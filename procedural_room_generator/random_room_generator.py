from collections import namedtuple

import bmesh
import bpy
import numpy as np
from mathutils import Vector

from . import camera_placer
from .config import RoomConfig
from .geometry_utils import random_sun_direction
from .object_placer import ObjectPlacer
from .material_generator import MaterialGenerator

Rect = namedtuple('Rect', ['x', 'y', 'w', 'd'])  # Center (x, y), width, depth


class RandomRoomGenerator:
    def __init__(self, seed=None, config=None):
        self.config = config or RoomConfig()

        # Initialize random seed
        if seed is not None:
            np.random.seed(seed)

        # Initialize material generator with same seed
        self.material_gen = MaterialGenerator(seed)

        # Store generated colors for consistency within a room
        self.room_colors = {
            'wall': self.material_gen.generate_wall_color(),
            'door': self.material_gen.generate_door_color(),
            'floor': self.material_gen.generate_floor_color()
        }
        self.room_colors['ceiling'] = self.material_gen.generate_ceiling_color(self.room_colors['wall'])
        self.room_colors['door_frame'] = self.material_gen.generate_frame_color(self.room_colors['door'])
        self.room_colors['window_frame'] = self.material_gen.generate_frame_color()

        # Clean scene
        self.clean_scene()

    def clean_scene(self):
        """Clear existing objects in the scene"""
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

        # Create new collections if they don't exist
        if "Walls" not in bpy.data.collections:
            walls_collection = bpy.data.collections.new("Walls")
            bpy.context.scene.collection.children.link(walls_collection)

        if "Doors" not in bpy.data.collections:
            doors_collection = bpy.data.collections.new("Doors")
            bpy.context.scene.collection.children.link(doors_collection)

        if "Windows" not in bpy.data.collections:
            windows_collection = bpy.data.collections.new("Windows")
            bpy.context.scene.collection.children.link(windows_collection)

        if "Furniture" not in bpy.data.collections:
            furniture_collection = bpy.data.collections.new("Furniture")
            bpy.context.scene.collection.children.link(furniture_collection)

    def generate_floor_plan(self, num_base_walls=4):
        """
        Generate a more realistic floor plan with varied shapes, recesses, and extensions.
        This creates open-plan living spaces with distinct areas and architectural interest.
        """
        # Start with a primary "core" area of the room (typically rectangular)
        width = np.random.uniform(self.config.ROOM_MIN_SIZE * 0.7, self.config.ROOM_MAX_SIZE * 0.7)
        depth = np.random.uniform(self.config.ROOM_MIN_SIZE * 0.7, self.config.ROOM_MAX_SIZE * 0.7)

        # Create the base room core
        core_vertices = [
            Vector((-width / 2, -depth / 2, 0)),
            Vector((width / 2, -depth / 2, 0)),
            Vector((width / 2, depth / 2, 0)),
            Vector((-width / 2, depth / 2, 0))
        ]

        # Determine architectural style complexity (affects how many features we add)
        complexity = np.random.uniform(0, 1.0)

        # Final vertices will be built from the core with modifications
        final_vertices = core_vertices.copy()

        # Add extensions/bays/alcoves if complexity warrants it
        num_features = int(np.random.triangular(1, 2, 5 if complexity > 0.6 else 3))

        # Track which walls we've modified to avoid overlapping features
        modified_wall_indices = set()

        for _ in range(num_features):
            # Select which architectural feature to add
            feature_type = np.random.choice(['extension', 'alcove', 'bay_window', 'angled_wall'])

            # Apply the selected architectural feature
            if feature_type == 'extension' and len(final_vertices) < 16:  # Limit total vertices
                final_vertices = self._add_extension(final_vertices, modified_wall_indices)
            elif feature_type == 'alcove':
                final_vertices = self._add_alcove(final_vertices, modified_wall_indices)
            elif feature_type == 'bay_window':
                final_vertices = self._add_bay_window(final_vertices, modified_wall_indices)
            elif feature_type == 'angled_wall':
                final_vertices = self._add_angled_wall(final_vertices, modified_wall_indices)

        # Sometimes add an open concept area with half walls/columns
        if complexity > 0.7 and np.random.random() > 0.5:
            # This will be handled separately in create_walls - just mark the area
            self.open_concept_area = {
                'position': self._find_open_concept_position(final_vertices),
                'size': Vector((np.random.uniform(2, 4), np.random.uniform(2, 4), 0))
            }
        else:
            self.open_concept_area = None

        return final_vertices

    def _add_extension(self, vertices, modified_wall_indices):
        """Add an extension to the room (creates additional living space in one direction)"""
        # Choose a wall that hasn't been modified yet
        available_walls = [i for i in range(len(vertices)) if i not in modified_wall_indices]
        if not available_walls:
            return vertices  # No unmodified walls left

        wall_idx = np.random.choice(available_walls)
        v1_idx = wall_idx
        v2_idx = (wall_idx + 1) % len(vertices)

        # Mark this wall as modified
        modified_wall_indices.add(wall_idx)

        # Get the vertices of the selected wall
        v1 = vertices[v1_idx]
        v2 = vertices[v2_idx]

        # Calculate wall direction and perpendicular direction
        wall_vec = v2 - v1
        wall_length = wall_vec.length
        wall_dir = wall_vec.normalized()
        perp_dir = Vector((-wall_dir.y, wall_dir.x, 0))  # 90-degree rotation

        # Extension parameters
        extension_depth = np.random.uniform(1.5, 4.0)  # How far it extends
        extension_width = np.random.uniform(wall_length * 0.3, wall_length * 0.8)  # How wide along the wall
        offset = np.random.uniform(0.1, wall_length - extension_width - 0.1)  # Position along wall

        # Calculate new vertices for the extension
        start_point = v1 + wall_dir * offset
        new_v1 = start_point
        new_v2 = start_point + perp_dir * extension_depth
        new_v3 = new_v2 + wall_dir * extension_width
        new_v4 = new_v1 + wall_dir * extension_width

        # Insert the new vertices into the list
        new_vertices = vertices.copy()
        new_vertices.insert(v1_idx + 1, new_v1)
        new_vertices.insert(v1_idx + 2, new_v2)
        new_vertices.insert(v1_idx + 3, new_v3)
        new_vertices.insert(v1_idx + 4, new_v4)

        # Remove the original v2 as it's replaced by the extension
        if v2_idx == 0:  # Wrap around case
            new_vertices.pop(4)  # After inserting 4 vertices, old v2 is now at index 4
        else:
            new_vertices.pop(v2_idx + 4)  # +4 because we added 4 vertices before it

        return new_vertices

    def _add_alcove(self, vertices, modified_wall_indices):
        """Add an alcove (recessed area) to the room"""
        # Choose a wall that hasn't been modified yet
        available_walls = [i for i in range(len(vertices)) if i not in modified_wall_indices]
        if not available_walls:
            return vertices  # No unmodified walls left

        wall_idx = np.random.choice(available_walls)
        v1_idx = wall_idx
        v2_idx = (wall_idx + 1) % len(vertices)

        # Mark this wall as modified
        modified_wall_indices.add(wall_idx)

        # Get the vertices of the selected wall
        v1 = vertices[v1_idx]
        v2 = vertices[v2_idx]

        # Calculate wall direction and perpendicular direction
        wall_vec = v2 - v1
        wall_length = wall_vec.length
        wall_dir = wall_vec.normalized()
        perp_dir = Vector((-wall_dir.y, wall_dir.x, 0))  # 90-degree rotation

        # Alcove parameters (negative for inward)
        alcove_depth = -np.random.uniform(0.8, 2.0)  # How deep it goes (negative for inward)
        alcove_width = np.random.uniform(wall_length * 0.3, wall_length * 0.6)
        offset = np.random.uniform(0.1, wall_length - alcove_width - 0.1)

        # Calculate new vertices for the alcove
        start_point = v1 + wall_dir * offset
        new_v1 = start_point
        new_v2 = start_point + perp_dir * alcove_depth
        new_v3 = new_v2 + wall_dir * alcove_width
        new_v4 = new_v1 + wall_dir * alcove_width

        # Insert the new vertices into the list
        new_vertices = vertices.copy()
        new_vertices.insert(v1_idx + 1, new_v1)
        new_vertices.insert(v1_idx + 2, new_v2)
        new_vertices.insert(v1_idx + 3, new_v3)
        new_vertices.insert(v1_idx + 4, new_v4)

        # Remove the original v2 as it's replaced
        if v2_idx == 0:  # Wrap around case
            new_vertices.pop(4)
        else:
            new_vertices.pop(v2_idx + 4)

        return new_vertices

    def _add_bay_window(self, vertices, modified_wall_indices):
        """Add a bay window (slightly angled outward extension, typically for windows)"""
        # Choose a wall that hasn't been modified yet
        available_walls = [i for i in range(len(vertices)) if i not in modified_wall_indices]
        if not available_walls:
            return vertices  # No unmodified walls left

        wall_idx = np.random.choice(available_walls)
        v1_idx = wall_idx
        v2_idx = (wall_idx + 1) % len(vertices)

        # Mark this wall as modified
        modified_wall_indices.add(wall_idx)

        # Get the vertices of the selected wall
        v1 = vertices[v1_idx]
        v2 = vertices[v2_idx]

        # Calculate wall direction and perpendicular direction
        wall_vec = v2 - v1
        wall_length = wall_vec.length
        wall_dir = wall_vec.normalized()
        perp_dir = Vector((-wall_dir.y, wall_dir.x, 0))  # 90-degree rotation

        # Bay window parameters
        bay_depth = np.random.uniform(0.6, 1.2)
        bay_width = np.random.uniform(wall_length * 0.2, wall_length * 0.5)
        offset = np.random.uniform(0.1, wall_length - bay_width - 0.1)

        # Bay windows typically have 3 segments
        start_point = v1 + wall_dir * offset
        new_v1 = start_point

        # First angled section
        angle1 = np.radians(np.random.uniform(20, 40))
        angled_dir1 = wall_dir.rotation_difference(
            wall_dir * np.cos(angle1) + perp_dir * np.sin(angle1)
        ).to_matrix() @ wall_dir
        new_v2 = new_v1 + angled_dir1 * (bay_width * 0.33)

        # Straight middle section
        new_v3 = new_v2 + perp_dir * bay_depth

        # Second angled section
        angle2 = np.radians(np.random.uniform(20, 40))
        angled_dir2 = wall_dir.rotation_difference(
            wall_dir * np.cos(-angle2) + perp_dir * np.sin(-angle2)
        ).to_matrix() @ wall_dir
        new_v4 = new_v3 + angled_dir2 * (bay_width * 0.33)

        # End point
        new_v5 = new_v1 + wall_dir * bay_width

        # Insert the new vertices into the list
        new_vertices = vertices.copy()
        new_vertices.insert(v1_idx + 1, new_v1)
        new_vertices.insert(v1_idx + 2, new_v2)
        new_vertices.insert(v1_idx + 3, new_v3)
        new_vertices.insert(v1_idx + 4, new_v4)
        new_vertices.insert(v1_idx + 5, new_v5)

        # Remove the original v2 as it's replaced
        if v2_idx == 0:  # Wrap around case
            new_vertices.pop(5)
        else:
            new_vertices.pop(v2_idx + 5)

        return new_vertices

    def _add_angled_wall(self, vertices, modified_wall_indices):
        """Replace a straight wall with an angled wall segment"""
        # Choose a wall that hasn't been modified yet
        available_walls = [i for i in range(len(vertices)) if i not in modified_wall_indices]
        if not available_walls:
            return vertices  # No unmodified walls left

        wall_idx = np.random.choice(available_walls)
        v1_idx = wall_idx

        # Mark this wall as modified
        modified_wall_indices.add(wall_idx)

        # Get the vertices of the selected wall
        v1 = vertices[v1_idx]
        v2 = vertices[(wall_idx + 1) % len(vertices)]

        # Calculate wall direction
        wall_vec = v2 - v1
        wall_length = wall_vec.length
        wall_dir = wall_vec.normalized()
        perp_dir = Vector((-wall_dir.y, wall_dir.x, 0))  # 90-degree rotation

        # Determine if the angle will create an inward or outward bend
        direction = 1 if np.random.random() > 0.5 else -1

        # Calculate the midpoint with some random offset
        offset_factor = np.random.uniform(0.3, 0.7)  # Position along the wall
        midpoint_on_wall = v1 + wall_dir * (wall_length * offset_factor)

        # Calculate how far to push the midpoint perpendicular to the wall
        push_distance = direction * np.random.uniform(0.5, 1.5)
        new_midpoint = midpoint_on_wall + perp_dir * push_distance

        # Insert the new vertex into the list
        new_vertices = vertices.copy()
        new_vertices.insert(v1_idx + 1, new_midpoint)

        return new_vertices

    def _find_open_concept_position(self, vertices):
        """Find a suitable position for an open concept area within the room"""
        # Compute the centroid of the room
        center = Vector((0, 0, 0))
        for v in vertices:
            center += v
        center /= len(vertices)

        # Find a random offset from the center, but ensure it stays inside the room
        max_offset = min(self.config.ROOM_MIN_SIZE * 0.2, 2.0)  # Limit how far we can go from center
        offset_x = np.random.uniform(-max_offset, max_offset)
        offset_y = np.random.uniform(-max_offset, max_offset)

        return center + Vector((offset_x, offset_y, 0))

    def create_floor(self, vertices):
        """Create floor from vertices"""
        # Create mesh and object
        mesh = bpy.data.meshes.new("Floor")
        obj = bpy.data.objects.new("Floor", mesh)

        # Link object to scene
        bpy.context.scene.collection.objects.link(obj)

        # Create mesh from vertices
        bm = bmesh.new()

        # Add vertices
        for v in vertices:
            bm.verts.new(v)

        # Make face
        bm.verts.ensure_lookup_table()
        if len(bm.verts) >= 3:  # Need at least 3 verts for a face
            bm.faces.new(bm.verts)

        # Update and free BMesh
        bm.to_mesh(mesh)
        bm.free()

        # Add material
        mat = self.create_material("Floor_Material", self.room_colors['floor'])
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)

        return obj

    def create_walls(self, vertices):
        """Create walls from floor vertices as a single continuous mesh"""
        walls_data = []

        # Get the collection for walls
        walls_collection = bpy.data.collections["Walls"]

        # Create a single mesh for all external walls
        wall_mesh = bpy.data.meshes.new("External_Walls")
        wall_obj = bpy.data.objects.new("External_Walls", wall_mesh)

        # Link object to walls collection
        walls_collection.objects.link(wall_obj)

        # Create mesh using bmesh
        bm = bmesh.new()

        # For each wall segment, we'll create vertices
        wall_verts_refs = []  # Store references to bmesh verts

        if not vertices or len(vertices) < 2:  # Not enough vertices to form walls
            bm.to_mesh(wall_mesh)  # Create empty mesh if no verts
            bm.free()
            return []  # No wall data

        # Create vertices for all walls
        for i in range(len(vertices)):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % len(vertices)]

            # Calculate wall direction and perpendicular
            direction = (v2 - v1).normalized()
            # Perpendicular pointing inward (assuming CCW vertices)
            perp = Vector((direction.y, -direction.x, 0))

            # Calculate the four corners of this wall segment at floor level
            # Outside corners
            out_v1_coord = v1 - perp * (self.config.WALL_THICKNESS / 2)
            out_v2_coord = v2 - perp * (self.config.WALL_THICKNESS / 2)
            # Inside corners
            in_v1_coord = v1 + perp * (self.config.WALL_THICKNESS / 2)
            in_v2_coord = v2 + perp * (self.config.WALL_THICKNESS / 2)

            # Create bmesh vertices at floor and ceiling level
            v_out1_floor = bm.verts.new((out_v1_coord.x, out_v1_coord.y, 0))
            v_out2_floor = bm.verts.new((out_v2_coord.x, out_v2_coord.y, 0))
            v_in1_floor = bm.verts.new((in_v1_coord.x, in_v1_coord.y, 0))
            v_in2_floor = bm.verts.new((in_v2_coord.x, in_v2_coord.y, 0))

            v_out1_ceil = bm.verts.new((out_v1_coord.x, out_v1_coord.y, self.config.WALL_HEIGHT))
            v_out2_ceil = bm.verts.new((out_v2_coord.x, out_v2_coord.y, self.config.WALL_HEIGHT))
            v_in1_ceil = bm.verts.new((in_v1_coord.x, in_v1_coord.y, self.config.WALL_HEIGHT))
            v_in2_ceil = bm.verts.new((in_v2_coord.x, in_v2_coord.y, self.config.WALL_HEIGHT))

            wall_segment_bverts = {
                'out1_floor': v_out1_floor, 'out2_floor': v_out2_floor,
                'in1_floor': v_in1_floor, 'in2_floor': v_in2_floor,
                'out1_ceil': v_out1_ceil, 'out2_ceil': v_out2_ceil,
                'in1_ceil': v_in1_ceil, 'in2_ceil': v_in2_ceil
            }
            wall_verts_refs.append(wall_segment_bverts)

            # Create faces for this wall segment
            bm.faces.new([v_out1_floor, v_out2_floor, v_out2_ceil, v_out1_ceil])  # Outside
            bm.faces.new([v_in2_floor, v_in1_floor, v_in1_ceil, v_in2_ceil])  # Inside (reversed for normal)
            bm.faces.new([v_out1_ceil, v_out2_ceil, v_in2_ceil, v_in1_ceil])  # Top

            wall_length = (v2 - v1).length
            angle = np.arctan2(direction.y, direction.x)
            walls_data.append((wall_obj, wall_length, v1, v2, direction, angle))

        # Connect adjacent wall segments at corners
        for i in range(len(wall_verts_refs)):
            curr_wall_bverts = wall_verts_refs[i]
            next_wall_bverts = wall_verts_refs[(i + 1) % len(wall_verts_refs)]

            # Outside corner face (end of current to start of next)
            bm.faces.new([
                curr_wall_bverts['out2_floor'], next_wall_bverts['out1_floor'],
                next_wall_bverts['out1_ceil'], curr_wall_bverts['out2_ceil']
            ])
            # Inside corner face
            bm.faces.new([
                next_wall_bverts['in1_floor'], curr_wall_bverts['in2_floor'],
                curr_wall_bverts['in2_ceil'], next_wall_bverts['in1_ceil']
            ])  # Reversed for normal

        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)  # Recalculate normals

        # Update the mesh
        bm.to_mesh(wall_mesh)
        bm.free()

        # Smart UV unwrap
        self.wall_uv_unwrap(wall_obj, margin=0.03)  # Slightly larger margin for bricks

        # Add material to the wall object
        mat = self.create_material("Wall_Material", self.room_colors['wall'])
        if wall_obj.data.materials:
            wall_obj.data.materials[0] = mat
        else:
            wall_obj.data.materials.append(mat)

        # Handle architectural features (pillars, half-walls) - these remain separate
        if hasattr(self, 'open_concept_area') and self.open_concept_area is not None:
            # Create pillars or half-walls to define the open concept area
            position = self.open_concept_area['position']
            size = self.open_concept_area['size']

            # Decide on the type of feature
            feature_style = np.random.choice(['pillars', 'half_wall', 'L_shaped_half_wall'])

            if feature_style == 'pillars':
                # Create 2-4 pillars
                pillar_count = np.random.randint(2, 5)
                pillar_radius = np.random.uniform(0.1, 0.2)
                pillar_height = self.config.WALL_HEIGHT

                for p in range(pillar_count):
                    # Calculate pillar position - arrange them in a line or L-shape
                    if pillar_count <= 2 or p < 2:
                        # Linear arrangement for 2 pillars or first 2 of more pillars
                        pillar_x = position.x + size.x * (p / (pillar_count - 1) if pillar_count > 1 else 0 - 0.5)
                        pillar_y = position.y
                    else:
                        # Make an L-shape for additional pillars
                        pillar_x = position.x + size.x * 0.5
                        pillar_y = position.y + size.y * ((p - 1) / (pillar_count - 2) if pillar_count > 2 else 0 - 0.5)

                    bpy.ops.mesh.primitive_cylinder_add(
                        radius=pillar_radius,
                        depth=pillar_height,
                        location=(pillar_x, pillar_y, pillar_height / 2)
                    )
                    pillar = bpy.context.active_object
                    pillar.name = f"Pillar_{p + 1}"

                    bpy.ops.collection.objects_remove_all()  # Remove from current scene collection
                    walls_collection.objects.link(pillar)  # Link to Walls collection

                    # For pillars and half-walls, use a slightly varied wall color:
                    pillar_color = list(self.room_colors['wall'])
                    # Make pillars slightly darker or lighter
                    variation = np.random.uniform(0.85, 1.15)
                    pillar_color[0] = min(1.0, pillar_color[0] * variation)
                    pillar_color[1] = min(1.0, pillar_color[1] * variation)
                    pillar_color[2] = min(1.0, pillar_color[2] * variation)

                    mat_pillar = self.create_material("Pillar_Material", tuple(pillar_color))

                    if pillar.data.materials:
                        pillar.data.materials[0] = mat_pillar
                    else:
                        pillar.data.materials.append(mat_pillar)


            elif feature_style == 'half_wall':
                half_wall_height = self.config.WALL_HEIGHT * np.random.uniform(0.4, 0.6)
                bpy.ops.mesh.primitive_cube_add(size=1)  # Creates at origin
                half_wall = bpy.context.active_object
                half_wall.name = "Half_Wall"

                half_wall.dimensions = (size.x, self.config.WALL_THICKNESS, half_wall_height)
                half_wall.location = (position.x, position.y, half_wall_height / 2)

                if np.random.random() > 0.5:
                    half_wall.rotation_euler.z = np.radians(np.random.uniform(-45, 45))

                bpy.ops.collection.objects_remove_all()
                walls_collection.objects.link(half_wall)

                # For pillars and half-walls, use a slightly varied wall color:
                pillar_color = list(self.room_colors['wall'])
                # Make pillars slightly darker or lighter
                variation = np.random.uniform(0.85, 1.15)
                pillar_color[0] = min(1.0, pillar_color[0] * variation)
                pillar_color[1] = min(1.0, pillar_color[1] * variation)
                pillar_color[2] = min(1.0, pillar_color[2] * variation)

                mat_half_wall = self.create_material("Half_Wall_Material", pillar_color)
                if half_wall.data.materials:
                    half_wall.data.materials[0] = mat_half_wall
                else:
                    half_wall.data.materials.append(mat_half_wall)


            elif feature_style == 'L_shaped_half_wall':
                half_wall_height = self.config.WALL_HEIGHT * np.random.uniform(0.4, 0.6)
                mat_l_half_wall = self.create_material("Half_Wall_Material", (1.0, 1.0, 1.0, 1.0))  # Shared material

                # First segment
                bpy.ops.mesh.primitive_cube_add(size=1)
                half_wall1 = bpy.context.active_object
                half_wall1.name = "Half_Wall_Segment1"
                half_wall1.dimensions = (size.x, self.config.WALL_THICKNESS, half_wall_height)
                half_wall1.location = (position.x, position.y, half_wall_height / 2)
                bpy.ops.collection.objects_remove_all()
                walls_collection.objects.link(half_wall1)
                if half_wall1.data.materials:
                    half_wall1.data.materials[0] = mat_l_half_wall
                else:
                    half_wall1.data.materials.append(mat_l_half_wall)

                # Second segment (perpendicular)
                bpy.ops.mesh.primitive_cube_add(size=1)
                half_wall2 = bpy.context.active_object
                half_wall2.name = "Half_Wall_Segment2"
                # Position relative to first segment end or center
                half_wall2.dimensions = (self.config.WALL_THICKNESS, size.y, half_wall_height)
                half_wall2.location = (position.x + size.x / 2 - self.config.WALL_THICKNESS / 2,
                                       position.y + size.y / 2 - self.config.WALL_THICKNESS / 2,
                                       # Crude L-shape connection
                                       half_wall_height / 2)
                bpy.ops.collection.objects_remove_all()
                walls_collection.objects.link(half_wall2)
                if half_wall2.data.materials:
                    half_wall2.data.materials[0] = mat_l_half_wall
                else:
                    half_wall2.data.materials.append(mat_l_half_wall)

        return walls_data

    def add_doors_and_windows(self, walls):
        """Add doors and windows to walls with no overlaps, using area-based and large-window logic."""

        for wall_data in walls:
            wall, length, v1, v2, direction, angle = wall_data

            min_window_gap = 0.2  # Minimum distance between doors/windows
            margin = 0.3  # Margin from wall ends

            reserved_spans = []  # List of (start, end) along wall for door/window reservations

            # First, door logic (probability per wall)
            add_door = np.random.random() < self.config.DOOR_PROBABILITY and length >= (
                        self.config.DOOR_WIDTH + 2 * margin)
            # door_span = None # Not used
            if add_door:
                max_door_start = length - self.config.DOOR_WIDTH - margin
                if max_door_start < margin:  # Not enough space even after check
                    add_door = False
                else:
                    door_start = np.random.uniform(margin, max_door_start)
                    door_end = door_start + self.config.DOOR_WIDTH
                    reserved_spans.append(
                        (door_start - min_window_gap, door_end + min_window_gap))
                    self.create_door(wall, door_start, angle, direction, v1)

            use_large_window = np.random.random() < 0.5

            if use_large_window:
                max_window_length = min(self.config.WINDOW_MAX_WIDTH * 1.7, length - 2 * margin)
                min_large_window_length = max(self.config.WINDOW_MAX_WIDTH, self.config.WINDOW_MIN_WIDTH * 1.3)
                if max_window_length > min_large_window_length:
                    window_width = np.random.uniform(min_large_window_length, max_window_length)
                    valid_start_pos = margin
                    valid_end_pos = length - window_width - margin

                    if valid_end_pos > valid_start_pos:  # Check if any valid range exists
                        valid_positions = []
                        test_starts = np.linspace(valid_start_pos, valid_end_pos, num=32)
                        for w_start in test_starts:
                            w_end = w_start + window_width
                            overlaps = any((w_end > rs and w_start < re) for (rs, re) in reserved_spans)
                            if not overlaps:
                                valid_positions.append(w_start)
                        if valid_positions:
                            window_start_pos = np.random.choice(valid_positions)
                            self.create_window(wall, window_start_pos, angle, direction, v1, window_width,
                                               np.random.uniform(self.config.WINDOW_MIN_HEIGHT,
                                                                 self.config.WINDOW_MAX_HEIGHT))
                            reserved_spans.append(
                                (window_start_pos - min_window_gap, window_start_pos + window_width + min_window_gap))
            else:
                wall_area = length * self.config.WALL_HEIGHT
                expected_window_area = wall_area * 0.19 * np.random.uniform(0.8, 1.15)
                max_num_windows = int(length // (self.config.WINDOW_MIN_WIDTH + margin))
                remaining_length_for_windows = length - 2 * margin
                current_total_window_area = 0
                # window_spans = [] # Not used
                attempts_count = 0
                while (current_total_window_area < expected_window_area and
                       remaining_length_for_windows > self.config.WINDOW_MIN_WIDTH and
                       attempts_count < max_num_windows):

                    max_w = min(self.config.WINDOW_MAX_WIDTH, remaining_length_for_windows)
                    min_w = min(self.config.WINDOW_MIN_WIDTH, max_w)
                    if min_w >= max_w: break  # No space for more windows

                    width = np.random.uniform(min_w, max_w)
                    height = np.random.uniform(self.config.WINDOW_MIN_HEIGHT, self.config.WINDOW_MAX_HEIGHT)

                    valid_start_range_begin = margin
                    valid_start_range_end = length - width - margin

                    placed_this_window = False
                    if valid_start_range_end > valid_start_range_begin:
                        for _ in range(5):  # Try to place 5 times
                            w_start = np.random.uniform(valid_start_range_begin, valid_start_range_end)
                            w_end = w_start + width
                            if all(w_end + min_window_gap <= rs or w_start - min_window_gap >= re for (rs, re) in
                                   reserved_spans):
                                self.create_window(wall, w_start, angle, direction, v1, width, height)
                                reserved_spans.append((w_start - min_window_gap, w_end + min_window_gap))
                                current_total_window_area += width * height
                                remaining_length_for_windows -= (width + min_window_gap)
                                attempts_count += 1
                                placed_this_window = True
                                break
                    if not placed_this_window:
                        break  # Could not place this window, stop trying for this wall

    def create_door(self, wall, position, angle, direction, start_point):
        """Create a door by cutting a hole in the wall"""
        doors_collection = bpy.data.collections["Doors"]
        frame_thickness = 0.06  # Door frame thickness
        frame_protrusion = 0.08  # How much frame extends from wall
        floor_cut_extension = 0.05 # 5cm below floor level

        door_center_offset = position + self.config.DOOR_WIDTH / 2
        door_center_world = start_point + direction * door_center_offset
        # door_center_world.z will define the center of the actual opening (passageway)
        door_center_world.z = (self.config.DOOR_HEIGHT - floor_cut_extension) / 2

        # Cutter for boolean (hole in the wall)
        # This cutter defines the raw opening from z=0 to z=self.config.DOOR_HEIGHT
        bpy.ops.mesh.primitive_cube_add(
            size=1, location=(door_center_world.x, door_center_world.y, door_center_world.z)
        )
        door_cutter = bpy.context.active_object
        door_cutter.dimensions = (self.config.DOOR_WIDTH,
                                  self.config.WALL_THICKNESS * 1.5,  # Ensure it cuts through
                                  self.config.DOOR_HEIGHT + floor_cut_extension) # Height of the raw opening
        door_cutter.rotation_euler.z = angle

        # Apply boolean modifier to wall
        bool_mod = wall.modifiers.new(name=f"Door_Cut_{len(wall.modifiers)}", type='BOOLEAN')
        bool_mod.operation = 'DIFFERENCE'
        bool_mod.object = door_cutter

        prev_active = bpy.context.view_layer.objects.active
        bpy.context.view_layer.objects.active = wall
        bpy.ops.object.modifier_apply(modifier=bool_mod.name)
        bpy.context.view_layer.objects.active = prev_active
        bpy.data.objects.remove(door_cutter)

        # --- Create the door frame ---
        door_center_world.z = self.config.DOOR_HEIGHT / 2  # Reset Z to center of the door opening
        frame_depth = self.config.WALL_THICKNESS + (2 * frame_protrusion)

        # Outer solid of the frame:
        # Its bottom should be at z=0. Its top at z = self.config.DOOR_HEIGHT + frame_thickness.
        # So, its height is self.config.DOOR_HEIGHT + frame_thickness.
        # Its Z center is (self.config.DOOR_HEIGHT + frame_thickness) / 2.
        outer_frame_actual_height = self.config.DOOR_HEIGHT + frame_thickness
        outer_frame_center_z = outer_frame_actual_height / 2

        bpy.ops.mesh.primitive_cube_add(
            size=1, location=(door_center_world.x, door_center_world.y, outer_frame_center_z) # Adjusted Z location
        )
        door_frame_outer = bpy.context.active_object
        door_frame_outer.name = "Door_Frame_Outer_Temp"
        door_frame_outer.dimensions = (self.config.DOOR_WIDTH + frame_thickness, # Overall width of frame assembly
                                       frame_depth,
                                       outer_frame_actual_height) # Overall height of frame assembly
        door_frame_outer.rotation_euler.z = angle

        # Inner cutter for the frame (defines the empty doorway space):
        # Its bottom should be at z=0. Its top at z = self.config.DOOR_HEIGHT.
        # So, its height is self.config.DOOR_HEIGHT.
        # Its Z center is self.config.DOOR_HEIGHT / 2 (which is door_center_world.z).
        inner_cutter_actual_height = self.config.DOOR_HEIGHT
        # inner_cutter_center_z is door_center_world.z

        bpy.ops.mesh.primitive_cube_add(
            size=1, location=(door_center_world.x, door_center_world.y, door_center_world.z) # Z loc is center of opening
        )
        door_frame_inner_cutter = bpy.context.active_object
        door_frame_inner_cutter.name = "Door_Frame_Inner_Cutter_Temp"
        # Width of the clear opening within the frame is self.config.DOOR_WIDTH - frame_thickness
        door_frame_inner_cutter.dimensions = (self.config.DOOR_WIDTH - frame_thickness,
                                              frame_depth * 1.2,  # Ensure complete cut through frame_depth
                                              inner_cutter_actual_height) # Adjusted height
        door_frame_inner_cutter.rotation_euler.z = angle

        # Apply boolean subtraction to create hollow frame (top and two sides)
        bool_mod_frame = door_frame_outer.modifiers.new(name="Door_Frame_Hole", type='BOOLEAN')
        bool_mod_frame.operation = 'DIFFERENCE'
        bool_mod_frame.object = door_frame_inner_cutter

        bpy.context.view_layer.objects.active = door_frame_outer
        bpy.ops.object.modifier_apply(modifier="Door_Frame_Hole")
        bpy.data.objects.remove(door_frame_inner_cutter) # Delete inner cutter

        door_frame_outer.name = "Door_Frame"
        bpy.ops.collection.objects_remove_all() # Remove from default scene collection before linking
        doors_collection.objects.link(door_frame_outer)

        mat_frame = self.create_material("Door_Frame_Material", self.room_colors['door_frame'])
        if door_frame_outer.data.materials:
            door_frame_outer.data.materials[0] = mat_frame
        else:
            door_frame_outer.data.materials.append(mat_frame)

        # Create the door panel (positioned slightly inside the frame and with clearance)
        # Its Z location (door_center_world.z) is self.config.DOOR_HEIGHT / 2.
        # Its height (self.config.DOOR_HEIGHT - frame_thickness - 0.02) ensures it sits
        # with a gap above z=0 and below the top of the frame opening.
        bpy.ops.mesh.primitive_cube_add(
            size=1, location=(door_center_world.x, door_center_world.y, door_center_world.z)
        )
        door_panel = bpy.context.active_object
        door_panel.name = "Door"
        door_panel.dimensions = (
            self.config.DOOR_WIDTH - frame_thickness, # Panel width for side clearance
            0.05, # Thickness of the door panel itself
            self.config.DOOR_HEIGHT # Panel height for top/bottom clearance
        )
        door_panel.rotation_euler.z = angle

        bpy.ops.collection.objects_remove_all()
        doors_collection.objects.link(door_panel)
        mat_door = self.create_material("Door_Material", self.room_colors['door'])
        if door_panel.data.materials:
            door_panel.data.materials[0] = mat_door
        else:
            door_panel.data.materials.append(mat_door)

    def create_window(self, wall, position, angle, direction, start_point, width, height):
        """Create a window by cutting a hole, adding frame and glass."""
        windows_collection = bpy.data.collections["Windows"]
        frame_thickness = 0.06
        frame_protrusion = 0.08
        frame_depth = self.config.WALL_THICKNESS + (2 * frame_protrusion)

        window_center_offset = position + width / 2
        window_center_world = start_point + direction * window_center_offset
        window_center_world.z = self.config.WINDOW_SILL_HEIGHT + height / 2

        # --- Cutter for the hole in the wall ---
        hole_width = width + frame_thickness
        hole_height = height + frame_thickness

        bpy.ops.mesh.primitive_cube_add(
            size=1, location=(window_center_world.x, window_center_world.y, window_center_world.z)
        )
        window_cutter = bpy.context.active_object
        window_cutter.name = "Window_Cutter_Temp"
        window_cutter.dimensions = (hole_width, self.config.WALL_THICKNESS * 1.5, hole_height)
        window_cutter.rotation_euler.z = angle

        bool_mod_wall = wall.modifiers.new(name=f"Window_Cut_{len(wall.modifiers)}", type='BOOLEAN')
        bool_mod_wall.operation = 'DIFFERENCE'
        bool_mod_wall.object = window_cutter

        prev_active = bpy.context.view_layer.objects.active
        bpy.context.view_layer.objects.active = wall
        bpy.ops.object.modifier_apply(modifier=bool_mod_wall.name)
        bpy.context.view_layer.objects.active = prev_active
        bpy.data.objects.remove(window_cutter)

        # --- Create the window frame ---
        bpy.ops.mesh.primitive_cube_add(
            size=1, location=(window_center_world.x, window_center_world.y, window_center_world.z)
        )
        frame_outer = bpy.context.active_object
        frame_outer.name = "Window_Frame_Outer_Temp"
        frame_outer.dimensions = (width + frame_thickness,
                                  frame_depth,
                                  height + frame_thickness)
        frame_outer.rotation_euler.z = angle

        # Inner box to subtract for the frame hole
        bpy.ops.mesh.primitive_cube_add(
            size=1, location=(window_center_world.x, window_center_world.y, window_center_world.z)
        )
        frame_inner_cutter = bpy.context.active_object
        frame_inner_cutter.name = "Window_Frame_Inner_Cutter_Temp"
        frame_inner_cutter.dimensions = (width - frame_thickness,
                                         frame_depth * 1.2,  # Ensure complete cut
                                         height - frame_thickness)
        frame_inner_cutter.rotation_euler.z = angle

        bool_mod_frame = frame_outer.modifiers.new(name="Frame_Hole", type='BOOLEAN')
        bool_mod_frame.operation = 'DIFFERENCE'
        bool_mod_frame.object = frame_inner_cutter

        bpy.context.view_layer.objects.active = frame_outer
        bpy.ops.object.modifier_apply(modifier="Frame_Hole")
        bpy.data.objects.remove(frame_inner_cutter)

        frame_outer.name = "Window_Frame"
        bpy.ops.collection.objects_remove_all()
        windows_collection.objects.link(frame_outer)

        mat_frame = self.create_material("Window_Frame_Material", self.room_colors['window_frame'])
        if frame_outer.data.materials:
            frame_outer.data.materials[0] = mat_frame
        else:
            frame_outer.data.materials.append(mat_frame)

        # --- Create window glass (positioned in center of frame) ---
        bpy.ops.mesh.primitive_cube_add(
            size=1, location=(window_center_world.x, window_center_world.y, window_center_world.z)
        )
        window_glass = bpy.context.active_object
        window_glass.name = "Window_Glass"
        window_glass.dimensions = (width - frame_thickness,
                                   0.02,  # Thin glass
                                   height - frame_thickness)
        window_glass.rotation_euler.z = angle

        # Hack: Turn off shadow rays for the glass.
        window_glass.visible_shadow = False

        bpy.ops.collection.objects_remove_all()
        windows_collection.objects.link(window_glass)

        # Glass material
        mat_glass_name = "Glass_Material"
        if mat_glass_name not in bpy.data.materials:
            mat_glass = bpy.data.materials.new(name=mat_glass_name)
            mat_glass.use_nodes = True
            bsdf = mat_glass.node_tree.nodes.get("Principled BSDF")
            if not bsdf:
                bsdf = mat_glass.node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')
                output = mat_glass.node_tree.nodes.get("Material Output")
                if not output: output = mat_glass.node_tree.nodes.new(type='ShaderNodeOutputMaterial')
                mat_glass.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

            bsdf.inputs['Base Color'].default_value = (0.8, 0.9, 1.0, 1.0)
            bsdf.inputs['Transmission Weight'].default_value = 1.0
            bsdf.inputs['Roughness'].default_value = 0.05
            bsdf.inputs['IOR'].default_value = 1.45
        else:
            mat_glass = bpy.data.materials[mat_glass_name]

        if window_glass.data.materials:
            window_glass.data.materials[0] = mat_glass
        else:
            window_glass.data.materials.append(mat_glass)

    def create_material(self, name, color, roughness=None, metallic=None):
        """Create a material with enhanced properties"""
        # Set default roughness values based on material type
        if roughness is None:
            if 'Wall' in name:
                roughness = np.random.uniform(0.4, 0.6)  # Walls have some texture
            elif 'Door' in name:
                roughness = np.random.uniform(0.2, 0.4)  # Doors are smoother
            elif 'Floor' in name:
                roughness = np.random.uniform(0.3, 0.5)  # Floors vary
            elif 'Frame' in name:
                roughness = np.random.uniform(0.2, 0.3)  # Frames are usually smooth
            else:
                roughness = 0.5

        if metallic is None:
            metallic = 0.0  # Most materials are non-metallic

        if name in bpy.data.materials:
            mat = bpy.data.materials[name]
            if mat.use_nodes and mat.node_tree:
                bsdf = mat.node_tree.nodes.get("Principled BSDF")
                if bsdf:
                    bsdf.inputs['Base Color'].default_value = color
                    bsdf.inputs['Roughness'].default_value = roughness
                    bsdf.inputs['Metallic'].default_value = metallic
            else:
                mat.diffuse_color = color
            return mat

        material = bpy.data.materials.new(name=name)
        material.use_nodes = True
        bsdf = material.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs['Base Color'].default_value = color
            bsdf.inputs['Roughness'].default_value = roughness
            bsdf.inputs['Metallic'].default_value = metallic
        else:
            material.diffuse_color = color

        # Experiment: Brick texture for walls
        # if 'Wall' in name:
        #     # Create texture coordinate node
        #     nodes = material.node_tree.nodes
        #     links = material.node_tree.links
        #     tex_coord = nodes.new(type='ShaderNodeTexCoord')
        #
        #     # Create mapping node for adjusting texture scale
        #     mapping = nodes.new(type='ShaderNodeMapping')
        #     mapping.vector_type = 'TEXTURE'
        #     # Scale the texture - adjust these values for brick size
        #     # mapping.inputs['Scale'].default_value[0] = np.random.uniform(3.0, 6.0)  # X scale
        #     # mapping.inputs['Scale'].default_value[1] = np.random.uniform(3.0, 6.0)  # Y scale
        #     # mapping.inputs['Scale'].default_value[2] = 1.0  # Z scale
        #
        #     # Create brick texture node
        #     brick_tex = nodes.new(type='ShaderNodeTexBrick')
        #     brick_tex.inputs['Color1'].default_value = color  # Main brick color
        #
        #     # Create slightly varied color for mortar
        #     mortar_color = list(color)
        #     mortar_factor = np.random.uniform(0.0, 0.9)  # Mortar color factor
        #     mortar_color[0] *= mortar_factor
        #     mortar_color[1] *= mortar_factor
        #     mortar_color[2] *= mortar_factor
        #     mortar_color[3] = 1.0  # Alpha
        #     brick_tex.inputs['Color2'].default_value = mortar_color  # Mortar color
        #
        #     # Adjust brick parameters
        #     brick_tex.inputs['Scale'].default_value = np.random.uniform(20.0, 55.0)  # Overall texture scale
        #     brick_tex.inputs['Mortar Size'].default_value = 0.02  # Size of gaps between bricks
        #     brick_tex.inputs['Bias'].default_value = 0.0  # Adjust for more or less mortar
        #     brick_tex.inputs['Brick Width'].default_value = 0.5  # Width of bricks
        #     brick_tex.inputs['Row Height'].default_value = 0.25  # Height of brick rows
        #
        #     # Create displacement node
        #     displacement = nodes.new(type='ShaderNodeDisplacement')
        #     displacement.inputs['Scale'].default_value = 0.02  # Displacement strength
        #     displacement.inputs['Midlevel'].default_value = 0.5  # Base level for displacement
        #
        #     # Create output node if it doesn't exist
        #     output = nodes.get("Material Output")
        #     if not output:
        #         output = nodes.new(type='ShaderNodeOutputMaterial')
        #
        #     # Connect nodes
        #     links.new(tex_coord.outputs['UV'], mapping.inputs['Vector'])
        #     links.new(mapping.outputs['Vector'], brick_tex.inputs['Vector'])
        #     links.new(brick_tex.outputs['Color'], bsdf.inputs['Base Color'])
        #     links.new(brick_tex.outputs['Fac'], displacement.inputs['Height'])
        #     links.new(displacement.outputs['Displacement'], output.inputs['Displacement'])
        #
        #     # Position nodes for better organization
        #     tex_coord.location = (-800, 0)
        #     mapping.location = (-600, 0)
        #     brick_tex.location = (-400, 0)
        #     bsdf.location = (-200, 0)
        #     displacement.location = (-200, -200)
        #     output.location = (0, 0)

        return material

    def wall_uv_unwrap(self, obj, margin=0.02):
        """Apply UV unwrapping to walls ensuring horizontal brick orientation"""
        # Store current active object and mode
        prev_active = bpy.context.view_layer.objects.active
        prev_mode = None
        if prev_active:
            prev_mode = prev_active.mode

        # Set the target object as active
        bpy.context.view_layer.objects.active = obj

        # Enter edit mode
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')

        # Use box projection from view for consistent orientation
        # This will ensure each face is projected from its normal direction
        bpy.ops.uv.cube_project(cube_size=1.0, scale_to_bounds=True)

        # Return to object mode
        bpy.ops.object.mode_set(mode='OBJECT')

        # Restore previous active object and mode
        bpy.context.view_layer.objects.active = prev_active
        if prev_active and prev_mode:
            bpy.ops.object.mode_set(mode=prev_mode)

    def add_ceiling(self, vertices):
        """Add ceiling to the room"""
        mesh = bpy.data.meshes.new("Ceiling")
        obj = bpy.data.objects.new("Ceiling", mesh)
        bpy.context.scene.collection.objects.link(obj)

        bm = bmesh.new()
        if vertices and len(vertices) >= 3:
            ceil_verts = [bm.verts.new(Vector((v.x, v.y, self.config.WALL_HEIGHT))) for v in vertices]
            bm.faces.new(ceil_verts)  # Assuming CCW, normals will point down, flip if needed
            # Flip normals if they point down (for ceiling, they should point down into room)
            # bmesh.ops.reverse_faces(bm, faces=bm.faces) # if you want them pointing up

        bm.to_mesh(mesh)
        bm.free()

        mat = self.create_material("Ceiling_Material", self.room_colors['ceiling'])
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)
        return obj

    def generate_room(self):
        """Main function to generate a random room"""
        vertices = self.generate_floor_plan()  # num_base_walls is default
        floor = self.create_floor(vertices)
        walls_data = self.create_walls(vertices)  # Renamed from 'walls' to 'walls_data'
        ceiling = self.add_ceiling(vertices)
        self.add_doors_and_windows(walls_data)
        self.setup_render()  # Scene-wide render settings

        return {
            "floor": floor,
            "walls_data": walls_data,  # Contains wall objects and geometric info
            "ceiling": ceiling,
            "vertices": vertices
        }

    def populate_with_objects(self, floor_vertices, floor_obj,
                              object_count=15, use_physics=True,
                              object_dataset_path=None):
        """Populate the room floor with random objects."""
        placer_seed = None
        if hasattr(self.config, 'seed') and self.config.seed is not None:
            placer_seed = self.config.seed  # Pass seed if available
        placer = ObjectPlacer(seed=placer_seed)

        object_paths = None
        if object_dataset_path:
            try:
                objects_data = placer.load_objects_from_dataset(
                    object_dataset_path,
                    category="",
                    max_objects=object_count
                )
                object_paths = [obj_data['path'] + '/meshes/model.obj'  # Adjust based on dataset
                                for obj_data in objects_data if 'path' in obj_data and 'name' in obj_data]
            except Exception as e:
                print(f"Could not load objects from dataset: {e}. Falling back.")

        placed_objects = placer.populate_floor(
            floor_vertices=floor_vertices,
            floor_obj=floor_obj,
            object_count=object_count,
            use_physics=use_physics,
            object_paths=object_paths,  # Can be None
            scale_range=(self.config.FURNITURE_MIN_SCALE, self.config.FURNITURE_MAX_SCALE) if hasattr(self.config,
                                                                                                      'FURNITURE_MIN_SCALE') else (
            0.5, 1.5)  # Example range
        )
        return placed_objects

    def generate_room_with_objects(self, populate_floor=True, object_count=20,
                                   use_physics=True, object_dataset_path=None):
        """Generate a complete room with objects and multiple camera views."""
        room_data = self.generate_room()
        placed_objects = []

        if populate_floor:
            placed_objects = self.populate_with_objects(
                floor_vertices=room_data['vertices'],
                floor_obj=room_data['floor'],
                object_count=object_count,
                use_physics=use_physics,
                object_dataset_path=object_dataset_path
            )
            room_data['placed_objects'] = placed_objects

        cameras, camera_styles = camera_placer.setup_multiple_cameras(
            vertices=room_data['vertices'],
            wall_thickness=self.config.WALL_THICKNESS,
            wall_height=self.config.WALL_HEIGHT,
            num_cameras=self.config.NUM_CAMERA_VIEWS if hasattr(self.config, 'NUM_CAMERA_VIEWS') else 5,
            target_objects=placed_objects  # Pass the list of bpy.types.Object
            # bpy context elements (scene, ops, data) will use defaults from camera_placer
        )
        room_data['cameras'] = cameras
        room_data['camera_styles'] = camera_styles

        return room_data

    def generate_room_with_furniture_and_objects(self,
                                                 num_furniture=5,
                                                 num_objects=20,
                                                 surface_placement_ratio=0.6,
                                                 object_dataset_path=None):
        """
        Generate a complete room with procedural furniture and intelligently placed objects.

        Args:
            num_furniture: Number of furniture pieces to generate
            num_objects: Total number of objects to place
            surface_placement_ratio: Ratio of objects to try placing on furniture surfaces
            object_dataset_path: Optional path to object dataset

        Returns:
            Dictionary with all room data including furniture and objects
        """
        # Generate base room
        room_data = self.generate_room()

        # Create furniture generator with same seed and materials
        from .furniture_generator import ProceduralFurnitureGenerator
        furniture_gen = ProceduralFurnitureGenerator(
            seed=self.config.seed if hasattr(self.config, 'seed') else None,
            material_generator=self.material_gen
        )

        # Place furniture in room
        furniture_objects = furniture_gen.place_furniture_in_room(
            room_data['vertices'],
            room_data['walls_data'],
            num_pieces=num_furniture
        )

        room_data['furniture'] = furniture_objects
        room_data['furniture_surfaces'] = furniture_gen.surfaces

        # Create object placer
        placer_seed = self.config.seed if hasattr(self.config, 'seed') else None
        placer = ObjectPlacer(seed=placer_seed)

        # Load or create objects
        if object_dataset_path:
            try:
                objects_data = placer.load_objects_from_dataset(
                    object_dataset_path,
                    category="",
                    max_objects=num_objects
                )
                object_paths = [obj_data['path'] + '/meshes/model.obj'
                                for obj_data in objects_data]
                objects = placer.load_objects_from_paths(
                    object_paths,
                    scale_range=(0.1, 0.5)  # Smaller scale for surface objects
                )
            except Exception as e:
                print(f"Could not load objects from dataset: {e}")
                objects = placer.create_simple_objects(num_objects)
        else:
            objects = placer.create_simple_objects(num_objects)

        # Place objects with furniture awareness
        placer.place_objects_with_furniture_awareness(
            objects=objects,
            floor_vertices=room_data['vertices'],
            floor_obj=room_data['floor'],
            furniture_surfaces=furniture_gen.surfaces,
            furniture_objects=furniture_objects,
            surface_placement_ratio=surface_placement_ratio
        )

        room_data['placed_objects'] = objects

        # Setup cameras with furniture as potential targets
        all_target_objects = furniture_objects + objects
        cameras, camera_styles = camera_placer.setup_multiple_cameras(
            vertices=room_data['vertices'],
            wall_thickness=self.config.WALL_THICKNESS,
            wall_height=self.config.WALL_HEIGHT,
            num_cameras=self.config.NUM_CAMERA_VIEWS if hasattr(self.config, 'NUM_CAMERA_VIEWS') else 5,
            target_objects=all_target_objects
        )

        room_data['cameras'] = cameras
        room_data['camera_styles'] = camera_styles

        return room_data

    def setup_render(self):
        """Setup render settings for a nice preview"""
        # bpy.context.scene.render.engine = 'CYCLES'
        # if bpy.context.preferences.addons['cycles'].preferences.has_active_device():
        #     bpy.context.scene.cycles.device = 'GPU'
        # bpy.context.scene.cycles.samples = 128

        world = bpy.context.scene.world
        if world:  # Ensure world exists
            world.use_nodes = True
            world_nodes = world.node_tree.nodes
            world_links = world.node_tree.links

            bg_node = world_nodes.get('Background')
            if not bg_node:  # If no background node, create one
                bg_node = world_nodes.new(type='ShaderNodeBackground')
                output_node = world_nodes.get('World Output')
                if not output_node: output_node = world_nodes.new(type='ShaderNodeOutputWorld')
                if output_node: world_links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])

            sky_node = world_nodes.get('Sky Texture')  # Check if sky texture already exists
            if not sky_node:
                sky_node = world_nodes.new(type='ShaderNodeTexSky')

            # Ensure link is correct
            existing_link_to_bg_color = None
            for link in world_links:
                if link.to_node == bg_node and link.to_socket == bg_node.inputs['Color']:
                    existing_link_to_bg_color = link
                    break

            if existing_link_to_bg_color and existing_link_to_bg_color.from_node != sky_node:
                world_links.remove(existing_link_to_bg_color)  # Remove incorrect link
                world_links.new(sky_node.outputs['Color'], bg_node.inputs['Color'])
            elif not existing_link_to_bg_color:
                world_links.new(sky_node.outputs['Color'], bg_node.inputs['Color'])

            sky_node.sun_direction = random_sun_direction()
            sky_node.turbidity = np.random.uniform(2.0, 5.0)  # Randomized turbidity
            sky_node.ground_albedo = np.random.uniform(0.1, 0.4)  # Randomized albedo
            # bg_node.inputs['Strength'].default_value = np.random.uniform(0.7, 1.3)  # Randomize strength slightly
