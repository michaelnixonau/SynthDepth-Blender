import bpy
import bmesh
import numpy as np
from mathutils import Vector, Matrix
from typing import List, Dict, Tuple, Optional
from collections import namedtuple

# Surface tracking structure
Surface = namedtuple('Surface', ['center', 'dimensions', 'normal', 'transform', 'furniture_obj', 'occupied_regions'])


class ProceduralFurnitureGenerator:
    """
    Generates procedural furniture with tracked surfaces for object placement.
    """

    def __init__(self, seed=None, material_generator=None):
        if seed is not None:
            np.random.seed(seed)

        self.material_generator = material_generator
        self.surfaces = []  # Track all placeable surfaces
        self.furniture_collection = self._get_or_create_collection("Furniture")

        # Default furniture colors
        self.default_colors = {
            'wood': (0.4, 0.25, 0.15, 1.0),
            'metal': (0.7, 0.7, 0.7, 1.0),
            'fabric': (0.3, 0.3, 0.4, 1.0),
            'marble': (0.9, 0.9, 0.9, 1.0),
            'glass': (0.8, 0.9, 1.0, 1.0)
        }

    def _get_or_create_collection(self, name):
        """Get or create a collection."""
        if name in bpy.data.collections:
            return bpy.data.collections[name]

        collection = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(collection)
        return collection

    def _create_material(self, name, color, roughness=None, metallic=None):
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
        return material

    def _add_surface(self, center, dimensions, normal, transform, furniture_obj):
        """Add a surface to the tracking list."""
        surface = Surface(
            center=center,
            dimensions=dimensions,  # (width, depth) in local space
            normal=normal,
            transform=transform,  # World transform matrix
            furniture_obj=furniture_obj,
            occupied_regions=[]  # List of (center, dimensions) tuples for occupied areas
        )
        self.surfaces.append(surface)
        return surface

    def create_table(self, location, size=None, style='modern'):
        """Create a procedural table with tracked top surface."""
        if size is None:
            # Random size within reasonable bounds
            width = np.random.uniform(0.8, 2.0)
            depth = np.random.uniform(0.6, 1.5)
            height = np.random.uniform(0.7, 0.8)
        else:
            width, depth, height = size

        # Create table top
        bpy.ops.mesh.primitive_cube_add(size=1, location=location)
        table_top = bpy.context.active_object
        table_top.name = "Table_Top"

        top_thickness = 0.05
        table_top.dimensions = (width, depth, top_thickness)
        table_top.location.z = location[2] + height - top_thickness / 2

        # Apply material
        wood_color = self._vary_color(self.default_colors['wood'])
        mat_top = self._create_material("Table_Wood", wood_color, roughness=0.3)
        table_top.data.materials.append(mat_top)

        # Track the top surface
        surface_center = Vector((location[0], location[1], location[2] + height))
        self._add_surface(
            center=surface_center,
            dimensions=(width * 0.9, depth * 0.9),  # Slightly smaller to avoid edges
            normal=Vector((0, 0, 1)),
            transform=table_top.matrix_world.copy(),
            furniture_obj=table_top
        )

        # Create legs based on style
        legs = []
        if style == 'modern':
            # 4 corner legs
            leg_positions = [
                (-width / 2 + 0.05, -depth / 2 + 0.05),
                (width / 2 - 0.05, -depth / 2 + 0.05),
                (-width / 2 + 0.05, depth / 2 - 0.05),
                (width / 2 - 0.05, depth / 2 - 0.05)
            ]

            for lx, ly in leg_positions:
                bpy.ops.mesh.primitive_cylinder_add(
                    radius=0.03,
                    depth=height - top_thickness,
                    location=(location[0] + lx, location[1] + ly, location[2] + (height - top_thickness) / 2)
                )
                leg = bpy.context.active_object
                leg.name = "Table_Leg"

                # Metal legs for modern style
                mat_leg = self._create_material("Table_Metal", self.default_colors['metal'], roughness=0.2,
                                                metallic=0.8)
                leg.data.materials.append(mat_leg)
                legs.append(leg)

        elif style == 'rustic':
            # Chunky wooden legs
            leg_width = 0.08
            leg_positions = [
                (-width / 2 + leg_width / 2, -depth / 2 + leg_width / 2),
                (width / 2 - leg_width / 2, -depth / 2 + leg_width / 2),
                (-width / 2 + leg_width / 2, depth / 2 - leg_width / 2),
                (width / 2 - leg_width / 2, depth / 2 - leg_width / 2)
            ]

            for lx, ly in leg_positions:
                bpy.ops.mesh.primitive_cube_add(
                    size=1,
                    location=(location[0] + lx, location[1] + ly, location[2] + (height - top_thickness) / 2)
                )
                leg = bpy.context.active_object
                leg.name = "Table_Leg"
                leg.dimensions = (leg_width, leg_width, height - top_thickness)
                leg.data.materials.append(mat_top)  # Same wood material
                legs.append(leg)

        # Join all parts
        bpy.ops.object.select_all(action='DESELECT')
        table_top.select_set(True)
        for leg in legs:
            leg.select_set(True)
        bpy.context.view_layer.objects.active = table_top
        bpy.ops.object.join()

        # Move to furniture collection
        for coll in table_top.users_collection:
            coll.objects.unlink(table_top)
        self.furniture_collection.objects.link(table_top)

        table_top.name = "Table"
        return table_top

    def create_cabinet(self, location, size=None, num_shelves=None):
        """Create a procedural cabinet with tracked shelf surfaces."""
        if size is None:
            width = np.random.uniform(0.8, 1.5)
            depth = np.random.uniform(0.4, 0.6)
            height = np.random.uniform(1.5, 2.2)
        else:
            width, depth, height = size

        if num_shelves is None:
            num_shelves = np.random.randint(2, 5)

        # Cabinet body (hollow box)
        thickness = 0.02

        # Create outer box
        bpy.ops.mesh.primitive_cube_add(size=1, location=location)
        cabinet_outer = bpy.context.active_object
        cabinet_outer.dimensions = (width, depth, height)
        cabinet_outer.location.z = location[2] + height / 2

        # Create inner box for hollowing
        bpy.ops.mesh.primitive_cube_add(size=1, location=location)
        cabinet_inner = bpy.context.active_object
        cabinet_inner.dimensions = (width - 2 * thickness, depth - 2 * thickness, height - 2 * thickness)
        cabinet_inner.location.z = location[2] + height / 2 + thickness

        # Boolean difference
        bool_mod = cabinet_outer.modifiers.new(name="Hollow", type='BOOLEAN')
        bool_mod.operation = 'DIFFERENCE'
        bool_mod.object = cabinet_inner

        bpy.context.view_layer.objects.active = cabinet_outer
        bpy.ops.object.modifier_apply(modifier=bool_mod.name)
        bpy.data.objects.remove(cabinet_inner)

        # Add back panel
        bpy.ops.mesh.primitive_cube_add(size=1)
        back_panel = bpy.context.active_object
        back_panel.dimensions = (width - 2 * thickness, thickness, height - 2 * thickness)
        back_panel.location = (location[0], location[1] - depth / 2 + thickness * 1.5, location[2] + height / 2)

        # Create shelves
        shelves = []
        shelf_spacing = (height - 2 * thickness) / (num_shelves + 1)

        for i in range(num_shelves):
            shelf_height = thickness + (i + 1) * shelf_spacing
            bpy.ops.mesh.primitive_cube_add(size=1)
            shelf = bpy.context.active_object
            shelf.name = f"Shelf_{i}"
            shelf.dimensions = (width - 2 * thickness, depth - 2 * thickness, thickness)
            shelf.location = (location[0], location[1], location[2] + shelf_height)
            shelves.append(shelf)

            # Track shelf surface
            surface_center = Vector((location[0], location[1], location[2] + shelf_height + thickness / 2))
            self._add_surface(
                center=surface_center,
                dimensions=(width - 2 * thickness - 0.1, depth - 2 * thickness - 0.1),
                normal=Vector((0, 0, 1)),
                transform=shelf.matrix_world.copy(),
                furniture_obj=shelf
            )

        # Apply materials
        wood_color = self._vary_color(self.default_colors['wood'])
        mat = self._create_material("Cabinet_Wood", wood_color, roughness=0.4)

        # Join all parts
        bpy.ops.object.select_all(action='DESELECT')
        cabinet_outer.select_set(True)
        back_panel.select_set(True)
        for shelf in shelves:
            shelf.select_set(True)
        bpy.context.view_layer.objects.active = cabinet_outer
        bpy.ops.object.join()

        cabinet_outer.data.materials.append(mat)

        # Move to furniture collection
        for coll in cabinet_outer.users_collection:
            coll.objects.unlink(cabinet_outer)
        self.furniture_collection.objects.link(cabinet_outer)

        cabinet_outer.name = "Cabinet"
        return cabinet_outer

    def create_sofa(self, location, size=None, style='modern'):
        """Create a procedural sofa."""
        if size is None:
            width = np.random.uniform(1.5, 2.5)
            depth = np.random.uniform(0.8, 1.0)
            height = np.random.uniform(0.7, 0.9)
        else:
            width, depth, height = size

        seat_height = height * 0.45

        # Create seat
        bpy.ops.mesh.primitive_cube_add(size=1, location=location)
        seat = bpy.context.active_object
        seat.name = "Sofa_Seat"
        seat.dimensions = (width, depth * 0.8, seat_height)
        seat.location.z = location[2] + seat_height / 2

        # Create backrest
        bpy.ops.mesh.primitive_cube_add(size=1)
        backrest = bpy.context.active_object
        backrest.name = "Sofa_Backrest"
        backrest.dimensions = (width, depth * 0.3, height - seat_height)
        backrest.location = (
        location[0], location[1] - depth * 0.35, location[2] + seat_height + (height - seat_height) / 2)

        # Create armrests
        armrests = []
        for side in [-1, 1]:
            bpy.ops.mesh.primitive_cube_add(size=1)
            armrest = bpy.context.active_object
            armrest.name = "Sofa_Armrest"
            armrest.dimensions = (0.15, depth * 0.8, height * 0.7)
            armrest.location = (location[0] + side * (width / 2 - 0.075), location[1], location[2] + height * 0.35)
            armrests.append(armrest)

        # Apply materials
        fabric_color = self._vary_color(self.default_colors['fabric'])
        mat = self._create_material("Sofa_Fabric", fabric_color, roughness=0.8)

        # Join all parts
        bpy.ops.object.select_all(action='DESELECT')
        seat.select_set(True)
        backrest.select_set(True)
        for armrest in armrests:
            armrest.select_set(True)
        bpy.context.view_layer.objects.active = seat
        bpy.ops.object.join()

        seat.data.materials.append(mat)

        # Add subdivision for smoother look
        subsurf = seat.modifiers.new(name="Subdivision", type='SUBSURF')
        subsurf.levels = 1
        subsurf.render_levels = 2

        # Move to furniture collection
        for coll in seat.users_collection:
            coll.objects.unlink(seat)
        self.furniture_collection.objects.link(seat)

        seat.name = "Sofa"
        return seat

    def create_shelf_unit(self, location, size=None, style='grid'):
        """Create a procedural shelf unit with multiple tracked surfaces."""
        if size is None:
            width = np.random.uniform(1.0, 2.0)
            depth = np.random.uniform(0.3, 0.5)
            height = np.random.uniform(1.5, 2.5)
        else:
            width, depth, height = size

        thickness = 0.025

        if style == 'grid':
            # Create a grid of compartments
            rows = np.random.randint(3, 6)
            cols = np.random.randint(2, 4)

            # Create frame
            parts = []

            # Vertical dividers
            for i in range(cols + 1):
                x_offset = -width / 2 + i * (width / cols)
                bpy.ops.mesh.primitive_cube_add(size=1)
                divider = bpy.context.active_object
                divider.dimensions = (thickness, depth, height)
                divider.location = (location[0] + x_offset, location[1], location[2] + height / 2)
                parts.append(divider)

            # Horizontal shelves
            for j in range(rows + 1):
                y_offset = j * (height / rows)
                bpy.ops.mesh.primitive_cube_add(size=1)
                shelf = bpy.context.active_object
                shelf.dimensions = (width, depth, thickness)
                shelf.location = (location[0], location[1], location[2] + y_offset)
                parts.append(shelf)

                # Track shelf surfaces (except bottom)
                if j > 0:
                    for i in range(cols):
                        compartment_center_x = location[0] - width / 2 + (i + 0.5) * (width / cols)
                        surface_center = Vector(
                            (compartment_center_x, location[1], location[2] + y_offset + thickness / 2))
                        compartment_width = (width / cols) - thickness - 0.05

                        self._add_surface(
                            center=surface_center,
                            dimensions=(compartment_width, depth - 0.05),
                            normal=Vector((0, 0, 1)),
                            transform=shelf.matrix_world.copy(),
                            furniture_obj=shelf
                        )

        else:  # 'ladder' style
            # Create asymmetric shelves
            num_shelves = np.random.randint(4, 7)
            parts = []

            # Side panels
            for side in [-1, 1]:
                bpy.ops.mesh.primitive_cube_add(size=1)
                panel = bpy.context.active_object
                panel.dimensions = (thickness, depth, height)
                panel.location = (location[0] + side * width / 2, location[1], location[2] + height / 2)
                parts.append(panel)

            # Random shelf arrangement
            for i in range(num_shelves):
                shelf_y = (i + 1) * height / (num_shelves + 1)
                shelf_width = np.random.uniform(width * 0.4, width * 0.9)
                shelf_offset = np.random.uniform(-width * 0.2, width * 0.2)

                bpy.ops.mesh.primitive_cube_add(size=1)
                shelf = bpy.context.active_object
                shelf.dimensions = (shelf_width, depth, thickness)
                shelf.location = (location[0] + shelf_offset, location[1], location[2] + shelf_y)
                parts.append(shelf)

                # Track surface
                surface_center = Vector(
                    (location[0] + shelf_offset, location[1], location[2] + shelf_y + thickness / 2))
                self._add_surface(
                    center=surface_center,
                    dimensions=(shelf_width - 0.1, depth - 0.05),
                    normal=Vector((0, 0, 1)),
                    transform=shelf.matrix_world.copy(),
                    furniture_obj=shelf
                )

        # Apply materials
        wood_color = self._vary_color(self.default_colors['wood'])
        mat = self._create_material("Shelf_Wood", wood_color, roughness=0.35)

        # Join all parts
        if parts:
            bpy.ops.object.select_all(action='DESELECT')
            for part in parts:
                part.select_set(True)
            bpy.context.view_layer.objects.active = parts[0]
            bpy.ops.object.join()

            shelf_unit = parts[0]
            shelf_unit.data.materials.append(mat)

            # Move to furniture collection
            for coll in shelf_unit.users_collection:
                coll.objects.unlink(shelf_unit)
            self.furniture_collection.objects.link(shelf_unit)

            shelf_unit.name = "Shelf_Unit"
            return shelf_unit

        return None

    def _vary_color(self, base_color):
        """Add slight variation to a color."""
        variation = 0.1
        return tuple(
            min(1.0, max(0.0, c + np.random.uniform(-variation, variation)))
            for c in base_color[:3]
        ) + (base_color[3],)

    def place_furniture_in_room(self, floor_vertices, wall_data, num_pieces=5):
        """Place furniture intelligently along walls and in room."""
        placed_furniture = []

        # Furniture types and their placement preferences
        furniture_types = [
            ('table', {'placement': 'center', 'min_clearance': 1.0}),
            ('cabinet', {'placement': 'wall', 'min_clearance': 0.3}),
            ('sofa', {'placement': 'wall', 'min_clearance': 0.5}),
            ('shelf_unit', {'placement': 'wall', 'min_clearance': 0.2}),
        ]

        # Get room bounds
        min_x = min(v.x for v in floor_vertices)
        max_x = max(v.x for v in floor_vertices)
        min_y = min(v.y for v in floor_vertices)
        max_y = max(v.y for v in floor_vertices)

        for i in range(num_pieces):
            furniture_type, preferences = furniture_types[i % len(furniture_types)]

            if preferences['placement'] == 'wall':
                # Place along a wall
                wall_idx = np.random.randint(0, len(wall_data))
                wall, length, v1, v2, direction, angle = wall_data[wall_idx]

                # Random position along wall
                position_along_wall = np.random.uniform(0.5, length - 0.5)
                wall_point = v1 + direction * position_along_wall

                # Offset from wall
                perp_dir = Vector((-direction.y, direction.x, 0))
                offset = preferences['min_clearance'] + 0.3
                location = (
                    wall_point.x + perp_dir.x * offset,
                    wall_point.y + perp_dir.y * offset,
                    0
                )

                # Create furniture facing away from wall
                if furniture_type == 'cabinet':
                    furniture = self.create_cabinet(location)
                elif furniture_type == 'sofa':
                    furniture = self.create_sofa(location)
                elif furniture_type == 'shelf_unit':
                    furniture = self.create_shelf_unit(location)
                else:
                    furniture = self.create_table(location)

                # Rotate to face away from wall
                furniture.rotation_euler.z = angle + np.pi / 2

            else:  # center placement
                # Place in open area
                location = (
                    np.random.uniform(min_x + 1, max_x - 1),
                    np.random.uniform(min_y + 1, max_y - 1),
                    0
                )

                if furniture_type == 'table':
                    furniture = self.create_table(location)
                else:
                    furniture = self.create_cabinet(location)

                # Random rotation
                furniture.rotation_euler.z = np.random.uniform(0, 2 * np.pi)

            placed_furniture.append(furniture)

        return placed_furniture

    def get_available_surfaces(self, min_area=0.01):
        """Get all available surfaces for object placement."""
        available = []
        for surface in self.surfaces:
            # Calculate available area
            total_area = surface.dimensions[0] * surface.dimensions[1]
            occupied_area = sum(
                region[1][0] * region[1][1]
                for region in surface.occupied_regions
            )

            if total_area - occupied_area >= min_area:
                available.append(surface)

        return available

    def place_object_on_surface(self, obj, surface, margin=0.05):
        """
        Place an object on a furniture surface.
        Returns True if successful, False if no space available.
        """
        # Get object bounding box
        obj_bbox = self._get_object_bounds(obj)
        obj_width = obj_bbox[1][0] - obj_bbox[0][0]
        obj_depth = obj_bbox[1][1] - obj_bbox[0][1]
        obj_height = obj_bbox[1][2] - obj_bbox[0][2]

        # Check if object fits on surface
        if (obj_width + 2 * margin > surface.dimensions[0] or
                obj_depth + 2 * margin > surface.dimensions[1]):
            return False

        # Find available position on surface
        position = self._find_available_position(
            surface, (obj_width, obj_depth), margin
        )

        if position is None:
            return False

        # Place object
        world_pos = surface.transform @ Vector((position[0], position[1], obj_height / 2))
        obj.location = world_pos

        # Random rotation around Z
        obj.rotation_euler.z = np.random.uniform(0, 2 * np.pi)

        # Mark area as occupied
        surface.occupied_regions.append((position, (obj_width, obj_depth)))

        return True

    def _get_object_bounds(self, obj):
        """Get object bounding box in local space."""
        mesh = obj.data
        verts = [obj.matrix_world @ v.co for v in mesh.vertices]

        if not verts:
            return (Vector((0, 0, 0)), Vector((0, 0, 0)))

        min_x = min(v.x for v in verts)
        max_x = max(v.x for v in verts)
        min_y = min(v.y for v in verts)
        max_y = max(v.y for v in verts)
        min_z = min(v.z for v in verts)
        max_z = max(v.z for v in verts)

        return (Vector((min_x, min_y, min_z)), Vector((max_x, max_y, max_z)))

    def _find_available_position(self, surface, obj_dimensions, margin):
        """Find an available position on a surface for an object."""
        obj_width, obj_depth = obj_dimensions

        # Simple grid-based search for available space
        grid_resolution = 0.1
        surface_width, surface_depth = surface.dimensions

        # Try random positions
        for _ in range(50):
            # Random position within surface bounds
            x = np.random.uniform(
                -surface_width / 2 + obj_width / 2 + margin,
                surface_width / 2 - obj_width / 2 - margin
            )
            y = np.random.uniform(
                -surface_depth / 2 + obj_depth / 2 + margin,
                surface_depth / 2 - obj_depth / 2 - margin
            )

            # Check if position overlaps with occupied regions
            overlaps = False
            for occupied_pos, occupied_dims in surface.occupied_regions:
                if (abs(x - occupied_pos[0]) < (obj_width + occupied_dims[0]) / 2 + margin and
                        abs(y - occupied_pos[1]) < (obj_depth + occupied_dims[1]) / 2 + margin):
                    overlaps = True
                    break

            if not overlaps:
                return (x, y)

        return None

    def clear_surfaces(self):
        """Clear all tracked surfaces."""
        self.surfaces = []


# Integration function for room generator
def add_furniture_to_room(room_generator, room_data, num_furniture=5, populate_surfaces=True):
    """
    Add procedural furniture to a generated room.

    Args:
        room_generator: The RandomRoomGenerator instance
        room_data: Dictionary containing room data (vertices, walls, etc.)
        num_furniture: Number of furniture pieces to add
        populate_surfaces: Whether to populate furniture surfaces with objects

    Returns:
        Updated room_data with furniture information
    """
    # Create furniture generator with same seed and material generator
    furniture_gen = ProceduralFurnitureGenerator(
        seed=room_generator.config.seed if hasattr(room_generator.config, 'seed') else None,
        material_generator=room_generator.material_gen
    )

    # Place furniture in room
    furniture_objects = furniture_gen.place_furniture_in_room(
        room_data['vertices'],
        room_data['walls_data'],
        num_pieces=num_furniture
    )

    # Store furniture data
    room_data['furniture'] = furniture_objects
    room_data['furniture_surfaces'] = furniture_gen.surfaces
    room_data['furniture_generator'] = furniture_gen

    return room_data