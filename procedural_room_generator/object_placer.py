import bpy
import bmesh
import numpy as np
import json
import random
import os
import blenderproc as bproc
from mathutils import Vector
from typing import List, Dict, Optional, Tuple


class ObjectPlacer:
    """
    Handles placement of random objects within room environments.
    Can work with or without physics simulation using BlenderProc's built-in methods.
    """

    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def load_objects_from_dataset(self, metadata_path: str, category: str = "",
                                  max_objects: int = 25) -> List[Dict]:
        """
        Load objects from a metadata JSON file.

        Args:
            metadata_path: Path to the metadata JSON file
            category: Category filter (empty string for all)
            max_objects: Maximum number of objects to load

        Returns:
            List of object data dictionaries
        """
        try:
            with open(metadata_path, 'r') as f:
                items = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Could not find metadata file at {metadata_path}")
            return []

        # Filter by category if specified
        if category:
            items = [item for item in items if item.get('category') == category]

        # Randomly select objects
        random.shuffle(items)
        return items[:max_objects]

    def load_objects_from_paths(self, object_paths: List[str],
                                scale_range: Tuple[float, float] = (1.0, 1.0)) -> List[bpy.types.Object]:
        """
        Load objects from file paths into Blender.

        Args:
            object_paths: List of file paths to 3D objects
            scale_range: Tuple of (min_scale, max_scale) for random scaling

        Returns:
            List of loaded Blender objects
        """
        objects = []

        for i, path in enumerate(object_paths):
            if not os.path.exists(path):
                print(f"Warning: Object file not found: {path}")
                continue

            try:
                # Load the object using BlenderProc when possible
                if path.endswith('.obj'):
                    # Use BlenderProc's loader for better integration
                    try:
                        loaded_objs = bproc.loader.load_obj(path)
                        if loaded_objs:
                            obj = loaded_objs[0]
                            obj.set_name(f"PlacedObject_{i}")

                            # Apply random scaling
                            # scale = 1
                            scale = np.random.uniform(scale_range[0], scale_range[1])
                            obj.set_scale([scale, scale, scale])

                            objects.append(obj.blender_obj)  # Get the underlying Blender object
                    except:
                        # Fallback to native Blender import
                        bpy.ops.wm.obj_import(filepath=path)
                        obj = bpy.context.active_object
                        if obj:
                            obj.name = f"PlacedObject_{i}"
                            scale = np.random.uniform(scale_range[0], scale_range[1])
                            obj.scale = (scale, scale, scale)
                            objects.append(obj)

                elif path.endswith('.blend'):
                    with bpy.data.libraries.load(path) as (data_from, data_to):
                        data_to.objects = data_from.objects
                    for obj in data_to.objects:
                        bpy.context.scene.collection.objects.link(obj)
                        objects.append(obj)
                else:
                    print(f"Warning: Unsupported file format: {path}")
                    continue

            except Exception as e:
                print(f"Error loading object from {path}: {e}")
                continue

        return objects

    def create_simple_objects(self, count: int = 10, object_types: List[str] = None) -> List[bpy.types.Object]:
        """
        Create simple geometric objects for testing.

        Args:
            count: Number of objects to create
            object_types: List of object types ('cube', 'sphere', 'cylinder', 'cone')

        Returns:
            List of created objects
        """
        if object_types is None:
            object_types = ['cube', 'sphere', 'cylinder', 'cone']

        objects = []

        for i in range(count):
            obj_type = random.choice(object_types)

            # Create object based on type
            if obj_type == 'cube':
                bpy.ops.mesh.primitive_cube_add()
            elif obj_type == 'sphere':
                bpy.ops.mesh.primitive_uv_sphere_add()
            elif obj_type == 'cylinder':
                bpy.ops.mesh.primitive_cylinder_add()
            elif obj_type == 'cone':
                bpy.ops.mesh.primitive_cone_add()

            obj = bpy.context.active_object
            obj.name = f"SimpleObject_{i}_{obj_type}"

            # Random scale
            scale = np.random.uniform(0.1, 0.5)
            obj.scale = (scale, scale, scale)

            # Random color material
            mat = bpy.data.materials.new(f"Material_{i}")
            mat.diffuse_color = (
                np.random.random(),
                np.random.random(),
                np.random.random(),
                1.0
            )
            obj.data.materials.append(mat)

            objects.append(obj)

        return objects

    def get_floor_bounds(self, floor_vertices: List[Vector], margin: float = 0.5) -> Tuple[Vector, Vector]:
        """
        Get the bounding box of the floor with optional margin.

        Args:
            floor_vertices: List of floor vertex positions
            margin: Margin to subtract from edges

        Returns:
            Tuple of (min_bounds, max_bounds) as Vector objects
        """
        if not floor_vertices:
            return Vector((0, 0, 0)), Vector((0, 0, 0))

        min_x = min(v.x for v in floor_vertices) + margin
        max_x = max(v.x for v in floor_vertices) - margin
        min_y = min(v.y for v in floor_vertices) + margin
        max_y = max(v.y for v in floor_vertices) - margin

        return Vector((min_x, min_y, 0)), Vector((max_x, max_y, 0))

    def point_in_polygon(self, point: Vector, vertices: List[Vector]) -> bool:
        """
        Check if a 2D point is inside a polygon using ray casting algorithm.

        Args:
            point: 2D point to test
            vertices: List of polygon vertices

        Returns:
            True if point is inside polygon
        """
        x, y = point.x, point.y
        n = len(vertices)
        inside = False

        p1x, p1y = vertices[0].x, vertices[0].y
        for i in range(1, n + 1):
            p2x, p2y = vertices[i % n].x, vertices[i % n].y
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def place_objects_randomly(self, objects: List[bpy.types.Object],
                               floor_vertices: List[Vector],
                               height_range: Tuple[float, float] = (2.0, 8.0),
                               margin: float = 0.5,
                               max_attempts: int = 100) -> None:
        """
        Place objects randomly within the floor bounds.

        Args:
            objects: List of objects to place
            floor_vertices: Vertices defining the floor boundary
            height_range: Range of heights to place objects at
            margin: Margin from walls
            max_attempts: Maximum placement attempts per object
        """
        min_bounds, max_bounds = self.get_floor_bounds(floor_vertices, margin)

        for obj in objects:
            placed = False
            attempts = 0

            while not placed and attempts < max_attempts:
                # Random position within bounds
                x = np.random.uniform(min_bounds.x, max_bounds.x)
                y = np.random.uniform(min_bounds.y, max_bounds.y)
                z = np.random.uniform(height_range[0], height_range[1])

                test_point = Vector((x, y, 0))

                # Check if point is inside the room polygon
                if self.point_in_polygon(test_point, floor_vertices):
                    obj.location = (x, y, z)

                    # Random rotation
                    obj.rotation_euler = (
                        np.random.uniform(0, 2 * np.pi),
                        np.random.uniform(0, 2 * np.pi),
                        np.random.uniform(0, 2 * np.pi)
                    )

                    placed = True

                attempts += 1

            if not placed:
                print(f"Warning: Could not place object {obj.name} after {max_attempts} attempts")

    def setup_physics_simulation(self, objects: List[bpy.types.Object],
                                 floor_obj: bpy.types.Object) -> None:
        """
        Setup physics simulation for objects and floor using BlenderProc's built-in methods.

        Args:
            objects: Objects that should fall and collide
            floor_obj: Floor object to act as collision surface
        """
        # Convert Blender objects to BlenderProc objects if needed and enable rigid body physics
        for obj in objects:
            try:
                # Try to use BlenderProc's method if the object supports it
                if hasattr(obj, 'enable_rigidbody'):
                    obj.enable_rigidbody(
                        active=True,
                        collision_shape='CONVEX_HULL',
                        mass=np.random.uniform(0.5, 2.0),
                        friction=0.5,
                    )
                else:
                    # Fallback: wrap in BlenderProc object
                    bproc_obj = bproc.types.MeshObject(obj)
                    bproc_obj.enable_rigidbody(
                        active=True,
                        collision_shape='CONVEX_HULL',
                        mass=np.random.uniform(0.5, 2.0),
                        friction=0.5,
                    )
            except Exception as e:
                print(f"Warning: Could not setup physics for object {obj.name}: {e}")
                # Fallback to original method if BlenderProc fails
                bpy.context.view_layer.objects.active = obj
                obj.select_set(True)
                bpy.ops.rigidbody.object_add(type='ACTIVE')
                obj.rigid_body.collision_shape = 'CONVEX_HULL'
                obj.rigid_body.mass = np.random.uniform(0.5, 2.0)
                obj.rigid_body.friction = 0.5
                obj.rigid_body.restitution = 0.1
                obj.select_set(False)

        # Setup floor as passive rigid body
        try:
            if hasattr(floor_obj, 'enable_rigidbody'):
                floor_obj.enable_rigidbody(active=False, collision_shape='MESH')
            else:
                bproc_floor = bproc.types.MeshObject(floor_obj)
                bproc_floor.enable_rigidbody(active=False, collision_shape='MESH')
        except Exception as e:
            print(f"Warning: Could not setup physics for floor: {e}")
            # Fallback to original method
            bpy.context.view_layer.objects.active = floor_obj
            floor_obj.select_set(True)
            bpy.ops.rigidbody.object_add(type='PASSIVE')
            floor_obj.rigid_body.collision_shape = 'MESH'
            floor_obj.select_set(False)

    def run_physics_simulation(self, min_simulation_time: float = 4.0,
                               max_simulation_time: float = 20.0,
                               check_object_interval: int = 1,
                               substeps_per_frame: int = 10,
                               solver_iters: int = 10) -> None:
        """
        Run physics simulation using BlenderProc's built-in method with improved object tracking.

        Args:
            min_simulation_time: Minimum simulation time in seconds
            max_simulation_time: Maximum simulation time in seconds
            check_object_interval: How often to check if objects have settled
            substeps_per_frame: Physics substeps per frame for accuracy
            solver_iters: Number of solver iterations per substep
        """
        # Store initial object count and names for tracking
        initial_object_names = []
        for obj in bpy.data.objects:
            if obj.rigid_body and obj.rigid_body.type == 'ACTIVE':
                initial_object_names.append(obj.name)

        initial_count = len(initial_object_names)
        print(f"Starting physics simulation with {initial_count} active objects")

        try:
            # Run BlenderProc simulation
            bproc.object.simulate_physics_and_fix_final_poses(
                min_simulation_time=min_simulation_time,
                max_simulation_time=max_simulation_time,
                check_object_interval=check_object_interval,
                substeps_per_frame=substeps_per_frame,
                solver_iters=solver_iters,
            )

            # Check how many objects remain
            print("Objects after simulation:")
            remaining_objects = []
            for name in initial_object_names:
                obj = bpy.data.objects.get(name)
                if obj.name and obj.rigid_body:
                    remaining_objects.append(obj)
                    print(f"- {name} at {obj.location}")
                    # If object fell too far, bring it back up
                    if obj.location.z < -5:
                        obj.location.z = 0.5
                        print(f"Rescued object {name} from falling too far")
                else:
                    print(f"Object {name} was removed during simulation")

            print(f"Physics simulation completed. {len(remaining_objects)}/{initial_count} objects remain.")

        except Exception as e:
            print(f"BlenderProc physics simulation failed: {e}")

    def place_objects_with_physics(self, objects: List[bpy.types.Object],
                                   floor_vertices: List[Vector],
                                   floor_obj: bpy.types.Object,
                                   initial_height_range: Tuple[float, float] = (3.0, 8.0),
                                   min_simulation_time: float = 4.0,
                                   max_simulation_time: float = 20.0) -> None:
        """
        Place objects using physics simulation for realistic placement with BlenderProc.

        Args:
            objects: Objects to place
            floor_vertices: Floor boundary vertices
            floor_obj: Floor object for collision
            initial_height_range: Initial height range for dropping objects
            min_simulation_time: Minimum simulation time in seconds
            max_simulation_time: Maximum simulation time in seconds
        """
        # First place objects randomly in the air
        self.place_objects_randomly(objects, floor_vertices, initial_height_range)

        # Setup physics using BlenderProc methods
        self.setup_physics_simulation(objects, floor_obj)

        # Run simulation using BlenderProc
        self.run_physics_simulation(
            min_simulation_time=min_simulation_time,
            max_simulation_time=max_simulation_time,
            check_object_interval=2,
            substeps_per_frame=10,
            solver_iters=10
        )

    def create_objects_collection(self, collection_name: str = "PlacedObjects") -> bpy.types.Collection:
        """
        Create or get a collection for placed objects.

        Args:
            collection_name: Name of the collection

        Returns:
            The collection object
        """
        if collection_name in bpy.data.collections:
            return bpy.data.collections[collection_name]

        collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(collection)
        return collection

    def populate_floor(self, floor_vertices: List[Vector],
                       floor_obj: bpy.types.Object,
                       object_count: int = 15,
                       use_physics: bool = True,
                       object_paths: Optional[List[str]] = None,
                       scale_range: Tuple[float, float] = (1.0, 1.0)) -> List[bpy.types.Object]:
        """
        Main method to populate floor with objects.

        Args:
            floor_vertices: Vertices defining the floor boundary
            floor_obj: Floor object
            object_count: Number of objects to place
            use_physics: Whether to use physics simulation
            object_paths: Optional list of object file paths to load
            scale_range: Scale range for objects

        Returns:
            List of placed objects
        """
        # Create collection for objects
        collection = self.create_objects_collection()

        # Load or create objects
        if object_paths:
            objects = self.load_objects_from_paths(object_paths, scale_range)
        else:
            objects = self.create_simple_objects(object_count)

        # Move objects to collection
        object_names = [obj.name for obj in objects] # Save names for later.
        for obj in objects:
            # Remove from default collection
            for coll in obj.users_collection:
                coll.objects.unlink(obj)
            # Add to our collection
            collection.objects.link(obj)

        # Place objects
        if use_physics and len(objects) > 0:
            self.place_objects_with_physics(objects, floor_vertices, floor_obj)
        else:
            # Place on floor level instead of in air
            height_range = (0.1, 0.5)  # Just above floor
            self.place_objects_randomly(objects, floor_vertices, height_range)

        print(f"Placed {len(objects)} objects on floor")
        return [bpy.data.objects.get(name) for name in object_names]


# Convenience function for integration with room generator
def populate_room_floor(floor_vertices: List[Vector],
                        floor_obj: bpy.types.Object,
                        object_count: int = 15,
                        use_physics: bool = True,
                        seed: Optional[int] = None) -> List[bpy.types.Object]:
    """
    Convenience function to populate a room floor with objects.

    Args:
        floor_vertices: Vertices defining the floor boundary
        floor_obj: Floor object
        object_count: Number of objects to place
        use_physics: Whether to use physics simulation
        seed: Random seed for reproducibility

    Returns:
        List of placed objects
    """
    placer = ObjectPlacer(seed=seed)
    return placer.populate_floor(
        floor_vertices=floor_vertices,
        floor_obj=floor_obj,
        object_count=object_count,
        use_physics=use_physics
    )