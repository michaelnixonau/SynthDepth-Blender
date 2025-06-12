import bpy
import numpy as np
from mathutils import Vector, Quaternion, Euler  # Added Quaternion, Euler
import random  # Moved to top

# Assuming geometry_utils.py is in the same directory or accessible via this relative path
# If random_room_generator.py uses `from .geometry_utils import ...`, this should also work.
from .geometry_utils import sample_farthest_camera_position


# Helper function (formerly _ensure_inside_room from RandomRoomGenerator)
def _ensure_inside_room(point, vertices, margin):
    """Ensure a point is inside the room polygon with margin from walls."""
    if not vertices:  # Guard against empty vertices
        return point

    center = Vector((0, 0, 0))
    for v in vertices:
        center += v
    center /= len(vertices)

    max_dist_from_center = 0
    if vertices:  # Ensure vertices is not an empty list before trying to get max
        max_dist_from_center = max((v - center).length for v in vertices) if len(vertices) > 0 else 0

    # Check if point might be outside (rough check)
    if (point - center).length > max_dist_from_center * 0.9:  # Heuristic
        direction_to_center = (center - point)
        if direction_to_center.length > 1e-6:  # Avoid normalizing zero vector
            direction_to_center.normalize()
            # Pull point back toward center by a heuristic amount
            point = point + direction_to_center * (margin + 0.1)
            # If point is at center and max_dist_from_center is 0, this logic is fine.
    return point


# Global list of camera styles
ALL_CAMERA_STYLES = [
    'eye_level_corner', 'low_angle', 'high_overview', 'wall_mounted',
    'object_focus', 'architectural', 'wide_establishing',
    'intimate_detail', 'dutch_angle', 'through_doorway'
]


def _calculate_camera_parameters_for_style(vertices, camera_style_req, target_objects, wall_thickness, wall_height):
    """
    Calculates camera location, rotation, focal length, and the effective style used.
    Does not create any Blender objects.
    Handles style selection if camera_style_req is None, and fallbacks for impossible styles.
    """
    current_styles_available = ALL_CAMERA_STYLES[:]
    if not target_objects or len(target_objects) == 0:
        if 'object_focus' in current_styles_available:
            current_styles_available.remove('object_focus')

    style_to_process = camera_style_req
    if style_to_process is None:
        if not current_styles_available:
            style_to_process = 'eye_level_corner'  # Ultimate fallback
        else:
            style_to_process = np.random.choice(current_styles_available)

    effective_style = style_to_process

    center = Vector((0, 0, 0))
    if vertices:
        for v in vertices: center += v
        center /= len(vertices)

    cam_loc = Vector((center.x, center.y - 3, 1.7))  # Default sensible location if other logic fails
    look_at = center.copy()
    focal_length = 35
    wall_margin = wall_thickness + 0.3

    # Style-specific logic
    if effective_style == 'eye_level_corner':
        if not vertices or len(vertices) < 3:  # Need at least 3 vertices for distinct corners
            print(f"Eye_level_corner: Not enough vertices. Falling back to 'high_overview'.")
            return _calculate_camera_parameters_for_style(vertices, 'high_overview', target_objects, wall_thickness,
                                                          wall_height)

        corner_idx = np.random.randint(len(vertices))
        corner = vertices[corner_idx].copy()

        direction_to_center = (center - corner)
        if direction_to_center.length < 1e-6:  # corner is effectively at center
            # Choose an arbitrary inward direction if room is tiny / just a point
            inward1 = Vector(vertices[(corner_idx + 1) % len(vertices)] - corner).normalized() if (vertices[(
                                                                                                                        corner_idx + 1) % len(
                vertices)] - corner).length > 1e-6 else Vector((0, 1, 0))
        else:
            inward1 = direction_to_center.normalized()

        cam_loc = corner + inward1 * wall_margin * 1.5  # Adjusted multiplier
        cam_loc.z = np.random.uniform(1.5, 1.8)

        opposite_corner_idx = (corner_idx + len(vertices) // 2) % len(vertices)
        opposite_corner = vertices[opposite_corner_idx].copy()
        look_at = opposite_corner + (center - opposite_corner) * 0.3
        look_at.z = np.random.uniform(1.0, 2.0)
        focal_length = np.random.uniform(24, 35)

    elif effective_style == 'low_angle':
        if vertices:
            cam_loc = sample_farthest_camera_position(vertices, wall_margin, n_samples=50)
        else:  # Fallback if no vertices
            cam_loc = Vector((center.x, center.y, 0.5))
        cam_loc.z = np.random.uniform(0.3, 0.8)
        look_at = center.copy()
        # Ensure look_at.z is above cam_loc.z and respects wall_height
        min_look_at_z = cam_loc.z + 0.5  # Look at least 0.5m above camera
        max_look_at_z = max(wall_height, min_look_at_z + 0.5) if wall_height > 0 else min_look_at_z + 1.0
        look_at_z_target_range_min = max(min_look_at_z, wall_height * 0.7 if wall_height > 0 else min_look_at_z)
        look_at.z = np.random.uniform(look_at_z_target_range_min, max_look_at_z)
        focal_length = np.random.uniform(20, 28)

    elif effective_style == 'high_overview':
        cam_loc = center + Vector(
            (np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), 0))  # Closer to center horizontal
        min_cam_z = 2.5  # Minimum height for overview
        cam_loc.z = max(min_cam_z, wall_height * np.random.uniform(0.85, 0.95) if wall_height > 0 else min_cam_z)
        look_at = center + Vector(
            (np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), 0))  # Slight offset target
        look_at.z = np.random.uniform(0, 0.5)  # Look towards floor
        focal_length = np.random.uniform(20, 35)

    elif effective_style == 'wall_mounted':
        if not vertices or len(vertices) < 2:
            print(f"Wall_mounted: Not enough vertices. Falling back to 'high_overview'.")
            return _calculate_camera_parameters_for_style(vertices, 'high_overview', target_objects, wall_thickness,
                                                          wall_height)

        wall_idx = np.random.randint(len(vertices))
        v1 = vertices[wall_idx].copy()
        v2 = vertices[(wall_idx + 1) % len(vertices)].copy()
        t = np.random.uniform(0.2, 0.8)
        wall_pos = v1 + (v2 - v1) * t
        wall_dir = (v2 - v1)
        if wall_dir.length < 1e-6:  # Vertices are coincident
            print(f"Wall_mounted: Wall segment too short. Falling back to 'high_overview'.")
            return _calculate_camera_parameters_for_style(vertices, 'high_overview', target_objects, wall_thickness,
                                                          wall_height)
        wall_dir.normalize()

        inward = Vector((-wall_dir.y, wall_dir.x, 0))
        if inward.dot(center - wall_pos) < 0: inward *= -1.0  # Ensure it points inward

        cam_loc = wall_pos + inward * wall_margin * 1.2  # Slightly closer to wall
        cam_loc.z = np.random.uniform(1.2, 2.2)
        look_at = wall_pos - inward * 5  # Look across room
        look_at.z = np.random.uniform(0.8, 1.8)
        focal_length = np.random.uniform(28, 50)

    elif effective_style == 'object_focus':
        # This style is pre-filtered if target_objects is empty when style_req=None.
        # This handles explicit request or if it was somehow chosen.
        if not target_objects or len(target_objects) == 0:
            print(f"Object_focus: No target objects. Falling back to 'eye_level_corner'.")
            return _calculate_camera_parameters_for_style(vertices, 'eye_level_corner', target_objects, wall_thickness,
                                                          wall_height)

        target_obj = random.choice(target_objects)
        obj_loc = target_obj.location.copy()
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(1.5, 3.0)
        cam_loc = obj_loc + Vector((np.cos(angle) * distance, np.sin(angle) * distance, np.random.uniform(0.5, 1.5)))
        cam_loc = _ensure_inside_room(cam_loc, vertices, wall_margin)
        look_at = obj_loc + Vector((0, 0, np.random.uniform(0.1, 0.3)))  # Look slightly above object base
        focal_length = np.random.uniform(50, 85)

    elif effective_style == 'architectural':
        features = []
        if "Doors" in bpy.data.collections:
            for obj in bpy.data.collections["Doors"].objects:
                if "Door_Frame" in obj.name: features.append(obj.location.copy())
        if "Windows" in bpy.data.collections:
            for obj in bpy.data.collections["Windows"].objects:
                if "Window_Frame" in obj.name: features.append(obj.location.copy())

        if features:
            feature_loc = random.choice(features)
            to_center = (center - feature_loc)
            if to_center.length < 1e-6:
                to_center = Vector((0, 1, 0))  # Default if feature is at center
            else:
                to_center.normalize()

            cam_loc = feature_loc + to_center * np.random.uniform(2, 4)
            cam_loc.z = np.random.uniform(1.3, 2.0)
            cam_loc = _ensure_inside_room(cam_loc, vertices, wall_margin)
            look_at = feature_loc
            focal_length = np.random.uniform(35, 50)
        else:
            print(f"Architectural style: No features found. Falling back to 'eye_level_corner'.")
            return _calculate_camera_parameters_for_style(vertices, 'eye_level_corner', target_objects, wall_thickness,
                                                          wall_height)

    elif effective_style == 'wide_establishing':
        if vertices:
            cam_loc = sample_farthest_camera_position(vertices, wall_margin * 1.5, n_samples=100)
        else:  # Fallback if no vertices
            cam_loc = Vector((center.x + 4, center.y + 4, 1.6))  # Arbitrary far-ish position
        cam_loc.z = np.random.uniform(1.4, 1.8)
        look_at = center.copy()
        look_at.z = np.random.uniform(1.0, 1.5)
        focal_length = np.random.uniform(16, 24)

    elif effective_style == 'intimate_detail':
        target_point = center.copy()  # Default if no other target found
        target_point.z = np.random.uniform(0.5, 1.0)

        if target_objects and len(target_objects) > 0 and np.random.random() > 0.5:
            target_point = random.choice(target_objects).location.copy()
        elif vertices:
            wall_point_idx = np.random.randint(len(vertices))
            wall_point = vertices[wall_point_idx].copy()
            target_point = wall_point + (center - wall_point) * np.random.uniform(0.1, 0.4)  # Point on wall
            target_point.z = np.random.uniform(0.5, 2.0)

        direction_from_target = (center - target_point)
        if direction_from_target.length < 1e-6:
            direction_from_target = Vector((1, 0, 0))  # Arbitrary if target is center
        else:
            direction_from_target.normalize()

        cam_loc = target_point + direction_from_target * np.random.uniform(0.8, 1.5)
        cam_loc.z = target_point.z + np.random.uniform(-0.2, 0.2)  # Similar height to target
        cam_loc = _ensure_inside_room(cam_loc, vertices, wall_margin)
        look_at = target_point
        focal_length = np.random.uniform(85, 135)

    elif effective_style == 'dutch_angle':
        if vertices:
            cam_loc = sample_farthest_camera_position(vertices, wall_margin, n_samples=50)
        else:  # Fallback if no vertices
            cam_loc = Vector((center.x - 2, center.y - 2, 1.5))
        cam_loc.z = np.random.uniform(1.0, 2.0)
        look_at = center + Vector((np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-0.5, 0.5)))
        focal_length = np.random.uniform(28, 40)

    elif effective_style == 'through_doorway':
        door_locs = []
        if "Doors" in bpy.data.collections:
            for obj in bpy.data.collections["Doors"].objects:
                if "Door_Frame" in obj.name: door_locs.append(obj.location.copy())

        if door_locs:
            door_loc = random.choice(door_locs)
            to_center_dir = (center - door_loc)
            if to_center_dir.length < 1e-6:
                to_center_dir = Vector((0, 1, 0))  # Default if door is at center
            else:
                to_center_dir.normalize()

            cam_loc = door_loc - to_center_dir * np.random.uniform(0.3, 0.7)  # Just outside/in doorway
            cam_loc.z = np.random.uniform(1.4, 1.7)
            look_at = center.copy() + to_center_dir * np.random.uniform(1, 3)  # Look into room towards center area
            look_at.z = np.random.uniform(0.8, 1.5)
            focal_length = np.random.uniform(24, 35)
        else:
            print(f"Through_doorway style: No doors found. Falling back to 'eye_level_corner'.")
            return _calculate_camera_parameters_for_style(vertices, 'eye_level_corner', target_objects, wall_thickness,
                                                          wall_height)

    else:  # Should not be reached if effective_style is one of ALL_CAMERA_STYLES
        print(f"Warning: Unknown camera style '{effective_style}'. Using defaults.")
        # Defaults (cam_loc, look_at, focal_length) are already set

    # Calculate final rotation (aim + roll)
    direction = look_at - cam_loc
    if direction.length < 1e-6: direction = Vector((0, 0, -1))  # Avoid issues if cam_loc is too close to look_at

    rot_quat_aim = direction.to_track_quat('-Z', 'Y')
    roll_to_apply = 0.0
    if effective_style == 'dutch_angle':
        roll_to_apply = np.random.uniform(-0.3, 0.3)  # approx +/- 17 degrees
    else:
        roll_to_apply = np.random.uniform(-0.05, 0.05)  # approx +/- 3 degrees

    if abs(roll_to_apply) > 1e-9:
        roll_quat = Quaternion((0, 0, 1), roll_to_apply)
        final_rot_quat = rot_quat_aim @ roll_quat
    else:
        final_rot_quat = rot_quat_aim

    final_rotation_euler = final_rot_quat.to_euler('XYZ')  # Default Euler order for cameras

    return cam_loc, final_rotation_euler, focal_length, effective_style


def setup_camera(vertices, camera_style=None, target_objects=None, wall_thickness=0, wall_height=0):
    """
    Improved camera placement. Creates a new camera object.
    Interface and behavior for single camera creation remain the same.
    """
    cam_loc, rotation_euler, focal_length, final_style = \
        _calculate_camera_parameters_for_style(
            vertices, camera_style, target_objects, wall_thickness, wall_height
        )

    bpy.ops.object.camera_add(location=cam_loc, rotation=rotation_euler)
    camera = bpy.context.active_object
    camera.name = f"Camera_{final_style}"
    camera.data.lens = focal_length

    if bpy.context.scene.camera is None:
        bpy.context.scene.camera = camera

    if "RoomLight" not in bpy.data.objects:
        center = Vector((0, 0, 0))
        if vertices:
            for v in vertices: center += v
            center /= len(vertices)

        light_height = wall_height * 0.95 if wall_height > 0 else 3.0
        bpy.ops.object.light_add(type='AREA', location=(center.x, center.y, light_height))
        light = bpy.context.active_object
        light.name = "RoomLight"

        max_dist_from_center = 0
        if vertices:
            max_dist_from_center = max((v - center).length for v in vertices) if len(vertices) > 0 else 10.0
            light.data.size = max_dist_from_center / 4 if max_dist_from_center > 0 else 2.5
        else:
            light.data.size = 2.5
        light.data.energy = 500  # Using a higher default energy, adjust as needed

    return camera, final_style


def setup_multiple_cameras(vertices, num_cameras=5, target_objects=None, wall_thickness=0, wall_height=0):
    """
    Sets up a SINGLE camera object with multiple keyframed poses.
    The interface (arguments and return values) remains the same.
    Returns a list containing 'num_cameras' references to the single camera object,
    and a list of styles used for each keyframe.
    """
    bpy.context.scene.frame_end = max(bpy.context.scene.frame_end, num_cameras)

    cam_name = "KeyframedMainCamera"
    if cam_name in bpy.data.objects and bpy.data.objects[cam_name].type == 'CAMERA':
        main_camera = bpy.data.objects[cam_name]
    else:
        if cam_name in bpy.data.objects:  # Exists but not a camera
            bpy.data.objects.remove(bpy.data.objects[cam_name], do_unlink=True)
        bpy.ops.object.camera_add(location=(0, 0, 0))
        main_camera = bpy.context.active_object
        main_camera.name = cam_name

    main_camera.rotation_mode = 'XYZ'  # Ensure consistent rotation mode

    if main_camera.animation_data:
        main_camera.animation_data_clear()
    if main_camera.data.animation_data:
        main_camera.data.animation_data_clear()

    bpy.context.scene.camera = main_camera

    if "RoomLight" not in bpy.data.objects:  # Copied light setup logic
        center = Vector((0, 0, 0))
        if vertices:
            for v in vertices: center += v
            center /= len(vertices)

        light_height = wall_height * 0.95 if wall_height > 0 else 3.0
        bpy.ops.object.light_add(type='AREA', location=(center.x, center.y, light_height))
        light = bpy.context.active_object
        light.name = "RoomLight"

        max_dist_from_center = 0
        if vertices:
            max_dist_from_center = max((v - center).length for v in vertices) if len(vertices) > 0 else 10.0
            light.data.size = max_dist_from_center / 4 if max_dist_from_center > 0 else 2.5
        else:
            light.data.size = 2.5
        light.data.energy = 250

    returned_cameras_list = []
    styles_used_for_keyframes = []

    selectable_styles = ALL_CAMERA_STYLES[:]
    if not target_objects or len(target_objects) == 0:
        if 'object_focus' in selectable_styles:
            selectable_styles.remove('object_focus')
    if not selectable_styles:
        selectable_styles.append('eye_level_corner')

    np.random.shuffle(selectable_styles)
    bpy.context.scene.timeline_markers.clear()

    for i in range(num_cameras):
        current_frame = i # Frame numbers start from 0 in Blender.
        style_to_try = selectable_styles[i % len(selectable_styles)]

        cam_loc, rotation_euler, focal_length, actual_style_applied = \
            _calculate_camera_parameters_for_style(
                vertices, style_to_try, target_objects, wall_thickness, wall_height
            )

        bpy.context.scene.frame_set(current_frame)
        main_camera.location = cam_loc
        main_camera.rotation_euler = rotation_euler
        main_camera.data.lens = focal_length

        main_camera.keyframe_insert(data_path="location", frame=current_frame)
        main_camera.keyframe_insert(data_path="rotation_euler", frame=current_frame)
        main_camera.data.keyframe_insert(data_path="lens", frame=current_frame)

        returned_cameras_list.append(main_camera)
        styles_used_for_keyframes.append(actual_style_applied)

        marker = bpy.context.scene.timeline_markers.new(
            f"View_{current_frame}_{actual_style_applied}", frame=current_frame
        )
        marker.camera = main_camera

    bpy.context.scene.frame_set(1)

    print(f"Configured 1 camera ('{main_camera.name}') with {num_cameras} keyframed views:")
    for i, style in enumerate(styles_used_for_keyframes):
        print(f"  Frame {i + 1}: Style '{style}'")

    return returned_cameras_list, styles_used_for_keyframes