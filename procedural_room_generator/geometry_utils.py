import random
import math
from mathutils import Vector
import numpy as np

def clamp(val, minv, maxv):
    return max(minv, min(maxv, val))

def point_in_polygon_2d(point, poly):
    """
    Returns True if the 2D point is inside the 2D polygon defined by poly (list of Vectors).
    Uses the ray casting algorithm.
    """
    x, y = point.x, point.y
    inside = False
    n = len(poly)
    for i in range(n):
        v1 = poly[i]
        v2 = poly[(i + 1) % n]
        xi, yi = v1.x, v1.y
        xj, yj = v2.x, v2.y
        if (yi > y) != (yj > y):
            x_intersect = (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
            if x < x_intersect:
                inside = not inside
    return inside

def distance_point_to_segment_2d(p, v1, v2):
    """Distance from point p to segment v1-v2 in 2D"""
    # project p onto segment
    a = v1.xy
    b = v2.xy
    ap = p.xy - a
    ab = b - a
    t = max(0.0, min(1.0, ap.dot(ab) / (ab.length_squared if ab.length_squared > 0 else 1.0)))
    proj = a + ab * t
    return (p.xy - proj).length

def sample_farthest_camera_position(vertices, margin, n_samples=64):
    """
    Find a camera position inside the room that is as far from the walls as possible.
    Returns the best candidate point.
    """
    # Room bounds
    xs = [v.x for v in vertices]
    ys = [v.y for v in vertices]
    min_x, max_x = min(xs) + margin, max(xs) - margin
    min_y, max_y = min(ys) + margin, max(ys) - margin

    # Sample a grid or random points and score by distance to closest wall
    candidates = []
    for _ in range(n_samples):
        x = np.random.uniform(min_x, max_x)
        y = np.random.uniform(min_y, max_y)
        p = Vector((x, y, 0))
        if not point_in_polygon_2d(p, vertices):
            continue
        # Score = minimum distance to any wall
        min_dist = min(distance_point_to_segment_2d(p, v1, v2) for v1, v2 in zip(vertices, vertices[1:] + vertices[:1]))
        candidates.append((min_dist, p))
    if not candidates:
        # fallback: centroid
        center = Vector((0, 0, 0))
        for v in vertices:
            center += v
        center /= len(vertices)
        return center
    candidates.sort(reverse=True)  # largest min_dist first
    return candidates[0][1]

def random_sun_direction():
    # Define ranges for daylight and twilight hours
    # Elevation (degrees): 0 = horizon, 90 = zenith (straight up)
    # Daylight: 10° to 70° above the horizon
    # Dusk/pre-dawn: -6° (just below) to 10° (just above) the horizon
    # Combine these ranges to avoid full nighttime elevations

    # Decide if we're in daylight or twilight period
    if random.random() < 0.75:
        # 75% chance for full daylight
        elevation_deg = random.uniform(10, 70)
    else:
        # 25% chance for twilight (dusk or pre-dawn)
        elevation_deg = random.uniform(-6, 10)

    azimuth_deg = random.uniform(0, 360)  # Any direction along the horizon

    # Convert to radians
    elevation_rad = math.radians(elevation_deg)
    azimuth_rad = math.radians(azimuth_deg)

    # Spherical to cartesian conversion
    x = math.cos(elevation_rad) * math.cos(azimuth_rad)
    y = math.cos(elevation_rad) * math.sin(azimuth_rad)
    z = math.sin(elevation_rad)

    # Return as a tuple
    return (x, y, z)