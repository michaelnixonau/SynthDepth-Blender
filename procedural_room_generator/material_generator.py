import numpy as np
from mathutils import Color


class MaterialGenerator:
    """Handles creation of randomized materials for room elements"""

    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    def generate_wall_color(self):
        """Generate randomized wall colors with weighted distribution"""
        color_type = np.random.random()

        if color_type < 0.60:  # 60% - Pure whites and off-whites
            # Various shades of white
            value = np.random.uniform(0.85, 1.0)

            # Sometimes add very subtle warmth or coolness
            if np.random.random() < 0.3:
                # Warm white
                r = value
                g = value * np.random.uniform(0.98, 1.0)
                b = value * np.random.uniform(0.94, 0.98)
            elif np.random.random() < 0.3:
                # Cool white
                r = value * np.random.uniform(0.96, 0.99)
                g = value * np.random.uniform(0.98, 1.0)
                b = value
            else:
                # Pure white/gray
                r = g = b = value

        elif color_type < 0.85:  # 25% - Pastels and light tints
            # Choose a base hue
            hue = np.random.uniform(0, 1)

            # High value (brightness) and low saturation for pastel effect
            saturation = np.random.uniform(0.05, 0.25)
            value = np.random.uniform(0.85, 0.95)

            # Convert HSV to RGB
            color = Color()
            color.hsv = (hue, saturation, value)
            r, g, b = color.r, color.g, color.b

        elif color_type < 0.95:  # 10% - Richer colors but still light
            # Medium saturation colors
            hue = np.random.uniform(0, 1)
            saturation = np.random.uniform(0.15, 0.4)
            value = np.random.uniform(0.7, 0.9)

            color = Color()
            color.hsv = (hue, saturation, value)
            r, g, b = color.r, color.g, color.b

        else:  # 5% - Statement colors (darker or more saturated)
            color_style = np.random.choice(['dark', 'saturated'])

            if color_style == 'dark':
                # Dark colors - could be dark gray, navy, forest green, etc.
                hue = np.random.uniform(0, 1)
                saturation = np.random.uniform(0.1, 0.5)
                value = np.random.uniform(0.2, 0.5)
            else:
                # Saturated colors - bold but not too dark
                hue = np.random.uniform(0, 1)
                saturation = np.random.uniform(0.5, 0.8)
                value = np.random.uniform(0.5, 0.7)

            color = Color()
            color.hsv = (hue, saturation, value)
            r, g, b = color.r, color.g, color.b

        return (r, g, b, 1.0)

    def generate_door_color(self):
        """Generate randomized door colors with weighted distribution"""
        color_type = np.random.random()

        if color_type < 0.70:  # 70% - Various woods and browns
            wood_style = np.random.choice(['light', 'medium', 'dark', 'red', 'yellow'])

            if wood_style == 'light':
                # Light woods - pine, birch, maple
                r = np.random.uniform(0.75, 0.9)
                g = np.random.uniform(0.65, 0.8)
                b = np.random.uniform(0.45, 0.6)

            elif wood_style == 'medium':
                # Medium woods - oak, standard brown
                r = np.random.uniform(0.5, 0.7)
                g = np.random.uniform(0.3, 0.5)
                b = np.random.uniform(0.15, 0.3)

            elif wood_style == 'dark':
                # Dark woods - walnut, mahogany
                r = np.random.uniform(0.25, 0.4)
                g = np.random.uniform(0.15, 0.25)
                b = np.random.uniform(0.08, 0.15)

            elif wood_style == 'red':
                # Reddish woods - cherry, mahogany tint
                r = np.random.uniform(0.5, 0.65)
                g = np.random.uniform(0.25, 0.35)
                b = np.random.uniform(0.15, 0.25)

            else:  # yellow
                # Yellowish woods - honey oak, bamboo
                r = np.random.uniform(0.7, 0.85)
                g = np.random.uniform(0.55, 0.7)
                b = np.random.uniform(0.3, 0.45)

        elif color_type < 0.85:  # 15% - White and gray doors
            # Modern painted doors
            value = np.random.uniform(0.85, 1.0)

            if np.random.random() < 0.7:
                # Pure white/gray
                r = g = b = value
            else:
                # Slightly warm or cool gray
                base = value
                variation = np.random.uniform(-0.02, 0.02)
                r = base + variation * np.random.choice([-1, 1])
                g = base
                b = base + variation * np.random.choice([-1, 1])
                # Clamp values
                r = max(0, min(1, r))
                b = max(0, min(1, b))

        else:  # 15% - Statement colors
            # Bold door colors - navy, forest green, burgundy, black, bright colors
            statement_type = np.random.choice(['classic', 'modern', 'bold'])

            if statement_type == 'classic':
                # Classic statement colors
                color_choice = np.random.choice(['navy', 'forest', 'burgundy', 'black'])

                if color_choice == 'navy':
                    r = np.random.uniform(0.05, 0.15)
                    g = np.random.uniform(0.1, 0.2)
                    b = np.random.uniform(0.3, 0.45)
                elif color_choice == 'forest':
                    r = np.random.uniform(0.05, 0.15)
                    g = np.random.uniform(0.2, 0.35)
                    b = np.random.uniform(0.05, 0.15)
                elif color_choice == 'burgundy':
                    r = np.random.uniform(0.4, 0.55)
                    g = np.random.uniform(0.05, 0.15)
                    b = np.random.uniform(0.1, 0.2)
                else:  # black
                    value = np.random.uniform(0.02, 0.1)
                    r = g = b = value

            elif statement_type == 'modern':
                # Modern colors - teal, sage, dusty blue, terracotta
                hue = np.random.choice([0.5, 0.33, 0.58, 0.08])  # Approximate hues
                saturation = np.random.uniform(0.3, 0.5)
                value = np.random.uniform(0.4, 0.6)

                color = Color()
                color.hsv = (hue, saturation, value)
                r, g, b = color.r, color.g, color.b

            else:  # bold
                # Really bold colors - bright red, yellow, blue, etc.
                hue = np.random.uniform(0, 1)
                saturation = np.random.uniform(0.7, 0.95)
                value = np.random.uniform(0.6, 0.85)

                color = Color()
                color.hsv = (hue, saturation, value)
                r, g, b = color.r, color.g, color.b

        return (r, g, b, 1.0)

    def generate_frame_color(self, door_color=None):
        """Generate door/window frame color - usually matches door or is white/neutral"""
        if door_color and np.random.random() < 0.6:
            # 60% chance to match door color (slightly lighter/darker)
            variation = np.random.uniform(0.8, 1.2)
            r = min(1.0, door_color[0] * variation)
            g = min(1.0, door_color[1] * variation)
            b = min(1.0, door_color[2] * variation)
            return (r, g, b, 1.0)
        else:
            # White or neutral frame
            value = np.random.uniform(0.85, 1.0)
            return (value, value, value, 1.0)

    def generate_ceiling_color(self, wall_color=None):
        """Generate ceiling color - usually white or slightly lighter than walls"""
        if wall_color and np.random.random() < 0.3:
            # 30% chance to tint based on wall color
            # Make it lighter than walls
            r = wall_color[0] + (1.0 - wall_color[0]) * 0.5
            g = wall_color[1] + (1.0 - wall_color[1]) * 0.5
            b = wall_color[2] + (1.0 - wall_color[2]) * 0.5
            return (r, g, b, 1.0)
        else:
            # Pure white or off-white ceiling
            value = np.random.uniform(0.92, 1.0)
            return (value, value, value, 1.0)

    def generate_floor_color(self):
        """Generate floor color - wood, tile, or concrete look"""
        floor_type = np.random.choice(['wood', 'tile', 'concrete'], p=[0.6, 0.25, 0.15])

        if floor_type == 'wood':
            # Similar to door woods but often different shade
            wood_tone = np.random.choice(['light', 'medium', 'dark'])

            if wood_tone == 'light':
                r = np.random.uniform(0.75, 0.85)
                g = np.random.uniform(0.65, 0.75)
                b = np.random.uniform(0.5, 0.6)
            elif wood_tone == 'medium':
                r = np.random.uniform(0.55, 0.65)
                g = np.random.uniform(0.4, 0.5)
                b = np.random.uniform(0.25, 0.35)
            else:  # dark
                r = np.random.uniform(0.25, 0.35)
                g = np.random.uniform(0.18, 0.25)
                b = np.random.uniform(0.12, 0.18)

        elif floor_type == 'tile':
            # Tile colors - often neutral or terracotta
            tile_style = np.random.choice(['white', 'gray', 'beige', 'terracotta'])

            if tile_style == 'white':
                value = np.random.uniform(0.85, 0.95)
                r = g = b = value
            elif tile_style == 'gray':
                value = np.random.uniform(0.4, 0.7)
                r = g = b = value
            elif tile_style == 'beige':
                r = np.random.uniform(0.75, 0.85)
                g = np.random.uniform(0.7, 0.8)
                b = np.random.uniform(0.6, 0.7)
            else:  # terracotta
                r = np.random.uniform(0.7, 0.8)
                g = np.random.uniform(0.45, 0.55)
                b = np.random.uniform(0.35, 0.45)

        else:  # concrete
            # Concrete colors - various grays
            value = np.random.uniform(0.45, 0.65)
            # Add slight color variation
            variation = np.random.uniform(-0.02, 0.02)
            r = value + variation
            g = value
            b = value - variation * 0.5
            # Clamp values
            r = max(0, min(1, r))
            g = max(0, min(1, g))
            b = max(0, min(1, b))

        return (r, g, b, 1.0)


# Integration with existing RandomRoomGenerator class
# Add this method to the RandomRoomGenerator class:

def init_material_generator(self):
    """Initialize the material generator with consistent seed"""
    seed = None
    if hasattr(self, 'config') and hasattr(self.config, 'seed'):
        seed = self.config.seed
    self.material_gen = MaterialGenerator(seed)


# Then update the material creation calls in the existing methods.
# Here are the modified methods to add to RandomRoomGenerator:

def create_enhanced_material(self, name, color, roughness=0.5, metallic=0.0):
    """Create a material with enhanced properties"""
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