import blenderproc as bproc
import os
import sys
import numpy as np

# Use current working directory instead of trying to detect script location
project_dir = os.getcwd()
print(f"Project directory: {project_dir}")

# Add project directory to Python path
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
    print(f"Added {project_dir} to sys.path")

# Check if generators directory exists
generators_path = os.path.join(project_dir, "procedural_room_generator")
print(f"Looking for generators at: {generators_path}")
print(f"Generators directory exists: {os.path.exists(generators_path)}")

if os.path.exists(generators_path):
    print(f"Contents of generators directory: {os.listdir(generators_path)}")
    # Check for __init__.py
    init_file = os.path.join(generators_path, "__init__.py")
    print(f"__init__.py exists: {os.path.exists(init_file)}")

# Try the import
try:
    from procedural_room_generator import RandomRoomGenerator
    print("Import successful!")
except ImportError as e:
    print(f"Import failed: {e}")

bproc.init()

#---------------------------------------------------------------------------------------------------
# GENERATE GEOMETRY
#---------------------------------------------------------------------------------------------------

def generate_random_room(seed=None):
    """Generate a random room with the given seed"""
    generator = RandomRoomGenerator(seed)
    return generator.generate_room_with_objects(
        object_dataset_path="../models_metadata.json",
        populate_floor=True,
        # object_count=12,  # Adjusted for a more manageable number of objects
        object_count=int(round(np.random.normal(80, 25)))
    )

#---------------------------------------------------------------------------------------------------
# RENDER STEP
#---------------------------------------------------------------------------------------------------

def renderOutput():
    # Render the whole pipeline
    data = bproc.renderer.render()

    # Write the data to a .hdf5 container
    output_dir = "output"
    bproc.writer.write_hdf5(output_dir, data, append_to_existing_output=True)

# Enable depth output for rendering
bproc.renderer.enable_depth_output(activate_antialiasing=False)

# Edit to enable multiple scenes per run.
SCENE_COUNT = 1
for i in range(SCENE_COUNT):
    print('\n\n')
    print(50 * '*')
    print(f'\nCreating scene {i} of {SCENE_COUNT}\n')

    bproc.clean_up()
    room_data = generate_random_room()

    # Set cam resolution
    bproc.camera.set_resolution(640, 420)

    bproc.renderer.set_light_bounces(
        diffuse_bounces=4,
        glossy_bounces=4,
        ao_bounces_render=1,
        transmission_bounces=12,
        volume_bounces=0,
        max_bounces=12
    )

    renderOutput()

    print(f'\nScene {i}/{SCENE_COUNT} complete\n')
    print(50 * '*')
