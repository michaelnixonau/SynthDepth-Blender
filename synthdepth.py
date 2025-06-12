import subprocess

total_scenes = 500

for i in range(0, total_scenes):
    subprocess.run([
        'blenderproc', 'run', 'generate_scene.py'
    ])