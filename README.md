# SynthDepth Blender

A pipeline for generating synthetic RGB-D datasets using BlenderProc and a procedural room generator.

## Project Structure

- `synthdepth.py`           Main entry point for generating multiple scenes.
- `generate_scene.py`       Script that builds and renders a single scene using BlenderProc.
- `make_png_dataset.py`     Convert rendered `.hdf5` files into RGB and depth PNG images.
- `procedural_room_generator/`  Python package for procedural room generation.

## Requirements

- Python 3.7 or higher
- Python packages:
  - [BlenderProc](https://github.com/DLR-RM/BlenderProc)
  - `numpy`
  - `h5py`
  - `Pillow`

You can install the Python dependencies with:

```bash
pip install numpy h5py Pillow blenderproc
```

## Usage

### 1. Generating synthetic depth data

The main entry point is `synthdepth.py`, which invokes BlenderProc on `generate_scene.py`
for a number of scenes.

Edit the `total_scenes` variable in `synthdepth.py` to change how many scenes are generated:

```python
# synthdepth.py
total_scenes = 500  # change this to the desired number of scenes
```

Before running, ensure that `generate_scene.py` is configured with the correct output directory:

```python
# generate_scene.py (around renderOutput)
output_dir = "/path/to/output_folder"  # set this to a valid directory on your system
```

Then run:

```bash
python synthdepth.py
```

This will create one `.hdf5` file per scene in the output directory.

### 2. Converting HDF5 outputs to PNG dataset

Once you have your rendered `.hdf5` files, use `make_png_dataset.py` to convert them into
separate RGB and depth PNG images:

```bash
python make_png_dataset.py --input /path/to/hdf5_folder --output /path/to/output_folder
```

By default, the script will use all available CPU cores. You can limit the number of workers:

```bash
python make_png_dataset.py \
    --input /path/to/hdf5_folder \
    --output /path/to/output_folder \
    --workers 4
```

The output folder will contain two subdirectories:

- `rgb/`   Normalized color images.
- `depth/` Normalized depth images (grayscale).