from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import numpy as np
import h5py
from PIL import Image
import glob
import json


def load_exclusions(dataset_dir: Path) -> set:
    """Load excluded files from the exclusion file."""
    exclusion_file = dataset_dir / "excluded_files.json"
    if exclusion_file.exists():
        try:
            with open(exclusion_file, 'r') as f:
                return set(json.load(f))
        except Exception as e:
            print(f"Warning: Could not load exclusions: {e}")
    return set()


def normalise_to_uint8(arr: np.ndarray, is_depth: bool = False) -> np.ndarray:
    arr = arr.astype(np.float32)
    if is_depth:
        arr = arr - arr.min()
        rng = arr.max() if arr.max() > 0 else 1.0
        arr = arr / rng
        arr = 1.0 - arr
    else:
        arr = arr - arr.min()
        rng = arr.max() if arr.max() > 0 else 1.0
        arr = arr / rng
    return (arr * 255).clip(0, 255).astype(np.uint8)


def get_next_index(folder: Path, prefix: str) -> int:
    pattern = str(folder / f"{prefix}_*.png")
    files = glob.glob(pattern)
    max_idx = -1
    for f in files:
        name = Path(f).stem
        try:
            idx = int(name.split("_")[1])
            if idx > max_idx:
                max_idx = idx
        except Exception:
            continue
    return max_idx + 1


def process_hdf5_file(args):
    h5_file, rgb_dir, depth_dir, idx = args
    with h5py.File(h5_file, "r") as f:
        rgb = f["/colors"][()]
        depth = f["/depth"][()]
    if rgb.ndim == 4:
        rgb = rgb[0]
    if depth.ndim == 3:
        if depth.shape[2] == 1:
            depth = depth[:, :, 0]
        else:
            depth = depth.mean(axis=2)
    rgb_img = Image.fromarray(normalise_to_uint8(rgb))
    depth_img = Image.fromarray(normalise_to_uint8(depth, is_depth=True), mode="L")
    rgb_filename = f"rgb_{idx:05d}.png"
    depth_filename = f"depth_{idx:05d}.png"
    rgb_img.save(rgb_dir / rgb_filename)
    depth_img.save(depth_dir / depth_filename)
    return str(h5_file)


def save_pngs_from_hdf5(input_dir: str, output_dir: str, num_workers: int = None):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Load exclusions
    excluded_files = load_exclusions(input_dir)

    rgb_dir = output_dir / "rgb"
    depth_dir = output_dir / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    start_idx = max(get_next_index(rgb_dir, "rgb"), get_next_index(depth_dir, "depth"))

    # Get all HDF5 files and filter out excluded ones
    all_hdf5_files = sorted(input_dir.glob("*.hdf5"))
    hdf5_files = [f for f in all_hdf5_files if f.name not in excluded_files]

    excluded_count = len(all_hdf5_files) - len(hdf5_files)
    if excluded_count > 0:
        print(f"Excluding {excluded_count} files based on exclusion list")

    args = [(h5_file, rgb_dir, depth_dir, idx) for idx, h5_file in enumerate(hdf5_files, start=start_idx)]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for f in executor.map(process_hdf5_file, args):
            print(f"Processed: {f}")

    print(f"\nProcessed {len(hdf5_files)} files total")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input folder with .hdf5 files")
    parser.add_argument("--output", required=True, help="Output folder")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (default: all CPUs)")
    args = parser.parse_args()
    save_pngs_from_hdf5(args.input, args.output, num_workers=args.workers)