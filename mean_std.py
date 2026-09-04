import argparse
from pathlib import Path

import cv2
import numpy as np

from project_paths import DATA_DIR


def image_mean_std(paths):
    mean_sum = 0.0
    square_sum = 0.0
    pixel_count = 0

    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        if image.ndim == 3:
            image = image[:, :, 0]
        image = image.astype(np.float64) / 255.0
        mean_sum += np.sum(image)
        square_sum += np.sum(np.square(image))
        pixel_count += image.size

    if pixel_count == 0:
        return 0.0, 0.0

    mean = mean_sum / pixel_count
    std = np.sqrt(square_sum / pixel_count - mean**2)
    return mean, std


def calculate_mean_std(folder_path: Path, file_pattern: str):
    if not folder_path.is_dir():
        raise FileNotFoundError(f"Image folder not found: {folder_path}")

    paths = sorted(
        path for path in folder_path.rglob("*.png")
        if file_pattern in path.name
    )
    print(f"Processing {len(paths)} files for {file_pattern}...")
    return image_mean_std(paths)


def calculate_dem_mean_std(folder_path: Path):
    if not folder_path.is_dir():
        raise FileNotFoundError(f"DEM folder not found: {folder_path}")

    paths = sorted(folder_path.rglob("*.png"))
    print(f"Processing {len(paths)} files for DEM...")
    return image_mean_std(paths)


def main():
    parser = argparse.ArgumentParser(description="Calculate water-depth and DEM image statistics.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_DIR / "hsinchu",
        help="regional dataset root containing tuflow/ and dem/",
    )
    parser.add_argument("--tuflow-dir", type=Path, help="override the TUFLOW directory")
    parser.add_argument("--dem-dir", type=Path, help="override the DEM directory")
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    tuflow_dir = (args.tuflow_dir or data_root / "tuflow").expanduser().resolve()
    dem_dir = (args.dem_dir or data_root / "dem").expanduser().resolve()

    print("=== Calculating statistics for depth (d) ===")
    d_mean, d_std = calculate_mean_std(tuflow_dir / "d", "_d_")
    print(f"Depth - Mean: {d_mean}, Std: {d_std}")

    print("\n=== Calculating statistics for DEM ===")
    dem_mean, dem_std = calculate_dem_mean_std(dem_dir)
    print(f"DEM - Mean: {dem_mean}, Std: {dem_std}")


if __name__ == "__main__":
    main()
