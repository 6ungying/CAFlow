import argparse
from pathlib import Path

import cv2
import numpy as np

from project_paths import OUTPUT_DIR


def contrast_stretching(image):
    image = (image - 200) / (255 - 200) * 255
    return np.clip(image, 0, 255).astype(np.uint8)


def read_image(path: Path, grayscale=False):
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_UNCHANGED
    image = cv2.imread(str(path), mode)
    if image is None:
        raise FileNotFoundError(f"Image not found or unreadable: {path}")
    if not grayscale and image.ndim == 3:
        image = image[:, :, 0]
    return image


def main():
    parser = argparse.ArgumentParser(description="Create cropped model-output visualizations.")
    parser.add_argument("image", type=Path, help="generated image to visualize")
    parser.add_argument(
        "--reference-image",
        type=Path,
        help="optional reference image used to produce an absolute-difference image",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "paper_vis" / "crop.png",
        help="cropped output image",
    )
    parser.add_argument(
        "--diff-output",
        type=Path,
        help="optional absolute-difference output image",
    )
    parser.add_argument(
        "--crop",
        type=int,
        nargs=4,
        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
        default=(30, 20, 130, 120),
        help="crop rectangle in pixels",
    )
    args = parser.parse_args()

    image = read_image(args.image.expanduser().resolve())
    image = contrast_stretching(image)

    left, top, right, bottom = args.crop
    if not (0 <= left < right <= image.shape[1] and 0 <= top < bottom <= image.shape[0]):
        raise ValueError(f"Crop {args.crop} is outside image dimensions {image.shape[1]}x{image.shape[0]}")

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image[top:bottom, left:right])

    if args.reference_image:
        reference = read_image(args.reference_image.expanduser().resolve(), grayscale=True)
        if reference.shape != image.shape:
            raise ValueError(f"Image shapes do not match: generated={image.shape}, reference={reference.shape}")
        if args.diff_output:
            diff_path = args.diff_output.expanduser().resolve()
            diff_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(diff_path), cv2.absdiff(image, reference))


if __name__ == "__main__":
    main()
