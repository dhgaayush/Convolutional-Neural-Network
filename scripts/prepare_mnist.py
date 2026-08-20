#!/usr/bin/env python3

"""
prepare_mnist.py

Downloads the official MNIST IDX dataset, validates it, and converts it
into the CSV format expected by this C++ CNN project.

Output format:
    label,pixel1,pixel2,...,pixel784

Outputs:
    resource/mnist_train.csv
    resource/mnist_test.csv

Raw downloads:
    data/raw/
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import shutil
import struct
import sys
import urllib.request
from pathlib import Path


# ---------------------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
RESOURCE_DIR = PROJECT_ROOT / "resource"


# ---------------------------------------------------------------------
# MNIST FILE DEFINITIONS
# ---------------------------------------------------------------------

BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"

FILES = {
    "train_images": {
        "filename": "train-images-idx3-ubyte.gz",
        "url": BASE_URL + "train-images-idx3-ubyte.gz",
        "sha256": (
            "440fcabf73cc546fa21475e81ea370265"
            "605f56be210a4024d2ca8f203523609"
        ),
    },
    "train_labels": {
        "filename": "train-labels-idx1-ubyte.gz",
        "url": BASE_URL + "train-labels-idx1-ubyte.gz",
        "sha256": (
            "3552534a0a558bbed6aed32b30c495cc"
            "a23d567ec52cac8be1a0730e8010255c"
        ),
    },
    "test_images": {
        "filename": "t10k-images-idx3-ubyte.gz",
        "url": BASE_URL + "t10k-images-idx3-ubyte.gz",
        "sha256": (
            "8d422c7b0a1c1c79245a5bcf07fe86e3"
            "3eeafee792b84584aec276f5a2dbc4e6"
        ),
    },
    "test_labels": {
        "filename": "t10k-labels-idx1-ubyte.gz",
        "url": BASE_URL + "t10k-labels-idx1-ubyte.gz",
        "sha256": (
            "f7ae60f92e00ec6debd23a6088c31dbd"
            "2371eca3ffa0defaefb259924204aec6"
        ),
    },
}


# ---------------------------------------------------------------------
# DOWNLOAD + INTEGRITY
# ---------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def download_file(url: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")

    print(f"Downloading: {url}")

    try:
        with urllib.request.urlopen(url) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    temporary.replace(destination)


def ensure_downloaded(info: dict) -> Path:
    destination = RAW_DIR / info["filename"]

    if destination.exists():
        print(f"Found cached file: {destination.name}")
    else:
        download_file(info["url"], destination)

    print(f"Verifying SHA-256: {destination.name}")

    actual_hash = sha256_file(destination)

    if actual_hash != info["sha256"]:
        destination.unlink(missing_ok=True)

        raise RuntimeError(
            f"Checksum verification failed for {destination.name}\n"
            f"Expected: {info['sha256']}\n"
            f"Actual:   {actual_hash}"
        )

    print("  OK")

    return destination


# ---------------------------------------------------------------------
# IDX PARSING
# ---------------------------------------------------------------------

def read_images(path: Path):
    with gzip.open(path, "rb") as file:
        magic, count, rows, cols = struct.unpack(
            ">IIII",
            file.read(16),
        )

        if magic != 2051:
            raise ValueError(
                f"{path.name}: invalid image magic number {magic}"
            )

        if rows != 28 or cols != 28:
            raise ValueError(
                f"{path.name}: expected 28x28 images, "
                f"got {rows}x{cols}"
            )

        expected_bytes = count * rows * cols
        data = file.read()

        if len(data) != expected_bytes:
            raise ValueError(
                f"{path.name}: expected {expected_bytes} pixel bytes, "
                f"got {len(data)}"
            )

    return count, rows, cols, data


def read_labels(path: Path):
    with gzip.open(path, "rb") as file:
        magic, count = struct.unpack(
            ">II",
            file.read(8),
        )

        if magic != 2049:
            raise ValueError(
                f"{path.name}: invalid label magic number {magic}"
            )

        data = file.read()

        if len(data) != count:
            raise ValueError(
                f"{path.name}: expected {count} labels, "
                f"got {len(data)}"
            )

    for label in data:
        if not 0 <= label <= 9:
            raise ValueError(
                f"{path.name}: invalid label value {label}"
            )

    return count, data


# ---------------------------------------------------------------------
# CSV GENERATION
# ---------------------------------------------------------------------

def write_csv(
    output_path: Path,
    images: bytes,
    labels: bytes,
    count: int,
    rows: int,
    cols: int,
    limit: int | None,
) -> None:
    if len(labels) != count:
        raise ValueError(
            f"Image/label count mismatch: "
            f"{count} images vs {len(labels)} labels"
        )

    if limit is None:
        limit = count
    else:
        limit = min(limit, count)

    pixels_per_image = rows * cols

    header = ["label"] + [
        f"pixel{i}" for i in range(1, pixels_per_image + 1)
    ]

    print(f"Writing {limit} samples -> {output_path}")

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(header)

        for index in range(limit):
            start = index * pixels_per_image
            end = start + pixels_per_image

            pixels = images[start:end]

            writer.writerow([labels[index], *pixels])

    print(f"  Wrote {limit} samples")


# ---------------------------------------------------------------------
# DATASET PREPARATION
# ---------------------------------------------------------------------

def prepare_split(
    image_path: Path,
    label_path: Path,
    output_path: Path,
    limit: int | None,
) -> None:
    image_count, rows, cols, images = read_images(image_path)
    label_count, labels = read_labels(label_path)

    if image_count != label_count:
        raise ValueError(
            f"Dataset mismatch: "
            f"{image_count} images vs {label_count} labels"
        )

    write_csv(
        output_path=output_path,
        images=images,
        labels=labels,
        count=image_count,
        rows=rows,
        cols=cols,
        limit=limit,
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download official MNIST IDX files and convert them "
            "to CSV for the C++ CNN project."
        )
    )

    parser.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help="Maximum number of training samples to export",
    )

    parser.add_argument(
        "--test-limit",
        type=int,
        default=None,
        help="Maximum number of test samples to export",
    )

    args = parser.parse_args()

    if args.train_limit is not None and args.train_limit <= 0:
        parser.error("--train-limit must be positive")

    if args.test_limit is not None and args.test_limit <= 0:
        parser.error("--test-limit must be positive")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    RESOURCE_DIR.mkdir(parents=True, exist_ok=True)

    print("========================================")
    print("MNIST Dataset Preparation Pipeline")
    print("========================================")
    print(f"Project root: {PROJECT_ROOT}")
    print()

    downloaded = {}

    for key, info in FILES.items():
        downloaded[key] = ensure_downloaded(info)

    print()
    print("Preparing training dataset...")

    prepare_split(
        image_path=downloaded["train_images"],
        label_path=downloaded["train_labels"],
        output_path=RESOURCE_DIR / "mnist_train.csv",
        limit=args.train_limit,
    )

    print()
    print("Preparing test dataset...")

    prepare_split(
        image_path=downloaded["test_images"],
        label_path=downloaded["test_labels"],
        output_path=RESOURCE_DIR / "mnist_test.csv",
        limit=args.test_limit,
    )

    print()
    print("========================================")
    print("MNIST preparation complete")
    print("========================================")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"\nError: {error}", file=sys.stderr)
        raise SystemExit(1)
