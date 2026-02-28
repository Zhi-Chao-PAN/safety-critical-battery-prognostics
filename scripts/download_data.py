"""
Download public battery datasets.

Usage:
    python scripts/download_data.py --dataset nasa
    python scripts/download_data.py --dataset all
"""

import argparse
import logging
import os
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent

DATASETS = {
    "nasa": {
        "url": "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip",
        "dest": ROOT / "data" / "battery_data",
        "description": "NASA PCoE Battery Dataset (B0005, B0006, B0007, B0018)",
    },
}


def download_file(url: str, dest: Path) -> Path:
    """Download a file with progress."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    filename = dest / url.split("/")[-1]
    if filename.exists():
        logger.info(f"Already exists: {filename}")
        return filename

    logger.info(f"Downloading: {url}")
    urlretrieve(url, str(filename))
    logger.info(f"Saved: {filename}")
    return filename


def extract_zip(zip_path: Path, dest: Path):
    """Extract zip file."""
    logger.info(f"Extracting: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest)
    logger.info(f"Extracted to: {dest}")


def download_nasa():
    """Download NASA PCoE battery dataset."""
    info = DATASETS["nasa"]
    dest = info["dest"]
    dest.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    mat_files = list(dest.glob("*.mat"))
    if len(mat_files) >= 4:
        logger.info(f"NASA data already present: {len(mat_files)} .mat files")
        return

    logger.info("Downloading NASA PCoE Battery Dataset...")
    logger.info("Note: If automatic download fails, manually download from:")
    logger.info("  https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository")
    logger.info(f"  Place .mat files (B0005, B0006, B0007, B0018) in: {dest}")

    try:
        zip_path = download_file(info["url"], dest.parent)
        extract_zip(zip_path, dest.parent)
        # Move .mat files to correct location
        for mat in (dest.parent).rglob("*.mat"):
            target = dest / mat.name
            if not target.exists():
                mat.rename(target)
                logger.info(f"  Moved: {mat.name}")
    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.info("Please download manually and place .mat files in:")
        logger.info(f"  {dest}")


def main():
    parser = argparse.ArgumentParser(description="Download battery datasets")
    parser.add_argument("--dataset", default="nasa", choices=["nasa", "all"])
    args = parser.parse_args()

    if args.dataset in ("nasa", "all"):
        download_nasa()

    logger.info("Done.")


if __name__ == "__main__":
    main()
