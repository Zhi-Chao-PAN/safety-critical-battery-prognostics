"""
Download NASA PCoE Battery Dataset.
Source: https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository
"""

import logging
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# NASA PCoE battery data URLs (direct .mat files)
# These are the commonly used Li-ion battery aging datasets
NASA_BATTERIES = {
    "B0005": "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set/B0005.mat",
    "B0006": "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set/B0006.mat",
    "B0007": "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set/B0007.mat",
    "B0018": "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set/B0018.mat",
}


def download_nasa(data_dir: str = "data/battery_data"):
    """Download NASA PCoE battery .mat files."""
    out = Path(data_dir)
    out.mkdir(parents=True, exist_ok=True)

    for bat_id, url in NASA_BATTERIES.items():
        filepath = out / f"{bat_id}.mat"
        if filepath.exists():
            logger.info(f"{bat_id}: Already exists, skipping")
            continue

        logger.info(f"Downloading {bat_id} from {url}...")
        try:
            urllib.request.urlretrieve(url, str(filepath))
            size_mb = filepath.stat().st_size / (1024 * 1024)
            logger.info(f"  Saved: {filepath} ({size_mb:.1f} MB)")
        except Exception as e:
            logger.error(f"  Failed to download {bat_id}: {e}")
            logger.info(f"  Manual download: {url}")

    logger.info("Done. Place .mat files in data/battery_data/ if auto-download failed.")


if __name__ == "__main__":
    download_nasa()
