"""
Offline High-Speed ETL Pipeline for Battery Micro Data (Phase 5.3)
War Zone 2: Cross-Battery Saturation with multiprocessing.

Resolves the "Dataloader Bottleneck" trap:
Dynamically parsing multi-gigabyte .xlsx or .txt files inside a PyTorch DataLoader 
will drastically bottleneck the GPU, dropping GPU utilization to <5%.

This ETL block runs Once:
1. Scans raw data directories for CALCE/NASA files.
2. Uses the CalceMicroParser to identify, slice, and 1D-interpolate the micro sequences.
3. Saves the strictly aligned PyTorch Tensors as `.pt` binaries.
4. During training, the runner's DataLoader simply loads these flat `.pt` files in milliseconds.

Now with multiprocessing for saturating 40GB RAM across all CPU cores.
"""

import argparse
import logging
import sys
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import torch

# Ensure src exists in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.calce_micro_parser import CalceMicroParser

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ETL_Pipeline")

def _process_single_file(file_path_str: str, output_dir_str: str, micro_steps: int) -> dict:
    """Worker function for multiprocessing pool."""
    file_path = Path(file_path_str)
    output_dir = Path(output_dir_str)
    parser = CalceMicroParser(target_micro_steps=micro_steps)

    try:
        tensors = parser.slice_cycles_and_align(filepath=str(file_path))

        if not tensors or len(tensors.get("i_app_micro", [])) == 0:
            return {"file": file_path.name, "status": "skipped", "cycles": 0}

        output_file = output_dir / f"{file_path.stem}_micro.pt"
        torch.save(tensors, output_file)

        return {"file": file_path.name, "status": "success", "cycles": len(tensors['i_app_micro'])}

    except Exception as e:
        return {"file": file_path.name, "status": "failed", "error": str(e)}

def run_etl(input_dir: Path, output_dir: Path, micro_steps: int = 100, num_workers: int = None):
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        logger.info("Please create this directory and place original CALCE .xlsx/.txt files there.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Target Output Directory: {output_dir}")

    # Supported raw extensions
    extensions = ('*.xlsx', '*.xls', '*.txt', '*.csv')
    raw_files = []
    for ext in extensions:
        raw_files.extend(input_dir.rglob(ext))

    if not raw_files:
        logger.warning(f"No raw battery data found in {input_dir}.")
        return

    if num_workers is None:
        num_workers = min(cpu_count(), len(raw_files))

    logger.info(f"Found {len(raw_files)} files. Launching {num_workers}-process parallel ETL...")

    worker_fn = partial(_process_single_file, output_dir_str=str(output_dir), micro_steps=micro_steps)
    file_paths = [str(f) for f in raw_files]

    with Pool(processes=num_workers) as pool:
        results = pool.map(worker_fn, file_paths)

    success = sum(1 for r in results if r["status"] == "success")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    failed = sum(1 for r in results if r["status"] == "failed")
    total_cycles = sum(r.get("cycles", 0) for r in results)

    logger.info("=" * 50)
    logger.info("ETL Pipeline Finished (Parallel Mode)")
    logger.info(f"Success: {success} | Skipped: {skipped} | Failed: {failed}")
    logger.info(f"Total Cycles Extracted: {total_cycles}")
    logger.info("=" * 50)

    for r in results:
        if r["status"] == "failed":
            logger.error(f"  FAILED: {r['file']} -> {r.get('error', 'unknown')}")

if __name__ == "__main__":
    cli = argparse.ArgumentParser(description="CALCE Data ETL PyTorch Compiler (Parallel)")
    cli.add_argument("--input_dir", type=str, default="data/raw/calce", help="Path to raw CALCE files.")
    cli.add_argument("--output_dir", type=str, default="data/processed/calce_micro", help="Path for .pt binaries.")
    cli.add_argument("--micro_steps", type=int, default=100, help="FDM alignment dimension.")
    cli.add_argument("--workers", type=int, default=None, help="Number of parallel workers.")
    args = cli.parse_args()

    run_etl(Path(args.input_dir), Path(args.output_dir), args.micro_steps, args.workers)
