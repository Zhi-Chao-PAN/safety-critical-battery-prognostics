"""
QLoRA Fine-Tuning Script for Chronos-T5 on Battery Degradation Data.

Pre-built locally for deployment on AutoDL (A40/A100).
Requires: peft, bitsandbytes, transformers >= 4.30.0

Usage (on GPU server):
    python scripts/run_chronos_finetune.py
    python scripts/run_chronos_finetune.py --config configs/chronos_finetune.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("chronos_finetune")


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def check_dependencies() -> dict[str, bool]:
    """Check if all required packages are installed."""
    deps = {}
    for pkg in ["peft", "bitsandbytes", "transformers", "accelerate", "chronos"]:
        try:
            __import__(pkg)
            deps[pkg] = True
        except ImportError:
            deps[pkg] = False
    return deps


def setup_quantization_config(quant_cfg: dict):
    """Create BitsAndBytesConfig for 4-bit quantization."""
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=quant_cfg.get("load_in_4bit", True),
        bnb_4bit_compute_dtype=getattr(torch, quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16")),
        bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
    )


def setup_lora_config(lora_cfg: dict):
    """Create LoRA configuration for PEFT."""
    from peft import LoraConfig, TaskType

    task_type_map = {
        "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
        "CAUSAL_LM": TaskType.CAUSAL_LM,
    }

    return LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        target_modules=lora_cfg.get("target_modules", ["q", "v"]),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        task_type=task_type_map.get(lora_cfg.get("task_type", "SEQ_2_SEQ_LM"), TaskType.SEQ_2_SEQ_LM),
    )


def prepare_data(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and split data into train/val DataFrames."""
    from src.data.unified_loader import UnifiedDataLoader

    loader = UnifiedDataLoader()
    df = loader.load_nasa(data_dir=str(ROOT / cfg["dataset"]["data_dir"]))

    train_bats = cfg["dataset"].get("train_batteries", ["B0005", "B0006", "B0007"])
    val_bats = cfg["dataset"].get("val_batteries", ["B0018"])

    train_df = df[df["battery_id"].isin(train_bats)]
    val_df = df[df["battery_id"].isin(val_bats)]

    logger.info(f"  Train: {len(train_df)} cycles from {train_bats}")
    logger.info(f"  Val:   {len(val_df)} cycles from {val_bats}")

    return train_df, val_df


def build_dataloaders(train_df: pd.DataFrame, val_df: pd.DataFrame, cfg: dict):
    """Build PyTorch DataLoaders using ChronosFineTuningDataset."""
    from src.data.chronos_dataset import get_chronos_dataloaders

    ds_cfg = cfg["dataset"]
    tr_cfg = cfg["training"]

    return get_chronos_dataloaders(
        train_df=train_df,
        val_df=val_df,
        context_length=ds_cfg.get("context_length", 64),
        prediction_length=ds_cfg.get("prediction_length", 20),
        stride=ds_cfg.get("stride", 5),
        batch_size=tr_cfg.get("batch_size", 16),
    )


def load_model_and_tokenizer(cfg: dict):
    """Load Chronos pipeline, extract tokenizer, and apply LoRA adapters to model."""
    from chronos import ChronosPipeline
    from peft import get_peft_model, prepare_model_for_kbit_training

    model_id = cfg["model"]["model_id"]
    lora_cfg = cfg.get("lora", {})
    quant_cfg = cfg.get("quantization", {})

    logger.info(f"  Loading Chronos pipeline: {model_id}")

    kwargs = {"device_map": "auto", "trust_remote_code": True}

    # Step 1: Load with quantization
    if quant_cfg.get("load_in_4bit", False):
        kwargs["quantization_config"] = setup_quantization_config(quant_cfg)
        logger.info("  Enabled 4-bit quantization (NF4) for loading")
    else:
        dtype_str = cfg["model"].get("torch_dtype", "float32")
        kwargs["torch_dtype"] = getattr(torch, dtype_str)

    pipeline = ChronosPipeline.from_pretrained(model_id, **kwargs)
    model = pipeline.model
    tokenizer = pipeline.tokenizer

    # Step 2: Prepare for k-bit training
    if quant_cfg.get("load_in_4bit", False):
        model = prepare_model_for_kbit_training(model)
        logger.info("  Prepared for k-bit training")

    # Step 3: Enable gradient checkpointing
    if cfg["training"].get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        logger.info("  Gradient checkpointing enabled")

    # Step 4: Apply LoRA
    if lora_cfg.get("enabled", True):
        lora_config = setup_lora_config(lora_cfg)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        logger.info("  LoRA adapters applied")

    return model, tokenizer


def compute_loss(model, tokenizer, batch: dict, device: str) -> torch.Tensor:
    """
    Compute auto-regressive loss for Chronos fine-tuning.
    
    Uses Chronos's native MeanScaleUniformBins tokenizer to quantize
    continuous capacities into vocabulary tokens for teacher forcing.
    """
    past = batch["past_values"].to(device)      # [B, ctx_len]
    future = batch["future_values"].to(device)   # [B, pred_len]

    # 1. Transform context to get input_ids and the normalization scale
    token_ids, attention_mask, scale = tokenizer.context_input_transform(past)

    # 2. Transform future labels using the SAME scale from context
    label_token_ids, label_attention_mask = tokenizer.label_input_transform(future, scale)

    # 3. Forward pass (T5 shifts labels to the right automatically for decoder input)
    outputs = model(
        input_ids=token_ids.to(device),
        attention_mask=attention_mask.to(device),
        labels=label_token_ids.to(device),
    )

    return outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=device)


def run_training(cfg: dict) -> None:
    """Main training loop."""
    tr_cfg = cfg["training"]

    # Set seed
    seed = tr_cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"  Device: {device}")

    if device == "cpu":
        logger.warning("  WARNING: Fine-tuning on CPU is extremely slow.")
        logger.warning("  This script is designed for GPU (AutoDL A40/A100).")
        logger.warning("  Running in dry-run mode for local validation only.")

    # Prepare data
    train_df, val_df = prepare_data(cfg)
    train_loader, val_loader = build_dataloaders(train_df, val_df, cfg)

    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches:   {len(val_loader)}")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(cfg)

    # Optimizer
    lr = tr_cfg.get("learning_rate", 1e-4)
    wd = tr_cfg.get("weight_decay", 0.01)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=wd,
    )

    # Training loop
    num_epochs = tr_cfg.get("num_epochs", 20)
    output_dir = Path(tr_cfg.get("output_dir", "checkpoints/chronos_finetune"))
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n  Starting training: {num_epochs} epochs, lr={lr}, wd={wd}")
    logger.info(f"  Output: {output_dir}")

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()
            loss = compute_loss(model, tokenizer, batch, device)

            if loss.requires_grad:
                loss.backward()
                max_norm = tr_cfg.get("max_grad_norm", 1.0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                loss = compute_loss(model, tokenizer, batch, device)
                val_loss += loss.item()
                n_val += 1

        avg_val_loss = val_loss / max(n_val, 1)

        logger.info(f"  Epoch {epoch+1}/{num_epochs}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = output_dir / "best_model"
            model.save_pretrained(str(save_path))
            logger.info(f"    Best model saved: {save_path}")

    # VRAM cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("  CUDA cache cleared")

    logger.info("  Training complete!")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chronos QLoRA Fine-Tuning")
    p.add_argument("--config", default=str(ROOT / "configs" / "chronos_finetune.yaml"))
    p.add_argument("--dry-run", action="store_true", help="Validate setup without training")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Check dependencies
    deps = check_dependencies()
    print("\n" + "=" * 60)
    print("  CHRONOS QLoRA FINE-TUNING - Dependency Check")
    print("=" * 60)
    for pkg, installed in deps.items():
        status = "OK" if installed else "MISSING"
        print(f"  {pkg:<20} {status}")

    missing = [k for k, v in deps.items() if not v]
    if missing:
        print(f"\n  Missing packages: {missing}")
        print("  Install with: pip install peft bitsandbytes")
        if not args.dry_run:
            print("  Use --dry-run to validate configuration without training.")
            return

    # Load config
    cfg = load_config(args.config)
    logger.info(f"  Config loaded: {args.config}")

    if args.dry_run:
        print("\n  DRY RUN: Validating data pipeline...")
        train_df, val_df = prepare_data(cfg)
        train_loader, val_loader = build_dataloaders(train_df, val_df, cfg)
        print(f"  Train: {len(train_loader)} batches")
        print(f"  Val:   {len(val_loader)} batches")

        batch = next(iter(train_loader))
        print(f"  Batch past_values shape: {batch['past_values'].shape}")
        print(f"  Batch future_values shape: {batch['future_values'].shape}")
        print("  DRY RUN validation PASSED")
        return

    run_training(cfg)


if __name__ == "__main__":
    main()
