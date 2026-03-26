"""
main_simple.py - Simplified, working entry point for quick start.

This is the "one-click demo" that actually works.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Project root
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main_simple")


def parse_args():
    p = argparse.ArgumentParser(description="Battery Prognostics - Quick Start Demo")
    p.add_argument("--dataset", default="sample", 
                   choices=["sample", "NASA", "CALCE"],
                   help="Dataset to use (default: sample)")
    return p.parse_args()


def create_sample_data():
    """Create a synthetic sample dataset for quick demonstration."""
    logger.info("Creating sample battery data (for demo)...")
    
    np.random.seed(42)
    n_cycles = 100
    
    # Synthetic capacity degradation
    capacity = 2.0 - 0.005 * np.arange(n_cycles) + 0.02 * np.random.randn(n_cycles)
    capacity = np.clip(capacity, 0.8, 2.0)
    
    # Create DataFrame
    df = pd.DataFrame({
        "cycle": np.arange(1, n_cycles + 1),
        "capacity": capacity,
        "battery_id": "SAMPLE001",
    })
    
    # Simple RUL calculation (EOL at 1.4 Ah)
    eol_threshold = 1.4
    rul_values = []
    for i in range(len(df)):
        remaining = len(df) - i - 1
        future = df["capacity"].iloc[i:]
        if (future < eol_threshold).any():
            eol_idx = np.argmax(future < eol_threshold)
            rul_values.append(eol_idx)
        else:
            rul_values.append(remaining)
    
    df["rul"] = rul_values
    
    logger.info(f"Sample data created: {len(df)} cycles")
    return df


def train_simple_model(df):
    """Train a simple baseline model (for demo)."""
    logger.info("Training simple baseline model...")
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    
    # Simple feature: use past 5 cycles
    def create_features(data):
        X, y = [], []
        for i in range(5, len(data)):
            X.append(data["capacity"].iloc[i-5:i].values)
            y.append(data["rul"].iloc[i])
        return np.array(X), np.array(y)
    
    X, y = create_features(df)
    
    if len(X) < 10:
        logger.warning("Not enough data for training")
        return None
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = np.mean(np.abs(y_pred - y_test))
    
    logger.info(f"Training complete. Test MAE: {mae:.2f} cycles")
    return model


def main():
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("  Battery RUL Prediction - Quick Start Demo")
    logger.info("=" * 60)
    
    # Step 1: Load or create data
    if args.dataset == "sample":
        df = create_sample_data()
    else:
        logger.warning(f"Dataset '{args.dataset}' requires additional setup.")
        logger.warning("Using sample data instead.")
        df = create_sample_data()
    
    # Step 2: Train a simple model
    model = train_simple_model(df)
    
    logger.info("=" * 60)
    logger.info("  Quick Start Demo Complete!")
    logger.info("  For the full version, please refer to the documentation.")
    logger.info("=" * 60)
    
    logger.info("\nNext steps:")
    logger.info("1. Check docs/ directory for detailed documentation")
    logger.info("2. Explore notebooks/ for interactive examples (if available)")
    logger.info("3. See README.md for project overview")


if __name__ == "__main__":
    main()
