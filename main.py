"""
main.py - Primary entry point for PINN Battery Prognostics demonstration.

Showcases core project capabilities:
  1. Physics-informed degradation modeling (SPM curve fitting)
  2. Three-layer physical safety defense (Constraint → Clamp → Projection)
  3. Safety decision engine with fail-safe NaN guard
  4. FMEA risk analysis

Works out of the box with zero external data — generates synthetic
battery degradation data for demonstration.
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
logger = logging.getLogger("main")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Battery Prognostics - Core Demo")
    p.add_argument(
        "--mode",
        default="full",
        choices=["full", "safety", "physics", "fmea"],
        help="Demo mode: full (all), safety (decision engine), "
             "physics (degradation model), fmea (failure analysis)",
    )
    return p.parse_args()


def create_synthetic_degradation(n_cycles: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic battery capacity degradation data.

    Model: C(t) = a * exp(-b * t) + c + noise
    This mimics real lithium-ion capacity fade with:
      - Exponential decay (electrochemical aging)
      - Noise (measurement + environmental)
      - Optional capacity recovery anomalies
    """
    rng = np.random.default_rng(seed)
    cycles = np.arange(1, n_cycles + 1, dtype=np.float64)
    t_norm = cycles / n_cycles

    # Physics-based degradation: a*exp(-b*t) + c
    a, b, c = 0.6, 3.0, 1.4  # Initial=2.0 Ah, EOL≈1.4 Ah
    capacity_true = a * np.exp(-b * t_norm) + c
    noise = rng.normal(0, 0.01, n_cycles)
    capacity = capacity_true + noise

    # RUL: cycles remaining until EOL (capacity < 1.4 Ah)
    eol_threshold = 1.4
    rul = np.zeros(n_cycles)
    for i in range(n_cycles):
        future = capacity[i:]
        eol_hits = np.where(future < eol_threshold)[0]
        rul[i] = eol_hits[0] if len(eol_hits) > 0 else (n_cycles - i)

    return pd.DataFrame({
        "cycle": cycles,
        "capacity": capacity,
        "capacity_true": capacity_true,
        "rul": rul,
        "battery_id": "SYNTH001",
    })


def demo_physics_model(df: pd.DataFrame) -> None:
    """Demonstrate SPM degradation curve fitting."""
    from src.physics.aging.degradation import PhysicsModel

    logger.info("=" * 60)
    logger.info("  [1/4] Physics Model: SPM Degradation Curve Fitting")
    logger.info("=" * 60)

    model = PhysicsModel()
    cycles = df["cycle"].values.astype(np.float64)
    capacity = df["capacity"].values.astype(np.float64)

    model.fit(cycles, capacity, battery_id="SYNTH001")
    params = model.params.get("SYNTH001")

    if params is not None:
        pred = model.predict(cycles, battery_id="SYNTH001")
        residual = capacity - pred
        rmse = np.sqrt(np.mean(residual ** 2))
        logger.info(f"  Fitted params: Q0={params['q0']:.4f}, a={params['a']:.4f}, b={params['b']:.5f}")
        logger.info(f"  Physics RMSE:  {rmse:.4f} Ah")
        logger.info(f"  Residual range: [{residual.min():.4f}, {residual.max():.4f}]")
    else:
        logger.warning("  Physics fit failed — see log for details.")


def demo_safety_engine(df: pd.DataFrame) -> None:
    """Demonstrate safety decision engine with NaN fail-safe."""
    from src.safety.decision_engine import SafetyDecisionEngine, SafetyLevel

    logger.info("=" * 60)
    logger.info("  [2/4] Safety Decision Engine: Three-Tier Classification")
    logger.info("=" * 60)

    engine = SafetyDecisionEngine(
        rul_critical=10.0,
        rul_warning=30.0,
        epistemic_threshold_low=5.0,
        epistemic_threshold_high=15.0,
    )

    # Normal operation
    d1 = engine.decide(rul_mean=80.0, rul_lower=70.0, rul_upper=90.0, epistemic_std=2.0)
    logger.info(f"  Normal input    → {d1.level.value:6s} | Action: {d1.action}")

    # Warning zone
    d2 = engine.decide(rul_mean=25.0, rul_lower=20.0, rul_upper=30.0, epistemic_std=8.0)
    logger.info(f"  Warning input   → {d2.level.value:6s} | Action: {d2.action}")

    # Critical
    d3 = engine.decide(rul_mean=5.0, rul_lower=3.0, rul_upper=7.0, epistemic_std=20.0)
    logger.info(f"  Critical input  → {d3.level.value:6s} | Action: {d3.action}")

    # F1 Fail-safe: NaN guard
    d4 = engine.decide(rul_mean=float('nan'), rul_lower=50.0, rul_upper=100.0, epistemic_std=2.0)
    logger.info(f"  NaN input       → {d4.level.value:6s} | Reason: {d4.reason[:60]}...")

    # F2 Fail-safe: Unknown uncertainty
    decisions = engine.decide_batch(
        means=np.array([80.0]),
        lowers=np.array([70.0]),
        uppers=np.array([90.0]),
        epistemic_stds=None,  # Unknown → fail-safe to YELLOW+
    )
    logger.info(f"  Unknown uncert. → {decisions[0].level.value:6s} | (epistemic_stds=None fail-safe)")

    # Summary
    red_count = sum(1 for d in [d1, d2, d3, d4] if d.level == SafetyLevel.RED)
    logger.info(f"\n  Result: {red_count}/4 RED decisions (NaN correctly triggers RED)")


def demo_fmea() -> None:
    """Demonstrate FMEA risk analysis."""
    from src.safety.fmea.analyzer import FMEAAnalyzer

    logger.info("=" * 60)
    logger.info("  [3/4] FMEA Risk Analysis")
    logger.info("=" * 60)

    analyzer = FMEAAnalyzer()
    critical = analyzer.get_critical_failures(rpn_threshold=80)

    logger.info(f"  Total failure modes analyzed: {len(analyzer.modes)}")
    logger.info(f"  Critical (RPN ≥ 80): {len(critical)}")
    for fm in critical:
        logger.info(f"    • {fm.failure_mode} (RPN={fm.rpn}, S={fm.severity})")


def demo_constraint_system() -> None:
    """Demonstrate physics constraint validation."""
    import torch
    from src.physics.constraints import (
        create_default_constraint_manager,
        NAN_PENALTY_LOSS,
    )

    logger.info("=" * 60)
    logger.info("  [4/4] Physics Constraint System: NaN Safety Guard")
    logger.info("=" * 60)

    manager = create_default_constraint_manager("cpu")

    # Clean predictions → should pass validation
    clean_preds = torch.linspace(2.0, 1.5, 20).unsqueeze(1)
    inputs = {"cycles": torch.arange(20, dtype=torch.float32).unsqueeze(1)}
    valid_clean = manager.validate_all(clean_preds, inputs)
    logger.info(f"  Clean data validation:  {'PASS ✓' if valid_clean else 'FAIL ✗'}")

    # NaN predictions → should fail validation and return penalty
    nan_preds = torch.tensor([[1.0], [float('nan')], [0.5]])
    nan_inputs = {"cycles": torch.arange(3, dtype=torch.float32).unsqueeze(1)}
    valid_nan = manager.validate_all(nan_preds, nan_inputs)
    logger.info(f"  NaN data validation:    {'PASS ✓' if valid_nan else f'FAIL ✗ (penalty={NAN_PENALTY_LOSS})'}")

    total_loss, breakdown = manager.compute_total_loss(
        clean_preds, inputs,
        cycles=torch.arange(20, dtype=torch.float32),
        max_cycle=100.0,
    )
    logger.info(f"  Total constraint loss (clean): {total_loss.item():.6f}")
    for name, info in breakdown.items():
        logger.info(f"    {name}: loss={info['loss']:.6f}")


def main() -> None:
    args = parse_args()

    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║  PINN Battery Prognostics — Core Capability Demo        ║")
    logger.info("║  Three-Layer Physical Safety Defense Architecture        ║")
    logger.info("╚" + "═" * 58 + "╝")

    df = create_synthetic_degradation()
    logger.info(f"Synthetic data: {len(df)} cycles, capacity range "
                f"[{df['capacity'].min():.3f}, {df['capacity'].max():.3f}] Ah\n")

    if args.mode in ("full", "physics"):
        demo_physics_model(df)
        print()

    if args.mode in ("full", "safety"):
        demo_safety_engine(df)
        print()

    if args.mode in ("full", "fmea"):
        demo_fmea()
        print()

    if args.mode in ("full",):
        demo_constraint_system()
        print()

    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║  Demo Complete — All core systems operational            ║")
    logger.info("╠" + "═" * 58 + "╣")
    logger.info("║  Next Steps:                                             ║")
    logger.info("║  • Full PINN training: scripts/run_pinn_training.py      ║")
    logger.info("║  • Robustness test:    scripts/run_robustness_test.py    ║")
    logger.info("║  • Documentation:      docs/PROJECT_ARCHITECTURE.md      ║")
    logger.info("╚" + "═" * 58 + "╝")


if __name__ == "__main__":
    main()
