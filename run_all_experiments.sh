#!/bin/bash
# run_all_experiments.sh
# Complete reproduction pipeline for Safety-Critical Battery Prognostics

set -e  # Exit on error

echo "========================================================"
echo "🚀 Starting Full Reproduction Pipeline"
echo "========================================================"

# 1. Run Quantitative Evaluation (Rigorous Leave-One-Out CV)
echo ""
echo "📊 [1/3] Running Rigorous Evaluation (may take time)..."
python -m src.evaluate_rigor

# 2. Run Deep Learning Baselines
echo ""
echo "🧠 [2/3] Training LSTM Neural Network Baseline..."
python src/train_nn_baseline.py

# 3. Generate Visualizations (Figures)
echo ""
echo "🎨 [3/3] Generating Publication-Quality Figures..."
# Note: evaluate_rigor generated results/metrics.csv and results/final_comparison_B0018.png
# We can add explicit plotting script here if needed.
# For now, we rely on the integrated plotting in compare_models/visualization modules.

echo ""
echo "========================================================"
echo "✅ Reproduction Complete!"
echo "   - Results saved to: results/rigor/ and results/"
echo "========================================================"
