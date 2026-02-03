#!/bin/bash
# run_all_experiments.sh
# Complete reproduction pipeline for Safety-Critical Battery Prognostics

set -e  # Exit on error

echo "========================================================"
echo "🚀 Starting Full Reproduction Pipeline"
echo "========================================================"

# 1. Run Quantitative Evaluation (Rigorous Leave-One-Out CV)
echo ""
echo "📊 [1/2] Running Rigorous Evaluation (Hierarchical Bayes vs LSTM)..."
# This script runs both models and saves metrics to results/rigor/
python -m src.evaluate_rigor

# 2. Visualizations are generated automatically by evaluate_rigor
echo ""
echo "🎨 [2/2] Verifying Visualizations..."
if [ -f "results/rigor/comparison_B0018_generated.png" ]; then
    echo "   - Found: results/rigor/comparison_B0018_generated.png"
else
    echo "   - Warning: Plot not found! Check src/evaluate_rigor.py output."
fi

echo ""
echo "========================================================"
echo "✅ Reproduction Complete!"
echo "   - Metrics: results/rigor/metrics_final.csv"
echo "   - Figures: results/rigor/"
echo "========================================================"
