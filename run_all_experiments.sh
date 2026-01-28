#!/bin/bash
# run_all_experiments.sh
# Complete reproduction pipeline for Spatial Bayesian vs Deep Learning

set -e  # Exit on error

echo "========================================================"
echo "🚀 Starting Full Reproduction Pipeline"
echo "========================================================"

# 1. Run Quantitative Evaluation (Rigorous 5-Fold CV)
echo ""
echo "📊 [1/3] Running Rigorous Evaluation (may take time)..."
python src/evaluate_rigor.py

# 2. Run Deep Learning Baselines
echo ""
echo "🧠 [2/3] Training Neural Network Baselines..."
python src/train_nn_baseline.py
# Optional: Spatial Embedding NN
# python src/train_nn_spatial.py

# 3. Generate Visualizations (Figures)
echo ""
echo "🎨 [3/3] Generating Publication-Quality Figures..."
python src/visualize_results.py

echo ""
echo "========================================================"
echo "✅ Reproduction Complete!"
echo "   - Results saved to: results/rigor/ and results/figures/"
echo "========================================================"
