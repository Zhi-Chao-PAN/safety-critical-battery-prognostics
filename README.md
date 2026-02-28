# Uncertainty-Aware Edge-Deployable Battery Health Management System

> Safety-critical battery prognostics with principled uncertainty quantification, physics-informed learning, and edge deployment capability.

## Highlights

- **7 Model Architectures**: LSTM, GRU, TCN, Transformer, PINN, Bayesian NN, Deep Ensemble
- **Physics-Informed**: Empirical degradation model (SEI + lithium plating) as PINN prior
- **Uncertainty Quantification**: MC Dropout, Variational Inference, Deep Ensembles with CRPS/NLL/PICP scoring
- **Uncertainty Decomposition**: Aleatoric vs epistemic separation
- **Safety Decision Engine**: Three-tier (GREEN/YELLOW/RED) adaptive safety classification
- **Edge Deployment**: ONNX export targeting Raspberry Pi 4B (<50ms inference)
- **15+ Engineered Features**: IC/DV curves, frequency domain, rolling statistics, trend analysis
- **Rigorous Evaluation**: LOGO-CV, nested CV, cross-dataset OOD, few-shot transfer splits

## Project Structure

```
├── main.py                     # Full pipeline entry point
├── pyproject.toml              # Dependencies & build config
├── configs/
│   └── default.yaml            # Experiment configuration
├── src/
│   ├── data/
│   │   ├── unified_loader.py   # Multi-dataset loader (NASA, CALCE, Oxford)
│   │   ├── validator.py        # Physical bounds validation
│   │   └── splitter.py         # LOGO-CV, nested CV, OOD, few-shot splits
│   ├── features/
│   │   └── extractor.py        # 15+ feature extraction pipeline
│   ├── models/
│   │   ├── base.py             # BatteryModel ABC (unified interface)
│   │   ├── lstm_model.py       # Bidirectional LSTM + Attention
│   │   ├── gru_model.py        # GRU + Attention
│   │   ├── tcn_model.py        # Temporal Convolutional Network
│   │   ├── transformer_model.py # Transformer Encoder + CLS token
│   │   ├── pinn_model.py       # Physics-Informed Neural Network
│   │   ├── bayesian_nn.py      # Bayesian NN (Variational Inference)
│   │   └── ensemble_model.py   # Deep Ensemble wrapper
│   ├── physics/
│   │   └── degradation.py      # Empirical fade + Arrhenius model
│   ├── uncertainty/
│   │   ├── scoring.py          # CRPS, NLL, PICP, MPIW, Interval Score
│   │   ├── calibration.py      # Reliability diagrams + isotonic recalibration
│   │   └── decomposition.py    # Aleatoric/epistemic decomposition
│   ├── safety/
│   │   └── decision_engine.py  # Three-tier safety classification
│   ├── training/
│   │   └── pipeline.py         # Training with checkpointing & logging
│   ├── evaluation/
│   │   └── benchmark.py        # Multi-seed benchmark runner
│   ├── deployment/
│   │   └── onnx_export.py      # ONNX export + edge predictor
│   └── ui/
│       └── visualization.py    # Publication-quality figures
├── tests/
│   ├── test_new_modules.py     # Core module tests
│   └── test_advanced_modules.py # PINN, pipeline, benchmark tests
└── data/
    └── battery_data/           # NASA PCoE .mat files
```

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run full pipeline
python main.py --device cuda --epochs 100

# Run specific models
python main.py --models lstm pinn transformer --seeds 42 43 44

# Run tests
pytest tests/ -v
```

## Models

| Model | Type | Uncertainty Method | Parameters |
|-------|------|-------------------|------------|
| LSTM + Attention | Sequence | MC Dropout | ~50K |
| GRU + Attention | Sequence | MC Dropout | ~40K |
| TCN | Convolutional | MC Dropout | ~30K |
| Transformer | Attention | MC Dropout | ~45K |
| PINN | Hybrid | MC Dropout + Physics | ~35K |
| Bayesian NN | Probabilistic | Variational Inference | ~35K |
| Deep Ensemble | Meta | Ensemble Variance | N × base |

## Evaluation Metrics

**Deterministic**: RMSE, MAE, MAPE, R²
**Probabilistic**: CRPS, NLL, PICP, MPIW, Interval Score, ENCE

## Key Innovation Points

1. **Uncertainty-Aware Edge Deployment**: Full UQ pipeline compressed to ONNX for RPi 4B
2. **Physics-Informed Bayesian Prior**: PINN learns residuals over empirical degradation model
3. **Adaptive Safety Decisions**: Uncertainty-calibrated three-tier safety classification
4. **Cross-Chemistry Transfer**: Few-shot adaptation across battery chemistries
5. **Comprehensive Comparison**: 7 architectures × 3 UQ methods × 4 split strategies

## Author

Zhichao Pan — Yangzhou University, Guangling College
