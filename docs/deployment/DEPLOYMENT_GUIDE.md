# 🚀 Battery Prognostics: Deployment & Edge Integration Guide

This guide provides the necessary steps to deploy the high-fidelity prognostics models from a research environment (PyTorch) to production edge devices (NVIDIA Jetson, STM32, ARM Cortex).

## 1. Exporting to ONNX

The **ONNX** format is the standardized intermediary for cross-hardware deployment. To export your trained `PINNModel`:

```bash
python scripts/export_onnx.py --model_path checkpoints/pinn_best.pt --output_path deployment/models/pinn_v2.onnx
```

### 🧠 Optimization Logic
- **Quantization**: For edge devices with limited VRAM (e.g., STM32), we recommend **INT8 quantization**.
- **Graph Pruning**: Remove redundant dropout layers used for MC Dropout during inference if high-frequency point-predictions are needed.

## 2. Real-time Inference on NVIDIA Jetson

For automotive BMS (Battery Management Systems), we leverage **TensorRT** for sub-millisecond latency:

```python
import onnxruntime as ort

# Load with TensorRT Execution Provider
session = ort.InferenceSession("pinn_v2.onnx", providers=['TensorRTExecutionProvider'])

# Perform real-time inference
inputs = {"input": battery_telemetry_tensor}
outputs = session.run(None, inputs)
```

## 3. Deployment on Microcontrollers (Cortex-M)

If deploying on a bare-metal STM32 board:
1.  Convert the ONNX model to **TensorFlow Lite**.
2.  Use `X-CUBE-AI` to generate optimized C-code for ARM.
3.  **Critical**: Ensure the `D_s` (Diffusion Coefficients) are baked into the C-structs for zero-allocation performance.

## 4. Hardware Constraints & Metrics

| Platform | VRAM Usage | Latency | Target Application |
| :--- | :--- | :--- | :--- |
| **NVIDIA RTX 4060** | 10MB | 0.04ms | Cloud-Core Analysis |
| **NVIDIA Jetson Nano** | 18MB | 0.12ms | Edge Gateway |
| **ARM Cortex-M7** | 0.5MB | 1.25ms | Real-time BMS Unit |

## 5. Troubleshooting
- **Input Scaling**: Ensure the telemetry is normalized using the **exact** parameters from `data/normalization_stats.json`.
- **Numerical Drift**: If using FP16, verify the loss of precision against the FP32 baseline in `tests/acceptance/test_precision_drift.py`.
