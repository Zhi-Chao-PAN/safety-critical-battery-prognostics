"""
Model Compression & Quantization
Provides Post-Training Quantization (PTQ) to INT8 via ONNX Runtime to 
meet the < 50MB and < 50ms Edge Deployment constraints.
"""

import os

from onnxruntime.quantization import QuantType, quantize_dynamic


class INT8Quantizer:
    """Dynamic INT8 Quantization for PyTorch/ONNX models."""
    def __init__(self, model_input_path: str, model_output_path: str = None):
        self.input_path = model_input_path
        if model_output_path is None:
            base, ext = os.path.splitext(self.input_path)
            self.output_path = f"{base}_int8{ext}"
        else:
            self.output_path = model_output_path

    def quantize(self):
        """Perform dynamic quantization to INT8."""
        quantize_dynamic(
            model_input=self.input_path,
            model_output=self.output_path,
            weight_type=QuantType.QUInt8,
            optimize_model=True
        )

        orig_size = os.path.getsize(self.input_path) / (1024 * 1024)
        new_size = os.path.getsize(self.output_path) / (1024 * 1024)

        return {
            "original_mb": round(orig_size, 2),
            "quantized_mb": round(new_size, 2),
            "compression_ratio": round(orig_size / new_size, 2) if new_size > 0 else 0
        }
