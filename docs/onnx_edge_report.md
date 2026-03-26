# ONNX Edge Deployment Profiling Report

**Hardware**: Intel Core Ultra 9-185H (CPU inference)
**Runs**: 1000 sequential inferences

## Model Size Comparison

| Format | Size | Compression |
|--------|------|-------------|
| FP32 ONNX | 0.022 MB | Baseline |
| INT8 ONNX | 0.011 MB | 48.5% smaller |

## Latency Benchmark

| Metric | FP32 | INT8 |
|--------|------|------|
| Mean Latency | 0.078 ms | 0.093 ms |
| P99 Latency | 0.263 ms | 0.394 ms |

## Verdict

> ✅ **PASS**: Mean inference latency **strictly < 50ms**. 
> Fully compatible with real-time BMS edge deployment requirements.