# 使用示例（Usage Examples）

本页面提供各种使用场景的示例代码。

---

## 目录

- [基础使用](#基础使用)
- [自定义数据](#自定义数据)
- [模型训练](#模型训练)
- [推理与预测](#推理与预测)
- [不确定性估计](#不确定性估计)
- [FMEA诊断](#fmea诊断)
- [ONNX部署](#onnx部署)
- [高级用法](#高级用法)

---

## 基础使用

### 快速开始

```python
from src.data_loader import BatteryDataLoader
from src.models.chronos_model import ChronosPINNModel

# 加载数据
loader = BatteryDataLoader(dataset="NASA", battery_id="B0005")
train_data, test_data = loader.load()

# 初始化模型
model = ChronosPINNModel(config="configs/default.yaml")

# 训练
model.train(train_data)

# 预测
predictions = model.predict(test_data)
print(predictions)
```

### 使用命令行

```bash
# 使用默认配置训练
python main.py --config configs/default.yaml

# 指定数据集和电池
python main.py --dataset CALCE --battery CS2_33

# 仅推理模式
python main.py --mode inference --checkpoint checkpoints/best_model.pt
```

---

## 自定义数据

### 准备自己的数据

```python
import pandas as pd
import torch
from src.data.validator import DataValidator

# 1. 准备数据（DataFrame格式）
data = pd.DataFrame({
    'cycle': range(1, 101),
    'capacity': [2.0 - 0.005 * i for i in range(100)],
    'voltage': [4.2 - 0.002 * i for i in range(100)],
    'current': [-1.0 for _ in range(100)],
    'temperature': [25.0 for _ in range(100)]
})

# 2. 验证数据格式
validator = DataValidator()
is_valid, errors = validator.validate(data)

if not is_valid:
    print("数据验证失败:", errors)
else:
    print("数据验证通过！")

# 3. 保存为项目支持的格式
data.to_csv("data/raw/my_battery.csv", index=False)
```

### 转换已有数据格式

```python
from src.data.calce_micro_parser import CALCEMicroParser

# 解析CALCE原始数据
parser = CALCEMicroParser()
parsed_data = parser.parse("data/raw/calce/CS2_33.xlsx")

# 保存为.pt格式
parser.save(parsed_data, "data/processed/my_battery.pt")
```

---

## 模型训练

### 完整训练流程

```python
import yaml
from src.training.pipeline import TrainingPipeline
from src.utils.config import Config

# 1. 加载配置
with open("configs/default.yaml", "r") as f:
    config_dict = yaml.safe_load(f)
config = Config(config_dict)

# 2. 创建训练流水线
pipeline = TrainingPipeline(config)

# 3. 准备数据
train_loader, val_loader, test_loader = pipeline.prepare_data()

# 4. 初始化模型
model = pipeline.setup_model()

# 5. 训练
results = pipeline.train(
    model,
    train_loader,
    val_loader,
    num_epochs=50,
    early_stopping_patience=10
)

# 6. 评估
test_metrics = pipeline.evaluate(model, test_loader)
print("测试集指标:", test_metrics)

# 7. 保存模型
pipeline.save_model(model, "checkpoints/my_model.pt")
```

### 自定义训练循环

```python
import torch
import torch.nn as nn
from torch.optim import Adam

# 假设已经有了model和dataloader
model = MyModel()
dataloader = MyDataLoader()

# 损失函数
criterion = nn.MSELoss()
physics_loss = PhysicsLoss()

# 优化器
optimizer = Adam(model.parameters(), lr=1e-3)

# 训练循环
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        x, y = batch

        # 前向传播
        y_pred = model(x)

        # 计算损失
        data_loss = criterion(y_pred, y)
        phys_loss = physics_loss(model, x)
        loss = data_loss + 0.1 * phys_loss

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # 打印进度
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
```

---

## 推理与预测

### 批量预测

```python
import torch
from src.models.chronos_model import ChronosPINNModel

# 加载模型
model = ChronosPINNModel.load_from_checkpoint("checkpoints/best_model.pt")
model.eval()

# 准备数据
test_data = torch.load("data/processed/test_data.pt")

# 批量预测
with torch.no_grad():
    predictions = model.predict_batch(test_data)

# 保存结果
torch.save(predictions, "results/predictions.pt")
print("预测完成！")
```

### 单样本预测

```python
import pandas as pd
import torch

# 加载模型
model = torch.load("checkpoints/best_model.pt")
model.eval()

# 准备单样本数据
single_sample = pd.read_csv("data/my_single_sample.csv")

# 预处理
from src.features.extractor import FeatureExtractor
extractor = FeatureExtractor()
features = extractor.extract(single_sample)

# 预测
with torch.no_grad():
    rul_prediction = model.predict_single(features)
    print(f"预测RUL: {rul_prediction:.2f} cycles")
```

### 可视化预测结果

```python
import matplotlib.pyplot as plt
from src.ui.visualization import plot_predictions

# 绘制预测对比图
fig = plot_predictions(
    true_values=true_rul,
    predicted_values=pred_rul,
    uncertainty_bounds=(lower_bound, upper_bound),
    title="RUL Prediction Results"
)

# 保存图片
fig.savefig("figures/prediction_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
```

---

## 不确定性估计

### 共形预测

```python
from src.uncertainty.conformal import ConformalPredictor

# 初始化共形预测器
conformal = ConformalPredictor(
    model=model,
    calibration_data=calib_loader,
    confidence_level=0.95
)

# 校准
conformal.calibrate()

# 预测（带不确定性）
test_point = next(iter(test_loader))
prediction, lower, upper = conformal.predict_with_uncertainty(test_point)

print(f"预测值: {prediction:.2f}")
print(f"95%置信区间: [{lower:.2f}, {upper:.2f}]")
```

### 可靠性分析

```python
from src.uncertainty.scoring import calculate_reliability

# 计算可靠性
reliability = calculate_reliability(
    predictions=all_predictions,
    lower_bounds=all_lower,
    upper_bounds=all_upper,
    true_values=all_true
)

print(f"实测覆盖率: {reliability['coverage']:.2%}")
print(f"预期覆盖率: {reliability['expected_coverage']:.2%}")
print(f"区间平均宽度: {reliability['mean_width']:.2f}")

# 绘制可靠性图
from src.ui.visualization import plot_reliability_diagram
fig = plot_reliability_diagram(reliability)
fig.savefig("figures/reliability_diagram.png")
```

---

## FMEA诊断

### 基础诊断

```python
from src.safety.fmea.analyzer import FMEAnalyzer
from src.safety.fmea.llm_agent import FMEAAgent

# 初始化分析器
analyzer = FMEAnalyzer()

# 分析模型输出
diagnostics = analyzer.analyze(
    predictions=predictions,
    physical_features=physical_features,
    thresholds={
        "concentration_gradient": 1000.0,
        "mechanical_stress": 200.0
    }
)

# 打印诊断结果
print("诊断结果:")
for issue in diagnostics.issues:
    print(f"- {issue.type}: {issue.severity}")
    print(f"  描述: {issue.description}")
```

### LLM代理诊断

```python
from src.safety.fmea.llm_agent import FMEAAgent

# 初始化LLM代理
agent = FMEAAgent(
    api_key="your-api-key-here",
    model="gpt-4"  # 或 deepseek-chat
)

# 生成FMEA报告
report = agent.generate_report(
    diagnostics=diagnostics,
    battery_info={
        "type": "Li-ion 18650",
        "cycles": 150,
        "temperature": 35.0
    }
)

# 打印报告
print(report)

# 保存JSON格式报告
report.save_json("results/fmea_report.json")
```

---

## ONNX部署

### 导出ONNX模型

```python
from src.deployment.onnx_export import ONNXExporter

# 加载PyTorch模型
model = torch.load("checkpoints/best_model.pt")

# 初始化导出器
exporter = ONNXExporter(model)

# 导出为FP32 ONNX
exporter.export(
    output_path="deployment/onnx/model_fp32.onnx",
    input_sample=torch.randn(1, 100),  # 示例输入
    opset_version=14
)

# 导出为INT8量化ONNX
exporter.export_quantized(
    output_path="deployment/onnx/model_int8.onnx",
    calibration_data=calib_loader
)
```

### ONNX推理

```python
import onnxruntime as ort

# 加载ONNX模型
session = ort.InferenceSession("deployment/onnx/model_int8.onnx")

# 准备输入
import numpy as np
input_data = np.random.randn(1, 100).astype(np.float32)

# 推理
outputs = session.run(
    None,
    {"input": input_data}
)

# 获取结果
rul_prediction = outputs[0][0]
print(f"预测RUL: {rul_prediction:.2f}")
```

### 性能分析

```python
from scripts.onnx_edge_profiler import ONNXProfiler

# 初始化性能分析器
profiler = ONNXProfiler("deployment/onnx/model_int8.onnx")

# 基准测试
benchmark_results = profiler.benchmark(
    num_runs=1000,
    warmup_runs=100
)

print(f"平均延迟: {benchmark_results['mean_latency']:.3f} ms")
print(f"P99延迟: {benchmark_results['p99_latency']:.3f} ms")
print(f"吞吐量: {benchmark_results['throughput']:.1f} samples/s")
```

---

## 高级用法

### 自定义物理模型

```python
from src.physics.electrochemistry.spm import SingleParticleModel

# 继承并自定义SPM
class MySPM(SingleParticleModel):
    def __init__(self, custom_params):
        super().__init__()
        self.custom_params = custom_params

    def calculate_degradation(self, state):
        # 自定义退化模型
        degradation = super().calculate_degradation(state)
        # 添加自定义逻辑
        degradation += self.custom_params["my_factor"] * state["temperature"]
        return degradation

# 使用自定义模型
my_spm = MySPM({"my_factor": 0.01})
```

### 特征工程

```python
from src.features.extractor import FeatureExtractor

# 自定义特征提取器
class MyFeatureExtractor(FeatureExtractor):
    def __init__(self):
        super().__init__()
        self.add_feature("my_custom_feature", self._compute_my_feature)

    def _compute_my_feature(self, data):
        # 计算自定义特征
        return data["voltage"].diff().abs().mean()

# 使用
extractor = MyFeatureExtractor()
features = extractor.extract(battery_data)
print(features.columns)
```

### 集成学习

```python
from src.models.ensemble_model import EnsembleModel

# 创建集成模型
ensemble = EnsembleModel([
    ("lstm", LSTMModel()),
    ("tcn", TCNModel()),
    ("transformer", TransformerModel()),
    ("hybrid", ChronosPINNModel())
])

# 训练集成
ensemble.train(train_data)

# 预测（加权平均）
predictions = ensemble.predict(test_data, weights=[0.1, 0.1, 0.2, 0.6])
```

---

## 更多示例

查看`notebooks/`目录获取更多交互式示例：
- `01_quick_start.ipynb` - 快速开始教程
- `quickstart.ipynb` - 完整入门指南

---

## 遇到问题？

请查看[FAQ.md](FAQ.md)或提交Issue！
