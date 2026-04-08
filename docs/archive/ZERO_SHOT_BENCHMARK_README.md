# 零样本跨数据集评测流水线 (Zero-Shot Cross-Dataset Benchmark)

> [!WARNING]
> Historical archive. This page is preserved for provenance and may include
> deprecated `target="rul"` PINN examples or older wording about model
> semantics. Use the active repository docs for current behavior.

## 🎯 项目概述

本项目实现了一个业界标杆级别的**零样本泛化评测流水线**，用于评估电池 RUL (Remaining Useful Life) 预测模型的跨数据集泛化能力。

**核心价值**：在 Dataset A (如 NASA) 上训练模型，直接在 Dataset B (如 CALCE/Oxford) 上测试，无需任何微调，真实反映模型的零样本泛化能力。

---

## 📦 核心组件

### 1. 核心类: `ZeroShotBenchmarkRunner`

**文件**: `src/evaluation/zero_shot_benchmark.py` (~1000 行)

**主要功能**:
- ✅ 零样本泛化测试 (Dataset A → Dataset B, 无微调)
- ✅ 支持多数据集: NASA PCoE、CALCE CS2、Oxford、MIT
- ✅ 自动特征推断
- ✅ 全面评测指标
- ✅ 自动化 Markdown 报告生成
- ✅ 丰富的可视化图表

**核心方法**:
```python
# 单组零样本评测
result = benchmark.run_zero_shot(
    model=pinn_model,
    model_name="PINN",
    train_dataset="nasa",
    test_dataset="calce",
    features=[...],
    target="rul"
)

# 完整跨数据集矩阵评测
benchmark.run_cross_dataset_matrix(
    model_class=PINNModel,
    model_name="PINN",
    datasets=["nasa", "calce", "oxford"]
)

# 生成 Markdown 报告
benchmark.generate_markdown_report(title="评测报告")

# 生成可视化图表
benchmark.generate_comparison_plots()
```

---

### 2. 命令行工具

**文件**: `scripts/run_zero_shot_benchmark.py` (~600 行)

**使用方法**:

```bash
# 单组跨数据集评测 (NASA → CALCE)
python scripts/run_zero_shot_benchmark.py \
    --model pinn \
    --train nasa \
    --test calce

# 完整跨数据集矩阵评测
python scripts/run_zero_shot_benchmark.py \
    --model pinn \
    --run-full-matrix \
    --datasets nasa calce oxford

# 使用 LSTM 基线模型
python scripts/run_zero_shot_benchmark.py \
    --model lstm \
    --train nasa \
    --test calce

# 查看帮助
python scripts/run_zero_shot_benchmark.py --help
```

**支持参数**:
- `--model`: 模型类型 (pinn, lstm, gru)
- `--train`: 训练数据集名称
- `--test`: 测试数据集名称
- `--run-full-matrix`: 运行完整跨数据集矩阵
- `--datasets`: 数据集列表
- `--results-dir`: 结果保存目录
- `-v, --verbose`: 详细日志输出

---

### 3. 使用示例

**文件**: `examples/zero_shot_benchmark_example.py` (~600 行)

**包含示例**:
1. **基础零样本评测** - NASA → CALCE 单组评测
2. **完整跨数据集矩阵** - 所有数据集组合评测
3. **自定义模型集成** - 如何将自定义 PINN 模型集成到评测流水线

---

## 📊 评测指标说明

### 1. 点预测精度指标

| 指标 | 全称 | 说明 | 目标值 |
|-----|------|------|--------|
| **RMSE** | Root Mean Squared Error | 均方根误差 | ↓ 越低越好 |
| **MAE** | Mean Absolute Error | 平均绝对误差 | ↓ 越低越好 |

### 2. 不确定性量化指标

| 指标 | 全称 | 说明 | 目标值 |
|-----|------|------|--------|
| **PICP** | Prediction Interval Coverage Probability | 95% 置信区间覆盖率 | ≈ 0.95 (理想) |
| **Coverage 80** | 80% Prediction Interval Coverage | 80% 置信区间覆盖率 | ≈ 0.80 (理想) |
| **Coverage 95** | 95% Prediction Interval Coverage | 95% 置信区间覆盖率 | ≈ 0.95 (理想) |

**PICP 解读**:
- **PICP ≈ 0.95**: 不确定性校准良好 (理想状态)
- **PICP < 0.95**: 预测过于自信 (under-confident)
- **PICP > 0.95**: 预测过于保守 (over-confident)

### 3. 概率预测质量指标

| 指标 | 全称 | 说明 | 目标值 |
|-----|------|------|--------|
| **CRPS** | Continuous Ranked Probability Score | 连续排序概率得分 | ↓ 越低越好 |
| **NLL** | Negative Log-Likelihood | 负对数似然 | ↓ 越低越好 |

### 4. 效率与风险指标

| 指标 | 说明 | 目标值 |
|-----|------|--------|
| **Inference Time (ms/sample)** | 推理时间（每样本毫秒） | ↓ 越低越好 |
| **Sharpe Ratio** | 风险调整后的预测质量 | ↑ 越高越好 |

---

## 📈 输出文件结构

运行评测后会生成以下文件：

```
results/zero_shot_benchmark/
├── zero_shot_benchmark_report.md        # 📄 Markdown 评测报告
│   ├── 执行摘要
│   ├── 跨数据集性能矩阵
│   ├── 统计分析 (t 检验)
│   ├── 详细结果表
│   └── 方法论说明
├── zero_shot_results.json              # 📊 JSON 格式结果 (可导入其他工具)
└── figures/
    ├── zero_shot_heatmap_rmse.png      # 🔥 RMSE 热力图
    ├── zero_shot_heatmap_mae.png       # 🔥 MAE 热力图
    ├── zero_shot_heatmap_picp.png    # 🔥 PICP 热力图 (关键指标！)
    ├── zero_shot_comparison.png        # 📈 同分布 vs 零样本对比
    └── zero_shot_boxplot.png           # 📦 指标分布箱线图
```

---

## 🚀 快速开始

### 方式 1: 运行演示脚本

```bash
# 运行演示 (仅展示功能，不实际运行模型)
python demo_zero_shot.py
```

### 方式 2: 单组跨数据集评测

```bash
# NASA → CALCE 零样本评测
python scripts/run_zero_shot_benchmark.py \
    --model pinn \
    --train nasa \
    --test calce \
    --results-dir results/nasa_to_calce
```

### 方式 3: 完整矩阵评测

```bash
# 评测所有数据集组合
python scripts/run_zero_shot_benchmark.py \
    --model pinn \
    --run-full-matrix \
    --datasets nasa calce \
    --results-dir results/full_matrix
```

### 方式 4: Python API

```python
from src.evaluation import ZeroShotBenchmarkRunner
from src.models.pinn_model import PINNModel

# 创建评测器
benchmark = ZeroShotBenchmarkRunner(
    results_dir="results/zero_shot",
    random_seed=42,
)

# 创建模型
model = PINNModel(input_dim=8, hidden_dims=[128, 64, 32])

# 运行零样本评测 (NASA → CALCE，无微调！)
result = benchmark.run_zero_shot(
    model=model,
    model_name="PINN_NASA_to_CALCE",
    train_dataset="nasa",
    test_dataset="calce",  # 零样本测试！
    features=["capacity", "discharge_time", "max_temp", "mean_temp"],
    target="rul",
)

# 查看结果
print(f"RMSE: {result.rmse:.4f}")
print(f"MAE: {result.mae:.4f}")
print(f"PICP: {result.picp:.4f}")  # 目标: ~0.95

# 生成完整报告
benchmark.generate_markdown_report()
benchmark.generate_comparison_plots()
```

---

## 📝 关键概念说明

### 什么是零样本泛化 (Zero-Shot Generalization)?

**定义**: 模型在训练数据集 A 上训练完成后，直接在未见过的数据集 B 上进行测试，**不进行任何微调或再训练**。

**为什么重要?**
1. **真实场景**: 实际应用中，目标域数据往往无法提前获取
2. **模型鲁棒性**: 测试模型对分布偏移的适应能力
3. **成本效益**: 避免为每个新场景收集标注数据

**PINN 的优势**:
- 物理约束提供了额外的正则化
- 对分布偏移具有更好的鲁棒性
- 在零样本场景下表现优于纯数据驱动模型

---

## 🔧 故障排除

### 常见问题

#### 1. 数据加载失败

**问题**: `FileNotFoundError: NASA data directory not found`

**解决方案**:
```bash
# 下载 NASA 数据集
python scripts/download_data.py --dataset nasa

# 或使用自定义路径
python scripts/run_zero_shot_benchmark.py \
    --train nasa --test calce \
    --nasa-dir /path/to/nasa/data \
    --calce-dir /path/to/calce/data
```

#### 2. CUDA 内存不足

**问题**: `RuntimeError: CUDA out of memory`

**解决方案**:
```python
# 在代码中添加
import torch
torch.cuda.empty_cache()

# 或使用 CPU
benchmark = ZeroShotBenchmarkRunner(device="cpu")
```

#### 3. 模型加载失败

**问题**: `ModuleNotFoundError: No module named 'src.models.pinn_model'`

**解决方案**:
```bash
# 确保在项目根目录运行
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 或在代码中添加
import sys
sys.path.insert(0, str(Path(__file__).parent))
```

---

## 📚 参考文献

1. **Zero-Shot Learning**: 
   - Palatucci, M., et al. (2009). "Zero-shot learning with semantic output codes." NIPS.

2. **Domain Adaptation**:
   - Ganin, Y., et al. (2016). "Domain-adversarial training of neural networks." JMLR.

3. **Physics-Informed Neural Networks**:
   - Raissi, M., et al. (2019). "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." JCP.

4. **Battery RUL Prediction**:
   - Che, Y., et al. (2022). "Deep learning for battery remaining useful life prediction: A comprehensive review." Renewable and Sustainable Energy Reviews.

---

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤:

1. **Fork 仓库**
2. **创建特性分支**: `git checkout -b feature/your-feature`
3. **提交更改**: `git commit -am 'Add some feature'`
4. **推送分支**: `git push origin feature/your-feature`
5. **创建 Pull Request**

### 代码规范

- 遵循 PEP 8 风格指南
- 添加类型注解
- 编写 docstring 文档
- 添加单元测试

---

## 📄 许可证

本项目采用 MIT 许可证 - 详情请参阅 [LICENSE](../LICENSE) 文件。

---

## 🙏 致谢

感谢以下开源项目和研究为本项目提供的灵感和基础:

- [PyTorch](https://pytorch.org/) - 深度学习框架
- [Pandas](https://pandas.pydata.org/) - 数据处理
- [Matplotlib](https://matplotlib.org/) / [Seaborn](https://seaborn.pydata.org/) - 可视化
- [SciPy](https://scipy.org/) - 科学计算

---

## 📞 联系方式

如有问题或建议，请通过以下方式联系:

- **GitHub Issues**: [提交问题](https://github.com/your-repo/issues)
- **Email**: your-email@example.com

---

**最后更新**: 2025年

**版本**: 1.0.0

---

*本项目致力于推动电池健康管理的可信赖 AI 技术发展* 🔋🤖
