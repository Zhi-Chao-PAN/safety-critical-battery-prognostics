# 零样本跨数据集评测流水线 - 快速开始指南

> [!WARNING]
> Historical archive. Commands and examples here may reflect older benchmark
> assumptions, including deprecated PINN target semantics. Check the active
> README before reusing them.

## 🚀 5分钟快速开始

### 步骤 1: 验证安装

```bash
# 验证核心类可以正常导入
python -c "from src.evaluation import ZeroShotBenchmarkRunner; print('✅ 安装成功')"
```

### 步骤 2: 运行演示

```bash
# 运行演示脚本（仅展示功能，不实际运行模型）
python demo_zero_shot.py
```

### 步骤 3: 运行实际评测

#### 方式 A: 命令行（推荐）

```bash
# 单组跨数据集评测 (NASA → CALCE)
python scripts/run_zero_shot_benchmark.py \
    --model pinn \
    --train nasa \
    --test calce \
    --results-dir results/nasa_to_calce

# 完整跨数据集矩阵评测
python scripts/run_zero_shot_benchmark.py \
    --model pinn \
    --run-full-matrix \
    --datasets nasa calce \
    --results-dir results/full_matrix
```

#### 方式 B: Python API

```python
from src.evaluation import ZeroShotBenchmarkRunner
from src.models.pinn_model import PINNModel

# 1. 创建评测器
benchmark = ZeroShotBenchmarkRunner(
    results_dir="results/zero_shot",
    random_seed=42,
)

# 2. 创建模型
model = PINNModel(
    input_dim=8,
    hidden_dims=[128, 64, 32],
    physics_weight=0.1
)

# 3. 运行零样本评测 (NASA → CALCE，无微调！)
result = benchmark.run_zero_shot(
    model=model,
    model_name="PINN_NASA_to_CALCE",
    train_dataset="nasa",
    test_dataset="calce",  # 零样本测试！
    features=["capacity", "discharge_time", "max_temp", "mean_temp"],
    target="rul"
)

# 4. 查看结果
print(f"\n评测结果:")
print(f"  RMSE: {result.rmse:.4f}")
print(f"  MAE:  {result.mae:.4f}")
print(f"  PICP: {result.picp:.4f}")  # 目标: ~0.95
print(f"  CRPS: {result.crps:.4f}")

# 5. 生成完整报告和可视化
report_path = benchmark.generate_markdown_report(
    title="零样本评测报告: NASA → CALCE"
)
plot_paths = benchmark.generate_comparison_plots()

print(f"\n报告已保存: {report_path}")
print(f"可视化图表: {len(plot_paths)} 张")
```

---

## 📊 输出文件

运行后会生成以下文件:

```
results/zero_shot_benchmark/
├── zero_shot_benchmark_report.md      # 📄 Markdown 评测报告
├── zero_shot_results.json              # 📊 JSON 格式结果
└── figures/
    ├── zero_shot_heatmap_rmse.png     # 🔥 RMSE 热力图
    ├── zero_shot_heatmap_mae.png      # 🔥 MAE 热力图
    ├── zero_shot_heatmap_picp.png     # 🔥 PICP 热力图
    ├── zero_shot_comparison.png       # 📈 对比图
    └── zero_shot_boxplot.png          # 📦 箱线图
```

---

## 🔧 常见问题

### Q1: 导入失败怎么办?

**A**: 确保在项目根目录运行，并添加项目到 PYTHONPATH:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python your_script.py
```

### Q2: 数据加载失败怎么办?

**A**: 确保数据文件存在于正确的位置:

```bash
# 检查数据目录
ls data/battery_data/  # NASA 数据
ls data/calce/         # CALCE 数据

# 如果缺失，运行数据下载脚本
python scripts/download_data.py --dataset nasa
python scripts/download_data.py --dataset calce
```

### Q3: CUDA 内存不足怎么办?

**A**: 使用 CPU 运行或减小 batch size:

```python
# 使用 CPU
benchmark = ZeroShotBenchmarkRunner(device="cpu")

# 或在代码中添加
import torch
torch.cuda.empty_cache()
```

---

## 📚 更多文档

- **完整文档**: `ZERO_SHOT_BENCHMARK_README.md`
- **交付确认书**: `ZERO_SHOT_BENCHMARK_SUMMARY.md`
- **使用示例**: `examples/zero_shot_benchmark_example.py`
- **演示脚本**: `demo_zero_shot.py`

---

## 🤝 技术支持

如有问题或建议，请联系:

- **GitHub Issues**: [提交问题](https://github.com/your-repo/issues)
- **Email**: your-email@example.com

---

## 🎉 开始使用

现在就开始使用零样本跨数据集评测流水线吧!

```bash
# 快速验证安装
python -c "from src.evaluation import ZeroShotBenchmarkRunner; print('✅ 安装成功')"

# 运行演示
python demo_zero_shot.py

# 运行实际评测
python scripts/run_zero_shot_benchmark.py --model pinn --train nasa --test calce
```

---

🔋🤖🚀 **祝你使用愉快!** 🚀🤖🔋
