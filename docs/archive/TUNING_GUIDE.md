# 电池寿命预测项目自动化超参搜索系统使用指南

## 概述

本系统为PINN电池寿命预测项目提供了完整的自动化超参搜索解决方案，集成了Optuna贝叶斯优化和W&B工业级监控，旨在寻找物理约束的最优解，展现MLOps工程素养。

## 系统架构

```
tune.py                    # 基础版超参搜索脚本
tune_enhanced.py          # 增强版（推荐），包含完整回调系统
├── PINNTuningObjective   # Optuna目标函数
├── TrainingCallback      # 训练回调基类
│   ├── WandbCallback    # W&B实时监控
│   ├── ModelCheckpointCallback # 模型保存
│   └── EarlyStoppingCallback   # 早期停止
└── 完整的MLOps流水线
```

## 核心功能

### 1. Optuna贝叶斯搜索
- **学习率**: Log-uniform分布 (1e-4 到 1e-2)
- **隐藏维度**: 类别选择 (32, 64, 128)
- **物理约束权重**: 
  - `lambda_physics`: 物理损失权重 (0.01-0.5)
  - `lambda_mono`: 单调性约束权重 (0.001-0.1)
- **自适应权重参数**: 动态调整约束强度
- **正则化参数**: dropout, weight_decay

### 2. W&B工业级监控
- 每个Trial独立W&B运行
- 实时上报训练曲线:
  - Total Loss, Data Loss, Physics Loss分离
  - 约束分解损失监控
  - 自适应权重可视化
- 超参重要性分析
- 实验对比和版本管理

### 3. 智能剪枝 (Pruning)
- **MedianPruner**: 中位数剪枝器
- **NaN/Inf检测**: 前30个epoch内自动终止
- **发散检测**: 损失超过10倍最佳值时剪枝
- **节约算力**: 在RTX 4060上最大化效率

### 4. 优雅保存
- **最佳模型**: 自动保存`.pth`权重文件
- **配置保存**: JSON格式的完整超参配置
- **时间戳**: 避免文件覆盖
- **检查点**: 训练过程中的中间保存

## 快速开始

### 1. 安装依赖
```bash
# 安装基础依赖
pip install -r requirements.txt

# 安装超参搜索依赖
pip install -r requirements_optuna.txt

# 或一次性安装所有依赖
pip install -r requirements.txt -r requirements_optuna.txt
```

### 2. 配置W&B（可选但推荐）
```bash
# 登录W&B
wandb login

# 或设置环境变量
export WANDB_API_KEY=your_api_key
```

### 3. 运行基础版搜索
```bash
# 基础配置，10个试验
python tune.py --n_trials 10

# 完整配置示例
python tune.py \
  --config configs/pinn_config.yaml \
  --n_trials 50 \
  --wandb_project "battery-pinn-optuna" \
  --save_dir "optuna_results" \
  --device cuda
```

### 4. 运行增强版搜索（推荐）
```bash
# 增强版，包含完整回调系统
python tune_enhanced.py --n_trials 30

# 生产环境配置
python tune_enhanced.py \
  --config configs/pinn_config.yaml \
  --n_trials 100 \
  --timeout 86400 \  # 24小时超时
  --wandb_project "battery-pinn-production" \
  --wandb_entity "your_team" \
  --save_dir "production_results" \
  --device cuda
```

## 命令行参数

### 通用参数
- `--config`: 配置文件路径 (默认: `configs/pinn_config.yaml`)
- `--n_trials`: 试验数量 (默认: 50/30)
- `--timeout`: 超时时间（秒）(默认: 无限制)
- `--device`: 训练设备 (`cuda`/`cpu`) (默认: `cuda`)

### W&B参数
- `--wandb_project`: W&B项目名称
- `--wandb_entity`: W&B实体/团队名称

### 输出参数
- `--save_dir`: 结果保存目录

## 输出文件结构

```
optuna_results/ 或 optuna_results_enhanced/
├── checkpoints/                    # 模型检查点
│   ├── model_trial_0001_epoch10_20250404_143022.pth
│   └── model_trial_0002_epoch25_20250404_143155.pth
├── best_config_20250404_143022.json    # 最佳配置
├── best_trial_info_20250404_143022.json # 最佳试验信息
├── all_trials.json                 # 所有试验结果
├── study_statistics.json           # 研究统计
├── param_importances.html          # 超参重要性图
├── optimization_history.html       # 优化历史图
└── tune.log                       # 运行日志
```

## 搜索空间详解

### 学习率 (learning_rate)
- **分布**: Log-uniform
- **范围**: 1e-4 到 1e-2
- **意义**: 控制参数更新步长，对收敛性至关重要

### 隐藏维度 (hidden_dim)
- **选项**: [32, 64, 128]
- **意义**: 网络容量，影响模型表达能力和过拟合风险

### 物理约束权重
- **lambda_physics**: 物理一致性损失权重
- **lambda_mono**: 单调性约束权重
- **自适应机制**: 根据电池生命周期动态调整

### 正则化参数
- **dropout**: 防止过拟合
- **weight_decay**: L2正则化强度

## 智能剪枝策略

### 1. 启动阶段保护
- `n_startup_trials=5`: 前5个试验不剪枝
- `n_warmup_steps=30`: 前30个epoch不剪枝

### 2. NaN/Inf检测
- 实时监控损失值
- 连续3次NaN/Inf自动终止
- 保存异常状态供调试

### 3. 发散检测
- 损失超过最佳值10倍时警告
- 连续3次发散自动剪枝
- 节约GPU算力

### 4. 早期停止
- 验证损失不再改善时停止
- 可配置的patience参数
- 避免过拟合

## W&B监控指标

### 实时训练曲线
- `total_loss`: 总损失
- `data_loss`: 数据拟合损失
- `constraint_loss`: 物理约束损失
- `constraint_*`: 各约束分解损失

### 自适应权重
- `lambda_physics_mean`: 平均物理权重
- `lambda_mono_mean`: 平均单调性权重

### 训练状态
- `best_loss`: 历史最佳损失
- `learning_rate`: 当前学习率
- `epoch`: 训练进度

## 最佳实践

### 1. 试验数量建议
- **探索阶段**: 20-30个试验
- **优化阶段**: 50-100个试验
- **生产调优**: 100+个试验

### 2. 设备配置
- **RTX 4060**: 启用混合精度训练
- **多GPU**: 可并行运行多个试验
- **CPU回退**: 自动检测CUDA可用性

### 3. 内存管理
- 批量大小根据VRAM调整
- 检查点保存间隔优化
- 及时清理完成试验

### 4. 监控策略
- 定期检查W&B仪表板
- 设置异常报警
- 保存关键试验的完整日志

## 故障排除

### 常见问题

#### 1. 导入错误
```bash
# 确保所有依赖已安装
pip install --upgrade -r requirements_optuna.txt

# 检查Python路径
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

#### 2. CUDA内存不足
```bash
# 减少批量大小
# 在配置文件中调整
batch_size: 16  # 默认32

# 启用梯度检查点
# 在PINNModel中配置
```

#### 3. W&B连接问题
```bash
# 检查网络连接
# 离线模式运行
wandb offline
python tune.py --wandb_project "local_test"

# 或禁用W&B
# 修改代码中wandb.init()调用
```

#### 4. 收敛问题
- 检查学习率范围
- 调整物理约束权重
- 验证数据预处理

### 调试模式
```python
# 在代码中添加调试输出
import logging
logging.basicConfig(level=logging.DEBUG)

# 或使用命令行
python tune.py --config configs/pinn_config.yaml --n_trials 1
```

## 扩展和定制

### 1. 添加新的搜索参数
```python
# 在_suggest_hyperparameters方法中添加
new_param = trial.suggest_float("new_param", 0.1, 1.0, log=True)
params["new_param"] = new_param
```

### 2. 自定义约束
```python
# 创建自定义约束管理器
from src.physics.constraints import ConstraintManager

custom_manager = ConstraintManager(device)
custom_manager.add_constraint(custom_constraint)
```

### 3. 集成新模型
```python
# 修改目标函数中的模型创建
from src.models.new_model import NewModel

model = NewModel(
    hidden_dim=params["hidden_dim"],
    lr=params["lr"],
    # ... 其他参数
)
```

### 4. 分布式优化
```python
# 使用Optuna的分布式存储
import optuna

study = optuna.create_study(
    storage="sqlite:///optuna.db",
    load_if_exists=True,
    # ... 其他参数
)
```

## 性能优化

### RTX 4060特定优化
1. **混合精度训练**: 自动启用
2. **批量处理**: 最大化VRAM利用率
3. **异步数据加载**: 减少CPU-GPU等待

### 内存优化
1. **梯度累积**: 模拟大批量训练
2. **检查点优化**: 选择性保存
3. **数据流优化**: 减少中间变量

### 计算优化
1. **并行试验**: 同时运行多个试验
2. **提前剪枝**: 快速淘汰不良配置
3. **缓存机制**: 复用中间结果

## 安全考虑

### 1. 数据安全
- 不保存原始敏感数据
- 检查点加密（可选）
- 访问控制

### 2. 计算安全
- 资源使用限制
- 异常处理机制
- 超时保护

### 3. 模型安全
- 完整性验证
- 版本控制
- 回滚机制

## 后续工作

### 短期改进
1. 多目标优化（精度 vs 速度）
2. 迁移学习集成
3. 实时进度可视化

### 长期规划
1. 自动机器学习（AutoML）
2. 神经网络架构搜索（NAS）
3. 生产环境部署流水线

## 联系我们

如有问题或建议，请通过项目Issue或邮件联系。

---

*最后更新: 2024年4月4日*
*版本: 1.0.0*