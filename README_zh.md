<div align="center">

# 🔋 电池容量建模研究审计
**有边界的合成基准与留一电芯容量重建实验**

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg?logo=pytorch)](https://pytorch.org/)
[![CI](https://github.com/Zhi-Chao-PAN/safety-critical-battery-prognostics/actions/workflows/ci.yml/badge.svg)](https://github.com/Zhi-Chao-PAN/safety-critical-battery-prognostics/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python)](https://python.org/)
[![GitHub stars](https://img.shields.io/github/stars/Zhi-Chao-PAN/safety-critical-battery-prognostics?style=social)](https://github.com/Zhi-Chao-PAN/safety-critical-battery-prognostics)

*用于容量空间模型、单调后处理与可复核负结果的研究代码*

</div>

---

> 说明：当前实验口径以英文 [README.md](README.md)、[三训练种子复核](experiments/logo_capacity_reconstruction/README.md) 与 [证据矩阵](docs/claim_evidence_matrix.md) 为准。旧报告保留作历史记录，可能含有更强的早期表述。

---

## ⭐ 如果这个项目对你有帮助，请给个Star！

---

## 📚 项目概述

本项目探索容量空间中的物理信息模型和数据驱动模型。当前真实电芯协议把已经观测到的带噪容量轨迹作为输入，因此评估的是容量重建/去噪，不是未来容量或 RUL 外推，也不构成部署安全证明。

最新的 3 个训练种子 × 6 个留一电芯复核得到：PINN 在 18 个 seed-fold 上的 RMSE 为 0.9814 ± 1.6640 Ah，LSTM 为 0.2221 ± 0.0352 Ah。两者的零上升违例都出现在共享 running-min 后处理之后。这组结果不支持 PINN 在该协议上优于 LSTM，也不支持把零违例解释为 PINN 独有的安全优势。

### 主要特点
- 物理信息神经网络（PINNs）与深度学习的混合架构
- 支持NASA和CALCE电池数据集
- 共形预测（Conformal Prediction）不确定性估计
- ONNX导出与边缘设备部署支持

---

## 🚀 快速开始

### 5分钟上手（演示版）

这是一个**确保能运行**的简化版本：

```bash
# 1. 克隆项目
git clone https://github.com/Zhi-Chao-PAN/safety-critical-battery-prognostics.git
cd safety-critical-battery-prognostics

# 2. 安装依赖
pip install -r requirements.txt

# 3. 运行快速演示
python main.py
```

这个演示会：
- 创建合成的电池数据
- 训练一个简单的基线模型
- 展示基本的预测结果

### 完整版本（需要数据集）

要运行完整版本，你需要：
1. 下载NASA或CALCE数据集
2. 配置数据路径
3. 详细步骤请参考 `docs/` 目录下的文档

---

## 📚 项目文档 (Documentation)

详细的文档位于 `docs/` 目录下：
- [**API 模块参考**](docs/technical/API_REFERENCE.md) - 核心类与函数接口说明
- [目录结构指南](docs/DIRECTORY_STRUCTURE.md) - 项目文件布局及其功能
- [部署与边缘集成](docs/deployment/DEPLOYMENT_GUIDE.md) - 从 ONNX 到硬件的部署 SOP
- [ISO 26262 案例](docs/industrial/ISO26262_Safety_Case.md) - 汽车工业功能安全分析方案

---

## 📁 项目结构

```
safety-critical-battery-prognostics/
├── src/                    # 源代码
│   ├── data/               # 数据加载与处理
│   ├── models/             # 模型定义
│   ├── features/           # 特征工程
│   ├── training/           # 训练流程
│   ├── evaluation/         # 评估指标
│   ├── uncertainty/        # 不确定性估计
│   ├── physics/            # 物理模型
│   ├── safety/             # 安全诊断
│   ├── deployment/         # 部署相关
│   └── utils/              # 工具函数
├── data/                   # 数据目录
├── docs/                   # 文档
├── notebooks/              # Jupyter notebooks
├── tests/                  # 单元测试
├── figures/                # 可视化结果
├── results/                # 实验结果
├── main.py                 # 快速演示入口（推荐先试这个）
├── _legacy_main.py         # 旧版完整入口（LEGACY - 不推荐新用户使用）
├── requirements.txt        # Python依赖
├── pyproject.toml          # 项目配置
└── README.md               # 本文件
```

---

## 📄 引用（Citation）

如果这个项目对你的研究有帮助，请引用：

```bibtex
@software{pan2026battery,
  author = {Pan, Zhichao},
  title = {Micro-Macro Time-Scale Decoupling for Battery RUL Prediction},
  year = {2026},
  url = {https://github.com/Zhi-Chao-PAN/safety-critical-battery-prognostics}
}
```

---

## 🤝 如何贡献

我们欢迎任何形式的贡献！请查看：
- [CONTRIBUTING.md](CONTRIBUTING.md) - 贡献指南
- [FAQ.md](docs/FAQ.md) - 常见问题
- [ROADMAP.md](docs/ROADMAP.md) - 项目路线图

---

## 📬 联系方式 (Contact)

- **Email**: [18652585856@163.com](mailto:18652585856@163.com)
- **Issues**: [GitHub Issues](https://github.com/Zhi-Chao-PAN/safety-critical-battery-prognostics/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Zhi-Chao-PAN/safety-critical-battery-prognostics/discussions)

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

---

<div align="center">
<i>本项目正在积极开发中。如果你觉得有帮助，请给个 ⭐ Star！</i>
</div>
