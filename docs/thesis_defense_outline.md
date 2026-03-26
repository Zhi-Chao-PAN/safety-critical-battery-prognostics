# 硕士学位论文答辩PPT大纲

## 论文题目
**微-宏时间尺度解耦的物理信息神经网络在电池无分布寿命预测中的应用**

*Micro-Macro Time-Scale Decoupling in Physi-Neural Networks for Distribution-Free Battery Prognostics*

---

## 第1页：封面页

- 论文题目
- 答辩人：
- 指导教师：
- 学科专业：
- 答辩日期：

---

## 第2页：研究背景与意义

### 研究背景
- 锂离子电池在新能源汽车、储能系统中的广泛应用
- 电池安全事故频发，寿命预测的重要性
- 传统方法面临"时间尺度黑洞"挑战

### 研究意义
- 理论意义：解决多时间尺度耦合的物理信息神经网络训练问题
- 实际意义：为BMS边缘设备提供可靠的电池寿命预测方案

---

## 第3页：国内外研究现状

### 数据驱动方法
- LSTM、TCN等序列模型
- 优点：能捕捉复杂非线性关系
- 缺点：缺乏物理可解释性，易过拟合

### 物理驱动方法
- 单粒子模型（SPM）、伪二维模型（P2D）
- 优点：有严格的物理理论支撑
- 缺点：计算复杂度高，难以实时应用

### 物理信息神经网络（PINNs）
- 结合数据驱动与物理驱动的优势
- 但面临多时间尺度耦合带来的OOM问题

---

## 第4页：问题陈述

### "时间尺度黑洞"问题
- 微观电化学过程：秒级时间尺度
- 宏观容量衰减：月级时间尺度
- 传统PINNs：直接桥接导致BPTT计算图爆炸

### 现有方法的局限性
- 显存占用高（~200-500MB）
- 无法满足边缘设备部署要求
- 缺乏可靠的不确定性估计

---

## 第5页：本文主要贡献

### 贡献1：微-宏时间尺度解耦架构
- 峰值显存：8.14 MB（>10x降低）
- 零OOM训练

### 贡献2：物理边界约束
- 可微分Sigmoid钳制层
- 100%物理合理性保证

### 贡献3：边缘设备就绪
- ONNX INT8推理：0.078ms
- 满足BMS实时要求

### 贡献4：无分布不确定性估计
- 共形分位数回归（CQR）
- 数学保证的95%覆盖

### 贡献5：安全诊断框架
- ISO 26262对齐的FMEA代理
- 自动生成安全报告

---

## 第6页：系统架构（配图：fig04_ablation_architecture.png）

### 整体架构
- 微观尺度：SPM-FDM沙箱
- 特征提取：物理特征提取器
- 宏观尺度：RUL预测网络
- 不确定性：共形预测模块

### 关键技术
- 计算图解耦
- 可微分物理约束
- 端到端训练

---

## 第7页：微观尺度：SPM-FDM沙箱

### 单粒子模型（SPM）
- Fick第二定律：$\frac{\partial c_s}{\partial t} = \frac{D_s}{r^2}\frac{\partial}{\partial r}\left(r^2\frac{\partial c_s}{\partial r}\right)$
- 边界条件：$r=0$时$\frac{\partial c_s}{\partial r}=0$；$r=R$时$c_s=c_{s,surf}$

### 有限差分法（FDM）离散化
- 空间离散：$N_r$个网格点
- 时间离散：欧拉积分
- 周期内计算，不跨周期展开

---

## 第8页：物理特征提取

### 提取的物理特征
- 最大径向浓度梯度：$\nabla c_{s,max}$
- 累积机械应力：$\sigma_{acc}$
- 表面浓度：$c_{s,surf}$

### 可微分约束
- Sigmoid钳制：$C_{pred} = C_{nom} \cdot \sigma(\cdot)$
- 保证：$0 < C_{pred} \le C_{nom}$

---

## 第9页：共形分位数回归（CQR）

### 为什么选择CQR？
- 贝叶斯MC-Dropout：无覆盖保证
- CQR：数学保证的有限样本覆盖

### CQR流程
1. 训练分位数回归模型
2. 在校准集上计算残差
3. 构造共形预测区间
4. 提供95%覆盖保证

---

## 第10页：实验设置

### 数据集
- **NASA数据集**：B0005-B0007, B0018
- **CALCE数据集**：CS2系列

### 评价指标
- RMSE、MAE（容量预测）
- RUL误差（寿命预测）
- 覆盖率（不确定性估计）

### 基线方法
- Pure Data-Driven（LSTM）
- Pure Physical（SPM-only）

---

## 第11页：NASA数据集结果（配图：fig10_prediction_comparison.png）

### RUL预测性能
| Battery ID | Actual RUL | Predicted RUL | Abs Error |
|------------|------------|---------------|-----------|
| B0005 | 23.27 | 20.00 | 3.27 |
| B0006 | 20.50 | 20.00 | 0.50 |
| B0007 | 34.00 | 20.00 | 14.00 |
| B0018 | 18.73 | 12.94 | 5.79 |

**平均绝对误差：5.89 cycles**

---

## 第12页：CALCE数据集结果（配图：fig03_comparison.png）

### 容量预测性能
| Model | RMSE (Ah) | MAE (Ah) |
|-------|-----------|----------|
| Pure Data-Driven | 0.089 | 0.072 |
| Pure Physical | 0.156 | 0.121 |
| **Ours** | **0.042** | **0.031** |

### 收敛速度
- MSE在1个epoch内从20M降至<38
- >3个数量级的提升

---

## 第13页：消融实验结果（配图：fig04_ablation_architecture.png, fig05_ablation_seqlen.png, fig06_ablation_hidden.png）

### 架构消融
| Variant | RMSE (Ah) | Physics | Uncertainty |
|---------|-----------|---------|-------------|
| No Physics | 0.081 | ❌ | MC Dropout |
| No Conformal | 0.052 | ✅ | MC Dropout |
| **Full Model** | **0.042** | ✅ | Conformal |

### 关键发现
- 物理约束：OOD鲁棒性提升78%
- 共形预测：精确95%覆盖

---

## 第14页：计算效率分析（配图：fig08_train_time.png, fig09_complexity.png）

### 显存占用
| Component | VRAM Usage |
|-----------|------------|
| Pure Data-Driven | ~200-500 MB |
| **Ours (Training)** | **8.14 MB** |
| **Ours (Inference)** | **<1 MB** |

### 边缘部署延迟
| Format | Mean Latency (ms) |
|--------|-------------------|
| FP32 ONNX | **0.078** |
| INT8 ONNX | 0.093 |

**比BMS要求快640倍**

---

## 第15页：不确定性估计验证（配图：fig_reliability_diagram.png）

### 共形预测覆盖
- **95%名义覆盖率**
- **实测覆盖率：94.8%**
- 数学保证的有限样本覆盖

### 可靠性图
- 预测概率与实际频率对齐良好
- 无过自信或欠自信问题

---

## 第16页：安全分析（配图：fig11_ood_dynamic_boundary.png）

### 物理约束满足
- **100%**预测满足$0 < C_{pred} \le C_{nom}$
- 无物理不可行的负容量预测

### FMEA诊断框架
- 锂沉积风险检测
- 机械断裂风险评估
- ISO 26262对齐的安全报告

---

## 第17页：总结与展望

### 工作总结
- 提出微-宏时间尺度解耦架构
- 在NASA和CALCE数据集上验证
- 实现边缘设备就绪的高性能预测

### 创新点
- 显存高效的解耦训练
- 物理约束保证
- 可靠的不确定性估计

### 未来工作
- 扩展到更多电池化学体系
- 实车数据验证与部署
- 多尺度多物理场耦合

---

## 第18页：致谢

感谢：
- 指导老师的悉心指导
- 实验室同学的帮助
- 家人的支持

---

## 附录：可视化图索引

| 图号 | 文件名 | 说明 |
|------|--------|------|
| 1 | fig01_degradation.png | 容量衰减曲线 |
| 2 | fig02_correlation.png | 相关性分析 |
| 3 | fig03_comparison.png | 方法对比 |
| 4 | fig04_ablation_architecture.png | 架构消融 |
| 5 | fig05_ablation_seqlen.png | 序列长度消融 |
| 6 | fig06_ablation_hidden.png | 隐藏层消融 |
| 7 | fig07_per_fold.png | 交叉验证结果 |
| 8 | fig08_train_time.png | 训练时间 |
| 9 | fig09_complexity.png | 模型复杂度 |
| 10 | fig10_prediction_comparison.png | 预测对比 |
| 11 | fig11_ood_dynamic_boundary.png | OOD动态边界 |
| 12 | fig_reliability_diagram.png | 可靠性图 |
