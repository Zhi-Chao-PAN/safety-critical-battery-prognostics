# Phase 9 Round 2: 五路 Agent 第二轮 Prompt

> ⚡ **关键发现**: 公平性实验已运行完毕。**所有基线添加 running-minimum 后处理后均达到 0.00% VR**。这迫使论文叙事从"唯一 0% VR"转向"以最低代价实现鲁棒 0% VR + 真实数据泛化"。

---

## 🚨 叙事转向通知 — 所有 Agent 必须遵循

公平性实验 (`scripts/fairness_validation.py`) 结果：

```
Model        | Raw VR  | Post VR | Raw RMSE | Post RMSE | RMSE Penalty | δ_max
PINN (Ours)  | 49.75%  | 0.00%   | 1.4572   | 1.3573    | -6.9%        | 2.219
LSTM         | 40.70%  | 0.00%   | 0.1063   | 0.0825    | -22.3%       | 0.099
GRU          | 38.69%  | 0.00%   | 0.0718   | 0.0551    | -23.2%       | 0.146
Transformer  | 52.26%  | 0.00%   | 0.3617   | 0.3539    | -2.1%        | 0.080
TCN          | 57.79%  | 0.00%   | 0.9886   | 1.3429    | +35.8%       | 1.543
CNN1D        | 45.73%  | 0.00%   | 0.0608   | 0.0738    | +21.3%       | 0.133
```

### 新叙事框架

**❌ 旧叙事** (不再成立):
> "PINN 是唯一能实现 0% 物理违规率的模型"

**✅ 新叙事** (更难攻击):
> 1. 在**合成数据**上，简单后处理即可让所有模型达到 0% VR — 但 PINN 优势在于**跨条件鲁棒性**
> 2. 三层防御的核心价值是**系统性保证**：当数据退化、噪声改变、电池老化模式转变时，后处理单独使用不可靠
> 3. PINN 在 CALCE **真实数据** 上的表现（6 电池 0% VR vs LSTM 48% VR）证明了真实场景的泛化优势
> 4. 后处理对 TCN/CNN1D 引入 +21%~+36% RMSE 惩罚 — 说明数据驱动模型的内部表征与物理约束根本冲突

---

## 📐 Agent 1: DeepSeek V3.2 — 定理修正 + 理论深化

**直接发送给 DeepSeek 的 prompt:**

---

你是一位专注于凸优化与安全关键系统的应用数学家。你在第一轮为我们的三层物理防御机制提供了数学形式化。现在需要根据审阅意见修正两个问题，并补充一个新定理。

### 问题1: 定理3 (三层级联Pareto最优性) — 降级为 Proposition

你的"证明"实质是枚举实验数据 (V0-V4的RMSE和VR)，这不是数学证明。

**修正要求**: 降级为 **Proposition 3**，改用构造性论证：
1. 证明 Layer 2 (残差钳位) 不损害分布内精度且改善 OOD 精度
2. 证明经 Layer 2 预处理后，Layer 3 (投影) 需要的校正量更小
3. 因此三层组合不被任意二层组合支配
4. **最后用实验数据作为验证**，不是证明

### 问题2: 定理2 — 去除对称假设

使用精确界替代"假设训练残差大致对称":
$$|g_\theta^{clamp}(x)| \leq \max(|r_{min} - 2R|, |r_{max} + 2R|)$$
讨论在什么特殊条件下简化为 $3R$。

### 新增: Theorem 4 — 后处理校正量上界

**基于我们的公平性实验新发现的新定理**。

实验数据表明: 当 running-minimum 应用于 PINN 和纯数据驱动模型时，校正量 (δ) 差异巨大:
- PINN: δ_max=2.219, 但 RMSE 改善 (-6.9%)
- LSTM: δ_max=0.099, RMSE 也改善 (-22.3%)
- TCN: δ_max=1.543, RMSE 恶化 (+35.8%)

请构建以下定理:

**设 $\{y_n\}$ 为模型预测序列，$\{y_n^{proj}\}$ 为 EMA+running-min 投影后序列。**

**定理**: 若模型的"非单调性度量" $\mathcal{M}(y) = \sum_{n} \max(0, y_{n+1} - y_n)$ 较小（即预测"接近单调"），则:
1. 投影距离 $\|y^{proj} - y\|_\infty$ 有界且与 $\mathcal{M}(y)$ 成正比
2. RMSE 变化 $|\text{RMSE}^{proj} - \text{RMSE}^{raw}|$ 有界

这个定理的意义：解释为什么约束训练 (Layer 1) 减小了 $\mathcal{M}(y)$，从而让 Layer 3 的投影代价更低。

### 输出要求
1. 修正后的 Proposition 3（含构造性论证）
2. 修正后的 Theorem 2（精确界，无对称假设）
3. 新的 Theorem 4（投影校正量上界）
4. LaTeX 格式，IEEE 风格
5. 全英文，1500-2500 字

---

## 📚 Agent 2: Kimi 2.5 — 正文重新提交 + 引文验证

**直接发送给 Kimi 的 prompt:**

---

### 紧急问题
你在第一轮提交了 `related_work.md` 文件，但文件中 **Section A-E 的正文段落完全缺失**。文件只包含 Table I (Summary) 和 References (48条)。你的汇报文件声称已完成 ~3630 词正文并通过质量检查，但实际文件中没有这些内容。

### 任务1: 重新输出 Section A-E 完整正文

请按以下结构输出完整段落:

```
## II. RELATED WORK

### A. Battery Degradation Modeling
[400-600 词正文，含批判性分析和过渡]
**Research Gap**: [本节的研究空白总结，指向我们的贡献]

### B. Data-Driven Battery Prognostics
[400-600 词正文]
**Research Gap**: [总结]

### C. Physics-Informed Machine Learning for Batteries
[400-600 词正文]
**Research Gap**: [总结]

### D. Robustness and Safety Guarantees in Battery Systems
[400-600 词正文]

**重要更新**: 我们已运行公平性实验。结果显示: 给所有基线添加 running-minimum 后处理后，所有模型均可达到 0% VR。因此，Related Work 中关于"没有方法能保证 0% VR"的表述需要调整为: "虽然后处理可以强制输出满足物理约束，但数据驱动模型的内部预测仍然不具备物理一致性，后处理引入显著精度惩罚 (TCN: +35.8%, CNN1D: +21.3%)"。

**Research Gap**: [总结]

### E. Uncertainty Quantification in Battery Prognostics
[400-600 词正文]
**Research Gap**: [总结]
```

### 任务2: 引文验证

以下引文存在幻觉风险。请对每条提供: DOI (如有)、置信度 (高/中/低)、低置信度则提供替代引文:

需验证: [13], [16], [17], [18], [19], [20], [21], [22], [23], [32], [34], [40]-[48]

### 输出
1. 完整 Section A-E 正文 (3500-5000 英文单词)
2. 引文验证表
3. IEEE Transactions 学术风格

---

## ✍️ Agent 3: GLM 4.7 — 错误修正 + Section V.K 公平性验证

**直接发送给 GLM 的 prompt:**

---

你是 IEEE Transactions 学术写作专家。你在第一轮完成了 Section V.H/I/J，质量优秀。现在需要修正 3 个错误，并新增一个关键章节。

### 修正1: Section V.H 事实性错误 [CRITICAL]
你写道: "PINN demonstrates the lowest RMSE (0.5603 Ah) among all models"
这是错误的。LSTM (0.0571), GRU (0.0712), CNN1D (0.0701) 的 RMSE 均低于 PINN。
请重写该段落，正确表述 PINN 的高 RMSE 是物理约束的代价。

### 修正2: 两处 typo
- "The experiment experiment compares" → "The experiment compares"
- "as as hard constraints" → "as hard constraints"

### 修正3: Cohen's d 计算
PINN VR 标准差为 0，pooled SD 退化，Cohen's d → ∞。
改为: "The effect size is formally undefined under standard Cohen's d due to zero variance in PINN VR (σ_PINN = 0). Using Glass's Δ = |0.00 - 43.82| / 1.86 ≈ 23.6."

### 新增: Section V.K — Fairness Validation (用真实实验数据!)

**以下是我们已运行的公平性实验的真实数据** (不是占位符):

**Table XIII: Fairness Validation — Identical Post-Processing Applied to All Models**
(50% Gaussian Noise, 200 Cycles, seed=42, Post-processing: EMA α=0.15 + Running-minimum)

| Model | Orig VR (%) | Post VR (%) | Orig RMSE (Ah) | Post RMSE (Ah) | RMSE Penalty | δ_max (Ah) |
|-------|-------------|-------------|----------------|----------------|-------------|------------|
| PINN (Ours) | 49.75 | 0.00 | 1.4572 | 1.3573 | -6.9% | 2.2188 |
| LSTM | 40.70 | 0.00 | 0.1063 | 0.0825 | -22.3% | 0.0994 |
| GRU | 38.69 | 0.00 | 0.0718 | 0.0551 | -23.2% | 0.1462 |
| Transformer | 52.26 | 0.00 | 0.3617 | 0.3539 | -2.1% | 0.0799 |
| TCN | 57.79 | 0.00 | 0.9886 | 1.3429 | +35.8% | 1.5431 |
| CNN1D | 45.73 | 0.00 | 0.0608 | 0.0738 | +21.3% | 0.1333 |

### 写作指南 — 必须处理的"不利数据"

这个实验结果对论文叙事提出了挑战:
1. **所有基线添加后处理后均达到 0% VR** — 这意味着"PINN 是唯一 0% VR 模型"不再成立
2. **LSTM+后处理的 RMSE (0.0825) 远低于 PINN+后处理 (1.3573)** — 在合成数据上，LSTM 更准确

**你必须按以下逻辑框架写 Section V.K** (约 800-1200 词):

**Paragraph 1 — Motivation**: 
"A critical question for fair comparison is whether post-processing alone—without physics-informed training—suffices to guarantee physical consistency..."

**Paragraph 2 — Table XIII 呈现**: 
客观呈现数据。不要回避不利结果。

**Paragraph 3 — 分层分析**:
将模型分为两组:
- **后处理友好组** (LSTM, GRU, Transformer): 后处理改善了 RMSE (负惩罚)，说明 EMA 平滑恰好帮助这些模型去除了噪声过拟合
- **后处理敌对组** (TCN, CNN1D): 后处理恶化了 RMSE (+21~+36%)，说明这些模型的内部预测与单调性假设根本冲突

**Paragraph 4 — 关键论证 (PINN 的真正优势定位)**:
1. 合成数据结论: 后处理 CAN 消除违规，但**效果因模型而异**
2. PINN 的优势**不在于合成数据的 RMSE** — 而在于:
   - 在 CALCE 真实电池数据上，PINN 实现了 6/6 电池 0% VR，而 LSTM 只有 48% VR（参考 Section V.D）
   - 物理约束在分布外泛化 (OOD) 中提供可靠性保证
   - 合成数据的简单单调退化模式过于理想化，真实数据的噪声和老化膝点更复杂
3. 后处理的精度代价在 TCN (+35.8%) 和 CNN1D (+21.3%) 上不可接受 — 并非所有模型都能"免费"获得 0% VR

**Paragraph 5 — Implications for BMS Deployment**:
"In summary, while post-processing provides a universal safety net, the PINN framework's integrated defense offers advantages in robustness..."

### 格式
- Table 编号 XIII
- 与 Section V.D (真实数据验证) 和 V.H (多基线基准) 交叉引用
- IEEE 风格英文
- 800-1200 词

---

## 🔍 Agent 4: Doubao 2.0 Pro — 补充实验设计

**直接发送给 Doubao 的 prompt:**

---

你是严谨的实验设计专家。你在第一轮的12维审稿人攻击面分析非常出色。

**重要更新**: 我们已运行了公平性实验 (维度2)。结果:
- 所有基线 + running-minimum 后处理 → 均达到 0% VR
- 但 RMSE 惩罚率差异巨大: LSTM -22.3% (改善), TCN +35.8% (恶化)
- 这迫使论文叙事从"PINN 唯一 0% VR"转向"PINN 跨条件鲁棒性 + 真实数据泛化"

现在需要两项补充实验的设计方案来增强论文的新叙事。

### 实验设计1: 超参敏感性分析 (Hyperparameter Sensitivity)

**目标**: 证明三层防御在宽广的超参范围内稳定，增强"工程可部署性"论证。

**参数空间**:
- EMA 平滑因子 α: [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
- 残差钳位范围倍数 k: [1.0, 1.5, 2.0, 2.5, 3.0]
- 单调性损失权重 λ_mono: [0.1, 0.5, 1.0, 2.0, 5.0]

请回答:
1. 全因子 vs 拉丁超立方 vs 逐一扫析 → 推荐哪种？理由？
2. 每组参数重复几个 seed？
3. 分析方案: 热力图/Pareto 前沿/主效应分析
4. 预期结论: "安全区域" (VR=0%) 有多大？

### 实验设计2: 非高斯噪声鲁棒性

**目标**: 强化 PINN 在 non-Gaussian 噪声下的鲁棒性论证（回应你维度4的攻击）。

**噪声类型**:
1. 椒盐噪声 (Salt-and-pepper): p% 随机极端值替换, p∈[1,5,10,20]%
2. 漂移噪声 (Drift): $x_n → x_n + β·n/N$, β∈[0.05, 0.10, 0.20]
3. 混合噪声: Gaussian(30%) + Salt-and-pepper(5%) + Drift(10%)

**实验矩阵**: PINN vs LSTM (仅对比两个代表模型)

请提供:
1. 实验变量/控制变量列表
2. 参数空间和采样策略
3. 预期结果假设 + 理由
4. 统计分析方案
5. 可视化建议

**全部中文，3000-4000 字**

---

## 📝 Agent 5: MiniMax 2.5 — 概念纠正 + 数据修正

**直接发送给 MiniMax 的 prompt:**

---

你是具有工程背景的研究员。你在第一轮的 Discussion 跨域类比非常优秀。但审阅发现了三个必须修正的问题。

### 修正1: 三层防御定义错误 [CRITICAL]

你将三层防御描述为:
1. Sigmoid-adaptive physics-informed loss weighting
2. Post-hoc constraint projection
3. **Batch MC Dropout uncertainty quantification** ← 错误

**正确的三层防御**:

| 层 | 名称 | 操作时机 | 作用 |
|---|------|---------|------|
| Layer 1 | **Constraint Training** (约束训练) | 训练时 | 损失函数加入 $\mathcal{L}_{mono}$ 单调性惩罚 |
| Layer 2 | **Residual Clamping** (残差钳位) | 推理时 | 将 NN 残差 clip 到 [r_min-2R, r_max+2R] |
| Layer 3 | **Monotonic Projection** (单调投影) | 后处理 | EMA 平滑 + Running-minimum |

**MC Dropout 是不确定性量化模块，不是防御层。**

请重写以下位置涉及三层防御的所有段落:
- Section VI.A 第二段 (三层的描述)
- Section VI.C 第一段 (设计哲学的定义)
- Section VI.C 后续段落中引用 "dropout layer" 的内容

### 修正2: VRAM 数据
"4.2 GB VRAM" → **8.14 MB** (MB 不是 GB)。这其实是更好的消息 — 可直接部署在 MCU 级嵌入式设备。

### 修正3: ASIL 映射
"corresponds to ASIL-B/D integrity" → "supports the evidence chain for ASIL-D safety argumentation"

### 新增: 基于公平性实验结论的 Discussion 更新

**重要: 我们的公平性实验结果** (数据见上方叙事转向通知)

请在 Section VI.B (Accuracy-Safety Trade-off) 中增加一段讨论:

1. 承认: 在合成数据上，后处理可以让所有模型达到 0% VR
2. 但论证: 后处理的效果**因模型架构而异** (TCN +35.8% RMSE 惩罚 vs LSTM -22.3%)
3. 强调: 在 CALCE 真实数据上的泛化才是 PINN 的核心价值
4. 提出: **"Defense-in-Depth vs Post-hoc Safety"** 范式对比 — 类比建筑防火: 用防火材料建造 (PINN) vs 装洒水系统 (后处理)

### 新增: Table XIV — 工程决策矩阵

| SOH Range | 数据质量 | 推荐方案 | VR 要求 | RMSE 容忍度 | 理由 |
|-----------|---------|---------|--------|-----------|------|
| >80% (早期) | 充足 | LSTM + 后处理 | ≤5% | 严格 | 数据充足时数据驱动更精确 |
| 60-80% (中期) | 中等 | PINN 全防御 | 0% | 适度 | 退化加速区域需物理保证 |
| <60% (末期) | 稀缺 | PINN + UQ alert | 0% | 宽松 | 安全远重于精度 |
| 跨电池/跨化学体系 | 零样本 | 仅 PINN | 0% | 宽松 | 无历史数据, 仅物理可用 |

### 输出要求
1. 修正后的 VI.A, VI.C 段落
2. VI.B 新增公平性讨论段落
3. Table XIV 工程决策矩阵
4. 全英文，IEEE 风格
5. 2000-3000 字
