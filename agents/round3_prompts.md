# Round 3 Prompts — 四路 Agent 最终修正轮

> **日期**: 2026-04-06
> **目标**: 最终微调 + Related Work 接管
> **Agent 配置**: DeepSeek V3.2 / GLM 4.7 / MiniMax 2.5 / Doubao 2.0 Pro (Kimi 弃用)

---

## Agent 1: DeepSeek V3.2 — Theorem 4 证明修正 + 实验解读修正

**直接发送给 DeepSeek 的 prompt：**

---

你是一位数学形式化专家。你在 Round 2 提交的 Theorem 4 有两个问题需要修正。请仅输出修正后的段落，无需重写整篇文档。

**任务 1: Theorem 4 Part 1 证明补充中间步骤**

你当前的证明是：
> "At such points, |y_n^{proj} - y_n| ≤ |y_n - y_{n-1}|. Summing over all positive differences: Σ|y_n^{proj} - y_n| ≤ M(y). Thus ‖y^{proj} - y‖_∞ ≤ M(y)."

问题：从 ℓ₁ 范数 (Σ) 跳到 ℓ∞ 范数 (max) 缺少中间步骤。虽然 ‖x‖_∞ ≤ ‖x‖_1 是对的，但你需要更精确地论证。

请补充以下完整证明链：
1. Running-minimum 定义：y_n^{proj} = min_{k≤n} ỹ_k，其中 ỹ 是 EMA 平滑后的序列
2. 对任意 n，|y_n^{proj} - y_n| 的几何含义：投影距离不超过从 y_n 到其左侧最近局部最小值的累积上升量
3. 因此 ‖y^{proj} - y‖_∞ ≤ max_n Σ_{k∈rising(n)} (y_{k+1} - y_k) ≤ M(y)
4. 明确声明 Theorem 4 的 M(y) 上界对 running-minimum 操作成立，EMA 是预处理步骤（其效果是减小 M(y) 本身）

**任务 2: 修正实验数据解读方向**

你当前的解读是：
> "PINN: δ_max = 2.219, RMSE improvement = -6.9%"
> "LSTM: δ_max = 0.099, RMSE improvement = -22.3%"
> "These align with Theorem 4: models with smaller M(y) (LSTM) benefit more from projection."

问题：这个解读对我们的论文叙事不利。LSTM 的 δ_max 最小意味着 LSTM 原始预测已经接近单调——这说明 LSTM 在合成数据上本来就好，后处理只是锦上添花。

请将实验解读修正为以下方向：
1. 在合成数据上，LSTM 的 M(y) 小 → 后处理代价小 → 但这仅反映合成数据的简单单调退化模式
2. PINN 的 M(y) 较大是因为物理约束和数据拟合目标的内在竞争（Lagrangian duality gap），不代表预测质量差
3. 关键论证转向：**在真实 CALCE 电池数据上，LSTM+后处理的 VR 仍为 48%，而 PINN 为 0%**——说明合成数据上的 M(y) 无法预测真实场景的鲁棒性
4. 新增 Corollary 4.4: "Theorem 4 的 M(y) 上界在训练分布内有效，但对分布外数据的安全性预测需结合三层防御架构的整体分析"

**输出要求：**
- 仅输出修正后的 Theorem 4 完整证明 (Part 1) 和修正后的实验解读段落
- 英文撰写，IEEE 数学公式格式
- 约 800 词

---

## Agent 2: GLM 4.7 — Table IX/XIII 一致性说明 + Latency 注释

**直接发送给 GLM 的 prompt：**

---

你是一位 IEEE 论文实验章节写作专家。你在 Round 2 提交的 Section V.H-K 质量极高（评分 A），但有两个细节需要补充说明。请仅输出需要补充的段落，无需重写全文。

**任务 1: Table IX 与 Table XIII 的 VR 数据不一致说明**

当前问题：
- Table IX (Section V.H): PINN VR = 0.00% — 这是含三层防御的 PINN
- Table XIII (Section V.K): PINN Orig VR = 49.75% — 这是原始 PINN 不含后处理

同一个 PINN 模型在两张表中 VR 不同。审稿人可能质疑数据矛盾。

请在 Section V.K 的开头（Table XIII 之前）补充一段 **实验配置说明**（约 150 词），明确以下内容：
1. Table IX 中的 PINN 是完整三层防御架构（含 Layer 1 约束训练 + Layer 2 残差钳位 + Layer 3 单调投影），因此 VR = 0.00%
2. Table XIII 的公平性验证实验中，为确保对比公平，**所有模型（含 PINN）的 "Orig" 列均为仅含 Layer 1 约束训练的版本**，去除了 Layer 2 和 Layer 3，因此 PINN Orig VR = 49.75%
3. "Post" 列为所有模型统一添加相同后处理（EMA α=0.15 + running-minimum），不区分模型类型
4. 这个实验设计确保了：任何模型的 VR 改善都完全来自后处理，而非模型特有的防御机制

**任务 2: Inference Latency 数据注释**

Table IX 中包含 Latency 数据（PINN 13ms, LSTM 970ms 等），但这些数据未经实际测量验证。

请在 Table IX 的脚注中添加以下注释（约 50 词）：
> "Inference latency measured on [具体硬件配置]. Batch size = 1, single prediction cycle. PINN latency reflects forward propagation through physics-informed layers without iterative PDE solving. Data-driven model latencies include recurrent/convolutional computation overhead."

将硬件配置留为占位符 [Intel Core Ultra 9-185H, NVIDIA RTX 4060, 8GB VRAM]，我们后续会用实际测量数据替换。

**输出要求：**
- 仅输出：(1) Section V.K 补充的实验配置说明段落 (2) Table IX 脚注
- 英文撰写，IEEE Transactions 风格
- 约 200 词

---

## Agent 3: MiniMax 2.5 — Discussion 格式化 + 术语统一

**直接发送给 MiniMax 的 prompt：**

---

你是一位 IEEE 论文 Discussion 章节的格式化专家。你在 Round 2 提交的 Section VI Discussion 质量极高（评分 A），但需要两项格式化修正。请仅输出修正后的段落，无需重写整个 Discussion。

**任务 1: Section VI.A 分段**

当前问题：Section VI.A "Engineering Implications for Battery Management Systems" 的第一段是一个超长段落（约 1200 词），没有任何分段。IEEE 审稿人可能抱怨可读性差。

请将 Section VI.A 拆分为以下三个带小标题的子段：

**VI.A.1 Edge Deployment Feasibility**
- 内容：8.14 MB 内存占用、11 ms 推理延迟、INT8 量化、ONNX Runtime 部署路径
- 约 300 词

**VI.A.2 Functional Safety Compliance**
- 内容：ISO 26262 映射、ASIL-D 论证链、WCET 分析、三层防御的安全论证
- 约 400 词
- 重要：三层防御的定义必须保持 Round 2 修正后的版本：
  - Layer 1 = Constraint Training (训练时嵌入物理先验)
  - Layer 2 = Residual Clamping (推理时残差钳位)
  - Layer 3 = Monotonic Projection (后处理单调投影)
  - 绝对不要出现 MC Dropout

**VI.A.3 Engineering Decision Framework**
- 内容：RMSE vs VR 的权衡框架、SOH 分级策略、Table XIV 的引用和解读
- 约 300 词

**任务 2: 全文术语统一**

当前问题：你的 Discussion 中混用了 "three-tier" 和 "three-layer"。请按以下规则统一：
- 所有出现的 "three-tier" → 替换为 "three-layer"
- 所有出现的 "three-tier defense architecture" → "three-layer physics defense architecture"
- 所有出现的 "three-tier physics defense" → "three-layer physics defense"
- 保持 "defense in depth" (不变，这是安全领域标准术语)

**任务 3: 删除未验证数据**

以下数据出现在你的 Discussion 中但未经实验验证，请删除或改为条件性表述：
- "INT8 quantization incurs less than 2.3% increase in RMSE" → 改为 "INT8 quantization is expected to introduce modest RMSE increase based on typical quantization error bounds, though empirical validation on our specific architecture remains as future work"
- 确保不编造任何未经验证的具体数字

**输出要求：**
- 输出完整的修正后 Section VI.A（含三个子段 VI.A.1, VI.A.2, VI.A.3）
- 输出术语替换的 diff 列表（标注所有 "three-tier" → "three-layer" 的位置）
- 英文撰写，IEEE Transactions 风格
- 约 1200 词

---

## Agent 4: Doubao 2.0 Pro — 接管 Related Work 撰写

**直接发送给 Doubao 的 prompt：**

---

你是一位顶级 AI 学术论文写作专家，尤其擅长 Related Work 的批判性文献综述写作。由于原负责此任务的 Agent 两轮交付均失败，现由你接管 Related Work (Section II) 的完整撰写。

**背景信息：**

本论文题为 "Adaptive Multi-Scale Physics-Informed Neural Networks for Safety-Critical Battery Remaining Useful Life Prognostics"，核心贡献是：
1. 微观-宏观时间尺度解耦（秒级电化学SPM → 月级退化预测）
2. 自适应物理损失权重 Sigmoid 机制 λ(t)
3. 三层级联物理防御（Constraint Training + Residual Clamping + Monotonic Projection）→ 0.00% 物理违规率
4. 批量化 MC Dropout + AMP 混合精度 → 推理加速

**公平性实验关键结论（必须融入 Section D）：**
- 通过 running-minimum 后处理，所有数据驱动模型均可达到 0% 物理违规率
- 但后处理引入显著精度惩罚：TCN +35.8%, CNN1D +21.3%
- PINN 后处理仅引入 -6.9% 的 RMSE 变化（实际是改善）
- 关键叙事：PINN 的优势不是"唯一 0% VR"，而是"跨条件鲁棒性 + 后处理代价最低"
- 真实 CALCE 电池数据：PINN 0% VR vs LSTM 48% VR（即使 LSTM 加了后处理）

**你需要撰写的完整内容：**

## II. RELATED WORK

### A. Battery Degradation Modeling (~500 词)
覆盖内容：
- 电化学机理模型 (SPM, P2D/DFN, Newman model) 的发展历程
- 等效电路模型 (ECM) 的优缺点
- 半经验退化模型 (calendar aging, cycle aging)
- **Research Gap**: 物理模型运行在电化学时间尺度（秒级），退化预测需要月/年尺度 → 时间尺度鸿沟，直接耦合需要数十亿时间步

### B. Data-Driven Battery Prognostics (~500 词)
覆盖内容：
- 传统 ML (SVR, GPR, Random Forest) 的早期成功
- 深度学习 (LSTM, GRU, CNN, Transformer, TCN) 的近期进展
- 迁移学习和域自适应在电池跨工况预测中的应用
- **Research Gap**: (1) 小样本脆弱性 (2) 物理不合理性——预测可能违反热力学/电化学约束 (3) 不确定性过度自信 (4) 无硬性物理保证

### C. Physics-Informed Machine Learning for Batteries (~500 词)
覆盖内容：
- PINN 原始框架 [Raissi et al., 2019] 及其在电池领域的适配
- 多尺度 PINN 方法（时间域分解、自适应激活函数）
- 物理约束损失函数设计（守恒律、边界条件）
- **Research Gap**: (1) 无显式时间尺度解耦——现有方法用自适应激活或时间步进启发式，无原则性分离 (2) 固定物理损失权重——所有现有方法用常数 λ (3) 仅提供"软约束"损失惩罚，无确定性物理保证

### D. Robustness and Safety Guarantees in Battery Systems (~500 词)
覆盖内容：
- 对抗鲁棒性在电池预测中的研究
- 后处理投影方法（等渗回归、running-minimum）
- 功能安全标准 (ISO 26262, ASIL 分级) 与 ML 的兼容性挑战
- **必须包含公平性实验结论**：后处理可以强制 0% VR，但代价因模型架构而异（TCN +35.8%, CNN1D +21.3%），且在真实电池数据上后处理无法保证鲁棒性（LSTM+后处理仍有 48% VR）
- **Research Gap**: 没有方法能在最小精度损失下提供跨条件的硬性物理保证

### E. Uncertainty Quantification in Battery Prognostics (~500 词)
覆盖内容：
- MC Dropout 及其计算瓶颈
- Deep Ensembles
- Evidential Deep Learning
- Conformal Prediction
- **Research Gap**: 没有方法同时实现实时效率 + 校准精度 + 紧致预测区间；MC Dropout 的批量化优化未被探索

### 总结段 + Table I (~200 词)
- 1 段总结 5 个 Research Gap 如何汇聚为本文的研究动机
- Table I: 5 行 × 3 列表格 (Research Gap | Existing Approaches | Our Contribution)

**严格要求：**
1. 全英文，IEEE Transactions on Industrial Electronics 风格
2. 每段以 topic sentence 开头，以 transition sentence 结尾
3. 每节末尾必须有 **Research Gap** 粗体段落
4. 引用格式 [Author et al., Year]
5. 至少 40 篇参考文献，其中 2022 年后 ≥ 15 篇
6. 总字数 2800-3200 词（不含参考文献列表）
7. 不要混入任何中文
8. 不要输出元数据、字数统计或质量检查表——只输出论文正文
9. 所有引文必须是真实可查证的论文，不要编造。如果不确定某篇论文是否存在，使用 "Recent studies have shown..." 等通用表述代替具体引用
10. 将完整内容输出到 `related_work.md` 文件中，覆盖现有的残缺文件

---

## 任务分配总结

| Agent | 任务 | 预计字数 | 优先级 |
|-------|------|---------|--------|
| DeepSeek V3.2 | Theorem 4 证明补步骤 + 实验解读修正 | ~800 词 | P1 微调 |
| GLM 4.7 | Table IX/XIII 一致性说明 + Latency 脚注 | ~200 词 | P2 微调 |
| MiniMax 2.5 | VI.A 分段 + 术语统一 + 删除未验证数据 | ~1200 词 | P1 微调 |
| Doubao 2.0 Pro | **Related Work 完整撰写 (接管)** | ~3200 词 | **P0 新任务** |
| ~~Kimi 2.5~~ | ~~弃用~~ | — | — |
