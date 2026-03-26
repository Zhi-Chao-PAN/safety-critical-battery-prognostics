# 项目核心架构规范 (Project Architecture)

> **[📝 AI Context Hand-off]**: 
> 本文档由系统的 AI Co-PI (SOTA Architect) 自动生成。在开启新的 AI 会话时，阅读此文档即可瞬间掌控全局技术堆栈与开发红线，实现零摩擦交接。

## 1. 架构定位与主动纠偏 (Architecture Positioning & Proactive Critique)
**【主动查漏补缺 (Proactive Critique)】**：
当前项目（`safety-critical-battery-prognostics`）**并非**标准的 Node.js/Web 前端工程（如 React/Vue/Angular），而是一个核心为 **Python 机器学习与数据科学**的算法工程项目，专注于电池健康管理、不确定性预测（基于 PyMC / BNN）及边缘部署（ONNX）。
因此，所有关于 "Axios、Tailwind、Vuex/Redux" 的前端标准范式在此架构下均不适用。本项目的 "UI/前端" 是一个基于数据流式渲染的 **Streamlit Dashboard**，主要用于科研呈现与模型指标可视化，而非高并发的客户端 Web 应用。

---

## 2. 核心技术栈概览 (Core Technology Stack)
- **底层架构语言**：Python (>= 3.10)
- **前端/UI 框架**：Streamlit (`src/ui/dashboard.py`)
- **UI 组件库与图表**：Plotly (`plotly.graph_objects`)
- **状态管理**：依托 Streamlit 自带的 `Session State` 与 `@st.cache_data` 实现组件缓存以及响应式重绘。
- **构建与依赖工具**：
  - 传统配置：`requirements.txt`
  - 现代化构建及格式约束：`pyproject.toml` (包含 Setuptools, Ruff, Mypy 等严格规范)
- **核心算法生态**：PyTorch (深度学习)、PyMC (概率编程与不确定性量化)、Scikit-learn、Numpy、Pandas。
- **模型推理与边缘部署**：ONNX (`onnxruntime`)

---

## 3. 核心目录结构梳理 (Directory Structure)

项目围绕数据流水线、模型训练、安全决策与可视化进行了极其严格的模块化解耦：

```text
safety-critical-battery-prognostics/
├── main.py                     # 全局 Pipeline 编排与启动入口
├── pyproject.toml              # 核心构建配置、Linting 规范与类型约束
├── src/                        # 核心业务与算法源码库
│   ├── ui/                     # [Frontend] 前端可视化目录（Streamlit Dashboard）
│   ├── data/                   # [Data] 数据总线（统一加载、清洗验证模块）
│   ├── features/               # [Feature] 特征工程抽取模块
│   ├── physics/                # [Physics] 物理先验机制模型（如等效电路参数、退化方程）
│   ├── models/                 # [Model] 网络模型架构定义 (如 TCN, PINN, BNN 等)
│   ├── training/               # [Training] 训练循环与优化器封装、Loss 调度
│   ├── evaluation/             # [Eval] 性能评估、泛化与 OOD 测试脚本
│   ├── safety/                 # [Engine] 安全决策推理引擎（阈值熔断机制）
│   ├── uncertainty/            # [Uncertainty] 认知/偶然不确定性量化与校准
│   ├── deployment/             # [Edge] 模型量化与边缘端部署转换 (ONNX)
│   └── utils/                  # [Utils] 通用工具链 (日志、配置解析等)
├── scripts/                    # 各种离线实验、可视化导出与分析脚本
├── docs/                       # 项目文档储备与进度沉淀 (Project Progress / Architecture)
├── configs/                    # 模型超参数、数据管道拦截阈值的结构化 YAML 文件
├── tests/                      # 单元测试与集成测试入口 (Pytest)
└── results/                    # 模型输出产物、图表报告、实验日志缓存
```

---

## 4. 重点业务代码约定与规范 (Coding Conventions)

### 4.1 数据请求与流转 (Data Fetching & Pipeline)
- **无常规 REST/Axios 请求**：由于本项目是纯数据计算管线，并不依赖外部微服务提供 HTTP API。数据流直接由 `src.data.unified_loader.UnifiedDataLoader` 从本地或挂载的高速存储 (`data/battery_data`) 中提取。
- **数据流强校验拦截**：数据在流入特征工程之前，必须经由 `src.data.validator.DataValidator` 产出校验报告。异常或越界数据将被清洗或引发管道断言中断，拒绝由于脏数据导致的数据污染。

### 4.2 状态管理 (State Management)
- **模型超参数与静态状态**：全局计算管线状态存储于 `configs/` 目录的结构化文件内，实现配置与逻辑分离。
- **UI 状态响应式更新**：可视化层没有采用 Redux 或 Vuex。取而代之的是 `Streamlit` 的单向数据流机制；跨交互状态通过 `@st.cache_data` 缓存昂贵的数据读取/模型推理过程；筛选条件或阈值拖动（如 `selected_bat`、`rul_critical`）通过 Streamlit 绑定的响应式组件变量即时下流，驱动整个页面无缝刷新。

### 4.3 样式与组件 (Styles and Components)
- **框架内置主导**：未使用 Tailwind CSS 或外部预处理器 (SCSS)，总体样式排版完全依赖 `Streamlit` 内置的流式网格（`st.columns`）和 Python 原生的 HTML/CSS 安全嵌入。例如对安全等级 `GREEN/YELLOW/RED` 卡片采用 `st.markdown(unsafe_allow_html=True)` 显式注入色彩。
- **渲染展示高度模块化**：可视化组件重度依赖于 `Plotly` 进行交互封装，所有视图逻辑严格收敛在 `src/ui/dashboard.py` 下，实现底层模型算法库与渲染表现层的高效伪隔离。

### 4.4 潜在技术债与注意事项 (Technical Debt & Edge Cases)
1. **显存/内存泄漏隐患 (VRAM/RAM Leakage)**：在 Streamlit 这种随每次访问重载整个脚本的框架中，若缓存超大规模 Pandas DataFrame 或加载 `torch` 主模型，极易导致隐形内存/显存泄露。处理方案：必须对 `@st.cache_data` 执行严格的 `max_entries` 限定，并适时应用 `torch.cuda.empty_cache()` 防御。
2. **阻断式同步架构瓶颈**：Streamlit 目前基于同步阻塞执行树设计。当底层连接 `SafetyDecisionEngine` 或大规模 OOD / PyMC 采样计算时，易导致 UI 线程“假死”（UI Blocking）。在后续向商用 SaaS 转化的演进过程中，应将重计算下推推至异步队列（如 Celery 分发至云端 A40 集群计算），并将 UI 架构重构为 FastAPI (后端) + React/Next.js (前端) 的标准异步微服务架构。

---

## 5. 开发执行红线 (Development Rules of Engagement)
- **职责隔离原则 (Separation of Concerns)**：
  若需增加新型可视化页面，必须且只能在 `src/ui/` 目录下操作，绝对禁止将 UI 渲染逻辑 (`st.write`, `matplotlib.pyplot.show`) 污染或混入底层算法核心 (`src/models/`, `src/physics/`) 中。
- **快速调试面板拉起**：
  使用命令 `streamlit run src/ui/dashboard.py` 启动 UI 联调系统。 
- **零容忍静态检查 (Zero-Tolerance Linting)**：
  CI/CD 提交流程强依赖 `ruff` 和 `mypy`。所有新增函数或修改必须携带极其严格的 Type Hints (类型注解)，并且必须在本地无 Warnings 抛出的情况下通过 `pytest` 测试。拒绝推断式 Lazy Code！
