# Core Architecture Guidelines (Project Architecture)

> **[📝 AI Context Hand-off]**:
> This document is automatically maintained by the system's AI Co-PI (SOTA Architect). Reviewing this document at the initiation of a new AI session provides immediate contextual mastery over the global tech stack and developmental red lines, enabling zero-friction onboarding.

## 1. Architectural Positioning & Proactive Critique

**[Proactive Critique]**:
The current project (`safety-critical-battery-prognostics`) is **not** a standard Node.js/Web front-end application (e.g., React/Vue/Angular). It is fundamentally a **Python-based Machine Learning & Data Science framework**, focusing strictly on battery health management, uncertainty quantification (via PyMC/BNN), and edge-native model deployment (ONNX).
Consequently, generic front-end architectural paradigms such as "Axios, Tailwind, Vuex/Redux" are strictly irrelevant here. The "UI/Frontend" of this project consists exclusively of a data-streaming **Streamlit Dashboard**, tailored for academic reporting and metric visualization—not a high-concurrency client-side web app.

---

## 2. Core Technology Stack Overview

- **Fundamental Architecture**: Python (>= 3.10)
- **Frontend / UI Rendering**: Streamlit (`src/ui/dashboard.py`)
- **UI Components & Charting**: Plotly (`plotly.graph_objects`)
- **State Management**: Executed natively via Streamlit's `Session State` and `@st.cache_data` for computational caching and reactive repainting.
- **Build Systems & Dependencies**:
  - Legacy parsing: `requirements.txt`
  - Modernized configuration and strict linting: `pyproject.toml` (integrates Setuptools, Ruff, Mypy typing constraints).
- **Algorithmic Ecosystem**: PyTorch (Deep Learning), PyMC (Probabilistic Programming / UQ), Scikit-learn, Numpy, Pandas.
- **Inference & Edge Exportation**: ONNX (`onnxruntime`)

---

## 3. Top-Level Hierarchy Abstraction

The repository is rigorously separated by data pipelines, model training, safety-decision engines, and visual rendering:

```text
safety-critical-battery-prognostics/
├── main.py                     # Global Pipeline orchestration and initialization
├── pyproject.toml              # Build settings, lint configurations, and rigid typing bounds
├── src/                        # Core algorithmic dataship and feature code
│   ├── ui/                     # [Frontend] Streamlit visualization dashboards
│   ├── data/                   # [Data] Ingestion bus (Loader, Cleaner, Validator)
│   ├── features/               # [Feature] Temporal engineering constructs
│   ├── physics/                # [Physics] Prior physical mechanisms (Equivalent Circuits, Degradation equations)
│   ├── models/                 # [Model] Neural Architectures (TCN, PINN, BNN, Chronos)
│   ├── training/               # [Training] Loop encapsulation, optimizers, dynamic loss scheduling
│   ├── evaluation/             # [Eval] Performance benchmarking, OOD generalization metrics
│   ├── safety/                 # [Engine] Decision-making limits and circuit breaker logic
│   ├── uncertainty/            # [Uncertainty] Aleatoric/Epistemic limits and conformal calibration
│   ├── deployment/             # [Edge] Model quantization (INT8) and ONNX translation workflows
│   └── utils/                  # [Utils] Generic tooling (logging, YAML parsers)
├── scripts/                    # Offline evaluation, exportation, and analytical utilities
├── docs/                       # Accumulated documentation and architectural hand-offs
├── configs/                    # Immutable YAML structuring model hyperparams and pipeline breakers
├── tests/                      # Pytest entrance for unit and strict integration tests
└── results/                    # Emitted artifacts, graphical reports, and cached tensor logs
```

---

## 4. Subsystem Conventions & Rules

### 4.1 Data Ingestion & Traversal
- **No REST/Axios Networking**: As a pure mathematical computing pipeline, there is zero reliance on external microservices HTTP APIs. The data stream is instantiated directly by `src.data.unified_loader.UnifiedDataLoader` via high-speed NVMe storage blocks located in `data/battery_data`.
- **Preemptive Data Integrity Validations**: Before engaging feature models, tensors must cross the `src.data.validator.DataValidator`. Any anomalous or Out-Of-Bounds geometries trigger immediate pipeline halting via assertions.

### 4.2 State Management
- **Hyperparameter & Static State**: Global configuration structures exist entirely inside `configs/`, strictly decoupling programmatic logic from operational parameters.
- **Reactive UI Flow**: Visualizations abandon Redux/Vuex entirely. Using Streamlit's unidirectional flow, computational burdens are constrained by `@st.cache_data`. Slider filters and threshold interactions (e.g., `selected_bat`, `rul_critical`) bind automatically to session variables, seamlessly driving top-down DOM repainting.

### 4.3 UI Styles and Layouts
- **Framework Immanent Primacy**: External pre-processors (Tailwind/SCSS) are forbidden. The grid-layout solely relies on native Streamlit columns (`st.columns`) and injected safe HTML/CSS strings if precision color manipulation is exceptionally mandated (e.g., GREEN/YELLOW/RED ISO cards).
- **Interface Modularization**: Visual rendering relies heavily upon Plotly. All pseudo-frontend logic is quarantined within `src/ui/dashboard.py`, ensuring a strictly hermetic barrier between foundational algorithms and the presentation layer.

### 4.4 Technical Debt & Edge Considerations
1. **Hidden VRAM/RAM Memory Leakages**: Due to Streamlit's structural execution mapping (reloading scripts upon every UX mutation), caching massive Pandas DataFrames or retaining large standard PyTorch network weights inside execution contexts is highly prone to catastrophic memory ballooning.
   *Mitigation*: Implement absolute `max_entries` limits on all `@st.cache_data` decorators, and actively employ `torch.cuda.empty_cache()` inside dense inference loops.
2. **Synchronous Execution Blockages**: The UI currently functions in a synchronous, blocking paradigm. Initiating heavy computation via `SafetyDecisionEngine` or massive OOD/PyMC MCMC sampling processes guarantees a frozen UI thread (UI Blocking). For future SaaS commercializations, consider shifting these heavy loads to a distributed asynchronous queue (e.g., Celery dispatch to A40 clusters), reconstructing the top-level REST with FastAPI + Next.js.

---

## 5. Non-Negotiable Development Stances

- **Absolute Separation of Concerns**:
  If building new visualization vectors, code **MUST ONLY** live within `src/ui/`. It is categorically prohibited to leak rendering abstractions (`st.write`, `matplotlib.pyplot.show`) into the algorithmic core (`src/models/`, `src/physics/`).
- **Debugging Boot Protocol**:
  Launch the local investigative dashboard strictly via: `streamlit run src/ui/dashboard.py`.
- **Zero-Tolerance Quality Gates**:
  CI/CD procedures hinge rigidly on `ruff` and `mypy` strict modes. All modified/new classes, tensors, and methods require explicit and sound **Type Hints**. It is strictly mandatory that all modifications cross local `pytest` assertions with 100% adherence and zero internal runtime warnings. Inferential, lazy logic is flatly rejected.
