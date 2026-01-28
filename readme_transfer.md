# README_TRANSFER.md

## Purpose
This document is a technical handover for AI assistants and new contributors. It captures current project state, architecture, decisions, and the prioritized roadmap so the project can be resumed without context loss.

---

## Project
**Repository:** ml-research-housing-price  
**Domain:** Research-grade ML + Bayesian hierarchical modeling for housing prices

**Goal:** Build a schema-driven, reproducible research pipeline supporting pooled and hierarchical Bayesian regression with spatial grouping, model comparison, and experiment tracking.

This is not an application/demo repo. It is a research engineering project emphasizing:
- Schema-driven modeling
- Reproducibility
- Hierarchical inference
- Spatial grouping
- Model comparison (WAIC/LOO)

---

## Data

**Path:**
```
data/processed/housing_with_spatial_clusters.csv
```

**Columns:**
```
median_income
house_age
avg_rooms
avg_bedrooms
population
latitude
longitude
price
spatial_cluster
```

**Semantics:**
- Target: `price`
- Core predictor: `median_income`
- Grouping variable (hierarchical): `spatial_cluster`

---

## Schema System (Core Architecture)

All models are driven by schema. Hard-coded column names are disallowed.

**Loader:**
```
src/utils/schema.py
```

**Current effective schema:**
```
dataset.path = data/processed/housing_with_spatial_clusters.csv

target.name = price

group.name = spatial_cluster

features.numeric = [
  median_income,
  house_age,
  avg_rooms,
  avg_bedrooms,
  population,
  latitude,
  longitude
]

modeling.standardize = [median_income, price]

bayesian.pooled.slope = median_income

bayesian.hierarchical.slope = median_income
bayesian.hierarchical.group = spatial_cluster
```

**Design principle:**
- Data ↔ Model decoupling
- Dataset swaps without code changes
- Single source of truth for column semantics

---

## Bayesian Models

### 1. Pooled Bayesian Regression

**File:**
```
src/train_bayes_pooled.py
```

**Model:**
```
y = α + β * x + ε
```

- Schema-driven
- Standardization via schema
- Stable sampling

**Output:**
```
results/bayes_pooled/trace_pooled.nc
```

**Status:**
- Converged
- R-hat ≈ 1
- ESS high
- No divergences

---

### 2. Hierarchical Bayesian Regression

**File:**
```
src/train_bayes_hierarchical.py
```

**Model:**
```
y = α_group[g] + β_group[g] * x + ε

α_group ~ Normal(μ_α, σ_α)
β_group ~ Normal(μ_β, σ_β)
```

- Group: spatial_cluster
- Slope: median_income

**Status:**
- Structure correct
- Sampling quality suboptimal

**Observed issues:**
- Divergences
- Low ESS
- R-hat > 1.01
- Funnel geometry

**Cause:**
- Weak group-level variation
- Hierarchical variance near zero
- Typical hierarchical funnel

**Required fixes (not yet implemented):**
- Non-centered parameterization
- Stronger priors on σ_α, σ_β
- Higher target_accept
- Reparameterization

---

## Model Comparison (Incomplete)

**File:**
```
src/compare_bayes_models.py
```

**Current issue:**
- ArviZ error: log_likelihood missing

**Cause:**
- PyMC sampling did not store log_likelihood

**Required changes:**
- Enable log_likelihood in pm.sample
- Or run pm.compute_log_likelihood(idata)

**Goal:**
- WAIC
- LOO
- Formal pooled vs hierarchical comparison

---

## Engineering Maturity

| Component | Status |
|------------|--------|
| Schema-driven system | Complete |
| CSV–Model decoupling | Complete |
| Pooled Bayesian | Stable |
| Hierarchical Bayesian | Structurally correct, needs optimization |
| Model comparison | Not implemented |
| Experiment tracking | Partial |
| Config system | Partial (schema only) |
| README / docs | Incomplete |

---

## Current Project Phase

**Research Engineering Phase**

Primary focus is no longer "write models" but:
- Stabilize hierarchical inference geometry
- Build model evaluation and comparison
- Improve reproducibility and experiment structure
- Document research assumptions

---

## Priority Roadmap

### Tier 1 — Inference Stability
1. Implement non-centered hierarchical parameterization
2. Tighten priors on σ_α, σ_β
3. Increase target_accept
4. Eliminate divergences
5. ESS > 1000
6. R-hat < 1.01

### Tier 2 — Model Comparison
7. Add log_likelihood recording
8. Implement WAIC / LOO
9. Complete compare_bayes_models.py
10. Formal pooled vs hierarchical evaluation

### Tier 3 — Engineering
11. experiments/ configuration
12. training config separation
13. structured logging
14. versioned results

### Tier 4 — Research Documentation
15. README methodology
16. model assumptions
17. Bayesian structure explanation
18. spatial modeling rationale

---

## Project Philosophy

This is a research-grade Bayesian modeling pipeline, not a production system.

Primary values:
- Correct probabilistic structure
- Clear modeling assumptions
- Reproducibility
- Extensibility
- Research interpretability

Accuracy is secondary to:
- Model correctness
- Inference quality
- Structural clarity

---

## Technical Stack

- Python
- pandas / numpy
- PyMC
- ArviZ
- Schema-driven modeling
- Hierarchical Bayesian inference
- Spatial grouping

---

## Handover Summary (for AI or new contributor)

This project already has:
- A schema-driven architecture
- Stable pooled Bayesian regression
- Structurally correct hierarchical Bayesian regression

Current work is focused on:
- Fixing hierarchical sampling geometry
- Implementing Bayesian model comparison
- Engineering the research pipeline

Do not treat this as a toy ML repo. It is structured as a research codebase intended to support statistical modeling and inference quality.

