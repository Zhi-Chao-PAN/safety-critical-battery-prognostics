# Claim-Evidence Matrix

This document separates repository claims into three buckets:

- **Verified facts**: directly supported by current scripts, code, and reported outputs.
- **Bounded conclusions**: supported within a specific protocol, but should not be generalized further.
- **Future work / unverified claims**: code paths or ideas that exist, but do not yet have repository-level evidence attached.
- The active README should summarize only the first two buckets and must not promote future-work items into headline results.

## Verified Facts

| Claim | Evidence | Scope / Limit |
|------|----------|---------------|
| The three-layer defense can enforce 0.00% monotonicity violation rate under the reported synthetic-noise setup. | `robustness_test.py`, `scripts/ablation_defense_layers.py`, `docs/comprehensive_experimental_results.md` Sections 7-8 | Synthetic noise protocol only. |
| Same-cell CALCE noise robustness is supported on 6 real cells with identical post-processing for PINN and baselines. | `scripts/validate_real_data.py`, `robustness_results/real_data_validation_report.md`, `README.md` | Same-cell only; not cross-cell evidence. In the current seeded rerun, both PINN and LSTM are at 0.00% VR. |
| Executed LOGO CALCE outputs are present for held-out-cell evaluation. | `scripts/validate_real_data_logo.py`, `robustness_results/real_data_logo_validation.png`, `robustness_results/real_data_logo_validation_report.md` | Confirms the protocol has been run and reported, not that it shows PINN superiority. |
| A seeded multi-corruption real-data stress suite is now attached to the repository. | `scripts/validate_real_data_stress_suite.py`, `robustness_results/real_data_stress_suite_report.md`, `robustness_results/real_data_stress_suite_summary.csv` | Varies corruption seeds and corruption families under fixed training seed 42. |
| PINN is a capacity-space model in the active training/evaluation stack. | `src/models/pinn_model.py`, `src/training/pipeline.py`, `tests/integration/test_integration_pipeline.py` | Applies to the active PINN implementation. |
| Capacity models can be evaluated against RUL through an explicit capacity-to-RUL adapter path. | `src/evaluation/capacity_to_rul.py`, `src/evaluation/target_adapter.py`, `tests/integration/test_benchmark_target_adapter.py` | Requires cycle-aligned trajectory context. |

## Bounded Conclusions

| Claim | Evidence | Why It Is Bounded |
|------|----------|-------------------|
| The real-data evaluation stack transfers from synthetic data to severe sensor corruption, but not as a PINN-only advantage. | `scripts/validate_real_data.py`, `docs/comprehensive_experimental_results.md` Section 9 | Evidence is same-cell noise robustness with shared post-processing; both PINN and LSTM are monotone in the seeded rerun, while PINN has worse RMSE. |
| Zero-shot cross-dataset evaluation is supported by the benchmarking stack. | `src/evaluation/zero_shot_benchmark.py`, `scripts/run_zero_shot_benchmark.py` | Interpretation depends on model semantics; capacity models are adapted to RUL during evaluation rather than trained directly on RUL. |
| The current LOGO cross-cell run supports a bounded safety claim, not a superiority claim. | `robustness_results/real_data_logo_validation_report.md`, `docs/comprehensive_experimental_results.md` Section 9.6 | In the reported held-out-cell run, both PINN and LSTM reach 0.00% VR under the shared post-processing stack, while PINN shows worse RMSE than LSTM. |
| The current multi-seed corruption suite extends that same bounded conclusion to extra corruption families under a fixed training seed. | `robustness_results/real_data_stress_suite_report.md`, `docs/comprehensive_experimental_results.md` Section 9.7 | Both models stay at 0.00% VR across Gaussian, bias-drift, impulse-spike, and missing-segment corruptions; the remaining gap is RMSE, not monotonicity. |

## Future Work / Unverified Claims

| Claim | Current Status | Next Step |
|------|----------------|-----------|
| Real CALCE conclusions will remain stable across different **training** seeds, not just corruption seeds. | Current same-cell/LOGO stress-suite reporting fixes training seed at 42 and varies only corruption seeds/families. | Add training-seed sweeps or nested repeats so the real-data bounds are not tied to one initialization. |
| The same bounded safety behavior holds under broader real sensor degradations such as periodic interference or calibration drift outside the current synthetic corruption families. | The active stress suite covers Gaussian noise, bias drift, impulse spikes, and missing segments only. | Add new corruption protocols and report them separately. |
| Archive-era zero-shot benchmark writeups fully match the current capacity-first semantics. | Historical documents remain preserved with warning banners. | Update or retire archive pages if they are promoted back to active documentation. |
