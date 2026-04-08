## Summary

- What changed?
- Why was this change needed?

## Validation

- [ ] `python -m pytest -q`
- [ ] Relevant targeted checks for the touched area were run

## Documentation

- [ ] README updated if user-facing behavior changed
- [ ] `docs/comprehensive_experimental_results.md` updated if experiment outputs changed
- [ ] `docs/claim_evidence_matrix.md` updated if any claim wording changed

## Evidence Boundary Check

- [ ] Same-cell results are not described as cross-cell results
- [ ] Synthetic, same-cell, LOGO, and zero-shot results remain clearly separated
- [ ] Any new headline claim is backed by a script and output artifact in `robustness_results/`

## Notes for Reviewers

- Anything risky, incomplete, or worth extra attention?
