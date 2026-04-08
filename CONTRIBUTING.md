# Contributing

Thanks for your interest in improving this repository.

This project mixes research code, evaluation scripts, and publication-facing documentation, so a good contribution is not just "code that runs". It should also preserve the repository's evidence boundaries and reproducibility.

For the original extended guide, see [CONTRIBUTING_zh.md](CONTRIBUTING_zh.md).

## Before You Start

- Read [README.md](README.md) for the current repository scope.
- Read [docs/claim_evidence_matrix.md](docs/claim_evidence_matrix.md) before changing any experiment summary, benchmark headline, or safety claim.
- Keep active documentation and executable scripts aligned. If you change a protocol, update the docs that describe it.

## Local Setup

```bash
git clone https://github.com/Zhi-Chao-PAN/safety-critical-battery-prognostics.git
cd safety-critical-battery-prognostics
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

## Validation Before Opening a PR

Run the checks that match your change scope. For most changes, start with:

```bash
python -m pytest -q
```

If you touched typing-sensitive modules, also run:

```bash
python -m mypy src
```

If you touched formatting or imports, also run:

```bash
python -m ruff check src tests
```

## Evidence Rules for Research Changes

- Do not describe same-cell validation as cross-cell validation.
- Do not promote a protocol to a headline claim unless the corresponding script has been executed and the output files are present in `robustness_results/`.
- Keep synthetic, same-cell, LOGO, and zero-shot results clearly separated.
- If a change affects model fairness or post-processing, document whether the same transformation is applied to every baseline.
- When in doubt, update [docs/comprehensive_experimental_results.md](docs/comprehensive_experimental_results.md) and [docs/claim_evidence_matrix.md](docs/claim_evidence_matrix.md) together.

## Pull Request Checklist

- The relevant tests pass locally.
- New behavior is covered by tests when practical.
- README and docs are updated if the user-facing behavior changed.
- Any new result claim cites the script and output artifact that support it.
- Temporary files, local caches, and scratch outputs are not included in the PR.

## Community

Please follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) in issues, pull requests, and discussions.
