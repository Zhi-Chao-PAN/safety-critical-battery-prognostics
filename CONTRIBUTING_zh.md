# Contributing Guide

👋 Thank you for your interest in contributing to the **Safety-Critical Battery Prognostics** project! We welcome a wide variety of contributions from the academic and industrial communities.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Environment Setup](#environment-setup)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Commit Message Conventions](#commit-message-conventions)
- [Pull Request Guidelines](#pull-request-guidelines)

---

## Code of Conduct

### Our Pledge
In the interest of fostering an open and welcoming environment, we pledge to:
- Respect diverse viewpoints and experiences.
- Accept constructive criticism gracefully.
- Focus on what is best for the community.
- Show empathy towards other community members.

### Unacceptable Behavior
- Use of sexualized language or imagery.
- Trolling, insulting/derogatory comments, and personal or political attacks.
- Public or private harassment.
- Publishing others' private information without explicit permission.
- Other conduct which could reasonably be considered inappropriate in a professional setting.

---

## How Can I Contribute?

### 🐛 Reporting Bugs
Found an issue? Please submit an Issue containing:
- A clear, descriptive title.
- Exact steps to reproduce the behavior.
- Expected behavior vs. Actual behavior.
- Environment details (Python version, dependency versions, OS).
- Stack traces or error logs.

### ✨ Proposing Features
Have a great idea? Submit a Feature Request outlining:
- The context and motivation for the feature.
- Proposed implementation details.
- Potential alternatives considered.

### 📝 Improving Documentation
Documentation can always be enhanced! You can:
- Fix typos or grammatical errors.
- Add clarifications and real-world examples.
- Translate documentation.
- Improve code-level docstrings.

### 💻 Code Contributions
We welcome PRs for:
- Bug fixes.
- New physics solvers or algorithmic features.
- Performance optimization (e.g., TensorRT integration).
- Codebase refactoring.
- Additional test coverage.

---

## Environment Setup

### 1. Fork and Clone
```bash
# Fork the repository on GitHub, then clone your fork:
git clone https://github.com/YOUR_USERNAME/safety-critical-battery-prognostics.git
cd safety-critical-battery-prognostics

# Add the upstream remote
git remote add upstream https://github.com/Zhi-Chao-PAN/safety-critical-battery-prognostics.git
```

### 2. Virtual Environment
```bash
# Create a venv
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (Linux/Mac)
source venv/bin/activate
```

### 3. Install Dependencies
```bash
# Core dependencies
pip install -r requirements.txt

# Development dependencies (testing, linting)
pip install -e ".[dev]"
```

### 4. Running Tests
```bash
# Execute the full suite
pytest

# Execute specific tests
pytest tests/unit/test_pinn_model.py -v

# Generate coverage reports
pytest --cov=src --cov-report=html
```

---

## Development Workflow

### 1. Sync with Upstream
```bash
git checkout main
git pull upstream main
```

### 2. Create a Feature Branch
```bash
# Use descriptive branch names
git checkout -b feature/your-feature-name
git checkout -b fix/issue-description
```

### 3. Develop
- Write clean code.
- Add or update corresponding unit/integration tests.
- Update documentation as required.

### 4. Commit Changes
```bash
git add .
git commit -m "feat: implement semi-implicit euler solver"
```

### 5. Push Branch
```bash
git push origin feature/your-feature-name
```

### 6. Create a Pull Request
- Open a PR against the `main` branch.
- Fill out the provided PR template.
- Link connected Issues.
- Request code review.

---

## Coding Standards

### Python Style
- Adhere strictly to [PEP 8](https://peps.python.org/pep-0008/).
- Use strong **Type Hints** `typing`.
- Maximum line length: 100 characters.
- Use **Google-style Docstrings** for all classes and functions.

### Example Code
```python
from typing import Optional
import torch

def predict_rul(
    capacity_sequence: torch.Tensor,
    context_length: int = 100,
    device: Optional[str] = None
) -> torch.Tensor:
    """Predicts battery Remaining Useful Life (RUL).

    Args:
        capacity_sequence: Tensorial sequence of capacities, shape (batch_size, seq_len).
        context_length: Historical context window size.
        device: Target computation device (auto-selected if None).

    Returns:
        Predicted RUL tensor, shape (batch_size,).

    Examples:
        >>> import torch
        >>> seq = torch.randn(1, 200)
        >>> rul = predict_rul(seq)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # Implementation logic...
    pass
```

### Testing Standards
- Write unit tests for all new functionalities.
- Place tests in the `tests/` directory mirroring the `src/` structure.
- Use the `pytest` framework.
- Naming convention: `test_<module_name>.py`.

---

## Commit Message Conventions

This project strictly follows [Conventional Commits](https://www.conventionalcommits.org/):

```text
<type>(<scope>): <subject>

<body>

<footer>
```

### Allowed Types
- `feat`: A new feature or algorithmic enhancement.
- `fix`: A bug fix.
- `docs`: Documentation only changes.
- `style`: Changes that do not affect runtime meaning (white-space, formatting).
- `refactor`: Code change that neither fixes a bug nor adds a feature.
- `perf`: A code change that improves performance.
- `test`: Adding missing tests or correcting existing tests.
- `chore`: Changes to the build process or auxiliary tools.

### Example
```text
feat(physics): implement backward euler stability scheme

- Replaced explicit forward euler with semi-implicit integration.
- Added Jacobian assertions for gradient checks.
- Updated coverage in tests/unit/test_spm.py.

Closes #123
```

---

## Pull Request Guidelines

### Pre-PR Checklist
- [ ] Code passes all linting (`ruff`) and type-checking (`mypy`) CI steps.
- [ ] Required tests have been added and pass locally.
- [ ] Relevant documentation has been updated.
- [ ] Code conforms to project SOTA guidelines.
- [ ] Commit messages follow the formatting rules.
- [ ] Branch is successfully rebased with upstream `main`.

### Code Review Process
- Maintainers will review the PR as swiftly as possible.
- Please be prepared to iterate based on architectural feedback.
- Maintain respectful communication.

---

## Need Help?

If you encounter issues during contribution:
1. Review the documentation and FAQ.
2. Search existing Issues and Discussions.
3. Open a new Discussion thread.
4. Leave a comment on related Issues.

---

## License

By contributing code, you agree that your contributions will be licensed under the project's MIT License.
