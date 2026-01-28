# Contributing to Spatial Bayesian vs Deep Learning

Thank you for your interest in contributing to this research project!

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/PanZhiChao666/spatial-bayes-vs-deep.git
   cd spatial-bayes-vs-deep
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_schema.py -v
```

## Code Style

- Use type annotations for function parameters and return values
- Include docstrings for all public functions (Google style)
- Follow PEP 8 guidelines
- Maximum line length: 100 characters

## Project Structure

```
├── config/           # Configuration files (schema.yaml)
├── data/             # Dataset files (raw, processed)
├── notebooks/        # Jupyter notebooks for exploration
├── results/          # Model outputs, figures, metrics
├── src/              # Source code
│   ├── utils/        # Utility modules
│   └── *.py          # Training and evaluation scripts
└── tests/            # Unit tests
```

## Reporting Issues

Please include:
- Python version
- Package versions (`pip freeze`)
- Full error traceback
- Steps to reproduce
