# tests/conftest.py
"""
Pytest configuration and shared fixtures.
"""

import pytest
import sys
from pathlib import Path

# Add src directory to Python path for all tests
@pytest.fixture(autouse=True)
def setup_path():
    """Automatically add src to path for all tests."""
    src_path = Path(__file__).parent.parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


@pytest.fixture
def project_root() -> Path:
    """Return project root directory."""
    return Path(__file__).parent.parent
