# tests/conftest.py
"""
Pytest configuration and shared fixtures.
"""

import shutil
import sys
import uuid
from pathlib import Path

import pytest


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


@pytest.fixture
def workspace_tmp_path(project_root: Path) -> Path:
    """Create a writable temp directory inside the repository workspace."""
    base_dir = project_root / "test_artifacts" / "pytest_local"
    base_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = base_dir / f"run-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
