# tests/test_schema.py
"""
Unit tests for schema configuration loading and validation.
"""

import pytest
import yaml
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestSchemaLoading:
    """Test suite for schema.yaml loading and validation."""
    
    @pytest.fixture
    def schema_path(self) -> Path:
        """Return path to schema configuration file."""
        return Path(__file__).parent.parent / "config" / "schema.yaml"
    
    @pytest.fixture
    def schema(self, schema_path: Path) -> dict:
        """Load and return schema configuration."""
        with open(schema_path, "r") as f:
            return yaml.safe_load(f)
    
    def test_schema_file_exists(self, schema_path: Path) -> None:
        """Verify schema.yaml file exists."""
        assert schema_path.exists(), f"Schema file not found: {schema_path}"
    
    def test_schema_has_required_keys(self, schema: dict) -> None:
        """Verify schema contains all required top-level keys."""
        required_keys = ["dataset", "target", "features", "bayesian"]
        for key in required_keys:
            assert key in schema, f"Missing required key: {key}"
    
    def test_dataset_path_exists(self, schema: dict) -> None:
        """Verify the dataset path in schema points to existing file."""
        data_path = Path(schema["dataset"]["path"])
        # Use relative path from project root
        project_root = Path(__file__).parent.parent
        full_path = project_root / data_path
        assert full_path.exists(), f"Dataset not found: {full_path}"
    
    def test_target_is_string(self, schema: dict) -> None:
        """Verify target name is a string."""
        assert isinstance(schema["target"]["name"], str)
        assert len(schema["target"]["name"]) > 0
    
    def test_features_numeric_is_list(self, schema: dict) -> None:
        """Verify numeric features is a non-empty list."""
        numeric_features = schema["features"]["numeric"]
        assert isinstance(numeric_features, list)
        assert len(numeric_features) > 0
    
    def test_bayesian_hierarchical_config(self, schema: dict) -> None:
        """Verify Bayesian hierarchical model configuration is valid."""
        hier = schema["bayesian"]["hierarchical"]
        assert "slope" in hier, "Missing 'slope' in hierarchical config"
        assert "group" in hier, "Missing 'group' in hierarchical config"
        
        # Slope can be string or list
        slope = hier["slope"]
        assert isinstance(slope, (str, list))
        
        # Group must be string
        assert isinstance(hier["group"], str)


class TestSchemaConsistency:
    """Test schema internal consistency."""
    
    @pytest.fixture
    def schema(self) -> dict:
        schema_path = Path(__file__).parent.parent / "config" / "schema.yaml"
        with open(schema_path, "r") as f:
            return yaml.safe_load(f)
    
    def test_slope_features_in_numeric(self, schema: dict) -> None:
        """Verify all slope features are in numeric features list."""
        numeric = schema["features"]["numeric"]
        slope = schema["bayesian"]["hierarchical"]["slope"]
        
        if isinstance(slope, str):
            slope = [slope]
        
        for feat in slope:
            assert feat in numeric, f"Slope feature '{feat}' not in numeric features"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
