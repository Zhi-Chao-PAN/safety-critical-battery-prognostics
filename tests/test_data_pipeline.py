# tests/test_data_pipeline.py
"""
Unit tests for data loading and preprocessing pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import yaml


class TestDataLoading:
    """Test suite for data loading functionality."""
    
    @pytest.fixture
    def project_root(self) -> Path:
        """Return project root directory."""
        return Path(__file__).parent.parent
    
    @pytest.fixture
    def schema(self, project_root: Path) -> dict:
        """Load schema configuration."""
        schema_path = project_root / "config" / "schema.yaml"
        with open(schema_path, "r") as f:
            return yaml.safe_load(f)
    
    @pytest.fixture
    def df(self, project_root: Path, schema: dict) -> pd.DataFrame:
        """Load the main dataset."""
        data_path = project_root / schema["dataset"]["path"]
        return pd.read_csv(data_path)
    
    def test_data_loads_successfully(self, df: pd.DataFrame) -> None:
        """Verify dataset loads without errors."""
        assert df is not None
        assert len(df) > 0
    
    def test_target_column_exists(self, df: pd.DataFrame, schema: dict) -> None:
        """Verify target column exists in dataset."""
        target = schema["target"]["name"]
        assert target in df.columns, f"Target '{target}' not in columns"
    
    def test_numeric_features_exist(self, df: pd.DataFrame, schema: dict) -> None:
        """Verify all numeric features exist in dataset."""
        for feat in schema["features"]["numeric"]:
            assert feat in df.columns, f"Feature '{feat}' not in columns"
    
    def test_group_column_exists(self, df: pd.DataFrame, schema: dict) -> None:
        """Verify group column for hierarchical model exists."""
        group = schema["bayesian"]["hierarchical"]["group"]
        assert group in df.columns, f"Group column '{group}' not in columns"
    
    def test_no_missing_in_target(self, df: pd.DataFrame, schema: dict) -> None:
        """Verify no missing values in target column."""
        target = schema["target"]["name"]
        assert df[target].isna().sum() == 0, "Target contains missing values"
    
    def test_numeric_features_dtype(self, df: pd.DataFrame, schema: dict) -> None:
        """Verify numeric features have numeric dtypes."""
        for feat in schema["features"]["numeric"]:
            assert pd.api.types.is_numeric_dtype(df[feat]), \
                f"Feature '{feat}' is not numeric type"


class TestDataPreprocessing:
    """Test data preprocessing functions."""
    
    def test_standardization(self) -> None:
        """Test that standardization produces zero mean and unit std."""
        # Create sample data
        data = np.random.randn(100) * 5 + 10  # mean=10, std=5
        
        # Standardize
        standardized = (data - data.mean()) / data.std()
        
        # Check properties
        assert np.abs(standardized.mean()) < 1e-10, "Mean should be ~0"
        assert np.abs(standardized.std() - 1) < 1e-10, "Std should be ~1"
    
    def test_standardization_preserves_shape(self) -> None:
        """Test that standardization preserves data shape."""
        data = np.random.randn(50, 3)
        standardized = (data - data.mean(axis=0)) / data.std(axis=0)
        
        assert data.shape == standardized.shape


class TestDataSplitting:
    """Test data splitting for cross-validation."""
    
    @pytest.fixture
    def sample_data(self) -> tuple:
        """Create sample data for testing."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 5)
        y = np.random.randn(n)
        groups = np.random.choice(["A", "B", "C", "D"], size=n)
        return X, y, groups
    
    def test_stratified_split_preserves_groups(self, sample_data: tuple) -> None:
        """Test that stratified split maintains group representation."""
        from sklearn.model_selection import StratifiedKFold
        
        X, y, groups = sample_data
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for train_idx, test_idx in skf.split(X, groups):
            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])
            
            # Each fold should have representation from multiple groups
            assert len(train_groups) >= 2, "Training set lacks group diversity"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
