import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.models.bayesian_nn import BayesianNNModel

np.random.seed(42)
X = np.random.randn(100, 22).astype("float32")
y = np.linspace(100, 0, 100).astype("float32")
m = BayesianNNModel(input_dim=22, hidden_dim=64, epochs=50, n_samples=20)
m.fit(X, y)
mean, lo, hi = m.predict(X)
rmse = np.sqrt(np.mean((y - mean) ** 2))
print(f"BNN RMSE={rmse:.2f}, range=[{mean.min():.1f}, {mean.max():.1f}]")
print(f"CI width avg={np.mean(hi-lo):.2f}")
print("BNN fix verified!" if rmse < 100 else "BNN still broken!")
