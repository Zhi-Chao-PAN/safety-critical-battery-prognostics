import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# =========================
# CONFIG
# =========================
DATA_PATH = "data/processed/housing.csv"
TARGET_COL = "price"
DROP_COL = "median_income"

ALPHA = 1.0

# =========================
# Load data
# =========================
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=[TARGET_COL, DROP_COL])
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# Model
# =========================
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=ALPHA))
])

pipeline.fit(X_train, y_train)

# =========================
# Evaluation
# =========================
y_pred = pipeline.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print("\n=== Ridge WITHOUT median_income ===")
print(f"RMSE: {rmse:.4f}")

# =========================
# Save model
# =========================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f"results/models/ridge_no_income_{timestamp}.joblib"

Path("results/models").mkdir(parents=True, exist_ok=True)
joblib.dump(pipeline, model_path)

print(f"Saved model to: {model_path}")
