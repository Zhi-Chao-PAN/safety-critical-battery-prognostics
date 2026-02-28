import re

results = {}
with open("logs/run.log") as f:
    current_model = None
    current_seed = None
    for line in f:
        m = re.search(r"=== (\w+) \| seed=(\d+) ===", line)
        if m:
            current_model = m.group(1)
            current_seed = int(m.group(2))
        m2 = re.search(r"Fold (\w+): RMSE=([\d.]+), CRPS=([\d.]+)", line)
        if m2 and current_model:
            key = f"{current_model}_s{current_seed}"
            if key not in results:
                results[key] = []
            results[key].append((float(m2.group(2)), float(m2.group(3))))

# Aggregate by model
from collections import defaultdict
model_agg = defaultdict(list)
for key in sorted(results.keys()):
    model = key.rsplit("_s", 1)[0]
    for rmse, crps in results[key]:
        model_agg[model].append((rmse, crps))

print(f"{'Model':<20} {'Seeds×Folds':>12} {'RMSE_avg':>10} {'RMSE_std':>10} {'CRPS_avg':>10}")
print("-" * 65)
import numpy as np
for model in model_agg:
    vals = model_agg[model]
    rmses = [v[0] for v in vals]
    crpss = [v[1] for v in vals]
    print(f"{model:<20} {len(vals):>12} {np.mean(rmses):>10.2f} {np.std(rmses):>10.2f} {np.mean(crpss):>10.2f}")
