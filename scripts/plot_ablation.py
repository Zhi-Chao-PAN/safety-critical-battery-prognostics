"""Generate ablation study figures for paper."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({'font.size': 11, 'figure.dpi': 150, 'savefig.bbox': 'tight'})
ROOT = Path(__file__).parent.parent
FIG = ROOT / "figures"
ABL = ROOT / "results" / "ablation"

# Fig 4: Architecture comparison
df = pd.read_csv(ABL / "architecture_comparison.csv")
summary = df.groupby("model").agg({"RMSE": ["mean", "std"], "CRPS": ["mean", "std"], "PICP": "mean"}).round(2)
summary.columns = ["RMSE_mean", "RMSE_std", "CRPS_mean", "CRPS_std", "PICP"]
summary = summary.sort_values("RMSE_mean")

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
models = summary.index.tolist()
x = np.arange(len(models))

axes[0].barh(x, summary["RMSE_mean"], xerr=summary["RMSE_std"], color="steelblue", alpha=0.8)
axes[0].set_yticks(x); axes[0].set_yticklabels(models)
axes[0].set_xlabel("RMSE"); axes[0].set_title("(a) RMSE by Architecture")

axes[1].barh(x, summary["CRPS_mean"], xerr=summary["CRPS_std"], color="coral", alpha=0.8)
axes[1].set_yticks(x); axes[1].set_yticklabels(models)
axes[1].set_xlabel("CRPS"); axes[1].set_title("(b) CRPS by Architecture")

axes[2].barh(x, summary["PICP"] * 100, color="seagreen", alpha=0.8)
axes[2].set_yticks(x); axes[2].set_yticklabels(models)
axes[2].set_xlabel("PICP (%)"); axes[2].set_title("(c) Coverage by Architecture")
axes[2].axvline(95, color="red", ls="--", lw=1, label="95% target")
axes[2].legend()

plt.tight_layout()
plt.savefig(FIG / "fig04_ablation_architecture.png")
plt.close()
print("Fig 4 saved.")

# Fig 5: Sequence length
df = pd.read_csv(ABL / "sequence_length.csv")
summary = df.groupby("model").agg({"RMSE": "mean", "CRPS": "mean"}).round(2)
seq_lens = [5, 10, 20, 30, 50]
rmses = [summary.loc[f"LSTM_seq{s}", "RMSE"] for s in seq_lens]
crps = [summary.loc[f"LSTM_seq{s}", "CRPS"] for s in seq_lens]

fig, ax1 = plt.subplots(figsize=(7, 4.5))
ax1.plot(seq_lens, rmses, "o-", color="steelblue", lw=2, label="RMSE")
ax1.set_xlabel("Sequence Length"); ax1.set_ylabel("RMSE", color="steelblue")
ax2 = ax1.twinx()
ax2.plot(seq_lens, crps, "s--", color="coral", lw=2, label="CRPS")
ax2.set_ylabel("CRPS", color="coral")
ax1.set_title("Effect of Sequence Length on LSTM Performance")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
plt.tight_layout()
plt.savefig(FIG / "fig05_ablation_seqlen.png")
plt.close()
print("Fig 5 saved.")

# Fig 6: Hidden dimension
df = pd.read_csv(ABL / "hidden_dimension.csv")
summary = df.groupby("model").agg({"RMSE": "mean", "train_time_s": "mean"}).round(2)
dims = [16, 32, 64, 128]
rmses = [summary.loc[f"LSTM_h{d}", "RMSE"] for d in dims]
times = [summary.loc[f"LSTM_h{d}", "train_time_s"] for d in dims]

fig, ax1 = plt.subplots(figsize=(7, 4.5))
ax1.plot(dims, rmses, "o-", color="steelblue", lw=2, label="RMSE")
ax1.set_xlabel("Hidden Dimension"); ax1.set_ylabel("RMSE", color="steelblue")
ax2 = ax1.twinx()
ax2.plot(dims, times, "s--", color="orange", lw=2, label="Train Time (s)")
ax2.set_ylabel("Train Time (s)", color="orange")
ax1.set_title("Effect of Hidden Dimension on LSTM")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
plt.tight_layout()
plt.savefig(FIG / "fig06_ablation_hidden.png")
plt.close()
print("Fig 6 saved.")

print("All ablation figures done.")
