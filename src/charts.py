"""Static charts for the README."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BLUE, ORANGE, INK, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.8, "axes.axisbelow": True})

# 1. Model comparison
res = pd.read_csv("outputs/model_comparison.csv", index_col=0)
fig, ax = plt.subplots(figsize=(8, 3.6))
labels = [i.split(". ", 1)[1] for i in res.index][::-1]
vals = res.auc.values[::-1]
colors = [BLUE if "defense" in l else "#86b6ef" for l in labels]
ax.barh(labels, vals - 0.5, left=0.5, color=colors, height=0.6)
for y, v in enumerate(vals):
    ax.text(v + 0.001, y, f"{v:.3f}", va="center", color=INK2, fontsize=10)
ax.set_xlim(0.5, 0.72); ax.set_xlabel("Out-of-sample AUC, 2024-25 test seasons (0.5 = coin flip)")
ax.grid(axis="y", visible=False)
ax.set_title("Adding defensive alignment does not improve accuracy", loc="left", fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig("outputs/model_comparison.png", dpi=160); plt.close(fig)

# 2. Calibration
t = pd.read_csv("outputs/test_predictions.csv")
t["bin"] = pd.qcut(t.p_convert, 10, labels=False)
cal = t.groupby("bin").agg(pred=("p_convert", "mean"), actual=("converted", "mean"), n=("converted", "size"))
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.plot([0.15, 0.95], [0.15, 0.95], color=INK2, lw=1, ls="--")
ax.text(0.80, 0.86, "perfect", color=INK2, fontsize=9, rotation=42)
ax.plot(cal.pred, cal.actual, color=BLUE, lw=2, marker="o", ms=8, mec=SURF, mew=2)
ax.set_xlim(0.15, 0.95); ax.set_ylim(0.15, 0.95)
ax.set_xlabel("Predicted conversion probability"); ax.set_ylabel("Actual conversion rate")
ax.set_title("Calibration on 2024-25 (deciles)", loc="left", fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig("outputs/calibration.png", dpi=160); plt.close(fig)

# 3. Short yardage: conversion by box count relative to blockers, run vs pass
d = pd.read_parquet("data/fourth_downs.parquet")
s = d[d.ydstogo <= 2].copy()
s["bmb"] = s.box_minus_blockers.clip(-2, 1)
g = s.groupby(["is_pass", "bmb"]).converted.agg(["mean", "size"]).reset_index()
fig, ax = plt.subplots(figsize=(7, 4.2))
for ip, col, name in [(0, BLUE, "Run"), (1, ORANGE, "Pass")]:
    r = g[g.is_pass == ip]
    ax.plot(r.bmb, r["mean"], color=col, lw=2, marker="o", ms=8, mec=SURF, mew=2)
    ax.text(r.bmb.iloc[-1] + 0.08, r["mean"].iloc[-1], name, color=INK, va="center", fontsize=10)
ax.set_xticks([-2, -1, 0, 1]); ax.set_xticklabels(["-2 or fewer", "-1", "0", "+1 or more"])
ax.set_xlim(-2.3, 1.5); ax.set_ylim(0.4, 0.85)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_xlabel("Defenders in box minus offensive blockers (OL + TE + RB)")
ax.set_ylabel("Conversion rate")
ax.set_title("4th & 1-2: runs convert ~70% no matter the box count", loc="left", fontsize=12, fontweight="bold")
ax.legend(handles=[plt.Line2D([], [], color=BLUE, lw=2, marker="o", label="Run"),
                   plt.Line2D([], [], color=ORANGE, lw=2, marker="o", label="Pass")],
          frameon=False, loc="lower left")
fig.tight_layout(); fig.savefig("outputs/short_yardage_box.png", dpi=160); plt.close(fig)

# 4. SHAP importance (top 12)
imp = pd.read_csv("outputs/shap_importance.csv", index_col=0).mean_abs_shap.head(12)[::-1]
DEF = {"defenders_in_box", "def_DL", "def_DB", "box_minus_blockers"}
fig, ax = plt.subplots(figsize=(7, 4.6))
ax.barh(imp.index, imp.values, color=[BLUE if i in DEF else "#b7d3f6" for i in imp.index], height=0.6)
ax.grid(axis="y", visible=False)
ax.set_xlabel("Mean |SHAP value| (log-odds)")
ax.set_title("What drives the prediction (dark = defensive features)", loc="left", fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig("outputs/feature_importance.png", dpi=160); plt.close(fig)
print(cal.round(3)); print(g)
