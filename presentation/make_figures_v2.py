"""Slide figures for the updated presentation (UCI + REFIT corpus, from-scratch model, evaluation), in the same
style as make_figures.py.

Usage (from the repo root, after generating the corpus and running scripts/analyze_eval.py):
    python presentation/make_figures_v2.py
Reads data/energydata_sml_large_train.jsonl, the two training notebooks' logs and paper/results.json.
"""
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(".")
OUT = ROOT / "presentation" / "figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "DejaVu Sans", "font.size": 17, "axes.titlesize": 18, "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 15,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.titlelocation": "left", "lines.linewidth": 2,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)


# 1. Training examples per task, UCI house vs REFIT homes
c = Counter()
for line in open(ROOT / "data" / "energydata_sml_large_train.jsonl"):
    r = json.loads(line)
    c[(r["task"], r["source"])] += 1
tasks = sorted({t for t, _ in c}, key=lambda t: c[(t, "uci")] + c[(t, "refit")])
uci = np.array([c[(t, "uci")] for t in tasks])
refit = np.array([c[(t, "refit")] for t in tasks])
fig, ax = plt.subplots(figsize=(7.4, 6.2))
y = np.arange(len(tasks))
ax.barh(y, uci, color=BLUE, height=0.7, label="UCI house (Belgium)")
ax.barh(y, refit, left=uci, color=ORANGE, height=0.7, label="REFIT homes (UK)")
for i, v in enumerate(uci + refit):
    ax.text(v + 300, i, f"{v:,}", va="center", color=INK, fontsize=14)
ax.set_yticks(y, [t.replace("_", " ") for t in tasks], fontsize=14)
ax.set_xlim(0, (uci + refit).max() * 1.2)
ax.set_xticks([0, 10000, 20000, 30000], ["0", "10k", "20k", "30k"])
ax.grid(axis="y", visible=False)
ax.set_xlabel("Training examples")
ax.legend(frameon=False, loc="lower right")
ax.set_title(f"{(uci + refit).sum():,} training examples")
save(fig, "fig_tasks_large.png")


# 2. From-scratch training: first (UCI-only) vs final (UCI + REFIT) run
def curve(path):
    nb = json.load(open(path))
    text = "".join("".join(o["text"]) for cell in nb["cells"] for o in cell.get("outputs", [])
                   if o.get("output_type") == "stream")
    pts = re.findall(r"step\s+(\d+): train loss ([\d.]+), val loss ([\d.]+)", text)
    return np.array([[int(s), float(t), float(v)] for s, t, v in pts if int(s) > 0])


v1 = curve(ROOT / "notebooks/reference/5_slm_from_scratch_qwen.v1_uci_only.ipynb")
v2 = curve(ROOT / "notebooks/5_slm_from_scratch_qwen.ipynb")
fig, ax = plt.subplots(figsize=(7.4, 4.9))
for d, color, name in ((v1, ORANGE, "First run: UCI only"), (v2, BLUE, "Final run: UCI + REFIT")):
    ax.plot(d[:, 0] / 1000, d[:, 2], color=color, marker="o", ms=4, label=f"{name}, validation")
    ax.plot(d[:, 0] / 1000, d[:, 1], color=color, ls="--", lw=1.5, alpha=0.8, label=f"{name}, training")
ax.set_yscale("log")
ax.set_xlabel("Training steps (thousands)")
ax.set_ylabel("Loss (log scale)")
ax.legend(frameon=False, fontsize=12, loc="upper right")
ax.set_title("More data closed the overfitting gap")
save(fig, "fig_scratch_loss.png")

# 3. Exact match per task, first vs final from-scratch model
res = json.load(open(ROOT / "paper" / "results.json"))
em = res["exact_match"]
keys = sorted(em, key=lambda k: em[k]["v2"])
labels = [f"{k.split('/')[0].replace('_', ' ')} ({'UCI' if k.endswith('uci') else 'REFIT'})" for k in keys]
fig, ax = plt.subplots(figsize=(8.2, 6.6))
y = np.arange(len(keys))
ax.barh(y + 0.2, [100 * em[k]["v1"] for k in keys], height=0.38, color=ORANGE, label="First run (UCI only)")
ax.barh(y - 0.2, [100 * em[k]["v2"] for k in keys], height=0.38, color=BLUE, label="Final run (UCI + REFIT)")
ax.set_yticks(y, labels, fontsize=13)
ax.set_xlim(0, 100)
ax.grid(axis="y", visible=False)
ax.set_xlabel("Answered exactly right (%)")
ax.legend(frameon=False, loc="lower right", fontsize=13)
ax.set_title("Exact match per task")
save(fig, "fig_exact_match_slide.png")

# 4. Accuracy vs amount of numbers in the prompt (window statistics)
bins = res["numbers_in_prompt"]["by_bin"]
fig, ax = plt.subplots(figsize=(6.6, 4.6))
x = np.arange(len(bins))
vals = [100 * b["em"] for b in bins]
ax.bar(x, vals, color=BLUE, width=0.6)
for i, v in enumerate(vals):
    ax.text(i, v + 2, f"{v:.0f}%", ha="center", color=INK, fontsize=15)
ax.set_xticks(x, [b["bin"].replace("--", "–").replace("$\\geq$", "≥") for b in bins])
ax.set_ylim(0, 75)
ax.grid(axis="x", visible=False)
ax.set_xlabel("Numbers in the question")
ax.set_ylabel("Exactly right (%)")
ax.set_title("More numbers, more arithmetic errors")
save(fig, "fig_numbers_slide.png")

print("wrote", ", ".join(["fig_tasks_large.png", "fig_scratch_loss.png", "fig_exact_match_slide.png", "fig_numbers_slide.png"]))
