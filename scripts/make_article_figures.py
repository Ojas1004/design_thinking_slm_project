"""Dataset and training figures for the research article (paper style), written to paper/figures/.

  fig_data.pdf  (a) distribution of UCI appliance energy, (b) autocorrelation up to one week
  fig_loss.pdf  validation loss of the first (UCI-only) and final (UCI + REFIT) training runs

usage (from the repo root): python scripts/make_article_figures.py
"""
import csv
import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "paper/figures"
BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

v = np.array([float(r["Appliances"].strip().strip('"')) for r in csv.DictReader(open("data/energydata_complete.csv"))])

fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 1.9), gridspec_kw={"width_ratios": [1, 1.9]})
a.hist(v, bins=np.arange(0, 1101, 20), color=BLUE, edgecolor="white", linewidth=0.4)
for q, lab in ((np.median(v), "median 60 Wh"), (np.percentile(v, 95), "95th pct. 330 Wh")):
    a.axvline(q, color=INK2, lw=0.7, ls="--")
    a.text(q + 15, a.get_ylim()[1] * (0.85 if q < 100 else 0.55), lab, fontsize=7, color=INK)
a.set_xlabel("appliance energy (Wh per 10 min)")
a.set_ylabel("readings")
a.set_title("(a)", fontsize=8, loc="left")
a.grid(axis="x", visible=False)

x = v - v.mean()
lags = np.arange(0, 1009)
acf = np.array([1.0] + [np.dot(x[:-k], x[k:]) / np.dot(x, x) for k in lags[1:]])
b.plot(lags / 6, acf, color=BLUE, lw=1)
b.axhline(0, color=INK2, lw=0.6)
for h, lab in ((24, "24 h"), (168, "168 h")):
    b.axvline(h, color=ORANGE, lw=0.8, ls="--")
    b.text(h - 2, 0.6, lab, ha="right", fontsize=7, color=INK)
b.set_xlim(0, 170)
b.set_xlabel("lag (hours)")
b.set_ylabel("autocorrelation")
b.set_title("(b)", fontsize=8, loc="left")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_data.pdf")
plt.close(fig)


def val_curve(path):
    nb = json.load(open(path))
    text = ""
    for c in nb["cells"]:
        for o in c.get("outputs", []):
            if o.get("output_type") == "stream":
                text += "".join(o["text"])
    pts = re.findall(r"step\s+(\d+): train loss ([\d.]+), val loss ([\d.]+)", text)
    return np.array([[int(s), float(t), float(vl)] for s, t, vl in pts])


v1 = val_curve("notebooks/reference/5_slm_from_scratch_qwen.v1_uci_only.ipynb")
v2 = val_curve("notebooks/5_slm_from_scratch_qwen.ipynb")
fig, ax = plt.subplots(figsize=(3.45, 1.9))
for d, color, name in ((v1, ORANGE, "v1: UCI-only corpus"), (v2, BLUE, "v2: UCI + REFIT corpus")):
    d = d[d[:, 0] > 0]
    ax.plot(d[:, 0] / 1000, d[:, 2], color=color, lw=1.2, marker="o", ms=2.5, label=f"{name}, validation")
    ax.plot(d[:, 0] / 1000, d[:, 1], color=color, lw=0.8, ls="--", alpha=0.8, label=f"{name}, training")
ax.set_xlabel("training steps of 8 examples (thousands)")
ax.set_ylabel("loss on answer tokens")
ax.set_yscale("log")
ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.45, -0.3), ncol=2, columnspacing=1, handlelength=1.8)
fig.savefig(f"{OUT}/fig_loss.pdf")
plt.close(fig)
print("v1 points", len(v1), "v2 points", len(v2))
