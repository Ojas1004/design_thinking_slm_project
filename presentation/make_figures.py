"""Make the presentation figures. Usage: python make_figures.py <repo root> <dir with data/*.parquet splits>"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1])
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

df = pd.read_csv(ROOT / "data" / "energydata_complete.csv", parse_dates=["date"])
a = df["Appliances"]
stats = {}

# 1. Daily appliance energy over the whole period, with the date splits shaded
daily = df.set_index("date")["Appliances"].resample("D").sum() / 1000
fig, ax = plt.subplots(figsize=(8.5, 4.2))
splits = [("Train", daily.index.min(), pd.Timestamp("2016-04-20"), BLUE),
          ("Val.", pd.Timestamp("2016-04-20"), pd.Timestamp("2016-05-08"), ORANGE),
          ("Test", pd.Timestamp("2016-05-08"), daily.index.max() + pd.Timedelta(days=1), AQUA)]
for name, s, e, c in splits:
    ax.axvspan(s, e, color=c, alpha=0.10, lw=0)
    ax.text(s + (e - s) / 2, daily.max() * 1.04, name, ha="center", color=INK2, fontsize=15)
ax.plot(daily.index, daily.values, color=BLUE)
ax.set_ylim(0, daily.max() * 1.12)
ax.set_ylabel("kWh per day")
import matplotlib.dates as mdates
ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.set_title("Appliance energy per day, 2016")
save(fig, "fig_timeseries.png")

# 2. Heavy-tailed distribution
med, p95 = a.median(), a.quantile(0.95)
top10 = a[a >= a.quantile(0.90)].sum() / a.sum()
stats.update(median=float(med), p95=float(p95), max=int(a.max()), top10_share=round(float(top10), 3),
             rows=len(df), start=str(df.date.min()), end=str(df.date.max()))
fig, ax = plt.subplots(figsize=(7, 4.8))
ax.hist(a, bins=np.arange(0, 1100, 20), color=BLUE, edgecolor=SURF, linewidth=1)
for v, lab, y in [(med, f"median {med:.0f} Wh", 0.88), (p95, f"95th percentile\n{p95:.0f} Wh", 0.55)]:
    ax.axvline(v, color=INK2, ls="--", lw=1.2)
    ax.text(v + 15, ax.get_ylim()[1] * y, lab, color=INK, fontsize=15, va="top")
ax.set_xlabel("Wh per 10 minutes")
ax.set_ylabel("Number of readings")
ax.set_title(f"Top 10% of readings hold {top10:.0%} of energy")
save(fig, "fig_distribution.png")

# 3. Average daily profile, weekday vs weekend
df["hour"] = df.date.dt.hour
df["weekend"] = df.date.dt.day_of_week >= 5
prof = df.groupby(["weekend", "hour"])["Appliances"].mean().unstack(0)
fig, ax = plt.subplots(figsize=(7, 4.8))
ax.plot(prof.index, prof[False], color=BLUE, marker="o", ms=5, label="Weekday")
ax.plot(prof.index, prof[True], color=ORANGE, marker="o", ms=5, label="Weekend")
ax.set_xlim(-0.5, 23.5); ax.set_xticks(range(0, 24, 3))
ax.set_xlabel("Hour of day"); ax.set_ylabel("Mean Wh per 10 min")
ax.legend(frameon=False, loc="upper left")
ax.set_title("Evening peak at 18:00")
save(fig, "fig_daily_profile.png")
stats["peak_hour_weekday"] = int(prof[False].idxmax())

# 4. Autocorrelation: useful signal is 1 day / 1 week back
x = a - a.mean()
def acf(l): return float((x[:-l].values * x[l:].values).mean() / x.var())
lags = list(range(1, 1009, 3))
vals = [acf(l) for l in lags]
stats.update(acf_1=round(acf(1), 2), acf_6=round(acf(6), 2), acf_36=round(acf(36), 2),
             acf_144=round(acf(144), 2), acf_1008=round(acf(1008), 2))
fig, ax = plt.subplots(figsize=(11, 4.0))
ax.plot([l / 6 for l in lags], vals, color=BLUE, lw=1.5)
ax.axhline(0, color=INK2, lw=0.8)
for l, lab in [(144, "same time\nyesterday"), (1008, "same time\nlast week")]:
    ax.axvline(l / 6, color=ORANGE, ls="--", lw=1.2)
    ax.text(l / 6 - 2, 0.55, lab, ha="right", color=INK, fontsize=15)
ax.set_xlabel("How far back (hours)"); ax.set_ylabel("Correlation with now")
ax.set_xlim(0, 170); ax.set_ylim(-0.16, 0.8)
ax.set_title("After a few hours, the signal is only a day or a week back")
save(fig, "fig_autocorr.png")

# 5. Misleading channels: T6 vs T_out, rv1 vs Appliances
r_t6 = df["T6"].corr(df["T_out"]); r_rv = df["rv1"].corr(df["Appliances"])
stats.update(r_t6_tout=round(float(r_t6), 3), r_rv1_app=round(float(r_rv), 3),
             rv_identical=bool((df.rv1 == df.rv2).all()))
s = df.sample(4000, random_state=0)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.3))
axs[0].scatter(s.T_out, s.T6, s=6, color=BLUE, alpha=0.5, lw=0)
axs[0].set_xlabel("Outdoor T_out (°C)"); axs[0].set_ylabel("Sensor T6 (°C)")
axs[0].set_title(f"“T6” is outside the house (r = {r_t6:.2f})", fontsize=17)
axs[1].scatter(s.rv1, s.Appliances, s=6, color=ORANGE, alpha=0.5, lw=0)
axs[1].set_xlabel("rv1 (random variable)"); axs[1].set_ylabel("Appliances (Wh)")
axs[1].set_title(f"rv1 / rv2 are pure noise (r = {r_rv:.2f})", fontsize=17)
save(fig, "fig_traps.png")

# 6. Task mix in the training split
tr = pd.read_parquet(Path(sys.argv[2]) / "data" / "train-00000-of-00001.parquet")
import pyarrow.parquet as pq
names = json.loads(pq.read_schema(Path(sys.argv[2]) / "data" / "train-00000-of-00001.parquet")
                   .metadata[b"huggingface"])["info"]["features"]["task"]["names"]
tr["task_name"] = tr["task"].map(lambda i: names[i])
refuse = {"unanswerable", "units_cost", "false_premise", "noise_features"}
counts = tr["task_name"].value_counts().sort_values()
stats["task_counts"] = counts.to_dict()
fig, ax = plt.subplots(figsize=(7.2, 6))
cols = [ORANGE if n in refuse else BLUE for n in counts.index]
ax.barh(counts.index, counts.values, color=cols, height=0.7)
for i, v in enumerate(counts.values):
    ax.text(v + 30, i, f"{v:,}", va="center", color=INK, fontsize=17)
ax.set_xlim(0, counts.max() * 1.18); ax.grid(axis="y", visible=False)
ax.set_xlabel("Training examples")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BLUE, label="Answer from the data"),
                   Patch(color=ORANGE, label="Know the limits / traps")],
          frameon=False, loc="lower right")
ax.set_title("14 task types in the training split")
save(fig, "fig_tasks.png")

# 7. Training and validation loss
st = json.load(open(ROOT / "out/qwen0.5b-energy-lora/checkpoint-1566/trainer_state.json"))
tl = [(h["step"], h["loss"]) for h in st["log_history"] if "loss" in h]
ev = [(h["step"], h["eval_loss"]) for h in st["log_history"] if "eval_loss" in h]
fig, ax = plt.subplots(figsize=(7, 4.8))
ax.plot(*zip(*tl), color=BLUE, label="Training loss")
ax.plot(*zip(*ev), color=ORANGE, marker="o", ms=7, label="Validation loss")
ax.set_yscale("log")
ax.annotate(f"{ev[-1][1]:.3f}", ev[-1], xytext=(0, 14), textcoords="offset points", ha="right", color=INK)
ax.set_xlabel("Training step"); ax.set_ylabel("Loss (log scale)")
ax.legend(frameon=False)
ax.set_title("Loss fell steadily, no overfitting")
save(fig, "fig_loss.png")

print(json.dumps(stats, indent=1, default=str))
