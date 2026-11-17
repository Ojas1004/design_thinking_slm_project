"""Statistical analysis of the large evaluation (evaluate_full.py) for the research article.

Reads the per-question records of the current model (v2: UCI + REFIT corpus) and the first model (v1: UCI-only
corpus), both scored on the same sampled test questions, and writes:
  paper/results.json            every number quoted in the article
  paper/tables/*.tex            LaTeX tables, \\input by main.tex
  paper/figures/fig_*.pdf       figures

usage (from the repo root): python scripts/analyze_eval.py
"""
import json
import math
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

V2 = "out/qwen3_scratch/eval_full.jsonl"
V1 = "out/qwen3_scratch_v1/eval_full.jsonl"
OUT = "paper"
os.makedirs(f"{OUT}/tables", exist_ok=True)
os.makedirs(f"{OUT}/figures", exist_ok=True)

FORECAST = ("forecast_next", "forecast_hour")
REFUSE_RE = re.compile(r"^(That cannot be|I cannot give|That date is outside|One 10-minute reading is)")
ABOUT_RE = re.compile(r"about (\d+) Wh")
NUM_RE = re.compile(r"\d+(?:\.\d+)?")
rng = np.random.default_rng(2026)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def load(path):
    return [json.loads(l) for l in open(path)]


v2, v1 = load(V2), load(V1)
assert [r["prompt"] for r in v2] == [r["prompt"] for r in v1], "the two runs must use the same questions"
for r in v2 + v1:
    r["em"] = r["model"] == r["reference"]
    r["template"] = NUM_RE.sub("#", r["model"]) == NUM_RE.sub("#", r["reference"])
    r["refused"] = bool(REFUSE_RE.match(r["model"]))


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def mcnemar(a, b):
    """exact McNemar test on paired booleans a (v1) and b (v2)"""
    b01 = sum((not x) and y for x, y in zip(a, b))   # v1 wrong, v2 right
    b10 = sum(x and (not y) for x, y in zip(a, b))   # v1 right, v2 wrong
    p = stats.binomtest(b01, b01 + b10, 0.5).pvalue if b01 + b10 else 1.0
    return b01, b10, p


def pct(x):
    return f"{100 * x:.1f}"


def fmt_p(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


res = {"n_questions": len(v2)}

# ------------------------------------------------------------------ sanity check of the refusal detector
ref_refuse = [bool(REFUSE_RE.match(r["reference"])) for r in v2]
agree = sum(rr == (not r["answerable"]) for rr, r in zip(ref_refuse, v2))
res["refusal_detector_agreement_on_references"] = agree / len(v2)

# ------------------------------------------------------------------ exact match per group, v1 vs v2
groups = defaultdict(list)
for i, r in enumerate(v2):
    groups[(r["task"], r["source"])].append(i)
exact_keys = sorted(k for k in groups if k[0] not in FORECAST)
rows, em_res = [], {}
for task, src in exact_keys:
    idx = groups[(task, src)]
    n = len(idx)
    a = [v1[i]["em"] for i in idx]
    b = [v2[i]["em"] for i in idx]
    p2, lo2, hi2 = wilson(sum(b), n)
    p1, lo1, hi1 = wilson(sum(a), n)
    t2 = sum(v2[i]["template"] for i in idx) / n
    b01, b10, p = mcnemar(a, b)
    em_res[f"{task}/{src}"] = dict(n=n, v2=p2, v2_ci=[lo2, hi2], v1=p1, v1_ci=[lo1, hi1], v2_template=t2,
                                   mcnemar_v2_only=b01, mcnemar_v1_only=b10, mcnemar_p=p)
    rows.append((task, src, n, p1, p2, lo2, hi2, t2, p))
res["exact_match"] = em_res

pooled = {}
for src in ("uci", "refit"):
    idx = [i for (t, s), ii in groups.items() if s == src and t not in FORECAST for i in ii]
    a = [v1[i]["em"] for i in idx]
    b = [v2[i]["em"] for i in idx]
    p2, lo2, hi2 = wilson(sum(b), len(b))
    p1, lo1, hi1 = wilson(sum(a), len(a))
    b01, b10, p = mcnemar(a, b)
    t2 = sum(v2[i]["template"] for i in idx) / len(idx)
    t1 = sum(v1[i]["template"] for i in idx) / len(idx)
    pooled[src] = dict(n=len(idx), v2=p2, v2_ci=[lo2, hi2], v1=p1, v1_ci=[lo1, hi1], v2_template=t2,
                       v1_template=t1, mcnemar_v2_only=b01, mcnemar_v1_only=b10, mcnemar_p=p)
res["exact_match_pooled"] = pooled

# macro average over tasks (each task weighted equally)
for src in ("uci", "refit"):
    vals2 = [em_res[k]["v2"] for k in em_res if k.endswith("/" + src)]
    vals1 = [em_res[k]["v1"] for k in em_res if k.endswith("/" + src)]
    pooled[src]["v2_macro"] = float(np.mean(vals2))
    pooled[src]["v1_macro"] = float(np.mean(vals1))

SRC = {"uci": "UCI", "refit": "REFIT"}
with open(f"{OUT}/tables/exact_match.tex", "w") as f:
    for task, src, n, p1, p2, lo2, hi2, t2, p in rows:
        name = task.replace("_", r"\_")
        f.write(f"{name} & {SRC[src]} & {n} & {pct(p1)} & \\textbf{{{pct(p2)}}} & [{pct(lo2)}, {pct(hi2)}] & "
                f"{pct(t2)} & {fmt_p(p)} \\\\\n")
    f.write("\\midrule\n")
    for src in ("uci", "refit"):
        o = pooled[src]
        f.write(f"\\textit{{pooled}} & {SRC[src]} & {o['n']} & {pct(o['v1'])} & \\textbf{{{pct(o['v2'])}}} & "
                f"[{pct(o['v2_ci'][0])}, {pct(o['v2_ci'][1])}] & {pct(o['v2_template'])} & {fmt_p(o['mcnemar_p'])} \\\\\n")

# ------------------------------------------------------------------ refusal / hallucination analysis (all questions)
def refusal(recs):
    tp = sum(r["refused"] and not r["answerable"] for r in recs)
    fn = sum((not r["refused"]) and not r["answerable"] for r in recs)
    fp = sum(r["refused"] and r["answerable"] for r in recs)
    tn = sum((not r["refused"]) and r["answerable"] for r in recs)
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    return dict(tp=tp, fn=fn, fp=fp, tn=tn, precision=prec, recall=rec,
                hallucination_rate=wilson(fn, tp + fn), over_refusal_rate=wilson(fp, fp + tn),
                f1=2 * prec * rec / (prec + rec) if prec + rec else float("nan"))


res["refusal"] = {"v2": refusal(v2), "v1_uci_only_questions": refusal([r for r in v1 if r["source"] == "uci"]),
                  "v2_uci_only_questions": refusal([r for r in v2 if r["source"] == "uci"])}
with open(f"{OUT}/tables/refusal.tex", "w") as f:
    for label, key in (("v1 (UCI questions)", "v1_uci_only_questions"), ("v2 (UCI questions)", "v2_uci_only_questions"),
                       ("v2 (all questions)", "v2")):
        r = res["refusal"][key]
        h, hl, hh = r["hallucination_rate"]
        o, ol, oh = r["over_refusal_rate"]
        f.write(f"{label} & {r['tp']} & {r['fn']} & {r['fp']} & {r['tn']} & {pct(r['precision'])} & {pct(r['recall'])} & "
                f"{pct(h)} [{pct(hl)}, {pct(hh)}] & {pct(o)} [{pct(ol)}, {pct(oh)}] \\\\\n")

# ------------------------------------------------------------------ forecasting
def history(prompt):
    stamped = re.findall(r"\d\d:\d\d  (\d+) Wh", prompt)
    if stamped:
        return [int(v) for v in stamped]
    listed = re.search(r"((?:\d+, )+\d+) \(Wh, oldest first", prompt)
    return [int(v) for v in listed.group(1).split(", ")] if listed else []


def boot_ci(d, n_boot=10000):
    d = np.asarray(d, float)
    bs = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


fc_res, fc_rows = {}, []
for task in FORECAST:
    for src in ("uci", "refit"):
        per_hour = 4 if src == "refit" else 6
        errs = {"model": [], "persistence": [], "mean": [], "v1": []}
        unparsed = 0
        for i in groups[(task, src)]:
            r = v2[i]
            got, want, hist = ABOUT_RE.search(r["model"]), ABOUT_RE.search(r["reference"]), history(r["prompt"])
            got1 = ABOUT_RE.search(v1[i]["model"])
            if not (got and want and hist):
                unparsed += 1
                continue
            y = int(want.group(1))
            if task == "forecast_next":
                pers, mean = hist[-1], float(np.mean(hist))
            else:
                pers, mean = sum(hist[-per_hour:]), float(np.mean(hist)) * per_hour
            errs["model"].append(abs(int(got.group(1)) - y))
            errs["persistence"].append(abs(pers - y))
            errs["mean"].append(abs(mean - y))
            errs["v1"].append(abs(int(got1.group(1)) - y) if got1 else np.nan)
        e = {k: np.asarray(v, float) for k, v in errs.items()}
        d = e["model"] - e["persistence"]
        w = stats.wilcoxon(e["model"], e["persistence"], zero_method="zsplit") if np.any(d != 0) else None
        mae = {k: float(np.nanmean(v)) for k, v in e.items()}
        cis = {k: boot_ci(v[~np.isnan(v)]) for k, v in e.items() if (~np.isnan(v)).any()}
        out = dict(n=len(e["model"]), unparsed=unparsed, mae=mae, mae_ci=cis,
                   rmse={k: float(np.sqrt(np.nanmean(v ** 2))) for k, v in e.items()},
                   diff_vs_persistence=float(d.mean()), diff_ci=boot_ci(d),
                   wilcoxon_p=float(w.pvalue) if w else 1.0,
                   skill_vs_persistence=1 - mae["model"] / mae["persistence"])
        fc_res[f"{task}/{src}"] = out
        fc_rows.append((task, src, out))
res["forecast"] = fc_res
with open(f"{OUT}/tables/forecast.tex", "w") as f:
    for task, src, o in fc_rows:
        name = task.replace("_", r"\_")
        v1m = f"{o['mae']['v1']:.1f}" if src == "uci" else "--"
        f.write(f"{name} & {SRC[src]} & {o['n']} & \\textbf{{{o['mae']['model']:.1f}}} & {v1m} & {o['mae']['persistence']:.1f} & "
                f"{o['mae']['mean']:.1f} & {o['diff_vs_persistence']:+.1f} [{o['diff_ci'][0]:+.1f}, {o['diff_ci'][1]:+.1f}] & "
                f"{fmt_p(o['wilcoxon_p'])} & {o['skill_vs_persistence']:+.2f} \\\\\n")

# ------------------------------------------------------------------ effect of the amount of numeric evidence in the prompt
numeric_tasks = ("window_stats",)   # one task, so the length effect is not confounded with task difficulty
idx = [i for i, r in enumerate(v2) if r["task"] in numeric_tasks]
cnt = np.array([len(NUM_RE.findall(v2[i]["prompt"])) for i in idx])
ok = np.array([v2[i]["em"] for i in idx])
pb = stats.pointbiserialr(ok.astype(float), cnt)
mw = stats.mannwhitneyu(cnt[ok], cnt[~ok], alternative="less")
bins = [0, 20, 30, 40, 1000]
blabels = ["<20", "20--29", "30--39", "$\\geq$40"]
by_bin = []
for lo, hi, lab in zip(bins[:-1], bins[1:], blabels):
    m = (cnt >= lo) & (cnt < hi)
    by_bin.append((lab, int(m.sum()), *wilson(int(ok[m].sum()), int(m.sum()))))
res["numbers_in_prompt"] = dict(tasks=numeric_tasks, n=len(idx), pointbiserial_r=float(pb.statistic),
                               pointbiserial_p=float(pb.pvalue), mannwhitney_p=float(mw.pvalue),
                               median_numbers_correct=float(np.median(cnt[ok])),
                               median_numbers_wrong=float(np.median(cnt[~ok])),
                               by_bin=[dict(bin=b[0], n=b[1], em=b[2], ci=[b[3], b[4]]) for b in by_bin])

# ------------------------------------------------------------------ error taxonomy on wrong answers (v2, computable tasks)
wrong = [r for r in v2 if r["task"] not in FORECAST and not r["em"]]
res["error_taxonomy"] = dict(
    wrong=len(wrong),
    numbers_only=sum(r["template"] for r in wrong),
    wrong_refusal=sum(r["refused"] and r["answerable"] for r in wrong),
    missed_refusal=sum((not r["refused"]) and not r["answerable"] for r in wrong),
)
res["error_taxonomy"]["other_wording"] = (res["error_taxonomy"]["wrong"] - res["error_taxonomy"]["numbers_only"]
                                         - res["error_taxonomy"]["wrong_refusal"] - res["error_taxonomy"]["missed_refusal"])

# ------------------------------------------------------------------ UCI vs REFIT on the shared computable tasks (v2)
shared = sorted({t for t, s in exact_keys if s == "refit"})
res["uci_vs_refit"] = {}
for t in shared:
    a = [v2[i]["em"] for i in groups[(t, "uci")]]
    b = [v2[i]["em"] for i in groups[(t, "refit")]]
    table = [[sum(a), len(a) - sum(a)], [sum(b), len(b) - sum(b)]]
    res["uci_vs_refit"][t] = dict(uci=sum(a) / len(a), refit=sum(b) / len(b), fisher_p=float(stats.fisher_exact(table).pvalue))

json.dump(res, open(f"{OUT}/results.json", "w"), indent=2)

# ------------------------------------------------------------------ figures
# Fig: exact match per task, v1 vs v2 with Wilson 95% CIs
labels = [f"{t.replace('_', ' ')} ({SRC[s]})" for t, s, *_ in rows]
y = np.arange(len(rows))[::-1]
fig, ax = plt.subplots(figsize=(3.45, 3.6))
for off, key, color, name in ((0.18, "v1", ORANGE, "v1: UCI-only corpus"), (-0.18, "v2", BLUE, "v2: UCI + REFIT corpus")):
    p = np.array([em_res[f"{t}/{s}"][key] for t, s, *_ in rows]) * 100
    ci = np.array([em_res[f"{t}/{s}"][key + "_ci"] for t, s, *_ in rows]) * 100
    ax.barh(y + off, p, height=0.34, color=color, label=name, edgecolor="white", linewidth=0.6)
    ax.errorbar(p, y + off, xerr=[np.clip(p - ci[:, 0], 0, None), np.clip(ci[:, 1] - p, 0, None)], fmt="none", ecolor=INK2, elinewidth=0.6, capsize=1.2)
ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.set_xlim(0, 100)
ax.set_xlabel("exact match on held-out test questions (%)")
ax.grid(axis="y", visible=False)
ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2, handlelength=1, columnspacing=1)
fig.savefig(f"{OUT}/figures/fig_exact_match.pdf")
plt.close(fig)

# Fig: forecast MAE with bootstrap CIs
fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.1))
for ax, task, title in zip(axes, FORECAST, ("next reading", "next hour (sum)")):
    xs = np.arange(2)
    for j, (key, color, name) in enumerate((("model", BLUE, "SLM (v2)"), ("persistence", ORANGE, "persistence"),
                                            ("mean", AQUA, "window mean"))):
        m = [fc_res[f"{task}/{s}"]["mae"][key] for s in ("uci", "refit")]
        ci = np.array([fc_res[f"{task}/{s}"]["mae_ci"][key] for s in ("uci", "refit")])
        pos = xs + (j - 1) * 0.26
        ax.bar(pos, m, width=0.24, color=color, label=name, edgecolor="white", linewidth=0.6)
        ax.errorbar(pos, m, yerr=[np.clip(np.array(m) - ci[:, 0], 0, None), np.clip(ci[:, 1] - np.array(m), 0, None)], fmt="none", ecolor=INK2,
                    elinewidth=0.6, capsize=1.2)
    ax.set_xticks(xs)
    ax.set_xticklabels(["UCI", "REFIT"])
    ax.set_title(title, fontsize=8, color=INK)
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("MAE (Wh)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=1, columnspacing=1)
fig.subplots_adjust(top=0.78, wspace=0.35)
fig.savefig(f"{OUT}/figures/fig_forecast.pdf")
plt.close(fig)

# Fig: exact match vs amount of numeric evidence in the prompt
fig, ax = plt.subplots(figsize=(3.45, 1.7))
ems = np.array([b[2] for b in by_bin]) * 100
lo = np.array([b[3] for b in by_bin]) * 100
hi = np.array([b[4] for b in by_bin]) * 100
xs = np.arange(len(by_bin))
ax.bar(xs, ems, width=0.55, color=BLUE, edgecolor="white")
ax.errorbar(xs, ems, yerr=[np.clip(ems - lo, 0, None), np.clip(hi - ems, 0, None)], fmt="none", ecolor=INK2, elinewidth=0.6, capsize=1.5)
ax.set_xticks(xs)
ax.set_xticklabels([b[0].replace("--", "–").replace("$\\geq$", "≥") + f"\n(n={b[1]})" for b in by_bin])
ax.set_xlabel("numbers in the prompt")
ax.set_ylabel("exact match (%)")
ax.set_ylim(0, 100)
ax.grid(axis="x", visible=False)
fig.savefig(f"{OUT}/figures/fig_numbers.pdf")
plt.close(fig)

print(json.dumps({k: res[k] for k in ("exact_match_pooled", "refusal", "error_taxonomy", "numbers_in_prompt",
                                       "uci_vs_refit", "refusal_detector_agreement_on_references")}, indent=1))
for k, o in fc_res.items():
    print(k, {m: round(v, 1) for m, v in o["mae"].items()}, "diff", round(o["diff_vs_persistence"], 1),
          [round(x, 1) for x in o["diff_ci"]], "p", round(o["wilcoxon_p"], 4))
