#!/usr/bin/env python3
"""Static summary figures from tokenizer_bench_results.json.

The dashboard is the thing you explore; this is the thing you paste into a PR.
Only `verified` cells are ranked against each other -- an engine that emitted
different ids did different work, so its MB/s is not comparable and is dropped
rather than quietly plotted next to the rest.
"""
import json
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = sys.argv[1] if len(sys.argv) > 1 else "tokenizer_bench_results.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "figs"

BG, FG, GRID = "#0d1117", "#e6edf3", "#21262d"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
    "axes.edgecolor": GRID, "grid.color": GRID, "font.size": 9,
})
COLOR = {
    "gigatoken": "#7ee787", "pipeline": "#58a6ff", "hf-tokenizers": "#f778ba",
    "tiktoken": "#ffa657", "fastokens": "#a5d6ff", "tokie": "#d2a8ff",
    "kitoken": "#ffdf5d", "wordchipper": "#79c0ff", "llamacpp": "#ff7b72",
    "blingfire": "#56d4dd", "minbpe": "#8b949e", "ai-tokenizer": "#e3b341",
}
# Colour alone does not separate these lines: fastokens/wordchipper/pipeline are
# three blues and tiktoken/ai-tokenizer two ambers, which is unreadable where
# curves cross. Dash pattern and marker shape carry the identity so the plot
# still reads in greyscale, or to a reader who cannot tell the blues apart.
STYLE = {
    "gigatoken":     ("-",  "o"),
    "pipeline":      ("-",  "s"),
    "hf-tokenizers": ("--", "o"),
    "tiktoken":      ("--", "^"),
    "fastokens":     (":",  "D"),
    "tokie":         (":",  "v"),
    "kitoken":       ("-.", "P"),
    "wordchipper":   ("-.", "X"),
    "llamacpp":      ((0, (3, 1, 1, 1)), "*"),
    "ai-tokenizer":  ((0, (5, 2)), "h"),
    "blingfire":     ((0, (1, 1)), "d"),
    "minbpe":        ((0, (7, 3)), "8"),
}


def style(e):
    return STYLE.get(e, ("-", "o"))

rep = json.load(open(SRC))
# cell[(model, corpus)][engine] = result, verified only
cell = defaultdict(dict)
allcell = defaultdict(dict)
for run in rep["runs"]:
    md = run["dataset_metadata"]
    key = (md["model"], md["corpus"])
    for e in run["results"]:
        allcell[key][e["tokenizer_name"]] = e
        if e.get("verified"):
            cell[key][e["tokenizer_name"]] = e

models = sorted({m for m, _ in cell})
corpora = sorted({c for _, c in cell})
engines = sorted({n for v in cell.values() for n in v})


def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else 0.0


# ---------------------------------------------------------------- figure 1
# Overall standing: median MB/s per engine over every cell it got right.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2))
rows = []
for e in engines:
    v = [r["mbps"] for k in cell for n, r in cell[k].items() if n == e]
    if v:
        rows.append((e, med(v), len(v)))
rows.sort(key=lambda r: r[1])
ax1.barh([r[0] for r in rows], [r[1] for r in rows],
         color=[COLOR.get(r[0], "#8b949e") for r in rows])
for i, (e, v, n) in enumerate(rows):
    ax1.text(v * 1.03, i, f"{v:.0f}  ({n} cells)", va="center", fontsize=8, color=FG)
ax1.set_xscale("log")
ax1.set_xlim(0.5, max(r[1] for r in rows) * 3)
ax1.set_xlabel("MB/s  (median over byte-exact cells, log scale)")
ax1.set_title("Overall throughput, 1 thread", loc="left", fontweight="bold")
ax1.grid(axis="x", alpha=0.3)

# Cost per byte for the same engines -- same data, the units a profiler thinks in.
rows2 = sorted(((e, med([r["ns_per_byte"] for k in cell for n, r in cell[k].items() if n == e]))
                for e in engines), key=lambda r: -r[1])
ax2.barh([r[0] for r in rows2], [r[1] for r in rows2],
         color=[COLOR.get(r[0], "#8b949e") for r in rows2])
for i, (e, v) in enumerate(rows2):
    ax2.text(v * 1.03, i, f"{v:.1f}", va="center", fontsize=8, color=FG)
ax2.set_xscale("log")
ax2.set_xlabel("ns per byte  (median, log scale)")
ax2.set_title("Same runs, per-byte cost", loc="left", fontweight="bold")
ax2.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/01-overall.png", dpi=150)

# ---------------------------------------------------------------- figure 2
# Per-model lines across corpora: where each engine wins and loses.
top = ["gigatoken", "pipeline", "hf-tokenizers", "tiktoken", "fastokens", "tokie"]
fig, axes = plt.subplots(2, 4, figsize=(20, 8.5), sharex=True)
show = [c for c in corpora if c in {
    "english", "code", "xnli", "chinese", "japanese", "korean", "russian",
    "arabic", "thai", "agentic-swe", "agentic-tools", "multilingual-mix"}]
for ax, m in zip(axes.flat, models):
    for e in top:
        ys = [cell[(m, c)].get(e, {}).get("mbps") for c in show]
        xs = [i for i, y in enumerate(ys) if y]
        if not xs:
            continue
        ls, mk = style(e)
        ax.plot(xs, [ys[i] for i in xs], marker=mk, ms=4, lw=1.6, linestyle=ls,
                markeredgecolor=BG, markeredgewidth=0.4,
                color=COLOR.get(e, "#8b949e"), label=e)
    ax.set_title(m, loc="left", fontsize=10, fontweight="bold")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.set_xticks(range(len(show)))
    ax.set_xticklabels(show, rotation=60, ha="right", fontsize=7)
for ax in axes.flat[len(models):]:
    ax.axis("off")
axes.flat[0].set_ylabel("MB/s (log)")
from matplotlib.lines import Line2D
axes.flat[len(models)].legend(
    [Line2D([], [], color=COLOR[e], marker=style(e)[1], linestyle=style(e)[0], ms=6, lw=2)
     for e in top], top,
    loc="center", frameon=False, fontsize=11, title="byte-exact cells only")
fig.suptitle("Throughput per model across corpora — every engine that matched the reference ids",
             x=0.01, ha="left", fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f"{OUT}/02-per-model.png", dpi=150)

# Same reasoning as STYLE, for the per-model lines in figure 3.
MODEL_STYLE = [("-", "o"), ("--", "s"), (":", "^"), ("-.", "D"),
               ((0, (5, 2)), "v"), ((0, (3, 1, 1, 1)), "P"), ((0, (1, 1)), "X")]

# ---------------------------------------------------------------- figure 3
# The controlled comparison: same project, same tokenizer.json, new encode path.
fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
pairs = [("pipeline", "hf-tokenizers", "pipeline (#2279+#2296) vs tokenizers 0.23.1"),
         ("pipeline", "gigatoken", "pipeline vs gigatoken"),
         ("gigatoken", "hf-tokenizers", "gigatoken vs tokenizers 0.23.1")]
for ax, (a, b, title) in zip(axes, pairs):
    for m in models:
        xs, ys = [], []
        for i, c in enumerate(show):
            ra, rb = cell[(m, c)].get(a), cell[(m, c)].get(b)
            if ra and rb:
                xs.append(i)
                ys.append(ra["mbps"] / rb["mbps"])
        if xs:
            ls, mk = MODEL_STYLE[models.index(m) % len(MODEL_STYLE)]
            ax.plot(xs, ys, marker=mk, ms=4, lw=1.4, linestyle=ls, label=m,
                    markeredgecolor=BG, markeredgewidth=0.4)
    ax.axhline(1, color="#8b949e", lw=1, ls="--")
    ax.set_yscale("log")
    ax.set_title(title, loc="left", fontsize=9.5, fontweight="bold")
    ax.set_ylabel(f"× faster than {b}")
    ax.set_xticks(range(len(show)))
    ax.set_xticklabels(show, rotation=60, ha="right", fontsize=7)
    ax.grid(alpha=0.25)
axes[0].legend(fontsize=7, frameon=False, ncol=2)
fig.tight_layout()
fig.savefig(f"{OUT}/03-headtohead.png", dpi=150)

# ---------------------------------------------------------------- figure 4
# Phase breakdown where the engine reports one, and footprint vs speed.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.6))
PH = [("normalization", "normalize", "#f778ba"),
      ("pre_tokenization", "pre-tokenize", "#ffa657"),
      ("core_encoding", "model (merges)", "#58a6ff"),
      ("post_processing", "post-process", "#7ee787")]
have = [e for e in engines if any(cell[k].get(e, {}).get("breakdown_nanoseconds") for k in cell)]
labels, stacks = [], []
for e in have:
    for m in models:
        r = cell.get((m, "english"), {}).get(e)
        if r and r.get("breakdown_nanoseconds"):
            bd = r["breakdown_nanoseconds"]
            nb = rep["runs"][0]["dataset_metadata"]["file_size_bytes"]
            labels.append(f"{e}\n{m}")
            stacks.append([bd.get(k, 0) / max(r["total_tokens_produced"], 1) for k, _, _ in PH])
if stacks:
    bot = [0] * len(stacks)
    for i, (_, name, col) in enumerate(PH):
        vals = [s[i] for s in stacks]
        ax1.bar(range(len(stacks)), vals, bottom=bot, color=col, label=name)
        bot = [b + v for b, v in zip(bot, vals)]
    ax1.set_xticks(range(len(stacks)))
    ax1.set_xticklabels(labels, fontsize=6.5, rotation=70, ha="right")
    ax1.set_ylabel("ns per token")
    ax1.legend(frameon=False, fontsize=8)
ax1.set_title("Where the time goes — english. Only hf-tokenizers reports a phase split;\n"
              "no other engine's API separates its stages, so none is invented for them.",
              loc="left", fontsize=9.5, fontweight="bold")
ax1.grid(axis="y", alpha=0.25)

for e in engines:
    pts = [(r["heap_load_mb"], r["rss_delta_mb"], r["mbps"])
           for k in cell for n, r in cell[k].items()
           if n == e and r.get("heap_load_mb") and r.get("rss_delta_mb")]
    if not pts:
        continue
    hx, rx, y = med([p[0] for p in pts]), med([p[1] for p in pts]), med([p[2] for p in pts])
    # The bar from live heap to RSS is the memory an engine allocates while
    # building and then frees -- gone, but still resident. Long bar = a loader
    # that constructs an intermediate; the dot is what it actually holds.
    ax2.plot([hx, rx], [y, y], color=COLOR.get(e, "#8b949e"), lw=1.2, alpha=0.45)
    ax2.scatter([rx], [y], s=22, marker="|", color=COLOR.get(e, "#8b949e"))
    ax2.scatter([hx], [y], s=95, color=COLOR.get(e, "#8b949e"), edgecolor=BG, zorder=3)
    ax2.annotate(e, (hx, y), fontsize=8, xytext=(-8, 7), ha="right",
                 textcoords="offset points", color=FG)
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel("MB held by the loaded tokenizer (dot = live heap, tick = RSS high-water)")
ax2.set_ylabel("MB/s (median)")
ax2.set_title("Speed against footprint — dot is what it holds, tick is what RSS charges it",
              loc="left", fontsize=9.5, fontweight="bold")
ax2.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(f"{OUT}/04-phases-footprint.png", dpi=150)

print(f"wrote 4 figures to {OUT}/")
print(f"cells: {len(cell)}  engines: {len(engines)}")

# ---------------------------------------------------------------- figure 5
# The metric matters more than the engine here: RSS and live heap disagree
# about who is small, and they disagree in a systematic direction.
import json as _json
try:
    fp = _json.load(open("footprint.json"))
except FileNotFoundError:
    fp = {}
if fp:
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(17, 5.8))
    ms = ["bert-wiki", "gpt2", "llama-2", "llama-3", "deepseek-v4", "mistral-nemo"]
    engs = sorted(fp, key=lambda e: sum(v["heap_load_mb"] for v in fp[e].values()) / len(fp[e]))
    engs = [e for e in engs if len(fp[e]) >= 4]
    w = 0.8 / len(ms)
    for j, m in enumerate(ms):
        for ax, key in ((axa, "heap_load_mb"), (axb, "rss_delta_mb")):
            vals = [fp[e].get(m, {}).get(key) or 0 for e in engs]
            ax.bar([i + j * w for i in range(len(engs))], vals, w,
                   label=m if ax is axa else None)
    for ax, t in ((axa, "Live heap of the loaded tokenizer — what it actually holds"),
                  (axb, "RSS delta, same runs — what the first dashboard reported")):
        ax.set_xticks([i + 0.4 for i in range(len(engs))])
        ax.set_xticklabels(engs, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("MB")
        ax.set_title(t, loc="left", fontsize=9.5, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
    axa.legend(fontsize=8, frameon=False, ncol=3, title="model")
    axa.annotate("pipeline holds the least\non every real vocabulary",
                 xy=(0.5, 0.82), xycoords="axes fraction", fontsize=9, color="#7ee787")
    axb.annotate("same engine, now looks worst —\nRSS keeps the pages of a parse it freed",
                 xy=(0.30, 0.82), xycoords="axes fraction", fontsize=9, color="#ff7b72")
    fig.suptitle("Why the footprint column was wrong: RSS is a high-water mark, "
                 "live heap is the resident structure", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{OUT}/05-memory-metric.png", dpi=150)
    print("wrote 05-memory-metric.png")
