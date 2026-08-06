#!/usr/bin/env python3
"""Regenerate the README's results tables from the run's JSON.

The tables used to be hand-maintained markdown, which is how they ended up
quoting an engine that was no longer in the matrix. They are now generated, so
the only way to change them is to re-run the benchmark.

HTML rather than markdown pipes: GitHub renders `<table>` fine, and this needs
`<sub>` captions and right-aligned numeric columns that markdown cannot express
without a wall of colons.

    python3 scripts/render_readme.py [results.json] [README.md]
"""
import json
import sys
from collections import defaultdict

SRC = sys.argv[1] if len(sys.argv) > 1 else "tokenizer_bench_results.json"
DST = sys.argv[2] if len(sys.argv) > 2 else "README.md"
BEGIN, END = "<!-- RESULTS:BEGIN -->", "<!-- RESULTS:END -->"

# Engines covering a single model would collapse the common-cell set to nothing,
# so they are kept out of the cross-engine ranking and reported on their own.
SINGLE_MODEL = {"minbpe", "mistral-common"}
REF = "hf-tokenizers"
LABEL = {
    "hf-tokenizers": 'tokenizers 0.23.1 <sub>reference</sub>',
    "pipeline": 'pipeline <a href="https://github.com/huggingface/tokenizers/pull/2279">#2279</a>'
                ' + <a href="https://github.com/huggingface/tokenizers/pull/2296">#2296</a>',
    "ai-tokenizer": "ai-tokenizer <sub>JS</sub>",
}

rep = json.load(open(SRC))
cell = defaultdict(dict)
for run in rep["runs"]:
    md = run["dataset_metadata"]
    for e in run["results"]:
        cell[(md["model"], md["corpus"])][e["tokenizer_name"]] = e

engines = sorted({n for v in cell.values() for n in v})
ranked = [e for e in engines if e not in SINGLE_MODEL
          and any(cell[k].get(e, {}).get("verified") for k in cell)]

# The common set: cells where every ranked engine ran AND matched the reference
# ids. Comparing medians over each engine's own cells silently rewards whichever
# engine declined the hard ones.
common = [k for k in cell
          if all(cell[k].get(e, {}).get("verified") for e in ranked)]


def med(xs):
    xs = sorted(xs)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def num(v, f="{:.1f}", dash="—"):
    return dash if v is None else f.format(v)


ref_med = med([cell[k][REF]["mbps"] for k in common if REF in cell[k]]) or 1.0

rows = []
for e in ranked:
    v = [cell[k][e]["mbps"] for k in common]
    rows.append((e, med(v), min(v), max(v)))
rows.sort(key=lambda r: -r[1])

out = [BEGIN, ""]
models = sorted({m for m, _ in common})
out.append(f"### The ranking — {len(ranked)} engines over the {len(common)} cells all of them verify")
out.append("")
by_model = defaultdict(list)
for m, c in sorted(common):
    by_model[m].append(c)
out.append("Cells: " + "; ".join(
    f"**{m}** × {{{', '.join(sorted(cs))}}}" for m, cs in sorted(by_model.items())) + ".")
out.append("Single thread, median of 5 timed passes over disjoint slices, warm, "
           "`add_special_tokens = false`, Apple M-series.")
out.append("")
out.append('<table>')
out.append('<thead><tr><th align="left">engine</th><th align="right">median MB/s</th>'
           '<th align="right">× ref</th><th align="right">min</th><th align="right">max</th></tr></thead>')
out.append("<tbody>")
for e, m, lo, hi in rows:
    name = LABEL.get(e, e)
    val = f"<b>{m:.1f}</b>" if e == rows[0][0] else f"{m:.1f}"
    out.append(f'<tr><td align="left">{name}</td><td align="right">{val}</td>'
               f'<td align="right">{m / ref_med:.1f}×</td>'
               f'<td align="right">{lo:.0f}</td><td align="right">{hi:.0f}</td></tr>')
out.append("</tbody></table>")
out.append("")

# --- coverage -------------------------------------------------------------
out.append("### Coverage — what each engine can actually do")
out.append("")
out.append("The other half of the picture, and the two must be read together: an engine high in "
           "the table above may be there partly because it declines the hard cells. "
           "**own-cells median is not cross-comparable** — it is each engine measured on whatever "
           "subset it supports.")
out.append("")
out.append('<table>')
out.append('<thead><tr><th align="left">engine</th><th align="right">cells</th>'
           '<th align="right">ids match</th><th align="right">ids differ</th>'
           '<th align="right">unsupported</th><th align="right">own-cells median</th>'
           '<th align="right">RAM</th><th align="right">package</th></tr></thead>')
out.append("<tbody>")
cov = []
for e in engines:
    ran = ok = bad = unsup = 0
    mbps, ram, pkg = [], [], None
    for k in cell:
        r = cell[k].get(e)
        if not r:
            continue
        if r.get("unsupported"):
            unsup += 1
            continue
        ran += 1
        if r.get("verified") is True:
            ok += 1
        elif r.get("verified") is False:
            bad += 1
        mbps.append(r["mbps"])
        if r.get("heap_load_mb"):
            ram.append(r["heap_load_mb"])
        pkg = pkg or r.get("crate_size_kb")
    cov.append((e, ran, ok, bad, unsup, med(mbps), med(ram), pkg))
for e, ran, ok, bad, unsup, m, ram, pkg in sorted(cov, key=lambda r: (-r[2], -r[1])):
    name = LABEL.get(e, e)
    okc = "—" if e == REF else (f"<b>{ok}</b>" if bad == 0 else str(ok))
    badc = "—" if e == REF else (f"<b>{bad}</b>" if bad else "0")
    out.append(f'<tr><td align="left">{name}</td><td align="right">{ran}</td>'
               f'<td align="right">{okc}</td><td align="right">{badc}</td>'
               f'<td align="right">{unsup}</td><td align="right">{num(m)}</td>'
               f'<td align="right">{num(ram, "{:.0f} MB")}</td>'
               f'<td align="right">{num(pkg, "{:.0f} kB")}</td></tr>')
out.append("</tbody></table>")
out.append("")
out.append('<sub>RAM is live heap held by the loaded tokenizer, not RSS — RSS is a high-water '
           'mark that never falls, so it bills a loader for the intermediate it already freed. '
           'See <code>core/src/mem.rs</code>.</sub>')
out.append("")
out.append(END)

readme = open(DST).read()
i, j = readme.index(BEGIN), readme.index(END) + len(END)
open(DST, "w").write(readme[:i] + "\n".join(out) + readme[j:])
print(f"README tables regenerated: {len(ranked)} ranked engines, {len(common)} common cells")
