# tokbench

A fair benchmark of tokenizer implementations.

One folder per engine, one Rust driver on top, **one timing loop** shared by all
of them, and an id-verification gate so a number is never published for an
engine that quietly computed something different.

```
core/          the fairness contract + the only timing loop
driver/        the Rust binary: runs the matrix, verifies, writes JSON
binsize/       minimal program linking one engine, to weigh it
engines/<name> one folder per engine
python/        the same protocol, reproduced for interpreted engines
dashboard.html single-file dark dashboard, drag-and-drop the JSON
```

## Why another benchmark

Most tokenizer comparisons are not comparisons. They time different functions
(one engine computes byte offsets, another does not), on different vocabularies,
with load time folded into encode time, at different thread counts, and without
ever checking that the two produced the same token ids. Every one of those makes
a fast engine look faster than it is.

tokbench fixes the measurement, not the result:

1. **Same work, verified.** Every engine's id stream is hashed (FNV-1a) and
   compared against the reference. Different ids → the cell is marked
   `mismatch` and is never ranked. Being fast at the wrong answer is not a win.
2. **Same clock, in-process.** Native engines are called directly through one
   `Engine::encode` trait and timed by the same `Instant`. No subprocess, no IPC.
3. **Load is not encode.** Vocabulary parsing and automaton construction happen
   before the timer and are reported separately as `load_ms`.
4. **Warm cache, stated.** One untimed pass precedes the timed ones, so
   cache-heavy engines are measured in the regime real loops reach.
   `--no-warmup` gives the cold contrast.
5. **One thread by default.** Engines that parallelise internally are flagged;
   a whole-machine number is never printed next to a single-core one unlabelled.
6. **Disclose the extra work.** An engine that also computes byte offsets keeps
   that cost in its number, and the report says so next to it.

The contract is written out in full at the top of [`core/src/lib.rs`](core/src/lib.rs).

## Results are only as honest as their caveats

Cross-engine medians are computed **only over cells every compared engine ran
and verified**. That matters more than it sounds: engines cover different model
families, so a median over each engine's own cells silently rewards the ones
that skip the hard cases. Measured effect of getting this wrong: pipeline 1.41×
understated, executorch 1.82× overstated.

### The ranking — 12 engines over the 12 cells all of them verify

llama-3 × {arabic, chinese, code, english, greek, hindi, thai} and
mistral-nemo × {chinese, code, english, greek, korean}. Single thread, median
of 5, warm, `add_special_tokens = false`, Apple M-series.

| engine | median MB/s | × ref | min | max |
|---|---:|---:|---:|---:|
| pipeline ([#2279](https://github.com/huggingface/tokenizers/pull/2279)) | **755.9** | 62.8× | 322 | 1175 |
| gigatoken | 605.6 | 48.9× | 334 | 1290 |
| wordchipper | 104.7 | 6.4× | 11 | 212 |
| iree | 77.4 | 7.0× | 34 | 104 |
| fastokens | 68.1 | 6.0× | 45 | 77 |
| ai-tokenizer (JS) | 59.1 | 4.4× | 36 | 105 |
| tokie | 36.8 | 2.9× | 15 | 121 |
| tiktoken | 33.6 | 2.7× | 20 | 43 |
| kitoken | 31.7 | 2.6× | 18 | 40 |
| tokenizers 0.23.1 (reference) | 12.4 | 1.0× | 6 | 23 |
| llamacpp | 6.4 | 0.4× | 2 | 18 |
| executorch | 2.1 | 0.2× | 2 | 4 |

`pipeline` is exempt from the verification gate for now, by request: it is an
in-progress PR and all 8 of its mismatches are albert (Unigram). The exemption
suppresses only the exclusion — its mismatch count is still reported in the
coverage table below, and its mismatched cells are still drawn as mismatched.
On this data it changes no ranking, because those cells fall outside the common
set regardless.

minbpe and mistral-common are excluded from this table: each supports only one
model, and including them would collapse the common set to zero cells. Their
own-cells figures are below.

### Coverage — what each engine can actually do

This is the other half of the picture, and the two must be read together: an
engine high in the table above may be there partly because it declines the
hard cells.

| engine | cells run | ids match | ids differ | unsupported | own-cells median | RSS | package |
|---|---:|---:|---:|---:|---:|---:|---:|
| kitoken | 70 | **70** | 0 | 0 | 29.6 | 23 MB | **63 kB** |
| tokenizers 0.23.1 (ref) | 70 | — | — | 0 | 12.2 | 40 MB | 192 kB |
| pipeline | 70 | 62 | 8 | 0 | 535.1 | 63 MB | — |
| tokie | 70 | 52 | 18 | 0 | 44.6 | 44 MB | 181 kB |
| iree | 70 | 37 | 33 | 0 | 72.2 | 15 MB | — |
| llamacpp | 60 | 47 | 13 | 10 | 6.6 | 31 MB | 215 kB |
| gigatoken | 50 | 50 | 0 | 20 | 605.6 | 89 MB | 5230 kB |
| fastokens | 40 | 40 | 0 | 30 | 67.9 | 153 MB | 690 kB |
| executorch | 40 | 40 | 0 | 30 | 3.9 | 47 MB | 1532 kB |
| tiktoken | 30 | 30 | 0 | 40 | 31.9 | 34 MB | 3699 kB |
| ai-tokenizer | 30 | 30 | 0 | 40 | 59.5 | — | 31133 kB |
| wordchipper | 30 | 29 | 1 | 40 | 78.1 | 105 MB | 259 kB |
| mistral-common | 10 | 10 | 0 | 60 | 13.6 | — | 6400 kB |
| minbpe | 10 | 10 | 0 | 60 | 1.0 | — | — |
| blingfire | 10 | 0 | **10** | 60 | 7.8 | 2 MB | 3 kB |

**own-cells median is NOT cross-comparable** — it is each engine measured on
whatever subset it supports. Use the ranking table for comparisons.

**kitoken is the only engine besides the reference that runs all 70 cells with
correct ids**, covering BPE, Unigram and WordPiece, at 2.6× the reference with
the smallest package in the set. Every engine above it in the ranking buys its
speed by supporting less.

### pipeline vs gigatoken

Measured head-to-head in one process, median of 7, idle machine — the only way
this question can be answered:

| | pipeline | gigatoken |
|---|---:|---:|
| median | **881 MB/s** | 590 MB/s |
| cells won | **15 / 20** | 5 / 20 |
| gpt2 | 698 | 554 (1.26× pipeline) |
| llama-3 | 947 | 612 (1.55× pipeline) |

The split is by script. gigatoken wins Latin and dense text (english 0.87×, code
0.82×); pipeline wins everything else, decisively (chinese 1.90×, korean 1.75×,
arabic 1.67×, greek 1.60×, russian 1.55×, thai 1.53×).

pipeline pays one cost gigatoken structurally does not: tk-encode exposes no
flat-`u32` entry point, so the adapter restates `Vec<PipelineToken>` as `u32`
while gigatoken writes straight into the caller's buffer. Measured by building
pipeline once with that copy removed: **+1% to +8%, median ~4%** — real, charged
to pipeline as an API-forced cost, and not the story.

### Multi-thread scaling

Same discipline: only engines whose ids **match** on the cell, and only the
cells all of them verify — llama-3/english and mistral-nemo/english.
Performance cores only; efficiency is against perfect linear from 1 thread.

| engine | 1t | 2t | 4t | 8t | 8t efficiency | × ref @8t |
|---|---:|---:|---:|---:|---:|---:|
| gigatoken | 1275 | 2480 | 4845 | **9517** | 93% | 200× |
| pipeline | 868 | 1686 | 3301 | 6483 | 94% | 137× |
| tokie | 119 | 230 | 448 | 865 | 91% | 18× |
| wordchipper | 117 | 231 | 448 | 778 | 85% | 16× |
| iree | 84 | 167 | 328 | 645 | 96% | 14× |
| fastokens | 58 | 112 | 235 | 399 | 86% | 8× |
| tiktoken | 37 | 69 | 135 | 252 | 86% | 5× |
| kitoken | 37 | 70 | 135 | 240 | 80% | 5× |
| llamacpp | 9 | 17 | 31 | 59 | 77% | 1.2× |
| **tokenizers 0.23.1** | 10 | 18 | 32 | 47 | **60%** | 1.0× |

The reference is the worst scaler in the set, and that is the number with the
most consequence: its shared BPE cache serialises threads, so its deficit
*grows* with core count instead of staying constant — 1.0× against pipeline's
137× at eight threads.

blingfire is excluded here. It posts 93% efficiency on gpt2/english, but 0/10
of its cells verify — it emits no whitespace tokens at all, so its curve
describes a different computation.

### Two hazards this benchmark had to solve

**Co-linking engines can silently corrupt an unrelated one.** executorch
statically links its own PCRE2 (`libpcre2-8.a`, 332 exported symbols, plus a
force-loaded `libregex_lookahead.a`); fastokens depends on the `pcre2` crate and
its speed rests on PCRE2 JIT. Linked into the same binary, executorch's copy
preempts it and **fastokens drops from ~55 MB/s to 3.4 — an 18× degradation with
correct ids**, so no verification gate would ever catch it. Bisected: fastokens
measures 61.5 / 56.3 / 61.1 MB/s beside iree / blingfire / llamacpp, and 3.4
beside executorch. executorch is therefore measured in its own process and
merged in, with its ids verified afterwards against the same cell's reference
hash (40/40 match).

**The footprint pass must not interleave with the timed passes.** Each cell's
RSS measurement costs ~12 child processes, each loading a full model; run
between cells, that churn evicts the next cell's warm pages. Footprint now runs
as a second pass over the whole matrix, after all timing is complete.

## Running it

```bash
make fixtures models          # corpora + per-engine model artifacts
make sizes                    # package sizes + binary deltas (optional)
make bench                    # or: cargo run --release -p tokbench --features rust-engines
open dashboard.html           # drop tokenizer_bench_results.json onto it
```

`make bench-open` does the run and opens the dashboard.

Useful flags: `--engine <name>` (repeatable), `--model <name>`, `--reps N`,
`--no-warmup`, `--no-memory` (skips the per-engine RSS child processes, which
roughly halves wall time).

## The engines

All sixteen are wired. Where an engine cannot run a cell it returns an explicit
`Unsupported` with the reason, and where its ids disagree with the reference the
cell is marked `differ` and excluded from every ranking.

| engine | language | class | status |
|---|---|---|---|
| [hf-tokenizers](engines/hf-tokenizers) | Rust | native | **wired** — reference + oracle, 4-phase instrumented |
| [pipeline](engines/pipeline) | Rust | native | **wired** — the target encode path, [tokenizers#2279](https://github.com/huggingface/tokenizers/pull/2279) |
| [kitoken](engines/kitoken) | Rust | native | **wired** — BPE + Unigram + WordPiece from one crate |
| [tokie](engines/tokie) | Rust | native | **wired, verified** |
| [tiktoken](engines/tiktoken) | Rust | native | **wired, verified** (needs derived `ranks.tiktoken`) |
| [fastokens](engines/fastokens) | Rust | native | **wired** — rejects tokenizer.json without `model.type` |
| [rust-gems-bpe](engines/rust-gems-bpe) | Rust | native | **wired** — cl100k/o200k only, see note below |
| [sentencepiece](engines/sentencepiece) | C++ | cffi | **wired** — needs `spiece.model`, builds libsentencepiece statically |
| [wordchipper](engines/wordchipper) | Rust | native | **wired** — 29/30 verified; `BpeBacktrack` selector, named in `version` |
| [gigatoken](engines/gigatoken) | Rust | native | **wired** — 50/50 verified; needs nightly + `-Z profile-rustflags`, links libpython |
| [blingfire](engines/blingfire) | C++ | cffi | **wired** — runs, but 0/10 verified: its GPT-2 model emits no whitespace tokens |
| [llamacpp](engines/llamacpp) | C++ | cffi | **wired** — 47/60 verified; slower than HF on every verified byte-level BPE |
| [iree](engines/iree) | C | cffi | **wired** — gpt2 10/10 byte-exact; 841 kB static lib, 9.6 s build, no CMake |
| [executorch](engines/executorch) | C++ | cffi | **wired** — 40/40 verified; must run in its own process (PCRE2 clash with fastokens) |
| [minbpe](engines/minbpe) | Python | subprocess | **wired** — 10/10 verified; the *floor*, not a competitor |
| [mistral-common](engines/mistral-common) | Python | subprocess | **wired** — 10/10 verified (needs `tekken.json`) |
| [ai-tokenizer](engines/ai-tokenizer) | JS | subprocess | **wired** — 30/30 verified; builds its Encoding from `ranks.tiktoken` |

### The controlled comparison

`hf-tokenizers` (released 0.23.1) and `pipeline`
([tokenizers#2279](https://github.com/huggingface/tokenizers/pull/2279), the
bitsplit + batched-model + fused-cache encode path) are the **same project
reading the same `tokenizer.json`**. Every other pairing in this table compares
across projects, where a difference could come from the vocabulary, the
pre-tokenizer, or a different idea of what a token is. This pairing isolates the
encode path itself, which makes it the one row where a speedup is unambiguously
attributable to the optimisation work.

It also makes the `verified` column load-bearing rather than decorative. A
rewritten merge loop and pre-tokenizer is exactly the change that can be fast
and subtly wrong on one script, so a mismatch here is a bug report, not a
benchmark result.

`pipeline` is timed through `encode_fast` (ids only); the reference goes through
`encode`, which also builds offsets, word ids and an attention mask. Neither is
silently equalised — the reference declares that extra work in `also_computes`,
and the dashboard prints it next to the number, so part of the gap is visibly
offset bookkeeping rather than raw encode speed.

Note that this engine tracks a **branch**, not a tag, because the PR is open.
Pin `rev = "..."` in `engines/pipeline/Cargo.toml` before quoting its number
anywhere durable.

Two more notes worth reading before trusting any row:

- **rust-gems `bpe` has no pre-tokenizer.** `encode_via_backtracking` consumes a
  whole document, so on its own its ids differ from every real tokenizer's.
  Only `bpe-openai`, which adds the split regex, is verifiable — so that is the
  only configuration benchmarked.
- **The `blingfire` crate cannot produce token ids.** It exposes `text_to_words`
  and `text_to_sentences`, which return strings. Benchmarking those against
  subword tokenizers would be the most misleading thing this repo could print.

## The dashboard

`dashboard.html` is a single file — drop `tokenizer_bench_results.json` on it.
The matrix is engine × model × corpus, which does not fit in a bar chart, so
the three main views are heatmaps:

- **engine × model** — median across every language; click a cell to focus it.
- **engine × corpus** for one model — where script coverage shows. CJK, Thai
  and Arabic behave nothing like Latin prose.
- **model × corpus** for one engine — one engine's whole surface at once.

Colour is **log-scaled**, because throughput here spans 1.7 → 1520 MB/s and a
linear ramp collapses everything except the winner into one dark cell. Cells
whose ids disagree with the reference are drawn with a red border and an ✗, and
"Hide unverified" removes them; they are never silently coloured as if they
were comparable results.

## Interpreted engines

Python and JS engines are **not** embedded via pyo3/N-API. They run in their own
runtime, and [`python/harness.py`](python/harness.py) reproduces the Rust
protocol step for step — same chunk boundaries, same warm-up, same median, same
FNV-1a hash — so their ids can still be verified against the reference.

Embedding would put them in the same process and make them *look* directly
comparable while charging every call a binding cost that belongs to the binding,
not the tokenizer. Running them as their users run them, and labelling the class
`subprocess`, is the honest option. Interpreter start-up is excluded (it lands in
`load_ms`); Python's per-call overhead is included, because a Python user pays it.

## Footprint

Two different questions, both reported, neither a substitute for the other:

- **`crate_size_kb`** — the published package you download (crates.io `.crate`,
  PyPI wheel, npm unpacked). From `scripts/package_size.py`.
- **`binary_delta_kb`** — stripped bytes added to a minimal program over a
  no-engine baseline. From `scripts/binsize.sh`.
- **`rss_delta_mb`** — resident memory once loaded and warmed, measured in a
  **dedicated child process per engine**. In one process the allocator hands
  engine B the pages engine A freed, reporting B's footprint as ~0.

## Output

The driver writes `tokenizer_bench_results.json`:

```json
{
  "dataset_metadata": { "file_size_bytes": 1048576, "total_characters": 1000000,
                        "corpus": "eng_Latn", "model": "gpt2", "reps": 5, "warmup": true },
  "results": [
    { "tokenizer_name": "tokie", "total_tokens_produced": 245277,
      "mean_execution_time_seconds": 0.0028,
      "breakdown_nanoseconds": { "normalization": 0, "pre_tokenization": 0,
                                 "core_encoding": 0, "post_processing": 0 },
      "engine_class": "native", "verified": true, "ids_hash": "47cdd1399a60de5a",
      "rss_delta_mb": 45.4, "crate_size_kb": 181.0 }
  ],
  "runs": [ "...one entry per model x corpus cell..." ]
}
```

`breakdown_nanoseconds` is **omitted** for engines whose API cannot separate its
stages. The dashboard renders that as "not instrumented" rather than inventing a
split, because an invented split is indistinguishable from a measured one once
it is a coloured bar.

## Adding an engine

Create `engines/<name>/`, implement two traits, add one line to
`driver/src/registry.rs`:

```rust
impl Build for Adapter {
    fn build(model: &Model) -> Result<Box<dyn Engine>, Unsupported> { ... }
}
impl Engine for Adapter {
    fn info(&self) -> Info { ... }                      // version, class, disclosures
    fn encode(&mut self, text: &str, out: &mut Ids) { } // the only timed call
    fn phases(&mut self, text: &str) -> Option<Phases> { None }  // optional
}
```

Use the library's ordinary public API — the one a user would call. If it forces
an allocation or a type conversion, that cost stays in the measurement, because
the user pays it too. Reaching into private internals to skip work the public
path performs is out of bounds.

## No benchmarking framework

Deliberate, and it is *less* code, not more. `divan` has no machine-readable
output (JSON/CSV is still a planned feature). `criterion`'s `estimates.json` is
documented as a private implementation detail that may change without warning,
needs `cargo-criterion` plus `harness = false`, and owns `fn main()`. Neither
expresses an engine × model matrix, and neither verifies that two engines
produced the same ids — the property the whole comparison rests on. Wrapping
either would mean parsing its output back into this schema. The measurement is
~25 lines in `tokbench_core::measure`, shared by every engine.

## Licence

Apache-2.0. Each engine remains under its own licence; this repository vendors
none of them.
