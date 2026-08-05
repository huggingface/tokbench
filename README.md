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

A real run on `gpt2` / English prose, 1.0 MB chunked, single thread, median of
5, warm cache, `add_special_tokens = false`, Apple M-series:

| engine | MB/s | ns/B | vs ref | ids | RSS | package |
|---|---:|---:|---:|:--:|---:|---:|
| tokie 0.1.4 | 423.1 | 2.25 | 57.4× | match | 43.2 MB | 181 kB |
| tiktoken-rs 0.12.0 | 22.3 | 42.8 | 3.0× | match | 12.7 MB | 3699 kB |
| pipeline ([#2279](https://github.com/huggingface/tokenizers/pull/2279)) | 10.7 | 89.3 | 1.4× | match | 33.2 MB | — |
| tokenizers 0.23.1 (reference) | 7.4 | 129.4 | 1.0× | — | 35.6 MB | 192 kB |

All four emit **245,277 tokens with the identical id hash** `47cdd1399a60de5a`,
which is the only reason the ranking means anything. Note the reference also
computes byte offsets, word ids and an attention mask on the same pass; the
others do not. That is a real part of the gap, and it is disclosed rather than
subtracted.

This is one model on one corpus on one machine, and it is not a league table:
tokie's 57× is on Latin prose with a warm pretoken cache, which is the regime
that suits it best. Run the full matrix before drawing conclusions.

Phase breakdown for the reference on that run:

```
normalization        1.5 µs    0.0%
pre-tokenization    90.5 ms   67.0%   <- the bottleneck
core encoding       30.7 ms   22.7%
post-processing     13.7 ms   10.1%
```

Pre-tokenization costs **2.9× the BPE merge loop**. It is broken out as its own
phase for exactly this reason — folding it into "normalization" would hide the
thing most worth optimising.

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

Status is stated plainly. "Scaffolded" means the folder holds the integration
contract and an explicit `Unsupported` reason — not a guess dressed up as a
measurement.

| engine | language | class | status |
|---|---|---|---|
| [hf-tokenizers](engines/hf-tokenizers) | Rust | native | **wired** — reference + oracle, 4-phase instrumented |
| [pipeline](engines/pipeline) | Rust | native | **wired** — the target encode path, [tokenizers#2279](https://github.com/huggingface/tokenizers/pull/2279) |
| [tokie](engines/tokie) | Rust | native | **wired, verified** |
| [tiktoken](engines/tiktoken) | Rust | native | **wired, verified** (needs derived `ranks.tiktoken`) |
| [fastokens](engines/fastokens) | Rust | native | **wired** — rejects tokenizer.json without `model.type` |
| [rust-gems-bpe](engines/rust-gems-bpe) | Rust | native | **wired** — cl100k/o200k only, see note below |
| [sentencepiece](engines/sentencepiece) | C++ | cffi | **wired** — needs `spiece.model`, builds libsentencepiece statically |
| [wordchipper](engines/wordchipper) | Rust | native | scaffolded — needs an explicit `SpanEncoderSelector` |
| [gigatoken](engines/gigatoken) | Rust | native | scaffolded — pin a git rev; watch its thread count |
| [blingfire](engines/blingfire) | C++ | cffi | scaffolded — needs `TextToIds`, **not** the `blingfire` crate |
| [llamacpp](engines/llamacpp) | C++ | cffi | scaffolded — needs a vocab-only GGUF |
| [iree](engines/iree) | C | cffi | scaffolded — best-matched foreign engine, reads tokenizer.json |
| [executorch](engines/executorch) | C++ | cffi | scaffolded — use `HFTokenizer`, name the variant |
| [minbpe](engines/minbpe) | Python | subprocess | runner written — the *floor*, not a competitor |
| [mistral-common](engines/mistral-common) | Python | subprocess | runner written — needs `tekken.json` |
| [ai-tokenizer](engines/ai-tokenizer) | JS | subprocess | runner written — needs a named encoding |

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
