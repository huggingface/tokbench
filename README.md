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

## Contributors

Initial development by @ArthurZucker, @SBrandeis, @McPatate, @LysandreJik
