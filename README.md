# tokbench

A fair benchmark of tokenizer implementations.

One folder per engine, one Rust driver on top, **one timing loop per
direction**, and an id-verification gate so a number is never published for an
engine that quietly computed something different.

```
core/          the measurement contract and the timing loops
driver/        the binary: runs the matrix, verifies, writes JSON
binsize/       a minimal program linking one engine, to weigh it
engines/<name> one adapter per engine
hf-jobs/       reproducible cloud runs on Hugging Face Jobs
dashboard.html single-file dashboard; drop the JSON onto it
```

## Why another benchmark

Most tokenizer comparisons are not comparisons. They time different functions
(one engine computes byte offsets, another does not), on different
vocabularies, with load time folded into encode time, at different thread
counts, and without ever checking that the two produced the same token ids.
Every one of those makes a fast engine look faster than it is.

tokbench fixes the measurement, not the result:

1. **Same work, verified.** Every engine's id stream is hashed (FNV-1a) and
   compared against the reference. Different ids, and the cell is marked
   `differ` and never ranked. Being fast at the wrong answer is not a win.
2. **Same clock, in-process.** Engines are called through one `Engine::encode`
   and timed by the same `Instant`. No subprocess, no IPC, no interpreter.
3. **Load is not encode.** Vocabulary parsing and automaton construction happen
   before the timer and are reported separately as `load_ms`.
4. **No timed pass ever re-encodes text.** See below — this is the one that
   decides whether you are measuring a tokenizer or a cache.
5. **One thread by default, and whose thread is stated.** Curves are labelled
   `native-threads` or `independent-instances`; the two are not the same claim.
6. **Disclose the extra work.** An engine that also computes byte offsets keeps
   that cost in its number, and the report says so next to it.
7. **Decode is timed over the reference's ids**, never over each engine's own
   encode output — otherwise an engine that merges harder feeds itself fewer,
   longer tokens and posts a better token rate for strictly less work.

## The corpus is the experiment

This is the part that matters most, and the part that is easiest to get wrong.

**A tokenizer's throughput is not a property of the tokenizer.** It is a
property of the tokenizer *and the text*. Every fast BPE implementation caches
pretokens — it looks up "have I already merged this word?" before doing any
work — so throughput tracks how often the input repeats itself. Measured here
on `pipeline` / gpt2, sweeping only the cache size and changing nothing else:

| cache slots | english | code | japanese | chinese |
|---|---:|---:|---:|---:|
| 0 (disabled) | 237 | 204 | 74 | 92 |
| 1 024 | 200 | 235 | 76 | 95 |
| 65 536 (default) | 220 | **307** | 72 | 96 |
| **cache is worth** | ~1.0× | **1.50×** | ~1.0× | ~1.0× |

Same engine, same vocabulary, same clock. The cache is worth 50% on source
code, where identifiers and indentation recur constantly, and **nothing at all
on Chinese and Japanese**, where pretokens barely recur — there are only so
many distinct words in English prose, and effectively no reusable pretokens in
unsegmented CJK. (English sits inside this cell's ~10% run-to-run variance.)

Three consequences follow, and they are the whole design:

**1. An unbounded cache is a benchmark result, not a product.** An engine that
never evicts — gigatoken seeds ~50k vocab entries at construction and its table
doubles rather than evicting — posts extraordinary numbers, and they are real
numbers about a real cache. But they describe a workload that has already been
seen. Production traffic is not a corpus you encode twice. The moment the text
stops repeating, the cache stops paying, and the ranking inverts: on CJK the
same engines that dominate English fall behind. Quote a cached number without
its cache state and its corpus, and you have not reported a result.

**2. So the harness never lets a timed pass see text twice.** The corpus is cut
into `reps + 1` **disjoint** slices: slice 0 warms the engine, and every timed
rep gets text the engine has never encountered. That is also the honest regime
— a real server is a warm process handed a new document. An earlier version
warmed on the same chunks it then re-timed, and a document-granularity cache
turned that into a **256× overstatement** (gigatoken on llama-2/dense) while a
word-granularity cache barely moved. Nothing from the outside distinguishes the
two, which is exactly why the harness has to remove the choice.
`--no-warmup` times only the genuinely cold first pass.

**3. So the corpora are mixed on purpose, and English is never the headline.**
`data/fixtures/` carries 14 scripts plus code, maths, agent traces and
special-token-dense text, because the spread between them is larger than the
spread between engines. On llama-3, one engine and one thread:

| engine | english | greek | chinese | english/chinese |
|---|---:|---:|---:|---:|
| pipeline | 172 | 90 | 49 | 3.5× |
| wordchipper | 89 | 50 | 44 | 2.0× |
| tokie | 79 | 21 | 9 | **8.5×** |

An English-only benchmark would rank these three in an order that reverses on
Chinese. Sweep at least one Latin and one CJK corpus before quoting anything.

**Equal ground.** Cross-engine medians are computed **only over cells every
compared engine ran and verified**. Engines cover different model families, so
a median over each engine's own cells silently rewards the ones that skip the
hard cases.

## Measuring

`tokbench` with no subcommand runs the full matrix. `tokbench measure <family>`
runs exactly one measurement and skips the rest. `--engine`, `--model` and
`--corpus` are repeatable; omitting one sweeps every value in that dimension.
`--compare-to` names one shared comparator and reports every target engine as a
multiplier against it.

```console
$ tokbench measure encode --engine pipeline --compare-to hf-tokenizers \
    --model gpt2 --model llama-3 --corpus english --corpus chinese --corpus code
tokbench measure encode: 2 engine(s) × 2 model(s) × 3 corpus/corpora, 5 reps each

model         corpora  median MB/s  median ns/B           vs hf-tokenizers
-------  ------------  -----------  -----------  -------------------------
gpt2     3/3 measured        231.4         4.12   ×35.74 on 3/3 comparable
llama-3  3/3 measured        180.7         5.28   ×23.22 on 3/3 comparable
```

```console
$ tokbench measure decode --engine pipeline --compare-to hf-tokenizers \
    --model gpt2 --model llama-3 --corpus english --corpus chinese
tokbench measure decode: 2 engine(s) × 2 model(s) × 2 corpus/corpora, 5 reps each

model         corpora  median MB/s  median ns/token          vs hf-tokenizers
-------  ------------  -----------  ---------------  ------------------------
gpt2     2/2 measured        314.2              8.2   ×7.79 on 2/2 comparable
llama-3  2/2 measured        367.8             10.1   ×6.89 on 2/2 comparable
```

```console
$ tokbench measure latency --engine pipeline --compare-to hf-tokenizers \
    --model gpt2 --corpus english
model  p50 us  p99 us  samples  p50 vs hf-tokenizers  p99 vs hf-tokenizers
-----  ------  ------  -------  --------------------  --------------------
gpt2     2.25    5.21     1000                ×35.04                ×20.95
```

The five families:

| family | what it answers |
|---|---|
| `encode` | MB/s and ns/byte of text → ids |
| `decode` | MB/s of text produced, and ns per input token |
| `latency` | p50/p99 for one short document — the serving number |
| `scaling` | throughput against thread count, with efficiency |
| `memory` | live heap after load and after encode, in an isolated child |

Every family writes the same JSON schema and keeps all raw per-corpus results;
the table is a median summary. Interactive runs show a progress bar, redirected
output does not. `measure memory` requires exactly one explicit `--corpus`.

**Decode** reports two rates because one number cannot answer both questions:
`decode_mbps` is text produced, on the same axis as encode; `decode_ns_per_token`
is the input-side cost, which is the one to compare when two engines emit text
of different lengths from the same ids. Correctness is checked one level along
— the decoded *text* is hashed against the reference's decoded text, not
against the original corpus, because a lowercasing or accent-stripping
normalizer makes `decode(encode(t)) != t` for a perfectly correct tokenizer.
An engine with no decode entry point reports `decode_unsupported` and is absent
from the decode ranking rather than scored zero in it.

**Scaling** labels who supplied the threads. `--scaling-mode auto` drives the
engine's own threads where it has them and otherwise runs independent
instances:

| label | meaning |
| --- | --- |
| `native-threads` | one engine told to use *n* threads, handed one batch. What a batch caller gets. |
| `native-threads`, `n/a @ 0T` | the library fans out but exposes no width knob, so one point at the width it chose. Not a curve, and none is invented. |
| `independent-instances` | *n* single-threaded instances in one process, fed by a shared cursor. |

**Padding is an axis, not a footnote.** `--padding off|longest|both` (both by
default). Padding is a large, uneven, and mandatory cost for anyone feeding
rectangular tensors, so the unpadded number alone is not usable for serving.
The harness never pads on an engine's behalf: an engine with no native padding
reports `no native padding` rather than an unpadded number wearing a padded
label.

**Cache ablation.** `--cache-capacity N` sizes the tokenizers v1 pipeline BPE
cache; `0` disables it. Omitting it keeps the upstream default of 65 536. The
value is recorded as `dataset_metadata.pipeline_cache_capacity`. Every engine
is also registered a second time as `<name>-no-cache`; engines with no way to
disable their caches report `unsupported` there rather than a number that
invites a wrong subtraction.

## Example results

Apple M3 Max, single thread, median of 5 timed passes over disjoint slices,
warm, `add_special_tokens = false`. gpt2 and llama-3 × 8 corpora = 16 cells.

**Encode**, over the 3 cells every engine below ran and verified:

| engine | median MB/s | × ref | min | max |
|---|---:|---:|---:|---:|
| pipeline | **90** | 10.7× | 49 | 172 |
| wordchipper | 50 | 6.0× | 44 | 89 |
| fastokens | 50 | 6.0× | 49 | 57 |
| tiktoken | 27 | 3.2× | 23 | 32 |
| kitoken | 25 | 3.0× | 22 | 33 |
| tokie | 21 | 2.5× | 9 | 79 |
| tokenizers 0.23.1 <sub>reference</sub> | 8 | 1.0× | 8 | 9 |

**Decode**, over the 5 cells every decode-capable engine verified:

| engine | median MB/s | × ref | ns/token |
|---|---:|---:|---:|
| tokie | **379** | 6.9× | 9.4 |
| pipeline | 337 | 6.1× | 11.2 |
| tiktoken | 257 | 4.6× | 16.9 |
| fastokens | 97 | 1.8× | 43.7 |
| tokenizers 0.23.1 <sub>reference</sub> | 55 | 1.0× | 78.1 |

**Coverage**, which must be read next to the ranking — an engine high in the
table above may be there partly because it declined the hard cells:

| engine | verified | ids differ | unsupported |
|---|---:|---:|---:|
| tokenizers 0.23.1 <sub>reference</sub> | 16 | — | 0 |
| pipeline | 16 | 0 | 0 |
| tiktoken | 16 | 0 | 0 |
| kitoken | 15 | 1 | 0 |
| tokie | 13 | 3 | 0 |
| wordchipper | 12 | 4 | 0 |
| fastokens | 8 | 0 | 8 |
| rust-gems-bpe | 0 | 0 | 16 |

Reproduce with `make bench`, or the exact commands above.

## The engines

Where an engine cannot run a cell it returns an explicit `Unsupported` with the
reason; where its ids disagree with the reference the cell is marked `differ`
and excluded from every ranking. Both are honest outcomes and neither is a
blank.

| engine | language | class | notes |
|---|---|---|---|
| [hf-tokenizers](engines/hf-tokenizers) | Rust | native | the reference and correctness oracle, `tokenizers` 0.23.1 |
| [pipeline](engines/pipeline) | Rust | native | [tk-encode 1.0.0-rc.0](https://github.com/huggingface/tokenizers), the rc0 encode path |
| [kitoken](engines/kitoken) | Rust | native | BPE + Unigram + WordPiece from one crate |
| [tokie](engines/tokie) | Rust | native | |
| [tiktoken](engines/tiktoken) | Rust | native | needs a derived `ranks.tiktoken` |
| [fastokens](engines/fastokens) | Rust | native | rejects a `tokenizer.json` without `model.type` |
| [rust-gems-bpe](engines/rust-gems-bpe) | Rust | native | `bpe-openai` only: plain `bpe` has no pre-tokenizer, so its ids are not comparable |
| [wordchipper](engines/wordchipper) | Rust | native | `BpeBacktrack` selector, named in `version` |
| [gigatoken](engines/gigatoken) | Rust | native | off by default; needs nightly and links libpython, see its `Cargo.toml` |
| [sentencepiece](engines/sentencepiece) | C++ | cffi | needs `spiece.model`; builds libsentencepiece statically |
| [llamacpp](engines/llamacpp) | C++ | cffi | needs a GGUF built by `engines/llamacpp/scripts/make_gguf.py` |
| [iree](engines/iree) | C | cffi | gpt2 only; static lib built by its vendor script |
| [executorch](engines/executorch) | C++ | cffi | must run in its own process: its static PCRE2 preempts fastokens' and silently degrades it ~18× with correct ids, which no verification gate would catch |

`pipeline` and `hf-tokenizers` are **the same project reading the same
`tokenizer.json`**, so the difference between them is the encode path and
nothing else. Every other pairing compares across projects, where a gap could
come from the vocabulary, the pre-tokenizer, or a different idea of what a
token is. That also makes the verification column load-bearing rather than
decorative: a rewritten merge loop is exactly the change that can be fast and
subtly wrong on one script, so a mismatch there is a bug report, not a result.

**Decode** is wired for `hf-tokenizers`, `pipeline`, `tokie`, `tiktoken` and
`fastokens`. The rest report `decode_unsupported`. That is one method per
adapter, and a welcome PR.

## Hugging Face Jobs

[`hf-jobs/`](hf-jobs/) runs the benchmark reproducibly in the cloud: an
immutable container, a pinned input-data revision, several complete
process-level repetitions, a recorded CPU and software environment, and raw
reports plus checksums written to a Storage Bucket. See
[`hf-jobs/README.md`](hf-jobs/README.md).

Fetch a finished Job's reports and open their median in the dashboard:

```bash
make dash BUCKET=huggingface/tokbench-results JOB_ID=<job-id>
```

A hardware flavor does not guarantee two Jobs land on the same physical CPU, so
absolute results stay machine-specific. What Jobs buys is reproducible
invocation and a measure of variance.

## Footprint

Three different questions, none a substitute for another:

- **`crate_size_kb`** — the published package you download. `scripts/package_size.py`.
- **`binary_delta_kb`** — stripped bytes added to a minimal program over a
  no-engine baseline. `scripts/binsize.sh`.
- **`heap_load_mb` / `heap_encode_mb`** — live heap after loading, and after
  encoding with caches populated. Measured in a dedicated child process per
  engine: in one process the allocator hands engine B the pages engine A freed
  and reports B's footprint as ~0. Live heap, not RSS — RSS is a high-water
  mark that never falls, so it bills a loader for an intermediate it already
  freed.

The footprint pass runs after all timing is complete. Interleaved, its ~12
child processes per cell evict the next cell's warm pages.

## Output

The driver writes `tokenizer_bench_results.json`:

```json
{
  "dataset_metadata": { "file_size_bytes": 1048576, "total_characters": 1000000,
                        "corpus": "english", "model": "gpt2", "reps": 5, "warmup": true },
  "results": [
    { "tokenizer_name": "tokie", "total_tokens_produced": 245277,
      "mean_execution_time_seconds": 0.0028, "mbps": 96.7, "ns_per_byte": 9.86,
      "engine_class": "native", "verified": true, "ids_hash": "47cdd1399a60de5a",
      "decode_mbps": 412.7, "decode_ns_per_token": 9.8,
      "decode_text_hash": "b3f1c0a29e4d5107", "decode_verified": true,
      "heap_load_mb": 45.4, "crate_size_kb": 181.0 }
  ],
  "runs": [ "...one entry per model × corpus cell..." ]
}
```

Every field is optional. An engine that could not run a cell carries
`unsupported` with the reason instead of a zero that would sort as "slow".

## Adding an engine

Create `engines/<name>/`, implement two traits, add one line to
`driver/src/registry.rs`:

```rust
impl Build for Adapter {
    fn build(model: &Model) -> Result<Box<dyn Engine>, Unsupported> { ... }
}
impl Engine for Adapter {
    fn info(&self) -> Info { ... }                      // version, class, disclosures
    fn encode(&mut self, text: &str, out: &mut Ids) { } // timed: encode
    fn decode(&mut self, ids: &[u32], out: &mut String) // timed: decode, optional
        -> Result<(), Unsupported> { ... }
}
```

Use the library's ordinary public API — the one a user would call. If it forces
an allocation or a type conversion, that cost stays in the measurement, because
the user pays it too. Reaching into private internals to skip work the public
path performs is out of bounds.

`decode` is the one method that may decline: the default returns `Unsupported`,
which keeps the engine out of the decode ranking instead of scoring it zero
there. Return `Err` rather than pushing a short string when an id cannot be
mapped — a truncated `out` would otherwise hash as a fast, wrong decode.

## No benchmarking framework

Deliberate, and it is *less* code. `divan` has no machine-readable output.
`criterion`'s `estimates.json` is a documented private implementation detail,
needs `cargo-criterion` plus `harness = false`, and owns `fn main()`. Neither
expresses an engine × model × corpus matrix, and neither verifies that two
engines produced the same ids — the property the whole comparison rests on.
The measurement is a few dozen lines in `tokbench_core`, shared by every engine.

## Licence

Apache-2.0. Each engine remains under its own licence; this repository vendors
none of them.

## Contributors

Initial development by @ArthurZucker, @SBrandeis, @McPatate, @LysandreJik
