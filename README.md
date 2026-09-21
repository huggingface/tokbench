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

**Throughput is not a property of the tokenizer. It is a property of the
tokenizer and the text.** Every fast BPE implementation caches pretokens, so
throughput tracks how often the input repeats itself. `pipeline` / gpt2, one
thread, median of 9 — the only thing changed is the cache:

| corpus | cache on | cache off | cache is worth | heap on | heap off |
|---|---:|---:|---:|---:|---:|
| agentic-swe | 332 | 174 | **1.90×** | 7.3 MB | 3.6 MB |
| math-latex | 238 | 188 | 1.26× | 7.2 MB | 3.6 MB |
| chat-llama3 | 307 | 288 | 1.07× | 7.2 MB | 3.6 MB |
| japanese | 77 | 76 | 1.01× | 9.2 MB | 3.7 MB |
| chinese | 88 | 92 | 0.95× | 11.5 MB | 3.7 MB |
| english | 228 | 238 | 0.96× | 7.2 MB | 3.6 MB |

The cache nearly doubles throughput on agent traces, where identifiers, diffs
and indentation recur constantly. On CJK it buys nothing and costs 5–8 MB,
because pretokens barely recur. On English prose it is a **net loss**.

That last row is worth dwelling on, because it is not what a cache is supposed
to do and the obvious explanations are both wrong. It is not a broken ablation
— the heap columns show the cache really is gone, and the driver now refuses to
call a row an ablation unless the cache-free half holds less live heap
(`cache_ablation_verified`; `pipeline-no-cache` once reported the cached engine
for as long as the config reader dropped `cache_capacity` on the floor). And it
is not eviction pressure — growing the table only makes it worse:

| cache slots | 0 | 65 536 | 262 144 | 1 048 576 | 4 194 304 |
|---|---:|---:|---:|---:|---:|
| english MB/s | **238** | 230 | 204 | 176 | 167 |

The cache does work, and it responds to repetition exactly as it should.
Feeding gpt2 synthetic text at a controlled recurrence rate:

| pretoken recurrence | 1.0× | 5.3× | 6.9× | 13.9× | 59.4× | 248× | 3655× |
|---|---:|---:|---:|---:|---:|---:|---:|
| cache is worth | 0.95× | 2.17× | 2.71× | **3.79×** | 3.40× | 2.70× | 2.31× |

So the payoff is `hit rate × (merge cost saved − lookup cost)`, and on real
English prose that product is ≈ 0. The merges this engine skips are cheap
enough that the hash lookup cancels them. **The same cache in front of a slower
merge loop is worth far more** — which is exactly why engines whose design rests
on an unbounded pretoken cache post their best numbers on English, and why that
number describes their merge loop's weakness as much as their cache's strength.
Three things follow:

**An unbounded cache measures repetition, not tokenization.** gigatoken seeds
~50k entries and doubles rather than evicting, and posts extraordinary numbers.
Production traffic is not a corpus you encode twice.

**So no timed pass ever sees text twice.** The corpus is cut into `reps + 1`
disjoint slices; slice 0 warms, each rep gets unseen text. Warming on the
chunks the reps then re-encode overstated gigatoken by **256×** (llama-2/dense)
and barely moved a word-granularity cache — indistinguishable from outside.
`--no-warmup` times the cold first pass.

**So the corpora are mixed, and English is never the headline.** 14 scripts
plus code, maths, agent traces and special-token-dense text. On llama-3:

| engine | english | greek | chinese | english/chinese |
|---|---:|---:|---:|---:|
| pipeline | 172 | 90 | 49 | 3.5× |
| wordchipper | 89 | 50 | 44 | 2.0× |
| tokie | 79 | 21 | 9 | **8.5×** |

An English-only benchmark ranks these three in an order that reverses on CJK.

**Equal ground.** Cross-engine medians are computed only over cells every
compared engine ran and verified, or the median rewards engines that skip the
hard cases.

**And they are public.** The corpora live in
[huggingface/tokbench-corpora](https://huggingface.co/datasets/huggingface/tokbench-corpora),
as parquet for reading and as the byte-identical `.txt` the driver chunks. A
config name there is exactly a `--corpus` argument here.

```python
load_dataset("huggingface/tokbench-corpora", "japanese")
```

`make fixtures` pulls the text; `CORPORA_REVISION=<sha>` pins it to a commit.
Two corpora are not redistributable and stay in the internal repo, which
`make fixtures` still reaches for: `code-mixed` carries copyleft source
verbatim, and `agentic-traces` has no recorded provenance. The dataset card
lists provenance and licence per corpus.

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

**Decode** reports two rates: `decode_mbps` (text produced, same axis as
encode) and `decode_ns_per_token` (input-side cost — the one to compare when
two engines emit different amounts of text from the same ids). The decoded
*text* is hashed against the reference's decoded text, not against the original
corpus: a lowercasing normalizer makes `decode(encode(t)) != t` for a correct
tokenizer. No decode entry point means `decode_unsupported`, absent from the
ranking rather than scored zero in it.

**Scaling** labels who supplied the threads. `--scaling-mode auto` drives the
engine's own threads where it has them and otherwise runs independent
instances:

| label | meaning |
| --- | --- |
| `native-threads` | one engine told to use *n* threads, handed one batch. What a batch caller gets. |
| `native-threads`, `n/a @ 0T` | the library fans out but exposes no width knob, so one point at the width it chose. Not a curve, and none is invented. |
| `independent-instances` | *n* single-threaded instances in one process, fed by a shared cursor. |

**Padding is an axis, not a footnote.** `--padding off|longest|both` (both by
default). It is a large, uneven, mandatory cost for anyone feeding rectangular
tensors. The harness never pads on an engine's behalf: no native padding means
`no native padding`, not an unpadded number wearing a padded label.

**Cache ablation.** `--cache-capacity N` sizes the tokenizers v1 BPE cache,
`0` disables it, omitting it keeps the upstream 65 536. Every engine is also
registered as `<name>-no-cache`; one with no way to disable its cache reports
`unsupported` there rather than a number that invites a wrong subtraction. When
both halves of a pair are measured, `cache_ablation_verified` records whether
the cache-free half actually held less live heap — a row that did not is not an
ablation result, whatever it measured.

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
`tokenizer.json`**, so the gap between them is the encode path and nothing
else. Every other pairing compares across projects, where a gap could come from
the vocabulary or a different idea of what a token is. It also makes the
verification column load-bearing: a rewritten merge loop is exactly the change
that is fast and subtly wrong on one script.

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
- **`heap_load_mb` / `heap_encode_mb`** — live heap after load, and after
  encode with caches populated. One child process per engine, because in one
  process the allocator hands engine B the pages engine A freed. Live heap, not
  RSS: RSS is a high-water mark that bills a loader for what it already freed.

The footprint pass runs after all timing. Interleaved, its ~12 child processes
per cell evict the next cell's warm pages.

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
      "heap_load_mb": 3.5, "heap_encode_mb": 7.2, "crate_size_kb": 181.0 }
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

Use the library's ordinary public API — the one a user would call. An
allocation or conversion it forces stays in the measurement, because the user
pays it too. Reaching into private internals to skip work the public path does
is out of bounds.

`decode` may decline: the default returns `Unsupported`. Return `Err` rather
than pushing a short string when an id cannot be mapped, or a truncated `out`
hashes as a fast, wrong decode.

## No benchmarking framework

Deliberate, and it is *less* code. `divan` has no machine-readable output;
`criterion`'s `estimates.json` is a documented private implementation detail
and owns `fn main()`. Neither expresses an engine × model × corpus matrix, and
neither verifies that two engines produced the same ids — the property the
whole comparison rests on.

## Licence

Apache-2.0. Each engine remains under its own licence; this repository vendors
none of them. The corpora are excerpts of public datasets under their own
licences — mostly ODC-By and MIT — listed per corpus in
`scripts/corpora_card.md`, which is the card published with the dataset.

## Contributors

Initial development by @ArthurZucker, @SBrandeis, @McPatate, @LysandreJik
