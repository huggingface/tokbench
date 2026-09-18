# Whose threads, and what a cache is actually worth

Two questions the scaling column could not answer before `Engine::set_threads`
and `Engine::build_without_cache` existed, measured on one machine (M-series,
10 performance cores), gpt2, ragged output, `--reps 5`, ids verified against
`tokenizers 0.23.1` in every cell.

Throughput is MB/s of input. `internal` means one engine instance told to use
*n* threads and handed one batch through its own batch API. `external` means
*n* independent single-thread instances over a shared work-stealing cursor —
what the harness does for a library with no threading of its own.

Both caches are pinned to **65,536 slots** so the comparison is design, not
memory budget.

## english — 14 MB, deduplicated, 100% unique lines

| mode | engine | 1T | 2T | 4T | 8T | speedup |
| --- | --- | --- | --- | --- | --- | --- |
| internal | **pipeline** | 242.7 | 582.0 | 823.6 | **1214.1** | 5.0x |
| internal | gigatoken | 195.5 | 352.4 | 509.8 | 508.6 | 2.6x |
| external | **pipeline** | 273.3 | 490.1 | 888.6 | **1609.8** | 5.9x |
| external | gigatoken | 206.2 | 399.1 | 688.3 | 1289.0 | 6.3x |

## chinese — 29.4 MB of `wikimedia/wikipedia` `20231101.zh`, 100% unique lines

| mode | engine | 1T | 2T | 4T | 8T | speedup |
| --- | --- | --- | --- | --- | --- | --- |
| internal | **pipeline** | 98.7 | 264.3 | 405.5 | **578.8** | 5.9x |
| internal | gigatoken | 60.7 | 118.8 | 207.9 | 314.8 | 5.2x |
| external | **pipeline** | 103.3 | 198.3 | 386.8 | **728.9** | 7.1x |
| external | gigatoken | 65.1 | 123.3 | 244.8 | 443.2 | 6.8x |

## The pool is worth about as much as sharding, at a fraction of the memory

Independent workers are the throughput ceiling, and it is not close for every
engine: gigatoken goes 508.6 -> 1289.0 on english at eight threads, 2.5x, and
97.9 -> 413.2 on the small Chinese corpus. Nothing is shared, so there is no
plan to build, no completion queue to drain and no gather.

The pipeline's own pool gets within **1.33x** of that ceiling on english
(1214.1 against 1609.8) and **1.26x** on Chinese (578.8 against 728.9), while
external mode pays for it in memory: *n* instances means *n* vocabularies and
*n* caches, each warming separately. At eight threads on a 256k-vocab model
that is the difference between one resident model and eight.

So "use the engine's own parallelism" costs the pipeline about a quarter of
peak throughput and saves most of the memory. For gigatoken the same choice
costs 2.5x, which is worth knowing before reading either engine's scaling
column: **the mode is not a detail, it can invert the ranking.** At its own
unbounded cache size gigatoken leads at one thread on english; at equal cache
the pipeline leads every cell in both tables.

## Corpus size gates gigatoken before anything else does

`chunk_target_bytes` floors chunks at `MIN_CHUNK_BYTES = 1 MiB` and
`encode_chunks_gathered` caps tasks at `current_num_threads().min(chunks.len())`.
A batch smaller than roughly 1 MiB per thread therefore cannot be split, and
the curve comes out flat for reasons that belong entirely to the corpus:

| corpus | measured | chunks | gigatoken internal speedup |
| --- | --- | --- | --- |
| english 5.2 MB | 4.3 MB | 4 | **1.0x** (365 -> 326) |
| chinese 5.2 MB | 4.3 MB | 4 | **1.6x** (60.2 -> 97.9) |
| english 14 MB | 11.7 MB | 11 | 2.6x |
| chinese 29.4 MB | 24.5 MB | 23 | **5.2x** |

Both "gigatoken does not scale" readings were this, not the engine. Given a
corpus that clears the floor its pool scales 5.2-6.3x, alongside everyone else.
tokbench's default scaling corpus is well under the floor, which is the single
most important thing to fix about the sweep.

## What gigatoken's cache is

It is memoised merge work, and that is the whole of it. The pretoken cache maps
a pretoken to the ids it encodes to, so a hit skips the BPE merge entirely --
gigatoken's own merge path is annotated as running on "~0.7% of" pretokens.
Bypass the cache (patched crate, `insert`/`insert_at`/`replace` made no-ops so
the table stays empty) and the engine is its raw merge loop:

| gpt2, one thread, 14 MB english | gigatoken | pipeline |
| --- | --- | --- |
| cache disabled | **60** | **172** |
| default | 400 | 244 |
| primed (ablation, not a throughput figure) | 741 | 432 |

The cache is worth 6.7x to gigatoken and 1.4x to the pipeline, because the
pipeline's merge loop is 2.9x faster to begin with. One engine is a fast merge
with a modest cache in front of it; the other is a slow merge with a very good
cache in front of it. That also explains the language dependence: English
pretokens recur about 8x (105,627 unique of 876,601) and Chinese ones about
1.3x (53,237 of 69,863), so the cache that carries gigatoken cannot be filled
on Chinese and the ranking there follows the merge loops instead.

`primed` encodes the measured text before timing it, which is the one thing
`measure_scaling` exists to avoid. It is an upper bound on a cache that has
already seen your corpus, never a throughput number.

## Reproducing

The mode switch is a harness capability (`Engine::set_threads`), so the
internal/external rows need no patching. The cache rows do: disabling
gigatoken's cache and capping its slot count both need a patched build, since
`ShortPretokenCache` is `pub(crate)` with no knob, and the pipeline's slot count
is reachable only because `tk-serialize` now reads `cache_capacity` from the
config. None of those ablation switches are in this repository.

Single cells on this machine carry roughly +-15% run to run; ratios below about
1.2x should not be read as differences.
