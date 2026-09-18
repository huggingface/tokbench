//! gigatoken (marcelroed) — Rust BPE whose speed comes from a hand-written
//! SWAR pre-tokenizer and a pretoken cache rather than a faster merge loop.
//!
//! Not published on crates.io (the PyPI `gigatoken` wheel is the distributed
//! artifact), so it enters as a git dependency pinned to an exact rev. The
//! package is `gigatoken`, its `[lib] name` is `gigatoken_rs`; Cargo.toml
//! renames it so the `use` paths below say what they mean.
//!
//! # Three build requirements a reader will hit
//!
//! Both extra flags are needed, every time:
//!
//! ```text
//! cargo +nightly -Z profile-rustflags <cmd> --features gigatoken
//! ```
//!
//! * **Nightly.** gigatoken's `src/lib.rs` starts with
//!   `#![feature(portable_simd)]`. A git dependency's `rust-toolchain.toml` is
//!   not honoured — rustup resolves the toolchain from the directory `cargo`
//!   is invoked in — so the whole tokbench workspace must be built with
//!   `cargo +nightly` whenever the `gigatoken` feature is on.
//! * **`-Z profile-rustflags`.** gigatoken's manifest sets `rustflags` under
//!   `[profile.profiling]`, which is still an unstable Cargo feature. Cargo
//!   parses a git dependency's entire manifest — profiles included, even
//!   though a dependency's profiles are then ignored — so without the flag the
//!   build fails during *resolution*, before one crate is compiled, with
//!   "feature `profile-rustflags` is required". gigatoken's own repo turns it
//!   on in its `.cargo/config.toml`, which does not travel with the dependency.
//! * **libpython.** `pyo3` is a non-optional dependency of gigatoken and the
//!   `extension-module` feature is not enabled, so pyo3's build script emits
//!   link flags for the CPython shared library. A cdylib for Python does not
//!   need that; a plain Rust *binary* does, and the tokbench driver is one.
//!   The link succeeds wherever a shared CPython exists, but it bakes an
//!   absolute path to that particular `libpython3.x.dylib` into the tokbench
//!   binary — `otool -L` shows it — so the binary stops working if that
//!   interpreter moves. This is a property of gigatoken's manifest, not of
//!   this adapter, and it is why the engine sits behind an off-by-default
//!   cargo feature.
//!
//! # Fairness
//!
//! * **Threads — two paths, and both are now measured.** gigatoken has two
//!   encode paths. `encode` here still calls the *serial* per-document one —
//!   `Tokenizer::encode_with_added_tokens_flat` for byte-level BPE,
//!   `SentencePieceBPE::encode_raw_cb` for byte-fallback — which is the same
//!   code gigatoken's own `encode_st` ("encode, single thread") bench
//!   measures, and which never touches the rayon pool. That keeps the encode
//!   row a one-core number directly comparable with every other row.
//!
//!   `encode_batch` calls the other one: `encode_docs_ragged` /
//!   `sp_encode_docs_ragged`, which fan documents (and even a single large
//!   document, split at pretoken-safe boundaries) across rayon. That is where
//!   the advertised "GB/s" comes from, and the scaling sweep drives it, so
//!   `internally_parallel` is now reported from the width actually installed
//!   rather than hardcoded `false`.
//!
//!   One caveat on the 1-thread point of that curve: gigatoken's own serial
//!   batch path, `encode_docs_ragged_serial`, is not re-exported at the crate
//!   root, so the adapter cannot reach it. A one-thread rayon pool stands in,
//!   which pays rayon dispatch the true serial path would not — so the
//!   1-thread number is, if anything, slightly pessimistic.
//!
//! * **The scaling corpus has to be large, or this engine cannot be measured
//!   at all.** `chunk_target_bytes` floors chunks at `MIN_CHUNK_BYTES` = 1 MiB
//!   and `encode_chunks_gathered` caps tasks at
//!   `current_num_threads().min(chunks.len())`. tokbench's ~4.3 MB scaling
//!   batch is therefore four chunks at *every* thread count, and the curve
//!   comes out flat — 365, 348, 309, 326 MB/s at 1/2/4/8 threads on
//!   gpt2/english. That flatness is the corpus, not the engine: on 13 MB of
//!   distinct English the same build does 378 -> 627 -> 866 MB/s over 1/2/4
//!   threads. Any engine with a coarse internal chunk floor needs the bigger
//!   batch before its curve means anything.
//! * **The pretoken cache is not "the main source of the speedup", it is
//!   very nearly the whole of it.**
//!
//!   ONLY THE `default` COLUMN BELOW IS A THROUGHPUT NUMBER. The other two are
//!   an ablation: `disabled` needs a patched build of the crate, and `primed`
//!   encodes the measured text before timing it, which is the one thing
//!   `measure_scaling` exists to avoid (it warms on a disjoint sixth, and
//!   `reused_text` flags any cell that replays input). `primed` is an upper
//!   bound on a cache that has already seen your corpus. Nothing should ever
//!   be quoted from it.
//!
//!   gpt2, one thread, 14 MB of deduplicated English (100% unique lines):
//!
//!   | cache state | gigatoken | pipeline |
//!   | --- | --- | --- |
//!   | disabled (patched crate) | 60 | 172 |
//!   | **default** | **400** | **244** |
//!   | primed (ablation, not throughput) | 741 | 432 |
//!
//!   So the cache is worth 6.7x to gigatoken and 1.4x to the pipeline, and
//!   bypassed entirely gigatoken is 2.9x *slower*. A gigatoken number without
//!   its cache state stated is not a result.
//!
//!   Part of that is structural rather than tuning: gigatoken seeds its
//!   pretoken cache with ~50k vocab entries at construction (see
//!   `ShortPretokenCache::with_at_least`) and the table doubles rather than
//!   evicting (`fn grow`), so it starts near a ~99.3% hit rate and keeps
//!   everything it sees — its merge path is annotated as running on "~0.7% of"
//!   pretokens. The pipeline's `WordCache` starts empty and evicts at a fixed
//!   65,536 slots.
//!
//!   **Most of the English gap is that size difference, not the design.** Cap
//!   both at the same slot count (gpt2, one thread, realistic warm) and it
//!   very nearly closes:
//!
//!   | slots | gigatoken | pipeline | giga/pipeline |
//!   | --- | --- | --- | --- |
//!   | 65,536 | 259.5 | 230.8 | **1.12x** |
//!   | 262,144 | 387.3 | 239.4 | 1.62x |
//!
//!   At equal memory gigatoken's 1.47x lead becomes 1.12x.
//!
//!   The 262,144 row is NOT a design difference, and an earlier version of
//!   this file said it was ("gigatoken turns capacity into throughput and the
//!   pipeline does not", from +49% against +4%). Two later measurements say
//!   otherwise:
//!
//!   * The pipeline's cache is already at a **95.87% hit rate** -- instrumented
//!     over 7.18M lookups on this corpus: 84.80% resolve from the home slot,
//!     11.06% displaced, **4.13% miss**. There is no room for capacity to buy
//!     +49%, because there are only 4% of lookups left to win.
//!   * Refusing to evict changes nothing. Turned into a bounded first-come
//!     table (ids identical), one thread on gpt2 reads 242.4 -> 245.9 at 16k
//!     slots, 241.6 -> 241.8 at 65k and 232.0 -> 247.5 at 262k. Signs go both
//!     ways, the spread is inside this cell's ~15% run variance, and the
//!     largest delta sits at the capacity with the *least* eviction pressure.
//!
//!   The likelier reading of gigatoken's +49% is that it is recovering from its
//!   own vocab seed: ~50k entries into a 65,536-slot budget leaves ~15k for the
//!   corpus, so the 65,536 row understates it and the 1.12x there is the fairer
//!   number, not the 1.62x.
//!
//!   What this adds up to is that the pipeline's word cache is close to maxed
//!   out, and no cache-side change reaches a 2x. Every lever tried landed in
//!   the noise: capacity +4%, eviction ~0%, one-cache-line probing capped at
//!   the 11.06% displaced lookups, four inline ids capped at the 3.79% of words
//!   that encode to exactly four (ported, byte-exact, +8% at one thread, then
//!   reverted), and prefetch and huge pages dead because throughput is flat
//!   from an 8 MB table to a 128 MB one.
//!
//!   Caveat on the method: capping gigatoken makes it a bounded *first-come*
//!   cache (inserts refused at the 3/4 load where it would have doubled) while
//!   the pipeline evicts on collision. Equal memory, different retention. On a
//!   Zipf workload first-come is roughly fine -- frequent items arrive early --
//!   but the ~50k vocab seed eats most of a 65,536-slot budget before the
//!   corpus gets any, which if anything understates gigatoken on that row.
//!
//!   **And it is a property of the language, not of the engine.** English
//!   pretokens recur about 8x (105,627 unique of 876,601); Chinese ones about
//!   1.3x (53,237 of 69,863, over 7,007 distinct characters against 350). So a
//!   growing cache fills itself on English in one pass and cannot on Chinese.
//!   gpt2, one thread, `chinese.txt`:
//!
//!   | cache state | gigatoken | pipeline |
//!   | --- | --- | --- |
//!   | **default** | **61.9** | **85.2** |
//!   | primed (ablation) | 221.4 | 86.4 |
//!
//!   On Chinese the pipeline's cache is worth nothing at all (1.01x) and
//!   gigatoken's is worth 3.58x — so with a realistic warm gigatoken drops
//!   *below* the pipeline, which wins by 1.37x. The English advantage is real
//!   but it belongs to the corpus.
//!
//!   Equal slot counts change nothing there, which is the point: neither
//!   engine can cache what does not recur.
//!
//!   | slots | gigatoken | pipeline | giga/pipeline |
//!   | --- | --- | --- | --- |
//!   | 65,536 | 56.3 | 86.0 | 0.65x |
//!   | 262,144 | 61.0 | 84.6 | 0.72x |
//!
//!   Four times the slots is +8% for gigatoken and -2% for the pipeline. Ids
//!   verified against `tokenizers 0.23.1` in all four cells.
//!
//!   The cache lives on the `Tokenizer`/`EncodeState` this struct owns, so it
//!   persists across `encode` calls. `tokbench_core::measure` walks the whole corpus once
//!   untimed before the clock starts, so the timed passes always see a warm
//!   cache — the regime a real `for doc in corpus` loop reaches, and the one
//!   every number from this engine describes. `--no-warmup` gives the cold
//!   contrast.
//!
//!   That was not true when this engine was first wired, and the bug was found
//!   here because this is the engine it distorted most: both branches of
//!   `measure` were byte-for-byte identical, so the "cold" path still made a
//!   full id-collecting pass that filled the cache before the clock started
//!   (gpt2/english measured 1401 MB/s `--no-warmup` against 1372 MB/s warm —
//!   noise, when the true gap should be large). `measure` now times only the
//!   first pass when warm-up is off, and collects ids afterwards.
//!
//! # Why `encode_with_added_tokens_flat` and not `memoized_encode_flat`
//!
//! `memoized_encode_flat` takes pre-split pretokens and skips added-token
//! handling — it is faster, and it is the wrong answer. The reference engine
//! (`tokenizers`, called with `add_special_tokens = false`) still splits the
//! input on added tokens before the model runs. `encode_with_added_tokens_flat`
//! is gigatoken's equivalent — its own docs say it "mirrors the full
//! HuggingFace `tokenizers` encode pipeline" — so it is the entry point whose
//! ids can legitimately be checked against the oracle. Picking the cheaper
//! call would buy throughput by doing less work, which rule 1 exists to catch.
//!
//! # Coverage
//!
//! `load_hf_slice` reads the same `tokenizer.json` as the reference and picks
//! its own backend from the model's `byte_fallback` flag: byte-level BPE, or a
//! SentencePiece-style BPE. It refuses WordPiece and Unigram by name, which
//! surfaces here as an ordinary [`Unsupported`] cell rather than a wrong
//! answer.

// The real adapter, compiled only when the vendored checkout is wired in;
// see Cargo.toml for why it cannot simply be a normal dependency.
#[cfg(gigatoken_wired)]
mod wired {
    use gigatoken_rs::load_tokenizer::hf::{load_hf_slice, HfTokenizer};
    use gigatoken_rs::EncodeState;
    use tokbench_core::{Build, Class, Engine, Ids, Info, Model, Padding, Unsupported};

    pub struct Adapter {
        /// gigatoken's own batch worker pool, forked per rayon slot. Held for
        /// the life of the engine because the pool caches one forked
        /// `Tokenizer` per slot -- rebuilding it per call would measure a
        /// permanently cold pool.
        workers: gigatoken_rs::WorkerPool,
        /// Sized by `set_threads`; `encode_docs_ragged` fans out over rayon's
        /// current pool, so this is how its width is set. `None` means one
        /// thread, which takes gigatoken's dedicated serial path instead.
        pool: Option<rayon::ThreadPool>,
        threads: usize,
        /// Owns the pretoken cache for the BPE backend, so it survives between
        /// `encode` calls exactly as it would in a user's loop.
        tok: HfTokenizer,
        /// The SentencePiece backend keeps its cache outside the model, in a
        /// caller-supplied `EncodeState`. Held here for the same reason: a
        /// per-call `EncodeState::new()` would throw the cache away every
        /// document and measure a permanently cold engine.
        sp_state: EncodeState,
    }

    impl Build for Adapter {
        /// `ShortPretokenCache` is `pub(crate)` with no capacity or bypass knob,
        /// and the ~50k vocab seed is applied unconditionally at construction.
        /// Turning it off needs a patched copy of the crate -- measured that way
        /// (no-op'd `insert`/`insert_at`/`replace`, table left permanently
        /// empty) it runs 60 MB/s against 400 cached and 741 primed, a 12.4x
        /// span. Worth knowing, not shippable.
        fn build_without_cache(_model: &Model) -> Result<Box<dyn Engine>, Unsupported> {
            Err(Unsupported(
                "gigatoken's pretoken cache is pub(crate) with no bypass; disabling it needs a \
                 patched build of the crate"
                    .into(),
            ))
        }

        fn build(model: &Model) -> Result<Box<dyn Engine>, Unsupported> {
            let path = model.tokenizer_json();
            if !path.exists() {
                return Err(Unsupported(format!(
                    "no tokenizer.json in {}",
                    model.dir.display()
                )));
            }
            // `load_hf_slice` (bytes) rather than a path-taking loader: reading
            // the file here keeps the error message ours, and this is all
            // pre-timer work anyway.
            let data = std::fs::read(&path)
                .map_err(|e| Unsupported(format!("cannot read {}: {e}", path.display())))?;
            let tok = load_hf_slice(&data)
                .map_err(|e| Unsupported(format!("gigatoken cannot load this config: {e}")))?;
            Ok(Box::new(Adapter {
                workers: gigatoken_rs::WorkerPool::new(),
                pool: None,
                threads: 1,
                tok,
                sp_state: EncodeState::new(),
            }))
        }
    }

    impl Engine for Adapter {
        fn info(&self) -> Info {
            Info {
                name: "gigatoken",
                // Not a crates.io version: the exact tree that was measured. The
                // manifest says 0.10.0, but 0.10.0 is not a fixed point in a repo
                // without releases, so the rev is the real answer and is what
                // Cargo.toml pins.
                version: "0.10.0 (git 34a1599f0c0ae7d7cd0d1c530e6522320158b360)",
                lang: "rust",
                class: Class::Native,
                url: "https://github.com/marcelroed/gigatoken",
                also_computes: "",
                // The single-document `encode` below is still the serial path.
                // The scaling sweep goes through `encode_batch`, which is
                // gigatoken's own rayon batch engine, and this then reports
                // the width that was actually installed.
                internally_parallel: self.threads > 1,
            }
        }

        fn encode(&mut self, text: &str, out: &mut Ids) {
            // Split the borrow: the SentencePiece call needs `&mut self.tok` and
            // `&mut self.sp_state` at the same time.
            let Self { tok, sp_state, .. } = self;
            match tok {
                // Appends `u32` ids straight into `out` — no intermediate Vec, no
                // conversion. That is the library's own output shape (its batch
                // engine fills chunk buffers the same way), so this is the
                // ordinary public path, not a harness-shaped shortcut.
                HfTokenizer::Bpe(bpe) => bpe.encode_with_added_tokens_flat(text.as_bytes(), out),
                // The SentencePiece backend delivers tokens per unit through a
                // callback and has no flat variant. The `TokenId` -> `u32` map is
                // a no-op at runtime (`#[repr(transparent)]`) but the extend is
                // real work, and it stays in the measurement.
                HfTokenizer::SentencePiece(sp) => {
                    sp.encode_raw_cb(sp_state, text, &mut |ids| {
                        out.extend(ids.iter().copied().map(u32::from));
                    });
                }
            }
        }

        /// Both backends have one: `encode_docs_ragged` for byte-level BPE,
        /// `sp_encode_docs_ragged` for SentencePiece. Both are re-exported at
        /// gigatoken's crate root even though `batch` itself is `pub(crate)`.
        fn has_native_batch(&self) -> bool {
            true
        }

        /// `encode_docs_ragged` fans out over rayon's current pool, so a sized
        /// pool is how gigatoken's batch engine is asked for `threads`.
        ///
        /// One thread is not "a pool of one": gigatoken ships a dedicated
        /// serial path precisely because touching rayon at all (even
        /// `current_num_threads()`) builds the global pool. `encode_batch`
        /// takes that path when `pool` is `None`, which is also the shape its
        /// own `parallel=false` binding promises.
        fn set_threads(&mut self, threads: usize) -> bool {
            if threads == 0 {
                return false;
            }
            // A one-thread POOL, not gigatoken's serial path: that path
            // (`encode_docs_ragged_serial`) is not re-exported at the crate
            // root, so it is unreachable from here. A pool of one still pins
            // the width honestly; it just pays rayon's dispatch, which every
            // other point on the curve also pays.
            self.pool = match rayon::ThreadPoolBuilder::new().num_threads(threads).build() {
                Ok(pool) => Some(pool),
                // Refuse rather than silently use the global pool, whose width
                // would then be reported as `threads`.
                Err(_) => return false,
            };
            self.threads = threads;
            true
        }

        /// gigatoken's own batch engine: chunks documents at pretoken-safe
        /// boundaries, encodes with pooled workers, returns one flat id buffer
        /// plus per-document row lengths. The ids are what tokbench compares,
        /// so the row lengths are dropped -- the work of producing them is not.
        fn encode_batch(&mut self, texts: &[&str], out: &mut Ids) {
            let workers = &self.workers;
            let run = || match &self.tok {
                HfTokenizer::Bpe(bpe) => {
                    let docs: Vec<&[u8]> = texts.iter().map(|t| t.as_bytes()).collect();
                    gigatoken_rs::encode_docs_ragged(workers, bpe, &docs)
                }
                // Takes no `WorkerPool`: it forks its own per chunk.
                HfTokenizer::SentencePiece(sp) => gigatoken_rs::sp_encode_docs_ragged(sp, texts),
            };
            let (flat, _row_lengths) = match &self.pool {
                Some(pool) => pool.install(run),
                None => run(),
            };
            out.extend_from_slice(&flat);
        }

        /// Ragged only, from Rust.
        ///
        /// gigatoken does pad, and its padding is even parallel, but it lives
        /// in the pyo3 layer: `bindings::padding::pad_truncate_matrix` is
        /// private, and the public `encode_batch_matrix` wants a
        /// `Python<'py>` token and `PyAny` inputs. There is no Rust-callable
        /// padded batch path, so the padded cell is reported as unsupported
        /// rather than filled by padding here -- which would measure tokbench's
        /// fill, not gigatoken's.
        ///
        /// Comparing gigatoken padded against the other engines means driving
        /// it through Python, the way `engines/*/run.py` engines already are.
        fn set_padding(&mut self, padding: Padding) -> bool {
            matches!(padding, Padding::Off)
        }
    }
}

#[cfg(gigatoken_wired)]
pub use wired::Adapter;

// Default build: no dependency, no libpython, no nightly — just an honest
// refusal that names what it would take.
#[cfg(not(gigatoken_wired))]
pub struct Adapter;

#[cfg(not(gigatoken_wired))]
impl tokbench_core::Build for Adapter {
    fn build(
        _model: &tokbench_core::Model,
    ) -> Result<Box<dyn tokbench_core::Engine>, tokbench_core::Unsupported> {
        Err(tokbench_core::Unsupported(
            "gigatoken not wired: its manifest declares `cargo-features`, which Cargo \
             rejects in a dependency and which breaks resolution for the whole \
             workspace. Run engines/gigatoken/scripts/enable_gigatoken.sh and build \
             with RUSTFLAGS='--cfg gigatoken_wired' on nightly."
                .into(),
        ))
    }
}
