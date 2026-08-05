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
//! * **Threads — this row is single-threaded, and that is a deliberate
//!   choice.** gigatoken has two encode paths. The headline "GB/s" figures
//!   come from `encode_docs_ragged`/`WorkerPool`, which fans documents (and
//!   even a single large document, split at pretoken-safe boundaries) across
//!   rayon; that is a whole-machine number. This adapter calls the *serial*
//!   per-document path instead — `Tokenizer::encode_with_added_tokens_flat`
//!   for byte-level BPE, `SentencePieceBPE::encode_raw_cb` for byte-fallback
//!   models — which is the same code gigatoken's own `encode_st` ("encode,
//!   single thread") bench measures, and which never touches the rayon pool.
//!   So `internally_parallel` is `false`, honestly: the number here is a
//!   one-core number directly comparable with every other row. It is NOT
//!   gigatoken's advertised throughput, and a writeup quoting it should say
//!   so — the multi-thread axis is where its batch engine belongs.
//! * **The pretoken cache** is the main source of the speedup, and it lives
//!   on the `Tokenizer`/`EncodeState` this struct owns, so it persists across
//!   `encode` calls. `tokbench_core::measure` walks the whole corpus once
//!   untimed before the clock starts, so the timed passes always see a warm
//!   cache — the regime a real `for doc in corpus` loop reaches, and the one
//!   every number from this engine describes. Be careful with the contrast:
//!   `--no-warmup` does NOT produce a cold number here, because that branch of
//!   `measure` still makes a full pass to collect ids for verification, which
//!   fills the cache just as thoroughly (measured on gpt2/english: 1401 MB/s
//!   `--no-warmup` vs 1372 MB/s warm — noise). A genuinely cold figure would
//!   need a fresh engine for each timed pass, which the harness does not do.
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
    use tokbench_core::{Build, Class, Engine, Ids, Info, Model, Unsupported};

    use gigatoken_rs::load_tokenizer::hf::{load_hf_slice, HfTokenizer};
    use gigatoken_rs::EncodeState;
    use tokbench_core::{Build, Class, Engine, Ids, Info, Model, Unsupported};

    pub struct Adapter {
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
                // See the module docs: the serial per-document entry points used
                // below never enter gigatoken's rayon batch engine.
                internally_parallel: false,
            }
        }

        fn encode(&mut self, text: &str, out: &mut Ids) {
            // Split the borrow: the SentencePiece call needs `&mut self.tok` and
            // `&mut self.sp_state` at the same time.
            let Self { tok, sp_state } = self;
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
    fn build(_model: &tokbench_core::Model)
        -> Result<Box<dyn tokbench_core::Engine>, tokbench_core::Unsupported>
    {
        Err(tokbench_core::Unsupported(
            "gigatoken not wired: its manifest declares `cargo-features`, which Cargo \
             rejects in a dependency and which breaks resolution for the whole \
             workspace. Run engines/gigatoken/scripts/enable_gigatoken.sh and build \
             with RUSTFLAGS='--cfg gigatoken_wired' on nightly."
                .into(),
        ))
    }
}
