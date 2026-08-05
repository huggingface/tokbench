//! wordchipper (zspacelabs) — "HPC Rust LLM tokenizer library", compatible
//! with nanochat/rustbpe and tiktoken vocabularies.
//!
//! STATUS: scaffolded, not yet wired. This returns `Unsupported`, so the
//! report shows an explicit blank with a reason instead of a missing row.
//!
//! Why it is not wired yet: unlike every other Rust engine here, wordchipper
//! has no single `from_file(...) -> Tokenizer` entry point. Encoding is
//! assembled from parts — a `UnifiedTokenVocab<T>` obtained through
//! `pretrained::factory::vocab_factory::load_vocab` (or
//! `pretrained::huggingface::vocab_from_hf_tokenizer`, which itself takes an
//! already-built `tokenizers::Tokenizer`), then a `TokenSpanEncoder` built with
//! an explicit `SpanEncoderSelector` (`BpeBacktrack`, `MergeHeap`,
//! `PriorityMerge`, `BufferSweep`, `TailSweep`), then `TokenEncoderOptions`.
//!
//! That choice is load-bearing for fairness: the selector picks the merge
//! algorithm, so "wordchipper's throughput" is really "throughput of the
//! selector someone chose". Wiring this correctly means either benchmarking
//! the crate's own default, or reporting each selector as its own row. Guessing
//! one here would publish a number under a name that does not identify what ran.
//!
//! To wire it:
//!   1. Enable the dependency in Cargo.toml (note `default-features = false`;
//!      the default `parallel` feature would make this engine multi-threaded
//!      while every other single-thread row is not — see
//!      `Info::internally_parallel`).
//!   2. Build the vocab from `model.tokenizer_json()`.
//!   3. Report the selector in `Info::version`, e.g. "0.9.2 (BpeBacktrack)".

use tokbench_core::{Build, Engine, Model, Unsupported};

pub struct Adapter;

impl Build for Adapter {
    fn build(_model: &Model) -> Result<Box<dyn Engine>, Unsupported> {
        Err(Unsupported(
            "not wired: wordchipper needs an explicit vocab + SpanEncoderSelector; \
             see engines/wordchipper/src/lib.rs"
                .into(),
        ))
    }
}
