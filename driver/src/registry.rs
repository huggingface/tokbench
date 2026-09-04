//! Which engines this binary was compiled with.
//!
//! Every engine folder exports one `Adapter` type implementing
//! [`tokbench_core::Build`], so registering it is a single line. Engines are
//! cargo features, which is what lets a machine without libsentencepiece still
//! build and run the rest of the matrix instead of failing wholesale.

use tokbench_core::{Build, Engine, Model, Unsupported};

pub type Ctor = fn(&Model) -> Result<Box<dyn Engine>, Unsupported>;

/// The correctness oracle. Every other engine's id stream is compared against
/// this one; without it the run reports throughput but cannot verify it.
pub const REFERENCE: &str = "hf-tokenizers";

/// Engines reached in-process (native Rust, C ABI, embedded interpreter).
// Built by successive pushes rather than a `vec![]` literal because every entry
// is `#[cfg]`-gated independently; a literal cannot express that.
#[allow(unused_mut, clippy::vec_init_then_push)]
pub fn native() -> Vec<(&'static str, Ctor)> {
    let mut v: Vec<(&'static str, Ctor)> = Vec::new();

    #[cfg(feature = "hf-tokenizers")]
    v.push((
        "hf-tokenizers",
        tokbench_hf_tokenizers::Adapter::build as Ctor,
    ));
    // The target encode path from tokenizers#2279: same project as the
    // reference, so this pairing is a controlled before/after.
    #[cfg(feature = "pipeline")]
    v.push(("pipeline", tokbench_pipeline::Adapter::build as Ctor));
    #[cfg(feature = "pipeline")]
    v.push((
        "pipeline-no-cache",
        tokbench_pipeline::NoCacheAdapter::build as Ctor,
    ));
    #[cfg(feature = "kitoken")]
    v.push(("kitoken", tokbench_kitoken::Adapter::build as Ctor));
    #[cfg(feature = "fastokens")]
    v.push(("fastokens", tokbench_fastokens::Adapter::build as Ctor));
    #[cfg(feature = "tokie")]
    v.push(("tokie", tokbench_tokie::Adapter::build as Ctor));
    #[cfg(feature = "tiktoken")]
    v.push(("tiktoken", tokbench_tiktoken::Adapter::build as Ctor));
    #[cfg(feature = "rust-gems-bpe")]
    v.push((
        "rust-gems-bpe",
        tokbench_rust_gems_bpe::Adapter::build as Ctor,
    ));
    #[cfg(feature = "wordchipper")]
    v.push(("wordchipper", tokbench_wordchipper::Adapter::build as Ctor));
    #[cfg(feature = "sentencepiece")]
    v.push((
        "sentencepiece",
        tokbench_sentencepiece::Adapter::build as Ctor,
    ));
    #[cfg(feature = "blingfire")]
    v.push(("blingfire", tokbench_blingfire::Adapter::build as Ctor));
    #[cfg(feature = "gigatoken")]
    v.push(("gigatoken", tokbench_gigatoken::Adapter::build as Ctor));
    #[cfg(feature = "llamacpp")]
    v.push(("llamacpp", tokbench_llamacpp::Adapter::build as Ctor));
    #[cfg(feature = "iree")]
    v.push(("iree", tokbench_iree::Adapter::build as Ctor));
    #[cfg(feature = "executorch")]
    v.push(("executorch", tokbench_executorch::Adapter::build as Ctor));

    v
}

/// Engines that only exist behind an interpreter, benchmarked by re-running
/// the SAME protocol inside that interpreter (`python/harness.py`) rather than
/// by embedding it. See `python/harness.py` for why that is the honest choice.
pub fn scripted() -> Vec<(&'static str, &'static str)> {
    vec![
        ("minbpe", "engines/minbpe/run.py"),
        ("mistral-common", "engines/mistral-common/run.py"),
        ("ai-tokenizer", "engines/ai-tokenizer/run.mjs"),
    ]
}
