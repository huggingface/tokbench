//! Which engines this binary was compiled with.
//!
//! Every engine folder exports one `Adapter` type implementing
//! [`tokbench_core::Build`], so registering it is a single line. Engines are
//! cargo features, which is what lets a machine without libsentencepiece still
//! build and run the rest of the matrix instead of failing wholesale.

use tokbench_core::{Build, Engine, Model, Unsupported};

pub type Ctor = fn(&Model, Option<usize>) -> Result<Box<dyn Engine>, Unsupported>;

/// The correctness oracle. Every other engine's id stream is compared against
/// this one; without it the run reports throughput but cannot verify it.
pub const REFERENCE: &str = "hf-tokenizers";

/// Engines reached in-process (native Rust, C ABI, embedded interpreter).
// Built by successive pushes rather than a `vec![]` literal because every entry
// is `#[cfg]`-gated independently; a literal cannot express that.
#[allow(unused_mut)]
pub fn native() -> Vec<(&'static str, Ctor)> {
    /// Register one engine twice: as itself, and as `<name>-no-cache`.
    ///
    /// The cache-free twin is a full row rather than a footnote, because a
    /// cache's contribution is otherwise unknowable from the outside -- and
    /// because it is exactly the number that was silently wrong before
    /// (`pipeline-no-cache` reported the cached engine for as long as the
    /// canonical reader ignored `cache_capacity`).
    ///
    /// `Build::build_without_cache` defaults to refusing, so an engine with no
    /// way to disable its caches contributes an explicit `unsupported: <why>`
    /// instead of a number that invites a wrong subtraction. Both halves share
    /// one `#[cfg]`, so a feature can never register only one of them.
    macro_rules! engine {
        ($v:ident, $feature:literal, $name:literal, $adapter:path) => {
            #[cfg(feature = $feature)]
            {
                $v.push(($name, <$adapter>::build_with_cache_capacity as Ctor));
                $v.push((
                    concat!($name, "-no-cache"),
                    <$adapter>::build_without_cache_with_capacity as Ctor,
                ));
            }
        };
    }

    let mut v: Vec<(&'static str, Ctor)> = Vec::new();

    engine!(
        v,
        "hf-tokenizers",
        "hf-tokenizers",
        tokbench_hf_tokenizers::Adapter
    );
    engine!(v, "pipeline", "pipeline", tokbench_pipeline::Adapter);
    engine!(v, "kitoken", "kitoken", tokbench_kitoken::Adapter);
    engine!(v, "fastokens", "fastokens", tokbench_fastokens::Adapter);
    engine!(v, "tokie", "tokie", tokbench_tokie::Adapter);
    engine!(v, "tiktoken", "tiktoken", tokbench_tiktoken::Adapter);
    engine!(
        v,
        "rust-gems-bpe",
        "rust-gems-bpe",
        tokbench_rust_gems_bpe::Adapter
    );
    engine!(
        v,
        "wordchipper",
        "wordchipper",
        tokbench_wordchipper::Adapter
    );
    engine!(
        v,
        "sentencepiece",
        "sentencepiece",
        tokbench_sentencepiece::Adapter
    );
    engine!(v, "blingfire", "blingfire", tokbench_blingfire::Adapter);
    engine!(v, "gigatoken", "gigatoken", tokbench_gigatoken::Adapter);
    engine!(v, "llamacpp", "llamacpp", tokbench_llamacpp::Adapter);
    engine!(v, "iree", "iree", tokbench_iree::Adapter);
    engine!(v, "executorch", "executorch", tokbench_executorch::Adapter);

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
