//! The HuggingFace **target encode path** — `PipelineTokenizer` from
//! [tokenizers#2279](https://github.com/huggingface/tokenizers/pull/2279)
//! (`poc/target-encode`, which carries bitsplit) with the metaspace-runs
//! pre-tokenizer from [#2296](https://github.com/huggingface/tokenizers/pull/2296)
//! stacked on top.
//!
//! This is the interesting row in the table: same project as the reference
//! engine, same `tokenizer.json`, so the difference between `hf-tokenizers`
//! and `pipeline` is the encode path itself rather than a different
//! vocabulary or a different idea of what a token is. Every other pairing here
//! compares across projects; this one is a controlled before/after.
//!
//! Which makes verification matter more here, not less. A rewritten merge loop
//! and pre-tokenizer is exactly the kind of change that can be very fast and
//! subtly wrong on some script, so the `verified` flag against
//! `tokenizers 0.23.1` is the whole point — a mismatch on any corpus is a bug
//! report, not a benchmark result.
//!
//! ## Getting the entry point wrong is easy — a note for the next person
//!
//! `tk_encode::Tokenizer::from_file(..).encode_fast(..)` compiles, runs, and
//! returns correct ids. It is also the **legacy** path this PR carries
//! alongside the new one, and it benchmarks ~100x slower than the work the PR
//! is actually about. The target path is a different type entirely:
//!
//! ```ignore
//! let legacy = tk_encode::Tokenizer::from_file(path)?;   // parse the config
//! let pipe = PipelineTokenizer::try_from(&legacy)?;       // build the fast path
//! pipe.encode_generic::<{ PipelineTokenizer::STAGE_POSTPROCESS }>(text, false)?;
//! ```
//!
//! `STAGE_POSTPROCESS` is the full pipeline; the lower `STAGE_*` constants are
//! the ablation ladder (frame / normalize / split / model) and must NOT be used
//! for a headline number, since they skip real work.
//!
//! ## Buffers, and the one cost this adapter adds
//!
//! `encode_generic` allocates a fresh output `Vec` sized from the input length,
//! which is a guess — measured 1 allocation plus 1-2 reallocations per call, the
//! reallocations where the guess undershoots on token-dense scripts. So this uses
//! `encode_generic_into` with a buffer the adapter reserves once and clears per
//! call: the configuration a server runs, and the one that matches how tokbench
//! hands every engine a caller-owned `out`.
//!
//! What is NOT elided is the `.map(|t| t.id)` below. `encode_generic` fills a
//! `Vec<PipelineToken>` (a one-field `{ id: u32 }` struct) while the harness
//! compares `Vec<u32>`, and tk-encode exposes no flat-`u32` entry point, so a
//! caller wanting ids pays this restatement too. Measured at +1% to +8%
//! (median ~4%) by building once with the copy removed.

use tk_encode::pipeline::{PipelineToken, PipelineTokenizer};
use tk_encode::Tokenizer;
use tokbench_core::{
    Build, Class, Engine, Ids, Info, Model as BenchModel, Unsupported,
};

pub struct Adapter {
    pipe: PipelineTokenizer,
    /// Reserved once and cleared per call. `encode_generic` allocates a fresh `Vec` sized from the
    /// input length, which is a guess: measured 1 allocation plus 1-2 reallocations per call, the
    /// reallocations being where the guess undershoots on token-dense scripts. tokbench hands every
    /// engine a caller-owned `out` for exactly this reason, so the engine should not be allocating
    /// behind it.
    scratch: Vec<PipelineToken>,
}

impl Build for Adapter {
    fn build(model: &BenchModel) -> Result<Box<dyn Engine>, Unsupported> {
        let path = model.tokenizer_json();
        if !path.exists() {
            return Err(Unsupported("no tokenizer.json".into()));
        }
        // The legacy tokenizer is only the config parser here — it is dropped
        // once the pipeline is built, and is never on the measured path.
        let legacy = Tokenizer::from_file(&path)
            .map_err(|e| Unsupported(format!("tk-encode cannot load this config: {e}")))?;
        let pipe = PipelineTokenizer::try_from(&legacy).map_err(|e| {
            Unsupported(format!(
                "PipelineTokenizer does not support this config yet: {e}"
            ))
        })?;
        Ok(Box::new(Adapter {
            pipe,
            scratch: Vec::new(),
        }))
    }
}

impl Engine for Adapter {
    fn info(&self) -> Info {
        Info {
            name: "pipeline",
            // Not a release, and not a literal: `build.rs` stamps the branch and short rev of the
            // tk-encode worktree this was compiled against, marking it DIRTY when that worktree has
            // uncommitted changes. A hand-written string here once outlived the checkout it named.
            version: env!("PIPELINE_TREE_VERSION"),
            lang: "rust",
            class: Class::Native,
            url: "https://github.com/huggingface/tokenizers/pull/2308",
            also_computes: "",
            internally_parallel: false,
        }
    }

    fn encode(&mut self, text: &str, out: &mut Ids) {
        self.scratch.clear();
        let r = self
            .pipe
            .encode_generic_into::<{ PipelineTokenizer::STAGE_POSTPROCESS }>(
                text,
                false,
                &mut self.scratch,
            );
        // On failure `out` stays short, which changes the id hash, so the
        // verification gate reports it instead of it passing as a fast run.
        if r.is_ok() {
            out.extend(self.scratch.iter().map(|t| t.id));
        }
    }

    // No phase breakdown on this branch: its `encode_generic` owns its buffers, so the ablation
    // ladder cannot be run with a caller-controlled cold cache the way it can on the
    // caller-owned-buffer branch. The harness reports "not instrumented" rather than a split
    // measured under a different cache regime from the headline.
}
