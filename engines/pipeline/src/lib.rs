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
//! On this branch `encode_generic` takes caller-owned `pre_tokens`, `scratch`
//! and `output`, so they live in the adapter and are reused — the
//! configuration a server runs and the one the PR's own bench measures.
//!
//! What is NOT elided is the `.map(|t| t.id)` below. `encode_generic` fills a
//! `Vec<PipelineToken>` (a one-field `{ id: u32 }` struct) while the harness
//! compares `Vec<u32>`, and tk-encode exposes no flat-`u32` entry point, so a
//! caller wanting ids pays this restatement too. Measured at +1% to +8%
//! (median ~4%) by building once with the copy removed.

use tk_encode::pipeline::{Model, PipelineModelScratch, PipelineToken, PipelineTokenizer, Span};
use tk_encode::Tokenizer;
use tokbench_core::{Build, Class, Engine, Ids, Info, Model as BenchModel, Unsupported};

pub struct Adapter {
    pipe: PipelineTokenizer,
    /// Caller-owned and reused, which is what this branch's `encode_generic`
    /// expects and what a server would do. Allocating per call would charge the
    /// engine a malloc plus a first-touch of the whole token array every
    /// document -- a cost that scales with TOKEN count, so it would penalise
    /// token-dense scripts rather than measure the encoder.
    pre_tokens: Vec<Span>,
    scratch: PipelineModelScratch,
    toks: Vec<PipelineToken>,
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
        let scratch = pipe.get_model().init_scratch();
        Ok(Box::new(Adapter {
            pipe,
            pre_tokens: Vec::new(),
            scratch,
            toks: Vec::new(),
        }))
    }
}

impl Engine for Adapter {
    fn info(&self) -> Info {
        Info {
            name: "pipeline",
            // Not a release: the exact stacked tree that was measured.
            version: "tk-encode (#2279 poc/target-encode + #2296 metaspace runs, e35dc99c)",
            lang: "rust",
            class: Class::Native,
            url: "https://github.com/huggingface/tokenizers/pull/2296",
            also_computes: "",
            internally_parallel: false,
        }
    }

    fn encode(&mut self, text: &str, out: &mut Ids) {
        self.toks.clear();
        let r = self
            .pipe
            .encode_generic::<{ PipelineTokenizer::STAGE_POSTPROCESS }>(
                text,
                false,
                &mut self.pre_tokens,
                &mut self.scratch,
                &mut self.toks,
            );
        // On failure `out` stays short, which changes the id hash, so the
        // verification gate reports it instead of it passing as a fast run.
        if r.is_ok() {
            out.extend(self.toks.iter().map(|t| t.id));
        }
    }
}
