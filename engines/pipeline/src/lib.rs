//! The HuggingFace **target encode path** — `PipelineTokenizer` from
//! [tokenizers#2279](https://github.com/huggingface/tokenizers/pull/2279)
//! ("bitsplit + batched model + fused cache probe").
//!
//! This is the interesting row in the table: same project as the reference
//! engine, same `tokenizer.json`, so the difference between `hf-tokenizers`
//! and `pipeline` is the encode path itself rather than a different
//! vocabulary or a different idea of what a token is. Every other pairing here
//! compares across projects; this one is a controlled before/after.
//!
//! ## Getting this wrong is easy — a note for the next person
//!
//! `tk_encode::Tokenizer::from_file(..).encode_fast(..)` compiles, runs, and
//! returns correct ids. It is also the **legacy** path that this PR carries
//! alongside the new one, and it benchmarks ~100x slower than the work the PR
//! is actually about. The target path is a different type entirely:
//!
//! ```ignore
//! let legacy = tk_encode::Tokenizer::from_file(path)?;      // parse the config
//! let pipe = PipelineTokenizer::try_from(&legacy)?;          // build the fast path
//! pipe.encode_generic::<{ PipelineTokenizer::STAGE_POSTPROCESS }>(
//!     text, add_special_tokens, &mut pre_tokens, &mut scratch, &mut out)?;
//! ```
//!
//! `STAGE_POSTPROCESS` is the full pipeline; the lower `STAGE_*` constants are
//! the ablation ladder (frame / normalize / split / model) and must NOT be used
//! for a headline number, since they skip real work.
//!
//! ## Why the buffers live in the adapter
//!
//! `encode_generic` writes into caller-owned `pre_tokens` and `scratch`. Those
//! are allocated once at build time and reused, which is the configuration the
//! PR's own `ab_giga` harness measures and the one a server would run. Creating
//! them per call would charge the engine an allocation and a first-touch of the
//! whole token array on every encode — a cost that scales with token count, so
//! it would quietly penalise token-dense corpora (Chinese emits ~3.5x the
//! tokens of English for the same bytes) rather than measuring the encoder.
//!
//! This is the same courtesy every other engine gets: `tokbench_core::measure`
//! hands each of them a reused `out` buffer.
//!
//! ## The one cost this adapter adds
//!
//! `encode_generic` fills a `Vec<PipelineToken>` (a one-field `{ id: u32 }`
//! struct), while the harness compares `Vec<u32>`. The `.map(|t| t.id)` copy
//! below is therefore adapter overhead the PR's own bench does not pay. It is
//! left in and timed rather than hidden with a transmute: it is small, it is
//! honest, and a caller wanting ids out of this API pays it too.

use tk_encode::pipeline::{Model, PipelineModelScratch, PipelineToken, PipelineTokenizer, Span};
use tk_encode::Tokenizer;
use tokbench_core::{Build, Class, Engine, Ids, Info, Model as BenchModel, Unsupported};

pub struct Adapter {
    pipe: PipelineTokenizer,
    /// Reused across calls; see the module docs.
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
            // Not a release: the branch head. Pin a rev before quoting this.
            version: "tk-encode 0.23.2-dev.0 (PR #2279, poc/target-encode)",
            lang: "rust",
            class: Class::Native,
            url: "https://github.com/huggingface/tokenizers/pull/2279",
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
