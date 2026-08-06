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
use std::time::Instant;

use tokbench_core::{
    Build, Class, Engine, Ids, Info, Model as BenchModel, Phases, Unsupported,
};

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

    /// Phase costs by **ablation**, not instrumentation.
    ///
    /// `encode_generic` is monomorphised on a `STAGE` constant, so the compiler
    /// emits a separate function per prefix of the pipeline: frame only, frame
    /// plus normalize, and so on up to the full encode. Each is real optimised
    /// code with no timing branches inside the hot loop -- the reason the PR
    /// carries the ladder at all -- so timing successive rungs and subtracting
    /// attributes cost without perturbing what is measured.
    ///
    /// Read it differently from `hf-tokenizers`' breakdown. That one runs each
    /// component and clocks it directly. This one subtracts two whole-pipeline
    /// timings, so a phase's bucket here absorbs any effect it has on the
    /// phases after it. The totals are honest; the attribution is a difference
    /// of totals.
    ///
    /// ## Two artefacts this has to avoid, both found by getting them wrong
    ///
    /// **The word cache makes rung order matter.** Run the ladder cold and in
    /// order and `STAGE_MODEL` fills the `WordCache`, so `STAGE_POSTPROCESS`
    /// -- timed next, on the same text -- reads almost free. Measured that way
    /// post-processing came out at 0.1% of gpt2/english against the reference
    /// engine's 9.6%, which is an artefact of the ladder, not a property of the
    /// pipeline. So a full encode runs first and is discarded: every rung then
    /// sees the same warm cache, which is also the state the headline number is
    /// measured in.
    ///
    /// **Minimum-of-N is the wrong statistic here.** Repeating a rung on one
    /// chunk replays it against a cache that only gets warmer, so the minimum
    /// converges on a best case the real encode never sees -- it reported 1.09
    /// ms for a corpus that takes ~7 ms to encode. The median of a few reps
    /// tracks the steady state instead.
    ///
    /// Differences can still come out slightly negative on a cheap stage
    /// swamped by noise; those clamp to zero rather than report negative time.
    ///
    /// ## What the totals here are, and are not
    ///
    /// Warming each chunk before timing it buys clean attribution at a price:
    /// these totals are the **fully warm** cost, and the headline number is not.
    /// The harness times disjoint slices, so a timed pass meets words this
    /// instance has often not seen; here every word is already in the
    /// `WordCache`. On gpt2/english that is ~1.2 ms against ~6.8 ms of real
    /// encode over the same chunks.
    ///
    /// So read the **shares**, which is what the chart stacks. Do not read a
    /// bar height here as this engine's encode time, and do not compare bar
    /// heights against `hf-tokenizers`, whose breakdown is instrumented and
    /// cold. Comparing the two engines' *proportions* is fine and is the point.
    ///
    /// Only ever called outside the timed loop.
    fn phases(&mut self, text: &str) -> Option<Phases> {
        const PHASE_REPS: usize = 5;

        // Warm first, discard. Every rung below then starts from the same cache
        // state, and it is the state the headline number is measured in.
        self.toks.clear();
        self.pipe
            .encode_generic::<{ PipelineTokenizer::STAGE_POSTPROCESS }>(
                text,
                false,
                &mut self.pre_tokens,
                &mut self.scratch,
                &mut self.toks,
            )
            .ok()?;

        // One monomorphised rung, median of `PHASE_REPS`. A macro because
        // `STAGE` is a const generic argument: a closure cannot be generic over
        // it, so each rung has to be spelled out as its own call.
        macro_rules! rung {
            ($stage:expr) => {{
                let mut runs = [0u64; PHASE_REPS];
                for slot in runs.iter_mut() {
                    self.toks.clear();
                    let t = Instant::now();
                    let r = self.pipe.encode_generic::<{ $stage }>(
                        text,
                        false,
                        &mut self.pre_tokens,
                        &mut self.scratch,
                        &mut self.toks,
                    );
                    *slot = t.elapsed().as_nanos() as u64;
                    r.ok()?;
                }
                runs.sort_unstable();
                runs[PHASE_REPS / 2]
            }};
        }

        let frame = rung!(PipelineTokenizer::STAGE_FRAME);
        let normalize = rung!(PipelineTokenizer::STAGE_NORMALIZE);
        let split = rung!(PipelineTokenizer::STAGE_SPLIT);
        let model = rung!(PipelineTokenizer::STAGE_MODEL);
        let post = rung!(PipelineTokenizer::STAGE_POSTPROCESS);

        // `frame` is the added-token scan and span setup that every rung pays.
        // It is not one of the four reported phases, so it is folded into
        // pre-tokenisation, which is where a reader looking for "the cost of
        // finding the pieces" would expect it.
        Some(Phases {
            normalization_ns: normalize.saturating_sub(frame),
            pre_tokenization_ns: frame + split.saturating_sub(normalize),
            core_encoding_ns: model.saturating_sub(split),
            post_processing_ns: post.saturating_sub(model),
        })
    }
}
