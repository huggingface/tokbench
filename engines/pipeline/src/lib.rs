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
    /// ## The cache has to be dropped between reps
    ///
    /// The `WordCache` lives in the caller-owned `BpeScratch`, so it survives
    /// an encode call -- which is the point of it, and right for the headline
    /// number, where every timed pass sees a slice of text the instance has not
    /// seen before. Here it is poison: this re-encodes *the same chunk* once
    /// per rep, so with the cache kept, rep 2 onward times a hash lookup rather
    /// than a merge. Both ways that went wrong before it was fixed:
    ///
    /// * Minimum of 9 reps reported **1.09 ms** for a corpus that takes ~7 ms
    ///   to encode -- the minimum was just the most-memoised pass.
    /// * Run in ladder order, `STAGE_MODEL` filled the cache and
    ///   `STAGE_POSTPROCESS`, timed next on the same text, came out at **0.1%**
    ///   of gpt2/english against the reference engine's 9.6%.
    ///
    /// So every rep gets a fresh scratch, allocated outside the clock. Each
    /// rung is measured from a cold cache, which also kills the ladder-order
    /// effect: no rung can warm the one after it.
    ///
    /// ## What a cold rep costs, and why it is still the right choice
    ///
    /// Dropping the cache is not free of consequence: it changes both the total
    /// and the attribution, because a cold merge loop does work a warm one
    /// skips. gpt2/english, same chunks, same code:
    ///
    /// ```text
    ///           total     normalize   pre-tokenize   model   post
    /// warm      1.15 ms      0.0%         36.2%      63.8%   0.0%
    /// cold      9.75 ms      0.0%          4.5%      95.5%   0.0%
    /// ```
    ///
    /// The headline encode of those chunks is ~6.8 ms, between the two: the
    /// harness reuses one instance across disjoint slices, so common words are
    /// cached and rare ones are not. Neither ablation regime reproduces that,
    /// and pretending otherwise would be the fabrication this module avoids.
    ///
    /// Cold is still right here for two reasons. It is the only regime that
    /// does not hand the engine credit for having already seen the exact text
    /// being timed. And `hf-tokenizers` keeps no such cache, so measuring cold
    /// puts both engines' breakdowns in the same regime -- which is what makes
    /// comparing their *proportions* legitimate.
    ///
    /// The attribution is still a difference of totals, so a phase's bucket
    /// absorbs any effect it has on the phases after it.
    ///
    /// Only ever called outside the timed loop.
    fn phases(&mut self, text: &str) -> Option<Phases> {
        const PHASE_REPS: usize = 5;

        // One monomorphised rung, median of `PHASE_REPS` cold reps. A macro
        // because `STAGE` is a const generic argument: a closure cannot be
        // generic over it, so each rung is spelled out as its own call.
        macro_rules! rung {
            ($stage:expr) => {{
                let mut runs = [0u64; PHASE_REPS];
                for slot in runs.iter_mut() {
                    // A fresh scratch is a fresh `WordCache`. Allocated before
                    // the clock starts, so it is not charged to the rung.
                    self.scratch = self.pipe.get_model().init_scratch();
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
