//! splintr — BPE, Unigram, SentencePiece and WordPiece behind one loader.
//!
//! Reads `tokenizer.json` directly through `from_json_path`, so it is measured
//! on the same artifact as the reference.
//!
//! `encode_raw` rather than `encode`: it returns the backend's content tokens
//! without the post-processor template (BOS/EOS, `[CLS] … [SEP]`), matching the
//! `add_special_tokens = false` the reference is called with, while still
//! matching special tokens that appear in the text. It is also the serial
//! per-document path; splintr's parallel entry points (`encode_batch`,
//! `encode_rayon`) are not called here.

use std::time::Instant;

use splintr::{AnyTokenizer, Backend};
use tokbench_core::{Build, Class, Engine, Ids, Info, Model, Phases, Unsupported};

pub struct Adapter {
    tok: AnyTokenizer,
}

impl Build for Adapter {
    fn build(model: &Model) -> Result<Box<dyn Engine>, Unsupported> {
        let path = model.tokenizer_json();
        if !path.exists() {
            return Err(Unsupported("no tokenizer.json".into()));
        }
        let tok = splintr::from_json_path(&path)
            .map_err(|e| Unsupported(format!("splintr cannot load this config: {e}")))?;
        Ok(Box::new(Adapter { tok }))
    }
}

impl Engine for Adapter {
    fn info(&self) -> Info {
        Info {
            name: "splintr",
            version: "0.19.1",
            lang: "rust",
            class: Class::Native,
            url: "https://github.com/ml-rust/splintr",
            also_computes: "",
            internally_parallel: false,
        }
    }

    fn encode(&mut self, text: &str, out: &mut Ids) {
        out.extend_from_slice(&self.tok.encode_raw(text));
    }

    /// Measured by subtracting rungs, the same way `pipeline` does it, because
    /// splintr does not instrument its own stages.
    ///
    /// Three timings: the whole `encode_raw`, then the normalizer alone, then
    /// the pre-tokenizer over the normalized text. Core encoding is what is
    /// left. Post-processing is genuinely zero rather than unmeasured —
    /// `encode_raw` is the no-template entry point, which is why it is the one
    /// the reference's `add_special_tokens = false` is compared against.
    ///
    /// The pre-tokenizer rung is `for_each_pre_token`, not the more obvious
    /// `pre_tokenize`: the latter allocates a `String` per piece and reverses
    /// the ByteLevel mapping for its caller, and charging pre-tokenization for
    /// work `encode` never does would misplace exactly the time this chart
    /// exists to locate.
    ///
    /// **Cold cache, like the other instrumented engines.** `clear_cache`
    /// first, so the merge loop is not credited for chunks a previous rep
    /// already resolved. The totals are therefore larger than the headline
    /// throughput, which is warm — the proportions are the point, not the sum.
    ///
    /// Only the BPE backend is reported. Unigram and WordPiece do their own
    /// splitting inside the model with no public rung between the normalizer
    /// and the ids, so there is nothing to measure without inventing it.
    fn phases(&mut self, text: &str) -> Option<Phases> {
        let Backend::Bpe(bpe) = self.tok.backend() else {
            return None;
        };

        bpe.clear_cache();
        let start = Instant::now();
        let ids = self.tok.encode_raw(text);
        let total = start.elapsed().as_nanos() as u64;
        std::hint::black_box(&ids);

        let start = Instant::now();
        let normalized = bpe.normalize(text);
        let normalization_ns = start.elapsed().as_nanos() as u64;

        let start = Instant::now();
        bpe.for_each_pre_token(&normalized, |piece| {
            std::hint::black_box(piece);
        });
        let pre_tokenization_ns = start.elapsed().as_nanos() as u64;

        Some(Phases {
            normalization_ns,
            pre_tokenization_ns,
            // `saturating_sub` because the two rungs are timed separately from
            // the total, so scheduling noise can leave them summing past it on
            // a corpus where both are small.
            core_encoding_ns: total.saturating_sub(normalization_ns + pre_tokenization_ns),
            post_processing_ns: 0,
        })
    }
}
