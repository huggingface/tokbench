//! HuggingFace `tokenizers` — the baseline, and the oracle every other
//! engine's ids are checked against.
//!
//! This engine is privileged in one way only: its id stream defines "correct".
//! It gets no advantage in measurement — it goes through the same
//! `tokbench_core::measure` as everything else.
//!
//! Two fairness notes specific to this crate:
//!
//! * `encode` is called with `add_special_tokens = false`. That is the common
//!   denominator across the matrix: tiktoken, rust-gems-bpe and the raw
//!   byte-level engines have no notion of a post-processor template, so
//!   leaving specials on would charge this engine for work the others are not
//!   asked to do and make the id streams incomparable.
//! * The full `encode` path also computes byte offsets and word ids, which
//!   several competitors do not. That cost is left in — it is what a caller of
//!   this API actually pays — and is disclosed through `Info::also_computes`
//!   so the report can say so next to the number.

use std::time::Instant;

use tokbench_core::{unsupported, Build, Class, Engine, Ids, Info, Model, Phases, Unsupported};
use tokenizers::tokenizer::{Model as _, Normalizer as _, PostProcessor as _, PreTokenizer as _};
use tokenizers::{NormalizedString, OffsetType, PreTokenizedString, Tokenizer};

pub struct Adapter {
    tok: Tokenizer,
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
        let tok = Tokenizer::from_file(&path)
            .map_err(|e| Unsupported(format!("tokenizers cannot load this config: {e}")))?;
        Ok(Box::new(Adapter { tok }))
    }
}

impl Engine for Adapter {
    fn info(&self) -> Info {
        Info {
            name: "hf-tokenizers",
            version: "0.23.1",
            lang: "rust",
            class: Class::Native,
            url: "https://github.com/huggingface/tokenizers",
            also_computes: "byte offsets, word ids, attention mask",
            internally_parallel: false,
        }
    }

    fn encode(&mut self, text: &str, out: &mut Ids) {
        // A cell that fails mid-run must not silently look fast. Leaving `out`
        // short changes the id hash, so verification flags it.
        if let Ok(enc) = self.tok.encode(text, false) {
            out.extend_from_slice(enc.get_ids());
        }
    }

    /// `skip_special_tokens = false` to match `encode`'s
    /// `add_special_tokens = false`: neither direction adds or removes
    /// anything the other did not.
    ///
    /// This is the decode oracle — every other engine's decoded text is
    /// compared against this one's.
    fn decode(&mut self, ids: &[u32], out: &mut String) -> Result<(), Unsupported> {
        match self.tok.decode(ids, false) {
            Ok(s) => {
                out.push_str(&s);
                Ok(())
            }
            Err(e) => unsupported(format!("decode failed: {e}")),
        }
    }

    /// Re-runs the pipeline stage by stage. This deliberately does NOT reuse
    /// `encode`: it walks normalizer → pre-tokenizer → model → post-processor
    /// by hand so each stage gets its own clock. It is only ever called
    /// outside the timed loop.
    fn phases(&mut self, text: &str) -> Option<Phases> {
        let mut normalized = NormalizedString::from(text);
        let t = Instant::now();
        if let Some(n) = self.tok.get_normalizer() {
            n.normalize(&mut normalized).ok()?;
        }
        let normalization_ns = t.elapsed().as_nanos() as u64;

        let mut pre = PreTokenizedString::from(normalized);
        let t = Instant::now();
        if let Some(p) = self.tok.get_pre_tokenizer() {
            p.pre_tokenize(&mut pre).ok()?;
        }
        let pre_tokenization_ns = t.elapsed().as_nanos() as u64;

        let model = self.tok.get_model();
        let t = Instant::now();
        pre.tokenize(|s| model.tokenize(s.get())).ok()?;
        let core_encoding_ns = t.elapsed().as_nanos() as u64;

        let t = Instant::now();
        let enc = pre.into_encoding(None, 0, OffsetType::Byte).ok()?;
        if let Some(pp) = self.tok.get_post_processor() {
            pp.process(enc, None, false).ok()?;
        }
        let post_processing_ns = t.elapsed().as_nanos() as u64;

        Some(Phases {
            normalization_ns,
            pre_tokenization_ns,
            core_encoding_ns,
            post_processing_ns,
        })
    }
}
