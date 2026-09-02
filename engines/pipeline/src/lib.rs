//! The HuggingFace **target encode path** — `tk-encode` from
//! [tokenizers#2279](https://github.com/huggingface/tokenizers/pull/2279)
//! ("bitsplit + batched model + fused cache probe").
//!
//! This is the interesting row in the table: it is the same project as the
//! reference engine, reading the same `tokenizer.json`, so the difference
//! between `hf-tokenizers` and `pipeline` is purely the new encode path rather
//! than a different vocabulary, a different pre-tokenizer, or a different
//! notion of what a token is. Every other engine in this repository is a
//! comparison across projects; this one is a controlled before/after.
//!
//! Which makes verification matter more here, not less. A rewrite of the merge
//! loop and pre-tokenizer is exactly the kind of change that can be very fast
//! and subtly wrong on some script, so the `verified` flag against
//! `tokenizers 0.23.1` is the whole point — a mismatch on any corpus is a bug
//! report, not a benchmark result.
//!
//! ## Why `encode_fast`
//!
//! `encode_fast` is the offset-free path. The reference engine is called
//! through `encode`, which also computes byte offsets, word ids and an
//! attention mask. Timing this side's *offset-computing* path against
//! competitors that only produce ids would be the wrong comparison, and timing
//! the reference's `encode_fast` while calling this one's `encode` would be the
//! wrong comparison in the other direction.
//!
//! Both are declared honestly instead of quietly equalised: the reference
//! discloses `also_computes: "byte offsets, word ids, attention mask"` and this
//! engine discloses nothing extra, so the report shows a reader that part of
//! any gap between the two is offset bookkeeping rather than raw encode speed.
//! The ids are identical either way, which is what `verified` checks.

use tokbench_core::{unsupported, Build, Class, Engine, Ids, Info, Model, Unsupported};

pub struct Adapter {
    tok: tk_encode::Tokenizer,
}

impl Build for Adapter {
    fn build(model: &Model) -> Result<Box<dyn Engine>, Unsupported> {
        let path = model.tokenizer_json();
        if !path.exists() {
            return Err(Unsupported("no tokenizer.json".into()));
        }
        let tok = tk_encode::Tokenizer::from_file(&path)
            .map_err(|e| Unsupported(format!("tk-encode cannot load this config: {e}")))?;
        Ok(Box::new(Adapter { tok }))
    }
}

impl Engine for Adapter {
    fn info(&self) -> Info {
        Info {
            name: "pipeline",
            // Not a release: the branch head. See the pinning note in Cargo.toml.
            version: "tk-encode 0.23.2-dev.0 (PR #2279, poc/target-encode)",
            lang: "rust",
            class: Class::Native,
            url: "https://github.com/huggingface/tokenizers/pull/2279",
            also_computes: "",
            internally_parallel: false,
        }
    }

    fn encode(&mut self, text: &str, out: &mut Ids) {
        if let Ok(enc) = self.tok.encode_fast(text, false) {
            out.extend_from_slice(enc.get_ids());
        }
    }

    /// There is no `decode_fast` counterpart to `encode_fast`: the PR's work
    /// is on the encode path, so decode goes through the ordinary entry point
    /// and this number is expected to sit near the reference's.
    fn decode(&mut self, ids: &[u32], out: &mut String) -> Result<(), Unsupported> {
        match self.tok.decode(ids, false) {
            Ok(s) => {
                out.push_str(&s);
                Ok(())
            }
            Err(e) => unsupported(format!("decode failed: {e}")),
        }
    }
}
