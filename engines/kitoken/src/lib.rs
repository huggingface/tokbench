//! kitoken — one crate covering BPE, Unigram and WordPiece.
//!
//! Notable in this lineup because its model coverage is broad: most of the
//! fast engines here handle byte-level BPE and give up on Unigram and
//! WordPiece, which is exactly where the id mismatches in this matrix cluster.
//! kitoken claims all three, so it is one of the few engines that can be held
//! to the reference across the whole model set rather than just the easy part
//! of it.
//!
//! It reads `tokenizer.json` directly through `from_tokenizers_file`, so it is
//! measured on the same artifact as the reference — no separate conversion
//! step that could silently change the vocabulary.
//!
//! `encode(text, false)` disables special-token recognition, matching the
//! `add_special_tokens = false` the reference is called with.
//!
//! Built with kitoken's default features rather than a hand-tuned set: that is
//! what `cargo add kitoken` gives you, so it is the configuration whose
//! throughput and binary size a reader would actually get. (`multiversion` and
//! `regex-perf` are among those defaults, so this is its fast path, not a
//! handicapped one.)

use tokbench_core::{Build, Class, Engine, Ids, Info, Model, Unsupported};

pub struct Adapter {
    tok: kitoken::Kitoken,
}

impl Build for Adapter {
    fn build(model: &Model) -> Result<Box<dyn Engine>, Unsupported> {
        let path = model.tokenizer_json();
        if !path.exists() {
            return Err(Unsupported("no tokenizer.json".into()));
        }
        let tok = kitoken::Kitoken::from_tokenizers_file(&path)
            .map_err(|e| Unsupported(format!("kitoken cannot load this config: {e}")))?;
        Ok(Box::new(Adapter { tok }))
    }
}

impl Engine for Adapter {
    fn info(&self) -> Info {
        Info {
            name: "kitoken",
            version: "0.11.0",
            lang: "rust",
            class: Class::Native,
            url: "https://github.com/Systemcluster/kitoken",
            also_computes: "",
            internally_parallel: false,
        }
    }

    fn encode(&mut self, text: &str, out: &mut Ids) {
        if let Ok(ids) = self.tok.encode(text, false) {
            out.extend(ids.iter().map(|&t| t as u32));
        }
    }
}
