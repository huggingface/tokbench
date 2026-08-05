//! fastokens — the Crusoe/NVIDIA-Dynamo byte-level BPE encoder, now published
//! by Atero. Reads HuggingFace `tokenizer.json` directly.
//!
//! `encode_ordinary` is used rather than `encode`: it is the "no special
//! tokens" path, which matches the `add_special_tokens = false` the reference
//! engine is called with. If a future version changes that meaning, the id
//! hash stops matching the reference and the report marks the cell as a
//! mismatch rather than quietly reporting a faster, different computation.

use tokbench_core::{Build, Class, Engine, Ids, Info, Model, Unsupported};

pub struct Adapter {
    tok: fastokens::Tokenizer,
}

impl Build for Adapter {
    fn build(model: &Model) -> Result<Box<dyn Engine>, Unsupported> {
        let path = model.tokenizer_json();
        if !path.exists() {
            return Err(Unsupported("no tokenizer.json".into()));
        }
        let tok = fastokens::Tokenizer::from_file(&path)
            .map_err(|e| Unsupported(format!("fastokens cannot load this config: {e}")))?;
        Ok(Box::new(Adapter { tok }))
    }
}

impl Engine for Adapter {
    fn info(&self) -> Info {
        Info {
            name: "fastokens",
            version: "0.3.1",
            lang: "rust",
            class: Class::Native,
            url: "https://github.com/Atero-ai/fastokens",
            also_computes: "",
            internally_parallel: false,
        }
    }

    fn encode(&mut self, text: &str, out: &mut Ids) {
        if let Ok(ids) = self.tok.encode_ordinary(text) {
            out.extend_from_slice(&ids);
        }
    }
}
