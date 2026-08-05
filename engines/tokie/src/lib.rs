//! tokie — chonkie's tokenizer: an Aho-Corasick backtracking encoder for BPE
//! and a double-array trie for WordPiece.
//!
//! tokie's native artifact is `.tkz`, a prebuilt format that front-loads
//! automaton construction. It also reads `tokenizer.json` directly via
//! `from_json`. Both are offered, and which one ran is visible in `load_ms`:
//!
//! * `model.tkz` when `make models` produced one — this is how tokie is meant
//!   to be deployed, and it is the fair configuration to report.
//! * `tokenizer.json` otherwise, so the engine still appears in the matrix
//!   instead of leaving a hole.
//!
//! Either way the *encode* path being timed is identical; only build cost
//! differs, and build cost is excluded from the timed region by design.

use tokbench_core::{Build, Class, Engine, Ids, Info, Model, Unsupported};

pub struct Adapter {
    tok: tokie::Tokenizer,
}

impl Build for Adapter {
    fn build(model: &Model) -> Result<Box<dyn Engine>, Unsupported> {
        if let Some(tkz) = model.artifact("model.tkz") {
            let tok = tokie::Tokenizer::from_file(&tkz)
                .map_err(|e| Unsupported(format!("tokie cannot read {}: {e}", tkz.display())))?;
            return Ok(Box::new(Adapter { tok }));
        }
        let json = model.tokenizer_json();
        if !json.exists() {
            return Err(Unsupported("no model.tkz and no tokenizer.json".into()));
        }
        let tok = tokie::Tokenizer::from_json(&json)
            .map_err(|e| Unsupported(format!("tokie cannot load this config: {e}")))?;
        Ok(Box::new(Adapter { tok }))
    }
}

impl Engine for Adapter {
    fn info(&self) -> Info {
        Info {
            name: "tokie",
            version: "0.1.4",
            lang: "rust",
            class: Class::Native,
            url: "https://github.com/chonkie-inc/tokie",
            also_computes: "",
            internally_parallel: false,
        }
    }

    fn encode(&mut self, text: &str, out: &mut Ids) {
        // `encode_ids` is the ids-only path; `encode`/`encode_with_offsets`
        // would additionally build an Encoding with offsets the reference is
        // not being charged for here.
        out.extend_from_slice(&self.tok.encode_ids(text, false));
    }
}
