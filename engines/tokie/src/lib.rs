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

use tokbench_core::{unsupported, Build, Class, Engine, Ids, Info, Model, Padding, Unsupported};

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

    /// `Tokenizer::encode_batch` fans out over its own stolen batches, so this
    /// engine must never be threaded from outside.
    fn has_native_batch(&self) -> bool {
        true
    }

    /// Always false, and that is a property of the library rather than a gap
    /// in this adapter: `encode_batch` reads its width from
    /// `thread::available_parallelism()` directly and exposes no knob, so
    /// there is no honest way to answer "use exactly n threads".
    ///
    /// Refusing here is what routes this engine to a single measurement at its
    /// own chosen width. Accepting and ignoring `threads` would label whatever
    /// width it picked as the requested one, which is worse than having no
    /// curve: it would look like a scaling result.
    fn set_threads(&mut self, _threads: usize) -> bool {
        false
    }

    /// `Tokenizer::encode_batch` — the library's own batch path, including its
    /// own parallelism.
    fn encode_batch(&mut self, texts: &[&str], out: &mut Ids) {
        for encoding in self.tok.encode_batch(texts, false) {
            out.extend_from_slice(&encoding.ids);
        }
    }

    /// `enable_padding` / `no_padding`. `PaddingParams::default()` is already
    /// `BatchLongest`, which is exactly [`Padding::Longest`].
    fn set_padding(&mut self, padding: Padding) -> bool {
        match padding {
            Padding::Off => {
                self.tok.no_padding();
            }
            Padding::Longest => {
                self.tok.enable_padding(tokie::PaddingParams::default());
            }
        }
        true
    }

    /// `Tokenizer::decode` returns `Option`, with `None` for a byte sequence
    /// that is not valid UTF-8. That is a real failure to reproduce the
    /// reference's text, so it becomes `Unsupported` rather than an empty
    /// push that would hash as a fast, wrong decode.
    fn decode(&mut self, ids: &[u32], out: &mut String) -> Result<(), Unsupported> {
        match self.tok.decode(ids) {
            Some(s) => {
                out.push_str(&s);
                Ok(())
            }
            None => unsupported("decode returned None (invalid utf-8)"),
        }
    }
}
