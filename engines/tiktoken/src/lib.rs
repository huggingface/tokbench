//! tiktoken (OpenAI), through `tiktoken-rs`.
//!
//! tiktoken does not read `tokenizer.json`; it wants a rank file and a split
//! pattern. Rather than reimplement the byte-level-vocab → ranks conversion
//! here in Rust, this engine consumes two artifacts that `make models`
//! generates from the SAME `tokenizer.json` every other engine loads:
//!
//! * `ranks.tiktoken` — the standard `<base64-token> <rank>` per line format.
//! * `pattern.txt`    — the pre-tokenizer regex.
//!
//! Doing the conversion once, offline, in `scripts/make_artifacts.py` keeps it
//! out of the measured path and keeps this adapter honest: if the conversion
//! were wrong, the ids would diverge from the reference and the report would
//! mark the cell as a mismatch instead of publishing a bogus speedup.
//!
//! No artifacts → `Unsupported`. A blank cell with a reason is a truthful
//! result; tiktoken genuinely cannot load an arbitrary Unigram or WordPiece
//! model, and pretending otherwise would be the dishonest option.

use base64::Engine as _;
// tiktoken-rs builds its tables with rustc-hash, and `CoreBPE::new` takes
// `FxHashMap` specifically — a `std::HashMap` will not coerce.
use rustc_hash::FxHashMap;
use tiktoken_rs::CoreBPE;
use tokbench_core::{Build, Class, Engine, Ids, Info, Model, Unsupported};

pub struct Adapter {
    bpe: CoreBPE,
}

impl Build for Adapter {
    fn build(model: &Model) -> Result<Box<dyn Engine>, Unsupported> {
        let Some(ranks_path) = model.artifact("ranks.tiktoken") else {
            return Err(Unsupported(
                "no ranks.tiktoken (run `make models`; only byte-level BPE converts)".into(),
            ));
        };
        let Some(pattern_path) = model.artifact("pattern.txt") else {
            return Err(Unsupported("no pattern.txt".into()));
        };

        let raw = std::fs::read_to_string(&ranks_path)
            .map_err(|e| Unsupported(format!("reading ranks.tiktoken: {e}")))?;
        let mut encoder: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
        encoder.reserve(raw.len() / 16);
        for (lineno, line) in raw.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let mut parts = line.split_whitespace();
            let (Some(tok), Some(rank)) = (parts.next(), parts.next()) else {
                return Err(Unsupported(format!(
                    "ranks.tiktoken:{}: malformed",
                    lineno + 1
                )));
            };
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(tok)
                .map_err(|e| Unsupported(format!("ranks.tiktoken:{}: {e}", lineno + 1)))?;
            let rank: u32 = rank
                .parse()
                .map_err(|e| Unsupported(format!("ranks.tiktoken:{}: {e}", lineno + 1)))?;
            encoder.insert(bytes, rank);
        }

        let pattern = std::fs::read_to_string(&pattern_path)
            .map_err(|e| Unsupported(format!("reading pattern.txt: {e}")))?;

        // Empty special-token map: the reference is called with
        // `add_special_tokens = false`, and `encode_ordinary` below never
        // consults specials anyway. Keeping it empty makes that explicit.
        let bpe = CoreBPE::new(encoder, FxHashMap::default(), pattern.trim())
            .map_err(|e| Unsupported(format!("CoreBPE::new: {e}")))?;
        Ok(Box::new(Adapter { bpe }))
    }
}

impl Engine for Adapter {
    fn info(&self) -> Info {
        Info {
            name: "tiktoken",
            version: "0.12.0",
            lang: "rust",
            class: Class::Native,
            url: "https://github.com/zurawiki/tiktoken-rs",
            also_computes: "",
            internally_parallel: false,
        }
    }

    fn encode(&mut self, text: &str, out: &mut Ids) {
        out.extend_from_slice(&self.bpe.encode_ordinary(text));
    }
}
