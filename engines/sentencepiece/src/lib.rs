//! Google SentencePiece (C++), through danieldk's `sentencepiece` crate.
//!
//! SentencePiece reads its own `.model` protobuf, not `tokenizer.json`. The
//! `spiece.model` artifact `make models` extracts comes from the same upstream
//! model directory as the `tokenizer.json` every other engine loads, so the
//! comparison is between two encoders of the SAME vocabulary.
//!
//! Expect this engine to be flagged as an id mismatch on models whose HF
//! conversion added a prefix space, changed the unknown-token policy, or
//! folded a normalizer into `tokenizer.json` that the raw `.model` does not
//! apply. That flag is the correct outcome and the reason verification exists:
//! it means the two are not computing the same function, and their speeds are
//! not comparable — not that one is faster.

use sentencepiece::SentencePieceProcessor;
use tokbench_core::{Build, Class, Engine, Ids, Info, Model, Unsupported};

pub struct Adapter {
    sp: SentencePieceProcessor,
}

impl Build for Adapter {
    fn build(model: &Model) -> Result<Box<dyn Engine>, Unsupported> {
        let path = model
            .artifact("spiece.model")
            .or_else(|| model.artifact("tokenizer.model"))
            .ok_or_else(|| {
                Unsupported("no spiece.model/tokenizer.model (not a SentencePiece model)".into())
            })?;
        let sp = SentencePieceProcessor::open(&path).map_err(|e| {
            Unsupported(format!("sentencepiece cannot open {}: {e}", path.display()))
        })?;
        Ok(Box::new(Adapter { sp }))
    }
}

impl Engine for Adapter {
    fn info(&self) -> Info {
        Info {
            name: "sentencepiece",
            version: "0.14.0 (libsentencepiece, static)",
            lang: "c++",
            class: Class::Cffi,
            url: "https://github.com/google/sentencepiece",
            // `encode` returns pieces with their ids and byte spans; the crate
            // offers no ids-only path, so that work is in the measurement.
            also_computes: "surface pieces + spans",
            internally_parallel: false,
        }
    }

    fn encode(&mut self, text: &str, out: &mut Ids) {
        if let Ok(pieces) = self.sp.encode(text) {
            out.extend(pieces.iter().map(|p| p.id));
        }
    }
}
