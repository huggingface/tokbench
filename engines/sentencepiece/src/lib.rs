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
