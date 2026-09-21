use base64::Engine as _;
use rustc_hash::FxHashMap;
use tiktoken_rs::CoreBPE;
use tokbench_core::{unsupported, Build, Class, Engine, Ids, Info, Model, Unsupported};

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

    fn decode(&mut self, ids: &[u32], out: &mut String) -> Result<(), Unsupported> {
        match self.bpe.decode(ids) {
            Ok(s) => {
                out.push_str(&s);
                Ok(())
            }
            Err(e) => unsupported(format!("decode failed: {e}")),
        }
    }
}
