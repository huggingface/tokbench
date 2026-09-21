use tokbench_core::{Build, Class, Engine, Ids, Info, Model, Unsupported};

pub struct Adapter {
    tok: &'static bpe_openai::Tokenizer,
    encoding: &'static str,
}

impl Build for Adapter {
    fn build(model: &Model) -> Result<Box<dyn Engine>, Unsupported> {
        let Some(marker) = model.artifact("bpe_openai.txt") else {
            return Err(Unsupported(
                "no bpe_openai.txt: `bpe` alone has no pre-tokenizer, so its ids are not \
                 comparable; only cl100k/o200k/voyage3-equivalent models can run here"
                    .into(),
            ));
        };
        let name = std::fs::read_to_string(&marker)
            .map_err(|e| Unsupported(format!("reading bpe_openai.txt: {e}")))?;
        let (tok, encoding): (&'static bpe_openai::Tokenizer, &'static str) = match name.trim() {
            "cl100k_base" => (bpe_openai::cl100k_base(), "cl100k_base"),
            "o200k_base" => (bpe_openai::o200k_base(), "o200k_base"),
            "voyage3_base" => (bpe_openai::voyage3_base(), "voyage3_base"),
            other => {
                return Err(Unsupported(format!(
                    "bpe-openai has no prebuilt encoding {other:?}"
                )))
            }
        };
        Ok(Box::new(Adapter { tok, encoding }))
    }
}

impl Engine for Adapter {
    fn info(&self) -> Info {
        Info {
            name: "rust-gems-bpe",
            version: "bpe 0.2.1 / bpe-openai 0.3.0",
            lang: "rust",
            class: Class::Native,
            url: "https://github.com/github/rust-gems",
            also_computes: "",
            internally_parallel: false,
        }
    }

    fn encode(&mut self, text: &str, out: &mut Ids) {
        let _ = self.encoding;
        out.extend_from_slice(&self.tok.encode(text));
    }
}
