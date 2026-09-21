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
            out.extend(ids.iter().copied());
        }
    }
}
