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
        out.extend_from_slice(&self.tok.encode_ids(text, false));
    }

    fn has_native_batch(&self) -> bool {
        true
    }

    fn set_threads(&mut self, _threads: usize) -> bool {
        false
    }

    fn encode_batch(&mut self, texts: &[&str], out: &mut Ids) {
        for encoding in self.tok.encode_batch(texts, false) {
            out.extend_from_slice(&encoding.ids);
        }
    }

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
