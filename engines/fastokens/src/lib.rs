use tokbench_core::{unsupported, Build, Class, Engine, Ids, Info, Model, Unsupported};

pub struct Adapter {
    tok: fastokens::Tokenizer,
    pool: Option<rayon::ThreadPool>,
    threads: usize,
}

impl Build for Adapter {
    fn build(model: &Model) -> Result<Box<dyn Engine>, Unsupported> {
        let path = model.tokenizer_json();
        if !path.exists() {
            return Err(Unsupported("no tokenizer.json".into()));
        }
        let tok = fastokens::Tokenizer::from_file(&path)
            .map_err(|e| Unsupported(format!("fastokens cannot load this config: {e}")))?;
        Ok(Box::new(Adapter {
            tok,
            pool: None,
            threads: 1,
        }))
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
            internally_parallel: self.threads > 1,
        }
    }

    fn encode(&mut self, text: &str, out: &mut Ids) {
        if let Ok(ids) = self.tok.encode_ordinary(text) {
            out.extend_from_slice(&ids);
        }
    }

    fn has_native_batch(&self) -> bool {
        true
    }

    fn set_threads(&mut self, threads: usize) -> bool {
        if threads == 0 {
            return false;
        }
        self.pool = if threads > 1 {
            match rayon::ThreadPoolBuilder::new().num_threads(threads).build() {
                Ok(pool) => Some(pool),
                Err(_) => return false,
            }
        } else {
            None
        };
        self.threads = threads;
        true
    }

    fn encode_batch(&mut self, texts: &[&str], out: &mut Ids) {
        let encode = || self.tok.encode_batch(texts, false);
        let encoded = match &self.pool {
            Some(pool) => pool.install(encode),
            None => encode(),
        };
        if let Ok(batch) = encoded {
            for ids in &batch {
                out.extend_from_slice(ids);
            }
        }
    }

    fn decode(&mut self, ids: &[u32], out: &mut String) -> Result<(), Unsupported> {
        match self.tok.decode(ids, false) {
            Ok(s) => {
                out.push_str(&s);
                Ok(())
            }
            Err(e) => unsupported(format!("decode failed: {e}")),
        }
    }
}
