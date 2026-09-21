use tokbench_core::{unsupported, Build, Class, Engine, Ids, Info, Model, Padding, Unsupported};
use tokenizers::Tokenizer;

pub struct Adapter {
    tok: Tokenizer,
    pool: Option<rayon::ThreadPool>,
    threads: usize,
}

impl Build for Adapter {
    fn build_without_cache(_model: &Model) -> Result<Box<dyn Engine>, Unsupported> {
        Err(Unsupported(
            "0.23.1 takes cache_capacity only through BpeBuilder, and the field is not in the \
             model config, so from_file cannot ask for it"
                .into(),
        ))
    }

    fn build(model: &Model) -> Result<Box<dyn Engine>, Unsupported> {
        let path = model.tokenizer_json();
        if !path.exists() {
            return Err(Unsupported(format!(
                "no tokenizer.json in {}",
                model.dir.display()
            )));
        }
        let tok = Tokenizer::from_file(&path)
            .map_err(|e| Unsupported(format!("tokenizers cannot load this config: {e}")))?;
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
            name: "hf-tokenizers",
            version: "0.23.1",
            lang: "rust",
            class: Class::Native,
            url: "https://github.com/huggingface/tokenizers",
            also_computes: "byte offsets, word ids, attention mask",
            internally_parallel: self.threads > 1,
        }
    }

    fn encode(&mut self, text: &str, out: &mut Ids) {
        if let Ok(enc) = self.tok.encode(text, false) {
            out.extend_from_slice(enc.get_ids());
        }
    }

    fn has_native_batch(&self) -> bool {
        true
    }

    fn set_threads(&mut self, threads: usize) -> bool {
        if threads == 0 {
            return false;
        }
        tokenizers::utils::parallelism::set_parallelism(threads > 1);
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
        let encode = || self.tok.encode_batch(texts.to_vec(), false);
        let encoded = match &self.pool {
            Some(pool) => pool.install(encode),
            None => encode(),
        };
        if let Ok(encodings) = encoded {
            for encoding in &encodings {
                out.extend_from_slice(encoding.get_ids());
            }
        }
    }

    fn set_padding(&mut self, padding: Padding) -> bool {
        match padding {
            Padding::Off => self.tok.with_padding(None),
            Padding::Longest => self
                .tok
                .with_padding(Some(tokenizers::PaddingParams::default())),
        };
        true
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
