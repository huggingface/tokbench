use tk_encode::pipeline::{EncodeOptions, Override, PipelineToken, PipelineTokenizer};
use tk_encode::PaddingParams;
use tokbench_core::{unsupported, Build, Class, Engine, Ids, Info, Model, Padding, Unsupported};

pub struct Adapter {
    tok: PipelineTokenizer,
    scratch: Vec<PipelineToken>,
    name: &'static str,
    options: EncodeOptions,
    threads: usize,
}

fn build(
    model: &Model,
    cache_capacity: Option<usize>,
    name: &'static str,
) -> Result<Box<dyn Engine>, Unsupported> {
    let path = model.tokenizer_json();
    if !path.exists() {
        return Err(Unsupported("no tokenizer.json".into()));
    }
    let canonical = tk_convert::canonicalize_file(&path)
        .map_err(|e| Unsupported(format!("tk-convert cannot upgrade this config: {e}")))?;
    let canonical = if let Some(cache_capacity) = cache_capacity {
        let mut value: serde_json::Value = serde_json::from_str(&canonical)
            .map_err(|e| Unsupported(format!("cannot parse canonical config: {e}")))?;
        let model = value
            .get_mut("model")
            .and_then(serde_json::Value::as_object_mut)
            .ok_or_else(|| Unsupported("canonical config has no model object".into()))?;
        if model.get("type").and_then(serde_json::Value::as_str) != Some("BPE") {
            return Err(Unsupported(
                "cache capacity is configurable only for the pipeline BPE model".into(),
            ));
        }
        model.insert(
            "cache_capacity".into(),
            serde_json::Value::from(cache_capacity),
        );
        serde_json::to_string(&value)
            .map_err(|e| Unsupported(format!("cannot write cache config: {e}")))?
    } else {
        canonical
    };
    let tok = tk_serialize::from_json(&canonical)
        .map_err(|e| Unsupported(format!("tk-serialize cannot read this config: {e}")))?;
    Ok(Box::new(Adapter {
        tok,
        scratch: Vec::new(),
        name,
        options: EncodeOptions {
            add_special_tokens: false,
            padding: Override::Off,
        },
        threads: 1,
    }))
}

impl Build for Adapter {
    fn build(model: &Model) -> Result<Box<dyn Engine>, Unsupported> {
        build(model, None, "pipeline")
    }

    fn build_with_cache_capacity(
        model: &Model,
        cache_capacity: Option<usize>,
    ) -> Result<Box<dyn Engine>, Unsupported> {
        build(model, cache_capacity, "pipeline")
    }

    fn build_without_cache(model: &Model) -> Result<Box<dyn Engine>, Unsupported> {
        build(model, Some(0), "pipeline-no-cache")
    }
}

impl Engine for Adapter {
    fn info(&self) -> Info {
        Info {
            name: self.name,
            version: "tk-encode 1.0.0-rc.0 (tokenizers-rc0 @ 199d9a13)",
            lang: "rust",
            class: Class::Native,
            url: "https://github.com/huggingface/tokenizers/tree/199d9a1338b1ffe672549f4dd80be49af13fab92",
            also_computes: "",
            internally_parallel: self.threads > 1,
        }
    }

    fn encode(&mut self, text: &str, out: &mut Ids) {
        self.scratch.clear();
        if self
            .tok
            .encode_into(text, &self.options, &mut self.scratch)
            .is_ok()
        {
            out.extend(self.scratch.iter().map(|t| t.id()));
        }
    }

    fn has_native_batch(&self) -> bool {
        true
    }

    fn set_threads(&mut self, threads: usize) -> bool {
        if threads == 0 {
            return false;
        }
        tk_encode::parallelism::set_num_threads(threads);
        self.threads = threads;
        true
    }

    fn encode_batch(&mut self, texts: &[&str], out: &mut Ids) {
        let owned: Vec<String> = texts.iter().map(|t| (*t).to_string()).collect();
        if let Ok(encodings) = self.tok.encode(owned, &self.options).wait() {
            for encoding in &encodings {
                out.extend(encoding.ids().iter().map(|t| t.id()));
            }
        }
    }

    fn set_padding(&mut self, padding: Padding) -> bool {
        self.options.padding = match padding {
            Padding::Off => Override::Off,
            Padding::Longest => Override::With(PaddingParams {
                pad_id: 0,
                ..PaddingParams::default()
            }),
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
