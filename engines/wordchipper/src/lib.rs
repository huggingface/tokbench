use std::sync::Arc;

use tokbench_core::{Build, Class, Engine, Ids, Info, Model, Unsupported};
use wordchipper::encoders::token_span_encoder::SpanEncoderSelector;
use wordchipper::pretrained::openai::{OA_CL100K_BASE_PATTERN, OA_GPT2_PATTERN};
use wordchipper::spanners::TextSpanningConfig;
use wordchipper::support::regex::{ConstRegexPattern, RegexPattern};
use wordchipper::vocab::io::load_base64_span_map_path;
use wordchipper::vocab::SpanMapVocab;
use wordchipper::{TokenEncoder, Tokenizer, TokenizerOptions, UnifiedTokenVocab};

const EQUIVALENT_SPELLINGS: &[(&str, ConstRegexPattern)] = &[
    (
        r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
        OA_GPT2_PATTERN,
    ),
    (
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        OA_CL100K_BASE_PATTERN,
    ),
];

pub struct Adapter {
    tok: Arc<Tokenizer<u32>>,
}

impl Build for Adapter {
    fn build(model: &Model) -> Result<Box<dyn Engine>, Unsupported> {
        let Some(ranks_path) = model.artifact("ranks.tiktoken") else {
            return Err(Unsupported(
                "no ranks.tiktoken (run `make models`; only byte-level BPE converts, \
                 which is also all wordchipper accepts)"
                    .into(),
            ));
        };
        let Some(pattern_path) = model.artifact("pattern.txt") else {
            return Err(Unsupported("no pattern.txt".into()));
        };

        let raw = std::fs::read_to_string(&pattern_path)
            .map_err(|e| Unsupported(format!("reading pattern.txt: {e}")))?;
        let raw = raw.trim();
        let pattern: RegexPattern = EQUIVALENT_SPELLINGS
            .iter()
            .find(|(hf, _)| *hf == raw)
            .map(|&(_, wc)| RegexPattern::from(wc))
            .unwrap_or_else(|| RegexPattern::from(raw));

        let mut span_map = load_base64_span_map_path::<u32, _>(&ranks_path)
            .map_err(|e| Unsupported(format!("wordchipper cannot read ranks.tiktoken: {e}")))?;

        loop {
            let unreachable: Vec<Vec<u8>> = span_map
                .keys()
                .filter(|span| {
                    span.len() > 1
                        && !(1..span.len()).any(|p| {
                            span_map.contains_key(&span[..p]) && span_map.contains_key(&span[p..])
                        })
                })
                .cloned()
                .collect();
            if unreachable.is_empty() {
                break;
            }
            for span in unreachable {
                span_map.remove(&span);
            }
        }

        let mut byte_ids: Vec<u32> = span_map
            .iter()
            .filter(|(span, _)| span.len() == 1)
            .map(|(_, &id)| id)
            .collect();
        let singles = byte_ids.len();
        byte_ids.sort_unstable();
        byte_ids.dedup();
        if singles != 256 || byte_ids.len() != 256 {
            return Err(Unsupported(format!(
                "vocabulary has {singles} single-byte tokens over {} distinct ids; \
                 wordchipper needs a 1:1 table for all 256 bytes",
                byte_ids.len()
            )));
        }

        let vocab = UnifiedTokenVocab::from_span_vocab(
            TextSpanningConfig::from_pattern(pattern),
            SpanMapVocab::from_span_map(span_map),
        )
        .map_err(|e| Unsupported(format!("wordchipper cannot load ranks.tiktoken: {e}")))?;

        let options = TokenizerOptions::default();

        let selector = options.encoder.effective_span_encoder();
        if selector != SpanEncoderSelector::SingleThreadDefault {
            return Err(Unsupported(format!(
                "expected the SingleThreadDefault span encoder, got {selector}; \
                 Info::version would name the wrong algorithm"
            )));
        }

        Ok(Box::new(Adapter {
            tok: options.build(Arc::new(vocab)),
        }))
    }
}

impl Engine for Adapter {
    fn info(&self) -> Info {
        Info {
            name: "wordchipper",
            version: "0.9.2 (BpeBacktrack)",
            lang: "rust",
            class: Class::Native,
            url: "https://github.com/zspacelabs/wordchipper",
            also_computes: "",
            internally_parallel: false,
        }
    }

    fn encode(&mut self, text: &str, out: &mut Ids) {
        let _ = self.tok.try_encode_append(text, out, None);
    }
}
