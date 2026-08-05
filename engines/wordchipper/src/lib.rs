//! wordchipper (zspacelabs) — "HPC Rust LLM tokenizer library", compatible
//! with nanochat/rustbpe and tiktoken vocabularies.
//!
//! Unlike every other Rust engine here, wordchipper has no single
//! `from_file(...) -> Tokenizer` entry point. A tokenizer is assembled from
//! parts: a `UnifiedTokenVocab<T>` (vocabulary + byte map + pair table + the
//! spanners config, which is where the pre-tokenizer regex and the special
//! tokens live), then `TokenizerOptions::build(vocab)`, which picks a span
//! encoder and a text spanner. Every decision below is one of those parts.
//!
//! # Where the vocabulary comes from
//!
//! The obvious route — `pretrained::huggingface::vocab_from_hf_tokenizer`,
//! which converts an already-built `tokenizers::Tokenizer` — **is not callable
//! in 0.9.2**. `src/pretrained/huggingface/mod.rs` is just `mod hf_factory;`
//! with no `pub` and no re-export, so the function is `pub` inside a private
//! module and invisible outside the crate. Enabling the `huggingface` feature
//! would therefore pull the entire `tokenizers` crate and buy nothing.
//!
//! So this adapter uses the loader wordchipper documents for exactly this job,
//! `vocab::io::load_base64_unified_vocab_path` (it is the worked example at the
//! top of that module), fed the two artifacts `make models` already derives
//! from the SAME `tokenizer.json` the reference engine loads:
//!
//! * `ranks.tiktoken` — `<base64 token bytes> <rank>` per line, which is
//!   precisely the format that loader parses.
//! * `pattern.txt`    — the pre-tokenizer regex.
//!
//! This is the same deal `engines/tiktoken` takes, for the same reason: the
//! byte-level-vocab → ranks conversion happens once, offline, in
//! `scripts/make_artifacts.py`, so it is out of the measured path and cannot be
//! silently wrong — a bad conversion changes the ids, and the report marks the
//! cell as a mismatch instead of publishing a bogus speedup.
//!
//! It also keeps this crate's dependency closure equal to wordchipper itself,
//! which matters because `binsize` links this engine to answer "what does
//! adding this library cost my binary". Parsing `tokenizer.json` here — with
//! `tokenizers`, or by hand with `serde_json` — would put a JSON parser that
//! wordchipper does not need into that number.
//!
//! Coverage is unaffected by taking the derived artifact rather than
//! `tokenizer.json`: wordchipper is byte-level BPE only (its own HF converter
//! rejects non-BPE models, models with an `unk` token, and any vocabulary
//! entry outside the byte-level alphabet), and `make_artifacts.py` declines to
//! write `ranks.tiktoken` for exactly that same set — Unigram, WordPiece,
//! SentencePiece-derived BPE, and models whose pre-tokenizer is several
//! sequential `Split` stages with no single-regex equivalent.
//!
//! # Why entries are dropped from the vocabulary, and which
//!
//! wordchipper does not store merges; it *derives* them, in
//! `SpanMapVocab::to_pair_vocab`, by splitting every vocabulary span at every
//! position and keeping the splits where both halves are themselves in the
//! vocabulary. A span with no such split gets no pair entry, the span vocab
//! and the derived pair vocab then disagree about which tokens exist, and
//! `UnifiedTokenVocab::new` rejects the whole vocabulary:
//! `"span vocab and pair vocab have different token sets"`. That is what
//! wordchipper does with gpt2 and mistral-nemo as `make_artifacts.py` emits
//! them, so this is not optional cleanup — without it the engine loads
//! nothing.
//!
//! What is unsplittable is exactly what BPE cannot build. A real merge is by
//! construction the concatenation of two vocabulary entries, so a span with no
//! such decomposition can never come out of the merge loop no matter what the
//! input is. Removing it is provably invisible in the ids. Concretely it is 1
//! entry for gpt2 (`<|endoftext|>`) and 1000 for mistral-nemo (`<unk>`,
//! `[INST]`, `[/AVAILABLE_TOOLS]`, …) — control tokens that survived the
//! byte-level round-trip because they happen to be spelled in ASCII, and that
//! `make_artifacts.py` therefore could not tell apart from ordinary tokens.
//! llama-3 loses nothing.
//!
//! The removal is iterated to a fixed point, because a span whose only
//! decomposition used a removed entry becomes undecomposable in turn — and by
//! the same argument is equally unreachable.
//!
//! They are dropped rather than re-registered as special tokens, which is the
//! other thing one could do with them. "Unsplittable" is not a synonym for
//! "added token": llama-3's `<|begin_of_text|>` and friends *are* added tokens
//! and they decompose fine, so they stay in the word vocabulary here. A rule
//! that misses the added tokens of one model cannot honestly be described as
//! recovering them for another. Dropping leaves this engine in exactly the
//! position `engines/tiktoken` is in — byte-level BPE over ranks, no
//! out-of-band special scanning — and leaves `hf-tokenizers` its already
//! disclosed extra work. Text that actually contains one of these strings
//! would then tokenize differently from the reference, and the harness would
//! report the cell as a mismatch, which is the correct outcome and not one
//! this adapter is entitled to paper over.
//!
//! # Why the pattern is normalised, and why that is not cheating
//!
//! wordchipper ships hand-written `logos` DFA lexers for the gpt2/r50k,
//! cl100k and o200k split patterns, and picks one by comparing the configured
//! pattern **string** for exact equality against its own constant
//! (`accelerators::get_regex_accelerator`). Its spelling of a pattern is not
//! HuggingFace's — it uses possessive quantifiers and a factored `'` prefix —
//! so handing it the HF spelling of the *same language* misses the accelerator
//! and silently measures the `fancy_regex` backtracking fallback instead.
//! That would not be "wordchipper's throughput"; it would be the throughput of
//! a path the library goes out of its way never to take on these models.
//!
//! `wordchipper`'s own HF converter does the same substitution (it maps a
//! `ByteLevel` pre-tokenizer straight onto `OA_GPT2_PATTERN` rather than
//! reconstructing the regex), so this follows the library's own policy rather
//! than inventing one. The table below is therefore short and is only for the
//! spellings that differ; an o200k model already arrives byte-identical to
//! `OA_O200K_BASE_PATTERN` and needs no entry. Anything unrecognised is passed
//! through verbatim as `RegexPattern::Adaptive`, which is the honest outcome:
//! mistral-nemo's pattern is o200k-*like* but not o200k (`\p{N}` where o200k
//! has `\p{N}{1,3}`, and no trailing `'s`/`'ll` group), so it gets no
//! accelerator and its number should say so. It does: mistral-nemo measures
//! 10-61 MB/s across the ten corpora where gpt2, on the mapped pattern,
//! measures 58-206 MB/s. That gap is the lexer, not the merge loop.
//!
//! Equivalence is asserted nowhere and *verified* everywhere: if a mapped
//! pattern were not equivalent, the spans would differ, the ids would differ,
//! and the cell would be reported as a mismatch rather than as a win.
//!
//! # A known divergence: llama-3 on Korean
//!
//! One cell of the matrix does not verify: llama-3 × korean, where this engine
//! emits 2,285 ids against the reference's 2,345. It is recorded here because
//! it is a property of wordchipper, not a defect in the wiring, and the number
//! should not be read as if it were.
//!
//! wordchipper has no merges list — `to_pair_vocab` reconstructs one from the
//! vocabulary by taking every split of every span whose halves are both in the
//! vocabulary. That reconstruction is lossy about *order*. On the pre-token
//! `Ġìĺ¤íĽĦ`, rank-ordered BPE merges `Ġ`+`ìĺ` first because that pair has the
//! lower id, which destroys the prefix `Ġìĺ¤` and yields
//! `[39623, 45780, 75309]`; `BpeBacktrack`, searching the reconstructed pair
//! table, instead finds the single token 124467 for the whole pre-token. The
//! merge it uses is a genuine listed merge of llama-3, so the output is a
//! legal — and shorter — tokenization; it is simply not the one the merge
//! ordering produces.
//!
//! This was checked rather than assumed: an id-rank BPE simulated over the
//! same vocabulary reproduces the reference's three ids exactly, so the
//! divergence is in the reconstruct-and-backtrack strategy and not in the
//! ranks, the pattern mapping, or the dropped entries above. The pre-token
//! boundaries agree on both sides. It is one cell out of thirty: gpt2 and
//! mistral-nemo verify on all ten corpora each, llama-3 on nine of ten.
//!
//! # Which span encoder ran
//!
//! `SpanEncoderSelector` chooses the merge algorithm — `BpeBacktrack`,
//! `MergeHeap`, `PriorityMerge`, `BufferSweep`, `TailSweep` — so a bare
//! "wordchipper: N MB/s" is meaningless. This uses the crate's own default
//! (`TokenEncoderOptions::span_encoder = None`, which resolves to
//! `SingleThreadDefault` for a non-concurrent build) rather than shopping for
//! the fastest one, and `Info::version` names the algorithm that resolves to,
//! `BpeBacktrack`, so the row identifies what actually ran.
//!
//! `Info::version` is a `&'static str` and cannot be computed, so `build`
//! checks that the effective selector really is `SingleThreadDefault` and
//! refuses to run otherwise. A wrong label is worse than a missing cell.
//!
//! # Why `parallel` is off
//!
//! `parallel` is one of wordchipper's three default features, and rule 5 of
//! the fairness contract makes the headline single-thread. It does **not**
//! make encoding multi-threaded on its own: `TokenEncoderOptions::default()`
//! has `parallel: false`, and `ParallelRayonEncoder` is only wrapped around
//! the encoder when that flag is true. The measured code path is byte-identical
//! with the feature on or off, so `internally_parallel: false` is accurate
//! either way.
//!
//! It is off anyway for two reasons. First, it turns the single-thread claim
//! from "the default happens to be false" into a property of the build.
//! Second, `parallel` implies `concurrent`, which pulls rayon into the
//! `binsize` measurement for a branch this engine never takes.
//!
//! There is a third effect worth recording because it cuts the other way:
//! `build_regex_lexer` only tries the `regex_automata` lexer when the
//! `concurrent` *feature* is absent or the `concurrent` *flag* is set. With
//! the default features and single-thread options — feature on, flag off —
//! that middle tier is skipped and an unaccelerated pattern falls all the way
//! to `fancy_regex`. Dropping the feature restores it. For every model this
//! engine can load with a mapped pattern the accelerator fires first and none
//! of this is reachable, but on an unrecognised pattern this build is the
//! faster of the two, and it would be dishonest to report the number without
//! saying so.
//!
//! # Special tokens
//!
//! The special vocabulary here is empty — the same position
//! `engines/tiktoken` is in with `encode_ordinary`, and the common denominator
//! the reference is held to (`add_special_tokens = false`). The `None` filter
//! passed to `try_encode_append` means "accept all specials", of which there
//! are none; it is the crate's own default rather than a restriction imposed
//! by the adapter. An empty special vocabulary also means
//! `TextSpannerBuilder` builds no special lexer, so no second scan of the
//! input happens per call.

use std::sync::Arc;

use tokbench_core::{Build, Class, Engine, Ids, Info, Model, Unsupported};
use wordchipper::encoders::token_span_encoder::SpanEncoderSelector;
use wordchipper::pretrained::openai::{OA_CL100K_BASE_PATTERN, OA_GPT2_PATTERN};
use wordchipper::spanners::TextSpanningConfig;
use wordchipper::support::regex::{ConstRegexPattern, RegexPattern};
use wordchipper::vocab::io::load_base64_span_map_path;
use wordchipper::vocab::SpanMapVocab;
use wordchipper::{TokenEncoder, Tokenizer, TokenizerOptions, UnifiedTokenVocab};

/// HuggingFace's spelling of a split pattern, paired with wordchipper's
/// spelling of the same language.
///
/// Only patterns whose two spellings differ need an entry — see the module
/// docs. Both of these are what `scripts/make_artifacts.py` writes into
/// `pattern.txt`: the first is the pattern a bare `ByteLevel` pre-tokenizer
/// implies (gpt2), the second is llama-3's own `Split` regex, which is the
/// cl100k pattern.
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

        // Everything from here to `options.build` is load-time and deliberately
        // so: base64 decode, the unreachable-span sweep, hashing the whole span
        // map and deriving the pair table from it all happen before `measure`
        // starts its clock.
        let mut span_map = load_base64_span_map_path::<u32, _>(&ranks_path)
            .map_err(|e| Unsupported(format!("wordchipper cannot read ranks.tiktoken: {e}")))?;

        // Drop the spans BPE cannot build, to a fixed point — see the module
        // docs. This is the same decomposition test `to_pair_vocab` applies,
        // so reaching a fixed point here is exactly the condition under which
        // `from_span_vocab` will accept the result.
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

        // `SpanMapVocab::from_span_map` panics unless the byte table it infers
        // is a bijection, and it fills any byte missing from the vocabulary
        // with the identity id — which can collide with a real token. Checking
        // here turns that panic into an `Unsupported` cell. Every byte-level
        // BPE vocabulary has all 256, so this is a guard, not a filter.
        // The ids themselves need not be 0..255 — mistral-nemo numbers its
        // byte tokens above its control block — they only have to be 256
        // distinct ids covering 256 distinct bytes.
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

        // Keeps `Info::version` honest — see the module docs.
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
            // `SingleThreadDefault` is documented as an alias for
            // `BpeBacktrack`, and `SpanEncoderSelector::span_encoder_builder`
            // maps it to `BpeBacktrackSpanEncoder`. Naming the algorithm
            // rather than the alias is what makes the row readable.
            version: "0.9.2 (BpeBacktrack)",
            lang: "rust",
            class: Class::Native,
            url: "https://github.com/zspacelabs/wordchipper",
            also_computes: "",
            internally_parallel: false,
        }
    }

    fn encode(&mut self, text: &str, out: &mut Ids) {
        // `Ids` is `Vec<u32>` and the tokenizer is `Tokenizer<u32>`, so this
        // appends straight into the harness's reused buffer — no intermediate
        // Vec, no id conversion. That is the API's own fast path
        // (`try_encode` allocates a Vec per call on top of this), not a
        // shortcut around the public surface.
        //
        // `None` = accept all special tokens; there are none (see module docs).
        // A failure leaves `out` short, which changes the id hash, so a broken
        // cell is reported as a mismatch rather than as a fast one.
        let _ = self.tok.try_encode_append(text, out, None);
    }
}
