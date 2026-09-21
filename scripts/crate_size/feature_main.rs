use std::collections::BTreeMap;
use std::hint::black_box;
use tk_encode::models::bpe::{BpeConfig, Merges, PipelineBPE, Vocab};
use tk_encode::pipeline::{EncodeOptions, PipelineModel, PipelinePostProcessor,
    PipelinePreTokenizer, PipelineTokenizer};
use tk_encode::vocab::bucket_added_vocabulary::AddedVocabulary;

fn bpe_tokenizer(pre: PipelinePreTokenizer) -> PipelineTokenizer {
    let vocab: Vocab = [("h", 0), ("e", 1), ("l", 2), ("o", 3),
        ("he", 4), ("hel", 5), ("hell", 6), ("hello", 7)]
        .into_iter().map(|(s, id)| (s.to_owned(), id)).collect();
    let merges: Merges = [("h", "e"), ("he", "l"), ("hel", "l"), ("hell", "o")]
        .into_iter().map(|(a, b)| (a.to_owned(), b.to_owned())).collect();
    let model = PipelineBPE::from_config(BpeConfig { vocab, merges,
        ..BpeConfig::default() }).unwrap();
    PipelineTokenizer::from_parts(AddedVocabulary::new(), Vec::new(), pre,
        PipelineModel::BPE(model), PipelinePostProcessor::default(), None,
        BTreeMap::new(), None, None)
}

fn main() {
    let pre = {
        #[cfg(feature = "unicode-scripts")]
        { PipelinePreTokenizer::UnicodeScripts(
            tk_encode::pre_tokenizers::unicode_scripts::UnicodeScripts::new()) }
        #[cfg(not(feature = "unicode-scripts"))]
        { PipelinePreTokenizer::Whitespace(
            tk_encode::pre_tokenizers::whitespace::Whitespace) }
    };
    let tok = bpe_tokenizer(pre);
    let inputs = {
        #[cfg(feature = "parallelism")]
        { vec!["hello".to_owned(); 8] }
        #[cfg(not(feature = "parallelism"))]
        { vec!["hello".to_owned()] }
    };
    black_box(tok.encode(inputs, &EncodeOptions::no_specials()).wait().unwrap());

    #[cfg(feature = "unigram")]
    {
        let model = tk_encode::models::unigram::Unigram::from(vec![
            ("<unk>".into(), 0.0), ("h".into(), -1.0), ("ello".into(), -2.0)],
            Some(0), false).unwrap();
        black_box(model.tokenize("hello").unwrap());
    }
    #[cfg(feature = "wordpiece")]
    {
        let vocab: ahash::AHashMap<String, u32> =
            [("[UNK]".to_owned(), 0), ("hello".to_owned(), 1)].into_iter().collect();
        let model = tk_encode::models::wordpiece::WordPiece::builder()
            .vocab(vocab).build().unwrap();
        black_box(model.tokenize("hello").unwrap());
    }
    #[cfg(feature = "wordlevel")]
    {
        let vocab: ahash::AHashMap<String, u32> =
            [("<unk>".to_owned(), 0), ("hello".to_owned(), 1)].into_iter().collect();
        let model = tk_encode::models::wordlevel::WordLevel::builder()
            .vocab(vocab).build().unwrap();
        black_box(model.tokenize("hello").unwrap());
    }
    #[cfg(feature = "normalizers")]
    {
        use tk_encode::pipeline::Normalizer;
        black_box(tk_encode::normalizers::unicode::NFC.normalize("e\u{301}", 0).unwrap());
    }
}
