use std::collections::BTreeMap;
use std::hint::black_box;

use tk_encode_size::models::bpe::{BpeConfig, Merges, Vocab, BPE};
use tk_encode_size::pipeline::{EncodeOptions, Model, PostProcessor, PreTokenizer, Tokenizer};
use tk_encode_size::vocab::bucket_added_vocabulary::AddedVocabulary;

fn bpe_tokenizer(pre: PreTokenizer) -> Tokenizer {
    let vocab: Vocab = [
        ("h", 0),
        ("e", 1),
        ("l", 2),
        ("o", 3),
        ("he", 4),
        ("hel", 5),
        ("hell", 6),
        ("hello", 7),
    ]
    .into_iter()
    .map(|(text, id)| (text.to_owned(), id))
    .collect();
    let merges: Merges = [("h", "e"), ("he", "l"), ("hel", "l"), ("hell", "o")]
        .into_iter()
        .map(|(left, right)| (left.to_owned(), right.to_owned()))
        .collect();
    let mut config = BpeConfig::default();
    config.vocab = vocab;
    config.merges = merges;
    let model = BPE::from_config(config).unwrap();
    Tokenizer::from_parts(
        AddedVocabulary::new(),
        Vec::new(),
        pre,
        Model::BPE(model),
        PostProcessor::default(),
        None,
        BTreeMap::new(),
        None,
        None,
    )
}

#[cfg(any(
    feature = "crate-unigram",
    feature = "crate-wordpiece",
    feature = "crate-wordlevel"
))]
fn tokenize<M: tk_encode_size::pipeline::Tokenize>(
    model: &M,
    text: &str,
) -> Vec<tk_encode_size::pipeline::Token> {
    let mut scratch = model.init_scratch();
    let mut tokens = Vec::new();
    model.tokenize(text, &mut scratch, &mut tokens).unwrap();
    tokens
}

fn main() {
    #[cfg(feature = "crate-convert")]
    black_box(
        tk_convert_size::canonicalize_str(
            r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]}}"#,
        )
        .ok(),
    );

    #[cfg(feature = "crate-train")]
    {
        use tk_train_size::Trainer;
        let mut trainer = tk_train_size::WordLevelTrainer::default();
        trainer
            .feed(["hello world"].into_iter(), |text| {
                Ok(text.split_whitespace().map(str::to_owned).collect())
            })
            .unwrap();
        let mut model = tk_encode_size::models::wordlevel::WordLevel::default();
        black_box(trainer.train(&mut model).unwrap());
    }

    let pre = {
        #[cfg(feature = "crate-unicode-scripts")]
        {
            PreTokenizer::UnicodeScripts(
                tk_encode_size::pre_tokenizers::unicode_scripts::UnicodeScripts::new(),
            )
        }
        #[cfg(not(feature = "crate-unicode-scripts"))]
        {
            PreTokenizer::Whitespace(tk_encode_size::pre_tokenizers::whitespace::Whitespace)
        }
    };
    let tokenizer = bpe_tokenizer(pre);
    let inputs = {
        #[cfg(feature = "crate-parallelism")]
        {
            vec!["hello".to_owned(); 8]
        }
        #[cfg(not(feature = "crate-parallelism"))]
        {
            vec!["hello".to_owned()]
        }
    };
    black_box(
        tokenizer
            .encode(inputs, &EncodeOptions::no_specials())
            .wait()
            .unwrap(),
    );

    #[cfg(feature = "crate-serialize")]
    {
        let path = std::env::args().nth(1).expect("tokenizer.json path");
        let tokenizer = tk_serialize_size::from_json_file(path).unwrap();
        black_box(
            tokenizer
                .encode("hello world", &EncodeOptions::no_specials())
                .wait()
                .unwrap(),
        );
    }
    #[cfg(feature = "crate-unigram")]
    {
        let model = tk_encode_size::models::unigram::Unigram::from(
            vec![
                ("<unk>".into(), 0.0),
                ("h".into(), -1.0),
                ("ello".into(), -2.0),
            ],
            Some(0),
            false,
        )
        .unwrap();
        black_box(tokenize(&model, "hello"));
    }
    #[cfg(feature = "crate-wordpiece")]
    {
        let vocab: ahash::AHashMap<String, u32> =
            [("[UNK]".to_owned(), 0), ("hello".to_owned(), 1)]
                .into_iter()
                .collect();
        let mut config = tk_encode_size::models::wordpiece::WordPieceConfig::default();
        config.vocab = vocab;
        let model = tk_encode_size::models::wordpiece::WordPiece::from_config(config).unwrap();
        black_box(tokenize(&model, "hello"));
    }
    #[cfg(feature = "crate-wordlevel")]
    {
        let vocab: ahash::AHashMap<String, u32> =
            [("<unk>".to_owned(), 0), ("hello".to_owned(), 1)]
                .into_iter()
                .collect();
        let model = tk_encode_size::models::wordlevel::WordLevel::builder()
            .vocab(vocab)
            .build()
            .unwrap();
        black_box(tokenize(&model, "hello"));
    }
    #[cfg(feature = "crate-normalizers")]
    {
        use tk_encode_size::pipeline::Normalize;
        black_box(
            tk_encode_size::normalizers::unicode::NFC
                .normalize("e\u{301}", 0)
                .unwrap(),
        );
    }
}
