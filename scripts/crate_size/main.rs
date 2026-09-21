use std::hint::black_box;

fn main() {
    #[cfg(feature = "convert")]
    black_box(tk_convert::canonicalize_str(
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]}}"#,
    ).ok());

    #[cfg(feature = "train")]
    {
        use tk_train::Trainer;
        let mut trainer = tk_train::WordLevelTrainer::default();
        trainer.feed(["hello world"].into_iter(), |s| {
            Ok(s.split_whitespace().map(str::to_owned).collect())
        }).unwrap();
        let mut model = tk_encode::models::wordlevel::WordLevel::default();
        black_box(trainer.train(&mut model).unwrap());
    }

    #[cfg(feature = "serialize")]
    {
        let path = std::env::args().nth(1).expect("tokenizer.json path");
        let tok = tk_serialize::from_json_file(path).unwrap();
        let out = tok
            .encode(black_box("hello world"), &tk_encode::pipeline::EncodeOptions::no_specials())
            .wait()
            .unwrap();
        println!("{}", out[0].len());
    }

    #[cfg(not(feature = "serialize"))]
    {
        use std::collections::BTreeMap;
        use tk_encode::models::bpe::{BpeConfig, Merges, PipelineBPE, Vocab};
        use tk_encode::pipeline::{
            EncodeOptions, PipelineModel, PipelinePostProcessor, PipelinePreTokenizer,
            PipelineTokenizer,
        };
        use tk_encode::vocab::bucket_added_vocabulary::AddedVocabulary;

        let vocab: Vocab = [
            ("h", 0), ("e", 1), ("l", 2), ("o", 3),
            ("he", 4), ("hel", 5), ("hell", 6), ("hello", 7),
        ].into_iter().map(|(s, id)| (s.to_owned(), id)).collect();
        let merges: Merges = [("h", "e"), ("he", "l"), ("hel", "l"), ("hell", "o")]
            .into_iter().map(|(a, b)| (a.to_owned(), b.to_owned())).collect();
        let model = PipelineBPE::from_config(BpeConfig {
            vocab, merges, ..BpeConfig::default()
        }).unwrap();
        let pre = PipelinePreTokenizer::Whitespace(
            tk_encode::pre_tokenizers::whitespace::Whitespace,
        );
        let tok = PipelineTokenizer::from_parts(
            AddedVocabulary::new(), Vec::new(), pre, PipelineModel::BPE(model),
            PipelinePostProcessor::default(), None, BTreeMap::new(), None, None,
        );
        let out = tok.encode(black_box("hello"), &EncodeOptions::no_specials())
            .wait().unwrap();
        println!("{}", out[0].len());
    }
}
