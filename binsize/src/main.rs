use std::path::PathBuf;

#[allow(unused_imports)]
use tokbench_core::{Build, Engine, Model, Unsupported};

#[allow(unreachable_code)]
fn engine(model: &Model) -> Option<Result<Box<dyn Engine>, Unsupported>> {
    #[cfg(feature = "hf-tokenizers")]
    return Some(tokbench_hf_tokenizers::Adapter::build(model));
    #[cfg(feature = "pipeline")]
    return Some(tokbench_pipeline::Adapter::build(model));
    #[cfg(feature = "kitoken")]
    return Some(tokbench_kitoken::Adapter::build(model));
    #[cfg(feature = "fastokens")]
    return Some(tokbench_fastokens::Adapter::build(model));
    #[cfg(feature = "tokie")]
    return Some(tokbench_tokie::Adapter::build(model));
    #[cfg(feature = "tiktoken")]
    return Some(tokbench_tiktoken::Adapter::build(model));
    #[cfg(feature = "rust-gems-bpe")]
    return Some(tokbench_rust_gems_bpe::Adapter::build(model));
    #[cfg(feature = "wordchipper")]
    return Some(tokbench_wordchipper::Adapter::build(model));
    #[cfg(feature = "sentencepiece")]
    return Some(tokbench_sentencepiece::Adapter::build(model));
    #[cfg(feature = "gigatoken")]
    return Some(tokbench_gigatoken::Adapter::build(model));
    #[cfg(feature = "llamacpp")]
    return Some(tokbench_llamacpp::Adapter::build(model));
    #[cfg(feature = "iree")]
    return Some(tokbench_iree::Adapter::build(model));
    #[cfg(feature = "executorch")]
    return Some(tokbench_executorch::Adapter::build(model));

    let _ = model;
    None
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(args.next().unwrap_or_else(|| ".".into()));
    let text = args
        .next()
        .unwrap_or_else(|| "The quick brown fox jumps 123.".into());

    let model = Model {
        name: "binsize".into(),
        dir,
    };

    match engine(&model) {
        None => println!("baseline: no engine linked"),
        Some(Err(e)) => println!("unsupported: {e}"),
        Some(Ok(mut e)) => {
            let mut out = Vec::new();
            e.encode(&text, &mut out);
            // Printing the count is what keeps the encode path reachable.
            println!("{} ids", out.len());
        }
    }
}
