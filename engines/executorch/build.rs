use std::path::{Path, PathBuf};

const REV: &str = "9b96c3941d8a1bd9dfe8261ac066f35d272c1959";

fn main() {
    println!("cargo::rustc-check-cfg=cfg(have_executorch)");
    println!("cargo:rerun-if-changed=shim/shim.cpp");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=TOKBENCH_EXECUTORCH_DIR");

    let engine_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let vendor = match std::env::var("TOKBENCH_EXECUTORCH_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => engine_dir.join("vendor"),
    };
    let src = vendor.join("src");
    let build = vendor.join("build");

    let archives = find_archives(&build);
    let has_core = archives.iter().any(|a| stem(a) == "tokenizers");
    if !has_core || !src.join("include").is_dir() {
        println!(
            "cargo:warning=tokbench-executorch: no vendored library at {} \
             — engine will report Unsupported. Run engines/executorch/scripts/vendor_executorch.sh",
            vendor.display()
        );
        return;
    }

    let mut cc = cc::Build::new();
    cc.cpp(true)
        .std("c++20")
        .file(engine_dir.join("shim/shim.cpp"))
        .include(src.join("include"))
        .include(src.join("third-party/json/single_include"))
        .include(src.join("third-party/re2"))
        .include(src.join("third-party/abseil-cpp"))
        .include(src.join("third-party/llama.cpp-unicode/include"))
        .include(src.join("third-party/sentencepiece"))
        .include(src.join("third-party/sentencepiece/src"))
        .warnings(false);
    cc.compile("tokbench_et_shim");

    let mut dirs: Vec<PathBuf> = archives
        .iter()
        .filter_map(|a| a.parent().map(Path::to_path_buf))
        .collect();
    dirs.sort();
    dirs.dedup();
    for d in &dirs {
        println!("cargo:rustc-link-search=native={}", d.display());
    }

    let ordered = order_archives(&archives);
    let emit = |names: &[String]| {
        for n in names {
            if n == "regex_lookahead" {
                println!("cargo:rustc-link-lib=static:+whole-archive={n}");
            } else {
                println!("cargo:rustc-link-lib=static={n}");
            }
        }
    };
    emit(&ordered);

    if !cfg!(target_vendor = "apple") {
        emit(&ordered);
    }

    println!("cargo::rustc-cfg=have_executorch");

    println!(
        "cargo:rustc-env=TOKBENCH_EXECUTORCH_VERSION=HFTokenizer @ {}",
        &REV[..9]
    );
}

fn stem(p: &Path) -> String {
    p.file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.strip_prefix("lib").unwrap_or(s))
        .unwrap_or_default()
        .to_string()
}

fn find_archives(build: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![build.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x == "a") {
                out.push(p);
            }
        }
    }
    out
}

fn order_archives(archives: &[PathBuf]) -> Vec<String> {
    let mut names: Vec<String> = archives.iter().map(|p| stem(p)).collect();
    names.sort();
    names.dedup();

    let rank = |n: &str| -> u8 {
        match n {
            "tokenizers" => 0,
            "regex_lookahead" => 1,
            _ if n.starts_with("pcre2") => 2,
            "sentencepiece" | "sentencepiece_train" => 3,
            "re2" => 4,
            _ => 5,
        }
    };
    names.sort_by_key(|n| rank(n));
    names
}
