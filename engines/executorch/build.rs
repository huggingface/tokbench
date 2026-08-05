//! Compile the C shim and link pytorch-labs/tokenizers, *if* someone has run
//! `scripts/vendor_executorch.sh`. If they have not, this script does nothing
//! and the crate still compiles — `build()` then reports `Unsupported` with
//! instructions. Nobody should have to build a C++ tree with five submodules
//! to run `cargo check` on the workspace.
//!
//! The detection is a single question: is there a `libtokenizers.a` under
//! `vendor/build`? Presence of the archive is what actually decides whether
//! the link can succeed, so it is what the cfg is keyed on.

use std::path::{Path, PathBuf};

/// Keep in sync with `REV` in `scripts/vendor_executorch.sh`.
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

    // --- compile the shim ---------------------------------------------------
    //
    // C++20 because upstream's CMakeLists sets CMAKE_CXX_STANDARD 20 and the
    // headers we include are compiled under it; a lower standard fails on
    // hf_tokenizer.h.
    let mut cc = cc::Build::new();
    cc.cpp(true)
        .std("c++20")
        .file(engine_dir.join("shim/shim.cpp"))
        .include(src.join("include"))
        // nlohmann/json.hpp, included by hf_tokenizer.h.
        .include(src.join("third-party/json/single_include"))
        // re2/re2.h and the llama.cpp unicode tables, reachable from the
        // pre-tokenizer headers. re2.h in turn includes absl/base/call_once.h,
        // so abseil's root has to be on the path too.
        .include(src.join("third-party/re2"))
        .include(src.join("third-party/abseil-cpp"))
        .include(src.join("third-party/llama.cpp-unicode/include"))
        // sentencepiece_processor.h, reachable via sentencepiece.h.
        .include(src.join("third-party/sentencepiece"))
        .include(src.join("third-party/sentencepiece/src"))
        .warnings(false);
    cc.compile("tokbench_et_shim");

    // --- link the C++ library ----------------------------------------------
    let mut dirs: Vec<PathBuf> = archives
        .iter()
        .filter_map(|a| a.parent().map(Path::to_path_buf))
        .collect();
    dirs.sort();
    dirs.dedup();
    for d in &dirs {
        println!("cargo:rustc-link-search=native={}", d.display());
    }

    // `regex_lookahead` must be whole-archived. Its only job is to run
    //
    //     static bool registered = register_override_fallback_regex(...);
    //
    // at load time (src/regex_lookahead.cpp). No symbol in it is referenced by
    // anything, so an ordinary link drops the object file and the tokenizer
    // silently reverts to RE2-only — which cannot compile the `\s+(?!\S)`
    // lookahead in a HuggingFace ByteLevel pre-tokenizer, so every byte-level
    // model would fail to load. Upstream's CMake solves this with
    // `target_link_options_shared_lib()` (-force_load / --whole-archive).
    //
    // We cannot reuse that here: `cargo:rustc-link-arg` applies only to the
    // targets of the package that emitted it, so a flag set by this build
    // script would never reach the link of the `tokbench` driver binary that
    // actually consumes this crate. The `+whole-archive` link modifier on
    // `rustc-link-lib` does propagate transitively, and rustc lowers it to the
    // right per-platform flag.
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

    // Abseil is a few dozen mutually-referencing archives. Apple's linker
    // resolves archives iteratively so order does not matter there, but an ELF
    // linker makes a single ordered pass and would need `--start-group`. That
    // flag is a `rustc-link-arg`, which (see above) will not propagate — so
    // emit the list a second time, which achieves the same fixpoint for the
    // one level of back-reference these libraries actually have.
    if !cfg!(target_vendor = "apple") {
        emit(&ordered);
    }

    println!("cargo::rustc-cfg=have_executorch");

    // The string the report prints. Built here rather than in src/lib.rs so
    // the pinned commit has one definition in the compiled artifact: "which
    // variant, at which revision" is the whole identity of this row, and
    // `executorch` on its own would not identify what ran.
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

/// Every `.a` in the build tree, which is where the archives stay: upstream
/// adds sentencepiece with `EXCLUDE_FROM_ALL`, so `cmake --install` omits
/// `libsentencepiece.a` even though `libtokenizers.a` has a PUBLIC dependency
/// on it. Consuming the build tree keeps every archive in one place.
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

/// Dependents before dependencies, which is what an ELF linker wants and what
/// Apple's linker does not care about.
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
            _ => 5, // abseil and anything else
        }
    };
    names.sort_by_key(|n| rank(n));
    names
}
