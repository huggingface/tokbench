use std::path::PathBuf;

fn main() {
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let libdir = root.join("vendor").join("lib");
    let lib = libdir.join("libiree_tokenizer.a");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", lib.display());
    println!(
        "cargo:rerun-if-changed={}",
        root.join("vendor").join("COMMIT").display()
    );
    println!("cargo:rustc-check-cfg=cfg(iree_available)");

    if !lib.exists() {
        println!(
            "cargo:warning=tokbench-iree: {} not found; the engine will report \
             Unsupported. Run `bash engines/iree/scripts/vendor_iree.sh` to build it.",
            lib.display()
        );
        return;
    }

    println!("cargo:rustc-link-search=native={}", libdir.display());
    println!("cargo:rustc-link-lib=static=iree_tokenizer");
    println!("cargo:rustc-cfg=iree_available");

    let commit = std::fs::read_to_string(root.join("vendor").join("COMMIT"))
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    let short = if commit.len() >= 12 {
        commit[..12].to_string()
    } else {
        "unknown".to_string()
    };
    println!("cargo:rustc-env=TOKBENCH_IREE_COMMIT={short}");
}
