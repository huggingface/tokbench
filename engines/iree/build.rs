//! Link the vendored IREE tokenizer static library — if it is there.
//!
//! The hard requirement is that this crate compiles for someone who has never
//! heard of `vendor_iree.sh`. `cargo check` on a fresh clone must not fail,
//! must not try to fetch anything, and must not need a C compiler. So this
//! build script does exactly one thing: it looks for
//! `vendor/lib/libiree_tokenizer.a` and, if present, emits link directives and
//! sets `cfg(iree_available)`. If absent it emits nothing but a hint, and
//! `src/lib.rs` compiles a stub whose `build()` returns `Unsupported` naming
//! the script to run.
//!
//! Note the split of labour: the C compilation happens in the vendor script,
//! not here. That is deliberate. Compiling ~110 IREE translation units from
//! build.rs would make every `cargo check` of the whole workspace depend on a
//! working C toolchain and on IREE's sources being present, which is exactly
//! the coupling the graceful-degradation rule exists to prevent. Vendoring is
//! an explicit, opt-in step; linking is automatic once it has been done.

use std::path::PathBuf;

fn main() {
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let libdir = root.join("vendor").join("lib");
    let lib = libdir.join("libiree_tokenizer.a");

    // Re-link when the vendored artifact appears, changes or is removed, and
    // when the pin moves.
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", lib.display());
    println!(
        "cargo:rerun-if-changed={}",
        root.join("vendor").join("COMMIT").display()
    );
    // Without this, newer cargo warns about the `iree_available` cfg it has
    // never seen declared.
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

    // Report the exact IREE revision that was linked, so the version column in
    // the report names a commit rather than a guess. Falls back to "unknown"
    // rather than to a plausible-looking lie.
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
