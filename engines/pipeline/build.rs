//! Stamps `Engine::info().version` from the tk-encode worktree this engine is actually compiled
//! against, at build time.
//!
//! It used to be a string literal. A run then recorded numbers under
//! `"tk-encode (#2279 poc/target-encode ...)"` while the path dep pointed at a checkout sitting on
//! a different branch with uncommitted changes — so the report named code that was never measured.
//! A literal cannot be kept in step with a path dep by hand; this can.
//!
//! The path comes from Cargo.toml, so there is one source of truth, and a dirty worktree is
//! reported as such rather than silently.
use std::process::Command;

fn git(tree: &str, args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(tree)
        .args(args)
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn main() {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    println!("cargo:rerun-if-changed={}", manifest.display());

    let dep = std::fs::read_to_string(&manifest).ok().and_then(|text| {
        text.lines()
            .find(|line| line.trim_start().starts_with("tk-encode = "))
            .map(str::to_string)
    });

    // A git dep pins a rev, and that rev *is* the answer -- nothing local to inspect and nothing
    // that can drift. Emit it and stop.
    //
    // Written as nested `if let` rather than a let-chain: this workspace is edition 2021, where let
    // chains are not accepted, and a build script that does not compile takes the whole engine with
    // it.
    if let Some(line) = dep.as_deref() {
        if let Some((_, rest)) = line.split_once("rev = \"") {
            if let Some((rev, _)) = rest.split_once('"') {
                println!("cargo:rustc-env=PIPELINE_TREE_VERSION=tk-encode {}", rev);
                return;
            }
        }
    }

    // A path dep points at a worktree, which CAN drift, so ask git what is actually there.
    let tree = dep
        .as_deref()
        .and_then(|line| line.split_once("path = \""))
        .and_then(|(_, rest)| rest.split_once('"'))
        .map(|(path, _)| path.to_string())
        // tk-encode lives at <tree>/tokenizers/tk-encode; the repo root is two levels up.
        .map(|dep| {
            std::path::Path::new(&dep)
                .ancestors()
                .nth(2)
                .map(|root| root.display().to_string())
                .unwrap_or(dep)
        });

    let version = match tree {
        Some(tree) => {
            println!("cargo:rerun-if-changed={tree}/.git/HEAD");
            let branch = git(&tree, &["rev-parse", "--abbrev-ref", "HEAD"]);
            let rev = git(&tree, &["rev-parse", "--short", "HEAD"]);
            match (branch, rev) {
                (Some(branch), Some(rev)) => {
                    // Uncommitted changes mean the rev does not describe what was compiled. Say so
                    // in the number's own label rather than leaving it to be discovered later.
                    let dirty = match git(&tree, &["status", "--porcelain"]) {
                        Some(status) if !status.is_empty() => " DIRTY",
                        _ => "",
                    };
                    format!("tk-encode {branch}@{rev}{dirty}")
                }
                _ => "tk-encode (worktree not a git checkout)".to_string(),
            }
        }
        None => "tk-encode (dependency not found in Cargo.toml)".to_string(),
    };
    println!("cargo:rustc-env=PIPELINE_TREE_VERSION={version}");
}
