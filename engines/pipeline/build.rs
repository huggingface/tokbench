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

    // The `path = "..."` of the tk-encode dependency, so this never disagrees with what is linked.
    let tree = std::fs::read_to_string(&manifest)
        .ok()
        .and_then(|text| {
            text.lines()
                .find(|line| line.trim_start().starts_with("tk-encode = "))
                .and_then(|line| line.split_once("path = \""))
                .and_then(|(_, rest)| rest.split_once('"'))
                .map(|(path, _)| path.to_string())
        })
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
        None => "tk-encode (path dep not found in Cargo.toml)".to_string(),
    };
    println!("cargo:rustc-env=PIPELINE_TREE_VERSION={version}");
}
