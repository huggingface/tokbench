//! Find an llama.cpp that is already on the machine, and link only against it.
//!
//! # Why this does not build llama.cpp
//!
//! The obvious wiring is `llama-cpp-2`, which vendors the upstream tree and
//! compiles it. That turns `cargo check -p tokbench-llamacpp` into a multi-
//! minute build of an entire inference engine — CUDA/Metal/BLAS backends, the
//! sampler, the KV cache, the server helpers — none of which affects the one
//! number this crate exists to produce. So instead: locate an installed
//! llama.cpp (`brew install llama.cpp`, a distro package, or a local build
//! pointed at by `LLAMA_CPP_DIR`), compile the 40-line shim in `src/shim.c`
//! against its headers, and link `-lllama`. That is a one-second build.
//!
//! Linking the shared `libllama` does not link the backends either: on macOS
//! and Linux the ggml backend libraries are separate shared objects that
//! `libllama` references by its own install name / SONAME, and with
//! `vocab_only = true` none of them is ever loaded at run time (verified: a
//! vocab-only load prints no backend or device line at all).
//!
//! # Why it must not fail the build
//!
//! Fairness rule: every engine in this workspace is optional and the workspace
//! has to build on a machine with no system libraries at all. A missing
//! llama.cpp is therefore not an error — it emits one `cargo:warning`, leaves
//! the `llamacpp` cfg unset, and `src/lib.rs` compiles to a stub whose
//! `build()` returns `Unsupported` with instructions. Nothing in here panics.
//!
//! # Where it looks, in order
//!
//! 1. `LLAMA_CPP_INCLUDE` / `LLAMA_CPP_LIB` — explicit override, `:`-separated.
//! 2. `LLAMA_CPP_DIR` — an install prefix or a built source tree; the usual
//!    subdirectories of both layouts are tried.
//! 3. `pkg-config llama` — what `brew install llama.cpp` and most distro
//!    packages register. Also the source of the reported version.
//! 4. Homebrew / FHS prefixes.
//!
//! `ggml.h` is searched for separately from `llama.h`: `llama.h` includes it,
//! but Homebrew splits llama.cpp and ggml into two formulae with two include
//! directories, so finding one does not imply finding the other.

use std::path::{Path, PathBuf};
use std::process::Command;

struct Found {
    /// Every include dir needed to compile `#include "llama.h"` — at least one
    /// holding `llama.h`, at least one holding `ggml.h`, often two different
    /// directories.
    includes: Vec<PathBuf>,
    libdir: PathBuf,
    /// Libraries to pass to the linker. One entry (`llama`) for a shared
    /// install; the ggml archives too when only static libs are present.
    libs: Vec<String>,
    /// True when the only llama library found is a `.a`, which means the ggml
    /// archives and the C++ runtime have to be named explicitly.
    static_link: bool,
    version: String,
}

fn main() {
    println!("cargo:rustc-check-cfg=cfg(llamacpp)");
    println!("cargo:rerun-if-changed=src/shim.c");
    for var in [
        "LLAMA_CPP_DIR",
        "LLAMA_CPP_INCLUDE",
        "LLAMA_CPP_LIB",
        "LLAMA_CPP_VERSION",
    ] {
        println!("cargo:rerun-if-env-changed={var}");
    }

    let Some(found) = locate() else {
        // `env!` in lib.rs needs this set on both paths or the stub will not
        // compile.
        println!("cargo:rustc-env=TOKBENCH_LLAMACPP_VERSION=absent");
        println!(
            "cargo:warning=llama.cpp not found (looked at pkg-config `llama`, \
             $LLAMA_CPP_DIR, Homebrew and /usr/local). tokbench-llamacpp will \
             report Unsupported. Install with `brew install llama.cpp`, or set \
             LLAMA_CPP_DIR to a built checkout."
        );
        return;
    };

    let mut build = cc::Build::new();
    build.file("src/shim.c").warnings(false);
    for inc in &found.includes {
        build.include(inc);
    }
    build.compile("tokbench_llama_shim");

    println!(
        "cargo:rustc-link-search=native={}",
        found.libdir.to_string_lossy()
    );
    for lib in &found.libs {
        println!("cargo:rustc-link-lib={lib}");
    }
    if found.static_link {
        // Static llama.cpp is C++ and does not carry its runtime with it.
        let cxx = if cfg!(target_os = "macos") {
            "c++"
        } else {
            "stdc++"
        };
        println!("cargo:rustc-link-lib=dylib={cxx}");
    }
    println!(
        "cargo:rustc-env=TOKBENCH_LLAMACPP_VERSION={}",
        found.version
    );
}

fn locate() -> Option<Found> {
    let mut inc_candidates: Vec<PathBuf> = Vec::new();
    let mut lib_candidates: Vec<PathBuf> = Vec::new();
    let mut version: Option<String> = None;

    // 1. Explicit override wins, and is the only way to be sure on a machine
    //    with several llama.cpp builds.
    if let Ok(v) = std::env::var("LLAMA_CPP_INCLUDE") {
        inc_candidates.extend(v.split(':').filter(|s| !s.is_empty()).map(PathBuf::from));
    }
    if let Ok(v) = std::env::var("LLAMA_CPP_LIB") {
        lib_candidates.extend(v.split(':').filter(|s| !s.is_empty()).map(PathBuf::from));
    }

    // 2. A prefix or a source tree. `build/bin` is where a cmake build of
    //    llama.cpp drops the shared objects; `ggml/include` is where the
    //    in-tree ggml headers live before installation.
    if let Ok(dir) = std::env::var("LLAMA_CPP_DIR") {
        let d = PathBuf::from(dir);
        for sub in ["include", "ggml/include", "src", "ggml/src"] {
            inc_candidates.push(d.join(sub));
        }
        for sub in ["lib", "lib64", "build/bin", "build/src", "build/ggml/src"] {
            lib_candidates.push(d.join(sub));
        }
        inc_candidates.push(d.clone());
    }

    // 3. pkg-config. Also the only place that knows the exact version, which
    //    the report prints — a throughput number without a version is not a
    //    result (see core's Info::version).
    if let Some(v) = pkg_config_var("--modversion") {
        // llama.cpp's .pc reports 0.0.<build>, and the upstream release tag for
        // that build is b<build>. Reporting "b9140" makes the row traceable to
        // a commit; reporting "0.0.9140" does not.
        version = Some(match v.strip_prefix("0.0.") {
            Some(build) => format!("b{build}"),
            None => v,
        });
    }
    if let Some(v) = pkg_config_var("--variable=includedir") {
        inc_candidates.push(PathBuf::from(v));
    }
    if let Some(v) = pkg_config_var("--variable=libdir") {
        lib_candidates.push(PathBuf::from(v));
    }

    // 4. Well-known prefixes. Homebrew keeps llama.cpp and ggml in separate
    //    formulae, hence both.
    for prefix in ["/opt/homebrew", "/usr/local", "/usr"] {
        for formula in ["opt/llama.cpp", "opt/ggml", ""] {
            let base = Path::new(prefix).join(formula);
            inc_candidates.push(base.join("include"));
            lib_candidates.push(base.join("lib"));
        }
    }

    // Resolve: keep every existing candidate that actually carries a header we
    // need, and require that both headers turned up somewhere.
    let mut includes: Vec<PathBuf> = Vec::new();
    let (mut have_llama_h, mut have_ggml_h) = (false, false);
    for c in inc_candidates {
        let llama = c.join("llama.h").is_file();
        let ggml = c.join("ggml.h").is_file();
        if !llama && !ggml {
            continue;
        }
        have_llama_h |= llama;
        have_ggml_h |= ggml;
        if !includes.contains(&c) {
            includes.push(c);
        }
    }
    if !have_llama_h || !have_ggml_h {
        return None;
    }

    // A shared library is strongly preferred: it is what the packaged builds
    // ship, and its recorded install name/SONAME lets the loader find ggml
    // without this crate having to name it.
    let mut fallback_static: Option<PathBuf> = None;
    let mut libdir: Option<PathBuf> = None;
    for c in &lib_candidates {
        if ["dylib", "so", "dll.a"]
            .iter()
            .any(|ext| c.join(format!("libllama.{ext}")).exists())
        {
            libdir = Some(c.clone());
            break;
        }
        if fallback_static.is_none() && c.join("libllama.a").exists() {
            fallback_static = Some(c.clone());
        }
    }
    let static_link = libdir.is_none();
    let libdir = libdir.or(fallback_static)?;

    let mut libs = vec!["llama".to_string()];
    if static_link {
        // Order matters for a static link: llama depends on ggml, ggml on
        // ggml-base. Only name the ones that are actually there, so a build
        // that fused them into one archive still links.
        for extra in ["ggml", "ggml-cpu", "ggml-base"] {
            if libdir.join(format!("lib{extra}.a")).exists() {
                libs.push(extra.to_string());
            }
        }
    }

    Some(Found {
        includes,
        libdir,
        libs,
        static_link,
        version: version
            .or_else(|| std::env::var("LLAMA_CPP_VERSION").ok())
            .unwrap_or_else(|| "unknown".to_string()),
    })
}

fn pkg_config_var(arg: &str) -> Option<String> {
    let out = Command::new("pkg-config").arg(arg).arg("llama").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?.trim().to_string();
    (!s.is_empty()).then_some(s)
}
