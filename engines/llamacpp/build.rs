use std::path::{Path, PathBuf};
use std::process::Command;

struct Found {
    includes: Vec<PathBuf>,
    libdir: PathBuf,
    libs: Vec<String>,
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
        let cxx = if cfg!(target_os = "macos") {
            "c++"
        } else {
            "stdc++"
        };
        println!("cargo:rustc-link-lib=dylib={cxx}");
    }
    println!("cargo:rustc-cfg=llamacpp");
    println!(
        "cargo:rustc-env=TOKBENCH_LLAMACPP_VERSION={}",
        found.version
    );
    eprintln!(
        "tokbench-llamacpp: linking llama.cpp {} from {}",
        found.version,
        found.libdir.to_string_lossy()
    );
}

fn locate() -> Option<Found> {
    let mut inc_candidates: Vec<PathBuf> = Vec::new();
    let mut lib_candidates: Vec<PathBuf> = Vec::new();
    let mut version: Option<String> = None;

    if let Ok(v) = std::env::var("LLAMA_CPP_INCLUDE") {
        inc_candidates.extend(v.split(':').filter(|s| !s.is_empty()).map(PathBuf::from));
    }
    if let Ok(v) = std::env::var("LLAMA_CPP_LIB") {
        lib_candidates.extend(v.split(':').filter(|s| !s.is_empty()).map(PathBuf::from));
    }

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

    if let Some(v) = pkg_config_var("--modversion") {
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

    for prefix in ["/opt/homebrew", "/usr/local", "/usr"] {
        for formula in ["opt/llama.cpp", "opt/ggml", ""] {
            let base = Path::new(prefix).join(formula);
            inc_candidates.push(base.join("include"));
            lib_candidates.push(base.join("lib"));
        }
    }

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
    let out = Command::new("pkg-config")
        .arg(arg)
        .arg("llama")
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?.trim().to_string();
    (!s.is_empty()).then_some(s)
}
