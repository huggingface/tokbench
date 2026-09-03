//! Find `libblingfiretokdll` if it is on this machine, and link it if so.
//!
//! # The one hard requirement: never break the workspace
//!
//! `tokbench-blingfire` must compile on a machine that has never heard of
//! BlingFire. Almost nobody has the library — it is not a crate, not a system
//! package, and not something `cargo build` can fetch — so a build script that
//! `panic!`s when it cannot find it would turn `--features blingfire` into a
//! landmine and, worse, would make the engine impossible to even typecheck in
//! CI.
//!
//! So this script never fails. It probes, and it reports the outcome to the
//! compiler as a single cfg flag:
//!
//! * **found** — emit the link directives and `cfg(blingfire_linked)`, so
//!   `src/lib.rs` compiles the FFI declarations and the real adapter.
//! * **not found** — emit nothing, so `src/lib.rs` compiles a stub whose
//!   `build()` returns `Unsupported` naming the script that would fix it.
//!
//! Both halves are type-checked the same way; the difference is which one is
//! `#[cfg]`-selected. `cargo check -p tokbench-blingfire` therefore passes in
//! both states, which is the property that matters.
//!
//! # Static before shared, and why that is not a size decision
//!
//! When `vendor/lib` holds the static archives they are preferred, because a
//! dynamically linked build of this crate links cleanly and then fails at run
//! time. The reason is a Cargo rule rather than anything about BlingFire: this
//! package is an rlib, and `cargo:rustc-link-arg` — the only way to inject an
//! `-rpath` — applies to artifacts of *this* package alone. It does not
//! propagate to the `tokbench` binary that ends up doing the link. So the
//! driver would be built with the right `-L` and no runtime search path, and
//! the loader would come up empty unless the user exported
//! `DYLD_LIBRARY_PATH`/`LD_LIBRARY_PATH` by hand.
//!
//! Static archives sidestep the whole question: the code is in the binary and
//! there is nothing to find at startup. `scripts/vendor_blingfire.sh` builds
//! them on the source path for exactly this reason. The shared library remains
//! a supported fallback (it is all Microsoft's prebuilt binaries offer), and
//! the vendor script rewrites its macOS install name to an absolute path so
//! that path works too.
//!
//! # Where it looks
//!
//! 1. `$BLINGFIRE_LIB_DIR` — an explicit override, checked first so a user can
//!    point at a system-wide or pip-installed copy without moving files.
//! 2. `engines/blingfire/vendor/lib` — where the vendor script puts things.
//! 3. `/usr/local/lib`, `/opt/homebrew/lib`, `/usr/lib` — for anyone who ran
//!    BlingFire's `make install`.

use std::env;
use std::path::PathBuf;

fn main() {
    // Re-probe when the override moves. The directory `rerun-if-changed` lines
    // below cover the library appearing or disappearing.
    println!("cargo:rerun-if-env-changed=BLINGFIRE_LIB_DIR");
    println!("cargo:rustc-check-cfg=cfg(blingfire_linked)");

    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let vendor = manifest.join("vendor");

    // NOTE: the *target* OS, not the host. A build script runs on the host, so
    // `cfg!(target_os = ...)` here would describe the wrong machine when
    // cross-compiling.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let shared = match target_os.as_str() {
        "macos" | "ios" => "libblingfiretokdll.dylib",
        "windows" => "blingfiretokdll.dll",
        _ => "libblingfiretokdll.so",
    };
    // libc++ on Apple/BSD, libstdc++ on GNU. BlingFire is C++, so a static
    // link has to name the standard library explicitly; a shared link does not
    // (the .so/.dylib already records its own dependency on it).
    let cxx = match target_os.as_str() {
        "macos" | "ios" | "freebsd" | "openbsd" => "c++",
        _ => "stdc++",
    };

    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Ok(d) = env::var("BLINGFIRE_LIB_DIR") {
        if !d.is_empty() {
            dirs.push(PathBuf::from(d));
        }
    }
    dirs.push(vendor.join("lib"));
    for p in ["/usr/local/lib", "/opt/homebrew/lib", "/usr/lib"] {
        dirs.push(PathBuf::from(p));
    }
    for d in &dirs {
        println!("cargo:rerun-if-changed={}", d.display());
    }

    // The version actually linked, recorded by the vendor script, so the row
    // in the report names the build that produced it rather than whatever
    // version this file was written against.
    let version = std::fs::read_to_string(vendor.join("PROVENANCE"))
        .ok()
        .and_then(|s| {
            s.lines().find_map(|l| {
                // "tag: v0.1.8" -> "0.1.8". The report prints bare version
                // numbers for every other engine; a stray `v` from the git tag
                // would make this the only row spelled differently.
                l.strip_prefix("tag:")
                    .map(|v| v.trim().trim_start_matches('v').to_string())
            })
        })
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "0.1.8".to_string());
    println!("cargo:rustc-env=TOKBENCH_BLINGFIRE_VERSION={version}");

    // Pass 1: static archives, in dependency order. `blingfiretokdll_static`
    // holds the exported C entry points and calls into `fsaClient`; CMake does
    // not merge one static library into another, so both must be named, and a
    // GNU-ld-style linker only resolves left to right.
    for d in &dirs {
        if d.join("libblingfiretokdll_static.a").is_file() && d.join("libfsaClient.a").is_file() {
            println!("cargo:rustc-link-search=native={}", d.display());
            println!("cargo:rustc-link-lib=static=blingfiretokdll_static");
            println!("cargo:rustc-link-lib=static=fsaClient");
            println!("cargo:rustc-link-lib=dylib={cxx}");
            println!("cargo:rustc-cfg=blingfire_linked");
            return;
        }
    }

    // Pass 2: the shared library.
    for d in &dirs {
        if d.join(shared).is_file() {
            println!("cargo:rustc-link-search=native={}", d.display());
            println!("cargo:rustc-link-lib=dylib=blingfiretokdll");
            println!("cargo:rustc-cfg=blingfire_linked");
            return;
        }
    }

    // Pass 3: nothing found. This is a normal, supported outcome — say so once
    // and compile the stub. Not an error, not a panic.
    println!(
        "cargo:warning=libblingfiretokdll not found; the blingfire engine will report \
         Unsupported. Run engines/blingfire/scripts/vendor_blingfire.sh to build it, \
         or set BLINGFIRE_LIB_DIR to a directory containing {shared}."
    );
}
