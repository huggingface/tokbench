//! Resident set size, for the "how much RAM does this engine actually hold"
//! column.
//!
//! # Why this is measured in a child process
//!
//! The number that matters is the memory a *loaded* tokenizer occupies: its
//! vocabulary, tries, automata and caches. Measuring several engines in one
//! process cannot produce that number. Allocators do not return freed pages to
//! the OS promptly, so engine B built after engine A is dropped reuses A's
//! pages and reports a delta near zero — making whichever engine ran second
//! look free. Peak RSS has the mirror problem: it attributes the high-water
//! mark of the whole run to every engine.
//!
//! So the driver re-spawns itself once per engine (`--memory <engine>`), and
//! each child reports its own numbers. One engine per process, no sharing, no
//! ordering effects.
//!
//! # What the two numbers mean
//!
//! * `delta` — RSS after loading the model and encoding, minus RSS at start-up.
//!   This is the engine's own footprint, with the runtime's baseline removed.
//! * `peak`  — the process high-water mark, which catches engines that
//!   transiently allocate far more than they retain (a common trait of
//!   builders that construct an intermediate vocabulary before compacting it).
//!   An engine can have a small `delta` and a large `peak`; that difference is
//!   what decides whether it fits in a constrained container.
//!
//! Zero dependencies here, like the rest of `core`: reading `/proc` or shelling
//! out to `ps` is enough, and pulling in a memory-instrumentation crate would
//! link an allocator shim into the very process whose memory is being measured.

use std::process::Command;

/// Current resident set size in bytes, or `None` if it cannot be determined.
pub fn current() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        if let Some(kb) = proc_status_kb("VmRSS:") {
            return Some(kb * 1024);
        }
    }
    ps_rss_kb().map(|kb| kb * 1024)
}

/// Peak resident set size ("high-water mark") in bytes.
///
/// Linux exposes this directly. Elsewhere there is no cheap dependency-free
/// way to read it, so this returns `None` and the report shows peak as
/// unavailable rather than silently substituting the current value — which
/// would understate transient spikes and quietly mislead.
pub fn peak() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        return proc_status_kb("VmHWM:").map(|kb| kb * 1024);
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

#[cfg(target_os = "linux")]
fn proc_status_kb(key: &str) -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|l| {
        let rest = l.strip_prefix(key)?;
        rest.split_whitespace().next()?.parse::<u64>().ok()
    })
}

/// `ps` reports RSS in kilobytes on both macOS and Linux. Slow (it forks), but
/// this is called twice per child process, never inside a timed region.
fn ps_rss_kb() -> Option<u64> {
    let pid = std::process::id();
    let out = Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse::<u64>()
        .ok()
}

/// RSS delta and peak around `f`, in bytes.
///
/// Only meaningful when the calling process does one engine and then exits;
/// see the module docs.
pub fn around<T>(f: impl FnOnce() -> T) -> (T, Option<u64>, Option<u64>) {
    let before = current();
    let value = f();
    let after = current();
    let delta = match (before, after) {
        // Saturating: a GC or allocator return can make `after` smaller, and a
        // negative footprint is not a meaningful thing to report.
        (Some(b), Some(a)) => Some(a.saturating_sub(b)),
        _ => None,
    };
    (value, delta, peak())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// RSS must be readable on the host, and must visibly grow when a large
    /// live allocation is held. Without this, a broken reader would silently
    /// report 0 MB for every engine and look like a great result.
    #[test]
    fn rss_moves_with_a_real_allocation() {
        let Some(before) = current() else {
            eprintln!("RSS unavailable on this platform; skipping");
            return;
        };
        assert!(before > 0, "a running process has non-zero RSS");

        // 64 MB, touched so the pages are actually resident, and kept alive
        // across the second reading.
        let mut big: Vec<u8> = vec![0; 64 << 20];
        for i in (0..big.len()).step_by(4096) {
            big[i] = 1;
        }
        let after = current().expect("RSS readable");
        assert!(
            after > before,
            "RSS should grow after touching 64 MB (before={before}, after={after})"
        );
        std::hint::black_box(&big);
    }
}
