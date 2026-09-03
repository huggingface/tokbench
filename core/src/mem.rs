//! How much memory an engine actually holds.
//!
//! # Why this is not RSS
//!
//! This column used to report the RSS delta around loading and warming an
//! engine. That number is misleading, and misleading in a specific direction:
//! RSS is a high-water mark, so it never falls when memory is freed. An engine
//! that parses `tokenizer.json` into hash maps, builds a compact structure from
//! them and drops the maps keeps every one of those pages resident, and gets
//! billed for memory it no longer holds.
//!
//! That is exactly the shape of a loader built around a minimal perfect hash,
//! so RSS reported the *opposite* of what such a design achieves — on llama-3
//! it ranked the engine holding the least (12 MB) as the one holding the most
//! (255 MB). The ranking was an artefact of the metric.
//!
//! So the footprint reported here is live heap: allocated and not yet freed,
//! right now. libc already tracks it, so this asks libc rather than shimming
//! the allocator — a `#[global_allocator]` wrapper would add atomics to every
//! allocation in the same binary that runs the timing loop, and would miss
//! allocations made by C and C++ engines through FFI, which this catches.
//!
//! # Still measured in a child process
//!
//! One engine per process, spawned by the driver (`--memory <engine>`).
//! Allocators do not return freed pages promptly, so several engines in one
//! process would let engine B reuse engine A's freed arenas and report a
//! footprint near zero. Live heap is less sensitive to this than RSS, but the
//! isolation is free and removes the ordering effect entirely.
//!
//! Caveat worth knowing: this counts `malloc`, not `mmap`. An engine that maps
//! a prebuilt image instead of allocating it reads low here, and that number is
//! a floor rather than a measurement.

/// Live heap bytes: allocated and not yet freed, at this instant.
///
/// `None` on platforms exposing neither interface, in which case the report
/// omits the column rather than substituting a number that means something
/// else.
pub fn live_heap() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        // <malloc/malloc.h>. Returned by value; five `size_t` fields.
        #[repr(C)]
        struct Mstats {
            bytes_total: usize,
            chunks_used: usize,
            bytes_used: usize,
            chunks_free: usize,
            bytes_free: usize,
        }
        extern "C" {
            fn mstats() -> Mstats;
        }
        // SAFETY: no arguments, no pointers, returns a plain POD by value.
        Some(unsafe { mstats() }.bytes_used as u64)
    }
    #[cfg(target_os = "linux")]
    {
        // <malloc.h>. `uordblks` is the live total from the main arena;
        // `hblkhd` covers the large allocations glibc serves with mmap, which a
        // vocabulary of any size will hit.
        #[repr(C)]
        struct MallInfo2 {
            arena: usize,
            ordblks: usize,
            smblks: usize,
            hblks: usize,
            hblkhd: usize,
            usmblks: usize,
            fsmblks: usize,
            uordblks: usize,
            fordblks: usize,
            keepcost: usize,
        }
        extern "C" {
            fn mallinfo2() -> MallInfo2;
        }
        // SAFETY: as above.
        let mi = unsafe { mallinfo2() };
        Some((mi.uordblks + mi.hblkhd) as u64)
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole point of this reader is the property RSS lacks: it must go
    /// back *down* when memory is freed. A reader that only ever grows would
    /// reproduce the exact bug this module was written to remove, and every
    /// engine would silently be ranked by its loader instead of its structures.
    #[test]
    fn live_heap_rises_and_falls() {
        let Some(base) = live_heap() else {
            eprintln!("live heap unavailable on this platform; skipping");
            return;
        };
        let big: Vec<u8> = vec![7; 64 << 20];
        let held = live_heap().expect("readable");
        assert!(
            held > base + (32 << 20),
            "64 MB held should show up (base={base}, held={held})"
        );
        drop(big);
        let freed = live_heap().expect("readable");
        assert!(
            freed < held - (32 << 20),
            "freeing 64 MB must reduce live heap (held={held}, freed={freed})"
        );
    }
}
