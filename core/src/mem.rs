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
