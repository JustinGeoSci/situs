// fuzz/fuzz_targets/query.rs

#![no_main]

//! `Index::query` against brute force. The input bytes decode to 1–8 query
//! windows and up to 512 entry boxes; every bound is drawn from a small table
//! (halves from −6 to 6, both zeros, ±1e300, and ±1 and ±2 times
//! `f64::MIN_POSITIVE`), so boxes often touch, coincide and share bounds,
//! the inputs that test the closed-interval rule. The target asserts:
//!
//! - no panic;
//! - for every window, the ids `query` reports, sorted, equal the ids of the
//!   entries whose boxes meet the window, sorted, by a closed-interval test
//!   written out here from the bounds (not `Rect::intersects`, so a defect in
//!   that test cannot hide in the oracle);
//! - `len` is the entry count, and `bounds` is the exact union of the boxes;
//! - `try_query` with a visitor that breaks after the `j`-th hit, `j` chosen
//!   from the input's last byte, visits exactly the first `j` ids of
//!   `query`'s sequence and returns that break; without a hit to break on it
//!   returns `Continue`. Checked on the first window.
//!
//! The bytes are decoded by hand rather than through the `arbitrary` crate.

use core::ops::ControlFlow;

mod common;

use common::{MAX_BOXES, meets, rect};
use libfuzzer_sys::fuzz_target;
use situs::{Index, Rect};

const MAX_WINDOWS: usize = 8;

/// `try_query` breaking after hit `j` (from `byte`) visits `query`'s first
/// `j` ids and returns `Break(j)`.
fn check_prefix(index: &Index, w: &Rect, sequence: &[u32], byte: u8) {
    let mut seen = Vec::new();
    if sequence.is_empty() {
        let got = index.try_query(w, |id| {
            seen.push(id);
            ControlFlow::Break(())
        });
        assert_eq!(
            got,
            ControlFlow::Continue(()),
            "try_query broke with no hit"
        );
        assert!(seen.is_empty());
        return;
    }
    let j = 1 + usize::from(byte) % sequence.len();
    let got = index.try_query(w, |id| {
        seen.push(id);
        if seen.len() == j {
            ControlFlow::Break(j)
        } else {
            ControlFlow::Continue(())
        }
    });
    assert_eq!(
        got,
        ControlFlow::Break(j),
        "try_query did not return its break"
    );
    assert_eq!(seen, sequence[..j], "try_query is not a prefix of query");
}

fuzz_target!(|data: &[u8]| {
    let Some((&head, rest)) = data.split_first() else {
        return;
    };
    let n_windows = 1 + usize::from(head) % MAX_WINDOWS;
    let mut chunks = rest.chunks_exact(4);
    let windows: Vec<Rect> = chunks.by_ref().take(n_windows).map(rect).collect();
    if windows.is_empty() {
        return;
    }
    let entries: Vec<(Rect, u32)> = chunks
        .take(MAX_BOXES)
        .enumerate()
        .map(|(i, b)| (rect(b), u32::try_from(i).expect("at most 512 entries")))
        .collect();

    let index = Index::bulk_load(entries.clone());
    assert_eq!(index.len(), entries.len());
    assert_eq!(index.is_empty(), entries.is_empty());
    match index.bounds() {
        None => assert!(
            entries.is_empty(),
            "an index of {} entries has no bounds",
            entries.len()
        ),
        Some(b) => {
            let min = |f: fn(&Rect) -> f64| {
                entries
                    .iter()
                    .map(|e| f(&e.0))
                    .fold(f64::INFINITY, f64::min)
            };
            let max = |f: fn(&Rect) -> f64| {
                entries
                    .iter()
                    .map(|e| f(&e.0))
                    .fold(f64::NEG_INFINITY, f64::max)
            };
            assert_eq!(
                (b.min_x(), b.min_y(), b.max_x(), b.max_y()),
                (
                    min(Rect::min_x),
                    min(Rect::min_y),
                    max(Rect::max_x),
                    max(Rect::max_y)
                ),
                "bounds are not the union of the boxes"
            );
        }
    }

    for (k, w) in windows.iter().enumerate() {
        let mut sequence = Vec::new();
        index.query(w, |id| sequence.push(id));
        if k == 0 {
            check_prefix(&index, w, &sequence, data[data.len() - 1]);
        }
        let mut got = sequence;
        got.sort_unstable();
        let mut want: Vec<u32> = entries
            .iter()
            .filter(|(r, _)| meets(r, w))
            .map(|(_, id)| *id)
            .collect();
        want.sort_unstable();
        assert_eq!(got, want, "query {w:?} differs from brute force");
    }
});
