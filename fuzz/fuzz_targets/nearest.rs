// fuzz/fuzz_targets/nearest.rs

#![no_main]

//! `Index::nearest_candidates` and `Index::browse` against brute force. The
//! input decodes a query point (two bytes through the bounds table), `k`
//! (one byte), and up to 512 boxes (`common`). The reference bounds for
//! each entry come from a one-entry index, so the target needs no private
//! access. The target asserts:
//!
//! - no panic; every `lower_sq` is finite and non-negative, never above its
//!   `upper_sq`;
//! - set identity: the candidates equal `{ e : lower_sq(e) <= τ }`, with τ
//!   the k-th smallest `upper_sq` over all entries, as a sorted list of
//!   `(id, lower, upper)`;
//! - monotonicity at the root: the root box's lower bound is at most every
//!   entry's lower bound;
//! - `browse` yields every stored entry exactly once, in nondecreasing
//!   `lower_sq`, with each entry's bounds equal to the reference;
//! - built with `--features exact-nearest`: on inputs whose coordinates are
//!   all half-integers, `nearest_exact` on both trees equals an exact `i128`
//!   brute force.

mod common;

use common::{MAX_BOXES, rect};
use libfuzzer_sys::fuzz_target;
use situs::{Index, Rect};

/// Checks `nearest_exact` under the `exact-nearest` feature. When every
/// decoded coordinate is a half-integer from the bounds table (|v| <= 6),
/// the exact squared distances, scaled by 4, are integers, and
/// `nearest_exact` on both trees is checked against an exact `i128` brute
/// force, the entries ordered by `(d², id)` and cut after the k-th and its
/// exact ties.
#[cfg(feature = "exact-nearest")]
fn check_exact(x: f64, y: f64, k: usize, boxes: &[(Rect, u32)]) {
    let half = |v: f64| {
        let t = v * 2.0;
        (v.abs() <= 6.0 && t == t.trunc()).then_some(t as i128)
    };
    let Some((px, py)) = half(x).zip(half(y)) else {
        return;
    };
    let mut all: Vec<(i128, u32)> = Vec::with_capacity(boxes.len());
    for (b, id) in boxes {
        let (Some(x0), Some(y0), Some(x1), Some(y1)) =
            (half(b.min_x()), half(b.min_y()), half(b.max_x()), half(b.max_y()))
        else {
            return;
        };
        let axis = |p: i128, lo: i128, hi: i128| if p < lo { lo - p } else if p > hi { p - hi } else { 0 };
        let (dx, dy) = (axis(px, x0, x1), axis(py, y0, y1));
        all.push((dx * dx + dy * dy, *id));
    }
    all.sort_unstable();
    let keep = if k >= all.len() {
        all.len()
    } else {
        let kth = all[k - 1].0;
        k + all[k..].iter().take_while(|e| e.0 == kth).count()
    };
    let want: Vec<u32> = all[..keep].iter().map(|e| e.1).collect();
    let got_static: Vec<u32> = Index::bulk_load(boxes.to_vec())
        .nearest_exact(x, y, k)
        .expect("finite point")
        .iter()
        .map(|n| n.id)
        .collect();
    assert_eq!(got_static, want, "static nearest_exact differs from the exact brute force at k = {k}");
    let got_dynamic: Vec<u32> = situs::DynamicIndex::bulk_load(boxes.to_vec())
        .nearest_exact(x, y, k)
        .expect("finite point")
        .iter()
        .map(|n| n.id)
        .collect();
    assert_eq!(got_dynamic, want, "dynamic nearest_exact differs from the exact brute force at k = {k}");
}

/// The reference `(lower, upper)` for one box, from a one-entry index
/// queried for its single candidate.
fn reference_bounds(x: f64, y: f64, b: Rect) -> (f64, f64) {
    let one = Index::bulk_load(vec![(b, 0)]);
    let got = one.nearest_candidates(x, y, 1).expect("finite point");
    (got[0].lower_sq, got[0].upper_sq)
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 {
        return;
    }
    let (head, rest) = data.split_at(3);
    let point = rect(&[head[0], head[0], head[1], head[1]]);
    let (x, y) = (point.min_x(), point.min_y());
    let boxes: Vec<(Rect, u32)> = rest
        .chunks_exact(4)
        .take(MAX_BOXES)
        .enumerate()
        .map(|(i, b)| (rect(b), u32::try_from(i).expect("at most 512 boxes")))
        .collect();
    let index = Index::bulk_load(boxes.clone());
    let k = 1 + usize::from(head[2]) % (boxes.len().max(1) + 2);

    let all: Vec<(u32, f64, f64)> = boxes
        .iter()
        .map(|(b, id)| {
            let (l, u) = reference_bounds(x, y, *b);
            assert!(l.is_finite() && l >= 0.0, "lower {l:e}");
            assert!(!u.is_nan() && u >= l, "bounds inverted: {l:e} > {u:e}");
            (*id, l, u)
        })
        .collect();

    // Monotonicity at the root.
    if let Some(root) = index.bounds() {
        let (root_l, _) = reference_bounds(x, y, root);
        for (id, l, _) in &all {
            assert!(
                root_l <= *l,
                "root lower {root_l:e} above entry {id}'s {l:e}"
            );
        }
    }

    // Set identity.
    let mut uppers: Vec<f64> = all.iter().map(|e| e.2).collect();
    uppers.sort_by(f64::total_cmp);
    let tau = if k <= uppers.len() {
        uppers[k - 1]
    } else {
        f64::INFINITY
    };
    let mut want: Vec<(u32, f64, f64)> = all.iter().copied().filter(|e| e.1 <= tau).collect();
    want.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    let mut got: Vec<(u32, f64, f64)> = index
        .nearest_candidates(x, y, k)
        .expect("finite point")
        .into_iter()
        .map(|n| (n.id, n.lower_sq, n.upper_sq))
        .collect();
    got.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    assert_eq!(got, want, "candidates differ from brute force at k = {k}");

    #[cfg(feature = "exact-nearest")]
    check_exact(x, y, k, &boxes);

    // browse: complete and sorted, with the reference bounds.
    let yielded: Vec<(u32, f64, f64)> = index
        .browse(x, y)
        .expect("finite point")
        .map(|n| (n.id, n.lower_sq, n.upper_sq))
        .collect();
    assert_eq!(yielded.len(), boxes.len());
    for w in yielded.windows(2) {
        assert!(w[0].1 <= w[1].1, "browse went backward: {w:?}");
    }
    let mut seen = yielded.clone();
    seen.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    let mut expect: Vec<(u32, f64, f64)> = all;
    expect.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    assert_eq!(seen, expect, "browse multiset differs from the entries");
});
