// src/static_index/search/tests.rs

//! Tests of the query: the fixed stack, early exit and the containment
//! shortcut.
//!
//! The query must answer exactly as a plain search does, and in exactly the
//! same order. The reference, `reference_query`, is a search with a heap
//! stack and no shortcut. Every test compares visit **sequences** with it rather
//! than sets.

#![allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]

use super::super::tests::{brute, entries, lattice, pseudo_random_boxes, small_counts};
use super::*;
use crate::tests::{CASES, r, rect};
use proptest::prelude::*;
use std::format;
use std::vec;
use std::vec::Vec;

// ===========================================================================
// The reference and the sequences
// ===========================================================================

/// A reference query with a heap stack and no shortcut, which tests every
/// leaf entry.
fn reference_query(index: &Index, window: &Rect, mut visit: impl FnMut(u32)) {
    if index.nodes.is_empty() {
        return;
    }
    let mut stack: Vec<usize> = Vec::new();
    stack.push(0);
    while let Some(i) = stack.pop() {
        let node = &index.nodes[i];
        if !node.rect.intersects(window) {
            continue;
        }
        if node.leaf {
            for (rect, id) in &index.entries[node.first..node.end] {
                if rect.intersects(window) {
                    visit(*id);
                }
            }
        } else {
            stack.extend((node.first..node.end).rev());
        }
    }
}

fn reference_sequence(index: &Index, window: &Rect) -> Vec<u32> {
    let mut ids = Vec::new();
    reference_query(index, window, |id| ids.push(id));
    ids
}

fn sequence(index: &Index, window: &Rect) -> Vec<u32> {
    let mut ids = Vec::new();
    index.query(window, |id| ids.push(id));
    ids
}

/// The search with the shortcut switched off.
fn sequence_without_shortcut(index: &Index, window: &Rect) -> Vec<u32> {
    let mut ids = Vec::new();
    let mut stack = Stack::new(index.depth);
    let ControlFlow::Continue(()) =
        index.search::<Infallible, false>(window, &mut stack, &mut |id| {
            ids.push(id);
            ControlFlow::Continue(())
        });
    ids
}

/// The windows that exercise the shortcut, namely the whole tree, every
/// node's own box (contained in itself) and a point at each node's corner,
/// plus the given ones.
fn windows_for(index: &Index, extra: &[Rect]) -> Vec<Rect> {
    let mut ws: Vec<Rect> = extra.to_vec();
    ws.extend(index.bounds());
    for n in &index.nodes {
        ws.push(n.rect);
        ws.push(r(
            n.rect.min_x(),
            n.rect.min_y(),
            n.rect.min_x(),
            n.rect.min_y(),
        ));
    }
    ws
}

/// Every window's sequence equals the reference's, with and without the
/// shortcut, and sorted equals brute force.
fn check_order(index: &Index, source: &[(Rect, u32)], windows: &[Rect]) {
    for w in windows {
        let want = reference_sequence(index, w);
        assert_eq!(sequence(index, w), want, "window {w:?}");
        assert_eq!(
            sequence_without_shortcut(index, w),
            want,
            "window {w:?}, no shortcut"
        );
        let mut sorted = want;
        sorted.sort_unstable();
        assert_eq!(sorted, brute(source, w), "window {w:?}");
    }
}

/// The largest stack height reached by one search.
fn high_water<const SHORTCUT: bool>(index: &Index, window: &Rect) -> usize {
    let mut stack = Stack::new(index.depth);
    let ControlFlow::Continue(()) = index.search::<Infallible, SHORTCUT>(
        window,
        &mut stack,
        &mut |_| ControlFlow::Continue(()),
    );
    stack.high
}

// ===========================================================================
// Order identity
// ===========================================================================

#[test]
fn visit_order_equals_the_reference_on_every_small_count() {
    let fixed = [
        r(0.0, 0.0, 0.0, 0.0),
        r(7.0, 7.0, 7.0, 7.0),
        r(-1.0, 3.5, 100.0, 3.5),
        r(2.5, 2.5, 11.5, 9.5),
        r(-10.0, -10.0, 1000.0, 1000.0),
    ];
    for n in small_counts() {
        let side = super::super::ceil_sqrt(n).max(1) as u32;
        let mut source = lattice(side);
        source.truncate(n);
        let index = Index::bulk_load(source.clone());
        // Every node's box for the small trees; a stride of them above 300.
        let mut ws = windows_for(&index, &fixed);
        if n > 300 {
            let stride = ws.len() / 200 + 1;
            ws = ws.into_iter().step_by(stride).collect();
            ws.extend(fixed);
        }
        check_order(&index, &source, &ws);
    }
}

#[test]
fn visit_order_equals_the_reference_on_pseudo_random_boxes() {
    let source = pseudo_random_boxes(5_000);
    let index = Index::bulk_load(source.clone());
    let ws = windows_for(
        &index,
        &[r(-50.0, -50.0, 50.0, 50.0), r(-1e300, -1e300, 1e300, 1e300)],
    );
    check_order(&index, &source, &ws);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(CASES))]

    /// Random entries and windows, plus the whole tree, one node's box, and
    /// a point at an entry's corner.
    #[test]
    fn visit_order_equals_the_reference(
        es in entries(300),
        ws in prop::collection::vec(rect(), 1..=6),
        k in any::<usize>(),
    ) {
        let index = Index::bulk_load(es.clone());
        let mut windows = ws;
        windows.extend(index.bounds());
        if !index.nodes.is_empty() {
            windows.push(index.nodes[k % index.nodes.len()].rect);
            let (e, _) = es[k % es.len()];
            windows.push(r(e.max_x(), e.min_y(), e.max_x(), e.min_y()));
        }
        check_order(&index, &es, &windows);
    }

    /// Breaking after the `j`-th hit visits exactly the first `j` entries of
    /// `query`'s sequence and returns that break, for every `j`.
    #[test]
    fn try_query_visits_a_prefix_and_returns_its_break(
        es in entries(100),
        w in rect(),
        k in any::<usize>(),
    ) {
        let index = Index::bulk_load(es);
        let mut windows = vec![w];
        windows.extend(index.bounds());
        if !index.nodes.is_empty() {
            windows.push(index.nodes[k % index.nodes.len()].rect);
        }
        for w in &windows {
            let seq = sequence(&index, w);
            let mut seen = Vec::new();
            let all = index.try_query(w, |id| {
                seen.push(id);
                ControlFlow::<usize>::Continue(())
            });
            prop_assert_eq!(all, ControlFlow::Continue(()));
            prop_assert_eq!(&seen, &seq);
            for j in 1..=seq.len() {
                let mut seen = Vec::new();
                let got = index.try_query(w, |id| {
                    seen.push(id);
                    if seen.len() == j { ControlFlow::Break(j) } else { ControlFlow::Continue(()) }
                });
                prop_assert_eq!(got, ControlFlow::Break(j));
                prop_assert_eq!(&seen[..], &seq[..j]);
            }
        }
    }
}

#[test]
fn try_query_on_an_empty_index_continues() {
    let index = Index::bulk_load(Vec::new());
    let got = index.try_query(&r(0.0, 0.0, 1.0, 1.0), |_| ControlFlow::Break(()));
    assert_eq!(got, ControlFlow::Continue(()));
}

/// The doc example's question, "does any entry meet this box?", answered
/// with one visit.
#[test]
fn try_query_stops_at_the_first_hit() {
    let index = Index::bulk_load(lattice(64));
    let mut visits = 0;
    let hit = index
        .try_query(&r(-1.0, -1.0, 100.0, 100.0), |_| {
            visits += 1;
            ControlFlow::Break(())
        })
        .is_break();
    assert!(hit);
    assert_eq!(visits, 1);
    assert!(
        index
            .try_query(&r(70.0, 70.0, 71.0, 71.0), |_| ControlFlow::Break(()))
            .is_continue()
    );
}

// ===========================================================================
// The containment shortcut walks the subtree
// ===========================================================================

/// The entries below node `i`, in walk order, which form the true subtree.
fn subtree_entries(index: &Index, i: usize) -> Vec<u32> {
    let n = &index.nodes[i];
    if n.leaf {
        return index.entries[n.first..n.end].iter().map(|e| e.1).collect();
    }
    (n.first..n.end)
        .flat_map(|c| subtree_entries(index, c))
        .collect()
}

/// The entries from node `i`'s first leaf's first entry to its last leaf's
/// end, the contiguous range the shortcut must never take.
fn contiguous_range(index: &Index, i: usize) -> Option<Vec<u32>> {
    let (mut lo, mut hi) = (i, i);
    while !index.nodes[lo].leaf {
        lo = index.nodes[lo].first;
    }
    while !index.nodes[hi].leaf {
        hi = index.nodes[hi].end - 1;
    }
    let (first, end) = (index.nodes[lo].first, index.nodes[hi].end);
    (first < end).then(|| index.entries[first..end].iter().map(|e| e.1).collect())
}

/// STR reorders each level after cutting it, so an internal node's entries
/// are not one contiguous range. This test builds trees where that is so,
/// asserts it for the node it queries, and checks the query with that
/// node's own box as the window, where the shortcut emits the node's whole
/// subtree.
#[test]
fn a_contained_subtree_is_not_one_contiguous_entry_range() {
    let mut witnessed = 0;
    for source in [pseudo_random_boxes(5_000), lattice(40), lattice(70)] {
        let index = Index::bulk_load(source.clone());
        for i in 0..index.nodes.len() {
            if index.nodes[i].leaf {
                continue;
            }
            let mut own = subtree_entries(&index, i);
            own.sort_unstable();
            let mut range = contiguous_range(&index, i).unwrap_or_default();
            range.sort_unstable();
            if range != own {
                witnessed += 1;
                let w = index.nodes[i].rect;
                assert_eq!(
                    sequence(&index, &w),
                    reference_sequence(&index, &w),
                    "node {i}"
                );
                let mut got = sequence(&index, &w);
                got.sort_unstable();
                assert_eq!(got, brute(&source, &w), "node {i}");
            }
        }
    }
    // The trap must be present, or the test demonstrates nothing.
    assert!(witnessed >= 10, "only {witnessed} non-contiguous subtrees");
}

// ===========================================================================
// Depth and the stack bound
// ===========================================================================

/// `lattice` truncated to `n` entries.
fn lattice_of(n: usize) -> Vec<(Rect, u32)> {
    let side = super::super::ceil_sqrt(n).max(1) as u32;
    let mut e = lattice(side);
    e.truncate(n);
    e
}

const DEPTHS: [(usize, usize); 10] = [
    (0, 0),
    (1, 1),
    (16, 1),
    (17, 2),
    (256, 2),
    (257, 3),
    (4096, 3),
    (4097, 4),
    (65_536, 4),
    (65_537, 5),
];

#[test]
fn depth_is_recorded_for_known_counts() {
    for (n, depth) in DEPTHS {
        let index = Index::bulk_load(lattice_of(n));
        assert_eq!(index.depth, depth, "n = {n}");
        assert!(index.depth <= MAX_DEPTH);
    }
    #[cfg(target_pointer_width = "64")]
    assert_eq!((MAX_DEPTH, STACK_CAPACITY), (17, 256));
    assert_eq!(STACK_CAPACITY, (Index::NODE_CAPACITY - 1) * MAX_DEPTH + 1);
}

/// The high-water mark never exceeds `(M − 1) · depth + 1`, on every tree
/// of `DEPTHS` and the pseudo-random set, with and without the shortcut. On
/// a full tree the whole-tree search without the shortcut reaches exactly
/// `(M − 1) · (depth − 1) + 1`, the count the crate doc derives.
#[test]
fn stack_height_stays_within_the_bound() {
    let m = Index::NODE_CAPACITY;
    let mut trees: Vec<Vec<(Rect, u32)>> = DEPTHS.iter().map(|&(n, _)| lattice_of(n)).collect();
    trees.push(pseudo_random_boxes(5_000));
    let mut deepest = 0;
    for source in trees {
        let index = Index::bulk_load(source);
        let bound = (m - 1) * index.depth + 1;
        let mut ws = windows_for(&index, &[r(0.5, 0.5, 0.5, 0.5), r(3.25, 3.25, 40.75, 9.5)]);
        if ws.len() > 400 {
            let stride = ws.len() / 400 + 1;
            ws = ws.into_iter().step_by(stride).collect();
            ws.extend(index.bounds());
        }
        for w in &ws {
            assert!(
                high_water::<true>(&index, w) <= bound,
                "depth {}",
                index.depth
            );
            assert!(
                high_water::<false>(&index, w) <= bound,
                "depth {}",
                index.depth
            );
        }
        deepest = deepest.max(index.depth);
    }
    assert!(deepest >= 5, "no tree of 65,537 entries was built");

    // The exact peak on full trees: 16², 16³, 16⁴ entries.
    for n in [256usize, 4096, 65_536] {
        let index = Index::bulk_load(lattice_of(n));
        let all = index.bounds().unwrap();
        assert_eq!(
            high_water::<false>(&index, &all),
            (m - 1) * (index.depth - 1) + 1,
            "n = {n}"
        );
    }
}

#[test]
#[should_panic(expected = "search stack")]
fn the_stack_panics_rather_than_drop_a_node() {
    let mut stack = Stack::new(MAX_DEPTH);
    for i in 0..=STACK_CAPACITY {
        stack.push(i);
    }
}

#[test]
fn the_stack_is_last_in_first_out() {
    let mut stack = Stack::new(2);
    assert_eq!(stack.pop(), None);
    for i in 0..STACK_CAPACITY.min(31) {
        stack.push(i);
    }
    assert_eq!(stack.pop(), Some(30));
    assert_eq!(stack.pop(), Some(29));
    assert_eq!(stack.len(), 29);
}

// ===========================================================================
// The timing ratio
// ===========================================================================

/// Ceilings for `new ÷ reference` per window size, and for the shortcut's
/// effect on 10% windows (`with ÷ without`), each set from a measured run
/// with a margin.
const CEILING_POINT: f64 = 1.0;
const CEILING_SMALL: f64 = 1.0;
const CEILING_TENTH: f64 = 0.75;
const CEILING_SHORTCUT_TENTH: f64 = 0.75;

/// `n` windows of side `side` (0 for points) on the 317 × 317 lattice, from
/// a fixed SplitMix64 stream, offset off the grid so that a window of side
/// `k + 0.8` meets `(k + 1)²` squares.
fn ratio_windows(n: usize, side: f64, seed: u64) -> Vec<Rect> {
    let mut state = seed;
    let mut next = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let span = 317.0 - side.ceil() - 1.0;
    (0..n)
        .map(|_| {
            let x = (next() % 1_000_000) as f64 / 1_000_000.0 * span;
            let y = (next() % 1_000_000) as f64 / 1_000_000.0 * span;
            let (x, y) = (x.floor() + 0.1, y.floor() + 0.1);
            r(x, y, x + side, y + side)
        })
        .collect()
}

/// The best of `rounds` timings of one pass over `windows`, and the number
/// of hits of the last pass.
fn time_pass(
    windows: &[Rect],
    rounds: usize,
    mut query: impl FnMut(&Rect, &mut u64),
) -> (f64, u64) {
    let mut best = f64::INFINITY;
    let mut hits = 0;
    for _ in 0..rounds {
        let mut h = 0u64;
        let t = std::time::Instant::now();
        for w in windows {
            query(w, &mut h);
        }
        best = best.min(t.elapsed().as_secs_f64());
        hits = core::hint::black_box(h);
    }
    (best, hits)
}

/// The timing ratio of the query against the reference, on 100,000 lattice
/// boxes and 10,000 windows of each of three sizes, with the shortcut's
/// effect on the largest. It runs in release only and prints the ratios.
#[test]
#[ignore = "timing test; run in release with --ignored query_ratio"]
fn query_ratio() {
    let index = Index::bulk_load(lattice_of(100_000));
    let sizes = [
        ("point", 0.0, CEILING_POINT),
        ("~10 entries", 2.8, CEILING_SMALL),
        ("~10%", 99.8, CEILING_TENTH),
    ];
    let rounds = 7;
    let mut failed = Vec::new();
    for (k, (name, side, ceiling)) in sizes.into_iter().enumerate() {
        let windows = ratio_windows(10_000, side, 0x5EED + k as u64);
        let (t_ref, h_ref) = time_pass(&windows, rounds, |w, h| {
            reference_query(&index, w, |id| *h += u64::from(id) + 1);
        });
        let (t_new, h_new) = time_pass(&windows, rounds, |w, h| {
            index.query(w, |id| *h += u64::from(id) + 1);
        });
        let (t_off, h_off) = time_pass(&windows, rounds, |w, h| {
            let mut stack = Stack::new(index.depth);
            let ControlFlow::Continue(()) =
                index.search::<Infallible, false>(w, &mut stack, &mut |id| {
                    *h += u64::from(id) + 1;
                    ControlFlow::Continue(())
                });
        });
        assert_eq!(
            (h_new, h_off),
            (h_ref, h_ref),
            "{name}: the variants disagree"
        );
        let ratio = t_new / t_ref;
        let shortcut = t_new / t_off;
        std::println!(
            "query_ratio {name:>12}: reference {:.4} s, new {:.4} s, new without shortcut {:.4} s; new/reference {ratio:.3} (ceiling {ceiling}), shortcut with/without {shortcut:.3}",
            t_ref,
            t_new,
            t_off
        );
        if ratio > ceiling {
            failed.push(std::format!("{name}: new/reference {ratio:.3} > {ceiling}"));
        }
        if k == 2 && shortcut > CEILING_SHORTCUT_TENTH {
            failed.push(std::format!(
                "{name}: shortcut with/without {shortcut:.3} > {CEILING_SHORTCUT_TENTH}"
            ));
        }
    }
    assert!(failed.is_empty(), "{failed:?}");
}
