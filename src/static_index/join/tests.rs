// src/static_index/join/tests.rs

//! Tests of `join` and `self_join`.
//!
//! A join that loses a pair gives no sign of the loss, so every test here is
//! an equality with brute force over all entry pairs, as a sorted multiset. Two identities tie the operations together in turn:
//! `join(A, B)` swapped is `join(B, A)`, and `join(A, A)` is `self_join(A)`
//! both ways plus every entry paired with itself.

#![allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]

use super::super::tests::{entries, lattice, pseudo_random_boxes};
use super::*;
use crate::tests::{CASES, r};
use proptest::prelude::*;
use std::format;
use std::vec::Vec;

// ===========================================================================
// Brute force and the identities
// ===========================================================================

type Pairs = Vec<(u32, u32)>;

/// Every `(a, b)` of `a_entries × b_entries` whose boxes meet, sorted.
fn brute_join(a_entries: &[(Rect, u32)], b_entries: &[(Rect, u32)]) -> Pairs {
    let mut out = Vec::new();
    for (ra, ia) in a_entries {
        for (rb, ib) in b_entries {
            if ra.intersects(rb) {
                out.push((*ia, *ib));
            }
        }
    }
    out.sort_unstable();
    out
}

/// Every unordered pair of distinct stored entries (`i < j` by input
/// position) whose boxes meet, each pair's ids sorted, then the list sorted.
fn brute_self(es: &[(Rect, u32)]) -> Pairs {
    let mut out = Vec::new();
    for (i, (ri, ii)) in es.iter().enumerate() {
        for (rj, ij) in &es[i + 1..] {
            if ri.intersects(rj) {
                out.push((*ii.min(ij), *ii.max(ij)));
            }
        }
    }
    out.sort_unstable();
    out
}

fn joined(a: &Index, b: &Index) -> Pairs {
    let mut out = Vec::new();
    a.join(b, |x, y| out.push((x, y)));
    out.sort_unstable();
    out
}

fn self_joined(a: &Index) -> Pairs {
    let mut out = Vec::new();
    a.self_join(|x, y| out.push((x.min(y), x.max(y))));
    out.sort_unstable();
    out
}

/// Every equality and identity check on one pair of entry sets. Both joins
/// equal brute force, `join` is symmetric, and `join(A, A)` is
/// `self_join(A)` both ways plus `(id, id)` for every stored entry.
fn check_join(a_entries: &[(Rect, u32)], b_entries: &[(Rect, u32)]) {
    let a = Index::bulk_load(a_entries.to_vec());
    let b = Index::bulk_load(b_entries.to_vec());
    let ab = joined(&a, &b);
    assert_eq!(ab, brute_join(a_entries, b_entries), "join(A, B)");
    let mut ba: Pairs = joined(&b, &a).into_iter().map(|(x, y)| (y, x)).collect();
    ba.sort_unstable();
    assert_eq!(ba, ab, "join(B, A) swapped differs from join(A, B)");
    for (index, source) in [(&a, a_entries), (&b, b_entries)] {
        let sj = self_joined(index);
        assert_eq!(sj, brute_self(source), "self_join");
        let mut both: Pairs = Vec::new();
        index.self_join(|x, y| both.extend([(x, y), (y, x)]));
        both.extend(source.iter().map(|(_, id)| (*id, *id)));
        both.sort_unstable();
        assert_eq!(
            joined(index, index),
            both,
            "join(A, A) is not self_join(A) both ways plus the diagonal"
        );
    }
}

/// `lattice` truncated to `n` entries.
fn lattice_of(n: usize) -> Vec<(Rect, u32)> {
    let side = super::super::ceil_sqrt(n).max(1) as u32;
    let mut e = lattice(side);
    e.truncate(n);
    e
}

/// The same entries with every id offset, so that the two sides of a join
/// are told apart.
fn offset_ids(es: &[(Rect, u32)], by: u32) -> Vec<(Rect, u32)> {
    es.iter().map(|(r, id)| (*r, id + by)).collect()
}

/// Every box shifted by `(dx, dy)`.
fn shifted(es: &[(Rect, u32)], dx: f64, dy: f64) -> Vec<(Rect, u32)> {
    es.iter()
        .map(|(b, id)| {
            (
                r(
                    b.min_x() + dx,
                    b.min_y() + dy,
                    b.max_x() + dx,
                    b.max_y() + dy,
                ),
                *id,
            )
        })
        .collect()
}

// ===========================================================================
// Fixed cases
// ===========================================================================

#[test]
fn join_with_an_empty_side_reports_nothing() {
    let some = lattice_of(20);
    check_join(&[], &some);
    check_join(&some, &[]);
    check_join(&[], &[]);
    let empty = Index::bulk_load(Vec::new());
    let mut n = 0;
    empty.self_join(|_, _| n += 1);
    empty.join(&Index::bulk_load(some), |_, _| n += 1);
    assert_eq!(n, 0);
}

#[test]
fn join_matches_brute_force_at_the_required_sizes() {
    let m = Index::NODE_CAPACITY;
    for n in [1, m, m + 1, m * m] {
        let a = lattice_of(n);
        check_join(&a, &offset_ids(&shifted(&a, 0.5, 0.25), 100_000));
        check_join(&a, &offset_ids(&lattice_of(n.max(2) - 1), 100_000));
    }
}

/// One entry against a tree of four levels, `M` against `M² + 1`, and
/// `M + 1` against `M³ + 1`, where the shallow side's leaf meets the deep
/// side's subtrees, the one-leaf case.
#[test]
fn join_across_trees_of_different_heights() {
    let m = Index::NODE_CAPACITY;
    let deep = offset_ids(&lattice_of(m * m * m + 1), 100_000);
    assert_eq!(Index::bulk_load(deep.clone()).depth, 4);
    check_join(&[(r(10.5, 10.5, 12.0, 12.0), 7)], &deep);
    check_join(&deep, &[(r(10.5, 10.5, 12.0, 12.0), 7)]);
    check_join(&[(r(-5.0, -5.0, 100.0, 100.0), 7)], &deep);
    check_join(&lattice_of(m), &offset_ids(&lattice_of(m * m + 1), 100_000));
    check_join(&shifted(&lattice_of(m + 1), 3.5, 2.0), &deep);
}

#[test]
fn join_reports_every_pair_of_identical_boxes() {
    let a: Vec<(Rect, u32)> = (0..100).map(|i| (r(1.0, 1.0, 2.0, 2.0), i)).collect();
    let b: Vec<(Rect, u32)> = (0..40).map(|i| (r(2.0, 2.0, 2.0, 2.0), 1000 + i)).collect();
    check_join(&a, &b);
    assert_eq!(
        joined(&Index::bulk_load(a.clone()), &Index::bulk_load(b)).len(),
        4000
    );
    assert_eq!(self_joined(&Index::bulk_load(a)).len(), 100 * 99 / 2);
}

/// Unit squares against unit squares shifted by a whole unit. Every contact
/// is along an edge or at a corner, so only the closed test finds them.
#[test]
fn join_finds_boxes_touching_at_an_edge_and_at_a_corner() {
    let a = lattice_of(400);
    let edge = offset_ids(&shifted(&a, 1.0, 0.0), 100_000);
    let corner = offset_ids(&shifted(&a, 1.0, 1.0), 100_000);
    check_join(&a, &edge);
    check_join(&a, &corner);
    // Two squares meeting at a single corner point.
    let one = [(r(0.0, 0.0, 1.0, 1.0), 1)];
    let other = [(r(1.0, 1.0, 2.0, 2.0), 2)];
    assert_eq!(
        joined(
            &Index::bulk_load(one.to_vec()),
            &Index::bulk_load(other.to_vec())
        ),
        [(1, 2)]
    );
}

#[test]
fn join_handles_extreme_magnitudes() {
    let big = 1e300;
    let tiny = f64::MIN_POSITIVE;
    let a = vec_of(&[
        (r(-big, -big, -big, -big), 0),
        (r(big, big, big, big), 1),
        (r(-big, -big, big, big), 2),
        (r(0.0, 0.0, tiny, tiny), 3),
        (r(-tiny, -tiny, 0.0, 0.0), 4),
        (r(tiny, tiny, 2.0 * tiny, 2.0 * tiny), 5),
        (r(-big, 0.0, big, 0.0), 6),
    ]);
    let b = vec_of(&[
        (r(-big, -big, -big, -big), 10),
        (r(big, big, big, big), 11),
        (r(0.0, 0.0, 0.0, 0.0), 12),
        (r(tiny, tiny, tiny, tiny), 13),
        (r(-tiny, -tiny, tiny, tiny), 14),
        (r(2.0 * tiny, 0.0, 1.0, 1.0), 15),
        (r(-big, -big, big, big), 16),
    ]);
    check_join(&a, &b);
    check_join(&a, &a);
}

fn vec_of(es: &[(Rect, u32)]) -> Vec<(Rect, u32)> {
    es.to_vec()
}

#[test]
fn join_matches_brute_force_on_pseudo_random_boxes() {
    let all = pseudo_random_boxes(3_000);
    let (a, b) = all.split_at(1_800);
    check_join(a, b);
}

// ===========================================================================
// Order
// ===========================================================================

/// The pairs of a fixed case, in visit order, recorded from this code, so
/// that a change of order is a visible decision rather than an accident.
#[test]
fn visit_order_of_a_fixed_case_is_recorded() {
    let a = Index::bulk_load(lattice_of(40));
    let b = Index::bulk_load(offset_ids(&shifted(&lattice_of(20), 2.5, 1.5), 100));
    let mut seq: Pairs = Vec::new();
    a.join(&b, |x, y| seq.push((x, y)));
    assert_eq!(seq.len(), JOIN_ORDER.len());
    assert_eq!(seq, JOIN_ORDER);
    let mut seq: Pairs = Vec::new();
    a.self_join(|x, y| seq.push((x, y)));
    assert_eq!(seq, SELF_JOIN_ORDER);
}

/// Within a `self_join` pair, the first entry is earlier in the index's
/// depth-first order, the order of a query that covers every entry.
#[test]
fn self_join_puts_the_earlier_entry_first() {
    for source in [pseudo_random_boxes(2_000), lattice_of(700)] {
        let index = Index::bulk_load(source);
        let all = index.bounds().unwrap();
        let mut position = std::collections::HashMap::new();
        let mut k = 0usize;
        index.query(&all, |id| {
            position.insert(id, k);
            k += 1;
        });
        let mut n = 0;
        index.self_join(|x, y| {
            assert!(position[&x] < position[&y], "({x}, {y}) out of order");
            n += 1;
        });
        assert!(n > 0);
    }
}

// ===========================================================================
// Property tests
// ===========================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(CASES))]

    /// Up to 500 entries a side, from the shared `rect()` strategy, with ids
    /// offset so that the sides are told apart.
    #[test]
    fn join_equals_brute_force(a in entries(500), b in entries(500)) {
        check_join(&a, &offset_ids(&b, 1_000_000));
    }

    /// One side a subset of the other, with shared ids, so that equal boxes
    /// and equal ids occur on both sides.
    #[test]
    fn join_of_overlapping_sets_equals_brute_force(a in entries(300), k in 0usize..300) {
        let b: Vec<(Rect, u32)> = a.iter().copied().skip(k % (a.len() + 1)).collect();
        check_join(&a, &b);
    }
}

// ===========================================================================
// The timing ratio
// ===========================================================================

/// Ceilings for `join ÷ one query per entry of B`, per data set, each set
/// from measured runs with a margin.
const CEILING_JOIN_LATTICE: f64 = 0.35;
const CEILING_JOIN_RANDOM: f64 = 0.6;

/// `n` boxes from a fixed SplitMix64 stream in a square of side `extent`,
/// each up to `size` on a side.
fn random_boxes(n: u32, extent: f64, size: f64, seed: u64, id0: u32) -> Vec<(Rect, u32)> {
    let mut state = seed;
    let mut next = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (z ^ (z >> 31)) as f64 / u64::MAX as f64
    };
    (0..n)
        .map(|i| {
            let (x, y) = (next() * extent, next() * extent);
            let (w, h) = (next() * size, next() * size);
            (r(x, y, x + w, y + h), id0 + i)
        })
        .collect()
}

fn best_of(rounds: usize, mut f: impl FnMut() -> u64) -> (f64, u64) {
    let mut best = f64::INFINITY;
    let mut out = 0;
    for _ in 0..rounds {
        let t = std::time::Instant::now();
        out = core::hint::black_box(f());
        best = best.min(t.elapsed().as_secs_f64());
    }
    (best, out)
}

/// The timing ratio of `join` against querying `A` once per entry of `B`,
/// on 50,000 boxes against 50,000 boxes, lattice and random. It runs in
/// release only and prints the ratios.
#[test]
#[ignore = "timing test; run in release with --ignored join_ratio"]
fn join_ratio() {
    let cases = [
        (
            "lattice",
            lattice_of(50_000),
            offset_ids(&shifted(&lattice_of(50_000), 0.5, 0.5), 1_000_000),
            CEILING_JOIN_LATTICE,
        ),
        (
            "random",
            random_boxes(50_000, 1_000.0, 6.0, 0xA11CE, 0),
            random_boxes(50_000, 1_000.0, 6.0, 0xB0B, 1_000_000),
            CEILING_JOIN_RANDOM,
        ),
    ];
    let mut failed = Vec::new();
    for (name, a_entries, b_entries, ceiling) in cases {
        let a = Index::bulk_load(a_entries);
        let b_index = Index::bulk_load(b_entries.clone());
        let mut n_pairs = 0usize;
        a.join(&b_index, |_, _| n_pairs += 1);
        let (t_join, n_join) = best_of(5, || {
            let mut n = 0u64;
            a.join(&b_index, |x, y| n += u64::from(x ^ y) | 1);
            n
        });
        let (t_query, n_query) = best_of(5, || {
            let mut n = 0u64;
            for (rb, ib) in &b_entries {
                a.query(rb, |x| n += u64::from(x ^ ib) | 1);
            }
            n
        });
        assert_eq!(
            n_join, n_query,
            "{name}: join and the per-entry queries disagree"
        );
        let ratio = t_join / t_query;
        std::println!(
            "join_ratio {name:>8}: join {t_join:.4} s, per-entry queries {t_query:.4} s, {n_pairs} pairs; join/queries {ratio:.3} (ceiling {ceiling})"
        );
        if ratio > ceiling {
            failed.push(format!("{name}: {ratio:.3} > {ceiling}"));
        }
    }
    assert!(failed.is_empty(), "{failed:?}");
}

/// `join(lattice of 40, lattice of 20 shifted by (2.5, 1.5), ids + 100)` in
/// visit order, recorded from this code.
const JOIN_ORDER: [(u32, u32); 68] = [
    (9, 100),
    (10, 100),
    (10, 101),
    (11, 101),
    (11, 102),
    (12, 102),
    (12, 103),
    (16, 100),
    (16, 105),
    (17, 100),
    (17, 101),
    (17, 105),
    (17, 106),
    (13, 103),
    (13, 104),
    (19, 102),
    (19, 103),
    (19, 107),
    (19, 108),
    (20, 103),
    (20, 104),
    (20, 108),
    (20, 109),
    (26, 107),
    (26, 108),
    (26, 112),
    (26, 113),
    (27, 108),
    (27, 109),
    (27, 113),
    (27, 114),
    (33, 112),
    (33, 113),
    (34, 113),
    (34, 114),
    (33, 117),
    (33, 118),
    (34, 118),
    (34, 119),
    (18, 101),
    (18, 102),
    (18, 106),
    (18, 107),
    (23, 105),
    (23, 110),
    (24, 105),
    (24, 106),
    (24, 110),
    (24, 111),
    (25, 106),
    (25, 107),
    (25, 111),
    (25, 112),
    (30, 110),
    (30, 115),
    (31, 110),
    (31, 111),
    (31, 115),
    (32, 111),
    (32, 112),
    (37, 115),
    (38, 115),
    (31, 116),
    (32, 116),
    (32, 117),
    (38, 116),
    (39, 116),
    (39, 117),
];

/// `self_join(lattice of 40)` in visit order, recorded from this code.
const SELF_JOIN_ORDER: [(u32, u32); 124] = [
    (0, 1),
    (0, 7),
    (0, 8),
    (1, 2),
    (1, 7),
    (1, 8),
    (1, 9),
    (2, 3),
    (2, 8),
    (2, 9),
    (2, 10),
    (3, 4),
    (3, 9),
    (3, 10),
    (3, 11),
    (4, 5),
    (4, 10),
    (4, 11),
    (4, 12),
    (5, 11),
    (5, 12),
    (7, 8),
    (7, 14),
    (7, 15),
    (8, 9),
    (8, 14),
    (8, 15),
    (8, 16),
    (9, 10),
    (9, 15),
    (9, 16),
    (9, 17),
    (10, 11),
    (10, 16),
    (10, 17),
    (11, 12),
    (11, 17),
    (14, 15),
    (15, 16),
    (16, 17),
    (5, 6),
    (5, 13),
    (11, 19),
    (12, 6),
    (12, 13),
    (12, 19),
    (12, 20),
    (10, 18),
    (11, 18),
    (12, 18),
    (14, 21),
    (14, 22),
    (15, 21),
    (15, 22),
    (15, 23),
    (16, 22),
    (16, 23),
    (16, 24),
    (17, 18),
    (17, 23),
    (17, 24),
    (17, 25),
    (6, 13),
    (13, 19),
    (13, 20),
    (19, 20),
    (19, 26),
    (19, 27),
    (20, 26),
    (20, 27),
    (26, 27),
    (26, 33),
    (26, 34),
    (27, 33),
    (27, 34),
    (33, 34),
    (19, 18),
    (19, 25),
    (26, 18),
    (26, 25),
    (26, 32),
    (33, 25),
    (33, 32),
    (33, 39),
    (18, 24),
    (18, 25),
    (21, 22),
    (21, 28),
    (21, 29),
    (22, 23),
    (22, 28),
    (22, 29),
    (22, 30),
    (23, 24),
    (23, 29),
    (23, 30),
    (23, 31),
    (24, 25),
    (24, 30),
    (24, 31),
    (24, 32),
    (25, 31),
    (25, 32),
    (28, 29),
    (28, 35),
    (28, 36),
    (29, 30),
    (29, 35),
    (29, 36),
    (29, 37),
    (30, 31),
    (30, 36),
    (30, 37),
    (30, 38),
    (31, 32),
    (31, 37),
    (31, 38),
    (31, 39),
    (32, 38),
    (32, 39),
    (35, 36),
    (36, 37),
    (37, 38),
    (38, 39),
];
