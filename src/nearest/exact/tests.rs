// src/nearest/exact/tests.rs

//! Tests of the exact ranking (`exact-nearest`).
//!
//! Every test runs on the three trees of the nearest-neighbor tests (the
//! static index, a bulk-loaded dynamic tree and a churned dynamic tree), and
//! the oracle reads each tree's own entries and positions, so the exact
//! order, ties included, is checked on every tree.
//!
//! - **Integer coordinates** up to `2^26`, with exact squared distances in
//!   `u128`, so the oracle is plain integer arithmetic.
//! - **The full range**. `write_exact_csv` writes cases spanning subnormal
//!   distances, overflow-scale coordinates and mixed magnitudes, for an
//!   independent check that recomputes every distance in exact rational
//!   arithmetic.

#![allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]

use super::super::bounds::bounds_sq;
use super::super::bounds::tests::{Rng, libm_exp2};
use super::super::tests::{Searched, churned, exact_sq_int, integer_entries, trees};
use crate::Rect;
use crate::dynamic::DynamicIndex;
use crate::nearest::{Neighbor, QueryError};
use crate::static_index::Index;
use crate::tests::{CASES, r};
use proptest::prelude::*;
use std::format;
use std::string::{String, ToString};
use std::vec;
use std::vec::Vec;

// ===========================================================================
// The integer oracle
// ===========================================================================

/// The exact answer for an integer query point. Every stored entry is ranked
/// by `(exact d², id, position)` and cut after the k-th and its exact ties,
/// with each entry's certified bounds.
fn brute_exact(index: &dyn Searched, x: i64, y: i64, k: usize) -> Vec<Neighbor> {
    let mut all: Vec<(u128, u32, usize, Rect)> = index
        .stored()
        .into_iter()
        .map(|(rect, id, pos)| (exact_sq_int(x, y, &rect), id, pos, rect))
        .collect();
    all.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    let keep = if k == 0 {
        0
    } else if k >= all.len() {
        all.len()
    } else {
        let kth = all[k - 1].0;
        k + all[k..].iter().take_while(|e| e.0 == kth).count()
    };
    all.truncate(keep);
    all.into_iter()
        .map(|(_, id, _, rect)| {
            let (lower_sq, upper_sq) = bounds_sq(x as f64, y as f64, &rect);
            Neighbor {
                id,
                lower_sq,
                upper_sq,
            }
        })
        .collect()
}

/// The fixed list of k for one tree: 0, 1, 2, the node capacity, and the
/// entry count less one, exactly, and plus five.
fn ks(n: usize) -> [usize; 7] {
    let m = Index::NODE_CAPACITY;
    [0, 1, 2, m, n.saturating_sub(1), n, n + 5]
}

/// Integer entries in one of three ranges: a tiny one (many exact ties), a
/// moderate one, and the full `2^26`.
fn integer_case() -> impl Strategy<Value = (Vec<(Rect, u32)>, i64, i64)> {
    (any::<u64>(), 1_usize..400, 0_usize..3).prop_map(|(seed, n, range)| {
        let bound = [8_i64, 1 << 12, 1 << 26][range];
        let mut rng = Rng::new(seed);
        let entries = integer_entries(&mut rng, n, bound);
        let (x, y) = (rng.int(bound), rng.int(bound));
        (entries, x, y)
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(CASES))]

    /// Integer coordinates up to 2^26. `nearest_exact` equals the exact
    /// brute force, tie extension and tie order included, on all three
    /// trees and for every k in the list.
    #[test]
    fn exact_equals_brute_force_on_integer_coordinates((es, x, y) in integer_case()) {
        for index in trees(&es) {
            for k in ks(es.len()) {
                let got = index.exact(x as f64, y as f64, k).unwrap();
                let want = brute_exact(&*index, x, y, k);
                prop_assert_eq!(got, want, "{} k = {}", index.which(), k);
            }
        }
    }

    /// Agreement with the candidates by certified bounds, and across trees.
    /// The exact result is a prefix-closed subset of the candidates (as a
    /// multiset), equal to them when no candidate lies beyond the k-th
    /// exactly, and the three trees return the same sequence (ids are
    /// unique here).
    #[test]
    fn exact_agrees_with_the_candidates_and_across_trees((es, x, y) in integer_case(), k in 0_usize..24) {
        let (px, py) = (x as f64, y as f64);
        let mut sequences: Vec<Vec<Neighbor>> = Vec::new();
        for index in trees(&es) {
            let exact = index.exact(px, py, k).unwrap();
            let cands = index.cands(px, py, k).unwrap();
            let key = |n: &Neighbor| (n.lower_sq.to_bits(), n.upper_sq.to_bits(), n.id);
            let mut have: Vec<_> = cands.iter().map(key).collect();
            have.sort_unstable();
            let mut sub: Vec<_> = exact.iter().map(key).collect();
            sub.sort_unstable();
            // Every exact entry is a candidate.
            let mut rest = have.clone();
            for e in &sub {
                let at = rest.iter().position(|c| c == e);
                prop_assert!(at.is_some(), "an exact entry is not a candidate");
                rest.remove(at.unwrap());
            }
            // Equal when no candidate lies beyond the k-th exactly.
            let d2 = |n: &Neighbor| {
                es.iter()
                    .filter(|(_, id)| *id == n.id)
                    .map(|(b, _)| exact_sq_int(x, y, b))
                    .next()
                    .unwrap()
            };
            if k > 0 && !exact.is_empty() {
                let kth = d2(&exact[exact.len().min(k) - 1]);
                let beyond = cands.iter().any(|c| d2(c) > kth);
                if !beyond {
                    prop_assert_eq!(&sub, &have);
                }
            }
            sequences.push(exact);
        }
        prop_assert_eq!(&sequences[0], &sequences[1]);
        prop_assert_eq!(&sequences[0], &sequences[2]);
    }
}

// ===========================================================================
// Fixed cases
// ===========================================================================

/// The bounds tie two entries whose exact squared distances are `2^54` and
/// `2^54 + 1`. The nearer one has the larger id, so an order by the lower
/// bound and then the id puts the wrong entry first. The exact ranking
/// returns the true nearest alone for k = 1.
#[test]
fn the_exact_order_overrules_a_lower_bound_tie_broken_by_id() {
    let p27 = libm_exp2(27);
    let near = r(p27, 0.0, p27, 0.0); // d² = 2⁵⁴
    let far = r(p27, 1.0, p27, 1.0); //  d² = 2⁵⁴ + 1, rounds to 2⁵⁴
    for index in trees(&[(far, 1), (near, 2)]) {
        let cands = index.cands(0.0, 0.0, 1).unwrap();
        assert_eq!(cands.iter().map(|n| n.id).collect::<Vec<_>>(), [1, 2]);
        assert_eq!(cands[0].lower_sq, cands[1].lower_sq, "the bounds tie");
        let one = index.exact(0.0, 0.0, 1).unwrap();
        assert_eq!(
            one.iter().map(|n| n.id).collect::<Vec<_>>(),
            [2],
            "{}",
            index.which()
        );
        let both = index.exact(0.0, 0.0, 2).unwrap();
        assert_eq!(both.iter().map(|n| n.id).collect::<Vec<_>>(), [2, 1]);
    }
}

/// Distances one unit in the last place of `d²` apart, at several scales.
/// The candidates by certified bounds include both, and the exact ranking
/// returns exactly the nearer one.
#[test]
fn near_ties_one_ulp_apart_are_resolved() {
    for e in [-500_i32, -30, 0, 26, 400] {
        let s = libm_exp2(e);
        // d² = 2s² and 2s² + s²·2⁻⁵¹·2, with the second box moved by a
        // relative 2⁻⁵² on one axis.
        let a = r(s, s, s, s);
        let b = r(s * (1.0 + f64::EPSILON), s, s * (1.0 + f64::EPSILON), s);
        for index in trees(&[(b, 1), (a, 2)]) {
            let one = index.exact(0.0, 0.0, 1).unwrap();
            assert_eq!(one.iter().map(|n| n.id).collect::<Vec<_>>(), [2], "e = {e}");
            assert!(!index.cands(0.0, 0.0, 1).unwrap().is_empty());
        }
    }
}

/// Exact ties, a point equidistant from several boxes and a point inside
/// several. All are returned, ordered by id.
#[test]
fn exact_ties_are_all_returned_in_id_order() {
    // Four unit boxes whose nearest corners are 5 from the origin, along
    // 3-4-5 triangles, and one farther box.
    let ring = [
        (r(3.0, 4.0, 4.0, 5.0), 9),
        (r(-5.0, -4.0, -4.0, -3.0), 3),
        (r(4.0, -4.0, 5.0, -3.0), 7),
        (r(-5.0, 3.0, -4.0, 4.0), 1),
        (r(40.0, 40.0, 41.0, 41.0), 0),
    ];
    for index in trees(&ring) {
        for k in 1..=4 {
            let got = index.exact(0.0, 0.0, k).unwrap();
            assert_eq!(
                got.iter().map(|n| n.id).collect::<Vec<_>>(),
                [1, 3, 7, 9],
                "k = {k}"
            );
        }
        let all = index.exact(0.0, 0.0, 5).unwrap();
        assert_eq!(
            all.iter().map(|n| n.id).collect::<Vec<_>>(),
            [1, 3, 7, 9, 0]
        );
    }
    // Inside several boxes, at distance zero from each.
    let nested: Vec<(Rect, u32)> = (0..12)
        .map(|i| {
            (
                r(-f64::from(i) - 1.0, -1.0, f64::from(i) + 1.0, 1.0),
                20 - i,
            )
        })
        .collect();
    for index in trees(&nested) {
        let got = index.exact(0.0, 0.0, 1).unwrap();
        let ids: Vec<u32> = got.iter().map(|n| n.id).collect();
        assert_eq!(ids, (9..=20).collect::<Vec<u32>>());
    }
}

/// `k = 0`, `k >= len`, an empty tree, and a non-finite query.
#[test]
fn the_edges_of_k_and_the_refusals() {
    let es = [
        (r(0.0, 0.0, 1.0, 1.0), 4),
        (r(5.0, 5.0, 6.0, 6.0), 2),
        (r(2.0, 2.0, 3.0, 3.0), 3),
    ];
    for index in trees(&es) {
        assert_eq!(index.exact(0.0, 0.0, 0).unwrap(), []);
        let all = index.exact(0.0, 0.0, 10).unwrap();
        assert_eq!(all.iter().map(|n| n.id).collect::<Vec<_>>(), [4, 3, 2]);
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(index.exact(bad, 0.0, 1), Err(QueryError::NonFinite));
            assert_eq!(index.exact(0.0, bad, 0), Err(QueryError::NonFinite));
        }
    }
    for empty in trees(&[]) {
        assert_eq!(empty.exact(1.0, 2.0, 3).unwrap(), []);
    }
    // A dynamic tree emptied by removal.
    let mut drained = DynamicIndex::bulk_load(vec![(r(1.0, 1.0, 2.0, 2.0), 1)]);
    assert!(drained.remove(&r(1.0, 1.0, 2.0, 2.0), 1));
    assert_eq!(drained.nearest_exact(0.0, 0.0, 2).unwrap(), []);
}

/// The clamp is the nearest point of the box. From `(60, 0)` the long box
/// is exactly 2 away (its nearest point is `(60, 2)`), and the point box at
/// `(60, −2 − 2⁻⁵¹)` is just farther, so the bounds keep both as candidates
/// for k = 1. Measured to the boxes' centers the order would flip, since
/// the long box's center, `(0, 2.5)`, is about 60 away.
#[test]
fn the_ranking_measures_to_the_box_not_its_center() {
    let long = r(-100.0, 2.0, 100.0, 3.0);
    let below = -2.0 - libm_exp2(-51);
    let point = r(60.0, below, 60.0, below);
    for index in trees(&[(point, 1), (long, 2)]) {
        assert_eq!(
            index.cands(60.0, 0.0, 1).unwrap().len(),
            2,
            "both are candidates"
        );
        let got = index.exact(60.0, 0.0, 1).unwrap();
        assert_eq!(
            got.iter().map(|n| n.id).collect::<Vec<_>>(),
            [2],
            "{}",
            index.which()
        );
    }
}

// ===========================================================================
// The full-range case file
// ===========================================================================

fn hex(v: f64) -> String {
    format!("{:016x}", v.to_bits())
}

/// Writes `situs_nearest_exact.csv` to the system temporary directory, for
/// an independent check in exact rational arithmetic. Each row is one case,
/// with the query point, `k`, the entries (`min_x:min_y:max_x:max_y:id` as
/// hex bit patterns and a decimal id, `;`-separated), and the ids
/// `nearest_exact` returned on the static index and on a churned dynamic
/// tree. Ids are unique within a case.
#[test]
#[ignore = "writes the case file situs_nearest_exact.csv to the temporary directory"]
fn write_exact_csv() {
    let mut rows: Vec<String> = Vec::new();
    let mut rng = Rng::new(0x5EED_B0CD_8000_0001);
    // Paired exponent regimes for the query and the boxes, as in the bounds
    // case file: ordinary, subnormal, overflow scale and mixed.
    let regimes: [(i32, i32); 5] = [
        (-8, 8),
        (-1074, -1000),
        (900, 1023),
        (-30, 30),
        (-520, -490),
    ];
    let mut push = |x: f64, y: f64, k: usize, es: &[(Rect, u32)]| {
        let s = Index::bulk_load(es.to_vec())
            .nearest_exact(x, y, k)
            .unwrap();
        let d = churned(es).nearest_exact(x, y, k).unwrap();
        let ids = |v: &[Neighbor]| {
            v.iter()
                .map(|n| n.id.to_string())
                .collect::<Vec<_>>()
                .join(";")
        };
        let boxes = es
            .iter()
            .map(|(b, id)| {
                format!(
                    "{}:{}:{}:{}:{id}",
                    hex(b.min_x()),
                    hex(b.min_y()),
                    hex(b.max_x()),
                    hex(b.max_y())
                )
            })
            .collect::<Vec<_>>()
            .join(";");
        rows.push(format!(
            "{},{},{k},{boxes},{},{}",
            hex(x),
            hex(y),
            ids(&s),
            ids(&d)
        ));
    };
    for round in 0..52_000_usize {
        let pe = regimes[round % regimes.len()];
        let be = regimes[(round / regimes.len()) % regimes.len()];
        let n = 2 + (rng.next() % 8) as usize;
        let (x, y) = (rng.float_in(pe.0, pe.1), rng.float_in(pe.0, pe.1));
        let mut es: Vec<(Rect, u32)> = Vec::with_capacity(n);
        for id in 0..n as u32 {
            let (a, b) = (rng.float_in(be.0, be.1), rng.float_in(be.0, be.1));
            let (c, d) = if rng.next().is_multiple_of(3) {
                (a, b) // a point
            } else {
                (rng.float_in(be.0, be.1), rng.float_in(be.0, be.1))
            };
            es.push((r(a.min(c), b.min(d), a.max(c), b.max(d)), id));
        }
        // One case in four adds near-ties, a copy of an entry moved by one
        // float on one bound.
        if rng.next().is_multiple_of(4) {
            let (b0, _) = es[0];
            let up = |v: f64| {
                f64::from_bits(if v >= 0.0 {
                    v.to_bits() + 1
                } else {
                    v.to_bits() - 1
                })
            };
            let moved = Rect::new(
                up(b0.min_x()),
                b0.min_y(),
                up(b0.max_x()).max(up(b0.min_x())),
                b0.max_y(),
            );
            if let Ok(m) = moved {
                es.push((m, n as u32));
            }
        }
        let k = 1 + (rng.next() % (es.len() as u64 + 1)) as usize;
        push(x, y, k, &es);
    }
    // The system temporary directory, so the test writes nothing inside or
    // outside the crate's own tree.
    let path = std::env::temp_dir().join("situs_nearest_exact.csv");
    std::println!("oracle CSV: {}", path.display());
    let header = "x,y,k,entries,static_ids,dynamic_ids";
    std::fs::write(&path, format!("{header}\n{}\n", rows.join("\n")))
        .expect("write the oracle CSV");
    assert!(rows.len() >= 50_000);
}
