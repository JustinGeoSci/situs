// src/nearest/tests.rs

//! Tests of the nearest-neighbor search. The candidate set is exactly
//! `{ e : lower_sq(e) ≤ τ }`, contains the true k nearest and does not
//! depend on the tree's shape, and `browse` yields in nondecreasing
//! `lower_sq`.
//!
//! The brute-force oracle recomputes every entry's bounds with the same
//! certified function (which the bounds tests verify against exact
//! arithmetic on their own), takes τ as the k-th smallest upper bound over
//! all entries, and keeps every entry whose lower bound is at most τ. For
//! integer coordinates the exact distances are recomputed in `u128`, so the
//! claim that the result contains the true k nearest is checked against
//! exact arithmetic rather than against the bounds.
//!
//! Every test runs on three trees over the same entries: the static index,
//! a bulk-loaded dynamic tree, and a dynamic tree built by interleaved
//! insertions and removals of decoys, whose shape is unlike either load's
//! and whose arena has been churned. The oracle reads each tree's own
//! entries and positions, so the exact order, ties included, is checked on
//! every tree.

#![allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]

use super::bounds::tests::{Rng, libm_exp2};
use super::*;
use crate::Rect;
use crate::dynamic::node::Child;
use crate::dynamic::{DynamicIndex, SLOTS};
use crate::static_index::tests::{entries, pseudo_random_boxes};
use crate::tests::{CASES, coord, r};
use proptest::prelude::*;
use std::format;
use std::vec;
use std::vec::Vec;

// ===========================================================================
// The three trees
// ===========================================================================

/// What the oracle needs from a tree: the searches, and the stored entries
/// with the positions the search breaks ties by.
pub(crate) trait Searched {
    fn which(&self) -> &'static str;
    fn count(&self) -> usize;
    fn cands(&self, x: f64, y: f64, k: usize) -> Result<Vec<Neighbor>, QueryError>;
    fn walk(&self, x: f64, y: f64) -> Result<Browse<'_>, QueryError>;
    /// Every stored entry with its search position.
    fn stored(&self) -> Vec<(Rect, u32, usize)>;
    #[cfg(feature = "exact-nearest")]
    fn exact(&self, x: f64, y: f64, k: usize) -> Result<Vec<Neighbor>, QueryError>;
}

impl Searched for Index {
    #[cfg(feature = "exact-nearest")]
    fn exact(&self, x: f64, y: f64, k: usize) -> Result<Vec<Neighbor>, QueryError> {
        self.nearest_exact(x, y, k)
    }
    fn which(&self) -> &'static str {
        "static"
    }
    fn count(&self) -> usize {
        self.len()
    }
    fn cands(&self, x: f64, y: f64, k: usize) -> Result<Vec<Neighbor>, QueryError> {
        self.nearest_candidates(x, y, k)
    }
    fn walk(&self, x: f64, y: f64) -> Result<Browse<'_>, QueryError> {
        self.browse(x, y)
    }
    fn stored(&self) -> Vec<(Rect, u32, usize)> {
        self.entries
            .iter()
            .enumerate()
            .map(|(pos, (rect, id))| (*rect, *id, pos))
            .collect()
    }
}

impl Searched for DynamicIndex {
    #[cfg(feature = "exact-nearest")]
    fn exact(&self, x: f64, y: f64, k: usize) -> Result<Vec<Neighbor>, QueryError> {
        self.nearest_exact(x, y, k)
    }
    fn which(&self) -> &'static str {
        "dynamic"
    }
    fn count(&self) -> usize {
        self.len()
    }
    fn cands(&self, x: f64, y: f64, k: usize) -> Result<Vec<Neighbor>, QueryError> {
        self.nearest_candidates(x, y, k)
    }
    fn walk(&self, x: f64, y: f64) -> Result<Browse<'_>, QueryError> {
        self.browse(x, y)
    }
    fn stored(&self) -> Vec<(Rect, u32, usize)> {
        // The search's position for an entry: its leaf's slot times the
        // slot count plus its place in the leaf.
        let mut out = Vec::new();
        let mut stack: Vec<usize> = self.root.into_iter().collect();
        while let Some(slot) = stack.pop() {
            for (at, child) in self.arena[slot].children().enumerate() {
                match *child {
                    Child::Node(s) => stack.push(s),
                    Child::Entry(rect, id) => out.push((rect, id, slot * SLOTS + at)),
                }
            }
        }
        out
    }
}

/// A decoy's id: far above every id the tests store.
const DECOY: u32 = 0xD000_0000;

/// A dynamic tree over `entries` built by churn: each entry is inserted
/// along with a decoy (the same box under a decoy id, or a box beside it),
/// a third of the decoys are removed as the build goes, every fifth live
/// entry is removed and reinserted, and the remaining decoys are removed at
/// the end. Its live entries are exactly `entries`.
pub(crate) fn churned(entries: &[(Rect, u32)]) -> DynamicIndex {
    let mut index = DynamicIndex::new();
    let mut decoys: Vec<(Rect, u32)> = Vec::new();
    for (i, (rect, id)) in entries.iter().enumerate() {
        index.insert(*rect, *id);
        let decoy_id = DECOY + i as u32;
        let decoy = if i % 2 == 0 {
            *rect
        } else {
            let (x, y) = (rect.max_x(), rect.max_y());
            Rect::new(x, y, x, y).unwrap()
        };
        index.insert(decoy, decoy_id);
        decoys.push((decoy, decoy_id));
        if i % 3 == 2 {
            let (b, d) = decoys.remove(0);
            assert!(index.remove(&b, d), "a decoy was lost");
        }
        if i % 5 == 4 {
            let (b, e) = entries[i / 2];
            assert!(index.remove(&b, e), "an entry was lost");
            index.insert(b, e);
        }
    }
    for (b, d) in decoys {
        assert!(index.remove(&b, d), "a decoy was lost");
    }
    assert_eq!(index.len(), entries.len());
    index
}

/// The three trees over the same entries.
pub(crate) fn trees(entries: &[(Rect, u32)]) -> Vec<std::boxed::Box<dyn Searched>> {
    vec![
        std::boxed::Box::new(Index::bulk_load(entries.to_vec())),
        std::boxed::Box::new(DynamicIndex::bulk_load(entries.to_vec())),
        std::boxed::Box::new(churned(entries)),
    ]
}

// ===========================================================================
// The oracle
// ===========================================================================

/// Every stored entry's `(lower, upper, id, pos)`, by the certified bounds.
fn all_bounds(index: &dyn Searched, x: f64, y: f64) -> Vec<(f64, f64, u32, usize)> {
    index
        .stored()
        .into_iter()
        .map(|(rect, id, pos)| {
            let (l, u) = bounds_sq(x, y, &rect);
            (l, u, id, pos)
        })
        .collect()
}

/// τ: the k-th smallest upper bound over all entries, `+∞` when `k > len`.
fn tau(index: &dyn Searched, x: f64, y: f64, k: usize) -> f64 {
    let mut uppers: Vec<f64> = all_bounds(index, x, y).iter().map(|e| e.1).collect();
    uppers.sort_by(f64::total_cmp);
    if k == 0 || k > uppers.len() {
        f64::INFINITY
    } else {
        uppers[k - 1]
    }
}

/// Brute force: `{ e : lower ≤ τ }`, sorted by `(lower, id, pos)`.
fn brute_candidates(index: &dyn Searched, x: f64, y: f64, k: usize) -> Vec<Neighbor> {
    if k == 0 {
        return Vec::new();
    }
    let t = tau(index, x, y, k);
    let mut kept: Vec<(f64, f64, u32, usize)> = all_bounds(index, x, y)
        .into_iter()
        .filter(|e| e.0 <= t)
        .collect();
    kept.sort_by(|a, b| {
        a.0.total_cmp(&b.0)
            .then_with(|| a.2.cmp(&b.2))
            .then_with(|| a.3.cmp(&b.3))
    });
    kept.into_iter()
        .map(|(l, u, id, _)| Neighbor {
            id,
            lower_sq: l,
            upper_sq: u,
        })
        .collect()
}

/// The identity checked everywhere. The search equals brute force on this
/// tree and point for each k in a fixed list: 0, 1, 2, the node capacity,
/// and the entry count less one, exactly, and plus five.
fn check_candidates(index: &dyn Searched, x: f64, y: f64) {
    let m = Index::NODE_CAPACITY;
    let n = index.count();
    for k in [0, 1, 2, m, n.saturating_sub(1), n, n + 5] {
        let got = index.cands(x, y, k).unwrap();
        let want = brute_candidates(index, x, y, k);
        assert_eq!(
            got,
            want,
            "{}: k = {k}, p = ({x:e}, {y:e}), n = {n}",
            index.which()
        );
    }
}

/// A result as a canonical multiset: sorted `(lower bits, upper bits, id)`.
pub(crate) fn as_set(found: &[Neighbor]) -> Vec<(u64, u64, u32)> {
    let mut set: Vec<(u64, u64, u32)> = found
        .iter()
        .map(|n| (n.lower_sq.to_bits(), n.upper_sq.to_bits(), n.id))
        .collect();
    set.sort_unstable();
    set
}

/// Exact squared distance to an integer-coordinate box, in `u128`.
pub(crate) fn exact_sq_int(x: i64, y: i64, b: &Rect) -> u128 {
    let axis = |p: i64, lo: f64, hi: f64| -> i128 {
        let (lo, hi) = (lo as i64, hi as i64);
        if p < lo {
            i128::from(lo - p)
        } else if p > hi {
            i128::from(p - hi)
        } else {
            0
        }
    };
    let dx = axis(x, b.min_x(), b.max_x());
    let dy = axis(y, b.min_y(), b.max_y());
    (dx * dx + dy * dy) as u128
}

/// Integer entries: unit boxes at pseudo-random integer corners.
pub(crate) fn integer_entries(rng: &mut Rng, n: usize, bound: i64) -> Vec<(Rect, u32)> {
    (0..n)
        .map(|i| {
            let (x, y) = (rng.int(bound) as f64, rng.int(bound) as f64);
            let (w, h) = ((rng.next() % 4) as f64, (rng.next() % 4) as f64);
            (r(x, y, x + w, y + h), i as u32)
        })
        .collect()
}

// ===========================================================================
// Fixed cases
// ===========================================================================

#[test]
fn an_empty_index_and_k_zero_give_empty_results() {
    for empty in trees(&[]) {
        assert_eq!(empty.cands(1.0, 2.0, 3).unwrap(), []);
        assert_eq!(empty.walk(1.0, 2.0).unwrap().count(), 0);
    }
    for one in trees(&[(r(0.0, 0.0, 1.0, 1.0), 9)]) {
        assert_eq!(one.cands(5.0, 5.0, 0).unwrap(), []);
        let got = one.cands(5.0, 5.0, 7).unwrap();
        assert_eq!(got.len(), 1, "{}", one.which());
        assert_eq!(got[0].id, 9);
    }
}

#[test]
fn a_non_finite_query_point_is_refused() {
    for index in trees(&[(r(0.0, 0.0, 1.0, 1.0), 1)]) {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(index.cands(bad, 0.0, 1), Err(QueryError::NonFinite));
            assert_eq!(index.cands(0.0, bad, 1), Err(QueryError::NonFinite));
            assert!(index.walk(bad, 0.0).is_err());
            assert!(index.walk(0.0, bad).is_err());
        }
    }
    // Refused before anything else, on an empty tree too.
    assert_eq!(
        DynamicIndex::new().nearest_candidates(f64::NAN, 0.0, 1),
        Err(QueryError::NonFinite)
    );
    assert_eq!(
        format!("{}", QueryError::NonFinite),
        "query point is NaN or infinite"
    );
}

/// Two entries built so that the second's lower bound equals τ exactly
/// (`L(B) = U(A) = 2⁵⁰ + 1`, bit for bit). The stopping rule is strict, so
/// the tied candidate is kept. The asserted values also pin the constants,
/// since a change to `K_LO` or `K_HI` moves them.
#[test]
fn a_candidate_whose_lower_bound_equals_tau_is_kept() {
    let p25 = libm_exp2(25);
    let a = r(p25, 0.0, p25, 0.0);
    let bx = p25 + libm_exp2(-25);
    let b = r(bx, 0.0, bx, 0.0);
    let (la, ua) = bounds_sq(0.0, 0.0, &a);
    let (lb, ub) = bounds_sq(0.0, 0.0, &b);
    let p50 = libm_exp2(50);
    assert_eq!((la, ua), (p50 - 1.0, p50 + 1.0));
    assert_eq!((lb, ub), (p50 + 1.0, p50 + 3.0));
    for index in trees(&[(a, 1), (b, 2)]) {
        let got = index.cands(0.0, 0.0, 1).unwrap();
        assert_eq!(got.iter().map(|n| n.id).collect::<Vec<_>>(), [1, 2]);
        check_candidates(&*index, 0.0, 0.0);
    }
}

/// A gap smaller than the bound width. Both entries are candidates for
/// k = 1, because their intervals overlap and the search does not guess.
#[test]
fn near_ties_within_the_bound_width_return_both() {
    let h = libm_exp2(-47);
    let a = r(6.0, 8.0, 6.0, 8.0);
    let b = r(6.0 + h, 8.0, 6.0 + h, 8.0);
    for index in trees(&[(a, 1), (b, 2)]) {
        let got = index.cands(0.0, 0.0, 1).unwrap();
        assert_eq!(got.iter().map(|n| n.id).collect::<Vec<_>>(), [1, 2]);
        check_candidates(&*index, 0.0, 0.0);
    }
}

#[test]
fn ties_at_distance_zero_return_every_containing_box() {
    let entries: Vec<(Rect, u32)> = (0..40)
        .map(|i| (r(-f64::from(i) - 1.0, -1.0, f64::from(i) + 1.0, 1.0), i))
        .collect();
    for index in trees(&entries) {
        for k in [1, 3, 40] {
            let got = index.cands(0.0, 0.0, k).unwrap();
            assert_eq!(
                got.len(),
                40,
                "{} k = {k}: a containing box was dropped",
                index.which()
            );
            assert!(got.iter().all(|n| n.lower_sq == 0.0));
        }
        check_candidates(&*index, 0.0, 0.0);
    }
}

#[test]
fn a_point_equidistant_from_two_boxes_returns_both_for_k_one() {
    let entries = [
        (r(-7.0, -1.0, -5.0, 1.0), 1),
        (r(5.0, -1.0, 7.0, 1.0), 2),
        (r(90.0, 90.0, 91.0, 91.0), 3),
    ];
    for index in trees(&entries) {
        let got = index.cands(0.0, 0.0, 1).unwrap();
        assert_eq!(got.iter().map(|n| n.id).collect::<Vec<_>>(), [1, 2]);
        check_candidates(&*index, 0.0, 0.0);
    }
}

#[test]
fn duplicate_entries_are_candidates_once_per_stored_entry() {
    let entries = [(r(3.0, 0.0, 4.0, 1.0), 5), (r(3.0, 0.0, 4.0, 1.0), 5)];
    for index in trees(&entries) {
        let got = index.cands(0.0, 0.0, 1).unwrap();
        assert_eq!(
            got.len(),
            2,
            "{}: a stored duplicate is its own candidate",
            index.which()
        );
        assert!(got.iter().all(|n| n.id == 5));
    }
}

/// The same entries in three input orders, on all three trees, give nine
/// shapes and one candidate set. The static trees agree in order too. The
/// dynamic ones agree as sets, because their positions differ and equal
/// `(lower, id)` may come in another order.
#[test]
fn the_result_does_not_depend_on_the_tree_shape() {
    let mut rng = Rng::new(0x5EED_B0CD_4000_0004);
    for n in [17, 300, 1200] {
        let base = integer_entries(&mut rng, n, 1 << 20);
        let mut reversed = base.clone();
        reversed.reverse();
        let mut shuffled = base.clone();
        for i in (1..shuffled.len()).rev() {
            shuffled.swap(i, (rng.next() as usize) % (i + 1));
        }
        let (i1, i2, i3) = (
            Index::bulk_load(base.clone()),
            Index::bulk_load(reversed.clone()),
            Index::bulk_load(shuffled.clone()),
        );
        let mut all: Vec<std::boxed::Box<dyn Searched>> = Vec::new();
        for order in [&base, &reversed, &shuffled] {
            all.extend(trees(order));
        }
        for (x, y) in [(0.0, 0.0), (1e6, -3.5), (-17.0, 123_456.0)] {
            for k in [1, 2, 7, n / 2, n] {
                let a = i1.nearest_candidates(x, y, k).unwrap();
                assert_eq!(
                    a,
                    i2.nearest_candidates(x, y, k).unwrap(),
                    "n = {n}, k = {k}"
                );
                assert_eq!(
                    a,
                    i3.nearest_candidates(x, y, k).unwrap(),
                    "n = {n}, k = {k}"
                );
                for t in &all {
                    assert_eq!(
                        as_set(&a),
                        as_set(&t.cands(x, y, k).unwrap()),
                        "{}: n = {n}, k = {k}",
                        t.which()
                    );
                }
            }
        }
    }
}

/// Integer coordinates, with exact distances in `u128`. The true k nearest
/// (with every tie at the k-th) are a subset of the result, and where the
/// exact distances are distinct integers the result has exactly k entries.
#[test]
fn candidates_contain_the_true_k_nearest_exactly() {
    let mut rng = Rng::new(0x5EED_B0CD_4000_0005);
    for _ in 0..300 {
        let n = 2 + (rng.next() % 400) as usize;
        let entries = integer_entries(&mut rng, n, 1 << 18);
        let (x, y) = (rng.int(1 << 18), rng.int(1 << 18));
        let mut exact: Vec<(u128, u32)> = entries
            .iter()
            .map(|(rect, id)| (exact_sq_int(x, y, rect), *id))
            .collect();
        exact.sort_unstable();
        for index in trees(&entries) {
            for k in [1usize, 2, 5, n] {
                let k = k.min(n);
                let got = index.cands(x as f64, y as f64, k).unwrap();
                let ids: Vec<u32> = got.iter().map(|n| n.id).collect();
                let dk = exact[k - 1].0;
                for (d, id) in exact.iter().filter(|(d, _)| *d <= dk) {
                    assert!(
                        ids.contains(id),
                        "{}: entry {id} at exact d² = {d} missing for k = {k} (d²ₖ = {dk})",
                        index.which()
                    );
                }
                // Distinct, well-separated integer distances: exactly k.
                let distinct = exact.windows(2).all(|w| w[0].0 != w[1].0);
                if distinct {
                    assert_eq!(got.len(), k, "k = {k} with distinct exact distances");
                }
            }
        }
    }
}

// ===========================================================================
// Distance defect classes
// ===========================================================================

/// Rounded ranking. Two entries' exact distances differ by 1 at 2⁵⁴, where
/// `f64` cannot represent the difference. The true nearest is always in the
/// result.
#[test]
fn distance_rounded_near_ties_keep_the_true_nearest() {
    let p27 = libm_exp2(27);
    let a = r(p27, 0.0, p27, 0.0); //        d² = 2⁵⁴
    let b = r(p27, 1.0, p27, 1.0); //        d² = 2⁵⁴ + 1, rounds to 2⁵⁴
    for index in trees(&[(b, 2), (a, 1)]) {
        let got = index.cands(0.0, 0.0, 1).unwrap();
        let ids: Vec<u32> = got.iter().map(|n| n.id).collect();
        assert!(ids.contains(&1), "the true nearest was dropped: {ids:?}");
        assert!(
            ids.contains(&2),
            "the indistinguishable tie was guessed away"
        );
        // Under the feature, the exact ranking resolves what the bounds
        // cannot. `a` is nearer by exactly 1 at 2⁵⁴, so it alone is
        // returned for k = 1, and it comes first for k = 2.
        #[cfg(feature = "exact-nearest")]
        {
            let one = index.exact(0.0, 0.0, 1).unwrap();
            assert_eq!(one.iter().map(|n| n.id).collect::<Vec<_>>(), [1]);
            let two = index.exact(0.0, 0.0, 2).unwrap();
            assert_eq!(two.iter().map(|n| n.id).collect::<Vec<_>>(), [1, 2]);
        }
    }
}

/// Underflow. Distances whose squares underflow are not ranked by the
/// rounded value. Everything indistinguishable is returned, and a truly
/// nearer subnormal entry is never lost to a farther one.
#[test]
fn distance_underflow_keeps_the_true_nearest() {
    let near = libm_exp2(-520); // d² = 2⁻¹⁰⁴⁰, below the threshold
    let far = libm_exp2(-200); //  d² = 2⁻⁴⁰⁰, far above it
    for index in trees(&[(r(far, 0.0, far, 0.0), 2), (r(near, 0.0, near, 0.0), 1)]) {
        let got = index.cands(0.0, 0.0, 1).unwrap();
        let ids: Vec<u32> = got.iter().map(|n| n.id).collect();
        assert_eq!(ids, [1], "underflow mis-ranked the subnormal distance");
    }
    // Entries indistinguishable below the threshold are all returned.
    let coincident: Vec<(Rect, u32)> = (0..5)
        .map(|i| {
            let dx = f64::from(i + 1) * libm_exp2(-1074);
            (r(dx, 0.0, dx, 0.0), i as u32)
        })
        .collect();
    for index in trees(&coincident) {
        let got = index.cands(0.0, 0.0, 1).unwrap();
        assert_eq!(got.len(), 5, "sub-resolution distances were guessed apart");
    }
}

/// Overflow. An entry whose squared distance overflows lies a valid, finite
/// lower bound away, never NaN, and never beats a moderate entry.
#[test]
fn distance_overflow_keeps_the_true_nearest() {
    let entries = [
        (r(1e300, 0.0, 1e300, 0.0), 2),
        (r(-1e308, -1e308, -1e308, -1e308), 3),
        (r(1000.0, 0.0, 1001.0, 1.0), 1),
    ];
    for index in trees(&entries) {
        let got = index.cands(0.0, 0.0, 1).unwrap();
        assert_eq!(got.iter().map(|n| n.id).collect::<Vec<_>>(), [1]);
        for n in &got {
            assert!(n.lower_sq.is_finite());
            assert!(!n.upper_sq.is_nan());
        }
        check_candidates(&*index, 0.0, 0.0);
    }
}

/// Empty and one-entry trees raise no panic and use no sentinel box, the
/// edge cases of a distance query. The dynamic tree is also emptied by
/// removal, so the case is reached by both paths.
#[test]
fn distance_empty_and_tiny_trees_are_safe() {
    for empty in trees(&[]) {
        assert_eq!(empty.cands(1e300, -1e300, 5).unwrap(), []);
        assert_eq!(empty.walk(0.0, 0.0).unwrap().count(), 0);
    }
    for one in trees(&[(r(-1.0, -1.0, 1.0, 1.0), 42)]) {
        assert_eq!(one.cands(0.0, 0.0, 3).unwrap().len(), 1);
        assert_eq!(one.walk(1e300, 1e300).unwrap().count(), 1);
    }
    let mut drained = DynamicIndex::bulk_load(vec![(r(-f64::MAX, 0.0, f64::MAX, 0.0), 1)]);
    assert!(drained.remove(&r(-f64::MAX, 0.0, f64::MAX, 0.0), 1));
    assert_eq!(drained.nearest_candidates(1e308, -1e308, 4).unwrap(), []);
    assert_eq!(drained.browse(0.0, 0.0).unwrap().count(), 0);
}

// ===========================================================================
// browse
// ===========================================================================

/// `browse` yields every stored entry exactly once, in nondecreasing
/// `lower_sq`, with ties by id and then position, and each entry's bounds
/// equal the direct computation. On the churned tree no removed decoy
/// appears.
#[test]
fn browse_yields_every_entry_in_nondecreasing_lower_order() {
    let source = pseudo_random_boxes(2_000);
    for index in trees(&source) {
        for (x, y) in [(0.0, 0.0), (37.5, -12.0), (1e300, 1e300)] {
            let yielded: Vec<Neighbor> = index.walk(x, y).unwrap().collect();
            assert_eq!(yielded.len(), index.count());
            assert!(
                yielded.iter().all(|n| n.id < DECOY),
                "a removed entry was yielded"
            );
            let mut expected = all_bounds(&*index, x, y);
            expected.sort_by(|a, b| {
                a.0.total_cmp(&b.0)
                    .then_with(|| a.2.cmp(&b.2))
                    .then_with(|| a.3.cmp(&b.3))
            });
            for (got, want) in yielded.iter().zip(&expected) {
                assert_eq!(
                    (got.lower_sq, got.upper_sq, got.id),
                    (want.0, want.1, want.2),
                    "{}",
                    index.which()
                );
            }
        }
    }
}

/// The browse contract, against exact arithmetic. Once a neighbor with
/// `lower_sq = ℓ` is yielded, every entry not yet yielded has exact
/// `d² ≥ ℓ`.
#[test]
fn browse_prefix_bounds_every_unseen_exact_distance() {
    let mut rng = Rng::new(0x5EED_B0CD_4000_0006);
    for _ in 0..50 {
        let n = 2 + (rng.next() % 200) as usize;
        let entries = integer_entries(&mut rng, n, 1 << 15);
        let (x, y) = (rng.int(1 << 15), rng.int(1 << 15));
        for index in trees(&entries) {
            let yielded: Vec<Neighbor> = index.walk(x as f64, y as f64).unwrap().collect();
            // Entries are unique by id here.
            for (i, n_i) in yielded.iter().enumerate() {
                for later in &yielded[i..] {
                    let d2 = entries
                        .iter()
                        .filter(|(_, id)| *id == later.id)
                        .map(|(rect, _)| exact_sq_int(x, y, rect))
                        .min()
                        .unwrap();
                    // ℓ ≤ every unseen exact d², tested as the exact
                    // integer comparison ceil(ℓ) ≤ d².
                    assert!(
                        n_i.lower_sq.ceil() as u128 <= d2,
                        "lower {} exceeds a later exact d² {}",
                        n_i.lower_sq,
                        d2
                    );
                }
            }
        }
    }
}

// ===========================================================================
// Property tests
// ===========================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(CASES))]

    /// Set identity against brute force for the fixed list of k, on up to
    /// 2,000 entries from the crate's box strategy and a point from its
    /// coordinate strategy, on each of the three trees.
    #[test]
    fn candidates_equal_brute_force(es in entries(2_000), x in coord(), y in coord()) {
        for index in trees(&es) {
            check_candidates(&*index, x, y);
        }
    }

    /// The set agreement. For the same live entries, the dynamic trees
    /// (bulk-loaded, and churned by insertions and removals) return the
    /// static index's candidate set, as sorted `(lower bits, upper bits,
    /// id)` triples.
    #[test]
    fn dynamic_and_static_candidate_sets_agree(
        es in entries(1_000),
        x in coord(),
        y in coord(),
        k in 0usize..40,
    ) {
        let static_index = Index::bulk_load(es.clone());
        let want = as_set(&static_index.nearest_candidates(x, y, k).unwrap());
        let loaded = DynamicIndex::bulk_load(es.clone());
        prop_assert_eq!(&as_set(&loaded.nearest_candidates(x, y, k).unwrap()), &want);
        let churn = churned(&es);
        prop_assert_eq!(&as_set(&churn.nearest_candidates(x, y, k).unwrap()), &want);
    }

    /// `browse` on random cases is complete and nondecreasing, and it never
    /// yields an entry that was removed.
    #[test]
    fn browse_is_sorted_and_complete(es in entries(300), x in coord(), y in coord()) {
        for index in trees(&es) {
            let yielded: Vec<Neighbor> = index.walk(x, y).unwrap().collect();
            prop_assert_eq!(yielded.len(), index.count());
            for w in yielded.windows(2) {
                prop_assert!(w[0].lower_sq <= w[1].lower_sq);
            }
            for n in &yielded {
                prop_assert!(n.id < DECOY, "a removed entry was yielded");
                prop_assert!(n.lower_sq <= n.upper_sq);
                prop_assert!(n.lower_sq.is_finite() && n.lower_sq >= 0.0);
            }
        }
    }
}
