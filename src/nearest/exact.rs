// src/nearest/exact.rs

//! Exact nearest-neighbor ranking, under the `exact-nearest` feature.
//!
//! The certified bounds of the parent module return a candidate set that
//! holds the true k nearest, but they cannot always establish which of two
//! candidates is nearer. Two entries whose bound intervals overlap are both
//! returned, unranked. This module ranks them exactly. The bounds and the
//! search remain in `situs`, and the one exact operation, comparing the
//! distances from the query point to two points, is
//! `arithmetica::compare_distance`.
//!
//! # The method
//!
//! 1. **Candidates:** `C` is the candidate set of
//!    [`Index::nearest_candidates`], unchanged.
//! 2. **Clamp:** For each candidate `e`, `q(e)` is the query point clamped to
//!    `e`'s box. Each coordinate is the query's own when it lies inside the
//!    box's interval on that axis, and otherwise the interval's nearer end,
//!    chosen by exact comparisons of stored values. `q(e)` is therefore a
//!    selection of input values rather than a computed one, and
//!    `|p − q(e)|` is exactly the Euclidean distance from `p` to the box.
//! 3. **Sort**: `C` is sorted by `compare_distance(p, q(a), q(b))`, then by id,
//!    then by internal position. `compare_distance` is exact for every
//!    finite input, so the first key is a total preorder by true distance,
//!    which the two tie-breaks refine into a total order.
//! 4. **Cut:** The first `k` are kept, then every following candidate whose
//!    distance equals the k-th's exactly.
//!
//! # Why the result is the exact answer over all entries
//!
//! Let `d²ₖ` be the true k-th smallest squared distance over every stored
//! entry. The candidate set contains every entry with `d² ≤ d²ₖ`, which is
//! the true k nearest and every entry exactly tied with the k-th (the
//! argument is on [`Index::nearest_candidates`]). An entry outside the set
//! has `lower_sq > τ ≥ d²ₖ`, so its true distance is strictly greater than
//! the k-th's, and it can be neither among the k nearest nor tied with the
//! k-th. Sorting the set exactly therefore orders every entry that can
//! appear in the answer, and the cut at the k-th, with its exact ties,
//! recovers the exact answer.
//!
//! # Why the comparator cannot fail
//!
//! `compare_distance` returns an error only for a non-finite coordinate.
//! The query point is checked before the search starts, and every stored
//! box is finite (a [`Rect`] cannot be built otherwise), so every clamped
//! point is finite too. The comparator therefore treats an error as
//! unreachable. A `debug_assert!` states it, and a release build, which
//! cannot reach it, orders the pair as equal rather than panicking.

use alloc::vec::Vec;
use core::cmp::Ordering;

use arithmetica::{Sign, compare_distance};

use crate::Rect;
use crate::dynamic::DynamicIndex;
use crate::nearest::{Item, Neighbor, QueryError, Tree, candidate_items};
use crate::static_index::Index;

// ===========================================================================
// The ranking, shared by both trees
// ===========================================================================

/// The point of `rect` nearest to `(x, y)`, the query clamped to the box.
///
/// Exact comparisons select stored values, and nothing is computed.
fn clamp(x: f64, y: f64, rect: &Rect) -> [f64; 2] {
    let axis = |v: f64, lo: f64, hi: f64| {
        if v < lo {
            lo
        } else if v > hi {
            hi
        } else {
            v
        }
    };
    [
        axis(x, rect.min_x(), rect.max_x()),
        axis(y, rect.min_y(), rect.max_y()),
    ]
}

/// The exact comparison of `a`'s and `b`'s distances from `p`, `Less` when
/// `a` is nearer.
fn by_distance(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> Ordering {
    match compare_distance(p, a, b) {
        Ok(Sign::Negative) => Ordering::Less,
        Ok(Sign::Zero) => Ordering::Equal,
        Ok(Sign::Positive) => Ordering::Greater,
        Err(_) => {
            debug_assert!(
                false,
                "compare_distance refused finite input: the query point is checked \
                 and every stored box is finite"
            );
            Ordering::Equal
        }
    }
}

/// `nearest_exact` over either tree (module doc).
fn exact(tree: Tree<'_>, x: f64, y: f64, k: usize) -> Result<Vec<Neighbor>, QueryError> {
    let candidates: Vec<Item> = candidate_items(tree, x, y, k)?;
    let p = [x, y];
    let mut ranked: Vec<(Item, [f64; 2])> = candidates
        .into_iter()
        .map(|item| (item, clamp(x, y, &tree.entry_rect(item.pos))))
        .collect();
    ranked.sort_by(|(a, qa), (b, qb)| {
        by_distance(p, *qa, *qb)
            .then_with(|| a.id.cmp(&b.id))
            .then_with(|| a.pos.cmp(&b.pos))
    });
    if ranked.len() > k {
        // The k-th, then every later candidate exactly as near.
        let kth = ranked[k - 1].1;
        let ties = ranked[k..]
            .iter()
            .take_while(|(_, q)| by_distance(p, kth, *q) == Ordering::Equal)
            .count();
        ranked.truncate(k + ties);
    }
    Ok(ranked
        .into_iter()
        .map(|(item, _)| item.neighbor())
        .collect())
}

// ===========================================================================
// The public methods
// ===========================================================================

impl Index {
    /// The `k` entries nearest to `(x, y)` by exact Euclidean distance to
    /// their boxes, plus every entry exactly as near as the k-th.
    ///
    /// The result is ordered by exact distance, and entries at exactly equal
    /// distance by id, then by internal position. Each [`Neighbor`] still
    /// carries its certified bounds, while the order and the cut are exact.
    /// The result has exactly `k` entries unless the k-th is exactly tied,
    /// and it never guesses between candidates the bounds alone could not
    /// order.
    ///
    /// **The ranking is exact by construction.** The candidates of
    /// [`Index::nearest_candidates`] contain every entry within the k-th
    /// true distance, ties included. Each is ranked by comparing the exact
    /// distances from the query point to the nearest points of the boxes
    /// (the query clamped to each box, a selection of stored values), with
    /// `arithmetica`'s exact `compare_distance`. Entries outside the
    /// candidate set are strictly farther than the k-th, so ranking the set
    /// ranks every entry that can be in the answer. The `nearest` module of
    /// the crate's source carries the full argument.
    ///
    /// `k = 0` gives an empty result, and `k >= len` gives every entry,
    /// exactly ordered.
    ///
    /// ```
    /// use situs::{Index, Rect};
    ///
    /// let r = |a, b, c, d| Rect::new(a, b, c, d).unwrap();
    /// // Squared distances from the origin are 2^54 and 2^54 + 1. They round
    /// // to the same f64, so the bounds cannot order them, and the exact
    /// // ranking can.
    /// let p27 = 134_217_728.0; // 2^27
    /// let index = Index::bulk_load(vec![(r(p27, 1.0, p27, 1.0), 1), (r(p27, 0.0, p27, 0.0), 2)]);
    /// assert_eq!(index.nearest_candidates(0.0, 0.0, 1).unwrap().len(), 2);
    /// let exact = index.nearest_exact(0.0, 0.0, 1).unwrap();
    /// assert_eq!(exact.len(), 1);
    /// assert_eq!(exact[0].id, 2);
    /// ```
    ///
    /// # Errors
    ///
    /// [`QueryError::NonFinite`] if `x` or `y` is NaN or infinite, checked
    /// before anything else.
    #[cfg_attr(docsrs, doc(cfg(feature = "exact-nearest")))]
    pub fn nearest_exact(&self, x: f64, y: f64, k: usize) -> Result<Vec<Neighbor>, QueryError> {
        exact(Tree::Static(self), x, y, k)
    }
}

impl DynamicIndex {
    /// The `k` entries nearest to `(x, y)` by exact Euclidean distance to
    /// their boxes, plus every entry exactly as near as the k-th.
    ///
    /// This is the ranking of [`Index::nearest_exact`], run by the same code,
    /// with the same guarantee and argument. For the same stored entries the
    /// two trees return the same entries in the same order, except that
    /// entries at exactly equal distance with equal ids follow each tree's
    /// internal position.
    ///
    /// # Errors
    ///
    /// [`QueryError::NonFinite`] if `x` or `y` is NaN or infinite.
    #[cfg_attr(docsrs, doc(cfg(feature = "exact-nearest")))]
    pub fn nearest_exact(&self, x: f64, y: f64, k: usize) -> Result<Vec<Neighbor>, QueryError> {
        exact(Tree::Dynamic(self), x, y, k)
    }
}

#[cfg(test)]
mod tests;
