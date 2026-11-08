// src/lib.rs

//! Spatial indexes over `f64` boxes, with no false negatives.
//!
//! `situs` answers two questions of a set of stored boxes: **which of them
//! might meet a query box**, and **which might lie nearest to a query point**. It
//! holds `(`[`Rect`]`, u32)` entries in an R-tree, either the static,
//! bulk-loaded [`Index`], built once and queried many times, or the
//! [`DynamicIndex`], which accepts insertions and removals as the set
//! changes. A box query returns every entry whose box meets the window, and
//! a nearest-neighbor query returns candidates ordered by certified bounds.
//! The index operates as a **candidate filter** rather than a decision
//! procedure. The caller decides exactly among the candidates, and `situs`
//! decides nothing geometric itself.
//!
//! The crate is `no_std` (with `alloc`), has no dependencies by default,
//! uses no `unsafe` code, and does not panic on any input it accepts.
//!
//! # The correctness argument
//!
//! The index carries one obligation, **no false negatives**: every entry whose
//! box meets the query box is returned. An extra entry would cost time
//! rather than correctness, since the caller decides exactly afterward. A
//! query in fact returns *exactly* the entries whose boxes meet the window
//! (see [`Index::query`]), because every leaf entry's own box is tested
//! before it is reported.
//!
//! No false negatives holds if and only if three conditions hold:
//!
//! 1. **Every entry's box contains its geometry.** This condition is the
//!    caller's obligation, met by taking the box as the minimum and maximum
//!    of stored coordinates, which is exact and requires no arithmetic.
//! 2. **Every node's box contains all of its children's boxes.** A node's box is
//!    the [`Rect::union`] of its children's boxes, a minimum and maximum of
//!    stored values and therefore exact. By induction, a box that meets an
//!    entry meets every ancestor of that entry, so the search never prunes
//!    an ancestor of a match.
//! 3. **The overlap test is the exact closed-interval test on stored values**,
//!    `a.min_x <= b.max_x && b.min_x <= a.max_x` and the same on `y`
//!    ([`Rect::intersects`]). The intervals are closed, so boxes that touch
//!    along an edge or at a corner meet, which is what a caller deciding
//!    exactly afterward requires. A contact *is* an intersection.
//!
//! **Every comparison in the crate is made on stored `f64` values, on
//! certified bounds whose error can only add candidates, or, under the
//! `exact-nearest` feature, on exact predicates from `arithmetica`. No
//! distance is returned as though it were exact.** The bounds, the searches
//! and the join remain in `situs`, and the one exact operation the feature
//! requires, comparing the distances from a query point to two stored
//! points, belongs to `arithmetica`.
//!
//! **The bulk loader is the one place where arithmetic enters.** It sorts
//! entries by the centers of their boxes, computed in floating point, and
//! those keys determine the *shape* of the tree rather than its *contents*. A
//! poor key produces a slower tree with the same answers. The tests attack
//! each of the three conditions separately, and a mutation test breaks each
//! condition in turn and fails a named test.
//!
//! # Bulk loading
//!
//! [`Index::bulk_load`] packs the entries by **Sort-Tile-Recursive** (STR)
//! packing (Leutenegger, Lopez & Edgington, "STR: A Simple and Efficient
//! Algorithm for R-Tree Packing", ICDE 1997). With `n` entries and node
//! capacity `M` ([`Index::NODE_CAPACITY`]), the method has four stages:
//!
//! - let `P = ⌈n / M⌉` be the number of leaves and `S = ⌈√P⌉`;
//! - sort the entries by the `x` of their box centers and cut them into `S`
//!   vertical slices of `S·M` consecutive entries, the last slice shorter;
//! - sort each slice by the `y` of the centers and pack consecutive runs of
//!   `M` into leaves;
//! - treat the leaf boxes as entries and repeat, level by level, until one
//!   node remains.
//!
//! A slice holds a multiple of `M` entries, so the runs of `M` over the
//! reordered array coincide with the per-slice runs, and a level is simply
//! the array cut into chunks of `M`, the last chunk possibly shorter.
//!
//! Hilbert-curve packing (Kamel & Faloutsos, "On Packing R-trees", CIKM
//! 1993) orders entries along a space-filling curve instead and gives
//! somewhat better query performance on some distributions. It remains the
//! documented alternative rather than a built one. STR is simpler to state
//! and to verify, its layout is a plain function of two sorts, and no
//! measured workload has required more.
//!
//! **The layout is a deterministic function of the input sequence.** Each sort
//! orders entries by their centers in `f64::total_cmp` order, then by entry
//! id, then by their position before the sort, which is the order a stable
//! sort by center and id would produce. A center is `min · 0.5 + max ·
//! 0.5`, which cannot overflow for finite bounds, since each product is at
//! most half its bound and their sum at most the larger bound in magnitude.
//! The rounding in that sum, and the loss of precision when the bounds are
//! subnormal, affect only the order.
//!
//! # Storage and search
//!
//! The nodes live in one `Vec` in level order, root first, and the entries
//! live in another, reordered into leaf order. Each node holds its box and
//! the index range of its children, nodes for an internal node and entries
//! for a leaf.
//!
//! **The containment shortcut** reports a whole subtree once the query window
//! *contains* its box. Every entry's box is contained in its leaf's box, and
//! every node's box in its parent's (condition 2). Containment is
//! transitive, and a box contained in the window meets it because the test
//! is closed. The search therefore reports that subtree without testing
//! anything below it. The containment test ([`Rect::contains_rect`]) is four
//! exact comparisons of stored values, so the shortcut adds no arithmetic
//! and can neither add an entry nor lose one.
//!
//! The shortcut **walks** the subtree rather than taking it as one range of the
//! entry array. STR reorders each level after its nodes are cut, so the
//! nodes of a level are not in child-range order, and a subtree's entries
//! are in general not contiguous. The range from its first leaf's first
//! entry to its last leaf's end can hold entries of other subtrees and miss
//! some of its own.
//!
//! **The search is iterative and runs on a fixed stack.** It pops a node, and
//! if the node is internal it pushes all of its children, in reverse, so
//! that they are visited in index order. At any moment the stack holds the
//! siblings not yet visited of each node on the current root-to-node path,
//! at most `M − 1` per level, plus the children just pushed. With `depth`
//! node levels the stack holds at most `(M − 1) · (depth − 1) + 1` indices,
//! below `(M − 1) · depth + 1`. The shortcut's subtree walk pushes onto the
//! same stack with the same discipline, starting from the node just popped,
//! so it is one more descent of the same kind and the bound holds through
//! it.
//!
//! The depth is bounded as well. STR packing makes `⌈n / M⌉` leaves, and
//! each level above divides the count by `M = 16 = 2⁴`, rounding up, so with
//! `n ≤ usize::MAX` there are at most `⌈usize::BITS / 4⌉ + 1` levels (17 on
//! a 64-bit target). The stack is therefore a fixed array of
//! `(M − 1) · 17 + 1 = 256` indices on a 64-bit target (2 KiB), held on the
//! call stack. No query allocates, and the search does not recurse. Were the
//! bound ever exceeded, the search would panic rather than drop a node,
//! since a dropped node would remove its whole subtree from the answer.
//!
//! **The visit order is the order of this depth-first walk.** The shortcut emits
//! a subtree in the order the plain search would visit it, because every
//! node and entry below a contained node meets the window and the walk is
//! the same walk. [`Index::try_query`], which can stop early, therefore
//! visits a prefix of [`Index::query`]'s sequence.
//!
//! # Joins
//!
//! [`Index::join`] reports every pair of entries, one from each of two
//! indexes, whose boxes meet, and [`Index::self_join`] reports every pair of
//! distinct entries of one index whose boxes meet, once each. The traversal
//! is the synchronized descent of Brinkhoff, Kriegel & Seeger ("Efficient
//! Processing of Spatial Joins Using R-trees", SIGMOD 1993). It walks pairs
//! of nodes, one from each tree, and descends a pair only when the two
//! boxes meet. The correctness argument is the query's, applied pairwise:
//!
//! - **No pair is lost:** If entry `a` (below node `N`) and entry `b` (below
//!   node `P`) have boxes that meet, they share a point. `N`'s box contains
//!   `a`'s and `P`'s contains `b`'s (condition 2), so both node boxes
//!   contain that point and meet. By induction every ancestor pair of
//!   `(a, b)` is descended, and the pair is reached.
//! - **No pair is added:** At two leaves, each entry pair is tested with the
//!   closed overlap test (condition 3) before it is reported.
//!
//! A leaf paired with an internal node is paired with each of that node's
//! children, which is how two trees of different heights meet. A join's
//! output can be quadratic in the sizes of the indexes, so it is never
//! collected. Each pair goes to the caller's visitor as it is found, and
//! only the pending node pairs are stored.
//!
//! # Nearest neighbors
//!
//! [`Index::nearest_candidates`] and [`Index::browse`] find the entries
//! nearest to a query point without computing any exact distance. For each
//! box the crate computes two `f64` values that bracket the exact squared
//! distance, `lower_sq ≤ d² ≤ upper_sq`, certified by a rounding-error
//! argument (see the `nearest` module documentation). A best-first search
//! over the lower bounds (Hjaltason & Samet, "Distance Browsing in Spatial
//! Databases", ACM TODS 24(2), 1999) then returns every entry whose interval
//! could place it among the k nearest. The result always includes the true
//! k nearest and every tie, and it never chooses between candidates whose
//! intervals overlap. The bounds reach the caller as bounds, and no computed
//! distance is presented as the exact one.
//!
//! Under the opt-in `exact-nearest` feature, `Index::nearest_exact` and
//! `DynamicIndex::nearest_exact` rank those candidates exactly, with
//! `arithmetica::compare_distance` on the nearest point of each box, and
//! return the exact k nearest plus every entry exactly tied with the k-th.
//! The default build has no dependencies, and the feature adds
//! `arithmetica`, which is also `no_std`.
//!
//! # The dynamic tree
//!
//! [`DynamicIndex`] is an R\*-tree (Beckmann, Kriegel, Schneider & Seeger,
//! SIGMOD 1990) that accepts insertions and exact removals, alongside the
//! read-only [`Index`]. The R\* heuristics are computed in `f64` and
//! determine only the tree's shape. Every node box is maintained as the
//! exact union of its children's boxes, so the argument above applies
//! unchanged, and [`DynamicIndex::bounds`] is always exact and tight. The
//! two types remain separate because their contracts differ. The static
//! index is compact and the fastest to query, and the dynamic tree can
//! change. Removal matches a box and an id exactly and recomputes every box
//! on its path, so the bounds shrink as entries leave.
//! [`DynamicIndex::bulk_load`] builds an initial tree by the static index's
//! packing, and nearest-neighbor candidates and browsing run on the dynamic
//! tree by the same search and the same argument as on the static index.
//!
//! # Not built
//!
//! - [`Index::join`], [`Index::self_join`], the early-exit
//!   [`Index::try_query`] and the containment shortcut have no dynamic
//!   counterparts. [`DynamicIndex::query`] tests every entry it reaches.
//! - Any distance presented as exact. Nearest-neighbor results carry
//!   certified bounds rather than a distance.
//! - An encoding of an index. An index is rebuilt from its entries, which
//!   the caller already holds.

#![no_std]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used)]
#![warn(missing_docs)]

extern crate alloc;

#[cfg(test)]
extern crate std;

mod dynamic;
mod nearest;
mod static_index;

pub use dynamic::DynamicIndex;
pub use nearest::{Browse, Neighbor, QueryError};
pub use static_index::Index;

// The README's examples run as doctests, so the published README cannot
// drift from the API. Compiled only by `cargo test --doc`.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

// ===========================================================================
// Rect
// ===========================================================================

/// An axis-aligned box with closed intervals on both axes.
///
/// # Invariants
///
/// - All four bounds are finite.
/// - `min_x <= max_x` and `min_y <= max_y`.
///
/// A box of zero width, zero height, or both (a segment, or a point) is
/// valid: it is a legitimate entry and a legitimate query. Only non-finite
/// and inverted bounds are refused, by [`Rect::new`], so a `Rect` that
/// exists satisfies the invariants.
///
/// The rule is four finite bounds, not inverted, and nothing more. A crate
/// whose own box type has the same validity rules converts to and from
/// `Rect` without a failure path, because each side accepts every value the
/// other can hold.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rect {
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
}

/// Why [`Rect::new`] refused its bounds.
///
/// Non-exhaustive: a match on it needs a wildcard arm, so that a later
/// variant is not a breaking change.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum RectError {
    /// A bound is NaN or infinite. Such a box has no extent to test, and a
    /// NaN bound would make every comparison false, so it is refused rather
    /// than stored.
    NonFinite,
    /// `min > max` on some axis. A zero-extent box (`min == max`) is valid;
    /// an inverted one would meet nothing and be silently useless.
    Inverted,
}

impl core::fmt::Display for RectError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonFinite => f.write_str("rect bound is NaN or infinite"),
            Self::Inverted => f.write_str("rect min exceeds max on an axis"),
        }
    }
}

impl core::error::Error for RectError {}

impl Rect {
    /// Builds a box from its bounds.
    ///
    /// Non-finiteness is checked before inversion, because a NaN makes the
    /// inversion comparison false and would otherwise be reported as a
    /// valid box.
    ///
    /// # Errors
    ///
    /// [`RectError::NonFinite`] if any bound is NaN or infinite;
    /// [`RectError::Inverted`] if `min_x > max_x` or `min_y > max_y`.
    pub fn new(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Result<Rect, RectError> {
        if !(min_x.is_finite() && min_y.is_finite() && max_x.is_finite() && max_y.is_finite()) {
            return Err(RectError::NonFinite);
        }
        if min_x > max_x || min_y > max_y {
            return Err(RectError::Inverted);
        }
        Ok(Rect {
            min_x,
            min_y,
            max_x,
            max_y,
        })
    }

    /// The lower `x` bound.
    pub fn min_x(&self) -> f64 {
        self.min_x
    }

    /// The lower `y` bound.
    pub fn min_y(&self) -> f64 {
        self.min_y
    }

    /// The upper `x` bound.
    pub fn max_x(&self) -> f64 {
        self.max_x
    }

    /// The upper `y` bound.
    pub fn max_y(&self) -> f64 {
        self.max_y
    }

    /// Do the two boxes share at least one point?
    ///
    /// The intervals are **closed**, so boxes that touch along an edge or at a
    /// corner intersect. The test is four exact comparisons of stored values
    /// and involves no arithmetic. It is condition 3 of the crate's
    /// correctness argument.
    pub fn intersects(&self, other: &Rect) -> bool {
        self.min_x <= other.max_x
            && other.min_x <= self.max_x
            && self.min_y <= other.max_y
            && other.min_y <= self.max_y
    }

    /// Is the point inside the box or on its boundary?
    ///
    /// The box is **closed**, as in [`Self::intersects`]. A NaN coordinate gives
    /// `false`, because every comparison with NaN is false, and an infinite
    /// coordinate lies outside every finite box.
    pub fn contains_point(&self, x: f64, y: f64) -> bool {
        x >= self.min_x && x <= self.max_x && y >= self.min_y && y <= self.max_y
    }

    /// The smallest box containing both.
    ///
    /// Each bound is one of the two inputs' bounds, chosen by exact
    /// comparison, so the result is exact and contains both operands, the
    /// property that establishes condition 2 of the correctness argument for
    /// node boxes. On a tie (`-0.0` against `0.0`, which compare equal)
    /// `self`'s bound is kept, so the operand order determines the result's
    /// bit patterns.
    #[must_use]
    pub fn union(&self, other: &Rect) -> Rect {
        Rect {
            min_x: if other.min_x < self.min_x {
                other.min_x
            } else {
                self.min_x
            },
            min_y: if other.min_y < self.min_y {
                other.min_y
            } else {
                self.min_y
            },
            max_x: if other.max_x > self.max_x {
                other.max_x
            } else {
                self.max_x
            },
            max_y: if other.max_y > self.max_y {
                other.max_y
            } else {
                self.max_y
            },
        }
    }

    /// Is `other` entirely inside this box, boundary included?
    ///
    /// The test is **closed**, so a box that shares an edge with this one is
    /// contained. It is exact.
    pub fn contains_rect(&self, other: &Rect) -> bool {
        other.min_x >= self.min_x
            && other.max_x <= self.max_x
            && other.min_y >= self.min_y
            && other.max_y <= self.max_y
    }

    /// The center of the box, as a sort key only.
    ///
    /// The center is `min · 0.5 + max · 0.5`, which cannot overflow for finite
    /// bounds. It is rounded, and it loses precision for subnormal bounds.
    /// Neither affects an answer, because a center determines where an entry
    /// sits in the tree rather than whether a query returns it.
    pub(crate) fn center(&self) -> (f64, f64) {
        (
            self.min_x * 0.5 + self.max_x * 0.5,
            self.min_y * 0.5 + self.max_y * 0.5,
        )
    }
}

#[cfg(test)]
mod tests;
