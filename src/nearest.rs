// src/nearest.rs

//! Nearest-neighbor candidates and distance browsing by certified bounds.
//!
//! `situs` holds no exact distance arithmetic, so it never ranks entries by
//! a computed distance. For a query point and each entry's box it computes
//! two `f64` values, `lower_sq ≤ d² ≤ upper_sq`, that bracket the exact
//! squared distance (the `bounds` module carries the proof). The candidate
//! sets and orderings it returns are stated in terms of these bounds rather
//! than a distance, so the error of a bound can only add a candidate and
//! never remove one.
//!
//! # The search
//!
//! Both operations run a best-first search over a min-heap keyed by the
//! lower bound, holding nodes (keyed by the bound of the node's box) and
//! entries. A fixed rule breaks ties (the bound, then nodes before entries,
//! then id, then internal position), so the pop order is deterministic.
//! Nodes come first at a tied key so that every entry tied at that key is
//! expanded into the heap before any of them is yielded. The tie order
//! among entries (id, then position) therefore holds globally rather than
//! only among the entries discovered so far.
//!
//! The search is complete because the bound is monotone and a node's box
//! contains every descendant's box (crate doc, condition 2). A node's key
//! is therefore at most every descendant entry's `lower_sq`, and nothing
//! below an unpopped key can beat a popped one.
//!
//! [`Index::nearest_candidates`] pops in key order, keeps the `k` smallest
//! `upper_sq` seen among popped entries (their running k-th smallest is
//! `τ*`, or `+∞` while fewer than `k` entries have been popped), and stops
//! when the next key exceeds `τ*`. The result is exactly
//! `{ e : lower_sq(e) ≤ τ }`, with `τ` the k-th smallest `upper_sq` over
//! **all** entries, independent of the tree's shape. The argument runs in four
//! steps:
//!
//! - `τ* ≥ τ` always, since `τ*` is the k-th smallest over a subset, and it
//!   never increases;
//! - every entry with `upper_sq ≤ τ*` is popped before the search stops,
//!   because its ancestors' keys are at most its `lower_sq ≤ upper_sq ≤ τ*`
//!   (monotonicity), so no ancestor is ever the key that stops the search;
//! - hence at stopping time the k entries with the globally smallest
//!   `upper_sq` have been popped, and `τ* = τ`;
//! - and every entry with `lower_sq ≤ τ` has been popped, by the same
//!   ancestor argument; popped entries with `lower_sq > τ` are filtered
//!   out.
//!
//! [`Index::browse`] runs the same heap without the cutoff and yields each
//! entry as it is popped, in nondecreasing `lower_sq`. This is distance
//! browsing in the sense of Hjaltason and Samet, "Distance Browsing in
//! Spatial Databases", ACM TODS 24(2), 1999.

use alloc::collections::BinaryHeap;
use alloc::vec::Vec;
use core::cmp::{Ordering, Reverse};

use crate::Rect;
use crate::dynamic::node::Child;
use crate::dynamic::{DynamicIndex, SLOTS};
use crate::static_index::Index;

pub(crate) mod bounds;

pub(crate) use bounds::bounds_sq;

// ===========================================================================
// Public types
// ===========================================================================

/// Certified bounds on the squared Euclidean distance from the query point
/// to one entry's box: `lower_sq <= d² <= upper_sq`, with `d²` the exact
/// value.
///
/// A neighbor carries bounds rather than a distance, so two entries' true
/// distances are ordered only where their intervals do not overlap. A
/// caller that must rank exactly decides with its own arithmetic among the
/// candidates returned.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Neighbor {
    /// The entry's id, as stored; not necessarily unique.
    pub id: u32,
    /// A certified lower bound on the exact squared distance. Finite and
    /// non-negative.
    pub lower_sq: f64,
    /// A certified upper bound on the exact squared distance. It is
    /// positive, never below `2⁻¹⁰¹⁹` even at distance zero, because it
    /// bounds the distance rather than equals it, and it is `+∞` where the
    /// squared distance can overflow `f64`.
    pub upper_sq: f64,
}

/// Why a nearest-neighbor query was refused.
///
/// The enum is non-exhaustive. A match on it needs a wildcard arm, so a
/// later variant is not a breaking change.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueryError {
    /// The query point was NaN or infinite. Every comparison with NaN is
    /// false, so such a point would silently match nothing, a false
    /// negative, and it is refused instead.
    NonFinite,
}

impl core::fmt::Display for QueryError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonFinite => f.write_str("query point is NaN or infinite"),
        }
    }
}

impl core::error::Error for QueryError {}

// ===========================================================================
// The frontier
// ===========================================================================

/// What the heap holds at one key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum ItemKind {
    /// An internal or leaf node, still to be expanded; before an entry at
    /// the same key (module doc).
    Node,
    /// An entry, ready to be yielded.
    Entry,
}

/// One frontier element: an entry or a node, keyed for the min-heap.
#[derive(Clone, Copy, Debug)]
struct Item {
    /// The key, the certified lower bound of the entry's or node's box.
    lower: f64,
    /// The entry's upper bound; unused for a node.
    upper: f64,
    kind: ItemKind,
    /// The entry's id; 0 for a node.
    id: u32,
    /// The entry's position in the entry array, or the node's index:
    /// the last tie-breaker, unique per kind.
    pos: usize,
}

impl Item {
    /// The entry this item stands for, as the public result type.
    fn neighbor(&self) -> Neighbor {
        Neighbor {
            id: self.id,
            lower_sq: self.lower,
            upper_sq: self.upper,
        }
    }
}

impl Ord for Item {
    fn cmp(&self, other: &Self) -> Ordering {
        // `lower` is never NaN (bounds module doc), so total_cmp is the
        // numeric order; it also makes ties bit-deterministic.
        self.lower
            .total_cmp(&other.lower)
            .then_with(|| self.kind.cmp(&other.kind))
            .then_with(|| self.id.cmp(&other.id))
            .then_with(|| self.pos.cmp(&other.pos))
    }
}

impl PartialOrd for Item {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Item {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Item {}

/// An `f64` ordered by `total_cmp`, for the cutoff's max-heap.
#[derive(Clone, Copy, Debug)]
struct TotalF64(f64);

impl Ord for TotalF64 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl PartialOrd for TotalF64 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for TotalF64 {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for TotalF64 {}

/// The k smallest `upper_sq` seen so far. `value()` is their maximum (τ*),
/// or `+∞` while fewer than k have been offered.
struct Cutoff {
    k: usize,
    worst: BinaryHeap<TotalF64>,
}

impl Cutoff {
    fn new(k: usize) -> Cutoff {
        Cutoff {
            k,
            worst: BinaryHeap::with_capacity(k.min(1024) + 1),
        }
    }

    fn offer(&mut self, upper: f64) {
        self.worst.push(TotalF64(upper));
        if self.worst.len() > self.k {
            self.worst.pop();
        }
    }

    fn value(&self) -> f64 {
        if self.worst.len() == self.k {
            self.worst.peek().map_or(f64::INFINITY, |t| t.0)
        } else {
            f64::INFINITY
        }
    }
}

/// The crate-private view of a tree that the search needs: a root, each
/// node's box and children, and leaf entries. Both index types provide it,
/// so one search and one proof serve both. The proof needs only that a
/// node's box contains every descendant's box (crate doc, condition 2),
/// which both trees keep exactly.
#[derive(Clone, Copy)]
enum Tree<'a> {
    Static(&'a Index),
    Dynamic(&'a DynamicIndex),
}

impl Tree<'_> {
    /// The root's box and position, or `None` for an empty tree.
    fn root(self) -> Option<(Rect, usize)> {
        match self {
            Tree::Static(index) => index.nodes.first().map(|root| (root.rect, 0)),
            Tree::Dynamic(index) => index.root.map(|slot| (index.arena[slot].rect, slot)),
        }
    }

    /// Pushes node `i`'s children (the entries of a leaf, the nodes
    /// otherwise) onto the frontier, each keyed by its own certified lower
    /// bound.
    ///
    /// Positions, the last tie-breaker, are unique per kind within a tree.
    /// In the static tree a position is the entry index or node index. In
    /// the dynamic tree it is the node slot, or for an entry its leaf's slot
    /// times the slot count plus its place in the leaf.
    fn push_children(self, i: usize, x: f64, y: f64, heap: &mut BinaryHeap<Reverse<Item>>) {
        match self {
            Tree::Static(index) => {
                let node = &index.nodes[i];
                if node.leaf {
                    for (pos, (rect, id)) in index.entries[node.first..node.end].iter().enumerate()
                    {
                        push_entry(heap, x, y, rect, *id, node.first + pos);
                    }
                } else {
                    for child in node.first..node.end {
                        push_node(heap, x, y, &index.nodes[child].rect, child);
                    }
                }
            }
            Tree::Dynamic(index) => {
                for (at, child) in index.arena[i].children().enumerate() {
                    match *child {
                        Child::Entry(rect, id) => push_entry(heap, x, y, &rect, id, i * SLOTS + at),
                        Child::Node(slot) => push_node(heap, x, y, &index.arena[slot].rect, slot),
                    }
                }
            }
        }
    }

    /// The box of the entry at search position `pos` (as `push_children`
    /// assigns positions).
    #[cfg(feature = "exact-nearest")]
    fn entry_rect(self, pos: usize) -> Rect {
        match self {
            Tree::Static(index) => index.entries[pos].0,
            Tree::Dynamic(index) => match index.arena[pos / SLOTS].child(pos % SLOTS) {
                Child::Entry(rect, _) => rect,
                Child::Node(_) => unreachable!("an entry position names an entry"),
            },
        }
    }

    /// The frontier seeded with the root, after validating the point.
    fn frontier(self, x: f64, y: f64) -> Result<BinaryHeap<Reverse<Item>>, QueryError> {
        if !x.is_finite() || !y.is_finite() {
            return Err(QueryError::NonFinite);
        }
        let mut heap = BinaryHeap::new();
        if let Some((rect, pos)) = self.root() {
            push_node(&mut heap, x, y, &rect, pos);
        }
        Ok(heap)
    }
}

fn push_entry(
    heap: &mut BinaryHeap<Reverse<Item>>,
    x: f64,
    y: f64,
    rect: &Rect,
    id: u32,
    pos: usize,
) {
    let (lower, upper) = bounds_sq(x, y, rect);
    heap.push(Reverse(Item {
        lower,
        upper,
        kind: ItemKind::Entry,
        id,
        pos,
    }));
}

fn push_node(heap: &mut BinaryHeap<Reverse<Item>>, x: f64, y: f64, rect: &Rect, pos: usize) {
    let (lower, _) = bounds_sq(x, y, rect);
    heap.push(Reverse(Item {
        lower,
        upper: f64::INFINITY,
        kind: ItemKind::Node,
        id: 0,
        pos,
    }));
}

// ===========================================================================
// nearest_candidates
// ===========================================================================

impl Index {
    /// The candidates for the `k` entries nearest to `(x, y)`: exactly the
    /// entries `e` with `lower_sq(e) <= τ`, where `τ` is the k-th smallest
    /// `upper_sq` over all entries (`+∞` when `k >= len`).
    ///
    /// **The result always contains the `k` entries truly nearest to the
    /// point, and every entry tied with the k-th.** Let `d²ₖ` be the k-th
    /// smallest exact squared distance. The k entries with the smallest
    /// `upper_sq` each have `d² ≤ upper_sq ≤ τ`, so at least k entries
    /// have `d² ≤ τ`, which means `d²ₖ ≤ τ`. Any entry among the true k
    /// nearest, or tied with the k-th, has `lower_sq ≤ d² ≤ d²ₖ ≤ τ`, so
    /// it is in the result.
    ///
    /// The result has exactly `k` entries unless distances near the k-th
    /// lie within a few ulps of each other, in which case it holds the
    /// extras rather than guessing among them. It depends only on the
    /// entries and the point rather than on the tree's shape. An entry
    /// stored twice is a candidate twice, and `k = 0` gives an empty
    /// result.
    ///
    /// The result is ordered by `(lower_sq, id)`, then by internal
    /// position. The order is deterministic but carries no meaning.
    ///
    /// ```
    /// use situs::{Index, Rect};
    ///
    /// let r = |a, b, c, d| Rect::new(a, b, c, d).unwrap();
    /// let index = Index::bulk_load(vec![(r(0.0, 0.0, 1.0, 1.0), 7), (r(5.0, 5.0, 6.0, 6.0), 9)]);
    ///
    /// let near = index.nearest_candidates(2.0, 2.0, 1).unwrap();
    /// assert_eq!(near.len(), 1);
    /// assert_eq!(near[0].id, 7);
    /// // d² = 2: the bounds bracket it.
    /// assert!(near[0].lower_sq <= 2.0 && 2.0 <= near[0].upper_sq);
    /// ```
    ///
    /// # Errors
    ///
    /// [`QueryError::NonFinite`] if `x` or `y` is NaN or infinite, checked
    /// before anything else. An empty index gives `Ok` of an empty vector.
    pub fn nearest_candidates(
        &self,
        x: f64,
        y: f64,
        k: usize,
    ) -> Result<Vec<Neighbor>, QueryError> {
        candidates(Tree::Static(self), x, y, k)
    }

    /// The entries in nondecreasing `lower_sq` (ties by id, then internal
    /// position), each exactly once per stored entry.
    ///
    /// The iterator holds one contract. Once a [`Neighbor`] with
    /// `lower_sq = ℓ` has been yielded, every entry not yet yielded has
    /// true `d² >= ℓ`. A caller refining candidates by its own exact
    /// distance stops once its best answer is at or below the next
    /// `lower_sq`.
    ///
    /// ```
    /// use situs::{Index, Rect};
    ///
    /// let r = |a, b, c, d| Rect::new(a, b, c, d).unwrap();
    /// let index = Index::bulk_load(vec![(r(0.0, 0.0, 1.0, 1.0), 7), (r(5.0, 5.0, 6.0, 6.0), 9)]);
    ///
    /// let order: Vec<u32> = index.browse(2.0, 2.0).unwrap().map(|n| n.id).collect();
    /// assert_eq!(order, [7, 9]);
    /// ```
    ///
    /// # Errors
    ///
    /// [`QueryError::NonFinite`] if `x` or `y` is NaN or infinite. An empty
    /// index gives an iterator that yields nothing.
    pub fn browse(&self, x: f64, y: f64) -> Result<Browse<'_>, QueryError> {
        browse(Tree::Static(self), x, y)
    }
}

// ===========================================================================
// The same search on the dynamic tree
// ===========================================================================

impl DynamicIndex {
    /// The candidates for the `k` entries nearest to `(x, y)`: exactly the
    /// entries `e` with `lower_sq(e) <= τ`, where `τ` is the k-th smallest
    /// `upper_sq` over all stored entries (`+∞` when `k >= len`).
    ///
    /// This is the search of [`Index::nearest_candidates`], run by the
    /// same code, and the guarantee and its argument are stated there. The
    /// result always contains the `k` entries truly nearest and every
    /// entry tied with the k-th. The argument needs only that every node's
    /// box contains its descendants' boxes, and this tree keeps every node
    /// box as the exact union of its children's. The set depends only on
    /// the stored entries and the point, so it equals the static index's
    /// set for the same entries. The order of entries with equal
    /// `(lower_sq, id)` follows the internal position, which differs
    /// between the two trees.
    ///
    /// ```
    /// use situs::{DynamicIndex, Rect};
    ///
    /// let r = |a, b, c, d| Rect::new(a, b, c, d).unwrap();
    /// let mut index = DynamicIndex::new();
    /// index.insert(r(0.0, 0.0, 1.0, 1.0), 7);
    /// index.insert(r(5.0, 5.0, 6.0, 6.0), 9);
    /// assert!(index.remove(&r(0.0, 0.0, 1.0, 1.0), 7));
    ///
    /// let near = index.nearest_candidates(2.0, 2.0, 1).unwrap();
    /// assert_eq!(near.len(), 1);
    /// assert_eq!(near[0].id, 9); // the removed entry is gone
    /// ```
    ///
    /// # Errors
    ///
    /// [`QueryError::NonFinite`] if `x` or `y` is NaN or infinite. An empty
    /// tree gives `Ok` of an empty vector.
    pub fn nearest_candidates(
        &self,
        x: f64,
        y: f64,
        k: usize,
    ) -> Result<Vec<Neighbor>, QueryError> {
        candidates(Tree::Dynamic(self), x, y, k)
    }

    /// The stored entries in nondecreasing `lower_sq` (ties by id, then
    /// internal position), each exactly once per stored entry.
    ///
    /// This is the iterator of [`Index::browse`], and it holds the same
    /// contract. Once a [`Neighbor`] with `lower_sq = ℓ` has been yielded,
    /// every entry not yet yielded has true `d² >= ℓ`. The iterator
    /// borrows the tree, so the tree cannot change while it runs.
    ///
    /// # Errors
    ///
    /// [`QueryError::NonFinite`] if `x` or `y` is NaN or infinite. An empty
    /// tree gives an iterator that yields nothing.
    pub fn browse(&self, x: f64, y: f64) -> Result<Browse<'_>, QueryError> {
        browse(Tree::Dynamic(self), x, y)
    }
}

/// The iterator of [`Index::browse`] and [`DynamicIndex::browse`], which
/// yields entries by nondecreasing `lower_sq`.
pub struct Browse<'a> {
    tree: Tree<'a>,
    x: f64,
    y: f64,
    heap: BinaryHeap<Reverse<Item>>,
}

impl Iterator for Browse<'_> {
    type Item = Neighbor;

    fn next(&mut self) -> Option<Neighbor> {
        while let Some(Reverse(top)) = self.heap.pop() {
            match top.kind {
                ItemKind::Entry => {
                    return Some(Neighbor {
                        id: top.id,
                        lower_sq: top.lower,
                        upper_sq: top.upper,
                    });
                }
                ItemKind::Node => self
                    .tree
                    .push_children(top.pos, self.x, self.y, &mut self.heap),
            }
        }
        None
    }
}

// ===========================================================================
// The search, shared by both trees
// ===========================================================================

/// The candidate search of [`Index::nearest_candidates`], over either tree.
fn candidates(tree: Tree<'_>, x: f64, y: f64, k: usize) -> Result<Vec<Neighbor>, QueryError> {
    Ok(candidate_items(tree, x, y, k)?
        .into_iter()
        .map(|e| e.neighbor())
        .collect())
}

/// The candidates as frontier items `(lower, id, position)`, in the order
/// `candidates` returns them. The position identifies the entry inside its
/// tree, so the exact ranking can read the entry's box back.
fn candidate_items(tree: Tree<'_>, x: f64, y: f64, k: usize) -> Result<Vec<Item>, QueryError> {
    let mut heap = tree.frontier(x, y)?;
    if k == 0 {
        return Ok(Vec::new());
    }
    let mut cutoff = Cutoff::new(k);
    let mut popped: Vec<Item> = Vec::new();
    while let Some(Reverse(top)) = heap.peek().copied() {
        if top.lower > cutoff.value() {
            break;
        }
        heap.pop();
        match top.kind {
            ItemKind::Entry => {
                cutoff.offer(top.upper);
                popped.push(top);
            }
            ItemKind::Node => tree.push_children(top.pos, x, y, &mut heap),
        }
    }
    let tau = cutoff.value();
    popped.retain(|e| e.lower <= tau);
    popped.sort_by(|a, b| {
        a.lower
            .total_cmp(&b.lower)
            .then_with(|| a.id.cmp(&b.id))
            .then_with(|| a.pos.cmp(&b.pos))
    });
    Ok(popped)
}

/// The iterator of [`Index::browse`], over either tree.
fn browse(tree: Tree<'_>, x: f64, y: f64) -> Result<Browse<'_>, QueryError> {
    Ok(Browse {
        tree,
        x,
        y,
        heap: tree.frontier(x, y)?,
    })
}

#[cfg(feature = "exact-nearest")]
mod exact;

#[cfg(test)]
mod tests;
