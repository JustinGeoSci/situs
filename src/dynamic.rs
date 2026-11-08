// src/dynamic.rs

//! The dynamic R\*-tree, [`DynamicIndex`], which accepts insertions and
//! removals.
//!
//! The static [`Index`](crate::Index) is read-only. This tree changes with
//! its contents and retains the same guarantee: a query reports exactly the
//! entries whose boxes meet the window, and a nearest-neighbor search runs
//! the static index's own search and proof.
//!
//! # Shape and contents
//!
//! **The R\* heuristics determine the tree's shape and nothing else.** Area,
//! margin, overlap and center distance (Beckmann, Kriegel, Schneider &
//! Seeger, "The R\*-tree: An Efficient and Robust Access Method for Points
//! and Rectangles", SIGMOD 1990) are computed in `f64`, while the
//! invariants on which the absence of false negatives rests are maintained
//! by exact `min` and `max` alone. The static index makes the same split,
//! with approximate sort keys and exact contents.
//!
//! The tests check three invariants after every insertion and every
//! removal:
//!
//! - **Tight, exact boxes:** Every node's box is the [`Rect::union`] of its
//!   children's boxes (its entries' boxes, for a leaf). The box is tight
//!   rather than merely containing, so [`DynamicIndex::bounds`] is exact.
//!   `union` selects stored values by exact comparison, so no arithmetic
//!   touches a stored box.
//! - **Balance:** Every leaf is at the same depth.
//! - **Contents:** The stored entries are exactly the multiset inserted.
//!
//! These are the static index's three conditions over a different storage,
//! and the query is the same search. No heuristic can break them, because
//! every structural change recomputes the boxes it touches by exact union.
//! A degenerate heuristic value, such as an area that overflows to infinity
//! or a NaN from `∞ − ∞`, can only select a worse shape. Every heuristic
//! comparison runs through one total order that places NaN past every
//! number, and no heuristic value is ever stored as a box.
//!
//! # Storage
//!
//! Nodes live in one arena (`Vec`) with a last-in, first-out free list of
//! slot indices. Each node holds its children inline, with room for one more
//! than the capacity, so an overflowing node holds its extra child while the
//! overflow is treated. No node allocates on its own, slots are reused in
//! last-in, first-out order, and every heuristic tie breaks by a fixed rule.
//! The same operations on the same starting tree therefore produce the same
//! arena, bit for bit.

use alloc::vec::Vec;

use crate::Rect;

#[cfg(any(test, fuzzing))]
mod check;
mod insert;
pub(crate) mod node;
mod query;
mod remove;

pub(crate) use node::{DNode, SLOTS};

/// A dynamic R\*-tree over `(Rect, u32)` entries: bulk-loaded, grown by
/// insertion, shrunk by exact removal, queried exactly.
///
/// The `u32` is the caller's identifier, as in the static
/// [`Index`](crate::Index); it need not be unique, and an entry stored
/// twice is two entries.
///
/// ```
/// use situs::{DynamicIndex, Rect};
///
/// let r = |a, b, c, d| Rect::new(a, b, c, d).unwrap();
/// let mut index = DynamicIndex::new();
/// index.insert(r(0.0, 0.0, 1.0, 1.0), 7);
/// index.insert(r(5.0, 5.0, 6.0, 6.0), 9);
///
/// let mut hits = Vec::new();
/// index.query(&r(1.0, 1.0, 2.0, 2.0), |id| hits.push(id));
/// assert_eq!(hits, [7]); // touching at a corner counts
/// assert_eq!(index.bounds(), Some(r(0.0, 0.0, 6.0, 6.0)));
/// ```
#[derive(Clone, Debug)]
pub struct DynamicIndex {
    /// The arena. A slot holds a live node or sits on the free list.
    pub(crate) arena: Vec<DNode>,
    /// Free slot indices, reused last in, first out.
    pub(crate) free: Vec<usize>,
    /// The root slot, or `None` for an empty tree.
    pub(crate) root: Option<usize>,
    /// The number of stored entries.
    pub(crate) len: usize,
}

impl DynamicIndex {
    /// Children per node and entries per leaf, the `M` of the R\*-tree.
    pub const NODE_CAPACITY: usize = 16;

    /// The least children in a non-root node, the `m` of the R\*-tree: 40%
    /// of the capacity, as the paper recommends.
    pub const MIN_FILL: usize = 6;

    /// How many children forced reinsertion removes from an overflowing
    /// node: 30% of the capacity, the paper's `p`.
    pub(crate) const REINSERT_COUNT: usize = 5;

    /// An empty tree.
    #[must_use]
    pub fn new() -> DynamicIndex {
        DynamicIndex {
            arena: Vec::new(),
            free: Vec::new(),
            root: None,
            len: 0,
        }
    }

    /// The number of stored entries.
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` if the tree holds no entry.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The box containing every entry, or `None` for an empty tree.
    ///
    /// The root's box, which is **exact and tight**. It is the exact union of
    /// every stored entry's box, recomputed from stored values at every
    /// change, never widened and never left stale.
    pub fn bounds(&self) -> Option<Rect> {
        self.root.map(|slot| self.arena[slot].rect)
    }

    /// Takes a slot from the free list, or grows the arena.
    pub(crate) fn allocate(&mut self, node: DNode) -> usize {
        if let Some(slot) = self.free.pop() {
            self.arena[slot] = node;
            slot
        } else {
            self.arena.push(node);
            self.arena.len() - 1
        }
    }

    /// Returns a slot to the free list. The slot's contents stay behind as
    /// inert data until the slot is reused. Insertion never frees a slot;
    /// removal frees the slots of detached nodes and of a replaced root.
    pub(crate) fn release(&mut self, slot: usize) {
        self.free.push(slot);
    }
}

impl Default for DynamicIndex {
    fn default() -> DynamicIndex {
        DynamicIndex::new()
    }
}

#[cfg(test)]
mod tests;
