// src/static_index.rs

//! This module implements the static index [`Index`] and its bulk load by
//! STR packing.
//!
//! The box query is in the `search` submodule and the joins in `join`. The
//! crate documentation states the correctness argument and the packing
//! method that the code here follows.

use alloc::vec::Vec;

use crate::Rect;

mod join;
mod search;

// ===========================================================================
// Index
// ===========================================================================

/// One node of the tree: its box and the index range of its children.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Node {
    pub(crate) rect: Rect,
    /// `[first, end)` into `Index::nodes` for an internal node, or into
    /// `Index::entries` for a leaf.
    pub(crate) first: usize,
    pub(crate) end: usize,
    pub(crate) leaf: bool,
}

/// A static R-tree over `(Rect, u32)` entries, bulk-loaded by STR packing.
///
/// The index is built once by [`Index::bulk_load`] and then queried, and it
/// cannot be changed afterward. The `u32` is the caller's identifier for the
/// entry, returned by [`Index::query`], and it need not be unique.
///
/// ```
/// use situs::{Index, Rect};
///
/// let r = |a, b, c, d| Rect::new(a, b, c, d).unwrap();
/// let index = Index::bulk_load(vec![(r(0.0, 0.0, 1.0, 1.0), 7), (r(5.0, 5.0, 6.0, 6.0), 9)]);
///
/// let mut hits = Vec::new();
/// index.query(&r(1.0, 1.0, 2.0, 2.0), |id| hits.push(id));
/// assert_eq!(hits, [7]); // touching at a corner counts
/// ```
#[derive(Clone, Debug)]
pub struct Index {
    /// Level order, root first; empty for an empty index.
    pub(crate) nodes: Vec<Node>,
    /// The entries, reordered into leaf order.
    pub(crate) entries: Vec<(Rect, u32)>,
    /// The number of node levels, root to leaves; 0 for an empty index. It
    /// bounds the search stack (`search` module) and is at most
    /// `search::MAX_DEPTH`.
    depth: usize,
}

impl Index {
    /// Entries per leaf and children per node, the `M` of STR packing.
    ///
    /// A capacity of 16 keeps a node's boxes within a few cache lines and
    /// the tree shallow, so a million entries make five levels.
    pub const NODE_CAPACITY: usize = 16;

    /// The heap bytes a built index of `n` entries holds, so that a caller
    /// can size an index before building it.
    ///
    /// Exactly `n · size_of::<(Rect, u32)>() + nodes(n) · size_of::<Node>()`,
    /// where `nodes(n)` is the node count STR packing produces: `⌈n / M⌉`
    /// leaves, then `⌈· / M⌉` nodes per level above until one node remains,
    /// and none for `n = 0`; `Node` is the index's private node record (a
    /// box and a child range). The count is a function of `n` alone, because
    /// every level is the level below cut into chunks of `M`.
    ///
    /// The figure omits four things:
    ///
    /// - Spare capacity in the entry vector. The figure assumes the vector
    ///   passed to [`Index::bulk_load`] has capacity equal to its length.
    ///   The index keeps that vector, so any spare capacity is held as well.
    /// - Spare capacity in the node vector, of which there is none. It is
    ///   allocated with exactly the node count.
    /// - The transient allocations made while building, one vector per
    ///   level, freed before `bulk_load` returns. They total at most the
    ///   node count again.
    /// - The inline size of the `Index` value itself
    ///   (`size_of::<Index>()`), which is not heap memory.
    ///
    /// The result saturates. For an `n` so large that the size overflows
    /// `usize`, it is `usize::MAX`, and no such index can be built.
    ///
    /// ```
    /// use situs::{Index, Rect};
    ///
    /// assert_eq!(Index::memory_size_for(0), 0);
    /// assert!(Index::memory_size_for(1_000) > 1_000 * core::mem::size_of::<(Rect, u32)>());
    /// assert_eq!(Index::memory_size_for(usize::MAX), usize::MAX);
    /// ```
    #[must_use]
    pub fn memory_size_for(n: usize) -> usize {
        let entries = n.checked_mul(core::mem::size_of::<(Rect, u32)>());
        let nodes = node_count(n).checked_mul(core::mem::size_of::<Node>());
        match (entries, nodes) {
            (Some(e), Some(m)) => e.saturating_add(m),
            _ => usize::MAX,
        }
    }

    /// Builds the index over `entries` by STR packing (see the crate doc).
    ///
    /// The build costs `O(n log n)`, from the sorts. The entry vector is
    /// reordered into leaf order and kept rather than copied. The build never
    /// panics, because every `Rect` is finite, so every sort key is finite
    /// and `total_cmp` orders all of them.
    #[must_use]
    pub fn bulk_load(mut entries: Vec<(Rect, u32)>) -> Index {
        if entries.is_empty() {
            return Index {
                nodes: Vec::new(),
                entries,
                depth: 0,
            };
        }

        // The leaves are the entries in STR order, cut into chunks of M.
        str_order(&mut entries, |e| e.0, |e| u64::from(e.1));
        let mut level: Vec<Node> = entries
            .chunks(Self::NODE_CAPACITY)
            .scan(0usize, |first, chunk| {
                let rect = union_all(chunk, |e| e.0);
                let node = Node {
                    rect,
                    first: *first,
                    end: *first + chunk.len(),
                    leaf: true,
                };
                *first += chunk.len();
                Some(node)
            })
            .collect();

        // The upper levels are built bottom up, each over the level below. A
        // node has no id, so equal centers fall back to the previous order,
        // which is itself deterministic.
        let mut levels: Vec<Vec<Node>> = Vec::new();
        while level.len() > 1 {
            let mut next: Vec<Node> = Vec::with_capacity(level.len().div_ceil(Self::NODE_CAPACITY));
            str_order(&mut level, |n| n.rect, |_| 0);
            let mut first = 0usize;
            for chunk in level.chunks(Self::NODE_CAPACITY) {
                next.push(Node {
                    rect: union_all(chunk, |n| n.rect),
                    first,
                    end: first + chunk.len(),
                    leaf: false,
                });
                first += chunk.len();
            }
            levels.push(level);
            level = next;
        }
        levels.push(level);
        let depth = levels.len();
        debug_assert!(
            depth <= search::MAX_DEPTH,
            "STR packing built {depth} levels, above the proven bound"
        );

        // The levels are flattened root first. A node in level `i` (counting
        // from the leaves) indexes level `i - 1`, which lands after it, so its
        // child range is offset by the size of everything above and including
        // level `i`.
        let mut nodes: Vec<Node> = Vec::with_capacity(levels.iter().map(Vec::len).sum());
        let mut offset = 0usize;
        for lvl in levels.iter().rev() {
            offset += lvl.len();
            for n in lvl {
                nodes.push(if n.leaf {
                    *n
                } else {
                    Node {
                        first: n.first + offset,
                        end: n.end + offset,
                        ..*n
                    }
                });
            }
        }

        Index {
            nodes,
            entries,
            depth,
        }
    }

    /// The number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// `true` if the index holds no entry.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The box containing every entry, or `None` for an empty index.
    ///
    /// It is the root's box, an exact union of every entry's box.
    pub fn bounds(&self) -> Option<Rect> {
        self.nodes.first().map(|n| n.rect)
    }
}

// ===========================================================================
// STR packing
// ===========================================================================

/// Reorders `items` into STR leaf order (crate doc, "Bulk loading").
///
/// `rect` reads an item's box, and `tie` breaks equal centers (the entry id
/// at the leaf level, a constant above it). Items that still compare equal
/// keep their previous relative order. Each pass sorts by `(center, tie,
/// previous position)`, a total order, so the unstable sort it uses
/// produces exactly the order a stable sort by `(center, tie)` would. The
/// centers are computed once per pass, as integer keys in `total_cmp`'s
/// order, so every comparison is an integer comparison.
pub(crate) fn str_order<T: Copy>(
    items: &mut [T],
    rect: impl Fn(&T) -> Rect + Copy,
    tie: impl Fn(&T) -> u64 + Copy,
) {
    let n = items.len();
    let leaves = n.div_ceil(Index::NODE_CAPACITY);
    let slices = ceil_sqrt(leaves);
    let slice_len = slices * Index::NODE_CAPACITY;

    sort_by_center(items, |t| rect(t).center().0, tie);
    for slice in items.chunks_mut(slice_len) {
        sort_by_center(slice, |t| rect(t).center().1, tie);
    }
}

/// The bits of `x`, mapped so that unsigned order is `f64::total_cmp`'s
/// order. A negative value (sign bit set) has every bit flipped, and any
/// other value only its sign bit.
fn total_key(x: f64) -> u64 {
    let b = x.to_bits();
    if b >> 63 == 1 { !b } else { b | (1 << 63) }
}

/// Sorts `items` by `key`, then `tie`, then present position, the order a
/// stable sort by `(key, tie)` gives, reached by an unstable sort over a
/// total order.
fn sort_by_center<T: Copy>(items: &mut [T], key: impl Fn(&T) -> f64, tie: impl Fn(&T) -> u64) {
    let mut order: Vec<(u64, u64, usize)> = items
        .iter()
        .enumerate()
        .map(|(i, t)| (total_key(key(t)), tie(t), i))
        .collect();
    order.sort_unstable();
    let sorted: Vec<T> = order.iter().map(|&(_, _, i)| items[i]).collect();
    items.copy_from_slice(&sorted);
}

/// The number of nodes STR packing builds over `n` entries: `⌈n / M⌉`
/// leaves, then `⌈· / M⌉` per level until one node remains; 0 for `n = 0`.
///
/// The count cannot overflow, since the sum is below `n / (M − 1) + depth`,
/// far under `usize::MAX` for any `n`.
fn node_count(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let mut level = n.div_ceil(Index::NODE_CAPACITY);
    let mut total = level;
    while level > 1 {
        level = level.div_ceil(Index::NODE_CAPACITY);
        total += level;
    }
    total
}

/// `⌈√n⌉`, in integers. `isqrt` is exact, and one is added when it falls
/// short. (`f64::sqrt` is not available in `core`.)
fn ceil_sqrt(n: usize) -> usize {
    let s = n.isqrt();
    if s * s < n { s + 1 } else { s }
}

/// The union of the boxes of a non-empty chunk, exact by [`Rect::union`].
///
/// `chunks` never yields an empty slice, so `chunk[0]` exists. Were that
/// ever false, the index would panic here rather than build a node with an
/// invented box, a failure that could lose entries silently.
fn union_all<T>(chunk: &[T], rect: impl Fn(&T) -> Rect) -> Rect {
    chunk[1..]
        .iter()
        .fold(rect(&chunk[0]), |acc, item| acc.union(&rect(item)))
}

#[cfg(test)]
pub(crate) mod tests;
