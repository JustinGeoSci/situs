// src/static_index/join.rs

//! The dual-tree join, [`Index::join`] and [`Index::self_join`].
//!
//! The crate doc ("Joins") states the correctness argument. A pair of nodes
//! is descended only if their boxes meet, which by condition 2 loses no
//! pair, and a pair of entries is reported only if their own boxes meet.
//! The traversal is the synchronized descent of Brinkhoff, Kriegel and
//! Seeger ("Efficient Processing of Spatial Joins Using R-trees", SIGMOD
//! 1993), without its search-space restriction.

use alloc::vec;
use alloc::vec::Vec;

use super::Index;
use crate::Rect;

impl Index {
    /// Calls `visit(a_id, b_id)` once for every pair of entries `a` in `self`
    /// and `b` in `other` whose boxes meet (closed: boxes that touch at an
    /// edge or a corner meet).
    ///
    /// A pair is a pair of *stored* entries, so an entry stored twice, or two
    /// entries with the same id, give one report per stored pair. The order
    /// of the reports is a deterministic function of both layouts rather
    /// than a meaningful one.
    ///
    /// The join reports exactly those pairs (crate doc, "Joins"). A pair of
    /// nodes is descended only if their boxes meet, and every ancestor pair
    /// of two meeting entries meets; a pair of entries is reported only after
    /// their own boxes are tested. The output can be quadratic in the sizes,
    /// so it is never collected, and each pair goes to `visit` as it is
    /// found. The pending node pairs live on a heap stack.
    ///
    /// ```
    /// use situs::{Index, Rect};
    ///
    /// let r = |a, b, c, d| Rect::new(a, b, c, d).unwrap();
    /// let roads = Index::bulk_load(vec![(r(0.0, 0.0, 10.0, 0.0), 1), (r(0.0, 5.0, 10.0, 5.0), 2)]);
    /// let lots = Index::bulk_load(vec![(r(2.0, -1.0, 3.0, 1.0), 10), (r(4.0, 1.0, 5.0, 4.0), 11)]);
    ///
    /// let mut pairs = Vec::new();
    /// roads.join(&lots, |road, lot| pairs.push((road, lot)));
    /// assert_eq!(pairs, [(1, 10)]); // lot 11 lies between the roads
    /// ```
    pub fn join(&self, other: &Index, mut visit: impl FnMut(u32, u32)) {
        if self.nodes.is_empty() || other.nodes.is_empty() {
            return;
        }
        let mut stack: Vec<(usize, usize)> = vec![(0, 0)];
        while let Some((i, j)) = stack.pop() {
            pair_step(self, other, i, j, &mut stack, &mut visit);
        }
    }

    /// Calls `visit(x_id, y_id)` once for every unordered pair of distinct
    /// stored entries of `self` whose boxes meet (closed). An entry is never
    /// paired with itself; two stored entries with equal boxes or equal ids
    /// are distinct entries and are paired.
    ///
    /// Within a pair, `x` is the entry earlier in the index's depth-first
    /// order, the order in which a query covering every entry reports them.
    /// That order is deterministic rather than meaningful, so a caller that
    /// needs `(min, max)` normalizes the pair. The order of the pairs is
    /// likewise deterministic and carries no meaning.
    ///
    /// The traversal is that of [`Index::join`], over pairs `(N, P)` of this
    /// tree's nodes. A node is paired with itself, which pairs its children
    /// `(cᵢ, cⱼ)` with `i ≤ j` (pairs within one child are found below
    /// `(cᵢ, cᵢ)`), and a leaf paired with itself tests its entries
    /// `(eᵢ, eⱼ)` with `i < j`. Two different nodes are joined as in
    /// [`Index::join`], and their subtrees are disjoint, so every pair is
    /// found exactly once.
    ///
    /// ```
    /// use situs::{Index, Rect};
    ///
    /// let r = |a, b, c, d| Rect::new(a, b, c, d).unwrap();
    /// let index = Index::bulk_load(vec![
    ///     (r(0.0, 0.0, 2.0, 2.0), 1),
    ///     (r(2.0, 2.0, 3.0, 3.0), 2), // touches 1 at a corner
    ///     (r(9.0, 9.0, 9.0, 9.0), 3),
    /// ]);
    ///
    /// let mut pairs = Vec::new();
    /// index.self_join(|x, y| pairs.push((x.min(y), x.max(y))));
    /// assert_eq!(pairs, [(1, 2)]);
    /// ```
    pub fn self_join(&self, mut visit: impl FnMut(u32, u32)) {
        if self.nodes.is_empty() {
            return;
        }
        let mut stack: Vec<(usize, usize)> = vec![(0, 0)];
        while let Some((i, j)) = stack.pop() {
            if i == j {
                self.diagonal_step(i, &mut stack, &mut visit);
            } else {
                pair_step(self, self, i, j, &mut stack, &mut visit);
            }
        }
    }

    /// One step of `self_join` on a node paired with itself. A node's box
    /// meets itself, so there is nothing to test at this level.
    fn diagonal_step(
        &self,
        i: usize,
        stack: &mut Vec<(usize, usize)>,
        visit: &mut impl FnMut(u32, u32),
    ) {
        let node = &self.nodes[i];
        if node.leaf {
            let leaf = &self.entries[node.first..node.end];
            for (x, (rx, idx)) in leaf.iter().enumerate() {
                for (ry, idy) in &leaf[x + 1..] {
                    if rx.intersects(ry) {
                        visit(*idx, *idy);
                    }
                }
            }
        } else {
            // Pushed in reverse, so pairs are taken in (i, j) order.
            for ci in (node.first..node.end).rev() {
                for cj in (ci..node.end).rev() {
                    if nodes_meet(&self.nodes[ci].rect, &self.nodes[cj].rect) {
                        stack.push((ci, cj));
                    }
                }
            }
        }
    }
}

/// Whether two node boxes meet, by the closed test of [`Rect::intersects`].
/// It is the condition under which a node pair is descended, named so that
/// the one place a node pair is decided is visible.
fn nodes_meet(a: &Rect, b: &Rect) -> bool {
    a.intersects(b)
}

/// One step of the synchronized descent on the node pair `(i, j)`, node `i`
/// of `a` and node `j` of `b`.
///
/// The pair is dropped unless the two boxes meet. Otherwise two leaves test
/// every entry pair. A leaf and an internal node pair the leaf with each
/// child of the other whose box meets it, which is how trees of different
/// heights meet. Two internal nodes pair each child of one with each child
/// of the other whose boxes meet. Pairs are pushed in reverse, so they are
/// taken in index order.
fn pair_step(
    a: &Index,
    b: &Index,
    i: usize,
    j: usize,
    stack: &mut Vec<(usize, usize)>,
    visit: &mut impl FnMut(u32, u32),
) {
    let (na, nb) = (&a.nodes[i], &b.nodes[j]);
    if !nodes_meet(&na.rect, &nb.rect) {
        return;
    }
    match (na.leaf, nb.leaf) {
        (true, true) => {
            for (ra, ida) in &a.entries[na.first..na.end] {
                for (rb, idb) in &b.entries[nb.first..nb.end] {
                    if ra.intersects(rb) {
                        visit(*ida, *idb);
                    }
                }
            }
        }
        (true, false) => {
            for cb in (nb.first..nb.end).rev() {
                if nodes_meet(&na.rect, &b.nodes[cb].rect) {
                    stack.push((i, cb));
                }
            }
        }
        (false, true) => {
            for ca in (na.first..na.end).rev() {
                if nodes_meet(&a.nodes[ca].rect, &nb.rect) {
                    stack.push((ca, j));
                }
            }
        }
        (false, false) => {
            for ca in (na.first..na.end).rev() {
                for cb in (nb.first..nb.end).rev() {
                    if nodes_meet(&a.nodes[ca].rect, &b.nodes[cb].rect) {
                        stack.push((ca, cb));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
