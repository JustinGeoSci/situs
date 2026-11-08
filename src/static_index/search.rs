// src/static_index/search.rs

//! The box query, [`Index::query`] and [`Index::try_query`].
//!
//! The crate doc ("Storage and search") states the arguments this code rests
//! on, namely the closed overlap test, the containment shortcut (and why its
//! subtree is walked rather than taken as one entry range), the stack bound
//! and the visit order. Nothing in this module allocates.

use core::convert::Infallible;
use core::ops::ControlFlow;

use super::Index;
use crate::Rect;

/// The most node levels STR packing can build over at most `usize::MAX`
/// entries, `⌈usize::BITS / 4⌉ + 1`, because `M = 2⁴`.
pub(super) const MAX_DEPTH: usize = (usize::BITS as usize).div_ceil(4) + 1;

/// The fixed capacity of the search stack, `(M − 1) · MAX_DEPTH + 1`.
pub(super) const STACK_CAPACITY: usize = (Index::NODE_CAPACITY - 1) * MAX_DEPTH + 1;

// ===========================================================================
// The fixed stack
// ===========================================================================

/// A stack of node indices in a fixed array, on the call stack.
pub(super) struct Stack {
    items: [usize; STACK_CAPACITY],
    len: usize,
    /// The bound for the tree being searched, `(M − 1) · depth + 1`, checked
    /// on every push in debug builds. `STACK_CAPACITY` is this bound at
    /// `MAX_DEPTH`.
    limit: usize,
    /// The largest `len` reached, for the tests of the bound.
    #[cfg(test)]
    high: usize,
}

impl Stack {
    /// An empty stack for searching a tree of `depth` node levels.
    pub(super) const fn new(depth: usize) -> Stack {
        Stack {
            items: [0; STACK_CAPACITY],
            len: 0,
            limit: (Index::NODE_CAPACITY - 1) * depth + 1,
            #[cfg(test)]
            high: 0,
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    /// Pushes a node index.
    ///
    /// The bound in the module doc makes overflow impossible. Were it ever
    /// broken, this panics rather than drop the node, since a dropped node
    /// removes its whole subtree from the answer, a false negative.
    fn push(&mut self, node: usize) {
        debug_assert!(
            self.len < self.limit,
            "search stack would exceed (M - 1) * depth + 1 = {}",
            self.limit
        );
        match self.items.get_mut(self.len) {
            Some(slot) => *slot = node,
            None => panic!("situs: search stack exceeded its proven bound of {STACK_CAPACITY}"),
        }
        self.len += 1;
        #[cfg(test)]
        {
            self.high = self.high.max(self.len);
        }
    }

    fn pop(&mut self) -> Option<usize> {
        self.len = self.len.checked_sub(1)?;
        Some(self.items[self.len])
    }
}

// ===========================================================================
// Query
// ===========================================================================

impl Index {
    /// Calls `visit` with the id of every entry whose box meets `window`.
    ///
    /// The query reports exactly those entries. A node is descended only if
    /// its box meets the window (which, by condition 2 of the crate's
    /// correctness argument, loses nothing), and a leaf entry is reported
    /// only if its own box meets the window. Below a node whose box the
    /// window contains, every entry is reported untested, because every one
    /// is a hit (the containment shortcut; crate doc, "Storage and search").
    /// An entry stored twice is reported twice. The order of visits is a
    /// deterministic function of the index layout rather than a meaningful
    /// one.
    ///
    /// The search is iterative, with a fixed stack on the call stack, so it
    /// never allocates.
    ///
    /// # Panics
    ///
    /// Never. The search stack's capacity is a proven bound on its height,
    /// and were that bound ever broken, the query would panic rather than
    /// silently drop part of the tree.
    pub fn query(&self, window: &Rect, mut visit: impl FnMut(u32)) {
        let ControlFlow::Continue(()) = self.try_query::<Infallible>(window, |id| {
            visit(id);
            ControlFlow::Continue(())
        });
    }

    /// Calls `visit` with the id of each entry whose box meets `window`, in
    /// the same order as [`Index::query`], until `visit` returns
    /// [`ControlFlow::Break`].
    ///
    /// Returns that `Break`, or `Continue(())` if every such entry was
    /// visited. The entries visited before a break are exactly a prefix of
    /// `query`'s sequence. Like `query`, it never allocates.
    ///
    /// ```
    /// use core::ops::ControlFlow;
    /// use situs::{Index, Rect};
    ///
    /// let r = |a, b, c, d| Rect::new(a, b, c, d).unwrap();
    /// let index = Index::bulk_load([(r(0.0, 0.0, 1.0, 1.0), 7), (r(5.0, 5.0, 6.0, 6.0), 9)].into());
    ///
    /// // Does any entry meet this box? Stop at the first one.
    /// let w = r(0.5, 0.5, 2.0, 2.0);
    /// assert!(index.try_query(&w, |_| ControlFlow::Break(())).is_break());
    ///
    /// // The first entry meeting a box, if any.
    /// let first = index.try_query(&r(4.0, 4.0, 9.0, 9.0), ControlFlow::Break);
    /// assert_eq!(first, ControlFlow::Break(9));
    /// ```
    ///
    /// # Panics
    ///
    /// Never, as for [`Index::query`].
    pub fn try_query<B>(
        &self,
        window: &Rect,
        mut visit: impl FnMut(u32) -> ControlFlow<B>,
    ) -> ControlFlow<B> {
        debug_assert!(self.depth <= MAX_DEPTH, "depth above the proven bound");
        let mut stack = Stack::new(self.depth);
        self.search::<B, true>(window, &mut stack, &mut visit)
    }

    /// The search behind both queries. `SHORTCUT` is `true` in every
    /// production path; the tests turn it off to measure what it saves.
    pub(super) fn search<B, const SHORTCUT: bool>(
        &self,
        window: &Rect,
        stack: &mut Stack,
        visit: &mut impl FnMut(u32) -> ControlFlow<B>,
    ) -> ControlFlow<B> {
        if self.nodes.is_empty() {
            return ControlFlow::Continue(());
        }
        stack.push(0);
        while let Some(i) = stack.pop() {
            let node = &self.nodes[i];
            if !node.rect.intersects(window) {
                continue;
            }
            if SHORTCUT && window.contains_rect(&node.rect) {
                self.emit_subtree(i, stack, visit)?;
            } else if node.leaf {
                for (rect, id) in &self.entries[node.first..node.end] {
                    if rect.intersects(window) {
                        visit(*id)?;
                    }
                }
            } else {
                // Pushed in reverse so that children are visited in index
                // order.
                for child in (node.first..node.end).rev() {
                    stack.push(child);
                }
            }
        }
        ControlFlow::Continue(())
    }

    /// Reports every entry below node `top` untested, because the window
    /// contains `top`'s box (crate doc, "Storage and search").
    ///
    /// Walks the subtree on the search stack, above its current height, with
    /// the search's own discipline, so the stack bound and the visit order
    /// are the search's. A break from `visit` ends the walk at once.
    fn emit_subtree<B>(
        &self,
        top: usize,
        stack: &mut Stack,
        visit: &mut impl FnMut(u32) -> ControlFlow<B>,
    ) -> ControlFlow<B> {
        let base = stack.len();
        stack.push(top);
        while stack.len() > base {
            let Some(i) = stack.pop() else { break };
            let n = &self.nodes[i];
            if n.leaf {
                for &(_, id) in &self.entries[n.first..n.end] {
                    visit(id)?;
                }
            } else {
                for c in (n.first..n.end).rev() {
                    stack.push(c);
                }
            }
        }
        ControlFlow::Continue(())
    }
}

#[cfg(test)]
mod tests;
