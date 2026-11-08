// src/dynamic/remove.rs

//! Exact removal from the dynamic tree, and the condensing that follows it.
//!
//! Removal follows Guttman's CondenseTree ("R-trees: A Dynamic Index
//! Structure for Spatial Searching", SIGMOD 1984), with the R\*-tree's
//! reinsertion of whole subtrees at their own level. The entry is found by
//! exact matching, and an underfull node on the path is detached and its
//! contents replanted. Every node left on the path has its box recomputed
//! as the exact union of what remains, and a root left with one child is
//! replaced by that child.
//!
//! Every box written here is an exact union of stored boxes. The only step
//! that consults a heuristic is the replanting, which runs through
//! insertion and determines only shape.

use alloc::vec::Vec;

use crate::Rect;
use crate::dynamic::DynamicIndex;
use crate::dynamic::node::Child;

impl DynamicIndex {
    /// Removes one stored entry equal to `(rect, id)` and returns `true`,
    /// or returns `false` if there is none.
    ///
    /// Equality is [`Rect`]'s `==` (bounds compared numerically, so `-0.0`
    /// equals `0.0`) together with the id, the same equality every query
    /// uses. If several stored entries are equal, one of them is removed,
    /// since they are the same point set with the same id and differ at
    /// most in the sign of a zero bound. Which one is removed is a
    /// deterministic function of the tree's state.
    ///
    /// **The search is complete.** A stored entry's box is contained in every
    /// ancestor's box, because every node box is the exact union of its
    /// children's, so every path to a matching entry passes the exact
    /// containment test ([`Rect::contains_rect`]). More than one child may
    /// contain `rect`, so the search descends into *every* child whose box
    /// contains it and backtracks when a subtree holds no match.
    ///
    /// **The bounds remain exact and tight.** Every node remaining on the path
    /// has its box recomputed as the exact union of its remaining children,
    /// so boxes shrink as entries leave, and [`DynamicIndex::bounds`] is
    /// always the exact union of the stored entries' boxes.
    ///
    /// ```
    /// use situs::{DynamicIndex, Rect};
    ///
    /// let r = |a, b, c, d| Rect::new(a, b, c, d).unwrap();
    /// let mut index = DynamicIndex::new();
    /// index.insert(r(0.0, 0.0, 1.0, 1.0), 7);
    /// index.insert(r(5.0, 5.0, 6.0, 6.0), 9);
    ///
    /// assert!(!index.remove(&r(0.0, 0.0, 1.0, 1.0), 9)); // wrong id
    /// assert!(index.remove(&r(5.0, 5.0, 6.0, 6.0), 9));
    /// assert_eq!(index.len(), 1);
    /// assert_eq!(index.bounds(), Some(r(0.0, 0.0, 1.0, 1.0))); // shrunk
    /// ```
    pub fn remove(&mut self, rect: &Rect, id: u32) -> bool {
        let Some(root) = self.root else {
            return false;
        };
        if !self.arena[root].rect.contains_rect(rect) {
            return false;
        }
        let mut path: Vec<usize> = Vec::with_capacity(8);
        let Some((leaf, at)) = self.find_entry(root, rect, id, &mut path) else {
            return false;
        };
        self.arena[leaf].remove(at);
        self.len -= 1;
        self.condense(&path);
        true
    }

    /// Depth-first search below `slot` for an entry equal to `(rect, id)`,
    /// into every child whose box contains `rect`, in child order. On
    /// success `path` holds the slots from `slot` down to the leaf, and the
    /// result is the leaf's slot and the entry's place in it. On failure
    /// `path` is left as it was. The recursion is as deep as the tree, a
    /// few levels.
    fn find_entry(
        &self,
        slot: usize,
        rect: &Rect,
        id: u32,
        path: &mut Vec<usize>,
    ) -> Option<(usize, usize)> {
        path.push(slot);
        let node = &self.arena[slot];
        for (at, child) in node.children().enumerate() {
            match *child {
                Child::Entry(stored, stored_id) => {
                    if stored_id == id && stored == *rect {
                        return Some((slot, at));
                    }
                }
                Child::Node(child_slot) => {
                    if self.arena[child_slot].rect.contains_rect(rect) {
                        if let Some(found) = self.find_entry(child_slot, rect, id, path) {
                            return Some(found);
                        }
                    }
                }
            }
        }
        path.pop();
        None
    }

    /// CondenseTree after an entry has left the last node of `path`.
    ///
    /// The pass runs bottom up. A non-root node left under the minimum fill
    /// is detached from its parent, its slot freed, and its children queued
    /// for replanting at its level. Every other node on the path has its box
    /// recomputed as the exact union of its remaining children. The queued
    /// children are then replanted by insertion, each at its own level, so
    /// every leaf stays at the same depth, and while the root is an internal
    /// node with one child, that child becomes the root.
    ///
    /// Only one child leaves any node on the path (the entry, or a detached
    /// child), so an internal root, which held at least two, retains at
    /// least one. Only a leaf root can be left empty.
    fn condense(&mut self, path: &[usize]) {
        // (child, the level of the node that should hold it)
        let mut orphans: Vec<(Child, u32)> = Vec::new();
        for depth in (1..path.len()).rev() {
            let slot = path[depth];
            if self.arena[slot].len() < Self::MIN_FILL {
                let parent = path[depth - 1];
                let at = self.arena[parent]
                    .children()
                    .position(|c| matches!(*c, Child::Node(s) if s == slot))
                    .expect("a path node is its parent's child");
                self.arena[parent].remove(at);
                let level = self.arena[slot].level;
                for child in self.arena[slot].children() {
                    orphans.push((*child, level));
                }
                self.release(slot);
            } else {
                self.arena[slot].rect = self.arena[slot].exact_box(self);
            }
        }
        let root = path[0];
        if self.arena[root].len() == 0 {
            // Only a leaf root empties, and then nothing was detached.
            self.release(root);
            self.root = None;
        } else {
            self.arena[root].rect = self.arena[root].exact_box(self);
        }

        // Each replanting is one insertion, with its own forced-reinsertion
        // record, as the R*-tree treats every insertion.
        for (child, level) in orphans {
            let mut reinserted: u64 = 0;
            self.place(child, level, &mut reinserted);
        }

        while let Some(root) = self.root {
            let node = &self.arena[root];
            if node.level == 0 || node.len() > 1 {
                break;
            }
            let Child::Node(only) = node.child(0) else {
                unreachable!("an internal node holds subtrees");
            };
            self.release(root);
            self.root = Some(only);
        }
    }
}
