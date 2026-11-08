// src/dynamic/check.rs

//! The structural invariant checker for the tests and the fuzz target.
//!
//! It compiles only for the tests and for fuzzing (`cfg(fuzzing)`, which
//! `cargo fuzz` sets), so the fuzz target checks the same invariants after
//! every operation as the tests do. It is never part of a normal build.

use alloc::vec;
use alloc::vec::Vec;

use crate::Rect;
use crate::dynamic::DynamicIndex;
use crate::dynamic::node::Child;

/// One entry as comparable bits.
fn entry_bits(rect: &Rect, id: u32) -> (u64, u64, u64, u64, u32) {
    (
        rect.min_x().to_bits(),
        rect.min_y().to_bits(),
        rect.max_x().to_bits(),
        rect.max_y().to_bits(),
        id,
    )
}

impl DynamicIndex {
    /// Panics unless the tree's invariants hold against `model`, the
    /// entries that should be stored:
    ///
    /// - **I1**. Every node's box equals the exact union of its children's
    ///   boxes, bit for bit.
    /// - **I2**. A node's children are exactly one level below it, so every leaf
    ///   is at the same depth.
    /// - **I3**. The stored entries are exactly the model's multiset, as bit
    ///   patterns.
    /// - **S1**. No node holds more than the capacity, a non-root node holds at
    ///   least the minimum fill, and an internal root holds at least two.
    ///
    /// The checker also verifies the arena's consistency (every reachable
    /// slot reached once, and no reachable slot on the free list), `len`
    /// against the model, and `bounds()` against the model's exact union.
    /// That last comparison is numeric rather than bitwise. When `-0.0` and
    /// `0.0` both occur among the entries' bounds, which zero survives a
    /// union depends on the fold order, and the tree folds in tree order
    /// while a model folds in its own. The two boxes are the same point set,
    /// and every comparison treats the zeros alike. The bitwise discipline
    /// belongs to I1, per node, where the child order is fixed.
    #[doc(hidden)]
    pub fn check_invariants(&self, model: &[(Rect, u32)]) {
        assert_eq!(self.len(), model.len(), "len differs from the model");
        assert_eq!(self.is_empty(), model.is_empty());
        let Some(root) = self.root else {
            assert!(model.is_empty(), "entries stored but no root");
            assert_eq!(self.bounds(), None);
            return;
        };

        let mut stored: Vec<(u64, u64, u64, u64, u32)> = Vec::new();
        let mut seen = vec![false; self.arena.len()];
        let mut stack = vec![root];
        while let Some(slot) = stack.pop() {
            assert!(!seen[slot], "slot {slot} is reachable twice");
            seen[slot] = true;
            let node = &self.arena[slot];
            // S1.
            assert!(
                node.len() <= Self::NODE_CAPACITY,
                "slot {slot} over capacity"
            );
            if slot == root {
                let least = if node.level == 0 { 1 } else { 2 };
                assert!(node.len() >= least, "root under its minimum");
            } else {
                assert!(
                    node.len() >= Self::MIN_FILL,
                    "slot {slot} holds {} children, under the minimum fill",
                    node.len()
                );
            }
            // I1: bitwise.
            let exact = node.exact_box(self);
            assert_eq!(
                entry_bits(&node.rect, 0),
                entry_bits(&exact, 0),
                "slot {slot}: the node box is not the exact union of its children"
            );
            // I2 through the levels; I3 collection.
            for child in node.children() {
                match *child {
                    Child::Node(child_slot) => {
                        assert!(node.level >= 1, "a leaf holds a subtree");
                        assert_eq!(
                            self.arena[child_slot].level,
                            node.level - 1,
                            "slot {slot}: a child is not one level down"
                        );
                        stack.push(child_slot);
                    }
                    Child::Entry(rect, id) => {
                        assert_eq!(node.level, 0, "an internal node holds an entry");
                        stored.push(entry_bits(&rect, id));
                    }
                }
            }
        }
        for &slot in &self.free {
            assert!(!seen[slot], "slot {slot} is both reachable and free");
        }
        // I3.
        let mut want: Vec<_> = model
            .iter()
            .map(|(rect, id)| entry_bits(rect, *id))
            .collect();
        stored.sort_unstable();
        want.sort_unstable();
        assert_eq!(stored, want, "the stored entries differ from the model");
        // bounds(): exact and tight over the model, numerically.
        let union = model
            .iter()
            .map(|(rect, _)| *rect)
            .reduce(|acc, b| acc.union(&b))
            .expect("the model is non-empty here");
        let got = self.bounds().expect("a non-empty tree has bounds");
        // `Rect`'s `==` compares the bounds numerically.
        assert!(
            got == union,
            "bounds() {got:?} is not the exact union of the model's boxes {union:?}"
        );
    }

    /// Every stored entry, in tree order (depth first, child order). For a
    /// model to tell which of several `==`-equal entries a removal took
    /// when their bounds differ only in the sign of a zero.
    #[doc(hidden)]
    pub fn stored_entries(&self) -> Vec<(Rect, u32)> {
        let mut out = Vec::with_capacity(self.len);
        let mut stack: Vec<usize> = self.root.into_iter().collect();
        while let Some(slot) = stack.pop() {
            for child in self.arena[slot].children().rev() {
                match *child {
                    Child::Node(s) => stack.push(s),
                    Child::Entry(rect, id) => out.push((rect, id)),
                }
            }
        }
        out
    }
}
