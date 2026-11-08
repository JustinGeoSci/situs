// src/dynamic/query.rs

//! The box query on the dynamic tree.

use alloc::vec::Vec;

use crate::Rect;
use crate::dynamic::DynamicIndex;
use crate::dynamic::node::Child;

impl DynamicIndex {
    /// Calls `visit` with the id of every entry whose box meets `window`.
    ///
    /// The query reports exactly those entries, by the argument the crate
    /// documentation makes for the static index. Every node's box contains
    /// its children's boxes (here it is their exact union), so descending
    /// only into nodes whose boxes meet the window loses nothing, and every
    /// entry's own box passes the closed overlap test before it is reported.
    /// An entry stored twice is reported twice. The visit order is a
    /// deterministic function of the tree's state rather than a meaningful
    /// one.
    pub fn query(&self, window: &Rect, mut visit: impl FnMut(u32)) {
        let Some(root) = self.root else {
            return;
        };
        let mut stack: Vec<usize> = Vec::new();
        stack.push(root);
        while let Some(slot) = stack.pop() {
            let node = &self.arena[slot];
            if !node.rect.intersects(window) {
                continue;
            }
            if node.level == 0 {
                for child in node.children() {
                    if let Child::Entry(rect, id) = *child {
                        if rect.intersects(window) {
                            visit(id);
                        }
                    }
                }
            } else {
                // Pushed in reverse so subtrees are visited in child order,
                // as the static search does.
                for child in node.children().rev() {
                    if let Child::Node(child_slot) = *child {
                        stack.push(child_slot);
                    }
                }
            }
        }
    }
}
