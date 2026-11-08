// src/dynamic/node.rs

//! The arena node: a box, a level, and an inline child array.

use crate::Rect;
use crate::dynamic::DynamicIndex;

/// Room for the capacity plus the one extra child an overflowing node
/// briefly holds while its overflow is treated.
pub(crate) const SLOTS: usize = DynamicIndex::NODE_CAPACITY + 1;

/// One child of a node: a subtree for an internal node, an entry for a
/// leaf. A node's children are all of one kind, by its level.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Child {
    /// An arena slot of a node one level down.
    Node(usize),
    /// A stored entry.
    Entry(Rect, u32),
}

impl Child {
    /// The child's box: an entry's own, or the subtree's node box.
    pub(crate) fn rect(&self, index: &DynamicIndex) -> Rect {
        match *self {
            Child::Node(slot) => index.arena[slot].rect,
            Child::Entry(rect, _) => rect,
        }
    }
}

/// One node: its box (always the exact union of its children's), its level
/// (0 for a leaf), and its children, inline.
#[derive(Clone, Debug)]
pub(crate) struct DNode {
    pub(crate) rect: Rect,
    /// 0 for a leaf; a node's children are at `level - 1`.
    pub(crate) level: u32,
    children: [Option<Child>; SLOTS],
    count: usize,
}

impl DNode {
    /// A node of the given level with one first child. A node is never
    /// empty, so construction takes the first child and its box.
    pub(crate) fn with_first(level: u32, first: Child, rect: Rect) -> DNode {
        let mut children = [None; SLOTS];
        children[0] = Some(first);
        DNode {
            rect,
            level,
            children,
            count: 1,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.count
    }

    pub(crate) fn children(&self) -> impl DoubleEndedIterator<Item = &Child> {
        self.children[..self.count].iter().flatten()
    }

    pub(crate) fn child(&self, at: usize) -> Child {
        self.children[at].expect("child within count")
    }

    /// Appends a child. The caller updates the box; a node may briefly hold
    /// `SLOTS` children while an overflow is being treated, never more.
    pub(crate) fn push(&mut self, child: Child) {
        assert!(self.count < SLOTS, "a node grew past its overflow slot");
        self.children[self.count] = Some(child);
        self.count += 1;
    }

    /// Removes and returns the child at `at`, keeping the order of the
    /// rest. The caller updates the box.
    pub(crate) fn remove(&mut self, at: usize) -> Child {
        let taken = self.children[at].expect("child within count");
        for i in at..self.count - 1 {
            self.children[i] = self.children[i + 1];
        }
        self.count -= 1;
        self.children[self.count] = None;
        taken
    }

    /// Replaces the children with `kept`, in order. The caller updates the
    /// box.
    pub(crate) fn set_children(&mut self, kept: &[Child]) {
        assert!(kept.len() <= SLOTS);
        self.children = [None; SLOTS];
        for (slot, child) in self.children.iter_mut().zip(kept.iter()) {
            *slot = Some(*child);
        }
        self.count = kept.len();
    }

    /// The exact union of the children's boxes: the one way a node box is
    /// ever computed. A node is never empty, so the union exists.
    pub(crate) fn exact_box(&self, index: &DynamicIndex) -> Rect {
        let mut boxes = self.children().map(|c| c.rect(index));
        let first = boxes.next().expect("a node is never empty");
        boxes.fold(first, |acc, r| acc.union(&r))
    }
}
