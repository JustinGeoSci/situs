// src/dynamic/insert.rs

//! Insertion into the dynamic tree, and its bulk loading.
//!
//! Insertion follows the R\*-tree (Beckmann, Kriegel, Schneider & Seeger,
//! SIGMOD 1990). ChooseSubtree descends by least overlap enlargement just
//! above the leaves and by least area enlargement elsewhere. The first
//! overflow at each level of one insertion is treated by forced
//! reinsertion, and later overflows split along the axis the margins choose,
//! at the distribution the overlaps choose. Bulk loading packs by STR, as
//! the static index does, with one adjustment that holds every node at or
//! above the minimum fill.
//!
//! Every heuristic value is an `f64` and determines only the shape, as the
//! module documentation sets out. Heuristic comparisons pass through
//! [`key`], which places NaN past every number, so an overflowed metric can
//! neither panic nor order two values inconsistently. No heuristic value is
//! ever stored as a box. Every box written anywhere in this file is an exact
//! union of stored values.

use alloc::vec::Vec;
use core::cmp::Ordering;

use crate::Rect;
use crate::dynamic::DynamicIndex;
use crate::dynamic::node::{Child, DNode};
use crate::static_index::str_order;

// ===========================================================================
// Heuristics: f64, shape only
// ===========================================================================

/// The single key every heuristic comparison passes through. NaN, which
/// `∞ − ∞` produces in a degenerate metric, maps to `+∞`, the worst value,
/// and `total_cmp` then orders everything deterministically.
fn key(v: f64) -> f64 {
    if v.is_nan() { f64::INFINITY } else { v }
}

fn cmp_key(a: f64, b: f64) -> Ordering {
    key(a).total_cmp(&key(b))
}

/// The box's area. May overflow to `+∞`; never stored.
fn area(r: &Rect) -> f64 {
    (r.max_x() - r.min_x()) * (r.max_y() - r.min_y())
}

/// The box's margin (half-perimeter). May overflow; never stored.
fn margin(r: &Rect) -> f64 {
    (r.max_x() - r.min_x()) + (r.max_y() - r.min_y())
}

/// The area shared by two boxes; `0.0` when they do not meet.
fn intersection_area(a: &Rect, b: &Rect) -> f64 {
    if !a.intersects(b) {
        return 0.0;
    }
    let w = a.max_x().min(b.max_x()) - a.min_x().max(b.min_x());
    let h = a.max_y().min(b.max_y()) - a.min_y().max(b.min_y());
    w * h
}

/// Squared distance between two box centers, for the reinsertion sort.
fn center_distance_sq(a: &Rect, b: &Rect) -> f64 {
    let (ax, ay) = a.center();
    let (bx, by) = b.center();
    let (dx, dy) = (ax - bx, ay - by);
    dx * dx + dy * dy
}

/// How much `grown` costs over `own` in area: `area(own ∪ grown) − area(own)`.
fn area_enlargement(own: &Rect, added: &Rect) -> f64 {
    area(&own.union(added)) - area(own)
}

// ===========================================================================
// Insertion
// ===========================================================================

impl DynamicIndex {
    /// Stores one entry. Infallible: a [`Rect`] is already valid, and ids
    /// need not be unique.
    ///
    /// The box of every node on the way is recomputed by exact union, so
    /// [`DynamicIndex::bounds`] stays exact and tight after every call.
    pub fn insert(&mut self, rect: Rect, id: u32) {
        self.len += 1;
        if self.root.is_none() {
            let leaf = DNode::with_first(0, Child::Entry(rect, id), rect);
            self.root = Some(self.allocate(leaf));
            return;
        }
        // One bit per level: has forced reinsertion run there during this
        // top-level insertion? (The root never reinserts.)
        let mut reinserted: u64 = 0;
        self.place(Child::Entry(rect, id), 0, &mut reinserted);
    }

    /// Places `child` into a node at `level`: the descent, the push, the
    /// exact box updates, and any overflow treatment. Reinsertion calls
    /// this too, with `level > 0` for a subtree being replanted.
    pub(crate) fn place(&mut self, child: Child, level: u32, reinserted: &mut u64) {
        let child_rect = child.rect(self);
        let path = self.choose_path(&child_rect, level);
        let target = *path.last().expect("the tree has a root");
        self.arena[target].push(child);
        self.recompute_boxes(&path);
        if self.arena[target].len() > Self::NODE_CAPACITY {
            self.treat_overflow(&path, reinserted);
        }
    }

    /// The slots from the root down to the chosen node at `level`.
    fn choose_path(&self, rect: &Rect, level: u32) -> Vec<usize> {
        let mut slot = self.root.expect("the tree has a root");
        let mut path = Vec::with_capacity(8);
        path.push(slot);
        while self.arena[slot].level > level {
            slot = self.choose_child(slot, rect);
            path.push(slot);
        }
        path
    }

    /// ChooseSubtree at one node: the child to descend into for `rect`.
    ///
    /// When the children are leaves, the choice is the least **overlap
    /// enlargement**, the increase in the summed intersection areas between
    /// the child's box and its siblings' that adding `rect` to the child
    /// would cause. Ties break by least area enlargement, then least area,
    /// then lowest position. When the children are internal, the choice is
    /// the least area enlargement, with the same ties.
    fn choose_child(&self, slot: usize, rect: &Rect) -> usize {
        let node = &self.arena[slot];
        let boxes: Vec<Rect> = node.children().map(|c| c.rect(self)).collect();
        let children_are_leaves = node.level == 1;
        let mut best: Option<(usize, f64, f64, f64)> = None;
        for (i, own) in boxes.iter().enumerate() {
            let grown = own.union(rect);
            let overlap_cost = if children_are_leaves {
                let mut cost = 0.0f64;
                for (j, other) in boxes.iter().enumerate() {
                    if i != j {
                        cost += intersection_area(&grown, other) - intersection_area(own, other);
                    }
                }
                cost
            } else {
                0.0
            };
            let enlargement = area_enlargement(own, rect);
            let own_area = area(own);
            let better = match &best {
                None => true,
                Some((_, bo, be, ba)) => {
                    cmp_key(overlap_cost, *bo)
                        .then_with(|| cmp_key(enlargement, *be))
                        .then_with(|| cmp_key(own_area, *ba))
                        == Ordering::Less
                }
            };
            if better {
                best = Some((i, overlap_cost, enlargement, own_area));
            }
        }
        let (at, ..) = best.expect("a node is never empty");
        match self.arena[slot].child(at) {
            Child::Node(child_slot) => child_slot,
            Child::Entry(..) => unreachable!("descent stops above the entries"),
        }
    }

    /// Recomputes, bottom up, the exact box of every node on `path`. Only
    /// exact unions are taken, so growth and shrinkage are handled alike.
    pub(crate) fn recompute_boxes(&mut self, path: &[usize]) {
        for &slot in path.iter().rev() {
            self.arena[slot].rect = self.arena[slot].exact_box(self);
        }
    }

    /// Treats an overflow at the last node of `path`, by forced reinsertion
    /// on the first overflow at that node's level (never at the root) and by
    /// a split otherwise.
    fn treat_overflow(&mut self, path: &[usize], reinserted: &mut u64) {
        let slot = *path.last().expect("path is never empty");
        let level = self.arena[slot].level;
        let is_root = path.len() == 1;
        let bit = 1u64 << level.min(63);
        if !is_root && *reinserted & bit == 0 {
            *reinserted |= bit;
            self.reinsert(path, reinserted);
        } else {
            self.split(path, reinserted);
        }
    }

    /// Forced reinsertion removes the children farthest from the center of
    /// the node's box, shrinks the boxes on the path, and replants the
    /// removed children at the same level, nearest first, the paper's close
    /// reinsert.
    fn reinsert(&mut self, path: &[usize], reinserted: &mut u64) {
        let slot = *path.last().expect("path is never empty");
        let level = self.arena[slot].level;
        let node_box = self.arena[slot].rect;
        let mut order: Vec<(f64, usize)> = self.arena[slot]
            .children()
            .enumerate()
            .map(|(i, c)| (center_distance_sq(&c.rect(self), &node_box), i))
            .collect();
        // Farthest first; ties by position, so the order is deterministic.
        order.sort_by(|a, b| cmp_key(b.0, a.0).then_with(|| a.1.cmp(&b.1)));
        let mut removed: Vec<Child> = Vec::with_capacity(Self::REINSERT_COUNT);
        let mut far: Vec<usize> = order[..Self::REINSERT_COUNT].iter().map(|e| e.1).collect();
        // Remove from the highest position down, so positions stay valid.
        far.sort_unstable_by(|a, b| b.cmp(a));
        for at in far {
            removed.push(self.arena[slot].remove(at));
        }
        self.recompute_boxes(path);
        // `removed` holds the farthest in removal (descending-position)
        // order; replant in increasing distance order instead.
        let mut by_distance: Vec<(f64, usize)> = removed
            .iter()
            .enumerate()
            .map(|(i, c)| (center_distance_sq(&c.rect(self), &node_box), i))
            .collect();
        by_distance.sort_by(|a, b| cmp_key(a.0, b.0).then_with(|| a.1.cmp(&b.1)));
        for (_, i) in by_distance {
            self.place(removed[i], level, reinserted);
        }
    }

    /// Splits the last node of `path` into two, by the R\* axis and
    /// distribution, and pushes the new node into the parent, treating any
    /// overflow there in turn. Splitting the root grows the tree by one
    /// level, so every leaf stays at the same depth.
    fn split(&mut self, path: &[usize], reinserted: &mut u64) {
        let slot = *path.last().expect("path is never empty");
        let level = self.arena[slot].level;
        let (first, second) = self.split_groups(slot);
        self.arena[slot].set_children(&first);
        self.arena[slot].rect = self.arena[slot].exact_box(self);
        let second_box = boxes_union(&second, self);
        let mut sibling = DNode::with_first(level, second[0], second_box);
        for child in &second[1..] {
            sibling.push(*child);
        }
        let sibling_slot = self.allocate(sibling);

        if path.len() == 1 {
            // The root split: a new root one level up holds both halves.
            let old_root_box = self.arena[slot].rect;
            let root = DNode::with_first(level + 1, Child::Node(slot), old_root_box);
            let root_slot = self.allocate(root);
            self.arena[root_slot].push(Child::Node(sibling_slot));
            self.arena[root_slot].rect = self.arena[root_slot].exact_box(self);
            self.root = Some(root_slot);
            return;
        }
        let parent_path = &path[..path.len() - 1];
        let parent = *parent_path.last().expect("a non-root node has a parent");
        self.arena[parent].push(Child::Node(sibling_slot));
        self.recompute_boxes(parent_path);
        if self.arena[parent].len() > Self::NODE_CAPACITY {
            self.treat_overflow(parent_path, reinserted);
        }
    }

    /// The R\* split of an overflowing node's children into two groups.
    ///
    /// ChooseSplitAxis sorts the children on each axis by lower bound, then
    /// upper bound, then position, and sums both groups' margins over the
    /// distributions that place `m` to `M + 1 − m` children in the first
    /// group. The axis with the smaller sum is taken, the x axis on a tie.
    /// ChooseSplitIndex then takes, on that axis's order, the distribution
    /// with the least intersection area between the group boxes, breaking
    /// ties by least total area and then by the first in sort order.
    fn split_groups(&self, slot: usize) -> (Vec<Child>, Vec<Child>) {
        let children: Vec<Child> = self.arena[slot].children().copied().collect();
        let m = Self::MIN_FILL;
        let splits = children.len() + 1 - 2 * m;

        let sorted_on = |by_y: bool| -> Vec<Child> {
            let mut order: Vec<(usize, Child)> = children.iter().copied().enumerate().collect();
            order.sort_by(|(ia, a), (ib, b)| {
                let (ra, rb) = (a.rect(self), b.rect(self));
                let (lo_a, hi_a, lo_b, hi_b) = if by_y {
                    (ra.min_y(), ra.max_y(), rb.min_y(), rb.max_y())
                } else {
                    (ra.min_x(), ra.max_x(), rb.min_x(), rb.max_x())
                };
                lo_a.total_cmp(&lo_b)
                    .then_with(|| hi_a.total_cmp(&hi_b))
                    .then_with(|| ia.cmp(ib))
            });
            order.into_iter().map(|(_, c)| c).collect()
        };

        let margin_sum = |sorted: &[Child]| -> f64 {
            let mut sum = 0.0f64;
            for k in m..=sorted.len() - m {
                let (a, b) = sorted.split_at(k);
                sum += margin(&boxes_union(a, self)) + margin(&boxes_union(b, self));
            }
            sum
        };

        let on_x = sorted_on(false);
        let on_y = sorted_on(true);
        let sorted = if cmp_key(margin_sum(&on_y), margin_sum(&on_x)) == Ordering::Less {
            on_y
        } else {
            on_x
        };

        let mut best: Option<(usize, f64, f64)> = None;
        for k in m..m + splits {
            let (a, b) = sorted.split_at(k);
            let (box_a, box_b) = (boxes_union(a, self), boxes_union(b, self));
            let overlap = intersection_area(&box_a, &box_b);
            let total_area = area(&box_a) + area(&box_b);
            let better = match &best {
                None => true,
                Some((_, bo, ba)) => {
                    cmp_key(overlap, *bo).then_with(|| cmp_key(total_area, *ba)) == Ordering::Less
                }
            };
            if better {
                best = Some((k, overlap, total_area));
            }
        }
        let (k, ..) = best.expect("at least one distribution exists");
        let (a, b) = sorted.split_at(k);
        (a.to_vec(), b.to_vec())
    }
}

/// The exact union of a non-empty child group's boxes.
fn boxes_union(children: &[Child], index: &DynamicIndex) -> Rect {
    let mut boxes = children.iter().map(|c| c.rect(index));
    let first = boxes.next().expect("a split group is never empty");
    boxes.fold(first, |acc, r| acc.union(&r))
}

// ===========================================================================
// Bulk loading
// ===========================================================================

impl DynamicIndex {
    /// Builds the initial tree by STR packing, the static index's method,
    /// with one adjustment. When a level's last chunk would hold fewer than
    /// the minimum fill, the last two chunks are rebalanced so both hold at
    /// least that many. The previous chunk is full and the capacity is at
    /// least twice the minimum, so the rebalance is always possible. Only
    /// the grouping changes rather than the order, and every box is an exact
    /// union. A level of one chunk is the root and is exempt.
    #[must_use]
    pub fn bulk_load(mut entries: Vec<(Rect, u32)>) -> DynamicIndex {
        let mut index = DynamicIndex::new();
        index.len = entries.len();
        if entries.is_empty() {
            return index;
        }
        str_order(&mut entries, |e| e.0, |e| u64::from(e.1));
        let mut level: Vec<usize> = chunk_sizes(entries.len())
            .into_iter()
            .scan(0usize, |taken, size| {
                let chunk = &entries[*taken..*taken + size];
                *taken += size;
                Some(index.load_node(0, chunk.iter().map(|&(r, id)| Child::Entry(r, id))))
            })
            .collect();
        let mut height = 1u32;
        while level.len() > 1 {
            let mut items: Vec<(Rect, usize)> =
                level.iter().map(|&s| (index.arena[s].rect, s)).collect();
            str_order(&mut items, |n| n.0, |_| 0);
            level = chunk_sizes(items.len())
                .into_iter()
                .scan(0usize, |taken, size| {
                    let chunk = &items[*taken..*taken + size];
                    *taken += size;
                    Some(index.load_node(height, chunk.iter().map(|&(_, s)| Child::Node(s))))
                })
                .collect();
            height += 1;
        }
        index.root = Some(level[0]);
        index
    }

    /// Allocates one node of `level` over `children`, its box the exact
    /// union of theirs.
    fn load_node(&mut self, level: u32, mut children: impl Iterator<Item = Child>) -> usize {
        let first = children.next().expect("a chunk is never empty");
        let mut node = DNode::with_first(level, first, first.rect(self));
        for child in children {
            node.push(child);
        }
        node.rect = node.exact_box(self);
        self.allocate(node)
    }
}

/// The chunk sizes of one level: full chunks of the capacity, with the last
/// two rebalanced when the final remainder is positive but below the
/// minimum fill. A single chunk (the root) is exempt.
fn chunk_sizes(n: usize) -> Vec<usize> {
    let (cap, min) = (DynamicIndex::NODE_CAPACITY, DynamicIndex::MIN_FILL);
    if n <= cap {
        return alloc::vec![n];
    }
    let mut sizes = Vec::with_capacity(n.div_ceil(cap));
    let full = n / cap;
    let rest = n % cap;
    for _ in 0..full {
        sizes.push(cap);
    }
    if rest > 0 {
        if rest < min {
            let moved = min - rest;
            let last_full = sizes.len() - 1;
            sizes[last_full] = cap - moved;
            sizes.push(rest + moved);
        } else {
            sizes.push(rest);
        }
    }
    sizes
}
