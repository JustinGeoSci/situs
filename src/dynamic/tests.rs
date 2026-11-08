// src/dynamic/tests.rs

//! Tests of the dynamic tree.
//!
//! Every test drives operations against a **model**, the plain vector of
//! entries that should be stored. After every operation, [`check_invariants`]
//! verifies four invariants against the model:
//!
//! - **I1**. Every node's box equals the exact union of its children's boxes,
//!   bit for bit.
//! - **I2**. Every leaf is at the same depth, checked through the level numbers
//!   (a node's children are exactly one level below it).
//! - **I3**. The stored entries are exactly the model's multiset, each in one
//!   leaf.
//! - **S1**. No node holds more than the capacity, a non-root node holds at
//!   least the minimum fill, and an internal root holds at least two.
//!
//! Queries are compared with brute force over the model, so the tree is
//! never trusted about its own contents.

#![allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use super::node::{Child, DNode};
use super::*;
use crate::nearest::bounds::tests::Rng;
use crate::static_index::tests::{entries, lattice, pseudo_random_boxes};
use crate::tests::r;
use proptest::prelude::*;
use std::vec;

// ===========================================================================
// The invariant checker and the model
// ===========================================================================

/// The invariant check (the `check` module): I1, I2, I3, S1, the arena,
/// `len` and `bounds()` against the model.
pub(crate) fn check_invariants(index: &DynamicIndex, model: &[(Rect, u32)]) {
    index.check_invariants(model);
}

/// Brute force over the model, ids sorted.
fn brute(model: &[(Rect, u32)], window: &Rect) -> Vec<u32> {
    let mut ids: Vec<u32> = model
        .iter()
        .filter(|(rect, _)| rect.intersects(window))
        .map(|(_, id)| *id)
        .collect();
    ids.sort_unstable();
    ids
}

/// The tree's answer, ids sorted.
fn queried(index: &DynamicIndex, window: &Rect) -> Vec<u32> {
    let mut ids = Vec::new();
    index.query(window, |id| ids.push(id));
    ids.sort_unstable();
    ids
}

/// The query check: 8 pseudo-random windows, the touching windows of the
/// static tests, and the whole extent, each against brute force.
fn check_queries(index: &DynamicIndex, model: &[(Rect, u32)], rng: &mut Rng) {
    let mut windows: Vec<Rect> = Vec::new();
    for _ in 0..8 {
        let c = |rng: &mut Rng| (rng.int(1 << 10) as f64) * 0.5;
        let (a, b, x, y) = (c(rng), c(rng), c(rng), c(rng));
        windows.push(r(a.min(x), b.min(y), a.max(x), b.max(y)));
    }
    if let Some((e, _)) = model.first() {
        windows.push(r(e.max_x(), e.max_y(), e.max_x() + 1.0, e.max_y() + 1.0));
        windows.push(r(e.min_x() - 1.0, e.min_y(), e.min_x(), e.max_y()));
    }
    if let Some(all) = index.bounds() {
        windows.push(all);
    }
    for w in &windows {
        assert_eq!(queried(index, w), brute(model, w), "window {w:?}");
    }
}

/// Inserts a sequence one entry at a time, checking the invariants after
/// every insertion and the queries after every tenth.
fn drive(
    start: DynamicIndex,
    start_model: Vec<(Rect, u32)>,
    inserts: &[(Rect, u32)],
    rng: &mut Rng,
) {
    let mut index = start;
    let mut model = start_model;
    check_invariants(&index, &model);
    for (i, (rect, id)) in inserts.iter().enumerate() {
        index.insert(*rect, *id);
        model.push((*rect, *id));
        check_invariants(&index, &model);
        if i % 10 == 9 {
            check_queries(&index, &model, rng);
        }
    }
    check_queries(&index, &model, rng);
}

// ===========================================================================
// Fixed sequences
// ===========================================================================

#[test]
fn an_empty_tree_answers_empty() {
    let index = DynamicIndex::new();
    assert!(index.is_empty());
    assert_eq!(index.len(), 0);
    assert_eq!(index.bounds(), None);
    let mut n = 0;
    index.query(&r(-1e300, -1e300, 1e300, 1e300), |_| n += 1);
    assert_eq!(n, 0);
    check_invariants(&index, &[]);
    assert_eq!(DynamicIndex::default().len(), 0);
}

#[test]
fn lattice_insertions_in_row_order_keep_the_invariants() {
    let mut rng = Rng::new(0x5EED_D101);
    let mut source = lattice(18);
    source.truncate(300);
    drive(DynamicIndex::new(), Vec::new(), &source, &mut rng);
}

#[test]
fn lattice_insertions_in_reverse_keep_the_invariants() {
    let mut rng = Rng::new(0x5EED_D102);
    let mut source = lattice(18);
    source.truncate(300);
    source.reverse();
    drive(DynamicIndex::new(), Vec::new(), &source, &mut rng);
}

#[test]
fn lattice_insertions_shuffled_keep_the_invariants() {
    let mut rng = Rng::new(0x5EED_D103);
    let mut source = lattice(18);
    source.truncate(300);
    for i in (1..source.len()).rev() {
        source.swap(i, (rng.next() as usize) % (i + 1));
    }
    drive(DynamicIndex::new(), Vec::new(), &source, &mut rng);
}

#[test]
fn a_thousand_identical_boxes_keep_the_invariants() {
    let mut rng = Rng::new(0x5EED_D104);
    let source: Vec<(Rect, u32)> = (0..1_000).map(|i| (r(1.0, 1.0, 2.0, 2.0), i)).collect();
    drive(DynamicIndex::new(), Vec::new(), &source, &mut rng);
    // And queried: every one comes back.
    let mut index = DynamicIndex::new();
    for (rect, id) in &source {
        index.insert(*rect, *id);
    }
    assert_eq!(queried(&index, &r(2.0, 2.0, 3.0, 3.0)).len(), 1_000);
    assert!(queried(&index, &r(2.1, 2.0, 3.0, 3.0)).is_empty());
}

#[test]
fn extreme_magnitudes_keep_the_invariants() {
    let big = 1e300;
    let tiny = f64::MIN_POSITIVE;
    let mut rng = Rng::new(0x5EED_D105);
    let source = vec![
        (r(-big, -big, -big, -big), 0),
        (r(big, big, big, big), 1),
        (r(-big, -big, big, big), 2),
        (r(0.0, 0.0, tiny, tiny), 3),
        (r(-tiny, -tiny, 0.0, 0.0), 4),
        (r(tiny, tiny, 2.0 * tiny, 2.0 * tiny), 5),
        (r(-big, 0.0, big, 0.0), 6),
        (r(-f64::MAX, -1.0, f64::MAX, 1.0), 7),
        (r(f64::MAX, f64::MAX, f64::MAX, f64::MAX), 8),
    ];
    // Repeated four times: enough identical extremes to overflow nodes and
    // drive reinsertion and splits through the degenerate heuristics.
    let mut all = Vec::new();
    for round in 0..4u32 {
        all.extend(source.iter().map(|&(b, id)| (b, id + 10 * round)));
    }
    drive(DynamicIndex::new(), Vec::new(), &all, &mut rng);
}

// ===========================================================================
// Bulk loading
// ===========================================================================

#[test]
fn bulk_load_of_every_small_count_keeps_the_invariants() {
    let m = DynamicIndex::NODE_CAPACITY;
    let mut counts: Vec<usize> = (0..=300).collect();
    for k in [m * m, m * m * m] {
        counts.extend([k - 1, k, k + 1]);
    }
    for n in counts {
        let side = (n as f64).sqrt().ceil().max(1.0) as u32;
        let mut source = lattice(side);
        source.truncate(n);
        let index = DynamicIndex::bulk_load(source.clone());
        check_invariants(&index, &source);
    }
}

#[test]
fn bulk_load_matches_brute_force_and_accepts_insertions() {
    let mut rng = Rng::new(0x5EED_D106);
    let source = pseudo_random_boxes(3_000);
    let (start, rest) = source.split_at(2_800);
    let index = DynamicIndex::bulk_load(start.to_vec());
    check_invariants(&index, start);
    drive(index, start.to_vec(), rest, &mut rng);
}

// ===========================================================================
// Structural defect classes
// ===========================================================================

/// Bulk loads at the counts where an STR loader that splits naively leaves
/// leaves under the minimum and uneven depths (one segment per leaf at 7–11
/// and 25–47 items), and at every count to 300. Fill and depth hold
/// everywhere.
#[test]
fn dynamic_undersized_leaves_and_uneven_depth() {
    for n in (7..=11).chain(25..=47) {
        let mut source = lattice(7);
        source.truncate(n);
        let index = DynamicIndex::bulk_load(source.clone());
        check_invariants(&index, &source);
    }
}

/// No node, the root included, grows past the capacity in any order, a
/// defect class R-tree loaders have shown. The sequence is 2,000 insertions
/// into one spot, then spread out.
#[test]
fn dynamic_oversized_root_cannot_happen() {
    let mut index = DynamicIndex::new();
    let mut model = Vec::new();
    for i in 0..2_000u32 {
        let spread = f64::from(i % 40);
        let b = r(spread, 0.0, spread + 1.0, 1.0);
        index.insert(b, i);
        model.push((b, i));
    }
    check_invariants(&index, &model);
}

/// Insertion into a bulk-loaded tree is ordinary, because the load retains
/// the fill invariants. A load that breaks them can make a later insertion
/// panic.
#[test]
fn dynamic_insert_after_bulk_load() {
    let mut rng = Rng::new(0x5EED_D107);
    for n in [7usize, 11, 25, 47, 100, 257] {
        let mut source = lattice(17);
        source.truncate(n);
        let index = DynamicIndex::bulk_load(source.clone());
        let extra: Vec<(Rect, u32)> = (0..40)
            .map(|i| {
                let x = (rng.int(20) as f64) * 0.5;
                let y = (rng.int(20) as f64) * 0.5;
                (r(x, y, x + 1.5, y + 1.5), 1_000 + i)
            })
            .collect();
        drive(index, source, &extra, &mut rng);
    }
}

/// Forced reinsertion neither loses nor duplicates an entry. Everything
/// lands in one tight cluster, so the same nodes overflow again and again
/// and reinsertion runs constantly, and the model comparison is exact.
#[test]
fn dynamic_reinsertion_keeps_every_entry() {
    let mut rng = Rng::new(0x5EED_D108);
    let source: Vec<(Rect, u32)> = (0..600u32)
        .map(|i| {
            let dx = f64::from(i % 7) * 0.25;
            let dy = f64::from(i % 11) * 0.25;
            (r(dx, dy, dx + 0.5, dy + 0.5), i)
        })
        .collect();
    drive(DynamicIndex::new(), Vec::new(), &source, &mut rng);
}

/// Center arithmetic on coordinates past half the maximum overflows, and a
/// NaN comparison can then panic. Here the heuristics may degenerate, but
/// nothing panics and the invariants hold.
#[test]
fn dynamic_extreme_magnitudes_never_panic() {
    let mut rng = Rng::new(0x5EED_D109);
    let huge = f64::MAX / 2.0;
    let source: Vec<(Rect, u32)> = (0..80u32)
        .map(|i| {
            let s = if i % 2 == 0 { 1.0 } else { -1.0 };
            let base = s * (huge + f64::from(i % 5) * 1e300);
            (r(base.min(s * huge), -1e308, base.max(s * huge), 1e308), i)
        })
        .collect();
    drive(DynamicIndex::new(), Vec::new(), &source, &mut rng);
}

/// An empty tree has no root at all rather than a sentinel box, and one
/// entry makes the bounds exactly that entry's box.
#[test]
fn dynamic_no_empty_sentinel() {
    let mut index = DynamicIndex::new();
    assert_eq!(index.bounds(), None);
    index.insert(r(2.0, 3.0, 5.0, 7.0), 42);
    assert_eq!(index.bounds(), Some(r(2.0, 3.0, 5.0, 7.0)));
    check_invariants(&index, &[(r(2.0, 3.0, 5.0, 7.0), 42)]);
}

/// Counts that are exact powers of the capacity are where a recursive
/// loader can fail to terminate. These loads terminate and hold the
/// invariants.
#[test]
fn dynamic_power_of_capacity_counts() {
    let m = DynamicIndex::NODE_CAPACITY;
    for n in [m - 1, m, m + 1, m * m - 1, m * m, m * m + 1, m * m * m] {
        let side = (n as f64).sqrt().ceil().max(1.0) as u32;
        let mut source = lattice(side);
        source.truncate(n);
        let index = DynamicIndex::bulk_load(source.clone());
        check_invariants(&index, &source);
    }
}

// ===========================================================================
// Property cases
// ===========================================================================

proptest! {
    // Half the crate's usual case count: 5,000 in release and 500 in
    // debug, where the per-insertion invariant check makes each case far
    // costlier than the static sweeps' cases.
    #![proptest_config(ProptestConfig::with_cases(crate::tests::CASES / 2))]

    /// A random sequence of up to 1,000 insertions from the crate's box
    /// strategy, starting from an empty tree or from a bulk load of a
    /// random prefix. Invariants after every insertion; queries after every
    /// tenth.
    #[test]
    fn insertions_keep_the_invariants(
        es in entries(1_000),
        prefix in any::<u16>(),
        seed in any::<u64>(),
    ) {
        let mut rng = Rng::new(seed);
        let cut = if es.is_empty() { 0 } else { usize::from(prefix) % (es.len() + 1) };
        let (start, rest) = es.split_at(cut);
        let index = DynamicIndex::bulk_load(start.to_vec());
        check_invariants(&index, start);
        drive(index, start.to_vec(), rest, &mut rng);
    }
}

// ===========================================================================
// Removal: the model and the operation driver
// ===========================================================================

/// One entry as comparable bits.
fn bits(rect: &Rect, id: u32) -> (u64, u64, u64, u64, u32) {
    (
        rect.min_x().to_bits(),
        rect.min_y().to_bits(),
        rect.max_x().to_bits(),
        rect.max_y().to_bits(),
        id,
    )
}

/// Does the model hold an entry equal to `(rect, id)` under the removal's
/// equality (`Rect`'s `==` and the id)?
fn model_has(model: &[(Rect, u32)], rect: &Rect, id: u32) -> bool {
    model.iter().any(|(m, mid)| *mid == id && m == rect)
}

/// Removes from the model the entry the tree removed. Equal entries are the
/// same point set and id but may differ in the sign of a zero bound, and in
/// that case the tree's choice is read from its stored entries, so the model
/// tracks the tree bit for bit and I3 remains a bitwise check.
fn model_remove(model: &mut Vec<(Rect, u32)>, index: &DynamicIndex, rect: &Rect, id: u32) {
    let matches: Vec<usize> = (0..model.len())
        .filter(|&i| model[i].1 == id && model[i].0 == *rect)
        .collect();
    assert!(!matches.is_empty(), "the model holds no match");
    let first = bits(&model[matches[0]].0, id);
    if matches.iter().all(|&i| bits(&model[i].0, id) == first) {
        model.swap_remove(matches[0]);
        return;
    }
    let stored = index.stored_entries();
    for &i in &matches {
        let b = bits(&model[i].0, id);
        let in_model = model.iter().filter(|(m, mid)| bits(m, *mid) == b).count();
        let in_tree = stored.iter().filter(|(m, mid)| bits(m, *mid) == b).count();
        if in_tree < in_model {
            model.swap_remove(i);
            return;
        }
    }
    panic!("the tree reported a removal but holds every matching entry");
}

/// Removes `(rect, id)` from both, checking `remove`'s return value against
/// the model's, then the invariants.
fn remove_checked(index: &mut DynamicIndex, model: &mut Vec<(Rect, u32)>, rect: &Rect, id: u32) {
    let expected = model_has(model, rect, id);
    let got = index.remove(rect, id);
    assert_eq!(got, expected, "remove({rect:?}, {id}) returned {got}");
    if got {
        model_remove(model, index, rect, id);
    }
    check_invariants(index, model);
}

/// One operation of the model-based suite.
#[derive(Clone, Debug)]
enum Op {
    /// Insert this entry.
    Insert(Rect, u32),
    /// Remove the model's entry at this index (modulo its length).
    RemoveExisting(u16),
    /// Remove this entry, which is usually absent (the model decides).
    RemoveRandom(Rect, u32),
    /// Remove an existing entry's box under a different id.
    RemoveOtherId(u16),
    /// Query this window against brute force.
    Query(Rect),
}

/// The operation mix: about half insertions, 30% removals of existing
/// entries, 10% removals of absent ones (half random, half an existing box
/// under another id), and 10% queries. Ids come from a small range, so
/// equal boxes under equal and unequal ids both occur.
fn op() -> impl Strategy<Value = Op> {
    prop_oneof![
        50 => (crate::tests::rect(), 0u32..64).prop_map(|(b, id)| Op::Insert(b, id)),
        30 => any::<u16>().prop_map(Op::RemoveExisting),
        5 => (crate::tests::rect(), 0u32..64).prop_map(|(b, id)| Op::RemoveRandom(b, id)),
        5 => any::<u16>().prop_map(Op::RemoveOtherId),
        10 => crate::tests::rect().prop_map(Op::Query),
    ]
}

/// Applies `ops`, checking after every one: the invariants (which include
/// `len` and `bounds()` against the model), `remove`'s return value, and,
/// after a query, equality with brute force.
fn drive_ops(index: &mut DynamicIndex, model: &mut Vec<(Rect, u32)>, ops: &[Op]) {
    for op in ops {
        match op {
            Op::Insert(b, id) => {
                index.insert(*b, *id);
                model.push((*b, *id));
                check_invariants(index, model);
            }
            Op::RemoveExisting(at) => {
                if model.is_empty() {
                    remove_checked(index, model, &r(0.0, 0.0, 1.0, 1.0), 0);
                } else {
                    let (b, id) = model[usize::from(*at) % model.len()];
                    remove_checked(index, model, &b, id);
                }
            }
            Op::RemoveRandom(b, id) => remove_checked(index, model, b, *id),
            Op::RemoveOtherId(at) => {
                if !model.is_empty() {
                    let (b, id) = model[usize::from(*at) % model.len()];
                    remove_checked(index, model, &b, id.wrapping_add(1));
                }
            }
            Op::Query(w) => {
                assert_eq!(queried(index, w), brute(model, w), "window {w:?}");
            }
        }
    }
}

// ===========================================================================
// Removal: fixed sequences
// ===========================================================================

/// 1,000 lattice boxes, inserted one at a time.
fn thousand() -> (DynamicIndex, Vec<(Rect, u32)>) {
    let mut source = lattice(32);
    source.truncate(1_000);
    let mut index = DynamicIndex::new();
    for (b, id) in &source {
        index.insert(*b, *id);
    }
    check_invariants(&index, &source);
    (index, source)
}

/// Removes `order` one at a time, checking the invariants (and so the exact
/// bounds) after each removal and the queries after every 25th, until the
/// tree is empty.
fn drain(
    mut index: DynamicIndex,
    mut model: Vec<(Rect, u32)>,
    order: &[(Rect, u32)],
    rng: &mut Rng,
) {
    for (i, (b, id)) in order.iter().enumerate() {
        remove_checked(&mut index, &mut model, b, *id);
        if i % 25 == 24 {
            check_queries(&index, &model, rng);
        }
    }
    assert!(index.is_empty());
    assert_eq!(index.root, None);
    assert_eq!(index.bounds(), None);
    assert_eq!(index.free.len(), index.arena.len(), "a slot leaked");
    assert!(
        !index.remove(&r(0.0, 0.0, 1.0, 1.0), 0),
        "removal from an empty tree"
    );
}

#[test]
fn removal_in_insertion_order_keeps_the_invariants() {
    let mut rng = Rng::new(0x5EED_D601);
    let (index, source) = thousand();
    drain(index, source.clone(), &source, &mut rng);
}

#[test]
fn removal_in_reverse_order_keeps_the_invariants() {
    let mut rng = Rng::new(0x5EED_D602);
    let (index, source) = thousand();
    let mut order = source.clone();
    order.reverse();
    drain(index, source, &order, &mut rng);
}

#[test]
fn removal_in_shuffled_order_keeps_the_invariants() {
    let mut rng = Rng::new(0x5EED_D603);
    let (index, source) = thousand();
    let mut order = source.clone();
    for i in (1..order.len()).rev() {
        order.swap(i, (rng.next() as usize) % (i + 1));
    }
    drain(index, source, &order, &mut rng);
}

/// Emptied, then refilled: the emptied tree is an ordinary empty tree, its
/// freed slots reused.
#[test]
fn a_tree_emptied_by_removal_accepts_insertions_again() {
    let mut rng = Rng::new(0x5EED_D604);
    let source = pseudo_random_boxes(400);
    let index = DynamicIndex::bulk_load(source.clone());
    let mut index = {
        let mut index = index;
        let mut model = source.clone();
        for (b, id) in &source {
            remove_checked(&mut index, &mut model, b, *id);
        }
        index
    };
    assert!(index.is_empty());
    assert_eq!(index.free.len(), index.arena.len(), "a slot leaked");
    let arena_before = index.arena.len();
    let mut model = Vec::new();
    for (b, id) in source.iter().take(300) {
        index.insert(*b, *id);
        model.push((*b, *id));
        check_invariants(&index, &model);
        if model.len() == 1 {
            assert_eq!(
                index.arena.len(),
                arena_before,
                "a freed slot was not reused"
            );
        }
    }
    check_queries(&index, &model, &mut rng);
}

/// 1,000 identical boxes under distinct ids, all removed: every removal must
/// find its id among identical boxes, so the search backtracks through
/// every containing child.
#[test]
fn a_thousand_identical_boxes_are_all_removed() {
    let mut rng = Rng::new(0x5EED_D605);
    let source: Vec<(Rect, u32)> = (0..1_000).map(|i| (r(1.0, 1.0, 2.0, 2.0), i)).collect();
    let mut index = DynamicIndex::new();
    for (b, id) in &source {
        index.insert(*b, *id);
    }
    let mut order = source.clone();
    for i in (1..order.len()).rev() {
        order.swap(i, (rng.next() as usize) % (i + 1));
    }
    drain(index, source, &order, &mut rng);
}

/// Duplicates of one `(rect, id)`: `true` as many times as stored, then
/// `false`, and the other entries untouched.
#[test]
fn duplicates_are_removed_one_at_a_time() {
    let mut index = DynamicIndex::bulk_load(lattice(6));
    let mut model = lattice(6);
    let dup = (r(2.5, 2.5, 3.5, 3.5), 77);
    for _ in 0..5 {
        index.insert(dup.0, dup.1);
        model.push(dup);
    }
    check_invariants(&index, &model);
    for _ in 0..5 {
        remove_checked(&mut index, &mut model, &dup.0, dup.1);
        assert!(model.len() >= 36);
    }
    assert!(
        !index.remove(&dup.0, dup.1),
        "a sixth removal found an entry"
    );
    check_invariants(&index, &lattice(6));
}

/// `-0.0` and `0.0` are equal for removal, as for every comparison in the
/// crate: an entry stored with `-0.0` bounds is removed by a box with `0.0`
/// bounds, and the reverse.
#[test]
fn removal_matches_negative_zero_against_zero() {
    let mut index = DynamicIndex::new();
    let neg = r(-0.0, -0.0, 1.0, -0.0);
    let pos = r(0.0, 0.0, 1.0, 0.0);
    index.insert(neg, 1);
    index.insert(pos, 2);
    let mut model = vec![(neg, 1), (pos, 2)];
    check_invariants(&index, &model);
    remove_checked(&mut index, &mut model, &pos, 1);
    remove_checked(&mut index, &mut model, &neg, 2);
    assert!(index.is_empty());
    // Both signs stored under one id: two removals by either sign.
    index.insert(neg, 3);
    index.insert(pos, 3);
    model = vec![(neg, 3), (pos, 3)];
    remove_checked(&mut index, &mut model, &pos, 3);
    remove_checked(&mut index, &mut model, &pos, 3);
    assert!(index.is_empty());
    assert!(!index.remove(&neg, 3));
}

// ===========================================================================
// Underflow and shortening
// ===========================================================================

/// Builds a tree by hand: a root over `groups` internal nodes, each over
/// `leaves` leaves of `per_leaf` lattice-ish entries, laid out so groups
/// and leaves occupy disjoint regions. Every box is an exact union; the
/// invariants are checked before it is returned.
fn handmade(groups: usize, leaves: usize, per_leaf: usize) -> (DynamicIndex, Vec<(Rect, u32)>) {
    let mut index = DynamicIndex::new();
    let mut model = Vec::new();
    let mut group_slots = Vec::new();
    let mut id = 0u32;
    for g in 0..groups {
        let mut leaf_slots = Vec::new();
        for l in 0..leaves {
            let mut node: Option<DNode> = None;
            for e in 0..per_leaf {
                let x = (g * 1_000 + l * 20) as f64;
                let y = e as f64;
                let b = r(x, y, x + 1.0, y + 0.5);
                model.push((b, id));
                let child = Child::Entry(b, id);
                id += 1;
                match node.as_mut() {
                    None => node = Some(DNode::with_first(0, child, b)),
                    Some(n) => n.push(child),
                }
            }
            let mut node = node.unwrap();
            node.rect = node.exact_box(&index);
            leaf_slots.push(index.allocate(node));
        }
        let mut node = DNode::with_first(
            1,
            Child::Node(leaf_slots[0]),
            index.arena[leaf_slots[0]].rect,
        );
        for &s in &leaf_slots[1..] {
            node.push(Child::Node(s));
        }
        node.rect = node.exact_box(&index);
        group_slots.push(index.allocate(node));
    }
    let mut root = DNode::with_first(
        2,
        Child::Node(group_slots[0]),
        index.arena[group_slots[0]].rect,
    );
    for &s in &group_slots[1..] {
        root.push(Child::Node(s));
    }
    root.rect = root.exact_box(&index);
    index.root = Some(index.allocate(root));
    index.len = model.len();
    check_invariants(&index, &model);
    (index, model)
}

/// A leaf forced below the minimum fill is detached and its entries
/// replanted; nothing above it underflows.
#[test]
fn a_leaf_under_the_minimum_is_condensed() {
    let m = DynamicIndex::MIN_FILL;
    let (mut index, mut model) = handmade(2, m + 1, m);
    let root_before = index.root;
    let (b, id) = model[0];
    remove_checked(&mut index, &mut model, &b, id);
    assert_eq!(index.root, root_before, "the root changed without cause");
    assert_eq!(index.arena[index.root.unwrap()].level, 2);
    assert_eq!(index.free.len(), 1, "exactly the underfull leaf was freed");
}

/// Underflow cascading up two levels: the leaf drops below the minimum, and
/// so does its parent once the leaf is detached; the parent's leaves are
/// replanted at their own level, the root is left with one child and the
/// tree is shortened.
#[test]
fn underflow_cascades_two_levels_and_the_root_shrinks() {
    let m = DynamicIndex::MIN_FILL;
    let (mut index, mut model) = handmade(2, m, m);
    let (b, id) = model[0];
    remove_checked(&mut index, &mut model, &b, id);
    let root = index.root.unwrap();
    // Three slots freed: the leaf, its parent, and the old root.
    assert_eq!(index.free.len(), 3);
    assert_eq!(index.arena[root].level, 1, "the tree was not shortened");
    assert_eq!(
        index.arena[root].len(),
        2 * m - 1,
        "the leaves were not replanted at their level"
    );
    let mut rng = Rng::new(0x5EED_D606);
    check_queries(&index, &model, &mut rng);
}

/// The root shrinking to one child, then to a leaf, then to nothing.
#[test]
fn the_root_shrinks_to_a_leaf_and_then_empties() {
    let mut rng = Rng::new(0x5EED_D607);
    let source: Vec<(Rect, u32)> = lattice(5).into_iter().take(17).collect();
    let mut index = DynamicIndex::new();
    for (b, id) in &source {
        index.insert(*b, *id);
    }
    assert_eq!(
        index.arena[index.root.unwrap()].level,
        1,
        "17 entries split the root"
    );
    let mut model = source.clone();
    let mut saw_leaf_root = false;
    for (b, id) in &source {
        remove_checked(&mut index, &mut model, b, *id);
        if let Some(root) = index.root {
            saw_leaf_root |= index.arena[root].level == 0;
        }
        check_queries(&index, &model, &mut rng);
    }
    assert!(saw_leaf_root, "the root never became a leaf");
    assert_eq!(index.root, None);
}

// ===========================================================================
// Removal-path defect classes
// ===========================================================================

/// Cached bounds that never shrink are a defect class of R-tree removal.
/// Here `bounds()` is the exact union of what remains after every removal,
/// including when the extreme entries leave first.
#[test]
fn dynamic_stale_bounds_after_removal() {
    let mut source = lattice(20);
    source.push((r(-1e300, -1e300, -1e300, -1e300), 9_000));
    source.push((r(1e300, 1e300, 1e300, 1e300), 9_001));
    let mut index = DynamicIndex::bulk_load(source.clone());
    let mut model = source.clone();
    remove_checked(
        &mut index,
        &mut model,
        &r(1e300, 1e300, 1e300, 1e300),
        9_001,
    );
    remove_checked(
        &mut index,
        &mut model,
        &r(-1e300, -1e300, -1e300, -1e300),
        9_000,
    );
    assert_eq!(index.bounds(), Some(r(0.0, 0.0, 20.0, 20.0)));
    // Peel the lattice from the outside in: the bounds follow exactly.
    let mut order = model.clone();
    order.sort_by(|a, b| {
        let key = |e: &(Rect, u32)| (e.0.min_x().max(e.0.min_y())) as i64;
        key(b).cmp(&key(a)).then(a.1.cmp(&b.1))
    });
    for (b, id) in order.iter().take(300) {
        remove_checked(&mut index, &mut model, b, *id);
    }
}

/// Condensing leaves no undersized node and no leaves at different depths,
/// the loader defect above reached through removal. Every count from a bulk
/// load at the sizes above is drained in a scattered order, with the
/// invariants checked after every removal.
#[test]
fn dynamic_condense_keeps_fill_and_depth() {
    let mut rng = Rng::new(0x5EED_D608);
    for n in (7..=11).chain(25..=47).chain([255, 256, 257, 300]) {
        let mut source = lattice(18);
        source.truncate(n);
        let index = DynamicIndex::bulk_load(source.clone());
        let mut order = source.clone();
        for i in (1..order.len()).rev() {
            order.swap(i, (rng.next() as usize) % (i + 1));
        }
        drain(index, source, &order, &mut rng);
    }
}

/// Draining a tree to empty can overflow an empty-tree sentinel and panic.
/// Here an emptied tree has no root, no bounds and no sentinel, and removal
/// from it is simply `false`, even with extreme boxes.
#[test]
fn dynamic_drain_to_empty_is_safe() {
    let mut rng = Rng::new(0x5EED_D609);
    let source = vec![
        (r(-f64::MAX, -f64::MAX, f64::MAX, f64::MAX), 0),
        (r(f64::MAX, f64::MAX, f64::MAX, f64::MAX), 1),
        (r(-f64::MAX, -f64::MAX, -f64::MAX, -f64::MAX), 2),
        (r(0.0, 0.0, f64::MIN_POSITIVE, f64::MIN_POSITIVE), 3),
    ];
    let index = DynamicIndex::bulk_load(source.clone());
    drain(index, source.clone(), &source, &mut rng);
    let mut empty = DynamicIndex::new();
    for (b, id) in &source {
        assert!(!empty.remove(b, *id));
    }
    check_invariants(&empty, &[]);
}

/// A total order that tells `-0.0` from `0.0` makes boxes touching at the
/// origin miss each other. Removal uses the same `==` as every query, so the
/// sign of a zero never hides an entry from removal, and the bounds after
/// removal treat the zeros alike.
#[test]
fn dynamic_signed_zero_never_hides_an_entry() {
    let mut index = DynamicIndex::new();
    let mut model = Vec::new();
    for i in 0..40u32 {
        let z = if i % 2 == 0 { 0.0 } else { -0.0 };
        let b = r(z, z, 1.0 + f64::from(i), 1.0);
        index.insert(b, i);
        model.push((b, i));
    }
    for i in 0..40u32 {
        let z = if i % 2 == 0 { -0.0 } else { 0.0 }; // the other sign
        remove_checked(&mut index, &mut model, &r(z, z, 1.0 + f64::from(i), 1.0), i);
    }
    assert!(index.is_empty());
}

/// Removal from a bulk-loaded tree complements the insertion test above. A
/// load at the same counts accepts removals and insertions interleaved.
#[test]
fn dynamic_remove_after_bulk_load() {
    let mut rng = Rng::new(0x5EED_D60A);
    for n in [7usize, 11, 25, 47, 100, 257] {
        let mut source = lattice(17);
        source.truncate(n);
        let mut index = DynamicIndex::bulk_load(source.clone());
        let mut model = source.clone();
        for step in 0..60u32 {
            if step % 3 == 0 || model.is_empty() {
                let x = (rng.int(20) as f64) * 0.5;
                let b = r(x, x, x + 1.5, x + 1.5);
                index.insert(b, 1_000 + step);
                model.push((b, 1_000 + step));
                check_invariants(&index, &model);
            } else {
                let (b, id) = model[(rng.next() as usize) % model.len()];
                remove_checked(&mut index, &mut model, &b, id);
            }
        }
        check_queries(&index, &model, &mut rng);
    }
}

/// Coordinates past half the maximum overflow the center arithmetic during
/// reinsertion, and condensing replants through the same heuristics. They
/// may degenerate, but nothing panics and the invariants hold.
#[test]
fn dynamic_extreme_magnitudes_through_condensing() {
    let mut rng = Rng::new(0x5EED_D60B);
    let huge = f64::MAX / 2.0;
    let source: Vec<(Rect, u32)> = (0..120u32)
        .map(|i| {
            let s = if i % 2 == 0 { 1.0 } else { -1.0 };
            let base = s * (huge + f64::from(i % 5) * 1e300);
            (r(base.min(s * huge), -1e308, base.max(s * huge), 1e308), i)
        })
        .collect();
    let mut index = DynamicIndex::new();
    for (b, id) in &source {
        index.insert(*b, *id);
    }
    let mut order = source.clone();
    for i in (1..order.len()).rev() {
        order.swap(i, (rng.next() as usize) % (i + 1));
    }
    drain(index, source, &order, &mut rng);
}

// ===========================================================================
// The model-based property suite
// ===========================================================================

proptest! {
    // 5,000 in release, 500 in debug, as for the insertion sweep.
    #![proptest_config(ProptestConfig::with_cases(crate::tests::CASES / 2))]

    /// Up to 2,000 interleaved operations from the operation mix, from an
    /// empty tree or a bulk load of up to 200 entries, with every check after
    /// every operation (`drive_ops`).
    #[test]
    fn interleaved_operations_match_the_model(
        start in entries(200),
        loaded in any::<bool>(),
        ops in prop::collection::vec(op(), 0..=2_000),
    ) {
        let (mut index, mut model) = if loaded {
            (DynamicIndex::bulk_load(start.clone()), start)
        } else {
            (DynamicIndex::new(), Vec::new())
        };
        check_invariants(&index, &model);
        drive_ops(&mut index, &mut model, &ops);
    }
}

// ===========================================================================
// Determinism
// ===========================================================================

/// FNV-1a over one node's identity: box bits, level, and children (slots,
/// or entry bits).
fn arena_digest(index: &DynamicIndex) -> u64 {
    let mut hash: u64 = 0xCBF2_9CE4_8422_2325;
    let mut eat = |v: u64| {
        for b in v.to_le_bytes() {
            hash = (hash ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01B3);
        }
    };
    eat(index.arena.len() as u64);
    eat(index.root.map_or(u64::MAX, |s| s as u64));
    let mut live = vec![false; index.arena.len()];
    let mut stack: Vec<usize> = index.root.into_iter().collect();
    while let Some(slot) = stack.pop() {
        live[slot] = true;
        let node = &index.arena[slot];
        for child in node.children() {
            if let Child::Node(s) = *child {
                stack.push(s);
            }
        }
    }
    for (slot, node) in index.arena.iter().enumerate() {
        if !live[slot] {
            eat(u64::MAX);
            continue;
        }
        eat(slot as u64);
        eat(u64::from(node.level));
        eat(node.rect.min_x().to_bits());
        eat(node.rect.min_y().to_bits());
        eat(node.rect.max_x().to_bits());
        eat(node.rect.max_y().to_bits());
        for child in node.children() {
            match *child {
                Child::Node(s) => eat(s as u64),
                Child::Entry(rect, id) => {
                    eat(rect.min_x().to_bits());
                    eat(rect.max_y().to_bits());
                    eat(u64::from(id));
                }
            }
        }
    }
    hash
}

/// The fixed 2,000-insertion sequence of the determinism test.
fn determinism_sequence() -> Vec<(Rect, u32)> {
    let mut rng = Rng::new(0x5EED_D10A);
    (0..2_000u32)
        .map(|i| {
            let x = (rng.int(400) as f64) * 0.5;
            let y = (rng.int(400) as f64) * 0.5;
            let w = (rng.next() % 8) as f64 * 0.5;
            let h = (rng.next() % 8) as f64 * 0.5;
            (r(x, y, x + w, y + h), i)
        })
        .collect()
}

/// The recorded layout of the fixed sequence. Any change to a heuristic or
/// a tie-break moves this digest, so the change becomes a visible decision.
const DETERMINISM_DIGEST: u64 = 0x65D9_8D1E_2466_5ACD;
const DETERMINISM_NODES: usize = 187;
const DETERMINISM_ROOT: usize = 19;
const DETERMINISM_LEVEL: u32 = 2;

#[test]
fn the_same_operations_give_the_same_arena() {
    let source = determinism_sequence();
    let build = || {
        let mut index = DynamicIndex::new();
        for (rect, id) in &source {
            index.insert(*rect, *id);
        }
        index
    };
    let (a, b) = (build(), build());
    assert_eq!(
        arena_digest(&a),
        arena_digest(&b),
        "two identical runs differ"
    );
    assert_eq!(a.arena.len(), b.arena.len());
    // Against the recorded values.
    assert_eq!(a.arena.len(), DETERMINISM_NODES, "the arena layout changed");
    assert_eq!(a.root, Some(DETERMINISM_ROOT));
    assert_eq!(a.arena[a.root.unwrap()].level, DETERMINISM_LEVEL);
    assert_eq!(
        arena_digest(&a),
        DETERMINISM_DIGEST,
        "the layout digest changed"
    );
    check_invariants(&a, &source);
}

/// The fixed 2,000-operation interleaved sequence of the extended
/// determinism test: insertions, removals of stored entries (chosen by a
/// fixed generator from the live set), and removals of absent ones, in about
/// the model suite's proportions.
fn interleaved_sequence() -> DynamicIndex {
    let mut rng = Rng::new(0x5EED_D60C);
    let mut index = DynamicIndex::new();
    let mut live: Vec<(Rect, u32)> = Vec::new();
    for i in 0..2_000u32 {
        let roll = rng.next() % 10;
        if roll < 6 || live.is_empty() {
            let x = (rng.int(400) as f64) * 0.5;
            let y = (rng.int(400) as f64) * 0.5;
            let w = (rng.next() % 8) as f64 * 0.5;
            let h = (rng.next() % 8) as f64 * 0.5;
            let b = r(x, y, x + w, y + h);
            index.insert(b, i);
            live.push((b, i));
        } else if roll < 9 {
            let at = (rng.next() as usize) % live.len();
            let (b, id) = live.swap_remove(at);
            assert!(index.remove(&b, id));
        } else {
            assert!(!index.remove(&r(-1.0, -1.0, -0.5, -0.5), i));
        }
    }
    check_invariants(&index, &live);
    index
}

/// The recorded layout after the interleaved sequence.
const INTERLEAVED_DIGEST: u64 = 0x1617_8E5C_362D_7019;
const INTERLEAVED_NODES: usize = 62;
const INTERLEAVED_ROOT: usize = 19;
const INTERLEAVED_LEVEL: u32 = 2;
const INTERLEAVED_FREE: usize = 0;

#[test]
fn the_same_interleaved_operations_give_the_same_arena() {
    let (a, b) = (interleaved_sequence(), interleaved_sequence());
    assert_eq!(
        arena_digest(&a),
        arena_digest(&b),
        "two identical runs differ"
    );
    assert_eq!(a.free, b.free, "the free lists differ");
    assert_eq!(a.arena.len(), INTERLEAVED_NODES, "the arena layout changed");
    assert_eq!(a.root, Some(INTERLEAVED_ROOT));
    assert_eq!(a.arena[a.root.unwrap()].level, INTERLEAVED_LEVEL);
    assert_eq!(a.free.len(), INTERLEAVED_FREE);
    assert_eq!(
        arena_digest(&a),
        INTERLEAVED_DIGEST,
        "the layout digest changed"
    );
}

// ===========================================================================
// Performance, recorded
// ===========================================================================

/// Ceilings set from a recorded release run, with a margin above it.
const CEILING_INSERT_SECONDS: f64 = 2.0;
const CEILING_QUERY_RATIO: f64 = 2.2;
const CEILING_REMOVE_SECONDS: f64 = 0.6;
const CEILING_QUERY_RATIO_AFTER: f64 = 2.0;

/// 100,000 random boxes inserted one at a time, then 10,000 window queries
/// against the same queries on a static index of the same entries; then
/// 50,000 of the entries removed in a scattered order, and the queries again
/// against a static index of the survivors. Release only. It prints all four
/// numbers.
#[test]
#[ignore = "timing; run in release with --ignored dynamic_perf"]
fn dynamic_perf() {
    let mut rng = Rng::new(0x5EED_D10B);
    let source: Vec<(Rect, u32)> = (0..100_000u32)
        .map(|i| {
            let x = (rng.next() % 2_000_000) as f64 / 1_000.0;
            let y = (rng.next() % 2_000_000) as f64 / 1_000.0;
            let w = (rng.next() % 12_000) as f64 / 1_000.0;
            let h = (rng.next() % 12_000) as f64 / 1_000.0;
            (r(x, y, x + w, y + h), i)
        })
        .collect();
    let windows: Vec<Rect> = (0..10_000)
        .map(|_| {
            let x = (rng.next() % 1_990_000) as f64 / 1_000.0;
            let y = (rng.next() % 1_990_000) as f64 / 1_000.0;
            (x, y)
        })
        .map(|(x, y)| r(x, y, x + 10.0, y + 10.0))
        .collect();

    let t = std::time::Instant::now();
    let mut dynamic = DynamicIndex::new();
    for (rect, id) in &source {
        dynamic.insert(*rect, *id);
    }
    let insert_time = t.elapsed().as_secs_f64();

    let static_index = crate::Index::bulk_load(source.clone());
    fn best_of<F: FnMut(&Rect, &mut u64)>(windows: &[Rect], mut f: F) -> (f64, u64) {
        let mut best = f64::INFINITY;
        let mut sum = 0u64;
        for _ in 0..5 {
            let mut h = 0u64;
            let t = std::time::Instant::now();
            for w in windows {
                f(w, &mut h);
            }
            best = best.min(t.elapsed().as_secs_f64());
            sum = core::hint::black_box(h);
        }
        (best, sum)
    }
    let (dyn_time, dyn_sum) = best_of(&windows, |w, h| {
        dynamic.query(w, |id| *h += u64::from(id) | 1);
    });
    let (stat_time, stat_sum) = best_of(&windows, |w, h| {
        static_index.query(w, |id| *h += u64::from(id) | 1);
    });
    assert_eq!(dyn_sum, stat_sum, "the two trees disagree");
    let ratio = dyn_time / stat_time;

    // Remove every other entry, in a scattered order, then run the same
    // queries against a static index of the survivors.
    let mut order: Vec<usize> = (0..source.len()).step_by(2).collect();
    for i in (1..order.len()).rev() {
        order.swap(i, (rng.next() as usize) % (i + 1));
    }
    let t = std::time::Instant::now();
    for &i in &order {
        let (rect, id) = source[i];
        assert!(dynamic.remove(&rect, id));
    }
    let remove_time = t.elapsed().as_secs_f64();
    let survivors: Vec<(Rect, u32)> = source.iter().skip(1).step_by(2).copied().collect();
    assert_eq!(dynamic.len(), survivors.len());
    let static_after = crate::Index::bulk_load(survivors);
    let (dyn_after, dyn_sum) = best_of(&windows, |w, h| {
        dynamic.query(w, |id| *h += u64::from(id) | 1);
    });
    let (stat_after, stat_sum) = best_of(&windows, |w, h| {
        static_after.query(w, |id| *h += u64::from(id) | 1);
    });
    assert_eq!(dyn_sum, stat_sum, "the two trees disagree after removals");
    let ratio_after = dyn_after / stat_after;

    std::println!(
        "dynamic_perf: 100,000 inserts in {insert_time:.3} s (ceiling {CEILING_INSERT_SECONDS}); \
         10,000 queries dynamic {dyn_time:.4} s / static {stat_time:.4} s = {ratio:.3} (ceiling {CEILING_QUERY_RATIO})"
    );
    std::println!(
        "dynamic_perf: 50,000 removals in {remove_time:.3} s (ceiling {CEILING_REMOVE_SECONDS}); \
         after removals dynamic {dyn_after:.4} s / static {stat_after:.4} s = {ratio_after:.3} (ceiling {CEILING_QUERY_RATIO_AFTER})"
    );
    assert!(
        insert_time <= CEILING_INSERT_SECONDS,
        "insert time {insert_time:.3} s over its ceiling"
    );
    assert!(
        ratio <= CEILING_QUERY_RATIO,
        "query ratio {ratio:.3} over its ceiling"
    );
    assert!(
        remove_time <= CEILING_REMOVE_SECONDS,
        "removal time {remove_time:.3} s over its ceiling"
    );
    assert!(
        ratio_after <= CEILING_QUERY_RATIO_AFTER,
        "query ratio after removals {ratio_after:.3} over its ceiling"
    );
}

// ===========================================================================
// Remaining properties
// ===========================================================================

/// The arena's overflow slot: a node can briefly hold capacity + 1, and the
/// constant in the node module says exactly that.
#[test]
fn the_overflow_slot_is_one_past_capacity() {
    assert_eq!(super::SLOTS, DynamicIndex::NODE_CAPACITY + 1);
    assert_eq!(DynamicIndex::MIN_FILL, 6);
    assert_eq!(DynamicIndex::REINSERT_COUNT, 5);
    const { assert!(DynamicIndex::NODE_CAPACITY >= 2 * DynamicIndex::MIN_FILL) }
}

/// Cloning is a deep copy: the clone evolves on its own.
#[test]
fn a_clone_is_independent() {
    let mut a = DynamicIndex::bulk_load(lattice(5));
    let model: Vec<(Rect, u32)> = lattice(5);
    let mut b = a.clone();
    b.insert(r(100.0, 100.0, 101.0, 101.0), 999);
    check_invariants(&a, &model);
    let mut model_b = model.clone();
    model_b.push((r(100.0, 100.0, 101.0, 101.0), 999));
    check_invariants(&b, &model_b);
    a.insert(r(-5.0, -5.0, -4.0, -4.0), 998);
    let mut model_a = model;
    model_a.push((r(-5.0, -5.0, -4.0, -4.0), 998));
    check_invariants(&a, &model_a);
}

/// Prints the recorded values for the determinism and performance tests.
#[test]
#[ignore = "prints the values the recorded constants pin"]
fn print_recorded_values() {
    let source = determinism_sequence();
    let mut index = DynamicIndex::new();
    for (rect, id) in &source {
        index.insert(*rect, *id);
    }
    std::println!(
        "DETERMINISM nodes={} root={} level={} digest={:#018x}",
        index.arena.len(),
        index.root.unwrap(),
        index.arena[index.root.unwrap()].level,
        arena_digest(&index)
    );
    let index = interleaved_sequence();
    std::println!(
        "INTERLEAVED nodes={} root={} level={} free={} digest={:#018x}",
        index.arena.len(),
        index.root.unwrap(),
        index.arena[index.root.unwrap()].level,
        index.free.len(),
        arena_digest(&index)
    );
}
