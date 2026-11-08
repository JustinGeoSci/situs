// fuzz/fuzz_targets/dynamic_ops.rs

#![no_main]

//! `DynamicIndex` under an operation sequence decoded from the input,
//! checked against a model.
//!
//! The model is the plain vector of entries that should be stored. Each
//! operation takes six bytes: an opcode, four bytes for a box
//! (`common::rect`), and one for an id or an index. The opcode, modulo 16,
//! selects the operation:
//!
//! - 0–6, insert the box under a small id (0–31, so equal boxes under
//!   equal and unequal ids both occur);
//! - 7–10, remove an entry the model holds;
//! - 11, remove the box under a small id (usually absent; the model
//!   decides);
//! - 12, remove an entry's box under another id;
//! - 13–14, query the box as a window against brute force;
//! - 15, nearest-neighbor candidates and browsing at a point.
//!
//! Every operation is followed by four checks. The first is the tree's own
//! invariant check, `check_invariants`, compiled under `cfg(fuzzing)`,
//! which covers exact node boxes bit for bit, uniform depth, fill bounds,
//! the exact stored multiset, and `len` and `bounds()` against the model.
//! The others compare `remove`'s return value with the model's, a query's
//! ids with brute force, and the nearest-neighbor candidate set with a
//! static index over the same entries.

mod common;

use common::{meets, rect};
use libfuzzer_sys::fuzz_target;
use situs::{DynamicIndex, Index, Neighbor, Rect};

/// The most operations decoded from one input.
const MAX_OPS: usize = 4096;

fn bits(b: &Rect, id: u32) -> (u64, u64, u64, u64, u32) {
    (
        b.min_x().to_bits(),
        b.min_y().to_bits(),
        b.max_x().to_bits(),
        b.max_y().to_bits(),
        id,
    )
}

/// Removes `(b, id)` from both and checks the return value. Equal entries
/// may differ in the sign of a zero bound, so the model removes the one the
/// tree no longer holds and tracks the tree bit for bit.
fn remove(index: &mut DynamicIndex, model: &mut Vec<(Rect, u32)>, b: &Rect, id: u32) {
    let matches: Vec<usize> = (0..model.len())
        .filter(|&i| model[i].1 == id && model[i].0 == *b)
        .collect();
    let got = index.remove(b, id);
    assert_eq!(got, !matches.is_empty(), "remove({b:?}, {id}) returned {got}");
    if !got {
        return;
    }
    let first = bits(&model[matches[0]].0, id);
    if matches.iter().all(|&i| bits(&model[i].0, id) == first) {
        model.swap_remove(matches[0]);
        return;
    }
    let stored = index.stored_entries();
    for &i in &matches {
        let want = bits(&model[i].0, id);
        let in_model = model.iter().filter(|(m, mid)| bits(m, *mid) == want).count();
        let in_tree = stored.iter().filter(|(m, mid)| bits(m, *mid) == want).count();
        if in_tree < in_model {
            model.swap_remove(i);
            return;
        }
    }
    panic!("the tree reported a removal but holds every matching entry");
}

/// A candidate result as a sorted multiset of `(lower bits, upper bits, id)`.
fn as_set(found: &[Neighbor]) -> Vec<(u64, u64, u32)> {
    let mut set: Vec<(u64, u64, u32)> = found
        .iter()
        .map(|n| (n.lower_sq.to_bits(), n.upper_sq.to_bits(), n.id))
        .collect();
    set.sort_unstable();
    set
}

fuzz_target!(|data: &[u8]| {
    let mut index = DynamicIndex::new();
    let mut model: Vec<(Rect, u32)> = Vec::new();
    for op in data.chunks_exact(6).take(MAX_OPS) {
        let b = rect(&op[1..5]);
        let small = u32::from(op[5] % 32);
        match op[0] % 16 {
            0..=6 => {
                index.insert(b, small);
                model.push((b, small));
            }
            7..=10 => {
                if !model.is_empty() {
                    let (e, id) = model[usize::from(op[5]) % model.len()];
                    remove(&mut index, &mut model, &e, id);
                }
            }
            11 => remove(&mut index, &mut model, &b, small),
            12 => {
                if !model.is_empty() {
                    let (e, id) = model[usize::from(op[5]) % model.len()];
                    remove(&mut index, &mut model, &e, id.wrapping_add(1));
                }
            }
            13 | 14 => {
                let mut got = Vec::new();
                index.query(&b, |id| got.push(id));
                got.sort_unstable();
                let mut want: Vec<u32> = model
                    .iter()
                    .filter(|(e, _)| meets(e, &b))
                    .map(|(_, id)| *id)
                    .collect();
                want.sort_unstable();
                assert_eq!(got, want, "query {b:?}");
            }
            _ => {
                let (x, y) = (b.min_x(), b.max_y());
                let k = usize::from(op[5] % 8);
                let static_index = Index::bulk_load(model.clone());
                let want = as_set(&static_index.nearest_candidates(x, y, k).expect("finite point"));
                let got = as_set(&index.nearest_candidates(x, y, k).expect("finite point"));
                assert_eq!(got, want, "candidates differ from the static index at k = {k}");
                let walked: Vec<Neighbor> = index.browse(x, y).expect("finite point").collect();
                assert_eq!(walked.len(), model.len(), "browse is not complete");
                for w in walked.windows(2) {
                    assert!(w[0].lower_sq <= w[1].lower_sq, "browse went backward: {w:?}");
                }
                let all = as_set(&walked);
                let want_all = as_set(&static_index.browse(x, y).expect("finite point").collect::<Vec<_>>());
                assert_eq!(all, want_all, "browse multiset differs from the entries");
            }
        }
        index.check_invariants(&model);
    }
});
