// fuzz/fuzz_targets/join.rs

#![no_main]

//! `Index::join` and `Index::self_join` against brute force. The input bytes
//! decode to up to 512 boxes (`common`), split into two sets `A` and `B` at a
//! point the first byte chooses. The target asserts:
//!
//! - no panic;
//! - `join(A, B)`, as a sorted multiset of id pairs, equals every pair of
//!   `A × B` whose boxes meet by the closed test written out from the
//!   bounds (`common::meets`);
//! - `join(B, A)`, each pair swapped, equals `join(A, B)`;
//! - `self_join(A)`, each pair's ids sorted, equals every unordered pair of
//!   distinct stored entries of `A` whose boxes meet; and no pair repeats an
//!   entry with itself.
//!
//! Ids are input positions, so every entry's id is unique and a self-pair
//! would show as `(i, i)`.

mod common;

use common::{MAX_BOXES, meets, rect};
use libfuzzer_sys::fuzz_target;
use situs::{Index, Rect};

fn sorted(mut v: Vec<(u32, u32)>) -> Vec<(u32, u32)> {
    v.sort_unstable();
    v
}

fuzz_target!(|data: &[u8]| {
    let Some((&head, rest)) = data.split_first() else {
        return;
    };
    let boxes: Vec<(Rect, u32)> = rest
        .chunks_exact(4)
        .take(MAX_BOXES)
        .enumerate()
        .map(|(i, b)| (rect(b), u32::try_from(i).expect("at most 512 boxes")))
        .collect();
    let split = usize::from(head) * boxes.len() / 255;
    let (a_entries, b_entries) = boxes.split_at(split);
    let a = Index::bulk_load(a_entries.to_vec());
    let b = Index::bulk_load(b_entries.to_vec());

    let mut want = Vec::new();
    for (ra, ia) in a_entries {
        for (rb, ib) in b_entries {
            if meets(ra, rb) {
                want.push((*ia, *ib));
            }
        }
    }
    let want = sorted(want);
    let mut got = Vec::new();
    a.join(&b, |x, y| got.push((x, y)));
    assert_eq!(sorted(got), want, "join(A, B) differs from brute force");
    let mut swapped = Vec::new();
    b.join(&a, |y, x| swapped.push((x, y)));
    assert_eq!(
        sorted(swapped),
        want,
        "join(B, A) swapped differs from join(A, B)"
    );

    let mut want = Vec::new();
    for (i, (ri, ii)) in a_entries.iter().enumerate() {
        for (rj, ij) in &a_entries[i + 1..] {
            if meets(ri, rj) {
                want.push((*ii.min(ij), *ii.max(ij)));
            }
        }
    }
    let mut got = Vec::new();
    a.self_join(|x, y| {
        assert_ne!(x, y, "self_join paired entry {x} with itself");
        got.push((x.min(y), x.max(y)));
    });
    assert_eq!(
        sorted(got),
        sorted(want),
        "self_join(A) differs from brute force"
    );
});
