// src/static_index/tests.rs

//! Tests of the static index.
//!
//! The crate's one obligation is no false negatives (crate doc), and a query
//! in fact returns exactly the entries whose boxes meet the window. Every
//! test here compares [`Index::query`] with brute force over the same
//! entries, as sorted id lists compared for equality, so a missing id is a
//! false negative and an extra id is a defect as well.
//!
//! The three correctness conditions are attacked separately. A mutation
//! test breaks each one in turn and names the test that fails: `intersects`
//! made open rather than closed, a node box taken from its children's
//! centers, and a leaf box taken from its first entry only.

#![allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]

use super::*;
use crate::tests::{CASES, r, rect};
use core::cmp::Ordering;
use proptest::prelude::*;
use std::vec;
use std::vec::Vec;

// ===========================================================================
// Builders and the oracle
// ===========================================================================

/// Brute force, every entry whose box meets `window`, ids sorted.
pub(super) fn brute(entries: &[(Rect, u32)], window: &Rect) -> Vec<u32> {
    let mut ids: Vec<u32> = entries
        .iter()
        .filter(|(rect, _)| rect.intersects(window))
        .map(|(_, id)| *id)
        .collect();
    ids.sort_unstable();
    ids
}

/// The index's answer, ids sorted.
fn queried(index: &Index, window: &Rect) -> Vec<u32> {
    let mut ids = Vec::new();
    index.query(window, |id| ids.push(id));
    ids.sort_unstable();
    ids
}

/// Builds the index and checks every window against brute force, plus the
/// structural invariants.
fn check(entries: &[(Rect, u32)], windows: &[Rect]) {
    let index = Index::bulk_load(entries.to_vec());
    assert_eq!(index.len(), entries.len());
    assert_eq!(index.is_empty(), entries.is_empty());
    check_structure(&index);
    for w in windows {
        assert_eq!(queried(&index, w), brute(entries, w), "window {w:?}");
    }
}

/// The structural invariants behind conditions 2 and 3. Every node's box
/// contains its children's, every entry is in exactly one leaf, the levels
/// are packed, and the root bounds everything.
pub(super) fn check_structure(index: &Index) {
    let n = index.entries.len();
    if n == 0 {
        assert!(index.nodes.is_empty());
        assert_eq!(index.bounds(), None);
        return;
    }
    let mut seen = vec![0usize; n];
    for (i, node) in index.nodes.iter().enumerate() {
        assert!(node.first < node.end, "node {i} is empty");
        assert!(
            node.end - node.first <= Index::NODE_CAPACITY,
            "node {i} over capacity"
        );
        if node.leaf {
            for e in &index.entries[node.first..node.end] {
                assert!(
                    node.rect.contains_rect(&e.0),
                    "leaf {i} does not contain {e:?}"
                );
            }
            for s in &mut seen[node.first..node.end] {
                *s += 1;
            }
        } else {
            assert!(node.first > i, "node {i}'s children precede it");
            for (k, child) in index.nodes[node.first..node.end].iter().enumerate() {
                assert!(
                    node.rect.contains_rect(&child.rect),
                    "node {i} does not contain child {}",
                    node.first + k
                );
            }
        }
    }
    assert!(
        seen.iter().all(|&c| c == 1),
        "an entry is in no leaf or in two"
    );
    // Every level is packed, with at most one node per level not full. It
    // need not be the last, because a level is reordered by its own STR sort
    // after its nodes are cut, so the short node can land anywhere and the
    // nodes' child ranges are not in array order either. Level 0 is the
    // root, and level `L + 1` is the union of level `L`'s child ranges,
    // which is contiguous because the layout is level order.
    let (mut lo, mut hi) = (0usize, 1usize);
    loop {
        let short = index.nodes[lo..hi]
            .iter()
            .filter(|n| n.end - n.first < Index::NODE_CAPACITY)
            .count();
        assert!(short <= 1, "level {lo}..{hi} has {short} short nodes");
        if index.nodes[lo].leaf {
            assert_eq!(hi, index.nodes.len(), "nodes after the leaf level");
            break;
        }
        let level = &index.nodes[lo..hi];
        (lo, hi) = (
            level.iter().map(|n| n.first).min().unwrap(),
            level.iter().map(|n| n.end).max().unwrap(),
        );
    }
    let root = index.bounds().unwrap();
    for e in &index.entries {
        assert!(root.contains_rect(&e.0));
    }
}

// ===========================================================================
// Index: fixed cases
// ===========================================================================

#[test]
fn empty_index_returns_nothing() {
    let index = Index::bulk_load(Vec::new());
    assert!(index.is_empty());
    assert_eq!(index.len(), 0);
    assert_eq!(index.bounds(), None);
    let mut n = 0;
    index.query(&r(-1e300, -1e300, 1e300, 1e300), |_| n += 1);
    assert_eq!(n, 0);
}

#[test]
fn one_entry_is_its_own_root() {
    let index = Index::bulk_load(vec![(r(1.0, 2.0, 3.0, 4.0), 42)]);
    assert_eq!(index.bounds(), Some(r(1.0, 2.0, 3.0, 4.0)));
    assert_eq!(index.nodes.len(), 1);
    assert_eq!(queried(&index, &r(3.0, 4.0, 9.0, 9.0)), [42]);
    assert_eq!(queried(&index, &r(3.1, 4.0, 9.0, 9.0)), Vec::<u32>::new());
}

/// Unit squares on a lattice, ids in row-major order.
pub(crate) fn lattice(side: u32) -> Vec<(Rect, u32)> {
    (0..side * side)
        .map(|i| {
            let (x, y) = (f64::from(i % side), f64::from(i / side));
            (r(x, y, x + 1.0, y + 1.0), i)
        })
        .collect()
}

/// The sizes at which the packing changes shape: 0, 1, M, M + 1, M², and
/// 10,000.
#[test]
fn required_sizes_match_brute_force() {
    let m = Index::NODE_CAPACITY as u32;
    let windows = [
        r(0.0, 0.0, 0.0, 0.0),
        r(0.5, 0.5, 0.5, 0.5),
        r(1.0, 1.0, 1.0, 1.0),
        r(3.0, 2.0, 7.0, 9.0),
        r(-10.0, -10.0, 1000.0, 1000.0),
        r(2.5, 2.5, 2.5, 7.0),
        r(99.0, 99.0, 101.0, 101.0),
    ];
    for n in [0u32, 1, m, m + 1, m * m, 10_000] {
        // A near-square lattice of n squares: side = ⌈√n⌉, truncated to n.
        let side = ceil_sqrt(n as usize).max(1) as u32;
        let mut entries = lattice(side);
        entries.truncate(n as usize);
        assert_eq!(entries.len(), n as usize);
        check(&entries, &windows);
    }
}

/// Every entry count from 0 to 300, and the counts around `M^k` and
/// `S·M·S` up to 5,000, where the structure holds and three windows match
/// brute force. These counts exercise the loader's fan-out and depth
/// arithmetic at every boundary, and the sweep is cheap.
#[test]
fn every_small_count_builds_a_sound_tree() {
    let m = Index::NODE_CAPACITY;
    let mut counts: Vec<usize> = (0..=300).collect();
    for k in [m * m, m * m * m] {
        counts.extend([k - 1, k, k + 1]);
    }
    for s in 2..=5usize {
        let slice = s * m;
        counts.extend([slice * s - 1, slice * s, slice * s + 1, slice * (s + 1)]);
    }
    let windows = [
        r(0.0, 0.0, 0.0, 0.0),
        r(7.0, 7.0, 7.0, 7.0),
        r(-1.0, 3.5, 100.0, 3.5),
    ];
    for n in counts {
        let side = ceil_sqrt(n).max(1) as u32;
        let mut entries = lattice(side);
        entries.truncate(n);
        check(&entries, &windows);
    }
}

#[test]
fn lattice_tree_is_packed_at_every_level() {
    // 256 entries: 16 leaves of 16, one root of 16.
    let index = Index::bulk_load(lattice(16));
    assert_eq!(index.nodes.len(), 17);
    assert!(!index.nodes[0].leaf);
    assert_eq!((index.nodes[0].first, index.nodes[0].end), (1, 17));
    assert!(
        index.nodes[1..]
            .iter()
            .all(|n| n.leaf && n.end - n.first == 16)
    );
    check_structure(&index);
    // 257 entries: 17 leaves, 2 internal nodes, one root.
    let mut e = lattice(17);
    e.truncate(257);
    let index = Index::bulk_load(e);
    assert_eq!(index.nodes.len(), 1 + 2 + 17);
    check_structure(&index);
}

#[test]
fn identical_entries_are_all_returned() {
    let entries: Vec<(Rect, u32)> = (0..1000).map(|i| (r(1.0, 1.0, 2.0, 2.0), i)).collect();
    let index = Index::bulk_load(entries.clone());
    check_structure(&index);
    assert_eq!(
        queried(&index, &r(2.0, 2.0, 3.0, 3.0)),
        brute(&entries, &r(2.0, 2.0, 3.0, 3.0))
    );
    assert_eq!(queried(&index, &r(2.0, 2.0, 3.0, 3.0)).len(), 1000);
    assert!(queried(&index, &r(2.1, 2.0, 3.0, 3.0)).is_empty());
}

#[test]
fn duplicate_ids_are_reported_once_per_entry() {
    let entries = vec![(r(0.0, 0.0, 1.0, 1.0), 5), (r(0.0, 0.0, 1.0, 1.0), 5)];
    let index = Index::bulk_load(entries);
    assert_eq!(queried(&index, &r(0.5, 0.5, 0.5, 0.5)), [5, 5]);
}

#[test]
fn extreme_magnitudes_are_handled() {
    let big = 1e300;
    let tiny = f64::MIN_POSITIVE;
    let entries = vec![
        (r(-big, -big, -big, -big), 0),
        (r(big, big, big, big), 1),
        (r(-big, -big, big, big), 2),
        (r(0.0, 0.0, tiny, tiny), 3),
        (r(-tiny, -tiny, 0.0, 0.0), 4),
        (r(tiny, tiny, 2.0 * tiny, 2.0 * tiny), 5),
        (r(-big, 0.0, big, 0.0), 6),
    ];
    let windows = [
        r(-big, -big, -big, -big),
        r(big, big, big, big),
        r(0.0, 0.0, 0.0, 0.0),
        r(tiny, tiny, tiny, tiny),
        r(-tiny, -tiny, tiny, tiny),
        r(2.0 * tiny, 0.0, 1.0, 1.0),
        r(-big, -big, big, big),
    ];
    check(&entries, &windows);
}

#[test]
fn layout_is_deterministic_across_builds() {
    let entries = lattice(23);
    let a = Index::bulk_load(entries.clone());
    let b = Index::bulk_load(entries);
    assert_eq!(a.entries, b.entries);
    assert_eq!(a.nodes.len(), b.nodes.len());
    for (x, y) in a.nodes.iter().zip(&b.nodes) {
        assert_eq!(
            (x.rect, x.first, x.end, x.leaf),
            (y.rect, y.first, y.end, y.leaf)
        );
    }
    // Visit order is a function of the layout, so it is deterministic too.
    let w = r(3.0, 3.0, 9.0, 9.0);
    let (mut va, mut vb) = (Vec::new(), Vec::new());
    a.query(&w, |id| va.push(id));
    b.query(&w, |id| vb.push(id));
    assert_eq!(va, vb);
}

#[test]
fn ceil_sqrt_is_exact() {
    for n in 0..5000usize {
        let s = ceil_sqrt(n);
        assert!(s * s >= n, "{n}");
        assert!(s == 0 || (s - 1) * (s - 1) < n, "{n}");
    }
}

// ===========================================================================
// Index: property tests
// ===========================================================================

pub(crate) fn entries(max: usize) -> impl Strategy<Value = Vec<(Rect, u32)>> {
    prop::collection::vec(rect(), 0..=max).prop_map(|rs| {
        rs.into_iter()
            .enumerate()
            .map(|(i, r)| (r, i as u32))
            .collect()
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(CASES))]

    /// Up to 300 entries with 8 windows each, which with `CASES` runs makes
    /// more than 80,000 queries per sweep in release, each against brute
    /// force.
    #[test]
    fn query_equals_brute_force(es in entries(300), ws in prop::collection::vec(rect(), 1..=8)) {
        check(&es, &ws);
    }

    /// Larger sets, fewer of them, up to 2,000 entries, which pack into
    /// three levels.
    #[test]
    fn query_equals_brute_force_on_larger_sets(es in entries(2_000), w in rect()) {
        check(&es, &[w]);
    }

    /// All entries identical, or identical but for id, with a window that
    /// touches them.
    #[test]
    fn coincident_entries(rect in rect(), n in 1usize..=50, w in rect()) {
        let es: Vec<(Rect, u32)> = (0..n as u32).map(|i| (rect, i)).collect();
        check(&es, &[w, rect, r(rect.max_x(), rect.max_y(), rect.max_x(), rect.max_y())]);
    }

    /// Windows built to touch an entry exactly at an edge or a corner.
    #[test]
    fn touching_windows(es in entries(200), k in 0usize..200, dx in 0.0f64..=3.0, dy in 0.0f64..=3.0) {
        if es.is_empty() {
            return Ok(());
        }
        let (e, _) = es[k % es.len()];
        let ws = [
            // Corner contact at the top right.
            r(e.max_x(), e.max_y(), e.max_x() + dx, e.max_y() + dy),
            // Edge contact on the left.
            r(e.min_x() - dx, e.min_y(), e.min_x(), e.max_y()),
            // Just past the right edge, which must miss this entry.
            {
                let lo = e.max_x() + f64::EPSILON * e.max_x().abs().max(1.0);
                r(lo, e.min_y(), lo.max(e.max_x() + 1.0), e.max_y())
            },
        ];
        check(&es, &ws);
    }
}

// ===========================================================================
// The recorded layout
// ===========================================================================

// The values below are a recorded layout, committed so that any change that
// moves an entry or a node fails these tests. The lattice is pinned in full,
// and the 5,000 pseudo-random boxes by a digest of the same sequences.

/// `n` boxes from a fixed SplitMix64 stream, ids `0..n`. Coordinates are
/// mostly halves in `[-100, 100]`, so boxes often touch, coincide and tie on
/// their centers. One in ten is drawn from a table of extremes.
pub(crate) fn pseudo_random_boxes(n: u32) -> Vec<(Rect, u32)> {
    const EXTREMES: [f64; 8] = [
        1e300,
        -1e300,
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        0.0,
        -0.0,
        1e6,
        -1e6,
    ];
    let mut state: u64 = 0x5EED_CA1C_0123_4567;
    let mut next = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut coord = move || {
        let v = next();
        if v % 10 == 0 {
            EXTREMES[((v >> 8) % 8) as usize]
        } else {
            f64::from(((v >> 8) % 401) as u32) * 0.5 - 100.0
        }
    };
    (0..n)
        .map(|id| {
            let (a, b, c, d) = (coord(), coord(), coord(), coord());
            let (min_x, max_x) = if a <= c { (a, c) } else { (c, a) };
            let (min_y, max_y) = if b <= d { (b, d) } else { (d, b) };
            (r(min_x, min_y, max_x, max_y), id)
        })
        .collect()
}

/// FNV-1a over a byte stream.
fn fnv1a(bytes: impl IntoIterator<Item = u8>) -> u64 {
    bytes.into_iter().fold(0xCBF2_9CE4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01B3)
    })
}

fn rect_bytes(r: &Rect) -> impl Iterator<Item = u8> {
    [r.min_x(), r.min_y(), r.max_x(), r.max_y()]
        .into_iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
}

/// Digests of the node sequence `(rect bits, first, end, leaf)` and of the
/// leaf-order entry sequence `(rect bits, id)`.
fn digests(index: &Index) -> (u64, u64) {
    let nodes = fnv1a(index.nodes.iter().flat_map(|n| {
        rect_bytes(&n.rect)
            .chain((n.first as u64).to_le_bytes())
            .chain((n.end as u64).to_le_bytes())
            .chain([u8::from(n.leaf)])
            .collect::<Vec<u8>>()
    }));
    let entries = fnv1a(
        index
            .entries
            .iter()
            .flat_map(|(r, id)| rect_bytes(r).chain(id.to_le_bytes()).collect::<Vec<u8>>()),
    );
    (nodes, entries)
}

/// Leaf-order ids of `lattice(23)`.
const LATTICE_23_IDS: [u32; 529] = [
    0, 1, 2, 3, 4, 23, 24, 25, 26, 27, 46, 47, 48, 49, 50, 69, 70, 71, 72, 73, 92, 93, 94, 95, 115,
    116, 117, 118, 138, 139, 140, 141, 161, 162, 163, 164, 184, 185, 186, 187, 207, 208, 209, 210,
    230, 231, 232, 233, 253, 254, 255, 256, 276, 277, 278, 279, 299, 300, 301, 302, 322, 323, 324,
    325, 345, 346, 347, 348, 368, 369, 370, 371, 391, 392, 393, 394, 414, 415, 416, 417, 437, 438,
    439, 440, 460, 461, 462, 463, 483, 484, 485, 486, 506, 507, 508, 509, 5, 6, 7, 8, 28, 29, 30,
    31, 51, 52, 53, 54, 74, 75, 76, 77, 96, 97, 98, 99, 100, 119, 120, 121, 122, 123, 142, 143,
    144, 145, 146, 165, 166, 167, 168, 169, 188, 189, 190, 191, 211, 212, 213, 214, 234, 235, 236,
    237, 257, 258, 259, 260, 280, 281, 282, 283, 303, 304, 305, 306, 326, 327, 328, 329, 349, 350,
    351, 352, 372, 373, 374, 375, 395, 396, 397, 398, 418, 419, 420, 421, 441, 442, 443, 444, 464,
    465, 466, 467, 487, 488, 489, 490, 510, 511, 512, 513, 9, 10, 11, 12, 32, 33, 34, 35, 55, 56,
    57, 58, 78, 79, 80, 81, 101, 102, 103, 104, 124, 125, 126, 127, 147, 148, 149, 150, 170, 171,
    172, 173, 192, 193, 194, 195, 196, 215, 216, 217, 218, 219, 238, 239, 240, 241, 242, 261, 262,
    263, 264, 265, 284, 285, 286, 287, 307, 308, 309, 310, 330, 331, 332, 333, 353, 354, 355, 356,
    376, 377, 378, 379, 399, 400, 401, 402, 422, 423, 424, 425, 445, 446, 447, 448, 468, 469, 470,
    471, 491, 492, 493, 494, 514, 515, 516, 517, 13, 14, 15, 16, 36, 37, 38, 39, 59, 60, 61, 62,
    82, 83, 84, 85, 105, 106, 107, 108, 128, 129, 130, 131, 151, 152, 153, 154, 174, 175, 176, 177,
    197, 198, 199, 200, 220, 221, 222, 223, 243, 244, 245, 246, 266, 267, 268, 269, 288, 289, 290,
    291, 292, 311, 312, 313, 314, 315, 334, 335, 336, 337, 338, 357, 358, 359, 360, 361, 380, 381,
    382, 383, 403, 404, 405, 406, 426, 427, 428, 429, 449, 450, 451, 452, 472, 473, 474, 475, 495,
    496, 497, 498, 518, 519, 520, 521, 17, 18, 19, 20, 40, 41, 42, 43, 63, 64, 65, 66, 86, 87, 88,
    89, 109, 110, 111, 112, 132, 133, 134, 135, 155, 156, 157, 158, 178, 179, 180, 181, 201, 202,
    203, 204, 224, 225, 226, 227, 247, 248, 249, 250, 270, 271, 272, 273, 293, 294, 295, 296, 316,
    317, 318, 319, 339, 340, 341, 342, 362, 363, 364, 365, 384, 385, 386, 387, 388, 407, 408, 409,
    410, 411, 430, 431, 432, 433, 434, 453, 454, 455, 456, 457, 476, 477, 478, 479, 499, 500, 501,
    502, 522, 523, 524, 525, 21, 22, 44, 45, 67, 68, 90, 91, 113, 114, 136, 137, 159, 160, 182,
    183, 205, 206, 228, 229, 251, 252, 274, 275, 297, 298, 320, 321, 343, 344, 366, 367, 389, 390,
    412, 413, 435, 436, 458, 459, 480, 481, 482, 503, 504, 505, 526, 527, 528,
];

/// `(min_x, min_y, max_x, max_y, first, end, leaf)` of every node of
/// `lattice(23)`, level order, root first.
const LATTICE_23_NODES: [(f64, f64, f64, f64, usize, usize, bool); 38] = [
    (0.0, 0.0, 23.0, 23.0, 1, 4, false),
    (0.0, 0.0, 23.0, 12.0, 4, 20, false),
    (21.0, 8.0, 23.0, 23.0, 36, 38, false),
    (0.0, 11.0, 23.0, 23.0, 20, 36, false),
    (0.0, 0.0, 5.0, 4.0, 0, 16, true),
    (5.0, 0.0, 9.0, 4.0, 96, 112, true),
    (9.0, 0.0, 13.0, 4.0, 192, 208, true),
    (13.0, 0.0, 17.0, 4.0, 288, 304, true),
    (17.0, 0.0, 21.0, 4.0, 384, 400, true),
    (21.0, 0.0, 23.0, 8.0, 480, 496, true),
    (0.0, 3.0, 5.0, 7.0, 16, 32, true),
    (4.0, 4.0, 9.0, 8.0, 112, 128, true),
    (9.0, 4.0, 13.0, 8.0, 208, 224, true),
    (13.0, 4.0, 17.0, 8.0, 304, 320, true),
    (17.0, 4.0, 21.0, 8.0, 400, 416, true),
    (0.0, 7.0, 4.0, 11.0, 32, 48, true),
    (4.0, 7.0, 9.0, 11.0, 128, 144, true),
    (8.0, 8.0, 13.0, 12.0, 224, 240, true),
    (13.0, 8.0, 17.0, 12.0, 320, 336, true),
    (17.0, 8.0, 21.0, 12.0, 416, 432, true),
    (0.0, 11.0, 4.0, 15.0, 48, 64, true),
    (4.0, 11.0, 8.0, 15.0, 144, 160, true),
    (8.0, 11.0, 13.0, 15.0, 240, 256, true),
    (12.0, 12.0, 17.0, 16.0, 336, 352, true),
    (17.0, 12.0, 21.0, 16.0, 432, 448, true),
    (0.0, 15.0, 4.0, 19.0, 64, 80, true),
    (4.0, 15.0, 8.0, 19.0, 160, 176, true),
    (8.0, 15.0, 12.0, 19.0, 256, 272, true),
    (12.0, 15.0, 17.0, 19.0, 352, 368, true),
    (16.0, 16.0, 21.0, 20.0, 448, 464, true),
    (20.0, 16.0, 23.0, 23.0, 512, 528, true),
    (0.0, 19.0, 4.0, 23.0, 80, 96, true),
    (4.0, 19.0, 8.0, 23.0, 176, 192, true),
    (8.0, 19.0, 12.0, 23.0, 272, 288, true),
    (12.0, 19.0, 16.0, 23.0, 368, 384, true),
    (16.0, 19.0, 21.0, 23.0, 464, 480, true),
    (21.0, 8.0, 23.0, 16.0, 496, 512, true),
    (22.0, 22.0, 23.0, 23.0, 528, 529, true),
];

const RANDOM_5000_NODES: usize = 336;
const RANDOM_5000_NODE_DIGEST: u64 = 0x1EEE_B2C9_79FF_4783;
const RANDOM_5000_ENTRY_DIGEST: u64 = 0xBD7B_9078_B817_8C1F;

#[test]
fn layout_of_the_lattice_equals_the_recorded_layout() {
    let source = lattice(23);
    let index = Index::bulk_load(source.clone());
    let ids: Vec<u32> = index.entries.iter().map(|e| e.1).collect();
    assert_eq!(ids, LATTICE_23_IDS);
    for (rect, id) in &index.entries {
        assert_eq!(
            rect_bytes(rect).collect::<Vec<u8>>(),
            rect_bytes(&source[*id as usize].0).collect::<Vec<u8>>()
        );
    }
    assert_eq!(index.nodes.len(), LATTICE_23_NODES.len());
    for (i, (n, want)) in index.nodes.iter().zip(&LATTICE_23_NODES).enumerate() {
        let (a, b, c, d, first, end, leaf) = *want;
        let got = [
            n.rect.min_x(),
            n.rect.min_y(),
            n.rect.max_x(),
            n.rect.max_y(),
        ]
        .map(f64::to_bits);
        assert_eq!(got, [a, b, c, d].map(f64::to_bits), "node {i} box");
        assert_eq!(
            (n.first, n.end, n.leaf),
            (first, end, leaf),
            "node {i} range"
        );
    }
}

#[test]
fn layout_of_pseudo_random_boxes_equals_the_recorded_layout() {
    let index = Index::bulk_load(pseudo_random_boxes(5_000));
    assert_eq!(index.nodes.len(), RANDOM_5000_NODES);
    assert_eq!(
        digests(&index),
        (RANDOM_5000_NODE_DIGEST, RANDOM_5000_ENTRY_DIGEST)
    );
    check_structure(&index);
}

// ===========================================================================
// memory_size_for
// ===========================================================================

/// The counts of `every_small_count_builds_a_sound_tree`: 0 to 300, around
/// `M²` and `M³`, and around `S·M·S` for `S` in 2 to 5.
pub(super) fn small_counts() -> Vec<usize> {
    let m = Index::NODE_CAPACITY;
    let mut counts: Vec<usize> = (0..=300).collect();
    for k in [m * m, m * m * m] {
        counts.extend([k - 1, k, k + 1]);
    }
    for s in 2..=5usize {
        let slice = s * m;
        counts.extend([slice * s - 1, slice * s, slice * s + 1, slice * (s + 1)]);
    }
    counts
}

/// Built from an entry vector whose capacity equals its length, the index
/// holds exactly what `memory_size_for` reports, the entry vector and the
/// node vector, each at its capacity.
#[test]
fn memory_size_for_is_exact_on_every_small_count() {
    for n in small_counts() {
        let side = ceil_sqrt(n).max(1) as u32;
        let mut entries: Vec<(Rect, u32)> = Vec::with_capacity(n);
        entries.extend(lattice(side).into_iter().take(n));
        assert_eq!(entries.capacity(), entries.len(), "n = {n}");
        let index = Index::bulk_load(entries);
        let held = index.entries.capacity() * core::mem::size_of::<(Rect, u32)>()
            + index.nodes.capacity() * core::mem::size_of::<Node>();
        assert_eq!(Index::memory_size_for(n), held, "n = {n}");
        assert_eq!(
            index.nodes.len(),
            index.nodes.capacity(),
            "n = {n}: spare node capacity"
        );
        assert_eq!(node_count(n), index.nodes.len(), "n = {n}");
    }
}

/// The closed form, recomputed in `u128` where nothing overflows, agrees
/// with `memory_size_for` wherever the size fits in `usize`, and
/// `memory_size_for` saturates wherever it does not.
#[test]
fn memory_size_for_matches_the_closed_form_and_saturates() {
    let e = core::mem::size_of::<(Rect, u32)>() as u128;
    let node = core::mem::size_of::<Node>() as u128;
    let reference = |n: usize| -> u128 {
        let m = Index::NODE_CAPACITY as u128;
        let (mut level, mut total) = (0u128, 0u128);
        if n > 0 {
            level = (n as u128).div_ceil(m);
            total = level;
        }
        while level > 1 {
            level = level.div_ceil(m);
            total += level;
        }
        n as u128 * e + total * node
    };
    let max = usize::MAX as u128;
    let edge = usize::MAX / core::mem::size_of::<(Rect, u32)>();
    for n in [
        1usize << 20,
        1 << 32,
        1 << 40,
        edge / 2,
        edge - 1,
        edge,
        edge + 1,
        usize::MAX / 2,
        usize::MAX - 1,
        usize::MAX,
    ] {
        let want = reference(n);
        let got = Index::memory_size_for(n);
        if want <= max {
            assert_eq!(got as u128, want, "n = {n}");
        } else {
            assert_eq!(got, usize::MAX, "n = {n}");
        }
    }
    assert_eq!(Index::memory_size_for(usize::MAX), usize::MAX);
    assert_eq!(Index::memory_size_for(0), 0);
}

/// The STR order by stable sorts, each pass a stable sort by center and
/// then tie. It is the reference for `str_order`.
fn str_order_stable<T: Copy>(
    items: &mut [T],
    rect: impl Fn(&T) -> Rect + Copy,
    tie: impl Fn(&T, &T) -> Ordering + Copy,
) {
    let leaves = items.len().div_ceil(Index::NODE_CAPACITY);
    let slice_len = ceil_sqrt(leaves) * Index::NODE_CAPACITY;
    items.sort_by(|a, b| {
        rect(a)
            .center()
            .0
            .total_cmp(&rect(b).center().0)
            .then_with(|| tie(a, b))
    });
    for slice in items.chunks_mut(slice_len) {
        slice.sort_by(|a, b| {
            rect(a)
                .center()
                .1
                .total_cmp(&rect(b).center().1)
                .then_with(|| tie(a, b))
        });
    }
}

/// `str_order` reorders exactly as the stable sorts do, on boxes drawn
/// from a few bounds (so centers tie often), with repeated ids at the leaf
/// level and no tie-break above it, at sizes up to several slices, and
/// with signed zeros and negative centers, where the integer keys must keep
/// `total_cmp`'s order.
#[test]
fn str_order_equals_the_stable_sorts() {
    let mut state = 0x5EED_0005u64;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) as u32
    };
    for case in 0..200u32 {
        let n = 1 + (next() % 900) as usize;
        let entries: Vec<(Rect, u32)> = (0..n)
            .map(|_| {
                let (x, y) = (f64::from(next() % 6) - 3.0, f64::from(next() % 6) - 3.0);
                (
                    r(x, y, x + f64::from(next() % 3), y + f64::from(next() % 3)),
                    next() % 8,
                )
            })
            .collect();
        let (mut a, mut b) = (entries.clone(), entries.clone());
        str_order(&mut a, |e| e.0, |e| u64::from(e.1));
        str_order_stable(&mut b, |e| e.0, |x, y| x.1.cmp(&y.1));
        assert_eq!(a, b, "leaf level, case {case}");
        let (mut a, mut b) = (entries.clone(), entries);
        str_order(&mut a, |e| e.0, |_| 0);
        str_order_stable(&mut b, |e| e.0, |_, _| Ordering::Equal);
        assert_eq!(a, b, "upper level, case {case}");
    }
}
