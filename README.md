# situs

Spatial indexes over `f64` boxes, with no false negatives.

`situs` answers two questions of a set of stored boxes: which of them might meet a query box, and which might lie nearest to a query point. The index operates as a **candidate filter** rather than a decision procedure. It returns every entry that could be an answer, and the caller decides exactly among them.

## The guarantee

The index never omits an answer. Every entry whose box meets the query box is returned, and every entry that could be among the k nearest to a query point is returned.

- **Every comparison is exact, or is made on a certified bound whose error can only add candidates.** Box tests compare stored `f64` values, and node boxes are the exact `min` and `max` of their children's boxes. Nearest-neighbor search uses `f64` bounds on each squared distance, `lower_sq ≤ d² ≤ upper_sq`, proven to bracket the exact value, so rounding can widen the candidate set but never shrink it.
- **No distance is presented as exact.** Nearest-neighbor results carry their certified bounds rather than a distance. Under the opt-in `exact-nearest` feature, candidates are ranked exactly by the exact distance comparison of [`arithmetica`](https://crates.io/crates/arithmetica).

A box query returns *exactly* the entries whose boxes meet the window. Arithmetic determines only the shape of the tree (the sort keys of the bulk load and the R\* heuristics), never its contents.

## Two index types

- **`Index`** is static and bulk-loaded by Sort-Tile-Recursive packing. It is compact and the fastest to query, suited to a set that is built once and queried many times.
- **`DynamicIndex`** is an R\*-tree with insertion and exact removal, by box and id, and its bounds shrink as entries leave. It suits a set that changes. It can also be bulk-loaded by the same packing as the static index.

Both hold `(Rect, u32)` entries. `Rect` is an axis-aligned box with closed intervals, defined by four finite bounds that are not inverted. Zero width and zero height are allowed, and touching boxes meet. `Rect::new` refuses anything else. An id is a key rather than a payload, so the index remains independent of what it indexes, and the caller maps each id back to its own data.

## Operations

| | `Index` | `DynamicIndex` |
|---|---|---|
| Bulk load (`bulk_load`) | ✓ | ✓ |
| Box query (`query`) | ✓ | ✓ |
| Early-exit box query without allocation (`try_query`) | ✓ | |
| Pairs of meeting boxes (`join`, `self_join`) | ✓ | |
| Nearest-neighbor candidates (`nearest_candidates`) and distance browsing (`browse`) | ✓ | ✓ |
| Exact k nearest (`nearest_exact`, `exact-nearest` feature) | ✓ | ✓ |
| `insert`, `remove` | | ✓ |
| `bounds`, `len`, `is_empty` | ✓ | ✓ |
| Memory needed for `n` entries (`memory_size_for`) | ✓ | |

On the static index, a query whose window contains a whole subtree's box reports that subtree without testing it further.

## Examples

```rust
use situs::{Index, Rect};

let a = Rect::new(0.0, 0.0, 1.0, 1.0).unwrap();
let b = Rect::new(5.0, 5.0, 6.0, 6.0).unwrap();
let index = Index::bulk_load(vec![(a, 0), (b, 1)]);

let mut hits = Vec::new();
index.query(&Rect::new(0.5, 0.5, 2.0, 2.0).unwrap(), |id| hits.push(id));
assert_eq!(hits, [0]);
```

```rust
use situs::{DynamicIndex, Rect};

let mut index = DynamicIndex::new();
let a = Rect::new(0.0, 0.0, 1.0, 1.0).unwrap();
let b = Rect::new(5.0, 5.0, 6.0, 6.0).unwrap();
index.insert(a, 0);
index.insert(b, 1);
assert!(index.remove(&a, 0));
assert_eq!(index.bounds(), Some(b));

let near = index.nearest_candidates(4.0, 4.0, 1).unwrap();
assert_eq!(near[0].id, 1);
assert!(near[0].lower_sq <= 2.0 && 2.0 <= near[0].upper_sq);
```

## What it does not do

- **It computes no exact distances.** Nearest-neighbor results are bounds, and exact ranking under `exact-nearest` compares distances without computing them.
- **The dynamic tree has no `join`, `self_join` or early-exit query in 0.1.**
- **It does not encode an index.** An index is rebuilt from its entries.

## Design

- No dependencies by default. The `exact-nearest` feature adds [`arithmetica`](https://crates.io/crates/arithmetica), also `no_std`, as the only dependency.
- `#![no_std]` with `alloc`.
- `#![forbid(unsafe_code)]`, and no panics on any input it accepts.
- Deterministic: the same entries in the same order give the same tree and the same results.
- Minimum supported Rust version: 1.85.

## References

- S. T. Leutenegger, M. A. Lopez, J. M. Edgington. *STR: A Simple and Efficient Algorithm for R-Tree Packing.* ICDE 1997.
- N. Beckmann, H.-P. Kriegel, R. Schneider, B. Seeger. *The R\*-tree: An Efficient and Robust Access Method for Points and Rectangles.* SIGMOD 1990.
- T. Brinkhoff, H.-P. Kriegel, B. Seeger. *Efficient Processing of Spatial Joins Using R-trees.* SIGMOD 1993.
- G. R. Hjaltason, H. Samet. *Distance Browsing in Spatial Databases.* ACM TODS 24(2), 1999.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.