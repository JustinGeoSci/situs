# situs

Spatial indexes over `f64` boxes, with no false negatives.

`situs` governs which entries a query must consider rather than which ones are the answer. Box queries return exactly the entries whose boxes meet the window; nearest-neighbor queries return a **candidate set** certified to contain the true nearest entries. The caller decides exactly among those candidates.

## The guarantee

**No false negatives.** Every entry whose box meets the query box is returned, and every entry that could be among the k nearest to a query point is returned.

- **Every comparison is exact, or a certified bound whose error can only add candidates.** Box tests compare stored `f64` values. Node boxes are the exact `min` and `max` of their children's boxes. Nearest-neighbor search uses `f64` bounds on each squared distance, `lower_sq ≤ d² ≤ upper_sq`, proven to bracket the exact value, so rounding can widen the candidate set but never shrink it.
- **No distance is presented as exact.** Nearest-neighbor results carry their certified bounds, not a distance. Under the opt-in `exact-nearest` feature, candidates are ranked exactly by `arithmetica`'s exact distance comparison.

Box queries return *exactly* the entries whose boxes meet the window. Arithmetic decides only the tree's shape (bulk-load sort keys, R\* heuristics), never its contents.

## Two index types

- **`Index`:** static, bulk-loaded by Sort-Tile-Recursive packing. Compact and the fastest to query. Use it when the set is built once and queried many times.
- **`DynamicIndex`:** an R\*-tree with insertion and exact removal (by box and id). Its bounds shrink as entries leave. Use it when the set changes.

Both hold `(Rect, u32)` entries. `Rect` is an axis-aligned box with closed intervals: four finite bounds, not inverted; zero width and height are allowed, and touching boxes meet. `Rect::new` refuses anything else. Ids are not payloads, so the index stays independent of what it indexes; the caller maps ids back to its own data.

## Operations

| | `Index` | `DynamicIndex` |
|---|---|---|
| Box query (`query`) | ✓ | ✓ |
| Early-exit box query (`try_query`, no allocation) | ✓ | — |
| Pairs of meeting boxes (`join`, `self_join`) | ✓ | — |
| Nearest-neighbor candidates (`nearest_candidates`) and distance browsing (`browse`) | ✓ | ✓ |
| Exact k nearest (`nearest_exact`, `exact-nearest` feature) | ✓ | ✓ |
| `insert`, `remove` | — | ✓ |
| `bounds`, `len`, `is_empty` | ✓ | ✓ |

On the static index, a query that contains a whole subtree's box reports that subtree without testing it further.

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

- **No exact distances.** Nearest-neighbor results are bounds; exact ranking, under `exact-nearest`, compares distances without computing them.
- **No `join`, `self_join` or early-exit query on the dynamic tree** in 0.1.
- **No encoding of an index.** Rebuild it from the entries.

## Design

- No dependencies by default. The `exact-nearest` feature adds [`arithmetica`](https://crates.io/crates/arithmetica), also `no_std`, as the only dependency.
- `#![no_std]` with `alloc`.
- `#![forbid(unsafe_code)]`; no panics on any input it accepts.
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
