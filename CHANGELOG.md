# Changelog

All notable changes to `situs` are recorded here. The project follows
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-11-08

First release.

### Boxes

- `Rect`: an axis-aligned box with closed intervals. `Rect::new` refuses
  non-finite and inverted bounds with `RectError` (`NonFinite`, `Inverted`;
  non-exhaustive). Accessors `min_x`, `min_y`, `max_x`, `max_y`; exact
  `intersects`, `contains_point`, `contains_rect` and `union`.

### The static index

- `Index::bulk_load`: an R-tree packed by Sort-Tile-Recursive, with
  `Index::NODE_CAPACITY` (16) and `Index::memory_size_for`.
- `query`, which reports exactly the entries whose boxes meet the window,
  and `try_query`, which allocates nothing and stops when the visitor
  returns `ControlFlow::Break`.
- `join` (every meeting pair across two indexes) and `self_join` (every
  meeting pair of distinct entries, once each).
- `bounds`, `len`, `is_empty`.

### The dynamic index

- `DynamicIndex`: an R*-tree with `new`, `bulk_load`, `insert` and
  `remove` (an exact match of box and id; returns whether an entry was
  removed). `DynamicIndex::NODE_CAPACITY` (16), `DynamicIndex::MIN_FILL`
  (6). Bounds are the exact union of the stored boxes after every change.
- `query`, `bounds`, `len`, `is_empty`; `Default`.

### Nearest neighbors (both index types)

- `nearest_candidates(x, y, k)`: every entry whose certified bounds could
  place it among the k nearest, as `Neighbor { id, lower_sq, upper_sq }`.
- `browse(x, y)`: a `Browse` iterator over entries by increasing lower
  bound.
- `QueryError::NonFinite` for a NaN or infinite query point
  (non-exhaustive).
- Under the `exact-nearest` feature, `nearest_exact(x, y, k)`: the exact k
  nearest plus every entry exactly tied with the k-th, ranked by
  `arithmetica::compare_distance`. Adds `arithmetica` 0.1.1 as an optional
  dependency.

[0.1.0]: https://github.com/JustinGeoSci/situs/releases/tag/v0.1.0