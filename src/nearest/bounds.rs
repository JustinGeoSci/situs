// src/nearest/bounds.rs

//! Certified bounds on the squared distance from a point to a box.
//!
//! [`bounds_sq`] computes two `f64` values `L ≤ d² ≤ U`, where `d²` is the
//! **exact** squared Euclidean distance from the query point to the box, taken
//! as a real number. The bounds are certified, meaning the inequalities hold
//! for every finite query point and every valid [`Rect`]. The proof below is
//! given in full because the nearest-neighbor guarantee rests on nothing
//! else.
//!
//! # The exact quantity
//!
//! The point of a box `[x₀, x₁] × [y₀, y₁]` nearest to `p = (x, y)` is the
//! clamp of `p` to the box, a selection among input values decided by exact
//! comparisons. Per axis, the exact distance is
//!
//! ```text
//! dx = x₀ − x   if x < x₀
//!      x − x₁   if x > x₁
//!      0        otherwise      (dy likewise; d² = dx² + dy²)
//! ```
//!
//! # The computation
//!
//! The computation takes `a = fl(dx)` (one subtraction, or exactly 0) and
//! `b = fl(dy)`, then `ŝ = fl(fl(a·a) + fl(b·b))`, with `fl` the IEEE 754
//! round-to-nearest of each operation. The bounds follow from `ŝ`:
//!
//! ```text
//! ŝ < T        →  L = 0,                U = U_SMALL
//! ŝ infinite   →  L = fl(MAX · K_LO),   U = +∞
//! otherwise    →  L = fl(ŝ · K_LO),     U = fl(ŝ · K_HI)
//! ```
//!
//! Here `T = 2⁻¹⁰²⁰`, `U_SMALL = 2⁻¹⁰¹⁹`, `K_LO = 1 − 2⁻⁵⁰` and
//! `K_HI = 1 + 2⁻⁵⁰` (both factors exactly representable), and `MAX` is the
//! largest finite `f64`.
//!
//! # The proof
//!
//! Let `u = 2⁻⁵³`, the unit roundoff.
//!
//! 1. **Subtraction:** `a = dx(1 + δ₁)` with `|δ₁| ≤ u`, by correct rounding.
//!    When the exact difference of two floats is subnormal it is exactly
//!    representable, so the subtraction never loses accuracy to underflow,
//!    and either `|δ₁| ≤ u` or `a` is exact. If `dx` is at or above the
//!    overflow threshold, `a = +∞`, a case item 6 resolves.
//! 2. **Squaring:** While `a²` is in the normal range,
//!    `fl(a·a) = a²(1 + δ₂)`, `|δ₂| ≤ u`. When `a²` is subnormal the error
//!    is absolute rather than relative, `fl(a·a) = a² + η` with
//!    `|η| ≤ 2⁻¹⁰⁷⁵` (half the subnormal spacing).
//! 3. **Addition:** Both operands are non-negative, so
//!    `fl(P + Q) = (P + Q)(1 + δ₃)`, `|δ₃| ≤ u`. A subnormal sum of
//!    subnormal operands is exact.
//! 4. **Together:** With `η` the combined absolute term,
//!    `|η| ≤ 2 · 2⁻¹⁰⁷⁵ = 2⁻¹⁰⁷⁴`, the computed sum is bounded on both sides:
//!
//!    ```text
//!    ŝ ∈ [ d²(1−u)⁴ − 2⁻¹⁰⁷⁴(1+u),  d²(1+u)⁴ + 2⁻¹⁰⁷⁴(1+u) ]
//!    ```
//!
//!    Let `c = 2⁻¹⁰⁷³ ≥ 2⁻¹⁰⁷⁴(1+u)`. Rearranged, the two inequalities the
//!    bounds must respect are
//!    `d² ≥ (ŝ − c)/(1+u)⁴` and `d² ≤ (ŝ + c)/(1−u)⁴`.
//! 5. **`L` when `T ≤ ŝ < ∞`:** Since `ŝ ≥ T = 2⁻¹⁰²⁰`, the absolute term
//!    satisfies `c ≤ u·ŝ` (`2⁻¹⁰⁷³ = 2⁻⁵³ · 2⁻¹⁰²⁰`). So
//!    `d² ≥ ŝ(1−u)/(1+u)⁴`. The computed lower bound satisfies
//!    `fl(ŝ·K_LO) ≤ ŝ(1−2⁻⁵⁰)(1+u) = ŝ(1−8u)(1+u)`, and
//!    `(1−8u)(1+u)⁵ ≤ 1−3u+O(u²) < 1−u`, so `fl(ŝ·K_LO) < d²`. The product
//!    cannot underflow (`ŝ·K_LO ≥ T/2` is normal) and cannot overflow
//!    (`K_LO < 1`).
//! 6. **Overflow:** `ŝ = +∞` requires the exact `a² + b²` (or `dx` itself)
//!    to reach the rounding threshold `2¹⁰²⁴(1 − 2⁻⁵⁴)`, so
//!    `d² ≥ 2¹⁰²⁴(1−2⁻⁵⁴)/(1+u)⁴ ≥ 2¹⁰²⁴(1 − 5u)`. The mapped bound is
//!    `fl(MAX·K_LO) ≤ 2¹⁰²⁴(1−u)(1−8u)(1+u) < 2¹⁰²⁴(1−8u) < d²`. `U = +∞`
//!    bounds anything from above. Infinities never meet a subtraction that
//!    could make NaN, because every operand along the way is non-negative,
//!    so `ŝ` is `+∞` or finite and never NaN.
//! 7. **`L` when `ŝ < T`:** `L = 0 ≤ d²` always. The bound depends on the
//!    threshold. Without it, `ŝ = 2⁻¹⁰⁷⁴` (a subnormal square rounded up)
//!    would give `fl(ŝ·K_LO) = 2⁻¹⁰⁷⁴` back (the product rounds up), which
//!    exceeds the true `d²` when `d² < 2⁻¹⁰⁷⁴`.
//! 8. **`U` when `T ≤ ŝ < ∞`:** Symmetrically, `d² ≤ ŝ(1+u)/(1−u)⁴` and
//!    `fl(ŝ·K_HI) ≥ ŝ(1+8u)(1−u)`, with `(1+8u)(1−u)⁵ ≥ 1+2u > 1+u`, so
//!    `fl(ŝ·K_HI) > d²`. If the product overflows, `U = +∞`, still valid.
//! 9. **`U` when `ŝ < T`:** `d² ≤ (ŝ + c)/(1−u)⁴ < (T + c)(1 + 5u) < 2T`, so
//!    the constant `U_SMALL = 2T = 2⁻¹⁰¹⁹` bounds `d²` from above. The
//!    constant is required rather than a convenience, since `ŝ` can be 0
//!    while `d² > 0` (a square that underflows), and `fl(ŝ·K_HI)` would
//!    then be 0, below `d²`.
//!
//! **The computation ends without a directed-rounding step.** A `next_down` on
//! `L` (and `next_up` on `U`) would serve the same inequality that `K_LO`
//! (`K_HI`) already serves with margin. The proof in items 5 and 8 closes
//! with `1−8u` against a need of about `1−5u`, strictly, so a directed step
//! could never separate a correct bound from an incorrect one. A component
//! whose removal no test can detect is either unnecessary or untested, and
//! this one would be unnecessary.
//!
//! # Monotonicity
//!
//! The lower bound is monotone in the box: **if `B ⊇ B′` then
//! `L(p, B) ≤ L(p, B′)`**. Per axis, the exact distance to `B` is at most the
//! exact distance to `B′`, because an enclosing interval is never farther
//! from `p`. Every step of the computation is monotone non-decreasing.
//! Correctly rounded subtraction is `fl` of the exact difference, and `fl`
//! is monotone. Squaring is monotone on non-negative values, and so are the
//! addition, the multiplication by a positive constant, the threshold step
//! (below `T` the bound is 0, at or above `T` it is at least
//! `fl(T·K_LO) > 0`), and the overflow mapping (`fl(MAX·K_LO)` is the value
//! the finite branch gives at `ŝ = MAX`, the largest finite `ŝ`). `L` is
//! therefore a monotone function of the per-axis exact distances.
//!
//! The search rests on this property. A node's box contains every
//! descendant's box, so the node's `L` is at most every descendant entry's
//! `L`, and a best-first search over `L` can stop without descending
//! further. A non-monotone step would not make a bound wrong; it would
//! break that stopping rule, and the broken rule would produce a false
//! negative.

use crate::Rect;

/// The value below which the absolute underflow term `c = 2⁻¹⁰⁷³` need not
/// be at most `u·ŝ`. Below it the bounds come from the two constants
/// instead (proof, items 7 and 9).
const UNDERFLOW_THRESHOLD: f64 = f64::MIN_POSITIVE * 4.0; // 2⁻¹⁰²⁰, exact

/// The upper bound used when `ŝ` is below the threshold, `2⁻¹⁰¹⁹`, which is
/// at or above every `d²` consistent with such an `ŝ` (proof, item 9).
const UPPER_BELOW_THRESHOLD: f64 = f64::MIN_POSITIVE * 8.0; // exact

/// The downward factor `1 − 2⁻⁵⁰`, exactly representable (proof, item 5).
const K_LO: f64 = 1.0 - f64::EPSILON * 4.0;

/// The upward factor `1 + 2⁻⁵⁰`, exactly representable (proof, item 8).
const K_HI: f64 = 1.0 + f64::EPSILON * 4.0;

/// The lower bound when `ŝ` overflows, the finite branch's value at the
/// largest finite `ŝ` (proof, item 6, and monotonicity).
const LOWER_AT_OVERFLOW: f64 = f64::MAX * K_LO;

/// The exact per-axis distance, rounded once, `fl(dx)`.
///
/// Exact comparisons of stored values decide the branch, and only the
/// subtraction rounds (proof, item 1). The result is `0.0` when `p` is
/// within the interval, ends included.
fn axis_distance(p: f64, lo: f64, hi: f64) -> f64 {
    if p < lo {
        lo - p
    } else if p > hi {
        p - hi
    } else {
        0.0
    }
}

/// `ŝ = fl(fl(a·a) + fl(b·b))` for the two per-axis distances.
fn squared_sum(x: f64, y: f64, rect: &Rect) -> f64 {
    let a = axis_distance(x, rect.min_x(), rect.max_x());
    let b = axis_distance(y, rect.min_y(), rect.max_y());
    a * a + b * b
}

/// Certified bounds `(L, U)` with `L ≤ d² ≤ U`, `d²` the exact squared
/// distance from the finite point `(x, y)` to `rect` (module doc).
///
/// `L` is finite and non-negative, and `U` is positive or `+∞`. Neither is
/// ever NaN. `L` is monotone, so a box containing another never receives
/// the larger `L`.
pub(crate) fn bounds_sq(x: f64, y: f64, rect: &Rect) -> (f64, f64) {
    let s = squared_sum(x, y, rect);
    if s < UNDERFLOW_THRESHOLD {
        return (0.0, UPPER_BELOW_THRESHOLD);
    }
    if s.is_infinite() {
        return (LOWER_AT_OVERFLOW, f64::INFINITY);
    }
    (s * K_LO, s * K_HI)
}

#[cfg(test)]
pub(crate) fn parts_for_tests(x: f64, y: f64, rect: &Rect) -> (f64, f64, f64) {
    let s = squared_sum(x, y, rect);
    let (l, u) = bounds_sq(x, y, rect);
    (s, l, u)
}

#[cfg(test)]
pub(crate) mod tests;
