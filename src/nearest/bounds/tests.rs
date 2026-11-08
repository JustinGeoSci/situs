// src/nearest/bounds/tests.rs

//! Tests of the certified bounds. The bounds bracket the exact value and are
//! monotone, and every regime of the proof (normal, subnormal and overflow)
//! is exercised at values chosen to make rounding go each way.

#![allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use super::*;
use crate::tests::r;
use std::format;
use std::string::String;
use std::vec::Vec;

/// A fixed SplitMix64 stream, which gives the same cases on every run.
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Rng {
        Rng(seed)
    }

    pub(crate) fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// An integer in `[-bound, bound]`.
    pub(crate) fn int(&mut self, bound: i64) -> i64 {
        let span = (2 * bound + 1) as u64;
        (self.next() % span).cast_signed() - bound
    }

    /// A finite `f64` with a random sign and significand and an exponent
    /// drawn from `[emin, emax]` (binades; -1075 reaches the subnormals).
    pub(crate) fn float_in(&mut self, emin: i32, emax: i32) -> f64 {
        let e = emin + (self.next() % (u64::from((emax - emin) as u32) + 1)) as i32;
        let m = 1.0 + (self.next() >> 12) as f64 * f64::EPSILON;
        let v = m * libm_exp2(e);
        if self.next().is_multiple_of(2) { v } else { -v }
    }
}

/// `2^e` for `e` in `[-1075, 1023]`, by exact halving/doubling.
pub(crate) fn libm_exp2(e: i32) -> f64 {
    let mut v = 1.0f64;
    let mut n = e;
    while n > 0 {
        v *= 2.0;
        n -= 1;
    }
    while n < 0 {
        v *= 0.5;
        n += 1;
    }
    v
}

/// The exact squared distance for integer inputs, in `u128`.
fn exact_sq_int(x: i64, y: i64, b: &Rect) -> u128 {
    let axis = |p: i64, lo: f64, hi: f64| -> i128 {
        let (lo, hi) = (lo as i64, hi as i64);
        if p < lo {
            i128::from(lo) - i128::from(p)
        } else if p > hi {
            i128::from(p) - i128::from(hi)
        } else {
            0
        }
    };
    let dx = axis(x, b.min_x(), b.max_x());
    let dy = axis(y, b.min_y(), b.max_y());
    (dx * dx + dy * dy) as u128
}

/// `L ≤ d² ≤ U` for an integer `d²`, tested as `ceil(L) ≤ d²` and
/// `floor(U) ≥ d²`, both exact in `u128`.
fn assert_brackets_int(x: i64, y: i64, b: &Rect, what: &str) {
    let d2 = exact_sq_int(x, y, b);
    let (l, u) = bounds_sq(x as f64, y as f64, b);
    assert!(l >= 0.0 && l.is_finite(), "{what}: L = {l}");
    assert!(
        l.ceil() as u128 <= d2,
        "{what}: L = {l:e} above d² = {d2} for p = ({x}, {y}), box {b:?}"
    );
    assert!(
        u.is_infinite() || u.floor() as u128 >= d2,
        "{what}: U = {u:e} below d² = {d2} for p = ({x}, {y}), box {b:?}"
    );
}

// ===========================================================================
// Brackets, exactly, on integers
// ===========================================================================

/// 150,000 fixed pseudo-random cases with integer coordinates up to 2²⁶ in
/// magnitude, where `d²` is exact in `u128` and `L ≤ d² ≤ U` holds exactly.
/// The sweep counts the cases where `ŝ` rounded above and below `d²`, so
/// each direction the two factors guard against is observed.
#[test]
fn bounds_bracket_the_exact_value_on_integer_coordinates() {
    let bound = 1 << 26;
    let mut rng = Rng::new(0x5EED_B0CD_5000_0001);
    let (mut above, mut below) = (0u64, 0u64);
    for _ in 0..150_000 {
        let (x, y) = (rng.int(bound), rng.int(bound));
        let (x0, x1) = {
            let (a, b) = (rng.int(bound), rng.int(bound));
            (a.min(b), a.max(b))
        };
        let (y0, y1) = {
            let (a, b) = (rng.int(bound), rng.int(bound));
            (a.min(b), a.max(b))
        };
        let b = r(x0 as f64, y0 as f64, x1 as f64, y1 as f64);
        assert_brackets_int(x, y, &b, "sweep");
        let d2 = exact_sq_int(x, y, &b);
        let (s, _, _) = parts_for_tests(x as f64, y as f64, &b);
        // In this regime every intermediate rounds at a spacing of whole
        // units or more, so ŝ is integer-valued and the cast is exact.
        assert_eq!(s.fract(), 0.0);
        let s_int = s as u128;
        if s_int > d2 {
            above += 1;
        } else if s_int < d2 {
            below += 1;
        }
    }
    // In this all-integer regime ŝ never rounds above d². A square is 0 or
    // 1 mod 4 (an odd square is 1 mod 8), so every rounding tie in the
    // squares and in their sum falls on an even significand below the exact
    // value. The above direction exists only off the integers. The
    // handpicked test below and the oracle's full-range cases cover it, and
    // the oracle requires both directions to occur.
    assert_eq!(
        above, 0,
        "integer ŝ rounded above d²: the parity argument is wrong"
    );
    assert!(below > 0, "no case had ŝ round below d²");
}

/// Off the integers, `ŝ` can round above `d²`, and the lower bound must
/// stay below the exact value anyway. With one axis, `a = fl(0.1)`,
/// `d² = a²` exactly as a real, and `ŝ = fl(a·a) > d²`. The FMA residual
/// `err = fl(a·a − ŝ) = a² − ŝ` is exact, so `err < 0` establishes the
/// direction, and `L ≤ d²` becomes `ŝ − L > −err`, both sides exact.
#[test]
fn the_sum_can_round_above_and_the_lower_bound_stays_below() {
    for a in [0.1f64, 0.3, 0.7, 1.1, 123.456, 1e-7] {
        let (s, l, _) = parts_for_tests(0.0, 0.0, &r(a, 0.0, a, 0.0));
        let err = a.mul_add(a, -s);
        if err < 0.0 {
            // ŝ rounded above the exact d² = a².
            assert!(
                s - l > -err,
                "a = {a}: L = {l:e} is not below d² (ŝ = {s:e}, err = {err:e})"
            );
        } else {
            assert!(l <= s, "a = {a}");
        }
    }
    // At least the first one must take the above-direction branch.
    let (s, _, _) = parts_for_tests(0.0, 0.0, &r(0.1, 0.0, 0.1, 0.0));
    assert!(
        0.1f64.mul_add(0.1, -s) < 0.0,
        "fl(0.1)² did not round above"
    );
}

/// The corner cases of the integer regime, inside, on an edge, at a corner,
/// and one unit outside each.
#[test]
fn bounds_at_contact_are_zero_lower() {
    let b = r(-3.0, -2.0, 5.0, 7.0);
    for (x, y) in [(0, 0), (-3, 0), (5, 7), (-3, -2), (5, -2), (0, 7)] {
        let d2 = exact_sq_int(x, y, &b);
        assert_eq!(d2, 0);
        let (l, u) = bounds_sq(x as f64, y as f64, &b);
        assert_eq!(l, 0.0);
        assert!(u >= 0.0);
    }
    for (x, y, d2) in [(-4, 0, 1u128), (6, 7, 1), (6, 8, 2), (-4, -3, 2)] {
        assert_eq!(exact_sq_int(x, y, &b), d2);
        assert_brackets_int(x, y, &b, "unit outside");
    }
}

// ===========================================================================
// The subnormal regime, at handpicked values
// ===========================================================================

/// The proof's items 7 and 9, at the values where rounding would go wrong
/// without them.
#[test]
fn bounds_bracket_handpicked_subnormal_distances() {
    // dx = 2⁻⁵³⁷, so d² = 2⁻¹⁰⁷⁴ exactly, which is representable.
    let dx = libm_exp2(-537);
    let b = r(dx, 0.0, dx, 0.0);
    let (l, u) = bounds_sq(0.0, 0.0, &b);
    let d2 = libm_exp2(-1074);
    assert!(l <= d2 && d2 <= u, "L = {l:e}, U = {u:e}, d² = {d2:e}");

    // dx = 1.5·2⁻⁵³⁸, so d² = 1.125·2⁻¹⁰⁷⁵, below 2⁻¹⁰⁷⁴, while a·a rounds
    // up to 2⁻¹⁰⁷⁴. L must stay below d², and so below 2⁻¹⁰⁷⁴ (proof,
    // item 7).
    let dx = 1.5 * libm_exp2(-538);
    let (s, l, u) = parts_for_tests(0.0, 0.0, &r(dx, 0.0, dx, 0.0));
    assert_eq!(s, libm_exp2(-1074), "a·a did not round up as constructed");
    assert!(l < libm_exp2(-1074), "L = {l:e} is not below d² < 2⁻¹⁰⁷⁴");
    assert!(u >= libm_exp2(-1074), "U = {u:e}");

    // dx = 1.25·2⁻⁵³⁸, where a·a underflows to exactly 0 while d² > 0. A
    // positive distance must keep a positive upper bound (proof, item 9).
    let dx = 1.25 * libm_exp2(-538);
    let (s, l, u) = parts_for_tests(0.0, 0.0, &r(dx, 0.0, dx, 0.0));
    assert_eq!(s, 0.0, "a·a did not underflow to zero as constructed");
    assert_eq!(l, 0.0);
    assert!(u > 0.0, "a positive distance got U = 0");

    // Both axes subnormal, and one subnormal against one huge.
    for (dx, dy) in [
        (libm_exp2(-540), libm_exp2(-541)),
        (libm_exp2(-1074), 0.0),
        (libm_exp2(-537), libm_exp2(-1074)),
        (libm_exp2(-540), 3.5),
        (libm_exp2(-1074), 1e300),
    ] {
        let (l, u) = bounds_sq(0.0, 0.0, &r(dx, dy, dx, dy));
        assert!(l.is_finite() && l >= 0.0);
        assert!(u > 0.0, "dx = {dx:e}, dy = {dy:e}: U = {u:e}");
        // The repeatable part of the bracket: the one-axis exact squares.
        if dy == 0.0 && dx == libm_exp2(-537) {
            assert!(l <= libm_exp2(-1074) && libm_exp2(-1074) <= u);
        }
    }
}

// ===========================================================================
// The overflow regime
// ===========================================================================

/// The proof's item 6. An infinite `ŝ` maps to a finite `L` and `U = +∞`,
/// and an overflowing subtraction behaves the same.
#[test]
fn bounds_at_overflow_are_finite_below_and_infinite_above() {
    // The square overflows, with dx = 2e300.
    let cases = [
        (-1e300, 0.0, r(1e300, 0.0, 1e300, 0.0)),
        // The subtraction overflows, with dx = 2·MAX exactly.
        (f64::MAX, 0.0, r(-f64::MAX, 0.0, -f64::MAX, 0.0)),
        // Both axes huge.
        (-1e300, -1e300, r(1e300, 1e300, 1e308, 1e308)),
        // One axis at MAX, the other moderate.
        (-f64::MAX, 3.0, r(f64::MAX, 5.0, f64::MAX, 5.0)),
    ];
    for (x, y, b) in cases {
        let (s, l, u) = parts_for_tests(x, y, &b);
        assert!(s.is_infinite(), "ŝ = {s:e} is not the overflow regime");
        assert_eq!(u, f64::INFINITY);
        assert!(l.is_finite(), "L = {l}");
        assert_eq!(l, LOWER_AT_OVERFLOW, "L is not the overflow constant");
    }
    assert_eq!(LOWER_AT_OVERFLOW, f64::MAX * K_LO);
    const { assert!(LOWER_AT_OVERFLOW < f64::MAX) }
    // Large but finite ŝ stays on the finite branch, below the mapped value.
    let (s, l, u) = parts_for_tests(0.0, 0.0, &r(1e154, 0.0, 1e154, 0.0));
    assert!(s.is_finite());
    assert!(l <= 1e308 && u >= 1e308);
    assert!(l <= LOWER_AT_OVERFLOW);
}

// ===========================================================================
// Monotonicity
// ===========================================================================

/// 120,000 fixed pseudo-random nested pairs `B ⊇ B′`, each satisfying
/// `L(p, B) ≤ L(p, B′)`. `B′` is built by shrinking `B` toward a point
/// inside it, so the pair is nested by construction. The points include
/// ones inside `B` but outside `B′`, where the inner distance is positive
/// and the outer is zero.
#[test]
fn lower_bound_is_monotone_on_nested_boxes() {
    let mut rng = Rng::new(0x5EED_B0CD_4000_0002);
    for round in 0..120_000 {
        // Mixed magnitudes, from everyday through huge to subnormal binades.
        let (emin, emax) = match round % 3 {
            0 => (-8, 8),
            1 => (900, 1023),
            _ => (-1074, -1000),
        };
        let mut c = || rng.float_in(emin, emax);
        let (a, b2, c1, d) = (c(), c(), c(), c());
        let outer = r(a.min(c1), b2.min(d), a.max(c1), b2.max(d));
        // The shrink picks an interior anchor by midpoint selection, then an
        // inner box between the anchor and one outer corner.
        let ax = outer.min_x() * 0.5 + outer.max_x() * 0.5;
        let ay = outer.min_y() * 0.5 + outer.max_y() * 0.5;
        let ax = ax.clamp(outer.min_x(), outer.max_x());
        let ay = ay.clamp(outer.min_y(), outer.max_y());
        let inner = r(
            ax.min(outer.max_x()),
            ay.min(outer.max_y()),
            outer.max_x(),
            outer.max_y(),
        );
        let (px, py) = (c(), c());
        let (lo_outer, _) = bounds_sq(px, py, &outer);
        let (lo_inner, _) = bounds_sq(px, py, &inner);
        assert!(
            lo_outer <= lo_inner,
            "L({px:e}, {py:e}) of {outer:?} = {lo_outer:e} > {lo_inner:e} of nested {inner:?}"
        );
        // A point inside the outer box and outside the inner one when the
        // boxes differ, where the outer L is 0.
        let qx = outer.min_x();
        let qy = outer.min_y();
        let (lo_o, _) = bounds_sq(qx, qy, &outer);
        let (lo_i, _) = bounds_sq(qx, qy, &inner);
        assert_eq!(lo_o, 0.0);
        assert!(lo_o <= lo_i);
    }
}

// ===========================================================================
// The case file for an exact check
// ===========================================================================

fn hex(v: f64) -> String {
    format!("{:016x}", v.to_bits())
}

/// Writes `situs_nearest_bounds.csv` to the system temporary directory,
/// more than 220,000 cases with the inputs, `ŝ`, `L` and `U` as hex bit
/// patterns, for an independent check against exact rational arithmetic.
/// The test is ignored in normal runs and is run explicitly.
#[test]
#[ignore = "writes the case file situs_nearest_bounds.csv to the temporary directory"]
fn write_bounds_csv() {
    let mut rows: Vec<String> = Vec::new();
    let mut push = |x: f64, y: f64, b: &Rect| {
        let (s, l, u) = parts_for_tests(x, y, b);
        rows.push(format!(
            "{},{},{},{},{},{},{},{},{}",
            hex(x),
            hex(y),
            hex(b.min_x()),
            hex(b.min_y()),
            hex(b.max_x()),
            hex(b.max_y()),
            hex(s),
            hex(l),
            hex(u)
        ));
    };

    // Structured cases: contact, subnormal distances, overflow, and mixed
    // magnitudes.
    let b = r(-3.0, -2.0, 5.0, 7.0);
    for (x, y) in [
        (0.0, 0.0),
        (-3.0, -2.0),
        (5.0, 7.0),
        (-3.0, 7.0),
        (6.0, 8.0),
        (-4.0, -3.0),
    ] {
        push(x, y, &b);
    }
    for e in -545..=-520 {
        for m in [1.0, 1.25, 1.5, 1.75] {
            let dx = m * libm_exp2(e);
            push(0.0, 0.0, &r(dx, 0.0, dx, 0.0));
            push(0.0, 0.0, &r(dx, dx, dx, dx));
        }
    }
    for dx in [1e300, 1e308, f64::MAX, 2.0 * libm_exp2(511)] {
        push(-dx, 0.0, &r(dx, 0.0, dx, 0.0));
        push(dx, dx, &r(-dx, -dx, -dx, -dx));
        push(-dx, 0.0, &r(dx, libm_exp2(-540), dx, libm_exp2(-540)));
    }

    // Random cases, over paired exponent regimes.
    let mut rng = Rng::new(0x5EED_B0CD_4000_0003);
    let regimes: [(i32, i32); 5] = [
        (-8, 8),
        (-1074, -1000),
        (900, 1023),
        (-30, 30),
        (-520, -490),
    ];
    for round in 0..220_000 {
        let (pe, be) = (
            regimes[round % regimes.len()],
            regimes[(round / regimes.len()) % regimes.len()],
        );
        let (x, y) = (rng.float_in(pe.0, pe.1), rng.float_in(pe.0, pe.1));
        let (a, b2) = (rng.float_in(be.0, be.1), rng.float_in(be.0, be.1));
        let (c1, d) = (rng.float_in(be.0, be.1), rng.float_in(be.0, be.1));
        push(x, y, &r(a.min(c1), b2.min(d), a.max(c1), b2.max(d)));
    }

    // The system temporary directory, so the test writes nothing inside or
    // outside the crate's own tree.
    let path = std::env::temp_dir().join("situs_nearest_bounds.csv");
    std::println!("oracle CSV: {}", path.display());
    let header = "x,y,min_x,min_y,max_x,max_y,s_hat,lower,upper";
    let body: String = rows.join("\n");
    std::fs::write(&path, format!("{header}\n{body}\n")).expect("write the oracle CSV");
    assert!(rows.len() > 220_000);
}
