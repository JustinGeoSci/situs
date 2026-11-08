// src/tests.rs

//! Tests of [`Rect`], and the builders the other test modules share.
//!
//! The index tests live beside the index (`static_index::tests`). This module
//! tests the box type the whole correctness argument rests on, on its own,
//! specifically the refusal of non-finite and inverted bounds, the closed
//! overlap test, and the exact union.

#![allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]

use super::*;
use proptest::prelude::*;
use std::format;

// ===========================================================================
// Shared builders
// ===========================================================================

/// A box from known-valid bounds.
pub(crate) fn r(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Rect {
    Rect::new(min_x, min_y, max_x, max_y).unwrap()
}

/// A coordinate, mostly a small integer or half (so boxes touch and
/// coincide) and sometimes an extreme.
pub(crate) fn coord() -> impl Strategy<Value = f64> {
    prop_oneof![
        8 => (-20i32..=20).prop_map(|v| f64::from(v) * 0.5),
        1 => (-1_000_000i32..=1_000_000).prop_map(f64::from),
        1 => prop_oneof![
            Just(1e300),
            Just(-1e300),
            Just(f64::MIN_POSITIVE),
            Just(-f64::MIN_POSITIVE),
            Just(2.0 * f64::MIN_POSITIVE),
            Just(0.0),
            Just(-0.0),
        ],
    ]
}

/// A box from two corners in any order. A repeated corner gives a point,
/// and equal `x` or `y` gives a segment.
pub(crate) fn rect() -> impl Strategy<Value = Rect> {
    (coord(), coord(), coord(), coord()).prop_map(|(a, b, c, d)| {
        let (min_x, max_x) = if a <= c { (a, c) } else { (c, a) };
        let (min_y, max_y) = if b <= d { (b, d) } else { (d, b) };
        Rect::new(min_x, min_y, max_x, max_y).unwrap()
    })
}

/// Cases per sweep, the full count in an optimized build and a tenth in
/// debug, where the brute-force oracle over a thousand entries runs slowly.
pub(crate) const CASES: u32 = if cfg!(debug_assertions) {
    1_000
} else {
    10_000
};

// ===========================================================================
// Rect
// ===========================================================================

#[test]
fn rect_refuses_non_finite_before_inverted() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(Rect::new(bad, 0.0, 1.0, 1.0), Err(RectError::NonFinite));
        assert_eq!(Rect::new(0.0, bad, 1.0, 1.0), Err(RectError::NonFinite));
        assert_eq!(Rect::new(0.0, 0.0, bad, 1.0), Err(RectError::NonFinite));
        assert_eq!(Rect::new(0.0, 0.0, 1.0, bad), Err(RectError::NonFinite));
        // NaN with inverted-looking bounds, where non-finite is the diagnosis.
        assert_eq!(Rect::new(bad, 5.0, 0.0, 0.0), Err(RectError::NonFinite));
    }
    assert_eq!(Rect::new(1.0, 0.0, 0.0, 1.0), Err(RectError::Inverted));
    assert_eq!(Rect::new(0.0, 1.0, 1.0, 0.0), Err(RectError::Inverted));
}

#[test]
fn rect_accepts_degenerate_boxes() {
    let p = r(2.0, 3.0, 2.0, 3.0);
    assert!(p.contains_point(2.0, 3.0));
    assert!(!p.contains_point(2.0, 3.000_000_1));
    let v = r(1.0, 0.0, 1.0, 5.0);
    assert!(v.intersects(&r(0.0, 2.0, 1.0, 2.0)), "touching at a corner");
    assert_eq!(v.intersects(&p), p.intersects(&v));
}

#[test]
fn rect_intersects_is_closed() {
    let a = r(0.0, 0.0, 2.0, 2.0);
    assert!(a.intersects(&r(2.0, 0.0, 4.0, 2.0)), "edge contact");
    assert!(a.intersects(&r(2.0, 2.0, 4.0, 4.0)), "corner contact");
    assert!(a.intersects(&r(0.5, 0.5, 1.5, 1.5)), "contained");
    assert!(!a.intersects(&r(2.000_000_000_000_001, 0.0, 4.0, 2.0)));
    assert!(!a.intersects(&r(0.0, -1.0, 2.0, -0.000_000_1)));
}

#[test]
fn rect_contains_point_is_false_for_nan_and_infinity() {
    let a = r(0.0, 0.0, 2.0, 2.0);
    assert!(a.contains_point(0.0, 2.0));
    assert!(!a.contains_point(f64::NAN, 1.0));
    assert!(!a.contains_point(1.0, f64::NAN));
    assert!(!a.contains_point(f64::INFINITY, 1.0));
    assert!(!a.contains_point(1.0, f64::NEG_INFINITY));
}

#[test]
fn rect_union_is_exact_and_keeps_self_on_ties() {
    let a = r(-1.0, 0.0, 3.0, 2.0);
    let b = r(0.0, -5.0, 1.0, 7.0);
    assert_eq!(a.union(&b), r(-1.0, -5.0, 3.0, 7.0));
    assert_eq!(b.union(&a), r(-1.0, -5.0, 3.0, 7.0));
    let z = r(-0.0, 0.0, 0.0, 0.0);
    let p = r(0.0, 0.0, -0.0, 0.0);
    assert_eq!(z.union(&p).min_x().to_bits(), (-0.0_f64).to_bits());
    assert_eq!(p.union(&z).min_x().to_bits(), 0.0_f64.to_bits());
    assert_eq!(z.union(&p).max_x().to_bits(), 0.0_f64.to_bits());
}

#[test]
fn rect_error_displays() {
    assert_eq!(
        format!("{}", RectError::NonFinite),
        "rect bound is NaN or infinite"
    );
    assert_eq!(
        format!("{}", RectError::Inverted),
        "rect min exceeds max on an axis"
    );
}

// ===========================================================================
// Rect: property tests
// ===========================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(CASES))]

    /// `Rect::new` accepts exactly finite, non-inverted bounds.
    #[test]
    fn rect_new_decides_by_the_invariants(
        a in prop::num::f64::ANY, b in prop::num::f64::ANY, c in prop::num::f64::ANY, d in prop::num::f64::ANY
    ) {
        let got = Rect::new(a, b, c, d);
        let finite = a.is_finite() && b.is_finite() && c.is_finite() && d.is_finite();
        if !finite {
            prop_assert_eq!(got, Err(RectError::NonFinite));
        } else if a > c || b > d {
            prop_assert_eq!(got, Err(RectError::Inverted));
        } else {
            let rect = got.unwrap();
            prop_assert_eq!((rect.min_x(), rect.min_y(), rect.max_x(), rect.max_y()), (a, b, c, d));
        }
    }

    /// `intersects` is symmetric and agrees with the interval definition,
    /// and `union` contains both boxes and is the smallest such box.
    #[test]
    fn rect_algebra(a in rect(), b in rect()) {
        prop_assert_eq!(a.intersects(&b), b.intersects(&a));
        let by_intervals = !(a.max_x() < b.min_x() || b.max_x() < a.min_x()
            || a.max_y() < b.min_y() || b.max_y() < a.min_y());
        prop_assert_eq!(a.intersects(&b), by_intervals);
        let u = a.union(&b);
        prop_assert!(u.contains_rect(&a) && u.contains_rect(&b));
        prop_assert_eq!(u.min_x(), a.min_x().min(b.min_x()));
        prop_assert_eq!(u.max_y(), a.max_y().max(b.max_y()));
        prop_assert!(a.contains_rect(&a));
    }
}
