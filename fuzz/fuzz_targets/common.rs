// fuzz/fuzz_targets/common.rs

//! Shared input decoding for the `situs` fuzz targets.
//!
//! Every bound is drawn from a small table (halves from −6 to 6, both zeros,
//! ±1e300, and ±1 and ±2 times `f64::MIN_POSITIVE`) indexed by input bytes,
//! so boxes often touch, coincide and share bounds, the inputs on which a
//! closed-interval index fails. The oracle's overlap test is written out
//! here from the bounds rather than taken from `Rect::intersects`, so a
//! defect in that test cannot hide in the oracle.

#![allow(dead_code)]

use situs::Rect;

/// The bounds a byte selects, by its value modulo the table length.
#[rustfmt::skip]
const BOUNDS: [f64; 32] = [
    -6.0, -5.5, -5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5,
    2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, -0.0, 1e300, -1e300, f64::MIN_POSITIVE,
    -f64::MIN_POSITIVE, 2.0 * f64::MIN_POSITIVE, -2.0 * f64::MIN_POSITIVE,
];

/// The most entry boxes a target decodes.
pub const MAX_BOXES: usize = 512;

/// A box from four bytes, two `x` bounds and two `y` bounds, each pair put
/// in order so that every decoded box is valid (a repeated bound gives a
/// segment or a point).
pub fn rect(b: &[u8]) -> Rect {
    let v = |i: usize| BOUNDS[usize::from(b[i]) % BOUNDS.len()];
    let (x0, x1) = (v(0), v(1));
    let (y0, y1) = (v(2), v(3));
    let (min_x, max_x) = if x0 <= x1 { (x0, x1) } else { (x1, x0) };
    let (min_y, max_y) = if y0 <= y1 { (y0, y1) } else { (y1, y0) };
    Rect::new(min_x, min_y, max_x, max_y).expect("ordered finite bounds make a Rect")
}

/// The closed-interval test, written from the bounds.
pub fn meets(a: &Rect, w: &Rect) -> bool {
    !(a.max_x() < w.min_x()
        || w.max_x() < a.min_x()
        || a.max_y() < w.min_y()
        || w.max_y() < a.min_y())
}
