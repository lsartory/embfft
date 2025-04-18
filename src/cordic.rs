/* embfft | cordic.rs
 * Copyright (c) 2025 L. Sartory
 * SPDX-License-Identifier: MIT
 */

//! CORDIC functions
//!
//! Useful for precomputing trigonometry tables

/******************************************************************************/

use core::f64::consts::PI;

/******************************************************************************/

include!(concat!(env!("OUT_DIR"), "/cordic_tables.rs"));

/******************************************************************************/

/// Compute the sine of an angle
///
/// The angle in radians must be comprised between -π/2 and +π/2
pub const fn sin(alpha: f64) -> f64 {
    const N: usize = 63;
    let mut theta = 0.0;
    let mut x = 1.0;
    let mut y = 0.0;
    let mut p2i = 1.0;

    assert!(alpha > -PI / 2.0 && alpha < PI / 2.0);

    let mut i = 0;
    while i < N {
        let sigma = if theta < alpha { 1.0 } else { -1.0 };
        theta += sigma * THETA_TABLE[i];
        (x, y) = (x - sigma * y * p2i, y + sigma * x * p2i);
        p2i /= 2.0;
        i += 1;
    }

    y * K_TABLE[N - 1]
}
