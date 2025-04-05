/* embfft | lib.rs
 * Copyright (c) 2025 L. Sartory
 * SPDX-License-Identifier: MIT
 */

/******************************************************************************/

#![no_std]
#![doc = include_str!("../README.md")]
#[warn(missing_docs)]

/******************************************************************************/

mod common;
mod cordic;
mod fft;

pub use crate::fft::EmbFft;
