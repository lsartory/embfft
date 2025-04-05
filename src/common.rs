/* embfft | common.rs
 * Copyright (c) 2025 L. Sartory
 * SPDX-License-Identifier: MIT
 */

/******************************************************************************/

pub struct Base<const N: usize>;

impl<const N: usize> Base<N> {
    /// Base 2 logarithm of the FFT size
    pub const LOG2_N: usize = {
        let mut x = N;
        let mut log2_n = 0;
        while x > 1 {
            x >>= 1;
            log2_n += 1;
        }
        log2_n
    };
    pub const IS_N_POW2: bool = {
        let mut n = 1;
        let mut i = Self::LOG2_N;
        while i > 0 {
            n <<= 1;
            i -= 1;
        }
        let is_n_pow2 = n == N;
        assert!(N >= 4, "The FFT data buffer size must be 4 or greater");
        assert!(is_n_pow2, "The FFT data buffer size must be a power of 2");
        is_n_pow2
    };

    /// Reverse all the bits in an integer
    ///
    /// Example: 0b11010010 --> 0b01001011
    /// Useful for sorting the output FFT values by frequency.
    pub const fn reverse_bits(x: usize) -> usize {
        let mut i = 0;
        let mut ret = 0;
        while i < Self::LOG2_N {
            ret |= ((x >> i) & 1) << ((Self::LOG2_N - 1) - i);
            i += 1;
        }
        ret
    }
}

/******************************************************************************/

pub struct SineTable<T, const N: usize>(T);

macro_rules! gen_sine_table {
    ($type: ty) => {
        {
            // TODO: the size should be N / 4...
            let mut table = [0.0; N];
            let mut i = 1;
            while i < N / 4 {
                table[i] = crate::cordic::sin(2.0 * core::f64::consts::PI * i as f64 / N as f64) as $type;
                i += 1;
            }
            table
        }
    }
}

impl<const N: usize> SineTable<f32, N> {
    pub const SINE_TABLE: [f32; N] = {
        gen_sine_table!(f32)
    };
}

impl<const N: usize> SineTable<f64, N> {
    pub const SINE_TABLE: [f64; N] = {
        gen_sine_table!(f64)
    };
}
