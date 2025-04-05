/* embfft | lib.rs
 * Copyright (c) 2025 L. Sartory
 * SPDX-License-Identifier: MIT
 */

/* Decimation in frequency fast Fourier transform */

/******************************************************************************/

#![no_std]
#![doc = include_str!("../README.md")]
#[warn(missing_docs)]

/******************************************************************************/

mod cordic;

/******************************************************************************/

/// The main EmbFft structure
///
/// It contains a reference to the input / output data, as well as information related to the
/// internal state.
pub struct EmbFft<'a, T, const N: usize> {
    data: &'a mut [(T, T); N],
    state: State,
    length: usize,
    step: usize,
    step_size: usize,
    top_ptr: usize,
    bottom_ptr: usize
}

/// Conversion state
#[derive(PartialEq)]
enum State {
    Step1,
    Step2,
    Step3,
    Step4,
    Step5,
    Step6,
    Reorder,
    Done
}

impl<'a, T, const N: usize> EmbFft<'a, T, N> {
    /// Base 2 logarithm of the FFT size
    const LOG2_N: usize = {
        let mut x = N;
        let mut log2_n = 0;
        while x > 1 {
            x >>= 1;
            log2_n += 1;
        }
        log2_n
    };
    const IS_N_POW2: bool = {
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
    const fn reverse_bits(x: usize) -> usize {
        let mut i = 0;
        let mut ret = 0;
        while i < Self::LOG2_N {
            ret |= ((x >> i) & 1) << ((Self::LOG2_N - 1) - i);
            i += 1;
        }
        ret
    }

    /// Initializes a new FFT conversion
    ///
    /// Use this function whenever a new conversion is required.
    pub fn new(data: &'a mut [(T, T); N]) -> Self {
        assert!(Self::IS_N_POW2);
        Self {
            data,
            state: State::Step1,
            length: N / 4,
            step: 0,
            step_size: 1,
            top_ptr: 0,
            bottom_ptr: 0
        }
    }

    /// Checks if the conversion is complete
    ///
    /// Use this together with the [`EmbFft::fft_iterate()`] function.
    pub fn is_done(&self) -> bool {
        self.state == State::Done
    }
}

/******************************************************************************/

macro_rules! gen_sine_table {
    ($type: ty) => {
        {
            // TODO: the size should be N / 4...
            let mut table = [0.0; N];
            let mut i = 1;
            while i < N / 4 {
                table[i] = cordic::sin(2.0 * core::f64::consts::PI * i as f64 / N as f64) as $type;
                i += 1;
            }
            table
        }
    }
}

macro_rules! gen_fft_iterate {
    ($self: ident) => {
        {
            let top;
            let bottom;

            match $self.state {
                State::Step1 => {
                    // Twiddle = 1
                    $self.bottom_ptr = $self.top_ptr + ($self.length << 1);
                    top = $self.data[$self.top_ptr];
                    bottom = $self.data[$self.bottom_ptr];
                    $self.data[$self.top_ptr].0 = 1.0 * bottom.0 + top.0;
                    $self.data[$self.top_ptr].1 = 1.0 * bottom.1 + top.1;
                    $self.data[$self.bottom_ptr].0 = -1.0 * bottom.0 + top.0;
                    $self.data[$self.bottom_ptr].1 = -1.0 * bottom.1 + top.1;
                    $self.top_ptr += 1;
                    $self.bottom_ptr += 1;
                    $self.step = $self.step_size;
                    if $self.step_size < N / 4 {
                        $self.state = State::Step2;
                    } else {
                        $self.state = State::Step3;
                    }
                },

                State::Step2 => {
                    // Twiddle = e^(-j * theta)
                    top = $self.data[$self.top_ptr];
                    bottom = $self.data[$self.bottom_ptr];
                    let temp = (-1.0 * bottom.0 + top.0, -1.0 * bottom.1 + top.1);
                    $self.data[$self.top_ptr].0 = 1.0 * bottom.0 + top.0;
                    $self.data[$self.top_ptr].1 = 1.0 * bottom.1 + top.1;
                    $self.data[$self.bottom_ptr].0 = temp.0 * Self::SINE_TABLE[N / 4 - $self.step] + temp.1 * Self::SINE_TABLE[$self.step];
                    $self.data[$self.bottom_ptr].1 = temp.1 * Self::SINE_TABLE[N / 4 - $self.step] - temp.0 * Self::SINE_TABLE[$self.step];
                    $self.top_ptr += 1;
                    $self.bottom_ptr += 1;
                    if $self.step < N / 4 - $self.step_size {
                        $self.step += $self.step_size;
                    } else {
                        $self.state = State::Step3;
                    }
                },

                State::Step3 => {
                    // Twiddle = -j
                    top = $self.data[$self.top_ptr];
                    bottom = $self.data[$self.bottom_ptr];
                    $self.data[$self.top_ptr].0 = 1.0 * bottom.0 + top.0;
                    $self.data[$self.top_ptr].1 = 1.0 * bottom.1 + top.1;
                    $self.data[$self.bottom_ptr].0 = -1.0 * bottom.1 + top.1;
                    $self.data[$self.bottom_ptr].1 = -1.0 * top.0 + bottom.0;
                    $self.top_ptr += 1;
                    $self.bottom_ptr += 1;
                    $self.step = $self.step_size;
                    if $self.step_size < N / 4 {
                        $self.state = State::Step4;
                    } else {
                        $self.state = State::Step5;
                    }
                },

                State::Step4 => {
                    // Twiddle = -j * e^(-j * theta)
                    top = $self.data[$self.top_ptr];
                    bottom = $self.data[$self.bottom_ptr];
                    let temp = (-1.0 * bottom.1 + top.1, -1.0 * top.0 + bottom.0);
                    $self.data[$self.top_ptr].0 = 1.0 * bottom.0 + top.0;
                    $self.data[$self.top_ptr].1 = 1.0 * bottom.1 + top.1;
                    $self.data[$self.bottom_ptr].0 = temp.0 * Self::SINE_TABLE[N / 4 - $self.step] + temp.1 * Self::SINE_TABLE[$self.step];
                    $self.data[$self.bottom_ptr].1 = temp.1 * Self::SINE_TABLE[N / 4 - $self.step] - temp.0 * Self::SINE_TABLE[$self.step];
                    $self.top_ptr += 1;
                    $self.bottom_ptr += 1;
                    if $self.step < N / 4 - $self.step_size {
                        $self.step += $self.step_size;
                    } else {
                        $self.state = State::Step5;
                    }
                },

                State::Step5 => {
                    // Check if we need to loop
                    if $self.bottom_ptr < N {
                        $self.top_ptr = $self.bottom_ptr;
                        $self.state = State::Step1;
                    } else if $self.length > 1 {
                        $self.length >>= 1;
                        $self.step_size <<= 1;
                        $self.top_ptr = 0;
                        $self.state = State::Step1;
                    } else {
                        $self.top_ptr = 0;
                        $self.bottom_ptr = 1;
                        $self.state = State::Step6;
                    }
                },

                State::Step6 => {
                    // Twiddle = 1
                    top = $self.data[$self.top_ptr];
                    bottom = $self.data[$self.bottom_ptr];
                    $self.data[$self.top_ptr].0 = 1.0 * bottom.0 + top.0;
                    $self.data[$self.top_ptr].1 = 1.0 * bottom.1 + top.1;
                    $self.data[$self.bottom_ptr].0 = -1.0 * bottom.0 + top.0;
                    $self.data[$self.bottom_ptr].1 = -1.0 * bottom.1 + top.1;
                    if $self.bottom_ptr < N - 2 {
                        $self.top_ptr += 2;
                        $self.bottom_ptr += 2;
                    } else {
                        $self.top_ptr = 0;
                        $self.bottom_ptr = 0;
                        $self.state = State::Reorder;
                    }
                },

                State::Reorder => {
                    // Ensure the output order is the same as the input
                    top = $self.data[$self.top_ptr];
                    bottom = $self.data[$self.bottom_ptr];
                    if $self.bottom_ptr > $self.top_ptr {
                        $self.data[$self.top_ptr] = bottom;
                        $self.data[$self.bottom_ptr] = top;
                    }
                    if $self.top_ptr < N - 1 {
                        $self.bottom_ptr = Self::reverse_bits($self.top_ptr + 1);
                        $self.top_ptr += 1;
                    } else {
                        $self.state = State::Done;
                    }
                },

                State::Done => {}
            }
        }
    }
}

/******************************************************************************/

impl<const N: usize> EmbFft<'_, f32, N> {
    const SINE_TABLE: [f32; N] = {
        gen_sine_table!(f32)
    };

    /// Non-blocking FFT computation with f32 precision
    ///
    /// Use this together with the [`EmbFft::is_done()`] function.
    /// For example:
    /// ```
    /// let mut data = [
    ///     (1.0f32, 1.0), (2.0, 2.0),
    ///     (3.0f32, 3.0), (4.0, 4.0),
    ///     (5.0f32, 5.0), (6.0, 6.0),
    ///     (7.0f32, 7.0), (8.0, 8.0)
    /// ];
    /// 
    /// let mut fft = embfft::EmbFft::new(&mut data);
    /// while !fft.is_done() {
    ///     fft.fft_iterate();
    ///     // Other actions can be performed here between two iterations
    /// }
    /// ```
    pub fn fft_iterate(&mut self) {
        gen_fft_iterate!(self);
    }

    /// Blocking FFT computation with f32 precision
    ///
    /// For example:
    /// ```
    /// let mut data = [
    ///     (1.0f32, 1.0), (2.0, 2.0),
    ///     (3.0f32, 3.0), (4.0, 4.0),
    ///     (5.0f32, 5.0), (6.0, 6.0),
    ///     (7.0f32, 7.0), (8.0, 8.0)
    /// ];
    /// embfft::EmbFft::new(&mut data).fft();
    /// ```
    pub fn fft(&mut self) {
        while self.state != State::Done {
            gen_fft_iterate!(self);
        }
    }
}

impl<const N: usize> EmbFft<'_, f64, N> {
    const SINE_TABLE: [f64; N] = {
        gen_sine_table!(f64)
    };

    /// Non-blocking FFT computation with f64 precision
    ///
    /// Use this together with the [`EmbFft::is_done()`] function.
    /// For example:
    /// ```
    /// let mut data = [
    ///     (1.0f64, 1.0), (2.0, 2.0),
    ///     (3.0f64, 3.0), (4.0, 4.0),
    ///     (5.0f64, 5.0), (6.0, 6.0),
    ///     (7.0f64, 7.0), (8.0, 8.0)
    /// ];
    /// 
    /// let mut fft = embfft::EmbFft::new(&mut data);
    /// while !fft.is_done() {
    ///     fft.fft_iterate();
    ///     // Other actions can be performed here between two iterations
    /// }
    /// ```
    pub fn fft_iterate(&mut self) {
        gen_fft_iterate!(self);
    }

    /// Blocking FFT computation with f64 precision
    ///
    /// For example:
    /// ```
    /// let mut data = [
    ///     (1.0f64, 1.0), (2.0, 2.0),
    ///     (3.0f64, 3.0), (4.0, 4.0),
    ///     (5.0f64, 5.0), (6.0, 6.0),
    ///     (7.0f64, 7.0), (8.0, 8.0)
    /// ];
    /// embfft::EmbFft::new(&mut data).fft();
    /// ```
    pub fn fft(&mut self) {
        while self.state != State::Done {
            gen_fft_iterate!(self);
        }
    }
}

/******************************************************************************/

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_ulps_eq;

    #[test]
    fn test_fft_f32() {
        let mut data: [(f32, f32); 64] = [
            ( 1.0, 0.0), ( 2.0, 0.0), ( 3.0, 0.0), ( 4.0, 0.0), ( 5.0, 0.0), ( 6.0, 0.0), ( 7.0, 0.0), ( 8.0, 0.0),
            ( 9.0, 0.0), (10.0, 0.0), (11.0, 0.0), (12.0, 0.0), (13.0, 0.0), (14.0, 0.0), (15.0, 0.0), (16.0, 0.0),
            (17.0, 0.0), (18.0, 0.0), (19.0, 0.0), (20.0, 0.0), (21.0, 0.0), (22.0, 0.0), (23.0, 0.0), (24.0, 0.0),
            (25.0, 0.0), (26.0, 0.0), (27.0, 0.0), (28.0, 0.0), (29.0, 0.0), (30.0, 0.0), (31.0, 0.0), (32.0, 0.0),
            (33.0, 0.0), (34.0, 0.0), (35.0, 0.0), (36.0, 0.0), (37.0, 0.0), (38.0, 0.0), (39.0, 0.0), (40.0, 0.0),
            (41.0, 0.0), (42.0, 0.0), (43.0, 0.0), (44.0, 0.0), (45.0, 0.0), (46.0, 0.0), (47.0, 0.0), (48.0, 0.0),
            (49.0, 0.0), (50.0, 0.0), (51.0, 0.0), (52.0, 0.0), (53.0, 0.0), (54.0, 0.0), (55.0, 0.0), (56.0, 0.0),
            (57.0, 0.0), (58.0, 0.0), (59.0, 0.0), (60.0, 0.0), (61.0, 0.0), (62.0, 0.0), (63.0, 0.0), (64.0, 0.0)
        ];

        let expected_data = [
            (2080.000000000,    0.000000000), ( -32.000000000,  651.374938965),
            ( -32.000000000,  324.901428223), ( -32.000000000,  215.726470947),
            ( -32.000000000,  160.874862671), ( -32.000000000,  127.751144409),
            ( -32.000000000,  105.489852905), ( -32.000000000,   89.433998108),
            ( -32.000000000,   77.254837036), ( -32.000000000,   67.658325195),
            ( -32.000000000,   59.867778778), ( -32.000000000,   53.388782501),
            ( -32.000000000,   47.891376495), ( -32.000000000,   43.147003174),
            ( -32.000000000,   38.992103577), ( -32.000000000,   35.306564331),
            ( -32.000000000,   32.000000000), ( -32.000000000,   29.003116608),
            ( -32.000000000,   26.261730194), ( -32.000000000,   23.732799530),
            ( -32.000000000,   21.381717682), ( -32.000000000,   19.180076599),
            ( -32.000000000,   17.104358673), ( -32.000000000,   15.134864807),
            ( -32.000000000,   13.254833221), ( -32.000000000,   11.449790955),
            ( -32.000000000,    9.707096100), ( -32.000000000,    8.015590668),
            ( -32.000000000,    6.365188599), ( -32.000000000,    4.746734619),
            ( -32.000000000,    3.151718140), ( -32.000000000,    1.572052002),
            ( -32.000000000,    0.000000000), ( -32.000000000,   -1.572082520),
            ( -32.000000000,   -3.151718140), ( -32.000000000,   -4.746765137),
            ( -32.000000000,   -6.365188599), ( -32.000000000,   -8.015590668),
            ( -32.000000000,   -9.707103729), ( -32.000000000,  -11.449790955),
            ( -32.000000000,  -13.254833221), ( -32.000000000,  -15.134864807),
            ( -32.000000000,  -17.104343414), ( -32.000000000,  -19.180065155),
            ( -32.000000000,  -21.381710052), ( -32.000000000,  -23.732810974),
            ( -32.000000000,  -26.261726379), ( -32.000000000,  -29.003128052),
            ( -32.000000000,  -32.000000000), ( -32.000000000,  -35.306552887),
            ( -32.000000000,  -38.992107391), ( -32.000000000,  -43.147006989),
            ( -32.000000000,  -47.891384125), ( -32.000000000,  -53.388763428),
            ( -32.000000000,  -59.867778778), ( -32.000000000,  -67.658317566),
            ( -32.000000000,  -77.254837036), ( -32.000000000,  -89.434005737),
            ( -32.000000000, -105.489868164), ( -32.000000000, -127.751144409),
            ( -32.000000000, -160.874862671), ( -32.000000000, -215.726470947),
            ( -32.000000000, -324.901428223), ( -32.000000000, -651.374877930)
        ];

        EmbFft::new(&mut data).fft();

        for (x, y) in core::iter::zip(data, expected_data) {
            assert_ulps_eq!(x.0, y.0);
            assert_ulps_eq!(x.1, y.1);
        }
    }

    #[test]
    fn test_fft_f64() {
        let mut data: [(f64, f64); 64] = [
            ( 1.0, 0.0), ( 2.0, 0.0), ( 3.0, 0.0), ( 4.0, 0.0), ( 5.0, 0.0), ( 6.0, 0.0), ( 7.0, 0.0), ( 8.0, 0.0),
            ( 9.0, 0.0), (10.0, 0.0), (11.0, 0.0), (12.0, 0.0), (13.0, 0.0), (14.0, 0.0), (15.0, 0.0), (16.0, 0.0),
            (17.0, 0.0), (18.0, 0.0), (19.0, 0.0), (20.0, 0.0), (21.0, 0.0), (22.0, 0.0), (23.0, 0.0), (24.0, 0.0),
            (25.0, 0.0), (26.0, 0.0), (27.0, 0.0), (28.0, 0.0), (29.0, 0.0), (30.0, 0.0), (31.0, 0.0), (32.0, 0.0),
            (33.0, 0.0), (34.0, 0.0), (35.0, 0.0), (36.0, 0.0), (37.0, 0.0), (38.0, 0.0), (39.0, 0.0), (40.0, 0.0),
            (41.0, 0.0), (42.0, 0.0), (43.0, 0.0), (44.0, 0.0), (45.0, 0.0), (46.0, 0.0), (47.0, 0.0), (48.0, 0.0),
            (49.0, 0.0), (50.0, 0.0), (51.0, 0.0), (52.0, 0.0), (53.0, 0.0), (54.0, 0.0), (55.0, 0.0), (56.0, 0.0),
            (57.0, 0.0), (58.0, 0.0), (59.0, 0.0), (60.0, 0.0), (61.0, 0.0), (62.0, 0.0), (63.0, 0.0), (64.0, 0.0)
        ];

        let expected_data = [
            (2080.000000000000000,    0.000000000000000), (-32.000000000000000,  651.374963999590136),
            ( -32.000000000000000,  324.901452403483631), (-32.000000000000000,  215.726476973279688),
            ( -32.000000000000000,  160.874863748027195), (-32.000000000000000,  127.751161080642859),
            ( -32.000000000000000,  105.489862686026299), (-32.000000000000000,   89.434008719695299),
            ( -32.000000000000000,   77.254833995939066), (-32.000000000000000,   67.658315441556596),
            ( -32.000000000000000,   59.867789177260548), (-32.000000000000000,   53.388774578672361),
            ( -32.000000000000000,   47.891384405295703), (-32.000000000000000,   43.147005231575143),
            ( -32.000000000000000,   38.992112818815343), (-32.000000000000000,   35.306559223471368),
            ( -32.000000000000000,   32.000000000000000), (-32.000000000000000,   29.003109408612701),
            ( -32.000000000000000,   26.261721306517153), (-32.000000000000000,   23.732817480705215),
            ( -32.000000000000000,   21.381716413417571), (-32.000000000000000,   19.180061877821572),
            ( -32.000000000000000,   17.104356350425391), (-32.000000000000000,   15.134872828522347),
            ( -32.000000000000000,   13.254833995939073), (-32.000000000000000,   11.449783082064791),
            ( -32.000000000000000,    9.707093875434992), (-32.000000000000000,    8.015582726121906),
            ( -32.000000000000000,    6.365195756149134), (-32.000000000000000,    4.746751601227075),
            ( -32.000000000000000,    3.151724907429411), (-32.000000000000000,    1.572059192623158),
            ( -32.000000000000000,    0.000000000000000), (-32.000000000000000,   -1.572059192622930),
            ( -32.000000000000000,   -3.151724907429269), (-32.000000000000000,   -4.746751601227089),
            ( -32.000000000000000,   -6.365195756149063), (-32.000000000000000,   -8.015582726121764),
            ( -32.000000000000000,   -9.707093875434900), (-32.000000000000000,  -11.449783082064613),
            ( -32.000000000000000,  -13.254833995939073), (-32.000000000000000,  -15.134872828522283),
            ( -32.000000000000000,  -17.104356350425405), (-32.000000000000000,  -19.180061877821579),
            ( -32.000000000000000,  -21.381716413417557), (-32.000000000000000,  -23.732817480705158),
            ( -32.000000000000000,  -26.261721306517074), (-32.000000000000000,  -29.003109408612545),
            ( -32.000000000000000,  -32.000000000000000), (-32.000000000000000,  -35.306559223471240),
            ( -32.000000000000000,  -38.992112818815279), (-32.000000000000000,  -43.147005231575015),
            ( -32.000000000000000,  -47.891384405295717), (-32.000000000000000,  -53.388774578672383),
            ( -32.000000000000000,  -59.867789177260505), (-32.000000000000000,  -67.658315441556496),
            ( -32.000000000000000,  -77.254833995939066), (-32.000000000000000,  -89.434008719695356),
            ( -32.000000000000000, -105.489862686026427), (-32.000000000000000, -127.751161080642916),
            ( -32.000000000000000, -160.874863748027281), (-32.000000000000000, -215.726476973279944),
            ( -32.000000000000000, -324.901452403483972), (-32.000000000000000, -651.374963999591046)
        ];

        EmbFft::new(&mut data).fft();

        for (x, y) in core::iter::zip(data, expected_data) {
            assert_ulps_eq!(x.0, y.0);
            assert_ulps_eq!(x.1, y.1);
        }
    }
}
