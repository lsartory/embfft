/* embfft | ifft.rs
 * Copyright (c) 2025 L. Sartory
 * SPDX-License-Identifier: MIT
 */

/******************************************************************************/

use crate::common::{Base, Float};

/******************************************************************************/

/// Decimation in time inverse fast Fourier transform
///
/// This structure contains a reference to the input / output data, as well as information related to the
/// internal state.
pub struct EmbIfft<'a, T, const N: usize> {
    data: &'a mut [(T, T); N],
    state: State,
    length: usize,
    step: usize,
    step_size: usize,
    top_idx: usize,
    bottom_idx: usize
}

/// Conversion state
#[derive(PartialEq)]
enum State {
    Reorder,
    Step1,
    Step2,
    Step3,
    Step4,
    Step5,
    Step6,
    Done
}

impl<'a, T: Float<N>, const N: usize> EmbIfft<'a, T, N> {
    /// Initializes a new IFFT conversion
    ///
    /// Use this function whenever a new conversion is required.
    pub fn new(data: &'a mut [(T, T); N]) -> Self {
        assert!(Base::<N>::IS_N_POW2);
        Self {
            data,
            state: State::Reorder,
            length: 1,
            step: 0,
            step_size: N / 4,
            top_idx: 0,
            bottom_idx: 0
        }
    }

    fn reorder(&mut self) {
        // Ensure the input order is reversed
        let top = self.data[self.top_idx];
        let bottom = self.data[self.bottom_idx];
        if self.bottom_idx > self.top_idx {
            self.data[self.top_idx] = bottom;
            self.data[self.bottom_idx] = top;
        }
        if self.top_idx < N - 1 {
            self.bottom_idx = Base::<N>::reverse_bits(self.top_idx + 1);
            self.top_idx += 1;
        } else {
            self.top_idx = 0;
            self.bottom_idx = 1;
            self.state = State::Step1;
        }
    }

    fn step1(&mut self) {
        // Twiddle = 1 / N
        let top = self.data[self.top_idx];
        let bottom = self.data[self.bottom_idx];
        self.data[self.top_idx].0 = (bottom.0 + top.0) * T::N_INV;
        self.data[self.top_idx].1 = (bottom.1 + top.1) * T::N_INV;
        self.data[self.bottom_idx].0 = (-bottom.0 + top.0) * T::N_INV;
        self.data[self.bottom_idx].1 = (-bottom.1 + top.1) * T::N_INV;
        if self.bottom_idx < N - 2 {
            self.top_idx += 2;
            self.bottom_idx += 2;
        } else {
            self.top_idx = 0;
            self.state = State::Step2;
        }
    }

    fn step2(&mut self) {
        // Twiddle = 1
        self.bottom_idx = self.top_idx + (self.length << 1);
        let top = self.data[self.top_idx];
        let bottom = self.data[self.bottom_idx];
        self.data[self.top_idx].0 = bottom.0 + top.0;
        self.data[self.top_idx].1 = bottom.1 + top.1;
        self.data[self.bottom_idx].0 = top.0 - bottom.0;
        self.data[self.bottom_idx].1 = top.1 - bottom.1;
        self.top_idx += 1;
        self.bottom_idx += 1;
        self.step = self.step_size;
        if self.step_size < N / 4 {
            self.state = State::Step3;
        } else {
            self.state = State::Step4;
        }
    }

    fn step3(&mut self) {
        // Twiddle = e^(+j * theta)
        let top = self.data[self.top_idx];
        let bottom = self.data[self.bottom_idx];
        let temp = (
            bottom.0 * T::SINE_TABLE[N / 4 - self.step] - bottom.1 * T::SINE_TABLE[self.step],
            bottom.1 * T::SINE_TABLE[N / 4 - self.step] + bottom.0 * T::SINE_TABLE[self.step]
        );
        self.data[self.top_idx].0 = top.0 + temp.0;
        self.data[self.top_idx].1 = top.1 + temp.1;
        self.data[self.bottom_idx].0 = top.0 - temp.0;
        self.data[self.bottom_idx].1 = top.1 - temp.1;
        self.top_idx += 1;
        self.bottom_idx += 1;
        if self.step < N / 4 - self.step_size {
            self.step += self.step_size;
        } else {
            self.state = State::Step4;
        }
    }

    fn step4(&mut self) {
        // Twiddle = +j
        let top = self.data[self.top_idx];
        let bottom = self.data[self.bottom_idx];
        self.data[self.top_idx].0 = top.0 - bottom.1;
        self.data[self.top_idx].1 = top.1 + bottom.0;
        self.data[self.bottom_idx].0 = top.0 + bottom.1;
        self.data[self.bottom_idx].1 = top.1 - bottom.0;
        self.top_idx += 1;
        self.bottom_idx += 1;
        self.step = self.step_size;
        if self.step_size < N / 4 {
            self.state = State::Step5;
        } else {
            self.state = State::Step6;
        }
    }

    fn step5(&mut self) {
        // Twiddle = +j * e^(+j * theta)
        let top = self.data[self.top_idx];
        let bottom = self.data[self.bottom_idx];
        let temp = (
            -bottom.1 * T::SINE_TABLE[N / 4 - self.step] - bottom.0 * T::SINE_TABLE[self.step],
            bottom.0 * T::SINE_TABLE[N / 4 - self.step] - bottom.1 * T::SINE_TABLE[self.step]
        );
        self.data[self.top_idx].0 = top.0 + temp.0;
        self.data[self.top_idx].1 = top.1 + temp.1;
        self.data[self.bottom_idx].0 = top.0 - temp.0;
        self.data[self.bottom_idx].1 = top.1 - temp.1;
        self.top_idx += 1;
        self.bottom_idx += 1;
        if self.step < N / 4 - self.step_size {
            self.step += self.step_size;
        } else {
            self.state = State::Step6;
        }
    }

    fn step6(&mut self) {
        // Check if we need to loop
        if self.bottom_idx < N {
            self.top_idx = self.bottom_idx;
            self.state = State::Step2;
        } else if self.step_size > 1 {
            self.length <<= 1;
            self.step_size >>= 1;
            self.top_idx = 0;
            self.state = State::Step2;
        } else {
            self.state = State::Done;
        }
    }

    /// Non-blocking IFFT computation
    ///
    /// Use this together with the [`EmbIfft::is_done()`] function.
    /// For example:
    /// ```
    /// let mut data = [
    ///     (1.0f32, 1.0), (2.0, 2.0),
    ///     (3.0f32, 3.0), (4.0, 4.0),
    ///     (5.0f32, 5.0), (6.0, 6.0),
    ///     (7.0f32, 7.0), (8.0, 8.0)
    /// ];
    ///
    /// let mut ifft = embfft::EmbIfft::new(&mut data);
    /// while !ifft.is_done() {
    ///     ifft.ifft_iterate();
    ///     // Other actions can be performed here between two iterations
    /// }
    /// ```
    pub fn ifft_iterate(&mut self) {
        match self.state {
            State::Reorder => { self.reorder(); },
            State::Step1 => { self.step1(); },
            State::Step2 => { self.step2(); },
            State::Step3 => { self.step3(); },
            State::Step4 => { self.step4(); },
            State::Step5 => { self.step5(); },
            State::Step6 => { self.step6(); },
            State::Done => {}
        }
    }

    /// Blocking IFFT computation
    ///
    /// For example:
    /// ```
    /// let mut data = [
    ///     (1.0f32, 1.0), (2.0, 2.0),
    ///     (3.0f32, 3.0), (4.0, 4.0),
    ///     (5.0f32, 5.0), (6.0, 6.0),
    ///     (7.0f32, 7.0), (8.0, 8.0)
    /// ];
    /// embfft::EmbIfft::new(&mut data).ifft();
    /// ```
    pub fn ifft(&mut self) {
        while self.state != State::Done {
            self.ifft_iterate();
        }
    }

    /// Checks if the conversion is complete
    ///
    /// Use this together with the [`EmbIfft::ifft_iterate()`] function.
    pub fn is_done(&self) -> bool {
        self.state == State::Done
    }
}

/******************************************************************************/

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_ulps_eq;

    #[test]
    fn test_ifft_f32() {
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
            (32.500000000,  0.000000000), (-0.500000000, -10.177733812),
            (-0.500000000, -5.076585194), (-0.500000000,  -3.370726203),
            (-0.500000000, -2.513669746), (-0.500000000,  -1.996111892),
            (-0.500000000, -1.648279104), (-0.500000000,  -1.397406386),
            (-0.500000000, -1.207106781), (-0.500000000,  -1.057161179),
            (-0.500000000, -0.935434206), (-0.500000000,  -0.834199603),
            (-0.500000000, -0.748302881), (-0.500000000,  -0.674171957),
            (-0.500000000, -0.609251763), (-0.500000000,  -0.551664988),
            (-0.500000000, -0.500000000), (-0.500000000,  -0.453173585),
            (-0.500000000, -0.410339395), (-0.500000000,  -0.370825273),
            (-0.500000000, -0.334089319), (-0.500000000,  -0.299688467),
            (-0.500000000, -0.267255568), (-0.500000000,  -0.236482388),
            (-0.500000000, -0.207106781), (-0.500000000,  -0.178902861),
            (-0.500000000, -0.151673342), (-0.500000000,  -0.125243480),
            (-0.500000000, -0.099456184), (-0.500000000,  -0.074167994),
            (-0.500000000, -0.049245702), (-0.500000000,  -0.024563425),
            (-0.500000000,  0.000000000), (-0.500000000,   0.024563425),
            (-0.500000000,  0.049245702), (-0.500000000,   0.074167994),
            (-0.500000000,  0.099456184), (-0.500000000,   0.125243480),
            (-0.500000000,  0.151673342), (-0.500000000,   0.178902861),
            (-0.500000000,  0.207106781), (-0.500000000,   0.236482388),
            (-0.500000000,  0.267255568), (-0.500000000,   0.299688467),
            (-0.500000000,  0.334089319), (-0.500000000,   0.370825273),
            (-0.500000000,  0.410339395), (-0.500000000,   0.453173585),
            (-0.500000000,  0.500000000), (-0.500000000,   0.551664988),
            (-0.500000000,  0.609251763), (-0.500000000,   0.674171957),
            (-0.500000000,  0.748302881), (-0.500000000,   0.834199603),
            (-0.500000000,  0.935434206), (-0.500000000,   1.057161179),
            (-0.500000000,  1.207106781), (-0.500000000,   1.397406386),
            (-0.500000000,  1.648279104), (-0.500000000,   1.996111892),
            (-0.500000000,  2.513669746), (-0.500000000,   3.370726203),
            (-0.500000000,  5.076585194), (-0.500000000,  10.177733812)
        ];

        EmbIfft::new(&mut data).ifft();

        for (x, y) in core::iter::zip(data, expected_data) {
            assert_ulps_eq!(x.0, y.0, max_ulps = 10);
            assert_ulps_eq!(x.1, y.1, max_ulps = 10);
        }
    }

    #[test]
    fn test_ifft_f64() {
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
            (32.500000000000000,  0.000000000000000), (-0.500000000000000, -10.177733812493605),
            (-0.500000000000000, -5.076585193804434), (-0.500000000000000,  -3.370726202707498),
            (-0.500000000000000, -2.513669746062925), (-0.500000000000000,  -1.996111891885044),
            (-0.500000000000000, -1.648279104469162), (-0.500000000000000,  -1.397406386245239),
            (-0.500000000000000, -1.207106781186548), (-0.500000000000000,  -1.057161178774320),
            (-0.500000000000000, -0.935434205894695), (-0.500000000000000,  -0.834199602791755),
            (-0.500000000000000, -0.748302881332745), (-0.500000000000000,  -0.674171956743360),
            (-0.500000000000000, -0.609251762793989), (-0.500000000000000,  -0.551664987866739),
            (-0.500000000000000, -0.500000000000000), (-0.500000000000000,  -0.453173584509572),
            (-0.500000000000000, -0.410339395414330), (-0.500000000000000,  -0.370825273136018),
            (-0.500000000000000, -0.334089318959649), (-0.500000000000000,  -0.299688466840962),
            (-0.500000000000000, -0.267255567975397), (-0.500000000000000,  -0.236482387945661),
            (-0.500000000000000, -0.207106781186548), (-0.500000000000000,  -0.178902860657261),
            (-0.500000000000000, -0.151673341803671), (-0.500000000000000,  -0.125243480095653),
            (-0.500000000000000, -0.099456183689830), (-0.500000000000000,  -0.074167993769174),
            (-0.500000000000000, -0.049245701678584), (-0.500000000000000,  -0.024563424884736),
            (-0.500000000000000,  0.000000000000000), (-0.500000000000000,   0.024563424884736),
            (-0.500000000000000,  0.049245701678584), (-0.500000000000000,   0.074167993769174),
            (-0.500000000000000,  0.099456183689830), (-0.500000000000000,   0.125243480095653),
            (-0.500000000000000,  0.151673341803671), (-0.500000000000000,   0.178902860657261),
            (-0.500000000000000,  0.207106781186548), (-0.500000000000000,   0.236482387945661),
            (-0.500000000000000,  0.267255567975397), (-0.500000000000000,   0.299688466840962),
            (-0.500000000000000,  0.334089318959649), (-0.500000000000000,   0.370825273136018),
            (-0.500000000000000,  0.410339395414330), (-0.500000000000000,   0.453173584509572),
            (-0.500000000000000,  0.500000000000000), (-0.500000000000000,   0.551664987866739),
            (-0.500000000000000,  0.609251762793989), (-0.500000000000000,   0.674171956743360),
            (-0.500000000000000,  0.748302881332745), (-0.500000000000000,   0.834199602791755),
            (-0.500000000000000,  0.935434205894695), (-0.500000000000000,   1.057161178774320),
            (-0.500000000000000,  1.207106781186548), (-0.500000000000000,   1.397406386245239),
            (-0.500000000000000,  1.648279104469162), (-0.500000000000000,   1.996111891885044),
            (-0.500000000000000,  2.513669746062925), (-0.500000000000000,   3.370726202707498),
            (-0.500000000000000,  5.076585193804434), (-0.500000000000000,  10.177733812493605)
        ];
        EmbIfft::new(&mut data).ifft();

        for (x, y) in core::iter::zip(data, expected_data) {
            assert_ulps_eq!(x.0, y.0, max_ulps = 75);
            assert_ulps_eq!(x.1, y.1, max_ulps = 75);
        }
    }
}
