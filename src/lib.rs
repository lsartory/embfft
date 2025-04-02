#![no_std]

/******************************************************************************/

mod cordic;

/******************************************************************************/

pub struct EmbFFT<'a, T, const N: usize> {
    data: &'a mut [(T, T); N],
    state: State,
    length: usize,
    step: usize,
    step_size: usize,
    top_ptr: usize,
    bottom_ptr: usize
}

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

// TODO: f64 impl

impl<'a, const N: usize> EmbFFT<'a, f32, N> {
    const SINE_TABLE: [f32; N] = {
        // TODO: the size should be N / 4...
        let mut table = [0.0; N];
        let mut i = 1;
        while i < N / 4 {
            table[i] = cordic::sin(2.0 * core::f64::consts::PI * i as f64 / N as f64) as f32;
            i += 1;
        }
        table
    };
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
        assert!(is_n_pow2, "The FFT data buffer size must be a power of 2");
        is_n_pow2
    };

    const fn reverse_bits(x: usize) -> usize {
        let mut i = 0;
        let mut ret = 0;
        while i < Self::LOG2_N {
            ret |= ((x >> i) & 1) << ((Self::LOG2_N - 1) - i);
            i += 1;
        }
        ret
    }

    pub fn new(data: &'a mut [(f32, f32); N]) -> Self {
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

    pub fn fft_iterate(&mut self) {
        let top;
        let bottom;

        match self.state {
            State::Step1 => {
                // Twiddle = 1
                self.bottom_ptr = self.top_ptr + (self.length << 1);
                top = self.data[self.top_ptr];
                bottom = self.data[self.bottom_ptr];
                self.data[self.top_ptr].0 = 1.0 * bottom.0 + top.0;
                self.data[self.top_ptr].1 = 1.0 * bottom.1 + top.1;
                self.data[self.bottom_ptr].0 = -1.0 * bottom.0 + top.0;
                self.data[self.bottom_ptr].1 = -1.0 * bottom.1 + top.1;
                self.top_ptr += 1;
                self.bottom_ptr += 1;
                self.step = self.step_size;
                if self.step_size < N / 4 {
                    self.state = State::Step2;
                } else {
                    self.state = State::Step3;
                }
            },

            State::Step2 => {
                // Twiddle = e^(-j * theta)
                top = self.data[self.top_ptr];
                bottom = self.data[self.bottom_ptr];
                let temp = (-1.0 * bottom.0 + top.0, -1.0 * bottom.1 + top.1);
                self.data[self.top_ptr].0 = 1.0 * bottom.0 + top.0;
                self.data[self.top_ptr].1 = 1.0 * bottom.1 + top.1;
                self.data[self.bottom_ptr].0 = temp.0 * Self::SINE_TABLE[N / 4 - self.step] + temp.1 * Self::SINE_TABLE[self.step];
                self.data[self.bottom_ptr].1 = temp.1 * Self::SINE_TABLE[N / 4 - self.step] - temp.0 * Self::SINE_TABLE[self.step];
                self.top_ptr += 1;
                self.bottom_ptr += 1;
                if self.step < N / 4 - self.step_size {
                    self.step += self.step_size;
                } else {
                    self.state = State::Step3;
                }
            },

            State::Step3 => {
                // Twiddle = -j
                top = self.data[self.top_ptr];
                bottom = self.data[self.bottom_ptr];
                self.data[self.top_ptr].0 = 1.0 * bottom.0 + top.0;
                self.data[self.top_ptr].1 = 1.0 * bottom.1 + top.1;
                self.data[self.bottom_ptr].0 = -1.0 * bottom.1 + top.1;
                self.data[self.bottom_ptr].1 = -1.0 * top.0 + bottom.0;
                self.top_ptr += 1;
                self.bottom_ptr += 1;
                self.step = self.step_size;
                if self.step_size < N / 4 {
                    self.state = State::Step4;
                } else {
                    self.state = State::Step5;
                }
            },

            State::Step4 => {
                // Twiddle = -j * e^(-j * theta)
                top = self.data[self.top_ptr];
                bottom = self.data[self.bottom_ptr];
                let temp = (-1.0 * bottom.1 + top.1, -1.0 * top.0 + bottom.0);
                self.data[self.top_ptr].0 = 1.0 * bottom.0 + top.0;
                self.data[self.top_ptr].1 = 1.0 * bottom.1 + top.1;
                self.data[self.bottom_ptr].0 = temp.0 * Self::SINE_TABLE[N / 4 - self.step] + temp.1 * Self::SINE_TABLE[self.step];
                self.data[self.bottom_ptr].1 = temp.1 * Self::SINE_TABLE[N / 4 - self.step] - temp.0 * Self::SINE_TABLE[self.step];
                self.top_ptr += 1;
                self.bottom_ptr += 1;
                if self.step < N / 4 - self.step_size {
                    self.step += self.step_size;
                } else {
                    self.state = State::Step5;
                }
            },

            State::Step5 => {
                // Check if we need to loop
                if self.bottom_ptr < N {
                    self.top_ptr = self.bottom_ptr;
                    self.state = State::Step1;
                } else if self.length > 1 {
                    self.length >>= 1;
                    self.step_size <<= 1;
                    self.top_ptr = 0;
                    self.state = State::Step1;
                } else {
                    self.top_ptr = 0;
                    self.bottom_ptr = 1;
                    self.state = State::Step6;
                }
            },

            State::Step6 => {
                // Twiddle = 1
                top = self.data[self.top_ptr];
                bottom = self.data[self.bottom_ptr];
                self.data[self.top_ptr].0 = 1.0 * bottom.0 + top.0;
                self.data[self.top_ptr].1 = 1.0 * bottom.1 + top.1;
                self.data[self.bottom_ptr].0 = -1.0 * bottom.0 + top.0;
                self.data[self.bottom_ptr].1 = -1.0 * bottom.1 + top.1;
                if self.bottom_ptr < N - 2 {
                    self.top_ptr += 2;
                    self.bottom_ptr += 2;
                } else {
                    self.top_ptr = 0;
                    self.bottom_ptr = 0;
                    self.state = State::Reorder;
                }
            },

            State::Reorder => {
                // Ensure the output order is the same as the input
                top = self.data[self.top_ptr];
                bottom = self.data[self.bottom_ptr];
                if self.bottom_ptr > self.top_ptr {
                    self.data[self.top_ptr] = bottom;
                    self.data[self.bottom_ptr] = top;
                }
                if self.top_ptr < N - 1 {
                    self.bottom_ptr = Self::reverse_bits(self.top_ptr + 1);
                    self.top_ptr += 1;
                } else {
                    self.state = State::Done;
                }
            },

            State::Done => {}
        }
    }

    pub fn fft(&mut self) {
        while self.state != State::Done {
            self.fft_iterate();
        }
    }
}

/******************************************************************************/

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_ulps_eq;

    #[test]
    fn test_fft() {
        let mut data = [
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

        EmbFFT::new(&mut data).fft();

        for (x, y) in core::iter::zip(data, expected_data) {
            assert_ulps_eq!(x.0, y.0);
            assert_ulps_eq!(x.1, y.1);
        }
    }
}
