# EmbFft

EmbFft is an easy to use Fast Fourier Transform library primarily meant for embedded systems.
It is based on a state machine that allows other tasks to be performed between iterations, without having to rely on multi-threading or async functions.


## Features

The library offers the following features:
* Easy to use
* Non-blocking processing
* Suitable for `no_std` environments
* No external dependencies at run time
* Low RAM requirements thanks to in-place conversion; no memory allocation is required
* All trigonometry-related computations are performed at compile time, only additions and multiplications are required at run time
* Supports any buffer size greater than 4, as long as it is a power of two
* Allows single-precision (f32) as well as double-precision (f64) conversions


## Limitations

Because of the FFT algorithm used, the following limitations exist:
* ROM space is required to store pre-computed sine tables
* Buffers must be a power of 2 in size


## Examples

### Non-blocking
```rust
let mut data = [
    (1.0f32, 1.0), (2.0, 2.0),
    (3.0f32, 3.0), (4.0, 4.0),
    (5.0f32, 5.0), (6.0, 6.0),
    (7.0f32, 7.0), (8.0, 8.0)
];

let mut fft = embfft::EmbFft::new(&mut data);
while !fft.is_done() {
    fft.fft_iterate();
    // Other actions can be performed here between two iterations
}

for x in data {
    println!("{:?}", x);
}
```

### Blocking
```rust
let mut data = [
    (1.0f32, 1.0), (2.0, 2.0),
    (3.0f32, 3.0), (4.0, 4.0),
    (5.0f32, 5.0), (6.0, 6.0),
    (7.0f32, 7.0), (8.0, 8.0)
];

embfft::EmbFft::new(&mut data).fft();

for x in data {
    println!("{:?}", x);
}
```


## Acknowledgement

Thanks to Robert Bristow-Johnson for the explanation of the algorithm, as well as for the example implementation.


## Changelog

| Date       | Version | Changes                                  |
|------------|---------|------------------------------------------|
| 2025-04-09 | 0.2.1   | Code cleanup and performance improvement |
| 2025-04-06 | 0.2.0   | Add inverse FFT                          |
| 2025-04-05 | 0.1.0   | Initial release                          |
