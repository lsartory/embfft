# EmbFft

EmbFft is an easy to use FFT library primarily meant for embedded systems.
It is based on a state machine that allows performing other tasks between iterations, without having to rely on multi-threading or async functions.

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

Because of the algorithm used, the following limitations exist:
* ROM space is required to hold the precomputed sine tables
* Buffers must have a power of 2 size
* Inverse FFT is not implemented yet

## Examples

### Non-blocking
```Rust
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
```Rust
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
