use std::env;
use std::fs::File;
use std::io::Write;
use std::path::Path;

fn compute_theta_table() -> [f64; 64] {
    (0..64)
        .map(|i| f64::atan2(1.0, f64::powf(2.0, i as _)))
        .collect::<Vec<f64>>()
        .try_into()
        .unwrap()
}

fn compute_k_table() -> [f64; 64] {
    let mut k = 1.0;
    (0..64)
        .map(|i| { k *= 1.0 / f64::sqrt(1.0 + f64::powf(2.0, -2.0 * i as f64)); k })
        .collect::<Vec<f64>>()
        .try_into()
        .unwrap()
}

fn main() {
    let out_dir = env::var_os("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("cordic_tables.rs");
    let mut f = File::create(dest_path).unwrap();

    let theta_table = compute_theta_table();
    writeln!(&mut f, "#[allow(clippy::approx_constant)]").unwrap();
    writeln!(&mut f, "const THETA_TABLE: [f64; {}] = {:?};", theta_table.len(), theta_table).unwrap();
    let k_table = compute_k_table();
    writeln!(&mut f, "const K_TABLE: [f64; {}] = {:?};", k_table.len(), k_table).unwrap();
}
