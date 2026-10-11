//! Wall time of the two-stage dense spectrum against the cyclic Jacobi solver.
//!
//! ```text
//! cargo bench --features jacobi,bench --bench eigh
//! ```
//!
//! Sizes are 30, 300, 1500 and 3000 unless arguments select a subset.
//! `EIGH_WHICH=eigh`, `jacobi`, or `both` (the default) selects a solver.
//! Each class is three random symmetric shifts of `M^T M`. The matrices
//! are built outside the timed section. Solvers are interleaved. Each
//! reported time is the median of five solves, with the range (max - min).
//! The timed section checks the residual of the first column. Host load
//! is the first field of `/proc/loadavg`. Ratios against faer are wall
//! times from `eigh_scale`; this binary does not call faer.

use std::env;
use std::fs;
use std::hint::black_box;
use std::time::Instant;

use ndarray::Array2;
use rgsaddle::{exact_eigh, jacobi_eigh};

fn next_open(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(0x5851_F42D_4C95_7F2D)
        .wrapping_add(0x1405_7B7E_F767_814F);
    let bits = (*state >> 11) as u32;
    (f64::from(bits) / f64::from(u32::MAX)).mul_add(2.0, -1.0)
}

fn random_spd(n: usize, seed: u64) -> Array2<f64> {
    let mut state = seed;
    let mut factor = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            factor[(i, j)] = next_open(&mut state);
        }
    }
    let mut a = factor.t().dot(&factor);
    for i in 0..n {
        a[(i, i)] += 0.5;
    }
    a
}

fn residual_first(a: &Array2<f64>, eval: f64, vecs: &ndarray::Array2<f64>) -> f64 {
    let n = a.nrows();
    let mut err = 0.0;
    for row in 0..n {
        let mut acc = 0.0;
        for col in 0..n {
            acc += a[(row, col)] * vecs[(col, 0)];
        }
        let diff = acc - eval * vecs[(row, 0)];
        err += diff * diff;
    }
    err.sqrt()
}

fn load() -> String {
    fs::read_to_string("/proc/loadavg")
        .unwrap_or_else(|_| "unknown".into())
        .split_whitespace()
        .next()
        .unwrap_or("unknown")
        .to_string()
}

const REPS: usize = 5;
const SEEDS: [u64; 3] = [0xE176_5ADD, 0xE176_0002, 0xE176_0003];

fn median_and_range(samples: &mut [f64]) -> (f64, f64) {
    samples.sort_by(f64::total_cmp);
    let median = samples[samples.len() / 2];
    let range = samples[samples.len() - 1] - samples[0];
    (median, range)
}

fn main() {
    let sizes: Vec<usize> = env::args()
        .skip(1)
        .filter(|arg| !arg.starts_with("--"))
        .map(|arg| {
            arg.parse::<usize>()
                .unwrap_or_else(|_| panic!("size must be an integer, got {arg}"))
        })
        .collect();
    let sizes = if sizes.is_empty() {
        vec![30, 300, 1500, 3000]
    } else {
        sizes
    };
    println!("rgsaddle {}", env!("CARGO_PKG_VERSION"));
    println!("solver two-stage dense spectrum vs cyclic Jacobi");
    println!("host_load_1m={}", load());
    for n in sizes {
        for (matrix, seed) in SEEDS.iter().enumerate() {
            println!(
                "building n={n} matrix={matrix} seed={seed:#x} load={}",
                load()
            );
            let a = random_spd(n, *seed);
            let which = env::var("EIGH_WHICH").unwrap_or_else(|_| "both".into());
            let run_eigh = which == "both" || which == "eigh" || which == "faer";
            // Five Jacobi solves at n = 1500 are hours. The default comparison
            // keeps Jacobi on the sizes that finish in a run of five.
            let run_jacobi = which == "jacobi" || (which == "both" && n < 1500);
            if run_eigh {
                black_box(exact_eigh(black_box(a.view())).expect("eigh warmup"));
            }
            if run_jacobi {
                black_box(jacobi_eigh(black_box(a.view())).expect("jacobi warmup"));
            }
            let mut eigh_samples = Vec::new();
            let mut jacobi_samples = Vec::new();
            let mut eigh_lowest = f64::NAN;
            let mut jacobi_lowest = f64::NAN;
            println!(
                "timing n={n} matrix={matrix} reps={REPS} interleaved load={}",
                load()
            );
            for _ in 0..REPS {
                if run_eigh {
                    let start = Instant::now();
                    let (evals, vecs) = exact_eigh(black_box(a.view())).expect("eigh");
                    let residual = residual_first(&a, evals[0], &vecs);
                    assert!(residual.is_finite());
                    eigh_samples.push(start.elapsed().as_secs_f64());
                    eigh_lowest = evals[0];
                    black_box(eigh_lowest);
                    black_box(residual);
                }
                if run_jacobi {
                    let start = Instant::now();
                    let (evals, vecs) = jacobi_eigh(black_box(a.view())).expect("jacobi");
                    let residual = residual_first(&a, evals[0], &vecs);
                    assert!(residual.is_finite());
                    jacobi_samples.push(start.elapsed().as_secs_f64());
                    jacobi_lowest = evals[0];
                    black_box(jacobi_lowest);
                    black_box(residual);
                }
            }
            let (eigh_median, eigh_range) = if run_eigh {
                median_and_range(&mut eigh_samples)
            } else {
                (f64::NAN, f64::NAN)
            };
            let (jacobi_median, jacobi_range) = if run_jacobi {
                median_and_range(&mut jacobi_samples)
            } else {
                (f64::NAN, f64::NAN)
            };
            let gap = if run_eigh && run_jacobi {
                (eigh_lowest - jacobi_lowest).abs()
            } else {
                f64::NAN
            };
            println!(
                "n={n} matrix={matrix} reps={REPS} eigh_median_s={eigh_median:.6} eigh_range_s={eigh_range:.6} jacobi_median_s={jacobi_median:.6} jacobi_range_s={jacobi_range:.6} lowest_gap={gap:.3e} load={}",
                load()
            );
        }
    }
}
