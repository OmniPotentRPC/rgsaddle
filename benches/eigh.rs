//! Wall time of the faer dense spectrum against the cyclic Jacobi solver.
//!
//! ```text
//! cargo bench --features jacobi --bench eigh
//! ```
//!
//! Sizes are 30, 300, 1500 and 3000 unless arguments select a subset.
//! `EIGH_WHICH=faer`, `jacobi`, or `both` (the default) selects a solver.
//! The matrix is a random symmetric shift of `M^T M`, positive definite,
//! built outside the timed section. Solvers are interleaved. Each reported
//! time is the median of five solves, with the range (max - min).
//! rgsaddle 0.2.2, faer 0.24.4, pulp runtime SIMD, `Par::Seq`.

use std::env;
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

fn random_spd(n: usize) -> Array2<f64> {
    let mut state = 0xE176_5ADD;
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

const REPS: usize = 5;

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
    println!("faer 0.24.4");
    println!("solver faer self-adjoint (pulp runtime SIMD) vs cyclic Jacobi");
    for n in sizes {
        println!("building n={n}");
        let a = random_spd(n);
        let which = env::var("EIGH_WHICH").unwrap_or_else(|_| "both".into());
        let run_faer = which == "both" || which == "faer";
        // Five Jacobi solves at n = 1500 are hours. The default comparison
        // keeps Jacobi on the sizes that finish in a run of five.
        let run_jacobi = which == "jacobi" || (which == "both" && n < 1500);
        if run_faer {
            black_box(exact_eigh(black_box(a.view())).expect("faer warmup"));
        }
        if run_jacobi {
            black_box(jacobi_eigh(black_box(a.view())).expect("jacobi warmup"));
        }
        let mut faer_samples = Vec::new();
        let mut jacobi_samples = Vec::new();
        let mut faer_lowest = f64::NAN;
        let mut jacobi_lowest = f64::NAN;
        println!("timing n={n} reps={REPS} interleaved");
        for _ in 0..REPS {
            if run_faer {
                let start = Instant::now();
                let (evals, _) = exact_eigh(black_box(a.view())).expect("faer");
                faer_samples.push(start.elapsed().as_secs_f64());
                faer_lowest = evals[0];
                black_box(faer_lowest);
            }
            if run_jacobi {
                let start = Instant::now();
                let (evals, _) = jacobi_eigh(black_box(a.view())).expect("jacobi");
                jacobi_samples.push(start.elapsed().as_secs_f64());
                jacobi_lowest = evals[0];
                black_box(jacobi_lowest);
            }
        }
        let (faer_median, faer_range) = if run_faer {
            median_and_range(&mut faer_samples)
        } else {
            (f64::NAN, f64::NAN)
        };
        let (jacobi_median, jacobi_range) = if run_jacobi {
            median_and_range(&mut jacobi_samples)
        } else {
            (f64::NAN, f64::NAN)
        };
        let gap = if run_faer && run_jacobi {
            (faer_lowest - jacobi_lowest).abs()
        } else {
            f64::NAN
        };
        println!(
            "n={n} reps={REPS} faer_median_s={faer_median:.6} faer_range_s={faer_range:.6} jacobi_median_s={jacobi_median:.6} jacobi_range_s={jacobi_range:.6} lowest_gap={gap:.3e}"
        );
    }
}
