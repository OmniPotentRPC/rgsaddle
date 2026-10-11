//! Wall time of the faer dense spectrum against the cyclic Jacobi solver.
//!
//! ```text
//! cargo bench --features jacobi --bench eigh
//! ```
//!
//! Sizes are 30, 300, 1500 and 3000 unless arguments select a subset.
//! `EIGH_WHICH=faer`, `jacobi`, or `both` (the default) selects a solver.
//! The matrix is a random symmetric shift of `M^T M`, positive definite,
//! built outside the timed section. Reported times are mean seconds per
//! solve: 20 repetitions at n = 30, 3 at n = 300, and 1 above that.
//! Sizes below 1500 take one untimed warmup. rgsaddle 0.2.2, faer 0.24.4,
//! pulp runtime SIMD.

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

fn mean_secs(reps: usize, mut body: impl FnMut()) -> f64 {
    let start = Instant::now();
    for _ in 0..reps {
        body();
    }
    start.elapsed().as_secs_f64() / f64::from(u32::try_from(reps).expect("reps"))
}

fn reps_for(n: usize) -> usize {
    if n <= 30 {
        20
    } else if n <= 300 {
        3
    } else {
        1
    }
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
        let reps = reps_for(n);
        let which = env::var("EIGH_WHICH").unwrap_or_else(|_| "both".into());
        let run_faer = which == "both" || which == "faer";
        let run_jacobi = which == "both" || which == "jacobi";
        let warmup = n < 1500;
        let mut faer_lowest = f64::NAN;
        let mut faer_secs = f64::NAN;
        if run_faer {
            if warmup {
                let _ = black_box(exact_eigh(black_box(a.view())).expect("faer warmup"));
            }
            println!("timing faer n={n} reps={reps}");
            faer_secs = mean_secs(reps, || {
                let (evals, _) = exact_eigh(black_box(a.view())).expect("faer");
                faer_lowest = evals[0];
                black_box(faer_lowest);
            });
        }
        let mut jacobi_lowest = f64::NAN;
        let mut jacobi_secs = f64::NAN;
        if run_jacobi {
            if warmup {
                let _ = black_box(jacobi_eigh(black_box(a.view())).expect("jacobi warmup"));
            }
            println!("timing jacobi n={n} reps={reps}");
            jacobi_secs = mean_secs(reps, || {
                let (evals, _) = jacobi_eigh(black_box(a.view())).expect("jacobi");
                jacobi_lowest = evals[0];
                black_box(jacobi_lowest);
            });
        }
        let gap = if run_faer && run_jacobi {
            (faer_lowest - jacobi_lowest).abs()
        } else {
            f64::NAN
        };
        println!(
            "n={n} reps={reps} faer_s={faer_secs:.6} jacobi_s={jacobi_secs:.6} lowest_gap={gap:.3e}"
        );
    }
}
