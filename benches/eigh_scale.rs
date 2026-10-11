//! Strong scaling of the two-stage dense eigensolver.
//!
//! ```text
//! cargo bench --features par,faer-unpinned --bench eigh_scale -- 1500 3000
//! ```
//!
//! Optional arguments are matrix sizes. `EIGH_REPS` overrides the
//! repetition count (default 7). Runs are interleaved. Each cell is
//! the median, with the range (max - min). Efficiency is
//! `median(1 thread) / (threads * median(threads))`.
//! The tile count does not read the pool size.

use std::env;
use std::hint::black_box;
use std::thread::available_parallelism;
use std::time::Instant;

use ndarray::Array2;
use rayon::ThreadPoolBuilder;
use rgsaddle::dense_eigh::{StageSeconds, faer_sequential, faer_with_threads};
use rgsaddle::eigensolve::{eigh_timed, partial_eigh_timed};

const DEFAULT_REPS: usize = 7;

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

/// Indefinite spectrum with a triple eigenvalue at the bottom.
/// `Q = I - 2 u u^T` and `A = Q diag(d) Q`.
fn clustered_indefinite(n: usize, seed: u64) -> Array2<f64> {
    let mut state = seed;
    let mut u = vec![0.0; n];
    let mut norm = 0.0;
    for value in &mut u {
        *value = next_open(&mut state);
        norm += *value * *value;
    }
    norm = norm.sqrt();
    for value in &mut u {
        *value /= norm;
    }
    let mut diag = vec![0.0; n];
    diag[0] = -1.5;
    diag[1] = -1.5;
    diag[2] = -1.5;
    for (i, value) in diag.iter_mut().enumerate().skip(3) {
        *value = 0.05 * i as f64;
    }
    let mut du = vec![0.0; n];
    let mut udu = 0.0;
    for i in 0..n {
        du[i] = diag[i] * u[i];
        udu += u[i] * du[i];
    }
    let mut a = Array2::zeros((n, n));
    for i in 0..n {
        a[(i, i)] = diag[i];
        for j in 0..n {
            a[(i, j)] += -2.0 * du[i] * u[j] - 2.0 * u[i] * du[j] + 4.0 * udu * u[i] * u[j];
        }
    }
    a
}

fn median_and_range(samples: &mut [f64]) -> (f64, f64) {
    samples.sort_by(f64::total_cmp);
    let median = samples[samples.len() / 2];
    let range = samples[samples.len() - 1] - samples[0];
    (median, range)
}

fn median_stages(samples: &mut [StageSeconds]) -> StageSeconds {
    let mut band: Vec<f64> = samples.iter().map(|stage| stage.band).collect();
    let mut bulge: Vec<f64> = samples.iter().map(|stage| stage.bulge).collect();
    let mut tridiagonal: Vec<f64> = samples.iter().map(|stage| stage.tridiagonal).collect();
    let mut back: Vec<f64> = samples.iter().map(|stage| stage.back_transform).collect();
    StageSeconds::from_parts(
        median_and_range(&mut band).0,
        median_and_range(&mut bulge).0,
        median_and_range(&mut tridiagonal).0,
        median_and_range(&mut back).0,
    )
}

fn print_stages(n: usize, threads: usize, wall: f64, stage: StageSeconds) {
    let sum = stage.band + stage.bulge + stage.tridiagonal + stage.back_transform;
    let share = |part: f64| if sum > 0.0 { part / sum } else { 0.0 };
    println!(
        "n={n} threads={threads} stage_band_s={:.6} stage_bulge_s={:.6} stage_tridiagonal_s={:.6} stage_back_s={:.6} share_band={:.3} share_bulge={:.3} share_tridiagonal={:.3} share_back={:.3} wall_s={wall:.6}",
        stage.band,
        stage.bulge,
        stage.tridiagonal,
        stage.back_transform,
        share(stage.band),
        share(stage.bulge),
        share(stage.tridiagonal),
        share(stage.back_transform),
    );
}

fn time_full(a: &Array2<f64>) -> (f64, StageSeconds) {
    let start = Instant::now();
    let (_, _, stage) = eigh_timed(black_box(a.view())).expect("eigh");
    (start.elapsed().as_secs_f64(), stage)
}

fn scale_matrix(label: &str, a: &Array2<f64>, threads: &[usize], reps: usize) {
    let n = a.nrows();
    println!("matrix={label} n={n}");
    let pools: Vec<_> = threads
        .iter()
        .map(|&count| {
            ThreadPoolBuilder::new()
                .num_threads(count)
                .build()
                .expect("thread pool")
        })
        .collect();
    for (count, pool) in threads.iter().zip(&pools) {
        pool.install(|| {
            black_box(time_full(a));
        });
        println!("warmup matrix={label} n={n} threads={count}");
    }
    let mut walls = vec![Vec::with_capacity(reps); threads.len()];
    let mut stages = vec![Vec::with_capacity(reps); threads.len()];
    for _ in 0..reps {
        for (slot, pool) in pools.iter().enumerate() {
            let (wall, stage) = pool.install(|| time_full(a));
            walls[slot].push(wall);
            stages[slot].push(stage);
        }
    }
    let (baseline, _) = median_and_range(&mut walls[0].clone());
    println!("matrix={label} n={n} threads median_s range_s efficiency");
    for (slot, &count) in threads.iter().enumerate() {
        let (median, range) = median_and_range(&mut walls[slot]);
        let efficiency = baseline / (count as f64 * median);
        println!(
            "matrix={label} n={n} threads={count} median_s={median:.6} range_s={range:.6} efficiency={efficiency:.3}"
        );
        print_stages(n, count, median, median_stages(&mut stages[slot]));
    }
}

fn partial_against_full(a: &Array2<f64>, threads: usize, reps: usize) {
    let n = a.nrows();
    let pool = ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("thread pool");
    pool.install(|| {
        black_box(time_full(a));
    });
    let mut full = Vec::with_capacity(reps);
    for _ in 0..reps {
        let wall = pool.install(|| time_full(a).0);
        full.push(wall);
    }
    let (full_median, full_range) = median_and_range(&mut full);
    println!(
        "partial n={n} threads={threads} k=full median_s={full_median:.6} range_s={full_range:.6}"
    );
    for k in [1usize, 5, 10] {
        pool.install(|| {
            black_box(partial_eigh_timed(black_box(a.view()), k).expect("partial warmup"));
        });
        let mut samples = Vec::with_capacity(reps);
        for _ in 0..reps {
            let wall = pool.install(|| {
                let start = Instant::now();
                black_box(partial_eigh_timed(black_box(a.view()), k).expect("partial"));
                start.elapsed().as_secs_f64()
            });
            samples.push(wall);
        }
        let (median, range) = median_and_range(&mut samples);
        println!(
            "partial n={n} threads={threads} k={k} median_s={median:.6} range_s={range:.6} versus_full={:.3}",
            median / full_median
        );
    }
}

fn main() {
    let reps = env::var("EIGH_REPS")
        .ok()
        .and_then(|text| text.parse().ok())
        .unwrap_or(DEFAULT_REPS);
    let sizes: Vec<usize> = env::args()
        .skip(1)
        .filter(|arg| !arg.starts_with("--"))
        .map(|arg| {
            arg.parse::<usize>()
                .unwrap_or_else(|_| panic!("size must be an integer, got {arg}"))
        })
        .collect();
    let sizes = if sizes.is_empty() {
        vec![1500, 3000]
    } else {
        sizes
    };
    let max_threads = available_parallelism().expect("thread count").get();
    let mut threads = vec![1_usize, 2, 4, max_threads.min(8)];
    threads.sort_unstable();
    threads.dedup();
    println!("rgsaddle {}", env!("CARGO_PKG_VERSION"));
    println!("solver two-stage Householder band, bulge chase, Cuppen tridiagonal");
    println!("band width 64, WY width 64, FMA product, tasks fixed from n, reps={reps}");
    println!("threads={threads:?} max={max_threads}");

    let quick = env::var_os("EIGH_QUICK").is_some();
    for &n in &sizes {
        println!("building spd n={n}");
        let spd = random_spd(n, 0x5CA1_E176);
        scale_matrix("spd", &spd, &threads, reps);
        if quick {
            continue;
        }
        println!("building clustered n={n}");
        let clustered = clustered_indefinite(n, 0xC105_7E12);
        scale_matrix("clustered", &clustered, &threads, reps);
        partial_against_full(&spd, 1, reps);
        partial_against_full(&spd, threads[threads.len() - 1], reps);
        if threads.contains(&1) {
            println!("faer sequential reference n={n}");
            black_box(faer_sequential(&spd).expect("seq warmup"));
            let mut samples = Vec::with_capacity(reps);
            for _ in 0..reps {
                let start = Instant::now();
                black_box(faer_sequential(&spd).expect("seq"));
                samples.push(start.elapsed().as_secs_f64());
            }
            let (median, range) = median_and_range(&mut samples);
            println!("faer_seq n={n} threads=1 median_s={median:.6} range_s={range:.6}");
            black_box(faer_with_threads(&spd, 8).expect("rayon warmup"));
            let mut wide = Vec::with_capacity(reps);
            for _ in 0..reps {
                let start = Instant::now();
                black_box(faer_with_threads(&spd, 8).expect("rayon"));
                wide.push(start.elapsed().as_secs_f64());
            }
            let (median, range) = median_and_range(&mut wide);
            println!("faer_rayon n={n} threads=8 median_s={median:.6} range_s={range:.6}");
        }
    }
}
