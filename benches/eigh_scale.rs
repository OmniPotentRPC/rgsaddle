//! Strong scaling of the shipped dense eigensolver.
//!
//! ```text
//! cargo bench --bench eigh_scale
//! ```
//!
//! `exact_eigh` passes `faer::Par::Seq`. Thread counts are rayon pools
//! around that call. Runs are interleaved. Each cell is the median of
//! seven solves, with the range (max - min). Efficiency is
//! `median(1 thread) / (threads * median(threads))`.
//! rgsaddle 0.2.2, faer 0.24.4.

use std::hint::black_box;
use std::thread::available_parallelism;
use std::time::Instant;

use ndarray::Array2;
use rayon::ThreadPoolBuilder;
use rgsaddle::exact_eigh;

const REPS: usize = 7;
const SIZES: [usize; 2] = [1500, 3000];

fn next_open(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(0x5851_F42D_4C95_7F2D)
        .wrapping_add(0x1405_7B7E_F767_814F);
    let bits = (*state >> 11) as u32;
    (f64::from(bits) / f64::from(u32::MAX)).mul_add(2.0, -1.0)
}

fn random_spd(n: usize) -> Array2<f64> {
    let mut state = 0x5CA1_E176;
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

fn median_and_range(samples: &mut [f64]) -> (f64, f64) {
    samples.sort_by(f64::total_cmp);
    let median = samples[samples.len() / 2];
    let range = samples[samples.len() - 1] - samples[0];
    (median, range)
}

fn main() {
    let max_threads = available_parallelism().expect("thread count").get();
    let mut threads = vec![1_usize, 2, 4, max_threads];
    threads.sort_unstable();
    threads.dedup();
    println!("rgsaddle {}", env!("CARGO_PKG_VERSION"));
    println!("faer 0.24.4");
    println!("solver exact_eigh pinned to faer::Par::Seq");
    println!("reps={REPS} interleaved median and range");
    println!("threads={threads:?} max={max_threads}");
    for n in SIZES {
        println!("building n={n}");
        let a = random_spd(n);
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
                black_box(exact_eigh(black_box(a.view())).expect("warmup"));
            });
            println!("warmup n={n} threads={count}");
        }
        let mut samples = vec![Vec::with_capacity(REPS); threads.len()];
        for _ in 0..REPS {
            for (slot, pool) in pools.iter().enumerate() {
                let seconds = pool.install(|| {
                    let start = Instant::now();
                    black_box(exact_eigh(black_box(a.view())).expect("faer"));
                    start.elapsed().as_secs_f64()
                });
                samples[slot].push(seconds);
            }
        }
        let (baseline, _) = median_and_range(&mut samples[0].clone());
        println!("n={n} threads median_s range_s efficiency");
        for (slot, &count) in threads.iter().enumerate() {
            let (median, range) = median_and_range(&mut samples[slot]);
            let efficiency = baseline / (count as f64 * median);
            println!(
                "n={n} threads={count} median_s={median:.6} range_s={range:.6} efficiency={efficiency:.3}"
            );
        }
    }
    println!(
        "limiter: exact_eigh is pinned to faer::Par::Seq, so extra threads do not enter the factorization"
    );
}
