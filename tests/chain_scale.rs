//! Strong scaling of [`rgsaddle::Chain::evaluate`].
//!
//! The default suite skips this test. Record the table with
//! `cargo test --release --test chain_scale -- --ignored --nocapture`.

use std::hint::black_box;
use std::time::Instant;

use ndarray::{Array2, ArrayView2, ArrayViewMut2};
use rgsaddle::stencil::{ImageId, Surface};
use rgsaddle::{Chain, EvalPool};

struct Work {
    rounds: usize,
}

impl Surface for Work {
    fn eval(
        &self,
        ids: &[ImageId],
        x: ArrayView2<f64>,
        e: &mut [f64],
        mut g: ArrayViewMut2<f64>,
    ) -> Result<(), rgsaddle::SaddleError> {
        for (row, _id) in ids.iter().enumerate() {
            let mut acc = x[(row, 0)];
            for _ in 0..self.rounds {
                acc = black_box(acc.mul_add(0.999_999_999, 1e-18));
            }
            e[row] = acc;
            g[(row, 0)] = acc;
        }
        Ok(())
    }
}

fn once(rows: usize, threads: usize, rounds: usize) -> std::time::Duration {
    let x = Array2::from_shape_fn((rows, 8), |(i, k)| (i + k) as f64 * 1e-3);
    let mut chain = Chain::new(x).expect("chain");
    let pool = EvalPool::new(threads).expect("pool");
    let surface = Work { rounds };
    let start = Instant::now();
    chain.evaluate(&surface, &pool).expect("evaluate");
    let elapsed = start.elapsed();
    black_box(chain.energies()[0]);
    elapsed
}

fn median(samples: &mut [u128]) -> u128 {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

#[test]
#[ignore = "timing table, run in release"]
fn chain_evaluate_strong_scaling() {
    let rows = 128;
    let rounds = 20_000;
    let max_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let mut counts = vec![1, 2, 4];
    if !counts.contains(&max_threads) {
        counts.push(max_threads);
    }
    let repeats = 5;
    let mut table: Vec<(usize, Vec<u128>)> = counts.iter().map(|n| (*n, Vec::new())).collect();
    for _ in 0..repeats {
        for (threads, samples) in &mut table {
            samples.push(once(rows, *threads, rounds).as_nanos());
        }
    }
    let serial = median(&mut table[0].1.clone());
    println!("rows {rows} rounds {rounds} repeats {repeats}");
    println!("threads median_ns min_ns max_ns efficiency");
    for (threads, samples) in &mut table {
        let low = *samples.iter().min().unwrap_or(&0);
        let high = *samples.iter().max().unwrap_or(&0);
        let mid = median(samples);
        let efficiency = if *threads == 0 || mid == 0 {
            0.0
        } else {
            (serial as f64) / (mid as f64) / (*threads as f64)
        };
        println!("{threads} {mid} {low} {high} {efficiency:.3}");
    }
    let one = {
        let mut chain = Chain::new(Array2::from_elem((8, 1), 0.25)).unwrap();
        chain
            .evaluate(&Work { rounds: 10 }, &EvalPool::new(1).unwrap())
            .unwrap();
        chain.energies().to_vec()
    };
    let many = {
        let mut chain = Chain::new(Array2::from_elem((8, 1), 0.25)).unwrap();
        chain
            .evaluate(
                &Work { rounds: 10 },
                &EvalPool::new(max_threads.max(2)).unwrap(),
            )
            .unwrap();
        chain.energies().to_vec()
    };
    for (a, b) in one.iter().zip(many.iter()) {
        assert_eq!(a.to_le_bytes(), b.to_le_bytes());
    }
}
