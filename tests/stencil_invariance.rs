//! Serial stencil kernels, hashed at several thread counts.
//!
//! The kernels are fixed-order dots. Spawning more threads repeats the
//! same serial evaluation and joins the hashes in spawn order. The hash
//! does not depend on the thread count. Inputs include -0, +0, a
//! subnormal, a quiet NaN, and an exact energy tie.

use ndarray::array;
use rgsaddle::projection::climbing_image_force;
use rgsaddle::tangent::{TangentKind, compute_tangent};
use sha2::{Digest, Sha256};
use std::time::{Duration, Instant};

fn specials() -> Vec<f64> {
    vec![
        -0.0,
        0.0,
        f64::from_bits(1),
        f64::from_bits(0x7ff8_0000_0000_0001),
        1.0,
        -1.0,
        0.5,
        2.0,
        -2.0,
    ]
}

fn push_bits(hasher: &mut Sha256, values: &[f64]) {
    for value in values {
        hasher.update(value.to_bits().to_le_bytes());
    }
}

fn one_run() -> [u8; 32] {
    let mut hasher = Sha256::new();
    let values = specials();
    for tangent_value in &values {
        for force_value in &values {
            let force = array![*force_value, 1.0];
            let tangent = array![*tangent_value, 0.0];
            let dneb = array![-0.0, 0.0];
            let got = climbing_image_force(force.view(), tangent.view(), dneb.view());
            push_bits(&mut hasher, got.as_slice().unwrap());
        }
    }
    // Exact tie: equal side energies, so both weighted formulas agree.
    let next = array![3.0, 4.0];
    let prev = array![3.0, 4.0];
    let tied = compute_tangent(
        TangentKind::Improved,
        next.view(),
        prev.view(),
        2.0,
        0.0,
        0.0,
    )
    .expect("tied tangent");
    push_bits(&mut hasher, tied.as_slice().unwrap());
    let uphill = compute_tangent(
        TangentKind::Improved,
        array![1.0, 0.0].view(),
        array![0.0, 1.0].view(),
        1.0,
        0.0,
        2.0,
    )
    .expect("uphill tangent");
    push_bits(&mut hasher, uphill.as_slice().unwrap());
    hasher.finalize().into()
}

fn hash_at(threads: usize) -> [u8; 32] {
    let handles: Vec<_> = (0..threads).map(|_| std::thread::spawn(one_run)).collect();
    let mut hasher = Sha256::new();
    for handle in handles {
        let piece = handle.join().expect("worker");
        hasher.update(piece);
    }
    hasher.finalize().into()
}

#[test]
fn stencil_hash_is_independent_of_thread_count() {
    let one = one_run();
    for threads in [1usize, 2, 3, 8] {
        let handles: Vec<_> = (0..threads).map(|_| std::thread::spawn(one_run)).collect();
        for handle in handles {
            assert_eq!(handle.join().expect("worker"), one);
        }
    }
    // Joining the per-thread hashes in spawn order is stable for a
    // fixed thread count, and the underlying digest does not change.
    assert_eq!(hash_at(1), hash_at(1));
    let joined_two = hash_at(2);
    assert_eq!(joined_two, hash_at(2));
    assert_ne!(hash_at(1), joined_two);
}

/// Fixed batch of 400 serial evaluations, split into contiguous
/// index-order chunks. The chunk count is the thread count. Each
/// evaluation is the same `one_run`, so the bits do not depend on
/// the split. Wall time is the strong-scaling measurement.
fn time_partitioned(threads: usize) -> Duration {
    let total = 400;
    let start = Instant::now();
    let mut handles = Vec::with_capacity(threads);
    for worker in 0..threads {
        let begin = worker * total / threads;
        let end = (worker + 1) * total / threads;
        handles.push(std::thread::spawn(move || {
            for _ in begin..end {
                let _ = one_run();
            }
        }));
    }
    for handle in handles {
        handle.join().expect("worker");
    }
    start.elapsed()
}

#[test]
fn strong_scaling_of_the_serial_kernel() {
    if std::env::var_os("RGSADDLE_SCALE").is_none() {
        return;
    }
    let threads: Vec<usize> = vec![
        1,
        2,
        4,
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
    ];
    let rounds = 5;
    let mut samples: Vec<Vec<Duration>> = threads.iter().map(|_| Vec::new()).collect();
    for _ in 0..rounds {
        for (index, _) in threads.iter().enumerate() {
            samples[index].push(time_partitioned(threads[index]));
        }
    }
    let t1 = median_ms(&samples[0]);
    println!("threads median_ms min_ms max_ms efficiency");
    for (threads, sample) in threads.iter().zip(&samples) {
        let med = median_ms(sample);
        let min = sample
            .iter()
            .map(|d| d.as_secs_f64() * 1e3)
            .fold(f64::INFINITY, f64::min);
        let max = sample
            .iter()
            .map(|d| d.as_secs_f64() * 1e3)
            .fold(0.0, f64::max);
        let efficiency = t1 / (med * *threads as f64);
        println!("{threads} {med:.3} {min:.3} {max:.3} {efficiency:.3}");
    }
}

fn median_ms(sample: &[Duration]) -> f64 {
    let mut values: Vec<f64> = sample.iter().map(|d| d.as_secs_f64() * 1e3).collect();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    values[values.len() / 2]
}
