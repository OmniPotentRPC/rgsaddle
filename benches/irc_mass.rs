// Copyright (c) rgsaddle developers

use criterion::{Criterion, criterion_group, criterion_main};
use ndarray::{Array1, ArrayView1};
use rgsaddle::{IrcConfig, IrcDirection, IrcSession, PointSurface, SaddleError};

/// `V = 1/4 (x0^2 - 1)^2 + 1/2 sum_{i>0} xi^2`, a saddle at the origin.
struct SaddleWell;

impl PointSurface for SaddleWell {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let x0 = x[0];
        let well = 0.25 * (x0 * x0 - 1.0).powi(2);
        let mut transverse = 0.0;
        let mut gradient = Array1::zeros(x.len());
        gradient[0] = x0 * x0 * x0 - x0;
        for index in 1..x.len() {
            transverse += 0.5 * x[index] * x[index];
            gradient[index] = x[index];
        }
        Ok((well + transverse, gradient))
    }
}

fn eight_steps(masses: [f64; 2]) -> Array1<f64> {
    let mut mode = Array1::zeros(6);
    mode[0] = 1.0;
    let mut session = IrcSession::new(
        IrcConfig::default()
            .with_dx(0.07)
            .with_force_tol(1.0e-12)
            .with_max_inner(8),
        Array1::zeros(6),
        Array1::from(masses.to_vec()),
        mode,
        IrcDirection::Forward,
    )
    .expect("IRC session");
    for _ in 0..8 {
        session.step(&SaddleWell).expect("IRC step");
    }
    session.position().to_owned()
}

fn irc_steps(c: &mut Criterion) {
    let mut group = c.benchmark_group("irc-eight-steps");
    group.bench_function("unequal-masses", |b| {
        b.iter(|| std::hint::black_box(eight_steps([1.008, 15.999])));
    });
    group.bench_function("unit-masses", |b| {
        b.iter(|| std::hint::black_box(eight_steps([1.0, 1.0])));
    });
    group.finish();
}

criterion_group!(benches, irc_steps);
criterion_main!(benches);
