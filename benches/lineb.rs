//! Force calls to converge one fixed-energy LI-NEB on the symmetric
//! Eckart barrier.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use ndarray::{Array1, Array2, ArrayView1};
use rgsaddle::{LinebConfig, LinebSession, LinebStatus, Masses, PointSurface, SaddleError};

struct CountedEckart {
    calls: AtomicUsize,
}

impl PointSurface for CountedEckart {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let s = 1.0 / x[0].cosh();
        Ok((s * s, Array1::from_elem(1, -2.0 * s * s * x[0].tanh())))
    }
}

fn converged_calls() -> usize {
    let turning = (1.0 / 0.25_f64).sqrt().acosh();
    let images = Array2::from_shape_fn((9, 1), |(i, _)| {
        let end = 0.5 * turning;
        -end + 2.0 * end * i as f64 / 8.0
    });
    let surface = CountedEckart {
        calls: AtomicUsize::new(0),
    };
    let spacing = turning * 2.0 / 8.0;
    let draft = LinebConfig::new(spacing, spacing);
    let panels = (2 * draft.quadrature_pairs) as f64;
    // Endpoint force `endpoint_scale * |V−E| / |∇V|`, inside one Simpson
    // panel of the converged chord.
    let tol = draft.endpoint_scale * spacing / panels.powi(2);
    let config = draft.with_opt_tol(tol).with_perp_tol(tol);
    let mut session = LinebSession::new(
        config,
        Masses::coordinates(Array1::ones(1).view()).unwrap(),
        images,
        0.25,
    )
    .unwrap();
    let report = session.run(&surface, 500).unwrap();
    assert_eq!(report.status, LinebStatus::Converged);
    surface.calls.load(Ordering::Relaxed)
}

fn eckart_force_calls(c: &mut Criterion) {
    let calls = converged_calls();
    eprintln!("one converged Eckart LI-NEB: {calls} force calls at one energy");
    let mut group = c.benchmark_group("lineb_eckart");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));
    group.throughput(Throughput::Elements(calls as u64));
    group.bench_function("converged_path", |b| b.iter(converged_calls));
    group.finish();
}

criterion_group!(benches, eckart_force_calls);
criterion_main!(benches);
