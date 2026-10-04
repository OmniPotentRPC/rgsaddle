//! The RAS controller uses maximum atomic displacement.

use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, ArrayView1};
use rgmin::ManifoldKind;
use rgsaddle::{
    PointSurface, RestrictedKind, SaddleError, SellaGeom, SellaSaddleConfig, SellaSaddleSession,
};

struct StiffQuadratic {
    evaluations: AtomicUsize,
}

impl PointSurface for StiffQuadratic {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        self.evaluations.fetch_add(1, Ordering::Relaxed);
        let mut energy = -0.5 * x[0] * x[0];
        let mut gradient = Array1::zeros(x.len());
        gradient[0] = -x[0];
        for i in [1, 4, 7] {
            energy += x[i] + 50.0 * x[i] * x[i];
            gradient[i] = 1.0 + 100.0 * x[i];
        }
        Ok((energy, gradient))
    }
}

#[test]
fn rejected_saddle_ras_model_shrinks_the_atomic_radius() {
    let surface = StiffQuadratic {
        evaluations: AtomicUsize::new(0),
    };
    let config = SellaSaddleConfig {
        delta: 0.1/9.0,
        eig: false,
        force_tol: 1e-12,
        restricted: RestrictedKind::RestrictedAtomicStep,
        ..SellaSaddleConfig::default()
    };
    let contraction = config.sigma_dec;
    let mut session = SellaSaddleSession::on(
        config,
        Array1::zeros(9),
        Array1::from(vec![1.0; 3]),
        SellaGeom::Kind(ManifoldKind::Euclidean),
    )
    .unwrap();
    let mut mode = Array1::zeros(9);
    mode[0] = 1.0;
    session.seed_mode(mode.view(), -1.0).unwrap();
    let initial_radius = session.delta();
    let report = session.step(&surface).unwrap();

    assert_eq!(surface.evaluations.load(Ordering::Relaxed), 2);
    for i in [1, 4, 7] {
        assert!((session.position()[i] + initial_radius).abs() < 1e-12);
    }
    assert!(report.energy > 0.0);
    assert!(report.rho < 0.0);
    assert!(!report.at_saddle);
    assert!(report.delta < initial_radius);
    assert!(
        (report.delta - contraction * initial_radius).abs() < 1e-12,
        "radius {} differs from atomic contraction {}",
        report.delta,
        contraction * initial_radius
    );
}
