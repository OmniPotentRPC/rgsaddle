use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, Array2, ArrayView1, array, s};
use rgsaddle::error::SaddleError;
use rgsaddle::minmode::PointSurface;
use rgsaddle::nichols::{HessianUpdate, Index1Config, Index1Session, Index1Status};
use rgsaddle::prfo_restricted::{PrfoKind, restricted_partitioned_rfo};

struct Quadratic {
    hessian: Array2<f64>,
    calls: AtomicUsize,
}

impl PointSurface for Quadratic {
    fn eval(&self, x: ArrayView1<'_, f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let gradient = self.hessian.dot(&x);
        Ok((0.5 * x.dot(&gradient), gradient))
    }
}

fn initial_hessian() -> Array2<f64> {
    // With masses [1, 4, 9], the weighted spectrum is [-2, 3, 5].
    array![[1.2, -4.8, 0.0], [-4.8, -0.8, 0.0], [0.0, 0.0, 45.0]]
}

#[test]
fn curvature_belongs_to_the_hessian_that_produced_the_step() {
    for update in [HessianUpdate::Powell, HessianUpdate::Bofill] {
        let h = initial_hessian();
        let surface = Quadratic {
            hessian: &h + &Array2::from_diag(&array![0.4, 0.8, 1.2]),
            calls: AtomicUsize::new(0),
        };
        let mut session = Index1Session::new(
            Index1Config {
                update,
                trust_radius: 0.1,
                force_tol: 0.0,
                ..Index1Config::default()
            },
            array![0.15, -0.25, 0.1],
            Some(h.clone()),
            Some(array![1.0, 4.0, 9.0]),
        )
        .unwrap();
        let report = session.step(&surface).unwrap();
        assert!((report.curvature + 2.0).abs() < 1e-14, "{report:?}");
        assert!(report.max_step <= 0.1);
        assert_eq!(surface.calls.load(Ordering::Relaxed), 2);
        let changed = (&session.hessian().unwrap() - &h).mapv(f64::abs).sum();
        assert!(
            changed > 1e-6,
            "the update must change the supplied Hessian"
        );
        session.step(&surface).unwrap();
        assert_eq!(surface.calls.load(Ordering::Relaxed), 3);
        session.reset();
        session.step(&surface).unwrap();
        assert_eq!(
            surface.calls.load(Ordering::Relaxed),
            11,
            "reset needs 2n finite differences"
        );
    }
}

#[test]
fn a_stationary_point_reports_curvature_without_a_displacement() {
    let h = initial_hessian();
    let surface = Quadratic {
        hessian: h.clone(),
        calls: AtomicUsize::new(0),
    };
    let mut session = Index1Session::new(
        Index1Config::default(),
        Array1::zeros(3),
        Some(h),
        Some(array![1.0, 4.0, 9.0]),
    )
    .unwrap();
    let report = session.step(&surface).unwrap();
    assert_eq!(report.status, Index1Status::Converged);
    assert_eq!(report.iteration, 0);
    assert_eq!(report.max_step, 0.0);
    assert!((report.curvature + 2.0).abs() < 1e-14);
    assert_eq!(surface.calls.load(Ordering::Relaxed), 1);
}

#[test]
fn restricted_steps_accept_strided_eigenvectors_and_masses() {
    let eigenvalues = array![-2.0, 3.0, 5.0];
    let gradient = array![0.2, -0.3, 0.4];
    let eigenvectors = array![[0.6, -0.8, 0.0], [0.8, 0.6, 0.0], [0.0, 0.0, 1.0]];
    let masses = array![1.0, 4.0, 9.0];
    let mut stored_vectors = Array2::from_elem((6, 6), f64::NAN);
    stored_vectors
        .slice_mut(s![..;2, ..;2])
        .assign(&eigenvectors);
    let mut stored_masses = Array1::from_elem(6, f64::NAN);
    stored_masses.slice_mut(s![..;2]).assign(&masses);
    for kind in [PrfoKind::Index1, PrfoKind::Minimize] {
        for radius in [0.001, 1.0] {
            let owned = restricted_partitioned_rfo(
                eigenvalues.view(),
                gradient.view(),
                eigenvectors.view(),
                masses.view(),
                radius,
                kind,
            )
            .unwrap();
            let strided = restricted_partitioned_rfo(
                eigenvalues.view(),
                gradient.view(),
                stored_vectors.slice(s![..;2, ..;2]),
                stored_masses.slice(s![..;2]),
                radius,
                kind,
            )
            .unwrap();
            assert_eq!(owned.mapv(f64::to_bits), strided.mapv(f64::to_bits));
            assert!(strided.dot(&strided).sqrt() <= radius);
        }
    }
}

#[test]
fn analytic_trust_step_distinguishes_weighted_and_updated_curvature() {
    for update in [HessianUpdate::Powell, HessianUpdate::Bofill] {
        let h = Array2::from_diag(&array![2.0, -8.0, 45.0]);
        let surface = Quadratic {
            hessian: &h * 2.0,
            calls: AtomicUsize::new(0),
        };
        let mut session = Index1Session::new(
            Index1Config {
                update,
                trust_radius: 0.125,
                force_tol: 0.0,
                ..Index1Config::default()
            },
            array![0.0, 0.25, 0.0],
            Some(h),
            Some(array![1.0, 4.0, 9.0]),
        )
        .unwrap();
        let report = session.step(&surface).unwrap();
        assert_eq!(report.curvature, -2.0);
        assert!((session.position()[1] - 0.125).abs() < 1e-14);
        assert_eq!(session.position()[0], 0.0);
        assert_eq!(session.position()[2], 0.0);
        assert!((session.hessian().unwrap()[(1, 1)] + 16.0).abs() < 1e-12);
        assert_eq!(surface.calls.load(Ordering::Relaxed), 2);
        let second = session.step(&surface).unwrap();
        assert!((second.curvature + 4.0).abs() < 1e-12);
        assert_eq!(surface.calls.load(Ordering::Relaxed), 3);
    }
}
