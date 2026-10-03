use ndarray::{Array1, Array2, ArrayView1, array};
use rgsaddle::{ForceGate, MinModeConfig, MinModeKind, MinModeSession, MinModeStatus, PointSurface, SaddleError};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Quadratic {
    diagonal: Array1<f64>,
    excluded: Array2<f64>,
    analytic: bool,
    points: Mutex<Vec<Array1<f64>>>,
    actions: AtomicUsize,
}

impl PointSurface for Quadratic {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        self.points.lock().unwrap().push(x.to_owned());
        let g = &self.diagonal * &x;
        Ok((0.5 * g.dot(&x), g))
    }

    fn hessian_vector(&self, _x: ArrayView1<f64>, v: ArrayView1<f64>) -> Result<Option<Array1<f64>>, SaddleError> {
        if self.analytic {
            self.actions.fetch_add(1, Ordering::Relaxed);
            Ok(Some(&self.diagonal * &v))
        } else {
            Ok(None)
        }
    }

    fn excluded_modes(&self, _x: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
        Ok(self.excluded.clone())
    }
}

#[test]
fn excluded_direction_is_absent_from_measured_actions_and_the_lowest_mode() {
    for kind in [MinModeKind::Dimer, MinModeKind::Lanczos] {
        for analytic in [false, true] {
            let surface = Quadratic {
                diagonal: array![-10.0, -2.0, 3.0],
                excluded: array![[1.0, 0.0, 0.0]],
                analytic,
                points: Mutex::new(Vec::new()),
                actions: AtomicUsize::new(0),
            };
            let config = MinModeConfig { kind, rotation_tol: 1e-10, ..Default::default() };
            let mut session = MinModeSession::new(config, Array1::zeros(3), array![1.0, 0.0, 0.0]).unwrap();
            let estimate = session.estimate_mode(&surface).unwrap();
            assert!((estimate.curvature + 2.0).abs() < 1e-9);
            assert!(estimate.mode[0].abs() < 1e-12);
            assert!((estimate.mode[1].abs() - 1.0).abs() < 1e-9);
            let points = surface.points.lock().unwrap();
            assert_eq!(estimate.evaluations, points.len());
            assert!(points.iter().all(|x| x[0] == 0.0));
            if analytic {
                assert_eq!(points.len(), 1);
                assert!(surface.actions.load(Ordering::Relaxed) > 0);
            } else {
                assert!(points.len() > 1);
                assert_eq!(surface.actions.load(Ordering::Relaxed), 0);
            }
        }
    }
}

#[test]
fn force_gate_setter_preserves_default_and_changes_physical_translation() {
    let surface = Quadratic {
        diagonal: array![-1.0, 1.0, 2.0],
        excluded: Array2::zeros((0, 3)),
        analytic: true,
        points: Mutex::new(Vec::new()),
        actions: AtomicUsize::new(0),
    };
    let config = MinModeConfig {
        method: rgmin::Method::Lbfgs { memory: 4 },
        force_tol: 1e-3,
        max_move: 1e-5,
        ..Default::default()
    };
    let start = array![-0.0008, 0.0008, 0.0004];
    let mode = array![1.0, 0.0, 0.0];
    let mut component = MinModeSession::new(config.clone(), start.clone(), mode.clone()).unwrap();
    let default_report = component.step(&surface).unwrap();
    assert_eq!(default_report.status, MinModeStatus::Converged);
    assert_eq!(default_report.iteration, 0);
    assert_eq!(component.position(), start.view());
    let mut atom = MinModeSession::new(config, start.clone(), mode).unwrap();
    atom.set_force_gate(ForceGate::MaxForceOnAtom);
    let atom_report = atom.step(&surface).unwrap();
    assert_eq!(atom_report.status, MinModeStatus::Running);
    assert_eq!(atom_report.iteration, 1);
    assert_ne!(atom.position(), start.view());
    let g = &surface.diagonal * &atom.position();
    assert!(g.dot(&g).sqrt() > 1e-3);
}
