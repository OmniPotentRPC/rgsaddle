//! Statistical convergence on coupled, non-separable surfaces.
#![cfg(feature = "ohtst")]

use ndarray::{Array1, ArrayView1, array};
use rgsaddle::{
    Hyperplane, PlaneOptimizer, PlaneOptimizerConfig, PlaneOptimizerStatus, PointSurface,
    SaddleError,
};

#[derive(Clone, Copy)]
struct CoupledWell {
    linear: f64,
    quadratic: f64,
}

impl CoupledWell {
    fn stiffness(self, x: f64) -> f64 {
        1.0 + 2.0 * self.linear * x + 2.0 * self.quadratic * x * x
    }

    fn mean_gradient(self, x: f64, thermal: f64) -> f64 {
        x * (x * x - 1.0) + (self.linear + 2.0 * self.quadratic * x) * thermal / self.stiffness(x)
    }

    fn optimal_plane(self, thermal: f64) -> Hyperplane {
        let mut x = self.linear * thermal;
        for _ in 0..20 {
            let h = 1.0e-6;
            let value = self.mean_gradient(x, thermal);
            let derivative = (self.mean_gradient(x + h, thermal)
                - self.mean_gradient(x - h, thermal))
                / (2.0 * h);
            x -= value / derivative;
        }
        assert!(self.mean_gradient(x, thermal).abs() < 1.0e-12);
        Hyperplane::new(array![x, 0.0], array![1.0, 0.0]).unwrap()
    }
}

impl PointSurface for CoupledWell {
    fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (x, y) = (q[0], q[1]);
        let well = x * x - 1.0;
        let energy = 0.25 * well * well + 0.5 * self.stiffness(x) * y * y;
        let gradient = array![
            x * well + (self.linear + 2.0 * self.quadratic * x) * y * y,
            self.stiffness(x) * y,
        ];
        Ok((energy, gradient))
    }
}

fn converges_from_optimal_plane(surface: CoupledWell) {
    let config = PlaneOptimizerConfig::default();
    let plane = surface.optimal_plane(config.sampler.thermal);
    let optimum = PlaneOptimizer::new(&plane, config)
        .unwrap()
        .run(&surface)
        .unwrap();
    assert_eq!(optimum.status, PlaneOptimizerStatus::Converged);
    let last = optimum.trace.last().unwrap();
    assert!(last.normal_force_error > 0.0);
    assert!(last.rotation_error > 0.0);
}

#[test]
fn weak_xy2_coupling_converges_under_defaults() {
    converges_from_optimal_plane(CoupledWell {
        linear: 0.1,
        quadratic: 0.0,
    });
}

#[test]
fn mixed_xy2_and_x2y2_coupling_converges_under_defaults() {
    converges_from_optimal_plane(CoupledWell {
        linear: 0.12,
        quadratic: 0.08,
    });
}
