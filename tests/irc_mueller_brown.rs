//! Two-dimensional Mueller-Brown surface (Mueller and Brown, Theor. Chim.
//! Acta 53, 75 (1979)) with unit weights, from the saddle near
//! (-0.822, 0.624) between the minima near (-0.558, 1.442) and
//! (-0.050, 0.467).

use ndarray::{Array1, Array2, ArrayView1, array};
use rgsaddle::{IrcConfig, IrcDirection, IrcSession, PointSurface, SaddleError};

const A: [f64; 4] = [-200.0, -100.0, -170.0, 15.0];
const AC: [f64; 4] = [-1.0, -1.0, -6.5, 0.7];
const B: [f64; 4] = [0.0, 0.0, 11.0, 0.6];
const C: [f64; 4] = [-10.0, -10.0, -6.5, 0.7];
const X0: [f64; 4] = [1.0, 0.0, -0.5, -1.0];
const Y0: [f64; 4] = [0.0, 0.5, 1.5, 1.0];

struct MuellerBrown;

impl MuellerBrown {
    fn hessian(x: ArrayView1<f64>) -> Array2<f64> {
        let mut h = Array2::zeros((2, 2));
        for i in 0..4 {
            let (dx, dy) = (x[0] - X0[i], x[1] - Y0[i]);
            let e = A[i] * (AC[i] * dx * dx + B[i] * dx * dy + C[i] * dy * dy).exp();
            let (ux, uy) = (2.0 * AC[i] * dx + B[i] * dy, B[i] * dx + 2.0 * C[i] * dy);
            h[(0, 0)] += e * (ux * ux + 2.0 * AC[i]);
            h[(0, 1)] += e * (ux * uy + B[i]);
            h[(1, 1)] += e * (uy * uy + 2.0 * C[i]);
        }
        h[(1, 0)] = h[(0, 1)];
        h
    }
}

impl PointSurface for MuellerBrown {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (mut energy, mut gx, mut gy) = (0.0, 0.0, 0.0);
        for i in 0..4 {
            let (dx, dy) = (x[0] - X0[i], x[1] - Y0[i]);
            let e = A[i] * (AC[i] * dx * dx + B[i] * dx * dy + C[i] * dy * dy).exp();
            energy += e;
            gx += e * (2.0 * AC[i] * dx + B[i] * dy);
            gy += e * (B[i] * dx + 2.0 * C[i] * dy);
        }
        Ok((energy, array![gx, gy]))
    }
}

/// Newton on the gradient until a step no longer shrinks, then the
/// eigenvector of the negative Hessian eigenvalue there.
fn saddle_and_mode() -> (Array1<f64>, Array1<f64>) {
    let mut x = array![-0.82, 0.62];
    let mut last = f64::INFINITY;
    for _ in 0..50 {
        let (_, g) = MuellerBrown.eval(x.view()).unwrap();
        let h = MuellerBrown::hessian(x.view());
        let det = h[(0, 0)] * h[(1, 1)] - h[(0, 1)] * h[(1, 0)];
        let step = array![
            (h[(1, 1)] * g[0] - h[(0, 1)] * g[1]) / det,
            (h[(0, 0)] * g[1] - h[(1, 0)] * g[0]) / det
        ];
        let size = step.dot(&step).sqrt();
        x -= &step;
        if size >= last {
            break;
        }
        last = size;
    }
    let h = MuellerBrown::hessian(x.view());
    let (a, b, d) = (h[(0, 0)], h[(0, 1)], h[(1, 1)]);
    let lowest = 0.5 * (a + d) - (0.25 * (a - d).powi(2) + b * b).sqrt();
    assert!(lowest < 0.0, "Newton left the saddle for {x:?}");
    let mode = array![b, lowest - a];
    let norm = mode.dot(&mode).sqrt();
    (x, mode / norm)
}

/// At radius 0.00625 the reverse branch meets quasi-Newton models with one
/// eigenvalue zero to round-off. A level shift short of the sphere there
/// gives a trial that raises the energy and halves the radius. At the
/// minimum of the model on the sphere no trial in the first 0.4 of arc is
/// rejected, so every step adds the full radius to the arc.
#[test]
fn reverse_branch_at_a_small_radius_never_halves_it() {
    let dx = 0.00625;
    let (saddle, mode) = saddle_and_mode();
    let mut session = IrcSession::new_euclidean(
        IrcConfig {
            dx,
            ..IrcConfig::default()
        },
        saddle,
        mode,
        IrcDirection::Reverse,
    )
    .unwrap();
    let mut arc = 0.0;
    for step in 1..=64 {
        let report = session.step(&MuellerBrown).unwrap();
        arc += dx;
        assert_eq!(report.arc, arc, "radius halved on step {step}");
    }
}
