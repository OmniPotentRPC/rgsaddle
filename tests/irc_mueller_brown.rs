//! Two-dimensional Mueller-Brown surface (Mueller and Brown, Theor. Chim.
//! Acta 53, 75 (1979)) with unit weights, from the saddle near
//! (-0.822, 0.624) between the minima near (-0.558, 1.442) and
//! (-0.050, 0.467).

use ndarray::{Array1, Array2, ArrayView1, array};
use rgsaddle::{IrcConfig, IrcDirection, IrcKind, IrcSession, PointSurface, SaddleError};

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
            kind: IrcKind::SellaSphere,
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

/// Arc from the saddle over which the path error is measured.
const ORDER_ARC: f64 = 0.4;
/// Coarsest radius; the fit takes `COARSEST / 2^i` for `i` in `1..=HALVINGS`.
const COARSEST: f64 = 0.1;
const HALVINGS: i32 = 4;
/// The chord of a reference step leaves the path by at most `k h^2 / 8`,
/// under 1e-8 at the path curvature `k` below 6 on these branches.
const REFERENCE_STEP: f64 = 1e-4;
/// Distance from the saddle along the mode where the reference starts. Off
/// the path by about `k d^2 / 2`, under 1e-11.
const REFERENCE_START: f64 = 1e-6;

fn descent(x: [f64; 2]) -> [f64; 2] {
    let (_, g) = MuellerBrown.eval(ArrayView1::from(&x)).unwrap();
    let norm = g[0].hypot(g[1]);
    [-g[0] / norm, -g[1] / norm]
}

fn shifted(x: [f64; 2], k: [f64; 2], h: f64) -> [f64; 2] {
    [x[0] + h * k[0], x[1] + h * k[1]]
}

/// Steepest descent `dx / dsigma = -g / |g|` by classical Runge-Kutta in arc
/// steps of `REFERENCE_STEP`, to one coarsest radius past `ORDER_ARC`.
fn reference_path(saddle: ArrayView1<f64>, launch: ArrayView1<f64>) -> Vec<[f64; 2]> {
    let h = REFERENCE_STEP;
    let mut x = shifted(
        [saddle[0], saddle[1]],
        [launch[0], launch[1]],
        REFERENCE_START,
    );
    let mut path = vec![[saddle[0], saddle[1]], x];
    let mut arc = REFERENCE_START;
    while arc < ORDER_ARC + COARSEST {
        let k1 = descent(x);
        let k2 = descent(shifted(x, k1, 0.5 * h));
        let k3 = descent(shifted(x, k2, 0.5 * h));
        let k4 = descent(shifted(x, k3, h));
        x = [
            x[0] + h / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]),
            x[1] + h / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]),
        ];
        arc += h;
        path.push(x);
    }
    path
}

fn distance_to(path: &[[f64; 2]], p: ArrayView1<f64>) -> f64 {
    path.windows(2)
        .map(|w| {
            let d = [w[1][0] - w[0][0], w[1][1] - w[0][1]];
            let r = [p[0] - w[0][0], p[1] - w[0][1]];
            let t = ((r[0] * d[0] + r[1] * d[1]) / (d[0] * d[0] + d[1] * d[1])).clamp(0.0, 1.0);
            (r[0] - t * d[0]).hypot(r[1] - t * d[1])
        })
        .fold(f64::INFINITY, f64::min)
}

/// `(dx, error)` for each fitted radius: the largest distance from the
/// reference of the points accepted within `ORDER_ARC` of arc. `force_tol`
/// 1e-4 puts the inner stop, `min(force_tol, 0.01)`, a hundredth of its
/// default, so each minimum on a sphere is resolved below the path error.
fn path_errors(kind: IrcKind, direction: IrcDirection) -> Vec<(f64, f64)> {
    let (saddle, mode) = saddle_and_mode();
    let mut reference = None;
    (1..=HALVINGS)
        .map(|i| {
            let dx = COARSEST / 2f64.powi(i);
            let mut session = IrcSession::new_euclidean(
                IrcConfig {
                    dx,
                    force_tol: 1e-4,
                    kind,
                    ..IrcConfig::default()
                },
                saddle.clone(),
                mode.clone(),
                direction,
            )
            .unwrap();
            let mut error: f64 = 0.0;
            for _ in 0..(ORDER_ARC / dx).round() as usize {
                session.step(&MuellerBrown).unwrap();
                let x = session.position();
                let reference = reference.get_or_insert_with(|| {
                    let launch = if (&x - &saddle).dot(&mode) > 0.0 {
                        mode.clone()
                    } else {
                        -&mode
                    };
                    reference_path(saddle.view(), launch.view())
                });
                error = error.max(distance_to(reference, x));
            }
            (dx, error)
        })
        .collect()
}

/// Least-squares slope of `ln y` against `ln x`.
fn log_slope(points: &[(f64, f64)]) -> f64 {
    let n = points.len() as f64;
    let (mx, my) = points.iter().fold((0.0, 0.0), |(a, b), (x, y)| {
        (a + x.ln() / n, b + y.ln() / n)
    });
    let (sxy, sxx) = points.iter().fold((0.0, 0.0), |(a, b), (x, y)| {
        let u = x.ln() - mx;
        (a + u * (y.ln() - my), b + u * u)
    });
    sxy / sxx
}

/// Let the path error at radius `s` be `C s^p (1 + c s / COARSEST)`, the
/// next term at most the leading one at the coarsest radius, which the fit
/// leaves out: `|c| <= 1`. The fitted slope is then `p` plus the slope of
/// `ln(1 + c s / COARSEST)` over the fitted radii. That slope rises with
/// `c`, so `c = -1` and `c = 1` bound it.
fn order_tolerance() -> f64 {
    [-1.0, 1.0]
        .iter()
        .map(|c| {
            let correction: Vec<(f64, f64)> = (1..=HALVINGS)
                .map(|i| {
                    let s = COARSEST / 2f64.powi(i);
                    (s, 1.0 + c * s / COARSEST)
                })
                .collect();
            log_slope(&correction).abs()
        })
        .fold(0.0, f64::max)
}

/// Gonzalez and Schlegel (J. Chem. Phys. 90, 2154 (1989)) join successive
/// points by an arc tangent to the gradient at both ends, so the path error
/// falls as the square of the radius.
#[test]
fn gs2_path_error_is_second_order_in_the_radius() {
    let tolerance = order_tolerance();
    for direction in [IrcDirection::Forward, IrcDirection::Reverse] {
        let errors = path_errors(IrcKind::Gs2, direction);
        let order = log_slope(&errors);
        assert!(
            (order - 2.0).abs() <= tolerance,
            "{direction:?} order {order} from {errors:?}, tolerance {tolerance}"
        );
    }
}

/// Every `Gs2` step at the finest fitted radius, with the inner stop of the
/// order fits, reaches that stop before its last allowed trial.
#[test]
fn gs2_steps_at_the_finest_radius_stop_before_the_last_trial() {
    let dx = COARSEST / 2f64.powi(HALVINGS);
    let (saddle, mode) = saddle_and_mode();
    for direction in [IrcDirection::Forward, IrcDirection::Reverse] {
        let config = IrcConfig {
            dx,
            force_tol: 1e-4,
            kind: IrcKind::Gs2,
            ..IrcConfig::default()
        };
        let max_inner = config.max_inner;
        let mut session =
            IrcSession::new_euclidean(config, saddle.clone(), mode.clone(), direction).unwrap();
        for step in 1..=(ORDER_ARC / dx).round() as usize {
            let report = session.step(&MuellerBrown).unwrap();
            assert!(
                report.inner_steps < max_inner,
                "{direction:?} step {step} used all {max_inner} trials"
            );
        }
    }
}

/// The minimum on the sphere about the last point leaves along the right
/// tangent with the wrong curvature (Gonzalez and Schlegel 1989, of the
/// Mueller and Brown step), so the path error falls as the radius.
#[test]
fn sella_sphere_path_error_is_first_order_in_the_radius() {
    let tolerance = order_tolerance();
    for direction in [IrcDirection::Forward, IrcDirection::Reverse] {
        let errors = path_errors(IrcKind::SellaSphere, direction);
        let order = log_slope(&errors);
        assert!(
            (order - 1.0).abs() <= tolerance,
            "{direction:?} order {order} from {errors:?}, tolerance {tolerance}"
        );
    }
}
