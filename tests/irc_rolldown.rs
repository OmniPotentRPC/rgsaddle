//! Two-atom double well: saddle at the origin, minima at x = ±1.
//! Rolling forward and reverse must leave the saddle in opposite ways.
//!
//! Gonzalez--Schlegel / Sella: every accepted point sits on the
//! mass-weighted sphere of radius `dx` about the previous accepted
//! point. That set is rgmin `IrcTrust`, not `ManifoldKind::Sphere`.

use ndarray::{Array1, ArrayView1};
use rgmin::IrcTrust;
use rgsaddle::{IrcConfig, IrcDirection, IrcSession, PointSurface, SaddleError};

struct DoubleWell;

impl PointSurface for DoubleWell {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        // Atom 0 moves in x; the rest are dummy coordinates.
        let t = x[0];
        let energy = (t * t - 1.0) * (t * t - 1.0);
        let mut g = Array1::zeros(x.len());
        g[0] = 4.0 * t * (t * t - 1.0);
        Ok((energy, g))
    }
}

#[test]
fn forward_and_reverse_leave_the_saddle_opposite_ways() {
    let saddle = Array1::zeros(6);
    let masses = Array1::from(vec![1.0, 1.0]);
    let mut mode = Array1::zeros(6);
    mode[0] = 1.0;
    let cfg = IrcConfig {
        dx: 0.2,
        force_tol: 1e-3,
        ..IrcConfig::default()
    };
    let mut session = IrcSession::new(
        cfg.clone(),
        saddle.clone(),
        masses.clone(),
        mode.clone(),
        IrcDirection::Forward,
    )
    .unwrap();
    let fwd = session.step(&DoubleWell).unwrap();
    let x_fwd = session.position()[0];
    assert!(x_fwd.abs() > 1e-8, "kick must leave the saddle");
    assert!(fwd.arc > 0.0);

    session.set_direction(IrcDirection::Reverse);
    let _rev = session.step(&DoubleWell).unwrap();
    let x_rev = session.position()[0];
    assert!(
        x_fwd * x_rev < 0.0,
        "forward {x_fwd} and reverse {x_rev} must have opposite signs"
    );
}

#[test]
fn kick_mode_comes_from_matrix_free_lanczos_not_a_full_heev() {
    let saddle = Array1::zeros(6);
    let masses = Array1::from(vec![1.0, 1.0]);
    let seed = Array1::from(vec![1.0, 0.1, 0.0, 0.0, 0.0, 0.0]);
    let mut session = IrcSession::from_surface(
        IrcConfig::default(),
        saddle,
        masses,
        seed,
        IrcDirection::Forward,
        &DoubleWell,
    )
    .unwrap();
    let _ = session.step(&DoubleWell).unwrap();
    let x0 = session.position()[0];
    assert!(x0.abs() > 1e-8, "Lanczos kick must leave the saddle");
}

#[test]
fn run_reaches_a_well() {
    let saddle = Array1::zeros(6);
    let masses = Array1::from(vec![1.0, 1.0]);
    let seed = Array1::from(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let mut session = IrcSession::from_surface(
        IrcConfig {
            dx: 0.15,
            force_tol: 0.05,
            ..IrcConfig::default()
        },
        saddle,
        masses,
        seed,
        IrcDirection::Forward,
        &DoubleWell,
    )
    .unwrap();
    let report = session.run(&DoubleWell, 40).unwrap();
    assert!(report.at_minimum, "force {} arc {}", report.max_force, report.arc);
    assert!(
        (session.position()[0].abs() - 1.0).abs() < 0.25,
        "x={}",
        session.position()[0]
    );
    assert!(report.energy < 0.1, "energy {}", report.energy);
}

/// Müller–Brown embedded in 6-D: the 2-D PES on (x, y) of atom 0,
/// plus a harmonic well on the dummy coordinates so the 6-D Hessian
/// is positive definite at each minimum.
struct MullerBrown;

const MB_A: [f64; 4] = [-200.0, -100.0, -170.0, 15.0];
const MB_A_COEF: [f64; 4] = [-1.0, -1.0, -6.5, 0.7];
const MB_B: [f64; 4] = [0.0, 0.0, 11.0, 0.6];
const MB_C: [f64; 4] = [-10.0, -10.0, -6.5, 0.7];
const MB_X0: [f64; 4] = [1.0, 0.0, -0.5, -1.0];
const MB_Y0: [f64; 4] = [0.0, 0.5, 1.5, 1.0];
const MB_K: f64 = 2.0;

impl PointSurface for MullerBrown {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (px, py) = (x[0], x[1]);
        let mut energy = 0.0;
        let mut gx = 0.0;
        let mut gy = 0.0;
        for i in 0..4 {
            let dx = px - MB_X0[i];
            let dy = py - MB_Y0[i];
            let e = MB_A[i]
                * (MB_A_COEF[i] * dx * dx + MB_B[i] * dx * dy + MB_C[i] * dy * dy).exp();
            energy += e;
            gx += e * (2.0 * MB_A_COEF[i] * dx + MB_B[i] * dy);
            gy += e * (MB_B[i] * dx + 2.0 * MB_C[i] * dy);
        }
        let mut g = Array1::zeros(x.len());
        g[0] = gx;
        g[1] = gy;
        for i in 2..x.len() {
            energy += 0.5 * MB_K * x[i] * x[i];
            g[i] = MB_K * x[i];
        }
        Ok((energy, g))
    }
}

fn fd_lowest_curvature(surface: &MullerBrown, x: ArrayView1<f64>, dr: f64) -> f64 {
    let (_, g0) = surface.eval(x).unwrap();
    let n = x.len();
    let mut h = ndarray::Array2::<f64>::zeros((n, n));
    for i in 0..n {
        let mut xp = x.to_owned();
        xp[i] += dr;
        let (_, gp) = surface.eval(xp.view()).unwrap();
        for j in 0..n {
            h[(j, i)] = (gp[j] - g0[j]) / dr;
        }
    }
    for i in 0..n {
        for j in i + 1..n {
            let s = 0.5 * (h[(i, j)] + h[(j, i)]);
            h[(i, j)] = s;
            h[(j, i)] = s;
        }
    }
    struct Dense(ndarray::Array2<f64>);
    impl rgmin::ApplyHessian for Dense {
        fn apply_hessian(
            &self,
            _x: ArrayView1<f64>,
            v: ArrayView1<f64>,
        ) -> Array1<f64> {
            self.0.dot(&v)
        }
    }
    let seed = Array1::from(vec![1.0, 0.3, 0.1, 0.0, 0.0, 0.0]);
    rgmin::lowest_eigenpair(&Dense(h), x, seed.view(), 6).value
}

fn roll_branch(direction: IrcDirection) -> (Array1<f64>, f64, f64, f64, bool) {
    let mut saddle = Array1::zeros(6);
    saddle[0] = -0.822;
    saddle[1] = 0.624;
    let masses = Array1::from(vec![1.0, 1.0]);
    let seed = Array1::from(vec![1.0, 0.2, 0.0, 0.0, 0.0, 0.0]);
    let mut session = IrcSession::from_surface(
        IrcConfig {
            dx: 0.1,
            force_tol: 0.05,
            max_inner: 10,
            ..IrcConfig::default()
        },
        saddle,
        masses,
        seed,
        direction,
        &MullerBrown,
    )
    .unwrap();
    let report = session.run(&MullerBrown, 40).unwrap();
    let x = session.position().to_owned();
    let curv = fd_lowest_curvature(&MullerBrown, x.view(), 1e-4);
    (x, report.energy, curv, report.arc, report.at_minimum)
}

#[test]
fn muller_brown_both_ways_ends_at_minima_with_positive_curvature() {
    let (xf, ef, cf, af, minf) = roll_branch(IrcDirection::Forward);
    let (xr, er, cr, ar, minr) = roll_branch(IrcDirection::Reverse);
    let sep = ((xf[0] - xr[0]) * (xf[0] - xr[0]) + (xf[1] - xr[1]) * (xf[1] - xr[1])).sqrt();
    assert!(
        sep > 0.5,
        "forward ({}, {}) E={ef} arc={af} min={minf} and reverse ({}, {}) E={er} arc={ar} min={minr} must be distinct wells (sep={sep})",
        xf[0],
        xf[1],
        xr[0],
        xr[1]
    );
    for (label, x, energy, curv) in [
        ("forward", xf, ef, cf),
        ("reverse", xr, er, cr),
    ] {
        assert!(
            curv > 0.0,
            "{label} ended at ({}, {}) E={energy} lambda_min={curv}",
            x[0],
            x[1]
        );
        assert!(
            energy < -70.0,
            "{label} energy {energy} is not a Müller–Brown well"
        );
    }
}

fn mw_step_on_sphere(masses: &Array1<f64>, dx: f64, from: &Array1<f64>, to: &Array1<f64>) -> bool {
    let s = to - from;
    let tr = IrcTrust::from_atom_masses(Array1::zeros(s.len()), masses.as_slice().unwrap(), dx);
    tr.on_bound(&s, 1e-8)
}

/// The GS2 / Sella set: `||sqrt(m) (x_k - x_{k-1})|| = dx`.
/// The mode lives on both atoms so unequal masses make the Euclidean
/// sphere of radius `dx` a different set.
#[test]
fn accepted_points_sit_on_the_mw_sphere() {
    let saddle = Array1::zeros(6);
    let masses = Array1::from(vec![1.0, 16.0]);
    let mut mode = Array1::zeros(6);
    mode[0] = 1.0;
    mode[3] = 1.0;
    let dx = 0.2;
    let cfg = IrcConfig {
        dx,
        force_tol: 1e-3,
        ..IrcConfig::default()
    };
    let mut session = IrcSession::new(
        cfg,
        saddle.clone(),
        masses.clone(),
        mode,
        IrcDirection::Forward,
    )
    .unwrap();

    let x0 = session.position().to_owned();
    let _ = session.step(&DoubleWell).unwrap();
    let x1 = session.position().to_owned();
    assert!(
        mw_step_on_sphere(&masses, dx, &x0, &x1),
        "kick left the MW-sphere: from={x0:?} to={x1:?}"
    );

    let _ = session.step(&DoubleWell).unwrap();
    let x2 = session.position().to_owned();
    assert!(
        mw_step_on_sphere(&masses, dx, &x1, &x2),
        "inner step left the MW-sphere about the last accepted point: from={x1:?} to={x2:?}"
    );
}
