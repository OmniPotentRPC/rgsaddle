//! Opt-in Gaussian-process dimer. These tests are the contract: an
//! inverse-distance kernel and its derivatives, marginal-likelihood
//! hyperparameters, a trust radius that stops on the inverse-distance
//! gap, rotation and translation on the surrogate, and a convergence
//! decision taken from the true force.

use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, ArrayView1, array};
use rgsaddle::{
    FitOptions, GaussianProcess, GpDimerConfig, GpDimerSession, GpDimerStatus, KernelKind,
    MinModeConfig, MinModeKind, Observation, PointSurface, SaddleError, fitting_evidence,
    max_log_pair_distance, observation_covariance,
};

struct QuadraticSaddle;

impl PointSurface for QuadraticSaddle {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let e = -x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
        Ok((e, array![-2.0 * x[0], 2.0 * x[1], 2.0 * x[2]]))
    }
}

struct Counting<S> {
    inner: S,
    calls: AtomicUsize,
}

impl<S: PointSurface> PointSurface for Counting<S> {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.inner.eval(x)
    }
}

fn sine_observations() -> Vec<Observation> {
    (0..6)
        .map(|i| {
            let x = i as f64 * 0.3;
            Observation {
                position: array![x],
                energy: x.sin(),
                gradient: array![x.cos()],
            }
        })
        .collect()
}

#[test]
fn gp_dimer_is_opt_in() {
    assert_eq!(MinModeKind::default(), MinModeKind::Dimer);
    assert_eq!(MinModeConfig::default().kind, MinModeKind::Dimer);
    let built = GpDimerConfig::default();
    assert_eq!(built.inner.kind, MinModeKind::Dimer);
    assert!(matches!(built.inner.method, rgmin::Method::Fire { .. }));
}

#[test]
fn inverse_distance_gap_is_rotation_invariant_and_flags_a_stretched_bond() {
    let base = array![0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 0.3, 1.0, 0.1];
    let mut shifted = base.clone();
    for c in shifted.iter_mut() {
        *c += 0.4;
    }
    let same = max_log_pair_distance(base.view(), shifted.view(), None, None).unwrap();
    assert!(same < 1e-12, "translation gap {same}");

    // 90 degree rotation about z.
    let mut turned = Array1::zeros(base.len());
    for a in 0..3 {
        let (x, y, z) = (base[3 * a], base[3 * a + 1], base[3 * a + 2]);
        turned[3 * a] = -y;
        turned[3 * a + 1] = x;
        turned[3 * a + 2] = z;
    }
    let spun = max_log_pair_distance(base.view(), turned.view(), None, None).unwrap();
    assert!(spun < 1e-12, "rotation gap {spun}");

    let mut stretched = base.clone();
    stretched[3] = 1.2 * 1.5;
    let gap = max_log_pair_distance(base.view(), stretched.view(), None, None).unwrap();
    assert!(
        (gap - 1.5_f64.ln()).abs() < 1e-10,
        "stretch by 3/2 should be ln(1.5), got {gap}"
    );
    // Koistinen's window is 2/3 < r'/r < 3/2, i.e. a gap of ln(1.5).
    assert!(gap > 1.2_f64.ln());
}

#[test]
fn inverse_distance_kernel_matches_finite_differences() {
    let x = array![0.1, -0.2, 0.3, 1.0, 0.4, -0.1, 0.2, 1.1, 0.5];
    let y = array![0.0, 0.2, -0.2, 0.9, -0.3, 0.4, -0.4, 0.8, 0.2];
    let kind = KernelKind::InverseDistance {
        cell: None,
        fixed: None,
    };
    let amplitude2 = 0.8;
    let length = 0.6;
    let block = observation_covariance(&kind, x.view(), y.view(), amplitude2, length).unwrap();
    let eps = 1e-6;
    let dim = x.len();
    for a in 0..dim {
        let mut xp = x.clone();
        let mut xm = x.clone();
        xp[a] += eps;
        xm[a] -= eps;
        let kp = observation_covariance(&kind, xp.view(), y.view(), amplitude2, length).unwrap();
        let km = observation_covariance(&kind, xm.view(), y.view(), amplitude2, length).unwrap();
        let fd = (kp[(0, 0)] - km[(0, 0)]) / (2.0 * eps);
        let analytic = block[(1 + a, 0)];
        assert!(
            (fd - analytic).abs() < 1e-6,
            "d k / d x[{a}]: fd {fd} analytic {analytic}"
        );
        for b in 0..dim {
            let mut yp = y.clone();
            let mut ym = y.clone();
            yp[b] += eps;
            ym[b] -= eps;
            let hp =
                observation_covariance(&kind, x.view(), yp.view(), amplitude2, length).unwrap();
            let hm =
                observation_covariance(&kind, x.view(), ym.view(), amplitude2, length).unwrap();
            let fd_h = (hp[(1 + a, 0)] - hm[(1 + a, 0)]) / (2.0 * eps);
            let analytic_h = block[(1 + a, 1 + b)];
            assert!(
                (fd_h - analytic_h).abs() < 2e-5,
                "d²k / d x[{a}] d y[{b}]: fd {fd_h} analytic {analytic_h}"
            );
        }
    }
}

#[test]
fn marginal_likelihood_fit_is_a_local_maximum() {
    let data = sine_observations();
    let kind = KernelKind::Cartesian;
    let opts = FitOptions {
        passes: 8,
        ..FitOptions::default()
    };
    let gp = GaussianProcess::fit(&data, &kind, &opts).unwrap();
    let here = fitting_evidence(&data, &kind, gp.amplitude2(), gp.length_scale()).unwrap();
    assert!(here.is_finite(), "evidence {here}");
    for (factor_a, factor_l) in [(0.5, 1.0), (2.0, 1.0), (1.0, 0.5), (1.0, 2.0)] {
        let there = fitting_evidence(
            &data,
            &kind,
            gp.amplitude2() * factor_a,
            gp.length_scale() * factor_l,
        )
        .unwrap();
        assert!(
            here + 1e-3 >= there,
            "fitted evidence {here} lost to amplitude×{factor_a} length×{factor_l} ({there})"
        );
    }
    // Derivative observations pin the training values.
    for obs in &data {
        let (energy, gradient) = gp.predict(obs.position.view()).unwrap();
        assert!(
            (energy - obs.energy).abs() < 1e-3,
            "energy {energy} vs {}",
            obs.energy
        );
        assert!(
            (gradient[0] - obs.gradient[0]).abs() < 1e-3,
            "gradient {gradient} vs {}",
            obs.gradient
        );
    }
    let mut worst = 0.0_f64;
    for i in 0..5 {
        let x = 0.15 + i as f64 * 0.3;
        let (energy, _) = gp.predict(array![x].view()).unwrap();
        worst = worst.max((energy - x.sin()).abs());
    }
    assert!(worst < 0.05, "held-out sine error {worst}");
}

#[test]
fn one_observation_hessian_on_the_quadratic() {
    let x = array![0.35, 0.4, -0.3];
    let (energy, gradient) = QuadraticSaddle.eval(x.view()).unwrap();
    let mut xp = x.clone();
    xp[0] += 0.05;
    let (ep, gp_) = QuadraticSaddle.eval(xp.view()).unwrap();
    let data = [
        Observation {
            position: x.clone(),
            energy,
            gradient: gradient.clone(),
        },
        Observation {
            position: xp,
            energy: ep,
            gradient: gp_,
        },
    ];
    let gp = GaussianProcess::fit(&data, &KernelKind::Cartesian, &FitOptions::default()).unwrap();
    let (_, g0) = gp.predict(x.view()).unwrap();
    for i in 0..3 {
        assert!(
            (g0[i] - gradient[i]).abs() < 1e-3,
            "grad {i}: {} vs {}",
            g0[i],
            gradient[i]
        );
    }
    let eps = 1e-4;
    let mut hess = [[0.0; 3]; 3];
    for j in 0..3 {
        let mut xp = x.clone();
        let mut xm = x.clone();
        xp[j] += eps;
        xm[j] -= eps;
        let (_, gp_) = gp.predict(xp.view()).unwrap();
        let (_, gm) = gp.predict(xm.view()).unwrap();
        for i in 0..3 {
            hess[i][j] = (gp_[i] - gm[i]) / (2.0 * eps);
        }
    }
    // True Hessian is diag(-2, 2, 2). One extra sample along x is enough
    // for that direction to be the most negative curvature of the surrogate.
    assert!(hess[0][0] < -1.0, "curvature along x is {}", hess[0][0]);
    assert!(hess[0][0] < hess[1][1], "y curvature {}", hess[1][1]);
    assert!(hess[0][0] < hess[2][2], "z curvature {}", hess[2][2]);
}

#[test]
fn predicted_gradient_matches_the_energy_slope() {
    let data = sine_observations();
    let gp = GaussianProcess::fit(&data, &KernelKind::Cartesian, &FitOptions::default()).unwrap();
    let x0 = 0.45;
    let (energy, gradient) = gp.predict(array![x0].view()).unwrap();
    let eps = 1e-5;
    let (ep, _) = gp.predict(array![x0 + eps].view()).unwrap();
    let (em, _) = gp.predict(array![x0 - eps].view()).unwrap();
    let slope = (ep - em) / (2.0 * eps);
    assert!(
        (slope - gradient[0]).abs() < 1e-4,
        "slope {slope} gradient {}",
        gradient[0]
    );
    let _ = energy;
}

#[test]
fn surrogate_walk_respects_the_trust_radius() {
    let config = GpDimerConfig {
        trust_radius: 0.05,
        max_inner_steps: 8,
        inner: MinModeConfig {
            max_move: 0.4,
            force_tol: 1e-6,
            ..MinModeConfig::default()
        },
        ..GpDimerConfig::default()
    };
    let start = array![0.35, 0.4, -0.3];
    let mut session = GpDimerSession::new(config, start.clone(), array![1.0, 0.3, 0.0]).unwrap();
    let report = session.step(&QuadraticSaddle).unwrap();
    let step = &session.position() - &start;
    let distance = step.dot(&step).sqrt();
    assert!(
        distance <= 0.08,
        "Cartesian trust radius exceeded: {distance}, report {report:?}"
    );
    assert!(distance > 1e-4, "surrogate did not move, {report:?}");
    assert_eq!(report.status, GpDimerStatus::Running);
}

#[test]
fn molecular_early_stop_uses_the_inverse_distance_gap() {
    // A 2% change in any pair distance ends the surrogate walk.
    let config = GpDimerConfig {
        trust_radius: 0.02,
        max_inner_steps: 6,
        ratio_at_limit: 0.0,
        inner: MinModeConfig {
            max_move: 0.3,
            force_tol: 1e-8,
            ..MinModeConfig::default()
        },
        ..GpDimerConfig::default()
    };
    let start = array![0.0, 0.0, 0.0, 1.1, 0.05, 0.0, 0.2, 0.9, 0.15];
    let mode = array![1.0, 0.0, 0.0, -0.5, 0.2, 0.0, 0.1, -0.2, 0.3];
    let mut session = GpDimerSession::new(config, start.clone(), mode).unwrap();
    let report = session.step(&ThreeBody).unwrap();
    let gap = max_log_pair_distance(start.view(), session.position(), None, None).unwrap();
    assert!(
        gap <= 0.035,
        "inverse-distance early stop let the gap reach {gap}, {report:?}"
    );
}

/// Three atoms, one soft bond along x and a stiff penalty on the rest,
/// so the dimer has a direction to walk.
struct ThreeBody;

impl PointSurface for ThreeBody {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let mut energy = 0.0;
        let mut gradient = Array1::zeros(x.len());
        let pairs = [(0, 1), (0, 2), (1, 2)];
        let targets = [1.3, 1.1, 1.2];
        for (pair, target) in pairs.iter().zip(targets) {
            let (i, j) = *pair;
            let mut d = [0.0; 3];
            let mut r2 = 0.0;
            for k in 0..3 {
                d[k] = x[3 * i + k] - x[3 * j + k];
                r2 += d[k] * d[k];
            }
            let r = r2.sqrt();
            let diff = r - target;
            let weight = if *pair == (0, 1) { 1.0 } else { 8.0 };
            energy += weight * diff * diff;
            let scale = 2.0 * weight * diff / r;
            for k in 0..3 {
                gradient[3 * i + k] += scale * d[k];
                gradient[3 * j + k] -= scale * d[k];
            }
        }
        Ok((energy, gradient))
    }
}

#[test]
fn convergence_is_decided_on_the_true_force() {
    let config = GpDimerConfig {
        max_inner_steps: 25,
        inner: MinModeConfig {
            force_tol: 1e-4,
            max_move: 0.15,
            ..MinModeConfig::default()
        },
        ..GpDimerConfig::default()
    };
    let start = array![0.35, 0.4, -0.3];
    let surface = Counting {
        inner: QuadraticSaddle,
        calls: AtomicUsize::new(0),
    };
    let mut session = GpDimerSession::new(config, start, array![0.8, 0.5, 0.1]).unwrap();
    let report = session.run(&surface, 30).unwrap();
    assert_eq!(
        report.status,
        GpDimerStatus::Converged,
        "max_force {} curvature {}",
        report.max_force,
        report.curvature
    );
    let x = session.position();
    for (i, c) in x.iter().enumerate() {
        assert!(c.abs() < 5e-3, "coord {i} = {c}");
    }
    let (_, gradient) = QuadraticSaddle.eval(x).unwrap();
    let true_force = gradient.iter().copied().map(f64::abs).fold(0.0, f64::max);
    assert!(true_force <= 1e-4, "true force {true_force}");
    assert!((report.max_force - true_force).abs() < 1e-8);
    assert!(report.curvature < 0.0, "curvature {}", report.curvature);
    let calls = surface.calls.load(Ordering::Relaxed);
    assert!(calls < 80, "true evaluations {calls}");
}

/// Müller-Brown saddle between the upper minima. The plain dimer stops
/// here; the GP dimer must as well, on the true force.
struct MullerBrown;

impl PointSurface for MullerBrown {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        const PREFACTOR: [f64; 4] = [-200.0, -100.0, -170.0, 15.0];
        const XX: [f64; 4] = [-1.0, -1.0, -6.5, 0.7];
        const XY: [f64; 4] = [0.0, 0.0, 11.0, 0.6];
        const YY: [f64; 4] = [-10.0, -10.0, -6.5, 0.7];
        const X0: [f64; 4] = [1.0, 0.0, -0.5, -1.0];
        const Y0: [f64; 4] = [0.0, 0.5, 1.5, 1.0];
        let (mut energy, mut gx, mut gy) = (0.0, 0.0, 0.0);
        for i in 0..4 {
            let (dx, dy) = (x[0] - X0[i], x[1] - Y0[i]);
            let term = PREFACTOR[i] * (XX[i] * dx * dx + XY[i] * dx * dy + YY[i] * dy * dy).exp();
            energy += term;
            gx += term * (2.0 * XX[i] * dx + XY[i] * dy);
            gy += term * (XY[i] * dx + 2.0 * YY[i] * dy);
        }
        energy += 2.0 * x[2] * x[2];
        Ok((energy, array![gx, gy, 4.0 * x[2]]))
    }
}

#[test]
fn gp_dimer_converges_on_muller_brown() {
    let config = GpDimerConfig {
        inner: MinModeConfig {
            force_tol: 1e-3,
            max_move: 0.2,
            ..MinModeConfig::default()
        },
        max_inner_steps: 25,
        ..GpDimerConfig::default()
    };
    let start = array![-0.6, 0.8, 0.0];
    let surface = Counting {
        inner: MullerBrown,
        calls: AtomicUsize::new(0),
    };
    let mut session = GpDimerSession::new(config, start, array![-0.8, -0.5, 0.02]).unwrap();
    let report = session.run(&surface, 40).unwrap();
    assert_eq!(
        report.status,
        GpDimerStatus::Converged,
        "force {} curvature {}",
        report.max_force,
        report.curvature
    );
    let x = session.position();
    assert!(
        (x[0] + 0.822).abs() < 2e-2 && (x[1] - 0.624).abs() < 2e-2 && x[2].abs() < 2e-2,
        "landed at {x}"
    );
    let (energy, gradient) = MullerBrown.eval(x).unwrap();
    let true_force = gradient.iter().copied().map(f64::abs).fold(0.0, f64::max);
    assert!(true_force <= 1e-3, "true force {true_force}");
    assert!((energy + 40.664843).abs() < 1e-3, "energy {energy}");
    let calls = surface.calls.load(Ordering::Relaxed);
    assert!(
        calls < 80,
        "true evaluations {calls} are not below the plain dimer"
    );
}
