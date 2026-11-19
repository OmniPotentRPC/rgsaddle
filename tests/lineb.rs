//! Eckart closed forms, a constant-Hessian head-to-head of the two rate
//! evaluators, and energy continuation on Müller-Brown.

use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, Array2, ArrayView1};
use rgsaddle::{
    Continuation, IrcConfig, IrcDirection, IrcKind, IrcSession, LinebConfig, LinebSession,
    LinebStatus, Masses, PointSurface, SaddleError, continue_from_mep, exact_eigh, hqtst_rate,
    hqtst_rate_interpolated_hessian,
};

struct Eckart;

impl PointSurface for Eckart {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let s = 1.0 / x[0].cosh();
        Ok((s * s, Array1::from_elem(1, -2.0 * s * s * x[0].tanh())))
    }
}

/// Inverted parabola `V = 1 − x²/2`. The Hessian is constant, so the
/// cubic interpolation of it is exact and a central Hessian-vector
/// product is exact.
struct Parabola;

impl PointSurface for Parabola {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        Ok((1.0 - 0.5 * x[0] * x[0], Array1::from_elem(1, -x[0])))
    }
}

const MB_A: [f64; 4] = [-200.0, -100.0, -170.0, 15.0];
const MB_AA: [f64; 4] = [-1.0, -1.0, -6.5, 0.7];
const MB_B: [f64; 4] = [0.0, 0.0, 11.0, 0.6];
const MB_C: [f64; 4] = [-10.0, -10.0, -6.5, 0.7];
const MB_X0: [f64; 4] = [1.0, 0.0, -0.5, -1.0];
const MB_Y0: [f64; 4] = [0.0, 0.5, 1.5, 1.0];

fn muller_brown(x: f64, y: f64) -> (f64, [f64; 2]) {
    let mut energy = 0.0;
    let mut gx = 0.0;
    let mut gy = 0.0;
    for i in 0..4 {
        let dx = x - MB_X0[i];
        let dy = y - MB_Y0[i];
        let exponent = MB_AA[i] * dx * dx + MB_B[i] * dx * dy + MB_C[i] * dy * dy;
        let weight = MB_A[i] * exponent.exp();
        energy += weight;
        gx += weight * (2.0 * MB_AA[i] * dx + MB_B[i] * dy);
        gy += weight * (MB_B[i] * dx + 2.0 * MB_C[i] * dy);
    }
    (energy, [gx, gy])
}

struct MullerBrown;

impl PointSurface for MullerBrown {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (energy, g) = muller_brown(x[0], x[1]);
        Ok((energy, Array1::from_vec(vec![g[0], g[1]])))
    }
}

/// An endpoint `δ` short of its turning point shortens the half period
/// by `√(2δ/|∇V|)` and feels the force `endpoint_scale * δ`. This
/// tolerance keeps both ends inside the quadrature slack of the closed
/// form test, `√(spacing / panels⁴ / |∇V|)`.
fn eckart_force_tolerance() -> (f64, LinebConfig) {
    let turning = (1.0 / 0.25_f64).sqrt().acosh();
    let spacing = turning * 2.0 / 8.0;
    let draft = LinebConfig::new(spacing, spacing);
    let panels = (2 * draft.quadrature_pairs) as f64;
    let tol = draft.endpoint_scale * spacing / (8.0 * panels.powi(4));
    (tol, draft.with_opt_tol(tol).with_perp_tol(tol))
}

fn eckart_path() -> rgsaddle::LinebPath {
    let turning = (1.0 / 0.25_f64).sqrt().acosh();
    let images = Array2::from_shape_fn((9, 1), |(i, _)| {
        let end = 0.5 * turning;
        -end + 2.0 * end * i as f64 / 8.0
    });
    let (_, config) = eckart_force_tolerance();
    let mut session = LinebSession::new(
        config,
        Masses::coordinates(Array1::ones(1).view()).unwrap(),
        images,
        0.25,
    )
    .unwrap();
    let report = session.run(&Eckart, 500).unwrap();
    assert_eq!(report.status, LinebStatus::Converged);
    session.path(&Eckart).unwrap()
}

/// Eckart that fails on its `fail_at`-th evaluation.
struct FailingEckart {
    calls: AtomicUsize,
    fail_at: usize,
}

impl PointSurface for FailingEckart {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        if self.calls.fetch_add(1, Ordering::Relaxed) + 1 == self.fail_at {
            return Err(SaddleError::Surface("injected".into()));
        }
        Eckart.eval(x)
    }
}

/// Only an image below `E + w` is retried. A surface error inside the
/// objective ends the step at the failing call.
#[test]
fn a_surface_error_inside_a_step_is_returned() {
    let turning = (1.0 / 0.25_f64).sqrt().acosh();
    let images = Array2::from_shape_fn((9, 1), |(i, _)| {
        let end = 0.5 * turning;
        -end + 2.0 * end * i as f64 / 8.0
    });
    let (_, config) = eckart_force_tolerance();
    let mut session = LinebSession::new(
        config,
        Masses::coordinates(Array1::ones(1).view()).unwrap(),
        images,
        0.25,
    )
    .unwrap();
    // Nine calls fix the turning width; the failure lands in the first
    // assembly of the objective.
    let surface = FailingEckart {
        calls: AtomicUsize::new(0),
        fail_at: 12,
    };
    let err = session.step(&surface).unwrap_err();
    assert!(matches!(err, SaddleError::Surface(_)), "{err}");
    assert_eq!(surface.calls.load(Ordering::Relaxed), 12);
}

#[test]
fn eckart_action_period_and_temperature_match_the_closed_form() {
    let energy = 0.25_f64;
    let path = eckart_path();
    let exact_action = std::f64::consts::PI * (2.0_f64.sqrt() - (2.0 * energy).sqrt());
    let exact_half = std::f64::consts::PI / (2.0 * energy).sqrt();
    let slope = (2.0 * energy) * (1.0 - energy).sqrt();
    let panels = (2 * path_pairs()) as f64;
    let length = path.cartesian[(path.cartesian.nrows() - 1, 0)] - path.cartesian[(0, 0)];
    let spacing = length / (path.cartesian.nrows() - 1) as f64;
    let slack = (spacing / panels.powi(4) / slope).sqrt();
    assert!(
        (path.action - exact_action).abs() <= slack,
        "action {} vs {exact_action}, slack {slack}",
        path.action
    );
    assert!(
        (path.half_period - exact_half).abs() <= slack,
        "half {} vs {exact_half}",
        path.half_period
    );
    assert!(
        (path.half_period - path.half_period_dynamics).abs() <= slack,
        "quadrature {} vs dynamics {}",
        path.half_period,
        path.half_period_dynamics
    );
    let period = 2.0 * path.half_period;
    assert!((path.temperature - 1.0 / period).abs() <= slack / (period * period));
    let line = (2.0 * path.action + energy * period) / 1.0;
    let rate = hqtst_rate(&Eckart, &path, 8).unwrap();
    assert!((rate.line_action - line).abs() <= slack);
    assert_eq!(rate.modes, 8 - 1);
    assert!(rate.temperature > 0.0);
}

fn path_pairs() -> usize {
    eckart_force_tolerance().1.quadrature_pairs
}

#[test]
fn constant_hessian_log_dets_agree() {
    // The Hessian is constant on the whole line, so the images only have
    // to stay where the spline is above the tunneling energy. A straight
    // chain inside the barrier makes the cubic the identity interpolant.
    let images = Array2::from_shape_fn((9, 1), |(i, _)| -0.3 + 0.6 * i as f64 / 8.0);
    let curvature = 1.0;
    let spacing = 0.6 / 8.0;
    let tol = curvature * spacing * f64::EPSILON.sqrt();
    let session = LinebSession::new(
        LinebConfig::new(tol, tol),
        Masses::coordinates(Array1::ones(1).view()).unwrap(),
        images,
        0.5,
    )
    .unwrap();
    let path = session.path(&Parabola).unwrap();
    let beads = 10;
    let direct = hqtst_rate(&Parabola, &path, beads).unwrap();
    let oracle = hqtst_rate_interpolated_hessian(&Parabola, &path, beads).unwrap();
    assert_eq!(direct.modes, beads - 1);
    assert_eq!(oracle.modes, direct.modes);
    // The oracle differences the gradient with step √ε · max|q| and the
    // replica products with ε^{1/3} times the path length. The absolute
    // noise of one Hessian entry is ε · |∇V| / step, at most √ε here,
    // where |∇V| is the coordinate and the curvature is 1. Each kept
    // mode can move the log-determinant by that much. The bound is
    // absolute.
    let entry_noise = f64::EPSILON.sqrt();
    let roundoff = entry_noise * direct.modes as f64;
    assert!(
        (direct.log_abs_det - oracle.log_abs_det).abs() <= roundoff,
        "matrix-free {} vs interpolated {}, bound {roundoff}",
        direct.log_abs_det,
        oracle.log_abs_det
    );
    // The rate's logarithm carries half the log-determinant, with a minus.
    assert!(
        (direct.log_rate_over_reactant - oracle.log_rate_over_reactant).abs() <= 0.5 * roundoff
    );
}

/// The unoptimized nine-image chain on `[−end, end]`, measured at
/// `energy`.
fn straight_path<S: PointSurface>(surface: &S, end: f64, energy: f64) -> rgsaddle::LinebPath {
    let images = Array2::from_shape_fn((9, 1), |(i, _)| -end + 2.0 * end * i as f64 / 8.0);
    let tol = 2.0 * end / 8.0 * f64::EPSILON.sqrt();
    LinebSession::new(
        LinebConfig::new(tol, tol),
        Masses::coordinates(Array1::ones(1).view()).unwrap(),
        images,
        energy,
    )
    .unwrap()
    .path(surface)
    .unwrap()
}

/// Both end knots sit exactly on the turning points `x = ±1` of the
/// parabola at `E = 1/2`. The half period of the inverted harmonic well
/// is `π` at every energy.
#[test]
fn turning_knots_at_both_ends_give_the_harmonic_half_period() {
    let path = straight_path(&Parabola, 1.0, 0.5);
    let panels = (2 * path_pairs()) as f64;
    let slack = (2.0 / 8.0 / panels.powi(4)).sqrt();
    assert!(
        (path.half_period - std::f64::consts::PI).abs() <= slack,
        "{}",
        path.half_period
    );
}

/// The inverted parabola lowered by `SHIFT`, a total energy far below
/// zero.
struct LoweredParabola;

const SHIFT: f64 = -2000.0;

impl PointSurface for LoweredParabola {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (energy, gradient) = Parabola.eval(x)?;
        Ok((energy + SHIFT, gradient))
    }
}

/// Knots `GAP` above the turning points of the lowered parabola, where
/// `V − E` rounds at `ε |SHIFT|`, the size of `GAP`. The half period is
/// still `π`, and the clock costs no more than at the parabola's own
/// energy zero.
#[test]
fn turning_knots_on_a_lowered_surface_keep_the_clock_cheap() {
    const GAP: f64 = 1e-12;
    let raised = straight_path(&Parabola, 1.0, 0.5 - GAP);
    let lowered = straight_path(&LoweredParabola, 1.0, 0.5 + SHIFT - GAP);
    let panels = (2 * path_pairs()) as f64;
    let slack = (2.0 / 8.0 / panels.powi(4)).sqrt();
    assert!(
        (lowered.half_period - std::f64::consts::PI).abs() <= slack,
        "{}",
        lowered.half_period
    );
    assert!(
        lowered.calls.measure <= raised.calls.measure,
        "lowered {} vs raised {}",
        lowered.calls.measure,
        raised.calls.measure
    );
}

/// Lowering every energy by a constant moves `ln(k Z_r)` by `−τ SHIFT / ℏ`
/// and nothing else, even where `k Z_r` itself overflows.
#[test]
fn an_energy_offset_moves_only_the_log_rate() {
    let raised = straight_path(&Parabola, 0.3, 0.5);
    let lowered = straight_path(&LoweredParabola, 0.3, 0.5 + SHIFT);
    let beads = 10;
    let reference = hqtst_rate(&Parabola, &raised, beads).unwrap();
    let rate = hqtst_rate(&LoweredParabola, &lowered, beads).unwrap();
    assert_eq!(rate.rate_over_reactant, f64::INFINITY);
    assert!(rate.log_rate_over_reactant.is_finite());
    let expected = -rate.period * SHIFT;
    let moved = rate.log_rate_over_reactant - reference.log_rate_over_reactant;
    // Each bead energy rounds at `ε |SHIFT|`, and the period and the
    // replicas at the relative `ε |SHIFT| / (V − E)` of the gap.
    let relative = 64.0 * f64::EPSILON * SHIFT.abs();
    assert!(
        (moved - expected).abs() <= relative * expected.abs(),
        "moved {moved}, expected {expected}"
    );
    assert!(
        (rate.log_abs_det - reference.log_abs_det).abs() <= relative * reference.log_abs_det.abs(),
        "{} vs {}",
        rate.log_abs_det,
        reference.log_abs_det
    );
}

fn unstable_mode(saddle: [f64; 2]) -> (Array1<f64>, f64) {
    let scale = saddle[0].abs().max(saddle[1].abs());
    let step = scale * f64::EPSILON.sqrt();
    let mut hessian = Array2::zeros((2, 2));
    for column in 0..2 {
        let mut plus = saddle;
        let mut minus = saddle;
        plus[column] += step;
        minus[column] -= step;
        let (_, gp) = muller_brown(plus[0], plus[1]);
        let (_, gm) = muller_brown(minus[0], minus[1]);
        for row in 0..2 {
            hessian[(row, column)] = (gp[row] - gm[row]) / (2.0 * step);
        }
    }
    for row in 0..2 {
        for column in row + 1..2 {
            let mean = 0.5 * (hessian[(row, column)] + hessian[(column, row)]);
            hessian[(row, column)] = mean;
            hessian[(column, row)] = mean;
        }
    }
    let (values, vectors) = exact_eigh(hessian.view()).unwrap();
    let lowest = if values[1] < values[0] { 1 } else { 0 };
    assert!(
        values[lowest] < 0.0,
        "Müller-Brown saddle curvature {} is not unstable",
        values[lowest]
    );
    (vectors.column(lowest).to_owned(), values[lowest].abs())
}

fn descending_branch(mode: &Array1<f64>, direction: IrcDirection, floor: f64) -> Vec<[f64; 2]> {
    let saddle = Array1::from_vec(vec![-0.82200156, 0.62431280]);
    let config = IrcConfig::default()
        .with_dx(0.08)
        .with_force_tol(1.0)
        .with_kind(IrcKind::Gs2);
    let mut session = IrcSession::new_euclidean(config, saddle, mode.clone(), direction).unwrap();
    let mut points = Vec::new();
    for _ in 0..40 {
        let report = session.step(&MullerBrown).unwrap();
        let position = session.position();
        points.push([position[0], position[1]]);
        if report.energy <= floor || report.energy.is_nan() || report.at_minimum {
            break;
        }
    }
    let end = points.last().copied().unwrap();
    let (energy, _) = muller_brown(end[0], end[1]);
    assert!(
        energy <= floor,
        "the reaction coordinate stopped at energy {energy}, above {floor}"
    );
    points
}

#[test]
fn muller_brown_continuation_lowers_the_energy_and_lengthens_the_period() {
    let saddle = [-0.82200156, 0.62431280];
    let (saddle_energy, _) = muller_brown(saddle[0], saddle[1]);
    let delta = 5.0;
    let (mode, _) = unstable_mode(saddle);
    let mut reactant = descending_branch(&mode, IrcDirection::Reverse, saddle_energy - 3.0 * delta);
    reactant.reverse();
    let mut rows = reactant;
    let saddle_index = rows.len();
    rows.push(saddle);
    rows.extend(descending_branch(
        &mode,
        IrcDirection::Forward,
        saddle_energy - 3.0 * delta,
    ));
    let energies: Vec<f64> = rows
        .iter()
        .map(|point| muller_brown(point[0], point[1]).0)
        .collect();
    let mut top = 0;
    for (index, energy) in energies.iter().enumerate() {
        if *energy > energies[top] {
            top = index;
        }
    }
    assert_eq!(top, saddle_index, "the IRC left the saddle uphill");
    let mep = Array2::from_shape_fn((rows.len(), 2), |(i, k)| rows[i][k]);
    let masses = Masses::coordinates(Array1::ones(2).view()).unwrap();
    let images = 7;
    let target = saddle_energy - delta;
    let mut bracket = 0;
    let mut bracket_gap = f64::INFINITY;
    for (index, point) in rows.iter().enumerate() {
        let gap = (muller_brown(point[0], point[1]).0 - target).abs();
        if gap < bracket_gap {
            bracket = index;
            bracket_gap = gap;
        }
    }
    let (_, gradient) = muller_brown(rows[bracket][0], rows[bracket][1]);
    let gradient_norm = (gradient[0] * gradient[0] + gradient[1] * gradient[1]).sqrt();
    assert!(
        gradient_norm > 0.0,
        "the MEP bracket at the first tunneling energy has no force"
    );
    let draft = LinebConfig::new(gradient_norm, gradient_norm).with_quadrature_pairs(2);
    // Endpoint force `endpoint_scale * |V−E| / |∇V|`. One image's share
    // of the energy step is the energy scale of this continuation.
    let tol = draft.endpoint_scale * delta / (images as f64) / gradient_norm;
    let config = draft.with_opt_tol(tol).with_perp_tol(tol);
    let paths = continue_from_mep(
        &MullerBrown,
        mep.view(),
        &masses,
        &config,
        &Continuation::new(delta, images, 2).with_max_steps(400),
    )
    .unwrap();
    assert_eq!(paths.len(), 2);
    assert!((paths[0].energy - (saddle_energy - delta)).abs() <= delta * f64::EPSILON * 16.0);
    assert!((paths[1].energy - (saddle_energy - 2.0 * delta)).abs() <= delta * f64::EPSILON * 16.0);
    let slope_scale = delta.abs();
    assert!(
        paths[1].half_period > paths[0].half_period + slope_scale * f64::EPSILON,
        "period {} did not grow from {}",
        paths[1].half_period,
        paths[0].half_period
    );
    let panels = (2 * config.quadrature_pairs) as f64;
    for path in &paths {
        assert!(
            (path.half_period - path.half_period_dynamics).abs()
                <= path.half_period.abs() / panels.powi(2),
            "quadrature {} vs dynamics {}",
            path.half_period,
            path.half_period_dynamics
        );
    }
    for beads in [8, 16, 32, 64] {
        let direct = hqtst_rate(&MullerBrown, &paths[0], beads).unwrap();
        let oracle = hqtst_rate_interpolated_hessian(&MullerBrown, &paths[0], beads).unwrap();
        eprintln!(
            "Müller-Brown P = {beads}: replica products {} calls, interpolated oracle {} calls, ln k difference {:e}",
            direct.evaluations,
            oracle.evaluations,
            direct.log_rate_over_reactant - oracle.log_rate_over_reactant
        );
        // Replicas `j` and `P − j` coincide. Each distinct replica costs
        // one gradient and two per central product, one product per
        // coordinate.
        assert_eq!(direct.evaluations, (beads / 2 + 1) * (1 + 2 * 2));
    }
}
