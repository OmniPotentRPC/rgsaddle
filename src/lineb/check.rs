//! Müller-Brown action and period against a fine grid, and the force
//! calls that grid's paths took.
//!
//! The two-dimensional illustration in Ásgeirsson, Arnaldsson and
//! Jónsson, J. Chem. Phys. 148, 102334 (2018) is a modified Müller-Brown
//! surface; this one keeps the standard constants. The paper's force-call
//! figure, an average of about 900 evaluations to compute the reactive
//! flux at each temperature (Sec. II.E), is for H2 desorbing from a
//! 216-atom Cu(110) slab on an empirical potential. rgpot serves a Cu-H
//! EAM of that slab over RPC as `CuH2`. This crate does not depend on
//! rgpot, so that comparison runs outside its tests, through a
//! [`PointSurface`] around the RPC client.
//!
//! The integrals below are a separate Simpson rule on the same cubic.
//! The allowance is that rule's own truncation:
//! the change from the library's panel count to the grid that has
//! settled at the roundoff of its sum.

use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, Array2, ArrayView1};

use super::action::l2;
use super::mass::Masses;
use super::optimize::{LinebConfig, LinebPath};
use super::path::{Curve, Level};
use crate::error::SaddleError;
use crate::minmode::PointSurface;
use crate::{
    Continuation, IrcConfig, IrcDirection, IrcKind, IrcSession, continue_from_mep, exact_eigh,
};

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

struct Sample {
    sigma: f64,
    speed: f64,
    slope: f64,
}

fn probe<S: PointSurface>(
    surface: &S,
    masses: &Masses,
    curve: &Curve,
    energy: f64,
    lambda: f64,
) -> Result<Sample, SaddleError> {
    let point = curve.at(lambda);
    let tangent = curve.tangent(lambda);
    let speed = l2(tangent.view());
    if speed <= 0.0 || speed.is_nan() {
        return Err(SaddleError::Invalid(
            "the reference grid hit a vanishing tangent".into(),
        ));
    }
    let cartesian = masses.to_cartesian(point.view())?;
    let (potential, gradient) = surface.eval(cartesian.view())?;
    let force = masses.weighted_force(gradient.view())?;
    let gap = potential - energy;
    Ok(Sample {
        sigma: if gap > 0.0 { (2.0 * gap).sqrt() } else { 0.0 },
        speed,
        slope: -force.dot(&tangent),
    })
}

fn integrands(sample: &Sample, u: Option<f64>) -> Result<(f64, f64), SaddleError> {
    let Some(u) = u else {
        if sample.sigma <= 0.0 || sample.sigma.is_nan() {
            return Err(SaddleError::Invalid(
                "the reference momentum vanishes inside a chord".into(),
            ));
        }
        return Ok((sample.sigma * sample.speed, sample.speed / sample.sigma));
    };
    let action = sample.sigma * sample.speed * 2.0 * u;
    let period = if sample.sigma > 0.0 && u > 0.0 {
        2.0 * u * sample.speed / sample.sigma
    } else if (u <= 0.0 || u.is_nan()) && (sample.sigma <= 0.0 || sample.sigma.is_nan()) {
        if sample.slope <= 0.0 || sample.slope.is_nan() {
            return Err(SaddleError::Invalid(
                "the reference potential does not rise off the turning point".into(),
            ));
        }
        2.0 * sample.speed / (2.0 * sample.slope).sqrt()
    } else if u <= 0.0 || u.is_nan() {
        0.0
    } else {
        return Err(SaddleError::Invalid(
            "the reference momentum vanishes before the end of the path".into(),
        ));
    };
    Ok((action, period))
}

/// Composite Simpson on `values`, spaced by `width`. Returns the integral
/// and the sum of the absolute weighted samples.
fn simpson(values: &[f64], width: f64) -> (f64, f64) {
    let intervals = values.len() - 1;
    let mut sum = 0.0;
    let mut absolute = 0.0;
    for (index, value) in values.iter().enumerate() {
        let weight = if index == 0 || index == intervals {
            1.0
        } else if index % 2 == 1 {
            4.0
        } else {
            2.0
        };
        let scaled = weight * width / 3.0;
        sum += scaled * value;
        absolute += scaled * value.abs();
    }
    (sum, absolute)
}

fn chord<S: PointSurface>(
    level: &Level<'_, S>,
    (left, right): (f64, f64),
    panels: usize,
    end: Option<bool>,
) -> Result<(f64, f64, f64, f64), SaddleError> {
    let mut action_samples = Vec::with_capacity(panels + 1);
    let mut period_samples = Vec::with_capacity(panels + 1);
    let width = match end {
        Some(_) => (right - left).sqrt() / panels as f64,
        None => (right - left) / panels as f64,
    };
    if width <= 0.0 || width.is_nan() {
        return Err(SaddleError::Stalled(
            "a reference panel no longer advances".into(),
        ));
    }
    for index in 0..=panels {
        let (lambda, u) = match end {
            Some(true) => {
                let u = width * index as f64;
                (left + u * u, Some(u))
            }
            Some(false) => {
                let u = width * index as f64;
                (right - u * u, Some(u))
            }
            None => (left + width * index as f64, None),
        };
        let mut sample = probe(
            level.surface,
            level.masses,
            level.curve,
            level.energy,
            lambda,
        )?;
        if end == Some(false) {
            // The barrier lies toward smaller `λ` on the end chord.
            sample.slope = -sample.slope;
        }
        let (action, period) = integrands(&sample, u)?;
        action_samples.push(action);
        period_samples.push(period);
    }
    let (action, action_abs) = simpson(&action_samples, width);
    let (period, period_abs) = simpson(&period_samples, width);
    Ok((action, period, action_abs, period_abs))
}

struct Grid {
    action: f64,
    half_period: f64,
    /// `|I(n) − I(converged)|` at the library panel count, plus the
    /// converged grid's own roundoff.
    action_error: f64,
    half_error: f64,
}

/// Independent Simpson rule on the cubic, doubled until the change is
/// within the roundoff of the weighted samples. The reported error is
/// the change from the library panel count to that converged grid,
/// plus the converged grid's roundoff.
fn fine_grid<S: PointSurface>(surface: &S, path: &LinebPath) -> Result<Grid, SaddleError> {
    let curve = Curve::new(path.weighted.view())?;
    let knots = curve.knots().to_vec();
    let level = Level {
        surface,
        masses: &path.masses,
        curve: &curve,
        energy: path.energy,
    };
    let last = knots.len() - 1;
    let library_panels = 2 * path.quadrature_pairs;
    let mut panels = 2usize;
    let mut previous: Option<(f64, f64)> = None;
    let mut at_library: Option<(f64, f64)> = None;
    loop {
        let mut action = 0.0;
        let mut half = 0.0;
        let mut action_abs = 0.0;
        let mut half_abs = 0.0;
        for chord_index in 0..last {
            let end = if chord_index == 0 {
                Some(true)
            } else if chord_index + 1 == last {
                Some(false)
            } else {
                None
            };
            let (da, dh, aa, ha) = chord(
                &level,
                (knots[chord_index], knots[chord_index + 1]),
                panels,
                end,
            )?;
            action += da;
            half += dh;
            action_abs += aa;
            half_abs += ha;
        }
        if panels == library_panels {
            at_library = Some((action, half));
        }
        if let Some((prev_action, prev_half)) = previous {
            let action_change = (action - prev_action).abs();
            let half_change = (half - prev_half).abs();
            let samples = (panels as f64) * (last as f64);
            let action_noise = action_abs * f64::EPSILON * samples;
            let half_noise = half_abs * f64::EPSILON * samples;
            if (action_change <= action_noise || action_change.is_nan())
                && (half_change <= half_noise || half_change.is_nan())
            {
                // A grid that has already settled before the library
                // panel count is at least as fine as that count.
                let (library_action, library_half) = match at_library {
                    Some(sampled) => sampled,
                    None => (action, half),
                };
                return Ok(Grid {
                    action,
                    half_period: half,
                    action_error: (action - library_action).abs() + action_change.max(action_noise),
                    half_error: (half - library_half).abs() + half_change.max(half_noise),
                });
            }
        }
        panels = panels
            .checked_mul(2)
            .ok_or_else(|| SaddleError::Stalled("the reference grid stopped advancing".into()))?;
        previous = Some((action, half));
    }
}

fn descending(mode: &Array1<f64>, direction: IrcDirection, floor: f64) -> Vec<[f64; 2]> {
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
    points
}

fn unstable_mode(saddle: [f64; 2]) -> Array1<f64> {
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
    assert!(values[lowest] < 0.0);
    vectors.column(lowest).to_owned()
}

/// The Müller-Brown MEP through the lower saddle down to `levels + 1`
/// energy steps below it, and the force norm where the first tunneling
/// energy crosses it.
fn muller_brown_mep(delta: f64, levels: usize) -> (Array2<f64>, f64) {
    let saddle = [-0.82200156, 0.62431280];
    let (saddle_energy, _) = muller_brown(saddle[0], saddle[1]);
    let mode = unstable_mode(saddle);
    let floor = saddle_energy - (levels as f64 + 1.0) * delta;
    let mut reactant = descending(&mode, IrcDirection::Reverse, floor);
    reactant.reverse();
    let mut rows = reactant;
    rows.push(saddle);
    rows.extend(descending(&mode, IrcDirection::Forward, floor));
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
    assert!(gradient_norm > 0.0);
    let mep = Array2::from_shape_fn((rows.len(), 2), |(i, k)| rows[i][k]);
    (mep, gradient_norm)
}

/// Endpoint force `endpoint_scale * |V−E| / |∇V|`. One image's share of
/// the energy step is the energy scale of the continuation.
fn muller_brown_config(delta: f64, images: usize, gradient_norm: f64) -> LinebConfig {
    let draft = LinebConfig::new(gradient_norm, gradient_norm);
    let tol = draft.endpoint_scale * delta / (images as f64) / gradient_norm;
    draft.with_opt_tol(tol).with_perp_tol(tol)
}

#[test]
fn muller_brown_action_and_period_match_a_fine_grid() {
    let delta = 5.0;
    let levels = 3;
    let images = 7;
    let (mep, gradient_norm) = muller_brown_mep(delta, levels);
    let config = muller_brown_config(delta, images, gradient_norm);
    let masses = Masses::coordinates(Array1::ones(2).view()).unwrap();
    let calls = AtomicUsize::new(0);
    let counted = crate::lineb::Counted {
        surface: &MullerBrown,
        calls: &calls,
    };
    let paths = continue_from_mep(
        &counted,
        mep.view(),
        &masses,
        &config,
        &Continuation::new(delta, images, levels).with_max_steps(800),
    )
    .unwrap();
    assert_eq!(paths.len(), levels);
    let total = calls.load(Ordering::Relaxed);
    let optimizer: usize = paths.iter().map(|path| path.calls.optimizer).sum();
    let measure: usize = paths.iter().map(|path| path.calls.measure).sum();
    let dynamics: usize = paths.iter().map(|path| path.calls.dynamics).sum();
    let seed = total - optimizer - measure - dynamics;
    eprintln!(
        "Müller-Brown force calls over {levels} energies: optimizer {optimizer}, measure {measure}, dynamics {dynamics}, seeds {seed}"
    );
    assert!(optimizer > 0 && measure > 0 && dynamics > 0);
    assert!(seed < total);
    for path in &paths {
        let grid = fine_grid(&MullerBrown, path).unwrap();
        assert!(
            (path.action - grid.action).abs() <= grid.action_error,
            "W({}) = {} vs fine grid {}, truncation {}",
            path.energy,
            path.action,
            grid.action,
            grid.action_error
        );
        assert!(
            (path.half_period - grid.half_period).abs() <= grid.half_error,
            "half period at {} = {} vs fine grid {}, truncation {}",
            path.energy,
            path.half_period,
            grid.half_period,
            grid.half_error
        );
    }
}
