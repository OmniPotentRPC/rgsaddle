//! The reactant normalization by thermodynamic integration over parallel
//! planes.
//!
//! For a unit normal `n` and the planes `H(ξ)` through `x_R + ξ n`,
//! `Z_R = ∫ Z_H(ξ) dξ` over the reactant basin, and the plane free energy
//! `A_R(ξ) = -kT ln Z_H(ξ)` has slope `⟨n · ∇V⟩_ξ`. Hence
//!
//! `Z_H(0) / Z_R = [∫ exp(-(A_R(ξ) - A_R(0)) / kT) dξ]⁻¹`,
//!
//! Eq. (7.3) of the OH-TST design survey. The harmonic standard deviation
//! of `ξ`, `σ_ξ² = kT nᵀ H_R⁻¹ n`, comes from conjugate gradients on
//! Hessian actions at the minimum and sets the spacing of the planes.
//!
//! The planes are sampled from the minimum outward, each from the last
//! configuration of its inner neighbour, and each side keeps its own
//! window. A side ends at the first plane where the rise of `A_R` passes
//! [`BasinConfig::cap`] `kT`, where `A_R` turns over at the edge of the
//! basin, or at [`BasinConfig::width`] standard deviations; a wall behind
//! the minimum and a shoulder before the barrier take as many planes as
//! they need and no more. A cubic spline through the mean forces, with
//! not-a-knot ends so that a cubic mean force is reproduced, integrates to
//! `A_R`. A side that turned over ends at the zero of the spline; past
//! the cap or the width, the integrand continues as the exponential of the
//! last slope. The interpolation error is estimated by leaving out each
//! interior plane in turn: at fourth order, doubling the spacing over two
//! intervals changes `ln ρ` by fifteen times their error. Planes are added
//! on both sides of the plane whose omission moves `ln ρ` most until that
//! estimate falls below [`BasinConfig::refine`] statistical errors.
//!
//! The spline is linear in the mean forces, so the delta method through
//! it gives the error of `ln ρ` exactly to first order. Stein control
//! variates from the Hessian at the minimum, [`BasinConfig::control`],
//! remove the harmonic part of the fluctuations of each mean force; on a
//! quadratic basin they remove all of it.
//!
//! This replaces the crossing estimator of Jóhannesson and Jónsson,
//! J. Chem. Phys. 115, 9644 (2001), Eq. 22, kept as
//! [`ReactantDensity::crossing_density`](crate::ohtst::ReactantDensity::crossing_density):
//! each crossing there is weighted by `1/|ξ̇|`, and under the flux-weighted
//! velocity distribution the second moment of that weight diverges
//! logarithmically at slow crossings.

use std::f64::consts::TAU;

use ndarray::{Array1, ArrayView1};

use super::anharmonic::hessian_action;
use super::control::{HarmonicReference, conjugate_gradients};
use super::plane::Hyperplane;
use super::rate::PlaneDensity;
use super::sample::{Controlled, LangevinConfig, held_directions, sample_plane_controlled};
use crate::spline::{Spline, simpson_rule};
use crate::{PointSurface, SaddleError};

/// Simpson pairs per spline interval of the integral.
const PAIRS: usize = 32;

/// Parallel-plane integration of the reactant normalization.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct BasinConfig {
    /// Sampler for every plane. The `i`-th plane sampled uses the seed
    /// plus `i`.
    pub sampler: LangevinConfig,
    /// Spacing of the first planes, in harmonic standard deviations of
    /// `ξ`.
    pub spacing: f64,
    /// Farthest plane on either side, in harmonic standard deviations.
    pub width: f64,
    /// Rise of the plane free energy, in `kT`, past which a side of the
    /// window takes no more planes.
    pub cap: f64,
    /// Planes are added while the interpolation error of `ln ρ` exceeds
    /// this many statistical errors.
    pub refine: f64,
    /// Nor are they added once the interpolation error of `ln ρ` is below
    /// this: the floor when control variates leave little noise.
    pub resolution: f64,
    /// Most planes added after the first pass.
    pub refinements: usize,
    /// Stein control variates in every plane from the Hessian at the
    /// minimum, unless [`LangevinConfig::control`] already holds a
    /// reference.
    pub control: bool,
    /// Gradient-difference step for Hessian actions at the minimum when
    /// the surface has no analytic one.
    pub hvp_step: f64,
    /// Relative residual of the solves for `H_R⁻¹ n` and the control
    /// coefficients.
    pub tolerance: f64,
}

impl Default for BasinConfig {
    fn default() -> Self {
        Self {
            sampler: LangevinConfig::default(),
            spacing: 1.0,
            width: 8.0,
            cap: 10.0,
            refine: 0.5,
            resolution: 1.0e-4,
            refinements: 8,
            control: true,
            hvp_step: 1.0e-4,
            tolerance: 1.0e-10,
        }
    }
}

impl BasinConfig {
    /// Set [`Self::sampler`].
    #[must_use]
    pub fn with_sampler(mut self, sampler: LangevinConfig) -> Self {
        self.sampler = sampler;
        self
    }

    /// Set [`Self::spacing`].
    #[must_use]
    pub fn with_spacing(mut self, spacing: f64) -> Self {
        self.spacing = spacing;
        self
    }

    /// Set [`Self::width`].
    #[must_use]
    pub fn with_width(mut self, width: f64) -> Self {
        self.width = width;
        self
    }

    /// Set [`Self::cap`].
    #[must_use]
    pub fn with_cap(mut self, cap: f64) -> Self {
        self.cap = cap;
        self
    }

    /// Set [`Self::refine`].
    #[must_use]
    pub fn with_refine(mut self, refine: f64) -> Self {
        self.refine = refine;
        self
    }

    /// Set [`Self::resolution`].
    #[must_use]
    pub fn with_resolution(mut self, resolution: f64) -> Self {
        self.resolution = resolution;
        self
    }

    /// Set [`Self::refinements`].
    #[must_use]
    pub fn with_refinements(mut self, refinements: usize) -> Self {
        self.refinements = refinements;
        self
    }

    /// Set [`Self::control`].
    #[must_use]
    pub fn with_control(mut self, control: bool) -> Self {
        self.control = control;
        self
    }

    /// Set [`Self::hvp_step`].
    #[must_use]
    pub fn with_hvp_step(mut self, hvp_step: f64) -> Self {
        self.hvp_step = hvp_step;
        self
    }

    /// Set [`Self::tolerance`].
    #[must_use]
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    fn check(&self) -> Result<(), SaddleError> {
        self.sampler.check()?;
        if !(self.spacing.is_finite() && self.spacing > 0.0) {
            return Err(SaddleError::Invalid(
                "the basin spacing must be finite and positive".into(),
            ));
        }
        if !(self.width.is_finite() && self.width >= self.spacing) {
            return Err(SaddleError::Invalid(
                "the basin width must be finite and at least one spacing".into(),
            ));
        }
        if !(self.cap.is_finite() && self.cap > 0.0) {
            return Err(SaddleError::Invalid(
                "the basin cap must be finite and positive".into(),
            ));
        }
        if !(nonneg_finite(self.refine) && nonneg_finite(self.resolution)) {
            return Err(SaddleError::Invalid(
                "the basin refinement thresholds must be finite and non-negative".into(),
            ));
        }
        if !(self.hvp_step.is_finite() && self.hvp_step > 0.0) {
            return Err(SaddleError::Invalid(
                "the basin difference step must be finite and positive".into(),
            ));
        }
        if !(self.tolerance > 0.0 && self.tolerance < 1.0) {
            return Err(SaddleError::Invalid(
                "the basin solve tolerance must lie in (0, 1)".into(),
            ));
        }
        Ok(())
    }
}

fn nonneg_finite(x: f64) -> bool {
    x.is_finite() && x >= 0.0
}

/// How a side of the basin window ended.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BasinEnd {
    /// `A_R` rose past [`BasinConfig::cap`]; the integrand continues past
    /// the last plane as the exponential of its slope.
    Cap,
    /// `A_R` turned over: the integral stops at the zero of the spline,
    /// the edge of the basin along the normal.
    Turn,
    /// The last plane sits at [`BasinConfig::width`] before the cap; the
    /// tail is extrapolated as for [`Self::Cap`].
    Width,
}

/// `Z_H / Z_R` on the plane through the minimum, from parallel planes.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct BasinDensity {
    /// The plane through the minimum, with the integration normal.
    pub plane: Hyperplane,
    /// The sampling temperature `kT`.
    pub thermal: f64,
    /// `Z_H(0) / Z_R` per unit mass-weighted length along the normal.
    pub density: f64,
    /// Statistical standard error of [`Self::density`].
    pub density_error: f64,
    /// The harmonic standard deviation `σ_ξ` of `ξ`.
    pub spread: f64,
    /// The harmonic density `1 / (√(2π) σ_ξ)`.
    pub harmonic_density: f64,
    /// Offsets `ξ` of the sampled planes, ascending.
    pub offsets: Vec<f64>,
    /// Mean forces `⟨n · ∇V⟩` on them, the slopes of `A_R`.
    pub slopes: Vec<f64>,
    /// Block standard errors of [`Self::slopes`].
    pub slope_errors: Vec<f64>,
    /// The errors of the slopes without the control variates, when they
    /// were applied.
    pub uncontrolled_errors: Option<Vec<f64>>,
    /// `A_R(ξ) - A_R(0)` at the offsets, from the spline.
    pub profile: Vec<f64>,
    /// The integration window behind and ahead of the minimum; the tails
    /// continue past a side that did not turn over.
    pub window: [f64; 2],
    /// How each side ended, behind and ahead.
    pub ends: [BasinEnd; 2],
    /// Fraction of `Z_R / Z_H(0)` in the extrapolated tails.
    pub tail: f64,
    /// Leave-one-out estimate of the interpolation error of `ln ρ`, not
    /// part of [`Self::density_error`].
    pub quadrature_error: f64,
    /// Surface evaluations, Hessian actions included.
    pub evaluations: usize,
}

impl PlaneDensity for BasinDensity {
    fn plane(&self) -> &Hyperplane {
        &self.plane
    }

    fn thermal(&self) -> f64 {
        self.thermal
    }

    fn density(&self) -> (f64, f64) {
        (self.density, self.density_error)
    }
}

/// Integrate the reactant normalization over planes normal to `normal`
/// around `minimum`. Excluded modes are projected out of `normal` when the
/// sampler holds them.
pub fn basin_density<S: PointSurface>(
    surface: &S,
    minimum: ArrayView1<f64>,
    normal: ArrayView1<f64>,
    config: &BasinConfig,
) -> Result<BasinDensity, SaddleError> {
    config.check()?;
    let n = minimum.len();
    if normal.len() != n {
        return Err(SaddleError::Shape(
            "the basin normal differs in length from the minimum".into(),
        ));
    }
    let excluded = held_directions(surface, minimum, None, config.sampler.hold_excluded)?;
    let mut direction = normal.to_owned();
    for t in &excluded {
        let c = t.dot(&direction);
        direction.scaled_add(-c, t);
    }
    let length = direction.dot(&direction).sqrt();
    if length.is_nan() || length <= 1.0e-12 {
        return Err(SaddleError::Invalid(
            "the basin normal lies in the excluded modes".into(),
        ));
    }
    direction /= length;
    let mut evaluations = 0;
    let solved = conjugate_gradients(
        |v| hessian_action(surface, minimum, v, config.hvp_step, &mut evaluations),
        &direction,
        &excluded,
        config.tolerance,
    )?;
    let compliance = solved.map_or(f64::NAN, |y| y.dot(&direction));
    if !(compliance.is_finite() && compliance > 0.0) {
        return Err(SaddleError::Invalid(
            "the Hessian at the minimum is not positive along the basin normal".into(),
        ));
    }
    let kt = config.sampler.thermal;
    let spread = (kt * compliance).sqrt();
    let sampler = if config.control && config.sampler.control.is_none() {
        config.sampler.clone().with_control(
            HarmonicReference::new(minimum.to_owned())
                .with_hvp_step(config.hvp_step)
                .with_tolerance(config.tolerance),
        )
    } else {
        config.sampler.clone()
    };
    let mut planes = Planes {
        surface,
        minimum,
        direction: &direction,
        sampler: &sampler,
        sampled: 0,
        evaluations,
    };

    let centre = planes.sample(0.0, None)?;
    let mut knots = Vec::new();
    let mut ends = [BasinEnd::Width; 2];
    for (side, sign) in [(0, -1.0), (1, 1.0)] {
        let (mut offset, mut slope, mut rise) = (0.0, centre.slope, 0.0);
        let mut risen = false;
        let mut start = centre.last.clone();
        for k in 1usize.. {
            let reach = k as f64 * config.spacing;
            if reach > config.width * (1.0 + 1.0e-12) {
                break;
            }
            let next = sign * reach * spread;
            let knot = planes.sample(next, Some(start.view()))?;
            rise += 0.5 * (slope + knot.slope) * (next - offset) / kt;
            (offset, slope) = (next, knot.slope);
            let outward = sign * knot.slope;
            let noise = knot.error;
            start = knot.last.clone();
            knots.push(knot);
            if risen && outward <= 0.0 && (outward < -2.0 * noise || rise > 0.5 * config.cap) {
                ends[side] = BasinEnd::Turn;
                break;
            }
            risen |= outward > 0.0;
            if rise > config.cap {
                ends[side] = BasinEnd::Cap;
                break;
            }
        }
    }
    knots.push(centre);
    knots.sort_by(|a, b| a.offset.total_cmp(&b.offset));
    let mut result = integrate(&knots, ends, kt)?;
    let mut added = 0;
    while added < config.refinements {
        let error = result.quadrature_error;
        if error <= config.refine * result.log_error || error <= config.resolution {
            break;
        }
        let Some(worst) = result.worst else {
            break;
        };
        for j in [worst, worst - 1] {
            if added == config.refinements {
                break;
            }
            let (left, right) = (&knots[j], &knots[j + 1]);
            let middle = 0.5 * (left.offset + right.offset);
            let inner = if left.offset.abs() <= right.offset.abs() {
                left
            } else {
                right
            };
            let start = inner.last.clone();
            let knot = planes.sample(middle, Some(start.view()))?;
            knots.insert(j + 1, knot);
            added += 1;
        }
        result = integrate(&knots, ends, kt)?;
    }

    let density = (-result.log_integral).exp();
    if !(density.is_finite() && density > 0.0) {
        return Err(SaddleError::NonFinite("basin integral"));
    }
    let uncontrolled_errors = knots
        .iter()
        .map(|k| k.uncontrolled)
        .collect::<Option<Vec<f64>>>();
    Ok(BasinDensity {
        plane: Hyperplane::new(minimum.to_owned(), direction.clone())?,
        thermal: kt,
        density,
        density_error: density * result.log_error,
        spread,
        harmonic_density: 1.0 / (TAU.sqrt() * spread),
        offsets: knots.iter().map(|k| k.offset).collect(),
        slopes: knots.iter().map(|k| k.slope).collect(),
        slope_errors: knots.iter().map(|k| k.error).collect(),
        uncontrolled_errors,
        profile: result.profile,
        window: result.window,
        ends,
        tail: result.tail,
        quadrature_error: result.quadrature_error,
        evaluations: planes.evaluations,
    })
}

/// One sampled plane.
struct Knot {
    offset: f64,
    slope: f64,
    error: f64,
    uncontrolled: Option<f64>,
    last: Array1<f64>,
}

struct Planes<'a, S> {
    surface: &'a S,
    minimum: ArrayView1<'a, f64>,
    direction: &'a Array1<f64>,
    sampler: &'a LangevinConfig,
    sampled: u64,
    evaluations: usize,
}

impl<S: PointSurface> Planes<'_, S> {
    fn sample(&mut self, offset: f64, start: Option<ArrayView1<f64>>) -> Result<Knot, SaddleError> {
        let mut point = self.minimum.to_owned();
        point.scaled_add(offset, self.direction);
        let plane = Hyperplane::new(point, self.direction.clone())?;
        let sampler = self
            .sampler
            .clone()
            .with_seed(self.sampler.seed.wrapping_add(self.sampled));
        self.sampled += 1;
        let sample = sample_plane_controlled(
            self.surface,
            &plane,
            start,
            &sampler,
            Controlled::NormalForce,
        )?;
        self.evaluations += sample.evaluations;
        let (slope, error) = (-sample.normal_force, sample.normal_force_error);
        if !(slope.is_finite() && error.is_finite() && error >= 0.0) {
            return Err(SaddleError::NonFinite("basin mean force"));
        }
        Ok(Knot {
            offset,
            slope,
            error,
            uncontrolled: sample.uncontrolled.map(|raw| raw.normal_force_error),
            last: sample.last,
        })
    }
}

/// `ln(Z_R / Z_H(0))` from the spline, with its statistical error and the
/// leave-one-out estimate of its interpolation error.
struct Integral {
    log_integral: f64,
    log_error: f64,
    quadrature_error: f64,
    /// The interior plane whose omission moves `ln ρ` most.
    worst: Option<usize>,
    window: [f64; 2],
    tail: f64,
    profile: Vec<f64>,
}

/// The zero of the outward slope `sign · S` past the first knot where it
/// is positive, between the knots where it changes sign.
fn turn_point(spline: &Spline, x: &[f64], f: &[f64], sign: f64) -> Result<f64, SaddleError> {
    let centre = x.partition_point(|&xi| xi < 0.0);
    let outward: Vec<usize> = if sign > 0.0 {
        (centre..x.len()).collect()
    } else {
        (0..=centre.min(x.len() - 1)).rev().collect()
    };
    let mut risen = false;
    for pair in outward.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        risen |= sign * f[a] > 0.0;
        if risen && sign * f[b] <= 0.0 {
            let (mut inside, mut outside) = (x[a], x[b]);
            for _ in 0..60 {
                let middle = 0.5 * (inside + outside);
                if sign * spline.value(middle) > 0.0 {
                    inside = middle;
                } else {
                    outside = middle;
                }
            }
            return Ok(0.5 * (inside + outside));
        }
    }
    Err(SaddleError::Invalid(
        "the basin side turned over but its mean force never changes sign".into(),
    ))
}

/// `∫ exp(-A_R / kT)` over the window and the tails for one set of planes,
/// with the nodes and weights scaled by `exp(-shift)`.
struct Quadrature {
    spline: Spline,
    nodes: Vec<(f64, f64)>,
    weights: Vec<f64>,
    /// Knot index and offset of each extrapolated tail.
    tails: Vec<(usize, f64)>,
    tail_weights: Vec<f64>,
    total: f64,
    log_integral: f64,
    window: [f64; 2],
}

fn quadrature(
    x: &[f64],
    f: &[f64],
    ends: [BasinEnd; 2],
    kt: f64,
) -> Result<Quadrature, SaddleError> {
    let k = x.len();
    let spline = Spline::new(x, f);
    let origin = spline.primitive(0.0);
    let lo = match ends[0] {
        BasinEnd::Turn => turn_point(&spline, x, f, -1.0)?,
        _ => x[0],
    };
    let hi = match ends[1] {
        BasinEnd::Turn => turn_point(&spline, x, f, 1.0)?,
        _ => x[k - 1],
    };
    let mut nodes = Vec::new();
    for j in 0..k - 1 {
        let (a, b) = (x[j].max(lo), x[j + 1].min(hi));
        if b > a {
            nodes.extend(simpson_rule(a, b, PAIRS));
        }
    }
    let mut tails = Vec::new();
    for (end, index, sign) in [(ends[0], 0, -1.0), (ends[1], k - 1, 1.0)] {
        if end == BasinEnd::Turn {
            continue;
        }
        if sign * f[index] > 0.0 {
            tails.push((index, x[index]));
        } else if end == BasinEnd::Width {
            return Err(SaddleError::Invalid(
                "the plane free energy does not rise at the edge of the basin window; widen it"
                    .into(),
            ));
        }
    }
    let reduced: Vec<f64> = nodes
        .iter()
        .map(|&(t, _)| -(spline.primitive(t) - origin) / kt)
        .collect();
    let tail_reduced: Vec<f64> = tails
        .iter()
        .map(|&(index, at)| -(spline.primitive(at) - origin) / kt + (kt / f[index].abs()).ln())
        .collect();
    let shift = reduced
        .iter()
        .chain(&tail_reduced)
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = nodes
        .iter()
        .zip(&reduced)
        .map(|(&(_, w), r)| w * (r - shift).exp())
        .collect();
    let tail_weights: Vec<f64> = tail_reduced.iter().map(|r| (r - shift).exp()).collect();
    let total = weights.iter().sum::<f64>() + tail_weights.iter().sum::<f64>();
    if !(total.is_finite() && total > 0.0) {
        return Err(SaddleError::NonFinite("basin integral"));
    }
    Ok(Quadrature {
        spline,
        nodes,
        weights,
        tails,
        tail_weights,
        total,
        log_integral: shift + total.ln(),
        window: [lo, hi],
    })
}

/// The integral, the delta-method error through the spline, and the
/// leave-one-out interpolation error. Omitting interior plane `i` doubles
/// the spacing over its two intervals, which at fourth order changes
/// `ln ρ` by fifteen times their error; every interval is counted twice,
/// so the error is the sum of the changes over thirty.
fn integrate(knots: &[Knot], ends: [BasinEnd; 2], kt: f64) -> Result<Integral, SaddleError> {
    let k = knots.len();
    let x: Vec<f64> = knots.iter().map(|k| k.offset).collect();
    let f: Vec<f64> = knots.iter().map(|k| k.slope).collect();
    let q = quadrature(&x, &f, ends, kt)?;

    let mut variance = 0.0;
    for i in 0..k {
        let mut unit = vec![0.0; k];
        unit[i] = 1.0;
        let basis = Spline::new(&x, &unit);
        let basis_origin = basis.primitive(0.0);
        let mut moment: f64 = q
            .nodes
            .iter()
            .zip(&q.weights)
            .map(|(&(t, _), w)| w * (basis.primitive(t) - basis_origin))
            .sum();
        let mut direct = 0.0;
        for (&(index, at), w) in q.tails.iter().zip(&q.tail_weights) {
            moment += w * (basis.primitive(at) - basis_origin);
            if index == i {
                direct -= w / f[index];
            }
        }
        let gradient = -moment / (kt * q.total) + direct / q.total;
        variance += (gradient * knots[i].error).powi(2);
    }

    let mut changes = 0.0;
    let mut worst = None;
    let mut largest = 0.0;
    for i in 1..k.saturating_sub(1) {
        let (mut xs, mut fs) = (x.clone(), f.clone());
        xs.remove(i);
        fs.remove(i);
        let Ok(other) = quadrature(&xs, &fs, ends, kt) else {
            continue;
        };
        let change = (other.log_integral - q.log_integral).abs();
        changes += change;
        if change > largest {
            largest = change;
            worst = Some(i);
        }
    }

    let origin = q.spline.primitive(0.0);
    let tail: f64 = q.tail_weights.iter().sum();
    Ok(Integral {
        log_integral: q.log_integral,
        log_error: variance.sqrt(),
        quadrature_error: changes / 30.0,
        worst,
        window: q.window,
        tail: tail / q.total,
        profile: x.iter().map(|&t| q.spline.primitive(t) - origin).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};

    /// `V = ½ qᵀ K q` with one excluded direction.
    struct Bowl {
        k: Array2<f64>,
    }

    impl PointSurface for Bowl {
        fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let g = self.k.dot(&q);
            Ok((0.5 * q.dot(&g), g))
        }

        fn excluded_modes(&self, _q: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
            Ok(array![[0.0, 0.0, 1.0]])
        }
    }

    fn bowl_case(control: bool) -> (BasinDensity, f64) {
        let surface = Bowl {
            k: array![[2.0, 0.6, 0.0], [0.6, 1.0, 0.0], [0.0, 0.0, 5.0]],
        };
        let kt = 0.3;
        let config = BasinConfig::default().with_control(control).with_sampler(
            LangevinConfig::default()
                .with_thermal(kt)
                .with_time_step(0.2)
                .with_records(2000)
                .with_seed(3),
        );
        let normal = array![1.0, 1.0, 0.7];
        let minimum = array![0.0, 0.0, 0.4];
        let result = basin_density(&surface, minimum.view(), normal.view(), &config).unwrap();
        let k_inv = array![[1.0, -0.6], [-0.6, 2.0]] / (2.0 - 0.36);
        let n = array![1.0, 1.0] / 2.0f64.sqrt();
        let spread = (kt * n.dot(&k_inv.dot(&n))).sqrt();
        assert!((result.spread - spread).abs() < 1e-8 * spread);
        (result, 1.0 / (TAU.sqrt() * spread))
    }

    /// On a quadratic basin the density is `1 / √(2π kT nᵀK⁻¹n)` on the
    /// free directions, the excluded one held. Each side ends on the cap
    /// at five standard deviations, where `A_R` has risen 12.5 `kT`.
    #[test]
    fn density_of_a_harmonic_basin() {
        let (result, exact) = bowl_case(false);
        assert!((result.harmonic_density - exact).abs() < 1e-8 * exact);
        assert!(
            (result.density - exact).abs() < 4.0 * result.density_error,
            "{} ± {} against {exact}",
            result.density,
            result.density_error
        );
        assert!(result.density_error < 0.03 * exact);
        assert!((result.plane.normal()[2]).abs() < 1e-12);
        assert_eq!(result.ends, [BasinEnd::Cap, BasinEnd::Cap]);
        assert!((result.offsets[0] + 5.0 * result.spread).abs() < 1e-9);
        assert!((result.offsets.last().unwrap() - 5.0 * result.spread).abs() < 1e-9);
        assert!(result.tail < 1e-5, "{}", result.tail);
    }

    /// With the control variates from the Hessian at the minimum, every
    /// mean force on a quadratic basin is exact and so is the density, to
    /// the Mills-ratio error of the exponential tails.
    #[test]
    fn control_variates_make_a_harmonic_basin_exact() {
        let (result, exact) = bowl_case(true);
        let raw = result.uncontrolled_errors.as_ref().unwrap();
        assert!(raw.iter().all(|e| *e > 1e-3), "{raw:?}");
        assert!(result.slope_errors.iter().all(|e| *e < 1e-10));
        assert!(
            (result.density - exact).abs() < 1e-6 * exact,
            "{} against {exact}",
            result.density
        );
        assert!(
            result.quadrature_error < 1e-6,
            "{}",
            result.quadrature_error
        );
    }

    fn wall_profile(xi: f64) -> f64 {
        0.5 * xi * xi - 0.3 * xi.powi(3) + 0.2 * xi.powi(4)
    }

    /// `V = U(ξ) + (y - ξ/2)²` with `U = ξ²/2 - 0.3 ξ³ + 0.2 ξ⁴`: one
    /// minimum at the origin, steep behind it along `ξ`, and a mean force
    /// `U'(ξ)` on every plane `ξ` = const.
    struct Wall;

    impl PointSurface for Wall {
        fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let (xi, y) = (q[0], q[1]);
            let slope = xi - 0.9 * xi * xi + 0.8 * xi.powi(3);
            let stretch = y - 0.5 * xi;
            Ok((
                wall_profile(xi) + stretch * stretch,
                array![slope - stretch, 2.0 * stretch],
            ))
        }
    }

    fn wall_density(kt: f64) -> f64 {
        let steps = 20_000;
        let h = 10.0 / steps as f64;
        let integral: f64 = (0..=steps)
            .map(|i| {
                let weight = if i == 0 || i == steps {
                    1.0
                } else if i % 2 == 1 {
                    4.0
                } else {
                    2.0
                };
                weight * (-wall_profile(-4.0 + i as f64 * h) / kt).exp()
            })
            .sum::<f64>()
            * h
            / 3.0;
        1.0 / integral
    }

    /// At `kT = 0.25`, `σ_ξ = 0.5` and the first planes sit every 0.5.
    /// `U/kT` is 12.6 at -1.5, past the cap of ten, so the side behind the
    /// minimum ends there, three planes out; ahead it is 4.5 at 1.5 and
    /// 11.2 at 2. The density is `1 / ∫ exp(-U/kT)` by Simpson's rule on
    /// `[-4, 6]`.
    #[test]
    fn the_cap_ends_a_side_at_a_wall() {
        let kt = 0.25;
        let config = BasinConfig::default()
            .with_control(false)
            .with_refinements(0)
            .with_sampler(
                LangevinConfig::default()
                    .with_thermal(kt)
                    .with_time_step(0.2)
                    .with_records(4000)
                    .with_seed(5),
            );
        let result = basin_density(
            &Wall,
            array![0.0, 0.0].view(),
            array![1.0, 0.0].view(),
            &config,
        )
        .unwrap();
        assert!((result.spread - 0.5).abs() < 1e-8);
        assert_eq!(result.offsets.len(), 8, "{:?}", result.offsets);
        assert!((result.offsets[0] + 1.5).abs() < 1e-7);
        assert!((result.offsets[7] - 2.0).abs() < 1e-7);
        assert_eq!(result.ends, [BasinEnd::Cap, BasinEnd::Cap]);
        let exact = wall_density(kt);
        assert!(
            (result.density - exact).abs()
                < 4.0 * result.density_error + result.quadrature_error * exact,
            "{} ± {} against {exact}",
            result.density,
            result.density_error
        );
        assert!(result.density_error < 0.02 * exact);
    }

    /// With the control variates the mean forces on the wall are exact,
    /// since the in-plane coupling is quadratic. The spline reproduces the
    /// cubic `U'`, so leaving out any plane changes nothing, no plane is
    /// added, and the density is exact to the Mills-ratio error of the
    /// tails.
    #[test]
    fn the_spline_reproduces_a_cubic_mean_force() {
        let kt = 0.25;
        let config = BasinConfig::default().with_sampler(
            LangevinConfig::default()
                .with_thermal(kt)
                .with_time_step(0.2)
                .with_records(400)
                .with_seed(5),
        );
        let result = basin_density(
            &Wall,
            array![0.0, 0.0].view(),
            array![1.0, 0.0].view(),
            &config,
        )
        .unwrap();
        assert_eq!(result.offsets.len(), 8, "{:?}", result.offsets);
        assert!(
            result.quadrature_error < 1e-9,
            "{}",
            result.quadrature_error
        );
        let miss = (result.density / wall_density(kt)).ln().abs();
        assert!(miss < 1e-6, "{miss}");
    }

    /// `U = ξ²/2 - 0.3 ξ³ + 0.2 ξ⁴ + 0.05 ξ⁵` has a quartic mean force,
    /// which the spline does not reproduce. With exact mean forces the
    /// refinement adds planes until the leave-one-out estimate is below
    /// the resolution, and the density then misses quadrature by less
    /// than that estimate.
    #[test]
    fn refinement_resolves_a_quartic_mean_force() {
        struct Quintic;
        impl PointSurface for Quintic {
            fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
                let (xi, y) = (q[0], q[1]);
                let stretch = y - 0.5 * xi;
                Ok((
                    quintic(xi) + stretch * stretch,
                    array![
                        xi - 0.9 * xi * xi + 0.8 * xi.powi(3) + 0.25 * xi.powi(4) - stretch,
                        2.0 * stretch
                    ],
                ))
            }
        }
        fn quintic(xi: f64) -> f64 {
            wall_profile(xi) + 0.05 * xi.powi(5)
        }
        let kt = 0.25;
        let sampler = LangevinConfig::default()
            .with_thermal(kt)
            .with_time_step(0.2)
            .with_records(400)
            .with_seed(5);
        let run = |refinements: usize| {
            let config = BasinConfig::default()
                .with_refinements(refinements)
                .with_sampler(sampler.clone());
            basin_density(
                &Quintic,
                array![0.0, 0.0].view(),
                array![1.0, 0.0].view(),
                &config,
            )
            .unwrap()
        };
        let steps = 20_000;
        let h = 10.0 / steps as f64;
        let integral: f64 = (0..=steps)
            .map(|i| {
                let weight = if i == 0 || i == steps {
                    1.0
                } else if i % 2 == 1 {
                    4.0
                } else {
                    2.0
                };
                weight * (-quintic(-3.0 + i as f64 * h) / kt).exp()
            })
            .sum::<f64>()
            * h
            / 3.0;
        let exact = 1.0 / integral;
        let coarse = run(0);
        let fine = run(8);
        let coarse_miss = (coarse.density / exact).ln().abs();
        let fine_miss = (fine.density / exact).ln().abs();
        assert!(
            coarse.quadrature_error > 1e-3,
            "{}",
            coarse.quadrature_error
        );
        assert!(coarse_miss < coarse.quadrature_error, "{coarse_miss}");
        assert!(fine.offsets.len() > coarse.offsets.len());
        assert!(fine.quadrature_error < 1e-3, "{}", fine.quadrature_error);
        assert!(fine_miss < fine.quadrature_error, "{fine_miss}");
    }

    /// `A_R` with a barrier ahead: `U = ξ²/2 - ξ³/3` peaks at `ξ = 1`,
    /// `1/6` above the minimum, 3.3 `kT` at `kT = 0.05`. The side ahead
    /// turns over there, and the window ends at the zero of the spline,
    /// which reproduces the quadratic mean force.
    #[test]
    fn a_side_turns_over_at_the_edge_of_the_basin() {
        struct Ledge;
        impl PointSurface for Ledge {
            fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
                let (xi, y) = (q[0], q[1]);
                Ok((
                    0.5 * xi * xi - xi.powi(3) / 3.0 + y * y,
                    array![xi - xi * xi, 2.0 * y],
                ))
            }
        }
        let kt = 0.05;
        let config = BasinConfig::default().with_sampler(
            LangevinConfig::default()
                .with_thermal(kt)
                .with_time_step(0.1)
                .with_records(400)
                .with_seed(2),
        );
        let result = basin_density(
            &Ledge,
            array![0.0, 0.0].view(),
            array![1.0, 0.0].view(),
            &config,
        )
        .unwrap();
        assert_eq!(result.ends, [BasinEnd::Cap, BasinEnd::Turn]);
        assert!((result.window[1] - 1.0).abs() < 1e-9, "{:?}", result.window);
        let u = |t: f64| 0.5 * t * t - t.powi(3) / 3.0;
        let steps = 20_000;
        let (a, b) = (-2.0, 1.0);
        let h = (b - a) / steps as f64;
        let integral: f64 = (0..=steps)
            .map(|i| {
                let weight = if i == 0 || i == steps {
                    1.0
                } else if i % 2 == 1 {
                    4.0
                } else {
                    2.0
                };
                weight * (-u(a + i as f64 * h) / kt).exp()
            })
            .sum::<f64>()
            * h
            / 3.0;
        let exact = 1.0 / integral;
        let miss = (result.density / exact).ln().abs();
        assert!(miss < 1e-5, "{miss}");
    }

    /// Either refinement threshold may be zero, `-0.0` included, and is
    /// refused when it is NaN, negative or infinite.
    #[test]
    fn refinement_thresholds_must_be_finite_and_non_negative() {
        let cases = [
            (0.5, true),
            (0.0, true),
            (-0.0, true),
            (f64::NAN, false),
            (-1.0, false),
            (-f64::MIN_POSITIVE, false),
            (f64::INFINITY, false),
            (f64::NEG_INFINITY, false),
        ];
        for (refine, refine_ok) in cases {
            for (resolution, resolution_ok) in cases {
                let checked = BasinConfig::default()
                    .with_refine(refine)
                    .with_resolution(resolution)
                    .check();
                assert_eq!(
                    checked.is_ok(),
                    refine_ok && resolution_ok,
                    "refine {refine}, resolution {resolution}: {checked:?}"
                );
            }
        }
    }
}
