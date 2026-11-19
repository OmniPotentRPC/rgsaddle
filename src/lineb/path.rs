//! Cubic path, the period `−dW/dE`, and half-period dynamics.
//!
//! Images are knots of a cubic spline in the chord-length parameter `λ`.
//! The one-way action along that spline is `W = ∫ √(2(V−E)) ds` and the
//! half period is `−∂W/∂E = ∫ ds / √(2(V−E))`. The same half period is
//! the travel time of a particle on the spline under the inverted
//! potential,
//!
//! `λ̈ = −(T · f + (T · ∂T/∂λ) λ̇²) / (T · T)`,
//!
//! with `T = ∂q/∂λ`. This is the constrained dynamics in appendix B of
//! the LI-NEB review (doi:10.1021/acs.jctc.5c00673), which Ásgeirsson,
//! Arnaldsson and Jónsson, J. Chem. Phys. 148, 102334 (2018), use to
//! read the temperature of an optimal tunneling path. The closed orbit
//! has period `τ = 2` half periods and temperature `ℏ / (k_B τ)`.

use ndarray::{Array1, Array2, ArrayView2};

use super::action::l2;
use super::mass::Masses;
use crate::error::SaddleError;
use crate::minmode::PointSurface;
use crate::spline::Spline;

#[derive(Clone, Debug)]
pub(crate) struct Curve {
    knots: Vec<f64>,
    coords: Vec<Spline>,
}

impl Curve {
    pub(crate) fn new(weighted: ArrayView2<f64>) -> Result<Self, SaddleError> {
        let n = weighted.nrows();
        let dof = weighted.ncols();
        if n < 3 {
            return Err(SaddleError::Shape(
                "a cubic tunneling path needs at least three images".into(),
            ));
        }
        let mut knots = Vec::with_capacity(n);
        knots.push(0.0);
        for j in 1..n {
            let mut length_sq = 0.0;
            for k in 0..dof {
                let d = weighted[(j, k)] - weighted[(j - 1, k)];
                length_sq += d * d;
            }
            let length = length_sq.sqrt();
            if length <= 0.0 || length.is_nan() {
                return Err(SaddleError::Invalid(
                    "a cubic tunneling path needs a positive chord between images".into(),
                ));
            }
            knots.push(knots[j - 1] + length);
        }
        let mut coords = Vec::with_capacity(dof);
        for k in 0..dof {
            let values: Vec<f64> = (0..n).map(|j| weighted[(j, k)]).collect();
            coords.push(Spline::new(&knots, &values));
        }
        Ok(Self { knots, coords })
    }

    pub(crate) fn length(&self) -> f64 {
        self.knots[self.knots.len() - 1]
    }

    #[cfg(any(test, feature = "lineb-oracle"))]
    pub(crate) fn knots(&self) -> &[f64] {
        &self.knots
    }

    pub(crate) fn at(&self, lambda: f64) -> Array1<f64> {
        Array1::from_iter(self.coords.iter().map(|spline| spline.value(lambda)))
    }

    pub(crate) fn tangent(&self, lambda: f64) -> Array1<f64> {
        Array1::from_iter(self.coords.iter().map(|spline| spline.derivative(lambda)))
    }

    fn curvature(&self, lambda: f64) -> Array1<f64> {
        Array1::from_iter(
            self.coords
                .iter()
                .map(|spline| spline.second_derivative(lambda)),
        )
    }
}

/// `(λ, t)` samples of the imaginary time along a spline, in path order.
pub(crate) type TimeTable = Vec<(f64, f64)>;

#[derive(Clone, Debug)]
pub(crate) struct Periods {
    /// `∫ √(2(V−E)) ds` along the spline, one way.
    pub action: f64,
    /// `∫ ds / √(2(V−E))`, one way. This is `−∂W/∂E`.
    pub half_period: f64,
    /// Travel time of the constrained dynamics from one end to the other.
    pub half_period_dynamics: f64,
    /// `(λ, t)` samples of the imaginary time along the spline.
    pub time_table: TimeTable,
}

struct Sample {
    sigma: f64,
    speed: f64,
    /// `∂V/∂λ = −f · T`. Positive when the path climbs into the barrier.
    slope: f64,
}

fn sample<S: PointSurface>(
    surface: &S,
    masses: &Masses,
    curve: &Curve,
    energy: f64,
    lambda: f64,
) -> Result<Sample, SaddleError> {
    let q = curve.at(lambda);
    let tangent = curve.tangent(lambda);
    let speed = l2(tangent.view());
    if speed <= 0.0 || speed.is_nan() {
        return Err(SaddleError::Invalid(
            "the cubic path has a vanishing tangent".into(),
        ));
    }
    let x = masses.to_cartesian(q.view())?;
    let (potential, gradient) = surface.eval(x.view())?;
    if !potential.is_finite() || gradient.iter().any(|component| !component.is_finite()) {
        return Err(SaddleError::NonFinite("path sample"));
    }
    let force = masses.weighted_force(gradient.view())?;
    let gap = potential - energy;
    let sigma = if gap > 0.0 { (2.0 * gap).sqrt() } else { 0.0 };
    Ok(Sample {
        sigma,
        speed,
        slope: -force.dot(&tangent),
    })
}

/// Period integrand in the end-point coordinate `u`, where
/// `λ = λ_turn ± u²` so `dλ = 2 u du`. On the end chord at the start the
/// barrier lies toward larger `λ`; on the one at the end, toward smaller.
fn end_rate(sample: &Sample, u: f64, from_start: bool) -> Result<f64, SaddleError> {
    let rise = if from_start {
        sample.slope
    } else {
        -sample.slope
    };
    let rate = if sample.sigma > 0.0 && u > 0.0 {
        2.0 * u * sample.speed / sample.sigma
    } else if (u <= 0.0 || u.is_nan()) && (sample.sigma <= 0.0 || sample.sigma.is_nan()) {
        if rise <= 0.0 || rise.is_nan() {
            return Err(SaddleError::Invalid(
                "the potential does not rise as the path enters the barrier".into(),
            ));
        }
        2.0 * sample.speed / (2.0 * rise).sqrt()
    } else if u <= 0.0 || u.is_nan() {
        // `V > E` at the knot, so the `u = 0` sample of `2 u / σ` is zero.
        0.0
    } else {
        return Err(SaddleError::Invalid(
            "the tunneling momentum vanishes before the end of the path".into(),
        ));
    };
    if !rate.is_finite() {
        return Err(SaddleError::NonFinite("period integrand"));
    }
    Ok(rate)
}

fn simpson_weight(i: usize, intervals: usize) -> f64 {
    if i == 0 || i == intervals {
        1.0
    } else if i % 2 == 1 {
        4.0
    } else {
        2.0
    }
}

/// Action, both half periods and the imaginary-time table. `pairs` is
/// the number of Simpson pairs on each chord. The quadrature and the
/// time table call `quadrature`; the constrained dynamics calls
/// `dynamics`, so the two costs can be counted apart.
pub(crate) fn measure<Q: PointSurface, D: PointSurface>(
    quadrature: &Q,
    dynamics_surface: &D,
    masses: &Masses,
    weighted: ArrayView2<f64>,
    energy: f64,
    pairs: usize,
) -> Result<Periods, SaddleError> {
    if pairs < 1 {
        return Err(SaddleError::Invalid(
            "the period quadrature needs at least one Simpson pair per chord".into(),
        ));
    }
    let curve = Curve::new(weighted)?;
    let level = Level {
        surface: quadrature,
        masses,
        curve: &curve,
        energy,
    };
    let (action, half_period, time_table) = integrate(&level, pairs)?;
    let half_period_dynamics = dynamics(dynamics_surface, masses, &curve, energy, pairs)?;
    Ok(Periods {
        action,
        half_period,
        half_period_dynamics,
        time_table,
    })
}

/// Action, half period and the `(λ, t)` time table from one pass of
/// samples. Interior chords add to the table by the trapezoid rule on
/// the Simpson samples; end chords add their adaptive clock.
fn integrate<S: PointSurface>(
    level: &Level<'_, S>,
    pairs: usize,
) -> Result<(f64, f64, TimeTable), SaddleError> {
    let mut action = 0.0;
    let mut half = 0.0;
    let mut table = vec![(0.0, 0.0)];
    let mut clock = 0.0;
    let knots = &level.curve.knots;
    let last = knots.len() - 1;
    for j in 0..last {
        let (left, right) = (knots[j], knots[j + 1]);
        if j == 0 || j + 1 == last {
            let chord = EndChord::new(left, right, j == 0)?;
            action += end_action(level, chord, pairs)?;
            let samples = end_clock(level, chord, pairs)?;
            let gained = samples.last().map(|(_, period)| *period).ok_or_else(|| {
                SaddleError::Invalid("the end chord produced no period samples".into())
            })?;
            for (lambda, period) in samples.into_iter().skip(1) {
                table.push((lambda, clock + period));
            }
            clock += gained;
            half += gained;
        } else {
            let (segment_action, segment_half) =
                open_integral(level, left, right, pairs, &mut table, &mut clock)?;
            action += segment_action;
            half += segment_half;
        }
    }
    if !(action.is_finite() && half.is_finite() && half > 0.0) {
        return Err(SaddleError::NonFinite("tunneling period"));
    }
    Ok((action, half, table))
}

fn open_integral<S: PointSurface>(
    level: &Level<'_, S>,
    left: f64,
    right: f64,
    pairs: usize,
    table: &mut Vec<(f64, f64)>,
    clock: &mut f64,
) -> Result<(f64, f64), SaddleError> {
    let intervals = 2 * pairs;
    let h = (right - left) / intervals as f64;
    let mut action = 0.0;
    let mut half = 0.0;
    let mut previous_rate = 0.0;
    for i in 0..=intervals {
        let weight = simpson_weight(i, intervals);
        let lambda = left + h * i as f64;
        let sample = level.sample(lambda)?;
        if sample.sigma <= 0.0 || sample.sigma.is_nan() {
            return Err(SaddleError::Invalid(
                "the tunneling momentum vanishes in the interior of the path".into(),
            ));
        }
        let rate = sample.speed / sample.sigma;
        action += weight * sample.sigma * sample.speed;
        half += weight * rate;
        if i > 0 {
            *clock += 0.5 * (previous_rate + rate) * h;
            table.push((lambda, *clock));
        }
        previous_rate = rate;
    }
    Ok((action * h / 3.0, half * h / 3.0))
}

/// The fixed-energy problem on one spline.
pub(crate) struct Level<'a, S> {
    pub(crate) surface: &'a S,
    pub(crate) masses: &'a Masses,
    pub(crate) curve: &'a Curve,
    pub(crate) energy: f64,
}

impl<S: PointSurface> Level<'_, S> {
    fn sample(&self, lambda: f64) -> Result<Sample, SaddleError> {
        sample(self.surface, self.masses, self.curve, self.energy, lambda)
    }
}

/// A chord that ends at a turning knot, in the coordinate `u` with
/// `λ = λ_turn ± u²`.
#[derive(Clone, Copy)]
struct EndChord {
    left: f64,
    right: f64,
    from_start: bool,
    u_max: f64,
}

impl EndChord {
    fn new(left: f64, right: f64, from_start: bool) -> Result<Self, SaddleError> {
        let u_max = (right - left).sqrt();
        if u_max <= 0.0 || u_max.is_nan() {
            return Err(SaddleError::Invalid(
                "an end chord of the tunneling path has no length".into(),
            ));
        }
        Ok(Self {
            left,
            right,
            from_start,
            u_max,
        })
    }

    /// `λ` at the distance `s` from the chord's path-order start.
    fn lambda(&self, s: f64) -> f64 {
        if self.from_start {
            self.left + s * s
        } else {
            let u = self.u_max - s;
            self.right - u * u
        }
    }

    /// `u` at the distance `s` from the chord's path-order start.
    fn u(&self, s: f64) -> f64 {
        if self.from_start { s } else { self.u_max - s }
    }
}

/// `∫ σ ds` on an end chord, Simpson in `u`.
fn end_action<S: PointSurface>(
    level: &Level<'_, S>,
    chord: EndChord,
    pairs: usize,
) -> Result<f64, SaddleError> {
    let intervals = 2 * pairs;
    let h = chord.u_max / intervals as f64;
    let mut action = 0.0;
    for i in 0..=intervals {
        let weight = simpson_weight(i, intervals);
        let u = h * i as f64;
        let lambda = if chord.from_start {
            chord.left + u * u
        } else {
            chord.right - u * u
        };
        let sample = level.sample(lambda)?;
        action += weight * sample.sigma * sample.speed * 2.0 * u;
    }
    Ok(action * h / 3.0)
}

/// Period samples along an end chord, in path order. `s` runs from the
/// interior of the chord toward the turning knot on the right end, and
/// from the turning knot inward on the left. A piece no wider than the
/// caller's panel is halved until Simpson and its two halves agree to
/// within the allowance plus the rounding of the integrand.
///
/// Next to the turning knot `V − E` is a difference of two nearly equal
/// energies. An error `δV` in `V` moves the integrand `1/σ` by
/// `δV/σ²` of itself, and once that exceeds the allowance, halving
/// cannot make the two estimates agree. `δV` is taken as
/// `ENERGY_ULPS · ε · |E|`.
fn end_clock<S: PointSurface>(
    level: &Level<'_, S>,
    chord: EndChord,
    pairs: usize,
) -> Result<Vec<(f64, f64)>, SaddleError> {
    let u_max = chord.u_max;
    let panels = (2 * pairs) as f64;
    let panel = u_max / panels;
    let tol = (chord.right - chord.left) / panels.powi(4);
    let mut s = 0.0;
    let mut period = 0.0;
    let mut out = vec![(chord.lambda(0.0), 0.0)];
    while u_max > s {
        let mut du = panel.min(u_max - s);
        loop {
            if du <= 0.0 || du.is_nan() {
                return Err(SaddleError::Stalled(
                    "the end-chord period step underflowed".into(),
                ));
            }
            let (coarse, coarse_rounding) = simpson_s(level, chord, s, s + du)?;
            let mid = s + 0.5 * du;
            let (left, left_rounding) = simpson_s(level, chord, s, mid)?;
            let (right, right_rounding) = simpson_s(level, chord, mid, s + du)?;
            let fine = left + right;
            let allowance = tol * du / u_max + coarse_rounding + left_rounding + right_rounding;
            if (fine - coarse).abs() <= allowance {
                let next = s + du;
                if next <= s || next.is_nan() {
                    return Err(SaddleError::Stalled(
                        "the end-chord period step stopped advancing".into(),
                    ));
                }
                period += fine + (fine - coarse) / 15.0;
                s = next;
                out.push((chord.lambda(s), period));
                break;
            }
            du *= 0.5;
        }
    }
    Ok(out)
}

/// Rounding of the energy, in units of `ε |E|`, that the end-chord
/// clock allows for.
const ENERGY_ULPS: f64 = 64.0;

/// Simpson on `[s0, s1]`, and the bound on its change from an error of
/// `ENERGY_ULPS · ε · |E|` in every sampled `V`.
fn simpson_s<S: PointSurface>(
    level: &Level<'_, S>,
    chord: EndChord,
    s0: f64,
    s1: f64,
) -> Result<(f64, f64), SaddleError> {
    let delta_v = ENERGY_ULPS * f64::EPSILON * level.energy.abs();
    let mid = 0.5 * (s0 + s1);
    let mut acc = 0.0;
    let mut rounding = 0.0;
    for (s, weight) in [(s0, 1.0), (mid, 4.0), (s1, 1.0)] {
        let taken = level.sample(chord.lambda(s))?;
        let rate = end_rate(&taken, chord.u(s), chord.from_start)?;
        acc += weight * rate;
        if taken.sigma > 0.0 {
            rounding += weight * rate * delta_v / (taken.sigma * taken.sigma);
        }
    }
    Ok((acc * (s1 - s0) / 6.0, rounding * (s1 - s0) / 6.0))
}

/// Fourth-order Runge–Kutta with step doubling. A step is accepted when
/// the two-half-step parameter agrees with the full step to within a
/// chord squared over the path length, the resolution of the spline.
/// The step doubles when that discrepancy is smaller by `2⁵`, the order
/// of the local Runge–Kutta error, and halves otherwise.
///
/// The dynamics sees only forces, so its energy drifts by the accepted
/// discrepancies. Into an end knot barely above `E` that drift can turn
/// the particle short of the knot. A turn within the summed allowance of
/// the accepted steps closes the gap `r` at the turning-point time
/// `√(2r/|a|)`.
fn dynamics<S: PointSurface>(
    surface: &S,
    masses: &Masses,
    curve: &Curve,
    energy: f64,
    pairs: usize,
) -> Result<f64, SaddleError> {
    let length = curve.length();
    let spacing = length / (curve.knots.len() - 1) as f64;
    // Simpson on a chord is fourth order in the panel width. The
    // Runge–Kutta discrepancy has to stay under that panel's truncation
    // or the half period drifts off the quadrature.
    let panels = (2 * pairs) as f64;
    let tolerance = spacing / panels.powi(4);
    let started = sample(surface, masses, curve, energy, 0.0)?;
    let mut velocity = started.sigma / started.speed;
    let accel0 = acceleration(surface, masses, curve, 0.0, velocity)?;
    if (accel0 <= 0.0 || accel0.is_nan()) && (velocity <= 0.0 || velocity.is_nan()) {
        return Err(SaddleError::Invalid(
            "the path does not move into the barrier at its first image".into(),
        ));
    }
    let driving = accel0.abs().max(velocity * velocity / spacing);
    let mut dt = (2.0 * spacing / driving).sqrt() / (2 * pairs) as f64;
    let mut lambda = 0.0;
    let mut time = 0.0;
    let mut accepted = 0.0;
    while lambda < length {
        let remaining = length - lambda;
        if remaining <= tolerance || remaining.is_nan() {
            // The last gap is inside one accepted step. On the way into a
            // turning point, v = √(2 |∂V/∂λ| · remaining), so the gap
            // takes twice remaining/v.
            if velocity > 0.0 {
                time += 2.0 * remaining / velocity;
            }
            break;
        }
        if dt <= 0.0 || dt.is_nan() {
            return Err(SaddleError::Stalled(format!(
                "the half-period step underflowed at parameter {lambda} of {length} with velocity {velocity}"
            )));
        }
        let (full_lambda, _) = rk4(surface, masses, curve, lambda, velocity, dt)?;
        let (mid_lambda, mid_velocity) = rk4(surface, masses, curve, lambda, velocity, 0.5 * dt)?;
        let (half_lambda, half_velocity) =
            rk4(surface, masses, curve, mid_lambda, mid_velocity, 0.5 * dt)?;
        let discrepancy = (full_lambda - half_lambda).abs();
        let advanced = half_lambda - lambda;
        // Keep the step moving toward the far end. A step that passes the
        // turning point and falls back has a non-positive end velocity.
        let forward = advanced > 0.0 && half_velocity > 0.0;
        if discrepancy <= tolerance && forward {
            if half_lambda >= length {
                time += dt * remaining / advanced;
                lambda = length;
            } else {
                let advanced_to = half_lambda;
                if advanced_to <= lambda || advanced_to.is_nan() {
                    return Err(SaddleError::Stalled(
                        "the half-period step stopped advancing along the path".into(),
                    ));
                }
                lambda = advanced_to;
                velocity = half_velocity;
                time += dt;
                accepted += 1.0;
                if discrepancy * 32.0 <= tolerance || discrepancy.is_nan() {
                    dt *= 2.0;
                }
            }
        } else if half_velocity <= 0.0 && remaining <= (accepted + 1.0) * tolerance {
            let accel = acceleration(surface, masses, curve, lambda, velocity)?;
            if accel >= 0.0 || accel.is_nan() {
                return Err(SaddleError::Invalid(
                    "the path turns back before its end without a turning point".into(),
                ));
            }
            time += (2.0 * remaining / -accel).sqrt();
            break;
        } else {
            dt *= 0.5;
        }
        if !time.is_finite() {
            return Err(SaddleError::NonFinite("half-period dynamics"));
        }
    }
    Ok(time)
}

fn rk4<S: PointSurface>(
    surface: &S,
    masses: &Masses,
    curve: &Curve,
    lambda: f64,
    velocity: f64,
    dt: f64,
) -> Result<(f64, f64), SaddleError> {
    let a1 = acceleration(surface, masses, curve, lambda, velocity)?;
    let v2 = velocity + 0.5 * dt * a1;
    let a2 = acceleration(surface, masses, curve, lambda + 0.5 * dt * velocity, v2)?;
    let v3 = velocity + 0.5 * dt * a2;
    let a3 = acceleration(surface, masses, curve, lambda + 0.5 * dt * v2, v3)?;
    let v4 = velocity + dt * a3;
    let a4 = acceleration(surface, masses, curve, lambda + dt * v3, v4)?;
    let lambda_next = lambda + dt * (velocity + 2.0 * v2 + 2.0 * v3 + v4) / 6.0;
    let velocity_next = velocity + dt * (a1 + 2.0 * a2 + 2.0 * a3 + a4) / 6.0;
    if !(lambda_next.is_finite() && velocity_next.is_finite()) {
        return Err(SaddleError::NonFinite("Runge-Kutta stage"));
    }
    Ok((lambda_next, velocity_next))
}

fn acceleration<S: PointSurface>(
    surface: &S,
    masses: &Masses,
    curve: &Curve,
    lambda: f64,
    velocity: f64,
) -> Result<f64, SaddleError> {
    let tangent = curve.tangent(lambda);
    let curvature = curve.curvature(lambda);
    let q = curve.at(lambda);
    let x = masses.to_cartesian(q.view())?;
    let (_, gradient) = surface.eval(x.view())?;
    if gradient.iter().any(|component| !component.is_finite()) {
        return Err(SaddleError::NonFinite("path acceleration"));
    }
    let force = masses.weighted_force(gradient.view())?;
    let scale = tangent.dot(&tangent);
    if scale <= 0.0 || scale.is_nan() {
        return Err(SaddleError::Invalid(
            "the cubic path has a vanishing tangent".into(),
        ));
    }
    let accel = -(tangent.dot(&force) + tangent.dot(&curvature) * velocity * velocity) / scale;
    if !accel.is_finite() {
        return Err(SaddleError::NonFinite("path acceleration"));
    }
    Ok(accel)
}

/// Replicas at `t_j = j τ / P` on the closed orbit, with the forward
/// chord parameter of each replica (the return half uses the same
/// parameter as its outbound twin).
pub(crate) struct Replicas {
    pub weighted: Array2<f64>,
    #[cfg(any(test, feature = "lineb-oracle"))]
    pub parameter: Vec<f64>,
    pub period: f64,
}

/// Replicas at `t_j = j τ / P`. The return half retraces the spline.
/// The period is twice the last time in `table`, the time table that
/// [`measure`] built on the same images.
pub(crate) fn replicas(
    weighted: ArrayView2<f64>,
    table: &[(f64, f64)],
    beads: usize,
) -> Result<Replicas, SaddleError> {
    if beads < 2 {
        return Err(SaddleError::Invalid(
            "a ring polymer needs at least two replicas".into(),
        ));
    }
    let curve = Curve::new(weighted)?;
    let half = table.last().map_or(0.0, |(_, time)| *time);
    if half <= 0.0 || half.is_nan() {
        return Err(SaddleError::Invalid(
            "the path has no imaginary-time duration".into(),
        ));
    }
    let period = 2.0 * half;
    let mut out = Array2::zeros((beads, weighted.ncols()));
    #[cfg(any(test, feature = "lineb-oracle"))]
    let mut parameter = Vec::with_capacity(beads);
    for bead in 0..beads {
        let time = period * bead as f64 / beads as f64;
        let forward = if time <= half { time } else { period - time };
        let lambda = invert_time(table, forward)?;
        out.row_mut(bead).assign(&curve.at(lambda));
        #[cfg(any(test, feature = "lineb-oracle"))]
        parameter.push(lambda);
    }
    Ok(Replicas {
        weighted: out,
        #[cfg(any(test, feature = "lineb-oracle"))]
        parameter,
        period,
    })
}

fn invert_time(table: &[(f64, f64)], time: f64) -> Result<f64, SaddleError> {
    let last = table.len() - 1;
    if time <= table[0].1 || time.is_nan() {
        return Ok(table[0].0);
    }
    if time >= table[last].1 || time.is_nan() {
        return Ok(table[last].0);
    }
    let mut lo = 0;
    let mut hi = last;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if table[mid].1 <= time || table[mid].1.is_nan() {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let (lambda_left, time_left) = table[lo];
    let (lambda_right, time_right) = table[hi];
    let span = time_right - time_left;
    if span <= 0.0 || span.is_nan() {
        return Err(SaddleError::Invalid(
            "the imaginary-time table stalled between samples".into(),
        ));
    }
    Ok(lambda_left + (lambda_right - lambda_left) * (time - time_left) / span)
}

/// `ℏ / (k_B τ)` for a full period `τ = 2` half periods.
pub(crate) fn temperature(half_period: f64, hbar: f64, boltzmann: f64) -> Result<f64, SaddleError> {
    let period = 2.0 * half_period;
    if !(hbar > 0.0 && boltzmann > 0.0 && period > 0.0) {
        return Err(SaddleError::Invalid(
            "temperature needs a positive period, hbar and Boltzmann constant".into(),
        ));
    }
    let value = hbar / (boltzmann * period);
    if !value.is_finite() {
        return Err(SaddleError::NonFinite("tunneling temperature"));
    }
    Ok(value)
}
