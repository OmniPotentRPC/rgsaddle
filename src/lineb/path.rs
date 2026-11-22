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

use std::collections::HashMap;
use std::sync::Mutex;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use super::action::l2;
#[path = "path_force_clock.rs"]
mod force_integrator;
pub(super) use force_integrator::{action_only, force_clock};

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
    #[cfg(test)]
    pub(crate) fn new(weighted: ArrayView2<f64>) -> Result<Self, SaddleError> {
        Self::with_end_tangents(weighted, None)
    }

    /// The cubic through the images. With `tangents` the two ends leave
    /// along the given unit directions. A particle released at rest at a
    /// turning point moves along the gradient there, so the gradient
    /// direction is the tangent of the orbit at each turn. The not-a-knot
    /// end follows the images alone and can differ from it by several
    /// degrees once the band curves between its first two images.
    pub(crate) fn with_end_tangents(
        weighted: ArrayView2<f64>,
        tangents: Option<&(Vec<f64>, Vec<f64>)>,
    ) -> Result<Self, SaddleError> {
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
        if let Some((start, end)) = tangents {
            if start.len() != dof || end.len() != dof {
                return Err(SaddleError::Shape(
                    "an end tangent needs one component per coordinate".into(),
                ));
            }
        }
        let mut coords = Vec::with_capacity(dof);
        for k in 0..dof {
            let values: Vec<f64> = (0..n).map(|j| weighted[(j, k)]).collect();
            coords.push(match tangents {
                Some((start, end)) => Spline::clamped(&knots, &values, start[k], end[k]),
                None => Spline::new(&knots, &values),
            });
        }
        Ok(Self { knots, coords })
    }

    pub(crate) fn length(&self) -> f64 {
        self.knots[self.knots.len() - 1]
    }

    #[cfg(any(test, feature = "lineb-interpolated-hessian"))]
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

/// Chord coordinate, cumulative imaginary time, and dλ/dt from the path momentum.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TimeKnot {
    lambda: f64,
    time: f64,
    velocity: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct TimeTable {
    knots: Vec<TimeKnot>,
    pub(crate) linear: bool,
    /// Unit tangents `∂q/∂λ` of the spline at its two turning points,
    /// along the surface gradient there. `None` leaves the not-a-knot ends.
    pub(crate) end_tangents: Option<(Vec<f64>, Vec<f64>)>,
}

struct Sample {
    energy: f64,
    gap: f64,
    lambda: f64,
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
        energy,
        gap,
        lambda,
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
    let rate = if u <= 0.0 || u.is_nan() {
        if rise <= 0.0 || rise.is_nan() {
            return Err(SaddleError::Invalid(
                "the potential does not rise as the path enters the barrier".into(),
            ));
        }
        2.0 * sample.speed / (2.0 * rise).sqrt()
    } else if sample.sigma > 0.0 {
        2.0 * u * sample.speed / sample.sigma
    } else {
        return Err(SaddleError::Invalid(format!(
            "the tunneling momentum vanishes before the end of the path at E={}, λ={}, V−E={}, u={}, inward slope={}",
            sample.energy, sample.lambda, sample.gap, u, rise
        )));
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

/// Move each end image onto `V = E` along the local full surface
/// gradient when that end is already within the optimizer's
/// equipotential residual `κ |V − E| ≤ opt_tol`.
/// Ends farther from the turning surface are left alone so
/// [`crate::lineb::optimize::LinebSession::path`] keeps its previous
/// behavior on bands that are not near-turning. Both signs of `V − E`
/// are corrected by the same Newton / bracketed root.
pub(crate) fn refine_near_turning_endpoints<S: PointSurface>(
    surface: &S,
    masses: &Masses,
    mut weighted: Array2<f64>,
    energy: f64,
    restraint: super::action::EndpointRestraint,
    opt_tol: f64,
) -> Result<Array2<f64>, SaddleError> {
    let last = weighted.nrows() - 1;
    for end in [0, last] {
        let refined = refine_endpoint_to_energy(
            surface,
            masses,
            weighted.row(end).to_owned(),
            energy,
            restraint,
            opt_tol,
        )?;
        weighted.row_mut(end).assign(&refined);
    }
    Ok(weighted)
}

fn refine_endpoint_to_energy<S: PointSurface>(
    surface: &S,
    masses: &Masses,
    point: Array1<f64>,
    energy: f64,
    restraint: super::action::EndpointRestraint,
    opt_tol: f64,
) -> Result<Array1<f64>, SaddleError> {
    let (potential, gradient) = eval_endpoint(surface, masses, point.view())?;
    let gap0 = potential - energy;
    let force = masses.weighted_force(gradient.view())?;
    let norm = l2(force.view());
    if norm <= 0.0 || norm.is_nan() {
        return Ok(point);
    }
    // Far from a converged turning point: leave the image for measure.
    if restraint.coefficient(norm) * gap0.abs() > opt_tol {
        return Ok(point);
    }
    if gap0 == 0.0 {
        return Ok(point);
    }

    // Root of V(q₀ + s f̂) − E along the local force direction. At s = 0
    // the directional derivative is −|f|, so the Newton step is
    // s = (V − E) / |f| for either sign of the residual.
    let mut hat = force;
    hat /= norm;
    let mut s = gap0 / norm;
    let mut s_below: Option<(f64, f64)> = None;
    let mut s_above: Option<(f64, f64)> = None;
    note_bracket(&mut s_below, &mut s_above, 0.0, gap0);

    for _ in 0..64 {
        let mut trial = point.clone();
        trial.scaled_add(s, &hat);
        let (trial_v, trial_g) = eval_endpoint(surface, masses, trial.view())?;
        let trial_gap = trial_v - energy;
        if trial_gap == 0.0 {
            return Ok(trial);
        }
        note_bracket(&mut s_below, &mut s_above, s, trial_gap);
        if let (Some((lo, _)), Some((hi, _))) = (s_below, s_above) {
            return root_along_ray(surface, masses, &point, &hat, lo, hi, energy);
        }
        let trial_force = masses.weighted_force(trial_g.view())?;
        // g(s) = V − E, g' = ∇_q V · f̂ = −f · f̂.
        let deriv = -trial_force.dot(&hat);
        if deriv == 0.0 || deriv.is_nan() {
            break;
        }
        let next = s - trial_gap / deriv;
        if next == s || next.is_nan() {
            break;
        }
        s = next;
    }
    // A one-sided Newton sequence can stall short of E; the doubled
    // first step crosses when the directional derivative keeps its sign.
    if s_below.is_none() || s_above.is_none() {
        let probe = 2.0 * (gap0 / norm);
        if probe != 0.0 && !probe.is_nan() {
            let mut trial = point.clone();
            trial.scaled_add(probe, &hat);
            let (trial_v, _) = eval_endpoint(surface, masses, trial.view())?;
            let trial_gap = trial_v - energy;
            if trial_gap == 0.0 {
                return Ok(trial);
            }
            note_bracket(&mut s_below, &mut s_above, probe, trial_gap);
        }
    }
    if let (Some((lo, _)), Some((hi, _))) = (s_below, s_above) {
        return root_along_ray(surface, masses, &point, &hat, lo, hi, energy);
    }
    Ok(point)
}

fn note_bracket(below: &mut Option<(f64, f64)>, above: &mut Option<(f64, f64)>, s: f64, gap: f64) {
    if gap < 0.0 {
        *below = Some((s, gap));
    } else if gap > 0.0 {
        *above = Some((s, gap));
    }
}

fn root_along_ray<S: PointSurface>(
    surface: &S,
    masses: &Masses,
    origin: &Array1<f64>,
    hat: &Array1<f64>,
    mut s_lo: f64,
    mut s_hi: f64,
    energy: f64,
) -> Result<Array1<f64>, SaddleError> {
    let at = |s: f64| -> Result<(Array1<f64>, f64), SaddleError> {
        let mut q = origin.clone();
        q.scaled_add(s, hat);
        let (v, _) = eval_endpoint(surface, masses, q.view())?;
        Ok((q, v - energy))
    };
    let (_, mut gap_lo) = at(s_lo)?;
    let (_, mut gap_hi) = at(s_hi)?;
    for _ in 0..64 {
        let mid = 0.5 * (s_lo + s_hi);
        if mid == s_lo || mid == s_hi || mid.is_nan() {
            if gap_lo == 0.0 {
                return Ok(at(s_lo)?.0);
            }
            if gap_hi == 0.0 {
                return Ok(at(s_hi)?.0);
            }
            return Ok(if gap_hi > 0.0 {
                at(s_hi)?.0
            } else {
                at(s_lo)?.0
            });
        }
        let (q, gap) = at(mid)?;
        if gap == 0.0 {
            return Ok(q);
        }
        if gap < 0.0 {
            s_lo = mid;
            gap_lo = gap;
        } else {
            s_hi = mid;
            gap_hi = gap;
        }
    }
    // The collapsed bracket is one float; prefer the evaluated side on
    // V = E, then the side above E so end-chord samples keep σ > 0.
    if gap_lo == 0.0 {
        Ok(at(s_lo)?.0)
    } else if gap_hi >= 0.0 {
        Ok(at(s_hi)?.0)
    } else {
        Ok(at(s_lo)?.0)
    }
}

fn eval_endpoint<S: PointSurface>(
    surface: &S,
    masses: &Masses,
    weighted: ArrayView1<f64>,
) -> Result<(f64, Array1<f64>), SaddleError> {
    let x = masses.to_cartesian(weighted)?;
    let (potential, gradient) = surface.eval(x.view())?;
    if !potential.is_finite() || gradient.iter().any(|component| !component.is_finite()) {
        return Err(SaddleError::NonFinite("turning endpoint"));
    }
    Ok((potential, gradient))
}

/// Spline action, quadrature half-period, and the replica imaginary-time table.
pub(super) fn quadrature<S: PointSurface>(
    surface: &S,
    masses: &Masses,
    curve: &Curve,
    energy: f64,
    pairs: usize,
) -> Result<(f64, f64, TimeTable), SaddleError> {
    if pairs < 1 {
        return Err(SaddleError::Invalid(
            "the period quadrature needs at least one Simpson pair per chord".into(),
        ));
    }
    // The quadrature nodes depend on the curve alone, apart from the
    // refinement near each turn. A pass over a placeholder surface records
    // the geometries it requests. They are measured as one batch, so a
    // provider may evaluate them concurrently. The integration then reads
    // those measurements and evaluates any other geometry on the surface.
    let recorder = Recorder {
        points: Mutex::new(Vec::new()),
        energy,
    };
    let _ = integrate(
        &Level {
            surface: &recorder,
            masses,
            curve,
            energy,
        },
        pairs,
    );
    let measured = Measured::new(surface, recorder.points.into_inner().unwrap_or_default())?;
    integrate(
        &Level {
            surface: &measured,
            masses,
            curve,
            energy,
        },
        pairs,
    )
}

fn geometry_key(x: ArrayView1<f64>) -> Vec<u64> {
    x.iter().map(|value| value.to_bits()).collect()
}

/// Records each distinct requested geometry and answers with a constant
/// barrier above the tunneling energy.
struct Recorder {
    points: Mutex<Vec<Array1<f64>>>,
    energy: f64,
}

impl PointSurface for Recorder {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let mut points = self
            .points
            .lock()
            .map_err(|_| SaddleError::Invalid("quadrature recorder poisoned".into()))?;
        if !points.iter().any(|seen| seen.view() == x) {
            points.push(x.to_owned());
        }
        Ok((self.energy + 1.0, Array1::from_elem(x.len(), 1e-3)))
    }
}

/// Batch measurements at recorded geometries, with the surface behind them.
struct Measured<'a, S> {
    surface: &'a S,
    known: HashMap<Vec<u64>, (f64, Array1<f64>)>,
}

impl<'a, S: PointSurface> Measured<'a, S> {
    fn new(surface: &'a S, points: Vec<Array1<f64>>) -> Result<Self, SaddleError> {
        let mut known = HashMap::with_capacity(points.len());
        if let Some(first) = points.first() {
            let mut rows = Array2::zeros((points.len(), first.len()));
            for (mut row, point) in rows.rows_mut().into_iter().zip(&points) {
                row.assign(point);
            }
            let observations = surface.eval_batch(rows.view())?;
            if observations.len() != points.len() {
                return Err(SaddleError::Shape(
                    "the surface must return one observation per quadrature geometry".into(),
                ));
            }
            for (point, observation) in points.iter().zip(observations) {
                known.insert(geometry_key(point.view()), observation);
            }
        }
        Ok(Self { surface, known })
    }
}

impl<S: PointSurface> PointSurface for Measured<'_, S> {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        match self.known.get(&geometry_key(x)) {
            Some(observation) => Ok(observation.clone()),
            None => self.surface.eval(x),
        }
    }

    fn eval_batch(&self, points: ArrayView2<f64>) -> Result<Vec<(f64, Array1<f64>)>, SaddleError> {
        points
            .rows()
            .into_iter()
            .map(|row| self.eval(row))
            .collect()
    }

    fn hessian_vector(
        &self,
        x: ArrayView1<f64>,
        v: ArrayView1<f64>,
    ) -> Result<Option<Array1<f64>>, SaddleError> {
        self.surface.hessian_vector(x, v)
    }
}

/// Action, half period and the `(λ, t)` time table from one pass of
/// samples. Interior chords use Simpson's duration and split it between
/// adjacent samples in proportion to their trapezoidal durations.
fn integrate<S: PointSurface>(
    level: &Level<'_, S>,
    pairs: usize,
) -> Result<(f64, f64, TimeTable), SaddleError> {
    let mut action = 0.0;
    let mut half = 0.0;
    let mut table = vec![TimeKnot {
        lambda: 0.,
        time: 0.,
        velocity: 0.,
    }];
    let mut clock = 0.0;
    let knots = &level.curve.knots;
    let last = knots.len() - 1;
    for j in 0..last {
        let (left, right) = (knots[j], knots[j + 1]);
        if j == 0 || j + 1 == last {
            let chord = EndChord::new(left, right, j == 0)?;
            action += end_action(level, chord, pairs)?;
            let samples = end_clock(level, chord, pairs)?;
            let gained = samples.last().map(|knot| knot.time).ok_or_else(|| {
                SaddleError::Invalid("the end chord produced no period samples".into())
            })?;
            for mut knot in samples.into_iter().skip(1) {
                knot.time += clock;
                table.push(knot);
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
    // A Hermite interval is monotone when both endpoint slopes lie in
    // [0, 3 Δλ/Δt]. Limit each shared knot against both adjacent intervals
    // to retain a continuous velocity. The orbit stands still at each turn.
    for i in 0..table.len() - 1 {
        let dt = table[i + 1].time - table[i].time;
        let dl = table[i + 1].lambda - table[i].lambda;
        if dt > 0. && dl >= 0. {
            let bound = 3. * dl / dt;
            table[i].velocity = table[i].velocity.min(bound);
            table[i + 1].velocity = table[i + 1].velocity.min(bound);
        }
    }
    table
        .last_mut()
        .expect("the clock contains its origin")
        .velocity = 0.;
    Ok((
        action,
        half,
        TimeTable {
            knots: table,
            linear: false,
            end_tangents: None,
        },
    ))
}

fn open_integral<S: PointSurface>(
    level: &Level<'_, S>,
    left: f64,
    right: f64,
    pairs: usize,
    table: &mut Vec<TimeKnot>,
    clock: &mut f64,
) -> Result<(f64, f64), SaddleError> {
    let intervals = 2 * pairs;
    let h = (right - left) / intervals as f64;
    let mut action = 0.0;
    let mut half = 0.0;
    let mut rates = [0.0; 3];
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
        if i == 0 {
            rates[0] = rate;
        } else if i % 2 == 1 {
            rates[1] = rate;
        } else {
            rates[2] = rate;
            let [a, b, c] = rates;
            let duration = (a + 4.0 * b + c) * h / 3.0;
            let midpoint = *clock + duration * (a + b) / (a + 2.0 * b + c);
            table.push(TimeKnot {
                lambda: lambda - h,
                time: midpoint,
                velocity: 1. / b,
            });
            *clock += duration;
            table.push(TimeKnot {
                lambda,
                time: *clock,
                velocity: 1. / c,
            });
            rates[0] = c;
        }
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
) -> Result<Vec<TimeKnot>, SaddleError> {
    let u_max = chord.u_max;
    let panels = (2 * pairs) as f64;
    let panel = u_max / panels;
    let tol = (chord.right - chord.left) / panels.powi(4);
    let mut s = 0.0;
    let mut period = 0.0;
    let mut out = vec![TimeKnot {
        lambda: chord.lambda(0.),
        time: 0.,
        velocity: 0.,
    }];
    while u_max > s {
        let mut du = panel.min(u_max - s);
        loop {
            if du <= 0.0 || du.is_nan() {
                return Err(SaddleError::Stalled(
                    "the end-chord period step underflowed".into(),
                ));
            }
            let (coarse, coarse_rounding, _) = simpson_s(level, chord, s, s + du)?;
            let mid = s + 0.5 * du;
            let (left, left_rounding, _) = simpson_s(level, chord, s, mid)?;
            let (right, right_rounding, velocity) = simpson_s(level, chord, mid, s + du)?;
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
                out.push(TimeKnot {
                    lambda: chord.lambda(s),
                    time: period,
                    velocity,
                });
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

pub(super) fn energy_roundoff(energy: f64) -> f64 {
    ENERGY_ULPS * f64::EPSILON * energy.abs()
}

/// Simpson on `[s0, s1]`, and the bound on its change from an error of
/// `ENERGY_ULPS · ε · |E|` in every sampled `V`.
fn simpson_s<S: PointSurface>(
    level: &Level<'_, S>,
    chord: EndChord,
    s0: f64,
    s1: f64,
) -> Result<(f64, f64, f64), SaddleError> {
    let delta_v = energy_roundoff(level.energy);
    let mid = 0.5 * (s0 + s1);
    let mut acc = 0.0;
    let mut rounding = 0.0;
    let mut velocity = 0.;
    for (s, weight) in [(s0, 1.0), (mid, 4.0), (s1, 1.0)] {
        let taken = level.sample(chord.lambda(s))?;
        // The spline coordinate can round to the turning knot even when
        // repeated panel additions leave a positive residual in u.
        let turn = if chord.from_start {
            chord.left
        } else {
            chord.right
        };
        let u = if taken.lambda == turn {
            0.0
        } else {
            chord.u(s)
        };
        velocity = taken.sigma / taken.speed;
        let rate = end_rate(&taken, u, chord.from_start)?;
        acc += weight * rate;
        if taken.sigma > 0.0 {
            rounding += weight * rate * delta_v / (taken.sigma * taken.sigma);
        }
    }
    Ok((acc * (s1 - s0) / 6.0, rounding * (s1 - s0) / 6.0, velocity))
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
pub(super) fn dynamics<S: PointSurface>(
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
    #[cfg(any(test, feature = "lineb-interpolated-hessian"))]
    pub parameter: Vec<f64>,
    pub period: f64,
}

/// Replicas at `t_j = j τ / P`. The return half retraces the spline.
/// The period is twice the last time in `table`, the time table that
/// [`quadrature`] built on the same images.
pub(crate) fn replicas(
    weighted: ArrayView2<f64>,
    clock: &TimeTable,
    beads: usize,
) -> Result<Replicas, SaddleError> {
    if beads < 2 {
        return Err(SaddleError::Invalid(
            "a ring polymer needs at least two replicas".into(),
        ));
    }
    let curve = Curve::with_end_tangents(weighted, clock.end_tangents.as_ref())?;
    let table = &clock.knots;
    let half = table.last().map_or(0.0, |knot| knot.time);
    if half <= 0.0 || half.is_nan() {
        return Err(SaddleError::Invalid(
            "the path has no imaginary-time duration".into(),
        ));
    }
    let period = 2.0 * half;
    let mut out = Array2::zeros((beads, weighted.ncols()));
    #[cfg(any(test, feature = "lineb-interpolated-hessian"))]
    let mut parameter = Vec::with_capacity(beads);
    for bead in 0..beads {
        // Mirror the integer phase so both legs present identical geometry
        // to a potential that caches energy and gradient by coordinate.
        let outbound = bead.min(beads - bead);
        let forward = period * outbound as f64 / beads as f64;
        let lambda = invert_time(table, forward, clock.linear)?;
        out.row_mut(bead).assign(&curve.at(lambda));
        #[cfg(any(test, feature = "lineb-interpolated-hessian"))]
        parameter.push(lambda);
    }
    Ok(Replicas {
        weighted: out,
        #[cfg(any(test, feature = "lineb-interpolated-hessian"))]
        parameter,
        period,
    })
}

fn invert_time(table: &[TimeKnot], time: f64, linear: bool) -> Result<f64, SaddleError> {
    let last = table.len() - 1;
    if time <= table[0].time || time.is_nan() {
        return Ok(table[0].lambda);
    }
    if time >= table[last].time || time.is_nan() {
        return Ok(table[last].lambda);
    }
    let mut lo = 0;
    let mut hi = last;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if table[mid].time <= time || table[mid].time.is_nan() {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let left = table[lo];
    let right = table[hi];
    let span = right.time - left.time;
    if span <= 0.0 || span.is_nan() {
        return Err(SaddleError::Invalid(
            "the imaginary-time table stalled between samples".into(),
        ));
    }
    if linear {
        return Ok(left.lambda + (right.lambda - left.lambda) * (time - left.time) / span);
    }
    let t = (time - left.time) / span;
    let u = 1. - t;
    Ok(left.lambda
        + t * t * (3. - 2. * t) * (right.lambda - left.lambda)
        + span * t * u * (u * left.velocity - t * right.velocity))
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    struct InvertedOscillator;

    impl PointSurface for InvertedOscillator {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            Ok(((1. - x[0] * x[0]) * 0.5, array![-x[0]]))
        }
    }

    #[test]
    fn reflected_replicas_reuse_the_outbound_geometry() {
        let images = array![[-1.], [0.], [1.]];
        let masses = Masses::coordinates(array![1.].view()).unwrap();
        let curve = Curve::new(images.view()).unwrap();
        let (_, _, table) = quadrature(&InvertedOscillator, &masses, &curve, 0., 4).unwrap();
        for count in [17, 240] {
            let ring = replicas(images.view(), &table, count).unwrap();
            for j in 1..count {
                assert_eq!(ring.weighted.row(j), ring.weighted.row(count - j));
                assert_eq!(ring.parameter[j], ring.parameter[count - j]);
            }
        }
    }

    #[test]
    fn reflected_harmonic_orbit_has_the_physical_turning_point_acceleration() {
        let images = array![[-1.], [0.], [1.]];
        let masses = Masses::coordinates(array![1.].view()).unwrap();
        let curve = Curve::new(images.view()).unwrap();
        let (_, _, table) = quadrature(&InvertedOscillator, &masses, &curve, 0., 4).unwrap();
        for beads in [120, 480] {
            let ring = replicas(images.view(), &table, beads).unwrap();
            let dt = ring.period / beads as f64;
            for j in [0, beads / 2] {
                let x = ring.weighted[(j, 0)];
                let previous = ring.weighted[((j + beads - 1) % beads, 0)];
                let next = ring.weighted[((j + 1) % beads, 0)];
                let acceleration = (previous - x + next - x) / dt.powi(2);
                assert!(
                    (acceleration + x).abs() < 0.01,
                    "P={beads}, x={x}, acceleration={acceleration}"
                );
            }
        }
    }
}
