//! One dimer translation with an Armijo merit and a shrinking length.
//!
//! The translation minimises the local merit of Gould, Ortner, and
//! Packwood, Math. Comp. (2016), doi:10.1090/mcom/3096, Algorithm 3.
//! The rotation stops when its residual is no larger than the
//! translation residual, down to their floor `TOL^v`. The dimer
//! length follows `dl/dt = -l` and stays on the order of the square
//! root of the accepted step: Zhang, Zhang, and Zheng, CSIAM Trans.
//! Appl. Math. (2023), doi:10.4208/csiam-am.so-2022-0010, equation
//! (13) and Remark 3.2, with the step-matched dimer error used by
//! Luo, Zheng, Cheng, and Zhang, SIAM J. Numer. Anal. (2022),
//! doi:10.1137/22m1487965. That shrink assumes an exact gradient. A
//! finite-difference curvature whose error, from the force precision,
//! would exceed the curvature is held at the length where the two
//! meet, and a rotation that does not reduce its residual stops.

use ndarray::{Array1, ArrayView1};

use crate::error::SaddleError;
use crate::minmode::PointSurface;
use rgmin::linesearch::conditions::armijo;

/// Centre-to-endpoint sample of a dimer of length `h`.
#[derive(Clone)]
struct DimerSample {
    /// `E_h = (E(x+hv) + E(x-hv)) / 2`.
    energy: f64,
    /// `∇_x E_h = (∇E(x+hv) + ∇E(x-hv)) / 2`.
    gradient: Array1<f64>,
    /// Finite-difference Hessian action `(∇E(x+hv) - ∇E(x-hv)) / (2h)`.
    curvature: f64,
    /// `(I - v v^T)` applied to that action.
    rotation: Array1<f64>,
    /// `(E(x+hv) - E(x-hv)) / (2h)`. Compared with `v · ∇_x E_h` this is
    /// the force error along the mode: an exact gradient makes the two
    /// agree up to the `O(h²)` truncation.
    energy_slope: f64,
}

/// Where the dimer is, and how long it is.
#[derive(Clone, Debug)]
pub struct DimerState {
    /// Centre.
    pub position: Array1<f64>,
    /// Orientation. A step normalises it.
    pub mode: Array1<f64>,
    /// Centre-to-endpoint distance. Positive.
    pub length: f64,
}

/// What one Armijo dimer step did.
#[derive(Clone, Copy, Debug)]
pub struct DimerStep {
    /// `||∇_x E_h||` at the accepted centre. On a step that does not
    /// move, this is the residual at the current centre.
    pub force: f64,
    /// `v · H_h v` at that centre.
    pub curvature: f64,
    /// Rotational force at that centre.
    pub rotation_residual: f64,
    /// Rotation was required only down to this value.
    pub rotation_tolerance: f64,
    /// Accepted translation length. Zero when the centre stays put.
    pub alpha: f64,
    /// Dimer length after the step.
    pub length: f64,
    /// Rotation iterations.
    pub rotations: usize,
    /// Translation residual under the force tolerance, negative curvature,
    /// and a rotation residual inside its tolerance.
    pub converged: bool,
}

/// Armijo backtracking whose slope at zero is known.
///
/// The opening trial is `alpha_max` until a step is accepted, then
/// `min(alpha_max, 2 * alpha)`. A probe returning `None` is refused.
/// The decrease test is [`armijo`] from rgmin. The opening length is
/// the doubled step of Gould, Ortner, and Packwood, Algorithm 3.
#[derive(Clone, Debug)]
struct MeritSearch {
    c: f64,
    shrink: f64,
    maxiter: usize,
    alpha: f64,
    alpha_max: f64,
}

impl MeritSearch {
    fn gould(alpha_max: f64) -> Self {
        Self {
            c: 0.1_f64.sqrt(),
            shrink: 0.5,
            maxiter: 40,
            alpha: 0.0,
            alpha_max,
        }
    }

    fn opening(&self) -> f64 {
        let doubled = if self.alpha > 0.0 {
            2.0 * self.alpha
        } else {
            self.alpha_max
        };
        doubled.min(self.alpha_max)
    }

    fn search<F>(&mut self, phi0: f64, slope: f64, mut probe: F) -> Option<f64>
    where
        F: FnMut(f64) -> Option<f64>,
    {
        if !(phi0.is_finite() && slope.is_finite() && slope < 0.0) {
            return None;
        }
        let mut alpha = self.opening();
        if !(alpha.is_finite() && alpha > 0.0) {
            return None;
        }
        for _ in 0..self.maxiter.max(1) {
            if let Some(phi) = probe(alpha)
                && phi.is_finite()
                && armijo(phi, phi0, alpha, slope, self.c)
            {
                self.alpha = alpha;
                return Some(alpha);
            }
            alpha *= self.shrink;
            if !(alpha.is_finite() && alpha >= 1e-16) {
                break;
            }
        }
        None
    }
}

/// Gould linesearch dimer.
///
/// `step` rotates, then accepts a translation of the local merit, then
/// shortens the dimer.
#[derive(Clone, Debug)]
pub struct ArmijoDimer {
    /// Armijo constant `Θ`. `√0.1` in the numerical section.
    pub theta: f64,
    /// A translation whose rotational force exceeds `Ψ` times the
    /// larger of the current rotational force and the translation
    /// residual is refused.
    pub psi: f64,
    /// Floor `TOL^v` on the rotation tolerance.
    pub tol_v: f64,
    /// Largest translation trial.
    pub alpha_max: f64,
    /// Largest rotation trial.
    pub beta_max: f64,
    /// Stop when `||∇_x E_h||` is at most this and the curvature is negative.
    pub force_tol: f64,
    /// Dimer length is not shortened below this.
    pub length_floor: f64,
    /// Rotation iterations before the translation is taken as it is.
    pub max_rotations: usize,
    /// Shrinks of one linesearch, counting the opening trial.
    pub max_shrinks: usize,
    /// Test hook. The published step keeps a noise-aware length.
    /// The legacy hook is the pure `min(l exp(-α), √α)` shrink, which
    /// is what a noisy force turns into a stalled finite difference.
    #[cfg(test)]
    legacy_length: bool,
    translation: MeritSearch,
    rotation: MeritSearch,
}

impl Default for ArmijoDimer {
    fn default() -> Self {
        Self::gould()
    }
}

impl ArmijoDimer {
    /// Numerical constants of Gould, Ortner, and Packwood, §4.1, and a
    /// length floor of `1e-8` so the shrinking length does not underflow
    /// the force difference.
    pub fn gould() -> Self {
        Self {
            theta: 0.1_f64.sqrt(),
            psi: 100.0,
            tol_v: 1e-1,
            alpha_max: 1.0,
            beta_max: 1.0,
            force_tol: 1e-5,
            length_floor: 1e-8,
            max_rotations: 20,
            max_shrinks: 40,
            #[cfg(test)]
            legacy_length: false,
            translation: MeritSearch::gould(1.0),
            rotation: MeritSearch::gould(1.0),
        }
    }

    pub(crate) fn seed_steps(&mut self, translation_alpha: f64, rotation_beta: f64) {
        if translation_alpha.is_finite() && translation_alpha > 0.0 {
            self.translation.alpha = translation_alpha;
        }
        if rotation_beta.is_finite() && rotation_beta > 0.0 {
            self.rotation.alpha = rotation_beta;
        }
    }

    pub(crate) fn taken_steps(&self) -> (f64, f64) {
        (self.translation.alpha, self.rotation.alpha)
    }

    fn controls(&mut self) {
        self.translation.c = self.theta;
        self.translation.shrink = 0.5;
        self.translation.maxiter = self.max_shrinks;
        self.translation.alpha_max = self.alpha_max;
        self.rotation.c = self.theta;
        self.rotation.shrink = 0.5;
        self.rotation.maxiter = self.max_shrinks;
        self.rotation.alpha_max = self.beta_max;
    }

    /// Rotate to the force-tied tolerance, then take one Armijo translation.
    pub fn step<S: PointSurface>(
        &mut self,
        surface: &S,
        state: &mut DimerState,
    ) -> Result<DimerStep, SaddleError> {
        self.controls();
        if !(state.length.is_finite() && state.length > 0.0) {
            return Err(SaddleError::Invalid(
                "dimer length must be positive and finite".into(),
            ));
        }
        let mode = unit_mode(state.mode.clone())?;
        let (mode, sample, rotations) = align(
            &mut self.rotation,
            surface,
            state.position.view(),
            mode,
            state.length,
            self.tol_v,
            self.max_rotations,
        )?;
        state.mode = mode;
        #[cfg(test)]
        let legacy = self.legacy_length;
        #[cfg(not(test))]
        let legacy = false;
        if !legacy {
            let noise_floor = resolve_dimer_length(
                surface,
                state.position.view(),
                state.mode.view(),
                state.length,
                self.length_floor,
                &sample,
            )?;
            let saved_floor = self.length_floor;
            if noise_floor > saved_floor {
                self.length_floor = noise_floor;
            }
            let moved = if noise_floor > state.length {
                state.length = noise_floor;
                let (mode, sample, rotations) = align(
                    &mut self.rotation,
                    surface,
                    state.position.view(),
                    state.mode.clone(),
                    state.length,
                    self.tol_v,
                    self.max_rotations,
                )?;
                state.mode = mode;
                translate(self, surface, state, &sample, rotations)
            } else {
                translate(self, surface, state, &sample, rotations)
            };
            self.length_floor = saved_floor;
            return moved;
        }
        let moved = translate(self, surface, state, &sample, rotations)?;
        Ok(moved)
    }
}

/// Local merit at a displacement `along = v · (y - x)` from the centre
/// where `parallel` is `v · ∇_x E_h` and `curvature` is `v · H_h v`.
///
/// Gould, Ortner, and Packwood, equation (15):
/// `F(y) = E_h(y) - 2 (v·∇_x E_h) (v·(y-x)) - λ (v·(y-x))²`.
pub fn local_merit(dimer_energy: f64, parallel: f64, curvature: f64, along: f64) -> f64 {
    dimer_energy - 2.0 * parallel * along - curvature * along * along
}

/// Rotation stops once the rotational force is at most this.
///
/// Algorithm 3 of Gould, Ortner, and Packwood (variable-metric form)
/// uses `max(||∇_x E_h||, TOL^v)`. The translation residual is the
/// force argument. `tol_v` is `TOL^v`.
pub fn rotation_tolerance(force_residual: f64, tol_v: f64) -> f64 {
    let force = if force_residual.is_finite() {
        force_residual.abs()
    } else {
        f64::INFINITY
    };
    let floor = if tol_v.is_finite() && tol_v >= 0.0 {
        tol_v
    } else {
        0.0
    };
    force.max(floor)
}

/// Dimer length after a step of duration `step`.
///
/// `l(t) = l exp(-step)` is the solution of `dl/dt = -l`. The accepted
/// linesearch length is that duration. The length is also held to
/// `sqrt(step)`, the matching `l² = O(step)` of Zhang, Zhang, and Zheng
/// (Remark 3.2). It is not shortened below `floor`, and it is not
/// lengthened.
pub fn shrink_dimer_length(length: f64, step: f64, floor: f64) -> f64 {
    if !(length.is_finite() && length > 0.0) {
        return length;
    }
    let tau = step.abs();
    if !(tau.is_finite() && tau > 0.0) {
        return length;
    }
    let decayed = length * (-tau).exp();
    let matched = tau.sqrt();
    let shrunk = decayed.min(matched);
    if floor.is_finite() && floor > 0.0 {
        shrunk.max(floor).min(length)
    } else {
        shrunk.min(length)
    }
}

/// Force scale read off an aligned dimer.
///
/// The rotational residual is `(I - vv) H_h v`. Times the dimer length
/// that is a force difference. Once rotation has stopped, the part that
/// remains is what this length cannot tell from error in the forces.
pub fn force_precision(rotation_residual: f64, length: f64) -> f64 {
    if rotation_residual.is_finite()
        && rotation_residual >= 0.0
        && length.is_finite()
        && length > 0.0
    {
        rotation_residual * length
    } else {
        0.0
    }
}

/// Shortest dimer whose central-difference curvature error stays at most
/// the curvature.
///
/// The action `(∇E(x+hv) - ∇E(x-hv)) / (2h)` moves by about
/// `precision / h` when each force is uncertain by `precision`. Asking
/// that error to stay under `|curvature|` means `h ≥ precision / |curvature|`.
pub fn noise_limited_length(precision: f64, curvature: f64) -> f64 {
    if !(precision.is_finite() && precision > 0.0) {
        return 0.0;
    }
    let scale = curvature.abs();
    if !(scale.is_finite() && scale > 0.0) {
        return 0.0;
    }
    let floor = precision / scale;
    if floor.is_finite() && floor > 0.0 {
        floor
    } else {
        0.0
    }
}

/// Dimer length after one accepted step, kept long enough for the forces.
///
/// The Gould–Zhang shrink still runs. It is not allowed below `floor`,
/// and it is not allowed below [`noise_limited_length`]. A length that
/// is already too short is raised, at most to the larger of the step
/// and `floor`, so a previous `1e-8` shrink can recover.
pub fn robust_dimer_length(
    length: f64,
    step: f64,
    floor: f64,
    force_precision: f64,
    curvature: f64,
) -> f64 {
    let limited = noise_limited_length(force_precision, curvature);
    let absolute = if floor.is_finite() && floor > 0.0 {
        floor
    } else {
        0.0
    };
    let guard = absolute.max(limited);
    let shrunk = shrink_dimer_length(length, step, 0.0);
    let held = if guard > 0.0 {
        shrunk.max(guard)
    } else {
        shrunk
    };
    let cap = step.abs().max(length).max(absolute);
    let cap = if cap.is_finite() && cap > 0.0 {
        cap
    } else {
        length
    };
    if held > cap { cap.max(shrunk) } else { held }
}

fn unit_mode(mode: Array1<f64>) -> Result<Array1<f64>, SaddleError> {
    let norm = mode.dot(&mode).sqrt();
    if !(norm.is_finite() && norm > 1e-14) {
        return Err(SaddleError::Invalid("dimer mode has no direction".into()));
    }
    Ok(mode / norm)
}

fn dimer_probe<S: PointSurface>(
    surface: &S,
    position: ArrayView1<f64>,
    mode: ArrayView1<f64>,
    length: f64,
) -> Result<DimerSample, SaddleError> {
    let plus = &position.to_owned() + &(&mode.to_owned() * length);
    let minus = &position.to_owned() - &(&mode.to_owned() * length);
    let (energy_plus, grad_plus) = surface.eval(plus.view())?;
    let (energy_minus, grad_minus) = surface.eval(minus.view())?;
    if grad_plus.len() != position.len() || grad_minus.len() != position.len() {
        return Err(SaddleError::Shape(
            "dimer endpoint gradient length changed".into(),
        ));
    }
    if !energy_plus.is_finite()
        || !energy_minus.is_finite()
        || grad_plus.iter().any(|value| !value.is_finite())
        || grad_minus.iter().any(|value| !value.is_finite())
    {
        return Err(SaddleError::NonFinite("dimer endpoints"));
    }
    let energy = 0.5 * (energy_plus + energy_minus);
    let gradient = (&grad_plus + &grad_minus) * 0.5;
    let action = (&grad_plus - &grad_minus) / (2.0 * length);
    let curvature = mode.dot(&action);
    let energy_slope = (energy_plus - energy_minus) / (2.0 * length);
    if !curvature.is_finite()
        || !energy_slope.is_finite()
        || action.iter().any(|value| !value.is_finite())
    {
        return Err(SaddleError::NonFinite("dimer curvature"));
    }
    let rotation = &action - &(&mode.to_owned() * curvature);
    Ok(DimerSample {
        energy,
        gradient,
        curvature,
        rotation,
        energy_slope,
    })
}

/// `|v · ∇_x E_h - (E(x+hv) - E(x-hv)) / (2h)|`.
///
/// This is a force. It does not shrink with the dimer length, so it is
/// the precision the finite-difference curvature is divided by.
fn directional_force_error(sample: &DimerSample, mode: ArrayView1<f64>) -> f64 {
    let from_force = sample.gradient.dot(&mode);
    let gap = (from_force - sample.energy_slope).abs();
    if gap.is_finite() && gap > 0.0 {
        gap
    } else {
        0.0
    }
}

fn curvatures_agree(left: f64, right: f64) -> bool {
    let scale = left.abs().max(right.abs());
    scale.is_finite() && scale > 0.0 && (left - right).abs() <= 0.25 * scale
}

/// Floor under the Gould shrink, raised when this length cannot support it.
///
/// An aligned dimer whose curvature clears the force-precision error
/// keeps `precision / |curvature|`. A rotation that missed its
/// tolerance, or a curvature smaller than that error, is not a
/// curvature yet: the length grows until two dimer widths report the
/// same curvature. The caller translates at the raised length and does
/// not shrink back under the floor.
fn resolve_dimer_length<S: PointSurface>(
    surface: &S,
    position: ArrayView1<f64>,
    mode: ArrayView1<f64>,
    length: f64,
    floor: f64,
    sample: &DimerSample,
) -> Result<f64, SaddleError> {
    let host = if floor.is_finite() && floor > 0.0 {
        floor
    } else {
        0.0
    };
    let precision = directional_force_error(sample, mode);
    // A single sample can meet the rotation tolerance with a curvature
    // that is the force error divided by the length. Trust that
    // curvature only when a wider dimer reports the same value.
    // Otherwise the length grows to the width where the two agree.
    // Growing without a ceiling ratchets: each step's cap is a multiple
    // of the length the previous step just adopted. 0.1 is the widest
    // centre-to-endpoint distance the search will introduce on its own.
    // A host floor above that is already a configured dimer length.
    let ceiling = 0.1_f64.max(host);
    let cap = if length < ceiling { ceiling } else { length };
    let mut trial = length;
    let mut previous = sample.curvature;
    for _ in 0..8 {
        if !(cap > trial * (1.0 + 1e-6)) {
            break;
        }
        let next = (trial * 4.0).min(cap);
        if !(next > trial * (1.0 + 1e-6)) {
            break;
        }
        let probed = dimer_probe(surface, position, mode, next)?;
        if curvatures_agree(previous, probed.curvature) {
            let prec = directional_force_error(&probed, mode).max(precision);
            let scale = previous.abs().max(probed.curvature.abs());
            // Agreement is not enough when the force error is still
            // larger than the curvature: both widths are then noise.
            let resolved = scale > 0.0 && prec <= scale * trial;
            if resolved && trial <= length * (1.0 + 1e-9) {
                let limited = noise_limited_length(prec, probed.curvature);
                return Ok(host.max(limited.min(length)));
            }
            if resolved {
                return Ok(host.max(trial));
            }
        }
        previous = probed.curvature;
        trial = next;
    }
    // No width up to the ceiling resolved the curvature. Translate
    // there instead of at a length whose finite difference is noise.
    Ok(host.max(cap))
}

fn translation_direction(gradient: &Array1<f64>, mode: &Array1<f64>) -> Array1<f64> {
    let parallel = gradient.dot(mode);
    -gradient + &(mode * (2.0 * parallel))
}

fn align<S: PointSurface>(
    stepper: &mut MeritSearch,
    surface: &S,
    position: ArrayView1<f64>,
    mut mode: Array1<f64>,
    length: f64,
    tol_v: f64,
    max_rotations: usize,
) -> Result<(Array1<f64>, DimerSample, usize), SaddleError> {
    let mut sample = dimer_probe(surface, position, mode.view(), length)?;
    let mut rotations = 0usize;
    loop {
        let residual = sample.rotation.dot(&sample.rotation).sqrt();
        let force = sample.gradient.dot(&sample.gradient).sqrt();
        let tolerance = rotation_tolerance(force, tol_v);
        if !residual.is_finite() || residual <= tolerance || rotations >= max_rotations {
            return Ok((mode, sample, rotations));
        }
        let scale = residual;
        if scale <= 1e-14 {
            return Ok((mode, sample, rotations));
        }
        // Curvature `C = v·H v` has derivative `-2 ||(I-vv)Hv||²` along
        // the unit-speed parameterisation whose angle is `beta * scale`
        // and whose tangent is the steepest-descent rotational force.
        let slope = -2.0 * scale * scale;
        let curvature0 = sample.curvature;
        let descent = -&sample.rotation;
        let mut failure: Option<SaddleError> = None;
        let accepted = stepper.search(curvature0, slope, |beta| {
            if failure.is_some() {
                return None;
            }
            let angle = beta * scale;
            let (sin_angle, cos_angle) = angle.sin_cos();
            let trial = &(&mode * cos_angle) + &(&(&descent / scale) * sin_angle);
            let trial = match unit_mode(trial) {
                Ok(mode) => mode,
                Err(err) => {
                    failure = Some(err);
                    return None;
                }
            };
            match dimer_probe(surface, position, trial.view(), length) {
                Ok(next) => Some(next.curvature),
                Err(err) => {
                    failure = Some(err);
                    None
                }
            }
        });
        if let Some(err) = failure {
            return Err(err);
        }
        let Some(accepted) = accepted else {
            return Ok((mode, sample, rotations));
        };
        let angle = accepted * scale;
        let (sin_angle, cos_angle) = angle.sin_cos();
        let previous_mode = mode.clone();
        let previous_sample = sample.clone();
        mode = unit_mode(&(&mode * cos_angle) + &(&(&descent / scale) * sin_angle))?;
        sample = dimer_probe(surface, position, mode.view(), length)?;
        rotations += 1;
        let updated = sample.rotation.dot(&sample.rotation).sqrt();
        // The tolerance is a floor on a residual that an exact gradient
        // can drive down. Additive force noise leaves a residual that
        // no angle reduces, and the loop would otherwise run out its
        // rotation budget on every translation.
        if !(updated.is_finite() && updated < residual) {
            mode = previous_mode;
            sample = previous_sample;
            return Ok((mode, sample, rotations));
        }
    }
}

fn report_of(
    sample: &DimerSample,
    tol_v: f64,
    alpha: f64,
    length: f64,
    rotations: usize,
    force_tol: f64,
) -> DimerStep {
    let force = sample.gradient.dot(&sample.gradient).sqrt();
    let rotation_residual = sample.rotation.dot(&sample.rotation).sqrt();
    let rotation_tolerance = rotation_tolerance(force, tol_v);
    let converged =
        force <= force_tol && sample.curvature < 0.0 && rotation_residual <= rotation_tolerance;
    DimerStep {
        force,
        curvature: sample.curvature,
        rotation_residual,
        rotation_tolerance,
        alpha,
        length,
        rotations,
        converged,
    }
}

fn translate<S: PointSurface>(
    dimer: &mut ArmijoDimer,
    surface: &S,
    state: &mut DimerState,
    sample: &DimerSample,
    rotations: usize,
) -> Result<DimerStep, SaddleError> {
    let force = sample.gradient.dot(&sample.gradient).sqrt();
    let rotation_residual = sample.rotation.dot(&sample.rotation).sqrt();
    let tolerance = rotation_tolerance(force, dimer.tol_v);
    if force <= dimer.force_tol && sample.curvature < 0.0 && rotation_residual <= tolerance {
        return Ok(report_of(
            sample,
            dimer.tol_v,
            0.0,
            state.length,
            rotations,
            dimer.force_tol,
        ));
    }
    let direction = translation_direction(&sample.gradient, &state.mode);
    let slope = -direction.dot(&direction);
    if !(slope.is_finite() && slope < 0.0) {
        return Ok(report_of(
            sample,
            dimer.tol_v,
            0.0,
            state.length,
            rotations,
            dimer.force_tol,
        ));
    }
    let parallel = sample.gradient.dot(&state.mode);
    let curvature = sample.curvature;
    let phi0 = sample.energy;
    let orient = dimer.psi * rotation_residual.max(force).max(1e-16);
    let origin = state.position.clone();
    let mode = state.mode.clone();
    let length = state.length;
    let mut failure: Option<SaddleError> = None;
    let accepted = dimer.translation.search(phi0, slope, |alpha| {
        if failure.is_some() {
            return None;
        }
        let trial = &origin + &(&direction * alpha);
        match dimer_probe(surface, trial.view(), mode.view(), length) {
            Ok(next) => {
                let residual = next.rotation.dot(&next.rotation).sqrt();
                if residual > orient {
                    return None;
                }
                let along = mode.dot(&(&trial - &origin));
                Some(local_merit(next.energy, parallel, curvature, along))
            }
            Err(err) => {
                failure = Some(err);
                None
            }
        }
    });
    if let Some(err) = failure {
        return Err(err);
    }
    let Some(accepted) = accepted else {
        return Ok(report_of(
            sample,
            dimer.tol_v,
            0.0,
            state.length,
            rotations,
            dimer.force_tol,
        ));
    };
    state.position = &origin + &(&direction * accepted);
    state.length = shrink_dimer_length(length, accepted, dimer.length_floor);
    let arrived = dimer_probe(
        surface,
        state.position.view(),
        state.mode.view(),
        state.length,
    )?;
    Ok(report_of(
        &arrived,
        dimer.tol_v,
        accepted,
        state.length,
        rotations,
        dimer.force_tol,
    ))
}

/// Forward Euler translation of length `alpha` after the same rotation.
///
/// The centre moves by `alpha * p` with `p = -(I - 2 v v^T) ∇_x E_h`.
/// The dimer length is left as it is. This is the fixed step Algorithm 1
/// takes once the orientation is the one Algorithm 3 would accept.
/// Compiled for the oscillation test. The library step is [`ArmijoDimer::step`].
#[cfg(test)]
fn euler_step<S: PointSurface>(
    dimer: &mut ArmijoDimer,
    surface: &S,
    state: &mut DimerState,
    alpha: f64,
) -> Result<DimerStep, SaddleError> {
    dimer.controls();
    let mode = unit_mode(state.mode.clone())?;
    let (mode, sample, rotations) = align(
        &mut dimer.rotation,
        surface,
        state.position.view(),
        mode,
        state.length,
        dimer.tol_v,
        dimer.max_rotations,
    )?;
    state.mode = mode;
    let direction = translation_direction(&sample.gradient, &state.mode);
    state.position = &state.position + &(&direction * alpha);
    let arrived = dimer_probe(
        surface,
        state.position.view(),
        state.mode.view(),
        state.length,
    )?;
    Ok(report_of(
        &arrived,
        dimer.tol_v,
        alpha,
        state.length,
        rotations,
        dimer.force_tol,
    ))
}

#[cfg(test)]
mod tests {
    use super::{
        ArmijoDimer, DimerState, euler_step, force_precision, local_merit, noise_limited_length,
        robust_dimer_length, rotation_tolerance, shrink_dimer_length,
    };
    use crate::error::SaddleError;
    use crate::minmode::PointSurface;
    use approx::assert_abs_diff_eq;
    use ndarray::{Array1, ArrayView1, array};

    /// `(x² - 1)² + y²`. Minima at `(±1, 0)`, index-1 saddle at the origin.
    /// Gould, Ortner, and Packwood, §4.2.
    struct GouldWell;

    impl PointSurface for GouldWell {
        fn eval(&self, r: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let (x, y) = (r[0], r[1]);
            let energy = (x * x - 1.0).powi(2) + y * y;
            Ok((energy, array![4.0 * x * (x * x - 1.0), 2.0 * y]))
        }
    }

    fn start() -> DimerState {
        DimerState {
            position: array![0.2, 1.0],
            mode: array![1.0, 1.0],
            length: 1e-3,
        }
    }

    #[test]
    fn rotation_tolerance_tracks_the_force_and_the_floor() {
        assert_abs_diff_eq!(rotation_tolerance(2.0, 0.1), 2.0, epsilon = 1e-15);
        assert_abs_diff_eq!(rotation_tolerance(1e-4, 0.1), 0.1, epsilon = 1e-15);
        assert_abs_diff_eq!(rotation_tolerance(-0.4, 0.1), 0.4, epsilon = 1e-15);
    }

    #[test]
    fn merit_at_the_centre_is_the_dimer_energy() {
        assert_abs_diff_eq!(local_merit(1.25, 0.5, -2.0, 0.0), 1.25, epsilon = 1e-15);
        // A pure slide along the mode: the quadratic term is `-λ s²`.
        assert_abs_diff_eq!(
            local_merit(1.25, 0.0, -2.0, 0.5),
            1.25 + 0.5,
            epsilon = 1e-15
        );
    }

    #[test]
    fn dimer_length_shrinks_with_the_step() {
        let decayed = shrink_dimer_length(1e-3, 0.5, 1e-8);
        assert!(decayed < 1e-3);
        assert_abs_diff_eq!(decayed, 1e-3 * (-0.5_f64).exp(), epsilon = 1e-15);
        // `l² = O(step)` binds when the exponential has not caught up.
        let matched = shrink_dimer_length(1.0, 1e-4, 1e-12);
        assert_abs_diff_eq!(matched, 1e-2, epsilon = 1e-15);
        assert_eq!(shrink_dimer_length(1e-3, 0.0, 1e-8), 1e-3);
        // The floor stops the shrink, and a length already under it stays.
        assert_eq!(shrink_dimer_length(1e-3, 40.0, 1e-6), 1e-6);
        assert_eq!(shrink_dimer_length(1e-9, 1.0, 1e-6), 1e-9);
    }

    #[test]
    fn fixed_step_oscillates_and_armijo_converges() {
        let surface = GouldWell;
        let mut fixed = ArmijoDimer::gould();
        fixed.force_tol = 1e-12;
        let mut state = start();
        let mut coordinate = Vec::new();
        for _ in 0..12 {
            euler_step(&mut fixed, &surface, &mut state, 0.5).expect("fixed step");
            coordinate.push(state.position[0]);
        }
        let mut flips = 0;
        for pair in coordinate.windows(2).skip(1) {
            if pair[0] * pair[1] < 0.0 {
                flips += 1;
            }
            assert!(
                pair[1].abs() > 5e-3,
                "fixed step collapsed onto the saddle at {pair:?}"
            );
        }
        assert!(
            flips >= 8,
            "fixed step did not keep changing sign: {coordinate:?}"
        );
        let (_, gradient) = surface.eval(state.position.view()).expect("gradient");
        let fixed_force = gradient.dot(&gradient).sqrt();
        assert!(
            fixed_force > 1e-2,
            "fixed step residual {fixed_force} is already a saddle"
        );

        let mut armijo = ArmijoDimer::gould();
        let mut state = start();
        let mut report = armijo.step(&surface, &mut state).expect("armijo step");
        for _ in 0..15 {
            if report.converged {
                break;
            }
            report = armijo.step(&surface, &mut state).expect("armijo step");
        }
        assert!(report.converged, "Armijo did not converge: {report:?}");
        assert!(state.position[0].abs() < 1e-3, "x = {}", state.position[0]);
        assert!(state.position[1].abs() < 1e-3, "y = {}", state.position[1]);
        assert!(
            state.length < 1e-3,
            "dimer length did not shrink: {}",
            state.length
        );
        assert!(report.curvature < 0.0);
    }

    #[test]
    fn noise_floor_keeps_the_curvature_error_under_the_curvature() {
        assert_abs_diff_eq!(force_precision(0.25, 4e-3), 1e-3, epsilon = 1e-15);
        assert_abs_diff_eq!(noise_limited_length(1e-3, -4.0), 2.5e-4, epsilon = 1e-15);
        let held = robust_dimer_length(1e-3, 0.5, 1e-8, 1e-4, -4.0);
        assert!(
            held < 1e-3,
            "exact-sized noise must still allow a shrink: {held}"
        );
        assert!(
            held + 1e-15 >= 2.5e-5,
            "held {held} is below the noise length"
        );
        // A length the old rule already drove under the noise floor comes back
        // up, and it does not come back further than the step that was taken.
        let raised = robust_dimer_length(1e-8, 0.2, 1e-8, 1e-3, -2.0);
        assert!(
            raised >= 4e-4,
            "tiny dimer was not raised to the noise length: {raised}"
        );
        assert!(
            raised <= 0.2 + 1e-15,
            "raised length {raised} exceeds the step"
        );
    }

    /// Gould well with a force error that changes at every distinct point.
    /// The energy stays exact, so the merit still has the saddle. The
    /// finite-difference curvature does not, once the dimer is shorter
    /// than the force error over the curvature.
    struct NoisyGould {
        sigma: f64,
    }

    fn force_error(x: f64, y: f64, salt: u64) -> f64 {
        let mixed = x.to_bits().wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ y.to_bits().wrapping_mul(0xBF58_476D_1CE4_E5B9)
            ^ salt;
        let mixed = mixed ^ (mixed >> 30);
        let unit = (mixed >> 11) as f64 / ((1u64 << 53) as f64);
        unit * 2.0 - 1.0
    }

    impl PointSurface for NoisyGould {
        fn eval(&self, r: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let (x, y) = (r[0], r[1]);
            let energy = (x * x - 1.0).powi(2) + y * y;
            let gx = 4.0 * x * (x * x - 1.0) + self.sigma * force_error(x, y, 0xA11C);
            let gy = 2.0 * y + self.sigma * force_error(x, y, 0xB0B5);
            Ok((energy, array![gx, gy]))
        }
    }

    fn run_noisy(legacy: bool) -> DimerState {
        // Force error 1e-4. At a dimer length of 1e-6 the curvature
        // error is about 100, and the old shrink keeps going down to
        // the 1e-8 floor. The same forces at a few 1e-3 are a small
        // fraction of the curvature of this well.
        let surface = NoisyGould { sigma: 1e-4 };
        let mut dimer = ArmijoDimer::gould();
        dimer.legacy_length = legacy;
        let mut state = start();
        state.length = 1e-6;
        for _ in 0..40 {
            let report = dimer.step(&surface, &mut state).expect("noisy step");
            if !state.position.iter().all(|value| value.is_finite()) || report.converged {
                break;
            }
        }
        state
    }

    #[test]
    fn additive_force_noise_stalls_the_old_shrink_and_the_floor_converges() {
        let legacy = run_noisy(true);
        let legacy_radius = if legacy.position.iter().all(|value| value.is_finite()) {
            legacy.position.dot(&legacy.position).sqrt()
        } else {
            f64::INFINITY
        };
        assert!(
            legacy.length < 1e-7,
            "old shrink stayed long: {}",
            legacy.length
        );
        assert!(
            legacy_radius > 0.2,
            "old shrink reached the saddle at radius {legacy_radius}"
        );

        let robust = run_noisy(false);
        assert!(
            robust.position.iter().all(|value| value.is_finite()),
            "noise-aware step left the finite plane: {:?}",
            robust.position
        );
        let radius = robust.position.dot(&robust.position).sqrt();
        assert!(
            robust.length > 1e-5,
            "noise-aware length collapsed: {} at radius {radius}",
            robust.length
        );
        assert!(
            radius < 5e-2,
            "noise-aware step stalled at radius {radius}, length {}",
            robust.length
        );
    }
}
