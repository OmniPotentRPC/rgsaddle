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
//! doi:10.1137/22m1487965.

use ndarray::{Array1, ArrayView1};

use crate::error::SaddleError;
use crate::minmode::PointSurface;
use rgmin::linesearch::conditions::armijo;

/// Centre-to-endpoint sample of a dimer of length `h`.
struct DimerSample {
    /// `E_h = (E(x+hv) + E(x-hv)) / 2`.
    energy: f64,
    /// `∇_x E_h = (∇E(x+hv) + ∇E(x-hv)) / 2`.
    gradient: Array1<f64>,
    /// Finite-difference Hessian action `(∇E(x+hv) - ∇E(x-hv)) / (2h)`.
    curvature: f64,
    /// `(I - v v^T)` applied to that action.
    rotation: Array1<f64>,
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
    if !curvature.is_finite() || action.iter().any(|value| !value.is_finite()) {
        return Err(SaddleError::NonFinite("dimer curvature"));
    }
    let rotation = &action - &(&mode.to_owned() * curvature);
    Ok(DimerSample {
        energy,
        gradient,
        curvature,
        rotation,
    })
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
        mode = unit_mode(&(&mode * cos_angle) + &(&(&descent / scale) * sin_angle))?;
        sample = dimer_probe(surface, position, mode.view(), length)?;
        rotations += 1;
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
        ArmijoDimer, DimerState, euler_step, local_merit, rotation_tolerance, shrink_dimer_length,
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
}
