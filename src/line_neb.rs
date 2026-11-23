//! Fixed-energy line-integral NEB (LI-NEB).
//!
//! The band is the nudged elastic band as practised in Goswami,
//! arXiv:2603.14737 (2026), doi:10.48550/arXiv.2603.14737, and
//! Goswami, Gunde, and Jónsson, Front. Chem. 14, 1807063 (2026),
//! doi:10.3389/fchem.2026.1807063, with the improved tangent of
//! Henkelman and Jónsson, J. Chem. Phys. 113, 9978 (2000),
//! doi:10.1063/1.1323224. Its objective is the line-integral extension
//! of Ásgeirsson, Arnaldsson, and Jónsson, J. Chem. Phys. 148, 102334
//! (2018), doi:10.1063/1.5007180: the optimal JWKB tunnelling path at a
//! fixed energy, which minimizes the abbreviated action between two
//! turning points. The optimizer is the global quick-min of eOn,
//! Goswami et al., eOn v2.9.0.1 (2026), doi:10.5281/zenodo.18529496.
//!
//! Coordinates are converted to `q_i = sqrt(m_i) x_i`; the abbreviated
//! action and all projections are evaluated in that Euclidean,
//! mass-scaled chart. [`fixed_energy_action`] is the trapezoidal
//! action over every image. The interior images move under the
//! transverse action force and a tangential spring; each endpoint moves
//! under the spring perpendicular to its force and a restraint that
//! pulls its energy to `E`. [`LineNebStatus::Converged`] needs all three
//! within the force tolerance.
//!
//! [`fixed_energy_mep_seed`] builds a starting band from an index-one
//! saddle and its unstable mode, for instance from a climbing-image band
//! on an IDPP path (Goswami, MethodsX (2026),
//! doi:10.1016/j.mex.2026.103899). [`crate::FixedEnergyTrajectory`]
//! integrates the motion along a converged path, the dynamics
//! Ásgeirsson, Arnaldsson, and Jónsson use to find the temperature at
//! which the path is an instanton.
//!
//! Units are deliberately not prescribed. Positions, potential
//! energies, masses, the reduced Planck constant, spring constant,
//! endpoint restraint, and time step must form one consistent unit
//! system. In particular, `reduced_planck` has units of
//! `energy * time`, `spring_k` maps a mass-scaled displacement to an
//! action-gradient force, and `endpoint_kappa * energy` has those same
//! force units.
//!
//! A session performs one host-visible step at a time. It has no run
//! loop and no force budget; the host decides when to stop calling
//! [`LineNebSession::step`].

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rgmin::{Control, Method, Oracle, Solver};

use crate::{BandSurface, SaddleError};

/// Policy for an LI-NEB session endpoint below the fixed energy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum EndpointBelowEnergy {
    /// Give the endpoint zero action weight while the endpoint restraint
    /// returns it to the fixed energy, as in the published LI-NEB scheme.
    #[default]
    Restrained,
    /// Stop the session with [`SaddleError::Invalid`].
    ///
    /// [`LineNebEndpointViolation`] carries the endpoint, `V - E`, and the
    /// one-based step. [`SaddleError`] gains no variant.
    Strict,
}

/// A strict LI-NEB session found an endpoint below the fixed energy.
///
/// [`LineNebSession::step`] returns this as [`SaddleError::Invalid`]. The
/// message names the endpoint (`first` or `last`), the signed difference
/// `V - E`, and the one-based step. [`LineNebSession::endpoint_violation`]
/// and [`Self::from_error`] read the same record. [`SaddleError`] is
/// unchanged.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct LineNebEndpointViolation {
    /// Endpoint name, either `first` or `last`.
    pub endpoint: &'static str,
    /// Endpoint potential energy minus the fixed energy, `V - E`.
    pub energy_difference: f64,
    /// One-based session step that detected the endpoint.
    pub step: usize,
}

impl LineNebEndpointViolation {
    /// The [`SaddleError::Invalid`] value a strict session returns.
    #[must_use]
    pub fn to_error(self) -> SaddleError {
        SaddleError::Invalid(self.message())
    }

    /// The violation when `err` is the strict-endpoint invalid parameter.
    #[must_use]
    pub fn from_error(err: &SaddleError) -> Option<Self> {
        let SaddleError::Invalid(message) = err else {
            return None;
        };
        parse_endpoint_violation(message)
    }

    fn message(self) -> String {
        format!(
            "LI-NEB {} endpoint lies below the fixed energy at step {}: V - E = {}",
            self.endpoint, self.step, self.energy_difference
        )
    }
}

fn parse_endpoint_violation(message: &str) -> Option<LineNebEndpointViolation> {
    let rest = message.strip_prefix("LI-NEB ")?;
    let (endpoint, rest) = if let Some(rest) =
        rest.strip_prefix("first endpoint lies below the fixed energy at step ")
    {
        ("first", rest)
    } else {
        (
            "last",
            rest.strip_prefix("last endpoint lies below the fixed energy at step ")?,
        )
    };
    let (step_text, difference_text) = rest.split_once(": V - E = ")?;
    let step: usize = step_text.parse().ok()?;
    if step_text != step.to_string() {
        return None;
    }
    let energy_difference: f64 = difference_text.parse().ok()?;
    if !energy_difference.is_finite() || difference_text != format!("{energy_difference}") {
        return None;
    }
    Some(LineNebEndpointViolation {
        endpoint,
        energy_difference,
        step,
    })
}

/// All dimensional controls for a fixed-energy LI-NEB session.
///
/// Start from [`LineNebConfig::default`] and chain the `with_*` setters.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct LineNebConfig {
    /// Uniform spring constant in the mass-scaled chart.
    pub spring_k: f64,
    /// Coefficient of the endpoint energy restraint.
    pub endpoint_kappa: f64,
    /// Convergence tolerance applied to force components.
    pub force_tol: f64,
    /// Fixed tunnelling energy `E`.
    pub energy: f64,
    /// One positive mass per physical coordinate (a 3N host repeats
    /// each atomic mass three times). `None` is unit mass on every
    /// coordinate.
    pub masses: Option<Array1<f64>>,
    /// Fixed time step of the eOn global quick-min variant.
    pub time_step: f64,
    /// Reduced Planck constant in the host's unit system.
    pub reduced_planck: f64,
    /// Behaviour when a running session carries an endpoint below `energy`.
    pub endpoint_below_energy: EndpointBelowEnergy,
}

impl Default for LineNebConfig {
    /// Unit spring constant and endpoint restraint, force tolerance
    /// 1e-3, fixed energy 0, unit masses, time step 0.1, unit reduced
    /// Planck constant, and [`EndpointBelowEnergy::Restrained`]. These
    /// are values in units where `hbar = 1`; a host in other units sets
    /// every field it relies on.
    fn default() -> Self {
        Self {
            spring_k: 1.0,
            endpoint_kappa: 1.0,
            force_tol: 1.0e-3,
            energy: 0.0,
            masses: None,
            time_step: 0.1,
            reduced_planck: 1.0,
            endpoint_below_energy: EndpointBelowEnergy::default(),
        }
    }
}

with_setters!(LineNebConfig {
    with_spring_k => spring_k: val f64,
    with_endpoint_kappa => endpoint_kappa: val f64,
    with_force_tol => force_tol: val f64,
    with_energy => energy: val f64,
    with_masses => masses: opt Array1<f64>,
    with_time_step => time_step: val f64,
    with_reduced_planck => reduced_planck: val f64,
    with_endpoint_below_energy => endpoint_below_energy: val EndpointBelowEnergy,
});

/// State of a host-stepped LI-NEB optimization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LineNebStatus {
    /// The force tolerance has not been met.
    Running,
    /// The interior perpendicular action force, the interior spring
    /// force, and the endpoint force are all within `force_tol`.
    Converged,
}

/// Diagnostics from the configuration evaluated after one step.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct LineNebReport {
    /// Whether this step met the force tolerance.
    pub status: LineNebStatus,
    /// Largest absolute component of the transverse action force over
    /// the interior images.
    pub max_interior_perpendicular: f64,
    /// Largest absolute component of the tangential spring force over
    /// the interior images. With one coordinate per image the
    /// transverse force vanishes identically and this is the whole
    /// interior force.
    pub max_interior_spring: f64,
    /// Largest absolute component of the complete optimized endpoint
    /// force (projected spring plus energy restraint).
    pub max_endpoint_optimized: f64,
    /// Abbreviated action of the configuration after the step. This is
    /// [`fixed_energy_action`] when both endpoints are at or above the
    /// fixed energy. Under [`EndpointBelowEnergy::Restrained`], an
    /// endpoint the optimization has carried below it has zero weight,
    /// as at a turning point, until the restraint brings it back.
    pub action: f64,
    /// Steps completed, counting this one.
    pub iteration: usize,
    /// Number of image rows passed to [`BandSurface::eval`] this step.
    pub surface_rows: usize,
}

#[derive(Clone, Debug)]
struct Evaluation {
    action: f64,
    optimization_force: Array2<f64>,
    max_interior_perpendicular: f64,
    max_interior_spring: f64,
    max_endpoint_optimized: f64,
}

fn validate_inputs(
    positions: ArrayView2<'_, f64>,
    energies: ArrayView1<'_, f64>,
    masses: ArrayView1<'_, f64>,
    energy: f64,
    reduced_planck: f64,
) -> Result<(), SaddleError> {
    if positions.nrows() < 3 || positions.ncols() == 0 {
        return Err(SaddleError::Shape(format!(
            "LI-NEB needs at least three nonempty images; got {} x {}",
            positions.nrows(),
            positions.ncols()
        )));
    }
    if energies.len() != positions.nrows() {
        return Err(SaddleError::Shape(
            "LI-NEB energies must have one value per image".into(),
        ));
    }
    if masses.len() != positions.ncols() {
        return Err(SaddleError::Shape(
            "LI-NEB masses must have one value per coordinate".into(),
        ));
    }
    if !positions.iter().all(|v| v.is_finite()) {
        return Err(SaddleError::NonFinite("LI-NEB positions"));
    }
    if !energies.iter().all(|v| v.is_finite()) {
        return Err(SaddleError::NonFinite("LI-NEB energies"));
    }
    if !masses.iter().all(|m| m.is_finite() && *m > 0.0) {
        return Err(SaddleError::Invalid(
            "LI-NEB masses must be positive and finite".into(),
        ));
    }
    if !energy.is_finite() {
        return Err(SaddleError::NonFinite("LI-NEB target energy"));
    }
    if !(reduced_planck.is_finite() && reduced_planck > 0.0) {
        return Err(SaddleError::Invalid(
            "LI-NEB reduced Planck constant must be positive and finite".into(),
        ));
    }
    Ok(())
}

fn mass_scaled(positions: ArrayView2<'_, f64>, masses: ArrayView1<'_, f64>) -> Array2<f64> {
    let sqrt_masses = masses.mapv(f64::sqrt);
    &positions * &sqrt_masses
}

/// How an endpoint below the fixed energy enters the action weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BelowEndpoint {
    /// Refuse it: the action is defined between turning points.
    Refuse,
    /// Give it zero weight, as at a turning point. A session uses this
    /// while the restraint is still pulling an endpoint back to `E`.
    Restrained,
    /// Stop a strict session and identify where the violation occurred.
    Strict { step: usize },
}

/// `xi_i = sqrt(2 (E_i - E)) / hbar` on every image, endpoints included.
fn action_weights(
    energies: ArrayView1<'_, f64>,
    energy: f64,
    reduced_planck: f64,
    below: BelowEndpoint,
) -> Result<Array1<f64>, SaddleError> {
    let last = energies.len() - 1;
    let mut xi = Array1::zeros(energies.len());
    for i in 0..=last {
        let excess = energies[i] - energy;
        if i == 0 || i == last {
            if excess < 0.0 {
                match below {
                    BelowEndpoint::Refuse => {
                        return Err(SaddleError::Invalid(format!(
                            "LI-NEB endpoint {i} lies below the fixed energy"
                        )));
                    }
                    BelowEndpoint::Restrained => continue,
                    BelowEndpoint::Strict { step } => {
                        return Err(LineNebEndpointViolation {
                            endpoint: if i == 0 { "first" } else { "last" },
                            energy_difference: excess,
                            step,
                        }
                        .to_error());
                    }
                }
            }
        } else if excess.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
            return Err(SaddleError::Invalid(format!(
                "LI-NEB interior image {i} must lie above the fixed energy"
            )));
        }
        xi[i] = (2.0 * excess).sqrt() / reduced_planck;
    }
    Ok(xi)
}

fn segments(q: ArrayView2<'_, f64>) -> Result<(Vec<Array1<f64>>, Array1<f64>), SaddleError> {
    let mut unit = Vec::with_capacity(q.nrows() - 1);
    let mut lengths = Array1::zeros(q.nrows() - 1);
    for i in 0..q.nrows() - 1 {
        let displacement = &q.row(i + 1) - &q.row(i);
        let length = displacement.dot(&displacement).sqrt();
        if !(length.is_finite() && length > 0.0) {
            return Err(SaddleError::Invalid(format!(
                "LI-NEB segment {i} must have positive finite length"
            )));
        }
        unit.push(displacement / length);
        lengths[i] = length;
    }
    Ok((unit, lengths))
}

/// Trapezoidal discretization of the fixed-energy abbreviated action.
///
/// With `q_i = sqrt(m) x_i` and `xi_i = sqrt(2 (E_i - E)) / hbar` on
/// every image, both endpoints included, the action is
/// `sum_i (xi_i + xi_{i+1}) |q_{i+1} - q_i| / 2`.
///
/// `positions` are physical coordinates and `masses` has one entry per
/// coordinate. An endpoint energy may equal `energy`, where it is a
/// classical turning point with zero weight, or lie above it. Every
/// interior energy must be strictly larger.
///
/// # Errors
///
/// [`SaddleError::Invalid`] when an endpoint lies below `energy`, when
/// an interior image does not lie above it, when a segment has no
/// length, or when a mass or `reduced_planck` is not positive.
/// [`SaddleError::Shape`] and [`SaddleError::NonFinite`] for
/// inconsistent or non-finite input.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use rgsaddle::fixed_energy_action;
/// let positions = array![[0.0], [1.0], [2.0]];
/// let masses = array![1.0];
/// let action = fixed_energy_action(
///     positions.view(),
///     array![4.0, 2.0, 4.0].view(),
///     masses.view(),
///     1.0,
///     1.0,
/// )
/// .expect("every image is at or above the fixed energy");
/// assert_eq!(action, 6.0_f64.sqrt() + 2.0_f64.sqrt());
/// ```
pub fn fixed_energy_action(
    positions: ArrayView2<'_, f64>,
    energies: ArrayView1<'_, f64>,
    masses: ArrayView1<'_, f64>,
    energy: f64,
    reduced_planck: f64,
) -> Result<f64, SaddleError> {
    validate_inputs(positions, energies, masses, energy, reduced_planck)?;
    let xi = action_weights(energies, energy, reduced_planck, BelowEndpoint::Refuse)?;
    trapezoid_action(positions, masses, &xi)
}

fn trapezoid_action(
    positions: ArrayView2<'_, f64>,
    masses: ArrayView1<'_, f64>,
    xi: &Array1<f64>,
) -> Result<f64, SaddleError> {
    let q = mass_scaled(positions, masses);
    let (_, lengths) = segments(q.view())?;
    let mut action = 0.0;
    for i in 0..lengths.len() {
        action += 0.5 * (xi[i] + xi[i + 1]) * lengths[i];
    }
    if !action.is_finite() {
        return Err(SaddleError::NonFinite("LI-NEB abbreviated action"));
    }
    Ok(action)
}

fn action_force_q(
    q: ArrayView2<'_, f64>,
    xi: &Array1<f64>,
    force_q: ArrayView2<'_, f64>,
    reduced_planck: f64,
) -> Result<Array2<f64>, SaddleError> {
    if force_q.raw_dim() != q.raw_dim() {
        return Err(SaddleError::Shape(
            "LI-NEB force shape must match positions".into(),
        ));
    }
    if !force_q.iter().all(|v| v.is_finite()) {
        return Err(SaddleError::NonFinite("LI-NEB physical force"));
    }
    let (unit, lengths) = segments(q)?;
    let mut force = Array2::zeros(q.raw_dim());
    for j in 1..q.nrows() - 1 {
        let physical = force_q.row(j).to_owned()
            * ((lengths[j - 1] + lengths[j]) / (xi[j] * reduced_planck * reduced_planck));
        let curvature = &unit[j] * (xi[j + 1] + xi[j]) - &unit[j - 1] * (xi[j] + xi[j - 1]);
        force.row_mut(j).assign(&(0.5 * (physical + curvature)));
    }
    if !force.iter().all(|v| v.is_finite()) {
        return Err(SaddleError::NonFinite("LI-NEB action force"));
    }
    Ok(force)
}

/// Negative gradient of [`fixed_energy_action`] with respect to the
/// physical coordinates.
///
/// The endpoint rows are zero because LI-NEB replaces their action
/// gradient by the endpoint spring and iso-energy restraint. The
/// endpoint weights still enter the interior rows through the two end
/// segments.
///
/// # Errors
///
/// The errors of [`fixed_energy_action`], and [`SaddleError::Shape`] or
/// [`SaddleError::NonFinite`] when `gradients` does not match.
pub fn fixed_energy_action_force(
    positions: ArrayView2<'_, f64>,
    energies: ArrayView1<'_, f64>,
    gradients: ArrayView2<'_, f64>,
    masses: ArrayView1<'_, f64>,
    energy: f64,
    reduced_planck: f64,
) -> Result<Array2<f64>, SaddleError> {
    validate_inputs(positions, energies, masses, energy, reduced_planck)?;
    if gradients.raw_dim() != positions.raw_dim() {
        return Err(SaddleError::Shape(
            "LI-NEB gradient shape must match positions".into(),
        ));
    }
    if !gradients.iter().all(|v| v.is_finite()) {
        return Err(SaddleError::NonFinite("LI-NEB gradients"));
    }
    let sqrt_masses = masses.mapv(f64::sqrt);
    let q = &positions * &sqrt_masses;
    let force_q = -&gradients / &sqrt_masses;
    let xi = action_weights(energies, energy, reduced_planck, BelowEndpoint::Refuse)?;
    let force = action_force_q(q.view(), &xi, force_q.view(), reduced_planck)?;
    Ok(force * &sqrt_masses)
}

fn store<T>(mutex: &Mutex<T>, value: T) -> bool {
    match mutex.lock() {
        Ok(mut guard) => {
            *guard = value;
            true
        }
        Err(_) => false,
    }
}

/// Largest magnitude of `values`; NaN if any element is NaN.
///
/// `f64::max` drops a NaN operand, so a plain fold would hide a bad force.
fn max_abs(values: impl Iterator<Item = f64>) -> f64 {
    values.fold(0.0_f64, |maximum, value| {
        if maximum.is_nan() || value.is_nan() {
            f64::NAN
        } else {
            maximum.max(value.abs())
        }
    })
}

/// Unit tangent of Henkelman and Jónsson at an interior image.
///
/// When the neighbour energies are equal the weighted tangent points
/// along `forward + backward`, the secant from the previous image to the
/// next, scaled by the larger energy difference. On a plateau of three
/// equal energies that scale is zero, so the secant is used unscaled.
fn improved_tangent(
    previous: ArrayView1<'_, f64>,
    current: ArrayView1<'_, f64>,
    next: ArrayView1<'_, f64>,
    previous_energy: f64,
    current_energy: f64,
    next_energy: f64,
) -> Result<Array1<f64>, SaddleError> {
    let backward = &current - &previous;
    let forward = &next - &current;
    let mut tangent = if next_energy > current_energy && current_energy > previous_energy {
        forward
    } else if next_energy < current_energy && current_energy < previous_energy {
        backward
    } else {
        let forward_difference = (next_energy - current_energy).abs();
        let backward_difference = (previous_energy - current_energy).abs();
        let large = if forward_difference > backward_difference {
            forward_difference
        } else {
            backward_difference
        };
        let small = if forward_difference < backward_difference {
            forward_difference
        } else {
            backward_difference
        };
        match next_energy.partial_cmp(&previous_energy) {
            Some(std::cmp::Ordering::Greater) => forward * large + backward * small,
            Some(std::cmp::Ordering::Less) => forward * small + backward * large,
            _ => forward + backward,
        }
    };
    let norm = tangent.dot(&tangent).sqrt();
    if !(norm.is_finite() && norm > 0.0) {
        return Err(SaddleError::Invalid(
            "LI-NEB tangent has no direction: the neighbouring images coincide".into(),
        ));
    }
    tangent /= norm;
    Ok(tangent)
}

fn evaluate(
    config: &LineNebConfig,
    masses: ArrayView1<'_, f64>,
    surface: &impl BandSurface,
    q: ArrayView2<'_, f64>,
    step: usize,
) -> Result<Evaluation, SaddleError> {
    let sqrt_masses = masses.mapv(f64::sqrt);
    let positions = &q / &sqrt_masses;
    let mut energies = Array1::zeros(q.nrows());
    let mut gradients = Array2::zeros(q.raw_dim());
    surface.eval(positions.view(), &mut energies, &mut gradients)?;
    if !energies.iter().all(|v| v.is_finite()) {
        return Err(SaddleError::NonFinite("LI-NEB surface energies"));
    }
    if !gradients.iter().all(|v| v.is_finite()) {
        return Err(SaddleError::NonFinite("LI-NEB surface gradients"));
    }
    let below = match config.endpoint_below_energy {
        EndpointBelowEnergy::Restrained => BelowEndpoint::Restrained,
        EndpointBelowEnergy::Strict => BelowEndpoint::Strict { step },
    };
    let xi = action_weights(energies.view(), config.energy, config.reduced_planck, below)?;
    let action = trapezoid_action(positions.view(), masses, &xi)?;
    let force_q = -&gradients / &sqrt_masses;
    let action_force = action_force_q(q, &xi, force_q.view(), config.reduced_planck)?;
    let (_, lengths) = segments(q)?;
    let mut optimization_force = Array2::zeros(q.raw_dim());
    let mut max_interior_perpendicular = 0.0_f64;
    let mut max_interior_spring = 0.0_f64;

    for j in 1..q.nrows() - 1 {
        let tangent = improved_tangent(
            q.row(j - 1),
            q.row(j),
            q.row(j + 1),
            energies[j - 1],
            energies[j],
            energies[j + 1],
        )?;
        let raw = action_force.row(j);
        let perpendicular = &raw - raw.dot(&tangent) * &tangent;
        max_interior_perpendicular = max_abs(
            perpendicular
                .iter()
                .copied()
                .chain([max_interior_perpendicular]),
        );
        let spring = config.spring_k * (lengths[j] - lengths[j - 1]) * tangent;
        max_interior_spring = max_abs(spring.iter().copied().chain([max_interior_spring]));
        optimization_force
            .row_mut(j)
            .assign(&(perpendicular + spring));
    }

    for (endpoint, neighbour) in [(0, 1), (q.nrows() - 1, q.nrows() - 2)] {
        let fq = force_q.row(endpoint).to_owned();
        let force_norm = fq.dot(&fq).sqrt();
        if !(force_norm.is_finite() && force_norm > 0.0) {
            return Err(SaddleError::Invalid(format!(
                "LI-NEB endpoint {endpoint} needs a nonzero physical force"
            )));
        }
        let force_hat = fq / force_norm;
        let spring = config.spring_k * (&q.row(neighbour) - &q.row(endpoint));
        let spring_perpendicular = &spring - spring.dot(&force_hat) * &force_hat;
        let restraint = config.endpoint_kappa * (energies[endpoint] - config.energy) * force_hat;
        optimization_force
            .row_mut(endpoint)
            .assign(&(spring_perpendicular + restraint));
    }

    if !optimization_force.iter().all(|v| v.is_finite()) {
        return Err(SaddleError::NonFinite("LI-NEB optimization force"));
    }
    let last = q.nrows() - 1;
    let max_endpoint_optimized = max_abs(
        optimization_force
            .row(0)
            .iter()
            .chain(optimization_force.row(last).iter())
            .copied(),
    );
    Ok(Evaluation {
        action,
        optimization_force,
        max_interior_perpendicular,
        max_interior_spring,
        max_endpoint_optimized,
    })
}

fn one_surface(
    surface: &impl BandSurface,
    position: ArrayView1<'_, f64>,
) -> Result<(f64, Array1<f64>), SaddleError> {
    let positions = position.insert_axis(ndarray::Axis(0));
    let mut energies = Array1::zeros(1);
    let mut gradients = Array2::zeros((1, position.len()));
    surface.eval(positions, &mut energies, &mut gradients)?;
    if !energies[0].is_finite() || !gradients.iter().all(|value| value.is_finite()) {
        return Err(SaddleError::NonFinite("MEP seed surface"));
    }
    Ok((energies[0], gradients.row(0).to_owned()))
}

/// Opening step as a fraction of a path length.
///
/// The fraction is `sqrt(f64::EPSILON)`, which is dimensionless, so the
/// step has the unit of `length_scale`. A uniform change of length unit
/// scales the step with the coordinates. The value is fixed for a given
/// length scale. The length scale is a distance along the path, never a
/// coordinate, so a translation of the path leaves the step unchanged.
fn opening_step(length_scale: f64) -> Result<f64, SaddleError> {
    if !(length_scale.is_finite() && length_scale > 0.0) {
        return Err(SaddleError::Invalid(
            "MEP opening step needs a positive finite length scale".into(),
        ));
    }
    let step = length_scale * f64::EPSILON.sqrt();
    if !(step.is_finite() && step > 0.0) {
        return Err(SaddleError::NonFinite("MEP opening step"));
    }
    Ok(step)
}

fn charge(budget: &mut usize) -> Result<(), SaddleError> {
    if *budget == 0 {
        return Err(SaddleError::Invalid(
            "work budget exhausted: MEP energy slice used its surface-call budget".into(),
        ));
    }
    *budget -= 1;
    Ok(())
}

/// Fraction of the first energy drop that sets the length scale of a branch.
const SCALE_DROP_FRACTION: f64 = 0.1;

/// Distance along `direction` from `start_q` at which the energy has fallen
/// to `level`.
///
/// The distance is the length scale of a branch: it is measured on the
/// surface, so it has the unit of the coordinates and does not depend on
/// where the path sits. The search brackets a crossing from a trial of 1
/// by doubling or halving, then bisects to a relative width of
/// `sqrt(f64::EPSILON)`. The level is a small fraction of the first energy
/// drop, so the crossing lies close to the saddle.
fn level_distance(
    surface: &impl BandSurface,
    start_q: ArrayView1<'_, f64>,
    direction: ArrayView1<'_, f64>,
    sqrt_masses: ArrayView1<'_, f64>,
    level: f64,
    budget: &mut usize,
) -> Result<f64, SaddleError> {
    let mut energy_at = |distance: f64| -> Result<f64, SaddleError> {
        let trial_q = start_q.to_owned() + distance * &direction;
        let trial = &trial_q / &sqrt_masses;
        charge(budget)?;
        Ok(one_surface(surface, trial.view())?.0)
    };
    let mut lower;
    let mut upper = 1.0;
    if energy_at(upper)? > level {
        loop {
            lower = upper;
            upper += upper;
            if !upper.is_finite() {
                return Err(SaddleError::Invalid(
                    "MEP length scale did not reach the first energy level".into(),
                ));
            }
            if energy_at(upper)? <= level {
                break;
            }
        }
    } else {
        loop {
            let trial = 0.5 * upper;
            if trial < f64::MIN_POSITIVE {
                return Err(SaddleError::Invalid(
                    "MEP length scale found no energy above the first level at the saddle".into(),
                ));
            }
            if energy_at(trial)? > level {
                lower = trial;
                break;
            }
            upper = trial;
        }
    }
    while upper - lower > f64::EPSILON.sqrt() * upper {
        let middle = lower + 0.5 * (upper - lower);
        if energy_at(middle)? > level {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    Ok(upper)
}

fn energy_slice(
    surface: &impl BandSurface,
    q: ArrayView1<'_, f64>,
    sqrt_masses: ArrayView1<'_, f64>,
    target: f64,
    spacing: f64,
    budget: &mut usize,
) -> Result<Array1<f64>, SaddleError> {
    let position = &q / &sqrt_masses;
    charge(budget)?;
    let (energy, gradient) = one_surface(surface, position.view())?;
    if energy <= target {
        return Ok(q.to_owned());
    }
    let force_q = -gradient / sqrt_masses;
    let force_norm = force_q.dot(&force_q).sqrt();
    if force_norm <= 0.0 || force_norm.is_nan() {
        return Err(SaddleError::Invalid(
            "MEP energy slice reached a stationary point above its target".into(),
        ));
    }
    let direction = force_q / force_norm;
    let mut lower = 0.0;
    // Linear in the spacing of the images already placed, so a change of
    // length unit scales the bracket with the path and a translation of
    // the path leaves it unchanged.
    let mut upper = opening_step(spacing)?;
    loop {
        let trial_q = q.to_owned() + upper * &direction;
        let trial = &trial_q / &sqrt_masses;
        charge(budget)?;
        if one_surface(surface, trial.view())?.0 <= target {
            break;
        }
        lower = upper;
        upper += upper;
        if !upper.is_finite() {
            return Err(SaddleError::Invalid(
                "MEP energy slice did not cross its target energy".into(),
            ));
        }
    }
    loop {
        let middle = lower + 0.5 * (upper - lower);
        if middle == lower || middle == upper {
            break;
        }
        let trial_q = q.to_owned() + middle * &direction;
        let trial = &trial_q / &sqrt_masses;
        charge(budget)?;
        if one_surface(surface, trial.view())?.0 > target {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    Ok(q.to_owned() + lower * direction)
}

/// Scalar controls for [`fixed_energy_mep_seed`].
///
/// Start from [`FixedEnergySeed::default`] and set both energies with
/// [`FixedEnergySeed::with_saddle_energy`] and
/// [`FixedEnergySeed::with_target_energy`].
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct FixedEnergySeed {
    /// Energy at the index-one saddle.
    pub saddle_energy: f64,
    /// Energy both branches descend to.
    pub target_energy: f64,
    /// Images on the joined path, including both ends. At least 3.
    pub images: usize,
    /// Surface calls available to every slice on both branches.
    pub max_evaluations: usize,
}

impl Default for FixedEnergySeed {
    /// Nine images and a budget of 10,000 surface calls. Both energies
    /// are 0, which [`fixed_energy_mep_seed`] refuses: the target must lie
    /// below the saddle, and the caller sets both.
    fn default() -> Self {
        Self {
            saddle_energy: 0.0,
            target_energy: 0.0,
            images: 9,
            max_evaluations: 10_000,
        }
    }
}

with_setters!(FixedEnergySeed {
    with_saddle_energy => saddle_energy: val f64,
    with_target_energy => target_energy: val f64,
    with_images => images: val usize,
    with_max_evaluations => max_evaluations: val usize,
});

/// Energy-sliced steepest-descent seed from an index-one saddle.
///
/// Each branch follows the mass-scaled force from the unstable mode and
/// locates equal energy drops by bracketing, then both branches are resampled
/// to [`FixedEnergySeed::images`] equal arc-length positions. The requested
/// image count sets the energy resolution. [`FixedEnergySeed::max_evaluations`]
/// is the surface-call budget for every slice on both branches.
///
/// The opening displacement off the saddle, and the first trial step of
/// each energy slice, is `sqrt(f64::EPSILON)` times a distance along the
/// path. The opening displacement uses the distance from the saddle, along
/// the mode of that branch, at which the energy has fallen by a tenth of the
/// first energy drop. A slice uses the distance between the last two images
/// already placed on its branch. Both are measured on the surface in
/// mass-scaled coordinates, so a translation of the saddle and the surface
/// together leaves the seed unchanged, and a uniform change of length unit
/// scales the step with the coordinates. The factor is dimensionless. The
/// seed is fixed for a given saddle, mode, masses, and [`FixedEnergySeed`].
///
/// # Errors
///
/// [`SaddleError::Invalid`] when the input is inconsistent, when the energy
/// along the mode never falls to the length-scale level, when a slice
/// meets a stationary point or never crosses its level, and, with a
/// message that starts `work budget exhausted`, when the budget is used
/// up. [`SaddleError::NonFinite`] when the surface returns a non-finite
/// value, and the errors of [`BandSurface::eval`].
pub fn fixed_energy_mep_seed(
    surface: &impl BandSurface,
    saddle: ArrayView1<'_, f64>,
    mode: ArrayView1<'_, f64>,
    masses: ArrayView1<'_, f64>,
    seed: FixedEnergySeed,
) -> Result<Array2<f64>, SaddleError> {
    if seed.images < 3
        || saddle.len() != mode.len()
        || saddle.len() != masses.len()
        || !masses.iter().all(|mass| mass.is_finite() && *mass > 0.0)
        || !(seed.saddle_energy.is_finite()
            && seed.target_energy.is_finite()
            && seed.target_energy < seed.saddle_energy)
    {
        return Err(SaddleError::Invalid(
            "fixed-energy MEP seed input is inconsistent".into(),
        ));
    }
    let sqrt_masses = masses.mapv(f64::sqrt);
    let saddle_q = &saddle * &sqrt_masses;
    let mode_q = &mode * &sqrt_masses;
    let mode_norm = mode_q.dot(&mode_q).sqrt();
    if mode_norm <= 0.0 || mode_norm.is_nan() {
        return Err(SaddleError::Invalid(
            "fixed-energy MEP seed mode must be nonzero".into(),
        ));
    }
    let direction = mode_q / mode_norm;
    if !saddle_q.iter().all(|value| value.is_finite()) {
        return Err(SaddleError::NonFinite("MEP coordinate"));
    }
    let first_drop = (seed.saddle_energy - seed.target_energy) / (seed.images - 1) as f64;
    let scale_level = seed.saddle_energy - SCALE_DROP_FRACTION * first_drop;
    let mut budget = seed.max_evaluations;
    let mut branches = Vec::with_capacity(2);
    for sign in [-1.0, 1.0] {
        let branch_direction = sign * &direction;
        let scale = level_distance(
            surface,
            saddle_q.view(),
            branch_direction.view(),
            sqrt_masses.view(),
            scale_level,
            &mut budget,
        )?;
        let mut spacing = scale;
        let mut branch = Vec::with_capacity(seed.images);
        branch.push(saddle_q.clone());
        let mut q = &saddle_q + opening_step(scale)? * &branch_direction;
        for slice in 1..seed.images {
            let fraction = slice as f64 / (seed.images - 1) as f64;
            let level = seed.saddle_energy + fraction * (seed.target_energy - seed.saddle_energy);
            let next = energy_slice(
                surface,
                q.view(),
                sqrt_masses.view(),
                level,
                spacing,
                &mut budget,
            )?;
            let step = &next - &branch[slice - 1];
            let moved = step.dot(&step).sqrt();
            if moved > 0.0 {
                spacing = moved;
            }
            q = next;
            branch.push(q.clone());
        }
        branches.push(branch);
    }
    branches[0].reverse();
    let mut joined = branches.remove(0);
    joined.extend(branches.remove(0).into_iter().skip(1));
    let mut arc = vec![0.0; joined.len()];
    for i in 1..joined.len() {
        let segment = &joined[i] - &joined[i - 1];
        arc[i] = arc[i - 1] + segment.dot(&segment).sqrt();
    }
    let total = *arc
        .last()
        .ok_or_else(|| SaddleError::Shape("empty MEP seed".into()))?;
    if !(total > 0.0 && total.is_finite()) {
        return Err(SaddleError::Invalid(
            "fixed-energy MEP seed has zero arc length".into(),
        ));
    }
    let mut result_q = Array2::zeros((seed.images, saddle.len()));
    for image in 0..seed.images {
        let wanted = total * image as f64 / (seed.images - 1) as f64;
        let right = arc
            .partition_point(|value| *value < wanted)
            .min(arc.len() - 1);
        if right == 0 {
            result_q.row_mut(image).assign(&joined[0]);
        } else {
            let left = right - 1;
            let fraction = (wanted - arc[left]) / (arc[right] - arc[left]);
            result_q
                .row_mut(image)
                .assign(&(&joined[left] + fraction * (&joined[right] - &joined[left])));
        }
    }
    Ok(result_q / &sqrt_masses)
}

/// Fixed-image-count LI-NEB session using rgmin's global eOn
/// quick-min velocity projection with a fixed time step.
pub struct LineNebSession {
    config: LineNebConfig,
    masses: Array1<f64>,
    q: Array2<f64>,
    solver: Solver,
    iteration: usize,
    endpoint_violation: Option<LineNebEndpointViolation>,
}

impl LineNebSession {
    /// Build a session on `positions` in physical coordinates.
    ///
    /// # Errors
    ///
    /// [`SaddleError::Shape`] or [`SaddleError::Invalid`] when the images,
    /// masses, or controls are not a fixed-energy band.
    /// [`SaddleError::NonFinite`] when a position or mass is not finite.
    pub fn new(config: LineNebConfig, positions: Array2<f64>) -> Result<Self, SaddleError> {
        let energies = Array1::zeros(positions.nrows());
        let masses = config
            .masses
            .clone()
            .unwrap_or_else(|| Array1::ones(positions.ncols()));
        validate_inputs(
            positions.view(),
            energies.view(),
            masses.view(),
            config.energy,
            config.reduced_planck,
        )?;
        for (name, value) in [
            ("spring constant", config.spring_k),
            ("endpoint restraint", config.endpoint_kappa),
            ("force tolerance", config.force_tol),
            ("time step", config.time_step),
        ] {
            if !(value.is_finite() && value > 0.0) {
                return Err(SaddleError::Invalid(format!(
                    "LI-NEB {name} must be positive and finite"
                )));
            }
        }
        let q = mass_scaled(positions.view(), masses.view());
        segments(q.view())?;
        let dim = q.len();
        let mut solver = Solver::new(
            Method::QuickMin,
            Control::default()
                .with_maxiter(usize::MAX)
                .with_gtol(0.0)
                .with_istep(config.time_step)
                .with_maxmove(None),
            dim,
        );
        solver.set_quickmin_options(rgmin::QuickMinOptions::eon());
        solver.set_atom_maxmove(0.0);
        Ok(Self {
            config,
            masses,
            q,
            solver,
            iteration: 0,
            endpoint_violation: None,
        })
    }

    /// Current physical positions, including the movable endpoints.
    pub fn positions(&self) -> Array2<f64> {
        &self.q / &self.masses.mapv(f64::sqrt)
    }

    /// Controls this session was built with. Unset masses stay `None`;
    /// the session uses unit masses for them.
    pub fn config(&self) -> &LineNebConfig {
        &self.config
    }

    /// Steps completed so far.
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// Strict-endpoint record from the latest [`Self::step`], when that
    /// step stopped because an endpoint was below the fixed energy.
    #[must_use]
    pub fn endpoint_violation(&self) -> Option<LineNebEndpointViolation> {
        self.endpoint_violation
    }

    /// Evaluate through [`BandSurface`] and take one global projected
    /// quick-min step. The host owns every call and its force budget.
    ///
    /// # Errors
    ///
    /// [`SaddleError::Invalid`] when [`EndpointBelowEnergy::Strict`] finds
    /// an endpoint below the fixed energy. The message names that endpoint
    /// (`first` or `last`), `V - E`, and this one-based step.
    /// [`Self::endpoint_violation`] returns the same record, and the
    /// session iteration does not advance. Also [`SaddleError::Shape`],
    /// [`SaddleError::NonFinite`], [`SaddleError::Solver`], and the errors
    /// of [`BandSurface::eval`].
    pub fn step<S: BandSurface>(&mut self, surface: &S) -> Result<LineNebReport, SaddleError> {
        self.endpoint_violation = None;
        let shape = self.q.raw_dim();
        let config = &self.config;
        let masses = self.masses.view();
        let step = self
            .iteration
            .checked_add(1)
            .ok_or_else(|| SaddleError::Solver("LI-NEB iteration counter overflow".into()))?;
        let error = Mutex::new(None);
        let latest: Mutex<Option<(Array1<f64>, Evaluation)>> = Mutex::new(None);
        let rows = AtomicUsize::new(0);
        let oracle = Oracle::unbounded(self.q.len(), |flat: ArrayView1<'_, f64>| {
            let q = match flat.to_owned().into_shape_with_order(shape) {
                Ok(q) => q,
                Err(err) => {
                    let _ = store(&error, Some(SaddleError::Shape(err.to_string())));
                    return (f64::INFINITY, Array1::zeros(flat.len()));
                }
            };
            rows.fetch_add(q.nrows(), Ordering::Relaxed);
            match evaluate(config, masses, surface, q.view(), step) {
                Ok(evaluation) => {
                    let gradient =
                        -Array1::from_iter(evaluation.optimization_force.iter().copied());
                    let action = evaluation.action;
                    if !store(&latest, Some((flat.to_owned(), evaluation))) {
                        let _ = store(
                            &error,
                            Some(SaddleError::Invalid(
                                "a shared evaluation lock was poisoned".into(),
                            )),
                        );
                        return (f64::INFINITY, Array1::zeros(flat.len()));
                    }
                    (action, gradient)
                }
                Err(err) => {
                    let _ = store(&error, Some(err));
                    (f64::INFINITY, Array1::zeros(flat.len()))
                }
            }
        });
        let mut flat = Array1::from_iter(self.q.iter().copied());
        let stepped = self.solver.step(&oracle, &mut flat);
        drop(oracle);
        let stored = match error.into_inner() {
            Ok(value) => value,
            Err(_) => {
                self.solver.forget_evaluation();
                return Err(SaddleError::Invalid(
                    "a shared evaluation lock was poisoned".into(),
                ));
            }
        };
        if let Some(err) = stored {
            self.solver.forget_evaluation();
            self.endpoint_violation = LineNebEndpointViolation::from_error(&err);
            return Err(err);
        }
        stepped.map_err(|err| SaddleError::Solver(err.to_string()))?;
        let accepted = flat.clone();
        let accepted_q = flat
            .into_shape_with_order(shape)
            .map_err(|err| SaddleError::Shape(err.to_string()))?;
        if !accepted_q.iter().all(|v| v.is_finite()) {
            self.solver.forget_evaluation();
            return Err(SaddleError::NonFinite("LI-NEB step"));
        }
        let cached = match latest.into_inner() {
            Ok(value) => value,
            Err(_) => {
                self.solver.forget_evaluation();
                return Err(SaddleError::Invalid(
                    "a shared evaluation lock was poisoned".into(),
                ));
            }
        };
        let evaluation = match cached {
            Some((evaluated, evaluation)) if evaluated == accepted => evaluation,
            _ => match evaluate(config, masses, surface, accepted_q.view(), step) {
                Ok(evaluation) => {
                    rows.fetch_add(accepted_q.nrows(), Ordering::Relaxed);
                    evaluation
                }
                Err(err) => {
                    self.solver.forget_evaluation();
                    self.endpoint_violation = LineNebEndpointViolation::from_error(&err);
                    return Err(err);
                }
            },
        };
        self.q = accepted_q;
        self.iteration = step;
        let status = if evaluation.max_interior_perpendicular <= self.config.force_tol
            && evaluation.max_interior_spring <= self.config.force_tol
            && evaluation.max_endpoint_optimized <= self.config.force_tol
        {
            LineNebStatus::Converged
        } else {
            LineNebStatus::Running
        };
        Ok(LineNebReport {
            status,
            max_interior_perpendicular: evaluation.max_interior_perpendicular,
            max_interior_spring: evaluation.max_interior_spring,
            max_endpoint_optimized: evaluation.max_endpoint_optimized,
            action: evaluation.action,
            iteration: self.iteration,
            surface_rows: rows.load(Ordering::Relaxed),
        })
    }
}

#[cfg(test)]
mod endpoint_violation_tests {
    use super::LineNebEndpointViolation;
    use crate::SaddleError;

    #[test]
    fn the_message_round_trips_the_signed_difference() {
        for (endpoint, energy_difference, step) in [
            ("first", -0.05640000000000003_f64, 1_usize),
            ("last", -1.0e-20, 2),
            ("first", (1.2_f64 * 1.2 - 1.0).powi(2) - 0.25, 7),
        ] {
            let violation = LineNebEndpointViolation {
                endpoint,
                energy_difference,
                step,
            };
            let err = violation.to_error();
            let SaddleError::Invalid(message) = &err else {
                panic!("strict endpoint is an invalid parameter: {err:?}");
            };
            assert!(message.starts_with(&format!(
                "LI-NEB {endpoint} endpoint lies below the fixed energy at step {step}:"
            )));
            let recovered = LineNebEndpointViolation::from_error(&err).expect("message parses");
            assert_eq!(recovered.endpoint, endpoint);
            assert_eq!(recovered.step, step);
            assert_eq!(
                recovered.energy_difference.to_bits(),
                energy_difference.to_bits()
            );
        }
        let refused = SaddleError::Invalid("LI-NEB endpoint 0 lies below the fixed energy".into());
        assert!(LineNebEndpointViolation::from_error(&refused).is_none());
    }
}

#[cfg(test)]
mod max_abs_tests {
    use super::max_abs;

    #[test]
    fn max_abs_takes_the_largest_magnitude() {
        assert_eq!(max_abs([1.0, -3.0, 2.0].into_iter()), 3.0);
        assert_eq!(max_abs(std::iter::empty()), 0.0);
    }

    #[test]
    fn max_abs_propagates_nan_from_any_position() {
        assert!(max_abs([f64::NAN, 1.0].into_iter()).is_nan());
        assert!(max_abs([1.0, f64::NAN, 2.0].into_iter()).is_nan());
        assert!(max_abs([1.0, 2.0, f64::NAN].into_iter()).is_nan());
    }
}

#[cfg(test)]
mod tangent_tests {
    use ndarray::array;

    use super::improved_tangent;

    #[test]
    fn a_plateau_takes_the_normalized_secant() {
        let previous = array![0.0, 0.0];
        let current = array![1.0, 3.0];
        let next = array![3.0, 4.0];
        let tangent =
            improved_tangent(previous.view(), current.view(), next.view(), 2.5, 2.5, 2.5).unwrap();
        assert_eq!(tangent, array![0.6, 0.8]);
    }

    #[test]
    fn equal_neighbours_point_along_the_secant() {
        let previous = array![0.0, 0.0];
        let current = array![1.0, 3.0];
        let next = array![3.0, 4.0];
        let tangent =
            improved_tangent(previous.view(), current.view(), next.view(), 1.0, 2.5, 1.0).unwrap();
        assert_eq!(tangent, array![0.6, 0.8]);
    }

    #[test]
    fn coincident_neighbours_have_no_tangent() {
        let previous = array![0.0, 0.0];
        let current = array![1.0, 3.0];
        let tangent = improved_tangent(
            previous.view(),
            current.view(),
            previous.view(),
            2.5,
            2.5,
            2.5,
        );
        assert!(tangent.is_err());
    }
}

#[cfg(test)]
mod opening_step_tests {
    use super::opening_step;

    #[test]
    fn the_opening_step_scales_linearly_with_length() {
        let unit = opening_step(1.0).unwrap();
        let scaled = opening_step(8.0).unwrap();
        assert_eq!(unit, f64::EPSILON.sqrt());
        assert_eq!(scaled, unit * 8.0);
        assert!(opening_step(0.0).is_err());
        assert!(opening_step(f64::NAN).is_err());
        assert!(opening_step(f64::INFINITY).is_err());
    }
}

#[cfg(test)]
mod level_distance_tests {
    use super::level_distance;
    use crate::{BandSurface, SaddleError};
    use ndarray::{Array1, Array2, ArrayView2, array};

    struct Cup {
        offset: f64,
        width: f64,
    }

    impl BandSurface for Cup {
        fn eval(
            &self,
            positions: ArrayView2<'_, f64>,
            energies: &mut Array1<f64>,
            gradients: &mut Array2<f64>,
        ) -> Result<(), SaddleError> {
            for (i, row) in positions.outer_iter().enumerate() {
                let x = (row[0] - self.offset) / self.width;
                energies[i] = 1.0 - x * x;
                gradients[(i, 0)] = -2.0 * x / self.width;
            }
            Ok(())
        }
    }

    fn distance(offset: f64, width: f64, level: f64) -> f64 {
        let mut budget = 10_000;
        level_distance(
            &Cup { offset, width },
            array![offset].view(),
            array![1.0].view(),
            array![1.0].view(),
            level,
            &mut budget,
        )
        .unwrap()
    }

    #[test]
    fn the_level_distance_ignores_translation_and_follows_length() {
        let exact = 0.5_f64.sqrt();
        let tolerance = 4.0 * f64::EPSILON.sqrt();
        for width in [1.0e-6, 1.0, 1.0e6] {
            let reference = distance(0.0, width, 0.5);
            assert!((reference / width - exact).abs() <= tolerance * exact);
            for offset in [-3.0e3, 17.0, 5.0e4] {
                let shifted = distance(offset, width, 0.5);
                assert!(
                    (shifted - reference).abs() <= 1.0e-6 * reference + 1.0e-12 * offset.abs(),
                    "width {width} offset {offset}: {shifted} against {reference}"
                );
            }
        }
    }

    struct Hill;

    impl BandSurface for Hill {
        fn eval(
            &self,
            positions: ArrayView2<'_, f64>,
            energies: &mut Array1<f64>,
            gradients: &mut Array2<f64>,
        ) -> Result<(), SaddleError> {
            for (i, row) in positions.outer_iter().enumerate() {
                let x = row[0];
                energies[i] = 1.0 / (1.0 + x * x);
                gradients[(i, 0)] = -2.0 * (x / (1.0 + x * x)) / (1.0 + x * x);
            }
            Ok(())
        }
    }

    #[test]
    fn the_level_distance_refuses_a_level_the_energy_never_reaches() {
        let mut budget = 10_000;
        let below = level_distance(
            &Hill,
            array![0.0].view(),
            array![1.0].view(),
            array![1.0].view(),
            -1.0,
            &mut budget,
        );
        assert!(matches!(
            below,
            Err(SaddleError::Invalid(message)) if message.contains("did not reach")
        ));
        let mut budget = 10_000;
        let above = level_distance(
            &Hill,
            array![0.0].view(),
            array![1.0].view(),
            array![1.0].view(),
            2.0,
            &mut budget,
        );
        assert!(matches!(
            above,
            Err(SaddleError::Invalid(message)) if message.contains("no energy above")
        ));
    }

    #[test]
    fn the_level_distance_spends_the_budget() {
        let mut budget = 3;
        let err = level_distance(
            &Cup {
                offset: 0.0,
                width: 1.0,
            },
            array![0.0].view(),
            array![1.0].view(),
            array![1.0].view(),
            0.5,
            &mut budget,
        );
        assert!(matches!(
            err,
            Err(SaddleError::Invalid(message)) if message.starts_with("work budget exhausted")
        ));
    }
}
