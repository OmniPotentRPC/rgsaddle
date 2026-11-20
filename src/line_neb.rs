//! Fixed-energy line-integral NEB (LI-NEB).
//!
//! This is the microcanonical path optimization of Ásgeirsson,
//! Arnaldsson, and Jónsson, J. Chem. Phys. 148, 102334 (2018),
//! doi:10.1063/1.5007180. Coordinates are converted to
//! `q_i = sqrt(m_i) x_i`; the abbreviated action and all projections
//! are evaluated in that Euclidean, mass-scaled chart.
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

/// All dimensional controls for a fixed-energy LI-NEB session.
#[derive(Clone, Debug)]
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
    /// each atomic mass three times).
    pub masses: Array1<f64>,
    /// Fixed time step of the eOn global quick-min variant.
    pub time_step: f64,
    /// Reduced Planck constant in the host's unit system.
    pub reduced_planck: f64,
}

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
    /// Abbreviated action of the configuration after the step.
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

fn action_weights(
    energies: ArrayView1<'_, f64>,
    energy: f64,
    reduced_planck: f64,
) -> Result<Array1<f64>, SaddleError> {
    let mut xi = Array1::zeros(energies.len());
    for i in 1..energies.len() - 1 {
        let excess = energies[i] - energy;
        if excess.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
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
/// `positions` are physical coordinates and `masses` has one entry per
/// coordinate. Endpoint energies may equal `energy`; every interior
/// energy must be strictly larger.
pub fn fixed_energy_action(
    positions: ArrayView2<'_, f64>,
    energies: ArrayView1<'_, f64>,
    masses: ArrayView1<'_, f64>,
    energy: f64,
    reduced_planck: f64,
) -> Result<f64, SaddleError> {
    validate_inputs(positions, energies, masses, energy, reduced_planck)?;
    let q = mass_scaled(positions, masses);
    let xi = action_weights(energies, energy, reduced_planck)?;
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
    energies: ArrayView1<'_, f64>,
    force_q: ArrayView2<'_, f64>,
    energy: f64,
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
    let xi = action_weights(energies, energy, reduced_planck)?;
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

/// Negative gradient of the discretized action with respect to the
/// physical coordinates.
///
/// The endpoint rows are zero because LI-NEB replaces their action
/// gradient by the endpoint spring and iso-energy restraint.
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
    let force = action_force_q(q.view(), energies, force_q.view(), energy, reduced_planck)?;
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
    surface: &impl BandSurface,
    q: ArrayView2<'_, f64>,
) -> Result<Evaluation, SaddleError> {
    let sqrt_masses = config.masses.mapv(f64::sqrt);
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
    let action = fixed_energy_action(
        positions.view(),
        energies.view(),
        config.masses.view(),
        config.energy,
        config.reduced_planck,
    )?;
    let force_q = -&gradients / &sqrt_masses;
    let action_force = action_force_q(
        q,
        energies.view(),
        force_q.view(),
        config.energy,
        config.reduced_planck,
    )?;
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

fn coordinate_extent(values: impl Iterator<Item = f64>) -> Result<f64, SaddleError> {
    let mut extent = 0.0;
    for value in values {
        let magnitude = value.abs();
        if !magnitude.is_finite() {
            return Err(SaddleError::NonFinite("MEP coordinate"));
        }
        if magnitude > extent {
            extent = magnitude;
        }
    }
    Ok(extent)
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

fn energy_slice(
    surface: &impl BandSurface,
    q: ArrayView1<'_, f64>,
    sqrt_masses: ArrayView1<'_, f64>,
    target: f64,
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
    let extent = coordinate_extent(q.iter().copied())?;
    if extent <= 0.0 {
        return Err(SaddleError::Invalid(
            "MEP energy slice has no coordinate extent".into(),
        ));
    }
    let mut upper = extent.sqrt() * f64::EPSILON.sqrt();
    if !(upper > 0.0 && upper.is_finite()) {
        return Err(SaddleError::NonFinite("MEP energy-slice step"));
    }
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
#[derive(Clone, Copy, Debug)]
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

/// Energy-sliced steepest-descent seed from an index-one saddle.
///
/// Each branch follows the mass-scaled force from the unstable mode and
/// locates equal energy drops by bracketing, then both branches are resampled
/// to [`FixedEnergySeed::images`] equal arc-length positions. The requested
/// image count sets the energy resolution. [`FixedEnergySeed::max_evaluations`]
/// is the surface-call budget for every slice on both branches.
///
/// # Errors
///
/// [`SaddleError::Invalid`] when the input is inconsistent, when a slice
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
    let extent = coordinate_extent(saddle_q.iter().copied())?;
    // A saddle at the origin has no coordinate extent. The kick is then
    // one normalized-mode length, which is 1 after the division above.
    let length = if extent > 0.0 {
        extent
    } else {
        direction.dot(&direction).sqrt()
    };
    let displacement = length.sqrt() * f64::EPSILON.sqrt();
    let mut budget = seed.max_evaluations;
    let mut branches = Vec::with_capacity(2);
    for sign in [-1.0, 1.0] {
        let mut branch = Vec::with_capacity(seed.images);
        branch.push(saddle_q.clone());
        let mut q = &saddle_q + sign * displacement * &direction;
        for slice in 1..seed.images {
            let fraction = slice as f64 / (seed.images - 1) as f64;
            let level = seed.saddle_energy + fraction * (seed.target_energy - seed.saddle_energy);
            q = energy_slice(surface, q.view(), sqrt_masses.view(), level, &mut budget)?;
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
    q: Array2<f64>,
    solver: Solver,
    iteration: usize,
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
        validate_inputs(
            positions.view(),
            energies.view(),
            config.masses.view(),
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
        let q = mass_scaled(positions.view(), config.masses.view());
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
            q,
            solver,
            iteration: 0,
        })
    }

    /// Current physical positions, including the movable endpoints.
    pub fn positions(&self) -> Array2<f64> {
        &self.q / &self.config.masses.mapv(f64::sqrt)
    }

    /// Controls this session was built with.
    pub fn config(&self) -> &LineNebConfig {
        &self.config
    }

    /// Steps completed so far.
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// Evaluate through [`BandSurface`] and take one global projected
    /// quick-min step. The host owns every call and its force budget.
    pub fn step<S: BandSurface>(&mut self, surface: &S) -> Result<LineNebReport, SaddleError> {
        let shape = self.q.raw_dim();
        let config = &self.config;
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
            match evaluate(config, surface, q.view()) {
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
            return Err(err);
        }
        stepped.map_err(|err| SaddleError::Solver(err.to_string()))?;
        let accepted = flat.clone();
        self.q = flat
            .into_shape_with_order(shape)
            .map_err(|err| SaddleError::Shape(err.to_string()))?;
        if !self.q.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("LI-NEB step"));
        }
        self.iteration = self
            .iteration
            .checked_add(1)
            .ok_or_else(|| SaddleError::Solver("LI-NEB iteration counter overflow".into()))?;
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
            _ => {
                let evaluation = evaluate(config, surface, self.q.view())?;
                rows.fetch_add(self.q.nrows(), Ordering::Relaxed);
                evaluation
            }
        };
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
