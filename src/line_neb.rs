//! Fixed-energy line-integral NEB (LI-NEB).
//!
//! This is the microcanonical path optimization of Ásgeirsson,
//! Arnaldsson, and Jónsson, J. Chem. Phys. 148, 102334 (2018),
//! doi:10.1063/1.5007180. Coordinates are converted to
//! `q_i = sqrt(m_i) x_i`; the abbreviated action and all projections
//! are evaluated in that Euclidean, mass-scaled chart.
//! The equation numbering used below is the explicit Appendix A
//! restatement by Zhang et al., ChemRxiv (2025),
//! doi:10.26434/chemrxiv-2025-lvfbr-v2.
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
///
/// Fields may be added; construct this with [`LineNebConfig::default`] and
/// the `with_*` setters.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct LineNebConfig {
    /// Uniform spring constant in the mass-scaled chart.
    pub spring_k: f64,
    /// Coefficient of the endpoint energy restraint.
    pub endpoint_kappa: f64,
    /// Convergence tolerance applied to each interior force norm.
    pub force_tol: f64,
    /// Convergence tolerance applied to each endpoint force norm.
    pub endpoint_force_tol: f64,
    /// Convergence tolerance applied to the sum of interior force norms.
    pub sum_force_tol: f64,
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

impl Default for LineNebConfig {
    fn default() -> Self {
        Self {
            spring_k: 1.0,
            endpoint_kappa: 1.0,
            force_tol: 0.001,
            endpoint_force_tol: 0.001,
            sum_force_tol: 0.001,
            energy: 0.0,
            masses: Array1::zeros(0),
            time_step: 0.01,
            reduced_planck: 1.0,
        }
    }
}

impl LineNebConfig {
    /// Set the uniform spring constant in mass-scaled coordinates.
    pub fn with_spring_k(mut self, spring_k: f64) -> Self {
        self.spring_k = spring_k;
        self
    }

    /// Set the endpoint restraint coefficient.
    pub fn with_endpoint_kappa(mut self, endpoint_kappa: f64) -> Self {
        self.endpoint_kappa = endpoint_kappa;
        self
    }

    /// Set all three convergence tolerances to one value.
    pub fn with_force_tol(mut self, force_tol: f64) -> Self {
        self.force_tol = force_tol;
        self.endpoint_force_tol = force_tol;
        self.sum_force_tol = force_tol;
        self
    }

    /// Set the endpoint-force convergence tolerance.
    pub fn with_endpoint_force_tol(mut self, endpoint_force_tol: f64) -> Self {
        self.endpoint_force_tol = endpoint_force_tol;
        self
    }

    /// Set the summed interior transverse-force tolerance.
    pub fn with_sum_force_tol(mut self, sum_force_tol: f64) -> Self {
        self.sum_force_tol = sum_force_tol;
        self
    }

    /// Set the fixed target energy used by the action.
    pub fn with_energy(mut self, energy: f64) -> Self {
        self.energy = energy;
        self
    }

    /// Set one positive mass per physical coordinate.
    pub fn with_masses(mut self, masses: Array1<f64>) -> Self {
        self.masses = masses;
        self
    }

    /// Set the fixed quick-min step size.
    pub fn with_time_step(mut self, time_step: f64) -> Self {
        self.time_step = time_step;
        self
    }

    /// Set the reduced Planck constant in host units.
    pub fn with_reduced_planck(mut self, reduced_planck: f64) -> Self {
        self.reduced_planck = reduced_planck;
        self
    }
}

/// State of a host-stepped LI-NEB optimization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LineNebStatus {
    /// The configured residuals are not yet below their tolerances.
    Running,
    /// All LI-NEB residuals satisfy their tolerances.
    Converged,
}

/// Diagnostics from the configuration evaluated after one step.
#[derive(Clone, Debug)]
pub struct LineNebReport {
    /// Session state after this step.
    pub status: LineNebStatus,
    /// Largest Euclidean norm of the complete optimized force over the
    /// interior images (transverse action plus tangential spring).
    pub max_interior_optimized: f64,
    /// Largest Euclidean norm of the transverse action force over the
    /// interior images.
    pub max_interior_perpendicular: f64,
    /// Sum of Euclidean transverse-force norms over interior images,
    /// the third convergence residual in Eq. (27) of the LI-NEB method.
    pub sum_interior_perpendicular: f64,
    /// Largest Euclidean norm of the complete optimized endpoint force
    /// (projected spring plus energy restraint).
    pub max_endpoint_optimized: f64,
    /// Discrete fixed-energy abbreviated action at the accepted path.
    pub action: f64,
    /// Completed LI-NEB steps in this session.
    pub iteration: usize,
    /// Number of image rows passed to [`BandSurface::eval`] this step.
    pub surface_rows: usize,
}

#[derive(Clone, Debug)]
struct Evaluation {
    action: f64,
    optimization_force: Array2<f64>,
    max_interior_optimized: f64,
    max_interior_perpendicular: f64,
    sum_interior_perpendicular: f64,
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
    // Zhang et al. (2025), Appendix A, Eq. (22), reproducing the 2018
    // LI-NEB derivation: xi_i = sqrt(2(V_i-E))/hbar at every image.
    let mut xi = Array1::zeros(energies.len());
    for i in 0..energies.len() {
        let excess = energies[i] - energy;
        if excess.partial_cmp(&0.0) == Some(std::cmp::Ordering::Less) {
            return Err(SaddleError::Invalid(format!(
                "LI-NEB image {i} must not lie below the fixed energy"
            )));
        }
        if i > 0
            && i + 1 < energies.len()
            && excess.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater)
        {
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
/// energy must be strictly larger; endpoints must not lie below it.
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
    // Zhang et al. (2025), Appendix A, Eq. (21), reproducing the 2018
    // LI-NEB trapezoidal discretization.
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
        // Zhang et al. (2025), Appendix A, Eq. (22), in mass-scaled
        // coordinates.
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
        let large = forward_difference.max(backward_difference);
        let small = forward_difference.min(backward_difference);
        match next_energy.total_cmp(&previous_energy) {
            std::cmp::Ordering::Greater => forward * large + backward * small,
            std::cmp::Ordering::Less => forward * small + backward * large,
            std::cmp::Ordering::Equal => (forward + backward) * large,
        }
    };
    let norm = tangent.dot(&tangent).sqrt();
    if !(norm.is_finite() && norm > 0.0) {
        return Err(SaddleError::Invalid(
            "LI-NEB improved tangent has no direction".into(),
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
    let mut max_interior_optimized = 0.0_f64;
    let mut max_interior_perpendicular = 0.0_f64;
    let mut sum_interior_perpendicular = 0.0_f64;

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
        let perpendicular_norm = perpendicular.dot(&perpendicular).sqrt();
        max_interior_perpendicular = max_interior_perpendicular.max(perpendicular_norm);
        sum_interior_perpendicular += perpendicular_norm;
        let spring = config.spring_k * (lengths[j] - lengths[j - 1]) * tangent;
        // LI-NEB transverse action force plus the parallel spring,
        // Zhang et al. (2025), Appendix A, Eqs. (24)-(25).
        let optimized = perpendicular + spring;
        max_interior_optimized = max_interior_optimized.max(optimized.dot(&optimized).sqrt());
        optimization_force.row_mut(j).assign(&optimized);
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
        // Turning-point restraint, Zhang et al. (2025), Appendix A, Eq. (26).
        optimization_force
            .row_mut(endpoint)
            .assign(&(spring_perpendicular + restraint));
    }

    if !optimization_force.iter().all(|v| v.is_finite()) {
        return Err(SaddleError::NonFinite("LI-NEB optimization force"));
    }
    let last = q.nrows() - 1;
    let first_endpoint = optimization_force.row(0);
    let last_endpoint = optimization_force.row(last);
    let max_endpoint_optimized = first_endpoint
        .dot(&first_endpoint)
        .sqrt()
        .max(last_endpoint.dot(&last_endpoint).sqrt());
    Ok(Evaluation {
        action,
        optimization_force,
        max_interior_optimized,
        max_interior_perpendicular,
        sum_interior_perpendicular,
        max_endpoint_optimized,
    })
}

/// Natural cubic interpolation of vector-valued data on a strict knot grid.
///
/// LI-NEB consumers use this for coordinates, parallel forces, and Hessian
/// elements. Sampling returns the value and its first two knot derivatives.
#[derive(Clone, Debug)]
pub struct NaturalCubic {
    knots: Vec<f64>,
    values: Vec<Vec<f64>>,
    second: Vec<Vec<f64>>,
}

/// A cubic interpolation value and its first two derivatives.
#[derive(Clone, Debug, PartialEq)]
pub struct CubicSample {
    /// Interpolated value at the sample point.
    pub value: Vec<f64>,
    /// First derivative with respect to the knot coordinate.
    pub first: Vec<f64>,
    /// Second derivative with respect to the knot coordinate.
    pub second: Vec<f64>,
}

impl NaturalCubic {
    /// Build a natural cubic interpolation on a strict knot grid.
    pub fn new(knots: Vec<f64>, values: Vec<Vec<f64>>) -> Result<Self, SaddleError> {
        if knots.len() != values.len() || knots.len() < 2 {
            return Err(SaddleError::Shape(
                "cubic interpolation needs at least two equally counted knots and rows".into(),
            ));
        }
        if !knots.iter().all(|knot| knot.is_finite())
            || knots.windows(2).any(|pair| pair[1] <= pair[0])
        {
            return Err(SaddleError::Invalid(
                "cubic interpolation knots must be finite and strictly increasing".into(),
            ));
        }
        let width = values[0].len();
        if width == 0 || values.iter().any(|row| row.len() != width) {
            return Err(SaddleError::Shape(
                "cubic interpolation rows must have one common nonzero width".into(),
            ));
        }
        let count = knots.len();
        let mut second = vec![vec![0.0; width]; count];
        if count > 2 {
            for component in 0..width {
                let interior = count - 2;
                let mut diagonal = vec![0.0; interior];
                let mut upper = vec![0.0; interior];
                let mut rhs = vec![0.0; interior];
                for row in 0..interior {
                    let i = row + 1;
                    let left = knots[i] - knots[i - 1];
                    let right = knots[i + 1] - knots[i];
                    diagonal[row] = 2.0 * (left + right);
                    if row + 1 < interior {
                        upper[row] = right;
                    }
                    rhs[row] = 6.0
                        * ((values[i + 1][component] - values[i][component]) / right
                            - (values[i][component] - values[i - 1][component]) / left);
                    if row > 0 {
                        let factor = left / diagonal[row - 1];
                        diagonal[row] -= factor * upper[row - 1];
                        rhs[row] -= factor * rhs[row - 1];
                    }
                }
                for row in (0..interior).rev() {
                    let next = if row + 1 < interior {
                        upper[row] * second[row + 2][component]
                    } else {
                        0.0
                    };
                    second[row + 1][component] = (rhs[row] - next) / diagonal[row];
                }
            }
        }
        if !values.iter().flatten().all(|value| value.is_finite())
            || !second.iter().flatten().all(|value| value.is_finite())
        {
            return Err(SaddleError::NonFinite("cubic interpolation"));
        }
        Ok(Self {
            knots,
            values,
            second,
        })
    }

    /// Return the strict knot grid.
    pub fn knots(&self) -> &[f64] {
        &self.knots
    }

    /// Sample value and first two derivatives at `x` within the knot interval.
    pub fn sample(&self, x: f64) -> Result<CubicSample, SaddleError> {
        if !x.is_finite() || x < self.knots[0] || x > self.knots[self.knots.len() - 1] {
            return Err(SaddleError::Invalid(
                "cubic interpolation sample lies outside its knot interval".into(),
            ));
        }
        let i = match self.knots.binary_search_by(|value| value.total_cmp(&x)) {
            Ok(index) => index.min(self.knots.len() - 2),
            Err(index) => index.saturating_sub(1).min(self.knots.len() - 2),
        };
        let h = self.knots[i + 1] - self.knots[i];
        let a = (self.knots[i + 1] - x) / h;
        let b = (x - self.knots[i]) / h;
        let width = self.values[i].len();
        let mut value = vec![0.0; width];
        let mut first = vec![0.0; width];
        let mut second = vec![0.0; width];
        for component in 0..width {
            let y0 = self.values[i][component];
            let y1 = self.values[i + 1][component];
            let m0 = self.second[i][component];
            let m1 = self.second[i + 1][component];
            value[component] =
                a * y0 + b * y1 + ((a * a * a - a) * m0 + (b * b * b - b) * m1) * h * h / 6.0;
            first[component] =
                (y1 - y0) / h + ((-3.0 * a * a + 1.0) * m0 + (3.0 * b * b - 1.0) * m1) * h / 6.0;
            second[component] = a * m0 + b * m1;
        }
        Ok(CubicSample {
            value,
            first,
            second,
        })
    }
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

fn descent_direction(
    surface: &impl BandSurface,
    q: ArrayView1<'_, f64>,
    sqrt_masses: ArrayView1<'_, f64>,
) -> Result<Array1<f64>, SaddleError> {
    let position = &q / &sqrt_masses;
    let (_, gradient) = one_surface(surface, position.view())?;
    let force_q = -gradient / sqrt_masses;
    let force_norm = force_q.dot(&force_q).sqrt();
    if force_norm.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return Err(SaddleError::Invalid(
            "MEP descent reached a stationary point above its target".into(),
        ));
    }
    Ok(force_q / force_norm)
}

fn rk4_descent(
    surface: &impl BandSurface,
    q: ArrayView1<'_, f64>,
    sqrt_masses: ArrayView1<'_, f64>,
    step: f64,
) -> Result<Array1<f64>, SaddleError> {
    let half = 0.5 * step;
    let k1 = descent_direction(surface, q, sqrt_masses)?;
    let k2_q = q.to_owned() + half * &k1;
    let k2 = descent_direction(surface, k2_q.view(), sqrt_masses)?;
    let k3_q = q.to_owned() + half * &k2;
    let k3 = descent_direction(surface, k3_q.view(), sqrt_masses)?;
    let k4_q = q.to_owned() + step * &k3;
    let k4 = descent_direction(surface, k4_q.view(), sqrt_masses)?;
    Ok(q.to_owned() + step * (&k1 + 2.0 * &k2 + 2.0 * &k3 + &k4) / 6.0)
}

fn crossing_on_segment(
    surface: &impl BandSurface,
    mut above: Array1<f64>,
    mut below: Array1<f64>,
    sqrt_masses: ArrayView1<'_, f64>,
    target: f64,
) -> Result<Array1<f64>, SaddleError> {
    loop {
        let middle = 0.5 * (&above + &below);
        if middle == above || middle == below {
            return Ok(below);
        }
        let position = &middle / &sqrt_masses;
        if one_surface(surface, position.view())?.0 > target {
            above = middle;
        } else {
            below = middle;
        }
    }
}

fn energy_slice(
    surface: &impl BandSurface,
    start: ArrayView1<'_, f64>,
    sqrt_masses: ArrayView1<'_, f64>,
    target: f64,
    step: &mut f64,
    path: &mut Vec<Array1<f64>>,
) -> Result<Array1<f64>, SaddleError> {
    let mut q = start.to_owned();
    let position = &q / &sqrt_masses;
    let mut energy = one_surface(surface, position.view())?.0;
    if energy <= target {
        return Ok(q);
    }
    loop {
        let full = rk4_descent(surface, q.view(), sqrt_masses, *step)?;
        let half_step = 0.5 * *step;
        let first_half = rk4_descent(surface, q.view(), sqrt_masses, half_step)?;
        let second_half = rk4_descent(surface, first_half.view(), sqrt_masses, half_step)?;
        let difference = &second_half - &full;
        let error = difference.dot(&difference).sqrt();
        let coordinate_scale = q
            .iter()
            .chain(second_half.iter())
            .map(|value| value.abs())
            .fold(1.0_f64, f64::max);
        let tolerance = f64::EPSILON.sqrt() * coordinate_scale;
        if error > tolerance {
            *step = half_step;
            if !(*step > 0.0 && step.is_finite()) || q == second_half {
                return Err(SaddleError::Stalled(
                    "adaptive MEP descent could not resolve a step".into(),
                ));
            }
            continue;
        }
        let trial_position = &second_half / &sqrt_masses;
        let trial_energy = one_surface(surface, trial_position.view())?.0;
        if trial_energy >= energy {
            *step = half_step;
            if q == second_half {
                return Err(SaddleError::Stalled(
                    "adaptive MEP descent made no downhill progress".into(),
                ));
            }
            continue;
        }
        if trial_energy <= target {
            let crossing = crossing_on_segment(surface, q, second_half, sqrt_masses, target)?;
            path.push(crossing.clone());
            return Ok(crossing);
        }
        q = second_half;
        energy = trial_energy;
        path.push(q.clone());
        if error < tolerance / 16.0 {
            *step += *step;
            if !step.is_finite() {
                return Err(SaddleError::NonFinite("adaptive MEP descent step"));
            }
        }
    }
}

/// Energy-sliced steepest-descent seed from an index-one saddle.
///
/// Each branch follows the mass-scaled force from the unstable mode and
/// locates equal energy drops by bracketing, then both branches are resampled
/// to `images` equal arc-length positions. The requested image count sets the
/// energy resolution; there is no coordinate step or iteration limit.
pub fn fixed_energy_mep_seed(
    surface: &impl BandSurface,
    saddle: ArrayView1<'_, f64>,
    mode: ArrayView1<'_, f64>,
    masses: ArrayView1<'_, f64>,
    saddle_energy: f64,
    target_energy: f64,
    images: usize,
) -> Result<Array2<f64>, SaddleError> {
    if images < 3
        || saddle.len() != mode.len()
        || saddle.len() != masses.len()
        || !masses.iter().all(|mass| mass.is_finite() && *mass > 0.0)
        || !(saddle_energy.is_finite()
            && target_energy.is_finite()
            && target_energy < saddle_energy)
    {
        return Err(SaddleError::Invalid(
            "fixed-energy MEP seed input is inconsistent".into(),
        ));
    }
    let sqrt_masses = masses.mapv(f64::sqrt);
    let saddle_q = &saddle * &sqrt_masses;
    let mode_q = &mode * &sqrt_masses;
    let mode_norm = mode_q.dot(&mode_q).sqrt();
    if mode_norm.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return Err(SaddleError::Invalid(
            "fixed-energy MEP seed mode must be nonzero".into(),
        ));
    }
    let direction = mode_q / mode_norm;
    let scale = saddle_q
        .iter()
        .map(|value| value.abs())
        .fold(1.0_f64, f64::max);
    let displacement = scale * f64::EPSILON.cbrt();
    let mut branches = Vec::with_capacity(2);
    for sign in [-1.0, 1.0] {
        let mut branch = Vec::new();
        branch.push(saddle_q.clone());
        let mut q = &saddle_q + sign * displacement * &direction;
        branch.push(q.clone());
        let mut step = displacement;
        for slice in 1..images {
            let fraction = slice as f64 / (images - 1) as f64;
            let level = saddle_energy + fraction * (target_energy - saddle_energy);
            q = energy_slice(
                surface,
                q.view(),
                sqrt_masses.view(),
                level,
                &mut step,
                &mut branch,
            )?;
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
    let mut result_q = Array2::zeros((images, saddle.len()));
    for image in 0..images {
        let wanted = total * image as f64 / (images - 1) as f64;
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
    /// Create a fixed-energy LI-NEB session from initial physical positions.
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
            ("interior force tolerance", config.force_tol),
            ("endpoint force tolerance", config.endpoint_force_tol),
            ("summed force tolerance", config.sum_force_tol),
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
        let control = Control::default()
            .with_maxiter(usize::MAX)
            .with_gtol(0.0)
            .with_istep(config.time_step);
        let mut solver = Solver::new(Method::QuickMin, control, dim);
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

    /// Read the immutable LI-NEB configuration.
    pub fn config(&self) -> &LineNebConfig {
        &self.config
    }

    /// Number of accepted LI-NEB steps so far.
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// Evaluate through [`BandSurface`] and take one global projected
    /// quick-min step. The host owns every call and its force budget.
    pub fn step<S: BandSurface>(&mut self, surface: &S) -> Result<LineNebReport, SaddleError> {
        let shape = self.q.raw_dim();
        let config = &self.config;
        let error = Mutex::new(None);
        let latest = Mutex::new(None);
        let rows = AtomicUsize::new(0);
        let oracle = Oracle::unbounded(self.q.len(), |flat: ArrayView1<'_, f64>| {
            let q = match flat.to_owned().into_shape_with_order(shape) {
                Ok(q) => q,
                Err(err) => {
                    *lock_unpoisoned(&error) = Some(SaddleError::Shape(err.to_string()));
                    return (f64::INFINITY, Array1::zeros(flat.len()));
                }
            };
            rows.fetch_add(q.nrows(), Ordering::Relaxed);
            match evaluate(config, surface, q.view()) {
                Ok(evaluation) => {
                    let gradient =
                        -Array1::from_iter(evaluation.optimization_force.iter().copied());
                    let action = evaluation.action;
                    *lock_unpoisoned(&latest) = Some(evaluation);
                    (action, gradient)
                }
                Err(err) => {
                    *lock_unpoisoned(&error) = Some(err);
                    (f64::INFINITY, Array1::zeros(flat.len()))
                }
            }
        });
        let previous_q = self.q.clone();
        let mut flat = Array1::from_iter(self.q.iter().copied());
        let stepped = self.solver.step(&oracle, &mut flat);
        drop(oracle);
        if let Some(err) = into_inner_unpoisoned(error) {
            self.solver.forget_evaluation();
            return Err(err);
        }
        stepped.map_err(|err| SaddleError::Solver(err.to_string()))?;
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
        let evaluation = into_inner_unpoisoned(latest)
            .ok_or_else(|| SaddleError::Solver("LI-NEB step produced no evaluation".into()))?;
        // All three residuals printed by Zhang et al. (2025), Appendix A,
        // Eq. (27), reproducing the 2018 LI-NEB convergence test.
        let status = if evaluation.max_interior_optimized <= self.config.force_tol
            && evaluation.max_endpoint_optimized <= self.config.endpoint_force_tol
            && evaluation.sum_interior_perpendicular <= self.config.sum_force_tol
        {
            LineNebStatus::Converged
        } else {
            LineNebStatus::Running
        };
        if status == LineNebStatus::Running && self.q == previous_q {
            return Err(SaddleError::Stalled(
                "LI-NEB optimizer made no progress above its convergence tolerances".into(),
            ));
        }
        Ok(LineNebReport {
            status,
            max_interior_optimized: evaluation.max_interior_optimized,
            max_interior_perpendicular: evaluation.max_interior_perpendicular,
            sum_interior_perpendicular: evaluation.sum_interior_perpendicular,
            max_endpoint_optimized: evaluation.max_endpoint_optimized,
            action: evaluation.action,
            iteration: self.iteration,
            surface_rows: rows.load(Ordering::Relaxed),
        })
    }
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poison) => poison.into_inner(),
    }
}

fn into_inner_unpoisoned<T>(mutex: Mutex<T>) -> T {
    match mutex.into_inner() {
        Ok(value) => value,
        Err(poison) => poison.into_inner(),
    }
}
