//! Ring-polymer centroid-hyperplane PMF on the unmodified PAFI walk.
//!
//! The one-dimensional double well is `(x² − 1)²`. Its barrier is the
//! free-energy difference of the ring partition function, integrated by
//! Gauss–Hermite on the Fourier modes of that closed-form potential.
//! Nothing in that integral calls the ring surface, the centroid path, or
//! the PAFI integrator. The constraint itself is the public [`Hyperplane`]
//! used by [`rgsaddle::HyperplaneSampler`]. The two-dimensional well adds
//! `2 y²`. Above the two-bead crossover the collapsed path is the instanton, and the
//! harmonic barrier is the same partition function with the quartic
//! replaced by its curvature. The seeds measure the anharmonic remainder.
//! The result is a static centroid quantum transition state theory free
//! energy, not a ring-polymer molecular dynamics rate.

use std::f64::consts::{PI, SQRT_2};
use std::num::NonZeroUsize;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, array, s};
use rgmin::{Control, Method, Oracle, Solver, StopNorm};
use rgsaddle::{
    ArcPath, CartesianLayout, Hyperplane, MinModeConfig, MinModeSession, MinModeStatus, PafiConfig,
    PafiIntegrator, PointSurface, RelaxConfig, RingPolymer, RingSettings, SaddleError, bead_study,
    centroid_hyperplane, centroid_path, collapsed, exact_eigh, pafi_profile,
};
#[cfg(feature = "readcon")]
use rgsaddle::{ConImage, ConSpecies, write_band};

/// Independent seeds in the two-dimensional comparison.
const SEEDS: usize = 6;
/// Blocks inside each constrained profile.
const BLOCKS: usize = 6;

#[link(name = "m")]
unsafe extern "C" {
    #[link_name = "erfc"]
    fn c_erfc(value: f64) -> f64;
}

fn erfc(value: f64) -> f64 {
    // SAFETY: C erfc accepts every finite f64 and returns a scalar value.
    unsafe { c_erfc(value) }
}

struct Flat;

impl PointSurface for Flat {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        Ok((0.0, Array1::zeros(x.len())))
    }

    fn hessian_vector(
        &self,
        x: ArrayView1<f64>,
        v: ArrayView1<f64>,
    ) -> Result<Option<Array1<f64>>, SaddleError> {
        if v.len() != x.len() {
            return Err(SaddleError::Shape(
                "the flat Hessian product must match the coordinate".into(),
            ));
        }
        Ok(Some(Array1::zeros(v.len())))
    }
}

struct NoHessian;

impl PointSurface for NoHessian {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        Ok((0.0, Array1::zeros(x.len())))
    }
}

struct ConstantTwo;

impl PointSurface for ConstantTwo {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        Ok((2.0, Array1::zeros(x.len())))
    }

    fn hessian_vector(
        &self,
        _x: ArrayView1<f64>,
        v: ArrayView1<f64>,
    ) -> Result<Option<Array1<f64>>, SaddleError> {
        Ok(Some(Array1::zeros(v.len())))
    }
}

struct DenseQuadratic;

impl PointSurface for DenseQuadratic {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        if x.len() != 2 {
            return Err(SaddleError::Shape(
                "the dense quadratic has two coordinates".into(),
            ));
        }
        let gradient = ndarray::array![2.0 * x[0] - 3.0 * x[1], -3.0 * x[0] + 4.0 * x[1]];
        Ok((0.5 * x.dot(&gradient), gradient))
    }

    fn hessian_vector(
        &self,
        x: ArrayView1<f64>,
        v: ArrayView1<f64>,
    ) -> Result<Option<Array1<f64>>, SaddleError> {
        if x.len() != 2 || v.len() != 2 {
            return Err(SaddleError::Shape(
                "the dense quadratic Hessian has two coordinates".into(),
            ));
        }
        Ok(Some(ndarray::array![
            2.0 * v[0] - 3.0 * v[1],
            -3.0 * v[0] + 4.0 * v[1]
        ]))
    }
}

/// `V(x) = (x² − 1)²`, so `V' = 4x(x² − 1)` and `V'' = 12x² − 4`.
struct DoubleWell;

fn quartic(x: f64) -> (f64, f64, f64) {
    let square = x * x;
    (
        (square - 1.0).powi(2),
        4.0 * x * (square - 1.0),
        12.0 * square - 4.0,
    )
}

/// `(energy, derivative, curvature)` of the quartic, the toy the partition function integrates.
fn quartic_point(x: f64) -> (f64, f64, f64) {
    quartic(x)
}

impl PointSurface for DoubleWell {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        if x.len() != 1 {
            return Err(SaddleError::Shape(
                "the double well is one coordinate".into(),
            ));
        }
        let (energy, gradient, _) = quartic(x[0]);
        Ok((energy, Array1::from_elem(1, gradient)))
    }

    fn hessian_vector(
        &self,
        x: ArrayView1<f64>,
        v: ArrayView1<f64>,
    ) -> Result<Option<Array1<f64>>, SaddleError> {
        if x.len() != 1 || v.len() != 1 {
            return Err(SaddleError::Shape(
                "the double well Hessian is one coordinate".into(),
            ));
        }
        let (_, _, curvature) = quartic(x[0]);
        Ok(Some(Array1::from_elem(1, curvature * v[0])))
    }
}

/// `V(x, y) = (x² − 1)² + 2 y²`.
struct Coupled;

fn transverse(y: f64) -> (f64, f64, f64) {
    (2.0 * y * y, 4.0 * y, 4.0)
}

impl PointSurface for Coupled {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        if x.len() != 2 {
            return Err(SaddleError::Shape(
                "the coupled well is two coordinates".into(),
            ));
        }
        let (along, along_force, _) = quartic(x[0]);
        let (across, across_force, _) = transverse(x[1]);
        Ok((along + across, ndarray::array![along_force, across_force]))
    }

    fn hessian_vector(
        &self,
        x: ArrayView1<f64>,
        v: ArrayView1<f64>,
    ) -> Result<Option<Array1<f64>>, SaddleError> {
        if x.len() != 2 || v.len() != 2 {
            return Err(SaddleError::Shape(
                "the coupled Hessian is two coordinates".into(),
            ));
        }
        let (_, _, along) = quartic(x[0]);
        let (_, _, across) = transverse(x[1]);
        Ok(Some(ndarray::array![along * v[0], across * v[1]]))
    }
}

struct Pin;

impl PointSurface for Pin {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        Ok((0.0, Array1::zeros(x.len())))
    }

    fn excluded_modes(&self, x: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
        let mut modes = Array2::zeros((1, x.len()));
        modes[(0, 0)] = 1.0;
        Ok(modes)
    }
}

/// Excludes both axes of a bead.
struct Lock;

impl PointSurface for Lock {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        Ok((0.0, Array1::zeros(x.len())))
    }

    fn excluded_modes(&self, x: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
        let mut modes = Array2::zeros((x.len(), x.len()));
        for index in 0..x.len() {
            modes[(index, index)] = 1.0;
        }
        Ok(modes)
    }
}

/// Excludes `e₀` on a bead with positive `x`, and `e₁` otherwise.
struct Split;

impl PointSurface for Split {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        Ok((0.0, Array1::zeros(x.len())))
    }

    fn excluded_modes(&self, x: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
        let mut modes = Array2::zeros((1, x.len()));
        if strictly_positive(x[0]) {
            modes[(0, 0)] = 1.0;
        } else {
            modes[(0, 1)] = 1.0;
        }
        Ok(modes)
    }
}

/// Excludes both axes when `x` is positive, and only `e₀` otherwise.
struct Half;

impl PointSurface for Half {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        Ok((0.0, Array1::zeros(x.len())))
    }

    fn excluded_modes(&self, x: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
        if strictly_positive(x[0]) {
            let mut modes = Array2::zeros((x.len(), x.len()));
            for index in 0..x.len() {
                modes[(index, index)] = 1.0;
            }
            Ok(modes)
        } else {
            let mut modes = Array2::zeros((1, x.len()));
            modes[(0, 0)] = 1.0;
            Ok(modes)
        }
    }
}

struct DiagonalExcluded;

impl PointSurface for DiagonalExcluded {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        Ok((0.0, Array1::zeros(x.len())))
    }

    fn excluded_modes(&self, x: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
        if x.len() != 2 {
            return Err(SaddleError::Shape(
                "diagonal excluded-mode surface needs two coordinates".into(),
            ));
        }
        Ok(ndarray::array![[1.0, 1.0]])
    }
}

struct MassWeightedTranslations;

impl PointSurface for MassWeightedTranslations {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        Ok((0.0, Array1::zeros(x.len())))
    }

    fn excluded_modes(&self, x: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
        if x.len() != 6 {
            return Err(SaddleError::Shape(
                "mass-weighted translation surface needs two Cartesian atoms".into(),
            ));
        }
        let heavy = 12.0_f64.sqrt();
        Ok(ndarray::array![
            [1.0, 0.0, 0.0, heavy, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, heavy, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, heavy],
        ])
    }
}

/// Excludes rigid translations and rotations of two Cartesian atoms.
struct RigidBody;

impl PointSurface for RigidBody {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        Ok((0.0, Array1::zeros(x.len())))
    }

    fn excluded_modes(&self, x: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
        if x.len() != 6 {
            return Err(SaddleError::Shape(
                "rigid-body test surface needs two Cartesian atoms".into(),
            ));
        }
        let mut modes = Array2::zeros((6, 6));
        for atom in 0..2 {
            let base = 3 * atom;
            modes[(0, base)] = 1.0;
            modes[(1, base + 1)] = 1.0;
            modes[(2, base + 2)] = 1.0;
            let px = x[base];
            let py = x[base + 1];
            let pz = x[base + 2];
            modes[(3, base)] = 0.0;
            modes[(3, base + 1)] = -pz;
            modes[(3, base + 2)] = py;
            modes[(4, base)] = pz;
            modes[(4, base + 1)] = 0.0;
            modes[(4, base + 2)] = -px;
            modes[(5, base)] = -py;
            modes[(5, base + 1)] = px;
            modes[(5, base + 2)] = 0.0;
        }
        Ok(modes)
    }
}

fn settings(beta: f64, dof: usize) -> RingSettings {
    RingSettings::default()
        .with_beta(beta)
        .with_hbar(1.0)
        .with_masses(Array1::ones(dof))
}

fn roundoff(scale: f64) -> f64 {
    f64::EPSILON.sqrt() * (1.0 + scale.abs())
}

fn strictly_positive(value: f64) -> bool {
    matches!(value.partial_cmp(&0.0), Some(std::cmp::Ordering::Greater))
}

fn strictly_negative(value: f64) -> bool {
    matches!(value.partial_cmp(&0.0), Some(std::cmp::Ordering::Less))
}

/// Two-sided normal coverage `P(|Z| > z) = 1/n²`, with `z = √2 erfc⁻¹(1/n²)`.
fn coverage(count: usize) -> Result<f64, SaddleError> {
    if count < 2 {
        return Err(SaddleError::Invalid(
            "a coverage factor needs at least two samples".into(),
        ));
    }
    Ok(SQRT_2 * inv_erfc(1.0 / (count as f64).powi(2))?)
}

fn inv_erfc(tail: f64) -> Result<f64, SaddleError> {
    if !(tail.is_finite() && tail > 0.0 && tail < 2.0) {
        return Err(SaddleError::Invalid(
            "an erfc tail must lie in (0, 2)".into(),
        ));
    }
    let (mut lo, mut hi) = if tail <= 1.0 {
        let mut hi = (-tail.ln()).sqrt();
        while erfc(hi) > tail {
            hi = hi + hi + 1.0;
            if !hi.is_finite() {
                return Err(SaddleError::NonFinite("inverse erfc"));
            }
        }
        (0.0, hi)
    } else {
        let mut lo = -(-(2.0 - tail).ln()).sqrt();
        while erfc(lo) < tail {
            lo = lo + lo - 1.0;
            if !lo.is_finite() {
                return Err(SaddleError::NonFinite("inverse erfc"));
            }
        }
        (lo, 0.0)
    };
    loop {
        let mid = 0.5 * (lo + hi);
        if !(mid > lo && mid < hi) || (hi - lo).abs() <= f64::EPSILON * (1.0 + mid.abs()) {
            if (erfc(mid) - tail).abs() > roundoff(tail) {
                return Err(SaddleError::Solver(
                    "inverse erfc did not reach the tail".into(),
                ));
            }
            return Ok(mid);
        }
        if erfc(mid) > tail {
            lo = mid;
        } else {
            hi = mid;
        }
    }
}

fn assert_close(actual: f64, expected: f64, tolerance: f64, label: &str) {
    let gap = (actual - expected).abs();
    assert!(
        gap <= tolerance,
        "{label}: {actual} against {expected}, gap {gap}, tolerance {tolerance}"
    );
}

fn assert_apart(actual: f64, expected: f64, tolerance: f64, label: &str) {
    let gap = (actual - expected).abs();
    assert!(
        gap > tolerance,
        "{label}: {actual} and {expected} differ by {gap}, inside tolerance {tolerance}"
    );
}

fn gauss_hermite(order: usize) -> Result<(Array1<f64>, Array1<f64>), SaddleError> {
    if order == 0 {
        return Err(SaddleError::Invalid(
            "Gauss-Hermite order starts at one".into(),
        ));
    }
    let mut jacobi = Array2::zeros((order, order));
    for node in 0..order - 1 {
        let sub = ((node as f64 + 1.0) / 2.0).sqrt();
        jacobi[(node, node + 1)] = sub;
        jacobi[(node + 1, node)] = sub;
    }
    let (nodes, vectors) = exact_eigh(jacobi.view())?;
    let mut weights = Array1::zeros(order);
    for node in 0..order {
        let leading = vectors[(0, node)];
        weights[node] = PI.sqrt() * leading * leading;
    }
    Ok((nodes, weights))
}

struct Mode {
    shape: Vec<f64>,
    lambda: f64,
}

fn spring_eigenvalue(beads: usize, beta: f64, wave: usize) -> f64 {
    let sine = (PI * wave as f64 / beads as f64).sin();
    4.0 * beads as f64 / (beta * beta) * sine * sine
}

/// The one-bead toy's centroid hyperplane: `tangent` is the physical axis.
fn toy_plane(beads: usize, centroid: f64) -> Result<Hyperplane, SaddleError> {
    centroid_hyperplane(
        beads,
        ndarray::array![centroid].view(),
        ndarray::array![1.0].view(),
    )
}

/// Orthonormal Fourier modes of a centroid [`Hyperplane`]. The centroid itself is not one of them.
fn relative_modes(plane: &Hyperplane, beta: f64) -> Result<Vec<Mode>, SaddleError> {
    let beads = plane.dim();
    if !strictly_positive(beta) {
        return Err(SaddleError::Invalid(
            "the centroid plane needs a positive beta".into(),
        ));
    }
    let mut modes = Vec::new();
    let half = beads / 2;
    for wave in 1..=half {
        let lambda = spring_eigenvalue(beads, beta, wave);
        if !strictly_positive(lambda) {
            return Err(SaddleError::Invalid(
                "a centroid mode has no spring eigenvalue".into(),
            ));
        }
        let nyquist = wave == half && beads.is_multiple_of(2);
        push_mode(&mut modes, plane, wave, false, lambda)?;
        if !nyquist {
            push_mode(&mut modes, plane, wave, true, lambda)?;
        }
    }
    if modes.len() != beads - 1 {
        return Err(SaddleError::Shape(
            "the centroid plane has one coordinate per bead except the centroid".into(),
        ));
    }
    Ok(modes)
}

fn push_mode(
    modes: &mut Vec<Mode>,
    plane: &Hyperplane,
    wave: usize,
    sine: bool,
    lambda: f64,
) -> Result<(), SaddleError> {
    let beads = plane.dim();
    let mut shape = vec![0.0; beads];
    for (bead, value) in shape.iter_mut().enumerate() {
        let angle = 2.0 * PI * wave as f64 * bead as f64 / beads as f64;
        *value = if sine { angle.sin() } else { angle.cos() };
    }
    let norm = shape.iter().map(|value| value * value).sum::<f64>().sqrt();
    if !strictly_positive(norm) {
        return Err(SaddleError::Invalid("a centroid mode vanished".into()));
    }
    for value in &mut shape {
        *value /= norm;
    }
    let mut shifted = plane.point().to_owned();
    for (value, component) in shifted.iter_mut().zip(&shape) {
        *value += *component;
    }
    let restored = plane.project(shifted.view());
    let mut drift = 0.0_f64;
    for (left, right) in restored.iter().zip(shifted.iter()) {
        if (left - right).abs() > drift {
            drift = (left - right).abs();
        }
    }
    if drift > roundoff(1.0) * beads as f64 {
        return Err(SaddleError::Invalid(
            "a quadrature mode left the centroid hyperplane".into(),
        ));
    }
    modes.push(Mode { shape, lambda });
    Ok(())
}

struct PlaneValue {
    free: f64,
    force: f64,
}

type Toy = fn(f64) -> (f64, f64, f64);

/// Free energy on one centroid [`Hyperplane`] from the closed-form potential.
///
/// The Gaussian is the spring plus the toy's curvature at the centroid.
/// The weight is the Taylor remainder of the toy, evaluated at the point
/// assembled from the plane's orthonormal Fourier modes. No ring surface
/// is evaluated.
fn free_at(
    plane: &Hyperplane,
    beta: f64,
    orders: &[usize],
    potential: Toy,
) -> Result<f64, SaddleError> {
    let modes = relative_modes(plane, beta)?;
    let dim = modes.len();
    let beads = plane.dim();
    let centroid = plane.point()[0];
    let (base, _, curvature) = potential(centroid);
    if dim == 0 {
        return Ok(base);
    }
    if orders.len() != dim {
        return Err(SaddleError::Shape(
            "each centroid mode needs its own quadrature order".into(),
        ));
    }
    let mut rules = Vec::with_capacity(dim);
    for &count in orders {
        let (nodes, weights) = gauss_hermite(count)?;
        let mut ln_weight = Vec::with_capacity(count);
        for weight in weights.iter() {
            if !strictly_positive(*weight) {
                return Err(SaddleError::Invalid(
                    "a Gauss-Hermite weight is not positive".into(),
                ));
            }
            ln_weight.push(weight.ln());
        }
        rules.push((nodes, ln_weight));
    }
    let mut log_prefactor = 0.0;
    let mut width = Vec::with_capacity(dim);
    for mode in &modes {
        let lambda = mode.lambda + curvature / beads as f64;
        if !strictly_positive(lambda) {
            return Err(SaddleError::Invalid(
                "a centroid mode is not bound, so this temperature is at or below the crossover"
                    .into(),
            ));
        }
        log_prefactor += 0.5 * (2.0 / (beta * lambda)).ln();
        width.push((2.0 / (beta * lambda)).sqrt());
    }
    let (_, slope, _) = potential(centroid);
    let mut index = vec![0usize; dim];
    let mut amplitude = vec![0.0; dim];
    let mut coordinate = vec![0.0; beads];
    let mut max_log = 0.0;
    let mut sum = 0.0;
    let mut started = false;
    let scale = 1.0 / beads as f64;
    loop {
        for mode in 0..dim {
            amplitude[mode] = rules[mode].0[index[mode]] * width[mode];
        }
        coordinate.fill(centroid);
        for mode in 0..dim {
            for (value, component) in coordinate.iter_mut().zip(&modes[mode].shape) {
                *value += amplitude[mode] * component;
            }
        }
        let mut remainder = 0.0;
        for value in &coordinate {
            let (energy, _, _) = potential(*value);
            let delta = value - centroid;
            remainder += energy - base - slope * delta - 0.5 * curvature * delta * delta;
        }
        let mut log_w = -beta * (base + scale * remainder);
        for mode in 0..dim {
            log_w += rules[mode].1[index[mode]];
        }
        if !started {
            max_log = log_w;
            sum = 1.0;
            started = true;
        } else if log_w <= max_log {
            sum += (log_w - max_log).exp();
        } else {
            sum = sum * (max_log - log_w).exp() + 1.0;
            max_log = log_w;
        }
        let mut axis = 0;
        while axis < dim {
            index[axis] += 1;
            if index[axis] < orders[axis] {
                break;
            }
            index[axis] = 0;
            axis += 1;
        }
        if axis == dim {
            break;
        }
    }
    if !strictly_positive(sum) {
        return Err(SaddleError::Invalid(
            "the partition function vanished".into(),
        ));
    }
    Ok(-(log_prefactor + max_log + sum.ln()) / beta)
}

/// Free energy and `dF/ds` on the centroid hyperplane. The slope is the central difference of [`free_at`].
fn plane(
    beads: usize,
    beta: f64,
    centroid: f64,
    orders: &[usize],
    potential: Toy,
) -> Result<PlaneValue, SaddleError> {
    let here = toy_plane(beads, centroid)?;
    let free = free_at(&here, beta, orders, potential)?;
    if relative_modes(&here, beta)?.is_empty() {
        let (_, force, _) = potential(centroid);
        return Ok(PlaneValue { free, force });
    }
    let step = f64::EPSILON.cbrt() * (1.0 + centroid.abs());
    let above = free_at(&toy_plane(beads, centroid + step)?, beta, orders, potential)?;
    let below = free_at(&toy_plane(beads, centroid - step)?, beta, orders, potential)?;
    let force = (above - below) / (2.0 * step);
    if !force.is_finite() {
        return Err(SaddleError::NonFinite("centroid mean force"));
    }
    Ok(PlaneValue { free, force })
}

struct Settled {
    barrier: f64,
    /// Quadrature order of each centroid mode. Empty for one bead.
    orders: Vec<usize>,
}

fn barrier_of(
    beads: usize,
    beta: f64,
    reactant: f64,
    saddle: f64,
    orders: &[usize],
    potential: Toy,
) -> Result<f64, SaddleError> {
    Ok(
        free_at(&toy_plane(beads, saddle)?, beta, orders, potential)?
            - free_at(&toy_plane(beads, reactant)?, beta, orders, potential)?,
    )
}

fn settled(
    beads: usize,
    beta: f64,
    reactant: f64,
    saddle: f64,
    potential: Toy,
) -> Result<Settled, SaddleError> {
    if beads == 1 {
        let (left, _, _) = potential(reactant);
        let (right, _, _) = potential(saddle);
        return Ok(Settled {
            barrier: right - left,
            orders: Vec::new(),
        });
    }
    let dim = beads - 1;
    let mut orders = vec![3usize; dim];
    let limit = (1.0 / f64::EPSILON).ln();
    loop {
        let mut refined = false;
        for mode in 0..dim {
            let before = barrier_of(beads, beta, reactant, saddle, &orders, potential)?;
            orders[mode] += 2;
            let nodes = orders.iter().map(|count| (*count as f64).ln()).sum::<f64>();
            if nodes > limit {
                return Err(SaddleError::Solver(
                    "the partition function lost a node below the roundoff of its mass".into(),
                ));
            }
            let after = barrier_of(beads, beta, reactant, saddle, &orders, potential)?;
            let scale = f64::EPSILON.sqrt() * (before.abs() + after.abs() + 1.0 / beta);
            if (after - before).abs() <= scale {
                orders[mode] -= 2;
            } else {
                refined = true;
            }
        }
        if !refined {
            return Ok(Settled {
                barrier: barrier_of(beads, beta, reactant, saddle, &orders, potential)?,
                orders,
            });
        }
    }
}

fn until_stable(
    beads: usize,
    beta: f64,
    centroid: f64,
    potential: Toy,
) -> Result<PlaneValue, SaddleError> {
    if beads == 1 {
        return plane(beads, beta, centroid, &[], potential);
    }
    let mut order = 3usize;
    let mut previous: Option<f64> = None;
    loop {
        let orders = vec![order; beads - 1];
        let value = plane(beads, beta, centroid, &orders, potential)?;
        if let Some(older) = previous {
            let scale = f64::EPSILON.sqrt() * (value.free.abs() + older.abs() + 1.0 / beta);
            if (value.free - older).abs() <= scale {
                return Ok(value);
            }
        }
        previous = Some(value.free);
        let next = order + 2;
        if (next as f64).ln() * (beads - 1) as f64 > (1.0 / f64::EPSILON).ln() {
            return Err(SaddleError::Solver(
                "the partition function lost a node below the roundoff of its mass".into(),
            ));
        }
        order = next;
    }
}

/// `4(x − 1)²`, curvature `8`.
fn quadratic_pair(x: f64) -> (f64, f64, f64) {
    let delta = x - 1.0;
    (4.0 * delta * delta, 8.0 * delta, 8.0)
}

fn gaussian_free(beads: usize, beta: f64, centroid: f64) -> Result<f64, SaddleError> {
    let mut free = quadratic_pair(centroid).0;
    let thermal = 1.0 / beta;
    for mode in relative_modes(&toy_plane(beads, centroid)?, beta)? {
        let lambda = mode.lambda + 8.0 / beads as f64;
        if !strictly_positive(lambda) {
            return Err(SaddleError::Invalid(
                "the quadratic ring is not bound".into(),
            ));
        }
        free += 0.5 * thermal * (beta * lambda / (2.0 * PI)).ln();
    }
    Ok(free)
}

/// Harmonic barrier. Transverse eigenvalues do not depend on the centroid, so they cancel.
fn harmonic_gap(beads: usize, beta: f64, reactant: f64, saddle: f64) -> Result<f64, SaddleError> {
    let potential = quartic(saddle).0 - quartic(reactant).0;
    let mut log_ratio = 0.0;
    for mode in relative_modes(&toy_plane(beads, reactant)?, beta)? {
        let at_saddle = mode.lambda + quartic(saddle).2 / beads as f64;
        let at_reactant = mode.lambda + quartic(reactant).2 / beads as f64;
        if !strictly_positive(at_saddle) || !strictly_positive(at_reactant) {
            return Err(SaddleError::Invalid(
                "a perpendicular eigenvalue is not positive, so this temperature is at or below the crossover".into(),
            ));
        }
        log_ratio += (at_saddle / at_reactant).ln();
    }
    Ok(potential + 0.5 / beta * log_ratio)
}

fn trapezoid(position: &[f64], force: &[f64], end: usize) -> Result<f64, SaddleError> {
    let mut value = 0.0;
    for index in 0..end {
        let step = position[index + 1] - position[index];
        value += 0.5 * (force[index] + force[index + 1]) * step;
    }
    if !value.is_finite() {
        return Err(SaddleError::NonFinite("centroid integral"));
    }
    Ok(value)
}

struct Integrated {
    images: Vec<Array1<f64>>,
    middle: usize,
}

fn integrated(
    found: &Searched,
    beads: usize,
    beta: f64,
    orders: &[usize],
    target: f64,
) -> Result<Integrated, SaddleError> {
    let mut per_leg = 2usize;
    loop {
        let images = path_through(&found.reactant, &found.saddle, &found.product, per_leg)?;
        let middle = per_leg - 1;
        let mut positions = Vec::with_capacity(middle + 1);
        let mut forces = Vec::with_capacity(middle + 1);
        for image in images.iter().take(middle + 1) {
            let value = plane(beads, beta, image[0], orders, quartic_point)?;
            positions.push(image[0]);
            forces.push(value.force);
        }
        let endpoint = plane(beads, beta, found.saddle[0], orders, quartic_point)?.free
            - plane(beads, beta, found.reactant[0], orders, quartic_point)?.free;
        let trap = trapezoid(&positions, &forces, middle)?;
        let truncation = (trap - endpoint).abs();
        if truncation <= target + roundoff(endpoint) {
            return Ok(Integrated { images, middle });
        }
        let next = per_leg
            .checked_mul(2)
            .ok_or_else(|| SaddleError::Solver("the centroid path spacing overflowed".into()))?;
        let span = (found.saddle[0] - found.reactant[0]).abs();
        let spacing = span / (next - 1) as f64;
        if spacing <= found.stationarity {
            return Err(SaddleError::Solver(
                "the centroid path reached the saddle's stationarity before its integral matched the partition function".into(),
            ));
        }
        per_leg = next;
    }
}

fn straight(start: &Array1<f64>, end: &Array1<f64>, count: usize) -> Vec<Array1<f64>> {
    (0..count)
        .map(|index| {
            let fraction = index as f64 / (count - 1) as f64;
            (1.0 - fraction) * start + fraction * end
        })
        .collect()
}

fn path_through(
    reactant: &Array1<f64>,
    saddle: &Array1<f64>,
    product: &Array1<f64>,
    per_leg: usize,
) -> Result<Vec<Array1<f64>>, SaddleError> {
    if per_leg < 2 {
        return Err(SaddleError::Invalid("a path leg needs its two ends".into()));
    }
    let mut images = straight(reactant, saddle, per_leg);
    let middle = per_leg - 1;
    for (value, bit) in images[middle].iter().zip(saddle.iter()) {
        if value.to_bits() != bit.to_bits() {
            return Err(SaddleError::Invalid(
                "the min-mode saddle is not an image of the path".into(),
            ));
        }
    }
    images.extend(straight(saddle, product, per_leg).into_iter().skip(1));
    Ok(images)
}

fn relax<S: PointSurface>(
    surface: &S,
    mut position: Array1<f64>,
) -> Result<Array1<f64>, SaddleError> {
    let controls = RelaxConfig::default().with_method(Method::lbfgs_with_memory(8));
    let n = position.len();
    let mut solver = Solver::new(
        controls.method.clone(),
        Control::default()
            .with_maxiter(controls.max_steps)
            .with_gtol(controls.force_tol)
            .with_istep(controls.initial_step)
            .with_stop_norm(StopNorm::Inf),
        n,
    );
    solver.set_atom_maxmove(controls.max_atom_move);
    let oracle = Oracle::unbounded(n, |x: ArrayView1<f64>| {
        surface
            .eval(x)
            .expect("analytic test surfaces evaluate finite coordinates")
    });
    solver
        .run(&oracle, &mut position)
        .map_err(|error| SaddleError::Solver(error.to_string()))?;
    let (_, gradient) = surface.eval(position.view())?;
    let mut largest = 0.0_f64;
    for value in &gradient {
        if value.abs() > largest {
            largest = value.abs();
        }
    }
    if !(largest.is_finite() && largest <= controls.force_tol) {
        return Err(SaddleError::Solver(format!(
            "relaxation left a force component {largest}"
        )));
    }
    Ok(position)
}

fn index_one<S: PointSurface>(
    surface: &S,
    start: Array1<f64>,
    mode: Array1<f64>,
) -> Result<(Array1<f64>, f64, f64), SaddleError> {
    let mut session = MinModeSession::new(MinModeConfig::default(), start, mode)?;
    let report = session.run(surface, RelaxConfig::default().max_steps)?;
    if report.status != MinModeStatus::Converged {
        return Err(SaddleError::Solver(format!(
            "saddle search ended as {:?}",
            report.status
        )));
    }
    if !strictly_negative(report.curvature) {
        return Err(SaddleError::Solver(format!(
            "saddle curvature {} is not negative",
            report.curvature
        )));
    }
    Ok((
        session.position().to_owned(),
        report.curvature,
        report.max_force,
    ))
}

/// `kT` at which the lowest ring mode is marginal: `λ_spring(k = 1) = −V''/P`.
fn marginal_temperature(
    beads: usize,
    barrier_curvature: f64,
    mass: f64,
    hbar: f64,
) -> Result<f64, SaddleError> {
    if !strictly_negative(barrier_curvature) {
        return Err(SaddleError::Invalid(
            "the marginal temperature needs a negative barrier curvature".into(),
        ));
    }
    let sine = (PI / beads as f64).sin();
    let beta_sq =
        4.0 * mass * (beads as f64).powi(2) * sine * sine / (-barrier_curvature * hbar * hbar);
    Ok(1.0 / beta_sq.sqrt())
}

fn curvature_at<S: PointSurface>(
    surface: &S,
    position: ArrayView1<f64>,
    axis: usize,
) -> Result<f64, SaddleError> {
    let mut direction = Array1::zeros(position.len());
    direction[axis] = 1.0;
    let Some(product) = surface.hessian_vector(position, direction.view())? else {
        return Err(SaddleError::Invalid(
            "curvature was requested without a Hessian product".into(),
        ));
    };
    Ok(product[axis])
}

fn ring_stride(beads: usize, beta: f64, well: f64, barrier: f64) -> Result<usize, SaddleError> {
    let wave = if beads == 1 { 0 } else { 1 };
    let stiff = spring_eigenvalue(beads, beta, beads / 2) + well / beads as f64;
    let soft = spring_eigenvalue(beads, beta, wave) + barrier / beads as f64;
    if !strictly_positive(stiff) || !strictly_positive(soft) {
        return Err(SaddleError::Invalid(
            "the stride needs a positive perpendicular eigenvalue".into(),
        ));
    }
    let ratio = stiff / soft;
    if !ratio.is_finite() {
        return Err(SaddleError::NonFinite("ring stride"));
    }
    Ok(ratio.ceil() as usize)
}

fn sampling(thermal: f64, time_step: f64, stride: usize, seed: u64) -> PafiConfig {
    let blocks = BLOCKS;
    PafiConfig::default()
        .with_thermal(thermal)
        .with_time_step(time_step)
        .with_stride(stride)
        .with_blocks(blocks)
        .with_samples(blocks * blocks * 4)
        .with_thermalize(stride * blocks)
        .with_seed(seed)
        .with_integrator(PafiIntegrator::EulerMaruyama)
        .with_fix_com(false)
        .with_retain(0)
        .with_target_error(None)
}

fn mean_sem(values: &[f64]) -> Result<(f64, f64), SaddleError> {
    if values.len() < 2 {
        return Err(SaddleError::Invalid(
            "a standard error needs two values".into(),
        ));
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / (values.len() - 1) as f64;
    Ok((mean, (variance / values.len() as f64).sqrt()))
}

struct Searched {
    reactant: Array1<f64>,
    product: Array1<f64>,
    saddle: Array1<f64>,
    /// Solver force over the curvature, the stationarity length.
    stationarity: f64,
}

fn locate<S: PointSurface>(
    surface: &S,
    reactant_start: Array1<f64>,
    product_start: Array1<f64>,
    saddle_start: Array1<f64>,
    mode: Array1<f64>,
) -> Result<Searched, SaddleError> {
    let reactant = relax(surface, reactant_start)?;
    let product = relax(surface, product_start)?;
    let (saddle, curvature, force) = index_one(surface, saddle_start, mode)?;
    Ok(Searched {
        reactant,
        product,
        saddle,
        stationarity: force / curvature.abs(),
    })
}

fn working_temperature<S: PointSurface>(
    surface: &S,
    found: &Searched,
) -> Result<(f64, f64, f64), SaddleError> {
    let well = curvature_at(surface, found.reactant.view(), 0)?;
    let barrier = curvature_at(surface, found.saddle.view(), 0)?;
    if !strictly_positive(well) || !strictly_negative(barrier) {
        return Err(SaddleError::Invalid(
            "the working temperature needs a bound well and a barrier".into(),
        ));
    }
    let marginal = marginal_temperature(2, barrier, 1.0, 1.0)?;
    Ok((marginal * (well / -barrier).sqrt(), well, barrier))
}

struct ProfileRun {
    sixteenth_barrier: f64,
    thirty_second_barrier: f64,
    blocking_error: f64,
    effective_samples: f64,
    tau: f64,
}

fn run_profile<S: PointSurface + Sync>(
    surface: &S,
    path: &ArcPath,
    thermal: f64,
    time_step: f64,
    stride: usize,
    seed: u64,
) -> Result<ProfileRun, SaddleError> {
    let seed_stride = (path.arcs().len() as u64).wrapping_add(1);
    let config = sampling(thermal, time_step, stride, seed.wrapping_add(seed_stride));
    let (_profile, planes) = pafi_profile(surface, path, &path.arcs(), &config)?;
    let half = sampling(
        thermal,
        0.5 * time_step,
        stride * 2,
        seed.wrapping_add(seed_stride.wrapping_mul(2)),
    );
    let (_half_profile, half_planes) = pafi_profile(surface, path, &path.arcs(), &half)?;
    let quarter = sampling(
        thermal,
        0.25 * time_step,
        stride * 4,
        seed.wrapping_add(seed_stride.wrapping_mul(3)),
    );
    let (_quarter_profile, quarter_planes) = pafi_profile(surface, path, &path.arcs(), &quarter)?;
    let eighth = sampling(
        thermal,
        0.125 * time_step,
        stride * 8,
        seed.wrapping_add(seed_stride.wrapping_mul(4)),
    );
    let (_eighth_profile, eighth_planes) = pafi_profile(surface, path, &path.arcs(), &eighth)?;
    let sixteenth = sampling(
        thermal,
        0.0625 * time_step,
        stride * 16,
        seed.wrapping_add(seed_stride.wrapping_mul(5)),
    );
    let (sixteenth_profile, sixteenth_planes) =
        pafi_profile(surface, path, &path.arcs(), &sixteenth)?;
    let thirty_second = sampling(
        thermal,
        0.03125 * time_step,
        stride * 32,
        seed.wrapping_add(seed_stride.wrapping_mul(6)),
    );
    let (thirty_second_profile, thirty_second_planes) =
        pafi_profile(surface, path, &path.arcs(), &thirty_second)?;
    let blocking_error =
        (4.0 * thirty_second_profile.barrier_error * thirty_second_profile.barrier_error
            + sixteenth_profile.barrier_error * sixteenth_profile.barrier_error)
            .sqrt();
    let mut observations = planes
        .iter()
        .chain(half_planes.iter())
        .chain(quarter_planes.iter())
        .chain(eighth_planes.iter())
        .chain(sixteenth_planes.iter())
        .chain(thirty_second_planes.iter());
    let Some(first) = observations.next() else {
        return Err(SaddleError::MissingSamples {
            context: "science profile planes",
        });
    };
    let mut effective_samples = first.n_eff;
    let mut tau = first.tau;
    for plane in observations {
        if plane.n_eff < effective_samples {
            effective_samples = plane.n_eff;
        }
        if plane.tau > tau {
            tau = plane.tau;
        }
    }
    Ok(ProfileRun {
        sixteenth_barrier: sixteenth_profile.barrier,
        thirty_second_barrier: thirty_second_profile.barrier,
        blocking_error,
        effective_samples,
        tau,
    })
}

fn assert_coherent(modes: &Array2<f64>, beads: usize, dof: usize) {
    for row in modes.rows() {
        let norm = row.iter().map(|value| value * value).sum::<f64>();
        assert_close(norm, 1.0, roundoff(1.0), "excluded-mode norm");
        for bead in 1..beads {
            for axis in 0..dof {
                assert_close(
                    row[bead * dof + axis],
                    row[axis],
                    roundoff(1.0),
                    "coherent bead",
                );
            }
        }
    }
}

fn project_rows(rows: ArrayView2<f64>, vector: ArrayView1<f64>) -> Array1<f64> {
    let mut projected = Array1::zeros(vector.len());
    for row in rows.rows() {
        let coefficient = row.dot(&vector);
        projected.scaled_add(coefficient, &row);
    }
    projected
}

fn plane_on(
    beads: usize,
    beta: f64,
    orders: &[usize],
    images: &[Array1<f64>],
    middle: usize,
) -> Result<(f64, f64), SaddleError> {
    let mut positions = Vec::with_capacity(middle + 1);
    let mut forces = Vec::with_capacity(middle + 1);
    for image in images.iter().take(middle + 1) {
        let value = plane(beads, beta, image[0], orders, quartic_point)?;
        positions.push(image[0]);
        forces.push(value.force);
    }
    let endpoint = plane(beads, beta, images[middle][0], orders, quartic_point)?.free
        - plane(beads, beta, images[0][0], orders, quartic_point)?.free;
    let trap = trapezoid(&positions, &forces, middle)?;
    Ok((trap, (trap - endpoint).abs()))
}

#[test]
fn inverse_erfc_round_trips() -> Result<(), SaddleError> {
    let tail = 1.0 / (SEEDS as f64).powi(2);
    let guess = inv_erfc(tail)?;
    assert_close(erfc(guess), tail, roundoff(tail), "erfc round trip");
    let (nodes, weights) = gauss_hermite(8)?;
    let mass = weights.sum();
    let second: f64 = nodes
        .iter()
        .zip(weights.iter())
        .map(|(node, weight)| node * node * weight)
        .sum();
    assert_close(mass, PI.sqrt(), roundoff(PI.sqrt()), "hermite mass");
    assert_close(
        second,
        0.5 * PI.sqrt(),
        roundoff(PI.sqrt()),
        "hermite second moment",
    );
    Ok(())
}

#[test]
fn one_bead_is_the_bare_potential() -> Result<(), SaddleError> {
    let ring = RingPolymer::new(&DoubleWell, 1, settings(2.0, 1))?;
    let coordinate = 0.3;
    let (bare, bare_force, _) = quartic(coordinate);
    let (energy, gradient) = ring.eval(ndarray::array![coordinate].view())?;
    let scale = roundoff(bare.abs() + bare_force.abs());
    assert_close(energy, bare, scale, "one-bead energy");
    assert_close(gradient[0], bare_force, scale, "one-bead force");
    let action = ring.action(ndarray::array![coordinate].view())?;
    assert_close(action, 2.0 * bare, scale, "one-bead action");
    Ok(())
}

#[test]
fn spring_matches_central_differences() -> Result<(), SaddleError> {
    let ring = RingPolymer::new(&Flat, 3, settings(1.7, 1))?;
    let coordinate = ndarray::array![0.2, -0.4, 0.7];
    let (energy, gradient) = ring.eval(coordinate.view())?;
    let step = f64::EPSILON.sqrt();
    for index in 0..coordinate.len() {
        let mut plus = coordinate.clone();
        let mut minus = coordinate.clone();
        plus[index] += step;
        minus[index] -= step;
        let (above, above_force) = ring.eval(plus.view())?;
        let (below, below_force) = ring.eval(minus.view())?;
        let slope = (above - below) / (2.0 * step);
        let scale = f64::EPSILON / step * (above.abs() + below.abs() + energy.abs() + 1.0);
        assert_close(slope, gradient[index], scale, "spring gradient");
        let mut direction = Array1::zeros(coordinate.len());
        direction[index] = 1.0;
        let Some(product) = ring.hessian_vector(coordinate.view(), direction.view())? else {
            return Err(SaddleError::Invalid(
                "the flat ring should expose its spring Hessian".into(),
            ));
        };
        let differenced = (&above_force - &below_force) / (2.0 * step);
        for entry in 0..coordinate.len() {
            let force_scale =
                f64::EPSILON / step * (above_force[entry].abs() + below_force[entry].abs() + 1.0);
            assert_close(
                differenced[entry],
                product[entry],
                force_scale,
                "spring Hessian",
            );
        }
    }
    Ok(())
}

#[test]
fn multi_bead_gradient_matches_central_differences() -> Result<(), SaddleError> {
    let ring = RingPolymer::new(
        &Coupled,
        3,
        RingSettings::default().with_masses(Array1::ones(2)),
    )?;
    let coordinate = ndarray::array![0.4, -0.2, -0.5, 0.3, 0.8, 0.1];
    let (energy, gradient) = ring.eval(coordinate.view())?;
    for index in 0..coordinate.len() {
        let mut step = f64::EPSILON.cbrt();
        loop {
            let slope = central(&ring, &coordinate, index, step)?;
            let half = central(&ring, &coordinate, index, step * 0.5)?;
            let truncation = (slope - half).abs() / 3.0;
            let noise = f64::EPSILON * (energy.abs() + 1.0) / (step * 0.5);
            if truncation <= noise {
                assert_close(
                    gradient[index],
                    half,
                    truncation + noise,
                    "multi-bead gradient",
                );
                break;
            }
            step *= 0.5;
            if step * (gradient[index].abs() + 1.0) <= f64::EPSILON * (energy.abs() + 1.0) {
                return Err(SaddleError::Solver(
                    "the gradient difference underflowed before its truncation reached roundoff"
                        .into(),
                ));
            }
        }
    }
    Ok(())
}

fn central(
    ring: &RingPolymer<'_, Coupled>,
    coordinate: &Array1<f64>,
    index: usize,
    step: f64,
) -> Result<f64, SaddleError> {
    let mut plus = coordinate.clone();
    let mut minus = coordinate.clone();
    plus[index] += step;
    minus[index] -= step;
    let (above, _) = ring.eval(plus.view())?;
    let (below, _) = ring.eval(minus.view())?;
    Ok((above - below) / (2.0 * step))
}

#[test]
fn centroid_hyperplane_is_the_shared_plane() -> Result<(), SaddleError> {
    let beads = 4usize;
    let centroid = 0.3;
    let plane = toy_plane(beads, centroid)?;
    assert_close(
        plane.distance(plane.point()),
        0.0,
        roundoff(1.0),
        "the collapsed image is on its hyperplane",
    );
    let coherent = (beads as f64).sqrt().recip();
    for component in plane.normal().iter() {
        assert_close(*component, coherent, roundoff(1.0), "coherent normal");
    }
    let mut shifted = plane.point().to_owned();
    shifted[0] += 0.2;
    shifted[1] -= 0.2;
    let restored = plane.project(shifted.view());
    assert_close(
        plane.distance(restored.view()),
        0.0,
        roundoff(1.0),
        "a relative displacement projects onto the hyperplane",
    );
    Ok(())
}

#[test]
fn fourier_modes_have_the_circulant_eigenvalue() -> Result<(), SaddleError> {
    let beads = 8;
    let beta = 1.5;
    let ring = RingPolymer::new(&DoubleWell, beads, settings(beta, 1))?;
    let collapsed_well = collapsed(beads, ndarray::array![1.0].view())?;
    let (_, _, well) = quartic(1.0);
    for wave in 0..beads / 2 + 1 {
        let mut mode = Array1::zeros(beads);
        for bead in 0..beads {
            mode[bead] = (2.0 * PI * wave as f64 * bead as f64 / beads as f64).cos();
        }
        let norm = mode.dot(&mode).sqrt();
        mode /= norm;
        let Some(product) = ring.hessian_vector(collapsed_well.view(), mode.view())? else {
            return Err(SaddleError::Invalid(
                "the double-well ring should expose a Hessian product".into(),
            ));
        };
        let rayleigh = mode.dot(&product);
        let sine = (PI * wave as f64 / beads as f64).sin();
        let lambda = 4.0 * beads as f64 / (beta * beta) * sine * sine + well / beads as f64;
        let scale = f64::EPSILON * (lambda.abs() + well.abs() + 1.0) * beads as f64;
        assert_close(rayleigh, lambda, scale, "fourier eigenvalue");
    }
    Ok(())
}

#[test]
fn partition_function_matches_the_gaussian() -> Result<(), SaddleError> {
    let beta = 1.5;
    let beads = 4;
    let centroid = 1.2;
    let value = until_stable(beads, beta, centroid, quadratic_pair)?;
    let analytic = gaussian_free(beads, beta, centroid)?;
    let scale = f64::EPSILON.sqrt() * (analytic.abs() + value.free.abs() + 1.0 / beta);
    assert_close(value.free, analytic, scale, "quadratic ring free energy");
    assert_close(
        value.force,
        quadratic_pair(centroid).1,
        scale,
        "quadratic mean force",
    );
    let classical = quartic_point(0.0).0 - quartic_point(-1.0).0;
    assert_close(classical, 1.0, roundoff(1.0), "one-bead partition barrier");
    Ok(())
}

#[test]
fn excluded_modes_are_the_coherent_ring_motion() -> Result<(), SaddleError> {
    let pinned = RingPolymer::new(&Pin, 3, settings(1.4, 2))?;
    let coordinate = ndarray::array![0.2, 0.5, -0.4, -0.2, 0.7, 0.1];
    let modes = pinned.excluded_modes(coordinate.view())?;
    assert_eq!(modes.nrows(), 1);
    assert_coherent(&modes, 3, 2);
    let (energy, _) = pinned.eval(coordinate.view())?;
    let mut shifted = coordinate.clone();
    for (value, mode) in shifted.iter_mut().zip(modes.row(0).iter()) {
        *value += 0.3 * mode;
    }
    let (held, _) = pinned.eval(shifted.view())?;
    assert_close(
        held,
        energy,
        roundoff(energy.abs() + 1.0),
        "coherent motion leaves the spring",
    );
    let mut local = coordinate.clone();
    local[0] += 0.3;
    let (broken, _) = pinned.eval(local.view())?;
    assert_apart(
        broken,
        energy,
        roundoff(energy.abs() + 1.0),
        "a one-bead move is not a spring symmetry",
    );

    let locked = RingPolymer::new(&Lock, 3, settings(1.0, 2))?;
    let both = locked.excluded_modes(coordinate.view())?;
    assert_eq!(both.nrows(), 2);
    assert_coherent(&both, 3, 2);
    let overlap = both.row(0).dot(&both.row(1));
    assert_close(overlap, 0.0, roundoff(1.0), "deduplicated modes");

    let split = RingPolymer::new(&Split, 2, settings(1.0, 2))?;
    let divided = ndarray::array![1.0, 0.0, -1.0, 0.0];
    let none = split.excluded_modes(divided.view())?;
    assert_eq!(none.nrows(), 0);

    let half = RingPolymer::new(&Half, 2, settings(1.0, 2))?;
    let shared = half.excluded_modes(divided.view())?;
    assert_eq!(shared.nrows(), 1);
    assert_coherent(&shared, 2, 2);
    assert_close(
        shared[(0, 1)],
        0.0,
        roundoff(1.0),
        "the unshared axis is not a ring mode",
    );
    Ok(())
}

#[test]
fn excluded_modes_preserve_arbitrary_coherent_subspaces() -> Result<(), SaddleError> {
    let diagonal = RingPolymer::new(&DiagonalExcluded, 1, settings(1.0, 2))?;
    let diagonal_modes = diagonal.excluded_modes(ndarray::array![0.2, -0.4].view())?;
    assert_eq!(diagonal_modes.nrows(), 1);
    let direction = ndarray::array![1.0, 1.0];
    let residual = &direction - &project_rows(diagonal_modes.view(), direction.view());
    assert_close(
        residual.dot(&residual).sqrt(),
        0.0,
        roundoff(direction.dot(&direction).sqrt()),
        "one-bead diagonal excluded mode",
    );

    let coordinate = ndarray::array![
        -0.9, 0.1, 0.0, 0.9, -0.1, 0.0, -0.7, 0.3, 0.2, 1.0, -0.2, -0.1
    ];
    let weighted_one = RingPolymer::new(&MassWeightedTranslations, 1, settings(1.0, 6))?;
    let one_modes = weighted_one.excluded_modes(coordinate.slice(s![..6]))?;
    assert_eq!(one_modes.nrows(), 3);

    let weighted = RingPolymer::new(&MassWeightedTranslations, 2, settings(1.0, 6))?;
    let weighted_modes = weighted.excluded_modes(coordinate.view())?;
    assert_eq!(weighted_modes.nrows(), 3);
    assert_coherent(&weighted_modes, 2, 6);
    Ok(())
}

#[test]
fn excluded_modes_keep_shared_rigid_generators() -> Result<(), SaddleError> {
    let ring = RingPolymer::new(&RigidBody, 2, settings(1.0, 6))?;
    let bead_a = array![-0.9, 0.1, 0.0, 0.9, -0.1, 0.0];
    let bead_b = array![-0.7, 0.3, 0.2, 1.0, -0.2, -0.1];
    let mut x = Array1::zeros(12);
    x.slice_mut(s![..6]).assign(&bead_a);
    x.slice_mut(s![6..]).assign(&bead_b);
    let modes = ring.excluded_modes(x.view())?;
    assert_eq!(modes.nrows(), 7);
    let mut rotation = Array1::zeros(12);
    for bead in 0..2 {
        for atom in 0..2 {
            let base = bead * 6 + 3 * atom;
            rotation[base] = -x[base + 1];
            rotation[base + 1] = x[base];
            rotation[base + 2] = 0.0;
        }
    }
    let projected = project_rows(modes.view(), rotation.view());
    let residual = (&rotation - &projected)
        .dot(&(&rotation - &projected))
        .sqrt();
    assert!(
        residual <= roundoff(1.0) * (1.0 + rotation.dot(&rotation).sqrt()),
        "shared rigid rotation left the coherent ring space"
    );
    Ok(())
}

#[test]
fn anisotropic_masses_do_not_exclude_rigid_rotations() -> Result<(), SaddleError> {
    let anisotropic = RingSettings::default()
        .with_beta(1.0)
        .with_hbar(1.0)
        .with_masses(array![1.0, 2.0, 1.0, 1.0, 2.0, 1.0]);
    let ring = RingPolymer::new(&RigidBody, 2, anisotropic)?;
    let x = array![
        -0.9, 0.1, 0.0, 0.9, -0.1, 0.0, -0.7, 0.3, 0.2, 1.0, -0.2, -0.1
    ];
    let modes = ring.excluded_modes(x.view())?;
    let mut rotation = Array1::zeros(12);
    for bead in 0..2 {
        for atom in 0..2 {
            let base = bead * 6 + 3 * atom;
            rotation[base] = -x[base + 1];
            rotation[base + 1] = x[base];
        }
    }
    let projected = project_rows(modes.view(), rotation.view());
    let residual = &rotation - &projected;
    assert!(
        residual.dot(&residual).sqrt() > 1.0e-3,
        "anisotropic spring rotation must not be an excluded zero mode"
    );
    Ok(())
}

#[test]
fn time_step_tracks_the_stiffest_spring() -> Result<(), SaddleError> {
    let beta = 2.0_f64.sqrt();
    let images = [
        ndarray::array![1.0],
        ndarray::array![0.0],
        ndarray::array![-1.0],
    ];
    let mobility = PafiConfig::default().mobility;
    let mut coarser = 0.0;
    for beads in [1usize, 2, 4, 8] {
        let ring = RingPolymer::new(&DoubleWell, beads, settings(beta, 1))?;
        let step = ring.time_step(mobility, &images)?;
        let lambda = spring_eigenvalue(beads, beta, beads / 2) + 8.0 / beads as f64;
        let expect = 1.0 / (mobility * lambda);
        assert_close(step, expect, roundoff(expect), "spring time step");
        if beads == 2 {
            coarser = step;
        }
        if beads == 8 {
            assert!(
                step < coarser,
                "the stiffer eight-bead spring takes a shorter step: {step} against {coarser}"
            );
        }
    }
    Ok(())
}

#[test]
fn time_step_uses_a_gershgorin_upper_bound() -> Result<(), SaddleError> {
    let ring = RingPolymer::new(&DenseQuadratic, 1, settings(1.0, 2))?;
    let mobility = PafiConfig::default().mobility;
    let step = ring.time_step(mobility, &[ndarray::array![0.0, 0.0]])?;
    let expected = 1.0 / (7.0 * mobility);
    assert_close(step, expected, roundoff(expected), "Gershgorin time step");
    Ok(())
}

#[test]
fn time_step_requires_images_and_hessian_products() -> Result<(), SaddleError> {
    let ring = RingPolymer::new(&NoHessian, 1, settings(1.0, 1))?;
    let mobility = PafiConfig::default().mobility;
    let no_images = ring.time_step(mobility, &[]).unwrap_err();
    assert!(matches!(no_images, SaddleError::MissingCurvatureImages));
    let no_hessian = ring
        .time_step(mobility, &[ndarray::array![0.0]])
        .unwrap_err();
    assert!(matches!(
        no_hessian,
        SaddleError::MissingHessianVector { image: 0 }
    ));
    Ok(())
}

#[test]
fn ring_arithmetic_rejects_nonfinite_results_and_underflow() -> Result<(), SaddleError> {
    let underflow = RingSettings::default()
        .with_beta(1.0e-200)
        .with_hbar(1.0)
        .with_masses(array![1.0]);
    assert!(matches!(
        RingPolymer::new(&Flat, 2, underflow.clone()),
        Err(SaddleError::NonFinite("ring-polymer spring thermal scale"))
    ));

    let one = RingPolymer::new(&Flat, 1, underflow)?;
    let (energy, gradient) = one.eval(array![0.0].view())?;
    assert_eq!(energy, 0.0);
    assert_eq!(gradient, array![0.0]);
    assert_eq!(
        one.hessian_vector(array![0.0].view(), array![1.0].view())?
            .expect("flat Hessian"),
        array![0.0]
    );

    let ordinary = RingPolymer::new(&Flat, 2, settings(1.0, 1))?;
    assert!(matches!(
        ordinary.eval(array![1.0e308, -1.0e308].view()),
        Err(SaddleError::NonFinite("ring-polymer energy"))
    ));
    assert!(matches!(
        ordinary.hessian_vector(array![0.0, 0.0].view(), array![1.0e308, -1.0e308].view()),
        Err(SaddleError::NonFinite("ring-polymer Hessian product"))
    ));

    let action = RingPolymer::new(
        &ConstantTwo,
        1,
        RingSettings::default()
            .with_beta(f64::MAX)
            .with_masses(array![1.0]),
    )?;
    assert!(matches!(
        action.action(array![0.0].view()),
        Err(SaddleError::NonFinite("ring-polymer action"))
    ));

    let dense = RingPolymer::new(&DenseQuadratic, 1, settings(1.0, 2))?;
    assert!(matches!(
        dense.time_step(f64::MAX, &[array![0.0, 0.0]]),
        Err(SaddleError::NonFinite("ring-polymer time step"))
    ));
    Ok(())
}

#[test]
fn bead_study_rejects_thermal_mismatch() -> Result<(), SaddleError> {
    let images = [array![-1.0], array![0.0], array![1.0]];
    let settings = settings(2.0, 1);
    let config = PafiConfig::default().with_thermal(0.7).with_fix_com(false);
    let error = bead_study(&DoubleWell, &images, &[1, 2], &settings, &config).unwrap_err();
    assert!(
        matches!(error, SaddleError::TemperatureMismatch { .. }),
        "bead study should report thermal/beta mismatch"
    );
    Ok(())
}

#[test]
fn bead_study_temperature_check_is_relative() {
    let images = [array![-1.0], array![0.0], array![1.0]];
    let settings = settings(1.0e9, 1);
    let config = PafiConfig::default()
        .with_thermal(1.0e-8)
        .with_fix_com(false);
    let error = bead_study(&DoubleWell, &images, &[1], &settings, &config)
        .expect_err("a tenfold thermal mismatch must not pass at small energy scales");
    assert!(matches!(error, SaddleError::TemperatureMismatch { .. }));
}

#[test]
fn bead_study_rejects_cartesian_com_for_low_dimensional_beads() {
    let images = [array![-1.0], array![0.0], array![1.0]];
    let settings = settings(1.0, 1);
    let error = bead_study(
        &DoubleWell,
        &images,
        &[1],
        &settings,
        &PafiConfig::default(),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        SaddleError::UnsupportedRingCenterOfMass { dof: 1 }
    ));
}

#[test]
fn direct_ring_pafi_rejects_com_for_wrapped_scalar_beads() -> Result<(), SaddleError> {
    let surface = Flat;
    let beads = 6;
    let ring = RingPolymer::new(&surface, beads, settings(1.0, 1))?;
    let images = [ndarray::array![-1.0], ndarray::array![1.0]];
    let path = centroid_path(&images, beads)?;
    let config = sampling(1.0, 0.01, 1, 3)
        .with_samples(4)
        .with_blocks(2)
        .with_fix_com(true)
        .with_com_atoms(NonZeroUsize::new(2).expect("two atoms"));
    let error = pafi_profile(&ring, &path, &path.arcs(), &config).unwrap_err();
    assert!(matches!(
        error,
        SaddleError::UnsupportedPafiComLayout {
            layout: CartesianLayout::Wrapped
        }
    ));
    Ok(())
}

#[test]
fn bead_study_rejects_steps_above_the_ring_bound() -> Result<(), SaddleError> {
    let images = [array![-1.0], array![0.0], array![1.0]];
    let settings = settings(2.0, 1);
    let ring = RingPolymer::new(&DoubleWell, 2, settings.clone())?;
    let bound = ring.time_step(PafiConfig::default().mobility, &images)?;
    let config = PafiConfig::default()
        .with_thermal(1.0 / settings.beta)
        .with_time_step(bound * 1.5)
        .with_fix_com(false);
    let error = bead_study(&DoubleWell, &images, &[2], &settings, &config).unwrap_err();
    assert!(matches!(error, SaddleError::TimeStepAboveBound { .. }));
    Ok(())
}

#[test]
fn rejects_a_ring_without_beads() {
    match RingPolymer::new(&Flat, 0, settings(1.0, 1)) {
        Err(SaddleError::Invalid(_)) => {}
        Err(error) => panic!("an empty ring failed as {error}"),
        Ok(_) => panic!("an empty ring was accepted"),
    }
}

#[cfg(not(debug_assertions))]
#[test]
fn bead_study_tracks_the_partition_function() -> Result<(), SaddleError> {
    let surface = DoubleWell;
    let found = locate(
        &surface,
        ndarray::array![-1.4],
        ndarray::array![1.25],
        ndarray::array![0.2],
        ndarray::array![1.0],
    )?;
    let (thermal, well, barrier_curvature) = working_temperature(&surface, &found)?;
    let beta = 1.0 / thermal;
    let z = coverage(SEEDS)?;
    let counts = [1usize, 2, 4, 8];
    let finest = *counts
        .last()
        .ok_or_else(|| SaddleError::Invalid("the bead study lists its counts".into()))?;
    let classical = settled(1, beta, found.reactant[0], found.saddle[0], quartic_point)?;
    let quantum = settled(
        finest,
        beta,
        found.reactant[0],
        found.saddle[0],
        quartic_point,
    )?;
    let separation = (classical.barrier - quantum.barrier).abs();
    if !strictly_positive(separation) {
        return Err(SaddleError::Solver(
            "the partition function did not separate the classical and finest rings".into(),
        ));
    }
    let stored = settings(beta, 1);
    let mobility = PafiConfig::default().mobility;
    let stride = ring_stride(finest, beta, well, barrier_curvature)?;
    let draws = (SEEDS * SEEDS * 4) as f64;
    let mut target = thermal / draws.sqrt() / z;
    let mut path = integrated(&found, finest, beta, &quantum.orders, target)?;
    let mut samples = {
        let ring = RingPolymer::new(&surface, finest, stored.clone())?;
        let step = ring.time_step(mobility, &path.images)?;
        bead_study(
            &surface,
            &path.images,
            &counts,
            &stored,
            &sampling(thermal, step, stride, 1),
        )?
    };
    loop {
        let mut unresolved = None;
        for sample in &samples {
            if sample.beads == 1 {
                continue;
            }
            let reference = settled(
                sample.beads,
                beta,
                found.reactant[0],
                found.saddle[0],
                quartic_point,
            )?;
            let (_, truncation) = plane_on(
                sample.beads,
                beta,
                &reference.orders,
                &path.images,
                path.middle,
            )?;
            let error = sample.profile.barrier_error;
            if error > roundoff(sample.profile.barrier) && truncation > error {
                unresolved = match unresolved {
                    Some(value) if value < error => Some(value),
                    _ => Some(error),
                };
            }
        }
        let Some(next) = unresolved else {
            break;
        };
        let previous = path.images.len();
        target = if next < target { next } else { target * 0.5 };
        path = integrated(&found, finest, beta, &quantum.orders, target)?;
        if path.images.len() <= previous {
            return Err(SaddleError::Solver(
                "the centroid path could not bring its truncation under the sampling error".into(),
            ));
        }
        let ring = RingPolymer::new(&surface, finest, stored.clone())?;
        let step = ring.time_step(mobility, &path.images)?;
        samples = bead_study(
            &surface,
            &path.images,
            &counts,
            &stored,
            &sampling(thermal, step, stride, 1),
        )?;
    }
    let mut classical_pafi = None;
    let mut quantum_pafi = None;
    let mut quantum_error = 0.0;
    for sample in &samples {
        let reference = settled(
            sample.beads,
            beta,
            found.reactant[0],
            found.saddle[0],
            quartic_point,
        )?;
        let (trap, truncation) = plane_on(
            sample.beads,
            beta,
            &reference.orders,
            &path.images,
            path.middle,
        )?;
        let barrier = sample.profile.barrier;
        let error = sample.profile.barrier_error;
        let gap = (barrier - reference.barrier).abs();
        let integral_gap = (barrier - trap).abs();
        let resolved = error > roundoff(barrier);
        let endpoint_ratio = if resolved { gap / error } else { gap };
        let integral_ratio = if resolved {
            integral_gap / error
        } else {
            integral_gap
        };
        if resolved {
            eprintln!(
                "P={} orders {:?} centroid PMF {barrier} ± {error} partition {} integral {trap} truncation {truncation} dt {} dt_error {} n_eff {} tau {} integral/error {integral_ratio} endpoint/error {endpoint_ratio}",
                sample.beads,
                reference.orders,
                reference.barrier,
                sample.time_step,
                sample.step_error,
                sample.effective_samples,
                sample.tau,
            );
        } else {
            eprintln!(
                "P={} orders {:?} centroid PMF {barrier} ± {error} partition {} integral {trap} truncation {truncation} dt {} dt_error {} n_eff {} tau {} integral gap {integral_ratio} endpoint gap {endpoint_ratio}",
                sample.beads,
                reference.orders,
                reference.barrier,
                sample.time_step,
                sample.step_error,
                sample.effective_samples,
                sample.tau,
            );
        }
        if sample.beads == 1 {
            assert_close(
                barrier,
                trap,
                roundoff(barrier.abs() + trap.abs()),
                "one-bead centroid PMF against the integral of its derivative",
            );
            assert!(
                gap <= truncation + roundoff(barrier.abs() + reference.barrier.abs()),
                "one-bead centroid PMF {barrier} against the partition function {}, gap {gap}, truncation {truncation}",
                reference.barrier
            );
        } else {
            // The reported barrier is the two-step linear extrapolation.
            // Its error includes both block errors and the step difference.
            let allowance = z * error + roundoff(barrier.abs() + trap.abs());
            assert!(
                integral_gap <= allowance,
                "centroid PMF {barrier} ± {error} is {integral_gap} from the partition-function integral {trap}, beyond allowance {allowance}"
            );
        }
        if sample.beads == 1 {
            classical_pafi = Some(barrier);
        }
        if sample.beads == finest {
            quantum_pafi = Some(barrier);
            quantum_error = error;
        }
    }
    let classical_pafi = classical_pafi
        .ok_or_else(|| SaddleError::Invalid("the study includes the classical ring".into()))?;
    let quantum_pafi = quantum_pafi
        .ok_or_else(|| SaddleError::Invalid("the study includes its finest ring".into()))?;
    assert_apart(
        classical.barrier,
        quantum.barrier,
        roundoff(classical.barrier.abs() + quantum.barrier.abs()),
        "classical partition barrier against the finest",
    );
    assert_apart(
        classical_pafi,
        quantum_pafi,
        z * quantum_error + roundoff(classical_pafi.abs() + quantum_pafi.abs()),
        "classical centroid PMF against the finest",
    );
    Ok(())
}

#[test]
fn anharmonic_correction_against_the_harmonic_instanton() -> Result<(), SaddleError> {
    let resolution = anharmonic_correction(SEEDS)?;
    match resolution {
        CorrectionResolution::Unresolved {
            correction,
            uncertainty,
        } => assert!(
            correction.abs() <= uncertainty,
            "typed unresolved correction {correction} exceeds uncertainty {uncertainty}"
        ),
        CorrectionResolution::Resolved {
            correction,
            uncertainty,
        } => assert!(
            correction.abs() > uncertainty,
            "typed resolved correction {correction} is inside uncertainty {uncertainty}"
        ),
    }
    Ok(())
}

#[derive(Debug)]
enum CorrectionResolution {
    Resolved { correction: f64, uncertainty: f64 },
    Unresolved { correction: f64, uncertainty: f64 },
}

fn anharmonic_correction(seeds: usize) -> Result<CorrectionResolution, SaddleError> {
    let surface = Coupled;
    let found = locate(
        &surface,
        ndarray::array![-1.35, 0.55],
        ndarray::array![1.15, -0.4],
        ndarray::array![0.25, 0.35],
        ndarray::array![1.0, 0.15],
    )?;
    let stationarity = found.stationarity + roundoff(found.stationarity);
    for (label, point) in [
        ("reactant", &found.reactant),
        ("saddle", &found.saddle),
        ("product", &found.product),
    ] {
        if point[1].abs() > stationarity {
            return Err(SaddleError::Solver(format!(
                "{label} left the barrier axis by {}",
                point[1]
            )));
        }
    }
    let (thermal, well, barrier_curvature) = working_temperature(&surface, &found)?;
    let beta = 1.0 / thermal;
    let z = coverage(SEEDS)?;
    let beads = 2;
    let reference = settled(
        beads,
        beta,
        found.reactant[0],
        found.saddle[0],
        quartic_point,
    )?;
    let harmonic = harmonic_gap(beads, beta, found.reactant[0], found.saddle[0])?;
    let correction = reference.barrier - harmonic;
    if !strictly_positive(correction.abs()) {
        return Err(SaddleError::Solver(
            "the partition function and the harmonic barrier did not separate".into(),
        ));
    }
    let draws = (BLOCKS * BLOCKS * 4) as f64;
    let mut target = thermal / draws.sqrt() / z;
    let mut path = integrated(&found, beads, beta, &reference.orders, target)?;
    let stored = settings(beta, 2);
    let ring = RingPolymer::new(&surface, beads, stored)?;
    let collapsed_saddle = collapsed(beads, found.saddle.view())?;
    let mut split = collapsed_saddle.clone();
    split[0] += MinModeConfig::default().dr;
    let mut ring_mode = Array1::zeros(split.len());
    for bead in 0..beads {
        ring_mode[bead * 2] = 1.0;
    }
    let (ring_saddle, ring_curvature, ring_force) = index_one(&ring, split, ring_mode)?;
    let allowance = found.stationarity + ring_force / ring_curvature.abs();
    let mut drift = 0.0_f64;
    for (left, right) in ring_saddle.iter().zip(collapsed_saddle.iter()) {
        if (left - right).abs() > drift {
            drift = (left - right).abs();
        }
    }
    assert!(
        drift <= allowance + roundoff(allowance),
        "ring saddle left the collapsed instanton by {drift}, allowance {allowance}"
    );
    let mobility = PafiConfig::default().mobility;
    let stride = ring_stride(beads, beta, well, barrier_curvature)?;
    let (
        mean,
        sem,
        trap,
        truncation,
        step_uncertainty,
        step_sampling_sem,
        seed_sem,
        blocking_sem,
        n_eff,
        tau,
    ) = loop {
        let step = ring.time_step(mobility, &path.images)?;
        let (mean, sem, step_uncertainty, step_sampling_sem, seed_sem, blocking_sem, n_eff, tau) = {
            let curve = centroid_path(&path.images, beads)?;
            let mut barriers = Vec::with_capacity(seeds);
            let mut step_differences = Vec::with_capacity(seeds);
            let mut block_variance = 0.0_f64;
            let mut n_eff = 0.0_f64;
            let mut tau = 0.0_f64;
            for seed in 0..seeds {
                let run = run_profile(&ring, &curve, thermal, step, stride, seed as u64)?;
                barriers.push(2.0 * run.thirty_second_barrier - run.sixteenth_barrier);
                step_differences.push(run.thirty_second_barrier - run.sixteenth_barrier);
                block_variance += run.blocking_error * run.blocking_error;
                n_eff += run.effective_samples;
                if run.tau > tau {
                    tau = run.tau;
                }
            }
            let (mean, seed_sem) = mean_sem(&barriers)?;
            let (step_difference, step_sampling_sem) = mean_sem(&step_differences)?;
            let step_uncertainty = step_difference.abs();
            let blocking_sem = block_variance.sqrt() / seeds as f64;
            let total = (seed_sem * seed_sem
                + step_uncertainty * step_uncertainty
                + step_sampling_sem * step_sampling_sem)
                .sqrt();
            (
                mean,
                total,
                step_uncertainty,
                step_sampling_sem,
                seed_sem,
                blocking_sem,
                n_eff,
                tau,
            )
        };
        let (trap, truncation) =
            plane_on(beads, beta, &reference.orders, &path.images, path.middle)?;
        if sem > roundoff(mean) && truncation > sem {
            let previous = path.images.len();
            target = if sem < target { sem } else { target * 0.5 };
            path = integrated(&found, beads, beta, &reference.orders, target)?;
            if path.images.len() <= previous {
                return Err(SaddleError::Solver(
                    "the centroid path could not bring its truncation under the sampling error"
                        .into(),
                ));
            }
            continue;
        }
        break (
            mean,
            sem,
            trap,
            truncation,
            step_uncertainty,
            step_sampling_sem,
            seed_sem,
            blocking_sem,
            n_eff,
            tau,
        );
    };
    let reported = mean - harmonic;
    let exact = reference.barrier - harmonic;
    let action = beta * (quartic(found.saddle[0]).0 - quartic(found.reactant[0]).0);
    let fluctuation = harmonic - (quartic(found.saddle[0]).0 - quartic(found.reactant[0]).0);
    let gap = (mean - reference.barrier).abs();
    let ratio = if strictly_positive(sem) {
        gap / sem
    } else {
        gap
    };
    let estimated_resolution_seeds =
        (seeds as f64 * (seed_sem / exact.abs()) * (seed_sem / exact.abs())).ceil() as usize;
    eprintln!(
        "n {seeds}; anharmonic correction {reported} ± {sem}; partition correction {exact}; integral {trap}; truncation {truncation}; seed SEM {seed_sem}; blocking SEM {blocking_sem}; step uncertainty {step_uncertainty}; step SEM {step_sampling_sem}; effective samples {n_eff}; tau {tau}; estimated seeds for one-SEM resolution {estimated_resolution_seeds}; S/hbar {action}; fluctuation {fluctuation}; kT {thermal}; gap/error {ratio}"
    );
    // The profile is extrapolated from dt/16 and dt/32. Its reported
    // uncertainty combines independent-seed, refinement-step, and
    // refinement-step-mean errors.
    let harmonic_distance = (harmonic - reference.barrier).abs() + z * sem;
    assert!(
        gap < harmonic_distance,
        "centroid PMF {mean} ± {sem} is {gap} from the partition function {}, farther than the harmonic barrier",
        reference.barrier
    );
    let correction_floor = z * sem + truncation + roundoff(reported);
    if reported.abs() > correction_floor {
        Ok(CorrectionResolution::Resolved {
            correction: reported,
            uncertainty: correction_floor,
        })
    } else {
        Ok(CorrectionResolution::Unresolved {
            correction: reported,
            uncertainty: correction_floor,
        })
    }
}

#[cfg(feature = "readcon")]
#[test]
fn con_orbit_is_a_ring_path() -> Result<(), SaddleError> {
    let symbols = ["H"];
    let masses = [1.0];
    let species = ConSpecies {
        symbols: &symbols,
        masses: &masses,
        cell: None,
    };
    let beads = 2;
    let images = [-1.0, -0.5, 0.0, 0.5, 1.0];
    let mut frames = Vec::new();
    let mut owned = Vec::new();
    for image in images {
        let fraction = 1.0 / (beads as f64 + 1.0);
        for bead in 0..beads {
            let spread = if bead == 0 { -fraction } else { fraction };
            owned.push(vec![image * (1.0 + spread), 0.0, 0.0]);
        }
    }
    for frame in &owned {
        frames.push(ConImage {
            positions: frame.as_slice(),
            energy: None,
            forces: None,
            neb_bead: None,
        });
    }
    let file = std::env::temp_dir().join(format!(
        "rgsaddle-ring-orbit-{}-{}.con",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    write_band(&file, &species, &frames)?;
    let (path, orbit) = rgsaddle::ring_path_from_con(&file, beads)?;
    let remove = std::fs::remove_file(&file);
    remove.map_err(|error| SaddleError::Surface(error.to_string()))?;
    assert_eq!(orbit.beads, beads);
    assert_eq!(orbit.masses.len(), 3);
    for mass in orbit.masses.iter() {
        assert_eq!(mass.to_bits(), 1.0_f64.to_bits());
    }
    let flat: Vec<f64> = owned.iter().take(beads).flatten().copied().collect();
    assert_eq!(orbit.orbit.len(), flat.len());
    for (value, bit) in orbit.orbit.iter().zip(&flat) {
        assert_eq!(value.to_bits(), bit.to_bits());
    }
    assert_eq!(path.arcs().len(), images.len());
    Ok(())
}
