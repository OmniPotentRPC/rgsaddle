//! Harmonic quantum transition-state theory on a line-integral path.
//!
//! Replicas sit at `t_j = j τ / P`. The ring-polymer Hessian in
//! mass-weighted coordinates is
//!
//! `(Hv)_i = κ (2 v_i − v_{i−1} − v_{i+1}) + ∇²V(q_i) v_i`,
//!
//! `κ = 1/(β_N² ℏ²)`, `β_N = τ / (ℏ P)`. Its fluctuation factor is the
//! product of the absolute eigenvalues except the permutation mode (and,
//! for a molecule, the rigid motions).
//!
//! [`hqtst_rate`] reads the surface through Hessian-vector products at
//! the replicas: the surface's own product when it has one, otherwise a
//! central gradient difference. Replicas `j` and `P − j` sit at the same
//! point of the retraced spline, so `⌊P/2⌋ + 1` replicas are evaluated,
//! each with one gradient and `dof` products for its mass-weighted
//! Hessian block. The ring operator is applied from those blocks and its
//! spectrum on the complement of the gauge modes comes from the shared
//! full-reorthogonalization Lanczos. Surface cost is linear in `P`.
//!
//! With the `lineb-interpolated-hessian` feature, image Hessians are interpolated onto
//! the ring. A bordered block elimination computes the projected log
//! absolute determinant and negative-mode count. An ill-conditioned
//! bead pivot selects the dense orthogonal projection.
//!
//! The rate times the reactant partition function is Richardson's
//! ring-polymer instanton expression (Int. Rev. Phys. Chem. 37, 171
//! (2018), eq. 52), which is the harmonic quantum TST prefactor:
//!
//! `k Z_r = Λ_N^{−1} (β_N ℏ)^{−1} √B_N  Π' (β_N ℏ η_k)^{−1}  exp(−β_N U)`.

use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(any(test, feature = "lineb-interpolated-hessian"))]
use ndarray::ArrayView2;
use ndarray::{Array1, Array2, ArrayView1, s};

use super::action::l2;
use super::align::rigid_directions;
use super::mass::Masses;
use super::optimize::LinebPath;
use super::path;
#[cfg(any(test, feature = "lineb-interpolated-hessian"))]
use super::path::Curve;
#[cfg(all(
    any(test, feature = "lineb-interpolated-hessian"),
    not(feature = "faer")
))]
use crate::eigensolve::exact_eigh;
use crate::error::SaddleError;
use crate::linalg::numerical_hvp;
use crate::mass_metric::MassMetric;
use crate::minmode::PointSurface;
use crate::sparse_hessian::lanczos_spectrum;
#[cfg(any(test, feature = "lineb-interpolated-hessian"))]
use crate::spline::Spline;

/// Ring-polymer instanton rate at the temperature of a tunneling path.
#[derive(Clone, Debug)]
pub struct HqtstRate {
    /// `ℏ / (k_B τ)` of the replica period.
    pub temperature: f64,
    /// Full imaginary-time period `τ`.
    pub period: f64,
    /// Euclidean action over `ℏ` from the line integral, `(2W + E τ) / ℏ`.
    pub line_action: f64,
    /// `β_N U_N` of the replicas.
    pub bead_action: f64,
    /// `Σ ln|λ|` over the kept eigenvalues of the ring-polymer Hessian.
    pub log_abs_det: f64,
    /// Number of strictly negative eigenvalues on the gauge-deflated space.
    pub negative_modes: usize,
    /// Cartesian displacement of the image-Hessian central difference, in the input length units.
    pub image_hessian_displacement: Option<f64>,
    /// How many eigenvalues that sum keeps.
    pub modes: usize,
    /// `ln(k Z_r)`.
    pub log_rate_over_reactant: f64,
    /// `k Z_r`, the exponential of the log. It carries `exp(−β_N U)` on
    /// the surface's own energy zero, so on a total energy far from zero
    /// it overflows to infinity or underflows to zero; the log is finite
    /// whenever this is returned.
    pub rate_over_reactant: f64,
    /// Surface geometries evaluated to form curvature blocks.
    pub curvature_evaluations: usize,
    /// Geometries evaluated for the replica energies and gradients.
    pub bead_evaluations: usize,
    /// Surface evaluations used to build the factor.
    pub evaluations: usize,
}

/// Harmonic quantum TST rate from Hessian-vector products at the
/// replicas. `beads` is `P` in `t_j = j τ / P`.
pub fn hqtst_rate<S: PointSurface>(
    surface: &S,
    path: &LinebPath<impl Sized>,
    beads: usize,
) -> Result<HqtstRate, SaddleError> {
    hqtst_rate_observed(surface, path, beads, |_, _, _| {})
}

/// Equal-time Cartesian configurations and energy gradients sampled for the rate.
#[derive(Clone, Debug)]
pub struct LinebRateOrbit {
    /// Rows cover one full period at the coordinates used by the rate evaluator.
    pub positions: Array2<f64>,
    /// Cartesian gradients in energy per length, in the same order as the positions.
    pub gradients: Array2<f64>,
    /// Largest absolute Cartesian component of the discrete equation of motion.
    pub max_abs_residual: f64,
    /// RMS residual over all Cartesian components of the ring.
    pub rms_residual: f64,
    /// Zero-based bead index of the maximum.
    pub max_bead: usize,
    /// Zero-based Cartesian component of the maximum.
    pub max_coordinate: usize,
}

/// Evaluate the harmonic rate and retain the bead observations used in it.
/// This performs the same surface calls as [`hqtst_rate`].
pub fn hqtst_rate_with_orbit<S: PointSurface>(
    surface: &S,
    path: &LinebPath<impl Sized>,
    beads: usize,
) -> Result<(HqtstRate, LinebRateOrbit), SaddleError> {
    let dof = path.masses.dof();
    let mut measured = Vec::new();
    let rate = hqtst_rate_observed(surface, path, beads, |_, x, g| {
        measured.push((x.to_owned(), g.to_owned()));
    })?;
    let mut positions = Array2::zeros((beads, dof));
    let mut gradients = Array2::zeros((beads, dof));
    for j in 0..beads {
        let (x, g) = &measured[j.min(beads - j)];
        positions.row_mut(j).assign(x);
        gradients.row_mut(j).assign(g);
    }
    let orbit = orbit_residual(path, rate.period, positions, gradients)?;
    Ok((rate, orbit))
}

fn orbit_residual(
    path: &LinebPath<impl Sized>,
    period: f64,
    positions: Array2<f64>,
    gradients: Array2<f64>,
) -> Result<LinebRateOrbit, SaddleError> {
    let (beads, dof) = positions.dim();
    let spring = (beads as f64 / period).powi(2);
    let mut rms_norm = 0.0_f64;
    let mut maximum = (0.0, 0, 0);
    for j in 0..beads {
        let prev = (j + beads - 1) % beads;
        let next = (j + 1) % beads;
        for k in 0..dof {
            let x = positions[(j, k)];
            let curvature = (x - positions[(prev, k)]) + (x - positions[(next, k)]);
            let residual = gradients[(j, k)] + path.masses.sqrt_at(k).powi(2) * spring * curvature;
            if !residual.is_finite() {
                return Err(SaddleError::NonFinite("replica force residual"));
            }
            rms_norm = rms_norm.hypot(residual);
            if residual.abs() > maximum.0 {
                maximum = (residual.abs(), j, k);
            }
        }
    }
    let rms_residual = rms_norm / (positions.len() as f64).sqrt();
    Ok(LinebRateOrbit {
        positions,
        gradients,
        max_abs_residual: maximum.0,
        rms_residual,
        max_bead: maximum.1,
        max_coordinate: maximum.2,
    })
}

fn hqtst_rate_observed<S: PointSurface>(
    surface: &S,
    path: &LinebPath<impl Sized>,
    beads: usize,
    mut observe: impl FnMut(usize, ArrayView1<'_, f64>, ArrayView1<'_, f64>),
) -> Result<HqtstRate, SaddleError> {
    let calls = AtomicUsize::new(0);
    let counted = crate::lineb::Counted {
        surface,
        calls: &calls,
    };
    let replicas = path::replicas(path.weighted.view(), &path.time_table, beads)?;
    let kappa = spring_constant(replicas.period, path.hbar, beads)?;
    let step = difference_step(path)?;
    let dof = replicas.weighted.ncols();
    let distinct = beads / 2 + 1;
    let mut energies = vec![0.0; beads];
    let mut blocks = Vec::with_capacity(distinct);
    for bead in 0..distinct {
        let cartesian = path.masses.to_cartesian(replicas.weighted.row(bead))?;
        let (energy, gradient) = counted.eval(cartesian.view())?;
        if gradient.len() != dof {
            return Err(SaddleError::Shape(
                "replica gradient dimensions differ".into(),
            ));
        }
        if !energy.is_finite() || gradient.iter().any(|component| !component.is_finite()) {
            return Err(SaddleError::NonFinite("ring-polymer replica"));
        }
        observe(bead, cartesian.view(), gradient.view());
        energies[bead] = energy;
        energies[(beads - bead) % beads] = energy;
        blocks.push(weighted_block(
            &counted,
            &path.masses,
            cartesian.view(),
            gradient.view(),
            step,
        )?);
    }
    let gauges = gauge_modes(&replicas.weighted, &path.masses)?;
    let spectrum = deflated_log_abs_det(beads * dof, &gauges, |vector| {
        let mut out = Array1::zeros(vector.len());
        for bead in 0..beads {
            let previous = if bead == 0 { beads - 1 } else { bead - 1 };
            let next = if bead + 1 == beads { 0 } else { bead + 1 };
            let block = vector.slice(s![bead * dof..(bead + 1) * dof]);
            let mut row = blocks[bead.min(beads - bead)].dot(&block);
            row.scaled_add(2.0 * kappa, &block);
            row.scaled_add(
                -kappa,
                &vector.slice(s![previous * dof..(previous + 1) * dof]),
            );
            row.scaled_add(-kappa, &vector.slice(s![next * dof..(next + 1) * dof]));
            out.slice_mut(s![bead * dof..(bead + 1) * dof]).assign(&row);
        }
        out
    })?;
    finish(
        path,
        &replicas.weighted,
        &energies,
        replicas.period,
        spectrum,
        None,
        RateWork {
            curvature: calls.load(Ordering::Relaxed) - distinct,
            beads: distinct,
        },
    )
}

/// The paper's finite-difference Hessians at the LI-NEB images, cubically
/// interpolated onto the replicas, then the same harmonic rate.
#[cfg(any(test, feature = "lineb-interpolated-hessian"))]
pub fn hqtst_rate_interpolated_hessian<S: PointSurface>(
    surface: &S,
    path: &LinebPath<impl Sized>,
    beads: usize,
) -> Result<HqtstRate, SaddleError> {
    let step = coordinate_scale(path.weighted.iter().copied())?;
    hqtst_rate_interpolated_hessian_with_step(surface, path, beads, step)
}

/// Interpolate image Hessians obtained using a prescribed positive Cartesian
/// displacement. The two gradients for each column are separated by twice
/// this displacement, independent of the displaced atom's mass.
#[cfg(any(test, feature = "lineb-interpolated-hessian"))]
pub fn hqtst_rate_interpolated_hessian_with_step<S: PointSurface>(
    surface: &S,
    path: &LinebPath<impl Sized>,
    beads: usize,
    step: f64,
) -> Result<HqtstRate, SaddleError> {
    Ok(hqtst_rate_interpolated_hessian_observed(surface, path, beads, Some(step), false)?.0)
}

/// Evaluate the image-interpolated rate. The optional orbit retains the actual
/// gradients from the full ring energy batch, including the return leg.
/// A supplied positive step is a central-difference Cartesian displacement.
#[cfg(any(test, feature = "lineb-interpolated-hessian"))]
pub fn hqtst_rate_interpolated_hessian_observed<S: PointSurface>(
    surface: &S,
    path: &LinebPath<impl Sized>,
    beads: usize,
    step: Option<f64>,
    return_orbit: bool,
) -> Result<(HqtstRate, Option<LinebRateOrbit>), SaddleError> {
    let step = match step {
        Some(step) => step,
        None => coordinate_scale(path.weighted.iter().copied())?,
    };
    if !step.is_finite() || step <= 0.0 {
        return Err(SaddleError::Invalid(
            "the image Hessian displacement must be finite and positive".into(),
        ));
    }
    let calls = std::sync::atomic::AtomicUsize::new(0);
    let counted = crate::lineb::Counted {
        surface,
        calls: &calls,
    };
    let hessians = image_hessians(&counted, &path.masses, path.weighted.view(), step)?;
    interpolated_rate(
        surface,
        path,
        beads,
        &hessians,
        Some(step),
        calls.load(Ordering::Relaxed),
        return_orbit,
    )
}

/// Interpolate independently measured Cartesian energy Hessians at the ordered
/// LI-NEB images. Each block uses the atom order and Cartesian axes of its
/// corresponding row in `path.cartesian`, in energy per squared input length. The rate's
/// evaluation count includes replica energies and excludes creation of the
/// supplied Hessians.
#[cfg(any(test, feature = "lineb-interpolated-hessian"))]
pub fn hqtst_rate_from_image_hessians<S: PointSurface>(
    surface: &S,
    path: &LinebPath<impl Sized>,
    beads: usize,
    cartesian_hessians: &[Array2<f64>],
) -> Result<HqtstRate, SaddleError> {
    if cartesian_hessians.len() != path.weighted.nrows() {
        return Err(SaddleError::Shape(
            "one Cartesian Hessian per path image is required".into(),
        ));
    }
    let weighted = cartesian_hessians
        .iter()
        .map(|h| weight_cartesian_hessian(h, &path.masses))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(interpolated_rate(surface, path, beads, &weighted, None, 0, false)?.0)
}

#[cfg(any(test, feature = "lineb-interpolated-hessian"))]
fn interpolated_rate<S: PointSurface>(
    surface: &S,
    path: &LinebPath<impl Sized>,
    beads: usize,
    hessians: &[Array2<f64>],
    step: Option<f64>,
    setup_evaluations: usize,
    return_orbit: bool,
) -> Result<(HqtstRate, Option<LinebRateOrbit>), SaddleError> {
    let replicas = path::replicas(path.weighted.view(), &path.time_table, beads)?;
    let kappa = spring_constant(replicas.period, path.hbar, beads)?;
    let curve =
        Curve::with_end_tangents(path.weighted.view(), path.time_table.end_tangents.as_ref())?;
    let blocks = interpolated_blocks(
        hessians,
        curve.knots(),
        &replicas.parameter,
        replicas.weighted.ncols(),
        kappa,
    )?;
    let gauges = gauge_modes(&replicas.weighted, &path.masses)?;
    let spectrum = ring_spectrum(&blocks, kappa, gauges)?;
    let mut coordinates = Array2::zeros(replicas.weighted.dim());
    for (mut point, row) in coordinates.outer_iter_mut().zip(replicas.weighted.rows()) {
        point.assign(&path.masses.to_cartesian(row)?);
    }
    let observations = surface.eval_batch(coordinates.view())?;
    if observations.len() != beads {
        return Err(SaddleError::Shape(
            "ring surface returned a different number of replicas".into(),
        ));
    }
    let mut energies = Vec::with_capacity(beads);
    let mut gradients = return_orbit.then(|| Array2::zeros(coordinates.dim()));
    for (i, (energy, gradient)) in observations.into_iter().enumerate() {
        if gradient.len() != coordinates.ncols() {
            return Err(SaddleError::Shape(
                "ring surface gradient length differs from the geometry".into(),
            ));
        }
        if !energy.is_finite() || gradient.iter().any(|component| !component.is_finite()) {
            return Err(SaddleError::NonFinite("ring-polymer replica"));
        }
        if let Some(rows) = gradients.as_mut() {
            rows.row_mut(i).assign(&gradient);
        }
        energies.push(energy);
    }
    let rate = finish(
        path,
        &replicas.weighted,
        &energies,
        replicas.period,
        spectrum,
        step,
        RateWork {
            curvature: setup_evaluations,
            beads,
        },
    )?;
    let orbit = gradients
        .map(|rows| orbit_residual(path, rate.period, coordinates, rows))
        .transpose()?;
    Ok((rate, orbit))
}

struct RateWork {
    curvature: usize,
    beads: usize,
}

fn finish(
    path: &LinebPath<impl Sized>,
    beads: &Array2<f64>,
    energies: &[f64],
    period: f64,
    spectrum: FluctuationSpectrum,
    image_hessian_displacement: Option<f64>,
    work: RateWork,
) -> Result<HqtstRate, SaddleError> {
    let FluctuationSpectrum {
        log_abs_det,
        negative_modes,
    } = spectrum;
    let bead_action = bead_action(beads, energies, period, path.hbar)?;
    let line_action = (2.0 * path.action + path.energy * period) / path.hbar;
    let modes = beads.len() - gauge_modes(beads, &path.masses)?.len();
    let log_rate = log_rate(beads, period, path.hbar, log_abs_det, modes, bead_action)?;
    Ok(HqtstRate {
        temperature: path::temperature(0.5 * period, path.hbar, path.boltzmann)?,
        period,
        line_action,
        bead_action,
        log_abs_det,
        negative_modes,
        image_hessian_displacement,
        modes,
        log_rate_over_reactant: log_rate,
        rate_over_reactant: if log_rate > f64::MAX.ln() {
            f64::INFINITY
        } else {
            log_rate.exp()
        },
        evaluations: work.curvature + work.beads,
        bead_evaluations: work.beads,
        curvature_evaluations: work.curvature,
    })
}

fn spring_constant(period: f64, hbar: f64, beads: usize) -> Result<f64, SaddleError> {
    if !(period > 0.0 && hbar > 0.0) || beads < 2 {
        return Err(SaddleError::Invalid(
            "the ring polymer needs a positive period and hbar and at least two replicas".into(),
        ));
    }
    let beta_n = period / (hbar * beads as f64);
    let kappa = 1.0 / (beta_n * beta_n * hbar * hbar);
    if !kappa.is_finite() {
        return Err(SaddleError::NonFinite("ring-polymer spring"));
    }
    Ok(kappa)
}

fn bead_action(
    beads: &Array2<f64>,
    energies: &[f64],
    period: f64,
    hbar: f64,
) -> Result<f64, SaddleError> {
    let count = beads.nrows();
    let beta_n = period / (hbar * count as f64);
    let mut kinetic = 0.0;
    for bead in 0..count {
        let previous = if bead == 0 { count - 1 } else { bead - 1 };
        let mut length_sq = 0.0;
        for coordinate in 0..beads.ncols() {
            let delta = beads[(bead, coordinate)] - beads[(previous, coordinate)];
            length_sq += delta * delta;
        }
        kinetic += length_sq;
    }
    let potential: f64 = energies.iter().sum();
    let action = kinetic / (2.0 * beta_n * hbar * hbar) + beta_n * potential;
    if !action.is_finite() {
        return Err(SaddleError::NonFinite("ring-polymer action"));
    }
    Ok(action)
}

fn log_rate(
    beads: &Array2<f64>,
    period: f64,
    hbar: f64,
    log_abs_det: f64,
    modes: usize,
    bead_action: f64,
) -> Result<f64, SaddleError> {
    let count = beads.nrows();
    let beta_n = period / (hbar * count as f64);
    let mut length_sq = 0.0;
    for bead in 0..count {
        let previous = if bead == 0 { count - 1 } else { bead - 1 };
        for coordinate in 0..beads.ncols() {
            let delta = beads[(bead, coordinate)] - beads[(previous, coordinate)];
            length_sq += delta * delta;
        }
    }
    if length_sq <= 0.0 || length_sq.is_nan() {
        return Err(SaddleError::Invalid(
            "the replicas do not separate, so B_N vanishes".into(),
        ));
    }
    let beta_hbar = beta_n * hbar;
    let log_rate = -0.5 * (2.0 * std::f64::consts::PI * beta_n * hbar * hbar).ln() - beta_hbar.ln()
        + 0.5 * length_sq.ln()
        - 0.5 * log_abs_det
        - modes as f64 * beta_hbar.ln()
        - bead_action;
    if !log_rate.is_finite() {
        return Err(SaddleError::NonFinite("instanton rate"));
    }
    Ok(log_rate)
}

/// Mass-weighted Hessian block `M^{−½} H M^{−½}` at one replica from
/// `dof` Hessian-vector products, symmetrized.
fn weighted_block<S: PointSurface>(
    surface: &S,
    masses: &Masses,
    cartesian: ArrayView1<f64>,
    gradient: ArrayView1<f64>,
    step: f64,
) -> Result<Array2<f64>, SaddleError> {
    let dof = cartesian.len();
    let mut block = Array2::zeros((dof, dof));
    for column in 0..dof {
        let mut direction = Array1::zeros(dof);
        direction[column] = masses.inv_sqrt_at(column);
        let product = match surface.hessian_vector(cartesian, direction.view())? {
            Some(product) => product,
            None => numerical_hvp(surface, cartesian, gradient, direction.view(), step, true)?,
        };
        if product.len() != dof {
            return Err(SaddleError::Shape(
                "a Hessian-vector product must match the coordinate length".into(),
            ));
        }
        if product.iter().any(|component| !component.is_finite()) {
            return Err(SaddleError::NonFinite("Hessian-vector product"));
        }
        for row in 0..dof {
            block[(row, column)] = product[row] / masses.sqrt_at(row);
        }
    }
    for row in 0..dof {
        for column in row + 1..dof {
            let mean = 0.5 * (block[(row, column)] + block[(column, row)]);
            block[(row, column)] = mean;
            block[(column, row)] = mean;
        }
    }
    Ok(block)
}

/// `ε^{1/3}` times the Cartesian length of the path: the roundoff-optimal
/// central step for a gradient difference on a path of that length.
fn difference_step(path: &LinebPath<impl Sized>) -> Result<f64, SaddleError> {
    let mut length = 0.0;
    for j in 1..path.cartesian.nrows() {
        let chord = &path.cartesian.row(j) - &path.cartesian.row(j - 1);
        length += l2(chord.view());
    }
    if length <= 0.0 || !length.is_finite() {
        return Err(SaddleError::Invalid(
            "the tunneling path has no Cartesian length, so a Hessian difference has no step"
                .into(),
        ));
    }
    Ok(length * f64::EPSILON.cbrt())
}

/// `√ε` times the largest absolute coordinate, the finite-difference
/// Hessian step of the oracle.
#[cfg(any(test, feature = "lineb-interpolated-hessian"))]
fn coordinate_scale(values: impl IntoIterator<Item = f64>) -> Result<f64, SaddleError> {
    let mut scale = 0.0_f64;
    for value in values {
        scale = scale.max(value.abs());
    }
    if scale <= 0.0 || scale.is_nan() {
        return Err(SaddleError::Invalid(
            "the tunneling coordinates are all zero, so a Hessian difference has no length".into(),
        ));
    }
    Ok(scale * f64::EPSILON.sqrt())
}

fn gauge_modes(beads: &Array2<f64>, masses: &Masses) -> Result<Vec<Array1<f64>>, SaddleError> {
    let mut modes = Vec::new();
    modes.push(velocity_mode(beads)?);
    if let Some(atoms) = masses.atoms {
        // The whole ring moves rigidly: its replicas are one molecule of
        // `P · atoms` atoms.
        let ring = Array1::from_iter(beads.iter().copied());
        let sqrt_mass = masses.sqrt_vector();
        let repeated =
            Array1::from_iter((0..beads.nrows()).flat_map(|_| sqrt_mass.iter().copied()));
        let ring_metric = MassMetric::from_square_roots(repeated.view())?;
        modes.extend(rigid_directions(
            ring.view(),
            &ring_metric,
            beads.nrows() * atoms,
        ));
    }
    Ok(orthonormalize(modes))
}

fn velocity_mode(beads: &Array2<f64>) -> Result<Array1<f64>, SaddleError> {
    let count = beads.nrows();
    let dof = beads.ncols();
    let mut mode = Array1::zeros(count * dof);
    for bead in 0..count {
        let previous = if bead == 0 { count - 1 } else { bead - 1 };
        for coordinate in 0..dof {
            mode[bead * dof + coordinate] =
                beads[(bead, coordinate)] - beads[(previous, coordinate)];
        }
    }
    let norm = l2(mode.view());
    if norm <= 0.0 || norm.is_nan() {
        return Err(SaddleError::Invalid(
            "the replicas do not move, so the permutation mode is undefined".into(),
        ));
    }
    mode /= norm;
    Ok(mode)
}

fn orthonormalize(modes: Vec<Array1<f64>>) -> Vec<Array1<f64>> {
    let mut kept: Vec<Array1<f64>> = Vec::new();
    for mut mode in modes {
        for earlier in &kept {
            let coeff = earlier.dot(&mode);
            mode.scaled_add(-coeff, earlier);
        }
        let norm = l2(mode.view());
        if norm > 0.0 {
            mode /= norm;
            kept.push(mode);
        }
    }
    kept
}

fn project(vector: &mut Array1<f64>, directions: &[Array1<f64>]) {
    for direction in directions {
        let coeff = direction.dot(vector);
        vector.scaled_add(-coeff, direction);
    }
}

struct FluctuationSpectrum {
    log_abs_det: f64,
    negative_modes: usize,
}

fn summarize_spectrum<'a>(
    values: impl Iterator<Item = &'a f64>,
) -> Result<FluctuationSpectrum, SaddleError> {
    let mut log_abs_det = 0.0;
    let mut negative_modes = 0;
    for &value in values {
        if value == 0.0 || !value.is_finite() {
            return Err(SaddleError::Invalid(
                "a kept ring-polymer eigenvalue vanishes or is nonfinite".into(),
            ));
        }
        log_abs_det += value.abs().ln();
        negative_modes += usize::from(value < 0.);
    }
    if !log_abs_det.is_finite() {
        return Err(SaddleError::NonFinite("fluctuation determinant"));
    }
    Ok(FluctuationSpectrum {
        log_abs_det,
        negative_modes,
    })
}

/// `Σ ln|λ|` of `apply` on the complement of the orthonormal `gauges`.
///
/// The spectrum is that of `Π A Π + G Gᵀ`, `Π = I − G Gᵀ`: the operator
/// on the complement, plus a unit eigenvalue on each gauge mode, which
/// adds nothing to the sum.
fn deflated_log_abs_det(
    length: usize,
    gauges: &[Array1<f64>],
    apply: impl Fn(ArrayView1<f64>) -> Array1<f64>,
) -> Result<FluctuationSpectrum, SaddleError> {
    if gauges.len() >= length {
        return Err(SaddleError::Invalid(
            "the fluctuation space is empty".into(),
        ));
    }
    let values = lanczos_spectrum(
        |vector: ArrayView1<f64>| {
            let mut kept = vector.to_owned();
            project(&mut kept, gauges);
            let mut out = apply(kept.view());
            project(&mut out, gauges);
            for gauge in gauges {
                out.scaled_add(gauge.dot(&vector), gauge);
            }
            out
        },
        length,
    )?;
    summarize_spectrum(values.iter())
}

#[cfg(any(test, feature = "lineb-interpolated-hessian"))]
fn image_hessians<S: PointSurface>(
    surface: &S,
    masses: &Masses,
    images: ArrayView2<f64>,
    step: f64,
) -> Result<Vec<Array2<f64>>, SaddleError> {
    let mut hessians = Vec::with_capacity(images.nrows());
    for row in images.rows() {
        let cartesian = masses.to_cartesian(row)?;
        let block = cartesian_image_hessian(surface, cartesian.view(), step)?;
        hessians.push(weight_cartesian_hessian(&block, masses)?);
    }
    Ok(hessians)
}

/// Cartesian energy Hessian from central gradient differences at one image.
/// Each column uses two gradients in an ordered batch, separated by twice
/// `step` in the surface's length units. The returned matrix precedes mass
/// weighting and symmetrization.
#[cfg(any(test, feature = "lineb-interpolated-hessian"))]
pub fn cartesian_image_hessian<S: PointSurface>(
    surface: &S,
    cartesian: ArrayView1<f64>,
    step: f64,
) -> Result<Array2<f64>, SaddleError> {
    if !step.is_finite() || step <= 0.0 || cartesian.is_empty() {
        return Err(SaddleError::Invalid(
            "a nonempty image and positive finite displacement are required".into(),
        ));
    }
    let dof = cartesian.len();
    let mut displacements = Array2::from_shape_fn((2 * dof, dof), |(_, j)| cartesian[j]);
    for column in 0..dof {
        displacements[(2 * column, column)] += step;
        displacements[(2 * column + 1, column)] -= step;
    }
    let samples = surface.eval_batch(displacements.view())?;
    if samples.len() != 2 * dof || samples.iter().any(|(_, g)| g.len() != dof) {
        return Err(SaddleError::Shape(
            "image Hessian gradient batch dimensions differ from the displacements".into(),
        ));
    }
    let mut hessian = Array2::zeros((dof, dof));
    for (column, pair) in samples.as_chunks::<2>().0.iter().enumerate() {
        for row in 0..dof {
            hessian[(row, column)] = (pair[0].1[row] - pair[1].1[row]) / (2.0 * step);
        }
    }
    if hessian.iter().any(|v| !v.is_finite()) {
        return Err(SaddleError::NonFinite("Cartesian image Hessian"));
    }
    Ok(hessian)
}

#[cfg(any(test, feature = "lineb-interpolated-hessian"))]
fn weight_cartesian_hessian(
    hessian: &Array2<f64>,
    masses: &Masses,
) -> Result<Array2<f64>, SaddleError> {
    let dof = masses.dof();
    if hessian.dim() != (dof, dof) {
        return Err(SaddleError::Shape(
            "Cartesian Hessian dimensions must match the geometry".into(),
        ));
    }
    if hessian.iter().any(|v| !v.is_finite()) {
        return Err(SaddleError::NonFinite("Cartesian image Hessian"));
    }
    let mut weighted_hessian = Array2::zeros((dof, dof));
    for row in 0..dof {
        for column in 0..dof {
            let symmetric = 0.5 * (hessian[(row, column)] + hessian[(column, row)]);
            weighted_hessian[(row, column)] =
                symmetric / (masses.sqrt_at(row) * masses.sqrt_at(column));
        }
    }
    Ok(weighted_hessian)
}

#[cfg(any(test, feature = "lineb-interpolated-hessian"))]
fn interpolated_blocks(
    hessians: &[Array2<f64>],
    knots: &[f64],
    parameters: &[f64],
    dof: usize,
    kappa: f64,
) -> Result<Vec<Array2<f64>>, SaddleError> {
    let mut elements = Vec::with_capacity(dof * dof);
    for row in 0..dof {
        for column in 0..dof {
            let values: Vec<f64> = hessians
                .iter()
                .map(|hessian| hessian[(row, column)])
                .collect();
            elements.push(Spline::new(knots, &values));
        }
    }
    Ok(parameters
        .iter()
        .map(|&lambda| {
            let mut block =
                Array2::from_shape_fn((dof, dof), |(r, c)| elements[r * dof + c].value(lambda));
            block.diag_mut().mapv_inplace(|value| value + 2. * kappa);
            block
        })
        .collect())
}

#[cfg(any(test, feature = "lineb-interpolated-hessian"))]
fn dense_ring(blocks: &[Array2<f64>], kappa: f64) -> Array2<f64> {
    let count = blocks.len();
    let dof = blocks[0].nrows();
    let mut ring = Array2::zeros((count * dof, count * dof));
    for (i, block) in blocks.iter().enumerate() {
        let start = i * dof;
        ring.slice_mut(s![start..start + dof, start..start + dof])
            .assign(block);
        for neighbour in [(i + 1) % count, (i + count - 1) % count] {
            for r in 0..dof {
                ring[(start + r, neighbour * dof + r)] -= kappa;
            }
        }
    }
    ring
}

#[cfg(any(test, feature = "lineb-interpolated-hessian"))]
fn ring_spectrum(
    blocks: &[Array2<f64>],
    kappa: f64,
    gauges: Vec<Array1<f64>>,
) -> Result<FluctuationSpectrum, SaddleError> {
    if let Some(spectrum) = bordered_ring(blocks, kappa, &gauges)? {
        return Ok(spectrum);
    }
    projected_log_abs_det(dense_ring(blocks, kappa), gauges)
}

/// For orthonormal gauge columns G, the bordered Hessian [A G; Gᵀ 0]
/// has the projected log absolute determinant and one extra positive and
/// negative eigenvalue per gauge. Eliminate each bead except the last one;
/// keeping that bead and the gauges in the border closes the periodic chain.
/// A poorly conditioned block requests the full symmetric projection.
#[cfg(any(test, feature = "lineb-interpolated-hessian"))]
fn bordered_ring(
    blocks: &[Array2<f64>],
    kappa: f64,
    gauges: &[Array1<f64>],
) -> Result<Option<FluctuationSpectrum>, SaddleError> {
    let last = blocks.len() - 1;
    let dof = blocks[0].nrows();
    let width = dof + gauges.len();
    let mut border = Array2::zeros((width, width));
    border.slice_mut(s![..dof, ..dof]).assign(&blocks[last]);
    for (g, gauge) in gauges.iter().enumerate() {
        for i in 0..dof {
            let value = gauge[last * dof + i];
            border[(i, dof + g)] = value;
            border[(dof + g, i)] = value;
        }
    }
    let mut update = Array2::<f64>::zeros((dof, dof));
    let mut coupling = Array2::<f64>::zeros((dof, width));
    let mut result = FluctuationSpectrum {
        log_abs_det: 0.,
        negative_modes: 0,
    };
    for (i, block) in blocks[..last].iter().enumerate() {
        let pivot = block - &update;
        for (g, gauge) in gauges.iter().enumerate() {
            for r in 0..dof {
                coupling[(r, dof + g)] += gauge[i * dof + r];
            }
        }
        // The two spring links both reach the anchor when there are two beads.
        for linked in [i == 0, i + 1 == last] {
            if linked {
                for r in 0..dof {
                    coupling[(r, r)] -= kappa;
                }
            }
        }
        let (values, vectors) = crate::eigensolve::exact_eigh(pivot.view())?;
        let scale = values
            .iter()
            .chain(coupling.iter())
            .fold(kappa, |a, &b| a.max(b.abs()));
        if values
            .iter()
            .any(|&value| value.abs() <= f64::EPSILON.sqrt() * scale)
        {
            return Ok(None);
        }
        let contribution = summarize_spectrum(values.iter())?;
        result.log_abs_det += contribution.log_abs_det;
        result.negative_modes += contribution.negative_modes;
        let mut scaled = vectors.clone();
        for (mut column, &value) in scaled.columns_mut().into_iter().zip(&values) {
            column /= value;
        }
        let inverse = scaled.dot(&vectors.t());
        let solved = inverse.dot(&coupling);
        border -= &coupling.t().dot(&solved);
        coupling = solved * kappa;
        update = inverse * (kappa * kappa);
    }
    let (values, _) = crate::eigensolve::exact_eigh(border.view())?;
    let largest = values.iter().fold(0.0_f64, |scale, &x| scale.max(x.abs()));
    if values
        .iter()
        .any(|&x| x.abs() <= f64::EPSILON * (width as f64) * largest)
    {
        return Ok(None);
    }
    let tail = summarize_spectrum(values.iter())?;
    result.log_abs_det += tail.log_abs_det;
    let Some(negative) = (result.negative_modes + tail.negative_modes).checked_sub(gauges.len())
    else {
        return Ok(None);
    };
    result.negative_modes = negative;
    // The bordered determinant also contains det(GᵀG). Its correction makes
    // the result invariant to the scale and mutual overlap of the gauge columns.
    if !gauges.is_empty() {
        let gram = Array2::from_shape_fn((gauges.len(), gauges.len()), |(i, j)| {
            gauges[i].dot(&gauges[j])
        });
        let (metric, _) = crate::eigensolve::exact_eigh(gram.view())?;
        let largest = metric.iter().copied().fold(0.0_f64, f64::max);
        if metric
            .iter()
            .any(|&value| value <= f64::EPSILON.sqrt() * largest)
        {
            return Ok(None);
        }
        result.log_abs_det -= metric.iter().map(|v| v.ln()).sum::<f64>();
    }
    Ok(Some(result))
}

/// Log absolute determinant of the symmetric ring Hessian restricted to
/// the orthogonal complement of the gauge modes. Each Householder similarity
/// moves one gauge onto a coordinate axis. The trailing principal block is
/// the Hessian on the retained space, including any unstable directions.
#[cfg(any(test, feature = "lineb-interpolated-hessian"))]
fn projected_log_abs_det(
    mut ring: Array2<f64>,
    mut gauges: Vec<Array1<f64>>,
) -> Result<FluctuationSpectrum, SaddleError> {
    let removed = gauges.len();
    if removed >= ring.nrows() {
        return Err(SaddleError::Invalid(
            "the fluctuation space is empty".into(),
        ));
    }
    for axis in 0..removed {
        let mut reflector = gauges[axis].slice(s![axis..]).to_owned();
        let norm = l2(reflector.view());
        if !(norm.is_finite() && norm > 0.) {
            return Err(SaddleError::Invalid(
                "dependent fluctuation gauge modes".into(),
            ));
        }
        reflector[0] += norm.copysign(reflector[0]);
        reflector /= l2(reflector.view());

        for gauge in &mut gauges[axis + 1..] {
            let mut tail = gauge.slice_mut(s![axis..]);
            let projection = reflector.dot(&tail);
            tail.scaled_add(-2. * projection, &reflector);
        }
        // For H = I - 2vvᵀ, HAH = A - vwᵀ - wvᵀ,
        // where w = 2(Av - (vᵀAv)v).
        let mut block = ring.slice_mut(s![axis.., axis..]);
        let mut w = block.dot(&reflector);
        let along = reflector.dot(&w);
        w.scaled_add(-along, &reflector);
        w *= 2.;
        for (i, mut row) in block.outer_iter_mut().enumerate() {
            row.scaled_add(-reflector[i], &w);
            row.scaled_add(-w[i], &reflector);
        }
    }
    let kept = ring.slice(s![removed.., removed..]);
    #[cfg(feature = "faer")]
    let values = {
        let n = kept.nrows();
        let matrix = faer::Mat::from_fn(n, n, |i, j| 0.5 * (kept[(i, j)] + kept[(j, i)]));
        matrix
            .self_adjoint_eigenvalues(faer::Side::Lower)
            .map_err(|_| SaddleError::Solver("ring Hessian eigensolver did not converge".into()))?
    };
    #[cfg(not(feature = "faer"))]
    let values = exact_eigh(kept)?.0;
    summarize_spectrum(values.iter())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cyclic_schur_complements_match_the_signed_dense_projection() {
        let blocks = vec![
            ndarray::array![[0., 1.], [1., 0.]],
            ndarray::array![[4., -0.3], [-0.3, 2.]],
            ndarray::array![[-2., 0.4], [0.4, 3.]],
            ndarray::array![[2., 0.7], [0.7, 1.]],
        ];
        let gauges = orthonormalize(vec![
            ndarray::array![2., 0., 1., 3., -1., 1., 0., 2.],
            ndarray::array![1., 2., -2., 0., 1., 1., 2., 0.],
        ]);
        let dense = projected_log_abs_det(dense_ring(&blocks, 0.5), gauges.clone()).unwrap();
        let schur = bordered_ring(&blocks, 0.5, &gauges).unwrap().unwrap();
        assert_eq!(schur.negative_modes, dense.negative_modes);
        assert!((schur.log_abs_det - dense.log_abs_det).abs() < 1e-11);
    }

    #[test]
    fn rescaling_and_mixing_gauges_preserves_the_border_determinant() {
        let blocks = vec![
            ndarray::array![[2.]],
            ndarray::array![[4.]],
            ndarray::array![[1.]],
            ndarray::array![[-2.]],
        ];
        let gauges = vec![
            ndarray::array![1., 2., 0., 0.],
            ndarray::array![2., 1., 3., 1.],
        ];
        let dense = projected_log_abs_det(dense_ring(&blocks, 0.4), gauges.clone()).unwrap();
        let schur = bordered_ring(&blocks, 0.4, &gauges).unwrap().unwrap();
        assert_eq!(schur.negative_modes, dense.negative_modes);
        assert!((schur.log_abs_det - dense.log_abs_det).abs() < 1e-11);
    }

    #[test]
    fn singular_bead_pivot_uses_the_full_projected_operator() {
        let blocks = vec![
            ndarray::array![[0.]],
            ndarray::array![[2.]],
            ndarray::array![[3.]],
        ];
        let gauges = orthonormalize(vec![ndarray::array![1., 2., 2.]]);
        assert!(bordered_ring(&blocks, 1., &gauges).unwrap().is_none());
        let dense = projected_log_abs_det(dense_ring(&blocks, 1.), gauges.clone()).unwrap();
        let measured = ring_spectrum(&blocks, 1., gauges).unwrap();
        assert_eq!(measured.negative_modes, dense.negative_modes);
        assert_eq!(measured.log_abs_det, dense.log_abs_det);
    }

    #[test]
    fn dense_projection_removes_gauges_coupled_to_an_indefinite_block() {
        let basis = ndarray::array![
            [1., 1., 1., 1.],
            [1., -1., 1., -1.],
            [1., 1., -1., -1.],
            [1., -1., -1., 1.]
        ] * 0.5;
        let blocks = ndarray::array![
            [2., 3., 0.6, 0.7],
            [3., 4., -1., 0.],
            [0.6, -1., -3., 0.5],
            [0.7, 0., 0.5, 2.]
        ];
        let matrix = basis.dot(&blocks).dot(&basis.t());
        let gauges = vec![basis.column(0).to_owned(), basis.column(1).to_owned()];
        let value = projected_log_abs_det(matrix, gauges).unwrap();
        // The retained 2x2 block has determinant -6.25.
        assert_eq!(value.negative_modes, 1);
        assert!((value.log_abs_det - 6.25_f64.ln()).abs() < 64. * f64::EPSILON);
    }

    /// A free ring's spring Hessian has eigenvalues
    /// `4 κ sin²(π k / P)` on each coordinate. Removing the constant
    /// mode leaves `Π = κ^{P−1} P²`.
    #[test]
    fn spring_log_det_matches_the_sine_product() {
        let beads = 8;
        let dof = 1;
        let kappa = 2.5;
        let apply = |vector: ArrayView1<f64>| {
            let mut out = Array1::zeros(vector.len());
            for bead in 0..beads {
                let previous = if bead == 0 { beads - 1 } else { bead - 1 };
                let next = if bead + 1 == beads { 0 } else { bead + 1 };
                out[bead] = kappa * (2.0 * vector[bead] - vector[previous] - vector[next]);
            }
            out
        };
        let mut constant = Array1::zeros(beads * dof);
        constant.fill(1.0);
        constant /= l2(constant.view());
        let log_det = deflated_log_abs_det(beads, &[constant], apply)
            .unwrap()
            .log_abs_det;
        let expected = (beads - 1) as f64 * kappa.ln() + 2.0 * (beads as f64).ln();
        // Relative to the expected log-determinant.
        let scale = expected.abs();
        assert!(
            (log_det - expected).abs() <= scale * f64::EPSILON * (beads as f64).powi(2),
            "{log_det} vs {expected}"
        );
    }

    /// A diagonal operator with distinct entries, larger than the dense
    /// Krylov cutoff of the shared Lanczos. The log-determinant is the
    /// sum of the logs of the entries.
    #[test]
    fn deflated_lanczos_multiplies_distinct_eigenvalues() {
        let n = 128;
        let apply = |vector: ArrayView1<f64>| {
            Array1::from_iter((0..n).map(|index| (index + 1) as f64 * vector[index]))
        };
        let log_det = deflated_log_abs_det(n, &[], apply).unwrap().log_abs_det;
        let expected: f64 = (1..=n).map(|k| (k as f64).ln()).sum();
        // Relative to the expected log-determinant.
        let scale = expected.abs();
        assert!(
            (log_det - expected).abs() <= scale * f64::EPSILON * (n as f64).powi(2),
            "{log_det} vs {expected}"
        );
    }
}
