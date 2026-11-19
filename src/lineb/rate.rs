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
//! The paper's finite-difference Hessians at the LI-NEB images, cubic
//! interpolation, and dense spectrum are compiled with the
//! `lineb-oracle` feature for tests and benches. They are not part of
//! the default build.
//!
//! The rate times the reactant partition function is Richardson's
//! ring-polymer instanton expression (Int. Rev. Phys. Chem. 37, 171
//! (2018), eq. 52), which is the harmonic quantum TST prefactor:
//!
//! `k Z_r = Λ_N^{−1} (β_N ℏ)^{−1} √B_N  Π' (β_N ℏ η_k)^{−1}  exp(−β_N U)`.

use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(any(test, feature = "lineb-oracle"))]
use ndarray::ArrayView2;
use ndarray::{Array1, Array2, ArrayView1, s};

use super::action::l2;
use super::align::rigid_directions;
use super::mass::Masses;
use super::optimize::LinebPath;
use super::path;
#[cfg(any(test, feature = "lineb-oracle"))]
use super::path::Curve;
#[cfg(any(test, feature = "lineb-oracle"))]
use crate::eigensolve::exact_eigh;
use crate::error::SaddleError;
use crate::linalg::numerical_hvp;
use crate::minmode::PointSurface;
use crate::sparse_hessian::lanczos_spectrum;
#[cfg(any(test, feature = "lineb-oracle"))]
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
    /// How many eigenvalues that sum keeps.
    pub modes: usize,
    /// `ln(k Z_r)`.
    pub log_rate_over_reactant: f64,
    /// `k Z_r`, the exponential of the log. It carries `exp(−β_N U)` on
    /// the surface's own energy zero, so on a total energy far from zero
    /// it overflows to infinity or underflows to zero; the log is finite
    /// whenever this is returned.
    pub rate_over_reactant: f64,
    /// Surface evaluations used to build the factor.
    pub evaluations: usize,
}

/// Harmonic quantum TST rate from Hessian-vector products at the
/// replicas. `beads` is `P` in `t_j = j τ / P`.
pub fn hqtst_rate<S: PointSurface>(
    surface: &S,
    path: &LinebPath,
    beads: usize,
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
        if !energy.is_finite() || gradient.iter().any(|component| !component.is_finite()) {
            return Err(SaddleError::NonFinite("ring-polymer replica"));
        }
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
    let log_abs_det = deflated_log_abs_det(beads * dof, &gauges, |vector| {
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
        log_abs_det,
        calls.load(Ordering::Relaxed),
    )
}

/// The paper's finite-difference Hessians at the LI-NEB images, cubically
/// interpolated onto the replicas, then the same rate. Oracle for
/// [`hqtst_rate`]. Not built unless the `lineb-oracle` feature is on
/// (tests and benches request it).
#[cfg(any(test, feature = "lineb-oracle"))]
pub fn hqtst_rate_interpolated_hessian<S: PointSurface>(
    surface: &S,
    path: &LinebPath,
    beads: usize,
) -> Result<HqtstRate, SaddleError> {
    let calls = std::sync::atomic::AtomicUsize::new(0);
    let counted = crate::lineb::Counted {
        surface,
        calls: &calls,
    };
    let replicas = path::replicas(path.weighted.view(), &path.time_table, beads)?;
    let kappa = spring_constant(replicas.period, path.hbar, beads)?;
    let step = coordinate_scale(path.weighted.iter().copied())?;
    let hessians = image_hessians(&counted, &path.masses, path.weighted.view(), step)?;
    let curve = Curve::new(path.weighted.view())?;
    let dense = interpolated_ring(
        &hessians,
        curve.knots(),
        &replicas.parameter,
        replicas.weighted.ncols(),
        kappa,
    )?;
    let log_abs_det = dense_kept_log_abs_det(dense.view(), &replicas.weighted, &path.masses)?;
    let mut energies = Vec::with_capacity(beads);
    for row in replicas.weighted.rows() {
        let cartesian = path.masses.to_cartesian(row)?;
        let (energy, gradient) = counted.eval(cartesian.view())?;
        if !energy.is_finite() || gradient.iter().any(|component| !component.is_finite()) {
            return Err(SaddleError::NonFinite("ring-polymer replica"));
        }
        energies.push(energy);
    }
    finish(
        path,
        &replicas.weighted,
        &energies,
        replicas.period,
        log_abs_det,
        calls.load(Ordering::Relaxed),
    )
}

fn finish(
    path: &LinebPath,
    beads: &Array2<f64>,
    energies: &[f64],
    period: f64,
    log_abs_det: f64,
    evaluations: usize,
) -> Result<HqtstRate, SaddleError> {
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
        modes,
        log_rate_over_reactant: log_rate,
        rate_over_reactant: log_rate.exp(),
        evaluations,
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
        direction[column] = 1.0 / masses.sqrt_mass[column];
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
            block[(row, column)] = product[row] / masses.sqrt_mass[row];
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
fn difference_step(path: &LinebPath) -> Result<f64, SaddleError> {
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
#[cfg(any(test, feature = "lineb-oracle"))]
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
        let sqrt_mass =
            Array1::from_iter((0..beads.nrows()).flat_map(|_| masses.sqrt_mass.iter().copied()));
        modes.extend(rigid_directions(
            ring.view(),
            sqrt_mass.view(),
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

/// `Σ ln|λ|` of `apply` on the complement of the orthonormal `gauges`.
///
/// The spectrum is that of `Π A Π + G Gᵀ`, `Π = I − G Gᵀ`: the operator
/// on the complement, plus a unit eigenvalue on each gauge mode, which
/// adds nothing to the sum.
fn deflated_log_abs_det(
    length: usize,
    gauges: &[Array1<f64>],
    apply: impl Fn(ArrayView1<f64>) -> Array1<f64>,
) -> Result<f64, SaddleError> {
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
    let mut sum = 0.0;
    for value in values.iter() {
        if *value == 0.0 || !value.is_finite() {
            return Err(SaddleError::Invalid(
                "the fluctuation operator has a vanishing eigenvalue".into(),
            ));
        }
        sum += value.abs().ln();
    }
    if !sum.is_finite() {
        return Err(SaddleError::NonFinite("fluctuation determinant"));
    }
    Ok(sum)
}

#[cfg(any(test, feature = "lineb-oracle"))]
fn image_hessians<S: PointSurface>(
    surface: &S,
    masses: &Masses,
    images: ArrayView2<f64>,
    step: f64,
) -> Result<Vec<Array2<f64>>, SaddleError> {
    let mut hessians = Vec::with_capacity(images.nrows());
    for row in images.rows() {
        hessians.push(finite_difference_hessian(surface, masses, row, step)?);
    }
    Ok(hessians)
}

#[cfg(any(test, feature = "lineb-oracle"))]
fn finite_difference_hessian<S: PointSurface>(
    surface: &S,
    masses: &Masses,
    weighted: ArrayView1<f64>,
    step: f64,
) -> Result<Array2<f64>, SaddleError> {
    let cartesian = masses.to_cartesian(weighted)?;
    let dof = cartesian.len();
    let mut hessian = Array2::zeros((dof, dof));
    for column in 0..dof {
        let mut plus = cartesian.clone();
        let mut minus = cartesian.clone();
        plus[column] += step;
        minus[column] -= step;
        let (_, gradient_plus) = surface.eval(plus.view())?;
        let (_, gradient_minus) = surface.eval(minus.view())?;
        for row in 0..dof {
            hessian[(row, column)] = (gradient_plus[row] - gradient_minus[row]) / (2.0 * step);
        }
    }
    let mut weighted_hessian = Array2::zeros((dof, dof));
    for row in 0..dof {
        for column in 0..dof {
            let symmetric = 0.5 * (hessian[(row, column)] + hessian[(column, row)]);
            weighted_hessian[(row, column)] =
                symmetric / (masses.sqrt_mass[row] * masses.sqrt_mass[column]);
        }
    }
    Ok(weighted_hessian)
}

#[cfg(any(test, feature = "lineb-oracle"))]
fn interpolated_ring(
    hessians: &[Array2<f64>],
    knots: &[f64],
    parameters: &[f64],
    dof: usize,
    kappa: f64,
) -> Result<Array2<f64>, SaddleError> {
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
    let count = parameters.len();
    let mut ring = Array2::zeros((count * dof, count * dof));
    for bead in 0..count {
        let lambda = parameters[bead];
        let previous = if bead == 0 { count - 1 } else { bead - 1 };
        let next = if bead + 1 == count { 0 } else { bead + 1 };
        for row in 0..dof {
            for column in 0..dof {
                let curvature = elements[row * dof + column].value(lambda);
                ring[(bead * dof + row, bead * dof + column)] = curvature;
            }
            ring[(bead * dof + row, bead * dof + row)] += 2.0 * kappa;
            ring[(bead * dof + row, previous * dof + row)] -= kappa;
            ring[(bead * dof + row, next * dof + row)] -= kappa;
        }
    }
    Ok(ring)
}

#[cfg(any(test, feature = "lineb-oracle"))]
fn dense_kept_log_abs_det(
    ring: ArrayView2<f64>,
    beads: &Array2<f64>,
    masses: &Masses,
) -> Result<f64, SaddleError> {
    let gauges = gauge_modes(beads, masses)?;
    let basis = fluctuation_basis(ring.nrows(), &gauges)?;
    let dim = basis.ncols();
    let mut reduced = Array2::zeros((dim, dim));
    for column in 0..dim {
        let image = ring.dot(&basis.column(column));
        for row in 0..dim {
            reduced[(row, column)] = basis.column(row).dot(&image);
        }
    }
    for row in 0..dim {
        for column in row + 1..dim {
            let mean = 0.5 * (reduced[(row, column)] + reduced[(column, row)]);
            reduced[(row, column)] = mean;
            reduced[(column, row)] = mean;
        }
    }
    let (values, _) = exact_eigh(reduced.view())?;
    let mut sum = 0.0;
    for value in values.iter() {
        if *value == 0.0 || !value.is_finite() {
            return Err(SaddleError::Invalid(
                "a kept ring-polymer eigenvalue vanishes".into(),
            ));
        }
        sum += value.abs().ln();
    }
    if !sum.is_finite() {
        return Err(SaddleError::NonFinite("fluctuation determinant"));
    }
    Ok(sum)
}

/// Orthonormal basis of the complement of `gauges`, one column each.
#[cfg(any(test, feature = "lineb-oracle"))]
fn fluctuation_basis(length: usize, gauges: &[Array1<f64>]) -> Result<Array2<f64>, SaddleError> {
    let dim = length.checked_sub(gauges.len()).ok_or_else(|| {
        SaddleError::Invalid("more gauge modes than ring-polymer coordinates".into())
    })?;
    if dim < 1 {
        return Err(SaddleError::Invalid(
            "the fluctuation space is empty".into(),
        ));
    }
    let resolved = f64::EPSILON * ((gauges.len() + dim) as f64).powi(2);
    let mut basis = Array2::zeros((length, dim));
    let mut filled = 0;
    for axis in 0..length {
        let mut probe = Array1::zeros(length);
        probe[axis] = 1.0;
        project(&mut probe, gauges);
        for column in 0..filled {
            let coeff = basis.column(column).dot(&probe);
            probe.scaled_add(-coeff, &basis.column(column));
        }
        project(&mut probe, gauges);
        let norm = l2(probe.view());
        if norm > resolved {
            probe /= norm;
            basis.column_mut(filled).assign(&probe);
            filled += 1;
            if filled == dim {
                return Ok(basis);
            }
        }
    }
    Err(SaddleError::Invalid(
        "the fluctuation space closed before every mode was found".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let log_det = deflated_log_abs_det(beads, &[constant], apply).unwrap();
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
        let log_det = deflated_log_abs_det(n, &[], apply).unwrap();
        let expected: f64 = (1..=n).map(|k| (k as f64).ln()).sum();
        // Relative to the expected log-determinant.
        let scale = expected.abs();
        assert!(
            (log_det - expected).abs() <= scale * f64::EPSILON * (n as f64).powi(2),
            "{log_det} vs {expected}"
        );
    }
}
