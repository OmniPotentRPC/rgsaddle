//! Ring-polymer surface for an unchanged PAFI walk.
//!
//! A bead count `P`, an inverse temperature `β` and a reduced Planck
//! constant `ℏ` turn any [`PointSurface`] into one ring-polymer
//! [`PointSurface`]. [`crate::pafi_profile`] then samples that surface
//! with the same hyperplane walk. The energy is the primitive
//! approximation of Feynman and Hibbs, written so its Boltzmann factor
//! at the physical temperature is the Euclidean weight `exp(-S/ℏ)`:
//!
//! `Δτ = β ℏ / P`,
//!
//! `U = Σ_i Σ_a m_a (q_i,a − q_{i+1,a})² / (2 ℏ Δτ β) + (1/P) Σ_i V(q_i)`.
//!
//! The first term is the spring. It uses the real mass of each degree of
//! freedom, once, on every link of the ring, including the link that
//! closes the last bead onto the first. `q_{P} = q_0`. The second term
//! is the bead average of the caller's potential, so `P = 1` is that
//! potential: the single bead is its own neighbour and the spring
//! vanishes. `S/ℏ = β U`.
//!
//! A collapsed path from [`centroid_path`] repeats the same tangent on
//! every bead, so [`crate::pafi_profile`] constrains only the centroid and
//! integrates its potential of mean force. [`centroid_hyperplane`] builds
//! that constraint as a [`Hyperplane`]. That is the static centroid quantum
//! transition state theory barrier, not a ring-polymer molecular dynamics
//! rate.
//! Parrinello and Rahman, J. Chem. Phys. 80, 860 (1984),
//! doi:10.1063/1.446740, sample this potential. Richardson and Althorpe,
//! J. Chem. Phys. 131, 214106 (2009), doi:10.1063/1.3267318, use the
//! same spring for the ring-polymer instanton. A general multi-bead path,
//! such as an orbit from `ring_path_from_con` when the `readcon` feature
//! is enabled, instead constrains the full `P d`-dimensional ring tangent.
//! Its hyperplanes mix centroid and internal modes and sample the free
//! energy along that collective ring-polymer path, not the centroid PMF.
//! The bead count is an argument of [`RingPolymer::new`] and of
//! [`bead_study`]. The caller supplies the overdamped step to
//! [`bead_study`], which rejects it when it exceeds the image-local
//! stiffness bound for any count. Neither function selects a bead count.
//!
//! Coordinates are bead-major. Bead `i` occupies
//! `[i · d, (i + 1) · d)`, where `d` is the length of one bead, the
//! length of `masses`. A centroid image repeats one geometry on every
//! bead ([`collapsed`]). An instanton orbit is those bead geometries
//! read from a `.con` by readcon-core, one frame per bead.
//!
//! # Examples
//!
//! ```
//! use ndarray::{Array1, ArrayView1, array};
//! use rgsaddle::{PointSurface, RingPolymer, RingSettings, SaddleError};
//!
//! struct Well;
//!
//! impl PointSurface for Well {
//!     fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
//!         let coordinate = x[0];
//!         let energy = (coordinate * coordinate - 1.0).powi(2);
//!         Ok((energy, array![4.0 * coordinate * (coordinate * coordinate - 1.0)]))
//!     }
//! }
//!
//! let well = Well;
//! let ring = RingPolymer::new(
//!     &well,
//!     1,
//!     RingSettings::default()
//!         .with_beta(2.0)
//!         .with_hbar(1.0)
//!         .with_masses(array![1.0]),
//! )?;
//! let (energy, gradient) = ring.eval(array![1.0].view())?;
//! assert!(energy.abs() <= f64::EPSILON);
//! assert!(gradient[0].abs() <= f64::EPSILON);
//! # Ok::<(), SaddleError>(())
//! ```

use std::f64::consts::PI;
use std::num::NonZeroUsize;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};

use crate::Hyperplane;
use crate::eigensolve::exact_eigh;
use crate::error::SaddleError;
use crate::linalg::modified_gram_schmidt;
use crate::minmode::{CartesianLayout, PointSurface};
use crate::pafi::{ArcPath, FreeEnergyProfile, PafiConfig, PafiIntegrator, pafi_profile};

/// Inverse temperature, `ℏ`, and the real masses of one bead.
///
/// Non-exhaustive: start from [`RingSettings::default`] and chain the
/// `with_*` setters. `beta` is `1/kT` in the energy unit of the bead
/// surface. `hbar` is `ℏ` in compatible energy-mass-length units:
/// `[ℏ] = sqrt([energy] [mass]) [length]`. An atomic-unit surface uses
/// `1`; with eV, amu, and Å use approximately
/// `0.064654148 sqrt(eV amu) Å`, not the eV·fs representation. `masses`
/// has one positive finite entry per degree of freedom, in the order the
/// bead surface evaluates.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct RingSettings {
    /// `β = 1/kT`.
    pub beta: f64,
    /// `ℏ`.
    pub hbar: f64,
    /// Mass of each degree of freedom of one bead.
    pub masses: Array1<f64>,
}

with_setters!(RingSettings {
    with_beta => beta: val f64,
    with_hbar => hbar: val f64,
    with_masses => masses: val Array1<f64>,
});

impl Default for RingSettings {
    /// One atomic-unit coordinate: `β = 1`, `ℏ = 1`, mass `1`.
    fn default() -> Self {
        Self {
            beta: 1.0,
            hbar: 1.0,
            masses: Array1::ones(1),
        }
    }
}

/// `P` beads of `surface` as one potential.
///
/// The potential is the `U` in the module documentation. [`crate::PafiConfig::thermal`]
/// is the physical `kT = 1/β` when `β` is [`RingSettings::beta`].
/// Rigid rotations are reported as excluded modes only when the bead
/// surface is Cartesian and each atom's three masses are bitwise equal;
/// anisotropic or non-Cartesian masses keep those directions active.
#[derive(Clone, Debug)]
pub struct RingPolymer<'a, S> {
    surface: &'a S,
    beads: NonZeroUsize,
    beta: f64,
    hbar: f64,
    masses: Array1<f64>,
}

impl<'a, S> RingPolymer<'a, S> {
    /// Ring of `beads` copies of `surface`.
    pub fn new(surface: &'a S, beads: usize, settings: RingSettings) -> Result<Self, SaddleError> {
        let beads = NonZeroUsize::new(beads)
            .ok_or_else(|| SaddleError::Invalid("a ring polymer needs at least one bead".into()))?;
        if !(settings.beta.is_finite() && settings.beta > 0.0) {
            return Err(SaddleError::Invalid(
                "ring-polymer beta must be finite and positive".into(),
            ));
        }
        if !(settings.hbar.is_finite() && settings.hbar > 0.0) {
            return Err(SaddleError::Invalid(
                "ring-polymer hbar must be finite and positive".into(),
            ));
        }
        if settings.masses.is_empty()
            || !settings
                .masses
                .iter()
                .all(|mass| mass.is_finite() && *mass > 0.0)
        {
            return Err(SaddleError::Invalid(
                "ring-polymer masses must be finite and positive, one per degree of freedom".into(),
            ));
        }
        let ring = Self {
            surface,
            beads,
            beta: settings.beta,
            hbar: settings.hbar,
            masses: settings.masses,
        };
        if ring.beads.get() > 1 {
            let thermal_action = ring.beta * ring.hbar;
            let thermal_action_squared = thermal_action * thermal_action;
            if !(thermal_action_squared.is_finite() && thermal_action_squared > 0.0) {
                return Err(SaddleError::NonFinite("ring-polymer spring thermal scale"));
            }
            for axis in 0..ring.dof() {
                if !ring.link(axis).is_finite() {
                    return Err(SaddleError::NonFinite("ring-polymer spring stiffness"));
                }
                ring.axis_spring(axis)?;
            }
        }
        Ok(ring)
    }

    /// Bead count `P`.
    pub fn beads(&self) -> usize {
        self.beads.get()
    }

    /// Degrees of freedom of one bead.
    pub fn dof(&self) -> usize {
        self.masses.len()
    }

    /// Inverse temperature `β` in the energy unit of the bead surface.
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// `ℏ` in the units of the bead surface.
    pub fn hbar(&self) -> f64 {
        self.hbar
    }

    /// Mass of each degree of freedom of one bead.
    pub fn masses(&self) -> ArrayView1<'_, f64> {
        self.masses.view()
    }

    /// Euclidean action over `ℏ`, `S/ℏ = β U`.
    pub fn action(&self, x: ArrayView1<f64>) -> Result<f64, SaddleError>
    where
        S: PointSurface,
    {
        let (energy, _) = self.eval(x)?;
        let action = self.beta * energy;
        if !action.is_finite() {
            return Err(SaddleError::NonFinite("ring-polymer action"));
        }
        Ok(action)
    }

    fn link(&self, dof: usize) -> f64 {
        let thermal_action = self.beta * self.hbar;
        self.masses[dof] * self.beads.get() as f64 / (2.0 * thermal_action * thermal_action)
    }

    /// Stiffest spring eigenvalue on one axis, `4 m P (β ℏ)^{-2} sin²(π k / P)` at `k = ⌊P/2⌋`.
    fn axis_spring(&self, axis: usize) -> Result<f64, SaddleError> {
        if self.beads.get() == 1 {
            return Ok(0.0);
        }
        let count = self.beads.get() as f64;
        let sine = (PI * (self.beads.get() / 2) as f64 / count).sin();
        let spring =
            4.0 * self.masses[axis] * count / (self.beta * self.hbar).powi(2) * sine * sine;
        if !spring.is_finite() {
            return Err(SaddleError::NonFinite("ring-polymer spring stiffness"));
        }
        Ok(spring)
    }

    /// Image-local overdamped step bound `1/(μ λ)` for this bead count.
    ///
    /// `λ` is the stiffest spring eigenvalue plus a Gershgorin upper bound
    /// on the largest bead-surface Hessian eigenvalue, divided by the bead
    /// count. The full Hessian is reconstructed from Hessian-vector products
    /// at every supplied image. This bound certifies those images only: it
    /// does not bound curvature between them or elsewhere in the sampled
    /// hyperplanes. Supply images representative of the region to be sampled.
    ///
    /// Empty `images`, an unavailable Hessian-vector product, or a
    /// non-positive combined bound returns a typed [`SaddleError`].
    pub fn time_step(&self, mobility: f64, images: &[Array1<f64>]) -> Result<f64, SaddleError>
    where
        S: PointSurface,
    {
        if !(mobility.is_finite() && mobility > 0.0) {
            return Err(SaddleError::Invalid(
                "the ring time step needs a finite positive mobility".into(),
            ));
        }
        if images.is_empty() {
            return Err(SaddleError::MissingCurvatureImages);
        }
        let dof = self.dof();
        for image in images {
            if image.len() != dof || !image.iter().all(|value| value.is_finite()) {
                return Err(SaddleError::Shape(
                    "a ring time step needs finite images of one bead".into(),
                ));
            }
        }
        let mut stiff = 0.0;
        for axis in 0..dof {
            let spring = self.axis_spring(axis)?;
            if spring > stiff {
                stiff = spring;
            }
        }
        let mut curvature = 0.0_f64;
        for (index, image) in images.iter().enumerate() {
            let value = self.max_positive_curvature_bound(image.view(), index)?;
            if value > curvature {
                curvature = value;
            }
        }
        let bound = stiff + curvature / self.beads.get() as f64;
        if !bound.is_finite() {
            return Err(SaddleError::NonFinite("ring-polymer stiffness bound"));
        }
        if bound <= 0.0 {
            return Err(SaddleError::NonPositiveStiffnessBound { bound });
        }
        let step = 1.0 / (mobility * bound);
        if !(step.is_finite() && step > 0.0) {
            return Err(SaddleError::NonFinite("ring-polymer time step"));
        }
        Ok(step)
    }

    fn max_positive_curvature_bound(
        &self,
        image: ArrayView1<f64>,
        image_index: usize,
    ) -> Result<f64, SaddleError>
    where
        S: PointSurface,
    {
        let dof = self.dof();
        if image.len() != dof || !image.iter().all(|value| value.is_finite()) {
            return Err(SaddleError::Shape(
                "a ring time step needs finite images of one bead".into(),
            ));
        }
        let mut row_sums: Array1<f64> = Array1::zeros(dof);
        let mut direction = Array1::zeros(dof);
        for column in 0..dof {
            direction[column] = 1.0;
            let Some(product) = self.surface.hessian_vector(image, direction.view())? else {
                return Err(SaddleError::MissingHessianVector { image: image_index });
            };
            direction[column] = 0.0;
            if product.len() != dof {
                return Err(SaddleError::Shape(
                    "a bead Hessian product must match the bead".into(),
                ));
            }
            if !product.iter().all(|value| value.is_finite()) {
                return Err(SaddleError::NonFinite("bead Hessian product"));
            }
            for row in 0..dof {
                row_sums[row] += product[row].abs();
                if !row_sums[row].is_finite() {
                    return Err(SaddleError::NonFinite("ring-polymer stiffness bound"));
                }
            }
        }
        let mut bound = 0.0_f64;
        for row_sum in row_sums {
            if row_sum > bound {
                bound = row_sum;
            }
        }
        Ok(bound)
    }

    fn check_layout(
        &self,
        x: ArrayView1<f64>,
        v: Option<ArrayView1<f64>>,
    ) -> Result<(), SaddleError> {
        let n = self.beads.get() * self.dof();
        if x.len() != n {
            return Err(SaddleError::Shape(
                "ring-polymer coordinates must be beads times one bead".into(),
            ));
        }
        if !x.iter().all(|value| value.is_finite()) {
            return Err(SaddleError::NonFinite("ring-polymer coordinates"));
        }
        if let Some(direction) = v
            && (direction.len() != n || !direction.iter().all(|value| value.is_finite()))
        {
            return Err(SaddleError::Shape(
                "ring-polymer Hessian direction must match the coordinates and be finite".into(),
            ));
        }
        Ok(())
    }
}

impl<S: PointSurface> PointSurface for RingPolymer<'_, S> {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        self.check_layout(x, None)?;
        let dof = self.dof();
        let scale = 1.0 / self.beads.get() as f64;
        let mut energy = 0.0;
        let mut gradient = Array1::zeros(x.len());
        for bead in 0..self.beads.get() {
            let start = bead * dof;
            let (bead_energy, bead_gradient) =
                self.surface.eval(x.slice(s![start..start + dof]))?;
            if bead_gradient.len() != dof {
                return Err(SaddleError::Shape(
                    "a bead gradient must match the bead".into(),
                ));
            }
            if !bead_energy.is_finite() {
                return Err(SaddleError::NonFinite("bead energy"));
            }
            if !bead_gradient.iter().all(|value| value.is_finite()) {
                return Err(SaddleError::NonFinite("bead gradient"));
            }
            energy += scale * bead_energy;
            gradient
                .slice_mut(s![start..start + dof])
                .assign(&(&bead_gradient * scale));
        }
        if self.beads.get() > 1 {
            for bead in 0..self.beads.get() {
                let next = (bead + 1) % self.beads;
                for axis in 0..dof {
                    let left = x[bead * dof + axis];
                    let right = x[next * dof + axis];
                    let gap = left - right;
                    let coefficient = self.link(axis);
                    energy += coefficient * gap * gap;
                    let force = 2.0 * coefficient * gap;
                    gradient[bead * dof + axis] += force;
                    gradient[next * dof + axis] -= force;
                }
            }
        }
        if !energy.is_finite() {
            return Err(SaddleError::NonFinite("ring-polymer energy"));
        }
        if !gradient.iter().all(|value| value.is_finite()) {
            return Err(SaddleError::NonFinite("ring-polymer gradient"));
        }
        Ok((energy, gradient))
    }

    fn hessian_vector(
        &self,
        x: ArrayView1<f64>,
        v: ArrayView1<f64>,
    ) -> Result<Option<Array1<f64>>, SaddleError> {
        self.check_layout(x, Some(v))?;
        let dof = self.dof();
        let scale = 1.0 / self.beads.get() as f64;
        let mut product = Array1::zeros(x.len());
        for bead in 0..self.beads.get() {
            let start = bead * dof;
            let Some(bead_product) = self.surface.hessian_vector(
                x.slice(s![start..start + dof]),
                v.slice(s![start..start + dof]),
            )?
            else {
                return Ok(None);
            };
            if bead_product.len() != dof {
                return Err(SaddleError::Shape(
                    "a bead Hessian product must match the bead".into(),
                ));
            }
            if !bead_product.iter().all(|value| value.is_finite()) {
                return Err(SaddleError::NonFinite("bead Hessian product"));
            }
            product
                .slice_mut(s![start..start + dof])
                .assign(&(&bead_product * scale));
        }
        if self.beads.get() > 1 {
            for bead in 0..self.beads.get() {
                let previous = (bead + self.beads.get() - 1) % self.beads;
                let next = (bead + 1) % self.beads;
                for axis in 0..dof {
                    let here = v[bead * dof + axis];
                    let before = v[previous * dof + axis];
                    let after = v[next * dof + axis];
                    product[bead * dof + axis] +=
                        2.0 * self.link(axis) * (2.0 * here - before - after);
                }
            }
        }
        if !product.iter().all(|value| value.is_finite()) {
            return Err(SaddleError::NonFinite("ring-polymer Hessian product"));
        }
        Ok(Some(product))
    }

    fn excluded_modes(&self, x: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
        self.check_layout(x, None)?;
        let dof = self.dof();
        let mut bases = Vec::with_capacity(self.beads.get());
        for bead in 0..self.beads.get() {
            let start = bead * dof;
            let modes = self
                .surface
                .excluded_modes(x.slice(s![start..start + dof]))?;
            if modes.ncols() != dof {
                return Err(SaddleError::Shape(
                    "a bead excluded mode must match the bead".into(),
                ));
            }
            if modes.iter().any(|value| !value.is_finite()) {
                return Err(SaddleError::NonFinite("ring-polymer excluded mode"));
            }
            bases.push(orthonormal_columns(modes.view()));
        }
        let Some(reference) = bases.first() else {
            return Err(SaddleError::MissingExcludedBasis);
        };
        if reference.ncols() == 0 || bases.iter().any(|basis| basis.ncols() == 0) {
            return Ok(Array2::zeros((0, x.len())));
        }
        if self.beads.get() == 1 {
            return Ok(reference.t().to_owned());
        }
        let mut kept = Vec::new();
        for direction in coherent_intersection(&bases)? {
            let mut candidate = Array1::zeros(x.len());
            for bead in 0..self.beads.get() {
                let start = bead * dof;
                candidate
                    .slice_mut(s![start..start + dof])
                    .assign(&direction);
            }
            candidate /= (self.beads.get() as f64).sqrt();
            kept.push(candidate);
        }
        let residual_bound = f64::EPSILON * dof as f64 * self.beads.get() as f64;
        let isotropic_cartesian_masses = dof.is_multiple_of(3)
            && self.surface.cartesian_layout() != CartesianLayout::NonCartesian
            && self.masses.as_slice().is_some_and(|masses| {
                masses.as_chunks::<3>().0.iter().all(|mass| {
                    mass[0].to_bits() == mass[1].to_bits() && mass[0].to_bits() == mass[2].to_bits()
                })
            });
        let rotations = if isotropic_cartesian_masses {
            rigid_rotation_generators(x, dof, self.beads.get())
        } else {
            Vec::new()
        };
        for candidate in rotations {
            let mut coherent = true;
            for (bead, basis) in bases.iter().enumerate() {
                let start = bead * dof;
                let local = candidate.slice(s![start..start + dof]);
                let projected = project_onto_basis(basis, local);
                let mut residual_norm = 0.0_f64;
                let mut local_norm = 0.0_f64;
                for axis in 0..dof {
                    let residual = local[axis] - projected[axis];
                    residual_norm += residual * residual;
                    local_norm += local[axis] * local[axis];
                }
                if residual_norm > residual_bound * (1.0 + local_norm) {
                    coherent = false;
                    break;
                }
            }
            if coherent {
                kept.push(candidate);
            }
        }
        if kept.is_empty() {
            return Ok(Array2::zeros((0, x.len())));
        }
        let mut columns = Array2::zeros((x.len(), kept.len()));
        for (index, mode) in kept.iter().enumerate() {
            columns.column_mut(index).assign(mode);
        }
        let eps = f64::EPSILON * x.len() as f64;
        Ok(modified_gram_schmidt(columns.view(), None, eps)
            .t()
            .to_owned())
    }

    fn cartesian_layout(&self) -> CartesianLayout {
        if self.beads == NonZeroUsize::MIN {
            self.surface.cartesian_layout()
        } else {
            CartesianLayout::Wrapped
        }
    }
}

/// Orthonormal columns from mode rows. A column already in the span is dropped.
fn orthonormal_columns(modes: ArrayView2<f64>) -> Array2<f64> {
    if modes.nrows() == 0 {
        return Array2::zeros((modes.ncols(), 0));
    }
    let mut normalized_rows = Vec::new();
    for row in modes.rows() {
        let norm = row.dot(&row).sqrt();
        if norm.is_finite() && norm > 0.0 {
            normalized_rows.push(row.to_owned() / norm);
        }
    }
    if normalized_rows.is_empty() {
        return Array2::zeros((modes.ncols(), 0));
    }
    let mut basis: Vec<Array1<f64>> = Vec::new();
    let relative = f64::EPSILON * (modes.nrows() + modes.ncols()) as f64;
    for row in normalized_rows {
        let mut vector = row.clone();
        for _ in 0..2 {
            for direction in &basis {
                let coefficient = direction.dot(&vector);
                vector.scaled_add(-coefficient, direction);
            }
        }
        let norm = vector.dot(&vector).sqrt();
        if norm > relative {
            basis.push(vector / norm);
        }
    }
    if basis.is_empty() {
        return Array2::zeros((modes.ncols(), 0));
    }
    let mut columns = Array2::zeros((modes.ncols(), basis.len()));
    for (index, column) in basis.iter().enumerate() {
        columns.column_mut(index).assign(column);
    }
    columns
}

fn project_onto_basis(basis: &Array2<f64>, vector: ArrayView1<f64>) -> Array1<f64> {
    let mut projected = Array1::zeros(vector.len());
    for column in 0..basis.ncols() {
        let direction = basis.column(column);
        let coefficient = direction.dot(&vector);
        projected.scaled_add(coefficient, &direction);
    }
    projected
}

fn coherent_intersection(bases: &[Array2<f64>]) -> Result<Vec<Array1<f64>>, SaddleError> {
    let Some(reference) = bases.first() else {
        return Err(SaddleError::MissingExcludedBasis);
    };
    let rank = reference.ncols();
    let dof = reference.nrows();
    let mut residual_gram = Array2::zeros((rank, rank));
    for basis in bases.iter().skip(1) {
        let mut residuals = Vec::with_capacity(rank);
        for column in 0..rank {
            let direction = reference.column(column);
            residuals.push(&direction - &project_onto_basis(basis, direction));
        }
        for row in 0..rank {
            for column in 0..rank {
                residual_gram[(row, column)] += residuals[row].dot(&residuals[column]);
            }
        }
    }
    let (values, vectors) = exact_eigh(residual_gram.view())?;
    let roundoff = f64::EPSILON * dof as f64 * bases.len() as f64;
    let mut intersection = Vec::new();
    for index in 0..values.len() {
        if values[index] <= roundoff {
            intersection.push(reference.dot(&vectors.column(index)));
        }
    }
    Ok(intersection)
}

fn rigid_rotation_generators(x: ArrayView1<f64>, dof: usize, beads: usize) -> Vec<Array1<f64>> {
    let mut generators = Vec::new();
    if dof.is_multiple_of(3) {
        let mut center = [0.0_f64; 3];
        let atoms = beads * (dof / 3);
        if atoms > 0 {
            for bead in 0..beads {
                for atom in 0..(dof / 3) {
                    center[0] += x[bead * dof + 3 * atom];
                    center[1] += x[bead * dof + 3 * atom + 1];
                    center[2] += x[bead * dof + 3 * atom + 2];
                }
            }
            for value in &mut center {
                *value /= atoms as f64;
            }
            for axis in 0..3 {
                let mut rotation = Array1::zeros(x.len());
                for bead in 0..beads {
                    for atom in 0..(dof / 3) {
                        let start = bead * dof + 3 * atom;
                        let rx = x[start] - center[0];
                        let ry = x[start + 1] - center[1];
                        let rz = x[start + 2] - center[2];
                        let (dx, dy, dz) = match axis {
                            0 => (0.0, -rz, ry),
                            1 => (rz, 0.0, -rx),
                            _ => (-ry, rx, 0.0),
                        };
                        rotation[start] = dx;
                        rotation[start + 1] = dy;
                        rotation[start + 2] = dz;
                    }
                }
                let norm = rotation.dot(&rotation).sqrt();
                if norm.is_finite() && norm > 0.0 {
                    generators.push(rotation / norm);
                }
            }
        }
    }
    generators
}

/// Flat centroid hyperplane through one physical image.
///
/// The point is [`collapsed`] `image`. The normal is `tangent` repeated
/// on every bead, then normalized by [`Hyperplane::new`]. `tangent` has
/// one entry per degree of freedom of the bead. This is the public
/// [`crate::projector::Hyperplane`] used by [`crate::HyperplaneSampler`].
pub fn centroid_hyperplane(
    beads: usize,
    image: ArrayView1<f64>,
    tangent: ArrayView1<f64>,
) -> Result<Hyperplane, SaddleError> {
    if beads == 0 {
        return Err(SaddleError::Invalid(
            "a centroid hyperplane needs at least one bead".into(),
        ));
    }
    if image.is_empty() || image.len() != tangent.len() {
        return Err(SaddleError::Shape(
            "a centroid hyperplane needs one tangent entry per bead coordinate".into(),
        ));
    }
    if !image
        .iter()
        .chain(tangent.iter())
        .all(|value| value.is_finite())
    {
        return Err(SaddleError::NonFinite("centroid hyperplane"));
    }
    let point = collapsed(beads, image)?;
    let mut normal = Array1::zeros(point.len());
    let dof = image.len();
    for bead in 0..beads {
        for axis in 0..dof {
            normal[bead * dof + axis] = tangent[axis];
        }
    }
    Hyperplane::new(point, normal)
}

/// Every bead at `geometry`.
pub fn collapsed(beads: usize, geometry: ArrayView1<f64>) -> Result<Array1<f64>, SaddleError> {
    if beads == 0 || geometry.is_empty() || !geometry.iter().all(|value| value.is_finite()) {
        return Err(SaddleError::Invalid(
            "a collapsed ring needs a positive bead count and a finite geometry".into(),
        ));
    }
    let mut ring = Array1::zeros(beads * geometry.len());
    for bead in 0..beads {
        let start = bead * geometry.len();
        ring.slice_mut(s![start..start + geometry.len()])
            .assign(&geometry);
    }
    Ok(ring)
}

/// Centroid path: each image of a physical path, repeated on every bead.
///
/// The images are the geometries [`crate::relax_to_tolerance`] and a
/// saddle search already produced. This function does not move them.
pub fn centroid_path(images: &[Array1<f64>], beads: usize) -> Result<ArcPath, SaddleError> {
    if images.is_empty() {
        return Err(SaddleError::Invalid(
            "a centroid path needs the images of the physical path".into(),
        ));
    }
    let width = images[0].len();
    let mut frames = Vec::with_capacity(images.len());
    for image in images {
        if image.len() != width {
            return Err(SaddleError::Shape(
                "centroid images must share one geometry".into(),
            ));
        }
        frames.push(collapsed(beads, image.view())?);
    }
    ArcPath::from_positions(&frames)
}

/// One bead count in a [`bead_study`].
#[derive(Clone, Debug)]
pub struct BeadSample {
    /// Bead count of this sample.
    pub beads: usize,
    /// Potential of mean force on the centroid path at [`Self::beads`].
    ///
    /// `barrier` and `barrier_error` are zero-step extrapolations. The
    /// per-plane `free_energy`, `gradient`, and errors are the values from
    /// the finest, one-eighth-step run and are not extrapolated. Because
    /// [`centroid_path`] repeats each physical image `P` times, `profile.arc`
    /// is `sqrt(P)` times the physical path arc and `profile.gradient` is
    /// divided by `sqrt(P)`; free energies and barriers are invariant.
    pub profile: FreeEnergyProfile,
    /// Estimated leading finite-step correction from the quarter- and
    /// eighth-step barriers.
    pub step_error: f64,
    /// Smallest effective sample count among both refinement step sizes.
    pub effective_samples: f64,
    /// Largest integrated autocorrelation time among both refinement step sizes.
    pub tau: f64,
    /// Time step requested by the caller.
    pub time_step: f64,
}

/// PAFI on the centroid path at each requested bead count.
///
/// `images` is a physical path, reactant first. Each entry of `beads` is
/// one ring. The caller's [`PafiConfig::time_step`] is used as-is when it
/// does not exceed that ring's stiffness bound from [`RingPolymer::time_step`].
/// A larger step returns [`SaddleError::TimeStepAboveBound`].
/// Every sample is run at quarter and eighth steps. Those two refinement
/// results give a first-order zero-step extrapolation for
/// [`PafiIntegrator::EulerMaruyama`] and a second-order extrapolation for
/// [`PafiIntegrator::LeimkuhlerMatthews`]. `profile.samples` counts only
/// the one-eighth-step run. The two refinements use bead-count-specific
/// independent seed streams.
///
/// [`PafiConfig::max_jump`] retains its ordinary Cartesian convention of
/// grouping coordinates in threes. For scalar, planar, or delocalized
/// instanton rings, set it to `None` (or choose a physically justified
/// larger cap) so the default atom-wise basin test does not reject valid
/// ring fluctuations.
/// [`PafiConfig::target_error`] applies independently to the quarter- and
/// eighth-step profiles; the combined extrapolated error is not itself that
/// target.
pub fn bead_study<S: PointSurface + Sync>(
    surface: &S,
    images: &[Array1<f64>],
    beads: &[usize],
    settings: &RingSettings,
    config: &PafiConfig,
) -> Result<Vec<BeadSample>, SaddleError> {
    if beads.is_empty() {
        return Err(SaddleError::Invalid(
            "a bead study needs at least one bead count".into(),
        ));
    }
    if config.fix_com {
        return Err(SaddleError::UnsupportedRingCenterOfMass {
            dof: settings.masses.len(),
        });
    }
    if !(settings.beta.is_finite() && settings.beta > 0.0)
        || !(config.thermal.is_finite() && config.thermal > 0.0)
    {
        return Err(SaddleError::Invalid(
            "ring PAFI beta and thermal energy must be finite and positive".into(),
        ));
    }
    let expected_thermal = settings.beta.recip();
    let relative_mismatch = (config.thermal * settings.beta - 1.0).abs();
    if !(expected_thermal.is_finite() && expected_thermal > 0.0 && relative_mismatch.is_finite())
        || relative_mismatch > f64::EPSILON.sqrt()
    {
        return Err(SaddleError::TemperatureMismatch {
            beta: settings.beta,
            thermal: config.thermal,
            expected_thermal,
        });
    }
    let mut samples = Vec::with_capacity(beads.len());
    let requested_step = config.checked_time_step()?;
    for &count in beads {
        let ring = RingPolymer::new(surface, count, settings.clone())?;
        let bound = ring.time_step(config.mobility, images)?;
        if requested_step > bound {
            return Err(SaddleError::TimeStepAboveBound {
                provided: requested_step,
                bound,
            });
        }
        let step = requested_step;
        let path = centroid_path(images, count)?;
        let arcs = path.arcs();
        let quarter_seed = ring_seed(config.seed, count, 3);
        let eighth_seed = ring_seed(config.seed, count, 4);
        let quarter = config
            .clone()
            .with_time_step(step * 0.25)
            .with_seed(quarter_seed)
            .with_stride(scaled_steps(config.stride, 4, "stride")?)
            .with_thermalize(scaled_steps(config.thermalize, 4, "thermalize")?);
        let (quarter_profile, quarter_planes) = pafi_profile(&ring, &path, &arcs, &quarter)?;
        let eighth = config
            .clone()
            .with_time_step(step * 0.125)
            .with_seed(eighth_seed)
            .with_stride(scaled_steps(config.stride, 8, "stride")?)
            .with_thermalize(scaled_steps(config.thermalize, 8, "thermalize")?);
        let (mut profile, eighth_planes) = pafi_profile(&ring, &path, &arcs, &eighth)?;
        let difference = (profile.barrier - quarter_profile.barrier).abs();
        let step_error = match config.integrator {
            PafiIntegrator::EulerMaruyama => {
                profile.barrier = 2.0 * profile.barrier - quarter_profile.barrier;
                profile.barrier_error = (4.0 * profile.barrier_error * profile.barrier_error
                    + quarter_profile.barrier_error * quarter_profile.barrier_error)
                    .sqrt();
                difference
            }
            PafiIntegrator::LeimkuhlerMatthews => {
                profile.barrier = (4.0 * profile.barrier - quarter_profile.barrier) / 3.0;
                profile.barrier_error = ((16.0 * profile.barrier_error * profile.barrier_error
                    + quarter_profile.barrier_error * quarter_profile.barrier_error)
                    / 9.0)
                    .sqrt();
                difference / 3.0
            }
        };
        let (effective_samples, tau) = conservative_correlation(&quarter_planes, &eighth_planes)?;
        samples.push(BeadSample {
            beads: count,
            profile,
            step_error,
            effective_samples,
            tau,
            time_step: step,
        });
    }
    Ok(samples)
}

fn ring_seed(base: u64, beads: usize, refinement: u64) -> u64 {
    let mut value = base
        ^ (beads as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ refinement.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn scaled_steps(value: usize, factor: usize, field: &'static str) -> Result<usize, SaddleError> {
    value.checked_mul(factor).ok_or_else(|| {
        SaddleError::Invalid(format!(
            "ring PAFI {field} overflows when the time step is divided by {factor}"
        ))
    })
}

fn conservative_correlation(
    quarter_planes: &[crate::pafi::HyperplaneAverage],
    eighth_planes: &[crate::pafi::HyperplaneAverage],
) -> Result<(f64, f64), SaddleError> {
    let mut observations = quarter_planes.iter().chain(eighth_planes.iter());
    let Some(first) = observations.next() else {
        return Err(SaddleError::MissingSamples {
            context: "ring PAFI planes",
        });
    };
    check_correlation(first.n_eff, first.tau)?;
    let mut effective_samples = first.n_eff;
    let mut tau = first.tau;
    for plane in observations {
        check_correlation(plane.n_eff, plane.tau)?;
        if plane.n_eff < effective_samples {
            effective_samples = plane.n_eff;
        }
        if plane.tau > tau {
            tau = plane.tau;
        }
    }
    Ok((effective_samples, tau))
}

fn check_correlation(n_eff: f64, tau: f64) -> Result<(), SaddleError> {
    if n_eff.is_finite() && tau.is_finite() {
        Ok(())
    } else {
        Err(SaddleError::NonFinite("ring PAFI correlation"))
    }
}

/// Instanton orbit read from a `.con`: one frame per bead, in bead order.
#[cfg(feature = "readcon")]
#[derive(Clone, Debug)]
pub struct RingCon {
    /// Bead coordinates concatenated in frame order.
    pub orbit: Array1<f64>,
    /// Mass of each Cartesian degree of freedom of one bead.
    pub masses: Array1<f64>,
    /// Frames consumed as beads.
    pub beads: usize,
}

/// One instanton orbit from a `.con`.
///
/// readcon-core reads the frames. Frame `i` is bead `i`, and the frame's
/// Cartesian coordinates are that bead. Every frame has the same atom
/// count and the same masses. The mass of an atom is copied onto its
/// three Cartesian degrees of freedom.
#[cfg(feature = "readcon")]
pub fn orbit_from_con(path: impl AsRef<std::path::Path>) -> Result<RingCon, SaddleError> {
    let frames = crate::io::frames_from_con(path)?;
    if frames.is_empty() {
        return Err(SaddleError::Invalid(
            "an instanton orbit needs one frame per bead".into(),
        ));
    }
    pack_frames(&frames, frames.len())
}

/// Path whose frames are rings read from a `.con`.
///
/// Frames are grouped in order, `beads` frames per ring. The first group
/// is the first path image. A file needs at least two groups.
#[cfg(feature = "readcon")]
pub fn ring_path_from_con(
    path: impl AsRef<std::path::Path>,
    beads: usize,
) -> Result<(ArcPath, RingCon), SaddleError> {
    if beads == 0 {
        return Err(SaddleError::Invalid(
            "a ring-polymer path needs at least one bead".into(),
        ));
    }
    let frames = crate::io::frames_from_con(path)?;
    if frames.is_empty() || !frames.len().is_multiple_of(beads) {
        return Err(SaddleError::Invalid(
            "ring-polymer frames must form complete rings of the requested bead count".into(),
        ));
    }
    let rings = frames.len() / beads;
    if rings < 2 {
        return Err(SaddleError::Invalid(
            "a ring-polymer path needs at least two orbits".into(),
        ));
    }
    let first = pack_frames(&frames[..beads], beads)?;
    let mut images = Vec::with_capacity(rings);
    images.push(first.orbit.clone());
    for ring in 1..rings {
        let group = &frames[ring * beads..(ring + 1) * beads];
        let packed = pack_frames(group, beads)?;
        same_masses(&first.masses, &packed.masses)?;
        images.push(packed.orbit);
    }
    let sample = RingCon {
        orbit: first.orbit,
        masses: first.masses,
        beads,
    };
    Ok((ArcPath::from_positions(&images)?, sample))
}

#[cfg(feature = "readcon")]
fn pack_frames(frames: &[crate::io::MolecularFrame], beads: usize) -> Result<RingCon, SaddleError> {
    if frames.len() != beads {
        return Err(SaddleError::Shape(
            "an orbit group must contain one frame per bead".into(),
        ));
    }
    let masses = cartesian_masses(&frames[0])?;
    let width = frames[0].positions.len();
    if width != masses.len() {
        return Err(SaddleError::Shape(
            "bead coordinates and Cartesian masses differ in length".into(),
        ));
    }
    let mut orbit = Array1::zeros(beads * width);
    for (bead, frame) in frames.iter().enumerate() {
        if frame.positions.len() != width || !frame.positions.iter().all(|value| value.is_finite())
        {
            return Err(SaddleError::Shape(
                "every bead frame must be finite and the same length".into(),
            ));
        }
        same_masses(&masses, &cartesian_masses(frame)?)?;
        let start = bead * width;
        orbit
            .slice_mut(s![start..start + width])
            .assign(&frame.positions);
    }
    Ok(RingCon {
        orbit,
        masses,
        beads,
    })
}

#[cfg(feature = "readcon")]
fn cartesian_masses(frame: &crate::io::MolecularFrame) -> Result<Array1<f64>, SaddleError> {
    if frame.positions.len() != frame.masses.len() * 3 {
        return Err(SaddleError::Shape(
            "a bead frame must hold three Cartesian coordinates per atom".into(),
        ));
    }
    let mut masses = Array1::zeros(frame.positions.len());
    for (atom, &mass) in frame.masses.iter().enumerate() {
        if !(mass.is_finite() && mass > 0.0) {
            return Err(SaddleError::Invalid(
                "a bead mass must be finite and positive".into(),
            ));
        }
        for axis in 0..3 {
            masses[atom * 3 + axis] = mass;
        }
    }
    Ok(masses)
}

#[cfg(feature = "readcon")]
fn same_masses(reference: &Array1<f64>, other: &Array1<f64>) -> Result<(), SaddleError> {
    if reference.len() != other.len()
        || reference
            .iter()
            .zip(other.iter())
            .any(|(left, right)| left.to_bits() != right.to_bits())
    {
        return Err(SaddleError::Invalid(
            "every bead frame must carry the same masses".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::check_correlation;
    use crate::SaddleError;

    #[test]
    fn correlation_rejects_nonfinite_plane_statistics() {
        for (n_eff, tau) in [
            (f64::NAN, 0.5),
            (f64::INFINITY, 0.5),
            (1.0, f64::NAN),
            (1.0, f64::INFINITY),
        ] {
            assert!(matches!(
                check_correlation(n_eff, tau),
                Err(SaddleError::NonFinite("ring PAFI correlation"))
            ));
        }
    }
}
