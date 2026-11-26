//! Mass weights. Coordinates stored by the session are `q = M^{1/2} x`.

use ndarray::{Array1, ArrayView1};

use crate::error::SaddleError;
use crate::mass_metric::{MassMetric, MetricCovector, MetricPoint};

/// Per-coordinate masses, and the atom count when the system is a molecule.
#[derive(Clone, Debug)]
pub struct Masses {
    metric: MassMetric,
    /// `Some(atoms)` turns on quaternion alignment. The three coordinates
    /// of an atom share one atomic mass.
    pub(crate) atoms: Option<usize>,
    /// Fit each image to its aligned neighbour instead of the first image.
    pub(crate) adjacent: bool,
}

impl Masses {
    /// One positive finite mass per coordinate. No rigid alignment.
    pub fn coordinates(mass: ArrayView1<f64>) -> Result<Self, SaddleError> {
        if mass.is_empty() {
            return Err(SaddleError::Shape(
                "a tunneling path needs a coordinate".into(),
            ));
        }
        for &value in mass {
            if !(value.is_finite() && value > 0.0) {
                return Err(SaddleError::Invalid(
                    "coordinate masses must be finite and positive".into(),
                ));
            }
        }
        Ok(Self {
            metric: MassMetric::coordinates(mass, mass.len())?,
            atoms: None,
            adjacent: false,
        })
    }

    /// One positive finite mass per atom, expanded to three coordinates,
    /// with quaternion alignment.
    pub fn molecule(atom_mass: &[f64]) -> Result<Self, SaddleError> {
        if atom_mass.len() < 2 {
            return Err(SaddleError::Invalid(
                "quaternion alignment needs at least two atoms".into(),
            ));
        }
        for &value in atom_mass {
            if !(value.is_finite() && value > 0.0) {
                return Err(SaddleError::Invalid(
                    "atomic masses must be finite and positive".into(),
                ));
            }
        }
        let cartesian_len = atom_mass.len().checked_mul(3).ok_or_else(|| {
            SaddleError::Shape("atom count does not fit in a coordinate length".into())
        })?;
        Ok(Self {
            metric: MassMetric::atoms(Array1::from(atom_mass.to_vec()).view(), cartesian_len)?,
            atoms: Some(atom_mass.len()),
            adjacent: false,
        })
    }

    /// Choose the reference of the quaternion alignment for a molecule.
    ///
    /// By default every image is fitted to the first image. With
    /// `adjacent` each image is fitted to its aligned neighbour, so every
    /// chord has its minimum mass-weighted length. Ásgeirsson, Arnaldsson
    /// and Jónsson, J. Chem. Phys. 148, 102334 (2018) minimize the
    /// deviation of adjacent images in this way. The two choices agree
    /// for short paths and differ once the band turns far from its first
    /// image. The setting has no effect without atomic masses.
    pub fn with_adjacent_alignment(mut self, adjacent: bool) -> Self {
        self.adjacent = adjacent;
        self
    }

    pub(crate) fn dof(&self) -> usize {
        self.metric.len()
    }

    pub(crate) fn metric(&self) -> &MassMetric {
        &self.metric
    }

    pub(crate) fn sqrt_at(&self, index: usize) -> f64 {
        self.metric.sqrt_at(index)
    }

    pub(crate) fn inv_sqrt_at(&self, index: usize) -> f64 {
        self.metric.inv_sqrt_at(index)
    }

    pub(crate) fn sqrt_vector(&self) -> Array1<f64> {
        self.metric.sqrt_vector()
    }

    pub(crate) fn to_weighted(
        &self,
        cartesian: ArrayView1<f64>,
    ) -> Result<Array1<f64>, SaddleError> {
        if cartesian.len() != self.dof() {
            return Err(SaddleError::Shape(
                "Cartesian vector length must match the masses".into(),
            ));
        }
        self.metric.weight_point(MetricPoint::new(cartesian))
    }

    pub(crate) fn to_cartesian(
        &self,
        weighted: ArrayView1<f64>,
    ) -> Result<Array1<f64>, SaddleError> {
        if weighted.len() != self.dof() {
            return Err(SaddleError::Shape(
                "mass-weighted vector length must match the masses".into(),
            ));
        }
        self.metric.unweight_point(MetricPoint::new(weighted))
    }

    /// Mass-weighted force `−M^{−1/2} ∇_x V` from a Cartesian gradient.
    pub(crate) fn weighted_force(
        &self,
        gradient: ArrayView1<f64>,
    ) -> Result<Array1<f64>, SaddleError> {
        if gradient.len() != self.dof() {
            return Err(SaddleError::Shape(
                "gradient length must match the masses".into(),
            ));
        }
        let mut force = gradient.to_owned();
        for value in force.iter_mut() {
            *value = -*value;
        }
        self.metric
            .weight_covector(MetricCovector::new(force.view()))
    }
}
