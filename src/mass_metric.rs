//! Diagonal mass metric.
//!
//! A point and a tangent transform as `q = M^{1/2} x`. A covector, whether
//! a force or an energy gradient, transforms as `c_q = M^{-1/2} c`. The
//! metric stores each square root and inverse square root once. When every
//! mass is one, or when the metric is built as a unit metric, those vectors
//! are not stored and the maps copy their input.
//!
//! The input wrappers are different types, so a force cannot be passed to a
//! point or tangent map. The same wrapper is used on either side of a map.
//! The method name is the direction: `weight_*` multiplies a point or a
//! tangent by `M^{1/2}` and divides a covector by it, and `unweight_*` is
//! the inverse.
//!
//! Masses are numbers in the caller's units. The metric does not name a
//! coordinate system beyond the length of the mass list: one mass per
//! coordinate, or one mass per atom expanded to three coordinates.
//!
//! # Examples
//!
//! ```
//! use ndarray::array;
//! use rgsaddle::{MassMetric, MetricCovector, MetricPoint, MetricTangent, SaddleError};
//!
//! let metric = MassMetric::coordinates(array![1.0, 4.0].view(), 2)?;
//! let weighted = metric.weight_point(MetricPoint::new(array![2.0, 3.0].view()))?;
//! assert_eq!(weighted.as_slice(), Some(&[2.0, 6.0][..]));
//! let gradient = metric.weight_covector(MetricCovector::new(array![2.0, 6.0].view()))?;
//! assert_eq!(gradient.as_slice(), Some(&[2.0, 3.0][..]));
//! let tangent = metric.weight_tangent(MetricTangent::new(array![2.0, 3.0].view()))?;
//! assert_eq!(
//!     metric.norm(MetricTangent::new(array![2.0, 3.0].view()))?,
//!     tangent.dot(&tangent).sqrt()
//! );
//! # Ok::<(), SaddleError>(())
//! ```

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::error::SaddleError;

/// View of a point.
///
/// [`MassMetric::weight_point`] and [`MassMetric::unweight_point`] accept
/// this wrapper and do not accept a [`MetricTangent`] or a [`MetricCovector`].
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct MetricPoint<'a> {
    components: ArrayView1<'a, f64>,
}

/// View of a tangent vector.
///
/// [`MassMetric::weight_tangent`], [`MassMetric::unweight_tangent`],
/// [`MassMetric::inner`], and [`MassMetric::norm`] accept this wrapper.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct MetricTangent<'a> {
    components: ArrayView1<'a, f64>,
}

/// View of a covector, such as a force or an energy gradient.
///
/// [`MassMetric::weight_covector`] and [`MassMetric::unweight_covector`]
/// accept this wrapper and do not accept a point or a tangent.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct MetricCovector<'a> {
    components: ArrayView1<'a, f64>,
}

impl<'a> MetricPoint<'a> {
    /// Wrap `components`.
    #[must_use]
    pub fn new(components: ArrayView1<'a, f64>) -> Self {
        Self { components }
    }

    /// The wrapped components.
    #[must_use]
    pub fn components(self) -> ArrayView1<'a, f64> {
        self.components
    }
}

impl<'a> MetricTangent<'a> {
    /// Wrap `components`.
    #[must_use]
    pub fn new(components: ArrayView1<'a, f64>) -> Self {
        Self { components }
    }

    /// The wrapped components.
    #[must_use]
    pub fn components(self) -> ArrayView1<'a, f64> {
        self.components
    }
}

impl<'a> MetricCovector<'a> {
    /// Wrap `components`.
    #[must_use]
    pub fn new(components: ArrayView1<'a, f64>) -> Self {
        Self { components }
    }

    /// The wrapped components.
    #[must_use]
    pub fn components(self) -> ArrayView1<'a, f64> {
        self.components
    }
}

#[derive(Clone, Debug)]
enum Scale {
    Identity {
        len: usize,
    },
    Diagonal {
        sqrt_mass: Array1<f64>,
        inv_sqrt_mass: Array1<f64>,
    },
}

#[derive(Clone, Copy)]
enum MapOp {
    Multiply,
    Divide,
}

/// Checked diagonal mass metric.
///
/// Build one with [`MassMetric::coordinates`], [`MassMetric::atoms`], or
/// [`MassMetric::unit`]. A non-positive, non-finite, or wrongly sized mass
/// list is an error. The metric does not replace a bad mass with a default.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct MassMetric {
    scale: Scale,
}

impl MassMetric {
    /// One positive finite mass per coordinate.
    ///
    /// `masses.len()` must equal `dimension`, and `dimension` must be nonzero.
    ///
    /// # Errors
    ///
    /// [`SaddleError::Shape`] when the length is wrong.
    /// [`SaddleError::NonFinite`] when a mass is not finite.
    /// [`SaddleError::Invalid`] when a mass is not positive.
    pub fn coordinates(masses: ArrayView1<'_, f64>, dimension: usize) -> Result<Self, SaddleError> {
        if dimension == 0 || masses.len() != dimension {
            return Err(SaddleError::Shape(
                "one positive mass per coordinate".into(),
            ));
        }
        Ok(Self {
            scale: coordinate_scale(masses)?,
        })
    }

    /// One positive finite mass per atom, repeated on three coordinates.
    ///
    /// `cartesian_len` must equal three times the number of atoms, and that
    /// number must be nonzero.
    ///
    /// # Errors
    ///
    /// [`SaddleError::Shape`] when the length is wrong or the atom count
    /// does not fit in a coordinate length. [`SaddleError::NonFinite`] when
    /// a mass is not finite. [`SaddleError::Invalid`] when a mass is not
    /// positive.
    pub fn atoms(
        atom_masses: ArrayView1<'_, f64>,
        cartesian_len: usize,
    ) -> Result<Self, SaddleError> {
        let dof = atom_masses.len().checked_mul(3).ok_or_else(|| {
            SaddleError::Shape("atom count does not fit in a coordinate length".into())
        })?;
        if atom_masses.is_empty() || dof != cartesian_len {
            return Err(SaddleError::Shape(
                "one positive mass per atom, three coordinates each".into(),
            ));
        }
        Ok(Self {
            scale: atom_scale(atom_masses)?,
        })
    }

    /// Square-root weights that were already checked.
    ///
    /// A ring of images repeats one image's weights. Building a new metric
    /// from the squared roots and taking the square root again would not
    /// reproduce those weights.
    pub(crate) fn from_square_roots(sqrt_mass: ArrayView1<'_, f64>) -> Result<Self, SaddleError> {
        if sqrt_mass.is_empty() {
            return Err(SaddleError::Shape(
                "a mass metric needs a coordinate".into(),
            ));
        }
        let mut all_unit = true;
        for &root in sqrt_mass {
            if !root.is_finite() {
                return Err(SaddleError::NonFinite("mass"));
            }
            if root <= 0.0 {
                return Err(SaddleError::Invalid("masses must be positive".into()));
            }
            if root != 1.0 {
                all_unit = false;
            }
        }
        if all_unit {
            return Ok(Self {
                scale: Scale::Identity {
                    len: sqrt_mass.len(),
                },
            });
        }
        let mut inv_sqrt_mass = Array1::zeros(sqrt_mass.len());
        for (index, &root) in sqrt_mass.iter().enumerate() {
            inv_sqrt_mass[index] = 1.0 / root;
        }
        Ok(Self {
            scale: Scale::Diagonal {
                sqrt_mass: sqrt_mass.to_owned(),
                inv_sqrt_mass,
            },
        })
    }

    /// Unit mass on each of `dimension` coordinates.
    ///
    /// The maps copy. No square-root vector is stored.
    ///
    /// # Errors
    ///
    /// [`SaddleError::Shape`] when `dimension` is zero.
    pub fn unit(dimension: usize) -> Result<Self, SaddleError> {
        if dimension == 0 {
            return Err(SaddleError::Shape(
                "a mass metric needs a coordinate".into(),
            ));
        }
        Ok(Self {
            scale: Scale::Identity { len: dimension },
        })
    }

    /// Coordinate dimension.
    #[must_use]
    pub fn len(&self) -> usize {
        match &self.scale {
            Scale::Identity { len } => *len,
            Scale::Diagonal { sqrt_mass, .. } => sqrt_mass.len(),
        }
    }

    /// `false`. A metric rejects an empty mass list at construction.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// `true` when every coordinate has unit mass and the maps copy.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        matches!(&self.scale, Scale::Identity { .. })
    }

    /// Square root of the mass on coordinate `index`.
    ///
    /// # Errors
    ///
    /// [`SaddleError::Shape`] when `index` is outside the dimension.
    pub fn square_root(&self, index: usize) -> Result<f64, SaddleError> {
        if index >= self.len() {
            return Err(SaddleError::Shape(
                "mass metric index is outside the coordinate dimension".into(),
            ));
        }
        Ok(self.sqrt_at(index))
    }

    /// Inverse square root of the mass on coordinate `index`.
    ///
    /// The stored value is `1 / square_root(index)`.
    ///
    /// # Errors
    ///
    /// [`SaddleError::Shape`] when `index` is outside the dimension.
    pub fn inverse_square_root(&self, index: usize) -> Result<f64, SaddleError> {
        if index >= self.len() {
            return Err(SaddleError::Shape(
                "mass metric index is outside the coordinate dimension".into(),
            ));
        }
        Ok(self.inv_sqrt_at(index))
    }

    /// `M^{1/2} x` for a point.
    ///
    /// # Errors
    ///
    /// [`SaddleError::Shape`] when the point length is not the dimension.
    pub fn weight_point(&self, point: MetricPoint<'_>) -> Result<Array1<f64>, SaddleError> {
        self.map_checked(point.components, MapOp::Multiply)
    }

    /// `M^{-1/2} q` for a point.
    ///
    /// # Errors
    ///
    /// [`SaddleError::Shape`] when the point length is not the dimension.
    pub fn unweight_point(&self, point: MetricPoint<'_>) -> Result<Array1<f64>, SaddleError> {
        self.map_checked(point.components, MapOp::Divide)
    }

    /// `M^{1/2} v` for a tangent.
    ///
    /// # Errors
    ///
    /// [`SaddleError::Shape`] when the tangent length is not the dimension.
    pub fn weight_tangent(&self, tangent: MetricTangent<'_>) -> Result<Array1<f64>, SaddleError> {
        self.map_checked(tangent.components, MapOp::Multiply)
    }

    /// `M^{-1/2} v` for a tangent.
    ///
    /// # Errors
    ///
    /// [`SaddleError::Shape`] when the tangent length is not the dimension.
    pub fn unweight_tangent(&self, tangent: MetricTangent<'_>) -> Result<Array1<f64>, SaddleError> {
        self.map_checked(tangent.components, MapOp::Divide)
    }

    /// `M^{-1/2} c` for a covector.
    ///
    /// # Errors
    ///
    /// [`SaddleError::Shape`] when the covector length is not the dimension.
    pub fn weight_covector(
        &self,
        covector: MetricCovector<'_>,
    ) -> Result<Array1<f64>, SaddleError> {
        self.map_checked(covector.components, MapOp::Divide)
    }

    /// `M^{1/2} c` for a covector.
    ///
    /// # Errors
    ///
    /// [`SaddleError::Shape`] when the covector length is not the dimension.
    pub fn unweight_covector(
        &self,
        covector: MetricCovector<'_>,
    ) -> Result<Array1<f64>, SaddleError> {
        self.map_checked(covector.components, MapOp::Multiply)
    }

    /// Tangent inner product `u^T M v`.
    ///
    /// This is the Euclidean product of the mass-weighted tangents.
    ///
    /// # Errors
    ///
    /// [`SaddleError::Shape`] when either tangent length is not the dimension.
    pub fn inner(
        &self,
        left: MetricTangent<'_>,
        right: MetricTangent<'_>,
    ) -> Result<f64, SaddleError> {
        let left = self.weight_tangent(left)?;
        let right = self.weight_tangent(right)?;
        Ok(left.dot(&right))
    }

    /// Tangent norm `sqrt(u^T M u)`.
    ///
    /// # Errors
    ///
    /// [`SaddleError::Shape`] when the tangent length is not the dimension.
    pub fn norm(&self, tangent: MetricTangent<'_>) -> Result<f64, SaddleError> {
        let weighted = self.weight_tangent(tangent)?;
        Ok(weighted.dot(&weighted).sqrt())
    }

    /// Square-root weights, with ones for a unit metric.
    ///
    /// Callers that hand the weights to another crate clone this vector.
    /// The maps above do not.
    #[must_use]
    pub(crate) fn sqrt_vector(&self) -> Array1<f64> {
        match &self.scale {
            Scale::Identity { len } => Array1::ones(*len),
            Scale::Diagonal { sqrt_mass, .. } => sqrt_mass.clone(),
        }
    }

    #[must_use]
    pub(crate) fn sqrt_at(&self, index: usize) -> f64 {
        match &self.scale {
            Scale::Identity { .. } => 1.0,
            Scale::Diagonal { sqrt_mass, .. } => sqrt_mass[index],
        }
    }

    #[must_use]
    pub(crate) fn inv_sqrt_at(&self, index: usize) -> f64 {
        match &self.scale {
            Scale::Identity { .. } => 1.0,
            Scale::Diagonal { inv_sqrt_mass, .. } => inv_sqrt_mass[index],
        }
    }

    /// `M^{1/2} x` when `src` has the metric length.
    ///
    /// Debug builds assert that equality. Session code calls this only
    /// after its own length check. [`Self::weight_point`] and the other
    /// public maps return [`SaddleError::Shape`] for a mismatch.
    #[must_use]
    pub(crate) fn multiply_sqrt(&self, src: ArrayView1<'_, f64>) -> Array1<f64> {
        self.scaled(src, MapOp::Multiply)
    }

    /// `M^{-1/2} x` when `src` has the metric length.
    ///
    /// Debug builds assert that equality. The public maps return
    /// [`SaddleError::Shape`] for a mismatch.
    #[must_use]
    pub(crate) fn divide_sqrt(&self, src: ArrayView1<'_, f64>) -> Array1<f64> {
        self.scaled(src, MapOp::Divide)
    }

    #[must_use]
    pub(crate) fn weight_point_rows(&self, rows: ArrayView2<'_, f64>) -> Array2<f64> {
        self.map_rows(rows, MapOp::Multiply, false)
    }

    #[must_use]
    pub(crate) fn unweight_point_rows(&self, rows: ArrayView2<'_, f64>) -> Array2<f64> {
        self.map_rows(rows, MapOp::Divide, false)
    }

    /// `(-c) / sqrt(m)` on every row. The negation is on the covector, then
    /// the covector map divides.
    #[must_use]
    pub(crate) fn weight_force_rows(&self, rows: ArrayView2<'_, f64>) -> Array2<f64> {
        self.map_rows(rows, MapOp::Divide, true)
    }

    #[must_use]
    pub(crate) fn unweight_covector_rows(&self, rows: ArrayView2<'_, f64>) -> Array2<f64> {
        self.map_rows(rows, MapOp::Multiply, false)
    }

    fn map_checked(&self, src: ArrayView1<'_, f64>, op: MapOp) -> Result<Array1<f64>, SaddleError> {
        if src.len() != self.len() {
            return Err(SaddleError::Shape(
                "mass metric vector length must match the masses".into(),
            ));
        }
        Ok(self.scaled(src, op))
    }

    fn scaled(&self, src: ArrayView1<'_, f64>, op: MapOp) -> Array1<f64> {
        debug_assert_eq!(
            src.len(),
            self.len(),
            "mass metric vector length must match the masses"
        );
        match &self.scale {
            Scale::Identity { .. } => src.to_owned(),
            Scale::Diagonal { sqrt_mass, .. } => {
                let mut out = Array1::zeros(src.len());
                for index in 0..src.len() {
                    let weight = sqrt_mass[index];
                    out[index] = match op {
                        MapOp::Multiply => src[index] * weight,
                        MapOp::Divide => src[index] / weight,
                    };
                }
                out
            }
        }
    }

    fn map_rows(&self, rows: ArrayView2<'_, f64>, op: MapOp, negate: bool) -> Array2<f64> {
        debug_assert_eq!(
            rows.ncols(),
            self.len(),
            "mass metric row width must match the masses"
        );
        let mut out = rows.to_owned();
        match &self.scale {
            Scale::Identity { .. } => {
                if negate {
                    out.mapv_inplace(|value| -value);
                }
            }
            Scale::Diagonal { sqrt_mass, .. } => {
                let width = out.ncols();
                match (op, negate) {
                    (MapOp::Multiply, false) => {
                        for mut row in out.rows_mut() {
                            for index in 0..width {
                                row[index] *= sqrt_mass[index];
                            }
                        }
                    }
                    (MapOp::Divide, false) => {
                        for mut row in out.rows_mut() {
                            for index in 0..width {
                                row[index] /= sqrt_mass[index];
                            }
                        }
                    }
                    (MapOp::Divide, true) => {
                        for mut row in out.rows_mut() {
                            for index in 0..width {
                                row[index] = -row[index] / sqrt_mass[index];
                            }
                        }
                    }
                    (MapOp::Multiply, true) => {
                        for mut row in out.rows_mut() {
                            for index in 0..width {
                                row[index] = -row[index] * sqrt_mass[index];
                            }
                        }
                    }
                }
            }
        }
        out
    }
}

fn coordinate_scale(masses: ArrayView1<'_, f64>) -> Result<Scale, SaddleError> {
    let mut all_unit = true;
    for &mass in masses {
        check_mass(mass)?;
        if mass != 1.0 {
            all_unit = false;
        }
    }
    if all_unit {
        return Ok(Scale::Identity { len: masses.len() });
    }
    let mut sqrt_mass = Array1::zeros(masses.len());
    let mut inv_sqrt_mass = Array1::zeros(masses.len());
    for (index, &mass) in masses.iter().enumerate() {
        let root = mass.sqrt();
        sqrt_mass[index] = root;
        inv_sqrt_mass[index] = 1.0 / root;
    }
    Ok(Scale::Diagonal {
        sqrt_mass,
        inv_sqrt_mass,
    })
}

fn atom_scale(atom_masses: ArrayView1<'_, f64>) -> Result<Scale, SaddleError> {
    let mut all_unit = true;
    for &mass in atom_masses {
        check_mass(mass)?;
        if mass != 1.0 {
            all_unit = false;
        }
    }
    let len = atom_masses.len().saturating_mul(3);
    if all_unit {
        return Ok(Scale::Identity { len });
    }
    let mut sqrt_mass = Array1::zeros(len);
    let mut inv_sqrt_mass = Array1::zeros(len);
    for (atom, &mass) in atom_masses.iter().enumerate() {
        let root = mass.sqrt();
        let inverse = 1.0 / root;
        for axis in 0..3 {
            sqrt_mass[3 * atom + axis] = root;
            inv_sqrt_mass[3 * atom + axis] = inverse;
        }
    }
    Ok(Scale::Diagonal {
        sqrt_mass,
        inv_sqrt_mass,
    })
}

fn check_mass(mass: f64) -> Result<(), SaddleError> {
    if !mass.is_finite() {
        return Err(SaddleError::NonFinite("mass"));
    }
    if mass <= 0.0 {
        return Err(SaddleError::Invalid("masses must be positive".into()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn sample() -> MassMetric {
        MassMetric::coordinates(array![1.0, 4.0, 9.0].view(), 3).unwrap()
    }

    #[test]
    fn maps_round_trip_for_points_tangents_and_covectors() {
        let metric = sample();
        let values = array![0.5, -1.5, 0.25];
        let point = metric
            .weight_point(MetricPoint::new(values.view()))
            .unwrap();
        let point_back = metric
            .unweight_point(MetricPoint::new(point.view()))
            .unwrap();
        assert_eq!(point_back.as_slice(), values.as_slice());
        assert_eq!(point.as_slice(), Some(&[0.5, -3.0, 0.75][..]));

        let tangent = metric
            .weight_tangent(MetricTangent::new(values.view()))
            .unwrap();
        let tangent_back = metric
            .unweight_tangent(MetricTangent::new(tangent.view()))
            .unwrap();
        assert_eq!(tangent_back.as_slice(), values.as_slice());

        let covector = metric
            .weight_covector(MetricCovector::new(values.view()))
            .unwrap();
        let covector_back = metric
            .unweight_covector(MetricCovector::new(covector.view()))
            .unwrap();
        assert_eq!(covector_back.as_slice(), values.as_slice());
        assert_eq!(covector[0], 0.5);
        assert_eq!(covector[1], -0.75);
        assert_eq!(covector[2], values[2] / 3.0);
    }

    #[test]
    fn atom_masses_repeat_on_three_coordinates_and_round_trip() {
        let metric = MassMetric::atoms(array![1.0, 4.0].view(), 6).unwrap();
        assert_eq!(metric.len(), 6);
        assert!(!metric.is_identity());
        assert_eq!(metric.square_root(0).unwrap(), 1.0);
        assert_eq!(metric.square_root(3).unwrap(), 2.0);
        assert_eq!(metric.inverse_square_root(3).unwrap(), 0.5);
        assert_eq!(
            metric.square_root(3).unwrap() * metric.inverse_square_root(3).unwrap(),
            1.0
        );
        let values = array![1.0, -2.0, 0.5, 3.0, 0.25, -0.5];
        let weighted = metric
            .weight_point(MetricPoint::new(values.view()))
            .unwrap();
        let back = metric
            .unweight_point(MetricPoint::new(weighted.view()))
            .unwrap();
        assert_eq!(back.as_slice(), values.as_slice());
    }

    #[test]
    fn unit_masses_copy_without_scaling() {
        let metric = MassMetric::unit(4).unwrap();
        assert!(metric.is_identity());
        assert!(metric.sqrt_vector().iter().all(|weight| *weight == 1.0));
        let explicit = MassMetric::coordinates(array![1.0, 1.0, 1.0, 1.0].view(), 4).unwrap();
        assert!(explicit.is_identity());
        let values = array![1.0, -0.0, 3.5, 8.0];
        for metric in [&metric, &explicit] {
            let point = metric
                .weight_point(MetricPoint::new(values.view()))
                .unwrap();
            assert_eq!(point.as_slice(), values.as_slice());
            let tangent = metric
                .unweight_tangent(MetricTangent::new(values.view()))
                .unwrap();
            assert_eq!(tangent.as_slice(), values.as_slice());
            let covector = metric
                .weight_covector(MetricCovector::new(values.view()))
                .unwrap();
            assert_eq!(covector.as_slice(), values.as_slice());
        }
    }

    #[test]
    fn bad_masses_and_wrong_lengths_are_rejected() {
        assert!(matches!(
            MassMetric::coordinates(array![].view(), 0),
            Err(SaddleError::Shape(_))
        ));
        assert!(matches!(
            MassMetric::coordinates(array![1.0, 2.0].view(), 3),
            Err(SaddleError::Shape(_))
        ));
        assert!(matches!(
            MassMetric::coordinates(array![0.0].view(), 1),
            Err(SaddleError::Invalid(_))
        ));
        assert!(matches!(
            MassMetric::coordinates(array![-2.0].view(), 1),
            Err(SaddleError::Invalid(_))
        ));
        assert!(matches!(
            MassMetric::coordinates(array![f64::NAN].view(), 1),
            Err(SaddleError::NonFinite(_))
        ));
        assert!(matches!(
            MassMetric::coordinates(array![f64::INFINITY].view(), 1),
            Err(SaddleError::NonFinite(_))
        ));
        assert!(matches!(
            MassMetric::atoms(array![].view(), 0),
            Err(SaddleError::Shape(_))
        ));
        assert!(matches!(
            MassMetric::atoms(array![1.0].view(), 4),
            Err(SaddleError::Shape(_))
        ));
        assert!(matches!(
            MassMetric::atoms(array![1.0, f64::NEG_INFINITY].view(), 6),
            Err(SaddleError::NonFinite(_))
        ));
        assert!(matches!(MassMetric::unit(0), Err(SaddleError::Shape(_))));
        let metric = sample();
        assert!(matches!(
            metric.weight_point(MetricPoint::new(array![1.0, 2.0].view())),
            Err(SaddleError::Shape(_))
        ));
        assert!(matches!(
            metric.weight_covector(MetricCovector::new(array![1.0].view())),
            Err(SaddleError::Shape(_))
        ));
        assert!(matches!(metric.square_root(3), Err(SaddleError::Shape(_))));
    }

    #[test]
    fn inner_product_matches_the_weighted_vectors() {
        let metric = sample();
        let left = array![0.5, -1.5, 0.25];
        let right = array![1.0, 0.5, -0.5];
        let weighted_left = metric
            .weight_tangent(MetricTangent::new(left.view()))
            .unwrap();
        let weighted_right = metric
            .weight_tangent(MetricTangent::new(right.view()))
            .unwrap();
        let inner = metric
            .inner(
                MetricTangent::new(left.view()),
                MetricTangent::new(right.view()),
            )
            .unwrap();
        assert_eq!(inner, weighted_left.dot(&weighted_right));
        assert_eq!(inner, -3.625);
        let norm = metric.norm(MetricTangent::new(left.view())).unwrap();
        assert_eq!(norm, weighted_left.dot(&weighted_left).sqrt());
    }

    fn assert_shape(result: Result<Array1<f64>, SaddleError>) {
        assert!(matches!(result, Err(SaddleError::Shape(_))));
    }

    #[test]
    fn public_maps_reject_a_different_length() {
        let metric = sample();
        let short = array![1.0, 2.0];
        let matched = array![1.0, 4.0, 9.0];
        let long = array![1.0, 2.0, 3.0, 4.0];
        assert_shape(metric.weight_point(MetricPoint::new(long.view())));
        assert_shape(metric.unweight_point(MetricPoint::new(short.view())));
        assert_shape(metric.weight_tangent(MetricTangent::new(short.view())));
        assert_shape(metric.unweight_tangent(MetricTangent::new(long.view())));
        assert_shape(metric.weight_covector(MetricCovector::new(long.view())));
        assert_shape(metric.unweight_covector(MetricCovector::new(short.view())));
        assert!(matches!(
            metric.inner(
                MetricTangent::new(short.view()),
                MetricTangent::new(matched.view()),
            ),
            Err(SaddleError::Shape(_))
        ));
        assert!(matches!(
            metric.inner(
                MetricTangent::new(matched.view()),
                MetricTangent::new(long.view()),
            ),
            Err(SaddleError::Shape(_))
        ));
        assert!(matches!(
            metric.norm(MetricTangent::new(short.view())),
            Err(SaddleError::Shape(_))
        ));
    }

    #[test]
    fn unchecked_maps_match_the_public_result_on_equal_lengths() {
        let metric = sample();
        let values = array![0.5, -1.5, 0.25];
        assert_eq!(
            metric.multiply_sqrt(values.view()),
            metric
                .weight_point(MetricPoint::new(values.view()))
                .unwrap()
        );
        assert_eq!(
            metric.divide_sqrt(values.view()),
            metric
                .unweight_point(MetricPoint::new(values.view()))
                .unwrap()
        );
        let rows = array![[0.5, -1.5, 0.25], [1.0, -2.0, 0.5]];
        let weighted = metric.weight_point_rows(rows.view());
        let forced = metric.weight_force_rows(rows.view());
        for (index, row) in rows.outer_iter().enumerate() {
            let point = metric.weight_point(MetricPoint::new(row)).unwrap();
            assert_eq!(weighted.row(index), point);
            let covector = metric.weight_covector(MetricCovector::new(row)).unwrap();
            assert_eq!(forced.row(index), -covector);
        }
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "mass metric vector length must match the masses")]
    fn multiply_sqrt_rejects_a_short_vector() {
        let _ = sample().multiply_sqrt(array![1.0, 2.0].view());
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "mass metric vector length must match the masses")]
    fn divide_sqrt_rejects_a_long_vector() {
        let _ = sample().divide_sqrt(array![1.0, 2.0, 3.0, 4.0].view());
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "mass metric vector length must match the masses")]
    fn identity_map_rejects_a_different_length() {
        let metric = MassMetric::unit(3).unwrap();
        let _ = metric.multiply_sqrt(array![1.0].view());
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "mass metric row width must match the masses")]
    fn point_rows_reject_a_different_width() {
        let rows = array![[1.0, 2.0], [3.0, 4.0]];
        let _ = sample().weight_point_rows(rows.view());
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "mass metric row width must match the masses")]
    fn unweight_point_rows_reject_a_different_width() {
        let rows = array![[1.0, 2.0, 3.0, 4.0]];
        let _ = sample().unweight_point_rows(rows.view());
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "mass metric row width must match the masses")]
    fn force_rows_reject_a_different_width() {
        let rows = array![[1.0, 2.0]];
        let _ = sample().weight_force_rows(rows.view());
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "mass metric row width must match the masses")]
    fn covector_rows_reject_a_different_width() {
        let rows = array![[1.0, 2.0, 3.0, 4.0]];
        let _ = sample().unweight_covector_rows(rows.view());
    }
}
