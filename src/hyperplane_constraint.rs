//! A hyperplane whose normal is orthogonal to rigid directions.
//!
//! [`HyperplaneConstraint::orthogonal_to`] removes those directions by
//! modified Gram-Schmidt and drops a row only inside Higham's roundoff
//! bound. The estimator uses that same unit normal. The type lives
//! outside the `ohtst` feature so path sampling can hold a center of
//! mass when that feature is off.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

use crate::error::SaddleError;
use crate::projector::Hyperplane;

/// The plane `(x - origin) · normal = 0`, with `normal` a unit vector
/// orthogonal to every retained rigid direction.
///
/// The plane itself is a [`Hyperplane`]. Fields may be added. Build one
/// with [`HyperplaneConstraint::through`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct HyperplaneConstraint {
    plane: Hyperplane,
    /// Orthonormal rows. [`HyperplaneConstraint::project`] removes them before the
    /// normal, which is how a center-of-mass constraint is held.
    rigid: Array2<f64>,
}

impl HyperplaneConstraint {
    /// Unit vector in the direction of `normal` after its projection on
    /// the rows of `directions` is removed.
    ///
    /// `directions` has one row per forbidden direction and one column
    /// per coordinate. An empty matrix only normalizes `normal`. A row
    /// whose residual against the rows already kept is no larger than the
    /// roundoff of that product lies in their span and is dropped. The
    /// normal is then normalized. A residual within that same roundoff
    /// means the normal itself lies in the forbidden subspace.
    pub fn orthogonal_to(
        normal: ArrayView1<f64>,
        directions: ArrayView2<f64>,
    ) -> Result<Array1<f64>, SaddleError> {
        let basis = orthonormal_basis(directions, normal.len())?;
        project_and_normalize(normal, basis.view())
    }

    /// Unit normal and the derivative of that unit normal.
    ///
    /// `rate` is the derivative, with respect to the same parameter, of
    /// the vector passed as `normal`, taken before the projection. The
    /// returned rate is the derivative of the unit vector
    /// [`Self::orthogonal_to`] returns.
    pub fn orthogonal_rate(
        normal: ArrayView1<f64>,
        rate: ArrayView1<f64>,
        directions: ArrayView2<f64>,
    ) -> Result<(Array1<f64>, Array1<f64>), SaddleError> {
        if rate.len() != normal.len() {
            return Err(SaddleError::Shape(
                "hyperplane rate must match the normal".into(),
            ));
        }
        if !rate.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("hyperplane rate"));
        }
        let basis = orthonormal_basis(directions, normal.len())?;
        let mut projected = normal.to_owned();
        let mut projected_rate = rate.to_owned();
        for row in basis.rows() {
            let along = projected.dot(&row);
            projected.scaled_add(-along, &row);
            let along_rate = projected_rate.dot(&row);
            projected_rate.scaled_add(-along_rate, &row);
        }
        let length = norm(projected.view());
        let scale = norm(normal);
        if !(scale.is_finite() && scale > 0.0) {
            return Err(SaddleError::Invalid("hyperplane normal vanishes".into()));
        }
        let floor = roundoff_floor(normal.len(), basis.nrows(), scale)?;
        if !(length.is_finite() && length > floor) {
            return Err(SaddleError::Invalid(
                "hyperplane normal lies in the rigid subspace".into(),
            ));
        }
        let unit = &projected / length;
        // d(w/|w|) = w'/|w| - w (w · w') / |w|³, and w' is orthogonal to
        // the same constant directions w is.
        let radial = projected.dot(&projected_rate);
        let mut derivative = &projected_rate / length;
        derivative.scaled_add(-radial / length.powi(3), &projected);
        Ok((unit, derivative))
    }

    /// Orthonormal rows spanning `directions`.
    ///
    /// A row with no component outside the rows already kept is omitted.
    /// `width` is the normal's length, so an empty direction matrix still
    /// has the right number of columns.
    pub fn orthonormal_basis(
        directions: ArrayView2<f64>,
        width: usize,
    ) -> Result<Array2<f64>, SaddleError> {
        orthonormal_basis(directions, width)
    }

    /// Remove the rows of `basis` from `vector`.
    ///
    /// `basis` is the matrix [`Self::orthonormal_basis`] returns: one
    /// orthonormal row per direction.
    pub fn project_out(vector: &mut Array1<f64>, basis: ArrayView2<f64>) {
        for row in basis.rows() {
            let along = vector.dot(&row);
            vector.scaled_add(-along, &row);
        }
    }

    /// The plane through `origin` with the given normal, rigid directions
    /// removed from that normal.
    pub fn through(
        origin: Array1<f64>,
        normal: ArrayView1<f64>,
        directions: ArrayView2<f64>,
    ) -> Result<Self, SaddleError> {
        if origin.len() != normal.len() || !origin.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::Shape(
                "hyperplane origin must be finite and match the normal".into(),
            ));
        }
        let rigid = orthonormal_basis(directions, normal.len())?;
        let unit = project_and_normalize(normal, rigid.view())?;
        Ok(Self {
            plane: Hyperplane::new(origin, unit)?,
            rigid,
        })
    }

    /// The plane, without its rigid directions.
    pub fn plane(&self) -> &Hyperplane {
        &self.plane
    }

    /// Unit constraint normal.
    pub fn normal(&self) -> ArrayView1<'_, f64> {
        self.plane.normal()
    }

    /// Point the plane passes through.
    pub fn origin(&self) -> ArrayView1<'_, f64> {
        self.plane.point()
    }

    /// Orthonormal rigid directions, one per row.
    pub fn rigid(&self) -> ArrayView2<'_, f64> {
        self.rigid.view()
    }

    /// Remove the rigid directions and then the normal from `vector`.
    pub fn project(&self, vector: &mut Array1<f64>) {
        Self::project_out(vector, self.rigid.view());
        let along = vector.dot(&self.plane.normal());
        vector.scaled_add(-along, &self.plane.normal());
    }

    /// Move `x` along the normal until it lies on the plane.
    pub fn restore(&self, x: &mut Array1<f64>) {
        let slip = (&*x - &self.plane.point()).dot(&self.plane.normal());
        x.scaled_add(-slip, &self.plane.normal());
    }
}

/// Three uniform translations, one row per Cartesian axis, with a one on
/// that axis of every atom. The input of [`HyperplaneConstraint::orthogonal_to`]
/// when a host asks for a fixed center of mass and the surface does not
/// name its own rigid modes.
pub fn uniform_translations(n_atoms: usize) -> Array2<f64> {
    let mut modes = Array2::zeros((3, n_atoms * 3));
    for atom in 0..n_atoms {
        for axis in 0..3 {
            modes[(axis, 3 * atom + axis)] = 1.0;
        }
    }
    modes
}

fn orthonormal_basis(
    directions: ArrayView2<f64>,
    width: usize,
) -> Result<Array2<f64>, SaddleError> {
    if directions.nrows() > 0 && directions.ncols() != width {
        return Err(SaddleError::Shape(
            "hyperplane directions must match the normal".into(),
        ));
    }
    let mut kept: Vec<Array1<f64>> = Vec::new();
    for row in directions.axis_iter(Axis(0)) {
        if !row.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("hyperplane direction"));
        }
        let mut residual = row.to_owned();
        for earlier in &kept {
            let along = residual.dot(earlier);
            residual.scaled_add(-along, earlier);
        }
        let scale = norm(row);
        let length = norm(residual.view());
        let floor = roundoff_floor(width, kept.len(), scale)?;
        if length > floor {
            residual /= length;
            kept.push(residual);
        }
    }
    let mut basis = Array2::zeros((kept.len(), width));
    for (index, row) in kept.iter().enumerate() {
        basis.row_mut(index).assign(row);
    }
    Ok(basis)
}

fn project_and_normalize(
    normal: ArrayView1<f64>,
    basis: ArrayView2<f64>,
) -> Result<Array1<f64>, SaddleError> {
    if !normal.iter().all(|v| v.is_finite()) {
        return Err(SaddleError::NonFinite("hyperplane normal"));
    }
    let scale = norm(normal);
    if !(scale.is_finite() && scale > 0.0) {
        return Err(SaddleError::Invalid("hyperplane normal vanishes".into()));
    }
    let mut residual = normal.to_owned();
    HyperplaneConstraint::project_out(&mut residual, basis);
    let length = norm(residual.view());
    let floor = roundoff_floor(normal.len(), basis.nrows(), scale)?;
    if !(length.is_finite() && length > floor) {
        return Err(SaddleError::Invalid(
            "hyperplane normal lies in the rigid subspace".into(),
        ));
    }
    residual /= length;
    Ok(residual)
}

fn norm(vector: ArrayView1<f64>) -> f64 {
    vector.dot(&vector).sqrt()
}

/// Residual a vector that lies exactly in a span of `passes` directions
/// can carry after modified Gram-Schmidt.
///
/// One dot product and one axpy of length `d` accumulate at most
/// `γ_d = d u / (1 - d u)` times the vector's norm, with `u` machine
/// epsilon (Higham, Accuracy and Stability of Numerical Algorithms,
/// SIAM, 2nd ed., 2002, Eq. (3.8)). `passes` such projections, plus the
/// comparison itself, multiply that bound by `1 + passes`.
fn roundoff_floor(length: usize, passes: usize, scale: f64) -> Result<f64, SaddleError> {
    if !(scale.is_finite() && scale >= 0.0) {
        return Err(SaddleError::NonFinite("hyperplane scale"));
    }
    let unit = f64::EPSILON;
    let extent = length as f64 * unit;
    if extent >= 1.0 {
        return Err(SaddleError::Invalid(
            "hyperplane dimension is past the Gram-Schmidt roundoff bound".into(),
        ));
    }
    let gamma = extent / (1.0 - extent);
    Ok((1.0 + passes as f64) * gamma * scale)
}
