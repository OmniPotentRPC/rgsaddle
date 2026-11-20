//! Tangent projectors and affine constraint restoration.
//!
//! A projector removes forbidden components from vectors. A constraint
//! additionally restores a point to its level set. These two operations
//! are kept separate because band and minimum-mode methods only transform
//! forces, while PAFI also restores each sampled configuration.

use std::collections::BTreeMap;
use std::sync::Arc;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Zip, s};
use rgmin::Manifold;

use crate::SaddleError;
use crate::minmode::{FiniteDifference, MinModeConfig, MinModeSession, ModeEstimate, PointSurface};
use crate::region::Region;

/// Projection of an ambient vector onto an allowed tangent space.
pub trait TangentProjector {
    /// Return the allowed component of `vector`.
    fn project(&self, vector: ArrayView1<'_, f64>) -> Array1<f64>;

    /// Replace `vector` by its allowed component.
    fn project_in_place(&self, vector: &mut Array1<f64>);
}

/// A tangent projector with a level set onto which points can be restored.
pub trait ProjectedConstraint: TangentProjector {
    /// Restore `point` to the constraint level set.
    fn restore_in_place(&self, point: &mut Array1<f64>);
}

/// Coordinates in which a backend receives a hyperplane.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HyperplaneCoordinates {
    /// `q = M^(1/2) x`; the normal and velocities use the Euclidean metric.
    #[default]
    MassWeighted,
    /// Cartesian coordinates. The backend must apply its configured masses.
    Cartesian,
}

/// Backend-independent constrained-ensemble request.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct HyperplaneSampleBudget {
    /// `kT` in the engine's energy units.
    pub thermal: f64,
    /// Seed of all stochastic engine operations.
    pub seed: u64,
    /// Optional minimum warm-up floor before estimator-side equilibration.
    pub thermalize: usize,
    /// Engine steps between records.
    pub stride: usize,
    /// Number of records.
    pub samples: usize,
    /// Coordinate convention of the plane and returned vectors.
    pub coordinates: HyperplaneCoordinates,
    /// Ask the engine to return positions at every record.
    pub positions: bool,
    /// Coupling parameter for a thermodynamic-interpolation request.
    pub lambda: Option<f64>,
    /// Named scalar observables requested at every record.
    pub observables: Vec<String>,
    /// Hard symmetry-unique region and any additional predicates.
    pub region: Option<Arc<dyn Region>>,
}

impl Default for HyperplaneSampleBudget {
    fn default() -> Self {
        Self {
            thermal: 0.02585,
            seed: 1,
            thermalize: 500,
            stride: 5,
            samples: 2_000,
            coordinates: HyperplaneCoordinates::MassWeighted,
            positions: true,
            lambda: None,
            observables: Vec::new(),
            region: None,
        }
    }
}

impl HyperplaneSampleBudget {
    /// Set [`Self::thermal`].
    pub fn with_thermal(mut self, thermal: f64) -> Self {
        self.thermal = thermal;
        self
    }
    /// Set [`Self::seed`].
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
    /// Set [`Self::thermalize`].
    pub fn with_thermalize(mut self, thermalize: usize) -> Self {
        self.thermalize = thermalize;
        self
    }
    /// Set [`Self::stride`].
    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }
    /// Set [`Self::samples`].
    pub fn with_samples(mut self, samples: usize) -> Self {
        self.samples = samples;
        self
    }
    /// Set [`Self::coordinates`].
    pub fn with_coordinates(mut self, coordinates: HyperplaneCoordinates) -> Self {
        self.coordinates = coordinates;
        self
    }
    /// Set [`Self::positions`].
    pub fn with_positions(mut self, positions: bool) -> Self {
        self.positions = positions;
        self
    }
    /// Set the thermodynamic coupling parameter.
    pub fn with_lambda(mut self, lambda: f64) -> Self {
        self.lambda = Some(lambda);
        self
    }
    /// Request one named scalar observable.
    pub fn with_observable(mut self, observable: impl Into<String>) -> Self {
        self.observables.push(observable.into());
        self
    }

    /// Restrict sampling to a hard region.
    pub fn with_region(mut self, region: Arc<dyn Region>) -> Self {
        self.region = Some(region);
        self
    }
}

/// One engine record on a constrained hyperplane.
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct HyperplaneObservation {
    /// Mechanical constraint force along the plane's positive unit normal,
    /// `-normal · gradient`.
    pub constraint_force: f64,
    /// Complete PAFI plane-gradient record, when the backend computes it.
    ///
    /// When absent, PAFI reconstructs the value from the constraint force and
    /// geometry.
    pub pafi_gradient: Option<f64>,
    /// PAFI metric/Jacobian correction `psi`; zero for a fixed flat plane.
    pub jacobian: f64,
    /// Position when requested.
    pub position: Option<Array1<f64>>,
    /// Force after removing constrained directions, when exposed by the engine.
    pub projected_force: Option<Array1<f64>>,
    /// Velocity component removed by the constraint integrator, when exposed.
    pub projected_velocity: Option<Array1<f64>>,
    /// Estimator-selected scalar observables, keyed by a stable public name.
    ///
    /// Thermodynamic integration uses this for quantities such as
    /// `potential.full_minus_harmonic` without exposing an engine type.
    pub observables: BTreeMap<String, f64>,
}

impl HyperplaneObservation {
    /// Set [`Self::constraint_force`].
    pub fn with_constraint_force(mut self, constraint_force: f64) -> Self {
        self.constraint_force = constraint_force;
        self
    }

    /// Set [`Self::pafi_gradient`].
    pub fn with_pafi_gradient(mut self, pafi_gradient: f64) -> Self {
        self.pafi_gradient = Some(pafi_gradient);
        self
    }

    /// Set [`Self::jacobian`].
    pub fn with_jacobian(mut self, jacobian: f64) -> Self {
        self.jacobian = jacobian;
        self
    }

    /// Set [`Self::position`].
    pub fn with_position(mut self, position: Array1<f64>) -> Self {
        self.position = Some(position);
        self
    }

    /// Set [`Self::projected_force`].
    pub fn with_projected_force(mut self, projected_force: Array1<f64>) -> Self {
        self.projected_force = Some(projected_force);
        self
    }

    /// Set [`Self::projected_velocity`].
    pub fn with_projected_velocity(mut self, projected_velocity: Array1<f64>) -> Self {
        self.projected_velocity = Some(projected_velocity);
        self
    }

    /// Insert a named scalar observable.
    pub fn with_observable(mut self, name: impl Into<String>, value: f64) -> Self {
        self.observables.insert(name.into(), value);
        self
    }
}

/// Records and engine accounting returned by [`HyperplaneSampler`].
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct HyperplaneEnsemble {
    /// Records in engine order.
    pub observations: Vec<HyperplaneObservation>,
    /// Force/energy evaluations reported by the engine.
    pub evaluations: usize,
    /// Accepted fraction for a Metropolized engine, otherwise one.
    pub acceptance: f64,
}

/// External or analytic engine that samples a constrained ensemble.
///
/// Implementations own integration, thermostats, masses, units and engine
/// handles. No engine-specific type crosses this interface.
pub trait HyperplaneSampler {
    /// Generate records on `plane`.
    fn sample_hyperplane(
        &mut self,
        plane: &Hyperplane,
        budget: &HyperplaneSampleBudget,
    ) -> Result<HyperplaneEnsemble, SaddleError>;

    /// Potential energy at an arbitrary configuration, when the backend
    /// supports mapped-configuration BAR or MBAR.
    fn energy(&mut self, _position: ArrayView1<f64>) -> Result<Option<f64>, SaddleError> {
        Ok(None)
    }

    /// Orthonormal directions held fixed in addition to the plane normal.
    fn constrained_directions(
        &mut self,
        _plane: &Hyperplane,
    ) -> Result<Vec<Array1<f64>>, SaddleError> {
        Ok(Vec::new())
    }
}

/// A point-dependent [`rgmin::Manifold`] projector bound at one point.
///
/// This adapter puts Sella internal-coordinate constraints and rgmin's
/// sphere, Grassmann and quotient manifolds behind [`TangentProjector`]
/// without changing their retractions.
#[derive(Clone, Copy, Debug)]
pub struct ManifoldTangent<'a, M: ?Sized> {
    manifold: &'a M,
    point: &'a Array1<f64>,
}

impl<'a, M: Manifold + ?Sized> ManifoldTangent<'a, M> {
    /// Bind `manifold` at `point`.
    pub fn new(manifold: &'a M, point: &'a Array1<f64>) -> Self {
        Self { manifold, point }
    }
}

impl<M: Manifold + ?Sized> TangentProjector for ManifoldTangent<'_, M> {
    fn project(&self, vector: ArrayView1<'_, f64>) -> Array1<f64> {
        self.manifold.project(self.point, &vector.to_owned())
    }

    fn project_in_place(&self, vector: &mut Array1<f64>) {
        *vector = self.manifold.project(self.point, vector);
    }
}

/// The plane `{q : normal · (q - point) = 0}` in mass-weighted coordinates.
///
/// [`Hyperplane::new`] stores a unit normal for OH-TST. [`Hyperplane::from_path`]
/// additionally retains the path derivative used to apply the algebraically
/// equivalent `nnᵀ/(nᵀn)` projector in PAFI's historical operation order.
#[derive(Clone, Debug, PartialEq)]
pub struct Hyperplane {
    point: Array1<f64>,
    normal: Array1<f64>,
    projection_normal: Option<Array1<f64>>,
}

/// Lowest eigenpairs of a Hessian restricted to a hyperplane.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct HyperplaneModes {
    /// Orthonormal eigenvectors, one column per mode.
    pub modes: Array2<f64>,
    /// Ritz values in ascending order.
    pub curvatures: Array1<f64>,
    /// Restricted Hessian actions performed by block Lanczos.
    pub actions: usize,
    /// Surface energy/gradient evaluations used by finite differences.
    pub evaluations: usize,
}

impl Hyperplane {
    /// The plane through `point` normal to `normal`; the stored normal is
    /// normalized.
    pub fn new(point: Array1<f64>, normal: Array1<f64>) -> Result<Self, SaddleError> {
        check_plane(&point, &normal)?;
        let norm = normal.dot(&normal).sqrt();
        Ok(Self {
            point,
            normal: normal / norm,
            projection_normal: None,
        })
    }

    /// A path plane retaining the unnormalized derivative for bitwise-stable
    /// projection and restoration.
    pub fn from_path(
        point: ArrayView1<'_, f64>,
        derivative: ArrayView1<'_, f64>,
    ) -> Result<Self, SaddleError> {
        let raw = derivative.to_owned();
        let mut plane = Self::new(point.to_owned(), raw.clone())?;
        plane.projection_normal = Some(raw);
        Ok(plane)
    }

    /// The plane through a minimum-mode session's position and mode.
    pub fn from_min_mode(session: &MinModeSession) -> Result<Self, SaddleError> {
        Self::new(session.position().to_owned(), session.mode().to_owned())
    }

    /// The plane through `position` normal to an estimated lowest mode.
    pub fn from_mode(position: ArrayView1<f64>, mode: &ModeEstimate) -> Result<Self, SaddleError> {
        Self::new(position.to_owned(), mode.mode.clone())
    }

    /// Map a Cartesian plane to mass-weighted coordinates.
    pub fn from_cartesian(
        position: ArrayView1<f64>,
        normal: ArrayView1<f64>,
        masses: &[f64],
    ) -> Result<Self, SaddleError> {
        let sqrt_mass = sqrt_masses(masses, position.len())?;
        if normal.len() != position.len() {
            return Err(SaddleError::Shape(
                "Cartesian plane point and normal need one length".into(),
            ));
        }
        Self::new(&position * &sqrt_mass, &normal / &sqrt_mass)
    }

    /// Map this mass-weighted plane back to Cartesian coordinates.
    pub fn to_cartesian(&self, masses: &[f64]) -> Result<(Array1<f64>, Array1<f64>), SaddleError> {
        let sqrt_mass = sqrt_masses(masses, self.dim())?;
        let normal = &self.normal * &sqrt_mass;
        let norm = normal.dot(&normal).sqrt();
        Ok((&self.point / &sqrt_mass, normal / norm))
    }

    /// A point on the plane.
    pub fn point(&self) -> ArrayView1<'_, f64> {
        self.point.view()
    }

    /// Unit normal pointing from the reactant side to the product side.
    pub fn normal(&self) -> ArrayView1<'_, f64> {
        self.normal.view()
    }

    /// Configuration-space dimension.
    pub fn dim(&self) -> usize {
        self.point.len()
    }

    /// Signed mass-weighted distance from `q` to the plane.
    pub fn distance(&self, q: ArrayView1<f64>) -> f64 {
        let mut distance = 0.0;
        for i in 0..self.dim() {
            distance += self.normal[i] * (q[i] - self.point[i]);
        }
        distance
    }

    /// Orthogonally restore `q` to the plane.
    pub fn project(&self, q: ArrayView1<f64>) -> Array1<f64> {
        let mut out = q.to_owned();
        self.restore_in_place(&mut out);
        out
    }

    /// The same plane represented by the projection of `center`.
    pub fn recentered(&self, center: ArrayView1<f64>) -> Self {
        Self {
            point: self.project(center),
            normal: self.normal.clone(),
            projection_normal: self.projection_normal.clone(),
        }
    }

    /// Reverse the normal when `toward` is on the negative side.
    pub fn oriented_toward(mut self, toward: ArrayView1<f64>) -> Self {
        if self.distance(toward) < 0.0 {
            self.normal.mapv_inplace(|value| -value);
            if let Some(raw) = &mut self.projection_normal {
                raw.mapv_inplace(|value| -value);
            }
        }
        self
    }

    /// Make the normal orthogonal to the rows of `modes`.
    pub fn orthogonal_to(&self, modes: ArrayView2<f64>) -> Result<Self, SaddleError> {
        if modes.nrows() > 0 && modes.ncols() != self.dim() {
            return Err(SaddleError::Shape(
                "modes differ in length from the plane".into(),
            ));
        }
        let mut normal = self.normal.clone();
        for direction in orthonormal_rows(modes, self.dim()) {
            let coefficient = direction.dot(&normal);
            normal.scaled_add(-coefficient, &direction);
        }
        Self::new(self.point.clone(), normal)
    }

    /// Pack `[point; normal]`.
    pub fn packed(&self) -> Array1<f64> {
        let n = self.dim();
        let mut packed = Array1::zeros(2 * n);
        packed.slice_mut(s![..n]).assign(&self.point);
        packed.slice_mut(s![n..]).assign(&self.normal);
        packed
    }

    /// Unpack `[point; normal]`, normalizing the normal block.
    pub fn from_packed(packed: ArrayView1<f64>) -> Result<Self, SaddleError> {
        if packed.is_empty() || !packed.len().is_multiple_of(2) {
            return Err(SaddleError::Shape(
                "a packed hyperplane has an even nonzero length".into(),
            ));
        }
        let n = packed.len() / 2;
        Self::new(
            packed.slice(s![..n]).to_owned(),
            packed.slice(s![n..]).to_owned(),
        )
    }

    /// Apply the Hessian on the tangent space of this plane and any
    /// additional constrained rows.
    ///
    /// The input is projected before the action and the result afterwards.
    /// [`PointSurface::hessian_vector`] is used when available; otherwise
    /// the finite difference selected by `config` is used.
    pub fn hessian_vector<S: PointSurface>(
        &self,
        surface: &S,
        vector: ArrayView1<f64>,
        constrained_rows: ArrayView2<f64>,
        config: &MinModeConfig,
    ) -> Result<Array1<f64>, SaddleError> {
        if vector.len() != self.dim() {
            return Err(SaddleError::Shape(
                "Hessian vector differs in length from the plane".into(),
            ));
        }
        let basis = self.constraint_basis(constrained_rows)?;
        let vector = project_rows(vector, &basis);
        let (_, gradient) = surface.eval(self.point())?;
        let mut evaluations = 1;
        restricted_hessian_action(
            surface,
            self.point(),
            gradient.view(),
            vector.view(),
            &basis,
            config,
            &mut evaluations,
        )
    }

    /// Lowest `count` Hessian eigenpairs restricted to this plane and any
    /// additional constrained rows.
    ///
    /// This calls the same reorthogonalized block-Lanczos workspace used by
    /// [`crate::IndexKSession`]. The returned vectors are ambient columns
    /// orthogonal to the plane normal and every independent constrained row.
    pub fn lowest_modes<S: PointSurface>(
        &self,
        surface: &S,
        constrained_rows: ArrayView2<f64>,
        count: usize,
        config: &MinModeConfig,
    ) -> Result<HyperplaneModes, SaddleError> {
        let basis = self.constraint_basis(constrained_rows)?;
        let free = self.dim().saturating_sub(basis.len());
        if count == 0 || count > free {
            return Err(SaddleError::Invalid(format!(
                "requested {count} hyperplane modes from {free} free directions"
            )));
        }
        let mut candidates = Array2::zeros((self.dim(), self.dim()));
        for column in 0..self.dim() {
            let mut axis = Array1::zeros(self.dim());
            axis[column] = 1.0;
            candidates
                .column_mut(column)
                .assign(&project_rows(axis.view(), &basis));
        }
        let seed = crate::linalg::modified_gram_schmidt(candidates.view(), None, 1.0e-12);
        if seed.ncols() < count {
            return Err(SaddleError::Invalid(
                "hyperplane constraints leave too few independent directions".into(),
            ));
        }
        let seed = seed.slice(s![.., ..count]).to_owned();
        let (_, gradient) = surface.eval(self.point())?;
        let mut evaluations = 1;
        let mut apply = |vector: ArrayView1<f64>| {
            restricted_hessian_action(
                surface,
                self.point(),
                gradient.view(),
                vector,
                &basis,
                config,
                &mut evaluations,
            )
        };
        let (modes, curvatures, actions) = crate::indexk::block_lanczos(
            seed.view(),
            config.rotation_tol,
            free.div_ceil(count).max(2),
            &mut apply,
        )?;
        Ok(HyperplaneModes {
            modes,
            curvatures,
            actions,
            evaluations,
        })
    }

    fn constraint_basis(
        &self,
        constrained_rows: ArrayView2<f64>,
    ) -> Result<Vec<Array1<f64>>, SaddleError> {
        if constrained_rows.nrows() > 0 && constrained_rows.ncols() != self.dim() {
            return Err(SaddleError::Shape(
                "constrained rows differ in length from the plane".into(),
            ));
        }
        if constrained_rows.iter().any(|value| !value.is_finite()) {
            return Err(SaddleError::NonFinite("hyperplane constrained rows"));
        }
        let mut rows = Array2::zeros((constrained_rows.nrows() + 1, self.dim()));
        rows.row_mut(0).assign(&self.normal);
        if constrained_rows.nrows() > 0 {
            rows.slice_mut(s![1.., ..]).assign(&constrained_rows);
        }
        Ok(orthonormal_rows(rows.view(), self.dim()))
    }

    fn projection(&self) -> (ArrayView1<'_, f64>, f64) {
        match &self.projection_normal {
            Some(raw) => (raw.view(), raw.dot(raw)),
            None => (self.normal.view(), 1.0),
        }
    }
}

fn project_rows(vector: ArrayView1<f64>, rows: &[Array1<f64>]) -> Array1<f64> {
    let mut out = vector.to_owned();
    for row in rows {
        let coefficient = row.dot(&out);
        out.scaled_add(-coefficient, row);
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn restricted_hessian_action<S: PointSurface>(
    surface: &S,
    point: ArrayView1<f64>,
    gradient: ArrayView1<f64>,
    vector: ArrayView1<f64>,
    constraints: &[Array1<f64>],
    config: &MinModeConfig,
    evaluations: &mut usize,
) -> Result<Array1<f64>, SaddleError> {
    let vector = project_rows(vector, constraints);
    let action = match surface.hessian_vector(point, vector.view())? {
        Some(action) => action,
        None => {
            let mut plus = point.to_owned();
            plus.scaled_add(config.dr, &vector);
            let (_, forward) = surface.eval(plus.view())?;
            *evaluations += 1;
            match config.difference {
                FiniteDifference::Forward => (&forward - &gradient) / config.dr,
                FiniteDifference::Central => {
                    let mut minus = point.to_owned();
                    minus.scaled_add(-config.dr, &vector);
                    let (_, backward) = surface.eval(minus.view())?;
                    *evaluations += 1;
                    (forward - backward) / (2.0 * config.dr)
                }
            }
        }
    };
    if action.len() != point.len() {
        return Err(SaddleError::Shape(
            "surface Hessian action differs in length from the plane".into(),
        ));
    }
    if action.iter().any(|value| !value.is_finite()) {
        return Err(SaddleError::NonFinite(
            "restricted hyperplane Hessian action",
        ));
    }
    Ok(project_rows(action.view(), constraints))
}

impl TangentProjector for Hyperplane {
    fn project(&self, vector: ArrayView1<'_, f64>) -> Array1<f64> {
        let mut out = vector.to_owned();
        self.project_in_place(&mut out);
        out
    }

    fn project_in_place(&self, vector: &mut Array1<f64>) {
        let (normal, denominator) = self.projection();
        let coefficient = vector.dot(&normal) / denominator;
        *vector -= &(normal.to_owned() * coefficient);
    }
}

impl ProjectedConstraint for Hyperplane {
    fn restore_in_place(&self, point: &mut Array1<f64>) {
        let (normal, denominator) = self.projection();
        let correction = (&*point - &self.point).dot(&normal) / denominator;
        *point -= &(normal.to_owned() * correction);
    }
}

fn check_plane(point: &Array1<f64>, normal: &Array1<f64>) -> Result<(), SaddleError> {
    if point.is_empty() || point.len() != normal.len() {
        return Err(SaddleError::Shape(
            "hyperplane point and normal need one nonzero length".into(),
        ));
    }
    if !point.iter().chain(normal).all(|value| value.is_finite()) {
        return Err(SaddleError::NonFinite("hyperplane"));
    }
    let norm = normal.dot(normal).sqrt();
    if norm <= 0.0 {
        return Err(SaddleError::Invalid("hyperplane normal vanishes".into()));
    }
    Ok(())
}

pub(crate) fn orthonormal_rows(rows: ArrayView2<f64>, n: usize) -> Vec<Array1<f64>> {
    let mut basis: Vec<Array1<f64>> = Vec::with_capacity(rows.nrows());
    for row in rows.rows() {
        if row.len() != n {
            continue;
        }
        let scale = row.dot(&row).sqrt();
        let mut vector = row.to_owned();
        for _ in 0..2 {
            for direction in &basis {
                let coefficient = direction.dot(&vector);
                vector.scaled_add(-coefficient, direction);
            }
        }
        let norm = vector.dot(&vector).sqrt();
        if scale > 0.0 && norm > 1.0e-10 * scale {
            basis.push(vector / norm);
        }
    }
    basis
}

pub(crate) fn sqrt_masses(masses: &[f64], n: usize) -> Result<Array1<f64>, SaddleError> {
    if masses.len() != n {
        return Err(SaddleError::Shape("one mass per coordinate".into()));
    }
    if !masses.iter().all(|mass| mass.is_finite() && *mass > 0.0) {
        return Err(SaddleError::Invalid(
            "masses must be finite and positive".into(),
        ));
    }
    Ok(masses.iter().map(|mass| mass.sqrt()).collect())
}

/// Orthogonal complement of an axis known by its caller to have unit norm.
///
/// This adapter deliberately does not renormalize or divide by
/// `axis · axis`: band, dimer and Lanczos paths already supply unit axes,
/// and their historical arithmetic is `v - (v · axis) axis`.
#[derive(Clone, Copy, Debug)]
pub struct UnitNormal<'a> {
    axis: ArrayView1<'a, f64>,
}

impl<'a> UnitNormal<'a> {
    /// Bind the caller-validated unit `axis`.
    pub fn new(axis: ArrayView1<'a, f64>) -> Self {
        Self { axis }
    }
}

impl TangentProjector for UnitNormal<'_> {
    fn project(&self, vector: ArrayView1<'_, f64>) -> Array1<f64> {
        let parallel = vector.dot(&self.axis);
        let mut out = Array1::zeros(vector.len());
        Zip::from(out.view_mut())
            .and(vector)
            .and(self.axis)
            .for_each(|slot, &value, &axis| *slot = value - axis * parallel);
        out
    }

    fn project_in_place(&self, vector: &mut Array1<f64>) {
        let parallel = vector.dot(&self.axis);
        Zip::from(vector)
            .and(self.axis)
            .for_each(|value, &axis| *value -= axis * parallel);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    struct DiagonalHessian;

    impl PointSurface for DiagonalHessian {
        fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let diagonal = array![1.0, 2.0, 4.0, 8.0];
            let gradient = &diagonal * &q;
            Ok((0.5 * q.dot(&gradient), gradient))
        }

        fn hessian_vector(
            &self,
            _q: ArrayView1<f64>,
            vector: ArrayView1<f64>,
        ) -> Result<Option<Array1<f64>>, SaddleError> {
            Ok(Some(&array![1.0, 2.0, 4.0, 8.0] * &vector))
        }
    }

    #[test]
    fn projection_and_restore_satisfy_the_plane() {
        let point = array![0.4, -0.2, 0.7];
        let normal = array![2.0, -1.0, 0.5];
        let plane = Hyperplane::from_path(point.view(), normal.view()).unwrap();
        let projected = TangentProjector::project(&plane, array![1.0, 3.0, -2.0].view());
        assert!(projected.dot(&normal).abs() < 1e-14);
        let mut position = array![-0.3, 1.2, 0.1];
        plane.restore_in_place(&mut position);
        assert!((&position - &point).dot(&normal).abs() < 1e-14);
    }

    #[test]
    fn restricted_lanczos_matches_the_projected_hessian() {
        let plane =
            Hyperplane::new(array![0.0, 0.0, 0.0, 0.0], array![0.0, 0.0, 0.0, 1.0]).unwrap();
        let rows = array![[0.0, 0.0, 1.0, 0.0]];
        let config = MinModeConfig::default().with_rotation_tol(1.0e-12);
        let modes = plane
            .lowest_modes(&DiagonalHessian, rows.view(), 2, &config)
            .unwrap();
        assert_eq!(modes.curvatures.len(), 2);
        assert!((modes.curvatures[0] - 1.0).abs() < 1.0e-12);
        assert!((modes.curvatures[1] - 2.0).abs() < 1.0e-12);
        assert!(modes.modes.column(0)[0].abs() > 1.0 - 1.0e-12);
        assert!(modes.modes.column(1)[1].abs() > 1.0 - 1.0e-12);
        for column in modes.modes.columns() {
            assert!(column.dot(&plane.normal()).abs() < 1.0e-12);
            assert!(column.dot(&rows.row(0)).abs() < 1.0e-12);
        }
        let action = plane
            .hessian_vector(
                &DiagonalHessian,
                array![3.0, -2.0, 7.0, 9.0].view(),
                rows.view(),
                &config,
            )
            .unwrap();
        assert_eq!(action, array![3.0, -4.0, 0.0, 0.0]);
    }
}
