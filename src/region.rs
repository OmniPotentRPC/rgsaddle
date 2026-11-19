//! Hard, symmetry-aware regions for constrained sampling.
//!
//! A [`Region`] is an unbiased geometric restriction: molecular dynamics
//! reflects at its faces and Monte Carlo rejects a proposal outside it.
//! [`SymmetryUniqueRegion::from_source`] is the Voronoi cell of one
//! representative among symmetry images that preserve a hyperplane.
//! [`SymmetryUniqueRegion::from_crystal_operations`] builds the asymmetric
//! unit of one hop's dividing plane. Its degeneracy is the order of the
//! joint stabilizer of that hop, not the number of equivalent hops.

use std::fmt::Debug;
use std::sync::Arc;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::{Hyperplane, SaddleError};

/// The face of a hard region that a configuration crossed.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct BoundaryViolation {
    /// Outward ambient unit normal at the violated face.
    pub normal: Array1<f64>,
    /// Signed distance past a planar face. Predicate-only boundaries may
    /// omit it, in which case MD returns to the previous contained point.
    pub penetration: Option<f64>,
}

impl BoundaryViolation {
    /// A locally planar violation.
    pub fn planar(normal: Array1<f64>, penetration: f64) -> Result<Self, SaddleError> {
        if !(penetration.is_finite() && penetration > 0.0) {
            return Err(SaddleError::Invalid(
                "boundary penetration must be finite and positive".into(),
            ));
        }
        Ok(Self {
            normal: unit(normal)?,
            penetration: Some(penetration),
        })
    }

    /// A predicate violation with no penetration depth.
    pub fn predicate(normal: Array1<f64>) -> Result<Self, SaddleError> {
        Ok(Self {
            normal: unit(normal)?,
            penetration: None,
        })
    }
}

/// A hard configuration-space region.
pub trait Region: Debug + Send + Sync {
    /// Whether `point` lies in the closed region.
    fn contains(&self, point: ArrayView1<'_, f64>) -> Result<bool, SaddleError> {
        Ok(self.violation(point)?.is_none())
    }

    /// The violated face and its outward normal, or `None` inside.
    fn violation(
        &self,
        point: ArrayView1<'_, f64>,
    ) -> Result<Option<BoundaryViolation>, SaddleError>;

    /// Number of equivalent in-plane images represented by this region.
    fn degeneracy(&self) -> usize {
        1
    }

    /// Exact planar faces, when an external engine can consume them.
    fn planar_boundaries(&self) -> Option<Vec<PlanarBoundary>> {
        None
    }
}

/// The half-space `normal · (q - point) <= 0`.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanarBoundary {
    point: Array1<f64>,
    normal: Array1<f64>,
}

impl PlanarBoundary {
    /// Construct a half-space from a face point and outward normal.
    pub fn new(point: Array1<f64>, normal: Array1<f64>) -> Result<Self, SaddleError> {
        if point.is_empty() || point.len() != normal.len() {
            return Err(SaddleError::Shape(
                "boundary point and normal need one nonzero length".into(),
            ));
        }
        if !point.iter().all(|value| value.is_finite()) {
            return Err(SaddleError::NonFinite("boundary point"));
        }
        Ok(Self {
            point,
            normal: unit(normal)?,
        })
    }

    /// Construct the orientation whose closed interior contains `inside`.
    pub fn containing(
        point: Array1<f64>,
        normal: Array1<f64>,
        inside: ArrayView1<'_, f64>,
    ) -> Result<Self, SaddleError> {
        let mut boundary = Self::new(point, normal)?;
        if boundary.signed_distance(inside)? > 0.0 {
            boundary.normal.mapv_inplace(|value| -value);
        }
        Ok(boundary)
    }

    /// Point on the face.
    pub fn point(&self) -> ArrayView1<'_, f64> {
        self.point.view()
    }

    /// Outward unit normal.
    pub fn normal(&self) -> ArrayView1<'_, f64> {
        self.normal.view()
    }

    /// Signed distance, positive outside.
    pub fn signed_distance(&self, point: ArrayView1<'_, f64>) -> Result<f64, SaddleError> {
        if point.len() != self.point.len() {
            return Err(SaddleError::Shape(
                "boundary and configuration dimensions differ".into(),
            ));
        }
        Ok((&point - &self.point).dot(&self.normal))
    }
}

impl Region for PlanarBoundary {
    fn violation(
        &self,
        point: ArrayView1<'_, f64>,
    ) -> Result<Option<BoundaryViolation>, SaddleError> {
        let penetration = self.signed_distance(point)?;
        if penetration > 0.0 {
            Ok(Some(BoundaryViolation::planar(
                self.normal.clone(),
                penetration,
            )?))
        } else {
            Ok(None)
        }
    }

    fn planar_boundaries(&self) -> Option<Vec<PlanarBoundary>> {
        Some(vec![self.clone()])
    }
}

/// Intersection of a primary symmetry region and optional unit-degeneracy
/// predicates such as a path slab or basin exclusion.
#[derive(Clone, Debug)]
pub struct RegionIntersection {
    regions: Vec<Arc<dyn Region>>,
    degeneracy: usize,
}

impl Default for RegionIntersection {
    fn default() -> Self {
        Self {
            regions: Vec::new(),
            degeneracy: 1,
        }
    }
}

impl RegionIntersection {
    /// Validate an intersection. More than one non-unit degeneracy is
    /// rejected because multiplying overlapping symmetry counts would
    /// double count configurations.
    pub fn new(regions: Vec<Arc<dyn Region>>) -> Result<Self, SaddleError> {
        let non_unit: Vec<usize> = regions
            .iter()
            .map(|region| region.degeneracy())
            .filter(|degeneracy| *degeneracy != 1)
            .collect();
        if non_unit.len() > 1 {
            return Err(SaddleError::Invalid(
                "an intersection may carry only one symmetry degeneracy".into(),
            ));
        }
        if regions.iter().any(|region| region.degeneracy() == 0) {
            return Err(SaddleError::Invalid(
                "region degeneracy must be positive".into(),
            ));
        }
        let degeneracy = match non_unit.first() {
            Some(degeneracy) => *degeneracy,
            None => 1,
        };
        Ok(Self {
            degeneracy,
            regions,
        })
    }

    /// Components in test order.
    pub fn regions(&self) -> &[Arc<dyn Region>] {
        &self.regions
    }
}

impl Region for RegionIntersection {
    fn violation(
        &self,
        point: ArrayView1<'_, f64>,
    ) -> Result<Option<BoundaryViolation>, SaddleError> {
        for region in &self.regions {
            if let Some(violation) = region.violation(point)? {
                return Ok(Some(violation));
            }
        }
        Ok(None)
    }

    fn degeneracy(&self) -> usize {
        self.degeneracy
    }

    fn planar_boundaries(&self) -> Option<Vec<PlanarBoundary>> {
        let mut boundaries = Vec::new();
        for region in &self.regions {
            boundaries.extend(region.planar_boundaries()?);
        }
        Some(boundaries)
    }
}

/// One affine point- or space-group operation `q -> linear q + translation`.
///
/// Non-exhaustive: start from [`AffineSymmetry::default`] and use setters,
/// or use [`AffineSymmetry::new`].
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct AffineSymmetry {
    /// Linear coordinate map.
    pub linear: Array2<f64>,
    /// Nonsymmorphic or lattice translation.
    pub translation: Array1<f64>,
}

impl AffineSymmetry {
    /// Construct one affine operation.
    pub fn new(linear: Array2<f64>, translation: Array1<f64>) -> Self {
        Self {
            linear,
            translation,
        }
    }

    /// Set [`Self::linear`].
    #[must_use]
    pub fn with_linear(mut self, linear: Array2<f64>) -> Self {
        self.linear = linear;
        self
    }

    /// Set [`Self::translation`].
    #[must_use]
    pub fn with_translation(mut self, translation: Array1<f64>) -> Self {
        self.translation = translation;
        self
    }

    fn apply(&self, point: ArrayView1<'_, f64>) -> Result<Array1<f64>, SaddleError> {
        if self.linear.nrows() != point.len()
            || self.linear.ncols() != point.len()
            || self.translation.len() != point.len()
        {
            return Err(SaddleError::Shape(
                "symmetry operation and configuration dimensions differ".into(),
            ));
        }
        if self
            .linear
            .iter()
            .chain(self.translation.iter())
            .any(|value| !value.is_finite())
        {
            return Err(SaddleError::NonFinite("symmetry operation"));
        }
        Ok(self.linear.dot(&point) + &self.translation)
    }
}

/// Generic source of point- or space-group operations.
pub trait SymmetrySource {
    /// Operations to test for preservation of the sampled plane.
    fn operations(&self) -> &[AffineSymmetry];

    /// Displacement from `reference` to an image in the source's metric.
    fn displacement(
        &self,
        reference: ArrayView1<'_, f64>,
        image: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, SaddleError> {
        if reference.len() != image.len() {
            return Err(SaddleError::Shape(
                "symmetry image and representative dimensions differ".into(),
            ));
        }
        Ok(&image - &reference)
    }
}

/// Caller-supplied symmetry operations.
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct SymmetryOperations {
    /// Point- or space-group operations.
    pub operations: Vec<AffineSymmetry>,
}

impl SymmetryOperations {
    /// Set [`Self::operations`].
    #[must_use]
    pub fn with_operations(mut self, operations: Vec<AffineSymmetry>) -> Self {
        self.operations = operations;
        self
    }
}

impl SymmetrySource for SymmetryOperations {
    fn operations(&self) -> &[AffineSymmetry] {
        &self.operations
    }
}

/// Symmetry operations whose atomic displacements use a periodic
/// [`minimage::Cell`].
#[derive(Clone, Debug)]
pub struct PeriodicSymmetryOperations {
    source: SymmetryOperations,
    cell: minimage::Cell,
}

impl PeriodicSymmetryOperations {
    /// Attach a periodic cell to caller-supplied operations.
    pub fn new(source: SymmetryOperations, cell: minimage::Cell) -> Self {
        Self { source, cell }
    }

    /// Periodic cell used for image displacements.
    pub fn cell(&self) -> &minimage::Cell {
        &self.cell
    }
}

impl SymmetrySource for PeriodicSymmetryOperations {
    fn operations(&self) -> &[AffineSymmetry] {
        self.source.operations()
    }

    fn displacement(
        &self,
        reference: ArrayView1<'_, f64>,
        image: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, SaddleError> {
        if reference.len() != image.len() || !reference.len().is_multiple_of(3) {
            return Err(SaddleError::Shape(
                "periodic symmetry configurations need matching atom triples".into(),
            ));
        }
        let mut displacement = Array1::zeros(reference.len());
        for atom in 0..reference.len() / 3 {
            let start = 3 * atom;
            let from = [reference[start], reference[start + 1], reference[start + 2]];
            let to = [image[start], image[start + 1], image[start + 2]];
            let delta = self.cell.displacement_euclidean(from, to);
            displacement[start] = delta[0];
            displacement[start + 1] = delta[1];
            displacement[start + 2] = delta[2];
        }
        Ok(displacement)
    }
}

/// A three-dimensional crystal point-group operation.
///
/// `permutation[source] = target` describes where each atom is placed. If
/// it is absent, [`SymmetryUniqueRegion::from_crystal_operations`] derives
/// it by species-aware nearest-image matching against the reactant.
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct CrystalSymmetryOperation {
    /// Orthogonal Cartesian matrix.
    pub rotation: Array2<f64>,
    /// Translation applied after rotation.
    pub translation: Option<Array1<f64>>,
    /// Centre about which the rotation acts.
    pub centre: Option<Array1<f64>>,
    /// Optional source-to-target atom permutation.
    pub permutation: Option<Vec<usize>>,
}

impl CrystalSymmetryOperation {
    /// Set [`Self::rotation`].
    #[must_use]
    pub fn with_rotation(mut self, rotation: Array2<f64>) -> Self {
        self.rotation = rotation;
        self
    }

    /// Set [`Self::translation`].
    #[must_use]
    pub fn with_translation(mut self, translation: Array1<f64>) -> Self {
        self.translation = Some(translation);
        self
    }

    /// Set [`Self::centre`].
    #[must_use]
    pub fn with_centre(mut self, centre: Array1<f64>) -> Self {
        self.centre = Some(centre);
        self
    }

    /// Set [`Self::permutation`].
    #[must_use]
    pub fn with_permutation(mut self, permutation: Vec<usize>) -> Self {
        self.permutation = Some(permutation);
        self
    }
}

/// Validation and matching controls for crystal symmetry operations.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct CrystalSymmetryConfig {
    /// Maximum absolute entry error in `R^T R - I`.
    pub orthogonality_tolerance: f64,
    /// Maximum atom distance when matching a transformed reactant.
    pub matching_tolerance: f64,
    /// Maximum atom distance for identifying two saddle images.
    pub orbit_tolerance: f64,
    /// Periodic cell used for every nearest-image displacement.
    pub cell: Option<minimage::Cell>,
}

impl Default for CrystalSymmetryConfig {
    fn default() -> Self {
        let geometric_tolerance = 256.0 * f64::EPSILON.sqrt();
        Self {
            orthogonality_tolerance: 64.0 * f64::EPSILON.sqrt(),
            matching_tolerance: geometric_tolerance,
            orbit_tolerance: geometric_tolerance,
            cell: None,
        }
    }
}

impl CrystalSymmetryConfig {
    /// Set [`Self::orthogonality_tolerance`].
    #[must_use]
    pub fn with_orthogonality_tolerance(mut self, tolerance: f64) -> Self {
        self.orthogonality_tolerance = tolerance;
        self
    }

    /// Set [`Self::matching_tolerance`].
    #[must_use]
    pub fn with_matching_tolerance(mut self, tolerance: f64) -> Self {
        self.matching_tolerance = tolerance;
        self
    }

    /// Set [`Self::orbit_tolerance`].
    #[must_use]
    pub fn with_orbit_tolerance(mut self, tolerance: f64) -> Self {
        self.orbit_tolerance = tolerance;
        self
    }

    /// Use `cell` for periodic nearest-image matching and orbit distances.
    #[must_use]
    pub fn with_cell(mut self, cell: minimage::Cell) -> Self {
        self.cell = Some(cell);
        self
    }
}

/// Voronoi cell of one representative among the symmetry images on a
/// hyperplane.
///
/// [`Region::degeneracy`] is the number of in-plane copies the cell stands
/// for. [`Self::from_source`] counts images of the supplied representative.
/// [`Self::from_crystal_operations`] counts the joint stabilizer of one hop.
#[derive(Clone, Debug)]
pub struct SymmetryUniqueRegion {
    representative: Array1<f64>,
    boundaries: Vec<PlanarBoundary>,
    degeneracy: usize,
}

/// Path multiplicity and in-plane asymmetric unit of one crystal hop.
///
/// Non-exhaustive result of [`SymmetryUniqueRegion::from_crystal_operations`].
/// The fields are already split: pass [`Self::path_multiplicity`] to
/// `RateMultiplicity::with_paths`, and use [`Self::region_degeneracy`] or
/// [`Region::degeneracy`] as the in-plane factor.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct CrystalSymmetryRegion {
    /// Asymmetric unit of the selected hop's dividing plane.
    pub region: SymmetryUniqueRegion,
    /// Distinct saddle images under the reactant stabilizer.
    pub path_multiplicity: usize,
    /// Order of the joint stabilizer of the reactant and that saddle.
    pub region_degeneracy: usize,
}

impl SymmetryUniqueRegion {
    /// Lift crystal operations into one hop's in-plane asymmetric unit.
    ///
    /// Operations that stabilize `reactant` form the reactant stabilizer.
    /// [`CrystalSymmetryRegion::path_multiplicity`] is the number of distinct
    /// images of `saddle` under that stabilizer. Pass it to
    /// `RateMultiplicity::with_paths`.
    /// [`CrystalSymmetryRegion::region_degeneracy`] is the number of supplied
    /// operations that stabilize both endpoints. Those operations map the
    /// dividing plane of this hop onto itself. The region is the asymmetric
    /// unit of that plane, and [`Region::degeneracy`] returns the same order.
    ///
    /// Jóhannesson and Jónsson, J. Chem. Phys. 115, 9644 (2001),
    /// doi:10.1063/1.1415499, Secs. II D and III D, keep the thermal sample in
    /// one symmetric part of the hyperplane by reflecting at the in-plane
    /// mirror between equivalent directions.
    ///
    /// Permutations are either supplied or obtained by species-aware
    /// nearest-image matching. The supplied group must include the identity.
    /// A lattice translation is not the identity: `operation_is_identity`
    /// rejects a translation whose absolute component exceeds
    /// `matching_tolerance`.
    ///
    /// ```
    /// use ndarray::{Array2, array};
    /// use rgsaddle::{
    ///     CrystalSymmetryConfig, CrystalSymmetryOperation, Region, SymmetryUniqueRegion,
    /// };
    ///
    /// let reactant = array![[0.0, 0.0, 0.0]];
    /// let saddle = array![[1.0, 0.0, 0.0]];
    /// let identity = CrystalSymmetryOperation::default().with_rotation(Array2::eye(3));
    /// let mirror = CrystalSymmetryOperation::default().with_rotation(array![
    ///     [1.0, 0.0, 0.0],
    ///     [0.0, -1.0, 0.0],
    ///     [0.0, 0.0, 1.0],
    /// ]);
    /// let built = SymmetryUniqueRegion::from_crystal_operations(
    ///     reactant.view(),
    ///     saddle.view(),
    ///     &[0],
    ///     &[identity, mirror],
    ///     &CrystalSymmetryConfig::default(),
    /// )
    /// .unwrap();
    /// assert_eq!(built.path_multiplicity, 1);
    /// assert_eq!(built.region_degeneracy, 2);
    /// assert_eq!(built.region.degeneracy(), built.region_degeneracy);
    /// let paths = built.path_multiplicity;
    /// let in_plane = built.region_degeneracy;
    /// assert_eq!(paths * in_plane, 2);
    /// ```
    pub fn from_crystal_operations(
        reactant: ArrayView2<'_, f64>,
        saddle: ArrayView2<'_, f64>,
        species: &[usize],
        operations: &[CrystalSymmetryOperation],
        config: &CrystalSymmetryConfig,
    ) -> Result<CrystalSymmetryRegion, SaddleError> {
        validate_crystal_input(reactant, saddle, species, operations, config)?;
        let saddle_flat = flatten_geometry(saddle);
        let mut saddle_images = vec![saddle_flat.clone()];
        let mut joint = Vec::new();
        let mut reactant_stabilizer = 0_usize;

        for (index, operation) in operations.iter().enumerate() {
            validate_crystal_operation(operation, reactant.nrows(), species, config)?;
            let Some(permutation) = reactant_permutation(reactant, operation, species, config)?
            else {
                continue;
            };
            reactant_stabilizer += 1;
            let image = mapped_flat(saddle, operation, &permutation, config)?;
            if same_geometry(&saddle_flat, &image, config)? {
                joint.push((index, permutation));
                continue;
            }
            let mut distinct = true;
            for known in &saddle_images {
                if same_geometry(known, &image, config)? {
                    distinct = false;
                    break;
                }
            }
            if distinct {
                saddle_images.push(image);
            }
        }
        if !operations
            .iter()
            .any(|operation| operation_is_identity(operation, config))
        {
            return Err(SaddleError::Invalid(
                "the crystal symmetry group must include the identity".into(),
            ));
        }
        if reactant_stabilizer == 0 {
            return Err(SaddleError::Invalid(
                "no supplied operation stabilizes the reactant".into(),
            ));
        }
        if joint.is_empty() {
            return Err(SaddleError::Invalid(
                "no supplied operation stabilizes both the reactant and the saddle".into(),
            ));
        }
        let normal = hop_displacement(reactant, saddle, config)?;
        let images = separating_images(&saddle_flat, &normal, operations, &joint, config)?;
        let region = asymmetric_unit(&saddle_flat, &images, joint.len())?;
        Ok(CrystalSymmetryRegion {
            region,
            path_multiplicity: saddle_images.len(),
            region_degeneracy: joint.len(),
        })
    }

    /// Build the cell from operations that map `plane` onto itself.
    ///
    /// Distinctness and plane-preservation tolerances are derived from
    /// machine epsilon, dimension and coordinate scale; no physical
    /// tolerance is guessed.
    pub fn from_source<S: SymmetrySource + ?Sized>(
        plane: &Hyperplane,
        representative: Array1<f64>,
        source: &S,
    ) -> Result<Self, SaddleError> {
        if representative.len() != plane.dim() {
            return Err(SaddleError::Shape(
                "symmetry representative and plane dimensions differ".into(),
            ));
        }
        let scale = 1.0 + representative.dot(&representative).sqrt();
        let tolerance = f64::EPSILON.sqrt() * plane.dim() as f64 * scale;
        if plane.distance(representative.view()).abs() > tolerance {
            return Err(SaddleError::Invalid(
                "symmetry representative does not lie on the plane".into(),
            ));
        }
        let mut images = vec![representative.clone()];
        for operation in source.operations() {
            if operation.linear.nrows() != plane.dim() || operation.linear.ncols() != plane.dim() {
                return Err(SaddleError::Shape(
                    "symmetry operation and plane dimensions differ".into(),
                ));
            }
            let gram = operation.linear.t().dot(&operation.linear);
            let orthogonality_error = gram
                .indexed_iter()
                .map(|((row, column), value)| {
                    let target = if row == column { 1.0 } else { 0.0 };
                    (value - target).abs()
                })
                .fold(0.0_f64, f64::max);
            if orthogonality_error > tolerance / scale {
                return Err(SaddleError::Invalid(
                    "symmetry operation is not orthogonal in sampling coordinates".into(),
                ));
            }
            let mapped_point = operation.apply(plane.point())?;
            let mapped_normal = operation.linear.dot(&plane.normal());
            let normal_length = mapped_normal.dot(&mapped_normal).sqrt();
            if !(normal_length.is_finite() && normal_length > 0.0) {
                return Err(SaddleError::Invalid(
                    "symmetry operation maps the plane normal to zero".into(),
                ));
            }
            let parallel = mapped_normal.dot(&plane.normal()).abs() / normal_length;
            if plane.distance(mapped_point.view()).abs() > tolerance
                || 1.0 - parallel > tolerance / scale
            {
                continue;
            }
            let raw_image = operation.apply(representative.view())?;
            let displacement = source.displacement(representative.view(), raw_image.view())?;
            let image = &representative + &displacement;
            if plane.distance(image.view()).abs() > tolerance {
                continue;
            }
            if images
                .iter()
                .all(|known| (&image - known).dot(&(&image - known)).sqrt() > tolerance)
            {
                images.push(image);
            }
        }
        let mut boundaries = Vec::with_capacity(images.len().saturating_sub(1));
        for image in images.iter().skip(1) {
            let direction = image - &representative;
            let midpoint = 0.5 * (image + &representative);
            boundaries.push(PlanarBoundary::containing(
                midpoint,
                direction,
                representative.view(),
            )?);
        }
        Ok(Self {
            representative,
            boundaries,
            degeneracy: images.len(),
        })
    }

    /// Representative configuration of this cell.
    pub fn representative(&self) -> ArrayView1<'_, f64> {
        self.representative.view()
    }
}

fn reactant_permutation(
    reactant: ArrayView2<'_, f64>,
    operation: &CrystalSymmetryOperation,
    species: &[usize],
    config: &CrystalSymmetryConfig,
) -> Result<Option<Vec<usize>>, SaddleError> {
    let transformed = transform_unpermuted(reactant, operation)?;
    match &operation.permutation {
        Some(permutation) => {
            if maps_reactant(transformed.view(), reactant, permutation, config)? {
                Ok(Some(permutation.clone()))
            } else {
                Ok(None)
            }
        }
        None => derive_permutation(transformed.view(), reactant, species, config),
    }
}

fn mapped_flat(
    reference: ArrayView2<'_, f64>,
    operation: &CrystalSymmetryOperation,
    permutation: &[usize],
    config: &CrystalSymmetryConfig,
) -> Result<Array1<f64>, SaddleError> {
    let transformed = transform_unpermuted(reference, operation)?;
    let permuted = permute_geometry(transformed.view(), permutation)?;
    unwrap_against(reference, permuted.view(), config)
}

fn hop_displacement(
    reactant: ArrayView2<'_, f64>,
    saddle: ArrayView2<'_, f64>,
    config: &CrystalSymmetryConfig,
) -> Result<Array1<f64>, SaddleError> {
    let mut normal = Array1::zeros(reactant.len());
    for atom in 0..reactant.nrows() {
        let displacement = displacement3(reactant.row(atom), saddle.row(atom), config)?;
        for axis in 0..3 {
            normal[3 * atom + axis] = displacement[axis];
        }
    }
    Ok(normal)
}

fn reject_normal(seed: &Array1<f64>, normal: &Array1<f64>) -> Array1<f64> {
    let norm_sq = normal.dot(normal);
    if !(norm_sq.is_finite() && norm_sq > 0.0) {
        return seed.clone();
    }
    let coefficient = seed.dot(normal) / norm_sq;
    seed - &(normal * coefficient)
}

fn in_plane_probe(
    saddle: &Array1<f64>,
    normal: &Array1<f64>,
    seed: &Array1<f64>,
    config: &CrystalSymmetryConfig,
) -> Option<Array1<f64>> {
    let offset = reject_normal(seed, normal);
    let raw = offset.dot(&offset).sqrt();
    if !(raw.is_finite() && raw > config.orbit_tolerance) {
        return None;
    }
    let hop = normal.dot(normal).sqrt();
    let target = if hop.is_finite() && hop > config.orbit_tolerance {
        hop
    } else {
        config.orbit_tolerance * (seed.len() as f64 + 1.0)
    };
    Some(saddle + &(offset * (target / raw)))
}

fn unflatten_geometry(flat: ArrayView1<'_, f64>) -> Result<Array2<f64>, SaddleError> {
    if flat.is_empty() || !flat.len().is_multiple_of(3) {
        return Err(SaddleError::Shape(
            "a crystal configuration needs a positive multiple of three coordinates".into(),
        ));
    }
    let mut geometry = Array2::zeros((flat.len() / 3, 3));
    for atom in 0..geometry.nrows() {
        for axis in 0..3 {
            geometry[[atom, axis]] = flat[3 * atom + axis];
        }
    }
    Ok(geometry)
}

fn probe_orbit(
    probe: ArrayView1<'_, f64>,
    operations: &[CrystalSymmetryOperation],
    joint: &[(usize, Vec<usize>)],
    config: &CrystalSymmetryConfig,
) -> Result<Vec<Array1<f64>>, SaddleError> {
    let geometry = unflatten_geometry(probe)?;
    let mut images = vec![probe.to_owned()];
    for (index, permutation) in joint {
        let image = mapped_flat(geometry.view(), &operations[*index], permutation, config)?;
        let mut distinct = true;
        for known in &images {
            if same_geometry(known, &image, config)? {
                distinct = false;
                break;
            }
        }
        if distinct {
            images.push(image);
        }
    }
    Ok(images)
}

fn separating_images(
    saddle: &Array1<f64>,
    normal: &Array1<f64>,
    operations: &[CrystalSymmetryOperation],
    joint: &[(usize, Vec<usize>)],
    config: &CrystalSymmetryConfig,
) -> Result<Vec<Array1<f64>>, SaddleError> {
    if joint.len() == 1 {
        return Ok(vec![saddle.clone()]);
    }
    let mut seed = Array1::zeros(saddle.len());
    for axis in 0..saddle.len() {
        seed[axis] = (axis + 1) as f64;
        let Some(probe) = in_plane_probe(saddle, normal, &seed, config) else {
            continue;
        };
        let images = probe_orbit(probe.view(), operations, joint, config)?;
        if images.len() == joint.len() {
            return Ok(images);
        }
    }
    Err(SaddleError::Invalid(
        "no point in the dividing plane separates the joint stabilizer".into(),
    ))
}

/// Dirichlet cell of `images` on the dividing plane through `saddle`.
///
/// The representative is the probe. Each face passes through the saddle, so
/// the fixed point of the joint stabilizer lies on the boundary of the cell.
fn asymmetric_unit(
    saddle: &Array1<f64>,
    images: &[Array1<f64>],
    degeneracy: usize,
) -> Result<SymmetryUniqueRegion, SaddleError> {
    if degeneracy == 0 || images.len() != degeneracy {
        return Err(SaddleError::Invalid(
            "the in-plane asymmetric unit must match the joint stabilizer".into(),
        ));
    }
    let representative = match images.first() {
        Some(representative) => representative.clone(),
        None => {
            return Err(SaddleError::Invalid(
                "an in-plane asymmetric unit needs a representative".into(),
            ));
        }
    };
    let mut boundaries = Vec::new();
    for image in images.iter().skip(1) {
        let direction = image - &representative;
        boundaries.push(PlanarBoundary::containing(
            saddle.clone(),
            direction,
            representative.view(),
        )?);
    }
    Ok(SymmetryUniqueRegion {
        representative,
        boundaries,
        degeneracy,
    })
}

fn validate_crystal_input(
    reactant: ArrayView2<'_, f64>,
    saddle: ArrayView2<'_, f64>,
    species: &[usize],
    operations: &[CrystalSymmetryOperation],
    config: &CrystalSymmetryConfig,
) -> Result<(), SaddleError> {
    if reactant.ncols() != 3
        || saddle.ncols() != 3
        || reactant.nrows() == 0
        || reactant.raw_dim() != saddle.raw_dim()
        || species.len() != reactant.nrows()
    {
        return Err(SaddleError::Shape(
            "crystal geometries need matching nonempty N by 3 rows and one species per atom".into(),
        ));
    }
    if operations.is_empty() {
        return Err(SaddleError::Invalid(
            "a crystal symmetry group needs at least one operation".into(),
        ));
    }
    if reactant
        .iter()
        .chain(saddle.iter())
        .any(|value| !value.is_finite())
    {
        return Err(SaddleError::NonFinite("crystal geometry"));
    }
    for tolerance in [
        config.orthogonality_tolerance,
        config.matching_tolerance,
        config.orbit_tolerance,
    ] {
        if !(tolerance.is_finite() && tolerance > 0.0) {
            return Err(SaddleError::Invalid(
                "crystal symmetry tolerances must be finite and positive".into(),
            ));
        }
    }
    Ok(())
}

/// Report whether `operation` is the identity.
///
/// A lattice translation is not the identity. A translation whose absolute
/// component exceeds `matching_tolerance` is rejected, as is a permutation
/// that moves an atom or a rotation that is not the identity within
/// `orthogonality_tolerance`.
fn operation_is_identity(
    operation: &CrystalSymmetryOperation,
    config: &CrystalSymmetryConfig,
) -> bool {
    if operation.rotation.dim() != (3, 3) {
        return false;
    }
    let rotation_error = operation
        .rotation
        .indexed_iter()
        .map(|((row, column), value)| {
            let target = if row == column { 1.0 } else { 0.0 };
            (value - target).abs()
        })
        .fold(0.0_f64, f64::max);
    if rotation_error > config.orthogonality_tolerance {
        return false;
    }
    if let Some(translation) = &operation.translation
        && translation
            .iter()
            .any(|value| value.abs() > config.matching_tolerance)
    {
        return false;
    }
    if let Some(permutation) = &operation.permutation
        && permutation
            .iter()
            .enumerate()
            .any(|(index, target)| *target != index)
    {
        return false;
    }
    true
}

fn validate_crystal_operation(
    operation: &CrystalSymmetryOperation,
    atom_count: usize,
    species: &[usize],
    config: &CrystalSymmetryConfig,
) -> Result<(), SaddleError> {
    if operation.rotation.dim() != (3, 3) {
        return Err(SaddleError::Shape(
            "crystal symmetry rotations must be 3 by 3".into(),
        ));
    }
    if operation.rotation.iter().any(|value| !value.is_finite()) {
        return Err(SaddleError::NonFinite("crystal symmetry rotation"));
    }
    for vector in [&operation.translation, &operation.centre]
        .into_iter()
        .flatten()
    {
        if vector.len() != 3 {
            return Err(SaddleError::Shape(
                "crystal symmetry translations and centres need three components".into(),
            ));
        }
        if vector.iter().any(|value| !value.is_finite()) {
            return Err(SaddleError::NonFinite("crystal symmetry offset"));
        }
    }
    let gram = operation.rotation.t().dot(&operation.rotation);
    let orthogonality_error = gram
        .indexed_iter()
        .map(|((row, column), value)| {
            let target = if row == column { 1.0 } else { 0.0 };
            (value - target).abs()
        })
        .fold(0.0_f64, f64::max);
    if orthogonality_error > config.orthogonality_tolerance {
        return Err(SaddleError::Invalid(
            "crystal symmetry rotation is not orthogonal".into(),
        ));
    }
    if let Some(permutation) = &operation.permutation {
        if permutation.len() != atom_count {
            return Err(SaddleError::Shape(
                "crystal symmetry permutation length differs from the atom count".into(),
            ));
        }
        let mut seen = vec![false; atom_count];
        for (source, target) in permutation.iter().copied().enumerate() {
            if target >= atom_count || seen[target] {
                return Err(SaddleError::Invalid(
                    "crystal symmetry permutation is not bijective".into(),
                ));
            }
            if species[source] != species[target] {
                return Err(SaddleError::Invalid(
                    "crystal symmetry permutation maps different species".into(),
                ));
            }
            seen[target] = true;
        }
    }
    Ok(())
}

fn transform_unpermuted(
    geometry: ArrayView2<'_, f64>,
    operation: &CrystalSymmetryOperation,
) -> Result<Array2<f64>, SaddleError> {
    let zero = Array1::zeros(3);
    let centre = match &operation.centre {
        Some(centre) => centre,
        None => &zero,
    };
    let translation = match &operation.translation {
        Some(translation) => translation,
        None => &zero,
    };
    let mut transformed = Array2::zeros(geometry.raw_dim());
    for atom in 0..geometry.nrows() {
        let rotated = operation.rotation.dot(&(&geometry.row(atom) - centre));
        transformed
            .row_mut(atom)
            .assign(&(&rotated + centre + translation));
    }
    Ok(transformed)
}

fn displacement3(
    from: ArrayView1<'_, f64>,
    to: ArrayView1<'_, f64>,
    config: &CrystalSymmetryConfig,
) -> Result<[f64; 3], SaddleError> {
    if from.len() != 3 || to.len() != 3 {
        return Err(SaddleError::Shape(
            "nearest-image displacement needs two three-vectors".into(),
        ));
    }
    match &config.cell {
        Some(cell) => {
            Ok(cell.displacement_euclidean([from[0], from[1], from[2]], [to[0], to[1], to[2]]))
        }
        None => Ok([to[0] - from[0], to[1] - from[1], to[2] - from[2]]),
    }
}

fn distance3(
    from: ArrayView1<'_, f64>,
    to: ArrayView1<'_, f64>,
    config: &CrystalSymmetryConfig,
) -> Result<f64, SaddleError> {
    let displacement = displacement3(from, to, config)?;
    Ok(displacement
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt())
}

fn maps_reactant(
    transformed: ArrayView2<'_, f64>,
    reactant: ArrayView2<'_, f64>,
    permutation: &[usize],
    config: &CrystalSymmetryConfig,
) -> Result<bool, SaddleError> {
    for (source, target) in permutation.iter().copied().enumerate() {
        if distance3(transformed.row(source), reactant.row(target), config)?
            > config.matching_tolerance
        {
            return Ok(false);
        }
    }
    Ok(true)
}

fn derive_permutation(
    transformed: ArrayView2<'_, f64>,
    reactant: ArrayView2<'_, f64>,
    species: &[usize],
    config: &CrystalSymmetryConfig,
) -> Result<Option<Vec<usize>>, SaddleError> {
    let atom_count = transformed.nrows();
    let mut candidates = vec![Vec::<(usize, f64)>::new(); atom_count];
    for source in 0..atom_count {
        for target in 0..atom_count {
            if species[source] == species[target] {
                let distance = distance3(transformed.row(source), reactant.row(target), config)?;
                if distance <= config.matching_tolerance {
                    candidates[source].push((target, distance));
                }
            }
        }
        candidates[source].sort_by(|left, right| left.1.total_cmp(&right.1));
        if candidates[source].is_empty() {
            return Ok(None);
        }
    }
    let mut target_to_source = vec![None; atom_count];
    for source in 0..atom_count {
        let mut visited = vec![false; atom_count];
        if !augment_matching(source, &candidates, &mut target_to_source, &mut visited) {
            return Ok(None);
        }
    }
    let mut permutation = vec![0; atom_count];
    for (target, source) in target_to_source.into_iter().enumerate() {
        match source {
            Some(source) => permutation[source] = target,
            None => return Ok(None),
        }
    }
    Ok(Some(permutation))
}

fn augment_matching(
    source: usize,
    candidates: &[Vec<(usize, f64)>],
    target_to_source: &mut [Option<usize>],
    visited: &mut [bool],
) -> bool {
    for (target, _) in &candidates[source] {
        if visited[*target] {
            continue;
        }
        visited[*target] = true;
        let available = match target_to_source[*target] {
            Some(previous) => augment_matching(previous, candidates, target_to_source, visited),
            None => true,
        };
        if available {
            target_to_source[*target] = Some(source);
            return true;
        }
    }
    false
}

fn permute_geometry(
    transformed: ArrayView2<'_, f64>,
    permutation: &[usize],
) -> Result<Array2<f64>, SaddleError> {
    if transformed.nrows() != permutation.len() {
        return Err(SaddleError::Shape(
            "transformed geometry and permutation sizes differ".into(),
        ));
    }
    let mut result = Array2::zeros(transformed.raw_dim());
    for (source, target) in permutation.iter().copied().enumerate() {
        result.row_mut(target).assign(&transformed.row(source));
    }
    Ok(result)
}

fn unwrap_against(
    reference: ArrayView2<'_, f64>,
    image: ArrayView2<'_, f64>,
    config: &CrystalSymmetryConfig,
) -> Result<Array1<f64>, SaddleError> {
    let mut unwrapped = Array1::zeros(reference.len());
    for atom in 0..reference.nrows() {
        let displacement = displacement3(reference.row(atom), image.row(atom), config)?;
        for axis in 0..3 {
            unwrapped[3 * atom + axis] = reference[[atom, axis]] + displacement[axis];
        }
    }
    Ok(unwrapped)
}

fn same_geometry(
    left: &Array1<f64>,
    right: &Array1<f64>,
    config: &CrystalSymmetryConfig,
) -> Result<bool, SaddleError> {
    if left.len() != right.len() || !left.len().is_multiple_of(3) {
        return Err(SaddleError::Shape(
            "saddle orbit images need matching atom triples".into(),
        ));
    }
    for atom in 0..left.len() / 3 {
        let start = 3 * atom;
        let left_atom = left.slice(ndarray::s![start..start + 3]);
        let right_atom = right.slice(ndarray::s![start..start + 3]);
        if distance3(left_atom, right_atom, config)? > config.orbit_tolerance {
            return Ok(false);
        }
    }
    Ok(true)
}

fn flatten_geometry(geometry: ArrayView2<'_, f64>) -> Array1<f64> {
    Array1::from_iter(geometry.iter().copied())
}

impl Region for SymmetryUniqueRegion {
    fn violation(
        &self,
        point: ArrayView1<'_, f64>,
    ) -> Result<Option<BoundaryViolation>, SaddleError> {
        for boundary in &self.boundaries {
            if let Some(violation) = boundary.violation(point)? {
                return Ok(Some(violation));
            }
        }
        Ok(None)
    }

    fn degeneracy(&self) -> usize {
        self.degeneracy
    }

    fn planar_boundaries(&self) -> Option<Vec<PlanarBoundary>> {
        Some(self.boundaries.clone())
    }
}

/// Region between the neighbouring hyperplanes of a path.
#[derive(Clone, Debug)]
pub struct PathSlab {
    boundaries: Vec<PlanarBoundary>,
}

impl PathSlab {
    /// Orient each supplied neighbouring plane so that `inside` is in its
    /// closed half-space. One neighbour is valid at a path endpoint.
    pub fn new(
        inside: ArrayView1<'_, f64>,
        previous: Option<&Hyperplane>,
        next: Option<&Hyperplane>,
    ) -> Result<Self, SaddleError> {
        let mut boundaries = Vec::with_capacity(2);
        for neighbour in [previous, next].into_iter().flatten() {
            boundaries.push(PlanarBoundary::containing(
                neighbour.point().to_owned(),
                neighbour.normal().to_owned(),
                inside,
            )?);
        }
        if boundaries.is_empty() {
            return Err(SaddleError::Invalid(
                "a path slab needs at least one neighbouring plane".into(),
            ));
        }
        Ok(Self { boundaries })
    }
}

impl Region for PathSlab {
    fn violation(
        &self,
        point: ArrayView1<'_, f64>,
    ) -> Result<Option<BoundaryViolation>, SaddleError> {
        for boundary in &self.boundaries {
            if let Some(violation) = boundary.violation(point)? {
                return Ok(Some(violation));
            }
        }
        Ok(None)
    }

    fn planar_boundaries(&self) -> Option<Vec<PlanarBoundary>> {
        Some(self.boundaries.clone())
    }
}

/// Reflect `velocity` specularly in an outward normal after both have been
/// projected into the active constraint tangent space.
pub fn reflect_velocity(
    velocity: &mut Array1<f64>,
    outward_normal: ArrayView1<'_, f64>,
    constraint: &impl crate::TangentProjector,
) -> Result<(), SaddleError> {
    if velocity.len() != outward_normal.len() {
        return Err(SaddleError::Shape(
            "velocity and boundary normal dimensions differ".into(),
        ));
    }
    let tangent = constraint.project(outward_normal);
    let norm = tangent.dot(&tangent).sqrt();
    if !(norm.is_finite() && norm > 0.0) {
        return Err(SaddleError::Invalid(
            "boundary has no normal component within the constraint".into(),
        ));
    }
    let tangent = tangent / norm;
    let component = velocity.dot(&tangent);
    velocity.scaled_add(-2.0 * component, &tangent);
    constraint.project_in_place(velocity);
    Ok(())
}

fn unit(vector: Array1<f64>) -> Result<Array1<f64>, SaddleError> {
    if vector.is_empty() || vector.iter().any(|value| !value.is_finite()) {
        return Err(SaddleError::NonFinite("boundary normal"));
    }
    let norm = vector.dot(&vector).sqrt();
    if norm > 0.0 {
        Ok(vector / norm)
    } else {
        Err(SaddleError::Invalid("boundary normal vanishes".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ProjectedConstraint;
    use ndarray::array;

    #[test]
    fn reflection_preserves_plane_and_speed() {
        let plane = Hyperplane::new(array![0.0, 0.0, 0.0], array![1.0, 1.0, 0.0]).unwrap();
        let mut velocity = plane.project(array![0.4, -0.4, 1.7].view());
        let speed = velocity.dot(&velocity);
        reflect_velocity(&mut velocity, array![0.0, 0.0, 1.0].view(), &plane).unwrap();
        assert!(velocity.dot(&plane.normal()).abs() <= 32.0 * f64::EPSILON);
        assert!((velocity.dot(&velocity) - speed).abs() <= 64.0 * f64::EPSILON * speed);
        let mut point = array![0.3, -0.2, 4.0];
        plane.restore_in_place(&mut point);
        assert!(plane.distance(point.view()).abs() <= 32.0 * f64::EPSILON);
    }

    #[test]
    fn symmetry_cell_has_explicit_degeneracy() {
        let plane = Hyperplane::new(array![0.0, 0.0], array![1.0, 0.0]).unwrap();
        let identity = AffineSymmetry::new(Array2::eye(2), Array1::zeros(2));
        let mirror = AffineSymmetry::new(array![[1.0, 0.0], [0.0, -1.0]], Array1::zeros(2));
        let source = SymmetryOperations::default().with_operations(vec![identity, mirror]);
        let region = SymmetryUniqueRegion::from_source(&plane, array![0.0, 1.0], &source).unwrap();
        assert_eq!(region.degeneracy(), 2);
        assert!(region.contains(array![0.0, 0.5].view()).unwrap());
        assert!(!region.contains(array![0.0, -0.5].view()).unwrap());
    }

    #[test]
    fn one_symmetric_cell_times_degeneracy_matches_the_full_line() {
        let plane = Hyperplane::new(array![0.0, 0.0], array![1.0, 0.0]).unwrap();
        let source = SymmetryOperations::default().with_operations(vec![
            AffineSymmetry::new(Array2::eye(2), Array1::zeros(2)),
            AffineSymmetry::new(array![[1.0, 0.0], [0.0, -1.0]], Array1::zeros(2)),
        ]);
        let region = SymmetryUniqueRegion::from_source(&plane, array![0.0, 1.0], &source).unwrap();
        let thermal = 0.2;
        let steps = 20_000;
        let limit = 3.0;
        let integrate = |start: f64, end: f64| {
            let width = (end - start) / steps as f64;
            (0..steps)
                .map(|index| {
                    let y = start + (index as f64 + 0.5) * width;
                    (-((y * y - 1.0).powi(2)) / thermal).exp() * width
                })
                .sum::<f64>()
        };
        let full = integrate(-limit, limit);
        let unique = integrate(0.0, limit);
        let reconstructed = region.degeneracy() as f64 * unique;
        let quadrature_roundoff = 128.0 * f64::EPSILON * full.abs();
        assert!((reconstructed - full).abs() <= quadrature_roundoff);
    }

    #[test]
    fn an_intersection_cannot_double_count_two_symmetry_orbits() {
        #[derive(Debug)]
        struct Twice;
        impl Region for Twice {
            fn violation(
                &self,
                _point: ArrayView1<'_, f64>,
            ) -> Result<Option<BoundaryViolation>, SaddleError> {
                Ok(None)
            }
            fn degeneracy(&self) -> usize {
                2
            }
        }
        assert!(RegionIntersection::new(vec![Arc::new(Twice), Arc::new(Twice)]).is_err());
        assert_eq!(RegionIntersection::default().degeneracy(), 1);
    }

    #[test]
    fn periodic_symmetry_images_use_minimum_image_displacements() {
        let plane = Hyperplane::new(array![0.0, 0.0, 0.0], array![1.0, 0.0, 0.0]).unwrap();
        let source = SymmetryOperations::default().with_operations(vec![
            AffineSymmetry::new(Array2::eye(3), Array1::zeros(3)),
            AffineSymmetry::new(Array2::eye(3), array![0.0, 9.0, 0.0]),
        ]);
        let periodic = PeriodicSymmetryOperations::new(
            source,
            minimage::Cell::ortho(10.0, 10.0, 10.0).unwrap(),
        );
        let region =
            SymmetryUniqueRegion::from_source(&plane, array![0.0, 0.0, 0.0], &periodic).unwrap();
        assert_eq!(region.degeneracy(), 2);
        assert!(region.contains(array![0.0, 0.2, 0.0].view()).unwrap());
        assert!(!region.contains(array![0.0, -0.8, 0.0].view()).unwrap());
    }
}
