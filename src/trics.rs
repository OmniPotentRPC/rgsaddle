//! Translation rotation internal coordinates for path construction.
//!
//! The primitive coordinates follow the TRIC construction of Wang and
//! Song, J. Chem. Phys. 144, 214108 (2016).  Bonded fragments carry
//! ordinary distances, angles and dihedrals together with three
//! translations and three rotations.  Eigenvectors of the primitive
//! Wilson metric form a nonredundant coordinate system.
//!
//! The segmented interpolation follows the algorithmic ideas in the
//! BSD licensed geomeTRIC reference implementation.  No geomeTRIC
//! source is included here.
//!
//! # Length unit
//!
//! Every absolute length threshold in this module is calibrated in
//! angstrom: the splice tolerance, the inversion move cap, the endpoint
//! agreement of an interpolated segment, and the coincidence guards of
//! the primitives.  [`TricsConfig::length_per_angstrom`] states how long
//! one angstrom is in the caller's coordinates, and [`trics_path`]
//! converts to angstrom before it works and back before it returns.  A
//! path in bohr or nanometres therefore needs only that one option.  The
//! lower level items ([`TricSystem`], [`interpolate_tric_segment`], and
//! the primitives) take coordinates in angstrom and do no conversion.
//! Only the finite difference step of the primitive derivatives follows
//! the size of the atoms it moves, so it never depends on the unit.

use std::cmp::Ordering;

use ndarray::{Array1, Array2, ArrayView1};

use crate::error::SaddleError;
use crate::internal::{CartAxis, Rotation, Translation};

const DEFAULT_BOND_FACTOR: f64 = 1.35;
const DEFAULT_RANK_TOLERANCE: f64 = 0.000_001;
/// Finite difference step relative to the extent of the moved atoms.
const DIFFERENCE_STEP: f64 = 0.000_001;
const VECTOR_TOLERANCE: f64 = 0.000_000_000_001;
const ANGLE_WARNING_COSINE: f64 = 0.906_307_787_036_649_9;
const ANGLE_UNSTABLE_COSINE: f64 = 0.965_925_826_289_068_3;
const ANGLE_FAILURE_COSINE: f64 = 0.996_194_698_091_745_5;
const LINEAR_ANGLE_COSINE: f64 = ANGLE_WARNING_COSINE;
const OUT_OF_PLANE_COSINE: f64 = 0.95;
const INVERSION_DAMPING_GROWTH: f64 = 1.2;
const INVERSION_REJECTION_LIMIT: usize = 5;
const SPLICE_DAMPING_REDUCTION: f64 = 0.8;
const SPLICE_DAMPING_MINIMUM: f64 = 0.2;
const SPLICE_DAMPING_RESET_STREAK: usize = 3;
const MINIMUM_SPLICE_LENGTH: usize = 2;
const MAXIMUM_SPLICE_LENGTH: usize = 6;
/// Endpoint agreement of an interpolated segment, in angstrom.
const INTERPOLATION_ENDPOINT_TOLERANCE: f64 = 0.01;
const DEFAULT_INVERSION_TOLERANCE: f64 = 0.000_000_01;
/// Largest single atom move of one inversion step, in angstrom.
const DEFAULT_INVERSION_MOVE: f64 = 0.1;
/// Paper convergence threshold for the maximum atom splice mismatch.
pub const TRICS_SPLICE_TOLERANCE_ANGSTROM: f64 = 0.001_8;

/// Controls for constructing a nonredundant TRIC system.
///
/// The bond graph is inferred from mutual nearest neighbour scales.
/// Explicit bonds replace that inference when supplied.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct TricConfig {
    /// Multiplicative threshold used by inferred bonds.
    pub bond_factor: f64,
    /// Absolute eigenvalue threshold for the primitive Wilson metric.
    pub rank_tolerance: f64,
    /// Optional zero based atom pairs defining the bond graph.
    pub bonds: Option<Vec<(usize, usize)>>,
}

with_setters!(TricConfig {
    with_bond_factor => bond_factor: val f64,
    with_rank_tolerance => rank_tolerance: val f64,
    with_bonds => bonds: opt Vec<(usize, usize)>,
});

impl Default for TricConfig {
    fn default() -> Self {
        Self {
            bond_factor: DEFAULT_BOND_FACTOR,
            rank_tolerance: DEFAULT_RANK_TOLERANCE,
            bonds: None,
        }
    }
}

/// Controls for iterative internal to Cartesian inversion.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct TricInversionConfig {
    /// Maximum Newton iterations for one Cartesian image.
    pub max_iterations: usize,
    /// Largest internal coordinate residual accepted, in angstrom and
    /// radian.
    pub tolerance: f64,
    /// Largest displacement of one atom in one iteration, in angstrom.
    pub max_atom_move: f64,
}

with_setters!(TricInversionConfig {
    with_max_iterations => max_iterations: val usize,
    with_tolerance => tolerance: val f64,
    with_max_atom_move => max_atom_move: val f64,
});

impl Default for TricInversionConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            tolerance: DEFAULT_INVERSION_TOLERANCE,
            max_atom_move: DEFAULT_INVERSION_MOVE,
        }
    }
}

/// Controls for TRICS path construction and splicing.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct TricsConfig {
    /// Coordinate construction controls.
    pub coordinates: TricConfig,
    /// Internal to Cartesian inversion controls.
    pub inversion: TricInversionConfig,
    /// Largest diagnostic level considered stable.
    pub diagnostic_threshold: u8,
    /// Maximum atom mismatch that ends endpoint splicing, in angstrom.
    pub splice_tolerance: f64,
    /// Maximum endpoint splice iterations.
    pub max_splice_iterations: usize,
    /// Length of one angstrom in the caller's coordinates.
    ///
    /// One for a path in angstrom, 0.1 for nanometres, and about 1.8897
    /// for bohr. Every length control of this configuration
    /// ([`TricsConfig::splice_tolerance`] and the inversion
    /// [`TricInversionConfig::max_atom_move`] and
    /// [`TricInversionConfig::tolerance`]) is stated in angstrom whatever
    /// this value is.
    pub length_per_angstrom: f64,
}

with_setters!(TricsConfig {
    with_coordinates => coordinates: val TricConfig,
    with_inversion => inversion: val TricInversionConfig,
    with_diagnostic_threshold => diagnostic_threshold: val u8,
    with_splice_tolerance => splice_tolerance: val f64,
    with_max_splice_iterations => max_splice_iterations: val usize,
    with_length_per_angstrom => length_per_angstrom: val f64,
});

impl Default for TricsConfig {
    fn default() -> Self {
        Self {
            coordinates: TricConfig::default(),
            inversion: TricInversionConfig::default(),
            diagnostic_threshold: 1,
            splice_tolerance: TRICS_SPLICE_TOLERANCE_ANGSTROM,
            max_splice_iterations: 1000,
            length_per_angstrom: 1.0,
        }
    }
}

/// One overlapping segment selected from the diagnostic matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct TricsSegment {
    /// First image in the segment.
    pub start: usize,
    /// Last image in the segment.
    pub end: usize,
    /// Path image whose TRIC system interpolates this segment.
    pub system: usize,
}

/// A completed TRICS path and its splice metadata.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct TricsReport {
    /// Interpolated Cartesian images.
    pub path: Array2<f64>,
    /// Overlapping diagnostic segments.
    pub segments: Vec<TricsSegment>,
    /// Endpoint splice iterations. Zero means one system covered the path.
    pub splice_iterations: usize,
}

/// One primitive coordinate used to span a TRIC system.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum TricPrimitive {
    /// Distance between two atoms.
    Distance(usize, usize),
    /// Angle `a-b-c`.
    Angle(usize, usize, usize),
    /// One of two bends spanning a linear angle.
    LinearBend(usize, usize, usize, [f64; 3]),
    /// Dihedral `a-b-c-d`.
    Dihedral(usize, usize, usize, usize),
    /// Mean fragment position along one Cartesian axis.
    Translation(Translation),
    /// Fragment orientation relative to the system reference.
    Rotation(Rotation),
    /// Radius of gyration scaled fragment rotation.
    ScaledRotation(Rotation, f64),
    /// Out of plane coordinate `a-b-c-d`.
    OutOfPlane(usize, usize, usize, usize),
}

impl TricPrimitive {
    /// Coordinate value at a packed `3N` Cartesian geometry.
    pub fn value(&self, positions: ArrayView1<'_, f64>) -> Result<f64, SaddleError> {
        check_positions(positions)?;
        match self {
            Self::Distance(a, b) => distance(positions, *a, *b),
            Self::Angle(a, b, c) => angle(positions, *a, *b, *c),
            Self::LinearBend(a, b, c, direction) => linear_bend(positions, *a, *b, *c, *direction),
            Self::Dihedral(a, b, c, d) => dihedral(positions, *a, *b, *c, *d),
            Self::Translation(coordinate) => coordinate.value(positions),
            Self::Rotation(coordinate) => coordinate.value(positions),
            Self::ScaledRotation(coordinate, factor) => Ok(coordinate.value(positions)? * factor),
            Self::OutOfPlane(a, b, c, d) => out_of_plane(positions, *a, *b, *c, *d),
        }
    }

    /// Cartesian derivative of this coordinate.
    pub fn derivative(&self, positions: ArrayView1<'_, f64>) -> Result<Array1<f64>, SaddleError> {
        check_positions(positions)?;
        match self {
            Self::Distance(a, b) => distance_derivative(positions, *a, *b),
            Self::Angle(a, b, c) => angle_derivative(positions, *a, *b, *c),
            Self::Translation(coordinate) => coordinate.gradient(positions),
            Self::Rotation(coordinate) => coordinate.gradient(positions),
            Self::ScaledRotation(coordinate, factor) => {
                Ok(coordinate.gradient(positions)? * *factor)
            }
            Self::Dihedral(..) | Self::LinearBend(..) | Self::OutOfPlane(..) => {
                central_derivative(self, positions)
            }
        }
    }

    /// Numerical stability level from zero through six.
    ///
    /// Zero is regular. One warns that an angular primitive is close
    /// to linear, two marks it unstable under small perturbations, and
    /// three or above marks imminent coordinate failure. Out-of-plane
    /// coordinates sum two adjacent angle scores up to six.
    pub fn diagnostic(&self, positions: ArrayView1<'_, f64>) -> u8 {
        match self {
            Self::Angle(a, b, c) => angular_diagnostic(positions, &[(*a, *b, *c)]),
            Self::Dihedral(a, b, c, d) => {
                angular_diagnostic(positions, &[(*a, *b, *c), (*b, *c, *d)])
            }
            Self::LinearBend(a, b, c, _) => linear_angular_diagnostic(positions, *a, *b, *c),
            Self::OutOfPlane(a, b, c, d) => out_of_plane_diagnostic(positions, *a, *b, *c, *d),
            Self::Distance(a, b) => match distance(positions, *a, *b) {
                Ok(_) => 0,
                Err(_) => 3,
            },
            Self::Translation(_) | Self::Rotation(_) | Self::ScaledRotation(..) => {
                match self.value(positions) {
                    Ok(_) => 0,
                    Err(_) => 3,
                }
            }
        }
    }

    fn difference(
        &self,
        left: ArrayView1<'_, f64>,
        right: ArrayView1<'_, f64>,
    ) -> Result<f64, SaddleError> {
        let mut difference = self.value(left)? - self.value(right)?;
        if matches!(self, Self::Dihedral(..) | Self::OutOfPlane(..)) {
            let period = std::f64::consts::TAU;
            if difference > std::f64::consts::PI {
                difference -= period;
            } else if difference < -std::f64::consts::PI {
                difference += period;
            }
        }
        Ok(difference)
    }
}

/// A nonredundant TRIC basis tied to one reference geometry.
#[derive(Clone, Debug)]
pub struct TricSystem {
    reference: Array1<f64>,
    primitives: Vec<TricPrimitive>,
    basis: Array2<f64>,
    rank_tolerance: f64,
}

impl TricSystem {
    /// Build a TRIC system from a packed `3N` reference geometry.
    pub fn new(reference: ArrayView1<'_, f64>, config: &TricConfig) -> Result<Self, SaddleError> {
        check_positions(reference)?;
        check_config(config)?;
        let n_atoms = reference.len() / 3;
        let bonds = match &config.bonds {
            Some(bonds) => validate_bonds(bonds, n_atoms)?,
            None => infer_bonds(reference, config.bond_factor)?,
        };
        let fragments = fragments(n_atoms, &bonds);
        let mut primitives = molecular_primitives(&bonds, n_atoms, reference)?;
        for fragment in fragments {
            for axis in CartAxis::ALL {
                primitives.push(TricPrimitive::Translation(Translation::new(
                    fragment.clone(),
                    axis,
                )?));
            }
            if fragment.len() > 1 {
                let refpos = fragment_positions(reference, &fragment);
                let rotation_scale = radius_of_gyration(&refpos);
                for axis in CartAxis::ALL {
                    primitives.push(TricPrimitive::ScaledRotation(
                        Rotation::new(fragment.clone(), axis, refpos.clone())?,
                        rotation_scale,
                    ));
                }
            }
        }
        let primitive_b = primitive_wilson_b(&primitives, reference)?;
        let metric = primitive_b.dot(&primitive_b.t());
        let basis = nonredundant_basis(&metric, config.rank_tolerance)?;
        if basis.ncols() == 0 {
            return Err(SaddleError::Invalid(
                "TRIC construction found no independent coordinates".into(),
            ));
        }
        Ok(Self {
            reference: reference.to_owned(),
            primitives,
            basis,
            rank_tolerance: config.rank_tolerance,
        })
    }

    /// Reference geometry used to define rotations and delocalization.
    pub fn reference(&self) -> ArrayView1<'_, f64> {
        self.reference.view()
    }

    /// Primitive coordinates before delocalization.
    pub fn primitives(&self) -> &[TricPrimitive] {
        &self.primitives
    }

    /// Primitive to nonredundant transformation, one coordinate per column.
    pub fn basis(&self) -> &Array2<f64> {
        &self.basis
    }

    /// Number of nonredundant coordinates.
    pub fn dimension(&self) -> usize {
        self.basis.ncols()
    }

    /// Nonredundant coordinate difference `q(left) - q(right)`.
    pub fn difference(
        &self,
        left: ArrayView1<'_, f64>,
        right: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, SaddleError> {
        check_same_positions(left, right)?;
        let mut primitive = Array1::zeros(self.primitives.len());
        for (slot, coordinate) in primitive.iter_mut().zip(&self.primitives) {
            *slot = coordinate.difference(left, right)?;
        }
        Ok(self.basis.t().dot(&primitive))
    }

    /// Wilson B matrix of the nonredundant coordinates.
    pub fn wilson_b(&self, positions: ArrayView1<'_, f64>) -> Result<Array2<f64>, SaddleError> {
        let primitive = primitive_wilson_b(&self.primitives, positions)?;
        Ok(self.basis.t().dot(&primitive))
    }

    /// Highest primitive stability diagnostic at `positions`.
    pub fn diagnostic(&self, positions: ArrayView1<'_, f64>) -> u8 {
        self.primitives
            .iter()
            .map(|primitive| primitive.diagnostic(positions))
            .max()
            .unwrap_or(0)
    }

    /// Iteratively invert an internal displacement from `start`.
    pub fn back_transform(
        &self,
        start: ArrayView1<'_, f64>,
        displacement: ArrayView1<'_, f64>,
        config: &TricInversionConfig,
    ) -> Result<Array1<f64>, SaddleError> {
        check_positions(start)?;
        check_inversion_config(config)?;
        if start.len() != self.reference.len() || displacement.len() != self.dimension() {
            return Err(SaddleError::Shape(
                "TRIC inversion dimensions do not match the coordinate system".into(),
            ));
        }
        let mut positions = start.to_owned();
        let mut remaining = displacement.to_owned();
        let initial_error = remaining.dot(&remaining).sqrt();
        if initial_error <= config.tolerance {
            return Ok(positions);
        }
        let mut best_error = initial_error;
        let mut damping = 1.0;
        let mut rejected = 0;
        for _ in 0..config.max_iterations {
            let b = self.wilson_b(positions.view())?;
            let gram = b.dot(&b.t());
            let multipliers = solve_metric(&gram, &remaining, self.rank_tolerance)?;
            let mut step = cap_atom_step(b.t().dot(&multipliers), config.max_atom_move);
            step *= damping;
            let trial = &positions + &step;
            if trial.iter().any(|value| !value.is_finite()) {
                return Err(SaddleError::NonFinite("TRIC back transformation"));
            }
            let actual = self.difference(trial.view(), positions.view())?;
            let next_remaining = &remaining - &actual;
            let error = next_remaining.dot(&next_remaining).sqrt();
            if error > best_error {
                damping *= 0.5;
                rejected += 1;
                if rejected >= INVERSION_REJECTION_LIMIT {
                    return Err(SaddleError::Solver(
                        "TRIC back transformation rejected five damped steps".into(),
                    ));
                }
                continue;
            }
            positions = trial;
            remaining = next_remaining;
            best_error = error;
            rejected = 0;
            damping = (damping * INVERSION_DAMPING_GROWTH).min(1.0);
            if error <= config.tolerance {
                return Ok(positions);
            }
            if max_atom_norm(&step) <= config.tolerance {
                return Err(SaddleError::Solver(
                    "TRIC back transformation stalled before reaching tolerance".into(),
                ));
            }
        }
        Err(SaddleError::Solver(
            "TRIC back transformation did not converge".into(),
        ))
    }
}

/// Interpolate one segment in a fixed TRIC metric.
///
/// The endpoints are copied exactly. Each interior image advances by
/// the same internal displacement and is recovered by iterative
/// internal to Cartesian inversion.
pub fn interpolate_tric_segment(
    system: &TricSystem,
    left: ArrayView1<'_, f64>,
    right: ArrayView1<'_, f64>,
    n_images: usize,
    config: &TricInversionConfig,
) -> Result<Array2<f64>, SaddleError> {
    check_same_positions(left, right)?;
    if n_images < 2 {
        return Err(SaddleError::Shape(
            "TRIC interpolation needs at least two images".into(),
        ));
    }
    match interpolate_one_direction(system, left, right, n_images, config) {
        Ok(path) => Ok(path),
        Err(forward_error) => {
            match interpolate_one_direction(system, right, left, n_images, config) {
                Ok(backward) => Ok(Array2::from_shape_fn(
                    backward.raw_dim(),
                    |(row, column)| backward[(n_images - row - 1, column)],
                )),
                Err(_) => Err(forward_error),
            }
        }
    }
}

fn interpolate_one_direction(
    system: &TricSystem,
    left: ArrayView1<'_, f64>,
    right: ArrayView1<'_, f64>,
    n_images: usize,
    config: &TricInversionConfig,
) -> Result<Array2<f64>, SaddleError> {
    let total = system.difference(right, left)?;
    let increment = total / (n_images - 1) as f64;
    let mut path = Array2::zeros((n_images, left.len()));
    path.row_mut(0).assign(&left);
    let mut current = left.to_owned();
    for image in 1..n_images {
        current = system.back_transform(current.view(), increment.view(), config)?;
        path.row_mut(image).assign(&current);
    }
    if aligned_max_atom_difference(current.view(), right)? > INTERPOLATION_ENDPOINT_TOLERANCE {
        return Err(SaddleError::Solver(
            "TRIC interpolation did not reach its aligned endpoint".into(),
        ));
    }
    path.row_mut(n_images - 1).assign(&right);
    Ok(path)
}

/// Build one TRIC system per image of an initial path.
pub fn tric_systems(
    path: ndarray::ArrayView2<'_, f64>,
    config: &TricConfig,
) -> Result<Vec<TricSystem>, SaddleError> {
    if path.nrows() < 2 || path.ncols() == 0 || !path.ncols().is_multiple_of(3) {
        return Err(SaddleError::Shape(
            "a TRIC path needs at least two 3N images".into(),
        ));
    }
    path.outer_iter()
        .map(|image| TricSystem::new(image, config))
        .collect()
}

/// Diagnostic matrix `D`.
///
/// Row `i` evaluates the TRIC system built at image `i` on every path
/// image `j`. Entries range from zero for a regular system to six
/// when both adjacent angles of an out-of-plane coordinate are near
/// linear.
pub fn diagnostic_matrix(
    systems: &[TricSystem],
    path: ndarray::ArrayView2<'_, f64>,
) -> Result<Array2<u8>, SaddleError> {
    if systems.len() != path.nrows() {
        return Err(SaddleError::Shape(
            "the TRIC diagnostic needs one system per path image".into(),
        ));
    }
    if systems
        .iter()
        .any(|system| system.reference.len() != path.ncols())
    {
        return Err(SaddleError::Shape(
            "a TRIC system and path image have different dimensions".into(),
        ));
    }
    Ok(Array2::from_shape_fn(
        (systems.len(), path.nrows()),
        |(system, image)| systems[system].diagnostic(path.row(image)),
    ))
}

/// Find overlapping low diagnostic segments through `D`.
///
/// This is the discrete car through the maze procedure. Each row
/// contributes contiguous stable intervals. The selected chain starts
/// at the first image, ends at the last, and maximizes forward progress
/// while retaining an overlap with the preceding interval.
pub fn tric_segments(
    diagnostic: ndarray::ArrayView2<'_, u8>,
    threshold: u8,
) -> Result<Vec<TricsSegment>, SaddleError> {
    if diagnostic.nrows() == 0 || diagnostic.nrows() != diagnostic.ncols() {
        return Err(SaddleError::Shape(
            "the TRIC diagnostic matrix must be nonempty and square".into(),
        ));
    }
    let n = diagnostic.nrows();
    for system in 0..n {
        if (0..n).all(|image| diagnostic[(system, image)] <= threshold) {
            return Ok(vec![TricsSegment {
                start: 0,
                end: n - 1,
                system,
            }]);
        }
    }
    let maze_segments = maze_path(diagnostic, threshold)?;
    let splice_length = splice_length(n);
    Ok(fill_splice_segments_with_systems(
        &maze_segments,
        diagnostic,
        threshold,
        splice_length,
        n,
    ))
}

/// Refine an initial Cartesian path with TRICS.
///
/// The supplied path may be linear, IDPP, or caller generated. One
/// nonredundant TRIC system is built per image, low diagnostic segments
/// are interpolated in their own metrics, and overlapping endpoints are
/// iterated until their maximum atom mismatch reaches the configured
/// threshold.
///
/// Coordinates are in the caller's length unit, which
/// [`TricsConfig::length_per_angstrom`] relates to the angstrom of the
/// thresholds. The first and last image are returned exactly as given.
pub fn trics_path(
    initial: ndarray::ArrayView2<'_, f64>,
    config: &TricsConfig,
) -> Result<TricsReport, SaddleError> {
    check_trics_config(config)?;
    if initial.iter().any(|value| !value.is_finite()) {
        return Err(SaddleError::NonFinite("TRICS path coordinates"));
    }
    let unit = config.length_per_angstrom;
    if unit == 1.0 {
        return trics_path_in_angstrom(initial, config);
    }
    let scaled = initial.mapv(|value| value / unit);
    if scaled.iter().any(|value| !value.is_finite()) {
        return Err(SaddleError::NonFinite("TRICS path in angstrom"));
    }
    let mut report = trics_path_in_angstrom(scaled.view(), config)?;
    report.path.mapv_inplace(|value| value * unit);
    if report.path.iter().any(|value| !value.is_finite()) {
        return Err(SaddleError::NonFinite("TRICS path in caller units"));
    }
    let last = report.path.nrows() - 1;
    report.path.row_mut(0).assign(&initial.row(0));
    report.path.row_mut(last).assign(&initial.row(last));
    Ok(report)
}

fn trics_path_in_angstrom(
    initial: ndarray::ArrayView2<'_, f64>,
    config: &TricsConfig,
) -> Result<TricsReport, SaddleError> {
    let systems = tric_systems(initial, &config.coordinates)?;
    let diagnostic = diagnostic_matrix(&systems, initial)?;
    let segments = tric_segments(diagnostic.view(), config.diagnostic_threshold)?;
    if segments.len() == 1 {
        let segment = segments[0];
        let mut candidates = vec![segment.system];
        candidates.extend((0..systems.len()).filter(|&system| {
            system != segment.system
                && (0..initial.nrows())
                    .all(|image| diagnostic[(system, image)] <= config.diagnostic_threshold)
        }));
        candidates.retain(|&system| {
            (0..initial.nrows())
                .all(|image| diagnostic[(system, image)] <= config.diagnostic_threshold)
        });
        let mut first_error = None;
        for system in candidates {
            match interpolate_tric_segment(
                &systems[system],
                initial.row(0),
                initial.row(initial.nrows() - 1),
                initial.nrows(),
                &config.inversion,
            ) {
                Ok(path) => {
                    return Ok(TricsReport {
                        path,
                        segments: vec![TricsSegment { system, ..segment }],
                        splice_iterations: 0,
                    });
                }
                Err(error) if first_error.is_none() => first_error = Some(error),
                Err(_) => {}
            }
        }
        return Err(first_error.unwrap_or_else(|| {
            SaddleError::Solver("TRICS found no stable coordinate system".into())
        }));
    }
    splice_path(initial, &systems, segments, config)
}

fn maze_path(
    diagnostic: ndarray::ArrayView2<'_, u8>,
    threshold: u8,
) -> Result<Vec<TricsSegment>, SaddleError> {
    let n = diagnostic.nrows();
    let mut okay = Array2::from_shape_fn((n, n), |index| diagnostic[index] <= threshold);
    if !okay[(0, 0)] || !okay[(n - 1, n - 1)] {
        return Err(SaddleError::Invalid(
            "TRICS could not enter or leave the diagnostic maze".into(),
        ));
    }
    for _ in 0..n * n {
        let previous = okay.clone();
        for system in 0..n {
            for image in 0..n {
                if (system == 0 && image == 0) || (system == n - 1 && image == n - 1) {
                    continue;
                }
                let forward_blocked = image < n - 1
                    && !previous[(system, image + 1)]
                    && (system + 1 >= n || !previous[(system + 1, image)]);
                let backward_blocked = image > 0
                    && !previous[(system, image - 1)]
                    && (system == 0 || !previous[(system - 1, image)]);
                if forward_blocked || backward_blocked {
                    okay[(system, image)] = false;
                }
            }
        }
        if okay == previous {
            break;
        }
    }

    let mut candidates: Vec<Option<Vec<TricsSegment>>> = vec![None; (n / 2).max(1)];
    for (size, candidate) in candidates.iter_mut().enumerate() {
        for start in 0..n {
            if let Some(segments) = drive_maze(&okay, start, size, false) {
                keep_maze_candidate(candidate, segments);
            }
        }
        for start in (0..n).rev() {
            if let Some(segments) = drive_maze(&okay, start, size, true) {
                keep_maze_candidate(candidate, segments);
            }
        }
    }
    let minimum = candidates
        .iter()
        .filter_map(|candidate| candidate.as_ref().map(Vec::len))
        .min()
        .ok_or_else(|| SaddleError::Invalid("TRICS diagnostic maze has no path".into()))?;
    candidates
        .into_iter()
        .rev()
        .flatten()
        .find(|segments| segments.len() == minimum)
        .ok_or_else(|| SaddleError::Invalid("TRICS diagnostic maze has no path".into()))
}

fn drive_maze(
    okay: &Array2<bool>,
    start_system: usize,
    size: usize,
    reverse: bool,
) -> Option<Vec<TricsSegment>> {
    let n = okay.nrows();
    let mut system = start_system;
    let mut image = if reverse { n - 1 } else { 0 };
    if !okay[(system, image)] {
        return None;
    }
    let mut last = image;
    let mut along_images = true;
    let mut segments = Vec::new();
    for _ in 0..=2 * n {
        if (!reverse && image == n - 1) || (reverse && image == 0) {
            let (start, end) = if reverse { (0, last) } else { (last, n - 1) };
            segments.push(TricsSegment { start, end, system });
            if reverse {
                segments.reverse();
            }
            return Some(segments);
        }
        if !maze_square_is_open(okay, system, image, size) {
            return None;
        }
        if along_images {
            let next_image = if reverse {
                image.checked_sub(1)
            } else {
                image.checked_add(1).filter(|next| *next < n)
            }?;
            if maze_square_is_open(okay, system, next_image, size) {
                image = next_image;
            } else {
                along_images = false;
                let (start, end) = if reverse {
                    (image, last)
                } else {
                    (last, image)
                };
                segments.push(TricsSegment { start, end, system });
                last = next_image;
                system = if reverse {
                    system.checked_sub(1)?
                } else {
                    system.checked_add(1).filter(|next| *next < n)?
                };
            }
        } else {
            let next_system = if reverse {
                system.checked_sub(1)
            } else {
                system.checked_add(1).filter(|next| *next < n)
            };
            if let Some(next) = next_system
                && maze_square_is_open(okay, next, image, size)
            {
                system = next;
                continue;
            }
            along_images = true;
            image = if reverse {
                image.checked_sub(1)?
            } else {
                image.checked_add(1).filter(|next| *next < n)?
            };
        }
    }
    None
}

fn maze_square_is_open(okay: &Array2<bool>, system: usize, image: usize, size: usize) -> bool {
    let row_start = system.saturating_sub(size);
    let row_end = (system + size + 1).min(okay.nrows());
    let column_start = image.saturating_sub(size);
    let column_end = (image + size + 1).min(okay.ncols());
    (row_start..row_end).all(|row| (column_start..column_end).all(|column| okay[(row, column)]))
}

fn keep_maze_candidate(current: &mut Option<Vec<TricsSegment>>, candidate: Vec<TricsSegment>) {
    let replace = match current {
        Some(existing) if candidate.len() > existing.len() => false,
        Some(existing) if candidate.len() < existing.len() => true,
        Some(existing) => {
            let shortest = candidate
                .iter()
                .map(|segment| segment.end - segment.start)
                .min()
                .unwrap_or(0);
            let existing_shortest = existing
                .iter()
                .map(|segment| segment.end - segment.start)
                .min()
                .unwrap_or(0);
            shortest > existing_shortest
        }
        None => true,
    };
    if replace {
        *current = Some(candidate);
    }
}

fn splice_length(n_images: usize) -> usize {
    let rounded = ((n_images as f64) / 10.0).round() as usize;
    let even = rounded + rounded % 2;
    match (
        even.cmp(&MINIMUM_SPLICE_LENGTH),
        even.cmp(&MAXIMUM_SPLICE_LENGTH),
    ) {
        (Ordering::Less, _) => MINIMUM_SPLICE_LENGTH,
        (_, Ordering::Greater) => MAXIMUM_SPLICE_LENGTH,
        _ => even,
    }
}

fn merge_segment_pair(
    segments: &[TricsSegment],
    merge_index: usize,
    diagnostic: ndarray::ArrayView2<'_, u8>,
    threshold: u8,
) -> Vec<TricsSegment> {
    let mut merged = Vec::with_capacity(segments.len() - 1);
    for (index, &segment) in segments.iter().enumerate() {
        if index == merge_index {
            let start = segment.start;
            let end = segments[index + 1].end;
            let system = best_system(diagnostic, start, end, segment.system, threshold);
            merged.push(TricsSegment { start, end, system });
        } else if index != merge_index + 1 {
            merged.push(segment);
        }
    }
    merged
}

#[cfg(test)]
fn fill_splice_segments(
    segments: &[(usize, usize)],
    splice_length: usize,
    n_images: usize,
) -> Vec<(usize, usize)> {
    let diagnostic = Array2::zeros((n_images, n_images));
    let typed: Vec<_> = segments
        .iter()
        .map(|&(start, end)| TricsSegment {
            start,
            end,
            system: start,
        })
        .collect();
    fill_splice_segments_with_systems(&typed, diagnostic.view(), 0, splice_length, n_images)
        .into_iter()
        .map(|segment| (segment.start, segment.end))
        .collect()
}

fn fill_splice_segments_with_systems(
    segments: &[TricsSegment],
    diagnostic: ndarray::ArrayView2<'_, u8>,
    threshold: u8,
    splice_length: usize,
    n_images: usize,
) -> Vec<TricsSegment> {
    let mut filled = segments.to_vec();
    while filled.len() > 2
        && filled
            .iter()
            .map(|segment| segment.end - segment.start)
            .min()
            .is_some_and(|length| length < splice_length)
    {
        let merge_index = (0..filled.len() - 1)
            .filter(|&index| {
                let start = filled[index].start;
                let end = filled[index + 1].end;
                (0..diagnostic.nrows()).any(|system| {
                    (start..=end).all(|image| diagnostic[(system, image)] <= threshold)
                })
            })
            .min_by_key(|&index| {
                merge_segment_pair(&filled, index, diagnostic, threshold)
                    .iter()
                    .map(|segment| segment.end - segment.start)
                    .max()
                    .unwrap_or(0)
            });
        let Some(merge_index) = merge_index else {
            break;
        };
        filled = merge_segment_pair(&filled, merge_index, diagnostic, threshold);
    }
    let half = splice_length / 2;
    let mut spliced: Vec<_> = filled
        .iter()
        .map(|segment| {
            let start = segment.start.saturating_sub(half);
            let end = (segment.end + half).min(n_images - 1);
            TricsSegment {
                start,
                end,
                system: best_system(diagnostic, start, end, segment.system, threshold),
            }
        })
        .collect();
    if spliced.len() > 2 && spliced[1].start == 0 {
        spliced.remove(0);
        filled.remove(0);
        filled[0].start = 0;
    }
    if spliced.len() > 2 && spliced[spliced.len() - 2].end == n_images - 1 {
        spliced.pop();
        filled.pop();
        let last = filled.len() - 1;
        filled[last].end = n_images - 1;
    }
    if spliced.len() == 2 {
        if n_images >= splice_length + 2
            && (spliced[1].start == 0
                || spliced[0].end == n_images - 1
                || spliced[0].end + 1 - spliced[0].start < splice_length
                || spliced[1].end + 1 - spliced[1].start < splice_length)
        {
            let boundary = filled[0].end.clamp(half, n_images - 2 - half);
            let first_end = boundary + half;
            let second_start = boundary + 1 - half;
            spliced[0] = TricsSegment {
                start: 0,
                end: first_end,
                system: best_system(diagnostic, 0, first_end, filled[0].system, threshold),
            };
            spliced[1] = TricsSegment {
                start: second_start,
                end: n_images - 1,
                system: best_system(
                    diagnostic,
                    second_start,
                    n_images - 1,
                    filled[1].system,
                    threshold,
                ),
            };
        } else if n_images == 3 {
            spliced[0] = TricsSegment {
                start: 0,
                end: 1,
                system: best_system(diagnostic, 0, 1, filled[0].system, threshold),
            };
            spliced[1] = TricsSegment {
                start: 1,
                end: 2,
                system: best_system(diagnostic, 1, 2, filled[1].system, threshold),
            };
        }
    }
    spliced
}

fn best_system(
    diagnostic: ndarray::ArrayView2<'_, u8>,
    start: usize,
    end: usize,
    preferred: usize,
    threshold: u8,
) -> usize {
    let n = diagnostic.nrows();
    if preferred < n && (start..=end).all(|image| diagnostic[(preferred, image)] <= threshold) {
        return preferred;
    }
    let midpoint = (start + end) / 2;
    (0..n)
        .min_by_key(|&system| {
            let level = (start..=end)
                .map(|image| diagnostic[(system, image)])
                .max()
                .unwrap_or(u8::MAX);
            (
                level > threshold,
                level,
                usize::from(system != preferred),
                usize::from(!(start..=end).contains(&system)),
                system.abs_diff(midpoint),
            )
        })
        .unwrap_or(start)
}

fn segment_candidates(
    segments: &[TricsSegment],
    systems: &[TricSystem],
    initial: ndarray::ArrayView2<'_, f64>,
    threshold: u8,
) -> Vec<Vec<usize>> {
    let half = splice_length(initial.nrows()) / 2;
    segments
        .iter()
        .enumerate()
        .map(|(index, segment)| {
            let midpoint = (segment.start + segment.end) / 2;
            let core_start = if index == 0 {
                segment.start
            } else {
                (segment.start + half).min(segment.end)
            };
            let core_end = if index + 1 == segments.len() {
                segment.end
            } else {
                segment.end.saturating_sub(half).max(core_start)
            };
            let expanded_level = |system: usize| {
                (segment.start..=segment.end)
                    .map(|image| systems[system].diagnostic(initial.row(image)))
                    .max()
                    .unwrap_or(u8::MAX)
            };
            let core_level = |system: usize| {
                (core_start..=core_end)
                    .map(|image| systems[system].diagnostic(initial.row(image)))
                    .max()
                    .unwrap_or(u8::MAX)
            };
            let mut candidates = vec![segment.system];
            for system in segment.start..=segment.end {
                if !candidates.contains(&system) {
                    candidates.push(system);
                }
            }
            let has_strict = candidates
                .iter()
                .any(|&system| expanded_level(system) <= threshold);
            candidates.retain(|&system| {
                if has_strict {
                    expanded_level(system) <= threshold
                } else {
                    core_level(system) <= threshold && expanded_level(system) < 3
                }
            });
            candidates.sort_by_key(|&system| {
                let endpoint_priority = if index == 0 && system == segment.start
                    || index + 1 == segments.len() && system == segment.end
                {
                    0
                } else {
                    1
                };
                (
                    expanded_level(system),
                    core_level(system),
                    endpoint_priority,
                    usize::from(system != segment.system),
                    system.abs_diff(midpoint),
                )
            });
            candidates
        })
        .collect()
}

fn splice_path(
    initial: ndarray::ArrayView2<'_, f64>,
    systems: &[TricSystem],
    mut segments: Vec<TricsSegment>,
    config: &TricsConfig,
) -> Result<TricsReport, SaddleError> {
    let mut endpoints: Vec<[Array1<f64>; 2]> = segments
        .iter()
        .map(|segment| {
            [
                initial.row(segment.start).to_owned(),
                initial.row(segment.end).to_owned(),
            ]
        })
        .collect();
    let segment_candidates =
        segment_candidates(&segments, systems, initial, config.diagnostic_threshold);
    let mut interpolated = Vec::new();
    let splice_length = segments
        .windows(2)
        .map(|pair| pair[0].end + 1 - pair[1].start)
        .min()
        .unwrap_or_else(|| splice_length(initial.nrows()));
    let mut damping = 1.0;
    let mut last_mismatch = f64::INFINITY;
    let mut decrease_streak = 0;
    for iteration in 1..=config.max_splice_iterations {
        interpolated.clear();
        for index in 0..segments.len() {
            let segment = segments[index];
            let endpoint = &endpoints[index];
            let mut result = None;
            for &system in &segment_candidates[index] {
                if let Ok(path) = interpolate_tric_segment(
                    &systems[system],
                    endpoint[0].view(),
                    endpoint[1].view(),
                    segment.end - segment.start + 1,
                    &config.inversion,
                ) {
                    result = Some((system, path));
                    break;
                }
            }
            let Some((system, path)) = result else {
                return Err(SaddleError::Solver(
                    "TRICS could not interpolate a selected segment".into(),
                ));
            };
            segments[index].system = system;
            interpolated.push(path);
        }
        let mismatch = splice_mismatch(&interpolated, splice_length);
        if mismatch <= config.splice_tolerance {
            let path = concatenate_segments(initial.nrows(), &segments, &interpolated);
            return Ok(TricsReport {
                path,
                segments,
                splice_iterations: iteration,
            });
        }
        if mismatch < last_mismatch {
            decrease_streak += 1;
            if decrease_streak >= SPLICE_DAMPING_RESET_STREAK && damping < 1.0 {
                damping = 1.0;
            }
        } else {
            decrease_streak = 0;
            damping = (damping * SPLICE_DAMPING_REDUCTION).max(SPLICE_DAMPING_MINIMUM);
        }
        last_mismatch = mismatch;
        update_splice_endpoints(&interpolated, &mut endpoints, splice_length, damping);
    }
    Err(SaddleError::Solver(
        "TRICS splicing did not converge".into(),
    ))
}

fn splice_mismatch(paths: &[Array2<f64>], splice_length: usize) -> f64 {
    let mut mismatch = 0.0_f64;
    for pair in 0..paths.len() - 1 {
        let left_end = paths[pair].nrows() - 1;
        let left_start = left_end + 1 - splice_length;
        mismatch = mismatch.max(max_atom_difference(
            &paths[pair].row(left_start).to_owned(),
            &paths[pair + 1].row(0).to_owned(),
        ));
        mismatch = mismatch.max(max_atom_difference(
            &paths[pair].row(left_end).to_owned(),
            &paths[pair + 1].row(splice_length - 1).to_owned(),
        ));
    }
    mismatch
}

fn update_splice_endpoints(
    paths: &[Array2<f64>],
    endpoints: &mut [[Array1<f64>; 2]],
    splice_length: usize,
    damping: f64,
) {
    for pair in 0..paths.len() - 1 {
        let left_end = paths[pair].nrows() - 1;
        let left_start = left_end + 1 - splice_length;
        let right_end = splice_length - 1;
        let new_left_end = &paths[pair + 1].row(right_end) * damping
            + &paths[pair].row(left_end) * (1.0 - damping);
        let new_right_start =
            &paths[pair].row(left_start) * damping + &paths[pair + 1].row(0) * (1.0 - damping);
        endpoints[pair][1].assign(&new_left_end);
        endpoints[pair + 1][0].assign(&new_right_start);
    }
}

fn concatenate_segments(
    n_images: usize,
    segments: &[TricsSegment],
    paths: &[Array2<f64>],
) -> Array2<f64> {
    let dof = paths[0].ncols();
    let mut result = Array2::zeros((n_images, dof));
    let mut segment = 0;
    for image in 0..n_images {
        while segment + 1 < segments.len() {
            let boundary = (segments[segment + 1].start + segments[segment].end) / 2;
            if image <= boundary {
                break;
            }
            segment += 1;
        }
        result
            .row_mut(image)
            .assign(&paths[segment].row(image - segments[segment].start));
    }
    result
}

fn max_atom_difference(left: &Array1<f64>, right: &Array1<f64>) -> f64 {
    left.iter()
        .zip(right)
        .collect::<Vec<_>>()
        .chunks(3)
        .map(|atom| {
            atom.iter()
                .map(|(left, right)| {
                    let difference = *left - *right;
                    difference * difference
                })
                .sum::<f64>()
                .sqrt()
        })
        .fold(0.0_f64, f64::max)
}

fn aligned_max_atom_difference(
    moving: ArrayView1<'_, f64>,
    target: ArrayView1<'_, f64>,
) -> Result<f64, SaddleError> {
    check_same_positions(moving, target)?;
    let n_atoms = moving.len() / 3;
    if n_atoms < 2 {
        return Ok(max_atom_difference(&moving.to_owned(), &target.to_owned()));
    }
    let indices: Vec<_> = (0..n_atoms).collect();
    let reference = fragment_positions(target, &indices);
    let quaternion = Rotation::new(indices, CartAxis::X, reference.clone())?.quaternion(moving)?;
    let [w, x, y, z] = quaternion;
    let rotation = [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ];
    let moving_positions = fragment_positions(moving, &(0..n_atoms).collect::<Vec<_>>());
    let moving_center = [
        moving_positions.column(0).sum() / n_atoms as f64,
        moving_positions.column(1).sum() / n_atoms as f64,
        moving_positions.column(2).sum() / n_atoms as f64,
    ];
    let target_center = [
        reference.column(0).sum() / n_atoms as f64,
        reference.column(1).sum() / n_atoms as f64,
        reference.column(2).sum() / n_atoms as f64,
    ];
    let mut maximum = 0.0_f64;
    for atom_index in 0..n_atoms {
        let centered = [
            moving_positions[(atom_index, 0)] - moving_center[0],
            moving_positions[(atom_index, 1)] - moving_center[1],
            moving_positions[(atom_index, 2)] - moving_center[2],
        ];
        let aligned = [
            dot(rotation[0], centered) + target_center[0],
            dot(rotation[1], centered) + target_center[1],
            dot(rotation[2], centered) + target_center[2],
        ];
        let difference = sub(
            aligned,
            [
                reference[(atom_index, 0)],
                reference[(atom_index, 1)],
                reference[(atom_index, 2)],
            ],
        );
        maximum = maximum.max(norm(difference));
    }
    Ok(maximum)
}

fn check_trics_config(config: &TricsConfig) -> Result<(), SaddleError> {
    check_config(&config.coordinates)?;
    check_inversion_config(&config.inversion)?;
    if config.diagnostic_threshold > 6
        || !(config.splice_tolerance.is_finite() && config.splice_tolerance > 0.0)
        || config.max_splice_iterations == 0
    {
        return Err(SaddleError::Invalid(
            "TRICS needs a diagnostic threshold from zero to six and positive splice controls"
                .into(),
        ));
    }
    if !(config.length_per_angstrom.is_finite() && config.length_per_angstrom > 0.0) {
        return Err(SaddleError::Invalid(
            "TRICS length_per_angstrom must be positive and finite".into(),
        ));
    }
    Ok(())
}

fn check_config(config: &TricConfig) -> Result<(), SaddleError> {
    if !(config.bond_factor.is_finite() && config.bond_factor > 1.0) {
        return Err(SaddleError::Invalid(
            "TRIC bond factor must be finite and greater than one".into(),
        ));
    }
    if !(config.rank_tolerance.is_finite()
        && config.rank_tolerance > 0.0
        && config.rank_tolerance < 1.0)
    {
        return Err(SaddleError::Invalid(
            "TRIC rank tolerance must lie between zero and one".into(),
        ));
    }
    Ok(())
}

fn check_inversion_config(config: &TricInversionConfig) -> Result<(), SaddleError> {
    if config.max_iterations == 0
        || !(config.tolerance.is_finite() && config.tolerance > 0.0)
        || !(config.max_atom_move.is_finite() && config.max_atom_move > 0.0)
    {
        return Err(SaddleError::Invalid(
            "TRIC inversion needs positive finite controls".into(),
        ));
    }
    Ok(())
}

fn check_positions(positions: ArrayView1<'_, f64>) -> Result<(), SaddleError> {
    if positions.is_empty() || !positions.len().is_multiple_of(3) {
        return Err(SaddleError::Shape(
            "TRIC coordinates must be a nonempty 3N vector".into(),
        ));
    }
    if positions.iter().any(|value| !value.is_finite()) {
        return Err(SaddleError::NonFinite("TRIC coordinates"));
    }
    Ok(())
}

fn check_same_positions(
    left: ArrayView1<'_, f64>,
    right: ArrayView1<'_, f64>,
) -> Result<(), SaddleError> {
    check_positions(left)?;
    check_positions(right)?;
    if left.len() != right.len() {
        return Err(SaddleError::Shape(
            "TRIC geometries have different dimensions".into(),
        ));
    }
    Ok(())
}

fn atom(positions: ArrayView1<'_, f64>, index: usize) -> Result<[f64; 3], SaddleError> {
    let start = 3 * index;
    if start + 2 >= positions.len() {
        return Err(SaddleError::Shape(
            "TRIC atom index exceeds the geometry".into(),
        ));
    }
    Ok([positions[start], positions[start + 1], positions[start + 2]])
}

fn sub(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn norm(vector: [f64; 3]) -> f64 {
    dot(vector, vector).sqrt()
}

fn scale(vector: [f64; 3], factor: f64) -> [f64; 3] {
    [vector[0] * factor, vector[1] * factor, vector[2] * factor]
}

fn distance(positions: ArrayView1<'_, f64>, left: usize, right: usize) -> Result<f64, SaddleError> {
    let length = norm(sub(atom(positions, left)?, atom(positions, right)?));
    if length > VECTOR_TOLERANCE {
        Ok(length)
    } else {
        Err(SaddleError::Invalid(
            "TRIC distance has coincident atoms".into(),
        ))
    }
}

fn distance_derivative(
    positions: ArrayView1<'_, f64>,
    left: usize,
    right: usize,
) -> Result<Array1<f64>, SaddleError> {
    let vector = sub(atom(positions, left)?, atom(positions, right)?);
    let length = norm(vector);
    if length <= VECTOR_TOLERANCE {
        return Err(SaddleError::Invalid(
            "TRIC distance has coincident atoms".into(),
        ));
    }
    let unit = scale(vector, 1.0 / length);
    let mut derivative = Array1::zeros(positions.len());
    for axis in 0..3 {
        derivative[3 * left + axis] = unit[axis];
        derivative[3 * right + axis] = -unit[axis];
    }
    Ok(derivative)
}

fn angle(
    positions: ArrayView1<'_, f64>,
    left: usize,
    center: usize,
    right: usize,
) -> Result<f64, SaddleError> {
    let u = sub(atom(positions, left)?, atom(positions, center)?);
    let v = sub(atom(positions, right)?, atom(positions, center)?);
    let denominator = norm(u) * norm(v);
    if denominator <= VECTOR_TOLERANCE {
        return Err(SaddleError::Invalid(
            "TRIC angle has a zero length leg".into(),
        ));
    }
    let raw_cosine = dot(u, v) / denominator;
    let cosine = if (-1.0..=1.0).contains(&raw_cosine) {
        raw_cosine
    } else {
        raw_cosine.signum()
    };
    Ok(cosine.acos())
}

fn angle_derivative(
    positions: ArrayView1<'_, f64>,
    left: usize,
    center: usize,
    right: usize,
) -> Result<Array1<f64>, SaddleError> {
    let u = sub(atom(positions, left)?, atom(positions, center)?);
    let v = sub(atom(positions, right)?, atom(positions, center)?);
    let (nu, nv) = (norm(u), norm(v));
    let cosine = dot(u, v) / (nu * nv);
    let sine = (1.0 - cosine * cosine).max(0.0).sqrt();
    if nu <= VECTOR_TOLERANCE || nv <= VECTOR_TOLERANCE || sine <= VECTOR_TOLERANCE {
        return Err(SaddleError::Invalid(
            "TRIC angle derivative is singular".into(),
        ));
    }
    let left_gradient = scale(
        sub(scale(u, cosine / nu), scale(v, 1.0 / nv)),
        1.0 / (nu * sine),
    );
    let right_gradient = scale(
        sub(scale(v, cosine / nv), scale(u, 1.0 / nu)),
        1.0 / (nv * sine),
    );
    let center_gradient = scale(
        [
            left_gradient[0] + right_gradient[0],
            left_gradient[1] + right_gradient[1],
            left_gradient[2] + right_gradient[2],
        ],
        -1.0,
    );
    let mut derivative = Array1::zeros(positions.len());
    for axis in 0..3 {
        derivative[3 * left + axis] = left_gradient[axis];
        derivative[3 * center + axis] = center_gradient[axis];
        derivative[3 * right + axis] = right_gradient[axis];
    }
    Ok(derivative)
}

fn angular_diagnostic(positions: ArrayView1<'_, f64>, angles: &[(usize, usize, usize)]) -> u8 {
    let mut result = 0;
    for &(left, center, right) in angles {
        let Ok(value) = angle(positions, left, center, right) else {
            return 3;
        };
        let cosine = value.cos().abs();
        let level = if cosine > ANGLE_FAILURE_COSINE {
            3
        } else if cosine > ANGLE_UNSTABLE_COSINE {
            2
        } else if cosine >= ANGLE_WARNING_COSINE {
            1
        } else {
            0
        };
        result = result.max(level);
    }
    result
}

fn linear_bend(
    positions: ArrayView1<'_, f64>,
    left: usize,
    center: usize,
    right: usize,
    direction: [f64; 3],
) -> Result<f64, SaddleError> {
    let u = sub(atom(positions, left)?, atom(positions, center)?);
    let v = sub(atom(positions, right)?, atom(positions, center)?);
    let (nu, nv) = (norm(u), norm(v));
    if nu <= VECTOR_TOLERANCE || nv <= VECTOR_TOLERANCE {
        return Err(SaddleError::Invalid(
            "TRIC linear bend has a zero length leg".into(),
        ));
    }
    Ok(dot(
        [
            u[0] / nu + v[0] / nv,
            u[1] / nu + v[1] / nv,
            u[2] / nu + v[2] / nv,
        ],
        direction,
    ))
}

fn linear_angular_diagnostic(
    positions: ArrayView1<'_, f64>,
    left: usize,
    center: usize,
    right: usize,
) -> u8 {
    let Ok(value) = angle(positions, left, center, right) else {
        return 3;
    };
    let cosine = value.cos().abs();
    if cosine < 0.4 {
        3
    } else if cosine < 0.6 {
        2
    } else if cosine < 0.8 {
        1
    } else {
        0
    }
}

fn perpendicular_directions(axis: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let seed = if axis[0].abs() <= axis[1].abs() && axis[0].abs() <= axis[2].abs() {
        [1.0, 0.0, 0.0]
    } else if axis[1].abs() <= axis[2].abs() {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    let first_raw = cross(axis, seed);
    let first = scale(first_raw, 1.0 / norm(first_raw));
    let second_raw = cross(axis, first);
    let second = scale(second_raw, 1.0 / norm(second_raw));
    (first, second)
}

fn dihedral(
    positions: ArrayView1<'_, f64>,
    a: usize,
    b: usize,
    c: usize,
    d: usize,
) -> Result<f64, SaddleError> {
    let b0 = sub(atom(positions, a)?, atom(positions, b)?);
    let b1 = sub(atom(positions, c)?, atom(positions, b)?);
    let b2 = sub(atom(positions, d)?, atom(positions, c)?);
    let b1_norm = norm(b1);
    if b1_norm <= VECTOR_TOLERANCE {
        return Err(SaddleError::Invalid(
            "TRIC dihedral has a zero central bond".into(),
        ));
    }
    let axis = scale(b1, 1.0 / b1_norm);
    let v = sub(b0, scale(axis, dot(b0, axis)));
    let w = sub(b2, scale(axis, dot(b2, axis)));
    if norm(v) <= VECTOR_TOLERANCE || norm(w) <= VECTOR_TOLERANCE {
        return Err(SaddleError::Invalid(
            "TRIC dihedral is undefined at a linear angle".into(),
        ));
    }
    Ok(dot(cross(axis, v), w).atan2(dot(v, w)))
}

fn out_of_plane(
    positions: ArrayView1<'_, f64>,
    a: usize,
    b: usize,
    c: usize,
    d: usize,
) -> Result<f64, SaddleError> {
    let first = sub(atom(positions, b)?, atom(positions, a)?);
    let middle = sub(atom(positions, c)?, atom(positions, b)?);
    let last = sub(atom(positions, d)?, atom(positions, c)?);
    let normal_last = cross(middle, last);
    let normal_first = cross(first, middle);
    if norm(middle) <= VECTOR_TOLERANCE
        || norm(normal_last) <= VECTOR_TOLERANCE
        || norm(normal_first) <= VECTOR_TOLERANCE
    {
        return Err(SaddleError::Invalid(
            "TRIC out of plane coordinate is singular".into(),
        ));
    }
    let numerator = dot(first, normal_last) * norm(middle);
    let denominator = dot(normal_last, normal_first);
    Ok(numerator.atan2(denominator))
}

fn out_of_plane_diagnostic(
    positions: ArrayView1<'_, f64>,
    a: usize,
    b: usize,
    c: usize,
    d: usize,
) -> u8 {
    let first = match angle(positions, a, b, c) {
        Ok(value) => value.cos().abs(),
        Err(_) => f64::INFINITY,
    };
    let second = match angle(positions, b, c, d) {
        Ok(value) => value.cos().abs(),
        Err(_) => f64::INFINITY,
    };
    let level = |cosine: f64| {
        if cosine > 0.99 {
            3
        } else if cosine > 0.98 {
            2
        } else if cosine > 0.95 {
            1
        } else {
            0
        }
    };
    let left = level(first);
    let right = level(second);
    left + right
}

fn angle_normal(
    positions: ArrayView1<'_, f64>,
    a: usize,
    b: usize,
    c: usize,
) -> Result<[f64; 3], SaddleError> {
    let normal = cross(
        sub(atom(positions, a)?, atom(positions, b)?),
        sub(atom(positions, c)?, atom(positions, b)?),
    );
    let length = norm(normal);
    if length <= VECTOR_TOLERANCE {
        return Err(SaddleError::Invalid("TRIC angle normal is singular".into()));
    }
    Ok(scale(normal, 1.0 / length))
}

fn unique_atoms(indices: &[usize]) -> Vec<usize> {
    let mut atoms = Vec::with_capacity(indices.len());
    for &index in indices {
        if !atoms.contains(&index) {
            atoms.push(index);
        }
    }
    atoms
}

/// Finite difference step as a fraction of the largest separation among
/// the atoms a primitive moves, so it follows the size of the molecule in
/// whatever length unit the coordinates use.
fn difference_step(positions: ArrayView1<'_, f64>, atoms: &[usize]) -> f64 {
    let mut extent = 0.0_f64;
    for (position, &first) in atoms.iter().enumerate() {
        for &second in &atoms[position + 1..] {
            let separation: f64 = (0..3)
                .map(|axis| {
                    let difference = positions[3 * first + axis] - positions[3 * second + axis];
                    difference * difference
                })
                .sum::<f64>()
                .sqrt();
            extent = extent.max(separation);
        }
    }
    if extent.is_finite() && extent > 0.0 {
        DIFFERENCE_STEP * extent
    } else {
        DIFFERENCE_STEP
    }
}

fn central_derivative(
    coordinate: &TricPrimitive,
    positions: ArrayView1<'_, f64>,
) -> Result<Array1<f64>, SaddleError> {
    let mut derivative = Array1::zeros(positions.len());
    let mut trial = positions.to_owned();
    let atoms = match *coordinate {
        TricPrimitive::LinearBend(a, b, c, _) => unique_atoms(&[a, b, c]),
        TricPrimitive::Dihedral(a, b, c, d) | TricPrimitive::OutOfPlane(a, b, c, d) => {
            unique_atoms(&[a, b, c, d])
        }
        _ => (0..positions.len() / 3).collect(),
    };
    let n_atoms = positions.len() / 3;
    if let Some(atom_index) = atoms.iter().find(|&&index| index >= n_atoms) {
        return Err(SaddleError::Shape(format!(
            "atom index {atom_index} out of bounds for {n_atoms} atoms"
        )));
    }
    let step = difference_step(positions, &atoms);
    for atom_index in atoms {
        for axis in 0..3 {
            let slot = 3 * atom_index + axis;
            trial[slot] = positions[slot] + step;
            let plus = coordinate.value(trial.view())?;
            trial[slot] = positions[slot] - step;
            let minus = coordinate.value(trial.view())?;
            trial[slot] = positions[slot];
            let mut difference = plus - minus;
            if matches!(
                coordinate,
                TricPrimitive::Dihedral(..) | TricPrimitive::OutOfPlane(..)
            ) {
                if difference > std::f64::consts::PI {
                    difference -= std::f64::consts::TAU;
                } else if difference < -std::f64::consts::PI {
                    difference += std::f64::consts::TAU;
                }
            }
            derivative[slot] = difference / (2.0 * step);
        }
    }
    Ok(derivative)
}

fn primitive_wilson_b(
    primitives: &[TricPrimitive],
    positions: ArrayView1<'_, f64>,
) -> Result<Array2<f64>, SaddleError> {
    let mut matrix = Array2::zeros((primitives.len(), positions.len()));
    for (row, primitive) in primitives.iter().enumerate() {
        matrix
            .row_mut(row)
            .assign(&primitive.derivative(positions)?);
    }
    Ok(matrix)
}

fn solve_metric(
    matrix: &Array2<f64>,
    right: &Array1<f64>,
    tolerance: f64,
) -> Result<Array1<f64>, SaddleError> {
    let n = matrix.nrows();
    if matrix.ncols() != n || right.len() != n {
        return Err(SaddleError::Shape(
            "TRIC linear solve dimensions differ".into(),
        ));
    }
    let (values, vectors) = crate::eigensolve::exact_eigh(matrix.view())?;
    let mut solution = Array1::zeros(n);
    for (mode, &value) in values.iter().enumerate() {
        if value.abs() <= tolerance {
            continue;
        }
        let vector = vectors.column(mode);
        let coefficient = vector.dot(right) / value;
        solution += &(&vector * coefficient);
    }
    Ok(solution)
}

fn cap_atom_step(mut step: Array1<f64>, max_move: f64) -> Array1<f64> {
    let largest = max_atom_norm(&step);
    if largest > max_move {
        step *= max_move / largest;
    }
    step
}

fn max_atom_norm(step: &Array1<f64>) -> f64 {
    step.as_slice().map_or(0.0, |values| {
        values
            .chunks(3)
            .map(|atom| atom.iter().map(|value| value * value).sum::<f64>().sqrt())
            .fold(0.0_f64, f64::max)
    })
}

fn validate_bonds(
    bonds: &[(usize, usize)],
    n_atoms: usize,
) -> Result<Vec<(usize, usize)>, SaddleError> {
    let mut checked = Vec::with_capacity(bonds.len());
    for &(left, right) in bonds {
        if left == right || left >= n_atoms || right >= n_atoms {
            return Err(SaddleError::Shape(
                "TRIC bond contains an invalid atom pair".into(),
            ));
        }
        let pair = (left.min(right), left.max(right));
        if !checked.contains(&pair) {
            checked.push(pair);
        }
    }
    checked.sort_unstable();
    Ok(checked)
}

fn infer_bonds(
    positions: ArrayView1<'_, f64>,
    factor: f64,
) -> Result<Vec<(usize, usize)>, SaddleError> {
    let n_atoms = positions.len() / 3;
    if n_atoms < 2 {
        return Ok(Vec::new());
    }
    let mut distances = Array2::from_elem((n_atoms, n_atoms), f64::INFINITY);
    let mut nearest = vec![f64::INFINITY; n_atoms];
    for left in 0..n_atoms {
        for right in left + 1..n_atoms {
            let separation = distance(positions, left, right)?;
            distances[(left, right)] = separation;
            distances[(right, left)] = separation;
            nearest[left] = nearest[left].min(separation);
            nearest[right] = nearest[right].min(separation);
        }
    }
    let mut scale = nearest.clone();
    scale.sort_by(|left, right| match left.partial_cmp(right) {
        Some(ordering) => ordering,
        None => Ordering::Equal,
    });
    let median = scale[scale.len() / 2];
    let global_limit = factor * median;
    let mut bonds = Vec::new();
    for left in 0..n_atoms {
        for right in left + 1..n_atoms {
            let local_limit = factor * (nearest[left] + nearest[right]) * 0.5;
            if distances[(left, right)] <= local_limit.min(global_limit) {
                bonds.push((left, right));
            }
        }
    }
    Ok(bonds)
}

fn fragments(n_atoms: usize, bonds: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let mut adjacency = vec![Vec::new(); n_atoms];
    for &(left, right) in bonds {
        adjacency[left].push(right);
        adjacency[right].push(left);
    }
    let mut seen = vec![false; n_atoms];
    let mut result = Vec::new();
    for start in 0..n_atoms {
        if seen[start] {
            continue;
        }
        let mut stack = vec![start];
        let mut fragment = Vec::new();
        seen[start] = true;
        while let Some(atom) = stack.pop() {
            fragment.push(atom);
            for &next in &adjacency[atom] {
                if !seen[next] {
                    seen[next] = true;
                    stack.push(next);
                }
            }
        }
        fragment.sort_unstable();
        result.push(fragment);
    }
    result
}

fn molecular_primitives(
    bonds: &[(usize, usize)],
    n_atoms: usize,
    reference: ArrayView1<'_, f64>,
) -> Result<Vec<TricPrimitive>, SaddleError> {
    let mut adjacency = vec![Vec::new(); n_atoms];
    let mut primitives = Vec::new();
    for &(left, right) in bonds {
        adjacency[left].push(right);
        adjacency[right].push(left);
        primitives.push(TricPrimitive::Distance(left, right));
    }
    for neighbors in &mut adjacency {
        neighbors.sort_unstable();
    }
    for (center, neighbors) in adjacency.iter().enumerate() {
        for first in 0..neighbors.len() {
            for second in first + 1..neighbors.len() {
                let left = neighbors[first];
                let right = neighbors[second];
                let value = angle(reference, left, center, right)?;
                if value.cos().abs() >= LINEAR_ANGLE_COSINE {
                    let bond = sub(atom(reference, left)?, atom(reference, center)?);
                    let axis = scale(bond, 1.0 / norm(bond));
                    let (first_direction, second_direction) = perpendicular_directions(axis);
                    primitives.push(TricPrimitive::LinearBend(
                        left,
                        center,
                        right,
                        first_direction,
                    ));
                    primitives.push(TricPrimitive::LinearBend(
                        left,
                        center,
                        right,
                        second_direction,
                    ));
                } else {
                    primitives.push(TricPrimitive::Angle(left, center, right));
                }
            }
        }
        if neighbors.len() == 3 {
            let permutations = [
                [neighbors[0], neighbors[1], neighbors[2]],
                [neighbors[0], neighbors[2], neighbors[1]],
                [neighbors[1], neighbors[0], neighbors[2]],
                [neighbors[1], neighbors[2], neighbors[0]],
                [neighbors[2], neighbors[0], neighbors[1]],
                [neighbors[2], neighbors[1], neighbors[0]],
            ];
            let mut candidates = Vec::new();
            for [b, c, d] in permutations {
                let first_cosine = angle(reference, center, b, c)?.cos().abs();
                let second_cosine = angle(reference, b, c, d)?.cos().abs();
                if first_cosine > OUT_OF_PLANE_COSINE || second_cosine > OUT_OF_PLANE_COSINE {
                    continue;
                }
                let normals = angle_normal(reference, center, b, c).and_then(|first| {
                    angle_normal(reference, b, c, d).map(|second| (first, second))
                });
                if let Ok((first, second)) = normals
                    && dot(first, second).abs() >= OUT_OF_PLANE_COSINE
                {
                    let span = distance(reference, center, b)?.max(distance(reference, center, c)?);
                    candidates.push((b, c, d, span));
                }
            }
            if let Some(maximum) = candidates
                .iter()
                .map(|candidate| candidate.3)
                .max_by(f64::total_cmp)
                && let Some(&(b, c, d, _)) = candidates
                    .iter()
                    .find(|candidate| candidate.3 >= 0.8 * maximum)
            {
                primitives.retain(|primitive| {
                    !matches!(
                        primitive,
                        TricPrimitive::Angle(left, middle, right)
                            if *middle == center
                                && ((*left == b && *right == c)
                                    || (*left == c && *right == b))
                    )
                });
                primitives.push(TricPrimitive::OutOfPlane(center, b, c, d));
            }
        }
    }
    let mut dihedrals = Vec::new();
    for &(b, c) in bonds {
        for &a in &adjacency[b] {
            if a == c {
                continue;
            }
            for &d in &adjacency[c] {
                if d == b || d == a {
                    continue;
                }
                let forward = (a, b, c, d);
                let reverse = (d, c, b, a);
                let canonical = if forward <= reverse { forward } else { reverse };
                if !dihedrals.contains(&canonical) {
                    dihedrals.push(canonical);
                }
            }
        }
    }
    dihedrals.sort_unstable();
    primitives.extend(
        dihedrals
            .into_iter()
            .map(|(a, b, c, d)| TricPrimitive::Dihedral(a, b, c, d)),
    );
    Ok(primitives)
}

fn fragment_positions(positions: ArrayView1<'_, f64>, fragment: &[usize]) -> Array2<f64> {
    Array2::from_shape_fn((fragment.len(), 3), |(row, axis)| {
        positions[3 * fragment[row] + axis]
    })
}

fn radius_of_gyration(positions: &Array2<f64>) -> f64 {
    let count = positions.nrows() as f64;
    let centroid = [
        positions.column(0).sum() / count,
        positions.column(1).sum() / count,
        positions.column(2).sum() / count,
    ];
    let squared = positions
        .outer_iter()
        .map(|row| {
            (row[0] - centroid[0]) * (row[0] - centroid[0])
                + (row[1] - centroid[1]) * (row[1] - centroid[1])
                + (row[2] - centroid[2]) * (row[2] - centroid[2])
        })
        .sum::<f64>();
    (squared / count).sqrt()
}

fn nonredundant_basis(metric: &Array2<f64>, tolerance: f64) -> Result<Array2<f64>, SaddleError> {
    let (values, vectors) = crate::eigensolve::exact_eigh(metric.view())?;
    let mut indices: Vec<usize> = (0..values.len())
        .filter(|&index| values[index].abs() > tolerance)
        .collect();
    indices.sort_by(
        |&left, &right| match values[right].partial_cmp(&values[left]) {
            Some(ordering) => ordering,
            None => Ordering::Equal,
        },
    );
    Ok(Array2::from_shape_fn(
        (metric.nrows(), indices.len()),
        |(row, column)| vectors[(row, indices[column])],
    ))
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    use super::*;

    const TEST_STEP: f64 = 0.000_000_1;
    const TEST_TOLERANCE: f64 = 0.000_02;

    #[test]
    fn primitive_wilson_rows_match_finite_differences() {
        let positions = array![0.0, 0.0, 0.0, 1.2, 0.1, 0.0, 2.0, 0.9, 0.2, 2.8, 1.0, 1.1];
        let primitives = [
            TricPrimitive::Distance(0, 1),
            TricPrimitive::Angle(0, 1, 2),
            TricPrimitive::Dihedral(0, 1, 2, 3),
            TricPrimitive::OutOfPlane(0, 1, 2, 3),
        ];
        for primitive in primitives {
            let derivative = primitive.derivative(positions.view()).unwrap();
            for slot in 0..positions.len() {
                let mut plus = positions.clone();
                let mut minus = positions.clone();
                plus[slot] += TEST_STEP;
                minus[slot] -= TEST_STEP;
                let numerical =
                    primitive.difference(plus.view(), minus.view()).unwrap() / (2.0 * TEST_STEP);
                assert_abs_diff_eq!(derivative[slot], numerical, epsilon = TEST_TOLERANCE);
            }
        }
    }

    #[test]
    fn a_tric_basis_is_nonredundant() {
        let water = array![0.0, 0.0, 0.0, 0.96, 0.0, 0.0, -0.24, 0.93, 0.0];
        let config = TricConfig::default().with_bonds(Some(vec![(0, 1), (0, 2)]));
        let system = TricSystem::new(water.view(), &config).unwrap();
        assert_eq!(system.dimension(), water.len());
        let b = system.wilson_b(water.view()).unwrap();
        assert_eq!(b.dim(), (water.len(), water.len()));
    }

    #[test]
    fn inferred_bonds_keep_a_chain_sparse() {
        let chain = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.2, 0.0];
        let bonds = infer_bonds(chain.view(), DEFAULT_BOND_FACTOR).unwrap();
        assert_eq!(bonds, vec![(0, 1), (1, 2)]);
    }

    #[test]
    fn diagnostic_matrix_flags_a_collinear_angle() {
        let bent = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.7, 0.7, 0.0];
        let linear = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let path = ndarray::stack(ndarray::Axis(0), &[bent.view(), linear.view()]).unwrap();
        let config = TricConfig::default().with_bonds(Some(vec![(0, 1), (1, 2)]));
        let systems = tric_systems(path.view(), &config).unwrap();
        let diagnostic = diagnostic_matrix(&systems, path.view()).unwrap();
        assert_eq!(diagnostic[(0, 0)], 0);
        assert_eq!(diagnostic[(0, 1)], 3);
    }

    #[test]
    fn diagnostics_follow_the_reference_boundaries() {
        let geometry = |degrees: f64| {
            let radians = degrees.to_radians();
            array![
                radians.cos(),
                radians.sin(),
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                1.0 - radians.cos(),
                radians.sin(),
                0.1
            ]
        };
        let dihedral = TricPrimitive::Dihedral(0, 1, 2, 3);
        assert_eq!(dihedral.diagnostic(geometry(154.9).view()), 0);
        assert_eq!(dihedral.diagnostic(geometry(155.1).view()), 1);
        assert_eq!(dihedral.diagnostic(geometry(165.1).view()), 2);
        assert_eq!(dihedral.diagnostic(geometry(175.1).view()), 3);

        let linear = TricPrimitive::LinearBend(0, 1, 2, [0.0, 1.0, 0.0]);
        let linear_geometry = |cosine: f64| {
            let sine = (1.0 - cosine * cosine).sqrt();
            array![cosine, sine, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        };
        assert_eq!(linear.diagnostic(linear_geometry(0.81).view()), 0);
        assert_eq!(linear.diagnostic(linear_geometry(0.79).view()), 1);
        assert_eq!(linear.diagnostic(linear_geometry(0.59).view()), 2);
        assert_eq!(linear.diagnostic(linear_geometry(0.39).view()), 3);

        let out_of_plane = TricPrimitive::OutOfPlane(0, 1, 2, 3);
        let out_of_plane_geometry = |first: f64, second: f64| {
            let first = first.to_radians();
            let second = second.to_radians();
            array![
                first.cos(),
                first.sin(),
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                1.0 - second.cos(),
                second.sin(),
                0.0
            ]
        };
        assert_eq!(
            out_of_plane.diagnostic(out_of_plane_geometry(20.0, 90.0).view()),
            0
        );
        assert_eq!(
            out_of_plane.diagnostic(out_of_plane_geometry(16.0, 90.0).view()),
            1
        );
        assert_eq!(
            out_of_plane.diagnostic(out_of_plane_geometry(10.0, 90.0).view()),
            2
        );
        assert_eq!(
            out_of_plane.diagnostic(out_of_plane_geometry(5.0, 90.0).view()),
            3
        );
        assert_eq!(
            out_of_plane.diagnostic(out_of_plane_geometry(5.0, 5.0).view()),
            6
        );
        assert_eq!(
            out_of_plane.diagnostic(out_of_plane_geometry(16.0, 10.0).view()),
            3
        );
        assert_eq!(
            out_of_plane.diagnostic(out_of_plane_geometry(16.0, 5.0).view()),
            4
        );
    }

    #[test]
    fn out_of_plane_replaces_one_trigonal_angle() {
        let trigonal = array![
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            -0.5,
            0.866_025_403_784,
            0.0,
            -0.5,
            -0.866_025_403_784,
            0.0
        ];
        let primitives =
            molecular_primitives(&[(0, 1), (0, 2), (0, 3)], 4, trigonal.view()).unwrap();
        assert_eq!(
            primitives
                .iter()
                .filter(|primitive| matches!(primitive, TricPrimitive::Angle(..)))
                .count(),
            2
        );
        assert_eq!(
            primitives
                .iter()
                .filter(|primitive| matches!(primitive, TricPrimitive::OutOfPlane(..)))
                .count(),
            1
        );
    }

    #[test]
    fn maze_segments_receive_the_reference_overlap() {
        let segments = fill_splice_segments(&[(0, 4), (5, 9), (10, 14)], 2, 15);
        assert_eq!(segments, vec![(0, 5), (4, 10), (9, 14)]);
        assert_eq!(segments[0].1 - segments[1].0 + 1, 2);
        assert_eq!(segments[1].1 - segments[2].0 + 1, 2);
    }

    #[test]
    fn splice_endpoint_update_uses_opposite_overlap_ends() {
        let left = array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0]
        ];
        let right = array![
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
            [12.0, 0.0, 0.0],
            [13.0, 0.0, 0.0],
            [14.0, 0.0, 0.0]
        ];
        let mut endpoints = vec![
            [left.row(0).to_owned(), left.row(4).to_owned()],
            [right.row(0).to_owned(), right.row(4).to_owned()],
        ];
        update_splice_endpoints(&[left, right], &mut endpoints, 2, 1.0);
        assert_eq!(endpoints[0][1], array![11.0, 0.0, 0.0]);
        assert_eq!(endpoints[1][0], array![3.0, 0.0, 0.0]);
    }

    #[test]
    fn aligned_endpoint_error_removes_rigid_motion() {
        let target = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let moving = array![2.0, -1.0, 0.5, 2.0, 0.0, 0.5, 1.0, -1.0, 0.5];
        assert!(
            aligned_max_atom_difference(moving.view(), target.view()).unwrap() < TEST_TOLERANCE
        );
    }

    #[test]
    fn back_transformation_round_trips_an_internal_step() {
        let start = array![0.0, 0.0, 0.0, 0.96, 0.0, 0.0, -0.24, 0.93, 0.0];
        let target = array![0.1, -0.05, 0.02, 1.02, 0.08, -0.03, -0.18, 0.98, 0.06];
        let config = TricConfig::default().with_bonds(Some(vec![(0, 1), (0, 2)]));
        let system = TricSystem::new(start.view(), &config).unwrap();
        let displacement = system.difference(target.view(), start.view()).unwrap();
        let recovered = system
            .back_transform(
                start.view(),
                displacement.view(),
                &TricInversionConfig::default(),
            )
            .unwrap();
        let error = &recovered - &target;
        assert!(error.dot(&error).sqrt() < TEST_TOLERANCE);
    }

    #[test]
    fn segment_interpolation_keeps_exact_endpoints() {
        let left = array![0.0, 0.0, 0.0, 0.96, 0.0, 0.0, -0.24, 0.93, 0.0];
        let right = array![0.2, 0.0, 0.0, 1.18, 0.1, 0.0, -0.1, 1.0, 0.1];
        let config = TricConfig::default().with_bonds(Some(vec![(0, 1), (0, 2)]));
        let system = TricSystem::new(left.view(), &config).unwrap();
        let path = interpolate_tric_segment(
            &system,
            left.view(),
            right.view(),
            5,
            &TricInversionConfig::default(),
        )
        .unwrap();
        assert_eq!(path.row(0), left);
        assert_eq!(path.row(4), right);
        assert!(path.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn one_stable_system_skips_splicing() {
        let left = array![0.0, 0.0, 0.0, 0.96, 0.0, 0.0, -0.24, 0.93, 0.0];
        let right = array![0.2, 0.0, 0.0, 1.18, 0.1, 0.0, -0.1, 1.0, 0.1];
        let mut initial = Array2::zeros((5, left.len()));
        for image in 0..5 {
            let fraction = image as f64 / 4.0;
            initial
                .row_mut(image)
                .assign(&(&left * (1.0 - fraction) + &right * fraction));
        }
        let config = TricsConfig::default()
            .with_coordinates(TricConfig::default().with_bonds(Some(vec![(0, 1), (0, 2)])));
        let report = trics_path(initial.view(), &config).unwrap();
        assert_eq!(report.segments.len(), 1);
        assert_eq!(report.splice_iterations, 0);
        let direct = interpolate_tric_segment(
            &TricSystem::new(left.view(), &config.coordinates).unwrap(),
            left.view(),
            right.view(),
            5,
            &config.inversion,
        )
        .unwrap();
        assert!(
            report
                .path
                .iter()
                .zip(&direct)
                .all(|(left, right)| (left - right).abs() <= TEST_TOLERANCE)
        );
    }

    #[test]
    fn splice_converges_across_a_linear_bend() {
        let angles = [
            60.0_f64, 55.0, 50.0, 45.0, 15.0, 10.0, 5.0, 0.0, -5.0, -10.0, -15.0, -45.0, -50.0,
            -55.0, -60.0,
        ];
        let mut initial = Array2::zeros((angles.len(), 9));
        for (image, degrees) in angles.into_iter().enumerate() {
            let radians = degrees.to_radians();
            initial.row_mut(image).assign(&array![
                -1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                radians.cos(),
                radians.sin(),
                0.0
            ]);
        }
        let config = TricsConfig::default()
            .with_coordinates(TricConfig::default().with_bonds(Some(vec![(0, 1), (1, 2)])))
            .with_inversion(
                TricInversionConfig::default()
                    .with_max_iterations(200)
                    .with_tolerance(0.000_001)
                    .with_max_atom_move(0.05),
            )
            .with_diagnostic_threshold(1)
            .with_splice_tolerance(0.01);
        let report = trics_path(initial.view(), &config).unwrap();
        assert!(report.segments.len() > 1);
        assert!(report.splice_iterations > 0);
        assert!(report.path.iter().all(|value| value.is_finite()));
        for pair in report.path.outer_iter().collect::<Vec<_>>().windows(2) {
            let step = &pair[1] - &pair[0];
            assert!(step.dot(&step).sqrt() < 1.0);
        }
    }

    fn water_path() -> (Array2<f64>, TricsConfig) {
        let left = array![0.0, 0.0, 0.0, 0.96, 0.0, 0.0, -0.24, 0.93, 0.0];
        let right = array![0.2, 0.0, 0.0, 1.18, 0.1, 0.0, -0.1, 1.0, 0.1];
        let mut initial = Array2::zeros((5, left.len()));
        for image in 0..5 {
            let fraction = image as f64 / 4.0;
            initial
                .row_mut(image)
                .assign(&(&left * (1.0 - fraction) + &right * fraction));
        }
        let config = TricsConfig::default()
            .with_coordinates(TricConfig::default().with_bonds(Some(vec![(0, 1), (0, 2)])));
        (initial, config)
    }

    /// The bends sit away from the 5 and 15 degree diagnostic boundaries, so
    /// a change of unit that moves a cosine by rounding cannot move a level.
    fn linear_bend_path() -> (Array2<f64>, TricsConfig) {
        let angles = [
            60.0_f64, 55.0, 50.0, 45.0, 16.0, 10.0, 6.5, 0.0, -6.5, -10.0, -16.0, -45.0, -50.0,
            -55.0, -60.0,
        ];
        let mut initial = Array2::zeros((angles.len(), 9));
        for (image, degrees) in angles.into_iter().enumerate() {
            let radians = degrees.to_radians();
            initial.row_mut(image).assign(&array![
                -1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                radians.cos(),
                radians.sin(),
                0.0
            ]);
        }
        let config = TricsConfig::default()
            .with_coordinates(TricConfig::default().with_bonds(Some(vec![(0, 1), (1, 2)])))
            .with_inversion(
                TricInversionConfig::default()
                    .with_max_iterations(200)
                    .with_tolerance(0.000_001)
                    .with_max_atom_move(0.05),
            )
            .with_diagnostic_threshold(1)
            .with_splice_tolerance(0.01);
        (initial, config)
    }

    #[test]
    fn the_same_path_in_other_length_units_gives_the_same_geometry() {
        const BOHR_PER_ANGSTROM: f64 = 1.889_726_124_565_062;
        for (initial, config) in [water_path(), linear_bend_path()] {
            let reference = trics_path(initial.view(), &config).unwrap();
            assert!(initial.nrows() < 6 || reference.segments.len() > 1);
            for unit in [0.1, BOHR_PER_ANGSTROM, 1.0e-10, 1.0e3] {
                let scaled = initial.mapv(|value| value * unit);
                let report = trics_path(
                    scaled.view(),
                    &config.clone().with_length_per_angstrom(unit),
                )
                .unwrap();
                assert_eq!(report.segments, reference.segments, "unit {unit}");
                assert_eq!(
                    report.splice_iterations, reference.splice_iterations,
                    "unit {unit}"
                );
                assert_eq!(report.path.row(0), scaled.row(0));
                let last = scaled.nrows() - 1;
                assert_eq!(report.path.row(last), scaled.row(last));
                let largest = report
                    .path
                    .iter()
                    .zip(&reference.path)
                    .map(|(got, want)| (got - want * unit).abs())
                    .fold(0.0_f64, f64::max);
                assert!(
                    largest <= 1.0e-9 * unit,
                    "unit {unit}: path differs by {largest}"
                );
            }
        }
    }

    #[test]
    fn the_length_unit_must_be_positive_and_finite() {
        let (initial, config) = water_path();
        for unit in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                trics_path(
                    initial.view(),
                    &config.clone().with_length_per_angstrom(unit)
                ),
                Err(SaddleError::Invalid(_))
            ));
        }
    }

    #[test]
    fn the_difference_step_follows_the_size_of_the_atoms() {
        let rows = [
            [0.0, 0.0, 0.0],
            [1.4, 0.1, 0.0],
            [2.1, 1.3, 0.2],
            [3.0, 1.2, 1.4],
        ];
        let primitive = TricPrimitive::Dihedral(0, 1, 2, 3);
        let reference = primitive
            .derivative(crate::internal::pack_cart(&rows).view())
            .unwrap();
        for unit in [1.0e-6, 1.0e-3, 1.0e3, 1.0e6] {
            let scaled: Vec<[f64; 3]> = rows
                .iter()
                .map(|p| [p[0] * unit, p[1] * unit, p[2] * unit])
                .collect();
            let derivative = primitive
                .derivative(crate::internal::pack_cart(&scaled).view())
                .unwrap();
            for (got, want) in derivative.iter().zip(&reference) {
                assert!(
                    (got * unit - want).abs() <= 1.0e-6 * (1.0 + want.abs()),
                    "unit {unit}: {got} against {want}"
                );
            }
        }
    }

    #[test]
    fn large_basis_eigensolver_builds_fifty_plus_primitives() {
        let n_atoms = 20;
        let mut coords = Vec::with_capacity(3 * n_atoms);
        for index in 0..n_atoms {
            let t = index as f64 * 0.7;
            coords.push(1.5 * t.cos() + 0.05 * index as f64);
            coords.push(1.5 * t.sin());
            coords.push(0.9 * index as f64);
        }
        let start = Array1::from_vec(coords);
        let bonds: Vec<(usize, usize)> = (0..n_atoms - 1).map(|i| (i, i + 1)).collect();
        let config = TricConfig::default().with_bonds(Some(bonds));
        let system = TricSystem::new(start.view(), &config).unwrap();
        assert!(
            system.primitives().len() >= 50,
            "expected >= 50 primitives, got {}",
            system.primitives().len()
        );
        let u_gram = system.basis().t().dot(system.basis());
        assert_eq!(u_gram.nrows(), 3 * n_atoms);
        for row in 0..u_gram.nrows() {
            for col in 0..u_gram.ncols() {
                let expected = if row == col { 1.0 } else { 0.0 };
                assert!(
                    (u_gram[(row, col)] - expected).abs() < 1e-10,
                    "u_gram[({row}, {col})] = {}, expected {expected}",
                    u_gram[(row, col)]
                );
            }
        }
        let wilson = system.wilson_b(start.view()).unwrap();
        let gram = wilson.dot(&wilson.t());
        assert_eq!(gram.nrows(), 3 * n_atoms);
        for row in 0..gram.nrows() {
            assert!(gram[(row, row)] > 1e-6);
            for col in 0..gram.ncols() {
                if row != col {
                    assert!(
                        gram[(row, col)].abs() < 1e-8,
                        "off-diagonal gram[({row}, {col})] = {}",
                        gram[(row, col)]
                    );
                }
            }
        }
        let mut target = start.clone();
        target[0] += 0.01;
        target[4] -= 0.01;
        target[3 * (n_atoms - 1) + 2] += 0.015;
        let displacement = system.difference(target.view(), start.view()).unwrap();
        let recovered = system
            .back_transform(
                start.view(),
                displacement.view(),
                &TricInversionConfig::default(),
            )
            .unwrap();
        let diff = &recovered - &target;
        assert!(diff.dot(&diff).sqrt() < 1e-5);
    }

    #[test]
    fn back_transform_rejects_nonconvergent_or_worsening_steps() {
        let start = array![0.0, 0.0, 0.0, 0.96, 0.0, 0.0, -0.24, 0.93, 0.0];
        let target = array![0.25, -0.15, 0.1, 1.15, 0.2, -0.1, -0.05, 1.08, 0.15];
        let config = TricConfig::default().with_bonds(Some(vec![(0, 1), (0, 2)]));
        let system = TricSystem::new(start.view(), &config).unwrap();
        let displacement = system.difference(target.view(), start.view()).unwrap();

        let one_iter = TricInversionConfig::default()
            .with_max_iterations(1)
            .with_tolerance(1e-12)
            .with_max_atom_move(0.01);
        assert!(matches!(
            system.back_transform(start.view(), displacement.view(), &one_iter),
            Err(SaddleError::Solver(_))
        ));

        let tiny_move = TricInversionConfig::default()
            .with_max_iterations(32)
            .with_tolerance(1e-4)
            .with_max_atom_move(1e-7);
        assert!(matches!(
            system.back_transform(start.view(), displacement.view(), &tiny_move),
            Err(SaddleError::Solver(_))
        ));

        let near_linear = array![-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.999_999, 0.001, 0.0];
        let mut impossible = Array1::zeros(system.basis().ncols());
        for (idx, primitive) in system.primitives().iter().enumerate() {
            if matches!(primitive, TricPrimitive::Angle(1, 0, 2)) {
                for mode in 0..system.basis().ncols() {
                    impossible[mode] = -system.basis()[(idx, mode)];
                }
            }
        }
        let strict = TricInversionConfig::default()
            .with_max_iterations(10)
            .with_tolerance(1e-8)
            .with_max_atom_move(0.5);
        assert!(matches!(
            system.back_transform(near_linear.view(), impossible.view(), &strict),
            Err(SaddleError::Solver(_))
        ));
    }

    #[test]
    fn distance_gradient_and_diagnostics_guard_singular_geometries() {
        let coincident = array![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert!(matches!(
            TricPrimitive::Distance(0, 1).derivative(coincident.view()),
            Err(SaddleError::Invalid(_))
        ));
        assert_eq!(
            TricPrimitive::Distance(0, 1).diagnostic(coincident.view()),
            3
        );

        let valid = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let invalid_short = array![0.0, 0.0, 0.0];
        let translation =
            TricPrimitive::Translation(Translation::new(vec![0, 1, 2], CartAxis::X).unwrap());
        assert_eq!(translation.diagnostic(valid.view()), 0);
        assert_eq!(translation.diagnostic(invalid_short.view()), 3);

        let refpos = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let rotation =
            TricPrimitive::Rotation(Rotation::new(vec![0, 1, 2], CartAxis::Z, refpos).unwrap());
        assert_eq!(rotation.diagnostic(valid.view()), 0);
        assert_eq!(rotation.diagnostic(invalid_short.view()), 3);

        let path = array![
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.1, 0.0, 0.0]
        ];
        assert!(
            trics_path(
                path.view(),
                &TricsConfig::default().with_diagnostic_threshold(6)
            )
            .is_ok()
        );
        assert!(matches!(
            trics_path(
                path.view(),
                &TricsConfig::default().with_diagnostic_threshold(7)
            ),
            Err(SaddleError::Invalid(_))
        ));
    }

    #[test]
    fn angle_diagnostic_threshold_matches_linear_bend_switch_boundary() {
        let config = TricConfig::default().with_bonds(Some(vec![(0, 1), (1, 2)]));
        let below_rad = (180.0_f64 - 154.9).to_radians();
        let below = array![
            -1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            below_rad.cos(),
            below_rad.sin(),
            0.0
        ];
        let sys_below = TricSystem::new(below.view(), &config).unwrap();
        assert!(
            sys_below
                .primitives()
                .iter()
                .any(|p| matches!(p, TricPrimitive::Angle(0, 1, 2)))
        );
        assert!(
            !sys_below
                .primitives()
                .iter()
                .any(|p| matches!(p, TricPrimitive::LinearBend { .. }))
        );
        assert_eq!(sys_below.diagnostic(below.view()), 0);

        let above_rad = (180.0_f64 - 155.1).to_radians();
        let above = array![
            -1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            above_rad.cos(),
            above_rad.sin(),
            0.0
        ];
        assert_eq!(sys_below.diagnostic(above.view()), 1);

        let sys_above = TricSystem::new(above.view(), &config).unwrap();
        assert!(
            sys_above
                .primitives()
                .iter()
                .any(|p| matches!(p, TricPrimitive::LinearBend { .. }))
        );
        assert!(
            !sys_above
                .primitives()
                .iter()
                .any(|p| matches!(p, TricPrimitive::Angle(0, 1, 2)))
        );
        assert_eq!(sys_above.diagnostic(above.view()), 0);
    }

    #[test]
    fn maze_path_and_candidates_respect_edge_systems_and_thresholds() {
        let mut diag = Array2::from_elem((8, 8), 3_u8);
        for col in 0..=5 {
            diag[(0, col)] = 0;
        }
        for row in 0..=7 {
            diag[(row, 3)] = 0;
        }
        for col in 3..=7 {
            diag[(7, col)] = 0;
        }
        let segments = maze_path(diag.view(), 0).unwrap();
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].system, 0);
        assert_eq!(segments[1].system, 7);

        let spliced = fill_splice_segments_with_systems(&segments, diag.view(), 0, 2, 8);
        assert_eq!(spliced.len(), 2);

        let angles = [30.0_f64, 20.0, 0.0, 20.0, 30.0];
        let mut initial = Array2::zeros((angles.len(), 9));
        for (image, degrees) in angles.into_iter().enumerate() {
            let radians = degrees.to_radians();
            initial.row_mut(image).assign(&array![
                -1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                radians.cos(),
                radians.sin(),
                0.0
            ]);
        }
        let config = TricConfig::default().with_bonds(Some(vec![(0, 1), (1, 2)]));
        let systems = tric_systems(initial.view(), &config).unwrap();
        let seg = [TricsSegment {
            start: 0,
            end: 2,
            system: 2,
        }];
        let candidates = segment_candidates(&seg, &systems, initial.view(), 0);
        assert!(!candidates[0].is_empty());
        for &sys_idx in &candidates[0] {
            let max_diag = (0..=2)
                .map(|img| systems[sys_idx].diagnostic(initial.row(img)))
                .max()
                .unwrap();
            assert_eq!(max_diag, 0);
        }
        assert!(!candidates[0].contains(&0));
    }
}
