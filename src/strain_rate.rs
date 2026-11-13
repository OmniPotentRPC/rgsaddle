// The tensor loops index both sides of a product. `is_multiple_of` is the
// same test as `% 3` on a length the rest of this file already writes that way.
#![allow(clippy::needless_range_loop, clippy::manual_is_multiple_of)]

//! Strain response of one migration event, for any potential and cell.
//!
//! The elastic dipole is the rank-2 force moment of a point defect.
//! Varvenne and Clouet, Phys. Rev. B 96, 224103 (2017),
//! doi:10.1103/PhysRevB.96.224103, Eq. (5), write its interaction with an
//! applied strain as `-P : ε`. Their Eq. (8) on a fixed cell is
//! `P = -V σ`. The paragraph after that equation removes a nonzero
//! homogeneous stress of the perfect cell, which is the correction used
//! here:
//!
//! `P = -V (σ_defect - σ_bulk)`,
//!
//! then symmetrized, because the defect has no net torque (their Eq. (1)).
//! The same tensor is the first moment of the Kanzaki forces when those
//! forces are known.
//!
//! A hop's barrier at an imposed strain follows by subtracting the two
//! interaction energies:
//!
//! `ΔE(ε) = ΔE0 - (P_saddle - P_minimum) : ε`.
//!
//! Dederichs and Schroeder, Phys. Rev. B 17, 2524 (1978),
//! doi:10.1103/PhysRevB.17.2524, show that a homogeneous strain changes
//! the jump rate through the dipole of the saddle relative to the
//! equilibrium site. [`StrainResponse::barrier`] evaluates that line for
//! every symmetry image of the hop and returns the lowest one, together
//! with a trust flag. Past the validity bound, or when the predicted
//! barrier falls through the floor, the flag is [`Trust::NeedsSearch`]
//! and [`StrainResponse::refine`] runs a fixed-cell solid dimer from the
//! strained saddle. The step is short of the minimum-escape length, and a
//! point that has left the predicted geometry or does not connect to the
//! reactant is not cached.
//!
//! The activation volume is the stress derivative of the activation
//! Gibbs energy, `Ω = ∂ΔG/∂τ`. With `ε = S τ`,
//!
//! `Ω = -W^{-1} S^T W (P_saddle - P_minimum)`,
//!
//! where `W = diag(1, 1, 1, 2, 2, 2)` is the Voigt weight that makes
//! `P : ε` agree with the full sum over both off-diagonal entries.
//! `τ` is the applied tension. Positive tension expands the cell.

use std::fmt;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::cell_log::M3;
use crate::elastic::{LocalField, Quadrupole, quadrupole_energy};
use crate::error::SaddleError;
use crate::minmode::{MinModeConfig, MinModeStatus, PointSurface};
use crate::solid_dimer::{IndexCheck, SolidDimer, SolidDimerConfig};

/// Absolute barrier error, in eV, used as the default validity tolerance.
pub const LINEAR_FOLD_EV: f64 = 0.01;

/// An eigenvalue below this is a negative mode, matching [`crate::IndexCheck::first_order`].
pub const NEGATIVE_CURVATURE: f64 = -1.0e-4;

/// Squared overlap with the translational subspace that drops a mode.
const TRANSLATION_OVERLAP2: f64 = 0.25;

/// Boltzmann constant in eV/K. This is `k / e` with the SI values of `k` and `e`.
pub const BOLTZMANN_EV_PER_K: f64 = 8.617333262145e-5;

const ELEMENTARY_CHARGE_C: f64 = 1.602176634e-19;
const ATOMIC_MASS_KG: f64 = 1.66053906660e-27;
const ANGSTROM2_M2: f64 = 1.0e-20;

/// Voigt order xx, yy, zz, yz, xz, xy. Off-diagonal entries are tensor components.
pub const VOIGT: [(usize, usize); 6] = [(0, 0), (1, 1), (2, 2), (2, 1), (2, 0), (1, 0)];

/// Infinitesimal strain `ε`, with `h = h0 (I + ε)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Strain {
    tensor: M3,
}

impl Strain {
    /// The zero strain.
    pub fn zero() -> Self {
        Self {
            tensor: [[0.0; 3]; 3],
        }
    }

    /// Symmetrize `tensor` and refuse a non-finite entry.
    pub fn from_tensor(tensor: M3) -> Result<Self, SaddleError> {
        let tensor = symmetrize(tensor);
        if !finite_m3(tensor) {
            return Err(SaddleError::NonFinite("strain"));
        }
        Ok(Self { tensor })
    }

    /// Right strain with `ε_yx = gamma` and every other entry zero.
    pub fn shear_yx(gamma: f64) -> Self {
        Self {
            tensor: [[0.0, 0.0, 0.0], [gamma, 0.0, 0.0], [0.0, 0.0, 0.0]],
        }
    }

    /// Right strain with `ε_zx = gamma` and every other entry zero.
    pub fn shear_xz(gamma: f64) -> Self {
        Self {
            tensor: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [gamma, 0.0, 0.0]],
        }
    }

    /// Uniaxial extension along x.
    pub fn uniaxial_x(eps: f64) -> Self {
        Self {
            tensor: [[eps, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        }
    }

    /// Isotropic extension `ε I`.
    pub fn hydrostatic(eps: f64) -> Self {
        Self {
            tensor: [[eps, 0.0, 0.0], [0.0, eps, 0.0], [0.0, 0.0, eps]],
        }
    }

    /// The symmetric tensor.
    pub fn tensor(self) -> M3 {
        self.tensor
    }

    /// Frobenius norm, `sqrt(Σ ε_ij²)`.
    pub fn norm(self) -> f64 {
        let mut sum = 0.0;
        for row in &self.tensor {
            for value in row {
                sum += value * value;
            }
        }
        sum.sqrt()
    }

    /// `R ε R^T`.
    pub fn transform(self, rotation: M3) -> Self {
        Self {
            tensor: transform_tensor(self.tensor, rotation),
        }
    }
}

/// Symmetric elastic dipole, in eV.
///
/// The sign is Varvenne and Clouet's: the interaction energy in an applied
/// strain is `-P : ε`. Build one with [`Self::from_stress`] or
/// [`Self::from_forces`]. [`Self::from_tensor`] accepts a tensor that
/// already uses that sign.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ElasticDipole {
    tensor: M3,
}

impl ElasticDipole {
    /// `P = -V (σ_defect - σ_bulk)`, symmetrized.
    ///
    /// `volume` is the cell volume. `defect` and `bulk` are Cauchy stresses
    /// of the defective cell and the perfect cell at the same homogeneous
    /// strain. The bulk subtraction is the homogeneous-stress correction
    /// after Eq. (8) of Varvenne and Clouet. [`Self::from_residual_stress`]
    /// is this identification under periodic boundaries.
    pub fn from_stress(volume: f64, defect: M3, bulk: M3) -> Result<Self, SaddleError> {
        Ok(Self {
            tensor: elastic_dipole(volume, defect, bulk)?,
        })
    }

    /// Dipole of one defect from the residual stress of a fixed periodic cell.
    ///
    /// The cell vectors stay at the perfect crystal, the atoms relax, and a
    /// residual stress remains because the average strain is held at zero.
    /// Varvenne and Clouet, Phys. Rev. B 96, 224103 (2017),
    /// doi:10.1103/PhysRevB.96.224103, Eq. (8), give `P = -V σ` for that
    /// stress. The perfect-cell stress is removed as in the paragraph
    /// after their Eq. (8).
    ///
    /// Pasianot, Philos. Mag. Lett. 96, 447 (2016),
    /// doi:10.1080/09500839.2016.1250965, shows that this P is the dipole
    /// of one defect, not of the periodic array: the images contribute a
    /// strain of zero cell average, and holding the cell fixed removes
    /// that homogeneous image strain from the measurement. No further
    /// term is added to P. The same P is what Varvenne, Bruneval,
    /// Marinica, and Clouet, Phys. Rev. B 88, 134102 (2013),
    /// doi:10.1103/PhysRevB.88.134102, insert into the image-interaction
    /// correction of the formation energy. A naive pile of infinite-medium
    /// fields is not the strain inside the cell; the periodic field is
    /// [`crate::CubicElasticity::periodic_strain`].
    pub fn from_residual_stress(volume: f64, defect: M3, bulk: M3) -> Result<Self, SaddleError> {
        Self::from_stress(volume, defect, bulk)
    }

    /// First moment of a force distribution, symmetrized.
    ///
    /// `P_ij = Σ_a (X_i^a F_j^a + X_j^a F_i^a) / 2`. The resultant and the
    /// torque must already vanish, as they do for Kanzaki forces of a
    /// relaxed defect. Positions and forces are Cartesian.
    pub fn from_forces(positions: &[[f64; 3]], forces: &[[f64; 3]]) -> Result<Self, SaddleError> {
        if positions.len() != forces.len() {
            return Err(SaddleError::Shape(
                "one force vector per atom in the dipole sum".into(),
            ));
        }
        let mut resultant = [0.0; 3];
        let mut torque = [0.0; 3];
        let mut raw = [[0.0; 3]; 3];
        for (x, f) in positions.iter().zip(forces) {
            if !x.iter().chain(f.iter()).all(|v| v.is_finite()) {
                return Err(SaddleError::NonFinite("dipole force"));
            }
            for i in 0..3 {
                resultant[i] += f[i];
                for j in 0..3 {
                    raw[i][j] += x[i] * f[j];
                }
            }
            torque[0] += x[1] * f[2] - x[2] * f[1];
            torque[1] += x[2] * f[0] - x[0] * f[2];
            torque[2] += x[0] * f[1] - x[1] * f[0];
        }
        let scale = raw
            .iter()
            .flatten()
            .fold(0.0_f64, |acc, v| acc.max(v.abs()))
            .max(1.0);
        let slack = 1e-8 * scale;
        if resultant.iter().any(|v| v.abs() > slack) || torque.iter().any(|v| v.abs() > slack) {
            return Err(SaddleError::Invalid(
                "a force dipole needs zero resultant and zero torque".into(),
            ));
        }
        Ok(Self {
            tensor: symmetrize(raw),
        })
    }

    /// A tensor that already uses the Varvenne sign. It is symmetrized.
    pub fn from_tensor(tensor: M3) -> Result<Self, SaddleError> {
        let tensor = symmetrize(tensor);
        if !finite_m3(tensor) {
            return Err(SaddleError::NonFinite("elastic dipole"));
        }
        Ok(Self { tensor })
    }

    /// The symmetric tensor.
    pub fn tensor(self) -> M3 {
        self.tensor
    }

    /// `R P R^T`.
    pub fn transform(self, rotation: M3) -> Self {
        Self {
            tensor: transform_tensor(self.tensor, rotation),
        }
    }

    /// `P : ε`, summed over all nine entries.
    pub fn contract(self, strain: Strain) -> f64 {
        double_dot(self.tensor, strain.tensor)
    }
}

/// One symmetry image of a hop.
#[derive(Clone, Debug, PartialEq)]
pub struct HopVariant {
    pub name: String,
    /// Rotation that produced this image from the reference hop. `I` for the reference.
    pub rotation: M3,
    pub minimum: ElasticDipole,
    pub saddle: ElasticDipole,
}

impl HopVariant {
    /// `ΔE0 - P_saddle : ε_saddle + P_minimum : ε_minimum`.
    ///
    /// The interaction energy of a dipole is `-P : ε`. When the minimum
    /// and the saddle sit in different strains, the barrier keeps both.
    /// [`Self::linear_barrier`] is the case where those strains are equal.
    pub fn barrier_at(&self, barrier0: f64, strain_minimum: Strain, strain_saddle: Strain) -> f64 {
        barrier0 - self.saddle.contract(strain_saddle) + self.minimum.contract(strain_minimum)
    }

    /// `ΔE0 - (P_saddle - P_minimum) : ε`.
    pub fn linear_barrier(&self, barrier0: f64, strain: Strain) -> f64 {
        self.barrier_at(barrier0, strain, strain)
    }
}

/// One symmetry image of a hop, evaluated in one local strain.
#[derive(Clone, Debug, PartialEq)]
pub struct HopInField {
    pub name: String,
    pub barrier: f64,
    pub prefactor_hz: Option<f64>,
    pub trust: Trust,
    /// Strain at the minimum.
    pub strain: Strain,
    /// Strain at the saddle. Equal to `strain` when the hop is evaluated
    /// in one homogeneous field.
    pub strain_saddle: Strain,
}

/// How far the dipole line is trusted.
///
/// A missing bound does not constrain that check. The floor always
/// applies: a predicted barrier at or below [`Self::floor`] needs a search.
#[derive(Clone, Debug, PartialEq)]
pub struct TrustPolicy {
    /// Largest Frobenius norm of `ε` for which the line is used.
    pub radius: Option<f64>,
    /// Estimated absolute error `quadratic * |ε|²`.
    pub quadratic: Option<f64>,
    /// The quadratic estimate may not exceed this, in eV.
    pub error_tolerance: f64,
    /// Unstable-mode curvature and its strain slope.
    pub curvature: Option<CurvatureBound>,
    /// Predicted barrier at or below this needs a search.
    pub floor: f64,
}

impl Default for TrustPolicy {
    fn default() -> Self {
        Self {
            radius: None,
            quadratic: None,
            error_tolerance: LINEAR_FOLD_EV,
            curvature: None,
            floor: 0.0,
        }
    }
}

/// Separation inside which a direct saddle replaces the dipole line.
///
/// `samples` are `(separation, absolute error)` of the dipole line against
/// a direct saddle. The radius is the smallest sampled separation at and
/// beyond which every sample is within `tolerance`. A nearer defect is
/// searched directly. At and beyond the radius the dipole barrier is used.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DirectCutoff {
    radius: f64,
}

impl DirectCutoff {
    /// Radius, in the same length unit as the samples.
    pub fn radius(self) -> f64 {
        self.radius
    }

    /// The measured errors stay within `tolerance` from `radius` outward.
    pub fn from_errors(samples: &[(f64, f64)], tolerance: f64) -> Result<Self, SaddleError> {
        if !(tolerance.is_finite() && tolerance >= 0.0) {
            return Err(SaddleError::Invalid(
                "the direct-cutoff tolerance must be finite and non-negative".into(),
            ));
        }
        if samples.is_empty() {
            return Err(SaddleError::Invalid(
                "a direct cutoff needs at least one measured separation".into(),
            ));
        }
        for (separation, error) in samples {
            if !(separation.is_finite() && *separation >= 0.0) {
                return Err(SaddleError::Invalid(
                    "a direct-cutoff separation must be finite and non-negative".into(),
                ));
            }
            if !(error.is_finite() && *error >= 0.0) {
                return Err(SaddleError::Invalid(
                    "a direct-cutoff error must be finite and non-negative".into(),
                ));
            }
        }
        let mut radii: Vec<f64> = samples.iter().map(|(separation, _)| *separation).collect();
        radii.sort_by(f64::total_cmp);
        radii.dedup();
        for radius in radii {
            let inside = samples
                .iter()
                .filter(|(separation, _)| *separation + 1.0e-9 >= radius)
                .all(|(_, error)| *error <= tolerance);
            if inside {
                return Ok(Self { radius });
            }
        }
        Err(SaddleError::Invalid(
            "the dipole line is not inside the tolerance at any measured separation".into(),
        ))
    }

    /// True when `separation` is inside the radius and needs a direct saddle.
    pub fn search_directly(self, separation: f64) -> bool {
        separation < self.radius
    }

    /// True when the dipole barrier is the value used at `separation`.
    pub fn use_dipole(self, separation: f64) -> bool {
        separation >= self.radius
    }
}

/// One measured dipole error, labelled by hop.
///
/// `orientation` is the caller's label for the dumbbell axis or the
/// sense of the hop. Samples that share a kind and an orientation get
/// one cutoff.
#[derive(Clone, Debug, PartialEq)]
pub struct HopErrorSample {
    pub kind: String,
    pub orientation: String,
    pub separation: f64,
    pub absolute_error: f64,
}

/// Where one hop may use the dipole line.
///
/// `radius` is `None` when no measured tail of this hop stays inside
/// the tolerance. That is a hop with no safe cutoff in the cells that
/// were searched.
#[derive(Clone, Debug, PartialEq)]
pub struct HopCutoff {
    pub kind: String,
    pub orientation: String,
    pub radius: Option<f64>,
}

/// One cutoff per `(kind, orientation)`.
///
/// The radius is the smallest sampled separation at and beyond which
/// every sample of that hop is within `tolerance`.
pub fn cutoffs_by_hop(
    samples: &[HopErrorSample],
    tolerance: f64,
) -> Result<Vec<HopCutoff>, SaddleError> {
    if !(tolerance.is_finite() && tolerance >= 0.0) {
        return Err(SaddleError::Invalid(
            "the direct-cutoff tolerance must be finite and non-negative".into(),
        ));
    }
    if samples.is_empty() {
        return Err(SaddleError::Invalid(
            "a per-hop cutoff needs at least one measured separation".into(),
        ));
    }
    let mut order = Vec::new();
    for sample in samples {
        if sample.kind.is_empty() || sample.orientation.is_empty() {
            return Err(SaddleError::Invalid(
                "a hop error needs a kind and an orientation".into(),
            ));
        }
        if !(sample.separation.is_finite() && sample.separation >= 0.0) {
            return Err(SaddleError::Invalid(
                "a hop separation must be finite and non-negative".into(),
            ));
        }
        if !(sample.absolute_error.is_finite() && sample.absolute_error >= 0.0) {
            return Err(SaddleError::Invalid(
                "a hop error must be finite and non-negative".into(),
            ));
        }
        let key = (sample.kind.clone(), sample.orientation.clone());
        if !order.contains(&key) {
            order.push(key);
        }
    }
    let mut out = Vec::new();
    for (kind, orientation) in order {
        let group: Vec<(f64, f64)> = samples
            .iter()
            .filter(|sample| sample.kind == kind && sample.orientation == orientation)
            .map(|sample| (sample.separation, sample.absolute_error))
            .collect();
        let radius = DirectCutoff::from_errors(&group, tolerance)
            .ok()
            .map(|cutoff| cutoff.radius());
        out.push(HopCutoff {
            kind,
            orientation,
            radius,
        });
    }
    Ok(out)
}

/// Second-order bound from the softening of the unstable mode.
///
/// `λ(ε) = λ0 + slope : ε`. `λ0` is negative. The line is trusted while
/// `λ` stays negative and at least as negative as `fraction * λ0`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CurvatureBound {
    pub lambda0: f64,
    pub slope: M3,
    pub fraction: f64,
}

/// Why [`StrainResponse::barrier`] refused the dipole line.
#[derive(Clone, Debug, PartialEq)]
pub enum SearchReason {
    /// `|ε|` is past the supplied radius.
    Radius { norm: f64, radius: f64 },
    /// The quadratic error estimate is past the tolerance.
    Quadratic {
        estimated_error: f64,
        tolerance: f64,
    },
    /// The unstable mode has softened past the curvature bound.
    Softened { lambda: f64, limit: f64 },
    /// The predicted barrier is at or below the floor.
    BelowFloor { barrier: f64, floor: f64 },
}

impl fmt::Display for SearchReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Radius { norm, radius } => {
                write!(f, "strain norm {norm} is past the radius {radius}")
            }
            Self::Quadratic {
                estimated_error,
                tolerance,
            } => write!(
                f,
                "quadratic error {estimated_error} eV is past {tolerance} eV"
            ),
            Self::Softened { lambda, limit } => {
                write!(f, "mode curvature {lambda} is softer than {limit}")
            }
            Self::BelowFloor { barrier, floor } => {
                write!(
                    f,
                    "barrier {barrier} eV is at or below the floor {floor} eV"
                )
            }
        }
    }
}

/// Whether the number in an [`Estimate`] may be used as a rate.
#[derive(Clone, Debug, PartialEq)]
pub enum Trust {
    /// The dipole line is inside the validity bound.
    Linear,
    /// A cached explicit saddle at this strain.
    Explicit,
    /// Run a search. The estimate still carries the dipole prediction.
    NeedsSearch(SearchReason),
}

/// Barrier of one hop at one strain.
#[derive(Clone, Debug, PartialEq)]
pub struct Estimate {
    pub barrier: f64,
    pub prefactor_hz: Option<f64>,
    /// Name of the symmetry image that produced `barrier`.
    pub variant: String,
    pub trust: Trust,
}

/// Reference geometry for [`StrainResponse::refine`].
///
/// `positions` is the saddle and `mode` is its unstable direction, both
/// in the reference cell. `minimum_energy` is the reactant energy in the
/// cell the search will use. The surface is evaluated in Cartesian
/// coordinates; bind it to that cell before calling refine.
pub struct SaddleSeed {
    pub positions: Array1<f64>,
    pub cell: M3,
    pub mode: Array1<f64>,
    pub minimum_energy: f64,
}

/// Per-atom cap for a search that starts on a predicted saddle.
///
/// [`MinModeConfig::max_move`] defaults to 0.2 Å, and while the lowest
/// curvature is not negative the translation steps that far along the
/// mode. That length is how a search leaves a minimum. On a predicted
/// saddle it crosses a soft barrier: the point can sit a tenth of an
/// angstrom away, on the side where the curvature is still positive, and
/// one step of 0.2 Å ends on a different index-1 saddle. 0.02 Å is short
/// of that crossing. A cap of 0.05 Å still had not met the force gate on
/// the softest barrier of the copper table after 400 steps.
const REFINE_MAX_MOVE: f64 = 0.02;

/// Force gate for a refinement that starts on the predicted saddle.
///
/// [`MinModeConfig::force_tol`] defaults to 1e-3. After the curvature
/// has gone negative the force passes 2e-3 while the 0.02 Å cap is
/// still several times `|F|/|λ|`. The next translation throws the
/// force back up and FIRE repeats the cycle, for an energy under
/// 1 meV. 2e-3 is the gate of the climb from the minimum.
const REFINE_FORCE_TOL: f64 = 2.0e-3;

/// Smallest neighborhood kept around the predicted saddle, in Å.
///
/// The supplied mode may be shorter than the relaxation of a soft
/// saddle. A refinement that moves an atom farther than the hop, and
/// farther than this, has left the prediction.
const REFINE_TRUST_FLOOR: f64 = 0.4;

/// RMS displacement from the predicted saddle below which the reactant
/// quench is skipped. A different index-1 point on this hop sits several
/// tenths of an angstrom away; the prediction itself is already within
/// a few hundredths of the explicit saddle when the image is the right one.
const REFINE_SKIP_RMS: f64 = 0.08;

#[derive(Clone, Debug, PartialEq)]
struct CachedBarrier {
    strain: M3,
    variant: String,
    barrier: f64,
    prefactor_hz: Option<f64>,
    /// Cartesian saddle and its unstable mode. Empty after a JSON
    /// round trip: the document stores the barrier, not the geometry.
    positions: Array1<f64>,
    mode: Array1<f64>,
}

/// One migration event: the zero-strain barrier, the two dipoles, and
/// the symmetry images of that hop.
#[derive(Clone, Debug, PartialEq)]
pub struct StrainResponse {
    barrier0: f64,
    minimum: ElasticDipole,
    saddle: ElasticDipole,
    prefactor_hz: Option<f64>,
    prefactor_slope: Option<M3>,
    quadrupole_minimum: Option<Quadrupole>,
    quadrupole_saddle: Option<Quadrupole>,
    variants: Vec<HopVariant>,
    policy: TrustPolicy,
    cache: Vec<CachedBarrier>,
}

impl StrainResponse {
    /// Reference hop. The only variant is this pair, named `reference`.
    pub fn new(
        barrier0: f64,
        minimum: ElasticDipole,
        saddle: ElasticDipole,
    ) -> Result<Self, SaddleError> {
        if !barrier0.is_finite() {
            return Err(SaddleError::NonFinite("zero-strain barrier"));
        }
        Ok(Self {
            barrier0,
            minimum,
            saddle,
            prefactor_hz: None,
            prefactor_slope: None,
            quadrupole_minimum: None,
            quadrupole_saddle: None,
            variants: vec![HopVariant {
                name: "reference".to_string(),
                rotation: identity(),
                minimum,
                saddle,
            }],
            policy: TrustPolicy::default(),
            cache: Vec::new(),
        })
    }

    /// Vineyard prefactor at zero strain, and optionally `∂ν/∂ε`.
    pub fn with_prefactor(mut self, hertz: f64, slope: Option<M3>) -> Result<Self, SaddleError> {
        if !(hertz.is_finite() && hertz > 0.0) {
            return Err(SaddleError::Invalid(
                "prefactor must be finite and positive".into(),
            ));
        }
        if let Some(slope) = slope
            && !finite_m3(slope)
        {
            return Err(SaddleError::NonFinite("prefactor slope"));
        }
        self.prefactor_hz = Some(hertz);
        self.prefactor_slope = slope.map(symmetrize);
        Ok(self)
    }

    /// Replace the images. Each hop keeps its own dipoles.
    pub fn with_variants(mut self, variants: Vec<HopVariant>) -> Result<Self, SaddleError> {
        if variants.is_empty() {
            return Err(SaddleError::Invalid(
                "a strain response needs at least one hop".into(),
            ));
        }
        for variant in &variants {
            if variant.name.is_empty() {
                return Err(SaddleError::Invalid("a hop needs a name".into()));
            }
        }
        self.variants = variants;
        Ok(self)
    }

    /// Images of the reference hop under `rotations`, with duplicate
    /// dipoles collapsed. The identity, when present, stays first.
    pub fn with_point_group(self, rotations: &[M3]) -> Result<Self, SaddleError> {
        let mut variants = Vec::new();
        for (index, rotation) in rotations.iter().enumerate() {
            let rotation = *rotation;
            if !finite_m3(rotation) {
                return Err(SaddleError::NonFinite("point-group rotation"));
            }
            let minimum = self.minimum.transform(rotation);
            let saddle = self.saddle.transform(rotation);
            let duplicate = variants.iter().any(|prior: &HopVariant| {
                tensor_close(prior.minimum.tensor(), minimum.tensor())
                    && tensor_close(prior.saddle.tensor(), saddle.tensor())
            });
            if duplicate {
                continue;
            }
            variants.push(HopVariant {
                name: format!("v{index}"),
                rotation,
                minimum,
                saddle,
            });
        }
        self.with_variants(variants)
    }

    /// Replace the validity bound.
    pub fn with_policy(mut self, policy: TrustPolicy) -> Result<Self, SaddleError> {
        if !(policy.error_tolerance.is_finite() && policy.error_tolerance >= 0.0) {
            return Err(SaddleError::Invalid(
                "the barrier tolerance must be finite and non-negative".into(),
            ));
        }
        if !policy.floor.is_finite() {
            return Err(SaddleError::NonFinite("barrier floor"));
        }
        if let Some(radius) = policy.radius
            && !(radius.is_finite() && radius >= 0.0)
        {
            return Err(SaddleError::Invalid(
                "the strain radius must be finite and non-negative".into(),
            ));
        }
        if let Some(quadratic) = policy.quadratic
            && !(quadratic.is_finite() && quadratic >= 0.0)
        {
            return Err(SaddleError::Invalid(
                "the quadratic error coefficient must be finite and non-negative".into(),
            ));
        }
        if let Some(bound) = policy.curvature {
            if !(bound.lambda0.is_finite() && bound.lambda0 < 0.0) {
                return Err(SaddleError::Invalid(
                    "the unstable curvature must be finite and negative".into(),
                ));
            }
            if !(bound.fraction.is_finite() && (0.0..=1.0).contains(&bound.fraction)) {
                return Err(SaddleError::Invalid(
                    "the curvature fraction must lie in [0, 1]".into(),
                ));
            }
            if !finite_m3(bound.slope) {
                return Err(SaddleError::NonFinite("curvature slope"));
            }
        }
        self.policy = policy;
        Ok(self)
    }

    /// Attach higher force moments. The dipole already stored is replaced
    /// by the dipole of each multipole, and existing images are rotated again.
    pub fn with_multipoles(
        mut self,
        minimum: crate::elastic::ElasticMultipole,
        saddle: crate::elastic::ElasticMultipole,
    ) -> Self {
        self.minimum = minimum.dipole();
        self.saddle = saddle.dipole();
        self.quadrupole_minimum = minimum.quadrupole();
        self.quadrupole_saddle = saddle.quadrupole();
        for variant in &mut self.variants {
            variant.minimum = self.minimum.transform(variant.rotation);
            variant.saddle = self.saddle.transform(variant.rotation);
        }
        self
    }

    /// Lowest barrier among the symmetry images at a homogeneous strain.
    pub fn barrier(&self, eps: &Strain) -> Estimate {
        self.barrier_in(&LocalField::homogeneous(*eps))
    }

    /// Barrier of every symmetry image at one homogeneous strain.
    ///
    /// The images are the hops of the mobile defect. A strain that breaks
    /// the point group, such as the field of a neighbouring defect, splits
    /// them. [`Self::barrier`] keeps only the lowest.
    pub fn hops_at(&self, strain: &Strain) -> Vec<HopInField> {
        let field = LocalField::homogeneous(*strain);
        self.variants
            .iter()
            .map(|variant| {
                let barrier = self.image_barrier(variant, &field);
                HopInField {
                    name: variant.name.clone(),
                    barrier,
                    prefactor_hz: self.prefactor_at(strain, variant),
                    trust: self.trust_of(strain, variant, barrier),
                    strain: *strain,
                    strain_saddle: *strain,
                }
            })
            .collect()
    }

    /// Every image when the minimum and the saddle sit in different strains.
    ///
    /// The barrier is [`HopVariant::barrier_at`]. A hop toward a neighbouring
    /// defect and the hop away from it are the two images whose saddles lie
    /// on opposite sides of the minimum.
    pub fn hops_at_ends(&self, strain_minimum: &Strain, strain_saddle: &Strain) -> Vec<HopInField> {
        self.variants
            .iter()
            .map(|variant| {
                let barrier = variant.barrier_at(self.barrier0, *strain_minimum, *strain_saddle);
                HopInField {
                    name: variant.name.clone(),
                    barrier,
                    prefactor_hz: self.prefactor_at(strain_minimum, variant),
                    trust: self.trust_of(strain_minimum, variant, barrier),
                    strain: *strain_minimum,
                    strain_saddle: *strain_saddle,
                }
            })
            .collect()
    }

    /// Dipole line plus `-(1/2) (Q_saddle - Q_minimum) : ∇ε` when a
    /// quadrupole and a strain gradient are both present.
    pub fn barrier_in(&self, field: &LocalField) -> Estimate {
        let (variant, barrier) = self.lowest(field);
        let eps = &field.strain;
        if let Some(cached) = self.cached(eps, &variant.name) {
            return Estimate {
                barrier: cached.barrier,
                prefactor_hz: cached.prefactor_hz,
                variant: variant.name.clone(),
                trust: Trust::Explicit,
            };
        }
        let prefactor_hz = self.prefactor_at(eps, variant);
        let trust = self.trust_of(eps, variant, barrier);
        Estimate {
            barrier,
            prefactor_hz,
            variant: variant.name.clone(),
            trust,
        }
    }

    /// Arrhenius rate `ν exp(-ΔE / kT)` when the line is trusted.
    pub fn rate_hz(&self, eps: &Strain, temperature_k: f64) -> Result<f64, SaddleError> {
        if !(temperature_k.is_finite() && temperature_k > 0.0) {
            return Err(SaddleError::Invalid(
                "temperature must be finite and positive".into(),
            ));
        }
        let estimate = self.barrier(eps);
        let (Trust::Linear | Trust::Explicit) = estimate.trust else {
            return Err(SaddleError::Invalid(
                "the dipole line is not trusted at this strain; refine it".into(),
            ));
        };
        let Some(prefactor) = estimate.prefactor_hz else {
            return Err(SaddleError::Invalid(
                "this event has no Vineyard prefactor".into(),
            ));
        };
        Ok(arrhenius(prefactor, estimate.barrier, temperature_k))
    }

    /// Affine-strain `seed` and refine that prediction with a fixed-cell
    /// solid dimer.
    ///
    /// The translation cap is [`REFINE_MAX_MOVE`], not the 0.2 Å
    /// minimum-escape step, and the force gate is [`REFINE_FORCE_TOL`],
    /// not the 1e-3 default of [`MinModeConfig`]. A result is cached
    /// only when it is an index-1 saddle and still in the neighborhood
    /// of the predicted geometry. A point that stays within
    /// [`REFINE_SKIP_RMS`] of the prediction, with negative curvature,
    /// is accepted without a quench. Anything farther, but still inside
    /// the neighborhood, is checked with one downhill quench. A point
    /// that has left the neighborhood, or that does not reach the
    /// reactant, is not cached.
    pub fn refine<S: PointSurface>(
        &mut self,
        eps: &Strain,
        surface: &S,
        seed: &SaddleSeed,
        max_steps: usize,
    ) -> Result<Estimate, SaddleError> {
        if max_steps == 0 {
            return Err(SaddleError::Invalid(
                "refine needs at least one dimer step".into(),
            ));
        }
        let positions = deform_vectors(&seed.positions, eps.tensor())?;
        let mode = deform_vectors(&seed.mode, eps.tensor())?;
        let cell = deform_cell(seed.cell, eps.tensor());
        let search = MinModeConfig {
            force_tol: REFINE_FORCE_TOL,
            max_move: REFINE_MAX_MOVE,
            ..MinModeConfig::default()
        };
        let found = run_frozen_dimer(
            surface,
            &search,
            &positions,
            cell,
            &mode,
            seed.minimum_energy,
            max_steps,
        )?;
        let moved = largest_atom_shift(&positions, &found.positions);
        let trust = largest_atom_length(&mode).max(REFINE_TRUST_FLOOR);
        if moved > trust {
            return Err(SaddleError::Invalid(format!(
                "the refined saddle left the predicted geometry (an atom moved {moved:.4} A; the neighborhood is {trust:.4} A)"
            )));
        }
        let rms = rms_shift(&positions, &found.positions);
        if !(found.curvature < 0.0 && rms <= REFINE_SKIP_RMS) {
            confirm_reactant(surface, &found, seed.minimum_energy)?;
        }
        let variant = self.lowest(&LocalField::homogeneous(*eps)).0.name.clone();
        self.cache.retain(|row| {
            !(row.variant == variant && strain_distance(row.strain, eps.tensor()) < 1e-8)
        });
        self.cache.push(CachedBarrier {
            strain: eps.tensor(),
            variant,
            barrier: found.barrier,
            prefactor_hz: self.prefactor_hz,
            positions: found.positions,
            mode: found.mode,
        });
        Ok(self.barrier(eps))
    }

    /// Saddle geometry cached by [`Self::refine`] for the winning image
    /// at `eps`. `None` when that strain has not been refined in this
    /// process. A JSON round trip keeps the barrier and drops the
    /// coordinates.
    pub fn refined_saddle(&self, eps: &Strain) -> Option<(Array1<f64>, Array1<f64>)> {
        let variant = self.lowest(&LocalField::homogeneous(*eps)).0;
        self.cache
            .iter()
            .find(|row| {
                row.variant == variant.name
                    && strain_distance(row.strain, eps.tensor()) < 1e-8
                    && !row.positions.is_empty()
            })
            .map(|row| (row.positions.clone(), row.mode.clone()))
    }

    /// Reference dipoles, before the point-group images.
    pub fn minimum(&self) -> ElasticDipole {
        self.minimum
    }

    /// Reference saddle dipole.
    pub fn saddle(&self) -> ElasticDipole {
        self.saddle
    }

    /// Zero-strain barrier.
    pub fn barrier0(&self) -> f64 {
        self.barrier0
    }

    /// Symmetry images, reference frame included.
    pub fn variants(&self) -> &[HopVariant] {
        &self.variants
    }

    /// Validity bound.
    pub fn policy(&self) -> &TrustPolicy {
        &self.policy
    }

    /// JSON document an AKMC store can keep with the event.
    pub fn to_json(&self) -> String {
        let mut out = String::new();
        out.push_str("{\n");
        out.push_str("  \"format\": \"rgsaddle.strain_response\",\n");
        out.push_str("  \"version\": 1,\n");
        out.push_str(&format!("  \"barrier0_eV\": {},\n", num(self.barrier0)));
        out.push_str("  \"prefactor_Hz\": ");
        push_option(&mut out, self.prefactor_hz);
        out.push_str(",\n  \"prefactor_slope\": ");
        match self.prefactor_slope {
            Some(slope) => push_matrix(&mut out, slope),
            None => out.push_str("null"),
        }
        out.push_str(",\n  \"p_minimum\": ");
        push_matrix(&mut out, self.minimum.tensor());
        out.push_str(",\n  \"p_saddle\": ");
        push_matrix(&mut out, self.saddle.tensor());
        out.push_str(",\n  \"q_minimum\": ");
        push_optional_moment(&mut out, self.quadrupole_minimum);
        out.push_str(",\n  \"q_saddle\": ");
        push_optional_moment(&mut out, self.quadrupole_saddle);
        out.push_str(",\n  \"policy\": ");
        push_policy(&mut out, &self.policy);
        out.push_str(",\n  \"variants\": [\n");
        for (i, variant) in self.variants.iter().enumerate() {
            out.push_str("    {\"name\": ");
            push_string(&mut out, &variant.name);
            out.push_str(", \"rotation\": ");
            push_matrix(&mut out, variant.rotation);
            out.push_str(", \"p_minimum\": ");
            push_matrix(&mut out, variant.minimum.tensor());
            out.push_str(", \"p_saddle\": ");
            push_matrix(&mut out, variant.saddle.tensor());
            out.push('}');
            if i + 1 != self.variants.len() {
                out.push(',');
            }
            out.push('\n');
        }
        out.push_str("  ],\n  \"cache\": [\n");
        for (i, row) in self.cache.iter().enumerate() {
            out.push_str("    {\"variant\": ");
            push_string(&mut out, &row.variant);
            out.push_str(", \"strain\": ");
            push_matrix(&mut out, row.strain);
            out.push_str(&format!(", \"barrier_eV\": {}", num(row.barrier)));
            out.push_str(", \"prefactor_Hz\": ");
            push_option(&mut out, row.prefactor_hz);
            out.push('}');
            if i + 1 != self.cache.len() {
                out.push(',');
            }
            out.push('\n');
        }
        out.push_str("  ]\n}\n");
        out
    }

    /// Read a document written by [`Self::to_json`].
    pub fn from_json(text: &str) -> Result<Self, SaddleError> {
        let value = parse_json(text)?;
        let obj = value
            .as_object()
            .ok_or_else(|| SaddleError::Invalid("strain response json must be an object".into()))?;
        let kind = obj_get(obj, "format").and_then(Json::as_str).unwrap_or("");
        if kind != "rgsaddle.strain_response" {
            return Err(SaddleError::Invalid(
                "json format is not rgsaddle.strain_response".into(),
            ));
        }
        let minimum = ElasticDipole::from_tensor(required_matrix(obj, "p_minimum")?)?;
        let saddle = ElasticDipole::from_tensor(required_matrix(obj, "p_saddle")?)?;
        let mut response = Self::new(required_f64(obj, "barrier0_eV")?, minimum, saddle)?;
        if let Some(hz) = obj_get(obj, "prefactor_Hz").and_then(Json::as_f64) {
            let slope = match obj_get(obj, "prefactor_slope") {
                None | Some(Json::Null) => None,
                Some(value) => Some(value.as_matrix().ok_or_else(|| {
                    SaddleError::Invalid("prefactor_slope must be a 3x3 matrix".into())
                })?),
            };
            response = response.with_prefactor(hz, slope)?;
        }
        response.quadrupole_minimum = optional_moment(obj, "q_minimum")?;
        response.quadrupole_saddle = optional_moment(obj, "q_saddle")?;
        if let Some(policy) = obj_get(obj, "policy").and_then(Json::as_object) {
            response = response.with_policy(policy_from_json(policy)?)?;
        }
        if let Some(variants) = obj_get(obj, "variants").and_then(Json::as_array) {
            let mut hops = Vec::new();
            for item in variants {
                let variant = item.as_object().ok_or_else(|| {
                    SaddleError::Invalid("a hop variant must be an object".into())
                })?;
                hops.push(HopVariant {
                    name: required_str(variant, "name")?.to_string(),
                    rotation: required_matrix(variant, "rotation")?,
                    minimum: ElasticDipole::from_tensor(required_matrix(variant, "p_minimum")?)?,
                    saddle: ElasticDipole::from_tensor(required_matrix(variant, "p_saddle")?)?,
                });
            }
            if !hops.is_empty() {
                response = response.with_variants(hops)?;
            }
        }
        if let Some(cache) = obj_get(obj, "cache").and_then(Json::as_array) {
            for item in cache {
                let row = item
                    .as_object()
                    .ok_or_else(|| SaddleError::Invalid("a cache row must be an object".into()))?;
                response.cache.push(CachedBarrier {
                    strain: required_matrix(row, "strain")?,
                    variant: required_str(row, "variant")?.to_string(),
                    barrier: required_f64(row, "barrier_eV")?,
                    prefactor_hz: obj_get(row, "prefactor_Hz").and_then(Json::as_f64),
                    positions: Array1::zeros(0),
                    mode: Array1::zeros(0),
                });
            }
        }
        Ok(response)
    }

    fn lowest(&self, field: &LocalField) -> (&HopVariant, f64) {
        let mut best = &self.variants[0];
        let mut barrier = self.image_barrier(best, field);
        for variant in self.variants.iter().skip(1) {
            let value = self.image_barrier(variant, field);
            if value < barrier {
                barrier = value;
                best = variant;
            }
        }
        (best, barrier)
    }

    fn image_barrier(&self, variant: &HopVariant, field: &LocalField) -> f64 {
        let mut barrier = variant.linear_barrier(self.barrier0, field.strain);
        if let (Some(gradient), Some(minimum), Some(saddle)) = (
            field.gradient,
            self.quadrupole_minimum,
            self.quadrupole_saddle,
        ) {
            let minimum = rotate_moment(minimum, variant.rotation);
            let saddle = rotate_moment(saddle, variant.rotation);
            let mut delta = minimum;
            for i in 0..3 {
                for j in 0..3 {
                    for k in 0..3 {
                        delta[i][j][k] = saddle[i][j][k] - minimum[i][j][k];
                    }
                }
            }
            barrier -= quadrupole_energy(delta, gradient);
        }
        barrier
    }

    fn cached(&self, eps: &Strain, variant: &str) -> Option<&CachedBarrier> {
        self.cache
            .iter()
            .find(|row| row.variant == variant && strain_distance(row.strain, eps.tensor()) < 1e-8)
    }

    fn prefactor_at(&self, eps: &Strain, variant: &HopVariant) -> Option<f64> {
        let base = self.prefactor_hz?;
        let Some(slope) = self.prefactor_slope else {
            return Some(base);
        };
        let slope = transform_tensor(slope, variant.rotation);
        let value = base + double_dot(slope, eps.tensor());
        if value.is_finite() && value > 0.0 {
            Some(value)
        } else {
            None
        }
    }

    fn trust_of(&self, eps: &Strain, variant: &HopVariant, barrier: f64) -> Trust {
        if barrier <= self.policy.floor {
            return Trust::NeedsSearch(SearchReason::BelowFloor {
                barrier,
                floor: self.policy.floor,
            });
        }
        let norm = eps.norm();
        if let Some(radius) = self.policy.radius
            && norm > radius
        {
            return Trust::NeedsSearch(SearchReason::Radius { norm, radius });
        }
        if let Some(quadratic) = self.policy.quadratic {
            let estimated_error = quadratic * norm * norm;
            if estimated_error > self.policy.error_tolerance {
                return Trust::NeedsSearch(SearchReason::Quadratic {
                    estimated_error,
                    tolerance: self.policy.error_tolerance,
                });
            }
        }
        if let Some(bound) = self.policy.curvature {
            let slope = transform_tensor(bound.slope, variant.rotation);
            let lambda = bound.lambda0 + double_dot(slope, eps.tensor());
            let limit = bound.fraction * bound.lambda0;
            if lambda > limit {
                return Trust::NeedsSearch(SearchReason::Softened { lambda, limit });
            }
        }
        Trust::Linear
    }
}

/// `ν exp(-ΔE / kT)` in Hz. A huge negative exponent returns 0.
pub fn arrhenius(prefactor_hz: f64, barrier: f64, temperature_k: f64) -> f64 {
    let exponent = -barrier / (BOLTZMANN_EV_PER_K * temperature_k);
    if exponent < -700.0 {
        return 0.0;
    }
    prefactor_hz * exponent.exp()
}

/// Proper rotations of the cubic point group, identity first. There are 24.
pub fn cubic_rotations() -> Vec<M3> {
    let mut perms = Vec::new();
    let mut prefix = Vec::new();
    let mut used = [false; 3];
    permute(&mut prefix, &mut used, &mut perms);
    let mut out = vec![identity()];
    for perm in perms {
        for mask in 0..8 {
            let sign = [
                if mask & 1 == 0 { 1.0 } else { -1.0 },
                if mask & 2 == 0 { 1.0 } else { -1.0 },
                if mask & 4 == 0 { 1.0 } else { -1.0 },
            ];
            let mut rotation = [[0.0; 3]; 3];
            for i in 0..3 {
                rotation[i][perm[i]] = sign[i];
            }
            if det(rotation) <= 0.0 || tensor_close(rotation, identity()) {
                continue;
            }
            if out.iter().any(|prior| tensor_close(*prior, rotation)) {
                continue;
            }
            out.push(rotation);
        }
    }
    out
}

fn permute(prefix: &mut Vec<usize>, used: &mut [bool; 3], out: &mut Vec<[usize; 3]>) {
    if prefix.len() == 3 {
        out.push([prefix[0], prefix[1], prefix[2]]);
        return;
    }
    for i in 0..3 {
        if used[i] {
            continue;
        }
        used[i] = true;
        prefix.push(i);
        permute(prefix, used, out);
        prefix.pop();
        used[i] = false;
    }
}

/// Quarter turns about x. `1` sends `[110]` to `[101]`.
pub fn rotation_x(quarter_turns: i32) -> M3 {
    match quarter_turns.rem_euclid(4) {
        0 => identity(),
        1 => [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        2 => [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        _ => [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
    }
}

fn rotate_moment(moment: Quadrupole, rotation: M3) -> Quadrupole {
    let mut out = [[[0.0; 3]; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                let mut sum = 0.0;
                for a in 0..3 {
                    for b in 0..3 {
                        for c in 0..3 {
                            sum +=
                                rotation[i][a] * rotation[j][b] * rotation[k][c] * moment[a][b][c];
                        }
                    }
                }
                out[i][j][k] = sum;
            }
        }
    }
    out
}

fn identity() -> M3 {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

fn deform_vectors(values: &Array1<f64>, eps: M3) -> Result<Array1<f64>, SaddleError> {
    if values.len() % 3 != 0 {
        return Err(SaddleError::Shape(
            "a Cartesian vector has length divisible by 3".into(),
        ));
    }
    let mut deform = eps;
    for i in 0..3 {
        deform[i][i] += 1.0;
    }
    let mut out = Array1::zeros(values.len());
    for atom in 0..values.len() / 3 {
        let row = [values[3 * atom], values[3 * atom + 1], values[3 * atom + 2]];
        for c in 0..3 {
            out[3 * atom + c] =
                row[0] * deform[0][c] + row[1] * deform[1][c] + row[2] * deform[2][c];
        }
    }
    Ok(out)
}

struct FrozenSaddle {
    positions: Array1<f64>,
    mode: Array1<f64>,
    barrier: f64,
    curvature: f64,
}

fn run_frozen_dimer<S: PointSurface>(
    surface: &S,
    search: &MinModeConfig,
    positions: &Array1<f64>,
    cell: M3,
    mode: &Array1<f64>,
    minimum_energy: f64,
    max_steps: usize,
) -> Result<FrozenSaddle, SaddleError> {
    let mut dimer = SolidDimer::new(
        SolidDimerConfig::frozen(search.clone()).with_index_check(IndexCheck::first_order()),
        positions.clone(),
        cell,
        mode.clone(),
    )?;
    let mut report = dimer.step(surface)?;
    let mut steps = 1usize;
    while report.status == MinModeStatus::Running && steps < max_steps {
        report = dimer.step(surface)?;
        steps += 1;
    }
    if report.status != MinModeStatus::Converged || report.rejected {
        return Err(SaddleError::Invalid(format!(
            "the strained saddle search did not stop on an index-1 saddle (status {:?}, rejected {}, curvature {}, steps {steps})",
            report.status, report.rejected, report.curvature
        )));
    }
    let (energy, _) = surface.eval(dimer.positions())?;
    let barrier = energy - minimum_energy;
    if !barrier.is_finite() {
        return Err(SaddleError::NonFinite("explicit barrier"));
    }
    Ok(FrozenSaddle {
        positions: dimer.positions().to_owned(),
        mode: dimer.mode().to_owned(),
        barrier,
        curvature: report.curvature,
    })
}

/// One downhill quench. Used only when the refined point has moved
/// farther than [`REFINE_SKIP_RMS`] from the prediction. The side that
/// drops the energy is minimized to a loose force gate.
fn confirm_reactant<S: PointSurface>(
    surface: &S,
    found: &FrozenSaddle,
    minimum_energy: f64,
) -> Result<(), SaddleError> {
    let norm = found.mode.dot(&found.mode).sqrt();
    if !(norm.is_finite() && norm > 1e-14) {
        return Err(SaddleError::Invalid(
            "the refined saddle has no unstable mode".into(),
        ));
    }
    let unit = &found.mode / norm;
    let mut best_sign = 1.0;
    let mut best_energy = f64::INFINITY;
    for sign in [1.0, -1.0] {
        let trial = &found.positions + &(&unit * (sign * 0.2));
        let (energy, _) = surface.eval(trial.view())?;
        if energy < best_energy {
            best_energy = energy;
            best_sign = sign;
        }
    }
    let mut trial = &found.positions + &(&unit * (best_sign * 0.25));
    let energy = quench(surface, &mut trial, 1.0e-2, 40)?;
    let gap = energy - minimum_energy;
    if gap.abs() > 5.0e-2 {
        return Err(SaddleError::Invalid(format!(
            "the refined saddle does not connect to the reactant (barrier {:.6} eV, quench gap {gap:.6} eV)",
            found.barrier
        )));
    }
    Ok(())
}

fn quench<S: PointSurface>(
    surface: &S,
    position: &mut Array1<f64>,
    force_tol: f64,
    max_steps: usize,
) -> Result<f64, SaddleError> {
    let mut solver = rgmin::Solver::new(
        rgmin::Method::Fire {
            kind: rgmin::FireKind::V2,
        },
        rgmin::Control {
            maxiter: usize::MAX,
            gtol: 0.0,
            ..rgmin::Control::default()
        },
        position.len(),
    );
    solver.set_atom_maxmove(0.05);
    let mut energy = 0.0;
    for _ in 0..max_steps {
        let oracle = rgmin::Oracle::unbounded(position.len(), |x: ArrayView1<f64>| {
            surface
                .eval(x)
                .unwrap_or_else(|_| (f64::INFINITY, Array1::zeros(x.len())))
        });
        solver
            .step(&oracle, position)
            .map_err(|err| SaddleError::Solver(err.to_string()))?;
        let (value, gradient) = surface.eval(position.view())?;
        energy = value;
        let max_component = gradient.iter().map(|value| value.abs()).fold(0.0, f64::max);
        if max_component < force_tol {
            break;
        }
    }
    if !energy.is_finite() {
        return Err(SaddleError::NonFinite("reactant quench"));
    }
    Ok(energy)
}

fn largest_atom_length(values: &Array1<f64>) -> f64 {
    let mut max_atom = 0.0_f64;
    let mut index = 0;
    while index + 3 <= values.len() {
        let r = (values[index] * values[index]
            + values[index + 1] * values[index + 1]
            + values[index + 2] * values[index + 2])
            .sqrt();
        max_atom = max_atom.max(r);
        index += 3;
    }
    max_atom
}

fn largest_atom_shift(origin: &Array1<f64>, trial: &Array1<f64>) -> f64 {
    largest_atom_length(&(trial - origin))
}

fn rms_shift(origin: &Array1<f64>, trial: &Array1<f64>) -> f64 {
    let delta = trial - origin;
    let atoms = delta.len() / 3;
    if atoms == 0 {
        return 0.0;
    }
    (delta.dot(&delta) / atoms as f64).sqrt()
}

fn deform_cell(cell: M3, eps: M3) -> M3 {
    let mut deform = eps;
    for i in 0..3 {
        deform[i][i] += 1.0;
    }
    let mut h = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            h[i][j] =
                cell[i][0] * deform[0][j] + cell[i][1] * deform[1][j] + cell[i][2] * deform[2][j];
        }
    }
    h
}

/// `(P_ij + P_ji) / 2`.
pub fn symmetrize(m: M3) -> M3 {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = 0.5 * (m[i][j] + m[j][i]);
        }
    }
    out
}

/// Sum over all nine entries, `A_ij B_ij`.
pub fn double_dot(a: M3, b: M3) -> f64 {
    let mut sum = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            sum += a[i][j] * b[i][j];
        }
    }
    sum
}

/// `P' = R P R^T` for a rotation `R` that sends column vectors to `R v`.
pub fn transform_tensor(tensor: M3, rotation: M3) -> M3 {
    let mut rotated = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut sum = 0.0;
            for k in 0..3 {
                for l in 0..3 {
                    sum += rotation[i][k] * tensor[k][l] * rotation[j][l];
                }
            }
            rotated[i][j] = sum;
        }
    }
    symmetrize(rotated)
}

/// Symmetric tensor components in [`VOIGT`] order.
pub fn voigt(m: M3) -> [f64; 6] {
    let mut out = [0.0; 6];
    for (slot, (i, j)) in VOIGT.iter().enumerate() {
        out[slot] = 0.5 * (m[*i][*j] + m[*j][*i]);
    }
    out
}

/// Symmetric matrix from [`VOIGT`] components.
pub fn from_voigt(v: [f64; 6]) -> M3 {
    let mut out = [[0.0; 3]; 3];
    for (slot, (i, j)) in VOIGT.iter().enumerate() {
        out[*i][*j] = v[slot];
        out[*j][*i] = v[slot];
    }
    out
}

/// Varvenne dipole `P = -V (σ_defect - σ_bulk)`, symmetrized.
pub fn elastic_dipole(volume: f64, defect_stress: M3, bulk_stress: M3) -> Result<M3, SaddleError> {
    if !(volume.is_finite() && volume > 0.0) {
        return Err(SaddleError::Invalid(
            "elastic dipole needs a positive finite volume".into(),
        ));
    }
    if !finite_m3(defect_stress) || !finite_m3(bulk_stress) {
        return Err(SaddleError::NonFinite("elastic dipole stress"));
    }
    let mut raw = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            raw[i][j] = -volume * (defect_stress[i][j] - bulk_stress[i][j]);
        }
    }
    Ok(symmetrize(raw))
}

/// Activation volume `Ω = ∂ΔG/∂τ` in Å³.
///
/// `compliance` maps a Voigt tension `(xx, yy, zz, yz, xz, xy)` onto the
/// Voigt strain it produces. Both vectors use tensor components, not
/// engineering shear.
pub fn activation_volume(
    minimum: M3,
    saddle: M3,
    compliance: &[[f64; 6]; 6],
) -> Result<M3, SaddleError> {
    if !compliance.iter().flatten().all(|value| value.is_finite()) {
        return Err(SaddleError::NonFinite("compliance"));
    }
    let delta = voigt(sub(symmetrize(saddle), symmetrize(minimum)));
    let mut weighted = [0.0; 6];
    for (i, slot) in weighted.iter_mut().enumerate() {
        *slot = voigt_weight(i) * delta[i];
    }
    let mut transformed = [0.0; 6];
    for (i, slot) in transformed.iter_mut().enumerate() {
        for (j, weight) in weighted.iter().enumerate() {
            *slot += compliance[j][i] * weight;
        }
    }
    let mut omega = [0.0; 6];
    for (i, slot) in omega.iter_mut().enumerate() {
        *slot = -transformed[i] / voigt_weight(i);
    }
    Ok(from_voigt(omega))
}

/// Invert a 6×6 stiffness, in the same Voigt layout as [`activation_volume`].
pub fn compliance_from_stiffness(stiffness: &[[f64; 6]; 6]) -> Result<[[f64; 6]; 6], SaddleError> {
    if !stiffness.iter().flatten().all(|value| value.is_finite()) {
        return Err(SaddleError::NonFinite("stiffness"));
    }
    invert6(stiffness)
}

/// `sqrt(eV / (u Å²))` in rad/s, the factor that turns `sqrt(λ / m)` into an angular frequency.
pub fn angular_scale_rad_s() -> f64 {
    (ELEMENTARY_CHARGE_C / (ATOMIC_MASS_KG * ANGSTROM2_M2)).sqrt()
}

/// Vineyard prefactor in Hz.
///
/// `min_lambda` and `sad_lambda` are positive Cartesian Hessian eigenvalues
/// in eV/Å². Translations are already removed, and the saddle's negative
/// eigenvalue is not included. Every atom has the same mass `mass_amu`.
/// `ν_i = ω_i / 2π` with `ω_i = sqrt(λ_i / m)`, and the prefactor is
/// `(∏ ν_min) / (∏ ν_sad)`.
pub fn vineyard_prefactor(
    min_lambda: &[f64],
    sad_lambda: &[f64],
    mass_amu: f64,
) -> Result<f64, SaddleError> {
    if !(mass_amu.is_finite() && mass_amu > 0.0) {
        return Err(SaddleError::Invalid(
            "Vineyard mass must be finite and positive".into(),
        ));
    }
    if min_lambda.is_empty() {
        return Err(SaddleError::Invalid(
            "Vineyard needs at least one positive minimum eigenvalue".into(),
        ));
    }
    if min_lambda.len() < sad_lambda.len() {
        return Err(SaddleError::Invalid(
            "the saddle product is longer than the minimum product".into(),
        ));
    }
    let mut log_ratio = 0.0;
    for lambda in min_lambda {
        if !(lambda.is_finite() && *lambda > 0.0) {
            return Err(SaddleError::Invalid(
                "minimum eigenvalues in the Vineyard product must be positive".into(),
            ));
        }
        log_ratio += 0.5 * lambda.ln();
    }
    for lambda in sad_lambda {
        if !(lambda.is_finite() && *lambda > 0.0) {
            return Err(SaddleError::Invalid(
                "saddle eigenvalues in the Vineyard product must be positive".into(),
            ));
        }
        log_ratio -= 0.5 * lambda.ln();
    }
    let extra = (min_lambda.len() - sad_lambda.len()) as f64;
    log_ratio -= 0.5 * extra * mass_amu.ln();
    log_ratio -= extra * (2.0 * std::f64::consts::PI).ln();
    let hertz = log_ratio.exp() * angular_scale_rad_s().powf(extra);
    if !hertz.is_finite() || hertz <= 0.0 {
        return Err(SaddleError::NonFinite("Vineyard prefactor"));
    }
    Ok(hertz)
}

/// Positive eigenvalues kept for a Vineyard product, after translations are removed.
#[derive(Clone, Debug, PartialEq)]
pub struct TranslationalSplit {
    pub negative: Vec<f64>,
    pub positive: Vec<f64>,
    pub translations: usize,
}

/// Drop translational eigenvectors and split the rest at [`NEGATIVE_CURVATURE`].
///
/// `eigenvectors` are columns. `translations` are rows, one rigid shift per
/// row, not necessarily orthonormal. A mode whose squared overlap with that
/// subspace is at least 0.25 is a translation and is left out of both products.
pub fn split_translational_modes(
    eigenvalues: &[f64],
    eigenvectors: ArrayView2<f64>,
    translations: ArrayView2<f64>,
) -> Result<TranslationalSplit, SaddleError> {
    if eigenvalues.len() != eigenvectors.ncols() {
        return Err(SaddleError::Shape(
            "one eigenvalue per eigenvector column".into(),
        ));
    }
    if translations.ncols() != eigenvectors.nrows() && translations.nrows() != 0 {
        return Err(SaddleError::Shape(
            "translation rows must match the eigenvector length".into(),
        ));
    }
    let basis = orthonormal_rows(translations);
    let mut negative = Vec::new();
    let mut positive = Vec::new();
    let mut translations_found = 0usize;
    for (index, lambda) in eigenvalues.iter().enumerate() {
        if !lambda.is_finite() {
            return Err(SaddleError::NonFinite("hessian eigenvalue"));
        }
        let mode = eigenvectors.column(index);
        if translation_overlap2(mode, &basis) >= TRANSLATION_OVERLAP2 {
            translations_found += 1;
            continue;
        }
        if *lambda < NEGATIVE_CURVATURE {
            negative.push(*lambda);
        } else if *lambda > 0.0 {
            positive.push(*lambda);
        }
    }
    Ok(TranslationalSplit {
        negative,
        positive,
        translations: translations_found,
    })
}

fn orthonormal_rows(translations: ArrayView2<f64>) -> Array2<f64> {
    let mut kept: Vec<Vec<f64>> = Vec::new();
    for row in translations.rows() {
        let mut v: Vec<f64> = row.iter().copied().collect();
        for basis in &kept {
            let c = dot_slice(&v, basis);
            for (slot, basis_value) in v.iter_mut().zip(basis) {
                *slot -= c * basis_value;
            }
        }
        let norm = dot_slice(&v, &v).sqrt();
        if norm > 1e-12 {
            for value in &mut v {
                *value /= norm;
            }
            kept.push(v);
        }
    }
    let n = translations.ncols();
    let mut out = Array2::zeros((kept.len(), n));
    for (i, row) in kept.iter().enumerate() {
        for (j, value) in row.iter().enumerate() {
            out[(i, j)] = *value;
        }
    }
    out
}

fn translation_overlap2(mode: ArrayView1<f64>, basis: &Array2<f64>) -> f64 {
    let norm2 = dot_view(mode, mode);
    if norm2 <= 0.0 {
        return 1.0;
    }
    let mut projected = 0.0;
    for row in basis.rows() {
        let c = dot_view(mode, row);
        projected += c * c;
    }
    projected / norm2
}

fn dot_slice(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn dot_view(left: ArrayView1<f64>, right: ArrayView1<f64>) -> f64 {
    left.iter().zip(right.iter()).map(|(a, b)| a * b).sum()
}

fn voigt_weight(index: usize) -> f64 {
    if index < 3 { 1.0 } else { 2.0 }
}

fn sub(a: M3, b: M3) -> M3 {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[i][j] - b[i][j];
        }
    }
    out
}

fn finite_m3(m: M3) -> bool {
    m.iter().flatten().all(|value| value.is_finite())
}

fn tensor_close(a: M3, b: M3) -> bool {
    strain_distance(a, b) < 1e-8
}

fn strain_distance(a: M3, b: M3) -> f64 {
    let mut acc = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            let d = a[i][j] - b[i][j];
            acc += d * d;
        }
    }
    acc.sqrt()
}

fn det(m: M3) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

fn invert6(matrix: &[[f64; 6]; 6]) -> Result<[[f64; 6]; 6], SaddleError> {
    let mut a = [[0.0; 12]; 6];
    for i in 0..6 {
        for j in 0..6 {
            a[i][j] = matrix[i][j];
            a[i][j + 6] = if i == j { 1.0 } else { 0.0 };
        }
    }
    for col in 0..6 {
        let mut pivot = col;
        let mut best = a[col][col].abs();
        for row in col + 1..6 {
            let value = a[row][col].abs();
            if value > best {
                best = value;
                pivot = row;
            }
        }
        if best < 1e-18 {
            return Err(SaddleError::Invalid(
                "stiffness is singular, so the compliance is undefined".into(),
            ));
        }
        a.swap(col, pivot);
        let diag = a[col][col];
        for j in 0..12 {
            a[col][j] /= diag;
        }
        for row in 0..6 {
            if row == col {
                continue;
            }
            let factor = a[row][col];
            for j in 0..12 {
                a[row][j] -= factor * a[col][j];
            }
        }
    }
    let mut out = [[0.0; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            out[i][j] = a[i][j + 6];
        }
    }
    Ok(out)
}

fn num(value: f64) -> String {
    if value == 0.0 {
        "0".to_string()
    } else {
        format!("{value}")
    }
}

fn push_option(out: &mut String, value: Option<f64>) {
    match value {
        Some(value) => out.push_str(&num(value)),
        None => out.push_str("null"),
    }
}

fn push_string(out: &mut String, text: &str) {
    out.push('"');
    out.push_str(text);
    out.push('"');
}

fn push_optional_moment(out: &mut String, moment: Option<Quadrupole>) {
    match moment {
        None => out.push_str("null"),
        Some(moment) => {
            out.push('[');
            for i in 0..3 {
                if i > 0 {
                    out.push(',');
                }
                out.push('[');
                for j in 0..3 {
                    if j > 0 {
                        out.push(',');
                    }
                    out.push('[');
                    for k in 0..3 {
                        if k > 0 {
                            out.push(',');
                        }
                        out.push_str(&num(moment[i][j][k]));
                    }
                    out.push(']');
                }
                out.push(']');
            }
            out.push(']');
        }
    }
}

fn optional_moment(obj: &[(String, Json)], key: &str) -> Result<Option<Quadrupole>, SaddleError> {
    match obj_get(obj, key) {
        None | Some(Json::Null) => Ok(None),
        Some(value) => value
            .as_moment()
            .map(Some)
            .ok_or_else(|| SaddleError::Invalid(format!("json moment {key} must be 3x3x3"))),
    }
}

fn push_matrix(out: &mut String, m: M3) {
    out.push('[');
    for i in 0..3 {
        if i > 0 {
            out.push(',');
        }
        out.push('[');
        for j in 0..3 {
            if j > 0 {
                out.push(',');
            }
            out.push_str(&num(m[i][j]));
        }
        out.push(']');
    }
    out.push(']');
}

fn push_policy(out: &mut String, policy: &TrustPolicy) {
    out.push('{');
    out.push_str("\"radius\": ");
    push_option(out, policy.radius);
    out.push_str(", \"quadratic\": ");
    push_option(out, policy.quadratic);
    out.push_str(&format!(
        ", \"error_tolerance_eV\": {}",
        num(policy.error_tolerance)
    ));
    out.push_str(&format!(", \"floor_eV\": {}", num(policy.floor)));
    out.push_str(", \"curvature\": ");
    match policy.curvature {
        None => out.push_str("null"),
        Some(bound) => {
            out.push_str("{\"lambda0\": ");
            out.push_str(&num(bound.lambda0));
            out.push_str(", \"slope\": ");
            push_matrix(out, bound.slope);
            out.push_str(", \"fraction\": ");
            out.push_str(&num(bound.fraction));
            out.push('}');
        }
    }
    out.push('}');
}

fn policy_from_json(obj: &[(String, Json)]) -> Result<TrustPolicy, SaddleError> {
    let curvature = match obj_get(obj, "curvature") {
        None | Some(Json::Null) => None,
        Some(value) => {
            let bound = value
                .as_object()
                .ok_or_else(|| SaddleError::Invalid("curvature bound must be an object".into()))?;
            Some(CurvatureBound {
                lambda0: required_f64(bound, "lambda0")?,
                slope: required_matrix(bound, "slope")?,
                fraction: required_f64(bound, "fraction")?,
            })
        }
    };
    Ok(TrustPolicy {
        radius: obj_get(obj, "radius").and_then(Json::as_f64),
        quadratic: obj_get(obj, "quadratic").and_then(Json::as_f64),
        error_tolerance: obj_get(obj, "error_tolerance_eV")
            .and_then(Json::as_f64)
            .unwrap_or(LINEAR_FOLD_EV),
        curvature,
        floor: obj_get(obj, "floor_eV")
            .and_then(Json::as_f64)
            .unwrap_or(0.0),
    })
}

#[derive(Clone, Debug)]
enum Json {
    Null,
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    fn as_object(&self) -> Option<&[(String, Json)]> {
        match self {
            Self::Obj(pairs) => Some(pairs),
            _ => None,
        }
    }

    fn as_array(&self) -> Option<&[Json]> {
        match self {
            Self::Arr(items) => Some(items),
            _ => None,
        }
    }

    fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Num(value) => Some(*value),
            _ => None,
        }
    }

    fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(text) => Some(text),
            _ => None,
        }
    }

    fn as_matrix(&self) -> Option<M3> {
        let rows = self.as_array()?;
        if rows.len() != 3 {
            return None;
        }
        let mut out = [[0.0; 3]; 3];
        for (i, row) in rows.iter().enumerate() {
            let cols = row.as_array()?;
            if cols.len() != 3 {
                return None;
            }
            for (j, value) in cols.iter().enumerate() {
                out[i][j] = value.as_f64()?;
            }
        }
        Some(out)
    }

    fn as_moment(&self) -> Option<Quadrupole> {
        let rows = self.as_array()?;
        if rows.len() != 3 {
            return None;
        }
        let mut out = [[[0.0; 3]; 3]; 3];
        for (i, row) in rows.iter().enumerate() {
            let mid = row.as_array()?;
            if mid.len() != 3 {
                return None;
            }
            for (j, layer) in mid.iter().enumerate() {
                let cols = layer.as_array()?;
                if cols.len() != 3 {
                    return None;
                }
                for (k, value) in cols.iter().enumerate() {
                    out[i][j][k] = value.as_f64()?;
                }
            }
        }
        Some(out)
    }
}

fn obj_get<'a>(obj: &'a [(String, Json)], key: &str) -> Option<&'a Json> {
    obj.iter()
        .find(|(name, _)| name == key)
        .map(|(_, value)| value)
}

fn required_f64(obj: &[(String, Json)], key: &str) -> Result<f64, SaddleError> {
    obj_get(obj, key)
        .and_then(Json::as_f64)
        .ok_or_else(|| SaddleError::Invalid(format!("json is missing the number {key}")))
}

fn required_str<'a>(obj: &'a [(String, Json)], key: &str) -> Result<&'a str, SaddleError> {
    obj_get(obj, key)
        .and_then(Json::as_str)
        .ok_or_else(|| SaddleError::Invalid(format!("json is missing the string {key}")))
}

fn required_matrix(obj: &[(String, Json)], key: &str) -> Result<M3, SaddleError> {
    obj_get(obj, key)
        .and_then(Json::as_matrix)
        .ok_or_else(|| SaddleError::Invalid(format!("json is missing the matrix {key}")))
}

struct Parser<'a> {
    bytes: &'a [u8],
    index: usize,
}

fn parse_json(text: &str) -> Result<Json, SaddleError> {
    let mut parser = Parser {
        bytes: text.as_bytes(),
        index: 0,
    };
    let value = parser.parse_value()?;
    parser.skip();
    if parser.index != parser.bytes.len() {
        return Err(SaddleError::Invalid(
            "trailing data after the strain response json".into(),
        ));
    }
    Ok(value)
}

impl<'a> Parser<'a> {
    fn skip(&mut self) {
        while self.index < self.bytes.len() && self.bytes[self.index].is_ascii_whitespace() {
            self.index += 1;
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.skip();
        self.bytes.get(self.index).copied()
    }

    fn eat(&mut self, expected: u8) -> Result<(), SaddleError> {
        if self.peek() != Some(expected) {
            return Err(SaddleError::Invalid("unexpected json token".into()));
        }
        self.index += 1;
        Ok(())
    }

    fn parse_value(&mut self) -> Result<Json, SaddleError> {
        match self.peek() {
            Some(b'n') => self.parse_literal(b"null", Json::Null),
            Some(b't') => self.parse_literal(b"true", Json::Num(1.0)),
            Some(b'f') => self.parse_literal(b"false", Json::Num(0.0)),
            Some(b'"') => Ok(Json::Str(self.parse_string()?)),
            Some(b'[') => self.parse_array(),
            Some(b'{') => self.parse_object(),
            Some(b'-') | Some(b'+') | Some(b'0'..=b'9') => Ok(Json::Num(self.parse_number()?)),
            _ => Err(SaddleError::Invalid("invalid json value".into())),
        }
    }

    fn parse_literal(&mut self, literal: &[u8], value: Json) -> Result<Json, SaddleError> {
        if self.bytes[self.index..].starts_with(literal) {
            self.index += literal.len();
            Ok(value)
        } else {
            Err(SaddleError::Invalid("invalid json literal".into()))
        }
    }

    fn parse_array(&mut self) -> Result<Json, SaddleError> {
        self.eat(b'[')?;
        let mut items = Vec::new();
        if self.peek() == Some(b']') {
            self.index += 1;
            return Ok(Json::Arr(items));
        }
        loop {
            items.push(self.parse_value()?);
            match self.peek() {
                Some(b',') => self.index += 1,
                Some(b']') => {
                    self.index += 1;
                    break;
                }
                _ => return Err(SaddleError::Invalid("unclosed json array".into())),
            }
        }
        Ok(Json::Arr(items))
    }

    fn parse_object(&mut self) -> Result<Json, SaddleError> {
        self.eat(b'{')?;
        let mut pairs = Vec::new();
        if self.peek() == Some(b'}') {
            self.index += 1;
            return Ok(Json::Obj(pairs));
        }
        loop {
            if self.peek() != Some(b'"') {
                return Err(SaddleError::Invalid(
                    "json object key must be a string".into(),
                ));
            }
            let key = self.parse_string()?;
            self.eat(b':')?;
            pairs.push((key, self.parse_value()?));
            match self.peek() {
                Some(b',') => self.index += 1,
                Some(b'}') => {
                    self.index += 1;
                    break;
                }
                _ => return Err(SaddleError::Invalid("unclosed json object".into())),
            }
        }
        Ok(Json::Obj(pairs))
    }

    fn parse_string(&mut self) -> Result<String, SaddleError> {
        self.eat(b'"')?;
        let mut out = String::new();
        while self.index < self.bytes.len() {
            let byte = self.bytes[self.index];
            self.index += 1;
            match byte {
                b'"' => return Ok(out),
                b'\\' => {
                    let escaped = self
                        .bytes
                        .get(self.index)
                        .copied()
                        .ok_or_else(|| SaddleError::Invalid("truncated json string".into()))?;
                    self.index += 1;
                    match escaped {
                        b'"' | b'\\' | b'/' => out.push(escaped as char),
                        b'n' => out.push('\n'),
                        _ => {
                            return Err(SaddleError::Invalid(
                                "unsupported json string escape".into(),
                            ));
                        }
                    }
                }
                _ => out.push(byte as char),
            }
        }
        Err(SaddleError::Invalid("truncated json string".into()))
    }

    fn parse_number(&mut self) -> Result<f64, SaddleError> {
        let start = self.index;
        if matches!(self.bytes.get(self.index), Some(b'+' | b'-')) {
            self.index += 1;
        }
        while matches!(self.bytes.get(self.index), Some(b'0'..=b'9')) {
            self.index += 1;
        }
        if self.bytes.get(self.index) == Some(&b'.') {
            self.index += 1;
            while matches!(self.bytes.get(self.index), Some(b'0'..=b'9')) {
                self.index += 1;
            }
        }
        if matches!(self.bytes.get(self.index), Some(b'e' | b'E')) {
            self.index += 1;
            if matches!(self.bytes.get(self.index), Some(b'+' | b'-')) {
                self.index += 1;
            }
            while matches!(self.bytes.get(self.index), Some(b'0'..=b'9')) {
                self.index += 1;
            }
        }
        let text = std::str::from_utf8(&self.bytes[start..self.index])
            .map_err(|_| SaddleError::Invalid("invalid json number".into()))?;
        text.parse::<f64>()
            .map_err(|_| SaddleError::Invalid(format!("invalid json number {text}")))
    }
}
