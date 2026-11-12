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
//! strained saddle.
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
    /// after Eq. (8) of Varvenne and Clouet.
    pub fn from_stress(volume: f64, defect: M3, bulk: M3) -> Result<Self, SaddleError> {
        Ok(Self {
            tensor: elastic_dipole(volume, defect, bulk)?,
        })
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
    /// `ΔE0 - (P_saddle - P_minimum) : ε`.
    pub fn linear_barrier(&self, barrier0: f64, strain: Strain) -> f64 {
        let mut delta = [[0.0; 3]; 3];
        let saddle = self.saddle.tensor();
        let minimum = self.minimum.tensor();
        for i in 0..3 {
            for j in 0..3 {
                delta[i][j] = saddle[i][j] - minimum[i][j];
            }
        }
        barrier0 - double_dot(delta, strain.tensor())
    }
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
/// `minimum_energy` is the energy of the reactant at the strained cell.
/// The surface is evaluated in Cartesian coordinates; bind it to that cell
/// before calling refine.
pub struct SaddleSeed {
    pub positions: Array1<f64>,
    pub cell: M3,
    pub mode: Array1<f64>,
    pub minimum_energy: f64,
}

#[derive(Clone, Debug, PartialEq)]
struct CachedBarrier {
    strain: M3,
    variant: String,
    barrier: f64,
    prefactor_hz: Option<f64>,
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

    /// Lowest barrier among the symmetry images at `eps`.
    pub fn barrier(&self, eps: &Strain) -> Estimate {
        let (variant, barrier) = self.lowest(eps);
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

    /// Affine-strain `seed`, run a fixed-cell solid dimer, require an
    /// index-1 saddle, and cache that barrier for the winning image.
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
            force_tol: 1e-6,
            max_move: 0.2,
            ..MinModeConfig::default()
        };
        let mut dimer = SolidDimer::new(
            SolidDimerConfig::frozen(search).with_index_check(IndexCheck::first_order()),
            positions,
            cell,
            mode,
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
        let barrier = energy - seed.minimum_energy;
        if !barrier.is_finite() {
            return Err(SaddleError::NonFinite("explicit barrier"));
        }
        let variant = self.lowest(eps).0.name.clone();
        self.cache.retain(|row| {
            !(row.variant == variant && strain_distance(row.strain, eps.tensor()) < 1e-8)
        });
        self.cache.push(CachedBarrier {
            strain: eps.tensor(),
            variant,
            barrier,
            prefactor_hz: self.prefactor_hz,
        });
        Ok(self.barrier(eps))
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
                });
            }
        }
        Ok(response)
    }

    fn lowest(&self, eps: &Strain) -> (&HopVariant, f64) {
        let mut best = &self.variants[0];
        let mut barrier = best.linear_barrier(self.barrier0, *eps);
        for variant in self.variants.iter().skip(1) {
            let value = variant.linear_barrier(self.barrier0, *eps);
            if value < barrier {
                barrier = value;
                best = variant;
            }
        }
        (best, barrier)
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
