//! Strain-dependent harmonic rates from elastic dipoles.
//!
//! [`SolidDimer::elastic_dipole`](crate::SolidDimer::elastic_dipole) is the
//! virial `V σ`, and `σ = (1/V) ∂E/∂ε` for a right strain at fixed
//! fractional coordinates. The dipole in the interaction energy has the
//! opposite sign. Varvenne and Clouet, Phys. Rev. B 96, 224103 (2017),
//! doi:10.1103/PhysRevB.96.224103, Eq. (5), write that energy as `-P : ε`.
//! Their Eq. (8) is `P = -V σ` on a fixed cell. The paragraph after Eq. (8)
//! removes a nonzero homogeneous stress of the perfect cell, which is the
//! correction applied here:
//!
//! `P = -V (σ_defect - σ_bulk)`,
//!
//! then symmetrized, because the defect has no net torque (their Eq. (1)).
//!
//! The barrier at an imposed strain follows by subtracting the two
//! interaction energies. The bulk term cancels:
//!
//! `ΔE(ε) = ΔE0 - (P_saddle - P_minimum) : ε`.
//!
//! Dederichs and Schroeder, Phys. Rev. B 17, 2524 (1978),
//! doi:10.1103/PhysRevB.17.2524, show that a homogeneous strain changes the
//! jump rate through the dipole of the saddle relative to the equilibrium
//! site. Their elasto-diffusion constants measure that saddle dipole. The
//! activation volume used here is the stress derivative of the activation
//! Gibbs energy, `Ω = ∂ΔG/∂τ`, with `τ` the applied tension (positive
//! tension expands the cell; it is minus [`JointChart`](crate::JointChart)'s
//! external stress). With `ε = S τ`,
//!
//! `Ω = -W^{-1} S^T W (P_saddle - P_minimum)`,
//!
//! where `W = diag(1, 1, 1, 2, 2, 2)` is the Voigt weight that makes
//! `P : ε` agree with the full sum over both off-diagonal entries.
//!
//! Fixed-cell barriers are Helmholtz. Variable-cell barriers are the
//! enthalpy `E + V0 σ_ext : ε` and are not substituted into the linear
//! column. Past a recorded error of [`LINEAR_FOLD_EV`], or where the
//! explicit search finds no barrier, [`StrainRateTable::lookup`] says so
//! instead of returning the dipole line as a rate.

use std::fmt;
use std::fs;
use std::path::Path;
use std::str::FromStr;

use ndarray::{Array2, ArrayView1, ArrayView2};

use crate::cell_log::M3;
use crate::error::SaddleError;

/// Absolute error, in eV, past which the dipole line is not a rate.
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

/// Which barrier a lookup reads. The three are never substituted for each other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ensemble {
    /// Fixed cell. Potential-energy difference.
    Helmholtz,
    /// Variable cell. Enthalpy difference.
    Gibbs,
    /// `ΔE0 - (P_saddle - P_minimum) : ε` while that line is inside the fold.
    Linear,
}

/// Where a query sits relative to the recorded dipole error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fold {
    Holds,
    Failed,
    Unchecked,
}

/// Whether the dipole line is still within [`LINEAR_FOLD_EV`] of the fixed-cell barrier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinearStatus {
    Holds,
    Failed,
}

impl LinearStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Holds => "holds",
            Self::Failed => "failed",
        }
    }

    fn parse(text: &str) -> Result<Self, SaddleError> {
        match text {
            "holds" => Ok(Self::Holds),
            "failed" => Ok(Self::Failed),
            _ => Err(SaddleError::Invalid(format!(
                "linear status must be holds or failed, got {text}"
            ))),
        }
    }
}

/// One side of an explicit search, fixed-cell or variable-cell.
#[derive(Clone, Debug, PartialEq)]
pub struct ExplicitSide {
    /// Positive barrier in eV. `None` means the search found no barrier.
    pub barrier: Option<f64>,
    /// Vineyard prefactor in Hz at this geometry. Absent when there is no barrier
    /// or the Hessian was not indexed as a single negative mode.
    pub prefactor_hz: Option<f64>,
    /// Index-check phrase, for example `index 1` or `no barrier`.
    pub index: String,
}

impl ExplicitSide {
    /// A positive barrier and its index-check phrase.
    pub fn barrier(
        barrier: f64,
        prefactor_hz: Option<f64>,
        index: impl Into<String>,
    ) -> Result<Self, SaddleError> {
        if !(barrier.is_finite() && barrier > 0.0) {
            return Err(SaddleError::Invalid(
                "an explicit barrier must be finite and positive".into(),
            ));
        }
        if let Some(hz) = prefactor_hz
            && !(hz.is_finite() && hz > 0.0)
        {
            return Err(SaddleError::Invalid(
                "a Vineyard prefactor must be finite and positive".into(),
            ));
        }
        Ok(Self {
            barrier: Some(barrier),
            prefactor_hz,
            index: index.into(),
        })
    }

    /// The saddle on this path is gone.
    pub fn no_barrier() -> Self {
        Self {
            barrier: None,
            prefactor_hz: None,
            index: "no barrier".to_string(),
        }
    }

    /// `true` when the search did not return a positive barrier.
    pub fn is_no_barrier(&self) -> bool {
        self.barrier.is_none()
    }
}

/// Dipoles of the reference minimum and saddle, and the barrier at zero strain.
#[derive(Clone, Copy, Debug)]
pub struct DipoleModel {
    pub minimum: M3,
    pub saddle: M3,
    pub barrier0: f64,
}

impl DipoleModel {
    /// `P_saddle - P_minimum`.
    pub fn difference(self) -> M3 {
        sub(self.saddle, self.minimum)
    }

    /// `ΔE0 - (P_saddle - P_minimum) : ε`.
    pub fn linear_barrier(self, strain: M3) -> f64 {
        self.barrier0 - double_dot(self.difference(), strain)
    }
}

/// One hop, with its own dipoles. Symmetry copies share `barrier0`.
#[derive(Clone, Debug)]
pub struct HopVariant {
    pub name: String,
    pub minimum: M3,
    pub saddle: M3,
    pub barrier0: f64,
}

impl HopVariant {
    /// `ΔE0 - (P_saddle - P_minimum) : ε` for this hop.
    pub fn linear_barrier(&self, strain: M3) -> f64 {
        self.barrier0 - double_dot(sub(self.saddle, self.minimum), strain)
    }
}

/// One strain and one hop. The three barriers belong to that hop.
#[derive(Clone, Debug, PartialEq)]
pub struct StrainPoint {
    pub direction: String,
    /// `<110>` hop label, for example `110` or `101`. Empty when the table has one hop.
    pub variant: String,
    /// Dumbbell axis of the explicit saddle, Cartesian, not necessarily unit length.
    pub hop: [f64; 3],
    pub strain: M3,
    pub linear: f64,
    pub linear_status: LinearStatus,
    pub helmholtz: ExplicitSide,
    pub gibbs: ExplicitSide,
    /// `|linear - helmholtz|` when the fixed-cell search returned a barrier.
    pub linear_error: Option<f64>,
}

/// Harmonic transition-state rate.
#[derive(Clone, Debug, PartialEq)]
pub struct HarmonicRate {
    pub k_hz: f64,
    pub barrier: f64,
    pub prefactor_hz: f64,
    /// The barrier and prefactor were interpolated along a recorded ray.
    pub interpolated: bool,
    /// The dipole line has been compared with an explicit barrier on this ray.
    pub validated: bool,
    /// Hop that produced this rate. Empty when the table has one hop.
    pub variant: String,
}

/// Result of [`StrainRateTable::lookup`].
#[derive(Clone, Debug, PartialEq)]
pub enum Rate {
    Harmonic(HarmonicRate),
    /// The explicit search found no barrier on this ensemble.
    NoBarrier,
    /// The dipole line is past the fold. `predicted` is not a rate.
    LinearFailed {
        predicted: f64,
    },
}

impl fmt::Display for Rate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Harmonic(rate) => write!(
                f,
                "{:.6e} Hz, barrier {:.8} eV, prefactor {:.6e} Hz{}{}",
                rate.k_hz,
                rate.barrier,
                rate.prefactor_hz,
                if rate.interpolated {
                    ", interpolated"
                } else {
                    ""
                },
                if rate.validated {
                    ""
                } else {
                    ", not validated"
                },
            ),
            Self::NoBarrier => f.write_str("no barrier"),
            Self::LinearFailed { predicted } => {
                write!(f, "linear model failed, predicted {predicted:.8} eV")
            }
        }
    }
}

/// Positive eigenvalues kept for a Vineyard product, after translations are removed.
#[derive(Clone, Debug, PartialEq)]
pub struct TranslationalSplit {
    pub negative: Vec<f64>,
    pub positive: Vec<f64>,
    pub translations: usize,
}

/// Grid of explicit barriers and the dipole line they are compared with.
#[derive(Clone, Debug)]
pub struct StrainRateTable {
    model: DipoleModel,
    variants: Vec<HopVariant>,
    fold: f64,
    activation_volume: Option<M3>,
    points: Vec<StrainPoint>,
}

impl StrainRateTable {
    /// Build a table. Dipoles are symmetrized. Duplicate strains are refused.
    pub fn try_new(
        mut model: DipoleModel,
        points: Vec<StrainPoint>,
        activation_volume: Option<M3>,
    ) -> Result<Self, SaddleError> {
        model.minimum = symmetrize(model.minimum);
        model.saddle = symmetrize(model.saddle);
        if !model.barrier0.is_finite() || !finite_m3(model.minimum) || !finite_m3(model.saddle) {
            return Err(SaddleError::NonFinite("dipole model"));
        }
        if let Some(omega) = activation_volume
            && !finite_m3(omega)
        {
            return Err(SaddleError::NonFinite("activation volume"));
        }
        for (i, point) in points.iter().enumerate() {
            if point.direction.is_empty() || point.direction.contains([',', '"', '\n']) {
                return Err(SaddleError::Invalid(
                    "strain direction must be a non-empty token without commas".into(),
                ));
            }
            for index in [&point.helmholtz.index, &point.gibbs.index] {
                if index.contains([',', '"', '\n']) {
                    return Err(SaddleError::Invalid(
                        "index text must not contain a comma or a quote".into(),
                    ));
                }
            }
            if !finite_m3(point.strain) || !point.linear.is_finite() {
                return Err(SaddleError::NonFinite("strain point"));
            }
            if let Some(error) = point.linear_error
                && !(error.is_finite() && error >= 0.0)
            {
                return Err(SaddleError::Invalid(
                    "linear error must be finite and non-negative".into(),
                ));
            }
            for other in points.iter().take(i) {
                if strain_distance(point.strain, other.strain) < 1e-10
                    && point.variant == other.variant
                {
                    return Err(SaddleError::Invalid(
                        "two strain-table rows have the same strain and hop".into(),
                    ));
                }
            }
        }
        let variants = vec![HopVariant {
            name: "reference".to_string(),
            minimum: model.minimum,
            saddle: model.saddle,
            barrier0: model.barrier0,
        }];
        Ok(Self {
            model,
            variants,
            fold: LINEAR_FOLD_EV,
            activation_volume: activation_volume.map(symmetrize),
            points,
        })
    }

    /// Replace the single reference hop with one dipole per variant.
    ///
    /// `lookup` then returns the lowest barrier among them. A hop whose
    /// explicit search found no barrier makes the result [`Rate::NoBarrier`].
    pub fn with_hop_variants(mut self, variants: Vec<HopVariant>) -> Result<Self, SaddleError> {
        if variants.is_empty() {
            return Err(SaddleError::Invalid(
                "a strain table needs at least one hop".into(),
            ));
        }
        for variant in &variants {
            if variant.name.is_empty() || variant.name.contains([',', '"', '\n']) {
                return Err(SaddleError::Invalid(
                    "hop name must be a non-empty token without commas".into(),
                ));
            }
            if !variant.barrier0.is_finite()
                || !finite_m3(variant.minimum)
                || !finite_m3(variant.saddle)
            {
                return Err(SaddleError::NonFinite("hop variant"));
            }
        }
        self.variants = variants
            .into_iter()
            .map(|mut variant| {
                variant.minimum = symmetrize(variant.minimum);
                variant.saddle = symmetrize(variant.saddle);
                variant
            })
            .collect();
        Ok(self)
    }

    /// Hops the linear model can choose among.
    pub fn variants(&self) -> &[HopVariant] {
        &self.variants
    }

    /// Lowest dipole barrier at `strain`, and the hop that produces it.
    pub fn minimum_linear(&self, strain: M3) -> (String, f64) {
        let mut best_name = self.variants[0].name.clone();
        let mut best = f64::INFINITY;
        for variant in &self.variants {
            let barrier = variant.linear_barrier(strain);
            if barrier < best {
                best = barrier;
                best_name = variant.name.clone();
            }
        }
        (best_name, best)
    }

    /// Reference dipoles and the zero-strain barrier.
    pub fn model(&self) -> DipoleModel {
        self.model
    }

    /// Recorded strains, in insertion order.
    pub fn points(&self) -> &[StrainPoint] {
        &self.points
    }

    /// `Ω = ∂ΔG/∂τ` at the reference state, in Å³, when the host supplied a compliance.
    pub fn activation_volume(&self) -> Option<M3> {
        self.activation_volume
    }

    /// Fold used when the rows were judged, in eV.
    pub fn fold(&self) -> f64 {
        self.fold
    }

    /// Rate at `strain` and `temperature_k` for one ensemble.
    ///
    /// Helmholtz and Gibbs interpolate along a recorded ray and return
    /// [`Rate::NoBarrier`] when a bracketing row has no barrier. They do not
    /// extrapolate. Linear uses the dipole formula and the zero-strain
    /// fixed-cell prefactor. It returns [`Rate::LinearFailed`] once the
    /// recorded absolute error exceeds the fold, or once an explicit
    /// fixed-cell row on that ray has no barrier.
    pub fn lookup(
        &self,
        strain: M3,
        temperature_k: f64,
        ensemble: Ensemble,
    ) -> Result<Rate, SaddleError> {
        if !(temperature_k.is_finite() && temperature_k > 0.0) {
            return Err(SaddleError::Invalid(
                "temperature must be finite and positive".into(),
            ));
        }
        if !finite_m3(strain) {
            return Err(SaddleError::NonFinite("query strain"));
        }
        if ensemble == Ensemble::Linear {
            return self.lookup_linear(strain, temperature_k);
        }
        let exact = self.exact_points(strain);
        if !exact.is_empty() {
            return self.rate_from_exact(&exact, ensemble, temperature_k);
        }
        let hit = self.locate(strain)?;
        let side = match ensemble {
            Ensemble::Helmholtz => &hit.side_helmholtz,
            Ensemble::Gibbs => &hit.side_gibbs,
            Ensemble::Linear => unreachable!("linear is handled above"),
        };
        if side.no_barrier {
            return Ok(Rate::NoBarrier);
        }
        let barrier = side.barrier.ok_or_else(|| {
            SaddleError::Invalid("explicit barrier is missing on a barrier row".into())
        })?;
        let prefactor = side.prefactor_hz.ok_or_else(|| {
            SaddleError::Invalid(
                "Vineyard prefactor was not measured at this explicit strain".into(),
            )
        })?;
        Ok(self.harmonic(
            prefactor,
            barrier,
            hit.interpolated,
            true,
            &hit.variant,
            temperature_k,
        ))
    }

    fn rate_from_exact(
        &self,
        points: &[&StrainPoint],
        ensemble: Ensemble,
        temperature_k: f64,
    ) -> Result<Rate, SaddleError> {
        if points
            .iter()
            .any(|point| explicit_side(point, ensemble).is_no_barrier())
        {
            return Ok(Rate::NoBarrier);
        }
        let mut best: Option<&StrainPoint> = None;
        for point in points {
            let Some(barrier) = explicit_side(point, ensemble).barrier else {
                continue;
            };
            if best.is_none_or(|current| {
                barrier
                    < explicit_side(current, ensemble)
                        .barrier
                        .unwrap_or(f64::INFINITY)
            }) {
                best = Some(*point);
            }
        }
        let point = best.ok_or_else(|| {
            SaddleError::Invalid("explicit barrier is missing on a barrier row".into())
        })?;
        let barrier = explicit_side(point, ensemble)
            .barrier
            .expect("barrier was selected");
        let prefactor = explicit_side(point, ensemble).prefactor_hz.ok_or_else(|| {
            SaddleError::Invalid(
                "Vineyard prefactor was not measured at this explicit strain".into(),
            )
        })?;
        Ok(self.harmonic(
            prefactor,
            barrier,
            false,
            true,
            &point.variant,
            temperature_k,
        ))
    }

    fn harmonic(
        &self,
        prefactor: f64,
        barrier: f64,
        interpolated: bool,
        validated: bool,
        variant: &str,
        temperature_k: f64,
    ) -> Rate {
        Rate::Harmonic(HarmonicRate {
            k_hz: arrhenius(prefactor, barrier, temperature_k),
            barrier,
            prefactor_hz: prefactor,
            interpolated,
            validated,
            variant: variant.to_string(),
        })
    }

    fn exact_points(&self, strain: M3) -> Vec<&StrainPoint> {
        self.points
            .iter()
            .filter(|point| strain_distance(point.strain, strain) < 1e-8)
            .collect()
    }

    /// CSV that round-trips through [`Self::from_csv`].
    pub fn to_csv(&self) -> String {
        let mut lines = vec![
            "record,direction,e00,e01,e02,e10,e11,e12,e20,e21,e22,linear_barrier_eV,linear_status,helmholtz_barrier_eV,gibbs_barrier_eV,vineyard_helmholtz_Hz,vineyard_gibbs_Hz,index_helmholtz,index_gibbs,linear_error_eV,p_xx,p_yy,p_zz,p_yz,p_xz,p_xy,barrier0_eV,omega_xx,omega_yy,omega_zz,omega_yz,omega_xz,omega_xy,variant,hop_x,hop_y,hop_z"
                .to_string(),
        ];
        for variant in &self.variants {
            lines.push(dipole_variant_row(
                "dipole_minimum",
                variant.name.as_str(),
                variant.minimum,
            ));
            lines.push(dipole_variant_row(
                "dipole_saddle",
                variant.name.as_str(),
                variant.saddle,
            ));
        }
        lines.push(scalar_record("barrier0", 26, self.model.barrier0));
        if let Some(omega) = self.activation_volume {
            lines.push(voigt_record("activation_volume", 27, voigt(omega)));
        }
        for point in &self.points {
            lines.push(point_row(point));
        }
        lines.join("\n") + "\n"
    }

    /// Read a table written by [`Self::to_csv`].
    pub fn from_csv(text: &str) -> Result<Self, SaddleError> {
        let mut rows = text.lines().filter(|line| !line.trim().is_empty());
        let header = rows
            .next()
            .ok_or_else(|| SaddleError::Invalid("strain table csv is empty".into()))?;
        let columns: Vec<&str> = header.split(',').collect();
        let col = |name: &str| -> Result<usize, SaddleError> {
            columns
                .iter()
                .position(|item| *item == name)
                .ok_or_else(|| SaddleError::Invalid(format!("csv is missing {name}")))
        };
        let record = col("record")?;
        let direction = col("direction")?;
        let strain_cols = [
            col("e00")?,
            col("e01")?,
            col("e02")?,
            col("e10")?,
            col("e11")?,
            col("e12")?,
            col("e20")?,
            col("e21")?,
            col("e22")?,
        ];
        let linear = col("linear_barrier_eV")?;
        let linear_status = col("linear_status")?;
        let helmholtz = col("helmholtz_barrier_eV")?;
        let gibbs = col("gibbs_barrier_eV")?;
        let vine_h = col("vineyard_helmholtz_Hz")?;
        let vine_g = col("vineyard_gibbs_Hz")?;
        let index_h = col("index_helmholtz")?;
        let index_g = col("index_gibbs")?;
        let error = col("linear_error_eV")?;
        let p_cols = [
            col("p_xx")?,
            col("p_yy")?,
            col("p_zz")?,
            col("p_yz")?,
            col("p_xz")?,
            col("p_xy")?,
        ];
        let barrier0_col = col("barrier0_eV")?;
        let o_cols = [
            col("omega_xx")?,
            col("omega_yy")?,
            col("omega_zz")?,
            col("omega_yz")?,
            col("omega_xz")?,
            col("omega_xy")?,
        ];
        let variant_col = columns.iter().position(|item| *item == "variant");
        let hop_x = columns.iter().position(|item| *item == "hop_x");
        let hop_y = columns.iter().position(|item| *item == "hop_y");
        let hop_z = columns.iter().position(|item| *item == "hop_z");
        let mut minima = Vec::new();
        let mut saddles = Vec::new();
        let mut barrier0 = None;
        let mut activation = None;
        let mut points = Vec::new();
        for (line_no, line) in rows.enumerate() {
            let fields = split_csv(line);
            let take = |index: usize| -> Result<&str, SaddleError> {
                fields.get(index).copied().ok_or_else(|| {
                    SaddleError::Invalid(format!("csv row {} is short", line_no + 2))
                })
            };
            let variant_name = variant_col
                .and_then(|index| fields.get(index).copied())
                .filter(|text| !text.is_empty())
                .unwrap_or("reference")
                .to_string();
            match take(record)? {
                "dipole_minimum" => {
                    minima.push((variant_name, from_voigt(read_voigt(&fields, p_cols)?)));
                }
                "dipole_saddle" => {
                    saddles.push((variant_name, from_voigt(read_voigt(&fields, p_cols)?)));
                }
                "barrier0" => barrier0 = Some(parse_f64(take(barrier0_col)?, "barrier0")?),
                "activation_volume" => {
                    activation = Some(from_voigt(read_voigt(&fields, o_cols)?));
                }
                "point" => {
                    let mut strain = [[0.0; 3]; 3];
                    for (slot, index) in strain_cols.iter().enumerate() {
                        strain[slot / 3][slot % 3] = parse_f64(take(*index)?, "strain")?;
                    }
                    let helmholtz_text = take(helmholtz)?;
                    let gibbs_text = take(gibbs)?;
                    let hop = [
                        optional_col(&fields, hop_x)?,
                        optional_col(&fields, hop_y)?,
                        optional_col(&fields, hop_z)?,
                    ];
                    points.push(StrainPoint {
                        direction: take(direction)?.to_string(),
                        variant: if variant_name == "reference" && variant_col.is_none() {
                            String::new()
                        } else {
                            variant_name
                        },
                        hop,
                        strain,
                        linear: parse_f64(take(linear)?, "linear barrier")?,
                        linear_status: LinearStatus::parse(take(linear_status)?)?,
                        helmholtz: read_side(helmholtz_text, take(vine_h)?, take(index_h)?)?,
                        gibbs: read_side(gibbs_text, take(vine_g)?, take(index_g)?)?,
                        linear_error: optional_f64(take(error)?)?,
                    });
                }
                other => {
                    return Err(SaddleError::Invalid(format!("unknown csv record {other}")));
                }
            }
        }
        let barrier0 =
            barrier0.ok_or_else(|| SaddleError::Invalid("csv is missing barrier0".into()))?;
        if minima.is_empty() || saddles.is_empty() {
            return Err(SaddleError::Invalid("csv is missing a dipole pair".into()));
        }
        let model = DipoleModel {
            minimum: minima[0].1,
            saddle: saddles[0].1,
            barrier0,
        };
        let mut table = Self::try_new(model, points, activation)?;
        if minima.len() > 1 || saddles.len() > 1 {
            let mut hops = Vec::new();
            for (name, minimum) in &minima {
                let saddle = saddles
                    .iter()
                    .find(|(saddle_name, _)| saddle_name == name)
                    .map(|(_, tensor)| *tensor)
                    .ok_or_else(|| {
                        SaddleError::Invalid(format!("csv is missing dipole_saddle for {name}"))
                    })?;
                hops.push(HopVariant {
                    name: name.clone(),
                    minimum: *minimum,
                    saddle,
                    barrier0,
                });
            }
            table = table.with_hop_variants(hops)?;
        }
        Ok(table)
    }

    /// JSON that round-trips through [`Self::from_json`].
    pub fn to_json(&self) -> String {
        let mut out = String::new();
        out.push_str("{\n");
        out.push_str("  \"format\": \"rgsaddle.strain_rate\",\n");
        out.push_str("  \"version\": 1,\n");
        out.push_str(&format!("  \"linear_fold_eV\": {},\n", num(self.fold)));
        out.push_str(&format!(
            "  \"barrier0_eV\": {},\n",
            num(self.model.barrier0)
        ));
        out.push_str("  \"energy_unit\": \"eV\",\n");
        out.push_str("  \"prefactor_unit\": \"Hz\",\n");
        out.push_str("  \"activation_volume_unit\": \"A3\",\n");
        out.push_str(
            "  \"dipole_sign\": \"P = -V (stress_defect - stress_bulk); dE = dE0 - (P_saddle - P_minimum):strain\",\n",
        );
        out.push_str("  \"p_minimum\": ");
        push_matrix(&mut out, self.model.minimum);
        out.push_str(",\n  \"p_saddle\": ");
        push_matrix(&mut out, self.model.saddle);
        out.push_str(",\n  \"activation_volume_A3\": ");
        match self.activation_volume {
            Some(omega) => push_matrix(&mut out, omega),
            None => out.push_str("null"),
        }
        out.push_str(",\n  \"variants\": [\n");
        for (i, variant) in self.variants.iter().enumerate() {
            out.push_str("    {\n");
            out.push_str(&format!("      \"name\": \"{}\",\n", variant.name));
            out.push_str(&format!(
                "      \"barrier0_eV\": {},\n",
                num(variant.barrier0)
            ));
            out.push_str("      \"p_minimum\": ");
            push_matrix(&mut out, variant.minimum);
            out.push_str(",\n      \"p_saddle\": ");
            push_matrix(&mut out, variant.saddle);
            out.push_str("\n    }");
            if i + 1 != self.variants.len() {
                out.push(',');
            }
            out.push('\n');
        }
        out.push_str("  ],\n  \"points\": [\n");
        for (i, point) in self.points.iter().enumerate() {
            out.push_str("    {\n");
            out.push_str(&format!("      \"direction\": \"{}\",\n", point.direction));
            out.push_str(&format!("      \"variant\": \"{}\",\n", point.variant));
            out.push_str(&format!(
                "      \"hop\": [{}, {}, {}],\n",
                num(point.hop[0]),
                num(point.hop[1]),
                num(point.hop[2])
            ));
            out.push_str("      \"strain\": ");
            push_matrix(&mut out, point.strain);
            out.push_str(",\n");
            out.push_str(&format!(
                "      \"linear_barrier_eV\": {},\n",
                num(point.linear)
            ));
            out.push_str(&format!(
                "      \"linear_status\": \"{}\",\n",
                point.linear_status.as_str()
            ));
            push_side(&mut out, "helmholtz", &point.helmholtz);
            push_side(&mut out, "gibbs", &point.gibbs);
            out.push_str("      \"linear_error_eV\": ");
            match point.linear_error {
                Some(error) => out.push_str(&num(error)),
                None => out.push_str("null"),
            }
            out.push_str("\n    }");
            if i + 1 != self.points.len() {
                out.push(',');
            }
            out.push('\n');
        }
        out.push_str("  ]\n}\n");
        out
    }

    /// Read a table written by [`Self::to_json`].
    pub fn from_json(text: &str) -> Result<Self, SaddleError> {
        let value = parse_json(text)?;
        let obj = value
            .as_object()
            .ok_or_else(|| SaddleError::Invalid("strain table json must be an object".into()))?;
        let kind = obj_get(obj, "format").and_then(Json::as_str).unwrap_or("");
        if kind != "rgsaddle.strain_rate" {
            return Err(SaddleError::Invalid(
                "json format is not rgsaddle.strain_rate".into(),
            ));
        }
        let version = obj_get(obj, "version")
            .and_then(Json::as_f64)
            .unwrap_or(0.0);
        if (version - 1.0).abs() > 1e-12 {
            return Err(SaddleError::Invalid(
                "only strain table json version 1 is read".into(),
            ));
        }
        let model = DipoleModel {
            minimum: required_matrix(obj, "p_minimum")?,
            saddle: required_matrix(obj, "p_saddle")?,
            barrier0: required_f64(obj, "barrier0_eV")?,
        };
        let activation = match obj_get(obj, "activation_volume_A3") {
            None | Some(Json::Null) => None,
            Some(value) => Some(value.as_matrix().ok_or_else(|| {
                SaddleError::Invalid("activation_volume_A3 must be a 3x3 matrix".into())
            })?),
        };
        let points_json = obj_get(obj, "points")
            .and_then(Json::as_array)
            .ok_or_else(|| SaddleError::Invalid("json is missing points".into()))?;
        let mut points = Vec::new();
        for item in points_json {
            let point = item
                .as_object()
                .ok_or_else(|| SaddleError::Invalid("a strain point must be an object".into()))?;
            points.push(StrainPoint {
                direction: required_str(point, "direction")?.to_string(),
                variant: obj_get(point, "variant")
                    .and_then(Json::as_str)
                    .unwrap_or("")
                    .to_string(),
                hop: hop_from_json(point)?,
                strain: required_matrix(point, "strain")?,
                linear: required_f64(point, "linear_barrier_eV")?,
                linear_status: LinearStatus::parse(required_str(point, "linear_status")?)?,
                helmholtz: side_from_json(point, "helmholtz")?,
                gibbs: side_from_json(point, "gibbs")?,
                linear_error: match obj_get(point, "linear_error_eV") {
                    None | Some(Json::Null) => None,
                    Some(value) => Some(value.as_f64().ok_or_else(|| {
                        SaddleError::Invalid("linear_error_eV must be a number".into())
                    })?),
                },
            });
        }
        let mut table = Self::try_new(model, points, activation)?;
        if let Some(variants) = obj_get(obj, "variants").and_then(Json::as_array) {
            let mut hops = Vec::new();
            for item in variants {
                let variant = item.as_object().ok_or_else(|| {
                    SaddleError::Invalid("a hop variant must be an object".into())
                })?;
                hops.push(HopVariant {
                    name: required_str(variant, "name")?.to_string(),
                    minimum: required_matrix(variant, "p_minimum")?,
                    saddle: required_matrix(variant, "p_saddle")?,
                    barrier0: obj_get(variant, "barrier0_eV")
                        .and_then(Json::as_f64)
                        .unwrap_or(table.model.barrier0),
                });
            }
            if !hops.is_empty() {
                table = table.with_hop_variants(hops)?;
            }
        }
        Ok(table)
    }

    /// Read `.json` or `.csv`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, SaddleError> {
        let path = path.as_ref();
        let text = fs::read_to_string(path)
            .map_err(|err| SaddleError::Invalid(format!("reading {}: {err}", path.display())))?;
        match path.extension().and_then(|ext| ext.to_str()) {
            Some("json") => Self::from_json(&text),
            Some("csv") => Self::from_csv(&text),
            _ => Err(SaddleError::Invalid(
                "strain table path must end in .json or .csv".into(),
            )),
        }
    }

    /// Write JSON beside the path's extension. `.csv` writes CSV.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), SaddleError> {
        let path = path.as_ref();
        let text = match path.extension().and_then(|ext| ext.to_str()) {
            Some("json") => self.to_json(),
            Some("csv") => self.to_csv(),
            _ => {
                return Err(SaddleError::Invalid(
                    "strain table path must end in .json or .csv".into(),
                ));
            }
        };
        fs::write(path, text)
            .map_err(|err| SaddleError::Invalid(format!("writing {}: {err}", path.display())))
    }

    fn lookup_linear(&self, strain: M3, temperature_k: f64) -> Result<Rate, SaddleError> {
        let (variant, predicted) = self.minimum_linear(strain);
        if !predicted.is_finite() {
            return Err(SaddleError::NonFinite("linear barrier"));
        }
        let prefactor = self.reference_prefactor()?;
        if strain_norm(strain) < 1e-12 {
            return Ok(self.harmonic(prefactor, predicted, false, true, &variant, temperature_k));
        }
        let located = match self.ray(strain) {
            Ok(ray) => ray,
            Err(_) => {
                return Ok(self.harmonic(
                    prefactor,
                    predicted,
                    false,
                    false,
                    &variant,
                    temperature_k,
                ));
            }
        };
        let validated = match self.fold_of(&located) {
            Fold::Failed => {
                return Ok(Rate::LinearFailed { predicted });
            }
            Fold::Holds => true,
            Fold::Unchecked => false,
        };
        Ok(self.harmonic(
            prefactor,
            predicted,
            false,
            validated,
            &variant,
            temperature_k,
        ))
    }

    fn reference_prefactor(&self) -> Result<f64, SaddleError> {
        self.points
            .iter()
            .find(|point| strain_norm(point.strain) < 1e-12)
            .and_then(|point| point.helmholtz.prefactor_hz)
            .ok_or_else(|| {
                SaddleError::Invalid(
                    "the zero-strain row needs a fixed-cell Vineyard prefactor".into(),
                )
            })
    }

    fn fold_of(&self, ray: &Ray) -> Fold {
        let mut samples: Vec<(f64, &StrainPoint)> = ray.samples.clone();
        samples.sort_by(|left, right| {
            left.0
                .partial_cmp(&right.0)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let target = ray.query_scale;
        let mut max_scale = 0.0_f64;
        for (scale, point) in &samples {
            max_scale = max_scale.max(*scale);
            if *scale > 1e-12 && *scale <= target + 1e-12 && point.helmholtz.is_no_barrier() {
                return Fold::Failed;
            }
        }
        if target > max_scale + 1e-8 {
            return Fold::Unchecked;
        }
        let mut lower = None;
        let mut upper = None;
        for (scale, point) in &samples {
            let Some(error) = point.linear_error else {
                continue;
            };
            if *scale <= target + 1e-12 {
                lower = Some((*scale, error));
            }
            if *scale + 1e-12 >= target && upper.is_none() {
                upper = Some((*scale, error));
            }
        }
        match (lower, upper) {
            (Some((left, left_error)), Some((right, right_error))) => {
                let span = right - left;
                let mixed = if span.abs() < 1e-14 {
                    left_error
                } else {
                    let weight = (target - left) / span;
                    left_error + weight * (right_error - left_error)
                };
                if mixed > self.fold {
                    Fold::Failed
                } else {
                    Fold::Holds
                }
            }
            _ => Fold::Unchecked,
        }
    }

    fn locate(&self, strain: M3) -> Result<Located, SaddleError> {
        if let Some(point) = self
            .points
            .iter()
            .find(|point| strain_distance(point.strain, strain) < 1e-8)
        {
            return Ok(Located::exact(point));
        }
        let ray = self.ray(strain)?;
        let helmholtz = interpolate_side(&ray, true)?;
        let gibbs = interpolate_side(&ray, false)?;
        Ok(Located {
            side_helmholtz: helmholtz,
            side_gibbs: gibbs,
            interpolated: true,
            variant: self.minimum_linear(strain).0,
        })
    }

    fn ray(&self, strain: M3) -> Result<Ray, SaddleError> {
        let winner = self.minimum_linear(strain).0;
        let mut best: Option<(f64, String, M3)> = None;
        for point in &self.points {
            if strain_norm(point.strain) < 1e-12 {
                continue;
            }
            if self.variants.len() > 1 && !point.variant.is_empty() && point.variant != winner {
                continue;
            }
            let scale = parallel_scale(strain, point.strain);
            let Some(scale) = scale else {
                continue;
            };
            let residual = strain_distance(strain, scale_m3(point.strain, scale));
            if best.as_ref().is_none_or(|(prev, _, _)| residual < *prev) {
                best = Some((residual, point.direction.clone(), point.strain));
            }
        }
        let Some((residual, direction, basis)) = best else {
            return Err(SaddleError::Invalid(
                "the query strain is not on a recorded ray".into(),
            ));
        };
        if residual > 1e-6 * strain_norm(strain).max(1.0) {
            return Err(SaddleError::Invalid(
                "the query strain is not on a recorded ray".into(),
            ));
        }
        let query_scale = parallel_scale(strain, basis).unwrap_or(1.0);
        let mut samples = Vec::new();
        for point in &self.points {
            if strain_norm(point.strain) < 1e-12 {
                samples.push((0.0, point));
                continue;
            }
            if point.direction != direction {
                continue;
            }
            if self.variants.len() > 1 && !point.variant.is_empty() && point.variant != winner {
                continue;
            }
            if let Some(scale) = parallel_scale(point.strain, basis) {
                samples.push((scale, point));
            }
        }
        if samples.iter().all(|(scale, _)| scale.abs() < 1e-12) {
            return Err(SaddleError::Invalid(
                "the recorded ray has no finite strain".into(),
            ));
        }
        Ok(Ray {
            query_scale,
            samples,
        })
    }
}

struct SideSample {
    no_barrier: bool,
    barrier: Option<f64>,
    prefactor_hz: Option<f64>,
}

struct Located {
    side_helmholtz: SideSample,
    side_gibbs: SideSample,
    interpolated: bool,
    variant: String,
}

impl Located {
    fn exact(point: &StrainPoint) -> Self {
        Self {
            side_helmholtz: side_sample(&point.helmholtz),
            side_gibbs: side_sample(&point.gibbs),
            interpolated: false,
            variant: point.variant.clone(),
        }
    }
}

struct Ray<'a> {
    query_scale: f64,
    samples: Vec<(f64, &'a StrainPoint)>,
}

fn explicit_side(point: &StrainPoint, ensemble: Ensemble) -> &ExplicitSide {
    match ensemble {
        Ensemble::Helmholtz => &point.helmholtz,
        Ensemble::Gibbs => &point.gibbs,
        Ensemble::Linear => unreachable!("linear rows are not an explicit side"),
    }
}

fn side_sample(side: &ExplicitSide) -> SideSample {
    SideSample {
        no_barrier: side.is_no_barrier(),
        barrier: side.barrier,
        prefactor_hz: side.prefactor_hz,
    }
}

fn interpolate_side(ray: &Ray<'_>, helmholtz: bool) -> Result<SideSample, SaddleError> {
    let mut samples: Vec<(f64, &StrainPoint)> = ray.samples.clone();
    samples.sort_by(|left, right| {
        left.0
            .partial_cmp(&right.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let target = ray.query_scale;
    let mut lower: Option<(f64, &StrainPoint)> = None;
    let mut upper: Option<(f64, &StrainPoint)> = None;
    for (scale, point) in &samples {
        if *scale <= target + 1e-10 {
            lower = Some((*scale, *point));
        }
        if *scale >= target - 1e-10 && upper.is_none() {
            upper = Some((*scale, *point));
        }
    }
    let (Some((left_s, left)), Some((right_s, right))) = (lower, upper) else {
        return Err(SaddleError::Invalid(
            "the query strain is outside the recorded ray".into(),
        ));
    };
    let left_side = if helmholtz {
        &left.helmholtz
    } else {
        &left.gibbs
    };
    let right_side = if helmholtz {
        &right.helmholtz
    } else {
        &right.gibbs
    };
    if left_side.is_no_barrier() && (target - left_s).abs() <= 1e-8 {
        return Ok(SideSample {
            no_barrier: true,
            barrier: None,
            prefactor_hz: None,
        });
    }
    if right_side.is_no_barrier() && (target - right_s).abs() <= 1e-8 {
        return Ok(SideSample {
            no_barrier: true,
            barrier: None,
            prefactor_hz: None,
        });
    }
    if left_side.is_no_barrier() || right_side.is_no_barrier() {
        return Err(SaddleError::Invalid(
            "the recorded ray crosses a saddle annihilation; query a recorded strain".into(),
        ));
    }
    let (Some(left_b), Some(right_b)) = (left_side.barrier, right_side.barrier) else {
        return Err(SaddleError::Invalid(
            "a bracketing row is missing its barrier".into(),
        ));
    };
    let (Some(left_p), Some(right_p)) = (left_side.prefactor_hz, right_side.prefactor_hz) else {
        return Err(SaddleError::Invalid(
            "a bracketing row is missing its Vineyard prefactor".into(),
        ));
    };
    let span = right_s - left_s;
    let weight = if span.abs() < 1e-14 {
        0.0
    } else {
        (target - left_s) / span
    };
    let barrier = left_b + weight * (right_b - left_b);
    let prefactor = (left_p.ln() + weight * (right_p.ln() - left_p.ln())).exp();
    Ok(SideSample {
        no_barrier: false,
        barrier: Some(barrier),
        prefactor_hz: Some(prefactor),
    })
}

/// Judge the dipole line against one explicit fixed-cell barrier.
pub fn judge_linear(linear: f64, explicit: Option<f64>, fold: f64) -> (LinearStatus, Option<f64>) {
    match explicit {
        Some(value) => {
            let error = (linear - value).abs();
            let status = if error > fold {
                LinearStatus::Failed
            } else {
                LinearStatus::Holds
            };
            (status, Some(error))
        }
        None => (LinearStatus::Failed, None),
    }
}

/// Right strain with `ε_yx = gamma` and every other entry zero.
///
/// This is the simple shear of the copper fixture: `h = h0 (I + ε)`.
pub fn shear_yx(gamma: f64) -> M3 {
    [[0.0, 0.0, 0.0], [gamma, 0.0, 0.0], [0.0, 0.0, 0.0]]
}

/// Right strain with `ε_zx = gamma` and every other entry zero.
pub fn shear_xz(gamma: f64) -> M3 {
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [gamma, 0.0, 0.0]]
}

/// Uniaxial extension along x.
pub fn uniaxial_x(eps: f64) -> M3 {
    [[eps, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
}

/// Isotropic extension `ε I`.
pub fn hydrostatic(eps: f64) -> M3 {
    [[eps, 0.0, 0.0], [0.0, eps, 0.0], [0.0, 0.0, eps]]
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

/// Symmetric tensor components in [`VOIGT`] order.
///
/// Off-diagonal entries are the average of the two matrix entries.
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

/// Quarter turns about x. `1` sends `[110]` to `[101]`.
pub fn rotation_x(quarter_turns: i32) -> M3 {
    match quarter_turns.rem_euclid(4) {
        0 => [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        1 => [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        2 => [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        _ => [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
    }
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
/// engineering shear. `Ω : τ` uses the same nine-entry sum as [`double_dot`].
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

fn scale_m3(m: M3, scale: f64) -> M3 {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = scale * m[i][j];
        }
    }
    out
}

fn finite_m3(m: M3) -> bool {
    m.iter().flatten().all(|value| value.is_finite())
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

fn strain_norm(m: M3) -> f64 {
    strain_distance(m, [[0.0; 3]; 3])
}

fn parallel_scale(query: M3, basis: M3) -> Option<f64> {
    let mut basis_norm2 = 0.0;
    let mut dot = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            basis_norm2 += basis[i][j] * basis[i][j];
            dot += query[i][j] * basis[i][j];
        }
    }
    if basis_norm2 < 1e-24 {
        return None;
    }
    let scale = dot / basis_norm2;
    let residual = strain_distance(query, scale_m3(basis, scale));
    let scale_ref = strain_norm(query).max(strain_norm(basis));
    if residual <= 1e-6 * scale_ref.max(1e-8) {
        Some(scale)
    } else {
        None
    }
}

fn arrhenius(prefactor_hz: f64, barrier: f64, temperature_k: f64) -> f64 {
    let exponent = -barrier / (BOLTZMANN_EV_PER_K * temperature_k);
    if exponent < -700.0 {
        return 0.0;
    }
    prefactor_hz * exponent.exp()
}

fn num(value: f64) -> String {
    if value == 0.0 {
        return "0".to_string();
    }
    format!("{value}")
}

const CSV_COLUMNS: usize = 37;

fn dipole_variant_row(record: &str, variant: &str, dipole: M3) -> String {
    let mut fields = blank_record(record);
    let values = voigt(dipole);
    for (offset, value) in values.iter().enumerate() {
        fields[20 + offset] = num(*value);
    }
    fields[33] = variant.to_string();
    fields.join(",")
}

fn blank_record(record: &str) -> Vec<String> {
    let mut fields = vec![String::new(); CSV_COLUMNS];
    fields[0] = record.to_string();
    fields
}

fn voigt_record(record: &str, start: usize, values: [f64; 6]) -> String {
    let mut fields = blank_record(record);
    for (offset, value) in values.iter().enumerate() {
        fields[start + offset] = num(*value);
    }
    fields.join(",")
}

fn scalar_record(record: &str, column: usize, value: f64) -> String {
    let mut fields = blank_record(record);
    fields[column] = num(value);
    fields.join(",")
}

fn point_row(point: &StrainPoint) -> String {
    let mut fields = Vec::new();
    fields.push("point".to_string());
    fields.push(point.direction.clone());
    for i in 0..3 {
        for j in 0..3 {
            fields.push(num(point.strain[i][j]));
        }
    }
    fields.push(num(point.linear));
    fields.push(point.linear_status.as_str().to_string());
    fields.push(barrier_cell(&point.helmholtz));
    fields.push(barrier_cell(&point.gibbs));
    fields.push(optional_num(point.helmholtz.prefactor_hz));
    fields.push(optional_num(point.gibbs.prefactor_hz));
    fields.push(point.helmholtz.index.clone());
    fields.push(point.gibbs.index.clone());
    fields.push(optional_num(point.linear_error));
    while fields.len() < 33 {
        fields.push(String::new());
    }
    fields.push(point.variant.clone());
    fields.push(num(point.hop[0]));
    fields.push(num(point.hop[1]));
    fields.push(num(point.hop[2]));
    fields.join(",")
}

fn barrier_cell(side: &ExplicitSide) -> String {
    match side.barrier {
        Some(value) => num(value),
        None => "no barrier".to_string(),
    }
}

fn optional_col(fields: &[&str], index: Option<usize>) -> Result<f64, SaddleError> {
    let Some(index) = index else {
        return Ok(0.0);
    };
    let text = fields.get(index).copied().unwrap_or("");
    if text.is_empty() {
        Ok(0.0)
    } else {
        parse_f64(text, "hop")
    }
}

fn optional_num(value: Option<f64>) -> String {
    match value {
        Some(value) => num(value),
        None => String::new(),
    }
}

fn read_side(barrier: &str, prefactor: &str, index: &str) -> Result<ExplicitSide, SaddleError> {
    if barrier.is_empty() || barrier == "no barrier" {
        return Ok(ExplicitSide {
            barrier: None,
            prefactor_hz: None,
            index: if index.is_empty() {
                "no barrier".to_string()
            } else {
                index.to_string()
            },
        });
    }
    ExplicitSide::barrier(
        parse_f64(barrier, "barrier")?,
        optional_f64(prefactor)?,
        index,
    )
}

fn read_voigt(fields: &[&str], columns: [usize; 6]) -> Result<[f64; 6], SaddleError> {
    let mut out = [0.0; 6];
    for (slot, index) in columns.iter().enumerate() {
        let text = fields.get(*index).copied().unwrap_or("");
        out[slot] = parse_f64(text, "voigt")?;
    }
    Ok(out)
}

fn optional_f64(text: &str) -> Result<Option<f64>, SaddleError> {
    if text.is_empty() {
        Ok(None)
    } else {
        Ok(Some(parse_f64(text, "number")?))
    }
}

fn parse_f64(text: &str, what: &str) -> Result<f64, SaddleError> {
    let value = f64::from_str(text.trim())
        .map_err(|_| SaddleError::Invalid(format!("bad {what}: {text}")))?;
    if !value.is_finite() {
        return Err(SaddleError::NonFinite("csv number"));
    }
    Ok(value)
}

fn split_csv(line: &str) -> Vec<&str> {
    line.split(',').collect()
}

fn push_matrix(out: &mut String, m: M3) {
    out.push('[');
    for (i, row) in m.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push('[');
        for (j, value) in row.iter().enumerate() {
            if j > 0 {
                out.push_str(", ");
            }
            out.push_str(&num(*value));
        }
        out.push(']');
    }
    out.push(']');
}

fn push_side(out: &mut String, name: &str, side: &ExplicitSide) {
    out.push_str(&format!("      \"{name}_barrier_eV\": "));
    match side.barrier {
        Some(value) => out.push_str(&num(value)),
        None => out.push_str("null"),
    }
    out.push_str(",\n");
    out.push_str(&format!("      \"{name}_prefactor_Hz\": "));
    match side.prefactor_hz {
        Some(value) => out.push_str(&num(value)),
        None => out.push_str("null"),
    }
    out.push_str(",\n");
    out.push_str(&format!("      \"{name}_index\": \"{}\",\n", side.index));
}

#[allow(clippy::needless_range_loop)]
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
            Self::Str(value) => Some(value),
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

fn hop_from_json(obj: &[(String, Json)]) -> Result<[f64; 3], SaddleError> {
    match obj_get(obj, "hop") {
        None | Some(Json::Null) => Ok([0.0; 3]),
        Some(value) => {
            let items = value
                .as_array()
                .ok_or_else(|| SaddleError::Invalid("hop must be three numbers".into()))?;
            if items.len() != 3 {
                return Err(SaddleError::Invalid("hop must be three numbers".into()));
            }
            let mut hop = [0.0; 3];
            for (slot, item) in items.iter().enumerate() {
                hop[slot] = item
                    .as_f64()
                    .ok_or_else(|| SaddleError::Invalid("hop must be three numbers".into()))?;
            }
            Ok(hop)
        }
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
        .ok_or_else(|| SaddleError::Invalid(format!("json is missing {key}")))
}

fn required_str<'a>(obj: &'a [(String, Json)], key: &str) -> Result<&'a str, SaddleError> {
    obj_get(obj, key)
        .and_then(Json::as_str)
        .ok_or_else(|| SaddleError::Invalid(format!("json is missing {key}")))
}

fn required_matrix(obj: &[(String, Json)], key: &str) -> Result<M3, SaddleError> {
    obj_get(obj, key)
        .and_then(Json::as_matrix)
        .ok_or_else(|| SaddleError::Invalid(format!("json is missing matrix {key}")))
}

fn side_from_json(obj: &[(String, Json)], name: &str) -> Result<ExplicitSide, SaddleError> {
    let barrier_key = format!("{name}_barrier_eV");
    let prefactor_key = format!("{name}_prefactor_Hz");
    let index_key = format!("{name}_index");
    let index = required_str(obj, &index_key)?.to_string();
    let barrier = match obj_get(obj, &barrier_key) {
        None | Some(Json::Null) => None,
        Some(value) => Some(
            value
                .as_f64()
                .ok_or_else(|| SaddleError::Invalid(format!("{barrier_key} must be a number")))?,
        ),
    };
    let prefactor =
        match obj_get(obj, &prefactor_key) {
            None | Some(Json::Null) => None,
            Some(value) => Some(value.as_f64().ok_or_else(|| {
                SaddleError::Invalid(format!("{prefactor_key} must be a number"))
            })?),
        };
    match barrier {
        Some(value) => ExplicitSide::barrier(value, prefactor, index),
        None => Ok(ExplicitSide {
            barrier: None,
            prefactor_hz: None,
            index,
        }),
    }
}

struct Parser<'a> {
    text: &'a [u8],
    index: usize,
}

fn parse_json(text: &str) -> Result<Json, SaddleError> {
    let mut parser = Parser {
        text: text.as_bytes(),
        index: 0,
    };
    let value = parser.value()?;
    parser.skip();
    if parser.index != parser.text.len() {
        return Err(SaddleError::Invalid(
            "trailing data after the strain table json".into(),
        ));
    }
    Ok(value)
}

impl<'a> Parser<'a> {
    fn skip(&mut self) {
        while let Some(byte) = self.text.get(self.index)
            && byte.is_ascii_whitespace()
        {
            self.index += 1;
        }
    }

    fn peek(&mut self) -> Result<u8, SaddleError> {
        self.skip();
        self.text
            .get(self.index)
            .copied()
            .ok_or_else(|| SaddleError::Invalid("unexpected end of strain table json".into()))
    }

    fn bump(&mut self) -> Result<u8, SaddleError> {
        let byte = self.peek()?;
        self.index += 1;
        Ok(byte)
    }

    fn value(&mut self) -> Result<Json, SaddleError> {
        match self.peek()? {
            b'n' => self.null(),
            b'"' => Ok(Json::Str(self.string()?)),
            b'[' => self.array(),
            b'{' => self.object(),
            b'-' | b'0'..=b'9' => Ok(Json::Num(self.number()?)),
            other => Err(SaddleError::Invalid(format!(
                "unexpected json byte {other}"
            ))),
        }
    }

    fn null(&mut self) -> Result<Json, SaddleError> {
        for expected in b"null" {
            if self.bump()? != *expected {
                return Err(SaddleError::Invalid("bad json null".into()));
            }
        }
        Ok(Json::Null)
    }

    fn array(&mut self) -> Result<Json, SaddleError> {
        self.bump()?;
        let mut items = Vec::new();
        if self.peek()? == b']' {
            self.bump()?;
            return Ok(Json::Arr(items));
        }
        loop {
            items.push(self.value()?);
            match self.bump()? {
                b']' => return Ok(Json::Arr(items)),
                b',' => continue,
                _ => {
                    return Err(SaddleError::Invalid(
                        "json array expected a comma or a closing bracket".into(),
                    ));
                }
            }
        }
    }

    fn object(&mut self) -> Result<Json, SaddleError> {
        self.bump()?;
        let mut pairs = Vec::new();
        if self.peek()? == b'}' {
            self.bump()?;
            return Ok(Json::Obj(pairs));
        }
        loop {
            if self.peek()? != b'"' {
                return Err(SaddleError::Invalid(
                    "json object key must be a string".into(),
                ));
            }
            let key = self.string()?;
            if self.bump()? != b':' {
                return Err(SaddleError::Invalid("json object expected a colon".into()));
            }
            pairs.push((key, self.value()?));
            match self.bump()? {
                b'}' => return Ok(Json::Obj(pairs)),
                b',' => continue,
                _ => {
                    return Err(SaddleError::Invalid(
                        "json object expected a comma or a closing brace".into(),
                    ));
                }
            }
        }
    }

    fn string(&mut self) -> Result<String, SaddleError> {
        if self.bump()? != b'"' {
            return Err(SaddleError::Invalid("json string expected a quote".into()));
        }
        let mut out = String::new();
        loop {
            let byte = self
                .text
                .get(self.index)
                .copied()
                .ok_or_else(|| SaddleError::Invalid("unterminated json string".into()))?;
            self.index += 1;
            match byte {
                b'"' => return Ok(out),
                b'\\' => {
                    let escaped = self
                        .text
                        .get(self.index)
                        .copied()
                        .ok_or_else(|| SaddleError::Invalid("bad json escape".into()))?;
                    self.index += 1;
                    match escaped {
                        b'"' | b'\\' => out.push(escaped as char),
                        b'n' => out.push('\n'),
                        _ => {
                            return Err(SaddleError::Invalid("unsupported json escape".into()));
                        }
                    }
                }
                byte if byte < 0x80 => out.push(byte as char),
                _ => {
                    return Err(SaddleError::Invalid(
                        "strain table strings are ascii".into(),
                    ));
                }
            }
        }
    }

    fn number(&mut self) -> Result<f64, SaddleError> {
        self.skip();
        let start = self.index;
        let mut saw_digit = false;
        if self.text.get(self.index) == Some(&b'-') {
            self.index += 1;
        }
        while self
            .text
            .get(self.index)
            .is_some_and(|byte| byte.is_ascii_digit())
        {
            saw_digit = true;
            self.index += 1;
        }
        if self.text.get(self.index) == Some(&b'.') {
            self.index += 1;
            while self
                .text
                .get(self.index)
                .is_some_and(|byte| byte.is_ascii_digit())
            {
                saw_digit = true;
                self.index += 1;
            }
        }
        if matches!(self.text.get(self.index), Some(b'e' | b'E')) {
            self.index += 1;
            if matches!(self.text.get(self.index), Some(b'+' | b'-')) {
                self.index += 1;
            }
            while self
                .text
                .get(self.index)
                .is_some_and(|byte| byte.is_ascii_digit())
            {
                self.index += 1;
            }
        }
        if !saw_digit {
            return Err(SaddleError::Invalid("bad json number".into()));
        }
        let text = std::str::from_utf8(&self.text[start..self.index])
            .map_err(|_| SaddleError::Invalid("bad json number".into()))?;
        f64::from_str(text).map_err(|_| SaddleError::Invalid(format!("bad json number {text}")))
    }
}
