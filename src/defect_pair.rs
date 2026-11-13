//! Saddles and finite-temperature rates of a defect pair.
//!
//! The caller supplies a potential, a periodic cell, one relaxed
//! configuration, and a catalogue of hops (the product geometry of each
//! symmetry-distinct move). Nothing here is specific to a material. For
//! each hop the search
//!
//! 1. minimizes the reactant and the product,
//! 2. climbs from the reactant side of that hop with the solid-state dimer
//!    ([`crate::SolidDimer`]), on a frozen cell or on the joint chart,
//! 3. requires one negative projected eigenvalue from [`crate::index_at`],
//! 4. pushes off both sides of the unstable mode, minimizes, and keeps the
//!    hop only when those minima are the reactant and the product,
//! 5. builds a climbing-image solid-state NEB through that saddle, in two
//!    phases: a plain relaxation, then the climb,
//! 6. integrates the projected average force along that path
//!    ([`crate::pafi`]) and shoots the Bennett–Chandler factor
//!    ([`crate::bennett`]) from the free-energy maximum.
//!
//! The rate at temperature `T` is
//!
//! `k(T) = κ ν exp(-ΔF / kT)`.
//!
//! `ΔF` is the PAFI potential of mean force from the reactant hyperplane
//! to the dividing surface. `ν` is the reactant attempt frequency along
//! the path tangent, `ν = (ω / 2π)` with `ω = sqrt(τ · H τ / m)`. `κ` is
//! the Bennett–Chandler plateau. In the harmonic limit `ΔF` holds the
//! transverse entropy and this expression agrees with Vineyard's rate
//! when `κ = 1`. The Vineyard rate `ν_h exp(-ΔE0 / kT)` is reported beside
//! it. Vineyard, J. Phys. Chem. Solids 3, 121 (1957),
//! doi:10.1016/0022-3697(57)90059-8.
//!
//! A hop is `toward`, `away`, or `tangential` by the change in the
//! minimum-image separation of the mobile-defect centroid from the fixed
//! defect. The shell label is the caller's.
//!
//! [`RateCatalogue`] is the document `rgsaddle.defect_pair_rates`, version
//! 1. An amsel absorbing-Markov step reads `rate_Hz` as the first-order
//! exit rate of one channel. The channel key is
//! `(shell, direction, hop, temperature, cell_repeats)`.

use ndarray::{Array1, Array2, ArrayView1, s};
use rgmin::{FireKind, Method};

use crate::band::{BandConfig, BandStatus, CiConfig};
use crate::bennett::{self, ShootingConfig, Transmission};
use crate::cell_log::{M3, m3_det, m3_inv, m3_mul, m3_to_row9};
use crate::eigensolve::exact_eigh;
use crate::error::SaddleError;
use crate::minmode::{MinModeStatus, PointSurface};
use crate::pafi::{self, ArcPath, FreeEnergyProfile, HyperplaneAverage, PafiConfig};
use crate::solid_dimer::{
    CellMask, JointChart, SolidDimer, SolidDimerConfig, finite_difference_cauchy_stress,
};
use crate::ssneb::SolidNeb;
use crate::strain_rate::{
    self, ElasticDipole, Json, double_dot, obj_get, parse_json, required_f64,
    required_matrix, required_str, split_translational_modes,
};

/// Which way the mobile defect moves relative to the fixed defect.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DirectionClass {
    Toward,
    Away,
    Tangential,
}

impl DirectionClass {
    /// `toward`, `away`, or `tangential`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Toward => "toward",
            Self::Away => "away",
            Self::Tangential => "tangential",
        }
    }

    fn parse(text: &str) -> Result<Self, SaddleError> {
        match text {
            "toward" => Ok(Self::Toward),
            "away" => Ok(Self::Away),
            "tangential" => Ok(Self::Tangential),
            _ => Err(SaddleError::Invalid(
                "direction must be toward, away, or tangential".into(),
            )),
        }
    }
}

/// Classify a hop by the change in separation.
///
/// `after - before < -tol` is toward the fixed defect, `> tol` is away,
/// and the rest is tangential.
pub fn classify_separation(
    before: f64,
    after: f64,
    tol: f64,
) -> Result<DirectionClass, SaddleError> {
    if !(before.is_finite() && after.is_finite() && tol.is_finite() && tol >= 0.0) {
        return Err(SaddleError::Invalid(
            "separation and tolerance must be finite, and the tolerance non-negative".into(),
        ));
    }
    let delta = after - before;
    if delta < -tol {
        Ok(DirectionClass::Toward)
    } else if delta > tol {
        Ok(DirectionClass::Away)
    } else {
        Ok(DirectionClass::Tangential)
    }
}

/// Minimum-image separation between `fixed` and the centroid of `mobile`.
pub fn pair_separation(
    positions: ArrayView1<f64>,
    cell: M3,
    fixed: [f64; 3],
    mobile: &[usize],
) -> Result<f64, SaddleError> {
    let center = mobile_center(positions, cell, mobile)?;
    Ok(norm3(mic(sub3(center, fixed), cell)?))
}

/// One symmetry-distinct hop. `product` is a Cartesian geometry of the
/// same length as the reactant, before it is minimized.
#[derive(Clone, Debug)]
pub struct HopTemplate {
    pub name: String,
    pub product: Array1<f64>,
    /// Start the dimer here when the caller already has a saddle guess.
    pub saddle: Option<Array1<f64>>,
    /// Initial unstable direction. Used with [`Self::saddle`].
    pub mode: Option<Array1<f64>>,
    /// Caller label for the mobile-defect orientation. Empty if it has none.
    pub orientation: String,
    /// Caller label for the product basin.
    pub final_state: String,
}

/// The relaxed pair and the hops to search.
#[derive(Clone, Debug)]
pub struct PairSetup {
    pub positions: Array1<f64>,
    pub cell: M3,
    /// Fixed defect, Cartesian, in the same frame as `positions`.
    pub fixed: [f64; 3],
    /// Atoms whose centroid is the mobile defect.
    pub mobile: Vec<usize>,
    /// Caller label, stored on every row. A shell index or a distance.
    pub shell: String,
    /// Cell repeats, stored on every row. Zero if the cell is not a cube of repeats.
    pub cell_repeats: u32,
    pub hops: Vec<HopTemplate>,
}

/// Frozen cell, or the joint chart with every lower-triangular strain free.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CellMode {
    Frozen,
    Variable,
}

/// Search, sampling, and the unit system of the rate.
///
/// `temperatures` are in the unit implied by `boltzmann`: kelvin when
/// `boltzmann` is [`crate::BOLTZMANN_EV_PER_K`]. `mass` and `angular_scale`
/// convert `sqrt(λ / mass)` into a frequency. For eV, Å and amu,
/// `angular_scale` is [`crate::angular_scale_rad_s`].
#[derive(Clone, Debug)]
pub struct DefectPairConfig {
    pub dimer: SolidDimerConfig,
    pub cell_mode: CellMode,
    pub external_stress: M3,
    /// Subtracted in `P = -V (σ - σ_bulk)`. Zero if the perfect cell is unstressed.
    pub bulk_stress: M3,
    pub band: BandConfig,
    pub images: usize,
    pub max_dimer_steps: usize,
    pub max_band_steps: usize,
    pub max_relax_steps: usize,
    pub relax_tol: f64,
    pub relax_step: f64,
    pub max_move: f64,
    pub seed_fraction: f64,
    pub push: f64,
    pub connect_rmsd: f64,
    pub direction_tol: f64,
    /// `thermal` is replaced by `boltzmann * T` for each temperature.
    pub pafi: PafiConfig,
    /// `thermal` is replaced the same way.
    pub shooting: ShootingConfig,
    pub temperatures: Vec<f64>,
    pub boltzmann: f64,
    pub mass: f64,
    pub angular_scale: f64,
    pub harmonic_step: f64,
    /// Atoms in the harmonic Hessian. `None` is the whole cell.
    pub harmonic_atoms: Option<Vec<usize>>,
    /// Hyperplanes, including the ends. Zero uses every NEB image.
    pub planes: usize,
    /// When false, the search still reports ΔE0, the band and the
    /// zero-temperature integral, and does not sample a thermal profile.
    pub finite_temperature: bool,
    /// A converged path whose zero-temperature PAFI integral misses `dE0`
    /// by more than this does not receive a finite-temperature rate.
    pub zero_t_tol: f64,
}

impl Default for DefectPairConfig {
    fn default() -> Self {
        Self {
            dimer: SolidDimerConfig::frozen(crate::MinModeConfig::default())
                .with_index_check(crate::IndexCheck::first_order()),
            cell_mode: CellMode::Frozen,
            external_stress: [[0.0; 3]; 3],
            bulk_stress: [[0.0; 3]; 3],
            band: BandConfig {
                min_mode: None,
                autoneb: None,
                zoom: None,
                climbing: Some(CiConfig::always()),
                ..BandConfig::default()
            },
            images: 7,
            max_dimer_steps: 80,
            max_band_steps: 80,
            max_relax_steps: 80,
            relax_tol: 1.0e-3,
            relax_step: 0.1,
            max_move: 0.08,
            seed_fraction: 0.35,
            push: 0.15,
            connect_rmsd: 0.3,
            direction_tol: 0.05,
            pafi: PafiConfig::default(),
            shooting: ShootingConfig::default(),
            temperatures: vec![300.0],
            boltzmann: strain_rate::BOLTZMANN_EV_PER_K,
            mass: 1.0,
            angular_scale: strain_rate::angular_scale_rad_s(),
            harmonic_step: 1.0e-3,
            harmonic_atoms: None,
            planes: 0,
            finite_temperature: true,
            zero_t_tol: 0.02,
        }
    }
}

/// Why a hop did not receive a rate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HopOutcome {
    Rated,
    NotConverged,
    NotIndex1,
    Disconnected,
}

impl HopOutcome {
    /// `rated`, `not_converged`, `not_index1`, or `disconnected`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Rated => "rated",
            Self::NotConverged => "not_converged",
            Self::NotIndex1 => "not_index1",
            Self::Disconnected => "disconnected",
        }
    }

    fn parse(text: &str) -> Result<Self, SaddleError> {
        match text {
            "rated" => Ok(Self::Rated),
            "not_converged" => Ok(Self::NotConverged),
            "not_index1" => Ok(Self::NotIndex1),
            "disconnected" => Ok(Self::Disconnected),
            _ => Err(SaddleError::Invalid("unknown hop outcome".into())),
        }
    }
}

/// One channel at one temperature.
#[derive(Clone, Debug)]
pub struct RateEntry {
    pub shell: String,
    pub direction: DirectionClass,
    pub hop: String,
    pub temperature: f64,
    pub cell_repeats: u32,
    pub n_atoms: usize,
    pub separation_before: f64,
    pub separation_after: f64,
    pub outcome: HopOutcome,
    pub connected: bool,
    pub index1: bool,
    pub band_converged: bool,
    pub reactant_energy: f64,
    pub product_energy: f64,
    pub saddle_energy: f64,
    /// Potential, or enthalpy when an external stress is set.
    pub d_e0: f64,
    pub d_f: Option<f64>,
    pub d_f_error: Option<f64>,
    /// `ΔE0 - kT ln(ν_h / ν)`.
    pub d_f_harmonic: Option<f64>,
    /// Attempt frequency along the path tangent, in the `angular_scale` unit.
    pub nu: Option<f64>,
    /// Vineyard prefactor in the same unit.
    pub vineyard: Option<f64>,
    pub kappa: Option<f64>,
    pub kappa_error: Option<f64>,
    pub harmonic_rate: Option<f64>,
    pub rate: Option<f64>,
    /// `∫ (∇E · n) ds` with the system held on the path. On a minimum-energy
    /// path this is `dE0`.
    pub zero_t: Option<f64>,
    /// Max band force, in energy per length, after the climbing phase.
    pub band_force: f64,
    /// Recorded PAFI samples, summed over planes.
    pub pafi_samples: usize,
    /// Force evaluations spent on the PAFI sampling.
    pub pafi_evaluations: usize,
    pub saddle_stress: M3,
    pub saddle_dipole: M3,
    pub negative_eigenvalue: Option<f64>,
    /// Mobile-defect orientation, as supplied on the hop.
    pub orientation: String,
    /// Product basin, as supplied on the hop.
    pub final_state: String,
}

/// The document written by [`defect_pair_rates`].
#[derive(Clone, Debug)]
pub struct RateCatalogue {
    pub entries: Vec<RateEntry>,
}

impl RateCatalogue {
    /// Append every entry of `other`.
    pub fn merge(&mut self, other: RateCatalogue) {
        self.entries.extend(other.entries);
    }

    /// `rgsaddle.defect_pair_rates` version 1.
    pub fn to_json(&self) -> String {
        let mut out = String::new();
        out.push_str("{\n");
        out.push_str("  \"format\": \"rgsaddle.defect_pair_rates\",\n");
        out.push_str("  \"version\": 1,\n");
        out.push_str("  \"rate\": \"k = kappa * nu * exp(-dF/kT)\",\n");
        out.push_str("  \"nu\": \"(1/2pi) * sqrt((tau·H·tau)/mass) * angular_scale\",\n");
        out.push_str(
            "  \"dF\": \"PAFI potential of mean force from the reactant to the dividing surface\",\n",
        );
        out.push_str(
            "  \"kappa\": \"Bennett-Chandler positive-flux transmission coefficient\",\n",
        );
        out.push_str(
            "  \"harmonic_rate\": \"Vineyard prefactor * exp(-dE0/kT)\",\n",
        );
        out.push_str("  \"entries\": [\n");
        for (i, entry) in self.entries.iter().enumerate() {
            out.push_str("    {\n");
            push_kv(&mut out, "shell", &json_string(&entry.shell), true);
            push_kv(&mut out, "direction", &json_string(entry.direction.as_str()), true);
            push_kv(&mut out, "hop", &json_string(&entry.hop), true);
            push_kv(&mut out, "temperature", &num(entry.temperature), true);
            push_kv(&mut out, "cell_repeats", &entry.cell_repeats.to_string(), true);
            push_kv(&mut out, "n_atoms", &entry.n_atoms.to_string(), true);
            push_kv(&mut out, "separation_before", &num(entry.separation_before), true);
            push_kv(&mut out, "separation_after", &num(entry.separation_after), true);
            push_kv(&mut out, "outcome", &json_string(entry.outcome.as_str()), true);
            push_kv(&mut out, "connected", if entry.connected { "true" } else { "false" }, true);
            push_kv(&mut out, "index1", if entry.index1 { "true" } else { "false" }, true);
            push_kv(
                &mut out,
                "band_converged",
                if entry.band_converged { "true" } else { "false" },
                true,
            );
            push_kv(&mut out, "reactant_energy", &num(entry.reactant_energy), true);
            push_kv(&mut out, "product_energy", &num(entry.product_energy), true);
            push_kv(&mut out, "saddle_energy", &num(entry.saddle_energy), true);
            push_kv(&mut out, "dE0", &num(entry.d_e0), true);
            push_kv(&mut out, "dF", &opt_num(entry.d_f), true);
            push_kv(&mut out, "dF_error", &opt_num(entry.d_f_error), true);
            push_kv(&mut out, "dF_harmonic", &opt_num(entry.d_f_harmonic), true);
            push_kv(&mut out, "nu", &opt_num(entry.nu), true);
            push_kv(&mut out, "vineyard", &opt_num(entry.vineyard), true);
            push_kv(&mut out, "kappa", &opt_num(entry.kappa), true);
            push_kv(&mut out, "kappa_error", &opt_num(entry.kappa_error), true);
            push_kv(&mut out, "harmonic_rate", &opt_num(entry.harmonic_rate), true);
            push_kv(&mut out, "rate_Hz", &opt_num(entry.rate), true);
            push_kv(&mut out, "zero_T", &opt_num(entry.zero_t), true);
            push_kv(&mut out, "band_force", &num(entry.band_force), true);
            push_kv(&mut out, "pafi_samples", &entry.pafi_samples.to_string(), true);
            push_kv(
                &mut out,
                "pafi_evaluations",
                &entry.pafi_evaluations.to_string(),
                true,
            );
            push_kv(&mut out, "saddle_stress", &matrix_json(entry.saddle_stress), true);
            push_kv(&mut out, "saddle_dipole", &matrix_json(entry.saddle_dipole), true);
            push_kv(
                &mut out,
                "negative_eigenvalue",
                &opt_num(entry.negative_eigenvalue),
                true,
            );
            push_kv(&mut out, "orientation", &json_string(&entry.orientation), true);
            push_kv(&mut out, "final_state", &json_string(&entry.final_state), false);
            out.push_str("    }");
            if i + 1 != self.entries.len() {
                out.push(',');
            }
            out.push('\n');
        }
        out.push_str("  ]\n}\n");
        out
    }

    /// Parse a document this crate wrote.
    pub fn from_json(text: &str) -> Result<Self, SaddleError> {
        let value = parse_json(text)?;
        let obj = value.as_object().ok_or_else(|| {
            SaddleError::Invalid("defect-pair catalogue must be an object".into())
        })?;
        let kind = obj_get(obj, "format").and_then(Json::as_str).unwrap_or("");
        if kind != "rgsaddle.defect_pair_rates" {
            return Err(SaddleError::Invalid(
                "json format is not rgsaddle.defect_pair_rates".into(),
            ));
        }
        let rows = obj_get(obj, "entries")
            .and_then(Json::as_array)
            .ok_or_else(|| SaddleError::Invalid("catalogue is missing entries".into()))?;
        let mut entries = Vec::with_capacity(rows.len());
        for row in rows {
            let item = row
                .as_object()
                .ok_or_else(|| SaddleError::Invalid("a catalogue entry must be an object".into()))?;
            entries.push(RateEntry {
                shell: required_str(item, "shell")?.to_string(),
                direction: DirectionClass::parse(required_str(item, "direction")?)?,
                hop: required_str(item, "hop")?.to_string(),
                temperature: required_f64(item, "temperature")?,
                cell_repeats: required_f64(item, "cell_repeats")? as u32,
                n_atoms: required_f64(item, "n_atoms")? as usize,
                separation_before: required_f64(item, "separation_before")?,
                separation_after: required_f64(item, "separation_after")?,
                outcome: HopOutcome::parse(required_str(item, "outcome")?)?,
                connected: flag(item, "connected")?,
                index1: flag(item, "index1")?,
                band_converged: flag(item, "band_converged")?,
                reactant_energy: required_f64(item, "reactant_energy")?,
                product_energy: required_f64(item, "product_energy")?,
                saddle_energy: required_f64(item, "saddle_energy")?,
                d_e0: required_f64(item, "dE0")?,
                d_f: optional_f64(item, "dF")?,
                d_f_error: optional_f64(item, "dF_error")?,
                d_f_harmonic: optional_f64(item, "dF_harmonic")?,
                nu: optional_f64(item, "nu")?,
                vineyard: optional_f64(item, "vineyard")?,
                kappa: optional_f64(item, "kappa")?,
                kappa_error: optional_f64(item, "kappa_error")?,
                harmonic_rate: optional_f64(item, "harmonic_rate")?,
                rate: optional_f64(item, "rate_Hz")?,
                zero_t: optional_f64(item, "zero_T")?,
                band_force: optional_f64(item, "band_force")?.unwrap_or(0.0),
                pafi_samples: optional_f64(item, "pafi_samples")?.unwrap_or(0.0) as usize,
                pafi_evaluations: optional_f64(item, "pafi_evaluations")?.unwrap_or(0.0) as usize,
                saddle_stress: required_matrix(item, "saddle_stress")?,
                saddle_dipole: required_matrix(item, "saddle_dipole")?,
                negative_eigenvalue: optional_f64(item, "negative_eigenvalue")?,
                orientation: optional_str(item, "orientation").unwrap_or("").to_string(),
                final_state: optional_str(item, "final_state").unwrap_or("").to_string(),
            });
        }
        Ok(Self { entries })
    }
}

/// Find every hop's saddle and its finite-temperature rates.
pub fn defect_pair_rates<S: PointSurface + Sync>(
    surface: &S,
    setup: &PairSetup,
    config: &DefectPairConfig,
) -> Result<RateCatalogue, SaddleError> {
    check_setup(setup)?;
    check_config(config)?;
    let n_atoms = setup.positions.len() / 3;
    let (reactant, reactant_energy) = relax(
        surface,
        setup.positions.clone(),
        setup.cell,
        config,
    )?;
    let mut entries = Vec::new();
    for hop in &setup.hops {
        let (product, product_energy) = relax(surface, hop.product.clone(), setup.cell, config)?;
        let before = pair_separation(reactant.view(), setup.cell, setup.fixed, &setup.mobile)?;
        let after = pair_separation(product.view(), setup.cell, setup.fixed, &setup.mobile)?;
        let direction = classify_separation(before, after, config.direction_tol)?;
        let searched = search_hop(
            surface,
            setup,
            config,
            hop,
            &reactant,
            reactant_energy,
            &product,
            product_energy,
        )?;
        for &temperature in &config.temperatures {
            let mut entry = blank_entry(
                setup,
                hop,
                direction,
                temperature,
                n_atoms,
                before,
                after,
                reactant_energy,
                product_energy,
            );
            fill_entry(surface, config, &searched, temperature, &mut entry)?;
            entries.push(entry);
        }
    }
    Ok(RateCatalogue { entries })
}

struct Searched {
    outcome: HopOutcome,
    connected: bool,
    index1: bool,
    band_converged: bool,
    saddle: Array1<f64>,
    saddle_cell: M3,
    saddle_energy: f64,
    d_e0: f64,
    stress: M3,
    dipole: M3,
    negative: Option<f64>,
    path: Option<ArcPath>,
    reactant: Array1<f64>,
    reactant_energy: f64,
    band_force: f64,
}

fn search_hop<S: PointSurface>(
    surface: &S,
    setup: &PairSetup,
    config: &DefectPairConfig,
    hop: &HopTemplate,
    reactant: &Array1<f64>,
    reactant_energy: f64,
    product: &Array1<f64>,
    _product_energy: f64,
) -> Result<Searched, SaddleError> {
    let failed = |outcome: HopOutcome| -> Searched {
        Searched {
            outcome,
            connected: false,
            index1: false,
            band_converged: false,
            saddle: reactant.clone(),
            saddle_cell: setup.cell,
            saddle_energy: reactant_energy,
            d_e0: 0.0,
            stress: [[0.0; 3]; 3],
            dipole: [[0.0; 3]; 3],
            negative: None,
            path: None,
            reactant: reactant.clone(),
            reactant_energy,
            band_force: 0.0,
        }
    };
    let product_mode = mic_positions(product, reactant, setup.cell)?;
    if norm1(&product_mode) < 1.0e-8 && hop.mode.is_none() {
        return Ok(failed(HopOutcome::NotConverged));
    }
    let mut seeds = Vec::new();
    if let Some(hint) = &hop.saddle {
        let mode = hop.mode.clone().unwrap_or_else(|| product_mode.clone());
        seeds.push((hint.clone(), mode));
    }
    if let Ok(mixed) = mix(reactant, product, config.seed_fraction, setup.cell) {
        seeds.push((mixed, product_mode));
    }
    if seeds.is_empty() {
        return Ok(failed(HopOutcome::NotConverged));
    }
    let mut dimer_config = config.dimer.clone();
    dimer_config.mask = match config.cell_mode {
        CellMode::Frozen => CellMask::frozen(),
        CellMode::Variable => CellMask::all(),
    };
    dimer_config.external_stress = config.external_stress;
    let mut found_saddle = None;
    let mut last_negative = None;
    let mut saw_index = false;
    let mut disconnected = false;
    for (seed, mode) in seeds {
        if norm1(&mode) < 1.0e-8 {
            continue;
        }
        let mut dimer = SolidDimer::new(dimer_config.clone(), seed, setup.cell, mode)?;
        let report = dimer.run(surface, config.max_dimer_steps)?;
        if report.status != MinModeStatus::Converged {
            continue;
        }
        let index1 = !report.rejected && report.negative_modes == Some(1);
        if !index1 {
            saw_index = true;
            last_negative = report.eigenvalues.first().copied();
            continue;
        }
        let saddle = dimer.positions().to_owned();
        let saddle_cell = dimer.cell();
        let push = cartesian_push(&dimer, config.push)?;
        let (plus, _) = relax(surface, &saddle + &push, saddle_cell, config)?;
        let (minus, _) = relax(surface, &saddle - &push, saddle_cell, config)?;
        if links(&plus, &minus, reactant, product, setup.cell, config.connect_rmsd)? {
            found_saddle = Some((dimer, saddle, saddle_cell, report));
            break;
        }
        disconnected = true;
    }
    let Some((dimer, saddle, _found_cell, report)) = found_saddle else {
        if disconnected {
            return Ok(failed(HopOutcome::Disconnected));
        }
        if saw_index {
            let mut row = failed(HopOutcome::NotIndex1);
            row.negative = last_negative;
            return Ok(row);
        }
        return Ok(failed(HopOutcome::NotConverged));
    };
    let saddle_cell = dimer.cell();
    let push = cartesian_push(&dimer, config.push)?;
    let (plus, _) = relax(surface, &saddle + &push, saddle_cell, config)?;
    let (minus, _) = relax(surface, &saddle - &push, saddle_cell, config)?;
    let connected = links(&plus, &minus, reactant, product, setup.cell, config.connect_rmsd)?;
    let (saddle_energy, _) = surface.eval_in_cell(saddle.view(), &m3_to_row9(saddle_cell))?;
    let d_e0 = enthalpy(saddle_energy, saddle_cell, setup.cell, config.external_stress)?
        - enthalpy(reactant_energy, setup.cell, setup.cell, config.external_stress)?;
    let stress = cauchy(surface, saddle.view(), saddle_cell)?;
    let volume = m3_det(saddle_cell).abs();
    let dipole = ElasticDipole::from_stress(volume, stress, config.bulk_stress)?.tensor();
    if !connected {
        return Ok(Searched {
            outcome: HopOutcome::Disconnected,
            connected: false,
            index1: true,
            band_converged: false,
            saddle,
            saddle_cell,
            saddle_energy,
            d_e0,
            stress,
            dipole,
            negative: report.eigenvalues.first().copied(),
            path: None,
            reactant: reactant.clone(),
            reactant_energy,
            band_force: 0.0,
        });
    }
    let (path, band_converged, band_force) = climb_path(
        surface,
        config,
        reactant,
        &saddle,
        product,
        setup.cell,
        saddle_cell,
    )?;
    Ok(Searched {
        outcome: HopOutcome::Rated,
        connected: true,
        index1: true,
        band_converged,
        saddle,
        saddle_cell,
        saddle_energy,
        d_e0,
        stress,
        dipole,
        negative: report.eigenvalues.first().copied(),
        path: Some(path),
        reactant: reactant.clone(),
        reactant_energy,
        band_force,
    })
}

fn fill_entry<S: PointSurface + Sync>(
    surface: &S,
    config: &DefectPairConfig,
    searched: &Searched,
    temperature: f64,
    entry: &mut RateEntry,
) -> Result<(), SaddleError> {
    entry.outcome = searched.outcome;
    entry.connected = searched.connected;
    entry.index1 = searched.index1;
    entry.band_converged = searched.band_converged;
    entry.band_force = searched.band_force;
    entry.saddle_energy = searched.saddle_energy;
    entry.d_e0 = searched.d_e0;
    entry.saddle_stress = searched.stress;
    entry.saddle_dipole = searched.dipole;
    entry.negative_eigenvalue = searched.negative;
    entry.reactant_energy = searched.reactant_energy;
    let Some(path) = searched.path.as_ref() else {
        return Ok(());
    };
    if !searched.band_converged {
        entry.outcome = HopOutcome::NotConverged;
        return Ok(());
    }
    let thermal = config.boltzmann * temperature;
    if !(thermal.is_finite() && thermal > 0.0) {
        return Err(SaddleError::Invalid(
            "kT must be finite and positive".into(),
        ));
    }
    let harmonic = harmonic_rates(
        surface,
        config,
        searched.reactant.view(),
        searched.saddle.view(),
        path,
        searched.d_e0,
        thermal,
    )?;
    entry.nu = harmonic.nu;
    entry.vineyard = harmonic.vineyard;
    entry.d_f_harmonic = harmonic.d_f_harmonic;
    entry.harmonic_rate = harmonic.rate;
    let arcs = plane_arcs(path, config.planes);
    let zero = pafi::zero_temperature_barrier(surface, path, &arcs)?;
    entry.zero_t = Some(zero.barrier);
    if (zero.barrier - searched.d_e0).abs() > config.zero_t_tol {
        entry.outcome = HopOutcome::NotConverged;
        entry.d_f = Some(zero.barrier);
        entry.d_f_error = Some((zero.barrier - searched.d_e0).abs());
        return Ok(());
    }
    if !config.finite_temperature {
        return Ok(());
    }
    let mut pafi_config = config.pafi.clone();
    pafi_config.thermal = thermal;
    let (profile, planes) = pafi::pafi_profile(surface, path, &arcs, &pafi_config)?;
    entry.d_f = Some(profile.barrier);
    entry.d_f_error = Some(profile.barrier_error);
    entry.pafi_samples = profile.samples;
    entry.pafi_evaluations = planes.iter().map(|plane| plane.evaluations).sum();
    let transmission = shoot(surface, config, path, &profile, &planes, thermal)?;
    entry.kappa = Some(transmission.kappa);
    entry.kappa_error = Some(transmission.error);
    if let Some(nu) = harmonic.nu
        && (0.0..=1.0).contains(&transmission.kappa)
    {
        entry.rate = Some(arrhenius(transmission.kappa * nu, profile.barrier, thermal));
    }
    Ok(())
}

struct Harmonic {
    nu: Option<f64>,
    vineyard: Option<f64>,
    d_f_harmonic: Option<f64>,
    rate: Option<f64>,
}

fn harmonic_rates<S: PointSurface>(
    surface: &S,
    config: &DefectPairConfig,
    reactant: ArrayView1<f64>,
    saddle: ArrayView1<f64>,
    path: &ArcPath,
    d_e0: f64,
    thermal: f64,
) -> Result<Harmonic, SaddleError> {
    let atoms = config.harmonic_atoms.as_deref();
    let (min_h, min_vals, min_vecs) = hessian(surface, reactant, atoms, config.harmonic_step)?;
    let (_sad_h, sad_vals, sad_vecs) = hessian(surface, saddle, atoms, config.harmonic_step)?;
    let (min_positive, sad_positive) = if atoms.is_none() {
        let translations = surface.excluded_modes(reactant)?;
        let min_split = split_translational_modes(&min_vals, min_vecs.view(), translations.view())?;
        let sad_translations = surface.excluded_modes(saddle)?;
        let sad_split =
            split_translational_modes(&sad_vals, sad_vecs.view(), sad_translations.view())?;
        (min_split.positive, sad_split.positive)
    } else {
        (
            positives(&min_vals),
            positives(&sad_vals),
        )
    };
    let geometry = path.geometry(path.bounds().0)?;
    let tangent = unit(&geometry.tangent)?;
    let lambda = curvature_along(&min_h, tangent.view(), atoms, reactant.len());
    let nu = if lambda > 0.0 {
        Some(attempt_frequency(lambda, config.mass, config.angular_scale)?)
    } else {
        None
    };
    let vineyard = if !min_positive.is_empty()
        && !sad_positive.is_empty()
        && min_positive.len() > sad_positive.len()
    {
        vineyard_frequency(&min_positive, &sad_positive, config.mass, config.angular_scale).ok()
    } else {
        None
    };
    let d_f_harmonic = match (nu, vineyard) {
        (Some(nu), Some(vineyard)) if vineyard > 0.0 && nu > 0.0 => {
            Some(d_e0 - thermal * (vineyard / nu).ln())
        }
        _ => None,
    };
    let rate = vineyard.map(|prefactor| arrhenius(prefactor, d_e0, thermal));
    Ok(Harmonic {
        nu,
        vineyard,
        d_f_harmonic,
        rate,
    })
}

fn shoot<S: PointSurface>(
    surface: &S,
    config: &DefectPairConfig,
    path: &ArcPath,
    profile: &FreeEnergyProfile,
    planes: &[HyperplaneAverage],
    thermal: f64,
) -> Result<Transmission, SaddleError> {
    let dividing = profile.dividing;
    let geometry = path.geometry(profile.arc[dividing])?;
    let starts = if planes[dividing].retained.is_empty() {
        vec![geometry.positions.clone()]
    } else {
        planes[dividing].retained.clone()
    };
    let mut shooting = config.shooting.clone();
    shooting.thermal = thermal;
    shooting.mass = config.mass;
    bennett::bennett_chandler(
        surface,
        &starts,
        geometry.positions.view(),
        geometry.tangent.view(),
        &shooting,
    )
}

fn climb_path<S: PointSurface>(
    surface: &S,
    config: &DefectPairConfig,
    reactant: &Array1<f64>,
    saddle: &Array1<f64>,
    product: &Array1<f64>,
    cell: M3,
    saddle_cell: M3,
) -> Result<(ArcPath, bool, f64), SaddleError> {
    let images = config.images.max(3);
    let mid = images / 2;
    let idpp = crate::idpp::IdppConfig {
        max_iterations: 2000,
        grad_tol: 1.0e-3,
        max_move: 0.2,
        memory: 12,
    };
    let wrapped = crate::mic::Cell::from_ase(cell).map_err(|error| {
        SaddleError::Invalid(format!("IDPP cell: {error}"))
    })?;
    let left = crate::idpp::idpp_path(
        reactant.view(),
        saddle.view(),
        mid + 1,
        Some(&wrapped),
        &idpp,
    )?;
    let right = crate::idpp::idpp_path(
        saddle.view(),
        product.view(),
        images - mid,
        Some(&wrapped),
        &idpp,
    )?;
    let mut band = Array2::zeros((images, reactant.len()));
    for i in 0..=mid {
        band.row_mut(i).assign(&left.row(i));
    }
    for i in 1..(images - mid) {
        band.row_mut(mid + i).assign(&right.row(i));
    }
    let _ = saddle_cell;
    // L-BFGS with climbing armed on the first step follows the highest
    // image. An IDPP guess can put that image on a collision, several eV
    // above the dimer saddle, and the two-loop step then runs up the
    // wall. The guards in the band reset a capped step; they do not make
    // that image the saddle. Relax with climbing off, then climb. If the
    // quasi-Newton climb still runs away, repeat it with FIRE, or with
    // ODE12r when the caller was already on FIRE.
    let mut plain = bare_band(&config.band);
    plain.climbing = None;
    let start_cells = vec![cell; images];
    let (mut relaxed, mut relaxed_cells, mut relaxed_force) = run_band(
        surface,
        config,
        plain,
        band.clone(),
        cell,
        Some(start_cells.clone()),
        false,
    )?;
    if relaxed_force > config.band.force_tol * 5.0
        && !matches!(config.band.method, Method::Fire { .. })
    {
        let mut fire = bare_band(&config.band);
        fire.climbing = None;
        fire.method = Method::Fire { kind: FireKind::V2 };
        fire.time_step = config.band.time_step.min(0.1);
        let (pos, cells, force) = run_band(
            surface,
            config,
            fire,
            band,
            cell,
            Some(start_cells),
            false,
        )?;
        if force < relaxed_force {
            relaxed = pos;
            relaxed_cells = cells;
            relaxed_force = force;
        }
    }
    let mut climbing = bare_band(&config.band);
    climbing.climbing = Some(CiConfig::always());
    let (climbed, _, climbed_force) = run_band(
        surface,
        config,
        climbing.clone(),
        relaxed.clone(),
        cell,
        Some(relaxed_cells.clone()),
        true,
    )?;
    let (cartesian, force) =
        if climbed_force <= config.band.force_tol || climbed_force <= relaxed_force {
            (climbed, climbed_force)
        } else {
            let mut retry = climbing;
            retry.method = if matches!(config.band.method, Method::Fire { .. }) {
                Method::Ode12r
            } else {
                Method::Fire { kind: FireKind::V2 }
            };
            retry.time_step = config.band.time_step.min(0.1);
            let (again, _, again_force) = run_band(
                surface,
                config,
                retry,
                relaxed,
                cell,
                Some(relaxed_cells),
                true,
            )?;
            if again_force < climbed_force {
                (again, again_force)
            } else {
                (climbed, climbed_force)
            }
        };
    let (saddle_energy, _) = surface.eval(saddle.view())?;
    let mut highest = f64::NEG_INFINITY;
    for row in cartesian.rows() {
        let (energy, _) = surface.eval(row)?;
        highest = highest.max(energy);
    }
    let mut frames = Vec::with_capacity(cartesian.nrows());
    for row in cartesian.rows() {
        frames.push(row.to_owned());
    }
    let tol = config.band.force_tol;
    let converged = force <= tol && highest <= saddle_energy + config.zero_t_tol.max(0.05);
    Ok((ArcPath::from_positions(&frames)?, converged, force))
}

fn bare_band(config: &BandConfig) -> BandConfig {
    let mut band = config.clone();
    band.min_mode = None;
    band.autoneb = None;
    band.zoom = None;
    band.solid = None;
    band.exp_precon = None;
    band.cell = None;
    band
}

fn run_band<S: PointSurface>(
    surface: &S,
    config: &DefectPairConfig,
    band_config: BandConfig,
    positions: Array2<f64>,
    reference: M3,
    cells: Option<Vec<M3>>,
    abort_on_spike: bool,
) -> Result<(Array2<f64>, Vec<M3>, f64), SaddleError> {
    let tol = band_config.force_tol;
    let n_images = positions.nrows();
    let image_cells = cells.unwrap_or_else(|| vec![reference; n_images]);
    let mut neb = match config.cell_mode {
        CellMode::Frozen => SolidNeb::frozen(band_config, positions)?,
        CellMode::Variable => SolidNeb::joint(
            band_config,
            reference,
            positions,
            &image_cells,
            CellMask::all(),
            config.external_stress,
            1.0,
        )?,
    };
    let mut steps = 0usize;
    let mut report = neb.step(surface)?;
    steps += 1;
    let mut best_force = report.max_force;
    let mut best = neb.cartesian()?;
    let mut best_cells = neb.cells()?.unwrap_or_default();
    while report.status == BandStatus::Running && steps < config.max_band_steps {
        report = neb.step(surface)?;
        steps += 1;
        if report.max_force < best_force {
            best_force = report.max_force;
            best = neb.cartesian()?;
            best_cells = neb.cells()?.unwrap_or_default();
        } else if abort_on_spike && best_force < tol.max(0.05) && report.max_force > best_force * 8.0
        {
            break;
        }
    }
    let (cartesian, found_cells, force) = if report.max_force <= best_force {
        (neb.cartesian()?, neb.cells()?.unwrap_or_default(), report.max_force)
    } else {
        (best, best_cells, best_force)
    };
    Ok((cartesian, found_cells, force))
}

fn cartesian_push(dimer: &SolidDimer, push: f64) -> Result<Array1<f64>, SaddleError> {
    let n = dimer.positions().len();
    let mode = dimer.mode().to_owned();
    let q = dimer.generalized().to_owned();
    let mut step = Array1::zeros(n);
    if let Some(chart) = dimer.chart() {
        let eps = chart.strain(q.view())?;
        let deform = deform_of(eps);
        for atom in 0..n / 3 {
            let row = [mode[3 * atom], mode[3 * atom + 1], mode[3 * atom + 2]];
            for c in 0..3 {
                step[3 * atom + c] =
                    row[0] * deform[0][c] + row[1] * deform[1][c] + row[2] * deform[2][c];
            }
        }
    } else {
        let length = mode.len().min(n);
        step.slice_mut(s![..length]).assign(&mode.slice(s![..length]));
    }
    let scale = max_atomic(&step);
    if scale == 0.0 {
        return Err(SaddleError::Invalid(
            "the unstable mode has no atomic component".into(),
        ));
    }
    Ok(step * (push / scale))
}

/// Largest per-atom length. `push` is this length, not the 3N Euclidean norm,
/// so a localized mode on a large cell still leaves the saddle.
fn max_atomic(step: &Array1<f64>) -> f64 {
    let mut scale = 0.0_f64;
    let n = step.len() / 3;
    for atom in 0..n {
        let mut acc = 0.0;
        for c in 0..3 {
            let value = step[3 * atom + c];
            acc += value * value;
        }
        scale = scale.max(acc.sqrt());
    }
    scale
}

fn deform_of(eps: M3) -> M3 {
    [
        [1.0 + eps[0][0], eps[0][1], eps[0][2]],
        [eps[1][0], 1.0 + eps[1][1], eps[1][2]],
        [eps[2][0], eps[2][1], 1.0 + eps[2][2]],
    ]
}

fn links(
    plus: &Array1<f64>,
    minus: &Array1<f64>,
    reactant: &Array1<f64>,
    product: &Array1<f64>,
    cell: M3,
    tol: f64,
) -> Result<bool, SaddleError> {
    let plus_r = hausdorff(plus, reactant, cell)?;
    let plus_p = hausdorff(plus, product, cell)?;
    let minus_r = hausdorff(minus, reactant, cell)?;
    let minus_p = hausdorff(minus, product, cell)?;
    let forward = plus_p <= tol && minus_r <= tol;
    let backward = plus_r <= tol && minus_p <= tol;
    Ok(forward || backward)
}

/// Largest gap between the two point clouds. Identical atoms make an index
/// RMSD depend on which atom landed in the dumbbell; the clouds do not.
fn hausdorff(left: &Array1<f64>, right: &Array1<f64>, cell: M3) -> Result<f64, SaddleError> {
    Ok(directed_gap(left, right, cell)?.max(directed_gap(right, left, cell)?))
}

fn directed_gap(left: &Array1<f64>, right: &Array1<f64>, cell: M3) -> Result<f64, SaddleError> {
    let n = left.len() / 3;
    let m = right.len() / 3;
    if n == 0 || n != m {
        return Err(SaddleError::Shape(
            "connectivity geometries must have the same number of atoms".into(),
        ));
    }
    let mut worst = 0.0;
    for i in 0..n {
        let origin = [left[3 * i], left[3 * i + 1], left[3 * i + 2]];
        let mut best = f64::MAX;
        for j in 0..m {
            let delta = mic(
                [
                    right[3 * j] - origin[0],
                    right[3 * j + 1] - origin[1],
                    right[3 * j + 2] - origin[2],
                ],
                cell,
            )?;
            let r = norm3(delta);
            if r < best {
                best = r;
            }
        }
        if best > worst {
            worst = best;
        }
    }
    Ok(worst)
}

fn relax<S: PointSurface>(
    surface: &S,
    mut positions: Array1<f64>,
    cell: M3,
    config: &DefectPairConfig,
) -> Result<(Array1<f64>, f64), SaddleError> {
    if config.cell_mode == CellMode::Variable {
        return relax_joint(surface, positions, cell, config);
    }
    let mut solver = rgmin::Solver::new(
        rgmin::Method::Fire {
            kind: rgmin::FireKind::V2,
        },
        rgmin::Control {
            maxiter: usize::MAX,
            gtol: 0.0,
            istep: config.relax_step,
            ..rgmin::Control::default()
        },
        positions.len(),
    );
    solver.set_atom_maxmove(config.max_move);
    let flat = m3_to_row9(cell);
    for _ in 0..config.max_relax_steps {
        let oracle = rgmin::Oracle::unbounded(positions.len(), |x: ArrayView1<f64>| {
            surface
                .eval_in_cell(x, &flat)
                .unwrap_or_else(|_| (f64::NAN, Array1::zeros(x.len())))
        });
        solver
            .step(&oracle, &mut positions)
            .map_err(|error| SaddleError::Solver(error.to_string()))?;
        let (_, gradient) = surface.eval_in_cell(positions.view(), &flat)?;
        if max_abs(&gradient) < config.relax_tol {
            break;
        }
    }
    let (energy, _) = surface.eval_in_cell(positions.view(), &flat)?;
    Ok((positions, energy))
}

fn relax_joint<S: PointSurface>(
    surface: &S,
    positions: Array1<f64>,
    cell: M3,
    config: &DefectPairConfig,
) -> Result<(Array1<f64>, f64), SaddleError> {
    let n_atoms = positions.len() / 3;
    let chart = JointChart::new(
        cell,
        n_atoms,
        CellMask::all(),
        config.external_stress,
        1.0,
    )?;
    let mut q = Array1::zeros(chart.dimension());
    q.slice_mut(s![..positions.len()]).assign(&positions);
    let mut solver = rgmin::Solver::new(
        rgmin::Method::Fire {
            kind: rgmin::FireKind::V2,
        },
        rgmin::Control {
            maxiter: usize::MAX,
            gtol: 0.0,
            istep: config.relax_step,
            ..rgmin::Control::default()
        },
        q.len(),
    );
    solver.set_atom_maxmove(config.max_move);
    for _ in 0..config.max_relax_steps {
        let oracle = rgmin::Oracle::unbounded(q.len(), |q: ArrayView1<f64>| {
            match joint_gradient(surface, &chart, q) {
                Ok(pair) => pair,
                Err(_) => (f64::NAN, Array1::zeros(q.len())),
            }
        });
        solver
            .step(&oracle, &mut q)
            .map_err(|error| SaddleError::Solver(error.to_string()))?;
        let (_, gradient) = joint_gradient(surface, &chart, q.view())?;
        if max_abs(&gradient) < config.relax_tol {
            break;
        }
    }
    let (energy, _) = joint_gradient(surface, &chart, q.view())?;
    let (positions, _) = chart.geometry(q.view())?;
    Ok((positions, energy))
}

fn joint_gradient<S: PointSurface>(
    surface: &S,
    chart: &JointChart,
    q: ArrayView1<f64>,
) -> Result<(f64, Array1<f64>), SaddleError> {
    let (positions, cell) = chart.geometry(q)?;
    let flat = m3_to_row9(cell);
    let (energy, gradient) = surface.eval_in_cell(positions.view(), &flat)?;
    let stress = cauchy(surface, positions.view(), cell)?;
    let generalized = chart.gradient(q, gradient.view(), stress)?;
    Ok((chart.enthalpy(energy, q)?, generalized))
}

fn cauchy<S: PointSurface>(
    surface: &S,
    positions: ArrayView1<f64>,
    cell: M3,
) -> Result<M3, SaddleError> {
    let flat = m3_to_row9(cell);
    if let Some(raw) = surface.cauchy_stress(positions, &flat)? {
        return Ok(unflat(&raw));
    }
    finite_difference_cauchy_stress(surface, positions, cell, 1.0e-5)
}

fn hessian<S: PointSurface>(
    surface: &S,
    positions: ArrayView1<f64>,
    atoms: Option<&[usize]>,
    step: f64,
) -> Result<(Array2<f64>, Vec<f64>, Array2<f64>), SaddleError> {
    if !(step.is_finite() && step > 0.0) {
        return Err(SaddleError::Invalid(
            "harmonic difference step must be positive".into(),
        ));
    }
    let coords: Vec<usize> = match atoms {
        None => (0..positions.len()).collect(),
        Some(atoms) => {
            let mut coords = Vec::with_capacity(atoms.len() * 3);
            for &atom in atoms {
                if atom >= positions.len() / 3 {
                    return Err(SaddleError::Invalid(
                        "harmonic atom is outside the cell".into(),
                    ));
                }
                coords.extend([3 * atom, 3 * atom + 1, 3 * atom + 2]);
            }
            coords
        }
    };
    if coords.is_empty() {
        return Err(SaddleError::Invalid(
            "the harmonic region is empty".into(),
        ));
    }
    let n = coords.len();
    let mut matrix = Array2::zeros((n, n));
    for (col, &coordinate) in coords.iter().enumerate() {
        let mut plus = positions.to_owned();
        let mut minus = positions.to_owned();
        plus[coordinate] += step;
        minus[coordinate] -= step;
        let (_, g_plus) = surface.eval(plus.view())?;
        let (_, g_minus) = surface.eval(minus.view())?;
        for (row, &other) in coords.iter().enumerate() {
            matrix[(row, col)] = (g_plus[other] - g_minus[other]) / (2.0 * step);
        }
    }
    for i in 0..n {
        for j in 0..i {
            let mean = 0.5 * (matrix[(i, j)] + matrix[(j, i)]);
            matrix[(i, j)] = mean;
            matrix[(j, i)] = mean;
        }
    }
    let (values, vectors) = exact_eigh(matrix.view())?;
    Ok((matrix, values.to_vec(), vectors))
}

fn curvature_along(hessian: &Array2<f64>, tangent: ArrayView1<f64>, atoms: Option<&[usize]>, n_full: usize) -> f64 {
    let coords: Vec<usize> = match atoms {
        None => (0..n_full).collect(),
        Some(atoms) => atoms.iter().flat_map(|atom| [3 * atom, 3 * atom + 1, 3 * atom + 2]).collect(),
    };
    let mut lambda = 0.0;
    for (i, &ci) in coords.iter().enumerate() {
        for (j, &cj) in coords.iter().enumerate() {
            lambda += tangent[ci] * hessian[(i, j)] * tangent[cj];
        }
    }
    lambda
}

fn positives(values: &[f64]) -> Vec<f64> {
    values.iter().copied().filter(|value| *value > 1.0e-4).collect()
}

fn vineyard_frequency(
    min_lambda: &[f64],
    sad_lambda: &[f64],
    mass: f64,
    angular: f64,
) -> Result<f64, SaddleError> {
    if !(mass.is_finite() && mass > 0.0 && angular.is_finite() && angular > 0.0) {
        return Err(SaddleError::Invalid(
            "Vineyard mass and angular scale must be positive".into(),
        ));
    }
    if min_lambda.len() <= sad_lambda.len() {
        return Err(SaddleError::Invalid(
            "the saddle product is not shorter than the minimum product".into(),
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
    log_ratio -= 0.5 * extra * mass.ln();
    log_ratio -= extra * (2.0 * std::f64::consts::PI).ln();
    let frequency = log_ratio.exp() * angular.powf(extra);
    if !frequency.is_finite() || frequency <= 0.0 {
        return Err(SaddleError::NonFinite("Vineyard prefactor"));
    }
    Ok(frequency)
}

/// `ν = angular * sqrt(λ / mass) / 2π`.
pub fn attempt_frequency(lambda: f64, mass: f64, angular: f64) -> Result<f64, SaddleError> {
    if !(lambda.is_finite() && lambda > 0.0 && mass.is_finite() && mass > 0.0) {
        return Err(SaddleError::Invalid(
            "attempt frequency needs a positive curvature and mass".into(),
        ));
    }
    if !(angular.is_finite() && angular > 0.0) {
        return Err(SaddleError::Invalid(
            "angular scale must be positive".into(),
        ));
    }
    Ok(angular * (lambda / mass).sqrt() / std::f64::consts::TAU)
}

fn arrhenius(prefactor: f64, barrier: f64, thermal: f64) -> f64 {
    let exponent = -barrier / thermal;
    if exponent < -700.0 {
        0.0
    } else {
        prefactor * exponent.exp()
    }
}

fn plane_arcs(path: &ArcPath, planes: usize) -> Vec<f64> {
    let nodes = path.arcs();
    if planes < 2 {
        return nodes;
    }
    let (start, end) = path.bounds();
    (0..planes)
        .map(|i| start + (end - start) * (i as f64) / (planes - 1) as f64)
        .collect()
}

fn blank_entry(
    setup: &PairSetup,
    hop: &HopTemplate,
    direction: DirectionClass,
    temperature: f64,
    n_atoms: usize,
    before: f64,
    after: f64,
    reactant_energy: f64,
    product_energy: f64,
) -> RateEntry {
    RateEntry {
        shell: setup.shell.clone(),
        direction,
        hop: hop.name.clone(),
        temperature,
        cell_repeats: setup.cell_repeats,
        n_atoms,
        separation_before: before,
        separation_after: after,
        outcome: HopOutcome::NotConverged,
        connected: false,
        index1: false,
        band_converged: false,
        reactant_energy,
        product_energy,
        saddle_energy: reactant_energy,
        d_e0: 0.0,
        d_f: None,
        d_f_error: None,
        d_f_harmonic: None,
        nu: None,
        vineyard: None,
        kappa: None,
        kappa_error: None,
        harmonic_rate: None,
        rate: None,
        zero_t: None,
        band_force: 0.0,
        pafi_samples: 0,
        pafi_evaluations: 0,
        saddle_stress: [[0.0; 3]; 3],
        saddle_dipole: [[0.0; 3]; 3],
        negative_eigenvalue: None,
        orientation: hop.orientation.clone(),
        final_state: hop.final_state.clone(),
    }
}

fn check_setup(setup: &PairSetup) -> Result<(), SaddleError> {
    let n = setup.positions.len();
    if n == 0 || !n.is_multiple_of(3) || !setup.positions.iter().all(|v| v.is_finite()) {
        return Err(SaddleError::Shape(
            "defect-pair positions must be a finite 3N vector".into(),
        ));
    }
    if m3_det(setup.cell).abs() < 1.0e-18 {
        return Err(SaddleError::Invalid(
            "defect-pair cell must have positive volume".into(),
        ));
    }
    if setup.mobile.is_empty() || setup.mobile.iter().any(|atom| atom >= &(n / 3)) {
        return Err(SaddleError::Invalid(
            "mobile atoms must be a non-empty set of indices".into(),
        ));
    }
    if setup.hops.is_empty() {
        return Err(SaddleError::Invalid(
            "the hop catalogue is empty".into(),
        ));
    }
    for hop in &setup.hops {
        if hop.product.len() != n {
            return Err(SaddleError::Shape(
                "a hop product must have the reactant length".into(),
            ));
        }
        if hop.name.is_empty() {
            return Err(SaddleError::Invalid("a hop needs a name".into()));
        }
    }
    Ok(())
}

fn check_config(config: &DefectPairConfig) -> Result<(), SaddleError> {
    if config.dimer.index.is_none() {
        return Err(SaddleError::Invalid(
            "defect-pair search requires the solid-dimer index check".into(),
        ));
    }
    if config.images < 3 {
        return Err(SaddleError::Invalid(
            "the climbing image band needs at least three images".into(),
        ));
    }
    if config.temperatures.is_empty()
        || config.temperatures.iter().any(|t| !(t.is_finite() && *t > 0.0))
    {
        return Err(SaddleError::Invalid(
            "at least one positive finite temperature is required".into(),
        ));
    }
    if !(config.boltzmann.is_finite() && config.boltzmann > 0.0) {
        return Err(SaddleError::Invalid(
            "Boltzmann's constant must be positive".into(),
        ));
    }
    if !(config.seed_fraction.is_finite() && (0.0..1.0).contains(&config.seed_fraction)) {
        return Err(SaddleError::Invalid(
            "the dimer seed fraction must lie in [0, 1)".into(),
        ));
    }
    if !(config.push.is_finite() && config.push > 0.0 && config.connect_rmsd > 0.0) {
        return Err(SaddleError::Invalid(
            "the push and the connectivity tolerance must be positive".into(),
        ));
    }
    if !(config.zero_t_tol.is_finite() && config.zero_t_tol > 0.0) {
        return Err(SaddleError::Invalid(
            "the zero-temperature tolerance must be positive".into(),
        ));
    }
    Ok(())
}

fn enthalpy(energy: f64, cell: M3, reference: M3, external: M3) -> Result<f64, SaddleError> {
    let eps = strain_between(reference, cell)?;
    let volume = m3_det(reference).abs();
    Ok(energy + volume * double_dot(external, eps))
}

fn strain_between(reference: M3, cell: M3) -> Result<M3, SaddleError> {
    let prod = m3_mul(m3_inv(reference)?, cell);
    Ok([
        [prod[0][0] - 1.0, prod[0][1], prod[0][2]],
        [prod[1][0], prod[1][1] - 1.0, prod[1][2]],
        [prod[2][0], prod[2][1], prod[2][2] - 1.0],
    ])
}

fn mix(start: &Array1<f64>, end: &Array1<f64>, fraction: f64, cell: M3) -> Result<Array1<f64>, SaddleError> {
    let delta = mic_positions(end, start, cell)?;
    Ok(start + delta * fraction)
}

fn mic_positions(end: &Array1<f64>, start: &Array1<f64>, cell: M3) -> Result<Array1<f64>, SaddleError> {
    if end.len() != start.len() {
        return Err(SaddleError::Shape(
            "geometries in one hop must share a length".into(),
        ));
    }
    let mut out = Array1::zeros(start.len());
    for atom in 0..start.len() / 3 {
        let delta = mic(
            [
                end[3 * atom] - start[3 * atom],
                end[3 * atom + 1] - start[3 * atom + 1],
                end[3 * atom + 2] - start[3 * atom + 2],
            ],
            cell,
        )?;
        for c in 0..3 {
            out[3 * atom + c] = delta[c];
        }
    }
    Ok(out)
}

fn mobile_center(positions: ArrayView1<f64>, cell: M3, mobile: &[usize]) -> Result<[f64; 3], SaddleError> {
    let origin = mobile[0];
    let base = [
        positions[3 * origin],
        positions[3 * origin + 1],
        positions[3 * origin + 2],
    ];
    let mut acc = [0.0; 3];
    for &atom in mobile {
        let point = [
            positions[3 * atom],
            positions[3 * atom + 1],
            positions[3 * atom + 2],
        ];
        let delta = mic(sub3(point, base), cell)?;
        for c in 0..3 {
            acc[c] += base[c] + delta[c];
        }
    }
    let scale = mobile.len() as f64;
    Ok([acc[0] / scale, acc[1] / scale, acc[2] / scale])
}

fn mic(delta: [f64; 3], cell: M3) -> Result<[f64; 3], SaddleError> {
    let inv = m3_inv(cell)?;
    let mut s = [0.0; 3];
    for i in 0..3 {
        s[i] = delta[0] * inv[0][i] + delta[1] * inv[1][i] + delta[2] * inv[2][i];
        s[i] -= s[i].round();
    }
    Ok([
        s[0] * cell[0][0] + s[1] * cell[1][0] + s[2] * cell[2][0],
        s[0] * cell[0][1] + s[1] * cell[1][1] + s[2] * cell[2][1],
        s[0] * cell[0][2] + s[1] * cell[1][2] + s[2] * cell[2][2],
    ])
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn norm3(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn norm1(v: &Array1<f64>) -> f64 {
    v.dot(v).sqrt()
}

fn unit(v: &Array1<f64>) -> Result<Array1<f64>, SaddleError> {
    let scale = norm1(v);
    if scale == 0.0 {
        return Err(SaddleError::Invalid("a path tangent vanishes".into()));
    }
    Ok(v / scale)
}

fn max_abs(gradient: &Array1<f64>) -> f64 {
    gradient.iter().map(|v| v.abs()).fold(0.0, f64::max)
}

fn unflat(cell: &[f64; 9]) -> M3 {
    [
        [cell[0], cell[1], cell[2]],
        [cell[3], cell[4], cell[5]],
        [cell[6], cell[7], cell[8]],
    ]
}

fn flag(obj: &[(String, Json)], key: &str) -> Result<bool, SaddleError> {
    let value = required_f64(obj, key)?;
    Ok(value > 0.5)
}

fn optional_str<'a>(obj: &'a [(String, Json)], key: &str) -> Option<&'a str> {
    obj_get(obj, key).and_then(Json::as_str)
}

fn optional_f64(obj: &[(String, Json)], key: &str) -> Result<Option<f64>, SaddleError> {
    match obj_get(obj, key) {
        None | Some(Json::Null) => Ok(None),
        Some(value) => value.as_f64().map(Some).ok_or_else(|| {
            SaddleError::Invalid(format!("json field {key} must be a number or null"))
        }),
    }
}

fn num(value: f64) -> String {
    if value == 0.0 {
        "0".to_string()
    } else {
        format!("{value:.16e}")
    }
}

fn opt_num(value: Option<f64>) -> String {
    match value {
        Some(value) => num(value),
        None => "null".to_string(),
    }
}

fn json_string(text: &str) -> String {
    let mut out = String::from("\"");
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            _ => out.push(ch),
        }
    }
    out.push('"');
    out
}

fn matrix_json(matrix: M3) -> String {
    format!(
        "[[{},{},{}],[{},{},{}],[{},{},{}]]",
        num(matrix[0][0]),
        num(matrix[0][1]),
        num(matrix[0][2]),
        num(matrix[1][0]),
        num(matrix[1][1]),
        num(matrix[1][2]),
        num(matrix[2][0]),
        num(matrix[2][1]),
        num(matrix[2][2])
    )
}

fn push_kv(out: &mut String, key: &str, value: &str, comma: bool) {
    out.push_str("      \"");
    out.push_str(key);
    out.push_str("\": ");
    out.push_str(value);
    if comma {
        out.push(',');
    }
    out.push('\n');
}

