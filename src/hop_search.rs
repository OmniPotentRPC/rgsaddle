//! Zero-temperature hop search for a defect pair.
//!
//! Given a potential, a periodic cell, and one hop template, this module
//!
//! 1. minimizes the reactant and the product,
//! 2. climbs from the reactant side with the solid-state dimer
//!    ([`crate::SolidDimer`]), on a frozen cell or on the joint chart,
//! 3. requires one negative projected eigenvalue from [`crate::index_at`],
//! 4. pushes off both sides of the unstable mode, minimizes, and keeps the
//!    hop only when those minima are the reactant and the product,
//! 5. builds a climbing-image solid-state NEB through that saddle.
//!
//! [`search_hop`] is the documented entry point. It returns a
//! [`HopSearchResult`] with the saddle, path, enthalpy barrier, and
//! Hessian-derived saddle data that a finite-temperature rate evaluation
//! consumes. The rate half lives in [`crate::defect_pair`].

use ndarray::{Array1, Array2, ArrayView1, s};
use rayon::prelude::*;
use rgmin::{FireKind, Method};

use crate::band::{BandConfig, BandStatus, CiConfig};
use crate::cell_log::{M3, m3_det, m3_inv, m3_mul, m3_to_row9};
use crate::eigensolve::exact_eigh;
use crate::error::SaddleError;
use crate::minmode::{MinModeStatus, PointSurface};
use crate::pafi::ArcPath;
use crate::relax::{RelaxConfig, max_atom_force, relax_to_tolerance};
use crate::solid_dimer::{
    CellMask, JointChart, SolidDimer, SolidDimerConfig, finite_difference_cauchy_stress,
};
use crate::ssneb::SolidNeb;
use crate::strain_rate::{ElasticDipole, double_dot};

/// Which way the mobile defect moves relative to the fixed defect.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DirectionClass {
    /// The separation shrinks by more than the tolerance.
    Toward,
    /// The separation grows by more than the tolerance.
    Away,
    /// The separation changes by at most the tolerance.
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

    pub(crate) fn parse(text: &str) -> Result<Self, SaddleError> {
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

/// A dimer start. `mode` is the initial unstable direction, not a displacement
/// that has to be normalized by the caller.
#[derive(Clone, Debug)]
pub struct DimerSeed {
    /// Start geometry, Cartesian.
    pub positions: Array1<f64>,
    /// Initial unstable direction.
    pub mode: Array1<f64>,
}

/// An index-1 saddle and the two basins reached by pushing along its mode.
#[derive(Clone, Debug)]
pub struct BasinSaddle {
    /// Saddle geometry, Cartesian.
    pub saddle: Array1<f64>,
    /// Energy at the saddle.
    pub saddle_energy: f64,
    /// Lowest projected eigenvalue at the saddle.
    pub negative_eigenvalue: Option<f64>,
    /// Unstable mode at the saddle. The lower basin lies along one sign of this.
    pub mode: Array1<f64>,
    /// Minimum reached from `saddle + push`.
    pub plus: Array1<f64>,
    /// Minimum reached from `saddle - push`.
    pub minus: Array1<f64>,
    /// Energy of `plus`.
    pub plus_energy: f64,
    /// Energy of `minus`.
    pub minus_energy: f64,
}

/// One symmetry-distinct hop. `product` is a Cartesian geometry of the
/// same length as the reactant, before it is minimized.
#[derive(Clone, Debug)]
pub struct HopTemplate {
    /// Caller label, stored on every row.
    pub name: String,
    /// Product geometry, Cartesian.
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
    /// Relaxed reactant, Cartesian, three entries per atom.
    pub positions: Array1<f64>,
    /// Periodic cell, rows are lattice vectors.
    pub cell: M3,
    /// Fixed defect, Cartesian, in the same frame as `positions`.
    pub fixed: [f64; 3],
    /// Atoms whose centroid is the mobile defect.
    pub mobile: Vec<usize>,
    /// Caller label, stored on every row. A shell index or a distance.
    pub shell: String,
    /// Cell repeats, stored on every row. Zero if the cell is not a cube of repeats.
    pub cell_repeats: u32,
    /// Catalogue searched from this reactant.
    pub hops: Vec<HopTemplate>,
}

/// Frozen cell, or the joint chart with every lower-triangular strain free.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CellMode {
    /// The cell is held fixed.
    Frozen,
    /// Every lower-triangular strain is free.
    Variable,
}

/// Zero-temperature hop search settings.
///
/// Built from the matching fields of [`crate::DefectPairConfig`] so the
/// rate catalogue can keep one config while this module borrows only what
/// the search needs. Non-exhaustive: start from [`HopSearchConfig::default`]
/// and chain the `with_*` setters.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct HopSearchConfig {
    /// Saddle search. The cell mask is set from `cell_mode`.
    pub dimer: SolidDimerConfig,
    /// Frozen or variable cell.
    pub cell_mode: CellMode,
    /// Applied load with the sign of a pressure, positive inward, as
    /// [`SolidDimerConfig::external_stress`]. Nonzero makes barriers
    /// enthalpies.
    pub external_stress: M3,
    /// Subtracted in `P = -V (σ - σ_bulk)`. Zero if the perfect cell is unstressed.
    pub bulk_stress: M3,
    /// Climbing-image band settings.
    pub band: BandConfig,
    /// Band images, including the ends. At least 3.
    pub images: usize,
    /// Most dimer steps per hop.
    pub max_dimer_steps: usize,
    /// Most band steps per phase.
    pub max_band_steps: usize,
    /// Every minimization: the reactant, the products and the pushes off
    /// a saddle. It runs to `relax.force_tol` on the largest per-atom force
    /// and reaching `relax.max_steps` is an error. The pushes decide which
    /// minima a saddle connects, so the default follows the steepest-descent
    /// flow with [`Method::Ode12r`].
    pub relax: RelaxConfig,
    /// Dimer start, this fraction of the way from reactant to product.
    pub seed_fraction: f64,
    /// Largest atomic displacement of the push off each side of the saddle.
    pub push: f64,
    /// Largest point-cloud gap that still names a minimum.
    pub connect_rmsd: f64,
    /// Separation change below which a hop is tangential.
    pub direction_tol: f64,
    /// Central finite-difference step of the harmonic Hessian.
    pub harmonic_step: f64,
    /// Atoms in the harmonic Hessian. `None` is the whole cell.
    pub harmonic_atoms: Option<Vec<usize>>,
    /// Hyperplanes, including the ends. Zero uses every NEB image.
    pub planes: usize,
    /// A converged path whose zero-temperature barrier misses `dE0` by more
    /// than this is not treated as band-converged for rate work.
    pub zero_t_tol: f64,
}

with_setters!(HopSearchConfig {
    with_dimer => dimer: val SolidDimerConfig,
    with_cell_mode => cell_mode: val CellMode,
    with_external_stress => external_stress: val M3,
    with_bulk_stress => bulk_stress: val M3,
    with_band => band: val BandConfig,
    with_images => images: val usize,
    with_max_dimer_steps => max_dimer_steps: val usize,
    with_max_band_steps => max_band_steps: val usize,
    with_relax => relax: val RelaxConfig,
    with_seed_fraction => seed_fraction: val f64,
    with_push => push: val f64,
    with_connect_rmsd => connect_rmsd: val f64,
    with_direction_tol => direction_tol: val f64,
    with_harmonic_step => harmonic_step: val f64,
    with_harmonic_atoms => harmonic_atoms: opt Vec<usize>,
    with_planes => planes: val usize,
    with_zero_t_tol => zero_t_tol: val f64,
});

impl Default for HopSearchConfig {
    fn default() -> Self {
        Self {
            dimer: SolidDimerConfig::frozen(crate::MinModeConfig::default())
                .with_index_check(crate::IndexCheck::first_order()),
            cell_mode: CellMode::Frozen,
            external_stress: [[0.0; 3]; 3],
            bulk_stress: [[0.0; 3]; 3],
            band: BandConfig::default()
                .with_min_mode(None)
                .with_autoneb(None)
                .with_zoom(None)
                .with_climbing(CiConfig::always()),
            images: 7,
            max_dimer_steps: 80,
            max_band_steps: 80,
            relax: RelaxConfig::default()
                .with_method(Method::Ode12r)
                .with_max_atom_move(0.08),
            seed_fraction: 0.35,
            push: 0.15,
            connect_rmsd: 0.3,
            direction_tol: 0.05,
            harmonic_step: 1.0e-3,
            harmonic_atoms: None,
            planes: 0,
            zero_t_tol: 0.02,
        }
    }
}

/// Why a hop did not receive a rate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HopOutcome {
    /// Connected index-1 saddle with a rate.
    Rated,
    /// The dimer or the band did not converge, or the zero-temperature integral missed `dE0`.
    NotConverged,
    /// The dimer stopped on a point that is not index 1.
    NotIndex1,
    /// The push does not return to the reactant and the product.
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

    pub(crate) fn parse(text: &str) -> Result<Self, SaddleError> {
        match text {
            "rated" => Ok(Self::Rated),
            "not_converged" => Ok(Self::NotConverged),
            "not_index1" => Ok(Self::NotIndex1),
            "disconnected" => Ok(Self::Disconnected),
            _ => Err(SaddleError::Invalid("unknown hop outcome".into())),
        }
    }
}

/// Saddle, path, barriers, and Hessian-derived data of one zero-temperature hop.
///
/// The finite-temperature rate half consumes this record: the climbing path
/// for PAFI, `d_e0` as the enthalpy barrier, and `negative_eigenvalue` with
/// the saddle stress and dipole as the Hessian-derived saddle data.
#[derive(Clone, Debug)]
pub struct HopSearchResult {
    /// Search status used by the rate half.
    pub outcome: HopOutcome,
    /// Both pushes return to the named minima.
    pub connected: bool,
    /// One negative projected eigenvalue at the saddle.
    pub index1: bool,
    /// The climbing band met its force tolerance.
    pub band_converged: bool,
    /// Saddle geometry, Cartesian.
    pub saddle: Array1<f64>,
    /// Energy at the saddle.
    pub saddle_energy: f64,
    /// Enthalpy barrier from the reactant to the saddle.
    pub d_e0: f64,
    /// Cauchy stress at the saddle.
    pub stress: M3,
    /// `-V (σ_saddle - σ_bulk)` at the saddle.
    pub dipole: M3,
    /// Lowest projected eigenvalue at the saddle.
    pub negative_eigenvalue: Option<f64>,
    /// Climbing-image path through the saddle, when connected.
    pub path: Option<ArcPath>,
    /// Minimized reactant geometry.
    pub reactant: Array1<f64>,
    /// Minimized reactant energy.
    pub reactant_energy: f64,
    /// Max band force after the climbing phase.
    pub band_force: f64,
}

/// Zero-temperature search of one hop from a setup and template.
///
/// Relaxes the reactant and the product, climbs an index-1 saddle with the
/// solid-state dimer, checks connectivity, and builds the climbing-image
/// band. The returned [`HopSearchResult`] is what a finite-temperature rate
/// evaluation consumes.
pub fn search_hop<S: PointSurface>(
    surface: &S,
    setup: &PairSetup,
    hop: &HopTemplate,
    config: &HopSearchConfig,
) -> Result<HopSearchResult, SaddleError> {
    check_pair_geometry(setup)?;
    check_hop_template(hop, setup.positions.len())?;
    check_config(config)?;
    let (reactant, reactant_energy) = relax(surface, setup.positions.clone(), setup.cell, config)?;
    let (product, product_energy) = relax(surface, hop.product.clone(), setup.cell, config)?;
    search_hop_at(
        surface,
        setup,
        config,
        hop,
        &reactant,
        reactant_energy,
        &product,
        product_energy,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn search_hop_at<S: PointSurface>(
    surface: &S,
    setup: &PairSetup,
    config: &HopSearchConfig,
    hop: &HopTemplate,
    reactant: &Array1<f64>,
    reactant_energy: f64,
    product: &Array1<f64>,
    _product_energy: f64,
) -> Result<HopSearchResult, SaddleError> {
    let failed = |outcome: HopOutcome| -> HopSearchResult {
        HopSearchResult {
            outcome,
            connected: false,
            index1: false,
            band_converged: false,
            saddle: reactant.clone(),
            saddle_energy: reactant_energy,
            d_e0: 0.0,
            stress: [[0.0; 3]; 3],
            dipole: [[0.0; 3]; 3],
            negative_eigenvalue: None,
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
    let dimer_config = config
        .dimer
        .clone()
        .with_mask(match config.cell_mode {
            CellMode::Frozen => CellMask::frozen(),
            CellMode::Variable => CellMask::all(),
        })
        .with_external_stress(config.external_stress);
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
        if links(
            &plus,
            &minus,
            reactant,
            product,
            setup.cell,
            config.connect_rmsd,
        )? {
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
            row.negative_eigenvalue = last_negative;
            return Ok(row);
        }
        return Ok(failed(HopOutcome::NotConverged));
    };
    let saddle_cell = dimer.cell();
    let push = cartesian_push(&dimer, config.push)?;
    let (plus, _) = relax(surface, &saddle + &push, saddle_cell, config)?;
    let (minus, _) = relax(surface, &saddle - &push, saddle_cell, config)?;
    let connected = links(
        &plus,
        &minus,
        reactant,
        product,
        setup.cell,
        config.connect_rmsd,
    )?;
    let (saddle_energy, _) = surface.eval_in_cell(saddle.view(), &m3_to_row9(saddle_cell))?;
    let d_e0 = enthalpy(
        saddle_energy,
        saddle_cell,
        setup.cell,
        config.external_stress,
    )? - enthalpy(
        reactant_energy,
        setup.cell,
        setup.cell,
        config.external_stress,
    )?;
    let stress = cauchy(surface, saddle.view(), saddle_cell)?;
    let volume = m3_det(saddle_cell).abs();
    let dipole = ElasticDipole::from_stress(volume, stress, config.bulk_stress)?.tensor();
    if !connected {
        return Ok(HopSearchResult {
            outcome: HopOutcome::Disconnected,
            connected: false,
            index1: true,
            band_converged: false,
            saddle,
            saddle_energy,
            d_e0,
            stress,
            dipole,
            negative_eigenvalue: report.eigenvalues.first().copied(),
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
    Ok(HopSearchResult {
        outcome: HopOutcome::Rated,
        connected: true,
        index1: true,
        band_converged,
        saddle,
        saddle_energy,
        d_e0,
        stress,
        dipole,
        negative_eigenvalue: report.eigenvalues.first().copied(),
        path: Some(path),
        reactant: reactant.clone(),
        reactant_energy,
        band_force,
    })
}

fn climb_path<S: PointSurface>(
    surface: &S,
    config: &HopSearchConfig,
    reactant: &Array1<f64>,
    saddle: &Array1<f64>,
    product: &Array1<f64>,
    cell: M3,
    saddle_cell: M3,
) -> Result<(ArcPath, bool, f64), SaddleError> {
    let images = config.images.max(3);
    let mid = images / 2;
    let idpp = crate::idpp::IdppConfig::default()
        .with_max_iterations(2000)
        .with_grad_tol(1.0e-3)
        .with_max_move(0.2)
        .with_memory(12);
    let wrapped = crate::mic::Cell::from_ase(cell)
        .map_err(|error| SaddleError::Invalid(format!("IDPP cell: {error}")))?;
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
    let plain = bare_band(&config.band).with_climbing(None);
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
        let fire = bare_band(&config.band)
            .with_climbing(None)
            .with_method(Method::fire(FireKind::V2))
            .with_time_step(config.band.time_step.min(0.1));
        let (pos, cells, force) =
            run_band(surface, config, fire, band, cell, Some(start_cells), false)?;
        if force < relaxed_force {
            relaxed = pos;
            relaxed_cells = cells;
            relaxed_force = force;
        }
    }
    let climbing = bare_band(&config.band).with_climbing(CiConfig::always());
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
            let retry = climbing
                .with_method(if matches!(config.band.method, Method::Fire { .. }) {
                    Method::Ode12r
                } else {
                    Method::fire(FireKind::V2)
                })
                .with_time_step(config.band.time_step.min(0.1));
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
    config
        .clone()
        .with_min_mode(None)
        .with_autoneb(None)
        .with_zoom(None)
        .with_solid(None)
        .with_exp_precon(None)
        .with_cell(None)
}

fn run_band<S: PointSurface>(
    surface: &S,
    config: &HopSearchConfig,
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
        } else if abort_on_spike
            && best_force < tol.max(0.05)
            && report.max_force > best_force * 8.0
        {
            break;
        }
    }
    let (cartesian, found_cells, force) = if report.max_force <= best_force {
        (
            neb.cartesian()?,
            neb.cells()?.unwrap_or_default(),
            report.max_force,
        )
    } else {
        (best, best_cells, best_force)
    };
    Ok((cartesian, found_cells, force))
}

/// Run every seed. Index-1 saddles are returned with both pushed basins.
/// A seed that does not converge, or that is not index 1, is omitted.
pub fn seeded_saddles<S: PointSurface + Sync>(
    surface: &S,
    config: &HopSearchConfig,
    cell: M3,
    seeds: &[DimerSeed],
) -> Vec<BasinSaddle> {
    seeds
        .par_iter()
        .filter_map(|seed| one_seed(surface, config, cell, seed).ok())
        .collect()
}

fn one_seed<S: PointSurface>(
    surface: &S,
    config: &HopSearchConfig,
    cell: M3,
    seed: &DimerSeed,
) -> Result<BasinSaddle, SaddleError> {
    if seed.mode.dot(&seed.mode) < 1.0e-16 {
        return Err(SaddleError::Invalid("dimer seed mode vanishes".into()));
    }
    let mut dimer = SolidDimer::new(
        config.dimer.clone(),
        seed.positions.clone(),
        cell,
        seed.mode.clone(),
    )?;
    let report = dimer.run(surface, config.max_dimer_steps)?;
    if report.status != MinModeStatus::Converged
        || report.rejected
        || report.negative_modes != Some(1)
    {
        return Err(SaddleError::Invalid("seed is not an index-1 saddle".into()));
    }
    let saddle = dimer.positions().to_owned();
    let mode = dimer.mode().to_owned();
    let push = cartesian_push(&dimer, config.push)?;
    let (plus, plus_energy) = relax(surface, &saddle + &push, cell, config)?;
    let (minus, minus_energy) = relax(surface, &saddle - &push, cell, config)?;
    let (saddle_energy, _) = surface.eval(saddle.view())?;
    Ok(BasinSaddle {
        saddle,
        saddle_energy,
        negative_eigenvalue: report.eigenvalues.first().copied(),
        mode,
        plus,
        minus,
        plus_energy,
        minus_energy,
    })
}

/// Follow `config.relax` from `positions` until the energy is at or below
/// `floor` or the largest per-atom force is below `config.relax.force_tol`.
///
/// The floor ends a descent from a saddle well before a deep minimum, so a
/// band can end there instead of across the whole drop. Reaching
/// `config.relax.max_steps` first, or a non-finite energy, is an error.
pub fn descend_to_floor<S: PointSurface>(
    surface: &S,
    mut positions: Array1<f64>,
    cell: M3,
    config: &HopSearchConfig,
    floor: f64,
) -> Result<(Array1<f64>, f64), SaddleError> {
    let relax = &config.relax;
    relax.check()?;
    let flat = m3_to_row9(cell);
    let (mut energy, gradient) = surface.eval_in_cell(positions.view(), &flat)?;
    if !energy.is_finite() {
        return Err(SaddleError::NonFinite("descent start energy"));
    }
    if energy <= floor || max_atom_force(gradient.view()) < relax.force_tol {
        return Ok((positions, energy));
    }
    let mut solver = rgmin::Solver::new(
        relax.method.clone(),
        rgmin::Control::default()
            .with_maxiter(usize::MAX)
            .with_gtol(0.0)
            .with_istep(relax.initial_step),
        positions.len(),
    );
    solver.set_atom_maxmove(relax.max_atom_move);
    let oracle = rgmin::Oracle::unbounded(positions.len(), |x: ArrayView1<f64>| {
        surface
            .eval_in_cell(x, &flat)
            .unwrap_or_else(|_| (f64::NAN, Array1::from_elem(x.len(), f64::NAN)))
    });
    let mut largest = f64::INFINITY;
    for _ in 0..relax.max_steps {
        solver
            .step(&oracle, &mut positions)
            .map_err(|error| SaddleError::Solver(error.to_string()))?;
        let (next, gradient) = surface.eval_in_cell(positions.view(), &flat)?;
        if !next.is_finite() {
            return Err(SaddleError::NonFinite("descent energy"));
        }
        energy = next;
        largest = max_atom_force(gradient.view());
        if energy <= floor || largest < relax.force_tol {
            return Ok((positions, energy));
        }
    }
    Err(SaddleError::Solver(format!(
        "descent not converged: energy {energy:.6} above the floor {floor:.6} and largest atom force {largest:.3e} not below {:.1e} after {} steps",
        relax.force_tol, relax.max_steps
    )))
}

/// Climbing band from the reactant through `saddle` to a nearby downhill
/// geometry. Springs are energy-weighted so images stay near the barrier.
#[allow(clippy::too_many_arguments)]
pub fn segment_band<S: PointSurface>(
    surface: &S,
    config: &HopSearchConfig,
    reactant: &Array1<f64>,
    saddle: &Array1<f64>,
    downhill: &Array1<f64>,
    cell: M3,
    k_min: f64,
    k_max: f64,
) -> Result<(ArcPath, bool, f64), SaddleError> {
    let band = config
        .band
        .clone()
        .with_spring(crate::spring::SpringKind::EnergyWeighted { k_min, k_max });
    let config = config.clone().with_band(band);
    climb_path(surface, &config, reactant, saddle, downhill, cell, cell)
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
        step.slice_mut(s![..length])
            .assign(&mode.slice(s![..length]));
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
/// RMSD depend on which atom of a split site moved; the clouds do not.
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

pub(crate) fn relax<S: PointSurface>(
    surface: &S,
    positions: Array1<f64>,
    cell: M3,
    config: &HopSearchConfig,
) -> Result<(Array1<f64>, f64), SaddleError> {
    if config.cell_mode == CellMode::Variable {
        return relax_joint(surface, positions, cell, config);
    }
    let done = relax_to_tolerance(surface, positions, cell, &config.relax)?;
    Ok((done.positions, done.energy))
}

fn relax_joint<S: PointSurface>(
    surface: &S,
    positions: Array1<f64>,
    cell: M3,
    config: &HopSearchConfig,
) -> Result<(Array1<f64>, f64), SaddleError> {
    let n_atoms = positions.len() / 3;
    let chart = JointChart::new(cell, n_atoms, CellMask::all(), config.external_stress, 1.0)?;
    let mut q = Array1::zeros(chart.dimension());
    q.slice_mut(s![..positions.len()]).assign(&positions);
    let relax = &config.relax;
    let mut solver = rgmin::Solver::new(
        relax.method.clone(),
        rgmin::Control::default()
            .with_maxiter(relax.max_steps)
            .with_gtol(relax.force_tol)
            .with_stop_norm(rgmin::StopNorm::MaxAtom)
            .with_istep(relax.initial_step),
        q.len(),
    );
    solver.set_atom_maxmove(relax.max_atom_move);
    let oracle = rgmin::Oracle::unbounded(q.len(), |q: ArrayView1<f64>| {
        match joint_gradient(surface, &chart, q) {
            Ok(pair) => pair,
            Err(_) => (f64::NAN, Array1::zeros(q.len())),
        }
    });
    let report = solver
        .run(&oracle, &mut q)
        .map_err(|error| SaddleError::Solver(error.to_string()))?;
    let (energy, gradient) = joint_gradient(surface, &chart, q.view())?;
    let largest = max_atom_force(gradient.view());
    if !(energy.is_finite() && largest < relax.force_tol) {
        return Err(SaddleError::Solver(format!(
            "joint relaxation not converged: largest generalized force {largest:.3e} is not below {:.1e} after {} of {} steps ({:?})",
            relax.force_tol, report.steps, relax.max_steps, report.status
        )));
    }
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

#[allow(clippy::type_complexity)]
pub(crate) fn hessian<S: PointSurface>(
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
        return Err(SaddleError::Invalid("the harmonic region is empty".into()));
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

pub(crate) fn plane_arcs(path: &ArcPath, planes: usize) -> Vec<f64> {
    let nodes = path.arcs();
    if planes < 2 {
        return nodes;
    }
    let (start, end) = path.bounds();
    (0..planes)
        .map(|i| start + (end - start) * (i as f64) / (planes - 1) as f64)
        .collect()
}

pub(crate) fn check_pair_geometry(setup: &PairSetup) -> Result<(), SaddleError> {
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
    Ok(())
}

pub(crate) fn check_hop_template(hop: &HopTemplate, n: usize) -> Result<(), SaddleError> {
    if hop.product.len() != n {
        return Err(SaddleError::Shape(
            "a hop product must have the reactant length".into(),
        ));
    }
    if hop.name.is_empty() {
        return Err(SaddleError::Invalid("a hop needs a name".into()));
    }
    Ok(())
}

/// Validate a [`PairSetup`], including a non-empty hop catalogue.
pub(crate) fn check_setup(setup: &PairSetup) -> Result<(), SaddleError> {
    check_pair_geometry(setup)?;
    if setup.hops.is_empty() {
        return Err(SaddleError::Invalid("the hop catalogue is empty".into()));
    }
    for hop in &setup.hops {
        check_hop_template(hop, setup.positions.len())?;
    }
    Ok(())
}

pub(crate) fn check_config(config: &HopSearchConfig) -> Result<(), SaddleError> {
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
    config.relax.check()?;
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

fn mix(
    start: &Array1<f64>,
    end: &Array1<f64>,
    fraction: f64,
    cell: M3,
) -> Result<Array1<f64>, SaddleError> {
    let delta = mic_positions(end, start, cell)?;
    Ok(start + delta * fraction)
}

fn mic_positions(
    end: &Array1<f64>,
    start: &Array1<f64>,
    cell: M3,
) -> Result<Array1<f64>, SaddleError> {
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

fn mobile_center(
    positions: ArrayView1<f64>,
    cell: M3,
    mobile: &[usize],
) -> Result<[f64; 3], SaddleError> {
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

#[allow(clippy::needless_range_loop)]
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

pub(crate) fn norm1(v: &Array1<f64>) -> f64 {
    v.dot(v).sqrt()
}

pub(crate) fn unit(v: &Array1<f64>) -> Result<Array1<f64>, SaddleError> {
    let scale = norm1(v);
    if scale == 0.0 {
        return Err(SaddleError::Invalid("a path tangent vanishes".into()));
    }
    Ok(v / scale)
}

fn unflat(cell: &[f64; 9]) -> M3 {
    [
        [cell[0], cell[1], cell[2]],
        [cell[3], cell[4], cell[5]],
        [cell[6], cell[7], cell[8]],
    ]
}
