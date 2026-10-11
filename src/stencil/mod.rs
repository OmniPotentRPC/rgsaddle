//! One builder for the band methods and the point methods.
//!
//! Goswami, Gunde, and Jónsson, Front. Chem. 14, 1807063 (2026),
//! doi:10.3389/fchem.2026.1807063, treat the climbing image and the
//! dimer translation as one Householder reflection. The band side of
//! this builder is CI-NEB, AutoNEB, and zoom-NEB. Zoom-NEB follows
//! eOn 3.4.0 (doi:10.5281/zenodo.23166734) with fixed window ends.
//! [`ZoomEnds::FreePerp`] is ORCA's free-end variant and is off unless
//! asked for. Energy-weighted springs follow Ásgeirsson et al.,
//! J. Chem. Theory Comput. 17, 4929 (2021),
//! doi:10.1021/acs.jctc.1c00462. AutoNEB follows Kolsbjerg, Groves,
//! and Hammer, J. Chem. Phys. 145, 094107 (2016),
//! doi:10.1063/1.4961868.
//!
//! The force in every report is the maximum absolute component
//! ([`crate::force::ForceGate::LinfNorm`]). There is no SIMD path and
//! no BLAS call on this path. Image blocks are a fixed partition, so
//! the result does not depend on the thread count.

mod parallel;
mod point;
pub mod spring_ew;
pub mod zoom;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::band::{BandConfig, BandReport, BandSession, BandStatus, BandSurface, CiConfig};
use crate::error::SaddleError;
use crate::force::ForceGate;
use crate::minmode::MinModeConfig;
use crate::spring::SpringKind;

pub use parallel::{ImagePool, image_block};
pub use spring_ew::{K_MAX_EV_PER_A2, K_MIN_EV_PER_A2};
pub use zoom::{Interp, ZoomEnds, ZoomForce, ZoomParams, ZoomWatch, ZoomWindow};

/// Stable identity of one image. It survives insertion and a zoom pack.
pub type ImageId = u64;

/// C ordinal of a band or point method.
pub fn method_ordinal(method: &Method) -> i32 {
    match method {
        Method::Band(BandMethod::Plain) => 0,
        Method::Band(BandMethod::CiNeb) => 1,
        Method::Band(BandMethod::AutoNeb(_)) => 2,
        Method::Band(BandMethod::Zoom(_)) => 3,
        Method::Point(PointMethod::Dimer) => 4,
        Method::Point(PointMethod::Lanczos) => 5,
        Method::Point(PointMethod::Art) => 6,
    }
}

/// C ordinal of a preset.
pub fn preset_ordinal(preset: Preset) -> i32 {
    match preset {
        Preset::CiNeb => 0,
        Preset::EonZoom => 1,
        Preset::OrcaZoomNebTs => 2,
        Preset::AutoNeb => 3,
        Preset::Dimer => 4,
        Preset::Lanczos => 5,
        Preset::Art => 6,
        Preset::Plain => 7,
    }
}

/// C ordinal of a phase.
pub fn phase_ordinal(phase: Phase) -> i32 {
    match phase {
        Phase::Band => 0,
        Phase::Zoom => 1,
        Phase::Insert => 2,
        Phase::Align => 3,
        Phase::MinMode => 4,
        Phase::Ef => 5,
        Phase::Rotate => 6,
        Phase::Translate => 7,
    }
}

/// What re-meshes the band. Each variant has a C ordinal.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum BandMethod {
    /// Fixed images, no climbing.
    Plain,
    /// Fixed images, climbing image armed by [`CiConfig`].
    CiNeb,
    /// Kolsbjerg, Groves, and Hammer (2016) insertion, then a climb.
    AutoNeb(AutoNebParams),
    /// CI-NEB until the climbing image is stable, then one zoom.
    Zoom(ZoomParams),
}

/// How the springs are chosen.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum SpringModel {
    Uniform {
        k: f64,
    },
    /// Recomputed from the energies of the same evaluation.
    EnergyWeighted {
        k_min: f64,
        k_max: f64,
    },
    OnsagerMachlup {
        k: f64,
    },
    /// Host-supplied list. Length `n_images - 1`.
    Explicit {
        ks: Vec<f64>,
    },
}

/// What runs after the band, if anything.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Refine {
    None,
    /// OCI-NEB hand-off. Needs rgsaddle#11, which is not on this branch.
    Ocineb(OcinebParams),
    /// NEB-TS hand-off. The tangent-overlap P-RFO adapter is follow-up work.
    EigenvectorFollowing(EfParams),
}

/// Point method. Each variant has a C ordinal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PointMethod {
    /// Henkelman and Jónsson (1999) dimer.
    Dimer,
    /// Lanczos minimum mode.
    Lanczos,
    /// ART nouveau, once `feat/art-nouveau` is on main.
    Art,
}

/// Band or point.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Method {
    Band(BandMethod),
    Point(PointMethod),
}

/// Named recipe. The rustdoc lists every value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Preset {
    /// CI-NEB, uniform spring `k = 5`, eOn climbing trigger, CI tolerance `1e-3`.
    #[default]
    CiNeb,
    /// eOn 3.4.0 zoom.
    ///
    /// CI-NEB, improved tangent, NEB projection, uniform spring `k = 5`
    /// eV/Å^2, climbing trigger factor 0.5 and trigger force 0, zoom
    /// auto `alpha = 0.5` and `offset = 1`, cubic Hermite, fixed ends,
    /// stability 5, relative force 10, CI-only convergence with slack
    /// 10. FIRE, `max_move = 0.2`. Source: eOn `neb_options` at 3.4.0
    /// and doi:10.5281/zenodo.23166734.
    EonZoom,
    /// ORCA ZOOM-NEB-TS parameters converted to eV and Å.
    ///
    /// Energy-weighted springs 0.972 to 9.72 eV/Å^2 (ORCA
    /// `SpringConst` 0.01 and `SpringConst2` 0.1 Eh/Bohr^2), linear
    /// pack, free ends, `Tol_Scale` 10, zoom force 5.142067 eV/Å
    /// (ORCA `Tol_Turn_On_Zoom` 0.1 Eh/Bohr), stability 5,
    /// eigenvector-following activate 0.102844 eV/Å (ORCA 2e-3 Eh/Bohr).
    /// The P-RFO hand-off returns an error on this branch.
    OrcaZoomNebTs,
    /// AutoNEB at [`AutoNebParams::default`].
    AutoNeb,
    /// Dimer through [`crate::minmode::MinModeSession`].
    Dimer,
    /// Lanczos through [`crate::minmode::MinModeSession`].
    Lanczos,
    /// Refused until `feat/art-nouveau` is on main.
    Art,
    /// Fixed band, climbing off.
    Plain,
}

/// AutoNEB controls on the stencil. Mirrors [`crate::autoneb::AutoNebTuning`].
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct AutoNebParams {
    pub n_simul: usize,
    pub n_max: usize,
    pub local_force_tol: f64,
    pub local_max_steps: usize,
    pub climb_max_steps: usize,
    pub space_energy_ratio: f64,
    pub climb: bool,
    pub full_band: bool,
}

impl Default for AutoNebParams {
    fn default() -> Self {
        Self {
            n_simul: 3,
            n_max: 7,
            local_force_tol: 1e-1,
            local_max_steps: 100,
            climb_max_steps: 2000,
            space_energy_ratio: 0.5,
            climb: true,
            full_band: false,
        }
    }
}

/// Placeholder for the OCI-NEB hand-off (rgsaddle#11).
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct OcinebParams {
    pub trust: f64,
}

impl Default for OcinebParams {
    fn default() -> Self {
        Self { trust: 0.1 }
    }
}

/// NEB-TS hand-off tolerances. `activate` is the climbing-image force
/// at which the band would stop. ORCA's TS default is 2e-3 Eh/Bohr.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct EfParams {
    pub activate: f64,
}

impl Default for EfParams {
    fn default() -> Self {
        Self { activate: 0.102844 }
    }
}

/// When a band reports convergence.
///
/// `max_force` in the report is the maximum absolute component. With
/// `image_scale`, the climbing image must be within `ci_tol` and every
/// other moving image within `image_scale * ci_tol` (ORCA `Tol_Scale`).
/// With `image_scale == None`, only the climbing image is required,
/// and convergence is refused while any moving image exceeds
/// `band_slack * ci_tol` (eOn `climbing_image_band_slack`). Without
/// climbing, every interior image must be within `ci_tol`. `ci_rms_tol`
/// adds the root-mean-square test on the climbing image.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct Convergence {
    pub ci_tol: f64,
    pub ci_rms_tol: Option<f64>,
    pub image_scale: Option<f64>,
    pub band_slack: f64,
}

impl Default for Convergence {
    fn default() -> Self {
        Self {
            ci_tol: 1e-3,
            ci_rms_tol: None,
            image_scale: None,
            band_slack: 10.0,
        }
    }
}

/// Full configuration. One leaf per C key.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct StencilConfig {
    pub method: Method,
    pub spring: SpringModel,
    pub refine: Refine,
    pub climb: CiConfig,
    pub converge: Convergence,
    pub band: BandConfig,
    pub point: MinModeConfig,
    pub threads: usize,
}

impl Default for StencilConfig {
    fn default() -> Self {
        Self::from_preset(Preset::CiNeb)
    }
}

impl StencilConfig {
    /// Every default of `preset`.
    pub fn from_preset(preset: Preset) -> Self {
        let mut config = Self {
            method: Method::Band(BandMethod::CiNeb),
            spring: SpringModel::Uniform { k: 5.0 },
            refine: Refine::None,
            climb: CiConfig {
                trigger_factor: 0.5,
                trigger_force: 0.0,
            },
            converge: Convergence::default(),
            band: BandConfig::default(),
            point: MinModeConfig::default(),
            threads: 1,
        };
        match preset {
            Preset::CiNeb => {}
            Preset::Plain => {
                config.method = Method::Band(BandMethod::Plain);
                config.band.climbing = None;
            }
            Preset::EonZoom => {
                config.method = Method::Band(BandMethod::Zoom(ZoomParams::default()));
            }
            Preset::OrcaZoomNebTs => {
                config.method = Method::Band(BandMethod::Zoom(ZoomParams {
                    stability: 5,
                    force: ZoomForce::Abs(5.142067),
                    window: ZoomWindow::Auto {
                        alpha: 0.5,
                        offset: 1,
                    },
                    interpolation: Interp::Linear,
                    ends: ZoomEnds::FreePerp,
                    max_iterations: 0,
                }));
                config.spring = SpringModel::EnergyWeighted {
                    k_min: K_MIN_EV_PER_A2,
                    k_max: K_MAX_EV_PER_A2,
                };
                config.converge.image_scale = Some(10.0);
                config.refine = Refine::EigenvectorFollowing(EfParams::default());
            }
            Preset::AutoNeb => {
                config.method = Method::Band(BandMethod::AutoNeb(AutoNebParams::default()));
            }
            Preset::Dimer => config.method = Method::Point(PointMethod::Dimer),
            Preset::Lanczos => config.method = Method::Point(PointMethod::Lanczos),
            Preset::Art => config.method = Method::Point(PointMethod::Art),
        }
        config
    }
}

/// Where a step spent its work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Phase {
    Band,
    Zoom,
    Insert,
    Align,
    MinMode,
    Ef,
    Rotate,
    Translate,
}

/// One step.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct StencilReport {
    pub phase: Phase,
    pub status: BandStatus,
    pub max_force: f64,
    pub ci_index: Option<usize>,
    pub iteration: usize,
    pub surface_rows: usize,
    pub total_rows: usize,
    /// Band indices before the pack. Set on the zoom step.
    pub zoom_window: Option<(usize, usize)>,
    pub curvature: Option<f64>,
    /// Name of the force measure. This branch uses the maximum absolute component.
    pub force_gate: &'static str,
}

/// Host surface. A point evaluation is a one-row call.
///
/// `ids` stay with the image across insertion and the zoom pack.
pub trait StencilSurface: Sync {
    fn eval(
        &self,
        ids: &[ImageId],
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError>;
}

impl<S: BandSurface + ?Sized> StencilSurface for S {
    fn eval(
        &self,
        _ids: &[ImageId],
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        BandSurface::eval(self, positions, energies, gradients)
    }
}

/// Adapts a [`crate::minmode::PointSurface`] to one row per call, in order.
pub struct PointRows<S>(pub S);

impl<S: crate::minmode::PointSurface> StencilSurface for PointRows<S> {
    fn eval(
        &self,
        _ids: &[ImageId],
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (row, energy_slot) in energies.iter_mut().enumerate() {
            let (energy, gradient) = self.0.eval(positions.row(row))?;
            *energy_slot = energy;
            gradients.row_mut(row).assign(&gradient);
        }
        Ok(())
    }
}

/// Fluent builder. `build` checks every combination.
pub struct StencilBuilder {
    initial: Array2<f64>,
    point: bool,
    config: StencilConfig,
    asked_zoom: bool,
    asked_auto: bool,
    asked_spring: bool,
    asked_refine: bool,
    asked_idpp: bool,
}

impl StencilBuilder {
    fn new(initial: Array2<f64>, point: bool) -> Self {
        let config = if point {
            StencilConfig::from_preset(Preset::Dimer)
        } else {
            StencilConfig::default()
        };
        Self {
            initial,
            point,
            config,
            asked_zoom: false,
            asked_auto: false,
            asked_spring: false,
            asked_refine: false,
            asked_idpp: false,
        }
    }

    pub fn plain(mut self) -> Self {
        self.config.method = Method::Band(BandMethod::Plain);
        self.config.band.climbing = None;
        self
    }

    pub fn ci_neb(mut self) -> Self {
        self.config.method = Method::Band(BandMethod::CiNeb);
        self
    }

    pub fn autoneb(mut self, params: AutoNebParams) -> Self {
        self.asked_auto = true;
        self.config.method = Method::Band(BandMethod::AutoNeb(params));
        self
    }

    pub fn zoom(mut self, params: ZoomParams) -> Self {
        self.asked_zoom = true;
        self.config.method = Method::Band(BandMethod::Zoom(params));
        self
    }

    pub fn energy_weighted(mut self, k_min: f64, k_max: f64) -> Self {
        self.asked_spring = true;
        self.config.spring = SpringModel::EnergyWeighted { k_min, k_max };
        self
    }

    pub fn uniform(mut self, k: f64) -> Self {
        self.asked_spring = true;
        self.config.spring = SpringModel::Uniform { k };
        self
    }

    pub fn explicit_springs(mut self, ks: Vec<f64>) -> Self {
        self.asked_spring = true;
        self.config.spring = SpringModel::Explicit { ks };
        self
    }

    pub fn ocineb(mut self, params: OcinebParams) -> Self {
        self.asked_refine = true;
        self.config.refine = Refine::Ocineb(params);
        self
    }

    pub fn ef(mut self, params: EfParams) -> Self {
        self.asked_refine = true;
        self.config.refine = Refine::EigenvectorFollowing(params);
        self
    }

    pub fn dimer(mut self) -> Self {
        self.config.method = Method::Point(PointMethod::Dimer);
        self.point = true;
        self
    }

    pub fn lanczos(mut self) -> Self {
        self.config.method = Method::Point(PointMethod::Lanczos);
        self.point = true;
        self
    }

    pub fn art(mut self) -> Self {
        self.config.method = Method::Point(PointMethod::Art);
        self.point = true;
        self
    }

    pub fn threads(mut self, threads: usize) -> Self {
        self.config.threads = threads;
        self
    }

    pub fn preset(mut self, preset: Preset) -> Self {
        let threads = self.config.threads;
        self.config = StencilConfig::from_preset(preset);
        self.config.threads = threads;
        self.point = matches!(preset, Preset::Dimer | Preset::Lanczos | Preset::Art);
        self.asked_zoom = matches!(preset, Preset::EonZoom | Preset::OrcaZoomNebTs);
        self.asked_auto = matches!(preset, Preset::AutoNeb);
        self.asked_spring = matches!(preset, Preset::OrcaZoomNebTs);
        self.asked_refine = matches!(preset, Preset::OrcaZoomNebTs);
        self.asked_idpp = false;
        self
    }

    /// IDPP initial paths are not on this branch.
    pub fn idpp(mut self) -> Self {
        self.asked_idpp = true;
        self
    }

    pub fn converge(mut self, converge: Convergence) -> Self {
        self.config.converge = converge;
        self
    }

    pub fn build(self) -> Result<Stencil, SaddleError> {
        Stencil::open(
            self.config,
            self.initial,
            BuilderFlags {
                asked_zoom: self.asked_zoom,
                asked_auto: self.asked_auto,
                asked_spring: self.asked_spring,
                asked_refine: self.asked_refine,
                asked_idpp: self.asked_idpp,
            },
        )
    }
}

struct BuilderFlags {
    asked_zoom: bool,
    asked_auto: bool,
    asked_spring: bool,
    asked_refine: bool,
    asked_idpp: bool,
}

struct BandRun {
    session: BandSession,
    ids: Vec<ImageId>,
    next_id: ImageId,
    watch: Option<ZoomWatch>,
    zoom_due: bool,
    zoomed: bool,
    window: Option<(usize, usize)>,
    post_steps: usize,
    zoom_cap: usize,
    free_ends: bool,
    total_rows: usize,
    iteration: usize,
    ew: Option<(f64, f64)>,
    template: BandConfig,
    cell: Option<crate::band::Cell>,
    refine: Refine,
    plain: bool,
}

enum Engine {
    Band(Box<BandRun>),
    Auto(Box<AutoRun>),
    Point(Box<point::PointRun>),
}

struct AutoRun {
    search: crate::autoneb::AutoNeb,
    ids: Vec<ImageId>,
    next_id: ImageId,
    seen_insertions: usize,
    total_rows: usize,
}

/// Band or point search.
pub struct Stencil {
    engine: Engine,
    pool: ImagePool,
    converge: Convergence,
    threads: usize,
}

impl Stencil {
    /// Rows are images, endpoints included.
    pub fn band(initial: Array2<f64>) -> StencilBuilder {
        StencilBuilder::new(initial, false)
    }

    /// One geometry.
    pub fn point(x0: Array1<f64>) -> StencilBuilder {
        let mut initial = Array2::zeros((1, x0.len()));
        initial.row_mut(0).assign(&x0);
        StencilBuilder::new(initial, true)
    }

    /// Build from a filled config. `initial` is the band, or one row for a point method.
    pub fn new(config: StencilConfig, initial: Array2<f64>) -> Result<Self, SaddleError> {
        let flags = BuilderFlags {
            asked_zoom: matches!(config.method, Method::Band(BandMethod::Zoom(_))),
            asked_auto: matches!(config.method, Method::Band(BandMethod::AutoNeb(_))),
            asked_spring: !matches!(config.spring, SpringModel::Uniform { k } if k == 5.0),
            asked_refine: !matches!(config.refine, Refine::None),
            asked_idpp: false,
        };
        Self::open(config, initial, flags)
    }

    fn open(
        config: StencilConfig,
        initial: Array2<f64>,
        flags: BuilderFlags,
    ) -> Result<Self, SaddleError> {
        validate(&config, &initial, &flags)?;
        let threads = config.threads;
        let (engine, rows, dof) = match &config.method {
            Method::Point(kind) => {
                let run =
                    point::PointRun::new(config.point.clone(), *kind, initial.row(0).to_owned())?;
                (Engine::Point(Box::new(run)), 1, initial.ncols())
            }
            Method::Band(BandMethod::AutoNeb(params)) => {
                let mut band = apply_spring(config.band.clone(), &config.spring, initial.nrows())?;
                band.climbing = Some(config.climb);
                band.autoneb = Some(crate::autoneb::AutoNebTuning {
                    n_simul: params.n_simul,
                    n_max: params.n_max,
                    local_force_tol: params.local_force_tol,
                    local_max_steps: params.local_max_steps,
                    climb_max_steps: params.climb_max_steps,
                    space_energy_ratio: params.space_energy_ratio,
                    climb: params.climb,
                    full_band: params.full_band,
                    threads,
                    record_samples: false,
                });
                let n_user = initial.nrows();
                let search = crate::autoneb::AutoNeb::new(band, initial)?;
                let mut ids: Vec<ImageId> = (0..n_user as ImageId).collect();
                let mut next_id = n_user as ImageId;
                for &after in search.insertions() {
                    ids.insert(after + 1, next_id);
                    next_id += 1;
                }
                let seen_insertions = search.insertions().len();
                let rows = search.positions().nrows();
                let dof = search.positions().ncols();
                (
                    Engine::Auto(Box::new(AutoRun {
                        search,
                        ids,
                        next_id,
                        seen_insertions,
                        total_rows: 0,
                    })),
                    rows,
                    dof,
                )
            }
            Method::Band(method) => {
                let n = initial.nrows();
                let dof = initial.ncols();
                let (zoom, plain) = match method {
                    BandMethod::Zoom(params) => (Some(*params), false),
                    BandMethod::Plain => (None, true),
                    BandMethod::CiNeb => (None, false),
                    BandMethod::AutoNeb(_) => unreachable!("autoneb is a separate engine"),
                };
                let mut band = apply_spring(config.band.clone(), &config.spring, n)?;
                if plain {
                    band.climbing = None;
                } else {
                    band.climbing = Some(config.climb);
                }
                band.force_tol = config.converge.ci_tol;
                band.autoneb = None;
                let template = band.clone();
                let mut session = BandSession::new(band, initial)?;
                session.set_force_gate(ForceGate::LinfNorm);
                let ew = match config.spring {
                    SpringModel::EnergyWeighted { k_min, k_max } => Some((k_min, k_max)),
                    _ => None,
                };
                (
                    Engine::Band(Box::new(BandRun {
                        session,
                        ids: (0..n as ImageId).collect(),
                        next_id: n as ImageId,
                        watch: zoom.map(ZoomWatch::new),
                        zoom_due: false,
                        zoomed: false,
                        window: None,
                        post_steps: 0,
                        zoom_cap: zoom.map(|params| params.max_iterations).unwrap_or(0),
                        free_ends: false,
                        total_rows: 0,
                        iteration: 0,
                        ew,
                        template,
                        cell: config.band.cell,
                        refine: config.refine.clone(),
                        plain,
                    })),
                    n,
                    dof,
                )
            }
        };
        // The pool is sized for the band that exists at build. AutoNEB
        // grows inside its own pool; this one serves the fixed band.
        let pool = ImagePool::new(threads.max(1), rows.max(1), dof)?;
        Ok(Self {
            engine,
            pool,
            converge: config.converge,
            threads,
        })
    }

    pub fn step<S: StencilSurface + ?Sized>(
        &mut self,
        surface: &S,
    ) -> Result<StencilReport, SaddleError> {
        match &mut self.engine {
            Engine::Point(run) => run.step(surface),
            Engine::Auto(run) => step_auto(run, surface),
            Engine::Band(run) => step_band(run, &self.pool, &self.converge, surface),
        }
    }

    pub fn run<S: StencilSurface + ?Sized>(
        &mut self,
        surface: &S,
        max_steps: usize,
    ) -> Result<StencilReport, SaddleError> {
        let mut last = None;
        for _ in 0..max_steps {
            let report = self.step(surface)?;
            let done = report.status == BandStatus::Converged;
            last = Some(report);
            if done {
                break;
            }
        }
        last.ok_or_else(|| SaddleError::Invalid("stencil run needs at least one step".into()))
    }

    pub fn positions(&self) -> ArrayView2<'_, f64> {
        match &self.engine {
            Engine::Band(run) => run.session.positions(),
            Engine::Auto(run) => run.search.positions(),
            Engine::Point(run) => run.matrix(),
        }
    }

    pub fn point_position(&self) -> Option<ArrayView1<'_, f64>> {
        match &self.engine {
            Engine::Point(run) => Some(run.positions()),
            _ => None,
        }
    }

    pub fn energies(&self) -> Option<Array1<f64>> {
        match &self.engine {
            Engine::Band(run) => run.session.evaluation().map(|eval| eval.energies),
            Engine::Auto(run) => Some(
                run.search
                    .energies()
                    .iter()
                    .map(|energy| energy.unwrap_or(f64::NAN))
                    .collect(),
            ),
            Engine::Point(_) => None,
        }
    }

    pub fn climbing_image(&self) -> Option<usize> {
        match &self.engine {
            Engine::Band(run) => run.session.climbing_image(),
            Engine::Auto(run) => run.search.climbing_image(),
            Engine::Point(_) => None,
        }
    }

    pub fn image_ids(&self) -> &[ImageId] {
        match &self.engine {
            Engine::Band(run) => &run.ids,
            Engine::Auto(run) => &run.ids,
            Engine::Point(_) => &[],
        }
    }

    pub fn mode(&self) -> Option<ArrayView1<'_, f64>> {
        match &self.engine {
            Engine::Point(run) => Some(run.mode()),
            _ => None,
        }
    }

    pub fn curvature(&self) -> Option<f64> {
        match &self.engine {
            Engine::Point(run) => Some(run.curvature()),
            _ => None,
        }
    }

    pub fn spring_constants(&self) -> Vec<f64> {
        match &self.engine {
            Engine::Band(run) => run.session.spring_constants(),
            Engine::Auto(run) => run.search.springs().to_vec(),
            Engine::Point(_) => Vec::new(),
        }
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Band indices of the packed window, after the zoom step.
    pub fn zoom_window(&self) -> Option<(usize, usize)> {
        match &self.engine {
            Engine::Band(run) => run.window,
            _ => None,
        }
    }
}

fn validate(
    config: &StencilConfig,
    initial: &Array2<f64>,
    flags: &BuilderFlags,
) -> Result<(), SaddleError> {
    if flags.asked_idpp {
        return Err(SaddleError::Invalid(
            "IDPP initial path needs rgsaddle#10".into(),
        ));
    }
    if config.threads == 0 {
        return Err(SaddleError::Invalid(
            "stencil needs at least one image-evaluation thread".into(),
        ));
    }
    if flags.asked_zoom && flags.asked_auto {
        return Err(SaddleError::Invalid(
            "AutoNEB and zoom-NEB cannot be combined; both re-mesh the band".into(),
        ));
    }
    let point = matches!(config.method, Method::Point(_));
    if point && (flags.asked_spring || flags.asked_zoom || flags.asked_auto || flags.asked_refine) {
        return Err(SaddleError::Invalid(
            "a point stencil cannot take band options (spring, zoom, autoneb, refine)".into(),
        ));
    }
    if point && initial.nrows() != 1 {
        return Err(SaddleError::Invalid(
            "a point method needs one row; the initial array has more".into(),
        ));
    }
    if !point && initial.nrows() < 2 {
        return Err(SaddleError::Invalid(
            "a band method needs at least the two endpoints".into(),
        ));
    }
    match &config.spring {
        SpringModel::EnergyWeighted { k_min, k_max } => {
            if !(k_min.is_finite() && k_max.is_finite() && *k_min > 0.0 && *k_max > 0.0) {
                return Err(SaddleError::Invalid(
                    "energy-weighted springs need finite positive bounds".into(),
                ));
            }
            if k_min > k_max {
                return Err(SaddleError::Invalid(
                    "energy-weighted springs need k_min at most k_max".into(),
                ));
            }
        }
        SpringModel::Explicit { ks } if !point && ks.len() + 1 != initial.nrows() => {
            // AutoNEB may start from the endpoints and grow, so the
            // explicit list is checked against the array that was passed.
            if !matches!(config.method, Method::Band(BandMethod::AutoNeb(_))) {
                return Err(SaddleError::Invalid(format!(
                    "explicit springs need {} constants for {} images",
                    initial.nrows().saturating_sub(1),
                    initial.nrows()
                )));
            }
        }
        SpringModel::Uniform { k } if !k.is_finite() => {
            return Err(SaddleError::Invalid(
                "the uniform spring constant must be finite".into(),
            ));
        }
        _ => {}
    }
    if matches!(
        config.method,
        Method::Band(BandMethod::Zoom(_)) | Method::Band(BandMethod::AutoNeb(_))
    ) && config.band.solid.is_some()
    {
        return Err(SaddleError::Invalid(
            "zoom-NEB and AutoNEB do not carry the solid-state block".into(),
        ));
    }
    if let Method::Band(BandMethod::Zoom(_)) = &config.method
        && config.band.climbing.is_none()
    {
        return Err(SaddleError::Invalid(
            "zoom-NEB needs a climbing image".into(),
        ));
    }
    if matches!(config.refine, Refine::Ocineb(_)) {
        return Err(SaddleError::Invalid(
            "OCI-NEB refine needs rgsaddle#11".into(),
        ));
    }
    Ok(())
}

fn apply_spring(
    mut band: BandConfig,
    spring: &SpringModel,
    _n: usize,
) -> Result<BandConfig, SaddleError> {
    band.spring = match spring {
        SpringModel::Uniform { k } => SpringKind::Uniform { k: *k },
        SpringModel::EnergyWeighted { k_min, k_max } => SpringKind::EnergyWeighted {
            k_min: *k_min,
            k_max: *k_max,
        },
        SpringModel::OnsagerMachlup { k: _ } => {
            return Err(SaddleError::Invalid(
                "Onsager-Machlup springs need the host displacement list on BandConfig".into(),
            ));
        }
        SpringModel::Explicit { ks } => SpringKind::Weighted { ks: ks.clone() },
    };
    let _ = k_placeholder(spring);
    Ok(band)
}

fn k_placeholder(spring: &SpringModel) -> f64 {
    match spring {
        SpringModel::Uniform { k } | SpringModel::OnsagerMachlup { k } => *k,
        SpringModel::EnergyWeighted { k_min, .. } => *k_min,
        SpringModel::Explicit { ks } => ks.first().copied().unwrap_or(0.0),
    }
}

struct PoolSurface<'a, S: ?Sized> {
    surface: &'a S,
    pool: &'a ImagePool,
    ids: &'a [ImageId],
    free_ends: bool,
}

impl<S: StencilSurface + ?Sized> BandSurface for PoolSurface<'_, S> {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        let n = positions.nrows();
        let ids: &[ImageId] = if n == self.ids.len() {
            self.ids
        } else if !self.free_ends && self.ids.len() >= 2 && n + 2 == self.ids.len() {
            &self.ids[1..self.ids.len() - 1]
        } else {
            return Err(SaddleError::Shape(format!(
                "stencil surface expected {} ids and saw {n} rows",
                self.ids.len()
            )));
        };
        self.pool
            .eval(self.surface, ids, positions, energies, gradients)
    }
}

fn step_band<S: StencilSurface + ?Sized>(
    run: &mut BandRun,
    pool: &ImagePool,
    converge: &Convergence,
    surface: &S,
) -> Result<StencilReport, SaddleError> {
    if run.zoom_due {
        return pack_zoom(run);
    }
    if run.zoomed && run.zoom_cap > 0 && run.post_steps >= run.zoom_cap {
        return Ok(idle_report(run, Phase::Band));
    }
    maybe_handoff(run)?;
    let wrapped = PoolSurface {
        surface,
        pool,
        ids: &run.ids,
        free_ends: run.free_ends,
    };
    let report = run.session.step(&wrapped)?;
    run.total_rows += report.surface_rows;
    run.iteration += 1;
    if run.zoomed {
        run.post_steps += 1;
    }
    let judged = judge(run, converge, &report);
    if let Some(watch) = run.watch.as_mut()
        && !run.zoomed
        && watch.observe(
            run.ids.len(),
            !run.plain && run.session.climbing_image().is_some() || report.ci_index.is_some(),
            report.ci_index,
            judged.ci_force,
            converge.ci_tol,
        )
    {
        run.zoom_due = true;
    }
    Ok(StencilReport {
        phase: Phase::Band,
        status: if judged.converged {
            BandStatus::Converged
        } else {
            BandStatus::Running
        },
        max_force: judged.max_force,
        ci_index: report.ci_index,
        iteration: run.iteration,
        surface_rows: report.surface_rows,
        total_rows: run.total_rows,
        zoom_window: None,
        curvature: None,
        force_gate: "max-abs-component",
    })
}

struct Judged {
    converged: bool,
    max_force: f64,
    ci_force: f64,
}

fn judge(run: &BandRun, converge: &Convergence, report: &BandReport) -> Judged {
    let Some(projected) = run.session.last_projected() else {
        return Judged {
            converged: false,
            max_force: report.max_force,
            ci_force: report.max_force,
        };
    };
    let n_rows = projected.nrows();
    let mut ci_force = report.max_force;
    let mut worst = 0.0_f64;
    let mut ci_rms = 0.0_f64;
    let ci_row = report.ci_index.and_then(|ci| {
        if run.free_ends {
            (ci < n_rows).then_some(ci)
        } else {
            ci.checked_sub(1).filter(|row| *row < n_rows)
        }
    });
    for (row, force) in projected.outer_iter().enumerate() {
        let mut linf = 0.0_f64;
        let mut sum = 0.0_f64;
        for value in force.iter() {
            linf = linf.max(value.abs());
            sum += value * value;
        }
        let rms = if force.is_empty() {
            0.0
        } else {
            (sum / force.len() as f64).sqrt()
        };
        if linf > worst {
            worst = linf;
        }
        if Some(row) == ci_row {
            ci_force = linf;
            ci_rms = rms;
        }
    }
    let climbing = !run.plain && report.ci_index.is_some();
    let converged = if !climbing {
        projected
            .outer_iter()
            .all(|force| row_linf(force) <= converge.ci_tol)
    } else {
        let mut ok = ci_force <= converge.ci_tol;
        if let Some(rms_tol) = converge.ci_rms_tol {
            ok &= ci_rms <= rms_tol;
        }
        if let Some(scale) = converge.image_scale {
            ok &= projected.outer_iter().enumerate().all(|(row, force)| {
                Some(row) == ci_row || row_linf(force) <= scale * converge.ci_tol
            });
        } else {
            ok &= worst <= converge.band_slack * converge.ci_tol;
        }
        ok
    };
    Judged {
        converged,
        max_force: if climbing { ci_force } else { worst },
        ci_force,
    }
}

fn row_linf(force: ArrayView1<f64>) -> f64 {
    force
        .iter()
        .fold(0.0_f64, |max, value| max.max(value.abs()))
}

fn idle_report(run: &BandRun, phase: Phase) -> StencilReport {
    StencilReport {
        phase,
        status: BandStatus::Running,
        max_force: f64::INFINITY,
        ci_index: run.session.climbing_image(),
        iteration: run.iteration,
        surface_rows: 0,
        total_rows: run.total_rows,
        zoom_window: run.window,
        curvature: None,
        force_gate: "max-abs-component",
    }
}

fn maybe_handoff(run: &BandRun) -> Result<(), SaddleError> {
    if let Refine::EigenvectorFollowing(params) = &run.refine
        && let Some(ci) = run.session.climbing_image()
        && let Some(projected) = run.session.last_projected()
    {
        let row = if run.free_ends {
            ci
        } else {
            ci.saturating_sub(1)
        };
        if projected.nrows() > row && row_linf(projected.row(row)) <= params.activate {
            return Err(SaddleError::Invalid(
                "eigenvector-following refine needs a tangent-overlap P-RFO adapter; prfo.rs is unchanged on this branch"
                    .into(),
            ));
        }
    }
    Ok(())
}

fn pack_zoom(run: &mut BandRun) -> Result<StencilReport, SaddleError> {
    let params = run
        .watch
        .as_ref()
        .map(ZoomWatch::params)
        .ok_or_else(|| SaddleError::Invalid("zoom pack without zoom parameters".into()))?;
    let evaluation = run.session.evaluation().ok_or_else(|| {
        SaddleError::Invalid("zoom pack needs the energies of the last evaluation".into())
    })?;
    let ci = run
        .session
        .climbing_image()
        .ok_or_else(|| SaddleError::Invalid("zoom pack needs a climbing image".into()))?;
    let energies: Vec<f64> = evaluation.energies.iter().copied().collect();
    let Some((lo, hi)) = zoom::select_window(&energies, ci, params.window) else {
        run.zoom_due = false;
        run.zoomed = true;
        return Ok(idle_report(run, Phase::Zoom));
    };
    let positions = run.session.positions().to_owned();
    let count = positions.nrows();
    let packed = zoom::pack_window(
        positions.view(),
        lo,
        hi,
        count,
        params.interpolation,
        run.cell.as_ref(),
    );
    let mut ids = Vec::with_capacity(count);
    ids.push(run.ids[lo]);
    for _ in 1..count.saturating_sub(1) {
        ids.push(run.next_id);
        run.next_id += 1;
    }
    if count > 1 {
        ids.push(run.ids[hi]);
    }
    let mut band = run.template.clone();
    if let Some((k_min, k_max)) = run.ew {
        band.spring = SpringKind::EnergyWeighted { k_min, k_max };
    }
    band.autoneb = None;
    band.solid = None;
    let end_energies = [energies[lo], energies[hi]];
    let free = params.ends == ZoomEnds::FreePerp;
    let mut session = if free {
        BandSession::new_free_ends(band, packed)?
    } else {
        let mut session = BandSession::new(band, packed)?;
        session.seed_endpoint_energies(end_energies)?;
        session
    };
    session.set_force_gate(ForceGate::LinfNorm);
    run.session = session;
    run.ids = ids;
    run.free_ends = free;
    run.zoom_due = false;
    run.zoomed = true;
    run.window = Some((lo, hi));
    run.post_steps = 0;
    run.iteration += 1;
    Ok(StencilReport {
        phase: Phase::Zoom,
        status: BandStatus::Running,
        max_force: 0.0,
        ci_index: None,
        iteration: run.iteration,
        surface_rows: 0,
        total_rows: run.total_rows,
        zoom_window: Some((lo, hi)),
        curvature: None,
        force_gate: "max-abs-component",
    })
}

fn step_auto<S: StencilSurface + ?Sized>(
    run: &mut AutoRun,
    surface: &S,
) -> Result<StencilReport, SaddleError> {
    let wrapped = AutoSurface { surface };
    let before_rows = run.search.surface_rows();
    let before = run.search.insertions().len();
    let report = run.search.step(&wrapped)?;
    let after = run.search.insertions().len();
    if after > before {
        for &index in &run.search.insertions()[run.seen_insertions..] {
            if index < run.ids.len() {
                run.ids.insert(index + 1, run.next_id);
                run.next_id += 1;
            }
        }
        run.seen_insertions = after;
    }
    let delta = report.surface_rows.saturating_sub(before_rows);
    let phase = if run.search.took_insertion() {
        Phase::Insert
    } else {
        Phase::Band
    };
    run.total_rows = report.surface_rows;
    Ok(StencilReport {
        phase,
        status: report.status,
        max_force: report.max_force,
        ci_index: report.ci_index,
        iteration: report.iteration,
        surface_rows: delta,
        total_rows: run.total_rows,
        zoom_window: None,
        curvature: None,
        force_gate: "max-abs-component",
    })
}

struct AutoSurface<'a, S: ?Sized> {
    surface: &'a S,
}

impl<S: StencilSurface + ?Sized> BandSurface for AutoSurface<'_, S> {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        // AutoNEB's own pool splits the block. The ids here are the
        // positions in that block, not the stable stencil ids. The
        // stencil assigns stable ids from the insertion list.
        let ids = vec![0; positions.nrows()];
        self.surface.eval(&ids, positions, energies, gradients)
    }
}
