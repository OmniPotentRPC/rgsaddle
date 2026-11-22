//! One builder for a point stencil.
//!
//! Dimer and Lanczos construct a [`MinModeSession`],
//! so a preset with the same settings repeats that session's positions.
//! Band presets stay on `feat/band-methods-zoom`.
//! Goswami, Gunde, and Jónsson, Front. Chem. (2026),
//! doi:10.3389/fchem.2026.1807063, Eq. 9.
//! Henkelman and Jónsson, J. Chem. Phys. 111, 7010 (1999),
//! doi:10.1063/1.480097.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, Array2, ArrayView1};
use rgmin::{Accept, Control, FireKind, Method, Oracle, Solver};

use crate::error::SaddleError;
use crate::force::ForceGate;
use crate::minmode::{
    ArtNConfig, ClimbCap, FiniteDifference, MinModeConfig, MinModeKind, MinModeSession,
    MinModeStatus, climb_displacement, translated_gradient,
};
/// Error text [`BandBuilder::build`] returns until `feat/band-methods-zoom` lands.
pub const BAND_PRESET_SEAM: &str = "band presets stay on feat/band-methods-zoom";

/// Closed preset list. A band value is accepted by the builder and
/// rejected by [`BandBuilder::build`] and by [`PointBuilder::build`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Preset {
    /// The Min.
    Min,
    /// The Dimer.
    Dimer,
    /// The Lanczos.
    Lanczos,
    /// The Art.
    Art,
    /// The EigenFollow.
    EigenFollow,
    /// The Neb.
    Neb,
    /// The Dneb.
    Dneb,
    /// The CiNeb.
    CiNeb,
    /// The EnergyWeightedNeb.
    EnergyWeightedNeb,
    /// The ZoomNeb.
    ZoomNeb,
    /// The AutoNeb.
    AutoNeb,
    /// The OciNeb.
    OciNeb,
    /// The Idpp.
    Idpp,
    /// The SIdpp.
    SIdpp,
}

impl Preset {
    /// Name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Min => "min",
            Self::Dimer => "dimer",
            Self::Lanczos => "lanczos",
            Self::Art => "art",
            Self::EigenFollow => "eigen-follow",
            Self::Neb => "neb",
            Self::Dneb => "dneb",
            Self::CiNeb => "ci-neb",
            Self::EnergyWeightedNeb => "energy-weighted-neb",
            Self::ZoomNeb => "zoom-neb",
            Self::AutoNeb => "autoneb",
            Self::OciNeb => "oci-neb",
            Self::Idpp => "idpp",
            Self::SIdpp => "s-idpp",
        }
    }

    /// From name.
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "min" => Self::Min,
            "dimer" => Self::Dimer,
            "lanczos" => Self::Lanczos,
            "art" => Self::Art,
            "eigen-follow" => Self::EigenFollow,
            "neb" => Self::Neb,
            "dneb" => Self::Dneb,
            "ci-neb" => Self::CiNeb,
            "energy-weighted-neb" => Self::EnergyWeightedNeb,
            "zoom-neb" => Self::ZoomNeb,
            "autoneb" => Self::AutoNeb,
            "oci-neb" => Self::OciNeb,
            "idpp" => Self::Idpp,
            "s-idpp" => Self::SIdpp,
            _ => return None,
        })
    }

    /// To abi.
    pub fn to_abi(self) -> i32 {
        match self {
            Self::Min => 0,
            Self::Dimer => 1,
            Self::Lanczos => 2,
            Self::Art => 3,
            Self::EigenFollow => 4,
            Self::Neb => 5,
            Self::Dneb => 6,
            Self::CiNeb => 7,
            Self::EnergyWeightedNeb => 8,
            Self::ZoomNeb => 9,
            Self::AutoNeb => 10,
            Self::OciNeb => 11,
            Self::Idpp => 12,
            Self::SIdpp => 13,
        }
    }

    /// From abi.
    pub fn from_abi(value: i32) -> Option<Self> {
        Some(match value {
            0 => Self::Min,
            1 => Self::Dimer,
            2 => Self::Lanczos,
            3 => Self::Art,
            4 => Self::EigenFollow,
            5 => Self::Neb,
            6 => Self::Dneb,
            7 => Self::CiNeb,
            8 => Self::EnergyWeightedNeb,
            9 => Self::ZoomNeb,
            10 => Self::AutoNeb,
            11 => Self::OciNeb,
            12 => Self::Idpp,
            13 => Self::SIdpp,
            _ => return None,
        })
    }

    /// Is point.
    pub fn is_point(self) -> bool {
        matches!(
            self,
            Self::Min | Self::Dimer | Self::Lanczos | Self::Art | Self::EigenFollow
        )
    }
}

/// What happens to the true force along the tangent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Climb {
    /// The Descend.
    Descend,
    /// The Reflect.
    Reflect,
    /// The Ascend.
    Ascend,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// The StencilStatus.
pub enum StencilStatus {
    /// The Running.
    Running,
    /// The Converged.
    Converged,
}

#[derive(Clone, Copy, Debug)]
/// The StencilReport.
pub struct StencilReport {
    /// Whether this step is still running or has converged.
    pub status: StencilStatus,
    /// Largest absolute force component, in the host's force units.
    pub max_force: f64,
    /// Curvature along the mode, energy per length squared.
    pub curvature: f64,
    /// Number of steps taken.
    pub iteration: usize,
    /// Surface evaluations performed by this call.
    pub evaluations: usize,
}

enum Engine {
    Mode(Box<MinModeSession>),
    Eigen(Box<EigenEngine>),
    Min(Box<MinEngine>),
}

struct MinEngine {
    x: Array1<f64>,
    force_tol: f64,
    solver: Solver,
    iteration: usize,
    max_force: f64,
}

struct EigenEngine {
    x: Array1<f64>,
    mode: Array1<f64>,
    curvature: f64,
    dr: f64,
    force_tol: f64,
    max_move: f64,
    solver: Solver,
    iteration: usize,
    max_force: f64,
}

/// Point search. [`Stencil::point`] fills the defaults.
pub struct PointBuilder {
    x: Array1<f64>,
    preset: Preset,
    mode: Option<Array1<f64>>,
    dr: f64,
    rotation_tol: f64,
    rotation_angle_tol: Option<f64>,
    max_rotations: usize,
    krylov_dim: usize,
    force_tol: f64,
    max_move: f64,
    threads: usize,
    art_threshold: f64,
    art_push_steps: usize,
    artn: Option<ArtNConfig>,
    climb: Option<Climb>,
    window: bool,
    mode_solver: bool,
}

/// Band search. [`BandBuilder::build`] is the seam with the band branch.
pub struct BandBuilder {
    rows: usize,
    dof: usize,
    finite: bool,
    threads: usize,
}

/// A point stencil. Band construction returns [`BAND_PRESET_SEAM`].
///
/// # Examples
///
/// A dimer search on the double well `(x² − 1)² + 2y² + 2z²`, set up with
/// the builder from [`Stencil::point`].
///
/// ```
/// use ndarray::{Array1, ArrayView1, array};
/// use rgsaddle::{PointSurface, Preset, SaddleError, Stencil, StencilStatus};
///
/// struct DoubleWell;
///
/// impl PointSurface for DoubleWell {
///     fn eval(&self, r: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
///         let (x, y, z) = (r[0], r[1], r[2]);
///         let energy = (x * x - 1.0).powi(2) + 2.0 * y * y + 2.0 * z * z;
///         Ok((energy, array![4.0 * x * (x * x - 1.0), 4.0 * y, 4.0 * z]))
///     }
/// }
///
/// let mut stencil = Stencil::point(array![0.3, 0.1, 0.0])
///     .preset(Preset::Dimer)
///     .mode(array![1.0, 0.5, 0.0])
///     .force_tol(1e-4)
///     .max_move(0.1)
///     .build()?;
/// let report = stencil.run(&DoubleWell, 200)?;
///
/// assert_eq!(report.status, StencilStatus::Converged);
/// assert!(stencil.curvature() < 0.0);
/// assert!(stencil.position().iter().all(|c| c.abs() < 1e-3));
/// # Ok::<(), SaddleError>(())
/// ```
pub struct Stencil {
    engine: Engine,
    threads: usize,
}

impl Stencil {
    /// Start a point search at `x`, Cartesian length 3N. The defaults
    /// are a dimer along the first coordinate axis, `force_tol` 1e-3 and
    /// `max_move` 0.2; the [`PointBuilder`] setters change them.
    pub fn point(x: Array1<f64>) -> PointBuilder {
        PointBuilder {
            x,
            preset: Preset::Dimer,
            mode: None,
            dr: 1e-3,
            rotation_tol: 1e-4,
            rotation_angle_tol: None,
            max_rotations: 20,
            krylov_dim: 12,
            force_tol: 1e-3,
            max_move: 0.2,
            threads: 1,
            art_threshold: ArtNConfig::default().eigenvalue_threshold,
            art_push_steps: ArtNConfig::default().lanczos_start,
            artn: None,
            climb: None,
            window: false,
            mode_solver: true,
        }
    }

    /// Start a band search over the rows of `x`, one image per row.
    /// [`BandBuilder::build`] returns [`BAND_PRESET_SEAM`] for now.
    pub fn band(x: Array2<f64>) -> BandBuilder {
        BandBuilder {
            rows: x.nrows(),
            dof: x.ncols(),
            finite: x.iter().all(|v| v.is_finite()),
            threads: 1,
        }
    }

    /// How many images may be evaluated at once.
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Cartesian coordinates, length 3N, in the host's length unit.
    pub fn position(&self) -> ArrayView1<'_, f64> {
        match &self.engine {
            Engine::Mode(session) => session.position(),
            Engine::Eigen(engine) => engine.x.view(),
            Engine::Min(engine) => engine.x.view(),
        }
    }

    /// Estimated lowest-curvature direction. A stored mode is normalized.
    pub fn mode(&self) -> Option<ArrayView1<'_, f64>> {
        match &self.engine {
            Engine::Mode(session) => Some(session.mode()),
            Engine::Eigen(engine) => Some(engine.mode.view()),
            Engine::Min(_) => None,
        }
    }

    /// Curvature along the mode, energy per length squared.
    pub fn curvature(&self) -> f64 {
        match &self.engine {
            Engine::Mode(session) => session.curvature(),
            Engine::Eigen(engine) => engine.curvature,
            Engine::Min(_) => f64::NAN,
        }
    }

    /// Take one step of the preset on `surface`.
    pub fn step<S: crate::minmode::PointSurface>(
        &mut self,
        surface: &S,
    ) -> Result<StencilReport, SaddleError> {
        match &mut self.engine {
            Engine::Mode(session) => Ok(from_min(session.step(surface)?)),
            Engine::Eigen(engine) => eigen_step(engine, surface),
            Engine::Min(engine) => min_step(engine, surface),
        }
    }

    /// Loop [`Self::step`] until convergence or `max_steps`.
    pub fn run<S: crate::minmode::PointSurface>(
        &mut self,
        surface: &S,
        max_steps: usize,
    ) -> Result<StencilReport, SaddleError> {
        if let Engine::Mode(session) = &mut self.engine {
            return Ok(from_min(session.run(surface, max_steps)?));
        }
        let mut report = self.step(surface)?;
        while report.status == StencilStatus::Running && report.iteration < max_steps {
            report = self.step(surface)?;
        }
        Ok(report)
    }
}

impl PointBuilder {
    /// Search method. [`Preset::Dimer`] by default. A band preset is
    /// rejected by [`Self::build`].
    pub fn preset(mut self, preset: Preset) -> Self {
        self.preset = preset;
        self
    }

    /// Estimated lowest-curvature direction. A stored mode is normalized.
    pub fn mode(mut self, mode: Array1<f64>) -> Self {
        self.mode = Some(mode);
        self
    }

    /// Finite-difference displacement, in the host's length unit.
    pub fn dr(mut self, dr: f64) -> Self {
        self.dr = dr;
        self
    }

    /// The rotation stops when the rotational force falls under this value.
    pub fn rotation_tol(mut self, rotation_tol: f64) -> Self {
        self.rotation_tol = rotation_tol;
        self
    }

    /// The dimer also stops when one rotation's angle falls under this many radians. Zero disables that test.
    /// Unset, a dimer keeps the default five-degree stop at every `rotation_tol`.
    pub fn rotation_angle_tol(mut self, rotation_angle_tol: f64) -> Self {
        self.rotation_angle_tol = Some(rotation_angle_tol);
        self
    }

    /// Most dimer rotations, or Lanczos actions, in one estimate.
    pub fn max_rotations(mut self, max_rotations: usize) -> Self {
        self.max_rotations = max_rotations;
        self
    }

    /// Largest Krylov subspace used for one Lanczos estimate.
    pub fn krylov_dim(mut self, krylov_dim: usize) -> Self {
        self.krylov_dim = krylov_dim;
        self
    }

    /// The search stops when the largest absolute force component falls under this value, in the host's force units.
    pub fn force_tol(mut self, force_tol: f64) -> Self {
        self.force_tol = force_tol;
        self
    }

    /// Largest per-atom displacement, in the host's length unit.
    pub fn max_move(mut self, max_move: f64) -> Self {
        self.max_move = max_move;
        self
    }

    /// How many images may be evaluated at once.
    pub fn threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }

    /// Art threshold.
    pub fn art_threshold(mut self, art_threshold: f64) -> Self {
        self.art_threshold = art_threshold;
        self
    }

    /// Art push steps.
    pub fn art_push_steps(mut self, art_push_steps: usize) -> Self {
        self.art_push_steps = art_push_steps;
        self
    }

    /// Enable the ART nouveau protocol with these controls.
    ///
    /// ART nouveau uses the Lanczos minimum-mode estimator, regardless
    /// of the otherwise selected point preset.
    pub fn artn(mut self, artn: ArtNConfig) -> Self {
        self.artn = Some(artn);
        self
    }

    /// Climb.
    pub fn climb(mut self, climb: Climb) -> Self {
        self.climb = Some(climb);
        self
    }

    /// Mark a window mesh. A point search rejects it in [`Self::build`].
    pub fn window(mut self) -> Self {
        self.window = true;
        self
    }

    /// ART climbs with Lanczos after the push. `false` is rejected.
    pub fn mode_solver(mut self, enabled: bool) -> Self {
        self.mode_solver = enabled;
        self
    }

    /// Check the settings and start the search.
    pub fn build(self) -> Result<Stencil, SaddleError> {
        validate_point(
            self.preset,
            self.climb,
            self.threads,
            self.window,
            self.mode_solver,
        )?;
        if self.x.is_empty() {
            return Err(SaddleError::Shape(
                "a point stencil needs a position".into(),
            ));
        }
        if self.x.iter().any(|v| !v.is_finite()) {
            return Err(SaddleError::NonFinite("position"));
        }
        if !(self.dr.is_finite() && self.dr > 0.0) {
            return Err(SaddleError::Invalid(
                "finite-difference step must be positive and finite".into(),
            ));
        }
        if !self.art_threshold.is_finite() {
            return Err(SaddleError::Invalid(
                "ART curvature threshold must be finite".into(),
            ));
        }
        let mode = match &self.mode {
            Some(mode) => unit_direction(mode.clone())?,
            None => basis(self.x.len()),
        };
        if mode.len() != self.x.len() {
            return Err(SaddleError::Shape("mode must match the position".into()));
        }
        let threads = self.threads;
        let dof = self.x.len();
        let engine = match self.preset {
            Preset::Dimer | Preset::Lanczos | Preset::Art => {
                Engine::Mode(Box::new(mode_session(&self, mode)?))
            }
            Preset::EigenFollow => Engine::Eigen(Box::new(EigenEngine {
                x: self.x,
                mode,
                curvature: f64::NAN,
                dr: self.dr,
                force_tol: self.force_tol,
                max_move: self.max_move,
                solver: force_solver(Method::lbfgs_with_memory(20), dof, self.max_move)?,
                iteration: 0,
                max_force: f64::NAN,
            })),
            Preset::Min => Engine::Min(Box::new(MinEngine {
                x: self.x,
                force_tol: self.force_tol,
                solver: force_solver(Method::fire(FireKind::V2), dof, self.max_move)?,
                iteration: 0,
                max_force: f64::NAN,
            })),
            _ => {
                return Err(SaddleError::Invalid(BAND_PRESET_SEAM.into()));
            }
        };
        Ok(Stencil { engine, threads })
    }
}

impl BandBuilder {
    /// Band method. Not read until band presets land.
    pub fn preset(self, _preset: Preset) -> Self {
        self
    }

    /// How many images may be evaluated at once.
    pub fn threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }

    /// Check the shape, the positions and the thread count, then return
    /// [`BAND_PRESET_SEAM`] as [`SaddleError::Invalid`].
    pub fn build(self) -> Result<Stencil, SaddleError> {
        if self.rows == 0 || self.dof == 0 {
            return Err(SaddleError::Shape(
                "a band needs images and degrees of freedom".into(),
            ));
        }
        if !self.finite {
            return Err(SaddleError::NonFinite("band positions"));
        }
        if self.threads == 0 {
            return Err(SaddleError::Invalid(
                "chain evaluation needs at least one thread".into(),
            ));
        }
        Err(SaddleError::Invalid(BAND_PRESET_SEAM.into()))
    }
}

fn validate_point(
    preset: Preset,
    climb: Option<Climb>,
    threads: usize,
    window: bool,
    mode_solver: bool,
) -> Result<(), SaddleError> {
    if threads == 0 {
        return Err(SaddleError::Invalid(
            "chain evaluation needs at least one thread".into(),
        ));
    }
    if window {
        return Err(SaddleError::Invalid("a window is a band mesh".into()));
    }
    if !preset.is_point() {
        return Err(SaddleError::Invalid(BAND_PRESET_SEAM.into()));
    }
    if preset == Preset::Min && climb == Some(Climb::Reflect) {
        return Err(SaddleError::Invalid(
            "Climb::Reflect needs a tangent".into(),
        ));
    }
    if matches!(
        preset,
        Preset::Dimer | Preset::Lanczos | Preset::Art | Preset::EigenFollow
    ) && climb == Some(Climb::Descend)
    {
        return Err(SaddleError::Invalid(
            "this preset reflects or ascends along its tangent".into(),
        ));
    }
    if preset == Preset::Art && !mode_solver {
        return Err(SaddleError::Invalid("ART needs a mode solver".into()));
    }
    Ok(())
}

fn mode_session(builder: &PointBuilder, mode: Array1<f64>) -> Result<MinModeSession, SaddleError> {
    let kind = if matches!(builder.preset, Preset::Lanczos | Preset::Art) || builder.artn.is_some()
    {
        MinModeKind::Lanczos
    } else {
        MinModeKind::Dimer
    };
    let mut config = MinModeConfig::default()
        .with_kind(kind)
        .with_dr(builder.dr)
        .with_difference(FiniteDifference::Forward)
        .with_rotation_tol(builder.rotation_tol)
        .with_max_rotations(builder.max_rotations)
        .with_krylov_dim(builder.krylov_dim)
        .with_force_tol(builder.force_tol)
        .with_max_move(builder.max_move)
        .with_method(Method::fire(FireKind::V2))
        .with_seed_curvature(false)
        .with_saddle_check(None);
    if let Some(rotation_angle_tol) = builder.rotation_angle_tol {
        config = config.with_rotation_angle_tol(rotation_angle_tol);
    } else if kind == MinModeKind::Dimer {
        config = config.with_rotation_angle_tol(MinModeConfig::default().rotation_angle_tol);
    }
    if builder.preset == Preset::Art || builder.artn.is_some() {
        let artn = builder.artn.clone().unwrap_or_else(|| {
            ArtNConfig::default()
                .with_eigenvalue_threshold(builder.art_threshold)
                .with_lanczos_start(builder.art_push_steps)
        });
        config = config.with_artn(artn);
    }
    MinModeSession::new(config, builder.x.clone(), mode)
}

fn force_solver(method: Method, n: usize, max_move: f64) -> Result<Solver, SaddleError> {
    crate::band::check_force_driven(&method)?;
    let control = Control::default()
        .with_maxiter(usize::MAX)
        .with_gtol(0.0)
        .with_istep(1.0)
        .with_maxmove(None);
    let lbfgs = matches!(method, Method::Lbfgs { .. });
    let mut solver = Solver::new(method, control, n);
    solver.set_atom_maxmove(max_move);
    if lbfgs {
        solver.set_accept(Accept::Step);
    }
    Ok(solver)
}

fn from_min(report: crate::minmode::MinModeReport) -> StencilReport {
    StencilReport {
        status: match report.status {
            MinModeStatus::Converged => StencilStatus::Converged,
            MinModeStatus::Running => StencilStatus::Running,
        },
        max_force: report.max_force,
        curvature: report.curvature,
        iteration: report.iteration,
        evaluations: report.evaluations,
    }
}

fn basis(n: usize) -> Array1<f64> {
    let mut mode = Array1::zeros(n);
    if n > 0 {
        mode[0] = 1.0;
    }
    mode
}

fn unit_direction(mode: Array1<f64>) -> Result<Array1<f64>, SaddleError> {
    if mode.iter().any(|v| !v.is_finite()) {
        return Err(SaddleError::NonFinite("mode"));
    }
    let scale = mode.dot(&mode).sqrt();
    if !(scale.is_finite() && scale > 1e-14) {
        return Err(SaddleError::Invalid("mode must be nonzero".into()));
    }
    Ok(mode.mapv(|v| v / scale))
}

fn eigen_step<S: crate::minmode::PointSurface>(
    engine: &mut EigenEngine,
    surface: &S,
) -> Result<StencilReport, SaddleError> {
    let (_energy, gradient) = surface.eval(engine.x.view())?;
    let hessian = dense_hessian(surface, &engine.x, engine.dr)?;
    let (values, vectors) = crate::eigensolve::exact_eigh(hessian.view())?;
    let mut mode = vectors.column(0).to_owned();
    largest_component_positive(&mut mode);
    let curvature = values[0];
    let previous = engine.mode.clone();
    engine.mode = mode;
    engine.curvature = curvature;
    let max_force = ForceGate::LinfNorm.value(gradient.view());
    engine.max_force = max_force;
    let calls = AtomicUsize::new(1 + 2 * engine.x.len());
    if max_force <= engine.force_tol && curvature < 0.0 {
        return Ok(StencilReport {
            status: StencilStatus::Converged,
            max_force,
            curvature,
            iteration: engine.iteration,
            evaluations: calls.load(Ordering::Relaxed),
        });
    }
    let turned = previous.dot(&engine.mode).abs().min(1.0).acos();
    if turned > crate::minmode::MODE_RESET_ANGLE {
        engine.solver.forget();
    } else {
        engine.solver.forget_evaluation();
    }
    let tau = engine.mode.clone();
    if curvature >= 0.0 && engine.max_move.is_finite() && engine.max_move > 0.0 {
        let climbed = climb_displacement(
            engine.x.view(),
            gradient.view(),
            tau.view(),
            curvature,
            engine.max_move,
            ClimbCap::Atom,
        );
        let (_energy, _gradient) = surface.eval(climbed.view())?;
        calls.fetch_add(1, Ordering::Relaxed);
        engine.solver.forget_evaluation();
        engine.x = climbed;
        engine.iteration += 1;
        return Ok(StencilReport {
            status: StencilStatus::Running,
            max_force,
            curvature,
            iteration: engine.iteration,
            evaluations: calls.load(Ordering::Relaxed),
        });
    }
    let start = engine.x.clone();
    let slot: Mutex<Option<SaddleError>> = Mutex::new(None);
    let oracle = Oracle::unbounded(engine.x.len(), |xv: ArrayView1<f64>| {
        let answer = if xv == start.view() {
            Ok((0.0, gradient.clone()))
        } else {
            calls.fetch_add(1, Ordering::Relaxed);
            surface.eval(xv)
        };
        match answer {
            Ok((energy, g)) => {
                let eff = translated_gradient(&g, &tau, curvature);
                (energy, eff)
            }
            Err(err) => {
                remember_error(&slot, err);
                (f64::INFINITY, Array1::zeros(xv.len()))
            }
        }
    });
    let mut x = engine.x.clone();
    let stepped = engine.solver.step(&oracle, &mut x);
    if let Some(err) = take_error(&slot) {
        return Err(err);
    }
    stepped.map_err(|err| SaddleError::Solver(err.to_string()))?;
    engine.x = x;
    engine.iteration += 1;
    Ok(StencilReport {
        status: StencilStatus::Running,
        max_force,
        curvature,
        iteration: engine.iteration,
        evaluations: calls.load(Ordering::Relaxed),
    })
}

fn min_step<S: crate::minmode::PointSurface>(
    engine: &mut MinEngine,
    surface: &S,
) -> Result<StencilReport, SaddleError> {
    let (energy, gradient) = surface.eval(engine.x.view())?;
    let max_force = ForceGate::LinfNorm.value(gradient.view());
    engine.max_force = max_force;
    if max_force <= engine.force_tol {
        return Ok(StencilReport {
            status: StencilStatus::Converged,
            max_force,
            curvature: f64::NAN,
            iteration: engine.iteration,
            evaluations: 1,
        });
    }
    let start = engine.x.clone();
    let slot: Mutex<Option<SaddleError>> = Mutex::new(None);
    let oracle = Oracle::unbounded(engine.x.len(), |xv: ArrayView1<f64>| {
        if xv == start.view() {
            return (energy, gradient.clone());
        }
        match surface.eval(xv) {
            Ok(pair) => pair,
            Err(err) => {
                remember_error(&slot, err);
                (f64::INFINITY, Array1::zeros(xv.len()))
            }
        }
    });
    let mut x = engine.x.clone();
    let stepped = engine.solver.step(&oracle, &mut x);
    if let Some(err) = take_error(&slot) {
        return Err(err);
    }
    stepped.map_err(|err| SaddleError::Solver(err.to_string()))?;
    engine.x = x;
    engine.iteration += 1;
    Ok(StencilReport {
        status: StencilStatus::Running,
        max_force,
        curvature: f64::NAN,
        iteration: engine.iteration,
        evaluations: 1,
    })
}

fn remember_error(slot: &Mutex<Option<SaddleError>>, err: SaddleError) {
    let mut guard = match slot.lock() {
        Ok(guard) => guard,
        Err(poison) => poison.into_inner(),
    };
    if guard.is_none() {
        *guard = Some(err);
    }
}

fn take_error(slot: &Mutex<Option<SaddleError>>) -> Option<SaddleError> {
    match slot.lock() {
        Ok(mut guard) => guard.take(),
        Err(poison) => poison.into_inner().take(),
    }
}

fn dense_hessian<S: crate::minmode::PointSurface>(
    surface: &S,
    x: &Array1<f64>,
    dr: f64,
) -> Result<Array2<f64>, SaddleError> {
    let n = x.len();
    let mut hessian = Array2::zeros((n, n));
    for column in 0..n {
        let mut plus = x.clone();
        let mut minus = x.clone();
        plus[column] += dr;
        minus[column] -= dr;
        let (_ep, gp) = surface.eval(plus.view())?;
        let (_em, gm) = surface.eval(minus.view())?;
        if gp.len() != n || gm.len() != n {
            return Err(SaddleError::Shape(
                "hessian gradient length does not match the position".into(),
            ));
        }
        for row in 0..n {
            hessian[(row, column)] = (gp[row] - gm[row]) / (2.0 * dr);
        }
    }
    for row in 0..n {
        for column in 0..row {
            let mean = 0.5 * (hessian[(row, column)] + hessian[(column, row)]);
            hessian[(row, column)] = mean;
            hessian[(column, row)] = mean;
        }
    }
    Ok(hessian)
}

fn largest_component_positive(mode: &mut Array1<f64>) {
    let mut best = 0;
    for index in 1..mode.len() {
        if mode[index].abs() > mode[best].abs() {
            best = index;
        }
    }
    if mode[best] < 0.0 {
        mode.mapv_inplace(|value| -value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::minmode::PointSurface;
    use ndarray::ArrayView1;

    struct Quadratic {
        cxx: f64,
        cyy: f64,
    }

    impl PointSurface for Quadratic {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let energy = 0.5 * self.cxx * x[0] * x[0] + 0.5 * self.cyy * x[1] * x[1];
            Ok((energy, array_grad(self.cxx * x[0], self.cyy * x[1])))
        }
    }

    fn array_grad(x: f64, y: f64) -> Array1<f64> {
        Array1::from_vec(vec![x, y])
    }

    fn bowl() -> Quadratic {
        Quadratic {
            cxx: 4.0,
            cyy: 10.0,
        }
    }

    fn saddle() -> Quadratic {
        Quadratic {
            cxx: -4.0,
            cyy: 10.0,
        }
    }

    #[test]
    fn positive_curvature_moves_only_along_the_mode() {
        for preset in [Preset::Dimer, Preset::Lanczos] {
            let mut search = Stencil::point(array_grad(0.4, 0.3))
                .preset(preset)
                .mode(array_grad(1.0, 0.0))
                .max_rotations(4)
                .krylov_dim(4)
                .build()
                .unwrap();
            search.step(&bowl()).unwrap();
            assert_eq!(
                search.position()[1].to_le_bytes(),
                (0.3_f64).to_le_bytes(),
                "{preset:?}"
            );
            assert_ne!(search.position()[0].to_le_bytes(), (0.4_f64).to_le_bytes());
        }
    }

    #[test]
    fn negative_curvature_keeps_the_perpendicular_force() {
        let mut search = Stencil::point(array_grad(0.2, 0.3))
            .preset(Preset::Dimer)
            .mode(array_grad(1.0, 0.0))
            .max_rotations(4)
            .build()
            .unwrap();
        search.step(&saddle()).unwrap();
        assert_ne!(search.position()[1].to_le_bytes(), (0.3_f64).to_le_bytes());
        assert!(search.curvature() < 0.0);
    }

    #[test]
    fn lanczos_presets_reject_invalid_rotation_angle_tolerances() {
        for preset in [Preset::Lanczos, Preset::Art] {
            for rotation_angle_tol in [-1.0, f64::NAN] {
                let err = Stencil::point(array_grad(0.0, 0.0))
                    .preset(preset)
                    .rotation_angle_tol(rotation_angle_tol)
                    .build()
                    .err()
                    .expect("build rejects the invalid rotation angle tolerance");
                assert!(
                    err.to_string()
                        .contains("rotation tolerances must be finite and non-negative"),
                    "{preset:?}: {err}"
                );
            }
        }
    }

    #[test]
    fn art_push_relaxes_only_perpendicular_to_the_seed() {
        let mut search = Stencil::point(array_grad(0.4, 0.3))
            .preset(Preset::Art)
            .mode(array_grad(1.0, 0.0))
            .art_threshold(0.0)
            .art_push_steps(5)
            .build()
            .unwrap();
        search.step(&bowl()).unwrap();
        assert!(search.position()[0] > 0.4);
        assert!(search.position()[1].abs() < 0.3);
        assert!(search.curvature().is_nan());

        let mut opened = Stencil::point(array_grad(0.2, 0.3))
            .mode(array_grad(1.0, 0.0))
            .artn(ArtNConfig::default().with_lanczos_start(0))
            .max_rotations(4)
            .krylov_dim(4)
            .build()
            .unwrap();
        opened.step(&saddle()).unwrap();
        assert_ne!(opened.position()[1].to_le_bytes(), (0.3_f64).to_le_bytes());
        assert!(opened.curvature() < 0.0);
    }

    #[test]
    fn eigen_follow_uses_a_positive_largest_component() {
        let mut search = Stencil::point(array_grad(0.2, 0.3))
            .preset(Preset::EigenFollow)
            .build()
            .unwrap();
        search.step(&saddle()).unwrap();
        let mode = search.mode().unwrap();
        let mut best = 0;
        if mode[1].abs() > mode[0].abs() {
            best = 1;
        }
        assert!(mode[best] > 0.0);
        assert!(search.curvature() < 0.0);
    }

    #[test]
    fn a_minimiser_walks_down_the_bowl() {
        let mut search = Stencil::point(array_grad(0.5, -0.2))
            .preset(Preset::Min)
            .force_tol(1e-12)
            .max_move(0.05)
            .build()
            .unwrap();
        let start = search.position()[0];
        search.step(&bowl()).unwrap();
        assert!(search.position()[0].abs() < start.abs());
    }

    #[test]
    fn band_presets_and_incoherent_rows_are_rejected() {
        let err = Stencil::band(Array2::zeros((3, 2)))
            .preset(Preset::CiNeb)
            .build()
            .err()
            .expect("build rejects this row");
        assert!(err.to_string().contains(BAND_PRESET_SEAM));

        let err = Stencil::point(array_grad(0.0, 0.0))
            .preset(Preset::Neb)
            .build()
            .err()
            .expect("build rejects this row");
        assert!(err.to_string().contains(BAND_PRESET_SEAM));

        let err = Stencil::point(array_grad(0.0, 0.0))
            .preset(Preset::Min)
            .climb(Climb::Reflect)
            .build()
            .err()
            .expect("build rejects this row");
        assert!(err.to_string().contains("Climb::Reflect"));

        let err = Stencil::point(array_grad(0.0, 0.0))
            .window()
            .build()
            .err()
            .expect("build rejects this row");
        assert!(err.to_string().contains("window"));

        let err = Stencil::point(array_grad(0.0, 0.0))
            .preset(Preset::Art)
            .mode_solver(false)
            .build()
            .err()
            .expect("build rejects this row");
        assert!(err.to_string().contains("mode solver"));

        let err = Stencil::point(array_grad(0.0, 0.0))
            .threads(0)
            .build()
            .err()
            .expect("build rejects this row");
        assert!(err.to_string().contains("at least one thread"));
    }

    #[test]
    fn preset_names_and_abi_values_agree() {
        for value in 0..14 {
            let preset = Preset::from_abi(value).unwrap();
            assert_eq!(preset.to_abi(), value);
            assert_eq!(Preset::from_name(preset.name()), Some(preset));
        }
        assert!(Preset::from_abi(14).is_none());
        assert!(Preset::from_name("zoom").is_none());
    }
}
