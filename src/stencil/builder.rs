//! One builder for a point stencil.
//!
//! Dimer and Lanczos construct a [`MinModeSession`](crate::minmode::MinModeSession),
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
    FiniteDifference, MinModeConfig, MinModeKind, MinModeSession, MinModeStatus,
    convex_displacement, translated_gradient,
};
use crate::stencil::{ascend, curvature_along};

use super::control::{Action, ArtControl, Controller, Report as ControlReport, Role, StencilView};

/// Band `Stencil::build` returns this until `feat/band-methods-zoom` lands.
pub const BAND_PRESET_SEAM: &str = "band presets stay on feat/band-methods-zoom";

/// Closed preset list. A band value is accepted by the builder and
/// rejected by [`BandBuilder::build`] and by [`PointBuilder::build`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Preset {
    Min,
    Dimer,
    Lanczos,
    Art,
    EigenFollow,
    Neb,
    Dneb,
    CiNeb,
    EnergyWeightedNeb,
    ZoomNeb,
    AutoNeb,
    OciNeb,
    Idpp,
    SIdpp,
}

impl Preset {
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
    Descend,
    Reflect,
    Ascend,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StencilStatus {
    Running,
    Converged,
}

#[derive(Clone, Copy, Debug)]
pub struct StencilReport {
    pub status: StencilStatus,
    pub max_force: f64,
    pub curvature: f64,
    pub iteration: usize,
    pub evaluations: usize,
}

enum Engine {
    Mode(Box<MinModeSession>),
    Art(Box<ArtEngine>),
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

struct ArtEngine {
    x: Array1<f64>,
    mode: Array1<f64>,
    curvature: f64,
    dr: f64,
    force_tol: f64,
    max_move: f64,
    max_rotations: usize,
    rotation_tol: f64,
    rotation_angle_tol: f64,
    krylov_dim: usize,
    control: ArtControl,
    session: Option<MinModeSession>,
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
    rotation_angle_tol: f64,
    max_rotations: usize,
    krylov_dim: usize,
    force_tol: f64,
    max_move: f64,
    threads: usize,
    art_threshold: f64,
    art_push_steps: usize,
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
pub struct Stencil {
    engine: Engine,
    threads: usize,
}

impl Stencil {
    pub fn point(x: Array1<f64>) -> PointBuilder {
        PointBuilder {
            x,
            preset: Preset::Dimer,
            mode: None,
            dr: 1e-3,
            rotation_tol: 1e-4,
            rotation_angle_tol: crate::minmode::DEFAULT_ROTATION_ANGLE_TOL,
            max_rotations: 20,
            krylov_dim: 12,
            force_tol: 1e-3,
            max_move: 0.2,
            threads: 1,
            art_threshold: 0.0,
            art_push_steps: 20,
            climb: None,
            window: false,
            mode_solver: true,
        }
    }

    pub fn band(x: Array2<f64>) -> BandBuilder {
        BandBuilder {
            rows: x.nrows(),
            dof: x.ncols(),
            finite: x.iter().all(|v| v.is_finite()),
            threads: 1,
        }
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    pub fn position(&self) -> ArrayView1<'_, f64> {
        match &self.engine {
            Engine::Mode(session) => session.position(),
            Engine::Art(engine) => engine.x.view(),
            Engine::Eigen(engine) => engine.x.view(),
            Engine::Min(engine) => engine.x.view(),
        }
    }

    pub fn mode(&self) -> Option<ArrayView1<'_, f64>> {
        match &self.engine {
            Engine::Mode(session) => Some(session.mode()),
            Engine::Art(engine) => Some(engine.mode.view()),
            Engine::Eigen(engine) => Some(engine.mode.view()),
            Engine::Min(_) => None,
        }
    }

    pub fn curvature(&self) -> f64 {
        match &self.engine {
            Engine::Mode(session) => session.curvature(),
            Engine::Art(engine) => engine.curvature,
            Engine::Eigen(engine) => engine.curvature,
            Engine::Min(_) => f64::NAN,
        }
    }

    pub fn step<S: crate::minmode::PointSurface>(
        &mut self,
        surface: &S,
    ) -> Result<StencilReport, SaddleError> {
        match &mut self.engine {
            Engine::Mode(session) => Ok(from_min(session.step(surface)?)),
            Engine::Art(engine) => art_step(engine, surface),
            Engine::Eigen(engine) => eigen_step(engine, surface),
            Engine::Min(engine) => min_step(engine, surface),
        }
    }

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
    pub fn preset(mut self, preset: Preset) -> Self {
        self.preset = preset;
        self
    }

    pub fn mode(mut self, mode: Array1<f64>) -> Self {
        self.mode = Some(mode);
        self
    }

    pub fn dr(mut self, dr: f64) -> Self {
        self.dr = dr;
        self
    }

    pub fn rotation_tol(mut self, rotation_tol: f64) -> Self {
        self.rotation_tol = rotation_tol;
        self
    }

    pub fn rotation_angle_tol(mut self, rotation_angle_tol: f64) -> Self {
        self.rotation_angle_tol = rotation_angle_tol;
        self
    }

    pub fn max_rotations(mut self, max_rotations: usize) -> Self {
        self.max_rotations = max_rotations;
        self
    }

    pub fn krylov_dim(mut self, krylov_dim: usize) -> Self {
        self.krylov_dim = krylov_dim;
        self
    }

    pub fn force_tol(mut self, force_tol: f64) -> Self {
        self.force_tol = force_tol;
        self
    }

    pub fn max_move(mut self, max_move: f64) -> Self {
        self.max_move = max_move;
        self
    }

    pub fn threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }

    pub fn art_threshold(mut self, art_threshold: f64) -> Self {
        self.art_threshold = art_threshold;
        self
    }

    pub fn art_push_steps(mut self, art_push_steps: usize) -> Self {
        self.art_push_steps = art_push_steps;
        self
    }

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
            Preset::Dimer | Preset::Lanczos => Engine::Mode(Box::new(mode_session(&self, mode)?)),
            Preset::Art => Engine::Art(Box::new(ArtEngine {
                x: self.x,
                mode,
                curvature: f64::NAN,
                dr: self.dr,
                force_tol: self.force_tol,
                max_move: self.max_move,
                max_rotations: self.max_rotations,
                rotation_tol: self.rotation_tol,
                rotation_angle_tol: self.rotation_angle_tol,
                krylov_dim: self.krylov_dim,
                control: ArtControl::new(self.art_threshold, self.art_push_steps),
                session: None,
                iteration: 0,
                max_force: f64::NAN,
            })),
            Preset::EigenFollow => Engine::Eigen(Box::new(EigenEngine {
                x: self.x,
                mode,
                curvature: f64::NAN,
                dr: self.dr,
                force_tol: self.force_tol,
                max_move: self.max_move,
                solver: force_solver(Method::Lbfgs { memory: 20 }, dof, self.max_move)?,
                iteration: 0,
                max_force: f64::NAN,
            })),
            Preset::Min => Engine::Min(Box::new(MinEngine {
                x: self.x,
                force_tol: self.force_tol,
                solver: force_solver(Method::Fire { kind: FireKind::V2 }, dof, self.max_move)?,
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
    pub fn preset(self, _preset: Preset) -> Self {
        self
    }

    pub fn threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }

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
    let kind = if builder.preset == Preset::Lanczos {
        MinModeKind::Lanczos
    } else {
        MinModeKind::Dimer
    };
    let config = MinModeConfig {
        kind,
        dr: builder.dr,
        difference: FiniteDifference::Forward,
        rotation_tol: builder.rotation_tol,
        rotation_angle_tol: builder.rotation_angle_tol,
        max_rotations: builder.max_rotations,
        krylov_dim: builder.krylov_dim,
        force_tol: builder.force_tol,
        max_move: builder.max_move,
        method: Method::Fire { kind: FireKind::V2 },
    };
    MinModeSession::new(config, builder.x.clone(), mode)
}

fn art_mode_config(engine: &ArtEngine) -> MinModeConfig {
    MinModeConfig {
        kind: MinModeKind::Lanczos,
        dr: engine.dr,
        difference: FiniteDifference::Forward,
        rotation_tol: engine.rotation_tol,
        rotation_angle_tol: engine.rotation_angle_tol,
        max_rotations: engine.max_rotations,
        krylov_dim: engine.krylov_dim,
        force_tol: engine.force_tol,
        max_move: engine.max_move,
        method: Method::Fire { kind: FireKind::V2 },
    }
}

fn force_solver(method: Method, n: usize, max_move: f64) -> Result<Solver, SaddleError> {
    crate::band::check_force_driven(&method)?;
    let control = Control {
        maxiter: usize::MAX,
        gtol: 0.0,
        istep: 1.0,
        maxmove: None,
        ftol_rel: None,
        noise_floor: None,
    };
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

fn capped_step(x: &Array1<f64>, g_eff: &Array1<f64>, max_move: f64) -> Array1<f64> {
    let nrm = g_eff.dot(g_eff).sqrt();
    if !(nrm.is_finite() && nrm > 1e-15) {
        return x.clone();
    }
    let alpha = (max_move / nrm).min(1.0);
    x - &(g_eff * alpha)
}

fn forward_curvature<S: crate::minmode::PointSurface>(
    surface: &S,
    x: &Array1<f64>,
    gradient: &Array1<f64>,
    mode: &Array1<f64>,
    dr: f64,
) -> Result<f64, SaddleError> {
    let shifted = x + &(mode * dr);
    let (_energy, g_plus) = surface.eval(shifted.view())?;
    // Same order as the min-mode forward action: divide, then dot.
    let action = (&g_plus - gradient) / dr;
    Ok(curvature_along(action.view(), mode.view()))
}

fn art_step<S: crate::minmode::PointSurface>(
    engine: &mut ArtEngine,
    surface: &S,
) -> Result<StencilReport, SaddleError> {
    if engine.session.is_some() {
        return art_session_step(engine, surface);
    }
    let (_energy, gradient) = surface.eval(engine.x.view())?;
    let curvature = forward_curvature(surface, &engine.x, &gradient, &engine.mode, engine.dr)?;
    engine.curvature = curvature;
    let max_force = ForceGate::LinfNorm.value(gradient.view());
    engine.max_force = max_force;
    let converged = max_force <= engine.force_tol && curvature < 0.0;
    let last = ControlReport {
        status: if converged {
            super::control::StencilStatus::Converged
        } else {
            super::control::StencilStatus::Running
        },
        max_force,
        curvature,
    };
    let roles = [Role::Walker];
    let view = StencilView {
        roles: &roles,
        curvature,
        max_force,
    };
    match engine.control.decide(&view, &last) {
        Action::Stop(_) => Ok(StencilReport {
            status: StencilStatus::Converged,
            max_force,
            curvature,
            iteration: engine.iteration,
            evaluations: 2,
        }),
        Action::Push => {
            let eff = ascend(gradient.view(), engine.mode.view());
            engine.x = capped_step(&engine.x, &eff, engine.max_move);
            engine.iteration += 1;
            Ok(StencilReport {
                status: StencilStatus::Running,
                max_force,
                curvature,
                iteration: engine.iteration,
                evaluations: 2,
            })
        }
        Action::Climb | Action::Step => {
            let config = art_mode_config(engine);
            engine.session = Some(MinModeSession::new(
                config,
                engine.x.clone(),
                engine.mode.clone(),
            )?);
            let mut out = art_session_step(engine, surface)?;
            out.evaluations = out.evaluations.saturating_add(2);
            Ok(out)
        }
    }
}

fn art_session_step<S: crate::minmode::PointSurface>(
    engine: &mut ArtEngine,
    surface: &S,
) -> Result<StencilReport, SaddleError> {
    let (raw, x, mode, curvature) = {
        let session = engine
            .session
            .as_mut()
            .ok_or_else(|| SaddleError::Invalid("ART climb has no mode solver".into()))?;
        let raw = session.step(surface)?;
        (
            raw,
            session.position().to_owned(),
            session.mode().to_owned(),
            session.curvature(),
        )
    };
    engine.x = x;
    engine.mode = mode;
    engine.curvature = curvature;
    engine.max_force = raw.max_force;
    engine.iteration += 1;
    let mut report = from_min(raw);
    report.iteration = engine.iteration;
    Ok(report)
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
        let climbed = convex_displacement(
            engine.x.view(),
            gradient.view(),
            tau.view(),
            engine.max_move,
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
    fn art_push_stays_on_the_seed_until_the_well_opens() {
        let mut search = Stencil::point(array_grad(0.4, 0.3))
            .preset(Preset::Art)
            .mode(array_grad(1.0, 0.0))
            .art_threshold(0.0)
            .art_push_steps(5)
            .build()
            .unwrap();
        search.step(&bowl()).unwrap();
        assert_eq!(search.position()[1].to_le_bytes(), (0.3_f64).to_le_bytes());
        assert!(search.curvature() > 0.0);

        let mut opened = Stencil::point(array_grad(0.2, 0.3))
            .preset(Preset::Art)
            .mode(array_grad(1.0, 0.0))
            .art_threshold(0.0)
            .max_rotations(4)
            .krylov_dim(4)
            .build()
            .unwrap();
        opened.step(&saddle()).unwrap();
        assert_ne!(opened.position()[1].to_le_bytes(), (0.3_f64).to_le_bytes());
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
