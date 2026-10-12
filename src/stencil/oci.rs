//! Off-path climbing image.
//!
//! Goswami, Gunde, and Jónsson, Front. Chem. 14, 1807063 (2026),
//! doi:10.3389/fchem.2026.1807063. CI-NEB runs until the climbing index
//! has been stable for five steps and the climbing force is under
//! `lambda` times the initial peak force. The dimer then follows the
//! reflected force. Each translation is the Newton step `|F| / |λ|`,
//! capped by the band maximum move. The point kept is the one with the
//! smallest force, which is the first-order saddle. Positive
//! curvature stops the walk. A walk that never lowers the force restores
//! the saved image. A drop in the climbing force reparameterizes the path, except
//! when the climber is already inside the force tolerance. A poor
//! alignment lowers the next trigger threshold. Ásgeirsson et al.,
//! J. Chem. Theory Comput. 17, 4929 (2021), doi:10.1021/acs.jctc.1c00462,
//! is the static one-shot handover this loop replaces.

use ndarray::{Array1, Array2, ArrayView1, Axis};

use crate::error::SaddleError;
use crate::minmode::{MinModeConfig, MinModeKind, MinModeSession, PointSurface};

use super::Surface;

pub const LAMBDA: f64 = 0.31;
pub const ALPHA_TOL: f64 = 0.85;
pub const KAPPA: usize = 5;
pub const MAX_MMF: usize = 1000;

pub struct OciControl {
    pub lambda: f64,
    pub alpha_tol: f64,
    pub kappa: usize,
    pub max_mmf: usize,
    pub f0: Option<f64>,
    pub t_mmf: f64,
    pub stable: usize,
    pub prev_ci: Option<usize>,
    pub f_ci: f64,
    pub cached_mode: Option<Array1<f64>>,
    pub skip: bool,
}

impl OciControl {
    pub fn new() -> Self {
        Self {
            lambda: LAMBDA,
            alpha_tol: ALPHA_TOL,
            kappa: KAPPA,
            max_mmf: MAX_MMF,
            f0: None,
            t_mmf: f64::INFINITY,
            stable: 0,
            prev_ci: None,
            f_ci: f64::INFINITY,
            cached_mode: None,
            skip: false,
        }
    }

    pub fn trigger(&self) -> bool {
        self.prev_ci.is_some() && self.stable >= self.kappa && self.f_ci < self.t_mmf
    }

    pub fn observe(&mut self, ci: Option<usize>, f_ci: f64, f0: Option<f64>) {
        match (self.prev_ci, ci) {
            (Some(prev), Some(now)) if prev == now => self.stable += 1,
            (_, Some(now)) => {
                self.stable = 0;
                self.cached_mode = None;
                self.prev_ci = Some(now);
            }
            _ => {
                self.stable = 0;
                self.cached_mode = None;
                self.prev_ci = None;
            }
        }
        if self.f0.is_none()
            && let Some(peak) = f0
            && peak.is_finite()
            && peak > 0.0
        {
            self.f0 = Some(peak);
            self.t_mmf = self.lambda * peak;
        }
        if f_ci.is_finite() {
            self.f_ci = f_ci;
        }
    }
}

pub struct DimerEnd {
    pub position: Array1<f64>,
    pub gradient: Array1<f64>,
    pub energy: f64,
    pub mode: Array1<f64>,
    pub curvature: f64,
    pub alpha: f64,
    pub evaluations: usize,
    pub positive_curvature: bool,
    /// Leave the climbing image where it was. Set on positive curvature
    /// and when no translation lowered the force.
    pub restore: bool,
}

struct Row<'a, S> {
    surface: &'a S,
    id: u64,
}

impl<S: Surface> PointSurface for Row<'_, S> {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let coords = x.to_owned().insert_axis(Axis(0));
        let mut energy = [0.0];
        let mut gradient = Array2::zeros((1, x.len()));
        self.surface
            .eval(&[self.id], coords.view(), &mut energy, gradient.view_mut())?;
        Ok((energy[0], gradient.row(0).to_owned()))
    }
}

pub struct DimerRequest {
    pub alpha_tol: f64,
    pub max_mmf: usize,
    pub force_tol: f64,
    pub max_move: f64,
}

fn linf(force: &Array1<f64>) -> f64 {
    force.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
}

/// Newton step on the translated force, capped by the band move.
///
/// The translated force is the reflection while the curvature is
/// negative. Its length equals the true force, so dividing by `|λ|`
/// is the Newton length. Empty-memory L-BFGS with that scale takes
/// the same step; the cap is the trust radius.
fn newton_step(
    gradient: &Array1<f64>,
    mode: &Array1<f64>,
    curvature: f64,
    limit: f64,
) -> Array1<f64> {
    let eff = if curvature < 0.0 {
        super::reflect(gradient.view(), mode.view())
    } else {
        super::ascend(gradient.view(), mode.view())
    };
    let mut step = &eff * (-1.0 / curvature.abs().max(1e-8));
    let length = step.dot(&step).sqrt();
    let cap = limit.max(1e-8);
    if length > cap {
        step *= cap / length;
    }
    step
}

/// Forward-difference curvature along a unit mode. The rotator's
/// Fourier model extrapolates the action, and on an anharmonic surface
/// that extrapolation can change sign away from the true eigenvalue.
fn fd_curvature<S: Surface>(
    adapter: &Row<'_, S>,
    x: &Array1<f64>,
    gradient: &Array1<f64>,
    mode: &Array1<f64>,
    dr: f64,
) -> Result<f64, SaddleError> {
    let shifted = x + &(mode * dr);
    let gp = adapter.eval(shifted.view())?.1;
    Ok((&gp - gradient).dot(mode) / dr)
}

/// Run the dimer at the climbing image.
///
/// Rotation runs before every translation. The step follows the
/// reflected force by at most the Newton length `|F| / |λ|`. The
/// installed point is the lowest force on that walk. Positive
/// curvature stops the walk. A walk that never lowers the force
/// restores `saved`.
pub fn run_dimer<S: Surface>(
    surface: &S,
    id: u64,
    saved: Array1<f64>,
    gradient: Array1<f64>,
    tangent: Array1<f64>,
    cached: Option<Array1<f64>>,
    request: &DimerRequest,
) -> Result<DimerEnd, SaddleError> {
    let mode = match cached {
        Some(mode) if mode.len() == saved.len() && mode.iter().any(|v| v.abs() > 0.0) => mode,
        _ => tangent.clone(),
    };
    let config = MinModeConfig {
        kind: MinModeKind::Dimer,
        force_tol: request.force_tol,
        max_move: request.max_move,
        max_rotations: 8,
        rotation_tol: 1e-2,
        ..MinModeConfig::default()
    };
    let dr = config.dr;
    let mut session = MinModeSession::new(config, saved.clone(), mode)?;
    session.set_position(saved.clone(), Some(gradient.clone()))?;
    let adapter = Row { surface, id };
    let mut x = saved.clone();
    let mut g = gradient.clone();
    let start_force = linf(&gradient);
    let mut evals = 0usize;
    let mut best_force = f64::INFINITY;
    let mut best_pos = saved.clone();
    let mut best_mode = session.mode().to_owned();
    let mut best_curvature = f64::NAN;
    let mut best_alpha = 1.0;
    let mut curvature = f64::NAN;
    let mut alpha = 1.0;
    let mut positive = false;
    let mut converged = false;
    let mut stalled = 0usize;
    let budget = request.max_mmf.min(16);
    for _ in 0..budget {
        let estimate = session.estimate_mode(&adapter)?;
        evals += estimate.evaluations;
        let mode_now = estimate.mode;
        alpha = mode_now.dot(&tangent).abs().min(1.0);
        curvature = fd_curvature(&adapter, &x, &g, &mode_now, dr)?;
        evals += 1;
        if curvature > 0.0 {
            positive = true;
            break;
        }
        if alpha < request.alpha_tol {
            break;
        }
        let x_new = &x + &newton_step(&g, &mode_now, curvature, request.max_move);
        let (_, g_new) = adapter.eval(x_new.view())?;
        evals += 1;
        let force = linf(&g_new);
        if force < best_force {
            best_force = force;
            best_pos = x_new.clone();
            best_mode = mode_now.clone();
            best_curvature = curvature;
            best_alpha = alpha;
        }
        if force <= request.force_tol {
            converged = true;
            break;
        }
        if force < start_force {
            stalled = 0;
        } else {
            stalled += 1;
        }
        if stalled >= 3 && best_force >= start_force {
            break;
        }
        x = x_new;
        g = g_new;
        session.set_position(x.clone(), Some(g.clone()))?;
    }
    let improved = best_force < start_force || converged;
    if !improved {
        return Ok(DimerEnd {
            position: saved,
            gradient,
            energy: f64::NAN,
            mode: best_mode,
            curvature,
            alpha,
            evaluations: evals,
            positive_curvature: positive,
            restore: true,
        });
    }
    let (energy, gradient) = adapter.eval(best_pos.view())?;
    evals += 1;
    Ok(DimerEnd {
        position: best_pos,
        gradient,
        energy,
        mode: best_mode,
        curvature: best_curvature,
        alpha: best_alpha,
        evaluations: evals,
        positive_curvature: false,
        restore: false,
    })
}
