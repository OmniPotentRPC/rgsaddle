//! Off-path climbing image.
//!
//! Goswami, Gunde, and Jónsson, Front. Chem. 14, 1807063 (2026),
//! doi:10.3389/fchem.2026.1807063. CI-NEB runs until the climbing index
//! has been stable for five steps and the climbing force is under
//! `lambda` times the initial peak force. The dimer then follows the
//! lowest mode. Positive curvature restores the saved image. A drop in
//! the climbing force reparameterizes the path. A poor alignment lowers
//! the next trigger threshold. Ásgeirsson et al., J. Chem. Theory
//! Comput. 17, 4929 (2021), doi:10.1021/acs.jctc.1c00462, is the static
//! one-shot handover this loop replaces.

use ndarray::{Array1, Array2, ArrayView1, Axis};

use crate::error::SaddleError;
use crate::minmode::{MinModeConfig, MinModeKind, MinModeSession, MinModeStatus, PointSurface};
use rgmin::Method;

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
    pub method: Method,
}

/// Run the dimer at the climbing image. Positive curvature restores `saved`.
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
    let mut session = MinModeSession::new(
        MinModeConfig {
            kind: MinModeKind::Dimer,
            force_tol: request.force_tol,
            max_move: request.max_move,
            method: request.method.clone(),
            ..MinModeConfig::default()
        },
        saved.clone(),
        mode,
    )?;
    session.set_position(saved.clone(), Some(gradient))?;
    let adapter = Row { surface, id };
    let mut best_curvature = f64::INFINITY;
    let mut best_pos = saved.clone();
    let mut best_mode = session.mode().to_owned();
    let mut evals = 0usize;
    let mut curvature = f64::NAN;
    let mut alpha = 1.0;
    let mut positive = false;
    for _ in 0..request.max_mmf {
        let report = session.step(&adapter)?;
        evals += report.evaluations;
        curvature = report.curvature;
        let mode_now = session.mode().to_owned();
        alpha = mode_now.dot(&tangent).abs().min(1.0);
        if curvature > 0.0 {
            positive = true;
            best_pos = saved.clone();
            best_mode = mode_now;
            break;
        }
        if curvature < best_curvature {
            best_curvature = curvature;
            best_pos = session.position().to_owned();
            best_mode = mode_now;
        }
        if alpha < request.alpha_tol || report.status == MinModeStatus::Converged {
            break;
        }
    }
    let (energy, gradient) = adapter.eval(best_pos.view())?;
    evals += 1;
    Ok(DimerEnd {
        position: best_pos,
        gradient,
        energy,
        mode: best_mode,
        curvature,
        alpha,
        evaluations: evals,
        positive_curvature: positive,
    })
}
