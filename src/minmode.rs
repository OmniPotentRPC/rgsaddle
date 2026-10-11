//! Minimum-mode saddle search: find the lowest curvature direction,
//! invert the force along it, take one step. Stepping, like the band.
//!
//! A session is long-lived: [`MinModeSession::set_position`] moves it
//! to a new point (with the host's gradient there, when the host has
//! one) and keeps the mode as the next rotation's seed, and
//! [`MinModeSession::estimate_mode`] refreshes the mode without a
//! translation, for a host that climbs with its own optimizer.

use std::f64::consts::PI;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, Array2, ArrayView1};
use rgmin::{Accept, Control, Method, Oracle, Solver};

use crate::error::SaddleError;
use crate::force::ForceGate;
use crate::kappa::{Complement, KappaDimerConfig, kappa_dimer_force};

/// The caller's surface for a single geometry.
pub trait PointSurface: Sync {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError>;

    /// Optional analytic Cartesian Hessian action. None selects gradient differences.
    fn hessian_vector(
        &self,
        _x: ArrayView1<f64>,
        _v: ArrayView1<f64>,
    ) -> Result<Option<Array1<f64>>, SaddleError> {
        Ok(None)
    }

    /// Rigid or constrained Cartesian directions, one direction per row.
    /// Molecular callers supply translations and rotations; fixed substrates
    /// supply only their actual symmetries. These are evaluated at the query.
    fn excluded_modes(&self, x: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
        Ok(Array2::zeros((0, x.len())))
    }

    /// Energy and Cartesian gradient in a periodic cell, row-major 3x3.
    ///
    /// Default ignores the cell and calls [`Self::eval`].
    fn eval_in_cell(
        &self,
        x: ArrayView1<f64>,
        _cell: &[f64; 9],
    ) -> Result<(f64, Array1<f64>), SaddleError> {
        self.eval(x)
    }

    /// Optional `dE/dC`, row-major 3x3. `None` means the session
    /// finite-differences [`Self::eval_in_cell`].
    fn cell_grad(
        &self,
        x: ArrayView1<f64>,
        cell: &[f64; 9],
    ) -> Result<Option<[f64; 9]>, SaddleError> {
        let _ = (x, cell);
        Ok(None)
    }
}

/// Finite difference behind every Hessian action.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FiniteDifference {
    /// `(g(x + dr v) - g(x)) / dr`: one gradient per action (the centre
    /// is known), error `dr/2 T[v,v,v]`.
    #[default]
    Forward,
    /// `(g(x + dr v) - g(x - dr v)) / (2 dr)`: two gradients per action,
    /// error `dr^2/6 Q[v,v,v,v]`.
    Central,
}

/// How the lowest mode is estimated.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MinModeKind {
    /// Finite-difference dimer (Henkelman-Jonsson 1999) rotated by the
    /// modified Newton step of Heyden, Bell, and Keil (2005) and
    /// Kastner and Sherwood (2008) along conjugate rotation
    /// directions: one gradient per rotation, the gradient at the
    /// rotated dimer is extrapolated.
    #[default]
    Dimer,
    /// Lanczos on finite-difference Hessian actions: a Krylov
    /// subspace per compute, one gradient per action, stopped when
    /// the lowest Ritz pair's residual meets `rotation_tol`.
    Lanczos,
}

impl MinModeKind {
    /// The discriminant used by the C and Python interfaces.
    pub fn to_abi(self) -> i32 {
        match self {
            Self::Dimer => 0,
            Self::Lanczos => 1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MinModeConfig {
    pub kind: MinModeKind,
    /// Dimer separation / finite-difference displacement, Angstrom.
    /// The curvature is a forward difference against the centre
    /// gradient, with truncation error `dr/2 * T[v,v,v]` (`T` the third
    /// derivative) and noise error `2 eps_g / dr` for gradient noise
    /// `eps_g`; see `validation/fd_curvature.py`.
    pub dr: f64,
    /// Forward (default) or central difference for the actions.
    pub difference: FiniteDifference,
    /// Rotation stops when the rotational force `|H v - (v.H v) v|`
    /// falls under this. For Lanczos it is the Ritz residual bound.
    pub rotation_tol: f64,
    /// Dimer rotation also stops when the optimal rotation angle of an
    /// iteration falls under this many radians (eOn
    /// `converged_angle`). Zero disables the angle test.
    pub rotation_angle_tol: f64,
    pub max_rotations: usize,
    /// Krylov dimension for [`MinModeKind::Lanczos`].
    pub krylov_dim: usize,
    /// Ritz vectors kept when a Lanczos basis fills before the residual
    /// test. Zero keeps the single-pass stop at [`Self::krylov_dim`].
    /// Wu and Simon, SIAM J. Matrix Anal. Appl. 22, 602 (2000),
    /// doi:10.1137/S0895479898334605.
    pub lanczos_restart_keep: usize,
    /// Recycle the previous second Ritz vector when
    /// `(theta_2 - theta_1) / |theta_1|` is strictly below this.
    /// The vector is only a direction: the next estimate applies a fresh
    /// Hessian action and never reuses the old one.
    pub lanczos_recycle_gap: f64,
    /// Translation stops when max|F| falls under this and the
    /// curvature along the lowest mode is negative.
    pub force_tol: f64,
    pub max_move: f64,
    pub method: Method,
}

impl Default for MinModeConfig {
    fn default() -> Self {
        Self {
            kind: MinModeKind::Dimer,
            dr: 1e-3,
            difference: FiniteDifference::Forward,
            rotation_tol: 1e-4,
            rotation_angle_tol: 0.0,
            max_rotations: 20,
            krylov_dim: 12,
            lanczos_restart_keep: 2,
            lanczos_recycle_gap: 0.2,
            force_tol: 1e-3,
            max_move: 0.2,
            method: Method::Fire {
                kind: rgmin::FireKind::V2,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MinModeStatus {
    Running,
    Converged,
}

#[derive(Clone, Debug)]
pub struct MinModeReport {
    pub status: MinModeStatus,
    pub max_force: f64,
    pub curvature: f64,
    /// Rotation iterations (dimer) or Hessian actions (Lanczos).
    pub rotations: usize,
    pub iteration: usize,
    /// Surface evaluations during this call, centre included.
    pub evaluations: usize,
}

/// A refreshed lowest mode at the current point.
#[derive(Clone, Debug)]
pub struct ModeEstimate {
    pub mode: Array1<f64>,
    pub curvature: f64,
    /// Rotation iterations (dimer) or Hessian actions (Lanczos).
    pub rotations: usize,
    /// Surface evaluations, the centre included when it was neither
    /// cached nor supplied by the host.
    pub evaluations: usize,
    /// Lanczos reached the residual bound (or an invariant subspace).
    /// False when the action budget or, with `lanczos_restart_keep = 0`,
    /// the Krylov dimension stopped the basis first. The dimer sets this
    /// when the rotation exits before `max_rotations`.
    pub converged: bool,
}

/// Rotation of the mode between two steps past which the translation
/// optimizer's memory is dropped. The effective gradient is `R g` with
/// `R = I - 2 tau tau'`; when `tau` turns by `theta`,
/// `|R' - R|_2 = 2 sin(theta)` (`validation/mode_reset.py`), so the
/// effective Hessian `R H R` that the stored pairs describe moves by up
/// to `4 sin(theta) |H|`. Under eOn's converged angle (5 degrees) that
/// is at most `0.35 |H|` and the dimer counts the mode as unchanged: the
/// pairs and the FIRE velocity stay and only the cached evaluation
/// goes. Past it the mode is a different one and the memory is reset.
pub const MODE_RESET_ANGLE: f64 = 5.0 * PI / 180.0;

/// A first-order saddle has a vanishing force and one negative
/// curvature. A vanishing force alone also describes a minimum, which
/// the inverted-force step then leaves; the gate therefore requires
/// `curvature < 0` alongside `max_force <= force_tol`.
fn converged(max_force: f64, curvature: f64, force_tol: f64) -> bool {
    max_force <= force_tol && curvature < 0.0
}

fn normalize(mut v: Array1<f64>) -> Array1<f64> {
    let n = v.dot(&v).sqrt();
    if n > 1e-14 {
        v /= n;
    }
    v
}

/// A host mode must admit the unit normalization used by the curvature
/// and reflected-force contracts. Reject invalid norms before division.
fn checked_mode(mode: Array1<f64>) -> Result<Array1<f64>, SaddleError> {
    let norm = mode.dot(&mode).sqrt();
    if !(norm.is_finite() && norm > 1e-14) {
        return Err(SaddleError::Invalid(
            "mode must have a finite norm above 1e-14".into(),
        ));
    }
    Ok(mode / norm)
}

/// The surface, counting the calls one compute makes and checking
/// what comes back.
struct Counted<'a, S: PointSurface> {
    surface: &'a S,
    calls: &'a AtomicUsize,
    space: Option<&'a Complement>,
}

impl<S: PointSurface> Counted<'_, S> {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let (e, g) = self.surface.eval(x)?;
        if g.len() != x.len() {
            return Err(SaddleError::Shape("surface gradient length changed".into()));
        }
        if !e.is_finite() || !g.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("min-mode gradient"));
        }
        Ok((e, g))
    }

    /// Gradient at `x + step * unit`.
    fn gradient_along(
        &self,
        x: ArrayView1<f64>,
        unit: ArrayView1<f64>,
        step: f64,
    ) -> Result<Array1<f64>, SaddleError> {
        let shifted = &x + &(&unit * step);
        Ok(self.eval(shifted.view())?.1)
    }

    /// Finite-difference Hessian action `H unit`: forward against the
    /// centre gradient (one call) or central (two calls).
    fn action(
        &self,
        x: ArrayView1<f64>,
        g0: ArrayView1<f64>,
        unit: ArrayView1<f64>,
        config: &MinModeConfig,
    ) -> Result<Array1<f64>, SaddleError> {
        let lifted;
        let direction = if let Some(space) = self.space {
            lifted = space.lift(unit);
            lifted.view()
        } else {
            unit
        };
        let action = if let Some(hv) = self.surface.hessian_vector(x, direction)? {
            if hv.len() != x.len() {
                return Err(SaddleError::Shape(
                    "surface Hessian action length changed".into(),
                ));
            }
            if !hv.iter().all(|v| v.is_finite()) {
                return Err(SaddleError::NonFinite("surface Hessian action"));
            }
            hv
        } else {
            let dr = config.dr;
            match config.difference {
                FiniteDifference::Forward => (&self.gradient_along(x, direction, dr)? - &g0) / dr,
                FiniteDifference::Central => {
                    let plus = self.gradient_along(x, direction, dr)?;
                    let minus = self.gradient_along(x, direction, -dr)?;
                    (&plus - &minus) / (2.0 * dr)
                }
            }
        };
        Ok(match self.space {
            Some(space) => space.reduce(action.view()),
            None => action,
        })
    }
}

/// Component of `h` perpendicular to the unit vector `n`.
fn perpendicular(h: &Array1<f64>, n: &Array1<f64>) -> Array1<f64> {
    h - &(n * h.dot(n))
}

/// Dimer rotation by the modified Newton step.
///
/// With `h(n) = (g(x + dr n) - g0) / dr`, the curvature along the unit
/// vector `n` is `C = h.n` and its gradient on the sphere is
/// `2 (h - C n)`. Rotating in the plane of `n` and a unit `theta`
/// perpendicular to it, `n(phi) = n cos phi + theta sin phi`, a
/// quadratic surface gives `C(phi) = a0/2 + a1 cos 2phi + b1 sin 2phi`
/// with `b1 = h.theta`. One gradient at a trial angle `phi1` fixes
/// `a1`, the minimizing `phi` follows in closed form, and the gradient
/// at the rotated dimer is the linear combination
/// `g(phi) = g0 + sin(phi1 - phi)/sin(phi1) (g1 - g0)
///          + sin(phi)/sin(phi1) (g1' - g0)`, which for the Hessian
/// action is `h(phi) = w_old h + w_trial h1` with the same weights,
/// exact on a quadratic surface (`validation/dimer_rotation.py`), so
/// the next iteration starts without a gradient call. Rotation
/// directions are Polak-Ribiere conjugate on the rotational force,
/// with the previous direction carried along the rotation.
fn rotate_dimer<S: PointSurface>(
    surface: &Counted<'_, S>,
    x: ArrayView1<f64>,
    g0: ArrayView1<f64>,
    mode: Array1<f64>,
    config: &MinModeConfig,
) -> Result<(Array1<f64>, f64, usize), SaddleError> {
    let mut n = normalize(mode);
    let mut h = surface.action(x, g0, n.view(), config)?;
    let mut curvature = h.dot(&n);
    let mut rotations = 0;
    // Rotational force (descent direction of the curvature on the
    // sphere) and the conjugate direction of the previous iteration.
    let mut force_prev: Option<Array1<f64>> = None;
    let mut dir_prev: Option<Array1<f64>> = None;
    loop {
        let force = -perpendicular(&h, &n);
        let force_norm = force.dot(&force).sqrt();
        if force_norm <= config.rotation_tol || rotations >= config.max_rotations {
            break;
        }
        let mut dir = force.clone();
        if let (Some(fp), Some(dp)) = (&force_prev, &dir_prev) {
            let gamma = ((&force - fp).dot(&force) / fp.dot(fp)).max(0.0);
            if gamma.is_finite() {
                dir = &dir + &(dp * gamma);
            }
        }
        dir = perpendicular(&dir, &n);
        let dir_norm = dir.dot(&dir).sqrt();
        if dir_norm <= 1e-14 * (1.0 + force_norm) {
            break;
        }
        let theta = &dir / dir_norm;
        // dC/dphi at phi = 0 is 2 h.theta (negative along a descent
        // direction). The trial angle is the Kastner-Sherwood guess
        // from a harmonic model, kept inside (1e-4, pi/4].
        let b1 = h.dot(&theta);
        let phi1 = (0.5 * (b1.abs() / (curvature.abs() + 1e-300)).atan()).clamp(1e-4, 0.25 * PI);
        let n1 = normalize(&(&n * phi1.cos()) + &(&theta * phi1.sin()));
        let h1 = surface.action(x, g0, n1.view(), config)?;
        let c1 = h1.dot(&n1);
        rotations += 1;
        // Fourier fit through C(0), C'(0) = 2 b1, and C(phi1).
        let a1 = (curvature - c1 + b1 * (2.0 * phi1).sin()) / (1.0 - (2.0 * phi1).cos());
        let a0 = 2.0 * (curvature - a1);
        let c_at = |p: f64| 0.5 * a0 + a1 * (2.0 * p).cos() + b1 * (2.0 * p).sin();
        let mut phi = 0.5 * (b1 / a1).atan();
        if c_at(phi) > c_at(phi + 0.5 * PI) {
            phi += 0.5 * PI;
        }
        if phi > 0.5 * PI {
            phi -= PI;
        }
        let c_min = c_at(phi);
        let slack = 1e-10 * (curvature.abs() + c1.abs() + 1.0);
        if !(c_min.is_finite() && phi.is_finite()) || c_min > c1.min(curvature) + slack {
            // The harmonic model does not hold here (anharmonic or
            // noisy surface): keep the better of the two measured
            // dimers and continue from it.
            if c1 < curvature {
                n = n1;
                h = h1;
                curvature = c1;
            }
            force_prev = Some(force);
            dir_prev = Some(theta * dir_norm);
            if config.rotation_angle_tol > 0.0 && phi1 < config.rotation_angle_tol {
                break;
            }
            continue;
        }
        let (s, c) = phi.sin_cos();
        let s1 = phi1.sin();
        let w_old = (phi1 - phi).sin() / s1;
        let w_trial = s / s1;
        h = &(&h * w_old) + &(&h1 * w_trial);
        let theta_rotated = &(&theta * c) - &(&n * s);
        n = normalize(&(&n * c) + &(&theta * s));
        curvature = h.dot(&n);
        force_prev = Some(force);
        dir_prev = Some(theta_rotated * dir_norm);
        if config.rotation_angle_tol > 0.0 && phi.abs() < config.rotation_angle_tol {
            break;
        }
    }
    Ok((n, curvature, rotations))
}

struct LanczosOut {
    mode: Array1<f64>,
    curvature: f64,
    actions: usize,
    converged: bool,
    /// Second-lowest Ritz vector and `(theta_2 - theta_1) / |theta_1|`.
    second: RitzTail,
}

/// Second-lowest Ritz vector and its gap ratio.
type RitzTail = Option<(Array1<f64>, f64)>;

/// Mode, curvature, actions, residual flag, and the optional second vector.
type ModeSample = (Array1<f64>, f64, usize, bool, RitzTail);

struct LanczosBasis {
    q: Vec<Array1<f64>>,
    alpha: Vec<f64>,
    beta: Vec<f64>,
    w: Array1<f64>,
    beta_next: f64,
    actions: usize,
}

/// `Q^T w` over columns in index order, then `w - Q (Q^T w)`.
///
/// The product stays in this function so a deterministic emulated
/// kernel can replace it later. The sum is one dot per column, in
/// column order, which does not depend on the thread count.
const REORTH_SERIAL_BELOW: usize = 65_536;
const REORTH_CHUNK: usize = 4_096;

fn reorthogonalize(basis: &[Array1<f64>], mut w: Array1<f64>) -> Array1<f64> {
    for column in basis {
        let overlap = column_overlap(w.view(), column.view());
        if w.len() < REORTH_SERIAL_BELOW {
            w = &w - &(column * overlap);
        } else {
            let ws = w.as_slice_mut().expect("mode storage is contiguous");
            let cs = column.as_slice().expect("mode storage is contiguous");
            scale_sub(ws, cs, overlap);
        }
    }
    w
}

/// One dot. Short vectors use the serial `ndarray` product, which is
/// the single-pass Lanczos overlap. Long vectors split into fixed
/// chunks and add the partials in chunk order, so the bits do not
/// depend on the thread count.
fn column_overlap(w: ArrayView1<f64>, column: ArrayView1<f64>) -> f64 {
    if w.len() < REORTH_SERIAL_BELOW {
        return w.dot(&column);
    }
    use rayon::prelude::{IntoParallelIterator, ParallelIterator};
    let starts: Vec<usize> = (0..w.len()).step_by(REORTH_CHUNK).collect();
    let partials: Vec<f64> = starts
        .into_par_iter()
        .map(|start| {
            let end = (start + REORTH_CHUNK).min(w.len());
            let mut sum = 0.0;
            for i in start..end {
                sum += w[i] * column[i];
            }
            sum
        })
        .collect();
    let mut sum = 0.0;
    for partial in partials {
        sum += partial;
    }
    sum
}

fn scale_sub(w: &mut [f64], column: &[f64], overlap: f64) {
    use rayon::prelude::{IndexedParallelIterator, ParallelIterator, ParallelSliceMut};
    // Same chunk width as [`column_overlap`]. Each chunk writes its
    // own indices, so the update matches at one thread and at many.
    w.par_chunks_mut(REORTH_CHUNK)
        .enumerate()
        .for_each(|(chunk, dest)| {
            let start = chunk * REORTH_CHUNK;
            for (offset, value) in dest.iter_mut().enumerate() {
                *value -= column[start + offset] * overlap;
            }
        });
}

/// Lanczos: build a Krylov basis of finite-difference Hessian actions
/// and take the lowest Ritz vector. The residual is `beta_j |s_j|`
/// (Paige 1971). A seed that already is the mode costs one action.
///
/// With `lanczos_restart_keep = 0` the basis stops at `krylov_dim`,
/// which is the previous single pass. A positive keep thick-restarts
/// that basis (Wu and Simon, SIAM J. Matrix Anal. Appl. 22, 602 (2000),
/// doi:10.1137/S0895479898334605) until the residual meets
/// `rotation_tol` or `max_rotations` actions are spent.
/// `recycle`, when the caller kept it, is a direction from the previous
/// point. It is orthogonalized and costs one new action. The old action
/// is not reused.
fn lanczos_mode<S: PointSurface>(
    surface: &Counted<'_, S>,
    x: ArrayView1<f64>,
    g0: ArrayView1<f64>,
    seed: Array1<f64>,
    config: &MinModeConfig,
    recycle: Option<&Array1<f64>>,
) -> Result<LanczosOut, SaddleError> {
    let n = seed.len();
    let m = config.krylov_dim.min(n).max(1);
    let mut q: Vec<Array1<f64>> = Vec::with_capacity(m);
    let mut alpha: Vec<f64> = Vec::with_capacity(m);
    let mut beta: Vec<f64> = Vec::with_capacity(m);
    q.push(normalize(seed));
    if let Some(extra) = recycle.filter(|_| config.lanczos_restart_keep > 0) {
        return lanczos_recycled(surface, x, g0, config, q, extra);
    }

    let mut actions = 0;
    loop {
        let j = alpha.len();
        let hv = surface.action(x, g0, q[j].view(), config)?;
        actions += 1;
        let a = hv.dot(&q[j]);
        alpha.push(a);
        let mut w = &hv - &(&q[j] * a);
        if j > 0 {
            w = &w - &(&q[j - 1] * beta[j - 1]);
        }
        w = reorthogonalize(&q, w);
        let b = w.dot(&w).sqrt();
        let (theta, s) = lowest_ritz(&alpha, &beta);
        let second = if config.lanczos_restart_keep > 0 {
            ritz_pair(&alpha, &beta, &q).2
        } else {
            None
        };
        let residual = b * s[j].abs();
        let ok = residual <= config.rotation_tol || b <= 1e-12;
        let budget = config.lanczos_restart_keep > 0 && actions >= config.max_rotations;
        let cap = alpha.len() == m;
        if ok || budget || (cap && config.lanczos_restart_keep == 0) {
            return Ok(LanczosOut {
                mode: combine_ritz(&q, &s),
                curvature: theta,
                actions,
                converged: ok,
                second,
            });
        }
        if cap {
            return lanczos_restarted(
                surface,
                x,
                g0,
                config,
                LanczosBasis {
                    q,
                    alpha,
                    beta,
                    w,
                    beta_next: b,
                    actions,
                },
            );
        }
        beta.push(b);
        q.push(w / b);
    }
}

fn combine_ritz(q: &[Array1<f64>], s: &[f64]) -> Array1<f64> {
    let mut mode = Array1::zeros(q[0].len());
    for (qi, si) in q.iter().zip(s.iter()) {
        mode = &mode + &(qi * *si);
    }
    normalize(mode)
}

fn ritz_pair(alpha: &[f64], beta: &[f64], q: &[Array1<f64>]) -> (f64, Vec<f64>, RitzTail) {
    let (evals, evecs) = tridiagonal_eigen(alpha, beta);
    let mut order: Vec<usize> = (0..evals.len()).collect();
    order.sort_by(|&i, &j| evals[i].total_cmp(&evals[j]));
    let lowest = order[0];
    let s: Vec<f64> = evecs.iter().map(|row| row[lowest]).collect();
    let second = order.get(1).map(|&col| {
        let coeffs: Vec<f64> = evecs.iter().map(|row| row[col]).collect();
        let gap = if evals[lowest].abs() > 0.0 {
            (evals[col] - evals[lowest]) / evals[lowest].abs()
        } else {
            f64::INFINITY
        };
        (combine_ritz(q, &coeffs), gap)
    });
    (evals[lowest], s, second)
}

/// Tridiagonal factor of a Lanczos projection. The single-pass return
/// still goes through [`lowest_ritz`], so `lanczos_restart_keep = 0`
/// does not call this.
fn tridiagonal_eigen(alpha: &[f64], beta: &[f64]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let k = alpha.len();
    let mut t = vec![vec![0.0; k]; k];
    for i in 0..k {
        t[i][i] = alpha[i];
        if i + 1 < k {
            t[i][i + 1] = beta[i];
            t[i + 1][i] = beta[i];
        }
    }
    projected_eigen(&t)
}

fn dense_eigen(proj: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    projected_eigen(proj)
}

/// Eigenpairs of a small symmetric projection.
///
/// Householder reduction, then an implicit-shift QL. Columns of the
/// second matrix are the eigenvectors, `v[row][col]`. Duersch, Shao,
/// Yang, and Gu, SIAM J. Sci. Comput. 40, C655 (2018),
/// doi:10.1137/17M1129830, is the stable reference for this tridiagonal
/// eigenproblem at larger order. The Krylov projection stays small
/// enough for the QL. [`lowest_ritz`] keeps its Jacobi sweep so a keep
/// of zero matches the single pass.
#[allow(clippy::needless_range_loop)]
fn projected_eigen(a: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    if n == 1 {
        return (vec![a[0][0]], vec![vec![1.0]]);
    }
    let mut symmetric = a.to_vec();
    for i in 0..n {
        for j in 0..i {
            let value = 0.5 * (symmetric[i][j] + symmetric[j][i]);
            symmetric[i][j] = value;
            symmetric[j][i] = value;
        }
    }
    let (mut diag, mut off, mut vectors) = householder_tridiagonal(&symmetric);
    implicit_shift_ql(&mut diag, &mut off, &mut vectors);
    (diag, vectors)
}

struct Arrow {
    q: Vec<Array1<f64>>,
    proj: Vec<Vec<f64>>,
    /// Next column that still needs a Hessian action.
    acted: usize,
}

fn arrow_from_tridiagonal(
    q: &[Array1<f64>],
    alpha: &[f64],
    beta: &[f64],
    w: Array1<f64>,
    b: f64,
    k_keep: usize,
) -> Arrow {
    let (evals, evecs) = tridiagonal_eigen(alpha, beta);
    let mut order: Vec<usize> = (0..evals.len()).collect();
    order.sort_by(|&i, &j| evals[i].total_cmp(&evals[j]));
    let k = k_keep.min(order.len().saturating_sub(1)).max(1);
    let m = q.len();
    let mut kept = Vec::with_capacity(k + 1);
    let mut thetas = Vec::with_capacity(k);
    let mut couplings = Vec::with_capacity(k);
    for &col in &order[..k] {
        let coeffs: Vec<f64> = evecs.iter().map(|row| row[col]).collect();
        let mut y = Array1::zeros(q[0].len());
        for (qi, si) in q.iter().zip(coeffs.iter()) {
            y = &y + &(qi * *si);
        }
        kept.push(y);
        thetas.push(evals[col]);
        couplings.push(b * evecs[m - 1][col]);
    }
    kept.push(&w / b);
    let n = kept.len();
    let mut proj = vec![vec![0.0; n]; n];
    for i in 0..k {
        proj[i][i] = thetas[i];
        proj[i][k] = couplings[i];
        proj[k][i] = couplings[i];
    }
    Arrow {
        q: kept,
        proj,
        acted: k,
    }
}

fn lanczos_restarted<S: PointSurface>(
    surface: &Counted<'_, S>,
    x: ArrayView1<f64>,
    g0: ArrayView1<f64>,
    config: &MinModeConfig,
    built: LanczosBasis,
) -> Result<LanczosOut, SaddleError> {
    let LanczosBasis {
        q,
        alpha,
        beta,
        w,
        beta_next: b,
        actions,
    } = built;
    let k_keep = config.lanczos_restart_keep;
    if k_keep == 0 || q.len() < 2 {
        let (theta, s, second) = ritz_pair(&alpha, &beta, &q);
        return Ok(LanczosOut {
            mode: combine_ritz(&q, &s),
            curvature: theta,
            actions,
            converged: false,
            second,
        });
    }
    let arrow = arrow_from_tridiagonal(&q, &alpha, &beta, w, b, k_keep);
    lanczos_arrow(surface, x, g0, config, arrow, actions)
}

fn lanczos_recycled<S: PointSurface>(
    surface: &Counted<'_, S>,
    x: ArrayView1<f64>,
    g0: ArrayView1<f64>,
    config: &MinModeConfig,
    mut q: Vec<Array1<f64>>,
    extra: &Array1<f64>,
) -> Result<LanczosOut, SaddleError> {
    let hv = surface.action(x, g0, q[0].view(), config)?;
    let mut actions = 1;
    let a = hv.dot(&q[0]);
    let w = reorthogonalize(&q, &hv - &(&q[0] * a));
    let b = w.dot(&w).sqrt();
    let mut direction = reorthogonalize(&q, extra.clone());
    let norm = direction.dot(&direction).sqrt();
    if norm <= 1e-12 {
        let (theta, s, second) = ritz_pair(&[a], &[], &q);
        let residual = b * s[0].abs();
        return Ok(LanczosOut {
            mode: combine_ritz(&q, &s),
            curvature: theta,
            actions,
            converged: residual <= config.rotation_tol || b <= 1e-12,
            second,
        });
    }
    let _ = (w, b);
    direction /= norm;
    q.push(direction);
    let mut proj = vec![vec![0.0; 2]; 2];
    proj[0][0] = a;
    proj[0][1] = 0.0;
    proj[1][0] = 0.0;
    actions += 1;
    let hv2 = surface.action(x, g0, q[1].view(), config)?;
    proj[1][1] = hv2.dot(&q[1]);
    proj[0][1] = hv2.dot(&q[0]);
    proj[1][0] = proj[0][1];
    let mut acc = &hv2 - &(&q[0] * proj[0][1]);
    acc = &acc - &(&q[1] * proj[1][1]);
    acc = reorthogonalize(&q, acc);
    let b2 = acc.dot(&acc).sqrt();
    let (theta, s, residual, second) = dense_ritz(&proj, &q, b2);
    let ok = residual <= config.rotation_tol || b2 <= 1e-12;
    let m = config.krylov_dim.min(q[0].len()).max(1);
    if ok || actions >= config.max_rotations || q.len() >= m {
        return Ok(LanczosOut {
            mode: combine_ritz(&q, &s),
            curvature: theta,
            actions,
            converged: ok,
            second,
        });
    }
    for row in proj.iter_mut() {
        row.push(0.0);
    }
    let mut col = vec![0.0; q.len() + 1];
    col[1] = b2;
    proj.push(col);
    q.push(&acc / b2);
    lanczos_arrow(surface, x, g0, config, Arrow { q, proj, acted: 2 }, actions)
}

fn lanczos_arrow<S: PointSurface>(
    surface: &Counted<'_, S>,
    x: ArrayView1<f64>,
    g0: ArrayView1<f64>,
    config: &MinModeConfig,
    mut arrow: Arrow,
    mut actions: usize,
) -> Result<LanczosOut, SaddleError> {
    let m = config.krylov_dim.min(arrow.q[0].len()).max(1);
    let k_keep = config.lanczos_restart_keep.max(1);
    loop {
        if arrow.acted >= arrow.q.len() {
            return Ok(arrow_out(&arrow, actions, false, 0.0));
        }
        if actions >= config.max_rotations {
            return Ok(arrow_out(&arrow, actions, false, 0.0));
        }
        let j = arrow.acted;
        let hv = surface.action(x, g0, arrow.q[j].view(), config)?;
        actions += 1;
        let n = arrow.q.len();
        for i in 0..n {
            let dot = hv.dot(&arrow.q[i]);
            arrow.proj[i][j] = dot;
            arrow.proj[j][i] = dot;
        }
        let mut w = hv;
        for i in 0..n {
            w = &w - &(&arrow.q[i] * arrow.proj[i][j]);
        }
        w = reorthogonalize(&arrow.q, w);
        let b = w.dot(&w).sqrt();
        let (theta, s, residual, second) = dense_ritz(&arrow.proj, &arrow.q, b);
        let ok = residual <= config.rotation_tol || b <= 1e-12;
        if ok || actions >= config.max_rotations {
            return Ok(LanczosOut {
                mode: combine_ritz(&arrow.q, &s),
                curvature: theta,
                actions,
                converged: ok,
                second,
            });
        }
        if arrow.q.len() == m {
            if k_keep + 1 >= arrow.q.len() {
                return Ok(LanczosOut {
                    mode: combine_ritz(&arrow.q, &s),
                    curvature: theta,
                    actions,
                    converged: false,
                    second,
                });
            }
            arrow = arrow_from_dense(&arrow, &w, b, k_keep);
            continue;
        }
        for row in arrow.proj.iter_mut() {
            row.push(0.0);
        }
        let mut col = vec![0.0; n + 1];
        col[j] = b;
        arrow.proj.push(col);
        arrow.q.push(&w / b);
        arrow.acted = j + 1;
    }
}

fn arrow_out(arrow: &Arrow, actions: usize, converged: bool, b: f64) -> LanczosOut {
    let n = arrow.acted.min(arrow.q.len()).max(1);
    let proj: Vec<Vec<f64>> = arrow.proj[..n]
        .iter()
        .map(|row| row[..n].to_vec())
        .collect();
    let (theta, s, _, second) = dense_ritz(&proj, &arrow.q[..n], b);
    LanczosOut {
        mode: combine_ritz(&arrow.q[..n], &s),
        curvature: theta,
        actions,
        converged,
        second,
    }
}

fn dense_ritz(proj: &[Vec<f64>], q: &[Array1<f64>], b: f64) -> (f64, Vec<f64>, f64, RitzTail) {
    let (evals, evecs) = dense_eigen(proj);
    let mut order: Vec<usize> = (0..evals.len()).collect();
    order.sort_by(|&i, &j| evals[i].total_cmp(&evals[j]));
    let lowest = order[0];
    let s: Vec<f64> = evecs.iter().map(|row| row[lowest]).collect();
    let residual = b * s.last().copied().unwrap_or(0.0).abs();
    let second = order.get(1).and_then(|&col| {
        if col >= evecs[0].len() {
            return None;
        }
        let coeffs: Vec<f64> = evecs.iter().map(|row| row[col]).collect();
        let gap = if evals[lowest].abs() > 0.0 {
            (evals[col] - evals[lowest]) / evals[lowest].abs()
        } else {
            f64::INFINITY
        };
        Some((combine_ritz(q, &coeffs), gap))
    });
    (evals[lowest], s, residual, second)
}

fn arrow_from_dense(arrow: &Arrow, w: &Array1<f64>, b: f64, k_keep: usize) -> Arrow {
    let (evals, evecs) = dense_eigen(&arrow.proj);
    let mut order: Vec<usize> = (0..evals.len()).collect();
    order.sort_by(|&i, &j| evals[i].total_cmp(&evals[j]));
    let k = k_keep.min(order.len().saturating_sub(1)).max(1);
    let m = arrow.q.len();
    let mut kept = Vec::with_capacity(k + 1);
    let mut thetas = Vec::with_capacity(k);
    let mut couplings = Vec::with_capacity(k);
    for &col in &order[..k] {
        let coeffs: Vec<f64> = evecs.iter().map(|row| row[col]).collect();
        let mut y = Array1::zeros(arrow.q[0].len());
        for (qi, si) in arrow.q.iter().zip(coeffs.iter()) {
            y = &y + &(qi * *si);
        }
        kept.push(y);
        thetas.push(evals[col]);
        couplings.push(b * evecs[m - 1][col]);
    }
    kept.push(w / b);
    let n = kept.len();
    let mut proj = vec![vec![0.0; n]; n];
    for i in 0..k {
        proj[i][i] = thetas[i];
        proj[i][k] = couplings[i];
        proj[k][i] = couplings[i];
    }
    Arrow {
        q: kept,
        proj,
        acted: k,
    }
}

#[allow(clippy::needless_range_loop)]
/// Symmetric Householder reduction. Returns the diagonal, the
/// subdiagonal in Numerical Recipes layout (`off[0] = 0`, `off[i]`
/// couples rows `i - 1` and `i`), and the orthogonal factor as columns.
fn householder_tridiagonal(a_in: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
    let n = a_in.len();
    let mut a = vec![vec![0.0; n + 1]; n + 1];
    for i in 1..=n {
        for j in 1..=n {
            a[i][j] = a_in[i - 1][j - 1];
        }
    }
    let mut diag = vec![0.0; n + 1];
    let mut off = vec![0.0; n + 1];
    for i in (2..=n).rev() {
        let l = i - 1;
        let mut h = 0.0;
        if l > 1 {
            let mut scale = 0.0;
            for k in 1..=l {
                scale += a[i][k].abs();
            }
            if scale == 0.0 {
                off[i] = a[i][l];
            } else {
                for k in 1..=l {
                    a[i][k] /= scale;
                    h += a[i][k] * a[i][k];
                }
                let pivot = a[i][l];
                let signed = if pivot >= 0.0 { -h.sqrt() } else { h.sqrt() };
                off[i] = scale * signed;
                h -= pivot * signed;
                a[i][l] = pivot - signed;
                let mut dot_u = 0.0;
                for j in 1..=l {
                    a[j][i] = a[i][j] / h;
                    let mut row = 0.0;
                    for k in 1..=j {
                        row += a[j][k] * a[i][k];
                    }
                    for k in (j + 1)..=l {
                        row += a[k][j] * a[i][k];
                    }
                    off[j] = row / h;
                    dot_u += off[j] * a[i][j];
                }
                let half = dot_u / (h + h);
                for j in 1..=l {
                    let column = a[i][j];
                    let corrected = off[j] - half * column;
                    off[j] = corrected;
                    for k in 1..=j {
                        a[j][k] -= column * off[k] + corrected * a[i][k];
                    }
                }
            }
        } else {
            off[i] = a[i][l];
        }
        diag[i] = h;
    }
    diag[1] = 0.0;
    off[1] = 0.0;
    for i in 1..=n {
        let l = i - 1;
        if diag[i] != 0.0 {
            for j in 1..=l {
                let mut product = 0.0;
                for k in 1..=l {
                    product += a[i][k] * a[k][j];
                }
                for k in 1..=l {
                    a[k][j] -= product * a[k][i];
                }
            }
        }
        diag[i] = a[i][i];
        a[i][i] = 1.0;
        for j in 1..=l {
            a[j][i] = 0.0;
            a[i][j] = 0.0;
        }
    }
    let mut vectors = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            vectors[i][j] = a[i + 1][j + 1];
        }
    }
    (diag[1..=n].to_vec(), off[1..=n].to_vec(), vectors)
}

/// Implicit-shift QL for a symmetric tridiagonal matrix.
///
/// `diag` is overwritten with the eigenvalues. `off` uses the
/// Householder layout. `vectors` is multiplied on the right by the QL
/// factors, so a Householder factor in `vectors` becomes the
/// eigenvectors of the original matrix.
fn implicit_shift_ql(diag: &mut [f64], off_in: &mut [f64], vectors: &mut [Vec<f64>]) {
    let n = diag.len();
    if n <= 1 {
        return;
    }
    let mut d = vec![0.0; n + 1];
    let mut e = vec![0.0; n + 1];
    d[1..=n].copy_from_slice(diag);
    e[1..=n].copy_from_slice(off_in);
    for i in 2..=n {
        e[i - 1] = e[i];
    }
    e[n] = 0.0;
    for l in 1..=n {
        let mut iter = 0;
        loop {
            let mut m = l;
            while m < n {
                let scale = d[m].abs() + d[m + 1].abs();
                if e[m].abs() + scale == scale {
                    break;
                }
                m += 1;
            }
            if m == l {
                break;
            }
            iter += 1;
            if iter > 64 {
                break;
            }
            let shift = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let radius = (shift * shift + 1.0).sqrt();
            let denom = if shift >= 0.0 {
                shift + radius
            } else {
                shift - radius
            };
            let mut g = d[m] - d[l] + e[l] / denom;
            let mut s = 1.0;
            let mut c = 1.0;
            let mut p = 0.0;
            let mut underflow = false;
            let mut i = m;
            while i > l {
                i -= 1;
                let f = s * e[i];
                let b = c * e[i];
                let r = (f * f + g * g).sqrt();
                e[i + 1] = r;
                if r == 0.0 {
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    underflow = true;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                let rot = (d[i] - g) * s + 2.0 * c * b;
                p = s * rot;
                d[i + 1] = g + p;
                g = c * rot - b;
                let left = i - 1;
                let right = i;
                for row in vectors.iter_mut() {
                    let upper = row[right];
                    row[right] = s * row[left] + c * upper;
                    row[left] = c * row[left] - s * upper;
                }
            }
            if underflow {
                continue;
            }
            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }
    diag.copy_from_slice(&d[1..=n]);
}

/// Lowest eigenpair of the symmetric tridiagonal (alpha, beta), with
/// the eigenvector in the Lanczos basis.
fn lowest_ritz(alpha: &[f64], beta: &[f64]) -> (f64, Vec<f64>) {
    let k = alpha.len();
    let mut t = vec![vec![0.0; k]; k];
    for i in 0..k {
        t[i][i] = alpha[i];
        if i + 1 < k {
            t[i][i + 1] = beta[i];
            t[i + 1][i] = beta[i];
        }
    }
    let (evals, evecs) = jacobi_eigen(&mut t);
    let mut lowest = 0;
    for i in 1..k {
        if evals[i] < evals[lowest] {
            lowest = i;
        }
    }
    (evals[lowest], evecs.iter().map(|row| row[lowest]).collect())
}

/// Cyclic Jacobi for a small symmetric matrix. Returns eigenvalues
/// and the column-major eigenvector table `v[row][col]`.
// The rotation sweeps touch two columns of one row at a time, so an
// index loop is the readable form here.
#[allow(clippy::needless_range_loop)]
fn jacobi_eigen(a: &mut [Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut v = vec![vec![0.0; n]; n];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _sweep in 0..64 {
        let mut off = 0.0;
        for (i, row) in a.iter().enumerate() {
            for value in row.iter().skip(i + 1) {
                off += value * value;
            }
        }
        if off <= 1e-24 {
            break;
        }
        for p in 0..n {
            for qi in (p + 1)..n {
                if a[p][qi].abs() < 1e-18 {
                    continue;
                }
                let theta = (a[qi][qi] - a[p][p]) / (2.0 * a[p][qi]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let akp = a[k][p];
                    let akq = a[k][qi];
                    a[k][p] = c * akp - s * akq;
                    a[k][qi] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p][k];
                    let aqk = a[qi][k];
                    a[p][k] = c * apk - s * aqk;
                    a[qi][k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let vkp = v[k][p];
                    let vkq = v[k][qi];
                    v[k][p] = c * vkp - s * vkq;
                    v[k][qi] = s * vkp + c * vkq;
                }
            }
        }
    }
    let evals = (0..n).map(|i| a[i][i]).collect();
    (evals, v)
}

/// The surface's answer at one point.
#[derive(Clone)]
struct PointEval {
    x: Array1<f64>,
    /// `None` when the host supplied the gradient alone.
    energy: Option<f64>,
    gradient: Array1<f64>,
}

/// Effective gradient of a min-mode translation.
///
/// Henkelman and Jonsson, J. Chem. Phys. 111, 7010 (1999): while the
/// curvature along the unit mode is not negative, the step climbs that
/// mode alone. The effective gradient is minus the parallel part and
/// the perpendicular force is dropped. Once the curvature is negative
/// the perpendicular force stays and the parallel part is reflected,
/// `g - 2 (g · τ) τ`.
pub(crate) fn translated_gradient(
    gradient: &Array1<f64>,
    mode: &Array1<f64>,
    curvature: f64,
) -> Array1<f64> {
    let parallel = gradient.dot(mode);
    if curvature < 0.0 {
        gradient - &(mode * (2.0 * parallel))
    } else {
        mode * (-parallel)
    }
}

/// Stepping minimum-mode saddle search.
pub struct MinModeSession {
    config: MinModeConfig,
    force_gate: ForceGate,
    kappa: Option<KappaDimerConfig>,
    x: Array1<f64>,
    mode: Array1<f64>,
    curvature: f64,
    solver: Solver,
    iteration: usize,
    /// Gradient at `x`, when known: from the last translation's
    /// accepted point or from the host.
    centre: Option<PointEval>,
    /// Second-lowest Ritz vector from the previous Lanczos estimate,
    /// with `(theta_2 - theta_1) / |theta_1|`.
    recycle: Option<(Array1<f64>, f64)>,
    /// Residual test from the last Lanczos estimate. The dimer uses the
    /// rotation-cap test.
    mode_converged: bool,
    /// The first surface error raised inside the translation oracle
    /// during the current step; the rgmin oracle signature has no
    /// error channel of its own.
    surface_error: Mutex<Option<SaddleError>>,
}

impl MinModeSession {
    pub fn new(
        config: MinModeConfig,
        x: Array1<f64>,
        mode: Array1<f64>,
    ) -> Result<Self, SaddleError> {
        if x.len() != mode.len() || x.is_empty() {
            return Err(SaddleError::Shape(
                "position and mode must share a nonzero length".into(),
            ));
        }
        if !(config.dr.is_finite() && config.dr > 0.0) {
            return Err(SaddleError::Invalid(
                "finite-difference step must be positive and finite".into(),
            ));
        }
        let mode = checked_mode(mode)?;
        crate::band::check_force_driven(&config.method)?;
        let control = Control {
            maxiter: usize::MAX,
            gtol: 0.0,
            istep: 1.0,
            maxmove: None,
            ftol_rel: None,
        };
        let mut solver = Solver::new(config.method.clone(), control, x.len());
        // Per-atom cap over consecutive xyz triples (eOn
        // maxAtomMotionApplied). A non-positive max_move leaves the
        // step uncapped.
        solver.set_atom_maxmove(config.max_move);
        // The inverted force is not the gradient of the energy the
        // oracle reports, so a line search on that energy refuses good
        // steps and burns evaluations: L-BFGS takes its two-loop
        // direction with one oracle call (rgmin Accept::Step).
        if matches!(config.method, Method::Lbfgs { .. }) {
            solver.set_accept(Accept::Step);
        }
        Ok(Self {
            config,
            force_gate: ForceGate::LinfNorm,
            kappa: None,
            x,
            mode,
            curvature: f64::NAN,
            solver,
            iteration: 0,
            centre: None,
            recycle: None,
            mode_converged: false,
            surface_error: Mutex::new(None),
        })
    }

    /// Select the optional HiGHS feasible-set step; requires the `highs` feature.
    /// Disabled by default.
    pub fn set_highs(&mut self, enabled: bool) {
        self.solver.set_highs(enabled);
    }

    /// Select the convergence norm; the default is maximum absolute component.
    pub fn set_force_gate(&mut self, gate: ForceGate) {
        self.force_gate = gate;
    }

    /// Select the basin constraint without changing the rotation or translation method.
    pub fn set_kappa(&mut self, config: Option<KappaDimerConfig>) -> Result<(), SaddleError> {
        if let Some(c) = &config
            && (!c.beta.is_finite()
                || c.beta <= 0.0
                || !c.eigen.tol.is_finite()
                || c.eigen.tol <= 0.0)
        {
            return Err(SaddleError::Invalid(
                "kappa beta and tolerance must be positive and finite".into(),
            ));
        }
        self.kappa = config;
        self.reset();
        Ok(())
    }

    pub fn position(&self) -> ArrayView1<'_, f64> {
        self.x.view()
    }

    pub fn mode(&self) -> ArrayView1<'_, f64> {
        self.mode.view()
    }

    /// Curvature along [`MinModeSession::mode`] from the last
    /// estimate; NaN before the first.
    pub fn curvature(&self) -> f64 {
        self.curvature
    }

    /// Drop optimizer history and the cached centre gradient at a
    /// surface-epoch boundary. The mode stays as the next seed.
    pub fn reset(&mut self) {
        self.solver.forget();
        self.centre = None;
    }

    /// Move to `x` and keep the session: the mode stays as the next
    /// rotation's seed (a host stepping along a saddle search reuses
    /// it, so a mode that still holds costs one rotation evaluation),
    /// and the optimizer history stays. `gradient` is the host's
    /// energy gradient at `x` when it has one; the next compute then
    /// skips the centre evaluation.
    pub fn set_position(
        &mut self,
        x: Array1<f64>,
        gradient: Option<Array1<f64>>,
    ) -> Result<(), SaddleError> {
        if x.len() != self.x.len() {
            return Err(SaddleError::Shape("position length must stay fixed".into()));
        }
        if !x.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("position"));
        }
        self.centre = match gradient {
            Some(gradient) => {
                if gradient.len() != x.len() {
                    return Err(SaddleError::Shape(
                        "gradient must match the position".into(),
                    ));
                }
                if !gradient.iter().all(|v| v.is_finite()) {
                    return Err(SaddleError::NonFinite("min-mode gradient"));
                }
                Some(PointEval {
                    x: x.clone(),
                    energy: None,
                    gradient,
                })
            }
            None => None,
        };
        self.x = x;
        Ok(())
    }

    /// Replace the mode seed with a host-chosen direction.
    pub fn set_mode(&mut self, mode: Array1<f64>) -> Result<(), SaddleError> {
        if mode.len() != self.x.len() {
            return Err(SaddleError::Shape("mode must match the position".into()));
        }
        let mode = checked_mode(mode)?;
        // The solver's cached force at this point was inverted along the
        // old mode; a large turn also retires the stored pairs.
        let turned = self.mode.dot(&mode).abs().min(1.0).acos();
        if turned > MODE_RESET_ANGLE {
            self.solver.forget();
        } else {
            self.solver.forget_evaluation();
        }
        self.mode = mode;
        Ok(())
    }

    fn centre_gradient<S: PointSurface>(
        &mut self,
        surface: &Counted<'_, S>,
    ) -> Result<Array1<f64>, SaddleError> {
        if let Some(c) = &self.centre
            && c.x == self.x
        {
            return Ok(c.gradient.clone());
        }
        let (energy, gradient) = surface.eval(self.x.view())?;
        self.centre = Some(PointEval {
            x: self.x.clone(),
            energy: Some(energy),
            gradient: gradient.clone(),
        });
        Ok(gradient)
    }

    fn mode_at<S: PointSurface>(
        &self,
        surface: &Counted<'_, S>,
        position: ArrayView1<f64>,
        gradient: ArrayView1<f64>,
    ) -> Result<ModeSample, SaddleError> {
        let excluded = surface.surface.excluded_modes(position)?;
        if excluded.ncols() != position.len() || !excluded.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::Shape(
                "minimum-mode excluded directions".into(),
            ));
        }
        let recycle = self
            .recycle
            .as_ref()
            .and_then(|(vector, gap)| (*gap < self.config.lanczos_recycle_gap).then_some(vector));
        if excluded.nrows() == 0 {
            return self.estimate_in(surface, position, gradient, self.mode.clone(), recycle);
        }
        let mut space = Complement::new(position.len());
        for direction in excluded.rows() {
            space.exclude(direction);
        }
        if space.dimension() == 0 {
            return Err(SaddleError::Shape("minimum-mode space is empty".into()));
        }
        let mut seed = space.reduce(self.mode.view());
        if seed.dot(&seed).sqrt() <= 64.0 * f64::EPSILON {
            seed = Array1::from_iter(
                (0..space.dimension()).map(|i| ((i + 1) as f64 * 1.618033988749895).sin()),
            );
        }
        let reduced_recycle = recycle.map(|vector| space.reduce(vector.view()));
        let reduced = Counted {
            surface: surface.surface,
            calls: surface.calls,
            space: Some(&space),
        };
        let (mode, curvature, rotations, converged, second) = self.estimate_in(
            &reduced,
            position,
            gradient,
            checked_mode(seed)?,
            reduced_recycle.as_ref(),
        )?;
        let second = second.map(|(vector, gap)| (space.lift(vector.view()), gap));
        Ok((
            space.lift(mode.view()),
            curvature,
            rotations,
            converged,
            second,
        ))
    }

    fn estimate_in<S: PointSurface>(
        &self,
        surface: &Counted<'_, S>,
        position: ArrayView1<f64>,
        gradient: ArrayView1<f64>,
        seed: Array1<f64>,
        recycle: Option<&Array1<f64>>,
    ) -> Result<ModeSample, SaddleError> {
        match self.config.kind {
            MinModeKind::Dimer => {
                let (mode, curvature, rotations) =
                    rotate_dimer(surface, position, gradient, seed, &self.config)?;
                Ok((
                    mode,
                    curvature,
                    rotations,
                    rotations < self.config.max_rotations,
                    None,
                ))
            }
            MinModeKind::Lanczos => {
                let out = lanczos_mode(surface, position, gradient, seed, &self.config, recycle)?;
                Ok((
                    out.mode,
                    out.curvature,
                    out.actions,
                    out.converged,
                    out.second,
                ))
            }
        }
    }

    fn refresh_mode<S: PointSurface>(
        &mut self,
        surface: &Counted<'_, S>,
        g0: ArrayView1<f64>,
    ) -> Result<(f64, usize), SaddleError> {
        let (mode, curvature, rotations, converged, second) =
            self.mode_at(surface, self.x.view(), g0)?;
        self.mode = mode;
        self.curvature = curvature;
        self.mode_converged = converged;
        self.recycle = if self.config.lanczos_restart_keep > 0 {
            second
        } else {
            None
        };
        Ok((curvature, rotations))
    }

    /// Refresh the lowest mode at the current point without moving.
    /// The centre gradient comes from the cache, from
    /// [`MinModeSession::set_position`], or from one evaluation.
    pub fn estimate_mode<S: PointSurface>(
        &mut self,
        surface: &S,
    ) -> Result<ModeEstimate, SaddleError> {
        let calls = AtomicUsize::new(0);
        let counted = Counted {
            surface,
            calls: &calls,
            space: None,
        };
        // A central difference never reads the centre gradient, so a
        // rotation-only estimate does not evaluate it.
        let centre_known = self.centre.as_ref().is_some_and(|c| c.x == self.x);
        let g0 = if self.config.difference == FiniteDifference::Central && !centre_known {
            Array1::zeros(self.x.len())
        } else {
            self.centre_gradient(&counted)?
        };
        let (curvature, rotations) = self.refresh_mode(&counted, g0.view())?;
        Ok(ModeEstimate {
            mode: self.mode.clone(),
            curvature,
            rotations,
            evaluations: calls.load(Ordering::Relaxed),
            converged: self.mode_converged,
        })
    }

    /// One min-mode step: refresh the lowest mode, invert the force
    /// along it, take one solver step.
    ///
    /// A surface error inside the step returns `Err` with the position
    /// unchanged. The solver's internal state has seen the failed
    /// evaluation; a host that recovers the surface and steps again
    /// may call [`MinModeSession::reset`] first to drop it.
    pub fn step<S: PointSurface>(&mut self, surface: &S) -> Result<MinModeReport, SaddleError> {
        let calls = AtomicUsize::new(0);
        let counted = Counted {
            surface,
            calls: &calls,
            space: None,
        };
        let g0 = self.centre_gradient(&counted)?;
        let previous_mode = self.mode.clone();
        let (mut curvature, mut rotations) = self.refresh_mode(&counted, g0.view())?;
        // The solver cached the inverted force at this point under the
        // previous mode; the step must see the current one. The mode's
        // sign is arbitrary, so the angle reads |cos|.
        let turned = previous_mode.dot(&self.mode).abs().min(1.0).acos();
        if turned > MODE_RESET_ANGLE {
            self.solver.forget();
        } else {
            self.solver.forget_evaluation();
        }

        let max_force = self.force_gate.value(g0.view());
        if converged(max_force, curvature, self.config.force_tol) {
            return Ok(MinModeReport {
                status: MinModeStatus::Converged,
                max_force,
                curvature,
                rotations,
                iteration: self.iteration,
                evaluations: calls.load(Ordering::Relaxed),
            });
        }

        // Henkelman and Jonsson 1999: climb the mode alone while its
        // curvature is not negative, and reflect the parallel part
        // once the curvature is negative. The mode estimate for this
        // step is the one just refreshed; Lanczos and the dimer share it.
        let tau = self.mode.clone();
        let translation_curvature = curvature;
        let kappa = self.kappa.as_ref();
        let config = &self.config;
        let error_slot = &self.surface_error;
        if let Ok(mut slot) = error_slot.lock() {
            slot.take();
        }
        let last: Mutex<Option<PointEval>> = Mutex::new(None);
        // The solver re-reads the start under the new mode (its cached
        // force was inverted along the old one); the centre answers it
        // without a surface call when its energy is known.
        let start = self.x.clone();
        let start_energy = self
            .centre
            .as_ref()
            .filter(|c| c.x == start)
            .and_then(|c| c.energy);
        let oracle = Oracle::unbounded(self.x.len(), |xv: ArrayView1<f64>| {
            let answer = match start_energy {
                Some(e) if xv == start => Ok((e, g0.clone())),
                _ => counted.eval(xv),
            };
            let answer = answer.and_then(|(e, g)| {
                let eff = if let Some(kappa) = kappa {
                    let excluded = surface.excluded_modes(xv)?;
                    let out = kappa_dimer_force(
                        g.view(),
                        tau.view(),
                        excluded.view(),
                        |v| {
                            let magnitude = v.dot(&v).sqrt();
                            if magnitude == 0.0 {
                                return Ok(Array1::zeros(v.len()));
                            }
                            let unit = &v / magnitude;
                            Ok(counted.action(xv, g.view(), unit.view(), config)? * magnitude)
                        },
                        kappa,
                    )?;
                    -out.force
                } else {
                    translated_gradient(&g, &tau, translation_curvature)
                };
                Ok((e, g, eff))
            });
            match answer {
                Ok((e, g, eff)) => {
                    if let Ok(mut slot) = last.lock() {
                        *slot = Some(PointEval {
                            x: xv.to_owned(),
                            energy: Some(e),
                            gradient: g,
                        });
                    }
                    (e, eff)
                }
                Err(err) => {
                    if let Ok(mut slot) = error_slot.lock()
                        && slot.is_none()
                    {
                        *slot = Some(err);
                    }
                    (f64::INFINITY, Array1::zeros(xv.len()))
                }
            }
        });

        let mut x = self.x.clone();
        let stepped = self.solver.step(&oracle, &mut x);
        drop(oracle);
        // The surface error outranks whatever the solver made of an
        // infinite energy and a zero gradient. `x` is a local copy;
        // the session's position stays where it was.
        if let Some(err) = self.surface_error.lock().ok().and_then(|mut s| s.take()) {
            return Err(err);
        }
        stepped.map_err(|e| SaddleError::Solver(e.to_string()))?;
        // The oracle's last evaluation is the accepted point for every
        // stepper that ends on its trial (FIRE, Accept::Step); it is
        // this report's force and the next step's centre.
        let at_new = last.into_inner().ok().flatten().filter(|p| p.x == x);
        let centre = match at_new {
            Some(p) => p,
            None => {
                let (energy, gradient) = counted.eval(x.view())?;
                PointEval {
                    x: x.clone(),
                    energy: Some(energy),
                    gradient,
                }
            }
        };
        let max_force = self.force_gate.value(centre.gradient.view());
        if max_force <= self.config.force_tol && x != self.x {
            // Force and curvature in a convergence decision describe the
            // same point. The accepted gradient supplies the centre, so
            // confirmation pays only for the required Hessian actions.
            let (mode, accepted_curvature, accepted_rotations, accepted_converged, second) =
                self.mode_at(&counted, x.view(), centre.gradient.view())?;
            self.mode_converged = accepted_converged;
            self.recycle = if self.config.lanczos_restart_keep > 0 {
                second
            } else {
                None
            };
            let turned = self.mode.dot(&mode).abs().min(1.0).acos();
            if turned > MODE_RESET_ANGLE {
                self.solver.forget();
            } else {
                self.solver.forget_evaluation();
            }
            self.mode = mode;
            self.curvature = accepted_curvature;
            curvature = accepted_curvature;
            rotations += accepted_rotations;
        }
        self.x = x;
        self.iteration += 1;
        self.centre = Some(centre);
        let status = if converged(max_force, curvature, self.config.force_tol) {
            MinModeStatus::Converged
        } else {
            MinModeStatus::Running
        };
        Ok(MinModeReport {
            status,
            max_force,
            curvature,
            rotations,
            iteration: self.iteration,
            evaluations: calls.load(Ordering::Relaxed),
        })
    }

    /// Convenience loop over [`MinModeSession::step`].
    pub fn run<S: PointSurface>(
        &mut self,
        surface: &S,
        max_steps: usize,
    ) -> Result<MinModeReport, SaddleError> {
        let mut report = self.step(surface)?;
        while report.status == MinModeStatus::Running && self.iteration < max_steps {
            report = self.step(surface)?;
        }
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::translated_gradient;
    use approx::assert_abs_diff_eq;
    use ndarray::{Array1, array};

    #[test]
    fn positive_curvature_keeps_only_the_reversed_mode() {
        let gradient = array![1.0, -4.0, 0.5];
        let mode = array![1.0, 0.0, 0.0];
        let eff = translated_gradient(&gradient, &mode, 0.25);
        assert_abs_diff_eq!(eff[0], -1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(eff[1], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(eff[2], 0.0, epsilon = 1e-12);
        let flat = translated_gradient(&gradient, &mode, 0.0);
        assert_abs_diff_eq!(flat[1], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn negative_curvature_reflects_the_parallel_part() {
        let gradient = array![1.0, -4.0, 0.5];
        let mode = array![1.0, 0.0, 0.0];
        let eff = translated_gradient(&gradient, &mode, -1.5);
        assert_abs_diff_eq!(eff[0], -1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(eff[1], -4.0, epsilon = 1e-12);
        assert_abs_diff_eq!(eff[2], 0.5, epsilon = 1e-12);
    }

    #[test]
    fn projected_eigen_residuals_stay_small() {
        let samples = [
            vec![vec![2.0, 0.0], vec![0.0, -1.0]],
            vec![vec![1.0, 2.0], vec![2.0, 1.0]],
            vec![
                vec![4.0, 0.3, 0.0, 0.0],
                vec![0.3, -1.0, 0.2, 0.0],
                vec![0.0, 0.2, 0.5, 0.1],
                vec![0.0, 0.0, 0.1, 3.0],
            ],
        ];
        for matrix in samples {
            let (evals, evecs) = super::projected_eigen(&matrix);
            let n = matrix.len();
            assert_eq!(evals.len(), n);
            for col in 0..n {
                let mut residual = 0.0;
                for row in 0..n {
                    let mut product = 0.0;
                    for k in 0..n {
                        product += matrix[row][k] * evecs[k][col];
                    }
                    let diff = product - evals[col] * evecs[row][col];
                    residual += diff * diff;
                }
                assert!(
                    residual.sqrt() < 1e-8,
                    "col {col} residual {} evals {evals:?}",
                    residual.sqrt()
                );
            }
        }
    }

    #[test]
    fn reorthogonalization_matches_at_one_and_eight_threads() {
        let n = 70_000usize;
        let mut w = Array1::zeros(n);
        for (i, value) in w.iter_mut().enumerate() {
            *value = ((i + 1) as f64 * 0.017).sin();
        }
        let mut basis = Vec::new();
        for col in 0..3 {
            let mut q = Array1::zeros(n);
            for (i, value) in q.iter_mut().enumerate() {
                *value = ((i + 3) as f64 * 0.013 * (col + 1) as f64).sin();
            }
            basis.push(super::normalize(q));
        }
        let mut digests = Vec::new();
        for threads in [1usize, 2, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let digest = pool.install(|| {
                let out = super::reorthogonalize(&basis, w.clone());
                out.iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<u8>>()
            });
            digests.push(digest);
        }
        assert_eq!(digests[0], digests[1]);
        assert_eq!(digests[1], digests[2]);
    }

    /// Reorthogonalization at n = 3e5. The working set is the basis plus
    /// one vector. Modified Gram-Schmidt walks columns in order. Each
    /// dot and update is chunked, so the limit is memory bandwidth.
    #[test]
    #[ignore = "times reorthogonalization at n = 3e5"]
    fn reorthogonalization_scales_with_the_basis() {
        let n = 300_000usize;
        let mut w = Array1::from_elem(n, 0.0);
        for (i, value) in w.iter_mut().enumerate() {
            *value = ((i + 1) as f64 * 0.017).sin();
        }
        let pools: Vec<_> = [1usize, 2, 4, 8]
            .into_iter()
            .map(|threads| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap()
            })
            .collect();
        for k in [12usize, 20, 30] {
            let mut basis = Vec::with_capacity(k);
            for col in 0..k {
                let mut q = Array1::from_elem(n, 0.0);
                for (i, value) in q.iter_mut().enumerate() {
                    *value = ((i + 3) as f64 * 0.013 * (col + 1) as f64).sin();
                }
                basis.push(super::normalize(q));
            }
            let mut samples = vec![Vec::new(); pools.len()];
            for _ in 0..5 {
                for (slot, pool) in pools.iter().enumerate() {
                    let seconds = pool.install(|| {
                        let start = std::time::Instant::now();
                        let _ = super::reorthogonalize(&basis, w.clone());
                        start.elapsed().as_secs_f64()
                    });
                    samples[slot].push(seconds);
                }
            }
            let mut medians = Vec::new();
            for (slot, threads) in [1usize, 2, 4, 8].into_iter().enumerate() {
                samples[slot].sort_by(|a, b| a.total_cmp(b));
                let median = samples[slot][2];
                let range = samples[slot][4] - samples[slot][0];
                medians.push(median);
                let bytes = (k + 1) as f64 * n as f64 * 8.0 * 2.0;
                let gib_s = bytes / median / 1e9;
                eprintln!(
                    "k={k} threads={threads} median_s={median:.6} range_s={range:.6} GB/s={gib_s:.2}"
                );
            }
            let t1 = medians[0];
            for (slot, threads) in [2usize, 4, 8].into_iter().enumerate() {
                let efficiency = t1 / (threads as f64 * medians[slot + 1]);
                eprintln!("k={k} threads={threads} efficiency={efficiency:.3}");
            }
        }
    }
}
