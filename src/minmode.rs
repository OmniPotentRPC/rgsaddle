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

/// Lanczos: build a Krylov basis of finite-difference Hessian actions
/// and take the lowest Ritz vector. Stops once the Ritz residual
/// `beta_j |s_j|` (the norm of `H y - theta y` for the lowest Ritz
/// pair `(theta, y)` in exact arithmetic, Paige 1971) meets
/// `rotation_tol`, so a seed that already is the mode costs one action.
fn lanczos_mode<S: PointSurface>(
    surface: &Counted<'_, S>,
    x: ArrayView1<f64>,
    g0: ArrayView1<f64>,
    seed: Array1<f64>,
    config: &MinModeConfig,
) -> Result<(Array1<f64>, f64, usize), SaddleError> {
    let n = seed.len();
    let m = config.krylov_dim.min(n).max(1);
    let mut q: Vec<Array1<f64>> = Vec::with_capacity(m);
    let mut alpha: Vec<f64> = Vec::with_capacity(m);
    let mut beta: Vec<f64> = Vec::with_capacity(m);
    q.push(normalize(seed));

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
        // Full reorthogonalization: the Krylov dimension is small and
        // finite-difference actions are noisy.
        for qi in q.iter() {
            let overlap = w.dot(qi);
            w = &w - &(qi * overlap);
        }
        let b = w.dot(&w).sqrt();
        let (theta, s) = lowest_ritz(&alpha, &beta);
        let residual = b * s[j].abs();
        if residual <= config.rotation_tol || alpha.len() == m || b <= 1e-12 {
            let mut mode = Array1::zeros(n);
            for (qi, si) in q.iter().zip(s.iter()) {
                mode = &mode + &(qi * *si);
            }
            return Ok((normalize(mode), theta, actions));
        }
        beta.push(b);
        q.push(w / b);
    }
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
    ) -> Result<(Array1<f64>, f64, usize), SaddleError> {
        let excluded = surface.surface.excluded_modes(position)?;
        if excluded.ncols() != position.len() || !excluded.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::Shape(
                "minimum-mode excluded directions".into(),
            ));
        }
        let estimate = |surface: &Counted<'_, S>, seed| match self.config.kind {
            MinModeKind::Dimer => rotate_dimer(surface, position, gradient, seed, &self.config),
            MinModeKind::Lanczos => lanczos_mode(surface, position, gradient, seed, &self.config),
        };
        if excluded.nrows() == 0 {
            return estimate(surface, self.mode.clone());
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
        let reduced = Counted {
            surface: surface.surface,
            calls: surface.calls,
            space: Some(&space),
        };
        let (mode, curvature, rotations) = estimate(&reduced, checked_mode(seed)?)?;
        Ok((space.lift(mode.view()), curvature, rotations))
    }

    fn refresh_mode<S: PointSurface>(
        &mut self,
        surface: &Counted<'_, S>,
        g0: ArrayView1<f64>,
    ) -> Result<(f64, usize), SaddleError> {
        let (mode, curvature, rotations) = self.mode_at(surface, self.x.view(), g0)?;
        self.mode = mode;
        self.curvature = curvature;
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

        // Effective gradient: invert the component along the lowest
        // mode so the step climbs that direction and descends the
        // rest. Below a negative curvature this is the min-mode
        // force; above it, pure inversion still points off the ridge.
        let tau = self.mode.clone();
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
                    let par = g.dot(&tau);
                    &g - &(&tau * (2.0 * par))
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
            let (mode, accepted_curvature, accepted_rotations) =
                self.mode_at(&counted, x.view(), centre.gradient.view())?;
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
