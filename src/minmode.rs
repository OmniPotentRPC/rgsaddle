//! Minimum-mode saddle search: find the lowest curvature direction,
//! invert the force along it, take one step. Stepping, like the band.

use ndarray::{Array1, ArrayView1};
use rgmin::{Control, Method, Oracle, Solver};

use crate::error::SaddleError;

/// The caller's surface for a single geometry.
pub trait PointSurface: Sync {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError>;
}

/// How the lowest mode is estimated.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MinModeKind {
    /// Finite-difference dimer rotation (Henkelman-Jonsson 1999 with
    /// the Heyden 2005 rotation plane): one extra gradient per
    /// rotation iteration.
    #[default]
    Dimer,
    /// Lanczos on finite-difference Hessian actions: a Krylov
    /// subspace per compute, one gradient per action.
    Lanczos,
}

#[derive(Clone, Debug)]
pub struct MinModeConfig {
    pub kind: MinModeKind,
    /// Dimer separation / finite-difference displacement, Angstrom.
    pub dr: f64,
    /// Rotation stops when the rotational force falls under this.
    pub rotation_tol: f64,
    pub max_rotations: usize,
    /// Krylov dimension for [`MinModeKind::Lanczos`].
    pub krylov_dim: usize,
    /// Translation stops when max|F| falls under this.
    pub force_tol: f64,
    pub max_move: f64,
    pub method: Method,
}

impl Default for MinModeConfig {
    fn default() -> Self {
        Self {
            kind: MinModeKind::Dimer,
            dr: 1e-3,
            rotation_tol: 1e-4,
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
    pub rotations: usize,
    pub iteration: usize,
}

fn normalize(mut v: Array1<f64>) -> Array1<f64> {
    let n = v.dot(&v).sqrt();
    if n > 1e-14 {
        v /= n;
    }
    v
}

/// Finite-difference Hessian action `H v` at `x` with the forward
/// difference of gradients: one surface call per action.
fn hessian_action<S: PointSurface>(
    surface: &S,
    x: ArrayView1<f64>,
    g0: ArrayView1<f64>,
    v: ArrayView1<f64>,
    dr: f64,
) -> Result<Array1<f64>, SaddleError> {
    let unit = normalize(v.to_owned());
    let shifted = &x + &(&unit * dr);
    let (_, g1) = surface.eval(shifted.view())?;
    Ok((&g1 - &g0) / dr)
}

/// Rayleigh quotient of the finite-difference Hessian along `v`.
fn curvature_along(hv: ArrayView1<f64>, v: ArrayView1<f64>) -> f64 {
    hv.dot(&v)
}

/// Dimer rotation: steepest-descent rotation of the mode toward the
/// lowest curvature, stopping on the rotational force.
fn rotate_dimer<S: PointSurface>(
    surface: &S,
    x: ArrayView1<f64>,
    g0: ArrayView1<f64>,
    mode: Array1<f64>,
    config: &MinModeConfig,
) -> Result<(Array1<f64>, f64, usize), SaddleError> {
    let mut tau = normalize(mode);
    let mut rotations = 0;
    let mut curvature;
    loop {
        let hv = hessian_action(surface, x, g0, tau.view(), config.dr)?;
        curvature = curvature_along(hv.view(), tau.view());
        // Rotational force: the component of H tau perpendicular to
        // tau, which vanishes exactly at an eigenvector.
        let perp = &hv - &(&tau * curvature);
        let perp_norm = perp.dot(&perp).sqrt();
        if perp_norm <= config.rotation_tol || rotations >= config.max_rotations {
            break;
        }
        // Rotate against the perpendicular curvature component. The
        // step is the normalized perpendicular direction scaled by a
        // Rayleigh-based trust factor; renormalizing keeps tau a unit
        // vector without a line search.
        let theta = normalize(perp);
        let scale = (perp_norm / (curvature.abs() + perp_norm)).min(0.5);
        tau = normalize(&tau - &(&theta * scale));
        rotations += 1;
    }
    Ok((tau, curvature, rotations))
}

/// Lanczos: build a Krylov basis of finite-difference Hessian actions
/// and take the lowest Ritz vector.
fn lanczos_mode<S: PointSurface>(
    surface: &S,
    x: ArrayView1<f64>,
    g0: ArrayView1<f64>,
    seed: Array1<f64>,
    config: &MinModeConfig,
) -> Result<(Array1<f64>, f64, usize), SaddleError> {
    let n = seed.len();
    let m = config.krylov_dim.min(n).max(1);
    let mut q: Vec<Array1<f64>> = Vec::with_capacity(m);
    let mut alpha = Vec::with_capacity(m);
    let mut beta: Vec<f64> = Vec::with_capacity(m);
    q.push(normalize(seed));

    let mut actions = 0;
    for j in 0..m {
        let hv = hessian_action(surface, x, g0, q[j].view(), config.dr)?;
        actions += 1;
        let a = hv.dot(&q[j]);
        alpha.push(a);
        if j + 1 == m {
            break;
        }
        let mut w = &hv - &(&q[j] * a);
        if j > 0 {
            let b = beta[j - 1];
            w = &w - &(&q[j - 1] * b);
        }
        // Full reorthogonalization: the Krylov dimension is small and
        // finite-difference actions are noisy.
        for qi in q.iter() {
            let overlap = w.dot(qi);
            w = &w - &(qi * overlap);
        }
        let b = w.dot(&w).sqrt();
        if b <= 1e-12 {
            break;
        }
        beta.push(b);
        q.push(w / b);
    }

    // Lowest eigenpair of the symmetric tridiagonal (alpha, beta) by
    // inverse-free bisection-free QL: the dimension is tiny, so a
    // dense Jacobi sweep on the tridiagonal is cheapest to get right.
    let k = alpha.len();
    let mut t = vec![vec![0.0; k]; k];
    for i in 0..k {
        t[i][i] = alpha[i];
        if i + 1 < k && i < beta.len() {
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
    let mut mode = Array1::zeros(n);
    for (i, qi) in q.iter().enumerate().take(k) {
        mode = &mode + &(qi * evecs[i][lowest]);
    }
    Ok((normalize(mode), evals[lowest], actions))
}

/// Cyclic Jacobi for a small symmetric matrix. Returns eigenvalues
/// and the column-major eigenvector table `v[row][col]`.
fn jacobi_eigen(a: &mut [Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut v = vec![vec![0.0; n]; n];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _sweep in 0..64 {
        let mut off = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                off += a[i][j] * a[i][j];
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

/// Stepping minimum-mode saddle search.
pub struct MinModeSession {
    config: MinModeConfig,
    x: Array1<f64>,
    mode: Array1<f64>,
    solver: Solver,
    iteration: usize,
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
        let control = Control {
            maxiter: usize::MAX,
            gtol: 0.0,
            istep: 1.0,
            maxmove: Some(config.max_move),
        };
        let solver = Solver::new(config.method.clone(), control, x.len());
        Ok(Self {
            config,
            x,
            mode: normalize(mode),
            solver,
            iteration: 0,
        })
    }

    pub fn position(&self) -> ArrayView1<'_, f64> {
        self.x.view()
    }

    pub fn mode(&self) -> ArrayView1<'_, f64> {
        self.mode.view()
    }

    /// Drop optimizer history at a surface-epoch boundary.
    pub fn reset(&mut self) {
        self.solver.forget();
    }

    /// One min-mode step: refresh the lowest mode, invert the force
    /// along it, take one solver step.
    pub fn step<S: PointSurface>(&mut self, surface: &S) -> Result<MinModeReport, SaddleError> {
        let (_, g0) = surface.eval(self.x.view())?;
        if !g0.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("min-mode gradient"));
        }
        let (mode, curvature, rotations) = match self.config.kind {
            MinModeKind::Dimer => rotate_dimer(
                surface,
                self.x.view(),
                g0.view(),
                self.mode.clone(),
                &self.config,
            )?,
            MinModeKind::Lanczos => lanczos_mode(
                surface,
                self.x.view(),
                g0.view(),
                self.mode.clone(),
                &self.config,
            )?,
        };
        self.mode = mode;

        let force = -&g0;
        let max_force = force.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        if max_force <= self.config.force_tol {
            return Ok(MinModeReport {
                status: MinModeStatus::Converged,
                max_force,
                curvature,
                rotations,
                iteration: self.iteration,
            });
        }

        // Effective gradient: invert the component along the lowest
        // mode so the step climbs that direction and descends the
        // rest. Below a negative curvature this is the min-mode
        // force; above it, pure inversion still points off the ridge.
        let tau = self.mode.clone();
        let config_dr = self.config.dr;
        let kind = self.config.kind;
        let _ = (config_dr, kind);
        let oracle = Oracle::unbounded(self.x.len(), move |xv: ArrayView1<f64>| {
            match surface.eval(xv) {
                Ok((e, g)) => {
                    let par = g.dot(&tau);
                    let eff = &g - &(&tau * (2.0 * par));
                    (e, eff)
                }
                Err(_) => (f64::INFINITY, Array1::zeros(xv.len())),
            }
        });

        let mut x = self.x.clone();
        self.solver
            .step(&oracle, &mut x)
            .map_err(|e| SaddleError::Solver(e.to_string()))?;
        drop(oracle);
        self.x = x;
        self.iteration += 1;

        let (_, g_new) = surface.eval(self.x.view())?;
        let max_force = g_new.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let status = if max_force <= self.config.force_tol {
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
