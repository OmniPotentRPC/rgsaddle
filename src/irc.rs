//! Intrinsic reaction coordinate session: roll from a first-order
//! saddle to a minimum, one branch at a time.
//!
//! Inner geometry is rgmin `IrcTrust` (Sella IRCTrustRegion /
//! Gonzalez--Schlegel MW sphere) on `ManifoldKind::MwRigid`. The
//! increment is rgmin `qn_irc_restricted` (Sella QuasiNewtonIRC)
//! with a persistent MW BFGS Hessian. Ambient algebra goes through
//! `rgmin::vecops`. The host owns the loop. `run` is a convenience
//! over `step`.

use ndarray::{Array1, ArrayView1};
use rgmin::vecops::{axpy, dot, nrm2, nrminf, Vector};
use rgmin::{
    mw_pair, qn_irc_restricted, sqrt_masses_3n, BfgsModel, Control, EigensolverKind, IrcTrust,
    ManifoldKind, Method, Solver,
};

use crate::error::SaddleError;
use crate::minmode::PointSurface;

/// Which side of the imaginary mode to follow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IrcDirection {
    Forward,
    Reverse,
}

/// Outer IRC controls.
#[derive(Clone, Debug)]
pub struct IrcConfig {
    /// Mass-weighted sphere radius (Sella `dx`).
    pub dx: f64,
    pub force_tol: f64,
    pub max_move: f64,
    pub method: Method,
    pub max_inner: usize,
    /// Finite-difference Hessian action length for the matrix-free kick.
    pub dr: f64,
    /// Krylov dimension for the lowest-mode kick.
    pub krylov_dim: usize,
    /// Closed lowest-mode backend. Unlinked kinds fail closed.
    pub eigen_kind: EigensolverKind,
}

impl Default for IrcConfig {
    fn default() -> Self {
        Self {
            dx: 0.1,
            force_tol: 0.05,
            max_move: 0.2,
            method: Method::Steepest,
            max_inner: 10,
            dr: 1e-3,
            krylov_dim: 12,
            eigen_kind: EigensolverKind::Lanczos,
        }
    }
}

/// One accepted IRC point.
#[derive(Clone, Debug)]
pub struct IrcReport {
    pub energy: f64,
    pub max_force: f64,
    pub arc: f64,
    pub inner_steps: usize,
    pub at_minimum: bool,
}

/// Forward or reverse roll-down over an rgmin `Solver`.
pub struct IrcSession {
    config: IrcConfig,
    x: Array1<f64>,
    x_saddle: Array1<f64>,
    masses: Array1<f64>,
    d1: Array1<f64>,
    mode: Array1<f64>,
    solver: Solver,
    hess: BfgsModel,
    first: bool,
    arc: f64,
    last_step: Option<Array1<f64>>,
    last_outer: Option<Array1<f64>>,
}

impl IrcSession {
    /// `mode` is the Cartesian imaginary-mode direction (not yet scaled).
    pub fn new(
        config: IrcConfig,
        saddle: Array1<f64>,
        masses: Array1<f64>,
        mode: Array1<f64>,
        direction: IrcDirection,
    ) -> Result<Self, SaddleError> {
        let n3 = saddle.len();
        if n3 == 0 || n3 % 3 != 0 {
            return Err(SaddleError::Shape(
                "saddle must be a nonzero 3N Cartesian".into(),
            ));
        }
        if masses.len() * 3 != n3 || mode.len() != n3 {
            return Err(SaddleError::Shape(
                "masses (N) and mode (3N) must match the saddle".into(),
            ));
        }
        let control = Control {
            maxiter: usize::MAX,
            gtol: 0.0,
            istep: 1.0,
            maxmove: Some(config.max_move),
        };
        let mut solver = Solver::new(config.method.clone(), control, n3);
        // SE(3) quotient needs leftover internal modes: N >= 3.
        if masses.len() >= 3 {
            solver.set_manifold(ManifoldKind::MwRigid);
            solver.set_masses(masses.clone());
        }
        let mut session = Self {
            config,
            x: saddle.clone(),
            x_saddle: saddle,
            masses,
            d1: Array1::zeros(n3),
            mode,
            solver,
            hess: BfgsModel::identity(n3),
            first: true,
            arc: 0.0,
            last_step: None,
            last_outer: None,
        };
        session.set_direction(direction);
        Ok(session)
    }

    /// Estimate the imaginary mode with matrix-free Lanczos (FD Hessian
    /// actions). This is the ChASE / PRIMME / Sella lesson: IRC needs
    /// one extremal pair, not a full ELPA / SLATE heev of H.
    pub fn from_surface<S: crate::minmode::PointSurface>(
        config: IrcConfig,
        saddle: Array1<f64>,
        masses: Array1<f64>,
        seed: Array1<f64>,
        direction: IrcDirection,
        surface: &S,
    ) -> Result<Self, SaddleError> {
        let (_, g0) = surface.eval(saddle.view())?;
        let mm = crate::minmode::MinModeConfig {
            dr: config.dr,
            krylov_dim: config.krylov_dim,
            eigen_kind: config.eigen_kind,
            ..crate::minmode::MinModeConfig::default()
        };
        let (mode, _curv, _actions) =
            crate::minmode::lanczos_mode(surface, saddle.view(), g0.view(), seed, &mm)?;
        Self::new(config, saddle, masses, mode, direction)
    }

    pub fn position(&self) -> ArrayView1<'_, f64> {
        self.x.view()
    }

    pub fn reset(&mut self) {
        self.solver.forget();
        self.hess.forget();
    }

    /// Restore the saddle and flip the kick sign.
    pub fn set_direction(&mut self, direction: IrcDirection) {
        self.x = self.x_saddle.clone();
        self.d1 = self.kick_vector(direction);
        self.first = true;
        self.arc = 0.0;
        self.last_step = None;
        self.last_outer = None;
        self.solver.forget();
        self.hess.forget();
    }

    fn kick_vector(&self, direction: IrcDirection) -> Array1<f64> {
        let sqrtm = sqrt_masses_3n(self.masses.as_slice().unwrap_or(&[]));
        let mut mw = Vector::from_host(self.mode.clone());
        for i in 0..mw.host_view().len() {
            mw.host_mut()[i] *= sqrtm[i].max(1e-16);
        }
        let n = nrm2(mw.host_view());
        if n <= 1e-16 {
            return Array1::zeros(self.mode.len());
        }
        let mut v0ts = Array1::zeros(self.mode.len());
        for i in 0..self.mode.len() {
            v0ts[i] = self.config.dx * (mw.host_view()[i] / n) / sqrtm[i].max(1e-16);
        }
        // Sella: first nonzero of v0ts is positive, then reverse is -v0ts.
        if let Some(&v) = v0ts.iter().find(|v| v.abs() > 1e-16) {
            if v < 0.0 {
                v0ts.mapv_inplace(|c| -c);
            }
        }
        match direction {
            IrcDirection::Forward => v0ts,
            IrcDirection::Reverse => v0ts.mapv(|c| -c),
        }
    }

    fn trust(&self) -> IrcTrust {
        IrcTrust::from_atom_masses(
            self.d1.clone(),
            self.masses.as_slice().unwrap_or(&[]),
            self.config.dx,
        )
    }

    /// Sella: project `g` orthogonal to the path in the MW metric.
    fn path_force(&self, g: &Array1<f64>) -> Array1<f64> {
        path_projected_force(g, &self.d1, self.masses.as_slice().unwrap_or(&[]))
    }

    /// Sella `QuasiNewtonIRC` plus `IRCTrustRegion` on the current
    /// MW sphere. The BFGS model is identity until the first accepted
    /// inner pair; GS2 equality is `IrcTrust`.
    fn restricted_increment(&self, g: &Array1<f64>) -> Array1<f64> {
        let (evals, evecs) = self.hess.eigh();
        qn_irc_restricted(&self.trust(), &evals, &evecs, g, self.hess.is_posdef())
    }

    /// One Sella `IRC.step`: optional kick, then inner GS2 on the
    /// current MW sphere, then `d1 = 0` for the next outer sphere.
    pub fn step<S: PointSurface>(&mut self, surface: &S) -> Result<IrcReport, SaddleError> {
        let kicked = self.first;
        if self.first {
            axpy(1.0, self.d1.view(), &mut self.x);
            self.first = false;
        }
        let x_start = self.x.clone();

        let (energy0, g0) = surface.eval(self.x.view())?;
        if !g0.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("irc gradient"));
        }
        let max_force0 = nrminf(g0.view());
        let fmax_inner = self.config.force_tol.min(0.01);
        let mut inner_steps = 0;
        let mut energy = energy0;
        let mut g = g0;
        let mut max_force = max_force0;
        let mut last_interior = false;
        for _ in 0..self.config.max_inner {
            inner_steps += 1;
            let s = self.restricted_increment(&g);
            let interior = self.trust().cons(&s) + 1e-8 < self.config.dx;
            if inner_steps == 1 {
                if let Some(prev) = &self.last_outer {
                    // Reversal is a well overshoot only after the BFGS
                    // model is positive definite. At a TS the kick and
                    // -g need not be aligned.
                    if self.hess.is_posdef() && dot(prev.view(), s.view()) < 0.0 {
                        self.d1.fill(0.0);
                        self.last_step = None;
                        self.solver.forget();
                        return Ok(IrcReport {
                            energy,
                            max_force,
                            arc: self.arc,
                            inner_steps,
                            at_minimum: true,
                        });
                    }
                }
            }
            if let Some(prev) = &self.last_step {
                if !kicked && self.hess.is_posdef() && dot(prev.view(), s.view()) < 0.0 {
                    self.d1.fill(0.0);
                    self.last_step = None;
                    self.solver.forget();
                    return Ok(IrcReport {
                        energy,
                        max_force,
                        arc: self.arc,
                        inner_steps,
                        at_minimum: true,
                    });
                }
            }
            self.last_step = Some(s.clone());
            last_interior = interior;
            axpy(1.0, s.view(), &mut self.d1);
            axpy(1.0, s.view(), &mut self.x);
            let ev = surface.eval(self.x.view())?;
            energy = ev.0;
            let y = &ev.1 - &g;
            let sqrtm = sqrt_masses_3n(self.masses.as_slice().unwrap_or(&[]));
            let (s_mw, y_mw) = mw_pair(&s, &y, &sqrtm);
            self.hess.update(&s_mw, &y_mw);
            g = ev.1;
            max_force = nrminf(g.view());
            let g_path = self.path_force(&g);
            let fmax_path = nrminf(g_path.view());
            let bound = self.trust().on_bound(&Array1::zeros(s.len()), 1e-8);
            if (bound && fmax_path <= fmax_inner)
                || (max_force <= self.config.force_tol && interior)
            {
                break;
            }
        }
        let mut outer = self.x.clone();
        axpy(-1.0, x_start.view(), &mut outer);
        self.last_outer = Some(outer);
        self.d1.fill(0.0);
        self.last_step = None;
        self.arc += self.config.dx;
        self.solver.forget();
        Ok(IrcReport {
            energy,
            max_force,
            arc: self.arc,
            inner_steps,
            // Sella: the kick outer step is never a minimum, even when
            // |F| is already under fmax (a TS neighborhood).
            at_minimum: !kicked
                && max_force <= self.config.force_tol
                && last_interior,
        })
    }

    /// Convenience loop over [`IrcSession::step`]; nothing more.
    pub fn run<S: PointSurface>(
        &mut self,
        surface: &S,
        max_steps: usize,
    ) -> Result<IrcReport, SaddleError> {
        let mut report = self.step(surface)?;
        let mut n = 1;
        while !report.at_minimum && n < max_steps {
            report = self.step(surface)?;
            n += 1;
        }
        Ok(report)
    }
}

/// Mass-weighted path-orthogonal force. Algebra is `rgmin::vecops`.
fn path_projected_force(g: &Array1<f64>, d1: &Array1<f64>, masses: &[f64]) -> Array1<f64> {
    let sqrtm = sqrt_masses_3n(masses);
    let n = g.len().min(d1.len()).min(sqrtm.len());
    let mut d1m = Array1::zeros(n);
    let mut gm = Array1::zeros(n);
    for i in 0..n {
        d1m[i] = d1[i] * sqrtm[i];
        gm[i] = g[i] / sqrtm[i].max(1e-16);
    }
    let dn = nrm2(d1m.view());
    if dn > 1e-16 {
        let proj = dot(d1m.view(), gm.view()) / dn;
        axpy(-proj / dn, d1m.view(), &mut gm);
    }
    let mut force = Array1::zeros(g.len());
    for i in 0..n {
        force[i] = -gm[i] * sqrtm[i];
    }
    force
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Well;

    impl PointSurface for Well {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let t = x[0];
            let mut g = Array1::zeros(x.len());
            g[0] = 4.0 * t * (t * t - 1.0);
            Ok(((t * t - 1.0).powi(2), g))
        }
    }

    fn session(dx: f64, masses: Array1<f64>, saddle: Array1<f64>) -> IrcSession {
        let mut mode = Array1::zeros(saddle.len());
        mode[0] = 1.0;
        IrcSession::new(
            IrcConfig {
                dx,
                force_tol: 1e-3,
                ..IrcConfig::default()
            },
            saddle,
            masses,
            mode,
            IrcDirection::Forward,
        )
        .unwrap()
    }

    fn mw_radius(x: &Array1<f64>, center: &Array1<f64>, masses: &Array1<f64>) -> f64 {
        let s = x - center;
        IrcTrust::from_atom_masses(Array1::zeros(x.len()), masses.as_slice().unwrap(), 0.0).cons(&s)
    }

    #[test]
    fn kick_lands_on_the_mw_sphere() {
        let dx = 0.15;
        let masses = Array1::from(vec![1.0, 4.0]);
        let saddle = Array1::from(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let mut irc = session(dx, masses.clone(), saddle.clone());
        let _ = irc.step(&Well).unwrap();
        let r = mw_radius(&irc.position().to_owned(), &saddle, &masses);
        assert!((r - dx).abs() < 1e-10, "cons={r} dx={dx}");
    }

    struct Flat;

    impl PointSurface for Flat {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            Ok((0.0, Array1::zeros(x.len())))
        }
    }

    #[test]
    fn unequal_mass_kick_is_not_the_euclidean_sphere() {
        let dx = 0.2;
        let masses = Array1::from(vec![1.0, 16.0]);
        let saddle = Array1::from(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let mut mode = Array1::zeros(6);
        mode[0] = 1.0;
        mode[3] = 1.0;
        let mut irc = IrcSession::new(
            IrcConfig {
                dx,
                force_tol: 1e-12,
                max_inner: 1,
                ..IrcConfig::default()
            },
            saddle.clone(),
            masses.clone(),
            mode,
            IrcDirection::Forward,
        )
        .unwrap();
        let _ = irc.step(&Flat).unwrap();
        let x = irc.position().to_owned();
        let r = mw_radius(&x, &saddle, &masses);
        assert!((r - dx).abs() < 1e-10, "cons={r} dx={dx}");
        let eucl = (&x - &saddle).iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            (eucl - dx).abs() > 1e-6,
            "unequal-mass kick must not sit on the Euclidean sphere (eucl={eucl})"
        );
    }

    #[test]
    fn inner_gs2_step_stays_on_the_mw_sphere() {
        let dx = 0.15;
        let masses = Array1::from(vec![1.0, 4.0]);
        let saddle = Array1::from(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let mut irc = session(dx, masses.clone(), saddle.clone());
        let _ = irc.step(&Well).unwrap();
        let kicked = irc.position().to_owned();
        let _report = irc.step(&Well).unwrap();
        let r = mw_radius(&irc.position().to_owned(), &kicked, &masses);
        assert!(
            (r - dx).abs() < 1e-8,
            "inner step left the MW sphere: cons={r} dx={dx}"
        );
    }
}
