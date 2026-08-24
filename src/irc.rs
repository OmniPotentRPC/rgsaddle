//! Intrinsic reaction coordinate session: roll from a first-order
//! saddle to a minimum, one branch at a time.
//!
//! Inner geometry is rgmin `IrcTrust` (Sella IRCTrustRegion /
//! Gonzalez--Schlegel MW sphere). The host owns the loop. `run` is a
//! convenience over `step`.

use ndarray::{Array1, ArrayView1};
use rgmin::{sqrt_masses_3n, Control, IrcTrust, Method, Solver};

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
}

impl Default for IrcConfig {
    fn default() -> Self {
        Self {
            dx: 0.1,
            force_tol: 0.05,
            max_move: 0.2,
            method: Method::Steepest,
            max_inner: 10,
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
    first: bool,
    arc: f64,
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
        let solver = Solver::new(config.method.clone(), control, n3);
        let mut session = Self {
            config,
            x: saddle.clone(),
            x_saddle: saddle,
            masses,
            d1: Array1::zeros(n3),
            mode,
            solver,
            first: true,
            arc: 0.0,
        };
        session.set_direction(direction);
        Ok(session)
    }

    pub fn position(&self) -> ArrayView1<'_, f64> {
        self.x.view()
    }

    pub fn reset(&mut self) {
        self.solver.forget();
    }

    /// Restore the saddle and flip the kick sign.
    pub fn set_direction(&mut self, direction: IrcDirection) {
        self.x = self.x_saddle.clone();
        self.d1 = self.kick_vector(direction);
        self.first = true;
        self.arc = 0.0;
        self.solver.forget();
    }

    fn kick_vector(&self, direction: IrcDirection) -> Array1<f64> {
        let sqrtm = sqrt_masses_3n(self.masses.as_slice().unwrap_or(&[]));
        let mut mw = Array1::zeros(self.mode.len());
        for i in 0..self.mode.len() {
            let sm = sqrtm[i].max(1e-16);
            mw[i] = self.mode[i] * sm;
        }
        let n = mw.iter().map(|v| v * v).sum::<f64>().sqrt();
        if n <= 1e-16 {
            return Array1::zeros(self.mode.len());
        }
        let sign = match direction {
            IrcDirection::Forward => 1.0,
            IrcDirection::Reverse => -1.0,
        };
        let mut d1 = Array1::zeros(self.mode.len());
        for i in 0..self.mode.len() {
            d1[i] = sign * self.config.dx * (mw[i] / n) / sqrtm[i].max(1e-16);
        }
        // Sella: first nonzero component of v0ts is positive.
        if let Some(&v) = d1.iter().find(|v| v.abs() > 1e-16) {
            if v < 0.0 && matches!(direction, IrcDirection::Forward) {
                d1.mapv_inplace(|c| -c);
            }
        }
        d1
    }

    fn trust(&self) -> IrcTrust {
        IrcTrust::from_atom_masses(self.d1.clone(), self.masses.as_slice().unwrap_or(&[]), self.config.dx)
    }

    /// One outer IRC move: first call kicks; later calls take a
    /// steepest step projected onto the MW sphere and accumulate `d1`.
    pub fn step<S: PointSurface>(&mut self, surface: &S) -> Result<IrcReport, SaddleError> {
        if self.first {
            self.x = &self.x + &self.d1;
            self.first = false;
            self.arc += self.config.dx;
            let (energy, g) = surface.eval(self.x.view())?;
            let max_force = g.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            return Ok(IrcReport {
                energy,
                max_force,
                arc: self.arc,
                inner_steps: 0,
                at_minimum: max_force <= self.config.force_tol,
            });
        }

        let (energy0, g0) = surface.eval(self.x.view())?;
        if !g0.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("irc gradient"));
        }
        let max_force0 = g0.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        if max_force0 <= self.config.force_tol {
            return Ok(IrcReport {
                energy: energy0,
                max_force: max_force0,
                arc: self.arc,
                inner_steps: 0,
                at_minimum: true,
            });
        }

        let trust = self.trust();
        let mut inner_steps = 0;
        let mut energy = energy0;
        let mut max_force = max_force0;
        for _ in 0..self.config.max_inner {
            inner_steps += 1;
            let mut s = -&g0;
            let gn = s.iter().map(|v| v * v).sum::<f64>().sqrt();
            if gn > 1e-16 {
                s.mapv_inplace(|v| v * (self.config.dx / gn));
            }
            s = trust.project(&s);
            self.d1 = &self.d1 + &s;
            let trial = &self.x + &s;
            let ev = surface.eval(trial.view())?;
            energy = ev.0;
            let g = ev.1;
            max_force = g.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            self.x = trial;
            if trust.on_bound(&s, 1e-8) || max_force <= self.config.force_tol {
                break;
            }
        }
        self.d1.fill(0.0);
        self.arc += self.config.dx;
        self.solver.forget();
        Ok(IrcReport {
            energy,
            max_force,
            arc: self.arc,
            inner_steps,
            at_minimum: max_force <= self.config.force_tol,
        })
    }
}
