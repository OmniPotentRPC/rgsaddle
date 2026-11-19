//! Fixed-energy LI-NEB.
//!
//! One rgmin step on the nudged action gradient. The host owns the loop;
//! [`LinebSession::run`] repeats `step`. FIRE, L-BFGS (accepted on the
//! step, because the nudged force is not the gradient of the action),
//! Barzilai–Borwein, quick-min and ODE12r are the force-driven methods.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rgmin::{Accept, Control, Method, Oracle, Solver};

use super::action;
use super::mass::Masses;
use super::path::{self, Periods};
use crate::band::check_force_driven;
use crate::error::SaddleError;
use crate::minmode::PointSurface;

/// Controls for a fixed-energy optimal tunneling path.
///
/// The spring `k` and the endpoint scale are the values of appendix A
/// of doi:10.1021/acs.jctc.5c00673, which states them as the products
/// `k Δt² = 0.1` and `κ |∇V| Δt² = 0.2` at a unit time step. The
/// endpoint force is `κ (V − E) f̂` with `κ = endpoint_scale / |f|`.
///
/// `opt_tol` and `perp_tol` are required. They are a force scale of the
/// surface being searched (its gradient, its curvature, or the noise in
/// an evaluation), and this method does not substitute one.
///
/// Non-exhaustive: start from [`LinebConfig::new`] and chain setters.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct LinebConfig {
    /// Spring constant `k` in the parallel spacing force.
    pub spring: f64,
    /// `κ |f|` for the endpoint equipotential force.
    pub endpoint_scale: f64,
    /// Stop when every nudged force's norm is at most this.
    pub opt_tol: f64,
    /// Stop when every interior perpendicular action gradient's norm is
    /// at most this.
    pub perp_tol: f64,
    /// Initial FIRE / quick-min time step (rgmin `Control::istep`).
    ///
    /// A FIRE step moves an image by about `time_step² |F|` from rest,
    /// with `F` the nudged force in mass-weighted units, and FIRE grows
    /// the step from here. The default 0.1 converges the Müller-Brown
    /// continuation (barrier of order 100 energy units, coordinates of
    /// order one, unit masses) with every other setting at its default;
    /// 0.2 and above bend the band until the action gradient itself
    /// points out of the feasible set. A surface with a stiffer action
    /// (larger force over a smaller `√(V − E)`) needs a smaller step.
    pub time_step: f64,
    /// Largest step. With atomic masses it is the largest displacement
    /// of one atom of one image; otherwise it is the Euclidean length of
    /// the whole band step. Mass-weighted units. Zero leaves the step
    /// uncapped.
    pub max_move: f64,
    /// Regularization width `w` of the momentum near the turning
    /// surface, as a fraction of the height of the highest starting
    /// image above `E`. Interior images stay at `V ≥ E + w`.
    pub turning_width: f64,
    /// `ℏ` in the temperature `ℏ / (k_B τ)`. It does not move the path.
    pub hbar: f64,
    /// Boltzmann constant in the same temperature.
    pub boltzmann: f64,
    /// Force-driven stepper. FIRE 2.0 by default.
    pub method: Method,
    /// Simpson pairs on each chord when the period is integrated.
    pub quadrature_pairs: usize,
}

with_setters!(LinebConfig {
    with_spring => spring: val f64,
    with_endpoint_scale => endpoint_scale: val f64,
    with_opt_tol => opt_tol: val f64,
    with_perp_tol => perp_tol: val f64,
    with_time_step => time_step: val f64,
    with_max_move => max_move: val f64,
    with_turning_width => turning_width: val f64,
    with_hbar => hbar: val f64,
    with_boltzmann => boltzmann: val f64,
    with_method => method: val Method,
    with_quadrature_pairs => quadrature_pairs: val usize,
});

impl LinebConfig {
    /// Paper spring and endpoint scale, time step 0.1, unit `ℏ` and
    /// `k_B`, FIRE 2.0.
    ///
    /// `opt_tol` stops the nudged force and `perp_tol` stops the
    /// perpendicular action gradient. Both are absolute force norms in
    /// the mass-weighted coordinates.
    pub fn new(opt_tol: f64, perp_tol: f64) -> Self {
        Self {
            spring: 0.1,
            endpoint_scale: 0.2,
            opt_tol,
            perp_tol,
            time_step: 0.1,
            max_move: 0.2,
            turning_width: 1e-3,
            hbar: 1.0,
            boltzmann: 1.0,
            method: Method::fire(rgmin::FireKind::V2),
            quadrature_pairs: 4,
        }
    }
}

impl LinebConfig {
    fn check(&self) -> Result<(), SaddleError> {
        if !(self.spring.is_finite() && self.spring > 0.0) {
            return Err(SaddleError::Invalid(
                "the tunneling spring must be finite and positive".into(),
            ));
        }
        if !(self.endpoint_scale.is_finite() && self.endpoint_scale > 0.0) {
            return Err(SaddleError::Invalid(
                "the endpoint scale must be finite and positive".into(),
            ));
        }
        if !(self.opt_tol.is_finite()
            && self.opt_tol >= 0.0
            && self.perp_tol.is_finite()
            && self.perp_tol >= 0.0)
        {
            return Err(SaddleError::Invalid(
                "tunneling force tolerances must be finite and non-negative".into(),
            ));
        }
        if !(self.time_step.is_finite() && self.time_step > 0.0) {
            return Err(SaddleError::Invalid(
                "the tunneling time step must be finite and positive".into(),
            ));
        }
        if !(self.max_move.is_finite() && self.max_move >= 0.0) {
            return Err(SaddleError::Invalid(
                "the tunneling step cap must be finite and non-negative".into(),
            ));
        }
        if !(self.turning_width > 0.0 && self.turning_width < 1.0) {
            return Err(SaddleError::Invalid(
                "the turning width must be a fraction strictly between 0 and 1".into(),
            ));
        }
        if !(self.hbar.is_finite()
            && self.hbar > 0.0
            && self.boltzmann.is_finite()
            && self.boltzmann > 0.0)
        {
            return Err(SaddleError::Invalid(
                "hbar and the Boltzmann constant must be finite and positive".into(),
            ));
        }
        if self.quadrature_pairs < 1 {
            return Err(SaddleError::Invalid(
                "the period quadrature needs at least one Simpson pair per chord".into(),
            ));
        }
        check_force_driven(&self.method)
    }
}

/// Whether a fixed-energy path has met its force tolerances.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinebStatus {
    /// The perpendicular action gradient and the nudged force are within
    /// their tolerances.
    Converged,
    /// Another step is available.
    Running,
}

/// One step of a fixed-energy search.
#[derive(Clone, Debug)]
pub struct LinebReport {
    /// The status.
    pub status: LinebStatus,
    /// Discrete abbreviated action `Σ (σ_j + σ_{j+1}) |Δq| / 2`.
    pub action: f64,
    /// Largest norm of an interior perpendicular action gradient.
    pub perp_max: f64,
    /// Largest norm of a nudged force, endpoints included.
    pub opt_max: f64,
    /// Steps completed, counting this one.
    pub iteration: usize,
    /// Surface evaluations inside optimizer steps so far.
    pub evaluations: usize,
}

/// Surface evaluations behind one [`LinebPath`], by phase.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct LinebCalls {
    /// Inside [`LinebSession::step`], over the whole session.
    pub optimizer: usize,
    /// In [`LinebSession::path`]: the action and period quadrature, the
    /// imaginary-time table, and the force assembly when no step has
    /// left one at the current images.
    pub measure: usize,
    /// In [`LinebSession::path`]: the Runge–Kutta constrained dynamics.
    pub dynamics: usize,
}

impl LinebCalls {
    /// Every evaluation.
    pub fn total(&self) -> usize {
        self.optimizer + self.measure + self.dynamics
    }
}

/// A converged (or inspected) optimal tunneling path and both periods.
#[derive(Clone, Debug)]
pub struct LinebPath {
    /// Cartesian images of the aligned mass-weighted path.
    pub cartesian: Array2<f64>,
    /// Tunneling energy.
    pub energy: f64,
    /// Discrete action the optimizer minimized.
    pub discrete_action: f64,
    /// Spline action `∫ √(2(V−E)) ds`.
    pub action: f64,
    /// `−∂W/∂E` along the spline, one way.
    pub half_period: f64,
    /// Half-period constrained dynamics along the same spline.
    pub half_period_dynamics: f64,
    /// `ℏ / (k_B τ)` from [`Self::half_period`].
    pub temperature: f64,
    /// `ℏ / (k_B τ)` from [`Self::half_period_dynamics`].
    pub temperature_dynamics: f64,
    /// Surface evaluations by phase.
    pub calls: LinebCalls,
    pub(crate) weighted: Array2<f64>,
    /// `(λ, t)` imaginary time along the spline, for the replicas.
    pub(crate) time_table: path::TimeTable,
    pub(crate) masses: Masses,
    #[cfg(test)]
    pub(crate) quadrature_pairs: usize,
    pub(crate) hbar: f64,
    pub(crate) boltzmann: f64,
}

/// Fixed-energy line-integral nudged elastic band.
///
/// Images are stored in mass-weighted coordinates. Every image moves,
/// including the endpoints, which are held on `V = E`.
pub struct LinebSession {
    config: LinebConfig,
    masses: Masses,
    energy: f64,
    weighted: Array2<f64>,
    solver: Solver,
    /// Cap installed at construction, restored after a retried trial.
    cap: StepCap,
    /// A retried trial tightened the step length. The next step
    /// releases that length.
    cap_held: bool,
    /// Absolute regularization width `w`, fixed on the first step.
    width: Option<f64>,
    /// Assembly at [`Self::weighted`], when the last step ended on a
    /// point it had evaluated.
    held: Option<action::Assembly>,
    evaluations: AtomicUsize,
    iteration: usize,
}

impl LinebSession {
    /// `cartesian` has one image per row, in the same coordinates the
    /// surface evaluates. Every image must lie strictly above `energy`.
    pub fn new(
        config: LinebConfig,
        masses: Masses,
        cartesian: Array2<f64>,
        energy: f64,
    ) -> Result<Self, SaddleError> {
        config.check()?;
        if !(energy.is_finite()) {
            return Err(SaddleError::Invalid(
                "the tunneling energy must be finite".into(),
            ));
        }
        if cartesian.nrows() < 3 || cartesian.ncols() != masses.dof() {
            return Err(SaddleError::Shape(
                "LI-NEB needs at least three images and one mass per coordinate".into(),
            ));
        }
        if cartesian.iter().any(|value| !value.is_finite()) {
            return Err(SaddleError::NonFinite("initial tunneling path"));
        }
        let mut weighted = Array2::zeros(cartesian.dim());
        for i in 0..cartesian.nrows() {
            weighted
                .row_mut(i)
                .assign(&masses.to_weighted(cartesian.row(i))?);
        }
        let dof = cartesian.nrows() * cartesian.ncols();
        let control = Control::default()
            .with_maxiter(usize::MAX)
            .with_gtol(0.0)
            .with_istep(config.time_step)
            .with_maxmove(None);
        let mut solver = Solver::new(config.method.clone(), control, dof);
        let cap = if config.max_move <= 0.0 {
            StepCap::None
        } else if masses.atoms.is_some() {
            StepCap::Atom(config.max_move)
        } else {
            StepCap::Band(config.max_move)
        };
        cap.install(&mut solver);
        if matches!(config.method, Method::Lbfgs { .. }) {
            solver.set_accept(Accept::Step);
            solver.set_lbfgs_neb_guards(true);
        }
        Ok(Self {
            config,
            masses,
            energy,
            weighted,
            solver,
            cap,
            cap_held: false,
            width: None,
            held: None,
            evaluations: AtomicUsize::new(0),
            iteration: 0,
        })
    }

    /// Mass-weighted images the solver holds.
    pub fn weighted(&self) -> ArrayView2<'_, f64> {
        self.weighted.view()
    }

    /// Cartesian images corresponding to [`Self::weighted`].
    pub fn cartesian(&self) -> Result<Array2<f64>, SaddleError> {
        let mut out = Array2::zeros(self.weighted.dim());
        for i in 0..self.weighted.nrows() {
            out.row_mut(i)
                .assign(&self.masses.to_cartesian(self.weighted.row(i))?);
        }
        Ok(out)
    }

    /// `w` from [`LinebConfig::turning_width`] and the highest image's
    /// height above the energy. Every interior image must be at
    /// `V ≥ E + w`.
    fn turning_width<S: PointSurface>(&self, surface: &S) -> Result<f64, SaddleError> {
        let n = self.weighted.nrows();
        let mut top = f64::NEG_INFINITY;
        let mut lowest_interior = f64::INFINITY;
        for (i, row) in self.weighted.rows().into_iter().enumerate() {
            let x = self.masses.to_cartesian(row)?;
            let (potential, _) = surface.eval(x.view())?;
            if !potential.is_finite() {
                return Err(SaddleError::NonFinite("tunneling image"));
            }
            top = top.max(potential);
            if i > 0 && i + 1 < n {
                lowest_interior = lowest_interior.min(potential);
            }
        }
        let width = self.config.turning_width * (top - self.energy);
        if width <= 0.0 || width.is_nan() || lowest_interior - self.energy < width {
            return Err(SaddleError::Invalid(format!(
                "every interior tunneling image must start at V >= E + {width}"
            )));
        }
        Ok(width)
    }

    /// One optimizer step.
    ///
    /// A trial that carries an interior image below `E + w` is outside
    /// the feasible set. That trial alone is retried, at half the length
    /// that left the set, until it lands inside or no shorter step moves.
    /// Any other error from the surface or the assembly is returned.
    pub fn step<S: PointSurface>(&mut self, surface: &S) -> Result<LinebReport, SaddleError> {
        let n = self.weighted.nrows();
        let dof = self.weighted.ncols();
        let flat_dof = n * dof;
        let calls = &self.evaluations;
        let counted = crate::lineb::Counted { surface, calls };
        let width = match self.width {
            Some(width) => width,
            None => {
                let width = self.turning_width(&counted)?;
                self.width = Some(width);
                width
            }
        };
        if self.cap_held {
            self.solver.set_maxmove(f64::INFINITY);
            self.cap.install(&mut self.solver);
            self.cap_held = false;
        }
        let config = self.config.clone();
        let masses = self.masses.clone();
        let energy = self.energy;
        let mut x = action::pack(self.weighted.view());
        let before = x.clone();
        // `f64::INFINITY` marks "the stepper's own length", not a bound.
        let mut cap = f64::INFINITY;
        let assembly = loop {
            // The last two feasible points and their assemblies. The
            // solver accepts its last evaluation or, after a rejection,
            // the one before it.
            let evaluated: Mutex<Vec<(Array1<f64>, action::Assembly)>> = Mutex::new(Vec::new());
            let outcome = Mutex::new(Trial::Inside);
            let oracle = Oracle::unbounded(flat_dof, |trial: ArrayView1<f64>| {
                let images = action::unpack(trial, n, dof);
                let assembled = action::assemble(
                    &counted,
                    &masses,
                    images.view(),
                    energy,
                    config.spring,
                    config.endpoint_scale,
                    width,
                );
                let mut state = outcome.lock().unwrap_or_else(|e| e.into_inner());
                match assembled {
                    Ok(Some(assembly)) => {
                        let mut force = action::pack(assembly.drive.view());
                        force *= -1.0;
                        let value = assembly.action;
                        let mut slots = evaluated.lock().unwrap_or_else(|e| e.into_inner());
                        if slots.len() == 2 {
                            slots.remove(0);
                        }
                        slots.push((trial.to_owned(), assembly));
                        (value, force)
                    }
                    Ok(None) => {
                        if !matches!(*state, Trial::Failed(_)) {
                            *state = Trial::Outside(action::l2((&trial - &before).view()));
                        }
                        (f64::INFINITY, Array1::zeros(flat_dof))
                    }
                    Err(err) => {
                        *state = Trial::Failed(err);
                        (f64::INFINITY, Array1::zeros(flat_dof))
                    }
                }
            });
            let stepped = self.solver.step(&oracle, &mut x);
            drop(oracle);
            match outcome.into_inner().unwrap_or_else(|e| e.into_inner()) {
                Trial::Failed(err) => {
                    // The solver cached the failed point's infinite value
                    // and zero force; the next step must evaluate again.
                    self.solver.forget_evaluation();
                    return Err(err);
                }
                Trial::Outside(attempted) => {
                    self.solver.forget_evaluation();
                    x.assign(&before);
                    let shorter = 0.5 * cap.min(attempted);
                    if shorter <= 0.0 || shorter.is_nan() {
                        return Err(SaddleError::Stalled(format!(
                            "no shorter step keeps the interior images at V >= E + {width}"
                        )));
                    }
                    cap = shorter;
                    self.solver.set_maxmove(cap);
                    if matches!(self.cap, StepCap::Atom(_)) {
                        self.solver.set_atom_maxmove(cap);
                    }
                    self.cap_held = true;
                }
                Trial::Inside => {
                    stepped.map_err(|err| SaddleError::Solver(err.to_string()))?;
                    let mut slots = evaluated.into_inner().unwrap_or_else(|e| e.into_inner());
                    let cached = match slots.iter().rposition(|(at, _)| *at == x) {
                        Some(index) => Some(slots.swap_remove(index).1),
                        None if x == before => self.held.take(),
                        None => None,
                    };
                    break match cached {
                        Some(assembly) => assembly,
                        None => action::assemble(
                            &counted,
                            &masses,
                            action::unpack(x.view(), n, dof).view(),
                            energy,
                            config.spring,
                            config.endpoint_scale,
                            width,
                        )?
                        .ok_or_else(|| {
                            SaddleError::Solver(
                                "the solver accepted a point outside the feasible set".into(),
                            )
                        })?,
                    };
                }
            }
        };
        let delta = &x - &before;
        let displacement = action::l2(delta.view());
        self.weighted = action::unpack(x.view(), n, dof);
        self.iteration += 1;
        self.held = Some(assembly.clone());
        let converged = assembly.perp_max <= config.perp_tol && assembly.opt_max <= config.opt_tol;
        if (displacement <= 0.0 || displacement.is_nan()) && !converged {
            return Err(SaddleError::Stalled(format!(
                "tunneling path unchanged with nudged force {} above {}",
                assembly.opt_max, config.opt_tol
            )));
        }
        Ok(LinebReport {
            status: if converged {
                LinebStatus::Converged
            } else {
                LinebStatus::Running
            },
            action: assembly.action,
            perp_max: assembly.perp_max,
            opt_max: assembly.opt_max,
            iteration: self.iteration,
            evaluations: self.evaluations.load(Ordering::Relaxed),
        })
    }

    /// Step until the tolerances are met or `max_steps` steps have been
    /// taken. The report of the last step is returned either way.
    pub fn run<S: PointSurface>(
        &mut self,
        surface: &S,
        max_steps: usize,
    ) -> Result<LinebReport, SaddleError> {
        let mut report = self.step(surface)?;
        while report.status == LinebStatus::Running && self.iteration < max_steps {
            report = self.step(surface)?;
        }
        Ok(report)
    }

    /// Spline action, both half periods and both temperatures of the
    /// current images.
    pub fn path<S: PointSurface>(&self, surface: &S) -> Result<LinebPath, SaddleError> {
        let measure_calls = AtomicUsize::new(0);
        let dynamics_calls = AtomicUsize::new(0);
        let measured = crate::lineb::Counted {
            surface,
            calls: &measure_calls,
        };
        let dynamics = crate::lineb::Counted {
            surface,
            calls: &dynamics_calls,
        };
        let assembly = match &self.held {
            Some(assembly) => assembly.clone(),
            None => {
                let width = match self.width {
                    Some(width) => width,
                    None => self.turning_width(&measured)?,
                };
                action::assemble(
                    &measured,
                    &self.masses,
                    self.weighted.view(),
                    self.energy,
                    self.config.spring,
                    self.config.endpoint_scale,
                    width,
                )?
                .ok_or_else(|| {
                    SaddleError::Invalid(
                        "an interior tunneling image is below the feasible energy".into(),
                    )
                })?
            }
        };
        let periods: Periods = path::measure(
            &measured,
            &dynamics,
            &self.masses,
            assembly.aligned.view(),
            self.energy,
            self.config.quadrature_pairs,
        )?;
        let mut cartesian = Array2::zeros(assembly.aligned.dim());
        for i in 0..assembly.aligned.nrows() {
            cartesian
                .row_mut(i)
                .assign(&self.masses.to_cartesian(assembly.aligned.row(i))?);
        }
        Ok(LinebPath {
            cartesian,
            energy: self.energy,
            discrete_action: assembly.action,
            action: periods.action,
            half_period: periods.half_period,
            half_period_dynamics: periods.half_period_dynamics,
            temperature: path::temperature(
                periods.half_period,
                self.config.hbar,
                self.config.boltzmann,
            )?,
            temperature_dynamics: path::temperature(
                periods.half_period_dynamics,
                self.config.hbar,
                self.config.boltzmann,
            )?,
            calls: LinebCalls {
                optimizer: self.evaluations.load(Ordering::Relaxed),
                measure: measure_calls.load(Ordering::Relaxed),
                dynamics: dynamics_calls.load(Ordering::Relaxed),
            },
            weighted: assembly.aligned,
            time_table: periods.time_table,
            masses: self.masses.clone(),
            #[cfg(test)]
            quadrature_pairs: self.config.quadrature_pairs,
            hbar: self.config.hbar,
            boltzmann: self.config.boltzmann,
        })
    }
}

/// The step cap a session restores after a retried trial.
#[derive(Clone, Copy, Debug)]
enum StepCap {
    None,
    /// Per-atom displacement.
    Atom(f64),
    /// Euclidean length of the whole band step.
    Band(f64),
}

impl StepCap {
    fn install(self, solver: &mut Solver) {
        match self {
            StepCap::None => {}
            StepCap::Atom(cap) => solver.set_atom_maxmove(cap),
            StepCap::Band(cap) => solver.set_maxmove(cap),
        }
    }
}

/// What the objective saw during one solver step.
enum Trial {
    Inside,
    /// An interior image fell below `E + w` at this step length.
    Outside(f64),
    Failed(SaddleError),
}
