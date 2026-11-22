//! Fixed-energy LI-NEB.
//!
//! One rgmin step on the nudged action gradient. The host owns the loop;
//! [`LinebSession::run`] repeats `step`. FIRE, L-BFGS (accepted on the
//! step, because the nudged force is not the gradient of the action),
//! Barzilai–Borwein, quick-min and ODE12r are the force-driven methods.
//! [`LinebConfig::with_projected_verlet`] opts into a fixed-time-step
//! projected velocity Verlet that uses [`rgmin::quickmin_project`].

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rgmin::{Accept, Control, Method, Oracle, Solver, quickmin_project};

use super::action;
use super::mass::Masses;
use super::path;
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
    /// Fixed coefficient κ of (V − E) along the unit surface force.
    /// When set, this replaces the gradient-normalized endpoint scale.
    pub endpoint_kappa: Option<f64>,
    /// Apply opt_tol only to the endpoints, and perp_tol to interior images.
    pub endpoint_convergence: bool,
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
    /// Use piecewise-linear chord coordinates between imaginary-time samples.
    /// This permits comparison with sampled linear clocks. The default Hermite
    /// clock uses the physical velocities and stops at each turning point.
    pub linear_replica_clock: bool,
    /// Propagate the signed inverted-potential force interpolated from the images.
    pub interpolated_force_clock: bool,
    /// Simpson pairs on each chord when the period is integrated.
    pub quadrature_pairs: usize,
    /// Fixed-time-step projected velocity Verlet instead of [`Self::method`].
    ///
    /// Off by default. When set, each step is velocity Verlet at
    /// [`Self::time_step`] with the velocity replaced by
    /// [`rgmin::quickmin_project`] after the kick. Feasible-set retries,
    /// force caching, and surface-error returns match the default path.
    pub projected_verlet: bool,
    /// Use unit mass-weighted surface-gradient directions as the endpoint
    /// tangents of the measured path spline.
    pub gradient_end_tangents: bool,
}

with_setters!(LinebConfig {
    with_spring => spring: val f64,
    with_endpoint_convergence => endpoint_convergence: val bool,
    with_endpoint_scale => endpoint_scale: val f64,
    with_opt_tol => opt_tol: val f64,
    with_perp_tol => perp_tol: val f64,
    with_time_step => time_step: val f64,
    with_max_move => max_move: val f64,
    with_turning_width => turning_width: val f64,
    with_hbar => hbar: val f64,
    with_boltzmann => boltzmann: val f64,
    with_method => method: val Method,
    with_interpolated_force_clock => interpolated_force_clock: val bool,
    with_linear_replica_clock => linear_replica_clock: val bool,
    with_quadrature_pairs => quadrature_pairs: val usize,
    with_projected_verlet => projected_verlet: val bool,
    with_gradient_end_tangents => gradient_end_tangents: val bool,
});

impl LinebConfig {
    /// Use a constant endpoint coefficient as in Eqs. (19) and (21).
    pub fn with_endpoint_kappa(mut self, kappa: f64) -> Self {
        self.endpoint_kappa = Some(kappa);
        self
    }

    fn endpoint_restraint(&self) -> action::EndpointRestraint {
        match self.endpoint_kappa {
            Some(kappa) => action::EndpointRestraint::Fixed(kappa),
            None => action::EndpointRestraint::Normalized(self.endpoint_scale),
        }
    }

    /// Gradient-normalized endpoint force, time step 0.1, unit `ℏ` and
    /// `k_B`, FIRE 2.0.
    ///
    /// `opt_tol` stops the nudged force and `perp_tol` stops the
    /// perpendicular action gradient. Both are absolute force norms in
    /// the mass-weighted coordinates.
    pub fn new(opt_tol: f64, perp_tol: f64) -> Self {
        Self {
            spring: 0.1,
            endpoint_scale: 0.2,
            endpoint_kappa: None,
            endpoint_convergence: false,
            opt_tol,
            perp_tol,
            time_step: 0.1,
            max_move: 0.2,
            turning_width: 1e-3,
            hbar: 1.0,
            boltzmann: 1.0,
            method: Method::fire(rgmin::FireKind::V2),
            quadrature_pairs: 4,
            linear_replica_clock: false,
            interpolated_force_clock: false,
            projected_verlet: false,
            gradient_end_tangents: false,
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
        if !self.endpoint_restraint().is_valid() {
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
    /// Largest nudged force norm at either endpoint.
    pub endpoint_max: f64,
    /// Steps completed, counting this one.
    pub iteration: usize,
    /// Surface evaluations inside optimizer steps so far.
    pub evaluations: usize,
}

/// Surface evaluations behind one [`LinebPath`], by phase.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct LinebCalls {
    /// Evaluations preparing the seed at this energy.
    pub seed: usize,
    seed_in_optimizer: usize,
    /// Measurement evaluations preparing or refining the turning-point path.
    pub preparation: usize,
    /// Session evaluations including seed clipping by [`LinebSession::from_seed`].
    pub optimizer: usize,
    /// In [`LinebSession::path`]: the action and period quadrature, the
    /// imaginary-time table, and the force assembly when no step has
    /// left one at the current images.
    pub measure: usize,
    /// In [`LinebSession::path`]: the Runge–Kutta constrained dynamics.
    pub dynamics: usize,
}

impl LinebCalls {
    /// Optimizer evaluations after seed preparation.
    pub fn steps(&self) -> usize {
        self.optimizer - self.seed_in_optimizer
    }

    /// Action quadrature and imaginary-time table evaluations after path preparation.
    pub fn quadrature(&self) -> usize {
        self.measure - self.preparation
    }

    /// All path evaluations, including external continuation seeds.
    pub fn total_with_seed(&self) -> usize {
        self.seed + self.steps() + self.measure + self.dynamics
    }

    /// Session and measurement evaluations, excluding external continuation seeds.
    pub fn total(&self) -> usize {
        self.optimizer + self.measure + self.dynamics
    }
}

/// Measured agreement between the force-interpolated return turn and the
/// optimized turning image on the opposite end of the path.
#[derive(Clone, Copy, Debug)]
pub struct LinebTurning {
    /// Index of the image from which the trajectory starts at rest.
    pub launch_image: usize,
    /// Mass-weighted arc length of the prepared coordinate spline.
    pub path_arc_length: f64,
    /// Physical return arc coordinate measured from the first image.
    pub return_arc: f64,
    /// Maximum absolute Cartesian coordinate difference from the opposite image.
    pub max_cartesian_displacement: f64,
    /// Measured potential at the return turn minus the requested tunneling energy.
    pub energy_residual: f64,
    /// Measured potential at the return turn minus that at the opposite image.
    pub endpoint_energy_difference: f64,
}

/// A converged (or inspected) optimal tunneling path and both periods.
#[derive(Clone, Debug)]
pub struct LinebPath<D = f64> {
    /// Cartesian images of the aligned mass-weighted path.
    pub cartesian: Array2<f64>,
    /// Tunneling energy.
    pub energy: f64,
    /// Discrete action the optimizer minimized.
    pub discrete_action: f64,
    /// Spline action `∫ √(2(V−E)) ds`.
    pub action: f64,
    /// Turning-to-turning time of the selected replica clock.
    pub half_period: f64,
    /// Return-turn measurement when using the interpolated image force.
    pub turning: Option<LinebTurning>,
    /// Half-period constrained dynamics along the same spline.
    pub half_period_dynamics: D,
    /// `ℏ / (k_B τ)` from [`Self::half_period`].
    pub temperature: f64,
    /// `ℏ / (k_B τ)` from [`Self::half_period_dynamics`].
    pub temperature_dynamics: D,
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

impl<D> LinebPath<D> {
    /// Place a full ring of replicas at equal intervals of imaginary time.
    /// Each row contains one Cartesian geometry, with the first turning
    /// point at row zero. The return leg retraces the outbound spline.
    pub fn cartesian_replicas(&self, beads: usize) -> Result<Array2<f64>, SaddleError> {
        let mut ring = path::replicas(self.weighted.view(), &self.time_table, beads)?.weighted;
        for mut row in ring.rows_mut() {
            let cartesian = self.masses.to_cartesian(row.view())?;
            row.assign(&cartesian);
        }
        Ok(ring)
    }
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
    /// Velocity for [`LinebConfig::projected_verlet`]. Unused otherwise.
    velocity: Array1<f64>,
    evaluations: AtomicUsize,
    seed_evaluations: usize,
    iteration: usize,
}

impl LinebSession {
    /// Number of completed optimizer iterations, including the last stalled step.
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// Prepare a single energy from a full reaction path or a turning-point seed.
    /// If both endpoints lie below the requested energy, trim the seed to the
    /// two crossings around its highest image and retain its image count.
    /// Endpoints on or above the target retain an already prepared band.
    /// When one endpoint is above and the other below, both must lie
    /// within the energy roundoff tolerance to form a saved turning pair.
    /// A seed spanning just one side of the barrier is invalid.
    pub fn from_seed<S: PointSurface>(
        surface: &S,
        config: LinebConfig,
        masses: Masses,
        cartesian: Array2<f64>,
        energy: f64,
    ) -> Result<Self, SaddleError> {
        Self::from_seed_with_interpolation(surface, config, masses, cartesian, energy, false)
    }

    /// As in [`Self::from_seed`], with straight interpolation between the
    /// turning surfaces when `linear_seed` is true and the seed spans both minima.
    pub fn from_seed_with_interpolation<S: PointSurface>(
        surface: &S,
        config: LinebConfig,
        masses: Masses,
        cartesian: Array2<f64>,
        energy: f64,
        linear_seed: bool,
    ) -> Result<Self, SaddleError> {
        let mut session = Self::new(config, masses, cartesian.clone(), energy)?;
        let counted = super::Counted {
            surface,
            calls: &session.evaluations,
        };
        if let Some(seed) = super::continue_energy::clip_seed(
            &counted,
            &session.masses,
            cartesian.view(),
            energy,
            linear_seed,
        )? {
            for (mut weighted, row) in session.weighted.rows_mut().into_iter().zip(seed.rows()) {
                weighted.assign(&session.masses.to_weighted(row)?);
            }
        }
        session.seed_evaluations = session.evaluations.load(Ordering::Relaxed);
        Ok(session)
    }

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
        let velocity = Array1::zeros(if config.projected_verlet { dof } else { 0 });
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
            velocity,
            evaluations: AtomicUsize::new(0),
            seed_evaluations: 0,
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
        let width = match self.width {
            Some(width) => width,
            None => {
                let width = self.turning_width(&crate::lineb::Counted {
                    surface,
                    calls: &self.evaluations,
                })?;
                self.width = Some(width);
                width
            }
        };
        if self.config.projected_verlet {
            self.step_projected_verlet(surface, width)
        } else {
            self.step_solver(surface, width)
        }
    }

    fn step_solver<S: PointSurface>(
        &mut self,
        surface: &S,
        width: f64,
    ) -> Result<LinebReport, SaddleError> {
        let n = self.weighted.nrows();
        let dof = self.weighted.ncols();
        let flat_dof = n * dof;
        let calls = &self.evaluations;
        let counted = crate::lineb::Counted { surface, calls };
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
                    config.endpoint_restraint(),
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
                    // The optimizer has advanced its velocity or curvature history
                    // to the rejected trial. Restart it at the accepted coordinates.
                    self.solver.forget();
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
                            config.endpoint_restraint(),
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
        self.finish_step(before.view(), x.view(), assembly)
    }

    /// Fixed-`Δt` projected velocity Verlet on the nudged force.
    ///
    /// Half-kick, drift under the configured displacement bounds with the
    /// mid-step velocity rescaled to the step that was taken, evaluate,
    /// complete the kick, then [`quickmin_project`]. An interior image
    /// below `E + w` zeros the velocity and retries at half the attempted
    /// length; a surface error is returned without treating the trial as
    /// a feasibility failure.
    fn step_projected_verlet<S: PointSurface>(
        &mut self,
        surface: &S,
        width: f64,
    ) -> Result<LinebReport, SaddleError> {
        let before = action::pack(self.weighted.view());
        let config = &self.config;
        let dt = config.time_step;
        let counted = crate::lineb::Counted {
            surface,
            calls: &self.evaluations,
        };
        let assemble = |images: ArrayView2<'_, f64>| {
            action::assemble(
                &counted,
                &self.masses,
                images,
                self.energy,
                config.spring,
                config.endpoint_restraint(),
                width,
            )
        };
        let old = match &self.held {
            Some(assembly) => action::pack(assembly.drive.view()),
            None => action::pack(
                assemble(self.weighted.view())?
                    .ok_or_else(|| {
                        SaddleError::Invalid("the starting band is outside the feasible set".into())
                    })?
                    .drive
                    .view(),
            ),
        };
        let half_kick = &old * (0.5 * dt);
        let mut drift = (&self.velocity + &half_kick) * dt;
        self.cap.apply(&mut drift);
        loop {
            let trial = &before + &drift;
            let images = action::unpack(trial.view(), self.weighted.nrows(), self.weighted.ncols());
            if let Some(assembly) = assemble(images.view())? {
                let new_force = action::pack(assembly.drive.view());
                let velocity = drift / dt + &new_force * (0.5 * dt);
                self.velocity = quickmin_project(&velocity, &new_force);
                return self.finish_step(before.view(), trial.view(), assembly);
            }
            // Restart from rest at the accepted band and halve the infeasible move.
            let limit = 0.5 * action::l2(drift.view());
            if trial == before || limit <= 0.0 {
                return Err(SaddleError::Stalled(format!(
                    "no shorter step keeps the interior images at V >= E + {width}"
                )));
            }
            drift = &half_kick * dt;
            self.cap.apply(&mut drift);
            StepCap::Band(limit).apply(&mut drift);
        }
    }

    fn finish_step(
        &mut self,
        before: ArrayView1<'_, f64>,
        x: ArrayView1<'_, f64>,
        assembly: action::Assembly,
    ) -> Result<LinebReport, SaddleError> {
        let n = self.weighted.nrows();
        let dof = self.weighted.ncols();
        let config = &self.config;
        let displacement = action::l2((&x - &before).view());
        self.weighted = action::unpack(x, n, dof);
        self.iteration += 1;
        self.held = Some(assembly.clone());
        let tested_force = if config.endpoint_convergence {
            assembly.endpoint_max
        } else {
            assembly.opt_max
        };
        let converged = assembly.perp_max <= config.perp_tol && tested_force <= config.opt_tol;
        if (displacement <= 0.0 || displacement.is_nan()) && !converged {
            return Err(SaddleError::Stalled(format!(
                "tunneling path unchanged at energy {} on iteration {}: tested nudged force {} (tolerance {}), perpendicular force {} (tolerance {})",
                self.energy,
                self.iteration,
                tested_force,
                config.opt_tol,
                assembly.perp_max,
                config.perp_tol
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
            endpoint_max: assembly.endpoint_max,
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

    /// Cartesian image centers with both near-turning endpoints refined to the
    /// requested energy. These are the centers used by the interpolated rate.
    pub fn refined_cartesian_images<S: PointSurface>(
        &self,
        surface: &S,
    ) -> Result<Array2<f64>, SaddleError> {
        let (weighted, _) = self.prepared_images(surface)?;
        self.cartesian_images(&weighted)
    }

    fn prepared_images<S: PointSurface>(
        &self,
        surface: &S,
    ) -> Result<(Array2<f64>, f64), SaddleError> {
        let assembly = match &self.held {
            Some(assembly) => assembly.clone(),
            None => {
                let width = match self.width {
                    Some(width) => width,
                    None => self.turning_width(surface)?,
                };
                action::assemble(
                    surface,
                    &self.masses,
                    self.weighted.view(),
                    self.energy,
                    self.config.spring,
                    self.config.endpoint_restraint(),
                    width,
                )?
                .ok_or_else(|| {
                    SaddleError::Invalid(
                        "an interior tunneling image is below the feasible energy".into(),
                    )
                })?
            }
        };
        let weighted = path::refine_near_turning_endpoints(
            surface,
            &self.masses,
            assembly.aligned,
            self.energy,
            self.config.endpoint_restraint(),
            self.config.opt_tol,
        )?;
        Ok((weighted, assembly.action))
    }

    fn cartesian_images(&self, weighted: &Array2<f64>) -> Result<Array2<f64>, SaddleError> {
        let mut cartesian = Array2::zeros(weighted.dim());
        for (mut row, q) in cartesian.outer_iter_mut().zip(weighted.rows()) {
            row.assign(&self.masses.to_cartesian(q)?);
        }
        Ok(cartesian)
    }

    /// Spline action, both half periods and both temperatures of the
    /// current images.
    pub fn path<S: PointSurface>(&self, surface: &S) -> Result<LinebPath, SaddleError> {
        self.measured_path(surface, |counted, curve, active| {
            let period = match active {
                Some(period) => period,
                None => path::dynamics(
                    counted,
                    &self.masses,
                    curve,
                    self.energy,
                    self.config.quadrature_pairs,
                )?,
            };
            Ok((
                period,
                path::temperature(period, self.config.hbar, self.config.boltzmann)?,
            ))
        })
    }

    /// Spline action, quadrature temperature, and the imaginary-time ring at
    /// the current images. The independent constrained-dynamics observables
    /// carry the unit type. This path supports the same rate evaluators.
    pub fn quadrature_path<S: PointSurface>(
        &self,
        surface: &S,
    ) -> Result<LinebPath<()>, SaddleError> {
        self.measured_path(surface, |_, _, _| Ok(((), ())))
    }

    /// Unit mass-weighted gradient directions at the two end images,
    /// oriented along increasing path parameter.
    fn gradient_tangents<S: PointSurface>(
        &self,
        surface: &S,
        weighted: &Array2<f64>,
    ) -> Result<(Vec<f64>, Vec<f64>), SaddleError> {
        let last = weighted.nrows() - 1;
        let tangent = |end: usize, neighbour: usize| -> Result<Vec<f64>, SaddleError> {
            let x = self.masses.to_cartesian(weighted.row(end))?;
            let (_, gradient) = surface.eval(x.view())?;
            let force = self.masses.weighted_force(gradient.view())?;
            let norm = action::l2(force.view());
            if !(norm.is_finite() && norm > 0.0) {
                return Err(SaddleError::Invalid(
                    "a turning point needs a finite nonzero gradient".into(),
                ));
            }
            // The path parameter runs from image 0 to the last image.
            let mut along = 0.0;
            for k in 0..force.len() {
                along += force[k] * (weighted[(neighbour, k)] - weighted[(end, k)]);
            }
            let toward_neighbour = if along >= 0.0 { 1.0 } else { -1.0 };
            let sign = if end == 0 {
                toward_neighbour
            } else {
                -toward_neighbour
            };
            Ok(force.iter().map(|f| sign * f / norm).collect())
        };
        Ok((tangent(0, 1)?, tangent(last, last - 1)?))
    }

    fn measured_path<S: PointSurface, D>(
        &self,
        surface: &S,
        dynamics_clock: impl FnOnce(
            &crate::lineb::Counted<'_, S>,
            &path::Curve,
            Option<f64>,
        ) -> Result<(D, D), SaddleError>,
    ) -> Result<LinebPath<D>, SaddleError> {
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
        let (weighted, discrete_action) = self.prepared_images(&measured)?;
        let preparation = measure_calls.load(Ordering::Relaxed);
        let end_tangents = if self.config.gradient_end_tangents {
            Some(self.gradient_tangents(&measured, &weighted)?)
        } else {
            None
        };
        let curve = path::Curve::with_end_tangents(weighted.view(), end_tangents.as_ref())?;
        let pairs = self.config.quadrature_pairs;
        let (action, half_period, time_table, turning) = if self.config.interpolated_force_clock {
            let action = path::action_only(&measured, &self.masses, &curve, self.energy, pairs)?;
            let (half, table, turning) =
                path::force_clock(&dynamics, &self.masses, &curve, self.energy, pairs)?;
            (action, half, table, Some(turning))
        } else {
            let (action, half, mut table) =
                path::quadrature(&measured, &self.masses, &curve, self.energy, pairs)?;
            table.linear = self.config.linear_replica_clock;
            (action, half, table, None)
        };
        let mut time_table = time_table;
        time_table.end_tangents = end_tangents;
        let (half_period_dynamics, temperature_dynamics) =
            dynamics_clock(&dynamics, &curve, turning.map(|_| half_period))?;
        Ok(LinebPath {
            cartesian: self.cartesian_images(&weighted)?,
            energy: self.energy,
            discrete_action,
            action,
            half_period,
            half_period_dynamics,
            turning,
            temperature: path::temperature(half_period, self.config.hbar, self.config.boltzmann)?,
            temperature_dynamics,
            calls: LinebCalls {
                seed: self.seed_evaluations,
                seed_in_optimizer: self.seed_evaluations,
                preparation,
                optimizer: self.evaluations.load(Ordering::Relaxed),
                measure: measure_calls.load(Ordering::Relaxed),
                dynamics: dynamics_calls.load(Ordering::Relaxed),
            },
            weighted,
            time_table,
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
    fn apply(self, drift: &mut Array1<f64>) {
        let (length, limit) = match self {
            Self::None => return,
            Self::Band(limit) => (action::l2(drift.view()), limit),
            Self::Atom(limit) => {
                let largest = drift
                    .exact_chunks(3)
                    .into_iter()
                    .map(action::l2)
                    .fold(0.0, f64::max);
                (largest, limit)
            }
        };
        if length > limit {
            *drift *= limit / length;
        }
    }

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
