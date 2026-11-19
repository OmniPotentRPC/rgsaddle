//! Fixed-cell relaxation to an explicit force tolerance.
//!
//! A quantity read off a relaxed configuration (a defect's identity, its
//! stress, a pair's separation) must not depend on how the minimizer
//! travelled there. [`relax_to_tolerance`] therefore stops only when the
//! largest per-atom force norm, `max_i |F_i|`, is below
//! [`RelaxConfig::force_tol`]. The step cap [`RelaxConfig::max_steps`] is
//! a guard against a run that does not settle: reaching it is an error,
//! never a result.
//!
//! The minimizer is any [`rgmin::Method`] that a [`rgmin::Solver`]
//! session steps. A converged end state is a minimum, but which minimum
//! still depends on the path when the start lies near a basin boundary:
//! FIRE carries momentum and L-BFGS extrapolates, so from one start they
//! can settle in different basins, each below the tolerance. When the
//! basin itself is the result, follow the steepest-descent flow with
//! [`rgmin::Method::Ode12r`] and check it against a second integration of
//! the flow, such as [`rgmin::Method::Steepest`] under a smaller per-atom
//! cap.

use ndarray::{Array1, ArrayView1};
use rgmin::{Control, FireKind, Method, Oracle, Solver, StopNorm};

use crate::cell_log::{M3, m3_to_row9};
use crate::error::SaddleError;
use crate::minmode::PointSurface;

/// Default largest per-atom force norm, in the surface's force units
/// (eV/Å for a surface in eV and Å).
pub const DEFAULT_FORCE_TOL: f64 = 1.0e-3;

/// Default step cap. Reaching it is reported as non-convergence.
pub const DEFAULT_MAX_STEPS: usize = 10_000;

/// Controls for [`relax_to_tolerance`].
///
/// Start from [`RelaxConfig::default`] and chain the `with_*` setters.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct RelaxConfig {
    /// Minimizer. FIRE 2.0 by default.
    pub method: Method,
    /// Largest per-atom force norm accepted as converged.
    pub force_tol: f64,
    /// Most solver steps before the run is reported as not converged.
    pub max_steps: usize,
    /// Largest displacement of one atom in one step, in length units.
    pub max_atom_move: f64,
    /// FIRE time step, and the opening line-search step of the
    /// line-search methods.
    pub initial_step: f64,
}

impl Default for RelaxConfig {
    /// FIRE 2.0, [`DEFAULT_FORCE_TOL`], [`DEFAULT_MAX_STEPS`], 0.1 per-atom
    /// cap and 0.1 initial step.
    fn default() -> Self {
        Self {
            method: Method::fire(FireKind::V2),
            force_tol: DEFAULT_FORCE_TOL,
            max_steps: DEFAULT_MAX_STEPS,
            max_atom_move: 0.1,
            initial_step: 0.1,
        }
    }
}

with_setters!(RelaxConfig {
    with_method => method: val Method,
    with_force_tol => force_tol: val f64,
    with_max_steps => max_steps: val usize,
    with_max_atom_move => max_atom_move: val f64,
    with_initial_step => initial_step: val f64,
});

impl RelaxConfig {
    pub(crate) fn check(&self) -> Result<(), SaddleError> {
        if !(self.force_tol.is_finite() && self.force_tol > 0.0) {
            return Err(SaddleError::Invalid(
                "relaxation force tolerance must be finite and positive".into(),
            ));
        }
        if self.max_steps == 0 {
            return Err(SaddleError::Invalid(
                "relaxation step cap must be at least one".into(),
            ));
        }
        if !(self.max_atom_move.is_finite() && self.max_atom_move > 0.0) {
            return Err(SaddleError::Invalid(
                "per-atom step cap must be finite and positive".into(),
            ));
        }
        if !(self.initial_step.is_finite() && self.initial_step > 0.0) {
            return Err(SaddleError::Invalid(
                "initial step must be finite and positive".into(),
            ));
        }
        Ok(())
    }
}

/// A configuration whose largest per-atom force is below the tolerance.
#[derive(Clone, Debug)]
pub struct Relaxation {
    /// Relaxed Cartesian positions.
    pub positions: Array1<f64>,
    /// Energy at [`Self::positions`].
    pub energy: f64,
    /// Largest per-atom force norm at [`Self::positions`].
    pub max_force: f64,
    /// Solver steps taken.
    pub steps: usize,
}

/// Largest Euclidean norm over the `(x, y, z)` triples of `gradient`.
pub fn max_atom_force(gradient: ArrayView1<f64>) -> f64 {
    StopNorm::MaxAtom.of(gradient)
}

/// Relax `positions` in the fixed `cell` until the largest per-atom force
/// norm is below `config.force_tol`.
///
/// The force is measured on the returned positions, after the last step.
/// When `config.max_steps` steps leave it at or above the tolerance, or
/// the minimizer stops early on another test, the result is
/// [`SaddleError::Solver`] naming the step count and the force reached.
pub fn relax_to_tolerance<S: PointSurface>(
    surface: &S,
    positions: Array1<f64>,
    cell: M3,
    config: &RelaxConfig,
) -> Result<Relaxation, SaddleError> {
    config.check()?;
    if positions.is_empty() || !positions.len().is_multiple_of(3) {
        return Err(SaddleError::Shape(
            "positions must hold whole (x, y, z) triples".into(),
        ));
    }
    let flat = m3_to_row9(cell);
    let (energy, gradient) = surface.eval_in_cell(positions.view(), &flat)?;
    if !energy.is_finite() {
        return Err(SaddleError::NonFinite("relaxation start energy"));
    }
    let start_force = max_atom_force(gradient.view());
    if start_force < config.force_tol {
        return Ok(Relaxation {
            positions,
            energy,
            max_force: start_force,
            steps: 0,
        });
    }
    let mut solver = Solver::new(
        config.method.clone(),
        Control::default()
            .with_maxiter(config.max_steps)
            .with_gtol(config.force_tol)
            .with_stop_norm(StopNorm::MaxAtom)
            .with_istep(config.initial_step),
        positions.len(),
    );
    solver.set_atom_maxmove(config.max_atom_move);
    let oracle = Oracle::unbounded(positions.len(), |x: ArrayView1<f64>| {
        surface
            .eval_in_cell(x, &flat)
            .unwrap_or_else(|_| (f64::NAN, Array1::from_elem(x.len(), f64::NAN)))
    });
    let mut positions = positions;
    let report = solver
        .run(&oracle, &mut positions)
        .map_err(|error| SaddleError::Solver(error.to_string()))?;
    let (energy, gradient) = surface.eval_in_cell(positions.view(), &flat)?;
    let max_force = max_atom_force(gradient.view());
    if !(energy.is_finite() && max_force < config.force_tol) {
        return Err(SaddleError::Solver(format!(
            "relaxation not converged: largest atom force {max_force:.3e} is not below {:.1e} after {} of {} steps ({:?})",
            config.force_tol, report.steps, config.max_steps, report.status
        )));
    }
    Ok(Relaxation {
        positions,
        energy,
        max_force,
        steps: report.steps,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Independent harmonic wells about `centre` with stiffness `k`.
    struct Wells {
        centre: Array1<f64>,
        k: Array1<f64>,
    }

    impl PointSurface for Wells {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let d = &x - &self.centre;
            let g = &d * &self.k;
            Ok((0.5 * d.dot(&g), g))
        }
    }

    fn wells() -> (Wells, Array1<f64>) {
        let centre = Array1::from(vec![0.0, 0.1, -0.2, 1.0, 1.1, 0.9]);
        let k = Array1::from(vec![1.0, 4.0, 9.0, 2.0, 0.5, 6.0]);
        let start = Array1::from(vec![0.4, -0.3, 0.1, 1.6, 0.2, 1.3]);
        (Wells { centre, k }, start)
    }

    fn cell() -> M3 {
        [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
    }

    #[test]
    fn fire_and_lbfgs_reach_the_same_minimum_below_the_tolerance() {
        let (surface, start) = wells();
        let tol = 1.0e-6;
        let fire = relax_to_tolerance(
            &surface,
            start.clone(),
            cell(),
            &RelaxConfig::default().with_force_tol(tol),
        )
        .unwrap();
        let lbfgs = relax_to_tolerance(
            &surface,
            start,
            cell(),
            &RelaxConfig::default()
                .with_force_tol(tol)
                .with_method(Method::lbfgs()),
        )
        .unwrap();
        assert!(fire.max_force < tol && lbfgs.max_force < tol);
        for ((a, b), c) in fire
            .positions
            .iter()
            .zip(lbfgs.positions.iter())
            .zip(surface.centre.iter())
        {
            assert!(
                (a - c).abs() < 1.0e-5 && (b - c).abs() < 1.0e-5,
                "{a} {b} {c}"
            );
        }
    }

    #[test]
    fn the_step_cap_is_an_error_not_a_result() {
        let (surface, start) = wells();
        let err = relax_to_tolerance(
            &surface,
            start,
            cell(),
            &RelaxConfig::default()
                .with_force_tol(1.0e-8)
                .with_max_steps(3),
        )
        .unwrap_err();
        assert!(
            matches!(err, SaddleError::Solver(ref m) if m.contains("not converged")),
            "{err}"
        );
    }

    #[test]
    fn a_converged_start_takes_no_step() {
        let (surface, _) = wells();
        let done = relax_to_tolerance(
            &surface,
            surface.centre.clone(),
            cell(),
            &RelaxConfig::default(),
        )
        .unwrap();
        assert_eq!(done.steps, 0);
        assert_eq!(done.max_force, 0.0);
    }

    #[test]
    fn invalid_controls_are_refused() {
        let (surface, start) = wells();
        for config in [
            RelaxConfig::default().with_force_tol(0.0),
            RelaxConfig::default().with_max_steps(0),
            RelaxConfig::default().with_max_atom_move(f64::NAN),
            RelaxConfig::default().with_initial_step(-1.0),
        ] {
            assert!(matches!(
                relax_to_tolerance(&surface, start.clone(), cell(), &config),
                Err(SaddleError::Invalid(_))
            ));
        }
    }

    #[test]
    fn max_atom_force_takes_the_largest_triple() {
        let g = Array1::from(vec![3.0, 4.0, 0.0, 0.0, 0.0, 6.0]);
        assert_eq!(max_atom_force(g.view()), 6.0);
    }
}
