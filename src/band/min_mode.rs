//! Min-mode refinement of the climbing image (OCI-NEB).
//!
//! Goswami, Gunde, and Jónsson, Front. Chem. 14, 1807063 (2026),
//! doi:10.3389/fchem.2026.1807063, Algorithm 1. This is the one
//! implementation: [`crate::ocineb::OcinebSession`] runs it with the
//! published controls. Once the climbing image has kept its index for
//! `stable_steps` band steps and its largest per-atom force is under
//! the trigger, that image alone climbs its lowest mode while the
//! other images stay where they are. The first refinement step only
//! estimates the mode, seeded with the band tangent (or the mode cached
//! by the last successful refinement), and the climb starts when the
//! curvature is negative and `|v · τ| >= alignment_tol`. A climb
//! continues while that overlap holds. A positive curvature restores
//! the image to where the refinement began unless an aligned point was
//! visited. An alignment failure, a positive curvature after such a
//! point, or the step budget leaves the image at the visited aligned
//! point of lowest force. The trigger thresholds are
//! [`crate::ocineb::initial_threshold`], and a refinement re-arms at
//! [`crate::ocineb::success_threshold`] (eq. 15) when it lowered the
//! force and at [`crate::ocineb::backoff_threshold`] (eqs. 13-14)
//! otherwise.
//!
//! The lowest mode is Lanczos with the improved relative stop of
//! Olsen, Kroes, Henkelman, Arnaldsson, and Jónsson, J. Chem. Phys.
//! 121, 9776 (2004), doi:10.1063/1.1809574, or the dimer. The climb is
//! the order-1 P-RFO step on a dense TS-BFGS model (`climb`). The model
//! starts at `h_0 I`, `h_0` the finite-difference curvature along the
//! force perpendicular to the mode, and carries the estimated curvature
//! on the mode with every other direction a minimum. The mode is
//! estimated again, and the model takes its curvature, when the force
//! grows past `refresh_factor` times its lowest value since the last
//! estimate or when the model's lowest mode turns away from the
//! previous one.

use ndarray::{Array1, ArrayView1, Axis, s};

use super::climb::ModelClimb;
use super::{BandReport, BandSession, BandStatus, BandSurface};
use crate::error::SaddleError;
use crate::minmode::{
    FiniteDifference, MinModeConfig, MinModeKind, MinModeSession, ModeEstimate, PointSurface,
    translated_gradient,
};
use crate::ocineb::{
    backoff_threshold, initial_threshold, mode_alignment, should_trigger, success_threshold,
    walk_helped,
};
use crate::tangent::compute_tangent;

use std::sync::atomic::{AtomicUsize, Ordering};

/// Overlap with the previous model mode under which the model's lowest
/// mode counts as a different direction.
const MODE_TURN: f64 = 0.7;

/// OCI-NEB controls. The trigger, stability count and alignment are
/// looser than the published `λ_rel = 0.31`, `κ = 5`, `α_tol = 0.85`:
/// the P-RFO climb follows its own model mode, so it starts from a
/// rougher band and keeps going while the mode turns. The budget is
/// 1000 steps per refinement. The Lanczos estimate keeps eight actions
/// and stops when the lowest Ritz value changes by under 5%.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct MinModeRefine {
    /// `λ_rel`: a refinement waits until the climbing image's largest
    /// per-atom force is under this fraction of the largest per-atom
    /// band force of the first evaluation.
    pub trigger_factor: f64,
    /// Absolute trigger on the largest per-atom force, read only when
    /// `trigger_factor` is 0.
    pub trigger_force: f64,
    /// `α_tol`: the climb starts and continues while the lowest mode
    /// keeps `|v · τ|` at or above this.
    pub alignment_tol: f64,
    /// `κ`: band steps the same image must keep its index.
    pub stable_steps: usize,
    /// Climb steps of one refinement before the band resumes.
    pub max_steps: usize,
    /// Put the image back where the refinement began when the
    /// refinement did not lower its force.
    pub restore_unhelpful: bool,
    pub kind: MinModeKind,
    pub krylov_dim: usize,
    /// Relative change of the lowest Ritz value that ends a Lanczos
    /// estimate.
    pub lanczos_rel_tol: f64,
    /// Finite-difference displacement of the Hessian actions and of the
    /// `h_0` probe.
    pub dr: f64,
    pub difference: FiniteDifference,
    /// Dimer rotation stop and budget.
    pub rotation_tol: f64,
    pub max_rotations: usize,
    /// Growth of the force over its lowest value since the last mode
    /// estimate that triggers a new one.
    pub refresh_factor: f64,
}

impl Default for MinModeRefine {
    fn default() -> Self {
        Self {
            trigger_factor: 0.5,
            trigger_force: 0.0,
            alignment_tol: 0.3,
            stable_steps: 3,
            max_steps: 1000,
            restore_unhelpful: false,
            kind: MinModeKind::Lanczos,
            krylov_dim: 8,
            lanczos_rel_tol: 0.05,
            dr: 1e-3,
            difference: FiniteDifference::Forward,
            rotation_tol: 1e-4,
            max_rotations: 20,
            refresh_factor: 1.5,
        }
    }
}

impl MinModeRefine {
    pub(super) fn validate(&self) -> Result<(), SaddleError> {
        let positive = |v: f64| v.is_finite() && v > 0.0;
        let non_negative = |v: f64| v.is_finite() && v >= 0.0;
        if !(non_negative(self.trigger_factor)
            && non_negative(self.trigger_force)
            && positive(self.dr)
            && positive(self.lanczos_rel_tol)
            && non_negative(self.rotation_tol)
            && self.refresh_factor.is_finite()
            && self.refresh_factor >= 1.0
            && self.alignment_tol.is_finite()
            && (0.0..=1.0).contains(&self.alignment_tol)
            && self.max_steps > 0
            && self.max_rotations > 0
            && self.krylov_dim > 0)
        {
            return Err(SaddleError::Invalid(
                "min-mode refinement needs a non-negative trigger, a positive step, \
                 tolerance, budget and Krylov size, a refresh factor of at least 1, \
                 and an alignment in [0, 1]"
                    .into(),
            ));
        }
        Ok(())
    }
}

/// What the next band step does while the refinement is on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RefinePhase {
    Band,
    Align,
    Climb,
}

/// The last refinement step, for hosts that report phases.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RefineStep {
    pub align: bool,
    pub alignment: f64,
    pub rotations: usize,
    /// The refinement ended on a positive curvature or a mode that left
    /// the tangent cone.
    pub fell_back: bool,
}

#[derive(Clone)]
struct Point {
    x: Array1<f64>,
    energy: f64,
    gradient: Array1<f64>,
}

struct Run {
    image: usize,
    saved: Point,
    entry_force: f64,
    tangent: Array1<f64>,
    session: MinModeSession,
    climb: Option<ModelClimb>,
    current: Point,
    /// Position and gradient before the last climb step.
    previous: Option<(Array1<f64>, Array1<f64>)>,
    best: Option<(f64, Point, Array1<f64>)>,
    alignment: f64,
    steps: usize,
    last_mode: Option<Array1<f64>>,
    /// Lowest force since the last mode estimate.
    refresh_gate: f64,
    since_refresh: usize,
}

impl Run {
    fn estimate<P: PointSurface>(&mut self, surface: &P) -> Result<ModeEstimate, SaddleError> {
        self.session
            .set_position(self.current.x.clone(), Some(self.current.gradient.clone()))?;
        self.session.estimate_mode(surface)
    }

    /// The `h_0` probe along the force perpendicular to `mode`, and the
    /// model that carries `curvature` on `mode`.
    fn start_climb<P: PointSurface>(
        &mut self,
        surface: &P,
        mode: &Array1<f64>,
        curvature: f64,
        dr: f64,
        max_move: f64,
    ) -> Result<(), SaddleError> {
        let g = &self.current.gradient;
        let mut d = g - &(mode * mode.dot(g));
        let norm = d.dot(&d).sqrt();
        let mut h0 = 1.0;
        if norm > 0.0 {
            d /= norm;
            let probe = &self.current.x + &(&d * dr);
            let (_, gp) = surface.eval(probe.view())?;
            h0 = ((&gp - g).dot(&d) / dr).abs().max(1e-3);
        }
        let mut climb = ModelClimb::new(g.len(), h0, max_move);
        climb.set_mode(mode, curvature)?;
        self.climb = Some(climb);
        Ok(())
    }
}

/// Trigger, stability latch, cached mode and the running refinement.
#[derive(Default)]
pub(super) struct RefineState {
    threshold: Option<f64>,
    stable: usize,
    prev_ci: Option<usize>,
    cached_mode: Option<Array1<f64>>,
    run: Option<Run>,
    converged: bool,
    last: Option<RefineStep>,
}

/// One band image as a point surface.
struct ImageSurface<'a, S: BandSurface + ?Sized> {
    surface: &'a S,
    image: usize,
    rows: &'a AtomicUsize,
}

impl<S: BandSurface + ?Sized> PointSurface for ImageSurface<'_, S> {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        self.rows.fetch_add(1, Ordering::Relaxed);
        let (energy, gradient) = self.surface.eval_image(self.image, x)?;
        if gradient.len() != x.len() {
            return Err(SaddleError::Shape(
                "image gradient length does not match the position".into(),
            ));
        }
        if !energy.is_finite() || !gradient.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("climbing image"));
        }
        Ok((energy, gradient))
    }
}

enum Outcome {
    /// Back to where the refinement began, after a positive curvature.
    Restore,
    /// The best aligned point, or the current one.
    Leave,
}

impl BandSession {
    pub(crate) fn refine_phase(&self) -> RefinePhase {
        match &self.refine.run {
            None => RefinePhase::Band,
            Some(run) if run.climb.is_none() => RefinePhase::Align,
            Some(_) => RefinePhase::Climb,
        }
    }

    pub(crate) fn refine_threshold(&self) -> Option<f64> {
        self.refine.threshold
    }

    pub(crate) fn refine_stability(&self) -> usize {
        self.refine.stable
    }

    pub(crate) fn refine_last(&self) -> Option<RefineStep> {
        self.refine.last
    }

    /// After a band step: count the steps the climbing image held and
    /// start a refinement when the latch and the trigger allow one.
    pub(super) fn refine_latch(&mut self, status: BandStatus) -> Result<(), SaddleError> {
        let ci = self.state.ci();
        let refine = &mut self.refine;
        if ci.is_some() && ci == refine.prev_ci {
            refine.stable += 1;
        } else {
            refine.stable = 0;
            refine.cached_mode = None;
        }
        refine.prev_ci = ci;
        let Some(config) = self.config.min_mode else {
            return Ok(());
        };
        if self.config.climbing.is_none() {
            return Ok(());
        }
        if self.refine.threshold.is_none()
            && let Some(f0) = self.state.atom_baseline()
        {
            self.refine.threshold = Some(initial_threshold(
                f0,
                config.trigger_factor,
                config.trigger_force,
                self.config.force_tol,
            ));
        }
        if status == BandStatus::Running && self.refine.run.is_none() {
            self.refine_begin(&config)?;
        }
        Ok(())
    }

    /// One refinement step, or `None` when the band steps instead.
    pub(super) fn refine_step<S: BandSurface>(
        &mut self,
        surface: &S,
    ) -> Result<Option<BandReport>, SaddleError> {
        self.refine.last = None;
        let Some(config) = self.config.min_mode else {
            return Ok(None);
        };
        if self.refine.converged {
            return Ok(Some(self.refine_report(BandStatus::Converged, None)));
        }
        let Some(run) = self.refine.run.as_ref() else {
            return Ok(None);
        };
        self.state.rows.store(0, Ordering::Relaxed);
        if run.climb.is_none() {
            return self.refine_align(&config, surface).map(Some);
        }
        let max_move = self.config.max_move;
        let force_gate = self.state.force_gate;
        let run = self.refine.run.as_mut().expect("refinement run");
        let image_surface = ImageSurface {
            surface,
            image: run.image,
            rows: &self.state.rows,
        };
        let gate = force_gate.value(run.current.gradient.view());
        let mut measured = None;
        let mut rotations = 0;
        if let Some((x, g)) = run.previous.take() {
            let climb = run.climb.as_mut().expect("climb model");
            climb.update(
                &(&run.current.x - &x),
                &(&run.current.gradient - &g),
                run.current.energy,
            );
            run.refresh_gate = run.refresh_gate.min(gate);
            run.since_refresh += 1;
            let (lowest, _) = climb.lowest()?;
            let turned = run
                .last_mode
                .as_ref()
                .is_some_and(|m| m.dot(&lowest).abs() < MODE_TURN);
            let grew = gate > config.refresh_factor * run.refresh_gate && run.since_refresh >= 2;
            if turned || grew {
                let estimate = run.estimate(&image_surface)?;
                run.climb
                    .as_mut()
                    .expect("climb model")
                    .set_mode(&estimate.mode, estimate.curvature)?;
                run.refresh_gate = gate;
                run.since_refresh = 0;
                rotations = estimate.rotations;
                measured = Some(estimate.curvature);
            }
        }
        let climb = run.climb.as_mut().expect("climb model");
        let (mode, model_curvature) = climb.lowest()?;
        let curvature = measured.unwrap_or(model_curvature);
        run.alignment = mode_alignment(mode.view(), run.tangent.view());
        run.last_mode = Some(mode.clone());
        let alignment = run.alignment;
        let step = RefineStep {
            align: false,
            alignment,
            rotations,
            fell_back: false,
        };
        self.iteration += 1;
        if curvature > 0.0 {
            let outcome = if run.best.is_some() {
                Outcome::Leave
            } else {
                Outcome::Restore
            };
            self.refine_finish(&config, outcome);
            return Ok(Some(self.refine_stop(
                RefineStep {
                    fell_back: true,
                    ..step
                },
                BandStatus::Running,
                curvature,
            )));
        }
        if alignment >= config.alignment_tol && run.best.as_ref().is_none_or(|(f, _, _)| gate < *f)
        {
            run.best = Some((gate, run.current.clone(), mode.clone()));
        }
        if gate <= self.config.force_tol {
            self.refine.converged = true;
            return Ok(Some(self.refine_stop(
                step,
                BandStatus::Converged,
                curvature,
            )));
        }
        if alignment < config.alignment_tol || run.steps >= config.max_steps {
            let fell_back = alignment < config.alignment_tol;
            self.refine_finish(&config, Outcome::Leave);
            return Ok(Some(self.refine_stop(
                RefineStep { fell_back, ..step },
                BandStatus::Running,
                curvature,
            )));
        }
        let delta = climb.step(&run.current.gradient, run.current.energy, max_move)?;
        let next = &run.current.x + &delta;
        let (energy, gradient) = image_surface.eval(next.view())?;
        run.steps += 1;
        run.previous = Some((
            std::mem::replace(&mut run.current.x, next),
            std::mem::replace(&mut run.current.gradient, gradient),
        ));
        run.current.energy = energy;
        let image = run.image;
        let point = run.current.clone();
        let projected = -translated_gradient(&point.gradient, &mode, curvature);
        self.put_image(image, &point, projected.view());
        Ok(Some(self.refine_stop(step, BandStatus::Running, curvature)))
    }

    /// The first refinement step: estimate the mode at the climbing
    /// image and start the climb when it is a negative, aligned mode.
    /// The image does not move.
    fn refine_align<S: BandSurface>(
        &mut self,
        config: &MinModeRefine,
        surface: &S,
    ) -> Result<BandReport, SaddleError> {
        let max_move = self.config.max_move;
        let force_tol = self.config.force_tol;
        let gate_of = self.state.force_gate;
        let run = self.refine.run.as_mut().expect("refinement run");
        let image_surface = ImageSurface {
            surface,
            image: run.image,
            rows: &self.state.rows,
        };
        let estimate = run.estimate(&image_surface)?;
        let alignment = mode_alignment(estimate.mode.view(), run.tangent.view());
        run.alignment = alignment;
        let gate = gate_of.value(run.current.gradient.view());
        let step = RefineStep {
            align: true,
            alignment,
            rotations: estimate.rotations,
            fell_back: true,
        };
        self.iteration += 1;
        if estimate.curvature > 0.0 {
            self.refine_finish(config, Outcome::Restore);
            return Ok(self.refine_stop(step, BandStatus::Running, estimate.curvature));
        }
        if alignment < config.alignment_tol {
            self.refine_finish(config, Outcome::Leave);
            return Ok(self.refine_stop(step, BandStatus::Running, estimate.curvature));
        }
        let step = RefineStep {
            fell_back: false,
            ..step
        };
        run.best = Some((gate, run.current.clone(), estimate.mode.clone()));
        if gate <= force_tol {
            self.refine.converged = true;
            return Ok(self.refine_stop(step, BandStatus::Converged, estimate.curvature));
        }
        run.start_climb(
            &image_surface,
            &estimate.mode,
            estimate.curvature,
            config.dr,
            max_move,
        )?;
        run.refresh_gate = gate;
        run.since_refresh = 0;
        Ok(self.refine_stop(step, BandStatus::Running, estimate.curvature))
    }

    /// Start a refinement when the latch and the trigger allow one.
    fn refine_begin(&mut self, config: &MinModeRefine) -> Result<(), SaddleError> {
        let Some(image) = self.state.ci() else {
            return Ok(());
        };
        let Some(threshold) = self.refine.threshold else {
            return Ok(());
        };
        let x = self.interior_flat();
        let (energies, gradient) = {
            let slot = self.state.last.lock().unwrap_or_else(|e| e.into_inner());
            let Some(last) = slot.as_ref().filter(|l| l.x == x) else {
                return Ok(());
            };
            (last.energies.clone(), last.gradients.row(image).to_owned())
        };
        let entry_force = max_atom(gradient.view());
        if !should_trigger(
            true,
            Some(image),
            self.positions.nrows() - 2,
            self.refine.stable as i64,
            config.stable_steps as i64 - 1,
            entry_force,
            threshold,
        ) {
            return Ok(());
        }
        let tangent = self.image_tangent(image, &energies)?;
        let seed = self
            .refine
            .cached_mode
            .clone()
            .unwrap_or_else(|| tangent.clone());
        let position = self.positions.row(image).to_owned();
        let mode_config = MinModeConfig {
            kind: config.kind,
            dr: config.dr,
            difference: config.difference,
            rotation_tol: config.rotation_tol,
            max_rotations: config.max_rotations,
            krylov_dim: config.krylov_dim,
            force_tol: self.config.force_tol,
            max_move: self.config.max_move,
            ..MinModeConfig::default()
        };
        let mut session = MinModeSession::new(mode_config, position.clone(), seed)?;
        if config.kind == MinModeKind::Lanczos {
            session.set_lanczos_rel_tol(Some(config.lanczos_rel_tol))?;
        }
        session.set_force_gate(self.state.force_gate);
        let entry = Point {
            x: position,
            energy: energies[image],
            gradient,
        };
        self.refine.run = Some(Run {
            image,
            saved: entry.clone(),
            entry_force,
            tangent,
            session,
            climb: None,
            current: entry,
            previous: None,
            best: None,
            alignment: 1.0,
            steps: 0,
            last_mode: None,
            refresh_gate: f64::INFINITY,
            since_refresh: 0,
        });
        Ok(())
    }

    /// End the refinement, place the image, re-arm the trigger, and
    /// hand the band its optimizer back without history.
    fn refine_finish(&mut self, config: &MinModeRefine, outcome: Outcome) {
        let Some(run) = self.refine.run.take() else {
            return;
        };
        let f0 = self.state.atom_baseline().unwrap_or(run.entry_force);
        let force_tol = self.config.force_tol;
        let backoff = |alpha: f64| backoff_threshold(alpha, f0, config.trigger_factor, force_tol);
        match outcome {
            Outcome::Restore => {
                self.restore_image(&run);
                self.refine.cached_mode = None;
                self.refine.threshold = Some(backoff(0.0));
            }
            Outcome::Leave => {
                let (point, mode) = match run.best {
                    Some((_, ref point, ref mode)) => (point.clone(), Some(mode.clone())),
                    None => (run.current.clone(), None),
                };
                let f_new = max_atom(point.gradient.view());
                let helped = walk_helped(f_new, run.entry_force, false);
                if helped {
                    let projected = -&point.gradient;
                    self.put_image(run.image, &point, projected.view());
                    self.refine.threshold = Some(success_threshold(
                        run.entry_force,
                        f_new,
                        f0,
                        config.trigger_factor,
                        force_tol,
                    ));
                    self.refine.cached_mode = mode;
                } else {
                    if config.restore_unhelpful {
                        self.restore_image(&run);
                        self.refine.cached_mode = None;
                    } else {
                        let projected = -&point.gradient;
                        self.put_image(run.image, &point, projected.view());
                    }
                    self.refine.threshold = Some(backoff(run.alignment));
                }
            }
        }
        self.refine.stable = 0;
        self.forget_optimizer();
    }

    fn restore_image(&mut self, run: &Run) {
        let projected = -&run.saved.gradient;
        self.put_image(run.image, &run.saved, projected.view());
    }

    /// Write one image and its surface answer into the band and its
    /// last evaluation. The other rows keep their cached values.
    fn put_image(&mut self, image: usize, point: &Point, projected: ArrayView1<f64>) {
        self.positions.row_mut(image).assign(&point.x);
        let x = self.interior_flat();
        let dof = self.positions.ncols();
        let mut slot = self.state.last.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(last) = slot.as_mut() {
            last.x = x;
            last.energies[image] = point.energy;
            last.gradients.row_mut(image).assign(&point.gradient);
            if last.projected.len() == (self.positions.nrows() - 2) * dof {
                last.projected
                    .slice_mut(s![(image - 1) * dof..image * dof])
                    .assign(&projected);
            }
        }
    }

    fn image_tangent(
        &self,
        image: usize,
        energies: &Array1<f64>,
    ) -> Result<Array1<f64>, SaddleError> {
        let pos = self.positions.row(image);
        let mut next = &self.positions.row(image + 1) - &pos;
        let mut prev = &pos - &self.positions.row(image - 1);
        if let Some(cell) = &self.config.cell {
            cell.minimum_image(&mut next);
            cell.minimum_image(&mut prev);
        }
        compute_tangent(
            self.config.tangent,
            next.view(),
            prev.view(),
            energies[image],
            energies[image - 1],
            energies[image + 1],
        )
        .map_err(|e| SaddleError::Invalid(format!("image {image}: {e}")))
    }

    fn refine_stop(&mut self, step: RefineStep, status: BandStatus, curvature: f64) -> BandReport {
        self.refine.last = Some(step);
        self.refine_report(status, Some(curvature))
    }

    fn refine_report(&self, status: BandStatus, curvature: Option<f64>) -> BandReport {
        let image = self.state.ci();
        let max_force = image
            .and_then(|i| {
                let slot = self.state.last.lock().unwrap_or_else(|e| e.into_inner());
                slot.as_ref()
                    .map(|last| self.state.force_gate.value(last.gradients.row(i)))
            })
            .unwrap_or(f64::INFINITY);
        let rows = if status == BandStatus::Converged && curvature.is_none() {
            0
        } else {
            self.state.rows.load(Ordering::Relaxed)
        };
        BandReport {
            status,
            max_force,
            ci_index: image,
            iteration: self.iteration,
            surface_rows: rows,
            curvature,
        }
    }
}

/// Largest per-atom norm of a flat `3N` vector.
pub(super) fn max_atom(v: ArrayView1<f64>) -> f64 {
    v.exact_chunks(3)
        .into_iter()
        .map(|a| a.dot(&a).sqrt())
        .fold(0.0, f64::max)
}

/// The default one-row evaluation behind [`BandSurface::eval_image`].
pub(super) fn one_row<S: BandSurface + ?Sized>(
    surface: &S,
    x: ArrayView1<f64>,
) -> Result<(f64, Array1<f64>), SaddleError> {
    let row = x.insert_axis(Axis(0));
    let mut energies = Array1::zeros(1);
    let mut gradients = ndarray::Array2::zeros((1, x.len()));
    surface.eval(row, &mut energies, &mut gradients)?;
    Ok((energies[0], gradients.row(0).to_owned()))
}
