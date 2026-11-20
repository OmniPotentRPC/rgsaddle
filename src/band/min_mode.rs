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
//! curvature is negative and `|v · τ| >= alignment_tol`. With
//! `climb_alignment` a climb continues only while that overlap holds.
//! A positive curvature restores the image to where the refinement
//! began unless an aligned point was visited. An alignment failure, a
//! positive curvature after such a point, or the step budget leaves the
//! image at the visited aligned point of lowest force. A positive
//! curvature of the model is measured again before it ends the climb.
//! The trigger thresholds are [`crate::ocineb::initial_threshold`], and
//! a refinement re-arms at [`crate::ocineb::success_threshold`]
//! (eq. 15) when it lowered the force and at
//! [`crate::ocineb::backoff_threshold`] (eqs. 13-14) otherwise, capped
//! by `retry_factor` times the force it started from. Those re-armed
//! values belong to the image that climbed: when the band's climbing
//! image changes index the threshold returns to its initial value. A
//! refinement that leaves the image where it began keeps the band
//! optimizer's history.
//!
//! The lowest mode is Lanczos with the improved relative stop of
//! Olsen, Kroes, Henkelman, Arnaldsson, and Jónsson, J. Chem. Phys.
//! 121, 9776 (2004), doi:10.1063/1.1809574, or the dimer. The climb is
//! the order-1 P-RFO step on a dense TS-BFGS model (`climb`) fit to the
//! pair-stretch Hessian and the estimate's Krylov block, following the
//! estimated mode by overlap. The mode is estimated again, its block
//! imposed and its direction followed, once the followed curvature has
//! kept under `curvature_retained` of the last measured value or
//! the followed mode turned under `min_overlap`, at most every
//! `refresh_interval` climb steps.

use ndarray::{Array1, ArrayView1, Axis, s};

use super::climb::ModelClimb;
use super::pair_model::rigid_basis;
use super::{BandReport, BandSession, BandStatus, BandSurface, Cell, zero_fixed};
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

/// OCI-NEB controls. The defaults hand the climbing image off as soon
/// as it has kept its index for one band step, at any force, where the
/// published controls wait for `λ_rel = 0.31` and `κ = 5`; the start
/// keeps the published `α_tol = 0.85`. The P-RFO climb follows its own
/// mode by overlap, so it does not end when that mode leaves the band
/// tangent, and a refinement that did not lower the force re-arms only
/// under half the force it started from. The budget is 1000 steps per
/// refinement. The Lanczos estimate keeps up to twenty actions, which
/// fit the climb model, and stops when the lowest Ritz value changes by
/// under 5%.
///
/// Non-exhaustive: start from [`MinModeRefine::default`] and chain the
/// `with_*` setters.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct MinModeRefine {
    /// `λ_rel`: a refinement waits until the climbing image's largest
    /// per-atom force is under this fraction of the largest per-atom
    /// band force of the first evaluation. Infinite hands the image off
    /// at any force; that needs `retry_factor`.
    pub trigger_factor: f64,
    /// Absolute trigger on the largest per-atom force, read only when
    /// `trigger_factor` is 0.
    pub trigger_force: f64,
    /// `α_tol`: the climb starts when the lowest mode has `|v · τ|` at
    /// or above this.
    pub alignment_tol: f64,
    /// The climb also ends once the followed mode's `|v · τ|` falls
    /// under `alignment_tol`.
    pub climb_alignment: bool,
    /// A refinement that did not lower the force also re-arms under this
    /// fraction of the force it started from. `None` keeps the
    /// published back-off alone.
    pub retry_factor: Option<f64>,
    /// `κ`: band steps the same image must keep its index.
    pub stable_steps: usize,
    /// Climb steps of one refinement before the band resumes.
    pub max_steps: usize,
    /// Put the image back where the refinement began when the
    /// refinement did not lower its force.
    pub restore_unhelpful: bool,
    /// Which estimator produces the lowest mode.
    pub kind: MinModeKind,
    /// Largest Krylov subspace used for one Lanczos estimate.
    pub krylov_dim: usize,
    /// Relative change of the lowest Ritz value that ends a Lanczos
    /// estimate.
    pub lanczos_rel_tol: f64,
    /// Finite-difference displacement of the Hessian actions, and of the
    /// probe that joins a dimer mode in the climb model.
    pub dr: f64,
    /// Forward or central differences for Hessian actions.
    pub difference: FiniteDifference,
    /// Dimer rotation stop and budget.
    pub rotation_tol: f64,
    /// Most dimer rotations, or Lanczos actions, in one estimate.
    pub max_rotations: usize,
    /// The mode is estimated again once the followed model curvature
    /// is above this fraction of the curvature last measured.
    pub curvature_retained: f64,
    /// The mode is estimated again once the followed mode's overlap with
    /// the one followed a step before is under this.
    pub min_overlap: f64,
    /// Climb steps after an estimate before the next one.
    pub refresh_interval: usize,
}

with_setters!(MinModeRefine {
    with_trigger_factor => trigger_factor: val f64,
    with_trigger_force => trigger_force: val f64,
    with_alignment_tol => alignment_tol: val f64,
    with_climb_alignment => climb_alignment: val bool,
    with_retry_factor => retry_factor: opt f64,
    with_stable_steps => stable_steps: val usize,
    with_max_steps => max_steps: val usize,
    with_restore_unhelpful => restore_unhelpful: val bool,
    with_kind => kind: val MinModeKind,
    with_krylov_dim => krylov_dim: val usize,
    with_lanczos_rel_tol => lanczos_rel_tol: val f64,
    with_dr => dr: val f64,
    with_difference => difference: val FiniteDifference,
    with_rotation_tol => rotation_tol: val f64,
    with_max_rotations => max_rotations: val usize,
    with_curvature_retained => curvature_retained: val f64,
    with_min_overlap => min_overlap: val f64,
    with_refresh_interval => refresh_interval: val usize,
});

impl Default for MinModeRefine {
    fn default() -> Self {
        Self {
            trigger_factor: f64::INFINITY,
            trigger_force: 0.0,
            alignment_tol: 0.85,
            climb_alignment: false,
            retry_factor: Some(0.5),
            stable_steps: 1,
            max_steps: 1000,
            restore_unhelpful: false,
            kind: MinModeKind::Lanczos,
            krylov_dim: 20,
            lanczos_rel_tol: 0.05,
            dr: 1e-3,
            difference: FiniteDifference::Forward,
            rotation_tol: 1e-4,
            max_rotations: 20,
            curvature_retained: 0.3,
            min_overlap: 0.8,
            refresh_interval: 4,
        }
    }
}

impl MinModeRefine {
    pub(super) fn validate(&self) -> Result<(), SaddleError> {
        let positive = |v: f64| v.is_finite() && v > 0.0;
        let non_negative = |v: f64| v.is_finite() && v >= 0.0;
        let fraction = |v: f64| (0.0..=1.0).contains(&v);
        let retry = match self.retry_factor {
            Some(f) => positive(f),
            None => self.trigger_factor.is_finite(),
        };
        if !(self.trigger_factor >= 0.0
            && non_negative(self.trigger_force)
            && retry
            && positive(self.dr)
            && positive(self.lanczos_rel_tol)
            && non_negative(self.rotation_tol)
            && fraction(self.alignment_tol)
            && fraction(self.curvature_retained)
            && fraction(self.min_overlap)
            && self.max_steps > 0
            && self.max_rotations > 0
            && self.krylov_dim > 0)
        {
            return Err(SaddleError::Invalid(
                "min-mode refinement needs a non-negative trigger, a positive step, \
                 tolerance, budget and Krylov size, a positive retry factor with an \
                 infinite trigger, and the alignment, retained curvature and overlap \
                 in [0, 1]"
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
pub(super) struct Point {
    pub(super) x: Array1<f64>,
    pub(super) energy: f64,
    pub(super) gradient: Array1<f64>,
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
    /// Climb steps since the last mode estimate.
    since_refresh: usize,
    /// The climb starts off a saddle of higher index; its seed is the
    /// negative mode closest to the tangent, which need not meet
    /// `alignment_tol`.
    escape: bool,
}

impl Run {
    fn estimate<P: PointSurface>(&mut self, surface: &P) -> Result<ModeEstimate, SaddleError> {
        self.session
            .set_position(self.current.x.clone(), Some(self.current.gradient.clone()))?;
        self.session.estimate_mode(surface)
    }

    /// The Krylov block of the last estimate, or the estimated mode
    /// with its curvature when the estimate kept none.
    fn block(&self, estimate: &ModeEstimate) -> Vec<(Array1<f64>, Array1<f64>)> {
        let block = self.session.krylov_block();
        if block.is_empty() {
            vec![(estimate.mode.clone(), &estimate.mode * estimate.curvature)]
        } else {
            block.to_vec()
        }
    }

    /// The climb model fit to the estimate's block, off the `rigid`
    /// directions. Without a Krylov block (a dimer estimate) one Hessian
    /// action along the force perpendicular to the mode joins the mode.
    fn start_climb<P: PointSurface>(
        &mut self,
        surface: &P,
        estimate: &ModeEstimate,
        dr: f64,
        max_move: f64,
        cell: Option<&Cell>,
        rigid: Option<ndarray::Array2<f64>>,
    ) -> Result<(), SaddleError> {
        let mut block = self.block(estimate);
        if self.session.krylov_block().is_empty() {
            let g = &self.current.gradient;
            let mode = &estimate.mode;
            let mut d = g - &(mode * mode.dot(g));
            let norm = d.dot(&d).sqrt();
            if norm > 0.0 {
                d /= norm;
                let probe = &self.current.x + &(&d * dr);
                let (_, gp) = surface.eval(probe.view())?;
                block.push((d, (&gp - g) / dr));
            }
        }
        self.climb = Some(ModelClimb::fitted(
            &self.current.x,
            cell,
            rigid,
            &block,
            &estimate.mode,
            estimate.curvature,
            max_move,
        ));
        self.since_refresh = 0;
        Ok(())
    }

    /// Impose the estimate's block on the climb model and follow its mode.
    fn refresh(&mut self, estimate: &ModeEstimate) {
        let block = self.block(estimate);
        self.climb.as_mut().expect("climb model").refresh(
            &block,
            &estimate.mode,
            estimate.curvature,
        );
        self.since_refresh = 0;
    }
}

/// The rigid motions a band without its mean force removes: the
/// translations, and the rotations too when there is no cell. With
/// atoms fixed they are the fixed coordinates instead, so the climb
/// model, its steps and its secant pairs live on the free block.
pub(super) fn rigid_motions(
    x: &Array1<f64>,
    cell: Option<&Cell>,
    remove_translation: bool,
    fixed: &[bool],
) -> Option<ndarray::Array2<f64>> {
    if fixed.iter().any(|&f| f) {
        let held: Vec<usize> = fixed
            .iter()
            .enumerate()
            .filter(|(_, f)| **f)
            .flat_map(|(a, _)| 3 * a..3 * a + 3)
            .collect();
        let mut q = ndarray::Array2::zeros((x.len(), held.len()));
        for (j, &c) in held.iter().enumerate() {
            q[(c, j)] = 1.0;
        }
        return Some(q);
    }
    remove_translation
        .then(|| rigid_basis(x, cell.is_none()))
        .flatten()
}

/// Trigger, stability latch, cached mode and the running refinement.
#[derive(Default)]
pub(super) struct RefineState {
    threshold: Option<f64>,
    stable: usize,
    prev_ci: Option<usize>,
    cached_mode: Option<Array1<f64>>,
    run: Option<Run>,
    pub(super) converged: bool,
    last: Option<RefineStep>,
    /// An escape started a climb; it runs under the default controls
    /// when the band has none.
    escaped: bool,
}

/// One band image as a point surface.
pub(super) struct ImageSurface<'a, S: BandSurface + ?Sized> {
    pub(super) surface: &'a S,
    pub(super) image: usize,
    pub(super) rows: &'a AtomicUsize,
    /// The band's fixed mask; those gradient components read as zero.
    pub(super) fixed: &'a [bool],
}

impl<S: BandSurface + ?Sized> PointSurface for ImageSurface<'_, S> {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        self.rows.fetch_add(1, Ordering::Relaxed);
        let (energy, mut gradient) = self.surface.eval_image(self.image, x)?;
        if gradient.len() != x.len() {
            return Err(SaddleError::Shape(
                "image gradient length does not match the position".into(),
            ));
        }
        if !energy.is_finite() || !gradient.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("climbing image"));
        }
        zero_fixed(self.fixed, &mut gradient);
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

    /// The refinement controls: the band's own, or the defaults for a
    /// climb an escape started on a band without them.
    fn refine_config(&self) -> Option<MinModeRefine> {
        self.config
            .min_mode
            .or_else(|| self.refine.escaped.then(MinModeRefine::default))
    }

    /// The mode the refinement of `image` last estimated, if one runs.
    pub(super) fn refine_mode(&self, image: usize) -> Option<Array1<f64>> {
        self.refine
            .run
            .as_ref()
            .filter(|run| run.image == image)
            .map(|run| run.session.mode().to_owned())
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
            if refine.prev_ci.is_some() {
                refine.threshold = None;
            }
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
        if status == BandStatus::Running && self.refine.run.is_none() && !self.zoom_pending() {
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
        let Some(config) = self.refine_config() else {
            return Ok(None);
        };
        self.state.rows.store(0, Ordering::Relaxed);
        if self.refine.converged {
            self.refresh_exp(surface)?;
            return Ok(Some(self.refine_report(BandStatus::Converged, None)));
        }
        let Some(run) = self.refine.run.as_ref() else {
            return Ok(None);
        };
        if run.climb.is_none() {
            return self.refine_align(&config, surface).map(Some);
        }
        let max_move = self.config.max_move;
        let force_gate = self.state.force_gate;
        let cell = self.config.cell;
        let remove_translation = self.config.drops_translation();
        let run = self.refine.run.as_mut().expect("refinement run");
        let fixed = self.state.fixed.as_slice();
        let image_surface = ImageSurface {
            surface,
            image: run.image,
            rows: &self.state.rows,
            fixed,
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
            climb.set_rigid(rigid_motions(
                &run.current.x,
                cell.as_ref(),
                remove_translation,
                fixed,
            ));
            run.since_refresh += 1;
            climb.lowest()?;
            if run.since_refresh >= config.refresh_interval
                && climb.stale(config.curvature_retained, config.min_overlap)
            {
                let estimate = run.estimate(&image_surface)?;
                run.refresh(&estimate);
                rotations = estimate.rotations;
                measured = Some(estimate.curvature);
            }
        }
        let (mut mode, mut curvature) = run.climb.as_mut().expect("climb model").lowest()?;
        if let Some(c) = measured {
            curvature = c;
        } else if curvature > 0.0 {
            // Near convergence the TS-BFGS update on short steps can lift
            // the model's lowest eigenvalue over zero while the surface
            // curvature is still negative, so only a measured one ends
            // the climb.
            let estimate = run.estimate(&image_surface)?;
            run.refresh(&estimate);
            rotations = estimate.rotations;
            curvature = estimate.curvature;
            mode = run.climb.as_mut().expect("climb model").lowest()?.0;
        }
        let climb = run.climb.as_mut().expect("climb model");
        run.alignment = mode_alignment(mode.view(), run.tangent.view());
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
            self.refresh_exp(surface)?;
            return Ok(Some(self.refine_stop(
                RefineStep {
                    fell_back: true,
                    ..step
                },
                BandStatus::Running,
                curvature,
            )));
        }
        if (run.escape || alignment >= config.alignment_tol)
            && run.best.as_ref().is_none_or(|(f, _, _)| gate < *f)
        {
            run.best = Some((gate, run.current.clone(), mode.clone()));
        }
        if gate <= self.config.force_tol {
            self.refine.converged = true;
            self.refresh_exp(surface)?;
            return Ok(Some(self.refine_stop(
                step,
                BandStatus::Converged,
                curvature,
            )));
        }
        let turned = config.climb_alignment && !run.escape && alignment < config.alignment_tol;
        if turned || run.steps >= config.max_steps {
            let fell_back = turned;
            self.refine_finish(&config, Outcome::Leave);
            self.refresh_exp(surface)?;
            return Ok(Some(self.refine_stop(
                RefineStep { fell_back, ..step },
                BandStatus::Running,
                curvature,
            )));
        }
        let mut delta = climb.step(&run.current.gradient, run.current.energy, max_move)?;
        zero_fixed(fixed, &mut delta);
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
        let cell = self.config.cell;
        let remove_translation = self.config.drops_translation();
        let run = self.refine.run.as_mut().expect("refinement run");
        let fixed = self.state.fixed.as_slice();
        let image_surface = ImageSurface {
            surface,
            image: run.image,
            rows: &self.state.rows,
            fixed,
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
            self.refresh_exp(surface)?;
            return Ok(self.refine_stop(step, BandStatus::Running, estimate.curvature));
        }
        if alignment < config.alignment_tol && !run.escape {
            self.refine_finish(config, Outcome::Leave);
            self.refresh_exp(surface)?;
            return Ok(self.refine_stop(step, BandStatus::Running, estimate.curvature));
        }
        let step = RefineStep {
            fell_back: false,
            ..step
        };
        run.best = Some((gate, run.current.clone(), estimate.mode.clone()));
        if gate <= force_tol {
            self.refine.converged = true;
            self.refresh_exp(surface)?;
            return Ok(self.refine_stop(step, BandStatus::Converged, estimate.curvature));
        }
        let rigid = rigid_motions(&run.current.x, cell.as_ref(), remove_translation, fixed);
        run.start_climb(
            &image_surface,
            &estimate,
            config.dr,
            max_move,
            cell.as_ref(),
            rigid,
        )?;
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
        let session = self.refine_session(config, position.clone(), seed)?;
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
            since_refresh: 0,
            escape: false,
        });
        Ok(())
    }

    fn refine_session(
        &self,
        config: &MinModeRefine,
        position: Array1<f64>,
        seed: Array1<f64>,
    ) -> Result<MinModeSession, SaddleError> {
        let mode_config = MinModeConfig::default()
            .with_kind(config.kind)
            .with_dr(config.dr)
            .with_difference(config.difference)
            .with_rotation_tol(config.rotation_tol)
            .with_max_rotations(config.max_rotations)
            .with_krylov_dim(config.krylov_dim)
            .with_force_tol(self.config.force_tol)
            .with_max_move(self.config.max_move)
            .with_saddle_check(None);
        let mut session = MinModeSession::new(mode_config, position, seed)?;
        if config.kind == MinModeKind::Lanczos {
            session.set_lanczos_rel_tol(Some(config.lanczos_rel_tol))?;
        }
        session.set_force_gate(self.state.force_gate);
        Ok(session)
    }

    /// Climb `image` again from `start`, seeded with `reaction`, after a
    /// push off a saddle of higher index at `saved`. A climb that ends
    /// on a positive curvature puts the image back at `saved`.
    pub(super) fn refine_escape(
        &mut self,
        image: usize,
        start: Point,
        saved: Point,
        reaction: Array1<f64>,
        tangent: Array1<f64>,
    ) -> Result<(), SaddleError> {
        self.refine.escaped = true;
        let config = self.refine_config().unwrap_or_default();
        let session = self.refine_session(&config, start.x.clone(), reaction)?;
        let projected = -&start.gradient;
        self.put_image(image, &start, projected.view());
        self.forget_optimizer();
        self.refine.converged = false;
        self.refine.stable = 0;
        self.refine.run = Some(Run {
            image,
            saved,
            entry_force: max_atom(start.gradient.view()),
            tangent,
            session,
            climb: None,
            current: start,
            previous: None,
            best: None,
            alignment: 1.0,
            steps: 0,
            since_refresh: 0,
            escape: true,
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
        let retry = config.retry_factor.map_or(f64::INFINITY, |f| {
            (f * run.entry_force).max(2.0 * force_tol)
        });
        let backoff =
            |alpha: f64| backoff_threshold(alpha, f0, config.trigger_factor, force_tol).min(retry);
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
        // An image left where the refinement began leaves the band as the
        // optimizer last saw it.
        if self.positions.row(run.image) != run.saved.x {
            self.forget_optimizer();
        }
    }

    fn restore_image(&mut self, run: &Run) {
        let projected = -&run.saved.gradient;
        self.put_image(run.image, &run.saved, projected.view());
    }

    /// Write one image and its surface answer into the band and its
    /// last evaluation. The other rows keep their cached surface values;
    /// tangents, springs, climbing forces and metric solves do not.
    pub(super) fn put_image(&mut self, image: usize, point: &Point, projected: ArrayView1<f64>) {
        self.positions.row_mut(image).assign(&point.x);
        self.pin_fixed();
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
        drop(slot);
        self.state.invalidate_assembly();
        self.solver.forget_evaluation();
        self.invalidate_stale_exp();
    }

    pub(super) fn image_tangent(
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
        zero_fixed(&self.config.fixed_atoms, &mut next);
        zero_fixed(&self.config.fixed_atoms, &mut prev);
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
        let max_force = match image {
            Some(i) => {
                let slot = match self.state.last.lock() {
                    Ok(guard) => guard,
                    Err(poisoned) => poisoned.into_inner(),
                };
                match slot.as_ref() {
                    Some(last) => self.state.force_gate.value(last.gradients.row(i)),
                    None => f64::INFINITY,
                }
            }
            None => f64::INFINITY,
        };
        let rows = self.state.rows.load(Ordering::Relaxed);
        BandReport {
            status,
            max_force,
            ci_index: image,
            iteration: self.iteration,
            surface_rows: rows,
            curvature,
            saddle_index: None,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::band::{BandConfig, CiConfig, assemble_band_with};
    use crate::minmode::MinModeConfig;
    use ndarray::{Array2, ArrayView2, array};
    use rgmin::Method;

    struct Ridge;

    impl BandSurface for Ridge {
        fn eval(
            &self,
            positions: ArrayView2<f64>,
            energies: &mut Array1<f64>,
            gradients: &mut Array2<f64>,
        ) -> Result<(), SaddleError> {
            for (i, row) in positions.outer_iter().enumerate() {
                energies[i] = 1.0 - row[0] * row[0];
                gradients[(i, 0)] = 1.0;
            }
            Ok(())
        }
    }

    #[test]
    fn first_band_step_after_refine_finish_uses_the_climbing_force() {
        let positions = array![[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let config = BandConfig::default()
            .with_method(Method::lbfgs())
            .with_climbing(CiConfig::always())
            .with_min_mode(None)
            .with_saddle_check(None)
            .with_remove_translation(false)
            .with_max_move(0.1);
        let mut band = BandSession::new(config, positions).expect("band");
        let surface = Ridge;
        let x = band.interior_flat();
        assemble_band_with(
            &band.config,
            &band.state,
            &surface,
            band.positions.row(0),
            band.positions.row(2),
            x.view(),
            None,
        )
        .expect("armed climbing assembly");
        assert_eq!(band.state.ci(), Some(1));

        let saved = Point {
            x: array![0.0, 0.0, 0.0],
            energy: 1.0,
            gradient: array![1.0, 0.0, 0.0],
        };
        let current = Point {
            x: array![0.2, 0.0, 0.0],
            energy: 0.96,
            gradient: array![1.0, 0.0, 0.0],
        };
        let refine_config = MinModeRefine::default().with_restore_unhelpful(false);
        band.refine.run = Some(Run {
            image: 1,
            saved,
            entry_force: 1.0,
            tangent: array![1.0, 0.0, 0.0],
            session: MinModeSession::new(
                MinModeConfig::default().with_saddle_check(None),
                current.x.clone(),
                array![1.0, 0.0, 0.0],
            )
            .expect("refinement session"),
            climb: None,
            current,
            previous: None,
            best: None,
            alignment: 1.0,
            steps: refine_config.max_steps,
            since_refresh: 0,
            escape: false,
        });
        band.refine_finish(&refine_config, Outcome::Leave);
        let refined_x = band.positions[(1, 0)];
        assert_eq!(refined_x, 0.2);

        band.step(&surface).expect("first band step");
        assert!(
            band.positions[(1, 0)] > refined_x,
            "the climbing image followed the raw negative gradient"
        );
    }
}
