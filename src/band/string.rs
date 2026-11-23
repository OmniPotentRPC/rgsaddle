//! Simplified string method of E, Ren, and Vanden-Eijnden, J. Chem.
//! Phys. 126, 164103 (2007), doi:10.1063/1.2720838.
//!
//! Each step evolves the interior images under the full potential force
//! (no spring, no tangent projection) through the band's
//! [`rgmin::Solver`] stepper, then places the interior images at equal
//! arc length via [`crate::reparametrize`] on either the not-a-knot
//! cubic spline ([`Reparametrization::NotAKnot`], the default) or the
//! polyline ([`Reparametrization::Linear`]). When
//! [`super::BandConfig::climbing`] arms an interior image, that image
//! reflects the potential force along the path tangent and stays
//! anchored during reparametrization so each side of it is spaced at
//! equal arc length on its own.

use std::sync::Mutex;
use std::sync::atomic::Ordering;

use ndarray::{Array1, Array2, ArrayView1, s};
use rgmin::{Exp, Method, Oracle};

use super::{
    BandConfig, BandMetrics, BandReport, BandSession, BandState, BandStatus, BandSurface,
    build_solver, climb_target, evaluate_band, max_abs, metric_force, min_mode, pin_fixed_rows,
    remove_net_translation, zero_fixed,
};
use crate::error::SaddleError;
use crate::reparam::{Reparametrization, reparametrize};
use crate::tangent::compute_tangent;

/// Simplified string configuration on [`BandConfig::string`].
///
/// E, Ren, and Vanden-Eijnden, J. Chem. Phys. 126, 164103 (2007),
/// doi:10.1063/1.2720838. Off (`None`) by default on [`BandConfig`],
/// which runs the nudged elastic band; setting `Some(StringConfig)`
/// evolves interior images under the full potential force and
/// reparametrizes to equal arc length after each step.
///
/// Non-exhaustive: start from [`StringConfig::default`],
/// [`StringConfig::spline`] or [`StringConfig::linear`] and chain the
/// `with_*` setters.
///
/// # Examples
///
/// ```
/// use ndarray::{Array1, Array2, ArrayView2};
/// use rgsaddle::{
///     BandConfig, BandSession, BandStatus, BandSurface, DEFAULT_FORCE_TOL, SaddleError,
///     StringConfig,
/// };
///
/// struct DoubleWell;
///
/// impl BandSurface for DoubleWell {
///     fn eval(
///         &self,
///         positions: ArrayView2<f64>,
///         energies: &mut Array1<f64>,
///         gradients: &mut Array2<f64>,
///     ) -> Result<(), SaddleError> {
///         for (i, r) in positions.outer_iter().enumerate() {
///             let (x, y, z) = (r[0], r[1], r[2]);
///             energies[i] = (x * x - 1.0).powi(2) + 2.0 * y * y + 2.0 * z * z;
///             gradients[(i, 0)] = 4.0 * x * (x * x - 1.0);
///             gradients[(i, 1)] = 4.0 * y;
///             gradients[(i, 2)] = 4.0 * z;
///         }
///         Ok(())
///     }
/// }
///
/// let initial = Array2::from_shape_fn((9, 3), |(i, k)| {
///     let t = i as f64 / 8.0;
///     [2.0 * t - 1.0, 0.3 * (std::f64::consts::PI * t).sin(), 0.0][k]
/// });
/// let config = BandConfig::default()
///     .with_string(StringConfig::default())
///     .with_force_tol(DEFAULT_FORCE_TOL)
///     .with_max_move(0.1)
///     .with_min_mode(None);
/// let mut session = BandSession::new(config, initial)?;
/// let report = session.run(&DoubleWell, 500)?;
/// assert_eq!(report.status, BandStatus::Converged);
/// let ci = report.ci_index.expect("climbing image");
/// assert!(session.positions().row(ci).iter().all(|c| c.abs() < 0.01));
/// # Ok::<(), SaddleError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct StringConfig {
    /// Equal arc-length placement after each evolution step.
    /// [`Reparametrization::NotAKnot`], the default, fits the
    /// not-a-knot cubic spline through the images in their polyline arc
    /// length; [`Reparametrization::Linear`] places on the polyline.
    pub reparametrization: Reparametrization,
}

with_setters!(StringConfig {
    with_reparametrization => reparametrization: val Reparametrization,
});

impl StringConfig {
    /// Equal arc-length placement on the not-a-knot cubic spline.
    #[must_use]
    pub fn spline() -> Self {
        Self {
            reparametrization: Reparametrization::NotAKnot,
        }
    }

    /// Equal arc-length placement on the polyline.
    #[must_use]
    pub fn linear() -> Self {
        Self {
            reparametrization: Reparametrization::Linear,
        }
    }
}

impl Default for StringConfig {
    fn default() -> Self {
        Self::spline()
    }
}

pub(super) fn check_string(config: &BandConfig) -> Result<(), SaddleError> {
    if config.solid.is_some() {
        return Err(SaddleError::Invalid(
            "the simplified string method does not carry a solid-state cell block".into(),
        ));
    }
    if config.dynamic.is_some() {
        return Err(SaddleError::Invalid(
            "the simplified string method evaluates every image; dynamic NEB is for the NEB".into(),
        ));
    }
    Ok(())
}

/// Reparametrized optimizer history across string steps.
#[derive(Clone, Debug, Default)]
pub(super) struct StringState {
    /// Secant pairs `(s, y)` of the reparametrized string steps for L-BFGS.
    pairs: Vec<(Array1<f64>, Array1<f64>)>,
    /// Previous reparametrized displacement `s`.
    prev_step: Option<Array1<f64>>,
    /// Stepper time scale adapted to transverse curvature.
    dt: Option<f64>,
    /// Climbing image index of the previous step.
    prev_ci: Option<usize>,
    /// Whether the preceding accepted string step met the force tolerance.
    converged_once: bool,
}

fn lock_guard<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// Evolution force of interior image `i`: the full potential force on a
/// non-climbing image (no spring, no tangent projection), and the
/// reflected potential force along the path tangent on an armed
/// climbing image. With an exponential metric installed, returns the
/// preconditioned driving force beside it.
#[allow(clippy::too_many_arguments)]
fn string_image_force(
    config: &BandConfig,
    state: &BandState,
    i: usize,
    climb: bool,
    endpoint_first: ArrayView1<f64>,
    endpoint_last: ArrayView1<f64>,
    x: ArrayView1<f64>,
    energies: &Array1<f64>,
    gradients: &Array2<f64>,
    metric: Option<(&Exp, Option<&[usize]>)>,
) -> Result<(Array1<f64>, Option<Array1<f64>>), SaddleError> {
    let dof = endpoint_first.len();
    let n_images = energies.len();
    if !climb {
        let mut evol = -&gradients.row(i);
        zero_fixed(&config.fixed_atoms, &mut evol);
        if config.drops_translation() {
            remove_net_translation(&mut evol);
        }
        let driving = match metric {
            None => None,
            Some((p, free)) => {
                let mut d = match free {
                    None => {
                        let s = p.solve(gradients.row(i)).ok_or_else(|| {
                            SaddleError::Invalid(
                                "exponential metric: the linear solve failed".into(),
                            )
                        })?;
                        -s
                    }
                    Some(free) => {
                        let gathered: Array1<f64> =
                            free.iter().map(|&c| gradients[(i, c)]).collect();
                        let s = p.solve(gathered.view()).ok_or_else(|| {
                            SaddleError::Invalid(
                                "exponential metric: the linear solve failed".into(),
                            )
                        })?;
                        let mut full = Array1::zeros(dof);
                        for (k, &c) in free.iter().enumerate() {
                            full[c] = -s[k];
                        }
                        full
                    }
                };
                if config.drops_translation() {
                    remove_net_translation(&mut d);
                }
                if !d.dot(&d).is_finite() {
                    return Err(SaddleError::NonFinite("preconditioned band force norm"));
                }
                Some(d)
            }
        };
        return Ok((evol, driving));
    }

    let row = |k: usize| -> ArrayView1<f64> {
        if k == 0 {
            endpoint_first
        } else if k == n_images - 1 {
            endpoint_last
        } else {
            x.slice_move(s![(k - 1) * dof..k * dof])
        }
    };
    let (pos_prev, pos, pos_next) = (row(i - 1), row(i), row(i + 1));
    let mut diff_next = &pos_next - &pos;
    let mut diff_prev = &pos - &pos_prev;
    if let Some(cell) = &config.cell {
        cell.minimum_image(&mut diff_next);
        cell.minimum_image(&mut diff_prev);
    }
    zero_fixed(&config.fixed_atoms, &mut diff_next);
    zero_fixed(&config.fixed_atoms, &mut diff_prev);
    let tangent = compute_tangent(
        config.tangent,
        diff_next.view(),
        diff_prev.view(),
        energies[i],
        energies[i - 1],
        energies[i + 1],
    )
    .map_err(|e| SaddleError::Invalid(format!("image {i}: {e}")))?;
    let force = -&gradients.row(i);
    let mut evol = crate::stencil::reflect(force.view(), tangent.view());
    zero_fixed(&config.fixed_atoms, &mut evol);
    if config.drops_translation() {
        remove_net_translation(&mut evol);
    }
    let driving = match metric {
        None => None,
        Some((p, free)) => {
            let zero_curv = Array1::zeros(dof);
            let mut d;
            match free {
                None => {
                    (d, evol) = metric_force(
                        state,
                        p,
                        gradients.row(i),
                        tangent.view(),
                        zero_curv.view(),
                        true,
                        None,
                    )?;
                }
                Some(free) => {
                    let gather = |v: ArrayView1<f64>| -> Array1<f64> {
                        free.iter().map(|&c| v[c]).collect()
                    };
                    let zero_free = Array1::zeros(free.len());
                    let (d_free, e_free) = metric_force(
                        state,
                        p,
                        gather(gradients.row(i)).view(),
                        gather(tangent.view()).view(),
                        zero_free.view(),
                        true,
                        None,
                    )?;
                    d = Array1::zeros(dof);
                    evol = Array1::zeros(dof);
                    for (k, &c) in free.iter().enumerate() {
                        d[c] = d_free[k];
                        evol[c] = e_free[k];
                    }
                }
            }
            if config.drops_translation() {
                remove_net_translation(&mut d);
                remove_net_translation(&mut evol);
            }
            if !d.dot(&d).is_finite() {
                return Err(SaddleError::NonFinite("preconditioned band force norm"));
            }
            Some(d)
        }
    };
    Ok((evol, driving))
}

/// Evaluate `x` and assemble the simplified-string evolution force.
/// Returns `(pseudo_energy, evol_force, step_force, climb_target)` where
/// `step_force` is the preconditioned driving force when a metric is
/// present and `evol_force` otherwise.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn assemble_string_with(
    config: &BandConfig,
    state: &BandState,
    surface: &dyn BandSurface,
    endpoint_first: ArrayView1<f64>,
    endpoint_last: ArrayView1<f64>,
    x: ArrayView1<f64>,
    metrics: Option<BandMetrics<'_>>,
) -> Result<(f64, Array1<f64>, Array1<f64>, Option<usize>), SaddleError> {
    let mut work = lock_guard(&state.work);
    let slot = evaluate_band(state, &mut work, surface, endpoint_first, endpoint_last, x)?;
    let eval = slot.as_ref().expect("evaluate_band leaves an evaluation");
    let (energies, gradients) = (&eval.energies, &eval.gradients);
    let n_images = energies.len();
    let dof = endpoint_first.len();

    let mut max_i = 1;
    for i in 2..n_images - 1 {
        if energies[i] > energies[max_i] {
            max_i = i;
        }
    }
    let target = climb_target(
        config,
        max_i,
        endpoint_first,
        endpoint_last,
        x,
        energies.view(),
        gradients.view(),
    );
    let climbing = config.climbing.is_some();
    let armed = climbing && state.armed.load(Ordering::Relaxed);
    if armed {
        let code = match target {
            Some(idx) => idx as i64,
            None => -1,
        };
        state.ci_index.store(code, Ordering::Relaxed);
    }
    let ci_at = if armed { target } else { None };
    let pseudo_energy: f64 = energies.slice(s![1..n_images - 1]).sum();

    let metric_of = |i: usize| metrics.and_then(|m| m.per_image.get(i - 1).map(|p| (p, m.free)));
    let mut evol = Array1::zeros((n_images - 2) * dof);
    let mut step_force = Array1::zeros((n_images - 2) * dof);
    for i in 1..n_images - 1 {
        let rows = s![(i - 1) * dof..i * dof];
        let (f, d) = string_image_force(
            config,
            state,
            i,
            ci_at == Some(i),
            endpoint_first,
            endpoint_last,
            x,
            energies,
            gradients,
            metric_of(i),
        )?;
        evol.slice_mut(rows).assign(&f);
        match d {
            Some(drv) => step_force.slice_mut(rows).assign(&drv),
            None => step_force.slice_mut(rows).assign(&f),
        }
    }
    if !evol.dot(&evol).is_finite() || !step_force.dot(&step_force).is_finite() {
        return Err(SaddleError::NonFinite("band force norm"));
    }
    Ok((pseudo_energy, evol, step_force, target))
}

struct ReparamTrial {
    band: Array2<f64>,
    x: Array1<f64>,
    evol_force: Array1<f64>,
    climb_target: Option<usize>,
}

impl BandSession {
    /// Install, replace, or clear the simplified string method
    /// ([`BandConfig::string`]). Optimizer history and the climbing
    /// state restart; cached endpoint energies stay.
    pub fn set_string(&mut self, string: Option<StringConfig>) -> Result<(), SaddleError> {
        if string.is_some() {
            if self.solid.is_some() {
                return Err(SaddleError::Invalid(
                    "the simplified string method does not carry a solid-state cell block".into(),
                ));
            }
            if self.rtr.is_some() {
                return Err(SaddleError::Invalid(
                    "the simplified string method uses the band's first-order stepper, \
                     not the trust-region stepper"
                        .into(),
                ));
            }
            if self.config.dynamic.is_some() {
                return Err(SaddleError::Invalid(
                    "the simplified string method evaluates every image; dynamic NEB is for the NEB"
                        .into(),
                ));
            }
        }
        self.config.string = string;
        self.restart();
        Ok(())
    }

    /// The simplified string configuration, if enabled.
    #[must_use]
    pub fn string(&self) -> Option<StringConfig> {
        self.config.string
    }

    pub(super) fn clear_string_state(&mut self) {
        self.string_state = StringState::default();
    }

    pub(super) fn step_string<S: BandSurface>(
        &mut self,
        surface: &S,
        string_cfg: StringConfig,
    ) -> Result<BandReport, SaddleError> {
        if self.rtr.is_some() {
            return Err(SaddleError::Invalid(
                "the simplified string method uses the band's first-order stepper, \
                 not the trust-region stepper"
                    .into(),
            ));
        }
        let n_images = self.positions.nrows();
        let dof = self.positions.ncols();
        let interior_dof = (n_images - 2) * dof;

        self.state.take_error();
        self.state.rows.store(0, Ordering::Relaxed);
        self.refresh_exp(surface)?;

        let endpoint_first = self.positions.row(0).to_owned();
        let endpoint_last = self.positions.row(n_images - 1).to_owned();
        let before = self.interior_flat();
        let metrics = match &self.exp {
            Some(exp) => Some(
                exp.fresh_metrics(before.view(), n_images - 2)
                    .ok_or_else(|| {
                        SaddleError::Invalid(
                            "exponential metric is not current at the start of the string step"
                                .into(),
                        )
                    })?,
            ),
            None => None,
        };

        let (e_start, evol_start, step_force_start, _) = assemble_string_with(
            &self.config,
            &self.state,
            surface,
            endpoint_first.view(),
            endpoint_last.view(),
            before.view(),
            metrics,
        )?;
        let anchor_before = self.state.ci();
        if anchor_before != self.string_state.prev_ci {
            self.string_state.pairs.clear();
            self.string_state.prev_step = None;
            if anchor_before.is_some() {
                self.solver.forget();
            }
            self.string_state.prev_ci = anchor_before;
        }

        let is_ode12r = !self.config.quickmin && matches!(self.config.method, Method::Ode12r);
        let mut x_evol = before.clone();

        let config = &self.config;
        let state = &self.state;
        let start_grad = -&step_force_start;
        let oracle = Oracle::unbounded(interior_dof, |xv: ArrayView1<f64>| {
            if xv == before.view() {
                return (e_start, start_grad.clone());
            }
            match assemble_string_with(
                config,
                state,
                surface,
                endpoint_first.view(),
                endpoint_last.view(),
                xv,
                metrics,
            ) {
                Ok((e, _, step_f, _)) => (e, -step_f),
                Err(err) => {
                    state.record_error(err);
                    (f64::INFINITY, Array1::zeros(interior_dof))
                }
            }
        });
        let stepped = self.solver.step(&oracle, &mut x_evol);
        drop(oracle);
        if let Some(err) = self.state.take_error() {
            self.solver.forget_evaluation();
            return Err(err);
        }
        stepped.map_err(|e| SaddleError::Solver(e.to_string()))?;

        if is_ode12r {
            if x_evol == before {
                if self.unmoved < 64 {
                    self.unmoved += 1;
                    self.state.restore_at(&before);
                    let raw_max = self.state.convergence_force(&evol_start);
                    *lock_guard(&self.state.last_max_force) = raw_max;
                    self.iteration += 1;
                    return Ok(BandReport {
                        status: BandStatus::Running,
                        max_force: raw_max,
                        ci_index: self.state.ci(),
                        iteration: self.iteration,
                        surface_rows: self.state.rows.load(Ordering::Relaxed),
                        curvature: None,
                        saddle_index: None,
                    });
                }
                let raw_max = self.state.convergence_force(&evol_start);
                if raw_max > self.config.force_tol {
                    return Err(SaddleError::Stalled(format!(
                        "band unchanged with max force {raw_max:e} above {:e}",
                        self.config.force_tol
                    )));
                }
            }
            self.unmoved = 0;
        } else {
            if !x_evol.iter().all(|v| v.is_finite()) {
                return Err(SaddleError::NonFinite("band step"));
            }
            if x_evol == before && max_abs(&evol_start) > 0.0 {
                let raw_max = self.state.convergence_force(&evol_start);
                if raw_max > self.config.force_tol {
                    return Err(SaddleError::Stalled(format!(
                        "band unchanged with max force {raw_max:e} above {:e}",
                        self.config.force_tol
                    )));
                }
            }
        }

        let mut band = self.positions.clone();
        for i in 1..n_images - 1 {
            band.row_mut(i)
                .assign(&x_evol.slice(s![(i - 1) * dof..i * dof]));
        }
        reparametrize(
            &mut band,
            anchor_before,
            self.config.cell.as_ref(),
            string_cfg.reparametrization,
        );
        pin_fixed_rows(&self.state.fixed, &mut band);
        let x_after: Array1<f64> = band
            .slice(s![1..n_images - 1, ..])
            .iter()
            .copied()
            .collect();
        let (_, evol_after, _, target) = assemble_string_with(
            &self.config,
            &self.state,
            surface,
            endpoint_first.view(),
            endpoint_last.view(),
            x_after.view(),
            metrics,
        )?;
        let trial = ReparamTrial {
            band,
            x: x_after,
            evol_force: evol_after,
            climb_target: target,
        };

        if !trial.x.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("band step"));
        }

        let mut s_vec = Array1::zeros(interior_dof);
        for i in 1..n_images - 1 {
            let mut d = &trial.band.row(i) - &self.positions.row(i);
            if let Some(cell) = &self.config.cell {
                cell.minimum_image(&mut d);
            }
            zero_fixed(&self.config.fixed_atoms, &mut d);
            s_vec.slice_mut(s![(i - 1) * dof..i * dof]).assign(&d);
        }

        let mut dx_evol = &x_evol - &before;
        zero_fixed(&self.config.fixed_atoms, &mut dx_evol);
        let dx_norm = dx_evol.dot(&dx_evol).sqrt();
        let f_norm = step_force_start.dot(&step_force_start).sqrt();
        let mut eff_force = if dx_norm > 0.0 && f_norm > 0.0 {
            &s_vec * (f_norm / dx_norm)
        } else {
            s_vec.clone()
        };
        if let Some(ci) = anchor_before {
            let rows = s![(ci - 1) * dof..ci * dof];
            eff_force
                .slice_mut(rows)
                .assign(&trial.evol_force.slice(rows));
        }
        if !eff_force.dot(&eff_force).is_finite() {
            return Err(SaddleError::NonFinite("band force norm"));
        }

        let mut max_component = self.state.convergence_force(&eff_force);
        let baseline = {
            let mut b = lock_guard(&self.state.baseline);
            *b.get_or_insert(max_component)
        };
        lock_guard(&self.state.atom_baseline)
            .get_or_insert_with(|| min_mode::max_atom(eff_force.view()));

        let mut armed_now = false;
        if let Some(ci_cfg) = &self.config.climbing
            && anchor_before.is_none()
            && !self.state.armed.load(Ordering::Relaxed)
            && (max_component < baseline * ci_cfg.trigger_factor
                || max_component < ci_cfg.trigger_force)
        {
            self.state.armed.store(true, Ordering::Relaxed);
            let code = match trial.climb_target {
                Some(idx) => idx as i64,
                None => -1,
            };
            self.state.ci_index.store(code, Ordering::Relaxed);
            if let Some(ci_i) = trial.climb_target {
                let (_, evol_ci, _, _) = assemble_string_with(
                    &self.config,
                    &self.state,
                    surface,
                    endpoint_first.view(),
                    endpoint_last.view(),
                    trial.x.view(),
                    metrics,
                )?;
                let rows = s![(ci_i - 1) * dof..ci_i * dof];
                eff_force.slice_mut(rows).assign(&evol_ci.slice(rows));
                max_component = self.state.convergence_force(&eff_force);
            }
            armed_now = true;
        }

        if trial.x == before && max_component > self.config.force_tol {
            return Err(SaddleError::Stalled(format!(
                "band unchanged with max force {max_component:e} above {:e}",
                self.config.force_tol
            )));
        }

        {
            let mut work = lock_guard(&self.state.work);
            let mut last = lock_guard(&self.state.last);
            if let Some(eval) = last.as_mut()
                && eval.x == trial.x
            {
                work.store_forces(eval, &eff_force, None);
                eval.assembly_ci = i64::MIN;
            }
        }
        *lock_guard(&self.state.last_max_force) = max_component;

        let y_vec = &evol_start - &trial.evol_force;
        let sy = s_vec.dot(&y_vec);
        let yy = y_vec.dot(&y_vec);

        if armed_now {
            self.string_state = StringState {
                pairs: Vec::new(),
                prev_step: None,
                dt: self.string_state.dt,
                prev_ci: self.state.ci(),
                converged_once: false,
            };
            self.solver.forget();
        } else if !self.config.quickmin
            && let Method::Lbfgs { memory, .. } = self.config.method
        {
            let uphill_kink = string_cfg.reparametrization == Reparametrization::Linear
                && s_vec.dot(&evol_start) <= 0.0;
            self.solver.forget();
            if sy > 0.0 && !uphill_kink {
                if memory > 0 {
                    if self.string_state.pairs.len() == memory {
                        self.string_state.pairs.remove(0);
                    }
                    self.string_state.pairs.push((s_vec, y_vec));
                }
                for (ps, py) in &self.string_state.pairs {
                    let _ = self.solver.push_pair(ps.view(), py.view());
                }
            } else {
                self.string_state.pairs.clear();
            }
        } else {
            let mut dt_cur = match self.string_state.dt {
                Some(dt) => dt,
                None => self.config.time_step,
            };
            let bounced = self.string_state.prev_step.as_ref().is_some_and(|prev| {
                let pn = prev.dot(prev).sqrt();
                let sn = s_vec.dot(&s_vec).sqrt();
                pn > 0.0 && sn > 0.0 && prev.dot(&s_vec) < -0.5 * pn * sn
            });
            if sy > 0.0 && yy > 0.0 {
                let alpha_bb = sy / yy;
                let ceiling = if self.config.quickmin {
                    alpha_bb.sqrt()
                } else {
                    match self.config.method {
                        Method::Fire {
                            kind: rgmin::FireKind::V2,
                            ..
                        } => (4.0 * alpha_bb).sqrt(),
                        Method::Fire { .. } => alpha_bb.sqrt(),
                        _ => alpha_bb,
                    }
                };
                if ceiling < dt_cur && ceiling > 0.0 && ceiling.is_finite() {
                    dt_cur = ceiling;
                }
            }
            if bounced {
                dt_cur *= 0.5;
            }
            self.string_state.dt = Some(dt_cur);
            self.string_state.prev_step = Some(s_vec);
            let method = if self.config.quickmin {
                Method::QuickMin
            } else {
                self.config.method.clone()
            };
            let mut stepped_cfg = self.config.clone();
            stepped_cfg.time_step = dt_cur;
            self.solver = build_solver(&stepped_cfg, &method, interior_dof);
            self.solver.set_highs(self.highs);
        }

        self.positions.assign(&trial.band);
        self.pin_fixed();
        self.iteration += 1;

        let under_tolerance = max_component <= self.config.force_tol;
        let status = if under_tolerance && self.string_state.converged_once {
            BandStatus::Converged
        } else {
            BandStatus::Running
        };
        self.string_state.converged_once = under_tolerance;
        if !self.zoom_latch(status) {
            self.refine_latch(status)?;
        }
        Ok(BandReport {
            status,
            max_force: max_component,
            ci_index: self.state.ci(),
            iteration: self.iteration,
            surface_rows: self.state.rows.load(Ordering::Relaxed),
            curvature: None,
            saddle_index: None,
        })
    }
}

/// Standalone simplified string session (E, Ren, and Vanden-Eijnden,
/// J. Chem. Phys. 126, 164103 (2007), doi:10.1063/1.2720838) with
/// preserved per-image [`crate::stencil::ImageId`]s and batched surface
/// evaluations.
///
/// [`GrowingStringSession`](super::GrowingStringSession) hands off to
/// this session once its reactant and product fragments join.
pub struct StringSession {
    session: BandSession,
}

impl StringSession {
    /// Open a simplified string session over `positions` (`n_images >= 3`
    /// rows of `3 * n_atoms`). Image IDs default to `0..n_images`.
    pub fn new(config: BandConfig, positions: Array2<f64>) -> Result<Self, SaddleError> {
        let ids: Vec<crate::stencil::ImageId> =
            (0..positions.nrows() as crate::stencil::ImageId).collect();
        Self::with_ids(config, ids, positions)
    }

    /// Open a simplified string session with explicit per-image `ids`.
    pub fn with_ids(
        mut config: BandConfig,
        ids: Vec<crate::stencil::ImageId>,
        positions: Array2<f64>,
    ) -> Result<Self, SaddleError> {
        if config.string.is_none() {
            config.string = Some(StringConfig::default());
        }
        if ids.len() != positions.nrows() {
            return Err(SaddleError::Shape(format!(
                "string session needs one image id per row; got {} ids for {} rows",
                ids.len(),
                positions.nrows()
            )));
        }
        let mut session = BandSession::new(config, positions)?;
        session.set_ids(ids)?;
        Ok(Self { session })
    }

    /// Take one simplified-string step on `surface`.
    pub fn step<S: BandSurface>(&mut self, surface: &S) -> Result<BandReport, SaddleError> {
        self.session.step(surface)
    }

    /// Loop [`StringSession::step`] until convergence or `max_steps`.
    pub fn run<S: BandSurface>(
        &mut self,
        surface: &S,
        max_steps: usize,
    ) -> Result<BandReport, SaddleError> {
        self.session.run(surface, max_steps)
    }

    /// Current string positions (`n_images x 3N`).
    #[must_use]
    pub fn positions(&self) -> ndarray::ArrayView2<'_, f64> {
        self.session.positions()
    }

    /// Replace the string positions.
    pub fn set_positions(&mut self, positions: Array2<f64>) -> Result<(), SaddleError> {
        self.session.set_positions(positions)
    }

    /// Stable image identifiers of the string rows.
    #[must_use]
    pub fn ids(&self) -> &[crate::stencil::ImageId] {
        self.session.ids()
    }

    /// Last cached evaluation (`energies`, `gradients`, `projected`), if valid.
    #[must_use]
    pub fn evaluation(&self) -> Option<super::BandEvaluation> {
        self.session.evaluation()
    }

    /// Index of the armed climbing image, if any.
    #[must_use]
    pub fn climbing_image(&self) -> Option<usize> {
        self.session.climbing_image()
    }

    /// Certified saddle index of the climbing image at convergence, if any.
    #[must_use]
    pub fn saddle_index(&self) -> Option<&crate::saddle_index::SaddleIndex> {
        self.session.saddle_index()
    }

    /// Number of steps taken in this string session.
    #[must_use]
    pub fn iteration(&self) -> usize {
        self.session.iteration()
    }

    /// Restart optimizer history and climbing state, keeping cached endpoint energies.
    pub fn restart(&mut self) {
        self.session.restart();
    }

    /// Reset optimizer history, climbing state, and all cached evaluations.
    pub fn reset(&mut self) {
        self.session.reset();
    }

    /// Set worker threads for batched image evaluation.
    pub fn set_image_threads(&mut self, threads: usize) -> Result<(), SaddleError> {
        self.session.set_image_threads(threads)
    }

    /// Worker thread count for batched image evaluation.
    #[must_use]
    pub fn image_threads(&self) -> usize {
        self.session.image_threads()
    }

    /// Replace the per-atom fixed mask.
    pub fn set_fixed_atoms(&mut self, fixed: Vec<bool>) -> Result<(), SaddleError> {
        self.session.set_fixed_atoms(fixed)
    }

    /// Borrow the underlying [`BandSession`].
    #[must_use]
    pub fn band_session(&self) -> &BandSession {
        &self.session
    }

    /// Consume into the underlying [`BandSession`].
    #[must_use]
    pub fn into_band_session(self) -> BandSession {
        self.session
    }

    pub(crate) fn seed_endpoint_energies(&mut self, energies: [f64; 2]) -> Result<(), SaddleError> {
        self.session.seed_endpoint_energies(energies)
    }
}
