//! The band session: assemble NEB forces on a caller surface, take
//! one rgmin solver step, report. Hosts own the loop.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicUsize, Ordering};

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use rgmin::{Accept, Control, Method, Oracle, Solver};

use crate::error::SaddleError;
use crate::force::ForceGate;
use crate::projection::{ProjectionKind, climbing_image_force, dneb_component, force_perp};
use crate::spring::SpringKind;
use crate::tangent::TangentKind;
use crate::tangent::compute_tangent;

/// Row-major 3x3 simulation cell: row `i` is lattice vector `a_i`.
/// Band mechanics need nothing more from periodicity; neighbor lists
/// (when a surface wants them) are vesin's job, not this crate's.
#[derive(Clone, Copy, Debug)]
pub struct Cell(pub [[f64; 3]; 3]);

impl Cell {
    /// Inverse of the row matrix, or `None` for a singular cell.
    fn inverse(&self) -> Option<[[f64; 3]; 3]> {
        let h = &self.0;
        let c00 = h[1][1] * h[2][2] - h[1][2] * h[2][1];
        let c01 = h[1][2] * h[2][0] - h[1][0] * h[2][2];
        let c02 = h[1][0] * h[2][1] - h[1][1] * h[2][0];
        let det = h[0][0] * c00 + h[0][1] * c01 + h[0][2] * c02;
        let scale = h.iter().flatten().fold(0.0_f64, |m, v| m.max(v.abs()));
        if !(det.is_finite() && det.abs() > 1e-12 * scale.powi(3)) {
            return None;
        }
        let inv_det = 1.0 / det;
        Some([
            [
                c00 * inv_det,
                (h[0][2] * h[2][1] - h[0][1] * h[2][2]) * inv_det,
                (h[0][1] * h[1][2] - h[0][2] * h[1][1]) * inv_det,
            ],
            [
                c01 * inv_det,
                (h[0][0] * h[2][2] - h[0][2] * h[2][0]) * inv_det,
                (h[0][2] * h[1][0] - h[0][0] * h[1][2]) * inv_det,
            ],
            [
                c02 * inv_det,
                (h[0][1] * h[2][0] - h[0][0] * h[2][1]) * inv_det,
                (h[0][0] * h[1][1] - h[0][1] * h[1][0]) * inv_det,
            ],
        ])
    }

    /// Wrap every per-atom 3-vector in `diff` to its nearest lattice
    /// image: fractional components `s = d H^-1` are shifted by
    /// `round(s)` and mapped back, `d - round(s) H`. A diagonal cell
    /// reduces to the per-axis wrap. For a strongly skewed triclinic
    /// cell the fractional wrap is the image inside the reduced unit
    /// cell, which is the minimum image whenever the displacement is
    /// shorter than half the shortest cell height. Axes with a zero
    /// row (a slab's vacuum direction, a molecule) are left unwrapped.
    pub fn minimum_image(&self, diff: &mut Array1<f64>) {
        let h = &self.0;
        let periodic: [bool; 3] = std::array::from_fn(|i| h[i].iter().any(|v| *v != 0.0));
        if periodic.iter().all(|p| *p)
            && let Some(inv) = self.inverse()
        {
            for a in 0..diff.len() / 3 {
                let d = [diff[3 * a], diff[3 * a + 1], diff[3 * a + 2]];
                let mut shift = [0.0; 3];
                for (i, si) in shift.iter_mut().enumerate() {
                    let s = d[0] * inv[0][i] + d[1] * inv[1][i] + d[2] * inv[2][i];
                    *si = s.round();
                }
                for c in 0..3 {
                    diff[3 * a + c] =
                        d[c] - (shift[0] * h[0][c] + shift[1] * h[1][c] + shift[2] * h[2][c]);
                }
            }
            return;
        }
        // Partially periodic or singular: wrap only along axes whose
        // lattice vector is the coordinate axis itself.
        for a in 0..diff.len() / 3 {
            for c in 0..3 {
                let l = h[c][c];
                let axis_aligned = (0..3).all(|k| k == c || h[c][k] == 0.0);
                if periodic[c] && axis_aligned && l > 0.0 {
                    let x = diff[3 * a + c];
                    diff[3 * a + c] = x - l * (x / l).round();
                }
            }
        }
    }
}

/// Climbing-image activation, the eOn trigger rule: CI arms when the
/// convergence force falls under `factor * baseline` or under the
/// absolute trigger.
#[derive(Clone, Copy, Debug)]
pub struct CiConfig {
    pub trigger_factor: f64,
    pub trigger_force: f64,
}

/// Band configuration. `force_tol` is on the max absolute component
/// of the projected force over interior images.
#[derive(Clone, Debug)]
pub struct BandConfig {
    pub tangent: TangentKind,
    pub spring: SpringKind,
    pub projection: ProjectionKind,
    pub climbing: Option<CiConfig>,
    pub cell: Option<Cell>,
    pub force_tol: f64,
    pub max_move: f64,
    /// Band stepper. FIRE by default (eOn's velocity NEB stepper).
    /// L-BFGS runs under rgmin's `Accept::Step`: the two-loop
    /// direction, capped per atom, one band evaluation per step and no
    /// energy test, since the projected force is not the gradient of
    /// the pseudo-energy.
    pub method: Method,
}

impl Default for BandConfig {
    fn default() -> Self {
        Self {
            tangent: TangentKind::Improved,
            spring: SpringKind::Uniform { k: 5.0 },
            projection: ProjectionKind::Neb,
            climbing: Some(CiConfig {
                trigger_factor: 0.5,
                trigger_force: 0.0,
            }),
            cell: None,
            force_tol: 1e-3,
            max_move: 0.2,
            method: Method::Fire {
                kind: rgmin::FireKind::V2,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BandStatus {
    Running,
    Converged,
}

/// One step's report.
#[derive(Clone, Debug)]
pub struct BandReport {
    pub status: BandStatus,
    pub max_force: f64,
    pub ci_index: Option<usize>,
    pub iteration: usize,
    /// Image evaluations the surface performed during this step: the
    /// sum of the row counts over every [`BandSurface::eval`] call.
    /// A warm step costs `n_images - 2`; the first step after
    /// [`BandSession::new`] or [`BandSession::reset`] adds the two
    /// endpoints.
    pub surface_rows: usize,
}

/// The caller's surface: fused energies and gradients for a run of
/// consecutive band images in one call (the batched channel is the
/// measured win; a per-image surface implements this with a loop).
///
/// Rows are interior images only, once the endpoint energies are
/// cached. The endpoints (band rows 0 and `n_images - 1`) never move,
/// their gradients are never read, and their energies feed only the
/// tangent, so the session evaluates them once: the first call after
/// [`BandSession::new`], [`BandSession::reset`], or a
/// [`BandSession::set_positions`] that moved an endpoint carries the
/// whole band (`positions.nrows() == n_images`, row `r` is image
/// `r`); every later call carries the `n_images - 2` interior images
/// in order (row `r` is image `r + 1`). A host that keys per-image
/// state (orbitals, densities) on the image index tells the two apart
/// by the row count.
pub trait BandSurface: Sync {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError>;
}

/// Rows within this distance (per Cartesian component, after the
/// minimum image) count as the same geometry in
/// [`BandSession::set_positions`]. A host that wraps atoms into its
/// cell moves a coordinate by a lattice vector, which the minimum
/// image removes up to rounding of order `1e-16 |L|`.
const SAME_ROW_TOL: f64 = 1e-10;

/// The surface's answer at one band geometry. `x` is the flat
/// interior; `energies` and `gradients` cover the whole band (the
/// endpoint gradient rows hold whatever the last whole-band
/// evaluation returned); `projected` is the assembled interior force
/// under the climbing state of the most recent assembly.
#[derive(Clone)]
struct BandEval {
    x: Array1<f64>,
    energies: Array1<f64>,
    gradients: Array2<f64>,
    projected: Array1<f64>,
}

/// Climbing state, the endpoint cache, and the last evaluation, shared
/// with the evaluation closure. Atomics and mutexes because the rgmin
/// oracle demands `Sync`; the oracle runs on the stepping thread.
struct BandState {
    force_gate: ForceGate,
    baseline: Mutex<Option<f64>>,
    /// Climbing armed: from then on the climbing image is the highest
    /// interior image of every evaluation (eOn's rule).
    armed: AtomicBool,
    ci_index: AtomicI64,
    /// The first surface or assembly error raised inside the oracle
    /// during the current step. The rgmin oracle signature has no
    /// error channel, so the closure parks the error here and
    /// [`BandSession::step`] raises it after the solver returns.
    surface_error: Mutex<Option<SaddleError>>,
    /// Max abs projected-force component at the most recent assembly.
    last_max_force: Mutex<f64>,
    /// Endpoint energies; valid while `Some`. The endpoints never
    /// move, so one evaluation serves every later step.
    endpoints: Mutex<Option<[f64; 2]>>,
    /// The last surface answer. An oracle call at the same interior
    /// reassembles from it instead of calling the surface, so a host
    /// that hands the band back unchanged (or a solver that dropped
    /// its own cache) costs no evaluation.
    last: Mutex<Option<BandEval>>,
    /// Rows evaluated by the surface during the current step.
    rows: AtomicUsize,
}

impl BandState {
    fn new() -> Self {
        Self {
            force_gate: ForceGate::LinfNorm,
            baseline: Mutex::new(None),
            armed: AtomicBool::new(false),
            ci_index: AtomicI64::new(-1),
            surface_error: Mutex::new(None),
            last_max_force: Mutex::new(f64::INFINITY),
            endpoints: Mutex::new(None),
            last: Mutex::new(None),
            rows: AtomicUsize::new(0),
        }
    }
    fn convergence_force(&self, force: &Array1<f64>) -> f64 {
        match self.force_gate {
            ForceGate::LinfNorm => max_abs(force),
            gate => gate.value(force.view()),
        }
    }
    fn ci(&self) -> Option<usize> {
        let i = self.ci_index.load(Ordering::Relaxed);
        (i >= 0).then_some(i as usize)
    }
    fn endpoints(&self) -> Option<[f64; 2]> {
        *self.endpoints.lock().unwrap_or_else(|e| e.into_inner())
    }
    fn set_endpoints(&self, value: Option<[f64; 2]>) {
        *self.endpoints.lock().unwrap_or_else(|e| e.into_inner()) = value;
    }
    fn last_max_force(&self) -> f64 {
        *self
            .last_max_force
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }
    fn drop_last(&self) {
        *self.last.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }
    /// The climbing baseline and the armed image belong to one
    /// relaxation; the surface caches stay.
    fn restart(&self) {
        *self.baseline.lock().unwrap_or_else(|e| e.into_inner()) = None;
        self.armed.store(false, Ordering::Relaxed);
        self.ci_index.store(-1, Ordering::Relaxed);
    }
    /// Keep the first error of a step; later evaluations at garbage
    /// positions add nothing.
    fn record_error(&self, err: SaddleError) {
        if let Ok(mut slot) = self.surface_error.lock()
            && slot.is_none()
        {
            *slot = Some(err);
        }
    }
    fn take_error(&self) -> Option<SaddleError> {
        self.surface_error
            .lock()
            .ok()
            .and_then(|mut slot| slot.take())
    }
}

/// Energies and gradients of the whole band at interior `x`: from the
/// last evaluation when `x` matches it bitwise, else from one surface
/// call (interior rows only once the endpoint energies are cached).
fn evaluate_band(
    state: &BandState,
    surface: &dyn BandSurface,
    endpoint_first: ArrayView1<f64>,
    endpoint_last: ArrayView1<f64>,
    x: ArrayView1<f64>,
) -> Result<(Array1<f64>, Array2<f64>), SaddleError> {
    let dof = endpoint_first.len();
    let n_images = x.len() / dof + 2;
    let mut slot = state.last.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(prev) = slot.as_ref()
        && prev.x == x
    {
        return Ok((prev.energies.clone(), prev.gradients.clone()));
    }
    let interior = x
        .to_shape((n_images - 2, dof))
        .map_err(|e| SaddleError::Shape(e.to_string()))?;
    let mut energies = Array1::zeros(n_images);
    // Endpoint gradient rows are never read; keep the last ones known.
    let mut gradients = match slot.as_ref() {
        Some(prev) => prev.gradients.clone(),
        None => Array2::zeros((n_images, dof)),
    };
    match state.endpoints() {
        Some([e_first, e_last]) => {
            let mut interior_e = Array1::zeros(n_images - 2);
            let mut interior_g = Array2::zeros((n_images - 2, dof));
            surface.eval(interior.view(), &mut interior_e, &mut interior_g)?;
            state.rows.fetch_add(n_images - 2, Ordering::Relaxed);
            energies[0] = e_first;
            energies[n_images - 1] = e_last;
            energies.slice_mut(s![1..n_images - 1]).assign(&interior_e);
            gradients
                .slice_mut(s![1..n_images - 1, ..])
                .assign(&interior_g);
        }
        None => {
            let mut full = Array2::zeros((n_images, dof));
            full.row_mut(0).assign(&endpoint_first);
            full.row_mut(n_images - 1).assign(&endpoint_last);
            full.slice_mut(s![1..n_images - 1, ..]).assign(&interior);
            surface.eval(full.view(), &mut energies, &mut gradients)?;
            state.rows.fetch_add(n_images, Ordering::Relaxed);
        }
    }
    if !energies.iter().all(|e| e.is_finite()) {
        return Err(SaddleError::NonFinite("band energies"));
    }
    if !gradients
        .slice(s![1..n_images - 1, ..])
        .iter()
        .all(|g| g.is_finite())
    {
        return Err(SaddleError::NonFinite("band gradients"));
    }
    if state.endpoints().is_none() {
        state.set_endpoints(Some([energies[0], energies[n_images - 1]]));
    }
    *slot = Some(BandEval {
        x: x.to_owned(),
        energies: energies.clone(),
        gradients: gradients.clone(),
        projected: Array1::zeros(0),
    });
    Ok((energies, gradients))
}

/// Projected force on interior image `i`, climbing when `climb`.
#[allow(clippy::too_many_arguments)]
fn image_force(
    config: &BandConfig,
    i: usize,
    climb: bool,
    endpoint_first: ArrayView1<f64>,
    endpoint_last: ArrayView1<f64>,
    x: ArrayView1<f64>,
    energies: &Array1<f64>,
    gradients: &Array2<f64>,
) -> Result<Array1<f64>, SaddleError> {
    let dof = endpoint_first.len();
    let n_images = energies.len();
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
    let mut pos_diff_next = &pos_next - &pos;
    let mut pos_diff_prev = &pos - &pos_prev;
    if let Some(cell) = &config.cell {
        cell.minimum_image(&mut pos_diff_next);
        cell.minimum_image(&mut pos_diff_prev);
    }
    let dist_next = pos_diff_next.dot(&pos_diff_next).sqrt();
    let dist_prev = pos_diff_prev.dot(&pos_diff_prev).sqrt();
    let tangent = compute_tangent(
        config.tangent,
        pos_diff_next.view(),
        pos_diff_prev.view(),
        energies[i],
        energies[i - 1],
        energies[i + 1],
    )
    .map_err(|e| SaddleError::Invalid(format!("image {i}: {e}")))?;
    let force = -&gradients.row(i);
    let spring = config.spring.compute(
        i,
        tangent.view(),
        dist_next,
        dist_prev,
        pos_diff_next.view(),
        pos_diff_prev.view(),
    );
    if climb {
        let dneb = if config.projection == ProjectionKind::DoublyNudged {
            let fp = force_perp(force.view(), tangent.view());
            dneb_component(spring.full.view(), tangent.view(), fp.view())
        } else {
            Array1::zeros(dof)
        };
        Ok(climbing_image_force(
            force.view(),
            tangent.view(),
            dneb.view(),
        ))
    } else {
        Ok(config
            .projection
            .project(force.view(), tangent.view(), &spring))
    }
}

fn max_abs(v: &Array1<f64>) -> f64 {
    v.iter().fold(0.0_f64, |m, c| m.max(c.abs()))
}

/// Assemble the projected band force at interior `x`. Returns
/// (pseudo-energy, projected interior forces, max abs component).
///
/// The climbing image is decided on this evaluation's energies, so
/// the force returned already carries the climb: once armed, the
/// highest interior image climbs; on the evaluation that arms it, the
/// climbing image's force is reassembled before returning.
fn assemble_band(
    config: &BandConfig,
    state: &BandState,
    surface: &dyn BandSurface,
    endpoint_first: ArrayView1<f64>,
    endpoint_last: ArrayView1<f64>,
    x: ArrayView1<f64>,
) -> Result<(f64, Array1<f64>, f64), SaddleError> {
    let (energies, gradients) = evaluate_band(state, surface, endpoint_first, endpoint_last, x)?;
    let n_images = energies.len();
    let dof = endpoint_first.len();

    let mut max_i = 1;
    for i in 2..n_images - 1 {
        if energies[i] > energies[max_i] {
            max_i = i;
        }
    }
    let climbing = config.climbing.is_some();
    let armed = climbing && state.armed.load(Ordering::Relaxed);
    if armed {
        state.ci_index.store(max_i as i64, Ordering::Relaxed);
    }
    let ci_at = if armed { Some(max_i) } else { None };

    let mut projected = Array1::zeros((n_images - 2) * dof);
    for i in 1..n_images - 1 {
        let f = image_force(
            config,
            i,
            ci_at == Some(i),
            endpoint_first,
            endpoint_last,
            x,
            &energies,
            &gradients,
        )?;
        projected.slice_mut(s![(i - 1) * dof..i * dof]).assign(&f);
    }
    let mut max_component = state.convergence_force(&projected);

    // Baseline capture and arming read the relaxation force of the
    // first evaluation and of this one.
    let baseline = {
        let mut b = state.baseline.lock().unwrap_or_else(|e| e.into_inner());
        *b.get_or_insert(max_component)
    };
    if let Some(ci) = &config.climbing
        && !armed
        && (max_component < baseline * ci.trigger_factor || max_component < ci.trigger_force)
    {
        state.armed.store(true, Ordering::Relaxed);
        state.ci_index.store(max_i as i64, Ordering::Relaxed);
        let f = image_force(
            config,
            max_i,
            true,
            endpoint_first,
            endpoint_last,
            x,
            &energies,
            &gradients,
        )?;
        projected
            .slice_mut(s![(max_i - 1) * dof..max_i * dof])
            .assign(&f);
        max_component = state.convergence_force(&projected);
    }

    if let Some(last) = state
        .last
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_mut()
        && last.x == x
    {
        last.projected.clone_from(&projected);
    }
    *state
        .last_max_force
        .lock()
        .unwrap_or_else(|e| e.into_inner()) = max_component;
    let pseudo_energy: f64 = energies.slice(s![1..n_images - 1]).sum();
    Ok((pseudo_energy, projected, max_component))
}

/// The oracle's value beside a projected (band) or inverted (min-mode)
/// force is not the potential of that force, so a method that tests
/// the value (a line search, an energy acceptance) refuses good steps.
/// FIRE, L-BFGS under `Accept::Step`, and Barzilai-Borwein (one oracle
/// call, no test under the session's `Accept::None`) step on the force
/// alone.
pub(crate) fn check_force_driven(method: &Method) -> Result<(), SaddleError> {
    match method {
        Method::Fire { .. } | Method::Lbfgs { .. } | Method::Bb => Ok(()),
        other => Err(SaddleError::Invalid(format!(
            "{other:?} tests the oracle value, which is not the potential of a \
             projected or inverted force; use FIRE, L-BFGS, or BB"
        ))),
    }
}

/// What the band holds at its current positions after a step: the
/// surface's energies (whole band) and gradients (whole band; the
/// endpoint rows are those of the last whole-band evaluation), and the
/// projected interior force the solver stepped on.
#[derive(Clone, Debug)]
pub struct BandEvaluation {
    pub energies: Array1<f64>,
    pub gradients: Array2<f64>,
    /// `(n_images - 2) x dof`, row `r` is image `r + 1`.
    pub projected: Array2<f64>,
}

struct CachedBandSurface<'a, S> {
    state: &'a BandState,
    surface: &'a S,
}

impl<S: BandSurface> BandSurface for CachedBandSurface<'_, S> {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        let last = positions.nrows() - 1;
        let x: Array1<f64> = positions.slice(s![1..last, ..]).iter().copied().collect();
        let (e, g) = evaluate_band(self.state, self.surface, positions.row(0), positions.row(last), x.view())?;
        energies.assign(&e);
        gradients.assign(&g);
        Ok(())
    }
}

/// Stepping band relaxation over an rgmin solver. Endpoints (rows 0
/// and `n_images - 1`) never move.
pub struct BandSession {
    config: BandConfig,
    positions: Array2<f64>,
    solver: Solver,
    state: BandState,
    iteration: usize,
    rtr: Option<crate::rtr::BandRtr>,
}

impl BandSession {
    pub fn new(config: BandConfig, initial: Array2<f64>) -> Result<Self, SaddleError> {
        let n_images = initial.nrows();
        let dof = initial.ncols();
        if n_images < 3 || dof == 0 || !dof.is_multiple_of(3) {
            return Err(SaddleError::Shape(format!(
                "band needs >= 3 images of 3N dof; got {n_images} x {dof}"
            )));
        }
        if let SpringKind::Weighted { ks } = &config.spring
            && ks.len() != n_images - 1
        {
            return Err(SaddleError::Shape(format!(
                "weighted springs need n_images - 1 = {} constants; got {}",
                n_images - 1,
                ks.len()
            )));
        }
        if let SpringKind::OnsagerMachlup { l_vecs, .. } = &config.spring
            && (l_vecs.len() != n_images || l_vecs.iter().any(|l| l.len() != dof))
        {
            return Err(SaddleError::Shape(format!(
                "Onsager-Machlup springs need {n_images} displacements of length {dof}"
            )));
        }
        if !initial.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("band positions"));
        }
        check_force_driven(&config.method)?;
        let interior_dof = (n_images - 2) * dof;
        let control = Control {
            maxiter: usize::MAX,
            gtol: 0.0,
            istep: 1.0,
            maxmove: None,
            ftol_rel: None,
        };
        let mut solver = Solver::new(config.method.clone(), control, interior_dof);
        // The flat band vector is consecutive xyz triples, so the cap
        // applies to the largest single-atom displacement (eOn
        // maxAtomMotionApplied), not to the L2 norm of the whole
        // band step. A non-positive max_move leaves the step uncapped.
        solver.set_atom_maxmove(config.max_move);
        // The projected band force is not the gradient of the summed
        // image energy the oracle reports, so an energy-decrease test
        // says nothing about a band step: L-BFGS takes its two-loop
        // direction with one oracle call (rgmin Accept::Step), as FIRE
        // steps unconditionally.
        if matches!(config.method, Method::Lbfgs { .. }) {
            solver.set_accept(Accept::Step);
        }
        Ok(Self {
            config,
            positions: initial,
            solver,
            state: BandState::new(),
            iteration: 0,
            rtr: None,
        })
    }

    /// Select the optional HiGHS feasible-set step; requires the `highs` feature.
    /// Disabled by default.
    pub fn set_highs(&mut self, enabled: bool) {
        self.solver.set_highs(enabled);
    }

    /// Select the convergence and climbing-trigger norm, retaining surface values.
    pub fn set_force_gate(&mut self, gate: ForceGate) {
        if self.state.force_gate != gate {
            self.state.force_gate = gate;
            self.restart();
        }
    }

    /// Select an optional trust-region stepper, retaining cached surface values.
    pub fn set_rtr(&mut self, config: Option<crate::rtr::RtrConfig>) -> Result<(), SaddleError> {
        if let Some(config) = config {
            config.validate()?;
        }
        self.rtr = config.map(|config| crate::rtr::BandRtr::new(config, false));
        self.restart();
        Ok(())
    }

    pub fn positions(&self) -> ArrayView2<'_, f64> {
        self.positions.view()
    }

    pub fn climbing_image(&self) -> Option<usize> {
        self.state.ci()
    }

    /// Replace the band (host-side move between steps: acquisition,
    /// reparameterization, or a resync of positions the host read
    /// back). Rows equal to the current ones up to the minimum image
    /// (within `1e-10` per component) keep the session's own copy, so
    /// a host that wraps atoms into its cell and hands the band back
    /// changes nothing: the endpoint energies and the last evaluation
    /// survive and the next step costs one interior evaluation. A
    /// moved endpoint drops the endpoint energies (the next evaluation
    /// carries the whole band); a moved interior image drops the last
    /// evaluation. Optimizer history survives; call
    /// [`BandSession::restart`] to drop it as well.
    pub fn set_positions(&mut self, positions: Array2<f64>) -> Result<(), SaddleError> {
        if positions.dim() != self.positions.dim() {
            return Err(SaddleError::Shape("set_positions shape".into()));
        }
        if !positions.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("band positions"));
        }
        let n_images = positions.nrows();
        let mut moved = false;
        for (i, new_row) in positions.outer_iter().enumerate() {
            let mut diff = &new_row - &self.positions.row(i);
            if let Some(cell) = &self.config.cell {
                cell.minimum_image(&mut diff);
            }
            if diff.iter().all(|d| d.abs() <= SAME_ROW_TOL) {
                continue;
            }
            self.positions.row_mut(i).assign(&new_row);
            if i == 0 || i == n_images - 1 {
                self.state.set_endpoints(None);
                // The interior may be unchanged, so the solver's own
                // cached force (assembled against the old endpoint)
                // would answer the next step; drop it, keep the memory.
                self.solver.forget_evaluation();
            }
            // The last evaluation's tangents read the neighbours, so
            // any moved row retires it.
            moved = true;
        }
        if moved {
            self.state.drop_last();
        }
        Ok(())
    }

    /// The model-update boundary: drop quasi-Newton history, the
    /// climbing baseline, the armed climbing image, and every cached
    /// surface value (endpoint energies, the last evaluation). The
    /// next evaluation carries the whole band. Use it when the surface
    /// itself changed (a surrogate refit, a new calculator setting).
    pub fn reset(&mut self) {
        self.restart();
        self.state.set_endpoints(None);
        self.state.drop_last();
    }

    /// Restart the relaxation on the same surface: drop quasi-Newton
    /// history (FIRE velocity, L-BFGS pairs), the climbing baseline,
    /// and the armed climbing image, keep the endpoint energies and
    /// the last evaluation. Mirrors eOn's optimizer reset after a
    /// reparameterization; with [`BandSession::set_positions`] for the
    /// moved images it costs one interior evaluation.
    pub fn restart(&mut self) {
        self.solver.forget();
        self.state.restart();
        if let Some(rtr) = &mut self.rtr {
            *rtr = crate::rtr::BandRtr::new(rtr.config, false);
            rtr.set_force_gate(self.state.force_gate);
        }
    }

    /// The last evaluation, if it was taken at the current positions:
    /// energies and gradients from the surface and the projected
    /// interior force the step used. `None` before the first step and
    /// after a change that dropped it.
    pub fn evaluation(&self) -> Option<BandEvaluation> {
        let x = self.interior_flat();
        let slot = self.state.last.lock().unwrap_or_else(|e| e.into_inner());
        let last = slot.as_ref().filter(|l| l.x == x)?;
        let n_images = self.positions.nrows();
        let dof = self.positions.ncols();
        let projected = last
            .projected
            .clone()
            .into_shape_with_order((n_images - 2, dof))
            .ok()?;
        Some(BandEvaluation {
            energies: last.energies.clone(),
            gradients: last.gradients.clone(),
            projected,
        })
    }

    fn interior_flat(&self) -> Array1<f64> {
        let n_images = self.positions.nrows();
        self.positions
            .slice(s![1..n_images - 1, ..])
            .iter()
            .copied()
            .collect()
    }

    fn scatter_interior(&mut self, flat: ArrayView1<f64>) {
        let n_images = self.positions.nrows();
        let dof = self.positions.ncols();
        for i in 1..n_images - 1 {
            self.positions
                .row_mut(i)
                .assign(&flat.slice(s![(i - 1) * dof..i * dof]));
        }
    }

    /// One solver step over the assembled band force. The host
    /// interleaves its policy between calls.
    ///
    /// A surface error inside the step returns `Err` with the band
    /// unchanged. The solver's internal state (FIRE velocity,
    /// quasi-Newton pairs) has seen the failed evaluation; a host that
    /// recovers the surface and steps again may call
    /// [`BandSession::restart`] first to drop it.
    pub fn step<S: BandSurface>(&mut self, surface: &S) -> Result<BandReport, SaddleError> {
        if self.rtr.is_some() {
            return self.step_rtr(surface);
        }
        let n_images = self.positions.nrows();
        let dof = self.positions.ncols();
        let interior_dof = (n_images - 2) * dof;

        let endpoint_first = self.positions.row(0).to_owned();
        let endpoint_last = self.positions.row(n_images - 1).to_owned();
        let config = &self.config;
        let state = &self.state;
        state.take_error();
        state.rows.store(0, Ordering::Relaxed);
        let oracle = Oracle::unbounded(interior_dof, |x: ArrayView1<f64>| {
            match assemble_band(
                config,
                state,
                surface,
                endpoint_first.view(),
                endpoint_last.view(),
                x,
            ) {
                Ok((e, f, _)) => (e, -f),
                Err(err) => {
                    state.record_error(err);
                    (f64::INFINITY, Array1::zeros(interior_dof))
                }
            }
        });

        let mut x = self.interior_flat();
        let stepped = self.solver.step(&oracle, &mut x);
        drop(oracle);
        // The surface error is the root cause of whatever the solver
        // did with an infinite energy and a zero gradient, so it
        // outranks a solver error. `x` is a local copy; the session's
        // positions stay where they were.
        if let Some(err) = self.state.take_error() {
            return Err(err);
        }
        stepped.map_err(|e| SaddleError::Solver(e.to_string()))?;
        self.scatter_interior(x.view());
        self.iteration += 1;

        // A solver that ends on a point it did not evaluate last (a
        // line search that backtracked) is assembled once more; the
        // cache answers when the accepted point was the last one.
        let at_last = self
            .state
            .last
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .is_some_and(|l| l.x == x && l.projected.len() == interior_dof);
        if !at_last {
            assemble_band(
                &self.config,
                &self.state,
                surface,
                endpoint_first.view(),
                endpoint_last.view(),
                x.view(),
            )?;
        }
        let max_force = self.state.last_max_force();
        let status = if max_force <= self.config.force_tol {
            BandStatus::Converged
        } else {
            BandStatus::Running
        };
        Ok(BandReport {
            status,
            max_force,
            ci_index: self.state.ci(),
            iteration: self.iteration,
            surface_rows: self.state.rows.load(Ordering::Relaxed),
        })
    }

    fn step_rtr<S: BandSurface>(&mut self, surface: &S) -> Result<BandReport, SaddleError> {
        self.state.take_error();
        self.state.rows.store(0, Ordering::Relaxed);
        let first = self.positions.row(0).to_owned();
        let last = self.positions.row(self.positions.nrows() - 1).to_owned();
        let initial = self.interior_flat();
        assemble_band(&self.config, &self.state, surface, first.view(), last.view(), initial.view())?;
        let mut trial = self.positions.clone();
        let cached = CachedBandSurface { state: &self.state, surface };
        let rtr = self.rtr.as_mut().expect("RTR configuration");
        rtr.climb = self.state.ci().is_some();
        rtr.set_force_gate(self.state.force_gate);
        rtr.step(&self.config, &cached, &mut trial)?;
        let interior: Array1<f64> = trial.slice(s![1..trial.nrows() - 1, ..]).iter().copied().collect();
        assemble_band(&self.config, &self.state, surface, first.view(), last.view(), interior.view())?;
        self.positions.assign(&trial);
        self.iteration += 1;
        let max_force = self.state.last_max_force();
        Ok(BandReport {
            status: if max_force <= self.config.force_tol { BandStatus::Converged } else { BandStatus::Running },
            max_force,
            ci_index: self.state.ci(),
            iteration: self.iteration,
            surface_rows: self.state.rows.load(Ordering::Relaxed),
        })
    }

    /// Convenience loop over [`BandSession::step`]; nothing more.
    pub fn run<S: BandSurface>(
        &mut self,
        surface: &S,
        max_steps: usize,
    ) -> Result<BandReport, SaddleError> {
        let mut report = self.step(surface)?;
        while report.status == BandStatus::Running && self.iteration < max_steps {
            report = self.step(surface)?;
        }
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn orthorhombic_cell_wraps_each_axis() {
        let cell = Cell([[10.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 6.0]]);
        let mut d = array![9.0, -5.0, 2.0];
        cell.minimum_image(&mut d);
        assert_abs_diff_eq!(d[0], -1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(d[1], 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(d[2], 2.0, epsilon = 1e-12);
    }

    #[test]
    fn triclinic_cell_wraps_along_the_lattice_vectors() {
        // Hexagonal a = 4: a2 = (2, 2 sqrt 3, 0). A displacement of
        // a2 + (0.1, 0.2, 0.3) is the short vector (0.1, 0.2, 0.3); the
        // diagonal-only wrap would leave x at 2.1.
        let r3 = 3.0_f64.sqrt();
        let cell = Cell([[4.0, 0.0, 0.0], [2.0, 2.0 * r3, 0.0], [0.0, 0.0, 10.0]]);
        let mut d = array![2.1, 2.0 * r3 + 0.2, 0.3];
        cell.minimum_image(&mut d);
        assert_abs_diff_eq!(d[0], 0.1, epsilon = 1e-12);
        assert_abs_diff_eq!(d[1], 0.2, epsilon = 1e-12);
        assert_abs_diff_eq!(d[2], 0.3, epsilon = 1e-12);
        // Whole lattice translations vanish.
        let mut t = array![6.0, 2.0 * r3, -10.0];
        cell.minimum_image(&mut t);
        for v in t.iter() {
            assert_abs_diff_eq!(*v, 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn slab_cell_leaves_the_vacuum_axis_alone() {
        let cell = Cell([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.0]]);
        let mut d = array![4.0, -3.0, 12.0];
        cell.minimum_image(&mut d);
        assert_abs_diff_eq!(d[0], -1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(d[1], 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(d[2], 12.0, epsilon = 1e-12);
    }
}
