//! The band session: assemble NEB forces on a caller surface, take
//! one rgmin solver step, report. Hosts own the loop.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicUsize, Ordering};

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use rgmin::{Accept, Control, Method, Oracle, Solver};

use crate::cell_log::{M3, m3_det};
use crate::error::SaddleError;
use crate::force::ForceGate;
use crate::projection::{ProjectionKind, climbing_image_force, dneb_component, force_perp};
use crate::solid_state::{
    cartesian_step, cell_neb_force, joint_displacement, orient_lower_triangular, pack_joint,
    solid_state_enthalpy, solid_state_jacobian, unpack_cell,
};
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

/// Per-image cells for the generalized solid-state band.
///
/// Sheppard, Xiao, Chemelewski, Johnson, and Henkelman, J. Chem. Phys.
/// 136, 074103 (2012). The Jacobian is fixed from the mean endpoint
/// volume when the session is built. Each cell is put in
/// lower-triangular form, and that image's positions rotate with it.
#[derive(Clone, Debug)]
pub struct SolidState {
    /// One cell per image, endpoints included.
    pub cells: Vec<Cell>,
    /// Hydrostatic pressure added to the Cauchy stress. A positive
    /// value pushes the cell inward.
    pub pressure: f64,
    /// Positive scale on the Jacobian. 1 is the 2012 value.
    pub weight: f64,
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
    /// Drop the mean Cartesian force of every image that has two or
    /// more atoms. eOn does this when every atom is free and leaves
    /// the mean in place when any atom is fixed, so a host with fixed
    /// atoms clears this flag.
    pub remove_translation: bool,
    /// Band stepper. FIRE by default (eOn's velocity NEB stepper).
    /// L-BFGS runs under rgmin's `Accept::Step`: the two-loop
    /// direction, capped per atom, one band evaluation per step and no
    /// energy test, since the projected force is not the gradient of
    /// the pseudo-energy. It also arms rgmin's NEB guards: a two-loop
    /// step that reaches the per-atom cap or faces away from the force
    /// drops its pairs and steps along the force, and an empty memory
    /// scales that step by 0.01.
    pub method: Method,
    /// Joint atomic and cell block. `None` is the atomic band.
    pub solid: Option<SolidState>,
    /// Ask for the quick-min step. The flag builds the solver as
    /// `Method::QuickMin`. The velocity is one vector over the interior
    /// coordinates, atoms and cell together. FIRE stays the default
    /// when the flag is clear.
    pub quickmin: bool,
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
            remove_translation: true,
            method: Method::Fire {
                kind: rgmin::FireKind::V2,
            },
            solid: None,
            quickmin: false,
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

    /// Energies, Cartesian gradients and Cauchy stresses for a
    /// solid-state band. `cells` has one entry per row of `positions`,
    /// in the same order as [`BandSurface::eval`]. `stresses` is
    /// `n_rows × 9`, row-major. The default refuses: an atomic surface
    /// has no stress.
    fn eval_cells(
        &self,
        positions: ArrayView2<f64>,
        cells: &[Cell],
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
        stresses: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        let _ = (positions, cells, energies, gradients, stresses);
        Err(SaddleError::Invalid(
            "solid-state band requires BandSurface::eval_cells".into(),
        ))
    }
}

/// Oriented cells and the Jacobian fixed at session build.
struct SolidRun {
    cells: Vec<Cell>,
    pressure: f64,
    jacobian: f64,
    reference: M3,
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
    /// Cauchy stress, whole band, row-major 9. Empty on an atomic band.
    stresses: Array2<f64>,
    /// Geometry the solid-state assembly evaluated. Empty on an atomic band.
    band_positions: Array2<f64>,
    band_cells: Vec<Cell>,
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
        stresses: Array2::zeros((0, 9)),
        band_positions: Array2::zeros((0, dof)),
        band_cells: Vec::new(),
    });
    Ok((energies, gradients))
}

/// Remove the mean Cartesian force (eOn's `zeroTranslation`). An image
/// with one atom has no internal force apart from the mean, so it
/// keeps its force.
pub(crate) fn remove_net_translation(force: &mut Array1<f64>) {
    let n = force.len() / 3;
    if n < 2 || force.len() != 3 * n {
        return;
    }
    let mut mean = [0.0; 3];
    for atom in 0..n {
        for (c, m) in mean.iter_mut().enumerate() {
            *m += force[3 * atom + c];
        }
    }
    for m in &mut mean {
        *m /= n as f64;
    }
    for atom in 0..n {
        for (c, m) in mean.iter().enumerate() {
            force[3 * atom + c] -= m;
        }
    }
}

fn prepare_solid(
    config: &BandConfig,
    mut positions: Array2<f64>,
) -> Result<(Array2<f64>, Option<SolidRun>), SaddleError> {
    let Some(spec) = &config.solid else {
        return Ok((positions, None));
    };
    if matches!(config.spring, SpringKind::OnsagerMachlup { .. }) {
        return Err(SaddleError::Invalid(
            "solid-state band does not use Onsager-Machlup springs".into(),
        ));
    }
    let n_images = positions.nrows();
    let dof = positions.ncols();
    if spec.cells.len() != n_images {
        return Err(SaddleError::Shape(format!(
            "solid-state band needs {n_images} cells; got {}",
            spec.cells.len()
        )));
    }
    if !spec.pressure.is_finite() {
        return Err(SaddleError::NonFinite("solid-state pressure"));
    }
    let mut cells = Vec::with_capacity(n_images);
    for (i, cell) in spec.cells.iter().enumerate() {
        if !cell.0.iter().flatten().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("solid-state cell"));
        }
        let mut oriented = cell.0;
        let mut row = positions.row(i).to_owned();
        orient_lower_triangular(&mut oriented, &mut row)?;
        positions.row_mut(i).assign(&row);
        cells.push(Cell(oriented));
    }
    let mean_volume = 0.5 * (m3_det(cells[0].0).abs() + m3_det(cells[n_images - 1].0).abs());
    let jacobian = solid_state_jacobian(mean_volume, dof / 3, spec.weight)?;
    Ok((
        positions,
        Some(SolidRun {
            reference: cells[0].0,
            cells,
            pressure: spec.pressure,
            jacobian,
        }),
    ))
}

/// Energies, gradients, stresses, band positions and band cells of one
/// solid-state evaluation.
type SolidEvaluation = (
    Array1<f64>,
    Array2<f64>,
    Array2<f64>,
    Array2<f64>,
    Vec<Cell>,
);

fn evaluate_solid(
    state: &BandState,
    surface: &dyn BandSurface,
    endpoint_first: ArrayView1<f64>,
    endpoint_last: ArrayView1<f64>,
    cell_first: Cell,
    cell_last: Cell,
    x: ArrayView1<f64>,
) -> Result<SolidEvaluation, SaddleError> {
    let dof = endpoint_first.len();
    let seg = dof + 9;
    let n_interior = x.len() / seg;
    let n_images = n_interior + 2;
    let mut slot = state.last.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(prev) = slot.as_ref()
        && prev.x == x
        && prev.band_cells.len() == n_images
    {
        return Ok((
            prev.energies.clone(),
            prev.gradients.clone(),
            prev.stresses.clone(),
            prev.band_positions.clone(),
            prev.band_cells.clone(),
        ));
    }
    let mut positions = Array2::zeros((n_images, dof));
    positions.row_mut(0).assign(&endpoint_first);
    positions.row_mut(n_images - 1).assign(&endpoint_last);
    let mut cells = vec![cell_first; n_images];
    cells[n_images - 1] = cell_last;
    for i in 0..n_interior {
        let start = i * seg;
        positions
            .row_mut(i + 1)
            .assign(&x.slice(s![start..start + dof]));
        let mut raw = [0.0; 9];
        for k in 0..9 {
            raw[k] = x[start + dof + k];
        }
        cells[i + 1] = Cell(unpack_cell(&raw));
    }
    let mut energies = Array1::zeros(n_images);
    let mut gradients = match slot.as_ref() {
        Some(prev) if prev.gradients.dim() == (n_images, dof) => prev.gradients.clone(),
        _ => Array2::zeros((n_images, dof)),
    };
    let mut stresses = Array2::zeros((n_images, 9));
    match state.endpoints() {
        Some([e_first, e_last]) => {
            let mut interior_e = Array1::zeros(n_interior);
            let mut interior_g = Array2::zeros((n_interior, dof));
            let mut interior_s = Array2::zeros((n_interior, 9));
            let interior_pos = positions.slice(s![1..n_images - 1, ..]);
            surface.eval_cells(
                interior_pos,
                &cells[1..n_images - 1],
                &mut interior_e,
                &mut interior_g,
                &mut interior_s,
            )?;
            state.rows.fetch_add(n_interior, Ordering::Relaxed);
            energies[0] = e_first;
            energies[n_images - 1] = e_last;
            energies.slice_mut(s![1..n_images - 1]).assign(&interior_e);
            gradients
                .slice_mut(s![1..n_images - 1, ..])
                .assign(&interior_g);
            stresses
                .slice_mut(s![1..n_images - 1, ..])
                .assign(&interior_s);
        }
        None => {
            surface.eval_cells(
                positions.view(),
                &cells,
                &mut energies,
                &mut gradients,
                &mut stresses,
            )?;
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
    if !stresses
        .slice(s![1..n_images - 1, ..])
        .iter()
        .all(|g| g.is_finite())
    {
        return Err(SaddleError::NonFinite("band stresses"));
    }
    if state.endpoints().is_none() {
        state.set_endpoints(Some([energies[0], energies[n_images - 1]]));
    }
    *slot = Some(BandEval {
        x: x.to_owned(),
        energies: energies.clone(),
        gradients: gradients.clone(),
        projected: Array1::zeros(0),
        stresses: stresses.clone(),
        band_positions: positions.clone(),
        band_cells: cells.clone(),
    });
    Ok((energies, gradients, stresses, positions, cells))
}

/// Joint projected force and its Cartesian image for one interior image.
#[allow(clippy::too_many_arguments)]
fn solid_image(
    config: &BandConfig,
    i: usize,
    climb: bool,
    positions: &Array2<f64>,
    cells: &[Cell],
    enthalpy: &Array1<f64>,
    gradients: &Array2<f64>,
    stresses: &Array2<f64>,
    jacobian: f64,
    pressure: f64,
) -> Result<(Array1<f64>, Array1<f64>), SaddleError> {
    let dof = positions.ncols();
    let (next_atomic, next_cell) = joint_displacement(
        cells[i].0,
        positions.row(i),
        cells[i + 1].0,
        positions.row(i + 1),
        jacobian,
    )?;
    let (prev_atomic, prev_cell) = joint_displacement(
        cells[i - 1].0,
        positions.row(i - 1),
        cells[i].0,
        positions.row(i),
        jacobian,
    )?;
    let diff_next = pack_joint(next_atomic.view(), next_cell);
    let diff_prev = pack_joint(prev_atomic.view(), prev_cell);
    let dist_next = diff_next.dot(&diff_next).sqrt();
    let dist_prev = diff_prev.dot(&diff_prev).sqrt();
    let tangent = compute_tangent(
        config.tangent,
        diff_next.view(),
        diff_prev.view(),
        enthalpy[i],
        enthalpy[i - 1],
        enthalpy[i + 1],
    )
    .map_err(|e| SaddleError::Invalid(format!("image {i}: {e}")))?;
    let atomic_force = gradients.row(i).mapv(|component| -component);
    let mut cauchy = [[0.0; 3]; 3];
    for k in 0..9 {
        cauchy[k / 3][k % 3] = stresses[(i, k)];
    }
    let volume = m3_det(cells[i].0).abs();
    let cell_force = cell_neb_force(cauchy, volume, jacobian, pressure)?;
    let force = pack_joint(atomic_force.view(), cell_force);
    let spring = config.spring.compute(
        i,
        tangent.view(),
        dist_next,
        dist_prev,
        diff_next.view(),
        diff_prev.view(),
    );
    let mut projected = if climb {
        let dneb = if config.projection == ProjectionKind::DoublyNudged {
            let perpendicular = force_perp(force.view(), tangent.view());
            dneb_component(spring.full.view(), tangent.view(), perpendicular.view())
        } else {
            Array1::zeros(force.len())
        };
        climbing_image_force(force.view(), tangent.view(), dneb.view())
    } else {
        config
            .projection
            .project(force.view(), tangent.view(), &spring)
    };
    if config.remove_translation {
        let mut atomic = projected.slice(s![..dof]).to_owned();
        remove_net_translation(&mut atomic);
        projected.slice_mut(s![..dof]).assign(&atomic);
    }
    let mut raw = [0.0; 9];
    for k in 0..9 {
        raw[k] = projected[dof + k];
    }
    let cleared = unpack_cell(&raw);
    let (delta_r, delta_h) = cartesian_step(
        positions.row(i),
        cells[i].0,
        projected.slice(s![..dof]),
        cleared,
        jacobian,
    )?;
    for k in 0..9 {
        projected[dof + k] = cleared[k / 3][k % 3];
    }
    Ok((projected, pack_joint(delta_r.view(), delta_h)))
}

#[allow(clippy::too_many_arguments)]
fn assemble_solid(
    config: &BandConfig,
    state: &BandState,
    surface: &dyn BandSurface,
    endpoint_first: ArrayView1<f64>,
    endpoint_last: ArrayView1<f64>,
    cell_first: Cell,
    cell_last: Cell,
    reference: M3,
    jacobian: f64,
    pressure: f64,
    x: ArrayView1<f64>,
) -> Result<(f64, Array1<f64>, f64), SaddleError> {
    let (energies, gradients, stresses, positions, cells) = evaluate_solid(
        state,
        surface,
        endpoint_first,
        endpoint_last,
        cell_first,
        cell_last,
        x,
    )?;
    let n_images = energies.len();
    let dof = endpoint_first.len();
    let seg = dof + 9;
    let mut enthalpy = Array1::zeros(n_images);
    for i in 0..n_images {
        enthalpy[i] = solid_state_enthalpy(energies[i], cells[i].0, reference, pressure)?;
    }
    let mut max_i = 1;
    for i in 2..n_images - 1 {
        if enthalpy[i] > enthalpy[max_i] {
            max_i = i;
        }
    }
    let endpoint_ceiling = enthalpy[0].max(enthalpy[n_images - 1]);
    let climb_target = (enthalpy[max_i] > endpoint_ceiling).then_some(max_i);
    let climbing = config.climbing.is_some();
    let armed = climbing && state.armed.load(Ordering::Relaxed);
    if armed {
        state
            .ci_index
            .store(climb_target.map_or(-1, |i| i as i64), Ordering::Relaxed);
    }
    let ci_at = if armed { climb_target } else { None };
    let mut joint = Array1::zeros(x.len());
    let mut cartesian = Array1::zeros(x.len());
    for i in 1..n_images - 1 {
        let (projected, step) = solid_image(
            config,
            i,
            ci_at == Some(i),
            &positions,
            &cells,
            &enthalpy,
            &gradients,
            &stresses,
            jacobian,
            pressure,
        )?;
        let start = (i - 1) * seg;
        joint.slice_mut(s![start..start + seg]).assign(&projected);
        cartesian.slice_mut(s![start..start + seg]).assign(&step);
    }
    let mut max_component = state.convergence_force(&joint);
    let baseline = {
        let mut slot = state.baseline.lock().unwrap_or_else(|e| e.into_inner());
        *slot.get_or_insert(max_component)
    };
    if let Some(ci) = &config.climbing
        && !armed
        && (max_component < baseline * ci.trigger_factor || max_component < ci.trigger_force)
    {
        state.armed.store(true, Ordering::Relaxed);
        if let Some(ci_i) = climb_target {
            state.ci_index.store(ci_i as i64, Ordering::Relaxed);
            let (projected, step) = solid_image(
                config, ci_i, true, &positions, &cells, &enthalpy, &gradients, &stresses, jacobian,
                pressure,
            )?;
            let start = (ci_i - 1) * seg;
            joint.slice_mut(s![start..start + seg]).assign(&projected);
            cartesian.slice_mut(s![start..start + seg]).assign(&step);
            max_component = state.convergence_force(&joint);
        }
    }
    if let Some(last) = state
        .last
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_mut()
        && last.x == x
    {
        last.projected.clone_from(&cartesian);
    }
    *state
        .last_max_force
        .lock()
        .unwrap_or_else(|e| e.into_inner()) = max_component;
    let pseudo_energy: f64 = enthalpy.slice(s![1..n_images - 1]).sum();
    Ok((pseudo_energy, cartesian, max_component))
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
    let mut projected = if climb {
        let dneb = if config.projection == ProjectionKind::DoublyNudged {
            let fp = force_perp(force.view(), tangent.view());
            dneb_component(spring.full.view(), tangent.view(), fp.view())
        } else {
            Array1::zeros(dof)
        };
        climbing_image_force(force.view(), tangent.view(), dneb.view())
    } else {
        config
            .projection
            .project(force.view(), tangent.view(), &spring)
    };
    if config.remove_translation {
        remove_net_translation(&mut projected);
    }
    Ok(projected)
}

fn max_abs(v: &Array1<f64>) -> f64 {
    v.iter().fold(0.0_f64, |m, c| m.max(c.abs()))
}

/// Assemble the projected band force at interior `x`. Returns
/// (pseudo-energy, projected interior forces, max abs component).
///
/// The climbing image is decided on this evaluation's energies, so
/// the force returned already carries the climb: once armed, the
/// highest interior image climbs while it lies above both fixed
/// endpoints (eOn's rule; a monotonic band keeps the spring on every
/// image); on the evaluation that arms it, the climbing image's force
/// is reassembled before returning.
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
    let endpoint_ceiling = energies[0].max(energies[n_images - 1]);
    let climb_target = (energies[max_i] > endpoint_ceiling).then_some(max_i);
    let climbing = config.climbing.is_some();
    let armed = climbing && state.armed.load(Ordering::Relaxed);
    if armed {
        state
            .ci_index
            .store(climb_target.map_or(-1, |i| i as i64), Ordering::Relaxed);
    }
    let ci_at = if armed { climb_target } else { None };

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
        if let Some(ci_i) = climb_target {
            state.ci_index.store(ci_i as i64, Ordering::Relaxed);
            let f = image_force(
                config,
                ci_i,
                true,
                endpoint_first,
                endpoint_last,
                x,
                &energies,
                &gradients,
            )?;
            projected
                .slice_mut(s![(ci_i - 1) * dof..ci_i * dof])
                .assign(&f);
            max_component = state.convergence_force(&projected);
        }
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
/// FIRE, L-BFGS under `Accept::Step`, Barzilai-Borwein (one oracle
/// call, no test under the session's `Accept::None`), quick-min, and
/// the Pulay residual subspace step on the force alone.
pub(crate) fn check_force_driven(method: &Method) -> Result<(), SaddleError> {
    match method {
        Method::Fire { .. }
        | Method::Lbfgs { .. }
        | Method::Bb
        | Method::QuickMin
        | Method::Diis { .. } => Ok(()),
        other => Err(SaddleError::Invalid(format!(
            "{other:?} tests the oracle value, which is not the potential of a \
             projected or inverted force; use FIRE, L-BFGS, BB, quick-min, or DIIS"
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
        let (e, g) = evaluate_band(
            self.state,
            self.surface,
            positions.row(0),
            positions.row(last),
            x.view(),
        )?;
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
    solid: Option<SolidRun>,
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
        let method = if config.quickmin {
            Method::QuickMin
        } else {
            config.method.clone()
        };
        check_force_driven(&method)?;
        let (positions, solid) = prepare_solid(&config, initial)?;
        let segment = if solid.is_some() { dof + 9 } else { dof };
        let interior_dof = (n_images - 2) * segment;
        let control = Control {
            maxiter: usize::MAX,
            gtol: 0.0,
            istep: 1.0,
            maxmove: None,
            ftol_rel: None,
        };
        let mut solver = Solver::new(method.clone(), control, interior_dof);
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
        if matches!(method, Method::Lbfgs { .. }) {
            solver.set_accept(Accept::Step);
            // eOn's L-BFGS resets: a two-loop step that reaches the
            // per-atom cap or faces away from the force drops its
            // pairs and steps along the force, and an empty memory
            // scales that step by 0.01.
            solver.set_lbfgs_neb_guards(true);
        }
        Ok(Self {
            config,
            positions,
            solver,
            state: BandState::new(),
            iteration: 0,
            rtr: None,
            solid,
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
        if self.solid.is_some() && config.is_some() {
            return Err(SaddleError::Invalid(
                "solid-state band does not use the trust-region stepper".into(),
            ));
        }
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

    /// Oriented cells, one per image, when the band carries the
    /// solid-state block. The Jacobian that built them stays fixed.
    pub fn image_cells(&self) -> Option<&[Cell]> {
        self.solid.as_ref().map(|solid| solid.cells.as_slice())
    }

    /// Jacobian fixed from the mean endpoint volume at session build.
    pub fn solid_jacobian(&self) -> Option<f64> {
        self.solid.as_ref().map(|solid| solid.jacobian)
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
        let x = if self.solid.is_some() {
            self.pack_interior()
        } else {
            self.interior_flat()
        };
        let slot = self.state.last.lock().unwrap_or_else(|e| e.into_inner());
        let last = slot.as_ref().filter(|l| l.x == x)?;
        let n_images = self.positions.nrows();
        let dof = self.positions.ncols();
        let n_interior = n_images - 2;
        let projected = if self.solid.is_some() {
            let seg = dof + 9;
            if last.projected.len() != n_interior * seg {
                return None;
            }
            let mut atomic = Array2::zeros((n_interior, dof));
            for i in 0..n_interior {
                let start = i * seg;
                atomic
                    .row_mut(i)
                    .assign(&last.projected.slice(s![start..start + dof]));
            }
            atomic
        } else {
            last.projected
                .clone()
                .into_shape_with_order((n_interior, dof))
                .ok()?
        };
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
        if self.solid.is_some() {
            if self.rtr.is_some() {
                return Err(SaddleError::Invalid(
                    "solid-state band does not use the trust-region stepper".into(),
                ));
            }
            return self.step_solid(surface);
        }
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

    fn pack_interior(&self) -> Array1<f64> {
        let n_images = self.positions.nrows();
        let dof = self.positions.ncols();
        let seg = dof + 9;
        let solid = self.solid.as_ref().expect("solid-state pack");
        let n = n_images - 2;
        let mut packed = Array1::zeros(n * seg);
        for i in 0..n {
            let start = i * seg;
            packed
                .slice_mut(s![start..start + dof])
                .assign(&self.positions.row(i + 1));
            let cell = pack_joint(ArrayView1::from(&[] as &[f64]), solid.cells[i + 1].0);
            packed.slice_mut(s![start + dof..start + seg]).assign(&cell);
        }
        packed
    }

    fn scatter_packed(&mut self, flat: ArrayView1<f64>) {
        let n_images = self.positions.nrows();
        let dof = self.positions.ncols();
        let seg = dof + 9;
        let solid = self.solid.as_mut().expect("solid-state scatter");
        for i in 0..n_images - 2 {
            let start = i * seg;
            self.positions
                .row_mut(i + 1)
                .assign(&flat.slice(s![start..start + dof]));
            let mut raw = [0.0; 9];
            for k in 0..9 {
                raw[k] = flat[start + dof + k];
            }
            solid.cells[i + 1] = Cell(unpack_cell(&raw));
        }
    }

    fn step_solid<S: BandSurface>(&mut self, surface: &S) -> Result<BandReport, SaddleError> {
        let n_images = self.positions.nrows();
        let dof = self.positions.ncols();
        let interior_dof = (n_images - 2) * (dof + 9);
        let endpoint_first = self.positions.row(0).to_owned();
        let endpoint_last = self.positions.row(n_images - 1).to_owned();
        let (cell_first, cell_last, reference, jacobian, pressure) = {
            let solid = self.solid.as_ref().expect("solid-state step");
            (
                solid.cells[0],
                solid.cells[n_images - 1],
                solid.reference,
                solid.jacobian,
                solid.pressure,
            )
        };
        let config = &self.config;
        let state = &self.state;
        state.take_error();
        state.rows.store(0, Ordering::Relaxed);
        let oracle = Oracle::unbounded(interior_dof, |x: ArrayView1<f64>| {
            match assemble_solid(
                config,
                state,
                surface,
                endpoint_first.view(),
                endpoint_last.view(),
                cell_first,
                cell_last,
                reference,
                jacobian,
                pressure,
                x,
            ) {
                Ok((energy, force, _)) => (energy, -force),
                Err(err) => {
                    state.record_error(err);
                    (f64::INFINITY, Array1::zeros(interior_dof))
                }
            }
        });
        let mut x = self.pack_interior();
        let stepped = self.solver.step(&oracle, &mut x);
        drop(oracle);
        if let Some(err) = self.state.take_error() {
            return Err(err);
        }
        stepped.map_err(|e| SaddleError::Solver(e.to_string()))?;
        self.scatter_packed(x.view());
        self.iteration += 1;
        let at_last = self
            .state
            .last
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .is_some_and(|last| last.x == x && last.projected.len() == interior_dof);
        if !at_last {
            assemble_solid(
                &self.config,
                &self.state,
                surface,
                endpoint_first.view(),
                endpoint_last.view(),
                cell_first,
                cell_last,
                reference,
                jacobian,
                pressure,
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
        assemble_band(
            &self.config,
            &self.state,
            surface,
            first.view(),
            last.view(),
            initial.view(),
        )?;
        let mut trial = self.positions.clone();
        let cached = CachedBandSurface {
            state: &self.state,
            surface,
        };
        let rtr = self.rtr.as_mut().expect("RTR configuration");
        rtr.climb = self.state.ci().is_some();
        rtr.set_force_gate(self.state.force_gate);
        rtr.step(&self.config, &cached, &mut trial)?;
        let interior: Array1<f64> = trial
            .slice(s![1..trial.nrows() - 1, ..])
            .iter()
            .copied()
            .collect();
        assemble_band(
            &self.config,
            &self.state,
            surface,
            first.view(),
            last.view(),
            interior.view(),
        )?;
        self.positions.assign(&trial);
        self.iteration += 1;
        let max_force = self.state.last_max_force();
        Ok(BandReport {
            status: if max_force <= self.config.force_tol {
                BandStatus::Converged
            } else {
                BandStatus::Running
            },
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

    struct Prescribed {
        energy: Vec<f64>,
        gradient: Array2<f64>,
    }

    impl BandSurface for Prescribed {
        fn eval(
            &self,
            positions: ArrayView2<f64>,
            energies: &mut Array1<f64>,
            gradients: &mut Array2<f64>,
        ) -> Result<(), SaddleError> {
            assert_eq!(positions.nrows(), self.energy.len());
            for (i, energy) in self.energy.iter().enumerate() {
                energies[i] = *energy;
            }
            gradients.assign(&self.gradient);
            Ok(())
        }
    }

    fn infinite_climb() -> BandConfig {
        BandConfig {
            climbing: Some(CiConfig {
                trigger_factor: 1.0,
                trigger_force: f64::INFINITY,
            }),
            ..BandConfig::default()
        }
    }

    /// Two atoms, three images at equal spacing; the tangent lies along
    /// the first atom's x. The force on the middle image is (2, 3, 0)
    /// and (5, 1, 0).
    fn two_atom_band() -> (Array2<f64>, Array2<f64>) {
        let positions = Array2::from_shape_vec(
            (3, 6),
            vec![
                0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 1.0, 0.0, 0.0, 5.0, 0.0, 0.0, 2.0, 0.0, 0.0, 5.0,
                0.0, 0.0,
            ],
        )
        .unwrap();
        let gradient = Array2::from_shape_vec(
            (3, 6),
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, -3.0, 0.0, -5.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0,
            ],
        )
        .unwrap();
        (positions, gradient)
    }

    fn assemble(
        config: &BandConfig,
        energy: Vec<f64>,
        positions: &Array2<f64>,
        gradient: Array2<f64>,
    ) -> (Array1<f64>, BandState) {
        let surface = Prescribed { energy, gradient };
        let state = BandState::new();
        let (_, projected, _) = assemble_band(
            config,
            &state,
            &surface,
            positions.row(0),
            positions.row(2),
            positions.row(1),
        )
        .unwrap();
        (projected, state)
    }

    fn assert_force(got: &Array1<f64>, want: &[f64]) {
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want) {
            assert_abs_diff_eq!(*g, *w, epsilon = 1e-12);
        }
    }

    #[test]
    fn net_translation_leaves_a_multi_atom_image_force() {
        let (positions, gradient) = two_atom_band();
        let (projected, _) = assemble(
            &BandConfig::default(),
            vec![0.0, 1.0, 5.0],
            &positions,
            gradient,
        );
        // The perpendicular force is (0, 3, 0, 5, 1, 0); its mean
        // (2.5, 2, 0) leaves atom 0 at (-2.5, 1, 0) and atom 1 at
        // (2.5, -1, 0).
        assert_force(&projected, &[-2.5, 1.0, 0.0, 2.5, -1.0, 0.0]);
    }

    #[test]
    fn trust_region_force_drops_net_translation() {
        let (positions, gradient) = two_atom_band();
        let surface = Prescribed {
            energy: vec![0.0, 1.0, 5.0],
            gradient,
        };
        let forces =
            crate::rtr::band_forces(&BandConfig::default(), &surface, positions.view(), None)
                .unwrap();
        assert_force(&forces.force, &[-2.5, 1.0, 0.0, 2.5, -1.0, 0.0]);
    }

    #[test]
    fn keep_translation_leaves_the_mean_in_the_force() {
        let (positions, gradient) = two_atom_band();
        let config = BandConfig {
            remove_translation: false,
            ..BandConfig::default()
        };
        let (projected, _) = assemble(&config, vec![0.0, 1.0, 5.0], &positions, gradient);
        assert_force(&projected, &[0.0, 3.0, 0.0, 5.0, 1.0, 0.0]);
    }

    #[test]
    fn one_atom_image_keeps_its_force() {
        let positions =
            Array2::from_shape_vec((3, 3), vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0])
                .unwrap();
        let gradient =
            Array2::from_shape_vec((3, 3), vec![0.0, 0.0, 0.0, -1.0, -2.0, -3.0, 0.0, 0.0, 0.0])
                .unwrap();
        let (projected, _) = assemble(
            &BandConfig::default(),
            vec![0.0, 1.0, 5.0],
            &positions,
            gradient,
        );
        assert_force(&projected, &[0.0, 2.0, 3.0]);
    }

    #[test]
    fn monotonic_band_keeps_the_spring_on_every_image() {
        let (positions, gradient) = two_atom_band();
        let (projected, state) =
            assemble(&infinite_climb(), vec![0.0, 1.0, 5.0], &positions, gradient);
        assert_force(&projected, &[-2.5, 1.0, 0.0, 2.5, -1.0, 0.0]);
        assert_eq!(state.ci(), None);
    }

    #[test]
    fn interior_peak_climbs() {
        let (positions, gradient) = two_atom_band();
        let (projected, state) =
            assemble(&infinite_climb(), vec![0.0, 4.0, 1.0], &positions, gradient);
        // Climbing force (-2, 3, 0, 5, 1, 0) minus its mean (1.5, 2, 0).
        assert_force(&projected, &[-3.5, 1.0, 0.0, 3.5, -1.0, 0.0]);
        assert_eq!(state.ci(), Some(1));
    }
}
