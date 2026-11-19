//! The band session: assemble NEB forces on a caller surface, take
//! one rgmin solver step, report. Hosts own the loop.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicUsize, Ordering};

use ndarray::{
    Array1, Array2, ArrayBase, ArrayView1, ArrayView2, ArrayViewMut1, DataMut, Ix1, Zip, s,
};
use rgmin::{Accept, Control, Exp, ExpParams, Method, Neighbours, Oracle, Solver};

use crate::cell_log::{M3, m3_det};
use crate::error::SaddleError;
use crate::force::ForceGate;
use crate::projection::{ProjectionKind, climbing_image_force, dneb_component, force_perp};
use crate::saddle_index::{SaddleIndex, SaddleIndexCheck};
use crate::solid_state::{
    cartesian_step, cell_neb_force, joint_displacement, orient_lower_triangular, pack_joint,
    solid_state_enthalpy, solid_state_jacobian, unpack_cell,
};
use crate::spring::SpringKind;
use crate::tangent::TangentKind;
use crate::tangent::compute_tangent;

mod climb;
mod free_end;
mod growing_string;
mod dynamic;
mod min_mode;
mod pair_model;
mod saddle_check;
mod string;
mod zoom;
pub use free_end::{FeaNebReport, FeaNebSession, FreeEndConfig, FreeEndReport, FreeEndSide};
pub use growing_string::{
    GrowingStringConfig, GrowingStringPhase, GrowingStringReport, GrowingStringSession,
};
pub use dynamic::DynamicNeb;
pub(crate) use dynamic::dynamic_mask;
pub use min_mode::MinModeRefine;
pub(crate) use min_mode::RefinePhase;
pub use string::{StringConfig, StringSession};
pub use zoom::ZoomConfig;

fn lock_guard<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

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

    /// Convert a non-singular three-axis periodic cell to [`minimage::Cell`].
    ///
    /// Returns `None` when any row is zero (a slab or open axis) or the
    /// lattice matrix is singular.
    pub fn as_minimage(&self) -> Option<minimage::Cell> {
        let periodic: [bool; 3] = std::array::from_fn(|i| self.0[i].iter().any(|v| *v != 0.0));
        if !periodic.iter().all(|p| *p) {
            return None;
        }
        self.inverse()?;
        minimage::Cell::from_ase(self.0).ok()
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
        self.minimum_image_view(diff.view_mut());
    }

    /// [`Cell::minimum_image`] on borrowed storage.
    pub(crate) fn minimum_image_view(&self, mut diff: ArrayViewMut1<f64>) {
        if let Some(cell) = self.as_minimage() {
            for a in 0..diff.len() / 3 {
                let d = [diff[3 * a], diff[3 * a + 1], diff[3 * a + 2]];
                let wrapped = cell.displacement([0.0; 3], d);
                diff[3 * a] = wrapped[0];
                diff[3 * a + 1] = wrapped[1];
                diff[3 * a + 2] = wrapped[2];
            }
            return;
        }
        let h = &self.0;
        let periodic: [bool; 3] = std::array::from_fn(|i| h[i].iter().any(|v| *v != 0.0));
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

    /// Nonzero lattice translations, along the axes
    /// [`Cell::minimum_image`] wraps, that can carry a wrapped
    /// displacement to another image shorter than `cutoff`. Empty when
    /// every wrapped axis has a cell height of more than twice the
    /// cutoff, where the minimum image is the only image in range.
    pub(crate) fn image_translations(&self, cutoff: f64) -> Vec<[f64; 3]> {
        // Bounds the enumeration on a cell far thinner than the cutoff.
        const MAX_REACH: f64 = 16.0;
        let h = &self.0;
        let periodic: [bool; 3] = std::array::from_fn(|i| h[i].iter().any(|v| *v != 0.0));
        // A wrapped displacement has a fractional component of at most
        // 1/2 along each wrapped axis, so the image n steps away along
        // it is at least (|n| - 1/2) heights long.
        let reach = |height: f64| (cutoff / height + 0.5).floor().clamp(0.0, MAX_REACH) as i64;
        let mut range = [0_i64; 3];
        if periodic.iter().all(|p| *p)
            && let Some(inv) = self.inverse()
        {
            for (i, slot) in range.iter_mut().enumerate() {
                let norm = (inv[0][i].powi(2) + inv[1][i].powi(2) + inv[2][i].powi(2)).sqrt();
                *slot = reach(1.0 / norm);
            }
        } else {
            for (c, slot) in range.iter_mut().enumerate() {
                let l = h[c][c];
                let axis_aligned = (0..3).all(|k| k == c || h[c][k] == 0.0);
                if periodic[c] && axis_aligned && l > 0.0 {
                    *slot = reach(l);
                }
            }
        }
        let mut out = Vec::new();
        for n0 in -range[0]..=range[0] {
            for n1 in -range[1]..=range[1] {
                for n2 in -range[2]..=range[2] {
                    if (n0, n1, n2) == (0, 0, 0) {
                        continue;
                    }
                    let n = [n0 as f64, n1 as f64, n2 as f64];
                    out.push(std::array::from_fn(|c| {
                        n[0] * h[0][c] + n[1] * h[1][c] + n[2] * h[2][c]
                    }));
                }
            }
        }
        out
    }
}

/// Climbing-image activation, the eOn trigger rule: CI arms when the
/// convergence force falls under `trigger_factor * baseline` (eOn
/// `ci_after_rel`, the baseline is the first evaluation's force) or
/// under `trigger_force` (eOn `ci_after`).
///
/// The default, [`CiConfig::after`]`(0.5)`, arms the climbing image
/// once the band has settled to a largest force component under 0.5
/// in the host's force units (eOn's recommended `ci_after`, eV/A), as
/// climbing-image NEB is run once the band is roughly converged
/// (Henkelman, Uberuaga and Jonsson, J. Chem. Phys. 113, 9901 (2000),
/// doi:10.1063/1.1329672). [`CiConfig::always`] climbs from the first
/// evaluation.
///
/// Non-exhaustive: start from [`CiConfig::after`],
/// [`CiConfig::after_rel`], [`CiConfig::always`] or
/// [`CiConfig::default`] and chain the `with_*` setters.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct CiConfig {
    /// The trigger factor.
    pub trigger_factor: f64,
    /// The trigger force.
    pub trigger_force: f64,
}

with_setters!(CiConfig {
    with_trigger_factor => trigger_factor: val f64,
    with_trigger_force => trigger_force: val f64,
});

impl CiConfig {
    /// Arm once the largest force component is under `force` (eOn
    /// `ci_after` with `ci_after_rel = 0`).
    pub fn after(force: f64) -> Self {
        Self {
            trigger_factor: 0.0,
            trigger_force: force,
        }
    }

    /// Arm once the force is under `factor` times the first
    /// evaluation's (eOn `ci_after_rel`).
    pub fn after_rel(factor: f64) -> Self {
        Self {
            trigger_factor: factor,
            trigger_force: 0.0,
        }
    }

    /// Climb from the first evaluation.
    pub fn always() -> Self {
        Self {
            trigger_factor: 1.0,
            trigger_force: f64::INFINITY,
        }
    }
}

impl Default for CiConfig {
    fn default() -> Self {
        Self::after(0.5)
    }
}

/// Per-image cells for the generalized solid-state band.
///
/// Sheppard, Xiao, Chemelewski, Johnson, and Henkelman, J. Chem. Phys.
/// 136, 074103 (2012). The Jacobian is fixed from the mean endpoint
/// volume when the session is built. Each cell is put in
/// lower-triangular form, and that image's positions rotate with it.
///
/// Fields may be added: build it with [`SolidState::new`] and the
/// `with_*` setters.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SolidState {
    /// One cell per image, endpoints included.
    pub cells: Vec<Cell>,
    /// Hydrostatic pressure added to the Cauchy stress. A positive
    /// value pushes the cell inward.
    pub pressure: f64,
    /// Positive scale on the Jacobian. 1 is the 2012 value.
    pub weight: f64,
}

impl SolidState {
    /// One cell per image, zero pressure, the 2012 Jacobian (weight 1).
    pub fn new(cells: Vec<Cell>) -> Self {
        Self {
            cells,
            pressure: 0.0,
            weight: 1.0,
        }
    }

    /// One cell per image, endpoints included.
    #[must_use]
    pub fn with_cells(mut self, cells: Vec<Cell>) -> Self {
        self.cells = cells;
        self
    }

    /// Hydrostatic pressure, energy per length cubed.
    #[must_use]
    pub fn with_pressure(mut self, pressure: f64) -> Self {
        self.pressure = pressure;
        self
    }

    /// Positive scale on the Jacobian.
    #[must_use]
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }
}

/// Band configuration. `force_tol` is on the max absolute component
/// of the projected force over interior images.
///
/// Non-exhaustive: start from [`BandConfig::default`] and chain the
/// `with_*` setters.
///
/// ```
/// use rgsaddle::{BandConfig, CiConfig};
///
/// let config = BandConfig::default()
///     .with_force_tol(1e-4)
///     .with_climbing(CiConfig::always())
///     .with_min_mode(None);
/// assert_eq!(config.force_tol, 1e-4);
/// assert!(config.min_mode.is_none());
/// ```
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct BandConfig {
    /// How the path direction at an interior image is chosen.
    pub tangent: TangentKind,
    /// How the springs between images are chosen.
    pub spring: SpringKind,
    /// How the true force and the spring are combined.
    pub projection: ProjectionKind,
    /// When set, a climbing image may arm. None keeps every image on the spring.
    pub climbing: Option<CiConfig>,
    /// Simulation cell. Row i is lattice vector a_i.
    pub cell: Option<Cell>,
    /// The search stops when the largest absolute force component falls under this value, in the host's force units.
    pub force_tol: f64,
    /// Largest per-atom displacement, in the host's length unit.
    pub max_move: f64,
    /// Drop the mean Cartesian force of every image that has two or
    /// more atoms. eOn does this when every atom is free and leaves
    /// the mean in place when any atom is fixed; a band with any atom
    /// set in [`BandConfig::fixed_atoms`] keeps the mean whatever this
    /// flag says.
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
    /// Quick-min variant when `quickmin` is set: `QuickMinOptions::rgmin`
    /// (default; FIRE time-step control, one projection),
    /// `QuickMinOptions::eon` (eOn's fixed `dt`), or
    /// `QuickMinOptions::per_atom` (a projection per atom).
    pub quickmin_options: rgmin::QuickMinOptions,
    /// Initial time step of FIRE and quick-min (rgmin `Control::istep`);
    /// quick-min `QuickMinOptions::eon` keeps it fixed. 1 by default. On
    /// the EAM Al vacancy hop and the Pt heptamer bands the eOn variant
    /// converges at 0.1 and 0.25 and not at 0.5 or 1.
    pub time_step: f64,
    /// `Some` selects AutoNEB on this configuration. `None` is a fixed band.
    pub autoneb: Option<crate::autoneb::AutoNebTuning>,
    /// OCI-NEB min-mode refinement of the armed climbing image
    /// (Goswami, Gunde, and Jónsson, Front. Chem. 14, 1807063 (2026)).
    /// On by default; it needs `climbing` and runs on the atomic band,
    /// not on the solid-state block. `None` keeps the plain CI-NEB.
    pub min_mode: Option<MinModeRefine>,
    /// Per-image exponential metric on the band solver. `None` runs
    /// the stepper in the Euclidean metric. Refused on a solid-state
    /// band, whose cell block has no pair geometry.
    pub exp_precon: Option<ExpBandPrecon>,
    /// Zoom-NEB: once the climbing image is stable, place the band on
    /// a window around it. `None`, the default, keeps the band whole.
    pub zoom: Option<ZoomConfig>,
    /// Per-atom fixed mask shared by every image, one entry per atom;
    /// empty (the default) or all `false` leaves every atom free and
    /// the band bitwise as without a mask. A fixed atom never moves:
    /// its gradient reads as zero, so it carries no force in any
    /// stepper, in the convergence norm or in the report; tangents and
    /// springs are formed on the free coordinates; the climbing image
    /// and the min-mode refinement step the free block only; and the
    /// exponential metric is assembled on the free atoms, with fixed
    /// atoms still counted as neighbours. With any atom fixed the mean
    /// force stays (see [`BandConfig::remove_translation`]). The
    /// endpoints must agree on every fixed atom up to the minimum image
    /// (1e-10 per component); interior images take the first image's
    /// fixed coordinates. Refused on a solid-state band.
    pub fixed_atoms: Vec<bool>,
    /// Certify the climbing image once the band converges: its lowest
    /// curvatures on the free coordinates, off the rigid motions the
    /// band removes, by block Lanczos on finite-difference Hessian
    /// actions (see [`mod@crate::saddle_index`]). The index is
    /// [`BandReport::saddle_index`] and the actions count in
    /// [`BandReport::surface_rows`]. On by default; the band itself
    /// steps as without it. With [`SaddleIndexCheck::escape`] set, a
    /// climbing image of index two or more is pushed off its extra
    /// negative modes and the min-mode refinement climbs again from
    /// there, seeded with the negative mode closest to the band
    /// tangent. `None` skips the check. A solid-state band is not
    /// checked.
    pub saddle_check: Option<SaddleIndexCheck>,
    /// Simplified string method of E, Ren, and Vanden-Eijnden, J. Chem.
    /// Phys. 126, 164103 (2007), doi:10.1063/1.2720838: evolve interior
    /// images under the full potential force (no spring, no tangent
    /// projection) and reparametrize to equal arc length after each
    /// step, anchoring the climbing image when armed. `None`, the
    /// default, runs the nudged elastic band.
    pub string: Option<StringConfig>,
    /// Free-end adaptive nudged elastic band (FEA-NEB) of Zhang, Zhang,
    /// Ye, and Zheng, J. Chem. Phys. 145, 094104 (2016),
    /// doi:10.1063/1.4962019: one terminal image moves subject to the
    /// energy level-set constraint \(E(x_{\mathrm{end}}) = E_{\mathrm{target}}\)
    /// with repeated adaptive cut-and-remesh around the detected saddle.
    /// `None`, the default, keeps both endpoints fixed.
    pub free_end: Option<FreeEndConfig>,
    /// Dynamic NEB ([`DynamicNeb`]). `None`, the default, evaluates
    /// every image on every step. Images under the threshold keep their
    /// last energy and force, stay in the spring and the tangent, and
    /// are evaluated again when a neighbour moves enough that the
    /// projected force exceeds the threshold.
    pub dynamic: Option<DynamicNeb>,
}

with_setters!(BandConfig {
    with_tangent => tangent: val TangentKind,
    with_spring => spring: val SpringKind,
    with_projection => projection: val ProjectionKind,
    with_climbing => climbing: opt CiConfig,
    with_cell => cell: opt Cell,
    with_force_tol => force_tol: val f64,
    with_max_move => max_move: val f64,
    with_remove_translation => remove_translation: val bool,
    with_method => method: val Method,
    with_solid => solid: opt SolidState,
    with_quickmin => quickmin: val bool,
    with_quickmin_options => quickmin_options: val rgmin::QuickMinOptions,
    with_time_step => time_step: val f64,
    with_autoneb => autoneb: opt crate::autoneb::AutoNebTuning,
    with_min_mode => min_mode: opt MinModeRefine,
    with_exp_precon => exp_precon: opt ExpBandPrecon,
    with_zoom => zoom: opt ZoomConfig,
    with_fixed_atoms => fixed_atoms: val Vec<bool>,
    with_saddle_check => saddle_check: opt SaddleIndexCheck,
    with_string => string: opt StringConfig,
    with_free_end => free_end: opt FreeEndConfig,
    with_dynamic => dynamic: opt DynamicNeb,
});

impl BandConfig {
    /// Whether [`BandConfig::fixed_atoms`] holds any atom.
    pub(crate) fn has_fixed(&self) -> bool {
        self.fixed_atoms.iter().any(|&f| f)
    }

    /// The mean force is dropped only on a band with every atom free.
    pub(crate) fn drops_translation(&self) -> bool {
        self.remove_translation && !self.has_fixed()
    }
}

/// Zero the coordinates of the atoms `fixed` holds in `v`, a flat run
/// of whole images of `fixed.len()` atoms each.
pub(crate) fn zero_fixed<S: DataMut<Elem = f64>>(fixed: &[bool], v: &mut ArrayBase<S, Ix1>) {
    let n = fixed.len();
    if n == 0 || !fixed.iter().any(|&f| f) {
        return;
    }
    for (k, c) in v.iter_mut().enumerate() {
        if fixed[(k / 3) % n] {
            *c = 0.0;
        }
    }
}

/// A non-empty fixed mask has one entry per atom.
pub(crate) fn check_fixed_len(config: &BandConfig, dof: usize) -> Result<(), SaddleError> {
    let n = config.fixed_atoms.len();
    if n != 0 && 3 * n != dof {
        return Err(SaddleError::Shape(format!(
            "fixed-atom mask has {n} entries for {} atoms",
            dof / 3
        )));
    }
    Ok(())
}

/// The fixed mask fits the band: one entry per atom, a free atom left,
/// no cell block, and endpoints that agree on every fixed atom up to
/// the minimum image.
pub(crate) fn check_fixed(
    config: &BandConfig,
    positions: ArrayView2<f64>,
) -> Result<(), SaddleError> {
    check_fixed_len(config, positions.ncols())?;
    if !config.has_fixed() {
        return Ok(());
    }
    if config.solid.is_some() {
        return Err(SaddleError::Invalid(
            "a solid-state band moves every atom with its cell; fixed atoms are not supported"
                .into(),
        ));
    }
    if config.fixed_atoms.iter().all(|&f| f) {
        return Err(SaddleError::Invalid(
            "every atom is fixed; a band needs a free atom".into(),
        ));
    }
    let last = positions.nrows() - 1;
    let mut diff = &positions.row(last) - &positions.row(0);
    if let Some(cell) = &config.cell {
        cell.minimum_image(&mut diff);
    }
    for (atom, _) in config.fixed_atoms.iter().enumerate().filter(|(_, f)| **f) {
        let gap = (0..3).fold(0.0_f64, |m, c| m.max(diff[3 * atom + c].abs()));
        if gap > SAME_ROW_TOL {
            return Err(SaddleError::Invalid(format!(
                "fixed atom {atom} differs between the endpoints by {gap:e}; \
                 a fixed atom must sit at the same place in every image"
            )));
        }
    }
    Ok(())
}

/// Copy the first row's fixed coordinates into every interior row.
pub(crate) fn pin_fixed_rows(fixed: &[bool], positions: &mut Array2<f64>) {
    if fixed.is_empty() {
        return;
    }
    let n_images = positions.nrows();
    for (atom, _) in fixed.iter().enumerate().filter(|(_, f)| **f) {
        for c in 3 * atom..3 * atom + 3 {
            let held = positions[(0, c)];
            for i in 1..n_images - 1 {
                positions[(i, c)] = held;
            }
        }
    }
}

/// Coordinate indices of the free atoms of one image.
fn free_coordinates(fixed: &[bool]) -> Vec<usize> {
    fixed
        .iter()
        .enumerate()
        .filter(|(_, f)| !**f)
        .flat_map(|(a, _)| 3 * a..3 * a + 3)
        .collect()
}

impl Default for BandConfig {
    fn default() -> Self {
        Self {
            tangent: TangentKind::Improved,
            spring: SpringKind::Uniform { k: 5.0 },
            projection: ProjectionKind::Neb,
            climbing: Some(CiConfig::default()),
            cell: None,
            force_tol: 1e-3,
            max_move: 0.2,
            remove_translation: true,
            method: Method::fire(rgmin::FireKind::V2),
            solid: None,
            quickmin: false,
            quickmin_options: rgmin::QuickMinOptions::rgmin(),
            time_step: 1.0,
            autoneb: None,
            min_mode: Some(MinModeRefine::default()),
            exp_precon: None,
            zoom: None,
            fixed_atoms: Vec::new(),
            saddle_check: Some(SaddleIndexCheck::default()),
            string: None,
            free_end: None,
            dynamic: None,
        }
    }
}

/// Exponential metric of Packwood, Kermode, Mones, Bernstein, Woolley,
/// Gould, Ortner and Csányi, J. Chem. Phys. 144, 164109 (2016), built
/// per interior image and applied inside the band projection as Makri,
/// Ortner and Kermode, J. Chem. Phys. 150, 094109 (2019), Eq. (9) and
/// (10), write it: the image's driving force is `-(P^{-1} g - t_P (t_P
/// . g)) + k ((d_next - d_prev) . P t_P) t_P` with `t_P` the tangent
/// normalised in the `P`-norm. The band solver then integrates that
/// force in the Euclidean metric, so [`Method::Ode12r`] is ODE12r-Exp
/// and FIRE or L-BFGS step on the same preconditioned force. The band
/// relaxes to the minimum energy path of the metric's geometry, where
/// the gradient lies along `P t_P`; the band force that convergence and
/// the report measure is the paper's residual, Eq. (11), `-(g - P t_P
/// (t_P . g))` per image, which vanishes on that path where the
/// Euclidean perpendicular force does not.
///
/// Pairs within `r_cut_factor * r_nn` of one another in the same image
/// enter the matrix, `r_nn` being the largest per-atom nearest-neighbour
/// distance over the interior images; a cell wraps the distances. The
/// matrices are rebuilt when any atom has moved more than `r_nn / 2`
/// since the last build; an energy scale that is not given is estimated
/// once by rgmin's sine probe on the first interior image, which costs
/// one surface row. Only a uniform spring is supported with the metric;
/// its constant `k` enters as the paper's dimensionless `kappa = k /
/// mu`. An armed climbing image reflects the preconditioned gradient in
/// the P-metric, `-(P^{-1} g - 2 t_P (t_P . g))`, whose fixed point is
/// the saddle.
///
/// Non-exhaustive: start from [`ExpBandPrecon::default`] and chain the
/// `with_*` setters.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct ExpBandPrecon {
    /// Exponent `A` of `exp(-A (r / r_nn - 1))`.
    pub a: f64,
    /// Diagonal stabilization `C_stab`.
    pub c_stab: f64,
    /// Energy scale `mu`. `None` lets rgmin estimate it from the band
    /// force on the first step, which costs band evaluations.
    pub mu: Option<f64>,
    /// Cutoff as a multiple of `r_nn`.
    pub r_cut_factor: f64,
}

with_setters!(ExpBandPrecon {
    with_a => a: val f64,
    with_c_stab => c_stab: val f64,
    with_mu => mu: opt f64,
    with_r_cut_factor => r_cut_factor: val f64,
});

impl Default for ExpBandPrecon {
    fn default() -> Self {
        Self {
            a: 3.0,
            c_stab: 0.1,
            mu: None,
            r_cut_factor: 2.0,
        }
    }
}

/// Identifies the metric matrices an assembly solved against. Zero is
/// the Euclidean metric; positive values number each [`ExpBandState`]
/// build on the session.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MetricBuildId(u64);

impl MetricBuildId {
    const EUCLIDEAN: Self = Self(0);
}

/// The band metric: one matrix per interior image and where they were built.
#[derive(Clone, Debug)]
struct ExpBandState {
    precon: ExpBandPrecon,
    build_id: Option<MetricBuildId>,
    built_at: Array1<f64>,
    r_nn: f64,
    /// Energy scale in force; `None` until estimated on the first step.
    mu: Option<f64>,
    /// One matrix per interior image, empty until the first step.
    metrics: Vec<Exp>,
    /// Coordinates the matrices act on when an atom is fixed; `None`
    /// is every coordinate.
    free: Option<Vec<usize>>,
}

impl ExpBandState {
    fn clear_metrics(&mut self) {
        self.build_id = None;
        self.built_at = Array1::zeros(0);
        self.r_nn = 0.0;
        self.metrics.clear();
    }

    fn is_stale(&self, x: ArrayView1<f64>, n_interior: usize) -> bool {
        self.build_id.is_none()
            || self.metrics.len() != n_interior
            || self.built_at.len() != x.len()
            || rgmin::needs_rebuild(
                rgmin::max_atom_displacement(self.built_at.view(), x),
                self.r_nn,
            )
    }

    /// Metrics valid at `x`, or `None` before the first build or once an
    /// interior atom has moved more than `r_nn / 2` from `built_at`.
    fn fresh_metrics(&self, x: ArrayView1<f64>, n_interior: usize) -> Option<BandMetrics<'_>> {
        let build_id = self.build_id?;
        if self.is_stale(x, n_interior) {
            return None;
        }
        Some(BandMetrics {
            build_id,
            built_at: self.built_at.view(),
            r_nn: self.r_nn,
            per_image: self.metrics.as_slice(),
            free: self.free.as_deref(),
        })
    }
}

/// The band metrics of one step: one matrix per interior image, on the
/// coordinates `free` lists, or on all of them.
#[derive(Clone, Copy)]
struct BandMetrics<'a> {
    build_id: MetricBuildId,
    built_at: ArrayView1<'a, f64>,
    r_nn: f64,
    per_image: &'a [Exp],
    free: Option<&'a [usize]>,
}

impl BandMetrics<'_> {
    fn is_fresh_at(&self, x: ArrayView1<f64>, n_interior: usize) -> bool {
        self.build_id != MetricBuildId::EUCLIDEAN
            && self.per_image.len() == n_interior
            && self.built_at.len() == x.len()
            && !rgmin::needs_rebuild(rgmin::max_atom_displacement(self.built_at, x), self.r_nn)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// The BandStatus.
pub enum BandStatus {
    /// The Running.
    Running,
    /// The Converged.
    Converged,
}

/// One step's report.
#[derive(Clone, Debug)]
pub struct BandReport {
    /// Whether this step is still running or has converged.
    pub status: BandStatus,
    /// Largest absolute force component, in the host's force units.
    pub max_force: f64,
    /// Index of the climbing image, or none when no image climbs.
    pub ci_index: Option<usize>,
    /// Number of steps taken.
    pub iteration: usize,
    /// Image evaluations the surface performed during this step: the
    /// sum of the row counts over every [`BandSurface::eval`] call.
    /// A warm step costs `n_images - 2`; the first step after
    /// [`BandSession::new`] or [`BandSession::reset`] adds the two
    /// endpoints.
    pub surface_rows: usize,
    /// Curvature along the climbing image's lowest mode on a step of
    /// the min-mode refinement; `None` on a band step.
    pub curvature: Option<f64>,
    /// Saddle index of the climbing image, certified when the band
    /// converged ([`BandConfig::saddle_check`]). `None` while the band
    /// runs, without a climbing image, or with the check off. A step
    /// that pushes off a saddle of higher index reports that index and
    /// runs on.
    pub saddle_index: Option<SaddleIndex>,
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
    /// Fill `energies[r]` and row `r` of `gradients` for each row `r` of
    /// `positions`.
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError>;

    /// Fill `energies[r]` and row `r` of `gradients` for each row `r` of
    /// `positions`, tagged with its stable [`crate::stencil::ImageId`].
    /// Defaults to [`BandSurface::eval`], and [`crate::stencil::Surface`]
    /// overrides it to forward `ids` to [`crate::stencil::Surface::eval`].
    fn eval_with_ids(
        &self,
        ids: &[crate::stencil::ImageId],
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        let _ = ids;
        self.eval(positions, energies, gradients)
    }

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

    /// Energy and Cartesian gradient of band image `image` alone. The
    /// min-mode refinement calls this for the climbing image; the
    /// default is a one-row [`BandSurface::eval`], so a host that keys
    /// per-image state on the row count overrides it.
    fn eval_image(
        &self,
        image: usize,
        position: ArrayView1<f64>,
    ) -> Result<(f64, Array1<f64>), SaddleError> {
        let _ = image;
        min_mode::one_row(self, position)
    }

    /// Energy and Cartesian gradient of band image `image` alone, for a
    /// dynamic-NEB refresh of one interior image. The default is a
    /// one-row [`BandSurface::eval`]. A host that keys per-image state
    /// on the band index overrides it. This is not the climbing-image
    /// request: [`BandSurface::eval_image`] stays that channel.
    fn eval_indexed(
        &self,
        image: usize,
        position: ArrayView1<f64>,
    ) -> Result<(f64, Array1<f64>), SaddleError> {
        let _ = image;
        min_mode::one_row(self, position)
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
/// evaluation returned); `projected` and `driving` are the assembled
/// residual and optional preconditioned force under the climbing state
/// of the most recent assembly.
#[derive(Clone)]
struct BandEval {
    x: Array1<f64>,
    energies: Array1<f64>,
    gradients: Array2<f64>,
    projected: Array1<f64>,
    driving: Array1<f64>,
    /// Climbing image used by the cached assembly; `i64::MIN` means none.
    assembly_ci: i64,
    /// Metric build used by the cached assembly; `None` when unassembled.
    assembly_metric: Option<MetricBuildId>,
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
    /// Largest per-atom force of the first assembly, the OCI-NEB `F_0`.
    atom_baseline: Mutex<Option<f64>>,
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
    /// The surface answer `last` replaced. A stepper that rejects its
    /// trial and stays at the start (ODE12r) finds the start here, so
    /// the rejection costs one evaluation, not two.
    previous: Mutex<Option<BandEval>>,
    /// Rows evaluated by the surface during the current step.
    rows: AtomicUsize,
    /// Number of exponential-metric PCG solves, for cache regression tests.
    pcg_solves: AtomicUsize,
    /// Monotonic identifier handed to each exponential-metric build.
    next_metric_id: AtomicU64,
    /// Buffers of the evaluation and assembly. Locked before `last`.
    work: Mutex<BandWork>,
    /// Image threads ([`BandSession::set_image_threads`]).
    pool: Option<crate::autoneb::ImageEvalPool>,
    /// The band's fixed mask, empty when every atom is free. Gradient
    /// components of fixed atoms are zeroed as the surface answers.
    fixed: Vec<bool>,
    /// Stable per-image identities when tracking across growth or remeshing.
    ids: Vec<crate::stencil::ImageId>,
    /// Dynamic NEB cache and the images pinned for the step in progress.
    dynamic: Mutex<dynamic::DynamicRun>,
}

impl BandState {
    fn new() -> Self {
        Self {
            fixed: Vec::new(),
            ids: Vec::new(),
            force_gate: ForceGate::LinfNorm,
            baseline: Mutex::new(None),
            atom_baseline: Mutex::new(None),
            armed: AtomicBool::new(false),
            ci_index: AtomicI64::new(-1),
            surface_error: Mutex::new(None),
            last_max_force: Mutex::new(f64::INFINITY),
            endpoints: Mutex::new(None),
            last: Mutex::new(None),
            previous: Mutex::new(None),
            rows: AtomicUsize::new(0),
            pcg_solves: AtomicUsize::new(0),
            next_metric_id: AtomicU64::new(1),
            work: Mutex::new(BandWork::default()),
            pool: None,
            dynamic: Mutex::new(dynamic::DynamicRun::default()),
        }
    }

    fn dynamic_run(&self) -> std::sync::MutexGuard<'_, dynamic::DynamicRun> {
        self.dynamic.lock().unwrap_or_else(|err| err.into_inner())
    }

    fn dynamic_enabled(&self) -> bool {
        self.dynamic_run().spec.is_some()
    }

    fn dynamic_spec(&self) -> Option<DynamicNeb> {
        self.dynamic_run().spec
    }

    fn install_dynamic(&self, spec: Option<DynamicNeb>) {
        self.dynamic_run().install(spec);
    }

    fn alloc_metric_id(&self) -> MetricBuildId {
        MetricBuildId(self.next_metric_id.fetch_add(1, Ordering::Relaxed))
    }
    /// One surface call, split into image blocks on the image threads.
    fn eval_rows(
        &self,
        surface: &dyn BandSurface,
        range: std::ops::Range<usize>,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        if range.end <= self.ids.len() && range.len() == positions.nrows() {
            let ids = &self.ids[range];
            return match &self.pool {
                Some(pool) => pool.eval_with_ids(surface, ids, positions, energies, gradients),
                None => surface.eval_with_ids(ids, positions, energies, gradients),
            };
        }
        match &self.pool {
            Some(pool) => pool.eval(surface, positions, energies, gradients),
            None => surface.eval(positions, energies, gradients),
        }
    }
    fn convergence_force(&self, force: &Array1<f64>) -> f64 {
        match self.force_gate {
            ForceGate::LinfNorm => max_abs(force),
            gate => gate.value(force.view()),
        }
    }
    fn atom_baseline(&self) -> Option<f64> {
        *self.atom_baseline.lock().unwrap_or_else(|e| e.into_inner())
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
        *self.previous.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }

    /// Make the start of a step the cached answer again when the
    /// stepper left the band there: the trial went into `last`, the
    /// start sits in `previous`. Returns whether `last` now holds `x`.
    fn restore_at(&self, x: &Array1<f64>) -> bool {
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        if last.as_ref().is_some_and(|l| l.x == *x) {
            return true;
        }
        let mut previous = self.previous.lock().unwrap_or_else(|e| e.into_inner());
        if previous.as_ref().is_some_and(|p| p.x == *x) {
            let start = previous.take();
            *previous = last.take();
            *last = start;
            if let Some(eval) = last.as_ref() {
                *self
                    .last_max_force
                    .lock()
                    .unwrap_or_else(|e| e.into_inner()) = self.convergence_force(&eval.projected);
            }
            return true;
        }
        false
    }
    /// Make `start` the cached answer again when it was taken at `x`
    /// and neither cache slot holds `x`.
    fn reinstate(&self, start: Option<BandEval>, x: &Array1<f64>) {
        let Some(start) = start.filter(|s| s.x == *x) else {
            return;
        };
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        let mut previous = self.previous.lock().unwrap_or_else(|e| e.into_inner());
        let held = |slot: &Option<BandEval>| slot.as_ref().is_some_and(|e| e.x == *x);
        if held(&last) || held(&previous) {
            return;
        }
        *previous = last.take();
        *last = Some(start);
    }

    fn invalidate_assembly(&self) {
        for slot in [&self.last, &self.previous] {
            let mut guard = match slot.lock() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
            if let Some(eval) = guard.as_mut() {
                eval.projected = Array1::zeros(0);
                eval.driving = Array1::zeros(0);
                eval.assembly_ci = i64::MIN;
                eval.assembly_metric = None;
            }
        }
    }

    /// Forget which climbing image an assembly belongs to.
    ///
    /// The projected force and its metric stamp stay, so
    /// [`BandSession::evaluation`] still reports the force the last
    /// step used. The next assembly misses this key, recaptures the
    /// baseline, and can arm again. Dropping the force here made
    /// `evaluation` report nothing after [`BandSession::restart`].
    fn retire_assembly_key(&self) {
        for slot in [&self.last, &self.previous] {
            let mut guard = match slot.lock() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
            if let Some(eval) = guard.as_mut() {
                eval.assembly_ci = i64::MIN;
            }
        }
    }

    /// The climbing baseline and the armed image belong to one
    /// relaxation; the surface caches and the last projected force stay.
    fn restart(&self) {
        self.retire_assembly_key();
        *self.baseline.lock().unwrap_or_else(|e| e.into_inner()) = None;
        *self.atom_baseline.lock().unwrap_or_else(|e| e.into_inner()) = None;
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

/// Buffers of one interior image's force. Each interior image owns
/// one, so the images can be assembled apart from one another.
#[derive(Default)]
struct ImageWork {
    diff_next: Array1<f64>,
    diff_prev: Array1<f64>,
    tangent: Array1<f64>,
    force: Array1<f64>,
    parallel: Array1<f64>,
    full: Array1<f64>,
    perp: Array1<f64>,
    dneb: Array1<f64>,
    scratch: Array1<f64>,
    l_next: Array1<f64>,
    l_here: Array1<f64>,
}

impl ImageWork {
    fn sized(dof: usize) -> Self {
        let zero = || Array1::zeros(dof);
        Self {
            diff_next: zero(),
            diff_prev: zero(),
            tangent: zero(),
            force: zero(),
            parallel: zero(),
            full: zero(),
            perp: zero(),
            dneb: zero(),
            scratch: zero(),
            l_next: zero(),
            l_here: zero(),
        }
    }
}

/// Storage that one band evaluation and assembly hand to the next.
///
/// `spare` is the evaluation that the cache dropped last, kept as the
/// storage of the next one, and `spare_projected` the projected force
/// buffer of a recycled evaluation. Both refill before they are read.
#[derive(Default)]
struct BandWork {
    images: Vec<ImageWork>,
    interior_e: Array1<f64>,
    interior_g: Array2<f64>,
    whole: Array2<f64>,
    ks: Vec<f64>,
    spare: Option<BandEval>,
    spare_projected: Array1<f64>,
    spare_driving: Array1<f64>,
}

/// One [`ImageWork`] of length `dof` per interior image.
fn fit_images(images: &mut Vec<ImageWork>, n_interior: usize, dof: usize) -> &mut [ImageWork] {
    if images.first().is_some_and(|w| w.tangent.len() != dof) {
        images.clear();
    }
    images.resize_with(n_interior, || ImageWork::sized(dof));
    images
}

impl BandWork {
    /// Storage for a new atomic evaluation of `n_images` rows: the
    /// recycled `spare` when it has that shape, a fresh one otherwise.
    /// Its forces are cleared, the mark of an unassembled point.
    fn take_eval(&mut self, n_images: usize, dof: usize) -> BandEval {
        let fits = |e: &BandEval| {
            e.x.len() == (n_images - 2) * dof && e.gradients.dim() == (n_images, dof)
        };
        match self.spare.take().filter(fits) {
            Some(mut eval) => {
                let projected = std::mem::replace(&mut eval.projected, Array1::zeros(0));
                if projected.len() > self.spare_projected.len() {
                    self.spare_projected = projected;
                }
                let driving = std::mem::replace(&mut eval.driving, Array1::zeros(0));
                if driving.len() > self.spare_driving.len() {
                    self.spare_driving = driving;
                }
                eval.assembly_ci = i64::MIN;
                eval.assembly_metric = None;
                eval
            }
            None => BandEval {
                x: Array1::zeros((n_images - 2) * dof),
                energies: Array1::zeros(n_images),
                gradients: Array2::zeros((n_images, dof)),
                projected: Array1::zeros(0),
                driving: Array1::zeros(0),
                assembly_ci: i64::MIN,
                assembly_metric: None,
                stresses: Array2::zeros((0, 9)),
                band_positions: Array2::zeros((0, dof)),
                band_cells: Vec::new(),
            },
        }
    }

    /// Store an assembled force in `target`, using `spare` when needed.
    fn store_force(target: &mut Array1<f64>, spare: &mut Array1<f64>, force: &Array1<f64>) {
        if target.len() != force.len() {
            std::mem::swap(target, spare);
        }
        if target.len() == force.len() {
            target.assign(force);
        } else {
            *target = force.clone();
        }
    }

    /// Store the residual and optional preconditioned driving force.
    fn store_forces(
        &mut self,
        eval: &mut BandEval,
        projected: &Array1<f64>,
        driving: Option<&Array1<f64>>,
    ) {
        Self::store_force(&mut eval.projected, &mut self.spare_projected, projected);
        match driving {
            Some(force) => {
                Self::store_force(&mut eval.driving, &mut self.spare_driving, force);
            }
            None => {
                // This residual was assembled without the metric. Keeping
                // a driving force beside it would let the next metric
                // read return this residual as if it were the
                // preconditioned one.
                if eval.driving.len() > self.spare_driving.len() {
                    std::mem::swap(&mut eval.driving, &mut self.spare_driving);
                }
                eval.driving = Array1::zeros(0);
            }
        }
    }
}

fn fit1(buffer: &mut Array1<f64>, len: usize) {
    if buffer.len() != len {
        *buffer = Array1::zeros(len);
    }
    buffer.fill(0.0);
}

fn fit2(buffer: &mut Array2<f64>, dim: (usize, usize)) {
    if buffer.dim() != dim {
        *buffer = Array2::zeros(dim);
    }
    buffer.fill(0.0);
}

/// Make `state.last` the surface's answer at interior `x` and return
/// it locked: kept when `x` matches it bitwise, swapped back from
/// `state.previous` when `x` matches that (a stepper that went back to
/// the point before its trial), else one surface call (interior rows
/// only once the endpoint energies are cached). The
/// surface receives zeroed energies and interior gradients, and on a
/// whole-band call the gradients of the evaluation it replaces, as
/// freshly built arrays would hold. The evaluation the cache drops
/// becomes the storage of the next one.
fn evaluate_band<'s>(
    state: &'s BandState,
    work: &mut BandWork,
    surface: &dyn BandSurface,
    endpoint_first: ArrayView1<f64>,
    endpoint_last: ArrayView1<f64>,
    x: ArrayView1<f64>,
) -> Result<std::sync::MutexGuard<'s, Option<BandEval>>, SaddleError> {
    let dof = endpoint_first.len();
    let n_images = x.len() / dof + 2;
    let mut slot = state.last.lock().unwrap_or_else(|e| e.into_inner());
    if slot.as_ref().is_some_and(|prev| prev.x == x) {
        return Ok(slot);
    }
    {
        let mut previous = state.previous.lock().unwrap_or_else(|e| e.into_inner());
        if previous.as_ref().is_some_and(|prev| prev.x == x) {
            std::mem::swap(&mut *slot, &mut *previous);
            return Ok(slot);
        }
    }
    let interior = x
        .to_shape((n_images - 2, dof))
        .map_err(|e| SaddleError::Shape(e.to_string()))?;
    let mut next = work.take_eval(n_images, dof);
    let answered = fill_band(
        state,
        work,
        surface,
        slot.as_ref(),
        &mut next,
        endpoint_first,
        endpoint_last,
        interior.view(),
    );
    if let Err(err) = answered {
        work.spare = Some(next);
        return Err(err);
    }
    next.x.assign(&x);
    if state.endpoints().is_none() {
        state.set_endpoints(Some([next.energies[0], next.energies[n_images - 1]]));
    }
    if let Some(old) = slot.take() {
        work.spare = state
            .previous
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .replace(old);
    }
    *slot = Some(next);
    Ok(slot)
}

/// One surface call into `next`, checked for finite values. `prev` is
/// the evaluation `next` replaces: its endpoint gradient rows stay
/// known when the call carries the interior alone.
#[allow(clippy::too_many_arguments)]
fn fill_band(
    state: &BandState,
    work: &mut BandWork,
    surface: &dyn BandSurface,
    prev: Option<&BandEval>,
    next: &mut BandEval,
    endpoint_first: ArrayView1<f64>,
    endpoint_last: ArrayView1<f64>,
    interior: ArrayView2<f64>,
) -> Result<(), SaddleError> {
    let n_images = next.energies.len();
    let dof = endpoint_first.len();
    let last = n_images - 1;
    next.energies.fill(0.0);
    match state.endpoints() {
        Some([e_first, e_last]) => {
            next.energies[0] = e_first;
            next.energies[last] = e_last;
            for end in [0, last] {
                match prev {
                    Some(prev) => next.gradients.row_mut(end).assign(&prev.gradients.row(end)),
                    None => next.gradients.row_mut(end).fill(0.0),
                }
            }
            if state.dynamic_enabled() {
                fill_dynamic_interior(state, work, surface, next, interior)?;
            } else {
                fit1(&mut work.interior_e, n_images - 2);
                fit2(&mut work.interior_g, (n_images - 2, dof));
                state.eval_rows(
                    surface,
                    1..last,
                    interior,
                    &mut work.interior_e,
                    &mut work.interior_g,
                )?;
                state.rows.fetch_add(n_images - 2, Ordering::Relaxed);
                next.energies
                    .slice_mut(s![1..last])
                    .assign(&work.interior_e);
                next.gradients
                    .slice_mut(s![1..last, ..])
                    .assign(&work.interior_g);
            }
        }
        None => {
            fit2(&mut work.whole, (n_images, dof));
            work.whole.row_mut(0).assign(&endpoint_first);
            work.whole.row_mut(last).assign(&endpoint_last);
            work.whole.slice_mut(s![1..last, ..]).assign(&interior);
            match prev {
                Some(prev) => next.gradients.assign(&prev.gradients),
                None => next.gradients.fill(0.0),
            }
            state.eval_rows(
                surface,
                0..n_images,
                work.whole.view(),
                &mut next.energies,
                &mut next.gradients,
            )?;
            state.rows.fetch_add(n_images, Ordering::Relaxed);
        }
    }
    if !next.energies.iter().all(|e| e.is_finite()) {
        return Err(SaddleError::NonFinite("band energies"));
    }
    if !next
        .gradients
        .slice(s![1..last, ..])
        .iter()
        .all(|g| g.is_finite())
    {
        return Err(SaddleError::NonFinite("band gradients"));
    }
    if !state.fixed.is_empty() {
        for mut row in next.gradients.outer_iter_mut() {
            zero_fixed(&state.fixed, &mut row);
        }
    }
    if state.dynamic_enabled() {
        remember_dynamic(state, interior, next);
    }
    Ok(())
}

/// Interior rows for dynamic NEB. An image whose coordinates still
/// equal its last answer reuses that energy and gradient. A full miss
/// stays one batched call. A partial miss asks for those images alone
/// through [`BandSurface::eval_indexed`].
fn fill_dynamic_interior(
    state: &BandState,
    work: &mut BandWork,
    surface: &dyn BandSurface,
    next: &mut BandEval,
    interior: ArrayView2<f64>,
) -> Result<(), SaddleError> {
    let n_interior = interior.nrows();
    let dof = interior.ncols();
    let mut reuse = Vec::with_capacity(n_interior);
    {
        let run = state.dynamic_run();
        for i in 0..n_interior {
            reuse.push(run.recall(i, interior.row(i)));
        }
    }
    let fresh: Vec<usize> = reuse
        .iter()
        .enumerate()
        .filter(|(_, hit)| hit.is_none())
        .map(|(i, _)| i)
        .collect();
    if fresh.len() == n_interior {
        fit1(&mut work.interior_e, n_interior);
        fit2(&mut work.interior_g, (n_interior, dof));
        state.eval_rows(
            surface,
            1..(n_interior + 1),
            interior,
            &mut work.interior_e,
            &mut work.interior_g,
        )?;
        state.rows.fetch_add(n_interior, Ordering::Relaxed);
        let last = n_interior + 1;
        next.energies
            .slice_mut(s![1..last])
            .assign(&work.interior_e);
        next.gradients
            .slice_mut(s![1..last, ..])
            .assign(&work.interior_g);
        return Ok(());
    }
    for (i, hit) in reuse.iter().enumerate() {
        if let Some((energy, gradient)) = hit {
            next.energies[i + 1] = *energy;
            next.gradients.row_mut(i + 1).assign(gradient);
        }
    }
    for i in fresh {
        let (energy, mut gradient) = surface.eval_indexed(i + 1, interior.row(i))?;
        if !energy.is_finite() || !gradient.iter().all(|component| component.is_finite()) {
            return Err(SaddleError::NonFinite("band gradients"));
        }
        zero_fixed(&state.fixed, &mut gradient);
        next.energies[i + 1] = energy;
        next.gradients.row_mut(i + 1).assign(&gradient);
        state.rows.fetch_add(1, Ordering::Relaxed);
    }
    Ok(())
}

/// Keep the surface answer of every interior image for the next dynamic step.
fn remember_dynamic(state: &BandState, interior: ArrayView2<f64>, next: &BandEval) {
    let mut run = state.dynamic_run();
    let n_interior = interior.nrows();
    if run.images.len() != n_interior {
        run.images = (0..n_interior).map(|_| None).collect();
    }
    for i in 0..n_interior {
        run.remember(
            i,
            interior.row(i),
            next.energies[i + 1],
            next.gradients.row(i + 1),
        );
    }
}

/// Remove the mean Cartesian force (eOn's `zeroTranslation`). An image
/// with one atom has no internal force apart from the mean, so it
/// keeps its force.
pub(crate) fn remove_net_translation<S: DataMut<Elem = f64>>(force: &mut ArrayBase<S, Ix1>) {
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
    if matches!(
        config.spring,
        SpringKind::OnsagerMachlup { .. } | SpringKind::OnsagerMachlupLive { .. }
    ) {
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
    if let Some(old) = slot.take() {
        *state.previous.lock().unwrap_or_else(|e| e.into_inner()) = Some(old);
    }
    *slot = Some(BandEval {
        x: x.to_owned(),
        energies: energies.clone(),
        gradients: gradients.clone(),
        projected: Array1::zeros(0),
        driving: Array1::zeros(0),
        assembly_ci: i64::MIN,
        assembly_metric: None,
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
    spring: &SpringKind,
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
    let spring = spring.compute(
        i,
        tangent.view(),
        dist_next,
        dist_prev,
        diff_next.view(),
        diff_prev.view(),
    )?;
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
    let climbing = config.climbing.is_some();
    let armed = climbing && state.armed.load(Ordering::Relaxed);
    if armed {
        state.ci_index.store(max_i as i64, Ordering::Relaxed);
    }
    let ci_at = armed.then_some(max_i);
    let spring = config.spring.at_energies(enthalpy.view());
    let mut joint = Array1::zeros(x.len());
    let mut cartesian = Array1::zeros(x.len());
    for i in 1..n_images - 1 {
        let (projected, step) = solid_image(
            config,
            &spring,
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
        state.ci_index.store(max_i as i64, Ordering::Relaxed);
        let (projected, step) = solid_image(
            config, &spring, max_i, true, &positions, &cells, &enthalpy, &gradients, &stresses,
            jacobian, pressure,
        )?;
        let start = (max_i - 1) * seg;
        joint.slice_mut(s![start..start + seg]).assign(&projected);
        cartesian.slice_mut(s![start..start + seg]).assign(&step);
        max_component = state.convergence_force(&joint);
    }
    // A force whose squared norm overflows defeats every solver's
    // dot products long before a component turns non-finite.
    if !joint.dot(&joint).is_finite() {
        return Err(SaddleError::NonFinite("band force norm"));
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

/// Projected force on interior image `i`, climbing when `climb`, and
/// with a metric the paper's preconditioned driving force beside it.
#[allow(clippy::too_many_arguments)]
fn image_force(
    config: &BandConfig,
    state: &BandState,
    spring_kind: &SpringKind,
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
    zero_fixed(&config.fixed_atoms, &mut pos_diff_next);
    zero_fixed(&config.fixed_atoms, &mut pos_diff_prev);
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
    let spring = spring_kind.compute(
        i,
        tangent.view(),
        dist_next,
        dist_prev,
        pos_diff_next.view(),
        pos_diff_prev.view(),
    )?;
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
    zero_fixed(&config.fixed_atoms, &mut projected);
    if config.drops_translation() {
        remove_net_translation(&mut projected);
    }
    let driving = match metric {
        Some((p, free)) => {
            let curvature = &pos_diff_next - &pos_diff_prev;
            let spring_k = match config.spring {
                SpringKind::Uniform { k } => Some(k),
                _ => None,
            };
            let mut driving;
            match free {
                None => {
                    (driving, projected) = metric_force(
                        state,
                        p,
                        gradients.row(i),
                        tangent.view(),
                        curvature.view(),
                        climb,
                        spring_k,
                    )?;
                }
                Some(free) => {
                    // P acts on the free block; the fixed coordinates
                    // carry no force in either vector.
                    let gather = |v: ArrayView1<f64>| -> Array1<f64> {
                        free.iter().map(|&c| v[c]).collect()
                    };
                    let (driving_free, projected_free) = metric_force(
                        state,
                        p,
                        gather(gradients.row(i)).view(),
                        gather(tangent.view()).view(),
                        gather(curvature.view()).view(),
                        climb,
                        spring_k,
                    )?;
                    driving = Array1::zeros(dof);
                    projected = Array1::zeros(dof);
                    for (k, &c) in free.iter().enumerate() {
                        driving[c] = driving_free[k];
                        projected[c] = projected_free[k];
                    }
                }
            }
            if config.drops_translation() {
                remove_net_translation(&mut driving);
                remove_net_translation(&mut projected);
            }
            if !driving.dot(&driving).is_finite() {
                return Err(SaddleError::NonFinite("preconditioned band force norm"));
            }
            Some(driving)
        }
        None => None,
    };
    Ok((projected, driving))
}

/// The preconditioned driving force of one image and the residual the
/// band measures, on the coordinates `p` acts on. `curvature` is
/// `d_next - d_prev`; `spring_k` is the uniform spring, if any.
fn metric_force(
    state: &BandState,
    p: &Exp,
    gradient: ArrayView1<f64>,
    tangent: ArrayView1<f64>,
    curvature: ArrayView1<f64>,
    climb: bool,
    spring_k: Option<f64>,
) -> Result<(Array1<f64>, Array1<f64>), SaddleError> {
    // Makri, Ortner and Kermode, Eq. (9) and (10): the gradient
    // through P^{-1}, the tangent normalised in the P-norm, and
    // the spring along that tangent with the P inner product.
    state.pcg_solves.fetch_add(1, Ordering::Relaxed);
    let s = p.solve(gradient).ok_or(SaddleError::Invalid(
        "exponential metric: the linear solve failed".into(),
    ))?;
    let pt = p.apply(tangent);
    let norm_p = tangent.dot(&pt).sqrt();
    if !(norm_p.is_finite() && norm_p > 0.0) {
        return Err(SaddleError::NonFinite("P-norm of the tangent"));
    }
    let t_p = &tangent / norm_p;
    let pt_p = &pt / norm_p;
    let along = t_p.dot(&gradient);
    let mut driving;
    let projected;
    if climb {
        // The climbing image reflects the preconditioned gradient
        // in the P-metric, -(P^{-1} g - 2 t_P (t_P . g)), whose
        // fixed point is g = 0; its residual is the P-image of
        // that force, -(g - 2 P t_P (t_P . g)).
        driving = -&s + &(&t_p * (2.0 * along));
        projected = -&gradient + &(&pt_p * (2.0 * along));
    } else {
        driving = -(&s - &(&t_p * along));
        if let Some(k) = spring_k {
            // The paper's spring constant is dimensionless, the
            // stiffness relative to the metric; `k` in
            // force-constant units keeps its meaning through the
            // metric's energy scale, kappa = k / mu.
            driving += &(&t_p * (k / p.mu() * curvature.dot(&pt_p)));
        }
        // The fixed point of the preconditioned projection is the
        // path with the gradient along P t_P, the minimum energy
        // path of the metric's geometry, where the Euclidean
        // perpendicular force does not vanish. The band force that
        // convergence and the report measure is the residual of
        // Eq. (11), -P grad_perp V_P = -(g - P t_P (t_P . g)),
        // zero on that path.
        projected = -&gradient + &(&pt_p * along);
    }
    Ok((driving, projected))
}

/// The springs of one band evaluation. A kind whose constants or
/// displacements follow the band resolves without building a new
/// [`SpringKind`]: the energy-weighted constants land in a reused
/// buffer, and the live Onsager-Machlup displacements are formed per
/// image from the gradients.
enum BandSpring<'a> {
    Kind(&'a SpringKind),
    Weighted(&'a [f64]),
    Live { k: f64, alpha: f64 },
}

impl<'a> BandSpring<'a> {
    fn at(kind: &'a SpringKind, energies: ArrayView1<f64>, ks: &'a mut Vec<f64>) -> Self {
        match kind {
            SpringKind::OnsagerMachlupLive { k } => BandSpring::Live {
                k: *k,
                alpha: crate::spring::onsager_machlup_scale(*k),
            },
            SpringKind::EnergyWeighted { k_min, k_max } => {
                crate::spring::energy_weighted_constants_into(energies, *k_min, *k_max, ks);
                BandSpring::Weighted(ks)
            }
            other => BandSpring::Kind(other),
        }
    }
}

/// Spring forces at interior image `i` into `parallel` and `full`, the
/// arithmetic of [`SpringKind::compute`] on the resolved springs.
#[allow(clippy::too_many_arguments)]
fn spring_into(
    spring: &BandSpring,
    i: usize,
    gradients: &Array2<f64>,
    dist_next: f64,
    dist_prev: f64,
    tangent: ArrayView1<f64>,
    diff_next: ArrayView1<f64>,
    diff_prev: ArrayView1<f64>,
    l_next: &mut Array1<f64>,
    l_here: &mut Array1<f64>,
    parallel: ArrayViewMut1<f64>,
    full: ArrayViewMut1<f64>,
) {
    use crate::stencil::Coupling;
    let weighted = |ks: &[f64]| Coupling::Weighted {
        k_next: ks[i],
        k_prev: ks[i - 1],
    };
    let coupling = match spring {
        BandSpring::Weighted(ks) => weighted(ks),
        BandSpring::Live { k, alpha } => {
            let n_images = gradients.nrows();
            if i + 1 == n_images - 1 {
                l_next.fill(0.0);
            } else {
                Zip::from(&mut *l_next)
                    .and(gradients.row(i + 1))
                    .for_each(|l, &g| *l = alpha * -g);
            }
            Zip::from(&mut *l_here)
                .and(gradients.row(i))
                .for_each(|l, &g| *l = alpha * -g);
            Coupling::Onsager {
                k: *k,
                l_next: l_next.view(),
                l_here: l_here.view(),
            }
        }
        BandSpring::Kind(SpringKind::Uniform { k }) => Coupling::Uniform { k: *k },
        BandSpring::Kind(SpringKind::Weighted { ks }) => weighted(ks),
        BandSpring::Kind(SpringKind::OnsagerMachlup { k, l_vecs }) => Coupling::Onsager {
            k: *k,
            l_next: l_vecs[i + 1].view(),
            l_here: l_vecs[i].view(),
        },
        BandSpring::Kind(SpringKind::OnsagerMachlupLive { k }) => {
            l_next.fill(0.0);
            l_here.fill(0.0);
            Coupling::Onsager {
                k: *k,
                l_next: l_next.view(),
                l_here: l_here.view(),
            }
        }
        BandSpring::Kind(SpringKind::EnergyWeighted { k_min, .. }) => {
            Coupling::Uniform { k: *k_min }
        }
    };
    crate::stencil::coupling_forces_into(
        &coupling, tangent, dist_next, dist_prev, diff_next, diff_prev, parallel, full,
    );
}

/// [`image_force`] without a metric, written into `out` through the
/// image's own buffers.
#[allow(clippy::too_many_arguments)]
fn image_force_into(
    config: &BandConfig,
    spring: &BandSpring,
    i: usize,
    climb: bool,
    endpoint_first: ArrayView1<f64>,
    endpoint_last: ArrayView1<f64>,
    x: ArrayView1<f64>,
    energies: &Array1<f64>,
    gradients: &Array2<f64>,
    w: &mut ImageWork,
    mut out: ArrayViewMut1<f64>,
) -> Result<(), SaddleError> {
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
    let ImageWork {
        diff_next,
        diff_prev,
        tangent,
        force,
        parallel,
        full,
        perp,
        dneb,
        scratch,
        l_next,
        l_here,
    } = w;
    Zip::from(&mut *diff_next)
        .and(pos_next)
        .and(pos)
        .for_each(|d, &a, &b| *d = a - b);
    Zip::from(&mut *diff_prev)
        .and(pos)
        .and(pos_prev)
        .for_each(|d, &a, &b| *d = a - b);
    if let Some(cell) = &config.cell {
        cell.minimum_image_view(diff_next.view_mut());
        cell.minimum_image_view(diff_prev.view_mut());
    }
    zero_fixed(&config.fixed_atoms, diff_next);
    zero_fixed(&config.fixed_atoms, diff_prev);
    let dist_next = diff_next.dot(&*diff_next).sqrt();
    let dist_prev = diff_prev.dot(&*diff_prev).sqrt();
    let rule = match config.tangent {
        TangentKind::Simple => crate::stencil::TangentRule::Simple,
        TangentKind::Improved => crate::stencil::TangentRule::Improved,
    };
    crate::stencil::unit_tangent_into(
        rule,
        diff_next.view(),
        diff_prev.view(),
        energies[i],
        energies[i - 1],
        energies[i + 1],
        tangent.view_mut(),
    )
    .map_err(|e| SaddleError::Invalid(format!("image {i}: {e}")))?;
    Zip::from(&mut *force)
        .and(gradients.row(i))
        .for_each(|f, &g| *f = -g);
    spring_into(
        spring,
        i,
        gradients,
        dist_next,
        dist_prev,
        tangent.view(),
        diff_next.view(),
        diff_prev.view(),
        l_next,
        l_here,
        parallel.view_mut(),
        full.view_mut(),
    );
    let doubly = config.projection == ProjectionKind::DoublyNudged;
    if doubly {
        crate::stencil::perp_into(force.view(), tangent.view(), perp.view_mut());
        crate::projection::dneb_component_into(
            full.view(),
            tangent.view(),
            perp.view(),
            dneb.view_mut(),
            scratch.view_mut(),
        );
    }
    if climb {
        if !doubly {
            dneb.fill(0.0);
        }
        crate::stencil::reflect_into(force.view(), tangent.view(), out.view_mut());
        Zip::from(&mut out).and(&*dneb).for_each(|o, &d| *o += d);
    } else {
        match config.projection {
            ProjectionKind::PlainElasticBand => Zip::from(&mut out)
                .and(&*full)
                .and(&*force)
                .for_each(|o, &s, &f| *o = s + f),
            ProjectionKind::Neb => {
                crate::stencil::perp_into(force.view(), tangent.view(), perp.view_mut());
                Zip::from(&mut out)
                    .and(&*parallel)
                    .and(&*perp)
                    .for_each(|o, &s, &p| *o = s + p);
            }
            ProjectionKind::DoublyNudged => Zip::from(&mut out)
                .and(&*parallel)
                .and(&*perp)
                .and(&*dneb)
                .for_each(|o, &s, &p, &d| *o = s + p + d),
        }
    }
    zero_fixed(&config.fixed_atoms, &mut out);
    if config.drops_translation() {
        remove_net_translation(&mut out);
    }
    Ok(())
}

fn max_abs(v: &Array1<f64>) -> f64 {
    v.iter().fold(0.0_f64, |m, c| m.max(c.abs()))
}

/// The interior image that climbs once the band is armed: the highest
/// one, unless an endpoint at or above it slopes down into the band.
///
/// Two minima bound a peak, so the saddle lies between a highest
/// interior image and a higher endpoint that is a minimum. An endpoint
/// that is not one can sit at the top of a slope the band rises along,
/// with no saddle in between; climbing there only drives the image up
/// the slope. Such an endpoint blocks the climb when its force along
/// the band, toward the next image, exceeds `force_tol`. An endpoint
/// whose gradient was never evaluated has a zero row and blocks nothing.
#[allow(clippy::too_many_arguments)]
/// Interior image that climbs on this evaluation, or `None` when an
/// endpoint at or above that image slopes down into the band.
///
/// `positions`, `energies` and `gradients` cover the whole band,
/// endpoints included. A caller that always climbs the highest
/// interior image, the way ASE's `climb=True` does on every force
/// call, drives that image up a slope that has no saddle between it
/// and the higher endpoint. This selection is the one
/// [`BandSession`] uses once climbing is armed.
pub fn climbing_image(
    config: &BandConfig,
    positions: ArrayView2<f64>,
    energies: ArrayView1<f64>,
    gradients: ArrayView2<f64>,
) -> Result<Option<usize>, SaddleError> {
    let n_images = positions.nrows();
    let dof = positions.ncols();
    if n_images < 3 || dof == 0 || !dof.is_multiple_of(3) {
        return Err(SaddleError::Shape(format!(
            "a band needs at least 3 images of 3N coordinates; got {n_images} x {dof}"
        )));
    }
    if energies.len() != n_images || gradients.dim() != (n_images, dof) {
        return Err(SaddleError::Shape(
            "energies and gradients must match the band".into(),
        ));
    }
    check_fixed_len(config, dof)?;
    let mut max_i = 1usize;
    for i in 2..n_images - 1 {
        if energies[i] > energies[max_i] {
            max_i = i;
        }
    }
    let mut flat = Array1::zeros((n_images - 2) * dof);
    for i in 1..n_images - 1 {
        flat.slice_mut(s![(i - 1) * dof..i * dof])
            .assign(&positions.row(i));
    }
    Ok(climb_target(
        config,
        max_i,
        positions.row(0),
        positions.row(n_images - 1),
        flat.view(),
        energies.view(),
        gradients.view(),
    ))
}

fn climb_target(
    config: &BandConfig,
    max_i: usize,
    endpoint_first: ArrayView1<f64>,
    endpoint_last: ArrayView1<f64>,
    x: ArrayView1<f64>,
    energies: ArrayView1<f64>,
    gradients: ArrayView2<f64>,
) -> Option<usize> {
    let dof = endpoint_first.len();
    let last = energies.len() - 1;
    let slopes_into_band = |end: usize| {
        let (pos, next) = if end == 0 {
            (endpoint_first, x.slice(s![..dof]))
        } else {
            (endpoint_last, x.slice(s![x.len() - dof..]))
        };
        let mut along = &next - &pos;
        if let Some(cell) = &config.cell {
            cell.minimum_image(&mut along);
        }
        zero_fixed(&config.fixed_atoms, &mut along);
        let length = along.dot(&along).sqrt();
        length > 0.0 && -gradients.row(end).dot(&along) / length > config.force_tol
    };
    let blocked = [0, last]
        .into_iter()
        .any(|end| energies[end] >= energies[max_i] && slopes_into_band(end));
    (!blocked).then_some(max_i)
}

/// Convergence force of each interior row of an unmasked projected force.
fn row_convergence(
    state: &BandState,
    projected: &Array1<f64>,
    n_interior: usize,
    dof: usize,
) -> Vec<f64> {
    (0..n_interior)
        .map(|i| {
            let row = projected.slice(s![i * dof..(i + 1) * dof]).to_owned();
            state.convergence_force(&row)
        })
        .collect()
}

/// Record the dynamic mask at `origin` from the evaluation already
/// stored there. A solver that cached this point does not ask for it
/// again, and the trial it does ask for must not be what pins the images.
fn seed_dynamic_decision(state: &BandState, origin: ArrayView1<f64>, force_tol: f64) {
    let Some(spec) = state.dynamic_spec() else {
        return;
    };
    let Some((projected, energies, n_interior, dof)) = cached_dynamic_inputs(state, origin) else {
        return;
    };
    let mut rows = Array2::zeros((n_interior, dof));
    for i in 0..n_interior {
        rows.row_mut(i)
            .assign(&projected.slice(s![i * dof..(i + 1) * dof]));
    }
    let Ok(interior) = origin.to_shape((n_interior, dof)) else {
        return;
    };
    let Ok(mask) = dynamic_mask(
        interior.view(),
        energies.slice(s![1..n_interior + 1]),
        &mut rows,
        spec,
    ) else {
        return;
    };
    let mut masked = Array1::zeros(origin.len());
    for i in 0..n_interior {
        masked
            .slice_mut(s![i * dof..(i + 1) * dof])
            .assign(&rows.row(i));
    }
    let visible = state.convergence_force(&masked);
    let row_gate = row_convergence(state, &projected, n_interior, dof);
    let _zeroed = state
        .dynamic_run()
        .decision(origin, &mask, visible, &row_gate, force_tol);
}

fn cached_dynamic_inputs(
    state: &BandState,
    origin: ArrayView1<f64>,
) -> Option<(Array1<f64>, Array1<f64>, usize, usize)> {
    let slot = state.last.lock().unwrap_or_else(|e| e.into_inner());
    let last = slot.as_ref()?;
    if last.x != origin {
        return None;
    }
    let n_images = last.energies.len();
    if n_images < 3 {
        return None;
    }
    let n_interior = n_images - 2;
    if n_interior == 0 || !origin.len().is_multiple_of(n_interior) {
        return None;
    }
    let dof = origin.len() / n_interior;
    if last.projected.len() != origin.len() || dof == 0 || !dof.is_multiple_of(3) {
        return None;
    }
    Some((
        last.projected.clone(),
        last.energies.clone(),
        n_interior,
        dof,
    ))
}

/// Assemble the projected band force at interior `x`. Returns
/// (pseudo-energy, projected interior forces, max abs component).
///
/// The climbing image is decided on this evaluation's energies, so
/// the force returned already carries the climb: once armed, the
/// highest interior image climbs, also when an endpoint is higher,
/// unless that endpoint slopes down into the band ([`climb_target`]);
/// on the evaluation that arms it, the climbing image's force is
/// reassembled before returning.
fn assemble_band(
    config: &BandConfig,
    state: &BandState,
    surface: &dyn BandSurface,
    endpoint_first: ArrayView1<f64>,
    endpoint_last: ArrayView1<f64>,
    x: ArrayView1<f64>,
) -> Result<(f64, Array1<f64>, f64), SaddleError> {
    let (energy, projected, max_component, _) = assemble_band_with(
        config,
        state,
        surface,
        endpoint_first,
        endpoint_last,
        x,
        None,
    )?;
    Ok((energy, projected, max_component))
}

#[allow(clippy::too_many_arguments)]
fn assemble_image_force(
    config: &BandConfig,
    state: &BandState,
    springs: &Springs<'_>,
    metric: Option<(&Exp, Option<&[usize]>)>,
    i: usize,
    climb: bool,
    endpoint_first: ArrayView1<f64>,
    endpoint_last: ArrayView1<f64>,
    x: ArrayView1<f64>,
    energies: &Array1<f64>,
    gradients: &Array2<f64>,
    work: &mut ImageWork,
    mut projected: ArrayViewMut1<f64>,
    mut driving: Option<ArrayViewMut1<f64>>,
) -> Result<(), SaddleError> {
    match springs {
        Springs::Metric(spring) => {
            let (force, step) = image_force(
                config,
                state,
                spring,
                i,
                climb,
                endpoint_first,
                endpoint_last,
                x,
                energies,
                gradients,
                metric,
            )?;
            projected.assign(&force);
            if let Some(out) = driving.as_mut() {
                match step.as_ref() {
                    Some(step) => out.assign(step),
                    None => out.assign(&force),
                }
            }
        }
        Springs::Band(spring) => image_force_into(
            config,
            spring,
            i,
            climb,
            endpoint_first,
            endpoint_last,
            x,
            energies,
            gradients,
            work,
            projected,
        )?,
    }
    Ok(())
}

/// The band assembly, and with `metrics` also the preconditioned driving
/// force the solver integrates in place of the projected one.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn assemble_band_with(
    config: &BandConfig,
    state: &BandState,
    surface: &dyn BandSurface,
    endpoint_first: ArrayView1<f64>,
    endpoint_last: ArrayView1<f64>,
    x: ArrayView1<f64>,
    metrics: Option<BandMetrics<'_>>,
) -> Result<(f64, Array1<f64>, f64, Option<Array1<f64>>), SaddleError> {
    let mut work = state.work.lock().unwrap_or_else(|e| e.into_inner());
    let mut slot = evaluate_band(state, &mut work, surface, endpoint_first, endpoint_last, x)?;
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
    let climb_target = climb_target(
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
        state
            .ci_index
            .store(climb_target.map_or(-1, |i| i as i64), Ordering::Relaxed);
    }
    let ci_at = if armed { climb_target } else { None };
    let pseudo_energy: f64 = energies.slice(s![1..n_images - 1]).sum();
    let n_interior = n_images - 2;
    let force_len = n_interior * dof;
    let assembly_ci = ci_at.map_or(-1, |i| i as i64);
    let cache_metric = match metrics {
        None => Some(MetricBuildId::EUCLIDEAN),
        Some(m) if m.is_fresh_at(x, n_interior) => Some(m.build_id),
        Some(_) => None,
    };
    if let Some(metric_id) = cache_metric
        && eval.projected.len() == force_len
        && eval.assembly_ci == assembly_ci
        && eval.assembly_metric == Some(metric_id)
        && (metrics.is_none() || eval.driving.len() == force_len)
        && !state.dynamic_enabled()
    {
        let projected = eval.projected.clone();
        let driving = metrics.map(|_| eval.driving.clone());
        let max_component = state.convergence_force(&projected);
        let mut last_max = match state.last_max_force.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        *last_max = max_component;
        return Ok((pseudo_energy, projected, max_component, driving));
    }

    let BandWork { images, ks, .. } = &mut *work;
    let springs = match metrics {
        Some(_) => Springs::Metric(config.spring.at_band(energies.view(), gradients.view())),
        None => Springs::Band(BandSpring::at(&config.spring, energies.view(), ks)),
    };
    let images = fit_images(images, n_images - 2, dof);
    let metric_of = |i: usize| metrics.and_then(|m| m.per_image.get(i - 1).map(|p| (p, m.free)));
    // An armed climbing image under the metric climbs in the P-metric,
    // so a metric slice carries every image.
    let mut projected = Array1::zeros((n_images - 2) * dof);
    let mut driving = metrics.map(|_| Array1::zeros((n_images - 2) * dof));
    let assemble_all = || -> Vec<Result<(), SaddleError>> {
        use rayon::prelude::*;
        match driving.as_mut() {
            Some(driving) => images
                .par_iter_mut()
                .zip(
                    projected
                        .as_slice_mut()
                        .expect("flat force")
                        .par_chunks_mut(dof),
                )
                .zip(
                    driving
                        .as_slice_mut()
                        .expect("flat driving force")
                        .par_chunks_mut(dof),
                )
                .enumerate()
                .map(|(index, ((work, projected), driving))| {
                    let i = index + 1;
                    assemble_image_force(
                        config,
                        state,
                        &springs,
                        metric_of(i),
                        i,
                        ci_at == Some(i),
                        endpoint_first,
                        endpoint_last,
                        x,
                        energies,
                        gradients,
                        work,
                        ArrayViewMut1::from(projected),
                        Some(ArrayViewMut1::from(driving)),
                    )
                })
                .collect(),
            None => images
                .par_iter_mut()
                .zip(
                    projected
                        .as_slice_mut()
                        .expect("flat force")
                        .par_chunks_mut(dof),
                )
                .enumerate()
                .map(|(index, (work, projected))| {
                    let i = index + 1;
                    assemble_image_force(
                        config,
                        state,
                        &springs,
                        metric_of(i),
                        i,
                        ci_at == Some(i),
                        endpoint_first,
                        endpoint_last,
                        x,
                        energies,
                        gradients,
                        work,
                        ArrayViewMut1::from(projected),
                        None,
                    )
                })
                .collect(),
        }
    };
    if let Some(pool) = &state.pool {
        for result in pool.install(assemble_all) {
            result?;
        }
    } else {
        for (index, work) in images.iter_mut().enumerate() {
            let i = index + 1;
            let rows = s![index * dof..(index + 1) * dof];
            assemble_image_force(
                config,
                state,
                &springs,
                metric_of(i),
                i,
                ci_at == Some(i),
                endpoint_first,
                endpoint_last,
                x,
                energies,
                gradients,
                work,
                projected.slice_mut(rows),
                driving.as_mut().map(|force| force.slice_mut(rows)),
            )?;
        }
    }
    let mut max_component = state.convergence_force(&projected);

    // Baseline capture and arming read the relaxation force of the
    // first evaluation and of this one.
    let baseline = {
        let mut b = state.baseline.lock().unwrap_or_else(|e| e.into_inner());
        *b.get_or_insert(max_component)
    };
    state
        .atom_baseline
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_or_insert_with(|| min_mode::max_atom(projected.view()));
    if let Some(ci) = &config.climbing
        && !armed
        && (max_component < baseline * ci.trigger_factor || max_component < ci.trigger_force)
    {
        state.armed.store(true, Ordering::Relaxed);
        state
            .ci_index
            .store(climb_target.map_or(-1, |i| i as i64), Ordering::Relaxed);
        if let Some(ci_i) = climb_target {
            let index = ci_i - 1;
            let rows = s![index * dof..(index + 1) * dof];
            assemble_image_force(
                config,
                state,
                &springs,
                metric_of(ci_i),
                ci_i,
                true,
                endpoint_first,
                endpoint_last,
                x,
                energies,
                gradients,
                &mut images[index],
                projected.slice_mut(rows),
                driving.as_mut().map(|force| force.slice_mut(rows)),
            )?;
            max_component = state.convergence_force(&projected);
        }
    }

    // A force whose squared norm overflows defeats every solver's
    // dot products long before a component turns non-finite.
    if !projected.dot(&projected).is_finite() {
        return Err(SaddleError::NonFinite("band force norm"));
    }
    // The solver sees forces ASE would hand the optimizer. The stored
    // evaluation keeps the projected force before that mask, which is
    // what convergence and [`BandSession::evaluation`] report.
    let stored = if let Some(spec) = state.dynamic_spec() {
        let saved = projected.clone();
        let saved_driving = driving.clone();
        let n_interior = n_images - 2;
        let mut rows = Array2::zeros((n_interior, dof));
        for i in 0..n_interior {
            rows.row_mut(i)
                .assign(&saved.slice(s![i * dof..(i + 1) * dof]));
        }
        let interior_pos = match x.to_shape((n_interior, dof)) {
            Ok(view) => view,
            Err(err) => return Err(SaddleError::Shape(err.to_string())),
        };
        let mask = dynamic_mask(
            interior_pos.view(),
            energies.slice(s![1..n_images - 1]),
            &mut rows,
            spec,
        )?;
        let mut masked_flat = Array1::zeros(saved.len());
        for i in 0..n_interior {
            masked_flat
                .slice_mut(s![i * dof..(i + 1) * dof])
                .assign(&rows.row(i));
        }
        let visible = state.convergence_force(&masked_flat);
        let row_gate = row_convergence(state, &saved, n_interior, dof);
        let zeroed = state
            .dynamic_run()
            .decision(x, &mask, visible, &row_gate, config.force_tol);
        if zeroed.len() != n_interior {
            return Err(SaddleError::Shape(
                "dynamic NEB image count changed during the step".into(),
            ));
        }
        for (i, &clear) in zeroed.iter().enumerate() {
            let range = s![i * dof..(i + 1) * dof];
            if clear {
                projected.slice_mut(range).fill(0.0);
                if let Some(drive) = driving.as_mut() {
                    drive.slice_mut(range).fill(0.0);
                }
            } else {
                projected.slice_mut(range).assign(&saved.slice(range));
                if let Some(drive) = driving.as_mut()
                    && let Some(saved_drive) = saved_driving.as_ref()
                {
                    drive.slice_mut(range).assign(&saved_drive.slice(range));
                }
            }
        }
        Some(saved)
    } else {
        None
    };
    if let Some(last) = slot.as_mut()
        && last.x == x
    {
        match cache_metric {
            Some(metric_id) => {
                let force = match &stored {
                    Some(force) => force,
                    None => &projected,
                };
                work.store_forces(last, force, driving.as_ref());
                last.assembly_ci = state.ci().map_or(-1, |i| i as i64);
                last.assembly_metric = Some(metric_id);
            }
            None => {
                last.projected = Array1::zeros(0);
                last.driving = Array1::zeros(0);
                last.assembly_ci = i64::MIN;
                last.assembly_metric = None;
            }
        }
    }
    *state
        .last_max_force
        .lock()
        .unwrap_or_else(|e| e.into_inner()) = max_component;
    Ok((pseudo_energy, projected, max_component, driving))
}

/// The springs of an assembly: resolved per evaluation into a
/// [`SpringKind`] for the metric path, in place otherwise.
enum Springs<'a> {
    Metric(std::borrow::Cow<'a, SpringKind>),
    Band(BandSpring<'a>),
}

/// A solver step must leave finite coordinates, and must move the band
/// unless the band was already within its tolerance; otherwise a host
/// loop on [`BandSession::step`] would spin on cached evaluations.
fn check_moved(
    state: &BandState,
    before: &Array1<f64>,
    after: &Array1<f64>,
    force_tol: f64,
) -> Result<(), SaddleError> {
    if !after.iter().all(|v| v.is_finite()) {
        return Err(SaddleError::NonFinite("band step"));
    }
    let max_force = state.last_max_force();
    if after == before && max_force > force_tol {
        return Err(SaddleError::Stalled(format!(
            "band unchanged with max force {max_force:e} above {force_tol:e}"
        )));
    }
    Ok(())
}

/// The oracle's value beside a projected (band) or inverted (min-mode)
/// force is not the potential of that force, so a method that tests
/// the value (a line search, an energy acceptance) refuses good steps.
/// FIRE, L-BFGS under `Accept::Step`, Barzilai-Borwein (one oracle
/// call, no test under the session's `Accept::None`), and quick-min
/// step on the force alone.
pub(crate) fn check_force_driven(method: &Method) -> Result<(), SaddleError> {
    match method {
        Method::Fire { .. }
        | Method::Lbfgs { .. }
        | Method::Bb
        | Method::QuickMin
        | Method::Ode12r => Ok(()),
        other => Err(SaddleError::Invalid(format!(
            "{other:?} tests the oracle value, which is not the potential of a \
             projected or inverted force; use FIRE, L-BFGS, BB, quick-min, or ODE12r"
        ))),
    }
}

/// What the band holds at its current positions after a step: the
/// surface's energies (whole band) and gradients (whole band; the
/// endpoint rows are those of the last whole-band evaluation), and the
/// projected interior force the solver stepped on.
#[derive(Clone, Debug)]
pub struct BandEvaluation {
    /// The energies.
    pub energies: Array1<f64>,
    /// Energy gradients, one row per image, in the same order as the positions.
    pub gradients: Array2<f64>,
    /// `(n_images - 2) x dof`, row `r` is image `r + 1`.
    pub projected: Array2<f64>,
}

struct CachedBandSurface<'a, S> {
    state: &'a BandState,
    surface: &'a S,
}

struct CachedEvaluationOnly;

impl BandSurface for CachedEvaluationOnly {
    fn eval(
        &self,
        _positions: ArrayView2<f64>,
        _energies: &mut Array1<f64>,
        _gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        Err(SaddleError::Invalid(
            "the cached band evaluation is unavailable".into(),
        ))
    }
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
        let mut work = self.state.work.lock().unwrap_or_else(|e| e.into_inner());
        let slot = evaluate_band(
            self.state,
            &mut work,
            self.surface,
            positions.row(0),
            positions.row(last),
            x.view(),
        )?;
        let eval = slot.as_ref().expect("evaluate_band leaves an evaluation");
        energies.assign(&eval.energies);
        gradients.assign(&eval.gradients);
        Ok(())
    }
}

/// The rgmin solver of a band: the per-atom cap, the quick-min options,
/// and L-BFGS under `Accept::Step` with eOn's NEB resets.
fn build_solver(config: &BandConfig, method: &Method, interior_dof: usize) -> Solver {
    let control = Control::default()
        .with_maxiter(usize::MAX)
        .with_gtol(0.0)
        .with_istep(config.time_step)
        .with_maxmove(None);
    let mut solver = Solver::new(method.clone(), control, interior_dof);
    // The flat band vector is consecutive xyz triples, so the cap
    // applies to the largest single-atom displacement (eOn
    // maxAtomMotionApplied), not to the L2 norm of the whole
    // band step. A non-positive max_move leaves the step uncapped.
    solver.set_atom_maxmove(config.max_move);
    if config.quickmin {
        solver.set_quickmin_options(config.quickmin_options.clone());
    }
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
    solver
}

/// Stepping band relaxation over an rgmin solver. Endpoints (rows 0
/// and `n_images - 1`) never move.
///
/// # Examples
///
/// A nudged elastic band on the double well
/// V = (x² − 1)² + 2y² + 2z², one atom with minima at (±1, 0, 0) and
/// the saddle at the origin. Nine images start on a path bowed off the
/// axis, and the host steps the band until it converges.
///
/// ```
/// use ndarray::{Array1, Array2, ArrayView2};
/// use rgsaddle::{BandConfig, BandSession, BandStatus, BandSurface, SaddleError};
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
/// let config = BandConfig::default().with_force_tol(1e-3).with_max_move(0.1);
/// let mut session = BandSession::new(config, initial)?;
/// let mut report = session.step(&DoubleWell)?;
/// while report.status == BandStatus::Running {
///     report = session.step(&DoubleWell)?;
/// }
///
/// assert_eq!(report.status, BandStatus::Converged);
/// let saddle = session.positions().row(report.ci_index.unwrap()).to_owned();
/// assert!(saddle.iter().all(|c| c.abs() < 1e-2));
/// # Ok::<(), SaddleError>(())
/// ```
pub struct BandSession {
    config: BandConfig,
    positions: Array2<f64>,
    solver: Solver,
    state: BandState,
    iteration: usize,
    rtr: Option<crate::rtr::BandRtr>,
    solid: Option<SolidRun>,
    refine: min_mode::RefineState,
    exp: Option<ExpBandState>,
    /// Consecutive ODE12r steps that rejected their trial.
    unmoved: usize,
    zoom: zoom::ZoomState,
    saddle: saddle_check::SaddleState,
    string_state: string::StringState,
    free_end_state: free_end::FreeEndState,
    highs: bool,
}

impl BandSession {
    /// New.
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
        if let SpringKind::OnsagerMachlupLive { k } = config.spring
            && !k.is_finite()
        {
            return Err(SaddleError::Invalid(format!(
                "Onsager-Machlup spring constant must be finite; got {k}"
            )));
        }
        if config.solid.is_some() && matches!(config.spring, SpringKind::OnsagerMachlupLive { .. })
        {
            return Err(SaddleError::Invalid(
                "a solid-state band has a cell block, so live Onsager-Machlup displacements \
                 (atomic force over 2k) do not apply; use a uniform, weighted, or energy-weighted spring"
                    .into(),
            ));
        }
        if !(config.force_tol.is_finite() && config.force_tol >= 0.0) {
            return Err(SaddleError::Invalid(
                "band force tolerance must be finite and non-negative".into(),
            ));
        }
        if let Some(spec) = config.dynamic {
            spec.check()?;
            if config.solid.is_some() {
                return Err(SaddleError::Invalid(
                    "dynamic NEB does not run on a solid-state band".into(),
                ));
            }
        }
        if let SpringKind::Uniform { k } = config.spring
            && !k.is_finite()
        {
            return Err(SaddleError::Invalid(
                "uniform spring constant must be finite".into(),
            ));
        }
        if let SpringKind::EnergyWeighted { k_min, k_max } = config.spring
            && !(k_min.is_finite() && k_max.is_finite() && 0.0 < k_min && k_min <= k_max)
        {
            return Err(SaddleError::Invalid(format!(
                "energy-weighted springs need 0 < k_min <= k_max; got {k_min} and {k_max}"
            )));
        }
        if !initial.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("band positions"));
        }
        check_fixed(&config, initial.view())?;
        let method = if config.quickmin {
            Method::QuickMin
        } else {
            config.method.clone()
        };
        check_force_driven(&method)?;
        if let Some(refine) = &config.min_mode {
            refine.validate()?;
        }
        if let Some(check) = &config.saddle_check {
            check.validate()?;
        }
        if let Some(zoom) = &config.zoom {
            zoom::check_zoom(&config, zoom)?;
        }
        if config.string.is_some() {
            string::check_string(&config)?;
        }
        if let Some(fe) = &config.free_end {
            free_end::check_free_end(&config, fe)?;
        }
        if !(config.time_step.is_finite() && config.time_step > 0.0) {
            return Err(SaddleError::Invalid(
                "band time step must be positive and finite".into(),
            ));
        }
        let (positions, solid) = prepare_solid(&config, initial)?;
        let segment = if solid.is_some() { dof + 9 } else { dof };
        let interior_dof = (n_images - 2) * segment;
        let solver = build_solver(&config, &method, interior_dof);
        let mut session = Self {
            config,
            positions,
            solver,
            state: BandState::new(),
            iteration: 0,
            rtr: None,
            solid,
            refine: min_mode::RefineState::default(),
            exp: None,
            unmoved: 0,
            zoom: zoom::ZoomState::default(),
            saddle: saddle_check::SaddleState::default(),
            string_state: string::StringState::default(),
            free_end_state: free_end::FreeEndState::default(),
            highs: false,
        };
        if session.config.has_fixed() {
            session.state.fixed = session.config.fixed_atoms.clone();
            session.pin_fixed();
        }
        session.state.install_dynamic(session.config.dynamic);
        if let Some(precon) = session.config.exp_precon {
            session.set_exp_precon(precon)?;
        }
        Ok(session)
    }

    /// Replace the fixed-atom mask ([`BandConfig::fixed_atoms`]) between
    /// steps: one entry per atom, empty or all `false` for none. The
    /// endpoints must agree on every fixed atom up to the minimum
    /// image; the interior images take the first image's fixed
    /// coordinates. Optimizer history, the climbing state, the
    /// min-mode refinement and the last evaluation are dropped, the
    /// endpoint energies kept, and an installed exponential metric is
    /// rebuilt on the new free atoms at the next step. On an error the
    /// session is unchanged.
    pub fn set_fixed_atoms(&mut self, fixed: Vec<bool>) -> Result<(), SaddleError> {
        let candidate = self.config.clone().with_fixed_atoms(fixed);
        if self.solid.is_some() && candidate.has_fixed() {
            return Err(SaddleError::Invalid(
                "a solid-state band moves every atom with its cell; fixed atoms are not supported"
                    .into(),
            ));
        }
        check_fixed(&candidate, self.positions.view())?;
        self.config.fixed_atoms = candidate.fixed_atoms;
        self.state.fixed = if self.config.has_fixed() {
            self.config.fixed_atoms.clone()
        } else {
            Vec::new()
        };
        self.pin_fixed();
        if let Some(exp) = self.exp.as_mut() {
            exp.free = self
                .config
                .has_fixed()
                .then(|| free_coordinates(&self.config.fixed_atoms));
            exp.clear_metrics();
        }
        self.state.drop_last();
        self.restart();
        Ok(())
    }

    /// The fixed-atom mask, empty when none was set.
    #[must_use]
    pub fn fixed_atoms(&self) -> &[bool] {
        &self.config.fixed_atoms
    }

    /// Give every interior image the first image's fixed coordinates.
    pub(crate) fn pin_fixed(&mut self) {
        pin_fixed_rows(&self.state.fixed, &mut self.positions);
    }

    /// Install the per-image exponential metric on the band solver, or
    /// rebuild it at the current positions. See [`ExpBandPrecon`].
    pub fn set_exp_precon(&mut self, precon: ExpBandPrecon) -> Result<(), SaddleError> {
        if self.solid.is_some() {
            return Err(SaddleError::Invalid(
                "the exponential metric has no pair geometry for a solid-state cell block".into(),
            ));
        }
        if !(precon.a.is_finite()
            && precon.a > 0.0
            && precon.c_stab.is_finite()
            && precon.c_stab > 0.0
            && precon.r_cut_factor.is_finite()
            && precon.r_cut_factor > 1.0)
            || precon.mu.is_some_and(|mu| !(mu.is_finite() && mu > 0.0))
        {
            return Err(SaddleError::Invalid(
                "exponential metric needs A > 0, C_stab > 0, r_cut_factor > 1 and mu > 0".into(),
            ));
        }
        if self.positions.ncols() < 6 {
            return Err(SaddleError::Invalid(
                "the exponential metric needs at least two atoms per image".into(),
            ));
        }
        if !matches!(self.config.spring, SpringKind::Uniform { .. }) {
            return Err(SaddleError::Invalid(
                "the exponential metric needs a uniform spring".into(),
            ));
        }
        self.config.exp_precon = Some(precon);
        // The matrices need the surface for the energy scale, so they
        // are built on the first step.
        self.exp = Some(ExpBandState {
            precon,
            build_id: None,
            built_at: Array1::zeros(0),
            r_nn: 0.0,
            mu: precon.mu,
            metrics: Vec::new(),
            free: self
                .config
                .has_fixed()
                .then(|| free_coordinates(&self.config.fixed_atoms)),
        });
        self.state.invalidate_assembly();
        self.solver.forget_evaluation();
        Ok(())
    }

    /// The installed band metric, if any.
    #[must_use]
    pub fn exp_precon(&self) -> Option<ExpBandPrecon> {
        self.config.exp_precon
    }

    /// Energy scale `mu` of the installed band metric: the configured
    /// value, or the one estimated on the first step; `None` before the
    /// first step when none was configured, and without a metric.
    #[must_use]
    pub fn exp_scale(&self) -> Option<f64> {
        self.exp.as_ref().and_then(|e| e.mu)
    }

    /// Pair lists of the interior images, one per image with the atom
    /// indices of that image, and the largest per-atom nearest-neighbour
    /// distance over all of them. Distances are wrapped by the cell.
    ///
    /// With atoms fixed every atom still counts for `r_nn` and the
    /// cutoff, as ASE's `Exp` does under `FixAtoms`; the lists index the
    /// free atoms in order, and a bond from a free atom to a fixed one
    /// is an anchor, which adds to the free atom's diagonal only.
    fn band_neighbours(&self, r_cut_factor: f64) -> Result<(Vec<Neighbours>, f64), SaddleError> {
        let fixed = &self.state.fixed;
        let free_index: Vec<Option<usize>> = {
            let mut next = 0;
            (0..self.positions.ncols() / 3)
                .map(|a| {
                    if fixed.get(a).copied().unwrap_or(false) {
                        None
                    } else {
                        next += 1;
                        Some(next - 1)
                    }
                })
                .collect()
        };
        let n_images = self.positions.nrows();
        let n_atoms = self.positions.ncols() / 3;
        let n_frames = n_images - 2;
        let mut points = Vec::with_capacity(n_frames * n_atoms);
        for row in self.positions.slice(s![1..n_images - 1, ..]).outer_iter() {
            points.extend(
                row.as_slice()
                    .expect("a band row is contiguous")
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .copied(),
            );
        }
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for point in &points {
            for axis in 0..3 {
                lo[axis] = lo[axis].min(point[axis]);
                hi[axis] = hi[axis].max(point[axis]);
            }
        }
        let diagonal = (0..3)
            .map(|axis| (hi[axis] - lo[axis]).powi(2))
            .sum::<f64>()
            .sqrt();
        let margin = (r_cut_factor.abs() + 2.0) * diagonal.max(1.0);
        let mut vectors = self.config.cell.map_or([[0.0; 3]; 3], |cell| cell.0);
        for axis in 0..3 {
            if vectors[axis].iter().all(|value| *value == 0.0) {
                vectors[axis][axis] = hi[axis] - lo[axis] + 2.0 * margin;
            }
        }
        let origin = std::array::from_fn(|axis| lo[axis] - margin);
        let cell = linkcell::Cell::from_vectors(vectors[0], vectors[1], vectors[2], origin)
            .map_err(|err| {
                SaddleError::Invalid(format!("exponential metric neighbour cell: {err}"))
            })?;

        let mut nearest_index = vec![-1; n_frames * n_atoms];
        let mut nearest_d2 = vec![f64::NAN; n_frames * n_atoms];
        linkcell::knearest_into_many(
            &points,
            n_atoms,
            n_frames,
            &cell,
            1,
            None,
            None,
            &mut nearest_index,
            Some(&mut nearest_d2),
        )
        .map_err(|err| {
            SaddleError::Invalid(format!("exponential metric nearest neighbours: {err}"))
        })?;
        let r_nn = nearest_d2
            .iter()
            .copied()
            .filter(|distance| distance.is_finite())
            .fold(0.0_f64, f64::max)
            .sqrt();
        if !(r_nn.is_finite() && r_nn > 0.0) {
            return Ok(((0..n_frames).map(|_| Neighbours::default()).collect(), r_nn));
        }
        let r_cut = r_cut_factor * r_nn;
        let mut lists = Vec::with_capacity(n_frames);
        for frame in 0..n_frames {
            let frame_points = &points[frame * n_atoms..(frame + 1) * n_atoms];
            let mut pairs =
                linkcell::pairs_within(frame_points, &cell, r_cut, None, Some(r_cut), true)
                    .map_err(|err| {
                        SaddleError::Invalid(format!("exponential metric cutoff neighbours: {err}"))
                    })?;
            pairs.retain(|pair| pair.i != pair.j);
            pairs.sort_by(|a, b| {
                (a.i, a.j)
                    .cmp(&(b.i, b.j))
                    .then(a.dist2.total_cmp(&b.dist2))
            });
            let mut last = None;
            let mut neigh = Neighbours::default();
            for pair in pairs {
                if last == Some((pair.i, pair.j)) {
                    continue;
                }
                last = Some((pair.i, pair.j));
                let r = pair.dist2.sqrt();
                match (free_index[pair.i], free_index[pair.j]) {
                    (Some(i), Some(j)) => {
                        neigh.i.push(i);
                        neigh.j.push(j);
                        neigh.r.push(r);
                    }
                    (Some(a), None) | (None, Some(a)) => neigh.anchor.push((a, r)),
                    (None, None) => {}
                }
            }
            lists.push(neigh);
        }
        Ok((lists, r_nn))
    }

    /// Build the per-image metrics at the current positions, estimating
    /// the energy scale on the first interior image when none is held.
    fn build_metrics<S: BandSurface>(&mut self, surface: &S) -> Result<(), SaddleError> {
        let Some(exp) = self.exp.as_ref() else {
            return Ok(());
        };
        let precon = exp.precon;
        let held_mu = exp.mu;
        let (lists, r_nn) = self.band_neighbours(precon.r_cut_factor)?;
        if !(r_nn.is_finite() && r_nn > 0.0) {
            return Err(SaddleError::Invalid(
                "exponential metric: no finite nearest-neighbour distance on the band".into(),
            ));
        }
        let params = ExpParams::default()
            .with_a(precon.a)
            .with_c_stab(precon.c_stab)
            .with_mu(held_mu)
            .with_r_nn(r_nn)
            .with_r_cut(None)
            .with_recalc_mu(false);
        let dof = self.positions.ncols();
        let free = exp.free.clone();
        // The metric reads the free coordinates of an image; the fixed
        // ones stay where the image has them.
        let gather = |row: ArrayView1<f64>| -> Array1<f64> {
            match &free {
                Some(free) => free.iter().map(|&c| row[c]).collect(),
                None => row.to_owned(),
            }
        };
        let first_row = self.positions.row(1).to_owned();
        let first = gather(first_row.view());
        // The sine probe of the mu estimate spans the whole image, fixed
        // atoms included; spanning the free atoms alone could leave it
        // flat, as for a single free atom.
        let params = match &free {
            Some(_) => {
                let mut lo = [f64::INFINITY; 3];
                let mut hi = [f64::NEG_INFINITY; 3];
                for atom in first_row.exact_chunks(3) {
                    for a in 0..3 {
                        lo[a] = lo[a].min(atom[a]);
                        hi[a] = hi[a].max(atom[a]);
                    }
                }
                params.with_lengths([hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]])
            }
            None => params,
        };
        let mu = match held_mu {
            Some(mu) => mu,
            None => {
                let rows = &self.state.rows;
                let grad = |p: ArrayView1<f64>| -> Array1<f64> {
                    let geometry = match &free {
                        Some(free) => {
                            let mut full = first_row.clone();
                            for (k, &c) in free.iter().enumerate() {
                                full[c] = p[k];
                            }
                            full.into_shape_with_order((1, dof)).expect("one image")
                        }
                        None => p.to_shape((1, dof)).expect("one image").to_owned(),
                    };
                    let mut energies = Array1::zeros(1);
                    let mut gradients = Array2::zeros((1, dof));
                    rows.fetch_add(1, Ordering::Relaxed);
                    match surface.eval(geometry.view(), &mut energies, &mut gradients) {
                        Ok(()) => gather(gradients.row(0)),
                        Err(_) => Array1::from_elem(p.len(), f64::NAN),
                    }
                };
                rgmin::estimate_mu(first.view(), &lists[0], &params, grad)
                    .map_err(|e| SaddleError::Solver(format!("exponential metric: {e}")))?
            }
        };
        let mut metrics = Vec::with_capacity(lists.len());
        for (k, neigh) in lists.iter().enumerate() {
            let row = gather(self.positions.row(k + 1));
            let metric = Exp::assemble(row.view(), neigh, &params, mu, r_nn)
                .map_err(|e| SaddleError::Solver(format!("exponential metric: {e}")))?;
            metrics.push(metric);
        }
        let build_id = self.state.alloc_metric_id();
        let flat = self.interior_flat();
        if let Some(exp) = self.exp.as_mut() {
            exp.build_id = Some(build_id);
            exp.built_at = flat;
            exp.r_nn = r_nn;
            exp.mu = Some(mu);
            exp.metrics = metrics;
            // The cached driving force was solved with the matrices just
            // replaced. Drop it on every slot, as set_exp_precon does,
            // so the next assembly cannot reuse it.
            self.state.invalidate_assembly();
            self.solver.forget_evaluation();
        }
        Ok(())
    }

    /// Build the band metrics on the first step and rebuild them when an
    /// atom has moved more than `r_nn / 2` since the last build. The
    /// energy scale found on the first build is kept.
    pub(super) fn refresh_exp<S: BandSurface>(&mut self, surface: &S) -> Result<(), SaddleError> {
        let Some(exp) = &self.exp else {
            return Ok(());
        };
        let n_interior = self.positions.nrows() - 2;
        let flat = self.interior_flat();
        if exp.is_stale(flat.view(), n_interior) {
            self.build_metrics(surface)?;
        }
        Ok(())
    }

    /// Drop built metric matrices once an interior atom has moved past
    /// `r_nn / 2` from where they were assembled, keeping `mu` for the
    /// next rebuild.
    pub(super) fn invalidate_stale_exp(&mut self) {
        let n_interior = self.positions.nrows() - 2;
        let flat = self.interior_flat();
        let stale = match &self.exp {
            Some(exp) => exp.build_id.is_some() && exp.is_stale(flat.view(), n_interior),
            None => false,
        };
        if stale {
            if let Some(exp) = self.exp.as_mut() {
                exp.clear_metrics();
            }
            self.state.invalidate_assembly();
            self.solver.forget_evaluation();
        }
    }

    /// Select the optional HiGHS feasible-set step; requires the `highs` feature.
    /// Disabled by default.
    pub fn set_highs(&mut self, enabled: bool) {
        self.highs = enabled;
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
        if self.config.string.is_some() && config.is_some() {
            return Err(SaddleError::Invalid(
                "the simplified string method uses the band's first-order stepper, \
                 not the trust-region stepper"
                    .into(),
            ));
        }
        if self.config.dynamic.is_some() && config.is_some() {
            return Err(SaddleError::Invalid(
                "dynamic NEB does not use the trust-region stepper".into(),
            ));
        }
        if let Some(config) = config {
            config.validate()?;
        }
        self.rtr = config.map(|config| crate::rtr::BandRtr::new(config, false));
        self.restart();
        Ok(())
    }

    /// Turn dynamic NEB on or off between steps. `None` restores a
    /// full evaluation of every image. The image cache is dropped, so
    /// the next step evaluates the band again. A solid-state band and
    /// the trust-region stepper are refused.
    pub fn set_dynamic(&mut self, dynamic: Option<DynamicNeb>) -> Result<(), SaddleError> {
        if let Some(spec) = dynamic {
            spec.check()?;
            if self.solid.is_some() {
                return Err(SaddleError::Invalid(
                    "dynamic NEB does not run on a solid-state band".into(),
                ));
            }
            if self.rtr.is_some() {
                return Err(SaddleError::Invalid(
                    "dynamic NEB does not use the trust-region stepper".into(),
                ));
            }
            if self.config.string.is_some() {
                return Err(SaddleError::Invalid(
                    "dynamic NEB is for the NEB; the simplified string method evaluates every image"
                        .into(),
                ));
            }
        }
        self.config.dynamic = dynamic;
        self.state.install_dynamic(dynamic);
        Ok(())
    }

    /// Evaluate the band on `threads` worker threads, kept for the
    /// session. One (the default) keeps the surface and force assembly
    /// on the calling thread; zero is an error. More than one also
    /// splits tangent, spring, projection and metric solves by image.
    ///
    /// Each surface call is split into `min(threads, rows)` contiguous
    /// image blocks, one [`BandSurface::eval`] per block, and the
    /// answers are stored at their image indices once every block has
    /// finished. When the surface's answer for a row
    /// depends on that row alone, the band, its forces and its
    /// trajectory are bitwise those of one thread, and the first
    /// failing image is the one reported. A surface that tells the
    /// whole-band evaluation from an interior one by the row count
    /// should stay at one thread, as
    /// [`crate::autoneb::AutoNebTuning::threads`] says.
    pub fn set_image_threads(&mut self, threads: usize) -> Result<(), SaddleError> {
        self.state.pool = match threads {
            1 => None,
            _ => Some(crate::autoneb::ImageEvalPool::new(threads)?),
        };
        Ok(())
    }

    /// The image thread count of [`BandSession::set_image_threads`].
    pub fn image_threads(&self) -> usize {
        self.state.pool.as_ref().map_or(1, |pool| pool.threads())
    }

    /// Cartesian coordinates. A band stores one row per image.
    pub fn positions(&self) -> ArrayView2<'_, f64> {
        self.positions.view()
    }

    /// Stable image identifiers, when installed via [`BandSession::set_ids`].
    #[must_use]
    pub fn ids(&self) -> &[crate::stencil::ImageId] {
        &self.state.ids
    }

    /// Install one stable [`crate::stencil::ImageId`] per band image so
    /// [`BandSurface::eval_with_ids`] and [`crate::stencil::Surface::eval`]
    /// receive those identifiers on whole-band and interior evaluations.
    pub fn set_ids(&mut self, ids: Vec<crate::stencil::ImageId>) -> Result<(), SaddleError> {
        if !ids.is_empty() && ids.len() != self.positions.nrows() {
            return Err(SaddleError::Shape(format!(
                "band image id count {} does not match {} images",
                ids.len(),
                self.positions.nrows()
            )));
        }
        self.state.ids = ids;
        Ok(())
    }

    /// Completed band iterations so far.
    #[must_use]
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// Oriented cells, one per image, when the band carries the
    /// solid-state block. The Jacobian that built them stays fixed.
    pub fn image_cells(&self) -> Option<&[Cell]> {
        self.solid.as_ref().map(|solid| solid.cells.as_slice())
    }

    /// Jacobian fixed from the mean endpoint volume at session build.
    /// Replace the initial time step of FIRE and quick-min
    /// (`BandConfig::time_step`). The solver is rebuilt, so optimizer
    /// history and the HiGHS selection go; cached surface values stay.
    pub fn set_time_step(&mut self, time_step: f64) -> Result<(), SaddleError> {
        if !(time_step.is_finite() && time_step > 0.0) {
            return Err(SaddleError::Invalid(
                "band time step must be positive and finite".into(),
            ));
        }
        self.config.time_step = time_step;
        let method = if self.config.quickmin {
            Method::QuickMin
        } else {
            self.config.method.clone()
        };
        let dof = self.positions.ncols();
        let segment = if self.solid.is_some() { dof + 9 } else { dof };
        let interior_dof = (self.positions.nrows() - 2) * segment;
        self.solver = build_solver(&self.config, &method, interior_dof);
        self.highs = false;
        self.clear_string_state();
        self.state.restart();
        self.refine = min_mode::RefineState::default();
        Ok(())
    }

    /// The band steps with quick-min (`BandConfig::quickmin`).
    pub fn is_quickmin(&self) -> bool {
        self.config.quickmin
    }

    /// Replace the quick-min variant of a quick-min band; the velocity
    /// and `dt` restart. A band on another method ignores the call.
    pub fn set_quickmin_options(&mut self, options: rgmin::QuickMinOptions) {
        if self.config.quickmin {
            self.solver.set_quickmin_options(options.clone());
            self.config.quickmin_options = options;
        }
    }

    /// Solid jacobian.
    pub fn solid_jacobian(&self) -> Option<f64> {
        self.solid.as_ref().map(|solid| solid.jacobian)
    }

    /// Climbing image.
    pub fn climbing_image(&self) -> Option<usize> {
        self.state.ci()
    }

    /// Drop optimizer history and keep the climbing image, its
    /// baseline, and the cached surface values. A hand-off that moved
    /// an image outside the solver uses this so the next band step
    /// does not inherit a stale quasi-Newton pair.
    pub(crate) fn forget_optimizer(&mut self) {
        self.solver.forget();
        self.clear_string_state();
        if let Some(rtr) = &mut self.rtr {
            let gate = self.state.force_gate;
            *rtr = crate::rtr::BandRtr::new(rtr.config, false);
            rtr.set_force_gate(gate);
        }
    }

    /// Replace the band (host-side move between steps: acquisition,
    /// reparameterization, or a resync of positions the host read
    /// back). Rows equal to the current ones up to the minimum image
    /// (within `1e-10` per component) keep the session's own copy, so
    /// a host that wraps atoms into its cell and hands the band back
    /// changes nothing: the endpoint energies and the last evaluation
    /// survive and the next step evaluates only the interior images. A
    /// moved endpoint drops the endpoint energies (the next evaluation
    /// carries the whole band); a moved interior image drops the last
    /// evaluation. Optimizer history survives; call
    /// [`BandSession::restart`] to drop it as well. With atoms fixed the
    /// new endpoints must agree on each of them, as at
    /// [`BandSession::new`], and the interior images take the first
    /// image's fixed coordinates.
    pub fn set_positions(&mut self, mut positions: Array2<f64>) -> Result<(), SaddleError> {
        if positions.dim() != self.positions.dim() {
            return Err(SaddleError::Shape("set_positions shape".into()));
        }
        if !positions.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("band positions"));
        }
        check_fixed(&self.config, positions.view())?;
        pin_fixed_rows(&self.state.fixed, &mut positions);
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
                self.free_end_state.fixed_endpoint_energy = None;
                // The interior may be unchanged, so the solver's own
                // cached force (assembled against the old endpoint)
                // would answer the next step; drop it, keep the memory.
                self.solver.forget_evaluation();
            }
            // The last evaluation's tangents read the neighbours, so
            // any moved row retires it.
            moved = true;
        }
        self.pin_fixed();
        if moved {
            self.invalidate_stale_exp();
            self.state.drop_last();
            self.refine = min_mode::RefineState::default();
            self.saddle.forget();
            self.clear_string_state();
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
        self.free_end_state.fixed_endpoint_energy = None;
        self.state.drop_last();
    }

    /// Set the force tolerance without dropping quasi-Newton pairs.
    pub(crate) fn set_force_tol(&mut self, force_tol: f64) -> Result<(), SaddleError> {
        if !(force_tol > 0.0 && force_tol.is_finite()) {
            return Err(SaddleError::Invalid(
                "band force tolerance must be finite and positive".into(),
            ));
        }
        self.config.force_tol = force_tol;
        Ok(())
    }

    /// Replace the climbing rule without dropping quasi-Newton pairs.
    /// The cached force is forgotten so the next step uses the new rule.
    pub(crate) fn set_climbing(&mut self, climbing: Option<CiConfig>) {
        self.config.climbing = climbing;
        self.state.restart();
        self.refine = min_mode::RefineState::default();
        self.solver.forget_evaluation();
    }

    /// Record endpoint energies the host already holds.
    ///
    /// The next evaluation asks only for the interior images, so the
    /// endpoint gradients stay unknown and never block the climbing
    /// image ([`climb_target`]). Call this before the first step when
    /// those two images were evaluated already, so a new window does
    /// not repeat them.
    pub(crate) fn seed_endpoint_energies(&mut self, energies: [f64; 2]) -> Result<(), SaddleError> {
        if !energies.iter().all(|energy| energy.is_finite()) {
            return Err(SaddleError::NonFinite("band endpoint energies"));
        }
        self.state.set_endpoints(Some(energies));
        Ok(())
    }

    /// Restart the relaxation on the same surface: drop quasi-Newton
    /// history (FIRE velocity, L-BFGS pairs), the climbing baseline,
    /// and the armed climbing image, keep the endpoint energies and
    /// the last evaluation. Mirrors eOn's optimizer reset after a
    /// reparameterization; with [`BandSession::set_positions`] for the
    /// moved images the next step evaluates only the interior images.
    pub fn restart(&mut self) {
        self.solver.forget();
        self.clear_string_state();
        self.state.restart();
        self.refine = min_mode::RefineState::default();
        self.saddle.forget();
        if let Some(rtr) = &mut self.rtr {
            *rtr = crate::rtr::BandRtr::new(rtr.config, false);
            rtr.set_force_gate(self.state.force_gate);
        }
    }

    fn expected_metric_at(&self, x: ArrayView1<f64>) -> Option<MetricBuildId> {
        if self.solid.is_some() || self.rtr.is_some() {
            return Some(MetricBuildId::EUCLIDEAN);
        }
        match &self.exp {
            Some(exp) => exp
                .fresh_metrics(x, self.positions.nrows() - 2)
                .map(|m| m.build_id),
            None => Some(MetricBuildId::EUCLIDEAN),
        }
    }

    fn has_assembled_at(&self, x: ArrayView1<f64>, force_len: usize) -> bool {
        let Some(expected) = self.expected_metric_at(x) else {
            return false;
        };
        let slot = match self.state.last.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        slot.as_ref().is_some_and(|last| {
            last.x == x
                && last.projected.len() == force_len
                && last.assembly_metric == Some(expected)
                && (expected == MetricBuildId::EUCLIDEAN || last.driving.len() == force_len)
        })
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
        let expected_metric = self.expected_metric_at(x.view())?;
        if self.solid.is_none() && self.refine.converged {
            let force_len = (self.positions.nrows() - 2) * self.positions.ncols();
            if !self.has_assembled_at(x.view(), force_len) {
                let has_surface_values = match self.state.last.lock() {
                    Ok(guard) => guard,
                    Err(poisoned) => poisoned.into_inner(),
                }
                .as_ref()
                .is_some_and(|last| last.x == x);
                if has_surface_values {
                    self.assemble_accepted(&CachedEvaluationOnly, x.view())
                        .ok()?;
                }
            }
        }
        let slot = match self.state.last.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        let last = slot.as_ref().filter(|l| {
            l.x == x && (self.solid.is_some() || l.assembly_metric == Some(expected_metric))
        })?;
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

    pub(super) fn cache_free_end_evaluation(
        &self,
        energies: Array1<f64>,
        gradients: Array2<f64>,
        projected: Array1<f64>,
    ) {
        let x = self.interior_flat();
        let assembly_metric = self.expected_metric_at(x.view());
        let evaluation = BandEval {
            x,
            energies,
            gradients,
            driving: projected.clone(),
            projected,
            assembly_ci: i64::MIN,
            assembly_metric,
            stresses: Array2::zeros((0, 0)),
            band_positions: Array2::zeros((0, 0)),
            band_cells: Vec::new(),
        };
        *lock_guard(&self.state.last) = Some(evaluation);
        *lock_guard(&self.state.previous) = None;
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
        self.pin_fixed();
    }

    /// One solver step over the assembled band force. The host
    /// interleaves its policy between calls.
    ///
    /// A surface error inside the step returns `Err` with the band
    /// unchanged. The solver's internal state (FIRE velocity,
    /// quasi-Newton pairs) has seen the failed evaluation; a host that
    /// recovers the surface and steps again may call
    /// [`BandSession::restart`] first to drop it.
    ///
    /// A converged step certifies the climbing image
    /// ([`BandConfig::saddle_check`]) at its first convergence at that
    /// position; later converged steps report the same certificate.
    pub fn step<S: BandSurface>(&mut self, surface: &S) -> Result<BandReport, SaddleError> {
        let mut report = self.step_inner(surface)?;
        if report.status == BandStatus::Converged {
            self.saddle_check(surface, &mut report)?;
        }
        Ok(report)
    }

    /// Assemble the accepted geometry, including the preconditioned
    /// driving force when a metric is installed. A plain residual
    /// assembly would clear that force.
    fn assemble_accepted<S: BandSurface>(
        &self,
        surface: &S,
        x: ArrayView1<f64>,
    ) -> Result<(), SaddleError> {
        let n_images = self.positions.nrows();
        let metrics = match (&self.exp, &self.rtr) {
            (Some(exp), None) => Some(exp.fresh_metrics(x, n_images - 2).ok_or_else(|| {
                SaddleError::Invalid(
                    "exponential metric is not current at the accepted band".into(),
                )
            })?),
            _ => None,
        };
        assemble_band_with(
            &self.config,
            &self.state,
            surface,
            self.positions.row(0),
            self.positions.row(n_images - 1),
            x,
            metrics,
        )?;
        Ok(())
    }

    fn step_inner<S: BandSurface>(&mut self, surface: &S) -> Result<BandReport, SaddleError> {
        if self.solid.is_some() {
            if self.rtr.is_some() {
                return Err(SaddleError::Invalid(
                    "solid-state band does not use the trust-region stepper".into(),
                ));
            }
            return self.step_solid(surface);
        }
        if self.config.climbing.is_some()
            && let Some(report) = self.refine_step(surface)?
        {
            return Ok(report);
        }
        if let Some(string_cfg) = self.config.string {
            return self.step_string(surface, string_cfg);
        }
        if let Some(fe_cfg) = self.config.free_end {
            let fe_rep = self.step_free_end_inner(surface, fe_cfg)?;
            return Ok(BandReport {
                status: fe_rep.status,
                max_force: fe_rep.max_force,
                ci_index: fe_rep.ci_index,
                iteration: fe_rep.iteration,
                surface_rows: fe_rep.surface_rows,
                curvature: None,
                saddle_index: None,
            });
        }
        if self.rtr.is_some() {
            if self.config.dynamic.is_some() {
                return Err(SaddleError::Invalid(
                    "dynamic NEB does not use the trust-region stepper".into(),
                ));
            }
            return self.step_rtr(surface);
        }
        let n_images = self.positions.nrows();
        let dof = self.positions.ncols();
        let interior_dof = (n_images - 2) * dof;
        // The metric probe's rows count in this step.
        self.state.take_error();
        self.state.rows.store(0, Ordering::Relaxed);
        self.refresh_exp(surface)?;

        let endpoint_first = self.positions.row(0).to_owned();
        let endpoint_last = self.positions.row(n_images - 1).to_owned();
        let mut x = self.interior_flat();
        let before = x.clone();
        let stepped = {
            let config = &self.config;
            let state = &self.state;
            let metrics = match &self.exp {
                Some(exp) => Some(exp.fresh_metrics(before.view(), n_images - 2).ok_or_else(
                    || {
                        SaddleError::Invalid(
                            "exponential metric is not current at the start of the band step"
                                .into(),
                        )
                    },
                )?),
                None => None,
            };
            if config.dynamic.is_some() {
                state.dynamic_run().begin_step(x.clone());
                // The solver reuses its cached force at this geometry, so
                // the oracle's first call is a trial and would not record
                // the mask. Record it from the evaluation already stored.
                seed_dynamic_decision(state, x.view(), config.force_tol);
            }
            let oracle = Oracle::unbounded(interior_dof, |query: ArrayView1<f64>| {
                let pinned = if config.dynamic.is_some() {
                    state.dynamic_run().pin(query)
                } else {
                    None
                };
                let y = match &pinned {
                    Some(pinned) => pinned.view(),
                    None => query,
                };
                match assemble_band_with(
                    config,
                    state,
                    surface,
                    endpoint_first.view(),
                    endpoint_last.view(),
                    y,
                    metrics,
                ) {
                    Ok((e, f, _, driving)) => (e, -driving.unwrap_or(f)),
                    Err(err) => {
                        state.record_error(err);
                        (f64::INFINITY, Array1::zeros(interior_dof))
                    }
                }
            });
            let stepped = self.solver.step(&oracle, &mut x);
            drop(oracle);
            stepped
        };
        // The surface error is the root cause of whatever the solver
        // did with an infinite energy and a zero gradient, so it
        // outranks a solver error. `x` is a local copy; the session's
        // positions stay where they were.
        if let Some(err) = self.state.take_error() {
            // The solver cached the failed point's infinite energy and
            // zero force; the next step must evaluate the band again.
            self.solver.forget_evaluation();
            return Err(err);
        }
        stepped.map_err(|e| SaddleError::Solver(e.to_string()))?;
        if self.config.dynamic.is_some() {
            self.state.dynamic_run().snap(&mut x);
        }
        // A dynamic step that no longer moves, with every force the
        // optimizer still sees under force_tol, is ASE's stop: images
        // under the scaled threshold do not count.
        let dynamic_settled = self.config.dynamic.is_some()
            && x == before
            && self
                .state
                .dynamic_run()
                .visible_max()
                .is_some_and(|visible| visible <= self.config.force_tol);
        // ODE12r keeps the band where it was when it rejects a trial and
        // retries with a shorter step, so an unchanged band is a stall
        // only once the step has shrunk past any use (64 rejections at
        // a quarter each).
        if dynamic_settled {
            self.unmoved = 0;
        } else if x == before && matches!(self.config.method, Method::Ode12r) && self.unmoved < 64 {
            self.unmoved += 1;
        } else {
            self.unmoved = 0;
            check_moved(&self.state, &before, &x, self.config.force_tol)?;
        }
        self.scatter_interior(x.view());
        self.iteration += 1;
        self.refresh_exp(surface)?;

        // A solver that ends on a point it did not evaluate last (a
        // line search that backtracked) is assembled once more. That
        // assembly carries the metric on every stepper: a plain residual
        // would drop the driving force, and the next step would solve
        // the metric again. The cache answers when the accepted point
        // was the last evaluation, and the start of the step comes back
        // from the previous slot when the stepper rejected its trial.
        let at_last = self.state.restore_at(&x) && self.has_assembled_at(x.view(), interior_dof);
        if !at_last {
            self.assemble_accepted(surface, x.view())?;
        }
        let max_force = self.state.last_max_force();
        let status = if max_force <= self.config.force_tol || dynamic_settled {
            BandStatus::Converged
        } else {
            BandStatus::Running
        };
        if !self.zoom_latch(status) {
            self.refine_latch(status)?;
        }
        Ok(BandReport {
            status,
            max_force,
            ci_index: self.state.ci(),
            iteration: self.iteration,
            surface_rows: self.state.rows.load(Ordering::Relaxed),
            curvature: None,
            saddle_index: None,
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
        let before = x.clone();
        let stepped = self.solver.step(&oracle, &mut x);
        drop(oracle);
        if let Some(err) = self.state.take_error() {
            // The solver cached the failed point's infinite energy and
            // zero force; the next step must evaluate the band again.
            self.solver.forget_evaluation();
            return Err(err);
        }
        stepped.map_err(|e| SaddleError::Solver(e.to_string()))?;
        check_moved(&self.state, &before, &x, self.config.force_tol)?;
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
            curvature: None,
            saddle_index: None,
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
        // The products and the trial push the start out of the cache;
        // a rejected step ends on it again.
        let start = self
            .state
            .last
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
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
        self.state.reinstate(start, &interior);
        assemble_band(
            &self.config,
            &self.state,
            surface,
            first.view(),
            last.view(),
            interior.view(),
        )?;
        self.positions.assign(&trial);
        self.pin_fixed();
        self.invalidate_stale_exp();
        self.iteration += 1;
        let max_force = self.state.last_max_force();
        let status = if max_force <= self.config.force_tol {
            BandStatus::Converged
        } else {
            BandStatus::Running
        };
        if !self.zoom_latch(status) {
            self.refine_latch(status)?;
        }
        Ok(BandReport {
            status,
            max_force,
            ci_index: self.state.ci(),
            iteration: self.iteration,
            surface_rows: self.state.rows.load(Ordering::Relaxed),
            curvature: None,
            saddle_index: None,
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

    #[test]
    fn image_translations_reach_every_image_in_range() {
        let cube = Cell([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]);
        assert!(cube.image_translations(4.0).is_empty());
        let slab = Cell([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.0]]);
        let steps = slab.image_translations(4.0);
        assert_eq!(steps.len(), 8);
        assert!(steps.iter().all(|t| t[2] == 0.0));
        let thin = Cell([[1e-3, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]);
        assert_eq!(thin.image_translations(4.0).len(), 32);
        // Every image of a wrapped displacement within the cutoff is the
        // wrapped one or one of the translations from it.
        let r3 = 3.0_f64.sqrt();
        let hex = Cell([[4.0, 0.0, 0.0], [2.0, 2.0 * r3, 0.0], [0.0, 0.0, 10.0]]);
        let cutoff = 4.0;
        let steps = hex.image_translations(cutoff);
        let fractions = [-0.5, -0.3, 0.0, 0.2, 0.5];
        for s0 in fractions {
            for s1 in fractions {
                for s2 in fractions {
                    let mut d = Array1::from_iter(
                        (0..3).map(|c| s0 * hex.0[0][c] + s1 * hex.0[1][c] + s2 * hex.0[2][c]),
                    );
                    hex.minimum_image(&mut d);
                    for n in (-3_i32..=3).flat_map(|a| {
                        (-3_i32..=3).flat_map(move |b| (-3_i32..=3).map(move |c| [a, b, c]))
                    }) {
                        if n == [0, 0, 0] {
                            continue;
                        }
                        let t: [f64; 3] = std::array::from_fn(|c| {
                            (0..3).map(|k| f64::from(n[k]) * hex.0[k][c]).sum()
                        });
                        let r = (0..3).map(|c| (d[c] + t[c]).powi(2)).sum::<f64>().sqrt();
                        if r < cutoff {
                            assert!(
                                steps
                                    .iter()
                                    .any(|u| (0..3).all(|c| (u[c] - t[c]).abs() < 1e-12)),
                                "image {n:?} of {d} at {r} is missing"
                            );
                        }
                    }
                }
            }
        }
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
        BandConfig::default().with_climbing(
            CiConfig::default()
                .with_trigger_factor(1.0)
                .with_trigger_force(f64::INFINITY),
        )
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
        let config = BandConfig::default().with_remove_translation(false);
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
    fn image_below_an_endpoint_still_climbs() {
        let (positions, gradient) = two_atom_band();
        let (projected, state) =
            assemble(&infinite_climb(), vec![0.0, 1.0, 5.0], &positions, gradient);
        // The tangent points at the higher endpoint, so the climb
        // inverts the parallel 2 as on an interior peak.
        assert_force(&projected, &[-3.5, 1.0, 0.0, 3.5, -1.0, 0.0]);
        assert_eq!(state.ci(), Some(1));
    }

    #[test]
    fn image_below_an_endpoint_that_slopes_into_the_band_keeps_the_spring() {
        // The higher endpoint pushes its first atom back along the band
        // with a force of 4: the band rises to it with no saddle between.
        let (positions, mut gradient) = two_atom_band();
        gradient[(2, 0)] = 4.0;
        let (projected, state) =
            assemble(&infinite_climb(), vec![0.0, 1.0, 5.0], &positions, gradient);
        assert_force(&projected, &[-2.5, 1.0, 0.0, 2.5, -1.0, 0.0]);
        assert_eq!(state.ci(), None);
    }

    #[test]
    fn endpoint_force_across_the_band_leaves_the_climb() {
        let (positions, mut gradient) = two_atom_band();
        gradient[(2, 1)] = 4.0;
        let (projected, state) =
            assemble(&infinite_climb(), vec![0.0, 1.0, 5.0], &positions, gradient);
        assert_force(&projected, &[-3.5, 1.0, 0.0, 3.5, -1.0, 0.0]);
        assert_eq!(state.ci(), Some(1));
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

    #[test]
    fn linkcell_band_neighbours_match_the_previous_pair_scan() {
        let cells = [
            Cell([[7.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 9.0]]),
            Cell([[7.0, 0.0, 0.0], [1.1, 7.5, 0.0], [-0.6, 0.8, 8.0]]),
        ];
        let mut seed = 0xbb67_ae85_84ca_a73b_u64;
        let mut random = || {
            seed = seed
                .wrapping_mul(2_862_933_555_777_941_757)
                .wrapping_add(3_037_000_493);
            (seed >> 11) as f64 / (1_u64 << 53) as f64
        };
        for cell in cells {
            let mic = linkcell::Cell::from_vectors(cell.0[0], cell.0[1], cell.0[2], [0.0; 3])
                .expect("periodic cell");
            let n_images = 6;
            let n_atoms = 24;
            let positions = Array2::from_shape_fn((n_images, 3 * n_atoms), |(image, col)| {
                let atom = col / 3;
                let axis = col % 3;
                let tag = (image * n_atoms + atom) as f64;
                let frac = [
                    (random() + 0.003 * tag).fract(),
                    (random() + 0.005 * tag).fract(),
                    (random() + 0.007 * tag).fract(),
                ];
                frac[0] * cell.0[0][axis] + frac[1] * cell.0[1][axis] + frac[2] * cell.0[2][axis]
            });
            let config = BandConfig::default()
                .with_cell(cell)
                .with_min_mode(None)
                .with_climbing(None);
            let session = BandSession::new(config, positions.clone()).expect("band");
            let (got, got_r_nn) = session.band_neighbours(2.0).expect("linkcell neighbours");

            let mut all_pairs = Vec::new();
            let mut want_r_nn = 0.0_f64;
            for image in 1..n_images - 1 {
                let row = positions.row(image);
                let mut nearest = vec![f64::INFINITY; n_atoms];
                let mut pairs = Vec::new();
                for i in 0..n_atoms {
                    for j in i + 1..n_atoms {
                        let p = [row[3 * i], row[3 * i + 1], row[3 * i + 2]];
                        let q = [row[3 * j], row[3 * j + 1], row[3 * j + 2]];
                        let distance = mic.dist2_euclidean(p, q).sqrt();
                        nearest[i] = nearest[i].min(distance);
                        nearest[j] = nearest[j].min(distance);
                        pairs.push((i, j, distance));
                    }
                }
                want_r_nn = nearest.into_iter().fold(want_r_nn, f64::max);
                all_pairs.push(pairs);
            }
            assert_abs_diff_eq!(got_r_nn, want_r_nn, epsilon = 2e-12);
            let cutoff = 2.0 * want_r_nn;
            for (actual, pairs) in got.iter().zip(all_pairs) {
                let expected: Vec<_> = pairs
                    .into_iter()
                    .filter(|(_, _, distance)| *distance < cutoff)
                    .collect();
                assert_eq!(actual.i.len(), expected.len());
                assert!(actual.anchor.is_empty());
                for (index, (i, j, distance)) in expected.into_iter().enumerate() {
                    assert_eq!((actual.i[index], actual.j[index]), (i, j));
                    assert_abs_diff_eq!(actual.r[index], distance, epsilon = 2e-12);
                }
            }
        }
    }

    #[test]
    fn cached_metric_force_does_not_repeat_pcg_solves() {
        let positions = Array2::from_shape_vec(
            (3, 9),
            vec![
                0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 0.0, 1.2, 0.1, 0.2, 0.1, 0.0, 1.2, 0.2, 0.0, 0.1,
                1.3, 0.2, 0.4, 0.2, 0.0, 1.3, 0.4, 0.1, 0.2, 1.4, 0.3,
            ],
        )
        .expect("positions");
        let gradient = Array2::from_shape_fn((3, 9), |(row, col)| {
            0.2 * (row + 1) as f64 - 0.03 * col as f64
        });
        let surface = Prescribed {
            energy: vec![0.0, 1.0, 0.2],
            gradient,
        };
        let config = BandConfig::default()
            .with_climbing(None)
            .with_min_mode(None)
            .with_remove_translation(false)
            .with_exp_precon(ExpBandPrecon::default().with_mu(1.0));
        let mut session = BandSession::new(config, positions).expect("band");
        session.refresh_exp(&surface).expect("metric");
        let x = session.interior_flat();
        let metrics = metric_view(&session);
        let first = assemble_band_with(
            &session.config,
            &session.state,
            &surface,
            session.positions.row(0),
            session.positions.row(2),
            x.view(),
            metrics,
        )
        .expect("first assembly");
        let solves = session.state.pcg_solves.load(Ordering::Relaxed);
        assert_eq!(solves, 1);
        let second = assemble_band_with(
            &session.config,
            &session.state,
            &surface,
            session.positions.row(0),
            session.positions.row(2),
            x.view(),
            metrics,
        )
        .expect("cached assembly");
        assert_eq!(session.state.pcg_solves.load(Ordering::Relaxed), solves);
        assert_eq!(first.1, second.1);
        assert_eq!(first.3, second.3);
    }

    struct Harmonic;

    impl BandSurface for Harmonic {
        fn eval(
            &self,
            positions: ArrayView2<f64>,
            energies: &mut Array1<f64>,
            gradients: &mut Array2<f64>,
        ) -> Result<(), SaddleError> {
            for (i, row) in positions.outer_iter().enumerate() {
                let mut energy = 0.0;
                for (c, &q) in row.iter().enumerate() {
                    let k = 1.0 + 0.1 * c as f64;
                    energy += 0.5 * k * q * q;
                    gradients[(i, c)] = k * q;
                }
                energies[i] = energy;
            }
            Ok(())
        }
    }

    struct BacktrackingBand {
        calls: AtomicUsize,
    }

    impl BandSurface for BacktrackingBand {
        fn eval(
            &self,
            positions: ArrayView2<f64>,
            energies: &mut Array1<f64>,
            gradients: &mut Array2<f64>,
        ) -> Result<(), SaddleError> {
            let call = self.calls.fetch_add(1, Ordering::Relaxed);
            let energy = match call {
                0 => 1.0,
                1 => 1.0 - f64::EPSILON,
                _ => 2.0,
            };
            energies.fill(energy);
            for i in 0..positions.nrows() {
                gradients[(i, 0)] = 0.001;
                gradients[(i, 1)] = -0.002;
            }
            Ok(())
        }
    }

    fn metric_view(session: &BandSession) -> Option<BandMetrics<'_>> {
        let x = session.interior_flat();
        session
            .exp
            .as_ref()
            .and_then(|exp| exp.fresh_metrics(x.view(), session.positions.nrows() - 2))
    }

    #[test]
    fn fire_finish_off_the_last_evaluation_keeps_the_driving_force() {
        let positions = Array2::from_shape_vec(
            (3, 9),
            vec![
                0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 0.0, 1.2, 0.1, 0.2, 0.1, 0.0, 1.2, 0.2, 0.0, 0.1,
                1.3, 0.2, 0.4, 0.2, 0.0, 1.3, 0.4, 0.1, 0.2, 1.4, 0.3,
            ],
        )
        .expect("positions");
        let config = BandConfig::default()
            .with_method(Method::fire(rgmin::FireKind::V2))
            .with_climbing(None)
            .with_min_mode(None)
            .with_saddle_check(None)
            .with_remove_translation(false)
            .with_exp_precon(ExpBandPrecon::default().with_mu(1.0));
        let mut session = BandSession::new(config, positions).expect("band");
        assert!(!matches!(session.config.method, Method::Ode12r));
        let surface = Harmonic;
        let report = session.step(&surface).expect("fire step");
        assert_eq!(report.status, BandStatus::Running);
        let accepted = session.interior_flat();
        let solves = session.state.pcg_solves.load(Ordering::Relaxed);
        let _ = assemble_band_with(
            &session.config,
            &session.state,
            &surface,
            session.positions.row(0),
            session.positions.row(2),
            accepted.view(),
            metric_view(&session),
        )
        .expect("accepted geometry is cached");
        assert_eq!(
            session.state.pcg_solves.load(Ordering::Relaxed),
            solves,
            "the fire step left no driving force at the accepted geometry"
        );

        // The finish used when the accepted point is not the oracle's
        // last evaluation. FIRE reaches it when a step leaves that point.
        let mut off = accepted.clone();
        off[0] += 0.05;
        let before = session.state.pcg_solves.load(Ordering::Relaxed);
        session
            .assemble_accepted(&surface, off.view())
            .expect("finish");
        let finished = session.state.pcg_solves.load(Ordering::Relaxed);
        assert!(
            finished > before,
            "the finish assembled a plain residual and did not solve the driving force"
        );
        let _ = assemble_band_with(
            &session.config,
            &session.state,
            &surface,
            session.positions.row(0),
            session.positions.row(2),
            off.view(),
            metric_view(&session),
        )
        .expect("cached finish");
        assert_eq!(session.state.pcg_solves.load(Ordering::Relaxed), finished);
    }

    #[test]
    fn lbfgs_step_finishes_the_accepted_point_with_the_metric() {
        let positions = Array2::from_shape_vec(
            (3, 9),
            vec![
                0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 0.0, 1.2, 0.1, 0.2, 0.1, 0.0, 1.2, 0.2, 0.0, 0.1,
                1.3, 0.2, 0.4, 0.2, 0.0, 1.3, 0.4, 0.1, 0.2, 1.4, 0.3,
            ],
        )
        .expect("positions");
        let config = BandConfig::default()
            .with_method(Method::lbfgs())
            .with_climbing(None)
            .with_min_mode(None)
            .with_saddle_check(None)
            .with_remove_translation(false)
            .with_exp_precon(ExpBandPrecon::default().with_mu(1.0));
        let mut session = BandSession::new(config, positions).expect("band");
        session
            .solver
            .set_lbfgs_step(rgmin::LbfgsStep::Wolfe)
            .expect("Wolfe L-BFGS");
        session.solver.set_geometry_step(false);
        session.solver.set_accept(Accept::None);
        let surface = BacktrackingBand {
            calls: AtomicUsize::new(0),
        };
        let before = session.interior_flat();
        let report = session.step(&surface).expect("L-BFGS step");
        assert_eq!(report.status, BandStatus::Running);
        let calls = surface.calls.load(Ordering::Relaxed);
        assert!(calls > 3, "the Wolfe search made {calls} surface calls");
        let accepted = session.interior_flat();
        assert_ne!(accepted, before);

        let solves = session.state.pcg_solves.load(Ordering::Relaxed);
        let _ = assemble_band_with(
            &session.config,
            &session.state,
            &surface,
            session.positions.row(0),
            session.positions.row(2),
            accepted.view(),
            metric_view(&session),
        )
        .expect("accepted geometry is cached");
        assert_eq!(
            session.state.pcg_solves.load(Ordering::Relaxed),
            solves,
            "the off-last finish did not cache the accepted driving force"
        );
    }

    #[test]
    fn rebuilding_the_metric_drops_the_cached_driving_force() {
        let positions = Array2::from_shape_vec(
            (3, 9),
            vec![
                0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 0.0, 1.2, 0.1, 0.2, 0.1, 0.0, 1.2, 0.2, 0.0, 0.1,
                1.3, 0.2, 0.4, 0.2, 0.0, 1.3, 0.4, 0.1, 0.2, 1.4, 0.3,
            ],
        )
        .expect("positions");
        let gradient = Array2::from_shape_fn((3, 9), |(row, col)| {
            0.2 * (row + 1) as f64 - 0.03 * col as f64
        });
        let surface = Prescribed {
            energy: vec![0.0, 1.0, 0.2],
            gradient,
        };
        let config = BandConfig::default()
            .with_climbing(None)
            .with_min_mode(None)
            .with_remove_translation(false)
            .with_exp_precon(ExpBandPrecon::default().with_mu(1.0));
        let mut session = BandSession::new(config, positions).expect("band");
        session.refresh_exp(&surface).expect("metric");
        let x = session.interior_flat();
        let _ = assemble_band_with(
            &session.config,
            &session.state,
            &surface,
            session.positions.row(0),
            session.positions.row(2),
            x.view(),
            metric_view(&session),
        )
        .expect("assembly");
        let solves = session.state.pcg_solves.load(Ordering::Relaxed);
        {
            let mut last = session.state.last.lock().unwrap_or_else(|e| e.into_inner());
            let eval = last.as_mut().expect("cached evaluation");
            eval.driving.fill(f64::NAN);
        }
        session.build_metrics(&surface).expect("rebuild");
        let again = assemble_band_with(
            &session.config,
            &session.state,
            &surface,
            session.positions.row(0),
            session.positions.row(2),
            x.view(),
            metric_view(&session),
        )
        .expect("assembly after rebuild");
        let driving = again.3.expect("driving force");
        assert!(driving.iter().all(|v| v.is_finite()));
        assert!(
            session.state.pcg_solves.load(Ordering::Relaxed) > solves,
            "the rebuilt metric reused the cached driving force"
        );
    }

    #[test]
    fn step_after_metric_rebuild_starts_with_the_new_driving_force() {
        let positions = Array2::from_shape_vec(
            (3, 9),
            vec![
                0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 0.0, 1.2, 0.1, 0.2, 0.1, 0.0, 1.2, 0.2, 0.0, 0.1,
                1.3, 0.2, 0.4, 0.2, 0.0, 1.3, 0.4, 0.1, 0.2, 1.4, 0.3,
            ],
        )
        .expect("positions");
        let config = BandConfig::default()
            .with_method(Method::fire(rgmin::FireKind::V2))
            .with_climbing(None)
            .with_min_mode(None)
            .with_saddle_check(None)
            .with_remove_translation(false)
            .with_exp_precon(ExpBandPrecon::default().with_mu(1.0));
        let mut session = BandSession::new(config, positions).expect("band");
        let surface = Harmonic;
        session.step(&surface).expect("first step");

        session.exp.as_mut().expect("metric state").precon.a *= 4.0;
        session.build_metrics(&surface).expect("metric rebuild");
        let solves = session.state.pcg_solves.load(Ordering::Relaxed);
        session.step(&surface).expect("step after rebuild");
        assert!(
            session.state.pcg_solves.load(Ordering::Relaxed) >= solves + 2,
            "the solver skipped the new-metric force at the accepted start"
        );
    }

    struct ShiftedSaddle;

    impl BandSurface for ShiftedSaddle {
        fn eval(
            &self,
            positions: ArrayView2<f64>,
            energies: &mut Array1<f64>,
            gradients: &mut Array2<f64>,
        ) -> Result<(), SaddleError> {
            let saddle = [3.2, 0.1, 0.0, 1.2, 0.2, 0.0, 0.1, 1.3, 0.2];
            for (i, row) in positions.outer_iter().enumerate() {
                let d0 = row[0] - saddle[0];
                let mut energy = 10.0 - d0 * d0;
                gradients[(i, 0)] = -2.0 * d0;
                for c in 1..9 {
                    let stiff = 3.0 + 0.5 * c as f64;
                    let dc = row[c] - saddle[c];
                    energy += 0.5 * stiff * dc * dc;
                    gradients[(i, c)] = stiff * dc;
                }
                energies[i] = energy;
            }
            Ok(())
        }
    }

    #[test]
    fn converged_refinement_past_rebuild_radius_matches_fresh_evaluation() {
        let positions = Array2::from_shape_vec(
            (4, 9),
            vec![
                -2.8, 0.1, 0.0, 1.2, 0.2, 0.0, 0.1, 1.3, 0.2, // endpoint 0
                0.2, 0.1, 0.0, 1.2, 0.2, 0.0, 0.1, 1.3, 0.2, // interior image 1
                -0.4, 0.5, 0.0, 0.6, 0.6, 0.0, -0.5, 1.7, 0.2, // interior image 2
                6.8, 0.1, 0.0, 1.2, 0.2, 0.0, 0.1, 1.3, 0.2, // endpoint 3
            ],
        )
        .expect("positions");
        let refine = MinModeRefine::default()
            .with_alignment_tol(0.2)
            .with_stable_steps(1);
        let config = BandConfig::default()
            .with_method(Method::fire(rgmin::FireKind::V2))
            .with_climbing(CiConfig::always())
            .with_min_mode(refine)
            .with_saddle_check(None)
            .with_remove_translation(false)
            .with_max_move(0.5)
            .with_force_tol(0.0001)
            .with_exp_precon(ExpBandPrecon::default().with_mu(1.0));

        let surface = ShiftedSaddle;
        let mut session = BandSession::new(config.clone(), positions).expect("band");
        let first = session.step(&surface).expect("initial band step");
        assert_eq!(first.status, BandStatus::Running);
        let exp = session.exp.as_ref().expect("metric built");
        let built_at = exp.built_at.clone();
        let initial_r_nn = exp.r_nn;
        assert_abs_diff_eq!(initial_r_nn, 1.49_f64.sqrt(), epsilon = 0.001);

        let mut report = first;
        while report.status == BandStatus::Running && session.iteration < 200 {
            report = session.step(&surface).expect("refinement step");
        }
        assert_eq!(report.status, BandStatus::Converged);
        assert!(
            report.curvature.is_some(),
            "convergence must return through refine_step"
        );
        let moved = rgmin::max_atom_displacement(built_at.view(), session.interior_flat().view());
        assert!(
            moved > 2.5 && rgmin::needs_rebuild(moved, initial_r_nn),
            "expected move past rebuild threshold {}, got {moved} (r_nn {initial_r_nn})",
            0.5 * initial_r_nn
        );

        let got = session.evaluation().expect("converged band evaluation");
        let converged_positions = session.positions().to_owned();
        let mut fresh = BandSession::new(config.with_min_mode(None), converged_positions)
            .expect("fresh band at converged geometry");
        fresh.refresh_exp(&surface).expect("fresh metric");
        fresh
            .assemble_accepted(&surface, fresh.interior_flat().view())
            .expect("fresh assembly");
        let want = fresh.evaluation().expect("fresh evaluation");

        assert_eq!(got.energies, want.energies);
        assert_eq!(
            got.gradients.slice(s![1..3, ..]),
            want.gradients.slice(s![1..3, ..])
        );
        for ((r, c), &g) in got.projected.indexed_iter() {
            let w = want.projected[(r, c)];
            assert_abs_diff_eq!(g, w, epsilon = 0.000000000001);
        }
    }

    #[test]
    fn set_exp_precon_forgets_the_cached_euclidean_solver_evaluation() {
        let positions = Array2::from_shape_vec(
            (3, 9),
            vec![
                0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 0.0, 1.2, 0.1, 0.2, 0.1, 0.0, 1.2, 0.2, 0.0, 0.1,
                1.3, 0.2, 0.4, 0.2, 0.0, 1.3, 0.4, 0.1, 0.2, 1.4, 0.3,
            ],
        )
        .expect("positions");
        let config = BandConfig::default()
            .with_method(Method::fire(rgmin::FireKind::V2))
            .with_climbing(None)
            .with_min_mode(None)
            .with_saddle_check(None)
            .with_remove_translation(false);
        let mut session = BandSession::new(config, positions).expect("band");
        let surface = Harmonic;
        session.step(&surface).expect("euclidean step");
        assert!(session.evaluation().is_some());

        let accepted = session.interior_flat();
        session
            .set_exp_precon(ExpBandPrecon::default().with_mu(1.0))
            .expect("install exponential metric");
        assert!(
            session.evaluation().is_none(),
            "evaluation must stay unavailable until the new metric is built"
        );

        let evaluated_start = AtomicBool::new(false);
        let expected_start = accepted.clone();
        let oracle = Oracle::unbounded(accepted.len(), |x: ArrayView1<f64>| {
            if x == expected_start.view() {
                evaluated_start.store(true, Ordering::Relaxed);
            }
            (1.0, Array1::from_elem(x.len(), 0.25))
        });
        let mut trial = accepted;
        session
            .solver
            .step(&oracle, &mut trial)
            .expect("solver step after set_exp_precon");
        assert!(
            evaluated_start.load(Ordering::Relaxed),
            "set_exp_precon left the Euclidean evaluation cached on the solver"
        );
    }

    #[test]
    fn metric_build_id_mismatch_on_identical_geometry_forces_rebuild() {
        let positions = Array2::from_shape_vec(
            (3, 9),
            vec![
                0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 0.0, 1.2, 0.1, 0.2, 0.1, 0.0, 1.2, 0.2, 0.0, 0.1,
                1.3, 0.2, 0.4, 0.2, 0.0, 1.3, 0.4, 0.1, 0.2, 1.4, 0.3,
            ],
        )
        .expect("positions");
        let gradient = Array2::from_shape_fn((3, 9), |(row, col)| {
            0.2 * (row + 1) as f64 - 0.03 * col as f64
        });
        let surface = Prescribed {
            energy: vec![0.0, 1.0, 0.2],
            gradient,
        };
        let config = BandConfig::default()
            .with_climbing(None)
            .with_min_mode(None)
            .with_saddle_check(None)
            .with_remove_translation(false)
            .with_exp_precon(ExpBandPrecon::default().with_mu(1.0));
        let mut session = BandSession::new(config, positions).expect("band");
        session.refresh_exp(&surface).expect("initial metric");
        let x = session.interior_flat();
        let first_id = session
            .exp
            .as_ref()
            .and_then(|exp| exp.build_id)
            .expect("first MetricBuildId");
        let first = assemble_band_with(
            &session.config,
            &session.state,
            &surface,
            session.positions.row(0),
            session.positions.row(2),
            x.view(),
            metric_view(&session),
        )
        .expect("first assembly");
        assert_eq!(session.state.pcg_solves.load(Ordering::Relaxed), 1);
        assert!(session.has_assembled_at(x.view(), x.len()));
        assert!(session.evaluation().is_some());

        let first_eval = session
            .state
            .last
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();

        {
            let exp_ref = session.exp.as_ref().expect("metric state");
            let euclidean_view = BandMetrics {
                build_id: MetricBuildId::EUCLIDEAN,
                built_at: x.view(),
                r_nn: exp_ref.r_nn,
                per_image: exp_ref.metrics.as_slice(),
                free: None,
            };
            assert!(
                !euclidean_view.is_fresh_at(x.view(), 1),
                "Euclidean MetricBuildId must not be accepted as a fresh exponential metric"
            );

            let exp = session.exp.as_mut().expect("metric state");
            assert_eq!(exp.built_at, x);
            exp.precon.a *= 4.0;
            exp.build_id = None;
        }
        session
            .refresh_exp(&surface)
            .expect("metric rebuild on identical geometry");
        let second_id = session
            .exp
            .as_ref()
            .and_then(|exp| exp.build_id)
            .expect("rebuilt MetricBuildId");
        assert_ne!(
            second_id, first_id,
            "refresh_exp skipped rebuilding the metric on identical geometry"
        );

        // Restore the intact `first_id` assembly at the identical geometry `x`
        // so only `MetricBuildId` differs between `last` and `session.exp`.
        *session.state.last.lock().unwrap_or_else(|e| e.into_inner()) = first_eval;
        assert!(
            !session.has_assembled_at(x.view(), x.len()),
            "has_assembled_at accepted a mismatched MetricBuildId at identical geometry"
        );
        assert!(
            session.evaluation().is_none(),
            "evaluation() returned forces solved under an older MetricBuildId"
        );

        let solves_before = session.state.pcg_solves.load(Ordering::Relaxed);
        let second = assemble_band_with(
            &session.config,
            &session.state,
            &CachedEvaluationOnly,
            session.positions.row(0),
            session.positions.row(2),
            x.view(),
            metric_view(&session),
        )
        .expect("reassembly under new MetricBuildId");
        assert_eq!(
            session.state.pcg_solves.load(Ordering::Relaxed),
            solves_before + 1,
            "MetricBuildId mismatch on identical geometry reused the cached driving force"
        );
        assert_ne!(second.1, first.1);
        assert_ne!(second.3, first.3);
        assert!(session.has_assembled_at(x.view(), x.len()));
        assert!(session.evaluation().is_some());
    }

    #[test]
    fn restart_keeps_the_projected_force_and_retires_the_climbing_key() {
        struct AlongX;
        impl BandSurface for AlongX {
            fn eval(
                &self,
                positions: ArrayView2<f64>,
                energies: &mut Array1<f64>,
                gradients: &mut Array2<f64>,
            ) -> Result<(), SaddleError> {
                for (i, row) in positions.outer_iter().enumerate() {
                    energies[i] = row[1];
                    gradients.row_mut(i).fill(0.0);
                    gradients[(i, 1)] = 1.0;
                }
                Ok(())
            }
        }
        let positions =
            Array2::from_shape_vec((3, 3), vec![-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0])
                .expect("positions");
        let config = BandConfig::default()
            .with_climbing(CiConfig::always())
            .with_min_mode(None)
            .with_remove_translation(false);
        let mut session = BandSession::new(config, positions).expect("band");
        session.step(&AlongX).expect("first step");
        let (projected, metric, assembly_ci) = {
            let slot = session.state.last.lock().unwrap_or_else(|e| e.into_inner());
            let eval = slot.as_ref().expect("assembly");
            (
                eval.projected.clone(),
                eval.assembly_metric,
                eval.assembly_ci,
            )
        };
        assert!(projected.iter().any(|value| *value != 0.0));
        assert!(metric.is_some());
        assert!(assembly_ci >= 0);
        assert!(session.climbing_image().is_some());

        session.restart();
        assert!(session.climbing_image().is_none());
        {
            let slot = session.state.last.lock().unwrap_or_else(|e| e.into_inner());
            let eval = slot.as_ref().expect("surface cache");
            assert_eq!(eval.projected, projected);
            assert_eq!(eval.assembly_metric, metric);
            assert_eq!(eval.assembly_ci, i64::MIN);
        }
        let kept = session
            .evaluation()
            .expect("restart keeps the last evaluation");
        assert_eq!(kept.projected.len(), projected.len());
        assert!(
            kept.projected
                .iter()
                .zip(projected.iter())
                .all(|(got, want)| got == want)
        );

        session.step(&AlongX).expect("step after restart");
        assert!(session.climbing_image().is_some());
    }
}
