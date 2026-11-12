//! Solid-state minimum-mode search on atoms plus cell strain.
//!
//! The translation is the dimer or Lanczos step already in
//! [`crate::minmode`]: the lowest mode of the generalized Hessian, the
//! force inverted along it, and one rgmin step on that vector.
//! Goswami, Gunde, and Jónsson, Front. Chem. 14, 1807063 (2026),
//! doi:10.3389/fchem.2026.1807063, Equation 3, is that reflected force.
//! The same search in eOn is the dimer of Goswami, Masterov, Kamath,
//! Peña-Torres, and Jónsson, J. Chem. Theory Comput. 21, 7935 (2025),
//! doi:10.1021/acs.jctc.5c00866. Force calls below are counted as in
//! Goswami, AIP Advances 15, 085210 (2025), doi:10.1063/5.0283639.
//!
//! The coordinate those steppers see is the solid-state dimer of Xiao,
//! Sheppard, Rogal, and Henkelman, J. Chem. Phys. 140, 174104 (2014),
//! doi:10.1063/1.4873437. Strain sits next to the atomic displacement
//! with the Jacobian of Sheppard, Xiao, Chemelewski, Johnson, and
//! Henkelman, J. Chem. Phys. 136, 074103 (2012), doi:10.1063/1.3684549,
//!
//! `J = V^{1/3} N^{1/6}`,
//!
//! fixed from the cell at the start of the search. The generalized
//! cell force is `-(V/J) σ` at that reference, with `σ` the Cauchy
//! stress, and the chain rule carries the same pairing out to a finite
//! right strain. A frozen cell drops the strain block and calls
//! [`PointSurface::eval`], so the step is the fixed-cell dimer.
//!
//! [`IndexCheck`] is optional. When it is set, a point that meets the
//! force tolerance is accepted only if the projected Hessian has one
//! negative eigenvalue. The three rigid translations, and any other
//! directions [`PointSurface::excluded_modes`] names, are removed
//! before that count. A higher index is left along its softest negative
//! mode, or the step is marked [`SolidDimerReport::rejected`] once the
//! escape budget is spent. A minimum of the enthalpy is not reported as
//! a barrier.

use std::sync::Mutex;

use ndarray::{Array1, Array2, ArrayView1, s};

use crate::cell_log::{M3, m3_det, m3_from_row9, m3_inv, m3_mul, m3_to_row9};
use crate::error::SaddleError;
use crate::minmode::{MinModeConfig, MinModeReport, MinModeSession, MinModeStatus, PointSurface};
use crate::solid_state::{orient_lower_triangular, solid_state_jacobian, zero_upper};

fn not_above(value: f64, floor: f64) -> bool {
    value.partial_cmp(&floor) != Some(std::cmp::Ordering::Greater)
}

/// Voigt order of the lower triangle: xx, yy, zz, yz, xz, xy.
pub const STRAIN_VOIGT: [(usize, usize); 6] = [(0, 0), (1, 1), (2, 2), (2, 1), (2, 0), (1, 0)];

/// Which of the six lower-triangular strains are free.
///
/// The order is [`STRAIN_VOIGT`]. A frozen mask is the fixed-cell dimer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CellMask {
    free: [bool; 6],
}

impl CellMask {
    /// Every lower-triangular strain is free.
    pub const fn all() -> Self {
        Self { free: [true; 6] }
    }

    /// No strain is free. The search is the fixed-cell dimer.
    pub const fn frozen() -> Self {
        Self { free: [false; 6] }
    }

    /// `free` is xx, yy, zz, yz, xz, xy.
    pub const fn voigt(free: [bool; 6]) -> Self {
        Self { free }
    }

    /// The six flags in [`STRAIN_VOIGT`] order.
    pub const fn flags(self) -> [bool; 6] {
        self.free
    }

    /// True when every strain is held.
    pub const fn is_frozen(self) -> bool {
        let mut i = 0;
        while i < 6 {
            if self.free[i] {
                return false;
            }
            i += 1;
        }
        true
    }
}

impl Default for CellMask {
    fn default() -> Self {
        Self::all()
    }
}

/// Settings of one solid-state dimer.
///
/// Non-exhaustive: build it with [`SolidDimerConfig::new`] and the
/// `with_*` setters. The defaults free every strain, apply no external
/// stress, and use the 2012 Jacobian.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SolidDimerConfig {
    /// Dimer or Lanczos search and its rgmin translator.
    pub search: MinModeConfig,
    /// Strains that enter the generalized vector.
    pub mask: CellMask,
    /// External Cauchy stress, added to the cell force. A positive
    /// diagonal entry pushes the cell inward, the same sign as
    /// [`crate::SolidState::pressure`]. The upper triangle is folded
    /// onto the lower one.
    pub external_stress: M3,
    /// Positive scale on the Jacobian. 1 is the 2012 value.
    pub weight: f64,
    /// Projected Hessian at a converged point. `None` accepts the
    /// minimum-mode test alone: negative curvature and a small force.
    pub index: Option<IndexCheck>,
    /// A negative curvature `λ` is converged only when the largest
    /// force component is at most `flat * |λ|`, and still at most the
    /// search tolerance. Zero leaves that tolerance unchanged. The
    /// harmonic residual along the mode is then about `flat` when the
    /// force sits on the mode.
    pub flat: f64,
}

impl SolidDimerConfig {
    /// `search` with every strain free, no external stress, weight 1.
    pub fn new(search: MinModeConfig) -> Self {
        Self {
            search,
            mask: CellMask::all(),
            external_stress: [[0.0; 3]; 3],
            weight: 1.0,
            index: None,
            flat: 0.0,
        }
    }

    /// The fixed-cell dimer. `step` calls [`PointSurface::eval`].
    pub fn frozen(search: MinModeConfig) -> Self {
        Self::new(search).with_mask(CellMask::frozen())
    }

    /// Strains that move.
    pub fn with_mask(mut self, mask: CellMask) -> Self {
        self.mask = mask;
        self
    }

    /// External Cauchy stress. The upper triangle is folded down.
    pub fn with_external_stress(mut self, stress: M3) -> Self {
        self.external_stress = stress;
        self
    }

    /// Hydrostatic pressure. A positive value pushes the cell inward.
    pub fn with_pressure(mut self, pressure: f64) -> Self {
        for (i, row) in self.external_stress.iter_mut().enumerate() {
            row[i] = pressure;
        }
        self
    }

    /// Positive scale on the Jacobian.
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }

    /// Require one negative projected eigenvalue at convergence.
    pub fn with_index_check(mut self, index: IndexCheck) -> Self {
        self.index = Some(index);
        self
    }

    /// Tighten a flat negative mode to a harmonic residual of `flat`.
    ///
    /// Zero restores the search tolerance. A non-positive value other
    /// than zero is refused by [`SolidDimer::new`].
    pub fn with_flat(mut self, flat: f64) -> Self {
        self.flat = flat;
        self
    }
}

/// How a converged solid-state dimer checks that it is index 1.
///
/// The lowest eigenvalues are a Lanczos decomposition of central
/// Hessian actions. Directions from [`PointSurface::excluded_modes`]
/// are projected out of every action, so a rigid translation is not a
/// negative mode. Build with [`IndexCheck::first_order`].
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct IndexCheck {
    /// How many lowest eigenvalues to resolve.
    pub lowest: usize,
    /// An eigenvalue below this is negative.
    pub negative: f64,
    /// Central-difference step of one Hessian action.
    pub step: f64,
    /// How many times a higher-index point may be left.
    pub escapes: usize,
    /// Generalized step along the softest extra negative mode.
    pub escape: f64,
}

impl IndexCheck {
    /// One negative eigenvalue below -1e-4, eight lowest modes,
    /// central step 1e-4, six escapes of length 0.02.
    pub const fn first_order() -> Self {
        Self {
            lowest: 8,
            negative: -1.0e-4,
            step: 1.0e-4,
            escapes: 6,
            escape: 0.02,
        }
    }

    /// Central-difference step.
    pub const fn with_step(mut self, step: f64) -> Self {
        self.step = step;
        self
    }

    /// How many higher-index points may be left. Zero rejects at once.
    pub const fn with_escapes(mut self, escapes: usize) -> Self {
        self.escapes = escapes;
        self
    }

    /// Displacement along the softest extra negative mode.
    pub const fn with_escape(mut self, escape: f64) -> Self {
        self.escape = escape;
        self
    }
}

/// Projected spectrum at one geometry.
///
/// Eigenvalues ascend. Eigenvectors are columns of `vectors`.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct IndexReport {
    /// Eigenvalues strictly below [`IndexCheck::negative`].
    pub negative: usize,
    pub values: Vec<f64>,
    pub vectors: Array2<f64>,
    /// Generalized surface evaluations spent on the Hessian actions.
    pub evaluations: usize,
}

impl IndexReport {
    /// Exactly one projected eigenvalue is below the cutoff.
    pub fn is_first_order(&self) -> bool {
        self.negative == 1
    }
}

/// Lowest projected eigenvalues of `surface` at `x`.
///
/// `x` is the coordinate [`PointSurface::eval`] reads: Cartesian for a
/// frozen cell, the generalized vector for a joint chart. The count
/// ignores directions [`PointSurface::excluded_modes`] returns.
pub fn index_at<S: PointSurface>(
    surface: &S,
    x: ArrayView1<f64>,
    check: IndexCheck,
) -> Result<IndexReport, SaddleError> {
    if x.is_empty() || !x.iter().all(|v| v.is_finite()) {
        return Err(SaddleError::Invalid(
            "index check needs a finite coordinate".into(),
        ));
    }
    if check.lowest == 0 || !(check.step.is_finite() && check.step > 0.0) {
        return Err(SaddleError::Invalid(
            "index check needs a positive difference step and at least one eigenvalue".into(),
        ));
    }
    if !check.negative.is_finite() || !(check.escape.is_finite() && check.escape >= 0.0) {
        return Err(SaddleError::Invalid(
            "index check cutoff and escape step must be finite".into(),
        ));
    }
    let modes = orthonormal_excluded(surface, x)?;
    let calls = std::cell::Cell::new(0usize);
    let (values, vectors) = ritz_lowest(surface, x, &modes, check, &calls)?;
    let negative = values
        .iter()
        .filter(|value| **value < check.negative)
        .count();
    Ok(IndexReport {
        negative,
        values,
        vectors,
        evaluations: calls.get(),
    })
}

impl Default for SolidDimerConfig {
    fn default() -> Self {
        Self::new(MinModeConfig::default())
    }
}

/// One solid-state dimer step.
#[derive(Clone, Debug)]
pub struct SolidDimerReport {
    pub status: MinModeStatus,
    pub max_force: f64,
    pub curvature: f64,
    pub rotations: usize,
    pub iteration: usize,
    /// Evaluations of the generalized surface, centre included.
    /// A frozen step counts [`PointSurface::eval`] calls.
    pub evaluations: usize,
    /// Cauchy stress at the accepted point when the joint surface
    /// recorded it. Frozen steps leave this empty; call
    /// [`SolidDimer::measure_stress`].
    pub stress: Option<M3>,
    /// Cell volume at the accepted point.
    pub volume: f64,
    /// Negative projected eigenvalues when this step tested the index.
    /// Empty when the check is off, or when a higher-index point was
    /// left and the search is still running.
    pub negative_modes: Option<usize>,
    /// The force tolerance was met and the projected Hessian is not
    /// index 1. [`Self::status`] stays [`MinModeStatus::Converged`]
    /// because the force test passed; this flag is the rejection.
    pub rejected: bool,
    /// Generalized evaluations spent on index checks so far, this step
    /// included.
    pub index_evaluations: usize,
    /// Projected eigenvalues at a point the index check accepted or
    /// rejected. Empty while the search is still leaving a higher-index point.
    pub eigenvalues: Vec<f64>,
}

/// Right-strain chart `h = h0 (I + ε)`, `r = r0 (I + ε)`.
///
/// `r0` is the atomic block of the generalized vector. The free
/// strains, in [`STRAIN_VOIGT`] order, are stored as `J ε`. `J` is
/// fixed from `h0`. The enthalpy is `E + V0 σ_ext : ε` with the upper
/// triangle of `ε` held at zero, so only the lower triangle of
/// `σ_ext` does work.
#[derive(Clone, Debug)]
pub struct JointChart {
    reference: M3,
    jacobian: f64,
    v0: f64,
    n_cart: usize,
    free: Vec<(usize, usize)>,
    external: M3,
}

impl JointChart {
    /// `cell` is the reference `h0`, lower triangular. `n_atoms` fixes
    /// the Jacobian together with `weight`.
    pub fn new(
        cell: M3,
        n_atoms: usize,
        mask: CellMask,
        external_stress: M3,
        weight: f64,
    ) -> Result<Self, SaddleError> {
        if n_atoms < 1 {
            return Err(SaddleError::Invalid(
                "solid-state dimer needs at least one atom".into(),
            ));
        }
        if !lower_triangular(&cell) {
            return Err(SaddleError::Invalid(
                "solid-state dimer reference cell must be lower triangular".into(),
            ));
        }
        let v0 = m3_det(cell).abs();
        let jacobian = solid_state_jacobian(v0, n_atoms, weight)?;
        let mut external = fold_lower(external_stress);
        zero_upper(&mut external);
        if !external.iter().flatten().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("solid-state external stress"));
        }
        let free = STRAIN_VOIGT
            .into_iter()
            .zip(mask.flags())
            .filter_map(|(ij, on)| on.then_some(ij))
            .collect::<Vec<_>>();
        Ok(Self {
            reference: cell,
            jacobian,
            v0,
            n_cart: n_atoms * 3,
            free,
            external,
        })
    }

    /// Jacobian `J`, fixed for the chart.
    pub fn jacobian(&self) -> f64 {
        self.jacobian
    }

    /// Reference cell `h0`.
    pub fn reference(&self) -> M3 {
        self.reference
    }

    /// Volume of the reference cell.
    pub fn reference_volume(&self) -> f64 {
        self.v0
    }

    /// Length of the generalized vector: `3N` plus the free strains.
    pub fn dimension(&self) -> usize {
        self.n_cart + self.free.len()
    }

    /// Number of free strains.
    pub fn n_strain(&self) -> usize {
        self.free.len()
    }

    /// Lower-triangular right strain stored in `q`.
    pub fn strain(&self, q: ArrayView1<f64>) -> Result<M3, SaddleError> {
        self.check_q(q)?;
        let mut eps = [[0.0; 3]; 3];
        for (k, &(i, j)) in self.free.iter().enumerate() {
            eps[i][j] = q[self.n_cart + k] / self.jacobian;
        }
        Ok(eps)
    }

    /// Cartesian positions and cell of a generalized vector.
    pub fn geometry(&self, q: ArrayView1<f64>) -> Result<(Array1<f64>, M3), SaddleError> {
        let eps = self.strain(q)?;
        let deform = deform(eps);
        let h = m3_mul(self.reference, deform);
        if not_above(m3_det(h).abs(), 1e-18) {
            return Err(SaddleError::Invalid(
                "solid-state dimer strain collapsed the cell".into(),
            ));
        }
        let r0 = q.slice(s![..self.n_cart]);
        let mut r = Array1::zeros(self.n_cart);
        for atom in 0..self.n_cart / 3 {
            let row = [r0[3 * atom], r0[3 * atom + 1], r0[3 * atom + 2]];
            let out = row_mul(row, deform);
            for c in 0..3 {
                r[3 * atom + c] = out[c];
            }
        }
        Ok((r, h))
    }

    /// `E + V0 σ_ext : ε(q)`.
    pub fn enthalpy(&self, energy: f64, q: ArrayView1<f64>) -> Result<f64, SaddleError> {
        if !energy.is_finite() {
            return Err(SaddleError::NonFinite("solid-state enthalpy"));
        }
        let eps = self.strain(q)?;
        let mut work = 0.0;
        for (i, row) in eps.iter().enumerate() {
            for (j, eps_ij) in row.iter().enumerate() {
                work += self.external[i][j] * eps_ij;
            }
        }
        Ok(energy + self.v0 * work)
    }

    /// Gradient of the enthalpy with respect to `q`.
    ///
    /// `g_cart` is `∂E/∂r` at fixed cell. `sigma` is the Cauchy stress
    /// of the current cell, `σ_ij = (1/V) ∂E/∂ε_ij` for an
    /// infinitesimal right strain at fixed fractional coordinates.
    /// At zero strain and zero external stress the cell block is
    /// `(V/J) σ`.
    pub fn gradient(
        &self,
        q: ArrayView1<f64>,
        g_cart: ArrayView1<f64>,
        sigma: M3,
    ) -> Result<Array1<f64>, SaddleError> {
        self.check_q(q)?;
        if g_cart.len() != self.n_cart || !g_cart.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::Shape("solid-state Cartesian gradient".into()));
        }
        if !sigma.iter().flatten().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("solid-state stress"));
        }
        let eps = self.strain(q)?;
        let deform = deform(eps);
        let h = m3_mul(self.reference, deform);
        let volume = m3_det(h).abs();
        if not_above(volume, 1e-18) {
            return Err(SaddleError::Invalid(
                "solid-state dimer strain collapsed the cell".into(),
            ));
        }
        let mut out = Array1::zeros(self.dimension());
        for atom in 0..self.n_cart / 3 {
            let g = [g_cart[3 * atom], g_cart[3 * atom + 1], g_cart[3 * atom + 2]];
            // ∂E/∂r0_a = sum_c M_ac g_c, from r = r0 M.
            for a in 0..3 {
                let mut acc = 0.0;
                for c in 0..3 {
                    acc += deform[a][c] * g[c];
                }
                out[3 * atom + a] = acc;
            }
        }
        // ∂E/∂ε = M^{-T} (V σ), then the external term V0 σ_ext.
        let pulled = m3_mul(transpose(m3_inv(deform)?), scale(volume, sigma));
        let d_eps = add(pulled, scale(self.v0, self.external));
        for (k, &(i, j)) in self.free.iter().enumerate() {
            out[self.n_cart + k] = d_eps[i][j] / self.jacobian;
        }
        Ok(out)
    }

    fn check_q(&self, q: ArrayView1<f64>) -> Result<(), SaddleError> {
        if q.len() != self.dimension() || !q.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::Shape(
                "solid-state generalized coordinate".into(),
            ));
        }
        Ok(())
    }
}

/// Central-difference Cauchy stress of [`PointSurface::eval_in_cell`].
///
/// Each of the nine right-strain entries is perturbed with the atoms
/// held at fixed fractional coordinates. `step` is the strain
/// increment. The result is `(1/V) ∂E/∂ε_ij` with the other entries
/// held fixed, so a potential that is not rotationally invariant can
/// return an unsymmetric tensor. The virial is `V` times this tensor.
pub fn finite_difference_cauchy_stress<S: PointSurface>(
    surface: &S,
    positions: ArrayView1<f64>,
    cell: M3,
    step: f64,
) -> Result<M3, SaddleError> {
    if !(step.is_finite() && step > 0.0) {
        return Err(SaddleError::Invalid(
            "stress finite difference needs a positive step".into(),
        ));
    }
    if !positions.len().is_multiple_of(3) || positions.is_empty() {
        return Err(SaddleError::Shape(
            "stress finite difference positions".into(),
        ));
    }
    let volume = m3_det(cell).abs();
    if not_above(volume, 1e-18) || !cell.iter().flatten().all(|v| v.is_finite()) {
        return Err(SaddleError::Invalid(
            "stress finite difference needs a cell with positive volume".into(),
        ));
    }
    let mut sigma = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut plus = [[0.0; 3]; 3];
            let mut minus = [[0.0; 3]; 3];
            plus[i][j] = step;
            minus[i][j] = -step;
            let e_plus = strained_energy(surface, positions, cell, plus)?;
            let e_minus = strained_energy(surface, positions, cell, minus)?;
            sigma[i][j] = (e_plus - e_minus) / (2.0 * step * volume);
        }
    }
    if !sigma.iter().flatten().all(|v| v.is_finite()) {
        return Err(SaddleError::NonFinite("finite-difference stress"));
    }
    Ok(sigma)
}

/// Virial `V σ` of a Cauchy stress.
pub fn virial(volume: f64, stress: M3) -> M3 {
    scale(volume, stress)
}

/// Solid-state dimer.
///
/// A frozen mask runs [`MinModeSession`] on the atomic coordinates and
/// [`PointSurface::eval`]. Any free strain runs the same session on the
/// generalized vector of [`JointChart`]. The elastic dipole at a
/// recorded saddle is [`SolidDimer::elastic_dipole`], `P = V σ`.
pub struct SolidDimer {
    inner: MinModeSession,
    chart: Option<JointChart>,
    snapshot: Mutex<Snapshot>,
    positions: Array1<f64>,
    cell: M3,
    stress: Option<M3>,
    volume: f64,
    energy: Option<f64>,
    index: Option<IndexCheck>,
    flat: f64,
    escapes_left: usize,
    escape_step: f64,
    rejected_at: Vec<Array1<f64>>,
    index_calls: usize,
}

#[derive(Clone, Debug)]
struct Snapshot {
    q: Array1<f64>,
    stress: M3,
    energy: f64,
    valid: bool,
}

impl Default for Snapshot {
    fn default() -> Self {
        Self {
            q: Array1::zeros(0),
            stress: [[0.0; 3]; 3],
            energy: 0.0,
            valid: false,
        }
    }
}

impl SolidDimer {
    /// `positions` is 3N Cartesian. `cell` has lattice vectors as rows.
    /// `mode` is either the atomic direction (length 3N) or the full
    /// generalized direction. A frozen cell accepts only the atomic
    /// length and does not rotate the frame. A free cell is turned
    /// lower triangular, and the atomic part of the mode rotates with
    /// the positions.
    pub fn new(
        config: SolidDimerConfig,
        mut positions: Array1<f64>,
        mut cell: M3,
        mut mode: Array1<f64>,
    ) -> Result<Self, SaddleError> {
        if positions.is_empty() || !positions.len().is_multiple_of(3) {
            return Err(SaddleError::Shape(
                "solid-state dimer positions must be 3N".into(),
            ));
        }
        if !positions.iter().all(|v| v.is_finite()) || !cell.iter().flatten().all(|v| v.is_finite())
        {
            return Err(SaddleError::NonFinite("solid-state dimer geometry"));
        }
        if let Some(index) = config.index {
            if index.lowest == 0 || !(index.step.is_finite() && index.step > 0.0) {
                return Err(SaddleError::Invalid(
                    "index check needs a positive difference step and at least one eigenvalue"
                        .into(),
                ));
            }
            if !(index.escape.is_finite() && index.escape >= 0.0) || !index.negative.is_finite() {
                return Err(SaddleError::Invalid(
                    "index check cutoff and escape step must be finite".into(),
                ));
            }
        }
        if !(config.flat.is_finite() && config.flat >= 0.0) {
            return Err(SaddleError::Invalid(
                "flat-saddle length must be finite and non-negative".into(),
            ));
        }
        let index = config.index;
        let flat = config.flat;
        let escapes_left = index.map(|check| check.escapes).unwrap_or(0);
        let escape_step = index.map(|check| check.escape).unwrap_or(0.0);
        let volume = m3_det(cell).abs();
        if not_above(volume, 1e-18) {
            return Err(SaddleError::Invalid(
                "solid-state dimer needs a cell with positive volume".into(),
            ));
        }
        let n = positions.len();
        let frozen = config.mask.is_frozen();
        if frozen {
            if mode.len() != n {
                return Err(SaddleError::Shape(
                    "frozen-cell dimer mode must be the atomic mode".into(),
                ));
            }
            let inner = MinModeSession::new(config.search, positions.clone(), mode)?;
            return Ok(Self {
                inner,
                chart: None,
                snapshot: Mutex::new(Snapshot::default()),
                positions,
                cell,
                stress: None,
                volume,
                energy: None,
                index,
                flat,
                escapes_left,
                escape_step,
                rejected_at: Vec::new(),
                index_calls: 0,
            });
        }
        let mode_slice = mode.as_slice_mut().expect("mode buffer");
        orient_frame(&mut cell, &mut positions, mode_slice)?;
        let chart = JointChart::new(
            cell,
            n / 3,
            config.mask,
            config.external_stress,
            config.weight,
        )?;
        let mode_q = match mode.len() {
            len if len == n => {
                let mut full = Array1::zeros(chart.dimension());
                full.slice_mut(s![..n]).assign(&mode);
                full
            }
            len if len == chart.dimension() => mode,
            _ => {
                return Err(SaddleError::Shape(
                    "solid-state dimer mode must be 3N or the generalized length".into(),
                ));
            }
        };
        let mut q = Array1::zeros(chart.dimension());
        q.slice_mut(s![..n]).assign(&positions);
        let inner = MinModeSession::new(config.search, q, mode_q)?;
        Ok(Self {
            inner,
            chart: Some(chart),
            snapshot: Mutex::new(Snapshot::default()),
            positions,
            cell,
            stress: None,
            volume,
            energy: None,
            index,
            flat,
            escapes_left,
            escape_step,
            rejected_at: Vec::new(),
            index_calls: 0,
        })
    }

    /// True when the cell block is absent and the step is the fixed-cell dimer.
    pub fn is_frozen(&self) -> bool {
        self.chart.is_none()
    }

    /// Jacobian. `None` on a frozen cell.
    pub fn jacobian(&self) -> Option<f64> {
        self.chart.as_ref().map(JointChart::jacobian)
    }

    /// Chart. `None` on a frozen cell.
    pub fn chart(&self) -> Option<&JointChart> {
        self.chart.as_ref()
    }

    /// Generalized coordinate the stepper holds. On a frozen cell this
    /// is the Cartesian position.
    pub fn generalized(&self) -> ArrayView1<'_, f64> {
        self.inner.position()
    }

    /// Cartesian positions.
    pub fn positions(&self) -> ArrayView1<'_, f64> {
        self.positions.view()
    }

    /// Current cell, lattice vectors as rows.
    pub fn cell(&self) -> M3 {
        self.cell
    }

    /// Generalized mode, unit length in the joint space.
    pub fn mode(&self) -> ArrayView1<'_, f64> {
        self.inner.mode()
    }

    /// Curvature along [`Self::mode`].
    pub fn curvature(&self) -> f64 {
        self.inner.curvature()
    }

    /// Enthalpy of the last joint evaluation at the accepted point.
    pub fn energy(&self) -> Option<f64> {
        self.energy
    }

    /// Cell volume.
    pub fn volume(&self) -> f64 {
        self.volume
    }

    /// Cauchy stress last recorded at the accepted point.
    pub fn stress(&self) -> Option<M3> {
        self.stress
    }

    /// Elastic dipole `P = V σ` at the recorded stress.
    pub fn elastic_dipole(&self) -> Option<M3> {
        self.stress.map(|sigma| virial(self.volume, sigma))
    }

    /// One minimum-mode step.
    pub fn step<S: PointSurface>(&mut self, surface: &S) -> Result<SolidDimerReport, SaddleError> {
        if self.chart.is_none() {
            let report = self.inner.step(surface)?;
            self.positions = self.inner.position().to_owned();
            self.stress = None;
            self.energy = None;
            return self.adjudicate(surface, report);
        }
        let chart = self.chart.clone().expect("joint chart");
        let snapshot = std::mem::take(&mut self.snapshot);
        let result = {
            let adapter = JointSurface {
                surface,
                chart: &chart,
                snapshot: &snapshot,
            };
            self.inner.step(&adapter)
        };
        let snap = snapshot
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .clone();
        self.snapshot = snapshot;
        let q = self.inner.position().to_owned();
        let (r, h) = chart.geometry(q.view())?;
        self.positions = r;
        self.cell = h;
        self.volume = m3_det(h).abs();
        let recorded = snap.valid && snap.q == q;
        if recorded {
            self.stress = Some(snap.stress);
            self.energy = Some(snap.energy);
        } else if result.is_ok() {
            // The mode estimate's last probe is not the accepted point.
            // Read the stress there without counting it as a translation.
            let sigma = load_stress(surface, self.positions.view(), self.cell)?;
            let flat = m3_to_row9(self.cell);
            let (energy, _) = surface.eval_in_cell(self.positions.view(), &flat)?;
            self.stress = Some(sigma);
            self.energy = Some(chart.enthalpy(energy, q.view())?);
        }
        let report = result?;
        self.adjudicate_joint(surface, report)
    }

    /// Projected index at the current generalized coordinate.
    ///
    /// Uses the same chart and the same excluded directions as
    /// [`Self::step`]. The search is not moved.
    pub fn index<S: PointSurface>(&self, surface: &S) -> Result<IndexReport, SaddleError> {
        let check = self.index.unwrap_or_else(IndexCheck::first_order);
        let q = self.inner.position().to_owned();
        if let Some(chart) = &self.chart {
            let snapshot = Mutex::new(Snapshot::default());
            let adapter = JointSurface {
                surface,
                chart,
                snapshot: &snapshot,
            };
            index_at(&adapter, q.view(), check)
        } else {
            index_at(surface, q.view(), check)
        }
    }

    /// Loop [`Self::step`] until convergence or `max_steps`.
    pub fn run<S: PointSurface>(
        &mut self,
        surface: &S,
        max_steps: usize,
    ) -> Result<SolidDimerReport, SaddleError> {
        let mut report = self.step(surface)?;
        while report.status == MinModeStatus::Running && report.iteration < max_steps {
            report = self.step(surface)?;
        }
        Ok(report)
    }

    /// Move to a generalized coordinate on the chart this search was
    /// built with.
    ///
    /// A frozen cell takes Cartesian positions. A free cell takes the
    /// joint vector of [`JointChart`]: non-affine positions, then the
    /// free strains as `J ε`, the same length as [`Self::generalized`].
    /// The reference cell, the external stress and `J` stay as they
    /// were at construction, so a later [`Self::step`] is still the
    /// enthalpy of that chart. Passing the current cell back into
    /// [`Self::new`] would start a different chart.
    pub fn set_generalized(&mut self, q: Array1<f64>) -> Result<(), SaddleError> {
        if let Some(chart) = &self.chart {
            let (r, h) = chart.geometry(q.view())?;
            self.inner.set_position(q, None)?;
            self.positions = r;
            self.cell = h;
            self.volume = m3_det(h).abs();
        } else {
            self.inner.set_position(q.clone(), None)?;
            self.positions = q;
        }
        self.stress = None;
        self.energy = None;
        *self
            .snapshot
            .lock()
            .unwrap_or_else(|poison| poison.into_inner()) = Snapshot::default();
        Ok(())
    }

    /// Replace the mode seed. The length is the generalized coordinate,
    /// and the direction is normalised.
    pub fn set_mode(&mut self, mode: Array1<f64>) -> Result<(), SaddleError> {
        self.inner.set_mode(mode)
    }

    /// Cauchy stress at the current geometry.
    ///
    /// Uses [`PointSurface::cauchy_stress`] when the surface supplies
    /// one, otherwise [`finite_difference_cauchy_stress`]. This is an
    /// extra evaluation, not part of [`Self::step`], so a frozen step
    /// stays identical to [`MinModeSession::step`].
    pub fn measure_stress<S: PointSurface>(&mut self, surface: &S) -> Result<M3, SaddleError> {
        let (positions, cell) = if let Some(chart) = &self.chart {
            chart.geometry(self.inner.position())?
        } else {
            (self.positions.clone(), self.cell)
        };
        let sigma = load_stress(surface, positions.view(), cell)?;
        self.positions = positions;
        self.cell = cell;
        self.volume = m3_det(cell).abs();
        self.stress = Some(sigma);
        Ok(sigma)
    }
}

fn map_report(
    report: MinModeReport,
    stress: Option<M3>,
    volume: f64,
    negative_modes: Option<usize>,
    rejected: bool,
    index_evaluations: usize,
    eigenvalues: Vec<f64>,
) -> SolidDimerReport {
    SolidDimerReport {
        status: report.status,
        max_force: report.max_force,
        curvature: report.curvature,
        rotations: report.rotations,
        iteration: report.iteration,
        evaluations: report.evaluations,
        stress,
        volume,
        negative_modes,
        rejected,
        index_evaluations,
        eigenvalues,
    }
}

impl SolidDimer {
    fn adjudicate_joint<S: PointSurface>(
        &mut self,
        surface: &S,
        report: MinModeReport,
    ) -> Result<SolidDimerReport, SaddleError> {
        let chart = self.chart.clone().expect("joint chart");
        let snapshot = Mutex::new(Snapshot::default());
        let adapter = JointSurface {
            surface,
            chart: &chart,
            snapshot: &snapshot,
        };
        self.adjudicate(&adapter, report)
    }

    /// A converged force is not a saddle until the projected index is 1.
    /// A flat mode stays running until the harmonic residual fits in
    /// [`SolidDimerConfig::flat`].
    fn adjudicate<S: PointSurface>(
        &mut self,
        surface: &S,
        mut report: MinModeReport,
    ) -> Result<SolidDimerReport, SaddleError> {
        let mut negative_modes = None;
        let mut rejected = false;
        let mut eigenvalues = Vec::new();
        if report.status == MinModeStatus::Converged && self.flat > 0.0 && report.curvature < -1e-3
        {
            let limit = (self.flat * report.curvature.abs()).max(1e-5);
            if report.max_force > limit && self.inner.force_tol() > limit * (1.0 + 1e-9) {
                self.inner.set_force_tol(limit)?;
                report.status = MinModeStatus::Running;
            }
        }
        if report.status == MinModeStatus::Converged
            && let Some(check) = self.index
        {
            let verdict = index_at(surface, self.inner.position(), check)?;
            report.evaluations += verdict.evaluations;
            self.index_calls += verdict.evaluations;
            if verdict.negative == 1 {
                negative_modes = Some(1);
                eigenvalues = verdict.values;
            } else if verdict.negative > 1
                && self.escapes_left > 0
                && self.leave(surface, &verdict)?
            {
                self.escapes_left -= 1;
                report.status = MinModeStatus::Running;
            } else {
                negative_modes = Some(verdict.negative);
                eigenvalues = verdict.values;
                rejected = true;
            }
        }
        Ok(map_report(
            report,
            self.stress,
            self.volume,
            negative_modes,
            rejected,
            self.index_calls,
            eigenvalues,
        ))
    }

    /// Step off a higher-index point along its softest negative mode.
    ///
    /// The lowest mode stays the seed, so the next translation still
    /// climbs that direction and minimises the rest. Returns false when
    /// neither sign lowers the enthalpy.
    fn leave<S: PointSurface>(
        &mut self,
        surface: &S,
        verdict: &IndexReport,
    ) -> Result<bool, SaddleError> {
        let q = self.inner.position().to_owned();
        self.rejected_at.push(q.clone());
        let soft = verdict.vectors.column(verdict.negative - 1).to_owned();
        let hard = verdict.vectors.column(0).to_owned();
        let (reference, _) = surface.eval(q.view())?;
        let mut sign = 0.0;
        let mut lowest = reference;
        for candidate in [1.0, -1.0] {
            let trial = &q + &(&soft * (candidate * 0.05));
            if !trial.iter().all(|value| value.is_finite()) {
                continue;
            }
            let (energy, _) = surface.eval(trial.view())?;
            if energy < lowest {
                lowest = energy;
                sign = candidate;
            }
        }
        if sign == 0.0 {
            return Ok(false);
        }
        let relaxed =
            walk_soft_positive(surface, &q, &soft, &hard, sign, self.escape_step.max(0.01))?;
        if rms_to_nearest(&relaxed, &self.rejected_at) <= 1e-4 {
            return Ok(false);
        }
        self.inner.reset();
        self.inner.set_position(relaxed, None)?;
        self.inner.set_mode(hard)?;
        self.pull_geometry()?;
        Ok(true)
    }

    fn pull_geometry(&mut self) -> Result<(), SaddleError> {
        if let Some(chart) = &self.chart {
            let q = self.inner.position().to_owned();
            let (positions, cell) = chart.geometry(q.view())?;
            self.positions = positions;
            self.cell = cell;
            self.volume = m3_det(cell).abs();
        } else {
            self.positions = self.inner.position().to_owned();
        }
        self.stress = None;
        self.energy = None;
        *self
            .snapshot
            .lock()
            .unwrap_or_else(|poison| poison.into_inner()) = Snapshot::default();
        Ok(())
    }
}

fn rms_to_nearest(point: &Array1<f64>, previous: &[Array1<f64>]) -> f64 {
    let mut nearest = f64::MAX;
    for earlier in previous {
        let mut sum = 0.0;
        for (left, right) in point.iter().zip(earlier.iter()) {
            let delta = left - right;
            sum += delta * delta;
        }
        let rms = (sum / point.len() as f64).sqrt();
        nearest = nearest.min(rms);
    }
    nearest
}

/// Lanczos with full reorthogonalisation. The Rayleigh-Ritz projection
/// of the actions is diagonalised directly, so the reorthogonalisation
/// coefficients are not dropped. The lowest values are returned once
/// they stop moving, or at a Krylov cap of 64.
fn ritz_lowest<S: PointSurface>(
    surface: &S,
    x: ArrayView1<f64>,
    modes: &[Array1<f64>],
    check: IndexCheck,
    calls: &std::cell::Cell<usize>,
) -> Result<(Vec<f64>, Array2<f64>), SaddleError> {
    let n = x.len();
    let k = check.lowest.min(n);
    let mut seed = Array1::zeros(n);
    for i in 0..n {
        seed[i] = ((i + 1) as f64 * 0.618_033_988_749_895).sin();
    }
    seed = remove_directions(&seed, modes);
    let norm = seed.dot(&seed).sqrt();
    if norm < 1e-14 {
        return Ok((vec![0.0; k], Array2::zeros((n, k))));
    }
    seed /= norm;
    let mut basis = vec![seed];
    let mut actions: Vec<Array1<f64>> = Vec::new();
    let cap = n.min(64.max(k.saturating_mul(4)));
    let mut previous: Option<Vec<f64>> = None;
    let mut kept: Option<(Vec<f64>, Array2<f64>)> = None;
    let mut confirmed = false;
    while basis.len() <= cap {
        let j = basis.len() - 1;
        let hq = projected_action(surface, x, basis[j].view(), modes, check.step, calls)?;
        if hq.len() != n {
            return Err(SaddleError::Shape(
                "index-check Hessian action length changed".into(),
            ));
        }
        let mut w = hq.clone();
        for _ in 0..2 {
            for q in &basis {
                let coeff = w.dot(q);
                w = &w - &(q * coeff);
            }
        }
        actions.push(hq);
        let off = w.dot(&w).sqrt();
        let m = actions.len();
        let chain_done = off < 1e-8 || m == n;
        let ready = chain_done || (m >= k && m.is_multiple_of(4));
        if ready {
            let (values, vectors) = ritz_from_actions(&basis[..m], &actions)?;
            let stable = previous.as_ref().is_some_and(|earlier| {
                values
                    .iter()
                    .zip(earlier.iter())
                    .take(k)
                    .all(|(now, then)| {
                        (now - then).abs() <= 1e-5 * (1.0 + now.abs().max(then.abs()))
                    })
            });
            previous = Some(values.iter().take(k).copied().collect());
            kept = Some((values, vectors));
            // One chain misses a repeated eigenvalue. A fresh direction
            // orthogonal to the basis picks up that copy.
            if m < n
                && (chain_done || (stable && !confirmed))
                && let Some(next) = fresh_direction(&basis[..m], modes, n)
            {
                confirmed = stable;
                basis.truncate(m);
                basis.push(next);
                continue;
            }
            if stable || chain_done || m == n {
                break;
            }
        }
        if m == n {
            break;
        }
        w /= off;
        basis.truncate(m);
        basis.push(w);
    }
    let Some((values, vectors)) = kept else {
        return Err(SaddleError::Solver(
            "index check did not build a Ritz pair".into(),
        ));
    };
    let take = k.min(values.len());
    Ok((
        values.into_iter().take(take).collect(),
        vectors.slice(ndarray::s![.., ..take]).to_owned(),
    ))
}

fn fresh_direction(basis: &[Array1<f64>], modes: &[Array1<f64>], n: usize) -> Option<Array1<f64>> {
    for seed in 0..n {
        let mut w = Array1::zeros(n);
        w[seed] = 1.0;
        w = remove_directions(&w, modes);
        for q in basis {
            let coeff = w.dot(q);
            w = &w - &(q * coeff);
        }
        let norm = w.dot(&w).sqrt();
        if norm > 1e-8 {
            return Some(w / norm);
        }
    }
    None
}

fn ritz_from_actions(
    basis: &[Array1<f64>],
    actions: &[Array1<f64>],
) -> Result<(Vec<f64>, Array2<f64>), SaddleError> {
    let m = actions.len().min(basis.len());
    let mut projected = Array2::zeros((m, m));
    for j in 0..m {
        for i in 0..m {
            projected[(i, j)] = basis[i].dot(&actions[j]);
        }
    }
    for i in 0..m {
        for j in (i + 1)..m {
            let mean = 0.5 * (projected[(i, j)] + projected[(j, i)]);
            projected[(i, j)] = mean;
            projected[(j, i)] = mean;
        }
    }
    let (vals, vecs) = crate::eigensolve::exact_eigh(projected.view())?;
    let n = basis.first().map(|column| column.len()).unwrap_or(0);
    let mut vectors = Array2::zeros((n, m));
    for column in 0..m {
        for (row_basis, q) in basis.iter().take(m).enumerate() {
            let weight = vecs[(row_basis, column)];
            for (i, value) in q.iter().enumerate() {
                vectors[(i, column)] += weight * value;
            }
        }
        let mut norm = 0.0_f64;
        for row in vectors.column(column) {
            norm += row * row;
        }
        norm = norm.sqrt();
        if norm > 0.0 {
            for row in vectors.column_mut(column) {
                *row /= norm;
            }
        }
    }
    Ok((vals.to_vec(), vectors))
}

/// Step along `soft` until its Rayleigh quotient is no longer negative.
///
/// The probe that chose `sign` is the downhill side. The hard mode is
/// held fixed so the walk stays on that ridge. A 0.02 Å step is past
/// the cubic scale of the copper soft mode, where one side of a 0.05 Å
/// probe rises.
fn walk_soft_positive<S: PointSurface>(
    surface: &S,
    start: &Array1<f64>,
    soft: &Array1<f64>,
    hard: &Array1<f64>,
    sign: f64,
    step: f64,
) -> Result<Array1<f64>, SaddleError> {
    let mut q = start.clone();
    for _ in 0..12 {
        q = &q + &(soft * (sign * step));
        let drift = (&q - start).dot(hard);
        q = &q - &(hard * drift);
        if rayleigh(surface, &q, soft)? >= -1e-4 {
            break;
        }
    }
    Ok(q)
}

fn rayleigh<S: PointSurface>(
    surface: &S,
    q: &Array1<f64>,
    direction: &Array1<f64>,
) -> Result<f64, SaddleError> {
    let step = 1e-4;
    let plus = q + &(direction * step);
    let minus = q - &(direction * step);
    let (_, g_plus) = surface.eval(plus.view())?;
    let (_, g_minus) = surface.eval(minus.view())?;
    if g_plus.len() != q.len() || g_minus.len() != q.len() {
        return Err(SaddleError::Shape(
            "index-check Rayleigh gradient length changed".into(),
        ));
    }
    let mut slope = 0.0;
    for i in 0..q.len() {
        slope += direction[i] * (g_plus[i] - g_minus[i]);
    }
    Ok(slope / (2.0 * step))
}

fn orthonormal_excluded<S: PointSurface>(
    surface: &S,
    x: ArrayView1<f64>,
) -> Result<Vec<Array1<f64>>, SaddleError> {
    let modes = surface.excluded_modes(x)?;
    if modes.ncols() != x.len() || !modes.iter().all(|value| value.is_finite()) {
        return Err(SaddleError::Shape("index-check excluded directions".into()));
    }
    let mut basis = Vec::new();
    for row in modes.rows() {
        let mut direction = row.to_owned();
        for earlier in &basis {
            let scale = direction.dot(earlier);
            direction = direction - earlier * scale;
        }
        let norm = direction.dot(&direction).sqrt();
        if norm > 1e-10 {
            direction /= norm;
            basis.push(direction);
        }
    }
    Ok(basis)
}

fn projected_action<S: PointSurface>(
    surface: &S,
    x: ArrayView1<f64>,
    direction: ArrayView1<f64>,
    modes: &[Array1<f64>],
    step: f64,
    calls: &std::cell::Cell<usize>,
) -> Result<Array1<f64>, SaddleError> {
    let direction = remove_directions(&direction.to_owned(), modes);
    let norm = direction.dot(&direction).sqrt();
    if norm < 1e-14 {
        return Ok(Array1::zeros(x.len()));
    }
    let unit = &direction / norm;
    let plus = x.to_owned() + &(&unit * step);
    let minus = x.to_owned() - &(&unit * step);
    let (_, g_plus) = surface.eval(plus.view())?;
    let (_, g_minus) = surface.eval(minus.view())?;
    calls.set(calls.get() + 2);
    if g_plus.len() != x.len() || g_minus.len() != x.len() {
        return Err(SaddleError::Shape(
            "index-check gradient length changed".into(),
        ));
    }
    if !g_plus.iter().all(|value| value.is_finite())
        || !g_minus.iter().all(|value| value.is_finite())
    {
        return Err(SaddleError::NonFinite("index-check gradient"));
    }
    let mut action = (g_plus - g_minus) * (norm / (2.0 * step));
    action = remove_directions(&action, modes);
    Ok(action)
}

fn remove_directions(vector: &Array1<f64>, modes: &[Array1<f64>]) -> Array1<f64> {
    let mut out = vector.clone();
    for _ in 0..2 {
        for mode in modes {
            let scale = out.dot(mode);
            out = out - mode * scale;
        }
    }
    out
}

struct JointSurface<'a, S> {
    surface: &'a S,
    chart: &'a JointChart,
    snapshot: &'a Mutex<Snapshot>,
}

impl<S: PointSurface> PointSurface for JointSurface<'_, S> {
    fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (r, h) = self.chart.geometry(q)?;
        let flat = m3_to_row9(h);
        let (energy, g_cart) = self.surface.eval_in_cell(r.view(), &flat)?;
        if g_cart.len() != r.len() {
            return Err(SaddleError::Shape(
                "solid-state surface gradient length".into(),
            ));
        }
        let sigma = load_stress(self.surface, r.view(), h)?;
        let enthalpy = self.chart.enthalpy(energy, q)?;
        let gradient = self.chart.gradient(q, g_cart.view(), sigma)?;
        if let Ok(mut slot) = self.snapshot.lock() {
            *slot = Snapshot {
                q: q.to_owned(),
                stress: sigma,
                energy: enthalpy,
                valid: true,
            };
        }
        Ok((enthalpy, gradient))
    }

    fn excluded_modes(&self, q: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
        let (r, _) = self.chart.geometry(q)?;
        let cart = self.surface.excluded_modes(r.view())?;
        if cart.nrows() == 0 {
            return Ok(Array2::zeros((0, q.len())));
        }
        if cart.ncols() != r.len() || !cart.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::Shape("solid-state excluded directions".into()));
        }
        // r = r0 (I + ε), so a Cartesian displacement dr is dr0 = dr (I + ε)^{-1}.
        let eps = self.chart.strain(q)?;
        let minv = m3_inv(deform(eps))?;
        let mut out = Array2::zeros((cart.nrows(), q.len()));
        let atoms = r.len() / 3;
        for row in 0..cart.nrows() {
            for atom in 0..atoms {
                let dr = [
                    cart[(row, 3 * atom)],
                    cart[(row, 3 * atom + 1)],
                    cart[(row, 3 * atom + 2)],
                ];
                for a in 0..3 {
                    let mut acc = 0.0;
                    for b in 0..3 {
                        acc += dr[b] * minv[b][a];
                    }
                    out[(row, 3 * atom + a)] = acc;
                }
            }
        }
        Ok(out)
    }
}

fn load_stress<S: PointSurface>(
    surface: &S,
    positions: ArrayView1<f64>,
    cell: M3,
) -> Result<M3, SaddleError> {
    let flat = m3_to_row9(cell);
    match surface.cauchy_stress(positions, &flat)? {
        Some(packed) => {
            if !packed.iter().all(|v| v.is_finite()) {
                return Err(SaddleError::NonFinite("solid-state stress"));
            }
            Ok(m3_from_row9(packed))
        }
        None => finite_difference_cauchy_stress(surface, positions, cell, 1e-5),
    }
}

fn strained_energy<S: PointSurface>(
    surface: &S,
    positions: ArrayView1<f64>,
    cell: M3,
    eps: M3,
) -> Result<f64, SaddleError> {
    let deform = deform(eps);
    let h = m3_mul(cell, deform);
    let mut r = Array1::zeros(positions.len());
    for atom in 0..positions.len() / 3 {
        let row = [
            positions[3 * atom],
            positions[3 * atom + 1],
            positions[3 * atom + 2],
        ];
        let out = row_mul(row, deform);
        for c in 0..3 {
            r[3 * atom + c] = out[c];
        }
    }
    let (energy, _) = surface.eval_in_cell(r.view(), &m3_to_row9(h))?;
    if !energy.is_finite() {
        return Err(SaddleError::NonFinite("strained energy"));
    }
    Ok(energy)
}

fn deform(eps: M3) -> M3 {
    let mut m = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = eps[i][j];
        }
        m[i][i] += 1.0;
    }
    m
}

fn lower_triangular(cell: &M3) -> bool {
    cell[0][1].abs() <= 1e-8 && cell[0][2].abs() <= 1e-8 && cell[1][2].abs() <= 1e-8
}

fn fold_lower(mut stress: M3) -> M3 {
    let upper = [
        (0, 1, stress[0][1]),
        (0, 2, stress[0][2]),
        (1, 2, stress[1][2]),
    ];
    for (i, j, value) in upper {
        if stress[j][i] == 0.0 {
            stress[j][i] = value;
        }
        stress[i][j] = 0.0;
    }
    stress
}

fn row_mul(row: [f64; 3], m: M3) -> [f64; 3] {
    [
        row[0] * m[0][0] + row[1] * m[1][0] + row[2] * m[2][0],
        row[0] * m[0][1] + row[1] * m[1][1] + row[2] * m[2][1],
        row[0] * m[0][2] + row[1] * m[1][2] + row[2] * m[2][2],
    ]
}

fn transpose(a: M3) -> M3 {
    [
        [a[0][0], a[1][0], a[2][0]],
        [a[0][1], a[1][1], a[2][1]],
        [a[0][2], a[1][2], a[2][2]],
    ]
}

fn scale(s: f64, a: M3) -> M3 {
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = s * a[i][j];
        }
    }
    c
}

fn add(a: M3, b: M3) -> M3 {
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = a[i][j] + b[i][j];
        }
    }
    c
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm3(a: [f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Same frame as [`orient_lower_triangular`], applied to the atomic mode too.
fn orient_frame(
    cell: &mut M3,
    positions: &mut Array1<f64>,
    mode: &mut [f64],
) -> Result<(), SaddleError> {
    let mut check_cell = *cell;
    let mut check_pos = positions.clone();
    orient_lower_triangular(&mut check_cell, &mut check_pos)?;
    let a = cell[0];
    let b = cell[1];
    let na = norm3(a);
    let e1 = [a[0] / na, a[1] / na, a[2] / na];
    let along = dot3(b, e1);
    let b_perp = [
        b[0] - along * e1[0],
        b[1] - along * e1[1],
        b[2] - along * e1[2],
    ];
    let nb = norm3(b_perp);
    let e2 = [b_perp[0] / nb, b_perp[1] / nb, b_perp[2] / nb];
    let e3 = cross3(e1, e2);
    let rotation = [
        [e1[0], e2[0], e3[0]],
        [e1[1], e2[1], e3[1]],
        [e1[2], e2[2], e3[2]],
    ];
    let n = positions.len();
    apply_rotation(positions, rotation);
    if mode.len() >= n {
        apply_rotation_prefix(mode, n, rotation);
    }
    *cell = check_cell;
    if positions
        .iter()
        .zip(check_pos.iter())
        .any(|(left, right)| (left - right).abs() > 1e-8)
    {
        return Err(SaddleError::Invalid(
            "solid-state orientation disagreed with the band frame".into(),
        ));
    }
    Ok(())
}

fn apply_rotation(positions: &mut Array1<f64>, rotation: M3) {
    for atom in 0..positions.len() / 3 {
        let row = [
            positions[3 * atom],
            positions[3 * atom + 1],
            positions[3 * atom + 2],
        ];
        let out = row_mul(row, rotation);
        for c in 0..3 {
            positions[3 * atom + c] = out[c];
        }
    }
}

fn apply_rotation_prefix(mode: &mut [f64], n: usize, rotation: M3) {
    for atom in 0..(n / 3) {
        let row = [mode[3 * atom], mode[3 * atom + 1], mode[3 * atom + 2]];
        let out = row_mul(row, rotation);
        for c in 0..3 {
            mode[3 * atom + c] = out[c];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{ArrayView1, array};
    use rgmin::Method;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn diagonal(x: f64, y: f64, z: f64) -> M3 {
        [[x, 0.0, 0.0], [0.0, y, 0.0], [0.0, 0.0, z]]
    }

    struct Well;

    impl PointSurface for Well {
        fn eval(&self, r: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let (x, y, z) = (r[0], r[1], r[2]);
            let energy = (x * x - 1.0).powi(2) + 2.0 * y * y + 2.0 * z * z;
            Ok((energy, array![4.0 * x * (x * x - 1.0), 4.0 * y, 4.0 * z]))
        }
    }

    struct Split {
        evals: AtomicUsize,
        cells: AtomicUsize,
    }

    impl PointSurface for Split {
        fn eval(&self, r: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            self.evals.fetch_add(1, Ordering::Relaxed);
            Well.eval(r)
        }

        fn eval_in_cell(
            &self,
            r: ArrayView1<f64>,
            cell: &[f64; 9],
        ) -> Result<(f64, Array1<f64>), SaddleError> {
            let _ = cell;
            self.cells.fetch_add(1, Ordering::Relaxed);
            self.eval(r)
        }
    }

    #[test]
    fn frozen_cell_matches_the_fixed_cell_dimer() {
        let start = array![0.4, 0.15, -0.05];
        let mode = array![0.2, 1.0, -0.4];
        let cell = diagonal(6.0, 5.0, 7.0);
        for method in [
            Method::Fire {
                kind: rgmin::FireKind::V2,
            },
            Method::Lbfgs { memory: 7 },
        ] {
            let search = MinModeConfig {
                method: method.clone(),
                force_tol: 1e-12,
                max_move: 0.08,
                max_rotations: 8,
                ..MinModeConfig::default()
            };
            let mut plain =
                MinModeSession::new(search.clone(), start.clone(), mode.clone()).unwrap();
            let mut solid = SolidDimer::new(
                SolidDimerConfig::frozen(search),
                start.clone(),
                cell,
                mode.clone(),
            )
            .unwrap();
            assert!(solid.is_frozen());
            assert!(solid.jacobian().is_none());
            let surface = Split {
                evals: AtomicUsize::new(0),
                cells: AtomicUsize::new(0),
            };
            for _ in 0..4 {
                let left = plain.step(&Well).unwrap();
                let right = solid.step(&surface).unwrap();
                assert_eq!(left.evaluations, right.evaluations);
                assert_eq!(left.curvature.to_bits(), right.curvature.to_bits());
                assert_eq!(left.max_force.to_bits(), right.max_force.to_bits());
                assert_eq!(left.status, right.status);
                for (a, b) in plain.position().iter().zip(solid.positions().iter()) {
                    assert_eq!(a, b);
                }
                for (a, b) in plain.mode().iter().zip(solid.mode().iter()) {
                    assert_eq!(a, b);
                }
            }
            assert_eq!(surface.cells.load(Ordering::Relaxed), 0);
            assert!(surface.evals.load(Ordering::Relaxed) > 0);
        }
    }

    /// E = (r_x - 0.2)^2 + (h_00 - 7)^2. The stress is the right-strain derivative.
    struct Shifted;

    impl PointSurface for Shifted {
        fn eval(&self, r: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            self.eval_in_cell(r, &[7.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 4.0])
        }

        fn eval_in_cell(
            &self,
            r: ArrayView1<f64>,
            cell: &[f64; 9],
        ) -> Result<(f64, Array1<f64>), SaddleError> {
            let dx = r[0] - 0.2;
            let dh = cell[0] - 7.0;
            let mut g = Array1::zeros(r.len());
            g[0] = 2.0 * dx;
            Ok((dx * dx + dh * dh, g))
        }
    }

    fn analytic_shift_stress(r: ArrayView1<f64>, cell: M3) -> M3 {
        let g0 = 2.0 * (r[0] - 0.2);
        let dh = 2.0 * (cell[0][0] - 7.0);
        let volume = m3_det(cell).abs();
        let mut sigma = [[0.0; 3]; 3];
        for (p, row) in sigma.iter_mut().enumerate() {
            for (q, entry) in row.iter_mut().enumerate() {
                let mut deriv = if q == 0 { g0 * r[p] } else { 0.0 };
                if q == 0 {
                    deriv += dh * cell[0][p];
                }
                *entry = deriv / volume;
            }
        }
        sigma
    }

    #[test]
    fn finite_difference_stress_matches_the_right_strain_derivative() {
        let cell = [[5.5, 0.0, 0.0], [0.4, 4.2, 0.0], [0.1, -0.2, 6.0]];
        let positions = array![0.7, -0.3, 0.2];
        let fd = finite_difference_cauchy_stress(&Shifted, positions.view(), cell, 1e-6).unwrap();
        let analytic = analytic_shift_stress(positions.view(), cell);
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(fd[i][j], analytic[i][j], epsilon = 1e-8);
            }
        }
        let dipole = virial(m3_det(cell).abs(), fd);
        assert_abs_diff_eq!(
            dipole[0][0],
            m3_det(cell).abs() * analytic[0][0],
            epsilon = 1e-8
        );
    }

    #[test]
    fn generalized_gradient_matches_a_finite_difference_of_the_enthalpy() {
        let cell = diagonal(4.0, 5.0, 6.0);
        let mut external = [[0.0; 3]; 3];
        external[0][0] = 0.15;
        external[1][0] = -0.04;
        let chart = JointChart::new(cell, 1, CellMask::all(), external, 1.0).unwrap();
        assert_abs_diff_eq!(
            chart.jacobian(),
            (4.0 * 5.0 * 6.0_f64).cbrt(),
            epsilon = 1e-12
        );
        let mut q = Array1::zeros(chart.dimension());
        q[0] = 0.55;
        q[1] = -0.2;
        q[2] = 0.15;
        // Finite right strain, so the chain rule is not the identity.
        q[3] = 0.12 * chart.jacobian();
        q[7] = -0.05 * chart.jacobian();
        let (r, h) = chart.geometry(q.view()).unwrap();
        let (energy, g_cart) = Shifted.eval_in_cell(r.view(), &m3_to_row9(h)).unwrap();
        let sigma = finite_difference_cauchy_stress(&Shifted, r.view(), h, 1e-6).unwrap();
        let gradient = chart.gradient(q.view(), g_cart.view(), sigma).unwrap();
        let enthalpy = chart.enthalpy(energy, q.view()).unwrap();
        let step = 1e-6;
        for k in 0..q.len() {
            let mut plus = q.clone();
            let mut minus = q.clone();
            plus[k] += step;
            minus[k] -= step;
            let e_plus = enthalpy_of(&chart, plus.view());
            let e_minus = enthalpy_of(&chart, minus.view());
            let fd = (e_plus - e_minus) / (2.0 * step);
            assert_abs_diff_eq!(gradient[k], fd, epsilon = 1e-6);
        }
        let _ = enthalpy;
    }

    fn enthalpy_of(chart: &JointChart, q: ArrayView1<f64>) -> f64 {
        let (r, h) = chart.geometry(q).unwrap();
        let (energy, _) = Shifted.eval_in_cell(r.view(), &m3_to_row9(h)).unwrap();
        chart.enthalpy(energy, q).unwrap()
    }

    /// Quadratic saddle in x, wells in y and z, and a cell term on h_00.
    struct QuadraticCell;

    impl PointSurface for QuadraticCell {
        fn eval(&self, r: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            self.eval_in_cell(r, &[4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.0])
        }

        fn eval_in_cell(
            &self,
            r: ArrayView1<f64>,
            cell: &[f64; 9],
        ) -> Result<(f64, Array1<f64>), SaddleError> {
            let energy = -0.5 * r[0] * r[0]
                + 2.0 * r[1] * r[1]
                + 2.0 * r[2] * r[2]
                + 0.5 * (cell[0] - 4.0) * (cell[0] - 4.0);
            Ok((energy, array![-r[0], 4.0 * r[1], 4.0 * r[2]]))
        }

        fn cauchy_stress(
            &self,
            r: ArrayView1<f64>,
            cell: &[f64; 9],
        ) -> Result<Option<[f64; 9]>, SaddleError> {
            let h = m3_from_row9(*cell);
            let volume = m3_det(h).abs();
            let g = [-r[0], 4.0 * r[1], 4.0 * r[2]];
            let dh = h[0][0] - 4.0;
            let mut sigma = [0.0; 9];
            for (p, &rp) in r.iter().take(3).enumerate() {
                for (q, &gq) in g.iter().enumerate() {
                    let mut deriv = gq * rp;
                    if q == 0 {
                        deriv += dh * h[0][p];
                    }
                    sigma[3 * p + q] = deriv / volume;
                }
            }
            Ok(Some(sigma))
        }
    }

    #[test]
    fn joint_curvature_matches_the_second_difference() {
        let search = MinModeConfig {
            method: Method::Lbfgs { memory: 8 },
            force_tol: 1e-9,
            max_move: 0.2,
            dr: 1e-5,
            rotation_tol: 1e-8,
            max_rotations: 12,
            ..MinModeConfig::default()
        };
        let mut dimer = SolidDimer::new(
            SolidDimerConfig::new(search),
            array![0.25, 0.04, -0.03],
            diagonal(4.0, 4.0, 4.0),
            array![1.0, 0.2, 0.0],
        )
        .unwrap();
        let report = dimer.run(&QuadraticCell, 40).unwrap();
        assert_eq!(report.status, MinModeStatus::Converged);
        assert!(report.curvature < 0.0);
        for value in dimer.positions().iter() {
            assert!(value.abs() < 1e-6, "position {value}");
        }
        let q = dimer.generalized().to_owned();
        let mode = dimer.mode().to_owned();
        let step = 1e-5;
        let plus = &q + &(&mode * step);
        let minus = &q - &(&mode * step);
        let second = (enthalpy_quadratic(plus.view()) + enthalpy_quadratic(minus.view())
            - 2.0 * enthalpy_quadratic(q.view()))
            / (step * step);
        assert_abs_diff_eq!(dimer.curvature(), second, epsilon = 1e-4);
        assert_abs_diff_eq!(dimer.curvature(), -1.0, epsilon = 1e-3);
        let stress = dimer.stress().expect("stress at the saddle");
        let dipole = dimer.elastic_dipole().unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(dipole[i][j], dimer.volume() * stress[i][j], epsilon = 1e-12);
            }
        }
    }

    fn enthalpy_quadratic(q: ArrayView1<f64>) -> f64 {
        let chart = JointChart::new(
            diagonal(4.0, 4.0, 4.0),
            1,
            CellMask::all(),
            [[0.0; 3]; 3],
            1.0,
        )
        .unwrap();
        let (r, h) = chart.geometry(q).unwrap();
        let (energy, _) = QuadraticCell
            .eval_in_cell(r.view(), &m3_to_row9(h))
            .unwrap();
        energy
    }

    #[test]
    fn orientation_matches_the_band_frame_and_rotates_the_mode() {
        let mut cell = [[3.0, 0.4, 0.2], [1.5, 4.0, -0.3], [0.2, 0.5, 5.0]];
        let mut positions = array![0.4, -0.2, 0.7, 1.0, 0.3, -0.1];
        let mut mode = array![0.2, 0.5, -0.1, -0.4, 0.3, 0.8];
        let mut band_cell = cell;
        let mut band_pos = positions.clone();
        orient_lower_triangular(&mut band_cell, &mut band_pos).unwrap();
        orient_frame(
            &mut cell,
            &mut positions,
            mode.as_slice_mut().expect("mode buffer"),
        )
        .unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(cell[i][j], band_cell[i][j], epsilon = 1e-12);
            }
        }
        for (left, right) in positions.iter().zip(band_pos.iter()) {
            assert_abs_diff_eq!(left, right, epsilon = 1e-12);
        }
        assert!(cell[0][1].abs() < 1e-12);
        let before = array![0.2, 0.5, -0.1, -0.4, 0.3, 0.8];
        assert_abs_diff_eq!(mode.dot(&mode), before.dot(&before), epsilon = 1e-12);
    }

    /// `E = -x² - y² + z²` is index 2 at the origin. `E = -x² + y² + z²` is index 1.
    struct IndexSurface {
        double: bool,
    }

    impl PointSurface for IndexSurface {
        fn eval(&self, r: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let (x, y, z) = (r[0], r[1], r[2]);
            let y_sign = if self.double { -1.0 } else { 1.0 };
            let energy = -x * x + y_sign * y * y + z * z;
            Ok((energy, array![-2.0 * x, 2.0 * y_sign * y, 2.0 * z]))
        }
    }

    fn placed(escapes: usize) -> SolidDimer {
        let search = MinModeConfig {
            force_tol: 1e-8,
            dr: 1e-4,
            rotation_tol: 1e-8,
            max_rotations: 8,
            max_move: 0.05,
            ..MinModeConfig::default()
        };
        SolidDimer::new(
            SolidDimerConfig::frozen(search).with_index_check(
                IndexCheck::first_order()
                    .with_escapes(escapes)
                    .with_escape(0.05),
            ),
            array![0.0, 0.0, 0.0],
            diagonal(5.0, 5.0, 5.0),
            array![1.0, 0.2, 0.0],
        )
        .unwrap()
    }

    #[test]
    fn index_check_rejects_a_double_index_point_and_accepts_a_saddle() {
        let double = IndexSurface { double: true };
        let saddle = IndexSurface { double: false };
        let origin = array![0.0, 0.0, 0.0];
        let rejected = index_at(&double, origin.view(), IndexCheck::first_order()).unwrap();
        let accepted = index_at(&saddle, origin.view(), IndexCheck::first_order()).unwrap();
        assert_eq!(rejected.negative, 2, "{:?}", rejected.values);
        assert!(!rejected.is_first_order());
        assert_eq!(accepted.negative, 1, "{:?}", accepted.values);
        assert!(accepted.is_first_order());
        assert!(rejected.values[0] < -1.5 && rejected.values[1] < -1.5);
        assert!(accepted.values[0] < -1.5 && accepted.values[1] > 1.5);

        let mut stayed = placed(0);
        let report = stayed.step(&double).unwrap();
        assert_eq!(report.status, MinModeStatus::Converged);
        assert!(report.rejected);
        assert_eq!(report.negative_modes, Some(2));
        assert!(stayed.positions().iter().all(|value| value.abs() < 1e-8));

        let mut moved = placed(4);
        let left = moved.step(&double).unwrap();
        assert_eq!(left.status, MinModeStatus::Running);
        assert!(!left.rejected);
        assert!(moved.positions().dot(&moved.positions()) > 1e-6);

        let mut kept = placed(0);
        let ok = kept.run(&saddle, 4).unwrap();
        assert_eq!(ok.status, MinModeStatus::Converged);
        assert!(!ok.rejected);
        assert_eq!(ok.negative_modes, Some(1));
    }

    /// A negative eigenvalue along a rigid translation is not an index.
    struct Translated {
        negative_relative: bool,
    }

    impl PointSurface for Translated {
        fn eval(&self, r: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let s2 = std::f64::consts::SQRT_2;
            let cx = (r[0] + r[3]) / s2;
            let ux = (r[0] - r[3]) / s2;
            let uy = (r[1] - r[4]) / s2;
            let sign = if self.negative_relative { -1.0 } else { 1.0 };
            let energy = sign * 0.5 * ux * ux - 0.5 * cx * cx + 0.5 * uy * uy;
            let d_cx = -cx;
            let d_ux = sign * ux;
            let d_uy = uy;
            let mut g = Array1::zeros(6);
            g[0] = (d_cx + d_ux) / s2;
            g[3] = (d_cx - d_ux) / s2;
            g[1] = d_uy / s2;
            g[4] = -d_uy / s2;
            Ok((energy, g))
        }

        fn excluded_modes(&self, x: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
            let mut modes = Array2::zeros((3, x.len()));
            for atom in 0..2 {
                for axis in 0..3 {
                    modes[(axis, 3 * atom + axis)] = 1.0;
                }
            }
            Ok(modes)
        }
    }

    #[test]
    fn index_check_does_not_count_a_rigid_translation() {
        let origin = Array1::zeros(6);
        let raw = Translated {
            negative_relative: false,
        };
        let projected = index_at(&raw, origin.view(), IndexCheck::first_order()).unwrap();
        assert_eq!(
            projected.negative, 0,
            "translation removed: {:?}",
            projected.values
        );
        struct Unprojected;
        impl PointSurface for Unprojected {
            fn eval(&self, r: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
                Translated {
                    negative_relative: false,
                }
                .eval(r)
            }
        }
        let bare = index_at(&Unprojected, origin.view(), IndexCheck::first_order()).unwrap();
        assert_eq!(bare.negative, 1, "raw translation: {:?}", bare.values);

        let saddle = Translated {
            negative_relative: true,
        };
        let one = index_at(&saddle, origin.view(), IndexCheck::first_order()).unwrap();
        assert_eq!(one.negative, 1, "{:?}", one.values);
        assert!(one.is_first_order());
    }

    #[test]
    fn index_check_accepts_the_quadratic_cell_saddle() {
        let search = MinModeConfig {
            method: Method::Lbfgs { memory: 8 },
            force_tol: 1e-9,
            max_move: 0.2,
            dr: 1e-5,
            rotation_tol: 1e-8,
            max_rotations: 12,
            ..MinModeConfig::default()
        };
        let mut dimer = SolidDimer::new(
            SolidDimerConfig::new(search).with_index_check(IndexCheck::first_order()),
            array![0.25, 0.04, -0.03],
            diagonal(4.0, 4.0, 4.0),
            array![1.0, 0.2, 0.0],
        )
        .unwrap();
        let report = dimer.run(&QuadraticCell, 40).unwrap();
        assert_eq!(report.status, MinModeStatus::Converged);
        assert!(!report.rejected, "{:?}", report.negative_modes);
        assert_eq!(report.negative_modes, Some(1));
        assert!(report.curvature < 0.0);
        for value in dimer.positions().iter() {
            assert!(value.abs() < 1e-6, "position {value}");
        }
    }
}
