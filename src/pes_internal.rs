//! Sella InternalPES / CellCartesianPES / CellInternalPES wrappers.
//!
//! [`InternalPes`] is a [`CartesianPes`] plus an internals chart
//! ([`Constraints`] equalities as coordinates). Wilson `B = dc/dx`.
//! An internals increment `dq` maps to Cartesian by the min-norm
//! solve `B dx = dq`, then [`CartesianPes::kick`].
//!
//! [`CellCartesianPes`] and [`CellInternalPes`] carry a
//! [`linkcell::Cell`]. Band / IRC MIC is still [`crate::mic`]; the
//! cell here is the PES wrapper Sella hangs on periodic systems.

use ndarray::{s, Array1, ArrayView1};
use rgmin::BfgsModel;

use crate::constraints::Constraints;
use crate::error::SaddleError;
use crate::mic::Cell;
use crate::minmode::PointSurface;
use crate::pes::{CartesianPes, HessUpdate};

/// Sella `InternalPES`: Cartesian geometry, internals chart, internals Hessian.
pub struct InternalPes {
    cart: CartesianPes,
    chart: Constraints,
    hess: BfgsModel,
    update: HessUpdate,
}

impl InternalPes {
    /// `x` is 3N Cartesian; `masses` is length N; `chart` is the internals.
    pub fn new(
        x: Array1<f64>,
        masses: Array1<f64>,
        chart: Constraints,
    ) -> Result<Self, SaddleError> {
        if chart.n_atoms() * 3 != x.len() {
            return Err(SaddleError::Shape(
                "internals chart atom count must match the 3N frame".into(),
            ));
        }
        if chart.equalities().is_empty() {
            return Err(SaddleError::Shape("internals chart is empty".into()));
        }
        let nint = chart.equalities().len();
        let mut pes = Self {
            cart: CartesianPes::new(x, masses)?,
            chart,
            hess: BfgsModel::identity(nint),
            update: HessUpdate::Bfgs,
        };
        pes.guess_hessian();
        Ok(pes)
    }

    /// Sella `Internals.guess_hessian` diagonal: translations 70,
    /// bonds 0.5, angles 0.2, dihedrals 0.1, rotations 1.
    pub fn guess_hessian(&mut self) {
        let nint = self.chart.equalities().len();
        if nint == 0 {
            return;
        }
        self.hess.forget();
        for (i, eq) in self.chart.equalities().iter().enumerate() {
            let lam = match eq {
                crate::constraints::Equality::Translation { .. } => 70.0,
                crate::constraints::Equality::Bond { .. } => 0.5,
                crate::constraints::Equality::Angle { .. } => 0.2,
                crate::constraints::Equality::Dihedral { .. } => 0.1,
                crate::constraints::Equality::Rotation { .. } => 1.0,
                crate::constraints::Equality::Displacement { .. } => 1.0,
            };
            let mut u = Array1::zeros(nint);
            u[i] = 1.0;
            self.hess.seed_mode(&u, lam);
        }
    }

    pub fn set_update(&mut self, update: HessUpdate) {
        self.update = update;
        self.cart.set_update(update);
    }

    /// Internals BFGS model (Sella `InternalPES.H`).
    pub fn hessian(&self) -> &BfgsModel {
        &self.hess
    }

    /// Number of internals (chart equalities).
    pub fn n_int(&self) -> usize {
        self.chart.equalities().len()
    }

    pub fn position(&self) -> ArrayView1<'_, f64> {
        self.cart.position()
    }

    pub fn chart(&self) -> &Constraints {
        &self.chart
    }

    pub fn cartesian(&self) -> &CartesianPes {
        &self.cart
    }

    /// Internals values at the current geometry.
    pub fn internals(&self) -> Result<Array1<f64>, SaddleError> {
        self.chart.values(self.cart.position())
    }

    /// Wilson B, `nint x 3N`.
    pub fn b_matrix(&self) -> Result<ndarray::Array2<f64>, SaddleError> {
        self.chart.jacobian(self.cart.position())
    }

    /// Internals gradient `B^{+T} g_cart` via the same min-norm solve
    /// as [`Constraints::scons`]: `(B B^T) y = B g`, so `y = B^{+T} g`.
    pub fn internals_grad(&self, g_cart: ArrayView1<f64>) -> Result<Array1<f64>, SaddleError> {
        let b = self.b_matrix()?;
        internals_grad_from_b(&b, g_cart)
    }

    /// Map `dq` to Cartesian by `B dx = dq` and kick the Cartesian PES.
    ///
    /// The internals Hessian stores `(dq, dg_int)` (Sella
    /// `InternalPES.kick`). The Cartesian model is updated with the
    /// matching `(dx, dg_cart)`.
    pub fn kick<S: PointSurface>(
        &mut self,
        surface: &S,
        dq: ArrayView1<f64>,
    ) -> Result<(f64, Array1<f64>), SaddleError> {
        let b = self.b_matrix()?;
        if dq.len() != b.nrows() {
            return Err(SaddleError::Shape(
                "internals increment must match the chart".into(),
            ));
        }
        let (_, g0) = surface.eval(self.cart.position())?;
        let g0_int = internals_grad_from_b(&b, g0.view())?;
        let dx = min_norm_dx(&b, &dq.to_owned());
        let (e, g1) = self.cart.kick_from(surface, dx.view(), g0.view())?;
        let g1_int = self.internals_grad(g1.view())?;
        let y = &g1_int - &g0_int;
        let s = dq.to_owned();
        match self.update {
            HessUpdate::Bfgs => self.hess.update(&s, &y),
            HessUpdate::TsBfgs => self.hess.update_ts(&s, &y),
        }
        Ok((e, g1))
    }

    pub fn reset(&mut self) {
        self.cart.reset();
        self.guess_hessian();
    }
}

/// Session PES: Cartesian BFGS, Sella InternalPES, or cell-packed Cartesian.
pub enum SellaPes {
    Cartesian(CartesianPes),
    Internal(InternalPes),
    Cell(CellCartesianPes),
    CellInternal(CellInternalPes),
}

impl SellaPes {
    pub fn position(&self) -> ArrayView1<'_, f64> {
        match self {
            Self::Cartesian(p) => p.position(),
            Self::Internal(p) => p.position(),
            Self::Cell(p) => p.position(),
            Self::CellInternal(p) => p.position(),
        }
    }

    pub fn set_update(&mut self, update: HessUpdate) {
        match self {
            Self::Cartesian(p) => p.set_update(update),
            Self::Internal(p) => p.set_update(update),
            Self::Cell(p) => p.set_update(update),
            Self::CellInternal(p) => p.set_update(update),
        }
    }

    pub fn reset(&mut self) {
        match self {
            Self::Cartesian(p) => p.reset(),
            Self::Internal(p) => p.reset(),
            Self::Cell(p) => p.reset(),
            Self::CellInternal(p) => p.reset(),
        }
    }

    pub fn cartesian(&self) -> Option<&CartesianPes> {
        match self {
            Self::Cartesian(p) => Some(p),
            Self::Internal(p) => Some(p.cartesian()),
            Self::Cell(p) => Some(p.cartesian()),
            Self::CellInternal(p) => Some(p.internals().cartesian()),
        }
    }

    pub fn internal(&self) -> Option<&InternalPes> {
        match self {
            Self::Internal(p) => Some(p),
            Self::CellInternal(p) => Some(p.internals()),
            Self::Cartesian(_) | Self::Cell(_) => None,
        }
    }

    pub fn internal_mut(&mut self) -> Option<&mut InternalPes> {
        match self {
            Self::Internal(p) => Some(p),
            Self::CellInternal(p) => Some(p.internals_mut()),
            Self::Cartesian(_) | Self::Cell(_) => None,
        }
    }

    pub fn cell(&self) -> Option<&CellCartesianPes> {
        match self {
            Self::Cell(p) => Some(p),
            Self::Cartesian(_) | Self::Internal(_) | Self::CellInternal(_) => None,
        }
    }

    pub fn cell_internal(&self) -> Option<&CellInternalPes> {
        match self {
            Self::CellInternal(p) => Some(p),
            _ => None,
        }
    }
}

/// Cartesian PES plus a periodic cell.
pub struct CellCartesianPes {
    cart: CartesianPes,
    cell: Cell,
    /// Sella `cell_mask`: which of the 9 Cartesian cell entries are free.
    mask: [bool; 9],
    /// Packed BFGS on `[x_cart; cell_params]`.
    hess: BfgsModel,
    update: HessUpdate,
}

impl CellCartesianPes {
    pub fn new(x: Array1<f64>, masses: Array1<f64>, cell: Cell) -> Result<Self, SaddleError> {
        let cart = CartesianPes::new(x, masses)?;
        let n = cart.position().len() + 9;
        Ok(Self {
            cart,
            cell,
            mask: [true; 9],
            hess: BfgsModel::identity(n),
            update: HessUpdate::Bfgs,
        })
    }

    pub fn set_update(&mut self, update: HessUpdate) {
        self.update = update;
        self.cart.set_update(update);
    }

    pub fn hessian(&self) -> &BfgsModel {
        &self.hess
    }

    pub fn n_cell_dof(&self) -> usize {
        self.mask.iter().filter(|b| **b).count()
    }

    /// Packed length: 3N plus the free cell entries.
    pub fn packed_len(&self) -> usize {
        self.cart.position().len() + self.n_cell_dof()
    }

    /// Row-major 3x3 lattice.
    pub fn cell9(&self) -> [f64; 9] {
        let [a, b, c] = self.lattice();
        [a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]]
    }

    /// Free cell entries in mask order.
    pub fn cell_params(&self) -> Array1<f64> {
        let c = self.cell9();
        let mut p = Array1::zeros(self.n_cell_dof());
        let mut k = 0;
        for i in 0..9 {
            if self.mask[i] {
                p[k] = c[i];
                k += 1;
            }
        }
        p
    }

    /// `[x_cart; cell_params]`.
    pub fn packed(&self) -> Array1<f64> {
        let x = self.cart.position();
        let p = self.cell_params();
        let mut out = Array1::zeros(x.len() + p.len());
        for (i, v) in x.iter().enumerate() {
            out[i] = *v;
        }
        for (i, v) in p.iter().enumerate() {
            out[x.len() + i] = *v;
        }
        out
    }

    fn apply_cell9(&mut self, c: [f64; 9]) -> Result<(), SaddleError> {
        self.cell = cell_from_lattice(
            [c[0], c[1], c[2]],
            [c[3], c[4], c[5]],
            [c[6], c[7], c[8]],
            self.cell.origin(),
        )?;
        Ok(())
    }

    /// Packed gradient: Cartesian `g` plus masked `dE/dC`.
    pub fn packed_grad<S: PointSurface>(
        &self,
        surface: &S,
        g_cart: ArrayView1<f64>,
    ) -> Result<Array1<f64>, SaddleError> {
        let c9 = self.cell9();
        let gcell = match surface.cell_grad(self.cart.position(), &c9)? {
            Some(g) => g,
            None => fd_cell_grad(surface, self.cart.position(), &c9, &self.mask, 1e-5)?,
        };
        let n = self.cart.position().len();
        let mut out = Array1::zeros(n + self.n_cell_dof());
        for i in 0..n {
            out[i] = g_cart[i];
        }
        let mut k = 0;
        for i in 0..9 {
            if self.mask[i] {
                out[n + k] = gcell[i];
                k += 1;
            }
        }
        Ok(out)
    }

    /// Kick a packed increment `[dx; dcell_params]`.
    pub fn kick_packed<S: PointSurface>(
        &mut self,
        surface: &S,
        d: ArrayView1<f64>,
    ) -> Result<(f64, Array1<f64>), SaddleError> {
        let n = self.cart.position().len();
        let ncell = self.n_cell_dof();
        if d.len() != n + ncell {
            return Err(SaddleError::Shape(
                "packed kick must be 3N plus free cell entries".into(),
            ));
        }
        let c9 = self.cell9();
        let (e0, g0) = surface.eval_in_cell(self.cart.position(), &c9)?;
        let g0p = self.packed_grad(surface, g0.view())?;
        let _ = e0;
        let mut c1 = c9;
        let mut k = 0;
        for i in 0..9 {
            if self.mask[i] {
                c1[i] += d[n + k];
                k += 1;
            }
        }
        self.apply_cell9(c1)?;
        let dx = d.slice(ndarray::s![..n]).to_owned();
        let (e, g1) = self.cart.kick_from(surface, dx.view(), g0.view())?;
        let g1p = self.packed_grad(surface, g1.view())?;
        let y = &g1p - &g0p;
        let s = d.to_owned();
        match self.update {
            HessUpdate::Bfgs => self.hess.update(&s, &y),
            HessUpdate::TsBfgs => self.hess.update_ts(&s, &y),
        }
        Ok((e, g1p))
    }

    pub fn reset(&mut self) {
        self.cart.reset();
        self.hess = BfgsModel::identity(self.packed_len());
    }

    pub fn cell(&self) -> &Cell {
        &self.cell
    }

    pub fn set_cell(&mut self, cell: Cell) {
        self.cell = cell;
    }

    /// Sella `cell_mask`, row-major 3x3. Rebuilds the packed Hessian.
    pub fn set_mask(&mut self, mask: [bool; 9]) {
        self.mask = mask;
        self.hess = BfgsModel::identity(self.packed_len());
    }

    pub fn mask(&self) -> [bool; 9] {
        self.mask
    }

    /// Zero cell-step entries the mask forbids.
    pub fn project_cell_step(&self, mut dcell: [f64; 9]) -> [f64; 9] {
        for i in 0..9 {
            if !self.mask[i] {
                dcell[i] = 0.0;
            }
        }
        dcell
    }

    /// Lattice vectors as three Cartesian columns of the cell.
    pub fn lattice(&self) -> [[f64; 3]; 3] {
        lattice_of(&self.cell)
    }

    /// Cell angles `α, β, γ` in degrees.
    pub fn angles_deg(&self) -> [f64; 3] {
        let [a, b, c] = self.lattice();
        [
            angle_deg(b, c),
            angle_deg(a, c),
            angle_deg(a, b),
        ]
    }

    /// Sella `maybe_niggli_reduce`: rewrite a skewed cell in place.
    ///
    /// Triggers when any angle is more than `angle_threshold` degrees
    /// from 90. The Cartesian Hessian is not rewritten (no log-cell
    /// block lives here).
    pub fn maybe_niggli_reduce(&mut self, angle_threshold: f64) -> Result<bool, SaddleError> {
        let angs = self.angles_deg();
        let max_dev = angs
            .iter()
            .map(|a| (a - 90.0).abs())
            .fold(0.0_f64, f64::max);
        if max_dev <= angle_threshold {
            return Ok(false);
        }
        let [a, b, c] = self.lattice();
        let (a2, b2, c2) = niggli_reduce_vectors(a, b, c);
        self.cell = cell_from_lattice(a2, b2, c2, self.cell.origin())?;
        // Dest packs Cartesian cell entries, not Sella log-strain, so
        // there is no Frechet T. Drop the mixed Hessian.
        self.hess = BfgsModel::identity(self.packed_len());
        Ok(true)
    }

    pub fn cartesian(&self) -> &CartesianPes {
        &self.cart
    }

    pub fn position(&self) -> ArrayView1<'_, f64> {
        self.cart.position()
    }

    pub fn kick<S: PointSurface>(
        &mut self,
        surface: &S,
        d: ArrayView1<f64>,
    ) -> Result<(f64, Array1<f64>), SaddleError> {
        self.cart.kick(surface, d)
    }
}

/// Internals PES plus a periodic cell.
pub struct CellInternalPes {
    inner: InternalPes,
    cell: Cell,
    mask: [bool; 9],
    hess: BfgsModel,
    update: HessUpdate,
}

impl CellInternalPes {
    pub fn new(
        x: Array1<f64>,
        masses: Array1<f64>,
        chart: Constraints,
        cell: Cell,
    ) -> Result<Self, SaddleError> {
        let inner = InternalPes::new(x, masses, chart)?;
        let n = inner.n_int() + 9;
        Ok(Self {
            inner,
            cell,
            mask: [true; 9],
            hess: BfgsModel::identity(n),
            update: HessUpdate::Bfgs,
        })
    }

    pub fn cell(&self) -> &Cell {
        &self.cell
    }

    pub fn set_cell(&mut self, cell: Cell) {
        self.cell = cell;
    }

    pub fn set_update(&mut self, update: HessUpdate) {
        self.update = update;
        self.inner.set_update(update);
    }

    pub fn set_mask(&mut self, mask: [bool; 9]) {
        self.mask = mask;
        self.hess = BfgsModel::identity(self.packed_len());
    }

    pub fn mask(&self) -> [bool; 9] {
        self.mask
    }

    pub fn hessian(&self) -> &BfgsModel {
        &self.hess
    }

    pub fn n_cell_dof(&self) -> usize {
        self.mask.iter().filter(|b| **b).count()
    }

    pub fn packed_len(&self) -> usize {
        self.inner.n_int() + self.n_cell_dof()
    }

    pub fn cell9(&self) -> [f64; 9] {
        let [a, b, c] = lattice_of(&self.cell);
        [a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]]
    }

    pub fn cell_params(&self) -> Array1<f64> {
        let c = self.cell9();
        let mut p = Array1::zeros(self.n_cell_dof());
        let mut k = 0;
        for i in 0..9 {
            if self.mask[i] {
                p[k] = c[i];
                k += 1;
            }
        }
        p
    }

    /// `[q_int; cell_params]`.
    pub fn packed(&self) -> Result<Array1<f64>, SaddleError> {
        let q = self.inner.internals()?;
        let p = self.cell_params();
        let mut out = Array1::zeros(q.len() + p.len());
        for (i, v) in q.iter().enumerate() {
            out[i] = *v;
        }
        for (i, v) in p.iter().enumerate() {
            out[q.len() + i] = *v;
        }
        Ok(out)
    }

    fn apply_cell9(&mut self, c: [f64; 9]) -> Result<(), SaddleError> {
        self.cell = cell_from_lattice(
            [c[0], c[1], c[2]],
            [c[3], c[4], c[5]],
            [c[6], c[7], c[8]],
            self.cell.origin(),
        )?;
        Ok(())
    }

    pub fn packed_grad<S: PointSurface>(
        &self,
        surface: &S,
        g_cart: ArrayView1<f64>,
    ) -> Result<Array1<f64>, SaddleError> {
        let g_int = self.inner.internals_grad(g_cart)?;
        let c9 = self.cell9();
        let gcell = match surface.cell_grad(self.inner.position(), &c9)? {
            Some(g) => g,
            None => fd_cell_grad(surface, self.inner.position(), &c9, &self.mask, 1e-5)?,
        };
        let n = g_int.len();
        let mut out = Array1::zeros(n + self.n_cell_dof());
        for i in 0..n {
            out[i] = g_int[i];
        }
        let mut k = 0;
        for i in 0..9 {
            if self.mask[i] {
                out[n + k] = gcell[i];
                k += 1;
            }
        }
        Ok(out)
    }

    /// Kick `[dq; dcell_params]`.
    pub fn kick_packed<S: PointSurface>(
        &mut self,
        surface: &S,
        d: ArrayView1<f64>,
    ) -> Result<(f64, Array1<f64>), SaddleError> {
        let nint = self.inner.n_int();
        let ncell = self.n_cell_dof();
        if d.len() != nint + ncell {
            return Err(SaddleError::Shape(
                "packed internals+cell kick length must match the chart".into(),
            ));
        }
        let c9 = self.cell9();
        let (e0, g0) = surface.eval_in_cell(self.inner.position(), &c9)?;
        let g0p = self.packed_grad(surface, g0.view())?;
        let _ = e0;
        let mut c1 = c9;
        let mut k = 0;
        for i in 0..9 {
            if self.mask[i] {
                c1[i] += d[nint + k];
                k += 1;
            }
        }
        self.apply_cell9(c1)?;
        let dq = d.slice(s![..nint]).to_owned();
        let (e, g1) = self.inner.kick(surface, dq.view())?;
        let g1p = self.packed_grad(surface, g1.view())?;
        let y = &g1p - &g0p;
        let s = d.to_owned();
        match self.update {
            HessUpdate::Bfgs => self.hess.update(&s, &y),
            HessUpdate::TsBfgs => self.hess.update_ts(&s, &y),
        }
        Ok((e, g1p))
    }

    /// Sella `maybe_niggli_reduce`. Packed Hessian is dropped.
    pub fn maybe_niggli_reduce(&mut self, angle_threshold: f64) -> Result<bool, SaddleError> {
        let [a, b, c] = lattice_of(&self.cell);
        let angs = [angle_deg(b, c), angle_deg(a, c), angle_deg(a, b)];
        let max_dev = angs
            .iter()
            .map(|x| (x - 90.0).abs())
            .fold(0.0_f64, f64::max);
        if max_dev <= angle_threshold {
            return Ok(false);
        }
        let (a2, b2, c2) = niggli_reduce_vectors(a, b, c);
        self.cell = cell_from_lattice(a2, b2, c2, self.cell.origin())?;
        self.hess = BfgsModel::identity(self.packed_len());
        Ok(true)
    }

    pub fn internals(&self) -> &InternalPes {
        &self.inner
    }

    pub fn internals_mut(&mut self) -> &mut InternalPes {
        &mut self.inner
    }

    pub fn position(&self) -> ArrayView1<'_, f64> {
        self.inner.position()
    }

    pub fn reset(&mut self) {
        self.inner.reset();
        self.hess = BfgsModel::identity(self.packed_len());
    }

    pub fn kick<S: PointSurface>(
        &mut self,
        surface: &S,
        dq: ArrayView1<f64>,
    ) -> Result<(f64, Array1<f64>), SaddleError> {
        self.inner.kick(surface, dq)
    }
}

fn fd_cell_grad<S: PointSurface>(
    surface: &S,
    x: ArrayView1<f64>,
    cell: &[f64; 9],
    mask: &[bool; 9],
    eps: f64,
) -> Result<[f64; 9], SaddleError> {
    let e0 = surface.eval_in_cell(x, cell)?.0;
    let mut g = [0.0; 9];
    for i in 0..9 {
        if !mask[i] {
            continue;
        }
        let mut cp = *cell;
        cp[i] += eps;
        let ep = surface.eval_in_cell(x, &cp)?.0;
        g[i] = (ep - e0) / eps;
    }
    Ok(g)
}

fn internals_grad_from_b(
    b: &ndarray::Array2<f64>,
    g_cart: ArrayView1<f64>,
) -> Result<Array1<f64>, SaddleError> {
    if b.nrows() == 0 {
        return Ok(Array1::zeros(0));
    }
    if g_cart.len() != b.ncols() {
        return Err(SaddleError::Shape(
            "cartesian gradient must match the 3N frame".into(),
        ));
    }
    let mut bg = Array1::zeros(b.nrows());
    for i in 0..b.nrows() {
        let mut acc = 0.0;
        for k in 0..b.ncols() {
            acc += b[(i, k)] * g_cart[k];
        }
        bg[i] = acc;
    }
    Ok(solve_bbt(b, &bg))
}

fn lattice_of(cell: &Cell) -> [[f64; 3]; 3] {
    let o = cell.origin();
    let a = cell.cartesian([1.0, 0.0, 0.0]);
    let b = cell.cartesian([0.0, 1.0, 0.0]);
    let c = cell.cartesian([0.0, 0.0, 1.0]);
    [
        [a[0] - o[0], a[1] - o[1], a[2] - o[2]],
        [b[0] - o[0], b[1] - o[1], b[2] - o[2]],
        [c[0] - o[0], c[1] - o[1], c[2] - o[2]],
    ]
}

fn cell_from_lattice(
    a: [f64; 3],
    b: [f64; 3],
    c: [f64; 3],
    origin: [f64; 3],
) -> Result<Cell, SaddleError> {
    Cell::from_vectors(a, b, c, origin).map_err(|_| {
        SaddleError::Shape("niggli produced a singular cell".into())
    })
}

fn dot3(u: [f64; 3], v: [f64; 3]) -> f64 {
    u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
}

fn norm3(u: [f64; 3]) -> f64 {
    dot3(u, u).sqrt()
}

fn angle_deg(u: [f64; 3], v: [f64; 3]) -> f64 {
    let nu = norm3(u);
    let nv = norm3(v);
    if nu < 1e-18 || nv < 1e-18 {
        return 0.0;
    }
    let c = (dot3(u, v) / (nu * nv)).clamp(-1.0, 1.0);
    c.acos() * 180.0 / std::f64::consts::PI
}

/// Niggli-reduce a row-major 3x3 cell in place. Returns whether it rewrote.
pub fn niggli_reduce_cell(cell: &mut [f64; 9], angle_threshold: f64) -> Result<bool, SaddleError> {
    let a = [cell[0], cell[1], cell[2]];
    let b = [cell[3], cell[4], cell[5]];
    let c = [cell[6], cell[7], cell[8]];
    let angs = [angle_deg(b, c), angle_deg(a, c), angle_deg(a, b)];
    let max_dev = angs
        .iter()
        .map(|x| (x - 90.0).abs())
        .fold(0.0_f64, f64::max);
    if max_dev <= angle_threshold {
        return Ok(false);
    }
    let (a2, b2, c2) = niggli_reduce_vectors(a, b, c);
    *cell = [
        a2[0], a2[1], a2[2], b2[0], b2[1], b2[2], c2[0], c2[1], c2[2],
    ];
    Ok(true)
}

/// Krivy–Gruber / Grosse-Kunstlere Niggli reduction of three lattice vectors.
pub fn niggli_reduce_vectors(
    mut a: [f64; 3],
    mut b: [f64; 3],
    mut c: [f64; 3],
) -> ([f64; 3], [f64; 3], [f64; 3]) {
    let eps = 1e-8;
    fn add(u: [f64; 3], s: f64, v: [f64; 3]) -> [f64; 3] {
        [u[0] + s * v[0], u[1] + s * v[1], u[2] + s * v[2]]
    }
    fn neg(u: [f64; 3]) -> [f64; 3] {
        [-u[0], -u[1], -u[2]]
    }
    for _ in 0..10_000 {
        let aa = dot3(a, a);
        let bb = dot3(b, b);
        let cc = dot3(c, c);
        let xi = 2.0 * dot3(b, c);
        let eta = 2.0 * dot3(a, c);
        let zeta = 2.0 * dot3(a, b);
        // (i)
        if aa > bb + eps || ((aa - bb).abs() <= eps && xi.abs() > eta.abs() + eps) {
            std::mem::swap(&mut a, &mut b);
            continue;
        }
        // (ii)
        if bb > cc + eps || ((bb - cc).abs() <= eps && eta.abs() > zeta.abs() + eps) {
            std::mem::swap(&mut b, &mut c);
            continue;
        }
        // (iii) type reduction
        if xi * eta * zeta > 0.0 {
            if xi < 0.0 {
                b = neg(b);
            }
            if eta < 0.0 {
                a = neg(a);
            }
            if zeta < 0.0 {
                // flipping a or b already handled two signs; flip c if needed
                if 2.0 * dot3(b, c) < 0.0 {
                    c = neg(c);
                }
            }
        } else {
            if xi > 0.0 {
                b = neg(b);
            }
            if eta > 0.0 {
                a = neg(a);
            }
            if 2.0 * dot3(a, b) > 0.0 {
                if 2.0 * dot3(b, c) > 0.0 {
                    c = neg(c);
                } else if 2.0 * dot3(a, c) > 0.0 {
                    c = neg(c);
                }
            }
        }
        let aa = dot3(a, a);
        let bb = dot3(b, b);
        let cc = dot3(c, c);
        let xi = 2.0 * dot3(b, c);
        let eta = 2.0 * dot3(a, c);
        let zeta = 2.0 * dot3(a, b);
        // (iv)
        if xi.abs() > bb + eps
            || ((xi - bb).abs() <= eps && 2.0 * eta < zeta - eps)
            || ((xi + bb).abs() <= eps && zeta < -eps)
        {
            let s = if xi > 0.0 { -1.0 } else { 1.0 };
            b = add(b, s, c);
            continue;
        }
        // (v)
        if eta.abs() > aa + eps
            || ((eta - aa).abs() <= eps && 2.0 * xi < zeta - eps)
            || ((eta + aa).abs() <= eps && zeta < -eps)
        {
            let s = if eta > 0.0 { -1.0 } else { 1.0 };
            a = add(a, s, c);
            continue;
        }
        // (vi)
        if zeta.abs() > aa + eps
            || ((zeta - aa).abs() <= eps && 2.0 * xi < eta - eps)
            || ((zeta + aa).abs() <= eps && eta < -eps)
        {
            let s = if zeta > 0.0 { -1.0 } else { 1.0 };
            a = add(a, s, b);
            continue;
        }
        // (vii) extra Grosse-Kunstlere
        let sum = xi.abs() + eta.abs() + zeta.abs() + aa + bb;
        if sum < cc - eps
            || ((sum - cc).abs() <= eps
                && 2.0 * (aa + eta) + zeta > eps)
        {
            let s = if xi + eta + zeta > 0.0 { -1.0 } else { 1.0 };
            c = add(c, s, a);
            c = add(c, s, b);
            continue;
        }
        break;
    }
    (a, b, c)
}

/// Min-norm `dx` for `B dx = dq`: `(B B^T) y = dq`, `dx = B^T y`.
fn min_norm_dx(b: &ndarray::Array2<f64>, dq: &Array1<f64>) -> Array1<f64> {
    let y = solve_bbt(b, dq);
    let mut dx = Array1::zeros(b.ncols());
    for t in 0..b.ncols() {
        let mut acc = 0.0;
        for i in 0..b.nrows() {
            acc += b[(i, t)] * y[i];
        }
        dx[t] = acc;
    }
    dx
}

fn solve_bbt(b: &ndarray::Array2<f64>, rhs: &Array1<f64>) -> Array1<f64> {
    let m = b.nrows();
    if m == 0 {
        return Array1::zeros(0);
    }
    let mut a = ndarray::Array2::<f64>::zeros((m, m));
    for i in 0..m {
        for k in 0..m {
            let mut acc = 0.0;
            for t in 0..b.ncols() {
                acc += b[(i, t)] * b[(k, t)];
            }
            a[(i, k)] = acc;
        }
        a[(i, i)] += 1e-14;
    }
    // Cholesky on the tiny internals Gramian.
    let n = m;
    let mut l = ndarray::Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut acc = a[(i, j)];
            for k in 0..j {
                acc -= l[(i, k)] * l[(j, k)];
            }
            if i == j {
                l[(i, i)] = acc.max(1e-30).sqrt();
            } else {
                l[(i, j)] = acc / l[(j, j)];
            }
        }
    }
    let mut y = Array1::zeros(n);
    for i in 0..n {
        let mut acc = rhs[i];
        for k in 0..i {
            acc -= l[(i, k)] * y[k];
        }
        y[i] = acc / l[(i, i)];
    }
    for i in (0..n).rev() {
        let mut acc = y[i];
        for k in (i + 1)..n {
            acc -= l[(k, i)] * y[k];
        }
        y[i] = acc / l[(i, i)];
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::internal::{CartAxis, Translation};
    use crate::minmode::PointSurface;
    use ndarray::{Array1, ArrayView1};

    struct Well;
    impl PointSurface for Well {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let t = x[0];
            let mut g = Array1::zeros(x.len());
            g[0] = 4.0 * t * (t * t - 1.0);
            Ok(((t * t - 1.0).powi(2), g))
        }
    }

    #[test]
    fn translation_kick_moves_the_com() {
        let x = Array1::zeros(6);
        let masses = Array1::from(vec![1.0, 1.0]);
        let mut chart = Constraints::new(2).unwrap();
        chart
            .fix_translation(Translation::all(2, CartAxis::X).unwrap(), x.view(), Some(0.0))
            .unwrap();
        let mut pes = InternalPes::new(x, masses, chart).unwrap();
        let dq = Array1::from(vec![0.2]);
        pes.kick(&Well, dq.view()).unwrap();
        // B_i = 1/2 on each atom x; min-norm dx is 0.2 on both x slots.
        assert!((pes.position()[0] - 0.2).abs() < 1e-12);
        assert!((pes.position()[3] - 0.2).abs() < 1e-12);
        let q = pes.internals().unwrap();
        assert!((q[0] - 0.2).abs() < 1e-12);
    }

    #[test]
    fn cell_wrappers_hold_the_linkcell() {
        let cart = CellCartesianPes::new(
            Array1::zeros(6),
            Array1::from(vec![1.0, 1.0]),
            Cell::ortho(10.0, 10.0, 10.0).unwrap(),
        )
        .unwrap();
        let _ = cart.cell();
        let mut cart = cart;
        cart.set_mask([
            true, false, false, false, true, false, false, false, true,
        ]);
        let step = cart.project_cell_step([1.0; 9]);
        assert_eq!(step[0], 1.0);
        assert_eq!(step[1], 0.0);
        assert_eq!(step[4], 1.0);
        let mut chart = Constraints::new(2).unwrap();
        chart.fix_com(Array1::zeros(6).view()).unwrap();
        let inner = CellInternalPes::new(
            Array1::zeros(6),
            Array1::from(vec![1.0, 1.0]),
            chart,
            Cell::ortho(10.0, 10.0, 10.0).unwrap(),
        )
        .unwrap();
        assert_eq!(inner.internals().chart().counts().ntrans, 3);
    }

    struct Quad;
    impl PointSurface for Quad {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let mut g = Array1::zeros(x.len());
            g[0] = 2.0 * x[0];
            Ok((x[0] * x[0], g))
        }
    }

    #[test]
    fn internals_kick_updates_the_int_hessian() {
        let mut x = Array1::zeros(6);
        x[0] = 0.4;
        let masses = Array1::from(vec![1.0, 1.0]);
        let mut chart = Constraints::new(2).unwrap();
        chart
            .fix_translation(Translation::all(2, CartAxis::X).unwrap(), x.view(), Some(0.0))
            .unwrap();
        let mut pes = InternalPes::new(x, masses, chart).unwrap();
        assert_eq!(pes.n_int(), 1);
        assert_eq!(pes.hessian().hessian().nrows(), 1);
        assert!(
            (pes.hessian().hessian()[(0, 0)] - 70.0).abs() < 1e-12,
            "translation guess {}",
            pes.hessian().hessian()[(0, 0)]
        );
        let h0 = pes.hessian().hessian()[(0, 0)];
        let dq = Array1::from(vec![0.15]);
        pes.kick(&Quad, dq.view()).unwrap();
        let h1 = pes.hessian().hessian()[(0, 0)];
        assert!(h1.is_finite());
        assert!((h1 - h0).abs() > 1e-18, "internals Hessian must accept a pair");
    }

    #[test]
    fn empty_chart_is_refused() {
        let chart = Constraints::new(2).unwrap();
        let err = InternalPes::new(Array1::zeros(6), Array1::from(vec![1.0, 1.0]), chart);
        assert!(err.is_err());
    }

    #[test]
    fn niggli_rewrites_a_skewed_cell() {
        let mut cell = [
            1.0, 0.0, 0.0, 0.9, 0.15, 0.0, 0.4, 0.5, 1.0,
        ];
        let a = [cell[0], cell[1], cell[2]];
        let b = [cell[3], cell[4], cell[5]];
        let c = [cell[6], cell[7], cell[8]];
        let before = [angle_deg(b, c), angle_deg(a, c), angle_deg(a, b)];
        let applied = niggli_reduce_cell(&mut cell, 20.0).unwrap();
        assert!(applied, "angles {before:?} should trigger");
        let a2 = [cell[0], cell[1], cell[2]];
        let b2 = [cell[3], cell[4], cell[5]];
        let c2 = [cell[6], cell[7], cell[8]];
        let vol = |u: [f64; 3], v: [f64; 3], w: [f64; 3]| {
            u[0] * (v[1] * w[2] - v[2] * w[1])
                - u[1] * (v[0] * w[2] - v[2] * w[0])
                + u[2] * (v[0] * w[1] - v[1] * w[0])
        };
        assert!((vol(a, b, c).abs() - vol(a2, b2, c2).abs()).abs() < 1e-8);
        let mut cart = CellCartesianPes::new(
            Array1::zeros(6),
            Array1::from(vec![1.0, 1.0]),
            Cell::from_vectors(a, b, c, [0.0, 0.0, 0.0]).unwrap(),
        )
        .unwrap();
        assert!(cart.maybe_niggli_reduce(20.0).unwrap());
        assert!(!cart.maybe_niggli_reduce(90.0).unwrap());
    }

    #[test]
    fn packed_cell_dof_follows_the_mask() {
        let mut cart = CellCartesianPes::new(
            Array1::zeros(6),
            Array1::from(vec![1.0, 1.0]),
            Cell::ortho(4.0, 5.0, 6.0).unwrap(),
        )
        .unwrap();
        assert_eq!(cart.n_cell_dof(), 9);
        assert_eq!(cart.packed_len(), 15);
        cart.set_mask([
            true, false, false, false, true, false, false, false, true,
        ]);
        assert_eq!(cart.n_cell_dof(), 3);
        assert_eq!(cart.packed_len(), 9);
        let p = cart.cell_params();
        assert!((p[0] - 4.0).abs() < 1e-12);
        assert!((p[1] - 5.0).abs() < 1e-12);
        assert!((p[2] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn packed_cell_internal_follows_the_mask() {
        let mut chart = Constraints::new(2).unwrap();
        chart
            .fix_translation(
                Translation::all(2, CartAxis::X).unwrap(),
                Array1::zeros(6).view(),
                Some(0.0),
            )
            .unwrap();
        let mut pes = CellInternalPes::new(
            Array1::zeros(6),
            Array1::from(vec![1.0, 1.0]),
            chart,
            Cell::ortho(4.0, 5.0, 6.0).unwrap(),
        )
        .unwrap();
        pes.set_mask([
            true, false, false, false, false, false, false, false, false,
        ]);
        assert_eq!(pes.n_cell_dof(), 1);
        assert_eq!(pes.packed_len(), 2);
        let p = pes.packed().unwrap();
        assert!((p[1] - 4.0).abs() < 1e-12);
    }
}
