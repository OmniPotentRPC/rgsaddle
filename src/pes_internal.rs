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

use ndarray::{Array1, ArrayView1};

use crate::constraints::Constraints;
use crate::error::SaddleError;
use crate::mic::Cell;
use crate::minmode::PointSurface;
use crate::pes::{CartesianPes, HessUpdate};

/// Sella `InternalPES`: Cartesian geometry, internals chart, MW Hessian.
pub struct InternalPes {
    cart: CartesianPes,
    chart: Constraints,
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
        Ok(Self {
            cart: CartesianPes::new(x, masses)?,
            chart,
        })
    }

    pub fn set_update(&mut self, update: HessUpdate) {
        self.cart.set_update(update);
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
        Ok(solve_bbt(&b, &bg))
    }

    /// Map `dq` to Cartesian by `B dx = dq` and kick the Cartesian PES.
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
        let dx = min_norm_dx(&b, &dq.to_owned());
        self.cart.kick(surface, dx.view())
    }

    pub fn reset(&mut self) {
        self.cart.reset();
    }
}

/// Cartesian PES plus a periodic cell.
pub struct CellCartesianPes {
    cart: CartesianPes,
    cell: Cell,
    /// Sella `cell_mask`: which of the 9 Cartesian cell entries are free.
    mask: [bool; 9],
}

impl CellCartesianPes {
    pub fn new(x: Array1<f64>, masses: Array1<f64>, cell: Cell) -> Result<Self, SaddleError> {
        Ok(Self {
            cart: CartesianPes::new(x, masses)?,
            cell,
            mask: [true; 9],
        })
    }

    pub fn cell(&self) -> &Cell {
        &self.cell
    }

    pub fn set_cell(&mut self, cell: Cell) {
        self.cell = cell;
    }

    /// Sella `cell_mask`, row-major 3x3.
    pub fn set_mask(&mut self, mask: [bool; 9]) {
        self.mask = mask;
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
}

impl CellInternalPes {
    pub fn new(
        x: Array1<f64>,
        masses: Array1<f64>,
        chart: Constraints,
        cell: Cell,
    ) -> Result<Self, SaddleError> {
        Ok(Self {
            inner: InternalPes::new(x, masses, chart)?,
            cell,
        })
    }

    pub fn cell(&self) -> &Cell {
        &self.cell
    }

    pub fn set_cell(&mut self, cell: Cell) {
        self.cell = cell;
    }

    pub fn internals(&self) -> &InternalPes {
        &self.inner
    }

    pub fn internals_mut(&mut self) -> &mut InternalPes {
        &mut self.inner
    }

    pub fn kick<S: PointSurface>(
        &mut self,
        surface: &S,
        dq: ArrayView1<f64>,
    ) -> Result<(f64, Array1<f64>), SaddleError> {
        self.inner.kick(surface, dq)
    }
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
}
