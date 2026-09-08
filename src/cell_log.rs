//! Sella cell log-deformation chart: `L = logm(F) * factor`,
//! `F = cell @ inv(orig)`.
//!
//! 3x3 `expm` / `logm` are scaling-and-squaring plus a Taylor series.
//! Dest packs either raw lattice entries or this `L` chart. Niggli
//! on the log chart transforms the cell Hessian by `T = J_old^{-1} J_new`.

use ndarray::{Array1, Array2};

use crate::error::SaddleError;
use crate::mic::Cell;

/// How the nine lattice numbers enter the packed PES.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CellChart {
    /// Masked row-major lattice entries.
    #[default]
    Entries,
    /// Sella `L = logm(F) * factor`.
    LogDeform,
}

impl CellChart {
    pub const fn to_abi(self) -> i32 {
        match self {
            Self::Entries => 0,
            Self::LogDeform => 1,
        }
    }

    pub const fn try_from_abi(v: i32) -> Option<Self> {
        match v {
            0 => Some(Self::Entries),
            1 => Some(Self::LogDeform),
            _ => None,
        }
    }
}

pub type M3 = [[f64; 3]; 3];

pub fn m3_from_row9(c: [f64; 9]) -> M3 {
    [[c[0], c[1], c[2]], [c[3], c[4], c[5]], [c[6], c[7], c[8]]]
}

pub fn m3_to_row9(a: M3) -> [f64; 9] {
    [
        a[0][0], a[0][1], a[0][2], a[1][0], a[1][1], a[1][2], a[2][0], a[2][1], a[2][2],
    ]
}

fn eye() -> M3 {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
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

fn sub(a: M3, b: M3) -> M3 {
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = a[i][j] - b[i][j];
        }
    }
    c
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

pub fn m3_mul(a: M3, b: M3) -> M3 {
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    c
}

fn frobenius(a: M3) -> f64 {
    let mut s = 0.0;
    for row in a {
        for value in row {
            s += value * value;
        }
    }
    s.sqrt()
}

pub fn m3_det(a: M3) -> f64 {
    a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
}

pub fn m3_inv(a: M3) -> Result<M3, SaddleError> {
    let d = m3_det(a);
    if d.abs() < 1e-18 {
        return Err(SaddleError::Shape("singular 3x3 cell".into()));
    }
    let mut c = [[0.0; 3]; 3];
    c[0][0] = (a[1][1] * a[2][2] - a[1][2] * a[2][1]) / d;
    c[0][1] = (a[0][2] * a[2][1] - a[0][1] * a[2][2]) / d;
    c[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) / d;
    c[1][0] = (a[1][2] * a[2][0] - a[1][0] * a[2][2]) / d;
    c[1][1] = (a[0][0] * a[2][2] - a[0][2] * a[2][0]) / d;
    c[1][2] = (a[0][2] * a[1][0] - a[0][0] * a[1][2]) / d;
    c[2][0] = (a[1][0] * a[2][1] - a[1][1] * a[2][0]) / d;
    c[2][1] = (a[0][1] * a[2][0] - a[0][0] * a[2][1]) / d;
    c[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) / d;
    Ok(c)
}

fn sqrtm_db(a: M3) -> Result<M3, SaddleError> {
    let mut y = a;
    let mut z = eye();
    for _ in 0..16 {
        let yi = m3_inv(y)?;
        let zi = m3_inv(z)?;
        let y1 = scale(0.5, add(y, zi));
        let z1 = scale(0.5, add(z, yi));
        if frobenius(sub(y1, y)) < 1e-14 {
            return Ok(y1);
        }
        y = y1;
        z = z1;
    }
    Ok(y)
}

/// Scaling-and-squaring Taylor `expm` on a 3x3.
pub fn expm_3x3(u: M3) -> M3 {
    let mut a = u;
    let mut s = 0u32;
    while frobenius(a) > 0.5 && s < 8 {
        a = scale(0.5, a);
        s += 1;
    }
    let mut term = eye();
    let mut sum = eye();
    for k in 1..14 {
        term = m3_mul(term, a);
        term = scale(1.0 / k as f64, term);
        sum = add(sum, term);
    }
    for _ in 0..s {
        sum = m3_mul(sum, sum);
    }
    sum
}

/// Inverse-scaling plus series `logm` on a 3x3 near GL+(3).
pub fn logm_3x3(f: M3) -> Result<M3, SaddleError> {
    let mut a = f;
    let mut s = 0u32;
    while frobenius(sub(a, eye())) > 0.3 && s < 12 {
        a = sqrtm_db(a)?;
        s += 1;
    }
    let x = sub(a, eye());
    let mut xk = x;
    let mut sum = x;
    for k in 2..18 {
        xk = m3_mul(xk, x);
        let sign = if k % 2 == 0 { -1.0 } else { 1.0 };
        sum = add(sum, scale(sign / k as f64, xk));
    }
    Ok(scale((1u32 << s) as f64, sum))
}

/// Finite-difference Frechet `d expm(U)[E]`.
pub fn expm_frechet(u: M3, e: M3) -> M3 {
    let n = frobenius(u).max(1.0);
    let eps = 1e-7 * n;
    let plus = expm_3x3(add(u, scale(eps, e)));
    let minus = expm_3x3(sub(u, scale(eps, e)));
    scale(0.5 / eps, sub(plus, minus))
}

/// Packed Hessian dest can rewrite (niggli `T`).
pub struct PackedHess {
    b: Array2<f64>,
}

impl PackedHess {
    pub fn identity(n: usize) -> Self {
        Self { b: Array2::eye(n) }
    }

    pub fn from_matrix(b: Array2<f64>) -> Self {
        Self { b }
    }

    pub fn hessian(&self) -> &Array2<f64> {
        &self.b
    }

    pub fn forget(&mut self) {
        let n = self.b.nrows();
        self.b = Array2::eye(n);
    }

    pub fn update(&mut self, s: &Array1<f64>, y: &Array1<f64>) {
        crate::qn::update_h(&mut self.b, s, y, crate::qn::HessUpdate::Bfgs);
    }

    pub fn update_ts(&mut self, s: &Array1<f64>, y: &Array1<f64>) {
        crate::qn::update_h(&mut self.b, s, y, crate::qn::HessUpdate::TsBfgs);
    }
}

/// Living cell chart: lattice, reference, mask, log vs entries.
pub struct CellState {
    cell: Cell,
    orig: [f64; 9],
    mask: [bool; 9],
    factor: f64,
    kind: CellChart,
}

impl CellState {
    pub fn new(cell: Cell, n_atoms: usize) -> Self {
        let orig = cell9_of(&cell);
        Self {
            cell,
            orig,
            mask: [true; 9],
            factor: (n_atoms as f64).max(1.0),
            kind: CellChart::Entries,
        }
    }

    pub fn cell(&self) -> &Cell {
        &self.cell
    }

    pub fn set_cell(&mut self, cell: Cell) {
        self.orig = cell9_of(&cell);
        self.cell = cell;
    }

    pub fn set_mask(&mut self, mask: [bool; 9]) {
        self.mask = mask;
    }

    pub fn mask(&self) -> [bool; 9] {
        self.mask
    }

    pub fn kind(&self) -> CellChart {
        self.kind
    }

    pub fn set_kind(&mut self, kind: CellChart) {
        self.kind = kind;
        if kind == CellChart::LogDeform {
            self.orig = cell9_of(&self.cell);
        }
    }

    pub fn set_factor(&mut self, factor: f64) {
        if factor > 0.0 {
            self.factor = factor;
        }
    }

    pub fn factor(&self) -> f64 {
        self.factor
    }

    pub fn n_cell_dof(&self) -> usize {
        self.mask.iter().filter(|b| **b).count()
    }

    pub fn cell9(&self) -> [f64; 9] {
        cell9_of(&self.cell)
    }

    pub fn lattice(&self) -> [[f64; 3]; 3] {
        lattice_of(&self.cell)
    }

    fn apply_cell9(&mut self, c: [f64; 9]) -> Result<(), SaddleError> {
        self.cell = cell_from_row9(c, self.cell.origin())?;
        Ok(())
    }

    pub fn params(&self) -> Result<Array1<f64>, SaddleError> {
        let raw = match self.kind {
            CellChart::Entries => self.cell9(),
            CellChart::LogDeform => m3_to_row9(self.log_deform()?),
        };
        let mut p = Array1::zeros(self.n_cell_dof());
        let mut k = 0;
        for (i, value) in raw.iter().enumerate() {
            if self.mask[i] {
                p[k] = *value;
                k += 1;
            }
        }
        Ok(p)
    }

    fn log_deform(&self) -> Result<M3, SaddleError> {
        let f = m3_mul(m3_from_row9(self.cell9()), m3_inv(m3_from_row9(self.orig))?);
        Ok(scale(self.factor, logm_3x3(f)?))
    }

    /// Add a masked increment in the active chart and write the cell.
    pub fn kick_params(&mut self, d: &Array1<f64>) -> Result<(), SaddleError> {
        if d.len() != self.n_cell_dof() {
            return Err(SaddleError::Shape(
                "cell increment must match the free cell DOF".into(),
            ));
        }
        match self.kind {
            CellChart::Entries => {
                let mut c = self.cell9();
                let mut k = 0;
                for (i, value) in c.iter_mut().enumerate() {
                    if self.mask[i] {
                        *value += d[k];
                        k += 1;
                    }
                }
                self.apply_cell9(c)
            }
            CellChart::LogDeform => {
                let mut l = m3_to_row9(self.log_deform()?);
                let mut k = 0;
                for (i, value) in l.iter_mut().enumerate() {
                    if self.mask[i] {
                        *value += d[k];
                        k += 1;
                    }
                }
                let u = scale(1.0 / self.factor, m3_from_row9(l));
                let f = expm_3x3(u);
                let c = m3_mul(f, m3_from_row9(self.orig));
                self.apply_cell9(m3_to_row9(c))
            }
        }
    }

    /// Convert `dE/dC` (row-major 9) into the active chart gradient.
    pub fn chart_grad(&self, g_c: [f64; 9]) -> Result<Array1<f64>, SaddleError> {
        let g_raw = match self.kind {
            CellChart::Entries => g_c,
            CellChart::LogDeform => {
                let j = self.jacobian_l_to_c()?;
                let mut g_l = [0.0; 9];
                for ij in 0..9 {
                    let mut acc = 0.0;
                    for ab in 0..9 {
                        acc += j[(ab, ij)] * g_c[ab];
                    }
                    g_l[ij] = acc;
                }
                g_l
            }
        };
        let mut out = Array1::zeros(self.n_cell_dof());
        let mut k = 0;
        for (i, value) in g_raw.iter().enumerate() {
            if self.mask[i] {
                out[k] = *value;
                k += 1;
            }
        }
        Ok(out)
    }

    /// `J[ab, ij] = d C_ab / d L_ij`.
    fn jacobian_l_to_c(&self) -> Result<Array2<f64>, SaddleError> {
        let orig = m3_from_row9(self.orig);
        let f = m3_mul(m3_from_row9(self.cell9()), m3_inv(orig)?);
        let u = logm_3x3(f)?;
        let mut j = Array2::<f64>::zeros((9, 9));
        for ij in 0..9 {
            let mut e = [[0.0; 3]; 3];
            e[ij / 3][ij % 3] = 1.0 / self.factor;
            let df = expm_frechet(u, e);
            let dc = m3_mul(df, orig);
            let row = m3_to_row9(dc);
            for ab in 0..9 {
                j[(ab, ij)] = row[ab];
            }
        }
        Ok(j)
    }

    /// Niggli-reduce. Returns `(applied, T_masked)` on the log chart.
    /// `T` is `ncell x ncell`. Entries chart returns `None` for `T`.
    pub fn maybe_niggli(
        &mut self,
        angle_threshold: f64,
        niggli_fn: impl Fn([f64; 3], [f64; 3], [f64; 3]) -> ([f64; 3], [f64; 3], [f64; 3]),
    ) -> Result<(bool, Option<Array2<f64>>), SaddleError> {
        let [a, b, c] = lattice_of(&self.cell);
        let angs = [angle_deg(b, c), angle_deg(a, c), angle_deg(a, b)];
        let max_dev = angs
            .iter()
            .map(|x| (x - 90.0).abs())
            .fold(0.0_f64, f64::max);
        if max_dev <= angle_threshold {
            return Ok((false, None));
        }
        let t = if self.kind == CellChart::LogDeform {
            Some(self.niggli_t(niggli_fn)?)
        } else {
            let (a2, b2, c2) = niggli_fn(a, b, c);
            self.cell = cell_from_vecs(a2, b2, c2, self.cell.origin())?;
            self.orig = cell9_of(&self.cell);
            None
        };
        Ok((true, t))
    }

    fn niggli_t(
        &mut self,
        niggli_fn: impl Fn([f64; 3], [f64; 3], [f64; 3]) -> ([f64; 3], [f64; 3], [f64; 3]),
    ) -> Result<Array2<f64>, SaddleError> {
        let j_old = self.jacobian_l_to_c()?;
        let [a, b, c] = lattice_of(&self.cell);
        let (a2, b2, c2) = niggli_fn(a, b, c);
        self.cell = cell_from_vecs(a2, b2, c2, self.cell.origin())?;
        self.orig = cell9_of(&self.cell);
        let orig = m3_from_row9(self.orig);
        let t_full = solve_3n(&j_old, &j_new_l0(orig, self.factor))?;
        let idx: Vec<usize> = (0..9).filter(|&i| self.mask[i]).collect();
        let m = idx.len();
        let mut t = Array2::<f64>::zeros((m, m));
        for (p, &ia) in idx.iter().enumerate() {
            for (q, &ib) in idx.iter().enumerate() {
                t[(p, q)] = t_full[(ia, ib)];
            }
        }
        Ok(t)
    }
}

fn j_new_l0(orig: M3, factor: f64) -> Array2<f64> {
    let mut j = Array2::<f64>::zeros((9, 9));
    for i in 0..3 {
        for (jcol, row) in orig.iter().enumerate() {
            let ij = 3 * i + jcol;
            for (q, value) in row.iter().enumerate() {
                let ab = 3 * i + q;
                j[(ab, ij)] = value / factor;
            }
        }
    }
    j
}

fn solve_3n(a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>, SaddleError> {
    let n = a.nrows();
    if a.ncols() != n || b.nrows() != n {
        return Err(SaddleError::Shape("niggli T solve shape".into()));
    }
    let mut out = Array2::<f64>::zeros((n, b.ncols()));
    for col in 0..b.ncols() {
        let mut rhs = Array1::zeros(n);
        for i in 0..n {
            rhs[i] = b[(i, col)];
        }
        let x = crate::eigensolve::solve_dense(a.view(), rhs.view())?;
        for i in 0..n {
            out[(i, col)] = x[i];
        }
    }
    Ok(out)
}

/// Apply `H_new = T^T H_old T` on the cell block of a packed Hessian.
pub fn transform_cell_block(h: &mut Array2<f64>, n_noncell: usize, t: &Array2<f64>) {
    let n = h.nrows();
    let m = t.nrows();
    if n != n_noncell + m || t.ncols() != m {
        return;
    }
    let mut h_cc = Array2::<f64>::zeros((m, m));
    for i in 0..m {
        for j in 0..m {
            h_cc[(i, j)] = h[(n_noncell + i, n_noncell + j)];
        }
    }
    let mut ht = Array2::<f64>::zeros((m, m));
    for i in 0..m {
        for j in 0..m {
            let mut acc = 0.0;
            for k in 0..m {
                acc += h_cc[(i, k)] * t[(k, j)];
            }
            ht[(i, j)] = acc;
        }
    }
    let mut ttht = Array2::<f64>::zeros((m, m));
    for i in 0..m {
        for j in 0..m {
            let mut acc = 0.0;
            for k in 0..m {
                acc += t[(k, i)] * ht[(k, j)];
            }
            ttht[(i, j)] = acc;
        }
    }
    for i in 0..m {
        for j in 0..m {
            h[(n_noncell + i, n_noncell + j)] = ttht[(i, j)];
        }
    }
    // coupling H_xc = H_xc @ T, H_cx = T^T @ H_cx
    let mut h_xc = Array2::<f64>::zeros((n_noncell, m));
    for i in 0..n_noncell {
        for j in 0..m {
            h_xc[(i, j)] = h[(i, n_noncell + j)];
        }
    }
    for i in 0..n_noncell {
        for j in 0..m {
            let mut acc = 0.0;
            for k in 0..m {
                acc += h_xc[(i, k)] * t[(k, j)];
            }
            h[(i, n_noncell + j)] = acc;
            h[(n_noncell + j, i)] = acc;
        }
    }
}

fn cell9_of(cell: &Cell) -> [f64; 9] {
    let [a, b, c] = lattice_of(cell);
    [a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]]
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

fn cell_from_row9(c: [f64; 9], origin: [f64; 3]) -> Result<Cell, SaddleError> {
    cell_from_vecs(
        [c[0], c[1], c[2]],
        [c[3], c[4], c[5]],
        [c[6], c[7], c[8]],
        origin,
    )
}

fn cell_from_vecs(
    a: [f64; 3],
    b: [f64; 3],
    c: [f64; 3],
    origin: [f64; 3],
) -> Result<Cell, SaddleError> {
    Cell::from_vectors(a, b, c, origin).map_err(|_| SaddleError::Shape("singular cell".into()))
}

fn angle_deg(u: [f64; 3], v: [f64; 3]) -> f64 {
    let nu = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
    let nv = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if nu < 1e-18 || nv < 1e-18 {
        return 0.0;
    }
    let c = ((u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (nu * nv)).clamp(-1.0, 1.0);
    c.acos() * 180.0 / std::f64::consts::PI
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expm_logm_roundtrip_near_identity() {
        let u = [[0.1, 0.02, 0.0], [0.0, -0.05, 0.01], [0.0, 0.0, 0.03]];
        let f = expm_3x3(u);
        let back = logm_3x3(f).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (back[i][j] - u[i][j]).abs() < 1e-8,
                    "logm(expm)[{i},{j}]={} vs {}",
                    back[i][j],
                    u[i][j]
                );
            }
        }
    }

    #[test]
    fn expm_of_zero_is_identity() {
        let f = expm_3x3([[0.0; 3]; 3]);
        assert!((f[0][0] - 1.0).abs() < 1e-14);
        assert!(f[0][1].abs() < 1e-14);
    }
}
