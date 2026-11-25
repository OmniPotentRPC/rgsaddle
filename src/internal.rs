//! Sella `Translation`, `Rotation`, and `Displacement` internals.
//!
//! Bond / angle / dihedral primitives live in vocn. These three
//! complete the Sella `internal.py` coordinate set that molecular
//! IRC and equality constraints need: fragment COM, Kabsch
//! quaternion rotation against a reference, and a quadratic
//! displacement form. Ambient reductions go through
//! [`rgmin::vecops`].
//!
//! Each coordinate is the level set `{y | c(y) = c(x)}`:
//! [`rgmin::Manifold`] `project` onto `ker(grad c)`, `retract` by a
//! tangent step plus Gauss-Newton restore, `transport` by projecting
//! at the arrival point.

use ndarray::{Array1, Array2, ArrayView1};
use rgmin::Manifold;
use rgmin::vecops::{axpy, dot, nrm2, sum};

const RESTORE_ITERS: usize = 12;
const RESTORE_TOL: f64 = 1e-12;
const RANK_TOL: f64 = 1e-10;

/// Cartesian axis a translation or rotation coordinate tracks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CartAxis {
    /// Cartesian x axis (slot 0).
    X,
    /// Cartesian y axis (slot 1).
    Y,
    /// Cartesian z axis (slot 2).
    Z,
}

impl CartAxis {
    /// The three axes in Sella order.
    pub const ALL: [Self; 3] = [Self::X, Self::Y, Self::Z];

    /// Packed Cartesian slot: `X = 0`, `Y = 1`, `Z = 2`.
    pub const fn index(self) -> usize {
        match self {
            Self::X => 0,
            Self::Y => 1,
            Self::Z => 2,
        }
    }
}

/// Mean Cartesian coordinate of a fragment along one axis.
///
/// Sella `internal.Translation`: `pos[:, dim].mean()`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Translation {
    indices: Vec<usize>,
    axis: CartAxis,
}

impl Translation {
    /// Fragment `indices` (atom ids) along `axis`.
    pub fn new(indices: Vec<usize>, axis: CartAxis) -> Result<Self, crate::SaddleError> {
        if indices.is_empty() {
            return Err(crate::SaddleError::Shape(
                "translation needs at least one atom".into(),
            ));
        }
        Ok(Self { indices, axis })
    }

    /// Every atom of an `n_atoms` frame.
    pub fn all(n_atoms: usize, axis: CartAxis) -> Result<Self, crate::SaddleError> {
        Self::new((0..n_atoms).collect(), axis)
    }

    /// Atom ids this coordinate reads.
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Tracked axis.
    pub fn axis(&self) -> CartAxis {
        self.axis
    }

    /// Sella `_translation`: mean of `pos[i, dim]`.
    pub fn value(&self, x: ArrayView1<f64>) -> Result<f64, crate::SaddleError> {
        let n = self.indices.len() as f64;
        let mut coords = Array1::zeros(self.indices.len());
        for (k, &i) in self.indices.iter().enumerate() {
            coords[k] = atom_coord(x, i, self.axis.index())?;
        }
        Ok(sum(coords.view()) / n)
    }

    /// Cartesian gradient, length `x`. Uniform `1/n` on the axis slots.
    pub fn gradient(&self, x: ArrayView1<f64>) -> Result<Array1<f64>, crate::SaddleError> {
        check_cart(x)?;
        let mut g = Array1::zeros(x.len());
        let w = 1.0 / self.indices.len() as f64;
        let ax = self.axis.index();
        for &i in &self.indices {
            let slot = 3 * i + ax;
            if slot >= x.len() {
                return Err(crate::SaddleError::Shape(
                    "translation atom index exceeds the 3N frame".into(),
                ));
            }
            g[slot] = w;
        }
        Ok(g)
    }
}

/// Kabsch quaternion rotation of a fragment against a centered reference.
///
/// Sella `internal.Rotation`: `2 q[axis+1] asinc(q[0])`, with `q` the
/// rightmost eigenvector of the 4x4 F-matrix. `refpos` is stored
/// centroid-free.
#[derive(Clone, Debug, PartialEq)]
pub struct Rotation {
    indices: Vec<usize>,
    axis: CartAxis,
    refpos: Array2<f64>,
}

impl Rotation {
    /// Fragment `indices`, rotation `axis`, reference Cartesian of
    /// those atoms (`n x 3`). The reference is centered on entry.
    pub fn new(
        indices: Vec<usize>,
        axis: CartAxis,
        refpos: Array2<f64>,
    ) -> Result<Self, crate::SaddleError> {
        if indices.len() < 2 {
            return Err(crate::SaddleError::Shape(
                "rotation needs at least two atoms".into(),
            ));
        }
        if refpos.nrows() != indices.len() || refpos.ncols() != 3 {
            return Err(crate::SaddleError::Shape(
                "rotation refpos must be n_indices x 3".into(),
            ));
        }
        let mut centered = refpos;
        center_rows(&mut centered);
        Ok(Self {
            indices,
            axis,
            refpos: centered,
        })
    }

    /// Atom ids this coordinate reads.
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Tracked axis.
    pub fn axis(&self) -> CartAxis {
        self.axis
    }

    /// Centered reference, `n x 3`.
    pub fn refpos(&self) -> &Array2<f64> {
        &self.refpos
    }

    /// Sella `_rotation`.
    pub fn value(&self, x: ArrayView1<f64>) -> Result<f64, crate::SaddleError> {
        let q = self.quaternion(x)?;
        Ok(2.0 * q[self.axis.index() + 1] * asinc(q[0]))
    }

    /// Cartesian gradient of the rotation coordinate, length `x`.
    pub fn gradient(&self, x: ArrayView1<f64>) -> Result<Array1<f64>, crate::SaddleError> {
        check_cart(x)?;
        let (lam, q) = self.f_eigen(x)?;
        let f = self.f_matrix(x)?;
        let a = asinc(q[0]);
        let da = asinc_deriv(q[0]);
        let axis = self.axis.index() + 1;
        let mut g = Array1::zeros(x.len());
        for (local, &atom) in self.indices.iter().enumerate() {
            for dim in 0..3 {
                let df = self.df_dpos(local, dim);
                let dq = eigvec_jvp(&f, lam, &q, &df);
                let dval = 2.0 * (dq[axis] * a + q[axis] * da * dq[0]);
                let slot = 3 * atom + dim;
                if slot >= x.len() {
                    return Err(crate::SaddleError::Shape(
                        "rotation atom index exceeds the 3N frame".into(),
                    ));
                }
                g[slot] = if dval.is_finite() { dval } else { 0.0 };
            }
        }
        Ok(g)
    }

    fn gather(&self, x: ArrayView1<f64>) -> Result<Array2<f64>, crate::SaddleError> {
        let mut pos = Array2::zeros((self.indices.len(), 3));
        for (local, &atom) in self.indices.iter().enumerate() {
            for dim in 0..3 {
                pos[(local, dim)] = atom_coord(x, atom, dim)?;
            }
        }
        Ok(pos)
    }

    /// Sign-fixed Kabsch quaternion for the reference fragment.
    pub fn quaternion(&self, x: ArrayView1<f64>) -> Result<[f64; 4], crate::SaddleError> {
        let (_, q) = self.f_eigen(x)?;
        Ok(q)
    }

    fn f_eigen(&self, x: ArrayView1<f64>) -> Result<(f64, [f64; 4]), crate::SaddleError> {
        let f = self.f_matrix(x)?;
        Ok(max_kabsch_quaternion(&f))
    }

    fn f_matrix(&self, x: ArrayView1<f64>) -> Result<[[f64; 4]; 4], crate::SaddleError> {
        let mut pos = self.gather(x)?;
        center_rows(&mut pos);
        Ok(kabsch_f(&pos, &self.refpos))
    }

    /// `dF / d pos[local, dim]`. `refpos` is centered, so
    /// `dR_{αβ} / d pos_{kγ} = δ_{αγ} ref_{kβ}`.
    fn df_dpos(&self, local: usize, dim: usize) -> [[f64; 4]; 4] {
        let mut d_r = [[0.0; 3]; 3];
        for (beta, value) in d_r[dim].iter_mut().enumerate() {
            *value = self.refpos[(local, beta)];
        }
        let d_rtr = self.refpos[(local, dim)];
        let d_ftop = [
            d_r[1][2] - d_r[2][1],
            d_r[2][0] - d_r[0][2],
            d_r[0][1] - d_r[1][0],
        ];
        let mut df = [[0.0; 4]; 4];
        df[0][0] = d_rtr;
        for k in 0..3 {
            df[0][k + 1] = d_ftop[k];
            df[k + 1][0] = d_ftop[k];
            df[k + 1][k + 1] = -d_rtr;
            for j in 0..3 {
                df[k + 1][j + 1] += d_r[k][j] + d_r[j][k];
            }
        }
        df
    }
}

/// Quadratic displacement `dx^T W dx` from a reference geometry.
///
/// Sella `internal.Displacement`. `W` is `3n x 3n` on the fragment
/// packing.
#[derive(Clone, Debug, PartialEq)]
pub struct Displacement {
    indices: Vec<usize>,
    refpos: Array2<f64>,
    w: Array2<f64>,
}

impl Displacement {
    /// Fragment `indices`, reference Cartesian `n x 3`, weight `3n x 3n`.
    pub fn new(
        indices: Vec<usize>,
        refpos: Array2<f64>,
        w: Array2<f64>,
    ) -> Result<Self, crate::SaddleError> {
        if indices.is_empty() {
            return Err(crate::SaddleError::Shape(
                "displacement needs at least one atom".into(),
            ));
        }
        if refpos.nrows() != indices.len() || refpos.ncols() != 3 {
            return Err(crate::SaddleError::Shape(
                "displacement refpos must be n_indices x 3".into(),
            ));
        }
        let n3 = 3 * indices.len();
        if w.nrows() != n3 || w.ncols() != n3 {
            return Err(crate::SaddleError::Shape(
                "displacement W must be 3n_indices x 3n_indices".into(),
            ));
        }
        Ok(Self { indices, refpos, w })
    }

    /// Identity-weighted displacement from `refpos`.
    pub fn identity(indices: Vec<usize>, refpos: Array2<f64>) -> Result<Self, crate::SaddleError> {
        let n3 = 3 * indices.len();
        Self::new(indices, refpos, Array2::eye(n3))
    }

    /// Atom ids this coordinate reads.
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Sella `_displacement`.
    pub fn value(&self, x: ArrayView1<f64>) -> Result<f64, crate::SaddleError> {
        let dx = self.dx(x)?;
        let wdx = self.w.dot(&dx);
        Ok(dot(dx.view(), wdx.view()))
    }

    /// Cartesian gradient `2 W dx`, scattered into the 3N frame.
    pub fn gradient(&self, x: ArrayView1<f64>) -> Result<Array1<f64>, crate::SaddleError> {
        check_cart(x)?;
        let dx = self.dx(x)?;
        let mut wdx = self.w.dot(&dx);
        wdx.mapv_inplace(|v| 2.0 * v);
        let mut g = Array1::zeros(x.len());
        for (local, &atom) in self.indices.iter().enumerate() {
            for dim in 0..3 {
                let slot = 3 * atom + dim;
                if slot >= x.len() {
                    return Err(crate::SaddleError::Shape(
                        "displacement atom index exceeds the 3N frame".into(),
                    ));
                }
                g[slot] = wdx[3 * local + dim];
            }
        }
        Ok(g)
    }

    fn dx(&self, x: ArrayView1<f64>) -> Result<Array1<f64>, crate::SaddleError> {
        let n = self.indices.len();
        let mut dx = Array1::zeros(3 * n);
        for (local, &atom) in self.indices.iter().enumerate() {
            for dim in 0..3 {
                dx[3 * local + dim] = atom_coord(x, atom, dim)? - self.refpos[(local, dim)];
            }
        }
        Ok(dx)
    }
}

impl Manifold for Translation {
    fn required_dim(&self, n: usize) -> Result<(), usize> {
        let need = packed_need(&self.indices);
        if n >= need && n.is_multiple_of(3) {
            Ok(())
        } else {
            Err(need)
        }
    }

    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        match self.gradient(x.view()) {
            Ok(g) => project_ker(&g, v),
            Err(_) => v.clone(),
        }
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        restore_level(x, v, |z| self.value(z), |z| self.gradient(z))
    }

    fn transport(&self, _x_from: &Array1<f64>, x_to: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        self.project(x_to, v)
    }
}

impl Manifold for Rotation {
    fn required_dim(&self, n: usize) -> Result<(), usize> {
        let need = packed_need(&self.indices);
        if n >= need && n.is_multiple_of(3) {
            Ok(())
        } else {
            Err(need)
        }
    }

    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        match self.gradient(x.view()) {
            Ok(g) => project_ker(&g, v),
            Err(_) => v.clone(),
        }
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        restore_level(x, v, |z| self.value(z), |z| self.gradient(z))
    }

    fn transport(&self, _x_from: &Array1<f64>, x_to: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        self.project(x_to, v)
    }
}

impl Manifold for Displacement {
    fn required_dim(&self, n: usize) -> Result<(), usize> {
        let need = packed_need(&self.indices);
        if n >= need && n.is_multiple_of(3) {
            Ok(())
        } else {
            Err(need)
        }
    }

    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        match self.gradient(x.view()) {
            Ok(g) => project_ker(&g, v),
            Err(_) => v.clone(),
        }
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        restore_level(x, v, |z| self.value(z), |z| self.gradient(z))
    }

    fn transport(&self, _x_from: &Array1<f64>, x_to: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        self.project(x_to, v)
    }
}

/// Which Sella internal a host packed into an internals chart.
///
/// Bond / angle / dihedral counts come from vocn; these three are
/// the remaining Sella slots MaxInternalStep weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InternalSlot {
    /// Rigid translation coordinate.
    Translation,
    /// Pairwise bond-length coordinate.
    Bond,
    /// Three-atom valence angle coordinate.
    Angle,
    /// Four-atom torsion dihedral coordinate.
    Dihedral,
    /// Non-standard internal coordinate such as [`Displacement`].
    Other,
    /// Kabsch quaternion rotation coordinate.
    Rotation,
}

fn packed_need(indices: &[usize]) -> usize {
    3 * (indices.iter().copied().max().unwrap_or(0) + 1)
}

/// Ambient projection onto `ker(g)`.
fn project_ker(g: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
    if g.len() != v.len() {
        return v.clone();
    }
    let gg = dot(g.view(), g.view());
    if gg <= RANK_TOL * RANK_TOL {
        return v.clone();
    }
    let mut out = v.clone();
    axpy(-dot(g.view(), v.view()) / gg, g.view(), &mut out);
    out
}

/// `x + v` then Gauss-Newton restore onto `{y | c(y) = c(x)}`.
fn restore_level<F, G>(x: &Array1<f64>, v: &Array1<f64>, value: F, gradient: G) -> Array1<f64>
where
    F: Fn(ArrayView1<f64>) -> Result<f64, crate::SaddleError>,
    G: Fn(ArrayView1<f64>) -> Result<Array1<f64>, crate::SaddleError>,
{
    if v.len() != x.len() {
        return x.clone();
    }
    let Ok(target) = value(x.view()) else {
        return x.clone();
    };
    let mut y = x.clone();
    axpy(1.0, v.view(), &mut y);
    for _ in 0..RESTORE_ITERS {
        let Ok(c) = value(y.view()) else {
            break;
        };
        let r = c - target;
        if r.abs() <= RESTORE_TOL {
            break;
        }
        let Ok(g) = gradient(y.view()) else {
            break;
        };
        let gg = dot(g.view(), g.view());
        if gg <= RANK_TOL * RANK_TOL {
            break;
        }
        axpy(-r / gg, g.view(), &mut y);
    }
    y
}

fn check_cart(x: ArrayView1<f64>) -> Result<(), crate::SaddleError> {
    if x.is_empty() || !x.len().is_multiple_of(3) {
        return Err(crate::SaddleError::Shape(
            "internals need a nonzero 3N Cartesian".into(),
        ));
    }
    Ok(())
}

fn atom_coord(x: ArrayView1<f64>, atom: usize, dim: usize) -> Result<f64, crate::SaddleError> {
    check_cart(x)?;
    let slot = 3 * atom + dim;
    if slot >= x.len() {
        return Err(crate::SaddleError::Shape(
            "atom index exceeds the 3N frame".into(),
        ));
    }
    Ok(x[slot])
}

fn center_rows(pos: &mut Array2<f64>) {
    let n = pos.nrows() as f64;
    if n == 0.0 {
        return;
    }
    for d in 0..3 {
        let c = sum(pos.column(d)) / n;
        for i in 0..pos.nrows() {
            pos[(i, d)] -= c;
        }
    }
}

/// Sella `R = dx.T @ refpos`. The rightmost quaternion is the inverse
/// Kabsch rotation: a +axis body rotation reads negative.
pub(crate) fn kabsch_f(pos: &Array2<f64>, refpos: &Array2<f64>) -> [[f64; 4]; 4] {
    let mut r = [[0.0; 3]; 3];
    for i in 0..pos.nrows() {
        for a in 0..3 {
            for b in 0..3 {
                r[a][b] += pos[(i, a)] * refpos[(i, b)];
            }
        }
    }
    let rtr = r[0][0] + r[1][1] + r[2][2];
    let ftop = [r[1][2] - r[2][1], r[2][0] - r[0][2], r[0][1] - r[1][0]];
    let mut f = [[0.0; 4]; 4];
    f[0][0] = rtr;
    for k in 0..3 {
        f[0][k + 1] = ftop[k];
        f[k + 1][0] = ftop[k];
        f[k + 1][k + 1] = -rtr;
        for j in 0..3 {
            f[k + 1][j + 1] += r[k][j] + r[j][k];
        }
    }
    f
}

/// Sella `asinc`: `arccos(x) / sqrt(1-x^2)`, Taylor near `x = 1`.
fn asinc(x: f64) -> f64 {
    if x < 0.97 {
        let c = x.clamp(-1.0, 0.97);
        c.acos() / (1.0 - c * c).sqrt()
    } else {
        asinc_taylor(x)
    }
}

fn asinc_taylor(x: f64) -> f64 {
    let y = x - 1.0;
    1.0 - y / 3.0 + 2.0 * y * y / 15.0 - 2.0 * y.powi(3) / 35.0 + 8.0 * y.powi(4) / 315.0
        - 8.0 * y.powi(5) / 693.0
        + 16.0 * y.powi(6) / 3003.0
        - 16.0 * y.powi(7) / 6435.0
        + 128.0 * y.powi(8) / 109395.0
        - 128.0 * y.powi(9) / 230945.0
}

fn asinc_deriv(x: f64) -> f64 {
    let h = 1e-6;
    (asinc(x + h) - asinc(x - h)) / (2.0 * h)
}

/// Gap under which two eigenvalues of the Kabsch F-matrix count as equal.
///
/// The F-matrix is quadratic in the coordinates, so its eigenvalues scale
/// with the square of the fragment size. The gap is therefore relative to
/// the largest eigenvalue magnitude and the same fragment read in any
/// length unit selects the same eigenspace.
fn degeneracy_tolerance(evals: &[f64; 4]) -> f64 {
    1e-10
        * evals
            .iter()
            .fold(0.0_f64, |scale, value| scale.max(value.abs()))
}

fn max_kabsch_quaternion(f: &[[f64; 4]; 4]) -> (f64, [f64; 4]) {
    let (evals, evecs) = jacobi4(*f);
    let mut imax = 0;
    for i in 1..4 {
        if evals[i] > evals[imax] {
            imax = i;
        }
    }
    let lam = evals[imax];
    let tol = degeneracy_tolerance(&evals);
    let mut q = [0.0; 4];
    for j in 0..4 {
        if (lam - evals[j]).abs() <= tol {
            let weight = evecs[0][j];
            for (qi, row) in q.iter_mut().zip(&evecs) {
                *qi += weight * row[j];
            }
        }
    }
    let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if norm > 1e-12 {
        for qi in &mut q {
            *qi /= norm;
        }
    } else {
        q = [
            evecs[0][imax],
            evecs[1][imax],
            evecs[2][imax],
            evecs[3][imax],
        ];
    }
    if q[0] < 0.0 {
        for v in &mut q {
            *v = -*v;
        }
    }
    (lam, q)
}

/// Rightmost-eigenvector JVP: `pinv(λI - F) (dF q - (q^T dF q) q)`.
fn eigvec_jvp(f: &[[f64; 4]; 4], lam: f64, q: &[f64; 4], df: &[[f64; 4]; 4]) -> [f64; 4] {
    let (evals, evecs) = jacobi4(*f);
    let tol = degeneracy_tolerance(&evals);
    let mut dfq = [0.0; 4];
    for i in 0..4 {
        for (dfij, qj) in df[i].iter().zip(q) {
            dfq[i] += dfij * qj;
        }
    }
    let mut dq = [0.0; 4];
    let mut g_e0 = [0.0; 4];
    for (j, &eval) in evals.iter().enumerate() {
        let gap = lam - eval;
        if gap <= tol {
            continue;
        }
        let vj = [evecs[0][j], evecs[1][j], evecs[2][j], evecs[3][j]];
        let v_dot_dfq = vj[0] * dfq[0] + vj[1] * dfq[1] + vj[2] * dfq[2] + vj[3] * dfq[3];
        let coeff = v_dot_dfq / gap;
        let coeff_e0 = vj[0] / gap;
        for i in 0..4 {
            dq[i] += coeff * vj[i];
            g_e0[i] += coeff_e0 * vj[i];
        }
    }
    if q[0].abs() > 1e-10 {
        let mut df_ge0 = [0.0; 4];
        for i in 0..4 {
            for (dfij, gk) in df[i].iter().zip(&g_e0) {
                df_ge0[i] += dfij * gk;
            }
        }
        for (j, &eval) in evals.iter().enumerate() {
            if (lam - eval).abs() <= tol {
                let vj = [evecs[0][j], evecs[1][j], evecs[2][j], evecs[3][j]];
                let proj =
                    (vj[0] * df_ge0[0] + vj[1] * df_ge0[1] + vj[2] * df_ge0[2] + vj[3] * df_ge0[3])
                        / q[0];
                for i in 0..4 {
                    dq[i] += proj * vj[i];
                }
            }
        }
    }
    let proj = dq[0] * q[0] + dq[1] * q[1] + dq[2] * q[2] + dq[3] * q[3];
    for i in 0..4 {
        dq[i] -= proj * q[i];
    }
    dq
}

pub(crate) fn jacobi4(mut a: [[f64; 4]; 4]) -> ([f64; 4], [[f64; 4]; 4]) {
    let mut v = [[0.0; 4]; 4];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _ in 0..32 {
        let mut off = 0.0;
        for (p, row) in a.iter().enumerate() {
            for value in row.iter().skip(p + 1) {
                off += value * value;
            }
        }
        let size = a
            .iter()
            .flatten()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        if off.sqrt() <= 1e-15 * size.min(1.0) {
            break;
        }
        for p in 0..4 {
            for q in (p + 1)..4 {
                let apq = a[p][q];
                if apq.abs() <= 1e-300 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * apq);
                let sign = if theta >= 0.0 { 1.0 } else { -1.0 };
                let t = sign / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for row in &mut a {
                    let aip = row[p];
                    let aiq = row[q];
                    row[p] = c * aip - s * aiq;
                    row[q] = s * aip + c * aiq;
                }
                let (before_q, from_q) = a.split_at_mut(q);
                for (api, aqi) in before_q[p].iter_mut().zip(&mut from_q[0]) {
                    let old_p = *api;
                    let old_q = *aqi;
                    *api = c * old_p - s * old_q;
                    *aqi = s * old_p + c * old_q;
                }
                for row in &mut v {
                    let vip = row[p];
                    let viq = row[q];
                    row[p] = c * vip - s * viq;
                    row[q] = s * vip + c * viq;
                }
            }
        }
    }
    ([a[0][0], a[1][1], a[2][2], a[3][3]], v)
}

/// Pack a 3N Cartesian from `n` atom xyz rows.
pub fn pack_cart(rows: &[[f64; 3]]) -> Array1<f64> {
    let mut x = Array1::zeros(3 * rows.len());
    for (i, p) in rows.iter().enumerate() {
        x[3 * i] = p[0];
        x[3 * i + 1] = p[1];
        x[3 * i + 2] = p[2];
    }
    x
}

/// Euclidean length through the vecops seam.
pub fn seam_nrm2(x: &Array1<f64>) -> f64 {
    nrm2(x.view())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use rgmin::Manifold;

    fn triangle() -> (Array1<f64>, Array2<f64>) {
        let rows = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.2, 0.8, 0.0]];
        let x = pack_cart(&rows);
        let mut refpos = Array2::zeros((3, 3));
        for i in 0..3 {
            for d in 0..3 {
                refpos[(i, d)] = rows[i][d];
            }
        }
        (x, refpos)
    }

    fn rot_z(p: [f64; 3], th: f64) -> [f64; 3] {
        let (s, c) = (th.sin(), th.cos());
        [c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]]
    }

    fn fd_grad<F: Fn(ArrayView1<f64>) -> f64>(f: F, x: &Array1<f64>, h: f64) -> Array1<f64> {
        let mut g = Array1::zeros(x.len());
        let mut xp = x.clone();
        for i in 0..x.len() {
            xp[i] = x[i] + h;
            let fp = f(xp.view());
            xp[i] = x[i] - h;
            let fm = f(xp.view());
            xp[i] = x[i];
            g[i] = (fp - fm) / (2.0 * h);
        }
        g
    }

    #[test]
    fn translation_is_the_fragment_mean() {
        let x = pack_cart(&[[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]]);
        let t = Translation::all(2, CartAxis::Y).unwrap();
        assert!((t.value(x.view()).unwrap() - 2.0).abs() < 1e-14);
        let g = t.gradient(x.view()).unwrap();
        assert!((g[1] - 0.5).abs() < 1e-14);
        assert!((g[4] - 0.5).abs() < 1e-14);
        assert!(g[0].abs() < 1e-14);
    }

    #[test]
    fn rotation_about_z_reads_the_angle() {
        let (x0, refpos) = triangle();
        let th = 0.15;
        let rows = [
            rot_z([0.0, 0.0, 0.0], th),
            rot_z([1.0, 0.0, 0.0], th),
            rot_z([0.2, 0.8, 0.0], th),
        ];
        let x = pack_cart(&rows);
        let rz = Rotation::new(vec![0, 1, 2], CartAxis::Z, refpos.clone()).unwrap();
        let val = rz.value(x.view()).unwrap();
        // Sella `R = dx.T @ refpos` is the inverse Kabsch angle: a
        // +Z body rotation reads `-theta`.
        assert!(
            (val + th).abs() < 1e-8,
            "rotation Z = {val}, expected {}; x0={x0:?}",
            -th
        );
        let rx = Rotation::new(vec![0, 1, 2], CartAxis::X, refpos.clone()).unwrap();
        let ry = Rotation::new(vec![0, 1, 2], CartAxis::Y, refpos).unwrap();
        assert!(rx.value(x.view()).unwrap().abs() < 1e-8);
        assert!(ry.value(x.view()).unwrap().abs() < 1e-8);
    }

    #[test]
    fn rotation_gradient_matches_central_difference() {
        let (_, refpos) = triangle();
        let th = 0.08;
        let rows = [
            rot_z([0.0, 0.0, 0.0], th),
            rot_z([1.0, 0.0, 0.0], th),
            rot_z([0.2, 0.8, 0.0], th),
        ];
        let x = pack_cart(&rows);
        let rz = Rotation::new(vec![0, 1, 2], CartAxis::Z, refpos).unwrap();
        let g = rz.gradient(x.view()).unwrap();
        let gfd = fd_grad(|v| rz.value(v).unwrap(), &x, 1e-6);
        let err = seam_nrm2(&(&g - &gfd));
        assert!(
            err < 2e-6,
            "rotation gradient err={err} anal={g:?} fd={gfd:?}"
        );
    }

    #[test]
    fn displacement_is_the_weighted_square() {
        let refpos = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let d = Displacement::identity(vec![0, 1], refpos).unwrap();
        let x = pack_cart(&[[0.1, 0.0, 0.0], [1.0, 0.2, 0.0]]);
        let val = d.value(x.view()).unwrap();
        assert!((val - (0.1 * 0.1 + 0.2 * 0.2)).abs() < 1e-14);
        let g = d.gradient(x.view()).unwrap();
        assert!((g[0] - 0.2).abs() < 1e-14);
        assert!((g[4] - 0.4).abs() < 1e-14);
    }

    #[test]
    fn translation_rejects_an_empty_fragment() {
        match Translation::new(vec![], CartAxis::X) {
            Err(crate::SaddleError::Shape(_)) => {}
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn translation_retract_stays_on_the_set() {
        let x = pack_cart(&[[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]]);
        let t = Translation::all(2, CartAxis::Y).unwrap();
        let c0 = t.value(x.view()).unwrap();
        let step = Array1::from_elem(6, 0.3);
        let y = t.retract(&x, &step);
        let c1 = t.value(y.view()).unwrap();
        assert!(
            (c0 - c1).abs() < 1e-12,
            "translation retract left the set: {c0} -> {c1}"
        );
        let g = t.gradient(x.view()).unwrap();
        let v = t.project(&x, &step);
        assert!(dot(g.view(), v.view()).abs() < 1e-12);
        let shift = Array1::from_vec(vec![0.0, 0.3, 0.0, 0.0, 0.3, 0.0]);
        let v_shift = t.project(&x, &shift);
        assert!(
            seam_nrm2(&v_shift) < 1e-12,
            "uniform Y shift must be vertical"
        );
        let w = t.transport(&x, &y, &v);
        let w_h = t.project(&y, &w);
        assert!(seam_nrm2(&(&w - &w_h)) < 1e-12);
        assert!(t.required_dim(6).is_ok());
        assert!(t.required_dim(5).is_err());
    }

    #[test]
    fn rotation_retract_stays_on_the_set() {
        let (_, refpos) = triangle();
        let th = 0.08;
        let rows = [
            rot_z([0.0, 0.0, 0.0], th),
            rot_z([1.0, 0.0, 0.0], th),
            rot_z([0.2, 0.8, 0.0], th),
        ];
        let x = pack_cart(&rows);
        let rz = Rotation::new(vec![0, 1, 2], CartAxis::Z, refpos).unwrap();
        let q = rz.quaternion(x.view()).unwrap();
        let qn = nrm2(Array1::from_vec(q.to_vec()).view());
        assert!((qn - 1.0).abs() < 1e-12, "quaternion left S^3: {q:?}");
        let c0 = rz.value(x.view()).unwrap();
        let mut step = Array1::zeros(9);
        step[0] = 0.05;
        step[1] = -0.04;
        step[3] = 0.02;
        step[4] = 0.03;
        let y = rz.retract(&x, &step);
        let c1 = rz.value(y.view()).unwrap();
        assert!(
            (c0 - c1).abs() < 1e-8,
            "rotation retract left the set: {c0} -> {c1}"
        );
        let g = rz.gradient(x.view()).unwrap();
        let v = rz.project(&x, &step);
        assert!(dot(g.view(), v.view()).abs() < 1e-10);
        let w = rz.transport(&x, &y, &v);
        let w_h = rz.project(&y, &w);
        assert!(seam_nrm2(&(&w - &w_h)) < 1e-10);
        let mut shifted = x.clone();
        for i in 0..3 {
            shifted[3 * i] += 0.4;
            shifted[3 * i + 1] -= 0.2;
        }
        assert!((rz.value(shifted.view()).unwrap() - c0).abs() < 1e-10);
    }

    fn collinear() -> (Array2<f64>, [[f64; 3]; 3]) {
        let rows = [[-0.9, 0.0, 0.0], [0.1, 0.0, 0.0], [1.3, 0.0, 0.0]];
        let mut refpos = Array2::zeros((3, 3));
        for i in 0..3 {
            for d in 0..3 {
                refpos[(i, d)] = rows[i][d];
            }
        }
        (refpos, rows)
    }

    /// A collinear fragment has no preferred twist about its own axis, so
    /// the top eigenvalue of the Kabsch F-matrix is doubly degenerate. The
    /// quaternion must be the member of that eigenspace closest to the
    /// identity, which carries no twist, instead of whatever vector the
    /// Jacobi sweep returns first. The eigenvalues grow with the square of
    /// the fragment size, so the check runs over coordinates from 1e-8 to
    /// 1e4: an absolute degeneracy gap lumps every eigenvalue of the small
    /// fragments together and reads their turn as no rotation at all.
    #[test]
    fn rotation_of_a_collinear_fragment_has_no_twist() {
        let (_, rows) = collinear();
        let th = 0.3;
        for scale in [1e-8, 1e-5, 1e-2, 1.0, 1e4] {
            let fixed = scaled_rows(&rows, scale);
            let refpos = Array2::from_shape_fn((3, 3), |(i, d)| fixed[i][d]);
            let turned: Vec<[f64; 3]> = fixed.iter().map(|&p| rot_z(p, th)).collect();
            let x = pack_cart(&turned);
            let rx = Rotation::new(vec![0, 1, 2], CartAxis::X, refpos.clone()).unwrap();
            let ry = Rotation::new(vec![0, 1, 2], CartAxis::Y, refpos.clone()).unwrap();
            let rz = Rotation::new(vec![0, 1, 2], CartAxis::Z, refpos).unwrap();
            let (vx, vy, vz) = (
                rx.value(x.view()).unwrap(),
                ry.value(x.view()).unwrap(),
                rz.value(x.view()).unwrap(),
            );
            assert!(
                (vz + th).abs() < 1e-8,
                "scale {scale}: rotation Z = {vz}, expected {}",
                -th
            );
            assert!(
                vx.abs() < 1e-8,
                "scale {scale}: twist about the fragment axis: {vx}"
            );
            assert!(
                vy.abs() < 1e-8,
                "scale {scale}: spurious tilt out of the plane: {vy}"
            );
            let q = rz.quaternion(x.view()).unwrap();
            let expected = [(0.5 * th).cos(), 0.0, 0.0, -(0.5 * th).sin()];
            for (got, want) in q.iter().zip(expected) {
                assert!(
                    (got - want).abs() < 1e-8,
                    "scale {scale}: quaternion {q:?}, expected {expected:?}"
                );
            }
        }
    }

    #[test]
    fn rotation_gradient_of_a_collinear_fragment_matches_central_difference() {
        let (_, rows) = collinear();
        let th = 0.2;
        for scale in [1e-2, 1.0, 1e3] {
            let fixed = scaled_rows(&rows, scale);
            let refpos = Array2::from_shape_fn((3, 3), |(i, d)| fixed[i][d]);
            let turned: Vec<[f64; 3]> = fixed.iter().map(|&p| rot_z(p, th)).collect();
            let x = pack_cart(&turned);
            let rz = Rotation::new(vec![0, 1, 2], CartAxis::Z, refpos).unwrap();
            let g = rz.gradient(x.view()).unwrap();
            let gfd = fd_grad(|v| rz.value(v).unwrap(), &x, 1e-6 * scale);
            let err = seam_nrm2(&(&g - &gfd)) * scale;
            assert!(
                err < 2e-5,
                "scale {scale}: collinear rotation gradient err={err} anal={g:?} fd={gfd:?}"
            );
            assert!(
                seam_nrm2(&g) * scale > 0.1,
                "scale {scale}: gradient vanished: {g:?}"
            );
        }
    }

    #[test]
    fn displacement_retract_stays_on_the_set() {
        let refpos = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let d = Displacement::identity(vec![0, 1], refpos).unwrap();
        let x = pack_cart(&[[0.1, 0.0, 0.0], [1.0, 0.2, 0.0]]);
        let c0 = d.value(x.view()).unwrap();
        let step = Array1::from_elem(6, 0.05);
        let y = d.retract(&x, &step);
        let c1 = d.value(y.view()).unwrap();
        assert!(
            (c0 - c1).abs() < 1e-10,
            "displacement retract left the set: {c0} -> {c1}"
        );
        let g = d.gradient(x.view()).unwrap();
        let v = d.project(&x, &step);
        assert!(dot(g.view(), v.view()).abs() < 1e-12);
        let w = d.transport(&x, &y, &v);
        let w_h = d.project(&y, &w);
        assert!(seam_nrm2(&(&w - &w_h)) < 1e-12);
    }

    fn scaled_rows(rows: &[[f64; 3]], scale: f64) -> Vec<[f64; 3]> {
        rows.iter()
            .map(|p| [p[0] * scale, p[1] * scale, p[2] * scale])
            .collect()
    }

    /// The Kabsch eigenspace is selected by a gap relative to the F-matrix
    /// size, so the same fragment turned by the same angle reads the same
    /// rotation and quaternion whether its coordinates are of order 1e-6,
    /// 1, or 1e6.
    #[test]
    fn kabsch_rotation_does_not_depend_on_the_length_scale() {
        let rows = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.2, 0.0],
            [0.0, 1.5, 0.3],
            [0.1, -0.4, 1.1],
        ];
        let th = 0.35;
        let mut reference_quaternion = None;
        for scale in [1e-10, 1e-6, 1e-3, 1.0, 1e3, 1e6] {
            let fixed = scaled_rows(&rows, scale);
            let turned: Vec<[f64; 3]> = fixed.iter().map(|&p| rot_z(p, th)).collect();
            let refpos = Array2::from_shape_fn((4, 3), |(i, d)| fixed[i][d]);
            let x = pack_cart(&turned);
            let rz = Rotation::new(vec![0, 1, 2, 3], CartAxis::Z, refpos).unwrap();
            let value = rz.value(x.view()).unwrap();
            assert!(
                (value + th).abs() < 1e-8,
                "scale {scale}: rotation Z = {value}, expected {}",
                -th
            );
            let q = rz.quaternion(x.view()).unwrap();
            let expected = [(0.5 * th).cos(), 0.0, 0.0, -(0.5 * th).sin()];
            for (got, want) in q.iter().zip(expected) {
                assert!(
                    (got - want).abs() < 1e-8,
                    "scale {scale}: quaternion {q:?}, expected {expected:?}"
                );
            }
            match reference_quaternion {
                None => reference_quaternion = Some(q),
                Some(first) => {
                    for (got, want) in q.iter().zip(first) {
                        assert!(
                            (got - want).abs() < 1e-8,
                            "scale {scale}: quaternion {q:?} differs from {first:?}"
                        );
                    }
                }
            }
        }
    }

    /// A fragment whose largest F eigenvalue is a single value must not be
    /// treated as degenerate because its coordinates are small.
    #[test]
    fn degeneracy_tolerance_follows_the_matrix_scale() {
        let evals = [4.0, 1.0, -2.0, -3.0];
        let small = evals.map(|value| value * 1e-12);
        assert!(degeneracy_tolerance(&small) < 1e-12 * 4.0 * 1e-9);
        assert!(degeneracy_tolerance(&small) / degeneracy_tolerance(&evals) < 1.1e-12);
        assert_eq!(degeneracy_tolerance(&[0.0; 4]), 0.0);
    }
}
