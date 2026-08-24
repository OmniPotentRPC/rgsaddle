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
    X,
    Y,
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

    fn quaternion(&self, x: ArrayView1<f64>) -> Result<[f64; 4], crate::SaddleError> {
        let (_, q) = self.f_eigen(x)?;
        Ok(q)
    }

    fn f_eigen(&self, x: ArrayView1<f64>) -> Result<(f64, [f64; 4]), crate::SaddleError> {
        let f = self.f_matrix(x)?;
        let (evals, evecs) = jacobi4(f);
        let mut imax = 0;
        for i in 1..4 {
            if evals[i] > evals[imax] {
                imax = i;
            }
        }
        let mut q = [
            evecs[0][imax],
            evecs[1][imax],
            evecs[2][imax],
            evecs[3][imax],
        ];
        if q[0] < 0.0 {
            for v in &mut q {
                *v = -*v;
            }
        }
        Ok((evals[imax], q))
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
        for beta in 0..3 {
            d_r[dim][beta] = self.refpos[(local, beta)];
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
        if n >= need && n % 3 == 0 {
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
        if n >= need && n % 3 == 0 {
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
        if n >= need && n % 3 == 0 {
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
    Translation,
    Bond,
    Angle,
    Dihedral,
    Other,
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
    if x.is_empty() || x.len() % 3 != 0 {
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
fn kabsch_f(pos: &Array2<f64>, refpos: &Array2<f64>) -> [[f64; 4]; 4] {
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

/// Rightmost-eigenvector JVP: `pinv(λI - F) (dF q - (q^T dF q) q)`.
fn eigvec_jvp(f: &[[f64; 4]; 4], lam: f64, q: &[f64; 4], df: &[[f64; 4]; 4]) -> [f64; 4] {
    let mut dfq = [0.0; 4];
    for i in 0..4 {
        for j in 0..4 {
            dfq[i] += df[i][j] * q[j];
        }
    }
    let mut ldot = 0.0;
    for i in 0..4 {
        ldot += q[i] * dfq[i];
    }
    let mut rhs = [0.0; 4];
    for i in 0..4 {
        rhs[i] = dfq[i] - ldot * q[i];
    }
    // `(λI - F + q q^T)` is invertible on the complement of `q`.
    let mut a = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            a[i][j] = -f[i][j] + q[i] * q[j];
        }
        a[i][i] += lam;
    }
    let mut dq = solve4(a, rhs);
    let mut proj = 0.0;
    for i in 0..4 {
        proj += dq[i] * q[i];
    }
    for i in 0..4 {
        dq[i] -= proj * q[i];
    }
    dq
}

fn solve4(mut a: [[f64; 4]; 4], mut b: [f64; 4]) -> [f64; 4] {
    for k in 0..4 {
        let mut piv = k;
        for i in (k + 1)..4 {
            if a[i][k].abs() > a[piv][k].abs() {
                piv = i;
            }
        }
        if a[piv][k].abs() < 1e-16 {
            continue;
        }
        if piv != k {
            a.swap(k, piv);
            b.swap(k, piv);
        }
        let akk = a[k][k];
        for i in (k + 1)..4 {
            let f = a[i][k] / akk;
            for j in k..4 {
                a[i][j] -= f * a[k][j];
            }
            b[i] -= f * b[k];
        }
    }
    let mut x = [0.0; 4];
    for i in (0..4).rev() {
        let mut acc = b[i];
        for j in (i + 1)..4 {
            acc -= a[i][j] * x[j];
        }
        x[i] = if a[i][i].abs() < 1e-16 {
            0.0
        } else {
            acc / a[i][i]
        };
    }
    x
}

fn jacobi4(mut a: [[f64; 4]; 4]) -> ([f64; 4], [[f64; 4]; 4]) {
    let mut v = [[0.0; 4]; 4];
    for i in 0..4 {
        v[i][i] = 1.0;
    }
    for _ in 0..32 {
        let mut off = 0.0;
        for p in 0..4 {
            for q in (p + 1)..4 {
                off += a[p][q] * a[p][q];
            }
        }
        if off.sqrt() <= 1e-15 {
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
                for i in 0..4 {
                    let aip = a[i][p];
                    let aiq = a[i][q];
                    a[i][p] = c * aip - s * aiq;
                    a[i][q] = s * aip + c * aiq;
                }
                for i in 0..4 {
                    let api = a[p][i];
                    let aqi = a[q][i];
                    a[p][i] = c * api - s * aqi;
                    a[q][i] = s * api + c * aqi;
                }
                for i in 0..4 {
                    let vip = v[i][p];
                    let viq = v[i][q];
                    v[i][p] = c * vip - s * viq;
                    v[i][q] = s * vip + c * viq;
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
        let w = t.transport(&x, &y, &v);
        let w_h = t.project(&y, &w);
        assert!(seam_nrm2(&(&w - &w_h)) < 1e-12);
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
}
