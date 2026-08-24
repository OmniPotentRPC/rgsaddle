//! Sella equality constraints on the Cartesian internals chart.
//!
//! `sella.internal.Constraints`: fix translations, rotations, and
//! selected bonds / angles / dihedrals. This is the residual and
//! the tangent projector, not the vocn primitive generator.
//! Bond / angle / dihedral *values* use the same Wilson B-matrix
//! formulas as gpr_optim `InternalCoordinates`; topology and dummy
//! atoms stay in vocn.
//!
//! The level set `{x | c(x) = c_target}` is a [`rgmin::Manifold`]:
//! `project` onto `ker(J)`, `retract` by a tangent step plus
//! Gauss-Newton restore, `transport` by projecting at the arrival
//! point.

use ndarray::{Array1, Array2, ArrayView1};
use rgmin::Manifold;
use rgmin::vecops::{axpy, dot, nrm2};

use crate::SaddleError;
use crate::internal::{CartAxis, Displacement, Rotation, Translation, pack_cart};

const RESTORE_ITERS: usize = 12;
const RESTORE_TOL: f64 = 1e-12;
const RANK_TOL: f64 = 1e-10;

/// One Sella equality: a coordinate plus its target.
#[derive(Clone, Debug)]
pub enum Equality {
    Translation {
        coord: Translation,
        target: f64,
    },
    Rotation {
        coord: Rotation,
        target: f64,
    },
    Displacement {
        coord: Displacement,
        target: f64,
    },
    /// Host-supplied pair. Evaluation is the Wilson bond, not vocn.
    Bond {
        atoms: [usize; 2],
        target: f64,
    },
    /// Host-supplied triple, vertex in the middle.
    Angle {
        atoms: [usize; 3],
        target: f64,
    },
    /// Host-supplied quadruple. Residual is wrapped to `(-pi, pi]`.
    Dihedral {
        atoms: [usize; 4],
        target: f64,
    },
}

impl Equality {
    fn value(&self, x: ArrayView1<f64>) -> Result<f64, SaddleError> {
        match self {
            Self::Translation { coord, .. } => coord.value(x),
            Self::Rotation { coord, .. } => coord.value(x),
            Self::Displacement { coord, .. } => coord.value(x),
            Self::Bond { atoms, .. } => bond_value(x, *atoms),
            Self::Angle { atoms, .. } => angle_value(x, *atoms),
            Self::Dihedral { atoms, .. } => dihedral_value(x, *atoms),
        }
    }

    fn gradient(&self, x: ArrayView1<f64>) -> Result<Array1<f64>, SaddleError> {
        match self {
            Self::Translation { coord, .. } => coord.gradient(x),
            Self::Rotation { coord, .. } => coord.gradient(x),
            Self::Displacement { coord, .. } => coord.gradient(x),
            Self::Bond { atoms, .. } => bond_grad(x, *atoms),
            Self::Angle { atoms, .. } => angle_grad(x, *atoms),
            Self::Dihedral { atoms, .. } => dihedral_grad(x, *atoms),
        }
    }

    fn target(&self) -> f64 {
        match self {
            Self::Translation { target, .. }
            | Self::Rotation { target, .. }
            | Self::Displacement { target, .. }
            | Self::Bond { target, .. }
            | Self::Angle { target, .. }
            | Self::Dihedral { target, .. } => *target,
        }
    }

    fn residual(&self, x: ArrayView1<f64>) -> Result<f64, SaddleError> {
        let raw = self.value(x)? - self.target();
        Ok(match self {
            Self::Dihedral { .. } => wrap_pi(raw),
            _ => raw,
        })
    }

    fn slot(&self) -> crate::internal::InternalSlot {
        use crate::internal::InternalSlot;
        match self {
            Self::Translation { .. } => InternalSlot::Translation,
            Self::Rotation { .. } => InternalSlot::Rotation,
            Self::Displacement { .. } => InternalSlot::Other,
            Self::Bond { .. } => InternalSlot::Bond,
            Self::Angle { .. } => InternalSlot::Angle,
            Self::Dihedral { .. } => InternalSlot::Dihedral,
        }
    }
}

/// Packed Sella internals counts for [`crate::restricted::MaxInternalStep`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InternalCounts {
    pub ntrans: usize,
    pub nbonds: usize,
    pub nangles: usize,
    pub ndihedrals: usize,
    pub nother: usize,
    pub nrotations: usize,
}

impl InternalCounts {
    /// Total active internals in Sella packing order.
    pub fn nint(self) -> usize {
        self.ntrans + self.nbonds + self.nangles + self.ndihedrals + self.nother + self.nrotations
    }
}

/// Equality internals on a 3N Cartesian frame.
#[derive(Clone, Debug)]
pub struct Constraints {
    n_atoms: usize,
    eqs: Vec<Equality>,
    /// Sella `ignore_rotation`: drop rotation residuals from `residual`.
    ignore_rotation: bool,
}

impl Constraints {
    /// Empty set on an `n_atoms` frame.
    pub fn new(n_atoms: usize) -> Result<Self, SaddleError> {
        if n_atoms == 0 {
            return Err(SaddleError::Shape(
                "constraints need at least one atom".into(),
            ));
        }
        Ok(Self {
            n_atoms,
            eqs: Vec::new(),
            ignore_rotation: true,
        })
    }

    /// Keep rotation residuals in `residual` (Sella `ignore_rotation=false`).
    pub fn keep_rotation_residual(mut self) -> Self {
        self.ignore_rotation = false;
        self
    }

    /// Atom count.
    pub fn n_atoms(&self) -> usize {
        self.n_atoms
    }

    /// Active equalities.
    pub fn equalities(&self) -> &[Equality] {
        &self.eqs
    }

    /// Sella packing counts.
    pub fn counts(&self) -> InternalCounts {
        let mut c = InternalCounts::default();
        for eq in &self.eqs {
            match eq.slot() {
                crate::internal::InternalSlot::Translation => c.ntrans += 1,
                crate::internal::InternalSlot::Bond => c.nbonds += 1,
                crate::internal::InternalSlot::Angle => c.nangles += 1,
                crate::internal::InternalSlot::Dihedral => c.ndihedrals += 1,
                crate::internal::InternalSlot::Other => c.nother += 1,
                crate::internal::InternalSlot::Rotation => c.nrotations += 1,
            }
        }
        c
    }

    /// Fix one translation. Target defaults to the current value.
    pub fn fix_translation(
        &mut self,
        coord: Translation,
        x: ArrayView1<f64>,
        target: Option<f64>,
    ) -> Result<(), SaddleError> {
        self.check_frame(x)?;
        let target = match target {
            Some(t) => t,
            None => coord.value(x)?,
        };
        self.eqs.push(Equality::Translation { coord, target });
        Ok(())
    }

    /// Fix the fragment (or whole-molecule) COM on every axis.
    pub fn fix_com(&mut self, x: ArrayView1<f64>) -> Result<(), SaddleError> {
        for axis in CartAxis::ALL {
            let coord = Translation::all(self.n_atoms, axis)?;
            self.fix_translation(coord, x, None)?;
        }
        Ok(())
    }

    /// Fix one rotation axis against the current geometry as reference.
    pub fn fix_rotation(
        &mut self,
        indices: Vec<usize>,
        axis: CartAxis,
        x: ArrayView1<f64>,
    ) -> Result<(), SaddleError> {
        self.check_frame(x)?;
        let refpos = gather_ref(x, &indices)?;
        let coord = Rotation::new(indices, axis, refpos)?;
        self.eqs.push(Equality::Rotation { coord, target: 0.0 });
        Ok(())
    }

    /// Fix all three rotation axes of `indices` (whole molecule if `None`).
    pub fn fix_orient(
        &mut self,
        indices: Option<Vec<usize>>,
        x: ArrayView1<f64>,
    ) -> Result<(), SaddleError> {
        let idx = indices.unwrap_or_else(|| (0..self.n_atoms).collect());
        for axis in CartAxis::ALL {
            self.fix_rotation(idx.clone(), axis, x)?;
        }
        Ok(())
    }

    /// Fix a quadratic displacement.
    pub fn fix_displacement(
        &mut self,
        coord: Displacement,
        x: ArrayView1<f64>,
        target: Option<f64>,
    ) -> Result<(), SaddleError> {
        self.check_frame(x)?;
        let target = match target {
            Some(t) => t,
            None => coord.value(x)?,
        };
        self.eqs.push(Equality::Displacement { coord, target });
        Ok(())
    }

    /// Fix a bond length. Target defaults to the current length.
    pub fn fix_bond(
        &mut self,
        atoms: [usize; 2],
        x: ArrayView1<f64>,
        target: Option<f64>,
    ) -> Result<(), SaddleError> {
        self.check_frame(x)?;
        let target = match target {
            Some(t) => t,
            None => bond_value(x, atoms)?,
        };
        self.eqs.push(Equality::Bond { atoms, target });
        Ok(())
    }

    /// Fix a valence angle (radians). Vertex is `atoms[1]`.
    pub fn fix_angle(
        &mut self,
        atoms: [usize; 3],
        x: ArrayView1<f64>,
        target: Option<f64>,
    ) -> Result<(), SaddleError> {
        self.check_frame(x)?;
        let target = match target {
            Some(t) => t,
            None => angle_value(x, atoms)?,
        };
        self.eqs.push(Equality::Angle { atoms, target });
        Ok(())
    }

    /// Fix a dihedral (radians).
    pub fn fix_dihedral(
        &mut self,
        atoms: [usize; 4],
        x: ArrayView1<f64>,
        target: Option<f64>,
    ) -> Result<(), SaddleError> {
        self.check_frame(x)?;
        let target = match target {
            Some(t) => t,
            None => dihedral_value(x, atoms)?,
        };
        self.eqs.push(Equality::Dihedral { atoms, target });
        Ok(())
    }

    /// Current internals values, length `equalities`.
    pub fn values(&self, x: ArrayView1<f64>) -> Result<Array1<f64>, SaddleError> {
        self.check_frame(x)?;
        let mut q = Array1::zeros(self.eqs.len());
        for (i, eq) in self.eqs.iter().enumerate() {
            q[i] = eq.value(x)?;
        }
        Ok(q)
    }

    /// Sella `Constraints.residual`.
    pub fn residual(&self, x: ArrayView1<f64>) -> Result<Array1<f64>, SaddleError> {
        self.check_frame(x)?;
        let mut r = Array1::zeros(self.eqs.len());
        for (i, eq) in self.eqs.iter().enumerate() {
            if self.ignore_rotation {
                if let Equality::Rotation { .. } = eq {
                    r[i] = 0.0;
                    continue;
                }
            }
            r[i] = eq.residual(x)?;
        }
        Ok(r)
    }

    /// Constraint Jacobian, `ncons x 3N`.
    pub fn jacobian(&self, x: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
        self.check_frame(x)?;
        let mut j = Array2::zeros((self.eqs.len(), x.len()));
        for (i, eq) in self.eqs.iter().enumerate() {
            let g = eq.gradient(x)?;
            for k in 0..x.len() {
                j[(i, k)] = g[k];
            }
        }
        Ok(j)
    }

    /// Sella `PES.get_scons`: min-norm correction `J s = -r`.
    pub fn scons(&self, x: ArrayView1<f64>) -> Result<Array1<f64>, SaddleError> {
        let r = self.residual(x)?;
        let j = self.jacobian(x)?;
        Ok(min_norm_solve(&j, &r).mapv(|v| -v))
    }

    /// `||residual||_2` through vecops.
    pub fn residual_norm(&self, x: ArrayView1<f64>) -> Result<f64, SaddleError> {
        Ok(nrm2(self.residual(x)?.view()))
    }

    /// Gauss-Newton restore onto the level set.
    pub fn restore(&self, mut x: Array1<f64>) -> Result<Array1<f64>, SaddleError> {
        for _ in 0..RESTORE_ITERS {
            let n = self.residual_norm(x.view())?;
            if n <= RESTORE_TOL {
                break;
            }
            let s = self.scons(x.view())?;
            axpy(1.0, s.view(), &mut x);
        }
        Ok(x)
    }

    fn check_frame(&self, x: ArrayView1<f64>) -> Result<(), SaddleError> {
        if x.len() != 3 * self.n_atoms {
            return Err(SaddleError::Shape(
                "constraint frame is not 3 N Cartesian".into(),
            ));
        }
        Ok(())
    }
}

impl Manifold for Constraints {
    fn required_dim(&self, n: usize) -> Result<(), usize> {
        if n == 3 * self.n_atoms {
            Ok(())
        } else {
            Err(3 * self.n_atoms)
        }
    }

    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        match self.jacobian(x.view()) {
            Ok(j) => project_null(&j, v),
            Err(_) => v.clone(),
        }
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        let mut y = x.clone();
        axpy(1.0, v.view(), &mut y);
        self.restore(y).unwrap_or_else(|_| x.clone())
    }

    fn transport(&self, _x_from: &Array1<f64>, x_to: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        self.project(x_to, v)
    }
}

fn gather_ref(x: ArrayView1<f64>, indices: &[usize]) -> Result<Array2<f64>, SaddleError> {
    let mut refpos = Array2::zeros((indices.len(), 3));
    for (local, &atom) in indices.iter().enumerate() {
        for dim in 0..3 {
            let slot = 3 * atom + dim;
            if slot >= x.len() {
                return Err(SaddleError::Shape(
                    "constraint atom index exceeds the 3N frame".into(),
                ));
            }
            refpos[(local, dim)] = x[slot];
        }
    }
    Ok(refpos)
}

fn wrap_pi(a: f64) -> f64 {
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut w = (a + std::f64::consts::PI) % two_pi;
    if w < 0.0 {
        w += two_pi;
    }
    w - std::f64::consts::PI
}

fn atom3(x: ArrayView1<f64>, i: usize) -> Result<[f64; 3], SaddleError> {
    let slot = 3 * i;
    if slot + 2 >= x.len() {
        return Err(SaddleError::Shape(
            "constraint atom index exceeds the 3N frame".into(),
        ));
    }
    Ok([x[slot], x[slot + 1], x[slot + 2]])
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn nrm3(a: [f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}

fn scale3(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scatter3(g: &mut Array1<f64>, atom: usize, v: [f64; 3]) {
    let slot = 3 * atom;
    g[slot] += v[0];
    g[slot + 1] += v[1];
    g[slot + 2] += v[2];
}

fn bond_value(x: ArrayView1<f64>, atoms: [usize; 2]) -> Result<f64, SaddleError> {
    let a = atom3(x, atoms[0])?;
    let b = atom3(x, atoms[1])?;
    let r = nrm3(sub3(b, a));
    if r <= f64::MIN_POSITIVE {
        return Err(SaddleError::NonFinite("constraint bond"));
    }
    Ok(r)
}

fn bond_grad(x: ArrayView1<f64>, atoms: [usize; 2]) -> Result<Array1<f64>, SaddleError> {
    let a = atom3(x, atoms[0])?;
    let b = atom3(x, atoms[1])?;
    let d = sub3(b, a);
    let r = nrm3(d);
    if r <= f64::MIN_POSITIVE {
        return Err(SaddleError::NonFinite("constraint bond"));
    }
    let u = scale3(d, 1.0 / r);
    let mut g = Array1::zeros(x.len());
    scatter3(&mut g, atoms[0], scale3(u, -1.0));
    scatter3(&mut g, atoms[1], u);
    Ok(g)
}

fn angle_value(x: ArrayView1<f64>, atoms: [usize; 3]) -> Result<f64, SaddleError> {
    let a = atom3(x, atoms[0])?;
    let b = atom3(x, atoms[1])?;
    let c = atom3(x, atoms[2])?;
    let u = sub3(a, b);
    let v = sub3(c, b);
    let denom = nrm3(u) * nrm3(v);
    if denom <= f64::MIN_POSITIVE {
        return Err(SaddleError::NonFinite("constraint angle"));
    }
    Ok((dot3(u, v) / denom).clamp(-1.0, 1.0).acos())
}

fn angle_grad(x: ArrayView1<f64>, atoms: [usize; 3]) -> Result<Array1<f64>, SaddleError> {
    let a = atom3(x, atoms[0])?;
    let b = atom3(x, atoms[1])?;
    let c = atom3(x, atoms[2])?;
    let u = sub3(a, b);
    let v = sub3(c, b);
    let ru = nrm3(u);
    let rv = nrm3(v);
    let denom = ru * rv;
    if denom <= f64::MIN_POSITIVE {
        return Err(SaddleError::NonFinite("constraint angle"));
    }
    let cos_t = (dot3(u, v) / denom).clamp(-1.0, 1.0);
    let sin2 = 1.0 - cos_t * cos_t;
    if sin2 <= f64::MIN_POSITIVE {
        return Err(SaddleError::NonFinite("constraint angle"));
    }
    let inv_sin = 1.0 / sin2.sqrt();
    let dtheta_du = scale3(
        sub3(scale3(v, 1.0 / denom), scale3(u, cos_t / (ru * ru))),
        -inv_sin,
    );
    let dtheta_dv = scale3(
        sub3(scale3(u, 1.0 / denom), scale3(v, cos_t / (rv * rv))),
        -inv_sin,
    );
    let mut g = Array1::zeros(x.len());
    scatter3(&mut g, atoms[0], dtheta_du);
    scatter3(&mut g, atoms[1], scale3(add3(dtheta_du, dtheta_dv), -1.0));
    scatter3(&mut g, atoms[2], dtheta_dv);
    Ok(g)
}

fn dihedral_value(x: ArrayView1<f64>, atoms: [usize; 4]) -> Result<f64, SaddleError> {
    let p0 = atom3(x, atoms[0])?;
    let p1 = atom3(x, atoms[1])?;
    let p2 = atom3(x, atoms[2])?;
    let p3 = atom3(x, atoms[3])?;
    let dx1 = sub3(p1, p0);
    let dx2 = sub3(p2, p1);
    let dx3 = sub3(p3, p2);
    let n1 = cross3(dx1, dx2);
    let n2 = cross3(dx2, dx3);
    let numer = dot3(dx2, cross3(n1, n2));
    let denom = nrm3(dx2) * dot3(n1, n2);
    if numer * numer + denom * denom <= f64::MIN_POSITIVE {
        return Err(SaddleError::NonFinite("constraint dihedral"));
    }
    Ok(numer.atan2(denom))
}

fn dihedral_grad(x: ArrayView1<f64>, atoms: [usize; 4]) -> Result<Array1<f64>, SaddleError> {
    let pts = [
        atom3(x, atoms[0])?,
        atom3(x, atoms[1])?,
        atom3(x, atoms[2])?,
        atom3(x, atoms[3])?,
    ];
    let dx1 = sub3(pts[1], pts[0]);
    let dx2 = sub3(pts[2], pts[1]);
    let dx3 = sub3(pts[3], pts[2]);
    let dx2_norm = nrm3(dx2);
    if dx2_norm <= f64::MIN_POSITIVE {
        return Err(SaddleError::NonFinite("constraint dihedral"));
    }
    let n1 = cross3(dx1, dx2);
    let n2 = cross3(dx2, dx3);
    let numer = dot3(dx2, cross3(n1, n2));
    let denom = dx2_norm * dot3(n1, n2);
    let atan_n2 = numer * numer + denom * denom;
    if atan_n2 <= f64::MIN_POSITIVE {
        return Err(SaddleError::NonFinite("constraint dihedral"));
    }
    let mut g = Array1::zeros(x.len());
    for local in 0..4 {
        for dim in 0..3 {
            let mut e = [0.0; 3];
            e[dim] = 1.0;
            let (ddx1, ddx2, ddx3) = match local {
                0 => (scale3(e, -1.0), [0.0; 3], [0.0; 3]),
                1 => (e, scale3(e, -1.0), [0.0; 3]),
                2 => ([0.0; 3], e, scale3(e, -1.0)),
                _ => ([0.0; 3], [0.0; 3], e),
            };
            let dn1 = add3(cross3(ddx1, dx2), cross3(dx1, ddx2));
            let dn2 = add3(cross3(ddx2, dx3), cross3(dx2, ddx3));
            let dnumer =
                dot3(ddx2, cross3(n1, n2)) + dot3(dx2, add3(cross3(dn1, n2), cross3(n1, dn2)));
            let dnorm = dot3(dx2, ddx2) / dx2_norm;
            let ddenom = dnorm * dot3(n1, n2) + dx2_norm * (dot3(dn1, n2) + dot3(n1, dn2));
            g[3 * atoms[local] + dim] = (denom * dnumer - numer * ddenom) / atan_n2;
        }
    }
    Ok(g)
}

/// Project `v` onto `ker(J)` by subtracting the row-space of `J`.
fn project_null(j: &Array2<f64>, v: &Array1<f64>) -> Array1<f64> {
    let ncons = j.nrows();
    let n = v.len().min(j.ncols());
    if ncons == 0 {
        return v.clone();
    }
    let mut basis: Vec<Array1<f64>> = Vec::new();
    for i in 0..ncons {
        let mut row = Array1::zeros(v.len());
        for k in 0..n {
            row[k] = j[(i, k)];
        }
        for b in &basis {
            let c = dot(row.view(), b.view());
            axpy(-c, b.view(), &mut row);
        }
        let nn = nrm2(row.view());
        if nn > RANK_TOL {
            row.mapv_inplace(|x| x / nn);
            basis.push(row);
        }
    }
    let mut out = v.clone();
    for b in &basis {
        let c = dot(out.view(), b.view());
        axpy(-c, b.view(), &mut out);
    }
    out
}

/// Min-norm `s` solving `J s = r` (row-space of `J`).
fn min_norm_solve(j: &Array2<f64>, r: &Array1<f64>) -> Array1<f64> {
    let ncons = j.nrows();
    let n = j.ncols();
    let mut s = Array1::zeros(n);
    if ncons == 0 {
        return s;
    }
    // Normal equations `(J J^T) y = r`, then `s = J^T y`.
    let mut a = Array2::<f64>::zeros((ncons, ncons));
    for i in 0..ncons {
        for k in 0..ncons {
            let mut acc = 0.0;
            for t in 0..n {
                acc += j[(i, t)] * j[(k, t)];
            }
            a[(i, k)] = acc;
        }
    }
    let y = solve_spd(&a, r);
    for t in 0..n {
        let mut acc = 0.0;
        for i in 0..ncons {
            acc += j[(i, t)] * y[i];
        }
        s[t] = acc;
    }
    s
}

fn solve_spd(a: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let n = b.len();
    if n == 0 {
        return Array1::zeros(0);
    }
    let mut m = a.clone();
    for i in 0..n {
        m[(i, i)] += 1e-14;
    }
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut acc = m[(i, j)];
            for k in 0..j {
                acc -= l[(i, k)] * l[(j, k)];
            }
            if i == j {
                l[(i, j)] = acc.max(0.0).sqrt();
            } else if l[(j, j)] > 1e-16 {
                l[(i, j)] = acc / l[(j, j)];
            }
        }
    }
    let mut y = Array1::zeros(n);
    for i in 0..n {
        let mut acc = b[i];
        for k in 0..i {
            acc -= l[(i, k)] * y[k];
        }
        y[i] = if l[(i, i)] > 1e-16 {
            acc / l[(i, i)]
        } else {
            0.0
        };
    }
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut acc = y[i];
        for k in (i + 1)..n {
            acc -= l[(k, i)] * x[k];
        }
        x[i] = if l[(i, i)] > 1e-16 {
            acc / l[(i, i)]
        } else {
            0.0
        };
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use rgmin::ManifoldKind;

    fn water() -> Array1<f64> {
        pack_cart(&[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
    }

    fn com(x: ArrayView1<f64>) -> [f64; 3] {
        let n = x.len() / 3;
        let mut c = [0.0; 3];
        for i in 0..n {
            c[0] += x[3 * i];
            c[1] += x[3 * i + 1];
            c[2] += x[3 * i + 2];
        }
        let f = n as f64;
        [c[0] / f, c[1] / f, c[2] / f]
    }

    #[test]
    fn com_residual_is_zero_at_construction() {
        let x = water();
        let mut cons = Constraints::new(3).unwrap();
        cons.fix_com(x.view()).unwrap();
        let r = cons.residual_norm(x.view()).unwrap();
        assert!(r < 1e-14, "r={r}");
        assert_eq!(cons.counts().ntrans, 3);
    }

    #[test]
    fn project_of_a_translation_vanishes_when_com_is_fixed() {
        let x = water();
        let mut cons = Constraints::new(3).unwrap();
        cons.fix_com(x.view()).unwrap();
        let shift = Array1::from_elem(9, 0.1);
        let v = cons.project(&x, &shift);
        let n = nrm2(v.view());
        assert!(n < 1e-12, "projected translation {n}");
    }

    #[test]
    fn retract_keeps_the_com() {
        let x = water();
        let mut cons = Constraints::new(3).unwrap();
        cons.fix_com(x.view()).unwrap();
        let c0 = com(x.view());
        let mut step = Array1::zeros(9);
        step[0] = 0.3;
        step[4] = -0.2;
        step[8] = 0.1;
        let y = cons.retract(&x, &step);
        let c1 = com(y.view());
        assert!(
            (c0[0] - c1[0]).abs() < 1e-10
                && (c0[1] - c1[1]).abs() < 1e-10
                && (c0[2] - c1[2]).abs() < 1e-10,
            "COM drifted {c0:?} -> {c1:?}"
        );
        let r = cons.residual_norm(y.view()).unwrap();
        assert!(r < 1e-10, "retract left the set: r={r}");
        let t = cons.transport(&x, &y, &step);
        let t_h = cons.project(&y, &t);
        assert!(nrm2((&t - &t_h).view()) < 1e-12);
    }

    #[test]
    fn fixed_bond_survives_a_retract() {
        let x = water();
        let mut cons = Constraints::new(3).unwrap();
        cons.fix_bond([0, 1], x.view(), None).unwrap();
        let r0 = bond_value(x.view(), [0, 1]).unwrap();
        let mut step = Array1::zeros(9);
        step[0] = 0.2;
        step[3] = -0.15;
        let y = cons.retract(&x, &step);
        let r1 = bond_value(y.view(), [0, 1]).unwrap();
        assert!((r0 - r1).abs() < 1e-10, "bond {r0} -> {r1}");
        assert!(cons.residual_norm(y.view()).unwrap() < 1e-10);
    }

    #[test]
    fn fixed_angle_survives_a_retract() {
        let x = water();
        let mut cons = Constraints::new(3).unwrap();
        cons.fix_angle([1, 0, 2], x.view(), None).unwrap();
        let a0 = angle_value(x.view(), [1, 0, 2]).unwrap();
        let mut step = Array1::zeros(9);
        step[3] = 0.05;
        step[7] = -0.04;
        let y = cons.retract(&x, &step);
        let a1 = angle_value(y.view(), [1, 0, 2]).unwrap();
        assert!((a0 - a1).abs() < 1e-9, "angle {a0} -> {a1}");
    }

    #[test]
    fn com_plus_orient_stays_on_the_rigid_quotient() {
        let x = water();
        let mut cons = Constraints::new(3).unwrap().keep_rotation_residual();
        cons.fix_com(x.view()).unwrap();
        cons.fix_orient(None, x.view()).unwrap();
        let mut step = Array1::zeros(9);
        step[0] = 0.2;
        step[4] = -0.15;
        step[8] = 0.05;
        let y = cons.retract(&x, &step);
        assert!(
            cons.residual_norm(y.view()).unwrap() < 1e-8,
            "left the constraint set"
        );
        let d = &y - &x;
        let d_h = ManifoldKind::RigidQuotient.project(&x, &d);
        // Restore may add a small vertical (constraint) correction;
        // the free motion is horizontal on the SE(3) quotient.
        let v = cons.project(&x, &step);
        let v_h = ManifoldKind::RigidQuotient.project(&x, &v);
        assert!(
            nrm2((&v - &v_h).view()) < 1e-8,
            "free step left the rigid quotient"
        );
        let _ = d_h;
    }

    #[test]
    fn bond_gradient_matches_central_difference() {
        let x = water();
        let g = bond_grad(x.view(), [0, 2]).unwrap();
        let mut gfd = Array1::zeros(9);
        let h = 1e-6;
        let mut xp = x.clone();
        for i in 0..9 {
            xp[i] = x[i] + h;
            let fp = bond_value(xp.view(), [0, 2]).unwrap();
            xp[i] = x[i] - h;
            let fm = bond_value(xp.view(), [0, 2]).unwrap();
            xp[i] = x[i];
            gfd[i] = (fp - fm) / (2.0 * h);
        }
        assert!(nrm2((&g - &gfd).view()) < 1e-8);
    }
}
