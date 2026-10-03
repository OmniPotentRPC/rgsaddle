//! Sella `linalg.py` helpers the steppers actually call.
//!
//! [`NumericalHessian`] is the finite-difference Hessian-vector
//! product (`NumericalHessian._matvec`). Reductions go through
//! [`rgmin::vecops`] so the `par` feature applies. A host that
//! needs the displaced point on a set calls [`numerical_hvp_on`]:
//! `project` / `retract` / `transport`. Dense eigen lives in
//! [`crate::eigensolve`]. Approximate Hessian lives in [`crate::qn`].
//! Sparse internals stay in vocn.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rgmin::Manifold;
use rgmin::vecops::{axpy, dot, nrm2};

use crate::error::SaddleError;
use crate::minmode::PointSurface;

/// Sella displacement sign: descend, else toward the origin, else first
/// nonzero component positive.
pub fn hessian_vec_sign(v: ArrayView1<f64>, g0: ArrayView1<f64>, x0: ArrayView1<f64>) -> f64 {
    let vdotg = dot(v, g0);
    if vdotg.abs() > 1e-4 {
        return if vdotg < 0.0 { 1.0 } else { -1.0 };
    }
    let vdotx = dot(v, x0);
    if vdotx.abs() > 1e-4 {
        return if vdotx < 0.0 { 1.0 } else { -1.0 };
    }
    for &vi in v.iter() {
        if vi > 1e-4 {
            return 1.0;
        }
        if vi < -1e-4 {
            return -1.0;
        }
    }
    1.0
}

/// Lift a reduced vector through Sella `Uproj`: `U @ v`.
pub fn lift_through(uproj: ArrayView2<f64>, v: ArrayView1<f64>) -> Array1<f64> {
    let mut out = Array1::zeros(uproj.nrows());
    for i in 0..uproj.nrows() {
        out[i] = dot(uproj.row(i), v);
    }
    out
}

/// Project a full-space vector through Sella `Uproj`: `U^T @ av`.
pub fn project_through(uproj: ArrayView2<f64>, av: ArrayView1<f64>) -> Array1<f64> {
    let mut out = Array1::zeros(uproj.ncols());
    for j in 0..uproj.ncols() {
        out[j] = dot(uproj.column(j), av);
    }
    out
}

/// Two-sided (or one-sided) Hessian-vector product of a surface.
///
/// Euclidean Sella `_matvec` without `Uproj`. A zero `v` returns zero.
pub fn numerical_hvp<S: PointSurface>(
    surface: &S,
    x0: ArrayView1<f64>,
    g0: ArrayView1<f64>,
    v: ArrayView1<f64>,
    eta: f64,
    threepoint: bool,
) -> Result<Array1<f64>, SaddleError> {
    numerical_hvp_proj(surface, x0, g0, v, eta, threepoint, None)
}

/// Sella `_matvec` with optional `Uproj` (free-subspace lift / project).
pub fn numerical_hvp_proj<S: PointSurface>(
    surface: &S,
    x0: ArrayView1<f64>,
    g0: ArrayView1<f64>,
    v: ArrayView1<f64>,
    eta: f64,
    threepoint: bool,
    uproj: Option<ArrayView2<f64>>,
) -> Result<Array1<f64>, SaddleError> {
    let v_full = match uproj {
        Some(u) => {
            if u.nrows() != x0.len() || u.ncols() != v.len() {
                return Err(SaddleError::Shape(
                    "Uproj rows must match the 3N frame and columns the reduced v".into(),
                ));
            }
            lift_through(u, v)
        }
        None => {
            if v.len() != x0.len() || g0.len() != x0.len() {
                return Err(SaddleError::Shape(
                    "numerical HVP length must match the 3N frame".into(),
                ));
            }
            v.to_owned()
        }
    };
    if g0.len() != x0.len() {
        return Err(SaddleError::Shape(
            "numerical HVP gradient length must match the 3N frame".into(),
        ));
    }
    let vn = nrm2(v_full.view());
    if vn < 1e-12 {
        return Ok(Array1::zeros(v.len()));
    }
    let sign = hessian_vec_sign(v_full.view(), g0, x0);
    let scale = vn * sign;
    let mut xp = x0.to_owned();
    axpy(eta / scale, v_full.view(), &mut xp);
    let (_, gplus) = surface.eval(xp.view())?;
    let av_full = if threepoint {
        let mut xm = x0.to_owned();
        axpy(-eta / scale, v_full.view(), &mut xm);
        let (_, gminus) = surface.eval(xm.view())?;
        let mut av = gplus;
        axpy(-1.0, gminus.view(), &mut av);
        av.mapv_inplace(|a| a * scale / (2.0 * eta));
        av
    } else {
        let mut av = gplus;
        axpy(-1.0, g0, &mut av);
        av.mapv_inplace(|a| a * scale / eta);
        av
    };
    match uproj {
        Some(u) => Ok(project_through(u, av_full.view())),
        None => Ok(av_full),
    }
}

/// Project an HVP direction onto `T_x`.
pub fn project_hvp<M: Manifold>(man: &M, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
    man.project(x, v)
}

/// Finite-difference displacement of `x` along `v`, retracted onto the set.
///
/// `sign` is Sella's displacement sign; the tangent increment is
/// `sign * eta / ||v|| * v`.
pub fn retract_hvp<M: Manifold>(
    man: &M,
    x: &Array1<f64>,
    v: &Array1<f64>,
    eta: f64,
    sign: f64,
) -> Array1<f64> {
    let vn = nrm2(v.view());
    if vn < 1e-12 {
        return x.clone();
    }
    let mut s = Array1::zeros(v.len());
    axpy(sign * eta / vn, v.view(), &mut s);
    let s = man.project(x, &s);
    man.retract(x, &s)
}

/// Vector transport of an HVP increment from `x_from` to `x_to`.
pub fn transport_hvp<M: Manifold>(
    man: &M,
    x_from: &Array1<f64>,
    x_to: &Array1<f64>,
    v: &Array1<f64>,
) -> Array1<f64> {
    man.transport(x_from, x_to, v)
}

/// Hessian-vector product whose displaced point stays on `man`.
///
/// Project `v` onto `T_x`, retract the finite-difference step, evaluate,
/// transport the Riemannian gradient back, and project the HVP.
pub fn numerical_hvp_on<S: PointSurface, M: Manifold>(
    man: &M,
    surface: &S,
    x0: ArrayView1<f64>,
    g0: ArrayView1<f64>,
    v: ArrayView1<f64>,
    eta: f64,
    threepoint: bool,
) -> Result<Array1<f64>, SaddleError> {
    if v.len() != x0.len() || g0.len() != x0.len() {
        return Err(SaddleError::Shape(
            "numerical HVP length must match the 3N frame".into(),
        ));
    }
    let x = x0.to_owned();
    let v_t = project_hvp(man, &x, &v.to_owned());
    let vn = nrm2(v_t.view());
    if vn < 1e-12 {
        return Ok(Array1::zeros(x.len()));
    }
    let sign = hessian_vec_sign(v_t.view(), g0, x0);
    let xp = retract_hvp(man, &x, &v_t, eta, sign);
    let (_, gplus) = surface.eval(xp.view())?;
    let gplus_r = man.egrad2rgrad(&xp, &gplus);
    let gplus_x = transport_hvp(man, &xp, &x, &gplus_r);
    let av = if threepoint {
        let xm = retract_hvp(man, &x, &v_t, eta, -sign);
        let (_, gminus) = surface.eval(xm.view())?;
        let gminus_r = man.egrad2rgrad(&xm, &gminus);
        let gminus_x = transport_hvp(man, &xm, &x, &gminus_r);
        let mut av = gplus_x;
        axpy(-1.0, gminus_x.view(), &mut av);
        av.mapv_inplace(|a| a * vn * sign / (2.0 * eta));
        av
    } else {
        let g0_r = man.egrad2rgrad(&x, &g0.to_owned());
        let mut av = gplus_x;
        axpy(-1.0, g0_r.view(), &mut av);
        av.mapv_inplace(|a| a * vn * sign / eta);
        av
    };
    Ok(project_hvp(man, &x, &av))
}

/// Sella `linalg.NumericalHessian`: cached FD HVP for a stepper.
#[derive(Clone, Debug)]
pub struct NumericalHessian {
    x0: Array1<f64>,
    g0: Array1<f64>,
    eta: f64,
    threepoint: bool,
    uproj: Option<Array2<f64>>,
    vs: Array2<f64>,
    avs: Array2<f64>,
}

impl NumericalHessian {
    /// Sella `NumericalHessian(func, x0, g0, eta, threepoint, Uproj)`.
    pub fn new(
        x0: Array1<f64>,
        g0: Array1<f64>,
        eta: f64,
        threepoint: bool,
        uproj: Option<Array2<f64>>,
    ) -> Result<Self, SaddleError> {
        if g0.len() != x0.len() {
            return Err(SaddleError::Shape(
                "NumericalHessian g0 must match x0".into(),
            ));
        }
        if let Some(u) = uproj.as_ref()
            && u.nrows() != x0.len()
        {
            return Err(SaddleError::Shape(
                "Uproj rows must match the 3N frame".into(),
            ));
        }
        let ntrue = x0.len();
        Ok(Self {
            x0,
            g0,
            eta,
            threepoint,
            uproj,
            vs: Array2::zeros((ntrue, 0)),
            avs: Array2::zeros((ntrue, 0)),
        })
    }

    /// Cached full-space trial directions (`NumericalHessian.Vs`).
    pub fn vs(&self) -> ArrayView2<'_, f64> {
        self.vs.view()
    }

    /// Cached full-space actions (`NumericalHessian.AVs`).
    pub fn avs(&self) -> ArrayView2<'_, f64> {
        self.avs.view()
    }

    /// Sella `_matvec`. Caches the full-space pair `(V, AV)`.
    pub fn matvec<S: PointSurface>(
        &mut self,
        surface: &S,
        v: ArrayView1<f64>,
    ) -> Result<Array1<f64>, SaddleError> {
        let v_full = match &self.uproj {
            Some(u) => lift_through(u.view(), v),
            None => v.to_owned(),
        };
        let av_full = numerical_hvp(
            surface,
            self.x0.view(),
            self.g0.view(),
            v_full.view(),
            self.eta,
            self.threepoint,
        )?;
        self.push_pair(v_full.view(), av_full.view());
        match &self.uproj {
            Some(u) => Ok(project_through(u.view(), av_full.view())),
            None => Ok(av_full),
        }
    }

    /// `_matvec` whose displaced point stays on `man`.
    pub fn matvec_on<S: PointSurface, M: Manifold>(
        &mut self,
        man: &M,
        surface: &S,
        v: ArrayView1<f64>,
    ) -> Result<Array1<f64>, SaddleError> {
        let v_full = match &self.uproj {
            Some(u) => lift_through(u.view(), v),
            None => v.to_owned(),
        };
        let av_full = numerical_hvp_on(
            man,
            surface,
            self.x0.view(),
            self.g0.view(),
            v_full.view(),
            self.eta,
            self.threepoint,
        )?;
        self.push_pair(v_full.view(), av_full.view());
        match &self.uproj {
            Some(u) => Ok(project_through(u.view(), av_full.view())),
            None => Ok(av_full),
        }
    }

    fn push_pair(&mut self, v: ArrayView1<f64>, av: ArrayView1<f64>) {
        let n = self.vs.nrows();
        let k = self.vs.ncols();
        let mut vs = Array2::zeros((n, k + 1));
        let mut avs = Array2::zeros((n, k + 1));
        for j in 0..k {
            for i in 0..n {
                vs[(i, j)] = self.vs[(i, j)];
                avs[(i, j)] = self.avs[(i, j)];
            }
        }
        for i in 0..n {
            vs[(i, k)] = v[i];
            avs[(i, k)] = av[i];
        }
        self.vs = vs;
        self.avs = avs;
    }
}

/// Modified Gram-Schmidt. Columns of `x` are orthonormalized against
/// each other and against the optional `y` basis. Columns with
/// residual below `eps` are dropped.
pub fn modified_gram_schmidt(
    x: ArrayView2<f64>,
    y: Option<ArrayView2<f64>>,
    eps: f64,
) -> Array2<f64> {
    let n = x.nrows();
    let mut cols: Vec<Array1<f64>> = Vec::new();
    if let Some(y) = y {
        for j in 0..y.ncols() {
            cols.push(y.column(j).to_owned());
        }
    }
    let y_keep = cols.len();
    for j in 0..x.ncols() {
        let mut v = x.column(j).to_owned();
        for b in &cols {
            let c = dot(v.view(), b.view());
            axpy(-c, b.view(), &mut v);
        }
        let nn = nrm2(v.view());
        if nn > eps {
            v.mapv_inplace(|a| a / nn);
            cols.push(v);
        }
    }
    let kept = cols.len() - y_keep;
    let mut out = Array2::zeros((n, kept));
    for (j, col) in cols.into_iter().skip(y_keep).enumerate() {
        for i in 0..n {
            out[(i, j)] = col[i];
        }
    }
    out
}

/// Symmetrize a rectangular pair `(V, AV)` the Sella `symm=2` way:
/// `0.5 (V^T AV + (V^T AV)^T)` projected back is done by the caller;
/// this returns the symmetric `V^T AV`.
pub fn symmetrize_vt_av(v: ArrayView2<f64>, av: ArrayView2<f64>) -> Array2<f64> {
    let k = v.ncols();
    let mut a = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in 0..k {
            a[(i, j)] = dot(v.column(i), av.column(j));
        }
    }
    for i in 0..k {
        for j in 0..i {
            let s = 0.5 * (a[(i, j)] + a[(j, i)]);
            a[(i, j)] = s;
            a[(j, i)] = s;
        }
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SaddleError;
    use crate::constraints::Constraints;
    use crate::internal::pack_cart;
    use ndarray::{Array1, Array2, ArrayView1, array};
    use rgmin::ManifoldKind;

    struct Quad;
    impl PointSurface for Quad {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            // E = x0^2 + 4 x1^2, g = (2 x0, 8 x1)
            Ok((
                x[0] * x[0] + 4.0 * x[1] * x[1],
                Array1::from(vec![2.0 * x[0], 8.0 * x[1]]),
            ))
        }
    }

    struct SphereQuad;
    impl PointSurface for SphereQuad {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            // E = 2 x0^2 + x1^2 + 3 x2^2
            Ok((
                2.0 * x[0] * x[0] + x[1] * x[1] + 3.0 * x[2] * x[2],
                Array1::from(vec![4.0 * x[0], 2.0 * x[1], 6.0 * x[2]]),
            ))
        }
    }

    #[test]
    fn hvp_recovers_a_diagonal_hessian() {
        let x = Array1::from(vec![0.3, -0.2]);
        let (_, g) = Quad.eval(x.view()).unwrap();
        let e0 = numerical_hvp(
            &Quad,
            x.view(),
            g.view(),
            Array1::from(vec![1.0, 0.0]).view(),
            1e-5,
            true,
        )
        .unwrap();
        let e1 = numerical_hvp(
            &Quad,
            x.view(),
            g.view(),
            Array1::from(vec![0.0, 1.0]).view(),
            1e-5,
            true,
        )
        .unwrap();
        assert!((e0[0] - 2.0).abs() < 1e-6, "H00={}", e0[0]);
        assert!(e0[1].abs() < 1e-6);
        assert!((e1[1] - 8.0).abs() < 1e-6, "H11={}", e1[1]);
        assert!(e1[0].abs() < 1e-6);
    }

    #[test]
    fn hvp_zero_vector_is_zero() {
        let x = Array1::from(vec![0.3, -0.2]);
        let (_, g) = Quad.eval(x.view()).unwrap();
        let av = numerical_hvp(
            &Quad,
            x.view(),
            g.view(),
            Array1::zeros(2).view(),
            1e-5,
            false,
        )
        .unwrap();
        assert!(nrm2(av.view()) < 1e-15);
    }

    #[test]
    fn hvp_proj_stays_in_the_subspace() {
        let x = Array1::from(vec![0.3, -0.2, 0.1]);
        let mut u = Array2::zeros((3, 2));
        u[(0, 0)] = 1.0;
        u[(1, 1)] = 1.0;
        let g = Array1::from(vec![2.0 * x[0], 8.0 * x[1], 0.0]);
        let v = Array1::from(vec![1.0, 0.0]);
        let av = numerical_hvp_proj(
            &Quad3,
            x.view(),
            g.view(),
            v.view(),
            1e-5,
            true,
            Some(u.view()),
        )
        .unwrap();
        assert_eq!(av.len(), 2);
        assert!((av[0] - 2.0).abs() < 1e-6, "H00={}", av[0]);
        assert!(av[1].abs() < 1e-6);
    }

    struct Quad3;
    impl PointSurface for Quad3 {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            Ok((
                x[0] * x[0] + 4.0 * x[1] * x[1],
                Array1::from(vec![2.0 * x[0], 8.0 * x[1], 0.0]),
            ))
        }
    }

    #[test]
    fn mgs_orthonormalizes_and_drops_a_duplicate() {
        let mut x = Array2::zeros((3, 3));
        x[(0, 0)] = 1.0;
        x[(1, 0)] = 1.0;
        x[(0, 1)] = 1.0;
        x[(1, 1)] = 1.0; // duplicate of col 0
        x[(2, 2)] = 1.0;
        let q = modified_gram_schmidt(x.view(), None, 1e-12);
        assert_eq!(q.ncols(), 2);
        let n0 = nrm2(q.column(0));
        let n1 = nrm2(q.column(1));
        assert!((n0 - 1.0).abs() < 1e-12);
        assert!((n1 - 1.0).abs() < 1e-12);
        assert!(dot(q.column(0), q.column(1)).abs() < 1e-12);
    }

    #[test]
    fn hvp_on_sphere_stays_on_the_set() {
        let man = ManifoldKind::Sphere;
        let x = array![0.0, 1.0, 0.0];
        let (_, g) = SphereQuad.eval(x.view()).unwrap();
        let v_amb = array![0.2, -0.1, 0.3];
        let s = project_hvp(&man, &x, &v_amb);
        assert!(
            dot(x.view(), s.view()).abs() < 1e-12,
            "projected step must be tangent: x·s={}",
            dot(x.view(), s.view())
        );
        let sign = hessian_vec_sign(s.view(), g.view(), x.view());
        let y = retract_hvp(&man, &x, &s, 1e-4, sign);
        assert!(
            (nrm2(y.view()) - 1.0).abs() < 1e-12,
            "retracted point left the sphere: ||y||={}",
            nrm2(y.view())
        );
        let t = transport_hvp(&man, &x, &y, &s);
        assert!(
            dot(y.view(), t.view()).abs() < 1e-12,
            "transported step leaves T_y: y·t={}",
            dot(y.view(), t.view())
        );
        let av = numerical_hvp_on(
            &man,
            &SphereQuad,
            x.view(),
            g.view(),
            v_amb.view(),
            1e-5,
            true,
        )
        .unwrap();
        assert!(av.iter().all(|a| a.is_finite()));
        assert!(
            dot(x.view(), av.view()).abs() < 1e-10,
            "HVP left T_x: x·Av={}",
            dot(x.view(), av.view())
        );
        let av_h = project_hvp(&man, &x, &av);
        assert!(nrm2((&av - &av_h).view()) < 1e-12);
    }

    #[test]
    fn hessian_cache_records_full_space_pairs() {
        let x = Array1::from(vec![0.3, -0.2]);
        let (_, g) = Quad.eval(x.view()).unwrap();
        let mut h = NumericalHessian::new(x, g, 1e-5, true, None).unwrap();
        let _ = h
            .matvec(&Quad, Array1::from(vec![1.0, 0.0]).view())
            .unwrap();
        assert_eq!(h.vs().ncols(), 1);
        assert_eq!(h.avs().ncols(), 1);
        assert!((h.avs()[(0, 0)] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn matvec_on_sphere_stays_on_the_set() {
        let man = ManifoldKind::Sphere;
        let x = array![0.0, 1.0, 0.0];
        let (_, g) = SphereQuad.eval(x.view()).unwrap();
        let mut h = NumericalHessian::new(x.clone(), g, 1e-5, false, None).unwrap();
        let av = h
            .matvec_on(&man, &SphereQuad, array![0.2, -0.1, 0.3].view())
            .unwrap();
        assert!(dot(x.view(), av.view()).abs() < 1e-10);
        assert_eq!(h.vs().ncols(), 1);
    }

    #[test]
    fn hvp_on_com_set_stays_on_the_set() {
        let x = pack_cart(&[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]);
        let mut cons = Constraints::new(3).unwrap();
        cons.fix_com(x.view()).unwrap();
        assert!(cons.residual_norm(x.view()).unwrap() < 1e-14);
        let g = Array1::from_elem(9, 0.1);
        let v = Array1::from_elem(9, 0.2);
        let y = retract_hvp(&cons, &x, &project_hvp(&cons, &x, &v), 1e-4, 1.0);
        let res = cons.residual_norm(y.view()).unwrap();
        assert!(res < 1e-10, "HVP retract left the COM set: {res}");
        let av =
            numerical_hvp_on(&cons, &Well9, x.view(), g.view(), v.view(), 1e-5, false).unwrap();
        let av_h = cons.project(&x, &av);
        assert!(nrm2((&av - &av_h).view()) < 1e-10);
    }

    struct Well9;
    impl PointSurface for Well9 {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let mut e = 0.0;
            let mut g = Array1::zeros(x.len());
            for i in 0..x.len() {
                e += x[i] * x[i];
                g[i] = 2.0 * x[i];
            }
            Ok((e, g))
        }
    }
}
