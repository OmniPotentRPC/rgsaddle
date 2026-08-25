//! Sella `linalg.py` helpers the steppers actually call.
//!
//! Finite-difference Hessian-vector products (`NumericalHessian`) and
//! modified Gram-Schmidt. Dense eigen lives in [`crate::eigensolve`].
//! Prefer rgmin / ndarray; this module does not grow a second LA stack.
//! Ambient reductions go through [`rgmin::vecops`] so `par` applies.
//! A host that needs the HVP increment on a set calls
//! [`project_hvp`] / [`retract_hvp`] / [`transport_hvp`].

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rgmin::Manifold;
use rgmin::vecops::{axpy, dot, nrm2};

use crate::error::SaddleError;
use crate::minmode::PointSurface;

/// `y = U v` with columns of `U` as the free basis (Sella `Uproj`).
fn apply_uproj(u: ArrayView2<f64>, v: ArrayView1<f64>) -> Array1<f64> {
    let mut y = Array1::zeros(u.nrows());
    let k = u.ncols().min(v.len());
    for j in 0..k {
        axpy(v[j], u.column(j), &mut y);
    }
    y
}

/// `y = U^T v`.
fn apply_uproj_t(u: ArrayView2<f64>, v: ArrayView1<f64>) -> Array1<f64> {
    let mut y = Array1::zeros(u.ncols());
    for j in 0..u.ncols() {
        y[j] = dot(u.column(j), v);
    }
    y
}

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

/// Two-sided (or one-sided) Hessian-vector product of a surface.
///
/// `uproj` is Sella `NumericalHessian.Uproj`: `v` is then the free
/// coordinate and the return is `U^T (H U v)`.
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

/// [`numerical_hvp`] with optional Sella `Uproj`.
pub fn numerical_hvp_proj<S: PointSurface>(
    surface: &S,
    x0: ArrayView1<f64>,
    g0: ArrayView1<f64>,
    v: ArrayView1<f64>,
    eta: f64,
    threepoint: bool,
    uproj: Option<ArrayView2<f64>>,
) -> Result<Array1<f64>, SaddleError> {
    if g0.len() != x0.len() {
        return Err(SaddleError::Shape(
            "numerical HVP length must match the 3N frame".into(),
        ));
    }
    let v_full = match uproj {
        Some(u) => {
            if u.nrows() != x0.len() {
                return Err(SaddleError::Shape(
                    "Uproj rows must match the 3N frame".into(),
                ));
            }
            if v.len() != u.ncols() {
                return Err(SaddleError::Shape(
                    "Uproj columns must match the free displacement".into(),
                ));
            }
            apply_uproj(u, v)
        }
        None => {
            if v.len() != x0.len() {
                return Err(SaddleError::Shape(
                    "numerical HVP length must match the 3N frame".into(),
                ));
            }
            v.to_owned()
        }
    };
    let vn = nrm2(v_full.view());
    if vn < 1e-12 {
        return Ok(Array1::zeros(match uproj {
            Some(u) => u.ncols(),
            None => x0.len(),
        }));
    }
    let sign = hessian_vec_sign(v_full.view(), g0, x0);
    let scale = vn * sign;
    let mut xp = x0.to_owned();
    axpy(eta / scale, v_full.view(), &mut xp);
    let (_, gplus) = surface.eval(xp.view())?;
    let mut av = if threepoint {
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
    if let Some(u) = uproj {
        av = apply_uproj_t(u, av.view());
    }
    Ok(av)
}

/// Project an HVP increment onto the tangent (Sella stay-on-set waist).
pub fn project_hvp<M: Manifold>(m: &M, x: &Array1<f64>, av: &Array1<f64>) -> Array1<f64> {
    m.project(x, av)
}

/// Retract along a projected HVP increment. The arrival stays on the set.
pub fn retract_hvp<M: Manifold>(m: &M, x: &Array1<f64>, av: &Array1<f64>) -> Array1<f64> {
    let s = m.project(x, av);
    m.retract(x, &s)
}

/// Vector transport of an HVP increment from `x_from` to `x_to`.
pub fn transport_hvp<M: Manifold>(
    m: &M,
    x_from: &Array1<f64>,
    x_to: &Array1<f64>,
    av: &Array1<f64>,
) -> Array1<f64> {
    m.transport(x_from, x_to, av)
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
            let mut c = Array1::zeros(n);
            for i in 0..n {
                c[i] = y[(i, j)];
            }
            cols.push(c);
        }
    }
    let y_keep = cols.len();
    for j in 0..x.ncols() {
        let mut v = Array1::zeros(n);
        for i in 0..n {
            v[i] = x[(i, j)];
        }
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
    use ndarray::{Array1, Array2, ArrayView1};

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
        let mut d = 0.0;
        for i in 0..3 {
            d += q[(i, 0)] * q[(i, 1)];
        }
        assert!(d.abs() < 1e-12, "not orthogonal: {d}");
    }

    #[test]
    fn hvp_proj_retr_transp_stays_on_the_sphere() {
        use rgmin::ManifoldKind;
        let x = Array1::from(vec![0.0, 1.0, 0.0]);
        let (_, g) = Quad3.eval(x.view()).unwrap();
        let v = Array1::from(vec![0.3, 0.0, -0.1]);
        let av = numerical_hvp(&Quad3, x.view(), g.view(), v.view(), 1e-5, true).unwrap();
        let man = ManifoldKind::Sphere;
        let s = project_hvp(&man, &x, &av);
        assert!(
            dot(x.view(), s.view()).abs() < 1e-12,
            "projected HVP must be tangent: x·s={}",
            dot(x.view(), s.view())
        );
        let y = retract_hvp(&man, &x, &av);
        assert!(
            (nrm2(y.view()) - 1.0).abs() < 1e-12,
            "retracted HVP left the sphere: ||y||={}",
            nrm2(y.view())
        );
        let t = transport_hvp(&man, &x, &y, &s);
        assert!(
            dot(y.view(), t.view()).abs() < 1e-12,
            "transported HVP leaves T_y: y·t={}",
            dot(y.view(), t.view())
        );
        let th = man.project(&y, &t);
        assert!(nrm2((&t - &th).view()) < 1e-12);
    }

    #[test]
    fn uproj_hvp_stays_in_the_free_chart() {
        use rgmin::ManifoldKind;
        let man = ManifoldKind::Sphere;
        let x = Array1::from(vec![0.0, 0.0, 1.0]);
        let (_, g) = Quad3.eval(x.view()).unwrap();
        let mut u = Array2::zeros((3, 2));
        u[(0, 0)] = 1.0;
        u[(1, 1)] = 1.0;
        let v_free = Array1::from(vec![0.4, -0.2]);
        let av_free = numerical_hvp_proj(
            &Quad3,
            x.view(),
            g.view(),
            v_free.view(),
            1e-5,
            true,
            Some(u.view()),
        )
        .unwrap();
        assert_eq!(av_free.len(), 2);
        let av_full = apply_uproj(u.view(), av_free.view());
        let s = project_hvp(&man, &x, &av_full);
        assert!(dot(x.view(), s.view()).abs() < 1e-12);
        let y = retract_hvp(&man, &x, &av_full);
        assert!((nrm2(y.view()) - 1.0).abs() < 1e-12);
        let t = transport_hvp(&man, &x, &y, &s);
        assert!(dot(y.view(), t.view()).abs() < 1e-12);
    }

    #[test]
    fn hvp_proj_retr_transp_stays_on_the_com_set() {
        use crate::constraints::Constraints;
        let x = Array1::from(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let mut cons = Constraints::new(3).unwrap();
        cons.fix_com(x.view()).unwrap();
        assert!(cons.residual_norm(x.view()).unwrap() < 1e-14);
        let (_, g) = QuadN.eval(x.view()).unwrap();
        let v = Array1::from_elem(9, 0.1);
        let av = numerical_hvp(&QuadN, x.view(), g.view(), v.view(), 1e-5, true).unwrap();
        let s = project_hvp(&cons, &x, &av);
        let sh = cons.project(&x, &s);
        assert!(nrm2((&s - &sh).view()) < 1e-12);
        let y = retract_hvp(&cons, &x, &av);
        let res = cons.residual_norm(y.view()).unwrap();
        assert!(res < 1e-10, "HVP retract left the COM set: {res}");
        let t = transport_hvp(&cons, &x, &y, &s);
        let th = cons.project(&y, &t);
        assert!(nrm2((&t - &th).view()) < 1e-12);
    }

    struct Quad3;
    impl PointSurface for Quad3 {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            Ok((
                x[0] * x[0] + 4.0 * x[1] * x[1] + 9.0 * x[2] * x[2],
                Array1::from(vec![2.0 * x[0], 8.0 * x[1], 18.0 * x[2]]),
            ))
        }
    }

    struct QuadN;
    impl PointSurface for QuadN {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let e: f64 = x.iter().map(|a| a * a).sum();
            Ok((e, x.mapv(|a| 2.0 * a)))
        }
    }
}
