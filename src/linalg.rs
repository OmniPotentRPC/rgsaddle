//! Sella `linalg.py` helpers the steppers actually call.
//!
//! Finite-difference Hessian-vector products (`NumericalHessian`) and
//! modified Gram-Schmidt. Dense eigen lives in [`crate::eigensolve`].
//! Prefer rgmin / ndarray; this module does not grow a second LA stack.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
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

/// Two-sided (or one-sided) Hessian-vector product of a surface.
pub fn numerical_hvp<S: PointSurface>(
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
    let vn = nrm2(v);
    if vn < 1e-12 {
        return Ok(Array1::zeros(x0.len()));
    }
    let sign = hessian_vec_sign(v, g0, x0);
    let scale = vn * sign;
    let mut xp = x0.to_owned();
    axpy(eta / scale, v, &mut xp);
    let (_, gplus) = surface.eval(xp.view())?;
    if threepoint {
        let mut xm = x0.to_owned();
        axpy(-eta / scale, v, &mut xm);
        let (_, gminus) = surface.eval(xm.view())?;
        let mut av = gplus;
        axpy(-1.0, gminus.view(), &mut av);
        av.mapv_inplace(|a| a * scale / (2.0 * eta));
        Ok(av)
    } else {
        let mut av = gplus;
        axpy(-1.0, g0, &mut av);
        av.mapv_inplace(|a| a * scale / eta);
        Ok(av)
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
            let mut acc = 0.0;
            for t in 0..v.nrows() {
                acc += v[(t, i)] * av[(t, j)];
            }
            a[(i, j)] = acc;
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
}
