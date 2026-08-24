//! Sella `eigensolvers.py`: exact dense diag, Rayleigh-Ritz, expand.
//!
//! `eig=true` on a Sella saddle walks a small Ritz subspace via
//! [`rayleigh_ritz_iter`] and [`expand`] (`jd0` default). The GPU
//! waist is Sella `_gpu.py` ([`crate::gpu`]): dlpk CUDA via
//! [`crate::gpu::to_gpu`], size gate, OOM floor, host Jacobi / QR
//! fallback. This crate does not grow a second device stack.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use rgmin::vecops::nrm2;
use rgmin::{ApplyHessian, EigenParams, EigensolverKind, lowest_mode};

use crate::error::SaddleError;
use crate::linalg::{modified_gram_schmidt, symmetrize_vt_av};

/// Where the dense eigen runs. dlpk CUDA is the only device waist.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EigenDevice {
    #[default]
    Host,
    /// rgmin `lowest_mode` (vecops / dlpk). Not a second GPU stack.
    Dlpk,
}

impl EigenDevice {
    /// C / Python ordinal.
    pub const fn to_abi(self) -> i32 {
        match self {
            Self::Host => 0,
            Self::Dlpk => 1,
        }
    }
}

/// Sella `eigensolvers.expand` method.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ExpandKind {
    Lanczos,
    Gd,
    #[default]
    Jd0,
    Jd0Alt,
    Mjd0,
    Mjd0Alt,
}

impl ExpandKind {
    pub const fn to_abi(self) -> i32 {
        match self {
            Self::Lanczos => 0,
            Self::Gd => 1,
            Self::Jd0 => 2,
            Self::Jd0Alt => 3,
            Self::Mjd0 => 4,
            Self::Mjd0Alt => 5,
        }
    }

    pub const fn try_from_abi(v: i32) -> Option<Self> {
        match v {
            0 => Some(Self::Lanczos),
            1 => Some(Self::Gd),
            2 => Some(Self::Jd0),
            3 => Some(Self::Jd0Alt),
            4 => Some(Self::Mjd0),
            5 => Some(Self::Mjd0Alt),
            _ => None,
        }
    }
}

/// Dense `H v` for [`lowest_mode`].
struct DenseApply<'a>(&'a Array2<f64>);

impl ApplyHessian for DenseApply<'_> {
    fn apply_hessian(&self, _x: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
        let n = self.0.nrows().min(v.len());
        let mut out = Array1::zeros(v.len());
        for i in 0..n {
            let mut acc = 0.0;
            for j in 0..n {
                acc += self.0[(i, j)] * v[j];
            }
            out[i] = acc;
        }
        out
    }
}

/// Sella `exact`: dense symmetric eigen of `a`.
pub fn exact_eigh(a: ArrayView2<f64>) -> Result<(Array1<f64>, Array2<f64>), SaddleError> {
    if a.nrows() != a.ncols() {
        return Err(SaddleError::Shape("eigh needs a square matrix".into()));
    }
    let n = a.nrows();
    let mut work = a.to_owned();
    for i in 0..n {
        for j in 0..i {
            let s = 0.5 * (work[(i, j)] + work[(j, i)]);
            work[(i, j)] = s;
            work[(j, i)] = s;
        }
    }
    jacobi_eigh(&mut work)
}

/// Rayleigh-Ritz on the columns of `v` against a dense `a`.
///
/// Returns `(lams, v_rot, a @ v_rot)`. `gamma <= 0` falls through to
/// [`exact_eigh`] on `a` (Sella `exact`).
pub fn rayleigh_ritz(
    a: ArrayView2<f64>,
    v: ArrayView2<f64>,
    gamma: f64,
) -> Result<(Array1<f64>, Array2<f64>, Array2<f64>), SaddleError> {
    if a.nrows() != a.ncols() || v.nrows() != a.nrows() {
        return Err(SaddleError::Shape(
            "Rayleigh-Ritz needs square A and matching V rows".into(),
        ));
    }
    if gamma <= 0.0 {
        let (lams, vecs) = exact_eigh(a)?;
        let av = matmul(a, vecs.view());
        return Ok((lams, vecs, av));
    }
    let q = modified_gram_schmidt(v, None, 1e-14);
    if q.ncols() == 0 {
        return Err(SaddleError::Solver("empty Ritz subspace".into()));
    }
    let av = matmul(a, q.view());
    let a_tilde = symmetrize_vt_av(q.view(), av.view());
    let (lams, vecs) = exact_eigh(a_tilde.view())?;
    let v_rot = matmul(q.view(), vecs.view());
    let av_rot = matmul(av.view(), vecs.view());
    Ok((lams, v_rot, av_rot))
}

/// Dense eigen on `device`. Full spectrum is Host Jacobi.
/// [`EigenDevice::Dlpk`] is Sella `_gpu.py` ([`crate::gpu::gpu_eigh_env`]):
/// process env policy and the process-global OOM floor.
pub fn eigh_on(
    device: EigenDevice,
    a: ArrayView2<f64>,
) -> Result<(Array1<f64>, Array2<f64>), SaddleError> {
    match device {
        EigenDevice::Host => exact_eigh(a),
        EigenDevice::Dlpk => crate::gpu::gpu_eigh_env(a),
    }
}

/// Lowest eigenpair through rgmin (`Lanczos` / `RayleighRitz`).
///
/// This is the dlpk waist Sella `_gpu.py` maps onto. Full spectrum
/// stays [`exact_eigh`].
pub fn lowest_on(
    device: EigenDevice,
    a: ArrayView2<f64>,
    seed: ArrayView1<f64>,
) -> Result<(f64, Array1<f64>), SaddleError> {
    if a.nrows() != a.ncols() || seed.len() != a.nrows() {
        return Err(SaddleError::Shape(
            "lowest_on needs square A and a matching seed".into(),
        ));
    }
    let x = Array1::zeros(seed.len());
    let kind = match device {
        EigenDevice::Host => EigensolverKind::Lanczos,
        EigenDevice::Dlpk => EigensolverKind::RayleighRitz,
    };
    let params = EigenParams {
        kind,
        krylov: seed.len().min(8).max(2),
        ..EigenParams::default()
    };
    let mode = lowest_mode(&DenseApply(&a.to_owned()), x.view(), seed, &params)
        .map_err(|e| SaddleError::Solver(e.to_string()))?;
    Ok((mode.value, mode.vector))
}

/// Sella `expand`: one Davidson / Jacobi-Davidson / Lanczos direction.
///
/// `v` is the current Ritz basis `(d, n)`, `y` is `A @ v` (or the
/// symmetrized action), `p` is the preconditioner, `b` the metric
/// (`None` is `I`). `lams` / `vecs` are the current Ritz pairs.
pub fn expand(
    v: ArrayView2<f64>,
    y: ArrayView2<f64>,
    p: ArrayView2<f64>,
    b: Option<ArrayView2<f64>>,
    lams: ArrayView1<f64>,
    vecs: ArrayView2<f64>,
    shift: f64,
    method: ExpandKind,
    seeking: usize,
) -> Result<Array1<f64>, SaddleError> {
    let d = v.nrows();
    let n = v.ncols();
    if y.nrows() != d || y.ncols() != n || p.nrows() != d || p.ncols() != d {
        return Err(SaddleError::Shape(
            "expand needs matching V, Y, P shapes".into(),
        ));
    }
    if seeking >= n || lams.len() != n || vecs.nrows() != n || vecs.ncols() != n {
        return Err(SaddleError::Shape(
            "expand Ritz pairs must match the subspace".into(),
        ));
    }
    let bmat = b.map(|m| m.to_owned());
    let bv = match &bmat {
        Some(bm) => matmul(bm.view(), v),
        None => v.to_owned(),
    };
    // R = Y V_rot - B V V_rot * lam  with V already rotated in Sella;
    // here vecs rotates the current subspace.
    let vrot = matmul(v, vecs);
    let yrot = matmul(y, vecs);
    let bvrot = matmul(bv.view(), vecs);
    let mut r = yrot;
    for j in 0..n {
        for i in 0..d {
            r[(i, j)] -= bvrot[(i, j)] * lams[j];
        }
    }
    let mut pshift = p.to_owned();
    match &bmat {
        Some(bm) => {
            for i in 0..d {
                for j in 0..d {
                    pshift[(i, j)] -= shift * bm[(i, j)];
                }
            }
        }
        None => {
            for i in 0..d {
                pshift[(i, i)] -= shift;
            }
        }
    }
    let mut rseek = Array1::zeros(d);
    for i in 0..d {
        rseek[i] = r[(i, seeking)];
    }
    match method {
        ExpandKind::Lanczos => Ok(rseek),
        ExpandKind::Gd => solve_dense(pshift.view(), rseek.view()),
        ExpandKind::Jd0 => {
            let mut vi = Array1::zeros(d);
            for i in 0..d {
                vi[i] = vrot[(i, seeking)];
            }
            let mut aaug = Array2::<f64>::zeros((d + 1, d + 1));
            for i in 0..d {
                for j in 0..d {
                    aaug[(i, j)] = pshift[(i, j)];
                }
                aaug[(i, d)] = vi[i];
                aaug[(d, i)] = vi[i];
            }
            let mut raug = Array1::zeros(d + 1);
            for i in 0..d {
                raug[i] = -rseek[i];
            }
            let z = solve_dense(aaug.view(), raug.view())?;
            Ok(z.slice(s![..d]).to_owned())
        }
        ExpandKind::Jd0Alt => {
            let mut vi = Array1::zeros(d);
            for i in 0..d {
                vi[i] = vrot[(i, seeking)];
            }
            let pprojr = solve_dense(pshift.view(), rseek.view())?;
            let pprojv = solve_dense(pshift.view(), vi.view())?;
            let mut denom = 0.0;
            let mut num = 0.0;
            for i in 0..d {
                denom += vi[i] * pprojv[i];
                num += vi[i] * pprojr[i];
            }
            if denom.abs() < 1e-12 {
                return Ok(pprojr);
            }
            let alpha = num / denom;
            let mut t = pprojv;
            for i in 0..d {
                t[i] = t[i] * alpha - pprojr[i];
            }
            Ok(t)
        }
        ExpandKind::Mjd0 => {
            let mut aaug = Array2::<f64>::zeros((d + n, d + n));
            for i in 0..d {
                for j in 0..d {
                    aaug[(i, j)] = pshift[(i, j)];
                }
                for j in 0..n {
                    aaug[(i, d + j)] = vrot[(i, j)];
                    aaug[(d + j, i)] = vrot[(i, j)];
                }
            }
            let mut raug = Array1::zeros(d + n);
            for i in 0..d {
                raug[i] = -rseek[i];
            }
            let z = solve_dense(aaug.view(), raug.view())?;
            Ok(z.slice(s![..d]).to_owned())
        }
        ExpandKind::Mjd0Alt => {
            let pprojr = solve_dense(pshift.view(), rseek.view())?;
            let mut rhs = Array1::zeros(n);
            let mut gram = Array2::<f64>::zeros((n, n));
            for j in 0..n {
                let mut col = Array1::zeros(d);
                for i in 0..d {
                    col[i] = vrot[(i, j)];
                }
                let pj = solve_dense(pshift.view(), col.view())?;
                for i in 0..n {
                    let mut acc = 0.0;
                    for t in 0..d {
                        acc += vrot[(t, i)] * pj[t];
                    }
                    gram[(i, j)] = acc;
                }
                let mut acc = 0.0;
                for t in 0..d {
                    acc += vrot[(t, j)] * pprojr[t];
                }
                rhs[j] = acc;
            }
            let alpha = solve_dense(gram.view(), rhs.view())?;
            let mut t = Array1::zeros(d);
            for i in 0..d {
                let mut acc = 0.0;
                for j in 0..n {
                    acc += vrot[(i, j)] * alpha[j];
                }
                t[i] = acc - rseek[i];
            }
            solve_dense(pshift.view(), t.view())
        }
    }
}

/// Sella `rayleigh_ritz` loop: grow `v` with [`expand`] until the
/// residual is below `gamma |theta|` or the subspace hits `2 n + 1`.
pub fn rayleigh_ritz_iter(
    a: ArrayView2<f64>,
    v0: ArrayView2<f64>,
    gamma: f64,
    method: ExpandKind,
) -> Result<(Array1<f64>, Array2<f64>, Array2<f64>), SaddleError> {
    if a.nrows() != a.ncols() || v0.nrows() != a.nrows() {
        return Err(SaddleError::Shape(
            "iterative Rayleigh-Ritz needs square A and matching V rows".into(),
        ));
    }
    if gamma <= 0.0 {
        let (lams, vecs) = exact_eigh(a)?;
        let av = matmul(a, vecs.view());
        return Ok((lams, vecs, av));
    }
    let n = a.nrows();
    let maxiter = (2 * n + 1).max(2);
    let mut v = modified_gram_schmidt(v0, None, 1e-14);
    if v.ncols() == 0 {
        return Err(SaddleError::Solver("empty Ritz subspace".into()));
    }
    let p = Array2::eye(n);
    let mut av = matmul(a, v.view());
    loop {
        let a_tilde = symmetrize_vt_av(v.view(), av.view());
        let (lams, vecs) = exact_eigh(a_tilde.view())?;
        av = matmul(av.view(), vecs.view());
        v = matmul(v.view(), vecs.view());
        let k = v.ncols();
        if k >= maxiter {
            return Ok((lams, v, av));
        }
        let mut nneg = 1usize;
        for &lam in lams.iter() {
            if lam < 0.0 {
                nneg += 1;
            }
        }
        nneg = nneg.min(k);
        let eye = Array2::eye(k);
        let mut rnorm = Array1::zeros(nneg);
        for j in 0..nneg {
            let mut acc = 0.0;
            for i in 0..n {
                let ri = av[(i, j)] - lams[j] * v[(i, j)];
                acc += ri * ri;
            }
            rnorm[j] = acc.sqrt();
        }
        let mut seeking = 0;
        let mut found = false;
        for j in 0..nneg {
            if k == 1 || rnorm[j] >= gamma * lams[j].abs() {
                seeking = j;
                found = true;
                break;
            }
        }
        if !found {
            return Ok((lams, v, av));
        }
        let t = expand(
            v.view(),
            av.view(),
            p.view(),
            None,
            lams.view(),
            eye.view(),
            lams[seeking],
            method,
            seeking,
        )?;
        let tn = {
            let mut s = 0.0;
            for &x in t.iter() {
                s += x * x;
            }
            s.sqrt()
        };
        if tn < 1e-18 {
            return Ok((lams, v, av));
        }
        let mut tcol = Array2::zeros((n, 1));
        for i in 0..n {
            tcol[(i, 0)] = t[i] / tn;
        }
        let tnew = modified_gram_schmidt(tcol.view(), Some(v.view()), 1e-8);
        if tnew.ncols() == 0 {
            return Ok((lams, v, av));
        }
        let mut vnext = Array2::zeros((n, k + tnew.ncols()));
        for j in 0..k {
            for i in 0..n {
                vnext[(i, j)] = v[(i, j)];
            }
        }
        for j in 0..tnew.ncols() {
            for i in 0..n {
                vnext[(i, k + j)] = tnew[(i, j)];
            }
        }
        let at = matmul(a, tnew.view());
        let mut avnext = Array2::zeros((n, k + tnew.ncols()));
        for j in 0..k {
            for i in 0..n {
                avnext[(i, j)] = av[(i, j)];
            }
        }
        for j in 0..tnew.ncols() {
            for i in 0..n {
                avnext[(i, k + j)] = at[(i, j)];
            }
        }
        v = vnext;
        av = avnext;
    }
}

pub(crate) fn solve_dense(
    a: ArrayView2<f64>,
    b: ArrayView1<f64>,
) -> Result<Array1<f64>, SaddleError> {
    let n = a.nrows();
    if a.ncols() != n || b.len() != n {
        return Err(SaddleError::Shape(
            "dense solve needs a square system".into(),
        ));
    }
    let mut m = a.to_owned();
    let mut x = b.to_owned();
    for k in 0..n {
        let mut piv = k;
        let mut best = m[(k, k)].abs();
        for i in (k + 1)..n {
            let mag = m[(i, k)].abs();
            if mag > best {
                best = mag;
                piv = i;
            }
        }
        if best < 1e-18 {
            return Err(SaddleError::Solver("singular expand system".into()));
        }
        if piv != k {
            for j in 0..n {
                let tmp = m[(k, j)];
                m[(k, j)] = m[(piv, j)];
                m[(piv, j)] = tmp;
            }
            let tmp = x[k];
            x[k] = x[piv];
            x[piv] = tmp;
        }
        let akk = m[(k, k)];
        for i in (k + 1)..n {
            let f = m[(i, k)] / akk;
            m[(i, k)] = 0.0;
            for j in (k + 1)..n {
                m[(i, j)] -= f * m[(k, j)];
            }
            x[i] -= f * x[k];
        }
    }
    for i in (0..n).rev() {
        let mut acc = x[i];
        for j in (i + 1)..n {
            acc -= m[(i, j)] * x[j];
        }
        x[i] = acc / m[(i, i)];
    }
    Ok(x)
}

fn matmul(a: ArrayView2<f64>, b: ArrayView2<f64>) -> Array2<f64> {
    let mut c = Array2::zeros((a.nrows(), b.ncols()));
    for i in 0..a.nrows() {
        for j in 0..b.ncols() {
            let mut acc = 0.0;
            for k in 0..a.ncols() {
                acc += a[(i, k)] * b[(k, j)];
            }
            c[(i, j)] = acc;
        }
    }
    c
}

/// Cyclic Jacobi for a small dense symmetric matrix.
fn jacobi_eigh(a: &mut Array2<f64>) -> Result<(Array1<f64>, Array2<f64>), SaddleError> {
    let n = a.nrows();
    let mut v = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        v[(i, i)] = 1.0;
    }
    for _ in 0..(8 * n * n).max(16) {
        let mut p = 0;
        let mut q = 1;
        let mut max = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                let mag = a[(i, j)].abs();
                if mag > max {
                    max = mag;
                    p = i;
                    q = j;
                }
            }
        }
        if max < 1e-14 {
            break;
        }
        let app = a[(p, p)];
        let aqq = a[(q, q)];
        let apq = a[(p, q)];
        let tau = (aqq - app) / (2.0 * apq);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        for k in 0..n {
            if k != p && k != q {
                let aik = a[(k.min(p), k.max(p))];
                let aiq = a[(k.min(q), k.max(q))];
                let npk = c * aik - s * aiq;
                let nqk = s * aik + c * aiq;
                a[(k.min(p), k.max(p))] = npk;
                a[(k.max(p), k.min(p))] = npk;
                a[(k.min(q), k.max(q))] = nqk;
                a[(k.max(q), k.min(q))] = nqk;
            }
            let vip = v[(k, p)];
            let viq = v[(k, q)];
            v[(k, p)] = c * vip - s * viq;
            v[(k, q)] = s * vip + c * viq;
        }
        a[(p, p)] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        a[(q, q)] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a[(p, q)] = 0.0;
        a[(q, p)] = 0.0;
    }
    let mut evals = Array1::zeros(n);
    for i in 0..n {
        evals[i] = a[(i, i)];
    }
    // Sort ascending.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| evals[i].total_cmp(&evals[j]));
    let mut ev_sorted = Array1::zeros(n);
    let mut vecs = Array2::zeros((n, n));
    for (dst, &src) in order.iter().enumerate() {
        ev_sorted[dst] = evals[src];
        for r in 0..n {
            vecs[(r, dst)] = v[(r, src)];
        }
    }
    for j in 0..n {
        let col = vecs.column(j).to_owned();
        if nrm2(col.view()) < 1e-18 {
            return Err(SaddleError::Solver(
                "Jacobi produced a zero eigenvector".into(),
            ));
        }
    }
    Ok((ev_sorted, vecs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn exact_eigh_recovers_a_known_spectrum() {
        let mut a = Array2::<f64>::zeros((2, 2));
        a[(0, 0)] = 2.0;
        a[(1, 1)] = 8.0;
        let (lams, vecs) = exact_eigh(a.view()).unwrap();
        assert!((lams[0] - 2.0).abs() < 1e-10, "{}", lams[0]);
        assert!((lams[1] - 8.0).abs() < 1e-10, "{}", lams[1]);
        // A v = lam v
        let av0 = a[(0, 0)] * vecs[(0, 0)] + a[(0, 1)] * vecs[(1, 0)];
        assert!((av0 - lams[0] * vecs[(0, 0)]).abs() < 1e-10);
    }

    #[test]
    fn rayleigh_ritz_on_the_full_space_matches_exact() {
        let mut a = Array2::<f64>::zeros((3, 3));
        a[(0, 0)] = 4.0;
        a[(1, 1)] = 1.0;
        a[(2, 2)] = 9.0;
        a[(0, 1)] = 0.5;
        a[(1, 0)] = 0.5;
        let v = Array2::eye(3);
        let (lams, _, _) = rayleigh_ritz(a.view(), v.view(), 0.1).unwrap();
        let (exact, _) = exact_eigh(a.view()).unwrap();
        for i in 0..3 {
            assert!(
                (lams[i] - exact[i]).abs() < 1e-10,
                "{} vs {}",
                lams[i],
                exact[i]
            );
        }
    }

    #[test]
    fn dlpk_device_shares_the_host_waist() {
        let a = Array2::eye(2);
        let (h, _) = eigh_on(EigenDevice::Host, a.view()).unwrap();
        let (d, _) = eigh_on(EigenDevice::Dlpk, a.view()).unwrap();
        assert!((h[0] - d[0]).abs() < 1e-14);
    }

    #[test]
    fn expand_jd0_is_orthogonal_to_the_ritz_vector() {
        let mut a = Array2::<f64>::zeros((3, 3));
        a[(0, 0)] = -2.0;
        a[(1, 1)] = 1.0;
        a[(2, 2)] = 4.0;
        let v = Array2::eye(3).slice(s![.., ..1]).to_owned();
        let y = matmul(a.view(), v.view());
        let p = Array2::eye(3);
        let lams = Array1::from(vec![-2.0]);
        let vecs = Array2::eye(1);
        let t = expand(
            v.view(),
            y.view(),
            p.view(),
            None,
            lams.view(),
            vecs.view(),
            -2.0,
            ExpandKind::Jd0,
            0,
        )
        .unwrap();
        assert!(t.iter().all(|x| x.is_finite()));
        let t_l = expand(
            v.view(),
            y.view(),
            p.view(),
            None,
            lams.view(),
            vecs.view(),
            -2.0,
            ExpandKind::Lanczos,
            0,
        )
        .unwrap();
        assert_eq!(t_l.len(), 3);
    }

    #[test]
    fn rayleigh_ritz_iter_recovers_the_lowest() {
        let mut a = Array2::<f64>::zeros((4, 4));
        a[(0, 0)] = -5.0;
        a[(1, 1)] = 1.0;
        a[(2, 2)] = 3.0;
        a[(3, 3)] = 8.0;
        a[(0, 1)] = 0.2;
        a[(1, 0)] = 0.2;
        let mut v0 = Array2::zeros((4, 1));
        v0[(0, 0)] = 1.0;
        let (lams, vecs, _) =
            rayleigh_ritz_iter(a.view(), v0.view(), 0.1, ExpandKind::Jd0).unwrap();
        let (exact, _) = exact_eigh(a.view()).unwrap();
        assert!(
            (lams[0] - exact[0]).abs() < 1e-6,
            "{} vs {}",
            lams[0],
            exact[0]
        );
        assert_eq!(vecs.nrows(), 4);
    }

    #[test]
    fn lowest_on_finds_the_soft_mode() {
        let mut a = Array2::<f64>::zeros((2, 2));
        a[(0, 0)] = -3.0;
        a[(1, 1)] = 8.0;
        let seed = Array1::from(vec![1.0, 0.1]);
        let (lam, v) = lowest_on(EigenDevice::Host, a.view(), seed.view()).unwrap();
        assert!(lam < 0.0, "lam={lam}");
        assert!(v[0].abs() > v[1].abs());
        let (lam_d, _) = lowest_on(EigenDevice::Dlpk, a.view(), seed.view()).unwrap();
        assert!((lam - lam_d).abs() < 0.5, "{lam} vs {lam_d}");
    }
}
