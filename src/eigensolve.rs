//! Sella `eigensolvers.py`: exact dense diag and Rayleigh-Ritz.
//!
//! `eig=true` on a Sella saddle walks a small Ritz subspace. The
//! GPU waist is dlpk CUDA ([`EigenDevice::Dlpk`] is the same Host
//! path until that feature is linked); this crate does not grow a
//! second device stack.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rgmin::{lowest_mode, ApplyHessian, EigenParams, EigensolverKind};
use rgmin::vecops::nrm2;

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
/// [`EigenDevice::Dlpk`] is the lowest-mode arm through rgmin.
pub fn eigh_on(
    device: EigenDevice,
    a: ArrayView2<f64>,
) -> Result<(Array1<f64>, Array2<f64>), SaddleError> {
    match device {
        EigenDevice::Host => exact_eigh(a),
        EigenDevice::Dlpk => {
            let (lams, vecs) = exact_eigh(a)?;
            Ok((lams, vecs))
        }
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
            return Err(SaddleError::Solver("Jacobi produced a zero eigenvector".into()));
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
            assert!((lams[i] - exact[i]).abs() < 1e-10, "{} vs {}", lams[i], exact[i]);
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
