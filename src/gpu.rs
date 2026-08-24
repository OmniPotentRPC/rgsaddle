//! Sella `_gpu.py`: optional GPU eigh / QR / project.
//!
//! The device waist is an rgmin [`vecops::Vector`] tagged with a
//! dlpk CUDA device. This crate does not grow a second GPU stack
//! (no torch). A CUDA tag without a kernel backend is refused and
//! the host Jacobi / Gram-Schmidt path runs, matching `_gpu.py`
//! CPU fallback. Size-gating and the OOM floor are the Sella
//! `SELLA_GPU_MIN_DIM` / `_oom_floor` contract.

use dlpk::sys::{DLDevice, DLDeviceType};
use ndarray::{Array1, Array2, ArrayView2};
use rgmin::vecops::{Vector, dot};

use crate::eigensolve::exact_eigh;
use crate::error::SaddleError;
use crate::linalg::modified_gram_schmidt;

/// Sella `SELLA_GPU_MIN_DIM` default.
pub const GPU_MIN_DIM: usize = 200;

/// Opt-out / size / OOM policy from `_gpu.py`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuPolicy {
    /// `SELLA_DISABLE_GPU`. When false the host path always runs.
    pub enabled: bool,
    /// Upload only when `n >= min_dim`.
    pub min_dim: usize,
    /// After a failure at dimension `N`, refuse `n >= N`.
    pub oom_floor: Option<usize>,
}

impl Default for GpuPolicy {
    fn default() -> Self {
        Self {
            enabled: true,
            min_dim: GPU_MIN_DIM,
            oom_floor: None,
        }
    }
}

impl GpuPolicy {
    /// Disabled policy (`SELLA_DISABLE_GPU=1`).
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }

    /// Sella `_gpu_ok(n)`.
    pub fn gpu_ok(&self, n: usize) -> bool {
        gpu_ok(self, n)
    }

    /// Sella `_record_oom(n)`.
    pub fn record_oom(&mut self, n: usize) {
        match self.oom_floor {
            None => self.oom_floor = Some(n),
            Some(floor) if n < floor => self.oom_floor = Some(n),
            _ => {}
        }
    }
}

/// Sella `_gpu_ok(n)`.
pub fn gpu_ok(policy: &GpuPolicy, n: usize) -> bool {
    policy.enabled && n >= policy.min_dim && policy.oom_floor.map(|floor| n < floor).unwrap_or(true)
}

/// dlpk CUDA device 0. The only GPU tag this crate will name.
pub fn cuda_device() -> DLDevice {
    DLDevice {
        device_type: DLDeviceType::kDLCUDA,
        device_id: 0,
    }
}

/// Sella `to_gpu`: claim host storage for dlpk CUDA.
///
/// Returns `None` when this build has no CUDA kernel backend.
/// The refusal is loud: device data is never staged through the host.
pub fn to_gpu(data: Array1<f64>) -> Option<Vector> {
    Vector::try_on(cuda_device(), data).ok()
}

/// Flatten `a` and try the CUDA tag. `None` if the backend is missing.
pub fn to_gpu_matrix(a: ArrayView2<f64>) -> Option<Vector> {
    let flat = Array1::from_iter(a.iter().copied());
    to_gpu(flat)
}

/// Sella `gpu_eigh_t`: stay on device. No kernel in this build.
pub fn gpu_eigh_t(_uploaded: &Vector) -> Option<(Array1<f64>, Array2<f64>)> {
    None
}

/// Sella `gpu_eigh`. GPU when beneficial, host Jacobi otherwise.
pub fn gpu_eigh(
    a: ArrayView2<f64>,
    policy: &mut GpuPolicy,
) -> Result<(Array1<f64>, Array2<f64>), SaddleError> {
    if a.nrows() != a.ncols() {
        return Err(SaddleError::Shape("gpu_eigh needs a square matrix".into()));
    }
    let n = a.nrows();
    if policy.gpu_ok(n) {
        if let Some(dev) = to_gpu_matrix(a) {
            if let Some(pair) = gpu_eigh_t(&dev) {
                return Ok(pair);
            }
            policy.record_oom(n);
        }
    }
    exact_eigh(a)
}

/// Sella `gpu_qr`: economy QR. GPU when beneficial, host MGS otherwise.
///
/// `Q` has orthonormal columns (Stiefel). `R = Q^T A`.
pub fn gpu_qr(
    a: ArrayView2<f64>,
    policy: &mut GpuPolicy,
) -> Result<(Array2<f64>, Array2<f64>), SaddleError> {
    let n = a.nrows();
    if policy.gpu_ok(n) {
        if to_gpu_matrix(a).is_some() {
            // A CUDA handle with no QR kernel is an OOM-equivalent miss.
            policy.record_oom(n);
        }
    }
    host_qr(a)
}

/// Sella `gpu_project`: `U^T H U` via rgmin vecops reductions.
pub fn gpu_project(
    h: ArrayView2<f64>,
    u: ArrayView2<f64>,
    policy: &mut GpuPolicy,
) -> Result<Array2<f64>, SaddleError> {
    if h.nrows() != h.ncols() || u.nrows() != h.nrows() {
        return Err(SaddleError::Shape(
            "gpu_project needs square H and matching U rows".into(),
        ));
    }
    let n = h.nrows();
    if policy.gpu_ok(n) {
        if to_gpu_matrix(h).is_some() && to_gpu_matrix(u).is_some() {
            policy.record_oom(n);
        }
    }
    Ok(host_project(h, u))
}

fn host_qr(a: ArrayView2<f64>) -> Result<(Array2<f64>, Array2<f64>), SaddleError> {
    if a.ncols() == 0 || a.nrows() == 0 {
        return Err(SaddleError::Shape("gpu_qr needs a nonempty matrix".into()));
    }
    let q = modified_gram_schmidt(a, None, 1e-14);
    if q.ncols() == 0 {
        return Err(SaddleError::Solver("gpu_qr empty range".into()));
    }
    let k = q.ncols();
    let n = a.ncols();
    let mut r = Array2::<f64>::zeros((k, n));
    for i in 0..k {
        for j in 0..n {
            r[(i, j)] = dot(q.column(i), a.column(j));
        }
    }
    Ok((q, r))
}

fn host_project(h: ArrayView2<f64>, u: ArrayView2<f64>) -> Array2<f64> {
    let n = h.nrows();
    let k = u.ncols();
    let mut hu = Array2::<f64>::zeros((n, k));
    for j in 0..k {
        for i in 0..n {
            hu[(i, j)] = dot(h.row(i), u.column(j));
        }
    }
    let mut out = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in 0..k {
            out[(i, j)] = dot(u.column(i), hu.column(j));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use rgmin::manifold::{Manifold, Sphere};
    use rgmin::vecops::nrm2;

    #[test]
    fn default_policy_gates_at_two_hundred() {
        let p = GpuPolicy::default();
        assert!(!p.gpu_ok(199));
        assert!(p.gpu_ok(200));
        assert!(!GpuPolicy::disabled().gpu_ok(1000));
    }

    #[test]
    fn cuda_tag_is_refused() {
        assert!(to_gpu(Array1::zeros(3)).is_none());
        assert_eq!(cuda_device().device_type, DLDeviceType::kDLCUDA);
    }

    #[test]
    fn gpu_eigh_recovers_a_known_spectrum() {
        let mut a = Array2::<f64>::zeros((2, 2));
        a[(0, 0)] = 2.0;
        a[(1, 1)] = 8.0;
        let mut policy = GpuPolicy {
            enabled: true,
            min_dim: 1,
            oom_floor: None,
        };
        let (lams, vecs) = gpu_eigh(a.view(), &mut policy).unwrap();
        assert!((lams[0] - 2.0).abs() < 1e-10);
        assert!((lams[1] - 8.0).abs() < 1e-10);
        let x = vecs.column(0).to_owned();
        let y = Sphere.retract(&x, &Sphere.project(&x, &Array1::from(vec![0.1, -0.2])));
        assert!((nrm2(y.view()) - 1.0).abs() < 1e-12);
    }
}
