//! Sella `_gpu.py`: optional GPU eigh / QR / project.
//!
//! The device waist is an rgmin [`vecops::Vector`] tagged with a
//! dlpk CUDA device. This crate does not grow a second GPU stack
//! (no torch). A CUDA tag without a kernel backend is refused and
//! the host Jacobi / Gram-Schmidt path runs, matching `_gpu.py`
//! CPU fallback. Opt-out is `RGSADDLE_DISABLE_GPU` /
//! `SELLA_DISABLE_GPU`. The size floor is `RGSADDLE_GPU_MIN_DIM` /
//! `SELLA_GPU_MIN_DIM` (default 200). A failed CUDA claim records
//! a process-global OOM floor so later calls of that size stay on
//! the host. A missing dlpk kernel is not an OOM.

use std::sync::Mutex;

use dlpk::sys::{DLDevice, DLDeviceType};
use ndarray::{Array1, Array2, ArrayView2};
use rgmin::vecops::{Vector, dot};

use crate::eigensolve::exact_eigh;
use crate::error::SaddleError;
use crate::linalg::modified_gram_schmidt;

/// Sella `SELLA_GPU_MIN_DIM` default.
pub const GPU_MIN_DIM: usize = 200;

static OOM_FLOOR: Mutex<Option<usize>> = Mutex::new(None);
static OOM_TEST_LOCK: Mutex<()> = Mutex::new(());

/// Opt-out / size policy from `_gpu.py`. The OOM floor is
/// process-global ([`oom_floor`]), matching Sella `_oom_floor`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuPolicy {
    /// `SELLA_DISABLE_GPU`. When false the host path always runs.
    pub enabled: bool,
    /// Upload only when `n >= min_dim`.
    pub min_dim: usize,
}

impl Default for GpuPolicy {
    fn default() -> Self {
        Self::from_env()
    }
}

impl GpuPolicy {
    /// Read the Sella / rgsaddle environment gate.
    pub fn from_env() -> Self {
        Self {
            enabled: !env_disabled(),
            min_dim: env_min_dim(),
        }
    }

    /// Disabled policy (`SELLA_DISABLE_GPU=1`).
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            min_dim: GPU_MIN_DIM,
        }
    }

    /// Sella `_gpu_ok(n)`.
    pub fn gpu_ok(&self, n: usize) -> bool {
        gpu_ok(self, n)
    }

    /// Sella `_record_oom(n)`: writes the process-global floor.
    pub fn record_oom(&mut self, n: usize) {
        record_oom(n);
    }
}

/// Sella `_gpu_ok(n)`.
pub fn gpu_ok(policy: &GpuPolicy, n: usize) -> bool {
    policy.enabled
        && cuda_available()
        && n >= policy.min_dim
        && oom_floor().is_none_or(|floor| n < floor)
}

fn flag_disables_gpu(raw: &str) -> bool {
    matches!(raw.to_ascii_lowercase().as_str(), "1" | "true" | "yes")
}

fn min_dim_from_raw(raw: &str) -> Option<usize> {
    raw.parse().ok()
}

fn env_disabled() -> bool {
    for key in ["RGSADDLE_DISABLE_GPU", "SELLA_DISABLE_GPU"] {
        if let Ok(v) = std::env::var(key) {
            if flag_disables_gpu(&v) {
                return true;
            }
        }
    }
    false
}

fn env_min_dim() -> usize {
    for key in ["RGSADDLE_GPU_MIN_DIM", "SELLA_GPU_MIN_DIM"] {
        if let Ok(v) = std::env::var(key) {
            if let Some(n) = min_dim_from_raw(&v) {
                return n;
            }
        }
    }
    GPU_MIN_DIM
}

/// Current Sella `_oom_floor`, if any.
pub fn oom_floor() -> Option<usize> {
    OOM_FLOOR.lock().ok().and_then(|g| *g)
}

/// Sella `_record_oom(n)`: refuse later offload for shapes `>= n`.
pub fn record_oom(n: usize) {
    if let Ok(mut g) = OOM_FLOOR.lock() {
        if g.is_none_or(|floor| n < floor) {
            *g = Some(n);
        }
    }
}

/// Drop the process-wide OOM floor (tests).
pub fn clear_oom_floor() {
    if let Ok(mut g) = OOM_FLOOR.lock() {
        *g = None;
    }
}

/// Hold while a test mutates the process-wide OOM floor.
pub fn lock_oom_for_test() -> std::sync::MutexGuard<'static, ()> {
    OOM_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner())
}

/// dlpk CUDA device 0. The only GPU tag this crate will name.
pub fn cuda_device() -> DLDevice {
    DLDevice {
        device_type: DLDeviceType::kDLCUDA,
        device_id: 0,
    }
}

/// True when `Vector::try_on(kDLCUDA)` has a kernel backend.
pub fn cuda_available() -> bool {
    Vector::try_on(cuda_device(), Array1::zeros(1)).is_ok()
}

/// Sella `to_gpu`: claim host storage for dlpk CUDA.
///
/// A missing CUDA backend is not an OOM. A failed claim after the
/// backend is present records [`record_oom`].
pub fn to_gpu(data: Array1<f64>) -> Option<Vector> {
    if !cuda_available() {
        return None;
    }
    let n = data.len();
    match Vector::try_on(cuda_device(), data) {
        Ok(v) => Some(v),
        Err(_) => {
            record_oom(n);
            None
        }
    }
}

/// Flatten `a` and try the CUDA tag. `None` if the backend is missing.
pub fn to_gpu_matrix(a: ArrayView2<f64>) -> Option<Vector> {
    let flat = Array1::from_iter(a.iter().copied());
    to_gpu(flat)
}

/// Sella `gpu_eigh_t`: stay on device. No kernel in this build.
///
/// A missing kernel is not a Sella OOM (`RuntimeError`/`MemoryError`).
pub fn gpu_eigh_t(_uploaded: &Vector) -> Option<(Array1<f64>, Array2<f64>)> {
    None
}

/// Sella `gpu_eigh` with the process env policy and global `_oom_floor`.
pub fn gpu_eigh_env(a: ArrayView2<f64>) -> Result<(Array1<f64>, Array2<f64>), SaddleError> {
    gpu_eigh(a, &mut GpuPolicy::from_env())
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
        let _ = to_gpu_matrix(a);
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
        let _ = to_gpu_matrix(h);
        let _ = to_gpu_matrix(u);
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
        let p = GpuPolicy {
            enabled: true,
            min_dim: GPU_MIN_DIM,
        };
        assert!(!p.gpu_ok(199));
        if cuda_available() {
            assert!(p.gpu_ok(200));
        } else {
            assert!(!p.gpu_ok(200));
        }
        assert!(!GpuPolicy::disabled().gpu_ok(1000));
    }

    #[test]
    fn default_reads_sella_env_keys() {
        assert_eq!(GpuPolicy::default(), GpuPolicy::from_env());
        assert!(flag_disables_gpu("1"));
        assert!(flag_disables_gpu("TRUE"));
        assert!(flag_disables_gpu("yes"));
        assert!(!flag_disables_gpu("0"));
        assert!(!flag_disables_gpu("false"));
        assert_eq!(min_dim_from_raw("64"), Some(64));
        assert_eq!(min_dim_from_raw("nope"), None);
    }

    #[test]
    fn cuda_tag_is_refused() {
        if !cuda_available() {
            assert!(to_gpu(Array1::zeros(3)).is_none());
        }
        assert_eq!(cuda_device().device_type, DLDeviceType::kDLCUDA);
    }

    #[test]
    fn missing_kernel_after_upload_is_not_oom() {
        let _guard = lock_oom_for_test();
        clear_oom_floor();
        let mut policy = GpuPolicy {
            enabled: true,
            min_dim: 1,
        };
        let a = Array2::<f64>::eye(2);
        let _ = gpu_eigh(a.view(), &mut policy).unwrap();
        let _ = gpu_qr(a.view(), &mut policy).unwrap();
        let _ = gpu_project(a.view(), a.view(), &mut policy).unwrap();
        assert_eq!(oom_floor(), None);
        clear_oom_floor();
    }

    #[test]
    fn oom_floor_is_process_global() {
        let _guard = lock_oom_for_test();
        clear_oom_floor();
        record_oom(128);
        assert_eq!(oom_floor(), Some(128));
        record_oom(256);
        assert_eq!(oom_floor(), Some(128));
        record_oom(64);
        assert_eq!(oom_floor(), Some(64));
        let p = GpuPolicy::from_env();
        assert!(!gpu_ok(&p, 64));
        clear_oom_floor();
        assert_eq!(oom_floor(), None);
    }

    #[test]
    fn gpu_eigh_recovers_a_known_spectrum() {
        let mut a = Array2::<f64>::zeros((2, 2));
        a[(0, 0)] = 2.0;
        a[(1, 1)] = 8.0;
        let mut policy = GpuPolicy {
            enabled: true,
            min_dim: 1,
        };
        let (lams, vecs) = gpu_eigh(a.view(), &mut policy).unwrap();
        assert!((lams[0] - 2.0).abs() < 1e-10);
        assert!((lams[1] - 8.0).abs() < 1e-10);
        let x = vecs.column(0).to_owned();
        let y = Sphere.retract(&x, &Sphere.project(&x, &Array1::from(vec![0.1, -0.2])));
        assert!((nrm2(y.view()) - 1.0).abs() < 1e-12);
    }
}
