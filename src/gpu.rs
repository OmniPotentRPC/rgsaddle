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
//! the host. A failed matrix claim records `A.shape[0]` (`nrows`),
//! not the flattened length. `to_gpu` itself returns `None` when
//! `SELLA_DISABLE_GPU=1`. A missing dlpk kernel is not an OOM.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

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

static FAIL_NEXT_CLAIM: AtomicBool = AtomicBool::new(false);

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

/// Hold while a test mutates process-wide GPU state (OOM floor or env).
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

/// Next [`to_gpu`] / [`to_gpu_matrix`] claim fails as a CUDA OOM.
///
/// Exercises Sella `_record_oom(A.shape[0])` without a device.
pub fn fail_next_cuda_claim() {
    FAIL_NEXT_CLAIM.store(true, Ordering::SeqCst);
}

/// Sella `to_gpu`: claim host storage for dlpk CUDA.
///
/// `SELLA_DISABLE_GPU=1` (or `RGSADDLE_DISABLE_GPU`) returns `None`
/// and does not upload (`sella/_gpu.py:24-25,61`). A missing CUDA
/// backend is not an OOM. A failed claim after the backend is
/// present records [`record_oom`] with `A.shape[0]`.
pub fn to_gpu(data: Array1<f64>) -> Option<Vector> {
    let n = data.len();
    try_cuda(data, n)
}

/// Flatten `a` and try the CUDA tag.
///
/// A failed claim records [`record_oom`] with `a.nrows()` (Sella
/// `A.shape[0]`), never the flattened `n * n`.
pub fn to_gpu_matrix(a: ArrayView2<f64>) -> Option<Vector> {
    let n = a.nrows();
    let flat = Array1::from_iter(a.iter().copied());
    try_cuda(flat, n)
}

fn try_cuda(data: Array1<f64>, oom_n: usize) -> Option<Vector> {
    if env_disabled() {
        let _ = FAIL_NEXT_CLAIM.swap(false, Ordering::SeqCst);
        return None;
    }
    if FAIL_NEXT_CLAIM.swap(false, Ordering::SeqCst) {
        record_oom(oom_n);
        return None;
    }
    if !cuda_available() {
        return None;
    }
    match Vector::try_on(cuda_device(), data) {
        Ok(v) => Some(v),
        Err(_) => {
            record_oom(oom_n);
            None
        }
    }
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

    struct EnvRestore {
        sella: Option<String>,
        rg: Option<String>,
    }

    impl EnvRestore {
        /// Isolate `SELLA_DISABLE_GPU=1` from `RGSADDLE_DISABLE_GPU`.
        fn disable_via_sella() -> Self {
            let this = Self {
                sella: std::env::var("SELLA_DISABLE_GPU").ok(),
                rg: std::env::var("RGSADDLE_DISABLE_GPU").ok(),
            };
            // SAFETY: caller holds OOM_TEST_LOCK, so GPU tests do not race on env.
            unsafe {
                std::env::remove_var("RGSADDLE_DISABLE_GPU");
                std::env::set_var("SELLA_DISABLE_GPU", "1");
            }
            this
        }
    }

    impl Drop for EnvRestore {
        fn drop(&mut self) {
            unsafe {
                match &self.sella {
                    Some(v) => std::env::set_var("SELLA_DISABLE_GPU", v),
                    None => std::env::remove_var("SELLA_DISABLE_GPU"),
                }
                match &self.rg {
                    Some(v) => std::env::set_var("RGSADDLE_DISABLE_GPU", v),
                    None => std::env::remove_var("RGSADDLE_DISABLE_GPU"),
                }
            }
        }
    }

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
        let _guard = lock_oom_for_test();
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
    fn to_gpu_returns_none_when_sella_disable_gpu() {
        let _guard = lock_oom_for_test();
        clear_oom_floor();
        let _env = EnvRestore::disable_via_sella();
        fail_next_cuda_claim();
        assert!(
            to_gpu(Array1::zeros(4)).is_none(),
            "SELLA_DISABLE_GPU=1 must make to_gpu return None"
        );
        assert!(to_gpu_matrix(Array2::<f64>::zeros((4, 4)).view()).is_none());
        assert_eq!(
            oom_floor(),
            None,
            "disabled to_gpu must not upload or record OOM"
        );
        clear_oom_floor();
    }

    #[test]
    fn to_gpu_matrix_oom_floor_is_n_not_n_squared() {
        let _guard = lock_oom_for_test();
        clear_oom_floor();
        let n = 8;
        fail_next_cuda_claim();
        let a = Array2::<f64>::zeros((n, n));
        assert!(to_gpu_matrix(a.view()).is_none());
        assert_eq!(oom_floor(), Some(n), "Sella to_gpu records A.shape[0]");
        assert_ne!(oom_floor(), Some(n * n));
        let p = GpuPolicy {
            enabled: true,
            min_dim: 1,
        };
        assert!(!gpu_ok(&p, n));
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
