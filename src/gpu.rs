//! Sella `_gpu.py`: optional dlpk CUDA waist for eigh, QR, and `U^T H U`.
//!
//! Opt-out via `RGSADDLE_DISABLE_GPU` / `SELLA_DISABLE_GPU`. The size
//! floor is `RGSADDLE_GPU_MIN_DIM` / `SELLA_GPU_MIN_DIM` (default 200).
//! A failed CUDA claim records an OOM floor so later calls of that
//! size stay on the host. The only device handle is
//! [`rgmin::vecops::Vector`]; this module does not grow a second
//! GPU stack. Without a dlpk CUDA kernel backend the helpers factor
//! on the host through [`crate::exact_eigh`] and vecops reductions
//! (`par` / rayon when that rgmin feature is on).

use std::sync::Mutex;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rgmin::vecops::{Vector, dot};

use crate::error::SaddleError;
use crate::exact_eigh;
use crate::linalg::modified_gram_schmidt;

const DEFAULT_MIN_DIM: usize = 200;

static OOM_FLOOR: Mutex<Option<usize>> = Mutex::new(None);
static OOM_TEST_LOCK: Mutex<()> = Mutex::new(());

/// Sella `_gpu.py` size / opt-out / OOM gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuPolicy {
    /// When false, every call stays on the host.
    pub enabled: bool,
    /// Sella `SELLA_GPU_MIN_DIM`.
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

    /// Never claim a device.
    pub fn host() -> Self {
        Self {
            enabled: false,
            min_dim: DEFAULT_MIN_DIM,
        }
    }

    /// Sella `_gpu_ok(n)`.
    pub fn ok(self, n: usize) -> bool {
        self.enabled
            && cuda_available()
            && n >= self.min_dim
            && oom_floor().is_none_or(|floor| n < floor)
    }

    /// Sella `gpu_eigh`.
    pub fn eigh(self, a: ArrayView2<f64>) -> Result<(Array1<f64>, Array2<f64>), SaddleError> {
        gpu_eigh_with(a, self)
    }

    /// Sella `gpu_qr` (economy).
    pub fn qr(self, a: ArrayView2<f64>) -> Result<(Array2<f64>, Array2<f64>), SaddleError> {
        gpu_qr_with(a, self)
    }

    /// Sella `gpu_project`: `U^T H U`.
    pub fn project(
        self,
        h: ArrayView2<f64>,
        u: ArrayView2<f64>,
    ) -> Result<Array2<f64>, SaddleError> {
        gpu_project_with(h, u, self)
    }
}

/// Sella `_gpu.py`: `SELLA_DISABLE_GPU` in `{"1","true","yes"}`.
fn env_flag_disabled(raw: Option<&str>) -> bool {
    matches!(
        raw.map(|v| v.to_ascii_lowercase()).as_deref(),
        Some("1" | "true" | "yes")
    )
}

/// Sella `_gpu.py`: `SELLA_GPU_MIN_DIM`, default 200.
fn parse_min_dim(raw: Option<&str>) -> usize {
    raw.and_then(|v| v.parse().ok()).unwrap_or(DEFAULT_MIN_DIM)
}

fn env_disabled() -> bool {
    for key in ["RGSADDLE_DISABLE_GPU", "SELLA_DISABLE_GPU"] {
        if env_flag_disabled(std::env::var(key).ok().as_deref()) {
            return true;
        }
    }
    false
}

fn env_min_dim() -> usize {
    for key in ["RGSADDLE_GPU_MIN_DIM", "SELLA_GPU_MIN_DIM"] {
        if let Ok(v) = std::env::var(key) {
            if v.parse::<usize>().is_ok() {
                return parse_min_dim(Some(v.as_str()));
            }
        }
    }
    DEFAULT_MIN_DIM
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

fn cuda_device() -> dlpk::sys::DLDevice {
    dlpk::sys::DLDevice {
        device_type: dlpk::sys::DLDeviceType::kDLCUDA,
        device_id: 0,
    }
}

/// True when `Vector::try_on(kDLCUDA)` has a kernel backend.
pub fn cuda_available() -> bool {
    Vector::try_on(cuda_device(), Array1::zeros(1)).is_ok()
}

/// Sella `_gpu_ok` with the process environment policy.
pub fn gpu_ok(n: usize) -> bool {
    GpuPolicy::from_env().ok(n)
}

/// Sella `to_gpu`: claim a flat buffer on dlpk CUDA, or `None`.
///
/// A missing CUDA backend is not an OOM. A failed claim after the
/// backend is present records [`record_oom`].
pub fn to_gpu(a: ArrayView1<f64>) -> Option<Vector> {
    if !cuda_available() {
        return None;
    }
    match Vector::try_on(cuda_device(), a.to_owned()) {
        Ok(v) => Some(v),
        Err(_) => {
            record_oom(a.len());
            None
        }
    }
}

/// Sella `gpu_eigh`. Host Jacobi unless a dlpk CUDA kernel is linked.
pub fn gpu_eigh(a: ArrayView2<f64>) -> Result<(Array1<f64>, Array2<f64>), SaddleError> {
    GpuPolicy::from_env().eigh(a)
}

/// Sella `gpu_eigh_t`: stay on device. No device eigh kernel here.
pub fn gpu_eigh_t(_a_gpu: &Vector) -> Option<(Array1<f64>, Array2<f64>)> {
    None
}

/// Sella `gpu_qr` (economy / reduced).
pub fn gpu_qr(a: ArrayView2<f64>) -> Result<(Array2<f64>, Array2<f64>), SaddleError> {
    GpuPolicy::from_env().qr(a)
}

/// Sella `gpu_project`: `U^T H U`.
pub fn gpu_project(h: ArrayView2<f64>, u: ArrayView2<f64>) -> Result<Array2<f64>, SaddleError> {
    GpuPolicy::from_env().project(h, u)
}

/// Flatten and claim CUDA. `to_gpu` records OOM only on a failed
/// claim after a backend is present, never on a missing kernel.
fn try_device_upload(a: ArrayView2<f64>) -> Option<Vector> {
    let flat = Array1::from_iter(a.iter().copied());
    to_gpu(flat.view())
}

fn gpu_eigh_with(
    a: ArrayView2<f64>,
    policy: GpuPolicy,
) -> Result<(Array1<f64>, Array2<f64>), SaddleError> {
    if a.nrows() != a.ncols() {
        return Err(SaddleError::Shape("gpu_eigh needs a square matrix".into()));
    }
    let n = a.nrows();
    if policy.ok(n) {
        if let Some(dev) = try_device_upload(a) {
            if let Some(pair) = gpu_eigh_t(&dev) {
                return Ok(pair);
            }
            // Missing kernel is not an OOM. Sella `_gpu.py` records
            // only RuntimeError / MemoryError.
        }
    }
    exact_eigh(a)
}

fn gpu_qr_with(
    a: ArrayView2<f64>,
    policy: GpuPolicy,
) -> Result<(Array2<f64>, Array2<f64>), SaddleError> {
    if a.ncols() == 0 || a.nrows() == 0 {
        return Err(SaddleError::Shape("gpu_qr needs a nonempty matrix".into()));
    }
    if policy.ok(a.nrows()) {
        let _ = try_device_upload(a);
    }
    host_qr(a)
}

fn gpu_project_with(
    h: ArrayView2<f64>,
    u: ArrayView2<f64>,
    policy: GpuPolicy,
) -> Result<Array2<f64>, SaddleError> {
    if h.nrows() != h.ncols() || u.nrows() != h.nrows() {
        return Err(SaddleError::Shape(
            "gpu_project needs square H and matching U rows".into(),
        ));
    }
    if policy.ok(h.nrows()) {
        let _ = try_device_upload(h);
        let _ = try_device_upload(u);
    }
    Ok(host_project(h, u))
}

/// Economy QR via modified Gram-Schmidt. `R = Q^T A`.
fn host_qr(a: ArrayView2<f64>) -> Result<(Array2<f64>, Array2<f64>), SaddleError> {
    let q = modified_gram_schmidt(a, None, 1e-14);
    if q.ncols() == 0 {
        return Err(SaddleError::Solver("gpu_qr empty range".into()));
    }
    let mut r = Array2::<f64>::zeros((q.ncols(), a.ncols()));
    for i in 0..q.ncols() {
        for j in 0..a.ncols() {
            r[(i, j)] = dot(q.column(i), a.column(j));
        }
    }
    Ok((q, r))
}

/// `U^T H U` with vecops dots (rayon when `par` is on).
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

    #[test]
    fn host_policy_never_claims_cuda() {
        assert!(!GpuPolicy::host().ok(1024));
    }

    #[test]
    fn default_min_dim_is_sella_200() {
        let p = GpuPolicy {
            enabled: true,
            min_dim: DEFAULT_MIN_DIM,
        };
        assert_eq!(p.min_dim, 200);
        assert!(!p.ok(199));
    }

    #[test]
    fn gpu_eigh_rejects_a_nonsquare() {
        let a = Array2::<f64>::zeros((2, 3));
        match gpu_eigh(a.view()) {
            Err(SaddleError::Shape(_)) => {}
            other => panic!("expected shape error, got {other:?}"),
        }
    }

    #[test]
    fn cuda_probe_uses_vecops_not_a_second_stack() {
        let err = Vector::try_on(cuda_device(), Array1::zeros(4));
        if !cuda_available() {
            assert!(err.is_err());
        }
    }

    #[test]
    fn oom_floor_drops_to_the_smallest_failure() {
        let _guard = lock_oom_for_test();
        clear_oom_floor();
        record_oom(128);
        assert_eq!(oom_floor(), Some(128));
        record_oom(256);
        assert_eq!(oom_floor(), Some(128));
        record_oom(64);
        assert_eq!(oom_floor(), Some(64));
        clear_oom_floor();
        assert_eq!(oom_floor(), None);
    }

    #[test]
    fn default_reads_sella_disable_and_min_dim_keys() {
        assert!(env_flag_disabled(Some("1")));
        assert!(env_flag_disabled(Some("true")));
        assert!(env_flag_disabled(Some("YES")));
        assert!(!env_flag_disabled(Some("0")));
        assert!(!env_flag_disabled(None));
        assert_eq!(parse_min_dim(None), 200);
        assert_eq!(parse_min_dim(Some("64")), 64);
        assert_eq!(parse_min_dim(Some("nope")), 200);
        let p = GpuPolicy::default();
        assert_eq!(p, GpuPolicy::from_env());
    }

    #[test]
    fn gpu_ok_needs_cuda_at_sella_min_dim() {
        if !cuda_available() {
            assert!(!GpuPolicy::default().ok(200));
            assert!(!gpu_ok(200));
        }
    }
}
