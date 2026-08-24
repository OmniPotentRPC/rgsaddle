//! Sella `_gpu.py`: dlpk is the waist; the eigenbasis stays on Stiefel.
//!
//! Algebra is rgmin `vecops`. Geometry is manopt proj / retr / transp.
//! This is the optional GPU factory, not a second device stack.

use ndarray::{Array1, Array2, array};
use rgmin::manifold::{Manifold, Sphere};
use rgmin::vecops::{Vector, dot, nrm2};
use rgsaddle::{
    EigenDevice, GpuPolicy, clear_oom_floor, cuda_available, eigh_on, gpu_eigh, gpu_eigh_t, gpu_ok,
    gpu_project, gpu_qr, lock_oom_for_test, record_oom, retract_qn, to_gpu,
};

fn hilbert(n: usize) -> Array2<f64> {
    let mut a = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            a[(i, j)] = 1.0 / ((i + j + 1) as f64);
        }
    }
    a
}

fn columns_on_stiefel(q: &Array2<f64>, tol: f64) -> bool {
    let k = q.ncols();
    for j in 0..k {
        if (nrm2(q.column(j)) - 1.0).abs() > tol {
            return false;
        }
        for i in 0..j {
            if dot(q.column(i), q.column(j)).abs() > tol {
                return false;
            }
        }
    }
    true
}

#[test]
fn gpu_eigh_eigenbasis_stays_on_stiefel() {
    let a = hilbert(4);
    let (lams, vecs) = gpu_eigh(a.view()).unwrap();
    assert_eq!(lams.len(), 4);
    assert!(
        columns_on_stiefel(&vecs, 1e-10),
        "gpu_eigh left the Stiefel set"
    );
    // A v = lam v
    for j in 0..4 {
        let mut av = Array1::zeros(4);
        for i in 0..4 {
            av[i] = dot(a.row(i), vecs.column(j));
        }
        for i in 0..4 {
            assert!((av[i] - lams[j] * vecs[(i, j)]).abs() < 1e-8);
        }
    }
}

#[test]
fn gpu_qr_q_stays_on_stiefel() {
    let mut a = Array2::<f64>::zeros((3, 2));
    a[(0, 0)] = 1.0;
    a[(1, 0)] = 2.0;
    a[(2, 0)] = 3.0;
    a[(0, 1)] = 0.0;
    a[(1, 1)] = 1.0;
    a[(2, 1)] = 1.0;
    let (q, r) = gpu_qr(a.view()).unwrap();
    assert_eq!(q.nrows(), 3);
    assert_eq!(q.ncols(), 2);
    assert!(columns_on_stiefel(&q, 1e-12), "gpu_qr Q left Stiefel");
    // A ≈ Q R
    for j in 0..2 {
        for i in 0..3 {
            let mut acc = 0.0;
            for k in 0..2 {
                acc += q[(i, k)] * r[(k, j)];
            }
            assert!((acc - a[(i, j)]).abs() < 1e-10, "QR reconstruct {i},{j}");
        }
    }
}

#[test]
fn gpu_project_matches_host_ut_h_u() {
    let mut h = Array2::<f64>::zeros((3, 3));
    h[(0, 0)] = 4.0;
    h[(1, 1)] = 1.0;
    h[(2, 2)] = 9.0;
    h[(0, 1)] = 0.5;
    h[(1, 0)] = 0.5;
    let u = Array2::eye(3).slice(ndarray::s![.., ..2]).to_owned();
    let r = gpu_project(h.view(), u.view()).unwrap();
    assert_eq!(r.nrows(), 2);
    assert_eq!(r.ncols(), 2);
    assert!((r[(0, 0)] - 4.0).abs() < 1e-12);
    assert!((r[(1, 1)] - 1.0).abs() < 1e-12);
    assert!((r[(0, 1)] - 0.5).abs() < 1e-12);
}

#[test]
fn gpu_eigh_then_retract_stays_on_the_sphere() {
    let a = hilbert(3);
    let (_lams, vecs) = gpu_eigh(a.view()).unwrap();
    let x = vecs.column(0).to_owned();
    assert!((nrm2(x.view()) - 1.0).abs() < 1e-12);
    let evals = Array1::ones(3);
    let evecs = Array2::<f64>::eye(3);
    let egrad = array![0.4, -0.2, 0.3];
    let y = retract_qn(&Sphere, &x, &evals, &evecs, &egrad, 0, 0.0);
    assert!(
        (nrm2(y.view()) - 1.0).abs() < 1e-12,
        "retracted gpu_eigh mode left the sphere: ||y||={}",
        nrm2(y.view())
    );
    let g = Sphere.egrad2rgrad(&x, &egrad);
    assert!(dot(x.view(), g.view()).abs() < 1e-12);
    let s = Sphere.project(&x, &g);
    let t = Sphere.transport(&x, &y, &s);
    assert!(
        dot(y.view(), t.view()).abs() < 1e-12,
        "transported step must be tangent at arrival"
    );
}

#[test]
fn gpu_ok_respects_min_dim_disable_and_oom_floor() {
    let _guard = lock_oom_for_test();
    clear_oom_floor();
    let host = GpuPolicy::host();
    assert!(!host.ok(512), "host policy must never claim a device");
    let gated = GpuPolicy {
        enabled: true,
        min_dim: 200,
    };
    assert!(!gated.ok(16), "below SELLA_GPU_MIN_DIM");
    record_oom(64);
    assert_eq!(rgsaddle::oom_floor(), Some(64));
    let wide = GpuPolicy {
        enabled: true,
        min_dim: 1,
    };
    assert!(!wide.ok(64), "OOM floor must skip this size");
    assert!(!wide.ok(128), "OOM floor skips larger shapes too");
    clear_oom_floor();
    // sella/_gpu.py:38 — `_gpu_ok` is false without a CUDA backend.
    if !cuda_available() {
        assert!(!wide.ok(8));
        assert!(!gpu_ok(512));
        assert!(
            !GpuPolicy::default().ok(200),
            "gpu_ok(200) must be false without CUDA"
        );
    }
}

#[test]
fn to_gpu_refuses_cuda_without_a_backend() {
    let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);
    if !cuda_available() {
        assert!(to_gpu(x.view()).is_none());
    }
    // The only storage handle is rgmin's Vector. A CPU wrap is the host arm.
    let v = Vector::from_host(x.clone());
    assert_eq!(v.host_view(), x.view());
    assert!(gpu_eigh_t(&v).is_none());
}

#[test]
fn dlpk_eigh_on_shares_the_gpu_factory() {
    let a = hilbert(3);
    let (h, hv) = eigh_on(EigenDevice::Host, a.view()).unwrap();
    let (d, dv) = eigh_on(EigenDevice::Dlpk, a.view()).unwrap();
    for i in 0..3 {
        assert!((h[i] - d[i]).abs() < 1e-10, "{} vs {}", h[i], d[i]);
    }
    assert!(columns_on_stiefel(&hv, 1e-10));
    assert!(columns_on_stiefel(&dv, 1e-10));
}

#[test]
fn eigh_on_dlpk_keeps_the_process_oom_floor() {
    let _guard = lock_oom_for_test();
    clear_oom_floor();
    record_oom(64);
    let a = hilbert(3);
    let _ = eigh_on(EigenDevice::Dlpk, a.view()).unwrap();
    assert_eq!(
        rgsaddle::oom_floor(),
        Some(64),
        "eigh_on Dlpk must keep the process-global OOM floor"
    );
    let wide = GpuPolicy {
        enabled: true,
        min_dim: 1,
    };
    assert!(!wide.ok(64));
    clear_oom_floor();
}

#[test]
fn missing_dlpk_kernel_is_not_an_oom() {
    let _guard = lock_oom_for_test();
    clear_oom_floor();
    let a = hilbert(4);
    let _ = gpu_eigh(a.view()).unwrap();
    let _ = gpu_qr(a.view()).unwrap();
    let u = Array2::eye(4);
    let _ = gpu_project(a.view(), u.view()).unwrap();
    assert_eq!(
        rgsaddle::oom_floor(),
        None,
        "gpu_eigh_t None must not record_oom"
    );
    if let Some(v) = to_gpu(Array1::zeros(4).view()) {
        assert!(gpu_eigh_t(&v).is_none());
        assert_eq!(rgsaddle::oom_floor(), None);
    }
    clear_oom_floor();
}
