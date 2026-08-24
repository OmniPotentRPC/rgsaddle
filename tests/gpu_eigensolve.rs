//! Sella `_gpu.py`: eigh / QR / project on the dlpk waist.
//!
//! Eigenvectors live on the sphere. After `gpu_eigh`, proj / retr /
//! transp must keep the point on the set. The CUDA tag is refused
//! when this build has no kernel backend (not a second GPU stack).

use ndarray::{Array1, Array2, array};
use rgmin::manifold::{Manifold, Sphere};
use rgmin::vecops::{dot, nrm2};
use rgsaddle::{
    EigenDevice, GPU_MIN_DIM, GpuPolicy, eigh_on, gpu_eigh, gpu_ok, gpu_project, gpu_qr, to_gpu,
};

fn diag3() -> Array2<f64> {
    let mut a = Array2::<f64>::zeros((3, 3));
    a[(0, 0)] = 4.0;
    a[(1, 1)] = 1.0;
    a[(2, 2)] = 9.0;
    a[(0, 1)] = 0.5;
    a[(1, 0)] = 0.5;
    a
}

#[test]
fn gpu_eigh_eigenvector_retract_stays_on_the_sphere() {
    let a = diag3();
    let mut policy = GpuPolicy {
        enabled: true,
        min_dim: 1,
        oom_floor: None,
    };
    let (lams, vecs) = gpu_eigh(a.view(), &mut policy).unwrap();
    assert_eq!(lams.len(), 3);
    let x = vecs.column(0).to_owned();
    assert!(
        (nrm2(x.view()) - 1.0).abs() < 1e-12,
        "eigenvector left the sphere: ||x||={}",
        nrm2(x.view())
    );
    let v_amb = array![0.2, -0.1, 0.3];
    let s = Sphere.project(&x, &v_amb);
    assert!(
        dot(x.view(), s.view()).abs() < 1e-12,
        "projected step must be tangent"
    );
    let y = Sphere.retract(&x, &s);
    assert!(
        (nrm2(y.view()) - 1.0).abs() < 1e-12,
        "retracted point left the sphere: ||y||={}",
        nrm2(y.view())
    );
    let t = Sphere.transport(&x, &y, &s);
    assert!(
        dot(y.view(), t.view()).abs() < 1e-12,
        "transported step must be tangent at arrival"
    );
}

#[test]
fn gpu_qr_q_is_on_stiefel() {
    let mut a = Array2::<f64>::zeros((3, 2));
    a[(0, 0)] = 1.0;
    a[(1, 0)] = 1.0;
    a[(2, 0)] = 0.0;
    a[(0, 1)] = 0.0;
    a[(1, 1)] = 1.0;
    a[(2, 1)] = 1.0;
    let mut policy = GpuPolicy {
        enabled: true,
        min_dim: 1,
        oom_floor: None,
    };
    let (q, r) = gpu_qr(a.view(), &mut policy).unwrap();
    assert_eq!(q.nrows(), 3);
    assert_eq!(q.ncols(), 2);
    assert_eq!(r.nrows(), 2);
    assert_eq!(r.ncols(), 2);
    for i in 0..2 {
        for j in 0..2 {
            let g = dot(q.column(i), q.column(j));
            let want = if i == j { 1.0 } else { 0.0 };
            assert!((g - want).abs() < 1e-12, "Q^T Q[{i},{j}]={g}");
        }
    }
}

#[test]
fn gpu_project_is_ut_h_u() {
    let h = diag3();
    let u = Array2::<f64>::eye(3).slice(ndarray::s![.., ..2]).to_owned();
    let mut policy = GpuPolicy {
        enabled: true,
        min_dim: 1,
        oom_floor: None,
    };
    let p = gpu_project(h.view(), u.view(), &mut policy).unwrap();
    assert_eq!(p.nrows(), 2);
    assert_eq!(p.ncols(), 2);
    assert!((p[(0, 0)] - 4.0).abs() < 1e-12);
    assert!((p[(1, 1)] - 1.0).abs() < 1e-12);
    assert!((p[(0, 1)] - 0.5).abs() < 1e-12);
    assert!((p[(1, 0)] - 0.5).abs() < 1e-12);
}

#[test]
fn dlpk_cuda_upload_is_refused_not_staged() {
    let a = Array1::zeros(4);
    assert!(
        to_gpu(a).is_none(),
        "CUDA Vector must be refused without a kernel backend"
    );
}

#[test]
fn size_gate_matches_sella_default() {
    let p = GpuPolicy::default();
    assert_eq!(p.min_dim, GPU_MIN_DIM);
    assert_eq!(GPU_MIN_DIM, 200);
    assert!(!gpu_ok(&p, 50));
    assert!(gpu_ok(&p, 200));
    let mut floored = p.clone();
    floored.record_oom(256);
    assert!(!gpu_ok(&floored, 300));
    assert!(gpu_ok(&floored, 200));
}

#[test]
fn eigh_on_dlpk_shares_the_host_spectrum() {
    let a = diag3();
    let (h, _) = eigh_on(EigenDevice::Host, a.view()).unwrap();
    let (d, _) = eigh_on(EigenDevice::Dlpk, a.view()).unwrap();
    for i in 0..3 {
        assert!((h[i] - d[i]).abs() < 1e-10, "{} vs {}", h[i], d[i]);
    }
}
