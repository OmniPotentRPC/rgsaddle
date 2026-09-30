//! Restricted-step partitioned RFO against the augmented-Hessian
//! eigenvalue problem, and the saddle searches that use it.

use ndarray::array;
use rgsaddle::{PrfoKind, partitioned_rfo_eigen, restricted_prfo_displacement};

fn close(got: &[f64], expect: &[f64], tol: f64) {
    assert_eq!(got.len(), expect.len());
    for (i, (a, b)) in got.iter().zip(expect).enumerate() {
        assert!(
            (a - b).abs() <= tol,
            "component {i}: {a} differs from {b} by {}",
            (a - b).abs()
        );
    }
}

#[test]
fn partition_matches_the_augmented_hessian_roots() {
    // Diagonal Hessian, so the eigenbasis is the coordinate basis.
    // The numbers are the larger root of the 2 by 2 on the lowest
    // mode and the lowest root of the complement.
    let h = array![[-2.0, 0.0], [0.0, 2.0]];
    let g = array![-0.8, -0.6];
    let dx =
        restricted_prfo_displacement(h.view(), g.view(), None, 10.0, PrfoKind::Index1).unwrap();
    close(
        dx.as_slice().unwrap(),
        &[-0.3507810593582122, 0.27698396494843347],
        1e-12,
    );

    let h3 = array![[-3.0, 0.0, 0.0], [0.0, -0.4, 0.0], [0.0, 0.0, 2.0]];
    let g3 = array![0.3, -0.1, 0.2];
    let dx =
        restricted_prfo_displacement(h3.view(), g3.view(), None, 10.0, PrfoKind::Index1).unwrap();
    close(
        dx.as_slice().unwrap(),
        &[0.09901951359278482, 4.080109606702181, -0.08249092414080715],
        1e-12,
    );
}

#[test]
fn trust_sphere_changes_the_mode_ratio() {
    let evals = array![-2.0, 2.0];
    let g = array![-0.8, -0.6];
    let pure = partitioned_rfo_eigen(evals.view(), g.view(), 1.0, PrfoKind::Index1).unwrap();
    let h = array![[-2.0, 0.0], [0.0, 2.0]];
    let restricted =
        restricted_prfo_displacement(h.view(), g.view(), None, 0.2, PrfoKind::Index1).unwrap();
    let norm = restricted.dot(&restricted).sqrt();
    assert!((norm - 0.2).abs() < 1e-8, "norm {norm}");
    let pure_ratio = pure[0] / pure[1];
    let restricted_ratio = restricted[0] / restricted[1];
    assert!(
        (pure_ratio - restricted_ratio).abs() > 1e-2,
        "pure {pure_ratio}, restricted {restricted_ratio}"
    );
    close(
        restricted.as_slice().unwrap(),
        &[-0.14655664955539524, 0.13609242620769563],
        1e-8,
    );
}
