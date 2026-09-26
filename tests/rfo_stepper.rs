//! Sella RationalFunctionOptimization: retracted point stays on the set.
//!
//! Algebra is rgmin `rfo_get_s`. Geometry is manopt proj / retr /
//! transp. This is a stepper, not a session.

use ndarray::{Array2, array};
use rgmin::manifold::{Manifold, Sphere};
use rgmin::vecops::{dot, nrm2};
use rgsaddle::{RationalFunctionOptimization, rfo_stepper};

#[test]
fn factory_matches_rfo_synonyms() {
    assert!(rfo_stepper("rfo", 0).is_some());
    assert!(rfo_stepper("rational function optimization", 1).is_some());
    assert!(rfo_stepper("qn", 0).is_none());
    assert!(rfo_stepper("prfo", 1).is_none());
}

#[test]
fn alpha_and_order_are_the_sella_contract() {
    let h = Array2::<f64>::eye(2);
    let g = array![2.0, 0.0];
    let stepper = RationalFunctionOptimization::new(0);
    assert_eq!(stepper.order, 0);
    assert!((RationalFunctionOptimization::ALPHA0 - 1.0).abs() < 1e-15);
    assert_eq!(RationalFunctionOptimization::SLOPE, 1.0);
    let s0 = stepper.get_s(&h, &g, 0.0);
    assert!(nrm2(s0.view()) < 1e-12);
    let n_lo = nrm2(stepper.get_s(&h, &g, 0.2).view());
    let n_hi = nrm2(stepper.get_s(&h, &g, 1.0).view());
    assert!(n_hi > n_lo, "Sella RFO slope is +1: {n_lo} !< {n_hi}");
    let s1 = RationalFunctionOptimization::new(1).get_s(&h, &g, 1.0);
    assert!(nrm2((&stepper.get_s(&h, &g, 1.0) - &s1).view()) > 1e-8);
}

#[test]
fn rfo_retract_stays_on_the_sphere() {
    let x = array![0.0, 1.0, 0.0];
    assert!((nrm2(x.view()) - 1.0).abs() < 1e-15);
    let h = Array2::<f64>::eye(3);
    let egrad = array![0.4, -0.2, 0.3];
    let y = RationalFunctionOptimization::new(0).step_on(&Sphere, &x, &h, &egrad, 1.0);
    assert!(
        (nrm2(y.view()) - 1.0).abs() < 1e-12,
        "retracted point left the sphere: ||y||={}",
        nrm2(y.view())
    );
}

#[test]
fn rfo_step_is_riemannian_proj_retr_transp() {
    let x = array![
        1.0 / 3.0_f64.sqrt(),
        1.0 / 3.0_f64.sqrt(),
        1.0 / 3.0_f64.sqrt()
    ];
    let h = Array2::<f64>::eye(3);
    let egrad = array![1.2, -0.4, 0.1];
    let stepper = RationalFunctionOptimization::new(0);
    let g = Sphere.egrad2rgrad(&x, &egrad);
    assert!(
        dot(x.view(), g.view()).abs() < 1e-12,
        "Riemannian gradient must be tangent"
    );
    let s_amb = stepper.get_s(&h, &g, 1.0);
    let s = Sphere.project(&x, &s_amb);
    assert!(
        dot(x.view(), s.view()).abs() < 1e-12,
        "projected RFO step must be tangent"
    );
    let y = Sphere.retract(&x, &s);
    assert!((nrm2(y.view()) - 1.0).abs() < 1e-12);
    let t = Sphere.transport(&x, &y, &s);
    assert!(
        dot(y.view(), t.view()).abs() < 1e-12,
        "transported step must be tangent at arrival"
    );
    let w = stepper.transport_step(&Sphere, &x, &y, &h, &egrad, 1.0);
    assert!(dot(y.view(), w.view()).abs() < 1e-12);
}
