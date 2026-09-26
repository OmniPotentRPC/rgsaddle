//! Sella QuasiNewton: the retracted point stays on the set.
//!
//! Algebra is rgmin `qn_get_s`. Geometry is manopt proj / retr /
//! transp. This is a stepper, not a session.

use ndarray::{Array1, Array2, array};
use rgmin::manifold::{Manifold, Sphere};
use rgmin::qn_get_s;
use rgmin::vecops::{dot, nrm2};
use rgsaddle::{QuasiNewton, get_stepper, retract_qn};

#[test]
fn get_stepper_matches_qn_synonyms() {
    assert!(get_stepper("qn").is_some());
    assert!(get_stepper("quasi-newton").is_some());
    assert!(get_stepper("newton").is_some());
    assert!(get_stepper("mmf").is_some());
    assert!(get_stepper("dimer").is_some());
    assert!(get_stepper("rfo").is_none());
    assert!(get_stepper("prfo").is_none());
}

#[test]
fn alpha_zero_is_signed_newton() {
    let evals = array![4.0, -2.0];
    let evecs = Array2::<f64>::eye(2);
    let g = array![8.0, 2.0];
    let qn = QuasiNewton::new(&evals, &evecs, &g, 1);
    let (s, _) = qn.get_s(0.0);
    // Sella flips the first `order` slots: L = (-|4|, |-2|) = (-4, 2);
    // s = -g/L = (2, -1).
    assert!((s[0] - 2.0).abs() < 1e-12);
    assert!((s[1] + 1.0).abs() < 1e-12);
}

#[test]
fn qn_retract_stays_on_the_sphere() {
    let x = array![0.0, 1.0, 0.0];
    assert!((nrm2(x.view()) - 1.0).abs() < 1e-15);
    let evals = Array1::ones(3);
    let evecs = Array2::<f64>::eye(3);
    let egrad = array![0.4, -0.2, 0.3];
    let y = retract_qn(&Sphere, &x, &evals, &evecs, &egrad, 0, 0.0);
    assert!(
        (nrm2(y.view()) - 1.0).abs() < 1e-12,
        "retracted point left the sphere: ||y||={}",
        nrm2(y.view())
    );
}

#[test]
fn qn_step_is_riemannian_proj_retr_transp() {
    let x = array![1.0 / 3.0_f64.sqrt(), 1.0 / 3.0_f64.sqrt(), 1.0 / 3.0_f64.sqrt()];
    let evals = Array1::from(vec![2.0, 1.0, 0.5]);
    let evecs = Array2::<f64>::eye(3);
    let egrad = array![1.2, -0.4, 0.1];
    let g = Sphere.egrad2rgrad(&x, &egrad);
    assert!(
        dot(x.view(), g.view()).abs() < 1e-12,
        "Riemannian gradient must be tangent"
    );
    let (s_amb, _) = qn_get_s(&evals, &evecs, &g, 0, 0.0);
    let s = Sphere.project(&x, &s_amb);
    assert!(
        dot(x.view(), s.view()).abs() < 1e-12,
        "projected QN step must be tangent"
    );
    let y = Sphere.retract(&x, &s);
    assert!((nrm2(y.view()) - 1.0).abs() < 1e-12);
    let t = Sphere.transport(&x, &y, &s);
    assert!(
        dot(y.view(), t.view()).abs() < 1e-12,
        "transported step must be tangent at arrival"
    );
}
