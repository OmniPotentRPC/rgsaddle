//! Sella TrustRegion: accepted QN step sits inside `delta`, and on
//! the bound when the unconstrained Newton step is longer.

use ndarray::{Array2, array};
use rgmin::manifold::{Euclidean, Manifold};
use rgsaddle::TrustRegion;

#[test]
fn accepted_qn_step_is_inside_delta() {
    let h = Array2::<f64>::eye(2) * 2.0;
    let g = array![2.0, 0.0];
    let acc = TrustRegion::new(10.0).restrict_qn(&h, &g).unwrap();
    assert!(acc.nrm2() <= 10.0);
    assert!((acc.step[0] + 1.0).abs() < 1e-10);
    assert!(acc.step[1].abs() < 1e-10);
}

#[test]
fn accepted_qn_step_sits_on_the_bound_when_newton_is_longer() {
    let h = Array2::<f64>::eye(2) * 2.0;
    let g = array![2.0, 0.0];
    let delta = 0.1;
    let acc = TrustRegion::new(delta).restrict_qn(&h, &g).unwrap();
    assert!(acc.nrm2() <= delta + 1e-10);
    assert!((acc.nrm2() - delta).abs() < 1e-9);
}

#[test]
fn retract_stays_on_the_trust_ball() {
    let h = Array2::<f64>::eye(3) * 2.0;
    let g = array![2.0, -1.0, 0.5];
    let delta = 0.2;
    let x = array![1.0, -2.0, 0.5];
    let acc = TrustRegion::new(delta).restrict_qn(&h, &g).unwrap();
    let y = Euclidean.retract(&x, &acc.step);
    let v = Euclidean.project(&x, &(&y - &x));
    let n = v.dot(&v).sqrt();
    assert!(n <= delta + 1e-10);
    let t = Euclidean.transport(&x, &y, &acc.step);
    assert!((t.dot(&t).sqrt() - acc.nrm2()).abs() < 1e-12);
}
