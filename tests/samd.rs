//! Sella SAMD: retracted point stays on the set.
//!
//! Algebra is Sella `samd.py` (velocity-Verlet + BDP). Geometry is
//! manopt proj / retr / transp. This is a session, not a stepper.

use ndarray::{Array1, ArrayView1, array};
use rgmin::ManifoldKind;
use rgmin::manifold::{Manifold, Sphere};
use rgmin::vecops::{dot, nrm2};
use rgsaddle::{
    PointSurface, SaddleError, SamdConfig, SamdSession, project_velocity, retract_samd,
    transport_velocity,
};

struct Well;

impl PointSurface for Well {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let mut g = Array1::zeros(x.len());
        g[0] = 2.0 * x[0];
        Ok((x[0] * x[0], g))
    }
}

#[test]
fn samd_retract_stays_on_the_sphere() {
    let x = array![0.0, 1.0, 0.0];
    assert!((nrm2(x.view()) - 1.0).abs() < 1e-15);
    let v = array![0.3, 0.0, -0.1];
    let g = array![0.4, -0.2, 0.3];
    let y = retract_samd(&Sphere, &x, &v, &g, 0.1);
    assert!(
        (nrm2(y.view()) - 1.0).abs() < 1e-12,
        "retracted point left the sphere: ||y||={}",
        nrm2(y.view())
    );
}

#[test]
fn samd_step_is_riemannian_proj_retr_transp() {
    let x = array![
        1.0 / 3.0_f64.sqrt(),
        1.0 / 3.0_f64.sqrt(),
        1.0 / 3.0_f64.sqrt()
    ];
    let v = array![0.2, -0.1, 0.05];
    let g = array![1.2, -0.4, 0.1];
    let dt = 0.1;
    let mut dx = Array1::zeros(3);
    rgmin::vecops::axpy(dt, v.view(), &mut dx);
    rgmin::vecops::axpy(-0.5 * dt * dt, g.view(), &mut dx);
    let s = Sphere.project(&x, &dx);
    assert!(
        dot(x.view(), s.view()).abs() < 1e-12,
        "projected Verlet increment must be tangent"
    );
    let y = Sphere.retract(&x, &s);
    assert!((nrm2(y.view()) - 1.0).abs() < 1e-12);
    let t = Sphere.transport(&x, &y, &s);
    assert!(
        dot(y.view(), t.view()).abs() < 1e-12,
        "transported increment must be tangent at arrival"
    );
    let vp = project_velocity(&Sphere, &x, &v);
    assert!(dot(x.view(), vp.view()).abs() < 1e-12);
    let vt = transport_velocity(&Sphere, &x, &y, &vp);
    assert!(dot(y.view(), vt.view()).abs() < 1e-12);
}

#[test]
fn samd_step_on_stays_on_the_sphere() {
    let x = array![0.0, 1.0, 0.0];
    let v0 = array![0.2, 0.0, 0.1];
    let mut sess = SamdSession::new(SamdConfig::default(), x, v0, &Well).unwrap();
    let r = array![0.1, -0.2, 0.05];
    let report = sess.step_on(&Sphere, &Well, r.view()).unwrap();
    assert!(report.energy.is_finite());
    assert!(report.kinetic.is_finite());
    let n = nrm2(sess.position());
    assert!(
        (n - 1.0).abs() < 1e-12,
        "SAMD point left the sphere: ||x||={n}"
    );
    assert!(
        dot(sess.position(), sess.velocity()).abs() < 1e-12,
        "SAMD velocity left the tangent space"
    );
}

#[test]
fn samd_step_on_preserves_rigid_quotient_com() {
    let x = Array1::from(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    let v0 = Array1::from_elem(9, 0.05);
    let mut sess = SamdSession::new(SamdConfig::default(), x, v0, &Well).unwrap();
    let r = Array1::from_elem(9, 0.1);
    sess.step_on(&ManifoldKind::RigidQuotient, &Well, r.view())
        .unwrap();
    let y = sess.position().to_owned();
    let v = sess.velocity().to_owned();
    let mut com = [0.0; 3];
    for i in 0..3 {
        com[0] += y[3 * i];
        com[1] += y[3 * i + 1];
        com[2] += y[3 * i + 2];
    }
    assert!((com[0] / 3.0 - 1.0 / 3.0).abs() < 1e-12);
    assert!((com[1] / 3.0 - 1.0 / 3.0).abs() < 1e-12);
    assert!((com[2] / 3.0).abs() < 1e-12);
    let vh = ManifoldKind::RigidQuotient.project(&y, &v);
    assert!(nrm2((&v - &vh).view()) < 1e-12);
}
