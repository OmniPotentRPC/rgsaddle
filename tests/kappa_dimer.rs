use approx::assert_abs_diff_eq;
use ndarray::{Array1, Array2, array};
use rgsaddle::kappa::{KappaDimerConfig, kappa_dimer_force};
use rgsaddle::SaddleError;

#[test]
fn constrained_mode_excludes_force_without_a_spurious_zero_eigenvalue() {
    let h = array![[-4.,0.,0.], [0.,2.,0.], [0.,0.,6.]];
    let g = array![3.,0.,0.];
    let out = kappa_dimer_force(g.view(), array![1.,0.,0.].view(),
        Array2::zeros((0,3)).view(), |v| Ok(h.dot(&v)), &KappaDimerConfig::default()).unwrap();
    assert_eq!(out.tangent_dimension, 2);
    assert_abs_diff_eq!(out.tangent_curvature, 2., epsilon=1e-8);
    assert_abs_diff_eq!(out.kappa, -2./3., epsilon=1e-8);
    assert_abs_diff_eq!(out.tangent_mode.dot(&g), 0., epsilon=1e-12);
    assert!(out.residual < 1e-6);
    let z = 5. * out.kappa;
    assert_abs_diff_eq!(out.force[0], 3. * (2./(1.+z.exp())-1.), epsilon=1e-8);
}

#[test]
fn second_unstable_mode_produces_downhill_restraint() {
    let h = array![[-4.,0.,0.], [0.,-2.,0.], [0.,0.,6.]];
    let out = kappa_dimer_force(array![0.1,0.,0.].view(), array![1.,0.,0.].view(),
        Array2::zeros((0,3)).view(), |v| Ok(h.dot(&v)), &KappaDimerConfig::default()).unwrap();
    assert_abs_diff_eq!(out.kappa, 20., epsilon=1e-8);
    assert_vector(&out.force, &array![-0.1,0.,0.], 1e-12);
}

#[test]
fn redundant_rigid_modes_and_force_are_removed_from_the_eigensolve() {
    let h = Array2::from_diag(&array![0.,0.,-4.,2.,7.]);
    let excluded = array![[1.,0.,0.,0.,0.], [0.,1.,0.,0.,0.], [2.,0.,0.,0.,0.]];
    let g = array![0.,0.,3.,0.,0.];
    let out = kappa_dimer_force(g.view(), g.view(), excluded.view(),
        |v| Ok(h.dot(&v)), &KappaDimerConfig::default()).unwrap();
    assert_eq!(out.tangent_dimension, 2);
    assert_abs_diff_eq!(out.tangent_curvature, 2., epsilon=1e-8);
    assert_abs_diff_eq!(out.force[0], 0., epsilon=1e-12);
    assert_abs_diff_eq!(out.force[1], 0., epsilon=1e-12);
}

#[test]
fn curvature_and_force_obey_rotation_and_energy_scaling() {
    let h = array![[1.,2.,0.], [2.,-3.,1.], [0.,1.,5.]];
    let g = array![1.,2.,3.];
    let m = array![3.,-2.,1.];
    let q = array![[1./3.,2./3.,2./3.], [2./3.,1./3.,-2./3.], [-2./3.,2./3.,-1./3.]];
    let cfg = KappaDimerConfig::default();
    let bare = kappa_dimer_force(g.view(),m.view(),Array2::zeros((0,3)).view(),
        |v| Ok(h.dot(&v)), &cfg).unwrap();
    let hg = q.dot(&h).dot(&q.t())*7.;
    let scaled = kappa_dimer_force((q.dot(&g)*7.).view(),q.dot(&m).view(),
        Array2::zeros((0,3)).view(), |v| Ok(hg.dot(&v)), &cfg).unwrap();
    assert_abs_diff_eq!(bare.kappa, scaled.kappa, epsilon=1e-8);
    assert_vector(&(q.dot(&bare.force)*7.), &scaled.force, 1e-7);
}

#[test]
fn zero_gradient_does_not_divide_or_evaluate_a_hessian() {
    let out = kappa_dimer_force(Array1::zeros(3).view(), array![1.,0.,0.].view(),
        Array2::zeros((0,3)).view(), |_| panic!("stationary force needs no kappa"),
        &KappaDimerConfig::default()).unwrap();
    assert_vector(&out.force, &Array1::zeros(3), 0.);
    assert_eq!(out.actions, 0);
}

#[test]
fn one_dimension_uses_the_pure_uphill_limit() {
    let out = kappa_dimer_force(array![2.].view(), array![1.].view(),
        Array2::zeros((0,1)).view(), |_| panic!("no tangent directions"),
        &KappaDimerConfig::default()).unwrap();
    assert_eq!(out.tangent_dimension,0);
    assert_abs_diff_eq!(out.force[0],2.,epsilon=1e-14);
}

#[test]
fn failed_hessian_action_is_returned_without_a_force() {
    let err = kappa_dimer_force(array![1.,2.,3.].view(), array![1.,0.,0.].view(),
        Array2::zeros((0,3)).view(), |_| Err(SaddleError::Surface("rejected HVP".into())),
        &KappaDimerConfig::default()).unwrap_err();
    assert!(err.to_string().contains("rejected HVP"));
}

#[test]
fn invalid_beta_and_nonfinite_inputs_are_rejected() {
    let g = array![1.,2.,3.];
    for beta in [0.,-1.,f64::NAN,f64::INFINITY] {
        let cfg=KappaDimerConfig { beta,..KappaDimerConfig::default() };
        assert!(kappa_dimer_force(g.view(),g.view(),Array2::zeros((0,3)).view(),
            |v| Ok(v.to_owned()), &cfg).is_err());
    }
    assert!(kappa_dimer_force(array![f64::NAN,0.,0.].view(),g.view(),
        Array2::zeros((0,3)).view(), |v| Ok(v.to_owned()), &KappaDimerConfig::default()).is_err());
}

fn assert_vector(actual: &Array1<f64>, expected: &Array1<f64>, tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (a,b) in actual.iter().zip(expected) {
        assert_abs_diff_eq!(a,b,epsilon=tolerance);
    }
}
