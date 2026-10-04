//! Physical radius checks for the Sella partitioned step.

use ndarray::{Array1, Array2, array};
use rgmin::vecops::nrm2;
use rgsaddle::{PartitionedRationalFunctionOptimization, prfo_trust_region};

fn check_binding_radius(evals: &Array1<f64>, g: &Array1<f64>, radius: f64) {
    let evecs = Array2::<f64>::eye(g.len());
    let step = prfo_trust_region(evals, &evecs, g, 1, radius);
    let length = nrm2(step.view());
    let roundoff = 8.0 * f64::EPSILON;
    assert!(
        length <= radius * (1.0 + roundoff),
        "step norm {length:e} exceeds radius {radius:e}"
    );
    assert!(
        length >= radius * (1.0 - 1e-10),
        "binding step norm {length:e} does not reach radius {radius:e}"
    );
    assert!(step[0] > 0.0);
    assert!(step[1] < 0.0);
}

#[test]
fn tiny_radii_bound_the_actual_partitioned_displacement() {
    let evals = array![-1.0, 4.0];
    let g = array![0.5, 0.4];
    for radius in [1e-12, 1e-20] {
        check_binding_radius(&evals, &g, radius);
    }
}

#[test]
fn unrestricted_step_requires_the_physical_radius() {
    let evals = array![-1.0, 4.0];
    let g = array![1e-6, 4e-7];
    let evecs = Array2::<f64>::eye(2);
    let full = PartitionedRationalFunctionOptimization::new(1).get_s(&evals, &evecs, &g, 1.0);
    let length = nrm2(full.view());
    assert!(length > 1e-8);
    check_binding_radius(&evals, &g, length * (1.0 - 1e-9));
}

#[test]
fn euclidean_clip_respects_radii_below_its_step_scale() {
    let step = array![1e-17, -2e-17];
    let radius = 1e-18;
    let trust = rgsaddle::restricted::TrustRegion::new(radius).unwrap();
    let clipped = trust.clip(&step);
    let length = nrm2(clipped.view());
    assert!(length <= radius * (1.0 + 8.0 * f64::EPSILON));
    assert!(length >= radius * (1.0 - 8.0 * f64::EPSILON));
    let zero = rgsaddle::restricted::TrustRegion::new(0.0).unwrap();
    assert_eq!(zero.clip(&step), Array1::<f64>::zeros(2));
}

#[test]
fn euclidean_radius_rejects_nonfinite_values() {
    for radius in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(rgsaddle::restricted::TrustRegion::new(radius).is_err());
    }
}
