//! Norm derivatives use the step scale without a denominator floor.

use ndarray::{Array1, array};
use rgsaddle::restricted::TrustRegion;

#[test]
fn norm_derivative_matches_the_step_direction_at_small_scales() {
    let trust = TrustRegion::new(1.0).unwrap();
    let derivative = array![3.0, 4.0];
    for scale in [1.0, 1e-20, 1e-200] {
        let step = &derivative * scale;
        assert!(
            (trust.cons_dalpha(&step, &derivative) - 5.0).abs() < 1e-14,
            "incorrect norm slope at step scale {scale:e}"
        );
    }
}

#[test]
fn norm_derivative_matches_a_transverse_finite_difference() {
    let trust = TrustRegion::new(1.0).unwrap();
    let step = array![3.0, 4.0];
    let derivative = array![2.0, -1.0];
    let h = 1e-4;
    let before = &step - &derivative * h;
    let after = &step + &derivative * h;
    let measured = (trust.cons(&after) - trust.cons(&before)) / (2.0 * h);
    let analytic = trust.cons_dalpha(&step, &derivative);
    assert!((analytic - 0.4).abs() < 1e-14);
    assert!((analytic - measured).abs() < 1e-9);
}

#[test]
fn zero_step_has_the_right_directional_norm_derivative() {
    let trust = TrustRegion::new(1.0).unwrap();
    let zero = Array1::<f64>::zeros(2);
    let derivative = array![3.0, 4.0];
    assert_eq!(trust.cons_dalpha(&zero, &derivative), 5.0);
    assert_eq!(trust.cons_dalpha(&zero, &zero), 0.0);
}

#[test]
fn zero_step_right_derivative_preserves_tiny_nonzero_values() {
    let trust = TrustRegion::new(1.0).unwrap();
    let zero = Array1::<f64>::zeros(2);
    let derivative = array![3e-200, 4e-200];
    let slope = trust.cons_dalpha(&zero, &derivative);
    assert!((slope / 5e-200 - 1.0).abs() < 1e-14);
}

#[test]
fn finite_derivative_does_not_overflow_its_dot_product() {
    let trust = TrustRegion::new(1.0).unwrap();
    let expected = 2.0_f64.sqrt() * 1e308;
    let slope = trust.cons_dalpha(&array![1.0, 1.0], &array![1e308, 1e308]);
    assert!(slope.is_finite());
    assert!((slope / expected - 1.0).abs() < 1e-14);

    let step = Array1::from(vec![1.0; 8]);
    let derivative = array![1e308, 1e308, 1e308, 1e308, 1e308, 1e308, -1e308, -1e308];
    let cancelled = trust.cons_dalpha(&step, &derivative);
    assert!(cancelled.is_finite());
    assert!((cancelled / expected - 1.0).abs() < 1e-14);
}
