//! Cartesian restrictions preserve their direction at extreme finite scales.

use ndarray::{Array1, array};
use rgsaddle::restricted::TrustRegion;

fn stable_length(v: &Array1<f64>) -> f64 {
    v.iter().fold(0.0_f64, |length, value| length.hypot(*value))
}

#[test]
fn tiny_cartesian_step_is_clipped_to_its_physical_radius() {
    let radius = 1e-201;
    let clipped = TrustRegion::new(radius).unwrap().clip(&array![3e-200, 4e-200]);
    assert!((stable_length(&clipped)/radius - 1.0).abs() < 8.0 * f64::EPSILON);
    assert!((clipped[0]/radius - 0.6).abs() < 8.0 * f64::EPSILON);
    assert!((clipped[1]/radius - 0.8).abs() < 8.0 * f64::EPSILON);
}

#[test]
fn large_cartesian_step_retains_its_direction_when_clipped() {
    let radius = 1.0;
    let clipped = TrustRegion::new(radius).unwrap().clip(&array![3e200, 4e200]);
    assert!((stable_length(&clipped)/radius - 1.0).abs() < 8.0 * f64::EPSILON);
    assert!((clipped[0] - 0.6).abs() < 8.0 * f64::EPSILON);
    assert!((clipped[1] - 0.8).abs() < 8.0 * f64::EPSILON);
}

#[test]
fn tiny_cartesian_steps_keep_the_interior_and_zero_radius_contracts() {
    let step = array![3e-200, 4e-200];
    assert_eq!(TrustRegion::new(6e-200).unwrap().clip(&step), step);
    assert_eq!(
        TrustRegion::new(0.0).unwrap().clip(&step),
        Array1::<f64>::zeros(2)
    );
}


#[test]
fn large_cartesian_step_reaches_a_tiny_representable_radius() {
    let radius = 1e-201;
    let clipped = TrustRegion::new(radius).unwrap().clip(&array![3e200, 4e200]);
    assert!((stable_length(&clipped)/radius - 1.0).abs() < 8.0 * f64::EPSILON);
    assert!((clipped[0]/radius - 0.6).abs() < 8.0 * f64::EPSILON);
    assert!((clipped[1]/radius - 0.8).abs() < 8.0 * f64::EPSILON);
}

#[test]
fn large_step_preserves_a_representable_small_component() {
    let radius = 1e100;
    let step = array![3e200, 4e200, 1e-200];
    let clipped = TrustRegion::new(radius).unwrap().clip(&step);
    assert!((stable_length(&clipped)/radius - 1.0).abs() < 8.0 * f64::EPSILON);
    assert!((clipped[2]/2e-301 - 1.0).abs() < 8.0 * f64::EPSILON);
}
