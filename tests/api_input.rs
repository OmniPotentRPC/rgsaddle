//! Inputs the sessions used to accept and then misread.

use ndarray::{ArrayView2, array};
use rgsaddle::{
    BandConfig, BandSession, BandSurface, Cell, MinModeConfig, MinModeSession, SaddleError,
    SolidState, SpringKind, band_forces, climbing_image,
};

struct Silent;

impl BandSurface for Silent {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut ndarray::Array1<f64>,
        gradients: &mut ndarray::Array2<f64>,
    ) -> Result<(), SaddleError> {
        let _ = (positions, energies, gradients);
        Ok(())
    }
}

fn three() -> ndarray::Array2<f64> {
    array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
}

#[test]
fn a_short_band_is_a_shape_error() {
    let positions = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let err = band_forces(&BandConfig::default(), &Silent, positions.view(), None).unwrap_err();
    assert!(matches!(err, SaddleError::Shape(_)), "{err}");
}

#[test]
fn a_non_finite_start_is_refused() {
    let built = MinModeSession::new(
        MinModeConfig::default(),
        array![f64::NAN, 0.0, 0.0],
        array![1.0, 0.0, 0.0],
    );
    assert!(matches!(built, Err(SaddleError::NonFinite(_))));
    let config = MinModeConfig {
        max_rotations: 0,
        ..MinModeConfig::default()
    };
    let built = MinModeSession::new(config, array![0.2, 0.0, 0.0], array![1.0, 0.0, 0.0]);
    assert!(matches!(built, Err(SaddleError::Invalid(_))));
}

#[test]
fn a_nan_force_tolerance_is_refused() {
    let config = BandConfig {
        force_tol: f64::NAN,
        min_mode: None,
        ..BandConfig::default()
    };
    let built = BandSession::new(config, three());
    assert!(matches!(built, Err(SaddleError::Invalid(_))));
}

#[test]
fn a_solid_band_refuses_live_onsager_machlup_springs() {
    let cell = Cell([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]);
    let config = BandConfig {
        min_mode: None,
        spring: SpringKind::OnsagerMachlupLive { k: 1.0 },
        solid: Some(SolidState::new(vec![cell, cell, cell])),
        ..BandConfig::default()
    };
    let built = BandSession::new(config, three());
    assert!(matches!(built, Err(SaddleError::Invalid(_))));
}

#[test]
fn an_endpoint_that_slopes_into_the_band_does_not_climb() {
    let positions = three();
    let energies = array![0.0, 1.0, 2.0];
    let mut into = ndarray::Array2::zeros((3, 3));
    into[(2, 0)] = 1.0;
    let config = BandConfig {
        force_tol: 1e-3,
        ..BandConfig::default()
    };
    let blocked = climbing_image(&config, positions.view(), energies.view(), into.view()).unwrap();
    assert_eq!(blocked, None);
    let mut away = ndarray::Array2::zeros((3, 3));
    away[(2, 0)] = -1.0;
    let climbed = climbing_image(&config, positions.view(), energies.view(), away.view()).unwrap();
    assert_eq!(climbed, Some(1));
}

#[cfg(feature = "capi")]
#[test]
fn the_c_projection_uses_the_same_climbing_rule() {
    use rgsaddle::capi::{RGSADDLE_OK, RgsaddleBandConfig, RgsaddleVersion, rgsaddle_band_project};
    let positions = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0];
    let energies = [0.0, 1.0, 2.0];
    let mut gradients = [0.0; 9];
    gradients[6] = 1.0;
    let config = RgsaddleBandConfig {
        version: RgsaddleVersion {
            major: 1,
            minor: 35,
        },
        flags: 0,
        tangent: 1,
        spring: 0,
        projection: 1,
        method: 0,
        spring_k: 0.1,
        spring_ks: std::ptr::null(),
        ci_trigger_factor: 1.0,
        ci_trigger_force: 0.0,
        cell: std::ptr::null(),
        force_tol: 1e-3,
        max_move: 0.2,
        memory: 0,
    };
    let mut forces = [0.0; 9];
    let mut ci = 7_i64;
    let status = unsafe {
        rgsaddle_band_project(
            &config,
            3,
            1,
            positions.as_ptr(),
            energies.as_ptr(),
            gradients.as_ptr(),
            forces.as_mut_ptr(),
            &mut ci,
        )
    };
    assert_eq!(status, RGSADDLE_OK);
    assert_eq!(ci, -1);
    assert!(forces.iter().all(|v| v.abs() < 1e-12), "{forces:?}");
}
