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
    let mut config = MinModeConfig::default();
    config.max_rotations = 0;
    let built = MinModeSession::new(config, array![0.2, 0.0, 0.0], array![1.0, 0.0, 0.0]);
    assert!(matches!(built, Err(SaddleError::Invalid(_))));
}

#[test]
fn a_nan_force_tolerance_is_refused() {
    let mut config = BandConfig::default();
    config.force_tol = f64::NAN;
    config.min_mode = None;
    let built = BandSession::new(config, three());
    assert!(matches!(built, Err(SaddleError::Invalid(_))));
}

#[test]
fn a_solid_band_refuses_live_onsager_machlup_springs() {
    let cell = Cell([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]);
    let mut config = BandConfig::default();
    config.min_mode = None;
    config.spring = SpringKind::OnsagerMachlupLive { k: 1.0 };
    config.solid = Some(SolidState::new(vec![cell, cell, cell]));
    let built = BandSession::new(config, three());
    assert!(matches!(built, Err(SaddleError::Invalid(_))));
}

#[test]
fn an_endpoint_that_slopes_into_the_band_does_not_climb() {
    let positions = three();
    let energies = array![0.0, 1.0, 2.0];
    let mut into = ndarray::Array2::zeros((3, 3));
    into[(2, 0)] = 1.0;
    let mut config = BandConfig::default();
    config.force_tol = 1e-3;
    let blocked = climbing_image(&config, positions.view(), energies.view(), into.view()).unwrap();
    assert_eq!(blocked, None);
    let mut away = ndarray::Array2::zeros((3, 3));
    away[(2, 0)] = -1.0;
    let climbed = climbing_image(&config, positions.view(), energies.view(), away.view()).unwrap();
    assert_eq!(climbed, Some(1));
}
