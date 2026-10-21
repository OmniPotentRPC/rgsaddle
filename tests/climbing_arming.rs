//! The default climbing image arms once the band has settled.

use ndarray::{Array1, Array2, ArrayView2, array};
use rgsaddle::{BandConfig, BandSession, BandSurface, CiConfig, SaddleError};

#[test]
fn the_climbing_image_arms_once_the_band_settles() {
    // CiConfig::default arms under a largest component of 0.5: with a
    // perpendicular force of 2 on a peaked band nothing climbs, with 0.1
    // the peak climbs.
    let band = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    struct Peak(f64);
    impl BandSurface for Peak {
        fn eval(
            &self,
            positions: ArrayView2<f64>,
            energies: &mut Array1<f64>,
            gradients: &mut Array2<f64>,
        ) -> Result<(), SaddleError> {
            for r in 0..positions.nrows() {
                let x = positions[(r, 0)];
                energies[r] = 1.0 - (x - 1.0) * (x - 1.0);
                gradients[(r, 0)] = 0.0;
                gradients[(r, 1)] = self.0;
                gradients[(r, 2)] = 0.0;
            }
            Ok(())
        }
    }
    let config = BandConfig {
        force_tol: 1e-8,
        ..BandConfig::default()
    };
    assert_eq!(config.climbing.map(|c| c.trigger_force), Some(0.5));
    let mut steep = BandSession::new(config.clone(), band.clone()).unwrap();
    assert_eq!(steep.step(&Peak(2.0)).unwrap().ci_index, None);
    let mut settled = BandSession::new(config, band).unwrap();
    assert_eq!(settled.step(&Peak(0.1)).unwrap().ci_index, Some(1));
}

#[test]
fn always_climbs_from_the_first_evaluation() {
    let ci = CiConfig::always();
    assert_eq!(ci.trigger_factor, 1.0);
    assert!(ci.trigger_force.is_infinite());
    let after = CiConfig::after(0.2);
    assert_eq!((after.trigger_factor, after.trigger_force), (0.0, 0.2));
    let rel = CiConfig::after_rel(0.8);
    assert_eq!((rel.trigger_factor, rel.trigger_force), (0.8, 0.0));
}
