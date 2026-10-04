//! The band runs quick-min when asked. FIRE stays the default.

use ndarray::{Array1, Array2, ArrayView2, array};
use rgsaddle::{BandConfig, BandSession, BandSurface, SaddleError, SpringKind};

#[test]
fn fire_remains_the_default_band_method() {
    let config = BandConfig::default();
    assert!(!config.quickmin);
    assert!(matches!(
        config.method,
        rgmin::Method::Fire {
            kind: rgmin::FireKind::V2
        }
    ));
}

/// `E = 1/2 y^2`. The improved tangent at this symmetric peak is `+x`,
/// the spring constant is zero, and the NEB force on the middle image
/// is `(0, -y, 0)`. From rest, with the session step equal to 1, the
/// quick-min displacement is that force. The per-atom cap of 0.2
/// shortens it, so `y` goes from 0.5 to 0.3 and `x` stays at 1.
struct Valley;

impl BandSurface for Valley {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (i, row) in positions.outer_iter().enumerate() {
            let y = row[1];
            energies[i] = 0.5 * y * y;
            gradients[(i, 0)] = 0.0;
            gradients[(i, 1)] = y;
            gradients[(i, 2)] = 0.0;
        }
        Ok(())
    }
}

#[test]
fn quickmin_steps_the_perpendicular_force() {
    let band = array![[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [2.0, 0.0, 0.0]];
    let built = BandSession::new(
        BandConfig {
            quickmin: true,
            climbing: None,
            spring: SpringKind::Uniform { k: 0.0 },
            max_move: 0.2,
            ..BandConfig::default()
        },
        band.clone(),
    );
    let Ok(mut session) = built else {
        panic!("quick-min session was refused");
    };
    session.step(&Valley).expect("quick-min step");
    let pos = session.positions();
    assert_eq!(pos.row(0), band.row(0));
    assert_eq!(pos.row(2), band.row(2));
    let mid = pos.row(1);
    assert!((mid[0] - 1.0).abs() < 1e-12, "x {}", mid[0]);
    assert!((mid[1] - 0.3).abs() < 1e-12, "y {}", mid[1]);
    assert!(mid[2].abs() < 1e-12, "z {}", mid[2]);
}
