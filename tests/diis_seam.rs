//! The band accepts the Pulay residual subspace and steps on the force.

use ndarray::{Array1, Array2, ArrayView2, array};
use rgsaddle::{BandConfig, BandSession, BandSurface, SaddleError, SpringKind};

/// `E = 1/2 y^2`. The improved tangent at this symmetric peak is `+x`,
/// the spring constant is zero, and the NEB force on the middle image
/// is `(0, -y, 0)`. The first Pulay step has one stored pair, so it is
/// a steepest kick of length `istep` (1). The per-atom cap of 0.2
/// shortens that kick, so `y` goes from 0.5 to 0.3 and `x` stays at 1.
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
fn diis_is_accepted_and_steps_the_perpendicular_force() {
    let band = array![[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [2.0, 0.0, 0.0]];
    let built = BandSession::new(
        BandConfig {
            method: rgmin::Method::diis(),
            climbing: None,
            spring: SpringKind::Uniform { k: 0.0 },
            max_move: 0.2,
            ..BandConfig::default()
        },
        band.clone(),
    );
    let Ok(mut session) = built else {
        panic!("DIIS session was refused");
    };
    session.step(&Valley).expect("DIIS step");
    let pos = session.positions();
    assert_eq!(pos.row(0), band.row(0));
    assert_eq!(pos.row(2), band.row(2));
    let mid = pos.row(1);
    assert!((mid[0] - 1.0).abs() < 1e-12, "x {}", mid[0]);
    assert!((mid[1] - 0.3).abs() < 1e-12, "y {}", mid[1]);
    assert!(mid[2].abs() < 1e-12, "z {}", mid[2]);
}

#[test]
fn steepest_descent_is_still_refused() {
    let band = array![[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [2.0, 0.0, 0.0]];
    let built = BandSession::new(
        BandConfig {
            method: rgmin::Method::Steepest,
            ..BandConfig::default()
        },
        band,
    );
    assert!(built.is_err());
}
