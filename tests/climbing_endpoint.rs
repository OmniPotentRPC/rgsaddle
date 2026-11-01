//! A band that rises to an endpoint which is not a minimum.
//!
//! eOn's NEB LJ fixture ends on the reactant with one atom pushed 0.5 A
//! into its neighbours, hundreds of eV up a wall. The band rises to that
//! endpoint with no saddle in between, so the CI-NEB there is a plain
//! NEB. This surface has the same shape: a curved valley from a minimum
//! at the origin, with the second endpoint on the valley floor partway up.

use ndarray::{Array1, Array2, ArrayView2, array};
use rgsaddle::{BandConfig, BandSession, BandStatus, BandSurface, SaddleError};

/// One atom in the valley `x^2 + 4 (y - 0.3 x^2)^2 + z^2`.
struct Valley;

impl BandSurface for Valley {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for r in 0..positions.nrows() {
            let (x, y, z) = (positions[(r, 0)], positions[(r, 1)], positions[(r, 2)]);
            let w = y - 0.3 * x * x;
            energies[r] = x * x + 4.0 * w * w + z * z;
            gradients[(r, 0)] = 2.0 * x - 4.8 * x * w;
            gradients[(r, 1)] = 8.0 * w;
            gradients[(r, 2)] = 2.0 * z;
        }
        Ok(())
    }
}

fn band() -> Array2<f64> {
    let (a, b) = (array![0.0, 0.0, 0.0], array![2.0, 1.2, 0.0]);
    let mut band = Array2::zeros((5, 3));
    for i in 0..5 {
        let t = i as f64 / 4.0;
        band.row_mut(i).assign(&(&a * (1.0 - t) + &b * t));
    }
    band
}

fn config() -> BandConfig {
    BandConfig {
        force_tol: 0.01,
        method: rgmin::Method::Lbfgs { memory: 20 },
        min_mode: None,
        ..BandConfig::default()
    }
}

#[test]
fn band_rising_to_a_sloped_endpoint_converges_without_a_climbing_image() {
    let mut session = BandSession::new(config(), band()).unwrap();
    let report = session.run(&Valley, 200).unwrap();
    assert_eq!(report.status, BandStatus::Converged, "{report:?}");
    assert_eq!(report.ci_index, None);
    let energies: Vec<f64> = session
        .positions()
        .rows()
        .into_iter()
        .map(|p| {
            let w = p[1] - 0.3 * p[0] * p[0];
            p[0] * p[0] + 4.0 * w * w
        })
        .collect();
    assert!(energies.windows(2).all(|e| e[0] < e[1]), "{energies:?}");
}
