//! The climbing image stays on a strict local energy maximum.
//!
//! eOn `NudgedElasticBand::updateForces` keeps the current climber while
//! its energy is above both neighbours. Re-picking the highest interior
//! image every evaluation hops that climber onto a shoulder.

use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, Array2, ArrayView2};
use rgsaddle::{BandConfig, BandSession, BandSurface, CiConfig, SaddleError};

struct TwoPeaks {
    evals: AtomicUsize,
}

impl BandSurface for TwoPeaks {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        let n = self.evals.fetch_add(1, Ordering::Relaxed);
        // Image 1 is a local maximum throughout. After the climber has
        // armed there, image 3 becomes the global interior maximum.
        // An interior-only evaluation is rows 1..n-2 of the band.
        let shoulder = n >= 2;
        let table = if shoulder {
            [0.0, 1.4, 0.2, 2.5, 0.0]
        } else {
            [0.0, 1.4, 0.2, 0.3, 0.0]
        };
        let offset = if positions.nrows() == table.len() {
            0
        } else {
            1
        };
        for (i, row) in positions.outer_iter().enumerate() {
            let _ = row;
            energies[i] = table[offset + i];
            gradients[(i, 0)] = 0.05;
            gradients[(i, 1)] = 0.0;
            gradients[(i, 2)] = 0.0;
        }
        Ok(())
    }
}

#[test]
fn the_climber_stays_while_it_is_a_local_maximum() {
    let mut band = Array2::zeros((5, 3));
    for i in 0..5 {
        band[(i, 0)] = i as f64;
    }
    let config = BandConfig {
        climbing: Some(CiConfig::always()),
        force_tol: 1e-8,
        max_move: 0.01,
        min_mode: None,
        ..BandConfig::default()
    };
    let mut session = BandSession::new(config, band).unwrap();
    let surface = TwoPeaks {
        evals: AtomicUsize::new(0),
    };
    let first = session.step(&surface).unwrap();
    assert_eq!(first.ci_index, Some(1), "the first peak should climb");
    let second = session.step(&surface).unwrap();
    assert_eq!(
        second.ci_index,
        Some(1),
        "a higher shoulder must not steal a climber that is still a local maximum, got {:?}",
        second.ci_index
    );
}
