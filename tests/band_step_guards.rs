//! A band step that cannot make progress returns a typed error instead
//! of Ok: a force whose squared norm overflows, and a step that leaves
//! the band unchanged while the force is above its tolerance.

use ndarray::{Array1, Array2, ArrayView2, array};
use rgsaddle::{BandConfig, BandSession, BandSurface, SaddleError};

/// A constant gradient on every image.
struct Constant([f64; 3]);

impl BandSurface for Constant {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for r in 0..positions.nrows() {
            energies[r] = positions[(r, 0)];
            for c in 0..3 {
                gradients[(r, c)] = self.0[c];
            }
        }
        Ok(())
    }
}

fn config() -> BandConfig {
    BandConfig {
        climbing: None,
        force_tol: 1e-8,
        ..BandConfig::default()
    }
}

#[test]
fn an_overflowing_force_norm_is_non_finite() {
    let band = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    let mut session = BandSession::new(config(), band.clone()).unwrap();
    // The perpendicular force is 1e200: finite, but its square is not.
    let err = session.step(&Constant([0.0, 1e200, 0.0])).unwrap_err();
    assert!(
        matches!(err, SaddleError::NonFinite("band force norm")),
        "{err}"
    );
    assert_eq!(session.positions(), band.view());
}

#[test]
fn a_step_that_cannot_move_the_band_is_stalled() {
    // At 1e11 one unit in the last place is 1.5e-5, so a step of order
    // 1e-7 rounds to no move while the force stays at 1e-7. (rgmin keeps
    // coordinates inside +-1e12, so the band stays below that.)
    let big = 1e11;
    let band = array![
        [big, big, big],
        [big + 1.0, big, big],
        [big + 2.0, big, big]
    ];
    let mut session = BandSession::new(config(), band.clone()).unwrap();
    let err = session.step(&Constant([0.0, 1e-7, 0.0])).unwrap_err();
    assert!(matches!(err, SaddleError::Stalled(_)), "{err}");
    assert_eq!(session.positions(), band.view());
}

/// Fails its first call, then answers as `Constant`.
struct FailOnce {
    failed: std::sync::atomic::AtomicBool,
    inner: Constant,
}

impl BandSurface for FailOnce {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        if !self.failed.swap(true, std::sync::atomic::Ordering::Relaxed) {
            return Err(SaddleError::Surface("first call".into()));
        }
        self.inner.eval(positions, energies, gradients)
    }
}

#[test]
fn a_failed_evaluation_is_not_reused_by_the_next_step() {
    let band = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    let mut session = BandSession::new(config(), band.clone()).unwrap();
    let surface = FailOnce {
        failed: std::sync::atomic::AtomicBool::new(false),
        inner: Constant([0.0, 1.0, 0.0]),
    };
    assert!(matches!(
        session.step(&surface),
        Err(SaddleError::Surface(_))
    ));
    assert_eq!(session.positions(), band.view());
    let report = session.step(&surface).unwrap();
    assert!(report.surface_rows > 0, "{report:?}");
    assert!(session.positions()[(1, 1)] < 0.0, "{:?}", session.positions());
}
