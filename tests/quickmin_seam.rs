//! The band can name quick-min before the pinned minimizer provides it.

use ndarray::array;
use rgsaddle::{BandConfig, BandSession};

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

#[test]
fn selecting_quickmin_fails_until_the_minimizer_publishes_it() {
    let built = BandSession::new(
        BandConfig {
            quickmin: true,
            ..BandConfig::default()
        },
        array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
    );
    let Err(err) = built else {
        panic!("quick-min was accepted");
    };
    let text = err.to_string();
    assert!(text.contains("Method::QuickMin"), "{text}");
}
