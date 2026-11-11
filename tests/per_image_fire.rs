//! A local FIRE step does not let one image's per-atom cap shrink the others.
//!
//! Sheppard, Terrell and Henkelman, J. Chem. Phys. 128, 134106 (2008),
//! doi:10.1063/1.2841941. The default stays one optimizer for the band.

use ndarray::{Array1, Array2, ArrayView2};
use rgsaddle::{BandConfig, BandSession, BandSurface, SaddleError};

struct Push {
    heavy: f64,
    light: f64,
}

impl BandSurface for Push {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        // Four images, one atom. Interior rows are 1 and 2 when the
        // whole band is evaluated, and 0 and 1 when only the interior is.
        let offset = if positions.nrows() == 4 { 0 } else { 1 };
        for i in 0..positions.nrows() {
            let band = offset + i;
            let gy = if band == 1 {
                -self.heavy
            } else if band == 2 {
                -self.light
            } else {
                0.0
            };
            energies[i] = 0.0;
            gradients[(i, 0)] = 0.0;
            gradients[(i, 1)] = gy;
            gradients[(i, 2)] = 0.0;
        }
        Ok(())
    }
}

fn config(per_image: bool) -> BandConfig {
    BandConfig {
        climbing: None,
        spring: rgsaddle::SpringKind::Uniform { k: 0.0 },
        force_tol: 1e-12,
        max_move: 0.2,
        time_step: 1.0,
        min_mode: None,
        method: rgmin::Method::Fire {
            kind: rgmin::FireKind::V1,
        },
        per_image_fire: per_image,
        ..BandConfig::default()
    }
}

fn band() -> Array2<f64> {
    Array2::from_shape_fn((4, 3), |(i, k)| if k == 0 { i as f64 } else { 0.0 })
}

#[test]
fn the_default_keeps_one_step_for_the_band() {
    assert!(!BandConfig::default().per_image_fire);
}

#[test]
fn one_image_hitting_the_cap_does_not_shrink_the_other() {
    let surface = Push {
        heavy: 100.0,
        light: 0.1,
    };
    let mut global = BandSession::new(config(false), band()).unwrap();
    let mut local = BandSession::new(config(true), band()).unwrap();
    global.step(&surface).unwrap();
    local.step(&surface).unwrap();
    let global_light = global.positions()[(2, 1)].abs();
    let local_light = local.positions()[(2, 1)].abs();
    assert!(
        local_light > 10.0 * global_light,
        "local {local_light} should clear the globally capped {global_light}"
    );
}
