//! eOn `FIRE::step` does not rescale the velocity when the per-atom cap
//! shortens the move. rgmin does. On a force that turns after that cap,
//! the next displacement follows the old velocity only when the scale
//! is left off.

use ndarray::{Array1, Array2, ArrayView2};
use rgsaddle::{BandConfig, BandSession, BandSurface, SaddleError};

struct Turn;

impl BandSurface for Turn {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for i in 0..positions.nrows() {
            let (fx, fy) = push(positions[(i, 0)]);
            energies[i] = 0.0;
            gradients[(i, 0)] = -fx;
            gradients[(i, 1)] = -fy;
            gradients[(i, 2)] = 0.0;
        }
        Ok(())
    }
}

fn push(x: f64) -> (f64, f64) {
    if x < 0.15 { (100.0, 1.0) } else { (1.0, 100.0) }
}

fn band() -> Array2<f64> {
    Array2::from_shape_fn((3, 3), |(i, k)| if k == 2 { i as f64 } else { 0.0 })
}

fn config() -> BandConfig {
    BandConfig {
        climbing: None,
        spring: rgsaddle::SpringKind::Uniform { k: 0.0 },
        // The plain band keeps the true force, so this step is the
        // single-atom integrator and not a nudged projection.
        projection: rgsaddle::ProjectionKind::PlainElasticBand,
        force_tol: 1e-12,
        max_move: 0.2,
        time_step: 0.1,
        min_mode: None,
        method: rgmin::Method::Fire {
            kind: rgmin::FireKind::V1,
        },
        ..BandConfig::default()
    }
}

fn pos_after(rescale: bool, steps: usize) -> (f64, f64) {
    let mut session = BandSession::new(config(), band()).unwrap();
    session.set_fire_clamp_rescale(rescale);
    for _ in 0..steps {
        session.step(&Turn).unwrap();
    }
    (session.positions()[(1, 0)], session.positions()[(1, 1)])
}

/// Inline `FIRE::step`: one atom, the same turn in the force, no
/// velocity rescale.
fn eon_xy(steps: usize) -> (f64, f64) {
    let mut x: f64 = 0.0;
    let mut y: f64 = 0.0;
    let mut vx: f64 = 0.0;
    let mut vy: f64 = 0.0;
    let mut dt: f64 = 0.1;
    let mut alpha: f64 = 0.1;
    let mut n_pos = 0usize;
    for _ in 0..steps {
        let (fx, fy) = push(x);
        vx += fx * dt;
        vy += fy * dt;
        let mut dx = vx * dt;
        let mut dy = vy * dt;
        let norm = (dx * dx + dy * dy).sqrt();
        if norm > 0.2 {
            dx *= 0.2 / norm;
            dy *= 0.2 / norm;
        }
        x += dx;
        y += dy;
        let (fx, fy) = push(x);
        let power = fx * vx + fy * vy;
        let vnorm = (vx * vx + vy * vy).sqrt();
        let fnorm = (fx * fx + fy * fy).sqrt();
        if fnorm > 0.0 {
            vx = (1.0 - alpha) * vx + alpha * fx / fnorm * vnorm;
            vy = (1.0 - alpha) * vy + alpha * fy / fnorm * vnorm;
        }
        if power >= 0.0 {
            n_pos += 1;
            if n_pos > 5 {
                dt = (dt * 1.1).min(0.25);
                alpha *= 0.99;
            }
        } else {
            dt *= 0.5;
            vx = 0.0;
            vy = 0.0;
            alpha = 0.1;
            n_pos = 0;
        }
    }
    (x, y)
}

#[test]
fn a_capped_fire_step_keeps_the_eon_velocity() {
    let kept = pos_after(false, 2);
    let rescaled = pos_after(true, 2);
    let reference = eon_xy(2);
    let split = (kept.0 - rescaled.0).hypot(kept.1 - rescaled.1);
    let miss = (kept.0 - reference.0).hypot(kept.1 - reference.1);
    assert!(
        miss < 1e-9,
        "kept {kept:?} reference {reference:?} rescaled {rescaled:?}"
    );
    assert!(
        split > 1e-2,
        "rescaling should leave the eOn path, split {split} kept {kept:?} rescaled {rescaled:?}"
    );
}
