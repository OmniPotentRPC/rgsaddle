//! Onsager-Machlup displacements rebuilt from the band forces at every
//! evaluation, as eOn's `buildSpringStrategy` does on each
//! `updateForces`, on a seven-image band over a double well.

use ndarray::{Array1, Array2, ArrayView2};
use rgsaddle::rtr::band_forces;
use rgsaddle::spring::onsager_machlup_displacements;
use rgsaddle::{BandConfig, BandSession, BandSurface, SaddleError, SpringKind};

const N: usize = 7;
const K: f64 = 5.0;

fn double_well_at(x: &[f64]) -> (f64, [f64; 3]) {
    let (a, b, c) = (x[0], x[1], x[2]);
    let e = (a * a - 1.0).powi(2) + 2.0 * b * b + 2.0 * c * c;
    (e, [4.0 * a * (a * a - 1.0), 4.0 * b, 4.0 * c])
}

struct DoubleWell;

impl BandSurface for DoubleWell {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (i, row) in positions.outer_iter().enumerate() {
            let (e, g) = double_well_at(row.as_slice().unwrap());
            energies[i] = e;
            gradients.row_mut(i).assign(&Array1::from(g.to_vec()));
        }
        Ok(())
    }
}

fn bowed_band() -> Array2<f64> {
    Array2::from_shape_fn((N, 3), |(i, c)| {
        let t = i as f64 / (N - 1) as f64;
        [-1.0 + 2.0 * t, 0.3 * (std::f64::consts::PI * t).sin(), 0.0][c]
    })
}

fn config(spring: SpringKind) -> BandConfig {
    BandConfig {
        spring,
        climbing: None,
        force_tol: 1e-5,
        max_move: 0.1,
        method: rgmin::Method::Lbfgs { memory: 10 },
        min_mode: None,
        ..BandConfig::default()
    }
}

fn displacements_at(positions: ArrayView2<f64>) -> Vec<Array1<f64>> {
    let mut energies = Array1::zeros(N);
    let mut gradients = Array2::zeros((N, 3));
    DoubleWell
        .eval(positions, &mut energies, &mut gradients)
        .unwrap();
    onsager_machlup_displacements(gradients.view(), K)
}

fn frozen(l_vecs: Vec<Array1<f64>>) -> SpringKind {
    SpringKind::OnsagerMachlup { k: K, l_vecs }
}

const LIVE: SpringKind = SpringKind::OnsagerMachlupLive { k: K };

#[test]
fn the_displacements_follow_the_forces_across_iterations() {
    let start = bowed_band();
    let l_start = displacements_at(start.view());
    assert!(
        l_start[1..N - 1]
            .iter()
            .all(|l| l.iter().any(|&v| v != 0.0))
    );
    let mut walker = BandSession::new(config(LIVE), start).unwrap();
    let mut previous = l_start.clone();
    for _ in 0..8 {
        walker.step(&DoubleWell).unwrap();
        let here = walker.positions().to_owned();
        let l_here = displacements_at(here.view());
        assert_ne!(l_here, previous, "the forces moved but L did not");
        let force = |s| band_forces(&config(s), &DoubleWell, here.view(), None).unwrap();
        assert_eq!(force(LIVE).force, force(frozen(l_here.clone())).force);
        assert_ne!(force(LIVE).force, force(frozen(l_start.clone())).force);
        let step = |s| {
            let mut session = BandSession::new(config(s), here.clone()).unwrap();
            session.step(&DoubleWell).unwrap();
            session.positions().to_owned()
        };
        assert_eq!(step(LIVE), step(frozen(l_here.clone())));
        assert_ne!(step(LIVE), step(frozen(l_start.clone())));
        previous = l_here;
    }
}

