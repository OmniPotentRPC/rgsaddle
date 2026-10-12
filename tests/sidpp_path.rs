//! S-IDPP path after construction and a fixed number of FIRE steps.
//!
//! The digest is SHA-256 of the little-endian positions. Threads 1, 2,
//! 3 and 8 share one hash. The bond stays longer than the linear chord
//! and longer than IDPP at the same spring and step count.

use ndarray::{Array1, Array2};
use rgsaddle::{BandSurface, Preset, SaddleError, Stencil};
use sha2::{Digest, Sha256};

const STEPS: usize = 12;

fn rotation() -> Array2<f64> {
    let start = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let end = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let n = 5;
    let mut images = Array2::zeros((n, 6));
    for i in 0..n {
        let t = i as f64 / (n - 1) as f64;
        for k in 0..6 {
            images[(i, k)] = (1.0 - t) * start[k] + t * end[k];
        }
    }
    images
}

struct Ignore;

impl BandSurface for Ignore {
    fn eval(
        &self,
        positions: ndarray::ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        let _ = (positions, energies, gradients);
        Err(SaddleError::Invalid(
            "S-IDPP does not evaluate the caller surface".into(),
        ))
    }
}

fn run(preset: Preset, threads: usize) -> Array2<f64> {
    let mut search = Stencil::band(rotation())
        .preset(preset)
        .threads(threads)
        .k(0.1)
        .force_tol(1e-6)
        .max_move(0.2)
        .build()
        .expect("path");
    for _ in 0..STEPS {
        search.step_band(&Ignore).expect("step");
    }
    search.positions()
}

fn digest(positions: &Array2<f64>) -> String {
    let mut hasher = Sha256::new();
    for value in positions.iter() {
        hasher.update(value.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn min_bond(images: &Array2<f64>) -> f64 {
    let mut best = f64::INFINITY;
    for row in images.outer_iter().skip(1).take(images.nrows() - 2) {
        let dx = row[3] - row[0];
        let dy = row[4] - row[1];
        let dz = row[5] - row[2];
        best = best.min((dx * dx + dy * dy + dz * dz).sqrt());
    }
    best
}

#[test]
fn sidpp_rotation_hash_is_stable_across_threads() {
    let linear = min_bond(&rotation());
    let idpp_bond = min_bond(&run(Preset::Idpp, 1));
    let mut hashes = Vec::new();
    for threads in [1usize, 2, 3, 8] {
        let positions = run(Preset::SIdpp, threads);
        let bond = min_bond(&positions);
        assert!(bond > linear + 0.15, "bond {bond} linear {linear}");
        assert!(bond > idpp_bond, "S-IDPP {bond} IDPP {idpp_bond}");
        assert!((positions[(0, 3)] - 1.0).abs() < 1e-12);
        assert!((positions[(4, 4)] - 1.0).abs() < 1e-12);
        hashes.push(digest(&positions));
    }
    assert!(hashes.windows(2).all(|pair| pair[0] == pair[1]));
    let recorded = include_str!("golden/sidpp.txt");
    let line = recorded.lines().next().expect("hash line");
    assert_eq!(hashes[0], line, "sidpp golden");
}
