//! OCI-NEB after the dimer has taken the climbing image.
//!
//! The digest is SHA-256 of the little-endian positions. Threads 1, 2,
//! 3 and 8 share one hash. The dimer reports a negative curvature, and
//! the path differs from CI-NEB at the same step count.

use ndarray::{Array2, ArrayViewMut2};
use rgsaddle::SaddleError;
use rgsaddle::stencil::{Preset, Stencil, Surface};
use sha2::{Digest, Sha256};

const STEPS: usize = 12;

fn band() -> Array2<f64> {
    let n = 7;
    let mut images = Array2::zeros((n, 2));
    for i in 0..n {
        let t = i as f64 / (n - 1) as f64;
        images[(i, 0)] = -1.0 + 2.0 * t;
        images[(i, 1)] = 0.4;
    }
    images
}

struct Well;

impl Surface for Well {
    fn eval(
        &self,
        _ids: &[u64],
        positions: ndarray::ArrayView2<f64>,
        energies: &mut [f64],
        mut gradients: ArrayViewMut2<f64>,
    ) -> Result<(), SaddleError> {
        for (row, energy) in energies.iter_mut().enumerate() {
            let x = positions[[row, 0]];
            let y = positions[[row, 1]];
            *energy = (x * x - 1.0).powi(2) + y * y;
            gradients[[row, 0]] = 4.0 * x * (x * x - 1.0);
            gradients[[row, 1]] = 2.0 * y;
        }
        Ok(())
    }
}

fn run(preset: Preset, threads: usize) -> (Array2<f64>, bool) {
    let mut search = Stencil::band(band())
        .preset(preset)
        .threads(threads)
        .climb_trigger(0.5, 1.0e300)
        .force_tol(1e-3)
        .max_move(0.2)
        .build()
        .expect("band");
    let mut dimer = false;
    for _ in 0..STEPS {
        let report = search.step_surface(&Well).expect("step");
        if report.curvature.is_finite() && report.curvature < 0.0 {
            dimer = true;
        }
    }
    (search.positions(), dimer)
}

fn digest(positions: &Array2<f64>) -> String {
    let mut hasher = Sha256::new();
    for value in positions.iter() {
        hasher.update(value.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

#[test]
fn oci_neb_hash_is_stable_and_differs_from_ci_neb() {
    let (ci, _) = run(Preset::CiNeb, 1);
    let ci_hash = digest(&ci);
    let mut hashes = Vec::new();
    for threads in [1usize, 2, 3, 8] {
        let (positions, dimer) = run(Preset::OciNeb, threads);
        assert!(dimer, "the dimer did not run");
        hashes.push(digest(&positions));
    }
    assert!(hashes.windows(2).all(|pair| pair[0] == pair[1]));
    assert_ne!(hashes[0], ci_hash, "OCI-NEB matched CI-NEB");
    let recorded = include_str!("golden/oci.txt");
    let line = recorded.lines().next().expect("hash line");
    assert_eq!(hashes[0], line, "oci golden");
}
