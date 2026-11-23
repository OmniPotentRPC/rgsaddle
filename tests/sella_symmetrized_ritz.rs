//! `SellaSaddleConfig::symmetrized_ritz` on a linear gradient field whose
//! Jacobian has an antisymmetric part, so every finite-difference
//! Hessian-vector product is asymmetric the way a noisy one is.

use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, Array2, ArrayView1};
use rgsaddle::{PointSurface, SaddleError, SellaSaddleConfig, SellaSaddleSession};

const N_ATOMS: usize = 4;

/// g = (H + K) x with H diagonal, one negative entry, and K antisymmetric.
struct Skewed {
    jacobian: Array2<f64>,
    hessian: Array1<f64>,
    evaluations: AtomicUsize,
}

impl Skewed {
    fn new() -> Self {
        let n = 3 * N_ATOMS;
        let hessian =
            Array1::from_shape_fn(n, |i| if i == 0 { -0.8 } else { 0.5 + 0.25 * i as f64 });
        let mut jacobian = Array2::from_diag(&hessian);
        for i in 0..n {
            for j in (i + 1)..n {
                let k = 0.02 * (((3 * i + 7 * j) % 11) as f64 - 5.0);
                jacobian[[i, j]] += k;
                jacobian[[j, i]] -= k;
            }
        }
        Self {
            jacobian,
            hessian,
            evaluations: AtomicUsize::new(0),
        }
    }
}

impl PointSurface for Skewed {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        self.evaluations.fetch_add(1, Ordering::Relaxed);
        let energy = 0.5 * (&self.hessian * &x).dot(&x);
        Ok((energy, self.jacobian.dot(&x)))
    }
}

fn start() -> Array1<f64> {
    Array1::from_shape_fn(3 * N_ATOMS, |i| {
        0.05 * ((i % 5) as f64 - 2.0) + 0.01 * i as f64
    })
}

fn walk(symmetrized: Option<bool>) -> (Vec<usize>, Array1<f64>) {
    let surface = Skewed::new();
    let mut session = SellaSaddleSession::new(
        SellaSaddleConfig::default().with_small_system_atom_ceiling(false),
        start(),
        Array1::from_elem(N_ATOMS, 1.0),
    )
    .unwrap();
    if let Some(on) = symmetrized {
        session.set_symmetrized_ritz(on);
    }
    let mut per_step = Vec::new();
    for _ in 0..4 {
        let before = surface.evaluations.load(Ordering::Relaxed);
        session.step(&surface).unwrap();
        per_step.push(surface.evaluations.load(Ordering::Relaxed) - before);
    }
    (per_step, session.position().to_owned())
}

/// Bits of the four-step walk with the default config and the atom
/// ceiling off, recorded before either option existed.
const DEFAULT_BITS: [u64; 12] = [
    0xbfff7a56b3b9309e,
    0xbfec29aa047b94c8,
    0xbfa8f3f517a51bc9,
    0x3fecdce5b34f0ac0,
    0x3fe15c67823c9e6c,
    0x3fa1bca4339c305c,
    0x3fe709b76c7b6199,
    0x3fd43e6e43dc47b6,
    0x3fbdb120d09b1650,
    0x3fe0d09fa3d0eaba,
    0x3fc87abcdd6c3fe7,
    0x3fadc094618b10c0,
];

#[test]
fn walk_without_either_option_is_unchanged() {
    assert!(!SellaSaddleConfig::default().symmetrized_ritz);
    for symmetrized in [None, Some(false)] {
        let (per_step, x) = walk(symmetrized);
        assert_eq!(per_step, [8, 2, 2, 2]);
        let bits: Vec<u64> = x.iter().map(|v| v.to_bits()).collect();
        assert_eq!(bits, DEFAULT_BITS);
    }
}

#[test]
fn symmetrized_ritz_stops_the_first_diagonalization_sooner() {
    let (per_step, x) = walk(Some(true));
    assert_eq!(per_step, [6, 2, 2, 2]);
    assert!(x.iter().all(|v| v.is_finite()));
    let bits: Vec<u64> = x.iter().map(|v| v.to_bits()).collect();
    assert_ne!(bits, DEFAULT_BITS);
}
