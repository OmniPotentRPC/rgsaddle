//! `SellaSaddleConfig::small_system_atom_ceiling` on a Cartesian saddle
//! whose unstable coordinate starts far from the saddle, so the first
//! partitioned rational function optimization step wants to move one
//! atom a long way.

use ndarray::{Array1, ArrayView1};
use rgsaddle::{PointSurface, SaddleError, SellaSaddleConfig, SellaSaddleSession};

/// `E = -x_0^2 / 2 + sum_{i > 0} x_i^2 / 2`, one unstable coordinate.
struct Spread;

impl PointSurface for Spread {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let mut energy = -0.5 * x[0] * x[0];
        let mut gradient = x.to_owned();
        gradient[0] = -x[0];
        for i in 1..x.len() {
            energy += 0.5 * x[i] * x[i];
        }
        Ok((energy, gradient))
    }
}

/// Atoms on a line along y, the first pushed 1.5 along x.
fn start(n_atoms: usize) -> Array1<f64> {
    let mut x = Array1::zeros(3 * n_atoms);
    for atom in 0..n_atoms {
        x[3 * atom + 1] = atom as f64;
    }
    x[0] = 1.5;
    x
}

fn largest_atom_move(before: &Array1<f64>, after: ArrayView1<f64>) -> f64 {
    let step = &after.to_owned() - before;
    step.exact_chunks(3)
        .into_iter()
        .map(|d| d.dot(&d).sqrt())
        .fold(0.0, f64::max)
}

struct Walk {
    moves: Vec<f64>,
    ceilings: Vec<f64>,
    at_saddle: bool,
    bits: Vec<u64>,
}

fn walk(config: SellaSaddleConfig, n_atoms: usize, ceiling: Option<bool>, steps: usize) -> Walk {
    let mut session =
        SellaSaddleSession::new(config, start(n_atoms), Array1::ones(n_atoms)).unwrap();
    if let Some(on) = ceiling {
        session.set_small_system_atom_ceiling(on);
    }
    let mut moves = Vec::new();
    let mut ceilings = Vec::new();
    let mut at_saddle = false;
    for _ in 0..steps {
        let before = session.position().to_owned();
        let report = session.step(&Spread).unwrap();
        moves.push(largest_atom_move(&before, session.position()));
        ceilings.push(report.delta);
        at_saddle = report.at_saddle;
        if at_saddle {
            break;
        }
    }
    let bits = session.position().iter().map(|v| v.to_bits()).collect();
    Walk {
        moves,
        ceilings,
        at_saddle,
        bits,
    }
}

const TOL: f64 = 1e-6;

#[test]
fn small_search_keeps_every_atom_move_under_the_ceiling_from_the_first_step() {
    // Four atoms leave six free coordinates, a Euclidean radius of 0.6.
    let off = walk(SellaSaddleConfig::default(), 4, Some(false), 1);
    assert!(
        off.moves[0] > 0.2,
        "without the ceiling the first step moved one atom {}",
        off.moves[0]
    );
    let on = walk(SellaSaddleConfig::default(), 4, Some(true), 8);
    assert!(
        on.moves[0] <= 0.1 * (1.0 + TOL),
        "the first step moved one atom {}",
        on.moves[0]
    );
    for k in 1..on.moves.len() {
        assert!(
            on.moves[k] <= on.ceilings[k - 1] * (1.0 + TOL),
            "step {k} moved one atom {} past the ceiling {}",
            on.moves[k],
            on.ceilings[k - 1]
        );
    }
}

#[test]
fn ceiling_is_never_looser_than_the_starting_euclidean_radius() {
    // delta 0.01 on six free coordinates is a radius of 0.06, under 0.1.
    let config = SellaSaddleConfig {
        delta: 0.01,
        ..SellaSaddleConfig::default()
    };
    let on = walk(config, 4, Some(true), 8);
    for (k, largest) in on.moves.iter().enumerate() {
        assert!(
            *largest <= 0.06 * (1.0 + TOL),
            "step {k} moved one atom {largest}, past the starting radius 0.06"
        );
    }
}

#[test]
fn small_search_with_the_ceiling_reaches_the_saddle() {
    let on = walk(SellaSaddleConfig::default(), 4, Some(true), 200);
    assert!(
        on.at_saddle,
        "no saddle after {} steps, last ceiling {}",
        on.moves.len(),
        on.ceilings.last().unwrap()
    );
}

#[test]
fn larger_search_is_the_same_with_the_ceiling_on_or_off() {
    // Six atoms leave twelve free coordinates, a radius of 1.2, which
    // carries the atom ceiling already.
    let off = walk(SellaSaddleConfig::default(), 6, Some(false), 6);
    let on = walk(SellaSaddleConfig::default(), 6, Some(true), 6);
    assert_eq!(on.bits, off.bits);
    assert_eq!(on.moves.len(), off.moves.len());
}

#[test]
fn without_iterative_diagonalization_the_ceiling_changes_nothing() {
    let config = || SellaSaddleConfig {
        eig: false,
        ..SellaSaddleConfig::default()
    };
    let off = walk(config(), 4, Some(false), 6);
    let on = walk(config(), 4, Some(true), 6);
    assert_eq!(on.bits, off.bits);
}
