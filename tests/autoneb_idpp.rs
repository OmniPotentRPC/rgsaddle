//! AutoNEB insertion with the opt-in IDPP interpolant.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, array};
use rgsaddle::band::Cell;
use rgsaddle::{
    AutoNeb, AutoNebInterpolant, AutoNebTuning, BandConfig, BandSurface, IdppConfig, SaddleError,
    idpp_path,
};

/// Default-path digest of [`crossing`] at base main f2ded3bb, one and
/// two image-evaluation threads, before the interpolant existed.
const BASE_DIGEST: u64 = 0x9c05_29d3_d1f6_5e1d;

/// Harmonic pair term on squared distances plus a flat-plane term on z.
struct Pairs;

impl BandSurface for Pairs {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (i, row) in positions.outer_iter().enumerate() {
            let atoms = row.len() / 3;
            let mut energy = 0.0;
            gradients.row_mut(i).fill(0.0);
            for a in 0..atoms {
                let z = row[3 * a + 2];
                energy += z * z;
                gradients[(i, 3 * a + 2)] += 2.0 * z;
                for b in (a + 1)..atoms {
                    let d = [
                        row[3 * a] - row[3 * b],
                        row[3 * a + 1] - row[3 * b + 1],
                        row[3 * a + 2] - row[3 * b + 2],
                    ];
                    let excess = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] - 1.0;
                    energy += 0.25 * excess * excess;
                    for c in 0..3 {
                        gradients[(i, 3 * a + c)] += excess * d[c];
                        gradients[(i, 3 * b + c)] -= excess * d[c];
                    }
                }
            }
            energies[i] = energy;
        }
        Ok(())
    }
}

/// Two atoms cross with a 0.02 impact parameter; a third stays put.
fn crossing() -> Array2<f64> {
    array![
        [0.0, 0.0, 0.0, 2.0, 0.02, 0.0, 1.0, 3.0, 0.0],
        [2.0, 1.0, 0.0, 0.0, 1.02, 0.0, 1.0, 3.0, 0.0],
    ]
}

fn tuning(threads: usize) -> AutoNebTuning {
    AutoNebTuning {
        n_simul: 1,
        n_max: 4,
        local_max_steps: 3,
        climb_max_steps: 3,
        threads,
        ..AutoNebTuning::default()
    }
}

fn digest(search: &AutoNeb) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET;
    let mut feed = |word: u64| {
        for byte in word.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(PRIME);
        }
    };
    for value in search.positions() {
        feed(value.to_bits());
    }
    for energy in search.energies() {
        feed(energy.unwrap_or(f64::NAN).to_bits());
    }
    feed(search.surface_rows() as u64);
    hash
}

fn run(tuning: AutoNebTuning, initial: Array2<f64>) -> AutoNeb {
    let config = BandConfig {
        autoneb: Some(tuning),
        ..BandConfig::default()
    };
    let mut search = AutoNeb::new(config, initial).unwrap();
    search.run(&Pairs).unwrap();
    search
}

fn idpp(threads: usize) -> AutoNebTuning {
    AutoNebTuning {
        interpolant: AutoNebInterpolant::Idpp(IdppConfig::default()),
        ..tuning(threads)
    }
}

fn pair(row: ArrayView1<f64>, a: usize, b: usize) -> f64 {
    (0..3)
        .map(|c| (row[3 * a + c] - row[3 * b + c]).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn same_bits(left: ArrayView1<f64>, right: ArrayView1<f64>) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits())
}

#[test]
fn the_default_path_keeps_the_base_digest_on_one_and_two_threads() {
    for threads in [1, 2] {
        let default = digest(&run(tuning(threads), crossing()));
        assert_eq!(default, BASE_DIGEST, "threads={threads} {default:#018x}");
        let explicit = AutoNebTuning {
            interpolant: AutoNebInterpolant::Linear,
            ..tuning(threads)
        };
        assert_eq!(digest(&run(explicit, crossing())), BASE_DIGEST);
    }
}

#[test]
fn an_idpp_fill_is_the_three_image_idpp_path_and_keeps_the_atoms_apart() {
    let ends = crossing();
    let config = BandConfig {
        autoneb: Some(idpp(1)),
        ..BandConfig::default()
    };
    let search = AutoNeb::new(config, ends.clone()).unwrap();
    let path = idpp_path(ends.row(0), ends.row(1), 3, None, &IdppConfig::default()).unwrap();
    assert_eq!(search.positions().nrows(), 3);
    assert!(same_bits(search.positions().row(1), path.row(1)));

    let config = BandConfig {
        autoneb: Some(tuning(1)),
        ..BandConfig::default()
    };
    let linear = AutoNeb::new(config, ends.clone()).unwrap();
    let target = 0.5 * (pair(ends.row(0), 0, 1) + pair(ends.row(1), 0, 1));
    assert!(pair(linear.positions().row(1), 0, 1) < 0.05 * target);
    let kept = pair(search.positions().row(1), 0, 1);
    assert!(kept > 0.5 * target, "{kept} against target {target}");
}

#[test]
fn an_idpp_run_differs_from_linear_and_two_threads_match_one() {
    let serial = run(idpp(1), crossing());
    let parallel = run(idpp(2), crossing());
    assert_eq!(digest(&serial), digest(&parallel));
    assert_ne!(digest(&serial), BASE_DIGEST);
    let linear = run(tuning(1), crossing());
    assert_eq!(serial.insertions().len(), linear.insertions().len());
    assert!(!same_bits(
        serial.positions().row(1),
        linear.positions().row(1)
    ));
}

#[test]
fn one_atom_has_no_pairs_so_idpp_places_the_linear_midpoint() {
    let ends = array![[0.0, 0.0, 0.0], [1.5, 0.5, 0.25]];
    for threads in [1, 2] {
        assert_eq!(
            digest(&run(idpp(threads), ends.clone())),
            digest(&run(tuning(threads), ends.clone())),
            "threads={threads}"
        );
    }
}

#[test]
fn a_bad_idpp_stop_and_a_singular_cell_are_refused() {
    let stop = AutoNebTuning {
        interpolant: AutoNebInterpolant::Idpp(IdppConfig {
            max_iterations: 0,
            ..IdppConfig::default()
        }),
        ..tuning(1)
    };
    let config = BandConfig {
        autoneb: Some(stop),
        ..BandConfig::default()
    };
    assert!(matches!(
        AutoNeb::new(config, crossing()),
        Err(SaddleError::Invalid(_))
    ));
    let slab = Cell([[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 0.0]]);
    let config = BandConfig {
        autoneb: Some(idpp(1)),
        cell: Some(slab),
        ..BandConfig::default()
    };
    assert!(matches!(
        AutoNeb::new(config, crossing()),
        Err(SaddleError::Invalid(_))
    ));
    let config = BandConfig {
        autoneb: Some(tuning(1)),
        cell: Some(slab),
        ..BandConfig::default()
    };
    assert!(AutoNeb::new(config, crossing()).is_ok());
}
