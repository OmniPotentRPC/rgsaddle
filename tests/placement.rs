//! A copper vacancy and a <100> dumbbell placed back onto the vacancy site.

use ndarray::Array1;
use rgsaddle::{push_past, site_occupancy, translate_onto_sites};

const A0: f64 = 3.615;
const N: usize = 4;

fn lattice() -> Vec<[f64; 3]> {
    let basis = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ];
    let mut sites = Vec::new();
    for i in 0..N {
        for j in 0..N {
            for k in 0..N {
                for b in basis {
                    sites.push([
                        (i as f64 + b[0]) * A0,
                        (j as f64 + b[1]) * A0,
                        (k as f64 + b[2]) * A0,
                    ]);
                }
            }
        }
    }
    sites
}

fn flat(rows: &[[f64; 3]]) -> Array1<f64> {
    Array1::from_iter(rows.iter().flatten().copied())
}

fn cell() -> [[f64; 3]; 3] {
    let l = N as f64 * A0;
    [[l, 0.0, 0.0], [0.0, l, 0.0], [0.0, 0.0, l]]
}

/// Site 0 is vacated, and a dumbbell of two atoms shares the last site, which
/// is a little over one lattice constant away across the cell face.
fn frenkel_pair() -> (Vec<[f64; 3]>, Vec<usize>, [f64; 3]) {
    let sites = lattice();
    let vacancy = sites[0];
    let host = sites.len() - 1;
    let mut atoms: Vec<[f64; 3]> = sites[1..].to_vec();
    let dumbbell = atoms.len() - 1;
    let c = sites[host];
    atoms[dumbbell] = [c[0] - 0.6, c[1], c[2]];
    atoms.push([c[0] + 0.6, c[1], c[2]]);
    (atoms, vec![dumbbell, dumbbell + 1], vacancy)
}

#[test]
fn the_dumbbell_fills_the_vacancy_and_the_crystal_is_perfect() {
    let (atoms, movers, vacancy) = frenkel_pair();
    let sites = lattice();
    let before = site_occupancy(flat(&sites).view(), flat(&atoms).view(), cell()).unwrap();
    assert_eq!(before.vacancy_count(), 1);
    assert_eq!(before.interstitial_count(), 1);
    assert!(!before.is_perfect());

    let placed = translate_onto_sites(
        flat(&atoms).view(),
        cell(),
        [true; 3],
        &movers,
        flat(&[vacancy]).view(),
    )
    .unwrap();
    assert_eq!(placed.pairs.len(), 1);
    let after = site_occupancy(flat(&sites).view(), placed.positions.view(), cell()).unwrap();
    assert!(after.is_perfect());
}

#[test]
fn the_nearer_dumbbell_atom_is_the_one_that_moves() {
    let (atoms, movers, vacancy) = frenkel_pair();
    let placed = translate_onto_sites(
        flat(&atoms).view(),
        cell(),
        [true; 3],
        &movers,
        flat(&[vacancy]).view(),
    )
    .unwrap();
    let (moved, _) = placed.pairs[0];
    let fold = |a: [f64; 3]| {
        let l = N as f64 * A0;
        let mut d2 = 0.0;
        for ax in 0..3 {
            let mut d = vacancy[ax] - a[ax];
            d -= l * (d / l).round();
            d2 += d * d;
        }
        d2.sqrt()
    };
    let other = movers.iter().copied().find(|&m| m != moved).unwrap();
    assert!(fold(atoms[moved]) < fold(atoms[other]));
    for atom in 0..atoms.len() {
        if atom != moved {
            for ax in 0..3 {
                assert_eq!(placed.positions[3 * atom + ax], atoms[atom][ax]);
            }
        }
    }
}

#[test]
fn a_push_past_a_hop_saddle_lands_beyond_the_midpoint() {
    let sites = lattice();
    let reactant = flat(&sites);
    let mut saddle_rows = sites.clone();
    let hop = [A0 / 4.0, A0 / 4.0, 0.0];
    for ax in 0..3 {
        saddle_rows[1][ax] += hop[ax];
    }
    let saddle = flat(&saddle_rows);
    let pushed = push_past(reactant.view(), saddle.view(), cell(), [true; 3], 0.5).unwrap();
    for ax in 0..3 {
        let expect = sites[1][ax] + 1.5 * hop[ax];
        assert!((pushed[3 + ax] - expect).abs() < 1e-12);
    }
    assert_eq!(pushed[0], reactant[0]);
}
