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
    for (atom, row) in atoms.iter().enumerate() {
        if atom != moved {
            for (ax, &value) in row.iter().enumerate() {
                assert_eq!(placed.positions[3 * atom + ax], value);
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
    for (value, step) in saddle_rows[1].iter_mut().zip(hop) {
        *value += step;
    }
    let saddle = flat(&saddle_rows);
    let pushed = push_past(reactant.view(), saddle.view(), cell(), [true; 3], 0.5).unwrap();
    for ax in 0..3 {
        let expect = sites[1][ax] + 1.5 * hop[ax];
        assert!((pushed[3 + ax] - expect).abs() < 1e-12);
    }
    assert_eq!(pushed[0], reactant[0]);
}

/// Splitmix64 uniform draws in `[-1, 1)`.
struct Draws(u64);

impl Draws {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        2.0 * (z >> 11) as f64 / (1u64 << 53) as f64 - 1.0
    }
}

fn combine(f: [f64; 3], cell: [[f64; 3]; 3]) -> [f64; 3] {
    let mut d = [0.0; 3];
    for (c, value) in d.iter_mut().enumerate() {
        *value = f[0] * cell[0][c] + f[1] * cell[1][c] + f[2] * cell[2][c];
    }
    d
}

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// The shortest of `d - n` over every lattice vector `n` with whole
/// coefficients up to `reach` along the periodic axes.
fn brute_force(d: [f64; 3], cell: [[f64; 3]; 3], periodic: [bool; 3], reach: i64) -> [f64; 3] {
    let span = |ax: usize| if periodic[ax] { reach } else { 0 };
    let mut best = d;
    for i in -span(0)..=span(0) {
        for j in -span(1)..=span(1) {
            for k in -span(2)..=span(2) {
                let n = combine([i as f64, j as f64, k as f64], cell);
                let v = [d[0] - n[0], d[1] - n[1], d[2] - n[2]];
                if norm(v) < norm(best) {
                    best = v;
                }
            }
        }
    }
    best
}

/// The fold `translate_onto_sites` applies to `d`, read off an atom at the
/// origin moved onto a site at `d`.
fn fold(d: [f64; 3], cell: [[f64; 3]; 3], periodic: [bool; 3]) -> [f64; 3] {
    let out = translate_onto_sites(
        flat(&[[0.0; 3]]).view(),
        cell,
        periodic,
        &[0],
        flat(&[d]).view(),
    )
    .unwrap();
    [out.positions[0], out.positions[1], out.positions[2]]
}

fn check_random_displacements(cell: [[f64; 3]; 3], periodic: [bool; 3], seed: u64) {
    let mut draws = Draws(seed);
    let scale = cell.iter().map(|&row| norm(row)).fold(0.0, f64::max);
    for sample in 0..3000 {
        let f = [draws.next(), draws.next(), draws.next()];
        let d = combine(f, cell);
        let got = fold(d, cell, periodic);
        let want = brute_force(d, cell, periodic, 6);
        assert!(
            (norm(got) - norm(want)).abs() <= 1e-9 * scale,
            "sample {sample}: |fold| {} but the nearest image is {} away",
            norm(got),
            norm(want)
        );
        for ax in 0..3 {
            assert!(
                (got[ax] - want[ax]).abs() <= 1e-9 * scale,
                "sample {sample}: fold {got:?}, nearest image {want:?}"
            );
        }
    }
}

const SHEARED: [[f64; 3]; 3] = [[10.0, 0.0, 0.0], [3.0, 10.0, 0.0], [0.0, 0.0, 10.0]];

fn fcc_primitive() -> [[f64; 3]; 3] {
    let h = A0 / 2.0;
    [[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]
}

fn hexagonal(a: f64, c: f64) -> [[f64; 3]; 3] {
    [
        [a, 0.0, 0.0],
        [-a / 2.0, a * 3f64.sqrt() / 2.0, 0.0],
        [0.0, 0.0, c],
    ]
}

/// A triclinic cell whose nearest image often lies two steps past the
/// rounded fractions, beyond a search of the next image only.
const SKEWED: [[f64; 3]; 3] = [[8.0, 0.0, 0.0], [13.0, 3.0, 0.0], [6.0, 5.0, 4.0]];

#[test]
fn a_sheared_cell_folds_to_the_nearest_image() {
    check_random_displacements(SHEARED, [true; 3], 1);
}

#[test]
fn an_fcc_primitive_cell_folds_to_the_nearest_image() {
    check_random_displacements(fcc_primitive(), [true; 3], 2);
}

#[test]
fn a_hexagonal_cell_folds_to_the_nearest_image() {
    check_random_displacements(hexagonal(2.95, 4.68), [true; 3], 3);
}

#[test]
fn a_skewed_cell_folds_to_the_nearest_image() {
    check_random_displacements(SKEWED, [true; 3], 4);
}

#[test]
fn a_hexagonal_displacement_takes_the_image_rounding_misses() {
    // Rounding the fractions (0.423, -0.450) leaves the displacement itself,
    // 6.58 Angstrom long. One whole a + b, (4.35, 7.53, 0), is closer.
    let cell = hexagonal(8.7, 10.0);
    let d = [5.64, -3.39, 0.0];
    assert!((norm(d) - 6.58).abs() < 5e-3);
    let got = fold(d, cell, [true; 3]);
    assert!((norm(got) - 4.34).abs() < 5e-3, "{}", norm(got));
    let want = brute_force(d, cell, [true; 3], 6);
    for ax in 0..3 {
        assert!((got[ax] - want[ax]).abs() < 1e-12);
    }
    assert!((got[0] - (5.64 - 4.35)).abs() < 1e-12);
}
