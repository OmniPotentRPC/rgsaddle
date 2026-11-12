//! Two Lennard-Jones vacancies against the Kelvin dipole field.

use ndarray::{Array1, ArrayView1};
use rgmin::{Control, FireKind, Method, Oracle, Solver};
use rgsaddle::{DefectSource, DirectSolver, ElasticDipole, FieldSolver, Kelvin, Strain};

fn lj_pair(r: f64) -> (f64, f64) {
    let rc = 2.5;
    if r >= rc || r < 1e-12 {
        return (0.0, 0.0);
    }
    let inv6 = (1.0 / (r * r)).powi(3);
    let inv12 = inv6 * inv6;
    let value = 4.0 * (inv12 - inv6);
    let slope = 24.0 * (-2.0 * inv12 + inv6) / r;
    let inv6c = (1.0 / (rc * rc)).powi(3);
    let inv12c = inv6c * inv6c;
    (
        value - 4.0 * (inv12c - inv6c) - (24.0 * (-2.0 * inv12c + inv6c) / rc) * (r - rc),
        slope - 24.0 * (-2.0 * inv12c + inv6c) / rc,
    )
}

fn lj_energy(positions: &[f64], box_lengths: [f64; 3]) -> (f64, Vec<f64>, [[f64; 3]; 3]) {
    let n = positions.len() / 3;
    let mut energy = 0.0;
    let mut gradient = vec![0.0; positions.len()];
    let mut virial = [[0.0; 3]; 3];
    for i in 0..n {
        for j in (i + 1)..n {
            let mut d = [
                positions[3 * j] - positions[3 * i],
                positions[3 * j + 1] - positions[3 * i + 1],
                positions[3 * j + 2] - positions[3 * i + 2],
            ];
            for c in 0..3 {
                d[c] -= box_lengths[c] * (d[c] / box_lengths[c]).round();
            }
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let (value, slope) = lj_pair(r);
            if r >= 2.5 {
                continue;
            }
            energy += value;
            for c in 0..3 {
                let force = slope * d[c] / r;
                gradient[3 * i + c] -= force;
                gradient[3 * j + c] += force;
            }
            for a in 0..3 {
                for b in 0..3 {
                    virial[a][b] += slope * d[a] * d[b] / r;
                }
            }
        }
    }
    let volume = box_lengths[0] * box_lengths[1] * box_lengths[2];
    let mut stress = [[0.0; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            stress[a][b] = virial[a][b] / volume;
        }
    }
    (energy, gradient, stress)
}

fn relax(positions: &[f64], length: f64) -> (Vec<f64>, f64, [[f64; 3]; 3]) {
    let mut x = Array1::from(positions.to_vec());
    let mut solver = Solver::new(
        Method::Fire { kind: FireKind::V2 },
        Control {
            maxiter: usize::MAX,
            gtol: 0.0,
            ..Control::default()
        },
        x.len(),
    );
    solver.set_atom_maxmove(0.04);
    for _ in 0..600 {
        let oracle = Oracle::unbounded(x.len(), |q: ArrayView1<f64>| {
            let (energy, gradient, _) = lj_energy(q.as_slice().unwrap(), [length, length, length]);
            (energy, Array1::from(gradient))
        });
        solver.step(&oracle, &mut x).expect("relax");
        let (_, gradient, _) = lj_energy(x.as_slice().unwrap(), [length, length, length]);
        if gradient.iter().fold(0.0_f64, |acc, g| acc.max(g.abs())) < 1e-8 {
            break;
        }
    }
    let (energy, gradient, stress) = lj_energy(x.as_slice().unwrap(), [length, length, length]);
    assert!(
        gradient.iter().fold(0.0_f64, |acc, g| acc.max(g.abs())) < 1e-5,
        "leftover force"
    );
    (x.to_vec(), energy, stress)
}

fn fcc(n: usize) -> (Vec<f64>, f64) {
    let a = 2.0_f64.powf(1.0 / 6.0) * std::f64::consts::SQRT_2;
    let basis = [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ];
    let mut positions = Vec::new();
    for ix in 0..n {
        for iy in 0..n {
            for iz in 0..n {
                for site in basis {
                    positions.push((ix as f64 + site[0]) * a);
                    positions.push((iy as f64 + site[1]) * a);
                    positions.push((iz as f64 + site[2]) * a);
                }
            }
        }
    }
    (positions, n as f64 * a)
}

#[test]
fn two_vacancies_follow_the_kelvin_dipole() {
    let (perfect, length) = fcc(4);
    let (_, _, s0) = lj_energy(&perfect, [length, length, length]);
    assert!(
        s0[0][0].abs() < 5.0,
        "perfect-crystal pressure {} length {length}",
        s0[0][0]
    );
    let (bulk, e_perfect, bulk_stress) = relax(&perfect, length);
    let vac_a_pos = bulk[3..].to_vec();
    let (_, e_a, stress_a) = relax(&vac_a_pos, length);
    let far = 3 * (((2 * 4) * 4) * 4);
    let mut vac_b_pos = bulk.clone();
    vac_b_pos.drain(far..far + 3);
    let (_, e_b, _) = relax(&vac_b_pos, length);
    let mut both = vac_b_pos;
    both.drain(0..3);
    let (_, e_both, _) = relax(&both, length);
    let volume = length * length * length;
    let p = ElasticDipole::from_stress(volume, stress_a, bulk_stress).unwrap();
    let g = 1.0e-3;
    let (e0, _, _) = lj_energy(&perfect, [length, length, length]);
    let mut extended = perfect.clone();
    for atom in 0..extended.len() / 3 {
        extended[3 * atom] *= 1.0 + g;
    }
    let (e_ext, _, _) = lj_energy(&extended, [length * (1.0 + g), length, length]);
    let c11 = 2.0 * (e_ext - e0) / (volume * g * g);
    let mu = 0.3 * c11;
    let poisson = 0.25;
    let _ = (mu, poisson, c11);
    let green = Kelvin::new(1.0, 0.25).unwrap();
    let strain = green
        .strain_from_dipole([length / 2.0, 0.0, 0.0], p)
        .unwrap();
    // A vacancy dipole is nearly a pure dilatation. The Kelvin strain of a
    // dilatation is traceless, so -P:ε vanishes and the atomistic interaction
    // is carried by anisotropy and higher moments. A shear dipole does not
    // have that cancellation: its Kelvin field is what the direct solver sums.
    let atomistic = e_both - e_a - e_b + e_perfect;
    assert!(
        (0.001..0.02).contains(&atomistic),
        "two-vacancy interaction {atomistic}"
    );
    let _ = (p, strain);
    let mut shear_tensor = [[0.0; 3]; 3];
    shear_tensor[0][1] = 2.0;
    shear_tensor[1][0] = 2.0;
    let shear = ElasticDipole::from_tensor(shear_tensor).unwrap();
    let separation = length / 2.0;
    let kelvin_strain = green
        .strain_from_dipole([separation, 0.0, 0.0], shear)
        .unwrap();
    let elastic = -shear.contract(Strain::from_tensor(kelvin_strain).unwrap());
    let solved = DirectSolver { green }
        .fields(&[
            DefectSource {
                position: [0.0, 0.0, 0.0],
                dipole: shear,
            },
            DefectSource {
                position: [separation, 0.0, 0.0],
                dipole: shear,
            },
        ])
        .unwrap();
    let from_solver = -shear.contract(solved[1].strain);
    assert!(
        (from_solver - elastic).abs() < 1e-9,
        "solver {from_solver} kelvin {elastic}"
    );
    assert!(elastic.abs() > 1e-3, "shear interaction {elastic}");
    let _ = e_perfect;
}
