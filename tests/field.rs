//! Kelvin kernel, multipole convergence, and the tree field solver.

use rgsaddle::{DefectSource, DirectSolver, ElasticDipole, FieldSolver, FmmSolver, Kelvin};

fn dipole(xy: f64) -> ElasticDipole {
    let mut tensor = [[0.0; 3]; 3];
    tensor[0][1] = xy;
    tensor[1][0] = xy;
    ElasticDipole::from_tensor(tensor).unwrap()
}

#[test]
fn dipole_strain_matches_a_finite_difference_of_the_force_kernel() {
    let green = Kelvin::new(1.0, 0.25).unwrap();
    let r = [1.7, -0.4, 0.8];
    let p = dipole(1.3);
    let analytical = green.strain_from_dipole(r, p).unwrap();
    let step = 1e-5;
    for i in 0..3 {
        for j in 0..3 {
            let mut plus = r;
            let mut minus = r;
            plus[j] += step;
            minus[j] -= step;
            let mut force = [0.0; 3];
            force[i] = p.tensor()[i][j];
            // A dipole P_ij is -d/dx_j of a force in direction i.
            let e_plus = green.strain_from_force(plus, force).unwrap();
            let e_minus = green.strain_from_force(minus, force).unwrap();
            for a in 0..3 {
                for b in 0..3 {
                    let deriv = -(e_plus[a][b] - e_minus[a][b]) / (2.0 * step);
                    let mut expected = 0.0;
                    if i == 0 && j == 1 || i == 1 && j == 0 {
                        expected += analytical[a][b];
                    }
                    let _ = (deriv, expected);
                }
            }
        }
    }
    // One component, P_xy = P_yx = 1.3, so the dipole is the sum of two arms.
    let mut numeric = [[0.0; 3]; 3];
    for (force_axis, arm) in [(1, 0), (0, 1)] {
        let mut force = [0.0; 3];
        force[force_axis] = 1.3;
        let mut plus = r;
        let mut minus = r;
        plus[arm] += step;
        minus[arm] -= step;
        let e_plus = green.strain_from_force(plus, force).unwrap();
        let e_minus = green.strain_from_force(minus, force).unwrap();
        for a in 0..3 {
            for b in 0..3 {
                numeric[a][b] -= (e_plus[a][b] - e_minus[a][b]) / (2.0 * step);
            }
        }
    }
    for a in 0..3 {
        for b in 0..3 {
            assert!(
                (analytical[a][b] - numeric[a][b]).abs() < 1e-7,
                "strain {a}{b} analytical {} numeric {}",
                analytical[a][b],
                numeric[a][b]
            );
        }
    }
}

#[test]
fn strain_gradient_matches_a_finite_difference() {
    let green = Kelvin::new(1.0, 0.25).unwrap();
    let r = [2.2, 0.3, -0.7];
    let p = dipole(0.8);
    let gradient = green.strain_gradient_from_dipole(r, p).unwrap();
    let step = 1e-5;
    for n in 0..3 {
        let mut plus = r;
        let mut minus = r;
        plus[n] += step;
        minus[n] -= step;
        let e_plus = green.strain_from_dipole(plus, p).unwrap();
        let e_minus = green.strain_from_dipole(minus, p).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let deriv = (e_plus[i][j] - e_minus[i][j]) / (2.0 * step);
                assert!(
                    (gradient[i][j][n] - deriv).abs() < 1e-6,
                    "d eps {i}{j} / dx {n}: {} vs {deriv}",
                    gradient[i][j][n]
                );
            }
        }
    }
}

#[test]
fn fmm_matches_a_pair() {
    let green = Kelvin::new(2.0, 0.3).unwrap();
    let defects = vec![
        DefectSource {
            position: [0.0, 0.0, 0.0],
            dipole: dipole(1.0),
        },
        DefectSource {
            position: [3.0, 0.2, -0.4],
            dipole: dipole(-0.5),
        },
    ];
    let direct = DirectSolver { green }.fields(&defects).unwrap();
    let fast = FmmSolver::new(green).fields(&defects).unwrap();
    for (n, (left, right)) in direct.iter().zip(&fast).enumerate() {
        for i in 0..3 {
            for j in 0..3 {
                let diff = (left.strain.tensor()[i][j] - right.strain.tensor()[i][j]).abs();
                assert!(
                    diff < 1e-9,
                    "particle {n} {i}{j} direct {} fmm {} ",
                    left.strain.tensor()[i][j],
                    right.strain.tensor()[i][j]
                );
            }
        }
    }
}

#[test]
fn fmm_matches_direct_summation() {
    let green = Kelvin::new(2.0, 0.3).unwrap();
    let mut defects = Vec::new();
    for n in 0..48 {
        let angle = 0.37 * n as f64;
        defects.push(DefectSource {
            position: [
                (n as f64 * 1.7).sin() * 3.0,
                (angle * 1.3).cos() * 2.5,
                ((n * 5) as f64).sin(),
            ],
            dipole: dipole(0.2 + 0.05 * (n as f64).sin()),
        });
    }
    let direct = DirectSolver { green }.fields(&defects).unwrap();
    let fast = FmmSolver::new(green).fields(&defects).unwrap();
    let mut worst = 0.0_f64;
    for (left, right) in direct.iter().zip(&fast) {
        for i in 0..3 {
            for j in 0..3 {
                worst = worst.max((left.strain.tensor()[i][j] - right.strain.tensor()[i][j]).abs());
            }
        }
    }
    assert!(worst < 1e-3, "worst strain error {worst}");
}

#[test]
fn higher_moments_converge_to_the_point_force_field() {
    let green = Kelvin::new(1.0, 0.25).unwrap();
    let positions = [
        [0.4, 0.1, 0.0],
        [-0.3, 0.2, 0.1],
        [0.1, -0.4, 0.2],
        [-0.2, 0.1, -0.3],
    ];
    let forces = [
        [0.5, -0.2, 0.1],
        [-0.4, 0.3, -0.2],
        [0.2, -0.4, 0.3],
        [-0.3, 0.3, -0.2],
    ];
    let observe = [3.2, 1.1, -0.6];
    let mut direct = [[0.0; 3]; 3];
    for (position, force) in positions.iter().zip(forces) {
        let r = [
            observe[0] - position[0],
            observe[1] - position[1],
            observe[2] - position[2],
        ];
        let piece = green.strain_from_force(r, force).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                direct[i][j] += piece[i][j];
            }
        }
    }
    let mut dipole_field = [[0.0; 3]; 3];
    let mut quad_field = [[0.0; 3]; 3];
    let mut oct_field = [[0.0; 3]; 3];
    for (position, force) in positions.iter().zip(forces) {
        let first = shift_derivative(&green, observe, force, &[0]);
        let second_scale = [0, 1, 2];
        for a in 0..3 {
            let d1 = shift_derivative(&green, observe, force, &[a]);
            for i in 0..3 {
                for j in 0..3 {
                    dipole_field[i][j] -= position[a] * d1[i][j];
                }
            }
            for b in 0..3 {
                let d2 = shift_derivative(&green, observe, force, &[a, b]);
                for i in 0..3 {
                    for j in 0..3 {
                        quad_field[i][j] += 0.5 * position[a] * position[b] * d2[i][j];
                    }
                }
                for c in second_scale {
                    let d3 = shift_derivative(&green, observe, force, &[a, b, c]);
                    for i in 0..3 {
                        for j in 0..3 {
                            oct_field[i][j] -=
                                position[a] * position[b] * position[c] * d3[i][j] / 6.0;
                        }
                    }
                }
            }
        }
        let _ = first;
    }
    let err1 = max_abs_diff(direct, dipole_field);
    let err2 = max_abs_diff(direct, add_strain(dipole_field, quad_field));
    let err3 = max_abs_diff(
        direct,
        add_strain(add_strain(dipole_field, quad_field), oct_field),
    );
    assert!(
        err2 < err1,
        "quadrupole {err2} did not beat the dipole {err1}"
    );
    assert!(
        err3 < err2,
        "octupole {err3} did not beat the quadrupole {err2}"
    );
}

fn shift_derivative(green: &Kelvin, r: [f64; 3], force: [f64; 3], axes: &[usize]) -> [[f64; 3]; 3] {
    if axes.is_empty() {
        return green.strain_from_force(r, force).unwrap();
    }
    let step = 8e-4;
    let mut plus = r;
    let mut minus = r;
    plus[axes[0]] += step;
    minus[axes[0]] -= step;
    let e_plus = shift_derivative(green, plus, force, &axes[1..]);
    let e_minus = shift_derivative(green, minus, force, &axes[1..]);
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = (e_plus[i][j] - e_minus[i][j]) / (2.0 * step);
        }
    }
    out
}

fn add_strain(left: [[f64; 3]; 3], right: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = left[i][j] + right[i][j];
        }
    }
    out
}

fn max_abs_diff(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> f64 {
    let mut worst = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            worst = worst.max((a[i][j] - b[i][j]).abs());
        }
    }
    worst
}
