//! Dipole finite differences, symmetry images, trust, and JSON.

use std::path::PathBuf;

use ndarray::{Array1, Array2, ArrayView1, array};
use rgmin::{Control, FireKind, Method, Oracle, Solver};
use rgsaddle::minmode::PointSurface;
use rgsaddle::{
    CurvatureBound, ElasticDipole, LINEAR_FOLD_EV, SaddleError, SaddleSeed, SearchReason, Strain,
    StrainResponse, Trust, TrustPolicy, activation_volume, compliance_from_stiffness,
    cubic_rotations, rotation_x, split_translational_modes, vineyard_prefactor,
};

fn det(h: [[f64; 3]; 3]) -> f64 {
    h[0][0] * (h[1][1] * h[2][2] - h[1][2] * h[2][1])
        - h[0][1] * (h[1][0] * h[2][2] - h[1][2] * h[2][0])
        + h[0][2] * (h[1][0] * h[2][1] - h[1][1] * h[2][0])
}

fn apply(
    positions: &Array1<f64>,
    cell: [[f64; 3]; 3],
    eps: [[f64; 3]; 3],
) -> (Array1<f64>, [[f64; 3]; 3]) {
    let mut deform = eps;
    for i in 0..3 {
        deform[i][i] += 1.0;
    }
    let mut h = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            h[i][j] =
                cell[i][0] * deform[0][j] + cell[i][1] * deform[1][j] + cell[i][2] * deform[2][j];
        }
    }
    let mut out = Array1::zeros(positions.len());
    for atom in 0..positions.len() / 3 {
        let row = [
            positions[3 * atom],
            positions[3 * atom + 1],
            positions[3 * atom + 2],
        ];
        for c in 0..3 {
            out[3 * atom + c] =
                row[0] * deform[0][c] + row[1] * deform[1][c] + row[2] * deform[2][c];
        }
    }
    (out, h)
}

fn lj_smooth(r: f64) -> (f64, f64) {
    let rc = 2.5;
    if r >= rc || r < 1e-12 {
        return (0.0, 0.0);
    }
    let inv = 1.0 / r;
    let inv2 = inv * inv;
    let inv6 = inv2 * inv2 * inv2;
    let inv12 = inv6 * inv6;
    let value = 4.0 * (inv12 - inv6);
    let slope = 4.0 * (-12.0 * inv12 + 6.0 * inv6) * inv;
    let (rc_v, rc_s) = {
        let inv = 1.0 / rc;
        let inv2 = inv * inv;
        let inv6 = inv2 * inv2 * inv2;
        let inv12 = inv6 * inv6;
        (
            4.0 * (inv12 - inv6),
            4.0 * (-12.0 * inv12 + 6.0 * inv6) * inv,
        )
    };
    (value - rc_v - rc_s * (r - rc), slope - rc_s)
}

fn mic(d: [f64; 3], cell: [[f64; 3]; 3]) -> [f64; 3] {
    let volume = det(cell);
    let mut inv = [[0.0; 3]; 3];
    inv[0][0] = (cell[1][1] * cell[2][2] - cell[1][2] * cell[2][1]) / volume;
    inv[0][1] = (cell[0][2] * cell[2][1] - cell[0][1] * cell[2][2]) / volume;
    inv[0][2] = (cell[0][1] * cell[1][2] - cell[0][2] * cell[1][1]) / volume;
    inv[1][0] = (cell[1][2] * cell[2][0] - cell[1][0] * cell[2][2]) / volume;
    inv[1][1] = (cell[0][0] * cell[2][2] - cell[0][2] * cell[2][0]) / volume;
    inv[1][2] = (cell[0][2] * cell[1][0] - cell[0][0] * cell[1][2]) / volume;
    inv[2][0] = (cell[1][0] * cell[2][1] - cell[1][1] * cell[2][0]) / volume;
    inv[2][1] = (cell[0][1] * cell[2][0] - cell[0][0] * cell[2][1]) / volume;
    inv[2][2] = (cell[0][0] * cell[1][1] - cell[0][1] * cell[1][0]) / volume;
    let s = [
        d[0] * inv[0][0] + d[1] * inv[1][0] + d[2] * inv[2][0],
        d[0] * inv[0][1] + d[1] * inv[1][1] + d[2] * inv[2][1],
        d[0] * inv[0][2] + d[1] * inv[1][2] + d[2] * inv[2][2],
    ];
    let w = [
        s[0] - s[0].round(),
        s[1] - s[1].round(),
        s[2] - s[2].round(),
    ];
    [
        w[0] * cell[0][0] + w[1] * cell[1][0] + w[2] * cell[2][0],
        w[0] * cell[0][1] + w[1] * cell[1][1] + w[2] * cell[2][1],
        w[0] * cell[0][2] + w[1] * cell[1][2] + w[2] * cell[2][2],
    ]
}

fn lj_cell(positions: &Array1<f64>, cell: [[f64; 3]; 3]) -> (f64, Array1<f64>, [[f64; 3]; 3]) {
    let n = positions.len() / 3;
    let mut energy = 0.0;
    let mut gradient = Array1::zeros(positions.len());
    let mut virial = [[0.0; 3]; 3];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = mic(
                [
                    positions[3 * j] - positions[3 * i],
                    positions[3 * j + 1] - positions[3 * i + 1],
                    positions[3 * j + 2] - positions[3 * i + 2],
                ],
                cell,
            );
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let (value, slope) = lj_smooth(r);
            if r >= 2.5 {
                continue;
            }
            energy += value;
            let inv_r = 1.0 / r;
            for c in 0..3 {
                let force = slope * d[c] * inv_r;
                gradient[3 * i + c] -= force;
                gradient[3 * j + c] += force;
            }
            for a in 0..3 {
                for b in 0..3 {
                    virial[a][b] += slope * d[a] * d[b] * inv_r;
                }
            }
        }
    }
    let volume = det(cell).abs();
    let mut stress = [[0.0; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            stress[a][b] = virial[a][b] / volume;
        }
    }
    (energy, gradient, stress)
}

fn descend(
    positions: &Array1<f64>,
    cell: [[f64; 3]; 3],
    steps: usize,
) -> (Array1<f64>, f64, [[f64; 3]; 3]) {
    let mut x = positions.clone();
    for _ in 0..steps {
        let (_, gradient, _) = lj_cell(&x, cell);
        let mut max_g = 0.0_f64;
        for g in gradient.iter() {
            max_g = max_g.max(g.abs());
        }
        if max_g < 1e-10 {
            break;
        }
        let step = 0.002_f64.min(0.02 / max_g.max(1e-8));
        for (coord, g) in x.iter_mut().zip(gradient.iter()) {
            *coord -= step * g;
        }
    }
    let (energy, _, stress) = lj_cell(&x, cell);
    (x, energy, stress)
}

fn relax_lj(positions: &Array1<f64>, cell: [[f64; 3]; 3]) -> (Array1<f64>, f64, [[f64; 3]; 3]) {
    let mut x = positions.clone();
    let mut solver = Solver::new(
        Method::Fire { kind: FireKind::V2 },
        Control {
            maxiter: usize::MAX,
            gtol: 0.0,
            ..Control::default()
        },
        x.len(),
    );
    solver.set_atom_maxmove(0.05);
    for _ in 0..400 {
        let oracle = Oracle::unbounded(x.len(), |q: ArrayView1<f64>| {
            let (energy, gradient, _) = lj_cell(&q.to_owned(), cell);
            (energy, gradient)
        });
        solver.step(&oracle, &mut x).expect("relax");
        let (_, gradient, _) = lj_cell(&x, cell);
        if gradient.iter().map(|v| v.abs()).fold(0.0, f64::max) < 1e-8 {
            break;
        }
    }
    let (energy, gradient, stress) = lj_cell(&x, cell);
    assert!(
        gradient.iter().map(|v| v.abs()).fold(0.0, f64::max) < 1e-6,
        "lj relax leftover force"
    );
    (x, energy, stress)
}

fn fcc_lj() -> (Array1<f64>, [[f64; 3]; 3]) {
    let a = 2.0_f64.powf(1.0 / 6.0) * std::f64::consts::SQRT_2;
    let basis = [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ];
    let n = 2;
    let mut values = Vec::new();
    for ix in 0..n {
        for iy in 0..n {
            for iz in 0..n {
                for site in basis {
                    values.push((ix as f64 + site[0]) * a);
                    values.push((iy as f64 + site[1]) * a);
                    values.push((iz as f64 + site[2]) * a);
                }
            }
        }
    }
    let length = n as f64 * a;
    (
        Array1::from(values),
        [[length, 0.0, 0.0], [0.0, length, 0.0], [0.0, 0.0, length]],
    )
}

#[test]
fn dipole_matches_the_strain_derivative_and_is_symmetric() {
    let (perfect, cell) = fcc_lj();
    let (bulk_pos, _, bulk_stress) = relax_lj(&perfect, cell);
    let vacancy = perfect.slice(ndarray::s![3..]).to_owned();
    let (defect_pos, _, defect_stress) = relax_lj(&vacancy, cell);
    let volume = det(cell).abs();
    let dipole = ElasticDipole::from_stress(volume, defect_stress, bulk_stress).unwrap();
    let tensor = dipole.tensor();
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (tensor[i][j] - tensor[j][i]).abs() < 1e-12,
                "P not symmetric at {i}{j}"
            );
        }
    }
    let step = 1e-5;
    let probes = [
        [[step, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, step, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, step]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, step], [0.0, step, 0.0]],
        [[0.0, 0.0, step], [0.0, 0.0, 0.0], [step, 0.0, 0.0]],
        [[0.0, step, 0.0], [step, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ];
    for eps in probes {
        let frozen = |positions: &Array1<f64>, sign: f64| {
            let (moved, strained) = apply(positions, cell, scale_eps(eps, sign));
            lj_cell(&moved, strained).0
        };
        let deriv = ((frozen(&defect_pos, 1.0) - frozen(&bulk_pos, 1.0))
            - (frozen(&defect_pos, -1.0) - frozen(&bulk_pos, -1.0)))
            / 2.0;
        let predicted = -dipole.contract(Strain::from_tensor(eps).unwrap());
        assert!(
            (deriv - predicted).abs() < 1e-6,
            "frozen excess derivative {deriv} vs -P:eps {predicted}"
        );
        let relaxed = |positions: &Array1<f64>, sign: f64| {
            let (moved, strained) = apply(positions, cell, scale_eps(eps, sign));
            descend(&moved, strained, 40).1
        };
        let deriv_relaxed = ((relaxed(&defect_pos, 1.0) - relaxed(&bulk_pos, 1.0))
            - (relaxed(&defect_pos, -1.0) - relaxed(&bulk_pos, -1.0)))
            / 2.0;
        assert!(
            (deriv_relaxed - predicted).abs() < 1e-5,
            "relaxed excess derivative {deriv_relaxed} vs -P:eps {predicted}"
        );
    }
}

fn scale_eps(eps: [[f64; 3]; 3], scale: f64) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = scale * eps[i][j];
        }
    }
    out
}

fn well(u: f64, gamma: f64) -> f64 {
    let gap = u * u - 1.0;
    gap * gap + 6.0 * (gamma - 0.2 * u).powi(2)
}

fn well_force(u: f64, gamma: f64) -> (f64, f64) {
    let du = 4.0 * u * (u * u - 1.0) - 2.4 * (gamma - 0.2 * u);
    let dg = 12.0 * (gamma - 0.2 * u);
    (du, dg)
}

fn newton_u(gamma: f64, start: f64) -> f64 {
    let mut u = start;
    for _ in 0..40 {
        let (du, _) = well_force(u, gamma);
        let (du_plus, _) = well_force(u + 1e-6, gamma);
        let slope = (du_plus - du) / 1e-6;
        if slope.abs() < 1e-12 {
            break;
        }
        u -= du / slope;
    }
    u
}

#[test]
fn linear_model_matches_the_explicit_barrier_at_small_strain() {
    let u_min = newton_u(0.0, 1.0);
    let u_sad = newton_u(0.0, 0.0);
    let (_, g_min) = well_force(u_min, 0.0);
    let (_, g_sad) = well_force(u_sad, 0.0);
    let stress_of = |g: f64| [[0.0, g, 0.0], [g, 0.0, 0.0], [0.0, 0.0, 0.0]];
    let minimum = ElasticDipole::from_stress(1.0, stress_of(g_min), [[0.0; 3]; 3]).unwrap();
    let saddle = ElasticDipole::from_stress(1.0, stress_of(g_sad), [[0.0; 3]; 3]).unwrap();
    let barrier0 = well(u_sad, 0.0) - well(u_min, 0.0);
    let response = StrainResponse::new(barrier0, minimum, saddle).unwrap();
    let gamma = 1.0e-5;
    let explicit = well(newton_u(gamma, u_sad), gamma) - well(newton_u(gamma, u_min), gamma);
    let estimate = response.barrier(&Strain::shear_yx(gamma));
    assert!(matches!(estimate.trust, Trust::Linear));
    assert!(
        (estimate.barrier - explicit).abs() < 1e-8,
        "linear {} explicit {explicit}",
        estimate.barrier
    );
}

#[test]
fn rotated_variant_matches_the_symmetry_related_shear() {
    let mut saddle = [[0.0; 3]; 3];
    saddle[0][1] = 12.85;
    saddle[1][0] = 12.85;
    let response = StrainResponse::new(
        0.5,
        ElasticDipole::from_tensor([[0.0; 3]; 3]).unwrap(),
        ElasticDipole::from_tensor(saddle).unwrap(),
    )
    .unwrap()
    .with_point_group(&[rotation_x(0), rotation_x(1), rotation_x(2), rotation_x(3)])
    .unwrap();
    let yx = response.barrier(&Strain::shear_yx(0.02));
    let xz = response.barrier(&Strain::shear_xz(0.02));
    assert!((yx.barrier - xz.barrier).abs() < 1e-12);
    assert_ne!(yx.variant, xz.variant);
    assert!(cubic_rotations().len() == 24);
}

#[test]
fn copper_fixture_needs_a_search_past_the_fold() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks/strain_response_cu.json");
    let text = std::fs::read_to_string(path).unwrap();
    let response = StrainResponse::from_json(&text)
        .unwrap()
        .with_point_group(&cubic_rotations())
        .unwrap();
    let small = response.barrier(&Strain::shear_yx(0.005));
    assert!(matches!(small.trust, Trust::Linear), "{:?}", small.trust);
    let folded = response.barrier(&Strain::shear_yx(0.02));
    match folded.trust {
        Trust::NeedsSearch(SearchReason::Quadratic {
            estimated_error,
            tolerance,
        }) => {
            assert!((tolerance - LINEAR_FOLD_EV).abs() < 1e-15);
            assert!(estimated_error > LINEAR_FOLD_EV);
        }
        other => panic!("expected the quadratic fold, got {other:?}"),
    }
    let xz = response.barrier(&Strain::shear_xz(0.02));
    assert!((folded.barrier - xz.barrier).abs() < 1e-9);
    assert!(
        (folded.barrier - 0.21690853542407557).abs() < 1e-9,
        "{}",
        folded.barrier
    );
}

#[test]
fn json_round_trip_keeps_the_hop_and_the_bound() {
    let response = StrainResponse::new(
        0.4,
        ElasticDipole::from_tensor([[1.0, 0.2, 0.0], [0.2, 0.0, 0.0], [0.0, 0.0, 0.0]]).unwrap(),
        ElasticDipole::from_tensor([[1.5, 0.4, 0.0], [0.4, 0.1, 0.0], [0.0, 0.0, 0.0]]).unwrap(),
    )
    .unwrap()
    .with_prefactor(1.0e12, None)
    .unwrap()
    .with_policy(TrustPolicy {
        radius: Some(0.03),
        quadratic: None,
        error_tolerance: LINEAR_FOLD_EV,
        curvature: Some(CurvatureBound {
            lambda0: -0.2,
            slope: [[0.0, 4.0, 0.0], [4.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            fraction: 0.5,
        }),
        floor: 0.0,
    })
    .unwrap();
    let again = StrainResponse::from_json(&response.to_json()).unwrap();
    let strain = Strain::shear_yx(0.04);
    let left = response.barrier(&strain);
    let right = again.barrier(&strain);
    assert!((left.barrier - right.barrier).abs() < 1e-12);
    assert_eq!(left.prefactor_hz, right.prefactor_hz);
    assert!(matches!(right.trust, Trust::NeedsSearch(_)));
}

struct Well;

impl PointSurface for Well {
    fn eval(&self, r: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (x, y, z) = (r[0], r[1], r[2]);
        let energy = (x * x - 1.0).powi(2) + 2.0 * y * y + 2.0 * z * z;
        Ok((energy, array![4.0 * x * (x * x - 1.0), 4.0 * y, 4.0 * z]))
    }
}

#[test]
fn refine_caches_an_index_one_saddle() {
    let response = StrainResponse::new(
        1.0,
        ElasticDipole::from_tensor([[0.0; 3]; 3]).unwrap(),
        ElasticDipole::from_tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).unwrap(),
    )
    .unwrap()
    .with_policy(TrustPolicy {
        radius: Some(0.01),
        ..TrustPolicy::default()
    })
    .unwrap();
    let strain = Strain::shear_yx(0.05);
    assert!(matches!(
        response.barrier(&strain).trust,
        Trust::NeedsSearch(_)
    ));
    let mut response = response;
    let estimate = response
        .refine(
            &strain,
            &Well,
            &SaddleSeed {
                positions: array![0.4, 0.1, -0.05],
                cell: [[6.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 7.0]],
                mode: array![1.0, 0.2, -0.1],
                minimum_energy: 0.0,
            },
            80,
        )
        .unwrap();
    assert!(matches!(estimate.trust, Trust::Explicit));
    assert!(
        (estimate.barrier - 1.0).abs() < 1e-6,
        "{}",
        estimate.barrier
    );
    assert!(matches!(response.barrier(&strain).trust, Trust::Explicit));
}

#[test]
fn activation_volume_is_the_stress_derivative() {
    let minimum = [[1.0, 0.2, 0.0], [0.2, -0.5, 0.1], [0.0, 0.1, 0.3]];
    let saddle = [[1.4, -0.1, 0.2], [-0.1, 0.2, 0.0], [0.2, 0.0, -0.4]];
    let mut identity = [[0.0; 6]; 6];
    for i in 0..6 {
        identity[i][i] = 1.0;
    }
    let omega = activation_volume(minimum, saddle, &identity).unwrap();
    let response = StrainResponse::new(
        0.4,
        ElasticDipole::from_tensor(minimum).unwrap(),
        ElasticDipole::from_tensor(saddle).unwrap(),
    )
    .unwrap();
    let step = 1e-6;
    let finite = (response.barrier(&Strain::uniaxial_x(step)).barrier - response.barrier0()) / step;
    assert!((finite - omega[0][0]).abs() < 1e-8);
    let mut stiffness = identity;
    stiffness[0][0] = 2.0;
    let compliance = compliance_from_stiffness(&stiffness).unwrap();
    assert!((compliance[0][0] - 0.5).abs() < 1e-12);
    let omega = activation_volume(minimum, saddle, &compliance).unwrap();
    let finite = (response
        .barrier(&Strain::uniaxial_x(compliance[0][0] * step))
        .barrier
        - response.barrier0())
        / step;
    assert!((finite - omega[0][0]).abs() < 1e-8);
}

#[test]
fn vineyard_prefactor_has_one_factor_of_frequency() {
    let mass = 4.0;
    let scale = rgsaddle::angular_scale_rad_s();
    let expected = scale * (1.0_f64 / mass).sqrt() / (2.0 * std::f64::consts::PI);
    let got = vineyard_prefactor(&[1.0], &[], mass).unwrap();
    assert!((got - expected).abs() / expected < 1e-12);
}

#[test]
fn translations_are_left_out_of_the_prefactor() {
    let n = 6;
    let mut vectors = Array2::zeros((n, n));
    for i in 0..n {
        vectors[(i, i)] = 1.0;
    }
    let mut translations = Array2::zeros((1, n));
    translations[(0, 0)] = 1.0;
    let values = [1e-12, -0.2, 0.5, 1.5, 2.0, 3.0];
    let split = split_translational_modes(&values, vectors.view(), translations.view()).unwrap();
    assert_eq!(split.translations, 1);
    assert_eq!(split.negative, vec![-0.2]);
    assert_eq!(split.positive, vec![0.5, 1.5, 2.0, 3.0]);
}

#[test]
fn force_dipole_matches_the_first_moment() {
    let positions = [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
    ];
    let forces = [
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
    ];
    let dipole = ElasticDipole::from_forces(&positions, &forces).unwrap();
    let tensor = dipole.tensor();
    assert!((tensor[0][1] - 2.0).abs() < 1e-12);
    assert!((tensor[1][0] - 2.0).abs() < 1e-12);
}
