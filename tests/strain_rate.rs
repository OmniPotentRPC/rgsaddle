//! Consumer of the strain-rate table: dipole finite differences, the
//! linear barrier, and lookup of a recorded no-barrier row.

use std::path::PathBuf;

use ndarray::{Array1, Array2, ArrayView1};
use rgmin::{Control, FireKind, Method, Oracle, Solver};
use rgsaddle::{
    Ensemble, ExplicitSide, HopVariant, LINEAR_FOLD_EV, LinearStatus, Rate, SaddleError,
    StrainPoint, StrainRateTable, activation_volume, compliance_from_stiffness, double_dot,
    elastic_dipole, hydrostatic, judge_linear, rotation_x, shear_xz, shear_yx,
    split_translational_modes, transform_tensor, vineyard_prefactor,
};

fn symmetrize(m: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = 0.5 * (m[i][j] + m[j][i]);
        }
    }
    out
}

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
    let dipole = elastic_dipole(volume, defect_stress, bulk_stress).unwrap();
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (dipole[i][j] - dipole[j][i]).abs() < 1e-12,
                "P not symmetric at {i}{j}"
            );
            assert!(
                (defect_stress[i][j] - defect_stress[j][i]).abs() < 1e-8,
                "pair stress not symmetric"
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
        let predicted = -double_dot(dipole, eps);
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
    let gamma0 = 0.0;
    let u_min = newton_u(gamma0, 1.0);
    let u_sad = newton_u(gamma0, 0.0);
    let (_, g_min) = well_force(u_min, gamma0);
    let (_, g_sad) = well_force(u_sad, gamma0);
    // The model's γ is the one-sided ε_yx of shear_yx. A symmetric virial
    // carries that same partial on both off-diagonal entries.
    let stress_of = |g: f64| [[0.0, g, 0.0], [g, 0.0, 0.0], [0.0, 0.0, 0.0]];
    let p_min = elastic_dipole(1.0, stress_of(g_min), [[0.0; 3]; 3]).unwrap();
    let p_sad = elastic_dipole(1.0, stress_of(g_sad), [[0.0; 3]; 3]).unwrap();
    assert!((p_min[1][0] - p_min[0][1]).abs() < 1e-12);
    let barrier0 = well(u_sad, gamma0) - well(u_min, gamma0);
    let model = rgsaddle::DipoleModel {
        minimum: p_min,
        saddle: p_sad,
        barrier0,
    };
    let gamma = 1.0e-5;
    let min_g = newton_u(gamma, u_min);
    let sad_g = newton_u(gamma, u_sad);
    let explicit = well(sad_g, gamma) - well(min_g, gamma);
    let linear = model.linear_barrier(shear_yx(gamma));
    assert!(
        (linear - explicit).abs() < 1e-8,
        "linear {linear} explicit {explicit} barrier0 {barrier0}"
    );
    let (status, error) = judge_linear(linear, Some(explicit), LINEAR_FOLD_EV);
    assert_eq!(status, LinearStatus::Holds);
    assert!(error.unwrap() < 1e-8);
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
    let delta = {
        let mut out = [[0.0; 3]; 3];
        let p_min = symmetrize(minimum);
        let p_sad = symmetrize(saddle);
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = p_sad[i][j] - p_min[i][j];
            }
        }
        out
    };
    for i in 0..3 {
        for j in 0..3 {
            assert!((omega[i][j] + delta[i][j]).abs() < 1e-12);
        }
    }
    let model = rgsaddle::DipoleModel {
        minimum: symmetrize(minimum),
        saddle: symmetrize(saddle),
        barrier0: 0.4,
    };
    let step = 1e-6;
    let mut tension = [[0.0; 3]; 3];
    tension[0][0] = step;
    let shifted = model.linear_barrier(tension) - model.barrier0;
    assert!((shifted / step - omega[0][0]).abs() < 1e-9);

    let mut stiffness = identity;
    stiffness[0][0] = 2.0;
    let compliance = compliance_from_stiffness(&stiffness).unwrap();
    assert!((compliance[0][0] - 0.5).abs() < 1e-12);
    let omega = activation_volume(minimum, saddle, &compliance).unwrap();
    let strain = hydrostatic(0.0);
    let _ = strain;
    let mut eps = [[0.0; 3]; 3];
    eps[0][0] = compliance[0][0] * step;
    let finite = (model.linear_barrier(eps) - model.barrier0) / step;
    assert!(
        (finite - omega[0][0]).abs() < 1e-8,
        "dG/dtau {finite} omega {}",
        omega[0][0]
    );
}

#[test]
fn vineyard_prefactor_has_one_factor_of_frequency() {
    let mass = 4.0;
    let scale = rgsaddle::angular_scale_rad_s();
    let expected = scale * (1.0_f64 / mass).sqrt() / (2.0 * std::f64::consts::PI);
    let got = vineyard_prefactor(&[1.0], &[], mass).unwrap();
    assert!((got - expected).abs() / expected < 1e-12);
    let two = vineyard_prefactor(&[4.0, 9.0], &[4.0], mass).unwrap();
    let ratio = scale * (9.0_f64 / mass).sqrt() / (2.0 * std::f64::consts::PI);
    assert!((two - ratio).abs() / ratio < 1e-12);
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

fn sample(
    direction: &str,
    strain: [[f64; 3]; 3],
    linear: f64,
    status: LinearStatus,
    helmholtz: ExplicitSide,
    gibbs: ExplicitSide,
    error: Option<f64>,
) -> StrainPoint {
    StrainPoint {
        direction: direction.to_string(),
        variant: String::new(),
        hop: [0.0; 3],
        strain,
        linear,
        linear_status: status,
        helmholtz,
        gibbs,
        linear_error: error,
    }
}

fn side(barrier: f64, prefactor: f64) -> ExplicitSide {
    ExplicitSide::barrier(barrier, Some(prefactor), "index 1").unwrap()
}

fn fixture_table() -> StrainRateTable {
    let pref = 1.0e12;
    let zero = sample(
        "reference",
        shear_yx(0.0),
        0.40,
        LinearStatus::Holds,
        side(0.40, pref),
        side(0.30, pref),
        Some(0.0),
    );
    let small = sample(
        "shear_yx",
        shear_yx(0.01),
        0.30,
        LinearStatus::Holds,
        side(0.305, pref * 0.9),
        side(0.20, pref * 0.8),
        Some(0.005),
    );
    let folded = sample(
        "shear_yx",
        shear_yx(0.05),
        0.05,
        LinearStatus::Failed,
        ExplicitSide::no_barrier(),
        ExplicitSide::no_barrier(),
        None,
    );
    StrainRateTable::try_new(
        rgsaddle::DipoleModel {
            minimum: [[0.0; 3]; 3],
            saddle: [[0.0; 3]; 3],
            barrier0: 0.40,
        },
        vec![zero, small, folded],
        None,
    )
    .unwrap()
}

#[test]
fn lookup_interpolates_and_flags_no_barrier_at_five_percent() {
    let table = fixture_table();
    let csv = table.to_csv();
    let json = table.to_json();
    let from_csv = StrainRateTable::from_csv(&csv).unwrap();
    let from_json = StrainRateTable::from_json(&json).unwrap();
    assert_eq!(from_csv.points().len(), 3);
    assert_eq!(from_json.points().len(), 3);
    assert!((from_json.model().barrier0 - 0.40).abs() < 1e-15);

    let mid = from_csv
        .lookup(shear_yx(0.005), 300.0, Ensemble::Helmholtz)
        .unwrap();
    match mid {
        Rate::Harmonic(rate) => {
            assert!(rate.interpolated);
            assert!((rate.barrier - 0.3525).abs() < 1e-12);
            let ln = 1.0e12_f64.ln() + 0.5 * ((1.0e12 * 0.9_f64).ln() - 1.0e12_f64.ln());
            assert!((rate.prefactor_hz.ln() - ln).abs() < 1e-9);
        }
        other => panic!("expected an interpolated rate, got {other}"),
    }

    let five = table
        .lookup(shear_yx(0.05), 500.0, Ensemble::Gibbs)
        .unwrap();
    assert!(matches!(five, Rate::NoBarrier), "{five}");
    let five_fixed = table
        .lookup(shear_yx(0.05), 500.0, Ensemble::Helmholtz)
        .unwrap();
    assert!(matches!(five_fixed, Rate::NoBarrier));
    let linear = table
        .lookup(shear_yx(0.05), 500.0, Ensemble::Linear)
        .unwrap();
    assert!(matches!(linear, Rate::LinearFailed { .. }), "{linear}");

    let across = table.lookup(shear_yx(0.03), 300.0, Ensemble::Helmholtz);
    assert!(
        across.is_err(),
        "a query between a barrier and no barrier is refused"
    );
}

#[test]
fn copper_table_flags_no_barrier_at_five_percent_variable_cell() -> Result<(), SaddleError> {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks/strain_rate_table_cu.csv");
    let table = StrainRateTable::open(&path)?;
    let mut found = false;
    for point in table.points() {
        let shear = point.strain[1][0];
        let others = point.strain[0][0].abs()
            + point.strain[0][1].abs()
            + point.strain[0][2].abs()
            + point.strain[1][1].abs()
            + point.strain[1][2].abs()
            + point.strain[2][0].abs()
            + point.strain[2][1].abs()
            + point.strain[2][2].abs();
        if point.direction == "shear_yx" && (shear - 0.05).abs() < 1e-12 && others < 1e-12 {
            assert!(
                point.gibbs.is_no_barrier(),
                "5% variable cell gibbs index {}",
                point.gibbs.index
            );
            let rate = table.lookup(point.strain, 300.0, Ensemble::Gibbs)?;
            assert!(matches!(rate, Rate::NoBarrier), "{rate}");
            found = true;
        }
    }
    assert!(found, "the copper table has no 5% shear_yx row");
    Ok(())
}

#[test]
fn rotated_dipole_matches_the_symmetry_related_shear() {
    let mut dipole = [[0.0; 3]; 3];
    dipole[0][0] = 1.0;
    dipole[1][1] = 2.0;
    dipole[2][2] = 3.0;
    dipole[0][1] = 12.85;
    dipole[1][0] = 12.85;
    let turned = transform_tensor(dipole, rotation_x(1));
    let plain = 0.5 - double_dot(dipole, shear_yx(0.02));
    let rotated = 0.5 - double_dot(turned, shear_xz(0.02));
    assert!((plain - rotated).abs() < 1e-9, "yx {plain} xz {rotated}");
}

#[test]
fn lookup_returns_the_lowest_hop() {
    let mut low = sample(
        "shear_yx",
        shear_yx(0.02),
        0.2,
        LinearStatus::Holds,
        side(0.2, 1.0e12),
        side(0.15, 1.0e12),
        Some(0.0),
    );
    low.variant = "110".to_string();
    let mut high = sample(
        "shear_yx",
        shear_yx(0.02),
        0.4,
        LinearStatus::Holds,
        side(0.4, 1.0e12),
        side(0.35, 1.0e12),
        Some(0.0),
    );
    high.variant = "101".to_string();
    let zero = sample(
        "reference",
        shear_yx(0.0),
        0.5,
        LinearStatus::Holds,
        side(0.5, 1.0e12),
        side(0.5, 1.0e12),
        Some(0.0),
    );
    let table = StrainRateTable::try_new(
        rgsaddle::DipoleModel {
            minimum: [[0.0; 3]; 3],
            saddle: [[0.0; 3]; 3],
            barrier0: 0.5,
        },
        vec![zero, low, high],
        None,
    )
    .unwrap()
    .with_hop_variants(vec![
        HopVariant {
            name: "110".into(),
            minimum: [[0.0; 3]; 3],
            saddle: [[0.0; 3]; 3],
            barrier0: 0.2,
        },
        HopVariant {
            name: "101".into(),
            minimum: [[0.0; 3]; 3],
            saddle: [[0.0; 3]; 3],
            barrier0: 0.4,
        },
    ])
    .unwrap();
    let rate = table
        .lookup(shear_yx(0.02), 300.0, Ensemble::Helmholtz)
        .unwrap();
    match rate {
        Rate::Harmonic(rate) => {
            assert!((rate.barrier - 0.2).abs() < 1e-12);
            assert_eq!(rate.variant, "110");
        }
        other => panic!("expected the lower hop, got {other}"),
    }
}

#[test]
fn matching_variant_equates_the_symmetry_related_copper_shears() -> Result<(), SaddleError> {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks/strain_rate_table_cu.csv");
    let table = StrainRateTable::open(&path)?;
    let yx = table
        .points()
        .iter()
        .find(|point| point.direction == "shear_yx" && (point.strain[1][0] - 0.02).abs() < 1e-12)
        .expect("shear_yx 0.02");
    let xz = table
        .points()
        .iter()
        .find(|point| point.direction == "shear_xz")
        .expect("shear_xz");
    assert_eq!(yx.variant, "110", "yx hop {:?}", yx.hop);
    assert_eq!(xz.variant, "101", "xz hop {:?}", xz.hop);
    assert!(
        (yx.linear - xz.linear).abs() < 1e-6,
        "linear yx {} xz {}",
        yx.linear,
        xz.linear
    );
    let yx_explicit = yx.helmholtz.barrier.expect("yx barrier");
    let xz_explicit = xz.helmholtz.barrier.expect("xz barrier");
    assert!(
        (yx_explicit - xz_explicit).abs() < 1e-4,
        "explicit yx {yx_explicit} xz {xz_explicit}"
    );
    Ok(())
}

#[test]
fn shear_yx_slope_matches_the_central_difference() -> Result<(), SaddleError> {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks/strain_rate_table_cu.csv");
    let table = StrainRateTable::open(&path)?;
    let at = |gamma: f64| {
        table
            .points()
            .iter()
            .find(|point| {
                point.variant == "110"
                    && point.direction == "shear_yx"
                    && (point.strain[1][0] - gamma).abs() < 1e-12
            })
            .unwrap_or_else(|| panic!("missing 110 shear {gamma}"))
            .helmholtz
            .barrier
            .expect("barrier")
    };
    let slope = (at(0.0025) - at(-0.0025)) / 0.005;
    let variant = table
        .variants()
        .iter()
        .find(|variant| variant.name == "110")
        .expect("110 dipole");
    let mut delta = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            delta[i][j] = variant.saddle[i][j] - variant.minimum[i][j];
        }
    }
    let model = -double_dot(delta, shear_yx(1.0));
    assert!(
        (slope - model).abs() < 1.0,
        "explicit slope {slope} dipole slope {model}"
    );
    Ok(())
}
