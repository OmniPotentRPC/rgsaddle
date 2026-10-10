//! IDPP initial path: collision, triclinic minimum image, eOn parity,
//! and CI-NEB force calls from the two starts.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, array};
use rgsaddle::mic::{self, Cell};
use rgsaddle::{
    BandConfig, BandSession, BandStatus, BandSurface, CiConfig, IdppConfig, SaddleError,
    SpringKind, TangentKind, idpp_path, linear_path,
};

fn pair_distance(pos: ArrayView1<f64>, i: usize, j: usize, cell: Option<&Cell>) -> f64 {
    let mut diff = array![
        pos[3 * i] - pos[3 * j],
        pos[3 * i + 1] - pos[3 * j + 1],
        pos[3 * i + 2] - pos[3 * j + 2],
    ];
    if let Some(cell) = cell {
        mic::wrap_difference(cell, &mut diff);
    }
    diff.dot(&diff).sqrt()
}

fn min_pair(pos: ArrayView1<f64>, cell: Option<&Cell>) -> f64 {
    let n = pos.len() / 3;
    let mut best = f64::MAX;
    for i in 0..n {
        for j in (i + 1)..n {
            best = best.min(pair_distance(pos, i, j, cell));
        }
    }
    best
}

/// Largest absolute minimum-image component between two geometries.
fn max_mic(cell: Option<&Cell>, got: ArrayView1<f64>, reference: &[f64]) -> f64 {
    let mut worst = 0.0_f64;
    let n = got.len() / 3;
    for atom in 0..n {
        let mut diff = array![
            got[3 * atom] - reference[3 * atom],
            got[3 * atom + 1] - reference[3 * atom + 1],
            got[3 * atom + 2] - reference[3 * atom + 2],
        ];
        if let Some(cell) = cell {
            mic::wrap_difference(cell, &mut diff);
        }
        for value in diff.iter() {
            worst = worst.max(value.abs());
        }
    }
    worst
}

#[test]
fn linear_interpolation_overlaps_and_idpp_separates() {
    // The straight line crosses with a 0.02 impact parameter, so the
    // single interior image puts the atoms on top of each other.
    let reactant = array![0.0, 0.0, 0.0, 2.0, 0.02, 0.0];
    let product = array![2.0, 1.0, 0.0, 0.0, 1.02, 0.0];
    let linear = linear_path(reactant.view(), product.view(), 3, None).unwrap();
    let idpp = idpp_path(
        reactant.view(),
        product.view(),
        3,
        None,
        &IdppConfig::default(),
    )
    .unwrap();
    assert!(min_pair(linear.row(1), None) < 0.05);
    assert!(min_pair(idpp.row(1), None) > 1.9);
    assert_eq!(idpp.row(0), reactant.view());
    assert_eq!(idpp.row(2), product.view());
    // pyeonclient 0.4.2 neb_idpp_path, eOn idppPath, gradient L2 < 1e-3.
    let reference = [1.0, -0.48892963988202126, 0.0, 1.0, 1.5089296398820433, 0.0];
    assert!(
        max_mic(None, idpp.row(1), &reference) < 3e-3,
        "{}",
        max_mic(None, idpp.row(1), &reference)
    );
}

#[test]
fn triclinic_path_uses_the_minimum_image_and_matches_eon() {
    let cell = Cell::from_ase([[6.0, 0.0, 0.0], [1.2, 5.0, 0.0], [0.3, 0.4, 7.0]]).unwrap();
    let reactant = array![0.4, 0.5, 0.5, 5.5, 0.8, 0.6, 3.0, 2.5, 3.5];
    let product = array![5.6, 0.7, 0.8, 0.5, 0.6, 0.4, 3.0, 2.5, 3.5];
    let linear = linear_path(reactant.view(), product.view(), 4, Some(&cell)).unwrap();
    let idpp = idpp_path(
        reactant.view(),
        product.view(),
        4,
        Some(&cell),
        &IdppConfig::default(),
    )
    .unwrap();
    // Short step: reactant x plus one third of the -0.8 minimum-image
    // displacement, not one third of the long 5.2 Cartesian gap.
    let straight = [
        0.4 - 0.8 / 3.0,
        0.5 + 0.2 / 3.0,
        0.5 + 0.3 / 3.0,
        5.5 + 1.0 / 3.0,
        0.8 - 0.2 / 3.0,
        0.6 - 0.2 / 3.0,
        3.0,
        2.5,
        3.5,
    ];
    assert!(max_mic(Some(&cell), linear.row(1), &straight) < 1e-12);
    assert!(min_pair(linear.row(1), Some(&cell)) < 0.4);
    assert!(min_pair(idpp.row(1), Some(&cell)) > 0.9);
    assert!(min_pair(idpp.row(2), Some(&cell)) > 0.9);
    // Same eOn idppPath stop as the free-space case. eOn stores a
    // lattice translate of the unwrapped image.
    let reference = [
        [
            0.39765117231697955,
            0.41911233268925996,
            0.658280915146819,
            5.569659890893558,
            0.8810438226906172,
            0.4755256045654346,
            2.9993556034562148,
            2.499843844620122,
            3.4995268136211037,
        ],
        [
            5.632425854139878,
            0.6073564853457033,
            0.8823594236787684,
            0.40147410048459753,
            0.6930186340335371,
            0.2849431138846925,
            2.999433378708923,
            2.4996248806207655,
            3.499364129103262,
        ],
    ];
    for (row, expect) in reference.iter().enumerate() {
        let err = max_mic(Some(&cell), idpp.row(row + 1), expect);
        assert!(err < 2e-4, "image {}: {err}", row + 1);
    }
}

/// Shifted 12-6 Lennard-Jones, epsilon = 1, sigma = 1, cutoff 15.
/// The constant shift is the unshifted energy at the cutoff, so the
/// pair energy and the force are zero beyond it. This is the rgpot LJ
/// kernel eOn calls for `PotType::LJ`.
fn lj_shifted(positions: ArrayView2<f64>, energies: &mut Array1<f64>, gradients: &mut Array2<f64>) {
    const RC: f64 = 15.0;
    let rc2 = 1.0 / (RC * RC);
    let rc6 = rc2 * rc2 * rc2;
    let shift = 4.0 * (rc6 * rc6 - rc6);
    let n_atoms = positions.ncols() / 3;
    for image in 0..positions.nrows() {
        let pos = positions.row(image);
        let mut energy = 0.0;
        let mut grad = vec![0.0; pos.len()];
        for i in 0..n_atoms {
            for j in (i + 1)..n_atoms {
                let dx = pos[3 * i] - pos[3 * j];
                let dy = pos[3 * i + 1] - pos[3 * j + 1];
                let dz = pos[3 * i + 2] - pos[3 * j + 2];
                let r = (dx * dx + dy * dy + dz * dz).sqrt();
                if r >= RC {
                    continue;
                }
                let sr2 = 1.0 / (r * r);
                let sr6 = sr2 * sr2 * sr2;
                let sr12 = sr6 * sr6;
                energy += 4.0 * (sr12 - sr6) - shift;
                let d_e_dr = 24.0 / r * (sr6 - 2.0 * sr12);
                let scale = d_e_dr / r;
                let (gx, gy, gz) = (scale * dx, scale * dy, scale * dz);
                grad[3 * i] += gx;
                grad[3 * i + 1] += gy;
                grad[3 * i + 2] += gz;
                grad[3 * j] -= gx;
                grad[3 * j + 1] -= gy;
                grad[3 * j + 2] -= gz;
            }
        }
        energies[image] = energy;
        for (k, g) in grad.iter().enumerate() {
            gradients[(image, k)] = *g;
        }
    }
}

struct Lj;

impl BandSurface for Lj {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        lj_shifted(positions, energies, gradients);
        Ok(())
    }
}

fn exchange_endpoints() -> (Array1<f64>, Array1<f64>) {
    // Minimized rgpot LJ endpoints of a four-atom exchange. The linear
    // image in the middle brings two atoms to 0.15.
    let reactant = array![
        0.0049206011764112,
        0.6512560235677834,
        0.1085426705946306,
        1.1175414471329614,
        0.6512560235677834,
        0.1085426705946306,
        0.0049206011764112,
        1.7487439764322168,
        0.2914573294053694,
        1.1175414471329614,
        1.7487439764322168,
        0.2914573294053694,
    ];
    let product = array![
        -0.0562331194558035,
        0.5115738336961295,
        0.0254474169714597,
        0.14218780528077,
        1.6079779458414332,
        0.1416653066629812,
        0.981179802547839,
        0.867332274327543,
        0.2538986841686863,
        1.177789608245941,
        1.963115946134895,
        0.3789885921968729,
    ];
    (reactant, product)
}

fn ci_neb(initial: Array2<f64>) -> (rgsaddle::BandReport, Array2<f64>) {
    let config = BandConfig {
        tangent: TangentKind::Improved,
        spring: SpringKind::Uniform { k: 5.0 },
        climbing: Some(CiConfig {
            trigger_factor: 0.5,
            trigger_force: 0.0,
        }),
        force_tol: 0.01,
        max_move: 0.2,
        remove_translation: true,
        method: rgmin::Method::Lbfgs { memory: 20 },
        ..BandConfig::default()
    };
    let mut session = BandSession::new(config, initial).unwrap();
    let mut rows = 0;
    let mut report = session.step(&Lj).unwrap();
    rows += report.surface_rows;
    while report.status == BandStatus::Running && report.iteration < 2500 {
        report = session.step(&Lj).unwrap();
        rows += report.surface_rows;
    }
    report.surface_rows = rows;
    (report, session.positions().to_owned())
}

#[test]
fn lennard_jones_matches_the_rgpot_endpoint_energy() {
    let (reactant, _) = exchange_endpoints();
    let positions = reactant.into_shape_with_order((1, 12)).unwrap();
    let mut energies = Array1::zeros(1);
    let mut gradients = Array2::zeros((1, 12));
    lj_shifted(positions.view(), &mut energies, &mut gradients);
    assert!(
        (energies[0] + 4.480618045083267).abs() < 1e-9,
        "{}",
        energies[0]
    );
}

#[test]
fn idpp_start_reaches_a_converged_band_in_fewer_force_calls() {
    let (reactant, product) = exchange_endpoints();
    let linear = linear_path(reactant.view(), product.view(), 6, None).unwrap();
    let idpp = idpp_path(
        reactant.view(),
        product.view(),
        6,
        None,
        &IdppConfig::default(),
    )
    .unwrap();
    let linear_clash = (1..5)
        .map(|i| min_pair(linear.row(i), None))
        .fold(f64::MAX, f64::min);
    let idpp_gap = (1..5)
        .map(|i| min_pair(idpp.row(i), None))
        .fold(f64::MAX, f64::min);
    assert!(linear_clash < 0.2, "{linear_clash}");
    assert!(idpp_gap > 1.0, "{idpp_gap}");

    let (linear_report, linear_end) = ci_neb(linear);
    let (idpp_report, idpp_end) = ci_neb(idpp);
    let max_interior = |path: &Array2<f64>| {
        let mut energies = Array1::zeros(path.nrows());
        let mut gradients = Array2::zeros(path.dim());
        lj_shifted(path.view(), &mut energies, &mut gradients);
        energies
            .iter()
            .skip(1)
            .take(path.nrows() - 2)
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    };
    eprintln!(
        "ci-neb force rows: linear {} (status {:?}, max force {}, interior energy {}), idpp {} (status {:?}, max force {}, interior energy {})",
        linear_report.surface_rows,
        linear_report.status,
        linear_report.max_force,
        max_interior(&linear_end),
        idpp_report.surface_rows,
        idpp_report.status,
        idpp_report.max_force,
        max_interior(&idpp_end)
    );
    assert_eq!(idpp_report.status, BandStatus::Converged);
    assert_eq!(linear_report.status, BandStatus::Converged);
    assert!(
        idpp_report.surface_rows < linear_report.surface_rows,
        "idpp {} linear {}",
        idpp_report.surface_rows,
        linear_report.surface_rows
    );
}

#[test]
#[cfg(feature = "capi")]
fn c_abi_idpp_path_copies_the_endpoints() {
    use rgsaddle::capi::{RGSADDLE_OK, RGSADDLE_SHAPE, rgsaddle_idpp_path};

    let reactant = [0.0, 0.0, 0.0, 2.0, 0.02, 0.0];
    let product = [2.0, 1.0, 0.0, 0.0, 1.02, 0.0];
    let mut out = [0.0; 18];
    let rc = unsafe {
        rgsaddle_idpp_path(
            3,
            2,
            reactant.as_ptr(),
            product.as_ptr(),
            std::ptr::null(),
            5000,
            1e-3,
            0.1,
            out.as_mut_ptr(),
        )
    };
    assert_eq!(rc, RGSADDLE_OK);
    assert_eq!(&out[..6], &reactant);
    assert_eq!(&out[12..], &product);
    let mid = Array1::from(out[6..12].to_vec());
    assert!(min_pair(mid.view(), None) > 1.9);
    let bad = unsafe {
        rgsaddle_idpp_path(
            2,
            2,
            reactant.as_ptr(),
            product.as_ptr(),
            std::ptr::null(),
            10,
            1e-3,
            0.1,
            out.as_mut_ptr(),
        )
    };
    assert_eq!(bad, RGSADDLE_SHAPE);
}
