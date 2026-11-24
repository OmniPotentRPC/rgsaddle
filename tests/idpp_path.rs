//! IDPP and S-IDPP initial paths: collision, triclinic minimum image,
//! eOn parity, a turning group, and NEB force calls from the starts.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, array};
use rgsaddle::mic::{self, Cell};
use rgsaddle::{
    BandConfig, BandSession, BandStatus, BandSurface, IdppConfig, SaddleError, SidppConfig,
    SpringKind, TangentKind, idpp_path, linear_path, sidpp_path,
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
    let cell = Cell::from_vectors(
        [6.0, 0.0, 0.0],
        [1.2, 5.0, 0.0],
        [0.3, 0.4, 7.0],
        [0.0, 0.0, 0.0],
    )
    .unwrap();
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
    let start = sidpp_start(&reactant, &product, 4, Some(&cell));
    assert_eq!(start.row(0), reactant.view());
    assert_eq!(start.row(3), product.view());
    assert!(min_pair(start.row(1), Some(&cell)) > 0.9);
    assert!(min_pair(start.row(2), Some(&cell)) > 0.9);
}

fn sidpp_start(r: &Array1<f64>, p: &Array1<f64>, n: usize, cell: Option<&Cell>) -> Array2<f64> {
    sidpp_path(r.view(), p.view(), n, cell, &SidppConfig::default()).unwrap()
}

/// An anchor pair on z and a three-atom group on the second atom,
/// turned 170 degrees about z, so every distance is the same at both
/// ends. The straight line takes the group through the axis.
fn rotor_endpoints() -> (Array1<f64>, Array1<f64>) {
    let group = [[1.0, 0.0, 2.0], [-0.5, 0.9, 2.0], [-0.45, -0.8, 2.1]];
    let (sin, cos) = 170.0_f64.to_radians().sin_cos();
    let mut reactant = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.5];
    let mut product = reactant.clone();
    for [x, y, z] in group {
        reactant.extend([x, y, z]);
        product.extend([cos * x - sin * y, sin * x + cos * y, z]);
    }
    (Array1::from(reactant), Array1::from(product))
}

fn interior_gap(path: &Array2<f64>) -> f64 {
    let n = path.nrows();
    (1..n - 1)
        .map(|i| min_pair(path.row(i), None))
        .fold(f64::MAX, f64::min)
}

/// Longest segment over the mean segment.
fn spacing_ratio(path: &Array2<f64>) -> f64 {
    let seg: Vec<f64> = (1..path.nrows())
        .map(|l| {
            (&path.row(l) - &path.row(l - 1))
                .mapv(|v| v * v)
                .sum()
                .sqrt()
        })
        .collect();
    seg.iter().copied().fold(0.0, f64::max) * seg.len() as f64 / seg.iter().sum::<f64>()
}

/// Largest miss of the interpolated distance over the rotor's bonds.
/// The cut 1.6 lies between the anchor bond, 1.5, and the closest
/// pair inside the group, 1.66.
fn bond_error(path: &Array2<f64>) -> f64 {
    let (last, n) = (path.nrows() - 1, path.ncols() / 3);
    let mut worst = 0.0_f64;
    for l in 1..last {
        let xi = l as f64 / last as f64;
        for (i, j) in (0..n).flat_map(|i| ((i + 1)..n).map(move |j| (i, j))) {
            let a = pair_distance(path.row(0), i, j, None);
            let b = pair_distance(path.row(last), i, j, None);
            if a < 1.6 && b < 1.6 {
                let miss = pair_distance(path.row(l), i, j, None) - (1.0 - xi) * a - xi * b;
                worst = worst.max(miss.abs());
            }
        }
    }
    worst
}

#[test]
fn sidpp_turns_a_group_without_a_clash_and_spaces_the_images() {
    let (r, p) = rotor_endpoints();
    let linear = linear_path(r.view(), p.view(), 9, None).unwrap();
    let idpp = idpp_path(r.view(), p.view(), 9, None, &IdppConfig::default()).unwrap();
    let start = sidpp_start(&r, &p, 9, None);
    assert_eq!(start.row(0), r.view());
    assert_eq!(start.row(8), p.view());
    assert!(interior_gap(&linear) < 0.16 && bond_error(&linear) > 0.6);
    // IDPP relaxes each image alone, so nothing spaces them.
    assert!(spacing_ratio(&idpp) > 3.0);
    let got = [
        interior_gap(&start),
        spacing_ratio(&start),
        bond_error(&start),
    ];
    for (value, golden) in got.iter().zip(GOLDEN_ROTOR) {
        assert!((value - golden).abs() < 1e-4, "{got:?}");
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

// The reactant is an index-2 stationary point of this cluster (Hessian
// eigenvalues near -7.9 and -2.2), and a climbing image climbs into it
// from either start, so the start comparison runs the band unclimbed.
fn neb(initial: Array2<f64>) -> (rgsaddle::BandReport, Array2<f64>) {
    let config = BandConfig::default()
        .with_tangent(TangentKind::Improved)
        .with_spring(SpringKind::Uniform { k: 5.0 })
        .with_climbing(None)
        .with_force_tol(0.01)
        .with_max_move(0.2)
        .with_remove_translation(true)
        .with_method(rgmin::Method::lbfgs_with_memory(20))
        .with_min_mode(None);
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

    let (linear_report, linear_end) = neb(linear);
    let (idpp_report, idpp_end) = neb(idpp);
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
        "neb force rows: linear {} (status {:?}, max force {}, interior energy {}), idpp {} (status {:?}, max force {}, interior energy {})",
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
fn sidpp_start_reaches_a_converged_band_in_fewer_force_calls_than_idpp() {
    let (r, p) = exchange_endpoints();
    let idpp = idpp_path(r.view(), p.view(), 6, None, &IdppConfig::default()).unwrap();
    let (idpp_report, _) = neb(idpp);
    let (report, _) = neb(sidpp_start(&r, &p, 6, None));
    assert_eq!(idpp_report.status, BandStatus::Converged);
    assert_eq!(report.status, BandStatus::Converged);
    assert_eq!(report.surface_rows, GOLDEN_EXCHANGE_ROWS);
    assert!(report.surface_rows < idpp_report.surface_rows);
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
    let singular = [0.0; 9];
    let rejected = unsafe {
        rgsaddle_idpp_path(
            3,
            2,
            reactant.as_ptr(),
            product.as_ptr(),
            singular.as_ptr(),
            10,
            1e-3,
            0.1,
            out.as_mut_ptr(),
        )
    };
    assert_eq!(rejected, rgsaddle::capi::RGSADDLE_INVALID_PARAMETER);
}

#[test]
#[cfg(feature = "capi")]
fn c_abi_trics_path_refines_a_supplied_path() {
    use rgsaddle::capi::{
        RGSADDLE_INVALID_PARAMETER, RGSADDLE_OK, RGSADDLE_SHAPE, rgsaddle_trics_path,
    };

    let reactant = array![0.0, 0.0, 0.0, 0.96, 0.0, 0.0, -0.24, 0.93, 0.0];
    let product = array![0.2, 0.0, 0.0, 1.18, 0.1, 0.0, -0.1, 1.0, 0.1];
    let initial = linear_path(reactant.view(), product.view(), 5, None).unwrap();
    let mut out = vec![0.0; initial.len()];
    let rc = unsafe {
        rgsaddle_trics_path(
            5,
            3,
            initial.as_ptr(),
            50,
            0.000_000_01,
            0.1,
            1,
            0.001_8,
            1000,
            1.0,
            out.as_mut_ptr(),
        )
    };
    assert_eq!(rc, RGSADDLE_OK);
    assert_eq!(&out[..reactant.len()], reactant.as_slice().unwrap());
    assert_eq!(
        &out[out.len() - product.len()..],
        product.as_slice().unwrap()
    );
    assert!(out.iter().all(|value| value.is_finite()));

    let mut in_place = initial.iter().copied().collect::<Vec<_>>();
    let rc_alias = unsafe {
        rgsaddle_trics_path(
            5,
            3,
            in_place.as_ptr(),
            50,
            0.000_000_01,
            0.1,
            1,
            0.001_8,
            1000,
            1.0,
            in_place.as_mut_ptr(),
        )
    };
    assert_eq!(rc_alias, RGSADDLE_OK);
    assert_eq!(in_place, out);

    let mut overlap = vec![0.0; initial.len() + 1];
    overlap[..initial.len()].copy_from_slice(initial.as_slice().unwrap());
    let rc_overlap = unsafe {
        rgsaddle_trics_path(
            5,
            3,
            overlap.as_ptr(),
            50,
            0.000_000_01,
            0.1,
            1,
            0.001_8,
            1000,
            1.0,
            overlap.as_mut_ptr().add(1),
        )
    };
    assert_eq!(rc_overlap, RGSADDLE_INVALID_PARAMETER);

    let rc_overflow_atoms = unsafe {
        rgsaddle_trics_path(
            5,
            i64::MAX,
            initial.as_ptr(),
            50,
            0.000_000_01,
            0.1,
            1,
            0.001_8,
            1000,
            1.0,
            out.as_mut_ptr(),
        )
    };
    assert_eq!(rc_overflow_atoms, RGSADDLE_SHAPE);

    let rc_overflow_total = unsafe {
        rgsaddle_trics_path(
            i64::MAX,
            2,
            initial.as_ptr(),
            50,
            0.000_000_01,
            0.1,
            1,
            0.001_8,
            1000,
            1.0,
            out.as_mut_ptr(),
        )
    };
    assert_eq!(rc_overflow_total, RGSADDLE_SHAPE);

    for (max_inv, diag, max_splice) in [(0, 1, 1000), (50, -1, 1000), (50, 7, 1000), (50, 1, 0)] {
        let rc_param = unsafe {
            rgsaddle_trics_path(
                5,
                3,
                initial.as_ptr(),
                max_inv,
                0.000_000_01,
                0.1,
                diag,
                0.001_8,
                max_splice,
                1.0,
                out.as_mut_ptr(),
            )
        };
        assert_eq!(rc_param, RGSADDLE_INVALID_PARAMETER);
    }
}

#[test]
#[cfg(feature = "capi")]
fn c_abi_trics_path_reports_solver_failures_and_crosses_a_linear_angle() {
    use rgsaddle::capi::{RGSADDLE_OK, RGSADDLE_SOLVER, rgsaddle_trics_path};

    let bend_path = |angles: &[f64], tail: bool| {
        let mut path = Vec::new();
        for degrees in angles {
            let (sine, cosine) = degrees.to_radians().sin_cos();
            path.extend([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, cosine, sine, 0.0]);
            if tail {
                path.extend([cosine + 0.4, sine + 0.7, 0.6]);
            }
        }
        path
    };
    let run = |path: &[f64], n_images: i64, n_atoms: i64, threshold: i32, iterations: i64| {
        let mut out = vec![0.0; path.len()];
        let rc = unsafe {
            rgsaddle_trics_path(
                n_images,
                n_atoms,
                path.as_ptr(),
                iterations,
                0.000_000_01,
                0.1,
                threshold,
                0.001_8,
                1000,
                1.0,
                out.as_mut_ptr(),
            )
        };
        (rc, out)
    };

    let blocked_maze = bend_path(&[60.0, 0.0, -60.0], false);
    assert_eq!(run(&blocked_maze, 3, 3, 0, 50).0, RGSADDLE_SOLVER);

    let singular_endpoint = bend_path(&[60.0, 40.0, 20.0, 0.0], true);
    assert_eq!(run(&singular_endpoint, 4, 4, 3, 3).0, RGSADDLE_SOLVER);

    let through_linear = bend_path(&[40.0, 10.0, 0.0, -10.0, -40.0], true);
    let (rc, out) = run(&through_linear, 5, 4, 1, 50);
    assert_eq!(rc, RGSADDLE_OK);
    assert_eq!(&out[..12], &through_linear[..12]);
    assert_eq!(&out[48..], &through_linear[48..]);
    assert!(out.iter().all(|value| value.is_finite()));
}

#[test]
fn rejects_a_short_path_and_a_bad_stop() {
    let reactant = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let product = array![0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
    let short = linear_path(reactant.view(), product.view(), 2, None).unwrap_err();
    assert!(matches!(short, SaddleError::Shape(_)), "{short}");
    let empty = idpp_path(
        reactant.view(),
        product.view(),
        3,
        None,
        &IdppConfig::default().with_max_iterations(0),
    )
    .unwrap_err();
    assert!(matches!(empty, SaddleError::Invalid(_)), "{empty}");
    let tol = idpp_path(
        reactant.view(),
        product.view(),
        3,
        None,
        &IdppConfig::default().with_grad_tol(f64::NAN),
    )
    .unwrap_err();
    assert!(matches!(tol, SaddleError::Invalid(_)), "{tol}");
    let mut bad = reactant.clone();
    bad[0] = f64::INFINITY;
    let non_finite =
        idpp_path(bad.view(), product.view(), 3, None, &IdppConfig::default()).unwrap_err();
    assert!(
        matches!(non_finite, SaddleError::NonFinite(_)),
        "{non_finite}"
    );
    for (time_step, max_iterations) in [(0.0, 10), (0.1, 0)] {
        let config = SidppConfig::default()
            .with_time_step(time_step)
            .with_max_iterations(max_iterations);
        let err = sidpp_path(reactant.view(), product.view(), 3, None, &config).unwrap_err();
        assert!(matches!(err, SaddleError::Invalid(_)), "{err}");
    }
}

/// Measured with `SidppConfig::default()`: the rotor's closest interior
/// pair, spacing ratio, and bond miss, and the exchange band's force
/// rows. The straight rotor line gives 0.152 and 0.637, IDPP a spacing
/// ratio of 3.284, and the IDPP exchange start 178 rows.
const GOLDEN_ROTOR: [f64; 3] = [1.096481, 1.003693, 0.008818];
const GOLDEN_EXCHANGE_ROWS: usize = 158;

#[test]
#[cfg(feature = "capi")]
fn c_abi_trics_path_honours_the_length_unit() {
    use rgsaddle::capi::{RGSADDLE_INVALID_PARAMETER, RGSADDLE_OK, rgsaddle_trics_path};

    let reactant = array![0.0, 0.0, 0.0, 0.96, 0.0, 0.0, -0.24, 0.93, 0.0];
    let product = array![0.2, 0.0, 0.0, 1.18, 0.1, 0.0, -0.1, 1.0, 0.1];
    let angstrom = linear_path(reactant.view(), product.view(), 5, None).unwrap();
    let run = |path: &[f64], unit: f64| {
        let mut out = vec![0.0; path.len()];
        let rc = unsafe {
            rgsaddle_trics_path(
                5,
                3,
                path.as_ptr(),
                50,
                0.000_000_01,
                0.1,
                1,
                0.001_8,
                1000,
                unit,
                out.as_mut_ptr(),
            )
        };
        (rc, out)
    };
    let (rc, reference) = run(angstrom.as_slice().unwrap(), 1.0);
    assert_eq!(rc, RGSADDLE_OK);
    for unit in [0.1, 1.889_726_124_565_062] {
        let scaled: Vec<f64> = angstrom.iter().map(|value| value * unit).collect();
        let (rc, out) = run(&scaled, unit);
        assert_eq!(rc, RGSADDLE_OK);
        assert_eq!(&out[..9], &scaled[..9]);
        assert_eq!(&out[36..], &scaled[36..]);
        for (got, want) in out.iter().zip(&reference) {
            assert!((got - want * unit).abs() <= 1e-9 * unit, "unit {unit}");
        }
    }
    for unit in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let (rc, _) = run(angstrom.as_slice().unwrap(), unit);
        assert_eq!(rc, RGSADDLE_INVALID_PARAMETER, "unit {unit}");
    }
}
