//! Alternating IDPP paths: mask, minimum image, equivariance, contacts.

use ndarray::{Array1, Array2, ArrayView1, array};
use rgsaddle::mic::{self, Cell};
use rgsaddle::{
    IdppConfig, SaddleError, SidppAlternatingConfig, idpp_path, idpp_path_fixed, linear_path,
    sidpp_alternating_inverse_distance_path, sidpp_alternating_path,
};

fn pair_distance(pos: ArrayView1<f64>, i: usize, j: usize) -> f64 {
    let dx = pos[3 * i] - pos[3 * j];
    let dy = pos[3 * i + 1] - pos[3 * j + 1];
    let dz = pos[3 * i + 2] - pos[3 * j + 2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn min_pair(pos: ArrayView1<f64>) -> f64 {
    let n = pos.len() / 3;
    let mut best = f64::MAX;
    for i in 0..n {
        for j in (i + 1)..n {
            best = best.min(pair_distance(pos, i, j));
        }
    }
    best
}

fn closest_interior(path: &Array2<f64>) -> (usize, f64) {
    let mut best_i = 1usize;
    let mut best = f64::MAX;
    for i in 1..path.nrows() - 1 {
        let distance = min_pair(path.row(i));
        if distance < best {
            best = distance;
            best_i = i;
        }
    }
    (best_i, best)
}

fn max_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .fold(0.0, |m, (x, y)| m.max((x - y).abs()))
}

fn fast() -> SidppAlternatingConfig {
    SidppAlternatingConfig::default()
        .with_short_iterations(6)
        .with_final_iterations(8)
        .with_pull(IdppConfig::default().with_max_iterations(8))
}

fn endpoints_match(path: &Array2<f64>, reactant: &Array1<f64>, product: &Array1<f64>) {
    assert_eq!(path.row(0), reactant.view());
    assert_eq!(path.row(path.nrows() - 1), product.view());
}

fn frozen_row(path: &Array2<f64>, atom: usize, held: &[f64]) {
    for image in 1..path.nrows() - 1 {
        for c in 0..3 {
            assert_eq!(
                path[(image, 3 * atom + c)],
                held[c],
                "image {image} atom {atom} component {c}"
            );
        }
    }
}

#[test]
fn empty_mask_matches_idpp_path() {
    let reactant = array![0.0, 0.0, 0.0, 1.0, 0.2, 0.0];
    let product = array![0.3, -0.1, 0.0, 0.0, 1.1, 0.2];
    let config = IdppConfig::default().with_max_iterations(5);
    let plain = idpp_path(reactant.view(), product.view(), 4, None, &config).unwrap();
    let masked = idpp_path_fixed(reactant.view(), product.view(), 4, None, &[], &config).unwrap();
    assert_eq!(plain, masked);
}

#[test]
fn frozen_atom_stays_on_the_reactant() {
    let reactant = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let product = array![0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let config = IdppConfig::default().with_max_iterations(80);
    let masked = idpp_path_fixed(
        reactant.view(),
        product.view(),
        3,
        None,
        &[true, false],
        &config,
    )
    .unwrap();
    let free = idpp_path(reactant.view(), product.view(), 3, None, &config).unwrap();
    endpoints_match(&masked, &reactant, &product);
    endpoints_match(&free, &reactant, &product);
    frozen_row(&masked, 0, &[0.0, 0.0, 0.0]);
    let moved = free[(1, 0)].hypot(free[(1, 1)]);
    assert!(
        moved > 1e-2,
        "unmasked atom 0 should leave the origin, moved {moved}"
    );
    let seq = sidpp_alternating_path(
        reactant.view(),
        product.view(),
        4,
        None,
        &[true, false],
        &fast(),
    )
    .unwrap();
    let geo = sidpp_alternating_inverse_distance_path(
        reactant.view(),
        product.view(),
        4,
        None,
        &[true, false],
        &fast(),
    )
    .unwrap();
    frozen_row(&seq, 0, &[0.0, 0.0, 0.0]);
    frozen_row(&geo, 0, &[0.0, 0.0, 0.0]);
    endpoints_match(&seq, &reactant, &product);
    endpoints_match(&geo, &reactant, &product);
}

#[test]
fn fixed_atom_that_moves_is_rejected_and_a_short_mask_is_a_shape_error() {
    let reactant = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let product = array![0.2, 0.0, 0.0, 0.0, 1.0, 0.0];
    let err = sidpp_alternating_path(
        reactant.view(),
        product.view(),
        3,
        None,
        &[true, false],
        &fast(),
    )
    .unwrap_err();
    assert!(matches!(err, SaddleError::Invalid(_)), "{err}");
    let err = idpp_path_fixed(
        reactant.view(),
        product.view(),
        3,
        None,
        &[true],
        &IdppConfig::default().with_max_iterations(1),
    )
    .unwrap_err();
    assert!(matches!(err, SaddleError::Shape(_)), "{err}");
    let err = sidpp_alternating_path(
        reactant.view(),
        product.view(),
        3,
        None,
        &[],
        &fast().with_alpha(0.0),
    )
    .unwrap_err();
    assert!(matches!(err, SaddleError::Invalid(_)), "{err}");
    let mut bad = reactant.clone();
    bad[3] = f64::NAN;
    let err =
        sidpp_alternating_path(bad.view(), product.view(), 3, None, &[], &fast()).unwrap_err();
    assert!(matches!(err, SaddleError::NonFinite(_)), "{err}");
}

#[test]
fn zero_iteration_reparam_matches_the_straight_minimum_image() {
    let cell = Cell::ortho(5.0, 5.0, 5.0).unwrap();
    let reactant = array![0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 2.0, 2.0, 2.0];
    let product = array![0.0, 0.0, 0.0, 4.8, 0.0, 0.0, 2.0, 2.0, 2.0];
    let fixed = [true, false, true];
    let config = SidppAlternatingConfig::default()
        .with_short_iterations(0)
        .with_final_iterations(0);
    let straight = linear_path(reactant.view(), product.view(), 5, Some(&cell)).unwrap();
    let cartesian = linear_path(reactant.view(), product.view(), 5, None).unwrap();
    let periodic = sidpp_alternating_path(
        reactant.view(),
        product.view(),
        5,
        Some(&cell),
        &fixed,
        &config,
    )
    .unwrap();
    let free =
        sidpp_alternating_path(reactant.view(), product.view(), 5, None, &fixed, &config).unwrap();
    assert!(
        max_diff(&straight, &cartesian) > 1.0,
        "cell and free linear paths collapsed"
    );
    let mut mic_gap = 0.0_f64;
    for i in 0..periodic.nrows() {
        let mut d = &periodic.row(i) - &straight.row(i);
        mic::wrap_difference(&cell, &mut d);
        for v in &d {
            mic_gap = mic_gap.max(v.abs());
        }
    }
    assert!(mic_gap < 1e-8, "periodic minimum-image gap {mic_gap}");
    assert!(
        (periodic[(2, 3)] - 0.05).abs() < 1e-6,
        "middle x {}",
        periodic[(2, 3)]
    );
    assert!(
        max_diff(&free, &cartesian) < 1e-8,
        "free {}",
        max_diff(&free, &cartesian)
    );
    frozen_row(&periodic, 0, &[0.0, 0.0, 0.0]);
    frozen_row(&periodic, 2, &[2.0, 2.0, 2.0]);
    endpoints_match(&periodic, &reactant, &product);
    let mut mic = array![4.8 - 0.3, 0.0, 0.0];
    mic::wrap_difference(&cell, &mut mic);
    println!(
        "periodic middle x {:.12} mic step {:.12}",
        periodic[(2, 3)],
        mic[0]
    );
}

#[test]
fn relaxed_periodic_path_stays_on_the_short_image() {
    let cell = Cell::ortho(5.0, 5.0, 5.0).unwrap();
    let reactant = array![0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 2.0, 2.0, 2.0];
    let product = array![0.0, 0.0, 0.0, 4.8, 0.0, 0.0, 2.0, 2.0, 2.0];
    let fixed = [true, false, true];
    let config = fast();
    for (label, path) in [
        (
            "sequential",
            sidpp_alternating_path(
                reactant.view(),
                product.view(),
                5,
                Some(&cell),
                &fixed,
                &config,
            )
            .unwrap(),
        ),
        (
            "geodesic",
            sidpp_alternating_inverse_distance_path(
                reactant.view(),
                product.view(),
                5,
                Some(&cell),
                &fixed,
                &config,
            )
            .unwrap(),
        ),
    ] {
        endpoints_match(&path, &reactant, &product);
        frozen_row(&path, 2, &[2.0, 2.0, 2.0]);
        for image in 1..4 {
            let x = path[(image, 3)];
            assert!(
                !(1.0..=4.0).contains(&x),
                "{label} image {image} crossed the long gap at x {x}"
            );
        }
    }
}

#[test]
fn permutation_and_rigid_motion_leave_the_path_equivariant() {
    let reactant = array![0.1, -0.2, 0.3, 1.2, 0.4, -0.1, 0.2, 1.0, 0.5];
    let product = array![0.4, 0.2, -0.3, 0.2, 1.3, 0.4, 1.1, 0.1, -0.2];
    let config = fast();
    let seq =
        sidpp_alternating_path(reactant.view(), product.view(), 4, None, &[], &config).unwrap();
    let geo = sidpp_alternating_inverse_distance_path(
        reactant.view(),
        product.view(),
        4,
        None,
        &[],
        &config,
    )
    .unwrap();
    let swap = |pos: &Array1<f64>| {
        let mut out = pos.clone();
        for c in 0..3 {
            out.swap(c, 6 + c);
        }
        out
    };
    let swap_path = |path: &Array2<f64>| {
        let mut out = path.clone();
        for row in 0..path.nrows() {
            for c in 0..3 {
                out.swap((row, c), (row, 6 + c));
            }
        }
        out
    };
    let reactant_s = swap(&reactant);
    let product_s = swap(&product);
    let seq_s =
        sidpp_alternating_path(reactant_s.view(), product_s.view(), 4, None, &[], &config).unwrap();
    let geo_s = sidpp_alternating_inverse_distance_path(
        reactant_s.view(),
        product_s.view(),
        4,
        None,
        &[],
        &config,
    )
    .unwrap();
    let seq_gap = max_diff(&seq_s, &swap_path(&seq));
    let geo_gap = max_diff(&geo_s, &swap_path(&geo));
    assert!(seq_gap < 1e-6, "sequential permutation {seq_gap}");
    assert!(geo_gap < 1e-6, "geodesic permutation {geo_gap}");

    let shift = |pos: &Array1<f64>| {
        let mut out = pos.clone();
        for atom in 0..3 {
            out[3 * atom] += 0.4;
            out[3 * atom + 1] -= 0.2;
            out[3 * atom + 2] += 0.7;
        }
        out
    };
    let shift_path = |path: &Array2<f64>| {
        let mut out = path.clone();
        for row in 0..path.nrows() {
            for atom in 0..3 {
                out[(row, 3 * atom)] += 0.4;
                out[(row, 3 * atom + 1)] -= 0.2;
                out[(row, 3 * atom + 2)] += 0.7;
            }
        }
        out
    };
    let seq_t = sidpp_alternating_path(
        shift(&reactant).view(),
        shift(&product).view(),
        4,
        None,
        &[],
        &config,
    )
    .unwrap();
    let geo_t = sidpp_alternating_inverse_distance_path(
        shift(&reactant).view(),
        shift(&product).view(),
        4,
        None,
        &[],
        &config,
    )
    .unwrap();
    let seq_shift = max_diff(&seq_t, &shift_path(&seq));
    let geo_shift = max_diff(&geo_t, &shift_path(&geo));
    assert!(seq_shift < 1e-6, "sequential translation {seq_shift}");
    assert!(geo_shift < 1e-6, "geodesic translation {geo_shift}");

    let rot = |pos: &Array1<f64>| {
        let mut out = pos.clone();
        for atom in 0..3 {
            let x = pos[3 * atom];
            let y = pos[3 * atom + 1];
            out[3 * atom] = -y;
            out[3 * atom + 1] = x;
        }
        out
    };
    let rot_path = |path: &Array2<f64>| {
        let mut out = path.clone();
        for row in 0..path.nrows() {
            for atom in 0..3 {
                let x = path[(row, 3 * atom)];
                let y = path[(row, 3 * atom + 1)];
                out[(row, 3 * atom)] = -y;
                out[(row, 3 * atom + 1)] = x;
            }
        }
        out
    };
    let seq_r = sidpp_alternating_path(
        rot(&reactant).view(),
        rot(&product).view(),
        4,
        None,
        &[],
        &config,
    )
    .unwrap();
    let geo_r = sidpp_alternating_inverse_distance_path(
        rot(&reactant).view(),
        rot(&product).view(),
        4,
        None,
        &[],
        &config,
    )
    .unwrap();
    let seq_rot = max_diff(&seq_r, &rot_path(&seq));
    let geo_rot = max_diff(&geo_r, &rot_path(&geo));
    assert!(seq_rot < 1e-6, "sequential rotation {seq_rot}");
    assert!(geo_rot < 1e-6, "geodesic rotation {geo_rot}");
}

#[test]
fn inverse_distance_pull_is_not_the_linear_target() {
    let reactant = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let product = array![0.0, 0.0, 0.0, 4.0, 0.0, 0.0];
    let fixed = [true, false];
    let config = SidppAlternatingConfig::default();
    let seq =
        sidpp_alternating_path(reactant.view(), product.view(), 5, None, &fixed, &config).unwrap();
    let geo = sidpp_alternating_inverse_distance_path(
        reactant.view(),
        product.view(),
        5,
        None,
        &fixed,
        &config,
    )
    .unwrap();
    endpoints_match(&seq, &reactant, &product);
    endpoints_match(&geo, &reactant, &product);
    frozen_row(&seq, 0, &[0.0, 0.0, 0.0]);
    frozen_row(&geo, 0, &[0.0, 0.0, 0.0]);
    let seq_d = pair_distance(seq.row(2), 0, 1);
    let geo_d = pair_distance(geo.row(2), 0, 1);
    println!("inverse middle sequential {seq_d:.12} geodesic {geo_d:.12}");
    assert!((seq_d - 2.5).abs() < 1e-9, "linear target middle {seq_d}");
    assert!((geo_d - 1.6).abs() < 1e-9, "inverse target middle {geo_d}");
}

#[test]
fn collision_contacts() {
    // Same endpoints as tests/idpp_path.rs. Five images, counting the
    // ends. The straight middle image puts the atoms 0.02 apart.
    let reactant = array![0.0, 0.0, 0.0, 2.0, 0.02, 0.0];
    let product = array![2.0, 1.0, 0.0, 0.0, 1.02, 0.0];
    let linear = linear_path(reactant.view(), product.view(), 5, None).unwrap();
    let idpp = idpp_path(
        reactant.view(),
        product.view(),
        5,
        None,
        &IdppConfig::default(),
    )
    .unwrap();
    let sequential = sidpp_alternating_path(
        reactant.view(),
        product.view(),
        5,
        None,
        &[],
        &SidppAlternatingConfig::default(),
    )
    .unwrap();
    let geodesic = sidpp_alternating_inverse_distance_path(
        reactant.view(),
        product.view(),
        5,
        None,
        &[],
        &SidppAlternatingConfig::default(),
    )
    .unwrap();
    for path in [&linear, &idpp, &sequential, &geodesic] {
        endpoints_match(path, &reactant, &product);
    }
    let (li, ld) = closest_interior(&linear);
    let (ii, id) = closest_interior(&idpp);
    let (si, sd) = closest_interior(&sequential);
    let (gi, gd) = closest_interior(&geodesic);
    println!(
        "contact endpoint {:.12}",
        pair_distance(reactant.view(), 0, 1)
    );
    println!("contact linear {ld:.12} image {li}");
    println!("contact idpp {id:.12} image {ii}");
    println!("contact sequential {sd:.12} image {si}");
    println!("contact geodesic {gd:.12} image {gi}");
    assert!((ld - 0.02).abs() < 1e-12, "linear {ld}");
    assert!((id - 1.995084722350).abs() < 1e-9, "idpp {id}");
    assert!((sd - 1.985794596637).abs() < 1e-9, "sequential {sd}");
    assert!((gd - 1.998473688789).abs() < 1e-9, "geodesic {gd}");
}

#[cfg(feature = "capi")]
#[test]
fn c_api_matches_the_rust_periodic_line() {
    use rgsaddle::capi::{
        RGSADDLE_INVALID_PARAMETER, RGSADDLE_OK, rgsaddle_sidpp_alternating_inverse_distance_path,
        rgsaddle_sidpp_alternating_path,
    };

    let reactant = [0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 2.0, 2.0, 2.0];
    let product = [0.0, 0.0, 0.0, 4.8, 0.0, 0.0, 2.0, 2.0, 2.0];
    let cell = [5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0];
    let mask = [1u8, 0, 1];
    let mut out = [0.0; 45];
    let rc = unsafe {
        rgsaddle_sidpp_alternating_path(
            5,
            3,
            reactant.as_ptr(),
            product.as_ptr(),
            cell.as_ptr(),
            mask.as_ptr(),
            0.33,
            0,
            0,
            1e-3,
            0.1,
            5.0,
            out.as_mut_ptr(),
        )
    };
    assert_eq!(rc, RGSADDLE_OK);
    let wrapped = Cell::ortho(5.0, 5.0, 5.0).unwrap();
    let rust = sidpp_alternating_path(
        Array1::from(reactant.to_vec()).view(),
        Array1::from(product.to_vec()).view(),
        5,
        Some(&wrapped),
        &[true, false, true],
        &SidppAlternatingConfig::default()
            .with_short_iterations(0)
            .with_final_iterations(0),
    )
    .unwrap();
    let mut worst = 0.0_f64;
    for (got, want) in out.iter().zip(rust.iter()) {
        worst = worst.max((got - want).abs());
    }
    assert!(worst < 1e-12, "c sequential {worst}");
    let rejected = unsafe {
        rgsaddle_sidpp_alternating_inverse_distance_path(
            5,
            3,
            reactant.as_ptr(),
            product.as_ptr(),
            std::ptr::null(),
            std::ptr::null(),
            1.0,
            1,
            1,
            1e-3,
            0.1,
            5.0,
            out.as_mut_ptr(),
        )
    };
    assert_eq!(rejected, RGSADDLE_INVALID_PARAMETER);
}
