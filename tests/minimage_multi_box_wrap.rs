use approx::assert_abs_diff_eq;
use ndarray::array;
use rgsaddle::mic::{self, Cell as MicCell};
use rgsaddle::sparse_hessian::SparsityPattern;
use rgsaddle::{
    Cell as BandCell, PeriodicSymmetryOperations, Reparametrization, SymmetryOperations,
    SymmetrySource, linear_path, reparametrize,
};

const WRAP_TOL: f64 = 1e-12;

#[test]
fn orthorhombic_separations_past_one_and_a_half_boxes_wrap_to_five() {
    let band_cell = BandCell([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]);
    let mi = band_cell.as_minimage().expect("orthorhombic minimage cell");

    for raw in [25.0_f64, 35.0_f64] {
        let dr = mi.displacement([0.0, 0.0, 0.0], [raw, 0.0, 0.0]);
        assert_abs_diff_eq!(dr[0], -5.0, epsilon = WRAP_TOL);
        assert_abs_diff_eq!(dr[1], 0.0, epsilon = WRAP_TOL);
        assert_abs_diff_eq!(dr[2], 0.0, epsilon = WRAP_TOL);
        assert_abs_diff_eq!(
            mi.dist2([0.0, 0.0, 0.0], [raw, 0.0, 0.0]).sqrt(),
            5.0,
            epsilon = WRAP_TOL
        );

        let mut band_diff = array![raw, -raw, 0.0];
        band_cell.minimum_image(&mut band_diff);
        assert_abs_diff_eq!(band_diff[0], -5.0, epsilon = WRAP_TOL);
        assert_abs_diff_eq!(band_diff[1], -5.0, epsilon = WRAP_TOL);
        assert_abs_diff_eq!(band_diff[2], 0.0, epsilon = WRAP_TOL);

        let mut mic_diff = array![raw, 0.0, 0.0];
        mic::wrap_difference(&mi, &mut mic_diff);
        assert_abs_diff_eq!(mic_diff[0], -5.0, epsilon = WRAP_TOL);

        let path = linear_path(
            array![0.0, 0.0, 0.0].view(),
            array![raw, 0.0, 0.0].view(),
            3,
            Some(&mi),
        )
        .expect("linear path");
        assert_abs_diff_eq!(path[(1, 0)], -2.5, epsilon = WRAP_TOL);

        let mut band = array![[0.0, 0.0, 0.0], [raw - 4.0, 0.0, 0.0], [raw, 0.0, 0.0]];
        reparametrize(&mut band, None, Some(&band_cell), Reparametrization::Linear);
        let mut mid_step = &band.row(1) - &band.row(0);
        band_cell.minimum_image(&mut mid_step);
        assert_abs_diff_eq!(mid_step[0].abs(), 2.5, epsilon = WRAP_TOL);

        let periodic_sym = PeriodicSymmetryOperations::new(SymmetryOperations::default(), mi);
        let sym_disp = periodic_sym
            .displacement(array![0.0, 0.0, 0.0].view(), array![raw, 0.0, 0.0].view())
            .expect("periodic symmetry displacement");
        assert_abs_diff_eq!(sym_disp.dot(&sym_disp).sqrt(), 5.0, epsilon = WRAP_TOL);

        #[cfg(feature = "ohtst")]
        {
            let perm = rgsaddle::ohtst::PermutationDistance::new(vec![0]).with_cell(mi);
            let dist = perm
                .distance(array![0.0, 0.0, 0.0].view(), array![raw, 0.0, 0.0].view())
                .expect("permutation distance");
            assert_abs_diff_eq!(dist, 5.0, epsilon = WRAP_TOL);
        }
    }

    let shifted = array![0.0, 0.0, 0.0, 21.0, 0.0, 0.0, 31.0, 0.0, 0.0];
    let pattern =
        SparsityPattern::from_positions(shifted.view(), Some(&mi), 2.0, 0.0).expect("pattern");
    assert_eq!(pattern.neighbors(0), &[1, 2]);
}

#[test]
fn triclinic_separations_past_one_and_a_half_boxes_wrap_to_minimum_image() {
    let rows = [
        [10.0, 0.0, 0.0],
        [5.0, 8.660_254_037_844_386, 0.0],
        [1.0, 2.0, 9.0],
    ];
    let band_cell = BandCell(rows);
    let mi = band_cell.as_minimage().expect("triclinic minimage cell");
    assert!(!mi.is_ortho());

    for multiple in [2.5_f64, 3.5_f64] {
        let raw = [multiple * rows[0][0], 0.0, 0.0];
        let dr = mi.displacement([0.0, 0.0, 0.0], raw);
        assert_abs_diff_eq!(dr[0], -5.0, epsilon = WRAP_TOL);
        assert_abs_diff_eq!(dr[1], 0.0, epsilon = WRAP_TOL);
        assert_abs_diff_eq!(dr[2], 0.0, epsilon = WRAP_TOL);

        let euc = mi.displacement_euclidean([0.0, 0.0, 0.0], raw);
        let euc_norm = (euc[0] * euc[0] + euc[1] * euc[1] + euc[2] * euc[2]).sqrt();
        assert_abs_diff_eq!(euc_norm, 5.0, epsilon = WRAP_TOL);

        let periodic_sym = PeriodicSymmetryOperations::new(SymmetryOperations::default(), mi);
        let sym_disp = periodic_sym
            .displacement(
                array![0.0, 0.0, 0.0].view(),
                array![raw[0], raw[1], raw[2]].view(),
            )
            .expect("triclinic periodic symmetry displacement");
        assert_abs_diff_eq!(sym_disp.dot(&sym_disp).sqrt(), 5.0, epsilon = WRAP_TOL);

        #[cfg(feature = "ohtst")]
        {
            let perm = rgsaddle::ohtst::PermutationDistance::new(vec![0]).with_cell(mi);
            let dist = perm
                .distance(
                    array![0.0, 0.0, 0.0].view(),
                    array![raw[0], raw[1], raw[2]].view(),
                )
                .expect("triclinic permutation distance");
            assert_abs_diff_eq!(dist, 5.0, epsilon = WRAP_TOL);
        }
    }

    let local = [1.25_f64, -0.75_f64, 0.5_f64];
    let shifted = [
        local[0] + 2.0 * rows[0][0] - 3.0 * rows[1][0] + 2.0 * rows[2][0],
        local[1] + 2.0 * rows[0][1] - 3.0 * rows[1][1] + 2.0 * rows[2][1],
        local[2] + 2.0 * rows[0][2] - 3.0 * rows[1][2] + 2.0 * rows[2][2],
    ];
    let mut band_diff = array![shifted[0], shifted[1], shifted[2]];
    band_cell.minimum_image(&mut band_diff);
    for c in 0..3 {
        assert_abs_diff_eq!(band_diff[c], local[c], epsilon = WRAP_TOL);
    }
    let from_mic = MicCell::from_ase(rows).expect("triclinic mic cell");
    let mut mic_diff = array![shifted[0], shifted[1], shifted[2]];
    mic::wrap_difference(&from_mic, &mut mic_diff);
    for c in 0..3 {
        assert_abs_diff_eq!(mic_diff[c], local[c], epsilon = WRAP_TOL);
    }
}
