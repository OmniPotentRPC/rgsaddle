//! Geodesic interpolation of Zhu, Thompson, and Martinez, J. Chem. Phys.
//! 150, 164103 (2019), doi:10.1063/1.5090303.
//!
//! On a flat chart the geodesic is the straight line (Appendix A) and the
//! 3-4-5 segment has length 5. On the Morse pair chart of Eq. (7), with
//! alpha 1.7 and beta 0.01, a one-coordinate dissociation from r = 2 to
//! r = 6 has geodesic length |q(6) - q(2)| = 0.9732933967063406, the
//! equality case of Eq. (B5) because every intermediate distance is feasible.

use ndarray::{Array1, Array2, ArrayView1};
use rgsaddle::{
    GeodesicChart, GeodesicConfig, MORSE_ALPHA, MORSE_BETA, MorsePairChart, SaddleError,
    geodesic_interpolate, geodesic_lengths, geodesic_path,
};

/// Eq. (7) length from r = 2 to r = 6 with r_e = 2, alpha 1.7, beta 0.01.
const MORSE_LENGTH: f64 = 0.9732933967063406;
/// Appendix A: the straight segment from (0, 0) to (3, 4) has length 5.
const FLAT_LENGTH: f64 = 5.0;

struct Flat;

impl GeodesicChart for Flat {
    fn n_coords(&self) -> usize {
        2
    }

    fn eval(
        &self,
        x: ArrayView1<f64>,
        q: &mut Array1<f64>,
        b: &mut ndarray::Array2<f64>,
    ) -> Result<(), SaddleError> {
        q.assign(&x);
        b.fill(0.0);
        b[(0, 0)] = 1.0;
        b[(1, 1)] = 1.0;
        Ok(())
    }
}

fn bowed_flat() -> Array2<f64> {
    Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 2.0, 2.0, 0.5, 3.0, 4.0]).unwrap()
}

fn morse() -> MorsePairChart {
    MorsePairChart::new(Array1::from_vec(vec![1.0, 1.0])).unwrap()
}

fn bowed_morse() -> Array2<f64> {
    // Second atom leaves the x axis, so a Cartesian chord cuts inside the
    // circle and Eq. (17) exceeds |q(6) - q(2)|.
    Array2::from_shape_vec(
        (4, 6),
        vec![
            0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0,
            0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0,
        ],
    )
    .unwrap()
}

#[test]
fn a_flat_geodesic_is_the_straight_segment_of_length_five() {
    let initial = bowed_flat();
    let before = geodesic_lengths(initial.view(), &Flat, 10).unwrap();
    assert!(
        before.length > FLAT_LENGTH + 0.5,
        "bowed length {}",
        before.length
    );
    let (path, report) = geodesic_path(initial.view(), &Flat, &GeodesicConfig::default()).unwrap();
    assert!(
        (report.length - FLAT_LENGTH).abs() < 1e-6,
        "{}",
        report.length
    );
    assert!(report.sweeps >= 1, "sweeps {}", report.sweeps);
    assert_eq!(report.images, initial.nrows());
    assert!(
        (report.lower - FLAT_LENGTH).abs() < 1e-6,
        "{}",
        report.lower
    );
    assert!(report.upper <= 1.1 * report.length + 1e-8);
    assert!((path[[0, 0]]).abs() < 1e-12 && (path[[0, 1]]).abs() < 1e-12);
    assert!((path[[3, 0]] - 3.0).abs() < 1e-12 && (path[[3, 1]] - 4.0).abs() < 1e-12);
    for i in 1..3 {
        let (x, y) = (path[[i, 0]], path[[i, 1]]);
        // The image lies on the line through (0, 0) and (3, 4).
        assert!((4.0 * x - 3.0 * y).abs() < 1e-4, "image {i} at {x}, {y}");
        assert!((0.0..=3.0).contains(&x), "{x}");
    }
}

#[test]
fn morse_dissociation_recovers_the_eq7_length() {
    assert!((MORSE_ALPHA - 1.7).abs() < 1e-15);
    assert!((MORSE_BETA - 0.01).abs() < 1e-15);
    let chart = morse();
    let initial = bowed_morse();
    let before = geodesic_lengths(initial.view(), &chart, 10).unwrap();
    assert!(
        before.length > MORSE_LENGTH + 0.05,
        "bowed Morse length {}",
        before.length
    );
    let (path, report) = geodesic_path(initial.view(), &chart, &GeodesicConfig::default()).unwrap();
    assert!(
        (report.length - MORSE_LENGTH).abs() < 1e-4,
        "length {} lower {} upper {}",
        report.length,
        report.lower,
        report.upper
    );
    assert!(report.sweeps >= 1, "sweeps {}", report.sweeps);
    assert!(report.images >= initial.nrows());
    assert!(report.lower + 1e-8 >= 0.95 * report.length);
    assert!(report.upper <= 1.1 * report.length + 1e-6);
    assert_eq!(
        path.row(0).as_slice().unwrap(),
        initial.row(0).as_slice().unwrap()
    );
    assert_eq!(
        path.row(path.nrows() - 1).as_slice().unwrap(),
        initial.row(initial.nrows() - 1).as_slice().unwrap()
    );
}

#[test]
fn interpolate_from_the_endpoints_has_the_same_morse_length() {
    let chart = morse();
    let ends = bowed_morse();
    let (path, report) = geodesic_interpolate(
        ends.row(0),
        ends.row(ends.nrows() - 1),
        5,
        &chart,
        &GeodesicConfig::default(),
    )
    .unwrap();
    assert_eq!(path.nrows(), 5);
    assert_eq!(report.images, 5);
    assert!(
        (report.length - MORSE_LENGTH).abs() < 1e-4,
        "{}",
        report.length
    );
    assert!(report.sweeps >= 1, "sweeps {}", report.sweeps);
}

#[test]
#[cfg(feature = "capi")]
fn c_abi_morse_and_chart_match_the_published_lengths() {
    use std::os::raw::c_void;

    use rgsaddle::capi::{
        RGSADDLE_INVALID_PARAMETER, RGSADDLE_OK, RgsaddleGeodesicConfig,
        rgsaddle_geodesic_config_default, rgsaddle_geodesic_morse, rgsaddle_geodesic_path,
    };

    extern "C" fn flat_chart(
        _ctx: *mut c_void,
        x: *const f64,
        dof: i64,
        q: *mut f64,
        b: *mut f64,
        n_coords: i64,
    ) -> i32 {
        if dof != 2 || n_coords != 2 || x.is_null() || q.is_null() || b.is_null() {
            return 1;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(x, q, 2);
            for i in 0..4 {
                *b.add(i) = 0.0;
            }
            *b = 1.0;
            *b.add(3) = 1.0;
        }
        0
    }

    let mut config = RgsaddleGeodesicConfig {
        version: rgsaddle::capi::RgsaddleVersion { major: 0, minor: 0 },
        flags: 0,
        step_tol: 0.0,
        grad_tol: 0.0,
        max_move: 0.0,
        lower_ratio: 0.0,
        upper_ratio: 0.0,
        max_sweeps: 0,
        max_iterations: 0,
        memory: 0,
        bound_samples: 0,
        max_images: 0,
    };
    assert_eq!(
        unsafe { rgsaddle_geodesic_config_default(&mut config) },
        RGSADDLE_OK
    );
    assert_eq!(config.version.minor, rgsaddle::capi::RGSADDLE_ABI_MINOR);

    let initial = bowed_flat();
    let mut out = vec![0.0; initial.len()];
    let mut length = 0.0;
    let mut lower = 0.0;
    let mut upper = 0.0;
    let rc = unsafe {
        rgsaddle_geodesic_path(
            initial.nrows() as i64,
            initial.ncols() as i64,
            initial.as_slice().unwrap().as_ptr(),
            2,
            Some(flat_chart),
            std::ptr::null_mut(),
            &config,
            out.as_mut_ptr(),
            &mut length,
            &mut lower,
            &mut upper,
        )
    };
    assert_eq!(rc, RGSADDLE_OK);
    assert!((length - FLAT_LENGTH).abs() < 1e-6, "{length}");

    extern "C" fn morse_chart(
        _ctx: *mut c_void,
        x: *const f64,
        dof: i64,
        q: *mut f64,
        b: *mut f64,
        n_coords: i64,
    ) -> i32 {
        if dof != 6 || n_coords != 1 || x.is_null() || q.is_null() || b.is_null() {
            return 1;
        }
        unsafe {
            let dx = *x.add(3) - *x;
            let dy = *x.add(4) - *x.add(1);
            let dz = *x.add(5) - *x.add(2);
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            if r <= 1e-12 {
                return 1;
            }
            let exp_term = (-0.85 * (r - 2.0)).exp();
            *q = exp_term + 0.02 / r;
            let dq_dr = -0.85 * exp_term - 0.02 / (r * r);
            for i in 0..6 {
                *b.add(i) = 0.0;
            }
            let scale = dq_dr / r;
            *b = -scale * dx;
            *b.add(1) = -scale * dy;
            *b.add(2) = -scale * dz;
            *b.add(3) = scale * dx;
            *b.add(4) = scale * dy;
            *b.add(5) = scale * dz;
        }
        0
    }

    let morse_bow = bowed_morse();
    let mut bow_out = vec![0.0; morse_bow.len()];
    let bow = unsafe {
        rgsaddle_geodesic_path(
            morse_bow.nrows() as i64,
            morse_bow.ncols() as i64,
            morse_bow.as_slice().unwrap().as_ptr(),
            1,
            Some(morse_chart),
            std::ptr::null_mut(),
            &config,
            bow_out.as_mut_ptr(),
            &mut length,
            &mut lower,
            &mut upper,
        )
    };
    assert_eq!(bow, RGSADDLE_OK);
    assert!((length - MORSE_LENGTH).abs() < 1e-4, "{length}");

    let morse_path = bowed_morse();
    let reactant = morse_path.row(0).to_vec();
    let product = morse_path.row(morse_path.nrows() - 1).to_vec();
    let radii = [1.0, 1.0];
    let mut mout = vec![0.0; 5 * 6];
    let mut mlen = 0.0;
    let mrc = unsafe {
        rgsaddle_geodesic_morse(
            5,
            2,
            reactant.as_ptr(),
            product.as_ptr(),
            radii.as_ptr(),
            std::ptr::null(),
            &config,
            mout.as_mut_ptr(),
            &mut mlen,
            &mut lower,
            &mut upper,
        )
    };
    assert_eq!(mrc, RGSADDLE_OK);
    assert!((mlen - MORSE_LENGTH).abs() < 1e-4, "{mlen}");
    assert_eq!(&mout[..6], reactant.as_slice());
    assert_eq!(&mout[mout.len() - 6..], product.as_slice());

    let bad = unsafe {
        rgsaddle_geodesic_path(
            1,
            2,
            initial.as_slice().unwrap().as_ptr(),
            2,
            Some(flat_chart),
            std::ptr::null_mut(),
            &config,
            out.as_mut_ptr(),
            &mut length,
            &mut lower,
            &mut upper,
        )
    };
    assert_eq!(bad, RGSADDLE_INVALID_PARAMETER);
}
