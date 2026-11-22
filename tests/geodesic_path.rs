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
    assert!(report.converged);
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
    assert!(report.converged);
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
        seed: 1,
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
            std::ptr::null(),
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
            std::ptr::null(),
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
            std::ptr::null(),
            &config,
            out.as_mut_ptr(),
            &mut length,
            &mut lower,
            &mut upper,
        )
    };
    assert_eq!(bad, RGSADDLE_INVALID_PARAMETER);
}

/// Eq. (7) for r_e = 2, alpha 1.7, beta 0.01.
fn pair_q(r: f64) -> f64 {
    (-MORSE_ALPHA * (r - 2.0) / 2.0).exp() + MORSE_BETA * 2.0 / r
}

fn periodic_box() -> minimage::Cell {
    minimage::Cell::ortho(10.0, 10.0, 10.0).unwrap()
}

/// Atom 0 stays at x = 2. Atom 1 is 1.5 from it at x = 0.5 and 2.5 from it at
/// x = -0.5. The wrapped endpoint stores that last position as x = 9.5, so a
/// naive difference of the two endpoints is 9.0 along x.
fn crossing_ends(wrapped: bool) -> (Array1<f64>, Array1<f64>) {
    let end = if wrapped { 9.5 } else { -0.5 };
    (
        Array1::from_vec(vec![2.0, 5.0, 5.0, 0.5, 5.0, 5.0]),
        Array1::from_vec(vec![2.0, 5.0, 5.0, end, 5.0, 5.0]),
    )
}

fn pair_distance(row: ndarray::ArrayView1<f64>, cell: &minimage::Cell) -> f64 {
    let d = cell.displacement([row[0], row[1], row[2]], [row[3], row[4], row[5]]);
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

#[test]
fn a_periodic_path_across_the_box_relaxes_like_the_unwrapped_one() {
    let cell = periodic_box();
    let periodic = MorsePairChart::new(Array1::from_vec(vec![1.0, 1.0]))
        .unwrap()
        .with_cell(cell);
    let free = morse();
    // Bowed through y and z. Atom 1 moves from 0.5 to -0.5 in x.
    let rows = |wrap: bool| {
        let (a, b) = crossing_ends(wrap);
        let xs = [0.5, 0.17, -0.17];
        let ys = [5.0, 5.7, 4.4, 5.0];
        let zs = [5.0, 5.3, 4.8, 5.0];
        let mut path = Array2::zeros((4, 6));
        path.row_mut(0).assign(&a);
        path.row_mut(3).assign(&b);
        for i in 1..3 {
            let x = if wrap && xs[i] < 0.0 {
                xs[i] + 10.0
            } else {
                xs[i]
            };
            path.row_mut(i)
                .assign(&Array1::from_vec(vec![2.0, 5.0, 5.0, x, ys[i], zs[i]]));
        }
        path
    };
    let config = GeodesicConfig::default();
    let (wrapped_path, wrapped) = geodesic_path(rows(true).view(), &periodic, &config).unwrap();
    let (free_path, unwrapped) = geodesic_path(rows(false).view(), &free, &config).unwrap();
    let expect = pair_q(1.5) - pair_q(2.5);
    assert!((wrapped.length - expect).abs() < 1e-4, "{}", wrapped.length);
    assert!(
        (wrapped.length - unwrapped.length).abs() < 1e-8,
        "{} vs {}",
        wrapped.length,
        unwrapped.length
    );
    assert_eq!(wrapped.images, unwrapped.images);
    for (u, v) in wrapped_path.iter().zip(free_path.iter()) {
        let d = u - v;
        let nearest = d - 10.0 * (d / 10.0).round();
        assert!(nearest.abs() < 1e-6, "{u} vs {v}");
    }
    // Read as free space, the same wrapped rows are a path of length ~q(1.5) - q(7.5).
    let naive = geodesic_lengths(rows(true).view(), &free, 10).unwrap();
    let proper = geodesic_lengths(rows(true).view(), &periodic, 10).unwrap();
    assert!(
        naive.length > proper.length + 0.2,
        "{} {}",
        naive.length,
        proper.length
    );
}

#[test]
fn periodic_interpolation_never_takes_the_long_way_around() {
    let cell = periodic_box();
    let chart = MorsePairChart::new(Array1::from_vec(vec![1.0, 1.0]))
        .unwrap()
        .with_cell(cell);
    let (a, b) = crossing_ends(true);
    let (path, report) = geodesic_interpolate(
        a.view(),
        b.view(),
        7,
        &chart,
        &GeodesicConfig::default().with_seed(3),
    )
    .unwrap();
    let expect = pair_q(1.5) - pair_q(2.5);
    assert!((report.length - expect).abs() < 1e-4, "{}", report.length);
    assert!((report.lower - expect).abs() < 1e-4, "{}", report.lower);
    let cell = periodic_box();
    let mut previous = 1.5 - 1e-9;
    for row in path.rows() {
        let r = pair_distance(row, &cell);
        assert!((1.5 - 1e-3..=2.5 + 1e-3).contains(&r), "pair distance {r}");
        assert!(r + 1e-3 >= previous, "{r} after {previous}");
        previous = r;
    }
    assert_eq!(path.row(0), a.view());
    assert_eq!(path.row(path.nrows() - 1), b.view());
}

#[test]
fn a_cell_needs_three_cartesians_per_atom() {
    let chart = MorsePairChart::new(Array1::from_vec(vec![1.0, 1.0]))
        .unwrap()
        .with_cell(periodic_box());
    let path = Array2::zeros((3, 5));
    assert!(matches!(
        geodesic_path(path.view(), &chart, &GeodesicConfig::default()),
        Err(SaddleError::Shape(_))
    ));
}

#[test]
#[cfg(feature = "capi")]
fn c_abi_periodic_chart_and_morse_follow_the_minimum_image() {
    use std::os::raw::c_void;

    use rgsaddle::capi::{
        RGSADDLE_OK, RGSADDLE_SHAPE, RgsaddleGeodesicConfig, rgsaddle_geodesic_config_default,
        rgsaddle_geodesic_morse, rgsaddle_geodesic_path,
    };

    // The host chart wraps through the same minimage cell as the library.
    extern "C" fn periodic_morse(
        ctx: *mut c_void,
        x: *const f64,
        dof: i64,
        q: *mut f64,
        b: *mut f64,
        n_coords: i64,
    ) -> i32 {
        if dof != 6 || n_coords != 1 || ctx.is_null() || x.is_null() || q.is_null() || b.is_null() {
            return 1;
        }
        unsafe {
            let cell = &*(ctx as *const minimage::Cell);
            let x = std::slice::from_raw_parts(x, 6);
            let d = cell.displacement([x[0], x[1], x[2]], [x[3], x[4], x[5]]);
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            if r <= 1e-12 {
                return 1;
            }
            let e = (-0.85 * (r - 2.0)).exp();
            *q = e + 0.02 / r;
            let scale = (-0.85 * e - 0.02 / (r * r)) / r;
            let b = std::slice::from_raw_parts_mut(b, 6);
            for c in 0..3 {
                b[c] = -scale * d[c];
                b[3 + c] = scale * d[c];
            }
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
        seed: 0,
    };
    assert_eq!(
        unsafe { rgsaddle_geodesic_config_default(&mut config) },
        RGSADDLE_OK
    );

    let lattice = [10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0];
    let host_cell = periodic_box();
    let (a, b) = crossing_ends(true);
    let n_images = 6usize;
    let mut path = Array2::zeros((n_images, 6));
    for i in 0..n_images {
        let t = i as f64 / (n_images - 1) as f64;
        // The wrapped way in x. y and z bow the interior so there is work to do.
        let bow = if i == 0 || i == n_images - 1 {
            0.0
        } else {
            0.5
        };
        path.row_mut(i).assign(&Array1::from_vec(vec![
            2.0,
            5.0,
            5.0,
            0.5 - t + if 0.5 - t < 0.0 { 10.0 } else { 0.0 },
            5.0 + bow,
            5.0,
        ]));
    }
    path.row_mut(0).assign(&a);
    path.row_mut(n_images - 1).assign(&b);
    let mut out = vec![0.0; path.len()];
    let (mut length, mut lower, mut upper) = (0.0, 0.0, 0.0);
    let rc = unsafe {
        rgsaddle_geodesic_path(
            n_images as i64,
            6,
            path.as_slice().unwrap().as_ptr(),
            1,
            Some(periodic_morse),
            &host_cell as *const minimage::Cell as *mut c_void,
            lattice.as_ptr(),
            &config,
            out.as_mut_ptr(),
            &mut length,
            &mut lower,
            &mut upper,
        )
    };
    assert_eq!(rc, RGSADDLE_OK);
    let expect = pair_q(1.5) - pair_q(2.5);
    assert!((length - expect).abs() < 1e-4, "{length}");
    for row in out.chunks(6) {
        let r = pair_distance(ndarray::ArrayView1::from(row), &host_cell);
        assert!((1.5 - 1e-3..=2.5 + 1e-3).contains(&r), "pair distance {r}");
    }

    let shape = unsafe {
        rgsaddle_geodesic_path(
            n_images as i64,
            5,
            path.as_slice().unwrap().as_ptr(),
            1,
            Some(periodic_morse),
            std::ptr::null_mut(),
            lattice.as_ptr(),
            &config,
            out.as_mut_ptr(),
            &mut length,
            &mut lower,
            &mut upper,
        )
    };
    assert_eq!(shape, RGSADDLE_SHAPE);

    let radii = [1.0, 1.0];
    let mut morse_out = vec![0.0; 5 * 6];
    let rc = unsafe {
        rgsaddle_geodesic_morse(
            5,
            2,
            a.as_ptr(),
            b.as_ptr(),
            radii.as_ptr(),
            lattice.as_ptr(),
            &config,
            morse_out.as_mut_ptr(),
            &mut length,
            &mut lower,
            &mut upper,
        )
    };
    assert_eq!(rc, RGSADDLE_OK);
    assert!((length - expect).abs() < 1e-4, "{length}");
    for row in morse_out.chunks(6) {
        let r = pair_distance(ndarray::ArrayView1::from(row), &host_cell);
        assert!((1.5 - 1e-3..=2.5 + 1e-3).contains(&r), "pair distance {r}");
    }
}

#[test]
fn a_chart_whose_wilson_matrix_cannot_be_allocated_is_a_shape_error() {
    struct Huge;
    impl GeodesicChart for Huge {
        fn n_coords(&self) -> usize {
            usize::MAX / 2
        }

        fn eval(
            &self,
            _x: ArrayView1<f64>,
            _q: &mut Array1<f64>,
            _b: &mut Array2<f64>,
        ) -> Result<(), SaddleError> {
            unreachable!("the size check comes before any allocation")
        }
    }
    let path = Array2::zeros((3, 4));
    assert!(matches!(
        geodesic_lengths(path.view(), &Huge, 4),
        Err(SaddleError::Shape(_))
    ));
}

#[test]
#[cfg(feature = "capi")]
fn c_abi_size_products_that_overflow_return_an_error_status() {
    use std::os::raw::c_void;
    use std::ptr::NonNull;

    use rgsaddle::capi::{
        RGSADDLE_INVALID_PARAMETER, rgsaddle_geodesic_morse, rgsaddle_geodesic_path,
    };

    extern "C" fn never_called(
        _ctx: *mut c_void,
        _x: *const f64,
        _dof: i64,
        _q: *mut f64,
        _b: *mut f64,
        _n_coords: i64,
    ) -> i32 {
        panic!("an overflowing size must be rejected before the chart runs");
    }

    // Nothing behind these pointers is read or written: every call below must
    // fail on its size arithmetic first.
    let input = NonNull::<f64>::dangling().as_ptr();
    let output = NonNull::<f64>::dangling().as_ptr();
    let (mut length, mut lower, mut upper) = (0.0, 0.0, 0.0);
    let mut path_call = |n_images: i64, dof: i64, n_coords: i64| unsafe {
        rgsaddle_geodesic_path(
            n_images,
            dof,
            input,
            n_coords,
            Some(never_called),
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            output,
            &mut length,
            &mut lower,
            &mut upper,
        )
    };
    // n_images * dof overflows usize.
    assert_eq!(path_call(i64::MAX, 3, 1), RGSADDLE_INVALID_PARAMETER);
    // n_images * dof is 2^59 and under the slice limit, but n_coords * dof is 2^64.
    assert_eq!(path_call(2, 1 << 58, 64), RGSADDLE_INVALID_PARAMETER);
    // n_images * dof fits in usize but is more bytes than a slice may span.
    assert_eq!(path_call(2, 1 << 62, 1), RGSADDLE_INVALID_PARAMETER);
    // n_images * dof is fine and n_coords * dof is more bytes than a slice may span.
    assert_eq!(path_call(2, 1 << 58, 1 << 5), RGSADDLE_INVALID_PARAMETER);

    let radii = NonNull::<f64>::dangling().as_ptr();
    let mut morse_call = |n_images: i64, n_atoms: i64| unsafe {
        rgsaddle_geodesic_morse(
            n_images,
            n_atoms,
            input,
            input,
            radii,
            std::ptr::null(),
            std::ptr::null(),
            output,
            &mut length,
            &mut lower,
            &mut upper,
        )
    };
    // 3 * n_atoms overflows usize.
    assert_eq!(morse_call(2, i64::MAX), RGSADDLE_INVALID_PARAMETER);
    // 3 * n_atoms fits, n_images * 3 * n_atoms overflows.
    assert_eq!(morse_call(i64::MAX, 1 << 40), RGSADDLE_INVALID_PARAMETER);
    // Both of those fit, n_pairs * 3 * n_atoms does not: about 2^63 pairs of 3 * 2^32 doubles.
    assert_eq!(morse_call(2, 1 << 32), RGSADDLE_INVALID_PARAMETER);
    // n_atoms * (n_atoms - 1) overflows usize.
    assert_eq!(
        morse_call(2, (1 << 32) + (1 << 31)),
        RGSADDLE_INVALID_PARAMETER
    );
}

#[test]
fn a_run_that_exhausts_max_sweeps_is_reported_as_not_converged() {
    let initial = bowed_flat();
    let tight = GeodesicConfig::default()
        .with_max_sweeps(1)
        .with_step_tol(1e-12);
    let (path, report) = geodesic_path(initial.view(), &Flat, &tight).unwrap();
    assert!(!report.converged, "one sweep cannot settle a bowed path");
    assert_eq!(report.sweeps, 1);
    assert!(path.iter().all(|v| v.is_finite()));
    assert!(report.length < geodesic_lengths(initial.view(), &Flat, 10).unwrap().length);

    let (_, settled) = geodesic_path(
        initial.view(),
        &Flat,
        &GeodesicConfig::default().with_step_tol(1e-12),
    )
    .unwrap();
    assert!(settled.converged);

    let ends = bowed_morse();
    let (_, interpolated) = geodesic_interpolate(
        ends.row(0),
        ends.row(ends.nrows() - 1),
        6,
        &morse(),
        &GeodesicConfig::default()
            .with_max_sweeps(1)
            .with_step_tol(1e-14),
    )
    .unwrap();
    assert!(!interpolated.converged);
}

#[test]
#[cfg(feature = "capi")]
fn c_abi_returns_not_converged_with_the_last_iterate() {
    use std::os::raw::c_void;

    use rgsaddle::capi::{
        RGSADDLE_NOT_CONVERGED, RGSADDLE_OK, RgsaddleGeodesicConfig,
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
        if dof != 2 || n_coords != 2 {
            return 1;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(x, q, 2);
            *b = 1.0;
            *b.add(1) = 0.0;
            *b.add(2) = 0.0;
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
        seed: 0,
    };
    assert_eq!(
        unsafe { rgsaddle_geodesic_config_default(&mut config) },
        RGSADDLE_OK
    );
    config.max_sweeps = 1;
    config.step_tol = 1e-12;

    let initial = bowed_flat();
    let mut out = vec![-1.0; initial.len()];
    let (mut length, mut lower, mut upper) = (0.0, 0.0, 0.0);
    let rc = unsafe {
        rgsaddle_geodesic_path(
            initial.nrows() as i64,
            2,
            initial.as_slice().unwrap().as_ptr(),
            2,
            Some(flat_chart),
            std::ptr::null_mut(),
            std::ptr::null(),
            &config,
            out.as_mut_ptr(),
            &mut length,
            &mut lower,
            &mut upper,
        )
    };
    assert_eq!(rc, RGSADDLE_NOT_CONVERGED);
    assert!(out.iter().all(|v| v.is_finite() && *v != -1.0), "{out:?}");
    assert!(length.is_finite() && length > 0.0);

    config.max_sweeps = 40;
    let rc = unsafe {
        rgsaddle_geodesic_path(
            initial.nrows() as i64,
            2,
            initial.as_slice().unwrap().as_ptr(),
            2,
            Some(flat_chart),
            std::ptr::null_mut(),
            std::ptr::null(),
            &config,
            out.as_mut_ptr(),
            &mut length,
            &mut lower,
            &mut upper,
        )
    };
    assert_eq!(rc, RGSADDLE_OK);

    config.max_sweeps = 1;
    config.step_tol = 1e-14;
    let ends = bowed_morse();
    let reactant = ends.row(0).to_vec();
    let product = ends.row(ends.nrows() - 1).to_vec();
    let mut morse_out = vec![-1.0; 6 * 6];
    let rc = unsafe {
        rgsaddle_geodesic_morse(
            6,
            2,
            reactant.as_ptr(),
            product.as_ptr(),
            [1.0, 1.0].as_ptr(),
            std::ptr::null(),
            &config,
            morse_out.as_mut_ptr(),
            &mut length,
            &mut lower,
            &mut upper,
        )
    };
    assert_eq!(rc, RGSADDLE_NOT_CONVERGED);
    assert!(morse_out.iter().all(|v| v.is_finite() && *v != -1.0));
}
