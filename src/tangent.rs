//! Band tangents, ported from eOn NEBTangent.cpp with identical
//! branch structure.
//!
//! A tangent is a unit vector in the 3N configuration space of one band
//! image. [`TangentKind::Simple`] and [`TangentKind::Improved`] are built
//! from the two neighbour displacements and the three image energies.
//! [`TangentKind::Improved`] is ASE's `method="improvedtangent"` in
//! `ase.mep.NEB` and eOn's improved `NEBTangent`.
//! [`TangentKind::Spline`] is a property of the whole band: the unit
//! `dx/ds` of a not-a-knot cubic through the images.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::band::Cell;
use crate::error::SaddleError;
use crate::reparam::{Spline, SplineEnds};

/// Which tangent estimate the band uses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TangentKind {
    /// Mills, Jónsson, and Schenter, Surf. Sci. 324, 305 (1995),
    /// doi:10.1016/0039-6028(94)00731-4: the forward difference alone,
    /// `R[i+1] - R[i]`, normalised.
    Simple,
    /// Henkelman and Jónsson, J. Chem. Phys. 113, 9978 (2000),
    /// doi:10.1063/1.1323224: the segment toward the higher neighbour at
    /// monotonic points, and an energy-weighted blend of both segments at
    /// extrema. This is the default.
    #[default]
    Improved,
    /// Makri, Ortner, and Kermode, J. Chem. Phys. 150, 094109 (2019),
    /// doi:10.1063/1.5064465: `dx/ds` of the not-a-knot cubic through the
    /// images, normalized, with `s` the normalized cumulative chord length.
    /// [`compute_tangent`] refuses it. [`spline_tangents`] builds it for
    /// the whole band. ASE's `NEB(method='spline')` uses this tangent and
    /// the spline curvature for the spring.
    Spline,
}

/// Unit tangent at one interior image.
///
/// `pos_diff_next` is `R[i+1] - R[i]` and `pos_diff_prev` is
/// `R[i] - R[i-1]`, both flat `3N` vectors in Angstrom, atom-major
/// `[x0, y0, z0, x1, ...]` (the host minimum-images both when periodic).
/// `energy`, `energy_prev` and `energy_next` are the energies in eV of
/// image `i`, `i-1` and `i+1`, in that order. [`TangentKind::Simple`] ignores
/// them. The result has length `3N` and unit norm.
///
/// # Errors
///
/// [`SaddleError::Invalid`] when the weighted tangent, the forward segment
/// and the backward segment all have a norm under `1e-10` Angstrom or are not
/// finite, which is an image sitting on both neighbours.
///
/// The arithmetic lives on [`crate::stencil::unit_tangent`].
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use rgsaddle::tangent::{TangentKind, compute_tangent};
///
/// // A collinear image on an uphill segment: the improved tangent is the
/// // forward segment, normalised.
/// let next = array![0.0, 0.0, 2.0];
/// let prev = array![0.0, 0.0, 1.0];
/// let t = compute_tangent(TangentKind::Improved, next.view(), prev.view(), 0.0, -1.0, 1.0)?;
/// assert_eq!(t, array![0.0, 0.0, 1.0]);
/// # Ok::<(), rgsaddle::SaddleError>(())
/// ```
pub fn compute_tangent(
    kind: TangentKind,
    pos_diff_next: ArrayView1<f64>,
    pos_diff_prev: ArrayView1<f64>,
    energy: f64,
    energy_prev: f64,
    energy_next: f64,
) -> Result<Array1<f64>, SaddleError> {
    let rule = match kind {
        TangentKind::Simple => crate::stencil::TangentRule::Simple,
        TangentKind::Improved => crate::stencil::TangentRule::Improved,
        TangentKind::Spline => {
            return Err(SaddleError::Invalid(
                "a spline tangent is a property of the whole band; use spline_tangents".into(),
            ));
        }
    };
    crate::stencil::unit_tangent(
        rule,
        pos_diff_next,
        pos_diff_prev,
        energy,
        energy_prev,
        energy_next,
    )
}

/// Unit tangents and curvature of the not-a-knot cubic through a band.
///
/// `parameter` is the normalized cumulative chord length, 0 at the first
/// image and 1 at the last. `tangents` are `dx/ds` at unit Euclidean
/// length. `curvature` is `d²x/ds²`. Both arrays have one row per image,
/// endpoints included, in the caller's length units (the tangent is
/// unitless and the curvature is one over length).
#[derive(Clone, Debug, PartialEq)]
pub struct SplineTangents {
    /// Normalized cumulative chord length, one entry per image.
    pub parameter: Vec<f64>,
    /// Unit `dx/ds` at every image, shape `(images, 3N)`.
    pub tangents: Array2<f64>,
    /// `d²x/ds²` at every image, shape `(images, 3N)`.
    pub curvature: Array2<f64>,
}

/// Fit [`SplineTangents`] through `positions`.
///
/// `positions` is `(images, 3N)`, at least two images. With `cell` set,
/// each chord is the minimum image; the spline still interpolates the
/// Cartesian rows. Three images give the interpolating parabola and four
/// or more the not-a-knot cubic, the same end condition as ASE's
/// `CubicSpline(..., bc_type='not-a-knot')`.
///
/// # Errors
///
/// [`SaddleError::Shape`] when the band is shorter than two images or a
/// row is not `3N`. [`SaddleError::NonFinite`] when a coordinate is not
/// finite. [`SaddleError::Invalid`] when the path has no length, an image
/// repeats its neighbour, the spline system has no solution, or `dx/ds`
/// vanishes.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use rgsaddle::tangent::spline_tangents;
///
/// let positions = array![
///     [0.0, 0.0, 0.0],
///     [1.0, 0.0, 0.0],
///     [2.0, 0.0, 0.0],
/// ];
/// let frame = spline_tangents(positions.view(), None)?;
/// assert!((frame.tangents[[1, 0]] - 1.0).abs() < 1e-12);
/// assert!(frame.curvature[[1, 0]].abs() < 1e-12);
/// # Ok::<(), rgsaddle::SaddleError>(())
/// ```
pub fn spline_tangents(
    positions: ArrayView2<f64>,
    cell: Option<&Cell>,
) -> Result<SplineTangents, SaddleError> {
    let (parameter, spline) = path_spline(positions, cell)?;
    let dof = positions.ncols();
    let mut tangents = Array2::zeros((positions.nrows(), dof));
    let mut curvature = Array2::zeros((positions.nrows(), dof));
    for (i, &s) in parameter.iter().enumerate() {
        let sample = spline.sample(s)?;
        let norm = sample.first.dot(&sample.first).sqrt();
        if !(norm.is_finite() && norm > 0.0) {
            return Err(SaddleError::Invalid(
                "spline tangent derivative vanished".into(),
            ));
        }
        let unit = &sample.first / norm;
        tangents.row_mut(i).assign(&unit);
        curvature.row_mut(i).assign(&sample.second);
    }
    Ok(SplineTangents {
        parameter,
        tangents,
        curvature,
    })
}

/// Place the interior images at equal steps of the normalized chord length.
///
/// The curve is the same not-a-knot cubic as [`spline_tangents`]. Image
/// `i` of `n` moves to parameter `i / (n - 1)`. The first and last rows
/// stay where they are. This is the redistribution of ASE's
/// `NEB(method='string')` with the identity preconditioner: uniform in
/// the chord-length parameter, not equal arc length of the spline.
///
/// # Errors
///
/// [`SaddleError::Shape`] when there are fewer than three images. The
/// other errors are those of [`spline_tangents`].
pub fn spline_resample(
    positions: ArrayView2<f64>,
    cell: Option<&Cell>,
) -> Result<Array2<f64>, SaddleError> {
    if positions.nrows() < 3 {
        return Err(SaddleError::Shape(
            "string redistribution needs at least three images".into(),
        ));
    }
    let spline = path_spline(positions, cell)?.1;
    let mut out = positions.to_owned();
    let last = (positions.nrows() - 1) as f64;
    for i in 1..positions.nrows() - 1 {
        let sample = spline.sample(i as f64 / last)?;
        out.row_mut(i).assign(&sample.value);
    }
    Ok(out)
}

/// Normalized chord-length parameter and the Cartesian spline on it.
fn path_spline(
    positions: ArrayView2<f64>,
    cell: Option<&Cell>,
) -> Result<(Vec<f64>, Spline), SaddleError> {
    let rows = positions.nrows();
    let dof = positions.ncols();
    if rows < 2 || dof == 0 || !dof.is_multiple_of(3) {
        return Err(SaddleError::Shape(
            "a spline tangent needs at least two images of 3N coordinates".into(),
        ));
    }
    if !positions.iter().all(|value| value.is_finite()) {
        return Err(SaddleError::NonFinite("spline tangent"));
    }
    let mut parameter = vec![0.0; rows];
    let mut acc = 0.0;
    for (i, slot) in parameter.iter_mut().enumerate().skip(1) {
        let mut step = (&positions.row(i) - &positions.row(i - 1)).to_owned();
        if let Some(cell) = cell {
            cell.minimum_image(&mut step);
        }
        let chord = step.dot(&step).sqrt();
        if !chord.is_finite() {
            return Err(SaddleError::NonFinite("spline tangent"));
        }
        acc += chord;
        *slot = acc;
    }
    if !(acc.is_finite() && acc > 0.0) {
        return Err(SaddleError::Invalid(
            "spline tangent path has no length".into(),
        ));
    }
    for value in &mut parameter {
        *value /= acc;
    }
    for pair in parameter.windows(2) {
        if pair[1] <= pair[0] {
            return Err(SaddleError::Invalid(
                "spline tangent knots must be strictly increasing; a repeated image has no tangent"
                    .into(),
            ));
        }
    }
    let spline = Spline::new(&parameter, &positions.to_owned(), SplineEnds::NotAKnot)?;
    Ok((parameter, spline))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn improved_picks_uphill_forward() {
        let next = array![1.0, 0.0];
        let prev = array![0.0, 1.0];
        let t = compute_tangent(
            TangentKind::Improved,
            next.view(),
            prev.view(),
            1.0,
            0.0,
            2.0,
        )
        .unwrap();
        assert_abs_diff_eq!(t[0], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(t[1], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn improved_picks_downhill_backward() {
        let next = array![1.0, 0.0];
        let prev = array![0.0, 1.0];
        let t = compute_tangent(
            TangentKind::Improved,
            next.view(),
            prev.view(),
            1.0,
            2.0,
            0.0,
        )
        .unwrap();
        assert_abs_diff_eq!(t[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(t[1], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn improved_weights_extremum_toward_steeper_side() {
        let next = array![1.0, 0.0];
        let prev = array![0.0, 1.0];
        // Maximum: prev side steeper (|dE_prev| = 2 > |dE_next| = 1).
        let t = compute_tangent(
            TangentKind::Improved,
            next.view(),
            prev.view(),
            3.0,
            1.0,
            2.0,
        )
        .unwrap();
        // energy_diff_prev (-2) < energy_diff_next (-1): next*max + prev*min.
        let expected = (array![1.0, 0.0] * 2.0 + array![0.0, 1.0] * 1.0) / 5.0f64.sqrt();
        assert_abs_diff_eq!(t[0], expected[0], epsilon = 1e-12);
        assert_abs_diff_eq!(t[1], expected[1], epsilon = 1e-12);
    }

    #[test]
    fn zero_tangent_falls_back_to_forward_difference() {
        let next = array![0.5, 0.0];
        let prev = array![-0.5, 0.0];
        // Extremum with equal, zero energy differences: combination is
        // zero-weighted; fallback normalizes the forward difference.
        let t = compute_tangent(
            TangentKind::Improved,
            next.view(),
            prev.view(),
            1.0,
            1.0,
            1.0,
        )
        .unwrap();
        assert_abs_diff_eq!(t[0], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn zero_forward_difference_falls_back_to_the_backward_one() {
        let next = array![0.0, 0.0];
        let prev = array![0.0, 0.3];
        let t =
            compute_tangent(TangentKind::Simple, next.view(), prev.view(), 1.0, 0.0, 2.0).unwrap();
        assert_abs_diff_eq!(t[1], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(t.dot(&t), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn coincident_images_have_no_tangent() {
        let zero = array![0.0, 0.0, 0.0];
        let err = compute_tangent(
            TangentKind::Improved,
            zero.view(),
            zero.view(),
            1.0,
            1.0,
            1.0,
        )
        .unwrap_err();
        assert!(matches!(err, SaddleError::Invalid(_)), "{err}");
    }

    #[test]
    fn spline_tangent_of_a_line_is_the_axis_and_resample_equalizes_chords() {
        let positions = array![
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.1, 0.0, 0.0],
        ];
        let frame = spline_tangents(positions.view(), None).unwrap();
        for row in frame.tangents.rows() {
            assert_abs_diff_eq!(row[0], 1.0, epsilon = 1e-12);
            assert_abs_diff_eq!(row[1], 0.0, epsilon = 1e-12);
            assert_abs_diff_eq!(row[2], 0.0, epsilon = 1e-12);
        }
        assert!(frame.curvature.iter().all(|value| value.abs() < 1e-12));
        let moved = spline_resample(positions.view(), None).unwrap();
        assert_abs_diff_eq!(moved[[1, 0]], 1.1 / 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(moved[[2, 0]], 2.2 / 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(moved[[0, 0]], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(moved[[3, 0]], 1.1, epsilon = 1e-12);
    }

    #[test]
    fn compute_tangent_refuses_the_spline_kind() {
        let step = array![1.0, 0.0, 0.0];
        let err = compute_tangent(TangentKind::Spline, step.view(), step.view(), 0.0, 0.0, 1.0)
            .unwrap_err();
        assert!(matches!(err, SaddleError::Invalid(_)), "{err}");
    }

    #[test]
    fn a_repeated_image_has_no_spline_tangent() {
        let positions = array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let err = spline_tangents(positions.view(), None).unwrap_err();
        assert!(matches!(err, SaddleError::Invalid(_)), "{err}");
    }
}
