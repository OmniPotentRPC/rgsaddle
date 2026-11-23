//! Geodesic interpolation on a caller supplied coordinate chart.
//!
//! Zhu, Thompson, and Martinez, J. Chem. Phys. 150, 164103 (2019),
//! doi:10.1063/1.5090303.
//!
//! The path length minimized here is the two point midpoint formula of
//! their Eq. (17): for each Cartesian segment, add the redundant
//! coordinate distance from each end to the image of the Cartesian
//! midpoint. On a flat chart that length is the polyline length, and
//! the geodesic is the straight line (Appendix A). The lower bound is
//! the sum of the endpoint chords in coordinate space (Eq. B5). The
//! upper bound subdivides each Cartesian segment into `bound_samples`
//! equal pieces (Eq. B18, ten pieces in the paper). `bound_samples` is
//! even, so one of those pieces ends on the Cartesian midpoint and the
//! chord sum is at least the midpoint length.
//!
//! Interior images move one at a time, forward and then back, until the
//! largest displacement of a pass falls under `step_tol` (Algorithm 2).
//! A pass that leaves the lower bound under `lower_ratio` times the
//! length, or the upper bound over `upper_ratio` times the length,
//! inserts Cartesian midpoints on the loose segments and starts again.
//! [`geodesic_interpolate`] builds the first path from the two endpoints
//! by their Algorithm 3. The target is the average of the redundant
//! coordinates, `q_mid = (q(A) + q(B)) / 2`, not the coordinates of the
//! Cartesian average. Ten starts alternate between the reactant (odd
//! starts) and the product (even starts), each with Gaussian noise of
//! scale 0.1 from the seeded generator of [`GeodesicConfig::seed`]. Each
//! start minimizes `|q(x) - q_mid|^2`, the shortest three point path
//! `[A, x, B]` is relaxed with the middle image free, and the images are
//! copied onto thirds of the path before the final relaxation.

use minimage::Cell;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1};

use crate::error::SaddleError;
use crate::stats::Rng;

/// Morse exponent of Zhu, Thompson, and Martinez, Eq. (7).
pub const MORSE_ALPHA: f64 = 1.7;
/// Morse weight of the inverse distance term, Eq. (7).
pub const MORSE_BETA: f64 = 0.01;

/// Distances at or below this are not a Morse pair. The inverse term is undefined there.
const MORSE_DISTANCE_FLOOR: f64 = 1e-12;
/// A coordinate chord shorter than this adds its length and skips its unit vector.
const CHORD_FLOOR: f64 = 1e-14;
/// Weight of each endpoint in a Cartesian midpoint.
const MIDPOINT_WEIGHT: f64 = 0.5;
/// First L-BFGS trial step, in the coordinate unit of the path.
const INITIAL_STEP: f64 = 1.0;
/// Noisy starts of Algorithm 3: odd starts begin at the reactant, even starts at the product.
const START_COUNT: usize = 10;
/// Scale of the Gaussian noise added to a start, in the path's length unit (Algorithm 3).
const START_NOISE: f64 = 0.1;

const DEFAULT_STEP_TOL: f64 = 1e-6;
const DEFAULT_GRAD_TOL: f64 = 1e-6;
const DEFAULT_MAX_MOVE: f64 = 0.5;
const DEFAULT_LOWER_RATIO: f64 = 0.95;
const DEFAULT_UPPER_RATIO: f64 = 1.1;
const DEFAULT_MAX_SWEEPS: usize = 40;
const DEFAULT_MAX_ITERATIONS: usize = 40;
const DEFAULT_MEMORY: usize = 8;
const DEFAULT_BOUND_SAMPLES: usize = 10;
const DEFAULT_MAX_IMAGES: usize = 32;
const DEFAULT_SEED: u64 = 0;

/// Redundant coordinates `q(x)` and the Wilson B matrix `dq/dx`.
///
/// `eval` writes `q` with length [`n_coords`](Self::n_coords) and `b`
/// with that many rows and one column per component of `x`. Both are
/// finite on success. [`n_coords`](Self::n_coords) does not change
/// between calls. The optimizers call `eval` from the thread that
/// entered them, and the chart must be [`Sync`] because the stepper
/// stores the objective as a shared value.
pub trait GeodesicChart: Sync {
    /// Number of redundant coordinates.
    fn n_coords(&self) -> usize;

    /// Periodic cell of the Cartesian rows, or `None` in free space.
    ///
    /// With a cell, a row holds one 3-vector per atom. Every midpoint,
    /// sample, and inserted image of the path moves along the
    /// [`minimage::Cell::displacement`] of its two ends, so a segment
    /// that crosses the box takes the short way. The chart must apply the
    /// same convention to its own distances. The cell should be
    /// tilt-reduced, the engine convention of [`minimage`].
    fn cell(&self) -> Option<&Cell> {
        None
    }

    /// Write `q(x)` and `dq/dx` (`b` is row major, one row per coordinate).
    ///
    /// # Errors
    ///
    /// [`SaddleError::Shape`] when `q` or `b` does not match `x`, and
    /// [`SaddleError::Invalid`] or [`SaddleError::NonFinite`] when the
    /// chart is undefined at `x`.
    fn eval(
        &self,
        x: ArrayView1<f64>,
        q: &mut Array1<f64>,
        b: &mut Array2<f64>,
    ) -> Result<(), SaddleError>;
}

/// Pairwise Morse coordinates of Eq. (7).
///
/// For atoms `i < j`, with `r` the minimum image distance and
/// `r_e` the sum of their covalent radii,
///
/// `q = exp(-alpha (r - r_e) / r_e) + beta r_e / r`.
///
/// Pairs are stored in lexicographic order. `radii.len()` is the atom
/// count, and every path row has `3 * radii.len()` Cartesian components.
///
/// Non-exhaustive: start from [`MorsePairChart::new`] or [`Default`] and
/// chain the `with_*` setters.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct MorsePairChart {
    /// Covalent radius of each atom, in the path's length unit. Finite and positive.
    ///
    /// Private so a replacement goes through [`Self::with_radii`], which repeats
    /// the check in [`Self::new`].
    radii: Array1<f64>,
    /// Exponent weight. The paper uses [`MORSE_ALPHA`]. Finite.
    pub alpha: f64,
    /// Inverse distance weight. The paper uses [`MORSE_BETA`]. Finite.
    pub beta: f64,
    /// Periodic cell. `None` is free space.
    pub cell: Option<Cell>,
}

with_setters!(MorsePairChart {
    with_alpha => alpha: val f64,
    with_beta => beta: val f64,
    with_cell => cell: opt Cell,
});

impl Default for MorsePairChart {
    fn default() -> Self {
        Self {
            radii: Array1::zeros(0),
            alpha: MORSE_ALPHA,
            beta: MORSE_BETA,
            cell: None,
        }
    }
}

impl MorsePairChart {
    /// Chart with the paper's alpha and beta, in free space.
    ///
    /// # Errors
    ///
    /// [`SaddleError::Shape`] when `radii` is empty, and
    /// [`SaddleError::Invalid`] when a radius is not finite and positive.
    pub fn new(radii: Array1<f64>) -> Result<Self, SaddleError> {
        check_morse_radii(&radii)?;
        Ok(Self {
            radii,
            alpha: MORSE_ALPHA,
            beta: MORSE_BETA,
            cell: None,
        })
    }

    /// Covalent radius of each atom.
    pub fn radii(&self) -> ndarray::ArrayView1<'_, f64> {
        self.radii.view()
    }

    /// Replace the covalent radii.
    ///
    /// # Errors
    ///
    /// [`SaddleError::Shape`] when `radii` is empty, and
    /// [`SaddleError::Invalid`] when a radius is not finite and positive.
    pub fn with_radii(mut self, radii: Array1<f64>) -> Result<Self, SaddleError> {
        check_morse_radii(&radii)?;
        self.radii = radii;
        Ok(self)
    }
}

fn check_morse_radii(radii: &Array1<f64>) -> Result<(), SaddleError> {
    if radii.is_empty() {
        return Err(SaddleError::Shape(
            "Morse chart needs a covalent radius per atom".into(),
        ));
    }
    if radii.iter().any(|r| !(r.is_finite() && *r > 0.0)) {
        return Err(SaddleError::Invalid(
            "Morse covalent radii must be finite and positive".into(),
        ));
    }
    Ok(())
}

impl GeodesicChart for MorsePairChart {
    fn n_coords(&self) -> usize {
        let n = self.radii.len();
        n.saturating_mul(n.saturating_sub(1)) / 2
    }

    fn cell(&self) -> Option<&Cell> {
        self.cell.as_ref()
    }

    fn eval(
        &self,
        x: ArrayView1<f64>,
        q: &mut Array1<f64>,
        b: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        let n = self.radii.len();
        let nq = self.n_coords();
        if n == 0 || x.len() != n.saturating_mul(3) || q.len() != nq || b.dim() != (nq, x.len()) {
            return Err(SaddleError::Shape(
                "Morse chart expects 3N Cartesians and one coordinate per pair".into(),
            ));
        }
        if !(self.alpha.is_finite() && self.beta.is_finite()) {
            return Err(SaddleError::Invalid(
                "Morse alpha and beta must be finite".into(),
            ));
        }
        if !x.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("geodesic path"));
        }
        q.fill(0.0);
        b.fill(0.0);
        let mut row = 0;
        for i in 0..n {
            let at_i = [x[3 * i], x[3 * i + 1], x[3 * i + 2]];
            for j in (i + 1)..n {
                let at_j = [x[3 * j], x[3 * j + 1], x[3 * j + 2]];
                let diff = match &self.cell {
                    Some(cell) => cell.displacement(at_j, at_i),
                    None => [at_i[0] - at_j[0], at_i[1] - at_j[1], at_i[2] - at_j[2]],
                };
                let r2 = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];
                if !r2.is_finite() {
                    return Err(SaddleError::NonFinite("geodesic pair distance"));
                }
                let r = r2.sqrt();
                let re = self.radii[i] + self.radii[j];
                if r <= MORSE_DISTANCE_FLOOR || re <= MORSE_DISTANCE_FLOOR {
                    return Err(SaddleError::Invalid(
                        "Morse pair distance is below the distance floor".into(),
                    ));
                }
                let exp_term = (-self.alpha * (r - re) / re).exp();
                if !exp_term.is_finite() {
                    return Err(SaddleError::NonFinite("geodesic Morse coordinate"));
                }
                let q_val = exp_term + self.beta * re / r;
                let dq_dr = -(self.alpha / re) * exp_term - self.beta * re / (r * r);
                let scale = dq_dr / r;
                if !q_val.is_finite() || !dq_dr.is_finite() || !scale.is_finite() {
                    return Err(SaddleError::NonFinite("geodesic Morse coordinate"));
                }
                let mut columns = [0.0; 3];
                for c in 0..3 {
                    let d = scale * diff[c];
                    if !d.is_finite() {
                        return Err(SaddleError::NonFinite("geodesic Morse coordinate"));
                    }
                    columns[c] = d;
                }
                q[row] = q_val;
                for c in 0..3 {
                    b[(row, 3 * i + c)] = columns[c];
                    b[(row, 3 * j + c)] = -columns[c];
                }
                row += 1;
            }
        }
        Ok(())
    }
}

/// Stop, step, and image cap for [`geodesic_path`].
///
/// Lengths and the step cap are in the path's coordinate unit. The ratios
/// compare the Eq. (B5) and Eq. (B18) bounds with the Eq. (17) length.
///
/// Non-exhaustive: start from [`GeodesicConfig::default`] and chain the
/// `with_*` setters.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct GeodesicConfig {
    /// Largest image displacement of a finished pass. Finite and positive.
    pub step_tol: f64,
    /// L2 norm of one image's length gradient at which that image stops. Finite and positive.
    pub grad_tol: f64,
    /// Largest Euclidean displacement of one L-BFGS step. Finite and positive.
    pub max_move: f64,
    /// Insert images when the lower bound is under this fraction of the length. In `(0, 1]`.
    pub lower_ratio: f64,
    /// Insert images when the upper bound exceeds this multiple of the length. At least 1.
    pub upper_ratio: f64,
    /// Forward and reverse passes, and restarts after an insertion. At least 1.
    pub max_sweeps: usize,
    /// L-BFGS iterations for one image inside one pass. At least 1.
    pub max_iterations: usize,
    /// L-BFGS vector pairs. At least 1.
    pub memory: usize,
    /// Cartesian pieces in the upper bound. Even and at least 2, so the equal
    /// partition includes the Cartesian midpoint. The paper uses 10.
    pub bound_samples: usize,
    /// Image cap, endpoints included, after midpoint insertion. At least 2.
    pub max_images: usize,
    /// Seed of the noisy starts of [`geodesic_interpolate`]. Any value; the same seed repeats the path.
    pub seed: u64,
}

with_setters!(GeodesicConfig {
    with_step_tol => step_tol: val f64,
    with_grad_tol => grad_tol: val f64,
    with_max_move => max_move: val f64,
    with_lower_ratio => lower_ratio: val f64,
    with_upper_ratio => upper_ratio: val f64,
    with_max_sweeps => max_sweeps: val usize,
    with_max_iterations => max_iterations: val usize,
    with_memory => memory: val usize,
    with_bound_samples => bound_samples: val usize,
    with_max_images => max_images: val usize,
    with_seed => seed: val u64,
});

impl Default for GeodesicConfig {
    fn default() -> Self {
        Self {
            step_tol: DEFAULT_STEP_TOL,
            grad_tol: DEFAULT_GRAD_TOL,
            max_move: DEFAULT_MAX_MOVE,
            lower_ratio: DEFAULT_LOWER_RATIO,
            upper_ratio: DEFAULT_UPPER_RATIO,
            max_sweeps: DEFAULT_MAX_SWEEPS,
            max_iterations: DEFAULT_MAX_ITERATIONS,
            memory: DEFAULT_MEMORY,
            bound_samples: DEFAULT_BOUND_SAMPLES,
            max_images: DEFAULT_MAX_IMAGES,
            seed: DEFAULT_SEED,
        }
    }
}

impl GeodesicConfig {
    /// Reject a non-finite or non-positive stop, a ratio outside the paper's range, or a zero count.
    ///
    /// # Errors
    ///
    /// [`SaddleError::Invalid`] when a field is outside the range documented on the struct.
    pub fn check(&self) -> Result<(), SaddleError> {
        let positive = [self.step_tol, self.grad_tol, self.max_move];
        if !positive.iter().all(|v| v.is_finite() && *v > 0.0) {
            return Err(SaddleError::Invalid(
                "geodesic step, gradient, and move tolerances must be finite and positive".into(),
            ));
        }
        if !(self.lower_ratio.is_finite() && self.lower_ratio > 0.0 && self.lower_ratio <= 1.0) {
            return Err(SaddleError::Invalid(
                "geodesic lower ratio must be finite and in (0, 1]".into(),
            ));
        }
        if !(self.upper_ratio.is_finite() && self.upper_ratio >= 1.0) {
            return Err(SaddleError::Invalid(
                "geodesic upper ratio must be finite and at least 1".into(),
            ));
        }
        if self.max_sweeps == 0 || self.max_iterations == 0 || self.memory == 0 {
            return Err(SaddleError::Invalid(
                "geodesic sweeps, iterations, and memory must be at least 1".into(),
            ));
        }
        if !bound_samples_include_midpoint(self.bound_samples) {
            return Err(SaddleError::Invalid(
                "geodesic bound samples must be even and at least 2 so the equal partition includes the Cartesian midpoint".into(),
            ));
        }
        if self.max_images < 2 {
            return Err(SaddleError::Invalid(
                "geodesic image cap must be at least 2".into(),
            ));
        }
        Ok(())
    }
}

/// Eq. (17) length and the two bounds, for one path.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct GeodesicLengths {
    /// Sum of the midpoint formula over segments.
    pub length: f64,
    /// Sum of the coordinate chords between segment ends.
    pub lower: f64,
    /// Sum of the subdivided Cartesian chords.
    pub upper: f64,
}

/// Result of [`geodesic_path`] or [`geodesic_interpolate`].
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct GeodesicReport {
    /// Eq. (17) length of the returned path.
    pub length: f64,
    /// Eq. (B5) lower bound.
    pub lower: f64,
    /// Eq. (B18) upper bound.
    pub upper: f64,
    /// Images in the returned path, endpoints included.
    pub images: usize,
    /// Forward and reverse passes actually run.
    pub sweeps: usize,
    /// False when the last round of sweeps used all of `max_sweeps` with the
    /// largest image displacement still above `step_tol`. The path is then the
    /// last iterate, not a stationary one. Loose bounds at the image cap do not
    /// clear this flag. Read `lower` and `upper` for those.
    pub converged: bool,
}

/// Length and bounds of `path` without moving it.
///
/// `path` has one image per row. `bound_samples` is the upper bound's
/// subdivision and must be at least 2.
///
/// # Errors
///
/// [`SaddleError::Shape`] when the path has fewer than two images or the
/// chart has no coordinates, [`SaddleError::Invalid`] when `bound_samples`
/// is odd or below 2, and the chart's own errors.
pub fn geodesic_lengths<C: GeodesicChart + ?Sized>(
    path: ArrayView2<f64>,
    chart: &C,
    bound_samples: usize,
) -> Result<GeodesicLengths, SaddleError> {
    Ok(measure(path, chart, bound_samples, None)?.total)
}

/// Minimize the geodesic length of `path`.
///
/// Row 0 and the last row stay at the input endpoints. Each interior row
/// is an L-BFGS minimization of the Eq. (17) length, with the analytic
/// gradient of Algorithm 1, swept forward and then back.
///
/// A round of sweeps that uses all of [`GeodesicConfig::max_sweeps`]
/// with an image still moving by more than `step_tol` ends the run with
/// [`GeodesicReport::converged`] false. The path returned is then the
/// last iterate. Callers that need a stationary path check that flag.
///
/// # Errors
///
/// [`SaddleError::Shape`] when the path has fewer than two rows or the
/// chart has no coordinates, [`SaddleError::Invalid`] from [`GeodesicConfig::check`],
/// [`SaddleError::NonFinite`] when an input is not finite, and the chart's
/// own errors on the starting path. A later trial the chart refuses is
/// discarded and that image keeps its last accepted point. Convergence
/// counts only an accepted step that meets the stop: a refused trial that
/// leaves the image unmoved does not settle the pass.
///
/// # Examples
///
/// ```
/// use ndarray::Array2;
/// use rgsaddle::{GeodesicChart, GeodesicConfig, SaddleError, geodesic_path};
///
/// struct Flat;
/// impl GeodesicChart for Flat {
///     fn n_coords(&self) -> usize { 2 }
///     fn eval(
///         &self,
///         x: ndarray::ArrayView1<f64>,
///         q: &mut ndarray::Array1<f64>,
///         b: &mut ndarray::Array2<f64>,
///     ) -> Result<(), SaddleError> {
///         q.assign(&x);
///         b.fill(0.0);
///         b[(0, 0)] = 1.0;
///         b[(1, 1)] = 1.0;
///         Ok(())
///     }
/// }
///
/// let path = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
/// let (_out, report) = geodesic_path(path.view(), &Flat, &GeodesicConfig::default())?;
/// // Appendix A: the straight segment from (0, 0) to (3, 4) has length 5.
/// assert!((report.length - 5.0).abs() < 1.0e-4);
/// # Ok::<(), SaddleError>(())
/// ```
pub fn geodesic_path<C: GeodesicChart + ?Sized>(
    path: ArrayView2<f64>,
    chart: &C,
    config: &GeodesicConfig,
) -> Result<(Array2<f64>, GeodesicReport), SaddleError> {
    config.check()?;
    check_path(path, chart.cell())?;
    if chart.n_coords() == 0 {
        return Err(SaddleError::Shape(
            "geodesic chart has no coordinates".into(),
        ));
    }
    let mut path = path.to_owned();
    // The first measure rejects a chart that is already undefined.
    measure(path.view(), chart, config.bound_samples, None)?;
    let mut sweeps = 0usize;
    let (lengths, converged) = loop {
        let mut moved = f64::MAX;
        for _ in 0..config.max_sweeps {
            moved = sweep(&mut path, chart, config)?;
            sweeps += 1;
            if moved <= config.step_tol {
                break;
            }
        }
        let settled = moved <= config.step_tol;
        let measured = measure(path.view(), chart, config.bound_samples, None)?;
        if !settled || bounds_tight(&measured.total, config) {
            break (measured.total, settled);
        }
        let grown = insert_midpoints(&path, &measured.segments, config, chart.cell());
        if grown.nrows() == path.nrows()
            || measure(grown.view(), chart, config.bound_samples, None).is_err()
        {
            break (measured.total, true);
        }
        path = grown;
    };
    let images = path.nrows();
    Ok((
        path,
        GeodesicReport {
            length: lengths.length,
            lower: lengths.lower,
            upper: lengths.upper,
            images,
            sweeps,
            converged,
        },
    ))
}

/// Geodesic path between two endpoints.
///
/// `n_images` counts the endpoints and must be at least 2. With three or
/// more images the interior rows start from the shortest seed of
/// Algorithm 3 and [`geodesic_path`] then minimizes the length.
///
/// # Errors
///
/// The errors of [`geodesic_path`], plus [`SaddleError::Shape`] when the
/// endpoints differ in length or `n_images` is below 2.
pub fn geodesic_interpolate<C: GeodesicChart + ?Sized>(
    reactant: ArrayView1<f64>,
    product: ArrayView1<f64>,
    n_images: usize,
    chart: &C,
    config: &GeodesicConfig,
) -> Result<(Array2<f64>, GeodesicReport), SaddleError> {
    config.check()?;
    let dof = reactant.len();
    if product.len() != dof || dof == 0 || n_images < 2 {
        return Err(SaddleError::Shape(
            "geodesic interpolation needs two matching endpoints and at least two images".into(),
        ));
    }
    if !reactant.iter().all(|v| v.is_finite()) || !product.iter().all(|v| v.is_finite()) {
        return Err(SaddleError::NonFinite("geodesic path"));
    }
    if chart.cell().is_some() && !dof.is_multiple_of(3) {
        return Err(SaddleError::Shape(
            "a periodic geodesic path holds three Cartesians per atom".into(),
        ));
    }
    if n_images < 3 {
        let mut path = Array2::zeros((n_images, dof));
        path.row_mut(0).assign(&reactant);
        path.row_mut(n_images - 1).assign(&product);
        return geodesic_path(path.view(), chart, config);
    }
    let mid = interior_seed(reactant, product, chart, config)?;
    let third = n_images / 3;
    let two_thirds = (2 * n_images) / 3;
    let mut path = Array2::zeros((n_images, dof));
    for i in 0..n_images {
        let slot = i + 1;
        if slot <= third {
            path.row_mut(i).assign(&reactant);
        } else if slot <= two_thirds {
            path.row_mut(i).assign(&mid);
        } else {
            path.row_mut(i).assign(&product);
        }
    }
    path.row_mut(0).assign(&reactant);
    path.row_mut(n_images - 1).assign(&product);
    geodesic_path(path.view(), chart, config)
}

/// Average redundant coordinates `(q(A) + q(B)) / 2` of Algorithm 3.
fn internal_midpoint<C: GeodesicChart + ?Sized>(
    reactant: ArrayView1<f64>,
    product: ArrayView1<f64>,
    chart: &C,
) -> Result<Array1<f64>, SaddleError> {
    let q_reactant = sample(chart, reactant)?.q;
    let q_product = sample(chart, product)?.q;
    let mut mid = Array1::zeros(q_reactant.len());
    for i in 0..mid.len() {
        mid[i] = MIDPOINT_WEIGHT * (q_reactant[i] + q_product[i]);
    }
    Ok(mid)
}

/// Best of the ten noisy starts of Algorithm 3, before any path adjustment.
///
/// Starts alternate between reactant and product. Each is pulled to
/// `q_mid` and the one whose three point path `[A, x, B]` is shortest wins.
fn pulled_start<C: GeodesicChart + ?Sized>(
    reactant: ArrayView1<f64>,
    product: ArrayView1<f64>,
    chart: &C,
    config: &GeodesicConfig,
) -> Result<Array1<f64>, SaddleError> {
    let dof = reactant.len();
    let target = internal_midpoint(reactant, product, chart)?;
    let mut rng = Rng::new(config.seed);
    let mut best: Option<(f64, Array1<f64>)> = None;
    for start in 0..START_COUNT {
        let base = if start % 2 == 0 { reactant } else { product };
        let mut x0 = base.to_owned();
        for value in x0.iter_mut() {
            *value += START_NOISE * rng.normal();
        }
        if sample(chart, x0.view()).is_err() {
            continue;
        }
        let Ok(matched) = pull_to_target(x0.view(), target.view(), chart, config) else {
            continue;
        };
        let mut trial = Array2::zeros((3, dof));
        trial.row_mut(0).assign(&reactant);
        trial.row_mut(1).assign(&matched);
        trial.row_mut(2).assign(&product);
        let Ok(lengths) = geodesic_lengths(trial.view(), chart, config.bound_samples) else {
            continue;
        };
        if best
            .as_ref()
            .is_none_or(|(shortest, _)| lengths.length < *shortest)
        {
            best = Some((lengths.length, matched));
        }
    }
    best.map(|(_, matched)| matched)
        .ok_or_else(|| SaddleError::Surface("geodesic seed search failed on every start".into()))
}

/// Middle image of Algorithm 3: the best noisy start, then the geodesic
/// procedure adjusts that image alone on the path `[A, x_mid, B]`.
fn interior_seed<C: GeodesicChart + ?Sized>(
    reactant: ArrayView1<f64>,
    product: ArrayView1<f64>,
    chart: &C,
    config: &GeodesicConfig,
) -> Result<Array1<f64>, SaddleError> {
    let dof = reactant.len();
    let mid = pulled_start(reactant, product, chart, config)?;
    let mut trial = Array2::zeros((3, dof));
    trial.row_mut(0).assign(&reactant);
    trial.row_mut(1).assign(&mid);
    trial.row_mut(2).assign(&product);
    // Three images only: the procedure moves the middle image and inserts none.
    let fixed = config.clone().with_max_images(3);
    let (adjusted, _) = geodesic_path(trial.view(), chart, &fixed)?;
    Ok(adjusted.row(1).to_owned())
}

struct Sample {
    q: Array1<f64>,
    b: Array2<f64>,
}

struct Segment {
    length: f64,
    lower: f64,
    upper: f64,
}

struct Measured {
    total: GeodesicLengths,
    segments: Vec<Segment>,
}

fn check_path(path: ArrayView2<f64>, cell: Option<&Cell>) -> Result<(), SaddleError> {
    if path.nrows() < 2 || path.ncols() == 0 {
        return Err(SaddleError::Shape(
            "geodesic path needs at least two images".into(),
        ));
    }
    if cell.is_some() && !path.ncols().is_multiple_of(3) {
        return Err(SaddleError::Shape(
            "a periodic geodesic path holds three Cartesians per atom".into(),
        ));
    }
    if path.iter().any(|v| !v.is_finite()) {
        return Err(SaddleError::NonFinite("geodesic path"));
    }
    Ok(())
}

fn sample<C: GeodesicChart + ?Sized>(chart: &C, x: ArrayView1<f64>) -> Result<Sample, SaddleError> {
    let nq = chart.n_coords();
    let dof = x.len();
    if nq == 0 || dof == 0 {
        return Err(SaddleError::Shape(
            "geodesic chart needs a coordinate and a degree of freedom".into(),
        ));
    }
    if nq
        .checked_mul(dof)
        .is_none_or(|cells| cells > isize::MAX as usize / size_of::<f64>())
    {
        return Err(SaddleError::Shape(
            "geodesic Wilson matrix is too large to allocate".into(),
        ));
    }
    let mut q = Array1::zeros(nq);
    let mut b = Array2::zeros((nq, dof));
    chart.eval(x, &mut q, &mut b)?;
    if q.iter().any(|v| !v.is_finite()) || b.iter().any(|v| !v.is_finite()) {
        return Err(SaddleError::NonFinite("geodesic chart"));
    }
    Ok(Sample { q, b })
}

/// `left + t (right - left)` for one row, the difference taken per atom
/// through [`Cell::displacement`] when the chart is periodic.
fn toward(
    cell: Option<&Cell>,
    left: ArrayView1<f64>,
    right: ArrayView1<f64>,
    t: f64,
) -> Array1<f64> {
    let mut out = Array1::zeros(left.len());
    match cell {
        None => {
            for k in 0..left.len() {
                out[k] = left[k] + t * (right[k] - left[k]);
            }
        }
        Some(cell) => {
            for atom in 0..left.len() / 3 {
                let at = 3 * atom;
                let from = [left[at], left[at + 1], left[at + 2]];
                let to = [right[at], right[at + 1], right[at + 2]];
                let step = cell.displacement(from, to);
                for c in 0..3 {
                    out[at + c] = from[c] + t * step[c];
                }
            }
        }
    }
    out
}

fn chord(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let mut acc = 0.0;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        acc += d * d;
    }
    acc.sqrt()
}

fn add_bt_u(grad: &mut ArrayViewMut1<f64>, b: &Array2<f64>, u: &Array1<f64>, scale: f64) {
    for i in 0..u.len() {
        let w = scale * u[i];
        if w == 0.0 {
            continue;
        }
        for j in 0..grad.len() {
            grad[j] += w * b[(i, j)];
        }
    }
}

fn measure<C: GeodesicChart + ?Sized>(
    path: ArrayView2<f64>,
    chart: &C,
    bound_samples: usize,
    mut grad: Option<&mut Array2<f64>>,
) -> Result<Measured, SaddleError> {
    check_path(path, chart.cell())?;
    if !bound_samples_include_midpoint(bound_samples) {
        return Err(SaddleError::Invalid(
            "geodesic bound samples must be even and at least 2 so the equal partition includes the Cartesian midpoint".into(),
        ));
    }
    let n = path.nrows();
    let dof = path.ncols();
    if let Some(g) = grad.as_mut() {
        if g.dim() != (n, dof) {
            return Err(SaddleError::Shape(
                "geodesic gradient has one row per image".into(),
            ));
        }
        g.fill(0.0);
    }
    let mut images = Vec::with_capacity(n);
    for i in 0..n {
        images.push(sample(chart, path.row(i))?);
    }
    let mut segments = Vec::with_capacity(n - 1);
    let mut length = 0.0;
    let mut lower = 0.0;
    let mut upper = 0.0;
    for i in 0..n - 1 {
        let left = path.row(i);
        let right = path.row(i + 1);
        let mid_x = toward(chart.cell(), left, right, MIDPOINT_WEIGHT);
        let mid = sample(chart, mid_x.view())?;
        let n_left = chord(&mid.q, &images[i].q);
        let n_right = chord(&mid.q, &images[i + 1].q);
        let seg_length = n_left + n_right;
        let seg_lower = chord(&images[i + 1].q, &images[i].q);
        let seg_upper = upper_bound(
            chart,
            left,
            right,
            &images[i].q,
            &images[i + 1].q,
            bound_samples,
        )?;
        if let Some(g) = grad.as_mut() {
            if n_left >= CHORD_FLOOR {
                let u = unit(&mid.q, &images[i].q, n_left);
                add_bt_u(&mut g.row_mut(i), &mid.b, &u, MIDPOINT_WEIGHT);
                add_bt_u(&mut g.row_mut(i), &images[i].b, &u, -1.0);
                add_bt_u(&mut g.row_mut(i + 1), &mid.b, &u, MIDPOINT_WEIGHT);
            }
            if n_right >= CHORD_FLOOR {
                let u = unit(&mid.q, &images[i + 1].q, n_right);
                add_bt_u(&mut g.row_mut(i), &mid.b, &u, MIDPOINT_WEIGHT);
                add_bt_u(&mut g.row_mut(i + 1), &mid.b, &u, MIDPOINT_WEIGHT);
                add_bt_u(&mut g.row_mut(i + 1), &images[i + 1].b, &u, -1.0);
            }
        }
        length += seg_length;
        lower += seg_lower;
        upper += seg_upper;
        segments.push(Segment {
            length: seg_length,
            lower: seg_lower,
            upper: seg_upper,
        });
    }
    if !length.is_finite() || !lower.is_finite() || !upper.is_finite() {
        return Err(SaddleError::NonFinite("geodesic length"));
    }
    Ok(Measured {
        total: GeodesicLengths {
            length,
            lower,
            upper,
        },
        segments,
    })
}

fn unit(a: &Array1<f64>, b: &Array1<f64>, norm: f64) -> Array1<f64> {
    let mut u = Array1::zeros(a.len());
    for i in 0..a.len() {
        u[i] = (a[i] - b[i]) / norm;
    }
    u
}

fn upper_bound<C: GeodesicChart + ?Sized>(
    chart: &C,
    left: ArrayView1<f64>,
    right: ArrayView1<f64>,
    q_left: &Array1<f64>,
    q_right: &Array1<f64>,
    samples: usize,
) -> Result<f64, SaddleError> {
    if !bound_samples_include_midpoint(samples) {
        return Err(SaddleError::Invalid(
            "geodesic bound samples must be even and at least 2 so the equal partition includes the Cartesian midpoint".into(),
        ));
    }
    let mut prev = q_left.clone();
    let mut total = 0.0;
    let denom = samples as f64;
    for k in 1..samples {
        // An even count has a node at 1/2. Write that node as 0.5 so it is
        // the Cartesian midpoint even when `k / samples` is not exact.
        let t = if k * 2 == samples {
            MIDPOINT_WEIGHT
        } else {
            (k as f64) / denom
        };
        let x = toward(chart.cell(), left, right, t);
        let next = sample(chart, x.view())?;
        total += chord(&next.q, &prev);
        prev = next.q;
    }
    total += chord(q_right, &prev);
    if !total.is_finite() {
        return Err(SaddleError::NonFinite("geodesic length"));
    }
    Ok(total)
}

fn bound_samples_include_midpoint(samples: usize) -> bool {
    samples >= 2 && samples.is_multiple_of(2)
}

fn bounds_tight(lengths: &GeodesicLengths, config: &GeodesicConfig) -> bool {
    lengths.lower + CHORD_FLOOR >= config.lower_ratio * lengths.length
        && lengths.upper <= config.upper_ratio * lengths.length + CHORD_FLOOR
}

fn segment_gap(seg: &Segment, config: &GeodesicConfig) -> f64 {
    let lower_gap = config.lower_ratio * seg.length - seg.lower;
    let upper_gap = seg.upper - config.upper_ratio * seg.length;
    if lower_gap > upper_gap {
        lower_gap
    } else {
        upper_gap
    }
}

fn segment_loose(seg: &Segment, config: &GeodesicConfig) -> bool {
    seg.lower + CHORD_FLOOR < config.lower_ratio * seg.length
        || seg.upper > config.upper_ratio * seg.length + CHORD_FLOOR
}

fn insert_midpoints(
    path: &Array2<f64>,
    segments: &[Segment],
    config: &GeodesicConfig,
    cell: Option<&Cell>,
) -> Array2<f64> {
    let n = path.nrows();
    if n < 2 || segments.len() != n - 1 || segments.is_empty() || path.nrows() >= config.max_images
    {
        return path.clone();
    }
    let mut loose: Vec<bool> = segments
        .iter()
        .map(|seg| segment_loose(seg, config))
        .collect();
    if !loose.iter().any(|flag| *flag) {
        let mut worst = 0usize;
        let mut worst_gap = segment_gap(&segments[0], config);
        for (i, seg) in segments.iter().enumerate().skip(1) {
            let gap = segment_gap(seg, config);
            if gap > worst_gap {
                worst = i;
                worst_gap = gap;
            }
        }
        loose[worst] = true;
    }
    let room = config.max_images - n;
    let marked = loose.iter().filter(|flag| **flag).count();
    if marked > room {
        let mut order: Vec<usize> = (0..segments.len()).filter(|&i| loose[i]).collect();
        order.sort_by(|&a, &b| {
            let ga = segment_gap(&segments[a], config);
            let gb = segment_gap(&segments[b], config);
            if ga < gb {
                std::cmp::Ordering::Less
            } else if ga > gb {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        });
        loose.fill(false);
        let keep_from = order.len() - room;
        for index in order.iter().skip(keep_from) {
            loose[*index] = true;
        }
    }
    let mut rows: Vec<Array1<f64>> = Vec::with_capacity(n + room);
    let dof = path.ncols();
    for (i, &split) in loose.iter().enumerate().take(n - 1) {
        rows.push(path.row(i).to_owned());
        if split {
            rows.push(toward(cell, path.row(i), path.row(i + 1), MIDPOINT_WEIGHT));
        }
    }
    rows.push(path.row(n - 1).to_owned());
    let mut out = Array2::zeros((rows.len(), dof));
    for (i, row) in rows.iter().enumerate() {
        out.row_mut(i).assign(row);
    }
    out
}

fn sweep<C: GeodesicChart + ?Sized>(
    path: &mut Array2<f64>,
    chart: &C,
    config: &GeodesicConfig,
) -> Result<f64, SaddleError> {
    let n = path.nrows();
    if n < 3 {
        return Ok(0.0);
    }
    let mut order: Vec<usize> = (1..n - 1).collect();
    let reverse: Vec<usize> = (1..n - 1).rev().collect();
    order.extend(reverse);
    let mut max_disp = 0.0;
    let mut accepted_stop = true;
    for index in order {
        let before = path.row(index).to_owned();
        if !relax_image(path, index, chart, config)? {
            accepted_stop = false;
        }
        let mut acc = 0.0;
        for k in 0..path.ncols() {
            let d = path[(index, k)] - before[k];
            acc += d * d;
        }
        let disp = acc.sqrt();
        if disp > max_disp {
            max_disp = disp;
        }
    }
    // A refused trial reverts the image, so the displacement is zero. That
    // zero is not an accepted step and must not settle the pass.
    if !accepted_stop && max_disp <= config.step_tol {
        max_disp = config.step_tol.next_up();
    }
    Ok(max_disp)
}

fn relax_image<C: GeodesicChart + ?Sized>(
    path: &mut Array2<f64>,
    index: usize,
    chart: &C,
    config: &GeodesicConfig,
) -> Result<bool, SaddleError> {
    use rgmin::{Accept, Control, Method, Oracle, Solver};

    let dof = path.ncols();
    let base = path.clone();
    let step_cap = config.max_move;
    let iterations = config.max_iterations;
    let grad_tol = config.grad_tol;
    let memory = config.memory;
    let samples = config.bound_samples;
    let mut y = path.row(index).to_owned();
    let oracle = Oracle::unbounded(dof, move |trial: ArrayView1<f64>| {
        let mut current = base.clone();
        current.row_mut(index).assign(&trial);
        let mut grad = Array2::zeros((current.nrows(), dof));
        match measure(current.view(), chart, samples, Some(&mut grad)) {
            Ok(measured) => {
                let row = grad.row(index).to_owned();
                if measured.total.length.is_finite() && row.iter().all(|v| v.is_finite()) {
                    (measured.total.length, row)
                } else {
                    refused_oracle(dof)
                }
            }
            // A refused geometry is non-finite, not a zero gradient. A zero
            // gradient would tell the solver the trial had already converged.
            Err(_) => refused_oracle(dof),
        }
    });
    let mut solver = Solver::new(
        Method::lbfgs_with_memory(memory),
        Control::default()
            .with_maxiter(iterations)
            .with_gtol(grad_tol)
            .with_istep(INITIAL_STEP)
            .with_maxmove(step_cap),
        dof,
    );
    solver.set_accept(Accept::Energy);
    let start = y.clone();
    let mut last_good = y.clone();
    let mut rejected = false;
    for _ in 0..iterations {
        match solver.step(&oracle, &mut y) {
            Ok(report) => {
                if !y.iter().all(|v| v.is_finite()) {
                    y.assign(&last_good);
                    path.row_mut(index).assign(&y);
                    rejected = true;
                    break;
                }
                path.row_mut(index).assign(&y);
                if measure(path.view(), chart, samples, None).is_err() {
                    y.assign(&last_good);
                    path.row_mut(index).assign(&y);
                    rejected = true;
                    break;
                }
                last_good.assign(&y);
                // `report` describes this accepted point. A refused trial
                // never reaches here.
                if report.grad_norm.is_finite() && report.grad_norm < grad_tol {
                    break;
                }
            }
            Err(_) => {
                y.assign(&last_good);
                path.row_mut(index).assign(&y);
                rejected = true;
                break;
            }
        }
    }
    path.row_mut(index).assign(&y);
    if measure(path.view(), chart, samples, None).is_err() {
        path.row_mut(index).assign(&start);
        rejected = true;
    }
    if !rejected {
        return Ok(true);
    }
    // The restored point settles the image only when its own gradient,
    // measured on that accepted geometry, meets the tolerance.
    Ok(image_gradient_meets(path, index, chart, samples, grad_tol))
}

fn refused_oracle(dof: usize) -> (f64, Array1<f64>) {
    (f64::NAN, Array1::from_elem(dof, f64::NAN))
}

fn image_gradient_meets<C: GeodesicChart + ?Sized>(
    path: &Array2<f64>,
    index: usize,
    chart: &C,
    samples: usize,
    grad_tol: f64,
) -> bool {
    let mut grad = Array2::zeros((path.nrows(), path.ncols()));
    let Ok(_) = measure(path.view(), chart, samples, Some(&mut grad)) else {
        return false;
    };
    let mut acc = 0.0;
    for value in grad.row(index) {
        acc += value * value;
    }
    let gnorm = acc.sqrt();
    gnorm.is_finite() && gnorm < grad_tol
}

fn pull_to_target<C: GeodesicChart + ?Sized>(
    start: ArrayView1<f64>,
    target: ArrayView1<f64>,
    chart: &C,
    config: &GeodesicConfig,
) -> Result<Array1<f64>, SaddleError> {
    use rgmin::{Accept, Control, Method, Oracle, Solver};

    let dof = start.len();
    let mut y = start.to_owned();
    let target = target.to_owned();
    let grad_tol = config.grad_tol;
    let oracle = Oracle::unbounded(dof, move |trial: ArrayView1<f64>| {
        match sample(chart, trial) {
            Ok(taken) => {
                if taken.q.len() != target.len() {
                    return refused_oracle(dof);
                }
                let mut energy = 0.0;
                let mut residual = Array1::zeros(taken.q.len());
                for i in 0..taken.q.len() {
                    residual[i] = taken.q[i] - target[i];
                    energy += MIDPOINT_WEIGHT * residual[i] * residual[i];
                }
                let mut grad = Array1::zeros(dof);
                add_bt_u(&mut grad.view_mut(), &taken.b, &residual, 1.0);
                if !energy.is_finite() || grad.iter().any(|value| !value.is_finite()) {
                    return refused_oracle(dof);
                }
                (energy, grad)
            }
            Err(_) => refused_oracle(dof),
        }
    });
    let mut solver = Solver::new(
        Method::lbfgs_with_memory(config.memory),
        Control::default()
            .with_maxiter(config.max_iterations)
            .with_gtol(grad_tol)
            .with_istep(INITIAL_STEP)
            .with_maxmove(config.max_move),
        dof,
    );
    solver.set_accept(Accept::Energy);
    let origin = y.clone();
    let mut last_good = y.clone();
    for _ in 0..config.max_iterations {
        match solver.step(&oracle, &mut y) {
            Ok(report) => {
                if !y.iter().all(|v| v.is_finite()) || sample(chart, y.view()).is_err() {
                    y.assign(&last_good);
                    break;
                }
                last_good.assign(&y);
                if report.grad_norm < grad_tol {
                    break;
                }
            }
            Err(_) => {
                y.assign(&last_good);
                break;
            }
        }
    }
    if sample(chart, y.view()).is_err() {
        y.assign(&origin);
    }
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn morse_coordinate_matches_eq7_and_its_wilson_b() {
        let chart = MorsePairChart::new(array![1.0, 1.0]).unwrap();
        let x = array![0.0, 0.0, 0.0, 2.4, 0.2, -0.1];
        let mut q = Array1::zeros(1);
        let mut b = Array2::zeros((1, 6));
        chart.eval(x.view(), &mut q, &mut b).unwrap();
        let dx = x[3] - x[0];
        let dy = x[4] - x[1];
        let dz = x[5] - x[2];
        let r = (dx * dx + dy * dy + dz * dz).sqrt();
        let re = 2.0;
        let exp_term = (-MORSE_ALPHA * (r - re) / re).exp();
        let expect = exp_term + MORSE_BETA * re / r;
        assert!((q[0] - expect).abs() < 1e-12);
        let eps = 1e-6;
        for k in 0..6 {
            let mut plus = x.clone();
            let mut minus = x.clone();
            plus[k] += eps;
            minus[k] -= eps;
            let mut qp = Array1::zeros(1);
            let mut qm = Array1::zeros(1);
            let mut scratch = Array2::zeros((1, 6));
            chart.eval(plus.view(), &mut qp, &mut scratch).unwrap();
            chart.eval(minus.view(), &mut qm, &mut scratch).unwrap();
            let finite = (qp[0] - qm[0]) / (2.0 * eps);
            assert!(
                (b[(0, k)] - finite).abs() < 1e-6,
                "column {k}: {} vs {finite}",
                b[(0, k)]
            );
        }
    }

    /// Compare every component of the analytic length gradient, endpoints included,
    /// with a central difference of the Eq. (17) length.
    fn assert_dense_gradient(chart: &MorsePairChart, path: &Array2<f64>, samples: usize) {
        let (rows, cols) = path.dim();
        let mut grad = Array2::zeros((rows, cols));
        measure(path.view(), chart, samples, Some(&mut grad)).unwrap();
        let eps = 1e-6;
        for image in 0..rows {
            for col in 0..cols {
                let mut plus = path.clone();
                let mut minus = path.clone();
                plus[(image, col)] += eps;
                minus[(image, col)] -= eps;
                let lp = geodesic_lengths(plus.view(), chart, samples).unwrap();
                let lm = geodesic_lengths(minus.view(), chart, samples).unwrap();
                let finite = (lp.length - lm.length) / (2.0 * eps);
                assert!(
                    (grad[(image, col)] - finite).abs() < 1e-5 * (1.0 + finite.abs()),
                    "image {image} col {col}: {} vs {finite}",
                    grad[(image, col)],
                );
            }
        }
    }

    #[test]
    fn length_gradient_matches_a_central_difference_in_every_component() {
        let chart = MorsePairChart::new(array![1.0, 1.0]).unwrap();
        let pair = Array2::from_shape_vec(
            (3, 6),
            vec![
                0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.1, -0.2, 0.0, 0.4, 2.2, 0.1, 0.0, 0.0, 0.0, 5.0,
                0.3, -0.2,
            ],
        )
        .unwrap();
        assert_dense_gradient(&chart, &pair, 4);

        let triple = MorsePairChart::new(array![1.0, 0.8, 1.2]).unwrap();
        let path = Array2::from_shape_vec(
            (3, 9),
            vec![
                0.0, 0.0, 0.0, 1.6, 0.1, 0.0, 3.4, -0.2, 0.3, 0.1, 0.2, -0.1, 1.9, 1.1, 0.2, 2.4,
                1.8, 0.4, 0.0, 0.0, 0.0, 0.3, 1.7, 0.0, 2.8, 2.9, -0.3,
            ],
        )
        .unwrap();
        assert_dense_gradient(&triple, &path, 4);
    }

    fn periodic_box() -> Cell {
        Cell::ortho(10.0, 10.0, 10.0).unwrap()
    }

    /// The pair coordinate of Eq. (7) for r_e = 2.
    fn pair_q(r: f64) -> f64 {
        (-MORSE_ALPHA * (r - 2.0) / 2.0).exp() + MORSE_BETA * 2.0 / r
    }

    /// Atom 1 sits 1.5 from atom 0 in the reactant and 2.5 in the product. The
    /// product places it at x = 9.5, so the naive difference is 8.0 along x.
    fn crossing_pair() -> Array2<f64> {
        Array2::from_shape_vec(
            (2, 6),
            vec![
                2.0, 5.0, 5.0, 0.5, 5.0, 5.0, //
                2.0, 5.0, 5.0, 9.5, 5.0, 5.0,
            ],
        )
        .unwrap()
    }

    fn periodic_pair() -> MorsePairChart {
        MorsePairChart::new(array![1.0, 1.0])
            .unwrap()
            .with_cell(periodic_box())
    }

    #[test]
    fn a_periodic_midpoint_takes_the_short_way_across_the_box() {
        let cell = periodic_box();
        let left = array![9.5, 5.0, 5.0];
        let right = array![0.5, 5.0, 5.0];
        let mid = toward(Some(&cell), left.view(), right.view(), 0.5);
        assert!((mid[0] - 10.0).abs() < 1e-12, "{}", mid[0]);
        let naive = toward(None, left.view(), right.view(), 0.5);
        assert!((naive[0] - 5.0).abs() < 1e-12);
        let quarter = toward(Some(&cell), left.view(), right.view(), 0.25);
        assert!((quarter[0] - 9.75).abs() < 1e-12, "{}", quarter[0]);
    }

    #[test]
    fn a_segment_across_the_box_has_the_chord_length_and_bounds() {
        let chart = periodic_pair();
        let path = crossing_pair();
        let measured = measure(path.view(), &chart, 10, None).unwrap();
        let expect = pair_q(1.5) - pair_q(2.5);
        // The pair distance runs 1.5, 2.0, 2.5 by the short way. Every sample lies
        // on that line, so the length and both bounds telescope to the chord.
        assert!(
            (measured.total.length - expect).abs() < 1e-12,
            "{}",
            measured.total.length
        );
        assert!(
            (measured.total.lower - expect).abs() < 1e-12,
            "{}",
            measured.total.lower
        );
        assert!(
            (measured.total.upper - expect).abs() < 1e-12,
            "{}",
            measured.total.upper
        );
    }

    #[test]
    fn an_inserted_image_crosses_the_box_with_the_segment() {
        let cell = periodic_box();
        let path = crossing_pair();
        let loose = Segment {
            length: 1.0,
            lower: 0.0,
            upper: 5.0,
        };
        let grown = insert_midpoints(&path, &[loose], &GeodesicConfig::default(), Some(&cell));
        assert_eq!(grown.nrows(), 3);
        assert!(grown[(1, 3)].abs() < 1e-12, "{}", grown[(1, 3)]);
        let free = insert_midpoints(
            &path,
            &[Segment {
                length: 1.0,
                lower: 0.0,
                upper: 5.0,
            }],
            &GeodesicConfig::default(),
            None,
        );
        assert!((free[(1, 3)] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn a_wrapped_cluster_matches_the_same_cluster_unwrapped_in_free_space() {
        let radii = array![1.0, 0.8, 1.2];
        let periodic = MorsePairChart::new(radii.clone())
            .unwrap()
            .with_cell(periodic_box());
        let free = MorsePairChart::new(radii).unwrap();
        // The cluster straddles the x = 0 face. Atoms 0 and 2 are wrapped to
        // x near 10 in the wrapped rows and left at negative x in the free rows.
        let unwrapped = Array2::from_shape_vec(
            (3, 9),
            vec![
                0.3, 5.0, 5.0, 1.5, 5.5, 5.0, 0.8, 4.0, 5.2, //
                0.1, 5.1, 5.0, 1.2, 5.4, 5.1, 0.3, 3.9, 5.2, //
                -0.4, 5.0, 5.0, 0.9, 5.4, 5.0, -0.2, 3.9, 5.1,
            ],
        )
        .unwrap();
        let mut wrapped = unwrapped.clone();
        for row in 0..3 {
            for atom in 0..3 {
                if wrapped[(row, 3 * atom)] < 0.0 {
                    wrapped[(row, 3 * atom)] += 10.0;
                }
            }
        }
        assert!(wrapped[(2, 0)] > 9.0);
        let mut grad_free = Array2::zeros((3, 9));
        let mut grad_wrapped = Array2::zeros((3, 9));
        let on_free = measure(unwrapped.view(), &free, 10, Some(&mut grad_free)).unwrap();
        let on_wrapped = measure(wrapped.view(), &periodic, 10, Some(&mut grad_wrapped)).unwrap();
        assert!((on_free.total.length - on_wrapped.total.length).abs() < 1e-10);
        assert!((on_free.total.lower - on_wrapped.total.lower).abs() < 1e-10);
        assert!((on_free.total.upper - on_wrapped.total.upper).abs() < 1e-10);
        for (u, v) in grad_free.iter().zip(grad_wrapped.iter()) {
            assert!((u - v).abs() < 1e-9, "{u} vs {v}");
        }
        let naive = measure(wrapped.view(), &free, 10, None).unwrap();
        assert!(
            (naive.total.length - on_wrapped.total.length).abs() > 0.05,
            "the box crossing must matter: {} vs {}",
            naive.total.length,
            on_wrapped.total.length
        );
    }

    #[test]
    fn the_periodic_length_gradient_matches_a_central_difference_across_the_box() {
        let chart = periodic_pair();
        let path = Array2::from_shape_vec(
            (3, 6),
            vec![
                2.0, 5.0, 5.0, 0.5, 5.2, 5.0, //
                2.0, 5.0, 5.0, 0.1, 5.3, 4.9, //
                2.0, 5.0, 5.0, 9.4, 5.1, 5.1,
            ],
        )
        .unwrap();
        assert_dense_gradient(&chart, &path, 4);
    }

    fn dissociation_ends() -> (Array1<f64>, Array1<f64>) {
        (
            array![0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
            array![0.0, 0.0, 0.0, 6.0, 0.0, 0.0],
        )
    }

    fn morse_q(chart: &MorsePairChart, x: &Array1<f64>) -> f64 {
        sample(chart, x.view()).unwrap().q[0]
    }

    #[test]
    fn the_target_is_the_average_of_the_coordinates() {
        let chart = MorsePairChart::new(array![1.0, 1.0]).unwrap();
        let (a, b) = dissociation_ends();
        let target = internal_midpoint(a.view(), b.view(), &chart).unwrap();
        let expect = 0.5 * (morse_q(&chart, &a) + morse_q(&chart, &b));
        assert!(
            (target[0] - expect).abs() < 1e-15,
            "{} vs {expect}",
            target[0]
        );
        let cartesian_average = (&a + &b) * 0.5;
        let wrong = morse_q(&chart, &cartesian_average);
        assert!(
            (wrong - expect).abs() > 0.3,
            "q(avg) {wrong}, avg(q) {expect}"
        );
    }

    #[test]
    fn the_pulled_start_reaches_the_average_coordinate() {
        let chart = MorsePairChart::new(array![1.0, 1.0]).unwrap();
        let (a, b) = dissociation_ends();
        let expect = 0.5 * (morse_q(&chart, &a) + morse_q(&chart, &b));
        let config = GeodesicConfig::default();
        let start = pulled_start(a.view(), b.view(), &chart, &config).unwrap();
        let reached = morse_q(&chart, &start);
        // q(x) lands on (q(A) + q(B)) / 2. The Cartesian average would land on q((A + B) / 2).
        assert!(
            (reached - expect).abs() < 1e-5,
            "q(x) {reached}, target {expect}"
        );
    }

    #[test]
    fn the_same_seed_repeats_the_noisy_starts_and_another_seed_changes_them() {
        let chart = MorsePairChart::new(array![1.0, 1.0, 1.0]).unwrap();
        let a = array![0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 3.5, 0.0, 0.0];
        let b = array![0.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 3.0, 0.5];
        let base = GeodesicConfig::default();
        let first = pulled_start(a.view(), b.view(), &chart, &base).unwrap();
        let again = pulled_start(a.view(), b.view(), &chart, &base).unwrap();
        assert_eq!(first, again);
        let other = pulled_start(a.view(), b.view(), &chart, &base.clone().with_seed(7)).unwrap();
        assert!(first.iter().zip(other.iter()).any(|(u, v)| u != v));
    }
}
