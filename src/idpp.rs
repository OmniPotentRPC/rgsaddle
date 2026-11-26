//! Image dependent pair potential initial path.
//!
//! Smidstrup, Pedersen, Stokbro, and Jonsson, J. Chem. Phys. 140,
//! 214106 (2014), doi:10.1063/1.4878664.
//!
//! Linear interpolation of the Cartesian endpoints can place two atoms
//! on top of each other. The path below keeps the endpoints and relaxes
//! every interior image toward the linearly interpolated pair distances
//! before any surface call. Distances and the endpoint step use the
//! minimum image.
//!
//! [`sidpp_path`] is the sequential start of Schmerwitz, Asgeirsson,
//! and Jonsson, J. Chem. Theory Comput. 20, 155 (2024),
//! doi:10.1021/acs.jctc.3c01111. It grows the band from both endpoints
//! on the same objective instead of starting from the straight line.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};

use crate::error::SaddleError;
use crate::mic::{self, Cell};
use crate::trics::{TricsConfig, trics_path};

/// Initial Cartesian path supplied to a band or Gaussian process band.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub enum PathInitializer {
    /// Straight minimum image interpolation.
    #[default]
    Linear,
    /// Image dependent pair potential interpolation.
    Idpp(IdppConfig),
    /// TRICS refinement of a configured seed path.
    Trics(TricsStartConfig),
}

/// Seed path refined by [`PathInitializer::Trics`].
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub enum TricsSeed {
    /// Straight Cartesian seed.
    #[default]
    Linear,
    /// IDPP seed.
    Idpp(IdppConfig),
}

/// Controls for a TRICS path initializer.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct TricsStartConfig {
    /// Path used before TRICS segmentation.
    pub seed: TricsSeed,
    /// TRICS coordinate, diagnostic, inversion, and splice controls.
    pub trics: TricsConfig,
}

with_setters!(TricsStartConfig {
    with_seed => seed: val TricsSeed,
    with_trics => trics: val TricsConfig,
});

/// Build a path with a generic initializer.
///
/// This is the common start option for NEB and Gaussian process NEB
/// drivers. TRICS currently requires unwrapped free space coordinates.
pub fn initialize_path(
    reactant: ArrayView1<f64>,
    product: ArrayView1<f64>,
    n_images: usize,
    cell: Option<&Cell>,
    initializer: &PathInitializer,
) -> Result<Array2<f64>, SaddleError> {
    match initializer {
        PathInitializer::Linear => linear_path(reactant, product, n_images, cell),
        PathInitializer::Idpp(config) => idpp_path(reactant, product, n_images, cell, config),
        PathInitializer::Trics(config) => {
            if cell.is_some() {
                return Err(SaddleError::Invalid(
                    "TRICS path initialization needs unwrapped free space coordinates".into(),
                ));
            }
            let initial = match &config.seed {
                TricsSeed::Linear => linear_path(reactant, product, n_images, None)?,
                TricsSeed::Idpp(idpp) => idpp_path(reactant, product, n_images, None, idpp)?,
            };
            Ok(trics_path(initial.view(), &config.trics)?.path)
        }
    }
}

/// Per-image stop for [`idpp_path`].
///
/// The defaults are 5000 L-BFGS steps, a gradient L2 norm of `1e-3`,
/// a per-atom step cap of `0.1`, and 20 L-BFGS pairs.
///
/// Non-exhaustive: start from [`IdppConfig::default`] and chain the
/// `with_*` setters.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct IdppConfig {
    /// L-BFGS iteration cap per image; at least 1. Default 5000.
    pub max_iterations: usize,
    /// Gradient L2 norm at which an image stops, in Angstrom^-3 (the IDPP
    /// objective is in Angstrom^-2); must be finite and positive. Default `1e-3`.
    pub grad_tol: f64,
    /// Largest per-atom displacement of one step, in Angstrom; finite and positive.
    /// Default 0.1.
    pub max_move: f64,
    /// Number of vector pairs a limited-memory stepper stores; at least 1. Default 20.
    pub memory: usize,
}

with_setters!(IdppConfig {
    with_max_iterations => max_iterations: val usize,
    with_grad_tol => grad_tol: val f64,
    with_max_move => max_move: val f64,
    with_memory => memory: val usize,
});

impl Default for IdppConfig {
    fn default() -> Self {
        Self {
            max_iterations: 5000,
            grad_tol: 1e-3,
            max_move: 0.1,
            memory: 20,
        }
    }
}

/// Straight path through the minimum image.
///
/// `n_images` counts the endpoints. Row 0 is `reactant` and the last
/// row is `product`. Interior row `i` is
/// `reactant + (i / (n_images - 1)) * mic(product - reactant)`.
/// `cell` is `None` in free space.
///
/// `reactant` and `product` are flat `3N` positions in Angstrom (atom-major) and
/// `n_images >= 3`. The result has `n_images` rows of `3N` columns.
///
/// # Errors
///
/// [`SaddleError::Shape`] when the endpoints differ in length, are empty or not a
/// multiple of 3, or `n_images < 3`; [`SaddleError::NonFinite`] when an endpoint
/// holds a non-finite value.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use rgsaddle::idpp::linear_path;
///
/// let reactant = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
/// let product = array![0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
/// let path = linear_path(reactant.view(), product.view(), 3, None)?;
/// // The middle image is the plain average of the endpoints.
/// assert!((path[(1, 3)] - 0.5).abs() < 0.001);
/// assert!((path[(1, 4)] - 0.5).abs() < 0.001);
/// # Ok::<(), rgsaddle::SaddleError>(())
/// ```
pub fn linear_path(
    reactant: ArrayView1<f64>,
    product: ArrayView1<f64>,
    n_images: usize,
    cell: Option<&Cell>,
) -> Result<Array2<f64>, SaddleError> {
    check_ends(reactant, product, n_images)?;
    let dof = reactant.len();
    let mut sep = &product - &reactant;
    if let Some(cell) = cell {
        mic::wrap_difference(cell, &mut sep);
    }
    let denom = (n_images - 1) as f64;
    sep /= denom;
    let mut path = Array2::zeros((n_images, dof));
    path.row_mut(0).assign(&reactant);
    path.row_mut(n_images - 1).assign(&product);
    for i in 1..n_images - 1 {
        let row = &reactant + &sep * (i as f64);
        path.row_mut(i).assign(&row);
    }
    Ok(path)
}

/// IDPP path. Same arguments as [`linear_path`].
///
/// Interior image `i` (fraction `xi = i / (n_images - 1)`) is minimized
/// on
///
/// `S = 1/2 sum_{a<b} (r_ab - d_ab)^2 / r_ab^4`,
///
/// where `r_ab` is the current minimum-image distance and `d_ab` is
/// `(1 - xi) r_ab(reactant) + xi r_ab(product)`. A pair closer than
/// `1e-4` is held at that floor so the weight stays finite. The
/// endpoints are copied through. The minimizer is L-BFGS. A step that
/// hits the iteration cap is kept.
///
/// Arguments and result as [`linear_path`]; `config` sets the L-BFGS stop.
///
/// # Errors
///
/// [`SaddleError::Invalid`] when `config` has a zero iteration cap or memory, or a
/// non-positive or non-finite gradient tolerance or step cap; the errors of
/// [`linear_path`]; and [`SaddleError::NonFinite`] or [`SaddleError::Shape`] from
/// the minimum-image distances.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use rgsaddle::idpp::{IdppConfig, idpp_path, linear_path};
///
/// // One atom swings around a second one at 1 Angstrom.
/// let reactant = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
/// let product = array![0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
/// let distance = |row: &[f64]| {
///     ((row[3] - row[0]).powi(2) + (row[4] - row[1]).powi(2) + (row[5] - row[2]).powi(2)).sqrt()
/// };
/// let line = linear_path(reactant.view(), product.view(), 3, None)?;
/// let path = idpp_path(reactant.view(), product.view(), 3, None, &IdppConfig::default())?;
/// // The straight line pulls the pair together; IDPP keeps the 1 Angstrom bond.
/// assert!(distance(line.row(1).as_slice().unwrap()) < 0.75);
/// assert!((distance(path.row(1).as_slice().unwrap()) - 1.0).abs() < 0.05);
/// # Ok::<(), rgsaddle::SaddleError>(())
/// ```
pub fn idpp_path(
    reactant: ArrayView1<f64>,
    product: ArrayView1<f64>,
    n_images: usize,
    cell: Option<&Cell>,
    config: &IdppConfig,
) -> Result<Array2<f64>, SaddleError> {
    check_config(config)?;
    let mut path = linear_path(reactant, product, n_images, cell)?;
    let d_init = distance_matrix(reactant, cell)?;
    let d_final = distance_matrix(product, cell)?;
    let denom = (n_images - 1) as f64;
    for i in 1..n_images - 1 {
        let xi = (i as f64) / denom;
        let target = &d_init * (1.0 - xi) + &d_final * xi;
        let mut image = path.row(i).to_owned();
        relax_image(&mut image, target.view(), cell, config)?;
        path.row_mut(i).assign(&image);
    }
    Ok(path)
}

/// One IDPP image between `left` and `right`, relaxed from `start`.
///
/// The target pair distances are the mean of the two ends, the
/// `xi = 1/2` row of [`idpp_path`]. With `start` at the minimum-image
/// midpoint this is the interior row of a three-image `idpp_path`.
pub(crate) fn idpp_midpoint(
    left: ArrayView1<f64>,
    right: ArrayView1<f64>,
    start: Array1<f64>,
    cell: Option<&Cell>,
    config: &IdppConfig,
) -> Result<Array1<f64>, SaddleError> {
    check_config(config)?;
    let xi = 0.5;
    let target = &distance_matrix(left, cell)? * (1.0 - xi) + &distance_matrix(right, cell)? * xi;
    let mut image = start;
    relax_image(&mut image, target.view(), cell, config)?;
    Ok(image)
}

/// Spring, stop, and step settings for [`sidpp_path`].
///
/// `spring = 1`, `force_tol = 0.01` and `rms_tol = 0.005` are the
/// settings of Schmerwitz, Asgeirsson, and Jonsson, Sec. 2.3.
/// `max_move` and `max_iterations` are the [`IdppConfig`] defaults,
/// spent once per growth stage. `time_step` is the fixed quick-min step.
///
/// Non-exhaustive: start from [`SidppConfig::default`] and chain the
/// `with_*` setters.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SidppConfig {
    /// Base spring constant `k` of Eqs. 5 and 10, in Angstrom^-4 (a force in
    /// Angstrom^-3 per Angstrom of stretch); finite and positive. Default 1.
    pub spring: f64,
    /// Largest NEB force component that ends a stage, in Angstrom^-3; finite and
    /// positive. Default 0.01.
    pub force_tol: f64,
    /// Root mean square NEB force that, with `force_tol`, ends the final stage, in
    /// Angstrom^-3; finite and positive. Default 0.005.
    pub rms_tol: f64,
    /// Initial time step of FIRE and quick-min; finite and positive. Default 0.1.
    pub time_step: f64,
    /// Largest per-atom displacement of one step, in Angstrom; finite and positive.
    /// Default 0.1.
    pub max_move: f64,
    /// Iteration cap per growth stage and for the final relaxation; at least 1.
    /// Default 5000.
    pub max_iterations: usize,
}

with_setters!(SidppConfig {
    with_spring => spring: val f64,
    with_force_tol => force_tol: val f64,
    with_rms_tol => rms_tol: val f64,
    with_time_step => time_step: val f64,
    with_max_move => max_move: val f64,
    with_max_iterations => max_iterations: val usize,
});

impl Default for SidppConfig {
    fn default() -> Self {
        let idpp = IdppConfig::default();
        Self {
            spring: 1.0,
            force_tol: 0.01,
            rms_tol: 0.005,
            time_step: 0.1,
            max_move: idpp.max_move,
            max_iterations: idpp.max_iterations,
        }
    }
}

/// Sequential IDPP path (S-IDPP). Same arguments as [`linear_path`].
///
/// Image `l` keeps the target of Eq. 1 for its final slot. The first
/// stage places images 1 and `n_images - 2` on the straight line, a
/// distance `d_id` (Eq. 11) from their endpoints. Each stage relaxes
/// the placed images under the NEB force of Eqs. 3 to 9, where the
/// segment across the gap between the two parts has the spring of
/// Eq. 10 and every other segment has `spring`. Once the frontier
/// image on one side is under `force_tol`, a new image goes on that
/// side, `d_id` from it along the Eq. 7 tangent, and `d_id` is
/// recomputed from the current path length. When both frontiers are
/// under `force_tol`, the side with fewer images grows, the reactant
/// side on a tie. The step never passes the middle of the gap. When
/// every slot is filled, the band relaxes with equal springs until
/// `force_tol` and `rms_tol` both hold. The minimizer is velocity
/// projection (quick-min). A stage that hits the iteration cap is
/// kept, and the side whose frontier force is smaller grows.
///
/// Arguments and result as [`linear_path`]; `config` sets the springs and stops.
///
/// # Errors
///
/// [`SaddleError::Invalid`] when `config` has a zero iteration cap or a
/// non-positive or non-finite setting, and the errors of [`idpp_path`].
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use rgsaddle::idpp::{SidppConfig, sidpp_path};
///
/// let reactant = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
/// let product = array![0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
/// let path = sidpp_path(reactant.view(), product.view(), 5, None, &SidppConfig::default())?;
/// assert_eq!(path.dim(), (5, 6));
/// // The endpoints are copied through.
/// assert_eq!(path.row(0), reactant);
/// assert_eq!(path.row(4), product);
/// # Ok::<(), rgsaddle::SaddleError>(())
/// ```
pub fn sidpp_path(
    reactant: ArrayView1<f64>,
    product: ArrayView1<f64>,
    n_images: usize,
    cell: Option<&Cell>,
    config: &SidppConfig,
) -> Result<Array2<f64>, SaddleError> {
    check_sidpp(config)?;
    let line = linear_path(reactant, product, n_images, cell)?;
    let d_init = distance_matrix(reactant, cell)?;
    let d_final = distance_matrix(product, cell)?;
    let last = n_images - 1;
    let target = |slot: usize| {
        let xi = slot as f64 / last as f64;
        &d_init * (1.0 - xi) + &d_final * xi
    };
    let dof = reactant.len();
    let seg = |a: &Array1<f64>, b: &Array1<f64>| norm(&mic_diff(a.view(), b.view(), cell));
    let mut images = vec![reactant.to_owned(), line.row(1).to_owned()];
    if n_images > 3 {
        images.push(line.row(last - 1).to_owned());
    }
    images.push(product.to_owned());
    let mut n_left = 1;
    let mut d_id = norm(&mic_diff(product, reactant, cell)) / last as f64;
    loop {
        let len = images.len();
        let full = len == n_images;
        let slot = |k: usize| if k <= n_left { k } else { last + k + 1 - len };
        let mut springs = vec![config.spring; len - 1];
        if !full {
            springs[n_left] = config.spring * d_id / seg(&images[n_left + 1], &images[n_left]);
        }
        let band = GrowBand {
            first: reactant.to_owned(),
            last: product.to_owned(),
            targets: (0..len).map(|k| target(slot(k))).collect(),
            springs,
            cell: cell.copied(),
        };
        let mut x = Array1::from_iter(images[1..len - 1].iter().flatten().copied());
        let image_max = |f: &Array1<f64>, i: usize| max_abs(f.slice(s![i * dof..(i + 1) * dof]));
        let front = [n_left - 1, n_left];
        let (values, force) = relax(&band, &mut x, config, |f| {
            if full {
                max_abs(f.view()) < config.force_tol
                    && (f.dot(f) / f.len() as f64).sqrt() < config.rms_tol
            } else {
                front.iter().any(|&i| image_max(f, i) < config.force_tol)
            }
        })?;
        images = band.images(x.view());
        if full {
            return Ok(Array2::from_shape_fn((n_images, dof), |(i, j)| {
                images[i][j]
            }));
        }
        d_id = images.windows(2).map(|w| seg(&w[1], &w[0])).sum::<f64>() / last as f64;
        let (f_left, f_right) = (image_max(&force, front[0]), image_max(&force, front[1]));
        let grow_left = if f_left < config.force_tol && f_right < config.force_tol {
            2 * n_left <= len - 2
        } else {
            f_left <= f_right
        };
        let b = if grow_left { n_left } else { n_left + 1 };
        let plus = mic_diff(images[b + 1].view(), images[b].view(), cell);
        let minus = mic_diff(images[b].view(), images[b - 1].view(), cell);
        let tau = tangent(&plus, &minus, &values[b - 1..b + 2], true);
        let step = d_id.min(0.5 * seg(&images[n_left + 1], &images[n_left]));
        let new = &images[b] + &(tau * if grow_left { step } else { -step });
        images.insert(n_left + 1, new);
        n_left += usize::from(grow_left);
    }
}

/// Partial S-IDPP band: the endpoints, one Eq. 1 target per image, and
/// one spring constant per segment.
#[derive(Clone)]
struct GrowBand {
    first: Array1<f64>,
    last: Array1<f64>,
    targets: Vec<Array2<f64>>,
    springs: Vec<f64>,
    cell: Option<Cell>,
}

impl GrowBand {
    fn images(&self, x: ArrayView1<f64>) -> Vec<Array1<f64>> {
        let dof = self.first.len();
        let mut images = vec![self.first.clone()];
        images.extend(x.exact_chunks(dof).into_iter().map(|c| c.to_owned()));
        images.push(self.last.clone());
        images
    }

    /// Objective of every image and the NEB force of Eqs. 3 to 9 on the
    /// moving images, flattened.
    fn force(&self, x: ArrayView1<f64>) -> Result<(Vec<f64>, Array1<f64>), SaddleError> {
        let dof = self.first.len();
        let images = self.images(x);
        let mut values = Vec::with_capacity(images.len());
        let mut grads = Vec::with_capacity(images.len());
        // Eq. 2 carries no 1/2, so the objective is twice the one that
        // idpp_value_grad minimizes.
        for (image, target) in images.iter().zip(&self.targets) {
            let (value, grad) = idpp_value_grad(image.view(), target.view(), self.cell.as_ref())?;
            values.push(2.0 * value);
            grads.push(grad * 2.0);
        }
        let mut force = Array1::zeros(x.len());
        for l in 1..images.len() - 1 {
            let plus = mic_diff(images[l + 1].view(), images[l].view(), self.cell.as_ref());
            let minus = mic_diff(images[l].view(), images[l - 1].view(), self.cell.as_ref());
            let tau = tangent(&plus, &minus, &values[l - 1..l + 2], false);
            let spring = self.springs[l] * norm(&plus) - self.springs[l - 1] * norm(&minus);
            let f = &tau * (grads[l].dot(&tau) + spring) - &grads[l];
            force.slice_mut(s![(l - 1) * dof..l * dof]).assign(&f);
        }
        Ok((values, force))
    }
}

/// Eq. 6 on a monotone triple of objective values, otherwise Eq. 7 on
/// the unit segment vectors. `weighted` always takes Eq. 7. A flat
/// triple weighs both segments equally.
fn tangent(plus: &Array1<f64>, minus: &Array1<f64>, s: &[f64], weighted: bool) -> Array1<f64> {
    let (prev, here, next) = (s[0], s[1], s[2]);
    let tau = if !weighted && next > here && here > prev {
        plus.clone()
    } else if !weighted && next < here && here < prev {
        minus.clone()
    } else {
        let (up, down) = ((next - here).abs(), (prev - here).abs());
        let (hi, lo) = (up.max(down), up.min(down));
        let (p, m) = (unit(plus), unit(minus));
        if hi == 0.0 {
            p + m
        } else if next > prev {
            p * hi + m * lo
        } else {
            p * lo + m * hi
        }
    };
    unit(&tau)
}

fn relax(
    band: &GrowBand,
    x: &mut Array1<f64>,
    config: &SidppConfig,
    done: impl Fn(&Array1<f64>) -> bool,
) -> Result<(Vec<f64>, Array1<f64>), SaddleError> {
    use rgmin::{Control, Method, Oracle, QuickMinOptions, Solver};

    let surface = band.clone();
    let oracle = Oracle::unbounded(x.len(), move |y: ArrayView1<f64>| match surface.force(y) {
        Ok((values, force)) => (values.iter().sum(), -force),
        Err(_) => (f64::NAN, Array1::from_elem(y.len(), f64::NAN)),
    });
    let control = Control::default()
        .with_maxiter(usize::MAX)
        .with_gtol(0.0)
        .with_istep(config.time_step)
        .with_maxmove(None);
    let mut solver = Solver::new(Method::QuickMin, control, x.len());
    solver.set_atom_maxmove(config.max_move);
    solver.set_quickmin_options(QuickMinOptions::eon());
    let mut state = band.force(x.view())?;
    for _ in 0..config.max_iterations {
        if done(&state.1) {
            break;
        }
        solver
            .step(&oracle, x)
            .map_err(|err| SaddleError::Solver(err.to_string()))?;
        if !x.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("S-IDPP image"));
        }
        state = band.force(x.view())?;
    }
    Ok(state)
}

fn mic_diff(a: ArrayView1<f64>, b: ArrayView1<f64>, cell: Option<&Cell>) -> Array1<f64> {
    let mut d = &a - &b;
    if let Some(cell) = cell {
        mic::wrap_difference(cell, &mut d);
    }
    d
}

fn norm(v: &Array1<f64>) -> f64 {
    v.dot(v).sqrt()
}

fn unit(v: &Array1<f64>) -> Array1<f64> {
    let n = norm(v);
    if n > 0.0 { v / n } else { v.clone() }
}

fn max_abs(v: ArrayView1<f64>) -> f64 {
    v.iter().fold(0.0, |m, x| m.max(x.abs()))
}

fn check_sidpp(config: &SidppConfig) -> Result<(), SaddleError> {
    let c = config;
    let positive = [c.spring, c.force_tol, c.rms_tol, c.time_step, c.max_move];
    if c.max_iterations == 0 || !positive.iter().all(|v| *v > 0.0 && v.is_finite()) {
        return Err(SaddleError::Invalid(
            "S-IDPP needs an iteration and finite positive settings".into(),
        ));
    }
    Ok(())
}

pub(crate) fn check_config(config: &IdppConfig) -> Result<(), SaddleError> {
    if config.max_iterations == 0 {
        return Err(SaddleError::Invalid(
            "IDPP needs at least one iteration".into(),
        ));
    }
    if !(config.grad_tol > 0.0 && config.grad_tol.is_finite()) {
        return Err(SaddleError::Invalid(
            "IDPP gradient tolerance must be finite and positive".into(),
        ));
    }
    if !(config.max_move > 0.0 && config.max_move.is_finite()) {
        return Err(SaddleError::Invalid(
            "IDPP max move must be finite and positive".into(),
        ));
    }
    if config.memory == 0 {
        return Err(SaddleError::Invalid("IDPP L-BFGS memory is empty".into()));
    }
    Ok(())
}

fn check_ends(
    reactant: ArrayView1<f64>,
    product: ArrayView1<f64>,
    n_images: usize,
) -> Result<(), SaddleError> {
    let dof = reactant.len();
    if product.len() != dof || dof == 0 || !dof.is_multiple_of(3) {
        return Err(SaddleError::Shape(format!(
            "IDPP endpoints must be matching 3N vectors; got {dof} and {}",
            product.len()
        )));
    }
    if n_images < 3 {
        return Err(SaddleError::Shape(format!(
            "IDPP path needs at least 3 images; got {n_images}"
        )));
    }
    if !reactant.iter().all(|v| v.is_finite()) {
        return Err(SaddleError::NonFinite("IDPP reactant"));
    }
    if !product.iter().all(|v| v.is_finite()) {
        return Err(SaddleError::NonFinite("IDPP product"));
    }
    Ok(())
}

/// Floor on a pair distance. Below it the `1/r^4` weight is undefined.
const R_FLOOR: f64 = 1e-4;

fn distance_matrix(pos: ArrayView1<f64>, cell: Option<&Cell>) -> Result<Array2<f64>, SaddleError> {
    let n = pos.len() / 3;
    let mut d = Array2::zeros((n, n));
    let mut diff = Array1::zeros(3);
    for i in 0..n {
        for j in (i + 1)..n {
            let r = pair_distance(pos, i, j, cell, &mut diff)?;
            d[(i, j)] = r;
            d[(j, i)] = r;
        }
    }
    Ok(d)
}

fn pair_distance(
    pos: ArrayView1<f64>,
    i: usize,
    j: usize,
    cell: Option<&Cell>,
    diff: &mut Array1<f64>,
) -> Result<f64, SaddleError> {
    for c in 0..3 {
        diff[c] = pos[3 * i + c] - pos[3 * j + c];
    }
    if let Some(cell) = cell {
        mic::wrap_difference(cell, diff);
    }
    let r2 = diff.dot(diff);
    if !r2.is_finite() {
        return Err(SaddleError::NonFinite("IDPP pair distance"));
    }
    Ok(r2.sqrt())
}

fn idpp_value_grad(
    pos: ArrayView1<f64>,
    target: ArrayView2<f64>,
    cell: Option<&Cell>,
) -> Result<(f64, Array1<f64>), SaddleError> {
    let n = pos.len() / 3;
    let mut energy = 0.0;
    let mut grad = Array1::zeros(pos.len());
    let mut diff = Array1::zeros(3);
    for i in 0..n {
        for j in (i + 1)..n {
            for c in 0..3 {
                diff[c] = pos[3 * i + c] - pos[3 * j + c];
            }
            if let Some(cell) = cell {
                mic::wrap_difference(cell, &mut diff);
            }
            let mut r = diff.dot(&diff).sqrt();
            if !r.is_finite() {
                return Err(SaddleError::NonFinite("IDPP pair distance"));
            }
            if r < R_FLOOR {
                r = R_FLOOR;
            }
            let delta = r - target[(i, j)];
            let r2 = r * r;
            let r4 = r2 * r2;
            energy += 0.5 * delta * delta / r4;
            // dS/dr = delta * (1 - 2 delta / r) / r^4
            let d_s_dr = (delta * (1.0 - 2.0 * delta / r)) / r4;
            let scale = d_s_dr / r;
            for c in 0..3 {
                let g = scale * diff[c];
                grad[3 * i + c] += g;
                grad[3 * j + c] -= g;
            }
        }
    }
    if !energy.is_finite() || !grad.iter().all(|g: &f64| g.is_finite()) {
        return Err(SaddleError::NonFinite("IDPP objective"));
    }
    Ok((energy, grad))
}

fn relax_image(
    image: &mut Array1<f64>,
    target: ArrayView2<f64>,
    cell: Option<&Cell>,
    config: &IdppConfig,
) -> Result<(), SaddleError> {
    use rgmin::{Accept, Control, Method, Oracle, Solver};

    let dim = image.len();
    let target = target.to_owned();
    let cell = cell.copied();
    let oracle = Oracle::unbounded(dim, move |y: ArrayView1<f64>| {
        match idpp_value_grad(y, target.view(), cell.as_ref()) {
            Ok(pair) => pair,
            Err(_) => (f64::NAN, Array1::from_elem(y.len(), f64::NAN)),
        }
    });
    let mut solver = Solver::new(
        Method::lbfgs_with_memory(config.memory),
        Control::default()
            .with_maxiter(config.max_iterations)
            .with_gtol(config.grad_tol)
            .with_istep(1.0)
            .with_maxmove(None),
        dim,
    );
    solver.set_atom_maxmove(config.max_move);
    solver.set_accept(Accept::Energy);
    for _ in 0..config.max_iterations {
        let report = solver
            .step(&oracle, image)
            .map_err(|err| SaddleError::Solver(err.to_string()))?;
        if !image.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("IDPP image"));
        }
        if report.grad_norm < config.grad_tol {
            return Ok(());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn gradient_matches_a_central_difference() {
        let pos = array![0.0, 0.0, 0.0, 1.2, 0.1, 0.0, 0.4, 1.1, 0.3];
        let cell = Cell::ortho(6.0, 6.0, 6.0).unwrap();
        let mut target = distance_matrix(pos.view(), Some(&cell)).unwrap();
        target *= 1.4;
        let (_e, grad) = idpp_value_grad(pos.view(), target.view(), Some(&cell)).unwrap();
        let eps = 1e-6;
        for k in 0..pos.len() {
            let mut plus = pos.clone();
            let mut minus = pos.clone();
            plus[k] += eps;
            minus[k] -= eps;
            let (ep, _) = idpp_value_grad(plus.view(), target.view(), Some(&cell)).unwrap();
            let (em, _) = idpp_value_grad(minus.view(), target.view(), Some(&cell)).unwrap();
            assert_abs_diff_eq!(grad[k], (ep - em) / (2.0 * eps), epsilon = 1e-6);
        }
    }
}
