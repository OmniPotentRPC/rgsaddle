//! Geodesic interpolation.
//!
//! Zhu, Thompson, and Martinez, J. Chem. Phys. 150, 164103 (2019),
//! doi:10.1063/1.5090303.

use std::ffi::c_void;
use std::slice;

use minimage::Cell;
use ndarray::{Array1, Array2, ArrayView1};

use super::{
    RGSADDLE_ABI_MAJOR, RGSADDLE_ABI_MINOR, RGSADDLE_ABI_MISMATCH, RGSADDLE_INVALID_PARAMETER,
    RGSADDLE_NOT_CONVERGED, RGSADDLE_OK, RGSADDLE_SHAPE, RgsaddleVersion, status_of,
};
use crate::error::SaddleError;
use crate::geodesic::{
    GeodesicChart, GeodesicConfig, MorsePairChart, geodesic_interpolate, geodesic_path,
};

/// Wire controls for [`rgsaddle_geodesic_path`]. One hundred four bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RgsaddleGeodesicConfig {
    /// ABI version this value was built against.
    pub version: RgsaddleVersion,
    /// Must be 0.
    pub flags: u64,
    /// Largest image displacement of a finished pass.
    pub step_tol: f64,
    /// L2 norm of one image's length gradient at which that image stops.
    pub grad_tol: f64,
    /// Largest Euclidean displacement of one L-BFGS step.
    pub max_move: f64,
    /// Insert images when the lower bound is under this fraction of the length.
    pub lower_ratio: f64,
    /// Insert images when the upper bound exceeds this multiple of the length.
    pub upper_ratio: f64,
    /// Forward and reverse passes, and restarts after an insertion.
    pub max_sweeps: i64,
    /// L-BFGS iterations for one image inside one pass.
    pub max_iterations: i64,
    /// L-BFGS vector pairs.
    pub memory: i64,
    /// Cartesian pieces in the upper bound.
    pub bound_samples: i64,
    /// Image cap, endpoints included.
    pub max_images: i64,
    /// Seed of the noisy starts of [`rgsaddle_geodesic_morse`].
    pub seed: u64,
}

/// Host chart. `q` has `n_coords` entries and `b` is row major
/// `n_coords * dof`. Return 0 on success.
pub type RgsaddleGeodesicChartFn =
    Option<extern "C" fn(*mut c_void, *const f64, i64, *mut f64, *mut f64, i64) -> i32>;

struct CallbackChart {
    n_coords: usize,
    fun: extern "C" fn(*mut c_void, *const f64, i64, *mut f64, *mut f64, i64) -> i32,
    ctx: *mut c_void,
    cell: Option<Cell>,
}

// SAFETY: the stepper calls the chart on the thread that entered the C
// function. The host context is not handed to another thread.
unsafe impl Send for CallbackChart {}
unsafe impl Sync for CallbackChart {}

impl GeodesicChart for CallbackChart {
    fn n_coords(&self) -> usize {
        self.n_coords
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
        if q.len() != self.n_coords || b.dim() != (self.n_coords, x.len()) {
            return Err(SaddleError::Shape(
                "geodesic chart buffers do not match the path".into(),
            ));
        }
        if !q.is_standard_layout() || !b.is_standard_layout() {
            return Err(SaddleError::Shape(
                "geodesic chart buffers are not contiguous".into(),
            ));
        }
        let Ok(dof) = i64::try_from(x.len()) else {
            return Err(SaddleError::Invalid(
                "geodesic dimension does not fit a C integer".into(),
            ));
        };
        let Ok(n_coords) = i64::try_from(self.n_coords) else {
            return Err(SaddleError::Invalid(
                "geodesic coordinate count does not fit a C integer".into(),
            ));
        };
        let x_owned = x.to_owned();
        let rc = (self.fun)(
            self.ctx,
            x_owned.as_ptr(),
            dof,
            q.as_mut_ptr(),
            b.as_mut_ptr(),
            n_coords,
        );
        if rc != RGSADDLE_OK {
            return Err(SaddleError::Surface(
                "geodesic chart rejected a geometry".into(),
            ));
        }
        Ok(())
    }
}

fn positive_usize(value: i64) -> Result<usize, i32> {
    match usize::try_from(value) {
        Ok(parsed) if parsed > 0 => Ok(parsed),
        _ => Err(RGSADDLE_INVALID_PARAMETER),
    }
}

fn config_from(ptr: *const RgsaddleGeodesicConfig) -> Result<GeodesicConfig, i32> {
    if ptr.is_null() {
        return Ok(GeodesicConfig::default());
    }
    let raw = unsafe { &*ptr };
    if raw.version.major != RGSADDLE_ABI_MAJOR {
        return Err(RGSADDLE_ABI_MISMATCH);
    }
    if raw.flags != 0 {
        return Err(RGSADDLE_INVALID_PARAMETER);
    }
    let config = GeodesicConfig {
        step_tol: raw.step_tol,
        grad_tol: raw.grad_tol,
        max_move: raw.max_move,
        lower_ratio: raw.lower_ratio,
        upper_ratio: raw.upper_ratio,
        max_sweeps: positive_usize(raw.max_sweeps)?,
        max_iterations: positive_usize(raw.max_iterations)?,
        memory: positive_usize(raw.memory)?,
        bound_samples: positive_usize(raw.bound_samples)?,
        max_images: positive_usize(raw.max_images)?,
        seed: raw.seed,
    };
    config.check().map_err(|err| status_of(&err))?;
    Ok(config)
}

/// Row-major 3x3 lattice, or null for free space.
fn cell_from(ptr: *const f64) -> Result<Option<Cell>, i32> {
    if ptr.is_null() {
        return Ok(None);
    }
    let raw = unsafe { slice::from_raw_parts(ptr, 9) };
    if !raw.iter().all(|v| v.is_finite()) {
        return Err(RGSADDLE_INVALID_PARAMETER);
    }
    Cell::from_vectors(
        [raw[0], raw[1], raw[2]],
        [raw[3], raw[4], raw[5]],
        [raw[6], raw[7], raw[8]],
        [0.0, 0.0, 0.0],
    )
    .map(Some)
    .map_err(|_| RGSADDLE_INVALID_PARAMETER)
}

/// `count` doubles as a slice length: no overflow, and under the `isize::MAX` bytes a slice may span.
fn buffer_len(count: usize) -> Result<usize, i32> {
    match count.checked_mul(size_of::<f64>()) {
        Some(bytes) if bytes <= isize::MAX as usize => Ok(count),
        _ => Err(RGSADDLE_INVALID_PARAMETER),
    }
}

/// `a * b` doubles, checked.
fn product_len(a: usize, b: usize) -> Result<usize, i32> {
    a.checked_mul(b)
        .ok_or(RGSADDLE_INVALID_PARAMETER)
        .and_then(buffer_len)
}

fn as_i64(value: usize) -> Result<i64, i32> {
    i64::try_from(value).map_err(|_| RGSADDLE_INVALID_PARAMETER)
}

fn write_f64(ptr: *mut f64, value: f64) {
    if !ptr.is_null() {
        unsafe { *ptr = value };
    }
}

fn write_path(path: &Array2<f64>, out: *mut f64) -> i32 {
    if out.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let dst = unsafe { slice::from_raw_parts_mut(out, path.len()) };
    for (i, value) in path.iter().enumerate() {
        dst[i] = *value;
    }
    RGSADDLE_OK
}

/// # Safety
/// `out` is a live `rgsaddle_geodesic_config_t` or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_geodesic_config_default(out: *mut RgsaddleGeodesicConfig) -> i32 {
    if out.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let config = GeodesicConfig::default();
    let Ok(max_sweeps) = as_i64(config.max_sweeps) else {
        return RGSADDLE_INVALID_PARAMETER;
    };
    let Ok(max_iterations) = as_i64(config.max_iterations) else {
        return RGSADDLE_INVALID_PARAMETER;
    };
    let Ok(memory) = as_i64(config.memory) else {
        return RGSADDLE_INVALID_PARAMETER;
    };
    let Ok(bound_samples) = as_i64(config.bound_samples) else {
        return RGSADDLE_INVALID_PARAMETER;
    };
    let Ok(max_images) = as_i64(config.max_images) else {
        return RGSADDLE_INVALID_PARAMETER;
    };
    unsafe {
        *out = RgsaddleGeodesicConfig {
            version: RgsaddleVersion {
                major: RGSADDLE_ABI_MAJOR,
                minor: RGSADDLE_ABI_MINOR,
            },
            flags: 0,
            step_tol: config.step_tol,
            grad_tol: config.grad_tol,
            max_move: config.max_move,
            lower_ratio: config.lower_ratio,
            upper_ratio: config.upper_ratio,
            max_sweeps,
            max_iterations,
            memory,
            bound_samples,
            max_images,
            seed: config.seed,
        };
    }
    RGSADDLE_OK
}

/// # Safety
/// `path` holds `n_images * dof` doubles. `out` holds the same number.
/// `chart` is a live callback or null. `cell` is 9 row-major lattice
/// entries or null, and a non-null `cell` needs `dof` divisible by 3.
/// `config` is a live config or null. `length`, `lower`, and `upper`
/// are live doubles or null. `ctx` is the host pointer passed back to
/// `chart`. The chart applies the same minimum image convention to its
/// own distances.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_geodesic_path(
    n_images: i64,
    dof: i64,
    path: *const f64,
    n_coords: i64,
    chart: RgsaddleGeodesicChartFn,
    ctx: *mut c_void,
    cell: *const f64,
    config: *const RgsaddleGeodesicConfig,
    out: *mut f64,
    length: *mut f64,
    lower: *mut f64,
    upper: *mut f64,
) -> i32 {
    let Ok(n_images) = positive_usize(n_images) else {
        return RGSADDLE_INVALID_PARAMETER;
    };
    let Ok(dof) = positive_usize(dof) else {
        return RGSADDLE_INVALID_PARAMETER;
    };
    let Ok(n_coords) = positive_usize(n_coords) else {
        return RGSADDLE_INVALID_PARAMETER;
    };
    if n_images < 2 {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let Some(fun) = chart else {
        return RGSADDLE_INVALID_PARAMETER;
    };
    if path.is_null() || out.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let total = match product_len(n_images, dof) {
        Ok(total) => total,
        Err(code) => return code,
    };
    // The chart fills `n_coords * dof` Wilson entries per sample.
    if let Err(code) = product_len(n_coords, dof) {
        return code;
    }
    if !cell.is_null() && dof % 3 != 0 {
        return RGSADDLE_SHAPE;
    }
    let cell = match cell_from(cell) {
        Ok(cell) => cell,
        Err(code) => return code,
    };
    let mut config = match config_from(config) {
        Ok(config) => config,
        Err(code) => return code,
    };
    // `out` has one row per input image, so midpoint insertion stops there.
    if config.max_images > n_images {
        config.max_images = n_images;
    }
    let raw = unsafe { slice::from_raw_parts(path, total) };
    let owned = match Array2::from_shape_vec((n_images, dof), raw.to_vec()) {
        Ok(value) => value,
        Err(_) => return RGSADDLE_SHAPE,
    };
    let adapter = CallbackChart {
        n_coords,
        fun,
        ctx,
        cell,
    };
    match geodesic_path(owned.view(), &adapter, &config) {
        Ok((optimized, report)) => {
            let status = write_path(&optimized, out);
            if status != RGSADDLE_OK {
                return status;
            }
            write_f64(length, report.length);
            write_f64(lower, report.lower);
            write_f64(upper, report.upper);
            if report.converged {
                RGSADDLE_OK
            } else {
                RGSADDLE_NOT_CONVERGED
            }
        }
        Err(err) => status_of(&err),
    }
}

/// # Safety
/// `reactant`, `product`, and `radii` each address `n_atoms` atoms
/// (`3 * n_atoms` doubles for the geometries, `n_atoms` doubles for the
/// radii). `out` holds `n_images * 3 * n_atoms` doubles. `cell` is 9
/// row-major lattice entries or null. `config` is a live config or null.
/// `length`, `lower`, and `upper` are live doubles or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_geodesic_morse(
    n_images: i64,
    n_atoms: i64,
    reactant: *const f64,
    product: *const f64,
    radii: *const f64,
    cell: *const f64,
    config: *const RgsaddleGeodesicConfig,
    out: *mut f64,
    length: *mut f64,
    lower: *mut f64,
    upper: *mut f64,
) -> i32 {
    let Ok(n_images) = positive_usize(n_images) else {
        return RGSADDLE_INVALID_PARAMETER;
    };
    let Ok(n_atoms) = positive_usize(n_atoms) else {
        return RGSADDLE_INVALID_PARAMETER;
    };
    if n_images < 2 || n_atoms < 2 {
        return RGSADDLE_SHAPE;
    }
    if reactant.is_null() || product.is_null() || radii.is_null() || out.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let Some(dof) = n_atoms.checked_mul(3) else {
        return RGSADDLE_INVALID_PARAMETER;
    };
    if let Err(code) = buffer_len(dof) {
        return code;
    }
    // `out` holds `n_images * dof` doubles and each sample fills one
    // Wilson row of `dof` entries per atom pair.
    if let Err(code) = product_len(n_images, dof) {
        return code;
    }
    let Some(n_pairs) = n_atoms.checked_mul(n_atoms - 1).map(|doubled| doubled / 2) else {
        return RGSADDLE_INVALID_PARAMETER;
    };
    if let Err(code) = product_len(n_pairs, dof) {
        return code;
    }
    let mut config = match config_from(config) {
        Ok(config) => config,
        Err(code) => return code,
    };
    // `out` has `n_images` rows, so midpoint insertion stops there.
    if config.max_images > n_images {
        config.max_images = n_images;
    }
    let reactant_raw = unsafe { slice::from_raw_parts(reactant, dof) };
    let product_raw = unsafe { slice::from_raw_parts(product, dof) };
    let radii_raw = unsafe { slice::from_raw_parts(radii, n_atoms) };
    let mut chart = match MorsePairChart::new(Array1::from_vec(radii_raw.to_vec())) {
        Ok(chart) => chart,
        Err(err) => return status_of(&err),
    };
    match cell_from(cell) {
        Ok(parsed) => chart = chart.with_cell(parsed),
        Err(code) => return code,
    }
    let reactant = match Array1::from_shape_vec(dof, reactant_raw.to_vec()) {
        Ok(value) => value,
        Err(_) => return RGSADDLE_SHAPE,
    };
    let product = match Array1::from_shape_vec(dof, product_raw.to_vec()) {
        Ok(value) => value,
        Err(_) => return RGSADDLE_SHAPE,
    };
    match geodesic_interpolate(reactant.view(), product.view(), n_images, &chart, &config) {
        Ok((path, report)) => {
            let status = write_path(&path, out);
            if status != RGSADDLE_OK {
                return status;
            }
            write_f64(length, report.length);
            write_f64(lower, report.lower);
            write_f64(upper, report.upper);
            if report.converged {
                RGSADDLE_OK
            } else {
                RGSADDLE_NOT_CONVERGED
            }
        }
        Err(err) => status_of(&err),
    }
}
