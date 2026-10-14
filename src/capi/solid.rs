//! C entry points of the solid-state band (ABI minor 16).

use super::*;
use crate::band::SolidBandSurface;
use crate::solid::SolidStateConfig;

/// Solid-state band settings. The caller stamps version and zeroes
/// flags.
#[repr(C)]
pub struct RgsaddleBandSolidConfig {
    pub version: RgsaddleVersion,
    pub flags: u64,
    /// Multiplies the Jacobian; values <= 0 select 1.
    pub weight: f64,
    /// Hydrostatic pressure, energy per length cubed.
    pub pressure: f64,
}

/// Surface request of a solid-state band: positions and cells in,
/// energies, gradients and Cauchy stresses out.
#[repr(C)]
pub struct RgsaddleSolidSurfaceRequest {
    pub version: RgsaddleVersion,
    pub flags: u64,
    pub n_images: i64,
    pub n_atoms: i64,
    pub positions: *const f64,
    pub cells: *const f64,
    pub energies: *mut f64,
    pub gradients: *mut f64,
    pub stresses: *mut f64,
    /// -1 for a batch. Otherwise the band image this call fills.
    pub image: i64,
}

pub type RgsaddleSolidSurfaceFn =
    extern "C" fn(*mut c_void, *mut RgsaddleSolidSurfaceRequest) -> i32;

struct CSolidSurface {
    f: RgsaddleSolidSurfaceFn,
    user: *mut c_void,
    n_atoms: i64,
    n_images: i64,
    per_image: bool,
    rows: std::sync::atomic::AtomicUsize,
}

// The sessions call the surface only from the thread that called step.
unsafe impl Sync for CSolidSurface {}
unsafe impl Send for CSolidSurface {}

impl CSolidSurface {
    #[allow(clippy::too_many_arguments)]
    fn call(
        &self,
        n_images: i64,
        image: i64,
        positions: &[f64],
        cells: &[f64],
        energies: &mut [f64],
        gradients: &mut [f64],
        stresses: &mut [f64],
    ) -> Result<(), SaddleError> {
        let mut req = RgsaddleSolidSurfaceRequest {
            version: RgsaddleVersion {
                major: RGSADDLE_ABI_MAJOR,
                minor: RGSADDLE_ABI_MINOR,
            },
            flags: if image >= 0 { 1 } else { 0 },
            n_images,
            n_atoms: self.n_atoms,
            positions: positions.as_ptr(),
            cells: cells.as_ptr(),
            energies: energies.as_mut_ptr(),
            gradients: gradients.as_mut_ptr(),
            stresses: stresses.as_mut_ptr(),
            image,
        };
        self.rows
            .fetch_add(energies.len(), std::sync::atomic::Ordering::Relaxed);
        let rc = (self.f)(self.user, &mut req);
        if rc != 0 {
            return Err(SaddleError::Surface(format!("host callback rc={rc}")));
        }
        Ok(())
    }
}

impl SolidBandSurface for CSolidSurface {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        cells: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
        stresses: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        let n_rows = positions.nrows() as i64;
        let dof = positions.ncols();
        let first_image: i64 = if n_rows == self.n_images { 0 } else { 1 };
        if self.per_image {
            for i in 0..positions.nrows() {
                let p: Vec<f64> = positions.row(i).to_vec();
                let c: Vec<f64> = cells.row(i).to_vec();
                let mut e = [0.0];
                let mut g = vec![0.0; dof];
                let mut st = [0.0; 9];
                self.call(
                    self.n_images,
                    first_image + i as i64,
                    &p,
                    &c,
                    &mut e,
                    &mut g,
                    &mut st,
                )?;
                energies[i] = e[0];
                gradients.row_mut(i).assign(&ndarray::ArrayView1::from(&g));
                stresses.row_mut(i).assign(&ndarray::ArrayView1::from(&st));
            }
            return Ok(());
        }
        let p: Vec<f64> = positions.iter().copied().collect();
        let c: Vec<f64> = cells.iter().copied().collect();
        let mut e = vec![0.0; energies.len()];
        let mut g = vec![0.0; gradients.len()];
        let mut st = vec![0.0; stresses.len()];
        self.call(n_rows, -1, &p, &c, &mut e, &mut g, &mut st)?;
        energies.assign(&ndarray::ArrayView1::from(&e));
        for i in 0..gradients.nrows() {
            for k in 0..dof {
                gradients[(i, k)] = g[i * dof + k];
            }
            for k in 0..9 {
                stresses[(i, k)] = st[i * 9 + k];
            }
        }
        Ok(())
    }
}

/// Band rows of `n_images` images: `3 n_atoms` positions then 9 cell
/// entries each.
fn solid_rows(n_images: usize, n_atoms: usize, positions: &[f64], cells: &[f64]) -> Array2<f64> {
    let dof = 3 * n_atoms;
    let mut rows = Array2::zeros((n_images, dof + 9));
    for i in 0..n_images {
        for k in 0..dof {
            rows[(i, k)] = positions[i * dof + k];
        }
        for k in 0..9 {
            rows[(i, dof + k)] = cells[i * 9 + k];
        }
    }
    rows
}

/// # Safety
/// `config` and `solid` must be valid; `positions` holds
/// `n_images * 3 * n_atoms` doubles and `cells` `n_images * 9`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_band_create_solid(
    config: *const RgsaddleBandConfig,
    solid: *const RgsaddleBandSolidConfig,
    n_images: i64,
    n_atoms: i64,
    positions: *const f64,
    cells: *const f64,
) -> *mut RgsaddleBand {
    if config.is_null()
        || solid.is_null()
        || positions.is_null()
        || cells.is_null()
        || n_images < 3
        || n_atoms < 1
    {
        return std::ptr::null_mut();
    }
    let (cfg, ss) = unsafe { (&*config, &*solid) };
    if cfg.version.major != RGSADDLE_ABI_MAJOR || ss.version.major != RGSADDLE_ABI_MAJOR {
        return std::ptr::null_mut();
    }
    // The trust-region stepper has no cell block.
    if cfg.method == 2 {
        return std::ptr::null_mut();
    }
    let (ni, na) = (n_images as usize, n_atoms as usize);
    let p = unsafe { slice::from_raw_parts(positions, ni * 3 * na) };
    let c = unsafe { slice::from_raw_parts(cells, ni * 9) };
    let rows = solid_rows(ni, na, p, c);
    let Some(mut band_config) = (unsafe { band_config_of(cfg, n_images, 3 * na + 9) }) else {
        return std::ptr::null_mut();
    };
    band_config.solid_state = Some(
        SolidStateConfig::new()
            .with_weight(if ss.weight > 0.0 { ss.weight } else { 1.0 })
            .with_pressure(ss.pressure),
    );
    match BandSession::new(band_config, rows) {
        Ok(session) => Box::into_raw(Box::new(RgsaddleBand {
            session,
            n_images,
            n_atoms,
            per_image: cfg.flags & RGSADDLE_BAND_PER_IMAGE != 0,
            solid: true,
        })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// # Safety
/// `band` and `out` must be valid; `surface` is called with `user`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_band_step_solid(
    band: *mut RgsaddleBand,
    surface: Option<RgsaddleSolidSurfaceFn>,
    user: *mut c_void,
    out: *mut RgsaddleReport,
) -> i32 {
    if band.is_null() {
        return RGSADDLE_NULL_BAND;
    }
    if out.is_null() {
        return RGSADDLE_NULL_REPORT;
    }
    let Some(f) = surface else {
        return RGSADDLE_NULL_SURFACE;
    };
    let band = unsafe { &mut *band };
    if !band.solid {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let cs = CSolidSurface {
        f,
        user,
        n_atoms: band.n_atoms,
        n_images: band.n_images,
        per_image: band.per_image,
        rows: Default::default(),
    };
    match band.session.step_solid(&cs) {
        Ok(report) => {
            let out = unsafe { &mut *out };
            stamp_report(out);
            out.status = i32::from(report.status == BandStatus::Converged);
            out.evaluations = i32::try_from(report.surface_rows).unwrap_or(i32::MAX);
            out.max_force = report.max_force;
            out.ci_index = report.ci_index.map(|i| i as i64).unwrap_or(-1);
            out.iteration = report.iteration as i64;
            out.curvature = 0.0;
            out.rotations = 0;
            RGSADDLE_OK
        }
        Err(e) => status_of(&e),
    }
}

/// # Safety
/// `out` must hold `n_images * 9` doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_band_cells(band: *const RgsaddleBand, out: *mut f64) -> i32 {
    if band.is_null() {
        return RGSADDLE_NULL_BAND;
    }
    let band = unsafe { &*band };
    if out.is_null() || !band.solid {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let dof = band.atom_dof();
    let dst = unsafe { slice::from_raw_parts_mut(out, band.n_images as usize * 9) };
    for (i, row) in band.session.positions().outer_iter().enumerate() {
        for k in 0..9 {
            dst[i * 9 + k] = row[dof + k];
        }
    }
    RGSADDLE_OK
}

/// # Safety
/// `positions` holds `n_images * 3 * n_atoms` doubles, `cells`
/// `n_images * 9`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_band_set_solid(
    band: *mut RgsaddleBand,
    positions: *const f64,
    cells: *const f64,
) -> i32 {
    if band.is_null() {
        return RGSADDLE_NULL_BAND;
    }
    let band = unsafe { &mut *band };
    if positions.is_null() || cells.is_null() || !band.solid {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let (ni, na) = (band.n_images as usize, band.n_atoms as usize);
    let p = unsafe { slice::from_raw_parts(positions, ni * 3 * na) };
    let c = unsafe { slice::from_raw_parts(cells, ni * 9) };
    match band.session.set_positions(solid_rows(ni, na, p, c)) {
        Ok(()) => RGSADDLE_OK,
        Err(e) => status_of(&e),
    }
}

/// # Safety
/// Each non-NULL output must hold its documented length: `stresses`
/// `n_images * 9`, `projected_cells` `(n_images - 2) * 9`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_band_solid_evaluation(
    band: *const RgsaddleBand,
    stresses: *mut f64,
    projected_cells: *mut f64,
) -> i32 {
    if band.is_null() {
        return RGSADDLE_NULL_BAND;
    }
    let band = unsafe { &*band };
    if !band.solid {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let Some(eval) = band.session.evaluation() else {
        return RGSADDLE_NO_EVALUATION;
    };
    let dof = band.atom_dof();
    let copy = |dst: *mut f64, src: &Array2<f64>| {
        if !dst.is_null() {
            let out = unsafe { slice::from_raw_parts_mut(dst, src.nrows() * 9) };
            for (i, row) in src.outer_iter().enumerate() {
                for k in 0..9 {
                    out[i * 9 + k] = row[dof + k];
                }
            }
        }
    };
    copy(stresses, &eval.gradients);
    copy(projected_cells, &eval.projected);
    RGSADDLE_OK
}

/// # Safety
/// `band` and `out` must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_band_solid_jacobian(band: *const RgsaddleBand, out: *mut f64) -> i32 {
    if band.is_null() {
        return RGSADDLE_NULL_BAND;
    }
    let band = unsafe { &*band };
    let Some(j) = band.session.solid_jacobian() else {
        return RGSADDLE_INVALID_PARAMETER;
    };
    if out.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    unsafe { *out = j };
    RGSADDLE_OK
}
