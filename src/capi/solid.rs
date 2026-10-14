//! C entry points of the solid-state band (ABI minor 16): one surface
//! request carries each image's positions and cell in and its energy,
//! gradient and Cauchy stress out.

use super::*;

/// Surface request of a solid-state band. The session stamps version
/// and flags; the row rules are those of [`RgsaddleSurfaceRequest`].
#[repr(C)]
pub struct RgsaddleSolidSurfaceRequest {
    pub version: RgsaddleVersion,
    pub flags: u64,
    pub n_images: i64,
    pub n_atoms: i64,
    pub positions: *const f64,
    /// Row-major 3x3 cell per image, lower triangular.
    pub cells: *const f64,
    pub energies: *mut f64,
    pub gradients: *mut f64,
    /// Row-major Cauchy stress per image, `(1/V) dE/deps` for the right
    /// strain `h <- h (I + eps)` at fixed fractional coordinates.
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
            flags: u64::from(image >= 0),
            n_images,
            n_atoms: self.n_atoms,
            positions: positions.as_ptr(),
            cells: cells.as_ptr(),
            energies: energies.as_mut_ptr(),
            gradients: gradients.as_mut_ptr(),
            stresses: stresses.as_mut_ptr(),
            image,
        };
        let rc = (self.f)(self.user, &mut req);
        if rc != 0 {
            return Err(SaddleError::Surface(format!("host callback rc={rc}")));
        }
        Ok(())
    }
}

impl BandSurface for CSolidSurface {
    fn eval(
        &self,
        _positions: ArrayView2<f64>,
        _energies: &mut Array1<f64>,
        _gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        Err(SaddleError::Invalid(
            "a solid-state surface evaluates with cells".into(),
        ))
    }

    fn eval_cells(
        &self,
        positions: ArrayView2<f64>,
        cells: &[Cell],
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
        stresses: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        let n = positions.nrows();
        if cells.len() != n {
            return Err(SaddleError::Shape(
                "solid-state request has a cell per image".into(),
            ));
        }
        let dof = positions.ncols();
        let flat_cell = |c: &Cell| -> Vec<f64> { c.0.iter().flatten().copied().collect() };
        let first_image: i64 = if n as i64 == self.n_images { 0 } else { 1 };
        if self.per_image {
            for i in 0..n {
                let p: Vec<f64> = positions.row(i).to_vec();
                let c = flat_cell(&cells[i]);
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
        let c: Vec<f64> = cells.iter().flat_map(flat_cell).collect();
        let mut e = vec![0.0; n];
        let mut g = vec![0.0; n * dof];
        let mut st = vec![0.0; n * 9];
        self.call(n as i64, -1, &p, &c, &mut e, &mut g, &mut st)?;
        for i in 0..n {
            energies[i] = e[i];
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
    if band.session.image_cells().is_none() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let cs = CSolidSurface {
        f,
        user,
        n_atoms: band.n_atoms,
        n_images: band.n_images,
        per_image: band.per_image,
    };
    match band.session.step(&cs) {
        Ok(report) => {
            let out = unsafe { &mut *out };
            stamp_report(out);
            out.status = i32::from(report.status == BandStatus::Converged);
            out.evaluations = i32::try_from(report.surface_rows).unwrap_or(i32::MAX);
            out.max_force = report.max_force;
            out.ci_index = report.ci_index.map_or(-1, |i| i as i64);
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
    let Some(cells) = band.session.image_cells() else {
        return RGSADDLE_INVALID_PARAMETER;
    };
    if out.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let dst = unsafe { slice::from_raw_parts_mut(out, cells.len() * 9) };
    for (i, cell) in cells.iter().enumerate() {
        for (k, v) in cell.0.iter().flatten().enumerate() {
            dst[i * 9 + k] = *v;
        }
    }
    RGSADDLE_OK
}

/// # Safety
/// `band` and `out` must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_band_solid_jacobian(
    band: *const RgsaddleBand,
    out: *mut f64,
) -> i32 {
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
