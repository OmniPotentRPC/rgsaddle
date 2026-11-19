//! C ABI over the stepping sessions. See `include/rgsaddle.h`.
//!
//! The host owns the loop here exactly as it does in Rust: create,
//! step until the report says converged, reset at a surface-epoch
//! boundary, free. No run-to-completion entry exists.
//!
//! [`rgsaddle_band_create`], [`rgsaddle_minmode_create`] and
//! [`rgsaddle_stencil_new`] each open with that loop written in Rust
//! against these exports.

use std::ffi::c_void;
use std::os::raw::c_char;
use std::slice;

use ndarray::{Array1, Array2, ArrayView2};
use rgmin::{FireKind, Method};

use crate::band::{BandConfig, BandSession, BandStatus, BandSurface, Cell, CiConfig, SolidState};
use crate::connect::{ConnectConfig, connect};
use crate::error::SaddleError;
use crate::geom::{TrustSchedule, trust_update};
use crate::minmode::{
    ArtNConfig, FiniteDifference, MinModeConfig, MinModeKind, MinModeSession, MinModeStatus,
    PointSurface, halve_raised_energy,
};
use crate::nichols::{
    HessianUpdate, Index1Config, Index1Session, Index1Status, NicholsMode, bofill_update,
    cap_max_abs, nichols_step, powell_update,
};
use crate::prfo::{PrfoKind, restricted_prfo_displacement};
use crate::projection::ProjectionKind;
use crate::saddle_index::SaddleIndexCheck;
use crate::spring::SpringKind;
use crate::tangent::TangentKind;

/// The RGSADDLE ABI MAJOR.
pub const RGSADDLE_ABI_MAJOR: u32 = 1;
/// The RGSADDLE ABI MINOR.
pub const RGSADDLE_ABI_MINOR: u32 = 46;

/// A band, min-mode or OCI-NEB config of this minor or newer certifies
/// the saddle index at convergence by default.
pub(crate) const SADDLE_INDEX_MINOR: u32 = 40;

/// Band config bit 0. The C step calls the surface once per image
/// carried by the evaluation: every image on the first evaluation
/// after create or set_positions, the interior images afterwards.
pub const RGSADDLE_BAND_PER_IMAGE: u64 = 1;
/// Band config bit 1. The projected force keeps its mean Cartesian
/// part, as eOn does when any atom is fixed. From ABI minor 39 a band
/// with a fixed-atom mask keeps it whether or not the bit is set.
pub const RGSADDLE_BAND_KEEP_TRANSLATION: u64 = 2;
/// Band config bit 2 (ABI minor 17). Climbing is on with the arming
/// rule as given: it arms once the force is under `ci_trigger_force` or
/// under `ci_trigger_factor` times the first force, either of which may
/// be 0. Without the bit, climbing is on only when `ci_trigger_factor`
/// is positive.
pub const RGSADDLE_BAND_CI_AFTER: u64 = 4;
/// Band config bit 3 (ABI minor 21). No min-mode refinement of the
/// climbing image: the plain CI-NEB of minor 20 and earlier.
pub const RGSADDLE_BAND_NO_MIN_MODE: u64 = 8;
/// Band config bit 4 (ABI minor 38). The trust-region stepper
/// (`RGSADDLE_METHOD_RTR`) places its trial images at equal arc length
/// on the not-a-knot cubic spline instead of the polyline. Read only
/// when the config minor is 38 or newer.
pub const RGSADDLE_BAND_SPLINE_REPARAM: u64 = 16;
/// Band config bit 5 (ABI minor 44). Simplified string method: evolve
/// interior images under the full potential force and reparametrize to
/// equal arc length after each step (`RGSADDLE_BAND_SPLINE_REPARAM`
/// selects the not-a-knot spline, else the polyline).
pub const RGSADDLE_BAND_STRING: u64 = 32;
pub(crate) const BAND_STRING_MINOR: u32 = 44;

/// The trust-region reparametrization a band config asks for.
pub(crate) fn rtr_reparametrization(cfg: &RgsaddleBandConfig) -> crate::Reparametrization {
    if cfg.version.minor >= 38 && cfg.flags & RGSADDLE_BAND_SPLINE_REPARAM != 0 {
        crate::Reparametrization::NotAKnot
    } else {
        crate::Reparametrization::Linear
    }
}

/// The simplified string configuration a band config asks for.
pub(crate) fn string_config(cfg: &RgsaddleBandConfig) -> Option<crate::StringConfig> {
    if cfg.version.minor >= BAND_STRING_MINOR && cfg.flags & RGSADDLE_BAND_STRING != 0 {
        Some(crate::StringConfig::default().with_reparametrization(rtr_reparametrization(cfg)))
    } else {
        None
    }
}

/// Request bit 0. Positions and gradients are one image.
pub const RGSADDLE_REQ_ONE_IMAGE: u64 = 1;
/// Request bit 1 (ABI minor 23). The request carries the band's
/// climbing image alone, for the min-mode refinement, and `image` is
/// its band index in the batched mode too. Set only for a band whose
/// config minor is 23 or newer.
pub const RGSADDLE_REQ_CLIMBING_IMAGE: u64 = 2;

/// The RGSADDLE OK.
pub const RGSADDLE_OK: i32 = 0;
/// The RGSADDLE NULL SESSION.
pub const RGSADDLE_NULL_SESSION: i32 = -1;
/// The RGSADDLE NULL BAND.
pub const RGSADDLE_NULL_BAND: i32 = -2;
/// The RGSADDLE NULL SURFACE.
pub const RGSADDLE_NULL_SURFACE: i32 = -3;
/// The RGSADDLE NULL REPORT.
pub const RGSADDLE_NULL_REPORT: i32 = -4;
/// The RGSADDLE SHAPE.
pub const RGSADDLE_SHAPE: i32 = -5;
/// The RGSADDLE SURFACE FAILED.
pub const RGSADDLE_SURFACE_FAILED: i32 = -6;
/// The RGSADDLE NON FINITE.
pub const RGSADDLE_NON_FINITE: i32 = -7;
/// The RGSADDLE SOLVER.
pub const RGSADDLE_SOLVER: i32 = -8;
/// The RGSADDLE ABI MISMATCH.
pub const RGSADDLE_ABI_MISMATCH: i32 = -9;
/// The RGSADDLE ALLOC.
pub const RGSADDLE_ALLOC: i32 = -10;
/// The RGSADDLE INVALID PARAMETER.
pub const RGSADDLE_INVALID_PARAMETER: i32 = -11;
/// The RGSADDLE NO EVALUATION.
pub const RGSADDLE_NO_EVALUATION: i32 = -12;
/// The RGSADDLE STALLED.
pub const RGSADDLE_STALLED: i32 = -13;
/// The constrained-ensemble tail still drifts after equilibration.
pub const RGSADDLE_NOT_EQUILIBRATED: i32 = -14;
/// The constrained-ensemble tail has too few effective samples.
pub const RGSADDLE_INSUFFICIENT_EFFECTIVE_SAMPLES: i32 = -15;

#[repr(C)]
#[derive(Clone, Copy)]
/// The RgsaddleVersion.
pub struct RgsaddleVersion {
    /// The major.
    pub major: u32,
    /// The minor.
    pub minor: u32,
}

#[repr(C)]
/// The RgsaddleSurfaceRequest.
pub struct RgsaddleSurfaceRequest {
    /// ABI version this value was built against.
    pub version: RgsaddleVersion,
    /// Bit flags. The header names each bit.
    pub flags: u64,
    /// Number of images, endpoints included.
    pub n_images: i64,
    /// Number of atoms in one image.
    pub n_atoms: i64,
    /// Positions, `n_images` rows of `3 * n_atoms`, row-major.
    pub positions: *const f64,
    /// One energy per row of `positions`, written by the host.
    pub energies: *mut f64,
    /// Energy gradients, one row per image, in the same order as the positions.
    pub gradients: *mut f64,
    /// -1 for the whole band. Otherwise the image this call fills.
    pub image: i64,
}

/// The RgsaddleSurfaceFn.
pub type RgsaddleSurfaceFn = extern "C" fn(*mut c_void, *mut RgsaddleSurfaceRequest) -> i32;

#[repr(C)]
/// The RgsaddleBandConfig.
pub struct RgsaddleBandConfig {
    /// ABI version this value was built against.
    pub version: RgsaddleVersion,
    /// Bit flags. The header names each bit.
    pub flags: u64,
    /// How the path direction at an interior image is chosen.
    pub tangent: i32,
    /// How the springs between images are chosen.
    pub spring: i32,
    /// How the true force and the spring are combined.
    pub projection: i32,
    /// Which rgmin stepper advances the assembled force.
    pub method: i32,
    /// Spring constant of the uniform and Onsager-Machlup springs.
    pub spring_k: f64,
    /// Weighted springs: `n_images - 1` constants, else NULL.
    pub spring_ks: *const f64,
    /// Climbing arms once the largest force component is under this
    /// factor times the first evaluation's. Climbing is off when it is
    /// not positive, unless [`RGSADDLE_BAND_CI_AFTER`] is set.
    pub ci_trigger_factor: f64,
    /// With [`RGSADDLE_BAND_CI_AFTER`], climbing also arms once the
    /// largest force component is under this value.
    pub ci_trigger_force: f64,
    /// Row-major 3x3 cell for minimum-image differences; NULL for none.
    pub cell: *const f64,
    /// The search stops when the largest absolute force component falls under this value, in the host's force units.
    pub force_tol: f64,
    /// Largest per-atom displacement, in the host's length unit.
    pub max_move: f64,
    /// Number of vector pairs a limited-memory stepper stores.
    pub memory: i64,
}

#[repr(C)]
/// The RgsaddleMinModeConfig.
pub struct RgsaddleMinModeConfig {
    /// ABI version this value was built against.
    pub version: RgsaddleVersion,
    /// Bit flags. The header names each bit.
    pub flags: u64,
    /// Which estimator produces the lowest mode.
    pub kind: i32,
    /// Which rgmin stepper advances the assembled force.
    pub method: i32,
    /// Finite-difference displacement, in the host's length unit.
    pub dr: f64,
    /// The rotation stops when the rotational force falls under this value.
    pub rotation_tol: f64,
    /// Most dimer rotations, or Lanczos actions, in one estimate.
    pub max_rotations: i64,
    /// Largest Krylov subspace used for one Lanczos estimate.
    pub krylov_dim: i64,
    /// The search stops when the largest absolute force component falls under this value, in the host's force units.
    pub force_tol: f64,
    /// Largest per-atom displacement, in the host's length unit.
    pub max_move: f64,
    /// ABI minor 5: dimer rotation angle tolerance in radians. The Rust
    /// default is five degrees; zero disables the test. A C config
    /// stores the caller's value. Read only when the minor is 5 or more.
    pub rotation_angle_tol: f64,
    /// ABI minor 5: 0 forward difference, 1 central.
    pub difference: i32,
    /// The reserved.
    pub reserved: i32,
}

#[repr(C)]
/// ART nouveau controls added in ABI minor 43.
pub struct RgsaddleArtNConfig {
    /// ABI version this value was built against.
    pub version: RgsaddleVersion,
    /// Reserved flags; set to zero.
    pub flags: u64,
    /// Start Lanczos after this many basin pushes.
    pub lanczos_start: i64,
    /// Enter activation below this lowest eigenvalue.
    pub eigenvalue_threshold: f64,
    /// Length of each basin push.
    pub basin_push: f64,
    /// Perpendicular relaxation steps after a basin push.
    pub basin_relaxations: i64,
    /// Perpendicular relaxation steps after an activation push.
    pub activation_relaxations: i64,
    /// Scale multiplying the projected gradient.
    pub relaxation_step: f64,
    /// Most Lanczos Hessian actions at one geometry.
    pub lanczos_actions: i64,
    /// Lower bound on the activation step's curvature magnitude.
    pub curvature_floor: f64,
}

#[repr(C)]
/// The RgsaddleReport.
pub struct RgsaddleReport {
    /// ABI version this value was built against.
    pub version: RgsaddleVersion,
    /// Bit flags. The header names each bit.
    pub flags: u64,
    /// Whether this step is still running or has converged.
    pub status: i32,
    /// Geometries the surface evaluated during the step (band image
    /// rows, or single points for the min-mode and index-1 sessions).
    pub evaluations: i32,
    /// Largest absolute force component, in the host's force units.
    pub max_force: f64,
    /// Index of the climbing image, or none when no image climbs.
    pub ci_index: i64,
    /// Number of steps taken.
    pub iteration: i64,
    /// Curvature along the mode, energy per length squared.
    pub curvature: f64,
    /// Dimer rotations or Lanczos actions in the step's mode estimate;
    /// 0 on a band step.
    pub rotations: i64,
}

/// Host surface behind the C callback. The pointers live only for the
/// duration of one call, which is exactly the request's lifetime.
struct CSurface {
    f: RgsaddleSurfaceFn,
    user: *mut c_void,
    n_atoms: i64,
    /// Length of the band (1 for a min-mode session). Maps the rows of
    /// an interior-only evaluation back to band image indices.
    n_images: i64,
    per_image: bool,
    /// Mark the refinement's requests with `RGSADDLE_REQ_CLIMBING_IMAGE`.
    climbing_request: bool,
    rows: std::sync::atomic::AtomicUsize,
}

// The C host is responsible for its own thread safety; the sessions
// call the surface only from the thread that called step.
unsafe impl Sync for CSurface {}
unsafe impl Send for CSurface {}

/// Write `values` into `out` in order.
///
/// # Safety
/// `out` must point at `n` writable doubles. A longer iterator panics,
/// the same as indexing that slice.
unsafe fn write_doubles(out: *mut f64, n: usize, values: impl Iterator<Item = f64>) {
    let dst = unsafe { slice::from_raw_parts_mut(out, n) };
    for (i, value) in values.enumerate() {
        dst[i] = value;
    }
}

impl CSurface {
    fn evaluation_rows_checked(&self) -> Option<i32> {
        i32::try_from(self.rows.load(std::sync::atomic::Ordering::Relaxed)).ok()
    }

    /// One geometry, with no per-image or climbing request flags.
    fn point(f: RgsaddleSurfaceFn, user: *mut c_void, n_atoms: i64) -> Self {
        Self {
            f,
            user,
            n_atoms,
            n_images: 1,
            per_image: false,
            climbing_request: false,
            rows: Default::default(),
        }
    }

    fn evaluation_rows(&self) -> i32 {
        i32::try_from(self.rows.load(std::sync::atomic::Ordering::Relaxed)).unwrap_or(i32::MAX)
    }

    fn call(
        &self,
        n_images: i64,
        image: i64,
        positions: &[f64],
        energies: &mut [f64],
        gradients: &mut [f64],
    ) -> Result<(), SaddleError> {
        let flags = if image >= 0 {
            RGSADDLE_REQ_ONE_IMAGE
        } else {
            0
        };
        self.request(flags, n_images, image, positions, energies, gradients)
    }

    fn request(
        &self,
        flags: u64,
        n_images: i64,
        image: i64,
        positions: &[f64],
        energies: &mut [f64],
        gradients: &mut [f64],
    ) -> Result<(), SaddleError> {
        let mut req = RgsaddleSurfaceRequest {
            version: RgsaddleVersion {
                major: RGSADDLE_ABI_MAJOR,
                minor: RGSADDLE_ABI_MINOR,
            },
            flags,
            n_images,
            n_atoms: self.n_atoms,
            positions: positions.as_ptr(),
            energies: energies.as_mut_ptr(),
            gradients: gradients.as_mut_ptr(),
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

impl BandSurface for CSurface {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        let n_rows = positions.nrows() as i64;
        let dof = gradients.ncols();
        // Row r is band image r on a whole-band evaluation and image
        // r + 1 on an interior-only one (see BandSurface).
        let first_image: i64 = if n_rows == self.n_images { 0 } else { 1 };
        if self.per_image {
            for i in 0..positions.nrows() {
                let flat: Vec<f64> = positions.row(i).iter().copied().collect();
                let mut e = [0.0];
                let mut g = vec![0.0; dof];
                self.call(self.n_images, first_image + i as i64, &flat, &mut e, &mut g)?;
                energies[i] = e[0];
                for c in 0..dof {
                    gradients[(i, c)] = g[c];
                }
            }
            return Ok(());
        }
        // Batched: n_images in the request counts the rows carried.
        let flat: Vec<f64> = positions.iter().copied().collect();
        let mut e = vec![0.0; energies.len()];
        let mut g = vec![0.0; gradients.len()];
        self.call(n_rows, -1, &flat, &mut e, &mut g)?;
        for (i, v) in e.iter().enumerate() {
            energies[i] = *v;
        }
        let dof = gradients.ncols();
        for i in 0..gradients.nrows() {
            for c in 0..dof {
                gradients[(i, c)] = g[i * dof + c];
            }
        }
        Ok(())
    }

    fn eval_image(
        &self,
        image: usize,
        position: ndarray::ArrayView1<f64>,
    ) -> Result<(f64, Array1<f64>), SaddleError> {
        let flat: Vec<f64> = position.iter().copied().collect();
        let mut e = [0.0];
        let mut g = vec![0.0; flat.len()];
        let image = image as i64;
        match (self.per_image, self.climbing_request) {
            (true, false) => self.call(self.n_images, image, &flat, &mut e, &mut g)?,
            (false, false) => self.call(1, -1, &flat, &mut e, &mut g)?,
            (true, true) => self.request(
                RGSADDLE_REQ_ONE_IMAGE | RGSADDLE_REQ_CLIMBING_IMAGE,
                self.n_images,
                image,
                &flat,
                &mut e,
                &mut g,
            )?,
            (false, true) => {
                self.request(RGSADDLE_REQ_CLIMBING_IMAGE, 1, image, &flat, &mut e, &mut g)?
            }
        }
        Ok((e[0], Array1::from(g)))
    }

    fn eval_indexed(
        &self,
        image: usize,
        position: ndarray::ArrayView1<f64>,
    ) -> Result<(f64, Array1<f64>), SaddleError> {
        let flat: Vec<f64> = position.iter().copied().collect();
        let mut e = [0.0];
        let mut g = vec![0.0; flat.len()];
        let image = image as i64;
        // One interior image, named by its band index. Not the
        // climbing-image request: a dynamic step refreshes whichever
        // image the spring pulled back over the threshold. n_images
        // stays the band length, as RGSADDLE_REQ_ONE_IMAGE requires;
        // the buffers hold the one image.
        self.request(
            RGSADDLE_REQ_ONE_IMAGE,
            self.n_images,
            image,
            &flat,
            &mut e,
            &mut g,
        )?;
        Ok((e[0], Array1::from(g)))
    }
}

impl PointSurface for CSurface {
    fn eval(&self, x: ndarray::ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let flat: Vec<f64> = x.iter().copied().collect();
        let mut e = vec![0.0; 1];
        let mut g = vec![0.0; flat.len()];
        self.call(1, -1, &flat, &mut e, &mut g)?;
        Ok((e[0], Array1::from(g)))
    }
}

/// The RgsaddleBand.
pub struct RgsaddleBand {
    session: BandSession,
    n_images: i64,
    n_atoms: i64,
    per_image: bool,
    climbing_request: bool,
}

/// The RgsaddleMinMode.
pub struct RgsaddleMinMode {
    session: MinModeSession,
    n_atoms: i64,
}

fn method_of(v: i32, memory: i64) -> Method {
    match v {
        1 => Method::lbfgs_with_memory(if memory > 0 { memory as usize } else { 20 }),
        4 => Method::Ode12r,
        _ => Method::fire(FireKind::V2),
    }
}

fn stamp_report(out: &mut RgsaddleReport) {
    out.version = RgsaddleVersion {
        major: RGSADDLE_ABI_MAJOR,
        minor: RGSADDLE_ABI_MINOR,
    };
    out.flags = 0;
}

pub(crate) fn status_of(err: &SaddleError) -> i32 {
    match err {
        SaddleError::Shape(_) => RGSADDLE_SHAPE,
        SaddleError::Surface(_) => RGSADDLE_SURFACE_FAILED,
        SaddleError::NonFinite(_) => RGSADDLE_NON_FINITE,
        SaddleError::Solver(_) => RGSADDLE_SOLVER,
        SaddleError::Invalid(_) => RGSADDLE_INVALID_PARAMETER,
        SaddleError::Stalled(_) => RGSADDLE_STALLED,
        SaddleError::NotEquilibrated { .. } => RGSADDLE_NOT_EQUILIBRATED,
        SaddleError::InsufficientEffectiveSamples { .. } => RGSADDLE_INSUFFICIENT_EFFECTIVE_SAMPLES,
    }
}

#[unsafe(no_mangle)]
/// Rgsaddle abi version.
pub extern "C" fn rgsaddle_abi_version() -> i32 {
    ((RGSADDLE_ABI_MAJOR << 16) | RGSADDLE_ABI_MINOR) as i32
}

/// # Safety
/// `out` must be a valid pointer to an `rgsaddle_version_t` or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_abi_stamp(out: *mut RgsaddleVersion) -> i32 {
    if out.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    unsafe {
        (*out).major = RGSADDLE_ABI_MAJOR;
        (*out).minor = RGSADDLE_ABI_MINOR;
    }
    RGSADDLE_OK
}

#[unsafe(no_mangle)]
/// Rgsaddle status name.
pub extern "C" fn rgsaddle_status_name(status: i32) -> *const c_char {
    let s: &'static str = match status {
        RGSADDLE_OK => "OK\0",
        RGSADDLE_NULL_SESSION => "NULL_SESSION\0",
        RGSADDLE_NULL_BAND => "NULL_BAND\0",
        RGSADDLE_NULL_SURFACE => "NULL_SURFACE\0",
        RGSADDLE_NULL_REPORT => "NULL_REPORT\0",
        RGSADDLE_SHAPE => "SHAPE\0",
        RGSADDLE_SURFACE_FAILED => "SURFACE_FAILED\0",
        RGSADDLE_NON_FINITE => "NON_FINITE\0",
        RGSADDLE_SOLVER => "SOLVER\0",
        RGSADDLE_ABI_MISMATCH => "ABI_MISMATCH\0",
        RGSADDLE_ALLOC => "ALLOC\0",
        RGSADDLE_INVALID_PARAMETER => "INVALID_PARAMETER\0",
        RGSADDLE_NO_EVALUATION => "NO_EVALUATION\0",
        RGSADDLE_STALLED => "STALLED\0",
        RGSADDLE_NOT_EQUILIBRATED => "NOT_EQUILIBRATED\0",
        RGSADDLE_INSUFFICIENT_EFFECTIVE_SAMPLES => "INSUFFICIENT_EFFECTIVE_SAMPLES\0",
        _ => "UNKNOWN\0",
    };
    s.as_ptr() as *const c_char
}

/// Open a band over `n_images` rows of `3 * n_atoms` positions.
/// NULL when the config or the shape is refused.
///
/// # Safety
/// `config` and `positions` must be valid for the declared shape.
///
/// # Examples
///
/// The C loop from Rust: a nudged elastic band on the double well
/// `(x² − 1)² + 2y² + 2z²`, one atom, seven images bowed off the axis.
/// The callback fills every row the request carries.
///
/// ```
/// use std::ffi::c_void;
/// use rgsaddle::capi::*;
///
/// extern "C" fn double_well(_user: *mut c_void, req: *mut RgsaddleSurfaceRequest) -> i32 {
///     let req = unsafe { &mut *req };
///     let rows = if req.flags & RGSADDLE_REQ_ONE_IMAGE != 0 { 1 } else { req.n_images as usize };
///     let dof = 3 * req.n_atoms as usize;
///     let x = unsafe { std::slice::from_raw_parts(req.positions, rows * dof) };
///     let e = unsafe { std::slice::from_raw_parts_mut(req.energies, rows) };
///     let g = unsafe { std::slice::from_raw_parts_mut(req.gradients, rows * dof) };
///     for i in 0..rows {
///         let r = &x[i * dof..(i + 1) * dof];
///         e[i] = (r[0] * r[0] - 1.0).powi(2) + 2.0 * r[1] * r[1] + 2.0 * r[2] * r[2];
///         g[i * dof..(i + 1) * dof]
///             .copy_from_slice(&[4.0 * r[0] * (r[0] * r[0] - 1.0), 4.0 * r[1], 4.0 * r[2]]);
///     }
///     0
/// }
///
/// let config = RgsaddleBandConfig {
///     version: RgsaddleVersion { major: RGSADDLE_ABI_MAJOR, minor: RGSADDLE_ABI_MINOR },
///     flags: 0,
///     tangent: 1,    // RGSADDLE_TANGENT_IMPROVED
///     spring: 0,     // RGSADDLE_SPRING_UNIFORM
///     projection: 1, // RGSADDLE_PROJECTION_NEB
///     method: 0,     // RGSADDLE_METHOD_FIRE
///     spring_k: 5.0,
///     spring_ks: std::ptr::null(),
///     ci_trigger_factor: 0.5,
///     ci_trigger_force: 0.0,
///     cell: std::ptr::null(),
///     force_tol: 1e-3,
///     max_move: 0.1,
///     memory: 0,
/// };
/// let mut positions = Vec::new();
/// for i in 0..7 {
///     let t = i as f64 / 6.0;
///     positions.extend([2.0 * t - 1.0, 0.3 * (std::f64::consts::PI * t).sin(), 0.0]);
/// }
/// let band = unsafe { rgsaddle_band_create(&config, 7, 1, positions.as_ptr()) };
/// assert!(!band.is_null());
/// let mut out = std::mem::MaybeUninit::<RgsaddleReport>::uninit();
/// let report = loop {
///     let user = std::ptr::null_mut();
///     let rc = unsafe { rgsaddle_band_step(band, Some(double_well), user, out.as_mut_ptr()) };
///     assert_eq!(rc, RGSADDLE_OK);
///     let report = unsafe { out.assume_init_ref() };
///     if report.status == 1 || report.iteration >= 2000 {
///         break report;
///     }
/// };
/// assert_eq!(report.status, 1);
/// unsafe { rgsaddle_band_positions(band, positions.as_mut_ptr()) };
/// let ci = report.ci_index as usize;
/// assert!(positions[3 * ci..3 * ci + 3].iter().all(|c| c.abs() < 1e-2));
/// unsafe { rgsaddle_band_free(band) };
/// ```
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_band_create(
    config: *const RgsaddleBandConfig,
    n_images: i64,
    n_atoms: i64,
    positions: *const f64,
) -> *mut RgsaddleBand {
    unsafe { band_from_parts(config, n_images, n_atoms, positions, None) }
}

/// # Safety
/// `cells` holds `n_images * 9` row-major lattice entries. The other
/// pointers follow [`rgsaddle_band_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_band_create_solid(
    config: *const RgsaddleBandConfig,
    n_images: i64,
    n_atoms: i64,
    positions: *const f64,
    cells: *const f64,
    pressure: f64,
    weight: f64,
) -> *mut RgsaddleBand {
    if cells.is_null() || n_images < 3 {
        return std::ptr::null_mut();
    }
    let raw = unsafe { slice::from_raw_parts(cells, (n_images as usize) * 9) };
    let mut parsed = Vec::with_capacity(n_images as usize);
    for image in 0..n_images as usize {
        let c = &raw[image * 9..image * 9 + 9];
        parsed.push(Cell([
            [c[0], c[1], c[2]],
            [c[3], c[4], c[5]],
            [c[6], c[7], c[8]],
        ]));
    }
    unsafe {
        band_from_parts(
            config,
            n_images,
            n_atoms,
            positions,
            Some(
                SolidState::new(parsed)
                    .with_pressure(pressure)
                    .with_weight(weight),
            ),
        )
    }
}

pub(super) struct ParsedBand {
    pub config: BandConfig,
    pub per_image: bool,
    pub method: i32,
}

/// # Safety
/// `spring_ks` and `cell`, when non-null, must point at the lengths
/// the band config documents.
pub(super) unsafe fn parse_band_config(
    cfg: &RgsaddleBandConfig,
    n_images: i64,
    dof: usize,
    solid: Option<SolidState>,
) -> Option<ParsedBand> {
    if cfg.version.major != RGSADDLE_ABI_MAJOR || n_images < 3 || dof == 0 {
        return None;
    }
    let spring = match cfg.spring {
        1 => {
            if cfg.spring_ks.is_null() {
                return None;
            }
            let ks = unsafe { slice::from_raw_parts(cfg.spring_ks, (n_images - 1) as usize) };
            SpringKind::Weighted { ks: ks.to_vec() }
        }
        2 => SpringKind::OnsagerMachlupLive { k: cfg.spring_k },
        _ => SpringKind::Uniform { k: cfg.spring_k },
    };
    let cell = if cfg.cell.is_null() {
        None
    } else {
        let c = unsafe { slice::from_raw_parts(cfg.cell, 9) };
        Some(Cell([
            [c[0], c[1], c[2]],
            [c[3], c[4], c[5]],
            [c[6], c[7], c[8]],
        ]))
    };
    Some(ParsedBand {
        config: BandConfig::default()
            .with_tangent(if cfg.tangent == 0 {
                TangentKind::Simple
            } else {
                TangentKind::Improved
            })
            .with_spring(spring)
            .with_projection(match cfg.projection {
                0 => ProjectionKind::PlainElasticBand,
                2 => ProjectionKind::DoublyNudged,
                _ => ProjectionKind::Neb,
            })
            .with_climbing(
                (cfg.ci_trigger_factor > 0.0 || cfg.flags & RGSADDLE_BAND_CI_AFTER != 0).then_some(
                    CiConfig::default()
                        .with_trigger_factor(cfg.ci_trigger_factor)
                        .with_trigger_force(cfg.ci_trigger_force),
                ),
            )
            .with_cell(cell)
            .with_force_tol(cfg.force_tol)
            .with_max_move(cfg.max_move)
            .with_remove_translation(cfg.flags & RGSADDLE_BAND_KEEP_TRANSLATION == 0)
            .with_method(method_of(cfg.method, cfg.memory))
            .with_solid(solid)
            .with_quickmin(cfg.method == 3)
            .with_quickmin_options(rgmin::QuickMinOptions::rgmin())
            .with_time_step(1.0)
            .with_autoneb(None)
            .with_min_mode(
                (cfg.flags & RGSADDLE_BAND_NO_MIN_MODE == 0)
                    .then(crate::band::MinModeRefine::default),
            )
            // Installed after creation through rgsaddle_band_set_exp_precon.
            .with_exp_precon(None)
            .with_zoom(None)
            .with_saddle_check(
                (cfg.version.minor >= SADDLE_INDEX_MINOR).then(SaddleIndexCheck::default),
            )
            .with_string(string_config(cfg)),
        per_image: cfg.flags & RGSADDLE_BAND_PER_IMAGE != 0,
        method: cfg.method,
    })
}

unsafe fn band_from_parts(
    config: *const RgsaddleBandConfig,
    n_images: i64,
    n_atoms: i64,
    positions: *const f64,
    solid: Option<SolidState>,
) -> *mut RgsaddleBand {
    if config.is_null() || positions.is_null() || n_images < 3 || n_atoms < 1 {
        return std::ptr::null_mut();
    }
    let cfg = unsafe { &*config };
    if cfg.version.major != RGSADDLE_ABI_MAJOR {
        return std::ptr::null_mut();
    }
    let dof = (3 * n_atoms) as usize;
    let total = dof * n_images as usize;
    let flat = unsafe { slice::from_raw_parts(positions, total) };
    let initial = match Array2::from_shape_vec((n_images as usize, dof), flat.to_vec()) {
        Ok(a) => a,
        Err(_) => return std::ptr::null_mut(),
    };
    let Some(parsed) = (unsafe { parse_band_config(cfg, n_images, dof, solid) }) else {
        return std::ptr::null_mut();
    };
    let per_image = parsed.per_image;
    let climbing_request = cfg.version.minor >= 22;
    let method = parsed.method;
    match BandSession::new(parsed.config, initial) {
        Ok(mut session) => {
            if method == 2
                && session
                    .set_rtr(Some(
                        crate::rtr::RtrConfig::default()
                            .with_radius_max(cfg.max_move * ((n_images - 2) as f64).sqrt())
                            .with_reparametrization(rtr_reparametrization(cfg)),
                    ))
                    .is_err()
            {
                return std::ptr::null_mut();
            }
            Box::into_raw(Box::new(RgsaddleBand {
                session,
                n_images,
                n_atoms,
                per_image,
                climbing_request,
            }))
        }
        Err(_) => std::ptr::null_mut(),
    }
}

struct GivenBand {
    energies: Array1<f64>,
    gradients: Array2<f64>,
}

impl BandSurface for GivenBand {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        if positions.nrows() != self.energies.len() || gradients.dim() != self.gradients.dim() {
            return Err(SaddleError::Shape(
                "projection storage does not match the band".into(),
            ));
        }
        energies.assign(&self.energies);
        gradients.assign(&self.gradients);
        Ok(())
    }
}

/// A C fixed-atom mask, one byte per atom, as booleans; `None` when
/// a byte is neither 0 nor 1.
///
/// # Safety
/// `mask` addresses `n_atoms` bytes and `n_atoms` is positive.
pub(super) unsafe fn fixed_mask(mask: *const u8, n_atoms: i64) -> Option<Vec<bool>> {
    let raw = unsafe { slice::from_raw_parts(mask, n_atoms as usize) };
    raw.iter()
        .map(|&v| match v {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        })
        .collect()
}

/// # Safety
/// `config`, the input pointers and `forces` must address the lengths
/// the header documents. `ci_index` may be null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_band_project(
    config: *const RgsaddleBandConfig,
    n_images: i64,
    n_atoms: i64,
    positions: *const f64,
    energies: *const f64,
    gradients: *const f64,
    forces: *mut f64,
    ci_index: *mut i64,
) -> i32 {
    unsafe {
        project_band(
            config, n_images, n_atoms, positions, energies, gradients, None, forces, ci_index,
        )
    }
}

/// [`rgsaddle_band_project`] with a per-atom fixed mask (ABI minor
/// 39): `fixed` is null or `n_atoms` bytes, 1 for a fixed atom.
///
/// # Safety
/// As [`rgsaddle_band_project`]; `fixed` is null or addresses
/// `n_atoms` bytes.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn rgsaddle_band_project_fixed(
    config: *const RgsaddleBandConfig,
    n_images: i64,
    n_atoms: i64,
    positions: *const f64,
    energies: *const f64,
    gradients: *const f64,
    fixed: *const u8,
    forces: *mut f64,
    ci_index: *mut i64,
) -> i32 {
    let mask = (!fixed.is_null()).then_some(fixed);
    unsafe {
        project_band(
            config, n_images, n_atoms, positions, energies, gradients, mask, forces, ci_index,
        )
    }
}

/// # Safety
/// As [`rgsaddle_band_project_fixed`].
#[allow(clippy::too_many_arguments)]
unsafe fn project_band(
    config: *const RgsaddleBandConfig,
    n_images: i64,
    n_atoms: i64,
    positions: *const f64,
    energies: *const f64,
    gradients: *const f64,
    fixed: Option<*const u8>,
    forces: *mut f64,
    ci_index: *mut i64,
) -> i32 {
    if config.is_null()
        || positions.is_null()
        || energies.is_null()
        || gradients.is_null()
        || forces.is_null()
        || n_images < 3
        || n_atoms < 1
    {
        return RGSADDLE_SHAPE;
    }
    let cfg = unsafe { &*config };
    let dof = (3 * n_atoms) as usize;
    let rows = n_images as usize;
    let total = dof * rows;
    let Some(mut parsed) = (unsafe { parse_band_config(cfg, n_images, dof, None) }) else {
        return RGSADDLE_INVALID_PARAMETER;
    };
    if let Some(fixed) = fixed {
        let Some(mask) = (unsafe { fixed_mask(fixed, n_atoms) }) else {
            return RGSADDLE_INVALID_PARAMETER;
        };
        parsed.config.fixed_atoms = mask;
    }
    let pos = match Array2::from_shape_vec(
        (rows, dof),
        unsafe { slice::from_raw_parts(positions, total) }.to_vec(),
    ) {
        Ok(a) => a,
        Err(_) => return RGSADDLE_SHAPE,
    };
    let energy = Array1::from(unsafe { slice::from_raw_parts(energies, rows) }.to_vec());
    let gradient = match Array2::from_shape_vec(
        (rows, dof),
        unsafe { slice::from_raw_parts(gradients, total) }.to_vec(),
    ) {
        Ok(a) => a,
        Err(_) => return RGSADDLE_SHAPE,
    };
    let ci = if parsed.config.climbing.is_some() {
        match crate::climbing_image(&parsed.config, pos.view(), energy.view(), gradient.view()) {
            Ok(index) => index,
            Err(err) => return status_of(&err),
        }
    } else {
        None
    };
    let given = GivenBand {
        energies: energy,
        gradients: gradient,
    };
    match crate::band_forces(&parsed.config, &given, pos.view(), ci) {
        Ok(projected) => {
            let out = unsafe { slice::from_raw_parts_mut(forces, total) };
            out.fill(0.0);
            for i in 1..rows - 1 {
                let start = (i - 1) * dof;
                for c in 0..dof {
                    out[i * dof + c] = projected.force[start + c];
                }
            }
            if !ci_index.is_null() {
                unsafe {
                    *ci_index = ci.map(|i| i as i64).unwrap_or(-1);
                }
            }
            RGSADDLE_OK
        }
        Err(err) => status_of(&err),
    }
}

/// ASE 3.29 DyNEB mask. See [`rgsaddle_dyneb_mask`] in the C header.
///
/// # Safety
/// `positions`, `energies` and `forces` address `n_images * 3 * n_atoms`
/// doubles. `frozen`, when non-null, addresses `n_images` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_dyneb_mask(
    n_images: i64,
    n_atoms: i64,
    positions: *const f64,
    energies: *const f64,
    forces: *mut f64,
    fmax: f64,
    scale_fmax: f64,
    frozen: *mut u8,
) -> i32 {
    if positions.is_null() || energies.is_null() || forces.is_null() || n_images < 3 || n_atoms < 1
    {
        return RGSADDLE_SHAPE;
    }
    let rows = n_images as usize;
    let dof = (3 * n_atoms) as usize;
    let total = rows * dof;
    let pos = unsafe { slice::from_raw_parts(positions, total) };
    let energy = unsafe { slice::from_raw_parts(energies, rows) };
    let force = unsafe { slice::from_raw_parts_mut(forces, total) };
    if !pos.iter().all(|v| v.is_finite())
        || !energy.iter().all(|v| v.is_finite())
        || !force.iter().all(|v| v.is_finite())
    {
        return RGSADDLE_NON_FINITE;
    }
    let Ok(interior_pos) = Array2::from_shape_vec((rows - 2, dof), pos[dof..total - dof].to_vec())
    else {
        return RGSADDLE_SHAPE;
    };
    let Ok(mut force_rows) =
        Array2::from_shape_vec((rows - 2, dof), force[dof..total - dof].to_vec())
    else {
        return RGSADDLE_SHAPE;
    };
    let interior_e = Array1::from(energy[1..rows - 1].to_vec());
    let spec = crate::DynamicNeb::default()
        .with_fmax(fmax)
        .with_scale_fmax(scale_fmax);
    let mask = match crate::band::dynamic_mask(
        interior_pos.view(),
        interior_e.view(),
        &mut force_rows,
        spec,
    ) {
        Ok(mask) => mask,
        Err(err) => return status_of(&err),
    };
    for (i, row) in force_rows.outer_iter().enumerate() {
        let start = (i + 1) * dof;
        for (c, value) in row.iter().enumerate() {
            force[start + c] = *value;
        }
    }
    if !frozen.is_null() {
        let flags = unsafe { slice::from_raw_parts_mut(frozen, rows) };
        flags[0] = 1;
        flags[rows - 1] = 1;
        for (i, image) in mask.iter().enumerate() {
            flags[i + 1] = u8::from(image.frozen);
        }
    }
    RGSADDLE_OK
}

/// # Safety
/// `band` and `out` must be valid; `surface` is called with `user`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_band_step(
    band: *mut RgsaddleBand,
    surface: Option<RgsaddleSurfaceFn>,
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
    let cs = CSurface {
        f,
        user,
        n_atoms: band.n_atoms,
        n_images: band.n_images,
        per_image: band.per_image,
        climbing_request: band.climbing_request,
        rows: Default::default(),
    };
    match band.session.step(&cs) {
        Ok(report) => {
            let out = unsafe { &mut *out };
            stamp_report(out);
            out.status = if report.status == BandStatus::Converged {
                1
            } else {
                0
            };
            out.evaluations = i32::try_from(report.surface_rows).unwrap_or(i32::MAX);
            out.max_force = report.max_force;
            out.ci_index = report.ci_index.map(|i| i as i64).unwrap_or(-1);
            out.iteration = report.iteration as i64;
            out.curvature = report.curvature.unwrap_or(0.0);
            out.rotations = 0;
            RGSADDLE_OK
        }
        Err(e) => status_of(&e),
    }
}

/// xtsci name of [`rgsaddle_band_step`].
///
/// # Safety
/// Same contract as [`rgsaddle_band_step`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn xts_band_step(
    band: *mut RgsaddleBand,
    surface: Option<RgsaddleSurfaceFn>,
    user: *mut c_void,
    out: *mut RgsaddleReport,
) -> i32 {
    unsafe { rgsaddle_band_step(band, surface, user, out) }
}

/// # Safety
/// `out` must hold `n_images * 3 * n_atoms` doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_band_positions(band: *const RgsaddleBand, out: *mut f64) -> i32 {
    if band.is_null() {
        return RGSADDLE_NULL_BAND;
    }
    if out.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let band = unsafe { &*band };
    let pos = band.session.positions();
    let total = (band.n_images * 3 * band.n_atoms) as usize;
    unsafe { write_doubles(out, total, pos.iter().copied()) };
    RGSADDLE_OK
}

/// # Safety
/// `positions` must hold `n_images * 3 * n_atoms` doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_band_set_positions(
    band: *mut RgsaddleBand,
    positions: *const f64,
) -> i32 {
    if band.is_null() {
        return RGSADDLE_NULL_BAND;
    }
    if positions.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let band = unsafe { &mut *band };
    let dof = (3 * band.n_atoms) as usize;
    let total = dof * band.n_images as usize;
    let flat = unsafe { slice::from_raw_parts(positions, total) };
    let arr = match Array2::from_shape_vec((band.n_images as usize, dof), flat.to_vec()) {
        Ok(a) => a,
        Err(_) => return RGSADDLE_SHAPE,
    };
    match band.session.set_positions(arr) {
        Ok(()) => RGSADDLE_OK,
        Err(e) => status_of(&e),
    }
}

/// # Safety
/// `band` must be a live session pointer or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_band_reset(band: *mut RgsaddleBand) -> i32 {
    if band.is_null() {
        return RGSADDLE_NULL_BAND;
    }
    unsafe { (*band).session.reset() };
    RGSADDLE_OK
}

/// # Safety
/// `band` must be a live session pointer or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_band_restart(band: *mut RgsaddleBand) -> i32 {
    if band.is_null() {
        return RGSADDLE_NULL_BAND;
    }
    unsafe { (*band).session.restart() };
    RGSADDLE_OK
}

/// # Safety
/// Each non-NULL output must hold its documented length:
/// `energies` n_images, `gradients` n_images * 3 * n_atoms,
/// `projected` (n_images - 2) * 3 * n_atoms.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_band_evaluation(
    band: *const RgsaddleBand,
    energies: *mut f64,
    gradients: *mut f64,
    projected: *mut f64,
) -> i32 {
    if band.is_null() {
        return RGSADDLE_NULL_BAND;
    }
    let band = unsafe { &*band };
    let Some(eval) = band.session.evaluation() else {
        return RGSADDLE_NO_EVALUATION;
    };
    let copy = |dst: *mut f64, src: &mut dyn Iterator<Item = f64>, len: usize| {
        if !dst.is_null() {
            let out = unsafe { slice::from_raw_parts_mut(dst, len) };
            for (o, v) in out.iter_mut().zip(src) {
                *o = v;
            }
        }
    };
    copy(
        energies,
        &mut eval.energies.iter().copied(),
        eval.energies.len(),
    );
    copy(
        gradients,
        &mut eval.gradients.iter().copied(),
        eval.gradients.len(),
    );
    copy(
        projected,
        &mut eval.projected.iter().copied(),
        eval.projected.len(),
    );
    RGSADDLE_OK
}

/// # Safety
/// `band` must come from [`rgsaddle_band_create`] and be freed once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_band_free(band: *mut RgsaddleBand) {
    if !band.is_null() {
        drop(unsafe { Box::from_raw(band) });
    }
}

/// Open a minimum-mode search at `position` with the mode guess
/// `mode`. NULL when the config or the shape is refused.
///
/// # Safety
/// `config`, `position`, and `mode` must be valid for `3 * n_atoms`.
///
/// # Examples
///
/// A dimer search from C, written in Rust: the double well
/// `(x² − 1)² + 2y² + 2z²`, one atom, started off the saddle at the
/// origin. The callback gets one geometry per request.
///
/// ```
/// use std::ffi::c_void;
/// use rgsaddle::capi::*;
///
/// extern "C" fn double_well(_user: *mut c_void, req: *mut RgsaddleSurfaceRequest) -> i32 {
///     let req = unsafe { &mut *req };
///     let r = unsafe { std::slice::from_raw_parts(req.positions, 3) };
///     unsafe {
///         *req.energies = (r[0] * r[0] - 1.0).powi(2) + 2.0 * r[1] * r[1] + 2.0 * r[2] * r[2];
///         std::slice::from_raw_parts_mut(req.gradients, 3)
///             .copy_from_slice(&[4.0 * r[0] * (r[0] * r[0] - 1.0), 4.0 * r[1], 4.0 * r[2]]);
///     }
///     0
/// }
///
/// let config = RgsaddleMinModeConfig {
///     version: RgsaddleVersion { major: RGSADDLE_ABI_MAJOR, minor: RGSADDLE_ABI_MINOR },
///     flags: 0,
///     kind: 0,   // RGSADDLE_MINMODE_DIMER
///     method: 0, // RGSADDLE_METHOD_FIRE
///     dr: 1e-3,
///     rotation_tol: 1e-4,
///     max_rotations: 20,
///     krylov_dim: 12,
///     force_tol: 1e-4,
///     max_move: 0.1,
///     rotation_angle_tol: 5f64.to_radians(),
///     difference: 0, // forward
///     reserved: 0,
/// };
/// let (start, guess) = ([0.3, 0.1, 0.0], [1.0, 0.5, 0.0]);
/// let session = unsafe { rgsaddle_minmode_create(&config, 1, start.as_ptr(), guess.as_ptr()) };
/// assert!(!session.is_null());
/// let mut out = std::mem::MaybeUninit::<RgsaddleReport>::uninit();
/// let report = loop {
///     let user = std::ptr::null_mut();
///     let rc = unsafe { rgsaddle_minmode_step(session, Some(double_well), user, out.as_mut_ptr()) };
///     assert_eq!(rc, RGSADDLE_OK);
///     let report = unsafe { out.assume_init_ref() };
///     if report.status == 1 || report.iteration >= 500 {
///         break report;
///     }
/// };
/// assert_eq!(report.status, 1);
/// assert!(report.curvature < 0.0);
/// let mut saddle = [0.0; 3];
/// unsafe { rgsaddle_minmode_position(session, saddle.as_mut_ptr()) };
/// assert!(saddle.iter().all(|c| c.abs() < 1e-3));
/// unsafe { rgsaddle_minmode_free(session) };
/// ```
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_minmode_create(
    config: *const RgsaddleMinModeConfig,
    n_atoms: i64,
    position: *const f64,
    mode: *const f64,
) -> *mut RgsaddleMinMode {
    if config.is_null() || position.is_null() || mode.is_null() || n_atoms < 1 {
        return std::ptr::null_mut();
    }
    let cfg = unsafe { &*config };
    if cfg.version.major != RGSADDLE_ABI_MAJOR {
        return std::ptr::null_mut();
    }
    let dof = (3 * n_atoms) as usize;
    let x = Array1::from(unsafe { slice::from_raw_parts(position, dof) }.to_vec());
    let m = Array1::from(unsafe { slice::from_raw_parts(mode, dof) }.to_vec());
    if cfg.method == 3 || cfg.method == 4 {
        return std::ptr::null_mut();
    }
    let mm_config = MinModeConfig::default()
        .with_kind(if cfg.kind == 1 {
            MinModeKind::Lanczos
        } else {
            MinModeKind::Dimer
        })
        .with_dr(cfg.dr)
        .with_rotation_tol(cfg.rotation_tol)
        .with_rotation_angle_tol(if cfg.version.minor >= 5 {
            cfg.rotation_angle_tol
        } else {
            0.0
        })
        .with_difference(if cfg.version.minor >= 5 && cfg.difference == 1 {
            FiniteDifference::Central
        } else {
            FiniteDifference::Forward
        })
        .with_max_rotations(cfg.max_rotations as usize)
        .with_krylov_dim(cfg.krylov_dim as usize)
        .with_force_tol(cfg.force_tol)
        .with_max_move(cfg.max_move)
        .with_method(method_of(cfg.method, 0))
        .with_seed_curvature(false)
        .with_saddle_check(
            (cfg.version.minor >= SADDLE_INDEX_MINOR).then(SaddleIndexCheck::default),
        );
    match MinModeSession::new(mm_config, x, m) {
        Ok(session) => Box::into_raw(Box::new(RgsaddleMinMode { session, n_atoms })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// # Safety
/// `session` and `out` must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_minmode_step(
    session: *mut RgsaddleMinMode,
    surface: Option<RgsaddleSurfaceFn>,
    user: *mut c_void,
    out: *mut RgsaddleReport,
) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    if out.is_null() {
        return RGSADDLE_NULL_REPORT;
    }
    let Some(f) = surface else {
        return RGSADDLE_NULL_SURFACE;
    };
    let session = unsafe { &mut *session };
    let cs = CSurface::point(f, user, session.n_atoms);
    match session.session.step(&cs) {
        Ok(report) => {
            let out = unsafe { &mut *out };
            stamp_report(out);
            out.status = if report.status == MinModeStatus::Converged {
                1
            } else {
                0
            };
            out.evaluations = i32::try_from(report.evaluations).unwrap_or(i32::MAX);
            out.max_force = report.max_force;
            out.ci_index = -1;
            out.iteration = report.iteration as i64;
            out.curvature = report.curvature;
            out.rotations = report.rotations as i64;
            RGSADDLE_OK
        }
        Err(e) => status_of(&e),
    }
}

/// # Safety
/// `session` and `out` must be valid; `surface` is called with `user`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_minmode_estimate(
    session: *mut RgsaddleMinMode,
    surface: Option<RgsaddleSurfaceFn>,
    user: *mut c_void,
    out: *mut RgsaddleReport,
) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    if out.is_null() {
        return RGSADDLE_NULL_REPORT;
    }
    let Some(f) = surface else {
        return RGSADDLE_NULL_SURFACE;
    };
    let session = unsafe { &mut *session };
    let cs = CSurface::point(f, user, session.n_atoms);
    match session.session.push_estimate(&cs) {
        Ok(est) => {
            let out = unsafe { &mut *out };
            stamp_report(out);
            out.status = 0;
            out.evaluations = i32::try_from(est.evaluations).unwrap_or(i32::MAX);
            out.max_force = f64::NAN;
            out.ci_index = -1;
            out.iteration = 0;
            out.curvature = est.curvature;
            out.rotations = est.rotations as i64;
            RGSADDLE_OK
        }
        Err(e) => status_of(&e),
    }
}

/// xtsci name of [`rgsaddle_minmode_estimate`].
///
/// # Safety
/// Same contract as [`rgsaddle_minmode_estimate`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn xts_minmode_estimate(
    session: *mut RgsaddleMinMode,
    surface: Option<RgsaddleSurfaceFn>,
    user: *mut c_void,
    out: *mut RgsaddleReport,
) -> i32 {
    unsafe { rgsaddle_minmode_estimate(session, surface, user, out) }
}

/// # Safety
/// `position` must hold `3 * n_atoms` doubles; `gradient` is NULL or
/// holds `3 * n_atoms` doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_minmode_set_position(
    session: *mut RgsaddleMinMode,
    position: *const f64,
    gradient: *const f64,
) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    if position.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let session = unsafe { &mut *session };
    let dof = (3 * session.n_atoms) as usize;
    let x = Array1::from(unsafe { slice::from_raw_parts(position, dof) }.to_vec());
    let g = (!gradient.is_null())
        .then(|| Array1::from(unsafe { slice::from_raw_parts(gradient, dof) }.to_vec()));
    match session.session.set_position(x, g) {
        Ok(()) => RGSADDLE_OK,
        Err(e) => status_of(&e),
    }
}

/// # Safety
/// `mode` must hold `3 * n_atoms` doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_minmode_set_mode(
    session: *mut RgsaddleMinMode,
    mode: *const f64,
) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    if mode.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let session = unsafe { &mut *session };
    let dof = (3 * session.n_atoms) as usize;
    let m = Array1::from(unsafe { slice::from_raw_parts(mode, dof) }.to_vec());
    match session.session.set_mode(m) {
        Ok(()) => RGSADDLE_OK,
        Err(e) => status_of(&e),
    }
}

/// # Safety
/// `out` must hold `3 * n_atoms` doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_minmode_position(
    session: *const RgsaddleMinMode,
    out: *mut f64,
) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    if out.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let session = unsafe { &*session };
    let dof = (3 * session.n_atoms) as usize;
    unsafe { write_doubles(out, dof, session.session.position().iter().copied()) };
    RGSADDLE_OK
}

/// # Safety
/// `out` must hold `3 * n_atoms` doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_minmode_mode(
    session: *const RgsaddleMinMode,
    out: *mut f64,
) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    if out.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let session = unsafe { &*session };
    let dof = (3 * session.n_atoms) as usize;
    unsafe { write_doubles(out, dof, session.session.mode().iter().copied()) };
    RGSADDLE_OK
}

/// # Safety
/// `session` must be a live pointer or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_minmode_reset(session: *mut RgsaddleMinMode) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    unsafe { (*session).session.reset() };
    RGSADDLE_OK
}

/// # Safety
/// `session` must come from [`rgsaddle_minmode_create`], freed once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_minmode_free(session: *mut RgsaddleMinMode) {
    if !session.is_null() {
        drop(unsafe { Box::from_raw(session) });
    }
}

#[repr(C)]
/// The RgsaddleIndex1Config.
pub struct RgsaddleIndex1Config {
    /// ABI version this value was built against.
    pub version: RgsaddleVersion,
    /// Bit flags. The header names each bit.
    pub flags: u64,
    /// The update.
    pub update: i32,
    /// Estimated lowest-curvature direction. A stored mode is normalized.
    pub mode: i32,
    /// The trust radius.
    pub trust_radius: f64,
    /// The search stops when the largest absolute force component falls under this value, in the host's force units.
    pub force_tol: f64,
    /// The fd dr.
    pub fd_dr: f64,
}

/// The RgsaddleIndex1.
pub struct RgsaddleIndex1 {
    session: Index1Session,
    n_atoms: i64,
}

fn nichols_mode(mode: i32) -> Option<NicholsMode> {
    match mode {
        0 => Some(NicholsMode::Minimize),
        1 => Some(NicholsMode::Index1),
        _ => None,
    }
}

/// # Safety
/// `gradient`, `evals`, and `evecs` are readable. `evecs` holds
/// `n * nmode` doubles, row-major, eigenvector `k` in column `k`.
/// `masses` is `n` doubles or NULL for unit mass. `displacement`
/// holds `n` doubles.
#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_nichols_step(
    n: i64,
    nmode: i64,
    gradient: *const f64,
    evals: *const f64,
    evecs: *const f64,
    masses: *const f64,
    trust_radius: f64,
    mode: i32,
    displacement: *mut f64,
) -> i32 {
    if n < 1
        || nmode < 1
        || gradient.is_null()
        || evals.is_null()
        || evecs.is_null()
        || displacement.is_null()
    {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let Some(mode) = nichols_mode(mode) else {
        return RGSADDLE_INVALID_PARAMETER;
    };
    let n = n as usize;
    let nmode = nmode as usize;
    if n.checked_mul(nmode).is_none() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let g = Array1::from(unsafe { slice::from_raw_parts(gradient, n) }.to_vec());
    let ev = Array1::from(unsafe { slice::from_raw_parts(evals, nmode) }.to_vec());
    let vectors = unsafe { slice::from_raw_parts(evecs, n * nmode) }.to_vec();
    let vectors = match Array2::from_shape_vec((n, nmode), vectors) {
        Ok(m) => m,
        Err(_) => return RGSADDLE_SHAPE,
    };
    let mass = if masses.is_null() {
        None
    } else {
        Some(Array1::from(
            unsafe { slice::from_raw_parts(masses, n) }.to_vec(),
        ))
    };
    match nichols_step(
        g.view(),
        ev.view(),
        vectors.view(),
        mass.as_ref().map(|m| m.view()),
        trust_radius,
        mode,
    ) {
        Ok(dx) => {
            unsafe { write_doubles(displacement, n, dx.iter().copied()) };
            RGSADDLE_OK
        }
        Err(e) => status_of(&e),
    }
}

/// # Safety
/// `gradient` holds `n` doubles. `hessian` holds `n * n` row-major
/// doubles, the Cartesian energy Hessian. `masses` holds `n` doubles
/// or is NULL for unit mass. `displacement` holds `n` doubles.
#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_prfo_step(
    n: i64,
    gradient: *const f64,
    hessian: *const f64,
    masses: *const f64,
    trust_radius: f64,
    mode: i32,
    displacement: *mut f64,
) -> i32 {
    if n < 1 || gradient.is_null() || hessian.is_null() || displacement.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let kind = match mode {
        0 => PrfoKind::Minimize,
        1 => PrfoKind::Index1,
        _ => return RGSADDLE_INVALID_PARAMETER,
    };
    let n = n as usize;
    let Some(len) = n.checked_mul(n) else {
        return RGSADDLE_INVALID_PARAMETER;
    };
    let g = Array1::from(unsafe { slice::from_raw_parts(gradient, n) }.to_vec());
    let raw = unsafe { slice::from_raw_parts(hessian, len) }.to_vec();
    let h = match Array2::from_shape_vec((n, n), raw) {
        Ok(m) => m,
        Err(_) => return RGSADDLE_SHAPE,
    };
    let mass = if masses.is_null() {
        None
    } else {
        Some(Array1::from(
            unsafe { slice::from_raw_parts(masses, n) }.to_vec(),
        ))
    };
    match restricted_prfo_displacement(
        h.view(),
        g.view(),
        mass.as_ref().map(|m| m.view()),
        trust_radius,
        kind,
    ) {
        Ok(dx) => {
            unsafe { write_doubles(displacement, n, dx.iter().copied()) };
            RGSADDLE_OK
        }
        Err(e) => status_of(&e),
    }
}

/// # Safety
/// `step` and `dgradient` hold `n` doubles. `hessian` holds `n * n`
/// row-major doubles and is updated in place.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_hessian_powell(
    n: i64,
    step: *const f64,
    dgradient: *const f64,
    hessian: *mut f64,
) -> i32 {
    hessian_update(n, step, dgradient, hessian, true)
}

/// # Safety
/// Same layout as [`rgsaddle_hessian_powell`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_hessian_bofill(
    n: i64,
    step: *const f64,
    dgradient: *const f64,
    hessian: *mut f64,
) -> i32 {
    hessian_update(n, step, dgradient, hessian, false)
}

fn hessian_update(
    n: i64,
    step: *const f64,
    dgradient: *const f64,
    hessian: *mut f64,
    powell: bool,
) -> i32 {
    if n < 1 || step.is_null() || dgradient.is_null() || hessian.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let n = n as usize;
    let s = Array1::from(unsafe { slice::from_raw_parts(step, n) }.to_vec());
    let y = Array1::from(unsafe { slice::from_raw_parts(dgradient, n) }.to_vec());
    let raw = unsafe { slice::from_raw_parts(hessian, n * n) }.to_vec();
    let mut h = match Array2::from_shape_vec((n, n), raw) {
        Ok(m) => m,
        Err(_) => return RGSADDLE_SHAPE,
    };
    let result = if powell {
        powell_update(&mut h, s.view(), y.view())
    } else {
        bofill_update(&mut h, s.view(), y.view())
    };
    match result {
        Ok(()) => {
            unsafe { write_doubles(hessian, n * n, h.iter().copied()) };
            RGSADDLE_OK
        }
        Err(e) => status_of(&e),
    }
}

/// # Safety
/// `raised` and `new_step` are writable. Neither pointer is read as a surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_halve_raised_energy(
    previous: f64,
    trial: f64,
    step: f64,
    raised: *mut i32,
    new_step: *mut f64,
) -> i32 {
    if raised.is_null() || new_step.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let decision = halve_raised_energy(previous, trial, step);
    unsafe {
        *raised = i32::from(decision.raised);
        *new_step = decision.step;
    }
    RGSADDLE_OK
}

/// # Safety
/// `radius_out` is writable. The call does not evaluate a surface.
#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_trust_update(
    radius: f64,
    predicted: f64,
    actual: f64,
    step_norm: f64,
    sigma_inc: f64,
    sigma_dec: f64,
    rho_inc: f64,
    rho_dec: f64,
    delta_min: f64,
    radius_out: *mut f64,
) -> i32 {
    if radius_out.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let schedule = TrustSchedule::default()
        .with_sigma_inc(sigma_inc)
        .with_sigma_dec(sigma_dec)
        .with_rho_inc(rho_inc)
        .with_rho_dec(rho_dec)
        .with_delta_min(delta_min);
    unsafe {
        *radius_out = trust_update(radius, predicted, actual, step_norm, &schedule);
    }
    RGSADDLE_OK
}

/// # Safety
/// `positions` holds `n_images * 3 * n_atoms` doubles and is rewritten
/// in place. `cell`, when not null, holds nine doubles. The call does
/// not evaluate a surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_reparametrize(
    n_images: i64,
    n_atoms: i64,
    positions: *mut f64,
    anchor: i64,
    cell: *const f64,
    scheme: i32,
) -> i32 {
    if positions.is_null() || n_images < 2 || n_atoms < 1 {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let Some(scheme) = crate::Reparametrization::from_abi(scheme) else {
        return RGSADDLE_INVALID_PARAMETER;
    };
    let (Ok(rows), Ok(atoms)) = (usize::try_from(n_images), usize::try_from(n_atoms)) else {
        return RGSADDLE_INVALID_PARAMETER;
    };
    let Some(total) = atoms.checked_mul(3).and_then(|dof| dof.checked_mul(rows)) else {
        return RGSADDLE_INVALID_PARAMETER;
    };
    let flat = unsafe { slice::from_raw_parts_mut(positions, total) };
    if !flat.iter().all(|v| v.is_finite()) {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let cell = if cell.is_null() {
        None
    } else {
        let c = unsafe { slice::from_raw_parts(cell, 9) };
        if !c.iter().all(|v| v.is_finite()) {
            return RGSADDLE_INVALID_PARAMETER;
        }
        Some(Cell([
            [c[0], c[1], c[2]],
            [c[3], c[4], c[5]],
            [c[6], c[7], c[8]],
        ]))
    };
    let anchor = usize::try_from(anchor).ok();
    let Ok(mut band) = Array2::from_shape_vec((rows, 3 * atoms), flat.to_vec()) else {
        return RGSADDLE_INVALID_PARAMETER;
    };
    crate::reparametrize(&mut band, anchor, cell.as_ref(), scheme);
    for (slot, value) in flat.iter_mut().zip(band.iter()) {
        *slot = *value;
    }
    RGSADDLE_OK
}

/// # Safety
/// `step` holds `n` doubles and is scaled in place.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_cap_max_abs(n: i64, step: *mut f64, trust_radius: f64) -> i32 {
    if n < 1 || step.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let n = n as usize;
    let mut dx = Array1::from(unsafe { slice::from_raw_parts(step, n) }.to_vec());
    match cap_max_abs(&mut dx, trust_radius) {
        Ok(()) => {
            unsafe { write_doubles(step, n, dx.iter().copied()) };
            RGSADDLE_OK
        }
        Err(e) => status_of(&e),
    }
}

/// # Safety
/// `config` and `position` are readable. `position` holds
/// `3 * n_atoms` doubles. `hessian` is `dof * dof` row-major or NULL
/// to build it by finite differences. `masses` is `dof` doubles or
/// NULL for unit mass.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_index1_create(
    config: *const RgsaddleIndex1Config,
    n_atoms: i64,
    position: *const f64,
    hessian: *const f64,
    masses: *const f64,
) -> *mut RgsaddleIndex1 {
    if config.is_null() || position.is_null() || n_atoms < 1 {
        return std::ptr::null_mut();
    }
    let cfg = unsafe { &*config };
    if cfg.version.major != RGSADDLE_ABI_MAJOR {
        return std::ptr::null_mut();
    }
    let dof = (3 * n_atoms) as usize;
    let x = Array1::from(unsafe { slice::from_raw_parts(position, dof) }.to_vec());
    let hess = if hessian.is_null() {
        None
    } else {
        let raw = unsafe { slice::from_raw_parts(hessian, dof * dof) }.to_vec();
        match Array2::from_shape_vec((dof, dof), raw) {
            Ok(m) => Some(m),
            Err(_) => return std::ptr::null_mut(),
        }
    };
    let mass = if masses.is_null() {
        None
    } else {
        Some(Array1::from(
            unsafe { slice::from_raw_parts(masses, dof) }.to_vec(),
        ))
    };
    let Some(mode) = nichols_mode(cfg.mode) else {
        return std::ptr::null_mut();
    };
    let update = match cfg.update {
        0 => HessianUpdate::Powell,
        1 => HessianUpdate::Bofill,
        _ => return std::ptr::null_mut(),
    };
    let session_config = Index1Config::default()
        .with_update(update)
        .with_mode(mode)
        .with_trust_radius(cfg.trust_radius)
        .with_force_tol(cfg.force_tol)
        .with_fd_dr(cfg.fd_dr);
    match Index1Session::new(session_config, x, hess, mass) {
        Ok(session) => Box::into_raw(Box::new(RgsaddleIndex1 { session, n_atoms })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// # Safety
/// `session` and `out` are valid. `surface` is called with `user`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_index1_step(
    session: *mut RgsaddleIndex1,
    surface: Option<RgsaddleSurfaceFn>,
    user: *mut c_void,
    out: *mut RgsaddleReport,
) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    if out.is_null() {
        return RGSADDLE_NULL_REPORT;
    }
    let Some(f) = surface else {
        return RGSADDLE_NULL_SURFACE;
    };
    let session = unsafe { &mut *session };
    let cs = CSurface::point(f, user, session.n_atoms);
    match session.session.step(&cs) {
        Ok(report) => {
            let out = unsafe { &mut *out };
            stamp_report(out);
            out.status = if report.status == Index1Status::Converged {
                1
            } else {
                0
            };
            out.evaluations = 0;
            out.max_force = report.max_force;
            out.ci_index = -1;
            out.iteration = report.iteration as i64;
            out.curvature = report.curvature;
            out.rotations = 0;
            RGSADDLE_OK
        }
        Err(e) => status_of(&e),
    }
}

/// # Safety
/// `out` holds `3 * n_atoms` doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_index1_position(
    session: *const RgsaddleIndex1,
    out: *mut f64,
) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    if out.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let session = unsafe { &*session };
    let dof = (3 * session.n_atoms) as usize;
    unsafe { write_doubles(out, dof, session.session.position().iter().copied()) };
    RGSADDLE_OK
}

/// # Safety
/// `out` holds `dof * dof` doubles. Fails with `RGSADDLE_SHAPE` when
/// the Hessian has not been built yet.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_index1_hessian(
    session: *const RgsaddleIndex1,
    out: *mut f64,
) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    if out.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let session = unsafe { &*session };
    let Some(h) = session.session.hessian() else {
        return RGSADDLE_SHAPE;
    };
    let dof = (3 * session.n_atoms) as usize;
    unsafe { write_doubles(out, dof * dof, h.iter().copied()) };
    RGSADDLE_OK
}

/// # Safety
/// `session` is a live pointer or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_index1_reset(session: *mut RgsaddleIndex1) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    unsafe { (*session).session.reset() };
    RGSADDLE_OK
}

/// # Safety
/// `session` comes from [`rgsaddle_index1_create`] and is freed once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_index1_free(session: *mut RgsaddleIndex1) {
    if !session.is_null() {
        drop(unsafe { Box::from_raw(session) });
    }
}

mod ocineb;
#[repr(C)]
/// The RgsaddleConnectConfig.
pub struct RgsaddleConnectConfig {
    /// ABI version this value was built against.
    pub version: RgsaddleVersion,
    /// Bit flags. The header names each bit.
    pub flags: u64,
    /// The search stops when the largest absolute force component falls under this value, in the host's force units.
    pub force_tol: f64,
    /// The max calls.
    pub max_calls: i64,
    /// The quench nodes.
    pub quench_nodes: i32,
}

#[repr(C)]
/// The RgsaddleConnectReport.
pub struct RgsaddleConnectReport {
    /// ABI version this value was built against.
    pub version: RgsaddleVersion,
    /// Bit flags. The header names each bit.
    pub flags: u64,
    /// Potential energy in the host's energy unit.
    pub energy: f64,
    /// The force norm.
    pub force_norm: f64,
    /// Which estimator produces the lowest mode.
    pub kind: i32,
    /// The hessian index.
    pub hessian_index: i32,
    /// Surface evaluations performed by this call.
    pub evaluations: i32,
}

/// # Safety
/// `left` and `right` each hold `3 * n_atoms` positions. `position`
/// has room for the same number. `surface` is called with one image
/// and must write the energy and the Cartesian gradient.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_connect(
    config: *const RgsaddleConnectConfig,
    n_atoms: i64,
    left: *const f64,
    right: *const f64,
    surface: Option<RgsaddleSurfaceFn>,
    user: *mut c_void,
    position: *mut f64,
    report: *mut RgsaddleConnectReport,
) -> i32 {
    if config.is_null() || report.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    if left.is_null() || right.is_null() || position.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let Some(surface) = surface else {
        return RGSADDLE_NULL_SURFACE;
    };
    if n_atoms <= 0 {
        return RGSADDLE_SHAPE;
    }
    let cfg = unsafe { &*config };
    if cfg.version.major != RGSADDLE_ABI_MAJOR {
        return RGSADDLE_ABI_MISMATCH;
    }
    if cfg.max_calls <= 0 || !(cfg.force_tol.is_finite() && cfg.force_tol > 0.0) {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let dof = (n_atoms as usize).saturating_mul(3);
    let left = unsafe { slice::from_raw_parts(left, dof) };
    let right = unsafe { slice::from_raw_parts(right, dof) };
    let settings = ConnectConfig::optbench(cfg.max_calls as usize)
        .with_forcetol(cfg.force_tol)
        .with_quench_nodes(cfg.quench_nodes != 0);
    let result = connect(
        |x| {
            let mut energy = 0.0;
            let mut gradient = vec![0.0; x.len()];
            let mut request = RgsaddleSurfaceRequest {
                version: RgsaddleVersion {
                    major: RGSADDLE_ABI_MAJOR,
                    minor: RGSADDLE_ABI_MINOR,
                },
                flags: 0,
                n_images: 1,
                n_atoms,
                positions: x.as_ptr(),
                energies: &mut energy,
                gradients: gradient.as_mut_ptr(),
                image: 0,
            };
            let status = surface(user, &mut request);
            if status != RGSADDLE_OK {
                return Err(SaddleError::Surface(format!(
                    "connect surface status {status}"
                )));
            }
            Ok((energy, gradient))
        },
        left,
        right,
        &settings,
    );
    match result {
        Ok(found) => {
            let dst = unsafe { slice::from_raw_parts_mut(position, dof) };
            dst.copy_from_slice(&found.position);
            unsafe {
                *report = RgsaddleConnectReport {
                    version: RgsaddleVersion {
                        major: RGSADDLE_ABI_MAJOR,
                        minor: RGSADDLE_ABI_MINOR,
                    },
                    flags: 0,
                    energy: found.energy,
                    force_norm: found.force_norm,
                    kind: found.kind as i32,
                    hessian_index: found.hessian_index,
                    evaluations: i32::try_from(found.evaluations).unwrap_or(i32::MAX),
                };
            }
            RGSADDLE_OK
        }
        Err(err) => status_of(&err),
    }
}

/// # Safety
/// `reactant` and `product` each hold `3 * n_atoms` doubles. `out`
/// holds `n_images * 3 * n_atoms` doubles. `cell` is 9 row-major
/// lattice entries or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_idpp_path(
    n_images: i64,
    n_atoms: i64,
    reactant: *const f64,
    product: *const f64,
    cell: *const f64,
    max_iterations: i64,
    grad_tol: f64,
    max_move: f64,
    out: *mut f64,
) -> i32 {
    if reactant.is_null() || product.is_null() || out.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    if n_images < 3 || n_atoms < 1 {
        return RGSADDLE_SHAPE;
    }
    let dof = (3 * n_atoms) as usize;
    let images = n_images as usize;
    let reactant = unsafe { slice::from_raw_parts(reactant, dof) };
    let product = unsafe { slice::from_raw_parts(product, dof) };
    let total = dof * images;
    let out = unsafe { slice::from_raw_parts_mut(out, total) };
    let parsed = if cell.is_null() {
        None
    } else {
        let raw = unsafe { slice::from_raw_parts(cell, 9) };
        match crate::mic::Cell::from_vectors(
            [raw[0], raw[1], raw[2]],
            [raw[3], raw[4], raw[5]],
            [raw[6], raw[7], raw[8]],
            [0.0, 0.0, 0.0],
        ) {
            Ok(cell) => Some(cell),
            Err(_) => return RGSADDLE_INVALID_PARAMETER,
        }
    };
    let config = crate::idpp::IdppConfig::default()
        .with_max_iterations(usize::try_from(max_iterations).unwrap_or(0))
        .with_grad_tol(grad_tol)
        .with_max_move(max_move);
    let reactant = match Array1::from_shape_vec(dof, reactant.to_vec()) {
        Ok(v) => v,
        Err(_) => return RGSADDLE_SHAPE,
    };
    let product = match Array1::from_shape_vec(dof, product.to_vec()) {
        Ok(v) => v,
        Err(_) => return RGSADDLE_SHAPE,
    };
    match crate::idpp_path(
        reactant.view(),
        product.view(),
        images,
        parsed.as_ref(),
        &config,
    ) {
        Ok(path) => {
            for (i, v) in path.iter().enumerate() {
                out[i] = *v;
            }
            RGSADDLE_OK
        }
        Err(err) => status_of(&err),
    }
}

mod indexk;
pub use indexk::*;
mod pafi_ohtst;
pub use pafi_ohtst::*;
mod sessions;
pub use sessions::*;
mod quickmin;
pub use quickmin::*;
mod saddle_index;
pub use saddle_index::*;
mod solid;
pub use solid::*;
mod stencil;
pub use stencil::*;
