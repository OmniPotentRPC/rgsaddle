//! OH-TST entry points, ABI minor 64.
//!
//! A point surface is one configuration of length `n`, not an image of
//! `3 N` atoms. Sampling, plane optimisation, reversible work (with MBAR),
//! the transmission coefficient and the rate are separate calls. The
//! callback may run on more than one thread: shooting does.

use std::os::raw::c_void;
use std::panic::AssertUnwindSafe;
use std::slice;

use ndarray::{Array1, Array2, ArrayView1};

use crate::error::SaddleError;
use crate::minmode::PointSurface;
use crate::ohtst::{
    AnalyticSurfaceDynamics, AnalyticSurfaceSampler, Hyperplane, LangevinConfig, MbarConfig,
    PlaneOptimizer, PlaneOptimizerConfig, PlaneOptimizerStatus, ReversibleWork,
    ReversibleWorkConfig, TransmissionConfig, WorkEstimate, mbar_free_energies, rate_from_parts,
    reversible_work, sample_plane, transmission_coefficient,
};

use super::{
    RGSADDLE_ABI_MAJOR, RGSADDLE_ABI_MINOR, RGSADDLE_ABI_MISMATCH, RGSADDLE_INVALID_PARAMETER,
    RGSADDLE_NOT_CONVERGED, RGSADDLE_NULL_SURFACE, RGSADDLE_OK, RGSADDLE_SHAPE, RGSADDLE_SOLVER,
    RgsaddleVersion, status_of,
    stencil::{clear_last_error, set_last_error},
};

/// Minor at which these entry points appeared. A config stamped earlier
/// is refused.
pub const RGSADDLE_OHTST_MINOR: u32 = 64;

/// Thermodynamic integration.
const ESTIMATE_INTEGRATION: i32 = 0;
/// BAR between neighbouring planes.
const ESTIMATE_BAR: i32 = 1;
/// MBAR across every plane.
const ESTIMATE_MBAR: i32 = 2;

/// The plane optimizer met its convergence test.
pub const RGSADDLE_OHTST_OPTIMIZE_CONVERGED: i32 = 0;
/// The plane optimizer reached its record ceiling while the norm test still requested growth.
pub const RGSADDLE_OHTST_OPTIMIZE_MAX_RECORDS: i32 = 1;
/// The plane optimizer reached its outer-iteration ceiling.
pub const RGSADDLE_OHTST_OPTIMIZE_MAX_ITERATIONS: i32 = 2;
/// The plane optimizer reached its surface-evaluation ceiling.
pub const RGSADDLE_OHTST_OPTIMIZE_MAX_EVALUATIONS: i32 = 3;

/// Point energy and gradient. `q` and `gradient` each hold `n` doubles.
pub type RgsaddlePointFn = extern "C" fn(*mut c_void, i64, *const f64, *mut f64, *mut f64) -> i32;

#[repr(C)]
/// Langevin sample of one plane.
pub struct RgsaddleOhtstSampleConfig {
    /// ABI version this value was built against.
    pub version: RgsaddleVersion,
    /// Bit flags. None are defined, and any nonzero value is rejected.
    pub flags: u64,
    /// `kT` in the energy unit of the surface.
    pub thermal: f64,
    /// Time step of the mass-weighted surface.
    pub time_step: f64,
    /// Langevin friction. Must be positive.
    pub friction: f64,
    /// Steps discarded before the first record.
    pub burn_in: u64,
    /// Steps between records. At least one.
    pub stride: u64,
    /// Records. At least two per block.
    pub records: u64,
    /// Most blocks for the standard error. At least two.
    pub blocks: u64,
    /// Seed of the noise.
    pub seed: u64,
}

#[repr(C)]
/// Averages from [`rgsaddle_ohtst_sample`].
pub struct RgsaddleOhtstSampleReport {
    /// ABI version written by the library.
    pub version: RgsaddleVersion,
    /// Bit flags. None are defined.
    pub flags: u64,
    /// Mean force along the normal, `⟨-ν · ∇V⟩`.
    pub normal_force: f64,
    /// Block standard error of [`Self::normal_force`].
    pub normal_force_error: f64,
    /// In-plane virial. At equilibrium it is the free dimension times `kT`.
    pub virial: f64,
    /// Block standard error of [`Self::virial`].
    pub virial_error: f64,
    /// Accepted fraction of the proposals.
    pub acceptance: f64,
    /// Records inside the blocks.
    pub records: u64,
    /// Surface evaluations.
    pub evaluations: u64,
}

#[repr(C)]
/// Trust-region climb of the plane free energy.
pub struct RgsaddleOhtstOptimizeConfig {
    /// ABI version this value was built against.
    pub version: RgsaddleVersion,
    /// Bit flags. None are defined, and any nonzero value is rejected.
    pub flags: u64,
    /// `kT` in the energy unit of the surface.
    pub thermal: f64,
    /// Time step of the mass-weighted surface.
    pub time_step: f64,
    /// Langevin friction. Must be positive.
    pub friction: f64,
    /// Steps discarded before the first record of each plane.
    pub burn_in: u64,
    /// Steps between records.
    pub stride: u64,
    /// Records of the first sample.
    pub records: u64,
    /// Most blocks for the standard error.
    pub blocks: u64,
    /// Seed of the noise.
    pub seed: u64,
    /// Outer trust-region iterations.
    pub max_iterations: u64,
    /// Hard cap on force/energy evaluations over the run.
    pub max_evaluations: u64,
    /// Largest sample, in records.
    pub max_records: u64,
    /// `theta` of the gradient norm test. Zero keeps the sample size fixed.
    pub norm_test: f64,
    /// Confidence multiplier of the statistical convergence test.
    pub confidence: f64,
    /// Absolute gradient and step tolerance.
    pub tolerance: f64,
    /// Longest move of the plane along its normal in one step.
    pub max_translation: f64,
    /// Turn of the normal, in radians, counted as long as [`Self::max_translation`].
    pub max_rotation: f64,
    /// Nonzero moves the plane along its normal.
    pub translate: i32,
    /// Nonzero turns the normal.
    pub rotate: i32,
    /// Converged samples in a row that end the climb. At least one.
    pub patience: u64,
}

#[repr(C)]
/// The last accepted plane of [`rgsaddle_ohtst_optimize`].
pub struct RgsaddleOhtstOptimizeReport {
    /// ABI version written by the library.
    pub version: RgsaddleVersion,
    /// Bit flags. None are defined.
    pub flags: u64,
    /// `rgsaddle_ohtst_optimize_status_t`.
    pub status: i32,
    /// Reserved. Written as zero.
    pub reserved: i32,
    /// Outer trust-region iterations.
    pub iterations: u64,
    /// Surface evaluations, metric actions included.
    pub evaluations: u64,
    /// Records in the last sample.
    pub records: u64,
    /// Mean force along the normal of the last plane.
    pub normal_force: f64,
    /// Its block standard error.
    pub normal_force_error: f64,
    /// Norm of the orientational gradient.
    pub rotation: f64,
    /// Root-sum-square block error of the orientational gradient.
    pub rotation_error: f64,
}

#[repr(C)]
/// Reversible work between planes, with BAR and optional MBAR.
pub struct RgsaddleOhtstWorkConfig {
    /// ABI version this value was built against.
    pub version: RgsaddleVersion,
    /// Bit flags. None are defined, and any nonzero value is rejected.
    pub flags: u64,
    /// `kT` in the energy unit of the surface.
    pub thermal: f64,
    /// Time step of the mass-weighted surface.
    pub time_step: f64,
    /// Langevin friction. Must be positive.
    pub friction: f64,
    /// Steps discarded before the first record of a cold start.
    pub burn_in: u64,
    /// Steps between records.
    pub stride: u64,
    /// Records on each node.
    pub records: u64,
    /// Most blocks for the standard error.
    pub blocks: u64,
    /// Seed of the noise.
    pub seed: u64,
    /// Nodes on each segment, ends included. Simpson weights need an odd count.
    pub nodes: u64,
    /// Configurations each node keeps for BAR and MBAR.
    pub bar_samples: u64,
    /// Nonzero solves MBAR across the nodes.
    pub mbar: i32,
    /// Which estimate [`RgsaddleOhtstWorkReport::work`] holds: 0 integration, 1 BAR, 2 MBAR.
    pub estimate: i32,
}

#[repr(C)]
/// `A(last) - A(first)`. Absent BAR or MBAR fields are NaN.
pub struct RgsaddleOhtstWorkReport {
    /// ABI version written by the library.
    pub version: RgsaddleVersion,
    /// Bit flags. None are defined.
    pub flags: u64,
    /// Work by [`RgsaddleOhtstWorkConfig::estimate`].
    pub work: f64,
    /// Standard error of [`Self::work`].
    pub error: f64,
    /// Work by thermodynamic integration.
    pub integrated_work: f64,
    /// Standard error of [`Self::integrated_work`].
    pub integrated_error: f64,
    /// Work by BAR. NaN when no configurations were kept.
    pub bar_work: f64,
    /// Standard error of [`Self::bar_work`]. NaN when BAR is off.
    pub bar_error: f64,
    /// Work by MBAR. NaN when MBAR is off.
    pub mbar_work: f64,
    /// Standard error of [`Self::mbar_work`]. NaN when MBAR is off.
    pub mbar_error: f64,
    /// Surface evaluations over the nodes.
    pub evaluations: u64,
}

#[repr(C)]
/// Shooting from one plane. Friction zero is Newtonian.
pub struct RgsaddleOhtstTransmissionConfig {
    /// ABI version this value was built against.
    pub version: RgsaddleVersion,
    /// Bit flags. None are defined, and any nonzero value is rejected.
    pub flags: u64,
    /// `kT` in the energy unit of the surface.
    pub thermal: f64,
    /// BAOAB time step.
    pub time_step: f64,
    /// Langevin friction. Zero is Newtonian.
    pub friction: f64,
    /// Steps of each half.
    pub steps: u64,
    /// Steps between checkpoints. The last step is always one.
    pub check_every: u64,
    /// Shots, spread evenly over the starts.
    pub shots: u64,
    /// Most blocks of consecutive shots. At least two, and no more than the shots.
    pub blocks: u64,
    /// Seed of the velocities. Shot `i` uses the seed plus `i`.
    pub seed: u64,
}

#[repr(C)]
/// Transmission coefficient counted at the first crossing.
pub struct RgsaddleOhtstTransmissionReport {
    /// ABI version written by the library.
    pub version: RgsaddleVersion,
    /// Bit flags. None are defined.
    pub flags: u64,
    /// `κ` at the end of the halves.
    pub kappa: f64,
    /// Block standard error of [`Self::kappa`].
    pub error: f64,
    /// Shots whose halves both reached a state.
    pub shots: u64,
    /// Shots left out because a half ended in no state.
    pub unassigned: u64,
    /// Shots from the reactant into the product.
    pub reactive: u64,
    /// Backward halves that crossed the plane again.
    pub recrossed: u64,
    /// Surface evaluations.
    pub evaluations: u64,
}

#[repr(C)]
/// Parts of [`rgsaddle_ohtst_rate`]. `work` is `ΔA` from the density's plane.
pub struct RgsaddleOhtstRateIn {
    /// ABI version this value was built against.
    pub version: RgsaddleVersion,
    /// Bit flags. None are defined, and any nonzero value is rejected.
    pub flags: u64,
    /// `kT` shared by the density and the work.
    pub thermal: f64,
    /// Reactant density `Q_Z / Q_R` on the first plane.
    pub density: f64,
    /// Standard error of [`Self::density`].
    pub density_error: f64,
    /// Reversible work from that plane to the optimal one.
    pub work: f64,
    /// Standard error of [`Self::work`].
    pub work_error: f64,
}

#[repr(C)]
/// `k = √(kT/2π) ρ exp(-ΔA/kT)` in the time unit of the mass-weighted surface.
pub struct RgsaddleOhtstRateOut {
    /// ABI version written by the library.
    pub version: RgsaddleVersion,
    /// Bit flags. None are defined.
    pub flags: u64,
    /// The rate.
    pub rate: f64,
    /// First-order standard error of [`Self::rate`].
    pub error: f64,
    /// `ln k`.
    pub log_rate: f64,
    /// Standard error of [`Self::log_rate`].
    pub log_rate_error: f64,
    /// `√(kT/2π)`.
    pub flux: f64,
}

struct PointFn {
    n: usize,
    eval: RgsaddlePointFn,
    user: usize,
}

impl PointSurface for PointFn {
    fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        if q.len() != self.n {
            return Err(SaddleError::Shape(
                "the surface was called at the wrong length".into(),
            ));
        }
        let position = q.to_vec();
        let mut energy = 0.0;
        let mut gradient = vec![0.0; self.n];
        let status = (self.eval)(
            self.user as *mut c_void,
            self.n as i64,
            position.as_ptr(),
            &mut energy,
            gradient.as_mut_ptr(),
        );
        if status != RGSADDLE_OK {
            return Err(SaddleError::Surface(format!(
                "point surface status {status}"
            )));
        }
        if !energy.is_finite() || gradient.iter().any(|value| !value.is_finite()) {
            return Err(SaddleError::NonFinite("point surface"));
        }
        Ok((energy, Array1::from_vec(gradient)))
    }
}

fn guard(body: impl FnOnce() -> i32) -> i32 {
    clear_last_error();
    match std::panic::catch_unwind(AssertUnwindSafe(body)) {
        Ok(code) => code,
        Err(_) => {
            set_last_error("OH-TST call panicked");
            RGSADDLE_SOLVER
        }
    }
}

/// BAR and MBAR report an infinite standard error when a variance
/// estimate overflows. The ABI carries that as a failure, never as `OK`.
fn require_finite_errors(work: &ReversibleWork) -> Result<(), SaddleError> {
    finite_errors([
        ("reversible work error", Some(work.error)),
        ("integrated work error", Some(work.integrated_error)),
        ("BAR work error", work.bar_error),
        ("MBAR work error", work.mbar_error),
    ])
}

fn finite_errors<const N: usize>(
    errors: [(&'static str, Option<f64>); N],
) -> Result<(), SaddleError> {
    match errors
        .into_iter()
        .find(|(_, error)| error.is_some_and(|value| !value.is_finite()))
    {
        Some((name, _)) => Err(SaddleError::NonFinite(name)),
        None => Ok(()),
    }
}

fn fail(err: &SaddleError) -> i32 {
    set_last_error(&err.to_string());
    status_of(err)
}

fn stamp() -> RgsaddleVersion {
    RgsaddleVersion {
        major: RGSADDLE_ABI_MAJOR,
        minor: RGSADDLE_ABI_MINOR,
    }
}

fn fail_code(code: i32, message: &str) -> i32 {
    set_last_error(message);
    code
}

fn accept_version(version: RgsaddleVersion) -> i32 {
    if version.major != RGSADDLE_ABI_MAJOR || version.minor < RGSADDLE_OHTST_MINOR {
        fail_code(
            RGSADDLE_ABI_MISMATCH,
            "OH-TST requires ABI major 1 and minor 64 or newer",
        )
    } else {
        RGSADDLE_OK
    }
}

fn as_usize(value: u64, what: &str) -> Result<usize, i32> {
    usize::try_from(value).map_err(|_| {
        fail_code(
            RGSADDLE_INVALID_PARAMETER,
            &format!("{what} does not fit in usize"),
        )
    })
}

/// A positive length stored in an `i64`. Negative values and values that
/// do not fit in `usize` are rejected, so a 32-bit target cannot truncate
/// a count before the allocation check.
fn positive_length(value: i64, what: &str) -> Result<usize, i32> {
    let length = usize::try_from(value).map_err(|_| {
        fail_code(
            RGSADDLE_INVALID_PARAMETER,
            &format!("{what} must be positive"),
        )
    })?;
    if length == 0 {
        return Err(fail_code(
            RGSADDLE_INVALID_PARAMETER,
            &format!("{what} must be positive"),
        ));
    }
    Ok(length)
}

/// Input flags. No bits are defined, so a nonzero word would ask for
/// behaviour this minor does not implement.
fn known_flags(flags: u64) -> i32 {
    if flags == 0 {
        RGSADDLE_OK
    } else {
        fail_code(RGSADDLE_INVALID_PARAMETER, "unknown OH-TST flags")
    }
}

fn finite_scalar(value: f64, what: &'static str) -> Result<f64, i32> {
    if !value.is_finite() {
        return Err(fail(&SaddleError::NonFinite(what)));
    }
    Ok(value)
}

fn finite_slice(ptr: *const f64, n: usize, what: &'static str) -> Result<Vec<f64>, i32> {
    if ptr.is_null() || n == 0 {
        return Err(fail_code(
            RGSADDLE_INVALID_PARAMETER,
            &format!("{what} must be non-null with positive length"),
        ));
    }
    if n > isize::MAX as usize / std::mem::size_of::<f64>() {
        return Err(fail_code(
            RGSADDLE_INVALID_PARAMETER,
            &format!("{what} length exceeds the addressable slice size"),
        ));
    }
    let values = unsafe { slice::from_raw_parts(ptr, n) };
    if values.iter().any(|value| !value.is_finite()) {
        return Err(fail(&SaddleError::NonFinite(what)));
    }
    Ok(values.to_vec())
}

fn write_owned(ptr: *mut f64, values: impl IntoIterator<Item = f64>) -> Result<(), i32> {
    let values: Vec<f64> = values.into_iter().collect();
    write_slice(ptr, &values)
}

fn write_slice(ptr: *mut f64, values: &[f64]) -> Result<(), i32> {
    if ptr.is_null() {
        return Err(fail_code(
            RGSADDLE_INVALID_PARAMETER,
            "output buffer pointer must be non-null",
        ));
    }
    let dst = unsafe { slice::from_raw_parts_mut(ptr, values.len()) };
    dst.copy_from_slice(values);
    Ok(())
}

fn plane_from(point: &[f64], normal: &[f64]) -> Result<Hyperplane, i32> {
    match Hyperplane::new(
        Array1::from_vec(point.to_vec()),
        Array1::from_vec(normal.to_vec()),
    ) {
        Ok(plane) => Ok(plane),
        Err(err) => Err(fail(&err)),
    }
}

#[allow(clippy::too_many_arguments)]
fn langevin_from(
    thermal: f64,
    time_step: f64,
    friction: f64,
    burn_in: u64,
    stride: u64,
    records: u64,
    blocks: u64,
    seed: u64,
    retain: u64,
) -> Result<LangevinConfig, i32> {
    let thermal = finite_scalar(thermal, "thermal")?;
    let time_step = finite_scalar(time_step, "time_step")?;
    let friction = finite_scalar(friction, "friction")?;
    Ok(LangevinConfig::default()
        .with_thermal(thermal)
        .with_time_step(time_step)
        .with_friction(friction)
        .with_burn_in(as_usize(burn_in, "burn_in")?)
        .with_stride(as_usize(stride, "stride")?)
        .with_records(as_usize(records, "records")?)
        .with_blocks(as_usize(blocks, "blocks")?)
        .with_seed(seed)
        .with_retain(as_usize(retain, "retain")?))
}

fn surface_of(eval: Option<RgsaddlePointFn>, user: *mut c_void, n: usize) -> Result<PointFn, i32> {
    let Some(eval) = eval else {
        return Err(fail_code(
            RGSADDLE_NULL_SURFACE,
            "point surface callback must be non-null",
        ));
    };
    Ok(PointFn {
        n,
        eval,
        user: user as usize,
    })
}

/// # Safety
/// `config` must be a valid pointer to one optimize config or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_ohtst_optimize_config_default(
    config: *mut RgsaddleOhtstOptimizeConfig,
) -> i32 {
    guard(|| {
        if config.is_null() {
            return fail_code(
                RGSADDLE_INVALID_PARAMETER,
                "optimize config pointer must be non-null",
            );
        }
        let default = PlaneOptimizerConfig::default();
        let sampler = &default.sampler;
        unsafe {
            *config = RgsaddleOhtstOptimizeConfig {
                version: stamp(),
                flags: 0,
                thermal: sampler.thermal,
                time_step: sampler.time_step,
                friction: sampler.friction,
                burn_in: sampler.burn_in as u64,
                stride: sampler.stride as u64,
                records: sampler.records as u64,
                blocks: sampler.blocks as u64,
                seed: sampler.seed,
                max_iterations: default.max_iterations as u64,
                max_evaluations: default.max_evaluations as u64,
                max_records: default.max_records as u64,
                norm_test: default.norm_test,
                confidence: default.confidence,
                tolerance: default.tolerance,
                max_translation: default.max_translation,
                max_rotation: default.max_rotation,
                translate: i32::from(default.translate),
                rotate: i32::from(default.rotate),
                patience: default.patience as u64,
            };
        }
        RGSADDLE_OK
    })
}

/// # Safety
/// `point`, `normal`, `centroid` and `grad_normal` each hold `n` doubles.
/// `surface` is called with that length and may run on more than one thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_ohtst_sample(
    config: *const RgsaddleOhtstSampleConfig,
    n: i64,
    point: *const f64,
    normal: *const f64,
    surface: Option<RgsaddlePointFn>,
    user: *mut c_void,
    centroid: *mut f64,
    grad_normal: *mut f64,
    report: *mut RgsaddleOhtstSampleReport,
) -> i32 {
    guard(|| unsafe {
        sample_call(
            config,
            n,
            point,
            normal,
            surface,
            user,
            centroid,
            grad_normal,
            report,
        )
    })
}

#[allow(clippy::too_many_arguments)]
unsafe fn sample_call(
    config: *const RgsaddleOhtstSampleConfig,
    n: i64,
    point: *const f64,
    normal: *const f64,
    surface: Option<RgsaddlePointFn>,
    user: *mut c_void,
    centroid: *mut f64,
    grad_normal: *mut f64,
    report: *mut RgsaddleOhtstSampleReport,
) -> i32 {
    if config.is_null() || report.is_null() || centroid.is_null() || grad_normal.is_null() {
        return fail_code(
            RGSADDLE_INVALID_PARAMETER,
            "config, centroid, grad_normal and report pointers must be non-null",
        );
    }
    let config = unsafe { &*config };
    let version = accept_version(config.version);
    if version != RGSADDLE_OK {
        return version;
    }
    let flags = known_flags(config.flags);
    if flags != RGSADDLE_OK {
        return flags;
    }
    let n = match positive_length(n, "dimension") {
        Ok(n) => n,
        Err(code) => return code,
    };
    let point = match finite_slice(point, n, "point") {
        Ok(values) => values,
        Err(code) => return code,
    };
    let normal = match finite_slice(normal, n, "normal") {
        Ok(values) => values,
        Err(code) => return code,
    };
    let plane = match plane_from(&point, &normal) {
        Ok(plane) => plane,
        Err(code) => return code,
    };
    let sampler = match langevin_from(
        config.thermal,
        config.time_step,
        config.friction,
        config.burn_in,
        config.stride,
        config.records,
        config.blocks,
        config.seed,
        0,
    ) {
        Ok(sampler) => sampler,
        Err(code) => return code,
    };
    let surface = match surface_of(surface, user, n) {
        Ok(surface) => surface,
        Err(code) => return code,
    };
    let sample = match sample_plane(&surface, &plane, None, &sampler) {
        Ok(sample) => sample,
        Err(err) => return fail(&err),
    };
    if write_owned(centroid, sample.centroid.iter().copied()).is_err()
        || write_owned(grad_normal, sample.grad_normal.iter().copied()).is_err()
    {
        return RGSADDLE_INVALID_PARAMETER;
    }
    unsafe {
        *report = RgsaddleOhtstSampleReport {
            version: stamp(),
            flags: 0,
            normal_force: sample.normal_force,
            normal_force_error: sample.normal_force_error,
            virial: sample.virial,
            virial_error: sample.virial_error,
            acceptance: sample.acceptance,
            records: sample.records as u64,
            evaluations: sample.evaluations as u64,
        };
    }
    RGSADDLE_OK
}

/// # Safety
/// `point` and `normal` each hold `n` doubles on the way in and on the way
/// out. `surface` may run on more than one thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_ohtst_optimize(
    config: *const RgsaddleOhtstOptimizeConfig,
    n: i64,
    point: *mut f64,
    normal: *mut f64,
    surface: Option<RgsaddlePointFn>,
    user: *mut c_void,
    report: *mut RgsaddleOhtstOptimizeReport,
) -> i32 {
    guard(|| unsafe { optimize_call(config, n, point, normal, surface, user, report) })
}

unsafe fn optimize_call(
    config: *const RgsaddleOhtstOptimizeConfig,
    n: i64,
    point: *mut f64,
    normal: *mut f64,
    surface: Option<RgsaddlePointFn>,
    user: *mut c_void,
    report: *mut RgsaddleOhtstOptimizeReport,
) -> i32 {
    if config.is_null() || report.is_null() {
        return fail_code(
            RGSADDLE_INVALID_PARAMETER,
            "config and report pointers must be non-null",
        );
    }
    let config = unsafe { &*config };
    let version = accept_version(config.version);
    if version != RGSADDLE_OK {
        return version;
    }
    let flags = known_flags(config.flags);
    if flags != RGSADDLE_OK {
        return flags;
    }
    let n = match positive_length(n, "dimension") {
        Ok(n) => n,
        Err(code) => return code,
    };
    let point_in = match finite_slice(point, n, "point") {
        Ok(values) => values,
        Err(code) => return code,
    };
    let normal_in = match finite_slice(normal, n, "normal") {
        Ok(values) => values,
        Err(code) => return code,
    };
    let plane = match plane_from(&point_in, &normal_in) {
        Ok(plane) => plane,
        Err(code) => return code,
    };
    let sampler = match langevin_from(
        config.thermal,
        config.time_step,
        config.friction,
        config.burn_in,
        config.stride,
        config.records,
        config.blocks,
        config.seed,
        0,
    ) {
        Ok(sampler) => sampler,
        Err(code) => return code,
    };
    let max_iterations = match as_usize(config.max_iterations, "max_iterations") {
        Ok(value) => value,
        Err(code) => return code,
    };
    let max_evaluations = match as_usize(config.max_evaluations, "max_evaluations") {
        Ok(value) => value,
        Err(code) => return code,
    };
    let max_records = match as_usize(config.max_records, "max_records") {
        Ok(value) => value,
        Err(code) => return code,
    };
    let norm_test = match finite_scalar(config.norm_test, "norm_test") {
        Ok(value) => value,
        Err(code) => return code,
    };
    let confidence = match finite_scalar(config.confidence, "confidence") {
        Ok(value) => value,
        Err(code) => return code,
    };
    let tolerance = match finite_scalar(config.tolerance, "tolerance") {
        Ok(value) => value,
        Err(code) => return code,
    };
    let max_translation = match finite_scalar(config.max_translation, "max_translation") {
        Ok(value) => value,
        Err(code) => return code,
    };
    let max_rotation = match finite_scalar(config.max_rotation, "max_rotation") {
        Ok(value) => value,
        Err(code) => return code,
    };
    let patience = match as_usize(config.patience, "patience") {
        Ok(value) if value >= 1 => value,
        Ok(_) => {
            return fail_code(RGSADDLE_INVALID_PARAMETER, "patience must be at least one");
        }
        Err(code) => return code,
    };
    let optimizer = PlaneOptimizerConfig::default()
        .with_sampler(sampler)
        .with_max_records(max_records)
        .with_norm_test(norm_test)
        .with_confidence(confidence)
        .with_tolerance(tolerance)
        .with_max_translation(max_translation)
        .with_max_rotation(max_rotation)
        .with_translate(config.translate != 0)
        .with_rotate(config.rotate != 0)
        .with_max_iterations(max_iterations)
        .with_max_evaluations(max_evaluations)
        .with_patience(patience);
    let surface = match surface_of(surface, user, n) {
        Ok(surface) => surface,
        Err(code) => return code,
    };
    let session = match PlaneOptimizer::new(&plane, optimizer) {
        Ok(session) => session,
        Err(err) => return fail(&err),
    };
    let optimum = match session.run(&surface) {
        Ok(optimum) => optimum,
        Err(err) => return fail(&err),
    };
    if write_owned(point, optimum.plane.point().iter().copied()).is_err()
        || write_owned(normal, optimum.plane.normal().iter().copied()).is_err()
    {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let (status, status_code) = match optimum.status {
        PlaneOptimizerStatus::Converged => (RGSADDLE_OHTST_OPTIMIZE_CONVERGED, RGSADDLE_OK),
        PlaneOptimizerStatus::MaxRecords => {
            (RGSADDLE_OHTST_OPTIMIZE_MAX_RECORDS, RGSADDLE_NOT_CONVERGED)
        }
        PlaneOptimizerStatus::MaxIterations => (
            RGSADDLE_OHTST_OPTIMIZE_MAX_ITERATIONS,
            RGSADDLE_NOT_CONVERGED,
        ),
        PlaneOptimizerStatus::MaxEvaluations => (
            RGSADDLE_OHTST_OPTIMIZE_MAX_EVALUATIONS,
            RGSADDLE_NOT_CONVERGED,
        ),
    };
    let last = optimum.trace.last();
    let rotation = last.map_or(0.0, |entry| entry.rotation);
    let rotation_error = last.map_or(0.0, |entry| entry.rotation_error);
    unsafe {
        *report = RgsaddleOhtstOptimizeReport {
            version: stamp(),
            flags: 0,
            status,
            reserved: 0,
            iterations: optimum.iterations as u64,
            evaluations: optimum.evaluations as u64,
            records: optimum.sample.records as u64,
            normal_force: optimum.sample.normal_force,
            normal_force_error: optimum.sample.normal_force_error,
            rotation,
            rotation_error,
        };
    }
    if status_code != RGSADDLE_OK {
        set_last_error(&format!(
            "the plane did not converge ({:?}): {} records, normal force {:.6e} +/- {:.6e}, rotation {:.6e} +/- {:.6e}",
            optimum.status,
            optimum.sample.records,
            optimum.sample.normal_force,
            optimum.sample.normal_force_error,
            rotation,
            rotation_error,
        ));
    }
    status_code
}

/// # Safety
/// `reduced` holds `samples * states` row-major doubles. `counts` holds
/// `states` positive counts that sum to `samples`. `initial` is `states`
/// doubles or NULL. Each output holds `states` or `states * states` doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_ohtst_mbar(
    samples: i64,
    states: i64,
    reduced: *const f64,
    counts: *const i64,
    initial: *const f64,
    free_energies: *mut f64,
    covariance: *mut f64,
    overlap: *mut f64,
) -> i32 {
    guard(|| unsafe {
        mbar_call(
            samples,
            states,
            reduced,
            counts,
            initial,
            free_energies,
            covariance,
            overlap,
        )
    })
}

#[allow(clippy::too_many_arguments)]
unsafe fn mbar_call(
    samples: i64,
    states: i64,
    reduced: *const f64,
    counts: *const i64,
    initial: *const f64,
    free_energies: *mut f64,
    covariance: *mut f64,
    overlap: *mut f64,
) -> i32 {
    if samples <= 0 || states < 2 {
        return fail_code(
            RGSADDLE_SHAPE,
            "MBAR needs at least one sample and two states",
        );
    }
    let Ok(samples) = usize::try_from(samples) else {
        return fail_code(RGSADDLE_SHAPE, "MBAR sample count does not fit in usize");
    };
    let Ok(states) = usize::try_from(states) else {
        return fail_code(RGSADDLE_SHAPE, "MBAR state count does not fit in usize");
    };
    // The covariance and the overlap are `states` by `states`. Reject the
    // product before any allocation so a 32-bit `usize` cannot wrap it.
    if samples.checked_mul(states).is_none() || states.checked_mul(states).is_none() {
        return fail_code(RGSADDLE_SHAPE, "MBAR matrix dimensions overflow usize");
    }
    let cells = samples * states;
    let state_square = states * states;
    let reduced_bytes = cells.checked_mul(std::mem::size_of::<f64>());
    let square_bytes = state_square.checked_mul(std::mem::size_of::<f64>());
    let count_bytes = states.checked_mul(std::mem::size_of::<i64>());
    if reduced_bytes.is_none_or(|bytes| bytes > isize::MAX as usize)
        || square_bytes.is_none_or(|bytes| bytes > isize::MAX as usize)
        || count_bytes.is_none_or(|bytes| bytes > isize::MAX as usize)
    {
        return fail_code(RGSADDLE_SHAPE, "MBAR matrix byte length exceeds isize::MAX");
    }
    if counts.is_null() || free_energies.is_null() || covariance.is_null() || overlap.is_null() {
        return fail_code(
            RGSADDLE_INVALID_PARAMETER,
            "counts, free_energies, covariance and overlap pointers must be non-null",
        );
    }
    if reduced.is_null() {
        return fail_code(RGSADDLE_INVALID_PARAMETER, "reduced must be non-null");
    }
    let reduced = unsafe { slice::from_raw_parts(reduced, cells) };
    if reduced
        .iter()
        .any(|value| value.is_nan() || *value == f64::NEG_INFINITY)
    {
        return fail(&SaddleError::NonFinite("reduced"));
    }
    let reduced = reduced.to_vec();
    let count_raw = unsafe { slice::from_raw_parts(counts, states) };
    let mut count_usize = Vec::with_capacity(states);
    for &count in count_raw {
        if count <= 0 {
            return fail_code(RGSADDLE_INVALID_PARAMETER, "MBAR counts must be positive");
        }
        match usize::try_from(count) {
            Ok(count) => count_usize.push(count),
            Err(_) => {
                return fail_code(
                    RGSADDLE_INVALID_PARAMETER,
                    "MBAR count does not fit in usize",
                );
            }
        }
    }
    let matrix = match Array2::from_shape_vec((samples, states), reduced) {
        Ok(matrix) => matrix,
        Err(_) => return fail_code(RGSADDLE_SHAPE, "MBAR reduced matrix shape mismatch"),
    };
    let mut config = MbarConfig::default();
    if !initial.is_null() {
        let start = match finite_slice(initial, states, "initial") {
            Ok(values) => values,
            Err(code) => return code,
        };
        config = config.with_initial(Array1::from_vec(start));
    }
    let solved = match mbar_free_energies(&matrix, &count_usize, &config) {
        Ok(solved) => solved,
        Err(err) => return fail(&err),
    };
    if write_owned(free_energies, solved.free_energies.iter().copied()).is_err()
        || write_owned(covariance, solved.covariance.iter().copied()).is_err()
        || write_owned(overlap, solved.overlap.iter().copied()).is_err()
    {
        return RGSADDLE_INVALID_PARAMETER;
    }
    RGSADDLE_OK
}

/// # Safety
/// `points` and `normals` each hold `n_planes * n` row-major doubles.
/// `start` holds `n` doubles or is NULL. `surface` may run on more than one thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_ohtst_work(
    config: *const RgsaddleOhtstWorkConfig,
    n: i64,
    n_planes: i64,
    points: *const f64,
    normals: *const f64,
    start: *const f64,
    surface: Option<RgsaddlePointFn>,
    user: *mut c_void,
    report: *mut RgsaddleOhtstWorkReport,
) -> i32 {
    guard(|| unsafe {
        work_call(
            config, n, n_planes, points, normals, start, surface, user, report,
        )
    })
}

#[allow(clippy::too_many_arguments)]
unsafe fn work_call(
    config: *const RgsaddleOhtstWorkConfig,
    n: i64,
    n_planes: i64,
    points: *const f64,
    normals: *const f64,
    start: *const f64,
    surface: Option<RgsaddlePointFn>,
    user: *mut c_void,
    report: *mut RgsaddleOhtstWorkReport,
) -> i32 {
    if config.is_null() || report.is_null() {
        return fail_code(
            RGSADDLE_INVALID_PARAMETER,
            "config and report pointers must be non-null",
        );
    }
    let config = unsafe { &*config };
    let version = accept_version(config.version);
    if version != RGSADDLE_OK {
        return version;
    }
    let flags = known_flags(config.flags);
    if flags != RGSADDLE_OK {
        return flags;
    }
    let n = match positive_length(n, "dimension") {
        Ok(n) => n,
        Err(code) => return code,
    };
    let Ok(n_planes) = usize::try_from(n_planes) else {
        return fail_code(RGSADDLE_INVALID_PARAMETER, "n_planes must be at least two");
    };
    if n_planes < 2 {
        return fail_code(RGSADDLE_INVALID_PARAMETER, "n_planes must be at least two");
    }
    let Some(cells) = n.checked_mul(n_planes) else {
        return fail_code(RGSADDLE_SHAPE, "plane array dimensions overflow usize");
    };
    let points = match finite_slice(points, cells, "points") {
        Ok(values) => values,
        Err(code) => return code,
    };
    let normals = match finite_slice(normals, cells, "normals") {
        Ok(values) => values,
        Err(code) => return code,
    };
    let mut planes = Vec::with_capacity(n_planes);
    for plane in 0..n_planes {
        let start_at = plane * n;
        let end = start_at + n;
        match plane_from(&points[start_at..end], &normals[start_at..end]) {
            Ok(plane) => planes.push(plane),
            Err(code) => return code,
        }
    }
    let start = if start.is_null() {
        None
    } else {
        match finite_slice(start, n, "start") {
            Ok(values) => Some(Array1::from_vec(values)),
            Err(code) => return code,
        }
    };
    let retain = config.bar_samples;
    let sampler = match langevin_from(
        config.thermal,
        config.time_step,
        config.friction,
        config.burn_in,
        config.stride,
        config.records,
        config.blocks,
        config.seed,
        retain,
    ) {
        Ok(sampler) => sampler,
        Err(code) => return code,
    };
    let estimate = match config.estimate {
        ESTIMATE_INTEGRATION => WorkEstimate::Integration,
        ESTIMATE_BAR => WorkEstimate::Bar,
        ESTIMATE_MBAR => WorkEstimate::Mbar,
        _ => {
            return fail_code(
                RGSADDLE_INVALID_PARAMETER,
                "estimate must be 0 (integration), 1 (BAR) or 2 (MBAR)",
            );
        }
    };
    let nodes = match as_usize(config.nodes, "nodes") {
        Ok(value) => value,
        Err(code) => return code,
    };
    let bar_samples = match as_usize(config.bar_samples, "bar_samples") {
        Ok(value) => value,
        Err(code) => return code,
    };
    let work_config = ReversibleWorkConfig::default()
        .with_sampler(sampler)
        .with_nodes(nodes)
        .with_bar_samples(bar_samples)
        .with_mbar(config.mbar != 0)
        .with_estimate(estimate);
    let surface = match surface_of(surface, user, n) {
        Ok(surface) => surface,
        Err(code) => return code,
    };
    let mut engine = AnalyticSurfaceSampler::new(&surface, work_config.sampler.clone());
    if let Some(start) = start {
        engine = engine.with_start(start);
    }
    let work = match reversible_work(&mut engine, &planes, &work_config) {
        Ok(work) => work,
        Err(err) => return fail(&err),
    };
    if let Err(err) = require_finite_errors(&work) {
        return fail(&err);
    }
    let mut output = RgsaddleOhtstWorkReport {
        version: stamp(),
        flags: 0,
        work: work.work,
        error: work.error,
        integrated_work: work.integrated_work,
        integrated_error: work.integrated_error,
        bar_work: f64::NAN,
        bar_error: f64::NAN,
        mbar_work: f64::NAN,
        mbar_error: f64::NAN,
        evaluations: work.evaluations as u64,
    };
    if let Some(value) = work.bar_work {
        output.bar_work = value;
    }
    if let Some(value) = work.bar_error {
        output.bar_error = value;
    }
    if let Some(value) = work.mbar_work {
        output.mbar_work = value;
    }
    if let Some(value) = work.mbar_error {
        output.mbar_error = value;
    }
    unsafe {
        *report = output;
    }
    RGSADDLE_OK
}

/// # Safety
/// `point` and `normal` each hold `n` doubles. `starts` holds
/// `n_starts * n` row-major doubles. `surface` may run on more than one thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_ohtst_transmission(
    config: *const RgsaddleOhtstTransmissionConfig,
    n: i64,
    point: *const f64,
    normal: *const f64,
    n_starts: i64,
    starts: *const f64,
    surface: Option<RgsaddlePointFn>,
    user: *mut c_void,
    report: *mut RgsaddleOhtstTransmissionReport,
) -> i32 {
    guard(|| unsafe {
        transmission_call(
            config, n, point, normal, n_starts, starts, surface, user, report,
        )
    })
}

#[allow(clippy::too_many_arguments)]
unsafe fn transmission_call(
    config: *const RgsaddleOhtstTransmissionConfig,
    n: i64,
    point: *const f64,
    normal: *const f64,
    n_starts: i64,
    starts: *const f64,
    surface: Option<RgsaddlePointFn>,
    user: *mut c_void,
    report: *mut RgsaddleOhtstTransmissionReport,
) -> i32 {
    if config.is_null() || report.is_null() {
        return fail_code(
            RGSADDLE_INVALID_PARAMETER,
            "config and report pointers must be non-null",
        );
    }
    let config = unsafe { &*config };
    let version = accept_version(config.version);
    if version != RGSADDLE_OK {
        return version;
    }
    let flags = known_flags(config.flags);
    if flags != RGSADDLE_OK {
        return flags;
    }
    let n = match positive_length(n, "dimension") {
        Ok(n) => n,
        Err(code) => return code,
    };
    let n_starts = match positive_length(n_starts, "n_starts") {
        Ok(n_starts) => n_starts,
        Err(code) => return code,
    };
    let Some(cells) = n.checked_mul(n_starts) else {
        return fail_code(RGSADDLE_SHAPE, "starts array dimensions overflow usize");
    };
    let point = match finite_slice(point, n, "point") {
        Ok(values) => values,
        Err(code) => return code,
    };
    let normal = match finite_slice(normal, n, "normal") {
        Ok(values) => values,
        Err(code) => return code,
    };
    let plane = match plane_from(&point, &normal) {
        Ok(plane) => plane,
        Err(code) => return code,
    };
    let packed = match finite_slice(starts, cells, "starts") {
        Ok(values) => values,
        Err(code) => return code,
    };
    let starts: Vec<Array1<f64>> = packed
        .chunks(n)
        .map(|row| Array1::from_vec(row.to_vec()))
        .collect();
    let thermal = match finite_scalar(config.thermal, "thermal") {
        Ok(value) => value,
        Err(code) => return code,
    };
    let time_step = match finite_scalar(config.time_step, "time_step") {
        Ok(value) => value,
        Err(code) => return code,
    };
    let friction = match finite_scalar(config.friction, "friction") {
        Ok(value) => value,
        Err(code) => return code,
    };
    let shots = match as_usize(config.shots, "shots") {
        Ok(value) => value,
        Err(code) => return code,
    };
    let transmission = TransmissionConfig::default()
        .with_thermal(thermal)
        .with_time_step(time_step)
        .with_friction(friction)
        .with_steps(match as_usize(config.steps, "steps") {
            Ok(value) => value,
            Err(code) => return code,
        })
        .with_check_every(match as_usize(config.check_every, "check_every") {
            Ok(value) => value,
            Err(code) => return code,
        })
        .with_shots(shots)
        .with_blocks(match as_usize(config.blocks, "blocks") {
            Ok(value) => value,
            Err(code) => return code,
        })
        .with_seed(config.seed);
    let surface = match surface_of(surface, user, n) {
        Ok(surface) => surface,
        Err(code) => return code,
    };
    let mut dynamics = AnalyticSurfaceDynamics::new(&surface);
    let coefficient = match transmission_coefficient(&mut dynamics, &plane, &starts, &transmission)
    {
        Ok(coefficient) => coefficient,
        Err(err) => return fail(&err),
    };
    unsafe {
        *report = RgsaddleOhtstTransmissionReport {
            version: stamp(),
            flags: 0,
            kappa: coefficient.kappa,
            error: coefficient.error,
            shots: coefficient.shots as u64,
            unassigned: coefficient.unassigned as u64,
            reactive: coefficient.reactive as u64,
            recrossed: coefficient.recrossed as u64,
            evaluations: coefficient.evaluations as u64,
        };
    }
    RGSADDLE_OK
}

/// # Safety
/// `input` and `out` are live structs. The caller has already paired the
/// density with a work path that starts on its plane.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_ohtst_rate(
    input: *const RgsaddleOhtstRateIn,
    out: *mut RgsaddleOhtstRateOut,
) -> i32 {
    guard(|| unsafe { rate_call(input, out) })
}

unsafe fn rate_call(input: *const RgsaddleOhtstRateIn, out: *mut RgsaddleOhtstRateOut) -> i32 {
    if input.is_null() || out.is_null() {
        return fail_code(
            RGSADDLE_INVALID_PARAMETER,
            "input and out pointers must be non-null",
        );
    }
    let input = unsafe { &*input };
    let version = accept_version(input.version);
    if version != RGSADDLE_OK {
        return version;
    }
    let flags = known_flags(input.flags);
    if flags != RGSADDLE_OK {
        return flags;
    }
    let rate = match rate_from_parts(
        input.thermal,
        input.density,
        input.density_error,
        input.work,
        input.work_error,
    ) {
        Ok(rate) => rate,
        Err(err) => return fail(&err),
    };
    unsafe {
        *out = RgsaddleOhtstRateOut {
            version: stamp(),
            flags: 0,
            rate: rate.rate,
            error: rate.error,
            log_rate: rate.log_rate,
            log_rate_error: rate.log_rate_error,
            flux: rate.flux,
        };
    }
    RGSADDLE_OK
}

#[cfg(test)]
mod tests {
    use super::super::RGSADDLE_NON_FINITE;
    use super::*;

    #[test]
    fn an_infinite_or_nan_error_is_not_ok() {
        for bad in [f64::INFINITY, f64::NAN, f64::NEG_INFINITY] {
            let err = finite_errors([("work error", Some(0.5)), ("BAR work error", Some(bad))])
                .unwrap_err();
            assert!(matches!(err, SaddleError::NonFinite("BAR work error")));
            assert_eq!(status_of(&err), RGSADDLE_NON_FINITE);
        }
    }

    #[test]
    fn finite_and_absent_errors_are_ok() {
        assert!(finite_errors([("work error", Some(0.5)), ("BAR work error", None)]).is_ok());
    }

    #[test]
    fn finite_slice_rejects_a_byte_length_above_isize() {
        let ptr = std::ptr::NonNull::<f64>::dangling().as_ptr();
        let n = isize::MAX as usize / std::mem::size_of::<f64>() + 1;
        assert_eq!(
            finite_slice(ptr, n, "values").unwrap_err(),
            RGSADDLE_INVALID_PARAMETER
        );
    }
}
