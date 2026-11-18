use std::ffi::c_void;
use std::slice;

use ndarray::{Array1, ArrayView1};

use crate::SaddleError;
use crate::minmode::PointSurface;
use crate::ohtst::{
    AnalyticSurfaceDynamics, Counting, Hyperplane, LangevinConfig, PlaneIntegrator,
    TransmissionCoefficient, TransmissionConfig, sample_plane, transmission_coefficient,
};
use crate::pafi::{ArcPath, FreeEnergyProfile, HyperplaneAverage, PafiConfig, pafi_profile};

use super::{
    RGSADDLE_ABI_MAJOR, RGSADDLE_ABI_MINOR, RGSADDLE_INVALID_PARAMETER, RGSADDLE_NO_EVALUATION,
    RGSADDLE_NULL_REPORT, RGSADDLE_NULL_SESSION, RGSADDLE_NULL_SURFACE, RGSADDLE_OK,
    RGSADDLE_SHAPE, RgsaddleVersion, status_of,
};

/// The point-surface callback used by the PAFI and OH-TST C handles.
pub type RgsaddlePointSurfaceFn =
    extern "C" fn(*mut c_void, i64, *const f64, *mut f64, *mut f64) -> i32;

struct CPointSurface {
    callback: RgsaddlePointSurfaceFn,
    user: *mut c_void,
    n_dof: usize,
}

// The C host owns thread-safety of `user`.
unsafe impl Send for CPointSurface {}
unsafe impl Sync for CPointSurface {}

impl PointSurface for CPointSurface {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        if x.len() != self.n_dof {
            return Err(SaddleError::Shape("point callback shape mismatch".into()));
        }
        let mut energy = 0.0;
        let mut gradient = vec![0.0; self.n_dof];
        let rc = (self.callback)(
            self.user,
            self.n_dof as i64,
            x.as_ptr(),
            &mut energy,
            gradient.as_mut_ptr(),
        );
        if rc != RGSADDLE_OK {
            return Err(SaddleError::Surface(format!("host callback rc={rc}")));
        }
        Ok((energy, Array1::from_vec(gradient)))
    }
}

/// PAFI configuration over the C ABI.
#[repr(C)]
pub struct RgsaddlePafiConfig {
    /// ABI version this value was built against.
    pub version: RgsaddleVersion,
    /// Bit flags. Reserved; write zero.
    pub flags: u64,
    /// Thermal energy `kT` in the surface's energy unit.
    pub thermal: f64,
    /// Overdamped mobility `mu`.
    pub mobility: f64,
    /// Overdamped integration time step.
    pub time_step: f64,
    /// Steps discarded per hyperplane before recording.
    pub thermalize: i64,
    /// Steps between recorded samples.
    pub stride: i64,
    /// Recorded samples per hyperplane.
    pub samples: i64,
    /// Contiguous blocks for block averaging.
    pub blocks: i64,
    /// Random seed for the Langevin noise.
    pub seed: u64,
    /// Nonzero removes center-of-mass drift for multi-atom paths.
    pub fix_com: i32,
    /// Number of configurations retained per hyperplane.
    pub retain: i64,
    /// Positive target barrier error enables adaptive rounds; non-positive runs one pass.
    pub target_error: f64,
    /// Maximum adaptive resampling rounds.
    pub max_rounds: i64,
}

/// PAFI barrier summary returned after `rgsaddle_pafi_run`.
#[repr(C)]
pub struct RgsaddlePafiResult {
    /// ABI version stamped by the library.
    pub version: RgsaddleVersion,
    /// Bit flags. Zeroed by the library.
    pub flags: u64,
    /// Free-energy barrier maximum relative to the reactant plane.
    pub barrier: f64,
    /// Propagated block standard error at the barrier.
    pub barrier_error: f64,
    /// Zero-based index of the dividing hyperplane in the profile.
    pub dividing: i64,
    /// Total recorded samples summed over hyperplanes and rounds.
    pub samples: i64,
    /// Total surface evaluations summed over hyperplanes and rounds.
    pub evaluations: i64,
}

/// Opaque PAFI session created by [`rgsaddle_pafi_create`].
pub struct RgsaddlePafi {
    config: PafiConfig,
    path: ArcPath,
    arcs: Vec<f64>,
    profile: Option<FreeEnergyProfile>,
    planes: Vec<HyperplaneAverage>,
}

fn stamp(out: &mut RgsaddleVersion) {
    out.major = RGSADDLE_ABI_MAJOR;
    out.minor = RGSADDLE_ABI_MINOR;
}

#[allow(clippy::manual_unwrap_or)]
fn as_i64_saturated(value: usize) -> i64 {
    match i64::try_from(value) {
        Ok(converted) => converted,
        Err(_) => i64::MAX,
    }
}

fn parse_pafi_config(config: &RgsaddlePafiConfig) -> Result<PafiConfig, SaddleError> {
    if config.version.major != RGSADDLE_ABI_MAJOR {
        return Err(SaddleError::Invalid("PAFI ABI major mismatch".into()));
    }
    let thermalize = usize::try_from(config.thermalize)
        .map_err(|_| SaddleError::Invalid("PAFI thermalize".into()))?;
    let stride =
        usize::try_from(config.stride).map_err(|_| SaddleError::Invalid("PAFI stride".into()))?;
    let samples =
        usize::try_from(config.samples).map_err(|_| SaddleError::Invalid("PAFI samples".into()))?;
    let blocks =
        usize::try_from(config.blocks).map_err(|_| SaddleError::Invalid("PAFI blocks".into()))?;
    let retain =
        usize::try_from(config.retain).map_err(|_| SaddleError::Invalid("PAFI retain".into()))?;
    let max_rounds = usize::try_from(config.max_rounds)
        .map_err(|_| SaddleError::Invalid("PAFI max rounds".into()))?;
    if config.target_error.is_nan() {
        return Err(SaddleError::Invalid("PAFI target error is NaN".into()));
    }
    let target_error = if config.target_error.is_finite() && config.target_error > 0.0 {
        Some(config.target_error)
    } else {
        None
    };
    let parsed = PafiConfig::default()
        .with_thermal(config.thermal)
        .with_mobility(config.mobility)
        .with_time_step(config.time_step)
        .with_thermalize(thermalize)
        .with_stride(stride)
        .with_samples(samples)
        .with_blocks(blocks)
        .with_seed(config.seed)
        .with_fix_com(config.fix_com != 0)
        .with_retain(retain)
        .with_target_error(target_error)
        .with_max_rounds(max_rounds);
    parsed.check()?;
    Ok(parsed)
}

fn parse_frames(
    n_frames: i64,
    n_dof: i64,
    frames: *const f64,
    arcs: *const f64,
    n_arcs: i64,
) -> Result<(ArcPath, Vec<f64>), SaddleError> {
    if n_frames < 2 || n_dof < 1 || frames.is_null() {
        return Err(SaddleError::Shape("PAFI path shape".into()));
    }
    let rows =
        usize::try_from(n_frames).map_err(|_| SaddleError::Shape("PAFI frame rows".into()))?;
    let dof = usize::try_from(n_dof).map_err(|_| SaddleError::Shape("PAFI dof".into()))?;
    let total = rows
        .checked_mul(dof)
        .ok_or_else(|| SaddleError::Shape("PAFI path size overflow".into()))?;
    if total > isize::MAX as usize / size_of::<f64>() {
        return Err(SaddleError::Shape("PAFI path exceeds address space".into()));
    }
    let raw = unsafe { slice::from_raw_parts(frames, total) };
    let mut points = Vec::with_capacity(rows);
    for i in 0..rows {
        points.push(Array1::from_vec(raw[i * dof..(i + 1) * dof].to_vec()));
    }
    let path = ArcPath::from_positions(&points)?;
    if arcs.is_null() {
        if n_arcs != 0 {
            return Err(SaddleError::Shape(
                "PAFI null arcs with nonzero count".into(),
            ));
        }
        let arc = path.arcs();
        return Ok((path, arc));
    }
    if n_arcs == 0 {
        let arc = path.arcs();
        return Ok((path, arc));
    }
    if n_arcs < 2 {
        return Err(SaddleError::Invalid(
            "PAFI needs at least two arc points".into(),
        ));
    }
    let count = usize::try_from(n_arcs).map_err(|_| SaddleError::Shape("PAFI arcs".into()))?;
    if count > isize::MAX as usize / size_of::<f64>() {
        return Err(SaddleError::Shape("PAFI arcs exceed address space".into()));
    }
    let arc = unsafe { slice::from_raw_parts(arcs, count) }.to_vec();
    for (i, value) in arc.iter().enumerate() {
        if !value.is_finite() {
            return Err(SaddleError::NonFinite("PAFI arc"));
        }
        if i > 0 && *value <= arc[i - 1] {
            return Err(SaddleError::Invalid(
                "PAFI arcs must strictly increase".into(),
            ));
        }
    }
    Ok((path, arc))
}

/// # Safety
/// `config` and `frames` are required. `frames` holds `n_frames * n_dof`
/// doubles row-major. `arcs`, when non-null and `n_arcs > 0`, holds
/// `n_arcs` increasing arc values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_pafi_create(
    config: *const RgsaddlePafiConfig,
    n_frames: i64,
    n_dof: i64,
    frames: *const f64,
    arcs: *const f64,
    n_arcs: i64,
) -> *mut RgsaddlePafi {
    if config.is_null() {
        return std::ptr::null_mut();
    }
    let cfg = unsafe { &*config };
    let parsed_config = match parse_pafi_config(cfg) {
        Ok(config) => config,
        Err(_) => return std::ptr::null_mut(),
    };
    let (path, arc_values) = match parse_frames(n_frames, n_dof, frames, arcs, n_arcs) {
        Ok(value) => value,
        Err(_) => return std::ptr::null_mut(),
    };
    Box::into_raw(Box::new(RgsaddlePafi {
        config: parsed_config,
        path,
        arcs: arc_values,
        profile: None,
        planes: Vec::new(),
    }))
}

/// # Safety
/// `session` and `config` must be valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_pafi_set_config(
    session: *mut RgsaddlePafi,
    config: *const RgsaddlePafiConfig,
) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    if config.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let parsed = match parse_pafi_config(unsafe { &*config }) {
        Ok(config) => config,
        Err(err) => return status_of(&err),
    };
    let session = unsafe { &mut *session };
    session.config = parsed;
    session.profile = None;
    session.planes.clear();
    RGSADDLE_OK
}

/// # Safety
/// `session` must be valid; `surface` is called with `user`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_pafi_run(
    session: *mut RgsaddlePafi,
    surface: Option<RgsaddlePointSurfaceFn>,
    user: *mut c_void,
) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    let Some(callback) = surface else {
        return RGSADDLE_NULL_SURFACE;
    };
    let session = unsafe { &mut *session };
    let n_dof = session
        .path
        .geometry(session.arcs[0])
        .map_err(|err| status_of(&err))
        .map(|g| g.positions.len());
    let n_dof = match n_dof {
        Ok(value) => value,
        Err(code) => return code,
    };
    let c_surface = CPointSurface {
        callback,
        user,
        n_dof,
    };
    match pafi_profile(&c_surface, &session.path, &session.arcs, &session.config) {
        Ok((profile, planes)) => {
            session.profile = Some(profile);
            session.planes = planes;
            RGSADDLE_OK
        }
        Err(err) => status_of(&err),
    }
}

/// # Safety
/// `session` and `out` are valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_pafi_result(
    session: *const RgsaddlePafi,
    out: *mut RgsaddlePafiResult,
) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    if out.is_null() {
        return RGSADDLE_NULL_REPORT;
    }
    let session = unsafe { &*session };
    let Some(profile) = session.profile.as_ref() else {
        return RGSADDLE_NO_EVALUATION;
    };
    let evaluations: i64 = session
        .planes
        .iter()
        .map(|plane| i64::try_from(plane.evaluations))
        .collect::<Result<Vec<_>, _>>()
        .map_or(i64::MAX, |values| values.into_iter().sum());
    let out = unsafe { &mut *out };
    stamp(&mut out.version);
    out.flags = 0;
    out.barrier = profile.barrier;
    out.barrier_error = profile.barrier_error;
    out.dividing = as_i64_saturated(profile.dividing);
    out.samples = as_i64_saturated(profile.samples);
    out.evaluations = evaluations;
    RGSADDLE_OK
}

/// # Safety
/// Every non-null array pointer must hold at least `capacity` doubles.
/// `written` may be null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_pafi_profile(
    session: *const RgsaddlePafi,
    arcs: *mut f64,
    gradients: *mut f64,
    gradient_errors: *mut f64,
    free_energy: *mut f64,
    free_energy_errors: *mut f64,
    capacity: i64,
    written: *mut i64,
) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    let session = unsafe { &*session };
    let Some(profile) = session.profile.as_ref() else {
        return RGSADDLE_NO_EVALUATION;
    };
    let n = profile.arc.len();
    let Ok(cap) = usize::try_from(capacity) else {
        return RGSADDLE_SHAPE;
    };
    if !written.is_null() {
        unsafe {
            *written = as_i64_saturated(n);
        }
    }
    if cap < n {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let copy = |ptr: *mut f64, src: &[f64]| {
        if !ptr.is_null() {
            let out = unsafe { slice::from_raw_parts_mut(ptr, src.len()) };
            out.copy_from_slice(src);
        }
    };
    copy(arcs, &profile.arc);
    copy(gradients, &profile.gradient);
    copy(gradient_errors, &profile.gradient_error);
    copy(free_energy, &profile.free_energy);
    copy(free_energy_errors, &profile.free_energy_error);
    RGSADDLE_OK
}

/// # Safety
/// `session` must come from `rgsaddle_pafi_create` and be freed once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_pafi_free(session: *mut RgsaddlePafi) {
    if !session.is_null() {
        drop(unsafe { Box::from_raw(session) });
    }
}

/// OH-TST configuration over the C ABI.
#[repr(C)]
pub struct RgsaddleOhtstConfig {
    /// ABI version this value was built against.
    pub version: RgsaddleVersion,
    /// Bit flags. Reserved; write zero.
    pub flags: u64,
    /// Thermal energy `kT` in the surface's energy unit.
    pub thermal: f64,
    /// Hyperplane sampler time step.
    pub sample_time_step: f64,
    /// Hyperplane sampler Langevin friction.
    pub sample_friction: f64,
    /// Hyperplane sampler burn-in steps.
    pub sample_burn_in: i64,
    /// Steps between hyperplane sampler records.
    pub sample_stride: i64,
    /// Hyperplane sampler records.
    pub sample_records: i64,
    /// Contiguous blocks for hyperplane sampler error estimation.
    pub sample_blocks: i64,
    /// Random seed for hyperplane sampling.
    pub sample_seed: u64,
    /// Number of plane configurations retained for shooting starts.
    pub sample_retain: i64,
    /// Hyperplane integrator: 0 BAOAB, 1 GHMC.
    pub sample_integrator: i32,
    /// Shooting trajectory Langevin friction (`>= 0`).
    pub shooting_friction: f64,
    /// Shooting trajectory integration time step.
    pub shooting_time_step: f64,
    /// Steps per forward and backward half-trajectory.
    pub shooting_steps: i64,
    /// Steps between transmission checkpoints.
    pub shooting_check_every: i64,
    /// Consecutive checkpoint commitment threshold, or 0 to run to the end.
    pub shooting_commit: i64,
    /// Total shooting trajectories.
    pub shooting_shots: i64,
    /// Contiguous blocks for shooting error estimation.
    pub shooting_blocks: i64,
    /// Random seed for shooting velocities and noise.
    pub shooting_seed: u64,
    /// Crossing counting rule: 0 final state, 1 first crossing.
    pub counting: i32,
}

/// OH-TST transmission summary returned after `rgsaddle_ohtst_run`.
#[repr(C)]
pub struct RgsaddleOhtstResult {
    /// ABI version stamped by the library.
    pub version: RgsaddleVersion,
    /// Bit flags. Zeroed by the library.
    pub flags: u64,
    /// Transmission coefficient `kappa` at the end of the halves.
    pub kappa: f64,
    /// Block standard error of `kappa`.
    pub error: f64,
    /// Shots whose halves both reached a state.
    pub shots: i64,
    /// Shots that ended unassigned.
    pub unassigned: i64,
    /// Shots counted as reactive.
    pub reactive: i64,
    /// Backward halves that recrossed the dividing plane.
    pub recrossed: i64,
    /// Surface evaluations spent during shooting.
    pub evaluations: i64,
}

/// Opaque OH-TST session created by [`rgsaddle_ohtst_create`].
pub struct RgsaddleOhtst {
    sampler: LangevinConfig,
    shooting: TransmissionConfig,
    plane: Hyperplane,
    transmission: Option<TransmissionCoefficient>,
}

fn parse_ohtst_config(
    config: &RgsaddleOhtstConfig,
    n_dof: usize,
) -> Result<(LangevinConfig, TransmissionConfig), SaddleError> {
    if config.version.major != RGSADDLE_ABI_MAJOR {
        return Err(SaddleError::Invalid("OH-TST ABI major mismatch".into()));
    }
    let to_usize = |value: i64, name: &str| -> Result<usize, SaddleError> {
        usize::try_from(value).map_err(|_| SaddleError::Invalid(format!("OH-TST {name}")))
    };
    let sample_integrator = match config.sample_integrator {
        0 => PlaneIntegrator::Baoab,
        1 => PlaneIntegrator::Ghmc,
        _ => {
            return Err(SaddleError::Invalid(
                "OH-TST sample integrator must be 0 (BAOAB) or 1 (GHMC)".into(),
            ));
        }
    };
    let counting = match config.counting {
        0 => Counting::Final,
        1 => Counting::FirstCrossing,
        _ => {
            return Err(SaddleError::Invalid(
                "OH-TST counting must be 0 (final) or 1 (first-crossing)".into(),
            ));
        }
    };
    let sample_retain = to_usize(config.sample_retain, "sample retain")?;
    if sample_retain == 0 {
        return Err(SaddleError::Invalid(
            "OH-TST sample retain must be positive".into(),
        ));
    }
    let sampler = LangevinConfig::default()
        .with_thermal(config.thermal)
        .with_time_step(config.sample_time_step)
        .with_friction(config.sample_friction)
        .with_burn_in(to_usize(config.sample_burn_in, "sample burn-in")?)
        .with_stride(to_usize(config.sample_stride, "sample stride")?)
        .with_records(to_usize(config.sample_records, "sample records")?)
        .with_blocks(to_usize(config.sample_blocks, "sample blocks")?)
        .with_seed(config.sample_seed)
        .with_retain(sample_retain)
        .with_integrator(sample_integrator);
    sampler.check()?;
    let shooting = TransmissionConfig::default()
        .with_thermal(config.thermal)
        .with_friction(config.shooting_friction)
        .with_time_step(config.shooting_time_step)
        .with_steps(to_usize(config.shooting_steps, "shooting steps")?)
        .with_check_every(to_usize(
            config.shooting_check_every,
            "shooting check interval",
        )?)
        .with_commit(to_usize(config.shooting_commit, "shooting commit")?)
        .with_shots(to_usize(config.shooting_shots, "shooting shots")?)
        .with_blocks(to_usize(config.shooting_blocks, "shooting blocks")?)
        .with_seed(config.shooting_seed)
        .with_counting(counting);
    shooting.check(n_dof)?;
    Ok((sampler, shooting))
}

/// # Safety
/// `config`, `point` and `normal` are required. `point` and `normal`
/// each hold `n_dof` doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_ohtst_create(
    config: *const RgsaddleOhtstConfig,
    n_dof: i64,
    point: *const f64,
    normal: *const f64,
) -> *mut RgsaddleOhtst {
    if config.is_null() || point.is_null() || normal.is_null() || n_dof < 1 {
        return std::ptr::null_mut();
    }
    let n = match usize::try_from(n_dof) {
        Ok(value) => value,
        Err(_) => return std::ptr::null_mut(),
    };
    if n > isize::MAX as usize / size_of::<f64>() {
        return std::ptr::null_mut();
    }
    let cfg = unsafe { &*config };
    let (sampler, shooting) = match parse_ohtst_config(cfg, n) {
        Ok(value) => value,
        Err(_) => return std::ptr::null_mut(),
    };
    let point = unsafe { slice::from_raw_parts(point, n) };
    let normal = unsafe { slice::from_raw_parts(normal, n) };
    let plane = match Hyperplane::new(
        Array1::from_vec(point.to_vec()),
        Array1::from_vec(normal.to_vec()),
    ) {
        Ok(plane) => plane,
        Err(_) => return std::ptr::null_mut(),
    };
    Box::into_raw(Box::new(RgsaddleOhtst {
        sampler,
        shooting,
        plane,
        transmission: None,
    }))
}

/// # Safety
/// `session` and `config` must be valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_ohtst_set_config(
    session: *mut RgsaddleOhtst,
    config: *const RgsaddleOhtstConfig,
) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    if config.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let session = unsafe { &mut *session };
    let (sampler, shooting) = match parse_ohtst_config(unsafe { &*config }, session.plane.dim()) {
        Ok(value) => value,
        Err(err) => return status_of(&err),
    };
    session.sampler = sampler;
    session.shooting = shooting;
    session.transmission = None;
    RGSADDLE_OK
}

/// # Safety
/// `session` must be valid; `surface` is called with `user`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_ohtst_run(
    session: *mut RgsaddleOhtst,
    surface: Option<RgsaddlePointSurfaceFn>,
    user: *mut c_void,
) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    let Some(callback) = surface else {
        return RGSADDLE_NULL_SURFACE;
    };
    let session = unsafe { &mut *session };
    let n_dof = session.plane.dim();
    let c_surface = CPointSurface {
        callback,
        user,
        n_dof,
    };
    let sample = match sample_plane(
        &c_surface,
        &session.plane,
        Some(session.plane.point()),
        &session.sampler,
    ) {
        Ok(sample) => sample,
        Err(err) => return status_of(&err),
    };
    let mut dynamics = AnalyticSurfaceDynamics::new(&c_surface);
    match transmission_coefficient(
        &mut dynamics,
        &session.plane,
        &sample.retained,
        &session.shooting,
    ) {
        Ok(transmission) => {
            session.transmission = Some(transmission);
            RGSADDLE_OK
        }
        Err(err) => status_of(&err),
    }
}

/// # Safety
/// `session` and `out` are valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_ohtst_result(
    session: *const RgsaddleOhtst,
    out: *mut RgsaddleOhtstResult,
) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    if out.is_null() {
        return RGSADDLE_NULL_REPORT;
    }
    let session = unsafe { &*session };
    let Some(transmission) = session.transmission.as_ref() else {
        return RGSADDLE_NO_EVALUATION;
    };
    let out = unsafe { &mut *out };
    stamp(&mut out.version);
    out.flags = 0;
    out.kappa = transmission.kappa;
    out.error = transmission.error;
    out.shots = as_i64_saturated(transmission.shots);
    out.unassigned = as_i64_saturated(transmission.unassigned);
    out.reactive = as_i64_saturated(transmission.reactive);
    out.recrossed = as_i64_saturated(transmission.recrossed);
    out.evaluations = as_i64_saturated(transmission.evaluations);
    RGSADDLE_OK
}

/// # Safety
/// `times` and `kappa_of_t`, when non-null, hold at least `capacity`
/// doubles. `written` may be null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_ohtst_curve(
    session: *const RgsaddleOhtst,
    times: *mut f64,
    kappa_of_t: *mut f64,
    capacity: i64,
    written: *mut i64,
) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    let session = unsafe { &*session };
    let Some(transmission) = session.transmission.as_ref() else {
        return RGSADDLE_NO_EVALUATION;
    };
    let Ok(cap) = usize::try_from(capacity) else {
        return RGSADDLE_SHAPE;
    };
    let n = transmission.time.len();
    if !written.is_null() {
        unsafe {
            *written = as_i64_saturated(n);
        }
    }
    if cap < n {
        return RGSADDLE_INVALID_PARAMETER;
    }
    if !times.is_null() {
        let out = unsafe { slice::from_raw_parts_mut(times, n) };
        out.copy_from_slice(&transmission.time);
    }
    if !kappa_of_t.is_null() {
        let out = unsafe { slice::from_raw_parts_mut(kappa_of_t, n) };
        out.copy_from_slice(&transmission.kappa_of_t);
    }
    RGSADDLE_OK
}

/// # Safety
/// `session` must come from `rgsaddle_ohtst_create` and be freed once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_ohtst_free(session: *mut RgsaddleOhtst) {
    if !session.is_null() {
        drop(unsafe { Box::from_raw(session) });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    extern "C" fn pafi_along_x(
        _user: *mut c_void,
        n_dof: i64,
        x: *const f64,
        energy: *mut f64,
        gradient: *mut f64,
    ) -> i32 {
        if n_dof < 1 || x.is_null() || energy.is_null() || gradient.is_null() {
            return RGSADDLE_INVALID_PARAMETER;
        }
        let point = unsafe { slice::from_raw_parts(x, n_dof as usize) };
        let grad = unsafe { slice::from_raw_parts_mut(gradient, n_dof as usize) };
        let coordinate = point[0];
        unsafe {
            *energy = (1.0 - coordinate * coordinate).powi(2);
        }
        grad.fill(0.0);
        grad[0] = -4.0 * coordinate * (1.0 - coordinate * coordinate);
        RGSADDLE_OK
    }

    extern "C" fn ohtst_quadratic(
        _user: *mut c_void,
        n_dof: i64,
        x: *const f64,
        energy: *mut f64,
        gradient: *mut f64,
    ) -> i32 {
        if n_dof != 2 || x.is_null() || energy.is_null() || gradient.is_null() {
            return RGSADDLE_INVALID_PARAMETER;
        }
        let point = unsafe { slice::from_raw_parts(x, 2) };
        let grad = unsafe { slice::from_raw_parts_mut(gradient, 2) };
        unsafe {
            *energy = 0.5 * (-point[0] * point[0] + 4.0 * point[1] * point[1]);
        }
        grad[0] = -point[0];
        grad[1] = 4.0 * point[1];
        RGSADDLE_OK
    }

    #[test]
    fn pafi_symbols_run_and_fill_outputs() {
        let mut config = RgsaddlePafiConfig {
            version: RgsaddleVersion {
                major: RGSADDLE_ABI_MAJOR,
                minor: RGSADDLE_ABI_MINOR,
            },
            flags: 0,
            thermal: 0.02,
            mobility: 1.0,
            time_step: 0.01,
            thermalize: 5,
            stride: 1,
            samples: 8,
            blocks: 2,
            seed: 3,
            fix_com: 0,
            retain: 1,
            target_error: 0.0,
            max_rounds: 1,
        };
        let mut frames = Vec::new();
        let n_frames = 41usize;
        for i in 0..n_frames {
            let x = -1.0 + 2.0 * i as f64 / (n_frames - 1) as f64;
            frames.push(x);
        }
        let mut arcs = Vec::with_capacity(n_frames);
        for i in 0..n_frames {
            arcs.push(i as f64 / (n_frames - 1) as f64 * 2.0);
        }
        let session = unsafe {
            rgsaddle_pafi_create(
                &config,
                n_frames as i64,
                1,
                frames.as_ptr(),
                arcs.as_ptr(),
                arcs.len() as i64,
            )
        };
        assert!(!session.is_null());

        let mut result = RgsaddlePafiResult {
            version: RgsaddleVersion { major: 0, minor: 0 },
            flags: 0,
            barrier: 0.0,
            barrier_error: 0.0,
            dividing: -1,
            samples: 0,
            evaluations: 0,
        };
        assert_eq!(
            unsafe { rgsaddle_pafi_result(session, &mut result) },
            RGSADDLE_NO_EVALUATION
        );

        config.samples = 200;
        assert_eq!(
            unsafe { rgsaddle_pafi_set_config(session, &config) },
            RGSADDLE_OK
        );
        assert_eq!(
            unsafe { rgsaddle_pafi_run(session, Some(pafi_along_x), std::ptr::null_mut()) },
            RGSADDLE_OK
        );
        assert_eq!(
            unsafe { rgsaddle_pafi_result(session, &mut result) },
            RGSADDLE_OK
        );
        let miss = (result.barrier - 1.0).abs();
        let floor = result.barrier_error.max(5.0e-3);
        assert!(
            miss < 2.0e-2 && miss < 4.0 * floor,
            "barrier {} miss {} err {}",
            result.barrier,
            miss,
            result.barrier_error
        );
        assert_eq!(result.dividing, 20);
        assert_eq!(result.version.major, RGSADDLE_ABI_MAJOR);

        let mut queried = 0i64;
        assert_eq!(
            unsafe {
                rgsaddle_pafi_profile(
                    session,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    0,
                    &mut queried,
                )
            },
            RGSADDLE_INVALID_PARAMETER
        );
        assert_eq!(queried, n_frames as i64);

        let mut out_arcs = vec![0.0; n_frames];
        let mut out_grad = vec![0.0; n_frames];
        let mut out_grad_err = vec![0.0; n_frames];
        let mut out_free = vec![0.0; n_frames];
        let mut out_free_err = vec![0.0; n_frames];
        let mut written = 0i64;
        assert_eq!(
            unsafe {
                rgsaddle_pafi_profile(
                    session,
                    out_arcs.as_mut_ptr(),
                    out_grad.as_mut_ptr(),
                    out_grad_err.as_mut_ptr(),
                    out_free.as_mut_ptr(),
                    out_free_err.as_mut_ptr(),
                    n_frames as i64,
                    &mut written,
                )
            },
            RGSADDLE_OK
        );
        assert_eq!(written, n_frames as i64);
        assert!((out_free[20] - 0.9975).abs() < 1e-12);
        assert!(out_free[n_frames - 1].abs() < 1e-12);
        unsafe { rgsaddle_pafi_free(session) };
    }

    #[test]
    fn ohtst_symbols_run_and_fill_outputs() {
        let config = RgsaddleOhtstConfig {
            version: RgsaddleVersion {
                major: RGSADDLE_ABI_MAJOR,
                minor: RGSADDLE_ABI_MINOR,
            },
            flags: 0,
            thermal: 0.5,
            sample_time_step: 0.1,
            sample_friction: 1.0,
            sample_burn_in: 300,
            sample_stride: 5,
            sample_records: 1000,
            sample_blocks: 16,
            sample_seed: 9,
            sample_retain: 512,
            sample_integrator: 0,
            shooting_friction: 1.5,
            shooting_time_step: 0.05,
            shooting_steps: 400,
            shooting_check_every: 40,
            shooting_commit: 0,
            shooting_shots: 4096,
            shooting_blocks: 32,
            shooting_seed: 17,
            counting: 0,
        };
        let point = [0.0, 0.0];
        let normal = [1.0, 0.0];
        let session = unsafe { rgsaddle_ohtst_create(&config, 2, point.as_ptr(), normal.as_ptr()) };
        assert!(!session.is_null());
        assert_eq!(
            unsafe { rgsaddle_ohtst_run(session, Some(ohtst_quadratic), std::ptr::null_mut()) },
            RGSADDLE_OK
        );
        let mut result = RgsaddleOhtstResult {
            version: RgsaddleVersion { major: 0, minor: 0 },
            flags: 0,
            kappa: 0.0,
            error: 0.0,
            shots: 0,
            unassigned: 0,
            reactive: 0,
            recrossed: 0,
            evaluations: 0,
        };
        assert_eq!(
            unsafe { rgsaddle_ohtst_result(session, &mut result) },
            RGSADDLE_OK
        );
        let kramers = (1.0 + (1.5_f64 / 2.0).powi(2)).sqrt() - 1.5_f64 / 2.0;
        assert!((result.kappa - kramers).abs() < 4.0 * result.error + f64::EPSILON);
        assert_eq!(result.version.major, RGSADDLE_ABI_MAJOR);

        let mut curve_t = vec![0.0; 16];
        let mut curve_kappa = vec![0.0; 16];
        let mut written = 0i64;
        assert_eq!(
            unsafe {
                rgsaddle_ohtst_curve(
                    session,
                    curve_t.as_mut_ptr(),
                    curve_kappa.as_mut_ptr(),
                    16,
                    &mut written,
                )
            },
            RGSADDLE_OK
        );
        assert!(written > 0);
        let final_index = usize::try_from(written - 1).expect("curve length");
        assert!(
            (curve_kappa[final_index] - kramers).abs() < 4.0 * result.error + f64::EPSILON,
            "kappa(t_end)={} expected={kramers} error={}",
            curve_kappa[final_index],
            result.error
        );

        let tilted = [25.0_f64.to_radians().cos(), 25.0_f64.to_radians().sin()];
        let tilt_session =
            unsafe { rgsaddle_ohtst_create(&config, 2, point.as_ptr(), tilted.as_ptr()) };
        assert!(!tilt_session.is_null());
        assert_eq!(
            unsafe {
                rgsaddle_ohtst_run(tilt_session, Some(ohtst_quadratic), std::ptr::null_mut())
            },
            RGSADDLE_OK
        );
        let mut tilted_result = RgsaddleOhtstResult {
            version: RgsaddleVersion { major: 0, minor: 0 },
            flags: 0,
            kappa: 0.0,
            error: 0.0,
            shots: 0,
            unassigned: 0,
            reactive: 0,
            recrossed: 0,
            evaluations: 0,
        };
        assert_eq!(
            unsafe { rgsaddle_ohtst_result(tilt_session, &mut tilted_result) },
            RGSADDLE_OK
        );
        let theta = 25.0_f64 * PI / 180.0;
        let geometric = (theta.cos().powi(2) - 0.25 * theta.sin().powi(2)).sqrt();
        let expected_tilted = kramers * geometric;
        assert!(
            (tilted_result.kappa - expected_tilted).abs()
                < 4.0 * tilted_result.error + f64::EPSILON
        );
        unsafe { rgsaddle_ohtst_free(tilt_session) };
        unsafe { rgsaddle_ohtst_free(session) };
    }
}
