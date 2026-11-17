//! C interface for the shared basin constrained dimer force.
use crate::{
    SaddleError,
    capi::*,
    kappa::{KappaDimerConfig, kappa_dimer_force},
    minmode::PointSurface,
    ohtst::{
        AnalyticSurfaceDynamics, Counting, Hyperplane, TransmissionConfig, transmission_coefficient,
    },
};
use ndarray::{Array1, ArrayView1, ArrayView2};
use rgmin::{EigenParams, EigensolverKind};
use std::{ffi::c_void, slice};

#[repr(C)]
#[derive(Clone, Copy)]
/// The RgsaddleKappaConfig.
pub struct RgsaddleKappaConfig {
    /// ABI version this value was built against.
    pub version: RgsaddleVersion,
    /// The beta.
    pub beta: f64,
    /// The tolerance.
    pub tolerance: f64,
    /// The max iterations.
    pub max_iterations: i64,
    /// The krylov dimension.
    pub krylov_dimension: i64,
    /// The eigen kind.
    pub eigen_kind: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
/// The RgsaddleKappaReport.
pub struct RgsaddleKappaReport {
    /// ABI version this value was built against.
    pub version: RgsaddleVersion,
    /// The kappa.
    pub kappa: f64,
    /// The tangent curvature.
    pub tangent_curvature: f64,
    /// The gamma parallel.
    pub gamma_parallel: f64,
    /// The gamma perpendicular.
    pub gamma_perpendicular: f64,
    /// The residual.
    pub residual: f64,
    /// The hessian actions.
    pub hessian_actions: i64,
    /// The tangent dimension.
    pub tangent_dimension: i64,
}

/// The RgsaddleHessianFn.
pub type RgsaddleHessianFn = extern "C" fn(*mut c_void, i64, *const f64, *mut f64) -> i32;

/// The point energy-and-gradient callback used by PAFI transmission.
pub type RgsaddlePointSurfaceFn =
    extern "C" fn(*mut c_void, i64, *const f64, *mut f64, *mut f64) -> i32;

#[repr(C)]
#[derive(Clone, Copy)]
/// Settings for forward/backward shooting from a PAFI dividing plane.
pub struct RgsaddlePafiTransmissionConfig {
    /// ABI version this value was built against.
    pub version: RgsaddleVersion,
    /// `kT` in the surface energy unit.
    pub thermal: f64,
    /// Langevin friction; zero is Newtonian.
    pub friction: f64,
    /// BAOAB time step.
    pub time_step: f64,
    /// Steps in each half trajectory.
    pub steps: i64,
    /// Steps between state checks.
    pub check_every: i64,
    /// Equal-state checks needed for commitment; zero runs every step.
    pub commit: i64,
    /// Number of shots.
    pub shots: i64,
    /// Most blocks used for the standard error.
    pub blocks: i64,
    /// Random seed.
    pub seed: u64,
    /// Zero for Bennett–Chandler final-state counting, one for first crossing.
    pub counting: i32,
    /// Whether to hold modes returned by the surface. C callbacks return none.
    pub hold_excluded: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
/// PAFI transmission estimate and trajectory counts.
pub struct RgsaddlePafiTransmissionReport {
    /// ABI version this value was built against.
    pub version: RgsaddleVersion,
    /// Transmission coefficient.
    pub kappa: f64,
    /// Block standard error.
    pub error: f64,
    /// Maximum-free-energy plane selected from the band.
    pub dividing: i64,
    /// Assigned shots entering the estimate.
    pub shots: i64,
    /// Shots whose half trajectory was not assigned.
    pub unassigned: i64,
    /// Reactive shots.
    pub reactive: i64,
    /// Backward halves that recrossed.
    pub recrossed: i64,
    /// Surface evaluations.
    pub evaluations: i64,
}

struct CPointSurface {
    callback: RgsaddlePointSurfaceFn,
    user: *mut c_void,
}

impl PointSurface for CPointSurface {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let mut energy = 0.0;
        let mut gradient = Array1::zeros(x.len());
        let status = (self.callback)(
            self.user,
            x.len() as i64,
            x.as_ptr(),
            &mut energy,
            gradient.as_mut_ptr(),
        );
        if status != 0 {
            return Err(SaddleError::Surface(format!(
                "point-surface callback status {status}"
            )));
        }
        Ok((energy, gradient))
    }
}

unsafe impl Sync for CPointSurface {}

/// Fill default PAFI transmission settings.
///
/// # Safety
/// `config` points to one writable configuration.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_pafi_transmission_config_default(
    config: *mut RgsaddlePafiTransmissionConfig,
) -> i32 {
    if config.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let default = TransmissionConfig::default().with_counting(Counting::Final);
    unsafe {
        *config = RgsaddlePafiTransmissionConfig {
            version: RgsaddleVersion {
                major: RGSADDLE_ABI_MAJOR,
                minor: RGSADDLE_ABI_MINOR,
            },
            thermal: default.thermal,
            friction: default.friction,
            time_step: default.time_step,
            steps: default.steps as i64,
            check_every: default.check_every as i64,
            commit: default.commit as i64,
            shots: default.shots as i64,
            blocks: default.blocks as i64,
            seed: default.seed,
            counting: 0,
            hold_excluded: i32::from(default.hold_excluded),
        };
    }
    RGSADDLE_OK
}

/// Select the maximum-free-energy PAFI plane and shoot its retained samples.
///
/// # Safety
/// `free_energy` has `n_planes` values; `points` and `normals` have
/// `n_planes * n_dof`; `start_offsets` has `n_planes + 1` monotonically
/// increasing entries beginning at zero; `starts` has
/// `start_offsets[n_planes] * n_dof` values. `masses` is null for unit
/// masses or has `n_dof` values. The callback writes one energy and
/// `n_dof` gradient values and must not unwind.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn rgsaddle_pafi_transmission(
    config: *const RgsaddlePafiTransmissionConfig,
    n_dof: i64,
    n_planes: i64,
    free_energy: *const f64,
    points: *const f64,
    normals: *const f64,
    start_offsets: *const i64,
    starts: *const f64,
    masses: *const f64,
    surface: Option<RgsaddlePointSurfaceFn>,
    user: *mut c_void,
    report: *mut RgsaddlePafiTransmissionReport,
) -> i32 {
    if config.is_null()
        || free_energy.is_null()
        || points.is_null()
        || normals.is_null()
        || start_offsets.is_null()
        || starts.is_null()
        || report.is_null()
        || n_dof <= 0
        || n_planes <= 0
    {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let Some(surface) = surface else {
        return RGSADDLE_NULL_SURFACE;
    };
    let cfg = unsafe { &*config };
    if cfg.version.major != RGSADDLE_ABI_MAJOR
        || cfg.steps <= 0
        || cfg.check_every <= 0
        || cfg.commit < 0
        || cfg.shots <= 0
        || cfg.blocks < 2
        || !matches!(cfg.counting, 0 | 1)
        || !matches!(cfg.hold_excluded, 0 | 1)
    {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let (n, planes) = (n_dof as usize, n_planes as usize);
    let Some(packed) = planes.checked_mul(n) else {
        return RGSADDLE_SHAPE;
    };
    let free = unsafe { slice::from_raw_parts(free_energy, planes) };
    if !free.iter().all(|v| v.is_finite()) {
        return RGSADDLE_NON_FINITE;
    }
    let mut dividing = 0;
    for i in 1..planes {
        if free[i] > free[dividing] {
            dividing = i;
        }
    }
    let offsets = unsafe { slice::from_raw_parts(start_offsets, planes + 1) };
    if offsets[0] != 0
        || offsets.iter().any(|&v| v < 0)
        || offsets.windows(2).any(|pair| pair[1] < pair[0])
    {
        return RGSADDLE_SHAPE;
    }
    let (first, last) = (offsets[dividing] as usize, offsets[dividing + 1] as usize);
    if first == last {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let Some(total) = (offsets[planes] as usize).checked_mul(n) else {
        return RGSADDLE_SHAPE;
    };
    if packed > isize::MAX as usize / size_of::<f64>()
        || total > isize::MAX as usize / size_of::<f64>()
    {
        return RGSADDLE_SHAPE;
    }
    let point_data = unsafe { slice::from_raw_parts(points, packed) };
    let normal_data = unsafe { slice::from_raw_parts(normals, packed) };
    let start_data = unsafe { slice::from_raw_parts(starts, total) };
    let plane = match Hyperplane::new(
        Array1::from_vec(point_data[dividing * n..(dividing + 1) * n].to_vec()),
        Array1::from_vec(normal_data[dividing * n..(dividing + 1) * n].to_vec()),
    ) {
        Ok(plane) => plane,
        Err(error) => return transmission_status(error),
    };
    let retained: Vec<Array1<f64>> = (first..last)
        .map(|i| Array1::from_vec(start_data[i * n..(i + 1) * n].to_vec()))
        .collect();
    let mut shooting = TransmissionConfig::default()
        .with_thermal(cfg.thermal)
        .with_friction(cfg.friction)
        .with_time_step(cfg.time_step)
        .with_steps(cfg.steps as usize)
        .with_check_every(cfg.check_every as usize)
        .with_commit(cfg.commit as usize)
        .with_shots(cfg.shots as usize)
        .with_blocks(cfg.blocks as usize)
        .with_seed(cfg.seed)
        .with_counting(if cfg.counting == 0 {
            Counting::Final
        } else {
            Counting::FirstCrossing
        })
        .with_hold_excluded(cfg.hold_excluded != 0);
    if !masses.is_null() {
        shooting = shooting.with_masses(unsafe { slice::from_raw_parts(masses, n) }.to_vec());
    }
    let callback = CPointSurface {
        callback: surface,
        user,
    };
    let mut dynamics = AnalyticSurfaceDynamics::new(&callback);
    match transmission_coefficient(&mut dynamics, &plane, &retained, &shooting) {
        Ok(out) => {
            unsafe {
                *report = RgsaddlePafiTransmissionReport {
                    version: RgsaddleVersion {
                        major: RGSADDLE_ABI_MAJOR,
                        minor: RGSADDLE_ABI_MINOR,
                    },
                    kappa: out.kappa,
                    error: out.error,
                    dividing: dividing as i64,
                    shots: out.shots as i64,
                    unassigned: out.unassigned as i64,
                    reactive: out.reactive as i64,
                    recrossed: out.recrossed as i64,
                    evaluations: out.evaluations as i64,
                };
            }
            RGSADDLE_OK
        }
        Err(error) => transmission_status(error),
    }
}

fn transmission_status(error: SaddleError) -> i32 {
    status_of(&error)
}

/// # Safety
/// `config` points to one writable configuration.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_kappa_config_default(config: *mut RgsaddleKappaConfig) -> i32 {
    if config.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let default = KappaDimerConfig::default();
    unsafe {
        *config = RgsaddleKappaConfig {
            version: RgsaddleVersion {
                major: RGSADDLE_ABI_MAJOR,
                minor: RGSADDLE_ABI_MINOR,
            },
            beta: default.beta,
            tolerance: default.eigen.tol,
            max_iterations: default.eigen.max_iter as i64,
            krylov_dimension: default.eigen.krylov as i64,
            eigen_kind: default.eigen.kind as i32,
        };
    }
    RGSADDLE_OK
}

/// Assemble equations (6)--(7) without evaluating a potential.
///
/// # Safety
/// Gradient, mode, and output each hold `n_dof` doubles. Excluded directions
/// are `n_excluded` consecutive rows of `n_dof` doubles, or null for zero rows.
/// Input and output buffers do not overlap. The callback writes `n_dof`
/// Hessian-action values, returns zero on success, and must not unwind.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_kappa_dimer_force(
    config: *const RgsaddleKappaConfig,
    n_dof: i64,
    gradient: *const f64,
    mode: *const f64,
    n_excluded: i64,
    excluded: *const f64,
    hessian: Option<RgsaddleHessianFn>,
    user: *mut c_void,
    force: *mut f64,
    report: *mut RgsaddleKappaReport,
) -> i32 {
    if config.is_null()
        || gradient.is_null()
        || mode.is_null()
        || force.is_null()
        || report.is_null()
        || n_dof <= 0
        || n_excluded < 0
        || (n_excluded > 0 && excluded.is_null())
    {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let Some(hessian) = hessian else {
        return RGSADDLE_NULL_SURFACE;
    };
    let cfg = unsafe { &*config };
    if cfg.version.major != RGSADDLE_ABI_MAJOR {
        return RGSADDLE_ABI_MISMATCH;
    }
    if cfg.max_iterations < 0 || cfg.krylov_dimension < 0 || !(0..=255).contains(&cfg.eigen_kind) {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let Some(kind) = EigensolverKind::from_ordinal(cfg.eigen_kind as u8) else {
        return RGSADDLE_INVALID_PARAMETER;
    };
    let n = n_dof as usize;
    let Some(count) = (n_excluded as usize).checked_mul(n) else {
        return RGSADDLE_SHAPE;
    };
    if n > isize::MAX as usize / size_of::<f64>() || count > isize::MAX as usize / size_of::<f64>()
    {
        return RGSADDLE_SHAPE;
    }
    let g = ArrayView1::from(unsafe { slice::from_raw_parts(gradient, n) });
    let m = ArrayView1::from(unsafe { slice::from_raw_parts(mode, n) });
    let excluded_slice = if count == 0 {
        &[]
    } else {
        unsafe { slice::from_raw_parts(excluded, count) }
    };
    let excluded = ArrayView2::from_shape((n_excluded as usize, n), excluded_slice).unwrap();
    let cfg = KappaDimerConfig::default().with_beta(cfg.beta).with_eigen(
        EigenParams::default()
            .with_kind(kind)
            .with_nev(1)
            .with_krylov(cfg.krylov_dimension as usize)
            .with_max_iter(cfg.max_iterations as usize)
            .with_tol(cfg.tolerance),
    );
    let result = kappa_dimer_force(
        g,
        m,
        excluded,
        |v| {
            let mut out = Array1::zeros(n);
            let status = hessian(user, n_dof, v.as_ptr(), out.as_mut_ptr());
            if status != 0 {
                return Err(SaddleError::Surface(format!(
                    "Hessian callback status {status}"
                )));
            }
            Ok(out)
        },
        &cfg,
    );
    match result {
        Ok(out) => {
            unsafe { slice::from_raw_parts_mut(force, n) }
                .copy_from_slice(out.force.as_slice().unwrap());
            unsafe {
                *report = RgsaddleKappaReport {
                    version: RgsaddleVersion {
                        major: RGSADDLE_ABI_MAJOR,
                        minor: RGSADDLE_ABI_MINOR,
                    },
                    kappa: out.kappa,
                    tangent_curvature: out.tangent_curvature,
                    gamma_parallel: out.gamma_parallel,
                    gamma_perpendicular: out.gamma_perpendicular,
                    residual: out.residual,
                    hessian_actions: out.actions as i64,
                    tangent_dimension: out.tangent_dimension as i64,
                };
            }
            RGSADDLE_OK
        }
        Err(SaddleError::Shape(_)) => RGSADDLE_SHAPE,
        Err(SaddleError::NonFinite(_)) => RGSADDLE_NON_FINITE,
        Err(SaddleError::Surface(_)) => RGSADDLE_SURFACE_FAILED,
        Err(SaddleError::Invalid(_)) => RGSADDLE_INVALID_PARAMETER,
        Err(SaddleError::Stalled(_)) => crate::capi::RGSADDLE_STALLED,
        Err(SaddleError::NotEquilibrated { .. }) => crate::capi::RGSADDLE_NOT_EQUILIBRATED,
        Err(SaddleError::InsufficientEffectiveSamples { .. }) => {
            crate::capi::RGSADDLE_INSUFFICIENT_EFFECTIVE_SAMPLES
        }
        Err(SaddleError::Solver(message)) => {
            eprintln!("kappa eigensolver: {message}");
            RGSADDLE_SOLVER
        }
    }
}
