//! Callback C ABI for engine-owned hyperplane sampling.

use std::ffi::c_void;
use std::slice;

use ndarray::Array1;

use crate::SaddleError;
use crate::capi::{
    RGSADDLE_ABI_MISMATCH, RGSADDLE_INVALID_PARAMETER, RGSADDLE_NON_FINITE, RGSADDLE_NULL_REPORT,
    RGSADDLE_NULL_SESSION, RGSADDLE_NULL_SURFACE, RGSADDLE_OK, RgsaddleVersion, status_of,
};
use crate::ohtst::{
    THERMODYNAMIC_DERIVATIVE, ThermodynamicIntegrationConfig, thermodynamic_integration,
};
use crate::projector::{
    Hyperplane, HyperplaneEnsemble, HyperplaneObservation, HyperplaneSampleBudget,
    HyperplaneSampler,
};

const HYPERPLANE_TI_ABI_MINOR: u32 = 42;

/// One callback sampling request.
#[repr(C)]
pub struct RgsaddleHyperplaneBudget {
    /// ABI version.
    pub version: RgsaddleVersion,
    /// Reserved flags.
    pub flags: u64,
    /// Thermal energy.
    pub thermal: f64,
    /// Deterministic seed.
    pub seed: u64,
    /// Minimum warm-up steps.
    pub thermalize: i64,
    /// Steps between records.
    pub stride: i64,
    /// Requested records.
    pub samples: i64,
    /// Lambda, or NaN outside TI.
    pub lambda: f64,
    /// Whether positions are requested.
    pub positions: i32,
}

/// One borrowed callback observation.
#[repr(C)]
pub struct RgsaddleHyperplaneObservation {
    /// Constraint force along the positive unit normal.
    pub constraint_force: f64,
    /// Position of `dimension` entries, or null when not requested.
    pub position: *const f64,
    /// Projected force of `dimension` entries, or null when unavailable.
    pub projected_force: *const f64,
    /// `V_full - V_harmonic`, or NaN when not requested.
    pub thermodynamic_derivative: f64,
}

/// Emit one observation. Borrowed arrays need only live through this call.
pub type RgsaddleHyperplaneEmitFn =
    extern "C" fn(*mut c_void, *const RgsaddleHyperplaneObservation) -> i32;

/// Engine callback for one complete constrained ensemble.
pub type RgsaddleSampleHyperplaneFn = extern "C" fn(
    *mut c_void,
    *const f64,
    *const f64,
    usize,
    *const RgsaddleHyperplaneBudget,
    RgsaddleHyperplaneEmitFn,
    *mut c_void,
) -> i32;

/// Opaque callback sampler.
pub struct RgsaddleHyperplaneSampler {
    context: *mut c_void,
    sample: RgsaddleSampleHyperplaneFn,
}

/// Fixed-plane TI settings.
#[repr(C)]
pub struct RgsaddleHyperplaneTiConfig {
    /// ABI version.
    pub version: RgsaddleVersion,
    /// Reserved flags.
    pub flags: u64,
    /// Thermal energy.
    pub thermal: f64,
    /// Gauss--Legendre node count.
    pub nodes: i64,
    /// Minimum warm-up at every node.
    pub thermalize: i64,
    /// Steps between records.
    pub stride: i64,
    /// Records at every node.
    pub samples: i64,
    /// Blocks at every node.
    pub blocks: i64,
    /// First node seed.
    pub seed: u64,
}

/// Fixed-plane TI result.
#[repr(C)]
pub struct RgsaddleHyperplaneTiResult {
    /// `F_full - F_harmonic`.
    pub free_energy: f64,
    /// Propagated block standard error.
    pub error: f64,
    /// Engine evaluations when supplied; callback samplers report records.
    pub evaluations: i64,
}

struct Emitted {
    dimension: usize,
    observations: Vec<HyperplaneObservation>,
    derivative_required: bool,
    error: Option<SaddleError>,
}

extern "C" fn emit_observation(
    context: *mut c_void,
    observation: *const RgsaddleHyperplaneObservation,
) -> i32 {
    if context.is_null() || observation.is_null() {
        return RGSADDLE_NULL_REPORT;
    }
    // SAFETY: both pointers are valid for the callback invocation by contract.
    let emitted = unsafe { &mut *context.cast::<Emitted>() };
    // SAFETY: checked non-null above and borrowed only for this call.
    let observation = unsafe { &*observation };
    let copy = |pointer: *const f64| {
        (!pointer.is_null()).then(|| {
            // SAFETY: callback arrays have `dimension` entries by contract.
            unsafe { Array1::from_vec(slice::from_raw_parts(pointer, emitted.dimension).to_vec()) }
        })
    };
    let mut record =
        HyperplaneObservation::default().with_constraint_force(observation.constraint_force);
    record.position = copy(observation.position);
    record.projected_force = copy(observation.projected_force);
    if observation.thermodynamic_derivative.is_finite() {
        record.observables.insert(
            THERMODYNAMIC_DERIVATIVE.into(),
            observation.thermodynamic_derivative,
        );
    } else if emitted.derivative_required {
        emitted.error = Some(SaddleError::NonFinite("thermodynamic derivative"));
        emitted.observations.push(record);
        return RGSADDLE_NON_FINITE;
    }
    emitted.observations.push(record);
    RGSADDLE_OK
}

impl HyperplaneSampler for RgsaddleHyperplaneSampler {
    fn sample_hyperplane(
        &mut self,
        plane: &Hyperplane,
        budget: &HyperplaneSampleBudget,
    ) -> Result<HyperplaneEnsemble, SaddleError> {
        let request = RgsaddleHyperplaneBudget {
            version: RgsaddleVersion {
                major: crate::capi::RGSADDLE_ABI_MAJOR,
                minor: crate::capi::RGSADDLE_ABI_MINOR,
            },
            flags: 0,
            thermal: budget.thermal,
            seed: budget.seed,
            thermalize: budget.thermalize as i64,
            stride: budget.stride as i64,
            samples: budget.samples as i64,
            lambda: budget.lambda.unwrap_or(f64::NAN),
            positions: i32::from(budget.positions),
        };
        let mut emitted = Emitted {
            dimension: plane.dim(),
            observations: Vec::with_capacity(budget.samples),
            derivative_required: budget.lambda.is_some(),
            error: None,
        };
        let status = (self.sample)(
            self.context,
            plane.point().as_ptr(),
            plane.normal().as_ptr(),
            plane.dim(),
            &request,
            emit_observation,
            (&mut emitted as *mut Emitted).cast(),
        );
        if let Some(error) = emitted.error {
            return Err(error);
        }
        if status != RGSADDLE_OK {
            return Err(SaddleError::Surface(format!(
                "hyperplane sampler callback status {status}"
            )));
        }
        Ok(HyperplaneEnsemble {
            evaluations: emitted.observations.len(),
            observations: emitted.observations,
            acceptance: 1.0,
        })
    }
}

/// Create an engine-owned callback sampler.
#[unsafe(no_mangle)]
pub extern "C" fn rgsaddle_hyperplane_sampler_create(
    context: *mut c_void,
    sample: Option<RgsaddleSampleHyperplaneFn>,
) -> *mut RgsaddleHyperplaneSampler {
    sample.map_or(std::ptr::null_mut(), |sample| {
        Box::into_raw(Box::new(RgsaddleHyperplaneSampler { context, sample }))
    })
}

/// Stable observable name for `V_full - V_harmonic`.
#[unsafe(no_mangle)]
pub extern "C" fn rgsaddle_thermodynamic_derivative_observable() -> *const std::ffi::c_char {
    static POINTER: std::sync::OnceLock<std::ffi::CString> = std::sync::OnceLock::new();
    let name = POINTER.get_or_init(|| {
        std::ffi::CString::new(THERMODYNAMIC_DERIVATIVE).expect("thermodynamic derivative name")
    });
    name.as_ptr()
}

/// Free a callback sampler. Null is accepted.
///
/// # Safety
///
/// A non-null pointer must come from
/// [`rgsaddle_hyperplane_sampler_create`] and must not have been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_hyperplane_sampler_free(sampler: *mut RgsaddleHyperplaneSampler) {
    if !sampler.is_null() {
        // SAFETY: ownership came from `rgsaddle_hyperplane_sampler_create`.
        drop(unsafe { Box::from_raw(sampler) });
    }
}

/// Run fixed-plane lambda TI through an engine callback.
///
/// # Safety
///
/// `sampler` must be a live sampler from the matching constructor. `point`
/// and `normal` must each address `dimension` doubles. `config` and
/// `result` must address initialized/readable and writable values,
/// respectively, for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_hyperplane_thermodynamic_integration(
    sampler: *mut RgsaddleHyperplaneSampler,
    point: *const f64,
    normal: *const f64,
    dimension: usize,
    config: *const RgsaddleHyperplaneTiConfig,
    result: *mut RgsaddleHyperplaneTiResult,
) -> i32 {
    if sampler.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    if point.is_null() || normal.is_null() {
        return RGSADDLE_NULL_SURFACE;
    }
    if config.is_null() || result.is_null() || dimension == 0 {
        return RGSADDLE_NULL_REPORT;
    }
    // SAFETY: pointers and lengths are checked above and owned by the caller.
    let point = unsafe { slice::from_raw_parts(point, dimension) };
    // SAFETY: same as `point`.
    let normal = unsafe { slice::from_raw_parts(normal, dimension) };
    // SAFETY: checked non-null above.
    let config = unsafe { &*config };
    if config.version.major != crate::capi::RGSADDLE_ABI_MAJOR
        || config.version.minor < HYPERPLANE_TI_ABI_MINOR
    {
        return RGSADDLE_ABI_MISMATCH;
    }
    if config.nodes <= 0
        || config.thermalize < 0
        || config.stride <= 0
        || config.samples <= 0
        || config.blocks <= 0
    {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let plane = match Hyperplane::new(
        Array1::from_vec(point.to_vec()),
        Array1::from_vec(normal.to_vec()),
    ) {
        Ok(plane) => plane,
        Err(error) => return status_of(&error),
    };
    let settings = ThermodynamicIntegrationConfig::default()
        .with_thermal(config.thermal)
        .with_nodes(config.nodes as usize)
        .with_thermalize(config.thermalize as usize)
        .with_stride(config.stride as usize)
        .with_samples(config.samples as usize)
        .with_blocks(config.blocks as usize)
        .with_seed(config.seed);
    // SAFETY: checked non-null and borrowed for this call.
    match thermodynamic_integration(unsafe { &mut *sampler }, &plane, &settings) {
        Ok(estimate) => {
            // SAFETY: checked non-null above.
            unsafe {
                *result = RgsaddleHyperplaneTiResult {
                    free_energy: estimate.free_energy,
                    error: estimate.error,
                    evaluations: estimate.evaluations as i64,
                };
            }
            RGSADDLE_OK
        }
        Err(error) => status_of(&error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    extern "C" fn sample(
        _context: *mut c_void,
        point: *const f64,
        _normal: *const f64,
        _dimension: usize,
        budget: *const RgsaddleHyperplaneBudget,
        emit: RgsaddleHyperplaneEmitFn,
        emit_context: *mut c_void,
    ) -> i32 {
        // SAFETY: the library supplies these pointers for the callback.
        let budget = unsafe { &*budget };
        for _ in 0..budget.samples {
            let observation = RgsaddleHyperplaneObservation {
                constraint_force: 0.0,
                position: point,
                projected_force: std::ptr::null(),
                thermodynamic_derivative: 0.0,
            };
            let status = emit(emit_context, &observation);
            if status != RGSADDLE_OK {
                return status;
            }
        }
        RGSADDLE_OK
    }

    #[test]
    fn thermodynamic_derivative_observable_is_stable() {
        // SAFETY: returns a process-static NUL-terminated string.
        let name =
            unsafe { std::ffi::CStr::from_ptr(rgsaddle_thermodynamic_derivative_observable()) };
        assert_eq!(name.to_str().unwrap(), THERMODYNAMIC_DERIVATIVE);
    }

    #[test]
    fn callback_ti_returns_the_emitted_constant() {
        let sampler = rgsaddle_hyperplane_sampler_create(std::ptr::null_mut(), Some(sample));
        let point = [0.0, 0.0];
        let normal = [1.0, 0.0];
        let config = RgsaddleHyperplaneTiConfig {
            version: RgsaddleVersion {
                major: crate::capi::RGSADDLE_ABI_MAJOR,
                minor: crate::capi::RGSADDLE_ABI_MINOR,
            },
            flags: 0,
            thermal: 0.2,
            nodes: 4,
            thermalize: 0,
            stride: 1,
            samples: 100,
            blocks: 4,
            seed: 3,
        };
        let mut result = RgsaddleHyperplaneTiResult {
            free_energy: f64::NAN,
            error: f64::NAN,
            evaluations: -1,
        };
        // SAFETY: all pointers remain valid through the call.
        let status = unsafe {
            rgsaddle_hyperplane_thermodynamic_integration(
                sampler,
                point.as_ptr(),
                normal.as_ptr(),
                point.len(),
                &config,
                &mut result,
            )
        };
        assert_eq!(status, RGSADDLE_OK);
        assert_eq!(result.free_energy, 0.0);
        assert_eq!(result.error, 0.0);
        assert_eq!(result.evaluations, 400);
        // SAFETY: sampler came from the matching constructor.
        unsafe { rgsaddle_hyperplane_sampler_free(sampler) };
    }

    fn integration_status(
        sample: RgsaddleSampleHyperplaneFn,
        config: &RgsaddleHyperplaneTiConfig,
    ) -> i32 {
        let sampler = rgsaddle_hyperplane_sampler_create(std::ptr::null_mut(), Some(sample));
        let point = [0.0, 0.0];
        let normal = [1.0, 0.0];
        let mut result = RgsaddleHyperplaneTiResult {
            free_energy: f64::NAN,
            error: f64::NAN,
            evaluations: -1,
        };
        // SAFETY: the pointers stay valid through the call, and the sampler
        // came from the matching constructor.
        unsafe {
            let status = rgsaddle_hyperplane_thermodynamic_integration(
                sampler,
                point.as_ptr(),
                normal.as_ptr(),
                point.len(),
                config,
                &mut result,
            );
            rgsaddle_hyperplane_sampler_free(sampler);
            status
        }
    }

    fn ti_config(samples: i64, blocks: i64) -> RgsaddleHyperplaneTiConfig {
        RgsaddleHyperplaneTiConfig {
            version: RgsaddleVersion {
                major: crate::capi::RGSADDLE_ABI_MAJOR,
                minor: crate::capi::RGSADDLE_ABI_MINOR,
            },
            flags: 0,
            thermal: 0.2,
            nodes: 1,
            thermalize: 0,
            stride: 1,
            samples,
            blocks,
            seed: 1,
        }
    }

    extern "C" fn sample_non_finite(
        _context: *mut c_void,
        _point: *const f64,
        _normal: *const f64,
        _dimension: usize,
        budget: *const RgsaddleHyperplaneBudget,
        emit: RgsaddleHyperplaneEmitFn,
        emit_context: *mut c_void,
    ) -> i32 {
        // SAFETY: the library supplies these pointers for the callback.
        let budget = unsafe { &*budget };
        for _ in 0..budget.samples {
            let observation = RgsaddleHyperplaneObservation {
                constraint_force: 0.0,
                position: std::ptr::null(),
                projected_force: std::ptr::null(),
                thermodynamic_derivative: f64::NAN,
            };
            let status = emit(emit_context, &observation);
            if status != RGSADDLE_OK {
                return status;
            }
        }
        RGSADDLE_OK
    }

    extern "C" fn sample_series(
        _context: *mut c_void,
        _point: *const f64,
        _normal: *const f64,
        _dimension: usize,
        budget: *const RgsaddleHyperplaneBudget,
        emit: RgsaddleHyperplaneEmitFn,
        emit_context: *mut c_void,
    ) -> i32 {
        // SAFETY: the library supplies these pointers for the callback.
        let budget = unsafe { &*budget };
        let kind = budget.seed;
        for index in 0..budget.samples {
            let value = if kind == 14 {
                let shift = if index < budget.samples / 2 {
                    0.0
                } else {
                    0.05
                };
                let alternation = if index % 2 == 0 { 1.0 } else { -1.0 };
                shift + alternation
            } else {
                0.01 * index as f64 + if index % 2 == 0 { 0.001 } else { -0.001 }
            };
            let observation = RgsaddleHyperplaneObservation {
                constraint_force: 0.0,
                position: std::ptr::null(),
                projected_force: std::ptr::null(),
                thermodynamic_derivative: value,
            };
            let status = emit(emit_context, &observation);
            if status != RGSADDLE_OK {
                return status;
            }
        }
        RGSADDLE_OK
    }

    #[test]
    fn hyperplane_ti_rejects_a_stale_minor() {
        let mut config = ti_config(100, 4);
        config.version.minor = HYPERPLANE_TI_ABI_MINOR - 1;
        assert_eq!(
            integration_status(sample, &config),
            crate::capi::RGSADDLE_ABI_MISMATCH
        );
        config.version.minor = HYPERPLANE_TI_ABI_MINOR;
        assert_eq!(
            integration_status(sample, &config),
            crate::capi::RGSADDLE_OK
        );
        config.version.major = 0;
        assert_eq!(
            integration_status(sample, &config),
            crate::capi::RGSADDLE_ABI_MISMATCH
        );
    }

    #[test]
    fn a_non_finite_derivative_is_not_an_invalid_parameter() {
        let config = ti_config(16, 4);
        assert_eq!(
            integration_status(sample_non_finite, &config),
            crate::capi::RGSADDLE_NON_FINITE
        );
    }

    #[test]
    fn split_half_drift_reports_not_equilibrated() {
        let mut config = ti_config(64, 4);
        config.seed = 14;
        assert_eq!(
            integration_status(sample_series, &config),
            crate::capi::RGSADDLE_NOT_EQUILIBRATED
        );
    }

    #[test]
    fn a_trend_reports_insufficient_effective_samples() {
        let mut config = ti_config(400, 8);
        config.seed = 15;
        assert_eq!(
            integration_status(sample_series, &config),
            crate::capi::RGSADDLE_INSUFFICIENT_EFFECTIVE_SAMPLES
        );
    }
}
