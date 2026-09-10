//! C interface for the shared basin constrained dimer force.
use crate::{
    SaddleError,
    capi::*,
    kappa::{KappaDimerConfig, kappa_dimer_force},
};
use ndarray::{Array1, ArrayView1, ArrayView2};
use rgmin::{EigenParams, EigensolverKind};
use std::{ffi::c_void, slice};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RgsaddleKappaConfig {
    pub version: RgsaddleVersion,
    pub beta: f64,
    pub tolerance: f64,
    pub max_iterations: i64,
    pub krylov_dimension: i64,
    pub eigen_kind: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RgsaddleKappaReport {
    pub version: RgsaddleVersion,
    pub kappa: f64,
    pub tangent_curvature: f64,
    pub gamma_parallel: f64,
    pub gamma_perpendicular: f64,
    pub residual: f64,
    pub hessian_actions: i64,
    pub tangent_dimension: i64,
}

pub type RgsaddleHessianFn = extern "C" fn(*mut c_void, i64, *const f64, *mut f64) -> i32;

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
    let cfg = KappaDimerConfig {
        beta: cfg.beta,
        eigen: EigenParams {
            kind,
            nev: 1,
            krylov: cfg.krylov_dimension as usize,
            max_iter: cfg.max_iterations as usize,
            tol: cfg.tolerance,
        },
    };
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
        Err(SaddleError::Solver(message)) => {
            eprintln!("kappa eigensolver: {message}");
            RGSADDLE_SOLVER
        }
    }
}
