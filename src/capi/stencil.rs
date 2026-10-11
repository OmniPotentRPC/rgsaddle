//! Opaque stencil config and the one-callback session.
//!
//! ABI minor 17. Existing `rgsaddle_band_*` symbols are unchanged.
//! Keys are one leaf of [`crate::stencil::StencilConfig`]. A wrong
//! key, a wrong type, or a key that does not apply to the method
//! returns a status and sets [`rgsaddle_last_error`].

use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};

use ndarray::{Array1, Array2, ArrayView2};
use rgmin::{FireKind, Method};

use crate::band::{BandStatus, Cell};
use crate::error::SaddleError;
use crate::minmode::{FiniteDifference, MinModeKind};
use crate::projection::ProjectionKind;
use crate::stencil::{
    AutoNebParams, BandMethod, EfParams, Interp, Method as StencilMethod, OcinebParams, Phase,
    PointMethod, Preset, Refine, SpringModel, Stencil, StencilConfig, StencilReport,
    StencilSurface, ZoomEnds, ZoomForce, ZoomParams, ZoomWindow, method_ordinal, phase_ordinal,
};
use crate::tangent::TangentKind;

use super::{
    RGSADDLE_INVALID_PARAMETER, RGSADDLE_NO_EVALUATION, RGSADDLE_NULL_SESSION, RGSADDLE_OK,
    status_of,
};

pub const RGSADDLE_STENCIL_KEY_METHOD: i32 = 1;
pub const RGSADDLE_STENCIL_KEY_SPRING_MODEL: i32 = 2;
pub const RGSADDLE_STENCIL_KEY_ZOOM_STABILITY: i32 = 3;
pub const RGSADDLE_STENCIL_KEY_ZOOM_WINDOW: i32 = 4;
pub const RGSADDLE_STENCIL_KEY_ZOOM_OFFSET: i32 = 5;
pub const RGSADDLE_STENCIL_KEY_ZOOM_INTERP: i32 = 6;
pub const RGSADDLE_STENCIL_KEY_ZOOM_ENDS: i32 = 7;
pub const RGSADDLE_STENCIL_KEY_ZOOM_MAX_ITERATIONS: i32 = 8;
pub const RGSADDLE_STENCIL_KEY_ZOOM_FORCE_KIND: i32 = 9;
pub const RGSADDLE_STENCIL_KEY_AUTONEB_N_SIMUL: i32 = 10;
pub const RGSADDLE_STENCIL_KEY_AUTONEB_N_MAX: i32 = 11;
pub const RGSADDLE_STENCIL_KEY_AUTONEB_LOCAL_MAX_STEPS: i32 = 12;
pub const RGSADDLE_STENCIL_KEY_AUTONEB_CLIMB_MAX_STEPS: i32 = 13;
pub const RGSADDLE_STENCIL_KEY_AUTONEB_CLIMB: i32 = 14;
pub const RGSADDLE_STENCIL_KEY_AUTONEB_FULL_BAND: i32 = 15;
pub const RGSADDLE_STENCIL_KEY_REFINE: i32 = 16;
pub const RGSADDLE_STENCIL_KEY_TANGENT: i32 = 17;
pub const RGSADDLE_STENCIL_KEY_PROJECTION: i32 = 18;
pub const RGSADDLE_STENCIL_KEY_STEPPER: i32 = 19;
pub const RGSADDLE_STENCIL_KEY_THREADS: i32 = 20;
pub const RGSADDLE_STENCIL_KEY_MINMODE_KIND: i32 = 21;
pub const RGSADDLE_STENCIL_KEY_MINMODE_DIFFERENCE: i32 = 22;
pub const RGSADDLE_STENCIL_KEY_MINMODE_MAX_ROTATIONS: i32 = 23;
pub const RGSADDLE_STENCIL_KEY_MINMODE_KRYLOV: i32 = 24;
pub const RGSADDLE_STENCIL_KEY_LBFGS_MEMORY: i32 = 25;

pub const RGSADDLE_STENCIL_KEY_K: i32 = 101;
pub const RGSADDLE_STENCIL_KEY_K_MIN: i32 = 102;
pub const RGSADDLE_STENCIL_KEY_K_MAX: i32 = 103;
pub const RGSADDLE_STENCIL_KEY_ZOOM_ALPHA: i32 = 104;
pub const RGSADDLE_STENCIL_KEY_ZOOM_FORCE: i32 = 105;
pub const RGSADDLE_STENCIL_KEY_AUTONEB_LOCAL_FORCE_TOL: i32 = 106;
pub const RGSADDLE_STENCIL_KEY_AUTONEB_SPACE_ENERGY_RATIO: i32 = 107;
pub const RGSADDLE_STENCIL_KEY_CI_TOL: i32 = 108;
pub const RGSADDLE_STENCIL_KEY_CI_RMS_TOL: i32 = 109;
pub const RGSADDLE_STENCIL_KEY_IMAGE_SCALE: i32 = 110;
pub const RGSADDLE_STENCIL_KEY_BAND_SLACK: i32 = 111;
pub const RGSADDLE_STENCIL_KEY_CI_TRIGGER_FACTOR: i32 = 112;
pub const RGSADDLE_STENCIL_KEY_CI_TRIGGER_FORCE: i32 = 113;
pub const RGSADDLE_STENCIL_KEY_MAX_MOVE: i32 = 114;
pub const RGSADDLE_STENCIL_KEY_EF_ACTIVATE: i32 = 115;
pub const RGSADDLE_STENCIL_KEY_OCINEB_TRUST: i32 = 116;
pub const RGSADDLE_STENCIL_KEY_MINMODE_DR: i32 = 117;
pub const RGSADDLE_STENCIL_KEY_MINMODE_ROTATION_TOL: i32 = 118;
pub const RGSADDLE_STENCIL_KEY_MINMODE_FORCE_TOL: i32 = 119;
pub const RGSADDLE_STENCIL_KEY_MINMODE_MAX_MOVE: i32 = 120;
pub const RGSADDLE_STENCIL_KEY_MINMODE_ROTATION_ANGLE_TOL: i32 = 121;

pub const RGSADDLE_STENCIL_KEY_EXPLICIT_KS: i32 = 201;
pub const RGSADDLE_STENCIL_KEY_CELL: i32 = 202;

pub const RGSADDLE_STENCIL_QUERY_MODE: i32 = 1;
pub const RGSADDLE_STENCIL_QUERY_CURVATURE: i32 = 2;
pub const RGSADDLE_STENCIL_QUERY_CLIMBING_INDEX: i32 = 3;
pub const RGSADDLE_STENCIL_QUERY_ZOOM_WINDOW: i32 = 4;
pub const RGSADDLE_STENCIL_QUERY_IMAGE_IDS: i32 = 5;
pub const RGSADDLE_STENCIL_QUERY_SHAPE: i32 = 6;

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::new("").unwrap());
}

fn set_last_error(text: &str) {
    let clean = text.replace('\0', " ");
    let owned = CString::new(clean).unwrap_or_else(|_| CString::new("stencil error").unwrap());
    LAST_ERROR.with(|slot| *slot.borrow_mut() = owned);
}

fn clear_last_error() {
    set_last_error("");
}

#[unsafe(no_mangle)]
pub extern "C" fn rgsaddle_last_error() -> *const c_char {
    LAST_ERROR.with(|slot| slot.borrow().as_ptr())
}

struct ConfigInner {
    config: StencilConfig,
    lbfgs_memory: usize,
    force_kind_abs: bool,
}

pub struct RgsaddleStencilConfig {
    inner: ConfigInner,
}

#[repr(C)]
pub struct RgsaddleStencilReport {
    pub struct_size: usize,
    pub phase: i32,
    pub status: i32,
    pub max_force: f64,
    pub ci_index: i64,
    pub iteration: i64,
    pub surface_rows: i64,
    pub total_evaluations: i64,
    pub zoom_lo: i64,
    pub zoom_hi: i64,
    pub curvature: f64,
}

pub struct RgsaddleStencil {
    stencil: Stencil,
    rows: usize,
    dof: usize,
    last: StencilReport,
}

fn preset_of(code: i32) -> Result<Preset, String> {
    match code {
        0 => Ok(Preset::CiNeb),
        1 => Ok(Preset::EonZoom),
        2 => Ok(Preset::OrcaZoomNebTs),
        3 => Ok(Preset::AutoNeb),
        4 => Ok(Preset::Dimer),
        5 => Ok(Preset::Lanczos),
        6 => Ok(Preset::Art),
        7 => Ok(Preset::Plain),
        _ => Err(format!("unknown stencil preset {code}")),
    }
}

fn is_point(config: &StencilConfig) -> bool {
    matches!(config.method, StencilMethod::Point(_))
}

fn band_key_error() -> String {
    "a point stencil cannot take band options (spring, zoom, autoneb, refine)".into()
}

fn point_key_error() -> String {
    "a min-mode key does not apply to a band method".into()
}

fn zoom_params(config: &StencilConfig) -> Result<ZoomParams, String> {
    match config.method {
        StencilMethod::Band(BandMethod::Zoom(params)) => Ok(params),
        _ => Err("zoom key does not apply to this method".into()),
    }
}

fn zoom_params_mut(config: &mut StencilConfig) -> Result<&mut ZoomParams, String> {
    match &mut config.method {
        StencilMethod::Band(BandMethod::Zoom(params)) => Ok(params),
        _ => Err("zoom key does not apply to this method".into()),
    }
}

fn auto_params(config: &StencilConfig) -> Result<&AutoNebParams, String> {
    match &config.method {
        StencilMethod::Band(BandMethod::AutoNeb(params)) => Ok(params),
        _ => Err("AutoNEB key does not apply to this method".into()),
    }
}

fn auto_params_mut(config: &mut StencilConfig) -> Result<&mut AutoNebParams, String> {
    match &mut config.method {
        StencilMethod::Band(BandMethod::AutoNeb(params)) => Ok(params),
        _ => Err("AutoNEB key does not apply to this method".into()),
    }
}

fn require_band(config: &StencilConfig) -> Result<(), String> {
    if is_point(config) {
        Err(band_key_error())
    } else {
        Ok(())
    }
}

fn require_point(config: &StencilConfig) -> Result<(), String> {
    if is_point(config) {
        Ok(())
    } else {
        Err(point_key_error())
    }
}

fn set_method(config: &mut StencilConfig, value: i64) -> Result<(), String> {
    config.method = match value {
        0 => StencilMethod::Band(BandMethod::Plain),
        1 => StencilMethod::Band(BandMethod::CiNeb),
        2 => {
            let params = auto_params(config).cloned().unwrap_or_default();
            StencilMethod::Band(BandMethod::AutoNeb(params))
        }
        3 => {
            let params = zoom_params(config).unwrap_or_default();
            StencilMethod::Band(BandMethod::Zoom(params))
        }
        4 => StencilMethod::Point(PointMethod::Dimer),
        5 => StencilMethod::Point(PointMethod::Lanczos),
        6 => StencilMethod::Point(PointMethod::Art),
        _ => return Err(format!("unknown stencil method {value}")),
    };
    if value == 0 {
        config.band.climbing = None;
    }
    Ok(())
}

fn spring_ordinal(spring: &SpringModel) -> i64 {
    match spring {
        SpringModel::Uniform { .. } => 0,
        SpringModel::EnergyWeighted { .. } => 1,
        SpringModel::OnsagerMachlup { .. } => 2,
        SpringModel::Explicit { .. } => 3,
    }
}

fn set_i64_inner(inner: &mut ConfigInner, key: i32, value: i64) -> Result<(), String> {
    let config = &mut inner.config;
    match key {
        RGSADDLE_STENCIL_KEY_METHOD => set_method(config, value),
        RGSADDLE_STENCIL_KEY_SPRING_MODEL => {
            require_band(config)?;
            let (k_min, k_max, k, ks) = match &config.spring {
                SpringModel::Uniform { k } => (0.972, 9.72, *k, Vec::new()),
                SpringModel::EnergyWeighted { k_min, k_max } => {
                    (*k_min, *k_max, *k_min, Vec::new())
                }
                SpringModel::OnsagerMachlup { k } => (0.972, 9.72, *k, Vec::new()),
                SpringModel::Explicit { ks } => (0.972, 9.72, 5.0, ks.clone()),
            };
            config.spring = match value {
                0 => SpringModel::Uniform { k },
                1 => SpringModel::EnergyWeighted { k_min, k_max },
                2 => SpringModel::OnsagerMachlup { k },
                3 => SpringModel::Explicit { ks },
                _ => return Err(format!("unknown spring model {value}")),
            };
            Ok(())
        }
        RGSADDLE_STENCIL_KEY_ZOOM_STABILITY => {
            zoom_params_mut(config)?.stability = value.max(0) as usize;
            Ok(())
        }
        RGSADDLE_STENCIL_KEY_ZOOM_WINDOW => {
            let params = zoom_params_mut(config)?;
            let offset = match params.window {
                ZoomWindow::Auto { offset, .. } | ZoomWindow::Manual { offset } => offset,
            };
            let alpha = match params.window {
                ZoomWindow::Auto { alpha, .. } => alpha,
                ZoomWindow::Manual { .. } => 0.5,
            };
            params.window = match value {
                0 => ZoomWindow::Auto { alpha, offset },
                1 => ZoomWindow::Manual { offset },
                _ => return Err(format!("unknown zoom window {value}")),
            };
            Ok(())
        }
        RGSADDLE_STENCIL_KEY_ZOOM_OFFSET => {
            let params = zoom_params_mut(config)?;
            let offset = value.max(0) as usize;
            params.window = match params.window {
                ZoomWindow::Auto { alpha, .. } => ZoomWindow::Auto { alpha, offset },
                ZoomWindow::Manual { .. } => ZoomWindow::Manual { offset },
            };
            Ok(())
        }
        RGSADDLE_STENCIL_KEY_ZOOM_INTERP => {
            zoom_params_mut(config)?.interpolation = match value {
                0 => Interp::Linear,
                1 => Interp::Cubic,
                _ => return Err(format!("unknown zoom interpolation {value}")),
            };
            Ok(())
        }
        RGSADDLE_STENCIL_KEY_ZOOM_ENDS => {
            zoom_params_mut(config)?.ends = match value {
                0 => ZoomEnds::Fixed,
                1 => ZoomEnds::FreePerp,
                _ => return Err(format!("unknown zoom ends {value}")),
            };
            Ok(())
        }
        RGSADDLE_STENCIL_KEY_ZOOM_MAX_ITERATIONS => {
            zoom_params_mut(config)?.max_iterations = value.max(0) as usize;
            Ok(())
        }
        RGSADDLE_STENCIL_KEY_ZOOM_FORCE_KIND => {
            let _ = zoom_params(config)?;
            inner.force_kind_abs = match value {
                0 => false,
                1 => true,
                _ => return Err(format!("unknown zoom force kind {value}")),
            };
            let params = zoom_params_mut(config)?;
            let magnitude = match params.force {
                ZoomForce::RelTol(scale) => scale,
                ZoomForce::Abs(force) => force,
            };
            params.force = if inner.force_kind_abs {
                ZoomForce::Abs(magnitude)
            } else {
                ZoomForce::RelTol(magnitude)
            };
            Ok(())
        }
        RGSADDLE_STENCIL_KEY_AUTONEB_N_SIMUL => {
            auto_params_mut(config)?.n_simul = value.max(0) as usize;
            Ok(())
        }
        RGSADDLE_STENCIL_KEY_AUTONEB_N_MAX => {
            auto_params_mut(config)?.n_max = value.max(0) as usize;
            Ok(())
        }
        RGSADDLE_STENCIL_KEY_AUTONEB_LOCAL_MAX_STEPS => {
            auto_params_mut(config)?.local_max_steps = value.max(0) as usize;
            Ok(())
        }
        RGSADDLE_STENCIL_KEY_AUTONEB_CLIMB_MAX_STEPS => {
            auto_params_mut(config)?.climb_max_steps = value.max(0) as usize;
            Ok(())
        }
        RGSADDLE_STENCIL_KEY_AUTONEB_CLIMB => {
            auto_params_mut(config)?.climb = value != 0;
            Ok(())
        }
        RGSADDLE_STENCIL_KEY_AUTONEB_FULL_BAND => {
            auto_params_mut(config)?.full_band = value != 0;
            Ok(())
        }
        RGSADDLE_STENCIL_KEY_REFINE => {
            require_band(config)?;
            config.refine = match value {
                0 => Refine::None,
                1 => Refine::Ocineb(OcinebParams::default()),
                2 => Refine::EigenvectorFollowing(EfParams::default()),
                _ => return Err(format!("unknown refine {value}")),
            };
            Ok(())
        }
        RGSADDLE_STENCIL_KEY_TANGENT => {
            require_band(config)?;
            config.band.tangent = match value {
                0 => TangentKind::Simple,
                1 => TangentKind::Improved,
                _ => return Err(format!("unknown tangent {value}")),
            };
            Ok(())
        }
        RGSADDLE_STENCIL_KEY_PROJECTION => {
            require_band(config)?;
            config.band.projection = match value {
                0 => ProjectionKind::PlainElasticBand,
                1 => ProjectionKind::Neb,
                2 => ProjectionKind::DoublyNudged,
                _ => return Err(format!("unknown projection {value}")),
            };
            Ok(())
        }
        RGSADDLE_STENCIL_KEY_STEPPER => {
            require_band(config)?;
            config.band.method = match value {
                0 => Method::Fire { kind: FireKind::V2 },
                1 => Method::Lbfgs {
                    memory: inner.lbfgs_memory,
                },
                _ => return Err(format!("unknown stepper {value}")),
            };
            config.band.quickmin = false;
            Ok(())
        }
        RGSADDLE_STENCIL_KEY_THREADS => {
            if value <= 0 {
                return Err("stencil needs at least one image-evaluation thread".into());
            }
            config.threads = value as usize;
            Ok(())
        }
        RGSADDLE_STENCIL_KEY_MINMODE_KIND => {
            require_point(config)?;
            config.point.kind = match value {
                0 => MinModeKind::Dimer,
                1 => MinModeKind::Lanczos,
                _ => return Err(format!("unknown min-mode kind {value}")),
            };
            Ok(())
        }
        RGSADDLE_STENCIL_KEY_MINMODE_DIFFERENCE => {
            require_point(config)?;
            config.point.difference = match value {
                0 => FiniteDifference::Forward,
                1 => FiniteDifference::Central,
                _ => return Err(format!("unknown difference {value}")),
            };
            Ok(())
        }
        RGSADDLE_STENCIL_KEY_MINMODE_MAX_ROTATIONS => {
            require_point(config)?;
            config.point.max_rotations = value.max(0) as usize;
            Ok(())
        }
        RGSADDLE_STENCIL_KEY_MINMODE_KRYLOV => {
            require_point(config)?;
            config.point.krylov_dim = value.max(0) as usize;
            Ok(())
        }
        RGSADDLE_STENCIL_KEY_LBFGS_MEMORY => {
            require_band(config)?;
            if value <= 0 {
                return Err("L-BFGS memory must be positive".into());
            }
            inner.lbfgs_memory = value as usize;
            if let Method::Lbfgs { memory } = &mut config.band.method {
                *memory = inner.lbfgs_memory;
            }
            Ok(())
        }
        RGSADDLE_STENCIL_KEY_K
        | RGSADDLE_STENCIL_KEY_K_MIN
        | RGSADDLE_STENCIL_KEY_K_MAX
        | RGSADDLE_STENCIL_KEY_ZOOM_ALPHA
        | RGSADDLE_STENCIL_KEY_ZOOM_FORCE
        | RGSADDLE_STENCIL_KEY_AUTONEB_LOCAL_FORCE_TOL
        | RGSADDLE_STENCIL_KEY_AUTONEB_SPACE_ENERGY_RATIO
        | RGSADDLE_STENCIL_KEY_CI_TOL
        | RGSADDLE_STENCIL_KEY_CI_RMS_TOL
        | RGSADDLE_STENCIL_KEY_IMAGE_SCALE
        | RGSADDLE_STENCIL_KEY_BAND_SLACK
        | RGSADDLE_STENCIL_KEY_CI_TRIGGER_FACTOR
        | RGSADDLE_STENCIL_KEY_CI_TRIGGER_FORCE
        | RGSADDLE_STENCIL_KEY_MAX_MOVE
        | RGSADDLE_STENCIL_KEY_EF_ACTIVATE
        | RGSADDLE_STENCIL_KEY_OCINEB_TRUST
        | RGSADDLE_STENCIL_KEY_MINMODE_DR
        | RGSADDLE_STENCIL_KEY_MINMODE_ROTATION_TOL
        | RGSADDLE_STENCIL_KEY_MINMODE_FORCE_TOL
        | RGSADDLE_STENCIL_KEY_MINMODE_MAX_MOVE
        | RGSADDLE_STENCIL_KEY_MINMODE_ROTATION_ANGLE_TOL => {
            Err("stencil key takes a double".into())
        }
        RGSADDLE_STENCIL_KEY_EXPLICIT_KS | RGSADDLE_STENCIL_KEY_CELL => {
            Err("stencil key takes an array".into())
        }
        _ => Err(format!("unknown stencil key {key}")),
    }
}

fn set_f64_inner(inner: &mut ConfigInner, key: i32, value: f64) -> Result<(), String> {
    let config = &mut inner.config;
    match key {
        RGSADDLE_STENCIL_KEY_K => {
            require_band(config)?;
            match &mut config.spring {
                SpringModel::Uniform { k } | SpringModel::OnsagerMachlup { k } => *k = value,
                SpringModel::EnergyWeighted { .. } | SpringModel::Explicit { .. } => {
                    return Err("K applies to a uniform or Onsager-Machlup spring".into());
                }
            }
            Ok(())
        }
        RGSADDLE_STENCIL_KEY_K_MIN => {
            require_band(config)?;
            match &mut config.spring {
                SpringModel::EnergyWeighted { k_min, .. } => *k_min = value,
                _ => return Err("K_MIN applies to energy-weighted springs".into()),
            }
            Ok(())
        }
        RGSADDLE_STENCIL_KEY_K_MAX => {
            require_band(config)?;
            match &mut config.spring {
                SpringModel::EnergyWeighted { k_max, .. } => *k_max = value,
                _ => return Err("K_MAX applies to energy-weighted springs".into()),
            }
            Ok(())
        }
        RGSADDLE_STENCIL_KEY_ZOOM_ALPHA => {
            let params = zoom_params_mut(config)?;
            match &mut params.window {
                ZoomWindow::Auto { alpha, .. } => *alpha = value,
                ZoomWindow::Manual { .. } => {
                    return Err("ZOOM_ALPHA applies to an automatic window".into());
                }
            }
            Ok(())
        }
        RGSADDLE_STENCIL_KEY_ZOOM_FORCE => {
            let abs = inner.force_kind_abs;
            let params = zoom_params_mut(config)?;
            params.force = if abs {
                ZoomForce::Abs(value)
            } else {
                ZoomForce::RelTol(value)
            };
            Ok(())
        }
        RGSADDLE_STENCIL_KEY_AUTONEB_LOCAL_FORCE_TOL => {
            auto_params_mut(config)?.local_force_tol = value;
            Ok(())
        }
        RGSADDLE_STENCIL_KEY_AUTONEB_SPACE_ENERGY_RATIO => {
            auto_params_mut(config)?.space_energy_ratio = value;
            Ok(())
        }
        RGSADDLE_STENCIL_KEY_CI_TOL => {
            require_band(config)?;
            config.converge.ci_tol = value;
            Ok(())
        }
        RGSADDLE_STENCIL_KEY_CI_RMS_TOL => {
            require_band(config)?;
            config.converge.ci_rms_tol = if value.is_nan() || value < 0.0 {
                None
            } else {
                Some(value)
            };
            Ok(())
        }
        RGSADDLE_STENCIL_KEY_IMAGE_SCALE => {
            require_band(config)?;
            config.converge.image_scale = if value.is_nan() || value < 0.0 {
                None
            } else {
                Some(value)
            };
            Ok(())
        }
        RGSADDLE_STENCIL_KEY_BAND_SLACK => {
            require_band(config)?;
            config.converge.band_slack = value;
            Ok(())
        }
        RGSADDLE_STENCIL_KEY_CI_TRIGGER_FACTOR => {
            require_band(config)?;
            config.climb.trigger_factor = value;
            Ok(())
        }
        RGSADDLE_STENCIL_KEY_CI_TRIGGER_FORCE => {
            require_band(config)?;
            config.climb.trigger_force = value;
            Ok(())
        }
        RGSADDLE_STENCIL_KEY_MAX_MOVE => {
            require_band(config)?;
            config.band.max_move = value;
            Ok(())
        }
        RGSADDLE_STENCIL_KEY_EF_ACTIVATE => {
            require_band(config)?;
            match &mut config.refine {
                Refine::EigenvectorFollowing(params) => params.activate = value,
                _ => return Err("EF_ACTIVATE applies to eigenvector-following refine".into()),
            }
            Ok(())
        }
        RGSADDLE_STENCIL_KEY_OCINEB_TRUST => {
            require_band(config)?;
            match &mut config.refine {
                Refine::Ocineb(params) => params.trust = value,
                _ => return Err("OCINEB_TRUST applies to OCI-NEB refine".into()),
            }
            Ok(())
        }
        RGSADDLE_STENCIL_KEY_MINMODE_DR => {
            require_point(config)?;
            config.point.dr = value;
            Ok(())
        }
        RGSADDLE_STENCIL_KEY_MINMODE_ROTATION_TOL => {
            require_point(config)?;
            config.point.rotation_tol = value;
            Ok(())
        }
        RGSADDLE_STENCIL_KEY_MINMODE_FORCE_TOL => {
            require_point(config)?;
            config.point.force_tol = value;
            Ok(())
        }
        RGSADDLE_STENCIL_KEY_MINMODE_MAX_MOVE => {
            require_point(config)?;
            config.point.max_move = value;
            Ok(())
        }
        RGSADDLE_STENCIL_KEY_MINMODE_ROTATION_ANGLE_TOL => {
            require_point(config)?;
            config.point.rotation_angle_tol = value;
            Ok(())
        }
        key if (1..=25).contains(&key) => Err("stencil key takes an integer".into()),
        RGSADDLE_STENCIL_KEY_EXPLICIT_KS | RGSADDLE_STENCIL_KEY_CELL => {
            Err("stencil key takes an array".into())
        }
        _ => Err(format!("unknown stencil key {key}")),
    }
}

fn get_i64_inner(inner: &ConfigInner, key: i32) -> Result<i64, String> {
    let config = &inner.config;
    match key {
        RGSADDLE_STENCIL_KEY_METHOD => Ok(method_ordinal(&config.method) as i64),
        RGSADDLE_STENCIL_KEY_SPRING_MODEL => {
            require_band(config)?;
            Ok(spring_ordinal(&config.spring))
        }
        RGSADDLE_STENCIL_KEY_ZOOM_STABILITY => Ok(zoom_params(config)?.stability as i64),
        RGSADDLE_STENCIL_KEY_ZOOM_WINDOW => Ok(match zoom_params(config)?.window {
            ZoomWindow::Auto { .. } => 0,
            ZoomWindow::Manual { .. } => 1,
        }),
        RGSADDLE_STENCIL_KEY_ZOOM_OFFSET => Ok(match zoom_params(config)?.window {
            ZoomWindow::Auto { offset, .. } | ZoomWindow::Manual { offset } => offset as i64,
        }),
        RGSADDLE_STENCIL_KEY_ZOOM_INTERP => Ok(match zoom_params(config)?.interpolation {
            Interp::Linear => 0,
            Interp::Cubic => 1,
        }),
        RGSADDLE_STENCIL_KEY_ZOOM_ENDS => Ok(match zoom_params(config)?.ends {
            ZoomEnds::Fixed => 0,
            ZoomEnds::FreePerp => 1,
        }),
        RGSADDLE_STENCIL_KEY_ZOOM_MAX_ITERATIONS => Ok(zoom_params(config)?.max_iterations as i64),
        RGSADDLE_STENCIL_KEY_ZOOM_FORCE_KIND => {
            let _ = zoom_params(config)?;
            Ok(i64::from(inner.force_kind_abs))
        }
        RGSADDLE_STENCIL_KEY_AUTONEB_N_SIMUL => Ok(auto_params(config)?.n_simul as i64),
        RGSADDLE_STENCIL_KEY_AUTONEB_N_MAX => Ok(auto_params(config)?.n_max as i64),
        RGSADDLE_STENCIL_KEY_AUTONEB_LOCAL_MAX_STEPS => {
            Ok(auto_params(config)?.local_max_steps as i64)
        }
        RGSADDLE_STENCIL_KEY_AUTONEB_CLIMB_MAX_STEPS => {
            Ok(auto_params(config)?.climb_max_steps as i64)
        }
        RGSADDLE_STENCIL_KEY_AUTONEB_CLIMB => Ok(i64::from(auto_params(config)?.climb)),
        RGSADDLE_STENCIL_KEY_AUTONEB_FULL_BAND => Ok(i64::from(auto_params(config)?.full_band)),
        RGSADDLE_STENCIL_KEY_REFINE => {
            require_band(config)?;
            Ok(match config.refine {
                Refine::None => 0,
                Refine::Ocineb(_) => 1,
                Refine::EigenvectorFollowing(_) => 2,
            })
        }
        RGSADDLE_STENCIL_KEY_TANGENT => {
            require_band(config)?;
            Ok(match config.band.tangent {
                TangentKind::Simple => 0,
                TangentKind::Improved => 1,
            })
        }
        RGSADDLE_STENCIL_KEY_PROJECTION => {
            require_band(config)?;
            Ok(match config.band.projection {
                ProjectionKind::PlainElasticBand => 0,
                ProjectionKind::Neb => 1,
                ProjectionKind::DoublyNudged => 2,
            })
        }
        RGSADDLE_STENCIL_KEY_STEPPER => {
            require_band(config)?;
            Ok(match config.band.method {
                Method::Lbfgs { .. } => 1,
                _ => 0,
            })
        }
        RGSADDLE_STENCIL_KEY_THREADS => Ok(config.threads as i64),
        RGSADDLE_STENCIL_KEY_MINMODE_KIND => {
            require_point(config)?;
            Ok(match config.point.kind {
                MinModeKind::Dimer => 0,
                MinModeKind::Lanczos => 1,
            })
        }
        RGSADDLE_STENCIL_KEY_MINMODE_DIFFERENCE => {
            require_point(config)?;
            Ok(match config.point.difference {
                FiniteDifference::Forward => 0,
                FiniteDifference::Central => 1,
            })
        }
        RGSADDLE_STENCIL_KEY_MINMODE_MAX_ROTATIONS => {
            require_point(config)?;
            Ok(config.point.max_rotations as i64)
        }
        RGSADDLE_STENCIL_KEY_MINMODE_KRYLOV => {
            require_point(config)?;
            Ok(config.point.krylov_dim as i64)
        }
        RGSADDLE_STENCIL_KEY_LBFGS_MEMORY => {
            require_band(config)?;
            Ok(inner.lbfgs_memory as i64)
        }
        key if (101..=121).contains(&key) => Err("stencil key takes a double".into()),
        RGSADDLE_STENCIL_KEY_EXPLICIT_KS | RGSADDLE_STENCIL_KEY_CELL => {
            Err("stencil key takes an array".into())
        }
        _ => Err(format!("unknown stencil key {key}")),
    }
}

fn get_f64_inner(inner: &ConfigInner, key: i32) -> Result<f64, String> {
    let config = &inner.config;
    match key {
        RGSADDLE_STENCIL_KEY_K => {
            require_band(config)?;
            match &config.spring {
                SpringModel::Uniform { k } | SpringModel::OnsagerMachlup { k } => Ok(*k),
                _ => Err("K applies to a uniform or Onsager-Machlup spring".into()),
            }
        }
        RGSADDLE_STENCIL_KEY_K_MIN => {
            require_band(config)?;
            match &config.spring {
                SpringModel::EnergyWeighted { k_min, .. } => Ok(*k_min),
                _ => Err("K_MIN applies to energy-weighted springs".into()),
            }
        }
        RGSADDLE_STENCIL_KEY_K_MAX => {
            require_band(config)?;
            match &config.spring {
                SpringModel::EnergyWeighted { k_max, .. } => Ok(*k_max),
                _ => Err("K_MAX applies to energy-weighted springs".into()),
            }
        }
        RGSADDLE_STENCIL_KEY_ZOOM_ALPHA => match zoom_params(config)?.window {
            ZoomWindow::Auto { alpha, .. } => Ok(alpha),
            ZoomWindow::Manual { .. } => Err("ZOOM_ALPHA applies to an automatic window".into()),
        },
        RGSADDLE_STENCIL_KEY_ZOOM_FORCE => Ok(match zoom_params(config)?.force {
            ZoomForce::RelTol(scale) => scale,
            ZoomForce::Abs(force) => force,
        }),
        RGSADDLE_STENCIL_KEY_AUTONEB_LOCAL_FORCE_TOL => Ok(auto_params(config)?.local_force_tol),
        RGSADDLE_STENCIL_KEY_AUTONEB_SPACE_ENERGY_RATIO => {
            Ok(auto_params(config)?.space_energy_ratio)
        }
        RGSADDLE_STENCIL_KEY_CI_TOL => {
            require_band(config)?;
            Ok(config.converge.ci_tol)
        }
        RGSADDLE_STENCIL_KEY_CI_RMS_TOL => {
            require_band(config)?;
            Ok(config.converge.ci_rms_tol.unwrap_or(f64::NAN))
        }
        RGSADDLE_STENCIL_KEY_IMAGE_SCALE => {
            require_band(config)?;
            Ok(config.converge.image_scale.unwrap_or(f64::NAN))
        }
        RGSADDLE_STENCIL_KEY_BAND_SLACK => {
            require_band(config)?;
            Ok(config.converge.band_slack)
        }
        RGSADDLE_STENCIL_KEY_CI_TRIGGER_FACTOR => {
            require_band(config)?;
            Ok(config.climb.trigger_factor)
        }
        RGSADDLE_STENCIL_KEY_CI_TRIGGER_FORCE => {
            require_band(config)?;
            Ok(config.climb.trigger_force)
        }
        RGSADDLE_STENCIL_KEY_MAX_MOVE => {
            require_band(config)?;
            Ok(config.band.max_move)
        }
        RGSADDLE_STENCIL_KEY_EF_ACTIVATE => {
            require_band(config)?;
            match &config.refine {
                Refine::EigenvectorFollowing(params) => Ok(params.activate),
                _ => Err("EF_ACTIVATE applies to eigenvector-following refine".into()),
            }
        }
        RGSADDLE_STENCIL_KEY_OCINEB_TRUST => {
            require_band(config)?;
            match &config.refine {
                Refine::Ocineb(params) => Ok(params.trust),
                _ => Err("OCINEB_TRUST applies to OCI-NEB refine".into()),
            }
        }
        RGSADDLE_STENCIL_KEY_MINMODE_DR => {
            require_point(config)?;
            Ok(config.point.dr)
        }
        RGSADDLE_STENCIL_KEY_MINMODE_ROTATION_TOL => {
            require_point(config)?;
            Ok(config.point.rotation_tol)
        }
        RGSADDLE_STENCIL_KEY_MINMODE_FORCE_TOL => {
            require_point(config)?;
            Ok(config.point.force_tol)
        }
        RGSADDLE_STENCIL_KEY_MINMODE_MAX_MOVE => {
            require_point(config)?;
            Ok(config.point.max_move)
        }
        RGSADDLE_STENCIL_KEY_MINMODE_ROTATION_ANGLE_TOL => {
            require_point(config)?;
            Ok(config.point.rotation_angle_tol)
        }
        key if (1..=25).contains(&key) => Err("stencil key takes an integer".into()),
        _ => Err(format!("unknown stencil key {key}")),
    }
}

fn fail(err: String) -> i32 {
    set_last_error(&err);
    RGSADDLE_INVALID_PARAMETER
}

fn fail_saddle(err: SaddleError) -> i32 {
    let text = match &err {
        SaddleError::Invalid(message) => message.clone(),
        other => other.to_string(),
    };
    set_last_error(&text);
    status_of(&err)
}

#[unsafe(no_mangle)]
pub extern "C" fn rgsaddle_stencil_config_new(preset: i32) -> *mut RgsaddleStencilConfig {
    match preset_of(preset) {
        Ok(preset) => {
            clear_last_error();
            let config = StencilConfig::from_preset(preset);
            let force_kind_abs = matches!(
                config.method,
                StencilMethod::Band(BandMethod::Zoom(ZoomParams {
                    force: ZoomForce::Abs(_),
                    ..
                }))
            );
            Box::into_raw(Box::new(RgsaddleStencilConfig {
                inner: ConfigInner {
                    config,
                    lbfgs_memory: 10,
                    force_kind_abs,
                },
            }))
        }
        Err(err) => {
            set_last_error(&err);
            std::ptr::null_mut()
        }
    }
}

/// # Safety
/// `cfg` is null or a live config.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_stencil_config_set_f64(
    cfg: *mut RgsaddleStencilConfig,
    key: i32,
    value: f64,
) -> i32 {
    if cfg.is_null() {
        return fail("stencil config is null".into());
    }
    match set_f64_inner(&mut unsafe { &mut *cfg }.inner, key, value) {
        Ok(()) => {
            clear_last_error();
            RGSADDLE_OK
        }
        Err(err) => fail(err),
    }
}

/// # Safety
/// `cfg` is null or a live config.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_stencil_config_set_i64(
    cfg: *mut RgsaddleStencilConfig,
    key: i32,
    value: i64,
) -> i32 {
    if cfg.is_null() {
        return fail("stencil config is null".into());
    }
    match set_i64_inner(&mut unsafe { &mut *cfg }.inner, key, value) {
        Ok(()) => {
            clear_last_error();
            RGSADDLE_OK
        }
        Err(err) => fail(err),
    }
}

/// # Safety
/// `cfg` is null or a live config. `values` has `n` doubles when `n > 0`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_stencil_config_set_array(
    cfg: *mut RgsaddleStencilConfig,
    key: i32,
    values: *const f64,
    n: usize,
) -> i32 {
    if cfg.is_null() {
        return fail("stencil config is null".into());
    }
    if n > 0 && values.is_null() {
        return fail("stencil array is null".into());
    }
    let slice = if n == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(values, n) }
    };
    let inner = &mut unsafe { &mut *cfg }.inner;
    let result = (|| -> Result<(), String> {
        match key {
            RGSADDLE_STENCIL_KEY_EXPLICIT_KS => {
                require_band(&inner.config)?;
                inner.config.spring = SpringModel::Explicit { ks: slice.to_vec() };
                Ok(())
            }
            RGSADDLE_STENCIL_KEY_CELL => {
                require_band(&inner.config)?;
                if slice.len() != 9 {
                    return Err("a cell needs 9 entries".into());
                }
                inner.config.band.cell = Some(Cell([
                    [slice[0], slice[1], slice[2]],
                    [slice[3], slice[4], slice[5]],
                    [slice[6], slice[7], slice[8]],
                ]));
                Ok(())
            }
            key if (1..=25).contains(&key) => Err("stencil key takes an integer".into()),
            key if (101..=121).contains(&key) => Err("stencil key takes a double".into()),
            _ => Err(format!("unknown stencil key {key}")),
        }
    })();
    match result {
        Ok(()) => {
            clear_last_error();
            RGSADDLE_OK
        }
        Err(err) => fail(err),
    }
}

/// # Safety
/// `cfg` is null or a live config. `out` is a double.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_stencil_config_get_f64(
    cfg: *const RgsaddleStencilConfig,
    key: i32,
    out: *mut f64,
) -> i32 {
    if cfg.is_null() || out.is_null() {
        return fail("stencil config get is null".into());
    }
    match get_f64_inner(&unsafe { &*cfg }.inner, key) {
        Ok(value) => {
            unsafe { *out = value };
            clear_last_error();
            RGSADDLE_OK
        }
        Err(err) => fail(err),
    }
}

/// # Safety
/// `cfg` is null or a live config. `out` is an int64.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_stencil_config_get_i64(
    cfg: *const RgsaddleStencilConfig,
    key: i32,
    out: *mut i64,
) -> i32 {
    if cfg.is_null() || out.is_null() {
        return fail("stencil config get is null".into());
    }
    match get_i64_inner(&unsafe { &*cfg }.inner, key) {
        Ok(value) => {
            unsafe { *out = value };
            clear_last_error();
            RGSADDLE_OK
        }
        Err(err) => fail(err),
    }
}

/// # Safety
/// `cfg` comes from [`rgsaddle_stencil_config_new`] and is freed once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_stencil_config_free(cfg: *mut RgsaddleStencilConfig) {
    if !cfg.is_null() {
        drop(unsafe { Box::from_raw(cfg) });
    }
}

type EvalFn =
    extern "C" fn(*mut c_void, usize, usize, *const u64, *const f64, *mut f64, *mut f64) -> i32;

struct CSurface {
    eval: EvalFn,
    user: *mut c_void,
}

unsafe impl Send for CSurface {}
unsafe impl Sync for CSurface {}

impl StencilSurface for CSurface {
    fn eval(
        &self,
        ids: &[u64],
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        let x = positions
            .as_slice()
            .ok_or_else(|| SaddleError::Invalid("stencil positions are not contiguous".into()))?;
        let e = energies
            .as_slice_mut()
            .ok_or_else(|| SaddleError::Invalid("stencil energies are not contiguous".into()))?;
        let g = gradients
            .as_slice_mut()
            .ok_or_else(|| SaddleError::Invalid("stencil gradients are not contiguous".into()))?;
        let code = (self.eval)(
            self.user,
            positions.nrows(),
            positions.ncols(),
            ids.as_ptr(),
            x.as_ptr(),
            e.as_mut_ptr(),
            g.as_mut_ptr(),
        );
        if code == 0 {
            Ok(())
        } else {
            Err(SaddleError::Surface(format!(
                "stencil callback returned {code}"
            )))
        }
    }
}

fn write_report(out: &mut RgsaddleStencilReport, report: &StencilReport) {
    let (zoom_lo, zoom_hi) = report
        .zoom_window
        .map(|(lo, hi)| (lo as i64, hi as i64))
        .unwrap_or((-1, -1));
    out.phase = phase_ordinal(report.phase);
    out.status = match report.status {
        BandStatus::Running => 0,
        BandStatus::Converged => 1,
    };
    out.max_force = report.max_force;
    out.ci_index = report.ci_index.map(|index| index as i64).unwrap_or(-1);
    out.iteration = report.iteration as i64;
    out.surface_rows = report.surface_rows as i64;
    out.total_evaluations = report.total_rows as i64;
    out.zoom_lo = zoom_lo;
    out.zoom_hi = zoom_hi;
    out.curvature = report.curvature.unwrap_or(f64::NAN);
}

fn check_report(
    out: *mut RgsaddleStencilReport,
) -> Result<&'static mut RgsaddleStencilReport, i32> {
    if out.is_null() {
        set_last_error("stencil report is null");
        return Err(RGSADDLE_INVALID_PARAMETER);
    }
    let report = unsafe { &mut *out };
    if report.struct_size < std::mem::size_of::<RgsaddleStencilReport>() {
        set_last_error("stencil report struct_size is short");
        return Err(RGSADDLE_INVALID_PARAMETER);
    }
    Ok(report)
}

/// # Safety
/// `cfg` is null or a live config. `x` has `rows * dof` doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_stencil_new(
    cfg: *const RgsaddleStencilConfig,
    rows: usize,
    dof: usize,
    x: *const f64,
) -> *mut RgsaddleStencil {
    if cfg.is_null() || x.is_null() || rows == 0 || dof == 0 {
        set_last_error("stencil create needs a config, rows, dof, and positions");
        return std::ptr::null_mut();
    }
    let values = unsafe { std::slice::from_raw_parts(x, rows * dof) };
    let initial = match Array2::from_shape_vec((rows, dof), values.to_vec()) {
        Ok(initial) => initial,
        Err(err) => {
            set_last_error(&err.to_string());
            return std::ptr::null_mut();
        }
    };
    let config = unsafe { &*cfg }.inner.config.clone();
    match Stencil::new(config, initial) {
        Ok(stencil) => {
            clear_last_error();
            let rows = stencil.positions().nrows();
            let dof = stencil.positions().ncols();
            Box::into_raw(Box::new(RgsaddleStencil {
                stencil,
                rows,
                dof,
                last: StencilReport {
                    phase: Phase::Band,
                    status: BandStatus::Running,
                    max_force: f64::NAN,
                    ci_index: None,
                    iteration: 0,
                    surface_rows: 0,
                    total_rows: 0,
                    zoom_window: None,
                    curvature: None,
                    force_gate: "max-abs-component",
                },
            }))
        }
        Err(err) => {
            fail_saddle(err);
            std::ptr::null_mut()
        }
    }
}

/// # Safety
/// `handle` is null or live. `eval` is a callback. `out` is a report whose
/// `struct_size` is at least the size of `rgsaddle_stencil_report_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_stencil_step(
    handle: *mut RgsaddleStencil,
    eval: Option<EvalFn>,
    user: *mut c_void,
    out: *mut RgsaddleStencilReport,
) -> i32 {
    if handle.is_null() {
        set_last_error("stencil is null");
        return RGSADDLE_NULL_SESSION;
    }
    let Some(eval) = eval else {
        return fail("stencil callback is null".into());
    };
    let report = match check_report(out) {
        Ok(report) => report,
        Err(code) => return code,
    };
    let session = unsafe { &mut *handle };
    let surface = CSurface { eval, user };
    match session.stencil.step(&surface) {
        Ok(step) => {
            session.rows = session.stencil.positions().nrows();
            session.dof = session.stencil.positions().ncols();
            session.last = step.clone();
            write_report(report, &step);
            clear_last_error();
            RGSADDLE_OK
        }
        Err(err) => fail_saddle(err),
    }
}

/// # Safety
/// Same as [`rgsaddle_stencil_step`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_stencil_run(
    handle: *mut RgsaddleStencil,
    eval: Option<EvalFn>,
    user: *mut c_void,
    max_steps: usize,
    out: *mut RgsaddleStencilReport,
) -> i32 {
    if handle.is_null() {
        set_last_error("stencil is null");
        return RGSADDLE_NULL_SESSION;
    }
    let Some(eval) = eval else {
        return fail("stencil callback is null".into());
    };
    let report = match check_report(out) {
        Ok(report) => report,
        Err(code) => return code,
    };
    let session = unsafe { &mut *handle };
    let surface = CSurface { eval, user };
    match session.stencil.run(&surface, max_steps) {
        Ok(step) => {
            session.rows = session.stencil.positions().nrows();
            session.dof = session.stencil.positions().ncols();
            session.last = step.clone();
            write_report(report, &step);
            clear_last_error();
            RGSADDLE_OK
        }
        Err(err) => fail_saddle(err),
    }
}

/// # Safety
/// `out` holds `rows * dof` doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_stencil_positions(
    handle: *const RgsaddleStencil,
    out: *mut f64,
) -> i32 {
    if handle.is_null() || out.is_null() {
        return fail("stencil positions are null".into());
    }
    let session = unsafe { &*handle };
    let positions = session.stencil.positions();
    let slice = match positions.as_slice() {
        Some(slice) => slice,
        None => return fail("stencil positions are not contiguous".into()),
    };
    if slice.len() != session.rows.saturating_mul(session.dof) {
        return fail("stencil position length does not match the band".into());
    }
    unsafe {
        std::ptr::copy_nonoverlapping(slice.as_ptr(), out, slice.len());
    }
    clear_last_error();
    RGSADDLE_OK
}

/// # Safety
/// `out` holds one double per image.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_stencil_energies(
    handle: *const RgsaddleStencil,
    out: *mut f64,
) -> i32 {
    if handle.is_null() || out.is_null() {
        return fail("stencil energies are null".into());
    }
    let session = unsafe { &*handle };
    let Some(energies) = session.stencil.energies() else {
        set_last_error("stencil has no energies");
        return RGSADDLE_NO_EVALUATION;
    };
    let slice = energies.as_slice().unwrap_or(&[]);
    unsafe {
        std::ptr::copy_nonoverlapping(slice.as_ptr(), out, slice.len());
    }
    clear_last_error();
    RGSADDLE_OK
}

/// # Safety
/// `out` holds `n` bytes of the value named by `key`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_stencil_query(
    handle: *const RgsaddleStencil,
    key: i32,
    out: *mut c_void,
    n: usize,
) -> i32 {
    if handle.is_null() || out.is_null() {
        return fail("stencil query is null".into());
    }
    let session = unsafe { &*handle };
    match key {
        RGSADDLE_STENCIL_QUERY_MODE => {
            let Some(mode) = session.stencil.mode() else {
                return fail("stencil has no mode".into());
            };
            if n < mode.len() {
                return fail("mode query is short".into());
            }
            unsafe {
                std::ptr::copy_nonoverlapping(mode.as_ptr(), out.cast::<f64>(), mode.len());
            }
        }
        RGSADDLE_STENCIL_QUERY_CURVATURE => {
            if n < 1 {
                return fail("curvature query is short".into());
            }
            let value = session.stencil.curvature().unwrap_or(f64::NAN);
            unsafe { *out.cast::<f64>() = value };
        }
        RGSADDLE_STENCIL_QUERY_CLIMBING_INDEX => {
            if n < 1 {
                return fail("climbing-index query is short".into());
            }
            let value = session
                .stencil
                .climbing_image()
                .map(|index| index as i64)
                .unwrap_or(-1);
            unsafe { *out.cast::<i64>() = value };
        }
        RGSADDLE_STENCIL_QUERY_ZOOM_WINDOW => {
            if n < 2 {
                return fail("zoom-window query is short".into());
            }
            let (lo, hi) = session
                .stencil
                .zoom_window()
                .map(|(lo, hi)| (lo as i64, hi as i64))
                .unwrap_or((-1, -1));
            let dest = unsafe { std::slice::from_raw_parts_mut(out.cast::<i64>(), 2) };
            dest[0] = lo;
            dest[1] = hi;
        }
        RGSADDLE_STENCIL_QUERY_IMAGE_IDS => {
            let ids = session.stencil.image_ids();
            if n < ids.len() {
                return fail("image-id query is short".into());
            }
            unsafe {
                std::ptr::copy_nonoverlapping(ids.as_ptr(), out.cast::<u64>(), ids.len());
            }
        }
        RGSADDLE_STENCIL_QUERY_SHAPE => {
            if n < 2 {
                return fail("shape query is short".into());
            }
            let dest = unsafe { std::slice::from_raw_parts_mut(out.cast::<i64>(), 2) };
            dest[0] = session.rows as i64;
            dest[1] = session.dof as i64;
        }
        _ => return fail(format!("unknown stencil query {key}")),
    }
    clear_last_error();
    RGSADDLE_OK
}

/// # Safety
/// `out` is a report whose `struct_size` is large enough.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_stencil_report(
    handle: *const RgsaddleStencil,
    out: *mut RgsaddleStencilReport,
) -> i32 {
    if handle.is_null() {
        set_last_error("stencil is null");
        return RGSADDLE_NULL_SESSION;
    }
    let report = match check_report(out) {
        Ok(report) => report,
        Err(code) => return code,
    };
    write_report(report, &unsafe { &*handle }.last);
    clear_last_error();
    RGSADDLE_OK
}

/// # Safety
/// `handle` comes from [`rgsaddle_stencil_new`] and is freed once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_stencil_free(handle: *mut RgsaddleStencil) {
    if !handle.is_null() {
        drop(unsafe { Box::from_raw(handle) });
    }
}

#[allow(dead_code)]
fn _cstr_keep(text: &CStr) -> &CStr {
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_keys_match_the_rust_ordinals() {
        let header = include_str!("../../include/rgsaddle.h");
        let define = |name: &str| -> i32 {
            header
                .lines()
                .find_map(|line| {
                    let rest = line.trim().strip_prefix("#define ")?;
                    let mut parts = rest.split_whitespace();
                    let found = parts.next()?;
                    if found == name {
                        let raw = parts.next()?.trim_end_matches('u');
                        Some(raw.parse().unwrap())
                    } else {
                        None
                    }
                })
                .unwrap_or_else(|| panic!("missing {name}"))
        };
        assert_eq!(
            define("RGSADDLE_STENCIL_KEY_METHOD"),
            RGSADDLE_STENCIL_KEY_METHOD
        );
        assert_eq!(define("RGSADDLE_STENCIL_KEY_K"), RGSADDLE_STENCIL_KEY_K);
        assert_eq!(
            define("RGSADDLE_STENCIL_KEY_EXPLICIT_KS"),
            RGSADDLE_STENCIL_KEY_EXPLICIT_KS
        );
        assert_eq!(define("RGSADDLE_STENCIL_METHOD_PLAIN"), 0);
        assert_eq!(define("RGSADDLE_STENCIL_METHOD_ART"), 6);
        assert_eq!(define("RGSADDLE_PRESET_PLAIN"), 7);
        assert_eq!(define("RGSADDLE_STENCIL_PHASE_TRANSLATE"), 7);
        assert_eq!(
            define("RGSADDLE_STENCIL_QUERY_SHAPE"),
            RGSADDLE_STENCIL_QUERY_SHAPE
        );
        assert_eq!(define("RGSADDLE_ABI_MINOR"), 17);
        assert_eq!(
            method_ordinal(&StencilMethod::Band(
                BandMethod::Zoom(ZoomParams::default())
            )),
            define("RGSADDLE_STENCIL_METHOD_ZOOM")
        );
        assert_eq!(
            phase_ordinal(Phase::Zoom),
            define("RGSADDLE_STENCIL_PHASE_ZOOM")
        );
    }
}
