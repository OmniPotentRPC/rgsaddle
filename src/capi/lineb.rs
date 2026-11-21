//! One-shot C ABI for fixed-energy LI-NEB and harmonic quantum TST rates.

use std::ffi::c_void;
use std::slice;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use super::stencil::{clear_last_error, fail, set_last_error};
use super::{
    CSurface, RGSADDLE_INVALID_PARAMETER, RGSADDLE_NULL_REPORT, RGSADDLE_NULL_SURFACE, RGSADDLE_OK,
    RGSADDLE_SHAPE, RGSADDLE_STALLED, RgsaddleSurfaceFn,
};
use crate::error::SaddleError;
use crate::lineb::hqtst_rate_interpolated_hessian_observed;
use crate::lineb::{
    LinebConfig, LinebSession, LinebStatus, Masses, hqtst_rate, hqtst_rate_with_orbit,
};
use crate::{BandSurface, PointSurface};

struct LinebSurface {
    callback: CSurface,
    batch: bool,
}

impl PointSurface for LinebSurface {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        PointSurface::eval(&self.callback, x)
    }

    fn eval_batch(&self, points: ArrayView2<f64>) -> Result<Vec<(f64, Array1<f64>)>, SaddleError> {
        if !self.batch || points.nrows() < 2 {
            return self.callback.eval_batch(points);
        }
        let mut energies = Array1::zeros(points.nrows());
        let mut gradients = Array2::zeros(points.raw_dim());
        BandSurface::eval(&self.callback, points, &mut energies, &mut gradients)?;
        Ok(energies
            .into_iter()
            .zip(gradients.rows().into_iter().map(|row| row.to_owned()))
            .collect())
    }
}

/// Fixed-energy LI-NEB controls for [`rgsaddle_lineb_solve`] (ABI minor 47).
///
/// Field order matches `rgsaddle_lineb_config_t` in `include/rgsaddle.h`.
/// Start from [`rgsaddle_lineb_config_default`].
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RgsaddleLinebConfig {
    /// Stop when every nudged force's norm is at most this.
    pub opt_tol: f64,
    /// Stop when every interior perpendicular action gradient's norm is
    /// at most this.
    pub perp_tol: f64,
    /// Spring constant `k` in the parallel spacing force.
    pub spring: f64,
    /// `κ |f|` for the endpoint equipotential force.
    pub endpoint_scale: f64,
    /// Initial FIRE / quick-min time step.
    pub time_step: f64,
    /// Largest step; zero leaves the step uncapped.
    pub max_move: f64,
    /// Turning-surface regularization width as a fraction of the highest
    /// starting image above `E`.
    pub turning_width: f64,
    /// `ℏ` in the temperature `ℏ / (k_B τ)`.
    pub hbar: f64,
    /// Boltzmann constant in the same temperature.
    pub boltzmann: f64,
    /// Simpson pairs on each chord when the period is integrated.
    pub quadrature_pairs: i64,
}

/// Path, temperature and rate summary from [`rgsaddle_lineb_solve`]
/// (ABI minor 47).
///
/// Field order matches `rgsaddle_lineb_result_t` in `include/rgsaddle.h`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RgsaddleLinebResult {
    /// Spline action `W`.
    pub action: f64,
    /// Half period from the action's energy derivative.
    pub half_period: f64,
    /// Half period from constrained dynamics along the path.
    pub half_period_dynamics: f64,
    /// Instanton temperature from the quadrature half period.
    pub temperature: f64,
    /// Instanton temperature from the dynamics half period.
    pub temperature_dynamics: f64,
    /// `ln(k Z_r)`, or NaN when `beads == 0`.
    pub log_rate_over_reactant: f64,
    /// Optimizer steps completed.
    pub iterations: i64,
    /// Surface evaluations inside optimizer steps.
    pub optimizer_calls: i64,
    /// Surface evaluations in the path quadrature and time table.
    pub measure_calls: i64,
    /// Surface evaluations in the half-period dynamics.
    pub dynamics_calls: i64,
    /// Surface evaluations in the rate, or 0 when `beads == 0`.
    pub rate_calls: i64,
}

fn config_from_c(cfg: &RgsaddleLinebConfig) -> Result<LinebConfig, i32> {
    if cfg.quadrature_pairs < 1 {
        return Err(RGSADDLE_INVALID_PARAMETER);
    }
    let Some(pairs) = usize::try_from(cfg.quadrature_pairs).ok() else {
        return Err(RGSADDLE_INVALID_PARAMETER);
    };
    Ok(LinebConfig::new(cfg.opt_tol, cfg.perp_tol)
        .with_spring(cfg.spring)
        .with_endpoint_scale(cfg.endpoint_scale)
        .with_time_step(cfg.time_step)
        .with_max_move(cfg.max_move)
        .with_turning_width(cfg.turning_width)
        .with_hbar(cfg.hbar)
        .with_boltzmann(cfg.boltzmann)
        .with_quadrature_pairs(pairs))
}

fn masses_from_c(atom_masses: &[f64], molecular: i32) -> Result<Masses, i32> {
    match molecular {
        0 => {
            let mut expanded = Array1::zeros(3 * atom_masses.len());
            for (atom, &mass) in atom_masses.iter().enumerate() {
                for axis in 0..3 {
                    expanded[3 * atom + axis] = mass;
                }
            }
            Masses::coordinates(expanded.view()).map_err(|err| fail(&err))
        }
        1 => Masses::molecule(atom_masses).map_err(|err| fail(&err)),
        2 => Masses::molecule(atom_masses)
            .map(|masses| masses.with_adjacent_alignment(true))
            .map_err(|err| fail(&err)),
        _ => Err(RGSADDLE_INVALID_PARAMETER),
    }
}

fn as_i64(value: usize) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

/// Fill `out` with the solver defaults for the given force tolerances
/// (ABI minor 47).
///
/// # Safety
/// `out` must be a valid writable pointer or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_lineb_config_default(
    opt_tol: f64,
    perp_tol: f64,
    out: *mut RgsaddleLinebConfig,
) -> i32 {
    if out.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let cfg = LinebConfig::new(opt_tol, perp_tol);
    // SAFETY: non-null writable config.
    unsafe {
        *out = RgsaddleLinebConfig {
            opt_tol: cfg.opt_tol,
            perp_tol: cfg.perp_tol,
            spring: cfg.spring,
            endpoint_scale: cfg.endpoint_scale,
            time_step: cfg.time_step,
            max_move: cfg.max_move,
            turning_width: cfg.turning_width,
            hbar: cfg.hbar,
            boltzmann: cfg.boltzmann,
            quadrature_pairs: as_i64(cfg.quadrature_pairs),
        };
    }
    RGSADDLE_OK
}

/// Optimize a fixed-energy tunneling path and optionally its HQ-TST rate
/// (ABI minor 47).
///
/// Positions are contiguous image, atom, xyz (`n_images * n_atoms * 3`).
/// `atom_masses` holds `n_atoms` positive masses: when `molecular` is 0
/// each mass expands to the three Cartesian coordinates with no rigid
/// alignment; when `molecular` is 1, quaternion alignment is used.
/// `beads == 0` returns the path only and writes NaN into
/// `log_rate_over_reactant`. An unconverged path returns the existing
/// non-success status (`RGSADDLE_STALLED`) after writing the last
/// images into `optimized_positions`.
///
/// # Safety
/// `cfg`, `positions`, `atom_masses`, `optimized_positions` and `out`
/// must be valid for the stated sizes when non-null. `surface` is
/// called with `user`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_lineb_solve(
    cfg: *const RgsaddleLinebConfig,
    n_images: i64,
    n_atoms: i64,
    positions: *const f64,
    atom_masses: *const f64,
    molecular: i32,
    energy: f64,
    max_steps: i64,
    beads: i64,
    surface: Option<RgsaddleSurfaceFn>,
    user: *mut c_void,
    optimized_positions: *mut f64,
    out: *mut RgsaddleLinebResult,
) -> i32 {
    // SAFETY: buffers and callback satisfy this entry's contract.
    unsafe {
        rgsaddle_lineb_solve_with_kappa(
            cfg,
            n_images,
            n_atoms,
            positions,
            atom_masses,
            molecular,
            energy,
            max_steps,
            beads,
            surface,
            user,
            optimized_positions,
            out,
            std::ptr::null(),
        )
    }
}

/// Solve with a fixed endpoint coefficient. A null coefficient selects
/// the gradient-normalized restraint in the supplied configuration.
///
/// # Safety
/// The buffer and callback contract matches [`rgsaddle_lineb_solve`].
/// A non-null `endpoint_kappa` must point to a readable double.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_lineb_solve_with_kappa(
    cfg: *const RgsaddleLinebConfig,
    n_images: i64,
    n_atoms: i64,
    positions: *const f64,
    atom_masses: *const f64,
    molecular: i32,
    energy: f64,
    max_steps: i64,
    beads: i64,
    surface: Option<RgsaddleSurfaceFn>,
    user: *mut c_void,
    optimized_positions: *mut f64,
    out: *mut RgsaddleLinebResult,
    endpoint_kappa: *const f64,
) -> i32 {
    // SAFETY: the buffers and coefficient follow the same contract.
    unsafe {
        rgsaddle_lineb_solve_options(
            cfg,
            n_images,
            n_atoms,
            positions,
            atom_masses,
            molecular,
            energy,
            max_steps,
            beads,
            surface,
            user,
            optimized_positions,
            out,
            endpoint_kappa,
            0,
        )
    }
}

/// Solve with configurable seed geometry, convergence, and integrator
/// (ABI minor 47).
///
/// # Safety
/// Pointer lifetimes and sizes follow [`rgsaddle_lineb_solve_with_kappa`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_lineb_solve_options(
    cfg: *const RgsaddleLinebConfig,
    n_images: i64,
    n_atoms: i64,
    positions: *const f64,
    atom_masses: *const f64,
    molecular: i32,
    energy: f64,
    max_steps: i64,
    beads: i64,
    surface: Option<RgsaddleSurfaceFn>,
    user: *mut c_void,
    optimized_positions: *mut f64,
    out: *mut RgsaddleLinebResult,
    endpoint_kappa: *const f64,
    options: u32,
) -> i32 {
    unsafe {
        rgsaddle_lineb_solve_counted(
            cfg,
            n_images,
            n_atoms,
            positions,
            atom_masses,
            molecular,
            energy,
            max_steps,
            beads,
            surface,
            user,
            optimized_positions,
            out,
            endpoint_kappa,
            options,
            std::ptr::null_mut(),
        )
    }
}

/// Disjoint potential surface evaluation counts for a successful solve.
#[repr(C)]
#[derive(Default)]
pub struct RgsaddleLinebCounts {
    /// Rows used to bracket, clip, or extend the input seed.
    pub seed: i64,
    /// Rows evaluated by path optimization steps.
    pub optimizer: i64,
    /// Rows used to assemble and refine the path before integration.
    pub preparation: i64,
    /// Rows used in action and imaginary-time quadrature.
    pub quadrature: i64,
    /// Rows used to construct or advance the trajectory.
    pub dynamics: i64,
    /// Rows used to evaluate the fluctuation prefactor.
    pub rate: i64,
}

/// Solve with optional per-phase evaluation counts.
///
/// # Safety
/// Follows [`rgsaddle_lineb_solve_options`]. A non-null counts pointer must
/// refer to distinct writable storage. Counts describe a successful solve.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_lineb_solve_counted(
    cfg: *const RgsaddleLinebConfig,
    n_images: i64,
    n_atoms: i64,
    positions: *const f64,
    atom_masses: *const f64,
    molecular: i32,
    energy: f64,
    max_steps: i64,
    beads: i64,
    surface: Option<RgsaddleSurfaceFn>,
    user: *mut c_void,
    optimized_positions: *mut f64,
    out: *mut RgsaddleLinebResult,
    endpoint_kappa: *const f64,
    options: u32,
    counts: *mut RgsaddleLinebCounts,
) -> i32 {
    unsafe {
        rgsaddle_lineb_solve_counted_diagnostics(
            cfg,
            n_images,
            n_atoms,
            positions,
            atom_masses,
            molecular,
            energy,
            max_steps,
            beads,
            surface,
            user,
            optimized_positions,
            out,
            endpoint_kappa,
            options,
            counts,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    }
}

/// Solve with per-phase counts and the inertia of the retained fluctuation operator.
/// A path-only solve or an error writes -1 to each requested mode count.
///
/// # Safety
/// Follows [`rgsaddle_lineb_solve_counted`]. Each non-null diagnostic pointer must
/// refer to distinct writable storage for one signed 64-bit integer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_lineb_solve_counted_diagnostics(
    cfg: *const RgsaddleLinebConfig,
    n_images: i64,
    n_atoms: i64,
    positions: *const f64,
    atom_masses: *const f64,
    molecular: i32,
    energy: f64,
    max_steps: i64,
    beads: i64,
    surface: Option<RgsaddleSurfaceFn>,
    user: *mut c_void,
    optimized_positions: *mut f64,
    out: *mut RgsaddleLinebResult,
    endpoint_kappa: *const f64,
    options: u32,
    counts: *mut RgsaddleLinebCounts,
    negative_modes: *mut i64,
    modes: *mut i64,
) -> i32 {
    unsafe {
        rgsaddle_lineb_solve_counted_orbit(
            cfg,
            n_images,
            n_atoms,
            positions,
            atom_masses,
            molecular,
            energy,
            max_steps,
            beads,
            surface,
            user,
            optimized_positions,
            out,
            endpoint_kappa,
            options,
            counts,
            negative_modes,
            modes,
            std::ptr::null_mut(),
        )
    }
}

/// Caller-owned Cartesian output buffers and discrete force residuals.
#[repr(C)]
pub struct RgsaddleLinebOrbit {
    /// Number of writable doubles in each supplied buffer.
    pub capacity: usize,
    /// Optional row-major bead positions.
    pub positions: *mut f64,
    /// Optional row-major bead gradient observations.
    pub gradients: *mut f64,
    /// Maximum absolute force residual in energy per length.
    pub max_abs_residual: f64,
    /// RMS force residual in energy per length.
    pub rms_residual: f64,
    /// Bead of the maximum residual, or -1 without a rate.
    pub max_bead: i64,
    /// Active Cartesian component of the maximum, or -1 without a rate.
    pub max_coordinate: i64,
}

/// Counted solve with optional observations from the rate's surface calls.
///
/// # Safety
/// The arguments obey the counted solve contract. A non-null orbit points to
/// distinct writable storage. Its optional buffers each hold capacity doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_lineb_solve_counted_orbit(
    cfg: *const RgsaddleLinebConfig,
    n_images: i64,
    n_atoms: i64,
    positions: *const f64,
    atom_masses: *const f64,
    molecular: i32,
    energy: f64,
    max_steps: i64,
    beads: i64,
    surface: Option<RgsaddleSurfaceFn>,
    user: *mut c_void,
    optimized_positions: *mut f64,
    out: *mut RgsaddleLinebResult,
    endpoint_kappa: *const f64,
    options: u32,
    counts: *mut RgsaddleLinebCounts,
    negative_modes: *mut i64,
    modes: *mut i64,
    orbit: *mut RgsaddleLinebOrbit,
) -> i32 {
    unsafe {
        rgsaddle_lineb_solve_counted_turning(
            cfg,
            n_images,
            n_atoms,
            positions,
            atom_masses,
            molecular,
            energy,
            max_steps,
            beads,
            surface,
            user,
            optimized_positions,
            out,
            endpoint_kappa,
            options,
            counts,
            negative_modes,
            modes,
            orbit,
            std::ptr::null_mut(),
        )
    }
}

/// Endpoint agreement of the active force-interpolated orbit.
#[repr(C)]
pub struct RgsaddleLinebTurning {
    /// ABI version used to produce this result.
    pub version: super::RgsaddleVersion,
    /// Reserved flags, zero.
    pub flags: u64,
    /// Index of the launch image, or -1 without a measured return turn.
    pub launch_image: i64,
    /// Mass-weighted arc length of the path.
    pub path_arc_length: f64,
    /// Return position in the same arc coordinate.
    pub return_arc: f64,
    /// Largest absolute Cartesian difference from the opposite image.
    pub max_cartesian_displacement: f64,
    /// Measured return potential minus the target energy.
    pub energy_residual: f64,
    /// Measured return potential minus that of the opposite image.
    pub endpoint_energy_difference: f64,
}

/// Solve with optional measurements of the force-interpolated return turn.
///
/// # Safety
/// Follows [`rgsaddle_lineb_solve_counted_orbit`]. A non-null turning pointer
/// must address one distinct writable result structure.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_lineb_solve_counted_turning(
    cfg: *const RgsaddleLinebConfig,
    n_images: i64,
    n_atoms: i64,
    positions: *const f64,
    atom_masses: *const f64,
    molecular: i32,
    energy: f64,
    max_steps: i64,
    beads: i64,
    surface: Option<RgsaddleSurfaceFn>,
    user: *mut c_void,
    optimized_positions: *mut f64,
    out: *mut RgsaddleLinebResult,
    endpoint_kappa: *const f64,
    options: u32,
    counts: *mut RgsaddleLinebCounts,
    negative_modes: *mut i64,
    modes: *mut i64,
    orbit: *mut RgsaddleLinebOrbit,
    turning: *mut RgsaddleLinebTurning,
) -> i32 {
    unsafe {
        rgsaddle_lineb_solve_counted_rate(
            cfg,
            n_images,
            n_atoms,
            positions,
            atom_masses,
            molecular,
            energy,
            max_steps,
            beads,
            surface,
            user,
            optimized_positions,
            out,
            endpoint_kappa,
            options,
            counts,
            negative_modes,
            modes,
            orbit,
            turning,
            std::ptr::null(),
            std::ptr::null_mut(),
        )
    }
}

/// Surface work and applied displacement of the selected harmonic rate.
#[repr(C)]
pub struct RgsaddleLinebRateDiagnostics {
    /// ABI version used to produce this result.
    pub version: super::RgsaddleVersion,
    /// Reserved flags, zero.
    pub flags: u64,
    /// Geometries evaluated to form Hessian blocks.
    pub curvature_evaluations: i64,
    /// Replica geometries queried for energies and gradients.
    pub bead_evaluations: i64,
    /// Actual Cartesian image-Hessian step, or NaN if no image Hessians were taken.
    pub image_hessian_step: f64,
}

/// Solve with bead or interpolated image Hessians and optional rate diagnostics.
///
/// # Safety
/// Follows [`rgsaddle_lineb_solve_counted_turning`]. Non-null step and diagnostics
/// pointers must address one readable double and one distinct writable structure.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_lineb_solve_counted_rate(
    cfg: *const RgsaddleLinebConfig,
    n_images: i64,
    n_atoms: i64,
    positions: *const f64,
    atom_masses: *const f64,
    molecular: i32,
    energy: f64,
    max_steps: i64,
    beads: i64,
    surface: Option<RgsaddleSurfaceFn>,
    user: *mut c_void,
    optimized_positions: *mut f64,
    out: *mut RgsaddleLinebResult,
    endpoint_kappa: *const f64,
    options: u32,
    counts: *mut RgsaddleLinebCounts,
    negative_modes: *mut i64,
    modes: *mut i64,
    orbit: *mut RgsaddleLinebOrbit,
    turning: *mut RgsaddleLinebTurning,
    image_hessian_step: *const f64,
    rate_diagnostics: *mut RgsaddleLinebRateDiagnostics,
) -> i32 {
    let mut rate_diagnostics = unsafe { rate_diagnostics.as_mut() };
    let image_hessian_step = unsafe { image_hessian_step.as_ref() }.copied();
    if let Some(d) = rate_diagnostics.as_deref_mut() {
        *d = RgsaddleLinebRateDiagnostics {
            version: super::RgsaddleVersion {
                major: super::RGSADDLE_ABI_MAJOR,
                minor: super::RGSADDLE_ABI_MINOR,
            },
            flags: 0,
            curvature_evaluations: 0,
            bead_evaluations: 0,
            image_hessian_step: f64::NAN,
        };
    }
    let mut turning = unsafe { turning.as_mut() };
    if let Some(t) = turning.as_deref_mut() {
        *t = RgsaddleLinebTurning {
            version: super::RgsaddleVersion {
                major: super::RGSADDLE_ABI_MAJOR,
                minor: super::RGSADDLE_ABI_MINOR,
            },
            flags: 0,
            launch_image: -1,
            path_arc_length: f64::NAN,
            return_arc: f64::NAN,
            max_cartesian_displacement: f64::NAN,
            energy_residual: f64::NAN,
            endpoint_energy_difference: f64::NAN,
        };
    }
    let mut orbit = unsafe { orbit.as_mut() };
    if let Some(o) = orbit.as_deref_mut() {
        o.max_abs_residual = f64::NAN;
        o.rms_residual = f64::NAN;
        o.max_bead = -1;
        o.max_coordinate = -1;
    }
    let mut negative_modes = unsafe { negative_modes.as_mut() };
    let mut modes = unsafe { modes.as_mut() };
    if let Some(value) = negative_modes.as_deref_mut() {
        *value = -1;
    }
    if let Some(value) = modes.as_deref_mut() {
        *value = -1;
    }
    let counts = unsafe { counts.as_mut() };
    let mut discarded = RgsaddleLinebCounts::default();
    let counts = counts.unwrap_or(&mut discarded);
    *counts = RgsaddleLinebCounts::default();
    clear_last_error();
    const LINEAR_SEED: u32 = 1;
    const ENDPOINT_CONVERGENCE: u32 = 2;
    const PROJECTED_VERLET: u32 = 4;
    const INTERPOLATED_FORCE_CLOCK: u32 = 8;
    const INTERPOLATED_HESSIAN: u32 = 16;
    const BATCH_SURFACE: u32 = 32;
    let interpolate = options & INTERPOLATED_HESSIAN != 0;
    if image_hessian_step.is_some_and(|v| !interpolate || !v.is_finite() || v <= 0.) {
        set_last_error(
            "image Hessian step requires interpolated Hessians and a positive finite displacement",
        );
        return RGSADDLE_INVALID_PARAMETER;
    }
    if options
        & !(LINEAR_SEED
            | ENDPOINT_CONVERGENCE
            | PROJECTED_VERLET
            | INTERPOLATED_FORCE_CLOCK
            | INTERPOLATED_HESSIAN
            | BATCH_SURFACE)
        != 0
    {
        set_last_error("unknown LI-NEB solver option");
        return RGSADDLE_INVALID_PARAMETER;
    }
    let endpoint_convergence = options & ENDPOINT_CONVERGENCE != 0;
    let projected_verlet = options & PROJECTED_VERLET != 0;
    if cfg.is_null()
        || positions.is_null()
        || atom_masses.is_null()
        || optimized_positions.is_null()
    {
        return RGSADDLE_INVALID_PARAMETER;
    }
    if out.is_null() {
        return RGSADDLE_NULL_REPORT;
    }
    let Some(f) = surface else {
        return RGSADDLE_NULL_SURFACE;
    };
    if n_images < 3 || n_atoms < 1 {
        return RGSADDLE_SHAPE;
    }
    if max_steps < 1 || beads < 0 || beads == 1 {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let Ok(n_images_usize) = usize::try_from(n_images) else {
        return RGSADDLE_SHAPE;
    };
    let Ok(n_atoms_usize) = usize::try_from(n_atoms) else {
        return RGSADDLE_SHAPE;
    };
    let Ok(max_steps_usize) = usize::try_from(max_steps) else {
        return RGSADDLE_INVALID_PARAMETER;
    };
    let Ok(beads_usize) = usize::try_from(beads) else {
        return RGSADDLE_INVALID_PARAMETER;
    };
    let dof = match n_atoms_usize.checked_mul(3) {
        Some(dof) => dof,
        None => return RGSADDLE_SHAPE,
    };
    if let Some(o) = orbit.as_deref() {
        let Some(n) = beads_usize.checked_mul(dof) else {
            return RGSADDLE_SHAPE;
        };
        if std::alloc::Layout::array::<f64>(n).is_err()
            || ((!o.positions.is_null() || !o.gradients.is_null()) && o.capacity < n)
        {
            set_last_error("LI orbit buffer cannot hold all bead coordinates");
            return RGSADDLE_SHAPE;
        }
    }
    let n_values = match n_images_usize.checked_mul(dof) {
        Some(n) if std::alloc::Layout::array::<f64>(n).is_ok() => n,
        _ => return RGSADDLE_SHAPE,
    };

    // SAFETY: pointers non-null; lengths checked above.
    let cfg = unsafe { &*cfg };
    let positions = unsafe { slice::from_raw_parts(positions, n_values) };
    let atom_masses = unsafe { slice::from_raw_parts(atom_masses, n_atoms_usize) };
    let out = unsafe { &mut *out };

    *out = RgsaddleLinebResult {
        action: f64::NAN,
        half_period: f64::NAN,
        half_period_dynamics: f64::NAN,
        temperature: f64::NAN,
        temperature_dynamics: f64::NAN,
        log_rate_over_reactant: f64::NAN,
        iterations: 0,
        optimizer_calls: 0,
        measure_calls: 0,
        dynamics_calls: 0,
        rate_calls: 0,
    };

    let config = match config_from_c(cfg) {
        Ok(config) => config,
        Err(status) => return status,
    };
    // SAFETY: the caller provides a readable coefficient or a null pointer.
    let config = match unsafe { endpoint_kappa.as_ref() } {
        Some(&kappa) if kappa.is_finite() && kappa > 0.0 => config.with_endpoint_kappa(kappa),
        Some(_) => {
            set_last_error("endpoint kappa must be finite and positive");
            return RGSADDLE_INVALID_PARAMETER;
        }
        None => config,
    };
    let masses = match masses_from_c(atom_masses, molecular) {
        Ok(masses) => masses,
        Err(status) => return status,
    };
    let cartesian = match Array2::from_shape_vec((n_images_usize, dof), positions.to_vec()) {
        Ok(cartesian) => cartesian,
        Err(_) => return RGSADDLE_SHAPE,
    };
    let cs = LinebSurface {
        callback: CSurface {
            f,
            user,
            n_atoms,
            n_images,
            per_image: false,
            climbing_request: false,
            rows: Default::default(),
        },
        batch: options & BATCH_SURFACE != 0,
    };
    let config = config
        .with_endpoint_convergence(endpoint_convergence)
        .with_projected_verlet(projected_verlet)
        .with_interpolated_force_clock(options & INTERPOLATED_FORCE_CLOCK != 0);
    let mut session = match LinebSession::from_seed_with_interpolation(
        &cs,
        config,
        masses,
        cartesian,
        energy,
        options & LINEAR_SEED != 0,
    ) {
        Ok(session) => session,
        Err(err) => return fail(&err),
    };
    let outcome = session.run(&cs, max_steps_usize);
    out.iterations = as_i64(session.iteration());
    out.optimizer_calls = as_i64(cs.callback.rows.load(std::sync::atomic::Ordering::Relaxed));
    let cartesian = match session.cartesian() {
        Ok(cartesian) => cartesian,
        Err(err) => return fail(&err),
    };
    // The session owns its input by this point. An in-place caller can
    // therefore use its seed buffer for the optimized coordinates.
    let optimized = unsafe { slice::from_raw_parts_mut(optimized_positions, n_values) };
    for (dst, src) in optimized.iter_mut().zip(cartesian.iter()) {
        *dst = *src;
    }
    let report = match outcome {
        Ok(report) => report,
        Err(err) => return fail(&err),
    };
    if report.status != LinebStatus::Converged {
        set_last_error(&format!(
            "LI-NEB exhausted {} iterations: nudged force {} (tolerance {}), perpendicular force {} (tolerance {})",
            report.iteration,
            if endpoint_convergence {
                report.endpoint_max
            } else {
                report.opt_max
            },
            cfg.opt_tol,
            report.perp_max,
            cfg.perp_tol
        ));
        return RGSADDLE_STALLED;
    }
    let path = match session.path(&cs) {
        Ok(path) => path,
        Err(err) => return fail(&err),
    };
    for (dst, src) in optimized.iter_mut().zip(path.cartesian.iter()) {
        *dst = *src;
    }
    if let (Some(t), Some(measured)) = (turning, path.turning) {
        t.launch_image = as_i64(measured.launch_image);
        t.path_arc_length = measured.path_arc_length;
        t.return_arc = measured.return_arc;
        t.max_cartesian_displacement = measured.max_cartesian_displacement;
        t.energy_residual = measured.energy_residual;
        t.endpoint_energy_difference = measured.endpoint_energy_difference;
    }
    counts.seed = as_i64(path.calls.seed);
    counts.optimizer = as_i64(path.calls.steps());
    counts.preparation = as_i64(path.calls.preparation);
    counts.quadrature = as_i64(path.calls.quadrature());
    counts.dynamics = as_i64(path.calls.dynamics);
    out.action = path.action;
    out.half_period = path.half_period;
    out.half_period_dynamics = path.half_period_dynamics;
    out.temperature = path.temperature;
    out.temperature_dynamics = path.temperature_dynamics;
    out.optimizer_calls = as_i64(path.calls.optimizer);
    out.measure_calls = as_i64(path.calls.measure);
    out.dynamics_calls = as_i64(path.calls.dynamics);
    if beads_usize == 0 {
        out.log_rate_over_reactant = f64::NAN;
        out.rate_calls = 0;
        return RGSADDLE_OK;
    }
    let rate = if interpolate {
        hqtst_rate_interpolated_hessian_observed(
            &cs,
            &path,
            beads_usize,
            image_hessian_step,
            orbit.is_some(),
        )
    } else if orbit.is_some() {
        hqtst_rate_with_orbit(&cs, &path, beads_usize)
            .map(|(rate, observed)| (rate, Some(observed)))
    } else {
        hqtst_rate(&cs, &path, beads_usize).map(|rate| (rate, None))
    };
    match rate {
        Ok((rate, observed)) => {
            if let (Some(o), Some(observed)) = (orbit, observed) {
                for (dst, values) in [
                    (o.positions, &observed.positions),
                    (o.gradients, &observed.gradients),
                ] {
                    if !dst.is_null() {
                        unsafe {
                            std::ptr::copy_nonoverlapping(values.as_ptr(), dst, values.len())
                        };
                    }
                }
                o.max_abs_residual = observed.max_abs_residual;
                o.rms_residual = observed.rms_residual;
                o.max_bead = as_i64(observed.max_bead);
                o.max_coordinate = as_i64(observed.max_coordinate);
            }
            if let Some(d) = rate_diagnostics {
                d.curvature_evaluations = as_i64(rate.curvature_evaluations);
                d.bead_evaluations = as_i64(rate.bead_evaluations);
                d.image_hessian_step = rate.image_hessian_displacement.unwrap_or(f64::NAN);
            }
            out.log_rate_over_reactant = rate.log_rate_over_reactant;
            out.rate_calls = as_i64(rate.evaluations);
            counts.rate = out.rate_calls;
            if let Some(value) = negative_modes {
                *value = as_i64(rate.negative_modes);
            }
            if let Some(value) = modes {
                *value = as_i64(rate.modes);
            }
            RGSADDLE_OK
        }
        Err(err) => fail(&err),
    }
}
