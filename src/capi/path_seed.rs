//! Masked and sequential IDPP paths (ABI minor 71).

use std::slice;

use ndarray::{Array1, Array2};

use super::{RGSADDLE_INVALID_PARAMETER, RGSADDLE_OK, RGSADDLE_SHAPE, fixed_mask, status_of};
use crate::error::SaddleError;
use crate::idpp::{
    IdppConfig, SidppAlternatingConfig, idpp_path_fixed, sidpp_alternating_inverse_distance_path,
    sidpp_alternating_path,
};
use crate::mic::Cell;

struct Inputs {
    reactant: Array1<f64>,
    product: Array1<f64>,
    cell: Option<Cell>,
    fixed: Vec<bool>,
    n_images: usize,
}

fn fits(count: usize) -> Result<(), i32> {
    match count.checked_mul(size_of::<f64>()) {
        Some(bytes) if bytes <= isize::MAX as usize => Ok(()),
        _ => Err(RGSADDLE_INVALID_PARAMETER),
    }
}

/// Row-major 3x3 lattice, or null for free space.
///
/// # Safety
/// A non-null `ptr` addresses 9 doubles.
unsafe fn cell_from(ptr: *const f64) -> Result<Option<Cell>, i32> {
    if ptr.is_null() {
        return Ok(None);
    }
    let raw = unsafe { slice::from_raw_parts(ptr, 9) };
    if !raw.iter().all(|v| v.is_finite()) {
        return Err(RGSADDLE_INVALID_PARAMETER);
    }
    Cell::from_vectors(
        [raw[0], raw[1], raw[2]],
        [raw[3], raw[4], raw[5]],
        [raw[6], raw[7], raw[8]],
        [0.0, 0.0, 0.0],
    )
    .map(Some)
    .map_err(|_| RGSADDLE_INVALID_PARAMETER)
}

/// # Safety
/// `reactant` and `product` each address `3 * n_atoms` doubles when the
/// shape checks pass. A non-null `mask` addresses `n_atoms` bytes. A
/// non-null `cell` addresses 9 doubles.
unsafe fn read_inputs(
    n_images: i64,
    n_atoms: i64,
    reactant: *const f64,
    product: *const f64,
    cell: *const f64,
    mask: *const u8,
) -> Result<Inputs, i32> {
    if reactant.is_null() || product.is_null() {
        return Err(RGSADDLE_INVALID_PARAMETER);
    }
    if n_images < 3 || n_atoms < 1 {
        return Err(RGSADDLE_SHAPE);
    }
    let n_images_us = usize::try_from(n_images).map_err(|_| RGSADDLE_SHAPE)?;
    let n_atoms_us = usize::try_from(n_atoms).map_err(|_| RGSADDLE_SHAPE)?;
    let dof = n_atoms_us
        .checked_mul(3)
        .ok_or(RGSADDLE_INVALID_PARAMETER)?;
    fits(dof)?;
    let total = dof
        .checked_mul(n_images_us)
        .ok_or(RGSADDLE_INVALID_PARAMETER)?;
    fits(total)?;
    let reactant = unsafe { slice::from_raw_parts(reactant, dof) }.to_vec();
    let product = unsafe { slice::from_raw_parts(product, dof) }.to_vec();
    let cell = unsafe { cell_from(cell) }?;
    let fixed = if mask.is_null() {
        Vec::new()
    } else {
        unsafe { fixed_mask(mask, n_atoms) }.ok_or(RGSADDLE_INVALID_PARAMETER)?
    };
    Ok(Inputs {
        reactant: Array1::from_vec(reactant),
        product: Array1::from_vec(product),
        cell,
        fixed,
        n_images: n_images_us,
    })
}

/// # Safety
/// `out` addresses `path.len()` doubles.
unsafe fn write_path(path: &Array2<f64>, out: *mut f64) {
    let dst = unsafe { slice::from_raw_parts_mut(out, path.len()) };
    for (i, value) in path.iter().enumerate() {
        dst[i] = *value;
    }
}

fn finish(result: Result<Array2<f64>, SaddleError>, out: *mut f64) -> i32 {
    match result {
        Ok(path) => {
            unsafe { write_path(&path, out) };
            RGSADDLE_OK
        }
        Err(err) => status_of(&err),
    }
}

fn count_of(value: i64) -> Result<usize, i32> {
    usize::try_from(value).map_err(|_| RGSADDLE_INVALID_PARAMETER)
}

#[allow(clippy::too_many_arguments)]
fn alternating_config(
    alpha: f64,
    short_iterations: i64,
    final_iterations: i64,
    force_tol: f64,
    max_move: f64,
    spring: f64,
    inverse: bool,
    pull: bool,
) -> Result<SidppAlternatingConfig, i32> {
    let short_iterations = count_of(short_iterations)?;
    let final_iterations = count_of(final_iterations)?;
    let mut config = SidppAlternatingConfig::default()
        .with_alpha(alpha)
        .with_short_iterations(short_iterations)
        .with_final_iterations(final_iterations)
        .with_force_tol(force_tol)
        .with_max_move(max_move)
        .with_spring(spring)
        .with_reparametrize(true)
        .with_inverse_distance(inverse)
        .with_pull_to_target(pull);
    if pull {
        config = config.with_pull(
            IdppConfig::default()
                .with_max_iterations(final_iterations)
                .with_grad_tol(force_tol)
                .with_max_move(max_move),
        );
    }
    Ok(config)
}

/// IDPP path with a frozen-atom mask. See `rgsaddle_idpp_fixed_path`.
///
/// # Safety
/// `reactant` and `product` each address `3 * n_atoms` doubles. `out`
/// addresses `n_images * 3 * n_atoms` doubles. `cell` is 9 row-major
/// lattice entries or null. `mask` is `n_atoms` bytes or null.
#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_idpp_fixed_path(
    n_images: i64,
    n_atoms: i64,
    reactant: *const f64,
    product: *const f64,
    cell: *const f64,
    mask: *const u8,
    max_iterations: i64,
    grad_tol: f64,
    max_move: f64,
    out: *mut f64,
) -> i32 {
    if out.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let inputs = match unsafe { read_inputs(n_images, n_atoms, reactant, product, cell, mask) } {
        Ok(inputs) => inputs,
        Err(code) => return code,
    };
    let config = IdppConfig::default()
        .with_max_iterations(usize::try_from(max_iterations).unwrap_or(0))
        .with_grad_tol(grad_tol)
        .with_max_move(max_move);
    finish(
        idpp_path_fixed(
            inputs.reactant.view(),
            inputs.product.view(),
            inputs.n_images,
            inputs.cell.as_ref(),
            &inputs.fixed,
            &config,
        ),
        out,
    )
}

/// Sequential IDPP path. See `rgsaddle_sidpp_alternating_path`.
///
/// # Safety
/// `reactant` and `product` each address `3 * n_atoms` doubles. `out`
/// addresses `n_images * 3 * n_atoms` doubles. `cell` is 9 row-major
/// lattice entries or null. `mask` is `n_atoms` bytes or null.
#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_sidpp_alternating_path(
    n_images: i64,
    n_atoms: i64,
    reactant: *const f64,
    product: *const f64,
    cell: *const f64,
    mask: *const u8,
    alpha: f64,
    short_iterations: i64,
    final_iterations: i64,
    force_tol: f64,
    max_move: f64,
    spring: f64,
    out: *mut f64,
) -> i32 {
    if out.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let inputs = match unsafe { read_inputs(n_images, n_atoms, reactant, product, cell, mask) } {
        Ok(inputs) => inputs,
        Err(code) => return code,
    };
    let config = match alternating_config(
        alpha,
        short_iterations,
        final_iterations,
        force_tol,
        max_move,
        spring,
        false,
        false,
    ) {
        Ok(config) => config,
        Err(code) => return code,
    };
    finish(
        sidpp_alternating_path(
            inputs.reactant.view(),
            inputs.product.view(),
            inputs.n_images,
            inputs.cell.as_ref(),
            &inputs.fixed,
            &config,
        ),
        out,
    )
}

/// Inverse-distance sequential path. See `rgsaddle_sidpp_alternating_inverse_distance_path`.
///
/// # Safety
/// `reactant` and `product` each address `3 * n_atoms` doubles. `out`
/// addresses `n_images * 3 * n_atoms` doubles. `cell` is 9 row-major
/// lattice entries or null. `mask` is `n_atoms` bytes or null.
#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_sidpp_alternating_inverse_distance_path(
    n_images: i64,
    n_atoms: i64,
    reactant: *const f64,
    product: *const f64,
    cell: *const f64,
    mask: *const u8,
    alpha: f64,
    short_iterations: i64,
    final_iterations: i64,
    force_tol: f64,
    max_move: f64,
    spring: f64,
    out: *mut f64,
) -> i32 {
    if out.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let inputs = match unsafe { read_inputs(n_images, n_atoms, reactant, product, cell, mask) } {
        Ok(inputs) => inputs,
        Err(code) => return code,
    };
    let config = match alternating_config(
        alpha,
        short_iterations,
        final_iterations,
        force_tol,
        max_move,
        spring,
        true,
        true,
    ) {
        Ok(config) => config,
        Err(code) => return code,
    };
    finish(
        sidpp_alternating_inverse_distance_path(
            inputs.reactant.view(),
            inputs.product.view(),
            inputs.n_images,
            inputs.cell.as_ref(),
            &inputs.fixed,
            &config,
        ),
        out,
    )
}
