//! Atom and image counts whose double count does not fit a slice are
//! refused with an error code before any host buffer is read.
//!
//! `i64::MAX / 4` atoms give a `3 * n_atoms` that fits a 64-bit `usize`
//! while its byte length does not, the case a truncating or unchecked
//! cast lets through to `slice::from_raw_parts`.

#![cfg(feature = "capi")]

use std::ffi::c_void;
use std::ptr;

use rgsaddle::capi::*;

const HUGE_ATOMS: i64 = i64::MAX / 4;

extern "C" fn never_called(_user: *mut c_void, _req: *mut RgsaddleSurfaceRequest) -> i32 {
    panic!("the surface must not be entered for a refused size");
}

fn version() -> RgsaddleVersion {
    RgsaddleVersion {
        major: RGSADDLE_ABI_MAJOR,
        minor: RGSADDLE_ABI_MINOR,
    }
}

fn autoneb_config() -> RgsaddleAutonebConfig {
    RgsaddleAutonebConfig {
        version: version(),
        flags: 0,
        projection: 1,
        tangent: 1,
        n_simul: 1,
        n_max: 3,
        local_max_steps: 10,
        climb_max_steps: 10,
        k: 1.0,
        force_tol: 1e-3,
        local_force_tol: 1e-2,
        max_move: 0.1,
        space_energy_ratio: 0.5,
    }
}

fn minmode_config(max_rotations: i64) -> RgsaddleMinModeConfig {
    RgsaddleMinModeConfig {
        version: version(),
        flags: 0,
        kind: 0,
        method: 0,
        dr: 1e-3,
        rotation_tol: 1e-4,
        max_rotations,
        krylov_dim: 3,
        force_tol: 1e-4,
        max_move: 0.1,
        rotation_angle_tol: 5f64.to_radians(),
        difference: 0,
        reserved: 0,
    }
}

fn band_config() -> RgsaddleBandConfig {
    RgsaddleBandConfig {
        version: version(),
        flags: 0,
        tangent: 1,
        spring: 0,
        projection: 1,
        method: 0,
        spring_k: 5.0,
        spring_ks: ptr::null(),
        ci_trigger_factor: 0.5,
        ci_trigger_force: 0.0,
        cell: ptr::null(),
        force_tol: 1e-3,
        max_move: 0.1,
        memory: 0,
    }
}

#[test]
fn autoneb_run_refuses_an_atom_count_past_a_slice() {
    let config = autoneb_config();
    let positions = [0.0; 6];
    let mut out = [0.0; 9];
    let mut report = std::mem::MaybeUninit::<RgsaddleAutonebReport>::uninit();
    for n_atoms in [HUGE_ATOMS, i64::MAX] {
        let rc = unsafe {
            rgsaddle_autoneb_run(
                &config,
                2,
                n_atoms,
                positions.as_ptr(),
                ptr::null(),
                ptr::null(),
                out.as_mut_ptr(),
                report.as_mut_ptr(),
                Some(never_called),
                ptr::null_mut(),
            )
        };
        assert_eq!(rc, RGSADDLE_SHAPE, "n_atoms {n_atoms}");
    }
}

#[test]
fn autoneb_run_refuses_an_image_count_past_a_slice() {
    let mut config = autoneb_config();
    config.n_max = i64::MAX / 2;
    let positions = [0.0; 6];
    let mut out = [0.0; 9];
    let mut report = std::mem::MaybeUninit::<RgsaddleAutonebReport>::uninit();
    let rc = unsafe {
        rgsaddle_autoneb_run(
            &config,
            2,
            1,
            positions.as_ptr(),
            ptr::null(),
            ptr::null(),
            out.as_mut_ptr(),
            report.as_mut_ptr(),
            Some(never_called),
            ptr::null_mut(),
        )
    };
    assert_eq!(rc, RGSADDLE_SHAPE);
}

#[test]
fn minmode_create_refuses_an_atom_count_past_a_slice() {
    let config = minmode_config(20);
    let x = [0.0; 3];
    let mode = [1.0, 0.0, 0.0];
    let session =
        unsafe { rgsaddle_minmode_create(&config, HUGE_ATOMS, x.as_ptr(), mode.as_ptr()) };
    assert!(session.is_null());
}

#[test]
fn minmode_create_refuses_a_negative_rotation_cap() {
    let config = minmode_config(-1);
    let x = [0.0; 3];
    let mode = [1.0, 0.0, 0.0];
    let session = unsafe { rgsaddle_minmode_create(&config, 1, x.as_ptr(), mode.as_ptr()) };
    assert!(session.is_null());
}

#[test]
fn band_create_refuses_an_atom_count_past_a_slice() {
    let config = band_config();
    let positions = [0.0; 9];
    let band = unsafe { rgsaddle_band_create(&config, 3, HUGE_ATOMS, positions.as_ptr()) };
    assert!(band.is_null());
}

#[test]
fn dyneb_mask_refuses_an_atom_count_past_a_slice() {
    let positions = [0.0; 9];
    let energies = [0.0; 3];
    let mut forces = [0.0; 9];
    let rc = unsafe {
        rgsaddle_dyneb_mask(
            3,
            HUGE_ATOMS,
            positions.as_ptr(),
            energies.as_ptr(),
            forces.as_mut_ptr(),
            0.05,
            0.0,
            ptr::null_mut(),
        )
    };
    assert_eq!(rc, RGSADDLE_SHAPE);
}

#[test]
fn idpp_path_refuses_an_atom_count_past_a_slice() {
    let reactant = [0.0; 3];
    let product = [1.0, 0.0, 0.0];
    let mut out = [0.0; 9];
    let rc = unsafe {
        rgsaddle_idpp_path(
            3,
            HUGE_ATOMS,
            reactant.as_ptr(),
            product.as_ptr(),
            ptr::null(),
            10,
            1e-3,
            0.1,
            out.as_mut_ptr(),
        )
    };
    assert_eq!(rc, RGSADDLE_SHAPE);
}

#[test]
fn prfo_step_refuses_a_hessian_past_a_slice() {
    // `n * n` fits a 64-bit `usize`, its byte length does not.
    let gradient = [0.0; 1];
    let hessian = [1.0; 1];
    let mut displacement = [0.0; 1];
    let rc = unsafe {
        rgsaddle_prfo_step(
            1 << 31,
            gradient.as_ptr(),
            hessian.as_ptr(),
            ptr::null(),
            0.1,
            0,
            displacement.as_mut_ptr(),
        )
    };
    assert_eq!(rc, RGSADDLE_INVALID_PARAMETER);
}
