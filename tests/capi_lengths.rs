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

fn irc_config() -> RgsaddleIrcConfig {
    RgsaddleIrcConfig {
        version: version(),
        flags: 0,
        dx: 0.1,
        force_tol: 1e-3,
        force_gate: 0,
        max_move: 0.1,
        max_inner: 1,
        kind: 0,
        direction: 0,
    }
}

fn sella_min_config() -> RgsaddleSellaMinConfig {
    RgsaddleSellaMinConfig {
        version: version(),
        flags: 0,
        delta: 0.1,
        force_tol: 1e-3,
        force_gate: 0,
    }
}

fn sella_saddle_config() -> RgsaddleSellaSaddleConfig {
    RgsaddleSellaSaddleConfig {
        version: version(),
        flags: 0,
        delta: 0.1,
        force_tol: 1e-3,
        force_gate: 0,
        order: 1,
    }
}

fn constraints_config() -> RgsaddleConstraintsConfig {
    RgsaddleConstraintsConfig {
        version: version(),
        flags: 0,
    }
}

fn samd_config() -> RgsaddleSamdConfig {
    RgsaddleSamdConfig {
        version: version(),
        flags: 0,
        dt: 1e-3,
        tau: 1.0,
        t0: 1.0,
        tf: 2.0,
        ngen: 1,
        exponential: 0,
    }
}

fn indexk_config() -> RgsaddleIndexKConfig {
    RgsaddleIndexKConfig {
        version: version(),
        flags: 0,
        index: 1,
        method: 0,
        dr: 1e-3,
        rotation_tol: 1e-4,
        max_rotations: 1,
        force_tol: 1e-3,
        max_move: 0.1,
        residual_tol: 1e-4,
        max_block_iterations: 1,
    }
}

/// Wire layout of `rgsaddle_ocineb_config_t`. The Rust type stays in a
/// private module, so the length tests call the C symbol directly.
#[repr(C)]
struct OcinebConfigWire {
    version: RgsaddleVersion,
    flags: u64,
    trigger_factor: f64,
    trigger_force: f64,
    angle_tol: f64,
    stability_count: i64,
    max_mmf_steps: i64,
    restore_unhelpful: i32,
    reserved: i32,
}

unsafe extern "C" {
    fn rgsaddle_ocineb_create(
        band: *const RgsaddleBandConfig,
        minmode: *const RgsaddleMinModeConfig,
        config: *const OcinebConfigWire,
        n_images: i64,
        n_atoms: i64,
        positions: *const f64,
    ) -> *mut c_void;
}

fn ocineb_config() -> OcinebConfigWire {
    OcinebConfigWire {
        version: version(),
        flags: 0,
        trigger_factor: 0.5,
        trigger_force: 0.0,
        angle_tol: 0.1,
        stability_count: 1,
        max_mmf_steps: 1,
        restore_unhelpful: 0,
        reserved: 0,
    }
}

fn irc_at(n_atoms: i64) -> *mut RgsaddleIrc {
    let config = irc_config();
    let buf = [0.0; 3];
    unsafe { rgsaddle_irc_create(&config, n_atoms, buf.as_ptr(), buf.as_ptr(), buf.as_ptr()) }
}

fn sella_min_at(n_atoms: i64) -> *mut RgsaddleSellaMin {
    let config = sella_min_config();
    let buf = [0.0; 3];
    unsafe { rgsaddle_sella_min_create(&config, n_atoms, buf.as_ptr(), buf.as_ptr()) }
}

fn sella_saddle_at(n_atoms: i64) -> *mut RgsaddleSellaSaddle {
    let config = sella_saddle_config();
    let buf = [0.0; 3];
    unsafe { rgsaddle_sella_saddle_create(&config, n_atoms, buf.as_ptr(), buf.as_ptr()) }
}

fn constraints_at(n_atoms: i64) -> *mut RgsaddleConstraints {
    let config = constraints_config();
    unsafe { rgsaddle_constraints_create(&config, n_atoms) }
}

fn samd_at(n_atoms: i64) -> *mut RgsaddleSamd {
    let config = samd_config();
    let buf = [0.0; 3];
    unsafe {
        rgsaddle_samd_create(
            &config,
            n_atoms,
            buf.as_ptr(),
            buf.as_ptr(),
            Some(never_called),
            ptr::null_mut(),
        )
    }
}

fn pes_at(n_atoms: i64) -> *mut RgsaddlePes {
    let buf = [0.0; 3];
    unsafe { rgsaddle_pes_create(n_atoms, buf.as_ptr(), buf.as_ptr(), 0, 0) }
}

fn internal_pes_at(
    n_atoms: i64,
    n_bonds: i64,
    n_angles: i64,
    n_dihedrals: i64,
) -> *mut RgsaddleInternalPes {
    let buf = [0.0; 3];
    let idx = [0_i64; 4];
    unsafe {
        rgsaddle_internal_pes_create_from_find(
            n_atoms,
            buf.as_ptr(),
            buf.as_ptr(),
            n_bonds,
            idx.as_ptr(),
            n_angles,
            idx.as_ptr(),
            n_dihedrals,
            idx.as_ptr(),
        )
    }
}

fn indexk_at(n_atoms: i64) -> *mut RgsaddleIndexK {
    let config = indexk_config();
    let buf = [0.0; 3];
    unsafe { rgsaddle_indexk_create(&config, n_atoms, buf.as_ptr(), ptr::null()) }
}

fn ocineb_at(n_images: i64, n_atoms: i64) -> *mut c_void {
    let band = band_config();
    let minmode = minmode_config(1);
    let config = ocineb_config();
    let buf = [0.0; 3];
    unsafe { rgsaddle_ocineb_create(&band, &minmode, &config, n_images, n_atoms, buf.as_ptr()) }
}

fn saddle_index_at(n_atoms: i64) -> i32 {
    let buf = [0.0; 3];
    let mut out = std::mem::MaybeUninit::<RgsaddleSaddleIndex>::uninit();
    unsafe {
        rgsaddle_saddle_index(
            ptr::null(),
            n_atoms,
            buf.as_ptr(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            Some(never_called),
            ptr::null_mut(),
            out.as_mut_ptr(),
            ptr::null_mut(),
            0,
        )
    }
}

#[test]
fn irc_refuses_a_negative_count() {
    assert!(irc_at(-1).is_null());
}

#[test]
fn irc_refuses_an_overflowing_product() {
    assert!(irc_at(i64::MAX).is_null());
}

#[test]
fn irc_refuses_a_span_past_a_slice() {
    assert!(irc_at(HUGE_ATOMS).is_null());
}

#[test]
fn sella_min_refuses_a_negative_count() {
    assert!(sella_min_at(-1).is_null());
}

#[test]
fn sella_min_refuses_an_overflowing_product() {
    assert!(sella_min_at(i64::MAX).is_null());
}

#[test]
fn sella_min_refuses_a_span_past_a_slice() {
    assert!(sella_min_at(HUGE_ATOMS).is_null());
}

#[test]
fn sella_saddle_refuses_a_negative_count() {
    assert!(sella_saddle_at(-1).is_null());
}

#[test]
fn sella_saddle_refuses_an_overflowing_product() {
    assert!(sella_saddle_at(i64::MAX).is_null());
}

#[test]
fn sella_saddle_refuses_a_span_past_a_slice() {
    assert!(sella_saddle_at(HUGE_ATOMS).is_null());
}

#[test]
fn constraints_refuses_a_negative_count() {
    assert!(constraints_at(-1).is_null());
}

#[test]
fn constraints_refuses_an_overflowing_product() {
    assert!(constraints_at(i64::MAX).is_null());
}

#[test]
fn constraints_refuses_a_span_past_a_slice() {
    assert!(constraints_at(HUGE_ATOMS).is_null());
}

#[test]
fn samd_refuses_a_negative_count() {
    assert!(samd_at(-1).is_null());
}

#[test]
fn samd_refuses_an_overflowing_product() {
    assert!(samd_at(i64::MAX).is_null());
}

#[test]
fn samd_refuses_a_span_past_a_slice() {
    assert!(samd_at(HUGE_ATOMS).is_null());
}

#[test]
fn pes_refuses_a_negative_count() {
    assert!(pes_at(-1).is_null());
}

#[test]
fn pes_refuses_an_overflowing_product() {
    assert!(pes_at(i64::MAX).is_null());
}

#[test]
fn pes_refuses_a_span_past_a_slice() {
    assert!(pes_at(HUGE_ATOMS).is_null());
}

#[test]
fn internal_pes_refuses_a_negative_count() {
    assert!(internal_pes_at(-1, 0, 0, 0).is_null());
    assert!(internal_pes_at(1, -1, 0, 0).is_null());
    assert!(internal_pes_at(1, 0, -1, 0).is_null());
    assert!(internal_pes_at(1, 0, 0, -1).is_null());
}

#[test]
fn internal_pes_refuses_an_overflowing_product() {
    assert!(internal_pes_at(i64::MAX, 0, 0, 0).is_null());
}

#[test]
fn internal_pes_bonds_refuse_an_overflowing_product() {
    assert!(internal_pes_at(1, i64::MAX, 0, 0).is_null());
}

#[test]
fn internal_pes_angles_refuse_an_overflowing_product() {
    assert!(internal_pes_at(1, 0, i64::MAX, 0).is_null());
}

#[test]
fn internal_pes_dihedrals_refuse_an_overflowing_product() {
    assert!(internal_pes_at(1, 0, 0, i64::MAX).is_null());
}

#[test]
fn internal_pes_refuses_a_span_past_a_slice() {
    assert!(internal_pes_at(HUGE_ATOMS, 0, 0, 0).is_null());
}

#[test]
fn internal_pes_bonds_refuse_a_span_past_a_slice() {
    assert!(internal_pes_at(1, HUGE_ATOMS, 0, 0).is_null());
}

#[test]
fn internal_pes_angles_refuse_a_span_past_a_slice() {
    assert!(internal_pes_at(1, 0, HUGE_ATOMS, 0).is_null());
}

#[test]
fn internal_pes_dihedrals_refuse_a_span_past_a_slice() {
    assert!(internal_pes_at(1, 0, 0, HUGE_ATOMS).is_null());
}

#[test]
fn indexk_refuses_a_negative_count() {
    assert!(indexk_at(-1).is_null());
}

#[test]
fn indexk_refuses_an_overflowing_product() {
    assert!(indexk_at(i64::MAX).is_null());
}

#[test]
fn indexk_refuses_a_span_past_a_slice() {
    assert!(indexk_at(HUGE_ATOMS).is_null());
}

#[test]
fn ocineb_refuses_a_negative_count() {
    assert!(ocineb_at(3, -1).is_null());
    assert!(ocineb_at(-1, 1).is_null());
}

#[test]
fn ocineb_refuses_an_overflowing_product() {
    assert!(ocineb_at(3, i64::MAX).is_null());
}

#[test]
fn ocineb_images_refuse_an_overflowing_product() {
    assert!(ocineb_at(i64::MAX, 1).is_null());
}

#[test]
fn ocineb_refuses_a_span_past_a_slice() {
    assert!(ocineb_at(3, HUGE_ATOMS).is_null());
}

#[test]
fn saddle_index_refuses_a_negative_count() {
    assert_eq!(saddle_index_at(-1), RGSADDLE_SHAPE);
}

#[test]
fn saddle_index_refuses_an_overflowing_product() {
    assert_eq!(saddle_index_at(i64::MAX), RGSADDLE_SHAPE);
}

#[test]
fn saddle_index_refuses_a_span_past_a_slice() {
    assert_eq!(saddle_index_at(HUGE_ATOMS), RGSADDLE_SHAPE);
}
