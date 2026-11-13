//! Bain path of a switched Lennard-Jones crystal through
//! rgsaddle_band_create_solid: tests/c/solid_bain_lj.c checks the
//! climbing image against the relaxed scan of
//! validation/bain_lj_reference.py.
//!
//! Skips when no C compiler is installed. A misconfigured compiler
//! fails the test. `RGSADDLE_REQUIRE_CAPI_SMOKE=1` turns the skip into
//! a failure.

#![cfg(feature = "capi")]

#[path = "common/abi_smoke.rs"]
mod abi_smoke;

use abi_smoke::Smoke;

#[test]
fn c_abi_solid_state_band_climbs_the_bain_path() {
    abi_smoke::run(&Smoke {
        label: "Bain-path C test",
        source: "tests/c/solid_bain_lj.c",
        output_stem: "rgsaddle_solid_bain_lj",
        cxx: false,
        compile_args: &["-std=c11", "-Wall", "-Wextra", "-Werror", "-O2"],
        link_libm: true,
        token: "RGSADDLE_BAIN_LJ_OK",
        echo: true,
    });
}
