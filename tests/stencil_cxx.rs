//! Compiles and runs tests/c/stencil_smoke.cpp against the built cdylib.
//! Skips when no C++ compiler is installed. A misconfigured compiler
//! fails the test. `RGSADDLE_REQUIRE_CAPI_SMOKE=1` turns the skip into
//! a failure.

#![cfg(feature = "capi")]

#[path = "common/abi_smoke.rs"]
mod abi_smoke;

use abi_smoke::Smoke;

#[test]
fn cxx_stencil_wrapper_steps_a_dimer() {
    abi_smoke::run(&Smoke {
        label: "C++ stencil smoke test",
        source: "tests/c/stencil_smoke.cpp",
        output_stem: "rgsaddle_stencil_cxx",
        cxx: true,
        compile_args: &["-std=c++17", "-Wall", "-Wextra", "-Werror"],
        link_libm: false,
        token: "RGSADDLE_STENCIL_CXX_OK",
        echo: false,
    });
}
