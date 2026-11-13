//! Compiles and runs tests/c/wrap_smoke.cpp against the built cdylib.
//! Skips when no C++ compiler is installed. A compiler that cannot
//! compile, including a `c++` that cannot find `<cstddef>`, fails the
//! test unless another driver can compile. `RGSADDLE_REQUIRE_CAPI_SMOKE=1`
//! turns the skip into a failure.

#![cfg(feature = "capi")]

#[path = "common/abi_smoke.rs"]
mod abi_smoke;

use abi_smoke::Smoke;

#[test]
fn cxx_session_wrapper_uses_the_official_abi() {
    abi_smoke::run(&Smoke {
        label: "C++ session smoke test",
        source: "tests/c/wrap_smoke.cpp",
        output_stem: "rgsaddle_cxx_smoke",
        cxx: true,
        compile_args: &["-std=c++17", "-Wall", "-Wextra", "-Werror"],
        link_libm: false,
        token: "RGSADDLE_CXX_WRAP_OK",
        echo: false,
    });
}
