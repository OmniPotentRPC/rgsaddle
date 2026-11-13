//! Compiles and runs tests/c/irc_abi.c against the built cdylib.
//! Skips when no C compiler is installed. A misconfigured compiler
//! fails the test. `RGSADDLE_REQUIRE_CAPI_SMOKE=1` turns the skip into
//! a failure.

#![cfg(feature = "capi")]

#[path = "common/abi_smoke.rs"]
mod abi_smoke;

use abi_smoke::Smoke;

#[test]
fn c_abi_irc_analytic_well() {
    abi_smoke::run(&Smoke {
        label: "IRC C ABI test",
        source: "tests/c/irc_abi.c",
        output_stem: "rgsaddle_irc_abi",
        cxx: false,
        compile_args: &["-std=c11", "-Wall", "-Wextra", "-Werror"],
        link_libm: true,
        token: "RGSADDLE_IRC_ABI_OK",
        echo: false,
    });
}
