//! Compiles and runs tests/c/abi_smoke.c against the built cdylib.
//! Skips when no C compiler is installed. A compiler that cannot
//! compile fails the test. `RGSADDLE_REQUIRE_CAPI_SMOKE=1` turns the
//! skip into a failure.

#[path = "common/abi_smoke.rs"]
mod abi_smoke;

use abi_smoke::Smoke;

#[test]
#[cfg_attr(not(feature = "capi"), ignore)]
fn c_abi_smoke() {
    abi_smoke::run(&Smoke {
        label: "C smoke test",
        source: "tests/c/abi_smoke.c",
        output_stem: "rgsaddle_abi_smoke",
        cxx: false,
        compile_args: &[],
        link_libm: true,
        token: "RGSADDLE_C_ABI_OK",
        echo: false,
    });
}
