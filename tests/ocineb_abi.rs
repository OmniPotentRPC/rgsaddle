//! Compiles and runs tests/c/ocineb_abi.c against the built cdylib.
//! Skips when no C compiler is installed. A misconfigured compiler
//! fails the test. `RGSADDLE_REQUIRE_CAPI_SMOKE=1` turns the skip into
//! a failure.

#[path = "common/abi_smoke.rs"]
mod abi_smoke;

use abi_smoke::Smoke;

#[test]
#[cfg_attr(not(feature = "capi"), ignore)]
fn ocineb_c_abi() {
    abi_smoke::run(&Smoke {
        label: "OCI-NEB C ABI test",
        source: "tests/c/ocineb_abi.c",
        output_stem: "rgsaddle_ocineb_abi",
        cxx: false,
        compile_args: &[],
        link_libm: true,
        token: "RGSADDLE_OCINEB_ABI_OK",
        echo: false,
    });
}
