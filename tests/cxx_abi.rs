#![cfg(feature = "capi")]

use std::path::PathBuf;
use std::process::Command;

include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/common/cdylib.rs"
));

#[test]
fn cxx_session_wrapper_uses_the_official_abi() {
    let compiler = std::env::var("CXX").unwrap_or_else(|_| "c++".into());
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out = std::env::temp_dir().join(format!("rgsaddle_cxx_smoke_{}", std::process::id()));
    let libdir = cdylib_dir();
    assert!(
        libdir.join("librgsaddle.so").exists(),
        "build the C ABI library before testing"
    );
    let build = Command::new(compiler)
        .arg("-std=c++17")
        .arg("-Wall")
        .arg("-Wextra")
        .arg("-Werror")
        .arg(root.join("tests/c/wrap_smoke.cpp"))
        .arg("-I")
        .arg(root.join("include"))
        .arg("-L")
        .arg(&libdir)
        .arg("-lrgsaddle")
        .arg("-o")
        .arg(&out)
        .output()
        .expect("compile the C++ session smoke test");
    assert!(
        build.status.success(),
        "C++ compile failed: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    let run = Command::new(&out)
        .env("LD_LIBRARY_PATH", &libdir)
        .output()
        .expect("run C++ session smoke test");
    assert!(
        run.status.success(),
        "C++ smoke failed: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert!(String::from_utf8_lossy(&run.stdout).contains("RGSADDLE_CXX_WRAP_OK"));
    std::fs::remove_file(out).expect("remove temporary smoke executable");
}

#[test]
fn cxx_landscape_limit_is_an_error_and_hessian_sees_x() {
    let compiler = std::env::var("CXX").unwrap_or_else(|_| "c++".into());
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out = std::env::temp_dir().join(format!(
        "rgsaddle_cxx_landscape_host_{}",
        std::process::id()
    ));
    let libdir = cdylib_dir();
    let build = Command::new(compiler)
        .arg("-std=c++17")
        .arg("-Wall")
        .arg("-Wextra")
        .arg("-Werror")
        .arg(root.join("tests/c/landscape_host.cpp"))
        .arg("-I")
        .arg(root.join("include"))
        .arg("-L")
        .arg(&libdir)
        .arg("-lrgsaddle")
        .arg("-o")
        .arg(&out)
        .output()
        .expect("compile the landscape host test");
    assert!(
        build.status.success(),
        "C++ compile failed: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    let run = Command::new(&out)
        .env("LD_LIBRARY_PATH", &libdir)
        .output()
        .expect("run the landscape host test");
    assert!(
        run.status.success(),
        "landscape host failed: {}\n{}",
        String::from_utf8_lossy(&run.stderr),
        String::from_utf8_lossy(&run.stdout)
    );
    assert!(String::from_utf8_lossy(&run.stdout).contains("RGSADDLE_LANDSCAPE_HOST_OK"));
    std::fs::remove_file(out).expect("remove temporary landscape host executable");
}

#[test]
fn cxx_landscape_guard_matches_the_symbol_minor() {
    let compiler = std::env::var("CXX").unwrap_or_else(|_| "c++".into());
    let cc = std::env::var("CC").unwrap_or_else(|_| "gcc".into());
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out = std::env::temp_dir().join(format!(
        "rgsaddle_cxx_landscape_guard_{}",
        std::process::id()
    ));
    let mock_c = std::env::temp_dir().join(format!("rgsaddle_mock_abi67_{}.c", std::process::id()));
    let mock_so =
        std::env::temp_dir().join(format!("rgsaddle_mock_abi67_{}.so", std::process::id()));
    let libdir = cdylib_dir();
    let build = Command::new(compiler)
        .arg("-std=c++17")
        .arg("-Wall")
        .arg("-Wextra")
        .arg("-Werror")
        .arg(root.join("tests/c/landscape_guard.cpp"))
        .arg("-I")
        .arg(root.join("include"))
        .arg("-L")
        .arg(&libdir)
        .arg("-lrgsaddle")
        .arg("-o")
        .arg(&out)
        .output()
        .expect("compile the landscape guard test");
    assert!(
        build.status.success(),
        "C++ compile failed: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    let run = Command::new(&out)
        .env("LD_LIBRARY_PATH", &libdir)
        .output()
        .expect("run the landscape guard test");
    assert!(
        run.status.success(),
        "landscape guard failed: {}\n{}",
        String::from_utf8_lossy(&run.stderr),
        String::from_utf8_lossy(&run.stdout)
    );
    assert!(String::from_utf8_lossy(&run.stdout).contains("RGSADDLE_LANDSCAPE_GUARD_OK"));

    // Also verify that rgsaddle::Landscape refuses a library reporting ABI 1.67.
    std::fs::write(
        &mock_c,
        "#include <stdint.h>\ntypedef struct { uint32_t major; uint32_t minor; } v_t;\nint32_t rgsaddle_abi_stamp(v_t *out) { if (!out) return -1; out->major = 1u; out->minor = 67u; return 0; }\n",
    )
    .expect("write mock ABI 1.67 C shim");
    let mock_build = Command::new(&cc)
        .arg("-shared")
        .arg("-fPIC")
        .arg(&mock_c)
        .arg("-o")
        .arg(&mock_so)
        .output()
        .expect("compile mock ABI 1.67 shim");
    assert!(
        mock_build.status.success(),
        "mock shim compile failed: {}",
        String::from_utf8_lossy(&mock_build.stderr)
    );
    let refused = Command::new(&out)
        .arg("--expect-refused")
        .env("LD_LIBRARY_PATH", &libdir)
        .env("LD_PRELOAD", &mock_so)
        .output()
        .expect("run the landscape guard refusal test");
    assert!(
        refused.status.success(),
        "landscape guard did not refuse 1.67: {}\n{}",
        String::from_utf8_lossy(&refused.stderr),
        String::from_utf8_lossy(&refused.stdout)
    );
    assert!(
        String::from_utf8_lossy(&refused.stdout).contains("RGSADDLE_LANDSCAPE_GUARD_REFUSED_OK")
    );

    let _ = std::fs::remove_file(mock_c);
    let _ = std::fs::remove_file(mock_so);
    std::fs::remove_file(out).expect("remove temporary landscape guard executable");
}
