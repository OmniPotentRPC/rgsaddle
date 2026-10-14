#![cfg(feature = "capi")]

use std::path::PathBuf;
use std::process::Command;

#[test]
fn cxx_stencil_wrapper_steps_a_dimer() {
    let compiler = std::env::var("CXX").unwrap_or_else(|_| "c++".into());
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out = std::env::temp_dir().join(format!("rgsaddle_stencil_cxx_{}", std::process::id()));
    let mut libdir = std::env::current_exe().unwrap();
    libdir.pop();
    if libdir.ends_with("deps") {
        libdir.pop();
    }
    assert!(
        libdir.join("librgsaddle.so").exists(),
        "build the C ABI library before testing"
    );
    let build = Command::new(compiler)
        .arg("-std=c++17")
        .arg("-Wall")
        .arg("-Wextra")
        .arg("-Werror")
        .arg(root.join("tests/c/stencil_smoke.cpp"))
        .arg("-I")
        .arg(root.join("include"))
        .arg("-L")
        .arg(&libdir)
        .arg("-lrgsaddle")
        .arg("-o")
        .arg(&out)
        .output()
        .expect("compile the C++ stencil smoke test");
    assert!(
        build.status.success(),
        "C++ compile failed: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    let run = Command::new(&out)
        .env("LD_LIBRARY_PATH", &libdir)
        .output()
        .expect("run C++ stencil smoke test");
    assert!(
        run.status.success(),
        "C++ stencil smoke failed: {}{}",
        String::from_utf8_lossy(&run.stderr),
        String::from_utf8_lossy(&run.stdout)
    );
    assert!(String::from_utf8_lossy(&run.stdout).contains("RGSADDLE_STENCIL_CXX_OK"));
    std::fs::remove_file(out).expect("remove temporary smoke executable");
}
