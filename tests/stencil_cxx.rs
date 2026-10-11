#![cfg(feature = "capi")]

use std::path::PathBuf;
use std::process::Command;

fn cxx_compiler() -> String {
    let candidates = [
        std::env::var("CXX").ok(),
        Some("g++".into()),
        Some("clang++".into()),
        Some("c++".into()),
    ];
    for candidate in candidates.into_iter().flatten() {
        let source = std::env::temp_dir().join(format!(
            "rgsaddle_stencil_cxx_probe_{}.cpp",
            std::process::id()
        ));
        let binary =
            std::env::temp_dir().join(format!("rgsaddle_stencil_cxx_probe_{}", std::process::id()));
        let _ = std::fs::write(&source, "#include <atomic>\nint main() { return 0; }\n");
        let output = Command::new(&candidate)
            .arg("-std=c++17")
            .arg(&source)
            .arg("-o")
            .arg(&binary)
            .output();
        let _ = std::fs::remove_file(&source);
        let _ = std::fs::remove_file(&binary);
        if output.is_ok_and(|output| output.status.success()) {
            return candidate;
        }
    }
    panic!("no C++ compiler can include <atomic>");
}

#[test]
fn cxx_stencil_wrapper_steps_a_dimer() {
    let compiler = cxx_compiler();
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
