#![cfg(feature = "capi")]

use std::path::PathBuf;
use std::process::Command;

fn libdir() -> PathBuf {
    let mut libdir = std::env::current_exe().unwrap();
    libdir.pop();
    if libdir.ends_with("deps") {
        libdir.pop();
    }
    if !libdir.join("librgsaddle.so").exists() {
        let deps = libdir.join("deps");
        if deps.join("librgsaddle.so").exists() {
            return deps;
        }
    }
    assert!(
        libdir.join("librgsaddle.so").exists(),
        "build the C ABI library before testing"
    );
    libdir
}

fn cxx_compiler() -> String {
    let mut tried = Vec::new();
    let candidates = [
        std::env::var("CXX").ok(),
        Some("g++".into()),
        Some("clang++".into()),
        Some("c++".into()),
    ];
    for candidate in candidates.into_iter().flatten() {
        if tried.iter().any(|seen: &String| seen == &candidate) {
            continue;
        }
        tried.push(candidate.clone());
        let probe = std::env::temp_dir().join(format!(
            "rgsaddle_cxx_probe_{}_{}",
            std::process::id(),
            tried.len()
        ));
        let source = std::env::temp_dir().join(format!(
            "rgsaddle_cxx_probe_{}_{}.cpp",
            std::process::id(),
            tried.len()
        ));
        let _ = std::fs::write(&source, "#include <cstddef>\nint main() { return 0; }\n");
        let output = Command::new(&candidate)
            .arg("-std=c++17")
            .arg(&source)
            .arg("-o")
            .arg(&probe)
            .output();
        let _ = std::fs::remove_file(&source);
        let _ = std::fs::remove_file(&probe);
        if output.is_ok_and(|output| output.status.success()) {
            return candidate;
        }
    }
    panic!("no working C++ compiler. Tried: {}", tried.join(", "));
}

#[test]
fn cxx_session_wrapper_uses_the_official_abi() {
    let compiler = cxx_compiler();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out = std::env::temp_dir().join(format!("rgsaddle_cxx_smoke_{}", std::process::id()));
    let libdir = libdir();
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
