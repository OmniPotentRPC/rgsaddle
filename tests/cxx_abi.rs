#![cfg(feature = "capi")]

use std::path::PathBuf;
use std::process::Command;

fn probe_cxx(compiler: &str) -> bool {
    let mut child = match Command::new(compiler)
        .args(["-std=c++17", "-x", "c++", "-c", "-", "-o", "/dev/null"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(_) => return false,
    };
    let written = child
        .stdin
        .as_mut()
        .and_then(|stdin| {
            use std::io::Write;
            stdin.write_all(b"#include <cstddef>\n").ok()
        })
        .is_some();
    drop(child.stdin.take());
    if !written {
        let _ = child.wait();
        return false;
    }
    match child.wait_with_output() {
        Ok(output) => output.status.success(),
        Err(_) => false,
    }
}

fn cxx_compiler() -> Result<String, String> {
    let mut tried = Vec::new();
    let mut candidates: Vec<String> = Vec::new();
    if let Ok(cxx) = std::env::var("CXX")
        && !cxx.is_empty()
    {
        candidates.push(cxx);
    }
    for name in ["g++", "clang++", "c++"] {
        if !candidates.iter().any(|have| have == name) {
            candidates.push(name.to_string());
        }
    }
    for compiler in candidates {
        tried.push(compiler.clone());
        if probe_cxx(&compiler) {
            return Ok(compiler);
        }
    }
    Err(format!(
        "no working C++ compiler; tried {}",
        tried.join(", ")
    ))
}

#[test]
fn cxx_session_wrapper_uses_the_official_abi() {
    let compiler = cxx_compiler().unwrap_or_else(|err| panic!("{err}"));
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out = std::env::temp_dir().join(format!("rgsaddle_cxx_smoke_{}", std::process::id()));
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
    let stdout = String::from_utf8_lossy(&run.stdout);
    assert!(stdout.contains("RGSADDLE_CXX_WRAP_OK"), "{stdout}");
    assert!(stdout.contains("RGSADDLE_CXX_ART_OK"), "{stdout}");
    std::fs::remove_file(out).expect("remove temporary smoke executable");
}
