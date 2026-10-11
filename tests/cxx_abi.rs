#![cfg(feature = "capi")]

use std::path::PathBuf;
use std::process::Command;

fn compiler_accepts_cstddef(compiler: &str) -> bool {
    let dir = std::env::temp_dir();
    let src = dir.join(format!(
        "rgsaddle_cxx_probe_{}_{}.cpp",
        std::process::id(),
        compiler.replace('/', "_")
    ));
    let bin = dir.join(format!(
        "rgsaddle_cxx_probe_{}_{}",
        std::process::id(),
        compiler.replace('/', "_")
    ));
    if std::fs::write(&src, "#include <cstddef>\nint main(){return 0;}\n").is_err() {
        return false;
    }
    let ok = Command::new(compiler)
        .arg(&src)
        .arg("-o")
        .arg(&bin)
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false);
    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&bin);
    ok
}

fn cxx_compiler() -> String {
    if let Ok(cxx) = std::env::var("CXX")
        && compiler_accepts_cstddef(&cxx)
    {
        return cxx;
    }
    if compiler_accepts_cstddef("c++") {
        return "c++".into();
    }
    if compiler_accepts_cstddef("g++") {
        return "g++".into();
    }
    std::env::var("CXX").unwrap_or_else(|_| "c++".into())
}

#[test]
fn cxx_session_wrapper_uses_the_official_abi() {
    let compiler = cxx_compiler();
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
    assert!(String::from_utf8_lossy(&run.stdout).contains("RGSADDLE_CXX_WRAP_OK"));
    std::fs::remove_file(out).expect("remove temporary smoke executable");
}
