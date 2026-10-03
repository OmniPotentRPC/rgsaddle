#![cfg(feature = "capi")]

use std::path::PathBuf;
use std::process::Command;

#[test]
fn c_abi_irc_analytic_well() {
    let compiler = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out = std::env::temp_dir().join(format!("rgsaddle_irc_abi_{}", std::process::id()));
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
        .arg("-std=c11")
        .arg("-Wall")
        .arg("-Wextra")
        .arg("-Werror")
        .arg(root.join("tests/c/irc_abi.c"))
        .arg("-I")
        .arg(root.join("include"))
        .arg("-L")
        .arg(&libdir)
        .arg("-lrgsaddle")
        .arg("-lm")
        .arg("-o")
        .arg(&out)
        .output()
        .expect("compile the IRC C ABI test");
    assert!(
        build.status.success(),
        "IRC C ABI test failed to build: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    let run = Command::new(&out)
        .env("LD_LIBRARY_PATH", &libdir)
        .output()
        .expect("run the IRC C ABI test");
    let stdout = String::from_utf8_lossy(&run.stdout);
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        run.status.success(),
        "IRC C ABI test failed:\nstdout: {stdout}\nstderr: {stderr}"
    );
    assert!(stdout.contains("RGSADDLE_IRC_ABI_OK"), "stdout: {stdout}");
    std::fs::remove_file(out).expect("remove temporary IRC executable");
}
