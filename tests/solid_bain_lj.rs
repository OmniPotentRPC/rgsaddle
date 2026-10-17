#![cfg(feature = "capi")]
//! Bain path of a switched Lennard-Jones crystal through
//! rgsaddle_band_create_solid: tests/c/solid_bain_lj.c checks the
//! climbing image against the relaxed scan of
//! validation/bain_lj_reference.py.

use std::path::PathBuf;
use std::process::Command;

#[test]
fn c_abi_solid_state_band_climbs_the_bain_path() {
    let compiler = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out = std::env::temp_dir().join(format!("rgsaddle_solid_bain_lj_{}", std::process::id()));
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
        .arg("-O2")
        .arg(root.join("tests/c/solid_bain_lj.c"))
        .arg("-I")
        .arg(root.join("include"))
        .arg("-L")
        .arg(&libdir)
        .arg("-lrgsaddle")
        .arg("-lm")
        .arg("-o")
        .arg(&out)
        .output()
        .expect("compile the Bain-path C test");
    assert!(
        build.status.success(),
        "Bain-path C test failed to build: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    let run = Command::new(&out)
        .env("LD_LIBRARY_PATH", &libdir)
        .output()
        .expect("run the Bain-path C test");
    let stdout = String::from_utf8_lossy(&run.stdout);
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        run.status.success(),
        "Bain-path C test failed:\nstdout: {stdout}\nstderr: {stderr}"
    );
    println!("{stdout}");
    assert!(stdout.contains("RGSADDLE_BAIN_LJ_OK"), "stdout: {stdout}");
    std::fs::remove_file(out).expect("remove temporary Bain-path executable");
}
