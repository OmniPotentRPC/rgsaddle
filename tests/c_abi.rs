//! Compiles and runs tests/c/abi_smoke.c against the built cdylib.
//! Skips when no C compiler is available.

use std::path::PathBuf;
use std::process::Command;

fn cc() -> Option<String> {
    [
        std::env::var("CC").ok(),
        Some("cc".into()),
        Some("gcc".into()),
    ]
    .into_iter()
    .flatten()
    .find(|c| Command::new(c).arg("--version").output().is_ok())
}

fn run_c(source: &str, marker: &str) {
    let Some(cc) = cc() else {
        eprintln!("no C compiler; skipping");
        return;
    };
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out = std::env::temp_dir().join(format!("rgsaddle_c_{}_{}", marker, std::process::id()));

    // The cdylib lives beside the test binary.
    let mut libdir = std::env::current_exe().unwrap();
    libdir.pop();
    if libdir.ends_with("deps") {
        libdir.pop();
    }
    if !libdir.join("librgsaddle.so").exists() {
        let deps = libdir.join("deps");
        if deps.join("librgsaddle.so").exists() {
            libdir = deps;
        }
    }

    let status = Command::new(&cc)
        .arg(root.join(source))
        .arg("-I")
        .arg(root.join("include"))
        .arg("-L")
        .arg(&libdir)
        .arg("-lrgsaddle")
        .arg("-lm")
        .arg("-o")
        .arg(&out)
        .status()
        .expect("compile the C test");
    assert!(status.success(), "{source} failed to build");

    let run = Command::new(&out)
        .env("LD_LIBRARY_PATH", &libdir)
        .output()
        .expect("run the C test");
    let stdout = String::from_utf8_lossy(&run.stdout);
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        run.status.success(),
        "{source} failed:\nstdout: {stdout}\nstderr: {stderr}"
    );
    assert!(stdout.contains(marker), "stdout: {stdout}");
    let _ = std::fs::remove_file(&out);
}

#[test]
#[cfg_attr(not(feature = "capi"), ignore)]
fn c_abi_smoke() {
    run_c("tests/c/abi_smoke.c", "RGSADDLE_C_ABI_OK");
}

#[test]
#[cfg_attr(not(feature = "capi"), ignore)]
fn c_fixed_mask() {
    run_c("tests/c/fixed_mask.c", "RGSADDLE_FIXED_MASK_OK");
}
