//! Compiles and runs tests/c/ocineb_abi.c against the built cdylib.

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

#[test]
#[cfg_attr(not(feature = "capi"), ignore)]
fn ocineb_c_abi() {
    let Some(cc) = cc() else {
        eprintln!("no C compiler; skipping");
        return;
    };
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out = std::env::temp_dir().join(format!("rgsaddle_ocineb_abi_{}", std::process::id()));
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
        .arg(root.join("tests/c/ocineb_abi.c"))
        .arg("-I")
        .arg(root.join("include"))
        .arg("-L")
        .arg(&libdir)
        .arg("-lrgsaddle")
        .arg("-lm")
        .arg("-o")
        .arg(&out)
        .status()
        .expect("compile the OCI-NEB C ABI test");
    assert!(status.success(), "OCI-NEB C ABI test failed to build");

    let run = Command::new(&out)
        .env("LD_LIBRARY_PATH", &libdir)
        .output()
        .expect("run the OCI-NEB C ABI test");
    let stdout = String::from_utf8_lossy(&run.stdout);
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        run.status.success(),
        "OCI-NEB C ABI test failed:\nstdout: {stdout}\nstderr: {stderr}"
    );
    assert!(
        stdout.contains("RGSADDLE_OCINEB_ABI_OK"),
        "stdout: {stdout}"
    );
    let _ = std::fs::remove_file(&out);
}
