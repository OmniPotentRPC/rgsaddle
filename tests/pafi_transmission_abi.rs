//! Compiles and runs tests/c/pafi_transmission_abi.c against the built cdylib.

#![cfg(all(feature = "capi", feature = "ohtst"))]

use std::path::PathBuf;
use std::process::Command;

include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/common/cdylib.rs"
));

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
fn pafi_transmission_c_abi() {
    let Some(cc) = cc() else {
        eprintln!("no C compiler; skipping");
        return;
    };
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out = std::env::temp_dir().join(format!(
        "rgsaddle_pafi_transmission_abi_{}",
        std::process::id()
    ));
    let libdir = cdylib_dir();

    let build = Command::new(&cc)
        .arg("-std=c11")
        .arg("-Wall")
        .arg("-Wextra")
        .arg("-Werror")
        .arg(root.join("tests/c/pafi_transmission_abi.c"))
        .arg("-I")
        .arg(root.join("include"))
        .arg("-L")
        .arg(&libdir)
        .arg("-lrgsaddle")
        .arg("-lm")
        .arg("-o")
        .arg(&out)
        .output()
        .expect("compile the PAFI transmission C ABI test");
    assert!(
        build.status.success(),
        "PAFI transmission C ABI test failed to build: {}",
        String::from_utf8_lossy(&build.stderr)
    );

    let run = Command::new(&out)
        .env("LD_LIBRARY_PATH", &libdir)
        .output()
        .expect("run the PAFI transmission C ABI test");
    let stdout = String::from_utf8_lossy(&run.stdout);
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        run.status.success(),
        "PAFI transmission C ABI test failed:\nstdout: {stdout}\nstderr: {stderr}"
    );
    assert!(
        stdout.contains("RGSADDLE_PAFI_TRANSMISSION_ABI_OK"),
        "stdout: {stdout}"
    );
    let _ = std::fs::remove_file(&out);
}
