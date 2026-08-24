//! Compiles and runs tests/c/abi_smoke.c against the built cdylib.
//! Skips when no C compiler is available.

use std::path::PathBuf;
use std::process::Command;

fn cc() -> Option<String> {
    for c in [
        std::env::var("CC").ok(),
        Some("cc".into()),
        Some("gcc".into()),
    ]
    .into_iter()
    .flatten()
    {
        if Command::new(&c).arg("--version").output().is_ok() {
            return Some(c);
        }
    }
    None
}

#[test]
#[cfg_attr(not(feature = "capi"), ignore)]
fn c_abi_smoke() {
    let Some(cc) = cc() else {
        eprintln!("no C compiler; skipping");
        return;
    };
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out = std::env::temp_dir().join(format!("rgsaddle_abi_smoke_{}", std::process::id()));

    // The cdylib lives beside the test binary.
    let mut libdir = std::env::current_exe().unwrap();
    libdir.pop();
    if libdir.ends_with("deps") {
        libdir.pop();
    }

    let status = Command::new(&cc)
        .arg(root.join("tests/c/abi_smoke.c"))
        .arg("-I")
        .arg(root.join("include"))
        .arg("-L")
        .arg(&libdir)
        .arg("-lrgsaddle")
        .arg("-lm")
        .arg("-o")
        .arg(&out)
        .status()
        .expect("compile the C smoke test");
    assert!(status.success(), "C smoke test failed to build");

    let run = Command::new(&out)
        .env("LD_LIBRARY_PATH", &libdir)
        .output()
        .expect("run the C smoke test");
    let stdout = String::from_utf8_lossy(&run.stdout);
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        run.status.success(),
        "C smoke test failed:\nstdout: {stdout}\nstderr: {stderr}"
    );
    assert!(stdout.contains("RGSADDLE_C_ABI_OK"), "stdout: {stdout}");
    let _ = std::fs::remove_file(&out);
}
