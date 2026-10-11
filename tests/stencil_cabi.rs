//! Compiles and runs tests/c/stencil_abi.c against the built library.

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
fn c_stencil_abi_drives_muller_brown() {
    let Some(cc) = cc() else {
        panic!("no C compiler");
    };
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out = std::env::temp_dir().join(format!("rgsaddle_stencil_abi_{}", std::process::id()));
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
    let build = Command::new(&cc)
        .arg("-std=c11")
        .arg("-Wall")
        .arg("-Wextra")
        .arg("-Werror")
        .arg(root.join("tests/c/stencil_abi.c"))
        .arg("-I")
        .arg(root.join("include"))
        .arg("-L")
        .arg(&libdir)
        .arg("-lrgsaddle")
        .arg("-lm")
        .arg("-o")
        .arg(&out)
        .output()
        .expect("compile stencil C test");
    assert!(
        build.status.success(),
        "C compile failed: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    let run = Command::new(&out)
        .env("LD_LIBRARY_PATH", &libdir)
        .output()
        .expect("run stencil C test");
    assert!(
        run.status.success(),
        "stdout {}\nstderr {}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    assert!(String::from_utf8_lossy(&run.stdout).contains("RGSADDLE_STENCIL_C_OK"));
    let _ = std::fs::remove_file(out);
}
