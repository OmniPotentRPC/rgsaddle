#![cfg(feature = "readcon")]

use std::path::Path;
use std::process::Command;

use rgsaddle::frame_from_con;

include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/common/cdylib.rs"
));

fn assert_no_mi_version(artifact: &Path) {
    let output = Command::new("nm")
        .args(["-g", "--defined-only"])
        .arg(artifact)
        .output()
        .expect("run nm on artifact");
    assert!(
        output.status.success(),
        "nm failed on {}: {}",
        artifact.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let colliding: Vec<&str> = stdout
        .lines()
        .filter(|line| {
            line.split_whitespace()
                .last()
                .is_some_and(|sym| sym.trim_start_matches('_') == "mi_version")
        })
        .collect();
    assert!(
        colliding.is_empty(),
        "{} defines mi_version from minimage:\n{}",
        artifact.display(),
        colliding.join("\n")
    );
}

#[test]
fn readcon_and_minimage_binary_exports_no_mi_version() {
    let frame = frame_from_con(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/triclinic.con"
    ))
    .expect("read triclinic.con");
    let cell = frame.cell.expect("triclinic cell");
    let ortho = minimage::Cell::ortho(10.0, 10.0, 10.0).expect("ortho cell");
    let d2 = ortho.dist2(cell.a(), cell.b());
    assert!(d2.is_finite());

    let exe = std::env::current_exe().expect("current test binary path");
    assert_no_mi_version(&exe);

    let libdir = cdylib_dir();
    assert_no_mi_version(&libdir.join("librgsaddle.so"));
    assert_no_mi_version(&libdir.join("librgsaddle.a"));
}
