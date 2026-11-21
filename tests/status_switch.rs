use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

/// Directory removed when the test returns, including after a panic.
struct WorkDir(PathBuf);

impl WorkDir {
    fn new() -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock precedes the unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "rgsaddle-status-switch-{}-{nanos}",
            std::process::id()
        ));
        fs::create_dir_all(&path).unwrap_or_else(|err| {
            panic!(
                "failed to create the status-switch directory {}: {err}",
                path.display()
            )
        });
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for WorkDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn status_switch_covers_every_enumerator() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source = root.join("tests/c/status_switch.cpp");
    // A fixed path under the shared temp directory is ETXTBSY when another
    // run on this machine is executing that binary.
    let work = WorkDir::new();
    let output = work.path().join("status-switch");
    assert_ne!(
        output,
        std::env::temp_dir().join("rgsaddle-status-switch"),
        "the status switch must not reuse a shared executable path"
    );
    let compile = Command::new("g++")
        .args([
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Wswitch",
            "-Werror",
            "-I",
        ])
        .arg(root.join("include"))
        .arg("-I")
        .arg(root.join("bind"))
        .arg(&source)
        .arg("-o")
        .arg(&output)
        .output()
        .expect("g++ is required to compile the status switch");
    assert!(
        compile.status.success(),
        "status switch failed to compile:\n{}",
        String::from_utf8_lossy(&compile.stderr)
    );
    let run = Command::new(&output)
        .output()
        .expect("status switch executable");
    assert!(
        run.status.success(),
        "status switch returned {}",
        run.status
    );
}
