use std::process::Command;

fn compiler() -> Command {
    let triple = host_triple();
    cc::Build::new()
        .cpp(true)
        .opt_level(0)
        .debug(false)
        .host(&triple)
        .target(&triple)
        .cargo_metadata(false)
        .get_compiler()
        .to_command()
}

fn host_triple() -> String {
    let output = Command::new("rustc")
        .arg("-vV")
        .output()
        .expect("rustc reports its host");
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .find_map(|line| line.strip_prefix("host: "))
        .expect("rustc host line")
        .to_string()
}

#[test]
fn status_switch_covers_every_enumerator() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source = root.join("tests/c/status_switch.cpp");
    let dir = tempfile::tempdir().expect("temp directory");
    let output = dir.path().join("rgsaddle-status-switch");
    let compile = compiler()
        .args(["-std=c++17", "-Wall", "-Wextra", "-Wswitch", "-Werror", "-I"])
        .arg(root.join("include"))
        .arg("-I")
        .arg(root.join("bind"))
        .arg(&source)
        .arg("-o")
        .arg(&output)
        .output()
        .expect("C++ compiler");
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
