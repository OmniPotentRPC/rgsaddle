/// Path of `librgsaddle.so`, built for this test run.
///
/// `cargo test` compiles the cdylib for the harness but does not copy
/// it into the profile directory, so a `librgsaddle.so` found there
/// may come from another feature set or an older tree. This runs
/// `cargo build --lib` with the features, profile, and target
/// directory of the running test binary and returns the cdylib that
/// cargo reports. Cargo releases the build-directory lock before it
/// runs test binaries, so the nested build shares `target/`. It sees
/// `RUSTFLAGS` and config files, not `--config` given to `cargo test`.
fn cdylib() -> std::path::PathBuf {
    static LIB: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();
    LIB.get_or_init(build_cdylib).clone()
}

/// Every feature in `[features]` except `default`, which only names
/// others. `build_cdylib` refuses a manifest feature missing here.
const CDYLIB_FEATURES: &[(&str, bool)] = &[
    ("capi", cfg!(feature = "capi")),
    ("par", cfg!(feature = "par")),
    ("highs", cfg!(feature = "highs")),
    ("readcon", cfg!(feature = "readcon")),
    ("readcon-db", cfg!(feature = "readcon-db")),
    ("chemfiles", cfg!(feature = "chemfiles")),
    ("verify", cfg!(feature = "verify")),
    ("faer", cfg!(feature = "faer")),
];

fn build_cdylib() -> std::path::PathBuf {
    use std::path::{Path, PathBuf};

    let manifest = Path::new(env!("CARGO_MANIFEST_PATH"));
    check_cdylib_features(manifest);
    let target_dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .parent()
        .expect("CARGO_TARGET_TMPDIR is <target-dir>/tmp");
    let features: Vec<&str> = CDYLIB_FEATURES
        .iter()
        .filter(|(_, on)| *on)
        .map(|(name, _)| *name)
        .collect();

    let mut command =
        std::process::Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()));
    command
        .args(["build", "--lib", "--message-format=json-render-diagnostics"])
        .arg("--manifest-path")
        .arg(manifest)
        .arg("--target-dir")
        .arg(target_dir)
        .arg("--no-default-features")
        .arg("--features")
        .arg(features.join(","));
    let (triple, profile) = build_layout(target_dir);
    if let Some(triple) = triple {
        command.arg("--target").arg(triple);
    }
    command.arg("--profile").arg(profile);
    let shown = format!("{command:?}");
    let output = command.output().expect("spawn cargo build --lib");
    assert!(
        output.status.success(),
        "{shown} failed with {}:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );

    let file_name = format!(
        "{}rgsaddle{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_SUFFIX
    );
    let stdout = String::from_utf8(output.stdout).expect("cargo JSON is UTF-8");
    let found: Vec<PathBuf> = stdout
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|message| {
            message["reason"] == "compiler-artifact"
                && message["manifest_path"].as_str().map(Path::new) == Some(manifest)
                && message["target"]["kind"]
                    .as_array()
                    .is_some_and(|kinds| kinds.iter().any(|kind| kind == "cdylib"))
        })
        .flat_map(|message| match message["filenames"].as_array() {
            Some(paths) => paths
                .iter()
                .filter_map(|path| path.as_str().map(PathBuf::from))
                .collect(),
            None => Vec::new(),
        })
        .filter(|path| path.file_name().is_some_and(|name| *name == *file_name))
        .collect();
    match found.as_slice() {
        [path] if path.is_file() => path.clone(),
        _ => panic!("{shown} reported {found:?} for {file_name}, expected one existing file"),
    }
}

fn check_cdylib_features(manifest: &std::path::Path) {
    let text = std::fs::read_to_string(manifest).expect("read Cargo.toml");
    let mut in_features = false;
    for line in text.lines().map(str::trim) {
        if line.starts_with('[') {
            in_features = line == "[features]";
        } else if let Some((name, _)) = line.split_once('=')
            && in_features
            && !line.starts_with('#')
        {
            let name = name.trim();
            assert!(
                name == "default" || CDYLIB_FEATURES.iter().any(|(known, _)| *known == name),
                "feature {name} is missing from CDYLIB_FEATURES in tests/common/cdylib.rs"
            );
        }
    }
}

/// `(--target triple, --profile name)` of the running test binary,
/// read from its path under the target directory: `<profile-dir>/...`
/// or `<triple>/<profile-dir>/...`. `cargo bench` also writes to
/// `release`, and its default profile builds the same library.
fn build_layout(target_dir: &std::path::Path) -> (Option<String>, String) {
    let exe = std::env::current_exe().expect("test executable");
    let parts: Vec<String> = exe
        .strip_prefix(target_dir)
        .unwrap_or_else(|_| {
            panic!(
                "test executable {} is outside the target directory {}",
                exe.display(),
                target_dir.display()
            )
        })
        .components()
        .map(|part| part.as_os_str().to_string_lossy().into_owned())
        .collect();
    let profile = |dir: &str| match dir {
        "debug" => "dev".to_owned(),
        other => other.to_owned(),
    };
    let first = parts[0].as_str();
    if first == "debug" || first == "release" {
        return (None, profile(first));
    }
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let is_triple = std::process::Command::new(rustc)
        .args(["--print", "target-list"])
        .output()
        .is_ok_and(|out| String::from_utf8_lossy(&out.stdout).lines().any(|t| t == first));
    if is_triple {
        (Some(first.to_owned()), profile(&parts[1]))
    } else {
        (None, profile(first))
    }
}
