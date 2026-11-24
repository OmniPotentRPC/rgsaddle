/// Directory that holds a freshly linked `librgsaddle.so` matching this
/// test binary's active features and profile.
///
/// `cargo test` compiles integration tests against the `rlib` without
/// uplifting `librgsaddle.so` into `target/<profile>`, so a prior
/// `cargo build --features capi` would leave a stale `capi` only shared
/// library behind when `cargo test --features capi,ohtst` runs. Invoking
/// `cargo build --lib` with the active feature set refreshes
/// `librgsaddle.so` before the C compiler links against it.
///
/// The build reuses the outer target directory and asks rustc for only the
/// `cdylib`. Cargo before 1.100 names the `rlib` of a mixed crate without a
/// hash, so a plain nested `cargo build` can replace the `rlib` that rustdoc
/// links later in the same `cargo test`. Restricting the nested rustc output
/// to the shared library avoids that replacement while sharing dependencies.
fn cdylib_dir() -> std::path::PathBuf {
    static DIR: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();
    DIR.get_or_init(build_fresh_cdylib).clone()
}

fn active_features() -> Vec<&'static str> {
    let mut features = Vec::new();
    if cfg!(feature = "capi") {
        features.push("capi");
    }
    if cfg!(feature = "par") {
        features.push("par");
    }
    if cfg!(feature = "highs") {
        features.push("highs");
    }
    if cfg!(feature = "ohtst") {
        features.push("ohtst");
    }
    if cfg!(feature = "lammps") {
        features.push("lammps");
    }
    if cfg!(feature = "readcon") {
        features.push("readcon");
    }
    if cfg!(feature = "readcon-db") {
        features.push("readcon-db");
    }
    if cfg!(feature = "chemfiles") {
        features.push("chemfiles");
    }
    if cfg!(feature = "verify") {
        features.push("verify");
    }
    if cfg!(feature = "faer") {
        features.push("faer");
    }
    if cfg!(feature = "lineb-oracle") {
        features.push("lineb-oracle");
    }
    if cfg!(feature = "lineb-interpolated-hessian") {
        features.push("lineb-interpolated-hessian");
    }
    if cfg!(feature = "pafi-repro") {
        features.push("pafi-repro");
    }
    features
}

fn locate_profile_dir(exe: &std::path::Path) -> std::path::PathBuf {
    for dir in exe.ancestors().skip(1) {
        let name = dir.file_name().and_then(|s| s.to_str());
        let from_deps = (name == Some("deps")).then(|| dir.parent()).flatten();
        if let Some(profile_dir) = from_deps {
            return profile_dir.to_path_buf();
        }
        let from_out = (name == Some("out"))
            .then_some(dir)
            .and_then(|d| d.parent())
            .and_then(|d| d.parent())
            .and_then(|d| d.parent())
            .filter(|b| b.file_name().and_then(|s| s.to_str()) == Some("build"))
            .and_then(|b| b.parent());
        if let Some(profile_dir) = from_out {
            return profile_dir.to_path_buf();
        }
    }
    panic!(
        "could not locate Cargo profile directory from {}",
        exe.display()
    );
}

fn build_fresh_cdylib() -> std::path::PathBuf {
    let exe = std::env::current_exe().expect("test executable");
    let profile_dir = locate_profile_dir(&exe);
    let profile_name = profile_dir
        .file_name()
        .and_then(|s| s.to_str())
        .expect("profile directory name")
        .to_owned();
    let parent = profile_dir.parent().expect("target directory");
    let is_target_root = |p: &std::path::Path| {
        p.join(".rustc_info.json").is_file() || p.join("CACHEDIR.TAG").is_file()
    };
    let (target_dir, target_triple) = if is_target_root(parent) {
        (parent.to_path_buf(), None)
    } else if let Some(grandparent) = parent.parent().filter(|p| is_target_root(p)) {
        (
            grandparent.to_path_buf(),
            parent
                .file_name()
                .and_then(|s| s.to_str())
                .map(str::to_owned),
        )
    } else {
        (parent.to_path_buf(), None)
    };

    let features = active_features();
    let mut nested_profile_dir = target_dir.clone();
    if let Some(triple) = &target_triple {
        nested_profile_dir.push(triple);
    }
    nested_profile_dir.push(&profile_name);

    let lib_path = nested_profile_dir.join("librgsaddle.so");

    let cargo = match std::env::var_os("CARGO") {
        Some(bin) => bin,
        None => std::ffi::OsString::from(env!("CARGO")),
    };
    let mut cmd = std::process::Command::new(cargo);
    cmd.arg("rustc")
        .arg("--manifest-path")
        .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml"))
        .arg("--lib")
        .arg("--locked")
        .arg("--no-default-features")
        .arg("--target-dir")
        .arg(&target_dir);
    if !features.is_empty() {
        cmd.arg("--features").arg(features.join(","));
    }
    if let Some(triple) = &target_triple {
        cmd.arg("--target").arg(triple);
    }
    match profile_name.as_str() {
        "debug" => {}
        "release" => {
            cmd.arg("--release");
        }
        other => {
            cmd.arg("--profile").arg(other);
        }
    }
    cmd.arg("--").arg("--crate-type=cdylib");
    let output = cmd.output().expect("invoke cargo rustc --lib for cdylib");
    assert!(
        output.status.success(),
        "cargo build --lib failed to refresh librgsaddle.so:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        lib_path.is_file(),
        "expected fresh cdylib at {}",
        lib_path.display()
    );
    nested_profile_dir
}
