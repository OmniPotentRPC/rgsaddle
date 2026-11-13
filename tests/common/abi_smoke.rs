//! Compile and run a C or C++ ABI smoke program against the cdylib
//! this `cargo test` just produced.
//!
//! Cargo keeps that library under the profile directory
//! (`target/debug/librgsaddle.so`, `target/debug/deps/`, or
//! `target/debug/build/<pkg>/<hash>/out/`). The smoke executable is
//! written to the temp directory, so the link records an absolute
//! runpath and the run also sets the loader path.
//!
//! The compiler comes from the `cc` crate, which honors `CC` or `CXX`.
//! When that variable is unset and the default driver cannot compile
//! (`c++` is sometimes Clang without `<cstddef>`), the harness tries
//! the other usual drivers. A compiler that is present but cannot
//! compile fails the test. The skip line is only for a machine with no
//! compiler at all, and `RGSADDLE_REQUIRE_CAPI_SMOKE=1` turns that skip
//! into a failure.

use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::SystemTime;

pub struct Smoke {
    pub label: &'static str,
    pub source: &'static str,
    pub output_stem: &'static str,
    pub cxx: bool,
    pub compile_args: &'static [&'static str],
    pub link_libm: bool,
    pub token: &'static str,
    pub echo: bool,
}

pub fn run(spec: &Smoke) {
    let compiler = match choose_compiler(spec.cxx) {
        Choice::Ready(compiler) => compiler,
        Choice::Missing(reason) => {
            note_skip(spec.label, &reason);
            return;
        }
        Choice::Misconfigured(reason) => {
            panic!(
                "{}: a compiler is installed but misconfigured.\n{reason}",
                spec.label
            );
        }
    };

    let library = find_library();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out = std::env::temp_dir().join(format!("{}_{}", spec.output_stem, std::process::id()));
    let _cleanup = RemoveFile(out.clone());

    let mut cmd = compiler.command();
    cmd.args(spec.compile_args);
    cmd.arg(root.join(spec.source));
    cmd.arg("-I").arg(root.join("include"));
    link_library(&mut cmd, &library);
    if spec.link_libm {
        cmd.arg("-lm");
    }
    cmd.arg("-o").arg(&out);

    let built = cmd.output().unwrap_or_else(|err| {
        panic!(
            "{}: could not run {}: {err}",
            spec.label,
            compiler.describe()
        )
    });
    assert!(
        built.status.success(),
        "{} failed to build with {}:\n{}",
        spec.label,
        compiler.describe(),
        String::from_utf8_lossy(&built.stderr)
    );

    let ran = run_linked(&out, &library);
    let stdout = String::from_utf8_lossy(&ran.stdout);
    let stderr = String::from_utf8_lossy(&ran.stderr);
    assert!(
        ran.status.success(),
        "{} failed:\nstdout: {stdout}\nstderr: {stderr}",
        spec.label
    );
    if spec.echo {
        println!("{stdout}");
    }
    assert!(
        stdout.contains(spec.token),
        "{} stdout missing {}:\n{stdout}",
        spec.label,
        spec.token
    );
}

enum Choice {
    Ready(Compiler),
    Missing(String),
    Misconfigured(String),
}

enum Compiler {
    Detected(cc::Tool),
    Fallback(PathBuf),
}

impl Compiler {
    fn command(&self) -> Command {
        match self {
            Compiler::Detected(tool) => tool.to_command(),
            Compiler::Fallback(path) => Command::new(path),
        }
    }

    fn describe(&self) -> String {
        match self {
            Compiler::Detected(tool) => tool.path().display().to_string(),
            Compiler::Fallback(path) => path.display().to_string(),
        }
    }

    fn program(&self) -> &Path {
        match self {
            Compiler::Detected(tool) => tool.path(),
            Compiler::Fallback(path) => path,
        }
    }
}

fn choose_compiler(cxx: bool) -> Choice {
    let explicit =
        std::env::var_os(if cxx { "CXX" } else { "CC" }).is_some_and(|value| !value.is_empty());
    let mut failures = Vec::new();
    let mut saw_compiler = false;
    let mut tried_default = false;

    match detect(cxx) {
        Ok(compiler) => {
            let default_driver = is_default_driver(compiler.program(), cxx);
            tried_default = default_driver;
            match probe(&compiler, cxx) {
                Probe::Works => return Choice::Ready(compiler),
                Probe::Absent(err) => {
                    if explicit {
                        return Choice::Misconfigured(err);
                    }
                    failures.push(err);
                }
                Probe::Failed(err) => {
                    saw_compiler = true;
                    failures.push(err);
                    if explicit || !default_driver {
                        return Choice::Misconfigured(failures.join("\n"));
                    }
                }
            }
        }
        Err(err) => {
            if explicit {
                return Choice::Misconfigured(format!(
                    "{} is set but the cc crate could not use it: {err}",
                    if cxx { "CXX" } else { "CC" }
                ));
            }
            failures.push(format!("cc crate: {err}"));
        }
    }

    for name in fallback_names(cxx) {
        if tried_default && is_default_driver(Path::new(name), cxx) {
            continue;
        }
        if !driver_exists(name) {
            continue;
        }
        let compiler = Compiler::Fallback(PathBuf::from(name));
        match probe(&compiler, cxx) {
            Probe::Works => return Choice::Ready(compiler),
            Probe::Absent(err) => failures.push(err),
            Probe::Failed(err) => {
                saw_compiler = true;
                failures.push(err);
            }
        }
    }

    if saw_compiler {
        Choice::Misconfigured(failures.join("\n"))
    } else {
        let names = fallback_names(cxx).join(", ");
        Choice::Missing(format!(
            "no {} compiler is installed (cc honors {}, then the default driver; also tried {names})",
            if cxx { "C++" } else { "C" },
            if cxx { "CXX" } else { "CC" }
        ))
    }
}

fn fallback_names(cxx: bool) -> &'static [&'static str] {
    if cxx {
        &["g++", "clang++", "c++"]
    } else {
        &["gcc", "clang", "cc"]
    }
}

fn is_default_driver(path: &Path, cxx: bool) -> bool {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    if cxx {
        name == "c++" || name == "c++.exe"
    } else {
        name == "cc" || name == "cc.exe"
    }
}

fn detect(cxx: bool) -> Result<Compiler, String> {
    let triple = host_triple()?;
    let mut build = cc::Build::new();
    build
        .cargo_metadata(false)
        .cargo_warnings(false)
        .opt_level(0)
        .debug(false)
        .target(&triple)
        .host(&triple)
        .out_dir(std::env::temp_dir());
    if cxx {
        build.cpp(true);
    }
    let tool = build.try_get_compiler().map_err(|err| err.to_string())?;
    Ok(Compiler::Detected(tool))
}

fn host_triple() -> Result<String, String> {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let output = Command::new(&rustc)
        .arg("-vV")
        .output()
        .map_err(|err| format!("could not run {rustc} -vV: {err}"))?;
    if !output.status.success() {
        return Err(format!(
            "{rustc} -vV failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .find_map(|line| line.strip_prefix("host: "))
        .map(|host| host.trim().to_string())
        .ok_or_else(|| format!("{rustc} -vV did not report a host"))
}

fn driver_exists(name: &str) -> bool {
    Command::new(name).arg("--version").output().is_ok()
}

enum Probe {
    Works,
    Absent(String),
    Failed(String),
}

fn probe(compiler: &Compiler, cxx: bool) -> Probe {
    let kind = if cxx { "cxx" } else { "c" };
    let src = std::env::temp_dir().join(format!(
        "rgsaddle_{kind}_probe_{}.{}",
        std::process::id(),
        if cxx { "cpp" } else { "c" }
    ));
    let bin = std::env::temp_dir().join(format!("rgsaddle_{kind}_probe_{}", std::process::id()));
    let source = if cxx {
        "#include <cstddef>\n#include <cstdio>\nint main(){return sizeof(std::size_t) ? 0 : 1;}\n"
    } else {
        "#include <stdio.h>\nint main(void){return 0;}\n"
    };
    if let Err(err) = std::fs::write(&src, source) {
        return Probe::Failed(format!("probe source: {err}"));
    }
    let mut cmd = compiler.command();
    cmd.arg(if cxx { "-std=c++17" } else { "-std=c11" });
    cmd.arg(&src).arg("-o").arg(&bin);
    let result = cmd.output();
    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&bin);
    match result {
        Ok(output) if output.status.success() => Probe::Works,
        Ok(output) => Probe::Failed(format!(
            "{}: {}",
            compiler.describe(),
            String::from_utf8_lossy(&output.stderr).trim()
        )),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            Probe::Absent(format!("{}: {err}", compiler.describe()))
        }
        Err(err) => Probe::Failed(format!("{}: {err}", compiler.describe())),
    }
}

fn note_skip(label: &str, reason: &str) {
    let message = format!("skipping {label}: {reason}");
    if std::env::var_os("RGSADDLE_REQUIRE_CAPI_SMOKE").is_some_and(|value| value == "1") {
        panic!("{message}");
    }
    eprintln!("{message}");
}

fn find_library() -> PathBuf {
    let exe = std::env::current_exe().expect("test executable path");
    let profile = profile_dir(&exe);
    let mut found = Vec::new();
    collect_dir(&profile, &mut found);
    collect_dir(&profile.join("deps"), &mut found);
    collect_build_outputs(&profile.join("build"), &mut found);
    let best = found
        .into_iter()
        .max_by(|left, right| library_rank(left).cmp(&library_rank(right)));
    best.unwrap_or_else(|| {
        panic!(
            "could not find librgsaddle.so, librgsaddle.dylib, or librgsaddle.a under {}",
            profile.display()
        )
    })
}

fn profile_dir(exe: &Path) -> PathBuf {
    let mut dir = exe.parent().unwrap_or(exe).to_path_buf();
    loop {
        if dir.join("build").is_dir() || dir.join("deps").is_dir() {
            return dir;
        }
        if !dir.pop() {
            panic!(
                "could not find the cargo profile directory from {}",
                exe.display()
            );
        }
    }
}

fn collect_build_outputs(dir: &Path, found: &mut Vec<PathBuf>) {
    collect_build_outputs_at(dir, 0, found);
}

fn collect_build_outputs_at(dir: &Path, depth: u32, found: &mut Vec<PathBuf>) {
    if depth > 6 {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        if path.file_name() == Some(OsStr::new("out")) {
            collect_dir(&path, found);
        } else {
            collect_build_outputs_at(&path, depth + 1, found);
        }
    }
}

fn collect_dir(dir: &Path, found: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() && is_rgsaddle_artifact(&path) {
            found.push(path);
        }
    }
}

fn is_rgsaddle_artifact(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    let Some(stem) = artifact_stem(name) else {
        return false;
    };
    stem == "librgsaddle"
        || stem == "rgsaddle"
        || stem.starts_with("librgsaddle-")
        || stem.starts_with("rgsaddle-")
}

fn artifact_stem(name: &str) -> Option<&str> {
    for ext in [".so", ".dylib", ".dll", ".a", ".lib"] {
        if let Some(stem) = name.strip_suffix(ext) {
            return Some(stem);
        }
    }
    None
}

fn library_rank(path: &Path) -> (u8, SystemTime) {
    let shared = matches!(
        path.extension().and_then(|ext| ext.to_str()),
        Some("so" | "dylib" | "dll")
    );
    let modified = std::fs::metadata(path)
        .and_then(|meta| meta.modified())
        .unwrap_or(SystemTime::UNIX_EPOCH);
    (u8::from(shared), modified)
}

fn link_library(cmd: &mut Command, library: &Path) {
    let dir = library
        .parent()
        .expect("library path has a parent directory");
    if is_static(library) {
        cmd.arg(library);
        #[cfg(unix)]
        cmd.args(["-ldl", "-lpthread"]);
        #[cfg(target_os = "macos")]
        cmd.args([
            "-framework",
            "CoreFoundation",
            "-framework",
            "Security",
            "-framework",
            "SystemConfiguration",
        ]);
        return;
    }
    let plain = library
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| {
            name == "librgsaddle.so" || name == "librgsaddle.dylib" || name == "rgsaddle.dll"
        });
    if plain {
        cmd.arg("-L").arg(dir);
        cmd.arg("-lrgsaddle");
    } else {
        cmd.arg(library);
    }
    let mut rpath = OsString::from("-Wl,-rpath,");
    rpath.push(dir);
    cmd.arg(rpath);
}

fn is_static(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|ext| ext.to_str()),
        Some("a" | "lib")
    )
}

fn run_linked(exe: &Path, library: &Path) -> Output {
    let mut cmd = Command::new(exe);
    if let Some(dir) = library.parent() {
        #[cfg(target_os = "macos")]
        {
            cmd.env("DYLD_LIBRARY_PATH", dir);
            cmd.env("DYLD_FALLBACK_LIBRARY_PATH", dir);
        }
        #[cfg(all(unix, not(target_os = "macos")))]
        {
            cmd.env("LD_LIBRARY_PATH", dir);
        }
        #[cfg(windows)]
        {
            let mut path = OsString::from(dir);
            if let Some(current) = std::env::var_os("PATH") {
                path.push(";");
                path.push(current);
            }
            cmd.env("PATH", path);
        }
    }
    cmd.output()
        .unwrap_or_else(|err| panic!("run {}: {err}", exe.display()))
}

struct RemoveFile(PathBuf);

impl Drop for RemoveFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}
