/// Directory that holds `librgsaddle.so`.
///
/// The final cdylib stays in the profile directory (`target/debug`).
/// Cargo 1.100 places the test binary under `build/<package>/<hash>/out`
/// instead of `deps`, so the library is an ancestor of the executable.
/// Older Cargo kept the binary in `deps` and the library one level up,
/// or the library inside `deps`.
fn cdylib_dir() -> std::path::PathBuf {
    let mut dir = std::env::current_exe().expect("test executable");
    dir.pop();
    let start = dir.clone();
    loop {
        if dir.join("librgsaddle.so").exists() {
            return dir;
        }
        let deps = dir.join("deps");
        if deps.join("librgsaddle.so").exists() {
            return deps;
        }
        if !dir.pop() {
            break;
        }
    }
    panic!(
        "build the C ABI library before testing (searched from {})",
        start.display()
    );
}
