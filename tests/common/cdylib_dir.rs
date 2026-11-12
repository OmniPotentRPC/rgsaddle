/// Directory that holds `librgsaddle.so`.
///
/// Cargo 1.100 puts the test binary under
/// `target/<profile>/build/<pkg>/<hash>/out` and the cdylib in
/// `target/<profile>` (hard-linked from another build-output directory).
/// Earlier cargo put the binary in `target/<profile>/deps` and the
/// cdylib next to that directory. Walk up from the executable until
/// one of those layouts matches.
fn cdylib_dir() -> std::path::PathBuf {
    let mut dir = std::env::current_exe().expect("test executable");
    dir.pop();
    let start = dir.clone();
    for _ in 0..8 {
        if dir.join("librgsaddle.so").is_file() {
            return dir;
        }
        let deps = dir.join("deps");
        if deps.join("librgsaddle.so").is_file() {
            return deps;
        }
        if !dir.pop() {
            break;
        }
    }
    start
}
