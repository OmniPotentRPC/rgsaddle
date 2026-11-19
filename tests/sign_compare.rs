fn compiler() -> std::process::Command {
    let output = std::process::Command::new("rustc")
        .arg("-vV")
        .output()
        .expect("rustc reports its host");
    let triple = String::from_utf8_lossy(&output.stdout)
        .lines()
        .find_map(|line| line.strip_prefix("host: "))
        .expect("rustc host line")
        .to_string();
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

#[test]
fn minmode_position_length_comparison_is_sign_clean() {
    let source = include_str!("../bind/ext.cpp");
    let start = source
        .find("void minmode_set_position")
        .expect("minmode_set_position");
    let body = &source[start..];
    let end = body.find("\n}\n").expect("function end");
    let function = &body[..end];
    let line = function
        .lines()
        .find(|line| line.contains("shape(0)") && line.contains("n_atoms"))
        .expect("length comparison");
    let expression = if line.contains("static_cast<std::int64_t>(x.shape(0))") {
        "static_cast<std::int64_t>(shape) != n_atoms * 3"
    } else {
        "shape != n_atoms * 3"
    };
    let code = format!(
        "#include <stddef.h>\n#include <stdint.h>\n#include <cstdint>\nint check(size_t shape, int64_t n_atoms) {{\n  return {expression};\n}}\nint main() {{ return 0; }}\n"
    );
    let dir = tempfile::tempdir().expect("temp directory");
    let source_path = dir.path().join("sign_compare.cpp");
    let output = dir.path().join("sign_compare");
    std::fs::write(&source_path, code).expect("write snippet");
    let compile = compiler()
        .arg("-std=c++17")
        .arg("-Wall")
        .arg("-Wextra")
        .arg("-Wsign-compare")
        .arg("-Werror")
        .arg(&source_path)
        .arg("-o")
        .arg(&output)
        .output()
        .expect("C++ compiler");
    assert!(
        compile.status.success(),
        "length comparison is not sign clean:\n{}",
        String::from_utf8_lossy(&compile.stderr)
    );
}
