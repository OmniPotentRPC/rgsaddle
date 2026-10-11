//! Cargo.toml and pyproject.toml publish the same version string.

use std::path::PathBuf;

fn section_version(text: &str, header: &str) -> String {
    let mut in_section = false;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            in_section = trimmed == header;
            continue;
        }
        if !in_section || trimmed.starts_with('#') {
            continue;
        }
        let Some(rest) = trimmed.strip_prefix("version") else {
            continue;
        };
        let rest = rest.trim_start();
        let Some(rest) = rest.strip_prefix('=') else {
            continue;
        };
        return rest.trim().trim_matches('"').to_string();
    }
    panic!("no version under {header}");
}

#[test]
fn cargo_and_pyproject_versions_agree() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let cargo = std::fs::read_to_string(root.join("Cargo.toml")).expect("read Cargo.toml");
    let py = std::fs::read_to_string(root.join("pyproject.toml")).expect("read pyproject.toml");
    let cargo_version = section_version(&cargo, "[package]");
    let py_version = section_version(&py, "[project]");
    assert_eq!(cargo_version, py_version);
    assert_eq!(cargo_version, env!("CARGO_PKG_VERSION"));
}
