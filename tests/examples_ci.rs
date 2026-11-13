//! Every file under `examples/` is executed by the examples CI job.

use std::fs;

#[test]
fn every_example_is_named_in_the_ci_job() {
    let workflow = include_str!("../.github/workflows/ci.yml");
    let mut missing = Vec::new();
    for entry in fs::read_dir("examples").expect("examples") {
        let name = entry.expect("entry").file_name();
        let name = name.to_string_lossy();
        if name.starts_with('.') {
            continue;
        }
        let stem = name.rsplit_once('.').map(|(stem, _)| stem).unwrap_or(&name);
        let rust = workflow.contains(&format!("--example {stem}"));
        let python = workflow.contains(&format!("examples/{name}"));
        if !rust && !python {
            missing.push(name.into_owned());
        }
    }
    assert!(missing.is_empty(), "CI does not run {missing:?}");
}
