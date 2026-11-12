//! The published pins shared with the rest of the stack.
const MANIFEST: &str = include_str!("../Cargo.toml");

#[test]
fn pins_match_the_shared_stack() {
    assert!(MANIFEST.contains("rgmin = \"0.5.2\""));
    assert!(MANIFEST.contains("tag = \"v0.3.5\""));
    assert!(MANIFEST.contains("readcon-core = { version = \"=0.17.1\""));
    assert!(MANIFEST.contains("readcon-db = { version = \"=0.2.0\""));
    assert!(MANIFEST.contains("minimage = \"=0.1.2\""));
}
