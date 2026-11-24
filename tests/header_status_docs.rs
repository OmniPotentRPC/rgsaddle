#[test]
fn header_documents_non_finite_and_abi_mismatch() {
    let header = include_str!("../include/rgsaddle.h");
    let non_finite = comment_before(header, "RGSADDLE_NON_FINITE");
    let mismatch = comment_before(header, "RGSADDLE_ABI_MISMATCH");
    assert!(
        non_finite.to_ascii_lowercase().contains("infinite"),
        "missing NON_FINITE documentation: {non_finite:?}"
    );
    assert!(
        mismatch.to_ascii_lowercase().contains("major"),
        "missing ABI_MISMATCH documentation: {mismatch:?}"
    );
    let source = include_str!("../src/capi.rs");
    let non_finite_doc = rust_doc_before(source, "pub const RGSADDLE_NON_FINITE");
    let mismatch_doc = rust_doc_before(source, "pub const RGSADDLE_ABI_MISMATCH");
    assert!(
        non_finite_doc.to_ascii_lowercase().contains("infinite"),
        "{non_finite_doc:?}"
    );
    assert!(
        mismatch_doc.to_ascii_lowercase().contains("major"),
        "{mismatch_doc:?}"
    );
}

fn comment_before(header: &str, name: &str) -> String {
    let index = header
        .find(name)
        .unwrap_or_else(|| panic!("{name} is missing from the header"));
    let before = &header[..index];
    let start = before.rfind("/**").unwrap_or(before.len());
    before[start..].to_string()
}

fn rust_doc_before(source: &str, item: &str) -> String {
    let index = source
        .find(item)
        .unwrap_or_else(|| panic!("{item} is missing"));
    let before = &source[..index];
    let start = before.rfind("///").unwrap_or(0);
    let block_start = before[..start].rfind("\n\n").map_or(0, |mark| mark + 2);
    before[block_start..index].to_string()
}
