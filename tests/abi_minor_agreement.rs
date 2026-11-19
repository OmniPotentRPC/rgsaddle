//! This series adds one ABI minor on top of the minor already on develop.

use std::fs;

fn read(path: &str) -> String {
    fs::read_to_string(path).unwrap_or_else(|err| panic!("read {path}: {err}"))
}

fn header_minor(header: &str) -> u32 {
    header
        .lines()
        .find_map(|line| {
            let rest = line.trim().strip_prefix("#define RGSADDLE_ABI_MINOR ")?;
            rest.strip_suffix('u')?.parse().ok()
        })
        .expect("RGSADDLE_ABI_MINOR")
}

#[test]
fn abi_minor_is_develop_plus_one_everywhere() {
    let root = env!("CARGO_MANIFEST_DIR");
    // include/rgsaddle.h on the develop base this series rebases onto is 44.
    let minor = 46u32;
    let header = read(&format!("{root}/include/rgsaddle.h"));
    assert_eq!(header_minor(&header), minor, "header minor");

    let capi = read(&format!("{root}/src/capi.rs"));
    assert!(capi.contains(&format!("pub const RGSADDLE_ABI_MINOR: u32 = {minor};")));

    let contract = read(&format!("{root}/tests/c/main_abi_contract.h"));
    assert!(contract.contains(&format!(
        "_Static_assert(RGSADDLE_ABI_MINOR == {minor}u, \"ABI minor\");"
    )));
    assert!(!contract.contains("abi47_"));
    assert!(contract.contains("abi46_signature_rgsaddle_band_set_dynamic"));
    assert!(contract.contains("abi46_signature_rgsaddle_dyneb_mask"));

    let fortran = read(&format!("{root}/fortran/rgsaddle.f90"));
    assert!(fortran.contains(&format!("RGSADDLE_ABI_MINOR = {minor}_c_int")));

    let handles = read(&format!("{root}/tests/fortran/test_handles.f90"));
    assert!(
        !handles.contains("version%minor = 44"),
        "Fortran handle test still stamps minor 44"
    );

    let pages = [
        format!("{root}/docs/orgmode/howto/c-abi.org"),
        format!("{root}/docs/orgmode/howto/fortran.org"),
        format!("{root}/docs/orgmode/howto.org"),
        format!("{root}/docs/orgmode/reference.org"),
        format!("{root}/docs/newsfragments/+dyneb.added.org"),
        format!("{root}/src/capi/sessions.rs"),
        format!("{root}/include/rgsaddle.h"),
    ];
    for path in &pages {
        let text = read(path);
        assert!(
            !text.contains("ABI minor 47") && !text.contains("minor 47") && !text.contains("1.47"),
            "{path} still says minor 46"
        );
    }

    let c_abi = read(&format!("{root}/docs/orgmode/howto/c-abi.org"));
    assert!(c_abi.contains("major 1, minor 46."));
    assert!(c_abi.contains("ABI minor 46 adds dynamic NEB."));
    assert!(c_abi.contains("at least minor 46 before resolving either symbol."));

    let fortran_doc = read(&format!("{root}/docs/orgmode/howto/fortran.org"));
    assert!(fortran_doc.contains("ABI major 1, minor 46"));
    assert!(fortran_doc.contains("from the C ABI at minor 46:"));

    let reference = read(&format!("{root}/docs/orgmode/reference.org"));
    assert!(reference.contains("=RGSADDLE_ABI_MINOR= is 45."));
    assert!(reference.contains("=rgsaddle_band_set_dynamic= (minor 46)"));
    assert!(reference.contains("=rgsaddle_dyneb_mask= (minor 46)"));

    assert!(header.contains("From ABI minor 46 a dynamic NEB step"));
    assert!(header.contains("ASE 3.29 DyNEB). ABI minor 46."));
    assert!(header.contains("ASE 3.29 DyNEB mask, ABI minor 46."));
    assert!(read(&format!("{root}/src/capi/sessions.rs")).contains("(ABI minor 46)"));
    assert!(read(&format!("{root}/docs/newsfragments/+dyneb.added.org")).contains("minor 46"));
    assert!(read(&format!("{root}/docs/orgmode/howto.org")).contains("ABI 1.46"));
}
