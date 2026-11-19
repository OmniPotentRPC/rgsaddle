//! The Fortran consumer smoke takes its ABI minor from the C header.

use std::fs;

fn header_minor() -> u32 {
    let header = fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/include/rgsaddle.h"))
        .expect("header");
    header
        .lines()
        .find_map(|line| {
            let rest = line.trim().strip_prefix("#define RGSADDLE_ABI_MINOR ")?;
            rest.strip_suffix('u')?.parse().ok()
        })
        .expect("RGSADDLE_ABI_MINOR")
}

#[test]
fn fortran_smoke_reads_the_header_minor() {
    let minor = header_minor();
    let script = fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/ci/fortran_abi.sh"))
        .expect("fortran abi script");
    assert!(
        script.contains("#define RGSADDLE_ABI_MINOR"),
        "ci/fortran_abi.sh does not read include/rgsaddle.h (minor {minor})"
    );
    let smoke: Vec<_> = script
        .lines()
        .filter(|line| line.contains("version%minor"))
        .collect();
    assert_eq!(smoke.len(), 1, "{smoke:?}");
    assert!(
        smoke[0].contains("version%minor /= ${abi_minor}"),
        "Fortran smoke hardcodes a minor instead of header value {minor}: {}",
        smoke[0].trim()
    );
}
