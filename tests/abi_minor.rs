//! The C ABI minor is written in three places that must agree: the Rust
//! constant, the header, and the expectation of the C contract tests.

#![cfg(feature = "capi")]

fn defined_minor(source: &str, name: &str) -> u32 {
    let line = source
        .lines()
        .find_map(|line| {
            line.strip_prefix("#define ")
                .and_then(|rest| rest.strip_prefix(name))
        })
        .unwrap_or_else(|| panic!("{name} is not defined"));
    line.trim()
        .trim_end_matches('u')
        .parse()
        .unwrap_or_else(|_| panic!("{name} is not a plain number: {line}"))
}

#[test]
fn the_abi_minor_is_the_same_in_the_header_the_constant_and_the_c_tests() {
    let header = include_str!("../include/rgsaddle.h");
    let contract = include_str!("c/main_abi_contract.h");
    let minor = rgsaddle::capi::RGSADDLE_ABI_MINOR;
    assert_eq!(defined_minor(header, "RGSADDLE_ABI_MINOR "), minor);
    assert_eq!(
        defined_minor(contract, "RGSADDLE_EXPECTED_ABI_MINOR "),
        minor
    );
}
