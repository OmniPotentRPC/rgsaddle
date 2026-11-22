/// The guide's comparison of the 600 K seed spread with the tabulated offsets
/// has to follow the differences of the numbers it lists.
#[test]
fn the_seed_spread_is_not_described_as_larger_than_the_offsets() {
    let text = include_str!("../docs/orgmode/howto/defect-pair-rates.org");
    assert!(
        !text.contains("which is larger than the offsets"),
        "the 0.0006 A spread is smaller than most listed offsets"
    );
    let rows = [
        (3.63195_f64, 3.631),
        (3.63817, 3.637),
        (3.64349, 3.643),
        (3.65095, 3.649),
        (3.65747, 3.655),
    ];
    let spread = 0.0006;
    let larger_than_spread = rows
        .iter()
        .filter(|(ours, table)| (ours - table) > spread)
        .count();
    assert_eq!(larger_than_spread, 4);
    assert!(text.contains("0.00095"));
    assert!(text.contains("0.00117"));
    assert!(text.contains("0.00195"));
    assert!(text.contains("0.00247"));
    assert!(text.contains("0.00049"));
    assert!(text.contains("smaller than the offsets"));
}
