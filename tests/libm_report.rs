//! Report-only bits of the transcendental calls used by the stencil.
//!
//! The sites are `dneb_component` (`atan` of a squared-norm ratio),
//! the dimer rotation (`atan`, `sin`, `cos`), and `set_mode` (`acos`).
//! This test prints the bits. It does not compare targets, and a
//! difference across ISAs is not a failure. See `validation/libm_audit.md`.

fn arg(index: usize) -> f64 {
    match index {
        0 => -0.0,
        1 => 0.0,
        2 => f64::from_bits(1),
        3 => f64::MIN_POSITIVE,
        _ => -4.0 + 8.0 * (index as f64) / 999.0,
    }
}

fn acos_arg(index: usize) -> f64 {
    match index {
        0 => -0.0,
        1 => 0.0,
        2 => -1.0,
        3 => 1.0,
        _ => -1.0 + 2.0 * (index as f64) / 999.0,
    }
}

#[test]
fn print_transcendental_bits() {
    for index in 0..1000 {
        let x = arg(index);
        let ratio = (x * x) / (1.0 + x * x);
        let angle = acos_arg(index);
        println!(
            "{index} {:016x} {:016x} {:016x} {:016x} {:016x}",
            ratio.atan().to_bits(),
            x.atan().to_bits(),
            x.sin().to_bits(),
            x.cos().to_bits(),
            angle.acos().to_bits()
        );
    }
}
