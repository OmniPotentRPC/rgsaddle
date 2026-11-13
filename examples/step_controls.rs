// Copyright (c) rgsaddle developers
//!
//! The two decisions a host can take from energies it already holds.
//!
//! ```text
//! cargo run --example step_controls
//! ```

use rgsaddle::{TrustSchedule, halve_raised_energy, trust_update, update_trust};

fn main() {
    let kept = halve_raised_energy(1.0, 1.0, 0.4);
    let halved = halve_raised_energy(1.0, 1.0002, 0.4);
    assert!(!kept.raised);
    assert_eq!(kept.step.to_bits(), 0.4_f64.to_bits());
    assert!(halved.raised);
    assert_eq!(halved.step.to_bits(), (0.4_f64 * 0.5).to_bits());

    let sch = TrustSchedule::minimum();
    let shrunk = trust_update(0.2, -1.0, 0.0, 0.2, &sch);
    assert_eq!(
        shrunk.to_bits(),
        update_trust(0.2, 0.0, 0.2, &sch).to_bits()
    );
    let held = trust_update(0.2, 1e-16, 1.0, 0.2, &sch);
    assert_eq!(held.to_bits(), 0.2_f64.to_bits());

    println!(
        "raised {} step {} radius {} held {}",
        halved.raised, halved.step, shrunk, held
    );
}
