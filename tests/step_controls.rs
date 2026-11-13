//! The public decisions against the comparisons the sessions already use.

use rgsaddle::{TrustSchedule, halve_raised_energy, trust_update, update_trust};

/// The translator's comparison, copied here so a change to the function
/// fails before the session is asked to take another surface call.
fn translator_raised(previous: f64, trial: f64) -> bool {
    trial > previous + 1e-4
}

#[test]
fn halve_matches_the_translator_bit_for_bit() {
    let samples = [
        (1.0, 1.0, 0.8),
        (1.0, 1.0 + 1e-4, 0.8),
        (1.0, 1.0 + 1e-4 + 1e-12, 0.8),
        (2.5, 2.4, 1.0),
        (0.0, 1e-3, 0.25),
        (f64::NAN, 1.0, 0.5),
        (1.0, f64::NAN, 0.5),
        (f64::INFINITY, 1.0, 0.3),
    ];
    for (previous, trial, step) in samples {
        let raised = translator_raised(previous, trial);
        let expected = if raised { step * 0.5 } else { step };
        let got = halve_raised_energy(previous, trial, step);
        assert_eq!(got.raised, raised, "{previous} {trial}");
        assert_eq!(got.step.to_bits(), expected.to_bits());
    }
}

#[test]
fn trust_update_matches_the_sella_sessions() {
    for sch in [TrustSchedule::minimum(), TrustSchedule::saddle()] {
        let cases = [
            (0.2, -1.0, 0.0, 0.2),
            (0.2, -0.4, -0.4, 0.2),
            (0.05, -0.2, -0.21, 0.05),
            (1.0, 0.3, 0.3, 0.1),
            (0.2, 1e-16, 4.0, 0.2),
            (0.2, -1e-14, 0.0, 0.2),
        ];
        for (radius, predicted, actual, step_norm) in cases {
            let predicted: f64 = predicted;
            let expected = if predicted.abs() >= 1e-14 {
                update_trust(radius, actual / predicted, step_norm, &sch)
            } else {
                radius
            };
            let got = trust_update(radius, predicted, actual, step_norm, &sch);
            assert_eq!(got.to_bits(), expected.to_bits(), "{predicted} {actual}");
        }
    }
}

#[cfg(feature = "capi")]
mod capi {
    use super::*;
    use rgsaddle::capi::{
        RGSADDLE_INVALID_PARAMETER, RGSADDLE_OK, rgsaddle_halve_raised_energy,
        rgsaddle_trust_update,
    };

    #[test]
    fn the_c_calls_match_the_rust_decisions() {
        let decision = halve_raised_energy(1.0, 1.0002, 0.4);
        let mut raised = -1;
        let mut step = 0.0;
        let status =
            unsafe { rgsaddle_halve_raised_energy(1.0, 1.0002, 0.4, &mut raised, &mut step) };
        assert_eq!(status, RGSADDLE_OK);
        assert_eq!(raised, i32::from(decision.raised));
        assert_eq!(step.to_bits(), decision.step.to_bits());
        let missing =
            unsafe { rgsaddle_halve_raised_energy(1.0, 1.0, 0.4, std::ptr::null_mut(), &mut step) };
        assert_eq!(missing, RGSADDLE_INVALID_PARAMETER);

        let sch = TrustSchedule::minimum();
        let radius = trust_update(0.2, -1.0, 0.0, 0.2, &sch);
        let mut out = 0.0;
        let status = unsafe {
            rgsaddle_trust_update(
                0.2,
                -1.0,
                0.0,
                0.2,
                sch.sigma_inc,
                sch.sigma_dec,
                sch.rho_inc,
                sch.rho_dec,
                sch.delta_min,
                &mut out,
            )
        };
        assert_eq!(status, RGSADDLE_OK);
        assert_eq!(out.to_bits(), radius.to_bits());
    }
}
