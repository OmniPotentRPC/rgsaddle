//! Synthetic estimates with known values and standard errors.

use rgsaddle::SaddleError;
use rgsaddle::ladder::*;

fn config() -> LadderConfig {
    LadderConfig::default()
}

fn decided(comparison: Comparison) -> Decision {
    let result = decide(&config(), comparison);
    assert!(result.is_ok(), "expected a climb, got {result:?}");
    result.unwrap()
}

fn started(value: f64) -> Ladder {
    let result = Ladder::start(
        config(),
        Estimate {
            value,
            error: 0.0,
            n: 0,
        },
    );
    assert!(result.is_ok(), "expected a ladder, got {result:?}");
    result.unwrap()
}

fn applied(ladder: &mut Ladder, comparison: Comparison) -> Decision {
    let result = ladder.apply(comparison);
    assert!(result.is_ok(), "expected a climb, got {result:?}");
    result.unwrap()
}

#[test]
fn defaults_are_the_named_thresholds() {
    let config = config();
    assert_eq!(config.relative_error, 0.03);
    assert_eq!(config.z_resolved, 2.0);
    assert_eq!(config.growth, 1.1);
    assert_eq!(config.pilot, 14);
    assert_eq!(config.cap, 2048);
    assert_eq!(config.temperature_tolerance, None);
    let changed = config.with_pilot(20);
    assert_eq!(changed.pilot, 20);
    assert_eq!(changed.cap, 2048);
}

#[test]
fn analytic_z_adopts_a_resolved_log_factor() {
    let decision = decided(Comparison::Analytic {
        log_phi: 0.5,
        error: 0.0078125,
        n: 32,
    });
    assert_eq!(decision.climb, Climb::Adopt);
    assert_eq!(decision.z, Some(64.0));
}

#[test]
fn analytic_z_stops_when_the_log_factor_is_noise() {
    let decision = decided(Comparison::Analytic {
        log_phi: 0.015625,
        error: 0.015625,
        n: 32,
    });
    assert_eq!(decision.climb, Climb::Stop);
    assert_eq!(decision.z, Some(1.0));
}

#[test]
fn a_small_log_factor_uses_the_absolute_log_error() {
    let decision = decided(Comparison::Analytic {
        log_phi: 0.0,
        error: 0.25,
        n: 14,
    });
    assert_eq!(decision.climb, Climb::Grow { n: 1070 });
    assert_eq!(decision.z, None);
}

#[test]
fn transmission_z_is_kappa_minus_one_over_its_error() {
    let decision = decided(Comparison::Transmission {
        kappa: 0.5,
        error: 0.0078125,
        n: 32,
    });
    assert_eq!(decision.climb, Climb::Adopt);
    assert_eq!(decision.z, Some(-64.0));
}

#[test]
fn transmission_stops_when_kappa_is_one_within_noise() {
    let decision = decided(Comparison::Transmission {
        kappa: 1.0,
        error: 0.015625,
        n: 32,
    });
    assert_eq!(decision.climb, Climb::Stop);
    assert_eq!(decision.z, Some(0.0));
}

#[test]
fn growing_does_not_compute_z() {
    let decision = decided(Comparison::Transmission {
        kappa: 0.5,
        error: 0.125,
        n: 14,
    });
    assert_eq!(decision.climb, Climb::Grow { n: 1070 });
    assert_eq!(decision.z, None);
}

#[test]
fn the_cap_computes_z_and_opens_a_large_difference() {
    let decision = decided(Comparison::Transmission {
        kappa: 0.5,
        error: 0.125,
        n: 2048,
    });
    assert_eq!(decision.climb, Climb::Open);
    assert_eq!(decision.z, Some(-4.0));
}

#[test]
fn the_cap_computes_z_and_stops_a_small_difference() {
    let decision = decided(Comparison::Transmission {
        kappa: 0.9921875,
        error: 0.03125,
        n: 2048,
    });
    assert_eq!(decision.climb, Climb::Stop);
    assert_eq!(decision.z, Some(-0.25));
}

#[test]
fn zero_error_does_not_form_a_z() {
    let same = decided(Comparison::Analytic {
        log_phi: 0.0,
        error: 0.0,
        n: 14,
    });
    assert_eq!(same.climb, Climb::Stop);
    assert_eq!(same.z, None);
    let moved = decided(Comparison::Analytic {
        log_phi: 0.5,
        error: 0.0,
        n: 14,
    });
    assert_eq!(moved.climb, Climb::Adopt);
    assert_eq!(moved.z, None);

    let one = decided(Comparison::Transmission {
        kappa: 1.0,
        error: 0.0,
        n: 14,
    });
    assert_eq!(one.climb, Climb::Stop);
    let half = decided(Comparison::Transmission {
        kappa: 0.5,
        error: 0.0,
        n: 14,
    });
    assert_eq!(half.climb, Climb::Adopt);

    let equal = Estimate {
        value: 1.0,
        error: 0.0,
        n: 8,
    };
    let same_rate = decided(Comparison::Independent {
        current: equal,
        next: equal,
    });
    assert_eq!(same_rate.climb, Climb::Stop);
    assert_eq!(same_rate.z, None);
    let other = Estimate {
        value: 2.0,
        error: 0.0,
        n: 8,
    };
    let differs = decided(Comparison::Independent {
        current: equal,
        next: other,
    });
    assert_eq!(differs.climb, Climb::Adopt);

    let noisy = Estimate {
        value: 1.5,
        error: 0.015625,
        n: 40,
    };
    let large_gap = decided(Comparison::Independent {
        current: equal,
        next: noisy,
    });
    assert_eq!(large_gap.climb, Climb::Adopt);
    assert_eq!(large_gap.z, None);
    let close = Estimate {
        value: 1.015625,
        error: 0.015625,
        n: 40,
    };
    let small_gap = decided(Comparison::Independent {
        current: equal,
        next: close,
    });
    assert_eq!(small_gap.climb, Climb::Stop);
}

#[test]
fn independent_rates_use_the_difference_of_the_rates() {
    let current = Estimate {
        value: 1.0,
        error: 0.015625,
        n: 40,
    };
    let next = Estimate {
        value: 2.0,
        error: 0.015625,
        n: 40,
    };
    let decision = decided(Comparison::Independent { current, next });
    let expected = 1.0 / 0.015625_f64.hypot(0.015625);
    assert_eq!(decision.climb, Climb::Adopt);
    assert_eq!(decision.z, Some(expected));
}

#[test]
fn a_longer_seed_repeats_the_prefix() {
    let short = prefix_values(11, 14);
    let long = prefix_values(11, 40);
    assert_eq!(long.len(), 40);
    assert_eq!(short.as_slice(), &long[..14]);
    assert_eq!(prefix_values(11, 14), short);
}

#[test]
fn temperatures_match_bits_or_a_named_tolerance() {
    let exact = config();
    assert!(temperatures_match(&exact, 300.0, 300.0));
    let next = f64::from_bits(300.0_f64.to_bits() + 1);
    assert!(!temperatures_match(&exact, 300.0, next));
    let wide = exact.clone().with_temperature_tolerance(1.0);
    assert!(temperatures_match(&wide, 300.0, 301.0));
    assert!(!temperatures_match(&wide, 300.0, 302.0));
    assert!(!temperatures_match(&exact, -0.0, 0.0));
    assert!(!temperatures_match(&exact, 0.0, -0.0));
    let payload = f64::from_bits(0x7ff8_0000_0000_0001);
    assert!(!temperatures_match(&exact, payload, payload));
    assert!(!temperatures_match(&wide, payload, payload));
}

#[test]
fn the_paper_band_gates_sampled_rungs_and_rejects_the_physical_scale() {
    assert!(paper_repro_passes(OHTST_PAPER_LOW));
    assert!(paper_repro_passes(OHTST_PAPER_HIGH));
    assert!(paper_repro_passes(4.34e-7));
    assert!(!paper_repro_passes(4.20148e-7));
    assert!(sampled_rungs_allowed(4.34e-7).is_ok());
    assert!(sampled_rungs_allowed(4.20148e-7).is_err());
    let scaled = physical_rate_per_s(4.34e-7);
    assert!(scaled.is_ok(), "expected a physical rate, got {scaled:?}");
    assert!(!paper_repro_passes(scaled.unwrap()));
}

#[test]
fn kappa_above_one_is_an_error() {
    let err = decide(
        &config(),
        Comparison::Transmission {
            kappa: 1.1,
            error: 0.01,
            n: 32,
        },
    );
    assert!(matches!(
        &err,
        Err(SaddleError::Invalid(text)) if text.contains("kappa")
    ));
}

#[test]
fn an_unresolved_factor_blocks_a_later_rung() {
    let mut ladder = started(10.0);
    let grow = applied(
        &mut ladder,
        Comparison::Analytic {
            log_phi: 0.0,
            error: 0.25,
            n: 14,
        },
    );
    assert_eq!(grow.climb, Climb::Grow { n: 1070 });
    assert_eq!(grow.z, None);
    let later = applied(
        &mut ladder,
        Comparison::Transmission {
            kappa: 0.5,
            error: 0.0078125,
            n: 32,
        },
    );
    assert_eq!(later.climb, Climb::Adopt);
    let recorded = ladder.record(300.0);
    assert!(recorded.is_ok(), "expected a record, got {recorded:?}");
    let record = recorded.unwrap();
    assert_eq!(record.rate, 10.0);
    assert_eq!(record.rung, 0);
    assert!(!record.open);
}

#[test]
fn an_open_climb_keeps_the_harmonic_rate() {
    let mut ladder = started(10.0);
    let open = applied(
        &mut ladder,
        Comparison::Analytic {
            log_phi: 1.0,
            error: 0.25,
            n: 2048,
        },
    );
    assert_eq!(open.climb, Climb::Open);
    assert!(open.z.is_some());
    let _ = applied(
        &mut ladder,
        Comparison::Transmission {
            kappa: 0.5,
            error: 0.0,
            n: 32,
        },
    );
    let recorded = ladder.record(300.0);
    assert!(recorded.is_ok(), "expected a record, got {recorded:?}");
    let record = recorded.unwrap();
    assert_eq!(record.rate, 10.0);
    assert!(record.open);
    assert_eq!(record.anharmonic, 1.0);
}

#[test]
fn adopting_both_factors_multiplies_them() {
    let mut ladder = started(8.0);
    applied(
        &mut ladder,
        Comparison::Analytic {
            log_phi: 2.0_f64.ln(),
            error: 0.0,
            n: 16,
        },
    );
    applied(
        &mut ladder,
        Comparison::Transmission {
            kappa: 0.5,
            error: 0.0,
            n: 16,
        },
    );
    let recorded = ladder.record(300.0);
    assert!(recorded.is_ok(), "expected a record, got {recorded:?}");
    let record = recorded.unwrap();
    assert_eq!(record.rate, 8.0);
    assert_eq!(record.anharmonic, 2.0);
    assert_eq!(record.kappa, 0.5);
    assert_eq!(record.rung, 2);
    assert!(!record.open);
}

#[test]
fn the_target_relative_error_computes_z() {
    let decision = decided(Comparison::Analytic {
        log_phi: 0.0,
        error: 0.03,
        n: 32,
    });
    assert_eq!(decision.climb, Climb::Stop);
    assert_eq!(decision.z, Some(0.0));
}

#[test]
fn an_equal_z_stops() {
    let decision = decided(Comparison::Analytic {
        log_phi: 0.0625,
        error: 0.03125,
        n: 2048,
    });
    assert_eq!(decision.climb, Climb::Stop);
    assert_eq!(decision.z, Some(2.0));
}

#[test]
fn one_zero_error_stops_on_the_threshold() {
    let current = Estimate {
        value: 1.0,
        error: 0.0,
        n: 8,
    };
    let next = Estimate {
        value: 1.0625,
        error: 0.03125,
        n: 40,
    };
    let decision = decided(Comparison::Independent { current, next });
    assert_eq!(decision.climb, Climb::Stop);
    assert_eq!(decision.z, None);
}

#[test]
fn a_large_log_keeps_the_absolute_log_error() {
    let decision = decided(Comparison::Analytic {
        log_phi: 2.0,
        error: 0.25,
        n: 14,
    });
    assert_eq!(decision.climb, Climb::Grow { n: 1070 });
    assert_eq!(decision.z, None);
}

#[test]
fn a_sampled_rate_replaces_an_unresolved_analytic_factor() {
    let mut ladder = started(10.0);
    let grow = applied(
        &mut ladder,
        Comparison::Analytic {
            log_phi: 0.0,
            error: 0.25,
            n: 14,
        },
    );
    assert_eq!(grow.climb, Climb::Grow { n: 1070 });
    let sampled = applied(
        &mut ladder,
        Comparison::Independent {
            current: Estimate {
                value: 10.0,
                error: 0.0,
                n: 0,
            },
            next: Estimate {
                value: 4.0,
                error: 0.0,
                n: 32,
            },
        },
    );
    assert_eq!(sampled.climb, Climb::Adopt);
    let record = ladder.record(300.0).unwrap();
    assert_eq!(record.rate, 4.0);
    assert_eq!(record.anharmonic, 0.4);
    assert_eq!(record.rung, 1);
    assert!(!record.open);
    assert_eq!(record.kappa, 1.0);
}

#[test]
fn a_sampled_rate_replaces_an_open_analytic_factor() {
    let mut ladder = started(10.0);
    let open = applied(
        &mut ladder,
        Comparison::Analytic {
            log_phi: 1.0,
            error: 0.25,
            n: 2048,
        },
    );
    assert_eq!(open.climb, Climb::Open);
    let sampled = applied(
        &mut ladder,
        Comparison::Independent {
            current: Estimate {
                value: 10.0,
                error: 0.0,
                n: 0,
            },
            next: Estimate {
                value: 6.0,
                error: 0.0,
                n: 40,
            },
        },
    );
    assert_eq!(sampled.climb, Climb::Adopt);
    let record = ladder.record(300.0).unwrap();
    assert_eq!(record.rate, 6.0);
    assert_eq!(record.rung, 1);
    assert!(!record.open);
}

#[test]
fn a_sampled_comparison_must_name_the_rate_in_hand() {
    let mut ladder = started(10.0);
    let err = ladder.apply(Comparison::Independent {
        current: Estimate {
            value: 11.0,
            error: 0.0,
            n: 0,
        },
        next: Estimate {
            value: 4.0,
            error: 0.0,
            n: 8,
        },
    });
    assert!(matches!(
        &err,
        Err(SaddleError::Invalid(text)) if text.contains("rate in hand")
    ));
    let record = ladder.record(300.0).unwrap();
    assert_eq!(record.rate, 10.0);
    assert_eq!(record.rung, 0);
}

#[test]
fn a_sampled_rate_does_not_follow_transmission() {
    let mut ladder = started(8.0);
    applied(
        &mut ladder,
        Comparison::Analytic {
            log_phi: 2.0_f64.ln(),
            error: 0.0,
            n: 16,
        },
    );
    applied(
        &mut ladder,
        Comparison::Transmission {
            kappa: 0.5,
            error: 0.0,
            n: 16,
        },
    );
    let err = ladder.apply(Comparison::Independent {
        current: Estimate {
            value: 8.0,
            error: 0.0,
            n: 16,
        },
        next: Estimate {
            value: 3.0,
            error: 0.0,
            n: 8,
        },
    });
    assert!(matches!(
        &err,
        Err(SaddleError::Invalid(text)) if text.contains("transmission")
    ));
    let record = ladder.record(300.0).unwrap();
    assert_eq!(record.rate, 8.0);
    assert_eq!(record.kappa, 0.5);
    assert_eq!(record.rung, 2);
}

#[test]
fn the_analytic_factor_does_not_multiply_a_sampled_rate() {
    let mut ladder = started(8.0);
    applied(
        &mut ladder,
        Comparison::Independent {
            current: Estimate {
                value: 8.0,
                error: 0.0,
                n: 0,
            },
            next: Estimate {
                value: 12.0,
                error: 0.0,
                n: 20,
            },
        },
    );
    let err = ladder.apply(Comparison::Analytic {
        log_phi: 2.0_f64.ln(),
        error: 0.0,
        n: 16,
    });
    assert!(matches!(
        &err,
        Err(SaddleError::Invalid(text)) if text.contains("plane factor")
    ));
    let record = ladder.record(300.0).unwrap();
    assert_eq!(record.rate, 12.0);
    assert_eq!(record.rung, 1);
}

#[test]
fn an_overflowed_product_is_rejected() {
    let mut ladder = started(1.0e10);
    let err = ladder.apply(Comparison::Analytic {
        log_phi: 700.0,
        error: 0.01,
        n: 32,
    });
    assert!(matches!(&err, Err(SaddleError::NonFinite(_))), "{err:?}");
    let record = ladder.record(300.0).unwrap();
    assert_eq!(record.rate, 1.0e10);
    assert!(record.rate.is_finite());
}

#[test]
fn an_underflowed_product_is_rejected() {
    let mut ladder = started(1.0e-200);
    let err = ladder.apply(Comparison::Analytic {
        log_phi: -400.0,
        error: 0.01,
        n: 32,
    });
    assert!(
        matches!(&err, Err(SaddleError::Invalid(text)) if text.contains("underflowed")),
        "{err:?}"
    );
    let record = ladder.record(300.0).unwrap();
    assert_eq!(record.rate, 1.0e-200);
}

#[test]
fn a_zero_error_overflow_does_not_store_a_non_finite_rate() {
    let mut ladder = started(1.0e10);
    let err = ladder.apply(Comparison::Analytic {
        log_phi: 700.0,
        error: 0.0,
        n: 32,
    });
    assert!(matches!(&err, Err(SaddleError::NonFinite(_))), "{err:?}");
    let record = ladder.record(300.0).unwrap();
    assert!(record.rate.is_finite() && record.rate_error.is_finite());
    assert_eq!(record.rate, 1.0e10);
}

#[test]
fn a_tiny_transmission_does_not_underflow_silently() {
    let tiny = f64::from_bits(1);
    let mut ladder = started(tiny);
    let err = ladder.apply(Comparison::Transmission {
        kappa: 0.5,
        error: 0.001,
        n: 32,
    });
    assert!(
        matches!(&err, Err(SaddleError::Invalid(text)) if text.contains("underflowed")),
        "{err:?}"
    );
    let record = ladder.record(300.0).unwrap();
    assert_eq!(record.rate, tiny);
    assert_eq!(record.rung, 0);
}

#[test]
fn a_sampled_factor_rejects_overflow_and_underflow() {
    let mut huge = started(1.0e-300);
    let overflow = huge.apply(Comparison::Independent {
        current: Estimate {
            value: 1.0e-300,
            error: 0.0,
            n: 0,
        },
        next: Estimate {
            value: 1.0e20,
            error: 0.0,
            n: 8,
        },
    });
    assert!(
        matches!(&overflow, Err(SaddleError::NonFinite(_))),
        "{overflow:?}"
    );
    assert_eq!(huge.record(300.0).unwrap().anharmonic, 1.0);

    let mut tiny = started(1.0e30);
    let underflow = tiny.apply(Comparison::Independent {
        current: Estimate {
            value: 1.0e30,
            error: 0.0,
            n: 0,
        },
        next: Estimate {
            value: 1.0e-300,
            error: 0.0,
            n: 8,
        },
    });
    assert!(
        matches!(&underflow, Err(SaddleError::Invalid(text)) if text.contains("underflowed")),
        "{underflow:?}"
    );
    assert_eq!(tiny.record(300.0).unwrap().anharmonic, 1.0);
}
