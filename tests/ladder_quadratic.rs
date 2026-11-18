#![cfg(feature = "ohtst")]

//! The ladder on a two-dimensional double well.
//!
//! `V = (x² − 1)² + 2 y²` has minima at `(±1, 0)` and a saddle at the
//! origin. The plane through the saddle, normal to the unstable mode, is
//! the dividing surface. The driver consumes the analytic free-energy
//! factor and the transmission coefficient that the OH-TST calls return.

use ndarray::{Array1, ArrayView1, array};
use rgsaddle::ladder::{Climb, Comparison, Estimate, Ladder, LadderConfig, decide};
use rgsaddle::{
    AnalyticSurfaceDynamics, AnharmonicConfig, Counting, Hyperplane, PointSurface, SaddleError,
    TransmissionConfig, anharmonic_correction, transmission_coefficient,
};

struct DoubleWell;

impl PointSurface for DoubleWell {
    fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (x, y) = (q[0], q[1]);
        let energy = (x * x - 1.0).powi(2) + 2.0 * y * y;
        Ok((energy, array![4.0 * x * (x * x - 1.0), 4.0 * y]))
    }
}

#[test]
fn the_double_well_climbs_with_the_same_rule_as_the_estimates() {
    let surface = DoubleWell;
    let config = LadderConfig::default();
    let minimum = array![1.0, 0.0];
    let plane = Hyperplane::new(array![0.0, 0.0], array![1.0, 0.0]).unwrap();
    let probes = AnharmonicConfig::default()
        .with_probes(config.pilot)
        .with_krylov(8)
        .with_seed(11);
    let correction = anharmonic_correction(&surface, minimum.view(), &plane, &probes).unwrap();
    let (log_phi, log_error) = correction.log_ratio(0.1);
    let analytic = Comparison::Analytic {
        log_phi,
        error: log_error,
        n: config.pilot,
    };
    let analytic_result = decide(&config, analytic);
    assert!(
        analytic_result.is_ok(),
        "expected a climb, got {analytic_result:?}"
    );
    let analytic_decision = analytic_result.unwrap();

    let starts = vec![array![0.0, 0.0]; 8];
    let shots = TransmissionConfig::default()
        .with_thermal(0.1)
        .with_friction(0.0)
        .with_time_step(0.02)
        .with_steps(80)
        .with_check_every(20)
        .with_shots(32)
        .with_blocks(4)
        .with_counting(Counting::FirstCrossing)
        .with_seed(11);
    let transmission = transmission_coefficient(
        &mut AnalyticSurfaceDynamics::new(&surface),
        &plane,
        &starts,
        &shots,
    )
    .unwrap();
    assert!(transmission.kappa > 0.0 && transmission.kappa <= 1.0);
    let shot = Comparison::Transmission {
        kappa: transmission.kappa,
        error: transmission.error,
        n: transmission.shots,
    };
    let shot_result = decide(&config, shot);
    assert!(shot_result.is_ok(), "expected a climb, got {shot_result:?}");
    let shot_decision = shot_result.unwrap();
    let relative = transmission.error / transmission.kappa;
    if relative > config.relative_error && transmission.shots < config.cap {
        assert_eq!(shot_decision.z, None);
        assert!(matches!(shot_decision.climb, Climb::Grow { .. }));
    } else {
        assert!(shot_decision.z.is_some() || transmission.error == 0.0);
    }

    let started = Ladder::start(
        config,
        Estimate {
            value: 1.0,
            error: 0.0,
            n: 0,
        },
    );
    assert!(started.is_ok(), "expected a ladder, got {started:?}");
    let mut ladder = started.unwrap();
    let applied_result = ladder.apply(analytic);
    assert!(
        applied_result.is_ok(),
        "expected a climb, got {applied_result:?}"
    );
    let applied = applied_result.unwrap();
    assert_eq!(applied, analytic_decision);
    match applied.climb {
        Climb::Grow { .. } | Climb::Open | Climb::Stop => {
            assert_eq!(ladder.record(300.0).unwrap().rate, 1.0);
        }
        Climb::Adopt => {
            let record = ladder.record(300.0).unwrap();
            assert_eq!(record.rung, 1);
            assert!(record.rate > 0.0);
        }
    }
    if !matches!(applied.climb, Climb::Grow { .. }) {
        let applied_shot = ladder.apply(shot).unwrap();
        assert_eq!(applied_shot.climb, shot_decision.climb);
        let record = ladder.record(300.0).unwrap();
        assert!(record.rate.is_finite() && record.rate > 0.0);
        assert!(record.kappa <= 1.0);
    }
}
