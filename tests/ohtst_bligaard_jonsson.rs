#![cfg(feature = "ohtst")]

//! OH-TST on the two-dimensional surfaces of Bligaard and Jónsson, Comput.
//! Phys. Commun. 169, 284–288 (2005), and Jóhannesson and Jónsson, J. Chem.
//! Phys. 115, 9644–9656 (2001).
//!
//! The angles below are the ones printed in Comput. Phys. Commun. 169,
//! 286–287 (2005), Sec. 2 and Fig. 2, checked here against quadrature of
//! `A = -kT ln ∫ exp(-V/kT)` along the line. That quadrature is independent
//! of the sampler. The peak printed as 1.24° at `T = 0.01 E_a/k_B` (p. 286)
//! is not the quadrature maximum. The 2005 paper does not print a rate with
//! an uncertainty. The LEPS rate `(4.5 ± 0.2)×10^{-7}` is J. Chem. Phys.
//! 115, 9651 (2001), Sec. III B, and is handled in `ohtst_leps`.
//! Units are those of the surfaces with unit masses, so lengths and
//! mass-weighted lengths agree.

use ndarray::array;
use rgsaddle::{
    AnalyticSurfaceDynamics, AnalyticSurfaceSampler, Hyperplane, LangevinConfig, PlaneOptimizer,
    PlaneOptimizerConfig, PlaneOptimizerStatus, ReversibleWorkConfig, TransmissionConfig,
    reversible_work, transmission_coefficient,
};

include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/common/ohtst_surfaces.rs"
));

use ohtst_surfaces::{ECKART_V0, EckartHo, eigen2, hessian, line_draws, line_free_energy};

/// Golden-section search for the maximum of `f` on `[a, b]`.
fn golden_max<F: Fn(f64) -> f64>(f: F, mut a: f64, mut b: f64, tol: f64) -> f64 {
    let ratio = 0.5 * (5.0f64.sqrt() - 1.0);
    let mut c = b - ratio * (b - a);
    let mut d = a + ratio * (b - a);
    let (mut fc, mut fd) = (f(c), f(d));
    while b - a > tol {
        if fc > fd {
            b = d;
            d = c;
            fd = fc;
            c = b - ratio * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + ratio * (b - a);
            fd = f(d);
        }
    }
    0.5 * (a + b)
}

/// The line through the Eckart saddle at the origin whose normal is
/// `(cos θ, sin θ)`.
fn eckart_line(degrees: f64) -> Hyperplane {
    let theta = degrees.to_radians();
    Hyperplane::new(array![0.0, 0.0], array![theta.cos(), theta.sin()]).unwrap()
}

fn degrees_of(plane: &Hyperplane) -> f64 {
    plane.normal()[1].atan2(plane.normal()[0]).to_degrees()
}

/// Quadrature free energy of [`eckart_line`], in units of `kT`.
fn eckart_absolute_free_energy(degrees: f64, kt: f64) -> f64 {
    let theta = degrees.to_radians();
    let direction = [-theta.sin(), theta.cos()];
    let cutoff = ECKART_V0 + 22.0 * kt;
    let mut lo = 0.0;
    let mut hi = 1.0;
    while EckartHo::energy(hi * direction[0], hi * direction[1]).0 < cutoff {
        hi *= 2.0;
    }
    for _ in 0..80 {
        let middle = 0.5 * (lo + hi);
        if EckartHo::energy(middle * direction[0], middle * direction[1]).0 < cutoff {
            lo = middle;
        } else {
            hi = middle;
        }
    }
    line_free_energy(
        EckartHo::energy,
        [0.0, 0.0],
        [theta.cos(), theta.sin()],
        kt,
        ECKART_V0,
        0.5 * (lo + hi),
    ) / kt
}

/// Angular free energy relative to the paper's reference line.
fn eckart_free_energy(degrees: f64, kt: f64) -> f64 {
    let reference = (-0.5 * 0.1_f64.atan()).to_degrees();
    eckart_absolute_free_energy(degrees, kt) - eckart_absolute_free_energy(reference, kt)
}

/// Maximum of the Eckart line free energy in a bracket of angles, with the
/// curvature there in `kT` per square radian.
fn eckart_maximum(kt: f64, lo: f64, hi: f64) -> (f64, f64) {
    let theta = golden_max(|t| eckart_free_energy(t, kt), lo, hi, 1.0e-6);
    let h = 0.05;
    let curvature = (eckart_free_energy(theta + h, kt) - 2.0 * eckart_free_energy(theta, kt)
        + eckart_free_energy(theta - h, kt))
        / h.to_radians().powi(2);
    (theta, curvature)
}

fn eckart_sampler(kt: f64, time_step: f64) -> LangevinConfig {
    LangevinConfig::default()
        .with_thermal(kt)
        .with_time_step(time_step)
        .with_friction(0.03)
        .with_burn_in(1_000)
        .with_stride(4)
        .with_records(4_000)
        .with_seed(7)
}

fn eckart_optimizer(kt: f64, time_step: f64) -> PlaneOptimizerConfig {
    PlaneOptimizerConfig::default()
        .with_sampler(eckart_sampler(kt, time_step))
        .with_translate(false)
        .with_max_rotation(0.05)
        .with_max_records(16_000)
        .with_max_iterations(60)
}

/// Turn a line about the saddle from `start` degrees to a maximum of its
/// free energy. Returns the angle and its error from the last sample's
/// torque error and the quadrature curvature.
fn eckart_turn(kt: f64, start: f64, time_step: f64, curvature: f64) -> (f64, f64) {
    let config = eckart_optimizer(kt, time_step).with_tolerance(1.0e-6);
    let optimum = PlaneOptimizer::new(&eckart_line(start), config)
        .unwrap()
        .run(&EckartHo)
        .unwrap();
    let last = optimum.trace.last().unwrap();
    let error = (last.rotation_error / kt / curvature.abs()).to_degrees();
    (degrees_of(&optimum.plane), error)
}

/// Simpson's rule on an odd number of equally spaced values spanning
/// `width`.
fn simpson_nodes(values: &[f64], width: f64) -> f64 {
    let m = values.len() - 1;
    let h = width / m as f64;
    let inner: f64 = values[1..m]
        .iter()
        .enumerate()
        .map(|(i, v)| if i % 2 == 0 { 4.0 * v } else { 2.0 * v })
        .sum();
    (values[0] + inner + values[m]) * h / 3.0
}

/// Slope of [`eckart_free_energy`] in `kT` per degree.
fn eckart_slope(degrees: f64, kt: f64) -> f64 {
    let h = 0.01;
    (eckart_free_energy(degrees + h, kt) - eckart_free_energy(degrees - h, kt)) / (2.0 * h)
}

/// Passes if the line at `paper` degrees is within `10⁻⁴ kT` below the
/// maximum at `quadrature`. A broad maximum fixes its angle poorly, so a
/// rounded angle is judged by the free energy it gives up.
fn assert_near_maximum(paper: f64, quadrature: f64, kt: f64) {
    let gap = eckart_free_energy(quadrature, kt) - eckart_free_energy(paper, kt);
    assert!(
        (0.0..1.0e-4).contains(&gap),
        "{paper}° is {gap} kT below the maximum at {quadrature}°"
    );
}

/// The angles of Bligaard and Jónsson, Comput. Phys. Commun. 169, 286–287
/// (2005), Sec. 2 and Fig. 2, against quadrature of the line free energy.
/// All angles are those of the normal from the x axis, which is the line's
/// angle from the vertical axis.
///
/// The line through the saddle across its unstable mode is at 1.16°. A
/// secondary maximum appears near 43° at `T = 0.1182 E_a/k_B` (quadrature:
/// at 0.11818, at 42.85°). The low and high maxima, at 1.33° and 71.2°,
/// are equally high at 0.2266 (quadrature: 0.22676). At 0.3 they are at
/// 1.43° and 75.6°, and the line across the valley walls,
/// `π/2 - arctan(1/10)`, is the limit at extreme temperature. The paper's
/// peak at 1.24° for `T = 0.01` is not reproduced by the stated
/// `V0 + 22 kT` cutoff: quadrature puts it at 1.170°, within 0.006° of the
/// saddle line, which is its low-temperature limit.
#[test]
fn eckart_angles_of_the_paper() {
    let reference = (-0.5 * 0.1_f64.atan()).to_degrees();
    assert_eq!(eckart_free_energy(reference, 0.3 * ECKART_V0), 0.0);
    let (_, _, mode) = eigen2(hessian(EckartHo::energy, [0.0, 0.0]));
    let saddle = (mode[1] / mode[0]).atan().to_degrees();
    assert!((saddle - 1.16).abs() < 0.005, "saddle line at {saddle}°");
    let cold = eckart_maximum(0.01 * ECKART_V0, 0.5, 3.0).0;
    assert!(cold > saddle && cold - saddle < 0.01, "{cold}° at T = 0.01");

    let steepest = |t: f64| {
        let kt = t * ECKART_V0;
        let at = golden_max(|d| eckart_slope(d, kt), 25.0, 65.0, 1.0e-3);
        (at, eckart_slope(at, kt))
    };
    assert!(steepest(0.1180).1 < 0.0 && steepest(0.1184).1 > 0.0);
    let onset = steepest(0.1182).0;
    assert!((onset - 43.0).abs() < 0.5, "onset at {onset}°");

    let maxima = |t: f64| {
        let kt = t * ECKART_V0;
        let low = eckart_maximum(kt, 0.5, 3.0).0;
        let high = eckart_maximum(kt, 40.0, 88.0).0;
        (
            low,
            high,
            eckart_free_energy(high, kt) - eckart_free_energy(low, kt),
        )
    };
    let (low, high, _) = maxima(0.2266);
    assert!((low - 1.33).abs() < 0.005, "low maximum at {low}°");
    assert_near_maximum(71.2, high, 0.2266 * ECKART_V0);
    assert!(maxima(0.2263).2 < 0.0 && maxima(0.2269).2 > 0.0);

    let kt = 0.3 * ECKART_V0;
    let (low, high, gap) = maxima(0.3);
    assert_near_maximum(1.43, low, kt);
    assert_near_maximum(75.6, high, kt);
    assert!(
        (gap - 0.467453).abs() < 1.0e-5,
        "{gap} kT between the maxima"
    );

    let kt = 1000.0 * ECKART_V0;
    let hot = golden_max(
        |d| {
            let theta = d.to_radians();
            line_free_energy(
                EckartHo::energy,
                [0.0, 0.0],
                [theta.cos(), theta.sin()],
                kt,
                ECKART_V0,
                2000.0,
            ) / kt
        },
        70.0,
        89.0,
        1.0e-4,
    );
    let walls = 90.0 - 0.1f64.atan().to_degrees();
    assert!((walls - 84.29).abs() < 0.005 && (hot - walls).abs() < 0.005);
}

/// Through the engine-neutral sampler trait, the rotation-only optimizer
/// reaches the cold low-angle maximum and, above the branch crossing, the
/// maximum in either basin. Each lands within four of its errors, the last
/// torque error over the quadrature curvature.
#[test]
fn eckart_optimizer_finds_both_maxima() {
    let cold_kt = 0.01 * ECKART_V0;
    let (cold, cold_curvature) = eckart_maximum(cold_kt, 0.5, 3.0);
    let (turned, error) = eckart_turn(cold_kt, 4.0, 8.0, cold_curvature);
    assert!(
        (turned - cold).abs() < 4.0 * error,
        "{turned}° ± {error}° against {cold}°"
    );

    let kt = 0.3 * ECKART_V0;
    let (low, low_curvature) = eckart_maximum(kt, 0.5, 3.0);
    let (high, high_curvature) = eckart_maximum(kt, 60.0, 88.0);
    let (turned, error) = eckart_turn(kt, 60.0, 1.0, high_curvature);
    assert!(
        (turned - high).abs() < 4.0 * error,
        "{turned}° ± {error}° against {high}°"
    );
    let (turned, error) = eckart_turn(kt, 4.0, 8.0, low_curvature);
    assert!(
        (turned - low).abs() < 4.0 * error,
        "{turned}° ± {error}° against {low}°"
    );
}

/// The sampled torque of a capped sample falls to zero, and the norm test
/// cannot certify a zero torque. With `tolerance` zero the run ends at the
/// record cap, not in an error and not as converged.
#[test]
fn eckart_capped_sample_without_tolerance_reports_the_record_cap() {
    let cold_kt = 0.01 * ECKART_V0;
    let kt = 0.3 * ECKART_V0;
    for (kt, start, time_step) in [(cold_kt, 4.0, 8.0), (kt, 60.0, 1.0), (kt, 4.0, 8.0)] {
        let optimum = PlaneOptimizer::new(&eckart_line(start), eckart_optimizer(kt, time_step))
            .unwrap()
            .run(&EckartHo)
            .unwrap();
        assert_eq!(optimum.status, PlaneOptimizerStatus::MaxRecords);
    }
}

/// The reversible work between the two maxima at `T = 0.3 E_a/k_B` against
/// quadrature. The path is split at 12° so that nine nodes a leg resolve
/// the narrow peak at 1.40°: Simpson's rule on the quadrature slopes at
/// the same nodes is within 0.002 kT of the quadrature difference, while a
/// single leg of 17 nodes would be 0.06 kT off.
#[test]
fn eckart_reversible_work_between_the_maxima() {
    let kt = 0.3 * ECKART_V0;
    let low = eckart_maximum(kt, 0.5, 3.0).0;
    let high = eckart_maximum(kt, 60.0, 88.0).0;
    let angles = [low, 12.0, high];
    let nodes = 9;
    let layout: f64 = angles
        .windows(2)
        .map(|leg| {
            let slopes: Vec<f64> = (0..nodes)
                .map(|i| eckart_slope(leg[0] + (leg[1] - leg[0]) * i as f64 / 8.0, kt))
                .collect();
            simpson_nodes(&slopes, leg[1] - leg[0])
        })
        .sum();
    let exact = eckart_free_energy(high, kt) - eckart_free_energy(low, kt);
    assert!((layout - exact).abs() < 0.005, "{layout} against {exact}");

    let planes: Vec<Hyperplane> = angles.iter().map(|&a| eckart_line(a)).collect();
    let config = ReversibleWorkConfig::default()
        .with_sampler(eckart_sampler(kt, 4.0))
        .with_nodes(nodes);
    let mut sampler = AnalyticSurfaceSampler::new(&EckartHo, config.sampler.clone());
    let work = reversible_work(&mut sampler, &planes, &config).unwrap();
    let (value, error) = (work.work / kt, work.error / kt);
    assert!(error < 0.06, "error {error} kT");
    assert!(
        (value - layout).abs() < 4.0 * error,
        "{value} ± {error} kT against {layout}"
    );
}

/// The exact rate does not depend on the dividing line, so under the
/// Newtonian dynamics of the surface `κ exp(-A/kT)` is the same on both
/// maxima at `T = 0.3 E_a/k_B`: `κ(1.405°) / κ(75.6°) = exp(-0.467453) =
/// 0.62660` by quadrature. Shots from exact draws on each line give
/// 0.26624 ± 0.00473 and 0.41809 ± 0.00611 counted at the first crossing,
/// a ratio of 0.63679 ± 0.01465. Nearly three in four trajectories recross
/// the low line, which runs 7° off the valley, and the final-state count
/// there flickers with the valley oscillation, so only first crossings are
/// counted.
#[test]
fn eckart_transmission_balances_the_free_energy_gap() {
    let kt = 0.3 * ECKART_V0;
    let low = eckart_maximum(kt, 0.5, 3.0).0;
    let high = eckart_maximum(kt, 60.0, 88.0).0;
    let gap = eckart_free_energy(high, kt) - eckart_free_energy(low, kt);
    assert!((gap - 0.467453).abs() < 1.0e-5);
    let expected = (-gap).exp();
    let shots = 8192;
    let kappa = |degrees: f64| {
        let theta = degrees.to_radians();
        let starts = line_draws(
            EckartHo::energy,
            [0.0, 0.0],
            [theta.cos(), theta.sin()],
            kt,
            60.0,
            shots,
            3,
        );
        let config = TransmissionConfig::default()
            .with_thermal(kt)
            .with_time_step(0.5)
            .with_steps(1600)
            .with_check_every(160)
            .with_shots(shots)
            .with_blocks(32);
        transmission_coefficient(
            &mut AnalyticSurfaceDynamics::new(&EckartHo),
            &eckart_line(degrees),
            &starts,
            &config,
        )
        .unwrap()
    };
    let (a, b) = (kappa(low), kappa(high));
    let ratio = a.kappa / b.kappa;
    let error = ratio * ((a.error / a.kappa).powi(2) + (b.error / b.kappa).powi(2)).sqrt();
    assert!(
        (ratio - expected).abs() < 4.0 * error,
        "{ratio} ± {error} against {expected}"
    );
    assert!(a.kappa < 1.0 && b.kappa < 1.0 && a.unassigned == 0);
    assert!(4 * a.recrossed > 2 * shots);
}
