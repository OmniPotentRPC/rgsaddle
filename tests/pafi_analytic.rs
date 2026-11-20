//! PAFI against a potential of mean force that can be integrated by hand.
//!
//! The estimator is Swinburne and Marinica, Phys. Rev. Lett. 120, 135503
//! (2018), doi:10.1103/PhysRevLett.120.135503: `dF/ds = ⟨(∇E · n)(1-ψ)⟩`.

use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, ArrayView1};
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};

use rgsaddle::{
    ArcPath, PafiConfig, PafiIntegrator, PathFrame, PathGeometry, PointSurface, RelaxConfig,
    SaddleError, anchored_barriers, average_hyperplane, balanced_barriers,
    flat_hyperplane_gradient, harmonic_mask_shift, integrate_profile, pafi_frames, pafi_profile,
    pafi_truncated, zero_temperature_barrier,
};

struct AlongX;

impl PointSurface for AlongX {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let coordinate = x[0];
        let energy = (1.0 - coordinate * coordinate).powi(2);
        let mut gradient = Array1::zeros(x.len());
        gradient[0] = -4.0 * coordinate * (1.0 - coordinate * coordinate);
        Ok((energy, gradient))
    }
}

/// `V = (1-x²)² + ½ k(x) y²`, with `k = 1.5 + 0.5 x`.
struct Widening;

impl PointSurface for Widening {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (energy, gx, gy) = widening(x[0], x[1]);
        Ok((energy, Array1::from_vec(vec![gx, gy])))
    }
}

fn widening(x: f64, y: f64) -> (f64, f64, f64) {
    let well = (1.0 - x * x).powi(2);
    let stiffness = 1.5 + 0.5 * x;
    let energy = well + 0.5 * stiffness * y * y;
    let d_well = -4.0 * x * (1.0 - x * x);
    let gx = d_well + 0.25 * y * y;
    let gy = stiffness * y;
    (energy, gx, gy)
}

fn line(samples: usize) -> Vec<Array1<f64>> {
    (0..samples)
        .map(|i| {
            let x = -1.0 + i as f64 / (samples - 1) as f64;
            Array1::from_vec(vec![x])
        })
        .collect()
}

#[test]
fn one_dimensional_barrier_is_the_potential() {
    let frames = line(41);
    let path = ArcPath::from_positions(&frames).expect("path");
    let config = PafiConfig::default()
        .with_thermal(0.02)
        .with_mobility(1.0)
        .with_time_step(0.01)
        .with_thermalize(5)
        .with_stride(1)
        .with_samples(8)
        .with_blocks(2)
        .with_seed(3)
        .with_fix_com(false)
        .with_retain(1)
        .with_target_error(None)
        .with_max_rounds(1);
    let geometry = path.geometry(0.5).expect("mid");
    // x = -0.5 sits halfway in arc length on this uniform line.
    assert!((geometry.positions[0] + 0.5).abs() < 1e-9);
    let config = config.with_samples(4);
    let plane = average_hyperplane(&AlongX, &geometry, &config).expect("plane");
    let expected = -4.0 * (-0.5) * (1.0 - 0.25);
    assert!(
        (plane.gradient - expected).abs() < 1e-9,
        "gradient {} expected {expected}",
        plane.gradient
    );
    assert!(plane.psi.abs() < 1e-9);
    assert!(plane.error < 1e-12);
    let arcs = path.arcs();
    let (profile, _) = pafi_profile(&AlongX, &path, &arcs, &config).expect("profile");
    assert!(
        (profile.barrier - 1.0).abs() < 2.0e-3,
        "barrier {}",
        profile.barrier
    );
    assert_eq!(profile.dividing, arcs.len() - 1);
}

#[test]
fn transverse_width_shifts_the_free_energy() {
    let thermal = 0.05;
    let truth = 1.0 + 0.5 * thermal * 1.5_f64.ln();
    let frames: Vec<Array1<f64>> = (0..11)
        .map(|i| {
            let x = -1.0 + i as f64 / 10.0;
            Array1::from_vec(vec![x, 0.0])
        })
        .collect();
    let path = ArcPath::from_positions(&frames).expect("path");
    let config = PafiConfig::default()
        .with_thermal(thermal)
        .with_mobility(1.0)
        .with_time_step(0.004)
        .with_thermalize(400)
        .with_stride(8)
        .with_samples(600)
        .with_blocks(6)
        .with_seed(11)
        .with_fix_com(false)
        .with_retain(2)
        .with_target_error(None)
        .with_max_rounds(1);
    let (profile, _) = pafi_profile(&Widening, &path, &path.arcs(), &config).expect("profile");
    let miss = (profile.barrier - truth).abs();
    assert!(
        miss < 0.02 && miss < 4.0 * profile.barrier_error.max(0.005),
        "barrier {} truth {truth} error {}",
        profile.barrier,
        profile.barrier_error
    );
}

/// On the plane `x = 0` of [`Widening`] the walk is the harmonic mode
/// `k = 1.5` in `y`, and `dF/ds = ⟨y²⟩ / 4 = kT / (4 k)`. The step is
/// `μ k dt = 1.2`, where an Euler–Maruyama walk samples `⟨y²⟩` 2.5 times
/// too wide.
#[test]
fn harmonic_plane_is_sampled_exactly_at_a_long_step() {
    let (thermal, stiffness) = (0.05, 1.5);
    let frames: Vec<Array1<f64>> = (0..11)
        .map(|i| Array1::from_vec(vec![-0.5 + i as f64 / 10.0, 0.0]))
        .collect();
    let path = ArcPath::from_positions(&frames).expect("path");
    let geometry = path.geometry(path.arcs()[5]).expect("middle");
    assert!(geometry.positions[0].abs() < 1e-12);
    assert!((geometry.tangent[0] - 1.0).abs() < 1e-12);
    let config = PafiConfig::default()
        .with_thermal(thermal)
        .with_mobility(1.0)
        .with_time_step(1.2 / stiffness)
        .with_thermalize(50)
        .with_stride(1)
        .with_samples(20_000)
        .with_blocks(20)
        .with_seed(5)
        .with_fix_com(false)
        .with_max_jump(None);
    let plane = average_hyperplane(&Widening, &geometry, &config).expect("plane");
    let truth = thermal / (4.0 * stiffness);
    assert!(plane.error < 0.05 * truth, "error {}", plane.error);
    assert!(
        (plane.gradient - truth).abs() < 4.0 * plane.error,
        "gradient {} truth {truth} error {}",
        plane.gradient,
        plane.error
    );
    let euler = average_hyperplane(
        &Widening,
        &geometry,
        &config.with_integrator(PafiIntegrator::EulerMaruyama),
    )
    .expect("Euler--Maruyama plane");
    // For x[n+1] = (1-a)x[n] + sqrt(2 h kT) R[n], the stationary
    // variance is (kT/k)/(1-a/2). Here a = μ k dt = 1.2.
    let euler_truth = truth / (1.0 - 1.2 / 2.0);
    assert!(
        (euler.gradient - euler_truth).abs() < 4.0 * euler.error,
        "Euler gradient {} truth {euler_truth} error {}",
        euler.gradient,
        euler.error
    );
}

#[test]
fn trajectory_ensemble_uses_the_reference_error() {
    let frames: Vec<Array1<f64>> = (0..11)
        .map(|i| Array1::from_vec(vec![-0.5 + i as f64 / 10.0, 0.0]))
        .collect();
    let path = ArcPath::from_positions(&frames).expect("path");
    let geometry = path.geometry(path.arcs()[5]).expect("middle");
    let trajectories = 4;
    let config = PafiConfig::default()
        .with_thermal(0.05)
        .with_time_step(0.01)
        .with_thermalize(20)
        .with_stride(2)
        .with_samples(40)
        .with_blocks(4)
        .with_trajectories(trajectories)
        .with_seed(31)
        .with_fix_com(false)
        .with_retain(0)
        .with_max_jump(None);
    let ensemble = average_hyperplane(&Widening, &geometry, &config).expect("ensemble");
    let means: Vec<f64> = (0..trajectories)
        .map(|index| {
            average_hyperplane(
                &Widening,
                &geometry,
                &config
                    .clone()
                    .with_trajectories(1)
                    .with_seed(config.seed + index as u64),
            )
            .expect("trajectory")
            .gradient
        })
        .collect();
    let mean = means.iter().sum::<f64>() / trajectories as f64;
    let error = means
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        .sqrt()
        / trajectories as f64;
    assert_eq!(ensemble.gradient, mean);
    assert_eq!(ensemble.error, error);
    assert_eq!(ensemble.samples, trajectories * config.samples);
}

#[test]
fn reverse_barrier_runs_from_the_last_plane() {
    let profile = integrate_profile(
        &[0.0, 1.0, 2.0, 3.0],
        &[0.0, 2.0, -1.0, -3.0],
        &[0.1, 0.1, 0.2, 0.2],
    )
    .expect("profile");
    assert_eq!(profile.dividing, 2);
    assert!((profile.barrier - 1.5).abs() < 1.0e-12);
    let (barrier, error) = profile.reverse_barrier();
    assert!((barrier - 2.0).abs() < 1.0e-12, "reverse {barrier}");
    assert!((error - 0.02_f64.sqrt()).abs() < 1.0e-12, "error {error}");
}

/// The profile above ends 0.5 below its start. Told that the ends differ
/// by 0.1, the two barriers weigh in by their errors and the remaining
/// -0.6 is drift.
#[test]
fn balanced_barriers_close_on_the_end_gap() {
    let profile = integrate_profile(
        &[0.0, 1.0, 2.0, 3.0],
        &[0.0, 2.0, -1.0, -3.0],
        &[0.1, 0.1, 0.2, 0.2],
    )
    .expect("profile");
    let balanced = profile.balanced(0.1).expect("balanced");
    let (forward, reverse) = ((1.5, profile.barrier_error), profile.reverse_barrier());
    let (weight_f, weight_r) = (forward.1.powi(-2), reverse.1.powi(-2));
    let mean = (weight_f * forward.0 + weight_r * (reverse.0 + 0.1)) / (weight_f + weight_r);
    assert!(
        (balanced.drift + 0.6).abs() < 1.0e-12,
        "drift {}",
        balanced.drift
    );
    assert!((balanced.forward - mean).abs() < 1.0e-12);
    assert!((balanced.forward - balanced.reverse - 0.1).abs() < 1.0e-12);
    let error = ((weight_f + weight_r).recip() + 0.09).sqrt();
    assert!(
        (balanced.error - error).abs() < 1.0e-12,
        "error {}",
        balanced.error
    );
    let even = balanced_barriers((1.0, 0.1), (0.8, 0.1), 0.0).expect("even");
    assert!((even.forward - 0.9).abs() < 1.0e-12 && (even.reverse - 0.9).abs() < 1.0e-12);
    assert!(balanced_barriers((1.0, 0.0), (0.8, 0.1), 0.0).is_err());
    assert!(balanced_barriers((1.0, 0.1), (f64::NAN, 0.1), 0.0).is_err());
}

#[test]
fn balanced_barriers_handle_tiny_positive_errors() {
    let base = balanced_barriers((1.0, 1.0e-153), (2.0, 1.0e-153), 0.25).expect("1e-153");
    assert_eq!(base.forward, 1.625);
    assert_eq!(base.reverse, 1.375);
    assert_eq!(base.error, 0.625);
    assert_eq!(base.drift, -1.25);

    for err in [1.0e-154, 1.0e-300, f64::MIN_POSITIVE] {
        let out = balanced_barriers((1.0, err), (2.0, err), 0.25).expect("equal tiny errors");
        assert_eq!(out.forward, base.forward);
        assert_eq!(out.reverse, base.reverse);
        assert_eq!(out.error, base.error);
        assert_eq!(out.drift, base.drift);

        let zero_drift =
            balanced_barriers((1.0, err), (2.0, err), -1.0).expect("zero drift tiny errors");
        assert!(zero_drift.forward.is_finite());
        assert!(zero_drift.reverse.is_finite());
        assert!(zero_drift.error.is_finite() && zero_drift.error >= 0.0);
        assert_eq!(zero_drift.drift, 0.0);

        let skewed =
            balanced_barriers((1.0, err), (2.0, 1.0), 0.25).expect("skewed tiny/unit errors");
        assert_eq!(skewed.forward, 1.0);
        assert_eq!(skewed.reverse, 0.75);
        assert!(skewed.error.is_finite());
    }

    assert!(matches!(
        balanced_barriers((f64::MAX, 1.0), (f64::MAX, 1.0), -f64::MAX),
        Err(SaddleError::NonFinite(_))
    ));
}

/// The same profile anchored on its forward barrier: the reverse barrier
/// is the forward one less the end gap, the error is the forward one, and
/// the drift is the same -0.6.
#[test]
fn anchored_barriers_keep_the_forward_barrier() {
    let profile = integrate_profile(
        &[0.0, 1.0, 2.0, 3.0],
        &[0.0, 2.0, -1.0, -3.0],
        &[0.1, 0.1, 0.2, 0.2],
    )
    .expect("profile");
    let anchored = profile.anchored(0.1).expect("anchored");
    assert!((anchored.forward - 1.5).abs() < 1.0e-12);
    assert!((anchored.reverse - 1.4).abs() < 1.0e-12);
    assert!((anchored.error - profile.barrier_error).abs() < 1.0e-12);
    assert!(
        (anchored.drift + 0.6).abs() < 1.0e-12,
        "drift {}",
        anchored.drift
    );
    assert!(anchored_barriers((1.0, 0.0), (0.8, 0.1), 0.0).is_err());
    assert!(anchored_barriers((1.0, 0.1), (0.8, 0.1), f64::INFINITY).is_err());
}

#[test]
fn zero_temperature_integral_recovers_the_barrier() {
    let frames: Vec<Array1<f64>> = (0..41)
        .map(|i| {
            let x = -1.0 + 2.0 * i as f64 / 40.0;
            Array1::from_vec(vec![x])
        })
        .collect();
    let path = ArcPath::from_positions(&frames).expect("path");
    let profile = zero_temperature_barrier(&AlongX, &path, &path.arcs()).expect("integral");
    let end = profile.free_energy.last().copied().unwrap_or(f64::NAN);
    assert!(
        (profile.barrier - 1.0).abs() < 3.0e-3 && end.abs() < 2.0e-2,
        "barrier {} end {end} dividing {}",
        profile.barrier,
        profile.dividing
    );
    assert!(profile.dividing > 0 && profile.dividing < profile.arc.len() - 1);
}

/// A corner between images. The spline passes through the corner image,
/// so the integral reaches the energy of the corner instead of cutting it.
#[test]
fn zero_temperature_integral_keeps_a_corner() {
    let frames = vec![
        Array1::from_vec(vec![-1.0, 0.0]),
        Array1::from_vec(vec![0.0, 0.4]),
        Array1::from_vec(vec![1.0, 0.0]),
    ];
    let path = ArcPath::from_positions(&frames).expect("path");
    let profile = zero_temperature_barrier(&Corner, &path, &path.arcs()).expect("integral");
    let expected = 1.0 + 4.0 * 0.16;
    assert!(
        (profile.barrier - expected).abs() < 2.0e-3,
        "barrier {} expected {expected}",
        profile.barrier
    );
}

/// `V = (1-x²)² + ½ k (y - a(1-x²))²` with `a = 0.5`, `k = 4`: a valley
/// bent up from the minima at `(±1, 0)` to the saddle at `(0, a)`.
struct Bent;

impl PointSurface for Bent {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (energy, gx, gy) = bent(x[0], x[1]);
        Ok((energy, Array1::from_vec(vec![gx, gy])))
    }
}

fn bent(x: f64, y: f64) -> (f64, f64, f64) {
    let (bend, stiffness) = (0.5, 4.0);
    let floor = bend * (1.0 - x * x);
    let rise = y - floor;
    let energy = (1.0 - x * x).powi(2) + 0.5 * stiffness * rise * rise;
    let gx = -4.0 * x * (1.0 - x * x) + stiffness * rise * 2.0 * bend * x;
    (energy, gx, stiffness * rise)
}

/// `d/ds` of `-kT ln ∫ exp(-V/kT) dl` over the line `X(s) + l m(s)`,
/// `m ⟂ X'(s)`, by the trapezoid rule on `|l| ≤ 1.5`. The limits do not
/// depend on `s`, so the derivative is the average of
/// `∇V · (X' + l m')` over the line, with `m' = (-t_y', t_x') / |t| -
/// m (t · t') / |t|²` for `t = X'`. These lines leave the valley well
/// inside that range.
fn line_free_energy_slope(path: &ArcPath, arc: f64, thermal: f64) -> f64 {
    let geometry = path.geometry(arc).expect("geometry");
    let (tx, ty) = (geometry.tangent[0], geometry.tangent[1]);
    let (cx, cy) = (geometry.curvature[0], geometry.curvature[1]);
    let norm = tx.hypot(ty);
    let (mx, my) = (-ty / norm, tx / norm);
    let along = (tx * cx + ty * cy) / (norm * norm);
    let (dmx, dmy) = (-cy / norm - mx * along, cx / norm - my * along);
    let (x, y) = (geometry.positions[0], geometry.positions[1]);
    let points = 30_001;
    let step = 3.0 / (points - 1) as f64;
    let samples: Vec<(f64, f64)> = (0..points)
        .map(|k| {
            let l = -1.5 + step * k as f64;
            let (energy, gx, gy) = bent(x + l * mx, y + l * my);
            (energy, gx * (tx + l * dmx) + gy * (ty + l * dmy))
        })
        .collect();
    let low = samples
        .iter()
        .map(|sample| sample.0)
        .fold(f64::INFINITY, f64::min);
    let weights: Vec<f64> = samples
        .iter()
        .map(|sample| (-(sample.0 - low) / thermal).exp())
        .collect();
    let trapezoid = |values: &dyn Fn(usize) -> f64| {
        (0..points).map(values).sum::<f64>() - 0.5 * (values(0) + values(points - 1))
    };
    let normalization = trapezoid(&|k| weights[k]);
    trapezoid(&|k| weights[k] * samples[k].1) / normalization
}

/// Five images on the floor of a bent valley at `kT = 0.1`. The gradient
/// on every plane is the derivative of the free energy of the spline
/// hyperplanes, `-kT ln Z` by quadrature over each, and the barrier is
/// the trapezoid sum of those derivatives. Hyperplanes normal to the
/// chords between the images have derivatives up to 0.18 away.
#[test]
fn bent_valley_follows_the_spline_hyperplanes() {
    let thermal = 0.1;
    let frames: Vec<Array1<f64>> = (0..5)
        .map(|i| {
            let x = -1.0 + 0.5 * i as f64;
            Array1::from_vec(vec![x, 0.5 * (1.0 - x * x)])
        })
        .collect();
    let path = ArcPath::from_positions(&frames).expect("path");
    let knots = path.arcs();
    let mut arcs = Vec::new();
    for pair in knots.windows(2) {
        arcs.push(pair[0]);
        arcs.push(0.5 * (pair[0] + pair[1]));
    }
    arcs.push(knots[knots.len() - 1]);
    let config = PafiConfig::default()
        .with_thermal(thermal)
        .with_mobility(1.0)
        .with_time_step(0.004)
        .with_thermalize(400)
        .with_stride(8)
        .with_samples(1500)
        .with_blocks(10)
        .with_seed(17)
        .with_fix_com(false)
        .with_retain(1);
    let (profile, planes) = pafi_profile(&Bent, &path, &arcs, &config).expect("profile");
    let exact: Vec<f64> = arcs
        .iter()
        .map(|arc| line_free_energy_slope(&path, *arc, thermal))
        .collect();
    for (k, (plane, truth)) in planes.iter().zip(&exact).enumerate() {
        assert!(
            (plane.gradient - truth).abs() < 4.0 * plane.error + 2.0e-3 && plane.error < 0.05,
            "plane {k}: dF/ds {} exact {truth} error {}",
            plane.gradient,
            plane.error
        );
        assert!(plane.psi < 1.0);
    }
    let errors = vec![0.0; arcs.len()];
    let truth = integrate_profile(&arcs, &exact, &errors).expect("exact profile");
    assert!(
        (profile.barrier - truth.barrier).abs() < 4.0 * profile.barrier_error + 1.0e-3,
        "barrier {} exact {} error {}",
        profile.barrier,
        truth.barrier,
        profile.barrier_error
    );
}

/// Not-a-knot splines reproduce a cubic, so the path through samples of
/// one is that cubic, with its first three derivatives, at any parameter.
#[test]
fn spline_reproduces_a_cubic() {
    let cubic = |s: f64| [s * s * s - 2.0 * s + 1.0, 0.5 * s * s + s, -s];
    let slope = |s: f64| [3.0 * s * s - 2.0, s + 1.0, -1.0];
    let bend = |s: f64| [6.0 * s, 1.0, 0.0];
    let jerk = [6.0, 0.0, 0.0];
    for knots in [vec![0.0, 0.4, 1.1, 2.0], vec![0.0, 0.3, 0.5, 1.2, 1.6, 2.0]] {
        let frames = knots
            .iter()
            .map(|s| PathFrame {
                positions: Array1::from_vec(cubic(*s).to_vec()),
                arc: *s,
            })
            .collect();
        let path = ArcPath::new(frames).expect("path");
        for k in 0..=40 {
            let s = 2.0 * k as f64 / 40.0;
            let geometry = path.geometry(s).expect("geometry");
            for (got, want) in [
                (&geometry.positions, cubic(s)),
                (&geometry.tangent, slope(s)),
                (&geometry.curvature, bend(s)),
                (&geometry.jerk, jerk),
            ] {
                for (a, b) in got.iter().zip(want) {
                    assert!((a - b).abs() < 1.0e-12, "s {s}: {a} against {b}");
                }
            }
        }
    }
}

/// Two frames are the straight line and three the parabola through them.
#[test]
fn two_frames_are_a_line_and_three_a_parabola() {
    let line = ArcPath::from_positions(&[
        Array1::from_vec(vec![0.0, 1.0]),
        Array1::from_vec(vec![3.0, 5.0]),
    ])
    .expect("line");
    let middle = line.geometry(2.5).expect("middle");
    assert!((middle.positions[0] - 1.5).abs() < 1.0e-12);
    assert!((middle.positions[1] - 3.0).abs() < 1.0e-12);
    assert!((middle.tangent[0] - 0.6).abs() < 1.0e-12);
    assert!(middle.curvature.iter().all(|value| *value == 0.0));
    let frames = [(0.0, 0.0), (1.0, 2.0), (3.0, 0.0)]
        .iter()
        .map(|(s, y)| PathFrame {
            positions: Array1::from_vec(vec![*y]),
            arc: *s,
        })
        .collect();
    let parabola = ArcPath::new(frames).expect("parabola");
    let truth = |s: f64| s * (3.0 - s);
    for s in [0.0, 0.5, 1.5, 2.0, 3.0] {
        let geometry = parabola.geometry(s).expect("geometry");
        assert!(
            (geometry.positions[0] - truth(s)).abs() < 1.0e-12,
            "s {s}: {}",
            geometry.positions[0]
        );
        assert!((geometry.curvature[0] + 2.0).abs() < 1.0e-12);
    }
}

/// The reversed path is the same curve from the other end.
#[test]
fn reversed_path_traverses_the_same_curve() {
    let frames: Vec<Array1<f64>> = (0..6)
        .map(|i| {
            let x = -1.0 + 0.4 * i as f64;
            Array1::from_vec(vec![x, 0.5 * (1.0 - x * x) + 0.1 * x * x * x])
        })
        .collect();
    let path = ArcPath::from_positions(&frames).expect("path");
    let reversed = path.reversed().expect("reversed");
    let (first, last) = path.bounds();
    assert_eq!(reversed.bounds(), (first, last));
    for k in 0..=50 {
        let s = first + (last - first) * k as f64 / 50.0;
        let ahead = path.geometry(s).expect("ahead");
        let back = reversed.geometry(first + last - s).expect("back");
        for (a, b, sign) in [
            (&ahead.positions, &back.positions, 1.0),
            (&ahead.tangent, &back.tangent, -1.0),
            (&ahead.curvature, &back.curvature, 1.0),
            (&ahead.jerk, &back.jerk, -1.0),
        ] {
            for (x, y) in a.iter().zip(b) {
                assert!((x - sign * y).abs() < 1.0e-12, "s {s}: {x} against {y}");
            }
        }
    }
}

struct Corner;

impl PointSurface for Corner {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (x0, y) = (x[0], x[1]);
        let well = (1.0 - x0 * x0).powi(2);
        let energy = well + 4.0 * y * y;
        let mut gradient = Array1::zeros(x.len());
        gradient[0] = -4.0 * x0 * (1.0 - x0 * x0);
        gradient[1] = 8.0 * y;
        Ok((energy, gradient))
    }
}

/// `V = h (y² - 1)² - b y + c x y` with `h = 0.4`, `c = 0.5` and the tilt
/// `b`: on the plane `x = 0`, a well near `y = -1` and a deeper one near
/// `y = 1`, told apart by `∂V/∂x = c y`.
struct TwoWells {
    tilt: f64,
}

const WELL_HEIGHT: f64 = 0.4;
const WELL_COUPLING: f64 = 0.5;

impl TwoWells {
    fn energy(&self, x: f64, y: f64) -> f64 {
        WELL_HEIGHT * (y * y - 1.0).powi(2) - self.tilt * y + WELL_COUPLING * x * y
    }

    fn slope(&self, x: f64, y: f64) -> f64 {
        4.0 * WELL_HEIGHT * y * (y * y - 1.0) - self.tilt + WELL_COUPLING * x
    }

    /// The line `y = -1` from `x = -1` to `x = 1`, and its plane at `x = 0`.
    fn plane() -> rgsaddle::PathGeometry {
        let path = ArcPath::from_positions(&[
            Array1::from_vec(vec![-1.0, -1.0]),
            Array1::from_vec(vec![1.0, -1.0]),
        ])
        .expect("path");
        path.geometry(1.0).expect("plane")
    }
}

impl PointSurface for TwoWells {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (along, across) = (x[0], x[1]);
        let gradient = vec![WELL_COUPLING * across, self.slope(along, across)];
        Ok((self.energy(along, across), Array1::from_vec(gradient)))
    }
}

/// The path sits in the upper well of [`TwoWells`] at `kT = 0.1`, `2 kT`
/// above the other behind a ridge `3 kT` high. With the basin test the
/// gradient is `c ⟨y⟩` over the basin of the path, `y` below the ridge, by
/// quadrature. Without it the walk crosses into the lower well and stays.
#[test]
fn basin_test_keeps_the_plane_in_the_basin_of_the_path() {
    let thermal = 0.1;
    let surface = TwoWells { tilt: 0.1 };
    let (mut below, mut above) = (-0.5, 0.5);
    for _ in 0..100 {
        let middle = 0.5 * (below + above);
        if surface.slope(0.0, middle) > 0.0 {
            below = middle;
        } else {
            above = middle;
        }
    }
    let ridge = 0.5 * (below + above);
    let points = 40_001;
    let step = (ridge + 3.0) / (points - 1) as f64;
    let (mut weight, mut moment) = (0.0, 0.0);
    for k in 0..points {
        let y = -3.0 + step * k as f64;
        let ends = if k == 0 || k == points - 1 { 0.5 } else { 1.0 };
        let w = ends * (-surface.energy(0.0, y) / thermal).exp();
        weight += w;
        moment += w * y;
    }
    let truth = WELL_COUPLING * moment / weight;
    let geometry = TwoWells::plane();
    let config = PafiConfig::default()
        .with_thermal(thermal)
        .with_mobility(1.0)
        .with_time_step(0.01)
        .with_thermalize(200)
        .with_stride(10)
        .with_samples(10_000)
        .with_blocks(500)
        .with_seed(5)
        .with_fix_com(false)
        .with_retain(0);
    let held = average_hyperplane(&surface, &geometry, &config).expect("held");
    assert!(
        (held.gradient - truth).abs() < 4.0 * held.error + 5.0e-3 && held.error < 0.01,
        "dF/ds {} quadrature {truth} error {}",
        held.gradient,
        held.error
    );
    assert!(held.rejected > 0, "the walk never left the basin");
    let free = average_hyperplane(&surface, &geometry, &config.with_max_jump(None)).expect("free");
    assert_eq!(free.rejected, 0);
    assert!(
        free.gradient > truth + 0.25,
        "dF/ds {} without the test, {truth} in the basin",
        free.gradient
    );
}

/// Tilted until the upper well of [`TwoWells`] holds the walk for no
/// longer than a stretch, the plane is refused rather than averaged over
/// the lower well.
#[test]
fn basin_test_refuses_a_plane_that_cannot_hold_the_basin() {
    let config = PafiConfig::default()
        .with_thermal(0.1)
        .with_mobility(1.0)
        .with_time_step(0.01)
        .with_thermalize(200)
        .with_stride(10)
        .with_samples(80)
        .with_blocks(4)
        .with_seed(5)
        .with_fix_com(false);
    match average_hyperplane(&TwoWells { tilt: 0.55 }, &TwoWells::plane(), &config) {
        Err(SaddleError::Solver(message)) => {
            assert!(message.contains("left the basin"), "{message}");
        }
        other => panic!("expected the plane to be refused, got {other:?}"),
    }
}

/// A target that is not a finite positive number cannot be reached or
/// missed, so it is refused where the configuration is checked rather
/// than read as a ratio that is never at or below it.
#[test]
fn target_error_must_be_finite_and_positive() {
    let frames = line(9);
    let path = ArcPath::from_positions(&frames).expect("path");
    let arcs = path.arcs();
    let base = PafiConfig::default()
        .with_thermal(0.02)
        .with_mobility(1.0)
        .with_time_step(0.01)
        .with_thermalize(5)
        .with_stride(1)
        .with_samples(8)
        .with_blocks(2)
        .with_fix_com(false)
        .with_retain(0)
        .with_max_rounds(3);
    for target in [
        f64::NAN,
        -f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        0.0,
        -0.5,
    ] {
        let config = base.clone().with_target_error(target);
        match pafi_profile(&AlongX, &path, &arcs, &config) {
            Err(SaddleError::Invalid(message)) => {
                assert!(message.contains("target error"), "{target}: {message}");
            }
            other => panic!("target {target} was not refused: {other:?}"),
        }
        let geometry = path.geometry(0.5).expect("mid");
        assert!(
            matches!(
                average_hyperplane(&AlongX, &geometry, &config),
                Err(SaddleError::Invalid(_))
            ),
            "average_hyperplane accepted target {target}"
        );
    }
    let (profile, planes) = pafi_profile(&AlongX, &path, &arcs, &base.with_target_error(1.0e3))
        .expect("a finite positive target is accepted");
    assert_eq!(planes.len(), arcs.len());
    assert!(profile.barrier.is_finite());
}

/// [`Widening`] with one fault, to see what the quench of the basin test
/// does with it.
enum Fault {
    /// Every evaluation is an error.
    Fails,
    /// Every gradient is one entry short.
    ShortGradient,
    /// The gradient reverses on every call, so every step goes against the
    /// force that follows it and FIRE never gains speed.
    Reversing,
    /// An error on the call after the first.
    FailsLater,
}

struct Faulty {
    fault: Fault,
    calls: AtomicUsize,
}

impl Faulty {
    fn new(fault: Fault) -> Self {
        Self {
            fault,
            calls: AtomicUsize::new(0),
        }
    }
}

impl PointSurface for Faulty {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let call = self.calls.fetch_add(1, Ordering::Relaxed);
        let (energy, gx, gy) = widening(x[0], x[1]);
        match self.fault {
            Fault::Fails => Err(SaddleError::Invalid("surface failed in the quench".into())),
            Fault::ShortGradient => Ok((energy, Array1::from_vec(vec![gx]))),
            Fault::Reversing => {
                let sign = if call & 1 == 0 { 1.0 } else { -1.0 };
                Ok((energy, Array1::from_vec(vec![sign * (gx + 1.0), sign * gy])))
            }
            Fault::FailsLater if call >= 1 => {
                Err(SaddleError::Invalid("surface failed in the quench".into()))
            }
            Fault::FailsLater => Ok((energy, Array1::from_vec(vec![gx, gy]))),
        }
    }
}

fn quench_config() -> PafiConfig {
    PafiConfig::default()
        .with_thermal(0.02)
        .with_mobility(1.0)
        .with_time_step(0.01)
        .with_thermalize(20)
        .with_stride(1)
        .with_samples(8)
        .with_blocks(2)
        .with_fix_com(false)
        .with_retain(0)
}

fn widening_plane() -> rgsaddle::PathGeometry {
    let frames: Vec<Array1<f64>> = (0..9)
        .map(|i| Array1::from_vec(vec![-1.0 + i as f64 * 0.25, 0.0]))
        .collect();
    ArcPath::from_positions(&frames)
        .expect("path")
        .geometry(0.5)
        .expect("mid")
}

#[test]
fn a_surface_error_in_the_quench_is_the_error_returned() {
    let geometry = widening_plane();
    for fault in [Fault::Fails, Fault::FailsLater] {
        let surface = Faulty::new(fault);
        match average_hyperplane(&surface, &geometry, &quench_config()) {
            Err(SaddleError::Invalid(message)) => {
                assert_eq!(message, "surface failed in the quench");
            }
            other => panic!("the surface error was hidden: {other:?}"),
        }
    }
}

#[test]
fn a_gradient_of_the_wrong_length_in_the_quench_is_a_shape_error() {
    let geometry = widening_plane();
    let surface = Faulty::new(Fault::ShortGradient);
    match average_hyperplane(&surface, &geometry, &quench_config()) {
        Err(SaddleError::Shape(message)) => {
            assert!(message.contains("quench gradient"), "{message}");
        }
        other => panic!("the shape error was hidden: {other:?}"),
    }
}

#[test]
fn a_quench_that_stalls_or_spends_its_cap_is_counted_and_a_converged_one_is_not() {
    let geometry = widening_plane();
    let converged =
        average_hyperplane(&Widening, &geometry, &quench_config()).expect("a consistent surface");
    assert_eq!(converged.unconverged_quenches, 0);
    let untested = average_hyperplane(&Widening, &geometry, &quench_config().with_max_jump(None))
        .expect("a consistent surface");
    assert_eq!(untested.unconverged_quenches, 0);
    let stalled = average_hyperplane(
        &Faulty::new(Fault::Reversing),
        &geometry,
        &quench_config()
            .with_max_jump(Some(1.0e6))
            .with_quench(RelaxConfig::default()),
    )
    .expect("a stalled quench ends where it is");
    assert!(
        stalled.unconverged_quenches > 0,
        "the quench of a reversing force was not reported"
    );
}

#[test]
fn a_quench_that_spends_its_step_cap_is_counted() {
    let geometry = widening_plane();
    let capped = quench_config().with_max_jump(Some(1.0e6)).with_quench(
        RelaxConfig::default()
            .with_force_tol(1.0e-12)
            .with_max_steps(2),
    );
    let average = average_hyperplane(&Widening, &geometry, &capped).expect("a capped quench ends");
    assert!(average.unconverged_quenches > 0);
}

const RING_STIFFNESS: f64 = 20.0;
const RING_STIFFENING: f64 = 1.0;
const RING_BARRIER: f64 = 1.0;
const RING_THERMAL: f64 = 0.1;

/// `V = A (1 + a sin²(2θ)) (r - 1)² + W sin²(2θ)`: minima on the axes and
/// saddles on the diagonals of a ring, so every hop between neighbouring
/// axes is the same and its path is curved. The ring is stiffer at the
/// saddle, and the free-energy barrier is `W + ½ kT ln(1 + a)`.
struct Ring;

impl PointSurface for Ring {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let r = x[0].hypot(x[1]);
        let theta = x[1].atan2(x[0]);
        let rise = (2.0 * theta).sin().powi(2);
        let stretch = r - 1.0;
        let stiffness = RING_STIFFNESS * (1.0 + RING_STIFFENING * rise);
        let energy = stiffness * stretch * stretch + RING_BARRIER * rise;
        let radial = 2.0 * stiffness * stretch;
        let angular = 2.0
            * (4.0 * theta).sin()
            * (RING_STIFFNESS * RING_STIFFENING * stretch * stretch + RING_BARRIER)
            / (r * r);
        let gradient = Array1::from_vec(vec![
            radial * x[0] / r - angular * x[1],
            radial * x[1] / r + angular * x[0],
        ]);
        Ok((energy, gradient))
    }
}

/// Unevenly spaced frames on the ring from the x axis to the y axis, the
/// saddle among them, turned by `turn` and mirrored across the x axis
/// when asked.
fn ring_frames(turn: f64, mirror: bool) -> Vec<Array1<f64>> {
    [
        0.0, 0.06, 0.15, 0.27, 0.42, 0.6, FRAC_PI_4, 1.0, 1.15, 1.3, 1.45, FRAC_PI_2,
    ]
    .iter()
    .map(|angle| {
        let angle = if mirror { -angle } else { *angle } + turn;
        Array1::from_vec(vec![angle.cos(), angle.sin()])
    })
    .collect()
}

/// The planes through a reversed band are the same planes, and the arc
/// length runs the other way.
#[test]
fn frame_planes_do_not_depend_on_the_traversal() {
    let frames = ring_frames(0.0, false);
    let back: Vec<Array1<f64>> = frames.iter().rev().cloned().collect();
    let forward = ArcPath::from_positions(&frames)
        .expect("path")
        .frame_planes()
        .expect("planes");
    let backward = ArcPath::from_positions(&back)
        .expect("path")
        .frame_planes()
        .expect("planes");
    let last = frames.len() - 1;
    let total = forward.arcs[last];
    for k in 0..=last {
        let sum = &forward.normals[k] + &backward.normals[last - k];
        assert!(sum.iter().all(|v| v.abs() < 1e-12), "normal {k}");
        let turn = &forward.curvatures[k] - &backward.curvatures[last - k];
        assert!(turn.iter().all(|v| v.abs() < 1e-10), "curvature {k}");
        assert!((forward.arcs[k] - (total - backward.arcs[last - k])).abs() < 1e-12);
    }
}

/// On the ring the normals are the tangents of the circle and the
/// curvature points to its centre, to second order in the spacing.
#[test]
fn frame_planes_follow_the_ring() {
    let frames = ring_frames(0.0, false);
    let planes = ArcPath::from_positions(&frames)
        .expect("path")
        .frame_planes()
        .expect("planes");
    for (k, frame) in frames.iter().enumerate() {
        let tangent = Array1::from_vec(vec![-frame[1], frame[0]]);
        let tilt = 1.0 - planes.normals[k].dot(&tangent);
        let bend = &planes.curvatures[k] + frame;
        assert!(tilt < 1e-3, "frame {k} tilts by {tilt}");
        assert!(bend.dot(&bend).sqrt() < 0.1, "frame {k} bends by {bend}");
    }
}

/// An atom that barely moves is left out of the planes at a share its
/// squared steps do not reach.
#[test]
fn frame_planes_keep_the_atoms_that_carry_the_path() {
    let frames: Vec<Array1<f64>> = (0..5)
        .map(|i| {
            let t = i as f64 / 4.0;
            Array1::from_vec(vec![t, 0.0, 0.0, 5.0 + 0.1 * t, 0.0, 0.0])
        })
        .collect();
    let path = ArcPath::from_positions(&frames).expect("path");
    let local = path.truncated_planes(0.95).expect("planes");
    assert_eq!(local.active, vec![0]);
    assert!(
        local
            .normals
            .iter()
            .all(|n| n[3] == 0.0 && (n[0] - 1.0).abs() < 1e-12)
    );
    assert!((local.arcs[4] - 1.0).abs() < 1e-12);
    let every = path.frame_planes().expect("planes");
    assert_eq!(every.active, vec![0, 1]);
    assert!((every.arcs[4] - 1.01_f64.sqrt()).abs() < 1e-12);
}

/// Over unevenly spaced frames the barrier is the same on the hop turned
/// onto the next axis and on the hop mirrored, a reversed band gives the
/// barrier back from the other end, and the barrier keeps the entropic
/// part `½ kT ln(1 + a)` to a tenth.
#[test]
fn frame_barrier_does_not_depend_on_the_frame() {
    let config = PafiConfig::default()
        .with_thermal(RING_THERMAL)
        .with_mobility(1.0)
        .with_time_step(0.01)
        .with_thermalize(200)
        .with_stride(1)
        .with_samples(20_000)
        .with_blocks(8)
        .with_seed(5)
        .with_fix_com(false)
        .with_retain(0)
        .with_target_error(None);
    let barrier = |frames: Vec<Array1<f64>>| {
        let path = ArcPath::from_positions(&frames).expect("path");
        let (profile, _) = pafi_frames(&Ring, &path, &config).expect("pafi");
        let back = profile.reverse_barrier();
        ((profile.barrier, profile.barrier_error), back)
    };
    let (built, built_back) = barrier(ring_frames(0.0, false));
    let (reversed, reversed_back) = barrier(ring_frames(0.0, false).into_iter().rev().collect());
    let (turned, _) = barrier(ring_frames(FRAC_PI_2, false));
    let (mirrored, _) = barrier(ring_frames(0.0, true));
    let entropic = 0.5 * RING_THERMAL * (1.0 + RING_STIFFENING).ln();
    eprintln!(
        "as built {built:?} back {built_back:?}, reversed {reversed:?} back {reversed_back:?}, turned {turned:?}, mirrored {mirrored:?}, exact {}",
        RING_BARRIER + entropic
    );
    let agree = |left: (f64, f64), right: (f64, f64)| {
        (left.0 - right.0).abs() < 3.0 * left.1.hypot(right.1)
    };
    assert!(agree(built, turned), "turned {turned:?} against {built:?}");
    assert!(
        agree(built, mirrored),
        "mirrored {mirrored:?} against {built:?}"
    );
    assert!(
        agree(built, reversed_back),
        "reversed {reversed_back:?} against {built:?}"
    );
    assert!(
        agree(built_back, reversed),
        "reversed {reversed:?} against {built_back:?}"
    );
    for (label, (value, error)) in [("as built", built), ("reversed", reversed)] {
        assert!(
            (value - RING_BARRIER - entropic).abs() < 3.0 * error + 0.1 * entropic,
            "{label}: {value} ± {error} against {}",
            RING_BARRIER + entropic
        );
    }
}

const BASIN_BARRIER: f64 = 0.4;
const BASIN_COUPLING: f64 = 0.3;
const BASIN_THERMAL: f64 = 0.1;
const BASIN_PLANE: f64 = 0.5;

/// `V = (1 - x²)² + B (y² - 1)² + λ x y`: the plane `x = ½` cuts two
/// basins in `y`, the one below the deeper, and `∂V/∂x = λ y - 4x(1 - x²)`
/// tells them apart.
struct TwoBasins;

impl PointSurface for TwoBasins {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (p, q) = (x[0], x[1]);
        let energy =
            (1.0 - p * p).powi(2) + BASIN_BARRIER * (q * q - 1.0).powi(2) + BASIN_COUPLING * p * q;
        let gradient = Array1::from_vec(vec![
            -4.0 * p * (1.0 - p * p) + BASIN_COUPLING * q,
            4.0 * BASIN_BARRIER * q * (q * q - 1.0) + BASIN_COUPLING * p,
        ]);
        Ok((energy, gradient))
    }
}

/// Boltzmann average of `∂V/∂x` on the plane over `y` in `(from, to)`.
fn basin_gradient(from: f64, to: f64) -> f64 {
    let steps = 40_000;
    let width = (to - from) / steps as f64;
    let (mut weight, mut sum) = (0.0, 0.0);
    for i in 0..=steps {
        let y = from + width * i as f64;
        let point = Array1::from_vec(vec![BASIN_PLANE, y]);
        let (energy, gradient) = TwoBasins.eval(point.view()).expect("energy");
        let w = (-energy / BASIN_THERMAL).exp();
        weight += w;
        sum += w * gradient[0];
    }
    sum / weight
}

/// The ridge in `y` between the two basins on the plane.
fn basin_top() -> f64 {
    let mut top = 0.0;
    for _ in 0..50 {
        let slope = 4.0 * BASIN_BARRIER * top * (top * top - 1.0) + BASIN_COUPLING * BASIN_PLANE;
        let bend = 4.0 * BASIN_BARRIER * (3.0 * top * top - 1.0);
        top -= slope / bend;
    }
    top
}

/// On a plane that also cuts a deeper basin, the jump test keeps the chain
/// in the basin it starts from, and the average is that basin's alone.
/// Without the test the chain falls into the other one.
#[test]
fn jump_test_keeps_the_basin_of_the_path() {
    let geometry = PathGeometry {
        positions: Array1::from_vec(vec![BASIN_PLANE, 1.0]),
        tangent: Array1::from_vec(vec![1.0, 0.0]),
        curvature: Array1::zeros(2),
        jerk: Array1::zeros(2),
        arc: 0.0,
    };
    let config = PafiConfig::default()
        .with_thermal(BASIN_THERMAL)
        .with_mobility(1.0)
        .with_time_step(0.01)
        .with_thermalize(200)
        .with_stride(1)
        .with_samples(40_000)
        .with_window(200)
        .with_blocks(8)
        .with_seed(3)
        .with_fix_com(false)
        .with_retain(0);
    let kept = average_hyperplane(
        &TwoBasins,
        &geometry,
        &config.clone().with_max_jump(Some(0.5)),
    )
    .expect("kept");
    let free = average_hyperplane(&TwoBasins, &geometry, &config.clone().with_max_jump(None))
        .expect("free");
    let top = basin_top();
    let upper = basin_gradient(top, 4.0);
    let lower = basin_gradient(-4.0, top);
    eprintln!(
        "kept {:.4} ± {:.4} after {} jumps, free {:.4} ± {:.4}, upper basin {upper:.4}, lower {lower:.4}",
        kept.gradient, kept.error, kept.jumps, free.gradient, free.error
    );
    assert!(kept.jumps > 0);
    assert_eq!(free.jumps, 0);
    assert!(
        (kept.gradient - upper).abs() < 3.0 * kept.error + 0.02,
        "kept {} ± {} against {upper}",
        kept.gradient,
        kept.error
    );
    assert!((free.gradient - upper).abs() > 0.25 * (upper - lower).abs());
}

/// A window far longer than a chain stays in the basin of the path keeps
/// nothing. The plane then halves its window until it keeps one, and the
/// average is still that basin's.
#[test]
fn jump_test_halves_a_window_no_chain_outlasts() {
    let geometry = PathGeometry {
        positions: Array1::from_vec(vec![BASIN_PLANE, 1.0]),
        tangent: Array1::from_vec(vec![1.0, 0.0]),
        curvature: Array1::zeros(2),
        jerk: Array1::zeros(2),
        arc: 0.0,
    };
    let config = PafiConfig::default()
        .with_thermal(BASIN_THERMAL)
        .with_mobility(1.0)
        .with_time_step(0.01)
        .with_thermalize(200)
        .with_stride(1)
        .with_samples(40_000)
        .with_window(40_000)
        .with_blocks(8)
        .with_seed(3)
        .with_fix_com(false)
        .with_retain(0)
        .with_max_jump(Some(0.5));
    let kept = average_hyperplane(&TwoBasins, &geometry, &config).expect("kept");
    let upper = basin_gradient(basin_top(), 4.0);
    eprintln!(
        "kept {:.4} ± {:.4} from {} samples after {} jumps, upper basin {upper:.4}",
        kept.gradient, kept.error, kept.samples, kept.jumps
    );
    assert!(kept.jumps >= 4, "{} jumps", kept.jumps);
    assert!(
        (kept.gradient - upper).abs() < 3.0 * kept.error + 0.02,
        "kept {} ± {} against {upper}",
        kept.gradient,
        kept.error
    );
}

/// Transverse stiffness of the rotated double well, and `kT`.
const MASK_STIFFNESS: f64 = 6.0;
const MASK_THERMAL: f64 = 0.05;

/// `U''` of `(1 - r²)²` at the minimum `r = -1` and at the saddle `r = 0`.
const MASK_CURVATURE_MINIMUM: f64 = 8.0;
const MASK_CURVATURE_SADDLE: f64 = -4.0;

/// Curvature of `V` in the plane normal to `(cos θ, sin θ)`, at a point
/// where the path curvature along `r` is `lambda` and the transverse
/// spring is [`MASK_STIFFNESS`]. `θ = 0` is the path tangent.
fn masked_curvature(lambda: f64, theta: f64) -> f64 {
    let (along, across) = (theta.cos(), theta.sin());
    lambda * across * across + MASK_STIFFNESS * along * along
}

/// `nᵀ H⁻¹ n` for that same normal. `H = λ ê_r ê_rᵀ + k ê_w ê_wᵀ`.
fn masked_compliance(lambda: f64, theta: f64) -> f64 {
    let (along, across) = (theta.cos(), theta.sin());
    along * along / lambda + across * across / MASK_STIFFNESS
}

/// The masked barrier is `1 + (kT/2) ln(κ_s / κ_m)`, and the compliance
/// shift returns the full-tangent barrier, which is 1 at every angle.
#[test]
fn harmonic_shift_removes_the_mask_angle() {
    for degrees in [0.0_f64, 15.0, 30.0, 45.0] {
        let theta = degrees.to_radians();
        let kappa_s = masked_curvature(MASK_CURVATURE_SADDLE, theta);
        let kappa_m = masked_curvature(MASK_CURVATURE_MINIMUM, theta);
        assert!(kappa_s > 0.0 && kappa_m > 0.0, "angle {degrees}");
        let raw = 1.0 + 0.5 * MASK_THERMAL * (kappa_s / kappa_m).ln();
        let corrected = harmonic_mask_shift(
            raw,
            masked_compliance(MASK_CURVATURE_MINIMUM, 0.0),
            masked_compliance(MASK_CURVATURE_MINIMUM, theta),
            masked_compliance(MASK_CURVATURE_SADDLE, 0.0),
            masked_compliance(MASK_CURVATURE_SADDLE, theta),
            MASK_THERMAL,
        )
        .expect("shift");
        let miss = (corrected - 1.0).abs();
        assert!(
            miss < f64::EPSILON.sqrt(),
            "{degrees}° raw {raw} corrected {corrected}"
        );
    }
    assert!(harmonic_mask_shift(1.0, 1.0, -1.0, 1.0, 1.0, MASK_THERMAL).is_err());
    assert!(harmonic_mask_shift(1.0, 1.0, 1.0, 1.0, 1.0, 0.0).is_err());
}

/// `V = U(r) + ½ k w²` with the path along `θ` from the path tangent.
struct RotatedWell {
    theta: f64,
}

impl PointSurface for RotatedWell {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (along, across) = (self.theta.cos(), self.theta.sin());
        let r = x[0] * along + x[1] * across;
        let w = -x[0] * across + x[1] * along;
        let well = (1.0 - r * r).powi(2);
        let energy = well + 0.5 * MASK_STIFFNESS * w * w;
        let slope = -4.0 * r * (1.0 - r * r);
        let gradient = Array1::from_vec(vec![
            slope * along - MASK_STIFFNESS * w * across,
            slope * across + MASK_STIFFNESS * w * along,
        ]);
        Ok((energy, gradient))
    }
}

fn rotated_frames(theta: f64) -> Vec<Array1<f64>> {
    (0..11)
        .map(|i| {
            let r = -1.0 + 2.0 * i as f64 / 10.0;
            Array1::from_vec(vec![r * theta.cos(), r * theta.sin()])
        })
        .collect()
}

fn mask_sampling() -> PafiConfig {
    PafiConfig::default()
        .with_thermal(MASK_THERMAL)
        .with_mobility(1.0)
        .with_time_step(0.004)
        .with_thermalize(400)
        .with_stride(4)
        .with_samples(2_000)
        .with_blocks(8)
        .with_seed(11)
        .with_fix_com(false)
        .with_retain(0)
        .with_target_error(None)
        .with_max_rounds(1)
}

/// The full tangent gives the barrier 1 at two angles. A share that keeps
/// only the first coordinate gives the masked barrier, and the compliance
/// shift puts it back on 1.
#[test]
fn truncated_plane_has_the_masked_barrier() {
    let config = mask_sampling();
    let theta = 30.0_f64.to_radians();
    let share = 0.5;
    let kappa_s = masked_curvature(MASK_CURVATURE_SADDLE, theta);
    let kappa_m = masked_curvature(MASK_CURVATURE_MINIMUM, theta);
    let masked = 1.0 + 0.5 * MASK_THERMAL * (kappa_s / kappa_m).ln();
    for angle in [0.0, theta] {
        let path = ArcPath::from_positions(&rotated_frames(angle)).expect("path");
        let (profile, _) =
            pafi_frames(&RotatedWell { theta: angle }, &path, &config).expect("full");
        eprintln!(
            "full θ={angle:.3} barrier {} ± {}",
            profile.barrier, profile.barrier_error
        );
        assert!(
            (profile.barrier - 1.0).abs() < 3.0 * profile.barrier_error,
            "full {angle}: {} ± {}",
            profile.barrier,
            profile.barrier_error
        );
    }
    let path = ArcPath::from_positions(&rotated_frames(theta)).expect("path");
    let planes = path.truncated_planes(share).expect("planes");
    assert_eq!(planes.active, vec![0]);
    let (profile, _) =
        pafi_truncated(&RotatedWell { theta }, &path, share, &config).expect("truncated");
    let corrected = harmonic_mask_shift(
        profile.barrier,
        masked_compliance(MASK_CURVATURE_MINIMUM, 0.0),
        masked_compliance(MASK_CURVATURE_MINIMUM, theta),
        masked_compliance(MASK_CURVATURE_SADDLE, 0.0),
        masked_compliance(MASK_CURVATURE_SADDLE, theta),
        MASK_THERMAL,
    )
    .expect("shift");
    eprintln!(
        "masked {} ± {} truth {masked}, corrected {corrected}",
        profile.barrier, profile.barrier_error
    );
    assert!(
        (profile.barrier - masked).abs() < 3.0 * profile.barrier_error,
        "masked {} ± {} against {masked}",
        profile.barrier,
        profile.barrier_error
    );
    assert!(
        (corrected - 1.0).abs() < 3.0 * profile.barrier_error,
        "corrected {corrected} ± {}",
        profile.barrier_error
    );
}

/// `(x - X) · ν = 0` with a fixed normal that is not the path tangent.
/// The flat-plane derivative is the derivative of `-kT ln Z`.
#[test]
fn flat_gradient_of_a_masked_normal_matches_the_plane() {
    let amplitude = 0.4;
    let stiffness = 12.0;
    let beta: f64 = 20.0;
    let thermal = beta.recip();
    let tilt: f64 = 0.3;
    let normal = [tilt.cos(), tilt.sin()];
    let tangent_of = |x: f64| -> [f64; 2] { [1.0, amplitude * x.cos()] };
    let energy_grad = |x: f64, y: f64| -> (f64, [f64; 2]) {
        let d = y - amplitude * x.sin();
        let well = (1.0 - x * x).powi(2);
        let energy = well + 0.5 * stiffness * d * d;
        let slope = -4.0 * x * (1.0 - x * x);
        (
            energy,
            [slope - stiffness * d * amplitude * x.cos(), stiffness * d],
        )
    };
    let plane = |x0: f64, ngrid: usize| -> (f64, f64) {
        let origin = [x0, amplitude * x0.sin()];
        let across = [-normal[1], normal[0]];
        let width = 6.0;
        let step = 2.0 * width / (ngrid - 1) as f64;
        let mut partition = 0.0;
        let mut weighted = 0.0;
        for i in 0..ngrid {
            let s = -width + i as f64 * step;
            let point = [origin[0] + s * across[0], origin[1] + s * across[1]];
            let (energy, gradient) = energy_grad(point[0], point[1]);
            let weight = (-beta * energy).exp();
            let displacement = Array1::from_vec(vec![point[0] - origin[0], point[1] - origin[1]]);
            let value = flat_hyperplane_gradient(
                Array1::from_vec(gradient.to_vec()).view(),
                displacement.view(),
                Array1::from_vec(normal.to_vec()).view(),
                Array1::zeros(2).view(),
                Array1::from_vec(tangent_of(x0).to_vec()).view(),
            )
            .expect("gradient");
            partition += weight;
            weighted += weight * value;
        }
        (-thermal * partition.ln(), weighted / partition)
    };
    let xs: Vec<f64> = (-8..=8).map(|i| i as f64 * 0.05).collect();
    let fine: Vec<(f64, f64)> = xs.iter().map(|x| plane(*x, 8_001)).collect();
    let h = xs[1] - xs[0];
    // The five-point rule has truncation C h^4. The same rule at 2h
    // differs from it by 15 C h^4, so that difference bounds the
    // truncation of the spacing-h derivative of this one log Z.
    for i in 4..xs.len() - 4 {
        let derivative = (-fine[i + 2].0 + 8.0 * fine[i + 1].0 - 8.0 * fine[i - 1].0
            + fine[i - 2].0)
            / (12.0 * h);
        let wider = (-fine[i + 4].0 + 8.0 * fine[i + 2].0 - 8.0 * fine[i - 2].0 + fine[i - 4].0)
            / (24.0 * h);
        let stencil = (derivative - wider).abs();
        let miss = (fine[i].1 - derivative).abs();
        assert!(
            miss < stencil.max(f64::EPSILON.sqrt()),
            "x {}: estimate {} derivative {derivative} stencil {stencil}",
            xs[i],
            fine[i].1
        );
    }
}

/// Two atoms, `q = x_0 - x_3`. The path tangent carries a uniform drift.
/// With the center of mass held, `q` is the constrained coordinate and the
/// gradient is `U'(q) / |dX/ds|` at every sample.
struct Relative;

impl PointSurface for Relative {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let q = x[0] - x[3];
        let energy = (1.0 - q * q).powi(2);
        let slope = -4.0 * q * (1.0 - q * q);
        let mut gradient = Array1::zeros(x.len());
        gradient[0] = slope;
        gradient[3] = -slope;
        for index in [1, 2, 4, 5] {
            gradient[index] = 8.0 * x[index];
        }
        Ok((energy, gradient))
    }
}

#[test]
fn center_of_mass_uses_the_same_normal_as_the_estimator() {
    let q = -0.5;
    let positions: Array1<f64> = Array1::from_vec(vec![0.6 * q, 0.0, 0.0, -0.4 * q, 0.0, 0.0]);
    let mut tangent: Array1<f64> = Array1::from_vec(vec![0.6, 0.0, 0.0, -0.4, 0.0, 0.0]);
    let speed = tangent.dot(&tangent).sqrt();
    tangent /= speed;
    let geometry = PathGeometry {
        positions,
        tangent,
        curvature: Array1::zeros(6),
        jerk: Array1::zeros(6),
        arc: 0.0,
    };
    let slope = -4.0 * q * (1.0 - q * q);
    let expected = slope / speed;
    let config = PafiConfig::default()
        .with_thermal(0.02)
        .with_mobility(1.0)
        .with_time_step(0.01)
        .with_thermalize(4)
        .with_stride(1)
        .with_samples(8)
        .with_blocks(2)
        .with_seed(3)
        .with_fix_com(true)
        .with_retain(0)
        .with_target_error(None)
        .with_max_rounds(1);
    let plane = average_hyperplane(&Relative, &geometry, &config).expect("plane");
    eprintln!(
        "com gradient {} ± {} expected {expected}",
        plane.gradient, plane.error
    );
    assert!(
        (plane.gradient - expected).abs() < f64::EPSILON.sqrt(),
        "gradient {} expected {expected}",
        plane.gradient
    );
    assert!(plane.error < f64::EPSILON.sqrt());
}

/// Rescaling the parameter of a path by `k` multiplies its tangent by `k`
/// and its curvature by `k²`, and moves nothing in the plane. The mean `ψ`
/// is a property of the plane, and `dF/ds` follows the parameter.
#[test]
fn psi_does_not_depend_on_the_parameter_of_the_path() {
    let config = PafiConfig::default()
        .with_thermal(0.05)
        .with_mobility(1.0)
        .with_time_step(0.004)
        .with_thermalize(100)
        .with_stride(4)
        .with_samples(200)
        .with_blocks(4)
        .with_seed(5)
        .with_fix_com(false)
        .with_retain(1)
        .with_window(50)
        .with_max_jump(None);
    let at = |scale: f64| {
        let geometry = PathGeometry {
            positions: Array1::from_vec(vec![-0.3, 0.1]),
            tangent: Array1::from_vec(vec![1.0, 0.3]) * scale,
            curvature: Array1::from_vec(vec![0.2, 0.8]) * scale * scale,
            jerk: Array1::zeros(2),
            arc: 0.0,
        };
        average_hyperplane(&Widening, &geometry, &config).expect("plane")
    };
    let base = at(1.0);
    assert!(base.psi.abs() > 1.0e-3, "psi {}", base.psi);
    for scale in [0.2, 5.0] {
        let scaled = at(scale);
        assert!(
            (scaled.psi - base.psi).abs() <= 1.0e-8 * base.psi.abs(),
            "scale {scale}: psi {} against {}",
            scaled.psi,
            base.psi
        );
        let expected = scale * base.gradient;
        assert!(
            (scaled.gradient - expected).abs() <= 1.0e-8 * expected.abs().max(1.0e-3),
            "scale {scale}: gradient {} against {expected}",
            scaled.gradient
        );
    }
}

#[test]
fn frame_controls_default_to_the_documented_values_and_have_setters() {
    let config = PafiConfig::default();
    assert_eq!(
        (config.relax_tol, config.jump_tol, config.window),
        (1.0e-4, 1.0e-2, 0)
    );
    let set = config
        .with_relax_tol(1.0e-6)
        .with_jump_tol(1.0e-3)
        .with_window(16);
    assert_eq!(
        (set.relax_tol, set.jump_tol, set.window),
        (1.0e-6, 1.0e-3, 16)
    );
}
