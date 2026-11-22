//! PAFI against a potential of mean force that can be integrated by hand.
//!
//! The estimator is Swinburne and Marinica, Phys. Rev. Lett. 120, 135503
//! (2018), doi:10.1103/PhysRevLett.120.135503: `dF/ds = ⟨(∇E · n)(1-ψ)⟩`.

use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, ArrayView1};
use rgsaddle::{
    ArcPath, PafiConfig, PafiIntegrator, PathFrame, PointSurface, RelaxConfig, SaddleError,
    anchored_barriers, average_hyperplane, balanced_barriers, integrate_profile, pafi_profile,
    zero_temperature_barrier,
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
