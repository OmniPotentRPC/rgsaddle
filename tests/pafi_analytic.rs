//! PAFI against a potential of mean force that can be integrated by hand.
//!
//! The estimator is Swinburne and Marinica, Phys. Rev. Lett. 120, 135503
//! (2018), doi:10.1103/PhysRevLett.120.135503: `dF/ds = ⟨(∇E · n)(1-ψ)⟩`.

use ndarray::{Array1, ArrayView1};
use rgsaddle::{
    ArcPath, PafiConfig, PointSurface, SaddleError, average_hyperplane, pafi_profile,
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
    let mut config = PafiConfig {
        thermal: 0.02,
        mobility: 1.0,
        time_step: 0.01,
        thermalize: 5,
        stride: 1,
        samples: 8,
        blocks: 2,
        seed: 3,
        fix_com: false,
        retain: 1,
        target_error: None,
        max_rounds: 1,
    };
    let geometry = path.geometry(0.5).expect("mid");
    // x = -0.5 sits halfway in arc length on this uniform line.
    assert!((geometry.positions[0] + 0.5).abs() < 1e-9);
    config.samples = 4;
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
    let config = PafiConfig {
        thermal,
        mobility: 1.0,
        time_step: 0.004,
        thermalize: 400,
        stride: 8,
        samples: 600,
        blocks: 6,
        seed: 11,
        fix_com: false,
        retain: 2,
        target_error: None,
        max_rounds: 1,
    };
    let (profile, _) = pafi_profile(&Widening, &path, &path.arcs(), &config).expect("profile");
    let miss = (profile.barrier - truth).abs();
    assert!(
        miss < 0.02 && miss < 4.0 * profile.barrier_error.max(0.005),
        "barrier {} truth {truth} error {}",
        profile.barrier,
        profile.barrier_error
    );
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

/// A corner between images. The integral is along the chords, so it
/// recovers the energy of the corner instead of cutting it.
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

