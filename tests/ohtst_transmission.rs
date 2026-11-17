#![cfg(feature = "ohtst")]

//! The transmission coefficient of planes through a quadratic saddle
//! against closed forms.
//!
//! On `V = ½ qᵀ H q` in mass-weighted coordinates, with eigenvalues `-λ₁`
//! and `λ₂`, no Newtonian trajectory recrosses the plane across the
//! unstable mode, and with Langevin friction `γ` its transmission
//! coefficient is that of Kramers, Physica 7, 284 (1940),
//!
//! `κ_K = √(1 + (γ/2ω_b)²) - γ/2ω_b`, `ω_b = √λ₁`.
//!
//! The rate does not depend on the plane, and a plane through the saddle
//! whose normal is turned by `θ` from the unstable mode has
//! `Z_H(θ) ∝ (λ₂ cos²θ - λ₁ sin²θ)^{-1/2}`, so
//!
//! `κ(θ) = κ_K √(cos²θ - (λ₁/λ₂) sin²θ)`.
//!
//! Starts are drawn from the exact Gaussian on each plane. The halves are
//! long enough that every trajectory has left the saddle for good, so the
//! side of the plane is the final state.

use ndarray::{Array1, ArrayView1, array};
use rgsaddle::{
    AnalyticSurfaceDynamics, Counting, Hyperplane, MassWeighted, PointSurface, SaddleError,
    TransmissionConfig, transmission_coefficient,
};

struct Quadratic {
    k: [[f64; 2]; 2],
}

impl PointSurface for Quadratic {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let g = array![
            self.k[0][0] * x[0] + self.k[0][1] * x[1],
            self.k[1][0] * x[0] + self.k[1][1] * x[1]
        ];
        Ok((0.5 * (x[0] * g[0] + x[1] * g[1]), g))
    }
}

fn kramers(friction: f64, barrier_frequency: f64) -> f64 {
    let r = friction / (2.0 * barrier_frequency);
    (1.0 + r * r).sqrt() - r
}

/// Standard normal deviates from SplitMix64 and Box–Muller.
struct Normal(u64);

impl Normal {
    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }

    fn draw(&mut self) -> f64 {
        let u = 1.0 - self.uniform();
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * self.uniform()).cos()
    }
}

/// `count` draws from `exp(-V/kT)` on the line through the origin normal
/// to `normal`, for the quadratic `k`.
fn line_starts(k: [[f64; 2]; 2], normal: [f64; 2], kt: f64, count: usize) -> Vec<Array1<f64>> {
    let t = [-normal[1], normal[0]];
    let stiffness =
        t[0] * (k[0][0] * t[0] + k[0][1] * t[1]) + t[1] * (k[1][0] * t[0] + k[1][1] * t[1]);
    assert!(stiffness > 0.0);
    let sigma = (kt / stiffness).sqrt();
    let mut normal_draws = Normal(17);
    (0..count)
        .map(|_| {
            let s = sigma * normal_draws.draw();
            array![s * t[0], s * t[1]]
        })
        .collect()
}

fn shooting(kt: f64, friction: f64, counting: Counting, shots: usize) -> TransmissionConfig {
    TransmissionConfig::default()
        .with_thermal(kt)
        .with_friction(friction)
        .with_time_step(0.05)
        .with_steps(400)
        .with_check_every(40)
        .with_shots(shots)
        .with_blocks(32)
        .with_counting(counting)
}

/// `H = diag(-1, 4)`, unit masses, `kT = 0.5`, 4096 shots. Across the
/// unstable mode every Newtonian shot scores one, which 256 shots show.
/// With `γ = 1.5`, `κ_K = 0.5` exactly, against 0.48975 ± 0.01139 by
/// Bennett–Chandler and 0.50415 ± 0.00681 at the first crossing. Turned by
/// 25° the plane has `κ = 0.88133` without friction, against 0.88647 ±
/// 0.00623 and 0.88696 ± 0.00469, and `0.44066` with `γ = 1.5`, against
/// 0.42847 ± 0.01047 and 0.44385 ± 0.00573. The first crossing has the
/// smaller error wherever trajectories recross.
#[test]
fn harmonic_saddle_against_kramers_and_turned_planes() {
    let k = [[-1.0, 0.0], [0.0, 4.0]];
    let surface = Quadratic { k };
    let kt = 0.5;
    for degrees in [0.0_f64, 25.0] {
        let theta = degrees.to_radians();
        let normal = [theta.cos(), theta.sin()];
        let plane = Hyperplane::new(array![0.0, 0.0], array![normal[0], normal[1]]).unwrap();
        let starts = line_starts(k, normal, kt, 4096);
        let geometric = (theta.cos().powi(2) - 0.25 * theta.sin().powi(2)).sqrt();
        for friction in [0.0, 1.5] {
            let exact = kramers(friction, 1.0) * geometric;
            let exact_case = degrees == 0.0 && friction == 0.0;
            let shots = if exact_case { 256 } else { 4096 };
            let mut errors = Vec::new();
            for counting in [Counting::Final, Counting::FirstCrossing] {
                let result = transmission_coefficient(
                    &mut AnalyticSurfaceDynamics::new(&surface),
                    &plane,
                    &starts,
                    &shooting(kt, friction, counting, shots),
                )
                .unwrap();
                assert!(
                    (result.kappa - exact).abs() <= 4.0 * result.error + 1.0e-12,
                    "{degrees}° γ {friction} {counting:?}: κ {} ± {} against {exact}",
                    result.kappa,
                    result.error
                );
                assert_eq!(result.unassigned, 0);
                assert_eq!(result.scores.len(), result.shots);
                let score_mean = result.scores.iter().sum::<f64>() / result.scores.len() as f64;
                assert!((score_mean - result.kappa).abs() < 1.0e-15);
                errors.push(result.error);
            }
            if exact_case {
                assert_eq!(errors, [0.0, 0.0]);
            } else {
                assert!(errors[1] < errors[0], "{errors:?}");
            }
        }
    }
}

/// The eigenvalues of a symmetric 2×2 matrix, ascending, and the unit
/// eigenvector of the first.
fn eigen(h: [[f64; 2]; 2]) -> (f64, f64, [f64; 2]) {
    let mean = 0.5 * (h[0][0] + h[1][1]);
    let radius = (0.25 * (h[0][0] - h[1][1]).powi(2) + h[0][1] * h[0][1]).sqrt();
    let low = mean - radius;
    let v = [h[0][1], low - h[0][0]];
    let norm = (v[0] * v[0] + v[1] * v[1]).sqrt();
    (low, mean + radius, [v[0] / norm, v[1] / norm])
}

/// A Cartesian saddle `K = [[-1, 0.3], [0.3, 2]]` with masses 1 and 4,
/// shot from a Cartesian plane with the masses, follows the trajectories
/// of the mass-weighted surface shot from the same plane in `q = M^{1/2} x`
/// with unit masses: the two give the same `κ` and the same counts. Both
/// match the closed form from the mass-weighted Hessian
/// `M^{-1/2} K M^{-1/2}`, whose unstable mode is 19.7° from the plane's
/// normal in `q`: 0.81404 without friction, against 0.81079 ± 0.00624 by
/// Bennett–Chandler and 0.81445 ± 0.00597 at the first crossing, and
/// 0.50476 with `γ = 1`, against 0.49536 ± 0.00984 and 0.50000 ± 0.00691.
/// The frames are compared with friction, where the noise passes through
/// the masses too.
#[test]
fn cartesian_masses_follow_the_mass_weighted_trajectories() {
    let k = [[-1.0, 0.3], [0.3, 2.0]];
    let masses = [1.0, 4.0];
    let surface = Quadratic { k };
    let kt = 0.5;
    let length = (1.0_f64 + 0.25).sqrt();
    let normal = [1.0 / length, 0.5 / length];
    let plane = Hyperplane::new(array![0.0, 0.0], array![normal[0], normal[1]]).unwrap();
    let weighted_plane =
        Hyperplane::from_cartesian(array![0.0, 0.0].view(), plane.normal(), &masses).unwrap();
    let starts = line_starts(k, normal, kt, 4096);
    let weighted_starts: Vec<Array1<f64>> = starts
        .iter()
        .map(|x| array![x[0] * masses[0].sqrt(), x[1] * masses[1].sqrt()])
        .collect();
    let weighted = MassWeighted::new(&surface, &masses).unwrap();

    let h = [
        [
            k[0][0] / masses[0],
            k[0][1] / (masses[0] * masses[1]).sqrt(),
        ],
        [
            k[1][0] / (masses[0] * masses[1]).sqrt(),
            k[1][1] / masses[1],
        ],
    ];
    let (unstable, stable, mode) = eigen(h);
    let n = weighted_plane.normal();
    let cos = (n[0] * mode[0] + n[1] * mode[1]).abs();
    let sin2 = 1.0 - cos * cos;
    assert!((cos.acos().to_degrees() - 19.7).abs() < 0.05);
    let geometric = (cos * cos - (-unstable / stable) * sin2).sqrt();

    for friction in [0.0, 1.0] {
        for counting in [Counting::Final, Counting::FirstCrossing] {
            let config = shooting(kt, friction, counting, 4096);
            let cartesian = transmission_coefficient(
                &mut AnalyticSurfaceDynamics::new(&surface),
                &plane,
                &starts,
                &config.clone().with_masses(masses.to_vec()),
            )
            .unwrap();
            if friction > 0.0 {
                let reference = transmission_coefficient(
                    &mut AnalyticSurfaceDynamics::new(&weighted),
                    &weighted_plane,
                    &weighted_starts,
                    &config,
                )
                .unwrap();
                assert!(
                    (cartesian.kappa - reference.kappa).abs() < 1.0e-12,
                    "{} against {}",
                    cartesian.kappa,
                    reference.kappa
                );
                assert_eq!(cartesian.reactive, reference.reactive);
                assert_eq!(cartesian.recrossed, reference.recrossed);
            }
            let exact = kramers(friction, (-unstable).sqrt()) * geometric;
            assert!(
                (cartesian.kappa - exact).abs() < 4.0 * cartesian.error,
                "γ {friction} {counting:?}: κ {} ± {} against {exact}",
                cartesian.kappa,
                cartesian.error
            );
        }
    }
}
