#![cfg(feature = "ohtst")]

//! Statistical and analytic contracts for the adaptive OH-TST string.

use std::f64::consts::TAU;
use std::sync::Arc;

use ndarray::{Array1, Array2, ArrayView1, array};
use rayon::prelude::*;
use rgsaddle::{
    ADAPTIVE_OHTST_MAX_SAMPLES, ADAPTIVE_OHTST_PILOT_SAMPLES, ADAPTIVE_OHTST_TARGET_LOG_ERROR,
    AdaptiveOhTstConfig, AdaptiveOhTstStatus, AffineSymmetry, AnalyticSurfaceSampler,
    BoundaryViolation, Hyperplane, HyperplaneEnsemble, HyperplaneObservation,
    HyperplaneSampleBudget, HyperplaneSampler, LangevinConfig, PlanarBoundary, PlaneDensity,
    PointSurface, RateMultiplicity, Region, SaddleError, SymmetryOperations, SymmetryUniqueRegion,
    adaptive_ohtst_sample_count, adaptive_ohtst_string_rate, reflect_velocity,
    sample_plane_with_sampler,
};

include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/common/ohtst_surfaces.rs"
));

use ohtst_surfaces::{ECKART_V0, EckartHo, line_free_energy};

struct Density {
    plane: Hyperplane,
    thermal: f64,
    error: f64,
}

impl PlaneDensity for Density {
    fn plane(&self) -> &Hyperplane {
        &self.plane
    }

    fn thermal(&self) -> f64 {
        self.thermal
    }

    fn density(&self) -> (f64, f64) {
        (1.0, self.error)
    }
}

struct ExactPlaneSampler<F> {
    thermal: f64,
    gradient: F,
    requested: Vec<usize>,
}

impl<F: Fn(&Hyperplane) -> (f64, f64)> HyperplaneSampler for ExactPlaneSampler<F> {
    fn sample_hyperplane(
        &mut self,
        plane: &Hyperplane,
        budget: &HyperplaneSampleBudget,
    ) -> Result<HyperplaneEnsemble, SaddleError> {
        self.requested.push(budget.samples);
        let (translation, rotation) = (self.gradient)(plane);
        let normal = plane.normal();
        let tangent = array![-normal[1], normal[0]];
        let observations = (0..budget.samples)
            .map(|index| {
                let sign = if index.is_multiple_of(2) { -1.0 } else { 1.0 };
                let arm = &tangent * sign;
                HyperplaneObservation::default()
                    .with_constraint_force(-translation + sign * rotation)
                    .with_position(&plane.point() + &arm)
                    .with_projected_force(&arm * -self.thermal)
            })
            .collect();
        let mut ensemble = HyperplaneEnsemble::default();
        ensemble.observations = observations;
        ensemble.evaluations = budget.samples;
        ensemble.acceptance = 1.0;
        Ok(ensemble)
    }
}

fn plane_at_angle(degrees: f64) -> Hyperplane {
    let angle = degrees.to_radians();
    Hyperplane::new(array![0.0, 0.0], array![angle.cos(), angle.sin()]).unwrap()
}

fn plane_angle(plane: &Hyperplane) -> f64 {
    plane.normal()[1].atan2(plane.normal()[0]).to_degrees()
}

fn eckart_free_energy(angle: f64, thermal: f64) -> f64 {
    let normal = [angle.cos(), angle.sin()];
    let direction = [-normal[1], normal[0]];
    let cutoff = ECKART_V0 + 22.0 * thermal;
    let mut low = 0.0;
    let mut high = 1.0;
    while EckartHo::energy(high * direction[0], high * direction[1]).0 < cutoff {
        high *= 2.0;
    }
    for _ in 0..80 {
        let middle = 0.5 * (low + high);
        if EckartHo::energy(middle * direction[0], middle * direction[1]).0 < cutoff {
            low = middle;
        } else {
            high = middle;
        }
    }
    line_free_energy(
        EckartHo::energy,
        [0.0, 0.0],
        normal,
        thermal,
        ECKART_V0,
        0.5 * (low + high),
    )
}

fn eckart_gradient(plane: &Hyperplane, thermal: f64) -> (f64, f64) {
    let angle = plane.normal()[1].atan2(plane.normal()[0]);
    let step = f64::EPSILON.cbrt() * (1.0 + angle.abs());
    let derivative = (eckart_free_energy(angle + step, thermal)
        - eckart_free_energy(angle - step, thermal))
        / (2.0 * step);
    (0.0, derivative)
}

fn config(thermal: f64) -> AdaptiveOhTstConfig {
    AdaptiveOhTstConfig::default()
        .with_thermal(thermal)
        .with_max_iterations(32)
        .with_max_planes(257)
}

fn twisted_estimate(temperature_ratio: f64) -> rgsaddle::AdaptiveOhTstEstimate {
    let thermal = temperature_ratio * ECKART_V0;
    let initial = [0.5, 20.0, 40.0, 60.0, 88.0].map(plane_at_angle).to_vec();
    let density = Density {
        plane: initial[0].clone(),
        thermal,
        error: ADAPTIVE_OHTST_TARGET_LOG_ERROR,
    };
    let mut sampler = ExactPlaneSampler {
        thermal,
        gradient: |plane: &Hyperplane| eckart_gradient(plane, thermal),
        requested: Vec::new(),
    };
    let estimate =
        adaptive_ohtst_string_rate(&mut sampler, &initial, &density, &config(thermal)).unwrap();
    assert_eq!(sampler.requested[0], ADAPTIVE_OHTST_PILOT_SAMPLES);
    estimate
}

#[test]
fn twisted_path_switches_at_the_entropy_selected_branch() {
    let cold = twisted_estimate(0.22);
    let hot = twisted_estimate(0.23);
    let cold_angle = plane_angle(&cold.dividing_plane);
    let hot_angle = plane_angle(&hot.dividing_plane);
    assert!(cold_angle < 2.0, "cold normal is {cold_angle} degrees");
    assert!(hot_angle > 70.0, "hot normal is {hot_angle} degrees");
    assert!(hot_angle - cold_angle > 68.0);
    assert!(cold.rate.rate.is_finite() && cold.rate.error.is_finite());
    assert!(hot.rate.rate.is_finite() && hot.rate.error.is_finite());
}

#[test]
fn twisted_rate_matches_line_quadrature_within_reported_error() {
    let thermal = 0.3 * ECKART_V0;
    let estimate = twisted_estimate(0.3);
    let start = 0.5f64.to_radians();
    let optimum = 75.60242543f64.to_radians();
    let barrier = eckart_free_energy(optimum, thermal) - eckart_free_energy(start, thermal);
    let exact = (thermal / TAU).sqrt() * (-barrier / thermal).exp();
    assert!(
        (estimate.rate.rate - exact).abs() <= estimate.rate.error,
        "{} +/- {} against {exact}",
        estimate.rate.rate,
        estimate.rate.error
    );
}

fn straight_gradient(plane: &Hyperplane, thermal: f64) -> (f64, f64) {
    let normal = plane.normal();
    let offset = normal.dot(&plane.point());
    let angle = normal[1].atan2(normal[0]);
    (-4.0 * offset * (1.0 - offset * offset), -thermal * angle)
}

fn straight_planes() -> Vec<Hyperplane> {
    (0..=4)
        .map(|index| {
            let x = -1.0 + index as f64 / 4.0;
            Hyperplane::new(array![x, 0.0], array![1.0, 0.0]).unwrap()
        })
        .collect()
}

#[test]
fn adaptive_planes_match_the_plain_finite_temperature_string() {
    let thermal = 0.2;
    let initial = straight_planes();
    let density = Density {
        plane: initial[0].clone(),
        thermal,
        error: 0.0,
    };
    let mut sampler = ExactPlaneSampler {
        thermal,
        gradient: |plane: &Hyperplane| straight_gradient(plane, thermal),
        requested: Vec::new(),
    };
    let estimate =
        adaptive_ohtst_string_rate(&mut sampler, &initial, &density, &config(thermal)).unwrap();
    assert!((estimate.profile.barrier - 1.0).abs() <= ADAPTIVE_OHTST_TARGET_LOG_ERROR * thermal);
    assert!(
        estimate
            .profile
            .planes
            .iter()
            .all(|plane| plane.normal()[1].abs() <= 64.0 * f64::EPSILON)
    );
}

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1_u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let radius = (-2.0 * self.uniform().ln()).sqrt();
        radius * (TAU * self.uniform()).cos()
    }
}

struct NoisySampler {
    sigma: f64,
    thermal: f64,
}

impl HyperplaneSampler for NoisySampler {
    fn sample_hyperplane(
        &mut self,
        plane: &Hyperplane,
        budget: &HyperplaneSampleBudget,
    ) -> Result<HyperplaneEnsemble, SaddleError> {
        let mut rng = Rng(budget
            .seed
            .wrapping_mul(0xd134_2543_de82_ef95)
            .wrapping_add(12_345));
        let normal = plane.normal();
        let tangent = array![-normal[1], normal[0]];
        let (translation, rotation) = straight_gradient(plane, self.thermal);
        let observations = (0..budget.samples)
            .map(|_| {
                let sign = if rng.uniform() < 0.5 { -1.0 } else { 1.0 };
                let arm = &tangent * sign;
                HyperplaneObservation::default()
                    .with_constraint_force(
                        -translation + sign * rotation + self.sigma * rng.normal(),
                    )
                    .with_position(&plane.point() + &arm)
                    .with_projected_force(&arm * -self.thermal)
            })
            .collect();
        let mut ensemble = HyperplaneEnsemble::default();
        ensemble.observations = observations;
        ensemble.evaluations = budget.samples;
        ensemble.acceptance = 1.0;
        Ok(ensemble)
    }
}

struct ProductionNoisySampler {
    noisy: NoisySampler,
    pilot_planes: usize,
}

impl HyperplaneSampler for ProductionNoisySampler {
    fn sample_hyperplane(
        &mut self,
        plane: &Hyperplane,
        budget: &HyperplaneSampleBudget,
    ) -> Result<HyperplaneEnsemble, SaddleError> {
        if self.pilot_planes == 0 {
            self.noisy.sample_hyperplane(plane, budget)
        } else {
            self.pilot_planes -= 1;
            NoisySampler {
                sigma: 0.0,
                thermal: self.noisy.thermal,
            }
            .sample_hyperplane(plane, budget)
        }
    }
}

struct CorrelatedSampler {
    sigma: f64,
    thermal: f64,
    phi: f64,
    drift: f64,
}

impl HyperplaneSampler for CorrelatedSampler {
    fn sample_hyperplane(
        &mut self,
        plane: &Hyperplane,
        budget: &HyperplaneSampleBudget,
    ) -> Result<HyperplaneEnsemble, SaddleError> {
        let mut rng = Rng(budget
            .seed
            .wrapping_mul(0xd134_2543_de82_ef95)
            .wrapping_add(12_345));
        let normal = plane.normal();
        let tangent = array![-normal[1], normal[0]];
        let (translation, rotation) = straight_gradient(plane, self.thermal);
        let phi = self.phi;
        let stride = budget.stride.max(1);
        let innovation = self.sigma * (1.0 - phi * phi).sqrt();
        let mut noise = self.sigma * rng.normal();
        let mut kept = Vec::with_capacity(budget.samples);
        let mut step = 0usize;
        while kept.len() < budget.samples {
            noise = phi * noise + innovation * rng.normal();
            step += 1;
            if step.is_multiple_of(stride) {
                kept.push(noise);
            }
        }
        let observations = kept
            .into_iter()
            .enumerate()
            .map(|(index, noise)| {
                let sign = if index.is_multiple_of(2) { -1.0 } else { 1.0 };
                let arm = &tangent * sign;
                let drift = self.drift * index as f64 / budget.samples as f64;
                HyperplaneObservation::default()
                    .with_constraint_force(-translation + sign * rotation + noise + drift)
                    .with_position(&plane.point() + &arm)
                    .with_projected_force(&arm * -self.thermal)
            })
            .collect();
        let mut ensemble = HyperplaneEnsemble::default();
        ensemble.observations = observations;
        ensemble.evaluations = budget.samples;
        ensemble.acceptance = 1.0;
        Ok(ensemble)
    }
}

fn tilted_straight_planes(angle: f64) -> Vec<Hyperplane> {
    (0..=4)
        .map(|index| {
            let x = -1.0 + index as f64 / 4.0;
            let local_angle: f64 = if index == 0 || index == 4 { 0.0 } else { angle };
            let normal = array![local_angle.cos(), local_angle.sin()];
            Hyperplane::new(&normal * x, normal).unwrap()
        })
        .collect()
}

#[test]
fn pilot_rejects_a_twenty_three_sigma_drift() {
    let thermal = 0.2;
    let initial = straight_planes();
    let density = Density {
        plane: initial[0].clone(),
        thermal,
        error: 0.0,
    };
    let mut sampler = CorrelatedSampler {
        sigma: 0.25,
        thermal,
        phi: 0.0,
        drift: 0.5,
    };
    let adaptive = config(thermal).with_sampler(
        LangevinConfig::default()
            .with_thermal(thermal)
            .with_seed(99 * 7_919),
    );
    match adaptive_ohtst_string_rate(&mut sampler, &initial, &density, &adaptive) {
        Err(SaddleError::NotEquilibrated { .. }) => {}
        Err(other) => panic!("unexpected drift failure: {other}"),
        Ok(estimate) => assert!(
            (estimate.profile.barrier - 1.0).abs() <= estimate.profile.barrier_error,
            "confident wrong barrier {} +/- {}",
            estimate.profile.barrier,
            estimate.profile.barrier_error
        ),
    }
}

fn exact_tilt_is_covered(angle: f64) {
    let thermal = 0.2;
    let initial = tilted_straight_planes(angle);
    let density = Density {
        plane: initial[0].clone(),
        thermal,
        error: 0.0,
    };
    let mut sampler = ExactPlaneSampler {
        thermal,
        gradient: |plane: &Hyperplane| straight_gradient(plane, thermal),
        requested: Vec::new(),
    };
    let estimate =
        adaptive_ohtst_string_rate(&mut sampler, &initial, &density, &config(thermal)).unwrap();
    let profile = &estimate.profile;
    assert_eq!(estimate.status, AdaptiveOhTstStatus::TargetMet);
    assert!(profile.barrier_error <= ADAPTIVE_OHTST_TARGET_LOG_ERROR * thermal);
    assert!(
        (profile.barrier - 1.0).abs() <= profile.barrier_error,
        "{} +/- {} (geom {} quad {})",
        profile.barrier,
        profile.barrier_error,
        profile.geometry_error,
        profile.quadrature_error
    );
}

#[test]
fn exact_tilts_of_a_tenth_and_a_half_radian_are_covered() {
    exact_tilt_is_covered(0.1);
    exact_tilt_is_covered(0.5);
}

#[test]
fn an_exact_tilted_string_moves_and_converges() {
    let thermal = 0.2;
    let initial = tilted_straight_planes(0.3);
    let density = Density {
        plane: initial[0].clone(),
        thermal,
        error: 0.0,
    };
    let mut sampler = ExactPlaneSampler {
        thermal,
        gradient: |plane: &Hyperplane| straight_gradient(plane, thermal),
        requested: Vec::new(),
    };
    let estimate =
        adaptive_ohtst_string_rate(&mut sampler, &initial, &density, &config(thermal)).unwrap();
    let final_angle = plane_angle(&estimate.profile.planes[estimate.profile.planes.len() / 2]);
    assert!(estimate.iterations > 0);
    let angular_budget = (2.0 * ADAPTIVE_OHTST_TARGET_LOG_ERROR).sqrt();
    assert!(
        final_angle.abs() <= angular_budget.to_degrees(),
        "final angle {final_angle}, budget {angular_budget}"
    );
    assert!(
        (estimate.profile.barrier - 1.0).abs() <= estimate.profile.barrier_error,
        "{} +/- {}",
        estimate.profile.barrier,
        estimate.profile.barrier_error
    );
}

#[test]
fn a_noisy_tilted_string_includes_position_uncertainty() {
    let thermal = 0.2;
    let initial = tilted_straight_planes(0.3);
    let density = Density {
        plane: initial[0].clone(),
        thermal,
        error: 0.0,
    };
    let mut sampler = CorrelatedSampler {
        sigma: 0.25,
        thermal,
        phi: 0.0,
        drift: 0.0,
    };
    let adaptive = config(thermal).with_sampler(
        LangevinConfig::default()
            .with_thermal(thermal)
            .with_seed(17 * 7_919),
    );
    let estimate = adaptive_ohtst_string_rate(&mut sampler, &initial, &density, &adaptive).unwrap();
    assert!(estimate.profile.samples_per_plane[0] > ADAPTIVE_OHTST_PILOT_SAMPLES);
    assert!(estimate.iterations > 0);
    let final_angle = plane_angle(&estimate.profile.planes[estimate.profile.planes.len() / 2]);
    assert!(
        final_angle.abs() < 0.3_f64.to_degrees(),
        "production relaxation left the tilt at {final_angle}"
    );
    assert!(
        (estimate.profile.barrier - 1.0).abs() <= estimate.profile.barrier_error,
        "{} +/- {}",
        estimate.profile.barrier,
        estimate.profile.barrier_error
    );
}

#[test]
fn profile_error_carries_sampling_geometry_and_quadrature() {
    let thermal = 0.2;
    let initial = tilted_straight_planes(0.3);
    let density = Density {
        plane: initial[0].clone(),
        thermal,
        error: 0.0,
    };
    let mut sampler = ExactPlaneSampler {
        thermal,
        gradient: |plane: &Hyperplane| straight_gradient(plane, thermal),
        requested: Vec::new(),
    };
    let estimate =
        adaptive_ohtst_string_rate(&mut sampler, &initial, &density, &config(thermal)).unwrap();
    let profile = estimate.profile;
    assert!(profile.sampled_barrier_error < profile.geometry_error);
    assert!(profile.geometry_error > 0.0);
    assert!(profile.quadrature_error > 0.0);
    assert!(profile.barrier_error > profile.sampled_barrier_error);
    assert!(profile.barrier_error > profile.geometry_error);
    assert!(profile.barrier_error > profile.quadrature_error);
}

#[test]
fn hitting_the_sample_cap_is_machine_readable() {
    let thermal = 0.2;
    let initial = straight_planes();
    let density = Density {
        plane: initial[0].clone(),
        thermal,
        error: 0.0,
    };
    for seed in 1..=64 {
        let mut sampler = NoisySampler {
            sigma: 0.5,
            thermal,
        };
        let adaptive = config(thermal).with_sampler(
            LangevinConfig::default()
                .with_thermal(thermal)
                .with_seed(seed * 7_919),
        );
        if let Ok(estimate) =
            adaptive_ohtst_string_rate(&mut sampler, &initial, &density, &adaptive)
            && estimate.profile.samples_per_plane[0] == ADAPTIVE_OHTST_MAX_SAMPLES
            && estimate.profile.sampled_barrier_error / thermal > ADAPTIVE_OHTST_TARGET_LOG_ERROR
        {
            assert_eq!(estimate.status, AdaptiveOhTstStatus::SampleCap);
            return;
        }
    }
    panic!("the deterministic high-noise campaign did not reach the sample cap");
}

#[test]
fn ar1_profile_error_is_autocorrelation_aware() {
    let thermal = 0.2;
    let initial = straight_planes();
    let density = Density {
        plane: initial[0].clone(),
        thermal,
        error: 0.0,
    };
    let mut estimates = Vec::new();
    let mut failures = 0;
    let outcomes: Vec<_> = (1_u64..=48)
        .into_par_iter()
        .map(|seed| {
            let mut sampler = CorrelatedSampler {
                sigma: 0.25,
                thermal,
                phi: 0.9,
                drift: 0.0,
            };
            let adaptive = config(thermal).with_sampler(
                LangevinConfig::default()
                    .with_thermal(thermal)
                    .with_seed(seed * 7_919),
            );
            adaptive_ohtst_string_rate(&mut sampler, &initial, &density, &adaptive)
        })
        .collect();
    for outcome in outcomes {
        match outcome {
            Ok(estimate) => {
                estimates.push((estimate.profile.barrier, estimate.profile.barrier_error))
            }
            Err(SaddleError::NotEquilibrated { .. }) => failures += 1,
            Err(other) => panic!("unexpected AR(1) failure: {other}"),
        }
    }
    assert!(estimates.len() >= 40, "{failures} stationarity failures");
    let mean = estimates.iter().map(|item| item.0).sum::<f64>() / estimates.len() as f64;
    let empirical = (estimates
        .iter()
        .map(|item| (item.0 - mean).powi(2))
        .sum::<f64>()
        / (estimates.len() - 1) as f64)
        .sqrt();
    let reported = estimates.iter().map(|item| item.1).sum::<f64>() / estimates.len() as f64;
    // Seven pilot blocks leave ν = 6. For a normal sample,
    // E[s]/σ = sqrt(2/ν) Γ((ν+1)/2) / Γ(ν/2) = sqrt(1/3) · (15/8) √π / 2.
    // This comparison uses the combined error. The sampled standard error
    // is checked on its own, at a fixed record count, by
    // `fixed_count_sampled_error_matches_the_barrier_deviation`.
    let ratio = (1.0_f64 / 3.0).sqrt() * (15.0 / 16.0) * std::f64::consts::PI.sqrt();
    assert!(
        reported >= ratio * empirical,
        "reported {reported}, empirical {empirical}, ratio {ratio}"
    );
}

fn noisy_run(seed: u64, sigma: f64) -> Result<(f64, f64, usize), SaddleError> {
    let thermal = 0.2;
    let initial = straight_planes();
    let density = Density {
        plane: initial[0].clone(),
        thermal,
        error: 0.0,
    };
    let sampler_config = LangevinConfig::default()
        .with_thermal(thermal)
        .with_seed(seed.wrapping_mul(7_919));
    let adaptive = config(thermal).with_sampler(sampler_config);
    let mut sampler = NoisySampler { sigma, thermal };
    let estimate = adaptive_ohtst_string_rate(&mut sampler, &initial, &density, &adaptive)?;
    Ok((
        estimate.profile.barrier,
        estimate.profile.barrier_error,
        estimate.profile.samples_per_plane[0],
    ))
}

fn assert_noisy_strings_survive_and_cover(sigma: f64) {
    let mut estimates = Vec::new();
    let mut failures = std::collections::BTreeMap::new();
    let outcomes: Vec<_> = (1_u64..=256)
        .into_par_iter()
        .map(|seed| noisy_run(1_000 * seed + 17, sigma))
        .collect();
    for outcome in outcomes {
        match outcome {
            Ok(estimate) => estimates.push(estimate),
            Err(error) => {
                *failures.entry(error.to_string()).or_insert(0_usize) += 1;
            }
        }
    }
    assert!(
        estimates.len() >= 248,
        "sigma {sigma}: only {} of 256 estimates returned: {failures:?}",
        estimates.len(),
    );
    let mean = estimates.iter().map(|estimate| estimate.0).sum::<f64>() / estimates.len() as f64;
    let empirical = (estimates
        .iter()
        .map(|estimate| (estimate.0 - mean).powi(2))
        .sum::<f64>()
        / (estimates.len() - 1) as f64)
        .sqrt();
    let reported =
        estimates.iter().map(|estimate| estimate.1).sum::<f64>() / estimates.len() as f64;
    let mean_count =
        estimates.iter().map(|estimate| estimate.2).sum::<usize>() as f64 / estimates.len() as f64;
    let coverage = estimates
        .iter()
        .filter(|estimate| (estimate.0 - 1.0).abs() <= 2.0 * estimate.1)
        .count() as f64
        / estimates.len() as f64;
    assert!(
        reported >= 0.9 * empirical,
        "sigma {sigma}: mean {mean}, count {mean_count}, reported {reported}, empirical {empirical}"
    );
    assert!(coverage >= 0.9, "sigma {sigma}: two-SE coverage {coverage}");
}

#[test]
fn moderately_noisy_strings_survive_and_profile_errors_cover() {
    assert_noisy_strings_survive_and_cover(0.25);
}

#[test]
fn strongly_noisy_strings_survive_and_profile_errors_cover() {
    assert_noisy_strings_survive_and_cover(0.5);
}

struct SymmetricDoubleWell;

impl PointSurface for SymmetricDoubleWell {
    fn eval(&self, q: ArrayView1<'_, f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let x2 = q[0] * q[0];
        let y2 = q[1] * q[1];
        Ok((
            (x2 - 1.0).powi(2) + (y2 - 1.0).powi(2),
            array![4.0 * q[0] * (x2 - 1.0), 4.0 * q[1] * (y2 - 1.0)],
        ))
    }
}

fn reflected_rate(region: Arc<dyn Region>) -> rgsaddle::AdaptiveOhTstEstimate {
    let thermal = 0.2;
    let planes: Vec<Hyperplane> = [-1.0, -0.5, 0.0]
        .map(|x| Hyperplane::new(array![x, 1.0], array![1.0, 0.0]).unwrap())
        .to_vec();
    let density = Density {
        plane: planes[0].clone(),
        thermal,
        error: 0.0,
    };
    let sampling = LangevinConfig::default()
        .with_thermal(thermal)
        .with_time_step(0.08)
        .with_friction(1.0)
        .with_burn_in(200)
        .with_stride(4)
        .with_blocks(3)
        .with_seed(31)
        .with_region(region);
    let mut sampler = AnalyticSurfaceSampler::new(&SymmetricDoubleWell, sampling.clone());
    adaptive_ohtst_string_rate(
        &mut sampler,
        &planes,
        &density,
        &config(thermal)
            .with_sampler(sampling)
            .with_max_iterations(8),
    )
    .unwrap()
}

#[test]
fn adaptive_rate_applies_the_sampled_region_degeneracy_once() {
    let plane = Hyperplane::new(array![0.0, 1.0], array![1.0, 0.0]).unwrap();
    let operations = SymmetryOperations::default().with_operations(vec![
        AffineSymmetry::new(Array2::eye(2), Array1::zeros(2)),
        AffineSymmetry::new(array![[1.0, 0.0], [0.0, -1.0]], Array1::zeros(2)),
    ]);
    let symmetry =
        Arc::new(SymmetryUniqueRegion::from_source(&plane, array![0.0, 1.0], &operations).unwrap());
    assert_eq!(symmetry.degeneracy(), 2);
    let same_wall = Arc::new(PlanarBoundary::new(array![0.0, 0.0], array![0.0, -1.0]).unwrap());
    let unique = reflected_rate(same_wall);
    let reconstructed = reflected_rate(symmetry.clone());
    assert_eq!(unique.profile.barrier, reconstructed.profile.barrier);
    assert_eq!(reconstructed.rate.rate / unique.rate.rate, 2.0);
    assert_eq!(
        unique.rate.log_rate_error,
        reconstructed.rate.log_rate_error
    );
    assert!(unique.rate.error > 0.0);
    assert_eq!(reconstructed.rate.error / unique.rate.error, 2.0);

    let explicit_images = unique
        .rate
        .multiplied(&RateMultiplicity::default().with_in_plane_images(2))
        .unwrap();
    let region_images = unique
        .rate
        .multiplied(&RateMultiplicity::default().with_region(symmetry.as_ref()))
        .unwrap();
    let paths = unique
        .rate
        .multiplied(&RateMultiplicity::default().with_paths(3))
        .unwrap();
    let paths_and_region = unique
        .rate
        .multiplied(
            &RateMultiplicity::default()
                .with_paths(3)
                .with_region(symmetry.as_ref()),
        )
        .unwrap();
    assert_eq!(explicit_images.rate / unique.rate.rate, 2.0);
    assert_eq!(region_images.rate, reconstructed.rate.rate);
    assert_eq!(paths.rate / unique.rate.rate, 3.0);
    assert_eq!(paths_and_region.rate / unique.rate.rate, 6.0);
    let operations = 6.0;
    let gamma = operations * f64::EPSILON / (1.0 - operations * f64::EPSILON);
    assert!((paths_and_region.error / unique.rate.error - 6.0).abs() <= gamma);
    assert!((region_images.log_rate - unique.rate.log_rate - 2.0_f64.ln()).abs() <= gamma);
    assert!((paths_and_region.log_rate - unique.rate.log_rate - 6.0_f64.ln()).abs() <= gamma);
}

#[test]
fn channel_mirror_reflection_matches_johannesson_jonsson() {
    // Mass-weighted specialization of JCP 115, 9644 (2001), Eqs. 16-17:
    // reflect the velocity in the channel-mapping mirror while retaining
    // the component allowed by the sampled hyperplane.
    let plane = Hyperplane::new(array![0.0, 0.0, 0.0], array![1.0, 1.0, 0.0]).unwrap();
    let raw_mirror_normal: Array1<f64> = array![2.0, -1.0, 2.0];
    let projected_mirror = plane.project(raw_mirror_normal.view());
    let mirror_normal = &projected_mirror / projected_mirror.dot(&projected_mirror).sqrt();
    let operations = 16 * plane.dim() + 8;
    let product = operations as f64 * f64::EPSILON;
    let gamma = product / (1.0 - product);
    assert!(raw_mirror_normal.dot(&plane.normal()).abs() > 0.0);
    assert!(mirror_normal.dot(&plane.normal()).abs() <= gamma);

    let raw_velocity = array![2.0, -1.0, 1.0];
    let original_velocity = plane.project(raw_velocity.view());
    let mut velocity = raw_velocity;
    let original_speed2 = original_velocity.dot(&original_velocity);
    let mirror = Array2::from_shape_fn((3, 3), |(row, column)| {
        let identity = if row == column { 1.0 } else { 0.0 };
        identity - 2.0 * mirror_normal[row] * mirror_normal[column]
    });
    let expected = mirror.dot(&original_velocity);
    reflect_velocity(&mut velocity, raw_mirror_normal.view(), &plane).unwrap();

    let scale = original_speed2.sqrt().max(1.0);
    let roundoff = gamma * scale;
    assert!(
        (&velocity - &expected)
            .iter()
            .all(|difference| difference.abs() <= roundoff)
    );
    assert!(velocity.dot(&plane.normal()).abs() <= roundoff);
    assert!((velocity.dot(&velocity) - original_speed2).abs() <= roundoff * scale);
    let mirror_tangent = array![2.0, -2.0, -3.0];
    assert!(
        (mirror_tangent.dot(&velocity) - mirror_tangent.dot(&original_velocity)).abs() <= roundoff
    );
    assert!(velocity.dot(&projected_mirror) <= 0.0);
    let unique = PlanarBoundary::new(Array1::zeros(3), projected_mirror.to_owned()).unwrap();
    let returned = &velocity * f64::EPSILON.sqrt();
    assert!(unique.contains(returned.view()).unwrap());
}

#[test]
fn sample_growth_uses_the_prescribed_rule_and_cap() {
    assert_eq!(adaptive_ohtst_sample_count(14, 0.06).unwrap(), 62);
    assert_eq!(adaptive_ohtst_sample_count(14, 3.0).unwrap(), 2048);
    assert!(adaptive_ohtst_sample_count(14, -0.01).is_err());
}

#[test]
fn density_uncertainty_does_not_drive_plane_resampling() {
    let thermal = 0.2;
    let initial = straight_planes();
    let density = Density {
        plane: initial[0].clone(),
        thermal,
        error: 0.5,
    };
    let mut sampler = ExactPlaneSampler {
        thermal,
        gradient: |plane: &Hyperplane| straight_gradient(plane, thermal),
        requested: Vec::new(),
    };
    let estimate =
        adaptive_ohtst_string_rate(&mut sampler, &initial, &density, &config(thermal)).unwrap();
    assert!(sampler.requested.iter().all(|count| *count == 14));
    assert!(estimate.rate.log_rate_error >= 0.5);
}

#[test]
fn rotating_a_bounded_plane_requires_the_wall_contact_derivative() {
    let thermal = 0.2;
    let initial = [0.0, 20.0, 40.0].map(plane_at_angle).to_vec();
    let density = Density {
        plane: initial[0].clone(),
        thermal,
        error: 0.0,
    };
    let region = Arc::new(PlanarBoundary::new(array![0.0, 0.0], array![0.0, -1.0]).unwrap());
    let bounded = config(thermal).with_sampler(
        LangevinConfig::default()
            .with_thermal(thermal)
            .with_region(region),
    );
    let mut sampler = ExactPlaneSampler {
        thermal,
        gradient: |_plane: &Hyperplane| (0.0, 0.0),
        requested: Vec::new(),
    };
    let error = adaptive_ohtst_string_rate(&mut sampler, &initial, &density, &bounded)
        .expect_err("rotating bounded planes need a wall-contact derivative");
    assert!(error.to_string().contains("wall-contact"));
}

#[derive(Debug)]
struct PredicateRegion;

impl Region for PredicateRegion {
    fn violation(
        &self,
        point: ArrayView1<'_, f64>,
    ) -> Result<Option<BoundaryViolation>, SaddleError> {
        if point[1] >= 0.0 {
            Ok(None)
        } else {
            Ok(Some(BoundaryViolation::predicate(array![0.0, -1.0])?))
        }
    }
}

#[test]
fn predicate_only_regions_require_wall_contact_derivatives() {
    let thermal = 0.2;
    let initial = straight_planes();
    let density = Density {
        plane: initial[0].clone(),
        thermal,
        error: 0.0,
    };
    let bounded = config(thermal).with_sampler(
        LangevinConfig::default()
            .with_thermal(thermal)
            .with_region(Arc::new(PredicateRegion)),
    );
    let mut sampler = ExactPlaneSampler {
        thermal,
        gradient: |plane: &Hyperplane| straight_gradient(plane, thermal),
        requested: Vec::new(),
    };
    let error = adaptive_ohtst_string_rate(&mut sampler, &initial, &density, &bounded)
        .expect_err("predicate-only boundaries lack wall-contact derivatives");
    assert!(error.to_string().contains("wall-contact"));
}

#[test]
fn oblique_faces_require_wall_contact_derivatives() {
    let thermal = 0.2;
    let initial = straight_planes();
    let density = Density {
        plane: initial[0].clone(),
        thermal,
        error: 0.0,
    };
    let boundary = Arc::new(PlanarBoundary::new(array![0.0, 0.0], array![1.0, 1.0]).unwrap());
    let bounded = config(thermal).with_sampler(
        LangevinConfig::default()
            .with_thermal(thermal)
            .with_region(boundary),
    );
    let mut sampler = ExactPlaneSampler {
        thermal,
        gradient: |plane: &Hyperplane| straight_gradient(plane, thermal),
        requested: Vec::new(),
    };
    let error = adaptive_ohtst_string_rate(&mut sampler, &initial, &density, &bounded)
        .expect_err("an oblique face changes under plane translation");
    assert!(error.to_string().contains("wall-contact"));
}

#[test]
fn an_in_step_rotation_rechecks_wall_contact_derivatives() {
    let thermal = 0.2;
    let initial = straight_planes();
    let density = Density {
        plane: initial[0].clone(),
        thermal,
        error: 0.0,
    };
    let boundary = Arc::new(PlanarBoundary::new(array![0.0, 0.0], array![0.0, -1.0]).unwrap());
    let bounded = config(thermal).with_sampler(
        LangevinConfig::default()
            .with_thermal(thermal)
            .with_region(boundary),
    );
    let mut sampler = ExactPlaneSampler {
        thermal,
        gradient: |_plane: &Hyperplane| (0.0, 0.1),
        requested: Vec::new(),
    };
    let error = adaptive_ohtst_string_rate(&mut sampler, &initial, &density, &bounded)
        .expect_err("a sampled torque must not rotate a bounded plane");
    assert!(error.to_string().contains("wall-contact"));
}

struct ScriptedPilot {
    base_seed: u64,
    thermal: f64,
    series: Vec<f64>,
}

impl HyperplaneSampler for ScriptedPilot {
    fn sample_hyperplane(
        &mut self,
        plane: &Hyperplane,
        budget: &HyperplaneSampleBudget,
    ) -> Result<HyperplaneEnsemble, SaddleError> {
        // The first pass pilots the five initial planes at salts 0..4.
        // Refinement salts start at the plane count, and production salts
        // start at the sample count, so both stay on the exact gradient.
        let pilot = budget.samples == ADAPTIVE_OHTST_PILOT_SAMPLES
            && budget.seed.wrapping_sub(self.base_seed) < 5;
        let (translation, rotation) = straight_gradient(plane, self.thermal);
        let normal = plane.normal();
        let tangent = array![-normal[1], normal[0]];
        let observations = (0..budget.samples)
            .map(|index| {
                let sign = if index.is_multiple_of(2) { -1.0 } else { 1.0 };
                let arm = &tangent * sign;
                let force = if pilot {
                    self.series[index]
                } else {
                    -translation + sign * rotation
                };
                HyperplaneObservation::default()
                    .with_constraint_force(force)
                    .with_position(&plane.point() + &arm)
                    .with_projected_force(&arm * -self.thermal)
            })
            .collect();
        let mut ensemble = HyperplaneEnsemble::default();
        ensemble.observations = observations;
        ensemble.evaluations = budget.samples;
        ensemble.acceptance = 1.0;
        Ok(ensemble)
    }
}

fn scripted_run(series: Vec<f64>) -> Result<rgsaddle::AdaptiveOhTstEstimate, SaddleError> {
    let thermal = 0.2;
    let initial = straight_planes();
    let density = Density {
        plane: initial[0].clone(),
        thermal,
        error: 0.0,
    };
    let base_seed = 50_000;
    let mut sampler = ScriptedPilot {
        base_seed,
        thermal,
        series,
    };
    let adaptive = config(thermal).with_sampler(
        LangevinConfig::default()
            .with_thermal(thermal)
            .with_seed(base_seed),
    );
    adaptive_ohtst_string_rate(&mut sampler, &initial, &density, &adaptive)
}

#[test]
fn pilot_split_half_drift_is_rejected() {
    let series = vec![
        4.0, 5.0, 4.0, 5.0, 4.0, 5.0, 0.0, 0.0, 11.0, 12.0, 11.0, 12.0, 11.0, 12.0,
    ];
    match scripted_run(series) {
        Err(SaddleError::NotEquilibrated { sigma_multiple, .. }) => {
            let family = (5.0 / ADAPTIVE_OHTST_TARGET_LOG_ERROR).sqrt();
            assert!((sigma_multiple - family).abs() <= 1.0e-9);
        }
        other => panic!("pilot drift was not rejected: {other:?}"),
    }
}

#[test]
fn pilot_familywise_threshold_accepts_a_moderate_drift() {
    let raw = [
        4.0, 5.0, 4.0, 5.0, 4.0, 5.0, 0.0, 0.0, 7.0, 8.0, 7.0, 8.0, 7.0, 8.0,
    ];
    let mean = raw.iter().sum::<f64>() / raw.len() as f64;
    let series = raw.iter().map(|value| 1.0e-8 * (value - mean)).collect();
    scripted_run(series).expect("a drift inside the familywise gate must be kept");
}

#[test]
fn pilot_effective_sample_gate_rejects_between_two_and_four() {
    let series = vec![
        0.0, 0.8, 1.4, 1.8, 2.0, 2.1, 2.0, 1.8, 1.4, 0.8, 0.2, -0.2, -0.4, -0.3,
    ];
    match scripted_run(series) {
        Err(SaddleError::InsufficientEffectiveSamples {
            n_eff, required, ..
        }) => {
            assert!(n_eff > 2.0 && n_eff < required);
            assert_eq!(required, 4.0);
        }
        other => panic!("effective-sample gate did not fire: {other:?}"),
    }
}

#[test]
fn pilot_autocorrelation_widens_the_half_error() {
    let raw = [
        -0.5941813929011218,
        -1.289345523322187,
        -1.5407113474250602,
        -1.980668954772438,
        -1.9453378211022716,
        -1.5690741670459318,
        0.0,
        0.0,
        -4.2483932502253445,
        -4.5911887417463015,
        -4.2483932502253445,
        -4.5911887417463015,
        -4.2483932502253445,
        -4.5911887417463015,
    ];
    let mean = raw.iter().sum::<f64>() / raw.len() as f64;
    let series = raw.iter().map(|value| 1.0e-8 * (value - mean)).collect();
    scripted_run(series).expect("the autocorrelation factor must keep this pilot");
}

#[test]
fn production_samples_use_equilibrate_not_the_pilot_policy() {
    let thermal = 0.2;
    let initial = straight_planes();
    let density = Density {
        plane: initial[0].clone(),
        thermal,
        error: 0.0,
    };
    let mut sampler = CorrelatedSampler {
        sigma: 0.25,
        thermal,
        phi: 0.0,
        drift: 0.25,
    };
    let adaptive = config(thermal).with_sampler(
        LangevinConfig::default()
            .with_thermal(thermal)
            .with_seed(19 * 7_919),
    );
    match adaptive_ohtst_string_rate(&mut sampler, &initial, &density, &adaptive) {
        Err(SaddleError::NotEquilibrated { sigma_multiple, .. }) => {
            assert!((sigma_multiple - 4.0).abs() <= 1.0e-9)
        }
        other => panic!("production drift was not rejected at 4 sigma: {other:?}"),
    }
}

#[test]
fn pilot_drift_respects_roundoff() {
    let step = 4.0 * f64::EPSILON;
    let mut series = vec![0.0; 14];
    for value in &mut series[8..] {
        *value = step;
    }
    scripted_run(series).expect("a one-ulp half-mean gap is roundoff, not drift");
}

#[test]
fn stationarity_roundoff_uses_the_projection_count() {
    let thermal = 0.2;
    let initial = straight_planes();
    let density = Density {
        plane: initial[0].clone(),
        thermal,
        error: 0.0,
    };
    let mut sampler = SpreadArmSampler { thermal };
    let estimate = adaptive_ohtst_string_rate(
        &mut sampler,
        &initial,
        &density,
        &config(thermal).with_max_iterations(4),
    )
    .expect("a gradient inside gamma_8m of one standard deviation is roundoff");
    assert_eq!(estimate.iterations, 0);
    for plane in &estimate.profile.planes {
        let normal = plane.normal();
        assert!((normal[0] - 1.0).abs() <= 1.0e-12 && normal[1].abs() <= 1.0e-12);
    }
    // Fourteen records and seven blocks divide evenly, so this series cannot
    // tell a floor width from a ceiling width.
    assert_fifteen_records_keep_seven_blocks();
}

struct SpreadArmSampler {
    thermal: f64,
}

impl HyperplaneSampler for SpreadArmSampler {
    fn sample_hyperplane(
        &mut self,
        plane: &Hyperplane,
        budget: &HyperplaneSampleBudget,
    ) -> Result<HyperplaneEnsemble, SaddleError> {
        // Fourteen records stay at the pilot count, so production uses seven
        // blocks of two. Every pair has mean force zero, and the seven block
        // gradients sit about 1.8e-15 above one standard deviation: inside
        // γ_24 of the projected gradient and outside γ_1.
        let forces = [
            1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0,
        ];
        let arms = [
            8.399_795_676_065_582e-1,
            -7.145_826_015_958_473e-2,
            -4.618_926_830_320_626e-1,
            -5.045_485_526_591_969_6e-2,
            9.680_664_521_810_538e-1,
            5.662_862_441_491_084_4e-2,
            -3.338_057_984_575_671e-1,
            7.763_202_930_857_585e-2,
            7.118_926_830_320_627e-1,
            -1.995_451_447_340_803e-1,
            -8.461_533_367_555_494e-1,
            -4.347_155_089_894_064e-1,
            -3.086_884_574_495_602e-3,
            -2.530_868_845_744_956e-1,
        ];
        let normal = plane.normal();
        let tangent = array![-normal[1], normal[0]];
        let observations = (0..budget.samples)
            .map(|index| {
                let slot = index % forces.len();
                let arm = &tangent * arms[slot];
                HyperplaneObservation::default()
                    .with_constraint_force(forces[slot])
                    .with_position(&plane.point() + &arm)
                    .with_projected_force(&arm * -self.thermal)
            })
            .collect();
        let mut ensemble = HyperplaneEnsemble::default();
        ensemble.observations = observations;
        ensemble.evaluations = budget.samples;
        ensemble.acceptance = 1.0;
        Ok(ensemble)
    }
}

fn symmetric_hop_planes() -> Vec<Hyperplane> {
    (0..=4)
        .map(|index| {
            let x = -1.0 + 0.5 * index as f64;
            Hyperplane::new(array![x, 0.0], array![1.0, 0.0]).unwrap()
        })
        .collect()
}

#[test]
fn an_interior_barrier_does_not_count_the_product_drop_as_quadrature() {
    let thermal = 0.2;
    let initial = symmetric_hop_planes();
    let density = Density {
        plane: initial[0].clone(),
        thermal,
        error: 0.0,
    };
    let mut sampler = ExactPlaneSampler {
        thermal,
        gradient: |plane: &Hyperplane| straight_gradient(plane, thermal),
        requested: Vec::new(),
    };
    let estimate =
        adaptive_ohtst_string_rate(&mut sampler, &initial, &density, &config(thermal)).unwrap();
    let profile = &estimate.profile;
    let end = profile.free_energy.len() - 1;
    assert!(profile.dividing < end);
    assert!(
        profile.planes[0]
            .normal()
            .dot(&profile.planes[end].normal())
            > 0.5
    );
    let drop = profile.barrier - profile.free_energy[end];
    assert!(drop > 0.5, "product drop {drop}");
    let quadrature_bound = thermal * ADAPTIVE_OHTST_TARGET_LOG_ERROR;
    assert!(
        profile.quadrature_error <= quadrature_bound,
        "quadrature {} exceeds the {quadrature_bound} target",
        profile.quadrature_error
    );
    assert_eq!(estimate.status, AdaptiveOhTstStatus::TargetMet);
    assert!(
        profile.barrier_error < drop,
        "total {} is the product drop {drop}",
        profile.barrier_error
    );
    assert!((profile.barrier - 1.0).abs() <= profile.barrier_error);
}

#[test]
fn target_met_uses_the_combined_error() {
    let thermal = 0.2;
    let initial = tilted_straight_planes(0.3);
    let density = Density {
        plane: initial[0].clone(),
        thermal,
        error: 0.5,
    };
    let mut sampler = ExactPlaneSampler {
        thermal,
        gradient: |plane: &Hyperplane| straight_gradient(plane, thermal),
        requested: Vec::new(),
    };
    let estimate =
        adaptive_ohtst_string_rate(&mut sampler, &initial, &density, &config(thermal)).unwrap();
    assert_eq!(estimate.status, AdaptiveOhTstStatus::Unresolved);
    assert!(estimate.profile.sampled_barrier_error / thermal <= ADAPTIVE_OHTST_TARGET_LOG_ERROR);
    assert!(estimate.profile.barrier_error / thermal > ADAPTIVE_OHTST_TARGET_LOG_ERROR);
    assert!(
        estimate
            .profile
            .samples_per_plane
            .iter()
            .all(|count| *count < ADAPTIVE_OHTST_MAX_SAMPLES)
    );
    assert!(
        sampler
            .requested
            .iter()
            .all(|count| *count == ADAPTIVE_OHTST_PILOT_SAMPLES),
        "density or a false cap grew the sample: {:?}",
        sampler.requested
    );
}

fn cosine_force(index: usize) -> f64 {
    (index as f64 * 0.62).cos()
}

fn block_force_error(values: &[f64], width: usize) -> (f64, f64, usize) {
    let blocks = values.len() / width;
    let kept = blocks * width;
    let mut means = Vec::with_capacity(blocks);
    for block in 0..blocks {
        let slice = &values[block * width..(block + 1) * width];
        means.push(slice.iter().sum::<f64>() / width as f64);
    }
    let mean = means.iter().sum::<f64>() / blocks as f64;
    let variance = means
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f64>()
        / (blocks - 1) as f64;
    (mean, (variance / blocks as f64).sqrt(), kept)
}

#[test]
fn shared_sampler_block_width_drops_the_tail() {
    let thermal = 0.2;
    let values: Vec<f64> = (0..14).map(cosine_force).collect();
    let plane = Hyperplane::new(array![0.0, 0.0], array![1.0, 0.0]).unwrap();
    let mut sampler = ScriptedForce {
        values: values.clone(),
    };
    let sampling = LangevinConfig::default()
        .with_thermal(thermal)
        .with_records(14)
        .with_blocks(4)
        .with_seed(11);
    let sample = sample_plane_with_sampler(&mut sampler, &plane, &sampling).unwrap();
    let (mean, error, kept) = block_force_error(&values, 4);
    let (_narrow_mean, narrow_error, _) = block_force_error(&values, 3);
    assert_eq!(sample.records, kept);
    assert!((sample.normal_force - mean).abs() <= 1e-12);
    assert!(
        (sample.normal_force_error - error).abs() <= 1e-12,
        "reported {} expected {error} (width 3 would be {narrow_error})",
        sample.normal_force_error
    );
    assert!((sample.normal_force_error - narrow_error).abs() > 1e-8);
}

struct ScriptedForce {
    values: Vec<f64>,
}

impl HyperplaneSampler for ScriptedForce {
    fn sample_hyperplane(
        &mut self,
        plane: &Hyperplane,
        budget: &HyperplaneSampleBudget,
    ) -> Result<HyperplaneEnsemble, SaddleError> {
        let normal = plane.normal();
        let tangent = array![-normal[1], normal[0]];
        let observations = (0..budget.samples)
            .map(|index| {
                let force = self.values[index % self.values.len()];
                HyperplaneObservation::default()
                    .with_constraint_force(force)
                    .with_position(&plane.point() + &tangent * 0.25)
                    .with_projected_force(&tangent * -0.2)
            })
            .collect();
        let mut ensemble = HyperplaneEnsemble::default();
        ensemble.observations = observations;
        ensemble.evaluations = budget.samples;
        ensemble.acceptance = 1.0;
        Ok(ensemble)
    }
}

fn scripted_plane(values: &[f64], blocks: usize) -> Result<rgsaddle::PlaneSample, SaddleError> {
    let plane = Hyperplane::new(array![0.0, 0.0], array![1.0, 0.0]).unwrap();
    let mut sampler = ScriptedForce {
        values: values.to_vec(),
    };
    let sampling = LangevinConfig::default()
        .with_thermal(0.2)
        .with_records(values.len())
        .with_blocks(blocks)
        .with_stride(1)
        .with_burn_in(0)
        .with_seed(1);
    sample_plane_with_sampler(&mut sampler, &plane, &sampling)
}

/// Fifteen records and seven blocks. The floor width is two, so seven blocks
/// remain and the last record is dropped. A ceiling width is three, and only
/// five blocks remain.
fn assert_fifteen_records_keep_seven_blocks() {
    let sample = scripted_plane(&[0.0; 15], 7).expect("fifteen equal records stay stationary");
    assert_eq!(sample.block_gradients.nrows(), 7);
    assert_eq!(sample.records, 14);
}

/// Fourteen monotone records and seven blocks. The integrated time leaves
/// `n_eff` below 4, so a literal floor of 4 and the caller floor both refuse
/// the series. `required` is the seven blocks.
fn assert_fourteen_record_floor_is_seven() {
    let values: Vec<f64> = (0..14).map(|index| index as f64).collect();
    match scripted_plane(&values, 7) {
        Err(SaddleError::InsufficientEffectiveSamples {
            n_eff, required, ..
        }) => {
            assert!(
                n_eff < required && required == 7.0,
                "n_eff {n_eff}, required {required}"
            );
        }
        other => panic!("fourteen monotone records did not report the caller floor: {other:?}"),
    }
}

struct PilotWidthSampler {
    requested: Vec<usize>,
    scale: f64,
}

impl HyperplaneSampler for PilotWidthSampler {
    fn sample_hyperplane(
        &mut self,
        plane: &Hyperplane,
        budget: &HyperplaneSampleBudget,
    ) -> Result<HyperplaneEnsemble, SaddleError> {
        self.requested.push(budget.samples);
        let normal = plane.normal();
        let tangent = array![-normal[1], normal[0]];
        let (translation, rotation) = straight_gradient(plane, 0.2);
        let observations = (0..budget.samples)
            .map(|index| {
                let force = if budget.samples == ADAPTIVE_OHTST_PILOT_SAMPLES {
                    cosine_force(index) * self.scale
                } else {
                    0.0
                };
                let sign = if index.is_multiple_of(2) { -1.0 } else { 1.0 };
                HyperplaneObservation::default()
                    .with_constraint_force(-translation + sign * rotation + force)
                    .with_position(&plane.point() + &tangent * sign)
                    .with_projected_force(&tangent * sign * -0.2)
            })
            .collect();
        let mut ensemble = HyperplaneEnsemble::default();
        ensemble.observations = observations;
        ensemble.evaluations = budget.samples;
        ensemble.acceptance = 1.0;
        Ok(ensemble)
    }
}

#[test]
fn pilot_block_width_follows_twice_the_autocorrelation() {
    let thermal = 0.2;
    let initial = straight_planes();
    let density = Density {
        plane: initial[0].clone(),
        thermal,
        error: 0.0,
    };
    let mut sampler = PilotWidthSampler {
        requested: Vec::new(),
        scale: 0.08,
    };
    let _ = adaptive_ohtst_string_rate(&mut sampler, &initial, &density, &config(thermal));
    // cos(0.62 i) on 14 records has τ ≈ 1.615, so ceil(2τ) = 4.
    // Width 4 requests 148 records; the naive width 2 requests 52.
    assert_eq!(sampler.requested[initial.len()], 148);
}

struct AttemptLog {
    calls: Vec<(usize, usize, usize)>,
}

impl HyperplaneSampler for AttemptLog {
    fn sample_hyperplane(
        &mut self,
        plane: &Hyperplane,
        budget: &HyperplaneSampleBudget,
    ) -> Result<HyperplaneEnsemble, SaddleError> {
        self.calls
            .push((budget.samples, budget.stride, budget.thermalize));
        let normal = plane.normal();
        let tangent = array![-normal[1], normal[0]];
        let observations = (0..budget.samples)
            .map(|index| {
                HyperplaneObservation::default()
                    .with_constraint_force(index as f64)
                    .with_position(&plane.point() + &tangent * 0.25)
                    .with_projected_force(&tangent * -0.2)
            })
            .collect();
        let mut ensemble = HyperplaneEnsemble::default();
        ensemble.observations = observations;
        ensemble.evaluations = budget.samples;
        ensemble.acceptance = 1.0;
        Ok(ensemble)
    }
}

#[test]
fn pilot_doubles_burn_in_and_stops_at_fourteen_strides() {
    let thermal = 0.2;
    let initial = straight_planes();
    let density = Density {
        plane: initial[0].clone(),
        thermal,
        error: 0.0,
    };
    let mut sampler = AttemptLog { calls: Vec::new() };
    let error = adaptive_ohtst_string_rate(&mut sampler, &initial, &density, &config(thermal));
    assert!(matches!(
        error,
        Err(SaddleError::InsufficientEffectiveSamples { .. })
    ));
    let pilot: Vec<_> = sampler
        .calls
        .iter()
        .copied()
        .filter(|call| call.0 == ADAPTIVE_OHTST_PILOT_SAMPLES)
        .collect();
    let strides: Vec<usize> = pilot.iter().map(|call| call.1).collect();
    let burn_in: Vec<usize> = pilot.iter().map(|call| call.2).collect();
    assert_eq!(strides, vec![5, 10, 20, 40]);
    assert_eq!(burn_in, vec![500, 1000, 2000, 4000]);
}

struct ProductionStride {
    calls: Vec<(usize, usize)>,
    seen: usize,
}

impl HyperplaneSampler for ProductionStride {
    fn sample_hyperplane(
        &mut self,
        plane: &Hyperplane,
        budget: &HyperplaneSampleBudget,
    ) -> Result<HyperplaneEnsemble, SaddleError> {
        self.seen += 1;
        self.calls.push((budget.samples, budget.stride));
        let normal = plane.normal();
        let tangent = array![-normal[1], normal[0]];
        let (translation, rotation) = straight_gradient(plane, 0.2);
        let pilot = self.seen <= 5;
        let doubled = budget.stride > 5;
        let observations = (0..budget.samples)
            .map(|index| {
                let sign = if index.is_multiple_of(2) { -1.0 } else { 1.0 };
                let force = if pilot || doubled {
                    -translation + sign * rotation
                } else {
                    index as f64
                };
                HyperplaneObservation::default()
                    .with_constraint_force(force)
                    .with_position(&plane.point() + &tangent * sign)
                    .with_projected_force(&tangent * sign * -0.2)
            })
            .collect();
        let mut ensemble = HyperplaneEnsemble::default();
        ensemble.observations = observations;
        ensemble.evaluations = budget.samples;
        ensemble.acceptance = 1.0;
        Ok(ensemble)
    }
}

#[test]
fn production_samples_do_not_inherit_the_pilot_stride() {
    let thermal = 0.2;
    let initial = straight_planes();
    let density = Density {
        plane: initial[0].clone(),
        thermal,
        error: 0.0,
    };
    let mut sampler = ProductionStride {
        calls: Vec::new(),
        seen: 0,
    };
    let error = adaptive_ohtst_string_rate(&mut sampler, &initial, &density, &config(thermal));
    assert!(
        matches!(error, Err(SaddleError::InsufficientEffectiveSamples { .. })),
        "production stride was doubled: {:?} ({error:?})",
        sampler.calls
    );
    assert!(sampler.calls.iter().all(|call| call.1 == 5));
}

#[test]
fn measured_errors_keep_the_integral_roundoff_floor() {
    let thermal = 0.2;
    let initial = straight_planes();
    let density = Density {
        plane: initial[0].clone(),
        thermal,
        error: 0.0,
    };
    let mut sampler = ExactPlaneSampler {
        thermal,
        gradient: |_plane: &Hyperplane| (0.0, 0.0),
        requested: Vec::new(),
    };
    let estimate =
        adaptive_ohtst_string_rate(&mut sampler, &initial, &density, &config(thermal)).unwrap();
    let planes = estimate.profile.planes.len();
    let dimension = initial[0].dim();
    let operations = (planes * (4 * dimension + 1)) as f64;
    let gamma = operations * f64::EPSILON / (1.0 - operations * f64::EPSILON);
    let floor = gamma * thermal;
    assert_eq!(estimate.profile.geometry_error, floor);
    assert_eq!(estimate.profile.quadrature_error, floor);
}

#[test]
fn doubled_grid_gap_is_part_of_the_quadrature_error() {
    let estimate = twisted_estimate(0.3);
    let planes = estimate.profile.planes.len();
    let dimension = estimate.profile.planes[0].dim();
    let operations = (planes * (4 * dimension + 1)) as f64;
    let gamma = operations * f64::EPSILON / (1.0 - operations * f64::EPSILON);
    let floor = gamma * estimate.profile.plane_samples[0].thermal;
    assert!(
        estimate.profile.quadrature_error > 100.0 * floor,
        "quadrature {} is only the floor {floor}",
        estimate.profile.quadrature_error
    );
}

#[test]
fn neighbor_smoothing_contributes_to_the_geometry_error() {
    let thermal = 0.2;
    let kink = 0.4_f64;
    let mut initial = [-0.5, 0.0, 0.5]
        .map(|x| Hyperplane::new(array![x, 0.0], array![1.0, 0.0]).unwrap())
        .to_vec();
    let normal = array![kink.cos(), kink.sin()];
    initial[1] = Hyperplane::new(array![0.0, 0.0], normal).unwrap();
    let density = Density {
        plane: initial[0].clone(),
        thermal,
        error: 0.0,
    };
    // Arms at 1 and 3 have a nonzero mean, so the block torque sees a change of
    // normal, and a nonzero spread, so the rotation metric exists. The centroid
    // correction cancels that mean, and the force is parallel to the string, so
    // the Newton probe does not move the kink. The force is one edge's share of
    // the target, small enough that refinement leaves the three planes in place.
    let edges = (initial.len() - 1) as f64;
    let force = thermal * ADAPTIVE_OHTST_TARGET_LOG_ERROR / edges;
    let mut sampler = OffsetArmSampler { force };
    let estimate =
        adaptive_ohtst_string_rate(&mut sampler, &initial, &density, &config(thermal)).unwrap();
    assert_eq!(estimate.iterations, 0);
    assert_eq!(estimate.profile.planes.len(), initial.len());
    let operations = (initial.len() * (4 * initial[0].dim() + 1)) as f64;
    let gamma = operations * f64::EPSILON / (1.0 - operations * f64::EPSILON);
    let floor = gamma * thermal;
    assert!(
        estimate.profile.geometry_error > 100.0 * floor,
        "geometry {} did not see the kink",
        estimate.profile.geometry_error
    );
}

struct OffsetArmSampler {
    force: f64,
}

impl HyperplaneSampler for OffsetArmSampler {
    fn sample_hyperplane(
        &mut self,
        plane: &Hyperplane,
        budget: &HyperplaneSampleBudget,
    ) -> Result<HyperplaneEnsemble, SaddleError> {
        let normal = plane.normal();
        let tangent = array![-normal[1], normal[0]];
        let observations = (0..budget.samples)
            .map(|index| {
                let arm = if index.is_multiple_of(2) { 1.0 } else { 3.0 };
                HyperplaneObservation::default()
                    .with_constraint_force(self.force)
                    .with_position(&plane.point() + &tangent * arm)
                    .with_projected_force(&tangent * -0.2)
            })
            .collect();
        let mut ensemble = HyperplaneEnsemble::default();
        ensemble.observations = observations;
        ensemble.evaluations = budget.samples;
        ensemble.acceptance = 1.0;
        Ok(ensemble)
    }
}

struct TorqueSampler {
    force: f64,
    torque: f64,
}

impl HyperplaneSampler for TorqueSampler {
    fn sample_hyperplane(
        &mut self,
        plane: &Hyperplane,
        budget: &HyperplaneSampleBudget,
    ) -> Result<HyperplaneEnsemble, SaddleError> {
        let normal = plane.normal();
        let tangent = array![-normal[1], normal[0]];
        let observations = (0..budget.samples)
            .map(|index| {
                let sign = if index.is_multiple_of(2) { 1.0 } else { -1.0 };
                let arm = &tangent * sign;
                HyperplaneObservation::default()
                    .with_constraint_force(self.force + sign * self.torque)
                    .with_position(&plane.point() + &arm)
                    .with_projected_force(&arm * -0.2)
            })
            .collect();
        let mut ensemble = HyperplaneEnsemble::default();
        ensemble.observations = observations;
        ensemble.evaluations = budget.samples;
        ensemble.acceptance = 1.0;
        Ok(ensemble)
    }
}

struct CapSampler {
    thermal: f64,
    pilot_sigma: f64,
    production_sigma: f64,
}

impl HyperplaneSampler for CapSampler {
    fn sample_hyperplane(
        &mut self,
        plane: &Hyperplane,
        budget: &HyperplaneSampleBudget,
    ) -> Result<HyperplaneEnsemble, SaddleError> {
        let mut rng = Rng(budget.seed.wrapping_mul(0xd134_2543_de82_ef95));
        let normal = plane.normal();
        let tangent = array![-normal[1], normal[0]];
        let (translation, rotation) = straight_gradient(plane, self.thermal);
        let pilot = budget.samples == ADAPTIVE_OHTST_PILOT_SAMPLES;
        let sigma = if pilot {
            self.pilot_sigma
        } else {
            self.production_sigma
        };
        let observations = (0..budget.samples)
            .map(|index| {
                let sign = if index.is_multiple_of(2) { -1.0 } else { 1.0 };
                let arm = &tangent * sign;
                HyperplaneObservation::default()
                    .with_constraint_force(-translation + sign * rotation + sigma * rng.normal())
                    .with_position(&plane.point() + &arm)
                    .with_projected_force(&arm * -self.thermal)
            })
            .collect();
        let mut ensemble = HyperplaneEnsemble::default();
        ensemble.observations = observations;
        ensemble.evaluations = budget.samples;
        ensemble.acceptance = 1.0;
        Ok(ensemble)
    }
}

fn gamma_ops(operations: f64) -> f64 {
    let product = operations * f64::EPSILON;
    product / (1.0 - product)
}

fn attach_wall(thermal: f64, wall: Arc<dyn Region>) -> AdaptiveOhTstConfig {
    let mut cfg = config(thermal);
    cfg.sampler = cfg.sampler.with_region(wall);
    cfg
}

#[test]
fn oblique_face_roundoff_uses_the_named_dot_count() {
    let thermal = 0.2;
    let initial = straight_planes();
    let dimension = 2.0;
    let gap = 0.5 * (gamma_ops(1.0) + gamma_ops(2.0 * dimension + 3.0));
    let wall = Arc::new(PlanarBoundary::new(array![0.0, 0.0], array![gap, 1.0]).unwrap());
    let dot = wall.normal().dot(&initial[0].normal()).abs();
    assert!(dot > gamma_ops(1.0) && dot <= gamma_ops(2.0 * dimension + 3.0));
    let mut sampler = ExactPlaneSampler {
        thermal,
        gradient: |plane: &Hyperplane| straight_gradient(plane, thermal),
        requested: Vec::new(),
    };
    adaptive_ohtst_string_rate(
        &mut sampler,
        &initial,
        &Density {
            plane: initial[0].clone(),
            thermal,
            error: 0.0,
        },
        &attach_wall(thermal, wall),
    )
    .expect("a face aligned within gamma(2D+3) is not an oblique cut");
}

#[test]
fn checked_cosine_accepts_a_two_ulp_normal_product() {
    let thermal = 0.2;
    let raw_a = [0.6034373567817739, 0.8620271682785496];
    let raw_b = [0.6034373634386266, 0.8620271752094535];
    let first = Hyperplane::new(array![-1.0, 0.0], array![raw_a[0], raw_a[1]]).unwrap();
    let second = Hyperplane::new(array![0.0, 0.0], array![raw_b[0], raw_b[1]]).unwrap();
    let third = Hyperplane::new(array![1.0, 0.0], array![raw_a[0], raw_a[1]]).unwrap();
    let excess = first.normal().dot(&second.normal()) - 1.0;
    let dimension = 2.0;
    assert!(excess > gamma_ops(1.0) && excess <= gamma_ops(2.0 * dimension + 3.0));
    let mut sampler = ExactPlaneSampler {
        thermal,
        gradient: |plane: &Hyperplane| straight_gradient(plane, thermal),
        requested: Vec::new(),
    };
    adaptive_ohtst_string_rate(
        &mut sampler,
        &[first.clone(), second, third],
        &Density {
            plane: first,
            thermal,
            error: 0.0,
        },
        &config(thermal).with_max_iterations(4),
    )
    .expect("gamma(2D+3) covers a two-ulp stored normal product");
}

#[test]
fn wall_rotation_roundoff_counts_the_updated_normal() {
    let thermal = 0.2;
    let dimension = 2.0;
    let gap = 0.5 * (gamma_ops(1.0) + gamma_ops(2.0 * dimension + 3.0));
    let wall = Arc::new(PlanarBoundary::new(array![0.0, 0.0], array![gap, 1.0]).unwrap());
    // A unit arm and a balanced torque t give a normal step t/kT. Then
    // 1 - cos = t^2 / (2 kT^2). The midpoint of gamma(1) and gamma(4D) is
    // inside the guard and outside gamma(1).
    let midpoint = 0.5 * (gamma_ops(1.0) + gamma_ops(4.0 * dimension));
    let torque = thermal * (2.0 * midpoint).sqrt();
    let planes = [
        Hyperplane::new(array![-1.0, 0.0], array![1.0, 0.0]).unwrap(),
        Hyperplane::new(array![0.0, 0.0], array![1.0, 0.0]).unwrap(),
        Hyperplane::new(array![1.0, 0.0], array![1.0, 0.0]).unwrap(),
    ];
    let mut sampler = TorqueSampler { force: 0.0, torque };
    adaptive_ohtst_string_rate(
        &mut sampler,
        &planes,
        &Density {
            plane: planes[0].clone(),
            thermal,
            error: 0.0,
        },
        &attach_wall(thermal, wall).with_max_iterations(2),
    )
    .expect("gamma(4D) allows this normalization deficit");
}

#[test]
fn temperature_roundoff_is_three_operations() {
    let thermal = 0.2;
    let shifted = thermal * (1.0 + 2.0 * f64::EPSILON);
    let difference = (shifted - thermal).abs();
    let scale = thermal.abs().max(shifted.abs());
    assert!(difference > gamma_ops(1.0) * scale && difference <= gamma_ops(3.0) * scale);
    let initial = straight_planes();
    let mut sampler = ExactPlaneSampler {
        thermal,
        gradient: |plane: &Hyperplane| straight_gradient(plane, thermal),
        requested: Vec::new(),
    };
    adaptive_ohtst_string_rate(
        &mut sampler,
        &initial,
        &Density {
            plane: initial[0].clone(),
            thermal: shifted,
            error: 0.0,
        },
        &config(thermal).with_max_iterations(4),
    )
    .expect("gamma(3) covers a two-ulp temperature gap");
}

#[test]
fn reactant_plane_roundoff_counts_the_point_difference() {
    let thermal = 0.2;
    let initial = straight_planes();
    let origin = initial[0].clone();
    let dimension = origin.dim() as f64;
    let point_scale = 1.0 + origin.point().dot(&origin.point()).sqrt();
    let shift = 0.5 * (gamma_ops(1.0) + gamma_ops(2.0 * dimension + 4.0)) * point_scale;
    let moved = Hyperplane::new(
        &origin.point() + &(&origin.normal() * shift),
        origin.normal().to_owned(),
    )
    .unwrap();
    let distance = origin.distance(moved.point()).abs();
    assert!(
        distance > gamma_ops(1.0) * point_scale
            && distance <= gamma_ops(2.0 * dimension + 4.0) * point_scale
    );
    let mut sampler = ExactPlaneSampler {
        thermal,
        gradient: |plane: &Hyperplane| straight_gradient(plane, thermal),
        requested: Vec::new(),
    };
    adaptive_ohtst_string_rate(
        &mut sampler,
        &initial,
        &Density {
            plane: moved,
            thermal,
            error: 0.0,
        },
        &config(thermal).with_max_iterations(4),
    )
    .expect("gamma(2D+4) covers the reactant point");
}

#[test]
fn a_precise_sample_at_the_record_cap_is_sample_cap() {
    let thermal = 0.2;
    let angle = 0.3;
    let initial = tilted_straight_planes(angle);
    // The pilot noise is kT/0.03, so the growth rule saturates at the cap.
    // A production noise of kT/10 keeps the 2048-record standard error under
    // the target without relaxing the tilt under the combined target.
    let mut sampler = CapSampler {
        thermal,
        pilot_sigma: thermal / ADAPTIVE_OHTST_TARGET_LOG_ERROR,
        production_sigma: 0.1 * thermal,
    };
    let estimate = adaptive_ohtst_string_rate(
        &mut sampler,
        &initial,
        &Density {
            plane: initial[0].clone(),
            thermal,
            error: 0.0,
        },
        &config(thermal),
    )
    .unwrap();
    assert_eq!(estimate.status, AdaptiveOhTstStatus::SampleCap);
    assert!(
        estimate
            .profile
            .samples_per_plane
            .iter()
            .all(|count| { *count == ADAPTIVE_OHTST_MAX_SAMPLES })
    );
    assert!(estimate.profile.sampled_barrier_error / thermal <= ADAPTIVE_OHTST_TARGET_LOG_ERROR);
    assert!(estimate.profile.barrier_error / thermal > ADAPTIVE_OHTST_TARGET_LOG_ERROR);
}

#[test]
fn an_odd_dividing_index_keeps_the_trapezoid_inside_its_bound() {
    let thermal = 0.2;
    let n = 32;
    let planes: Vec<_> = (0..n)
        .map(|index| {
            let x = -1.0 + index as f64 / (n - 1) as f64;
            Hyperplane::new(array![x, 0.0], array![1.0, 0.0]).unwrap()
        })
        .collect();
    let mut sampler = ExactPlaneSampler {
        thermal,
        gradient: |plane: &Hyperplane| straight_gradient(plane, thermal),
        requested: Vec::new(),
    };
    let estimate = adaptive_ohtst_string_rate(
        &mut sampler,
        &planes,
        &Density {
            plane: planes[0].clone(),
            thermal,
            error: 0.0,
        },
        &config(thermal).with_max_planes(n),
    )
    .unwrap();
    assert_eq!(estimate.profile.planes.len(), n);
    assert_eq!(estimate.profile.dividing % 2, 1);
    // f = 4x^3 - 4x on [-1, 0], so |f''| = |24x| <= 24. One trapezoid panel
    // of width h then errs by at most h^3/12 * 24 = 2 h^3.
    let spacing = 1.0 / (n - 1) as f64;
    let bound = 2.0 * spacing.powi(3);
    assert!(
        estimate.profile.quadrature_error <= bound,
        "quadrature {} exceeds the trapezoid bound {bound}",
        estimate.profile.quadrature_error
    );
}

#[test]
fn fixed_count_sampled_error_matches_the_barrier_deviation() {
    assert_fourteen_record_floor_is_seven();
    assert_fifteen_records_keep_seven_blocks();
    let thermal = 0.2;
    let chain: Vec<_> = (0..=16)
        .map(|index| {
            let x = index as f64 / 16.0 - 1.0;
            Hyperplane::new(array![x, 0.0], array![1.0, 0.0]).unwrap()
        })
        .collect();
    // One two-hundredth of kT. Kept runs stop at 14 records, so the sample
    // is not selected for a low error. The chain is fine enough that the
    // plane cap is not what holds the grid.
    let sigma = thermal / 200.0;
    let mut rows = Vec::new();
    for seed in 1..=40 {
        let mut sampler = NoisySampler { sigma, thermal };
        let adaptive = config(thermal).with_max_planes(chain.len()).with_sampler(
            LangevinConfig::default()
                .with_thermal(thermal)
                .with_seed(seed * 7_919),
        );
        let Ok(estimate) = adaptive_ohtst_string_rate(
            &mut sampler,
            &chain,
            &Density {
                plane: chain[0].clone(),
                thermal,
                error: 0.0,
            },
            &adaptive,
        ) else {
            continue;
        };
        if estimate.iterations != 0
            || estimate.profile.samples_per_plane[0] != ADAPTIVE_OHTST_PILOT_SAMPLES
        {
            continue;
        }
        rows.push((
            estimate.profile.barrier,
            estimate.profile.sampled_barrier_error,
        ));
    }
    assert!(
        rows.len() >= 8,
        "only {} stationary fixed-count runs",
        rows.len()
    );
    let mean = rows.iter().map(|item| item.0).sum::<f64>() / rows.len() as f64;
    let empirical = (rows.iter().map(|item| (item.0 - mean).powi(2)).sum::<f64>()
        / (rows.len() - 1) as f64)
        .sqrt();
    let reported = rows.iter().map(|item| item.1).sum::<f64>() / rows.len() as f64;
    // Seven blocks leave ν = 6. E[s]/σ = sqrt(2/ν) Γ((ν+1)/2) / Γ(ν/2)
    // = sqrt(1/3) * (15/16) * sqrt(π).
    let ratio = (1.0_f64 / 3.0).sqrt() * (15.0 / 16.0) * std::f64::consts::PI.sqrt();
    assert!(
        reported >= ratio * empirical,
        "sampled {reported}, empirical {empirical}, ratio {ratio}"
    );
    assert!(reported / thermal <= ADAPTIVE_OHTST_TARGET_LOG_ERROR);
}

fn fixed_count_iid_run(seed: u64) -> Result<(f64, f64, usize), SaddleError> {
    let thermal = 0.2;
    let chain: Vec<_> = (0..=16)
        .map(|index| {
            let x = index as f64 / 16.0 - 1.0;
            Hyperplane::new(array![x, 0.0], array![1.0, 0.0]).unwrap()
        })
        .collect();
    // One two-hundredth of kT. The pilot standard error stays under the
    // target, so every kept run stops at 14 records and is not selected
    // for a low error. The chain is fine enough that the plane cap is not
    // what holds the grid.
    let mut sampler = ProductionNoisySampler {
        noisy: NoisySampler {
            sigma: thermal / 200.0,
            thermal,
        },
        pilot_planes: chain.len(),
    };
    let adaptive = config(thermal).with_max_planes(chain.len()).with_sampler(
        LangevinConfig::default()
            .with_thermal(thermal)
            .with_seed(seed * 7_919),
    );
    let estimate = adaptive_ohtst_string_rate(
        &mut sampler,
        &chain,
        &Density {
            plane: chain[0].clone(),
            thermal,
            error: 0.0,
        },
        &adaptive,
    )?;
    Ok((
        estimate.profile.barrier,
        estimate.profile.sampled_barrier_error,
        estimate.profile.samples_per_plane[0],
    ))
}

/// Seeds 1..=200. Fourteen records use seven blocks, so the caller's floor
/// is `n_eff >= 7`. On this block the floor refuses 112 draws. The binomial
/// standard error is `sqrt(112 * 88 / 200) = 7.0`, and the window is three
/// of those standard errors. A literal floor of 4 refuses none of them:
/// the rejected `n_eff` values sit above 4 and below 7. Split-half
/// rejections are counted and are not turned into successes.
#[test]
fn fourteen_record_production_floor_is_the_caller_block_count() {
    assert_fifteen_records_keep_seven_blocks();
    let mut ise = 0usize;
    let mut ne = 0usize;
    let mut other = 0usize;
    let outcomes: Vec<_> = (1_u64..=200)
        .into_par_iter()
        .map(fixed_count_iid_run)
        .collect();
    for outcome in outcomes {
        match outcome {
            Ok(_) => {}
            Err(SaddleError::InsufficientEffectiveSamples {
                n_eff, required, ..
            }) => {
                assert!(
                    n_eff < required && required == 7.0,
                    "n_eff {n_eff}, required {required}"
                );
                ise += 1;
            }
            Err(SaddleError::NotEquilibrated { .. }) => ne += 1,
            Err(_) => other += 1,
        }
    }
    let low = 112usize.saturating_sub(21);
    let high = 112 + 21;
    assert!(
        (low..=high).contains(&ise),
        "floor refused {ise}, base 112, window {low}..={high}, split-half {ne}, other {other}"
    );
}

/// Seeds 1..=200, 1001..=1200, and 10001..=10200 are three disjoint,
/// pre-registered blocks. Coverage is the fraction of *returned* estimates
/// whose barrier lies inside two reported standard errors, and that fraction
/// must lie in 90%..=99%. For 200 Bernoulli trials at nominal 95% the
/// binomial standard error is 1.54 percentage points, so the band permits
/// 3.2 standard errors below nominal and 2.6 above. A floor rejection is not
/// a covered draw and is not dropped from the log: `required` must be the
/// caller's seven blocks. A literal floor of 4 reports `required = 4` and
/// fails that check. Split-half rejections stay in the `ne` count.
fn assert_fixed_count_coverage(first: u64) {
    assert_fourteen_record_floor_is_seven();
    assert_fifteen_records_keep_seven_blocks();
    let last = first + 199;
    let mut returned = 0usize;
    let mut covered = 0usize;
    let mut ne = 0usize;
    let mut other = 0usize;
    let outcomes: Vec<_> = (first..=last)
        .into_par_iter()
        .map(|seed| (seed, fixed_count_iid_run(seed)))
        .collect();
    for (seed, outcome) in outcomes {
        match outcome {
            Ok((barrier, error, records)) => {
                assert_eq!(records, ADAPTIVE_OHTST_PILOT_SAMPLES);
                returned += 1;
                if (barrier - 1.0).abs() <= 2.0 * error {
                    covered += 1;
                }
            }
            Err(SaddleError::InsufficientEffectiveSamples {
                n_eff, required, ..
            }) => {
                assert!(
                    n_eff < required && required == 7.0,
                    "seed {seed}: n_eff {n_eff}, required {required}"
                );
            }
            Err(SaddleError::NotEquilibrated { .. }) => ne += 1,
            Err(_) => other += 1,
        }
    }
    assert!(returned > 0, "seeds {first}..={last} returned nothing");
    let coverage = covered as f64 / returned as f64;
    assert!(
        (0.90..=0.99).contains(&coverage),
        "seeds {first}..={last}: {covered} of {returned} returned covered ({coverage}), split-half {ne}, other {other}"
    );
}

#[test]
fn fixed_count_sampled_error_has_two_sided_coverage_first_block() {
    assert_fixed_count_coverage(1);
}

#[test]
fn fixed_count_sampled_error_has_two_sided_coverage_second_block() {
    assert_fixed_count_coverage(1_001);
}

#[test]
fn fixed_count_sampled_error_has_two_sided_coverage_third_block() {
    assert_fixed_count_coverage(10_001);
}
