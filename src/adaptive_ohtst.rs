//! Adaptive finite-temperature strings of optimal hyperplanes.
//!
//! Unlike fixed-path PAFI, this method translates and rotates interior
//! hyperplanes under their sampled OH-TST free-energy gradients.

use ndarray::{Array1, ArrayView1, s};

use crate::SaddleError;
use crate::ohtst::{
    Hyperplane, LangevinConfig, OhTstRate, PlaneDensity, PlanePath, PlaneSample, RateMultiplicity,
    rate_from_parts, sample_plane_pilot_with_sampler, sample_plane_with_sampler,
};
use crate::projector::HyperplaneSampler;

/// Mandatory first sample count.
pub const ADAPTIVE_OHTST_PILOT_SAMPLES: usize = 14;
/// Target standard error of the sampled log-rate contribution.
pub const ADAPTIVE_OHTST_TARGET_LOG_ERROR: f64 = 0.03;
/// Largest retained sample count at one plane.
pub const ADAPTIVE_OHTST_MAX_SAMPLES: usize = 2048;
/// Prescribed sample-growth safety factor.
pub const ADAPTIVE_OHTST_SAMPLE_SAFETY: f64 = 1.1;

/// Controls for [`adaptive_ohtst_string_rate`].
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct AdaptiveOhTstConfig {
    /// Shared constrained-sampler controls. The method owns `records`.
    pub sampler: LangevinConfig,
    /// Geometry-relaxation iteration cap.
    pub max_iterations: usize,
    /// Plane cap after profile refinement.
    pub max_planes: usize,
}

impl Default for AdaptiveOhTstConfig {
    fn default() -> Self {
        Self {
            sampler: LangevinConfig::default(),
            max_iterations: 64,
            max_planes: 257,
        }
    }
}

impl AdaptiveOhTstConfig {
    /// Set all sampler controls except the method-owned record count.
    #[must_use]
    pub fn with_sampler(mut self, sampler: LangevinConfig) -> Self {
        self.sampler = sampler;
        self
    }

    /// Set `kT`.
    #[must_use]
    pub fn with_thermal(mut self, thermal: f64) -> Self {
        self.sampler.thermal = thermal;
        self
    }

    /// Set the geometry-relaxation cap.
    #[must_use]
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set the adaptive plane cap.
    #[must_use]
    pub fn with_max_planes(mut self, max_planes: usize) -> Self {
        self.max_planes = max_planes;
        self
    }

    fn check(&self, planes: &[Hyperplane]) -> Result<(), SaddleError> {
        if !(self.sampler.thermal.is_finite() && self.sampler.thermal > 0.0) {
            return Err(SaddleError::Invalid(
                "adaptive OH-TST needs a finite positive temperature".into(),
            ));
        }
        if self.max_iterations == 0 || planes.len() < 2 || self.max_planes < planes.len() {
            return Err(SaddleError::Invalid(
                "adaptive OH-TST caps must admit the initial plane chain".into(),
            ));
        }
        let dimension = planes[0].dim();
        if dimension == 0 || planes.iter().any(|plane| plane.dim() != dimension) {
            return Err(SaddleError::Shape(
                "adaptive OH-TST planes need one common nonzero dimension".into(),
            ));
        }
        if planes
            .windows(2)
            .any(|pair| pair[0].normal().dot(&pair[1].normal()) <= 0.0)
        {
            return Err(SaddleError::Invalid(
                "adaptive OH-TST plane normals must have consistent orientations".into(),
            ));
        }
        if let Some(region) = &self.sampler.region {
            let boundaries = region.planar_boundaries().ok_or_else(|| {
                SaddleError::Invalid(
                    "adaptive OH-TST needs wall-contact derivatives for a non-planar hard region"
                        .into(),
                )
            })?;
            if !boundaries.is_empty() {
                let reference = planes[0].normal();
                // One length-D dot, counted as 2D flops, then a subtraction
                // from 1, an absolute comparison, and the tolerance product.
                let roundoff = gamma_n(2 * dimension + 3)?;
                let rotates = planes
                    .iter()
                    .any(|plane| 1.0 - plane.normal().dot(&reference) > roundoff);
                let cuts_obliquely = boundaries
                    .iter()
                    .any(|boundary| boundary.normal().dot(&reference).abs() > roundoff);
                if rotates || cuts_obliquely {
                    return Err(SaddleError::Invalid(
                        "adaptive OH-TST needs the wall-contact derivative when a bounded plane rotates or moves across an oblique face"
                            .into(),
                    ));
                }
            }
        }
        Ok(())
    }
}

/// A free-energy profile on a relaxed finite-temperature plane string.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct AdaptiveOhTstProfile {
    /// Ordered relaxed planes.
    pub planes: Vec<Hyperplane>,
    /// Free energy relative to the first plane.
    pub free_energy: Vec<f64>,
    /// Propagated standard error at every plane.
    pub free_energy_error: Vec<f64>,
    /// Largest profile value.
    pub barrier: f64,
    /// Standard error of [`Self::barrier`].
    pub barrier_error: f64,
    /// Sampling contribution to [`Self::barrier_error`].
    pub sampled_barrier_error: f64,
    /// Measured barrier change under neighbor-smoothed normals and a trust-limited step.
    pub geometry_error: f64,
    /// Measured force-integral discrepancy from grid doubling and the offset rule.
    pub quadrature_error: f64,
    /// Index of the selected dividing plane.
    pub dividing: usize,
    /// Records requested at each final plane.
    pub samples_per_plane: Vec<usize>,
    /// Shared samples supplying translation, rotation, and work.
    pub plane_samples: Vec<PlaneSample>,
}

/// Whether adaptive plane sampling met its prescribed log-error target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdaptiveOhTstStatus {
    /// The total log-rate error, including geometry and quadrature, reached the target.
    TargetMet,
    /// The sample cap was reached with the total error still above the target.
    /// A sample that is already precise counts here when geometry or quadrature
    /// still misses the target: the cap, not the sample size, is what stopped the run.
    SampleCap,
    /// Sampling is already at its target, but geometry or quadrature keeps the
    /// total error above it, and the record cap was not reached. Further records
    /// would not change that.
    Unresolved,
}

/// Adaptive OH-TST string barrier composed with the one-way flux.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct AdaptiveOhTstEstimate {
    /// Rate and its propagated standard error.
    pub rate: OhTstRate,
    /// Relaxed path, profile, barrier, and profile uncertainty.
    pub profile: AdaptiveOhTstProfile,
    /// Plane at the profile maximum.
    pub dividing_plane: Hyperplane,
    /// Geometry-relaxation iterations.
    pub iterations: usize,
    /// Force or energy evaluations.
    pub evaluations: usize,
    /// Sampling termination status.
    pub status: AdaptiveOhTstStatus,
}

/// Apply the prescribed sample-growth rule.
pub fn adaptive_ohtst_sample_count(current: usize, error: f64) -> Result<usize, SaddleError> {
    if current == 0 || current > ADAPTIVE_OHTST_MAX_SAMPLES || !(error.is_finite() && error >= 0.0)
    {
        return Err(SaddleError::Invalid(
            "adaptive sampling needs a count inside its cap and finite non-negative error".into(),
        ));
    }
    let ratio = error / ADAPTIVE_OHTST_TARGET_LOG_ERROR;
    let proposed = (ADAPTIVE_OHTST_SAMPLE_SAFETY * current as f64 * ratio * ratio).ceil() as usize;
    Ok(proposed.max(current).min(ADAPTIVE_OHTST_MAX_SAMPLES))
}

/// Relax an OH-TST hyperplane string and estimate its rate.
pub fn adaptive_ohtst_string_rate<E, D>(
    sampler: &mut E,
    initial: &[Hyperplane],
    density: &D,
    config: &AdaptiveOhTstConfig,
) -> Result<AdaptiveOhTstEstimate, SaddleError>
where
    E: HyperplaneSampler,
    D: PlaneDensity + ?Sized,
{
    config.check(initial)?;
    let mut planes = initial.to_vec();
    let (pilot, mut evaluations) = sample_chain(
        sampler,
        &planes,
        config,
        ADAPTIVE_OHTST_PILOT_SAMPLES,
        0,
        true,
    )?;
    let pilot_profile = profile_from_samples(&planes, &pilot, ADAPTIVE_OHTST_PILOT_SAMPLES)?;
    let mut count = adaptive_ohtst_sample_count(
        ADAPTIVE_OHTST_PILOT_SAMPLES,
        pilot_profile.sampled_barrier_error / config.sampler.thermal,
    )?;
    let mut iterations = 0;

    loop {
        let sampled = sample_chain(sampler, &planes, config, count, count as u64, false)?;
        let mut samples = sampled.0;
        evaluations = checked_sum(evaluations, sampled.1)?;
        loop {
            let (next, converged) = string_step(&planes, &samples, config)?;
            if converged {
                break;
            }
            if iterations == config.max_iterations {
                return Err(SaddleError::Solver(
                    "adaptive OH-TST plane string did not converge".into(),
                ));
            }
            iterations += 1;
            planes = equal_arc_planes(&next, &samples)?;
            let sampled = sample_chain(
                sampler,
                &planes,
                config,
                count,
                (count as u64).wrapping_add(iterations as u64),
                false,
            )?;
            samples = sampled.0;
            evaluations = checked_sum(evaluations, sampled.1)?;
        }
        let refined = refine_profile(sampler, planes, samples, config, count, &mut evaluations)?;
        let resolved = resolve_measured_quadrature(
            sampler,
            refined.0,
            refined.1,
            config,
            count,
            &mut evaluations,
        )?;
        planes = resolved.0;
        samples = resolved.1;
        let profile = profile_from_samples(&planes, &samples, count)?;
        let mut rate = rate_from_profile(density, &profile, config.sampler.thermal)?;
        if let Some(region) = &config.sampler.region {
            rate = rate.multiplied(&RateMultiplicity::default().with_region(region.as_ref()))?;
        }
        let sampled_log_error = profile.sampled_barrier_error / config.sampler.thermal;
        let total_log_error = profile.barrier_error / config.sampler.thermal;
        if total_log_error <= ADAPTIVE_OHTST_TARGET_LOG_ERROR
            || sampled_log_error <= ADAPTIVE_OHTST_TARGET_LOG_ERROR
            || count == ADAPTIVE_OHTST_MAX_SAMPLES
        {
            // SampleCap means the cap was reached with the total target unmet,
            // even if the sampled error is already inside the target.
            // Unresolved is that split only below the cap.
            let status = if total_log_error <= ADAPTIVE_OHTST_TARGET_LOG_ERROR {
                AdaptiveOhTstStatus::TargetMet
            } else if sampled_log_error <= ADAPTIVE_OHTST_TARGET_LOG_ERROR
                && count != ADAPTIVE_OHTST_MAX_SAMPLES
            {
                AdaptiveOhTstStatus::Unresolved
            } else {
                AdaptiveOhTstStatus::SampleCap
            };
            let dividing_plane = planes[profile.dividing].clone();
            return Ok(AdaptiveOhTstEstimate {
                rate,
                profile,
                dividing_plane,
                iterations,
                evaluations,
                status,
            });
        }
        let next = adaptive_ohtst_sample_count(count, sampled_log_error)?;
        if next <= count {
            return Err(SaddleError::Solver(
                "adaptive OH-TST sample rule did not increase an unresolved count".into(),
            ));
        }
        count = next;
    }
}

fn sample_chain<E: HyperplaneSampler>(
    sampler: &mut E,
    planes: &[Hyperplane],
    config: &AdaptiveOhTstConfig,
    count: usize,
    salt: u64,
    pilot: bool,
) -> Result<(Vec<PlaneSample>, usize), SaddleError> {
    let mut samples = Vec::with_capacity(planes.len());
    let mut evaluations = 0;
    let pilot_false_rejection = ADAPTIVE_OHTST_TARGET_LOG_ERROR / planes.len() as f64;
    for (index, plane) in planes.iter().enumerate() {
        let sample = sample_one(
            sampler,
            plane,
            config,
            count,
            salt.wrapping_add(index as u64),
            pilot,
            pilot_false_rejection,
        )?;
        evaluations = checked_sum(evaluations, sample.evaluations)?;
        samples.push(sample);
    }
    Ok((samples, evaluations))
}

fn sample_one<E: HyperplaneSampler>(
    sampler: &mut E,
    plane: &Hyperplane,
    config: &AdaptiveOhTstConfig,
    count: usize,
    salt: u64,
    pilot: bool,
    pilot_false_rejection: f64,
) -> Result<PlaneSample, SaddleError> {
    let mut stride = config.sampler.stride.max(1);
    let mut burn_in = config.sampler.burn_in.max(1);
    // Fourteen records support τ ≤ 14/8 = 1.75. An AR(1) of parameter φ has
    // τ = (1+φ)/(2(1-φ)); φ = 0.9 is τ ≈ 9.5 and is rejected. Retaining every
    // s-th step replaces φ by φ^s. From the default stride of 5 the doublings
    // are 10, 20, and 40, and 0.9^10 ≈ 0.35 has τ ≈ 1.0. From stride 1 the
    // same cap stops at 8. The stride that passes is not reused for production
    // samples. A sampler that ignores stride still returns the effective-sample error.
    let stride_cap = stride.saturating_mul(ADAPTIVE_OHTST_PILOT_SAMPLES);
    loop {
        let mut local = sample_config(config, count, salt);
        local.stride = stride;
        local.burn_in = burn_in;
        let sampled = sample_with_policy(sampler, plane, &local, pilot, pilot_false_rejection);
        if pilot
            && matches!(
                sampled,
                Err(SaddleError::InsufficientEffectiveSamples { .. })
            )
            && stride.saturating_mul(2) <= stride_cap
        {
            stride = stride.saturating_mul(2);
            burn_in = burn_in.saturating_mul(2);
            continue;
        }
        return sampled;
    }
}

fn sample_with_policy<E: HyperplaneSampler>(
    sampler: &mut E,
    plane: &Hyperplane,
    local: &LangevinConfig,
    pilot: bool,
    pilot_false_rejection: f64,
) -> Result<PlaneSample, SaddleError> {
    if pilot {
        sample_plane_pilot_with_sampler(sampler, plane, local, pilot_false_rejection)
    } else {
        sample_plane_with_sampler(sampler, plane, local)
    }
}

fn sample_config(config: &AdaptiveOhTstConfig, count: usize, salt: u64) -> LangevinConfig {
    let blocks = if count == ADAPTIVE_OHTST_PILOT_SAMPLES {
        count / 2
    } else {
        let mut root = 1;
        while (root + 1) <= count / (root + 1) / (root + 1) {
            root += 1;
        }
        root
    };
    let mut local = config.sampler.clone();
    local.records = count;
    local.blocks = blocks.max(2);
    local.retain = local.retain.max(count);
    local.seed = local.seed.wrapping_add(salt);
    local
}

fn string_step(
    planes: &[Hyperplane],
    samples: &[PlaneSample],
    config: &AdaptiveOhTstConfig,
) -> Result<(Vec<Hyperplane>, bool), SaddleError> {
    if planes.len() != samples.len() {
        return Err(SaddleError::Shape(
            "adaptive OH-TST planes and samples differ in length".into(),
        ));
    }
    if planes.len() < 3 {
        return Ok((planes.to_vec(), true));
    }
    let scales: Vec<f64> = samples
        .iter()
        .map(rotation_scale)
        .collect::<Result<_, _>>()?;
    let mut next = planes.to_vec();
    let mut converged = true;
    for index in 1..planes.len() - 1 {
        let plane = &planes[index];
        let sample = &samples[index];
        let scale = scales[index];
        let tangent = string_tangent(&planes[index - 1], plane, &planes[index + 1], scale);
        let tangent_norm2 = tangent.dot(&tangent);
        if !(tangent_norm2.is_finite() && tangent_norm2 > 0.0) {
            return Err(SaddleError::Invalid(
                "adaptive OH-TST string tangent vanishes".into(),
            ));
        }
        let mut gradient = scaled_gradient(sample, scale);
        let unprojected_norm = gradient.dot(&gradient).sqrt();
        let along = gradient.dot(&tangent) / tangent_norm2;
        gradient.scaled_add(-along, &tangent);
        let norm2 = gradient.dot(&gradient);
        let trace_covariance = projected_gradient_variance(sample, scale, tangent.view())?;
        // Four length-m dots (the tangent norm, the unprojected norm, the
        // tangent coefficient, and the projected norm) and one axpy. Two
        // flops on each of the m components is the upper bound 8m.
        let gamma = gamma_n(8 * gradient.len())?;
        // γ bounds the absolute error of the projected gradient. Comparing
        // squared lengths would square that bound below the ulp of a physical
        // ||g||², so the added term is the expansion (σ + γ s)² − σ².
        let absolute = gamma * (unprojected_norm + config.sampler.thermal / scale);
        let sigma = trace_covariance.max(0.0).sqrt();
        let roundoff = (2.0 * sigma * absolute + absolute * absolute).sqrt();
        if norm2 <= trace_covariance + roundoff * roundoff {
            continue;
        }
        let left = generalized_distance(&planes[index - 1], plane, scales[index - 1], scale)?;
        let right = generalized_distance(plane, &planes[index + 1], scale, scales[index + 1])?;
        let no_crossing = 0.5 * left.min(right);
        if !(no_crossing.is_finite() && no_crossing > 0.0) {
            return Err(SaddleError::Invalid(
                "adaptive OH-TST neighboring planes are unresolved".into(),
            ));
        }
        // A rotation by one radian moves a typical retained configuration by
        // `scale`; equipartition supplies the initial rotational curvature
        // kT / scale². Translation uses the corresponding local plane spacing.
        // Neighboring sampled gradients provide secant lower bounds on
        // sharper curvature in each block, avoiding mixed-unit overshoot.
        let translation_scale = 0.5 * (left + right);
        let mut translation_curvature = config.sampler.thermal / translation_scale.powi(2);
        let mut rotation_curvature = config.sampler.thermal / scale.powi(2);
        for neighbor in [index - 1, index + 1] {
            let neighbor_gradient = scaled_gradient(&samples[neighbor], scales[neighbor]);
            let point_change = &planes[neighbor].point() - &plane.point();
            let point_distance = plane.normal().dot(&point_change).abs();
            if point_distance > 0.0 {
                let secant = (neighbor_gradient[0] - gradient[0]).abs() / point_distance;
                if secant > translation_curvature {
                    translation_curvature = secant;
                }
            }
            let angle =
                checked_cosine(plane.normal().dot(&planes[neighbor].normal()), plane.dim())?.acos();
            let rotation_distance = scale * angle;
            if rotation_distance > 0.0 {
                let change = &neighbor_gradient.slice(s![1..]) - &gradient.slice(s![1..]);
                let secant = change.dot(&change).sqrt() / rotation_distance;
                if secant > rotation_curvature {
                    rotation_curvature = secant;
                }
            }
        }
        if !(translation_curvature.is_finite()
            && translation_curvature > 0.0
            && rotation_curvature.is_finite()
            && rotation_curvature > 0.0)
        {
            return Err(SaddleError::NonFinite("adaptive OH-TST thermal curvature"));
        }
        let rotation_norm2 = gradient.slice(s![1..]).dot(&gradient.slice(s![1..]));
        let model_error = 0.5
            * (gradient[0].powi(2) / translation_curvature + rotation_norm2 / rotation_curvature);
        let geometry_budget =
            config.sampler.thermal * ADAPTIVE_OHTST_TARGET_LOG_ERROR / (planes.len() - 2) as f64;
        let mut displacement = gradient;
        displacement[0] /= translation_curvature;
        displacement
            .slice_mut(s![1..])
            .mapv_inplace(|value| value / rotation_curvature);
        let unconstrained = displacement.dot(&displacement).sqrt();
        // Half the neighboring separation is the largest step that cannot cross.
        let trust = no_crossing;
        let length = if unconstrained <= trust {
            1.0
        } else {
            trust / unconstrained
        };
        displacement *= length;
        let mut point = projected_centroid(sample);
        point.scaled_add(displacement[0], &plane.normal());
        let mut normal = plane.normal().to_owned() + &(&displacement.slice(s![1..]) / scale);
        let normal_length = normal.dot(&normal).sqrt();
        if !(normal_length.is_finite() && normal_length > 0.0) {
            return Err(SaddleError::Invalid(
                "adaptive OH-TST update erased a plane normal".into(),
            ));
        }
        normal /= normal_length;
        if normal.dot(&plane.normal()) <= 0.0 {
            return Err(SaddleError::Invalid(
                "adaptive OH-TST update reversed a plane normal".into(),
            ));
        }
        if let Some(region) = &config.sampler.region
            && region
                .planar_boundaries()
                .is_none_or(|boundaries| !boundaries.is_empty())
        {
            // Normalization of the updated normal and its dot with the old
            // normal: two length-D dots, two flops on each component.
            let roundoff = gamma_n(4 * plane.dim())?;
            if 1.0 - normal.dot(&plane.normal()) > roundoff {
                return Err(SaddleError::Invalid(
                    "adaptive OH-TST needs the wall-contact derivative before rotating a bounded plane"
                        .into(),
                ));
            }
        }
        if model_error <= geometry_budget {
            continue;
        }
        converged = false;
        next[index] = Hyperplane::new(point, normal)?;
    }
    Ok((next, converged))
}

fn projected_centroid(sample: &PlaneSample) -> Array1<f64> {
    let distance = sample.plane.distance(sample.centroid.view());
    &sample.centroid - &(&sample.plane.normal() * distance)
}

fn rotation_scale(sample: &PlaneSample) -> Result<f64, SaddleError> {
    if sample.retained.is_empty() {
        return Err(SaddleError::Invalid(
            "adaptive OH-TST needs retained configurations for its plane metric".into(),
        ));
    }
    let variance = sample
        .retained
        .iter()
        .map(|position| {
            let displacement = position - &sample.centroid;
            displacement.dot(&displacement)
        })
        .sum::<f64>()
        / sample.retained.len() as f64;
    let scale = variance.sqrt();
    if !(scale.is_finite() && scale > 0.0) {
        return Err(SaddleError::Invalid(
            "adaptive OH-TST sampled rotation metric vanishes".into(),
        ));
    }
    Ok(scale)
}

fn scaled_gradient(sample: &PlaneSample, scale: f64) -> Array1<f64> {
    let dimension = sample.plane.dim();
    let mut gradient = Array1::zeros(1 + dimension);
    gradient[0] = sample.grad_point.dot(&sample.plane.normal());
    gradient
        .slice_mut(s![1..])
        .assign(&(&sample.centroid_torque / scale));
    gradient
}

fn string_tangent(
    previous: &Hyperplane,
    current: &Hyperplane,
    next: &Hyperplane,
    scale: f64,
) -> Array1<f64> {
    let dimension = current.dim();
    let mut tangent = Array1::zeros(1 + dimension);
    tangent[0] = current.normal().dot(&(&next.point() - &previous.point()));
    let mut turn = &next.normal() - &previous.normal();
    let along = turn.dot(&current.normal());
    turn.scaled_add(-along, &current.normal());
    tangent.slice_mut(s![1..]).assign(&(turn * scale));
    tangent
}

/// Trace of the covariance of the projected sample-mean gradient.
///
/// The string skips a move when the squared gradient is within that trace
/// plus roundoff. This is one standard deviation of the mean gradient, not a
/// Bonferroni or Markov test. Convergence of the profile is the measured
/// quadrature and orientation errors against `0.03 kT`.
fn projected_gradient_variance(
    sample: &PlaneSample,
    scale: f64,
    tangent: ArrayView1<'_, f64>,
) -> Result<f64, SaddleError> {
    let blocks = sample.centroid_block_gradients.nrows();
    if blocks < 2 {
        return Err(SaddleError::Invalid(
            "adaptive OH-TST needs at least two gradient blocks".into(),
        ));
    }
    let dimension = sample.plane.dim();
    let tangent_norm2 = tangent.dot(&tangent);
    let mut projected = Vec::with_capacity(blocks);
    for row in sample.centroid_block_gradients.rows() {
        let mut value = Array1::zeros(1 + dimension);
        value[0] = row.slice(s![..dimension]).dot(&sample.plane.normal());
        value
            .slice_mut(s![1..])
            .assign(&(&row.slice(s![dimension..]) / scale));
        let along = value.dot(&tangent) / tangent_norm2;
        value.scaled_add(-along, &tangent);
        projected.push(value);
    }
    let mean = projected
        .iter()
        .fold(Array1::<f64>::zeros(1 + dimension), |sum, value| {
            sum + value
        })
        / blocks as f64;
    let sum = projected
        .iter()
        .map(|value| {
            let difference = value - &mean;
            difference.dot(&difference)
        })
        .sum::<f64>();
    let variance = sum / (blocks * (blocks - 1)) as f64;
    if variance.is_finite() && variance >= 0.0 {
        Ok(variance)
    } else {
        Err(SaddleError::NonFinite(
            "adaptive OH-TST gradient covariance",
        ))
    }
}

fn gamma_n(operations: usize) -> Result<f64, SaddleError> {
    let product = operations as f64 * f64::EPSILON;
    if product < 1.0 {
        Ok(product / (1.0 - product))
    } else {
        Err(SaddleError::Invalid(
            "floating-point operation count exceeds the roundoff model".into(),
        ))
    }
}

fn checked_cosine(cosine: f64, dimension: usize) -> Result<f64, SaddleError> {
    if (-1.0..=1.0).contains(&cosine) {
        return Ok(cosine);
    }
    let endpoint = if cosine > 1.0 { 1.0 } else { -1.0 };
    // The cosine is one length-D dot (2D flops) followed by a subtraction
    // from the endpoint, an absolute value, and the comparison.
    let rounding = gamma_n(2 * dimension + 3)?;
    if (cosine - endpoint).abs() <= rounding {
        Ok(endpoint)
    } else {
        Err(SaddleError::Invalid(
            "adaptive OH-TST plane normals are not normalized".into(),
        ))
    }
}

fn generalized_distance(
    left: &Hyperplane,
    right: &Hyperplane,
    left_scale: f64,
    right_scale: f64,
) -> Result<f64, SaddleError> {
    let displacement = &right.point() - &left.point();
    let angle = checked_cosine(left.normal().dot(&right.normal()), left.dim())?.acos();
    let scale = 0.5 * (left_scale + right_scale);
    let distance = displacement
        .dot(&displacement)
        .mul_add(1.0, (scale * angle).powi(2))
        .sqrt();
    if distance.is_finite() {
        Ok(distance)
    } else {
        Err(SaddleError::NonFinite(
            "adaptive OH-TST generalized distance",
        ))
    }
}

fn equal_arc_planes(
    planes: &[Hyperplane],
    samples: &[PlaneSample],
) -> Result<Vec<Hyperplane>, SaddleError> {
    let scales: Vec<f64> = samples
        .iter()
        .map(rotation_scale)
        .collect::<Result<_, _>>()?;
    let mut arc = Vec::with_capacity(planes.len());
    arc.push(0.0);
    for index in 1..planes.len() {
        let length = generalized_distance(
            &planes[index - 1],
            &planes[index],
            scales[index - 1],
            scales[index],
        )?;
        if length <= 0.0 {
            return Err(SaddleError::Invalid(
                "adaptive OH-TST plane string has a zero segment".into(),
            ));
        }
        arc.push(arc[index - 1] + length);
    }
    let total = arc[arc.len() - 1];
    let mut equal = Vec::with_capacity(planes.len());
    equal.push(planes[0].clone());
    for index in 1..planes.len() - 1 {
        let target = total * index as f64 / (planes.len() - 1) as f64;
        let segment = arc
            .windows(2)
            .position(|window| target <= window[1])
            .ok_or_else(|| {
                SaddleError::Solver(
                    "adaptive OH-TST equal-arc target lies outside the string".into(),
                )
            })?;
        let fraction = (target - arc[segment]) / (arc[segment + 1] - arc[segment]);
        equal.push(interpolate_plane(
            &planes[segment],
            &planes[segment + 1],
            fraction,
        )?);
    }
    equal.push(planes[planes.len() - 1].clone());
    Ok(equal)
}

fn interpolate_plane(
    from: &Hyperplane,
    to: &Hyperplane,
    fraction: f64,
) -> Result<Hyperplane, SaddleError> {
    if !(fraction.is_finite() && (0.0..=1.0).contains(&fraction)) {
        return Err(SaddleError::Invalid(
            "adaptive OH-TST interpolation fraction lies outside [0, 1]".into(),
        ));
    }
    let point = &from.point() + &((&to.point() - &from.point()) * fraction);
    let angle = checked_cosine(from.normal().dot(&to.normal()), from.dim())?.acos();
    let normal = if angle == 0.0 {
        from.normal().to_owned()
    } else {
        let sine = angle.sin();
        if sine == 0.0 {
            return Err(SaddleError::Invalid(
                "adaptive OH-TST cannot interpolate antiparallel normals".into(),
            ));
        }
        &from.normal() * ((1.0 - fraction) * angle).sin() / sine
            + &to.normal() * (fraction * angle).sin() / sine
    };
    Hyperplane::new(point, normal)
}

fn refine_profile<E: HyperplaneSampler>(
    sampler: &mut E,
    planes: Vec<Hyperplane>,
    samples: Vec<PlaneSample>,
    config: &AdaptiveOhTstConfig,
    count: usize,
    evaluations: &mut usize,
) -> Result<(Vec<Hyperplane>, Vec<PlaneSample>), SaddleError> {
    if planes.len() != samples.len() || planes.len() < 2 {
        return Err(SaddleError::Shape(
            "adaptive OH-TST refinement needs one sample per plane".into(),
        ));
    }
    let initial_edges = planes.len() - 1;
    let edge_budget =
        ADAPTIVE_OHTST_TARGET_LOG_ERROR * config.sampler.thermal / initial_edges as f64;
    let mut refined_planes = vec![planes[0].clone()];
    let mut refined_samples = vec![samples[0].clone()];
    let mut inserted = 0;
    let mut seed_salt = planes.len() as u64;
    for edge in 0..initial_edges {
        refine_edge(
            sampler,
            &planes[edge],
            &samples[edge],
            &planes[edge + 1],
            &samples[edge + 1],
            edge_budget,
            config,
            count,
            evaluations,
            &mut inserted,
            &mut seed_salt,
            &mut refined_planes,
            &mut refined_samples,
            planes.len(),
            false,
            false,
        )?;
    }
    Ok((refined_planes, refined_samples))
}

#[allow(clippy::too_many_arguments)]
fn refine_edge<E: HyperplaneSampler>(
    sampler: &mut E,
    from_plane: &Hyperplane,
    from_sample: &PlaneSample,
    to_plane: &Hyperplane,
    to_sample: &PlaneSample,
    quadrature_budget: f64,
    config: &AdaptiveOhTstConfig,
    count: usize,
    evaluations: &mut usize,
    inserted: &mut usize,
    seed_salt: &mut u64,
    output_planes: &mut Vec<Hyperplane>,
    output_samples: &mut Vec<PlaneSample>,
    initial_planes: usize,
    from_is_root: bool,
    to_is_root: bool,
) -> Result<(), SaddleError> {
    let path = PlanePath::legs(&[from_plane.clone(), to_plane.clone()])?;
    let from_direction = path.tangent(0, 0.0)?;
    let to_direction = path.tangent(0, 1.0)?;
    let (from, from_error) = from_sample.directional(from_direction.view())?;
    let (to, to_error) = to_sample.directional(to_direction.view())?;
    let family_error = ADAPTIVE_OHTST_TARGET_LOG_ERROR / config.max_planes as f64;
    let from_resolution = from_error / family_error.sqrt();
    let to_resolution = to_error / family_error.sqrt();
    let resolved_rise = !from_is_root && from > from_resolution;
    let resolved_fall = !to_is_root && to < -to_resolution;
    let root = if resolved_rise && resolved_fall {
        let denominator = from - to;
        if denominator > 0.0 {
            Some(from / denominator)
        } else {
            return Err(SaddleError::Solver(
                "adaptive OH-TST force root has a non-positive secant".into(),
            ));
        }
    } else {
        None
    };
    let (fraction, force_root) = match root {
        Some(fraction) => (fraction, true),
        None => (0.5, false),
    };
    let candidate = interpolate_plane(from_plane, to_plane, fraction)?;
    let middle = sample_one(
        sampler,
        &candidate,
        config,
        count,
        *seed_salt,
        false,
        ADAPTIVE_OHTST_TARGET_LOG_ERROR,
    )?;
    *seed_salt = seed_salt.wrapping_add(1);
    *evaluations = checked_sum(*evaluations, middle.evaluations)?;
    let root_resolved = if force_root {
        let (value, error) = middle.directional(path.tangent(0, fraction)?.view())?;
        let curvature = (to - from).abs();
        curvature > 0.0 && (value.abs() + error).powi(2) / (2.0 * curvature) <= quadrature_budget
    } else {
        false
    };

    let should_insert = if force_root {
        true
    } else {
        let middle_path =
            PlanePath::legs(&[from_plane.clone(), candidate.clone(), to_plane.clone()])?;
        let (from_half, from_half_error) =
            from_sample.directional(middle_path.tangent(0, 0.0)?.view())?;
        let (middle_left, middle_left_error) =
            middle.directional(middle_path.tangent(0, 1.0)?.view())?;
        let (middle_right, middle_right_error) =
            middle.directional(middle_path.tangent(1, 0.0)?.view())?;
        let (to_half, to_half_error) =
            to_sample.directional(middle_path.tangent(1, 1.0)?.view())?;
        let coarse = 0.5 * (from + to);
        let fine = 0.5 * (from_half + middle_left + middle_right + to_half);
        let sampled = 0.5
            * from_half_error
                .hypot(middle_left_error)
                .hypot(middle_right_error)
                .hypot(to_half_error);
        (fine - coarse).abs() > sampled / family_error.sqrt() + quadrature_budget
    };
    if !should_insert {
        output_planes.push(to_plane.clone());
        output_samples.push(to_sample.clone());
        return Ok(());
    }
    *inserted = inserted.checked_add(1).ok_or_else(|| {
        SaddleError::Invalid("adaptive OH-TST inserted-plane count overflows".into())
    })?;
    if initial_planes.saturating_add(*inserted) > config.max_planes {
        return Err(SaddleError::Solver(
            "adaptive OH-TST reached the plane cap before resolving its profile".into(),
        ));
    }
    let child_budget = 0.5 * quadrature_budget;
    refine_edge(
        sampler,
        from_plane,
        from_sample,
        &candidate,
        &middle,
        child_budget,
        config,
        count,
        evaluations,
        inserted,
        seed_salt,
        output_planes,
        output_samples,
        initial_planes,
        from_is_root,
        root_resolved,
    )?;
    refine_edge(
        sampler,
        &candidate,
        &middle,
        to_plane,
        to_sample,
        child_budget,
        config,
        count,
        evaluations,
        inserted,
        seed_salt,
        output_planes,
        output_samples,
        initial_planes,
        root_resolved,
        to_is_root,
    )
}

fn coarse_indices(count: usize) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..count).step_by(2).collect();
    if indices.last().copied() != Some(count - 1) {
        indices.push(count - 1);
    }
    indices
}

/// `|I_h - I_2h|` on the nodes already in hand, with a conservative standard error.
fn quadrature_gap(
    planes: &[Hyperplane],
    samples: &[PlaneSample],
) -> Result<(f64, f64), SaddleError> {
    let full = integrate_barrier(planes, samples)?;
    let indices = coarse_indices(planes.len());
    if indices.len() < 2 || indices.len() == planes.len() {
        return Ok((0.0, full.1));
    }
    let coarse_planes: Vec<Hyperplane> =
        indices.iter().map(|&index| planes[index].clone()).collect();
    let coarse_samples: Vec<PlaneSample> = indices
        .iter()
        .map(|&index| samples[index].clone())
        .collect();
    let coarse = integrate_barrier(&coarse_planes, &coarse_samples)?;
    Ok(((full.0 - coarse.0).abs(), full.1.hypot(coarse.1)))
}

fn integrate_barrier(
    planes: &[Hyperplane],
    samples: &[PlaneSample],
) -> Result<(f64, f64), SaddleError> {
    let path = PlanePath::legs(planes)?;
    let scales: Vec<f64> = samples
        .iter()
        .map(rotation_scale)
        .collect::<Result<_, _>>()?;
    let lengths: Vec<f64> = (0..planes.len() - 1)
        .map(|edge| {
            generalized_distance(
                &planes[edge],
                &planes[edge + 1],
                scales[edge],
                scales[edge + 1],
            )
        })
        .collect::<Result<_, _>>()?;
    let mut best = 0.0;
    let mut best_error = 0.0;
    for end in 1..planes.len() {
        let mut directions = vec![Array1::zeros(2 * path.dim()); end + 1];
        let paired_end = end - end % 2;
        for first_edge in (0..paired_end).step_by(2) {
            add_simpson_pair(&path, &lengths, first_edge, &mut directions)?;
        }
        if end % 2 == 1 {
            directions[end - 1] += &(&path.tangent(end - 1, 0.0)? * 0.5);
            directions[end] += &(&path.tangent(end - 1, 1.0)? * 0.5);
        }
        let mut value = 0.0;
        let mut variance = 0.0;
        for (sample, direction) in samples.iter().zip(&directions) {
            let (contribution, error) = sample.directional(direction.view())?;
            value += contribution;
            variance += error * error;
        }
        if value >= best {
            best = value;
            best_error = variance.sqrt();
        }
    }
    if best.is_finite() && best_error.is_finite() {
        Ok((best, best_error))
    } else {
        Err(SaddleError::NonFinite(
            "adaptive OH-TST doubled-grid integral",
        ))
    }
}

/// Barrier change when each interior plane is moved by its unresolved Newton step.
fn orientation_sensitivity(
    planes: &[Hyperplane],
    samples: &[PlaneSample],
    barrier: f64,
) -> Result<f64, SaddleError> {
    if planes.len() < 3 {
        return Ok(0.0);
    }
    let scales: Vec<f64> = samples
        .iter()
        .map(rotation_scale)
        .collect::<Result<_, _>>()?;
    let mut smoothed = planes.to_vec();
    for index in 1..planes.len() - 1 {
        if let Ok(mid) = interpolate_plane(&planes[index - 1], &planes[index + 1], 0.5) {
            // Keep the plane point; only the orientation is the geometric residual.
            if let Ok(oriented) =
                Hyperplane::new(planes[index].point().to_owned(), mid.normal().to_owned())
            {
                smoothed[index] = oriented;
            }
        }
    }
    let (smoothed_barrier, _) = integrate_barrier(&smoothed, samples)?;
    let mut accumulated = (smoothed_barrier - barrier).abs();
    for index in 1..planes.len() - 1 {
        let plane = &planes[index];
        let scale = scales[index];
        let gradient = scaled_gradient(&samples[index], scale);
        let left = generalized_distance(&planes[index - 1], plane, scales[index - 1], scale)?;
        let right = generalized_distance(plane, &planes[index + 1], scale, scales[index + 1])?;
        let translation_scale = 0.5 * (left + right);
        let thermal = samples[index].thermal;
        let translation_curvature = thermal / translation_scale.powi(2);
        let rotation_curvature = thermal / scale.powi(2);
        if !(translation_curvature.is_finite()
            && translation_curvature > 0.0
            && rotation_curvature.is_finite()
            && rotation_curvature > 0.0)
        {
            return Err(SaddleError::NonFinite(
                "adaptive OH-TST orientation curvature",
            ));
        }
        let tangent = string_tangent(&planes[index - 1], plane, &planes[index + 1], scale);
        let tangent_norm2 = tangent.dot(&tangent);
        let mut gradient = gradient;
        if tangent_norm2 > 0.0 {
            let along = gradient.dot(&tangent) / tangent_norm2;
            gradient.scaled_add(-along, &tangent);
        }
        let mut displacement = gradient;
        displacement[0] /= translation_curvature;
        displacement
            .slice_mut(s![1..])
            .mapv_inplace(|value| value / rotation_curvature);
        let unconstrained = displacement.dot(&displacement).sqrt();
        let trust = 0.5 * left.min(right);
        if unconstrained > trust && unconstrained > 0.0 {
            displacement *= trust / unconstrained;
        }
        let mut point = plane.point().to_owned();
        point.scaled_add(displacement[0], &plane.normal());
        let mut normal = plane.normal().to_owned();
        normal.scaled_add(1.0 / scale, &displacement.slice(s![1..]));
        let normal_length = normal.dot(&normal).sqrt();
        if !(normal_length.is_finite() && normal_length > 0.0) {
            return Err(SaddleError::Invalid(
                "adaptive OH-TST orientation probe erased a plane normal".into(),
            ));
        }
        normal /= normal_length;
        // A normal dot is 2D-1 flops. Skip a probe that does not move the normal.
        let roundoff = gamma_n(2 * plane.dim() + 1)?;
        if 1.0 - normal.dot(&plane.normal()) <= roundoff
            && plane.distance(point.view()).abs() <= roundoff * translation_scale
        {
            continue;
        }
        let mut probed = planes.to_vec();
        probed[index] = Hyperplane::new(point, normal)?;
        let (probed_barrier, _) = integrate_barrier(&probed, samples)?;
        let change = (probed_barrier - barrier).abs();
        accumulated = accumulated.hypot(change);
    }
    if accumulated.is_finite() {
        Ok(accumulated)
    } else {
        Err(SaddleError::NonFinite(
            "adaptive OH-TST orientation sensitivity",
        ))
    }
}

/// `|ΔA_path - ΔA_offset|` on the reactant-to-barrier prefix when those two
/// planes share an orientation, so the rotational change cancels and the
/// difference is quadrature. The drop past the barrier is not part of it.
fn translation_quadrature_gap(
    planes: &[Hyperplane],
    samples: &[PlaneSample],
    barrier: f64,
) -> Result<f64, SaddleError> {
    let first = planes
        .first()
        .ok_or_else(|| SaddleError::Invalid("adaptive OH-TST profile has no planes".into()))?;
    let last = planes
        .last()
        .ok_or_else(|| SaddleError::Invalid("adaptive OH-TST profile has no planes".into()))?;
    let orientation = gamma_n(2 * first.dim() + 1)?;
    if 1.0 - first.normal().dot(&last.normal()) > orientation {
        return Ok(0.0);
    }
    let offset = offset_simpson(planes, samples)?;
    Ok((barrier - offset).abs())
}

fn offset_simpson(planes: &[Hyperplane], samples: &[PlaneSample]) -> Result<f64, SaddleError> {
    let coordinate: Vec<f64> = planes
        .iter()
        .map(|plane| plane.normal().dot(&plane.point()))
        .collect();
    let force: Vec<f64> = samples.iter().map(|sample| -sample.normal_force).collect();
    let mut weight = vec![0.0; planes.len()];
    let end = planes.len() - 1;
    let paired = end - end % 2;
    for first in (0..paired).step_by(2) {
        let nodes = [
            coordinate[first],
            coordinate[first + 1],
            coordinate[first + 2],
        ];
        let span = nodes[2] - nodes[0];
        if span == 0.0 {
            continue;
        }
        let shifted = [0.0, nodes[1] - nodes[0], span];
        let mut local = [0.0; 3];
        for index in 0..3 {
            let others: Vec<f64> = shifted
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(other, value)| (other != index).then_some(value))
                .collect();
            let denominator = (shifted[index] - others[0]) * (shifted[index] - others[1]);
            if denominator == 0.0 {
                return Err(SaddleError::Invalid(
                    "adaptive OH-TST offset nodes coincide".into(),
                ));
            }
            let primitive = span.powi(3) / 3.0 - 0.5 * (others[0] + others[1]) * span.powi(2)
                + others[0] * others[1] * span;
            local[index] = primitive / denominator;
        }
        weight[first] += local[0];
        weight[first + 1] += local[1];
        weight[first + 2] += local[2];
    }
    if end % 2 == 1 {
        let step = coordinate[end] - coordinate[end - 1];
        weight[end - 1] += 0.5 * step;
        weight[end] += 0.5 * step;
    }
    let value: f64 = force
        .iter()
        .zip(&weight)
        .map(|(sample, panel)| sample * panel)
        .sum();
    if value.is_finite() {
        Ok(value)
    } else {
        Err(SaddleError::NonFinite("adaptive OH-TST offset integral"))
    }
}

fn resolve_measured_quadrature<E: HyperplaneSampler>(
    sampler: &mut E,
    mut planes: Vec<Hyperplane>,
    mut samples: Vec<PlaneSample>,
    config: &AdaptiveOhTstConfig,
    count: usize,
    evaluations: &mut usize,
) -> Result<(Vec<Hyperplane>, Vec<PlaneSample>), SaddleError> {
    let target = config.sampler.thermal * ADAPTIVE_OHTST_TARGET_LOG_ERROR;
    let multiple = (1.0 / ADAPTIVE_OHTST_TARGET_LOG_ERROR).sqrt();
    let mut salt = count as u64;
    loop {
        let (gap, gap_error) = quadrature_gap(&planes, &samples)?;
        if gap <= target || gap <= multiple * gap_error || planes.len() * 2 - 1 > config.max_planes
        {
            return Ok((planes, samples));
        }
        let mut fine_planes = Vec::with_capacity(planes.len() * 2 - 1);
        let mut fine_samples = Vec::with_capacity(planes.len() * 2 - 1);
        fine_planes.push(planes[0].clone());
        fine_samples.push(samples[0].clone());
        for edge in 0..planes.len() - 1 {
            let midpoint = interpolate_plane(&planes[edge], &planes[edge + 1], 0.5)?;
            salt = salt.wrapping_add(1);
            let sample = sample_one(
                sampler,
                &midpoint,
                config,
                count,
                salt,
                false,
                ADAPTIVE_OHTST_TARGET_LOG_ERROR,
            )?;
            *evaluations = checked_sum(*evaluations, sample.evaluations)?;
            fine_planes.push(midpoint);
            fine_samples.push(sample);
            fine_planes.push(planes[edge + 1].clone());
            fine_samples.push(samples[edge + 1].clone());
        }
        let (coarse_barrier, coarse_error) = integrate_barrier(&planes, &samples)?;
        let (fine_barrier, fine_error) = integrate_barrier(&fine_planes, &fine_samples)?;
        let gap = (fine_barrier - coarse_barrier).abs();
        let gap_error = coarse_error.hypot(fine_error);
        planes = fine_planes;
        samples = fine_samples;
        if gap <= target || gap <= multiple * gap_error {
            return Ok((planes, samples));
        }
    }
}

fn profile_from_samples(
    planes: &[Hyperplane],
    samples: &[PlaneSample],
    count: usize,
) -> Result<AdaptiveOhTstProfile, SaddleError> {
    let path = PlanePath::legs(planes)?;
    let scales: Vec<f64> = samples
        .iter()
        .map(rotation_scale)
        .collect::<Result<_, _>>()?;
    let lengths: Vec<f64> = (0..planes.len() - 1)
        .map(|edge| {
            generalized_distance(
                &planes[edge],
                &planes[edge + 1],
                scales[edge],
                scales[edge + 1],
            )
        })
        .collect::<Result<_, _>>()?;
    let mut free_energy = vec![0.0; planes.len()];
    let mut free_energy_error = vec![0.0; planes.len()];
    for end in 1..planes.len() {
        let mut directions = vec![Array1::zeros(2 * path.dim()); end + 1];
        let paired_end = end - end % 2;
        for first_edge in (0..paired_end).step_by(2) {
            add_simpson_pair(&path, &lengths, first_edge, &mut directions)?;
        }
        if end % 2 == 1 {
            directions[end - 1] += &(&path.tangent(end - 1, 0.0)? * 0.5);
            directions[end] += &(&path.tangent(end - 1, 1.0)? * 0.5);
        }
        let mut value = 0.0;
        let mut variance = 0.0;
        for (sample, direction) in samples.iter().zip(&directions) {
            let (contribution, error) = sample.directional(direction.view())?;
            value += contribution;
            variance += error * error;
        }
        free_energy[end] = value;
        free_energy_error[end] = variance.sqrt();
    }
    let (dividing, barrier) = free_energy
        .iter()
        .copied()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .ok_or_else(|| SaddleError::Invalid("adaptive OH-TST profile is empty".into()))?;
    let sampled_barrier_error = free_energy_error[dividing];
    let gap = quadrature_gap(planes, samples)?.0;
    // The offset rule is a quadrature check only through the barrier. The
    // free-energy drop past an interior dividing plane is not a gap.
    let offset_gap =
        translation_quadrature_gap(&planes[..=dividing], &samples[..=dividing], barrier)?;
    let orientation = orientation_sensitivity(planes, samples, barrier)?;
    // One packed dot per plane (two length-D dots: 4D flops) plus an addition
    // into the integral. γ times kT is the roundoff floor of that sum.
    let integral_operations = planes.len() * (4 * planes[0].dim() + 1);
    let roundoff_floor = gamma_n(integral_operations)? * samples[0].thermal;
    let geometry_error = orientation.max(roundoff_floor);
    // The generalized-arc Simpson and the offset Simpson are two coordinates
    // for the same samples. Their difference is the quadrature discrepancy;
    // the doubled-grid gap is the truncation piece.
    let quadrature_error = gap.max(offset_gap).max(roundoff_floor);
    let barrier_error = sampled_barrier_error
        .hypot(geometry_error)
        .hypot(quadrature_error);
    if !(barrier.is_finite()
        && barrier_error.is_finite()
        && free_energy.iter().all(|value| value.is_finite())
        && free_energy_error.iter().all(|value| value.is_finite()))
    {
        return Err(SaddleError::NonFinite("adaptive OH-TST profile"));
    }
    Ok(AdaptiveOhTstProfile {
        planes: planes.to_vec(),
        free_energy,
        free_energy_error,
        barrier,
        barrier_error,
        sampled_barrier_error,
        geometry_error,
        quadrature_error,
        dividing,
        samples_per_plane: vec![count; planes.len()],
        plane_samples: samples.to_vec(),
    })
}

fn add_simpson_pair(
    path: &PlanePath,
    lengths: &[f64],
    first_edge: usize,
    directions: &mut [Array1<f64>],
) -> Result<(), SaddleError> {
    let left = lengths[first_edge];
    let right = lengths[first_edge + 1];
    let total = left + right;
    let coordinates = [0.0, left, total];
    let mut weights = [0.0; 3];
    for index in 0..3 {
        let others: Vec<f64> = coordinates
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(other, value)| (other != index).then_some(value))
            .collect();
        let denominator = (coordinates[index] - others[0]) * (coordinates[index] - others[1]);
        if denominator == 0.0 {
            return Err(SaddleError::Invalid(
                "adaptive OH-TST Simpson nodes coincide".into(),
            ));
        }
        let primitive = total.powi(3) / 3.0 - 0.5 * (others[0] + others[1]) * total.powi(2)
            + others[0] * others[1] * total;
        weights[index] = primitive / denominator;
    }
    directions[first_edge] += &(&path.tangent(first_edge, 0.0)? * (weights[0] / left));
    let incoming = path.tangent(first_edge, 1.0)? / left;
    let outgoing = path.tangent(first_edge + 1, 0.0)? / right;
    directions[first_edge + 1] += &((&incoming + &outgoing) * (0.5 * weights[1]));
    directions[first_edge + 2] += &(&path.tangent(first_edge + 1, 1.0)? * (weights[2] / right));
    Ok(())
}

fn rate_from_profile<D: PlaneDensity + ?Sized>(
    density: &D,
    profile: &AdaptiveOhTstProfile,
    thermal: f64,
) -> Result<OhTstRate, SaddleError> {
    // Subtract the two temperatures, take the absolute value, and compare.
    let temperature_roundoff = gamma_n(3)? * thermal.abs().max(density.thermal().abs());
    if (density.thermal() - thermal).abs() > temperature_roundoff {
        return Err(SaddleError::Invalid(
            "the reactant density and adaptive string have different temperatures".into(),
        ));
    }
    let first = profile
        .planes
        .first()
        .ok_or_else(|| SaddleError::Invalid("adaptive OH-TST profile has no planes".into()))?;
    let point_scale = 1.0 + first.point().dot(&first.point()).sqrt();
    let plane_roundoff = gamma_n(2 * first.dim() + 4)? * point_scale;
    let normal_roundoff = gamma_n(2 * first.dim() + 1)?;
    if first.dim() != density.plane().dim()
        || first.normal().dot(&density.plane().normal()) < 1.0 - normal_roundoff
        || first.distance(density.plane().point()).abs() > plane_roundoff
    {
        return Err(SaddleError::Invalid(
            "the adaptive string does not start at the reactant-density plane".into(),
        ));
    }
    let (value, value_error) = density.density();
    if !(value.is_finite()
        && value > 0.0
        && value_error.is_finite()
        && value_error >= 0.0
        && profile.barrier.is_finite()
        && profile.barrier_error.is_finite()
        && profile.barrier_error >= 0.0)
    {
        return Err(SaddleError::Invalid(
            "adaptive OH-TST rate inputs must be finite and non-negative".into(),
        ));
    }
    rate_from_parts(
        thermal,
        value,
        value_error,
        profile.barrier,
        profile.barrier_error,
    )
}

fn checked_sum(left: usize, right: usize) -> Result<usize, SaddleError> {
    left.checked_add(right)
        .ok_or_else(|| SaddleError::Invalid("adaptive OH-TST evaluation count overflows".into()))
}
