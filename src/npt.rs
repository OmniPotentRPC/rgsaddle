//! Isotropic NPT sampling of a crystal lattice: Langevin dynamics with a
//! Berendsen barostat.
//!
//! The force evaluator supplies a tension-positive Cauchy stress.  The
//! reported pressure is therefore minus one third of its trace.

use ndarray::{Array1, ArrayView1};

use crate::{BOLTZMANN_EV_PER_K, Cell, SaddleError, angular_scale_rad_s};

/// Controls an isotropic NPT lattice run.
///
/// Non-exhaustive: start from [`NptConfig::default`] and chain the
/// `with_*` setters.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct NptConfig {
    /// Number of conventional cells along each lattice vector.
    pub repeats: usize,
    /// Initial number of integration steps.
    pub steps: usize,
    /// Largest number of integration steps after extensions.
    pub maximum_steps: usize,
    /// Number of steps added when the pressure criterion is not met.
    pub extension_steps: usize,
    /// Steps discarded before statistics are accumulated.
    pub equilibration_steps: usize,
    /// Number of production steps in one reported block. It sets
    /// [`NptReport::blocks`] and the two-block floor on the production
    /// length. The standard errors come from blocking the per-step series
    /// at every length, so they do not depend on this value.
    pub block_steps: usize,
    /// Velocity-Verlet time step in seconds.
    pub timestep_s: f64,
    /// Langevin friction in inverse seconds.
    pub langevin_gamma_per_s: f64,
    /// Berendsen pressure-coupling time in seconds.
    pub berendsen_tau_s: f64,
    /// Isotropic compressibility in inverse eV per cubic Angstrom.
    pub compressibility_per_ev_a3: f64,
    /// Target thermodynamic pressure in eV per cubic Angstrom.
    pub target_pressure_ev_a3: f64,
    /// Seed for the deterministic thermostat stream. Every seed, including
    /// consecutive ones, starts an unrelated stream.
    pub seed: u64,
}

with_setters!(NptConfig {
    with_repeats => repeats: val usize,
    with_steps => steps: val usize,
    with_maximum_steps => maximum_steps: val usize,
    with_extension_steps => extension_steps: val usize,
    with_equilibration_steps => equilibration_steps: val usize,
    with_block_steps => block_steps: val usize,
    with_timestep_s => timestep_s: val f64,
    with_langevin_gamma_per_s => langevin_gamma_per_s: val f64,
    with_berendsen_tau_s => berendsen_tau_s: val f64,
    with_compressibility_per_ev_a3 => compressibility_per_ev_a3: val f64,
    with_target_pressure_ev_a3 => target_pressure_ev_a3: val f64,
    with_seed => seed: val u64,
});

impl Default for NptConfig {
    fn default() -> Self {
        Self {
            repeats: 6,
            steps: 4_000,
            maximum_steps: 4_000,
            extension_steps: 1_000,
            equilibration_steps: 2_000,
            block_steps: 250,
            timestep_s: 2.0e-15,
            langevin_gamma_per_s: 1.0e12,
            berendsen_tau_s: 0.5e-12,
            compressibility_per_ev_a3: 1.0 / 0.87,
            target_pressure_ev_a3: 0.0,
            seed: 11,
        }
    }
}

/// Result of an NPT lattice run.
///
/// The standard errors are those of the mean of a correlated series, found
/// by blocking (Flyvbjerg and Petersen, J. Chem. Phys. 91, 461 (1989),
/// doi:10.1063/1.457480). Each is the largest estimate over the blocking
/// levels that keep at least sixteen blocks, so it is on the conservative
/// side for uncorrelated data. A `*_error_resolved` flag that is `false`
/// says the estimate was still rising at the longest block length the run
/// allows: the correlation time is not short against the run, and the
/// standard error is then a lower bound.
///
/// Non-exhaustive: read the fields.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct NptReport {
    /// Total number of integration steps.
    pub steps: usize,
    /// Number of complete production blocks of [`NptConfig::block_steps`].
    pub blocks: usize,
    /// Mean conventional lattice parameter in Angstrom.
    pub mean_lattice_a: f64,
    /// Standard error of the mean lattice parameter in Angstrom.
    pub lattice_standard_error: f64,
    /// Whether the lattice standard error reached a blocking plateau.
    pub lattice_error_resolved: bool,
    /// Mean thermodynamic pressure in eV per cubic Angstrom.
    pub mean_pressure: f64,
    /// Standard error of the mean pressure in eV per cubic Angstrom.
    pub pressure_standard_error: f64,
    /// Whether the pressure standard error reached a blocking plateau.
    pub pressure_error_resolved: bool,
    /// Indicative test that the pressure sits at the target: the pressure
    /// standard error is resolved and `|mean pressure - target| <= 2 SE`,
    /// where a series with no spread passes only at exactly zero residual.
    ///
    /// This is a screen, not a proof of equilibrium. It cannot see drift
    /// slower than the run, and a two standard error window misses a real
    /// bias about one time in twenty. Treat a `true` as "no evidence of
    /// bias at this run length", not as a measured zero pressure.
    pub pressure_converged: bool,
}

/// Seed expansion of Steele, Lea and Flood (the SplitMix64 generator).
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// xoshiro256** (Blackman and Vigna) with SplitMix64 seeding, the
/// initialisation its authors prescribe. Nearby seeds give unrelated
/// states, so no warm-up is needed.
#[derive(Clone, Copy)]
struct Rng {
    state: [u64; 4],
}

impl Rng {
    fn new(seed: u64) -> Self {
        let mut mixer = seed;
        let mut state = [0_u64; 4];
        for word in &mut state {
            *word = splitmix64(&mut mixer);
        }
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        let s = &mut self.state;
        let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform on `[0, 1)` with 53 random bits.
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64))
    }

    fn normal(&mut self) -> f64 {
        let u1 = 1.0 - self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

fn zero_mean(values: &mut Array1<f64>) {
    let atoms = values.len() / 3;
    let mut mean = [0.0; 3];
    for atom in 0..atoms {
        for c in 0..3 {
            mean[c] += values[3 * atom + c] / atoms as f64;
        }
    }
    for atom in 0..atoms {
        for c in 0..3 {
            values[3 * atom + c] -= mean[c];
        }
    }
}

fn determinant(cell: Cell) -> f64 {
    let h = cell.0;
    h[0][0] * (h[1][1] * h[2][2] - h[1][2] * h[2][1])
        - h[0][1] * (h[1][0] * h[2][2] - h[1][2] * h[2][0])
        + h[0][2] * (h[1][0] * h[2][1] - h[1][1] * h[2][0])
}

/// Fewest blocks a blocking level may keep and still be trusted. The
/// relative error of a standard error from `n` blocks is
/// `1 / sqrt(2 (n - 1))`, which is 18 per cent at sixteen.
const MIN_BLOCKS: usize = 16;

struct Blocked {
    mean: f64,
    standard_error: f64,
    resolved: bool,
}

fn mean_and_standard_error(values: &[f64]) -> (f64, f64) {
    let mut mean = 0.0;
    for (index, value) in values.iter().enumerate() {
        mean += (value - mean) / (index + 1) as f64;
    }
    if values.len() < 2 {
        return (mean, 0.0);
    }
    let spread = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>();
    let count = values.len() as f64;
    (mean, (spread / ((count - 1.0) * count)).sqrt())
}

/// Mean of a correlated series and the standard error of that mean by
/// blocking (Flyvbjerg and Petersen, 1989).
///
/// Averaging neighbouring pairs doubles the block length. While blocks are
/// shorter than the correlation time the standard error of the block means
/// rises with every doubling, and it stops rising once blocks are
/// independent. The estimate is the largest value over the levels that keep
/// at least [`MIN_BLOCKS`] blocks. It is `resolved` when the last doubling
/// raised it by no more than its own sampling error, and a series with no
/// spread is resolved with a standard error of zero. A series too short to
/// block twice cannot show a plateau and is not resolved.
fn blocked_statistics(values: &[f64]) -> Blocked {
    let mut level = values.to_vec();
    let mut estimates: Vec<(f64, usize)> = Vec::new();
    let mut mean = 0.0;
    while level.len() >= MIN_BLOCKS || (estimates.is_empty() && level.len() >= 2) {
        let (level_mean, standard_error) = mean_and_standard_error(&level);
        if estimates.is_empty() {
            mean = level_mean;
        }
        estimates.push((standard_error, level.len()));
        level = (0..level.len() / 2)
            .map(|pair| 0.5 * (level[2 * pair] + level[2 * pair + 1]))
            .collect();
    }
    if estimates.is_empty() {
        let (mean, _) = mean_and_standard_error(values);
        return Blocked {
            mean,
            standard_error: 0.0,
            resolved: false,
        };
    }
    let standard_error = estimates
        .iter()
        .fold(0.0_f64, |worst, &(estimate, _)| worst.max(estimate));
    let resolved = if estimates[0].0 == 0.0 {
        true
    } else if let [.., (previous, _), (last, blocks)] = estimates[..] {
        last - previous <= last / (2.0 * (blocks as f64 - 1.0)).sqrt()
    } else {
        false
    };
    Blocked {
        mean,
        standard_error,
        resolved,
    }
}

fn summarize(lattice: &[f64], pressure: &[f64], config: NptConfig, steps: usize) -> NptReport {
    let lattice = blocked_statistics(lattice);
    let pressure_stats = blocked_statistics(pressure);
    let residual = (pressure_stats.mean - config.target_pressure_ev_a3).abs();
    let within = if pressure_stats.standard_error == 0.0 {
        residual == 0.0
    } else {
        residual <= 2.0 * pressure_stats.standard_error
    };
    NptReport {
        steps,
        blocks: pressure.len() / config.block_steps,
        mean_lattice_a: lattice.mean,
        lattice_standard_error: lattice.standard_error,
        lattice_error_resolved: lattice.resolved,
        mean_pressure: pressure_stats.mean,
        pressure_standard_error: pressure_stats.standard_error,
        pressure_error_resolved: pressure_stats.resolved,
        pressure_converged: pressure_stats.resolved && within,
    }
}

fn extended_check(current: usize, extension: usize, maximum: usize) -> usize {
    current.saturating_add(extension).min(maximum)
}

/// Run isotropic NPT dynamics and return block standard errors.
///
/// `evaluate` returns `(energy, gradient, tension-positive stress)` for
/// Cartesian `positions` and the row-major `cell`.  Energy is accepted for
/// evaluator compatibility but is not used by the integrator.  A run extends
/// by `extension_steps` until the configured target-pressure rule is met or
/// `maximum_steps` is reached.
pub fn npt_lattice<F>(
    positions: &mut Array1<f64>,
    cell: &mut Cell,
    temperature_k: f64,
    mass_amu: f64,
    config: NptConfig,
    mut evaluate: F,
) -> Result<NptReport, SaddleError>
where
    F: FnMut(ArrayView1<'_, f64>, Cell) -> Result<(f64, Array1<f64>, [[f64; 3]; 3]), SaddleError>,
{
    let production = config.steps.saturating_sub(config.equilibration_steps);
    if positions.iter().any(|value| !value.is_finite()) {
        return Err(SaddleError::NonFinite("NPT initial positions"));
    }
    if cell.0.iter().flatten().any(|value| !value.is_finite()) {
        return Err(SaddleError::NonFinite("NPT initial cell"));
    }
    let initial_volume = determinant(*cell).abs();
    if !(initial_volume.is_finite() && initial_volume > 0.0) {
        return Err(SaddleError::Invalid(
            "NPT lattice sampling needs a positive finite cell volume".into(),
        ));
    }
    if positions.is_empty()
        || !positions.len().is_multiple_of(3)
        || config.repeats == 0
        || config.steps > config.maximum_steps
        || config.extension_steps == 0
        || config.block_steps == 0
        || production / config.block_steps < 2
        || !(temperature_k >= 0.0 && temperature_k.is_finite())
        || !(mass_amu > 0.0 && mass_amu.is_finite())
        || !(config.timestep_s.is_finite() && config.timestep_s > 0.0)
        || !(config.langevin_gamma_per_s.is_finite() && config.langevin_gamma_per_s > 0.0)
        || !(config.berendsen_tau_s.is_finite() && config.berendsen_tau_s > 0.0)
        || !(config.compressibility_per_ev_a3.is_finite() && config.compressibility_per_ev_a3 > 0.0)
        || !config.target_pressure_ev_a3.is_finite()
    {
        return Err(SaddleError::Invalid(
            "NPT lattice sampling needs finite physical controls and at least two production blocks"
                .into(),
        ));
    }

    let thermal = BOLTZMANN_EV_PER_K * temperature_k;
    let omega = angular_scale_rad_s();
    let dt = config.timestep_s * omega;
    let gamma = config.langevin_gamma_per_s / omega;
    let tau = config.berendsen_tau_s * omega;
    let mut rng = Rng::new(config.seed);
    let mut velocity = Array1::zeros(positions.len());
    let sigma = (thermal / mass_amu).sqrt();
    for value in velocity.iter_mut() {
        *value = sigma * rng.normal();
    }
    zero_mean(&mut velocity);

    let mut lattice_samples = Vec::with_capacity(
        config
            .maximum_steps
            .saturating_sub(config.equilibration_steps),
    );
    let mut pressure_samples = Vec::with_capacity(lattice_samples.capacity());
    let mut next_check = config.steps;
    let mut report = None;

    for step in 0..config.maximum_steps {
        let (_, gradient, _) = evaluate(positions.view(), *cell)?;
        if gradient.len() != positions.len() || gradient.iter().any(|value| !value.is_finite()) {
            return Err(SaddleError::NonFinite("NPT gradient"));
        }
        velocity = &velocity + &(&gradient * (-0.5 * dt / mass_amu));
        *positions = &*positions + &(&velocity * (0.5 * dt));

        let damp = (-gamma * dt).exp();
        let noise = (thermal / mass_amu * (1.0 - damp * damp)).sqrt();
        for value in velocity.iter_mut() {
            *value = damp * *value + noise * rng.normal();
        }
        zero_mean(&mut velocity);
        *positions = &*positions + &(&velocity * (0.5 * dt));

        let (_, gradient, stress) = evaluate(positions.view(), *cell)?;
        if gradient.len() != positions.len()
            || gradient.iter().any(|value| !value.is_finite())
            || stress.iter().flatten().any(|value| !value.is_finite())
        {
            return Err(SaddleError::NonFinite("NPT force or stress"));
        }
        velocity = &velocity + &(&gradient * (-0.5 * dt / mass_amu));

        let configurational_stress = (stress[0][0] + stress[1][1] + stress[2][2]) / 3.0;
        let kinetic_energy = velocity
            .iter()
            .map(|value| 0.5 * mass_amu * value * value)
            .sum::<f64>();
        let volume = determinant(*cell).abs();
        let kinetic_pressure = 2.0 * kinetic_energy / (3.0 * volume);
        let hydrostatic_stress = configurational_stress - kinetic_pressure;
        let drive = hydrostatic_stress + config.target_pressure_ev_a3;
        let volume_factor =
            (1.0 - config.compressibility_per_ev_a3 * dt / tau * drive).clamp(0.995, 1.005);
        let length_factor = volume_factor.cbrt();
        *positions *= length_factor;
        for row in &mut cell.0 {
            for value in row {
                *value *= length_factor;
            }
        }

        let completed = step + 1;
        if completed > config.equilibration_steps {
            lattice_samples.push(determinant(*cell).abs().cbrt() / config.repeats as f64);
            pressure_samples.push(-hydrostatic_stress);
        }
        if completed == next_check {
            let current = summarize(&lattice_samples, &pressure_samples, config, completed);
            if current.pressure_converged || completed == config.maximum_steps {
                return Ok(current);
            }
            report = Some(current);
            next_check = extended_check(next_check, config.extension_steps, config.maximum_steps);
        }
    }
    report.ok_or_else(|| SaddleError::Invalid("NPT run produced no report".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input_error(positions: &mut Array1<f64>, cell: &mut Cell, config: NptConfig) -> SaddleError {
        npt_lattice(positions, cell, 300.0, 1.0, config, |_, _| {
            panic!("invalid input reached the evaluator")
        })
        .unwrap_err()
    }

    #[test]
    fn npt_rejects_nonfinite_controls_cell_and_positions() {
        let position = Array1::zeros(3);
        let cell = Cell([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        for config in [
            NptConfig::default().with_langevin_gamma_per_s(f64::INFINITY),
            NptConfig::default().with_berendsen_tau_s(f64::INFINITY),
            NptConfig::default().with_compressibility_per_ev_a3(f64::INFINITY),
            NptConfig::default().with_target_pressure_ev_a3(f64::INFINITY),
        ] {
            assert!(matches!(
                input_error(&mut position.clone(), &mut cell.clone(), config),
                SaddleError::Invalid(_)
            ));
        }

        let mut zero_cell = Cell([[0.0; 3]; 3]);
        assert!(matches!(
            input_error(&mut position.clone(), &mut zero_cell, NptConfig::default()),
            SaddleError::Invalid(_)
        ));

        let mut nonfinite_position = position;
        nonfinite_position[0] = f64::NAN;
        assert!(matches!(
            input_error(
                &mut nonfinite_position,
                &mut cell.clone(),
                NptConfig::default()
            ),
            SaddleError::NonFinite("NPT initial positions")
        ));
    }

    #[test]
    fn npt_extension_check_saturates_at_the_maximum() {
        assert_eq!(
            extended_check(usize::MAX - 1, usize::MAX, usize::MAX),
            usize::MAX
        );
    }

    fn stream(seed: u64, length: usize) -> Vec<u64> {
        let mut rng = Rng::new(seed);
        (0..length).map(|_| rng.next_u64()).collect()
    }

    #[test]
    fn splitmix64_matches_the_published_stream() {
        let mut state = 0_u64;
        assert_eq!(splitmix64(&mut state), 0xE220_A839_7B1D_CDAF);
        assert_eq!(splitmix64(&mut state), 0x6E78_9E6A_A1B9_65F4);
        assert_eq!(splitmix64(&mut state), 0x06C4_5D18_8009_454F);
    }

    #[test]
    fn xoshiro_stream_is_pinned_for_a_zero_seed() {
        assert_eq!(
            stream(0, 3),
            [
                0x99EC_5F36_CB75_F2B4,
                0xBF6E_1F78_4956_452A,
                0x1A5F_849D_4933_E6E0
            ]
        );
    }

    #[test]
    fn distinct_seeds_give_distinct_streams() {
        let streams: Vec<Vec<u64>> = (0..64).map(|seed| stream(seed, 8)).collect();
        for (a, first) in streams.iter().enumerate() {
            for second in &streams[a + 1..] {
                assert_ne!(first, second);
                assert!(first.iter().zip(second).all(|(x, y)| x != y));
            }
        }
        for pair in (0..32_u64).map(|k| (stream(2 * k, 8), stream(2 * k + 1, 8))) {
            assert_ne!(pair.0, pair.1);
        }
    }

    #[test]
    fn nearby_seeds_are_uncorrelated_in_their_first_draw() {
        let draws: Vec<f64> = (0..4096_u64).map(|seed| Rng::new(seed).uniform()).collect();
        let mean = draws.iter().sum::<f64>() / draws.len() as f64;
        let variance =
            draws.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (draws.len() - 1) as f64;
        assert!((mean - 0.5).abs() < 0.02, "{mean}");
        assert!((variance - 1.0 / 12.0).abs() < 0.005, "{variance}");
        let lag = draws
            .windows(2)
            .map(|pair| (pair[0] - mean) * (pair[1] - mean))
            .sum::<f64>()
            / (draws.len() - 1) as f64
            / variance;
        assert!(lag.abs() < 0.05, "{lag}");
    }

    #[test]
    fn normal_draws_have_unit_variance() {
        let mut rng = Rng::new(7);
        let draws: Vec<f64> = (0..200_000).map(|_| rng.normal()).collect();
        let mean = draws.iter().sum::<f64>() / draws.len() as f64;
        let variance =
            draws.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (draws.len() - 1) as f64;
        assert!(mean.abs() < 0.01, "{mean}");
        assert!((variance - 1.0).abs() < 0.02, "{variance}");
    }

    fn autoregressive(seed: u64, phi: f64, length: usize) -> Vec<f64> {
        let mut rng = Rng::new(seed);
        let mut value = rng.normal() / (1.0 - phi * phi).sqrt();
        (0..length)
            .map(|_| {
                value = phi * value + rng.normal();
                value
            })
            .collect()
    }

    /// Standard error of the mean of a long stationary AR(1) series with
    /// unit innovations: `sqrt((1 + phi) / (1 - phi) / (1 - phi^2) / n)`.
    fn exact_standard_error(phi: f64, length: usize) -> f64 {
        ((1.0 + phi) / (1.0 - phi) / (1.0 - phi * phi) / length as f64).sqrt()
    }

    #[test]
    fn blocking_recovers_the_standard_error_of_correlated_data() {
        let phi = 0.995;
        let length = 1 << 17;
        let exact = exact_standard_error(phi, length);
        let seeds = 24;
        let mut ratio = 0.0;
        let mut resolved = 0;
        let mut naive_ratio = 0.0;
        for seed in 0..seeds {
            let series = autoregressive(seed, phi, length);
            let blocked = blocked_statistics(&series);
            ratio += blocked.standard_error / exact / seeds as f64;
            resolved += usize::from(blocked.resolved);
            let means: Vec<f64> = (0..series.len() / 250)
                .map(|block| series[250 * block..250 * (block + 1)].iter().sum::<f64>() / 250.0)
                .collect();
            naive_ratio += mean_and_standard_error(&means).1 / exact / seeds as f64;
        }
        eprintln!(
            "AR(1) phi {phi}: blocked over exact {ratio:.3}, 250-step blocks over exact \
             {naive_ratio:.3}, resolved {resolved} of {seeds}"
        );
        assert!((ratio - 1.0).abs() < 0.25, "blocked over exact {ratio}");
        assert!(resolved >= 3 * seeds as usize / 4, "resolved {resolved}");
        assert!(
            naive_ratio < 0.75,
            "250-step blocks should underestimate, got {naive_ratio}"
        );
    }

    #[test]
    fn blocking_does_not_resolve_a_run_shorter_than_the_correlation() {
        let phi = 0.995;
        let mut resolved = 0;
        let mut ratio = 0.0;
        let seeds = 24;
        for seed in 0..seeds {
            let series = autoregressive(100 + seed, phi, 1000);
            let blocked = blocked_statistics(&series);
            resolved += usize::from(blocked.resolved);
            ratio += blocked.standard_error / exact_standard_error(phi, 1000) / seeds as f64;
        }
        eprintln!("short AR(1): SE over exact {ratio:.3}, resolved {resolved} of {seeds}");
        assert!(resolved <= seeds as usize / 4, "resolved {resolved}");
        assert!(ratio < 0.8, "a short run is a lower bound, got {ratio}");
    }

    #[test]
    fn blocking_matches_the_plain_error_for_uncorrelated_data() {
        let length = 1 << 14;
        let mut ratio = 0.0;
        let seeds = 24;
        for seed in 0..seeds {
            let mut rng = Rng::new(200 + seed);
            let series: Vec<f64> = (0..length).map(|_| rng.normal()).collect();
            let blocked = blocked_statistics(&series);
            ratio += blocked.standard_error * (length as f64).sqrt() / seeds as f64;
        }
        eprintln!("iid: blocked SE over plain SE {ratio:.3}");
        assert!((1.0..1.2).contains(&ratio), "{ratio}");
    }

    #[test]
    fn a_series_without_spread_has_zero_error_and_is_resolved() {
        let blocked = blocked_statistics(&[3.6; 40]);
        assert_eq!(blocked.mean, 3.6);
        assert_eq!(blocked.standard_error, 0.0);
        assert!(blocked.resolved);
        let short = blocked_statistics(&[1.0, 2.0, 4.0]);
        assert!(!short.resolved);
        assert!(short.standard_error > 0.0);
    }
}
