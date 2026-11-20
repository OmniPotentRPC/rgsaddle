//! Isotropic Langevin--Berendsen sampling of a crystal lattice.
//!
//! The force evaluator supplies a tension-positive Cauchy stress.  The
//! reported pressure is therefore minus one third of its trace.

use ndarray::{Array1, ArrayView1};

use crate::{BOLTZMANN_EV_PER_K, Cell, SaddleError, angular_scale_rad_s};

/// Controls an isotropic NPT lattice run.
#[derive(Clone, Copy, Debug)]
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
    /// Steps discarded before block statistics are accumulated.
    pub equilibration_steps: usize,
    /// Number of production steps in one statistics block.
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
    /// Seed for the deterministic thermostat stream.
    pub seed: u64,
}

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

impl NptConfig {
    /// Set the conventional-cell repeat count.
    pub fn with_repeats(mut self, repeats: usize) -> Self {
        self.repeats = repeats;
        self
    }

    /// Set the initial integration length.
    pub fn with_steps(mut self, steps: usize) -> Self {
        self.steps = steps;
        self
    }

    /// Set the maximum integration length.
    pub fn with_maximum_steps(mut self, steps: usize) -> Self {
        self.maximum_steps = steps;
        self
    }

    /// Set the extension between pressure checks.
    pub fn with_extension_steps(mut self, steps: usize) -> Self {
        self.extension_steps = steps;
        self
    }

    /// Set the equilibration discard.
    pub fn with_equilibration_steps(mut self, steps: usize) -> Self {
        self.equilibration_steps = steps;
        self
    }

    /// Set the statistics block length.
    pub fn with_block_steps(mut self, steps: usize) -> Self {
        self.block_steps = steps;
        self
    }

    /// Set the velocity-Verlet time step.
    pub fn with_timestep_s(mut self, timestep_s: f64) -> Self {
        self.timestep_s = timestep_s;
        self
    }

    /// Set the Langevin friction.
    pub fn with_langevin_gamma_per_s(mut self, gamma_per_s: f64) -> Self {
        self.langevin_gamma_per_s = gamma_per_s;
        self
    }

    /// Set the Berendsen pressure-coupling time.
    pub fn with_berendsen_tau_s(mut self, tau_s: f64) -> Self {
        self.berendsen_tau_s = tau_s;
        self
    }

    /// Set the isotropic compressibility.
    pub fn with_compressibility_per_ev_a3(mut self, compressibility: f64) -> Self {
        self.compressibility_per_ev_a3 = compressibility;
        self
    }

    /// Set the target thermodynamic pressure.
    pub fn with_target_pressure_ev_a3(mut self, pressure: f64) -> Self {
        self.target_pressure_ev_a3 = pressure;
        self
    }

    /// Set the thermostat seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

/// Block-averaged result of an NPT lattice run.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct NptReport {
    /// Total number of integration steps.
    pub steps: usize,
    /// Number of complete production blocks.
    pub blocks: usize,
    /// Mean conventional lattice parameter in Angstrom.
    pub mean_lattice_a: f64,
    /// Standard error of the lattice block means in Angstrom.
    pub lattice_standard_error: f64,
    /// Effective number of production-step lattice samples after correlation.
    pub lattice_effective_samples: f64,
    /// Mean thermodynamic pressure in eV per cubic Angstrom.
    pub mean_pressure: f64,
    /// Standard error of the pressure block means.
    pub pressure_standard_error: f64,
    /// Effective number of production-step pressure samples after correlation.
    pub pressure_effective_samples: f64,
    /// Whether pressure blocking found a standard-error plateau.
    pub pressure_statistics_resolved: bool,
    /// Whether `|mean pressure - target| <= 2 SE`, including exact zero,
    /// with resolved statistics and at least 16 effective samples.
    pub pressure_converged: bool,
}

#[derive(Clone, Copy)]
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    fn uniform(&mut self) -> f64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64))
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1.0e-16);
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

const MIN_EFFECTIVE_SAMPLES: f64 = 16.0;
const MIN_PLATEAU_BLOCKS: usize = 16;

#[derive(Clone, Copy, Debug)]
struct BlockEstimate {
    mean: f64,
    standard_error: f64,
    effective_samples: f64,
    resolved: bool,
}

fn sample_variance(values: &[f64], mean: f64) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64
}

/// Flyvbjerg--Petersen blocking estimate for a correlated series.
///
/// Adjacent values are repeatedly averaged until successive standard-error
/// estimates overlap their sampling uncertainties. See Flyvbjerg
/// and Petersen, J. Chem. Phys. 91, 461--466 (1989),
/// doi:10.1063/1.457480. Failure to find that plateau is explicit.
fn block_statistics(values: &[f64]) -> BlockEstimate {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let raw_variance = sample_variance(values, mean);
    if raw_variance == 0.0 {
        return BlockEstimate {
            mean,
            standard_error: 0.0,
            effective_samples: values.len() as f64,
            resolved: true,
        };
    }

    let mut blocked = values.to_vec();
    let mut levels = Vec::new();
    while blocked.len() >= MIN_PLATEAU_BLOCKS {
        let level_mean = blocked.iter().sum::<f64>() / blocked.len() as f64;
        let variance = sample_variance(&blocked, level_mean);
        let standard_error = (variance / blocked.len() as f64).sqrt();
        let uncertainty = standard_error / (2.0 * (blocked.len() - 1) as f64).sqrt();
        levels.push((standard_error, uncertainty));

        let pairs = blocked.len() / 2;
        for index in 0..pairs {
            blocked[index] = 0.5 * (blocked[2 * index] + blocked[2 * index + 1]);
        }
        blocked.truncate(pairs);
    }

    let stable = |left: (f64, f64), right: (f64, f64)| {
        (left.0 - right.0).abs() <= 2.0 * left.1.hypot(right.1)
    };
    let plateau = levels
        .windows(2)
        .rev()
        .find(|window| stable(window[0], window[1]));
    let resolved = plateau.is_some();
    let standard_error = plateau
        .map(|window| window.iter().map(|level| level.0).fold(0.0, f64::max))
        .unwrap_or_else(|| levels.iter().map(|level| level.0).fold(0.0, f64::max));
    let effective_samples = if standard_error > 0.0 {
        (raw_variance / standard_error.powi(2)).min(values.len() as f64)
    } else {
        values.len() as f64
    };
    BlockEstimate {
        mean,
        standard_error,
        effective_samples,
        resolved,
    }
}

fn summarize(lattice: &[f64], pressure: &[f64], config: NptConfig, steps: usize) -> NptReport {
    let complete = lattice.len() / config.block_steps * config.block_steps;
    let lattice = block_statistics(&lattice[..complete]);
    let pressure = block_statistics(&pressure[..complete]);
    let residual = (pressure.mean - config.target_pressure_ev_a3).abs();
    let within_error = if pressure.standard_error == 0.0 {
        residual == 0.0
    } else {
        residual <= 2.0 * pressure.standard_error
    };
    let pressure_converged =
        within_error && pressure.resolved && pressure.effective_samples >= MIN_EFFECTIVE_SAMPLES;
    NptReport {
        steps,
        blocks: complete / config.block_steps,
        mean_lattice_a: lattice.mean,
        lattice_standard_error: lattice.standard_error,
        lattice_effective_samples: lattice.effective_samples,
        mean_pressure: pressure.mean,
        pressure_standard_error: pressure.standard_error,
        pressure_effective_samples: pressure.effective_samples,
        pressure_statistics_resolved: pressure.resolved,
        pressure_converged,
    }
}

/// Run isotropic NPT dynamics and return autocorrelation-aware standard errors.
///
/// `evaluate` returns `(energy, gradient, tension-positive stress)` for
/// Cartesian `positions` and the row-major `cell`.  Energy is accepted for
/// evaluator compatibility but is not used by the integrator.  A run extends
/// by `extension_steps` until the zero-pressure rule is met or
/// `maximum_steps` is reached.
///
/// Standard errors use recursive blocking of adjacent samples and require a
/// resolved plateau; see Flyvbjerg and Petersen, J. Chem. Phys. 91, 461--466
/// (1989), doi:10.1063/1.457480.
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
    if positions.is_empty()
        || !positions.len().is_multiple_of(3)
        || config.repeats == 0
        || config.steps > config.maximum_steps
        || config.extension_steps == 0
        || config.block_steps == 0
        || production / config.block_steps < 2
        || !(temperature_k >= 0.0 && temperature_k.is_finite())
        || !(mass_amu > 0.0 && mass_amu.is_finite())
        || !(config.timestep_s > 0.0
            && config.langevin_gamma_per_s > 0.0
            && config.berendsen_tau_s > 0.0
            && config.compressibility_per_ev_a3 > 0.0)
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
            next_check = (next_check + config.extension_steps).min(config.maximum_steps);
        }
    }
    report.ok_or_else(|| SaddleError::Invalid("NPT run produced no report".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocking_recovers_the_ar1_mean_variance() {
        let rho = 0.99_f64;
        let count = 1 << 17;
        let mut rng = Rng::new(0x5eed);
        let mut value = rng.normal() / (1.0 - rho * rho).sqrt();
        let mut series = Vec::with_capacity(count);
        for _ in 0..count {
            value = rho * value + rng.normal();
            series.push(value);
        }

        let estimate = block_statistics(&series);
        let stationary_variance = 1.0 / (1.0 - rho * rho);
        let expected_variance_of_mean =
            stationary_variance * (1.0 + rho) / ((1.0 - rho) * count as f64);
        let ratio = estimate.standard_error.powi(2) / expected_variance_of_mean;
        assert!(estimate.resolved, "{estimate:?}");
        assert!((0.75..1.25).contains(&ratio), "{estimate:?}, ratio {ratio}");

        let expected_effective = count as f64 * (1.0 - rho) / (1.0 + rho);
        let effective_ratio = estimate.effective_samples / expected_effective;
        assert!(
            (0.75..1.25).contains(&effective_ratio),
            "{estimate:?}, effective ratio {effective_ratio}"
        );
    }

    #[test]
    fn pressure_with_too_few_effective_samples_is_not_converged() {
        let series = [vec![-1.0; 16], vec![1.0; 16]].concat();
        let count = series.len();
        let target = series.iter().sum::<f64>() / count as f64;
        let config = NptConfig::default()
            .with_block_steps(1)
            .with_target_pressure_ev_a3(target);
        let report = summarize(&series, &series, config, count);

        assert!(
            report.pressure_effective_samples < MIN_EFFECTIVE_SAMPLES,
            "{report:?}"
        );
        assert!(!report.pressure_converged, "{report:?}");
    }
}
