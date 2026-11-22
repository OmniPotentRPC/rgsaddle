//! Isotropic NPT sampling of a cubic crystal lattice: Langevin dynamics
//! with a stochastic cell rescaling barostat.
//!
//! The force evaluator supplies a tension-positive Cauchy stress.  The
//! reported pressure is therefore minus one third of its trace, plus the
//! kinetic pressure.  The default barostat is stochastic cell rescaling
//! (Bernetti and Bussi, J. Chem. Phys. 153, 114107 (2020),
//! doi:10.1063/5.0020514), which samples the isotropic NPT ensemble.  The
//! Berendsen barostat remains available for equilibration and does not
//! report ensemble fluctuations.

use ndarray::{Array1, ArrayView1};

use crate::{BOLTZMANN_EV_PER_K, Cell, SaddleError, angular_scale_rad_s};

/// Which isotropic barostat [`npt_lattice`] applies.
///
/// Non-exhaustive: construct a variant. The default is
/// [`NptBarostat::StochasticCellRescaling`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum NptBarostat {
    /// Stochastic cell rescaling. Samples the isotropic NPT ensemble, so
    /// the volume variance does not depend on the coupling time.
    #[default]
    StochasticCellRescaling,
    /// Berendsen pressure coupling, for equilibration only. The volume
    /// distribution is not the NPT ensemble and depends on the coupling
    /// time. The report carries means and no ensemble fluctuation
    /// statistics.
    Berendsen,
}

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
    /// Barostat that advances the cell. The default samples NPT.
    /// [`NptBarostat::Berendsen`] is equilibration only.
    pub barostat: NptBarostat,
    /// Pressure-coupling time in seconds, for whichever [`Self::barostat`]
    /// is selected. The name is the historical Berendsen parameter. It is
    /// also τ_P in stochastic cell rescaling.
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
    with_barostat => barostat: val NptBarostat,
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
            barostat: NptBarostat::StochasticCellRescaling,
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
    /// [`NptBarostat::Berendsen`] leaves this `false`: the screen needs
    /// ensemble errors that barostat does not produce.
    pub pressure_converged: bool,
    /// Whether the standard errors are fluctuations of the NPT ensemble.
    /// `false` for [`NptBarostat::Berendsen`], which reports means only:
    /// both standard errors are zero and both resolved flags are `false`,
    /// and those zeros are not a measured variance.
    pub ensemble_fluctuations: bool,
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
fn blocked_statistics(values: &[f64]) -> Result<Blocked, SaddleError> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(SaddleError::NonFinite("NPT sample series"));
    }
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
        return Ok(Blocked {
            mean,
            standard_error: 0.0,
            resolved: false,
        });
    }
    // `f64::max` returns the other operand for a NaN, which would turn a
    // NaN estimate into a zero standard error, so every estimate is
    // checked before it is folded.
    let mut standard_error = 0.0_f64;
    for &(estimate, _) in &estimates {
        if !estimate.is_finite() {
            return Err(SaddleError::NonFinite("NPT standard error"));
        }
        standard_error = standard_error.max(estimate);
    }
    if !mean.is_finite() {
        return Err(SaddleError::NonFinite("NPT mean"));
    }
    let resolved = if estimates[0].0 == 0.0 {
        true
    } else if let [.., (previous, _), (last, blocks)] = estimates[..] {
        last - previous <= last / (2.0 * (blocks as f64 - 1.0)).sqrt()
    } else {
        false
    };
    Ok(Blocked {
        mean,
        standard_error,
        resolved,
    })
}

fn summarize(
    lattice: &[f64],
    pressure: &[f64],
    config: NptConfig,
    steps: usize,
) -> Result<NptReport, SaddleError> {
    let lattice = blocked_statistics(lattice)?;
    let pressure_stats = blocked_statistics(pressure)?;
    let blocks = pressure.len() / config.block_steps;
    // Berendsen does not sample the NPT volume distribution. Reporting a
    // blocking error would present a coupling-dependent number as an
    // ensemble fluctuation, so the means are kept and the errors are not.
    if config.barostat == NptBarostat::Berendsen {
        return Ok(NptReport {
            steps,
            blocks,
            mean_lattice_a: lattice.mean,
            lattice_standard_error: 0.0,
            lattice_error_resolved: false,
            mean_pressure: pressure_stats.mean,
            pressure_standard_error: 0.0,
            pressure_error_resolved: false,
            pressure_converged: false,
            ensemble_fluctuations: false,
        });
    }
    let residual = (pressure_stats.mean - config.target_pressure_ev_a3).abs();
    let within = if pressure_stats.standard_error == 0.0 {
        residual == 0.0
    } else {
        residual <= 2.0 * pressure_stats.standard_error
    };
    Ok(NptReport {
        steps,
        blocks,
        mean_lattice_a: lattice.mean,
        lattice_standard_error: lattice.standard_error,
        lattice_error_resolved: lattice.resolved,
        mean_pressure: pressure_stats.mean,
        pressure_standard_error: pressure_stats.standard_error,
        pressure_error_resolved: pressure_stats.resolved,
        pressure_converged: pressure_stats.resolved && within,
        ensemble_fluctuations: true,
    })
}

fn check_gradient(gradient: &Array1<f64>, length: usize) -> Result<(), SaddleError> {
    if gradient.len() != length {
        return Err(SaddleError::Shape(
            "NPT gradient length differs from the positions".into(),
        ));
    }
    if gradient.iter().any(|value| !value.is_finite()) {
        return Err(SaddleError::NonFinite("NPT gradient"));
    }
    Ok(())
}

/// Refuse every input the sampler cannot integrate, before any step runs
/// and before `positions` or `cell` are touched.
fn validate(
    positions: &Array1<f64>,
    cell: Cell,
    temperature_k: f64,
    mass_amu: f64,
    config: NptConfig,
) -> Result<(), SaddleError> {
    let production = config.steps.saturating_sub(config.equilibration_steps);
    let positive_finite = |value: f64| value.is_finite() && value > 0.0;
    // The thermostat removes the centre-of-mass momentum, which leaves
    // 3 (N - 1) thermal degrees of freedom. A single atom has none.
    if positions.len() == 3 {
        return Err(SaddleError::Invalid(
            "NPT sampling removes the centre-of-mass momentum and needs at least two atoms".into(),
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
        || !positive_finite(mass_amu)
        || !positive_finite(config.timestep_s)
        || !positive_finite(config.langevin_gamma_per_s)
        || !positive_finite(config.berendsen_tau_s)
        || !positive_finite(config.compressibility_per_ev_a3)
        || !config.target_pressure_ev_a3.is_finite()
    {
        return Err(SaddleError::Invalid(
            "NPT lattice sampling needs finite physical controls and at least two production blocks"
                .into(),
        ));
    }
    let omega = angular_scale_rad_s();
    if !positive_finite(config.timestep_s * omega)
        || !positive_finite(config.langevin_gamma_per_s / omega)
        || !positive_finite(config.berendsen_tau_s * omega)
    {
        return Err(SaddleError::Invalid(
            "NPT time controls overflow or underflow in internal units".into(),
        ));
    }
    if positions.iter().any(|value| !value.is_finite()) {
        return Err(SaddleError::NonFinite("NPT positions"));
    }
    if cell.0.iter().flatten().any(|value| !value.is_finite()) {
        return Err(SaddleError::NonFinite("NPT cell"));
    }
    let volume = determinant(cell).abs();
    if !positive_finite(volume) {
        return Err(SaddleError::Invalid(
            "NPT cell must have a finite, non-zero volume".into(),
        ));
    }
    require_cubic_cell(cell)?;
    if !extension_schedule_fits(config.steps, config.extension_steps, config.maximum_steps) {
        return Err(SaddleError::Invalid(
            "NPT extension_steps overflows the step counter".into(),
        ));
    }
    Ok(())
}

/// Relative tolerance on edge lengths and on the cosine of the cell angles.
/// A cell outside it is refused. It is not rescaled into a cube.
const CUBIC_TOLERANCE: f64 = 1.0e-8;

/// The reported lattice parameter is `cbrt(volume) / repeats`, which is the
/// conventional cubic lattice constant. A non-cubic cell is an error.
fn require_cubic_cell(cell: Cell) -> Result<(), SaddleError> {
    let rows = cell.0;
    let mut length = [0.0; 3];
    for (index, row) in rows.iter().enumerate() {
        length[index] = (row[0] * row[0] + row[1] * row[1] + row[2] * row[2]).sqrt();
    }
    let edge = length[0];
    let refused = || {
        SaddleError::Invalid(
            "NPT lattice sampling reports one lattice parameter and accepts only a cubic cell"
                .into(),
        )
    };
    if !(edge.is_finite() && edge > 0.0) {
        return Err(refused());
    }
    for len in length {
        if (len - edge).abs() > CUBIC_TOLERANCE * edge {
            return Err(refused());
        }
    }
    for left in 0..3 {
        for right in (left + 1)..3 {
            let dot = rows[left][0] * rows[right][0]
                + rows[left][1] * rows[right][1]
                + rows[left][2] * rows[right][2];
            if dot.abs() > CUBIC_TOLERANCE * edge * edge {
                return Err(refused());
            }
        }
    }
    Ok(())
}

/// Whether every checkpoint before `maximum` can add `extension` without
/// wrapping the step counter. A wrapped add skips the final report.
fn extension_schedule_fits(steps: usize, extension: usize, maximum: usize) -> bool {
    if extension == 0 {
        return false;
    }
    if steps > maximum {
        return false;
    }
    if steps == maximum {
        return true;
    }
    let Some(room) = usize::MAX.checked_sub(extension) else {
        return false;
    };
    if steps > room {
        return false;
    }
    // Every index below `maximum` can take the add.
    if maximum - 1 <= room {
        return true;
    }
    // Otherwise the last checkpoint that is still safe to add to must land
    // on `maximum` or beyond. Landing strictly between `room` and
    // `maximum` would overflow on the following add.
    let k_max = (room - steps) / extension;
    let last_safe = steps + k_max * extension;
    match last_safe.checked_add(extension) {
        Some(next) => next >= maximum,
        None => false,
    }
}

fn advance_checkpoint(
    current: usize,
    extension: usize,
    maximum: usize,
) -> Result<usize, SaddleError> {
    let sum = current.checked_add(extension).ok_or_else(|| {
        SaddleError::Invalid("NPT extension_steps overflows the step counter".into())
    })?;
    Ok(sum.min(maximum))
}

/// Run isotropic NPT dynamics and return block standard errors.
///
/// `evaluate` returns `(energy, gradient, tension-positive stress)` for
/// Cartesian `positions` and the row-major `cell`.  Energy is accepted for
/// evaluator compatibility but is not used by the integrator.  A run extends
/// by `extension_steps` until the pressure screen is met or `maximum_steps`
/// is reached.  The cell must be cubic: equal, orthogonal edges within a
/// relative tolerance of `1e-8`.  Outside that tolerance the call is
/// refused rather than reporting `cbrt(volume) / repeats` for a cell that
/// is not a cube.  The reported lattice parameter is that quotient, with
/// [`NptConfig::repeats`] the number of conventional cells along an edge.
///
/// The default barostat is stochastic cell rescaling.  Its log-volume
/// increment is the Berendsen drift plus the noise of Bernetti and Bussi
/// (J. Chem. Phys. 153, 114107 (2020), doi:10.1063/5.0020514, eq. 5).
/// Positions and the cell are scaled by the exponential of one third of
/// that increment, and the velocities by the inverse, because the pressure
/// uses the instantaneous kinetic energy.  [`NptBarostat::Berendsen`] keeps
/// the deterministic rescaling, limited to half a percent, and reports no
/// ensemble fluctuations.
///
/// # Errors
///
/// [`SaddleError::Invalid`] for a single atom, which has no thermal degrees
/// of freedom once the centre-of-mass momentum is removed, when `positions`
/// is empty or not a multiple of three, when a control is not finite and
/// positive (the target pressure only has to be finite), when the run is
/// too short for two production blocks, when the cell has no volume, when
/// the cell is not cubic, or when adding `extension_steps` to the step
/// counter would overflow. [`SaddleError::NonFinite`]
/// when `positions` or `cell` hold a NaN or an infinity, or when the
/// evaluator or the integration produces one; no partial report is
/// returned. Invalid input is refused before `evaluate` is called and
/// before `positions` or `cell` change.
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
    validate(positions, *cell, temperature_k, mass_amu, config)?;

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
        check_gradient(&gradient, positions.len())?;
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
        check_gradient(&gradient, positions.len())?;
        if stress.iter().flatten().any(|value| !value.is_finite()) {
            return Err(SaddleError::NonFinite("NPT stress"));
        }
        velocity = &velocity + &(&gradient * (-0.5 * dt / mass_amu));

        let configurational_stress = (stress[0][0] + stress[1][1] + stress[2][2]) / 3.0;
        let kinetic_energy = velocity
            .iter()
            .map(|value| 0.5 * mass_amu * value * value)
            .sum::<f64>();
        let volume = determinant(*cell).abs();
        if !(volume.is_finite() && volume > 0.0) {
            return Err(SaddleError::NonFinite("NPT cell volume"));
        }
        let kinetic_pressure = 2.0 * kinetic_energy / (3.0 * volume);
        let hydrostatic_stress = configurational_stress - kinetic_pressure;
        if !hydrostatic_stress.is_finite() {
            return Err(SaddleError::NonFinite("NPT pressure"));
        }
        let internal_pressure = -hydrostatic_stress;
        let coupling = config.compressibility_per_ev_a3 * dt / tau;
        let (length_factor, scale_velocities) = match config.barostat {
            NptBarostat::StochasticCellRescaling => {
                // Eq. 5 of Bernetti and Bussi, integrated on the log volume.
                // The Wiener increment has variance `dt`.
                let drift = -coupling * (config.target_pressure_ev_a3 - internal_pressure);
                let diffusion = (2.0 * thermal * coupling / volume).sqrt();
                if !drift.is_finite() || !diffusion.is_finite() {
                    return Err(SaddleError::NonFinite("NPT barostat volume factor"));
                }
                let d_log_volume = drift + diffusion * rng.normal();
                if !d_log_volume.is_finite() {
                    return Err(SaddleError::NonFinite("NPT barostat volume factor"));
                }
                ((d_log_volume / 3.0).exp(), true)
            }
            NptBarostat::Berendsen => {
                let drive = config.target_pressure_ev_a3 - internal_pressure;
                let volume_factor = (1.0 - coupling * drive).clamp(0.995, 1.005);
                if !volume_factor.is_finite() || volume_factor <= 0.0 {
                    return Err(SaddleError::NonFinite("NPT barostat volume factor"));
                }
                (volume_factor.cbrt(), false)
            }
        };
        if !length_factor.is_finite() || length_factor <= 0.0 {
            return Err(SaddleError::NonFinite("NPT barostat volume factor"));
        }
        *positions *= length_factor;
        for row in &mut cell.0 {
            for value in row {
                *value *= length_factor;
            }
        }
        if scale_velocities {
            velocity *= 1.0 / length_factor;
            if velocity.iter().any(|value| !value.is_finite()) {
                return Err(SaddleError::NonFinite("NPT velocity"));
            }
        }

        let completed = step + 1;
        if completed > config.equilibration_steps {
            let lattice = determinant(*cell).abs().cbrt() / config.repeats as f64;
            if !lattice.is_finite() {
                return Err(SaddleError::NonFinite("NPT lattice sample"));
            }
            lattice_samples.push(lattice);
            pressure_samples.push(-hydrostatic_stress);
        }
        if completed == next_check {
            let current = summarize(&lattice_samples, &pressure_samples, config, completed)?;
            if current.pressure_converged || completed == config.maximum_steps {
                return Ok(current);
            }
            report = Some(current);
            next_check =
                advance_checkpoint(next_check, config.extension_steps, config.maximum_steps)?;
        }
    }
    // A checkpoint that was never visited must not hand back the previous
    // one. The samples collected through the last step are the final state.
    if lattice_samples.is_empty() {
        return report.ok_or_else(|| SaddleError::Invalid("NPT run produced no report".into()));
    }
    summarize(
        &lattice_samples,
        &pressure_samples,
        config,
        config.maximum_steps,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

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
            let blocked = blocked_statistics(&series).unwrap();
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
            let blocked = blocked_statistics(&series).unwrap();
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
            let blocked = blocked_statistics(&series).unwrap();
            ratio += blocked.standard_error * (length as f64).sqrt() / seeds as f64;
        }
        eprintln!("iid: blocked SE over plain SE {ratio:.3}");
        assert!((1.0..1.2).contains(&ratio), "{ratio}");
    }

    #[test]
    fn a_series_without_spread_has_zero_error_and_is_resolved() {
        let blocked = blocked_statistics(&[3.6; 40]).unwrap();
        assert_eq!(blocked.mean, 3.6);
        assert_eq!(blocked.standard_error, 0.0);
        assert!(blocked.resolved);
        let short = blocked_statistics(&[1.0, 2.0, 4.0]).unwrap();
        assert!(!short.resolved);
        assert!(short.standard_error > 0.0);
    }

    #[test]
    fn blocking_refuses_a_series_with_a_non_finite_value() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut series = vec![3.6; 40];
            series[17] = bad;
            assert!(matches!(
                blocked_statistics(&series),
                Err(SaddleError::NonFinite(_))
            ));
        }
        // A NaN used to vanish into a zero standard error.
        assert!(blocked_statistics(&[f64::NAN; 40]).is_err());
        assert!(blocked_statistics(&[f64::NAN, 1.0, 2.0]).is_err());
    }

    #[test]
    fn extension_schedule_rejects_a_counter_that_would_wrap() {
        assert!(extension_schedule_fits(40, 1_000, 4_000));
        assert!(extension_schedule_fits(40, usize::MAX, 40));
        assert!(!extension_schedule_fits(40, usize::MAX, usize::MAX));
        assert!(!extension_schedule_fits(usize::MAX - 5, 10, usize::MAX));
        assert!(extension_schedule_fits(10, usize::MAX / 2, 1_000));
        assert!(!extension_schedule_fits(10, usize::MAX / 2, usize::MAX));
        assert!(advance_checkpoint(40, 1_000, 4_000).is_ok());
        assert!(advance_checkpoint(usize::MAX - 5, 10, usize::MAX).is_err());
    }

    #[test]
    fn blocking_refuses_a_series_whose_error_overflows() {
        let series: Vec<f64> = (0..40)
            .map(|index| if index % 2 == 0 { 1.0e200 } else { -1.0e200 })
            .collect();
        assert!(matches!(
            blocked_statistics(&series),
            Err(SaddleError::NonFinite(_))
        ));
    }
}
