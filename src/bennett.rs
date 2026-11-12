//! Bennett–Chandler transmission coefficient.
//!
//! Bennett, in Algorithms for Chemical Computations, ACS Symposium Series
//! 46, 63 (1977), and Chandler, J. Chem. Phys. 68, 2959 (1978),
//! doi:10.1063/1.436049, write the recrossing factor as the plateau of the
//! flux correlation through a dividing surface. For a surface with unit
//! normal `τ` and configurations drawn from the constrained ensemble,
//!
//! `κ(t) = ⟨ v_τ(0) θ(s(t)) ⟩ / ⟨ v_τ(0) θ(v_τ(0)) ⟩`,
//!
//! where `s(t) = (x(t) - x_surface) · τ` and `θ(s)` is 1 on the product
//! side. The numerator keeps the sign of the initial velocity, so a
//! trajectory that starts toward the reactant and finishes in the product
//! counts against the flux. The denominator is the positive flux. The
//! average is over Maxwell–Boltzmann velocities. Trajectories are
//! inertial Langevin, BAOAB,
//!
//! `m dv = (-∇E - m γ v) dt + sqrt(2 m γ kT) dW`,
//!
//! integrated by a velocity-Verlet / Ornstein–Uhlenbeck split. `γ = 0` is
//! Newtonian. The reported `κ` is `κ(t)` at the requested plateau record.
//! The standard error is the block standard error of that ratio.

use ndarray::{Array1, ArrayView1};

use crate::error::SaddleError;
use crate::minmode::PointSurface;

/// Inertial shooting from one dividing surface.
#[derive(Clone, Debug)]
pub struct ShootingConfig {
    /// `kT` in the energy unit of the surface.
    pub thermal: f64,
    /// Mass of every degree of freedom, in the surface's mass unit.
    pub mass: f64,
    /// Langevin friction `γ` in inverse time. Zero is Newtonian.
    pub friction: f64,
    /// Velocity-Verlet time step.
    pub time_step: f64,
    /// Integration steps after the initial velocity is drawn.
    pub steps: usize,
    /// Steps between stored values of `κ(t)`.
    pub record_every: usize,
    /// Shots. Starts are cycled if there are fewer configurations than shots.
    pub shots: usize,
    /// Blocks of shots for the standard error of the plateau value.
    pub blocks: usize,
    pub seed: u64,
    /// Which stored time is the plateau. Zero is the first record.
    pub plateau_record: usize,
}

impl Default for ShootingConfig {
    fn default() -> Self {
        Self {
            thermal: 0.02585,
            mass: 1.0,
            friction: 0.0,
            time_step: 1.0e-3,
            steps: 200,
            record_every: 10,
            shots: 64,
            blocks: 4,
            seed: 1,
            plateau_record: 0,
        }
    }
}

/// `κ(t)` and the plateau value.
#[derive(Clone, Debug)]
pub struct Transmission {
    /// Times of the stored records, in the same unit as [`ShootingConfig::time_step`].
    pub time: Vec<f64>,
    /// `κ(t)` at those times.
    pub kappa_of_t: Vec<f64>,
    /// `κ` at [`ShootingConfig::plateau_record`].
    pub kappa: f64,
    /// Block standard error of `kappa`.
    pub error: f64,
    /// Shots with `v_τ > 0` that entered the denominator.
    pub forward: usize,
}

/// Shoot from `starts` and return the Bennett–Chandler factor.
///
/// `tangent` is the product direction. It is normalised. `reference` is the
/// point on the dividing surface that defines `s = 0`; every start should
/// already lie on that plane.
pub fn bennett_chandler<S: PointSurface>(
    surface: &S,
    starts: &[Array1<f64>],
    reference: ArrayView1<f64>,
    tangent: ArrayView1<f64>,
    config: &ShootingConfig,
) -> Result<Transmission, SaddleError> {
    if starts.is_empty() {
        return Err(SaddleError::Invalid(
            "Bennett-Chandler needs at least one dividing-surface configuration".into(),
        ));
    }
    if !(config.thermal.is_finite() && config.thermal > 0.0) {
        return Err(SaddleError::Invalid(
            "Bennett-Chandler kT must be finite and positive".into(),
        ));
    }
    if !(config.mass.is_finite() && config.mass > 0.0) {
        return Err(SaddleError::Invalid(
            "Bennett-Chandler mass must be finite and positive".into(),
        ));
    }
    if !(config.friction.is_finite() && config.friction >= 0.0) {
        return Err(SaddleError::Invalid(
            "Bennett-Chandler friction must be finite and non-negative".into(),
        ));
    }
    if !(config.time_step.is_finite() && config.time_step > 0.0) || config.steps == 0 {
        return Err(SaddleError::Invalid(
            "Bennett-Chandler needs a positive time step and at least one step".into(),
        ));
    }
    if config.record_every == 0 || config.shots == 0 || config.blocks < 2 {
        return Err(SaddleError::Invalid(
            "Bennett-Chandler needs a record interval, one shot, and at least two blocks".into(),
        ));
    }
    if config.shots < config.blocks {
        return Err(SaddleError::Invalid(
            "Bennett-Chandler needs at least one shot per block".into(),
        ));
    }
    let n = reference.len();
    if tangent.len() != n || starts.iter().any(|start| start.len() != n) {
        return Err(SaddleError::Shape(
            "Bennett-Chandler reference, tangent and starts must share a length".into(),
        ));
    }
    let tangent_norm = tangent.dot(&tangent).sqrt();
    if !(tangent_norm.is_finite() && tangent_norm > 0.0) {
        return Err(SaddleError::Invalid(
            "Bennett-Chandler tangent vanishes".into(),
        ));
    }
    let tangent = &tangent / tangent_norm;
    let n_record = config.steps.div_ceil(config.record_every);
    if config.plateau_record >= n_record {
        return Err(SaddleError::Invalid(
            "Bennett-Chandler plateau record is past the trajectory".into(),
        ));
    }
    let mut rng = Rng::new(config.seed);
    let sigma_v = (config.thermal / config.mass).sqrt();
    let mut weights = vec![0.0; config.shots];
    let mut side = vec![vec![0.0; n_record]; config.shots];
    let mut forward = 0usize;
    for shot in 0..config.shots {
        let start = &starts[shot % starts.len()];
        let mut velocity = Array1::zeros(n);
        for value in velocity.iter_mut() {
            *value = sigma_v * rng.normal();
        }
        let v_parallel = velocity.dot(&tangent);
        if v_parallel > 0.0 {
            forward += 1;
        }
        weights[shot] = v_parallel;
        let mut x = start.clone();
        let mut force = force_of(surface, x.view())?;
        let half = 0.5 * config.time_step / config.mass;
        for step in 0..config.steps {
            velocity = &velocity + &(&force * half);
            x = &x + &(&velocity * (0.5 * config.time_step));
            if config.friction > 0.0 {
                let decay = (-config.friction * config.time_step).exp();
                let noise = (config.thermal / config.mass * (1.0 - decay * decay)).sqrt();
                for value in velocity.iter_mut() {
                    *value = decay * *value + noise * rng.normal();
                }
            }
            x = &x + &(&velocity * (0.5 * config.time_step));
            force = force_of(surface, x.view())?;
            velocity = &velocity + &(&force * half);
            if (step + 1).is_multiple_of(config.record_every) {
                let record = (step + 1) / config.record_every - 1;
                let coordinate = (&x - &reference).dot(&tangent);
                side[shot][record] = if coordinate > 0.0 { 1.0 } else { 0.0 };
            }
        }
    }
    if forward == 0 {
        return Err(SaddleError::Invalid(
            "Bennett-Chandler drew no positive flux".into(),
        ));
    }
    let mut kappa_of_t = Vec::with_capacity(n_record);
    let mut time = Vec::with_capacity(n_record);
    for record in 0..n_record {
        let mut numer = 0.0;
        let mut denom = 0.0;
        for shot in 0..config.shots {
            numer += weights[shot] * side[shot][record];
            if weights[shot] > 0.0 {
                denom += weights[shot];
            }
        }
        kappa_of_t.push(numer / denom);
        time.push((record + 1) as f64 * config.record_every as f64 * config.time_step);
    }
    let record = config.plateau_record;
    let width = config.shots / config.blocks;
    let mut block_kappa = Vec::with_capacity(config.blocks);
    for block in 0..config.blocks {
        let mut numer = 0.0;
        let mut denom = 0.0;
        for shot in block * width..(block + 1) * width {
            numer += weights[shot] * side[shot][record];
            if weights[shot] > 0.0 {
                denom += weights[shot];
            }
        }
        if denom > 0.0 {
            block_kappa.push(numer / denom);
        }
    }
    if block_kappa.len() < 2 {
        return Err(SaddleError::Invalid(
            "Bennett-Chandler blocks have no positive flux".into(),
        ));
    }
    let mean = block_kappa.iter().sum::<f64>() / block_kappa.len() as f64;
    let variance = block_kappa
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f64>()
        / (block_kappa.len() - 1) as f64;
    Ok(Transmission {
        kappa: kappa_of_t[record],
        error: (variance / block_kappa.len() as f64).sqrt(),
        time,
        kappa_of_t,
        forward,
    })
}

fn force_of<S: PointSurface>(surface: &S, x: ArrayView1<f64>) -> Result<Array1<f64>, SaddleError> {
    let (_energy, gradient) = surface.eval(x)?;
    if gradient.len() != x.len() || !gradient.iter().all(|v| v.is_finite()) {
        return Err(SaddleError::Shape(
            "Bennett-Chandler gradient must match the coordinate".into(),
        ));
    }
    Ok(-gradient)
}

struct Rng {
    s: [u64; 4],
}

impl Rng {
    fn new(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut s = [0; 4];
        for slot in &mut s {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut mixed = z;
            mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            *slot = mixed ^ (mixed >> 31);
        }
        Self { s }
    }

    fn u64(&mut self) -> u64 {
        let result = self.s[0]
            .wrapping_add(self.s[3])
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    fn normal(&mut self) -> f64 {
        let u1 = ((self.u64() >> 11) as f64 * (1.0 / ((1u64 << 53) as f64))).max(1.0e-16);
        let u2 = (self.u64() >> 11) as f64 * (1.0 / ((1u64 << 53) as f64));
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}
