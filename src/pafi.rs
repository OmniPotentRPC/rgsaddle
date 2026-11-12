//! Projected average force integration along a path.
//!
//! Swinburne and Marinica, Phys. Rev. Lett. 120, 135503 (2018),
//! doi:10.1103/PhysRevLett.120.135503, give a path-based free-energy
//! gradient that does not need a collective variable. The same estimator
//! is what LAMMPS `fix pafi` averages (Swinburne's implementation of that
//! paper): with `n = dX/ds` the path derivative, `dn/ds` its derivative,
//! and `ψ = (x - X) · dn/ds`,
//!
//! `dF/ds = ⟨ (∇E · n) (1 - ψ) ⟩`.
//!
//! The average is over the hyperplane `(x - X) · n = 0`. Sampling here is
//! overdamped Langevin (Brownian) dynamics in that plane. The parallel
//! piece of the drift and of the noise is removed, and the plane is
//! restored after every step. A center-of-mass projection is optional and
//! is the `com yes` choice of that fix. The integral of `dF/ds` from the
//! reactant is the potential of mean force. Block averages of the recorded
//! samples supply the standard error.

use ndarray::{Array1, ArrayView1};

use crate::error::SaddleError;
use crate::minmode::PointSurface;

/// Overdamped hyperplane sampling.
#[derive(Clone, Debug)]
pub struct PafiConfig {
    /// `kT` in the energy unit of the surface.
    pub thermal: f64,
    /// Mobility `μ` in `dx = -μ ∇E dt + sqrt(2 μ kT dt) ξ`.
    pub mobility: f64,
    /// Overdamped time step.
    pub time_step: f64,
    /// Steps discarded before the first recorded sample.
    pub thermalize: usize,
    /// Steps between recorded samples.
    pub stride: usize,
    /// Recorded samples. At least twice [`Self::blocks`].
    pub samples: usize,
    /// Contiguous blocks used for the standard error. At least 2.
    pub blocks: usize,
    /// Seed of the noise.
    pub seed: u64,
    /// Remove the mean Cartesian drift when the system has at least two atoms.
    pub fix_com: bool,
    /// How many recorded configurations to keep, evenly spaced.
    pub retain: usize,
}

impl Default for PafiConfig {
    fn default() -> Self {
        Self {
            thermal: 0.02585,
            mobility: 1.0,
            time_step: 1.0e-3,
            thermalize: 200,
            stride: 5,
            samples: 200,
            blocks: 5,
            seed: 1,
            fix_com: true,
            retain: 16,
        }
    }
}

impl PafiConfig {
    fn check(&self) -> Result<(), SaddleError> {
        if !(self.thermal.is_finite() && self.thermal > 0.0) {
            return Err(SaddleError::Invalid(
                "PAFI kT must be finite and positive".into(),
            ));
        }
        if !(self.mobility.is_finite() && self.mobility > 0.0) {
            return Err(SaddleError::Invalid(
                "PAFI mobility must be finite and positive".into(),
            ));
        }
        if !(self.time_step.is_finite() && self.time_step > 0.0) {
            return Err(SaddleError::Invalid(
                "PAFI time step must be finite and positive".into(),
            ));
        }
        if self.stride == 0 || self.blocks < 2 || self.samples < self.blocks * 2 {
            return Err(SaddleError::Invalid(
                "PAFI needs a positive stride, at least two blocks, and two samples per block"
                    .into(),
            ));
        }
        Ok(())
    }
}

/// One frame of a piecewise path, with its arc length from the reactant.
#[derive(Clone, Debug)]
pub struct PathFrame {
    pub positions: Array1<f64>,
    pub arc: f64,
}

/// Cubic Hermite path. The parameter is the polyline arc length.
#[derive(Clone, Debug)]
pub struct ArcPath {
    frames: Vec<PathFrame>,
    tangents: Vec<Array1<f64>>,
}

/// Position, path derivative `n = dX/ds`, and `dn/ds` at one arc length.
#[derive(Clone, Debug)]
pub struct PathGeometry {
    pub positions: Array1<f64>,
    pub tangent: Array1<f64>,
    pub curvature: Array1<f64>,
    pub arc: f64,
}

impl ArcPath {
    /// Build the path from ordered frames. Arc lengths must strictly increase.
    pub fn new(frames: Vec<PathFrame>) -> Result<Self, SaddleError> {
        if frames.len() < 2 {
            return Err(SaddleError::Invalid(
                "a PAFI path needs at least two frames".into(),
            ));
        }
        let n = frames[0].positions.len();
        if n == 0 {
            return Err(SaddleError::Shape("PAFI path frame is empty".into()));
        }
        for (i, frame) in frames.iter().enumerate() {
            if frame.positions.len() != n || !frame.positions.iter().all(|v| v.is_finite()) {
                return Err(SaddleError::Shape(
                    "PAFI path frames must share one finite length".into(),
                ));
            }
            if !frame.arc.is_finite() || (i > 0 && frame.arc <= frames[i - 1].arc) {
                return Err(SaddleError::Invalid(
                    "PAFI arc length must strictly increase".into(),
                ));
            }
        }
        let mut tangents = Vec::with_capacity(frames.len());
        for i in 0..frames.len() {
            let tangent = if i == 0 {
                difference(
                    &frames[1].positions,
                    &frames[0].positions,
                    frames[1].arc - frames[0].arc,
                )
            } else if i + 1 == frames.len() {
                difference(
                    &frames[i].positions,
                    &frames[i - 1].positions,
                    frames[i].arc - frames[i - 1].arc,
                )
            } else {
                difference(
                    &frames[i + 1].positions,
                    &frames[i - 1].positions,
                    frames[i + 1].arc - frames[i - 1].arc,
                )
            };
            if tangent.iter().all(|v| *v == 0.0) {
                return Err(SaddleError::Invalid(
                    "a PAFI frame has a zero tangent".into(),
                ));
            }
            tangents.push(tangent);
        }
        Ok(Self { frames, tangents })
    }

    /// Frames in order, reactant first.
    pub fn from_positions(positions: &[Array1<f64>]) -> Result<Self, SaddleError> {
        if positions.len() < 2 {
            return Err(SaddleError::Invalid(
                "a PAFI path needs at least two frames".into(),
            ));
        }
        let mut arc = 0.0;
        let mut frames = Vec::with_capacity(positions.len());
        frames.push(PathFrame {
            positions: positions[0].clone(),
            arc,
        });
        for pair in positions.windows(2) {
            let step = pair[1]
                .iter()
                .zip(pair[0].iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();
            if !(step.is_finite() && step > 0.0) {
                return Err(SaddleError::Invalid(
                    "PAFI frames must be finite and distinct".into(),
                ));
            }
            arc += step;
            frames.push(PathFrame {
                positions: pair[1].clone(),
                arc,
            });
        }
        Self::new(frames)
    }

    /// Arc length of the reactant and of the product.
    pub fn bounds(&self) -> (f64, f64) {
        (
            self.frames[0].arc,
            self.frames.last().expect("frames").arc,
        )
    }

    /// Arc length of every stored frame.
    pub fn arcs(&self) -> Vec<f64> {
        self.frames.iter().map(|frame| frame.arc).collect()
    }

    /// Hermite geometry at `arc`, clamped to the path.
    pub fn geometry(&self, arc: f64) -> Result<PathGeometry, SaddleError> {
        if !arc.is_finite() {
            return Err(SaddleError::NonFinite("PAFI arc"));
        }
        let last = self.frames.len() - 1;
        let arc = arc.clamp(self.frames[0].arc, self.frames[last].arc);
        let mut seg = last - 1;
        for i in 0..last {
            if arc <= self.frames[i + 1].arc {
                seg = i;
                break;
            }
        }
        let s0 = self.frames[seg].arc;
        let s1 = self.frames[seg + 1].arc;
        let dt = s1 - s0;
        let t = ((arc - s0) / dt).clamp(0.0, 1.0);
        let (h00, h10, h01, h11, d00, d10, d01, d11, c00, c10, c01, c11) = hermite(t);
        let x0 = &self.frames[seg].positions;
        let x1 = &self.frames[seg + 1].positions;
        let m0 = &self.tangents[seg];
        let m1 = &self.tangents[seg + 1];
        let n = x0.len();
        let mut position = Array1::zeros(n);
        let mut tangent = Array1::zeros(n);
        let mut curvature = Array1::zeros(n);
        for i in 0..n {
            position[i] = h00 * x0[i] + h10 * dt * m0[i] + h01 * x1[i] + h11 * dt * m1[i];
            let d_dt = d00 * x0[i] + d10 * dt * m0[i] + d01 * x1[i] + d11 * dt * m1[i];
            let d2_dt = c00 * x0[i] + c10 * dt * m0[i] + c01 * x1[i] + c11 * dt * m1[i];
            tangent[i] = d_dt / dt;
            curvature[i] = d2_dt / (dt * dt);
        }
        Ok(PathGeometry {
            positions: position,
            tangent,
            curvature,
            arc,
        })
    }
}

/// Mean free-energy gradient on one hyperplane.
#[derive(Clone, Debug)]
pub struct HyperplaneAverage {
    /// `dF/ds` in energy per arc length.
    pub gradient: f64,
    /// Standard error from the block means.
    pub error: f64,
    /// Mean of `ψ = (x - X) · dn/ds`.
    pub psi: f64,
    pub samples: usize,
    /// Configurations kept from the production run, on the plane.
    pub retained: Vec<Array1<f64>>,
}

/// Potential of mean force from the first plane.
#[derive(Clone, Debug)]
pub struct FreeEnergyProfile {
    pub arc: Vec<f64>,
    pub gradient: Vec<f64>,
    pub gradient_error: Vec<f64>,
    pub free_energy: Vec<f64>,
    pub free_energy_error: Vec<f64>,
    /// Maximum of the potential of mean force. The reactant is zero.
    pub barrier: f64,
    pub barrier_error: f64,
    /// Index of that maximum.
    pub dividing: usize,
}

/// Sample one hyperplane and return the PAFI gradient.
pub fn average_hyperplane<S: PointSurface>(
    surface: &S,
    geometry: &PathGeometry,
    config: &PafiConfig,
) -> Result<HyperplaneAverage, SaddleError> {
    config.check()?;
    let n = geometry.positions.len();
    if geometry.tangent.len() != n || geometry.curvature.len() != n {
        return Err(SaddleError::Shape(
            "PAFI tangent and curvature must match the position".into(),
        ));
    }
    if !geometry
        .tangent
        .iter()
        .chain(geometry.curvature.iter())
        .all(|v| v.is_finite())
    {
        return Err(SaddleError::NonFinite("PAFI path geometry"));
    }
    let tangent_norm = norm(&geometry.tangent);
    if !(tangent_norm.is_finite() && tangent_norm > 0.0) {
        return Err(SaddleError::Invalid("PAFI tangent vanishes".into()));
    }
    let mut x = geometry.positions.clone();
    let mut rng = Rng::new(config.seed);
    let sigma = (2.0 * config.mobility * config.thermal * config.time_step).sqrt();
    let drift_scale = config.mobility * config.time_step;
    let total = config.thermalize + config.stride * config.samples;
    let mut observations = Vec::with_capacity(config.samples);
    let mut psi_sum = 0.0;
    let mut retained = Vec::new();
    let retain_gap = config
        .samples
        .div_ceil(config.retain.max(1))
        .max(1);
    for step in 0..total {
        let (_energy, gradient) = surface.eval(x.view())?;
        if gradient.len() != n || !gradient.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::Shape(
                "PAFI gradient must match the path".into(),
            ));
        }
        if step >= config.thermalize && (step - config.thermalize).is_multiple_of(config.stride) {
            let (value, psi) = projected_gradient(&gradient, &x, geometry);
            if !value.is_finite() || !psi.is_finite() {
                return Err(SaddleError::NonFinite("PAFI projected gradient"));
            }
            let recorded = observations.len();
            observations.push(value);
            psi_sum += psi;
            if config.retain > 0 && recorded.is_multiple_of(retain_gap) && retained.len() < config.retain
            {
                retained.push(x.clone());
            }
        }
        let mut drift = gradient;
        project_out(&mut drift, geometry.tangent.view());
        let mut noise = Array1::zeros(n);
        for value in noise.iter_mut() {
            *value = rng.normal();
        }
        project_out(&mut noise, geometry.tangent.view());
        if config.fix_com && n >= 6 && n.is_multiple_of(3) {
            remove_com(&mut drift);
            remove_com(&mut noise);
            project_out(&mut drift, geometry.tangent.view());
            project_out(&mut noise, geometry.tangent.view());
        }
        x = &x - &(&drift * drift_scale) + &(&noise * sigma);
        let nn = geometry.tangent.dot(&geometry.tangent);
        let slip = (&x - &geometry.positions).dot(&geometry.tangent) / nn;
        x = &x - &(&geometry.tangent * slip);
    }
    if observations.len() != config.samples {
        return Err(SaddleError::Invalid(
            "PAFI did not record the requested samples".into(),
        ));
    }
    let (gradient, error) = block_mean(&observations, config.blocks);
    Ok(HyperplaneAverage {
        gradient,
        error,
        psi: psi_sum / config.samples as f64,
        samples: config.samples,
        retained,
    })
}

/// Integrate plane gradients with the trapezoid rule.
///
/// The dividing surface is the interior maximum of the potential of mean
/// force. With only two planes the maximum of the two is used.
pub fn integrate_profile(
    arcs: &[f64],
    gradients: &[f64],
    errors: &[f64],
) -> Result<FreeEnergyProfile, SaddleError> {
    if arcs.len() < 2 || arcs.len() != gradients.len() || arcs.len() != errors.len() {
        return Err(SaddleError::Shape(
            "PAFI integration needs matching arcs, gradients and errors".into(),
        ));
    }
    let mut free = vec![0.0; arcs.len()];
    for i in 1..arcs.len() {
        let ds = arcs[i] - arcs[i - 1];
        if !(ds.is_finite() && ds > 0.0) {
            return Err(SaddleError::Invalid(
                "PAFI arcs must strictly increase".into(),
            ));
        }
        if !gradients[i].is_finite() || !gradients[i - 1].is_finite() {
            return Err(SaddleError::NonFinite("PAFI gradient"));
        }
        free[i] = free[i - 1] + 0.5 * (gradients[i - 1] + gradients[i]) * ds;
    }
    let free_error = prefix_errors(arcs, errors)?;
    let dividing = dividing_index(&free);
    Ok(FreeEnergyProfile {
        barrier: free[dividing],
        barrier_error: free_error[dividing],
        dividing,
        arc: arcs.to_vec(),
        gradient: gradients.to_vec(),
        gradient_error: errors.to_vec(),
        free_energy: free,
        free_energy_error: free_error,
    })
}

/// Sample every requested arc and integrate.
pub fn pafi_profile<S: PointSurface>(
    surface: &S,
    path: &ArcPath,
    arcs: &[f64],
    config: &PafiConfig,
) -> Result<(FreeEnergyProfile, Vec<HyperplaneAverage>), SaddleError> {
    if arcs.len() < 2 {
        return Err(SaddleError::Invalid(
            "PAFI needs at least two hyperplanes".into(),
        ));
    }
    let mut planes = Vec::with_capacity(arcs.len());
    for (index, arc) in arcs.iter().enumerate() {
        let geometry = path.geometry(*arc)?;
        let mut local = config.clone();
        local.seed = config.seed.wrapping_add(index as u64).wrapping_add(1);
        planes.push(average_hyperplane(surface, &geometry, &local)?);
    }
    let gradients: Vec<f64> = planes.iter().map(|plane| plane.gradient).collect();
    let errors: Vec<f64> = planes.iter().map(|plane| plane.error).collect();
    let profile = integrate_profile(arcs, &gradients, &errors)?;
    Ok((profile, planes))
}

fn prefix_errors(arcs: &[f64], errors: &[f64]) -> Result<Vec<f64>, SaddleError> {
    let mut out = vec![0.0; arcs.len()];
    for i in 1..arcs.len() {
        let mut used = vec![0.0; arcs.len()];
        for k in 1..=i {
            let span = arcs[k] - arcs[k - 1];
            used[k - 1] += 0.5 * span;
            used[k] += 0.5 * span;
        }
        let mut variance = 0.0;
        for j in 0..=i {
            let sigma = errors[j];
            if !(sigma.is_finite() && sigma >= 0.0) {
                return Err(SaddleError::Invalid(
                    "PAFI gradient error must be finite and non-negative".into(),
                ));
            }
            variance += used[j] * used[j] * sigma * sigma;
        }
        out[i] = variance.sqrt();
    }
    Ok(out)
}

fn dividing_index(free: &[f64]) -> usize {
    let mut best = 0;
    for i in 1..free.len() {
        if free[i] > free[best] {
            best = i;
        }
    }
    best
}

fn projected_gradient(gradient: &Array1<f64>, x: &Array1<f64>, geometry: &PathGeometry) -> (f64, f64) {
    let psi = (x - &geometry.positions).dot(&geometry.curvature);
    let parallel = gradient.dot(&geometry.tangent);
    (parallel * (1.0 - psi), psi)
}

fn difference(end: &Array1<f64>, start: &Array1<f64>, ds: f64) -> Array1<f64> {
    (end - start) / ds
}

fn hermite(t: f64) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64) {
    let t2 = t * t;
    let t3 = t2 * t;
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;
    let d00 = 6.0 * t2 - 6.0 * t;
    let d10 = 3.0 * t2 - 4.0 * t + 1.0;
    let d01 = -6.0 * t2 + 6.0 * t;
    let d11 = 3.0 * t2 - 2.0 * t;
    let c00 = 12.0 * t - 6.0;
    let c10 = 6.0 * t - 4.0;
    let c01 = -12.0 * t + 6.0;
    let c11 = 6.0 * t - 2.0;
    (h00, h10, h01, h11, d00, d10, d01, d11, c00, c10, c01, c11)
}

fn project_out(vector: &mut Array1<f64>, tangent: ArrayView1<f64>) {
    let nn = tangent.dot(&tangent);
    if nn == 0.0 {
        return;
    }
    let coef = vector.dot(&tangent) / nn;
    *vector -= &(tangent.to_owned() * coef);
}

pub(crate) fn remove_com(vector: &mut Array1<f64>) {
    let n = vector.len() / 3;
    if n == 0 {
        return;
    }
    let mut mean = [0.0; 3];
    for atom in 0..n {
        for c in 0..3 {
            mean[c] += vector[3 * atom + c];
        }
    }
    for c in 0..3 {
        mean[c] /= n as f64;
    }
    for atom in 0..n {
        for c in 0..3 {
            vector[3 * atom + c] -= mean[c];
        }
    }
}

fn norm(vector: &Array1<f64>) -> f64 {
    vector.dot(vector).sqrt()
}

fn block_mean(values: &[f64], blocks: usize) -> (f64, f64) {
    let width = values.len() / blocks;
    let mut means = Vec::with_capacity(blocks);
    for block in 0..blocks {
        let slice = &values[block * width..(block + 1) * width];
        let mean = slice.iter().sum::<f64>() / slice.len() as f64;
        means.push(mean);
    }
    let mean = means.iter().sum::<f64>() / blocks as f64;
    let variance = means
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f64>()
        / (blocks - 1) as f64;
    (mean, (variance / blocks as f64).sqrt())
}

/// xoshiro256++. The stream is deterministic for a seed.
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
