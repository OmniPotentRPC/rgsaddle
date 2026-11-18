//! The rate through the optimal hyperplane.
//!
//! In mass-weighted coordinates every direction carries unit mass, so the
//! one-way flux through a plane per unit density is `⟨(ν · q̇)₊⟩ = √(kT/2π)`,
//! the paper's Eq. 23 with `μ = 1`, and
//!
//! ```text
//! k = √(kT/2π) ρ exp(-ΔA/kT),
//! ```
//!
//! with `ρ = Q_Z / Q_R` the density of the reactant distribution on the
//! first plane of the work path and `ΔA` the [`reversible_work`] from that
//! plane to the optimal one (Eqs. 2, 3 and 21). Any [`PlaneDensity`]
//! supplies `ρ`: [`basin_density`] integrates the mean force over planes
//! parallel to the first one, and [`reactant_density`] counts the
//! crossings of that plane by one sampled basin, the paper's Eq. 22, whose
//! weight `1/|ξ̇|` has a second moment that diverges at slow crossings.
//! The two errors add in `ln k`.
//!
//! With a Cartesian unit normal `n̂` the same flux is `√(kT/2πμ)` with
//! `μ = 1 / (n̂ᵀ M⁻¹ n̂)` ([`flux_mass`]), because `n̂ · ẋ` is Gaussian with
//! variance `kT Σ nᵢ²/mᵢ`. The paper's Eq. 24, `μ = Σ mᵢ nᵢ²`, comes from
//! the density of velocities along the ray `v n̂` instead, and the two
//! agree only when the masses that `n̂` touches are equal.
//!
//! The recrossing factor `κ` of the optimal plane comes from short
//! trajectories shot from the plane's own ensemble, and
//! [`OhTstRate::transmitted`] applies it. [`transmission_coefficient`]
//! counts first crossings or final states, assigns end states by side of
//! the plane or by quench, and shoots with masses on Cartesian surfaces;
//! [`plane_transmission`] is the plain Bennett–Chandler estimate.
//!
//! [`basin_density`]: crate::ohtst::basin_density
//! [`reactant_density`]: crate::ohtst::reactant_density
//! [`reversible_work`]: crate::ohtst::reversible_work
//! [`transmission_coefficient`]: crate::ohtst::transmission_coefficient

use std::f64::consts::TAU;

use ndarray::ArrayView1;

use crate::bennett::{ShootingConfig, Transmission, bennett_chandler};
use crate::error::SaddleError;
use crate::minmode::PointSurface;
use crate::ohtst::plane::Hyperplane;
use crate::ohtst::reactant::ReactantDensity;
use crate::ohtst::sample::PlaneSample;
use crate::ohtst::transmission::TransmissionEstimate;
use crate::ohtst::work::ReversibleWork;

/// The OH-TST rate and its parts, each with its standard error.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct OhTstRate {
    /// `k`, per unit of the time that mass-weighted coordinates imply.
    pub rate: f64,
    /// First-order standard error of [`Self::rate`], `k σ(ln k)`.
    pub error: f64,
    /// `ln k`.
    pub log_rate: f64,
    /// `σ(ln k)`.
    pub log_rate_error: f64,
    /// `√(kT/2π)`.
    pub flux: f64,
    /// `Q_Z / Q_R` on the reference plane.
    pub density: f64,
    /// Its standard error.
    pub density_error: f64,
    /// `ΔA` from the reference plane to the last plane of the work path.
    pub free_energy: f64,
    /// Its standard error.
    pub free_energy_error: f64,
}

impl OhTstRate {
    /// `κ k` and its first-order standard error.
    pub fn transmitted<T: TransmissionEstimate + ?Sized>(&self, transmission: &T) -> (f64, f64) {
        let (kappa, kappa_error) = transmission.kappa();
        let error = ((kappa * self.error).powi(2) + (self.rate * kappa_error).powi(2)).sqrt();
        (kappa * self.rate, error)
    }

    /// Apply distinct path and in-plane symmetry multiplicities.
    ///
    /// A path multiplicity (for example the 12 nearest-neighbour vacancy
    /// directions in fcc) and a region degeneracy count different sets of
    /// images. Put a given symmetry orbit in exactly one factor.
    pub fn multiplied(&self, multiplicity: &RateMultiplicity) -> Result<Self, SaddleError> {
        let factor = multiplicity.total()? as f64;
        let mut rate = self.clone();
        rate.rate *= factor;
        rate.error *= factor;
        rate.log_rate += factor.ln();
        Ok(rate)
    }
}

/// Explicit, separate multiplicities of a rate.
///
/// Non-exhaustive: start from [`RateMultiplicity::default`] and use setters.
///
/// [`SymmetryUniqueRegion::from_crystal_operations`](crate::SymmetryUniqueRegion::from_crystal_operations)
/// returns the two factors already split. The path multiplicity is the
/// argument of [`Self::with_paths`]; the region's degeneracy is the in-plane
/// factor.
///
/// ```
/// use ndarray::{Array2, array};
/// use rgsaddle::{
///     CrystalSymmetryConfig, CrystalSymmetryOperation, RateMultiplicity, SymmetryUniqueRegion,
/// };
///
/// let reactant = array![[0.0, 0.0, 0.0]];
/// let saddle = array![[1.0, 0.0, 0.0]];
/// let identity = CrystalSymmetryOperation::default().with_rotation(Array2::eye(3));
/// let mirror = CrystalSymmetryOperation::default().with_rotation(array![
///     [1.0, 0.0, 0.0],
///     [0.0, -1.0, 0.0],
///     [0.0, 0.0, 1.0],
/// ]);
/// let built = SymmetryUniqueRegion::from_crystal_operations(
///     reactant.view(),
///     saddle.view(),
///     &[0],
///     &[identity, mirror],
///     &CrystalSymmetryConfig::default(),
/// )
/// .unwrap();
/// let multiplicity = RateMultiplicity::default()
///     .with_paths(built.path_multiplicity)
///     .with_region(&built.region);
/// assert_eq!(multiplicity.paths, 1);
/// assert_eq!(multiplicity.in_plane_images, 2);
/// assert_eq!(built.region_degeneracy, multiplicity.in_plane_images);
/// assert_eq!(multiplicity.total().unwrap(), 2);
/// ```
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct RateMultiplicity {
    /// Number of equivalent reaction paths or hop directions.
    pub paths: usize,
    /// Degeneracy carried by one symmetry-unique in-plane region.
    pub in_plane_images: usize,
}

impl Default for RateMultiplicity {
    fn default() -> Self {
        Self {
            paths: 1,
            in_plane_images: 1,
        }
    }
}

impl RateMultiplicity {
    /// Set [`Self::paths`].
    #[must_use]
    pub fn with_paths(mut self, paths: usize) -> Self {
        self.paths = paths;
        self
    }

    /// Set [`Self::in_plane_images`].
    #[must_use]
    pub fn with_in_plane_images(mut self, in_plane_images: usize) -> Self {
        self.in_plane_images = in_plane_images;
        self
    }

    /// Set the in-plane factor from a hard region.
    #[must_use]
    pub fn with_region(mut self, region: &dyn crate::Region) -> Self {
        self.in_plane_images = region.degeneracy();
        self
    }

    /// Product of the two distinct factors.
    pub fn total(&self) -> Result<usize, SaddleError> {
        if self.paths == 0 || self.in_plane_images == 0 {
            return Err(SaddleError::Invalid(
                "rate multiplicities must be positive".into(),
            ));
        }
        self.paths
            .checked_mul(self.in_plane_images)
            .ok_or_else(|| SaddleError::Invalid("rate multiplicity overflow".into()))
    }
}

/// How close two unit normals must be before the planes are the same.
/// A normal is dimensionless, so this is an absolute gap, not a length.
const SAME_NORMAL_GAP: f64 = 1.0e-12;

/// A plane matches when its offset is within this fraction of `1 + |point|`.
/// The scale follows the plane away from the origin instead of fixing a length.
const SAME_OFFSET_FRACTION: f64 = 1.0e-9;

/// Temperatures match when they differ by less than this fraction of `kT`.
const SAME_TEMPERATURE_FRACTION: f64 = 1.0e-12;

fn same_plane(a: &Hyperplane, b: &Hyperplane) -> bool {
    let scale = 1.0 + a.point().dot(&a.point()).sqrt();
    a.dim() == b.dim()
        && a.normal().dot(&b.normal()) > 1.0 - SAME_NORMAL_GAP
        && a.distance(b.point()).abs() <= SAME_OFFSET_FRACTION * scale
}

/// An estimate of `Z_H / Z_R` per unit mass-weighted length along the
/// normal of a plane in the reactant basin, the factor [`ohtst_rate`]
/// takes.
pub trait PlaneDensity {
    /// The plane.
    fn plane(&self) -> &Hyperplane;
    /// The sampling temperature `kT`.
    fn thermal(&self) -> f64;
    /// The density and its standard error.
    fn density(&self) -> (f64, f64);
}

impl PlaneDensity for ReactantDensity {
    fn plane(&self) -> &Hyperplane {
        &self.plane
    }

    fn thermal(&self) -> f64 {
        self.thermal
    }

    fn density(&self) -> (f64, f64) {
        (self.density, self.density_error)
    }
}

/// Combine the reactant density on a reference plane with the work from
/// that plane to the optimal one. Both must come from the same `kT`, and
/// the work path must start at the density's plane. The density is either
/// [`BasinDensity`](crate::ohtst::BasinDensity), from parallel planes, or
/// [`ReactantDensity`], from one sampled basin.
pub fn ohtst_rate<D: PlaneDensity + ?Sized>(
    density: &D,
    work: &ReversibleWork,
) -> Result<OhTstRate, SaddleError> {
    let kt = density.thermal();
    let (value, value_error) = density.density();
    if (work.thermal - kt).abs() > SAME_TEMPERATURE_FRACTION * kt {
        return Err(SaddleError::Invalid(
            "the reactant density and the work come from different temperatures".into(),
        ));
    }
    let start = work
        .nodes
        .first()
        .ok_or_else(|| SaddleError::Invalid("the work path has no nodes".into()))?;
    if !same_plane(&start.plane, density.plane()) {
        return Err(SaddleError::Invalid(
            "the work path does not start at the plane of the reactant density".into(),
        ));
    }
    if !(value.is_finite() && value > 0.0) {
        return Err(SaddleError::Invalid(format!(
            "the reactant density {value} ± {value_error} is not positive; sample longer or move the plane toward the minimum"
        )));
    }
    rate_from_parts(kt, value, value_error, work.work, work.error)
}

/// `k = √(kT/2π) ρ exp(-ΔA/kT)` from the parts [`ohtst_rate`] multiplies.
///
/// `ρ` is the reactant density on the first plane of the work path and
/// `ΔA` is the reversible work from that plane. The caller has already
/// checked that both belong to the same temperature and the same plane.
pub fn rate_from_parts(
    thermal: f64,
    density: f64,
    density_error: f64,
    free_energy: f64,
    free_energy_error: f64,
) -> Result<OhTstRate, SaddleError> {
    if !(thermal.is_finite() && thermal > 0.0) {
        return Err(SaddleError::Invalid(
            "kT must be finite and positive".into(),
        ));
    }
    if !(density.is_finite() && density > 0.0) {
        return Err(SaddleError::Invalid(
            "the reactant density must be finite and positive".into(),
        ));
    }
    if !(density_error.is_finite() && free_energy.is_finite() && free_energy_error.is_finite()) {
        return Err(SaddleError::NonFinite("rate"));
    }
    if density_error < 0.0 || free_energy_error < 0.0 {
        return Err(SaddleError::Invalid(
            "standard errors of the density and the free energy must be non-negative".into(),
        ));
    }
    let flux = (thermal / TAU).sqrt();
    let log_rate = flux.ln() + density.ln() - free_energy / thermal;
    let log_rate_error =
        ((density_error / density).powi(2) + (free_energy_error / thermal).powi(2)).sqrt();
    let rate = log_rate.exp();
    Ok(OhTstRate {
        rate,
        error: rate * log_rate_error,
        log_rate,
        log_rate_error,
        flux,
        density,
        density_error,
        free_energy,
        free_energy_error,
    })
}

/// The Bennett–Chandler transmission coefficient of a sampled plane, shot
/// along its normal from the configurations the sample retained (see
/// [`LangevinConfig::retain`](crate::ohtst::LangevinConfig::retain)). The
/// surface is the mass-weighted one the plane was sampled on, so every
/// degree of freedom has unit mass.
pub fn plane_transmission<S: PointSurface>(
    surface: &S,
    sample: &PlaneSample,
    config: &ShootingConfig,
) -> Result<Transmission, SaddleError> {
    if config.mass != 1.0 {
        return Err(SaddleError::Invalid(
            "mass-weighted coordinates carry unit mass; set ShootingConfig::mass to 1".into(),
        ));
    }
    if sample.retained.is_empty() {
        return Err(SaddleError::Invalid(
            "the plane sample kept no configurations; raise LangevinConfig::retain".into(),
        ));
    }
    bennett_chandler(
        surface,
        &sample.retained,
        sample.plane.point(),
        sample.plane.normal(),
        config,
    )
}

/// The mass for flux along the Cartesian direction `normal`, normalized
/// here: `1 / Σ nᵢ²/mᵢ` with one mass per coordinate.
pub fn flux_mass(normal: ArrayView1<f64>, masses: ArrayView1<f64>) -> Result<f64, SaddleError> {
    if normal.len() != masses.len() || normal.is_empty() {
        return Err(SaddleError::Shape(
            "a normal and its masses differ in length".into(),
        ));
    }
    if !masses.iter().all(|m| m.is_finite() && *m > 0.0) {
        return Err(SaddleError::Invalid(
            "masses must be finite and positive".into(),
        ));
    }
    let norm2 = normal.dot(&normal);
    if !(norm2.is_finite() && norm2 > 0.0) {
        return Err(SaddleError::Invalid("the normal vanishes".into()));
    }
    let inverse: f64 = normal
        .iter()
        .zip(masses.iter())
        .map(|(n, m)| n * n / m)
        .sum::<f64>()
        / norm2;
    Ok(1.0 / inverse)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::minmode::PointSurface;
    use crate::ohtst::sample::{LangevinConfig, sample_plane};
    use crate::ohtst::work::{WorkEstimate, WorkNode};
    use ndarray::{Array1, array};

    fn simpson(f: impl Fn(f64) -> f64, a: f64, b: f64, steps: usize) -> f64 {
        let h = (b - a) / steps as f64;
        let mut sum = f(a) + f(b);
        for i in 1..steps {
            sum += f(a + i as f64 * h) * if i % 2 == 1 { 4.0 } else { 2.0 };
        }
        sum * h / 3.0
    }

    /// `⟨(n̂ · v)₊⟩` for `vᵢ ~ N(0, kT/mᵢ)` in two dimensions, with
    /// `n̂₀ > 0`. The inner integral starts where `n̂ · v` turns positive, so
    /// both integrands are smooth.
    fn mean_positive_flux(normal: [f64; 2], masses: [f64; 2], kt: f64) -> f64 {
        let c = normal[0] * (kt / masses[0]).sqrt();
        let d = normal[1] * (kt / masses[1]).sqrt();
        let phi = |t: f64| (-0.5 * t * t).exp() / TAU.sqrt();
        simpson(
            |b| {
                let lo = (-d * b / c).max(-12.0);
                if lo >= 12.0 {
                    0.0
                } else {
                    phi(b) * simpson(|a| (c * a + d * b) * phi(a), lo, 12.0, 2000)
                }
            },
            -12.0,
            12.0,
            400,
        )
    }

    #[test]
    fn flux_mass_is_the_marginal_of_the_normal_velocity() {
        let normal = [0.5f64.sqrt(), 0.5f64.sqrt()];
        let masses = [1.0, 3.0];
        let mu = flux_mass(array![1.0, 1.0].view(), array![1.0, 3.0].view()).unwrap();
        assert!((mu - 1.5).abs() < 1e-14);
        let kt = 0.7;
        let flux = mean_positive_flux(normal, masses, kt);
        assert!((flux - (kt / (TAU * mu)).sqrt()).abs() < 1e-8);
        let eq24: f64 = masses.iter().zip(normal).map(|(m, n)| m * n * n).sum();
        assert!((eq24 - 2.0).abs() < 1e-14);
        assert!((flux - (kt / (TAU * eq24)).sqrt()).abs() > 1e-2);
        let equal = flux_mass(array![0.6, 0.8].view(), array![2.0, 2.0].view()).unwrap();
        assert!((equal - 2.0).abs() < 1e-14);
    }

    /// `V = -½ x² + y²`. It separates, so a trajectory that leaves the line
    /// `x = 0` never returns and `κ = 1` there.
    struct Barrier;

    impl PointSurface for Barrier {
        fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            Ok((-0.5 * q[0] * q[0] + q[1] * q[1], array![-q[0], 2.0 * q[1]]))
        }
    }

    #[test]
    fn transmission_is_one_on_the_separable_dividing_line() {
        let kt = 0.2;
        let config = LangevinConfig::default()
            .with_thermal(kt)
            .with_time_step(0.2)
            .with_records(400)
            .with_retain(64);
        let shooting = ShootingConfig::default()
            .with_thermal(kt)
            .with_time_step(0.02)
            .with_steps(300)
            .with_record_every(50)
            .with_shots(256)
            .with_blocks(8)
            .with_plateau_record(5);
        let line = Hyperplane::new(array![0.0, 0.0], array![1.0, 0.0]).unwrap();
        let sample = sample_plane(&Barrier, &line, None, &config).unwrap();
        let kappa = plane_transmission(&Barrier, &sample, &shooting).unwrap();
        assert!((kappa.kappa - 1.0).abs() < 1e-12 && kappa.error < 1e-12);
        let tilted = Hyperplane::new(array![0.0, 0.0], array![0.5f64.cos(), 0.5f64.sin()]).unwrap();
        let sample = sample_plane(&Barrier, &tilted, None, &config).unwrap();
        let kappa = plane_transmission(&Barrier, &sample, &shooting).unwrap();
        assert!(kappa.kappa < 1.0 - 4.0 * kappa.error, "{kappa:?}");
        let heavy = shooting.with_mass(2.0);
        assert!(plane_transmission(&Barrier, &sample, &heavy).is_err());
    }

    struct Bowl;

    impl PointSurface for Bowl {
        fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            Ok((0.5 * q.dot(&q), q.to_owned()))
        }
    }

    #[test]
    fn rate_combines_density_and_work() {
        let plane = Hyperplane::new(array![-0.5, 0.0], array![1.0, 0.0]).unwrap();
        let config = LangevinConfig::default().with_records(200);
        let end = sample_plane(&Bowl, &plane, None, &config).unwrap();
        let node = WorkNode {
            leg: 0,
            s: 0.0,
            plane: plane.clone(),
            derivative: 0.0,
            error: 0.0,
            work: 0.0,
            bar_work: None,
            overlap: None,
            transport_weight: None,
            mbar_work: None,
        };
        let kt = 0.5;
        let work = ReversibleWork {
            work: 3.0,
            error: 0.1,
            estimate: WorkEstimate::Integration,
            integrated_work: 3.0,
            integrated_error: 0.1,
            nodes: vec![node],
            end,
            evaluations: 0,
            thermal: kt,
            bar_work: None,
            bar_error: None,
            mbar_work: None,
            mbar_error: None,
            overlap: None,
            warnings: Vec::new(),
        };
        let density = ReactantDensity {
            plane: plane.clone(),
            thermal: kt,
            density: 0.4,
            density_error: 0.02,
            crossing_density: 0.4,
            crossing_error: 0.05,
            behind: 0.3,
            tau: 1.0,
            records: 1,
            evaluations: 0,
            acceptance: 1.0,
            wall_hits: 0,
            last: array![0.0, 0.0],
        };
        let rate = ohtst_rate(&density, &work).unwrap();
        let expected = (kt / TAU).sqrt() * 0.4 * (-3.0f64 / kt).exp();
        let half = Transmission {
            time: vec![1.0],
            kappa_of_t: vec![0.5],
            kappa: 0.5,
            error: 0.05,
            forward: 1,
        };
        let (transmitted, transmitted_error) = rate.transmitted(&half);
        assert!((transmitted / expected - 0.5).abs() < 1e-12);
        let relative = ((0.02f64 / 0.4).powi(2) + (0.1f64 / kt).powi(2) + 0.1f64.powi(2)).sqrt();
        assert!((transmitted_error / transmitted - relative).abs() < 1e-12);
        assert!((rate.rate / expected - 1.0).abs() < 1e-12);
        let sigma = ((0.02f64 / 0.4).powi(2) + (0.1f64 / kt).powi(2)).sqrt();
        assert!((rate.log_rate_error - sigma).abs() < 1e-12);
        assert!((rate.error - expected * sigma).abs() < 1e-12 * expected);

        let mut hot = density.clone();
        hot.thermal = 0.6;
        assert!(ohtst_rate(&hot, &work).is_err());
        let mut elsewhere = density.clone();
        elsewhere.plane = Hyperplane::new(array![-0.4, 0.0], array![1.0, 0.0]).unwrap();
        assert!(ohtst_rate(&elsewhere, &work).is_err());
        let mut empty = density;
        empty.density = -0.01;
        assert!(ohtst_rate(&empty, &work).is_err());
        assert!(rate_from_parts(kt, 0.4, -0.02, 3.0, 0.1).is_err());
        assert!(rate_from_parts(kt, 0.4, 0.02, 3.0, -0.1).is_err());
    }

    #[test]
    fn path_and_in_plane_counts_are_not_double_applied() {
        let base = OhTstRate {
            rate: 2.0,
            error: 0.2,
            log_rate: 2.0f64.ln(),
            log_rate_error: 0.1,
            flux: 1.0,
            density: 1.0,
            density_error: 0.0,
            free_energy: 0.0,
            free_energy_error: 0.0,
        };
        let twelve_paths = base
            .multiplied(&RateMultiplicity::default().with_paths(12))
            .unwrap();
        let twelve_regions = base
            .multiplied(&RateMultiplicity::default().with_in_plane_images(12))
            .unwrap();
        assert_eq!(twelve_paths.rate, twelve_regions.rate);
        assert_eq!(twelve_paths.error, twelve_regions.error);
        assert_eq!(twelve_paths.rate, 24.0);
    }
}
