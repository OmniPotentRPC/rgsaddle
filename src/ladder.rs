//! A rate ladder above harmonic transition state theory.
//!
//! The rungs and the stop/adopt rule live in this module and name no
//! catalog type. `anharmonic_correction`
//! supplies `log φ`, `ohtst_rate` supplies a sampled
//! plane rate, and `TransmissionEstimate`
//! supplies `κ`. The rule decides whether the next rung replaces the
//! current rate.
//!
//! `z` is computed only once the relative error is at or under the target,
//! or the sample is already at the cap.
//!
//! ```
//! use rgsaddle::ladder::{Climb, Comparison, Decision, LadderConfig, decide};
//!
//! let decision = decide(
//!     &LadderConfig::default(),
//!     Comparison::Transmission {
//!         kappa: 0.5,
//!         error: 0.0078125,
//!         n: 32,
//!     },
//! );
//! assert!(matches!(
//!     decision,
//!     Ok(Decision {
//!         climb: Climb::Adopt,
//!         z: Some(-64.0),
//!     })
//! ));
//! ```

use crate::SaddleError;
use crate::stats::Rng;

/// Published OH-TST rate at the center of the Jóhannesson and Jónsson band.
pub const OHTST_PAPER_RATE: f64 = 4.5e-7;
/// Half-width of that band, `(4.5 ± 0.2) × 10^−7`.
pub const OHTST_PAPER_HALF_WIDTH: f64 = 0.2e-7;
/// Inclusive lower end of the published band.
pub const OHTST_PAPER_LOW: f64 = 4.3e-7;
/// Inclusive upper end of the published band.
pub const OHTST_PAPER_HIGH: f64 = 4.7e-7;

/// Sample size, thresholds, and the temperature match.
///
/// Non-exhaustive: start from [`LadderConfig::default`] and chain `with_*`.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct LadderConfig {
    /// Relative-error target `s*`.
    pub relative_error: f64,
    /// Adopt when `|z|` exceeds this.
    pub z_resolved: f64,
    /// Margin on the next sample size.
    pub growth: f64,
    /// First sample the caller draws.
    pub pilot: usize,
    /// Largest sample.
    pub cap: usize,
    /// `None` compares temperatures bit for bit. `Some(tol)` matches when
    /// the absolute gap is at most `tol`.
    pub temperature_tolerance: Option<f64>,
}

impl Default for LadderConfig {
    fn default() -> Self {
        Self {
            relative_error: 0.03,
            z_resolved: 2.0,
            growth: 1.1,
            pilot: 14,
            cap: 2048,
            temperature_tolerance: None,
        }
    }
}

with_setters!(LadderConfig {
    with_relative_error => relative_error: val f64,
    with_z_resolved => z_resolved: val f64,
    with_growth => growth: val f64,
    with_pilot => pilot: val usize,
    with_cap => cap: val usize,
    with_temperature_tolerance => temperature_tolerance: opt f64,
});

impl LadderConfig {
    fn check(&self) -> Result<(), SaddleError> {
        if !(self.relative_error.is_finite() && self.relative_error > 0.0) {
            return Err(SaddleError::Invalid(
                "the relative-error target must be finite and positive".into(),
            ));
        }
        if !(self.z_resolved.is_finite() && self.z_resolved > 0.0) {
            return Err(SaddleError::Invalid(
                "the z threshold must be finite and positive".into(),
            ));
        }
        if !(self.growth.is_finite() && self.growth >= 1.0) {
            return Err(SaddleError::Invalid(
                "the sample-size margin must be finite and at least one".into(),
            ));
        }
        if self.pilot == 0 || self.cap < self.pilot {
            return Err(SaddleError::Invalid(
                "the pilot is at least one and the cap is at least the pilot".into(),
            ));
        }
        if let Some(tol) = self.temperature_tolerance
            && !(tol.is_finite() && tol >= 0.0)
        {
            return Err(SaddleError::Invalid(
                "the temperature tolerance must be finite and non-negative".into(),
            ));
        }
        Ok(())
    }
}

/// A rate, or a factor, and the standard error of its mean.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Estimate {
    /// The value.
    pub value: f64,
    /// Standard error of [`Self::value`].
    pub error: f64,
    /// Samples that produced the estimate.
    pub n: usize,
}

/// What the next rung does to the rate in hand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Climb {
    /// The next rung does not replace the rate.
    Stop,
    /// Draw this many samples with the same seed.
    Grow {
        /// The next sample size.
        n: usize,
    },
    /// The next rung replaces the rate.
    Adopt,
    /// The sample is at the cap and the difference is unresolved.
    Open,
}

/// The climb, and `z` when the rule computed it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Decision {
    /// What to do with the next rung.
    pub climb: Climb,
    /// Absent while the sample is still growing, and when a standard error is zero.
    pub z: Option<f64>,
}

/// Which comparison [`decide`] runs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Comparison {
    /// `log φ` and its standard error, from `log_ratio`.
    Analytic {
        /// `log φ`.
        log_phi: f64,
        /// Standard error of `log φ`.
        error: f64,
        /// Probes.
        n: usize,
    },
    /// `κ` and its standard error, from `TransmissionEstimate`.
    Transmission {
        /// Transmission coefficient.
        kappa: f64,
        /// Standard error of `kappa`.
        error: f64,
        /// Shots that scored.
        n: usize,
    },
    /// A sampled rate against the rate in hand. The samples are independent.
    Independent {
        /// The rate in hand.
        current: Estimate,
        /// The sampled rate.
        next: Estimate,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    Plane,
    Transmission,
}

impl Comparison {
    fn stage(self) -> Stage {
        match self {
            Self::Analytic { .. } | Self::Independent { .. } => Stage::Plane,
            Self::Transmission { .. } => Stage::Transmission,
        }
    }

    fn n(self) -> usize {
        match self {
            Self::Analytic { n, .. } | Self::Transmission { n, .. } => n,
            Self::Independent { next, .. } => next.n,
        }
    }
}

/// The record a catalog reader stores. It has no catalog type of its own.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LadderRecord {
    /// Adopted rate, in the caller's time unit.
    pub rate: f64,
    /// Standard error of [`Self::rate`].
    pub rate_error: f64,
    /// Temperature the factors belong to.
    pub temperature_k: f64,
    /// Transmission coefficient that entered the rate, or 1 when it did not.
    pub kappa: f64,
    /// Standard error of [`Self::kappa`].
    pub kappa_error: f64,
    /// Free-energy factor that entered the rate, or 1 when it did not.
    pub anharmonic: f64,
    /// 0 harmonic, 1 plane free energy, 2 transmission.
    pub rung: u8,
    /// True when a rung reached the cap without resolving.
    pub open: bool,
}

/// Rates adopted so far, and the rung the next comparison has to finish.
#[derive(Clone, Debug)]
pub struct Ladder {
    config: LadderConfig,
    harmonic: Estimate,
    rate: Estimate,
    phi: f64,
    kappa: f64,
    kappa_error: f64,
    rung: u8,
    open: bool,
    waiting: Option<Stage>,
}

impl Ladder {
    /// Start from a harmonic rate. `n` may be zero: Vineyard is not a sample.
    pub fn start(config: LadderConfig, harmonic: Estimate) -> Result<Self, SaddleError> {
        config.check()?;
        if !(harmonic.value.is_finite() && harmonic.value > 0.0) {
            return Err(SaddleError::Invalid(
                "the harmonic rate must be finite and positive".into(),
            ));
        }
        if !(harmonic.error.is_finite() && harmonic.error >= 0.0) {
            return Err(SaddleError::Invalid(
                "the harmonic standard error must be finite and non-negative".into(),
            ));
        }
        Ok(Self {
            config,
            harmonic,
            rate: harmonic,
            phi: 1.0,
            kappa: 1.0,
            kappa_error: 0.0,
            rung: 0,
            open: false,
            waiting: None,
        })
    }

    /// Apply one comparison.
    ///
    /// The analytic factor and the sampled rate are the same rung, so a
    /// grow or an open analytic comparison does not block the sampled
    /// replacement. A grow blocks transmission until that plane rung stops
    /// or adopts. An open plane rung blocks transmission. The sampled
    /// comparison has to name the rate in hand, and a later rung is not
    /// applied backwards.
    pub fn apply(&mut self, comparison: Comparison) -> Result<Decision, SaddleError> {
        self.prepare(&comparison)?;
        let decision = decide(&self.config, comparison)?;
        let stage = comparison.stage();
        if self.blocks(stage) {
            return Ok(decision);
        }
        match decision.climb {
            Climb::Grow { .. } => self.waiting = Some(stage),
            Climb::Open => {
                self.open = true;
                self.waiting = None;
            }
            Climb::Stop => {
                self.waiting = None;
                if stage == Stage::Plane {
                    self.open = false;
                }
            }
            Climb::Adopt => {
                self.adopt(comparison)?;
                self.waiting = None;
                if stage == Stage::Plane {
                    self.open = false;
                }
            }
        }
        Ok(decision)
    }

    /// The adopted rate and its factors, at `temperature_k`.
    pub fn record(&self, temperature_k: f64) -> Result<LadderRecord, SaddleError> {
        if !(temperature_k.is_finite() && temperature_k > 0.0) {
            return Err(SaddleError::Invalid(
                "the record temperature must be finite and positive".into(),
            ));
        }
        Ok(LadderRecord {
            rate: self.rate.value,
            rate_error: self.rate.error,
            temperature_k,
            kappa: self.kappa,
            kappa_error: self.kappa_error,
            anharmonic: self.phi,
            rung: self.rung,
            open: self.open,
        })
    }

    fn prepare(&self, comparison: &Comparison) -> Result<(), SaddleError> {
        match *comparison {
            Comparison::Independent { current, .. } => {
                if !self.holds(current) {
                    return Err(SaddleError::Invalid(
                        "the sampled comparison is not against the rate in hand".into(),
                    ));
                }
                if self.rung >= 2 {
                    return Err(SaddleError::Invalid(
                        "a sampled rate does not replace a rate that already includes transmission"
                            .into(),
                    ));
                }
            }
            Comparison::Analytic { .. } => {
                if self.rung >= 1 {
                    return Err(SaddleError::Invalid(
                        "the analytic factor does not multiply a rate that already has a plane factor"
                            .into(),
                    ));
                }
            }
            Comparison::Transmission { .. } => {
                if self.rung >= 2 {
                    return Err(SaddleError::Invalid(
                        "transmission does not multiply a rate that already includes it".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    fn blocks(&self, stage: Stage) -> bool {
        let waiting_blocks = match self.waiting {
            Some(Stage::Plane) => stage == Stage::Transmission,
            Some(Stage::Transmission) => stage == Stage::Plane,
            None => false,
        };
        waiting_blocks || (self.open && stage == Stage::Transmission)
    }

    fn holds(&self, estimate: Estimate) -> bool {
        estimate.value.to_bits() == self.rate.value.to_bits()
            && estimate.error.to_bits() == self.rate.error.to_bits()
            && estimate.n == self.rate.n
    }

    fn adopt(&mut self, comparison: Comparison) -> Result<(), SaddleError> {
        match comparison {
            Comparison::Analytic { log_phi, error, n } => {
                let factor = log_phi.exp();
                if !factor.is_finite() {
                    return Err(SaddleError::NonFinite("free-energy factor"));
                }
                if factor <= 0.0 {
                    return Err(SaddleError::Invalid(
                        "the free-energy factor is not a positive rate".into(),
                    ));
                }
                let value = self.rate.value * factor;
                let rate_error = finite_product(factor * self.rate.error, value * error)?;
                self.store_rate(value, rate_error, n, factor)?;
                self.rung = 1;
            }
            Comparison::Transmission { kappa, error, n } => {
                let value = self.rate.value * kappa;
                let rate_error = finite_product(kappa * self.rate.error, self.rate.value * error)?;
                self.store_rate(value, rate_error, n, self.phi)?;
                self.kappa = kappa;
                self.kappa_error = error;
                self.rung = 2;
            }
            Comparison::Independent { next, .. } => {
                let phi = next.value / self.harmonic.value;
                self.store_rate(next.value, next.error, next.n, phi)?;
                self.rung = 1;
            }
        }
        Ok(())
    }

    fn store_rate(
        &mut self,
        value: f64,
        rate_error: f64,
        n: usize,
        phi: f64,
    ) -> Result<(), SaddleError> {
        if !value.is_finite() {
            return Err(SaddleError::NonFinite("adopted rate"));
        }
        if value == 0.0 {
            return Err(SaddleError::Invalid("the adopted rate underflowed".into()));
        }
        if value < 0.0 {
            return Err(SaddleError::Invalid(
                "the adopted rate must be positive".into(),
            ));
        }
        if !rate_error.is_finite() {
            return Err(SaddleError::NonFinite("adopted rate error"));
        }
        if rate_error < 0.0 {
            return Err(SaddleError::Invalid(
                "the adopted standard error must be non-negative".into(),
            ));
        }
        if !phi.is_finite() {
            return Err(SaddleError::NonFinite("free-energy factor"));
        }
        if phi == 0.0 {
            return Err(SaddleError::Invalid(
                "the free-energy factor underflowed".into(),
            ));
        }
        if phi < 0.0 {
            return Err(SaddleError::Invalid(
                "the free-energy factor must be positive".into(),
            ));
        }
        self.rate = Estimate {
            value,
            error: rate_error,
            n,
        };
        self.phi = phi;
        Ok(())
    }
}

fn finite_product(left: f64, right: f64) -> Result<f64, SaddleError> {
    if !left.is_finite() || !right.is_finite() {
        return Err(SaddleError::NonFinite("adopted rate error"));
    }
    let error = left.hypot(right);
    if error.is_finite() {
        Ok(error)
    } else {
        Err(SaddleError::NonFinite("adopted rate error"))
    }
}

/// Stop, grow, adopt, or leave the rung open.
pub fn decide(config: &LadderConfig, comparison: Comparison) -> Result<Decision, SaddleError> {
    config.check()?;
    let relative = relative_error(&comparison)?;
    let n = comparison.n();
    if n == 0 {
        return Err(SaddleError::Invalid(
            "a rung needs at least one sample".into(),
        ));
    }
    if relative > config.relative_error && n < config.cap {
        return Ok(Decision {
            climb: Climb::Grow {
                n: grown(config, n, relative),
            },
            z: None,
        });
    }
    let z = form_z(&comparison)?;
    let resolved = match z {
        Some(value) if value.abs() > config.z_resolved => Climb::Adopt,
        Some(_) => Climb::Stop,
        None => zero_climb(config, &comparison)?,
    };
    let climb = if relative > config.relative_error && resolved == Climb::Adopt {
        Climb::Open
    } else if relative > config.relative_error {
        Climb::Stop
    } else {
        resolved
    };
    Ok(Decision { climb, z })
}

/// True when `rate` lies on the inclusive published ends `4.3e-7` and `4.7e-7`.
///
/// A copper catalog rate or a diffusivity comparison calls
/// [`sampled_rungs_allowed`] before rung 1 or rung 2. The LEPS reproduction
/// is in the surface's own time unit. [`physical_rate_per_s`] is not applied
/// to it.
pub fn paper_repro_passes(rate: f64) -> bool {
    (OHTST_PAPER_LOW..=OHTST_PAPER_HIGH).contains(&rate)
}

/// Error when the OH-TST reproduction is outside the published band.
pub fn sampled_rungs_allowed(repro_rate: f64) -> Result<(), SaddleError> {
    if paper_repro_passes(repro_rate) {
        Ok(())
    } else {
        Err(SaddleError::Invalid(
            "sampled rungs stay closed until the OH-TST reproduction is inside the published band"
                .into(),
        ))
    }
}

/// Convert one coordinate-time rate to 1/s.
///
/// The surface energy is eV, the length is Å, and the mass in
/// `q = M^{1/2} x` is the atomic mass unit. The mass-weighted angular
/// frequency is `sqrt(λ / m)` in coordinate time, and
/// [`angular_scale_rad_s`](crate::angular_scale_rad_s) is the one factor
/// that turns it into rad/s. [`crate::vineyard_prefactor`] already includes
/// that factor once for a first-order saddle. Do not call this on a rate
/// that is already in Hz, and do not call it on the LEPS reproduction.
pub fn physical_rate_per_s(coordinate_rate: f64) -> Result<f64, SaddleError> {
    if !(coordinate_rate.is_finite() && coordinate_rate > 0.0) {
        return Err(SaddleError::Invalid(
            "the coordinate-time rate must be finite and positive".into(),
        ));
    }
    let rate = coordinate_rate * crate::angular_scale_rad_s();
    if !rate.is_finite() {
        return Err(SaddleError::NonFinite("physical rate"));
    }
    Ok(rate)
}

/// `None` compares the bits. `Some` matches an absolute gap of at most the tolerance.
pub fn temperatures_match(config: &LadderConfig, stored: f64, asked: f64) -> bool {
    if !stored.is_finite() || !asked.is_finite() {
        return false;
    }
    if stored.to_bits() == asked.to_bits() {
        return true;
    }
    match config.temperature_tolerance {
        Some(tol) if tol.is_finite() && tol >= 0.0 => (stored - asked).abs() <= tol,
        _ => false,
    }
}

/// `n` draws of the uniform stream. A longer `n` with the same seed repeats the prefix.
pub fn prefix_values(seed: u64, n: usize) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| rng.uniform()).collect()
}

fn relative_error(comparison: &Comparison) -> Result<f64, SaddleError> {
    match *comparison {
        Comparison::Analytic { log_phi, error, .. } => {
            check_sigma(error)?;
            if !log_phi.is_finite() {
                return Err(SaddleError::Invalid("log phi must be finite".into()));
            }
            Ok(error)
        }
        Comparison::Transmission { kappa, error, .. } => {
            check_sigma(error)?;
            if !(kappa.is_finite() && kappa > 0.0 && kappa <= 1.0) {
                return Err(SaddleError::Invalid("kappa must lie in (0, 1]".into()));
            }
            Ok(error / kappa)
        }
        Comparison::Independent { current, next } => {
            check_sigma(current.error)?;
            if !(current.value.is_finite() && current.value > 0.0) {
                return Err(SaddleError::Invalid(
                    "the rate in hand must be finite and positive".into(),
                ));
            }
            check_rate(next)?;
            Ok(next.error / next.value)
        }
    }
}

fn form_z(comparison: &Comparison) -> Result<Option<f64>, SaddleError> {
    let z = match *comparison {
        Comparison::Analytic { log_phi, error, .. } => {
            if error == 0.0 {
                None
            } else {
                Some(log_phi / error)
            }
        }
        Comparison::Transmission { kappa, error, .. } => {
            if error == 0.0 {
                None
            } else {
                Some((kappa - 1.0) / error)
            }
        }
        Comparison::Independent { current, next } => {
            if current.error == 0.0 || next.error == 0.0 {
                None
            } else {
                Some((next.value - current.value) / current.error.hypot(next.error))
            }
        }
    };
    if z.is_some_and(|value| !value.is_finite()) {
        return Err(SaddleError::NonFinite("z"));
    }
    Ok(z)
}

fn zero_climb(config: &LadderConfig, comparison: &Comparison) -> Result<Climb, SaddleError> {
    let adopt = match *comparison {
        Comparison::Analytic { log_phi, error, .. } => {
            if error != 0.0 {
                return Err(SaddleError::Invalid(
                    "a positive analytic error has a z".into(),
                ));
            }
            log_phi != 0.0
        }
        Comparison::Transmission { kappa, error, .. } => {
            if error != 0.0 {
                return Err(SaddleError::Invalid(
                    "a positive transmission error has a z".into(),
                ));
            }
            kappa != 1.0
        }
        Comparison::Independent { current, next } => {
            if current.error == 0.0 && next.error == 0.0 {
                current.value != next.value
            } else if current.error == 0.0 {
                (next.value - current.value).abs() > config.z_resolved * next.error
            } else if next.error == 0.0 {
                (next.value - current.value).abs() > config.z_resolved * current.error
            } else {
                return Err(SaddleError::Invalid(
                    "two positive rate errors have a z".into(),
                ));
            }
        }
    };
    Ok(if adopt { Climb::Adopt } else { Climb::Stop })
}

fn grown(config: &LadderConfig, n: usize, relative: f64) -> usize {
    let ratio = relative / config.relative_error;
    let raw = config.growth * n as f64 * ratio * ratio;
    let ceiled = if raw.is_finite() {
        let rounded = raw.ceil();
        if rounded >= usize::MAX as f64 {
            config.cap
        } else {
            rounded as usize
        }
    } else {
        config.cap
    };
    n.saturating_add(1).max(ceiled).min(config.cap)
}

fn check_sigma(error: f64) -> Result<(), SaddleError> {
    if error.is_finite() && error >= 0.0 {
        Ok(())
    } else {
        Err(SaddleError::Invalid(
            "a standard error must be finite and non-negative".into(),
        ))
    }
}

fn check_rate(estimate: Estimate) -> Result<(), SaddleError> {
    check_sigma(estimate.error)?;
    if estimate.n == 0 {
        return Err(SaddleError::Invalid(
            "a sampled rate needs at least one sample".into(),
        ));
    }
    if estimate.value.is_finite() && estimate.value > 0.0 {
        Ok(())
    } else {
        Err(SaddleError::Invalid(
            "a sampled rate must be finite and positive".into(),
        ))
    }
}
