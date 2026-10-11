//! One saddle-search configuration.
//!
//! Dimer, Lanczos, and ART nouveau are options on this handle. The
//! estimate and the reflected climb are a [`MinModeConfig`]. ART fields
//! are set with the same typed setters; setting one of them selects the
//! ART opening on that configuration. There is no string key.

use ndarray::Array1;

use crate::art::{ArtConfig, ArtDisplacement, ArtSupport};
use crate::error::SaddleError;
use crate::minmode::{FiniteDifference, MinModeConfig, MinModeKind};
use rgmin::Method;

/// ART fields that are not already on [`MinModeConfig`].
#[derive(Clone, Debug)]
struct ArtTune {
    eigen_threshold: f64,
    push_step: f64,
    max_pushes: usize,
    max_climbs: usize,
    perp_steps_push: usize,
    perp_steps_climb: usize,
    krylov_min: usize,
    activation_distance: f64,
    nsmooth: usize,
    push_over: f64,
}

impl From<&ArtConfig> for ArtTune {
    fn from(config: &ArtConfig) -> Self {
        Self {
            eigen_threshold: config.eigen_threshold,
            push_step: config.push_step,
            max_pushes: config.max_pushes,
            max_climbs: config.max_climbs,
            perp_steps_push: config.perp_steps_push,
            perp_steps_climb: config.perp_steps_climb,
            krylov_min: config.krylov_min,
            activation_distance: config.activation_distance,
            nsmooth: config.nsmooth,
            push_over: config.push_over,
        }
    }
}

/// Opaque saddle-search configuration: dimer, Lanczos, or ART nouveau.
#[derive(Clone, Debug)]
pub struct SearchConfig {
    mode: MinModeConfig,
    mode_touched: bool,
    art: Option<ArtTune>,
    displacement_seed: Option<u64>,
    displacement_amplitude: f64,
    support: ArtSupport,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl SearchConfig {
    /// Dimer defaults. ART setters switch on the opening and, until a
    /// mode field is set, adopt the ART nouveau Lanczos defaults.
    pub fn new() -> Self {
        Self {
            mode: MinModeConfig::default(),
            mode_touched: false,
            art: None,
            displacement_seed: None,
            displacement_amplitude: 0.15,
            support: ArtSupport::Global,
        }
    }

    pub fn art_enabled(&self) -> bool {
        self.art.is_some()
    }

    pub fn mode(&self) -> &MinModeConfig {
        &self.mode
    }

    pub fn set_kind(&mut self, kind: i32) -> Result<(), SaddleError> {
        self.mode_touched = true;
        self.mode.kind = match kind {
            0 => MinModeKind::Dimer,
            1 => MinModeKind::Lanczos,
            _ => {
                return Err(SaddleError::Invalid(
                    "saddle-search kind is dimer (0) or Lanczos (1)".into(),
                ));
            }
        };
        Ok(())
    }

    pub fn set_method(&mut self, method: i32) -> Result<(), SaddleError> {
        self.mode_touched = true;
        self.mode.method = match method {
            0 => Method::Fire {
                kind: rgmin::FireKind::V2,
            },
            1 => Method::Lbfgs { memory: 20 },
            _ => {
                return Err(SaddleError::Invalid(
                    "saddle-search method is FIRE (0) or L-BFGS (1)".into(),
                ));
            }
        };
        Ok(())
    }

    pub fn set_dr(&mut self, value: f64) -> Result<(), SaddleError> {
        self.mode_touched = true;
        positive("difference step", value)?;
        self.mode.dr = value;
        Ok(())
    }

    pub fn set_difference(&mut self, value: i32) -> Result<(), SaddleError> {
        self.mode_touched = true;
        self.mode.difference = match value {
            0 => FiniteDifference::Forward,
            1 => FiniteDifference::Central,
            _ => {
                return Err(SaddleError::Invalid(
                    "difference is forward (0) or central (1)".into(),
                ));
            }
        };
        Ok(())
    }

    pub fn set_rotation_tol(&mut self, value: f64) -> Result<(), SaddleError> {
        self.mode_touched = true;
        positive("rotation tolerance", value)?;
        self.mode.rotation_tol = value;
        Ok(())
    }

    pub fn set_rotation_angle_tol(&mut self, value: f64) -> Result<(), SaddleError> {
        self.mode_touched = true;
        if !(value.is_finite() && value >= 0.0) {
            return Err(SaddleError::Invalid(
                "rotation angle tolerance must be finite and non-negative".into(),
            ));
        }
        self.mode.rotation_angle_tol = value;
        Ok(())
    }

    pub fn set_max_rotations(&mut self, value: i64) -> Result<(), SaddleError> {
        self.mode_touched = true;
        self.mode.max_rotations = non_negative_count("max rotations", value)?;
        Ok(())
    }

    pub fn set_krylov_dim(&mut self, value: i64) -> Result<(), SaddleError> {
        self.mode_touched = true;
        let dim = non_negative_count("Krylov dimension", value)?;
        if dim < 1 {
            return Err(SaddleError::Invalid(
                "Krylov dimension must be at least 1".into(),
            ));
        }
        self.mode.krylov_dim = dim;
        Ok(())
    }

    pub fn set_force_tol(&mut self, value: f64) -> Result<(), SaddleError> {
        self.mode_touched = true;
        positive("force tolerance", value)?;
        self.mode.force_tol = value;
        Ok(())
    }

    pub fn set_max_move(&mut self, value: f64) -> Result<(), SaddleError> {
        self.mode_touched = true;
        positive("max move", value)?;
        self.mode.max_move = value;
        Ok(())
    }

    pub fn set_eigen_threshold(&mut self, value: f64) -> Result<(), SaddleError> {
        if !value.is_finite() {
            return Err(SaddleError::Invalid(
                "eigenvalue threshold must be finite".into(),
            ));
        }
        self.tune().eigen_threshold = value;
        Ok(())
    }

    pub fn set_push_step(&mut self, value: f64) -> Result<(), SaddleError> {
        positive("push step", value)?;
        self.tune().push_step = value;
        Ok(())
    }

    pub fn set_max_pushes(&mut self, value: i64) -> Result<(), SaddleError> {
        let count = non_negative_count("max pushes", value)?;
        self.tune().max_pushes = count;
        Ok(())
    }

    pub fn set_max_climbs(&mut self, value: i64) -> Result<(), SaddleError> {
        let count = non_negative_count("max climbs", value)?;
        self.tune().max_climbs = count;
        Ok(())
    }

    pub fn set_perp_steps_push(&mut self, value: i64) -> Result<(), SaddleError> {
        let count = non_negative_count("perpendicular push steps", value)?;
        self.tune().perp_steps_push = count;
        Ok(())
    }

    pub fn set_perp_steps_climb(&mut self, value: i64) -> Result<(), SaddleError> {
        let count = non_negative_count("perpendicular climb steps", value)?;
        self.tune().perp_steps_climb = count;
        Ok(())
    }

    pub fn set_krylov_min(&mut self, value: i64) -> Result<(), SaddleError> {
        let count = non_negative_count("minimum Krylov size", value)?;
        if count < 1 {
            return Err(SaddleError::Invalid(
                "minimum Krylov size must be at least 1".into(),
            ));
        }
        self.tune().krylov_min = count;
        Ok(())
    }

    pub fn set_activation_distance(&mut self, value: f64) -> Result<(), SaddleError> {
        if !(value.is_finite() && value >= 0.0) {
            return Err(SaddleError::Invalid(
                "activation distance must be finite and non-negative".into(),
            ));
        }
        self.tune().activation_distance = value;
        Ok(())
    }

    pub fn set_nsmooth(&mut self, value: i64) -> Result<(), SaddleError> {
        let count = non_negative_count("smooth steps", value)?;
        self.tune().nsmooth = count;
        Ok(())
    }

    pub fn set_push_over(&mut self, value: f64) -> Result<(), SaddleError> {
        if !(value.is_finite() && value >= 0.0) {
            return Err(SaddleError::Invalid(
                "push over the saddle must be finite and non-negative".into(),
            ));
        }
        self.tune().push_over = value;
        Ok(())
    }

    pub fn set_displacement_seed(&mut self, seed: u64) -> Result<(), SaddleError> {
        self.tune();
        self.displacement_seed = Some(seed);
        Ok(())
    }

    pub fn set_displacement_amplitude(&mut self, value: f64) -> Result<(), SaddleError> {
        positive("displacement amplitude", value)?;
        self.tune();
        self.displacement_amplitude = value;
        Ok(())
    }

    /// `None` is every coordinate. An empty slice is rejected.
    pub fn set_support(&mut self, indices: Option<&[usize]>) -> Result<(), SaddleError> {
        self.tune();
        self.support = match indices {
            None => ArtSupport::Global,
            Some([]) => {
                return Err(SaddleError::Invalid(
                    "a local support lists at least one coordinate".into(),
                ));
            }
            Some(indices) => ArtSupport::Local(indices.to_vec()),
        };
        Ok(())
    }

    /// Build the ART configuration. The mode block is the one set here.
    pub fn art_config(&self) -> Result<ArtConfig, SaddleError> {
        let Some(tune) = &self.art else {
            return Err(SaddleError::Invalid(
                "ART options were not set on the saddle-search configuration".into(),
            ));
        };
        ArtConfig::builder()
            .mode(self.mode.clone())
            .eigen_threshold(tune.eigen_threshold)
            .push_step(tune.push_step)
            .max_pushes(tune.max_pushes)
            .max_climbs(tune.max_climbs)
            .perp_steps_push(tune.perp_steps_push)
            .perp_steps_climb(tune.perp_steps_climb)
            .krylov_min(tune.krylov_min)
            .activation_distance(tune.activation_distance)
            .nsmooth(tune.nsmooth)
            .push_over(tune.push_over)
            .build()
    }

    /// Random when a seed was set, otherwise the host direction.
    pub fn displacement(
        &self,
        direction: Option<Array1<f64>>,
    ) -> Result<ArtDisplacement, SaddleError> {
        if self.art.is_none() {
            return Err(SaddleError::Invalid(
                "a displacement belongs to an ART search".into(),
            ));
        }
        let support = self.support.clone();
        let amplitude = self.displacement_amplitude;
        if let Some(seed) = self.displacement_seed {
            return Ok(ArtDisplacement::Random {
                seed,
                support,
                amplitude,
            });
        }
        let Some(direction) = direction else {
            return Err(SaddleError::Invalid(
                "a targeted ART search needs a direction".into(),
            ));
        };
        Ok(ArtDisplacement::Targeted {
            direction,
            support,
            amplitude,
        })
    }

    fn tune(&mut self) -> &mut ArtTune {
        if self.art.is_none() {
            if !self.mode_touched {
                self.mode = ArtConfig::default().mode;
            }
            self.art = Some(ArtTune::from(&ArtConfig::default()));
        }
        self.art.as_mut().unwrap()
    }
}

fn positive(name: &str, value: f64) -> Result<(), SaddleError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(SaddleError::Invalid(format!(
            "{name} must be finite and positive"
        )))
    }
}

fn non_negative_count(name: &str, value: i64) -> Result<usize, SaddleError> {
    if value < 0 {
        return Err(SaddleError::Invalid(format!("{name} must be non-negative")));
    }
    Ok(value as usize)
}

#[cfg(test)]
mod tests {
    #[test]
    fn art_options_share_the_minmode_block() {
        let mut config = super::SearchConfig::new();
        config.set_dr(1e-4).unwrap();
        config.set_eigen_threshold(-0.01).unwrap();
        config.set_push_step(0.3).unwrap();
        let art = config.art_config().unwrap();
        assert!((art.mode.dr - 1e-4).abs() < 1e-15);
        assert!((art.eigen_threshold + 0.01).abs() < 1e-15);
        assert!(config.set_push_step(0.0).is_err());
        assert!(config.set_kind(9).is_err());
    }
}
