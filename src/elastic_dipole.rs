//! Anisotropic point-defect interactions and periodic-image corrections.
//!
//! A point defect with dipole `P` has interaction energy `-P:ε`.  The
//! dipole follows Varvenne and Clouet, Phys. Rev. B 96, 224103 (2017),
//! doi:10.1103/PhysRevB.96.224103, and Clouet, Varvenne, and Jourdan,
//! Comput. Mater. Sci. 147, 49 (2018), doi:10.1016/j.commatsci.2018.01.053.
//! [`crate::ElasticDipole::from_stress`] is the residual-stress route,
//! [`crate::ElasticDipole::from_energy_strain_gradient`] is the homogeneous
//! strain route, and [`crate::ElasticDipole::from_forces`] is the Kanzaki
//! force-moment route.
//!
//! [`AnisotropicGreen`] evaluates Barnett's azimuthal integral and its
//! analytic second derivative.  It does not finite-difference the Green
//! function.  [`PeriodicElastic`] sums lattice images in growing spheres,
//! which fixes the otherwise conditional order of the `1/r³` sum.  The
//! periodic correction is the image term of Varvenne, Bruneval, Marinica,
//! and Clouet, Phys. Rev. B 88, 134102 (2013),
//! doi:10.1103/PhysRevB.88.134102.  Repeating the angular quadrature and the
//! outer image sphere supplies the reported numerical error estimate.
//! An acoustic inverse is performed only for a positive definite tensor.
//! Sylvester's criterion requires every leading principal minor of the
//! symmetric part to be positive, because a positive determinant still
//! allows two negative eigenvalues.  A corrected harmonic rate that is not
//! finite is rejected.

use crate::error::SaddleError;
use crate::strain_rate::{BOLTZMANN_EV_PER_K, ElasticDipole};

/// A fourth-rank Cartesian tensor, indexed `C[i][j][k][l]`.
pub type Tensor4 = [[[[f64; 3]; 3]; 3]; 3];

/// `∂²G_ik / ∂x_j∂x_l`, indexed `[i][j][k][l]`.
pub type GreenSecondDerivative = Tensor4;

/// Elastic stiffness tensor in energy per volume.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ElasticConstants {
    tensor: Tensor4,
}

impl ElasticConstants {
    /// Validate a tensor with the minor and major elastic symmetries.
    ///
    /// Each sampled acoustic tensor must be positive definite. A positive
    /// determinant does not prove that: two negative eigenvalues leave the
    /// determinant positive.
    pub fn from_tensor(tensor: Tensor4) -> Result<Self, SaddleError> {
        if !tensor
            .iter()
            .flatten()
            .flatten()
            .flatten()
            .all(|v| v.is_finite())
        {
            return Err(SaddleError::NonFinite("elastic constants"));
        }
        let scale = tensor
            .iter()
            .flatten()
            .flatten()
            .flatten()
            .fold(0.0_f64, |largest, value| largest.max(value.abs()))
            .max(1.0);
        let tolerance = 64.0 * f64::EPSILON * scale;
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    for l in 0..3 {
                        let value = tensor[i][j][k][l];
                        if (value - tensor[j][i][k][l]).abs() > tolerance
                            || (value - tensor[i][j][l][k]).abs() > tolerance
                            || (value - tensor[k][l][i][j]).abs() > tolerance
                        {
                            return Err(SaddleError::Invalid(
                                "elastic constants need minor and major symmetry".into(),
                            ));
                        }
                    }
                }
            }
        }
        let constants = Self { tensor };
        // Strong ellipticity requires a positive definite acoustic tensor in
        // every direction.  These crystal directions catch invalid
        // cubic/orthotropic inputs early.  Barnett quadrature checks every
        // direction it actually uses, and `invert3` applies Sylvester's
        // criterion rather than accepting a positive determinant.
        for direction in [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ] {
            let n = normalized(direction)?;
            invert3(acoustic(tensor, n))?;
        }
        Ok(constants)
    }

    /// Build the tensor from conventional Voigt order `xx, yy, zz, yz, xz, xy`.
    ///
    /// The shear entries are physical stiffnesses: for example
    /// `C[3][3] = C_yzyz`, so `σ_yz = 2 C[3][3] ε_yz`.
    pub fn from_voigt(voigt: [[f64; 6]; 6]) -> Result<Self, SaddleError> {
        if !voigt.iter().flatten().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("Voigt elastic constants"));
        }
        let pairs = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)];
        let mut tensor = [[[[0.0; 3]; 3]; 3]; 3];
        for (a, (i, j)) in pairs.iter().copied().enumerate() {
            for (b, (k, l)) in pairs.iter().copied().enumerate() {
                for (p, q) in [(i, j), (j, i)] {
                    for (r, s) in [(k, l), (l, k)] {
                        tensor[p][q][r][s] = voigt[a][b];
                    }
                }
            }
        }
        Self::from_tensor(tensor)
    }

    /// Cubic crystal with `C11`, `C12`, and `C44`.
    pub fn cubic(c11: f64, c12: f64, c44: f64) -> Result<Self, SaddleError> {
        let mut voigt = [[0.0; 6]; 6];
        for (i, row) in voigt.iter_mut().enumerate().take(3) {
            row[i] = c11;
            for (j, value) in row.iter_mut().enumerate().take(3) {
                if i != j {
                    *value = c12;
                }
            }
        }
        for (i, row) in voigt.iter_mut().enumerate().skip(3) {
            row[i] = c44;
        }
        Self::from_voigt(voigt)
    }

    /// Isotropic medium with Lamé constants `λ` and `μ`.
    pub fn isotropic(lambda: f64, shear_modulus: f64) -> Result<Self, SaddleError> {
        Self::cubic(lambda + 2.0 * shear_modulus, lambda, shear_modulus)
    }

    /// The full Cartesian tensor.
    pub fn tensor(self) -> Tensor4 {
        self.tensor
    }
}

/// Controls Barnett's azimuthal quadrature.
///
/// Non-exhaustive: start from [`Self::default`] and use the setters.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct AnisotropicGreenConfig {
    /// Points on the half-circle (at least 32 and a multiple of 8).  The
    /// periodic trapezoid rule is used together with a nested half mesh.
    pub angular_points: usize,
}

impl Default for AnisotropicGreenConfig {
    fn default() -> Self {
        Self { angular_points: 96 }
    }
}

with_setters!(AnisotropicGreenConfig {
    with_angular_points => angular_points: val usize,
});

/// One value and an absolute numerical error estimate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InteractionEstimate {
    /// Interaction energy.
    pub value: f64,
    /// Difference on doubling the angular mesh, plus image-shell error when applicable.
    pub error: f64,
}

/// Barnett anisotropic Green function for a homogeneous medium.
#[derive(Clone, Copy, Debug)]
pub struct AnisotropicGreen {
    constants: ElasticConstants,
    config: AnisotropicGreenConfig,
}

impl AnisotropicGreen {
    /// Use the default angular quadrature.
    pub fn new(constants: ElasticConstants) -> Self {
        Self {
            constants,
            config: AnisotropicGreenConfig::default(),
        }
    }

    /// Replace the quadrature controls.
    pub fn with_config(mut self, config: AnisotropicGreenConfig) -> Result<Self, SaddleError> {
        validate_angular_points(config.angular_points)?;
        self.config = config;
        Ok(self)
    }

    /// Elastic constants used by the kernel.
    pub fn constants(self) -> ElasticConstants {
        self.constants
    }

    /// Analytic second derivative of the anisotropic Green function.
    ///
    /// The error tensor is the absolute difference between the configured
    /// half-circle mesh and the nested mesh with half as many points.
    pub fn second_derivative(
        &self,
        separation: [f64; 3],
    ) -> Result<(GreenSecondDerivative, GreenSecondDerivative), SaddleError> {
        validate_angular_points(self.config.angular_points)?;
        let fine = barnett_second(
            self.constants.tensor,
            separation,
            self.config.angular_points,
        )?;
        let coarse = barnett_second(
            self.constants.tensor,
            separation,
            self.config.angular_points / 2,
        )?;
        let mut error = [[[[0.0; 3]; 3]; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    for l in 0..3 {
                        error[i][j][k][l] = (fine[i][j][k][l] - coarse[i][j][k][l]).abs();
                    }
                }
            }
        }
        Ok((fine, error))
    }

    /// `P_target,ij G_ik,jl P_source,kl`.
    pub fn dipole_interaction(
        &self,
        separation: [f64; 3],
        source: ElasticDipole,
        target: ElasticDipole,
    ) -> Result<InteractionEstimate, SaddleError> {
        let (kernel, error) = self.second_derivative(separation)?;
        Ok(InteractionEstimate {
            value: contract_kernel(target.tensor(), kernel, source.tensor()),
            error: contract_kernel_abs(target.tensor(), error, source.tensor()),
        })
    }
}

/// Controls the conditionally convergent periodic image sum.
///
/// Images are ordered by growing Cartesian spheres, not by a basis-dependent
/// lattice cube.  `image_shells` is the outer radius in units of the shortest
/// perpendicular cell width.  The error includes the change on dropping the
/// outermost shell.
///
/// Non-exhaustive: start from [`Self::default`] and use the setters.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct PeriodicElasticConfig {
    /// Barnett quadrature controls.
    pub green: AnisotropicGreenConfig,
    /// Radius of the spherical image sum in shortest-cell-width units.
    pub image_shells: usize,
}

impl Default for PeriodicElasticConfig {
    fn default() -> Self {
        Self {
            green: AnisotropicGreenConfig::default(),
            image_shells: 4,
        }
    }
}

with_setters!(PeriodicElasticConfig {
    with_green => green: val AnisotropicGreenConfig,
    with_image_shells => image_shells: val usize,
});

/// Infinite-cell correction for one periodic pair energy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ElasticCorrection {
    /// Pair interaction plus both defects' selected periodic self-images.
    pub periodic: f64,
    /// Pair interaction with no images.
    pub infinite: f64,
    /// `infinite - periodic`, to add to a periodic atomistic energy.
    pub correction: f64,
    /// Absolute numerical error estimate for `correction`.
    pub error: f64,
}

/// One two-defect point on an atomistic energy landscape.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ElasticPairState {
    /// Raw periodic atomistic energy.
    pub energy: f64,
    /// Centre and dipole of the first defect.
    pub first: DefectDipole,
    /// Centre and dipole of the second defect.
    pub second: DefectDipole,
}

/// Position and dipole of one point defect.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DefectDipole {
    /// Cartesian position.
    pub position: [f64; 3],
    /// Elastic dipole.
    pub dipole: ElasticDipole,
}

/// Infinite-cell energy for one landscape point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CorrectedEnergy {
    /// Input periodic energy.
    pub raw: f64,
    /// `raw + correction`.
    pub infinite: f64,
    /// Elastic correction added to `raw`.
    pub correction: f64,
    /// Absolute numerical error estimate.
    pub error: f64,
}

/// Reactant and saddle data for one rate-catalogue row.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ElasticRate {
    /// Reactant pair.
    pub reactant: ElasticPairState,
    /// Saddle pair.
    pub saddle: ElasticPairState,
    /// Attempt frequency.
    pub prefactor_hz: f64,
    /// Temperature.
    pub temperature_k: f64,
}

/// Infinite-cell barrier and harmonic rate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CorrectedRate {
    /// Raw periodic barrier.
    pub raw_barrier: f64,
    /// Infinite-cell barrier.
    pub infinite_barrier: f64,
    /// Absolute barrier error estimate.
    pub barrier_error: f64,
    /// Harmonic rate from the corrected barrier.
    pub rate_hz: f64,
}

/// Anisotropic interaction plus the spherical periodic-image sum.
#[derive(Clone, Copy, Debug)]
pub struct PeriodicElastic {
    green: AnisotropicGreen,
    config: PeriodicElasticConfig,
}

impl PeriodicElastic {
    /// Build a periodic corrector.
    pub fn new(
        constants: ElasticConstants,
        config: PeriodicElasticConfig,
    ) -> Result<Self, SaddleError> {
        if config.image_shells < 2 {
            return Err(SaddleError::Invalid(
                "the image error estimate needs at least two shells".into(),
            ));
        }
        let green = AnisotropicGreen::new(constants).with_config(config.green)?;
        Ok(Self { green, config })
    }

    /// Interaction in the central cell and all selected spherical images.
    pub fn periodic_interaction(
        &self,
        cell: &linkcell::Cell,
        first: DefectDipole,
        second: DefectDipole,
    ) -> Result<InteractionEstimate, SaddleError> {
        let separation = cell.displacement_euclidean(first.position, second.position);
        let (fine_pair, fine_pair_angular) = self.image_sum(
            cell,
            separation,
            first.dipole,
            second.dipole,
            self.config.image_shells,
            true,
        )?;
        let (coarse_pair, coarse_pair_angular) = self.image_sum(
            cell,
            separation,
            first.dipole,
            second.dipole,
            self.config.image_shells - 1,
            true,
        )?;
        let (fine_first, fine_first_angular) = self.image_sum(
            cell,
            [0.0; 3],
            first.dipole,
            first.dipole,
            self.config.image_shells,
            false,
        )?;
        let (coarse_first, coarse_first_angular) = self.image_sum(
            cell,
            [0.0; 3],
            first.dipole,
            first.dipole,
            self.config.image_shells - 1,
            false,
        )?;
        let (fine_second, fine_second_angular) = self.image_sum(
            cell,
            [0.0; 3],
            second.dipole,
            second.dipole,
            self.config.image_shells,
            false,
        )?;
        let (coarse_second, coarse_second_angular) = self.image_sum(
            cell,
            [0.0; 3],
            second.dipole,
            second.dipole,
            self.config.image_shells - 1,
            false,
        )?;
        // A periodic cell owns one half of every interaction with a copy of
        // the same defect.  Cross interactions need no half: the two ordered
        // defect pairs cancel it.
        let fine = fine_pair + 0.5 * (fine_first + fine_second);
        let coarse = coarse_pair + 0.5 * (coarse_first + coarse_second);
        let angular_error = fine_pair_angular
            + coarse_pair_angular
            + 0.5
                * (fine_first_angular
                    + coarse_first_angular
                    + fine_second_angular
                    + coarse_second_angular);
        Ok(InteractionEstimate {
            value: fine,
            error: (fine - coarse).abs() + angular_error,
        })
    }

    /// Correction from a periodic pair interaction to the infinite-cell pair.
    pub fn pair_correction(
        &self,
        cell: &linkcell::Cell,
        first: DefectDipole,
        second: DefectDipole,
    ) -> Result<ElasticCorrection, SaddleError> {
        let separation = cell.displacement_euclidean(first.position, second.position);
        let infinite = self
            .green
            .dipole_interaction(separation, first.dipole, second.dipole)?;
        let periodic = self.periodic_interaction(cell, first, second)?;
        Ok(ElasticCorrection {
            periodic: periodic.value,
            infinite: infinite.value,
            correction: infinite.value - periodic.value,
            error: infinite.error + periodic.error,
        })
    }

    /// Correct one point on an energy landscape.
    pub fn correct_energy(
        &self,
        cell: &linkcell::Cell,
        state: ElasticPairState,
    ) -> Result<CorrectedEnergy, SaddleError> {
        if !state.energy.is_finite() {
            return Err(SaddleError::NonFinite("periodic pair energy"));
        }
        let correction = self.pair_correction(cell, state.first, state.second)?;
        Ok(CorrectedEnergy {
            raw: state.energy,
            infinite: state.energy + correction.correction,
            correction: correction.correction,
            error: correction.error,
        })
    }

    /// Correct every point in a defect-pair energy landscape.
    pub fn correct_landscape(
        &self,
        cell: &linkcell::Cell,
        states: &[ElasticPairState],
    ) -> Result<Vec<CorrectedEnergy>, SaddleError> {
        states
            .iter()
            .copied()
            .map(|state| self.correct_energy(cell, state))
            .collect()
    }

    /// Correct barriers and harmonic rates for a rate catalogue.
    ///
    /// The corrected harmonic rate must be finite.
    pub fn correct_rate_catalogue(
        &self,
        cell: &linkcell::Cell,
        rates: &[ElasticRate],
    ) -> Result<Vec<CorrectedRate>, SaddleError> {
        rates
            .iter()
            .map(|rate| {
                if !(rate.prefactor_hz.is_finite() && rate.prefactor_hz > 0.0) {
                    return Err(SaddleError::Invalid(
                        "a rate prefactor must be finite and positive".into(),
                    ));
                }
                if !(rate.temperature_k.is_finite() && rate.temperature_k > 0.0) {
                    return Err(SaddleError::Invalid(
                        "a rate temperature must be finite and positive".into(),
                    ));
                }
                let reactant = self.correct_energy(cell, rate.reactant)?;
                let saddle = self.correct_energy(cell, rate.saddle)?;
                let raw_barrier = rate.saddle.energy - rate.reactant.energy;
                let infinite_barrier = saddle.infinite - reactant.infinite;
                let rate_hz = rate.prefactor_hz
                    * (-infinite_barrier / (BOLTZMANN_EV_PER_K * rate.temperature_k)).exp();
                if !rate_hz.is_finite() {
                    return Err(SaddleError::NonFinite("corrected harmonic rate"));
                }
                Ok(CorrectedRate {
                    raw_barrier,
                    infinite_barrier,
                    barrier_error: saddle.error + reactant.error,
                    rate_hz,
                })
            })
            .collect()
    }

    fn image_sum(
        &self,
        cell: &linkcell::Cell,
        separation: [f64; 3],
        source: ElasticDipole,
        target: ElasticDipole,
        shells: usize,
        include_central: bool,
    ) -> Result<(f64, f64), SaddleError> {
        let shortest = cell.widths().into_iter().fold(f64::INFINITY, f64::min);
        if !(shortest.is_finite() && shortest > 0.0) {
            return Err(SaddleError::Invalid(
                "periodic elastic correction needs a finite three-dimensional cell".into(),
            ));
        }
        let radius = shortest * shells as f64;
        let hinv = cell.hinv();
        let mut bounds = [0_i32; 3];
        for (i, slot) in bounds.iter_mut().enumerate() {
            let reciprocal_row =
                (hinv[0][i].powi(2) + hinv[1][i].powi(2) + hinv[2][i].powi(2)).sqrt();
            let bound = (radius * reciprocal_row).ceil() + 1.0;
            if bound > f64::from(i32::MAX) {
                return Err(SaddleError::Invalid(
                    "periodic image radius exceeds the lattice-index range".into(),
                ));
            }
            *slot = bound as i32;
        }
        let radius2 = radius * radius;
        let mut value = 0.0;
        let mut angular_error = 0.0;
        for na in -bounds[0]..=bounds[0] {
            for nb in -bounds[1]..=bounds[1] {
                for nc in -bounds[2]..=bounds[2] {
                    if !include_central && na == 0 && nb == 0 && nc == 0 {
                        continue;
                    }
                    let shift = cell.lattice_shift(na, nb, nc);
                    let shift2 = dot(shift, shift);
                    if shift2 > radius2 {
                        continue;
                    }
                    let r = [
                        separation[0] + shift[0],
                        separation[1] + shift[1],
                        separation[2] + shift[2],
                    ];
                    let estimate = self.green.dipole_interaction(r, source, target)?;
                    value += estimate.value;
                    angular_error += estimate.error;
                }
            }
        }
        Ok((value, angular_error))
    }
}

fn validate_angular_points(points: usize) -> Result<(), SaddleError> {
    if points < 32 || !points.is_multiple_of(8) {
        return Err(SaddleError::Invalid(
            "Barnett quadrature needs at least 32 points and a multiple of 8 so its nested half mesh has at least 16 points and a multiple of 4".into(),
        ));
    }
    Ok(())
}

fn validate_barnett_mesh(points: usize) -> Result<(), SaddleError> {
    if points < 16 || !points.is_multiple_of(4) {
        return Err(SaddleError::Invalid(
            "Barnett quadrature needs at least 16 points and a multiple of 4".into(),
        ));
    }
    Ok(())
}

#[allow(clippy::needless_range_loop)]
fn barnett_second(
    constants: Tensor4,
    separation: [f64; 3],
    points: usize,
) -> Result<GreenSecondDerivative, SaddleError> {
    validate_barnett_mesh(points)?;
    let radius = dot(separation, separation).sqrt();
    if !(radius.is_finite() && radius > f64::MIN_POSITIVE.sqrt()) {
        return Err(SaddleError::Invalid(
            "the anisotropic Green function is singular at the origin".into(),
        ));
    }
    let t = [
        separation[0] / radius,
        separation[1] / radius,
        separation[2] / radius,
    ];
    let (u, v) = plane(t)?;
    let mut sum = [[[[0.0; 3]; 3]; 3]; 3];
    for q in 0..points {
        let phi = std::f64::consts::PI * q as f64 / points as f64;
        let z = [
            phi.cos() * u[0] + phi.sin() * v[0],
            phi.cos() * u[1] + phi.sin() * v[1],
            phi.cos() * u[2] + phi.sin() * v[2],
        ];
        let ms = invert3(acoustic(constants, z))?;
        let mut zt = [[0.0; 3]; 3];
        for p in 0..3 {
            for w in 0..3 {
                zt[p][w] = z[p] * t[w] + t[p] * z[w];
            }
        }
        let mut f = [[0.0; 3]; 3];
        for i in 0..3 {
            for r in 0..3 {
                for j in 0..3 {
                    for p in 0..3 {
                        for n in 0..3 {
                            for w in 0..3 {
                                f[i][r] += constants[j][p][n][w] * ms[i][j] * ms[n][r] * zt[p][w];
                            }
                        }
                    }
                }
            }
        }
        let mut air = [[0.0; 3]; 3];
        for i in 0..3 {
            for r in 0..3 {
                for j in 0..3 {
                    for p in 0..3 {
                        for n in 0..3 {
                            for w in 0..3 {
                                let mf = f[i][j] * ms[n][r] + f[n][r] * ms[i][j];
                                let bracket =
                                    zt[p][w] * mf - 2.0 * ms[i][j] * ms[n][r] * t[p] * t[w];
                                air[i][r] += constants[j][p][n][w] * bracket;
                            }
                        }
                    }
                }
            }
        }
        for i in 0..3 {
            for s in 0..3 {
                for r in 0..3 {
                    for m in 0..3 {
                        sum[i][s][r][m] += -2.0 * zt[s][m] * f[i][r]
                            + 2.0 * t[s] * t[m] * ms[i][r]
                            + z[s] * z[m] * air[i][r];
                    }
                }
            }
        }
    }
    let scale = 1.0 / (4.0 * std::f64::consts::PI * points as f64 * radius * radius * radius);
    for value in sum.iter_mut().flatten().flatten().flatten() {
        *value *= scale;
    }
    Ok(sum)
}

fn acoustic(constants: Tensor4, direction: [f64; 3]) -> [[f64; 3]; 3] {
    let mut matrix = [[0.0; 3]; 3];
    for i in 0..3 {
        for k in 0..3 {
            for j in 0..3 {
                for l in 0..3 {
                    matrix[i][k] += constants[i][j][k][l] * direction[j] * direction[l];
                }
            }
        }
    }
    matrix
}

fn determinant3(matrix: [[f64; 3]; 3]) -> f64 {
    matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
}

fn symmetric_part(matrix: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let off_01 = f64::midpoint(matrix[0][1], matrix[1][0]);
    let off_02 = f64::midpoint(matrix[0][2], matrix[2][0]);
    let off_12 = f64::midpoint(matrix[1][2], matrix[2][1]);
    [
        [matrix[0][0], off_01, off_02],
        [off_01, matrix[1][1], off_12],
        [off_02, off_12, matrix[2][2]],
    ]
}

fn max_abs3(matrix: [[f64; 3]; 3]) -> f64 {
    matrix
        .iter()
        .flatten()
        .fold(0.0_f64, |largest, value| largest.max(value.abs()))
}

fn acoustic_positive_definite(matrix: [[f64; 3]; 3]) -> bool {
    // Positive definiteness is the quadratic form, so Sylvester's criterion
    // applies to the symmetric part.  Every leading principal minor must be
    // positive.  A positive determinant is not enough: two negative
    // eigenvalues leave it positive and the tensor indefinite.
    let symmetric = symmetric_part(matrix);
    let scale = max_abs3(symmetric);
    let relative = f64::EPSILON.sqrt();
    let minors = [
        symmetric[0][0],
        symmetric[0][0] * symmetric[1][1] - symmetric[0][1] * symmetric[1][0],
        determinant3(symmetric),
    ];
    let floors = [
        relative * scale,
        relative * scale * scale,
        relative * scale * scale * scale,
    ];
    minors
        .into_iter()
        .zip(floors)
        .all(|(minor, floor)| minor.is_finite() && minor > floor)
}

fn invert3(matrix: [[f64; 3]; 3]) -> Result<[[f64; 3]; 3], SaddleError> {
    let det = determinant3(matrix);
    let scale = max_abs3(matrix);
    let determinant_floor = f64::EPSILON.sqrt() * scale * scale * scale;
    if !acoustic_positive_definite(matrix) || !(det.is_finite() && det > determinant_floor) {
        return Err(SaddleError::Invalid(
            "elastic acoustic tensor is not positive definite".into(),
        ));
    }
    let inverse_det = 1.0 / det;
    Ok([
        [
            (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) * inverse_det,
            (matrix[0][2] * matrix[2][1] - matrix[0][1] * matrix[2][2]) * inverse_det,
            (matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1]) * inverse_det,
        ],
        [
            (matrix[1][2] * matrix[2][0] - matrix[1][0] * matrix[2][2]) * inverse_det,
            (matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0]) * inverse_det,
            (matrix[0][2] * matrix[1][0] - matrix[0][0] * matrix[1][2]) * inverse_det,
        ],
        [
            (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]) * inverse_det,
            (matrix[0][1] * matrix[2][0] - matrix[0][0] * matrix[2][1]) * inverse_det,
            (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]) * inverse_det,
        ],
    ])
}

fn plane(normal: [f64; 3]) -> Result<([f64; 3], [f64; 3]), SaddleError> {
    let axis = if normal[0].abs() <= normal[1].abs() && normal[0].abs() <= normal[2].abs() {
        [1.0, 0.0, 0.0]
    } else if normal[1].abs() <= normal[2].abs() {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    let u = normalized(cross(normal, axis))?;
    let v = cross(normal, u);
    Ok((u, v))
}

fn normalized(vector: [f64; 3]) -> Result<[f64; 3], SaddleError> {
    let length = dot(vector, vector).sqrt();
    if !(length.is_finite() && length > f64::MIN_POSITIVE.sqrt()) {
        return Err(SaddleError::Invalid(
            "a direction must be finite and nonzero".into(),
        ));
    }
    Ok([vector[0] / length, vector[1] / length, vector[2] / length])
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

#[allow(clippy::needless_range_loop)]
fn contract_kernel(left: [[f64; 3]; 3], kernel: Tensor4, right: [[f64; 3]; 3]) -> f64 {
    let mut value = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                for l in 0..3 {
                    value += left[i][j] * kernel[i][j][k][l] * right[k][l];
                }
            }
        }
    }
    value
}

#[allow(clippy::needless_range_loop)]
fn contract_kernel_abs(left: [[f64; 3]; 3], kernel: Tensor4, right: [[f64; 3]; 3]) -> f64 {
    let mut value = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                for l in 0..3 {
                    value += (left[i][j] * kernel[i][j][k][l] * right[k][l]).abs();
                }
            }
        }
    }
    value
}

#[cfg(test)]
mod tests {
    use super::invert3;
    use crate::error::SaddleError;

    fn multiply_residual(matrix: [[f64; 3]; 3], inverse: [[f64; 3]; 3]) -> f64 {
        let mut largest = 0.0_f64;
        for (i, row) in matrix.iter().enumerate() {
            let mut product = [0.0_f64; 3];
            for (entry, inverse_row) in row.iter().zip(inverse) {
                for (slot, coefficient) in product.iter_mut().zip(inverse_row) {
                    *slot += entry * coefficient;
                }
            }
            for (column, value) in product.iter().enumerate() {
                let diagonal = if i == column { 1.0 } else { 0.0 };
                largest = largest.max((value - diagonal).abs());
            }
        }
        largest
    }

    fn assert_rejected(matrix: [[f64; 3]; 3]) {
        let error = invert3(matrix).expect_err("acoustic tensor must be rejected");
        assert!(
            matches!(
                error,
                SaddleError::Invalid(ref message) if message.contains("not positive definite")
            ),
            "{error}"
        );
    }

    #[test]
    fn positive_definite_acoustic_tensor_inverts() {
        let matrix = [[4.0, 1.0, 0.2], [1.0, 3.0, 0.5], [0.2, 0.5, 2.0]];
        let inverse = invert3(matrix).expect("positive definite acoustic tensor");
        let residual = multiply_residual(matrix, inverse);
        assert!(
            residual <= 16.0 * f64::EPSILON,
            "inverse residual {residual}"
        );
    }

    #[test]
    fn negative_acoustic_eigenvalue_is_rejected() {
        // Eigenvalues 1, 1, -1. The determinant is negative.
        assert_rejected([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]);
    }

    #[test]
    fn indefinite_acoustic_tensor_with_positive_determinant_is_rejected() {
        // Eigenvalues 2, -1, -1. The determinant is 2.
        assert_rejected([[2.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]);
    }

    #[test]
    fn skewed_acoustic_tensor_with_indefinite_symmetric_part_is_rejected() {
        // Leading minors of this stored matrix are 1, 1, and 1. The
        // symmetric part is [[1, 2, 0], [2, 1, 0], [0, 0, 1]], whose
        // quadratic form takes both signs.
        assert_rejected([[1.0, 4.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    }
}
