//! Vineyard analysis on the atom-strain coordinate used by solid-state searches.
//!
//! [`crate::JointChart`] stores Cartesian reference positions followed by
//! `J epsilon`, where `J = V^(1/3) N^(1/6)`. Every coordinate therefore has
//! units of length. This module assigns the supplied atomic mass to every
//! component of that generalized coordinate. The resulting kinetic metric is
//! `mass * dq · dq`; it is the convention paired with the 2012 SS-NEB
//! Jacobian, not a fictitious barostat mass.
//!
//! The homogeneous strains are retained in both Hessians. Rigid translations
//! are identified as near-zero eigenvalues, and the saddle must have exactly
//! one unstable eigenvalue. The caller supplies every numerical scale:
//! finite-difference widths, stationarity tolerance, and the eigenvalue
//! tolerance justified from the Hessian error.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::cell_log::{m3_from_row9, m3_to_row9};
use crate::eigensolve::exact_eigh;
use crate::error::SaddleError;
use crate::minmode::PointSurface;
use crate::solid_dimer::{JointChart, finite_difference_cauchy_stress};
use crate::strain_rate::vineyard_prefactor;

/// Required numerical controls for a joint atom-strain Hessian.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct JointHessianConfig {
    /// Central-difference displacement in the generalized coordinate.
    pub coordinate_step: f64,
    /// Right-strain displacement used only when the surface has no stress.
    pub stress_step: f64,
    /// Largest accepted generalized-gradient component at the input point.
    pub stationary_tolerance: f64,
}

impl Default for JointHessianConfig {
    fn default() -> Self {
        Self {
            coordinate_step: 0.0,
            stress_step: 0.0,
            stationary_tolerance: 0.0,
        }
    }
}

with_setters!(JointHessianConfig {
    with_coordinate_step => coordinate_step: val f64,
    with_stress_step => stress_step: val f64,
    with_stationary_tolerance => stationary_tolerance: val f64,
});

impl JointHessianConfig {
    /// Construct controls without implicit numerical tolerances.
    pub const fn new(coordinate_step: f64, stress_step: f64, stationary_tolerance: f64) -> Self {
        Self {
            coordinate_step,
            stress_step,
            stationary_tolerance,
        }
    }

    fn validate(self) -> Result<Self, SaddleError> {
        if !(self.coordinate_step.is_finite() && self.coordinate_step > 0.0) {
            return Err(SaddleError::Invalid(
                "joint Hessian coordinate step must be positive and finite".into(),
            ));
        }
        if !(self.stress_step.is_finite() && self.stress_step > 0.0) {
            return Err(SaddleError::Invalid(
                "joint Hessian stress step must be positive and finite".into(),
            ));
        }
        if !(self.stationary_tolerance.is_finite() && self.stationary_tolerance >= 0.0) {
            return Err(SaddleError::Invalid(
                "joint Hessian stationarity tolerance must be finite and non-negative".into(),
            ));
        }
        Ok(self)
    }
}

/// Dense Hessian and the residual at the point where it was evaluated.
#[derive(Clone, Debug, PartialEq)]
pub struct JointHessian {
    /// Symmetric Hessian of enthalpy with respect to `(r0, J epsilon)`.
    pub matrix: Array2<f64>,
    /// Largest absolute component of the center generalized gradient.
    pub max_gradient: f64,
}

/// Build the dense atom-strain Hessian at a stationary point.
///
/// The Hessian is a central difference of [`JointChart::gradient`]. The
/// surface's analytic Cauchy stress is used when present; otherwise
/// [`finite_difference_cauchy_stress`] uses `config.stress_step`. The two
/// directed estimates of every off-diagonal entry are averaged.
pub fn joint_hessian<S: PointSurface>(
    surface: &S,
    chart: &JointChart,
    q: ArrayView1<f64>,
    config: JointHessianConfig,
) -> Result<JointHessian, SaddleError> {
    let config = config.validate()?;
    if q.len() != chart.dimension() || !q.iter().all(|value| value.is_finite()) {
        return Err(SaddleError::Shape(
            "joint Hessian coordinate does not match the chart".into(),
        ));
    }
    let center = joint_gradient(surface, chart, q, config.stress_step)?;
    let max_gradient = center
        .iter()
        .fold(0.0_f64, |size, value| size.max(value.abs()));
    if max_gradient > config.stationary_tolerance {
        return Err(SaddleError::Invalid(format!(
            "joint Hessian point has gradient {max_gradient:e} above the supplied stationarity \
             tolerance {:e}",
            config.stationary_tolerance
        )));
    }
    let dimension = q.len();
    let mut hessian = Array2::zeros((dimension, dimension));
    for column in 0..dimension {
        let mut plus = q.to_owned();
        let mut minus = q.to_owned();
        plus[column] += config.coordinate_step;
        minus[column] -= config.coordinate_step;
        let g_plus = joint_gradient(surface, chart, plus.view(), config.stress_step)?;
        let g_minus = joint_gradient(surface, chart, minus.view(), config.stress_step)?;
        for row in 0..dimension {
            hessian[(row, column)] = (g_plus[row] - g_minus[row]) / (2.0 * config.coordinate_step);
        }
    }
    for row in 0..dimension {
        for column in 0..row {
            let value = 0.5 * (hessian[(row, column)] + hessian[(column, row)]);
            hessian[(row, column)] = value;
            hessian[(column, row)] = value;
        }
    }
    if !hessian.iter().all(|value| value.is_finite()) {
        return Err(SaddleError::NonFinite("joint Hessian"));
    }
    Ok(JointHessian {
        matrix: hessian,
        max_gradient,
    })
}

fn joint_gradient<S: PointSurface>(
    surface: &S,
    chart: &JointChart,
    q: ArrayView1<f64>,
    stress_step: f64,
) -> Result<Array1<f64>, SaddleError> {
    let (positions, cell) = chart.geometry(q)?;
    let (_, gradient) = surface.eval_in_cell(positions.view(), &m3_to_row9(cell))?;
    let stress = match surface.cauchy_stress(positions.view(), &m3_to_row9(cell))? {
        Some(packed) => {
            if !packed.iter().all(|value| value.is_finite()) {
                return Err(SaddleError::NonFinite("joint Hessian stress"));
            }
            m3_from_row9(packed)
        }
        None => finite_difference_cauchy_stress(surface, positions.view(), cell, stress_step)?,
    };
    chart.gradient(q, gradient.view(), stress)
}

/// Required physical and numerical controls for a joint Vineyard product.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct JointVineyardConfig {
    /// Mass assigned to every component of `(r0, J epsilon)`, in atomic mass units.
    pub mass_amu: f64,
    /// Eigenvalues with absolute value at most this are excluded zero modes.
    pub mode_tolerance: f64,
}

impl Default for JointVineyardConfig {
    fn default() -> Self {
        Self {
            mass_amu: 0.0,
            mode_tolerance: 0.0,
        }
    }
}

with_setters!(JointVineyardConfig {
    with_mass_amu => mass_amu: val f64,
    with_mode_tolerance => mode_tolerance: val f64,
});

impl JointVineyardConfig {
    /// Construct controls without an implicit zero-mode threshold.
    pub const fn new(mass_amu: f64, mode_tolerance: f64) -> Self {
        Self {
            mass_amu,
            mode_tolerance,
        }
    }

    fn validate(self) -> Result<Self, SaddleError> {
        if !(self.mass_amu.is_finite() && self.mass_amu > 0.0) {
            return Err(SaddleError::Invalid(
                "joint Vineyard mass must be positive and finite".into(),
            ));
        }
        if !(self.mode_tolerance.is_finite() && self.mode_tolerance >= 0.0) {
            return Err(SaddleError::Invalid(
                "joint Vineyard mode tolerance must be finite and non-negative".into(),
            ));
        }
        Ok(self)
    }
}

/// Spectrum and prefactor of a variable-cell minimum and saddle.
#[derive(Clone, Debug, PartialEq)]
pub struct JointVineyard {
    /// Vineyard prefactor in hertz.
    pub prefactor_hz: f64,
    /// Complete minimum spectrum in `eV / Angstrom^2`.
    pub minimum_eigenvalues: Array1<f64>,
    /// Complete saddle spectrum in `eV / Angstrom^2`.
    pub saddle_eigenvalues: Array1<f64>,
    /// Number of excluded minimum zero modes.
    pub minimum_zero_modes: usize,
    /// Number of excluded saddle zero modes.
    pub saddle_zero_modes: usize,
    /// The unique unstable saddle eigenvalue.
    pub unstable_eigenvalue: f64,
}

/// Vineyard prefactor from atom-strain Hessians in the same [`JointChart`].
///
/// `mode_tolerance` must bound the Hessian discretization and surface noise.
/// Values in `[-mode_tolerance, mode_tolerance]` are reported and excluded.
/// The minimum must have no lower eigenvalue, the saddle exactly one, both
/// points must have the same number of zero modes, and their positive products
/// must differ by exactly the unstable mode.
pub fn joint_vineyard_prefactor(
    minimum_hessian: ArrayView2<f64>,
    saddle_hessian: ArrayView2<f64>,
    config: JointVineyardConfig,
) -> Result<JointVineyard, SaddleError> {
    let config = config.validate()?;
    if minimum_hessian.nrows() != saddle_hessian.nrows()
        || minimum_hessian.ncols() != saddle_hessian.ncols()
    {
        return Err(SaddleError::Shape(
            "joint Vineyard Hessians must have the same shape".into(),
        ));
    }
    let (minimum_eigenvalues, _) = exact_eigh(minimum_hessian)?;
    let (saddle_eigenvalues, _) = exact_eigh(saddle_hessian)?;
    let minimum = split_spectrum(&minimum_eigenvalues, config.mode_tolerance);
    let saddle = split_spectrum(&saddle_eigenvalues, config.mode_tolerance);
    if !minimum.negative.is_empty() {
        return Err(SaddleError::Invalid(format!(
            "joint Vineyard minimum has {} unstable modes",
            minimum.negative.len()
        )));
    }
    if saddle.negative.len() != 1 {
        return Err(SaddleError::Invalid(format!(
            "joint Vineyard saddle needs exactly one unstable mode; found {}",
            saddle.negative.len()
        )));
    }
    if minimum.zero != saddle.zero {
        return Err(SaddleError::Invalid(format!(
            "joint Vineyard zero-mode counts differ: minimum {}, saddle {}",
            minimum.zero, saddle.zero
        )));
    }
    let prefactor_hz = vineyard_prefactor(&minimum.positive, &saddle.positive, config.mass_amu)?;
    Ok(JointVineyard {
        prefactor_hz,
        minimum_eigenvalues,
        saddle_eigenvalues,
        minimum_zero_modes: minimum.zero,
        saddle_zero_modes: saddle.zero,
        unstable_eigenvalue: saddle.negative[0],
    })
}

struct Spectrum {
    negative: Vec<f64>,
    positive: Vec<f64>,
    zero: usize,
}

fn split_spectrum(values: &Array1<f64>, tolerance: f64) -> Spectrum {
    let mut negative = Vec::new();
    let mut positive = Vec::new();
    let mut zero = 0;
    for &value in values {
        if value < -tolerance {
            negative.push(value);
        } else if value > tolerance {
            positive.push(value);
        } else {
            zero += 1;
        }
    }
    Spectrum {
        negative,
        positive,
        zero,
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::{Array1, ArrayView1, array};

    use super::*;
    use crate::solid_dimer::CellMask;

    const CUBE: [f64; 9] = [4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.0];

    struct AtomStrainWell;

    impl PointSurface for AtomStrainWell {
        fn eval(&self, positions: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            self.eval_in_cell(positions, &CUBE)
        }

        fn eval_in_cell(
            &self,
            positions: ArrayView1<f64>,
            cell: &[f64; 9],
        ) -> Result<(f64, Array1<f64>), SaddleError> {
            let energy = positions[0].powi(2)
                + 1.5 * positions[1].powi(2)
                + 2.0 * positions[2].powi(2)
                + 2.5 * (cell[0] - 4.0).powi(2);
            Ok((
                energy,
                array![2.0 * positions[0], 3.0 * positions[1], 4.0 * positions[2]],
            ))
        }

        fn cauchy_stress(
            &self,
            _positions: ArrayView1<f64>,
            cell: &[f64; 9],
        ) -> Result<Option<[f64; 9]>, SaddleError> {
            let volume = cell[0] * cell[4] * cell[8];
            let mut stress = [0.0; 9];
            stress[0] = 5.0 * (cell[0] - 4.0) * cell[0] / volume;
            Ok(Some(stress))
        }
    }

    #[test]
    fn joint_hessian_contains_the_strain_curvature() {
        let cube = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]];
        let chart = JointChart::new(
            cube,
            1,
            CellMask::voigt([true, false, false, false, false, false]),
            [[0.0; 3]; 3],
            1.0,
        )
        .unwrap();
        let q = Array1::zeros(4);
        let result = joint_hessian(
            &AtomStrainWell,
            &chart,
            q.view(),
            JointHessianConfig::new(1.0e-4, 1.0e-5, 1.0e-12),
        )
        .unwrap();
        for (index, expected) in [2.0, 3.0, 4.0, 5.0].into_iter().enumerate() {
            assert_abs_diff_eq!(result.matrix[(index, index)], expected, epsilon = 1.0e-10);
        }
        assert_abs_diff_eq!(result.max_gradient, 0.0, epsilon = 1.0e-15);
    }

    #[test]
    fn joint_configs_require_explicit_scales_and_support_builders() {
        let cube = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]];
        let chart = JointChart::new(
            cube,
            1,
            CellMask::voigt([true, false, false, false, false, false]),
            [[0.0; 3]; 3],
            1.0,
        )
        .unwrap();
        let q = Array1::zeros(4);
        assert!(
            joint_hessian(
                &AtomStrainWell,
                &chart,
                q.view(),
                JointHessianConfig::default()
            )
            .is_err()
        );
        let hessian = JointHessianConfig::default()
            .with_coordinate_step(1.0e-4)
            .with_stress_step(1.0e-5)
            .with_stationary_tolerance(1.0e-12);
        assert!(joint_hessian(&AtomStrainWell, &chart, q.view(), hessian).is_ok());

        let minimum = Array2::from_diag(&array![0.0, 4.0]);
        let saddle = Array2::from_diag(&array![0.0, -1.0]);
        assert!(
            joint_vineyard_prefactor(
                minimum.view(),
                saddle.view(),
                JointVineyardConfig::default()
            )
            .is_err()
        );
        let vineyard = JointVineyardConfig::default()
            .with_mass_amu(4.0)
            .with_mode_tolerance(1.0e-10);
        assert!(joint_vineyard_prefactor(minimum.view(), saddle.view(), vineyard).is_ok());
    }

    #[test]
    fn joint_vineyard_keeps_strain_and_reports_zero_modes() {
        let minimum = Array2::from_diag(&array![0.0, 0.0, 0.0, 4.0, 9.0]);
        let saddle = Array2::from_diag(&array![0.0, 0.0, 0.0, -1.0, 16.0]);
        let report = joint_vineyard_prefactor(
            minimum.view(),
            saddle.view(),
            JointVineyardConfig::new(4.0, 1.0e-10),
        )
        .unwrap();
        let expected = angular_frequency_factor(4.0) * (4.0_f64 * 9.0 / 16.0).sqrt();
        assert_abs_diff_eq!(report.prefactor_hz, expected, epsilon = expected * 1.0e-12);
        assert_eq!(report.minimum_zero_modes, 3);
        assert_eq!(report.saddle_zero_modes, 3);
        assert_abs_diff_eq!(report.unstable_eigenvalue, -1.0, epsilon = 1.0e-12);
    }

    fn angular_frequency_factor(mass_amu: f64) -> f64 {
        crate::angular_scale_rad_s() / (2.0 * std::f64::consts::PI * mass_amu.sqrt())
    }
}
