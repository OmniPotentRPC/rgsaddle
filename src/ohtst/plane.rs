//! Mass-weighted and fixed-cell surfaces used with the shared hyperplane.

use ndarray::{Array1, Array2, ArrayView1};

use crate::error::SaddleError;
use crate::mass_metric::MassMetric;
use crate::minmode::PointSurface;
pub use crate::projector::Hyperplane;
pub(crate) use crate::projector::orthonormal_rows;
use crate::projector::sqrt_masses;

/// The three rigid-translation rows in mass-weighted coordinates.
///
/// `masses` and `n_atoms` both use one value per atom. The returned rows have
/// shape `(3, 3 * n_atoms)`: x, y and z translations with `sqrt(m_i)` in each
/// atom block, normalized to unit norm.
pub fn mass_weighted_rigid_translations(
    masses: &[f64],
    n_atoms: usize,
) -> Result<Array2<f64>, SaddleError> {
    if n_atoms == 0 {
        return Err(SaddleError::Invalid(
            "mass-weighted rigid translations need at least one atom".into(),
        ));
    }
    if masses.len() != n_atoms {
        return Err(SaddleError::Shape(
            "mass-weighted rigid translations need one mass per atom".into(),
        ));
    }
    if !masses.iter().all(|mass| mass.is_finite() && *mass > 0.0) {
        return Err(SaddleError::Invalid(
            "masses must be finite and positive".into(),
        ));
    }
    let metric = MassMetric::atoms(ArrayView1::from(masses), 3 * n_atoms)?;
    let normalization = masses.iter().sum::<f64>().sqrt();
    let mut rows = Array2::zeros((3, 3 * n_atoms));
    for atom in 0..n_atoms {
        let weight = metric.sqrt_at(3 * atom) / normalization;
        let base = 3 * atom;
        rows[(0, base)] = weight;
        rows[(1, base + 1)] = weight;
        rows[(2, base + 2)] = weight;
    }
    Ok(rows)
}

impl Hyperplane {
    /// Make the normal orthogonal to rigid translations under masses `masses`.
    pub fn orthogonal_to_mass_weighted_translations(
        &self,
        masses: &[f64],
        n_atoms: usize,
    ) -> Result<Self, SaddleError> {
        let translations = mass_weighted_rigid_translations(masses, n_atoms)?;
        self.orthogonal_to(translations.view())
    }
}

/// A Cartesian surface in mass-weighted coordinates `q = M^{1/2} x`.
///
/// The square-root weights are a [`MassMetric`]. A point is multiplied by
/// them and a gradient, which is a covector, is divided by them. The energy
/// is `V(M^{-1/2} q)` and the gradient `M^{-1/2} ∇V`. Unit mass in these
/// coordinates is the Cartesian dynamics with masses `M`, so the plane
/// sampler and the shooting step take mass 1 here. Hessian actions and
/// excluded modes are mapped when the wrapped surface has them.
pub struct MassWeighted<'a, S> {
    surface: &'a S,
    metric: MassMetric,
}

impl<'a, S: PointSurface> MassWeighted<'a, S> {
    /// One mass per coordinate.
    pub fn new(surface: &'a S, masses: &[f64]) -> Result<Self, SaddleError> {
        let sqrt_mass = sqrt_masses(masses, masses.len())?;
        Ok(Self {
            surface,
            metric: MassMetric::from_square_roots(sqrt_mass.view())?,
        })
    }

    /// One mass per atom, repeated over its three coordinates.
    pub fn per_atom(surface: &'a S, masses: &[f64]) -> Result<Self, SaddleError> {
        let expanded: Vec<f64> = masses.iter().flat_map(|m| [*m; 3]).collect();
        Self::new(surface, &expanded)
    }

    /// Masses, one per coordinate.
    pub fn masses(&self) -> Array1<f64> {
        self.metric.sqrt_vector().mapv(|root| root * root)
    }

    /// `q = M^{1/2} x`.
    pub fn to_mass_weighted(&self, x: ArrayView1<f64>) -> Result<Array1<f64>, SaddleError> {
        self.check(x.len())?;
        self.metric.multiply_sqrt(x)
    }

    /// `x = M^{-1/2} q`.
    pub fn to_cartesian(&self, q: ArrayView1<f64>) -> Result<Array1<f64>, SaddleError> {
        self.check(q.len())?;
        self.metric.divide_sqrt(q)
    }

    fn check(&self, n: usize) -> Result<(), SaddleError> {
        if n == self.metric.len() {
            Ok(())
        } else {
            Err(SaddleError::Shape(
                "mass-weighted surface length differs from the masses".into(),
            ))
        }
    }
}

impl<S: PointSurface> PointSurface for MassWeighted<'_, S> {
    fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let x = self.to_cartesian(q)?;
        let (energy, gradient) = self.surface.eval(x.view())?;
        if gradient.len() != q.len() {
            return Err(SaddleError::Shape(
                "surface gradient length differs from the masses".into(),
            ));
        }
        Ok((energy, self.metric.divide_sqrt(gradient.view())?))
    }

    fn hessian_vector(
        &self,
        q: ArrayView1<f64>,
        v: ArrayView1<f64>,
    ) -> Result<Option<Array1<f64>>, SaddleError> {
        let x = self.to_cartesian(q)?;
        let u = self.to_cartesian(v)?;
        let Some(action) = self.surface.hessian_vector(x.view(), u.view())? else {
            return Ok(None);
        };
        if action.len() != q.len() {
            return Err(SaddleError::Shape(
                "surface Hessian action length differs from the masses".into(),
            ));
        }
        Ok(Some(self.metric.divide_sqrt(action.view())?))
    }

    fn excluded_modes(&self, q: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
        let x = self.to_cartesian(q)?;
        let mut modes = self.surface.excluded_modes(x.view())?;
        if modes.nrows() > 0 && modes.ncols() != q.len() {
            return Err(SaddleError::Shape(
                "excluded modes differ in length from the masses".into(),
            ));
        }
        for mut row in modes.rows_mut() {
            let weighted = self.metric.multiply_sqrt(row.view())?;
            row.assign(&weighted);
        }
        Ok(modes)
    }
}

/// A surface evaluated in one fixed periodic cell, so that planes are
/// sampled at a set volume, for example at the lattice parameter of each
/// temperature. [`PointSurface::eval`] goes to
/// [`PointSurface::eval_in_cell`] with the stored cell. Hessian actions are
/// left to gradient differences, which see the cell; an analytic action of
/// the inner surface might not.
pub struct FixedCell<'a, S> {
    surface: &'a S,
    cell: [f64; 9],
}

impl<'a, S: PointSurface> FixedCell<'a, S> {
    /// `cell` is row-major 3x3, the layout of
    /// [`PointSurface::eval_in_cell`], with a nonzero volume.
    pub fn new(surface: &'a S, cell: [f64; 9]) -> Result<Self, SaddleError> {
        if !cell.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("cell"));
        }
        let volume = cell[0] * (cell[4] * cell[8] - cell[5] * cell[7])
            - cell[1] * (cell[3] * cell[8] - cell[5] * cell[6])
            + cell[2] * (cell[3] * cell[7] - cell[4] * cell[6]);
        if volume == 0.0 {
            return Err(SaddleError::Invalid("the cell has no volume".into()));
        }
        Ok(Self { surface, cell })
    }

    /// The cell, row-major.
    pub fn cell(&self) -> &[f64; 9] {
        &self.cell
    }
}

impl<S: PointSurface> PointSurface for FixedCell<'_, S> {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        self.surface.eval_in_cell(x, &self.cell)
    }

    fn excluded_modes(&self, x: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
        self.surface.excluded_modes(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ohtst::sample::{LangevinConfig, sample_plane};
    use ndarray::array;

    struct Quadratic;

    impl PointSurface for Quadratic {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let e = 0.5 * x[0] * x[0] + x[0] * x[1] + 2.0 * x[1] * x[1];
            Ok((e, array![x[0] + x[1], x[0] + 4.0 * x[1]]))
        }

        fn excluded_modes(&self, x: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
            let mut modes = Array2::zeros((1, x.len()));
            modes[(0, 0)] = 1.0;
            modes[(0, 1)] = 1.0;
            Ok(modes)
        }
    }

    fn center_of_mass(position: ArrayView1<'_, f64>, masses: &[f64]) -> [f64; 3] {
        let mut com = [0.0; 3];
        let total_mass = masses.iter().sum::<f64>();
        for (atom, mass) in masses.iter().enumerate() {
            let weight = mass.sqrt();
            let base = 3 * atom;
            com[0] += weight * position[base];
            com[1] += weight * position[base + 1];
            com[2] += weight * position[base + 2];
        }
        for value in &mut com {
            *value /= total_mass;
        }
        com
    }

    fn rigid_translation_axis(normal: ArrayView1<'_, f64>, masses: &[f64]) -> [f64; 3] {
        let mut axis = [0.0; 3];
        for (atom, mass) in masses.iter().enumerate() {
            let weight = mass.sqrt();
            let base = 3 * atom;
            axis[0] += weight * normal[base];
            axis[1] += weight * normal[base + 1];
            axis[2] += weight * normal[base + 2];
        }
        let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        if norm > 0.0 {
            axis[0] /= norm;
            axis[1] /= norm;
            axis[2] /= norm;
        }
        axis
    }

    #[test]
    fn cartesian_plane_maps_to_the_same_set() {
        let masses = [1.0, 3.0];
        let x0 = array![0.4, -0.2];
        let n = array![1.0, 1.0];
        let plane = Hyperplane::from_cartesian(x0.view(), n.view(), &masses).unwrap();
        let along = array![1.0, -1.0];
        for t in [-1.5, 0.3, 2.0] {
            let x = &x0 + &(&along * t);
            let q = array![x[0] * masses[0].sqrt(), x[1] * masses[1].sqrt()];
            assert!(plane.distance(q.view()).abs() < 1e-14);
        }
        let (back_point, back_normal) = plane.to_cartesian(&masses).unwrap();
        assert!((&back_point - &x0).iter().all(|d| d.abs() < 1e-14));
        let unit = &n / n.dot(&n).sqrt();
        assert!((&back_normal - &unit).iter().all(|d| d.abs() < 1e-14));
    }

    #[test]
    fn packing_recentering_and_orientation_keep_the_plane() {
        let plane = Hyperplane::new(array![1.0, 2.0, 0.0], array![0.0, 3.0, 4.0]).unwrap();
        assert!((plane.normal().dot(&plane.normal()) - 1.0).abs() < 1e-15);
        let back = Hyperplane::from_packed(plane.packed().view()).unwrap();
        assert_eq!(back, plane);
        let moved = plane.recentered(array![5.0, -1.0, 7.0].view());
        assert!(plane.distance(moved.point()).abs() < 1e-14);
        let flipped = plane.clone().oriented_toward(array![1.0, 0.0, 0.0].view());
        assert!(flipped.distance(array![1.0, 0.0, 0.0].view()) > 0.0);
        assert!(
            (&flipped.normal() + &plane.normal())
                .iter()
                .all(|d| d.abs() < 1e-15)
        );
        assert!(Hyperplane::new(array![0.0], array![0.0]).is_err());
        assert!(Hyperplane::new(array![0.0, 1.0], array![1.0]).is_err());
    }

    #[test]
    fn mass_weighted_translations_are_orthonormal() {
        let rows = mass_weighted_rigid_translations(&[1.0, 4.0, 9.0], 3).unwrap();
        assert_eq!(rows.dim(), (3, 9));
        for i in 0..3 {
            for j in 0..3 {
                let dot = rows.row(i).dot(&rows.row(j));
                let target = if i == j { 1.0 } else { 0.0 };
                assert!((dot - target).abs() < 1.0e-14);
            }
        }
        assert!(mass_weighted_rigid_translations(&[], 0).is_err());
        assert!(mass_weighted_rigid_translations(&[1.0], 2).is_err());
        assert!(mass_weighted_rigid_translations(&[1.0, 0.0], 2).is_err());
    }

    #[test]
    fn hyperplane_can_drop_mass_weighted_translations_from_its_normal() {
        let masses = [2.0, 5.0];
        let plane = Hyperplane::new(
            array![0.0, 0.0, 0.0, 0.2, -0.1, 0.3],
            array![1.0, 0.2, -0.4, 0.8, 0.1, 0.3],
        )
        .unwrap();
        let orthogonal = plane
            .orthogonal_to_mass_weighted_translations(&masses, masses.len())
            .unwrap();
        let translations = mass_weighted_rigid_translations(&masses, masses.len()).unwrap();
        for row in translations.rows() {
            assert!(row.dot(&orthogonal.normal()).abs() < 1.0e-12);
        }
    }

    struct TranslationInvariant {
        translations: Array2<f64>,
    }

    impl PointSurface for TranslationInvariant {
        fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            Ok((0.0, Array1::zeros(q.len())))
        }

        fn excluded_modes(&self, _q: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
            Ok(self.translations.clone())
        }
    }

    #[test]
    fn fixed_plane_sampling_has_no_com_drift_along_the_normal() {
        let masses = [1.0, 3.0];
        let translations = mass_weighted_rigid_translations(&masses, masses.len()).unwrap();
        let surface = TranslationInvariant {
            translations: translations.clone(),
        };
        let point = array![0.1, -0.2, 0.3, 0.6, 0.5, -0.4];
        let raw_normal = array![0.7, 0.3, -0.2, 0.4, 0.1, 0.8];
        let axis = rigid_translation_axis(raw_normal.view(), &masses);
        let axis_norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        assert!(axis_norm > 0.0);
        let plane = Hyperplane::new(point.clone(), raw_normal)
            .unwrap()
            .orthogonal_to_mass_weighted_translations(&masses, masses.len())
            .unwrap();
        let config = LangevinConfig::default()
            .with_thermal(0.02)
            .with_time_step(0.3)
            .with_records(200)
            .with_blocks(4)
            .with_seed(11)
            .with_retain(4);
        let sample = sample_plane(&surface, &plane, None, &config).unwrap();
        let com0 = center_of_mass(point.view(), &masses);
        for q in sample.retained.iter().chain(std::iter::once(&sample.last)) {
            let com = center_of_mass(q.view(), &masses);
            let drift = (com[0] - com0[0]) * axis[0]
                + (com[1] - com0[1]) * axis[1]
                + (com[2] - com0[2]) * axis[2];
            assert!(drift.abs() < 1.0e-12, "COM drift along normal is {drift}");
        }
    }

    #[test]
    fn mass_weighted_gradient_and_modes() {
        let masses = [2.0, 0.5];
        let surface = MassWeighted::new(&Quadratic, &masses).unwrap();
        let q = array![0.3, -0.7];
        let (_, g) = surface.eval(q.view()).unwrap();
        let h = 1e-6;
        for i in 0..2 {
            let mut plus = q.clone();
            let mut minus = q.clone();
            plus[i] += h;
            minus[i] -= h;
            let fd = (surface.eval(plus.view()).unwrap().0 - surface.eval(minus.view()).unwrap().0)
                / (2.0 * h);
            assert!((fd - g[i]).abs() < 1e-9);
        }
        let modes = surface.excluded_modes(q.view()).unwrap();
        assert!((modes[(0, 0)] - 2.0f64.sqrt()).abs() < 1e-15);
        assert!((modes[(0, 1)] - 0.5f64.sqrt()).abs() < 1e-15);
        assert!(
            surface
                .hessian_vector(q.view(), q.view())
                .unwrap()
                .is_none()
        );
        assert!(MassWeighted::new(&Quadratic, &[1.0, 0.0]).is_err());
    }

    #[test]
    fn mass_weighted_maps_match_direct_square_roots() {
        let masses = [2.0, 0.5];
        let surface = MassWeighted::new(&Quadratic, &masses).unwrap();
        let cartesian = array![0.3, -0.7];
        let sqrt_mass = array![masses[0].sqrt(), masses[1].sqrt()];
        let weighted = &cartesian * &sqrt_mass;
        let mapped = surface.to_mass_weighted(cartesian.view()).unwrap();
        for (actual, expected) in mapped.iter().zip(weighted.iter()) {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
        let back = surface.to_cartesian(weighted.view()).unwrap();
        for (actual, expected) in back.iter().zip(cartesian.iter()) {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
        let (_, gradient) = Quadratic.eval(cartesian.view()).unwrap();
        let expected_gradient = &gradient / &sqrt_mass;
        let (_, actual_gradient) = surface.eval(weighted.view()).unwrap();
        for (actual, expected) in actual_gradient.iter().zip(expected_gradient.iter()) {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
        let modes = surface.excluded_modes(weighted.view()).unwrap();
        assert_eq!(modes[(0, 0)].to_bits(), sqrt_mass[0].to_bits());
        assert_eq!(modes[(0, 1)].to_bits(), sqrt_mass[1].to_bits());

        let atom_masses = [1.0, 4.0, 9.0];
        let rows = mass_weighted_rigid_translations(&atom_masses, atom_masses.len()).unwrap();
        let normalization = atom_masses.iter().sum::<f64>().sqrt();
        for (atom, mass) in atom_masses.iter().enumerate() {
            let weight = mass.sqrt() / normalization;
            assert_eq!(rows[(0, 3 * atom)].to_bits(), weight.to_bits());
            assert_eq!(rows[(1, 3 * atom + 1)].to_bits(), weight.to_bits());
            assert_eq!(rows[(2, 3 * atom + 2)].to_bits(), weight.to_bits());
            assert_eq!(rows[(0, 3 * atom + 1)].to_bits(), 0.0_f64.to_bits());
        }
    }

    /// Energy `½|x|² + c₀₀`, gradient `x`: the first cell entry stands in
    /// for a volume-dependent energy.
    struct CellQuadratic;

    impl PointSurface for CellQuadratic {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            self.eval_in_cell(x, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        }

        fn eval_in_cell(
            &self,
            x: ArrayView1<f64>,
            cell: &[f64; 9],
        ) -> Result<(f64, Array1<f64>), SaddleError> {
            Ok((0.5 * x.dot(&x) + cell[0], x.to_owned()))
        }

        fn hessian_vector(
            &self,
            _x: ArrayView1<f64>,
            v: ArrayView1<f64>,
        ) -> Result<Option<Array1<f64>>, SaddleError> {
            Ok(Some(v.to_owned()))
        }
    }

    #[test]
    fn fixed_cell_evaluates_in_its_cell() {
        let cell = [1.1, 0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 0.0, 1.1];
        let surface = FixedCell::new(&CellQuadratic, cell).unwrap();
        let x = array![0.3, -0.4];
        let (energy, gradient) = surface.eval(x.view()).unwrap();
        assert!((energy - (0.125 + 1.1)).abs() < 1e-15);
        assert_eq!(gradient, x);
        assert_eq!(surface.cell(), &cell);
        assert!(
            surface
                .hessian_vector(x.view(), x.view())
                .unwrap()
                .is_none()
        );
        let flat = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        assert!(FixedCell::new(&CellQuadratic, flat).is_err());
        let mut broken = cell;
        broken[4] = f64::NAN;
        assert!(FixedCell::new(&CellQuadratic, broken).is_err());
    }

    struct FixedHessian {
        action: Array1<f64>,
    }

    impl PointSurface for FixedHessian {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            Ok((0.0, Array1::zeros(x.len())))
        }

        fn hessian_vector(
            &self,
            _x: ArrayView1<f64>,
            _v: ArrayView1<f64>,
        ) -> Result<Option<Array1<f64>>, SaddleError> {
            Ok(Some(self.action.clone()))
        }
    }

    fn assert_hessian_shape(masses: &[f64], action: Array1<f64>) {
        let inner = FixedHessian { action };
        let surface = MassWeighted::new(&inner, masses).unwrap();
        let q = Array1::zeros(masses.len());
        let result = surface.hessian_vector(q.view(), q.view());
        assert!(
            matches!(
                result,
                Err(SaddleError::Shape(ref message))
                    if message.contains("surface Hessian action length differs from the masses")
            ),
            "{result:?}"
        );
    }

    #[test]
    fn mass_weighted_surface_rejects_a_short_hessian_action() {
        assert_hessian_shape(&[4.0, 9.0], array![1.0]);
        assert_hessian_shape(&[1.0, 1.0], array![1.0]);
    }

    #[test]
    fn mass_weighted_surface_rejects_a_long_hessian_action() {
        assert_hessian_shape(&[1.0, 1.0], array![1.0, 2.0, 3.0]);
        assert_hessian_shape(&[4.0, 9.0], array![1.0, 2.0, 3.0]);
    }

    #[test]
    fn mass_weighted_hessian_divides_a_matched_action_by_sqrt_mass() {
        let inner = FixedHessian {
            action: array![2.0, 9.0],
        };
        let surface = MassWeighted::new(&inner, &[4.0, 9.0]).unwrap();
        let q = array![0.1, -0.2];
        let action = surface
            .hessian_vector(q.view(), q.view())
            .unwrap()
            .expect("analytic action");
        assert_eq!(action.as_slice(), Some(&[1.0, 3.0][..]));
    }
}
