//! One surface for a point search and a band.
//!
//! A point search is a one-row band. [`crate::minmode::PointSurface`] and
//! [`crate::band::BandSurface`] stay as they are for existing callers.
//! A type that implements [`Surface`] implements both of those traits.
//! [`FromPoint`] and [`FromBand`] lift an existing surface the other way.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut2};

use crate::band::{BandSurface, Cell};
use crate::error::SaddleError;
use crate::minmode::PointSurface;

/// Stable image identity. Row index is not an identity: insertion and
/// a window renumber rows, and this id does not.
pub type ImageId = u64;

/// Batched rows. `eval` writes `e[row]` and `g.row(row)` for each id.
///
/// Positions are flat Cartesian rows in Angstrom, energies are in eV and gradients
/// in eV/Angstrom. A type that implements this trait also implements
/// [`crate::minmode::PointSurface`] and [`crate::band::BandSurface`].
pub trait Surface: Sync {
    /// Energies and gradients of every row of `x` (`n x dof`, Angstrom).
    ///
    /// `ids` has `n` entries, `e` receives `n` energies in eV and `g` is `n x dof` and
    /// receives the energy gradients in eV/Angstrom (the negative of the forces).
    ///
    /// # Errors
    ///
    /// Any [`SaddleError`] the host raises; the callers do not use `e` or `g` then.
    fn eval(
        &self,
        ids: &[ImageId],
        x: ArrayView2<f64>,
        e: &mut [f64],
        g: ArrayViewMut2<f64>,
    ) -> Result<(), SaddleError>;

    /// Analytic Cartesian Hessian action `H v` in eV/Angstrom^2 times the unit of `v`,
    /// at `x`. The default is `None`, which selects a finite difference. Note the
    /// nesting: `Option<Result<..>>`, the reverse of
    /// [`crate::minmode::PointSurface::hessian_vector`].
    fn hessian_vector(
        &self,
        _x: ArrayView1<f64>,
        _v: ArrayView1<f64>,
    ) -> Option<Result<Array1<f64>, SaddleError>> {
        None
    }

    /// Energies, Cartesian gradients, and Cauchy stresses. `s` is `n × 9`.
    ///
    /// `x` is `n x dof` in Angstrom, `cells` has one cell per row, `e` receives energies
    /// in eV, `g` gradients in eV/Angstrom and `s` the row-major 3x3 Cauchy stress in
    /// eV/Angstrom^3.
    ///
    /// # Errors
    ///
    /// The default returns [`SaddleError::Invalid`]: a surface that does not evaluate
    /// cells refuses.
    fn eval_cells(
        &self,
        _x: ArrayView2<f64>,
        _cells: &[Cell],
        _e: &mut [f64],
        _g: ArrayViewMut2<f64>,
        _s: ArrayViewMut2<f64>,
    ) -> Result<(), SaddleError> {
        Err(SaddleError::Invalid(
            "surface does not evaluate cells".into(),
        ))
    }
}

/// Existing point surface viewed as one row.
///
/// The wrapped [`crate::minmode::PointSurface`] is evaluated once per row, so a
/// multi-row batch is a loop over its rows. [`Surface::eval`] fails with a
/// [`SaddleError::Shape`] when `ids`, `x`, `e` and `g` disagree in length, or the
/// point surface returns a gradient of the wrong length.
pub struct FromPoint<T>(pub T);

impl<T: PointSurface> Surface for FromPoint<T> {
    fn eval(
        &self,
        ids: &[ImageId],
        x: ArrayView2<f64>,
        e: &mut [f64],
        mut g: ArrayViewMut2<f64>,
    ) -> Result<(), SaddleError> {
        check_rows(ids, x, e, g.view())?;
        for (row, _) in ids.iter().enumerate() {
            let (energy, gradient) = self.0.eval(x.row(row))?;
            if gradient.len() != x.ncols() {
                return Err(SaddleError::Shape(
                    "point surface gradient length does not match the row".into(),
                ));
            }
            e[row] = energy;
            g.row_mut(row).assign(&gradient);
        }
        Ok(())
    }

    fn hessian_vector(
        &self,
        x: ArrayView1<f64>,
        v: ArrayView1<f64>,
    ) -> Option<Result<Array1<f64>, SaddleError>> {
        match self.0.hessian_vector(x, v) {
            Ok(Some(action)) => Some(Ok(action)),
            Ok(None) => None,
            Err(err) => Some(Err(err)),
        }
    }
}

/// Existing band surface viewed as a batch. The old trait has no image
/// ids, so the rows are evaluated in the order given.
///
/// The whole batch goes to [`crate::band::BandSurface::eval`] in one call.
/// [`Surface::eval`] fails with a [`SaddleError::Shape`] when the lengths disagree.
pub struct FromBand<T>(pub T);

impl<T: BandSurface> Surface for FromBand<T> {
    fn eval(
        &self,
        ids: &[ImageId],
        x: ArrayView2<f64>,
        e: &mut [f64],
        mut g: ArrayViewMut2<f64>,
    ) -> Result<(), SaddleError> {
        check_rows(ids, x, e, g.view())?;
        let mut energies = Array1::zeros(x.nrows());
        let mut gradients = Array2::zeros(x.dim());
        self.0.eval(x, &mut energies, &mut gradients)?;
        e.copy_from_slice(
            energies
                .as_slice()
                .ok_or_else(|| SaddleError::Shape("band energies are not contiguous".into()))?,
        );
        g.assign(&gradients);
        Ok(())
    }

    fn eval_cells(
        &self,
        x: ArrayView2<f64>,
        cells: &[Cell],
        e: &mut [f64],
        mut g: ArrayViewMut2<f64>,
        mut s: ArrayViewMut2<f64>,
    ) -> Result<(), SaddleError> {
        if e.len() != x.nrows() || g.dim() != x.dim() || s.dim() != (x.nrows(), 9) {
            return Err(SaddleError::Shape(
                "cell evaluation rows do not match the positions".into(),
            ));
        }
        let mut energies = Array1::zeros(x.nrows());
        let mut gradients = Array2::zeros(x.dim());
        let mut stresses = Array2::zeros((x.nrows(), 9));
        self.0
            .eval_cells(x, cells, &mut energies, &mut gradients, &mut stresses)?;
        e.copy_from_slice(
            energies
                .as_slice()
                .ok_or_else(|| SaddleError::Shape("band energies are not contiguous".into()))?,
        );
        g.assign(&gradients);
        s.assign(&stresses);
        Ok(())
    }
}

fn check_rows(
    ids: &[ImageId],
    x: ArrayView2<f64>,
    e: &[f64],
    g: ArrayView2<f64>,
) -> Result<(), SaddleError> {
    if ids.len() != x.nrows() || e.len() != x.nrows() || g.dim() != x.dim() {
        return Err(SaddleError::Shape(
            "surface rows, ids, energies, and gradients must match".into(),
        ));
    }
    Ok(())
}

impl<T: Surface + ?Sized> PointSurface for T {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let mut row = Array2::zeros((1, x.len()));
        row.row_mut(0).assign(&x);
        let mut energy = [0.0];
        let mut gradient = Array2::zeros((1, x.len()));
        Surface::eval(self, &[0], row.view(), &mut energy, gradient.view_mut())?;
        Ok((energy[0], gradient.row(0).to_owned()))
    }

    fn hessian_vector(
        &self,
        x: ArrayView1<f64>,
        v: ArrayView1<f64>,
    ) -> Result<Option<Array1<f64>>, SaddleError> {
        match Surface::hessian_vector(self, x, v) {
            None => Ok(None),
            Some(result) => result.map(Some),
        }
    }
}

impl<T: Surface + ?Sized> BandSurface for T {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        let ids: Vec<ImageId> = (0..positions.nrows() as ImageId).collect();
        self.eval_with_ids(&ids, positions, energies, gradients)
    }

    fn eval_with_ids(
        &self,
        ids: &[ImageId],
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        if ids.len() != positions.nrows()
            || energies.len() != positions.nrows()
            || gradients.dim() != positions.dim()
        {
            return Err(SaddleError::Shape(
                "band surface rows do not match the positions".into(),
            ));
        }
        let slice = energies
            .as_slice_mut()
            .ok_or_else(|| SaddleError::Shape("band energies are not contiguous".into()))?;
        Surface::eval(self, ids, positions, slice, gradients.view_mut())?;
        Ok(())
    }

    fn eval_cells(
        &self,
        positions: ArrayView2<f64>,
        cells: &[Cell],
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
        stresses: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        if energies.len() != positions.nrows()
            || gradients.dim() != positions.dim()
            || stresses.dim() != (positions.nrows(), 9)
        {
            return Err(SaddleError::Shape(
                "solid-state rows do not match the positions".into(),
            ));
        }
        let slice = energies
            .as_slice_mut()
            .ok_or_else(|| SaddleError::Shape("band energies are not contiguous".into()))?;
        Surface::eval_cells(
            self,
            positions,
            cells,
            slice,
            gradients.view_mut(),
            stresses.view_mut(),
        )?;
        Ok(())
    }

    fn hessian_vector(
        &self,
        positions: ArrayView2<f64>,
        direction: ArrayView2<f64>,
    ) -> Result<Option<Array2<f64>>, SaddleError> {
        let rows = direction.nrows();
        let cols = direction.ncols();
        if positions.nrows() != rows + 2 || positions.ncols() != cols {
            return Err(SaddleError::Shape(
                "band Hessian product rows do not match the positions".into(),
            ));
        }
        if rows == 0 {
            return Ok(None);
        }
        let mut out = Array2::zeros((rows, cols));
        for row in 0..rows {
            match Surface::hessian_vector(self, positions.row(row + 1), direction.row(row)) {
                None => {
                    if row > 0 {
                        return Err(SaddleError::Surface(
                            "analytic Hessian product stopped after earlier rows".into(),
                        ));
                    }
                    return Ok(None);
                }
                Some(result) => {
                    let action = result?;
                    if action.len() != cols || action.iter().any(|value| !value.is_finite()) {
                        return Err(if action.len() != cols {
                            SaddleError::Shape(
                                "analytic Hessian product length does not match the row".into(),
                            )
                        } else {
                            SaddleError::NonFinite("analytic Hessian product")
                        });
                    }
                    out.row_mut(row).assign(&action);
                }
            }
        }
        Ok(Some(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    struct Well;

    impl Surface for Well {
        fn eval(
            &self,
            ids: &[ImageId],
            x: ArrayView2<f64>,
            e: &mut [f64],
            mut g: ArrayViewMut2<f64>,
        ) -> Result<(), SaddleError> {
            check_rows(ids, x, e, g.view())?;
            for (row, id) in ids.iter().enumerate() {
                let _ = id;
                let x0 = x[(row, 0)];
                e[row] = x0 * x0;
                g[(row, 0)] = 2.0 * x0;
            }
            Ok(())
        }
    }

    struct Legacy;

    impl PointSurface for Legacy {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            Ok((x[0] * x[0], array![2.0 * x[0]]))
        }
    }

    #[test]
    fn a_surface_is_a_point_and_a_band() {
        let well = Well;
        let (energy, gradient) = PointSurface::eval(&well, array![0.5].view()).unwrap();
        assert_eq!(energy.to_le_bytes(), (0.25_f64).to_le_bytes());
        assert_eq!(gradient[0].to_le_bytes(), (1.0_f64).to_le_bytes());

        let positions = array![[0.5], [-1.0]];
        let mut energies = Array1::zeros(2);
        let mut gradients = Array2::zeros((2, 1));
        BandSurface::eval(&well, positions.view(), &mut energies, &mut gradients).unwrap();
        assert_eq!(energies[0].to_le_bytes(), (0.25_f64).to_le_bytes());
        assert_eq!(energies[1].to_le_bytes(), (1.0_f64).to_le_bytes());
        assert_eq!(gradients[(1, 0)].to_le_bytes(), (-2.0_f64).to_le_bytes());
    }

    #[test]
    fn a_point_surface_lifts_without_changing_the_value() {
        let lifted = FromPoint(Legacy);
        let x = array![[-1.5], [0.25]];
        let mut e = [0.0, 0.0];
        let mut g = Array2::zeros((2, 1));
        Surface::eval(&lifted, &[7, 9], x.view(), &mut e, g.view_mut()).unwrap();
        assert_eq!(e[0].to_le_bytes(), (2.25_f64).to_le_bytes());
        assert_eq!(g[(1, 0)].to_le_bytes(), (0.5_f64).to_le_bytes());
    }
}
