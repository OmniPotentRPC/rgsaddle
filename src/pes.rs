//! Cartesian PES wrapper: position, gradient, MW-BFGS Hessian, kick.
//!
//! Sella `peswrapper.PES` without internals. The Hessian lives in
//! rgmin [`BfgsModel`]. `kick` is a displacement plus a pair update.
//! Rigid projection is [`rgmin::ManifoldKind::MwRigid`] on the
//! session, not here.

use ndarray::{Array1, ArrayView1};
use rgmin::{mw_pair, sqrt_masses_3n, BfgsModel};

use crate::error::SaddleError;
use crate::minmode::PointSurface;

/// Cartesian geometry plus a persistent MW Hessian.
pub struct CartesianPes {
    x: Array1<f64>,
    masses: Array1<f64>,
    hess: BfgsModel,
}

impl CartesianPes {
    /// `x` is 3N Cartesian; `masses` is length N.
    pub fn new(x: Array1<f64>, masses: Array1<f64>) -> Result<Self, SaddleError> {
        if x.is_empty() || x.len() % 3 != 0 {
            return Err(SaddleError::Shape(
                "saddle must be a nonzero 3N Cartesian".into(),
            ));
        }
        if masses.len() * 3 != x.len() {
            return Err(SaddleError::Shape(
                "masses (N) must match the 3N Cartesian".into(),
            ));
        }
        let n = x.len();
        Ok(Self {
            x,
            masses,
            hess: BfgsModel::identity(n),
        })
    }

    pub fn position(&self) -> ArrayView1<'_, f64> {
        self.x.view()
    }

    pub fn hessian(&self) -> &BfgsModel {
        &self.hess
    }

    pub fn reset(&mut self) {
        self.hess.forget();
    }

    /// Sella `PES.kick`: displace, eval, store a MW BFGS pair.
    pub fn kick<S: PointSurface>(
        &mut self,
        surface: &S,
        d: ArrayView1<f64>,
    ) -> Result<(f64, Array1<f64>), SaddleError> {
        let (_, g0) = surface.eval(self.x.view())?;
        let n = self.x.len().min(d.len());
        for i in 0..n {
            self.x[i] += d[i];
        }
        let (e, g1) = surface.eval(self.x.view())?;
        if !g1.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("pes gradient"));
        }
        let y = &g1 - &g0;
        let s = d.slice(ndarray::s![..n]).to_owned();
        let sqrtm = sqrt_masses_3n(self.masses.as_slice().unwrap_or(&[]));
        let (sm, ym) = mw_pair(&s, &y, &sqrtm);
        self.hess.update(&sm, &ym);
        Ok((e, g1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    struct Well;
    impl PointSurface for Well {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let t = x[0];
            let mut g = Array1::zeros(x.len());
            g[0] = 4.0 * t * (t * t - 1.0);
            Ok(((t * t - 1.0).powi(2), g))
        }
    }

    #[test]
    fn kick_moves_and_records_a_pair() {
        let mut pes = CartesianPes::new(Array1::zeros(6), Array1::from(vec![1.0, 1.0])).unwrap();
        let mut d = Array1::zeros(6);
        d[0] = 0.1;
        let (e, g) = pes.kick(&Well, d.view()).unwrap();
        assert!((pes.position()[0] - 0.1).abs() < 1e-14);
        assert!(e > 0.0);
        assert!(g[0] < 0.0);
        assert!(pes.hessian().hessian().nrows() == 6);
    }
}
