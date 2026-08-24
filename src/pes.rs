//! Cartesian PES wrapper: position, gradient, Cartesian BFGS Hessian, kick.
//!
//! Sella `peswrapper.PES` without internals. The Hessian lives in
//! rgmin [`BfgsModel`] and is updated in Cartesian `(s, y)`, matching
//! Sella `_update_H`. IRC keeps a separate MW model on
//! [`crate::irc::IrcSession`].

use ndarray::{Array1, ArrayView1};
use rgmin::vecops::axpy;
use rgmin::BfgsModel;

use crate::error::SaddleError;
use crate::minmode::PointSurface;

/// How [`CartesianPes::kick`] updates the Hessian.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HessUpdate {
    #[default]
    Bfgs,
    TsBfgs,
}

impl HessUpdate {
    /// C / Python ordinal. Unknown values stay out of the enum.
    pub const fn to_abi(self) -> i32 {
        match self {
            Self::Bfgs => 0,
            Self::TsBfgs => 1,
        }
    }

    /// Inverse of [`Self::to_abi`]. Unknown ordinals are `None`.
    pub const fn try_from_abi(v: i32) -> Option<Self> {
        match v {
            0 => Some(Self::Bfgs),
            1 => Some(Self::TsBfgs),
            _ => None,
        }
    }
}

/// Cartesian geometry plus a persistent MW Hessian.
pub struct CartesianPes {
    x: Array1<f64>,
    masses: Array1<f64>,
    hess: BfgsModel,
    update: HessUpdate,
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
            update: HessUpdate::Bfgs,
        })
    }

    pub fn set_update(&mut self, update: HessUpdate) {
        self.update = update;
    }

    pub fn position(&self) -> ArrayView1<'_, f64> {
        self.x.view()
    }

    /// Per-atom masses, length N.
    pub fn masses(&self) -> ArrayView1<'_, f64> {
        self.masses.view()
    }

    pub fn hessian(&self) -> &BfgsModel {
        &self.hess
    }

    pub fn reset(&mut self) {
        self.hess.forget();
    }

    /// Add `d` into the living Cartesian frame. No surface eval.
    pub fn displace(&mut self, d: ArrayView1<f64>) -> Result<(), SaddleError> {
        if d.len() != self.x.len() {
            return Err(SaddleError::Shape(
                "displace d must match the 3N frame".into(),
            ));
        }
        axpy(1.0, d, &mut self.x);
        Ok(())
    }

    /// Sella `PES.kick`: displace, eval, store a Cartesian BFGS pair.
    ///
    /// Sella `_update_H(dx, dg)` is Cartesian. IRC keeps its own MW
    /// [`BfgsModel`] on [`crate::irc::IrcSession`].
    pub fn kick<S: PointSurface>(
        &mut self,
        surface: &S,
        d: ArrayView1<f64>,
    ) -> Result<(f64, Array1<f64>), SaddleError> {
        let (_, g0) = surface.eval(self.x.view())?;
        self.kick_from(surface, d, g0.view())
    }

    /// [`Self::kick`] when the caller already evaluated `g0` at `x`.
    pub fn kick_from<S: PointSurface>(
        &mut self,
        surface: &S,
        d: ArrayView1<f64>,
        g0: ArrayView1<f64>,
    ) -> Result<(f64, Array1<f64>), SaddleError> {
        if g0.len() != self.x.len() {
            return Err(SaddleError::Shape(
                "kick g0 must match the 3N frame".into(),
            ));
        }
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
        match self.update {
            HessUpdate::Bfgs => self.hess.update(&s, &y),
            HessUpdate::TsBfgs => self.hess.update_ts(&s, &y),
        }
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
