//! Order-1 P-RFO climb of one band image on a dense TS-BFGS model.
//!
//! Baker, J. Comput. Chem. 7, 385 (1986), doi:10.1002/jcc.540070402,
//! for the partitioned step; the TS-BFGS update and the trust schedule
//! are Sella's (Hermes, Sarsfield, and Zádor, J. Chem. Theory Comput.
//! 18, 6974 (2022), doi:10.1021/acs.jctc.2c00395). The trust radius
//! bounds the largest per-atom move and never exceeds the band's
//! `max_move`.
//!
//! The model starts from the identity scaled by the smallest positive
//! curvature measured over the Lanczos block, so it is not stiffer
//! than the surface along any sampled direction, and the block is then
//! imposed exactly ([`super::pair_model::impose`]).

use ndarray::{Array1, Array2};

use super::pair_model::impose;
use crate::error::SaddleError;
use crate::geom::{TrustSchedule, update_trust};
use crate::prfo::PartitionedRationalFunctionOptimization;
use crate::restricted::ras_cons;

pub(super) struct ModelClimb {
    b: Array2<f64>,
    spectrum: Option<(Array1<f64>, Array2<f64>)>,
    trust: f64,
    /// Energy, predicted change and per-atom size of the last step.
    pending: Option<(f64, f64, f64)>,
}

impl ModelClimb {
    /// `mu I` on `n` coordinates, `mu` the smallest `v.Hv / v.v` over
    /// the pairs of `block` with positive curvature (1 when there is
    /// none), then `block` imposed. Trust radius `max_move`.
    pub(super) fn fitted(n: usize, block: &[(Array1<f64>, Array1<f64>)], max_move: f64) -> Self {
        let p = Array2::<f64>::eye(n);
        let mut mu = f64::INFINITY;
        for (v, hv) in block {
            let measured = v.dot(hv);
            let model = v.dot(&p.dot(v));
            if measured > 0.0 && model > 0.0 {
                mu = mu.min(measured / model);
            }
        }
        if !mu.is_finite() {
            mu = 1.0;
        }
        let mut climb = Self {
            b: p * mu,
            spectrum: None,
            trust: max_move,
            pending: None,
        };
        climb.impose(block);
        climb
    }

    /// Make the model reproduce every measured pair of `block`.
    pub(super) fn impose(&mut self, block: &[(Array1<f64>, Array1<f64>)]) {
        impose(&mut self.b, block);
        self.spectrum = None;
    }

    /// TS-BFGS on the last step and the Sella trust update from the
    /// ratio of the actual to the predicted energy change.
    pub(super) fn update(&mut self, s: &Array1<f64>, y: &Array1<f64>, energy: f64) {
        if let Some((e0, predicted, size)) = self.pending.take()
            && predicted.abs() >= 1e-14
        {
            let rho = (energy - e0) / predicted;
            self.trust = update_trust(self.trust, rho, size, &TrustSchedule::saddle());
        }
        rgmin::ts_bfgs_update(&mut self.b, s, y);
        self.spectrum = None;
    }

    fn spectrum(&mut self) -> Result<&(Array1<f64>, Array2<f64>), SaddleError> {
        if self.spectrum.is_none() {
            self.spectrum = Some(eigh(&self.b)?);
        }
        Ok(self.spectrum.as_ref().expect("spectrum"))
    }

    /// The model's lowest eigenpair.
    pub(super) fn lowest(&mut self) -> Result<(Array1<f64>, f64), SaddleError> {
        let (vals, vecs) = self.spectrum()?;
        Ok((vecs.column(0).to_owned(), vals[0]))
    }

    /// The P-RFO step from `gradient` at `energy`, each atom's move at
    /// most the trust radius and `max_move`.
    pub(super) fn step(
        &mut self,
        gradient: &Array1<f64>,
        energy: f64,
        max_move: f64,
    ) -> Result<Array1<f64>, SaddleError> {
        self.trust = self.trust.min(max_move);
        let trust = self.trust;
        let (vals, vecs) = self.spectrum()?;
        let s = PartitionedRationalFunctionOptimization::new(1).get_s(
            vals,
            vecs,
            gradient,
            PartitionedRationalFunctionOptimization::ALPHA0,
        );
        let s = rgmin::ras_clip(&s, trust);
        let predicted = gradient.dot(&s) + 0.5 * s.dot(&self.b.dot(&s));
        self.pending = Some((energy, predicted, ras_cons(&s)));
        Ok(s)
    }
}

/// Ascending eigenpairs of a symmetric matrix, eigenvectors as columns.
fn eigh(matrix: &Array2<f64>) -> Result<(Array1<f64>, Array2<f64>), SaddleError> {
    let n = matrix.nrows();
    let symmetric = Array2::from_shape_fn((n, n), |(i, j)| 0.5 * (matrix[(i, j)] + matrix[(j, i)]));
    let (vals, vecs) = crate::eigensolve::faer_eigh(&symmetric)?;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| vals[i].total_cmp(&vals[j]));
    let vals = Array1::from_iter(order.iter().map(|&k| vals[k]));
    let vecs = Array2::from_shape_fn((n, n), |(i, c)| vecs[(i, order[c])]);
    Ok((vals, vecs))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(n: usize, i: usize) -> Array1<f64> {
        let mut v = Array1::zeros(n);
        v[i] = 1.0;
        v
    }

    /// Two atoms one unit apart along x, a third off the axis.
    fn triangle() -> Array1<f64> {
        Array1::from(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.4, 0.9, 0.0])
    }

    #[test]
    fn the_fitted_model_carries_the_measured_mode() {
        let x = triangle();
        let mut mode = Array1::zeros(9);
        mode[1] = 0.6;
        mode[4] = -0.8;
        let block = vec![(mode.clone(), &mode * -2.0)];
        let mut climb = ModelClimb::fitted(x.len(), &block, 0.1);
        let (lowest, curvature) = climb.lowest().unwrap();
        assert!((curvature + 2.0).abs() < 1e-10, "{curvature}");
        assert!((lowest.dot(&mode).abs() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn the_scale_keeps_the_model_under_the_measured_curvature() {
        let stretch = &unit(9, 3) - &unit(9, 0);
        let block = vec![(stretch.clone(), &stretch * 0.25)];
        let climb = ModelClimb::fitted(9, &block, 0.1);
        let other = unit(9, 8);
        assert!((other.dot(&climb.b.dot(&other)) - 0.25).abs() < 1e-12);
        assert!((stretch.dot(&climb.b.dot(&stretch)) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn the_step_climbs_the_mode_and_descends_the_rest() {
        let x = Array1::from(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let mode = unit(6, 1);
        let block = vec![(mode.clone(), &mode * -1.0)];
        let mut climb = ModelClimb::fitted(x.len(), &block, 1.0);
        let g = Array1::from(vec![0.0, 0.1, 0.0, 0.1, 0.0, 0.0]);
        let s = climb.step(&g, 0.0, 1.0).unwrap();
        assert!(s[1] > 0.0, "uphill along the mode: {s:?}");
        assert!(s[3] < 0.0, "downhill off the mode: {s:?}");
    }

    #[test]
    fn the_step_respects_the_per_atom_cap() {
        let x = Array1::from(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let mut climb = ModelClimb::fitted(x.len(), &[], 1.0);
        let g = Array1::from(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let s = climb.step(&g, 0.0, 0.05).unwrap();
        assert!(ras_cons(&s) <= 0.05 + 1e-12, "{s:?}");
    }
}
