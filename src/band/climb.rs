//! Order-1 P-RFO climb of one band image on a dense TS-BFGS model.
//!
//! Baker, J. Comput. Chem. 7, 385 (1986), doi:10.1002/jcc.540070402,
//! for the partitioned step; the TS-BFGS update and the trust schedule
//! are Sella's (Hermes, Sarsfield, and Zádor, J. Chem. Theory Comput.
//! 18, 6974 (2022), doi:10.1021/acs.jctc.2c00395). The trust radius
//! bounds the largest per-atom move and never exceeds the band's
//! `max_move`.
//!
//! The model starts from the pair-stretch Hessian of the image
//! ([`super::pair_model`]), scaled by the smallest ratio of measured to
//! model curvature over the positive directions of the Lanczos block,
//! so it is not stiffer than the surface along any sampled direction,
//! and the block is then imposed exactly. With `rigid` set the model,
//! the steps and the secant pairs live in the complement of the rigid
//! motions, which take no step.

use ndarray::{Array1, Array2};

use super::Cell;
use super::pair_model::{impose, pair_hessian, project, project_both};
use crate::error::SaddleError;
use crate::geom::{TrustSchedule, update_trust};
use crate::prfo::PartitionedRationalFunctionOptimization;
use crate::restricted::ras_cons;

pub(super) struct ModelClimb {
    b: Array2<f64>,
    /// Orthonormal rigid motions removed from the model, or `None`.
    rigid: Option<Array2<f64>>,
    spectrum: Option<(Array1<f64>, Array2<f64>)>,
    trust: f64,
    /// Energy, predicted change and per-atom size of the last step.
    pending: Option<(f64, f64, f64)>,
}

impl ModelClimb {
    /// `mu P` at `x`, `P` the pair-stretch model and `mu` the smallest
    /// `v.Hv / v.Pv` over the pairs of `block` with positive curvature
    /// (1 when there is none), then `block` imposed. Trust radius
    /// `max_move`.
    pub(super) fn fitted(
        x: &Array1<f64>,
        cell: Option<&Cell>,
        rigid: Option<Array2<f64>>,
        block: &[(Array1<f64>, Array1<f64>)],
        max_move: f64,
    ) -> Self {
        let mut p = pair_hessian(x, cell);
        if let Some(q) = &rigid {
            p = project_both(&p, q);
        }
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
            rigid,
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

    /// The rigid motions at the image's new position.
    pub(super) fn set_rigid(&mut self, rigid: Option<Array2<f64>>) {
        self.rigid = rigid;
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
        match &self.rigid {
            Some(q) => rgmin::ts_bfgs_update(&mut self.b, &project(s, q), &project(y, q)),
            None => rgmin::ts_bfgs_update(&mut self.b, s, y),
        }
        self.spectrum = None;
    }

    /// The model the step reads: the rigid motions carry a curvature a
    /// thousand times the model's largest entry, so they take no step.
    fn model(&self) -> Array2<f64> {
        match &self.rigid {
            Some(q) => {
                let mut m = project_both(&self.b, q);
                let scale = m.iter().fold(1.0_f64, |a, v| a.max(v.abs()));
                m += &(q.dot(&q.t()) * (1e3 * scale));
                m
            }
            None => self.b.clone(),
        }
    }

    fn spectrum(&mut self) -> Result<&(Array1<f64>, Array2<f64>), SaddleError> {
        if self.spectrum.is_none() {
            self.spectrum = Some(eigh(&self.model())?);
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
        let g = match &self.rigid {
            Some(q) => project(gradient, q),
            None => gradient.clone(),
        };
        let (vals, vecs) = self.spectrum()?;
        let s = PartitionedRationalFunctionOptimization::new(1).get_s(
            vals,
            vecs,
            &g,
            PartitionedRationalFunctionOptimization::ALPHA0,
        );
        let s = rgmin::ras_clip(&s, trust);
        let predicted = g.dot(&s) + 0.5 * s.dot(&self.b.dot(&s));
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
    use super::super::pair_model::rigid_basis;
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
        let mut climb = ModelClimb::fitted(&x, None, None, &block, 0.1);
        let (lowest, curvature) = climb.lowest().unwrap();
        assert!((curvature + 2.0).abs() < 1e-10, "{curvature}");
        assert!((lowest.dot(&mode).abs() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn the_scale_keeps_the_model_under_the_measured_curvature() {
        let x = triangle();
        let p = pair_hessian(&x, None);
        let stretch = &unit(9, 3) - &unit(9, 0);
        let along = stretch.dot(&p.dot(&stretch));
        let block = vec![(stretch.clone(), p.dot(&stretch) * 0.25)];
        let climb = ModelClimb::fitted(&x, None, None, &block, 0.1);
        let other = unit(9, 8);
        let scaled = other.dot(&climb.b.dot(&other));
        assert!((scaled - 0.25 * p[(8, 8)]).abs() < 1e-12, "{scaled}");
        assert!((stretch.dot(&climb.b.dot(&stretch)) - 0.25 * along).abs() < 1e-12);
    }

    #[test]
    fn rigid_motions_take_no_step() {
        let x = triangle();
        let q = rigid_basis(&x, true);
        let mut climb = ModelClimb::fitted(&x, None, q.clone(), &[], 1.0);
        let mut g = Array1::from_elem(9, 0.1);
        g[2] = -0.3;
        let s = climb.step(&g, 0.0, 1.0).unwrap();
        let q = q.unwrap();
        assert!(q.t().dot(&s).iter().all(|v| v.abs() < 1e-8), "{s}");
    }

    #[test]
    fn the_step_climbs_the_mode_and_descends_the_rest() {
        let x = Array1::from(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let mode = unit(6, 1);
        let block = vec![(mode.clone(), &mode * -1.0)];
        let mut climb = ModelClimb::fitted(&x, None, None, &block, 1.0);
        let g = Array1::from(vec![0.0, 0.1, 0.0, 0.1, 0.0, 0.0]);
        let s = climb.step(&g, 0.0, 1.0).unwrap();
        assert!(s[1] > 0.0, "uphill along the mode: {s:?}");
        assert!(s[3] < 0.0, "downhill off the mode: {s:?}");
    }

    #[test]
    fn the_step_respects_the_per_atom_cap() {
        let x = Array1::from(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let mut climb = ModelClimb::fitted(&x, None, None, &[], 1.0);
        let g = Array1::from(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let s = climb.step(&g, 0.0, 0.05).unwrap();
        assert!(ras_cons(&s) <= 0.05 + 1e-12, "{s:?}");
    }
}
