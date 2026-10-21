//! Order-1 P-RFO climb of one band image on a dense TS-BFGS model.
//!
//! Baker, J. Comput. Chem. 7, 385 (1986), doi:10.1002/jcc.540070402,
//! for the partitioned step, which maximizes along the followed mode
//! and minimizes along the rest. The mode is followed by overlap: each
//! step maximizes along the model eigenvector closest to the last
//! followed one, not along the lowest. The TS-BFGS update is Bofill's
//! (Int. J. Quantum Chem. 94, 324 (2003), doi:10.1002/qua.10709). The
//! trust radius bounds the largest per-atom move and never exceeds the
//! band's `max_move`: a ratio of actual to predicted energy change
//! outside `[1/4, 4]` halves the step just taken, and a ratio inside
//! `(3/4, 4/3)` from a step that reached the radius doubles it, the
//! rule of Nocedal and Wright (Numerical Optimization, 2nd ed.,
//! Algorithm 4.1) made symmetric under `rho -> 1/rho` because a saddle
//! step may raise or lower the energy.
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
    /// The followed mode, its model curvature and its overlap with the
    /// one followed before.
    follow: Array1<f64>,
    followed_curvature: f64,
    overlap: f64,
    /// Curvature of the last measured mode, at most `-1e-3`.
    reference: f64,
}

/// Smallest trust radius the ratio test shrinks to.
const TRUST_MIN: f64 = 1e-3;

impl ModelClimb {
    /// `mu P` at `x`, `P` the pair-stretch model and `mu` the smallest
    /// `v.Hv / v.Pv` over the pairs of `block` with positive curvature
    /// (1 when there is none), then `block` imposed and `mode`, measured
    /// at `curvature`, followed. Trust radius `max_move`.
    pub(super) fn fitted(
        x: &Array1<f64>,
        cell: Option<&Cell>,
        rigid: Option<Array2<f64>>,
        block: &[(Array1<f64>, Array1<f64>)],
        mode: &Array1<f64>,
        curvature: f64,
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
            follow: mode.clone(),
            followed_curvature: curvature,
            overlap: 1.0,
            reference: curvature.min(-1e-3),
        };
        climb.refresh(block, mode, curvature);
        climb
    }

    /// Make the model reproduce every measured pair of `block` and
    /// follow `mode`, measured at `curvature`, from here.
    pub(super) fn refresh(
        &mut self,
        block: &[(Array1<f64>, Array1<f64>)],
        mode: &Array1<f64>,
        curvature: f64,
    ) {
        impose(&mut self.b, block);
        self.spectrum = None;
        self.follow = self.free(mode);
        self.followed_curvature = curvature;
        self.overlap = 1.0;
        self.reference = curvature.min(-1e-3);
    }

    /// The followed mode has kept under `retained` of the curvature last
    /// measured, or turned under `min_overlap` from the one before.
    pub(super) fn stale(&self, retained: f64, min_overlap: f64) -> bool {
        self.followed_curvature > retained * self.reference || self.overlap < min_overlap
    }

    /// The rigid motions at the image's new position.
    pub(super) fn set_rigid(&mut self, rigid: Option<Array2<f64>>) {
        self.rigid = rigid;
        self.spectrum = None;
        self.follow = self.free(&self.follow);
    }

    /// `v` off the rigid motions, normalized; `v` itself when it is
    /// rigid or there are none, since the model then has no free
    /// direction closer to it.
    fn free(&self, v: &Array1<f64>) -> Array1<f64> {
        let Some(q) = &self.rigid else {
            return v.clone();
        };
        let p = project(v, q);
        let norm = p.dot(&p).sqrt();
        if norm > 1e-12 { p / norm } else { v.clone() }
    }

    /// TS-BFGS on the last step and the trust update from the ratio of
    /// the actual to the predicted energy change.
    pub(super) fn update(&mut self, s: &Array1<f64>, y: &Array1<f64>, energy: f64) {
        if let Some((e0, predicted, size)) = self.pending.take()
            && predicted.abs() >= 1e-14
        {
            let rho = (energy - e0) / predicted;
            if !(0.25..=4.0).contains(&rho) {
                self.trust = (0.5 * size).max(TRUST_MIN);
            } else if rho > 0.75 && rho < 4.0 / 3.0 && size > 0.9 * self.trust {
                self.trust *= 2.0;
            }
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

    /// Index of the model eigenvector closest to the followed mode.
    fn followed(&mut self) -> Result<usize, SaddleError> {
        let follow = self.follow.clone();
        let (_, vecs) = self.spectrum()?;
        let overlaps = vecs.t().dot(&follow);
        let mut k = 0;
        for i in 1..overlaps.len() {
            if overlaps[i].abs() > overlaps[k].abs() {
                k = i;
            }
        }
        Ok(k)
    }

    /// The followed eigenpair of the model, signed along the mode it
    /// follows, which it then replaces.
    pub(super) fn lowest(&mut self) -> Result<(Array1<f64>, f64), SaddleError> {
        let k = self.followed()?;
        let previous = self.follow.clone();
        let (vals, vecs) = self.spectrum()?;
        let mut v = vecs.column(k).to_owned();
        let curvature = vals[k];
        let overlap = v.dot(&previous);
        if overlap < 0.0 {
            v *= -1.0;
        }
        self.overlap = overlap.abs();
        self.followed_curvature = curvature;
        self.follow = v.clone();
        Ok((v, curvature))
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
        let k = self.followed()?;
        let (vals, vecs) = self.spectrum()?;
        let n = vals.len();
        let order: Vec<usize> = std::iter::once(k)
            .chain((0..n).filter(|&i| i != k))
            .collect();
        let vals = Array1::from_iter(order.iter().map(|&i| vals[i]));
        let vecs = Array2::from_shape_fn((n, n), |(r, c)| vecs[(r, order[c])]);
        let s = PartitionedRationalFunctionOptimization::new(1).get_s(
            &vals,
            &vecs,
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
        let mut climb = ModelClimb::fitted(&x, None, None, &block, &mode, -2.0, 0.1);
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
        let climb = ModelClimb::fitted(&x, None, None, &block, &stretch, 0.25 * along, 0.1);
        let other = unit(9, 8);
        let scaled = other.dot(&climb.b.dot(&other));
        assert!((scaled - 0.25 * p[(8, 8)]).abs() < 1e-12, "{scaled}");
        assert!((stretch.dot(&climb.b.dot(&stretch)) - 0.25 * along).abs() < 1e-12);
    }

    #[test]
    fn rigid_motions_take_no_step() {
        let x = triangle();
        let q = rigid_basis(&x, true);
        let mode = unit(9, 1);
        let mut climb = ModelClimb::fitted(&x, None, q.clone(), &[], &mode, -1.0, 1.0);
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
        let mut climb = ModelClimb::fitted(&x, None, None, &block, &mode, -1.0, 1.0);
        let g = Array1::from(vec![0.0, 0.1, 0.0, 0.1, 0.0, 0.0]);
        let s = climb.step(&g, 0.0, 1.0).unwrap();
        assert!(s[1] > 0.0, "uphill along the mode: {s:?}");
        assert!(s[3] < 0.0, "downhill off the mode: {s:?}");
    }

    #[test]
    fn the_step_respects_the_per_atom_cap() {
        let x = Array1::from(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let mut climb = ModelClimb::fitted(&x, None, None, &[], &unit(6, 0), -1.0, 1.0);
        let g = Array1::from(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let s = climb.step(&g, 0.0, 0.05).unwrap();
        assert!(ras_cons(&s) <= 0.05 + 1e-12, "{s:?}");
    }
}
