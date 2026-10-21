//! Order-1 P-RFO climb of one band image on a dense TS-BFGS model.
//!
//! Baker, J. Comput. Chem. 7, 385 (1986), doi:10.1002/jcc.540070402,
//! for the partitioned step; the TS-BFGS update and the trust schedule
//! are Sella's (Hermes, Sarsfield, and Zádor, J. Chem. Theory Comput.
//! 18, 6974 (2022), doi:10.1021/acs.jctc.2c00395). The trust radius
//! bounds the largest per-atom move and never exceeds the band's
//! `max_move`.

use ndarray::{Array1, Array2, Axis};

use crate::error::SaddleError;
use crate::geom::{TrustSchedule, update_trust};
use crate::prfo::PartitionedRationalFunctionOptimization;
use crate::restricted::ras_cons;

pub(super) struct ModelClimb {
    b: Array2<f64>,
    floor: f64,
    spectrum: Option<(Array1<f64>, Array2<f64>)>,
    trust: f64,
    /// Energy, predicted change and per-atom size of the last step.
    pending: Option<(f64, f64, f64)>,
}

impl ModelClimb {
    /// `h0 I` on `n` coordinates, trust radius `max_move`.
    pub(super) fn new(n: usize, h0: f64, max_move: f64) -> Self {
        Self {
            b: Array2::eye(n) * h0,
            floor: 1e-2 * h0,
            spectrum: None,
            trust: max_move,
            pending: None,
        }
    }

    /// Carry `curvature` on the unit `mode` and make every other model
    /// direction a minimum: `B <- P B P + c u u'`, `P = I - u u'`, with
    /// the eigenvalues of `P B P` under `floor` reflected up to it.
    pub(super) fn set_mode(
        &mut self,
        mode: &Array1<f64>,
        curvature: f64,
    ) -> Result<(), SaddleError> {
        let u = mode.view().insert_axis(Axis(1));
        let bu = self.b.dot(mode);
        let ubu = mode.dot(&bu);
        let bu = bu.view().insert_axis(Axis(1));
        let projected = &self.b - &bu.dot(&u.t()) - &u.dot(&bu.t()) + &(u.dot(&u.t()) * ubu);
        let (vals, vecs) = eigh(&projected)?;
        let lams = Array1::from_iter(vals.iter().zip(vecs.columns()).map(|(&lam, v)| {
            if v.dot(mode).abs() > 0.5 {
                0.0
            } else if lam < self.floor {
                lam.abs().max(self.floor)
            } else {
                lam
            }
        }));
        self.b = (&vecs * &lams).dot(&vecs.t()) + u.dot(&u.t()) * curvature;
        self.spectrum = None;
        Ok(())
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

    #[test]
    fn set_mode_leaves_one_negative_direction_on_the_mode() {
        let mut climb = ModelClimb::new(6, 2.0, 0.1);
        let mut spurious = Array1::zeros(6);
        spurious[4] = 1.0;
        climb.set_mode(&spurious, -3.0).unwrap();
        let mut mode = Array1::zeros(6);
        mode[0] = 0.6;
        mode[1] = 0.8;
        climb.set_mode(&mode, -1.5).unwrap();
        let (lowest, curvature) = climb.lowest().unwrap();
        assert!((curvature + 1.5).abs() < 1e-10, "{curvature}");
        assert!((lowest.dot(&mode).abs() - 1.0).abs() < 1e-10);
        let (vals, _) = climb.spectrum().unwrap();
        assert!(vals.iter().skip(1).all(|&v| v > 0.0), "{vals:?}");
    }

    #[test]
    fn the_step_climbs_the_mode_and_descends_the_rest() {
        let mut climb = ModelClimb::new(6, 1.0, 1.0);
        let mut mode = Array1::zeros(6);
        mode[0] = 1.0;
        climb.set_mode(&mode, -1.0).unwrap();
        let g = Array1::from(vec![0.1, 0.1, 0.0, 0.0, 0.0, 0.0]);
        let s = climb.step(&g, 0.0, 1.0).unwrap();
        assert!(s[0] > 0.0, "uphill along the mode: {s:?}");
        assert!(s[1] < 0.0, "downhill off the mode: {s:?}");
    }

    #[test]
    fn the_step_respects_the_per_atom_cap() {
        let mut climb = ModelClimb::new(6, 1e-3, 1.0);
        let g = Array1::from(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let s = climb.step(&g, 0.0, 0.05).unwrap();
        assert!(ras_cons(&s) <= 0.05 + 1e-12, "{s:?}");
    }
}
