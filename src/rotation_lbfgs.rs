//! L-BFGS direction in the dimer rotation tangent space.
//!
//! Kästner and Sherwood, J. Chem. Phys. 128, 014106 (2008),
//! doi:10.1063/1.2815812, pick the rotation plane with limited-memory
//! BFGS on the orientation. The angle inside that plane is unchanged.
//! The first direction is steepest descent. A pair whose curvature
//! `y · s` is not positive is left out, and the step is steepest
//! descent (Nocedal, Math. Comput. 35, 773 (1980),
//! doi:10.1090/S0025-5718-1980-0572855-7).

use std::collections::VecDeque;

use ndarray::Array1;

/// Pairs kept from earlier rotations of one estimate.
const MEMORY: usize = 8;

/// L-BFGS memory for one dimer rotation. Restarted with the estimate.
#[derive(Debug)]
pub(crate) struct RotationLbfgs {
    s: VecDeque<Array1<f64>>,
    y: VecDeque<Array1<f64>>,
    prev_x: Option<Array1<f64>>,
    prev_g: Option<Array1<f64>>,
}

impl RotationLbfgs {
    pub(crate) fn new() -> Self {
        Self {
            s: VecDeque::new(),
            y: VecDeque::new(),
            prev_x: None,
            prev_g: None,
        }
    }

    /// Descent direction for `gradient` at orientation `x`.
    ///
    /// `x` is the unit dimer axis and `gradient` is the curvature
    /// gradient in that tangent plane. The returned vector descends
    /// `gradient`. The caller projects it orthogonal to `x`.
    pub(crate) fn direction(&mut self, x: &Array1<f64>, gradient: &Array1<f64>) -> Array1<f64> {
        let steepest = -gradient;
        let step = if let (Some(prev_x), Some(prev_g)) = (&self.prev_x, &self.prev_g) {
            let s = x - prev_x;
            let y = gradient - prev_g;
            let ys = y.dot(&s);
            let yy = y.dot(&y);
            let scale = s.dot(&s).sqrt() * yy.sqrt();
            if ys.is_finite() && yy.is_finite() && yy > 0.0 && ys > 1e-16 * scale.max(1.0) {
                self.s.push_back(s);
                self.y.push_back(y);
                while self.s.len() > MEMORY {
                    self.s.pop_front();
                    self.y.pop_front();
                }
                let mut dir = two_loop(&self.s, &self.y, gradient);
                if !dir.iter().all(|value| value.is_finite()) {
                    steepest.clone()
                } else {
                    if dir.dot(gradient) > 0.0 {
                        dir.mapv_inplace(|value| -value);
                    }
                    dir
                }
            } else {
                steepest.clone()
            }
        } else {
            steepest
        };
        self.prev_x = Some(x.clone());
        self.prev_g = Some(gradient.clone());
        step
    }
}

/// Nocedal's two-loop recursion on `-gradient`, so the result is `-H g`.
/// The initial scaling is `(y · s) / (y · y)` from the newest pair.
fn two_loop(
    s: &VecDeque<Array1<f64>>,
    y: &VecDeque<Array1<f64>>,
    gradient: &Array1<f64>,
) -> Array1<f64> {
    let newest = s.len() - 1;
    let gamma = y[newest].dot(&s[newest]) / y[newest].dot(&y[newest]);
    let rho: Vec<f64> = s
        .iter()
        .zip(y.iter())
        .map(|(step, change)| 1.0 / change.dot(step))
        .collect();
    let mut alpha = vec![0.0; s.len()];
    let mut q = -gradient;
    for index in (0..s.len()).rev() {
        alpha[index] = rho[index] * s[index].dot(&q);
        q = &q - &y[index] * alpha[index];
    }
    let mut direction = &q * gamma;
    for index in 0..s.len() {
        let beta = rho[index] * y[index].dot(&direction);
        direction = &direction + &s[index] * (alpha[index] - beta);
    }
    direction
}
