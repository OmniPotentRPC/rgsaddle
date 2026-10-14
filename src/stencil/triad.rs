//! Borrowed three-point view and the operations on it.
//!
//! `reflect`, `perp`, and `ascend` are the climbing rules.
//! Goswami, Gunde, and Jónsson, Front. Chem. (2026),
//! doi:10.3389/fchem.2026.1807063, Eq. 3, Eq. 5, and Eq. 9.
//! Curvature is the rigid-pair coupling, Eq. 2, with the forward
//! difference using the half-separation δ = ΔR/2.

use ndarray::{Array1, ArrayView1};

use crate::error::SaddleError;

/// Inner product used by a stencil. The Euclidean product is
/// `ndarray`'s dot, the same product the band and the dimer already use.
pub trait Metric {
    fn inner(&self, left: ArrayView1<f64>, right: ArrayView1<f64>) -> f64;
}

/// Flat Cartesian product.
#[derive(Clone, Copy, Debug, Default)]
pub struct Euclidean;

impl Metric for Euclidean {
    fn inner(&self, left: ArrayView1<f64>, right: ArrayView1<f64>) -> f64 {
        left.dot(&right)
    }
}

/// One geometry on the stencil. The view borrows the caller's storage.
#[derive(Clone, Copy)]
pub struct Point<'a> {
    pub x: ArrayView1<'a, f64>,
    pub e: f64,
    pub g: ArrayView1<'a, f64>,
}

/// How far a rigid or extrapolated side sits from the centre.
///
/// `Fixed` is δ = ΔR/2. `FromNoise` records the gradient noise used
/// to pick that width; the session still stores the chosen δ.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Delta {
    Fixed(f64),
    FromNoise { eps_g: Option<f64> },
}

/// Where the two side points come from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Sides {
    None,
    Neighbours,
    Rigid { half: Delta },
    Extrapolated { half: Delta },
}

/// Centre and its two neighbours. The metric is a value, not a trait object.
#[derive(Clone, Copy)]
pub struct Triad<'a, M: Metric = Euclidean> {
    pub prev: Option<Point<'a>>,
    pub centre: Point<'a>,
    pub next: Option<Point<'a>>,
    pub metric: M,
}

impl<'a, M: Metric> Triad<'a, M> {
    pub fn new(centre: Point<'a>, metric: M) -> Self {
        Self {
            prev: None,
            centre,
            next: None,
            metric,
        }
    }

    /// Curvature along a unit axis.
    ///
    /// With both side gradients, `(g+ - g-) · axis / (2 δ)`.
    /// With only the forward side, `(g+ - g0) · axis / δ`, which is
    /// the forward difference against the centre at half-separation δ.
    /// Goswami, Gunde, and Jónsson, Front. Chem. (2026),
    /// doi:10.3389/fchem.2026.1807063, Eq. 2.
    pub fn curvature(&self, axis: ArrayView1<f64>, half: f64) -> Result<f64, SaddleError> {
        if !(half.is_finite() && half > 0.0) {
            return Err(SaddleError::Invalid(
                "stencil half-separation must be positive and finite".into(),
            ));
        }
        let Some(next) = self.next else {
            return Err(SaddleError::Invalid(
                "stencil curvature needs the forward side".into(),
            ));
        };
        let action = if let Some(prev) = self.prev {
            (&next.g.to_owned() - &prev.g.to_owned()) / (2.0 * half)
        } else {
            (&next.g.to_owned() - &self.centre.g.to_owned()) / half
        };
        Ok(self.metric.inner(action.view(), axis))
    }
}

/// Householder reflection `f - 2 (f · v) v`.
///
/// The subtraction is the band climbing-image expression, so a caller
/// that already used that spelling keeps its bits.
pub fn reflect(force: ArrayView1<f64>, axis: ArrayView1<f64>) -> Array1<f64> {
    let parallel = force.dot(&axis);
    &force - (axis.to_owned() * (2.0 * parallel))
}

/// Perpendicular part `f - (f · v) v`.
pub fn perp(force: ArrayView1<f64>, axis: ArrayView1<f64>) -> Array1<f64> {
    let parallel = force.dot(&axis);
    &force - (axis.to_owned() * parallel)
}

/// Climb along the axis while the curvature is not negative: `-(f · v) v`.
pub fn ascend(force: ArrayView1<f64>, axis: ArrayView1<f64>) -> Array1<f64> {
    let parallel = force.dot(&axis);
    axis.to_owned() * (-parallel)
}

/// Curvature of a finite-difference action, `action · axis`.
pub fn curvature_along(action: ArrayView1<f64>, axis: ArrayView1<f64>) -> f64 {
    action.dot(&axis)
}

/// Tangent rule. The same two branches as [`crate::tangent::TangentKind`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TangentRule {
    Simple,
    Improved,
}

fn normalize_tangent(
    tang: Array1<f64>,
    pos_diff_next: ArrayView1<f64>,
    pos_diff_prev: ArrayView1<f64>,
) -> Result<Array1<f64>, SaddleError> {
    for mut v in [tang, pos_diff_next.to_owned(), pos_diff_prev.to_owned()] {
        let norm = v.dot(&v).sqrt();
        if norm > 1e-10 && norm.is_finite() {
            v /= norm;
            return Ok(v);
        }
    }
    Err(SaddleError::Invalid(
        "band image coincides with both neighbours; no tangent".into(),
    ))
}

/// Unit tangent from the two minimum-imaged segments and the three energies.
pub fn unit_tangent(
    kind: TangentRule,
    pos_diff_next: ArrayView1<f64>,
    pos_diff_prev: ArrayView1<f64>,
    energy: f64,
    energy_prev: f64,
    energy_next: f64,
) -> Result<Array1<f64>, SaddleError> {
    let tang = match kind {
        TangentRule::Simple => pos_diff_next.to_owned(),
        TangentRule::Improved => {
            if energy_next > energy && energy > energy_prev {
                pos_diff_next.to_owned()
            } else if energy > energy_next && energy_prev > energy {
                pos_diff_prev.to_owned()
            } else {
                let energy_diff_prev = energy_prev - energy;
                let energy_diff_next = energy_next - energy;
                let min_diff = energy_diff_prev.abs().min(energy_diff_next.abs());
                let max_diff = energy_diff_prev.abs().max(energy_diff_next.abs());
                if energy_diff_prev > energy_diff_next {
                    &pos_diff_next * min_diff + &pos_diff_prev * max_diff
                } else {
                    &pos_diff_next * max_diff + &pos_diff_prev * min_diff
                }
            }
        }
    };
    normalize_tangent(tang, pos_diff_next, pos_diff_prev)
}

/// Parallel and full spring, the two pieces a projection consumes.
pub struct CouplingForces {
    pub parallel: Array1<f64>,
    pub full: Array1<f64>,
}

/// Coupling along the tangent. `Rigid` contributes no spring force.
pub enum Coupling<'a> {
    Uniform {
        k: f64,
    },
    Weighted {
        k_next: f64,
        k_prev: f64,
    },
    Onsager {
        k: f64,
        l_next: ArrayView1<'a, f64>,
        l_here: ArrayView1<'a, f64>,
    },
    Rigid,
}

/// Spring forces at one interior image. The arithmetic is the eOn
/// `NEBSpringForce` spelling.
pub fn coupling_forces(
    coupling: &Coupling<'_>,
    tangent: ArrayView1<f64>,
    dist_next: f64,
    dist_prev: f64,
    pos_diff_next: ArrayView1<f64>,
    pos_diff_prev: ArrayView1<f64>,
) -> CouplingForces {
    match coupling {
        Coupling::Uniform { k } => CouplingForces {
            parallel: tangent.to_owned() * (k * (dist_next - dist_prev)),
            full: (&pos_diff_next - &pos_diff_prev) * *k,
        },
        Coupling::Weighted { k_next, k_prev } => CouplingForces {
            parallel: tangent.to_owned() * (k_next * dist_next - k_prev * dist_prev),
            full: &pos_diff_next * *k_next - &pos_diff_prev * *k_prev,
        },
        Coupling::Onsager { k, l_next, l_here } => {
            let l_next = l_next.to_owned();
            let l_here = l_here.to_owned();
            let diff = &pos_diff_next - &pos_diff_prev + &l_next - &l_here;
            let f_om = diff * *k;
            let par = f_om.dot(&tangent);
            CouplingForces {
                parallel: tangent.to_owned() * par,
                full: Array1::zeros(tangent.len()),
            }
        }
        Coupling::Rigid => CouplingForces {
            parallel: Array1::zeros(tangent.len()),
            full: Array1::zeros(tangent.len()),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn le(values: &Array1<f64>) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn reflect_perp_and_ascend_match_the_historical_spelling() {
        let force = array![1.0, -4.0, 0.5, -0.0, 1e-20];
        let axis = array![1.0, 0.0, 0.0, 0.0, 0.0];
        let parallel = force.dot(&axis);
        let historical_reflect = &force - &(axis.to_owned() * (2.0 * parallel));
        let historical_perp = &force - (axis.to_owned() * parallel);
        let historical_ascend = axis.to_owned() * (-parallel);
        assert_eq!(
            le(&reflect(force.view(), axis.view())),
            le(&historical_reflect)
        );
        assert_eq!(le(&perp(force.view(), axis.view())), le(&historical_perp));
        assert_eq!(
            le(&ascend(force.view(), axis.view())),
            le(&historical_ascend)
        );
    }

    #[test]
    fn forward_curvature_is_the_half_separation_difference() {
        let x = array![0.0, 0.0];
        let g0 = array![1.0, -2.0];
        let gp = array![3.0, 0.0];
        let centre = Point {
            x: x.view(),
            e: 0.5,
            g: g0.view(),
        };
        let next_x = array![0.1, 0.0];
        let next = Point {
            x: next_x.view(),
            e: 0.6,
            g: gp.view(),
        };
        let triad = Triad {
            prev: None,
            centre,
            next: Some(next),
            metric: Euclidean,
        };
        let axis = array![1.0, 0.0];
        let half = 1e-3;
        let got = triad.curvature(axis.view(), half).unwrap();
        let action = (&gp - &g0) / half;
        assert_eq!(got.to_le_bytes(), action.dot(&axis).to_le_bytes());
    }
}
