//! Borrowed three-point view and the operations on it.
//!
//! `reflect`, `perp`, and `ascend` are the climbing rules.
//! Goswami, Gunde, and Jónsson, Front. Chem. (2026),
//! doi:10.3389/fchem.2026.1807063, Eq. 3, Eq. 5, and Eq. 9.
//! Curvature is the rigid-pair coupling, Eq. 2, with the forward
//! difference using the half-separation δ = ΔR/2.

use ndarray::{Array1, ArrayView1, ArrayViewMut1, Zip};

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
    let mut out = Array1::zeros(force.len());
    reflect_into(force, axis, out.view_mut());
    out
}

/// [`reflect`] written into `out`, element for element the same bits.
pub fn reflect_into(force: ArrayView1<f64>, axis: ArrayView1<f64>, out: ArrayViewMut1<f64>) {
    let twice = 2.0 * force.dot(&axis);
    Zip::from(out)
        .and(force)
        .and(axis)
        .for_each(|o, &f, &a| *o = f - a * twice);
}

/// Perpendicular part `f - (f · v) v`.
pub fn perp(force: ArrayView1<f64>, axis: ArrayView1<f64>) -> Array1<f64> {
    let mut out = Array1::zeros(force.len());
    perp_into(force, axis, out.view_mut());
    out
}

/// [`perp`] written into `out`.
pub fn perp_into(force: ArrayView1<f64>, axis: ArrayView1<f64>, out: ArrayViewMut1<f64>) {
    let parallel = force.dot(&axis);
    Zip::from(out)
        .and(force)
        .and(axis)
        .for_each(|o, &f, &a| *o = f - a * parallel);
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

/// Scale `v` to unit length when its norm is above `1e-10` and finite.
fn unit_in_place(mut v: ArrayViewMut1<f64>) -> bool {
    let norm = v.dot(&v).sqrt();
    if norm > 1e-10 && norm.is_finite() {
        v /= norm;
        return true;
    }
    false
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
    let mut out = Array1::zeros(pos_diff_next.len());
    unit_tangent_into(
        kind,
        pos_diff_next,
        pos_diff_prev,
        energy,
        energy_prev,
        energy_next,
        out.view_mut(),
    )?;
    Ok(out)
}

/// [`unit_tangent`] written into `out`. When the weighted tangent
/// vanishes, the forward and then the backward segment take its place.
pub fn unit_tangent_into(
    kind: TangentRule,
    pos_diff_next: ArrayView1<f64>,
    pos_diff_prev: ArrayView1<f64>,
    energy: f64,
    energy_prev: f64,
    energy_next: f64,
    mut out: ArrayViewMut1<f64>,
) -> Result<(), SaddleError> {
    enum Pick {
        Next,
        Prev,
        Mix { next: f64, prev: f64 },
    }
    let pick = match kind {
        TangentRule::Simple => Pick::Next,
        TangentRule::Improved => {
            if energy_next > energy && energy > energy_prev {
                Pick::Next
            } else if energy > energy_next && energy_prev > energy {
                Pick::Prev
            } else {
                let energy_diff_prev = energy_prev - energy;
                let energy_diff_next = energy_next - energy;
                let min_diff = energy_diff_prev.abs().min(energy_diff_next.abs());
                let max_diff = energy_diff_prev.abs().max(energy_diff_next.abs());
                if energy_diff_prev > energy_diff_next {
                    Pick::Mix {
                        next: min_diff,
                        prev: max_diff,
                    }
                } else {
                    Pick::Mix {
                        next: max_diff,
                        prev: min_diff,
                    }
                }
            }
        }
    };
    match pick {
        Pick::Next => out.assign(&pos_diff_next),
        Pick::Prev => out.assign(&pos_diff_prev),
        Pick::Mix { next, prev } => Zip::from(out.view_mut())
            .and(pos_diff_next)
            .and(pos_diff_prev)
            .for_each(|t, &n, &p| *t = n * next + p * prev),
    }
    if unit_in_place(out.view_mut()) {
        return Ok(());
    }
    for side in [pos_diff_next, pos_diff_prev] {
        out.assign(&side);
        if unit_in_place(out.view_mut()) {
            return Ok(());
        }
    }
    Err(SaddleError::Invalid(
        "band image coincides with both neighbours; no tangent".into(),
    ))
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
    let mut parallel = Array1::zeros(tangent.len());
    let mut full = Array1::zeros(tangent.len());
    coupling_forces_into(
        coupling,
        tangent,
        dist_next,
        dist_prev,
        pos_diff_next,
        pos_diff_prev,
        parallel.view_mut(),
        full.view_mut(),
    );
    CouplingForces { parallel, full }
}

/// [`coupling_forces`] written into `parallel` and `full`. The
/// Onsager-Machlup arm uses `full` as scratch before zeroing it.
#[allow(clippy::too_many_arguments)]
pub fn coupling_forces_into(
    coupling: &Coupling<'_>,
    tangent: ArrayView1<f64>,
    dist_next: f64,
    dist_prev: f64,
    pos_diff_next: ArrayView1<f64>,
    pos_diff_prev: ArrayView1<f64>,
    mut parallel: ArrayViewMut1<f64>,
    mut full: ArrayViewMut1<f64>,
) {
    match coupling {
        Coupling::Uniform { k } => {
            let along = k * (dist_next - dist_prev);
            Zip::from(parallel)
                .and(tangent)
                .for_each(|s, &t| *s = t * along);
            Zip::from(full)
                .and(pos_diff_next)
                .and(pos_diff_prev)
                .for_each(|s, &n, &p| *s = (n - p) * *k);
        }
        Coupling::Weighted { k_next, k_prev } => {
            let along = k_next * dist_next - k_prev * dist_prev;
            Zip::from(parallel)
                .and(tangent)
                .for_each(|s, &t| *s = t * along);
            Zip::from(full)
                .and(pos_diff_next)
                .and(pos_diff_prev)
                .for_each(|s, &n, &p| *s = n * *k_next - p * *k_prev);
        }
        Coupling::Onsager { k, l_next, l_here } => {
            Zip::from(full.view_mut())
                .and(pos_diff_next)
                .and(pos_diff_prev)
                .and(l_next)
                .and(l_here)
                .for_each(|s, &n, &p, &ln, &lh| *s = (n - p + ln - lh) * *k);
            let par = full.dot(&tangent);
            Zip::from(parallel)
                .and(tangent)
                .for_each(|s, &t| *s = t * par);
            full.fill(0.0);
        }
        Coupling::Rigid => {
            parallel.fill(0.0);
            full.fill(0.0);
        }
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
