//! Riemannian trust region with truncated conjugate gradients (RTR-tCG)
//! on the band.
//!
//! Absil, Baker, Gallivan, "Trust-region methods on Riemannian manifolds",
//! Found. Comput. Math. 7, 303 (2007). The band with `n` movable images is
//! a point `X` of the product manifold whose tangent space at `X` is the
//! direct sum of the chord-perpendicular subspaces `range(I - tau_i tau_i^T)`
//! (the nudging projection of NEB), so the Riemannian gradient is the
//! assembled NEB force with the sign flipped and a Hessian-vector product is
//! the projected directional derivative of that force. At `X` the model
//!
//! `m(eta) = f(X) + <grad f(X), eta> + 1/2 <Hess f(X)[eta], eta>`,
//! for `eta` in the tangent space `T_X M`,
//!
//! is minimised inside the ball `||eta|| <= Delta` by the Steihaug-Toint
//! truncated CG iteration run entirely in the tangent space; the caller
//! retracts the step (the images move, the endpoints stay), measures the
//! decrease and feeds the ratio to [`RtrRadius::update`]. The NEB force is
//! not the gradient of a potential once the springs and the climbing
//! reflection are in, so the "actual decrease" is the line integral of the
//! force along the step, `-1/2 <g(X) + g(X + eta), eta>` (trapezoid rule),
//! which is the energy decrease whenever the force is conservative.
//!
//! The constants in the radius update are the theorem's: accept when
//! `rho > rho' = 0.1`, shrink by 1/4 below `rho = 1/4`, grow by 2 above
//! `rho = 3/4` at the boundary, never past `Delta_bar`. Nothing here is a
//! per-fixture number.
//!
//! The shared subproblem solver gives the exact quadratic decrease in a
//! Euclidean tangent space, as for `f(x) = x^2` at `x = 1`:
//! ```
//! use ndarray::array;
//! use rgsaddle::truncated_cg;
//! let result = truncated_cg(
//!     |v| v.clone(), array![2.0].view(), |v| v * 2.0, 2.0, 1.0, 0.1, 10,
//! );
//! assert!((result.eta[0] + 1.0).abs() < 1e-12);
//! assert!((result.model_decrease - 1.0).abs() < 1e-12);
//! ```

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};

use crate::band::{BandConfig, BandSurface};
use crate::error::SaddleError;
use crate::mic::wrap_difference;
use crate::projection::{ProjectionKind, climbing_image_force, dneb_component, force_perp};
use crate::tangent::compute_tangent;

pub use rgmin::rtr::{RtrRadius, TcgResult, TcgStop, truncated_cg_projected as truncated_cg};

use rgmin::vecops::{dot, nrm2};

/// The assembled band at one point: per-image energies, chord tangents of
/// the interior images and the NEB force on the interior, flattened.
#[derive(Clone, Debug)]
pub struct BandForces {
    pub energies: Array1<f64>,
    /// Unit tangents of the interior images (rows `1..n-1`), `(n-2) x dof`.
    pub tangents: Array2<f64>,
    /// NEB force on the interior, `(n-2) * dof`.
    pub force: Array1<f64>,
    pub max_force: f64,
}

/// Assemble the NEB force of `positions` with the climbing image `ci` (an
/// interior index, or none). Same projection and spring rules as
/// [`crate::BandSession`], without the climbing state machine: the caller
/// decides which image climbs.
pub fn band_forces<S: BandSurface + ?Sized>(
    config: &BandConfig,
    surface: &S,
    positions: ArrayView2<f64>,
    ci: Option<usize>,
) -> Result<BandForces, SaddleError> {
    let n_images = positions.nrows();
    let dof = positions.ncols();
    if n_images < 3 {
        return Err(SaddleError::Solver(
            "a band needs at least one interior image".into(),
        ));
    }
    let mut energies = Array1::zeros(n_images);
    let mut gradients = Array2::zeros((n_images, dof));
    surface.eval(positions, &mut energies, &mut gradients)?;
    if !energies.iter().all(|e| e.is_finite()) {
        return Err(SaddleError::NonFinite("band energies"));
    }
    let mut tangents = Array2::zeros((n_images - 2, dof));
    let mut force = Array1::zeros((n_images - 2) * dof);
    for i in 1..n_images - 1 {
        let mut pos_diff_next = (&positions.row(i + 1) - &positions.row(i)).to_owned();
        let mut pos_diff_prev = (&positions.row(i) - &positions.row(i - 1)).to_owned();
        if let Some(cell) = &config.cell {
            wrap_difference(cell, &mut pos_diff_next);
            wrap_difference(cell, &mut pos_diff_prev);
        }
        let dist_next = nrm2(pos_diff_next.view());
        let dist_prev = nrm2(pos_diff_prev.view());
        let tangent = compute_tangent(
            config.tangent,
            pos_diff_next.view(),
            pos_diff_prev.view(),
            energies[i],
            energies[i - 1],
            energies[i + 1],
        );
        let true_force = -&gradients.row(i);
        let spring = config.spring.compute(
            i,
            tangent.view(),
            dist_next,
            dist_prev,
            pos_diff_next.view(),
            pos_diff_prev.view(),
            positions.row(i),
            positions.row(i - 1),
            positions.row(i + 1),
        );
        let image_force = if ci == Some(i) {
            let dneb = if config.projection == ProjectionKind::DoublyNudged {
                let fp = force_perp(true_force.view(), tangent.view());
                dneb_component(spring.full.view(), tangent.view(), fp.view())
            } else {
                Array1::zeros(dof)
            };
            climbing_image_force(true_force.view(), tangent.view(), dneb.view())
        } else {
            config
                .projection
                .project(true_force.view(), tangent.view(), &spring)
        };
        force
            .slice_mut(s![(i - 1) * dof..i * dof])
            .assign(&image_force);
        tangents.row_mut(i - 1).assign(&tangent);
    }
    let max_force = config.force_gate.value(force.view());
    Ok(BandForces {
        energies,
        tangents,
        force,
        max_force,
    })
}

/// Place the interior images at equal chord length along the current
/// polyline; with `anchor` an interior image, each side of it is spaced on
/// its own and the anchor stays. Endpoints never move.
pub fn reparametrize_equal_arc(positions: &mut Array2<f64>, anchor: Option<usize>) {
    let n = positions.nrows();
    if n < 3 {
        return;
    }
    let orig = positions.clone();
    let mut s = vec![0.0; n];
    for i in 1..n {
        let d = &orig.row(i) - &orig.row(i - 1);
        s[i] = s[i - 1] + d.dot(&d).sqrt();
    }
    let place = |positions: &mut Array2<f64>, j: usize, target: f64| {
        let mut k = 0;
        while k < n - 2 && s[k + 1] < target {
            k += 1;
        }
        let denom = s[k + 1] - s[k];
        let alpha = if denom > f64::MIN_POSITIVE {
            (target - s[k]) / denom
        } else {
            0.0
        };
        let p = &orig.row(k) + &((&orig.row(k + 1) - &orig.row(k)) * alpha);
        positions.row_mut(j).assign(&p);
    };
    match anchor {
        Some(a) if a > 0 && a < n - 1 => {
            let left = s[a] / a as f64;
            for j in 1..a {
                place(positions, j, j as f64 * left);
            }
            let right = (s[n - 1] - s[a]) / (n - 1 - a) as f64;
            for j in a + 1..n - 1 {
                place(positions, j, s[a] + (j - a) as f64 * right);
            }
        }
        _ => {
            let h = s[n - 1] / (n - 1) as f64;
            for j in 1..n - 1 {
                place(positions, j, j as f64 * h);
            }
        }
    }
}

/// Settings of the band trust-region step.
#[derive(Clone, Copy, Debug)]
pub struct RtrConfig {
    /// `Delta_bar`: largest trust radius, in the band's flattened metric.
    pub radius_max: f64,
    /// Inner stopping rule exponent (reference 1).
    pub theta: f64,
    /// Inner stopping rule factor (reference 0.1).
    pub kappa: f64,
    /// Truncated CG budget per step.
    pub max_cg: usize,
    /// Central-difference step of the force along a unit tangent direction.
    pub fd_step: f64,
}

impl Default for RtrConfig {
    fn default() -> Self {
        Self {
            radius_max: 1.0,
            theta: 1.0,
            kappa: 0.1,
            max_cg: 50,
            fd_step: 1e-4,
        }
    }
}

/// One accepted or rejected band step.
#[derive(Clone, Debug)]
pub struct RtrReport {
    pub accepted: bool,
    pub rho: f64,
    pub radius: f64,
    pub stop: TcgStop,
    pub cg_iterations: usize,
    pub model_decrease: f64,
    pub actual_decrease: f64,
    /// Max NEB-force component after the step (before it, when rejected).
    pub max_force: f64,
    pub ci: Option<usize>,
}

/// Band trust-region stepper. Endpoints (rows 0 and `n_images - 1`) never
/// move; the climbing image is the highest interior image once
/// `climb` is set, the plain band otherwise.
#[derive(Clone, Debug)]
pub struct BandRtr {
    pub config: RtrConfig,
    pub radius: RtrRadius,
    pub climb: bool,
    pub iteration: usize,
}

impl BandRtr {
    pub fn new(config: RtrConfig, climb: bool) -> Self {
        Self {
            config,
            radius: RtrRadius::new(config.radius_max),
            climb,
            iteration: 0,
        }
    }

    fn climbing_index(&self, energies: &Array1<f64>) -> Option<usize> {
        if !self.climb {
            return None;
        }
        let n = energies.len();
        let mut best = 1;
        for i in 1..n - 1 {
            if energies[i] > energies[best] {
                best = i;
            }
        }
        Some(best)
    }

    /// One RTR-tCG step on `positions` (modified in place when accepted).
    /// Two surface evaluations per Hessian-vector product, one for the trial
    /// point.
    pub fn step<S: BandSurface + ?Sized>(
        &mut self,
        band: &BandConfig,
        surface: &S,
        positions: &mut Array2<f64>,
    ) -> Result<RtrReport, SaddleError> {
        let n_images = positions.nrows();
        let dof = positions.ncols();
        let interior = (n_images - 2) * dof;
        let here = band_forces(band, surface, positions.view(), None)?;
        let ci = self.climbing_index(&here.energies);
        let here = if ci.is_some() {
            band_forces(band, surface, positions.view(), ci)?
        } else {
            here
        };
        let grad = -&here.force;
        let tangents = here.tangents.clone();
        let project = move |v: &Array1<f64>| -> Array1<f64> {
            let mut out = v.clone();
            for i in 0..n_images - 2 {
                let tau = tangents.row(i);
                let mut seg = out.slice_mut(s![i * dof..(i + 1) * dof]);
                let along = seg.dot(&tau);
                seg.scaled_add(-along, &tau);
            }
            out
        };
        let h = self.config.fd_step;
        let base = positions.clone();
        let hvp = |v: &Array1<f64>| -> Array1<f64> {
            let vn = nrm2(v.view());
            if !(vn > 0.0) {
                return Array1::zeros(interior);
            }
            let scatter = |sign: f64| {
                let mut x = base.clone();
                for i in 1..n_images - 1 {
                    let seg = v.slice(s![(i - 1) * dof..i * dof]);
                    let mut row = x.row_mut(i);
                    row.scaled_add(sign * h / vn, &seg);
                }
                x
            };
            let plus = band_forces(band, surface, scatter(1.0).view(), ci);
            let minus = band_forces(band, surface, scatter(-1.0).view(), ci);
            match (plus, minus) {
                (Ok(p), Ok(m)) => (&m.force - &p.force) * (vn / (2.0 * h)),
                _ => Array1::from_elem(interior, f64::INFINITY),
            }
        };
        let tcg = truncated_cg(
            project,
            grad.view(),
            hvp,
            self.radius.radius,
            self.config.theta,
            self.config.kappa,
            self.config.max_cg,
        );
        self.iteration += 1;
        if tcg.stop == TcgStop::ZeroGradient {
            return Ok(RtrReport {
                accepted: false,
                rho: f64::NAN,
                radius: self.radius.radius,
                stop: tcg.stop,
                cg_iterations: 0,
                model_decrease: 0.0,
                actual_decrease: 0.0,
                max_force: here.max_force,
                ci,
            });
        }
        let mut trial = positions.clone();
        for i in 1..n_images - 1 {
            let seg = tcg.eta.slice(s![(i - 1) * dof..i * dof]);
            let mut row = trial.row_mut(i);
            row.scaled_add(1.0, &seg);
        }
        let there = band_forces(band, surface, trial.view(), ci)?;
        let grad_there = -&there.force;
        let actual_decrease = -0.5 * dot((&grad + &grad_there).view(), tcg.eta.view());
        let eta_norm = nrm2(tcg.eta.view());
        let rho = if tcg.model_decrease > 0.0 {
            actual_decrease / tcg.model_decrease
        } else {
            f64::NEG_INFINITY
        };
        let accepted = self
            .radius
            .update(actual_decrease, tcg.model_decrease, eta_norm);
        // Retraction: the tangent step moves the images across the path, the
        // spacing along it is restored by re-parametrising the polyline at
        // equal chord length with the climbing image held where it landed.
        // The spring force therefore never enters the model; it is the
        // retraction's job, as in the string method.
        let there = if accepted {
            reparametrize_equal_arc(&mut trial, ci);
            positions.assign(&trial);
            band_forces(band, surface, positions.view(), ci)?
        } else {
            there
        };
        Ok(RtrReport {
            accepted,
            rho,
            radius: self.radius.radius,
            stop: tcg.stop,
            cg_iterations: tcg.iterations,
            model_decrease: tcg.model_decrease,
            actual_decrease,
            max_force: if accepted {
                there.max_force
            } else {
                here.max_force
            },
            ci,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};

    fn quad_hvp(h: &Array2<f64>) -> impl Fn(&Array1<f64>) -> Array1<f64> + '_ {
        move |v| h.dot(v)
    }

    fn identity(v: &Array1<f64>) -> Array1<f64> {
        v.clone()
    }

    #[test]
    fn euclidean_tcg_reaches_the_newton_point_inside_the_ball() {
        let h = Array2::from_diag(&array![1.0, 2.0, 3.0]);
        let g = array![1.0, -2.0, 0.5];
        let r = truncated_cg(identity, g.view(), quad_hvp(&h), 100.0, 1.0, 0.1, 50);
        let newton = array![-1.0, 1.0, -0.5 / 3.0];
        assert!(
            (r.eta.clone() - newton.clone()).mapv(f64::abs).sum() < 1e-10,
            "{:?}",
            r.eta
        );
        let expected = -(g.dot(&newton) + 0.5 * newton.dot(&h.dot(&newton)));
        assert!((r.model_decrease - expected).abs() < 1e-10);
        assert_eq!(r.stop, TcgStop::Residual);
    }

    #[test]
    fn negative_curvature_runs_to_the_boundary() {
        let h = Array2::from_diag(&array![-1.0, 2.0]);
        let g = array![1.0, 0.0];
        let r = truncated_cg(identity, g.view(), quad_hvp(&h), 0.5, 1.0, 0.1, 50);
        assert_eq!(r.stop, TcgStop::NegativeCurvature);
        assert!((nrm2(r.eta.view()) - 0.5).abs() < 1e-12);
        assert!(r.eta[0] < 0.0);
    }

    #[test]
    fn projected_tcg_stays_in_the_tangent_space() {
        // Projector onto the plane orthogonal to (1,1,1)/sqrt 3: the step
        // never leaves it and the model decrease is that of the projected
        // problem.
        let n = array![1.0, 1.0, 1.0] / 3f64.sqrt();
        let project = move |v: &Array1<f64>| v - &(&n * v.dot(&n));
        let h = Array2::from_diag(&array![1.0, 2.0, 3.0]);
        let g = array![1.0, -2.0, 0.5];
        let r = truncated_cg(project, g.view(), quad_hvp(&h), 100.0, 1.0, 0.1, 50);
        assert!(r.eta.sum().abs() < 1e-10, "{:?}", r.eta);
        assert!(r.model_decrease > 0.0);
    }

    #[test]
    fn radius_update_follows_the_reference_rule() {
        let mut r = RtrRadius::new(8.0);
        assert_eq!(r.radius, 1.0);
        assert!(!r.update(0.1, 1.0, 0.5)); // rho = 0.1: not accepted, shrink
        assert!((r.radius - 0.25).abs() < 1e-15);
        assert!(r.update(0.9, 1.0, 0.25)); // rho = 0.9 at the boundary: grow
        assert!((r.radius - 0.5).abs() < 1e-15);
        assert!(r.update(0.5, 1.0, 0.1)); // rho = 0.5 interior: unchanged
        assert!((r.radius - 0.5).abs() < 1e-15);
        assert!(!r.update(1.0, 0.0, 0.1)); // no predicted decrease: reject
    }
}
