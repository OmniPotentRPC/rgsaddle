//! Projected trust-region relaxation of a band.
//!
//! The truncated conjugate-gradient subproblem follows Absil, Baker and
//! Gallivan, Found. Comput. Math. 7, 303 (2007). Ordinary images move
//! perpendicular to their chord tangents; the climbing image retains
//! its reaction-path component.
//!
//! A trial combines this increment with equal-arc reparameterization,
//! on the polyline or on the not-a-knot spline ([`RtrConfig::reparametrization`]).
//! Its complete displacement is capped by the trust radius before evaluation.
//! The reported model decrease uses that displacement and a central
//! finite difference of the assembled force. The measured decrease is
//! its trapezoidal force integral. NEB forces are nonconservative, so this
//! band update is a heuristic and carries no energy-descent guarantee.
//!
//! This quadratic has its Newton point inside the trust sphere:
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

use crate::band::{BandConfig, BandSurface, zero_fixed};
use crate::error::SaddleError;
use crate::force::ForceGate;
use crate::projection::{ProjectionKind, climbing_image_force, dneb_component, force_perp};
use crate::reparam::Reparametrization;
use crate::tangent::compute_tangent;

pub use rgmin::rtr::{RtrRadius, TcgResult, TcgStop, truncated_cg_projected as truncated_cg};

use rgmin::vecops::{dot, nrm2};
use rgmin::{RigidMode, RigidRequest, RigidWorkspace};

/// The assembled band at one point: per-image energies, chord tangents of
/// the interior images and the NEB force on the interior, flattened.
#[derive(Clone, Debug)]
pub struct BandForces {
    /// Potential energy of every image, endpoints included, in the host's energy unit.
    pub energies: Array1<f64>,
    /// Unit tangents of the interior images (rows `1..n-1`), `(n-2) x dof`.
    pub tangents: Array2<f64>,
    /// NEB force on the interior, `(n-2) * dof`.
    pub force: Array1<f64>,
    /// Largest absolute force component, in the host's force units.
    pub max_force: f64,
}

/// Assemble the NEB force of `positions` with the climbing image `ci` (an
/// interior index, or none). Same projection and spring rules as
/// [`crate::BandSession`], without the climbing state machine: the caller
/// decides which image climbs. With [`BandConfig::fixed_atoms`] set the
/// fixed rows of the force are zero, and endpoints that place a fixed
/// atom differently are refused.
pub fn band_forces<S: BandSurface + ?Sized>(
    config: &BandConfig,
    surface: &S,
    positions: ArrayView2<f64>,
    ci: Option<usize>,
) -> Result<BandForces, SaddleError> {
    band_forces_with_gate(config, surface, positions, ci, ForceGate::LinfNorm)
}

fn band_forces_with_gate<S: BandSurface + ?Sized>(
    config: &BandConfig,
    surface: &S,
    positions: ArrayView2<f64>,
    ci: Option<usize>,
    force_gate: ForceGate,
) -> Result<BandForces, SaddleError> {
    let n_images = positions.nrows();
    let dof = positions.ncols();
    if n_images < 3 || dof == 0 || !dof.is_multiple_of(3) {
        return Err(SaddleError::Shape(format!(
            "a band needs at least 3 images of 3N coordinates; got {n_images} x {dof}"
        )));
    }
    crate::band::check_fixed(config, positions)?;
    let mut energies = Array1::zeros(n_images);
    let mut gradients = Array2::zeros((n_images, dof));
    surface.eval(positions, &mut energies, &mut gradients)?;
    if !gradients.iter().all(|g| g.is_finite()) {
        return Err(SaddleError::NonFinite("band gradients"));
    }
    for mut row in gradients.outer_iter_mut() {
        zero_fixed(&config.fixed_atoms, &mut row);
    }
    if !energies.iter().all(|e| e.is_finite()) {
        return Err(SaddleError::NonFinite("band energies"));
    }
    let is_string = config.string.is_some();
    let springs = config.spring.at_band(energies.view(), gradients.view());
    let mut tangents = Array2::zeros((n_images - 2, dof));
    let mut force = Array1::zeros((n_images - 2) * dof);
    for i in 1..n_images - 1 {
        let mut pos_diff_next = (&positions.row(i + 1) - &positions.row(i)).to_owned();
        let mut pos_diff_prev = (&positions.row(i) - &positions.row(i - 1)).to_owned();
        if let Some(cell) = &config.cell {
            cell.minimum_image(&mut pos_diff_next);
            cell.minimum_image(&mut pos_diff_prev);
        }
        zero_fixed(&config.fixed_atoms, &mut pos_diff_next);
        zero_fixed(&config.fixed_atoms, &mut pos_diff_prev);
        let dist_next = nrm2(pos_diff_next.view());
        let dist_prev = nrm2(pos_diff_prev.view());
        let true_force = -&gradients.row(i);
        let (tangent, mut image_force) = if is_string {
            if ci == Some(i) {
                let tangent = compute_tangent(
                    config.tangent,
                    pos_diff_next.view(),
                    pos_diff_prev.view(),
                    energies[i],
                    energies[i - 1],
                    energies[i + 1],
                )?;
                let reflected = climbing_image_force(
                    true_force.view(),
                    tangent.view(),
                    Array1::zeros(dof).view(),
                );
                (tangent, reflected)
            } else {
                let tangent = match compute_tangent(
                    config.tangent,
                    pos_diff_next.view(),
                    pos_diff_prev.view(),
                    energies[i],
                    energies[i - 1],
                    energies[i + 1],
                ) {
                    Ok(t) => t,
                    Err(_) => Array1::zeros(dof),
                };
                (tangent, true_force)
            }
        } else {
            let tangent = compute_tangent(
                config.tangent,
                pos_diff_next.view(),
                pos_diff_prev.view(),
                energies[i],
                energies[i - 1],
                energies[i + 1],
            )?;
            let spring = springs.compute(
                i,
                tangent.view(),
                dist_next,
                dist_prev,
                pos_diff_next.view(),
                pos_diff_prev.view(),
            )?;
            let projected = if ci == Some(i) {
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
            (tangent, projected)
        };
        zero_fixed(&config.fixed_atoms, &mut image_force);
        if config.drops_translation() {
            crate::band::remove_net_translation(&mut image_force);
        }
        force
            .slice_mut(s![(i - 1) * dof..i * dof])
            .assign(&image_force);
        tangents.row_mut(i - 1).assign(&tangent);
    }
    let max_force = force_gate.value(force.view());
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
    reparametrize_linear(positions, anchor, None);
}

/// Equal-arc interpolation using minimum-image chords, with fixed endpoints.
pub fn reparametrize_equal_arc_in_cell(
    positions: &mut Array2<f64>,
    anchor: Option<usize>,
    cell: &crate::Cell,
) {
    reparametrize_linear(positions, anchor, Some(cell));
}

pub(crate) fn reparametrize_linear(
    positions: &mut Array2<f64>,
    anchor: Option<usize>,
    cell: Option<&crate::Cell>,
) {
    let n = positions.nrows();
    if n < 3 {
        return;
    }
    let original = positions.clone();
    let mut orig = original.clone();
    let mut s = vec![0.0; n];
    for i in 1..n {
        let mut d = &original.row(i) - &original.row(i - 1);
        if let Some(cell) = cell {
            cell.minimum_image(&mut d);
        }
        let unwrapped = &orig.row(i - 1) + &d;
        orig.row_mut(i).assign(&unwrapped);
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
        let mut difference = &p - &original.row(j);
        if let Some(cell) = cell {
            cell.minimum_image(&mut difference);
        }
        positions
            .row_mut(j)
            .assign(&(&original.row(j) + &difference));
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
///
/// Non-exhaustive: start from [`RtrConfig::default`] and chain the
/// `with_*` setters.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
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
    /// Placement of the trial's interior images at equal arc length.
    /// [`Reparametrization::Linear`], the default, is the polyline.
    pub reparametrization: Reparametrization,
}

with_setters!(RtrConfig {
    with_radius_max => radius_max: val f64,
    with_theta => theta: val f64,
    with_kappa => kappa: val f64,
    with_max_cg => max_cg: val usize,
    with_fd_step => fd_step: val f64,
    with_reparametrization => reparametrization: val Reparametrization,
});

impl RtrConfig {
    pub(crate) fn validate(&self) -> Result<(), SaddleError> {
        if !(self.radius_max.is_finite()
            && self.radius_max > 0.0
            && self.theta.is_finite()
            && self.theta >= 0.0
            && self.kappa.is_finite()
            && self.kappa > 0.0
            && self.fd_step.is_finite()
            && self.fd_step > 0.0
            && self.max_cg > 0)
        {
            return Err(SaddleError::Invalid(
                "invalid band trust-region parameters".into(),
            ));
        }
        Ok(())
    }
}

impl Default for RtrConfig {
    fn default() -> Self {
        Self {
            radius_max: 1.0,
            theta: 1.0,
            kappa: 0.1,
            max_cg: 50,
            fd_step: 1e-4,
            reparametrization: Reparametrization::Linear,
        }
    }
}

/// One accepted or rejected band step.
#[derive(Clone, Debug)]
pub struct RtrReport {
    /// The accepted.
    pub accepted: bool,
    /// Ratio of the actual energy change to the model's prediction.
    pub rho: f64,
    /// The radius.
    pub radius: f64,
    /// The stop.
    pub stop: TcgStop,
    /// The cg iterations.
    pub cg_iterations: usize,
    /// The model decrease.
    pub model_decrease: f64,
    /// The actual decrease.
    pub actual_decrease: f64,
    /// Max NEB-force component after the step (before it, when rejected).
    pub max_force: f64,
    /// The ci.
    pub ci: Option<usize>,
}

/// Band trust-region stepper. Endpoints (rows 0 and `n_images - 1`) never
/// move; the climbing image is the highest interior image once
/// `climb` is set, the plain band otherwise.
#[derive(Clone, Debug)]
pub struct BandRtr {
    /// The config.
    pub config: RtrConfig,
    /// The radius.
    pub radius: RtrRadius,
    /// The climb.
    pub climb: bool,
    /// Number of steps taken.
    pub iteration: usize,
    force_gate: ForceGate,
    memo: HvpMemo,
}

/// Band forces of the Hessian-vector products around one base band
/// and climbing image. A rejected step leaves the base, the gradient
/// and so the truncated-CG directions as they were; the next step
/// reads their forces here instead of evaluating the same bands again.
/// The products of the step's displacement change with the radius, so
/// only the latest pair is kept. Keyed by the bits of the displaced
/// band.
#[derive(Clone, Debug, Default)]
struct HvpMemo {
    base: Option<(Array2<f64>, Option<usize>)>,
    products: Vec<(Array2<f64>, Array1<f64>)>,
    displacement: Vec<(Array2<f64>, Array1<f64>)>,
}

fn same_bits(a: &Array2<f64>, b: &Array2<f64>) -> bool {
    let rows_equal = |r: usize| {
        a.row(r)
            .iter()
            .zip(b.row(r))
            .all(|(x, y)| x.to_bits() == y.to_bits())
    };
    // Every key around one base shares its endpoint rows, so the
    // interior rows, where keys differ, are compared first.
    let n = a.nrows();
    a.dim() == b.dim() && (1..n.saturating_sub(1)).chain([0, n - 1]).all(rows_equal)
}

impl HvpMemo {
    /// Keep the entries when `base` and `ci` are those they were
    /// taken around, else drop them.
    fn around(&mut self, base: &Array2<f64>, ci: Option<usize>) {
        let kept = self
            .base
            .as_ref()
            .is_some_and(|(b, c)| *c == ci && same_bits(b, base));
        if !kept {
            self.products.clear();
            self.displacement.clear();
            self.base = Some((base.clone(), ci));
        }
    }
}

/// Where the products inside one trust step come from.
#[derive(Clone, Copy, PartialEq, Eq)]
enum HessianSource {
    /// The surface has not yet answered.
    Probe,
    /// The surface supplies analytic products.
    Analytic,
    /// The surface left the product empty, so the step differences forces.
    Difference,
}

fn accept_band_product(
    product: Array2<f64>,
    rows: usize,
    cols: usize,
) -> Result<Array2<f64>, SaddleError> {
    if product.nrows() != rows || product.ncols() != cols {
        return Err(SaddleError::Shape(
            "analytic band Hessian product has the wrong shape".into(),
        ));
    }
    if product.iter().any(|value| !value.is_finite()) {
        return Err(SaddleError::NonFinite("analytic band Hessian product"));
    }
    Ok(product)
}

/// Remove the net Cartesian translation with rgmin's rigid projector.
///
/// One atom has no internal translation to remove, matching
/// [`crate::band::remove_net_translation`].
fn project_out_translation(
    workspace: &mut RigidWorkspace,
    position: &[f64],
    vector: &mut Array1<f64>,
) {
    let atoms = vector.len() / 3;
    if atoms < 2 || vector.len() != 3 * atoms || position.len() != vector.len() {
        return;
    }
    let mut raw = vector.to_vec();
    let request = RigidRequest::cartesian(position, RigidMode::Translation);
    workspace.project(&mut raw, &request);
    for (slot, value) in vector.iter_mut().zip(raw) {
        *slot = value;
    }
}

/// Symmetric reflected Hessian on the climbing image:
/// `H v - (t · H v) t - (t · v) H t`.
fn reflected_hessian(
    action: ArrayView1<f64>,
    tangent: ArrayView1<f64>,
    tangent_action: ArrayView1<f64>,
    direction: ArrayView1<f64>,
) -> Array1<f64> {
    let along_action = tangent.dot(&action);
    let along_direction = tangent.dot(&direction);
    &action
        - &(&tangent.to_owned() * along_action)
        - &(&tangent_action.to_owned() * along_direction)
}

fn climbing_tangent_action<S: BandSurface + ?Sized>(
    surface: &S,
    base: ArrayView2<f64>,
    tangents: ArrayView2<f64>,
    ci: usize,
) -> Result<Option<Array1<f64>>, SaddleError> {
    let rows = tangents.nrows();
    let cols = tangents.ncols();
    let mut seed = Array2::zeros((rows, cols));
    seed.row_mut(ci - 1).assign(&tangents.row(ci - 1));
    match surface.hessian_vector(base, seed.view())? {
        None => Ok(None),
        Some(product) => {
            let product = accept_band_product(product, rows, cols)?;
            Ok(Some(product.row(ci - 1).to_owned()))
        }
    }
}

struct AnalyticParts<'a> {
    action: &'a Array2<f64>,
    direction: &'a Array2<f64>,
    base: &'a Array2<f64>,
    tangents: &'a Array2<f64>,
    climbing: Option<usize>,
    tangent_action: Option<&'a Array1<f64>>,
}

fn assemble_analytic_product(
    parts: &AnalyticParts<'_>,
    remove_translation: bool,
    fixed: &[bool],
    workspace: &mut RigidWorkspace,
) -> Result<Array1<f64>, SaddleError> {
    let n_images = parts.base.nrows();
    let dof = parts.base.ncols();
    let mut out = Array1::zeros((n_images - 2) * dof);
    for image in 0..n_images - 2 {
        let tangent = parts.tangents.row(image);
        let hv = parts.action.row(image);
        let mut segment = if parts.climbing == Some(image + 1) {
            let Some(along_tangent) = parts.tangent_action else {
                return Err(SaddleError::Invalid(
                    "climbing image analytic product needs the tangent action".into(),
                ));
            };
            reflected_hessian(
                hv,
                tangent,
                along_tangent.view(),
                parts.direction.row(image),
            )
        } else {
            crate::stencil::perp(hv, tangent)
        };
        zero_fixed(fixed, &mut segment);
        if remove_translation {
            let position = parts.base.row(image + 1).to_vec();
            project_out_translation(workspace, &position, &mut segment);
        }
        out.slice_mut(s![image * dof..(image + 1) * dof])
            .assign(&segment);
    }
    Ok(out)
}

fn keep_hvp_error(slot: &std::cell::RefCell<Option<SaddleError>>, error: SaddleError) {
    if slot.borrow().is_none() {
        *slot.borrow_mut() = Some(error);
    }
}

impl BandRtr {
    /// New.
    pub fn new(config: RtrConfig, climb: bool) -> Self {
        Self {
            config,
            radius: RtrRadius::new(config.radius_max),
            climb,
            iteration: 0,
            force_gate: ForceGate::LinfNorm,
            memo: HvpMemo::default(),
        }
    }

    /// Select the convergence norm without changing the trust model.
    pub fn set_force_gate(&mut self, gate: ForceGate) {
        if self.force_gate != gate {
            self.memo = HvpMemo::default();
        }
        self.force_gate = gate;
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
    ///
    /// Without [`BandSurface::hessian_vector`], each product takes two
    /// surface evaluations and the trial point takes one more. A supplied
    /// product replaces those probes: ordinary images use the component
    /// perpendicular to the tangent, and the climbing image uses the
    /// reflected Hessian. Net translation is removed with rgmin's rigid
    /// translation projector when the band drops it.
    pub fn step<S: BandSurface + ?Sized>(
        &mut self,
        band: &BandConfig,
        surface: &S,
        positions: &mut Array2<f64>,
    ) -> Result<RtrReport, SaddleError> {
        let n_images = positions.nrows();
        let dof = positions.ncols();
        self.config.validate()?;
        let here = band_forces_with_gate(band, surface, positions.view(), None, self.force_gate)?;
        let interior = (n_images - 2) * dof;
        let ci = self.climbing_index(&here.energies);
        let here = if ci.is_some() {
            band_forces_with_gate(band, surface, positions.view(), ci, self.force_gate)?
        } else {
            here
        };
        let grad = -&here.force;
        let tangents_hvp = here.tangents.clone();
        let tangents = tangents_hvp.clone();
        let project = move |v: &Array1<f64>| -> Array1<f64> {
            let mut out = v.clone();
            for i in 0..n_images - 2 {
                // The climbing force includes the reaction-path component.
                if ci == Some(i + 1) {
                    continue;
                }
                let tau = tangents.row(i);
                let mut seg = out.slice_mut(s![i * dof..(i + 1) * dof]);
                let along = seg.dot(&tau);
                seg.scaled_add(-along, &tau);
            }
            out
        };
        let h = self.config.fd_step;
        let base = positions.clone();
        let remove_translation = band.drops_translation();
        let fixed = band.fixed_atoms.clone();
        let tangent_action = match ci {
            Some(index) => {
                climbing_tangent_action(surface, base.view(), tangents_hvp.view(), index)?
            }
            None => None,
        };
        let source = std::cell::Cell::new(match (ci.is_some(), tangent_action.is_some()) {
            (true, true) => HessianSource::Analytic,
            (true, false) => HessianSource::Difference,
            (false, _) => HessianSource::Probe,
        });
        let translation = std::cell::RefCell::new(RigidWorkspace::new());
        self.memo.around(&base, ci);
        // Two bands per product, at most `max_cg` products per step.
        let products_capacity = 2 * self.config.max_cg;
        let memo = std::cell::RefCell::new(std::mem::take(&mut self.memo));
        // Set while the products of the step's displacement are taken,
        // which go to `fresh` and replace the previous step's pair.
        let displacing = std::cell::Cell::new(false);
        let fresh = std::cell::RefCell::new(Vec::new());
        let gate = self.force_gate;
        let force_at = |x: Array2<f64>| -> Result<Array1<f64>, SaddleError> {
            {
                let memo = memo.borrow();
                let mut held = memo.products.iter().chain(&memo.displacement);
                if let Some((_, force)) = held.find(|(key, _)| same_bits(key, &x)) {
                    return Ok(force.clone());
                }
            }
            let force = band_forces_with_gate(band, surface, x.view(), ci, gate)?.force;
            if displacing.get() {
                fresh.borrow_mut().push((x, force.clone()));
            } else {
                let mut memo = memo.borrow_mut();
                if memo.products.len() < products_capacity {
                    memo.products.push((x, force.clone()));
                }
            }
            Ok(force)
        };
        let hvp_error = std::cell::RefCell::new(None);
        let hvp = |v: &Array1<f64>| -> Array1<f64> {
            let vn = nrm2(v.view());
            if vn <= 0.0 || vn.is_nan() {
                return Array1::zeros(interior);
            }
            if source.get() != HessianSource::Difference {
                let direction = match Array2::from_shape_vec((n_images - 2, dof), v.to_vec()) {
                    Ok(direction) => direction,
                    Err(_) => {
                        keep_hvp_error(
                            &hvp_error,
                            SaddleError::Shape(
                                "analytic band Hessian product has the wrong shape".into(),
                            ),
                        );
                        return Array1::from_elem(interior, f64::INFINITY);
                    }
                };
                match surface.hessian_vector(base.view(), direction.view()) {
                    Ok(Some(product)) => match accept_band_product(product, n_images - 2, dof) {
                        Ok(action) => {
                            source.set(HessianSource::Analytic);
                            let parts = AnalyticParts {
                                action: &action,
                                direction: &direction,
                                base: &base,
                                tangents: &tangents_hvp,
                                climbing: ci,
                                tangent_action: tangent_action.as_ref(),
                            };
                            match assemble_analytic_product(
                                &parts,
                                remove_translation,
                                &fixed,
                                &mut translation.borrow_mut(),
                            ) {
                                Ok(out) => return out,
                                Err(error) => {
                                    keep_hvp_error(&hvp_error, error);
                                    return Array1::from_elem(interior, f64::INFINITY);
                                }
                            }
                        }
                        Err(error) => {
                            keep_hvp_error(&hvp_error, error);
                            return Array1::from_elem(interior, f64::INFINITY);
                        }
                    },
                    Ok(None) => {
                        if source.get() == HessianSource::Analytic {
                            keep_hvp_error(
                                &hvp_error,
                                SaddleError::Surface(
                                    "analytic band Hessian product was not supplied".into(),
                                ),
                            );
                            return Array1::from_elem(interior, f64::INFINITY);
                        }
                        source.set(HessianSource::Difference);
                    }
                    Err(error) => {
                        keep_hvp_error(&hvp_error, error);
                        return Array1::from_elem(interior, f64::INFINITY);
                    }
                }
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
            let plus = force_at(scatter(1.0));
            let minus = force_at(scatter(-1.0));
            match (plus, minus) {
                (Ok(p), Ok(m)) => (&m - &p) * (vn / (2.0 * h)),
                (Err(error), _) | (_, Err(error)) => {
                    keep_hvp_error(&hvp_error, error);
                    Array1::from_elem(interior, f64::INFINITY)
                }
            }
        };
        let tcg = truncated_cg(
            project,
            grad.view(),
            &hvp,
            self.radius.radius,
            self.config.theta,
            self.config.kappa,
            self.config.max_cg,
        );
        if let Some(error) = hvp_error.borrow_mut().take() {
            return Err(error);
        }
        self.iteration += 1;
        if tcg.stop == TcgStop::ZeroGradient {
            self.memo = memo.take();
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
        crate::reparam::reparametrize(
            &mut trial,
            ci,
            band.cell.as_ref(),
            self.config.reparametrization,
        );
        let mut displacement = Array1::zeros(interior);
        for i in 1..n_images - 1 {
            let mut difference = &trial.row(i) - &base.row(i);
            if let Some(cell) = &band.cell {
                cell.minimum_image(&mut difference);
            }
            displacement
                .slice_mut(s![(i - 1) * dof..i * dof])
                .assign(&difference);
        }
        zero_fixed(&band.fixed_atoms, &mut displacement);
        let length = nrm2(displacement.view());
        if length > self.radius.radius {
            displacement *= self.radius.radius / length;
        }
        for i in 1..n_images - 1 {
            let row = &base.row(i) + &displacement.slice(s![(i - 1) * dof..i * dof]);
            trial.row_mut(i).assign(&row);
        }
        displacing.set(true);
        let h_displacement = hvp(&displacement);
        if let Some(error) = hvp_error.borrow_mut().take() {
            return Err(error);
        }
        self.memo = memo.take();
        let fresh = fresh.take();
        if !fresh.is_empty() {
            self.memo.displacement = fresh;
        }
        let model_decrease = -dot(grad.view(), displacement.view())
            - 0.5 * dot(displacement.view(), h_displacement.view());
        // A displacement under the spacing of the coordinates leaves
        // the trial on the base.
        let there = if same_bits(&trial, &base) {
            here.clone()
        } else {
            band_forces_with_gate(band, surface, trial.view(), ci, self.force_gate)?
        };
        let grad_there = -&there.force;
        let actual_decrease = -0.5 * dot((&grad + &grad_there).view(), displacement.view());
        let eta_norm = nrm2(displacement.view());
        let rho = if model_decrease > 0.0 {
            actual_decrease / model_decrease
        } else {
            f64::NEG_INFINITY
        };
        let accepted = self
            .radius
            .update(actual_decrease, model_decrease, eta_norm);
        if accepted {
            positions.assign(&trial);
            self.memo = HvpMemo::default();
        }
        Ok(RtrReport {
            accepted,
            rho,
            radius: self.radius.radius,
            stop: tcg.stop,
            cg_iterations: tcg.iterations,
            model_decrease,
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

    /// Largest absolute coordinate gap allowed between one analytic trust
    /// step and the central difference of the same band force.
    const ANALYTIC_FINITE_DIFFERENCE_GAP: f64 = 1e-5;

    use crate::spring::SpringKind;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn quiet_band() -> crate::band::BandConfig {
        crate::band::BandConfig::default()
            .with_spring(SpringKind::Uniform { k: 0.0 })
            .with_remove_translation(true)
    }

    struct CountedQuadratic {
        curvatures: &'static [f64],
        analytic: bool,
        malformed: bool,
        evals: AtomicUsize,
        hvps: AtomicUsize,
    }

    impl BandSurface for CountedQuadratic {
        fn eval(
            &self,
            positions: ArrayView2<f64>,
            energies: &mut Array1<f64>,
            gradients: &mut Array2<f64>,
        ) -> Result<(), SaddleError> {
            self.evals.fetch_add(1, Ordering::Relaxed);
            for (image, row) in positions.outer_iter().enumerate() {
                let mut energy = 0.0;
                for (axis, &curvature) in self.curvatures.iter().enumerate() {
                    let coordinate = row[axis];
                    energy += 0.5 * curvature * coordinate * coordinate;
                    gradients[(image, axis)] = curvature * coordinate;
                }
                energies[image] = energy;
            }
            Ok(())
        }

        fn hessian_vector(
            &self,
            _positions: ArrayView2<f64>,
            direction: ArrayView2<f64>,
        ) -> Result<Option<Array2<f64>>, SaddleError> {
            if !self.analytic {
                return Ok(None);
            }
            self.hvps.fetch_add(1, Ordering::Relaxed);
            if self.malformed {
                return Ok(Some(Array2::zeros((1, 1))));
            }
            let mut action = Array2::zeros(direction.dim());
            for (image, row) in direction.outer_iter().enumerate() {
                for (axis, &curvature) in self.curvatures.iter().enumerate() {
                    action[(image, axis)] = curvature * row[axis];
                }
            }
            Ok(Some(action))
        }
    }

    fn counted(curvatures: &'static [f64], analytic: bool) -> CountedQuadratic {
        CountedQuadratic {
            curvatures,
            analytic,
            malformed: false,
            evals: AtomicUsize::new(0),
            hvps: AtomicUsize::new(0),
        }
    }

    fn max_abs_diff(left: &Array2<f64>, right: &Array2<f64>) -> f64 {
        left.iter()
            .zip(right.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max)
    }

    #[test]
    fn rigid_translation_matches_mean_removal() {
        let position = [0.2, -0.4, 0.1, 1.5, 0.3, -0.2, -0.7, 0.8, 0.4];
        let mut mean_free = array![0.4, -1.2, 0.3, 1.1, 0.5, -0.8, 0.2, 2.4, -0.1];
        let mut projected = mean_free.clone();
        crate::band::remove_net_translation(&mut mean_free);
        let mut workspace = RigidWorkspace::new();
        project_out_translation(&mut workspace, &position, &mut projected);
        let gap = mean_free
            .iter()
            .zip(projected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(gap < 1e-12, "{gap} {projected:?}");
    }

    #[test]
    fn analytic_hvp_skips_gradient_probes_and_matches_finite_differences() {
        const CURVATURES: [f64; 6] = [1.0, 4.0, 9.0, 2.0, 3.0, 5.0];
        let band = quiet_band();
        let start = Array2::from_shape_vec(
            (4, 6),
            vec![
                -1.5, 0.0, 0.0, -1.5, 1.0, 0.0, -0.4, 0.05, -0.02, -0.4, 1.04, 0.03, 0.45, -0.04,
                0.02, 0.45, 0.96, -0.01, 1.5, 0.0, 0.0, 1.5, 1.0, 0.0,
            ],
        )
        .expect("band shape");
        let analytic = counted(&CURVATURES, true);
        let mut analytic_positions = start.clone();
        let mut analytic_step = BandRtr::new(RtrConfig::default(), false);
        let analytic_report = analytic_step
            .step(&band, &analytic, &mut analytic_positions)
            .expect("analytic step");
        assert!(analytic_report.accepted);
        assert_eq!(
            analytic.evals.load(Ordering::Relaxed),
            2,
            "base evaluation and the trial, with no curvature probes"
        );
        assert!(analytic.hvps.load(Ordering::Relaxed) >= 1);

        let difference = counted(&CURVATURES, false);
        let mut difference_positions = start.clone();
        let mut difference_step = BandRtr::new(RtrConfig::default(), false);
        difference_step
            .step(&band, &difference, &mut difference_positions)
            .expect("finite difference step");
        assert!(difference.evals.load(Ordering::Relaxed) > 2);
        let gap = max_abs_diff(&analytic_positions, &difference_positions);
        assert!(
            gap < ANALYTIC_FINITE_DIFFERENCE_GAP,
            "analytic and finite difference positions differ by {gap}"
        );
    }

    #[test]
    fn climbing_image_uses_the_reflected_hessian() {
        // Negative curvature along the band, so the reflected operator is
        // positive definite and the Newton step stays inside the trust ball.
        const CURVATURES: [f64; 3] = [-2.0, 5.0, 8.0];
        let band = quiet_band();
        let options = RtrConfig::default().with_radius_max(8.0);
        let start = Array2::from_shape_vec(
            (3, 3),
            vec![-1.0, 0.2, -0.15, 0.35, 0.2, -0.15, 1.0, 0.2, -0.15],
        )
        .expect("band shape");
        let analytic = counted(&CURVATURES, true);
        let mut analytic_positions = start.clone();
        let mut analytic_step = BandRtr::new(options, true);
        analytic_step
            .step(&band, &analytic, &mut analytic_positions)
            .expect("climbing analytic step");
        assert_eq!(
            analytic.evals.load(Ordering::Relaxed),
            3,
            "two base evaluations and the trial, with no curvature probes"
        );
        assert!(analytic.hvps.load(Ordering::Relaxed) >= 1);
        assert!(
            (analytic_positions[(1, 0)] - start[(1, 0)]).abs() > 1e-3,
            "climbing image should move along the tangent"
        );

        let difference = counted(&CURVATURES, false);
        let mut difference_positions = start;
        let mut difference_step = BandRtr::new(options, true);
        difference_step
            .step(&band, &difference, &mut difference_positions)
            .expect("climbing finite difference step");
        let gap = max_abs_diff(&analytic_positions, &difference_positions);
        assert!(
            gap < ANALYTIC_FINITE_DIFFERENCE_GAP,
            "reflected Hessian and finite difference positions differ by {gap}"
        );
    }

    fn reflected_product(
        surface: &CountedQuadratic,
        band: &crate::band::BandConfig,
        base: &Array2<f64>,
        direction: &Array1<f64>,
    ) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let assembled = band_forces(band, surface, base.view(), Some(1)).expect("base force");
        let mut direction_rows = Array2::zeros((1, direction.len()));
        direction_rows.row_mut(0).assign(direction);
        let action = surface
            .hessian_vector(base.view(), direction_rows.view())
            .expect("direction product")
            .expect("analytic direction product");
        let mut tangent_rows = Array2::zeros(direction_rows.dim());
        tangent_rows.row_mut(0).assign(&assembled.tangents.row(0));
        let tangent_action = surface
            .hessian_vector(base.view(), tangent_rows.view())
            .expect("tangent product")
            .expect("analytic tangent product");
        let along = tangent_action.row(0).to_owned();
        let product = assemble_analytic_product(
            &AnalyticParts {
                action: &action,
                direction: &direction_rows,
                base,
                tangents: &assembled.tangents,
                climbing: Some(1),
                tangent_action: Some(&along),
            },
            band.drops_translation(),
            &band.fixed_atoms,
            &mut RigidWorkspace::new(),
        )
        .expect("reflected product");
        (
            product,
            action.row(0).to_owned(),
            assembled.tangents.row(0).to_owned(),
        )
    }

    /// Central difference of a climbing force, with the same scaling the
    /// trust step uses for a direction of length `vn`.
    fn central_force_difference(
        plus: &Array1<f64>,
        minus: &Array1<f64>,
        vn: f64,
        h: f64,
    ) -> Array1<f64> {
        (minus - plus) * (vn / (2.0 * h))
    }

    #[test]
    fn climbing_reflected_product_matches_central_force_difference() {
        // Straight band, so the tangent is the x axis and an eigenvector of
        // the diagonal Hessian. The reflected product then equals the
        // central difference of the climbing force at that tangent, and the
        // tangent component of Hv changes sign.
        const CURVATURES: [f64; 3] = [-2.0, 5.0, 8.0];
        let band = quiet_band();
        let surface = counted(&CURVATURES, true);
        let base = Array2::from_shape_vec(
            (3, 3),
            vec![-1.0, 0.2, -0.15, 0.35, 0.2, -0.15, 1.0, 0.2, -0.15],
        )
        .expect("band shape");
        let h = 1e-6;
        let direction = array![0.35, -0.2, 0.15];
        let (product, raw, tangent) = reflected_product(&surface, &band, &base, &direction);
        let vn = nrm2(direction.view());
        let climbing_at = |sign: f64| {
            let mut shifted = base.clone();
            shifted.row_mut(1).scaled_add(sign * h / vn, &direction);
            let mut energies = Array1::zeros(shifted.nrows());
            let mut gradients = Array2::zeros(shifted.dim());
            surface
                .eval(shifted.view(), &mut energies, &mut gradients)
                .expect("shifted energy");
            let true_force = -&gradients.row(1).to_owned();
            climbing_image_force(
                true_force.view(),
                tangent.view(),
                Array1::zeros(direction.len()).view(),
            )
        };
        let finite = central_force_difference(&climbing_at(1.0), &climbing_at(-1.0), vn, h);
        let gap = finite
            .iter()
            .zip(product.iter())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0, f64::max);
        assert!(
            finite.dot(&product) > 0.0,
            "reflected product and the climbing-force difference disagree in sign\n{product:?}\n{finite:?}"
        );
        assert!(
            gap < 1e-8,
            "reflected product and the central climbing-force difference differ by {gap}\n{product:?}\n{finite:?}"
        );
        let parallel_raw = tangent.dot(&raw);
        let parallel_product = tangent.dot(&product);
        assert!(
            parallel_raw < 0.0 && parallel_product > 0.0,
            "climbing product should reverse the tangent component of Hv ({parallel_raw} -> {parallel_product})"
        );
        assert!(
            (parallel_product + parallel_raw).abs() < 1e-8,
            "reversed tangent component {parallel_product} is not the negation of {parallel_raw}"
        );

        // Along the tangent the assembled band force has the same difference:
        // both segments stay on the line, so the tangent does not tilt.
        let along = tangent.clone();
        let (along_product, _, _) = reflected_product(&surface, &band, &base, &along);
        let along_norm = nrm2(along.view());
        let band_force_at = |sign: f64| {
            let mut shifted = base.clone();
            shifted.row_mut(1).scaled_add(sign * h / along_norm, &along);
            band_forces(&band, &surface, shifted.view(), Some(1))
                .expect("shifted band force")
                .force
        };
        let assembled =
            central_force_difference(&band_force_at(1.0), &band_force_at(-1.0), along_norm, h);
        let assembled_gap = assembled
            .iter()
            .zip(along_product.iter())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0, f64::max);
        assert!(
            assembled.dot(&along_product) > 0.0 && assembled_gap < 1e-7,
            "assembled climbing force difference {assembled:?} product {along_product:?} gap {assembled_gap}"
        );
    }

    #[test]
    fn malformed_analytic_product_is_rejected() {
        let band = quiet_band();
        let surface = CountedQuadratic {
            curvatures: &[1.0, 4.0, 9.0],
            analytic: true,
            malformed: true,
            evals: AtomicUsize::new(0),
            hvps: AtomicUsize::new(0),
        };
        let mut positions =
            Array2::from_shape_vec((3, 3), vec![-1.0, 0.0, 0.0, 0.2, 0.3, 0.1, 1.0, 0.0, 0.0])
                .expect("band shape");
        let mut step = BandRtr::new(RtrConfig::default(), false);
        let error = step
            .step(&band, &surface, &mut positions)
            .expect_err("wrong product shape");
        assert!(matches!(error, SaddleError::Shape(_)));
    }
}
