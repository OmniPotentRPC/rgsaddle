//! Double-ended growing string.
//!
//! Peters, Heyden, Bell, and Chakraborty, J. Chem. Phys. 120, 7877
//! (2004), doi:10.1063/1.1691018. Two fragments grow from the reactant
//! and the product. A frontier advances only after its force
//! perpendicular to the growth direction has fallen. On the alanine
//! dipeptide, where a linear guess is far from the path, that paper
//! finds the saddle with significantly fewer force evaluations than
//! the nudged elastic band or the zero-temperature string.
//!
//! Once the fragments meet, each node steps on the force normal to the
//! string, which is the search direction in that paper, and the nodes
//! are put back on equal arcs. The equal-arc step is the
//! reparameterization of E, Ren, and Vanden-Eijnden, J. Chem. Phys.
//! 126, 164103 (2007), doi:10.1063/1.2720838. The highest interior node
//! climbs, Henkelman, Uberuaga, and Jónsson, J. Chem. Phys. 113, 9901
//! (2000), doi:10.1063/1.1329672. The opt-in is [`grow_string`]. The
//! default band remains CI-NEB.

use ndarray::{Array1, Array2, ArrayView1, s};

use crate::band::{BandSurface, remove_net_translation};
use crate::error::SaddleError;
use crate::rtr::reparametrize_equal_arc;

/// Controls for [`grow_string`].
#[derive(Clone, Debug)]
pub struct GrowingString {
    /// Images including the endpoints. At least 3.
    pub images: usize,
    /// Stop when the climbing node's force norm is under this value.
    pub force_tol: f64,
    /// Largest per-atom displacement of one relaxation or string step.
    pub max_move: f64,
    /// A frontier may advance once its perpendicular force is under this.
    pub frontier_tol: f64,
    /// Force evaluations spent on one frontier before it advances anyway.
    pub relax_per_node: usize,
    /// String steps after the fragments join.
    pub max_string_steps: usize,
}

impl Default for GrowingString {
    fn default() -> Self {
        Self {
            images: 7,
            force_tol: 0.01,
            max_move: 0.2,
            frontier_tol: 0.05,
            relax_per_node: 25,
            max_string_steps: 2000,
        }
    }
}

/// The joined string and what it cost.
#[derive(Clone, Debug)]
pub struct GrowingStringReport {
    pub positions: Array2<f64>,
    /// Potential evaluations, one per image row.
    pub calls: usize,
    pub converged: bool,
    /// Climbing node, or `None` when the string is monotonic.
    pub ci_index: Option<usize>,
    /// Convergence scalar at the last string step: the climbing node's
    /// force norm, or the largest node norm when no node climbs.
    pub max_force: f64,
}

/// Grow a string from `reactant` to `product` and converge its climbing node.
pub fn grow_string<S: BandSurface + ?Sized>(
    reactant: ArrayView1<f64>,
    product: ArrayView1<f64>,
    config: &GrowingString,
    surface: &S,
) -> Result<GrowingStringReport, SaddleError> {
    if config.images < 3 {
        return Err(SaddleError::Invalid(
            "a growing string needs at least 3 images".into(),
        ));
    }
    if reactant.len() != product.len() || reactant.is_empty() || !reactant.len().is_multiple_of(3) {
        return Err(SaddleError::Shape(
            "growing-string endpoints must be matching 3N vectors".into(),
        ));
    }
    for (name, v) in [
        ("force_tol", config.force_tol),
        ("max_move", config.max_move),
        ("frontier_tol", config.frontier_tol),
    ] {
        if !(v.is_finite() && v > 0.0) {
            return Err(SaddleError::Invalid(format!(
                "growing-string {name} must be finite and positive"
            )));
        }
    }
    if config.relax_per_node == 0 || config.max_string_steps == 0 {
        return Err(SaddleError::Invalid(
            "growing-string budgets must be positive".into(),
        ));
    }
    let mut calls = 0usize;
    let mut left = vec![reactant.to_owned()];
    let mut right = vec![product.to_owned()];
    let sep = (&product.to_owned() - &reactant)
        .dot(&(&product.to_owned() - &reactant))
        .sqrt();
    let spacing = sep / (config.images - 1) as f64;
    while left.len() + right.len() < config.images {
        let gap = distance(left.last().unwrap().view(), right.last().unwrap().view());
        if gap < spacing.max(1e-8) {
            break;
        }
        advance(
            &mut left,
            right.last().unwrap(),
            spacing,
            config,
            surface,
            &mut calls,
        )?;
        if left.len() + right.len() >= config.images {
            break;
        }
        let gap = distance(left.last().unwrap().view(), right.last().unwrap().view());
        if gap < spacing.max(1e-8) {
            break;
        }
        advance(
            &mut right,
            left.last().unwrap(),
            spacing,
            config,
            surface,
            &mut calls,
        )?;
    }
    let mut images = join(&left, &right);
    if images.len() < 3 {
        return Err(SaddleError::Stalled(
            "growing string joined with no interior node".into(),
        ));
    }
    let (converged, ci, max_force) = evolve(&mut images, config, surface, &mut calls)?;
    let dof = reactant.len();
    let mut positions = Array2::zeros((images.len(), dof));
    for (i, image) in images.iter().enumerate() {
        positions.row_mut(i).assign(image);
    }
    Ok(GrowingStringReport {
        positions,
        calls,
        converged,
        ci_index: ci,
        max_force,
    })
}

fn advance<S: BandSurface + ?Sized>(
    side: &mut Vec<Array1<f64>>,
    toward: &Array1<f64>,
    spacing: f64,
    config: &GrowingString,
    surface: &S,
    calls: &mut usize,
) -> Result<(), SaddleError> {
    let frontier = side.last().unwrap();
    let mut dir = toward - frontier;
    let gap = dir.dot(&dir).sqrt();
    if gap < 1e-10 {
        return Ok(());
    }
    dir /= gap;
    let step = spacing.min(0.5 * gap);
    let mut node = frontier + &dir * step;
    for _ in 0..config.relax_per_node {
        let (_, g) = eval_one(surface, node.view(), calls)?;
        let mut force = -g;
        let parallel = force.dot(&dir);
        force = &force - &(&dir * parallel);
        remove_net_translation(&mut force);
        if force.dot(&force).sqrt() < config.frontier_tol {
            break;
        }
        let mut dx = &force * 0.1;
        cap_atoms(&mut dx, config.max_move);
        node = &node + &dx;
    }
    side.push(node);
    Ok(())
}

fn join(left: &[Array1<f64>], right: &[Array1<f64>]) -> Vec<Array1<f64>> {
    let mut images = left.to_vec();
    for node in right.iter().rev() {
        let close = images
            .last()
            .is_some_and(|prev| distance(prev.view(), node.view()) < 1e-8);
        if !close {
            images.push(node.clone());
        }
    }
    images
}

fn evolve<S: BandSurface + ?Sized>(
    images: &mut [Array1<f64>],
    config: &GrowingString,
    surface: &S,
    calls: &mut usize,
) -> Result<(bool, Option<usize>, f64), SaddleError> {
    let n = images.len();
    let dof = images[0].len();
    let n_int = n - 2;
    let mut vel = Array1::zeros(n_int * dof);
    let mut dt = 0.1_f64;
    let mut alpha = 0.1_f64;
    let mut n_pos = 0usize;
    let mut last_force = f64::INFINITY;
    let mut last_ci = None;
    let (mut energies, mut grads) = eval_all(surface, images, calls)?;
    for _ in 0..config.max_string_steps {
        let ci = climbing_node(&energies);
        let mut force = Array1::zeros(n_int * dof);
        for i in 1..n - 1 {
            let mut t = &images[i + 1] - &images[i - 1];
            let tn = t.dot(&t).sqrt();
            if tn > 1e-12 {
                t /= tn;
            }
            let mut f = -grads.row(i).to_owned();
            let parallel = f.dot(&t);
            // Peters steps on the normal force. The climbing node reflects
            // the parallel part instead. Reparameterization is in addition.
            if ci == Some(i) {
                f = &f - &(&t * (2.0 * parallel));
            } else {
                f = &f - &(&t * parallel);
            }
            remove_net_translation(&mut f);
            force.slice_mut(s![(i - 1) * dof..i * dof]).assign(&f);
        }
        let scalar = convergence(&force, dof, ci, config.force_tol);
        if scalar < config.force_tol {
            return Ok((true, ci, scalar));
        }
        // FIRE 1, Bitzek et al. and eOn: integrate, then mix on the new force.
        // A zero velocity is rest, not a failed step, so the time step
        // shrinks only when the new force opposes a velocity that was moving.
        vel = &vel + &force * dt;
        let proposed = &vel * dt;
        let mut dx = proposed.clone();
        cap_atoms(&mut dx, config.max_move);
        let proposed_norm = proposed.dot(&proposed).sqrt();
        let taken_norm = dx.dot(&dx).sqrt();
        if proposed_norm > 0.0 && taken_norm < proposed_norm {
            vel *= taken_norm / proposed_norm;
        }
        for i in 1..n - 1 {
            let step = dx.slice(s![(i - 1) * dof..i * dof]);
            images[i] = &images[i] + &step;
        }
        let mut band = Array2::zeros((n, dof));
        for (i, image) in images.iter().enumerate() {
            band.row_mut(i).assign(image);
        }
        // Reparameterize every few steps so the climb is not undone
        // on every iteration. Peters redistributes after the evolution.
        if n_pos % 5 == 4 {
            reparametrize_equal_arc(&mut band, ci);
            vel.fill(0.0);
        }
        for (i, image) in images.iter_mut().enumerate() {
            image.assign(&band.row(i));
        }
        let (e2, g2) = eval_all(surface, images, calls)?;
        energies = e2;
        grads = g2;
        let ci2 = climbing_node(&energies);
        let mut force2 = Array1::zeros(n_int * dof);
        for i in 1..n - 1 {
            let mut t = &images[i + 1] - &images[i - 1];
            let tn = t.dot(&t).sqrt();
            if tn > 1e-12 {
                t /= tn;
            }
            let mut f = -grads.row(i).to_owned();
            let parallel = f.dot(&t);
            if ci2 == Some(i) {
                f = &f - &(&t * (2.0 * parallel));
            } else {
                f = &f - &(&t * parallel);
            }
            remove_net_translation(&mut f);
            force2.slice_mut(s![(i - 1) * dof..i * dof]).assign(&f);
        }
        let power = force2.dot(&vel);
        let fnorm = force2.dot(&force2).sqrt();
        let vnorm = vel.dot(&vel).sqrt();
        if fnorm > 0.0 && vnorm > 0.0 {
            let a = alpha.clamp(0.0, 1.0);
            vel = &vel * (1.0 - a) + &force2 * (a * vnorm / fnorm);
        }
        if power > 0.0 {
            n_pos += 1;
            if n_pos > 5 {
                dt = (dt * 1.1).min(0.25);
                alpha *= 0.99;
            }
        } else if vnorm > 0.0 {
            dt = (dt * 0.5).max(1e-3);
            vel.fill(0.0);
            alpha = 0.1;
            n_pos = 0;
        }
        let scalar2 = convergence(&force2, dof, ci2, config.force_tol);
        last_force = scalar2;
        last_ci = ci2;
        if scalar2 < config.force_tol {
            return Ok((true, ci2, scalar2));
        }
    }
    Ok((false, last_ci, last_force))
}

fn climbing_node(energies: &Array1<f64>) -> Option<usize> {
    let n = energies.len();
    if n < 3 {
        return None;
    }
    let mut max_i = 1usize;
    for i in 2..n - 1 {
        if energies[i] > energies[max_i] {
            max_i = i;
        }
    }
    let higher = energies[0].max(energies[n - 1]);
    (energies[max_i] > higher).then_some(max_i)
}

fn convergence(force: &Array1<f64>, dof: usize, ci: Option<usize>, tol: f64) -> f64 {
    let n = force.len() / dof;
    let mut band = 0.0_f64;
    let mut local = None;
    for i in 0..n {
        let row = force.slice(s![i * dof..(i + 1) * dof]);
        let nrm = row.dot(&row).sqrt();
        band = band.max(nrm);
        if ci == Some(i + 1) {
            local = Some(nrm);
        }
    }
    match local {
        Some(value) if band <= 10.0 * tol => value,
        _ => band,
    }
}

fn eval_one<S: BandSurface + ?Sized>(
    surface: &S,
    x: ArrayView1<f64>,
    calls: &mut usize,
) -> Result<(f64, Array1<f64>), SaddleError> {
    let mut positions = Array2::zeros((1, x.len()));
    positions.row_mut(0).assign(&x);
    let mut energies = Array1::zeros(1);
    let mut gradients = Array2::zeros((1, x.len()));
    surface.eval(positions.view(), &mut energies, &mut gradients)?;
    *calls += 1;
    if !energies[0].is_finite() {
        return Err(SaddleError::NonFinite("growing-string energy"));
    }
    Ok((energies[0], gradients.row(0).to_owned()))
}

fn eval_all<S: BandSurface + ?Sized>(
    surface: &S,
    images: &[Array1<f64>],
    calls: &mut usize,
) -> Result<(Array1<f64>, Array2<f64>), SaddleError> {
    let n = images.len();
    let dof = images[0].len();
    let mut positions = Array2::zeros((n, dof));
    for (i, image) in images.iter().enumerate() {
        positions.row_mut(i).assign(image);
    }
    let mut energies = Array1::zeros(n);
    let mut gradients = Array2::zeros((n, dof));
    surface.eval(positions.view(), &mut energies, &mut gradients)?;
    *calls += n;
    if !energies.iter().all(|e| e.is_finite()) {
        return Err(SaddleError::NonFinite("growing-string energy"));
    }
    Ok((energies, gradients))
}

fn distance(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    let d = &a - &b;
    d.dot(&d).sqrt()
}

fn cap_atoms(step: &mut Array1<f64>, max_move: f64) {
    let mut max = 0.0_f64;
    let n = step.len() / 3;
    for i in 0..n {
        let nrm = (step[3 * i].powi(2) + step[3 * i + 1].powi(2) + step[3 * i + 2].powi(2)).sqrt();
        max = max.max(nrm);
    }
    if max > max_move && max.is_finite() {
        *step *= max_move / max;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayView2, array};

    struct Well;

    impl BandSurface for Well {
        fn eval(
            &self,
            positions: ArrayView2<f64>,
            energies: &mut Array1<f64>,
            gradients: &mut Array2<f64>,
        ) -> Result<(), SaddleError> {
            for (i, row) in positions.outer_iter().enumerate() {
                let (x, y, z) = (row[0], row[1], row[2]);
                energies[i] = (x * x - 1.0).powi(2) + 2.0 * y * y + 2.0 * z * z;
                gradients[(i, 0)] = 4.0 * x * (x * x - 1.0);
                gradients[(i, 1)] = 4.0 * y;
                gradients[(i, 2)] = 4.0 * z;
            }
            Ok(())
        }
    }

    #[test]
    fn the_growing_string_reaches_the_double_well_saddle() {
        let reactant = array![-1.0, 0.0, 0.0];
        let product = array![1.0, 0.0, 0.0];
        let config = GrowingString {
            images: 7,
            force_tol: 1e-2,
            max_string_steps: 400,
            ..GrowingString::default()
        };
        let report = grow_string(reactant.view(), product.view(), &config, &Well).unwrap();
        assert!(report.converged, "force {}", report.max_force);
        let ci = report.ci_index.expect("climbing node");
        let x = report.positions[(ci, 0)];
        assert!(x.abs() < 0.15, "saddle x {x}, calls {}", report.calls);
        assert!(report.calls > 0);
    }
}
