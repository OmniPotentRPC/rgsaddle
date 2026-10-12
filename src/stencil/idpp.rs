//! Image dependent pair potential.
//!
//! Smidstrup, Pedersen, Stokbro, and Jónsson, J. Chem. Phys. 140, 214106
//! (2014), doi:10.1063/1.4878664. The ideal distance between each atom
//! pair is the linear interpolation of the endpoint distances. The
//! objective weights the squared error by the inverse fourth power of
//! the current distance. The energy and the Cartesian force match the
//! ASE `IDPP` calculator, including both triangles of the pair matrix
//! and the factor of one half.

use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2};
use rgmin::{Accept, Control, Method, Oracle, Solver};

use crate::error::SaddleError;
use crate::stencil::{StencilReport, StencilStatus};

/// Pair distances of one image. `coords` is `3 N` Cartesian components.
pub fn pair_distances(coords: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
    let n = atom_count(coords.len())?;
    let mut out = Array2::zeros((n, n));
    for a in 0..n {
        for b in a + 1..n {
            let mut r2 = 0.0;
            for k in 0..3 {
                let delta = coords[3 * b + k] - coords[3 * a + k];
                r2 += delta * delta;
            }
            let dist = r2.sqrt();
            out[(a, b)] = dist;
            out[(b, a)] = dist;
        }
    }
    Ok(out)
}

/// Ideal distances for every image, including the frozen endpoints.
pub fn interpolated_targets(images: ArrayView2<f64>) -> Result<Vec<Array2<f64>>, SaddleError> {
    let n_img = images.nrows();
    if n_img < 3 {
        return Err(SaddleError::Shape(
            "IDPP needs at least three images".into(),
        ));
    }
    let d0 = pair_distances(images.row(0))?;
    let d1 = pair_distances(images.row(n_img - 1))?;
    let mut targets = Vec::with_capacity(n_img);
    let denom = (n_img - 1) as f64;
    for i in 0..n_img {
        let t = i as f64 / denom;
        targets.push(&d0 + &(&d1 - &d0) * t);
    }
    Ok(targets)
}

/// ASE energy and Cartesian force (`F = -dS/dx`) for one image.
pub fn energy_force(
    coords: ArrayView1<f64>,
    target: ArrayView2<f64>,
) -> Result<(f64, Array1<f64>), SaddleError> {
    let n = atom_count(coords.len())?;
    if target.nrows() != n || target.ncols() != n {
        return Err(SaddleError::Shape(
            "IDPP target distances must be one entry per atom pair".into(),
        ));
    }
    if !coords.iter().all(|value| value.is_finite())
        || !target.iter().all(|value| value.is_finite())
    {
        return Err(SaddleError::NonFinite("IDPP"));
    }
    let mut d = Array2::zeros((n, n));
    let mut disp = Array3::zeros((n, n, 3));
    for a in 0..n {
        for b in 0..n {
            if a == b {
                continue;
            }
            let mut r2 = 0.0;
            for k in 0..3 {
                let delta = coords[3 * b + k] - coords[3 * a + k];
                disp[(a, b, k)] = delta;
                r2 += delta * delta;
            }
            d[(a, b)] = r2.sqrt();
        }
    }
    let dd = &d - &target;
    for a in 0..n {
        d[(a, a)] = 1.0;
    }
    let mut energy = 0.0;
    let mut force = vec![0.0; 3 * n];
    for a in 0..n {
        for b in 0..n {
            let dist = d[(a, b)];
            let delta = dd[(a, b)];
            energy += 0.5 * (delta * delta / dist.powi(4));
            let coeff = delta * (1.0 - 2.0 * delta / dist) / dist.powi(5);
            for k in 0..3 {
                force[3 * b + k] += -2.0 * coeff * disp[(a, b, k)];
            }
        }
    }
    Ok((energy, Array1::from_vec(force)))
}

pub(crate) fn atom_count(dof: usize) -> Result<usize, SaddleError> {
    if dof == 0 || !dof.is_multiple_of(3) {
        return Err(SaddleError::Shape(
            "IDPP coordinates must come in groups of three".into(),
        ));
    }
    let n = dof / 3;
    if n < 2 {
        return Err(SaddleError::Shape("IDPP needs at least two atoms".into()));
    }
    Ok(n)
}

/// Tangent used by the IDPP NEB force.
///
/// `Improved` matches ASE. `Prenormalized` normalises each chord before
/// the weighted average, which sequential construction needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TangentStyle {
    Improved,
    Prenormalized,
}

/// Force on one interior image: maximum absolute component and RMS.
#[derive(Clone, Copy, Debug)]
pub(crate) struct InteriorLoad {
    pub max_abs: f64,
    pub rms: f64,
}

pub(crate) struct NebLoad {
    pub objective: f64,
    pub projected: Array1<f64>,
    pub max_component: f64,
    pub interiors: Vec<InteriorLoad>,
}

/// NEB force on the interior images. The true force is the IDPP force.
/// Translation is left in the force so a two-atom path can rotate.
pub(crate) fn neb_load(
    images: ArrayView2<f64>,
    targets: &[Array2<f64>],
    ks: &[f64],
    tangent_style: TangentStyle,
) -> Result<NebLoad, SaddleError> {
    let n = images.nrows();
    let dof = images.ncols();
    if targets.len() != n || ks.len() + 1 != n {
        return Err(SaddleError::Shape(
            "IDPP springs must match the image count".into(),
        ));
    }
    let mut energies = vec![0.0; n];
    let mut forces = Array2::zeros((n, dof));
    for i in 0..n {
        let (energy, force) = energy_force(images.row(i), targets[i].view())?;
        energies[i] = energy;
        forces.row_mut(i).assign(&force);
    }
    let mut projected = Array1::zeros((n - 2) * dof);
    let mut max_component = 0.0_f64;
    let mut interiors = Vec::with_capacity(n.saturating_sub(2));
    for i in 1..n - 1 {
        let pos_prev = images.row(i - 1);
        let pos = images.row(i);
        let pos_next = images.row(i + 1);
        let diff_next = &pos_next - &pos;
        let diff_prev = &pos - &pos_prev;
        let dist_next = diff_next.dot(&diff_next).sqrt();
        let dist_prev = diff_prev.dot(&diff_prev).sqrt();
        let tangent = tangent_of(
            tangent_style,
            diff_next.view(),
            diff_prev.view(),
            energies[i],
            energies[i - 1],
            energies[i + 1],
        )
        .map_err(|err| SaddleError::Invalid(format!("image {i}: {err}")))?;
        let true_force = forces.row(i).to_owned();
        let perp = super::perp(true_force.view(), tangent.view());
        let spring = tangent.to_owned() * (ks[i] * dist_next - ks[i - 1] * dist_prev);
        let neb = &perp + &spring;
        let mut sum_sq = 0.0_f64;
        let mut count = 0.0_f64;
        let mut max_abs = 0.0_f64;
        for value in neb.iter() {
            max_abs = max_abs.max(value.abs());
            max_component = max_component.max(value.abs());
            sum_sq += value * value;
            count += 1.0;
        }
        interiors.push(InteriorLoad {
            max_abs,
            rms: if count == 0.0 {
                0.0
            } else {
                (sum_sq / count).sqrt()
            },
        });
        let row = i - 1;
        projected
            .slice_mut(ndarray::s![row * dof..(row + 1) * dof])
            .assign(&neb);
    }
    let objective: f64 = energies[1..n - 1].iter().sum();
    Ok(NebLoad {
        objective,
        projected,
        max_component,
        interiors,
    })
}

pub(crate) fn tangent_of(
    style: TangentStyle,
    pos_diff_next: ArrayView1<f64>,
    pos_diff_prev: ArrayView1<f64>,
    energy: f64,
    energy_prev: f64,
    energy_next: f64,
) -> Result<Array1<f64>, SaddleError> {
    if style == TangentStyle::Improved {
        return super::unit_tangent(
            super::TangentRule::Improved,
            pos_diff_next,
            pos_diff_prev,
            energy,
            energy_prev,
            energy_next,
        );
    }
    let hat = |v: ArrayView1<f64>| {
        let norm = v.dot(&v).sqrt();
        if norm > 1e-15 && norm.is_finite() {
            v.to_owned() / norm
        } else {
            Array1::zeros(v.len())
        }
    };
    let tang = if energy_next > energy && energy > energy_prev {
        pos_diff_next.to_owned()
    } else if energy > energy_next && energy_prev > energy {
        pos_diff_prev.to_owned()
    } else {
        let next_hat = hat(pos_diff_next);
        let prev_hat = hat(pos_diff_prev);
        let diff_prev = energy_prev - energy;
        let diff_next = energy_next - energy;
        let min_diff = diff_prev.abs().min(diff_next.abs());
        let max_diff = diff_prev.abs().max(diff_next.abs());
        if diff_prev > diff_next {
            &next_hat * min_diff + &prev_hat * max_diff
        } else {
            &next_hat * max_diff + &prev_hat * min_diff
        }
    };
    let norm = tang.dot(&tang).sqrt();
    if norm > 1e-10 && norm.is_finite() {
        return Ok(tang / norm);
    }
    super::unit_tangent(
        super::TangentRule::Simple,
        pos_diff_next,
        pos_diff_prev,
        energy,
        energy_prev,
        energy_next,
    )
}

pub(crate) fn write_interior(
    images: &mut Array2<f64>,
    flat: ArrayView1<f64>,
) -> Result<(), SaddleError> {
    let n = images.nrows();
    let dof = images.ncols();
    let width = n.saturating_sub(2) * dof;
    if flat.len() != width {
        return Err(SaddleError::Shape(
            "IDPP step width does not match the interior images".into(),
        ));
    }
    for i in 1..n - 1 {
        let row = i - 1;
        for k in 0..dof {
            images[(i, k)] = flat[row * dof + k];
        }
    }
    Ok(())
}

/// IDPP relaxation of a fixed image count. Endpoints stay put.
pub struct IdppRun {
    images: Array2<f64>,
    targets: Vec<Array2<f64>>,
    ks: Vec<f64>,
    force_tol: f64,
    max_move: f64,
    method: Method,
    tangent: TangentStyle,
    solver: Solver,
    solver_n: usize,
    iteration: usize,
    max_force: f64,
}

impl IdppRun {
    pub fn new(
        images: Array2<f64>,
        k: f64,
        force_tol: f64,
        max_move: f64,
        method: Method,
    ) -> Result<Self, SaddleError> {
        check_band(&images, k, force_tol, max_move)?;
        let n = images.nrows();
        let ks = vec![k; n - 1];
        Self::from_parts(images, ks, force_tol, max_move, method)
    }

    pub fn from_parts(
        images: Array2<f64>,
        ks: Vec<f64>,
        force_tol: f64,
        max_move: f64,
        method: Method,
    ) -> Result<Self, SaddleError> {
        let targets = interpolated_targets(images.view())?;
        Self::from_targets(
            images,
            targets,
            ks,
            force_tol,
            max_move,
            method,
            TangentStyle::Improved,
        )
    }

    pub(crate) fn from_targets(
        images: Array2<f64>,
        targets: Vec<Array2<f64>>,
        ks: Vec<f64>,
        force_tol: f64,
        max_move: f64,
        method: Method,
        tangent: TangentStyle,
    ) -> Result<Self, SaddleError> {
        let k = *ks.first().unwrap_or(&1.0);
        check_band(&images, k, force_tol, max_move)?;
        if ks.len() + 1 != images.nrows() || targets.len() != images.nrows() {
            return Err(SaddleError::Shape(
                "IDPP springs must match the image count".into(),
            ));
        }
        if ks.iter().any(|value| !(value.is_finite() && *value > 0.0)) {
            return Err(SaddleError::Invalid(
                "spring constant must be finite and positive".into(),
            ));
        }
        let width = (images.nrows() - 2) * images.ncols();
        let solver = idpp_solver(method.clone(), width, max_move)?;
        Ok(Self {
            images,
            targets,
            ks,
            force_tol,
            max_move,
            method,
            tangent,
            solver,
            solver_n: width,
            iteration: 0,
            max_force: f64::NAN,
        })
    }

    pub(crate) fn interior_loads(&self) -> Result<Vec<InteriorLoad>, SaddleError> {
        Ok(neb_load(self.images.view(), &self.targets, &self.ks, self.tangent)?.interiors)
    }

    pub fn positions(&self) -> Array2<f64> {
        self.images.clone()
    }

    pub fn step(&mut self) -> Result<StencilReport, SaddleError> {
        let n = self.images.nrows();
        let dof = self.images.ncols();
        let width = (n - 2) * dof;
        if self.solver_n != width {
            self.solver = idpp_solver(self.method.clone(), width, self.max_move)?;
            self.solver_n = width;
        }
        let images0 = self.images.clone();
        let targets = self.targets.clone();
        let ks = self.ks.clone();
        let tangent = self.tangent;
        let oracle = Oracle::unbounded(width, move |flat: ArrayView1<f64>| {
            let mut trial = images0.clone();
            if write_interior(&mut trial, flat).is_err() {
                return (f64::INFINITY, Array1::zeros(width));
            }
            match neb_load(trial.view(), &targets, &ks, tangent) {
                Ok(load) => (load.objective, -load.projected),
                Err(_) => (f64::INFINITY, Array1::zeros(width)),
            }
        });
        let mut flat = interior_flat(self.images.view());
        self.solver
            .step(&oracle, &mut flat)
            .map_err(|err| SaddleError::Solver(err.to_string()))?;
        write_interior(&mut self.images, flat.view())?;
        let max_force =
            neb_load(self.images.view(), &self.targets, &self.ks, self.tangent)?.max_component;
        self.max_force = max_force;
        self.iteration += 1;
        let status = if max_force <= self.force_tol {
            StencilStatus::Converged
        } else {
            StencilStatus::Running
        };
        Ok(report(status, max_force, self.iteration))
    }
}

fn check_band(
    images: &Array2<f64>,
    k: f64,
    force_tol: f64,
    max_move: f64,
) -> Result<(), SaddleError> {
    if images.nrows() < 3 {
        return Err(SaddleError::Shape(
            "IDPP needs at least three images".into(),
        ));
    }
    atom_count(images.ncols())?;
    if !(k.is_finite() && k > 0.0) {
        return Err(SaddleError::Invalid(
            "spring constant must be finite and positive".into(),
        ));
    }
    if !(force_tol.is_finite() && force_tol > 0.0) {
        return Err(SaddleError::Invalid(
            "force tolerance must be finite and positive".into(),
        ));
    }
    if !(max_move.is_finite() && max_move > 0.0) {
        return Err(SaddleError::Invalid(
            "max move must be finite and positive".into(),
        ));
    }
    if !images.iter().all(|value| value.is_finite()) {
        return Err(SaddleError::NonFinite("IDPP positions"));
    }
    Ok(())
}

fn idpp_solver(method: Method, n: usize, max_move: f64) -> Result<Solver, SaddleError> {
    crate::band::check_force_driven(&method)?;
    let control = Control {
        maxiter: usize::MAX,
        gtol: 0.0,
        istep: 1.0,
        maxmove: None,
        ftol_rel: None,
    };
    let lbfgs = matches!(method, Method::Lbfgs { .. });
    let mut solver = Solver::new(method, control, n);
    solver.set_atom_maxmove(max_move);
    if lbfgs {
        solver.set_accept(Accept::Step);
    }
    Ok(solver)
}

fn report(status: StencilStatus, max_force: f64, iteration: usize) -> StencilReport {
    StencilReport {
        status,
        max_force,
        curvature: f64::NAN,
        iteration,
        evaluations: 0,
        total_evaluations: 0,
        phase: crate::stencil::PHASE_BAND,
        ci_index: None,
        zoom_window: None,
    }
}

pub(crate) fn interior_flat(images: ArrayView2<f64>) -> Array1<f64> {
    let n = images.nrows();
    let dof = images.ncols();
    let mut flat = Array1::zeros((n - 2) * dof);
    for i in 1..n - 1 {
        let row = i - 1;
        for k in 0..dof {
            flat[row * dof + k] = images[(i, k)];
        }
    }
    flat
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn a_rotation_keeps_a_longer_bond_than_the_linear_path() {
        let images = rotation_images();
        let linear_min = min_bond(&images);
        let mut run = IdppRun::new(
            images,
            0.1,
            1e-6,
            0.2,
            Method::Fire {
                kind: rgmin::FireKind::V2,
            },
        )
        .unwrap();
        for _ in 0..12 {
            run.step().unwrap();
        }
        let positions = run.positions();
        for row in 0..positions.nrows() {
            let (energy, _) = energy_force(positions.row(row), run.targets[row].view()).unwrap();
            assert!(energy.is_finite());
        }
        assert!(min_bond(&positions) > linear_min + 0.15);
        let ends = run.positions();
        assert!((ends[(0, 3)] - 1.0).abs() < 1e-12);
        assert!((ends[(4, 4)] - 1.0).abs() < 1e-12);
    }

    fn rotation_images() -> Array2<f64> {
        let start = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let end = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let n = 5;
        let mut images = Array2::zeros((n, 6));
        for i in 0..n {
            let t = i as f64 / (n - 1) as f64;
            for k in 0..6 {
                images[(i, k)] = (1.0 - t) * start[k] + t * end[k];
            }
        }
        images
    }

    fn min_bond(images: &Array2<f64>) -> f64 {
        let mut best = f64::INFINITY;
        for row in images.outer_iter().skip(1).take(images.nrows() - 2) {
            let dx = row[3] - row[0];
            let dy = row[4] - row[1];
            let dz = row[5] - row[2];
            best = best.min((dx * dx + dy * dy + dz * dz).sqrt());
        }
        best
    }

    #[test]
    fn one_atom_is_refused() {
        let coords = array![0.0, 0.0, 0.0];
        let err = pair_distances(coords.view()).unwrap_err();
        assert!(err.to_string().contains("at least two atoms"));
    }

    #[test]
    fn energy_and_force_match_ase() {
        // ase.mep.neb.IDPP on this geometry, mic=False.
        let coords = array![0.0, 0.0, 0.0, 1.2, 0.1, -0.2];
        let target = array![[0.0, 1.5], [1.5, 0.0]];
        let (energy, force) = energy_force(coords.view(), target.view()).unwrap();
        assert!((energy - 0.03514855874955593).abs() < 1e-12);
        let ase = [
            -0.36062223,
            -0.03005185,
            0.0601037,
            0.36062223,
            0.03005185,
            -0.0601037,
        ];
        for (got, want) in force.iter().zip(ase) {
            assert!((got - want).abs() < 1e-8, "{got} vs {want}");
        }
    }

    #[test]
    fn diagonal_target_does_not_diverge() {
        let coords = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let target = array![[0.0, 1.0], [1.0, 0.0]];
        let (energy, force) = energy_force(coords.view(), target.view()).unwrap();
        assert!(energy.abs() < 1e-12);
        assert!(force.iter().all(|value| value.abs() < 1e-12));
    }
}
