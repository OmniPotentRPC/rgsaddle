//! Discrete abbreviated action and the nudged force at fixed energy.
//!
//! Ásgeirsson, Arnaldsson and Jónsson, J. Chem. Phys. 148, 102334 (2018),
//! and the discretization written out by Zhang, Nimmrich, Johnson,
//! Schenter, Khalil, and Govind, J. Chem. Theory Comput. 21, 7517 (2025)
//! (the LI-NEB appendix of doi:10.1021/acs.jctc.5c00673).
//! In mass-weighted coordinates the one-way action is
//!
//! `W = Σ_j (σ_{j−1} + σ_j) / 2 · |q_j − q_{j−1}|`,
//!
//! `σ = √(2(V − E))`. The factor `1/ℏ` scales `W` but not its minimizer,
//! so the objective stored here is this integral. The negative gradient
//! at an interior image, with `ξ = σ` (`ℏ = 1` in the objective), is
//!
//! `g = ½ [(d_prev + d_next) σ' f − (σ + σ_prev) τ_prev + (σ_next + σ) τ_next]`,
//!
//! `σ' = dσ/dV`. The optimizer uses the regularized momentum
//!
//! `σ_w = √(Δ + √(Δ² + w²))`, `Δ = V − E`,
//!
//! which is `√(2Δ)` to a relative `w²/(8Δ²)` above the turning surface,
//! stays positive through it, and keeps `σ'` below `1/√w`. The endpoints
//! cross `V = E` freely. An interior image below `E + w` is outside the
//! feasible set, and [`assemble`] reports that as `Ok(None)`.
//!
//! The nudged force keeps the part of `g` perpendicular to the tangent
//! and adds the parallel spring `k (|q_next − q| − |q − q_prev|) τ`.
//! An endpoint keeps only the part of its spring orthogonal to the
//! physical force and is pulled onto `V = E` by `κ (V − E) f̂`, with
//! either fixed `κ` or fixed `κ |f|` according to the endpoint restraint.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use super::align::{self, Placement};
use super::mass::Masses;
use crate::error::SaddleError;
use crate::minmode::PointSurface;
use crate::projection::{ProjectionKind, force_perp};
use crate::spring::SpringKind;
use crate::tangent::{TangentKind, compute_tangent};

#[derive(Clone, Copy, Debug)]
pub(crate) enum EndpointRestraint {
    Normalized(f64),
    Fixed(f64),
}

impl EndpointRestraint {
    pub(crate) fn coefficient(self, gradient_norm: f64) -> f64 {
        match self {
            Self::Normalized(scale) => scale / gradient_norm,
            Self::Fixed(kappa) => kappa,
        }
    }

    pub(crate) fn is_valid(self) -> bool {
        let coefficient = self.coefficient(1.0);
        coefficient.is_finite() && coefficient > 0.0
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Assembly {
    /// Mass-weighted images after the rigid alignment, the geometry of
    /// the action.
    pub aligned: Array2<f64>,
    /// Nudged force in the solver's mass-weighted frame. The solver
    /// steps along this vector.
    pub drive: Array2<f64>,
    pub action: f64,
    pub perp_max: f64,
    pub opt_max: f64,
    pub endpoint_max: f64,
}

pub(crate) fn assemble<S: PointSurface>(
    surface: &S,
    masses: &Masses,
    weighted: ArrayView2<f64>,
    energy: f64,
    spring: f64,
    restraint: EndpointRestraint,
    width: f64,
) -> Result<Option<Assembly>, SaddleError> {
    let n = weighted.nrows();
    let dof = weighted.ncols();
    if n < 3 || dof != masses.dof() {
        return Err(SaddleError::Shape(
            "the action needs at least three images and one mass per coordinate".into(),
        ));
    }
    if !(energy.is_finite()
        && spring.is_finite()
        && spring > 0.0
        && restraint.is_valid()
        && width.is_finite()
        && width > 0.0)
    {
        return Err(SaddleError::Invalid(
            "tunneling energy, spring, endpoint scale and turning width must be finite, and the scales positive"
                .into(),
        ));
    }
    let (aligned, placements) = match masses.atoms {
        Some(atoms) => {
            let (aligned, placements) =
                align::align_images(weighted, masses.sqrt_mass.view(), atoms, masses.adjacent)?;
            (aligned, Some(placements))
        }
        None => (weighted.to_owned(), None),
    };

    let mut energies = Array1::zeros(n);
    let mut force = Array2::zeros((n, dof));
    // The surface sees the stored Cartesian geometries. Alignment is a
    // gauge: the energy matches, and the force is rotated below. The images
    // are independent, so they go to the surface as one batch and a
    // provider may evaluate them concurrently.
    let mut cartesian = Array2::zeros((n, dof));
    for i in 0..n {
        cartesian
            .row_mut(i)
            .assign(&masses.to_cartesian(weighted.row(i))?);
    }
    let observations = surface.eval_batch(cartesian.view())?;
    if observations.len() != n {
        return Err(SaddleError::Shape(
            "the surface must return one observation per image".into(),
        ));
    }
    for (i, (v, gradient)) in observations.into_iter().enumerate() {
        if !v.is_finite() || gradient.iter().any(|g| !g.is_finite()) {
            return Err(SaddleError::NonFinite("tunneling image"));
        }
        energies[i] = v;
        let mut f = masses.weighted_force(gradient.view())?;
        if let Some(placements) = &placements {
            rotate_into(&mut f, &placements[i]);
        }
        force.row_mut(i).assign(&f);
    }

    if energies
        .iter()
        .skip(1)
        .take(n - 2)
        .any(|&v| v - energy < width)
    {
        return Ok(None);
    }
    let (sigma, slope): (Vec<f64>, Vec<f64>) = energies
        .iter()
        .map(|&v| momentum(v - energy, width))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .unzip();
    let mut dist = Vec::with_capacity(n - 1);
    let mut step = Vec::with_capacity(n - 1);
    let mut unit = Vec::with_capacity(n - 1);
    for j in 0..n - 1 {
        let mut diff = aligned.row(j + 1).to_owned();
        diff -= &aligned.row(j);
        let length = l2(diff.view());
        if length <= 0.0 || length.is_nan() {
            return Err(SaddleError::Invalid(
                "two tunneling images coincide, so the path has no tangent".into(),
            ));
        }
        let mut direction = diff.clone();
        direction /= length;
        dist.push(length);
        step.push(diff);
        unit.push(direction);
    }

    let mut action = 0.0;
    for j in 0..n - 1 {
        action += 0.5 * (sigma[j] + sigma[j + 1]) * dist[j];
    }
    if !action.is_finite() {
        return Err(SaddleError::NonFinite("abbreviated action"));
    }

    let mut drive_aligned = Array2::zeros((n, dof));
    let mut perp_max = 0.0_f64;
    let mut opt_max = 0.0_f64;
    for j in 1..n - 1 {
        let mut g = Array1::zeros(dof);
        let both = dist[j - 1] + dist[j];
        g.scaled_add(0.5 * both * slope[j], &force.row(j));
        g.scaled_add(-0.5 * (sigma[j] + sigma[j - 1]), &unit[j - 1]);
        g.scaled_add(0.5 * (sigma[j] + sigma[j + 1]), &unit[j]);
        let tangent = compute_tangent(
            TangentKind::Improved,
            step[j].view(),
            step[j - 1].view(),
            energies[j],
            energies[j - 1],
            energies[j + 1],
        )?;
        perp_max = perp_max.max(l2(force_perp(g.view(), tangent.view()).view()));
        let springs = SpringKind::Uniform { k: spring }.compute(
            j,
            tangent.view(),
            dist[j],
            dist[j - 1],
            step[j].view(),
            step[j - 1].view(),
        )?;
        let nudged = ProjectionKind::Neb.project(g.view(), tangent.view(), &springs);
        opt_max = opt_max.max(l2(nudged.view()));
        drive_aligned.row_mut(j).assign(&nudged);
    }
    let mut endpoint_max = 0.0_f64;
    for (end, neighbour) in [(0, 1), (n - 1, n - 2)] {
        // Equations (20) and (22) use the full adjacent chord. Only
        // interior images have springs pulling from both sides.
        let mut spring_force = (&aligned.row(neighbour) - &aligned.row(end)) * spring;
        let f_norm = l2(force.row(end));
        if f_norm <= 0.0 || f_norm.is_nan() {
            return Err(SaddleError::Invalid(
                "an endpoint force vanishes, so the equipotential has no direction".into(),
            ));
        }
        let mut f_hat = force.row(end).to_owned();
        f_hat /= f_norm;
        let parallel = f_hat.dot(&spring_force);
        spring_force.scaled_add(-parallel, &f_hat);
        let kappa = restraint.coefficient(f_norm);
        spring_force.scaled_add(kappa * (energies[end] - energy), &f_hat);
        endpoint_max = endpoint_max.max(l2(spring_force.view()));
        drive_aligned.row_mut(end).assign(&spring_force);
    }

    let mut drive = drive_aligned.clone();
    if let (Some(atoms), Some(placements)) = (masses.atoms, &placements) {
        for (i, placement) in placements.iter().enumerate() {
            let mut row = drive.row(i).to_owned();
            align::restore_force(
                &mut row,
                placement,
                aligned.row(i),
                masses.sqrt_mass.view(),
                atoms,
            );
            drive.row_mut(i).assign(&row);
        }
    }

    Ok(Some(Assembly {
        aligned,
        drive,
        action,
        perp_max,
        opt_max: opt_max.max(endpoint_max),
        endpoint_max,
    }))
}

/// `σ_w(Δ)` and `dσ_w/dΔ`.
fn momentum(gap: f64, width: f64) -> Result<(f64, f64), SaddleError> {
    let root = gap.hypot(width);
    // `Δ + √(Δ² + w²)` cancels for `Δ < 0`; `w² / (√(Δ² + w²) − Δ)` does not.
    let inner = if gap >= 0.0 {
        gap + root
    } else {
        width * width / (root - gap)
    };
    let sigma = inner.sqrt();
    let slope = inner / (root * 2.0 * sigma);
    if !(sigma.is_finite() && slope.is_finite() && sigma > 0.0) {
        return Err(SaddleError::NonFinite("tunneling momentum"));
    }
    Ok((sigma, slope))
}

/// `R` maps the stored centred frame onto the aligned frame, so a
/// vector transforms by `R`.
fn rotate_into(force: &mut Array1<f64>, placement: &Placement) {
    let atoms = force.len() / 3;
    let mut rotated = Array1::zeros(force.len());
    for atom in 0..atoms {
        let f = [force[3 * atom], force[3 * atom + 1], force[3 * atom + 2]];
        for row in 0..3 {
            let mut acc = 0.0;
            for (col, component) in f.iter().enumerate() {
                acc += placement.rotation[row][col] * component;
            }
            rotated[3 * atom + row] = acc;
        }
    }
    *force = rotated;
}

pub(crate) fn l2(v: ArrayView1<f64>) -> f64 {
    v.dot(&v).sqrt()
}

/// Flat mass-weighted coordinates, image by image.
pub(crate) fn pack(images: ArrayView2<f64>) -> Array1<f64> {
    Array1::from_iter(images.iter().copied())
}

pub(crate) fn unpack(flat: ArrayView1<f64>, n: usize, dof: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, dof), |(i, k)| flat[i * dof + k])
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

    #[test]
    fn equally_spaced_endpoints_keep_the_projected_neighbour_spring() {
        let images = ndarray::array![[-1.0, 0.0], [0.0, 0.25], [1.0, 0.0]];
        let masses = Masses::coordinates(Array1::ones(2).view()).unwrap();
        let energy = Valley.eval(images.row(0)).unwrap().0;
        let spring = 2.0;
        let band = assemble(
            &Valley,
            &masses,
            images.view(),
            energy,
            spring,
            EndpointRestraint::Normalized(1.0),
            0.01,
        )
        .unwrap()
        .unwrap();
        for end in [0, 2] {
            // The surface normal points along x. Equations (20) and (22)
            // therefore retain the full y component of the adjacent chord.
            assert!((band.drive[[end, 1]] - spring * 0.25).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn molecular_action_and_drive_are_invariant_to_reversing_the_band() {
        struct HarmonicBond;
        impl PointSurface for HarmonicBond {
            fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
                let bond = &x.slice(ndarray::s![3..6]) - &x.slice(ndarray::s![..3]);
                let mut gradient = Array1::zeros(x.len());
                gradient.slice_mut(ndarray::s![..3]).assign(&(-&bond));
                gradient.slice_mut(ndarray::s![3..6]).assign(&bond);
                Ok((0.5 * bond.dot(&bond), gradient))
            }
        }
        let masses = Masses::molecule(&[12.0, 16.0, 1.0, 2.0])
            .unwrap()
            .with_adjacent_alignment(true);
        // Bond extension, bending and torsion make the best rotation depend on the pair.
        let mut images = Array2::zeros((5, 12));
        for (i, mut row) in images.rows_mut().into_iter().enumerate() {
            let t = i as f64 / 4.0;
            let xyz = ndarray::array![
                0.,
                0.,
                0.,
                1. + t,
                0.,
                0.,
                t,
                1. + t * t,
                0.,
                1. - t,
                t * t,
                1. + t
            ];
            row.assign(&masses.to_weighted(xyz.view()).unwrap());
        }
        let evaluate = |band| {
            assemble(
                &HarmonicBond,
                &masses,
                band,
                0.25,
                1.0,
                EndpointRestraint::Fixed(1.0),
                0.01,
            )
            .unwrap()
            .unwrap()
        };
        let forward = evaluate(images.view());
        let reverse = evaluate(images.slice(ndarray::s![..;-1, ..]));
        assert!(
            (forward.action - reverse.action).abs() < 1e-12,
            "forward {}, reverse {}",
            forward.action,
            reverse.action
        );
        for (a, b) in forward
            .drive
            .iter()
            .zip(reverse.drive.slice(ndarray::s![..;-1, ..]).iter())
        {
            assert!((a - b).abs() < 1e-12, "forward {a}, reverse {b}");
        }
    }

    #[test]
    fn fixed_endpoint_force_tracks_energy_error_at_both_ends() {
        let images = ndarray::array![[-1.0, 0.0], [0.0, 0.0], [1.5, 0.0]];
        let masses = Masses::coordinates(Array1::ones(2).view()).unwrap();
        let target = 0.25;
        let kappa = 2.0;
        let band = assemble(
            &Valley,
            &masses,
            images.view(),
            target,
            0.1,
            EndpointRestraint::Fixed(kappa),
            0.01,
        )
        .unwrap()
        .unwrap();
        for end in [0, 2] {
            let (energy, gradient) = Valley.eval(images.row(end)).unwrap();
            let expected = -gradient[0].signum() * kappa * (energy - target);
            assert!((band.drive[[end, 0]] - expected).abs() < 1e-14);
        }
    }

    /// Eckart barrier in `x` and a harmonic valley in `y`.
    struct Valley;

    impl PointSurface for Valley {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let s = 1.0 / x[0].cosh();
            let mut force = Array1::zeros(2);
            force[0] = -2.0 * s * s * x[0].tanh();
            force[1] = 4.0 * x[1];
            Ok((s * s + 2.0 * x[1] * x[1], force))
        }
    }

    /// Off the path, the drive is the negative action gradient. The
    /// spring lies along the tangent, so it does not enter this component.
    #[test]
    fn transverse_drive_matches_a_central_difference_of_the_action() {
        let masses = Masses::coordinates(Array1::ones(2).view()).unwrap();
        let images =
            Array2::from_shape_fn(
                (5, 2),
                |(i, k)| {
                    if k == 0 { -0.8 + 0.4 * i as f64 } else { 0.05 }
                },
            );
        let energy = 0.25;
        let spring = 0.1;
        let endpoint = 0.2;
        let width = 1e-3;
        let assemble_at = |images: ArrayView2<f64>| {
            assemble(
                &Valley,
                &masses,
                images,
                energy,
                spring,
                EndpointRestraint::Normalized(endpoint),
                width,
            )
            .unwrap()
            .unwrap()
        };
        let assembly = assemble_at(images.view());
        for image in 1..images.nrows() - 1 {
            // The probe length is the transverse coordinate itself.
            let delta = images[(image, 1)].abs() * f64::EPSILON.sqrt();
            let mut plus = images.clone();
            let mut minus = images.clone();
            plus[(image, 1)] += delta;
            minus[(image, 1)] -= delta;
            let up = assemble_at(plus.view()).action;
            let down = assemble_at(minus.view()).action;
            let slope = (up - down) / (2.0 * delta);
            let predicted = -assembly.drive[(image, 1)];
            // Relative to the larger of the two slopes. When both vanish,
            // the absolute measure is the action change across the probe.
            let mut scale = predicted.abs().max(slope.abs());
            if scale <= 0.0 || scale.is_nan() {
                scale = (up - down).abs();
            }
            assert!(
                (slope - predicted).abs() <= scale * delta.sqrt(),
                "image {image}: difference {slope} vs drive {predicted}"
            );
        }
    }
}
