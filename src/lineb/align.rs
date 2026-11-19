//! Mass-weighted quaternion alignment of a finite system.
//!
//! The path length of a molecule is the mass-weighted distance after the
//! optimal rigid motion is removed. The rotation is the Kabsch quaternion:
//! the largest eigenvector of the 4×4 matrix built from the mass-weighted
//! covariance, the same matrix eOn and ASE use for `align_to`, from the
//! quaternion helpers of [`crate::internal`]. The rigid projection of a
//! force is rgmin's, in the mass-weighted chart.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rgmin::{RigidMode, RigidRequest, RigidWorkspace};

use crate::error::SaddleError;
use crate::internal::{jacobi4, kabsch_f};

/// Rigid placement that takes one image onto the reference orientation.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Placement {
    /// Maps centered mass-weighted coordinates of this image onto the
    /// reference orientation.
    pub rotation: [[f64; 3]; 3],
}

/// Align every image to image 0. `images` is mass-weighted, one row per
/// image, length `3 * atoms`. The three coordinate masses of an atom are
/// one atomic mass.
pub(crate) fn align_images(
    images: ArrayView2<f64>,
    sqrt_mass: ArrayView1<f64>,
    atoms: usize,
) -> Result<(Array2<f64>, Vec<Placement>), SaddleError> {
    if atoms < 2 {
        return Err(SaddleError::Invalid(
            "quaternion alignment needs at least two atoms".into(),
        ));
    }
    let dof = 3 * atoms;
    if images.ncols() != dof || sqrt_mass.len() != dof {
        return Err(SaddleError::Shape(
            "alignment expects 3 coordinates and one mass weight per atom".into(),
        ));
    }
    check_atom_masses(sqrt_mass, atoms)?;
    let n = images.nrows();
    if n == 0 {
        return Err(SaddleError::Shape(
            "alignment needs at least one image".into(),
        ));
    }
    let reference = images.row(0).to_owned();
    let mut aligned = Array2::zeros((n, dof));
    let mut placements = Vec::with_capacity(n);
    for i in 0..n {
        let (row, placement) =
            align_to_reference(images.row(i), reference.view(), sqrt_mass, atoms)?;
        aligned.row_mut(i).assign(&row);
        placements.push(placement);
    }
    Ok((aligned, placements))
}

/// Rotate a mass-weighted force back to the stored orientation and remove
/// the rigid motions the alignment quotient does not feel. `aligned` is
/// the centred mass-weighted geometry the force was computed on.
pub(crate) fn restore_force(
    force: &mut Array1<f64>,
    placement: &Placement,
    aligned: ArrayView1<f64>,
    sqrt_mass: ArrayView1<f64>,
    atoms: usize,
) {
    let masses: Vec<f64> = (0..atoms)
        .map(|atom| sqrt_mass[3 * atom] * sqrt_mass[3 * atom])
        .collect();
    if let (Some(positions), Some(values)) = (aligned.as_slice(), force.as_slice_mut()) {
        let mut request = RigidRequest::cartesian(positions, RigidMode::TranslationRotation);
        request.masses = Some(&masses);
        request.mass_weighted_coords = true;
        RigidWorkspace::new().project(values, &request);
    }
    let mut rotated = Array1::zeros(force.len());
    for atom in 0..atoms {
        let f = [force[3 * atom], force[3 * atom + 1], force[3 * atom + 2]];
        for row in 0..3 {
            let mut acc = 0.0;
            for col in 0..3 {
                acc += placement.rotation[col][row] * f[col];
            }
            rotated[3 * atom + row] = acc;
        }
    }
    *force = rotated;
}

fn align_to_reference(
    image: ArrayView1<f64>,
    reference: ArrayView1<f64>,
    sqrt_mass: ArrayView1<f64>,
    atoms: usize,
) -> Result<(Array1<f64>, Placement), SaddleError> {
    let com = centre_of_mass(image, sqrt_mass, atoms)?;
    let com_ref = centre_of_mass(reference, sqrt_mass, atoms)?;
    let centered = center(image, sqrt_mass, com);
    let centered_ref = center(reference, sqrt_mass, com_ref);
    let rotation = kabsch(&centered, &centered_ref, atoms)?;
    let mut out = Array1::zeros(image.len());
    for atom in 0..atoms {
        let p = [
            centered[3 * atom],
            centered[3 * atom + 1],
            centered[3 * atom + 2],
        ];
        for row in 0..3 {
            let mut acc = 0.0;
            for col in 0..3 {
                acc += rotation[row][col] * p[col];
            }
            out[3 * atom + row] = acc;
        }
    }
    Ok((out, Placement { rotation }))
}

fn check_atom_masses(sqrt_mass: ArrayView1<f64>, atoms: usize) -> Result<(), SaddleError> {
    for atom in 0..atoms {
        let m = sqrt_mass[3 * atom];
        if !(m.is_finite() && m > 0.0) {
            return Err(SaddleError::Invalid(
                "atomic masses must be finite and positive".into(),
            ));
        }
        for axis in 1..3 {
            let other = sqrt_mass[3 * atom + axis];
            if (other - m).abs() > 0.0 {
                return Err(SaddleError::Invalid(
                    "the three coordinate masses of an atom are one atomic mass".into(),
                ));
            }
        }
    }
    Ok(())
}

fn centre_of_mass(
    q: ArrayView1<f64>,
    sqrt_mass: ArrayView1<f64>,
    atoms: usize,
) -> Result<[f64; 3], SaddleError> {
    let mut mass = 0.0;
    let mut moment = [0.0; 3];
    for atom in 0..atoms {
        let root = sqrt_mass[3 * atom];
        let m = root * root;
        mass += m;
        for axis in 0..3 {
            moment[axis] += root * q[3 * atom + axis];
        }
    }
    if mass <= 0.0 || mass.is_nan() {
        return Err(SaddleError::Invalid(
            "a molecule needs a positive total mass".into(),
        ));
    }
    Ok([moment[0] / mass, moment[1] / mass, moment[2] / mass])
}

/// `sqrt(m) (x - com)`, which is `q - sqrt(m) com`.
fn center(q: ArrayView1<f64>, sqrt_mass: ArrayView1<f64>, com: [f64; 3]) -> Array1<f64> {
    let mut out = q.to_owned();
    for i in 0..q.len() {
        out[i] -= sqrt_mass[i] * com[i % 3];
    }
    out
}

fn kabsch(
    source: &Array1<f64>,
    target: &Array1<f64>,
    atoms: usize,
) -> Result<[[f64; 3]; 3], SaddleError> {
    let rows =
        |q: &Array1<f64>| Array2::from_shape_fn((atoms, 3), |(atom, axis)| q[3 * atom + axis]);
    let (values, vectors) = jacobi4(kabsch_f(&rows(source), &rows(target)));
    if !values.iter().all(|value| value.is_finite()) {
        return Err(SaddleError::NonFinite("quaternion spectrum"));
    }
    let mut best = 0;
    for k in 1..4 {
        if values[k] > values[best] {
            best = k;
        }
    }
    let mut q = [
        vectors[0][best],
        vectors[1][best],
        vectors[2][best],
        vectors[3][best],
    ];
    if q[0] < 0.0 {
        for component in &mut q {
            *component = -*component;
        }
    }
    let (q0, q1, q2, q3) = (q[0], q[1], q[2], q[3]);
    Ok([
        [
            q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3,
            2.0 * (q1 * q2 - q0 * q3),
            2.0 * (q1 * q3 + q0 * q2),
        ],
        [
            2.0 * (q1 * q2 + q0 * q3),
            q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3,
            2.0 * (q2 * q3 - q0 * q1),
        ],
        [
            2.0 * (q1 * q3 - q0 * q2),
            2.0 * (q2 * q3 + q0 * q1),
            q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3,
        ],
    ])
}

/// Orthonormal translations and rotations at one mass-weighted geometry.
/// The rotations are about the origin; with the translations they span
/// the rigid motions about any centre. A mode with no length (a linear
/// molecule about its axis) is absent.
pub(crate) fn rigid_directions(
    positions: ArrayView1<f64>,
    sqrt_mass: ArrayView1<f64>,
    atoms: usize,
) -> Vec<Array1<f64>> {
    let mut modes = Vec::with_capacity(6);
    for axis in 0..3 {
        let mut mode = Array1::zeros(positions.len());
        for atom in 0..atoms {
            mode[3 * atom + axis] = sqrt_mass[3 * atom];
        }
        push_mode(&mut modes, mode);
    }
    for axis in 0..3 {
        let mut mode = Array1::zeros(positions.len());
        for atom in 0..atoms {
            let p = [
                positions[3 * atom],
                positions[3 * atom + 1],
                positions[3 * atom + 2],
            ];
            // q = sqrt(m) x, so the mass-weighted rotation mode is ω × q.
            let crossed = cross(axis, p);
            for component in 0..3 {
                mode[3 * atom + component] = crossed[component];
            }
        }
        push_mode(&mut modes, mode);
    }
    modes
}

fn push_mode(modes: &mut Vec<Array1<f64>>, mut mode: Array1<f64>) {
    for earlier in modes.iter() {
        let coeff = earlier.dot(&mode);
        mode.scaled_add(-coeff, earlier);
    }
    let norm = mode.dot(&mode).sqrt();
    if norm > 0.0 {
        mode /= norm;
        modes.push(mode);
    }
}

fn cross(axis: usize, p: [f64; 3]) -> [f64; 3] {
    match axis {
        0 => [0.0, -p[2], p[1]],
        1 => [p[2], 0.0, -p[0]],
        _ => [-p[1], p[0], 0.0],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn a_right_angle_returns_to_the_reference() {
        let sqrt_mass = array![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let reference = array![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        // 90 degrees about z, then a translation of (4, -2, 7).
        let moved = array![4.0, -1.0, 7.0, 3.0, -2.0, 7.0, 4.0, -2.0, 8.0];
        let images = ndarray::stack(ndarray::Axis(0), &[reference.view(), moved.view()]).unwrap();
        let (aligned, _) = align_images(images.view(), sqrt_mass.view(), 3).unwrap();
        let gap = &aligned.row(1) - &aligned.row(0);
        let scale = aligned.row(0).dot(&aligned.row(0)).sqrt();
        assert!(
            gap.dot(&gap).sqrt() <= scale * f64::EPSILON * 64.0,
            "{}",
            gap.dot(&gap).sqrt()
        );
    }

    /// The restored force has no component along a rigid motion of the
    /// mass-weighted geometry it was computed on.
    #[test]
    fn restored_force_has_no_rigid_component() {
        let masses = [1.0_f64, 16.0, 2.0, 12.0];
        let sqrt_mass = Array1::from_iter((0..12).map(|i| masses[i / 3].sqrt()));
        let aligned = Array1::from_iter((0..12).map(|i| ((i as f64) * 0.7).sin() * sqrt_mass[i]));
        let mut force = Array1::from_iter((0..12).map(|i| ((i as f64) * 1.3).cos()));
        let identity = Placement {
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        };
        let scale = force.dot(&force).sqrt();
        restore_force(&mut force, &identity, aligned.view(), sqrt_mass.view(), 4);
        for mode in rigid_directions(aligned.view(), sqrt_mass.view(), 4) {
            assert!(
                mode.dot(&force).abs() <= scale * f64::EPSILON * 64.0,
                "{}",
                mode.dot(&force)
            );
        }
    }
}
