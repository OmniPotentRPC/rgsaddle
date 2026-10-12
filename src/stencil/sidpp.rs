//! Sequential image dependent pair potential.
//!
//! Schmerwitz, Ásgeirsson, and Jónsson, arXiv:2310.04531. Images are
//! introduced from the endpoints. The long central gap uses a spring
//! scaled by the ideal spacing over the gap length. Once every
//! requested image exists, equal springs refine the path. The tangent
//! normalises each chord before the weighted average.

use ndarray::{Array1, Array2, ArrayView2};

use crate::error::SaddleError;
use rgmin::Method;

use super::idpp::{
    self, IdppRun, InteriorLoad, TangentStyle, energy_force, pair_distances, tangent_of,
};

const INNER_CAP: usize = 40;
const FMAX: f64 = 0.01;
const RMS: f64 = 0.005;

struct Build {
    d_r: Array2<f64>,
    d_p: Array2<f64>,
    n_req: usize,
    k: f64,
    max_move: f64,
    method: Method,
}

/// Build the sequential path, then hand back an equal-spring relaxation.
pub fn prepare(
    images: Array2<f64>,
    k: f64,
    force_tol: f64,
    max_move: f64,
    method: Method,
) -> Result<IdppRun, SaddleError> {
    let built = construct(&images, k, max_move, method.clone())?;
    let targets = idpp::interpolated_targets(built.view())?;
    let ks = vec![k; built.nrows() - 1];
    IdppRun::from_targets(
        built,
        targets,
        ks,
        force_tol,
        max_move,
        method,
        TangentStyle::Prenormalized,
    )
}

fn construct(
    images: &Array2<f64>,
    k: f64,
    max_move: f64,
    method: Method,
) -> Result<Array2<f64>, SaddleError> {
    idpp::atom_count(images.ncols())?;
    let n_req = images.nrows();
    if n_req < 3 {
        return Err(SaddleError::Shape(
            "S-IDPP needs at least three images".into(),
        ));
    }
    if !(k.is_finite() && k > 0.0) {
        return Err(SaddleError::Invalid(
            "spring constant must be finite and positive".into(),
        ));
    }
    let reactant = images.row(0).to_owned();
    let product = images.row(n_req - 1).to_owned();
    let chord = &product - &reactant;
    let chord_len = chord.dot(&chord).sqrt();
    if !(chord_len.is_finite() && chord_len > 1e-15) {
        return Err(SaddleError::Invalid("S-IDPP endpoints coincide".into()));
    }
    let hat = &chord / chord_len;
    let d0 = chord_len / (n_req - 1) as f64;
    let build = Build {
        d_r: pair_distances(reactant.view())?,
        d_p: pair_distances(product.view())?,
        n_req,
        k,
        max_move,
        method,
    };
    let mut left = vec![reactant.clone()];
    let mut right = vec![product.clone()];
    left.push(&reactant + &hat * d0);
    if n_req > 3 {
        right.push(&product - &hat * d0);
    }
    for _ in 0..n_req {
        let (left_load, right_load) = relax(&mut left, &mut right, &build)?;
        if left.len() + right.len() >= n_req {
            break;
        }
        add_one(&mut left, &mut right, &build, left_load, right_load)?;
    }
    if left.len() + right.len() != n_req {
        return Err(SaddleError::Invalid(
            "S-IDPP did not place every image".into(),
        ));
    }
    Ok(stack(&left, &right))
}

fn relax(
    left: &mut [Array1<f64>],
    right: &mut [Array1<f64>],
    build: &Build,
) -> Result<(Option<InteriorLoad>, Option<InteriorLoad>), SaddleError> {
    let path = stack(left, right);
    let targets = targets_for(path.view(), left.len(), build);
    let d_id = path_length(path.view()) / (build.n_req - 1) as f64;
    let ks = springs(path.view(), left.len(), build.k, d_id);
    let mut run = IdppRun::from_targets(
        path,
        targets,
        ks,
        FMAX,
        build.max_move,
        build.method.clone(),
        TangentStyle::Prenormalized,
    )?;
    for _ in 0..INNER_CAP {
        let loads = run.interior_loads()?;
        if frontier_done(&loads, left.len(), run.positions().nrows()) {
            break;
        }
        run.step()?;
    }
    let positions = run.positions();
    let loads = run.interior_loads()?;
    write_back(&positions, left, right);
    Ok((
        frontier_load(&loads, left.len().saturating_sub(1), positions.nrows()),
        frontier_load(&loads, left.len(), positions.nrows()),
    ))
}

fn add_one(
    left: &mut Vec<Array1<f64>>,
    right: &mut Vec<Array1<f64>>,
    build: &Build,
    left_load: Option<InteriorLoad>,
    right_load: Option<InteriorLoad>,
) -> Result<(), SaddleError> {
    let next_left = left.len();
    let next_right = build.n_req - 1 - right.len();
    if next_left > next_right {
        return Err(SaddleError::Invalid("S-IDPP image slots overlap".into()));
    }
    let prefer_left = match (left_load, right_load) {
        (Some(left_side), Some(right_side)) => {
            let left_ok = left_side.max_abs <= FMAX && left_side.rms <= RMS;
            let right_ok = right_side.max_abs <= FMAX && right_side.rms <= RMS;
            if left_ok != right_ok {
                left_ok
            } else {
                left_side.max_abs < right_side.max_abs
                    || (left_side.max_abs == right_side.max_abs && left_side.rms <= right_side.rms)
            }
        }
        (None, Some(_)) => false,
        _ => true,
    };
    let grow_left = if next_left == next_right {
        prefer_left || right_load.is_none()
    } else {
        prefer_left
    };
    let path = stack(left, right);
    let targets = targets_for(path.view(), left.len(), build);
    let d_id = path_length(path.view()) / (build.n_req - 1) as f64;
    if grow_left {
        let dir = frontier_direction(&path, &targets, left.len() - 1)?;
        let here = left.last().expect("reactant").clone();
        left.push(here + dir * d_id);
    } else {
        let dir = frontier_direction(&path, &targets, left.len())?;
        let here = right.last().expect("product").clone();
        right.push(here - dir * d_id);
    }
    Ok(())
}

/// Unit tangent at `index`, pointing from the reactant toward the product.
fn frontier_direction(
    path: &Array2<f64>,
    targets: &[Array2<f64>],
    index: usize,
) -> Result<Array1<f64>, SaddleError> {
    let n = path.nrows();
    if index == 0 || index + 1 >= n {
        let end = path.row(n - 1).to_owned() - path.row(0);
        return unit(end);
    }
    let energies = image_energies(path, targets)?;
    let diff_next = &path.row(index + 1) - &path.row(index);
    let diff_prev = &path.row(index) - &path.row(index - 1);
    let tang = tangent_of(
        TangentStyle::Prenormalized,
        diff_next.view(),
        diff_prev.view(),
        energies[index],
        energies[index - 1],
        energies[index + 1],
    )?;
    unit(tang)
}

fn image_energies(path: &Array2<f64>, targets: &[Array2<f64>]) -> Result<Vec<f64>, SaddleError> {
    let mut energies = Vec::with_capacity(path.nrows());
    for (row, target) in path.rows().into_iter().zip(targets.iter()) {
        let (energy, _) = energy_force(row, target.view())?;
        energies.push(energy);
    }
    Ok(energies)
}

fn springs(path: ArrayView2<f64>, left_len: usize, k: f64, d_id: f64) -> Vec<f64> {
    let n = path.nrows();
    let mut ks = vec![k; n.saturating_sub(1)];
    if left_len > 1 && n > left_len + 1 {
        let gap_at = left_len - 1;
        let delta = &path.row(gap_at + 1) - &path.row(gap_at);
        let gap = delta.dot(&delta).sqrt();
        if gap > 1e-15 && d_id.is_finite() && d_id > 0.0 {
            ks[gap_at] = (d_id / gap) * k;
        }
    }
    ks
}

fn targets_for(path: ArrayView2<f64>, left_len: usize, build: &Build) -> Vec<Array2<f64>> {
    let n = path.nrows();
    let right_len = n - left_len;
    let denom = (build.n_req - 1) as f64;
    let mut targets = Vec::with_capacity(n);
    for j in 0..n {
        let slot = if j < left_len {
            j
        } else {
            let offset = j - left_len;
            let right_index = right_len - 1 - offset;
            build.n_req - 1 - right_index
        };
        let t = slot as f64 / denom;
        targets.push(&build.d_r + &(&build.d_p - &build.d_r) * t);
    }
    targets
}

fn frontier_done(loads: &[InteriorLoad], left_len: usize, n: usize) -> bool {
    let left = frontier_load(loads, left_len.saturating_sub(1), n);
    let right = frontier_load(loads, left_len, n);
    let ready = |load: InteriorLoad| load.max_abs <= FMAX && load.rms <= RMS;
    match (left, right) {
        (Some(load), Some(other)) => ready(load) || ready(other),
        (Some(load), None) | (None, Some(load)) => ready(load),
        (None, None) => true,
    }
}

fn frontier_load(loads: &[InteriorLoad], path_index: usize, n: usize) -> Option<InteriorLoad> {
    if path_index == 0 || path_index + 1 >= n {
        return None;
    }
    loads.get(path_index - 1).copied()
}

fn stack(left: &[Array1<f64>], right: &[Array1<f64>]) -> Array2<f64> {
    let n = left.len() + right.len();
    let dof = left.first().map(|row| row.len()).unwrap_or(0);
    let mut out = Array2::zeros((n, dof));
    for (i, image) in left.iter().enumerate() {
        out.row_mut(i).assign(image);
    }
    for (offset, image) in right.iter().rev().enumerate() {
        out.row_mut(left.len() + offset).assign(image);
    }
    out
}

fn write_back(positions: &Array2<f64>, left: &mut [Array1<f64>], right: &mut [Array1<f64>]) {
    let left_len = left.len();
    for (i, image) in left.iter_mut().enumerate() {
        image.assign(&positions.row(i));
    }
    let right_len = right.len();
    for (index, image) in right.iter_mut().enumerate() {
        let path_index = left_len + (right_len - 1 - index);
        image.assign(&positions.row(path_index));
    }
}

fn path_length(path: ArrayView2<f64>) -> f64 {
    let mut total = 0.0;
    for i in 1..path.nrows() {
        let delta = &path.row(i) - &path.row(i - 1);
        total += delta.dot(&delta).sqrt();
    }
    total
}

fn unit(v: Array1<f64>) -> Result<Array1<f64>, SaddleError> {
    let norm = v.dot(&v).sqrt();
    if norm > 1e-15 && norm.is_finite() {
        Ok(v / norm)
    } else {
        Err(SaddleError::Invalid(
            "S-IDPP has no direction for the next image".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_rotation_starts_with_a_longer_bond_than_the_chord() {
        let images = rotation();
        let linear = min_bond(&images);
        let built = construct(
            &images,
            0.1,
            0.2,
            Method::Fire {
                kind: rgmin::FireKind::V2,
            },
        )
        .unwrap();
        assert!(min_bond(&built) > linear + 0.15);
        assert!((built[(0, 3)] - 1.0).abs() < 1e-12);
        assert!((built[(4, 4)] - 1.0).abs() < 1e-12);
    }

    fn rotation() -> Array2<f64> {
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
}
