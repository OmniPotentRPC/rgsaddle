//! Artificial force between two fragments.
//!
//! The force pulls the fragment centers together. Each atom in a
//! fragment receives an equal share. This is the force term an AFIR
//! contributes beside the physical force. The host owns the path search.

/// Pull fragments `a` and `b` together with total magnitude `alpha`.
///
/// `alpha` is finite and nonnegative. The returned force has one vector per row of
/// `positions`. Atoms in neither fragment are left at zero.
pub fn afir_force(
    positions: &[[f64; 3]],
    fragment_a: &[usize],
    fragment_b: &[usize],
    alpha: f64,
) -> Vec<[f64; 3]> {
    try_afir_force(positions, fragment_a, fragment_b, alpha)
        .expect("AFIR requires finite positions, distinct fragment centers and disjoint nonempty fragments")
}

/// Checked artificial force with explicit shape and numerical errors.
pub fn try_afir_force(
    positions: &[[f64; 3]],
    fragment_a: &[usize],
    fragment_b: &[usize],
    alpha: f64,
) -> Result<Vec<[f64; 3]>, crate::SaddleError> {
    use crate::SaddleError;
    if !alpha.is_finite() || alpha < 0.0 {
        return Err(SaddleError::Invalid("AFIR strength must be finite and nonnegative".into()));
    }
    if fragment_a.is_empty() || fragment_b.is_empty() {
        return Err(SaddleError::Shape("AFIR fragments must be nonempty".into()));
    }
    if !positions.iter().flatten().all(|x| x.is_finite()) {
        return Err(SaddleError::NonFinite("AFIR positions"));
    }
    let mut assigned = vec![false; positions.len()];
    for &index in fragment_a.iter().chain(fragment_b) {
        let Some(seen) = assigned.get_mut(index) else {
            return Err(SaddleError::Shape("AFIR fragment index is outside positions".into()));
        };
        if *seen {
            return Err(SaddleError::Invalid("AFIR fragments must be disjoint and contain unique atoms".into()));
        }
        *seen = true;
    }
    if alpha == 0.0 { return Ok(vec![[0.0; 3]; positions.len()]); }
    let ca = centroid(positions, fragment_a);
    let cb = centroid(positions, fragment_b);
    let delta = [cb[0] - ca[0], cb[1] - ca[1], cb[2] - ca[2]];
    let norm = delta[0].hypot(delta[1]).hypot(delta[2]);
    if !(norm.is_finite() && norm > 0.0) {
        return Err(SaddleError::Invalid("AFIR fragment centers must have a finite nonzero separation".into()));
    }
    let direction = [delta[0] / norm, delta[1] / norm, delta[2] / norm];
    let mut force = vec![[0.0; 3]; positions.len()];
    let share_a = alpha / fragment_a.len() as f64;
    let share_b = alpha / fragment_b.len() as f64;
    for &index in fragment_a {
        force[index] = [share_a * direction[0], share_a * direction[1], share_a * direction[2]];
    }
    for &index in fragment_b {
        force[index] = [-share_b * direction[0], -share_b * direction[1], -share_b * direction[2]];
    }
    Ok(force)
}

fn centroid(positions: &[[f64; 3]], indices: &[usize]) -> [f64; 3] {
    let mut sum = [0.0; 3];
    let n = indices.len() as f64;
    for &index in indices {
        for component in 0..3 {
            sum[component] += positions[index][component] / n;
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::afir_force;

    #[test]
    fn two_atoms_are_pulled_together() {
        let positions = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let force = afir_force(&positions, &[0], &[1], 1.0);
        assert!((force[0][0] - 1.0).abs() < 1e-12);
        assert!((force[1][0] + 1.0).abs() < 1e-12);
        assert!(force[0][1].abs() < 1e-12 && force[1][1].abs() < 1e-12);
    }
}

#[cfg(test)]
mod domain_tests {
    use super::try_afir_force;

    #[test]
    fn unequal_fragments_share_a_balanced_force() {
        let positions = [[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [2.0, 1.0, 0.0], [7.0, 5.0, 1.0]];
        let force = try_afir_force(&positions, &[0, 1], &[2], 4.0).unwrap();
        assert_eq!(force, vec![[2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [-4.0, 0.0, 0.0], [0.0; 3]]);
    }

    #[test]
    fn invalid_or_singular_fragments_are_rejected() {
        let positions = [[0.0; 3], [0.0; 3]];
        for (a, b) in [(vec![], vec![1]), (vec![0], vec![]), (vec![2], vec![1]),
                       (vec![0, 0], vec![1]), (vec![0], vec![0]), (vec![0], vec![1])] {
            assert!(try_afir_force(&positions, &a, &b, 1.0).is_err());
        }
        let separated = [[0.0; 3], [1.0, 0.0, 0.0]];
        for alpha in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(try_afir_force(&separated, &[0], &[1], alpha).is_err());
        }
        assert_eq!(try_afir_force(&positions, &[0], &[1], 0.0).unwrap(), vec![[0.0; 3]; 2]);
    }
}
