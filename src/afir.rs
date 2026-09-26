//! Artificial force between two fragments.
//!
//! The force pulls the fragment centers together. Each atom in a
//! fragment receives an equal share. This is the force term an AFIR
//! step adds to the true gradient. It is not a finished path search.

/// Pull fragments `a` and `b` together with total magnitude `alpha`.
///
/// `alpha` is positive. The returned force has one vector per row of
/// `positions`. Atoms in neither fragment are left at zero.
pub fn afir_force(
    positions: &[[f64; 3]],
    fragment_a: &[usize],
    fragment_b: &[usize],
    alpha: f64,
) -> Vec<[f64; 3]> {
    let ca = centroid(positions, fragment_a);
    let cb = centroid(positions, fragment_b);
    let delta = [cb[0] - ca[0], cb[1] - ca[1], cb[2] - ca[2]];
    let norm = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
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
    force
}

fn centroid(positions: &[[f64; 3]], indices: &[usize]) -> [f64; 3] {
    let mut sum = [0.0; 3];
    for &index in indices {
        sum[0] += positions[index][0];
        sum[1] += positions[index][1];
        sum[2] += positions[index][2];
    }
    let n = indices.len() as f64;
    [sum[0] / n, sum[1] / n, sum[2] / n]
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
