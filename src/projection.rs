//! Force projections, ported from eOn NEBForceProjection.cpp and
//! NEBProjection.cpp.

use crate::spring::SpringForces;
use ndarray::{Array1, ArrayView1};

/// Which band projection assembles the relaxed force.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ProjectionKind {
    /// Mills, Jonsson, Schenter (1995): full spring + full true force.
    PlainElasticBand,
    /// Jonsson, Mills, Jacobsen (1998): parallel spring +
    /// perpendicular true force.
    #[default]
    Neb,
    /// Trygubenko & Wales, JCP 120:2082 (2004): NEB plus the switched
    /// perpendicular spring correction.
    DoublyNudged,
}

/// True force with its tangent component removed.
pub fn force_perp(force: ArrayView1<f64>, tangent: ArrayView1<f64>) -> Array1<f64> {
    let par = force.dot(&tangent);
    &force - &(tangent.to_owned() * par)
}

/// Climbing-image force: `F - 2 (F . t) t + F_dneb`.
pub fn climbing_image_force(
    force: ArrayView1<f64>,
    tangent: ArrayView1<f64>,
    force_dneb: ArrayView1<f64>,
) -> Array1<f64> {
    let par = force.dot(&tangent);
    &force - &(tangent.to_owned() * (2.0 * par)) + &force_dneb
}

/// DNEB correction with the Trygubenko–Wales atan switching.
pub fn dneb_component(
    force_spring: ArrayView1<f64>,
    tangent: ArrayView1<f64>,
    f_perp: ArrayView1<f64>,
) -> Array1<f64> {
    let spring_par = force_spring.dot(&tangent);
    let spring_perp = &force_spring - &(tangent.to_owned() * spring_par);

    let spring_perp_norm = spring_perp.dot(&spring_perp).sqrt();
    let force_perp_norm = f_perp.dot(&f_perp).sqrt();
    if spring_perp_norm > 1e-10 && force_perp_norm > 1e-10 {
        let force_perp_hat = f_perp.to_owned() / force_perp_norm;
        let overlap = spring_perp.dot(&force_perp_hat);
        let mut dneb = &spring_perp - &(force_perp_hat * overlap);
        let switching = 2.0 / std::f64::consts::PI
            * ((force_perp_norm * force_perp_norm)
                / (spring_perp_norm * spring_perp_norm))
                .atan();
        dneb *= switching;
        return dneb;
    }
    Array1::zeros(tangent.len())
}

impl ProjectionKind {
    /// Projected relaxation force at one non-climbing interior image.
    pub fn project(
        &self,
        force: ArrayView1<f64>,
        tangent: ArrayView1<f64>,
        spring: &SpringForces,
    ) -> Array1<f64> {
        match self {
            ProjectionKind::PlainElasticBand => &spring.full + &force,
            ProjectionKind::Neb => {
                let fp = force_perp(force, tangent);
                &spring.parallel + &fp
            }
            ProjectionKind::DoublyNudged => {
                let fp = force_perp(force, tangent);
                let dneb = dneb_component(spring.full.view(), tangent, fp.view());
                &spring.parallel + &fp + &dneb
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn force_perp_is_orthogonal_to_tangent() {
        let f = array![1.0, 2.0, 3.0];
        let t = array![1.0, 0.0, 0.0];
        let fp = force_perp(f.view(), t.view());
        assert_abs_diff_eq!(fp.dot(&t), 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(fp[1], 2.0, epsilon = 1e-12);
    }

    #[test]
    fn climbing_force_inverts_the_parallel_component() {
        let f = array![1.0, 2.0];
        let t = array![1.0, 0.0];
        let zero = array![0.0, 0.0];
        let fc = climbing_image_force(f.view(), t.view(), zero.view());
        assert_abs_diff_eq!(fc[0], -1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(fc[1], 2.0, epsilon = 1e-12);
    }

    #[test]
    fn dneb_switching_stays_within_the_spring_perp() {
        let spring = array![0.0, 0.4];
        let t = array![1.0, 0.0];
        let fp = array![0.0, 0.1];
        let d = dneb_component(spring.view(), t.view(), fp.view());
        // spring_perp is parallel to f_perp here: the projection
        // removes it entirely.
        assert_abs_diff_eq!(d[1], 0.0, epsilon = 1e-12);
    }
}
