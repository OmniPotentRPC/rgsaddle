//! Force projections, ported from eOn NEBForceProjection.cpp and
//! NEBProjection.cpp.

use crate::spring::SpringForces;
use ndarray::{Array1, ArrayView1, ArrayViewMut1, Zip};

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
    crate::stencil::perp(force, tangent)
}

/// Climbing-image force: `F - 2 (F . t) t + F_dneb`.
///
/// The reflection is [`crate::stencil::reflect`] (Goswami, Gunde, and
/// Jónsson, Front. Chem. (2026), doi:10.3389/fchem.2026.1807063, Eq. 9).
pub fn climbing_image_force(
    force: ArrayView1<f64>,
    tangent: ArrayView1<f64>,
    force_dneb: ArrayView1<f64>,
) -> Array1<f64> {
    crate::stencil::reflect(force, tangent) + force_dneb
}

/// DNEB correction with the Trygubenko–Wales atan switching.
pub fn dneb_component(
    force_spring: ArrayView1<f64>,
    tangent: ArrayView1<f64>,
    f_perp: ArrayView1<f64>,
) -> Array1<f64> {
    let mut dneb = Array1::zeros(tangent.len());
    let mut scratch = Array1::zeros(tangent.len());
    dneb_component_into(
        force_spring,
        tangent,
        f_perp,
        dneb.view_mut(),
        scratch.view_mut(),
    );
    dneb
}

/// [`dneb_component`] written into `dneb`; `scratch` holds the unit
/// perpendicular force on the way.
pub(crate) fn dneb_component_into(
    force_spring: ArrayView1<f64>,
    tangent: ArrayView1<f64>,
    f_perp: ArrayView1<f64>,
    mut dneb: ArrayViewMut1<f64>,
    mut scratch: ArrayViewMut1<f64>,
) {
    crate::stencil::perp_into(force_spring, tangent, dneb.view_mut());
    let spring_perp_norm = dneb.dot(&dneb).sqrt();
    let force_perp_norm = f_perp.dot(&f_perp).sqrt();
    if spring_perp_norm > 1e-10 && force_perp_norm > 1e-10 {
        Zip::from(scratch.view_mut())
            .and(f_perp)
            .for_each(|h, &f| *h = f / force_perp_norm);
        let overlap = dneb.dot(&scratch);
        let switching = 2.0 / std::f64::consts::PI
            * ((force_perp_norm * force_perp_norm) / (spring_perp_norm * spring_perp_norm)).atan();
        Zip::from(dneb)
            .and(scratch.view())
            .for_each(|d, &h| *d = (*d - h * overlap) * switching);
        return;
    }
    dneb.fill(0.0);
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
