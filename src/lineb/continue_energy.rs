//! Energy continuation from a minimum-energy path.
//!
//! The first tunneling energy is the highest MEP energy minus `ΔE`. Its
//! seed is the MEP segment between the two points where the MEP crosses
//! that energy, resampled to equal mass-weighted chords. Each lower
//! energy starts from the previous optimal path with the endpoints walked
//! along the mass-weighted force until `V = E`, then resampled the same way.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use super::action::l2;
use super::mass::Masses;
use super::optimize::{LinebConfig, LinebPath, LinebSession, LinebStatus};
use crate::error::SaddleError;
use crate::minmode::PointSurface;

/// Energy ladder of [`continue_from_mep`].
///
/// The first tunneling energy is `E_saddle − delta` and each of the
/// `levels` energies is `delta` below the last. Every energy uses
/// `images` LI-NEB images and at most `max_steps` optimizer steps.
///
/// Non-exhaustive: start from [`Continuation::new`] and chain setters.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Continuation {
    /// Energy step `ΔE` between levels, and from the MEP top to the first.
    pub delta: f64,
    /// LI-NEB images at every energy.
    pub images: usize,
    /// Number of energies.
    pub levels: usize,
    /// Optimizer step budget at each energy. A level still running after
    /// it is an error.
    pub max_steps: usize,
}

with_setters!(Continuation {
    with_delta => delta: val f64,
    with_images => images: val usize,
    with_levels => levels: val usize,
    with_max_steps => max_steps: val usize,
});

impl Continuation {
    /// `levels` energies `delta` apart on `images` images, 1000 steps each.
    pub fn new(delta: f64, images: usize, levels: usize) -> Self {
        Self {
            delta,
            images,
            levels,
            max_steps: 1000,
        }
    }

    fn check(&self) -> Result<(), SaddleError> {
        if !(self.delta.is_finite() && self.delta > 0.0) {
            return Err(SaddleError::Invalid(
                "the tunneling energy step must be finite and positive".into(),
            ));
        }
        if self.images < 3 || self.levels < 1 {
            return Err(SaddleError::Invalid(
                "continuation needs at least one level and three images".into(),
            ));
        }
        Ok(())
    }
}

/// Optimal tunneling paths at `E_saddle − ΔE`, `E_saddle − 2ΔE`, …
pub fn continue_from_mep<S: PointSurface>(
    surface: &S,
    mep: ArrayView2<f64>,
    masses: &Masses,
    config: &LinebConfig,
    ladder: &Continuation,
) -> Result<Vec<LinebPath>, SaddleError> {
    ladder.check()?;
    let &Continuation {
        delta,
        images,
        levels,
        max_steps,
    } = ladder;
    if mep.nrows() < 3 || mep.ncols() != masses.dof() {
        return Err(SaddleError::Shape(
            "the MEP needs at least three images and one mass per coordinate".into(),
        ));
    }
    let energies = image_energies(surface, mep)?;
    let saddle = highest(&energies)?;
    let mut paths = Vec::with_capacity(levels);
    let mut energy = energies[saddle] - delta;
    let mut seed = bracket(surface, masses, mep, &energies, saddle, energy, images)?;
    for level in 0..levels {
        let mut session = LinebSession::new(config.clone(), masses.clone(), seed.clone(), energy)?;
        let report = session.run(surface, max_steps)?;
        if report.status != LinebStatus::Converged {
            return Err(SaddleError::Stalled(format!(
                "tunneling path at energy {energy} did not converge in {max_steps} steps, nudged force {}, perpendicular {}",
                report.opt_max, report.perp_max
            )));
        }
        let path = session.path(surface)?;
        if level + 1 < levels {
            energy -= delta;
            seed = extend_seed(surface, masses, &path, energy, images)?;
        }
        paths.push(path);
    }
    Ok(paths)
}

fn image_energies<S: PointSurface>(
    surface: &S,
    images: ArrayView2<f64>,
) -> Result<Vec<f64>, SaddleError> {
    let mut energies = Vec::with_capacity(images.nrows());
    for row in images.rows() {
        let (energy, gradient) = surface.eval(row)?;
        if !energy.is_finite() || gradient.iter().any(|component| !component.is_finite()) {
            return Err(SaddleError::NonFinite("MEP image"));
        }
        energies.push(energy);
    }
    Ok(energies)
}

fn highest(energies: &[f64]) -> Result<usize, SaddleError> {
    let mut best = 0;
    for (index, energy) in energies.iter().enumerate() {
        if *energy > energies[best] {
            best = index;
        }
    }
    if best == 0 || best + 1 == energies.len() {
        return Err(SaddleError::Invalid(
            "the highest MEP image is an endpoint, so it is not a barrier".into(),
        ));
    }
    Ok(best)
}

fn bracket<S: PointSurface>(
    surface: &S,
    masses: &Masses,
    mep: ArrayView2<f64>,
    energies: &[f64],
    saddle: usize,
    target: f64,
    images: usize,
) -> Result<Array2<f64>, SaddleError> {
    let left = crossing_toward_start(energies, saddle, target)?;
    let right = crossing_toward_end(energies, saddle, target)?;
    let mut weighted = Vec::new();
    weighted.push(onto_segment(
        surface,
        masses,
        mep.row(left),
        mep.row(left + 1),
        energies[left],
        energies[left + 1],
        target,
    )?);
    for index in (left + 1)..right {
        weighted.push(masses.to_weighted(mep.row(index))?);
    }
    weighted.push(onto_segment(
        surface,
        masses,
        mep.row(right),
        mep.row(right + 1),
        energies[right],
        energies[right + 1],
        target,
    )?);
    resample(&weighted, images).and_then(|weighted| to_cartesian(masses, weighted.view()))
}

fn crossing_toward_start(
    energies: &[f64],
    saddle: usize,
    target: f64,
) -> Result<usize, SaddleError> {
    let mut index = saddle;
    while index > 0 && energies[index - 1] > target {
        index -= 1;
    }
    if index == 0 {
        return Err(SaddleError::Invalid(
            "the MEP never falls to the tunneling energy on the reactant side".into(),
        ));
    }
    // The segment (index - 1, index) changes from at or below the target
    // to above it.
    if energies[index] <= target || energies[index].is_nan() {
        return Err(SaddleError::Invalid(
            "the reactant-side MEP crossing has no image above the tunneling energy".into(),
        ));
    }
    Ok(index - 1)
}

fn crossing_toward_end(energies: &[f64], saddle: usize, target: f64) -> Result<usize, SaddleError> {
    let mut index = saddle;
    let last = energies.len() - 1;
    while index < last && energies[index + 1] > target {
        index += 1;
    }
    if index == last {
        return Err(SaddleError::Invalid(
            "the MEP never falls to the tunneling energy on the product side".into(),
        ));
    }
    if energies[index] <= target || energies[index].is_nan() {
        return Err(SaddleError::Invalid(
            "the product-side MEP crossing has no image above the tunneling energy".into(),
        ));
    }
    Ok(index)
}

fn onto_segment<S: PointSurface>(
    surface: &S,
    masses: &Masses,
    left: ArrayView1<f64>,
    right: ArrayView1<f64>,
    left_energy: f64,
    right_energy: f64,
    target: f64,
) -> Result<Array1<f64>, SaddleError> {
    let product = (left_energy - target) * (right_energy - target);
    if product > 0.0 {
        return Err(SaddleError::Invalid(
            "an MEP segment does not cross the tunneling energy".into(),
        ));
    }
    let mut a = masses.to_weighted(left)?;
    let mut b = masses.to_weighted(right)?;
    let mut ea = left_energy;
    let mut eb = right_energy;
    loop {
        if let Some(above) = strictly_above(&a, ea, &b, eb, target) {
            let width = l2((&b - &a).view());
            if width == 0.0 || width.is_nan() {
                return Ok(above);
            }
            let mid = (&a + &b) * 0.5;
            // A midpoint that lands on an endpoint cannot shrink the bracket.
            if !(l2((&mid - &a).view()) > 0.0 && l2((&mid - &b).view()) > 0.0) {
                return Ok(above);
            }
            let x = masses.to_cartesian(mid.view())?;
            let (em, _) = surface.eval(x.view())?;
            if !em.is_finite() {
                return Err(SaddleError::NonFinite("equipotential bisection"));
            }
            // Sides are `V > E` and `V ≤ E`. A midpoint exactly at `E`
            // joins the side at or below, so the bracket keeps a point
            // strictly above.
            if (em > target) == (ea > target) {
                a = mid;
                ea = em;
            } else {
                b = mid;
                eb = em;
            }
        } else {
            return Err(SaddleError::Invalid(
                "the equipotential bisection has no point above the tunneling energy".into(),
            ));
        }
    }
}

/// The bracket endpoint that is still strictly above `target`. The seed
/// of a fixed-energy path has to start there; a point on or below `V = E`
/// has no tunneling momentum.
fn strictly_above(
    a: &Array1<f64>,
    ea: f64,
    b: &Array1<f64>,
    eb: f64,
    target: f64,
) -> Option<Array1<f64>> {
    if ea > target && (eb <= target || eb.is_nan() || (ea - target).abs() <= (eb - target).abs()) {
        Some(a.clone())
    } else if eb > target {
        Some(b.clone())
    } else {
        None
    }
}

fn extend_seed<S: PointSurface>(
    surface: &S,
    masses: &Masses,
    path: &LinebPath,
    target: f64,
    images: usize,
) -> Result<Array2<f64>, SaddleError> {
    let n = path.weighted.nrows();
    let left = walk_to_energy(
        surface,
        masses,
        path.weighted.row(0).to_owned(),
        path.weighted.row(1).to_owned(),
        target,
    )?;
    let right = walk_to_energy(
        surface,
        masses,
        path.weighted.row(n - 1).to_owned(),
        path.weighted.row(n - 2).to_owned(),
        target,
    )?;
    let mut polyline = Vec::with_capacity(n + 2);
    polyline.push(left);
    for index in 0..n {
        polyline.push(path.weighted.row(index).to_owned());
    }
    polyline.push(right);
    resample(&polyline, images).and_then(|weighted| to_cartesian(masses, weighted.view()))
}

fn walk_to_energy<S: PointSurface>(
    surface: &S,
    masses: &Masses,
    mut point: Array1<f64>,
    neighbor: Array1<f64>,
    target: f64,
) -> Result<Array1<f64>, SaddleError> {
    let spacing = l2((&point - &neighbor).view());
    if spacing <= 0.0 || spacing.is_nan() {
        return Err(SaddleError::Invalid(
            "an endpoint has no chord to walk along".into(),
        ));
    }
    let mut energy = potential(surface, masses, point.view())?;
    if energy <= target || energy.is_nan() {
        return Err(SaddleError::Invalid(
            "an endpoint is already at or below the next tunneling energy".into(),
        ));
    }
    // A step of one chord can cross the floor of a valley; it is halved
    // until the energy falls.
    let mut step = spacing;
    loop {
        let x = masses.to_cartesian(point.view())?;
        let (_, gradient) = surface.eval(x.view())?;
        let force = masses.weighted_force(gradient.view())?;
        let norm = l2(force.view());
        if norm <= 0.0 || norm.is_nan() {
            return Err(SaddleError::Invalid(
                "the downhill walk reached a vanishing force above the tunneling energy".into(),
            ));
        }
        let (next, next_energy) = loop {
            let mut next = point.clone();
            next.scaled_add(step / norm, &force);
            let advance = l2((&next - &point).view());
            if advance == 0.0 || advance.is_nan() {
                return Err(SaddleError::Stalled(
                    "the downhill walk stopped above the tunneling energy".into(),
                ));
            }
            let next_energy = potential(surface, masses, next.view())?;
            if next_energy < energy {
                break (next, next_energy);
            }
            step *= 0.5;
        };
        if next_energy <= target || next_energy.is_nan() {
            return onto_segment(
                surface,
                masses,
                masses.to_cartesian(point.view())?.view(),
                masses.to_cartesian(next.view())?.view(),
                energy,
                next_energy,
                target,
            );
        }
        point = next;
        energy = next_energy;
    }
}

fn potential<S: PointSurface>(
    surface: &S,
    masses: &Masses,
    weighted: ArrayView1<f64>,
) -> Result<f64, SaddleError> {
    let x = masses.to_cartesian(weighted)?;
    let (energy, gradient) = surface.eval(x.view())?;
    if !energy.is_finite() || gradient.iter().any(|component| !component.is_finite()) {
        return Err(SaddleError::NonFinite("continuation image"));
    }
    Ok(energy)
}

fn resample(points: &[Array1<f64>], images: usize) -> Result<Array2<f64>, SaddleError> {
    if points.len() < 2 || images < 2 {
        return Err(SaddleError::Shape(
            "resampling needs two polyline vertices and two images".into(),
        ));
    }
    let dof = points[0].len();
    let mut cumulative = vec![0.0];
    for pair in points.windows(2) {
        let chord = l2((&pair[1] - &pair[0]).view());
        cumulative.push(cumulative[cumulative.len() - 1] + chord);
    }
    let total = cumulative[cumulative.len() - 1];
    if total <= 0.0 || total.is_nan() {
        return Err(SaddleError::Invalid(
            "the polyline to resample has no length".into(),
        ));
    }
    let mut out = Array2::zeros((images, dof));
    out.row_mut(0).assign(&points[0]);
    out.row_mut(images - 1).assign(&points[points.len() - 1]);
    let mut segment = 0;
    for image in 1..images - 1 {
        let distance = total * image as f64 / (images - 1) as f64;
        while segment + 2 < cumulative.len()
            && (cumulative[segment + 1] <= distance || cumulative[segment + 1].is_nan())
        {
            segment += 1;
        }
        let (left, right) = (cumulative[segment], cumulative[segment + 1]);
        let span = right - left;
        let fraction = if span > 0.0 {
            (distance - left) / span
        } else {
            0.0
        };
        let mut row = points[segment].clone();
        let mut step = &points[segment + 1] - &points[segment];
        step *= fraction;
        row += &step;
        out.row_mut(image).assign(&row);
    }
    Ok(out)
}

fn to_cartesian(masses: &Masses, weighted: ArrayView2<f64>) -> Result<Array2<f64>, SaddleError> {
    let mut cartesian = Array2::zeros(weighted.dim());
    for i in 0..weighted.nrows() {
        cartesian
            .row_mut(i)
            .assign(&masses.to_cartesian(weighted.row(i))?);
    }
    Ok(cartesian)
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    struct Ramp;

    impl PointSurface for Ramp {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            Ok((x[0], array![1.0]))
        }
    }

    struct Well;

    impl PointSurface for Well {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            Ok((x[0] * x[0], array![2.0 * x[0]]))
        }
    }

    /// From `x = 1` on `V = x²` with a chord of 3, one chord along the
    /// force lands at `x = −2`, above the start. The halved step reaches
    /// `V = 1/4` and the walk ends on `V = 1/2`.
    #[test]
    fn a_walk_that_crosses_the_floor_halves_its_step() {
        let masses = Masses::coordinates(array![1.0].view()).unwrap();
        let point = walk_to_energy(&Well, &masses, array![1.0], array![4.0], 0.5).unwrap();
        let energy = point[0] * point[0];
        assert!(
            energy > 0.5 && energy - 0.5 <= 4.0 * f64::EPSILON,
            "{energy}"
        );
    }

    /// The first midpoint of `V = x` on `[-1, 1]` lands exactly on
    /// `E = 0`; the bisection still ends on a point strictly above.
    #[test]
    fn a_midpoint_on_the_energy_keeps_a_point_above() {
        let masses = Masses::coordinates(array![1.0].view()).unwrap();
        let point = onto_segment(
            &Ramp,
            &masses,
            array![-1.0].view(),
            array![1.0].view(),
            -1.0,
            1.0,
            0.0,
        )
        .unwrap();
        assert!(point[0] > 0.0 && point[0] <= f64::EPSILON, "{}", point[0]);
    }
}
