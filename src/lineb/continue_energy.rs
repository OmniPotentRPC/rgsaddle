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
    /// Interpolate the initial seed along the chord between its two MEP crossings.
    pub linear_seed: bool,
}

with_setters!(Continuation {
    with_linear_seed => linear_seed: val bool,
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
            linear_seed: false,
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
    continue_from_mep_with_observer(surface, mep, masses, config, ladder, |_, _| Ok(()))
}

/// Continue the energy ladder and call `observe(level, path)` at each completed
/// level, before any surface evaluation for the next seed. Levels are zero-based.
/// An observer can record provider counters or save the converged path. Its
/// error stops continuation.
pub fn continue_from_mep_with_observer<S: PointSurface>(
    surface: &S,
    mep: ArrayView2<f64>,
    masses: &Masses,
    config: &LinebConfig,
    ladder: &Continuation,
    observe: impl FnMut(usize, &LinebPath) -> Result<(), SaddleError>,
) -> Result<Vec<LinebPath>, SaddleError> {
    if mep.nrows() < 3 || mep.ncols() != masses.dof() {
        return Err(SaddleError::Shape(
            "the MEP needs at least three images and one mass per coordinate".into(),
        ));
    }
    continue_with_seed(
        surface,
        masses,
        config,
        ladder,
        |seeding| {
            let energies = image_energies(seeding, mep)?;
            let saddle = highest(&energies)?;
            let energy = energies[saddle] - ladder.delta;
            let seed = bracket(
                seeding,
                masses,
                mep,
                &energies,
                saddle,
                BracketSpec {
                    target: energy,
                    images: ladder.images,
                    linear_seed: ladder.linear_seed,
                },
            )?;
            Ok((energy, seed))
        },
        observe,
    )
}

/// Resume from the Cartesian turning-to-turning images of a converged path.
/// At the same energy the images are resampled; at lower energy both ends are
/// walked along the physical force. All seed evaluations belong to the new run.
#[allow(clippy::too_many_arguments)]
pub fn continue_from_path_with_observer<S: PointSurface>(
    surface: &S,
    cartesian: ArrayView2<f64>,
    saved_energy: f64,
    first_energy: f64,
    masses: &Masses,
    config: &LinebConfig,
    ladder: &Continuation,
    observe: impl FnMut(usize, &LinebPath) -> Result<(), SaddleError>,
) -> Result<Vec<LinebPath>, SaddleError> {
    if !saved_energy.is_finite() || !first_energy.is_finite() || first_energy > saved_energy {
        return Err(SaddleError::Invalid(
            "the restart energy must be at or below the saved energy".into(),
        ));
    }
    if cartesian.nrows() < 3
        || cartesian.ncols() != masses.dof()
        || !cartesian.iter().all(|v| v.is_finite())
    {
        return Err(SaddleError::Invalid(
            "the saved path needs finite Cartesian images with matching masses".into(),
        ));
    }
    continue_with_seed(
        surface,
        masses,
        config,
        ladder,
        |seeding| {
            let mut weighted = cartesian.to_owned();
            for (mut row, x) in weighted.rows_mut().into_iter().zip(cartesian.rows()) {
                row.assign(&masses.to_weighted(x)?);
            }
            let seed = if first_energy < saved_energy {
                extend_seed(
                    seeding,
                    masses,
                    weighted.view(),
                    first_energy,
                    ladder.images,
                )?
            } else if ladder.images == cartesian.nrows() {
                cartesian.to_owned()
            } else {
                let points = weighted
                    .rows()
                    .into_iter()
                    .map(|r| r.to_owned())
                    .collect::<Vec<_>>();
                to_cartesian(masses, resample(&points, ladder.images)?.view())?
            };
            Ok((first_energy, seed))
        },
        observe,
    )
}

fn continue_with_seed<S: PointSurface>(
    surface: &S,
    masses: &Masses,
    config: &LinebConfig,
    ladder: &Continuation,
    make_seed: impl FnOnce(&super::Counted<'_, S>) -> Result<(f64, Array2<f64>), SaddleError>,
    mut observe: impl FnMut(usize, &LinebPath) -> Result<(), SaddleError>,
) -> Result<Vec<LinebPath>, SaddleError> {
    ladder.check()?;
    let &Continuation {
        delta,
        images,
        levels,
        max_steps,
        ..
    } = ladder;
    let seed_calls = std::sync::atomic::AtomicUsize::new(0);
    let seeding = super::Counted {
        surface,
        calls: &seed_calls,
    };
    let (mut energy, mut seed) = make_seed(&seeding)?;
    let mut paths = Vec::with_capacity(levels);
    for level in 0..levels {
        let mut session = LinebSession::new(config.clone(), masses.clone(), seed.clone(), energy)?;
        let report = session.run(surface, max_steps)?;
        if report.status != LinebStatus::Converged {
            return Err(SaddleError::Stalled(format!(
                "tunneling path at energy {energy} did not converge in {max_steps} steps, nudged force {}, endpoint {}, perpendicular {}",
                report.opt_max, report.endpoint_max, report.perp_max
            )));
        }
        let mut path = session.path(surface)?;
        path.calls.seed = seed_calls.swap(0, std::sync::atomic::Ordering::Relaxed);
        observe(level, &path)?;
        if level + 1 < levels {
            energy -= delta;
            seed = extend_seed(&seeding, masses, path.weighted.view(), energy, images)?;
        }
        paths.push(path);
    }
    Ok(paths)
}

/// A minimum-to-minimum seed includes classically allowed tails on both
/// sides. Keep the barrier segment between the crossings of the target energy.
pub(super) fn clip_seed<S: PointSurface>(
    surface: &S,
    masses: &Masses,
    seed: ArrayView2<f64>,
    energy: f64,
    linear_seed: bool,
) -> Result<Option<Array2<f64>>, SaddleError> {
    let last = seed.nrows() - 1;
    let left = surface.eval(seed.row(0))?.0;
    if !left.is_finite() {
        return Err(SaddleError::NonFinite("reactant seed energy"));
    }
    let right = surface.eval(seed.row(last))?.0;
    if !right.is_finite() {
        return Err(SaddleError::NonFinite("product seed energy"));
    }
    // A prepared band lies inside the barrier; its ends relax on the turning surfaces.
    if left >= energy && right >= energy {
        return Ok(None);
    }
    // If a saved pair straddles E, both ends must lie within the PES resolution.
    let resolution = super::path::energy_roundoff(energy);
    if (left >= energy || right >= energy)
        && (energy - left).abs() <= resolution
        && (energy - right).abs() <= resolution
    {
        return Ok(None);
    }
    if left > energy || right > energy {
        let side = if left > energy { "reactant" } else { "product" };
        return Err(SaddleError::Invalid(format!(
            "seed lacks a {side}-side crossing of the tunneling energy {energy}: endpoint energies {left} and {right}"
        )));
    }
    let mut energies = vec![left];
    energies.extend(image_energies(
        surface,
        seed.slice(ndarray::s![1..last, ..]),
    )?);
    energies.push(right);
    let saddle = highest(&energies)?;
    bracket(
        surface,
        masses,
        seed,
        &energies,
        saddle,
        BracketSpec {
            target: energy,
            images: seed.nrows(),
            linear_seed,
        },
    )
    .map(Some)
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

struct BracketSpec {
    target: f64,
    images: usize,
    linear_seed: bool,
}

fn bracket<S: PointSurface>(
    surface: &S,
    masses: &Masses,
    mep: ArrayView2<f64>,
    energies: &[f64],
    saddle: usize,
    spec: BracketSpec,
) -> Result<Array2<f64>, SaddleError> {
    let left = crossing_toward_start(energies, saddle, spec.target)?;
    let right = crossing_toward_end(energies, saddle, spec.target)?;
    let mut weighted = Vec::new();
    weighted.push(onto_segment(
        surface,
        masses,
        mep.row(left),
        mep.row(left + 1),
        energies[left],
        energies[left + 1],
        spec.target,
    )?);
    if !spec.linear_seed {
        for index in (left + 1)..=right {
            weighted.push(masses.to_weighted(mep.row(index))?);
        }
    }
    weighted.push(onto_segment(
        surface,
        masses,
        mep.row(right),
        mep.row(right + 1),
        energies[right],
        energies[right + 1],
        spec.target,
    )?);
    resample(&weighted, spec.images).and_then(|weighted| to_cartesian(masses, weighted.view()))
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
    weighted: ArrayView2<f64>,
    target: f64,
    images: usize,
) -> Result<Array2<f64>, SaddleError> {
    let n = weighted.nrows();
    let left = walk_to_energy(
        surface,
        masses,
        weighted.row(0).to_owned(),
        weighted.row(1).to_owned(),
        target,
    )?;
    let right = walk_to_energy(
        surface,
        masses,
        weighted.row(n - 1).to_owned(),
        weighted.row(n - 2).to_owned(),
        target,
    )?;
    let mut polyline = Vec::with_capacity(n + 2);
    polyline.push(left);
    for index in 0..n {
        polyline.push(weighted.row(index).to_owned());
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

    #[test]
    fn a_one_sided_seed_reports_the_missing_crossing_in_either_direction() {
        let masses = Masses::coordinates(array![1.0].view()).unwrap();
        for (path, side) in [
            (array![[-1.0], [0.5], [1.0]], "product"),
            (array![[1.0], [0.5], [-1.0]], "reactant"),
        ] {
            let result = clip_seed(&Ramp, &masses, path.view(), 0.0, false);
            assert!(matches!(result, Err(SaddleError::Invalid(message)) if message.contains(side)));
        }
    }

    #[test]
    fn prepared_bands_and_roundoff_straddles_have_distinct_energy_conditions() {
        let masses = Masses::coordinates(array![1.0].view()).unwrap();
        let build = |ends: [f64; 2]| {
            let seed = array![[ends[0]], [3.0], [ends[1]]];
            LinebSession::from_seed(
                &Ramp,
                LinebConfig::new(1e-4, 1e-4),
                masses.clone(),
                seed,
                1.0,
            )
        };
        for ends in [
            [2.0, 2.5],
            [1.0 - f64::EPSILON, 1.0 + f64::EPSILON],
            [1.0 + f64::EPSILON, 1.0 - f64::EPSILON],
        ] {
            let seed = build(ends).unwrap().cartesian().unwrap();
            assert_eq!([seed[[0, 0]], seed[[2, 0]]], ends);
        }
        for (ends, side) in [
            ([1.0 - f64::EPSILON, 2.0], "product"),
            ([2.0, 1.0 - f64::EPSILON], "reactant"),
        ] {
            assert!(matches!(build(ends), Err(SaddleError::Invalid(s)) if s.contains(side)));
        }
    }

    #[test]
    fn saved_turning_points_can_straddle_the_target_by_roundoff() {
        let masses = Masses::coordinates(array![1.0].view()).unwrap();
        let seed = array![[1.0 - f64::EPSILON], [2.0], [1.0 + f64::EPSILON]];
        assert!(
            clip_seed(&Ramp, &masses, seed.view(), 1.0, false)
                .unwrap()
                .is_none()
        );
    }

    struct Well;

    #[test]
    fn seed_follows_the_bend_between_adjacent_crossings() {
        struct Ridge;

        impl PointSurface for Ridge {
            fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
                Ok((1.0 - x[0].powi(2), array![-2.0 * x[0], 0.0]))
            }
        }

        let mep = array![[-1.0, 0.0], [0.0, 1.0], [1.0, 0.0]];
        let masses = Masses::coordinates(array![1.0, 1.0].view()).unwrap();
        let session = LinebSession::from_seed(
            &Ridge,
            LinebConfig::new(0.01, 0.01),
            masses.clone(),
            mep.clone(),
            0.5,
        )
        .unwrap();
        let clipped = session.cartesian().unwrap();
        for end in [0, 2] {
            assert!((Ridge.eval(clipped.row(end)).unwrap().0 - 0.5).abs() < 1e-14);
        }
        assert!((clipped[[1, 1]] - 1.0).abs() < 1e-14);
        let energies = image_energies(&Ridge, mep.view()).unwrap();
        let seed = bracket(
            &Ridge,
            &masses,
            mep.view(),
            &energies,
            1,
            BracketSpec {
                target: 0.5,
                images: 3,
                linear_seed: false,
            },
        )
        .unwrap();
        let chord = bracket(
            &Ridge,
            &masses,
            mep.view(),
            &energies,
            1,
            BracketSpec {
                target: 0.5,
                images: 3,
                linear_seed: true,
            },
        )
        .unwrap();
        let midpoint = (&seed.row(0) + &seed.row(2)) * 0.5;
        assert!(l2((&chord.row(1) - &midpoint).view()) < 1e-12);
        assert!(l2((&seed.row(1) - &mep.row(1)).view()) < 1e-12);
    }

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
