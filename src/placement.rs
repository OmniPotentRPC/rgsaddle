//! Geometry placement: move atoms onto sites and push a geometry past a
//! saddle.
//!
//! Both operations build a starting geometry for a relaxation or a band.
//! Neither one relaxes anything, and neither knows what a defect is. The
//! caller names the atoms that move and the sites they move to.
//!
//! - [`translate_onto_sites`] pairs the chosen atoms with a list of sites
//!   by the nearest remaining pair first and puts each atom on its site.
//!   A vacancy centroid and the atoms of an interstitial cluster are one use;
//!   an adatom and a surface hollow are another.
//! - [`push_past`] extrapolates from the reactant through a saddle,
//!   `saddle + f * (saddle - reactant)`, the push that activation-relaxation
//!   searches make before they minimize. A factor of `0` returns the saddle.
//!
//! Every displacement goes through one minimum-image fold, the shortest
//! image under whole periodic lattice vectors. An axis is folded only when
//! `periodic` says so. In an orthogonal cell the fold rounds each periodic
//! fraction half toward zero, so a displacement that lies exactly half a
//! cell away keeps its sign. In a sheared or skewed cell rounding the
//! fractions is not the minimum image, and the neighbouring images are
//! searched. The `cell` rows are the lattice vectors, and an axis that is
//! not periodic still has to make the cell invertible, but its length is
//! never used.
//!
//! Positions, sites and the cell are in Angstrom, and positions and sites are
//! flat atom-major `[x0, y0, z0, x1, ...]`.

use ndarray::{Array1, ArrayView1};

use crate::cell_log::M3;
use crate::error::SaddleError;
use crate::occupancy::Images;

/// Atoms placed by [`translate_onto_sites`].
#[derive(Clone, Debug, PartialEq)]
pub struct SitePlacement {
    /// The geometry with each paired atom on its site, flat atom-major. Atoms
    /// that were not paired keep their positions.
    pub positions: Array1<f64>,
    /// `(atom, site)` for every pair, in the order the pairs were taken: the
    /// nearest pair first.
    pub pairs: Vec<(usize, usize)>,
}

/// Minimum-image fold of a displacement in one cell.
struct Fold(Images);

impl Fold {
    fn new(cell: M3, periodic: [bool; 3]) -> Result<Self, SaddleError> {
        if cell.iter().flatten().any(|v| !v.is_finite()) {
            return Err(SaddleError::NonFinite("placement cell"));
        }
        Ok(Self(Images::new(cell, periodic)?))
    }

    fn displacement(&self, d: [f64; 3]) -> [f64; 3] {
        self.0.displacement(d)
    }

    fn distance(&self, a: [f64; 3], b: [f64; 3]) -> f64 {
        let d = self.displacement([b[0] - a[0], b[1] - a[1], b[2] - a[2]]);
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
    }
}

fn point(flat: ArrayView1<f64>, index: usize) -> [f64; 3] {
    [flat[3 * index], flat[3 * index + 1], flat[3 * index + 2]]
}

/// Put the `movers` onto `sites`, nearest pair first.
///
/// Every remaining mover is compared with every remaining site under the
/// minimum image, the closest pair is taken, and the mover is put on the site
/// at `x + fold(site - x)`. That keeps the move the short way round and leaves
/// the atom outside the cell when the short way crosses a face. The next pair
/// is taken from what is left. A tie goes to the earlier entry of `movers`,
/// then to the lower site.
///
/// When there are more movers than sites the farthest movers keep their
/// positions. When there are more sites than movers the sites that are left
/// stay empty. The result is a placement, not a minimum: relax it before it is
/// used as an endpoint.
///
/// `positions` and `sites` are flat atom-major triples in Angstrom. `movers`
/// are 0-based atom indices and must be distinct. `cell` has one lattice vector
/// per row. `periodic` names the axes along which the minimum image is taken.
///
/// # Errors
///
/// - [`SaddleError::Shape`] when `positions` or `sites` is not a whole number of
///   triples, when `movers` or `sites` is empty, or when a mover is not an atom
///   of `positions`.
/// - [`SaddleError::Invalid`] when a mover is listed twice or the cell has no
///   volume.
/// - [`SaddleError::NonFinite`] when a coordinate or a cell entry is NaN or
///   infinite, or when a mover-site distance or a placed coordinate is not
///   finite, as when coordinates near `±f64::MAX` overflow their difference.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use rgsaddle::translate_onto_sites;
///
/// // The atom at x = 9.5 and the site at x = 0.5 are one Angstrom apart
/// // through the face of a 10 Angstrom cell.
/// let cell = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]];
/// let atoms = array![9.5, 0.0, 0.0];
/// let sites = array![0.5, 0.0, 0.0];
/// let out = translate_onto_sites(atoms.view(), cell, [true; 3], &[0], sites.view())?;
/// assert_eq!(out.positions[0], 10.5);
/// assert_eq!(out.pairs, vec![(0, 0)]);
/// # Ok::<(), rgsaddle::SaddleError>(())
/// ```
pub fn translate_onto_sites(
    positions: ArrayView1<f64>,
    cell: M3,
    periodic: [bool; 3],
    movers: &[usize],
    sites: ArrayView1<f64>,
) -> Result<SitePlacement, SaddleError> {
    if !positions.len().is_multiple_of(3) || !sites.len().is_multiple_of(3) {
        return Err(SaddleError::Shape(
            "positions and sites must hold whole (x, y, z) triples".into(),
        ));
    }
    if movers.is_empty() || sites.is_empty() {
        return Err(SaddleError::Shape(
            "translate_onto_sites needs at least one mover and one site".into(),
        ));
    }
    let n_atoms = positions.len() / 3;
    for (k, &atom) in movers.iter().enumerate() {
        if atom >= n_atoms {
            return Err(SaddleError::Shape(format!(
                "mover {atom} is not an atom of a {n_atoms}-atom geometry"
            )));
        }
        if movers[..k].contains(&atom) {
            return Err(SaddleError::Invalid(format!(
                "mover {atom} is listed twice"
            )));
        }
    }
    if positions.iter().chain(sites.iter()).any(|v| !v.is_finite()) {
        return Err(SaddleError::NonFinite("placement coordinates"));
    }
    let fold = Fold::new(cell, periodic)?;
    let mut out = positions.to_owned();
    let mut loose: Vec<usize> = movers.to_vec();
    let mut free: Vec<usize> = (0..sites.len() / 3).collect();
    let mut pairs = Vec::with_capacity(loose.len().min(free.len()));
    while !loose.is_empty() && !free.is_empty() {
        let mut best: Option<(usize, usize, f64)> = None;
        for (m, &atom) in loose.iter().enumerate() {
            for (s, &site) in free.iter().enumerate() {
                let d = fold.distance(point(positions, atom), point(sites, site));
                if !d.is_finite() {
                    return Err(SaddleError::NonFinite("placement distance"));
                }
                if best.is_none_or(|(_, _, nearest)| d < nearest) {
                    best = Some((m, s, d));
                }
            }
        }
        let Some((m, s, _)) = best else {
            break;
        };
        let atom = loose.remove(m);
        let site = free.remove(s);
        let from = point(positions, atom);
        let to = point(sites, site);
        let d = fold.displacement([to[0] - from[0], to[1] - from[1], to[2] - from[2]]);
        for ax in 0..3 {
            out[3 * atom + ax] = from[ax] + d[ax];
        }
        if (0..3).any(|ax| !out[3 * atom + ax].is_finite()) {
            return Err(SaddleError::NonFinite("placed position"));
        }
        pairs.push((atom, site));
    }
    Ok(SitePlacement {
        positions: out,
        pairs,
    })
}

/// Extrapolate from `reactant` through `saddle`: `saddle + factor * (saddle -
/// reactant)`.
///
/// The displacement of every atom is folded to its minimum image first, so a
/// saddle that sits across a cell face from the reactant is pushed the short
/// way. A `factor` of `0` returns the saddle, `1` doubles the distance from the
/// reactant, and a negative factor pulls back toward the reactant. The result
/// is not a minimum. Activation-relaxation searches relax it, with a factor of
/// about `0.1` to `1.5`, to leave the saddle on the product side.
///
/// `reactant` and `saddle` are flat atom-major triples of one atom ordering, in
/// Angstrom. Two empty geometries give an empty result.
///
/// # Errors
///
/// - [`SaddleError::Shape`] when the lengths differ or are not a whole number
///   of triples.
/// - [`SaddleError::NonFinite`] when `factor`, a coordinate or a cell entry is
///   NaN or infinite, or when a folded displacement or a pushed coordinate
///   overflows to infinity.
/// - [`SaddleError::Invalid`] when the cell has no volume.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use rgsaddle::push_past;
///
/// let cell = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]];
/// let reactant = array![0.0, 0.0, 0.0];
/// let saddle = array![1.0, 0.0, 0.0];
/// let pushed = push_past(reactant.view(), saddle.view(), cell, [true; 3], 0.5)?;
/// assert_eq!(pushed[0], 1.5);
/// # Ok::<(), rgsaddle::SaddleError>(())
/// ```
pub fn push_past(
    reactant: ArrayView1<f64>,
    saddle: ArrayView1<f64>,
    cell: M3,
    periodic: [bool; 3],
    factor: f64,
) -> Result<Array1<f64>, SaddleError> {
    if reactant.len() != saddle.len() || !saddle.len().is_multiple_of(3) {
        return Err(SaddleError::Shape(format!(
            "reactant has {} coordinates and saddle has {}; both must be whole triples of one length",
            reactant.len(),
            saddle.len()
        )));
    }
    if !factor.is_finite() {
        return Err(SaddleError::NonFinite("push factor"));
    }
    if reactant.iter().chain(saddle.iter()).any(|v| !v.is_finite()) {
        return Err(SaddleError::NonFinite("push coordinates"));
    }
    let fold = Fold::new(cell, periodic)?;
    let mut out = saddle.to_owned();
    for atom in 0..saddle.len() / 3 {
        let r = point(reactant, atom);
        let s = point(saddle, atom);
        let d = fold.displacement([s[0] - r[0], s[1] - r[1], s[2] - r[2]]);
        for ax in 0..3 {
            out[3 * atom + ax] = s[ax] + factor * d[ax];
        }
    }
    if out.iter().any(|v| !v.is_finite()) {
        return Err(SaddleError::NonFinite("pushed geometry"));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    const CUBE: M3 = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]];
    const ALL: [bool; 3] = [true; 3];

    fn flat(rows: &[[f64; 3]]) -> Array1<f64> {
        Array1::from_iter(rows.iter().flatten().copied())
    }

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-12, "{a} != {b}");
    }

    #[test]
    fn one_mover_goes_to_its_site_and_the_rest_stay() {
        let atoms = flat(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 5.0, 5.0]]);
        let sites = flat(&[[3.0, 3.0, 3.0]]);
        let out = translate_onto_sites(atoms.view(), CUBE, ALL, &[1], sites.view()).unwrap();
        for ax in 0..3 {
            close(out.positions[3 + ax], 3.0);
        }
        assert_eq!(
            out.positions.slice(ndarray::s![0..3]),
            atoms.slice(ndarray::s![0..3])
        );
        assert_eq!(
            out.positions.slice(ndarray::s![6..9]),
            atoms.slice(ndarray::s![6..9])
        );
        assert_eq!(out.pairs, vec![(1, 0)]);
    }

    #[test]
    fn the_short_way_through_a_face_leaves_the_cell() {
        let atoms = flat(&[[0.5, 0.0, 0.0]]);
        let sites = flat(&[[9.5, 0.0, 0.0]]);
        let out = translate_onto_sites(atoms.view(), CUBE, ALL, &[0], sites.view()).unwrap();
        close(out.positions[0], -0.5);

        let atoms = flat(&[[9.0, 1.0, 9.5]]);
        let sites = flat(&[[0.5, 9.0, 0.5]]);
        let out = translate_onto_sites(atoms.view(), CUBE, ALL, &[0], sites.view()).unwrap();
        close(out.positions[0], 10.5);
        close(out.positions[1], -1.0);
        close(out.positions[2], 10.5);
    }

    #[test]
    fn each_mover_takes_the_nearest_site_left() {
        let cell = [[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]];
        let atoms = flat(&[[0.0, 0.0, 0.0], [9.0, 9.0, 9.0], [5.0, 5.0, 5.0]]);
        let sites = flat(&[[1.0, 0.0, 0.0], [9.5, 9.5, 9.5]]);
        let out = translate_onto_sites(atoms.view(), cell, ALL, &[0, 1], sites.view()).unwrap();
        close(out.positions[0], 1.0);
        close(out.positions[3], 9.5);
        assert_eq!(out.positions[6], 5.0);
        assert_eq!(out.pairs.len(), 2);
    }

    #[test]
    fn the_closest_pair_is_taken_first() {
        // Atom 0 is nearer the shared site than atom 1, so atom 1 gets the far one.
        let cell = [[40.0, 0.0, 0.0], [0.0, 40.0, 0.0], [0.0, 0.0, 40.0]];
        let atoms = flat(&[[1.0, 0.0, 0.0], [5.0, 0.0, 0.0]]);
        let sites = flat(&[[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]]);
        let out = translate_onto_sites(atoms.view(), cell, ALL, &[0, 1], sites.view()).unwrap();
        assert_eq!(out.pairs, vec![(0, 0), (1, 1)]);
        close(out.positions[0], 0.0);
        close(out.positions[3], 20.0);
    }

    #[test]
    fn a_surplus_mover_stays_and_a_surplus_site_is_unused() {
        let open_x = [false, true, true];
        let atoms = flat(&[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]);
        let sites = flat(&[[8.0, 0.0, 0.0]]);
        let out = translate_onto_sites(atoms.view(), CUBE, open_x, &[0, 1], sites.view()).unwrap();
        assert_eq!(out.positions[0], 0.0);
        close(out.positions[3], 8.0);

        let sites = flat(&[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
        let out = translate_onto_sites(atoms.view(), CUBE, open_x, &[0], sites.view()).unwrap();
        close(out.positions[0], 1.0);
        assert_eq!(out.positions[3], 4.0);
        assert_eq!(out.pairs, vec![(0, 0)]);
    }

    #[test]
    fn a_displacement_of_half_a_cell_keeps_its_sign() {
        let atoms = flat(&[[0.0, 0.0, 0.0]]);
        let plus = translate_onto_sites(
            atoms.view(),
            CUBE,
            ALL,
            &[0],
            flat(&[[5.0, 0.0, 0.0]]).view(),
        )
        .unwrap();
        close(plus.positions[0], 5.0);
        let minus = translate_onto_sites(
            atoms.view(),
            CUBE,
            ALL,
            &[0],
            flat(&[[-5.0, 0.0, 0.0]]).view(),
        )
        .unwrap();
        close(minus.positions[0], -5.0);
        // 25 is two and a half cells away and folds to +5; -25 folds to -5.
        let far = translate_onto_sites(
            atoms.view(),
            CUBE,
            ALL,
            &[0],
            flat(&[[25.0, -25.0, 0.0]]).view(),
        )
        .unwrap();
        close(far.positions[0], 5.0);
        close(far.positions[1], -5.0);
    }

    #[test]
    fn an_open_axis_is_not_folded() {
        let atoms = flat(&[[0.0, 0.0, 0.0]]);
        let open = [false, false, true];
        let out = translate_onto_sites(
            atoms.view(),
            CUBE,
            open,
            &[0],
            flat(&[[25.0, -4.0, 9.0]]).view(),
        )
        .unwrap();
        close(out.positions[0], 25.0);
        close(out.positions[1], -4.0);
        close(out.positions[2], -1.0);
    }

    #[test]
    fn a_sheared_cell_folds_along_its_lattice_vectors() {
        // b = (3, 10, 0). The site is one b away plus 0.5 Angstrom along x.
        let cell = [[10.0, 0.0, 0.0], [3.0, 10.0, 0.0], [0.0, 0.0, 10.0]];
        let atoms = flat(&[[0.0, 0.0, 0.0]]);
        let sites = flat(&[[3.5, 10.0, 0.0]]);
        let out = translate_onto_sites(atoms.view(), cell, ALL, &[0], sites.view()).unwrap();
        close(out.positions[0], 0.5);
        close(out.positions[1], 0.0);
    }

    #[test]
    fn bad_input_is_named() {
        let atoms = flat(&[[0.0; 3]]);
        let sites = flat(&[[1.0; 3]]);
        let at = |m: &[usize], s: &Array1<f64>| {
            translate_onto_sites(atoms.view(), CUBE, ALL, m, s.view())
        };
        assert!(matches!(at(&[], &sites), Err(SaddleError::Shape(_))));
        assert!(matches!(
            at(&[0], &Array1::zeros(0)),
            Err(SaddleError::Shape(_))
        ));
        assert!(matches!(at(&[5], &sites), Err(SaddleError::Shape(_))));
        assert!(matches!(at(&[0, 0], &sites), Err(SaddleError::Invalid(_))));
        assert!(matches!(
            at(&[0], &array![1.0, 2.0]),
            Err(SaddleError::Shape(_))
        ));
        assert!(matches!(
            at(&[0], &array![f64::NAN, 0.0, 0.0]),
            Err(SaddleError::NonFinite(_))
        ));
        let flat_cell = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 0.0]];
        let bad = translate_onto_sites(atoms.view(), flat_cell, ALL, &[0], sites.view());
        assert!(matches!(bad, Err(SaddleError::Invalid(_))));
        let nan = [[f64::NAN, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]];
        let bad = translate_onto_sites(atoms.view(), nan, ALL, &[0], sites.view());
        assert!(matches!(bad, Err(SaddleError::NonFinite(_))));
    }

    #[test]
    fn an_overflowing_distance_is_an_error_not_a_panic() {
        const OPEN: [bool; 3] = [false; 3];
        let huge = f64::MAX;
        let atoms = flat(&[[huge, 0.0, 0.0]]);
        let sites = flat(&[[-huge, 0.0, 0.0]]);
        for periodic in [ALL, OPEN] {
            let out = translate_onto_sites(atoms.view(), CUBE, periodic, &[0], sites.view());
            assert!(
                matches!(out, Err(SaddleError::NonFinite(_))),
                "{periodic:?}: {out:?}"
            );
        }
        // The difference is finite, but its square is not.
        let atoms = flat(&[[0.0, 0.0, 0.0]]);
        let sites = flat(&[[1e200, 0.0, 0.0]]);
        let out = translate_onto_sites(atoms.view(), CUBE, OPEN, &[0], sites.view());
        assert!(matches!(out, Err(SaddleError::NonFinite(_))), "{out:?}");
        // One overflowing pair among finite ones still refuses the placement.
        let atoms = flat(&[[0.0, 0.0, 0.0], [huge, 0.0, 0.0]]);
        let sites = flat(&[[1.0, 0.0, 0.0], [-huge, 0.0, 0.0]]);
        let out = translate_onto_sites(atoms.view(), CUBE, ALL, &[0, 1], sites.view());
        assert!(matches!(out, Err(SaddleError::NonFinite(_))), "{out:?}");
    }

    #[test]
    fn an_overflowing_push_is_an_error_not_infinity() {
        let huge = f64::MAX;
        let reactant = flat(&[[-huge, 0.0, 0.0]]);
        let saddle = flat(&[[huge, 0.0, 0.0]]);
        for (periodic, factor) in [(ALL, 0.5), ([false; 3], 0.5), (ALL, 0.0)] {
            let out = push_past(reactant.view(), saddle.view(), CUBE, periodic, factor);
            assert!(
                matches!(out, Err(SaddleError::NonFinite(_))),
                "{periodic:?} {factor}: {out:?}"
            );
        }
        // The displacement is finite; the push past the saddle is not.
        let reactant = flat(&[[0.0, 0.0, 0.0]]);
        let saddle = flat(&[[huge, 0.0, 0.0]]);
        let out = push_past(reactant.view(), saddle.view(), CUBE, [false; 3], 1.0);
        assert!(matches!(out, Err(SaddleError::NonFinite(_))), "{out:?}");
    }

    #[test]
    fn a_push_of_zero_is_the_saddle_and_one_doubles_the_step() {
        let reactant = flat(&[[0.0, 0.0, 0.0]]);
        let saddle = flat(&[[1.0, 0.0, 0.0]]);
        let zero = push_past(reactant.view(), saddle.view(), CUBE, ALL, 0.0).unwrap();
        assert_eq!(zero, saddle);
        let one = push_past(reactant.view(), saddle.view(), CUBE, ALL, 1.0).unwrap();
        close(one[0], 2.0);
        let back = push_past(reactant.view(), saddle.view(), CUBE, ALL, -1.0).unwrap();
        close(back[0], 0.0);
    }

    #[test]
    fn a_push_takes_the_short_way_across_a_face() {
        let reactant = flat(&[[0.5, 0.0, 0.0]]);
        let saddle = flat(&[[9.5, 0.0, 0.0]]);
        let out = push_past(reactant.view(), saddle.view(), CUBE, ALL, 1.0).unwrap();
        close(out[0], 8.5);

        let reactant = flat(&[[9.0, 0.5, 1.0]]);
        let saddle = flat(&[[0.5, 9.0, 1.0]]);
        let out = push_past(reactant.view(), saddle.view(), CUBE, ALL, 1.0).unwrap();
        close(out[0], 2.0);
        close(out[1], 7.5);
        close(out[2], 1.0);
    }

    #[test]
    fn a_push_does_not_fold_an_open_axis() {
        let reactant = flat(&[[0.0, 0.0, 0.0]]);
        let saddle = flat(&[[8.0, 1.0, 0.0]]);
        let out = push_past(
            reactant.view(),
            saddle.view(),
            CUBE,
            [false, true, true],
            1.0,
        )
        .unwrap();
        close(out[0], 16.0);
        close(out[1], 2.0);
    }

    #[test]
    fn a_push_folds_by_more_than_one_cell_and_keeps_the_sign_at_half() {
        let reactant = flat(&[[0.0, 0.0, 0.0]]);
        let saddle = flat(&[[-25.0, 0.0, 0.0]]);
        let out = push_past(reactant.view(), saddle.view(), CUBE, ALL, 1.0).unwrap();
        close(out[0], -30.0);
    }

    #[test]
    fn empty_geometries_give_an_empty_push_and_bad_input_is_named() {
        let empty = Array1::<f64>::zeros(0);
        let out = push_past(empty.view(), empty.view(), CUBE, ALL, 1.0).unwrap();
        assert!(out.is_empty());
        let one = flat(&[[0.0; 3]]);
        let two = flat(&[[0.0; 3], [1.0; 3]]);
        assert!(matches!(
            push_past(one.view(), two.view(), CUBE, ALL, 0.5),
            Err(SaddleError::Shape(_))
        ));
        assert!(matches!(
            push_past(
                array![0.0, 0.0].view(),
                array![0.0, 0.0].view(),
                CUBE,
                ALL,
                0.5
            ),
            Err(SaddleError::Shape(_))
        ));
        for factor in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(matches!(
                push_past(one.view(), one.view(), CUBE, ALL, factor),
                Err(SaddleError::NonFinite(_))
            ));
        }
        assert!(matches!(
            push_past(
                array![f64::NAN, 0.0, 0.0].view(),
                one.view(),
                CUBE,
                ALL,
                0.5
            ),
            Err(SaddleError::NonFinite(_))
        ));
    }
}
