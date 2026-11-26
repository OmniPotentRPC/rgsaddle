//! Point defects by Wigner–Seitz site occupancy.
//!
//! Each atom belongs to the reference site nearest to it under the
//! minimum-image convention, which is the Wigner–Seitz cell of that site.
//! An empty site is a vacancy. A site holding `n > 1` atoms carries
//! `n - 1` interstitials. A configuration with as many atoms as sites and
//! every site singly occupied holds no point defect, whatever the local
//! distortion: a vacancy and an interstitial have recombined.
//!
//! Nordlund, Ghaly, Averback, Caturla, Diaz de la Rubia and Tarus,
//! Phys. Rev. B 57, 7556 (1998), doi:10.1103/PhysRevB.57.7556.
//!
//! Positions, sites and the cell are in Angstrom. The analysis has no
//! distance threshold: every atom goes to its nearest site.

use ndarray::ArrayView1;

use crate::cell_log::{M3, m3_det, m3_inv};
use crate::error::SaddleError;

/// Occupancy of every reference site.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SiteOccupancy {
    /// Atoms assigned to each site, in site order; one entry per reference
    /// site.
    pub counts: Vec<usize>,
    /// Index of the site each atom was assigned to, in atom order; one
    /// entry per atom.
    pub site_of: Vec<usize>,
}

impl SiteOccupancy {
    /// Indices of the sites with no atom (the vacancies), in site order.
    pub fn vacancies(&self) -> Vec<usize> {
        (0..self.counts.len())
            .filter(|&site| self.counts[site] == 0)
            .collect()
    }

    /// Indices of the sites with more than one atom, in site order.
    pub fn crowded(&self) -> Vec<usize> {
        (0..self.counts.len())
            .filter(|&site| self.counts[site] > 1)
            .collect()
    }

    /// Number of empty sites, the vacancy count.
    pub fn vacancy_count(&self) -> usize {
        self.counts.iter().filter(|&&n| n == 0).count()
    }

    /// `Σ max(n - 1, 0)` over the sites.
    pub fn interstitial_count(&self) -> usize {
        self.counts.iter().map(|&n| n.saturating_sub(1)).sum()
    }

    /// Whether every site holds exactly one atom. This is false whenever a
    /// vacancy and an interstitial coexist, and true after they recombine.
    pub fn is_perfect(&self) -> bool {
        self.counts.iter().all(|&n| n == 1)
    }
}

/// Assign every atom of `positions` to its nearest site of `sites`.
///
/// Both are flat Cartesian `(x, y, z)` triples in the periodic `cell`,
/// whose rows are the lattice vectors. Sites are binned on a fractional
/// grid, so the cost grows with the number of atoms, not with the
/// product of atoms and sites.
///
/// `sites` has length `3 * n_sites` and `positions` length `3 * n_atoms`,
/// both atom-major `[x0, y0, z0, x1, ...]` in Angstrom. `cell` is the
/// `[[ax, ay, az], [bx, by, bz], [cx, cy, cz]]` lattice, one vector per row,
/// in Angstrom, and every direction is periodic. Distances use the minimum
/// image. A tie goes to the site with the lower index.
///
/// # Errors
///
/// - [`SaddleError::Shape`] when `sites` is empty, or `sites` or `positions`
///   is not a whole number of triples.
/// - [`SaddleError::NonFinite`] when a coordinate is NaN or infinite.
/// - [`SaddleError::Invalid`] when the cell has zero volume, and the error
///   of [`crate::cell_log::m3_inv`] when it cannot be inverted.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use rgsaddle::site_occupancy;
///
/// // Two sites in a 4 Angstrom cubic cell; one atom, near the first site.
/// let cell = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]];
/// let sites = array![0.0, 0.0, 0.0, 2.0, 2.0, 2.0];
/// let atoms = array![0.1, 0.0, 0.0];
/// let occupancy = site_occupancy(sites.view(), atoms.view(), cell)?;
/// assert_eq!(occupancy.counts, vec![1, 0]);
/// assert_eq!(occupancy.vacancies(), vec![1]);
/// # Ok::<(), rgsaddle::SaddleError>(())
/// ```
pub fn site_occupancy(
    sites: ArrayView1<f64>,
    positions: ArrayView1<f64>,
    cell: M3,
) -> Result<SiteOccupancy, SaddleError> {
    if sites.is_empty() || !sites.len().is_multiple_of(3) || !positions.len().is_multiple_of(3) {
        return Err(SaddleError::Shape(
            "sites and positions must hold whole (x, y, z) triples, with at least one site".into(),
        ));
    }
    if sites.iter().chain(positions.iter()).any(|v| !v.is_finite()) {
        return Err(SaddleError::NonFinite("site occupancy coordinates"));
    }
    let volume = m3_det(cell).abs();
    if !(volume.is_finite() && volume > 0.0) {
        return Err(SaddleError::Invalid("the cell must have a volume".into()));
    }
    let inv = m3_inv(cell)?;
    let images = Images::new(cell, [true; 3])?;
    let n_sites = sites.len() / 3;
    let widths = plane_spacings(cell, volume);
    let spacing = (volume / n_sites as f64).cbrt();
    let mut bins = [1usize; 3];
    for k in 0..3 {
        bins[k] = ((widths[k] / spacing).floor() as usize).max(1);
    }
    let bin_width = (0..3)
        .map(|k| widths[k] / bins[k] as f64)
        .fold(f64::INFINITY, f64::min);
    let total = bins[0] * bins[1] * bins[2];
    let mut grid: Vec<Vec<usize>> = vec![Vec::new(); total];
    let mut site_frac = Vec::with_capacity(n_sites);
    for site in 0..n_sites {
        let s = wrap(fractional(point(sites, site), inv));
        grid[bin_index(bin_of(s, bins), bins)].push(site);
        site_frac.push(s);
    }
    let n_atoms = positions.len() / 3;
    let mut counts = vec![0usize; n_sites];
    let mut site_of = Vec::with_capacity(n_atoms);
    let reach = bins.iter().copied().max().unwrap_or(1).div_ceil(2);
    for atom in 0..n_atoms {
        let s = wrap(fractional(point(positions, atom), inv));
        let home = bin_of(s, bins);
        let mut best = (f64::INFINITY, usize::MAX);
        for ring in 0..=reach {
            visit_ring(home, ring, bins, |bin| {
                for &site in &grid[bin] {
                    let d2 = images.distance2(sub(s, site_frac[site]));
                    if d2 < best.0 || (d2 == best.0 && site < best.1) {
                        best = (d2, site);
                    }
                }
            });
            if best.1 != usize::MAX && best.0.sqrt() <= ring as f64 * bin_width {
                break;
            }
        }
        counts[best.1] += 1;
        site_of.push(best.1);
    }
    Ok(SiteOccupancy { counts, site_of })
}

fn point(flat: ArrayView1<f64>, index: usize) -> [f64; 3] {
    [flat[3 * index], flat[3 * index + 1], flat[3 * index + 2]]
}

fn fractional(r: [f64; 3], inv: M3) -> [f64; 3] {
    let mut s = [0.0; 3];
    for (i, value) in s.iter_mut().enumerate() {
        *value = r[0] * inv[0][i] + r[1] * inv[1][i] + r[2] * inv[2][i];
    }
    s
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn wrap(s: [f64; 3]) -> [f64; 3] {
    s.map(|v| {
        let w = v - v.floor();
        if w >= 1.0 { 0.0 } else { w }
    })
}

fn bin_of(s: [f64; 3], bins: [usize; 3]) -> [usize; 3] {
    let mut out = [0usize; 3];
    for k in 0..3 {
        out[k] = ((s[k] * bins[k] as f64) as usize).min(bins[k] - 1);
    }
    out
}

fn bin_index(b: [usize; 3], bins: [usize; 3]) -> usize {
    (b[0] * bins[1] + b[1]) * bins[2] + b[2]
}

/// Every bin whose periodic Chebyshev distance from `home` is `ring`,
/// each visited once.
fn visit_ring(home: [usize; 3], ring: usize, bins: [usize; 3], mut visit: impl FnMut(usize)) {
    let r = ring as i64;
    let mut seen = std::collections::HashSet::new();
    for dx in -r..=r {
        for dy in -r..=r {
            for dz in -r..=r {
                if dx.abs().max(dy.abs()).max(dz.abs()) != r {
                    continue;
                }
                let mut b = [0usize; 3];
                for (k, d) in [dx, dy, dz].into_iter().enumerate() {
                    b[k] = (home[k] as i64 + d).rem_euclid(bins[k] as i64) as usize;
                }
                let index = bin_index(b, bins);
                if seen.insert(index) {
                    visit(index);
                }
            }
        }
    }
}

/// Shortest lattice image of a displacement, taken along the periodic axes
/// of a cell.
///
/// Rounding the fractional coordinates is the minimum image only when the
/// lattice vectors are orthogonal. In a skewed cell the nearest image can
/// sit one or more steps past the rounded one, so the neighbouring images
/// are searched. The search is exact: an image whose coefficient along a
/// periodic vector `a_k` differs from the rounded one by `m` is at least
/// `(|m| - 1/2) h_k` long, where `h_k` is the spacing of the lattice planes
/// across `a_k`, so only offsets with `|m| <= |v| / h_k + 1/2` can beat the
/// rounded image `v`.
///
/// A component along an open axis is never folded. Only the part of a
/// displacement in the span of the periodic vectors changes between images,
/// so the rounding and the search use the coefficients of its projection
/// onto that span.
pub(crate) struct Images {
    cell: M3,
    periodic: [bool; 3],
    /// Inverse Gram matrix of the periodic vectors, with zero rows and
    /// columns for the open axes.
    gram_inv: M3,
    /// Plane spacing `h_k` within the periodic sublattice; unused on an
    /// open axis.
    spacing: [f64; 3],
    diagonal: bool,
    orthogonal: bool,
}

impl Images {
    /// The cell must have a volume, whichever axes are periodic.
    pub(crate) fn new(cell: M3, periodic: [bool; 3]) -> Result<Self, SaddleError> {
        let volume = m3_det(cell).abs();
        if !(volume.is_finite() && volume > 0.0) {
            return Err(SaddleError::Invalid("the cell must have a volume".into()));
        }
        let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let mut gram = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                gram[i][j] = if periodic[i] && periodic[j] {
                    dot(cell[i], cell[j])
                } else if i == j {
                    1.0
                } else {
                    0.0
                };
            }
        }
        let mut gram_inv = m3_inv(gram)?;
        let mut spacing = [0.0; 3];
        for i in 0..3 {
            for j in 0..3 {
                if !(periodic[i] && periodic[j]) {
                    gram_inv[i][j] = 0.0;
                }
            }
            if periodic[i] {
                spacing[i] = 1.0 / gram_inv[i][i].sqrt();
            }
        }
        let diagonal = (0..3).all(|i| (0..3).all(|j| i == j || cell[i][j] == 0.0));
        let orthogonal = (0..3).all(|i| (0..i).all(|j| dot(cell[i], cell[j]) == 0.0));
        Ok(Self {
            cell,
            periodic,
            gram_inv,
            spacing,
            diagonal,
            orthogonal,
        })
    }

    /// `d` moved by whole periodic lattice vectors to its shortest image.
    ///
    /// In an orthogonal cell each periodic coefficient is rounded into
    /// `[-1/2, 1/2]`, half toward zero, so a displacement of exactly half a
    /// cell keeps its sign. In any other cell the neighbouring images are
    /// searched, and a tie keeps the rounded image.
    pub(crate) fn displacement(&self, d: [f64; 3]) -> [f64; 3] {
        if !self.periodic.iter().any(|&p| p) {
            return d;
        }
        let mut whole = [0.0; 3];
        if self.diagonal {
            for ax in 0..3 {
                if self.periodic[ax] {
                    whole[ax] = shift(d[ax] / self.cell[ax][ax]);
                }
            }
        } else {
            let mut along = [0.0; 3];
            for (k, value) in along.iter_mut().enumerate() {
                *value = (0..3).map(|c| self.cell[k][c] * d[c]).sum();
            }
            let mut rest = [0.0; 3];
            for k in 0..3 {
                let g: f64 = (0..3).map(|j| self.gram_inv[k][j] * along[j]).sum();
                whole[k] = shift(g);
                rest[k] = g - whole[k];
            }
            if !self.orthogonal {
                let (offset, _) = self.nearest(rest);
                for k in 0..3 {
                    whole[k] += offset[k];
                }
            }
        }
        let mut out = d;
        for (c, value) in out.iter_mut().enumerate() {
            *value -= whole[0] * self.cell[0][c]
                + whole[1] * self.cell[1][c]
                + whole[2] * self.cell[2][c];
        }
        out
    }

    /// Squared length of the shortest image of the fractional displacement
    /// `ds`, every axis taken as periodic.
    fn distance2(&self, ds: [f64; 3]) -> f64 {
        self.nearest(ds.map(|s| s - shift(s))).1
    }

    /// The whole-cell offset `m` that makes `Σ (rest_k - m_k) a_k` shortest
    /// over the periodic axes, and that squared length. `rest` holds
    /// coefficients already rounded into `[-1/2, 1/2]`; the zero offset wins
    /// a tie, then the first offset in lexicographic order.
    fn nearest(&self, rest: [f64; 3]) -> ([f64; 3], f64) {
        let length2 = |m: [f64; 3]| {
            let mut d2 = 0.0;
            for c in 0..3 {
                let mut x = 0.0;
                for k in 0..3 {
                    if self.periodic[k] {
                        x += (rest[k] - m[k]) * self.cell[k][c];
                    }
                }
                d2 += x * x;
            }
            d2
        };
        let mut best = ([0.0; 3], length2([0.0; 3]));
        if !best.1.is_finite() {
            return best;
        }
        let length = best.1.sqrt();
        let reach: [i64; 3] = std::array::from_fn(|k| {
            if self.periodic[k] {
                (length / self.spacing[k] + 0.5).floor() as i64
            } else {
                0
            }
        });
        for mx in -reach[0]..=reach[0] {
            for my in -reach[1]..=reach[1] {
                for mz in -reach[2]..=reach[2] {
                    let m = [mx as f64, my as f64, mz as f64];
                    let d2 = length2(m);
                    if d2 < best.1 {
                        best = (m, d2);
                    }
                }
            }
        }
        best
    }
}

/// Whole cells to subtract from the fraction `s`, rounding half toward zero.
fn shift(s: f64) -> f64 {
    let steps = (s.abs() - 0.5).ceil().max(0.0);
    steps.copysign(s)
}

/// Distance between opposite faces of the cell, one per lattice vector.
fn plane_spacings(cell: M3, volume: f64) -> [f64; 3] {
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let norm = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [
        volume / norm(cross(cell[1], cell[2])),
        volume / norm(cross(cell[2], cell[0])),
        volume / norm(cross(cell[0], cell[1])),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn simple_cubic(n: usize, a: f64) -> (Array1<f64>, M3) {
        let mut values = Vec::new();
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    values.extend([i as f64 * a, j as f64 * a, k as f64 * a]);
                }
            }
        }
        let l = n as f64 * a;
        (
            Array1::from(values),
            [[l, 0.0, 0.0], [0.0, l, 0.0], [0.0, 0.0, l]],
        )
    }

    #[test]
    fn a_displaced_lattice_is_perfect() {
        let (sites, cell) = simple_cubic(5, 2.0);
        let mut atoms = sites.clone();
        for (i, v) in atoms.iter_mut().enumerate() {
            *v += 0.3 * ((i as f64) * 0.7).sin();
        }
        let occupancy = site_occupancy(sites.view(), atoms.view(), cell).unwrap();
        assert!(occupancy.is_perfect());
        assert_eq!(occupancy.vacancy_count(), 0);
        assert_eq!(occupancy.interstitial_count(), 0);
    }

    #[test]
    fn a_frenkel_pair_is_one_vacancy_and_one_interstitial() {
        let (sites, cell) = simple_cubic(5, 2.0);
        let mut atoms = sites.clone();
        // Atom 0 leaves its site for a point next to site 62, across the
        // periodic boundary from where it started.
        let target = [sites[3 * 62] + 0.4, sites[3 * 62 + 1], sites[3 * 62 + 2]];
        for c in 0..3 {
            atoms[c] = target[c] - 10.0;
        }
        let occupancy = site_occupancy(sites.view(), atoms.view(), cell).unwrap();
        assert_eq!(occupancy.vacancies(), vec![0]);
        assert_eq!(occupancy.crowded(), vec![62]);
        assert_eq!(occupancy.interstitial_count(), 1);
        assert!(!occupancy.is_perfect());
    }

    #[test]
    fn a_sheared_cell_matches_the_brute_force_assignment() {
        let (sites, mut cell) = simple_cubic(4, 2.5);
        cell[1][0] = 3.0;
        let atoms = sites.mapv(|v| v * 1.01 + 0.37);
        let occupancy = site_occupancy(sites.view(), atoms.view(), cell).unwrap();
        let inv = m3_inv(cell).unwrap();
        let images = Images::new(cell, [true; 3]).unwrap();
        for (atom, &site_of) in occupancy.site_of.iter().enumerate() {
            let s = fractional(point(atoms.view(), atom), inv);
            let mut best = (f64::INFINITY, 0);
            for site in 0..sites.len() / 3 {
                let d2 = images.distance2(sub(s, fractional(point(sites.view(), site), inv)));
                if d2 < best.0 {
                    best = (d2, site);
                }
            }
            assert_eq!(site_of, best.1, "atom {atom}");
        }
    }

    #[test]
    fn malformed_input_is_refused() {
        let (sites, cell) = simple_cubic(2, 2.0);
        assert!(matches!(
            site_occupancy(sites.view(), sites.slice(ndarray::s![..4]), cell),
            Err(SaddleError::Shape(_))
        ));
        let flat = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]];
        assert!(matches!(
            site_occupancy(sites.view(), sites.view(), flat),
            Err(SaddleError::Invalid(_))
        ));
    }
}
