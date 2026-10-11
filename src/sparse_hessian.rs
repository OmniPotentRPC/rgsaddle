//! Sparse Hessian by a direct Curtis-Powell-Reid partition.
//!
//! The atom graph is the sparsity pattern of the Hessian blocks.
//! [`distance2_colouring`] is a greedy smallest-last colouring of that
//! graph: two atoms share a colour only when they are neither neighbours
//! nor joined by a common neighbour. Displacing every atom of one colour
//! along one Cartesian axis then recovers each affected 3x3 block from a
//! single gradient difference (Curtis, Powell, and Reid, IMA J. Appl.
//! Math. 13, 117 (1974), <https://doi.org/10.1093/imamat/13.1.117>).
//! The blocks are symmetrised as `(H + H^T) / 2`.
//!
//! Powell and Toint's symmetric substitution (SIAM J. Numer. Anal. 16,
//! 1060 (1979), <https://doi.org/10.1137/0716078>) is not used. Star
//! colouring is not used. The distance-2 rule is the direct partition
//! in Gebremedhin, Manne, and Pothen, SIAM Rev. 47, 629 (2005),
//! <https://doi.org/10.1137/S0036144504444711>.
//!
//! [`sparse_hessian`] packs a block of colours into one
//! [`eval_displacements`] call. Colours in the block are independent and
//! run together. After the batch returns, each colour's blocks are
//! written back in colour index order. The block holds as many colours
//! as [`SparseHessianOpts::memory_budget`] can store, and every colour
//! when the whole problem fits. The budget is not the thread count.
//! Chunks hold 4 geometries, independent of the thread count.
//! [`SparseHessian`] implements [`rgmin::ApplyHessian`] and exports the
//! same matrix through [`SparseHessian::to_dense`]. The matrix is frozen
//! at the recovery point.

use ndarray::{Array1, Array2, ArrayView1};
use rgmin::ApplyHessian;
use rgmin::vecops::{axpy, dot};

use crate::eigensolve::exact_eigh;
use crate::error::SaddleError;
use crate::linalg::modified_gram_schmidt;
use crate::mic::Cell;
use crate::minmode::{FiniteDifference, PointSurface};

/// Geometries per parallel chunk. Independent of the rayon thread count.
const GEOM_CHUNK: usize = 4;

#[cfg(test)]
thread_local! {
    static BATCHES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static PHASES: std::cell::Cell<Option<PhaseRecord>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default)]
struct PhaseRecord {
    paint: f64,
    eval: f64,
    write: f64,
    sym: f64,
}

/// Full Lanczos basis at or below this dimension, so an index check on a
/// small Hessian matches a dense eigenpair.
const KRYLOV_FULL_MAX: usize = 96;

/// Default bytes for one colour block's geometry and gradient buffers.
const DEFAULT_MEMORY_BUDGET: usize = 256 * 1024 * 1024;

/// Relative residual `||H y - theta y|| / ||H||_est` that locks a Ritz pair.
const LANCZOS_TOL: f64 = 1e-8;

/// Atom-level symmetric sparsity pattern.
///
/// Off-diagonal only. The diagonal block is always present in the
/// Hessian and is not stored as a loop: a periodic image of an atom is
/// the same atom, so a cell smaller than the cutoff does not add an
/// edge. Column indices in each row are sorted and unique.
///
/// `interaction_range` passed to [`SparsityPattern::from_positions`] is
/// `r_c` for a pair potential (the Hessian block vanishes for `r > r_c`),
/// `2 r_c` for an atom-centred many-body model, and `r_c` times the
/// number of layers for a message-passing model.
#[derive(Clone, Debug)]
pub struct SparsityPattern {
    n_atoms: usize,
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
}

impl SparsityPattern {
    /// Number of atoms.
    pub fn n_atoms(&self) -> usize {
        self.n_atoms
    }

    /// CSR row pointers, length `n_atoms + 1`.
    pub fn row_ptr(&self) -> &[usize] {
        &self.row_ptr
    }

    /// Column indices of off-diagonal neighbours.
    pub fn col_idx(&self) -> &[usize] {
        &self.col_idx
    }

    /// Neighbours of `atom`, in ascending index order.
    pub fn neighbors(&self, atom: usize) -> &[usize] {
        let start = self.row_ptr[atom];
        let end = self.row_ptr[atom + 1];
        &self.col_idx[start..end]
    }

    /// Build a pattern from undirected pairs.
    ///
    /// A pair `(i, i)` is dropped. Each stored edge is written in both
    /// directions. Duplicate pairs are stored once.
    pub fn from_adjacency(n_atoms: usize, pairs: &[(usize, usize)]) -> Result<Self, SaddleError> {
        let mut rows = vec![Vec::new(); n_atoms];
        for &(i, j) in pairs {
            if i == j {
                continue;
            }
            if i >= n_atoms || j >= n_atoms {
                return Err(SaddleError::Invalid(
                    "adjacency endpoint out of range".into(),
                ));
            }
            rows[i].push(j);
            rows[j].push(i);
        }
        Ok(pack_rows(rows))
    }

    /// Pairs within `interaction_range + skin`.
    ///
    /// The search uses a linkcell binning on [`Cell`] primitives
    /// (`fractional`, `widths`, `lattice_shift`, `dist2`,
    /// `dist2_shifted`). The pinned linkcell crate exposes k-nearest
    /// queries, not a cutoff pair list, so the cutoff walk lives here.
    /// An open cluster (`cell == None`) is binned in Cartesian space.
    /// The range is inclusive. A periodic image of an atom is not stored
    /// as a loop. Coordinates are Cartesian, atom-major.
    ///
    /// `skin` is an extra pad. [`SparseHessianOpts::search_skin`] is
    /// `max(2 * step, skin)`, so a neighbour that is still outside the
    /// interaction range cannot enter it while one atom of a probe moves
    /// by `step` and the other stays put.
    pub fn from_positions(
        positions: ArrayView1<f64>,
        cell: Option<&Cell>,
        interaction_range: f64,
        skin: f64,
    ) -> Result<Self, SaddleError> {
        if !(interaction_range.is_finite() && interaction_range > 0.0) {
            return Err(SaddleError::Invalid(
                "interaction range must be positive and finite".into(),
            ));
        }
        if !(skin.is_finite() && skin >= 0.0) {
            return Err(SaddleError::Invalid(
                "sparsity skin must be non-negative and finite".into(),
            ));
        }
        if !positions.len().is_multiple_of(3) {
            return Err(SaddleError::Shape(
                "positions must hold 3 coordinates per atom".into(),
            ));
        }
        if positions.iter().any(|v| !v.is_finite()) {
            return Err(SaddleError::NonFinite("positions"));
        }
        let range = interaction_range + skin;
        let n_atoms = positions.len() / 3;
        let mut rows = vec![Vec::new(); n_atoms];
        let points = coords_of(positions);
        visit_cutoff(&points, cell, range, |i, j, _dr| {
            rows[i].push(j);
            rows[j].push(i);
        });
        Ok(pack_rows(rows))
    }
}

fn coords_at(positions: ArrayView1<f64>, atom: usize) -> [f64; 3] {
    [
        positions[3 * atom],
        positions[3 * atom + 1],
        positions[3 * atom + 2],
    ]
}

fn coords_of(positions: ArrayView1<f64>) -> Vec<[f64; 3]> {
    match positions.as_slice() {
        Some(raw) => raw.as_chunks::<3>().0.to_vec(),
        None => (0..positions.len() / 3)
            .map(|atom| coords_at(positions, atom))
            .collect(),
    }
}

/// Every image of an unordered pair whose Cartesian distance is at most
/// `range`. `dr` points from `i` to that image of `j`, with `i < j`.
/// A lattice image of one atom is not reported.
fn visit_cutoff(
    points: &[[f64; 3]],
    cell: Option<&Cell>,
    range: f64,
    mut visit: impl FnMut(usize, usize, [f64; 3]),
) {
    let n = points.len();
    if n < 2 || !(range.is_finite() && range > 0.0) {
        return;
    }
    match cell {
        None => visit_open(points, range, &mut visit),
        Some(cell) if ortho_single_image(cell, range) => {
            visit_ortho_mic(points, cell, range, &mut visit);
        }
        Some(cell) if n <= 64 => visit_brute_images(points, cell, range, &mut visit),
        Some(cell) => visit_binned_images(points, cell, range, &mut visit),
    }
}

fn ortho_single_image(cell: &Cell, range: f64) -> bool {
    if !cell.is_ortho() {
        return false;
    }
    let widths = cell.widths();
    widths.iter().all(|width| *width > 2.0 * range)
}

fn image_reach(cell: &Cell, range: f64) -> [i32; 3] {
    let widths = cell.widths();
    let mut reach = [1i32; 3];
    for axis in 0..3 {
        let width = widths[axis].abs().max(1e-15);
        let images = (range / width).ceil();
        let images = if images.is_finite() { images as i32 } else { 1 };
        // One extra shell covers a skewed metric past the perpendicular bound.
        reach[axis] = images.saturating_add(1).max(1);
    }
    reach
}

fn visit_brute_images(
    points: &[[f64; 3]],
    cell: &Cell,
    range: f64,
    visit: &mut impl FnMut(usize, usize, [f64; 3]),
) {
    let range2 = range * range;
    let reach = image_reach(cell, range);
    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            emit_images(points, cell, i, j, reach, range2, visit);
        }
    }
}

fn emit_images(
    points: &[[f64; 3]],
    cell: &Cell,
    i: usize,
    j: usize,
    reach: [i32; 3],
    range2: f64,
    visit: &mut impl FnMut(usize, usize, [f64; 3]),
) {
    for na in -reach[0]..=reach[0] {
        for nb in -reach[1]..=reach[1] {
            for nc in -reach[2]..=reach[2] {
                let shift = cell.lattice_shift(na, nb, nc);
                let dr = [
                    points[j][0] + shift[0] - points[i][0],
                    points[j][1] + shift[1] - points[i][1],
                    points[j][2] + shift[2] - points[i][2],
                ];
                let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                if d2 > 0.0 && d2 <= range2 {
                    visit(i, j, dr);
                }
            }
        }
    }
}

fn visit_open(points: &[[f64; 3]], range: f64, visit: &mut impl FnMut(usize, usize, [f64; 3])) {
    let n = points.len();
    let mut lo = [f64::MAX; 3];
    let mut hi = [f64::MIN; 3];
    for point in points {
        for axis in 0..3 {
            lo[axis] = lo[axis].min(point[axis]);
            hi[axis] = hi[axis].max(point[axis]);
        }
    }
    let inv = 1.0 / range;
    let dims = [0, 1, 2].map(|axis| {
        let span = ((hi[axis] - lo[axis]) * inv).floor();
        if span.is_finite() && span >= 0.0 {
            (span as usize).saturating_add(1).clamp(1, 4096)
        } else {
            1
        }
    });
    let bins = fill_bins(n, dims, |atom| {
        let point = points[atom];
        [0, 1, 2].map(|axis| {
            let slot = ((point[axis] - lo[axis]) * inv).floor();
            if slot.is_finite() && slot >= 0.0 {
                (slot as usize).min(dims[axis] - 1)
            } else {
                0
            }
        })
    });
    let range2 = range * range;
    for i in 0..n {
        let origin = bin_of_open(points[i], lo, inv, dims);
        for dz in -1isize..=1 {
            for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    let x = origin[0] as isize + dx;
                    let y = origin[1] as isize + dy;
                    let z = origin[2] as isize + dz;
                    if x < 0 || y < 0 || z < 0 {
                        continue;
                    }
                    let x = x as usize;
                    let y = y as usize;
                    let z = z as usize;
                    if x >= dims[0] || y >= dims[1] || z >= dims[2] {
                        continue;
                    }
                    let slot = bin_slot([x, y, z], dims);
                    for &j in &bins[slot] {
                        if j <= i {
                            continue;
                        }
                        let dr = [
                            points[j][0] - points[i][0],
                            points[j][1] - points[i][1],
                            points[j][2] - points[i][2],
                        ];
                        let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                        if d2 > 0.0 && d2 <= range2 {
                            visit(i, j, dr);
                        }
                    }
                }
            }
        }
    }
}

fn bin_of_open(point: [f64; 3], lo: [f64; 3], inv: f64, dims: [usize; 3]) -> [usize; 3] {
    [0, 1, 2].map(|axis| {
        let slot = ((point[axis] - lo[axis]) * inv).floor();
        if slot.is_finite() && slot >= 0.0 {
            (slot as usize).min(dims[axis] - 1)
        } else {
            0
        }
    })
}

fn visit_ortho_mic(
    points: &[[f64; 3]],
    cell: &Cell,
    range: f64,
    visit: &mut impl FnMut(usize, usize, [f64; 3]),
) {
    let n = points.len();
    let widths = cell.widths();
    let dims = [0, 1, 2].map(|axis| (widths[axis] / range).floor().max(1.0) as usize);
    let frac: Vec<[f64; 3]> = points
        .iter()
        .copied()
        .map(|point| cell.fractional(point))
        .collect();
    let bins = fill_bins(n, dims, |atom| {
        [0, 1, 2].map(|axis| {
            let slot = (frac[atom][axis] * dims[axis] as f64).floor();
            if slot.is_finite() && slot >= 0.0 {
                (slot as usize) % dims[axis]
            } else {
                0
            }
        })
    });
    let range2 = range * range;
    for i in 0..n {
        let origin = [0, 1, 2].map(|axis| {
            let slot = (frac[i][axis] * dims[axis] as f64).floor();
            if slot.is_finite() && slot >= 0.0 {
                (slot as usize) % dims[axis]
            } else {
                0
            }
        });
        for dz in -1isize..=1 {
            for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    let slot = bin_slot(
                        [
                            wrap_bin(origin[0] as isize + dx, dims[0]),
                            wrap_bin(origin[1] as isize + dy, dims[1]),
                            wrap_bin(origin[2] as isize + dz, dims[2]),
                        ],
                        dims,
                    );
                    for &j in &bins[slot] {
                        if j <= i {
                            continue;
                        }
                        let dr = cell.displacement(points[i], points[j]);
                        let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                        if d2 > 0.0 && d2 <= range2 {
                            visit(i, j, dr);
                        }
                    }
                }
            }
        }
    }
}

fn visit_binned_images(
    points: &[[f64; 3]],
    cell: &Cell,
    range: f64,
    visit: &mut impl FnMut(usize, usize, [f64; 3]),
) {
    let n = points.len();
    let widths = cell.widths();
    let dims = [0, 1, 2].map(|axis| {
        let bins = (widths[axis] / range).floor();
        if bins.is_finite() && bins >= 1.0 {
            bins as usize
        } else {
            1
        }
    });
    let stencil = [0, 1, 2].map(|axis| {
        let width = widths[axis].abs().max(1e-15);
        let frac_radius = range / width;
        let bins = (frac_radius * dims[axis] as f64).ceil();
        let bins = if bins.is_finite() { bins as isize } else { 1 };
        bins.saturating_add(1).clamp(1, dims[axis] as isize)
    });
    let frac: Vec<[f64; 3]> = points
        .iter()
        .copied()
        .map(|point| cell.fractional(point))
        .collect();
    let bins = fill_bins(n, dims, |atom| {
        [0, 1, 2].map(|axis| {
            let slot = (frac[atom][axis] * dims[axis] as f64).floor();
            if slot.is_finite() && slot >= 0.0 {
                (slot as usize) % dims[axis]
            } else {
                0
            }
        })
    });
    let reach = image_reach(cell, range);
    let range2 = range * range;
    for (i, frac_i) in frac.iter().enumerate() {
        let origin = [0, 1, 2].map(|axis| {
            let slot = (frac_i[axis] * dims[axis] as f64).floor();
            if slot.is_finite() && slot >= 0.0 {
                (slot as usize) % dims[axis]
            } else {
                0
            }
        });
        for dz in -stencil[2]..=stencil[2] {
            for dy in -stencil[1]..=stencil[1] {
                for dx in -stencil[0]..=stencil[0] {
                    let slot = bin_slot(
                        [
                            wrap_bin(origin[0] as isize + dx, dims[0]),
                            wrap_bin(origin[1] as isize + dy, dims[1]),
                            wrap_bin(origin[2] as isize + dz, dims[2]),
                        ],
                        dims,
                    );
                    for &j in &bins[slot] {
                        if j <= i {
                            continue;
                        }
                        emit_images(points, cell, i, j, reach, range2, visit);
                    }
                }
            }
        }
    }
}

fn fill_bins(n: usize, dims: [usize; 3], bin: impl Fn(usize) -> [usize; 3]) -> Vec<Vec<usize>> {
    let mut bins = vec![
        Vec::new();
        dims[0]
            .saturating_mul(dims[1])
            .saturating_mul(dims[2])
            .max(1)
    ];
    for atom in 0..n {
        let slot = bin_slot(bin(atom), dims);
        if slot < bins.len() {
            bins[slot].push(atom);
        }
    }
    bins
}

fn bin_slot(index: [usize; 3], dims: [usize; 3]) -> usize {
    index[0] + dims[0] * (index[1] + dims[1] * index[2])
}

fn wrap_bin(index: isize, dim: usize) -> usize {
    let dim = dim.max(1) as isize;
    let mut wrapped = index % dim;
    if wrapped < 0 {
        wrapped += dim;
    }
    wrapped as usize
}

fn pack_rows(mut rows: Vec<Vec<usize>>) -> SparsityPattern {
    let n_atoms = rows.len();
    let mut row_ptr = Vec::with_capacity(n_atoms + 1);
    let mut col_idx = Vec::new();
    row_ptr.push(0);
    for row in &mut rows {
        row.sort_unstable();
        row.dedup();
        col_idx.append(row);
        row_ptr.push(col_idx.len());
    }
    SparsityPattern {
        n_atoms,
        row_ptr,
        col_idx,
    }
}

/// Distance-2 colouring of an atom graph.
///
/// `lower_bound` is one plus the maximum number of other atoms at
/// graph distance 1 or 2. A greedy colouring uses at most that many
/// colours. Isolated atoms share colour 0.
#[derive(Clone, Debug)]
pub struct Colouring {
    colour: Vec<usize>,
    groups: Vec<Vec<usize>>,
    max_distance2_degree: usize,
}

impl Colouring {
    /// Number of colours. Empty graphs report 0.
    pub fn n_colours(&self) -> usize {
        self.groups.len()
    }

    /// Colour of each atom, in `0..n_colours`.
    pub fn colours(&self) -> &[usize] {
        &self.colour
    }

    /// Atoms of each colour, ascending, colour 0 first.
    pub fn groups(&self) -> &[Vec<usize>] {
        &self.groups
    }

    /// Maximum number of atoms at graph distance 1 or 2 from one atom.
    pub fn max_distance2_degree(&self) -> usize {
        self.max_distance2_degree
    }

    /// `max_distance2_degree + 1`, or 0 when the graph is empty.
    pub fn lower_bound(&self) -> usize {
        if self.colour.is_empty() {
            0
        } else {
            self.max_distance2_degree + 1
        }
    }
}

/// Greedy distance-2 colouring in smallest-last order.
///
/// The vertex of minimum remaining degree is removed first, ties broken
/// by the smaller index. Colouring runs in the reverse of that removal.
/// Each atom takes the smallest colour not used at distance 1 or 2.
pub fn distance2_colouring(pattern: &SparsityPattern) -> Colouring {
    let n = pattern.n_atoms();
    let max_distance2_degree = max_distance2_degree(pattern);
    if n == 0 {
        return Colouring {
            colour: Vec::new(),
            groups: Vec::new(),
            max_distance2_degree,
        };
    }
    let order = smallest_last_order(pattern);
    let colour = greedy_distance2(pattern, &order);
    let n_colours = colour.iter().copied().max().unwrap_or(0) + 1;
    let mut groups = vec![Vec::new(); n_colours];
    for (atom, &c) in colour.iter().enumerate() {
        groups[c].push(atom);
    }
    Colouring {
        colour,
        groups,
        max_distance2_degree,
    }
}

fn max_distance2_degree(pattern: &SparsityPattern) -> usize {
    let n = pattern.n_atoms();
    if n == 0 {
        return 0;
    }
    let mut stamp = vec![0u32; n];
    let mut mark = 1u32;
    let mut max_deg = 0usize;
    for i in 0..n {
        mark = mark.wrapping_add(1);
        if mark == 0 {
            stamp.fill(0);
            mark = 1;
        }
        stamp[i] = mark;
        let mut deg = 0usize;
        for &u in pattern.neighbors(i) {
            if stamp[u] != mark {
                stamp[u] = mark;
                deg += 1;
            }
            for &w in pattern.neighbors(u) {
                if stamp[w] != mark {
                    stamp[w] = mark;
                    deg += 1;
                }
            }
        }
        max_deg = max_deg.max(deg);
    }
    max_deg
}

fn smallest_last_order(pattern: &SparsityPattern) -> Vec<usize> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    let n = pattern.n_atoms();
    let mut degree = vec![0usize; n];
    let mut heap = BinaryHeap::with_capacity(n);
    for (i, deg) in degree.iter_mut().enumerate() {
        *deg = pattern.neighbors(i).len();
        heap.push(Reverse((*deg, i)));
    }
    let mut alive = vec![true; n];
    let mut removed = Vec::with_capacity(n);
    while let Some(Reverse((deg, atom))) = heap.pop() {
        if !alive[atom] || degree[atom] != deg {
            continue;
        }
        alive[atom] = false;
        removed.push(atom);
        for &neighbor in pattern.neighbors(atom) {
            if alive[neighbor] {
                degree[neighbor] -= 1;
                heap.push(Reverse((degree[neighbor], neighbor)));
            }
        }
    }
    removed.reverse();
    removed
}

fn greedy_distance2(pattern: &SparsityPattern, order: &[usize]) -> Vec<usize> {
    let n = pattern.n_atoms();
    let mut colour = vec![usize::MAX; n];
    let mut forbidden = vec![0u32; n];
    let mut stamp = 1u32;
    for &v in order {
        stamp = stamp.wrapping_add(1);
        if stamp == 0 {
            forbidden.fill(0);
            stamp = 1;
        }
        for &u in pattern.neighbors(v) {
            if colour[u] != usize::MAX {
                forbidden[colour[u]] = stamp;
            }
            for &w in pattern.neighbors(u) {
                if w != v && colour[w] != usize::MAX {
                    forbidden[colour[w]] = stamp;
                }
            }
        }
        let mut c = 0usize;
        while forbidden[c] == stamp {
            c += 1;
        }
        colour[v] = c;
    }
    colour
}

/// Finite-difference controls for one recovery.
///
/// `step` is the displacement `h`. A central difference truncates as
/// `O(h^2)`. A gradient known to absolute accuracy `eps_g` adds a
/// roundoff `O(eps_g / h)`. On a shifted-force Lennard-Jones line and
/// on the Gupta cluster, the block error falls by about 100 each time
/// `h` drops by 10, from `1e-2` through `1e-6`, then rises at `1e-7`.
/// The default `1e-6` is that measured minimum. An ECnoise estimate of
/// `h` plugs in through the same field.
///
/// `memory_budget` is the byte cap on one colour block's geometry buffer
/// and gradient buffer together. The number of colours in the block is
/// that budget divided by `probes * n_dof * 16`, at least one, and the
/// whole colouring when the product fits. It does not depend on how many
/// threads the caller has.
///
/// `skin` is the pattern pad. Build the pattern with
/// [`SparsityPattern::from_positions`] at
/// `interaction_range + search_skin()`, where [`Self::search_skin`] is
/// `max(2 * step, skin)`.
#[derive(Clone, Copy, Debug)]
pub struct SparseHessianOpts {
    pub difference: FiniteDifference,
    pub step: f64,
    pub memory_budget: usize,
    pub skin: f64,
}

impl SparseHessianOpts {
    /// `max(2 * step, skin)`, the pad added to an interaction range.
    pub fn search_skin(self) -> f64 {
        (2.0 * self.step).max(self.skin)
    }
}

impl Default for SparseHessianOpts {
    fn default() -> Self {
        Self {
            difference: FiniteDifference::Central,
            step: 1e-6,
            memory_budget: DEFAULT_MEMORY_BUDGET,
            skin: 0.0,
        }
    }
}

/// Recovered atom-block Hessian.
///
/// Storage is CSR of 3x3 blocks, diagonal included, columns sorted.
/// Block `(i, j)` is row-major and was written by the colour of atom
/// `j` before symmetrisation. [`Self::matvec`] and the Lanczos basis
/// walk rows in index order.
///
/// The matrix is frozen at the geometry where it was recovered.
/// [`ApplyHessian::apply_hessian`] does not rebuild it. A position or a
/// vector whose length is not [`Self::dof`] panics.
#[derive(Clone, Debug)]
pub struct SparseHessian {
    n_atoms: usize,
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
    blocks: Vec<[f64; 9]>,
    evaluations: usize,
    n_colours: usize,
}

impl SparseHessian {
    /// Atoms in the pattern.
    pub fn n_atoms(&self) -> usize {
        self.n_atoms
    }

    /// Cartesian dimension, `3 * n_atoms`.
    pub fn dof(&self) -> usize {
        self.n_atoms * 3
    }

    /// Surface evaluations used to build the matrix.
    ///
    /// Each atom colour is probed once per Cartesian axis. Central
    /// differences use two sides and forward differences use one, plus
    /// a single centre evaluation for the forward formula. The count is
    /// `n_colours * 3 * sides`, plus one when the difference is forward.
    pub fn evaluations(&self) -> usize {
        self.evaluations
    }

    /// Atom colours in the partition that produced the blocks.
    ///
    /// [`one_atom_hessian`] reports one colour per atom.
    pub fn n_colours(&self) -> usize {
        self.n_colours
    }

    /// CSR row pointers of the 3x3 blocks, including the diagonal.
    pub fn row_ptr(&self) -> &[usize] {
        &self.row_ptr
    }

    /// Block column indices.
    pub fn col_idx(&self) -> &[usize] {
        &self.col_idx
    }

    /// Row-major 3x3 blocks, one per stored column.
    pub fn blocks(&self) -> &[[f64; 9]] {
        &self.blocks
    }

    /// One Cartesian entry. Missing blocks read as zero.
    pub fn entry(&self, row: usize, col: usize) -> f64 {
        let n = self.dof();
        if row >= n || col >= n {
            return 0.0;
        }
        let i = row / 3;
        let j = col / 3;
        let a = row % 3;
        let c = col % 3;
        for p in self.row_ptr[i]..self.row_ptr[i + 1] {
            if self.col_idx[p] == j {
                return self.blocks[p][a * 3 + c];
            }
        }
        0.0
    }

    /// `y = H v` in row order. No surface evaluation.
    pub fn matvec(&self, v: ArrayView1<f64>) -> Array1<f64> {
        let n = self.dof();
        let mut y = Array1::zeros(n.max(v.len()));
        if v.len() != n {
            return Array1::zeros(v.len());
        }
        for i in 0..self.n_atoms {
            for p in self.row_ptr[i]..self.row_ptr[i + 1] {
                let j = self.col_idx[p];
                let block = &self.blocks[p];
                for a in 0..3 {
                    let mut sum = 0.0;
                    for c in 0..3 {
                        sum += block[a * 3 + c] * v[3 * j + c];
                    }
                    y[3 * i + a] += sum;
                }
            }
        }
        y
    }

    /// Dense export in Cartesian order. Entries outside the pattern are zero.
    pub fn to_dense(&self) -> Array2<f64> {
        let n = self.dof();
        let mut h = Array2::zeros((n, n));
        for i in 0..self.n_atoms {
            for p in self.row_ptr[i]..self.row_ptr[i + 1] {
                let j = self.col_idx[p];
                let block = &self.blocks[p];
                for a in 0..3 {
                    for c in 0..3 {
                        h[(3 * i + a, 3 * j + c)] = block[a * 3 + c];
                    }
                }
            }
        }
        h
    }

    /// Lowest `k` eigenpairs of the recovered matrix.
    ///
    /// Lanczos uses [`Self::matvec`] only. Inner products and axpy updates
    /// go through [`rgmin::vecops`], the shared reduction helper. No
    /// surface evaluation. Eigenvalues ascend. Eigenvectors are columns.
    /// Residuals are `||H y - theta y||`. A pair is accepted when that
    /// residual is at most `1e-8` times an estimate of `||H||`.
    /// The iteration cap returns [`SaddleError::Solver`] instead of an
    /// unconverged Ritz pair. At dimension 96 or below the basis is the
    /// whole space.
    pub fn lowest_eigenpairs(&self, k: usize) -> Result<LowestEigenpairs, SaddleError> {
        let n = self.dof();
        if k == 0 || n == 0 {
            return Ok(LowestEigenpairs {
                values: Array1::zeros(0),
                vectors: Array2::zeros((n, 0)),
                residuals: Array1::zeros(0),
            });
        }
        let k = k.min(n);
        lanczos_lowest(|v| self.matvec(v), n, k)
    }

    /// Dense Hessian with rigid and excluded directions removed.
    ///
    /// Rows of [`PointSurface::excluded_modes`] are the directions.
    /// They are orthonormalised, then
    /// `H_proj = (I - Q Q^T) H (I - Q Q^T)` is formed in index order.
    /// The product `H Q` is the sparse matvec.
    pub fn project_excluded<S: PointSurface>(
        &self,
        surface: &S,
        x: ArrayView1<f64>,
    ) -> Result<Array2<f64>, SaddleError> {
        let n = self.dof();
        if x.len() != n {
            return Err(SaddleError::Shape(
                "excluded-mode query must match the Hessian".into(),
            ));
        }
        let modes = surface.excluded_modes(x)?;
        if modes.ncols() != n {
            return Err(SaddleError::Shape(
                "excluded modes must have one column per coordinate".into(),
            ));
        }
        if modes.iter().any(|v| !v.is_finite()) {
            return Err(SaddleError::NonFinite("excluded modes"));
        }
        if modes.nrows() == 0 {
            return Ok(self.to_dense());
        }
        let q = modified_gram_schmidt(modes.t(), None, 1e-12);
        let nq = q.ncols();
        if nq == 0 {
            return Ok(self.to_dense());
        }
        let mut hq = Array2::zeros((n, nq));
        for k in 0..nq {
            let hv = self.matvec(q.column(k));
            for i in 0..n {
                hq[(i, k)] = hv[i];
            }
        }
        let mut gram = Array2::zeros((nq, nq));
        for a in 0..nq {
            for b in 0..nq {
                gram[(a, b)] = dot(q.column(a), hq.column(b));
            }
        }
        let mut qm = Array2::zeros((n, nq));
        for i in 0..n {
            for b in 0..nq {
                let mut sum = 0.0;
                for a in 0..nq {
                    sum += q[(i, a)] * gram[(a, b)];
                }
                qm[(i, b)] = sum;
            }
        }
        let mut h = self.to_dense();
        for i in 0..n {
            for j in 0..n {
                let mut corr = 0.0;
                for k in 0..nq {
                    corr +=
                        hq[(i, k)] * q[(j, k)] + q[(i, k)] * hq[(j, k)] - qm[(i, k)] * q[(j, k)];
                }
                h[(i, j)] -= corr;
            }
        }
        Ok(h)
    }
}

impl ApplyHessian for SparseHessian {
    fn apply_hessian(&self, x: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
        assert_eq!(
            x.len(),
            self.dof(),
            "SparseHessian is frozen at the recovery point and the position length must match"
        );
        assert_eq!(
            v.len(),
            self.dof(),
            "SparseHessian is frozen at the recovery point and the vector length must match"
        );
        self.matvec(v)
    }
}

/// Lowest eigenpairs of a frozen Hessian, with Lanczos residuals.
#[derive(Clone, Debug)]
pub struct LowestEigenpairs {
    pub values: Array1<f64>,
    pub vectors: Array2<f64>,
    pub residuals: Array1<f64>,
}

/// Evaluate packed geometries, `n_geom` rows of `n_dof` coordinates.
///
/// Recovery calls this once per colour block. The rows of one block are
/// the Cartesian probes of several colours and run together. A batch
/// evaluator can replace the body. With the `par` feature the rows run
/// in chunks of 4, combined in index order. Without it the same chunks
/// run in order on the calling thread. The chunk size is not the thread
/// count. Write-back of the blocks is outside this function and follows
/// colour index order.
pub fn eval_displacements<S: PointSurface>(
    surface: &S,
    geometries: &[f64],
    n_dof: usize,
    energies: &mut [f64],
    gradients: &mut [f64],
) -> Result<(), SaddleError> {
    if n_dof == 0 {
        if geometries.is_empty() && energies.is_empty() && gradients.is_empty() {
            return Ok(());
        }
        return Err(SaddleError::Shape(
            "a zero-width displacement batch must be empty".into(),
        ));
    }
    if !geometries.len().is_multiple_of(n_dof) {
        return Err(SaddleError::Shape(
            "displacement batch must be a multiple of the dimension".into(),
        ));
    }
    let n_geom = geometries.len() / n_dof;
    if energies.len() != n_geom || gradients.len() != geometries.len() {
        return Err(SaddleError::Shape(
            "displacement batch buffers do not match".into(),
        ));
    }
    if n_geom == 0 {
        return Ok(());
    }
    #[cfg(test)]
    BATCHES.with(|count| count.set(count.get() + 1));
    #[cfg(feature = "par")]
    {
        use rayon::prelude::*;
        let results: Vec<Result<(), SaddleError>> = geometries
            .par_chunks(n_dof * GEOM_CHUNK)
            .zip(energies.par_chunks_mut(GEOM_CHUNK))
            .zip(gradients.par_chunks_mut(n_dof * GEOM_CHUNK))
            .map(|((geom, energy), gradient)| {
                eval_geometry_chunk(surface, geom, n_dof, energy, gradient)
            })
            .collect();
        for result in results {
            result?;
        }
        Ok(())
    }
    #[cfg(not(feature = "par"))]
    {
        let mut offset = 0usize;
        let mut row = 0usize;
        while offset < geometries.len() {
            let rows = ((geometries.len() - offset) / n_dof).min(GEOM_CHUNK);
            let geom = &geometries[offset..offset + rows * n_dof];
            let energy = &mut energies[row..row + rows];
            let gradient = &mut gradients[offset..offset + rows * n_dof];
            eval_geometry_chunk(surface, geom, n_dof, energy, gradient)?;
            offset += rows * n_dof;
            row += rows;
        }
        Ok(())
    }
}

fn eval_geometry_chunk<S: PointSurface>(
    surface: &S,
    geom: &[f64],
    n_dof: usize,
    energy: &mut [f64],
    gradient: &mut [f64],
) -> Result<(), SaddleError> {
    let rows = geom.len() / n_dof;
    for r in 0..rows {
        let x = ArrayView1::from(&geom[r * n_dof..(r + 1) * n_dof]);
        let (e, g) = surface.eval(x)?;
        if g.len() != n_dof {
            return Err(SaddleError::Shape("surface gradient length changed".into()));
        }
        if !e.is_finite() || g.iter().any(|v| !v.is_finite()) {
            return Err(SaddleError::NonFinite("surface"));
        }
        energy[r] = e;
        let dst = &mut gradient[r * n_dof..(r + 1) * n_dof];
        match g.as_slice() {
            Some(src) => dst.copy_from_slice(src),
            None => {
                for (slot, value) in dst.iter_mut().zip(g.iter()) {
                    *slot = *value;
                }
            }
        }
    }
    Ok(())
}

/// Direct CPR recovery from a distance-2 colouring of `pattern`.
pub fn sparse_hessian<S: PointSurface>(
    surface: &S,
    x: ArrayView1<f64>,
    pattern: &SparsityPattern,
    opts: SparseHessianOpts,
) -> Result<SparseHessian, SaddleError> {
    let colouring = distance2_colouring(pattern);
    recover(surface, x, pattern, colouring.groups(), opts)
}

/// Seed Hessian for a prefactor or an index-1 start.
///
/// Same recovery as [`sparse_hessian`]. [`crate::nichols::Index1Session`]
/// calls this when [`crate::nichols::Index1Config::sparse_pattern`] is
/// set and no Hessian was supplied.
pub fn prefactor_hessian<S: PointSurface>(
    surface: &S,
    x: ArrayView1<f64>,
    pattern: &SparsityPattern,
    opts: SparseHessianOpts,
) -> Result<SparseHessian, SaddleError> {
    sparse_hessian(surface, x, pattern, opts)
}

/// One atom at a time, the eOn prefactor displacement.
///
/// Each atom is its own colour, so columns are not compressed. The
/// blocks and the symmetrisation match [`sparse_hessian`].
pub fn one_atom_hessian<S: PointSurface>(
    surface: &S,
    x: ArrayView1<f64>,
    pattern: &SparsityPattern,
    opts: SparseHessianOpts,
) -> Result<SparseHessian, SaddleError> {
    let groups: Vec<Vec<usize>> = (0..pattern.n_atoms()).map(|i| vec![i]).collect();
    recover(surface, x, pattern, &groups, opts)
}

fn recover<S: PointSurface>(
    surface: &S,
    x: ArrayView1<f64>,
    pattern: &SparsityPattern,
    groups: &[Vec<usize>],
    opts: SparseHessianOpts,
) -> Result<SparseHessian, SaddleError> {
    if !(opts.step.is_finite() && opts.step > 0.0) {
        return Err(SaddleError::Invalid(
            "finite-difference step must be positive and finite".into(),
        ));
    }
    if !x.len().is_multiple_of(3) || x.len() / 3 != pattern.n_atoms() {
        return Err(SaddleError::Shape(
            "position length must be three times the pattern atom count".into(),
        ));
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(SaddleError::NonFinite("position"));
    }
    let n_atoms = pattern.n_atoms();
    let n_dof = n_atoms * 3;
    let sides = match opts.difference {
        FiniteDifference::Forward => 1usize,
        FiniteDifference::Central => 2usize,
    };
    ensure_direct_partition(pattern, groups)?;
    let (row_ptr, col_idx) = block_csr(pattern);
    let mut blocks = vec![[0.0; 9]; col_idx.len()];
    let x_base: Vec<f64> = x.iter().copied().collect();
    let probes = 3 * sides;
    let per_block = colours_in_block(groups.len(), n_dof, probes, opts.memory_budget);
    let n_geom = per_block.saturating_mul(probes);
    let mut geometries = vec![0.0; n_geom.saturating_mul(n_dof)];
    let mut energies = vec![0.0; n_geom];
    let mut gradients = vec![0.0; n_geom.saturating_mul(n_dof)];
    let mut centre = vec![0.0; n_dof];
    let mut evaluations = 0usize;
    if opts.difference == FiniteDifference::Forward && n_dof > 0 {
        let mut energy = [0.0];
        eval_displacements(surface, &x_base, n_dof, &mut energy, &mut centre)?;
        evaluations += 1;
    }
    #[cfg(test)]
    let mut phases = PhaseRecord::default();
    let mut start = 0usize;
    while start < groups.len() {
        let end = (start + per_block).min(groups.len());
        let slice = &groups[start..end];
        let rows = slice.len() * probes;
        #[cfg(test)]
        let paint_started = std::time::Instant::now();
        paint_block(
            &mut geometries[..rows * n_dof],
            &x_base,
            slice,
            opts.step,
            sides,
        );
        #[cfg(test)]
        {
            phases.paint += paint_started.elapsed().as_secs_f64();
        }
        if n_dof > 0 {
            #[cfg(test)]
            let eval_started = std::time::Instant::now();
            eval_displacements(
                surface,
                &geometries[..rows * n_dof],
                n_dof,
                &mut energies[..rows],
                &mut gradients[..rows * n_dof],
            )?;
            #[cfg(test)]
            {
                phases.eval += eval_started.elapsed().as_secs_f64();
            }
            evaluations += rows;
        }
        #[cfg(test)]
        let write_started = std::time::Instant::now();
        for (local, group) in slice.iter().enumerate() {
            for axis in 0..3 {
                let plus = local * probes + axis * sides;
                let gp = &gradients[plus * n_dof..(plus + 1) * n_dof];
                let gm: &[f64] = if sides == 2 {
                    let minus = plus + 1;
                    &gradients[minus * n_dof..(minus + 1) * n_dof]
                } else {
                    &centre
                };
                let scale = if sides == 2 {
                    2.0 * opts.step
                } else {
                    opts.step
                };
                write_axis(
                    pattern,
                    &mut BlockStore {
                        row_ptr: &row_ptr,
                        col_idx: &col_idx,
                        blocks: &mut blocks,
                    },
                    group,
                    axis,
                    gp,
                    gm,
                    scale,
                )?;
            }
        }
        #[cfg(test)]
        {
            phases.write += write_started.elapsed().as_secs_f64();
        }
        start = end;
    }
    #[cfg(test)]
    let sym_started = std::time::Instant::now();
    symmetrise(n_atoms, &row_ptr, &col_idx, &mut blocks)?;
    #[cfg(test)]
    {
        phases.sym += sym_started.elapsed().as_secs_f64();
        PHASES.with(|slot| {
            if slot.get().is_some() {
                slot.set(Some(phases));
            }
        });
    }
    if blocks.iter().flatten().any(|v| !v.is_finite()) {
        return Err(SaddleError::NonFinite("sparse Hessian"));
    }
    Ok(SparseHessian {
        n_atoms,
        row_ptr,
        col_idx,
        blocks,
        evaluations,
        n_colours: groups.len(),
    })
}

fn block_csr(pattern: &SparsityPattern) -> (Vec<usize>, Vec<usize>) {
    let n = pattern.n_atoms();
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    row_ptr.push(0);
    for i in 0..n {
        let mut placed = false;
        for &j in pattern.neighbors(i) {
            if !placed && i < j {
                col_idx.push(i);
                placed = true;
            }
            col_idx.push(j);
        }
        if !placed {
            col_idx.push(i);
        }
        row_ptr.push(col_idx.len());
    }
    (row_ptr, col_idx)
}

fn colours_in_block(n_colours: usize, n_dof: usize, probes: usize, budget: usize) -> usize {
    if n_colours == 0 {
        return 0;
    }
    let per_colour = probes.saturating_mul(n_dof).saturating_mul(16);
    if per_colour == 0 {
        return n_colours;
    }
    (budget / per_colour).max(1).min(n_colours)
}

fn paint_block(
    geometries: &mut [f64],
    base: &[f64],
    groups: &[Vec<usize>],
    step: f64,
    sides: usize,
) {
    let n_dof = base.len();
    let probes = 3 * sides;
    for (local, group) in groups.iter().enumerate() {
        let start = local * probes * n_dof;
        let end = start + probes * n_dof;
        paint_colour(&mut geometries[start..end], base, group, step, sides);
    }
}

fn paint_colour(geometries: &mut [f64], base: &[f64], group: &[usize], step: f64, sides: usize) {
    let n_dof = base.len();
    for axis in 0..3 {
        for side in 0..sides {
            let sign = if side == 0 { 1.0 } else { -1.0 };
            let index = axis * sides + side;
            let dst = &mut geometries[index * n_dof..(index + 1) * n_dof];
            dst.copy_from_slice(base);
            for &atom in group {
                dst[3 * atom + axis] += sign * step;
            }
        }
    }
}

fn ensure_direct_partition(
    pattern: &SparsityPattern,
    groups: &[Vec<usize>],
) -> Result<(), SaddleError> {
    if is_direct_partition(pattern, groups) {
        Ok(())
    } else {
        Err(SaddleError::Invalid(
            "colour is not a direct Curtis-Powell-Reid partition".into(),
        ))
    }
}

/// True when every colour is a direct Curtis-Powell-Reid partition.
///
/// The closed neighbourhood `{j} ∪ N(j)` of the atoms in one colour is
/// pairwise disjoint. Exposed for a later Kani harness.
pub(crate) fn is_direct_partition(pattern: &SparsityPattern, groups: &[Vec<usize>]) -> bool {
    let n = pattern.n_atoms();
    let mut seen = vec![0u32; n];
    let mut generation = 0u32;
    for group in groups {
        generation = generation.wrapping_add(1);
        if generation == 0 {
            seen.fill(0);
            generation = 1;
        }
        let mark = generation;
        for &j in group {
            if j >= n || seen[j] == mark {
                return false;
            }
            seen[j] = mark;
            for &i in pattern.neighbors(j) {
                if i >= n || seen[i] == mark {
                    return false;
                }
                seen[i] = mark;
            }
        }
    }
    true
}

struct BlockStore<'a> {
    row_ptr: &'a [usize],
    col_idx: &'a [usize],
    blocks: &'a mut [[f64; 9]],
}

fn write_axis(
    pattern: &SparsityPattern,
    store: &mut BlockStore<'_>,
    group: &[usize],
    axis: usize,
    gp: &[f64],
    gm: &[f64],
    scale: f64,
) -> Result<(), SaddleError> {
    for &j in group {
        let diag = block_index(store.row_ptr, store.col_idx, j, j)?;
        for alpha in 0..3 {
            store.blocks[diag][alpha * 3 + axis] = (gp[3 * j + alpha] - gm[3 * j + alpha]) / scale;
        }
        for &i in pattern.neighbors(j) {
            let slot = block_index(store.row_ptr, store.col_idx, i, j)?;
            for alpha in 0..3 {
                store.blocks[slot][alpha * 3 + axis] =
                    (gp[3 * i + alpha] - gm[3 * i + alpha]) / scale;
            }
        }
    }
    Ok(())
}

fn block_index(
    row_ptr: &[usize],
    col_idx: &[usize],
    i: usize,
    j: usize,
) -> Result<usize, SaddleError> {
    let start = row_ptr[i];
    let end = row_ptr[i + 1];
    col_idx[start..end]
        .iter()
        .position(|&col| col == j)
        .map(|offset| start + offset)
        .ok_or_else(|| SaddleError::Shape("missing Hessian block".into()))
}

fn symmetrise(
    n_atoms: usize,
    row_ptr: &[usize],
    col_idx: &[usize],
    blocks: &mut [[f64; 9]],
) -> Result<(), SaddleError> {
    for i in 0..n_atoms {
        let p = block_index(row_ptr, col_idx, i, i)?;
        let b = blocks[p];
        let mut s = [0.0; 9];
        for a in 0..3 {
            for c in 0..3 {
                s[a * 3 + c] = 0.5 * (b[a * 3 + c] + b[c * 3 + a]);
            }
        }
        blocks[p] = s;
    }
    for i in 0..n_atoms {
        for p in row_ptr[i]..row_ptr[i + 1] {
            let j = col_idx[p];
            if j <= i {
                continue;
            }
            let q = block_index(row_ptr, col_idx, j, i)?;
            let bij = blocks[p];
            let bji = blocks[q];
            let mut sij = [0.0; 9];
            let mut sji = [0.0; 9];
            for a in 0..3 {
                for c in 0..3 {
                    let v = 0.5 * (bij[a * 3 + c] + bji[c * 3 + a]);
                    sij[a * 3 + c] = v;
                    sji[c * 3 + a] = v;
                }
            }
            blocks[p] = sij;
            blocks[q] = sji;
        }
    }
    Ok(())
}

fn lanczos_lowest(
    apply: impl Fn(ArrayView1<f64>) -> Array1<f64>,
    n: usize,
    k: usize,
) -> Result<LowestEigenpairs, SaddleError> {
    if n <= KRYLOV_FULL_MAX {
        lanczos_full(&apply, n, k)
    } else {
        lanczos_thick(&apply, n, k)
    }
}

fn lanczos_full(
    apply: &impl Fn(ArrayView1<f64>) -> Array1<f64>,
    n: usize,
    k: usize,
) -> Result<LowestEigenpairs, SaddleError> {
    let m = n;
    let mut basis: Vec<Array1<f64>> = Vec::with_capacity(m);
    basis.push(lanczos_start(n));
    let mut t = Array2::zeros((m, m));
    for j in 0..m {
        let mut w = apply(basis[j].view());
        if w.len() != n {
            return Err(SaddleError::Shape("Hessian action length changed".into()));
        }
        let alpha = reorthogonalise(&mut w, &basis[..=j]);
        t[(j, j)] = alpha;
        if j + 1 == m {
            break;
        }
        let beta = dot(w.view(), w.view()).sqrt();
        if beta <= 1e-12 * (1.0 + alpha.abs()) {
            match orthogonal_direction(&basis[..=j], n) {
                Some(next) => basis.push(next),
                None => {
                    t = t.slice(ndarray::s![..=j, ..=j]).to_owned();
                    break;
                }
            }
        } else {
            w.mapv_inplace(|a| a / beta);
            t[(j, j + 1)] = beta;
            t[(j + 1, j)] = beta;
            basis.push(w);
        }
    }
    let m = basis.len();
    let (vals, vecs) = exact_eigh(t.view())?;
    let take = k.min(m);
    let mut out_vals = Array1::zeros(take);
    let mut out_vecs = Array2::zeros((n, take));
    let mut residuals = Array1::zeros(take);
    for r in 0..take {
        out_vals[r] = vals[r];
        let mut y = Array1::zeros(n);
        for t_i in 0..m {
            axpy(vecs[(t_i, r)], basis[t_i].view(), &mut y);
        }
        let norm = dot(y.view(), y.view()).sqrt();
        if norm > 0.0 {
            y.mapv_inplace(|a| a / norm);
        }
        let hy = apply(y.view());
        let mut residual = hy;
        axpy(-vals[r], y.view(), &mut residual);
        residuals[r] = dot(residual.view(), residual.view()).sqrt();
        for i in 0..n {
            out_vecs[(i, r)] = y[i];
        }
    }
    Ok(LowestEigenpairs {
        values: out_vals,
        vectors: out_vecs,
        residuals,
    })
}

fn lanczos_thick(
    apply: &impl Fn(ArrayView1<f64>) -> Array1<f64>,
    n: usize,
    k: usize,
) -> Result<LowestEigenpairs, SaddleError> {
    let cap = (80 + 40 * k).clamp(k + 16, n);
    let cycle = (k + 16).clamp(k + 2, 48).min(n);
    let mut matvecs = 0usize;
    let mut norm_est = 0.0f64;
    let mut locked_y: Vec<Array1<f64>> = Vec::new();
    let mut locked_theta: Vec<f64> = Vec::new();
    let mut locked_res: Vec<f64> = Vec::new();
    let mut seed = vec![lanczos_start(n)];

    while locked_y.len() < k {
        if matvecs >= cap {
            return Err(SaddleError::Solver(
                "Lanczos iteration cap reached before the residual test".into(),
            ));
        }
        let room = n - locked_y.len();
        let m = cycle.min(room).max(1);
        let mut basis: Vec<Array1<f64>> = Vec::new();
        for vector in &seed {
            let mut next = vector.clone();
            orthogonalize_against(&mut next, &locked_y);
            orthogonalize_against(&mut next, &basis);
            let norm = dot(next.view(), next.view()).sqrt();
            if norm > 1e-12 {
                next.mapv_inplace(|value| value / norm);
                basis.push(next);
            }
            if basis.len() == m {
                break;
            }
        }
        if basis.is_empty() {
            let Some(next) = orthogonal_direction(&locked_y, n) else {
                return Err(SaddleError::Solver(
                    "Lanczos invariant subspace ended before the residual test".into(),
                ));
            };
            basis.push(next);
        }
        let mut applied: Vec<Array1<f64>> = Vec::with_capacity(m);
        while basis.len() < m || applied.len() < basis.len() {
            if applied.len() == basis.len() {
                let mut direction = applied[basis.len() - 1].clone();
                orthogonalize_against(&mut direction, &locked_y);
                let alpha = reorthogonalise(&mut direction, &basis);
                let beta = dot(direction.view(), direction.view()).sqrt();
                if beta <= 1e-12 * (1.0 + alpha.abs() + norm_est) {
                    match orthogonal_direction_excluding(&basis, &locked_y, n) {
                        Some(next) => basis.push(next),
                        None => break,
                    }
                } else {
                    direction.mapv_inplace(|value| value / beta);
                    basis.push(direction);
                }
            }
            if matvecs >= cap {
                return Err(SaddleError::Solver(
                    "Lanczos iteration cap reached before the residual test".into(),
                ));
            }
            let image = apply(basis[applied.len()].view());
            matvecs += 1;
            if image.len() != n {
                return Err(SaddleError::Shape("Hessian action length changed".into()));
            }
            norm_est = norm_est.max(dot(image.view(), image.view()).sqrt());
            applied.push(image);
        }
        let mb = applied.len().min(basis.len());
        if mb == 0 {
            return Err(SaddleError::Solver(
                "Lanczos iteration cap reached before the residual test".into(),
            ));
        }
        let mut projected = Array2::zeros((mb, mb));
        for j in 0..mb {
            for i in 0..mb {
                projected[(i, j)] = dot(basis[i].view(), applied[j].view());
            }
        }
        for i in 0..mb {
            for j in (i + 1)..mb {
                let value = 0.5 * (projected[(i, j)] + projected[(j, i)]);
                projected[(i, j)] = value;
                projected[(j, i)] = value;
            }
        }
        let (_vals, vecs) = exact_eigh(projected.view())?;
        let spanned = mb + locked_y.len() == n;
        let mut restart = Vec::new();
        for r in 0..mb {
            if locked_y.len() >= k {
                break;
            }
            let mut y = Array1::zeros(n);
            let mut hy = Array1::zeros(n);
            for i in 0..mb {
                axpy(vecs[(i, r)], basis[i].view(), &mut y);
                axpy(vecs[(i, r)], applied[i].view(), &mut hy);
            }
            orthogonalize_against(&mut y, &locked_y);
            let norm = dot(y.view(), y.view()).sqrt();
            if norm <= 1e-14 {
                continue;
            }
            y.mapv_inplace(|value| value / norm);
            hy.mapv_inplace(|value| value / norm);
            let theta = dot(y.view(), hy.view());
            norm_est = norm_est.max(theta.abs());
            let mut residual_vec = hy;
            axpy(-theta, y.view(), &mut residual_vec);
            orthogonalize_against(&mut residual_vec, &locked_y);
            let residual = dot(residual_vec.view(), residual_vec.view()).sqrt();
            let scale = norm_est.max(1e-30);
            if residual <= LANCZOS_TOL * scale || spanned {
                locked_y.push(y);
                locked_theta.push(theta);
                locked_res.push(residual);
            } else if restart.len() < k + 4 {
                restart.push(y);
            }
        }
        if locked_y.len() >= k {
            break;
        }
        if restart.is_empty() {
            return Err(SaddleError::Solver(
                "Lanczos iteration cap reached before the residual test".into(),
            ));
        }
        seed = restart;
    }
    let mut order: Vec<usize> = (0..locked_theta.len()).collect();
    order.sort_by(|&i, &j| locked_theta[i].total_cmp(&locked_theta[j]));
    let take = k.min(order.len());
    let mut values = Array1::zeros(take);
    let mut vectors = Array2::zeros((n, take));
    let mut residuals = Array1::zeros(take);
    for (column, &src) in order.iter().take(take).enumerate() {
        values[column] = locked_theta[src];
        residuals[column] = locked_res[src];
        for row in 0..n {
            vectors[(row, column)] = locked_y[src][row];
        }
    }
    Ok(LowestEigenpairs {
        values,
        vectors,
        residuals,
    })
}

fn orthogonalize_against(w: &mut Array1<f64>, basis: &[Array1<f64>]) {
    if basis.is_empty() {
        return;
    }
    for _pass in 0..2 {
        for q in basis {
            let coeff = dot(w.view(), q.view());
            axpy(-coeff, q.view(), w);
        }
    }
}

fn orthogonal_direction_excluding(
    basis: &[Array1<f64>],
    locked: &[Array1<f64>],
    n: usize,
) -> Option<Array1<f64>> {
    for seed in 0..n {
        let mut w = Array1::zeros(n);
        w[seed] = 1.0;
        orthogonalize_against(&mut w, locked);
        orthogonalize_against(&mut w, basis);
        let norm = dot(w.view(), w.view()).sqrt();
        if norm > 1e-12 {
            w.mapv_inplace(|value| value / norm);
            return Some(w);
        }
    }
    None
}

fn reorthogonalise(w: &mut Array1<f64>, basis: &[Array1<f64>]) -> f64 {
    let last = basis.len() - 1;
    let mut alpha = 0.0;
    for _pass in 0..2 {
        for (i, q) in basis.iter().enumerate() {
            let c = dot(w.view(), q.view());
            if i == last {
                alpha += c;
            }
            axpy(-c, q.view(), w);
        }
    }
    alpha
}

fn orthogonal_direction(basis: &[Array1<f64>], n: usize) -> Option<Array1<f64>> {
    for seed in 0..n {
        let mut w = Array1::zeros(n);
        w[seed] = 1.0;
        for _pass in 0..2 {
            for q in basis {
                let c = dot(w.view(), q.view());
                axpy(-c, q.view(), &mut w);
            }
        }
        let norm = dot(w.view(), w.view()).sqrt();
        if norm > 1e-12 {
            w.mapv_inplace(|a| a / norm);
            return Some(w);
        }
    }
    None
}

fn lanczos_start(n: usize) -> Array1<f64> {
    let mut v = Array1::zeros(n);
    if n == 0 {
        return v;
    }
    for i in 0..n {
        let t = (i as f64) * 0.618_033_988_749_894_9 + 0.125;
        v[i] = t.sin();
    }
    let scale = dot(v.view(), v.view()).sqrt();
    if scale > 0.0 {
        v.mapv_inplace(|a| a / scale);
    } else {
        v[0] = 1.0;
    }
    v
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use ndarray::{Array1, Array2, array};

    use super::*;
    use crate::eigensolve::exact_eigh;
    use crate::minmode::PointSurface;
    use crate::nichols::{Index1Config, Index1Session};

    fn pair_block(d2: f64, dv: f64, r: f64, u: [f64; 3]) -> [f64; 9] {
        let mut block = [0.0; 9];
        let radial = dv / r;
        for a in 0..3 {
            for c in 0..3 {
                let uu = u[a] * u[c];
                let eye = if a == c { 1.0 } else { 0.0 };
                block[a * 3 + c] = d2 * uu + radial * (eye - uu);
            }
        }
        block
    }

    fn lj_derivs(r: f64, sigma: f64) -> (f64, f64) {
        let s = sigma / r;
        let s2 = s * s;
        let s6 = s2 * s2 * s2;
        let s12 = s6 * s6;
        let dv = 24.0 / r * (s6 - 2.0 * s12);
        let d2 = 24.0 * (26.0 * s12 - 7.0 * s6) / (r * r);
        (dv, d2)
    }

    /// Shifted-force Lennard-Jones. Neighbours come from the geometry.
    struct ShiftedForce {
        sigma: f64,
        rc: f64,
        cell: Option<Cell>,
        calls: AtomicUsize,
    }

    impl ShiftedForce {
        fn open(sigma: f64, rc: f64) -> Self {
            Self {
                sigma,
                rc,
                cell: None,
                calls: AtomicUsize::new(0),
            }
        }

        fn periodic(sigma: f64, rc: f64, cell: Cell) -> Self {
            Self {
                sigma,
                rc,
                cell: Some(cell),
                calls: AtomicUsize::new(0),
            }
        }
    }

    fn shifted_force_derivs(r: f64, sigma: f64, rc: f64) -> Option<(f64, f64, f64)> {
        if !(r > 0.0 && r < rc) {
            return None;
        }
        let (dv, d2) = lj_derivs(r, sigma);
        let (dvc, _) = lj_derivs(rc, sigma);
        let sr2 = (sigma / r) * (sigma / r);
        let sr6 = sr2 * sr2 * sr2;
        let sr12 = sr6 * sr6;
        let energy = 4.0 * (sr12 - sr6);
        let src2 = (sigma / rc) * (sigma / rc);
        let src6 = src2 * src2 * src2;
        let src12 = src6 * src6;
        let vc = 4.0 * (src12 - src6);
        let shifted = energy - vc - (r - rc) * dvc;
        Some((shifted, dv - dvc, d2))
    }

    impl PointSurface for ShiftedForce {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            let mut g = Array1::zeros(x.len());
            let mut energy = 0.0;
            let points = coords_of(x);
            visit_cutoff(&points, self.cell.as_ref(), self.rc, |i, j, dr| {
                let r2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                let r = r2.sqrt();
                let Some((pot, dv, _)) = shifted_force_derivs(r, self.sigma, self.rc) else {
                    return;
                };
                energy += pot;
                for axis in 0..3 {
                    let u = dr[axis] / r;
                    g[3 * i + axis] -= dv * u;
                    g[3 * j + axis] += dv * u;
                }
            });
            Ok((energy, g))
        }
    }

    fn lj_analytic_blocks(
        positions: ArrayView1<f64>,
        cell: Option<&Cell>,
        sigma: f64,
        rc: f64,
    ) -> std::collections::HashMap<(usize, usize), [f64; 9]> {
        let mut blocks = std::collections::HashMap::new();
        let points = coords_of(positions);
        visit_cutoff(&points, cell, rc, |i, j, dr| {
            let r2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
            let r = r2.sqrt();
            let Some((_, dv, d2)) = shifted_force_derivs(r, sigma, rc) else {
                return;
            };
            let u = [dr[0] / r, dr[1] / r, dr[2] / r];
            let block = pair_block(d2, dv, r, u);
            add_pair_blocks(&mut blocks, i, j, &block);
        });
        blocks
    }

    fn add_pair_blocks(
        blocks: &mut std::collections::HashMap<(usize, usize), [f64; 9]>,
        i: usize,
        j: usize,
        block: &[f64; 9],
    ) {
        add_block(blocks, i, i, block, 1.0);
        add_block(blocks, j, j, block, 1.0);
        add_block(blocks, i, j, block, -1.0);
        add_block(blocks, j, i, block, -1.0);
    }

    fn add_block(
        blocks: &mut std::collections::HashMap<(usize, usize), [f64; 9]>,
        i: usize,
        j: usize,
        block: &[f64; 9],
        scale: f64,
    ) {
        let slot = blocks.entry((i, j)).or_insert([0.0; 9]);
        for (dst, src) in slot.iter_mut().zip(block.iter()) {
            *dst += scale * src;
        }
    }

    fn max_analytic_error(
        got: &SparseHessian,
        analytic: &std::collections::HashMap<(usize, usize), [f64; 9]>,
    ) -> f64 {
        let mut worst = 0.0_f64;
        for (&(i, j), reference) in analytic {
            let mut stored = [0.0; 9];
            if let Ok(slot) = block_index(got.row_ptr(), got.col_idx(), i, j) {
                stored = got.blocks()[slot];
            }
            worst = worst.max(block_error(&stored, reference));
        }
        for i in 0..got.n_atoms() {
            for p in got.row_ptr()[i]..got.row_ptr()[i + 1] {
                let j = got.col_idx()[p];
                if analytic.contains_key(&(i, j)) {
                    continue;
                }
                worst = worst.max(block_error(&got.blocks()[p], &[0.0; 9]));
            }
        }
        worst
    }

    fn block_error(got: &[f64; 9], reference: &[f64; 9]) -> f64 {
        let mut num = 0.0_f64;
        let mut den = 0.0_f64;
        for (g, r) in got.iter().zip(reference.iter()) {
            num = num.max((g - r).abs());
            den = den.max(r.abs());
        }
        if den < 1e-8 { num } else { num / den }
    }

    struct DoubleWell;

    impl PointSurface for DoubleWell {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let (xx, y, z) = (x[0], x[1], x[2]);
            let energy = (xx * xx - 1.0).powi(2) + 2.0 * y * y + 2.0 * z * z;
            let g = array![4.0 * xx * (xx * xx - 1.0), 4.0 * y, 4.0 * z];
            Ok((energy, g))
        }
    }

    struct Quadratic;

    impl PointSurface for Quadratic {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            Ok((0.5 * x.dot(&x), x.to_owned()))
        }
    }

    struct Translations<S> {
        inner: S,
    }

    impl<S: PointSurface> PointSurface for Translations<S> {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            self.inner.eval(x)
        }

        fn excluded_modes(&self, x: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
            let n_atoms = x.len() / 3;
            let mut modes = Array2::zeros((3, x.len()));
            for axis in 0..3 {
                for atom in 0..n_atoms {
                    modes[(axis, 3 * atom + axis)] = 1.0;
                }
            }
            Ok(modes)
        }
    }

    fn assert_distance2(pattern: &SparsityPattern, colouring: &Colouring) {
        assert!(colouring.n_colours() <= colouring.lower_bound());
        let colour = colouring.colours();
        for i in 0..pattern.n_atoms() {
            assert!(pattern.neighbors(i).binary_search(&i).is_err());
            for &j in pattern.neighbors(i) {
                assert_ne!(colour[i], colour[j], "edge {i}-{j}");
            }
            let mut seen = vec![false; pattern.n_atoms()];
            seen[i] = true;
            for &u in pattern.neighbors(i) {
                if !seen[u] {
                    seen[u] = true;
                    assert_ne!(colour[i], colour[u]);
                }
                for &w in pattern.neighbors(u) {
                    if !seen[w] {
                        seen[w] = true;
                        assert_ne!(colour[i], colour[w], "distance-2 {i}-{w}");
                    }
                }
            }
        }
    }

    fn u01(seed: u64, index: u64) -> f64 {
        let mut z = seed.wrapping_add(index.wrapping_mul(0x9E3779B97F4A7C15));
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        (z >> 11) as f64 / ((1u64 << 53) as f64)
    }

    #[test]
    fn path_is_distance2_coloured_within_the_bound() {
        let edges = [(0, 1), (1, 2), (2, 3)];
        let pattern = SparsityPattern::from_adjacency(4, &edges).unwrap();
        let colouring = distance2_colouring(&pattern);
        assert_distance2(&pattern, &colouring);
        assert!(colouring.n_colours() <= 3);
        assert!(colouring.lower_bound() >= colouring.n_colours());
    }

    #[test]
    fn isolates_share_a_colour_and_a_self_image_is_not_an_edge() {
        let cell = Cell::ortho(1.0, 1.0, 1.0).unwrap();
        let x = array![0.1, 0.2, 0.3];
        let pattern = SparsityPattern::from_positions(x.view(), Some(&cell), 2.0, 0.0).unwrap();
        assert!(pattern.neighbors(0).is_empty());
        let colouring = distance2_colouring(&pattern);
        assert_eq!(colouring.n_colours(), 1);
        assert_eq!(colouring.lower_bound(), 1);

        let wrapped = array![0.0, 0.0, 0.0, 0.9, 0.0, 0.0];
        let pattern =
            SparsityPattern::from_positions(wrapped.view(), Some(&cell), 0.3, 0.0).unwrap();
        assert_eq!(pattern.neighbors(0), &[1]);
        let colouring = distance2_colouring(&pattern);
        assert_eq!(colouring.n_colours(), 2);
        assert_ne!(colouring.colours()[0], colouring.colours()[1]);
    }

    #[test]
    fn random_cells_respect_the_distance2_bound() {
        for seed in 0..24u64 {
            let n = 1 + (seed % 7) as usize;
            let ax = 1.5 + u01(seed, 1);
            let by = 1.5 + u01(seed, 2);
            let cz = 1.5 + u01(seed, 3);
            let shear = 0.25;
            let cell = if seed % 5 == 0 {
                Cell::ortho(0.45, 0.5, 0.55).unwrap()
            } else {
                Cell::from_vectors(
                    [ax, u01(seed, 4) * shear, u01(seed, 5) * shear],
                    [u01(seed, 6) * shear, by, u01(seed, 7) * shear],
                    [u01(seed, 8) * shear, u01(seed, 9) * shear, cz],
                    [0.0, 0.0, 0.0],
                )
                .unwrap()
            };
            let mut coords = Vec::with_capacity(n * 3);
            for atom in 0..n {
                let s = [
                    u01(seed, 10 + 3 * atom as u64),
                    u01(seed, 11 + 3 * atom as u64),
                    u01(seed, 12 + 3 * atom as u64),
                ];
                let p = cell.cartesian(s);
                coords.extend_from_slice(&p);
            }
            let range = if seed % 5 == 0 {
                1.2
            } else {
                0.3 + u01(seed, 90)
            };
            let positions = Array1::from(coords);
            let pattern =
                SparsityPattern::from_positions(positions.view(), Some(&cell), range, 0.0).unwrap();
            let colouring = distance2_colouring(&pattern);
            assert_distance2(&pattern, &colouring);
            if n == 1 {
                assert!(pattern.neighbors(0).is_empty());
                assert_eq!(colouring.n_colours(), 1);
            }
        }
    }

    #[test]
    fn lj_central_blocks_match_the_analytic_hessian() {
        let coords = array![0.0, 0.0, 0.0, 1.3, 0.0, 0.0, 0.5, 1.2, 0.1, 8.0, 0.2, -0.3,];
        let pattern = SparsityPattern::from_positions(coords.view(), None, 3.0, 0.0).unwrap();
        assert!(pattern.neighbors(3).is_empty());
        let surface = ShiftedForce::open(1.0, 2.5);
        let opts = SparseHessianOpts {
            difference: FiniteDifference::Central,
            step: 1e-6,
            ..SparseHessianOpts::default()
        };
        let got = sparse_hessian(&surface, coords.view(), &pattern, opts).unwrap();
        assert_eq!(got.evaluations(), surface.calls.load(Ordering::Relaxed));
        assert_eq!(
            got.evaluations(),
            got.n_colours() * 3 * 2,
            "central differences have no centre evaluation"
        );

        let analytic = lj_analytic_blocks(coords.view(), None, 1.0, 2.5);
        let err = max_analytic_error(&got, &analytic);
        assert!(err < 1e-5, "relative block error {err}");
        let dense = got.to_dense();
        let n = dense.nrows();
        for i in 0..n {
            for j in 0..n {
                assert_eq!(dense[(i, j)], dense[(j, i)]);
            }
        }
        for col in 0..3 {
            for row in 0..9 {
                assert_eq!(got.entry(row, 9 + col), 0.0);
                assert_eq!(got.entry(9 + col, row), 0.0);
            }
        }
    }

    #[test]
    fn forward_count_includes_one_centre_evaluation() {
        let coords = array![0.0, 0.0, 0.0, 1.3, 0.0, 0.0];
        let pattern = SparsityPattern::from_positions(coords.view(), None, 2.0, 0.0).unwrap();
        let surface = ShiftedForce::open(1.0, 2.5);
        let opts = SparseHessianOpts {
            difference: FiniteDifference::Forward,
            step: 1e-5,
            ..SparseHessianOpts::default()
        };
        let got = sparse_hessian(&surface, coords.view(), &pattern, opts).unwrap();
        assert_eq!(got.evaluations(), got.n_colours() * 3 + 1);
        assert_eq!(got.evaluations(), surface.calls.load(Ordering::Relaxed));
    }

    #[test]
    fn quadratic_forward_and_central_recover_identity() {
        let coords = array![0.2, -0.4, 0.1, 1.0, 0.0, -0.2];
        let pattern = SparsityPattern::from_adjacency(2, &[]).unwrap();
        let colouring = distance2_colouring(&pattern);
        assert_eq!(colouring.n_colours(), 1);
        for difference in [FiniteDifference::Forward, FiniteDifference::Central] {
            let got = sparse_hessian(
                &Quadratic,
                coords.view(),
                &pattern,
                SparseHessianOpts {
                    difference,
                    step: 1e-4,
                    ..SparseHessianOpts::default()
                },
            )
            .unwrap();
            let dense = got.to_dense();
            for i in 0..6 {
                for j in 0..6 {
                    let expect = if i == j { 1.0 } else { 0.0 };
                    assert!((dense[(i, j)] - expect).abs() < 1e-12);
                }
            }
            let hv = got.apply_hessian(coords.view(), array![1.0, 0.0, 0.0, 0.0, 2.0, 0.0].view());
            assert!((hv[0] - 1.0).abs() < 1e-12);
            assert!((hv[4] - 2.0).abs() < 1e-12);
        }
    }

    fn dense_central<S: PointSurface>(surface: &S, x: ArrayView1<f64>, step: f64) -> Array2<f64> {
        let n = x.len();
        let mut h = Array2::zeros((n, n));
        for j in 0..n {
            let mut xp = x.to_owned();
            let mut xm = x.to_owned();
            xp[j] += step;
            xm[j] -= step;
            let (_, gp) = surface.eval(xp.view()).unwrap();
            let (_, gm) = surface.eval(xm.view()).unwrap();
            for i in 0..n {
                h[(i, j)] = (gp[i] - gm[i]) / (2.0 * step);
            }
        }
        for i in 0..n {
            for j in (i + 1)..n {
                let v = 0.5 * (h[(i, j)] + h[(j, i)]);
                h[(i, j)] = v;
                h[(j, i)] = v;
            }
        }
        h
    }

    fn lowest_overlap(values: &Array1<f64>, vectors: &Array2<f64>, dense: &Array2<f64>) {
        let (evals, evecs) = exact_eigh(dense.view()).unwrap();
        let gap = (values[0] - evals[0]).abs();
        assert!(gap < 1e-6, "eigenvalue gap {gap}");
        let mut overlap = 0.0;
        for i in 0..vectors.nrows() {
            overlap += vectors[(i, 0)] * evecs[(i, 0)];
        }
        assert!(overlap.abs() > 1.0 - 1e-6, "overlap {overlap}");
    }

    #[test]
    fn double_well_lowest_eigenpair_matches_dense_differences() {
        let x = array![0.0, 0.0, 0.0];
        let pattern = SparsityPattern::from_adjacency(1, &[]).unwrap();
        let step = 1e-5;
        let got = sparse_hessian(
            &DoubleWell,
            x.view(),
            &pattern,
            SparseHessianOpts {
                difference: FiniteDifference::Central,
                step,
                ..SparseHessianOpts::default()
            },
        )
        .unwrap();
        let dense = dense_central(&DoubleWell, x.view(), step);
        let solved = got.lowest_eigenpairs(1).unwrap();
        let values = &solved.values;
        let vectors = &solved.vectors;
        lowest_overlap(values, vectors, &dense);
        assert!((values[0] + 4.0).abs() < 1e-8);
        assert!(vectors[(0, 0)].abs() > 0.999);
        let before = got.evaluations();
        let _ = got.lowest_eigenpairs(1).unwrap();
        let _ = got.matvec(x.view());
        assert_eq!(got.evaluations(), before);
    }

    #[test]
    fn lj7_saddle_lowest_eigenpair_matches_dense_differences() {
        let coords = array![
            0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 6.0, 6.0, 6.0,
            0.0, 6.0, 0.0, 6.0,
        ];
        let pattern = SparsityPattern::from_positions(coords.view(), None, 2.0, 0.0).unwrap();
        assert_eq!(pattern.neighbors(0), &[1]);
        let surface = ShiftedForce::open(1.0, 2.5);
        let step = 1e-6;
        let opts = SparseHessianOpts {
            difference: FiniteDifference::Central,
            step,
            ..SparseHessianOpts::default()
        };
        let got = sparse_hessian(&surface, coords.view(), &pattern, opts).unwrap();
        let dense = got.to_dense();
        for i in 0..dense.nrows() {
            for j in 0..dense.ncols() {
                assert_eq!(dense[(i, j)], dense[(j, i)]);
            }
        }
        let reference = dense_central(&surface, coords.view(), step);
        let solved = got.lowest_eigenpairs(1).unwrap();
        let values = &solved.values;
        let vectors = &solved.vectors;
        lowest_overlap(values, vectors, &reference);
        assert!(values[0] < -1e-3, "lowest curvature {}", values[0]);
    }

    #[test]
    fn projected_hessian_removes_a_translation() {
        let coords = array![0.0, 0.0, 0.0, 1.3, 0.0, 0.0];
        let pattern = SparsityPattern::from_positions(coords.view(), None, 2.0, 0.0).unwrap();
        let surface = Translations {
            inner: ShiftedForce::open(1.0, 2.5),
        };
        let got = sparse_hessian(
            &surface,
            coords.view(),
            &pattern,
            SparseHessianOpts {
                difference: FiniteDifference::Central,
                step: 1e-6,
                ..SparseHessianOpts::default()
            },
        )
        .unwrap();
        let projected = got.project_excluded(&surface, coords.view()).unwrap();
        let n = projected.nrows();
        for axis in 0..3 {
            let mut mode = Array1::zeros(n);
            for atom in 0..2 {
                mode[3 * atom + axis] = 1.0 / 2f64.sqrt();
            }
            let mut rayleigh = 0.0;
            for i in 0..n {
                let mut hv = 0.0;
                for j in 0..n {
                    hv += projected[(i, j)] * mode[j];
                }
                rayleigh += mode[i] * hv;
            }
            assert!(rayleigh.abs() < 1e-8, "axis {axis} rayleigh {rayleigh}");
        }
    }

    #[test]
    fn repeated_recovery_is_bit_identical() {
        let mut raw = Vec::new();
        for i in 0..6 {
            raw.extend_from_slice(&[1.3 * i as f64, 0.0, 0.0]);
        }
        let coords = Array1::from(raw);
        let pattern = SparsityPattern::from_positions(coords.view(), None, 2.0, 0.0).unwrap();
        let surface = ShiftedForce::open(1.0, 2.5);
        let opts = SparseHessianOpts {
            difference: FiniteDifference::Central,
            step: 1e-6,
            ..SparseHessianOpts::default()
        };
        let a = sparse_hessian(&surface, coords.view(), &pattern, opts).unwrap();
        let b = sparse_hessian(&surface, coords.view(), &pattern, opts).unwrap();
        assert_eq!(a.blocks(), b.blocks());
        let one = one_atom_hessian(&surface, coords.view(), &pattern, opts).unwrap();
        assert_eq!(a.blocks(), one.blocks());
        assert!(a.n_colours() < one.n_colours());
    }

    #[test]
    fn index1_uses_the_sparse_seed_only_when_asked() {
        let mut coords = Vec::new();
        for i in 0..5 {
            coords.extend_from_slice(&[1.3 * i as f64, 0.0, 0.0]);
        }
        let x = Array1::from(coords);
        let pattern = SparsityPattern::from_positions(x.view(), None, 2.0, 0.0).unwrap();
        let colouring = distance2_colouring(&pattern);
        assert!(colouring.n_colours() < 5);
        let sparse_surface = ShiftedForce::open(1.0, 2.5);
        let dense_surface = ShiftedForce::open(1.0, 2.5);
        let mut sparse = Index1Session::new(
            Index1Config::default()
                .with_sparse_pattern(pattern)
                .with_force_tol(0.0)
                .with_trust_radius(1e-3)
                .with_fd_dr(1e-4),
            x.clone(),
            None,
            None,
        )
        .unwrap();
        sparse.step(&sparse_surface).unwrap();
        let sparse_calls = sparse_surface.calls.load(Ordering::Relaxed);
        assert_eq!(sparse_calls, colouring.n_colours() * 3 * 2 + 2);

        let mut dense = Index1Session::new(
            Index1Config::default()
                .with_force_tol(0.0)
                .with_trust_radius(1e-3)
                .with_fd_dr(1e-4),
            x,
            None,
            None,
        )
        .unwrap();
        dense.step(&dense_surface).unwrap();
        assert_eq!(dense_surface.calls.load(Ordering::Relaxed), 5 * 3 * 2 + 2);
        assert!(sparse_calls < dense_surface.calls.load(Ordering::Relaxed));
    }

    #[cfg(feature = "par")]
    fn fnv1a(words: &[f64]) -> u64 {
        let mut hash = 0xcbf29ce484222325u64;
        for value in words {
            for byte in value.to_ne_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x100000001b3);
            }
        }
        hash
    }

    #[cfg(feature = "par")]
    fn csr_words(h: &SparseHessian) -> Vec<f64> {
        h.blocks().iter().flatten().copied().collect()
    }

    #[cfg(feature = "par")]
    fn eigen_words(h: &SparseHessian) -> Vec<f64> {
        let solved = h.lowest_eigenpairs(1).unwrap();
        let mut words = solved.values.to_vec();
        words.extend(solved.vectors.iter().copied());
        words
    }

    #[cfg(feature = "par")]
    fn with_threads<T>(threads: usize, body: impl FnOnce() -> T + Send) -> T
    where
        T: Send,
    {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(body)
    }

    #[test]
    #[cfg(feature = "par")]
    fn rank_invariant_csr_and_eigenpairs_at_1_2_3_and_8_threads() {
        let a = 2f64.powf(1.0 / 6.0) * 2f64.sqrt();
        let coords = Array1::from(fcc_open(2, 2, 2, a));
        let cell = Cell::ortho(2.0 * a, 2.0 * a, 2.0 * a).unwrap();
        let cutoff = 2.5;
        let opts = SparseHessianOpts {
            difference: FiniteDifference::Central,
            step: 1e-6,
            ..SparseHessianOpts::default()
        };
        let pattern =
            SparsityPattern::from_positions(coords.view(), Some(&cell), cutoff, opts.search_skin())
                .unwrap();
        let surface = ShiftedForce::periodic(1.0, cutoff, cell);
        let mut csr_hashes = Vec::new();
        let mut eigen_hashes = Vec::new();
        for threads in [1usize, 2, 3, 8] {
            let recovered = with_threads(threads, || {
                sparse_hessian(&surface, coords.view(), &pattern, opts).unwrap()
            });
            csr_hashes.push(fnv1a(&csr_words(&recovered)));
            eigen_hashes.push(fnv1a(&eigen_words(&recovered)));
        }
        assert!(csr_hashes.iter().all(|h| *h == csr_hashes[0]));
        assert!(eigen_hashes.iter().all(|h| *h == eigen_hashes[0]));
    }

    #[test]
    #[cfg(feature = "par")]
    fn colour_block_strong_scaling_on_vacancy_cells() {
        let systems = [
            (14usize, "fcc_vacancy_10975_shifted_force"),
            (29, "fcc_vacancy_97555_shifted_force"),
        ];
        let max_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let mut counts = vec![1usize, 2, 4, 8, max_threads];
        counts.sort_unstable();
        counts.dedup();
        for (n, name) in systems {
            let (positions, cell) = fcc_vacancy_cell(n);
            let coords = Array1::from(positions);
            let rc = 2.5;
            let opts = SparseHessianOpts::default();
            let pattern =
                SparsityPattern::from_positions(coords.view(), Some(&cell), rc, opts.search_skin())
                    .unwrap();
            let surface = ShiftedForce::periodic(1.0, rc, cell);
            let colouring = distance2_colouring(&pattern);
            eprintln!(
                "{name} atoms {} colours {} dof {}",
                coords.len() / 3,
                colouring.n_colours(),
                coords.len()
            );
            let mut samples = vec![Vec::new(); counts.len()];
            let mut eval_frac = vec![Vec::new(); counts.len()];
            let mut paint_frac = vec![Vec::new(); counts.len()];
            let mut write_frac = vec![Vec::new(); counts.len()];
            let mut hashes = Vec::new();
            for _rep in 0..5 {
                for (i, &threads) in counts.iter().enumerate() {
                    let (recovered, phases, wall) = with_threads(threads, || {
                        PHASES.with(|slot| slot.set(Some(PhaseRecord::default())));
                        let started = std::time::Instant::now();
                        let recovered =
                            recover(&surface, coords.view(), &pattern, colouring.groups(), opts)
                                .unwrap();
                        let wall = started.elapsed().as_secs_f64();
                        let phases = PHASES.with(|slot| slot.get()).unwrap_or_default();
                        (recovered, phases, wall)
                    });
                    samples[i].push(wall);
                    let denom = (phases.eval + phases.paint + phases.write + phases.sym).max(1e-12);
                    eval_frac[i].push(phases.eval / denom);
                    paint_frac[i].push(phases.paint / denom);
                    write_frac[i].push((phases.write + phases.sym) / denom);
                    if hashes.len() == i {
                        hashes.push(fnv1a(&csr_words(&recovered)));
                    }
                }
            }
            assert!(hashes.iter().all(|hash| *hash == hashes[0]));
            let t1 = median(&samples[0]);
            eprintln!("{name} threads median_s min_s max_s efficiency");
            for (i, &threads) in counts.iter().enumerate() {
                let med = median(&samples[i]);
                let min = samples[i].iter().copied().fold(f64::MAX, f64::min);
                let max = samples[i].iter().copied().fold(0.0, f64::max);
                let efficiency = t1 / (threads as f64 * med);
                eprintln!("{threads} {med:.6} {min:.6} {max:.6} {efficiency:.3}");
            }
            let last = counts.len() - 1;
            let eval_m = median(&eval_frac[last]);
            let paint_m = median(&paint_frac[last]);
            let write_m = median(&write_frac[last]);
            let limiter = if eval_m >= paint_m && eval_m >= write_m {
                "surface evaluation"
            } else if paint_m >= write_m {
                "memory bandwidth of the geometry copies"
            } else {
                "serial assembly"
            };
            eprintln!(
                "{name} limiter at {} threads: {limiter} (eval {eval_m:.2}, paint {paint_m:.2}, assembly {write_m:.2})",
                counts[last]
            );
        }
    }

    fn median(values: &[f64]) -> f64 {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b));
        sorted[sorted.len() / 2]
    }

    fn nn_of(positions: &[f64]) -> f64 {
        let n = positions.len() / 3;
        let mut best = f64::MAX;
        for i in 0..n {
            for j in (i + 1)..n {
                let mut r2 = 0.0;
                for axis in 0..3 {
                    let d = positions[3 * i + axis] - positions[3 * j + axis];
                    r2 += d * d;
                }
                best = best.min(r2);
            }
        }
        best.sqrt()
    }

    fn lj38_positions() -> Vec<f64> {
        let a = 2f64.powf(1.0 / 6.0) * 2f64.sqrt();
        let all = fcc_open(4, 4, 4, a);
        let n = all.len() / 3;
        let mut c = [0.0; 3];
        for atom in all.as_chunks::<3>().0 {
            c[0] += atom[0];
            c[1] += atom[1];
            c[2] += atom[2];
        }
        c[0] /= n as f64;
        c[1] /= n as f64;
        c[2] /= n as f64;
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&i, &j| {
            let mut di = 0.0;
            let mut dj = 0.0;
            for axis in 0..3 {
                let pi = all[3 * i + axis] - c[axis];
                let pj = all[3 * j + axis] - c[axis];
                di += pi * pi;
                dj += pj * pj;
            }
            di.total_cmp(&dj).then(i.cmp(&j))
        });
        let mut out = Vec::with_capacity(38 * 3);
        for &i in order.iter().take(38) {
            out.extend_from_slice(&all[3 * i..3 * i + 3]);
        }
        out
    }

    fn fcc_open(nx: usize, ny: usize, nz: usize, a: f64) -> Vec<f64> {
        let basis = [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ];
        let mut out = Vec::new();
        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    for site in basis {
                        out.push((ix as f64 + site[0]) * a);
                        out.push((iy as f64 + site[1]) * a);
                        out.push((iz as f64 + site[2]) * a);
                    }
                }
            }
        }
        out
    }

    fn fcc_vacancy_cell(n: usize) -> (Vec<f64>, Cell) {
        let a = 2f64.powf(1.0 / 6.0) * 2f64.sqrt();
        let mut positions = fcc_open(n, n, n, a);
        positions.drain(0..3);
        let cell = Cell::ortho(n as f64 * a, n as f64 * a, n as f64 * a).unwrap();
        (positions, cell)
    }

    fn line_cluster(n: usize, spacing: f64) -> Vec<f64> {
        let mut raw = Vec::new();
        for i in 0..n {
            raw.extend_from_slice(&[spacing * i as f64, 0.0, 0.0]);
        }
        raw
    }

    #[test]
    fn two_memory_budgets_match_and_a_wide_budget_is_one_batch() {
        let coords = Array1::from(line_cluster(6, 1.3));
        let opts = SparseHessianOpts {
            step: 1e-6,
            ..SparseHessianOpts::default()
        };
        let pattern =
            SparsityPattern::from_positions(coords.view(), None, 2.5, opts.search_skin()).unwrap();
        assert!(distance2_colouring(&pattern).n_colours() > 1);
        let surface = ShiftedForce::open(1.0, 2.5);
        BATCHES.with(|count| count.set(0));
        let wide = sparse_hessian(&surface, coords.view(), &pattern, opts).unwrap();
        let wide_batches = BATCHES.with(|count| count.get());
        let narrow_opts = SparseHessianOpts {
            memory_budget: 1,
            ..opts
        };
        BATCHES.with(|count| count.set(0));
        let narrow = sparse_hessian(&surface, coords.view(), &pattern, narrow_opts).unwrap();
        let narrow_batches = BATCHES.with(|count| count.get());
        assert_eq!(wide.blocks(), narrow.blocks());
        assert!(wide_batches < narrow_batches);
        assert_eq!(wide_batches, 1);
        let wide_pairs = wide.lowest_eigenpairs(1).unwrap();
        let narrow_pairs = narrow.lowest_eigenpairs(1).unwrap();
        assert_eq!(wide_pairs.values, narrow_pairs.values);
        assert_eq!(wide_pairs.vectors, narrow_pairs.vectors);
    }

    #[test]
    fn direct_partition_rejects_a_neighbouring_pair() {
        let pattern = SparsityPattern::from_adjacency(3, &[(0, 1), (1, 2)]).unwrap();
        let colouring = distance2_colouring(&pattern);
        assert!(is_direct_partition(&pattern, colouring.groups()));
        assert!(!is_direct_partition(&pattern, &[vec![0, 1]]));
    }

    #[test]
    #[should_panic(expected = "frozen at the recovery point")]
    fn apply_hessian_panics_when_the_vector_length_does_not_match() {
        let coords = array![0.0, 0.0, 0.0];
        let pattern = SparsityPattern::from_adjacency(1, &[]).unwrap();
        let got = sparse_hessian(
            &Quadratic,
            coords.view(),
            &pattern,
            SparseHessianOpts::default(),
        )
        .unwrap();
        let _ = got.apply_hessian(coords.view(), array![1.0].view());
    }

    #[test]
    fn linkcell_pairs_match_brute_force_images() {
        for seed in 0..8u64 {
            let n = 4 + (seed % 4) as usize;
            let cell = if seed % 2 == 0 {
                Cell::ortho(0.7, 0.85, 1.05).unwrap()
            } else {
                Cell::from_vectors(
                    [1.2, 0.0, 0.0],
                    [0.9, 0.28, 0.0],
                    [0.75, 0.22, 0.31],
                    [0.0, 0.0, 0.0],
                )
                .unwrap()
            };
            let mut raw = Vec::new();
            for atom in 0..n {
                let s = [
                    u01(seed, 3 * atom as u64),
                    u01(seed, 3 * atom as u64 + 1),
                    u01(seed, 3 * atom as u64 + 2),
                ];
                raw.extend_from_slice(&cell.cartesian(s));
            }
            let positions = Array1::from(raw);
            let range = 1.1 + 0.2 * u01(seed, 99);
            let points = coords_of(positions.view());
            let brute = oracle_pairs(&points, Some(&cell), range);
            let mut binned = std::collections::BTreeSet::new();
            visit_binned_images(&points, &cell, range, &mut |i, j, _| {
                binned.insert((i, j));
            });
            assert_eq!(brute, binned, "seed {seed}");
            let pattern =
                SparsityPattern::from_positions(positions.view(), Some(&cell), range, 0.0).unwrap();
            assert_eq!(brute, pattern_pairs(&pattern));
        }
        let cell = Cell::ortho(30.0, 28.0, 26.0).unwrap();
        let mut raw = Vec::new();
        for atom in 0..40 {
            raw.extend_from_slice(&[
                u01(7, atom) * 30.0,
                u01(8, atom) * 28.0,
                u01(9, atom) * 26.0,
            ]);
        }
        let positions = Array1::from(raw);
        let points = coords_of(positions.view());
        let range = 2.5;
        let brute = oracle_pairs(&points, Some(&cell), range);
        let mut mic = std::collections::BTreeSet::new();
        visit_ortho_mic(&points, &cell, range, &mut |i, j, _| {
            mic.insert((i, j));
        });
        assert_eq!(brute, mic);
        let open = SparsityPattern::from_positions(positions.view(), None, range, 0.0).unwrap();
        assert_eq!(oracle_pairs(&points, None, range), pattern_pairs(&open));
    }

    fn pattern_pairs(pattern: &SparsityPattern) -> std::collections::BTreeSet<(usize, usize)> {
        let mut pairs = std::collections::BTreeSet::new();
        for i in 0..pattern.n_atoms() {
            for &j in pattern.neighbors(i) {
                if i < j {
                    pairs.insert((i, j));
                }
            }
        }
        pairs
    }

    fn oracle_pairs(
        points: &[[f64; 3]],
        cell: Option<&Cell>,
        range: f64,
    ) -> std::collections::BTreeSet<(usize, usize)> {
        let mut pairs = std::collections::BTreeSet::new();
        let range2 = range * range;
        match cell {
            None => {
                for i in 0..points.len() {
                    for j in (i + 1)..points.len() {
                        let dr = [
                            points[j][0] - points[i][0],
                            points[j][1] - points[i][1],
                            points[j][2] - points[i][2],
                        ];
                        let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                        if d2 > 0.0 && d2 <= range2 {
                            pairs.insert((i, j));
                        }
                    }
                }
            }
            Some(cell) => {
                let widths = cell.widths();
                let reach = [0, 1, 2].map(|axis| {
                    let images = (range / widths[axis].abs().max(1e-15)).ceil() as i32;
                    images + 4
                });
                for i in 0..points.len() {
                    for j in (i + 1)..points.len() {
                        for na in -reach[0]..=reach[0] {
                            for nb in -reach[1]..=reach[1] {
                                for nc in -reach[2]..=reach[2] {
                                    let shift = cell.lattice_shift(na, nb, nc);
                                    let d2 = cell.dist2_shifted(points[i], points[j], shift);
                                    if d2 > 0.0 && d2 <= range2 {
                                        pairs.insert((i, j));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        pairs
    }

    #[test]
    fn periodic_image_sum_matches_the_analytic_hessian() {
        let cell = Cell::ortho(1.2, 8.0, 8.0).unwrap();
        let coords = array![0.1, 0.2, 0.3, 0.8, 0.25, 0.3];
        let rc = 2.5;
        let opts = SparseHessianOpts {
            step: 1e-6,
            ..SparseHessianOpts::default()
        };
        let pattern =
            SparsityPattern::from_positions(coords.view(), Some(&cell), rc, opts.search_skin())
                .unwrap();
        let points = coords_of(coords.view());
        let mut images = 0usize;
        visit_cutoff(&points, Some(&cell), rc, |_, _, _| images += 1);
        assert!(images > 1, "images {images}");
        let surface = ShiftedForce::periodic(1.0, rc, cell);
        let got = sparse_hessian(&surface, coords.view(), &pattern, opts).unwrap();
        let analytic = lj_analytic_blocks(coords.view(), Some(&cell), 1.0, rc);
        let err = max_analytic_error(&got, &analytic);
        assert!(err < 1e-5, "relative block error {err}");
    }

    #[test]
    fn a_short_pattern_has_a_large_analytic_error() {
        let coords = Array1::from(line_cluster(4, 1.2));
        let opts = SparseHessianOpts {
            step: 1e-6,
            ..SparseHessianOpts::default()
        };
        let short =
            SparsityPattern::from_positions(coords.view(), None, 0.5, opts.search_skin()).unwrap();
        assert!(short.neighbors(0).is_empty());
        let surface = ShiftedForce::open(1.0, 2.5);
        let got = sparse_hessian(&surface, coords.view(), &short, opts).unwrap();
        let analytic = lj_analytic_blocks(coords.view(), None, 1.0, 2.5);
        let err = max_analytic_error(&got, &analytic);
        assert!(err > 0.1, "short pattern error {err}");
    }

    #[test]
    fn gupta_matches_a_finite_difference_hessian() {
        let coords = array![
            0.0, 0.0, 0.0, 1.15, 0.0, 0.0, 0.4, 1.05, 0.05, 1.05, 0.95, 0.15,
        ];
        let params = GuptaParams::demo();
        let surface = Gupta::new(params, None);
        let (energy, gradient) = surface.eval(coords.view()).unwrap();
        assert!(energy.is_finite());
        let h = 1e-6;
        for i in 0..coords.len() {
            let mut plus = coords.clone();
            let mut minus = coords.clone();
            plus[i] += h;
            minus[i] -= h;
            let (ep, _) = surface.eval(plus.view()).unwrap();
            let (em, _) = surface.eval(minus.view()).unwrap();
            let fd = (ep - em) / (2.0 * h);
            let gap = (fd - gradient[i]).abs();
            assert!(gap < 1e-6, "gradient {i} gap {gap}");
        }
        let dense = dense_central(&surface, coords.view(), h);
        let analytic = gupta_analytic_blocks(coords.view(), None, params);
        for i in 0..4 {
            for j in 0..4 {
                let reference = analytic.get(&(i, j)).copied().unwrap_or([0.0; 9]);
                for a in 0..3 {
                    for c in 0..3 {
                        let got = dense[(3 * i + a, 3 * j + c)];
                        let expect = reference[a * 3 + c];
                        let scale = expect.abs().max(1e-6);
                        let gap = (got - expect).abs() / scale;
                        assert!(gap < 1e-4, "block {i},{j} {a}{c} gap {gap}");
                    }
                }
            }
        }
    }

    #[test]
    fn step_sweep_shows_truncation() {
        let lj_coords = Array1::from(line_cluster(4, 1.2));
        let gupta_coords = lj_coords.clone();
        let params = GuptaParams::demo();
        let steps = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7];
        let mut lj_err = Vec::new();
        let mut gupta_err = Vec::new();
        eprintln!("h lj_rel gupta_rel");
        for step in steps {
            let opts = SparseHessianOpts {
                step,
                ..SparseHessianOpts::default()
            };
            let lj_pattern =
                SparsityPattern::from_positions(lj_coords.view(), None, 2.5, opts.search_skin())
                    .unwrap();
            let lj = sparse_hessian(
                &ShiftedForce::open(1.0, 2.5),
                lj_coords.view(),
                &lj_pattern,
                opts,
            )
            .unwrap();
            let lj_ref = lj_analytic_blocks(lj_coords.view(), None, 1.0, 2.5);
            lj_err.push(max_analytic_error(&lj, &lj_ref));
            let gupta_pattern = SparsityPattern::from_positions(
                gupta_coords.view(),
                None,
                2.0 * params.rc,
                opts.search_skin(),
            )
            .unwrap();
            let gupta = sparse_hessian(
                &Gupta::new(params, None),
                gupta_coords.view(),
                &gupta_pattern,
                opts,
            )
            .unwrap();
            let gupta_ref = gupta_analytic_blocks(gupta_coords.view(), None, params);
            gupta_err.push(max_analytic_error(&gupta, &gupta_ref));
            eprintln!(
                "{step:.1e} {} {}",
                lj_err.last().unwrap(),
                gupta_err.last().unwrap()
            );
        }
        assert!(
            lj_err[0] > lj_err[1] * 50.0,
            "LJ truncation branch {lj_err:?}"
        );
        assert!(
            gupta_err[0] > gupta_err[1] * 50.0,
            "Gupta truncation branch {gupta_err:?}"
        );
        assert!(lj_err[5] > lj_err[4], "LJ roundoff branch {lj_err:?}");
        assert!(
            gupta_err[5] > gupta_err[4],
            "Gupta roundoff branch {gupta_err:?}"
        );
        assert!((SparseHessianOpts::default().step - 1e-6).abs() < 1e-15);
        assert!(lj_err[4] <= lj_err[3] && lj_err[4] <= lj_err[5]);
        assert!(gupta_err[4] <= gupta_err[3] && gupta_err[4] <= gupta_err[5]);
    }

    #[test]
    fn lanczos_matches_exact_eigh_above_96_dof_and_on_500_atoms() {
        check_dimer_saddle(38);
        check_dimer_saddle(498);
    }

    fn check_dimer_saddle(spectators: usize) {
        let mut raw = vec![0.0, 0.0, 0.0, 1.5, 0.0, 0.0];
        let side = (spectators as f64).cbrt().ceil() as usize + 1;
        let mut count = 0usize;
        'place: for ix in 0..side {
            for iy in 0..side {
                for iz in 0..side {
                    if count == spectators {
                        break 'place;
                    }
                    raw.push(20.0 + 6.0 * ix as f64);
                    raw.push(6.0 * iy as f64);
                    raw.push(6.0 * iz as f64);
                    count += 1;
                }
            }
        }
        let coords = Array1::from(raw);
        assert!(coords.len() > 96);
        let rc = 2.5;
        let opts = SparseHessianOpts {
            step: 1e-6,
            ..SparseHessianOpts::default()
        };
        let pattern =
            SparsityPattern::from_positions(coords.view(), None, rc, opts.search_skin()).unwrap();
        let surface = ShiftedForce::open(1.0, rc);
        let got = sparse_hessian(&surface, coords.view(), &pattern, opts).unwrap();
        let solved = got.lowest_eigenpairs(1).unwrap();
        assert!(solved.values[0] < -1e-3, "curvature {}", solved.values[0]);
        let scale = solved.values[0].abs().max(1.0);
        assert!(
            solved.residuals[0] <= 1e-6 * scale,
            "residual {} at dof {}",
            solved.residuals[0],
            got.dof()
        );
        let dense = got.to_dense();
        let (evals, evecs) = exact_eigh(dense.view()).unwrap();
        let gap = (solved.values[0] - evals[0]).abs();
        assert!(gap < 1e-6, "dof {} gap {gap}", got.dof());
        let mut overlap = 0.0;
        for i in 0..got.dof() {
            overlap += solved.vectors[(i, 0)] * evecs[(i, 0)];
        }
        assert!(overlap.abs() > 1.0 - 1e-5, "overlap {overlap}");
    }

    #[derive(Clone, Copy)]
    struct GuptaParams {
        amplitude: f64,
        p: f64,
        q: f64,
        r0: f64,
        rc: f64,
    }

    impl GuptaParams {
        fn demo() -> Self {
            Self {
                amplitude: 0.2,
                p: 9.0,
                q: 3.0,
                r0: 1.15,
                rc: 1.9,
            }
        }
    }

    struct Bond {
        i: usize,
        j: usize,
        r: f64,
        u: [f64; 3],
        phi: f64,
        phip: f64,
        phipp: f64,
        psi: f64,
        psip: f64,
        psipp: f64,
    }

    fn smooth_cutoff(r: f64, rc: f64) -> Option<(f64, f64, f64)> {
        if !(r > 0.0 && r < rc) {
            return None;
        }
        let x = 1.0 - r / rc;
        Some((x * x * x, -3.0 * x * x / rc, 6.0 * x / (rc * rc)))
    }

    fn product(g: f64, gp: f64, gpp: f64, fc: f64, fp: f64, fpp: f64) -> (f64, f64, f64) {
        (g * fc, gp * fc + g * fp, gpp * fc + 2.0 * gp * fp + g * fpp)
    }

    fn gupta_bond(i: usize, j: usize, dr: [f64; 3], params: GuptaParams) -> Option<Bond> {
        let r2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
        let r = r2.sqrt();
        let (fc, fp, fpp) = smooth_cutoff(r, params.rc)?;
        let reduced = r / params.r0 - 1.0;
        let phi_bare = params.amplitude * (-params.p * reduced).exp();
        let phi_scale = -params.p / params.r0;
        let (phi, phip, phipp) = product(
            phi_bare,
            phi_bare * phi_scale,
            phi_bare * phi_scale * phi_scale,
            fc,
            fp,
            fpp,
        );
        let psi_bare = (-2.0 * params.q * reduced).exp();
        let psi_scale = -2.0 * params.q / params.r0;
        let (psi, psip, psipp) = product(
            psi_bare,
            psi_bare * psi_scale,
            psi_bare * psi_scale * psi_scale,
            fc,
            fp,
            fpp,
        );
        Some(Bond {
            i,
            j,
            r,
            u: [dr[0] / r, dr[1] / r, dr[2] / r],
            phi,
            phip,
            phipp,
            psi,
            psip,
            psipp,
        })
    }

    fn gupta_bonds(
        positions: ArrayView1<f64>,
        cell: Option<&Cell>,
        params: GuptaParams,
    ) -> Vec<Bond> {
        let points = coords_of(positions);
        let mut bonds = Vec::new();
        visit_cutoff(&points, cell, params.rc, |i, j, dr| {
            if let Some(bond) = gupta_bond(i, j, dr, params) {
                bonds.push(bond);
            }
        });
        bonds
    }

    struct Gupta {
        params: GuptaParams,
        cell: Option<Cell>,
        calls: AtomicUsize,
    }

    impl Gupta {
        fn new(params: GuptaParams, cell: Option<Cell>) -> Self {
            Self {
                params,
                cell,
                calls: AtomicUsize::new(0),
            }
        }
    }

    impl PointSurface for Gupta {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            let bonds = gupta_bonds(x, self.cell.as_ref(), self.params);
            let n = x.len() / 3;
            let mut rho = vec![0.0; n];
            let mut energy = 0.0;
            for bond in &bonds {
                energy += bond.phi;
                rho[bond.i] += bond.psi;
                rho[bond.j] += bond.psi;
            }
            let mut fp = vec![0.0; n];
            for i in 0..n {
                if rho[i] > 1e-18 {
                    energy -= rho[i].sqrt();
                    fp[i] = -0.5 / rho[i].sqrt();
                }
            }
            let mut gradient = Array1::zeros(x.len());
            for bond in &bonds {
                let weight = bond.phip + (fp[bond.i] + fp[bond.j]) * bond.psip;
                for axis in 0..3 {
                    gradient[3 * bond.i + axis] -= weight * bond.u[axis];
                    gradient[3 * bond.j + axis] += weight * bond.u[axis];
                }
            }
            Ok((energy, gradient))
        }
    }

    fn gupta_analytic_blocks(
        positions: ArrayView1<f64>,
        cell: Option<&Cell>,
        params: GuptaParams,
    ) -> std::collections::HashMap<(usize, usize), [f64; 9]> {
        let bonds = gupta_bonds(positions, cell, params);
        let n = positions.len() / 3;
        let mut rho = vec![0.0; n];
        for bond in &bonds {
            rho[bond.i] += bond.psi;
            rho[bond.j] += bond.psi;
        }
        let mut fp = vec![0.0; n];
        let mut fpp = vec![0.0; n];
        for i in 0..n {
            if rho[i] > 1e-18 {
                let root = rho[i].sqrt();
                fp[i] = -0.5 / root;
                fpp[i] = 0.25 / (rho[i] * root);
            }
        }
        let mut toward = vec![std::collections::HashMap::<usize, [f64; 3]>::new(); n];
        let mut capital = vec![[0.0; 3]; n];
        for bond in &bonds {
            let p = [
                bond.psip * bond.u[0],
                bond.psip * bond.u[1],
                bond.psip * bond.u[2],
            ];
            add_vec(toward[bond.i].entry(bond.j).or_insert([0.0; 3]), p, 1.0);
            add_vec(toward[bond.j].entry(bond.i).or_insert([0.0; 3]), p, -1.0);
            add_vec(&mut capital[bond.i], p, 1.0);
            add_vec(&mut capital[bond.j], p, -1.0);
        }
        let mut blocks = std::collections::HashMap::new();
        for bond in &bonds {
            let weight = bond.phip + (fp[bond.i] + fp[bond.j]) * bond.psip;
            let second = bond.phipp + (fp[bond.i] + fp[bond.j]) * bond.psipp;
            let block = pair_block(second, weight, bond.r, bond.u);
            add_pair_blocks(&mut blocks, bond.i, bond.j, &block);
        }
        for bond in &bonds {
            let p = toward[bond.i][&bond.j];
            let mut atoms = vec![bond.i, bond.j];
            atoms.extend(toward[bond.i].keys().copied());
            atoms.extend(toward[bond.j].keys().copied());
            atoms.sort_unstable();
            atoms.dedup();
            for atom in atoms {
                let di = density_gradient(&toward, &capital, bond.i, atom);
                let dj = density_gradient(&toward, &capital, bond.j, atom);
                let mut outer = [0.0; 9];
                for alpha in 0..3 {
                    for beta in 0..3 {
                        let embed = fpp[bond.i] * di[beta] + fpp[bond.j] * dj[beta];
                        outer[alpha * 3 + beta] = p[alpha] * embed;
                    }
                }
                add_block(&mut blocks, bond.i, atom, &outer, -1.0);
                add_block(&mut blocks, bond.j, atom, &outer, 1.0);
            }
        }
        blocks
    }

    fn add_vec(slot: &mut [f64; 3], value: [f64; 3], scale: f64) {
        for axis in 0..3 {
            slot[axis] += scale * value[axis];
        }
    }

    fn density_gradient(
        toward: &[std::collections::HashMap<usize, [f64; 3]>],
        capital: &[[f64; 3]],
        atom: usize,
        coord: usize,
    ) -> [f64; 3] {
        if atom == coord {
            [-capital[atom][0], -capital[atom][1], -capital[atom][2]]
        } else {
            toward[atom].get(&coord).copied().unwrap_or([0.0; 3])
        }
    }

    struct BenchSystem {
        name: &'static str,
        positions: Vec<f64>,
        cell: Option<Cell>,
        lj_sigma: Option<f64>,
        gupta: Option<GuptaParams>,
        rc: f64,
        pattern_range: f64,
    }

    fn bench_systems() -> Vec<BenchSystem> {
        let lj38 = BenchSystem {
            name: "lj38_shifted_force",
            positions: lj38_positions(),
            cell: None,
            lj_sigma: Some(1.0),
            gupta: None,
            rc: 2.5,
            pattern_range: 2.5,
        };
        let pt_raw = [
            7.494156838,
            7.43185154,
            14.66275315,
            7.491064071,
            12.37096012,
            14.66277188,
            10.34261175,
            7.43185155,
            14.66275313,
            10.3457045,
            12.37096013,
            14.66277186,
            11.76992719,
            9.90404444,
            14.6628811,
            6.066841393,
            9.904044422,
            14.66288113,
            8.918384291,
            9.902301057,
            14.65866693,
        ];
        let pt_nn = nn_of(&pt_raw);
        let pt_sigma = pt_nn / 2f64.powf(1.0 / 6.0);
        let pt = BenchSystem {
            name: "pt_heptamer_shifted_force",
            positions: pt_raw.to_vec(),
            cell: None,
            lj_sigma: Some(pt_sigma),
            gupta: None,
            rc: 2.5 * pt_sigma,
            pattern_range: 2.5 * pt_sigma,
        };
        let (vac_pos, vac_cell) = fcc_vacancy_cell(5);
        let vacancy = BenchSystem {
            name: "fcc_vacancy_499_shifted_force",
            positions: vac_pos.clone(),
            cell: Some(vac_cell),
            lj_sigma: Some(1.0),
            gupta: None,
            rc: 2.5,
            pattern_range: 2.5,
        };
        let gupta_rc = 1.5 * 2f64.powf(1.0 / 6.0);
        let gupta = BenchSystem {
            name: "fcc_vacancy_499_gupta",
            positions: vac_pos,
            cell: Some(vac_cell),
            lj_sigma: None,
            gupta: Some(GuptaParams {
                amplitude: 0.2,
                p: 9.0,
                q: 3.0,
                r0: 2f64.powf(1.0 / 6.0),
                rc: gupta_rc,
            }),
            rc: gupta_rc,
            pattern_range: 2.0 * gupta_rc,
        };
        vec![lj38, pt, vacancy, gupta]
    }

    enum BenchSurface {
        Lj(ShiftedForce),
        Gupta(Gupta),
    }

    impl PointSurface for BenchSurface {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            match self {
                Self::Lj(surface) => surface.eval(x),
                Self::Gupta(surface) => surface.eval(x),
            }
        }
    }

    fn bench_surface(system: &BenchSystem) -> BenchSurface {
        if let Some(params) = system.gupta {
            BenchSurface::Gupta(Gupta::new(params, system.cell))
        } else {
            let sigma = system.lj_sigma.unwrap();
            match system.cell {
                Some(cell) => BenchSurface::Lj(ShiftedForce::periodic(sigma, system.rc, cell)),
                None => BenchSurface::Lj(ShiftedForce::open(sigma, system.rc)),
            }
        }
    }

    fn bench_error(system: &BenchSystem, got: &SparseHessian, positions: ArrayView1<f64>) -> f64 {
        if let Some(params) = system.gupta {
            let analytic = gupta_analytic_blocks(positions, system.cell.as_ref(), params);
            max_analytic_error(got, &analytic)
        } else {
            let analytic = lj_analytic_blocks(
                positions,
                system.cell.as_ref(),
                system.lj_sigma.unwrap(),
                system.rc,
            );
            max_analytic_error(got, &analytic)
        }
    }

    fn rattle(positions: &[f64], seed: u64, amp: f64) -> Vec<f64> {
        positions
            .iter()
            .enumerate()
            .map(|(i, &p)| p + (u01(seed, i as u64) - 0.5) * 2.0 * amp)
            .collect()
    }

    #[test]
    fn hessian_bench_interleaves_cpr_and_one_atom() {
        let full = std::env::var("RGSADDLE_HESSIAN_BENCH").ok().as_deref() == Some("1");
        let systems = if full {
            bench_systems()
        } else {
            vec![bench_systems().into_iter().next().unwrap()]
        };
        let rattles = if full { 5 } else { 1 };
        let repeats = if full { 5 } else { 1 };
        let mut lines =
            vec!["system,method,seed,converged,iters,calls,energy,fmax,wall_s".to_string()];
        for system in &systems {
            eprintln!(
                "{} atoms {} rc {} pattern {}",
                system.name,
                system.positions.len() / 3,
                system.rc,
                system.pattern_range
            );
            let mut cpr_walls: Vec<Vec<f64>> = vec![Vec::new(); rattles];
            let mut atom_walls: Vec<Vec<f64>> = vec![Vec::new(); rattles];
            let mut errors = Vec::new();
            for seed in 0..rattles {
                let amp = 0.01 * system.rc.min(0.05);
                let rattled = rattle(&system.positions, seed as u64, amp);
                let x = Array1::from(rattled);
                let opts = SparseHessianOpts::default();
                let pattern = SparsityPattern::from_positions(
                    x.view(),
                    system.cell.as_ref(),
                    system.pattern_range,
                    opts.search_skin(),
                )
                .unwrap();
                let surface = bench_surface(system);
                let (energy, _) = surface.eval(x.view()).unwrap();
                let mut cpr_calls = 0usize;
                let mut atom_calls = 0usize;
                let mut colours = 0usize;
                let mut fmax = 0.0;
                for rep in 0..repeats {
                    let first_cpr = rep % 2 == 0;
                    let run = |method: &str| -> (f64, SparseHessian, usize) {
                        let started = std::time::Instant::now();
                        let recovered = if method == "cpr" {
                            sparse_hessian(&surface, x.view(), &pattern, opts).unwrap()
                        } else {
                            one_atom_hessian(&surface, x.view(), &pattern, opts).unwrap()
                        };
                        let wall = started.elapsed().as_secs_f64();
                        let calls = recovered.evaluations();
                        (wall, recovered, calls)
                    };
                    let (cpr, atom) = if first_cpr {
                        let cpr = run("cpr");
                        let atom = run("one_atom");
                        (cpr, atom)
                    } else {
                        let atom = run("one_atom");
                        let cpr = run("cpr");
                        (cpr, atom)
                    };
                    cpr_walls[seed].push(cpr.0);
                    atom_walls[seed].push(atom.0);
                    cpr_calls = cpr.2;
                    atom_calls = atom.2;
                    colours = cpr.1.n_colours();
                    if rep + 1 == repeats {
                        fmax = bench_error(system, &cpr.1, x.view()).max(bench_error(
                            system,
                            &atom.1,
                            x.view(),
                        ));
                    }
                }
                errors.push(fmax);
                assert!(
                    fmax < 1e-2,
                    "{} seed {seed} analytic error {fmax}",
                    system.name
                );
                let cpr_wall = median(&cpr_walls[seed]);
                let atom_wall = median(&atom_walls[seed]);
                lines.push(format!(
                    "{},cpr,{seed},1,1,{cpr_calls},{energy},{fmax},{cpr_wall}",
                    system.name
                ));
                lines.push(format!(
                    "{},one_atom,{seed},1,1,{atom_calls},{energy},{fmax},{atom_wall}",
                    system.name
                ));
                eprintln!(
                    "{} seed {seed} colours {colours} calls {cpr_calls}/{atom_calls} wall {cpr_wall:.6}/{atom_wall:.6} analytic {fmax}",
                    system.name
                );
            }
            if full {
                let cpr_all: Vec<f64> = cpr_walls.iter().flatten().copied().collect();
                let atom_all: Vec<f64> = atom_walls.iter().flatten().copied().collect();
                eprintln!(
                    "{} median wall cpr {} range {}..{} one_atom {} range {}..{} analytic max {}",
                    system.name,
                    median(&cpr_all),
                    cpr_all.iter().copied().fold(f64::MAX, f64::min),
                    cpr_all.iter().copied().fold(0.0, f64::max),
                    median(&atom_all),
                    atom_all.iter().copied().fold(f64::MAX, f64::min),
                    atom_all.iter().copied().fold(0.0, f64::max),
                    errors.iter().copied().fold(0.0, f64::max)
                );
            }
        }
        if full {
            let path = std::path::Path::new("target/sparse_hessian_bench.csv");
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::write(path, lines.join("\n") + "\n").unwrap();
        }
    }
}
