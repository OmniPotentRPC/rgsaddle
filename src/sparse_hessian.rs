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
//! [`sparse_hessian`] calls [`eval_displacements`] once per colour, in
//! colour order, with that colour's three Cartesian probes packed into
//! one batch. Chunks hold 4 geometries, independent of the thread count.
//! [`SparseHessian`] implements [`rgmin::ApplyHessian`] and exports the
//! same matrix through [`SparseHessian::to_dense`].

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

/// Full Lanczos basis at or below this dimension, so an index check on a
/// small Hessian matches a dense eigenpair.
const KRYLOV_FULL_MAX: usize = 96;

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

    /// Pairs within `interaction_range` under the minimum image of `cell`.
    ///
    /// `cell == None` is an open cluster: the distance is Euclidean.
    /// The range is inclusive. Coordinates are Cartesian, atom-major.
    pub fn from_positions(
        positions: ArrayView1<f64>,
        cell: Option<&Cell>,
        interaction_range: f64,
    ) -> Result<Self, SaddleError> {
        if !(interaction_range.is_finite() && interaction_range > 0.0) {
            return Err(SaddleError::Invalid(
                "interaction range must be positive and finite".into(),
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
        let n_atoms = positions.len() / 3;
        let range2 = interaction_range * interaction_range;
        let mut rows = vec![Vec::new(); n_atoms];
        for i in 0..n_atoms {
            let pi = coords_at(positions, i);
            for j in (i + 1)..n_atoms {
                let pj = coords_at(positions, j);
                let d2 = match cell {
                    Some(cell) => cell.dist2(pi, pj),
                    None => {
                        let dx = pi[0] - pj[0];
                        let dy = pi[1] - pj[1];
                        let dz = pi[2] - pj[2];
                        dx * dx + dy * dy + dz * dz
                    }
                };
                if d2 <= range2 {
                    rows[i].push(j);
                    rows[j].push(i);
                }
            }
        }
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
    let n = pattern.n_atoms();
    let mut alive = vec![true; n];
    let mut degree = Vec::with_capacity(n);
    for i in 0..n {
        degree.push(pattern.neighbors(i).len());
    }
    let mut removed = Vec::with_capacity(n);
    for _ in 0..n {
        let mut best = 0usize;
        while !alive[best] {
            best += 1;
        }
        for i in (best + 1)..n {
            if !alive[i] {
                continue;
            }
            if degree[i] < degree[best] || (degree[i] == degree[best] && i < best) {
                best = i;
            }
        }
        alive[best] = false;
        removed.push(best);
        for &u in pattern.neighbors(best) {
            if alive[u] {
                degree[u] -= 1;
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
/// `step` is the displacement `h`. An ECnoise estimate plugs in through
/// the same field.
#[derive(Clone, Copy, Debug)]
pub struct SparseHessianOpts {
    pub difference: FiniteDifference,
    pub step: f64,
}

impl Default for SparseHessianOpts {
    fn default() -> Self {
        Self {
            difference: FiniteDifference::Central,
            step: 1e-4,
        }
    }
}

/// Recovered atom-block Hessian.
///
/// Storage is CSR of 3x3 blocks, diagonal included, columns sorted.
/// Block `(i, j)` is row-major and was written by the colour of atom
/// `j` before symmetrisation. [`Self::matvec`] and the Lanczos basis
/// walk rows in index order.
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
    pub fn lowest_eigenpairs(&self, k: usize) -> Result<(Array1<f64>, Array2<f64>), SaddleError> {
        let n = self.dof();
        if k == 0 || n == 0 {
            return Ok((Array1::zeros(0), Array2::zeros((n, 0))));
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
        if x.len() != self.dof() || v.len() != self.dof() {
            return Array1::zeros(v.len());
        }
        self.matvec(v)
    }
}

/// Evaluate packed geometries, `n_geom` rows of `n_dof` coordinates.
///
/// The colour loop calls this once per colour, in colour order. A batch
/// evaluator can replace the body. With the `par` feature the rows run
/// in chunks of 4, combined in index order. Without it the same chunks
/// run in order on the calling thread.
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
    let (row_ptr, col_idx) = block_csr(pattern);
    let mut blocks = vec![[0.0; 9]; col_idx.len()];
    let x_base: Vec<f64> = x.iter().copied().collect();
    let probes = 3 * sides;
    let mut geometries = vec![0.0; probes * n_dof];
    let mut energies = vec![0.0; probes];
    let mut gradients = vec![0.0; probes * n_dof];
    let mut centre = vec![0.0; n_dof];
    let mut evaluations = 0usize;
    if opts.difference == FiniteDifference::Forward && n_dof > 0 {
        let mut energy = [0.0];
        eval_displacements(surface, &x_base, n_dof, &mut energy, &mut centre)?;
        evaluations += 1;
    }
    let mut seen = vec![0u32; n_atoms];
    let mut generation = 0u32;
    for group in groups {
        ensure_direct_partition(pattern, group, &mut seen, &mut generation)?;
        paint_colour(&mut geometries, &x_base, group, opts.step, sides);
        if n_dof > 0 {
            eval_displacements(surface, &geometries, n_dof, &mut energies, &mut gradients)?;
            evaluations += probes;
        }
        for axis in 0..3 {
            let plus = axis * sides;
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
    symmetrise(n_atoms, &row_ptr, &col_idx, &mut blocks)?;
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
    group: &[usize],
    seen: &mut [u32],
    generation: &mut u32,
) -> Result<(), SaddleError> {
    *generation = generation.wrapping_add(1);
    if *generation == 0 {
        seen.fill(0);
        *generation = 1;
    }
    let mark = *generation;
    for &j in group {
        if seen[j] == mark {
            return Err(SaddleError::Invalid(
                "colour is not a direct Curtis-Powell-Reid partition".into(),
            ));
        }
        seen[j] = mark;
        for &i in pattern.neighbors(j) {
            if seen[i] == mark {
                return Err(SaddleError::Invalid(
                    "colour is not a direct Curtis-Powell-Reid partition".into(),
                ));
            }
            seen[i] = mark;
        }
    }
    Ok(())
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
) -> Result<(Array1<f64>, Array2<f64>), SaddleError> {
    let m = if n <= KRYLOV_FULL_MAX {
        n
    } else {
        (k.saturating_mul(4).max(k.saturating_add(8))).min(n)
    };
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
        for i in 0..n {
            out_vecs[(i, r)] = y[i];
        }
    }
    Ok((out_vals, out_vecs))
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

    fn rel_block(got: &[f64; 9], reference: &[f64; 9]) -> f64 {
        let mut num = 0.0_f64;
        let mut den = 0.0_f64;
        for (g, r) in got.iter().zip(reference.iter()) {
            num = num.max((g - r).abs());
            den = den.max(r.abs());
        }
        num / den.max(1e-12)
    }

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

    fn lj_derivs(r: f64) -> (f64, f64) {
        let sr2 = 1.0 / (r * r);
        let sr6 = sr2 * sr2 * sr2;
        let sr12 = sr6 * sr6;
        let dv = 24.0 / r * (sr6 - 2.0 * sr12);
        let d2 = 24.0 * (26.0 * sr12 - 7.0 * sr6) / (r * r);
        (dv, d2)
    }

    struct Lj {
        pairs: Vec<(usize, usize)>,
        calls: AtomicUsize,
    }

    impl Lj {
        fn new(pattern: &SparsityPattern) -> Self {
            let mut pairs = Vec::new();
            for i in 0..pattern.n_atoms() {
                for &j in pattern.neighbors(i) {
                    if i < j {
                        pairs.push((i, j));
                    }
                }
            }
            Self {
                pairs,
                calls: AtomicUsize::new(0),
            }
        }
    }

    impl PointSurface for Lj {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            let mut g = Array1::zeros(x.len());
            let mut energy = 0.0;
            for &(i, j) in &self.pairs {
                let mut dr = [0.0; 3];
                let mut r2 = 0.0;
                for a in 0..3 {
                    dr[a] = x[3 * j + a] - x[3 * i + a];
                    r2 += dr[a] * dr[a];
                }
                let r = r2.sqrt();
                let sr2 = 1.0 / r2;
                let sr6 = sr2 * sr2 * sr2;
                let sr12 = sr6 * sr6;
                energy += 4.0 * (sr12 - sr6);
                let dv = 24.0 / r * (sr6 - 2.0 * sr12);
                for a in 0..3 {
                    let u = dr[a] / r;
                    g[3 * i + a] -= dv * u;
                    g[3 * j + a] += dv * u;
                }
            }
            Ok((energy, g))
        }
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
        let pattern = SparsityPattern::from_positions(x.view(), Some(&cell), 2.0).unwrap();
        assert!(pattern.neighbors(0).is_empty());
        let colouring = distance2_colouring(&pattern);
        assert_eq!(colouring.n_colours(), 1);
        assert_eq!(colouring.lower_bound(), 1);

        let wrapped = array![0.0, 0.0, 0.0, 0.9, 0.0, 0.0];
        let pattern = SparsityPattern::from_positions(wrapped.view(), Some(&cell), 0.3).unwrap();
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
                SparsityPattern::from_positions(positions.view(), Some(&cell), range).unwrap();
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
        let pattern = SparsityPattern::from_positions(coords.view(), None, 3.0).unwrap();
        assert!(pattern.neighbors(3).is_empty());
        let surface = Lj::new(&pattern);
        let opts = SparseHessianOpts {
            difference: FiniteDifference::Central,
            step: 1e-6,
        };
        let got = sparse_hessian(&surface, coords.view(), &pattern, opts).unwrap();
        assert_eq!(got.evaluations(), surface.calls.load(Ordering::Relaxed));
        assert_eq!(
            got.evaluations(),
            got.n_colours() * 3 * 2,
            "central differences have no centre evaluation"
        );

        let mut reference = vec![[0.0; 9]; got.blocks().len()];
        for i in 0..pattern.n_atoms() {
            for &j in pattern.neighbors(i) {
                if j < i {
                    continue;
                }
                let mut dr = [0.0; 3];
                let mut r2 = 0.0;
                for a in 0..3 {
                    dr[a] = coords[3 * j + a] - coords[3 * i + a];
                    r2 += dr[a] * dr[a];
                }
                let r = r2.sqrt();
                let (dv, d2) = lj_derivs(r);
                let u = [dr[0] / r, dr[1] / r, dr[2] / r];
                let block = pair_block(d2, dv, r, u);
                let p_ij = block_index(got.row_ptr(), got.col_idx(), i, j).unwrap();
                let p_ji = block_index(got.row_ptr(), got.col_idx(), j, i).unwrap();
                let p_ii = block_index(got.row_ptr(), got.col_idx(), i, i).unwrap();
                let p_jj = block_index(got.row_ptr(), got.col_idx(), j, j).unwrap();
                for a in 0..3 {
                    for c in 0..3 {
                        reference[p_ij][a * 3 + c] -= block[a * 3 + c];
                        reference[p_ji][c * 3 + a] -= block[a * 3 + c];
                        reference[p_ii][a * 3 + c] += block[a * 3 + c];
                        reference[p_jj][a * 3 + c] += block[a * 3 + c];
                    }
                }
            }
        }
        for (got_block, reference_block) in got.blocks().iter().zip(reference.iter()) {
            let err = rel_block(got_block, reference_block);
            assert!(err < 1e-6, "relative block error {err}");
        }
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
        let pattern = SparsityPattern::from_positions(coords.view(), None, 2.0).unwrap();
        let surface = Lj::new(&pattern);
        let opts = SparseHessianOpts {
            difference: FiniteDifference::Forward,
            step: 1e-5,
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
            },
        )
        .unwrap();
        let dense = dense_central(&DoubleWell, x.view(), step);
        let (values, vectors) = got.lowest_eigenpairs(1).unwrap();
        lowest_overlap(&values, &vectors, &dense);
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
        let pattern = SparsityPattern::from_positions(coords.view(), None, 2.0).unwrap();
        assert_eq!(pattern.neighbors(0), &[1]);
        let surface = Lj::new(&pattern);
        let step = 1e-6;
        let opts = SparseHessianOpts {
            difference: FiniteDifference::Central,
            step,
        };
        let got = sparse_hessian(&surface, coords.view(), &pattern, opts).unwrap();
        let dense = got.to_dense();
        for i in 0..dense.nrows() {
            for j in 0..dense.ncols() {
                assert_eq!(dense[(i, j)], dense[(j, i)]);
            }
        }
        let reference = dense_central(&surface, coords.view(), step);
        let (values, vectors) = got.lowest_eigenpairs(1).unwrap();
        lowest_overlap(&values, &vectors, &reference);
        assert!(values[0] < -1e-3, "lowest curvature {}", values[0]);
    }

    #[test]
    fn projected_hessian_removes_a_translation() {
        let coords = array![0.0, 0.0, 0.0, 1.3, 0.0, 0.0];
        let pattern = SparsityPattern::from_positions(coords.view(), None, 2.0).unwrap();
        let surface = Translations {
            inner: Lj::new(&pattern),
        };
        let got = sparse_hessian(
            &surface,
            coords.view(),
            &pattern,
            SparseHessianOpts {
                difference: FiniteDifference::Central,
                step: 1e-6,
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
        let pattern = SparsityPattern::from_positions(coords.view(), None, 2.0).unwrap();
        let surface = Lj::new(&pattern);
        let opts = SparseHessianOpts {
            difference: FiniteDifference::Central,
            step: 1e-6,
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
        let pattern = SparsityPattern::from_positions(x.view(), None, 2.0).unwrap();
        let colouring = distance2_colouring(&pattern);
        assert!(colouring.n_colours() < 5);
        let sparse_surface = Lj::new(&pattern);
        let dense_surface = Lj::new(&pattern);
        let mut sparse = Index1Session::new(
            Index1Config {
                sparse_pattern: Some(pattern),
                force_tol: 0.0,
                trust_radius: 1e-3,
                fd_dr: 1e-4,
                ..Index1Config::default()
            },
            x.clone(),
            None,
            None,
        )
        .unwrap();
        sparse.step(&sparse_surface).unwrap();
        let sparse_calls = sparse_surface.calls.load(Ordering::Relaxed);
        assert_eq!(sparse_calls, colouring.n_colours() * 3 * 2 + 2);

        let mut dense = Index1Session::new(
            Index1Config {
                sparse_pattern: None,
                force_tol: 0.0,
                trust_radius: 1e-3,
                fd_dr: 1e-4,
                ..Index1Config::default()
            },
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
        let (values, vectors) = h.lowest_eigenpairs(1).unwrap();
        let mut words = values.to_vec();
        words.extend(vectors.iter().copied());
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
        let cutoff = 2f64.powf(1.0 / 6.0) * 1.2;
        let pattern = SparsityPattern::from_positions(coords.view(), Some(&cell), cutoff).unwrap();
        let surface = Lj::new(&pattern);
        let opts = SparseHessianOpts {
            difference: FiniteDifference::Central,
            step: 1e-6,
        };
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
    fn colour_loop_strong_scaling_on_the_2048_atom_cell() {
        let a = 2f64.powf(1.0 / 6.0) * 2f64.sqrt();
        let coords = Array1::from(fcc_open(8, 8, 8, a));
        assert_eq!(coords.len() / 3, 2048);
        let cell = Cell::ortho(8.0 * a, 8.0 * a, 8.0 * a).unwrap();
        let cutoff = 2f64.powf(1.0 / 6.0) * 1.2;
        let pattern = SparsityPattern::from_positions(coords.view(), Some(&cell), cutoff).unwrap();
        let surface = Lj::new(&pattern);
        let opts = SparseHessianOpts {
            difference: FiniteDifference::Central,
            step: 1e-6,
        };
        let max_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let mut counts = vec![1usize, 2, 4, max_threads];
        counts.sort_unstable();
        counts.dedup();
        let mut samples = vec![Vec::new(); counts.len()];
        let mut hashes = Vec::new();
        for _rep in 0..5 {
            for (i, &threads) in counts.iter().enumerate() {
                let started = std::time::Instant::now();
                let recovered = with_threads(threads, || {
                    sparse_hessian(&surface, coords.view(), &pattern, opts).unwrap()
                });
                samples[i].push(started.elapsed().as_secs_f64());
                if hashes.len() == i {
                    hashes.push(fnv1a(&csr_words(&recovered)));
                }
            }
        }
        assert!(hashes.iter().all(|h| *h == hashes[0]));
        let t1 = median(&samples[0]);
        eprintln!("colour-loop scaling on lj_fcc_2048 (cutoff LJ, EMT/EAM is not linked)");
        eprintln!("threads median_s min_s max_s efficiency");
        for (i, &threads) in counts.iter().enumerate() {
            let med = median(&samples[i]);
            let min = samples[i].iter().copied().fold(f64::MAX, f64::min);
            let max = samples[i].iter().copied().fold(0.0, f64::max);
            let efficiency = t1 / (threads as f64 * med);
            eprintln!("{threads} {med:.6} {min:.6} {max:.6} {efficiency:.3}");
        }
        eprintln!(
            "limiter: colour blocks run in colour order, one batch of 3 axes times 2 sides, so extra threads only split that short batch and the neighbor-list force"
        );
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
                for a in 0..3 {
                    let d = positions[3 * i + a] - positions[3 * j + a];
                    r2 += d * d;
                }
                best = best.min(r2);
            }
        }
        best.sqrt()
    }

    struct SystemSpec {
        name: &'static str,
        positions: Vec<f64>,
        cell: Option<Cell>,
        sigma: f64,
        cutoff: f64,
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
            for a in 0..3 {
                let pi = all[3 * i + a] - c[a];
                let pj = all[3 * j + a] - c[a];
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
                    for b in basis {
                        out.push((ix as f64 + b[0]) * a);
                        out.push((iy as f64 + b[1]) * a);
                        out.push((iz as f64 + b[2]) * a);
                    }
                }
            }
        }
        out
    }

    fn vacancy(n: usize, a: f64) -> (Vec<f64>, Cell, f64, f64) {
        let mut positions = fcc_open(n, n, n, a);
        positions.drain(0..3);
        let nn = a / 2f64.sqrt();
        let sigma = nn / 2f64.powf(1.0 / 6.0);
        let cell = Cell::ortho(n as f64 * a, n as f64 * a, n as f64 * a).unwrap();
        (positions, cell, sigma, nn * 1.2)
    }

    fn bench_systems() -> Vec<SystemSpec> {
        let nn = 2f64.powf(1.0 / 6.0);
        let lj38 = SystemSpec {
            name: "lj38",
            positions: lj38_positions(),
            cell: None,
            sigma: 1.0,
            cutoff: nn * 1.2,
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
        let pt = SystemSpec {
            name: "pt_heptamer",
            positions: pt_raw.to_vec(),
            cell: None,
            sigma: pt_nn / nn,
            cutoff: pt_nn * 1.2,
        };
        let (al_pos, al_cell, al_sigma, al_cut) = vacancy(5, 4.05);
        let al = SystemSpec {
            name: "al_vacancy",
            positions: al_pos,
            cell: Some(al_cell),
            sigma: al_sigma,
            cutoff: al_cut,
        };
        let (cu_pos, cu_cell, cu_sigma, cu_cut) = vacancy(5, 3.61);
        let cu = SystemSpec {
            name: "cu_vacancy",
            positions: cu_pos,
            cell: Some(cu_cell),
            sigma: cu_sigma,
            cutoff: cu_cut,
        };
        let a = nn * 2f64.sqrt();
        let big = SystemSpec {
            name: "lj_fcc_2048",
            positions: fcc_open(8, 8, 8, a),
            cell: Some(Cell::ortho(8.0 * a, 8.0 * a, 8.0 * a).unwrap()),
            sigma: 1.0,
            cutoff: nn * 1.2,
        };
        vec![lj38, pt, al, cu, big]
    }

    struct ScaledLj {
        pairs: Vec<(usize, usize)>,
        sigma: f64,
        cell: Option<Cell>,
    }

    impl PointSurface for ScaledLj {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let mut g = Array1::zeros(x.len());
            let mut energy = 0.0;
            let sig2 = self.sigma * self.sigma;
            for &(i, j) in &self.pairs {
                let pi = [x[3 * i], x[3 * i + 1], x[3 * i + 2]];
                let pj = [x[3 * j], x[3 * j + 1], x[3 * j + 2]];
                let dr = match self.cell {
                    Some(cell) => cell.displacement(pi, pj),
                    None => [pj[0] - pi[0], pj[1] - pi[1], pj[2] - pi[2]],
                };
                let r2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                let r = r2.sqrt();
                let sr2 = sig2 / r2;
                let sr6 = sr2 * sr2 * sr2;
                let sr12 = sr6 * sr6;
                energy += 4.0 * (sr12 - sr6);
                let dv = 24.0 / r * (sr6 - 2.0 * sr12);
                for a in 0..3 {
                    let u = dr[a] / r;
                    g[3 * i + a] -= dv * u;
                    g[3 * j + a] += dv * u;
                }
            }
            Ok((energy, g))
        }
    }

    fn rattle(positions: &[f64], seed: u64, amp: f64) -> Vec<f64> {
        positions
            .iter()
            .enumerate()
            .map(|(i, &p)| p + (u01(seed, i as u64) - 0.5) * 2.0 * amp)
            .collect()
    }

    fn max_block_diff(a: &SparseHessian, b: &SparseHessian) -> f64 {
        a.blocks()
            .iter()
            .zip(b.blocks())
            .flat_map(|(p, q)| p.iter().zip(q.iter()))
            .fold(0.0, |m, (u, v)| m.max((u - v).abs()))
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
                "{} atoms {} sigma {} cutoff {}",
                system.name,
                system.positions.len() / 3,
                system.sigma,
                system.cutoff
            );
            let mut cpr_walls: Vec<Vec<f64>> = vec![Vec::new(); rattles];
            let mut atom_walls: Vec<Vec<f64>> = vec![Vec::new(); rattles];
            for seed in 0..rattles {
                let amp = 0.01 * system.cutoff;
                let rattled = rattle(&system.positions, seed as u64, amp);
                let x = Array1::from(rattled);
                let pattern =
                    SparsityPattern::from_positions(x.view(), system.cell.as_ref(), system.cutoff)
                        .unwrap();
                let surface = ScaledLj {
                    pairs: pairs_of(&pattern),
                    sigma: system.sigma,
                    cell: system.cell,
                };
                let (energy, _) = surface.eval(x.view()).unwrap();
                let opts = SparseHessianOpts {
                    difference: FiniteDifference::Central,
                    step: 1e-6,
                };
                let mut fmax = 0.0;
                let mut cpr_calls = 0usize;
                let mut atom_calls = 0usize;
                let mut colours = 0usize;
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
                    fmax = max_block_diff(&cpr.1, &atom.1);
                    cpr_calls = cpr.2;
                    atom_calls = atom.2;
                    colours = cpr.1.n_colours();
                    assert!(
                        fmax < 1e-8,
                        "{} seed {seed} max block error {fmax}",
                        system.name
                    );
                }
                let cpr_wall = median(&cpr_walls[seed]);
                let atom_wall = median(&atom_walls[seed]);
                lines.push(format!(
                    "{},cpr,{},1,1,{},{energy},{fmax},{cpr_wall}",
                    system.name, seed, cpr_calls
                ));
                lines.push(format!(
                    "{},one_atom,{},1,1,{},{energy},0,{atom_wall}",
                    system.name, seed, atom_calls
                ));
                eprintln!(
                    "{} seed {seed} colours {colours} calls {cpr_calls}/{atom_calls} wall {cpr_wall:.6}/{atom_wall:.6} fmax {fmax}",
                    system.name
                );
            }
            if full {
                let cpr_all: Vec<f64> = cpr_walls.iter().flatten().copied().collect();
                let atom_all: Vec<f64> = atom_walls.iter().flatten().copied().collect();
                eprintln!(
                    "{} median wall cpr {} range {}..{} one_atom {} range {}..{}",
                    system.name,
                    median(&cpr_all),
                    cpr_all.iter().copied().fold(f64::MAX, f64::min),
                    cpr_all.iter().copied().fold(0.0, f64::max),
                    median(&atom_all),
                    atom_all.iter().copied().fold(f64::MAX, f64::min),
                    atom_all.iter().copied().fold(0.0, f64::max)
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

    fn pairs_of(pattern: &SparsityPattern) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        for i in 0..pattern.n_atoms() {
            for &j in pattern.neighbors(i) {
                if i < j {
                    pairs.push((i, j));
                }
            }
        }
        pairs
    }
}
