//! Star colouring of a symmetric Hessian sparsity pattern.
//!
//! A star colouring is a proper vertex colouring in which every path on
//! four vertices uses at least three colours. Coleman and Moré showed that
//! this is a direct symmetric partition of a sparse Hessian (Math.
//! Programming 28, 243-270, 1984, doi:10.1007/BF02612334). The greedy
//! assignment used here follows Gebremedhin, Tarafdar, Manne and Pothen
//! (SIAM J. Sci. Comput. 29, 1042-1072, 2007, doi:10.1137/050639879).
//! Vertices are coloured in index order, so the result depends only on the
//! edge set. This module builds the pattern and the seed matrix. It does
//! not evaluate a surface.

use ndarray::{Array1, Array2, ArrayView1};

use crate::error::SaddleError;
use crate::mic::{self, Cell};

/// Star colouring of an undirected graph on `n` vertices.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StarColouring {
    colour: Vec<usize>,
    groups: Vec<Vec<usize>>,
}

impl StarColouring {
    /// Number of vertices.
    pub fn n(&self) -> usize {
        self.colour.len()
    }

    /// Number of colours. Isolated vertices share colour 0, so an empty
    /// graph on `n > 0` vertices has one colour.
    pub fn n_colours(&self) -> usize {
        self.groups.len()
    }

    /// Colour of each vertex, in `0..n_colours`.
    pub fn colour(&self) -> &[usize] {
        &self.colour
    }

    /// Vertices of each colour, in increasing index order.
    pub fn groups(&self) -> &[Vec<usize>] {
        &self.groups
    }

    /// Seed matrix `S` with shape `(n, n_colours)`.
    ///
    /// Column `c` is 1 on the vertices of colour `c` and 0 elsewhere.
    /// A directional derivative along that column compresses the Hessian
    /// columns of the colour.
    pub fn seed_matrix(&self) -> Array2<f64> {
        let mut seed = Array2::<f64>::zeros((self.n(), self.n_colours()));
        for (c, group) in self.groups.iter().enumerate() {
            for &vertex in group {
                seed[(vertex, c)] = 1.0;
            }
        }
        seed
    }
}

/// Greedy star colouring of an undirected graph.
///
/// `edges` are unordered pairs. A self-pair is ignored. Endpoints outside
/// `0..n` are an error. Duplicate pairs do not change the colouring.
pub fn star_colour(n: usize, edges: &[(usize, usize)]) -> Result<StarColouring, SaddleError> {
    let adj = adjacency(n, edges)?;
    let colour = greedy_star(&adj);
    Ok(pack(colour))
}

/// `true` when `colour` is a star colouring of the same graph.
pub fn is_star_colouring(
    n: usize,
    edges: &[(usize, usize)],
    colour: &[usize],
) -> Result<bool, SaddleError> {
    if colour.len() != n {
        return Err(SaddleError::Shape(
            "colouring length does not match the vertex count".into(),
        ));
    }
    let adj = adjacency(n, edges)?;
    if !proper(&adj, colour) {
        return Ok(false);
    }
    Ok(!has_bichromatic_path4(&adj, colour))
}

/// Coordinate pairs of a block-sparse Cartesian Hessian.
///
/// Each atom contributes a dense 3 by 3 diagonal block. Each atom pair
/// contributes a dense 3 by 3 off-diagonal block. The returned pairs index
/// the `3 n_atoms` Cartesian coordinates.
pub fn coordinate_edges(
    n_atoms: usize,
    atom_pairs: &[(usize, usize)],
) -> Result<Vec<(usize, usize)>, SaddleError> {
    let mut edges = Vec::new();
    for atom in 0..n_atoms {
        let base = 3 * atom;
        edges.push((base, base + 1));
        edges.push((base, base + 2));
        edges.push((base + 1, base + 2));
    }
    for &(i, j) in atom_pairs {
        if i >= n_atoms || j >= n_atoms {
            return Err(SaddleError::Shape(format!(
                "atom pair ({i}, {j}) is outside 0..{n_atoms}"
            )));
        }
        if i == j {
            continue;
        }
        for alpha in 0..3 {
            for beta in 0..3 {
                push_unordered(&mut edges, 3 * i + alpha, 3 * j + beta);
            }
        }
    }
    Ok(edges)
}

/// Atom pairs at distance `<= cutoff`.
///
/// `positions` is a `3 n_atoms` Cartesian vector. With a cell, the distance
/// is the minimum image. The pairs are ordered `i < j`.
pub fn interacting_atom_pairs(
    positions: ArrayView1<f64>,
    cutoff: f64,
    cell: Option<&Cell>,
) -> Result<Vec<(usize, usize)>, SaddleError> {
    if !positions.len().is_multiple_of(3) {
        return Err(SaddleError::Shape(
            "positions length must be a multiple of 3".into(),
        ));
    }
    if !(cutoff.is_finite() && cutoff >= 0.0) {
        return Err(SaddleError::Invalid(
            "cutoff must be finite and non-negative".into(),
        ));
    }
    if positions.iter().any(|v| !v.is_finite()) {
        return Err(SaddleError::NonFinite("interacting atom positions"));
    }
    let n_atoms = positions.len() / 3;
    let limit = cutoff * cutoff;
    let mut pairs = Vec::new();
    for i in 0..n_atoms {
        let a = [positions[3 * i], positions[3 * i + 1], positions[3 * i + 2]];
        for j in (i + 1)..n_atoms {
            let b = [positions[3 * j], positions[3 * j + 1], positions[3 * j + 2]];
            if min_image_distance2(a, b, cell) <= limit {
                pairs.push((i, j));
            }
        }
    }
    Ok(pairs)
}

/// Seed matrix for one Cartesian displacement per atom colour and axis.
///
/// The colouring is an atom colouring (`n_atoms` vertices). Column
/// `3 * colour + axis` is 1 on coordinate `3 * atom + axis` for every atom
/// of that colour. The shape is `(3 n_atoms, 3 n_colours)`.
pub fn atom_axis_seeds(
    n_atoms: usize,
    colouring: &StarColouring,
) -> Result<Array2<f64>, SaddleError> {
    if colouring.n() != n_atoms {
        return Err(SaddleError::Shape(
            "atom colouring length does not match the atom count".into(),
        ));
    }
    let n_colours = colouring.n_colours();
    let mut seed = Array2::<f64>::zeros((3 * n_atoms, 3 * n_colours));
    for (c, group) in colouring.groups().iter().enumerate() {
        for &atom in group {
            for axis in 0..3 {
                seed[(3 * atom + axis, 3 * c + axis)] = 1.0;
            }
        }
    }
    Ok(seed)
}

fn pack(colour: Vec<usize>) -> StarColouring {
    let n_colours = colour.iter().copied().max().map_or(0, |c| c + 1);
    let mut groups = vec![Vec::new(); n_colours];
    for (vertex, &c) in colour.iter().enumerate() {
        groups[c].push(vertex);
    }
    StarColouring { colour, groups }
}

fn adjacency(n: usize, edges: &[(usize, usize)]) -> Result<Vec<Vec<usize>>, SaddleError> {
    let mut adj = vec![Vec::new(); n];
    for &(i, j) in edges {
        if i >= n || j >= n {
            return Err(SaddleError::Shape(format!(
                "edge endpoint ({i}, {j}) is outside 0..{n}"
            )));
        }
        if i == j {
            continue;
        }
        adj[i].push(j);
        adj[j].push(i);
    }
    for neighbors in &mut adj {
        neighbors.sort_unstable();
        neighbors.dedup();
    }
    Ok(adj)
}

fn push_unordered(edges: &mut Vec<(usize, usize)>, i: usize, j: usize) {
    if i < j {
        edges.push((i, j));
    } else if j < i {
        edges.push((j, i));
    }
}

/// Smallest-colour greedy star colouring.
///
/// A colour already on a neighbour is forbidden. A colour on a vertex `x`
/// is also forbidden when assigning it here would close a two-coloured
/// path on four vertices through a neighbour of `x`.
fn greedy_star(adj: &[Vec<usize>]) -> Vec<usize> {
    let n = adj.len();
    const UNCOLOURED: usize = usize::MAX;
    let mut colour = vec![UNCOLOURED; n];
    let mut forbidden = vec![0usize; n + 1];
    for v in 0..n {
        let stamp = v + 1;
        for &w in &adj[v] {
            let cw = colour[w];
            if cw != UNCOLOURED {
                forbidden[cw] = stamp;
            }
            for &x in &adj[w] {
                if x == v {
                    continue;
                }
                let cx = colour[x];
                if cx == UNCOLOURED {
                    continue;
                }
                if cw == UNCOLOURED || adj[x].iter().any(|&y| y != w && colour[y] == cw) {
                    forbidden[cx] = stamp;
                }
            }
        }
        let mut c = 0;
        while forbidden[c] == stamp {
            c += 1;
        }
        colour[v] = c;
    }
    colour
}

fn proper(adj: &[Vec<usize>], colour: &[usize]) -> bool {
    adj.iter().enumerate().all(|(v, neighbors)| {
        neighbors
            .iter()
            .filter(|&&w| w > v)
            .all(|&w| colour[v] != colour[w])
    })
}

fn has_bichromatic_path4(adj: &[Vec<usize>], colour: &[usize]) -> bool {
    for (a, neighbors) in adj.iter().enumerate() {
        for &b in neighbors {
            if b < a {
                continue;
            }
            for &u in neighbors {
                if u == b {
                    continue;
                }
                for &v in &adj[b] {
                    if v == a || v == u {
                        continue;
                    }
                    let used = [colour[u], colour[a], colour[b], colour[v]];
                    if distinct_count(&used) < 3 {
                        return true;
                    }
                }
            }
        }
    }
    false
}

fn distinct_count(values: &[usize; 4]) -> usize {
    let mut n = 1;
    for i in 1..4 {
        if values[..i].iter().all(|&seen| seen != values[i]) {
            n += 1;
        }
    }
    n
}

fn min_image_distance2(a: [f64; 3], b: [f64; 3], cell: Option<&Cell>) -> f64 {
    let mut delta = Array1::from(vec![b[0] - a[0], b[1] - a[1], b[2] - a[2]]);
    if let Some(cell) = cell {
        mic::wrap_difference(cell, &mut delta);
    }
    delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn path_edges(n: usize) -> Vec<(usize, usize)> {
        (0..n.saturating_sub(1)).map(|i| (i, i + 1)).collect()
    }

    fn assert_star(n: usize, edges: &[(usize, usize)], colouring: &StarColouring) {
        assert_eq!(colouring.n(), n);
        assert!(is_star_colouring(n, edges, colouring.colour()).unwrap());
        let k = colouring.n_colours();
        assert!(k == 0 || colouring.colour().iter().all(|&c| c < k));
        let mut seen = vec![false; n];
        for (c, group) in colouring.groups().iter().enumerate() {
            for &v in group {
                assert!(!seen[v]);
                seen[v] = true;
                assert_eq!(colouring.colour()[v], c);
            }
        }
        assert!(seen.into_iter().all(|flag| flag));
    }

    #[test]
    fn isolated_vertices_share_one_colour() {
        let colouring = star_colour(5, &[]).unwrap();
        assert_eq!(colouring.n_colours(), 1);
        assert_eq!(colouring.colour(), &[0, 0, 0, 0, 0]);
        assert_star(5, &[], &colouring);
        let seed = colouring.seed_matrix();
        assert_eq!(seed.shape(), &[5, 1]);
        assert!(seed.iter().all(|v| *v == 1.0));
    }

    #[test]
    fn empty_graph_has_no_colours() {
        let colouring = star_colour(0, &[]).unwrap();
        assert_eq!(colouring.n(), 0);
        assert_eq!(colouring.n_colours(), 0);
        assert_eq!(colouring.seed_matrix().shape(), &[0, 0]);
    }

    #[test]
    fn a_path_uses_three_colours() {
        let edges = path_edges(7);
        let colouring = star_colour(7, &edges).unwrap();
        assert_eq!(colouring.n_colours(), 3);
        assert_eq!(colouring.colour(), &[0, 1, 0, 2, 0, 1, 0]);
        assert_star(7, &edges, &colouring);
    }

    #[test]
    fn a_star_uses_two_colours() {
        let edges: Vec<_> = (1..9).map(|leaf| (0, leaf)).collect();
        let colouring = star_colour(9, &edges).unwrap();
        assert_eq!(colouring.n_colours(), 2);
        assert_eq!(colouring.colour()[0], 0);
        assert!(colouring.colour()[1..].iter().all(|&c| c == 1));
        assert_star(9, &edges, &colouring);
    }

    #[test]
    fn a_four_cycle_is_not_two_coloured() {
        let edges = [(0, 1), (1, 2), (2, 3), (3, 0)];
        let colouring = star_colour(4, &edges).unwrap();
        assert_eq!(colouring.n_colours(), 3);
        assert_star(4, &edges, &colouring);
        assert!(!is_star_colouring(4, &edges, &[0, 1, 0, 1]).unwrap());
    }

    #[test]
    fn a_complete_graph_uses_one_colour_per_vertex() {
        let n = 6;
        let mut edges = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                edges.push((i, j));
            }
        }
        let colouring = star_colour(n, &edges).unwrap();
        assert_eq!(colouring.n_colours(), n);
        assert_star(n, &edges, &colouring);
    }

    #[test]
    fn edge_order_does_not_change_the_colouring() {
        let forward = path_edges(6);
        let mut reverse = forward.clone();
        reverse.reverse();
        let mut swapped: Vec<_> = forward.iter().map(|&(i, j)| (j, i)).collect();
        swapped.push((2, 2));
        swapped.push((1, 2));
        assert_eq!(
            star_colour(6, &forward).unwrap(),
            star_colour(6, &swapped).unwrap()
        );
        assert_eq!(
            star_colour(6, &forward).unwrap(),
            star_colour(6, &reverse).unwrap()
        );
    }

    #[test]
    fn an_endpoint_outside_the_graph_is_rejected() {
        let err = star_colour(3, &[(0, 3)]).unwrap_err();
        assert!(matches!(err, SaddleError::Shape(_)));
    }

    #[test]
    fn random_graphs_stay_star_coloured() {
        let mut state = 0x5eed_u64;
        for n in [1usize, 2, 5, 8, 12] {
            for _ in 0..8 {
                let mut edges = Vec::new();
                for i in 0..n {
                    for j in (i + 1)..n {
                        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                        if state % 3 == 0 {
                            edges.push((i, j));
                        }
                    }
                }
                let colouring = star_colour(n, &edges).unwrap();
                assert_star(n, &edges, &colouring);
            }
        }
    }

    #[test]
    fn two_separated_atoms_have_two_diagonal_blocks() {
        let edges = coordinate_edges(2, &[]).unwrap();
        let colouring = star_colour(6, &edges).unwrap();
        assert_eq!(colouring.n_colours(), 3);
        assert_star(6, &edges, &colouring);
        assert_eq!(colouring.colour()[0], colouring.colour()[3]);
        assert_eq!(colouring.colour()[1], colouring.colour()[4]);
        assert_eq!(colouring.colour()[2], colouring.colour()[5]);
    }

    #[test]
    fn a_dimer_block_is_complete() {
        let edges = coordinate_edges(2, &[(0, 1)]).unwrap();
        let colouring = star_colour(6, &edges).unwrap();
        assert_eq!(colouring.n_colours(), 6);
        assert_star(6, &edges, &colouring);
    }

    #[test]
    fn atom_axis_seeds_displace_one_colour_along_one_axis() {
        let edges: Vec<_> = (1..4).map(|leaf| (0, leaf)).collect();
        let colouring = star_colour(4, &edges).unwrap();
        let seed = atom_axis_seeds(4, &colouring).unwrap();
        assert_eq!(seed.shape(), &[12, 6]);
        assert_eq!(seed[(0, 0)], 1.0);
        assert_eq!(seed[(1, 0)], 0.0);
        assert_eq!(seed[(3, 3)], 1.0);
        assert!((seed.column(0).sum() - 1.0).abs() < 1e-12);
        let leaves = colouring.groups()[1].len() as f64;
        assert!((seed.column(3).sum() - leaves).abs() < 1e-12);
    }

    #[test]
    fn cutoff_pairs_use_the_minimum_image() {
        let positions = array![0.2, 0.0, 0.0, 9.9, 0.0, 0.0, 5.0, 0.0, 0.0];
        let open = interacting_atom_pairs(positions.view(), 1.0, None).unwrap();
        assert!(open.is_empty());
        let cell = Cell::ortho(10.0, 10.0, 10.0).unwrap();
        let wrapped = interacting_atom_pairs(positions.view(), 1.0, Some(&cell)).unwrap();
        assert_eq!(wrapped, vec![(0, 1)]);
        let err = interacting_atom_pairs(positions.view(), f64::NAN, None).unwrap_err();
        assert!(matches!(err, SaddleError::Invalid(_)));
    }

    #[test]
    fn a_cubic_lattice_uses_fewer_colours_than_columns() {
        let l = 12;
        let n = l * l * l;
        let index = |x, y, z| x + l * (y + l * z);
        let mut edges = Vec::new();
        for x in 0..l {
            for y in 0..l {
                for z in 0..l {
                    let i = index(x, y, z);
                    if x + 1 < l {
                        edges.push((i, index(x + 1, y, z)));
                    }
                    if y + 1 < l {
                        edges.push((i, index(x, y + 1, z)));
                    }
                    if z + 1 < l {
                        edges.push((i, index(x, y, z + 1)));
                    }
                }
            }
        }
        let started = std::time::Instant::now();
        let colouring = star_colour(n, &edges).unwrap();
        let elapsed = started.elapsed();
        assert_star(n, &edges, &colouring);
        assert!(colouring.n_colours() < n / 10);
        eprintln!(
            "cubic lattice L={l} atoms={n} edges={} colours={} elapsed_us={}",
            edges.len(),
            colouring.n_colours(),
            elapsed.as_micros()
        );
    }
}
