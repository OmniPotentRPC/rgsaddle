//! Automatic redundant internals from a Cartesian geometry.
//!
//! [`InternalTopology::from_geometry`] follows the construction of
//! Hermes, Sargsyan, Najm, and Zador, J. Chem. Theory Comput. 18, 6974
//! (2022), doi:10.1021/acs.jctc.2c00395, section 2.1:
//!
//! 1. Two atoms are bonded when their distance is under `sigma` times
//!    the sum of their covalent radii, `sigma = 1.25`. Fragments that
//!    stay apart are welded: `sigma` grows by 5 % and only pairs in
//!    different fragments are tried, until one fragment is left.
//! 2. Every graph edge is a bond. Two bonds that share an atom form an
//!    angle with the shared atom at the vertex. Two angles that share
//!    an edge and chain through it form a dihedral; a pair that would
//!    close an improper dihedral (H-N-H-H in ammonia) is not added.
//! 3. A bend within 15 degrees of 0 or 180 is not added. A vertex with
//!    three or more bonds takes the improper dihedral `a-b-d-c`, where
//!    `d` is the bonded atom closest to the vertex `b` other than `a`
//!    and `c`. A vertex bonded only to `a` and `c` gets a dummy atom 1
//!    length unit from `b` along `ba x bc`, pinned by a fixed bond and
//!    two 90 degree angles, and the improper dihedral `a-b-x-c`. When
//!    `|ba x bc|` is under one ten-thousandth for unit vectors, the dummy goes along
//!    `(I - ac ac^T) w`, `w` the Cartesian axis nearest to orthogonal
//!    to `ac`.
//!
//! The covalent radii are Cordero et al. (2008), see [`crate::covalent`].
//! Periodic cells enter through lattice images: a bond, angle or
//! dihedral stores the integer image of each atom, and the minimum
//! image is searched over every shift that can fall inside the cutoff.
//!
//! Three points are this crate's choice where the paper is silent. A
//! pair of bonds that is exactly equal in length is ordered by atomic
//! number and then by index, so the dummy side is reproducible. A
//! vertex with three or more bonds whose improper dihedral is undefined
//! (every other bond lies along the same line) falls back to the dummy
//! atom, as Sella's later releases do.
//!
//! The third is the near-planar centre. Bonds and angles alone fold at a
//! planar centre with three neighbours: the umbrella motion of ammonia
//! moves their sum to second order only, so a search that crosses the
//! plane meets a chart with no first-order handle on it. A three-neighbour
//! centre that sits inside no torsion gets the improper dihedral once
//! its three bends come within [`TopologyConfig::planar_tol`] of a full
//! turn, and [`InternalTopology::planar_watch`] lists the centres still
//! clear of that window so a search can rebuild when one enters it.
//! Sella 2.6 adds an improper at every such centre whatever its
//! geometry; this crate does not, and the chart still has full rank.
//!
//! The set is redundant by design. [`TopologyCheck`] measures the rank
//! of the Wilson matrix against the rigid quotient of the frame
//! (`3N - 6`, `3N - 5` for a line, `3N - 3` when the bond graph wraps
//! the cell) and whether every internal is invariant under the rigid
//! motions.

use std::collections::{BTreeSet, VecDeque};

use ndarray::{Array1, Array2, ArrayView1};
use rgmin::{Manifold, ManifoldKind};

use crate::cell_log::{lattice_of, m3_inv};
use crate::constraints::{Constraints, Equality};
use crate::covalent::{atomic_number, covalent_radius, element_symbol};
use crate::eigensolve::exact_eigh;
use crate::error::SaddleError;
use crate::mic::Cell;

/// Sella's cutoff on the singular values of the Wilson matrix: a value
/// under one millionth counts as zero.
pub const PAPER_SV_TOL: f64 = 1.0 / 1_000_000.0;

/// The paper's cutoff on `|ba x bc|` for unit vectors, under which the
/// dummy direction comes from the Cartesian axis rule.
const PAPER_CROSS_TOL: f64 = 1.0 / 10_000.0;

type Shift = [i32; 3];

/// A neighbour of an atom: the neighbour's index and the lattice image
/// it sits in, relative to its owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Nbr {
    atom: usize,
    shift: Shift,
}

/// How [`InternalTopology`] is built and maintained.
///
/// Non-exhaustive: start from [`TopologyConfig::default`] and chain the
/// `with_*` setters.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct TopologyConfig {
    /// `sigma` of the paper: atoms closer than `bond_scale` times the
    /// sum of their covalent radii are bonded.
    pub bond_scale: f64,
    /// Factor by which `sigma` grows each time fragments are welded.
    pub weld_growth: f64,
    /// Most welding rounds before the build gives up.
    pub max_weld_rounds: usize,
    /// A bend within this many radians of 0 or pi is near-linear.
    pub linear_tol: f64,
    /// A centre with three bonded neighbours is near-planar when the
    /// three bend angles among them fall short of a full turn by less
    /// than this many radians. Its umbrella motion is then a fold of the
    /// bond and angle chart, so the build adds the improper dihedral to
    /// every such centre that sits inside no torsion.
    pub planar_tol: f64,
    /// Distance from its vertex of a dummy atom, in the host's length
    /// unit. The paper's 1 angstrom.
    pub dummy_distance: f64,
    /// Singular values of the Wilson matrix under this count as zero.
    pub sv_tol: f64,
    /// Refuse a topology whose rank is short of the rigid quotient.
    pub require_complete: bool,
    /// Most Newton iterations of one back-transformation.
    pub back_transform_iterations: usize,
    /// A back-transformation has converged when its last Cartesian
    /// move is under this length.
    pub back_transform_tol: f64,
}

impl Default for TopologyConfig {
    fn default() -> Self {
        Self {
            bond_scale: 1.25,
            weld_growth: 1.05,
            max_weld_rounds: 512,
            linear_tol: 15.0 * std::f64::consts::PI / 180.0,
            planar_tol: 15.0 * std::f64::consts::PI / 180.0,
            dummy_distance: 1.0,
            sv_tol: PAPER_SV_TOL,
            require_complete: true,
            back_transform_iterations: 20,
            back_transform_tol: PAPER_SV_TOL,
        }
    }
}

with_setters!(TopologyConfig {
    with_bond_scale => bond_scale: val f64,
    with_weld_growth => weld_growth: val f64,
    with_max_weld_rounds => max_weld_rounds: val usize,
    with_linear_tol => linear_tol: val f64,
    with_planar_tol => planar_tol: val f64,
    with_dummy_distance => dummy_distance: val f64,
    with_sv_tol => sv_tol: val f64,
    with_require_complete => require_complete: val bool,
    with_back_transform_iterations => back_transform_iterations: val usize,
    with_back_transform_tol => back_transform_tol: val f64,
});

impl TopologyConfig {
    fn validate(&self) -> Result<(), SaddleError> {
        let positive = |v: f64| v.is_finite() && v > 0.0;
        if !positive(self.bond_scale)
            || !positive(self.dummy_distance)
            || !positive(self.sv_tol)
            || !positive(self.back_transform_tol)
        {
            return Err(SaddleError::Invalid(
                "topology scales and tolerances must be positive and finite".into(),
            ));
        }
        if !self.weld_growth.is_finite() || self.weld_growth <= 1.0 {
            return Err(SaddleError::Invalid(
                "weld_growth must exceed one so welding terminates".into(),
            ));
        }
        if !self.linear_tol.is_finite()
            || self.linear_tol <= 0.0
            || self.linear_tol >= std::f64::consts::FRAC_PI_2
        {
            return Err(SaddleError::Invalid(
                "linear_tol must lie strictly between 0 and pi/2".into(),
            ));
        }
        if !self.planar_tol.is_finite()
            || self.planar_tol <= 0.0
            || self.planar_tol >= std::f64::consts::FRAC_PI_2
        {
            return Err(SaddleError::Invalid(
                "planar_tol must lie strictly between 0 and pi/2".into(),
            ));
        }
        Ok(())
    }
}

/// One dummy atom and the real atoms it hangs on.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct DummyHost {
    /// Index of the dummy in the Wilson frame (`n_real + k`).
    pub frame_index: usize,
    /// The real atom at the vertex of the replaced bend.
    pub centre: usize,
    /// The two real atoms of the first bend the dummy replaced.
    pub partners: [usize; 2],
}

/// A bend within the near-linear window at build time, left out of the
/// chart: a dummy atom or an improper dihedral stands in for it.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct LinearBend {
    /// The real atom at the vertex.
    pub centre: usize,
    /// The two arms, as `(atom, lattice image)`.
    pub arms: [(usize, [i32; 3]); 2],
}

/// A centre with three bonded neighbours that was clear of the
/// near-planar window at build time and so has no improper dihedral. A
/// search that carries it into the window needs one.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct PlanarWatch {
    /// The real atom at the centre.
    pub centre: usize,
    /// The three neighbours, as `(atom, lattice image)`.
    pub arms: [(usize, [i32; 3]); 3],
}

/// Which frame slots are real atoms and which are dummies.
///
/// The Wilson frame lists the `n_real` atoms first, in the caller's
/// order, then the dummies in the order they were made.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AtomMap {
    n_real: usize,
    hosts: Vec<DummyHost>,
}

impl AtomMap {
    /// A frame of `n_real` atoms and no dummies.
    pub fn without_dummies(n_real: usize) -> Self {
        Self {
            n_real,
            hosts: Vec::new(),
        }
    }

    /// Number of real atoms.
    pub fn n_real(&self) -> usize {
        self.n_real
    }

    /// Number of dummy atoms.
    pub fn n_dummy(&self) -> usize {
        self.hosts.len()
    }

    /// Atoms in the Wilson frame, real and dummy.
    pub fn n_frame(&self) -> usize {
        self.n_real + self.hosts.len()
    }

    /// Frame slot of real atom `real`. Real atoms keep their index.
    pub fn frame_of_real(&self, real: usize) -> Option<usize> {
        (real < self.n_real).then_some(real)
    }

    /// Real atom in frame slot `frame`, or `None` for a dummy.
    pub fn real_of_frame(&self, frame: usize) -> Option<usize> {
        (frame < self.n_real).then_some(frame)
    }

    /// Whether frame slot `frame` holds a dummy.
    pub fn is_dummy(&self, frame: usize) -> bool {
        frame >= self.n_real && frame < self.n_frame()
    }

    /// Frame slot of dummy `dummy`.
    pub fn frame_of_dummy(&self, dummy: usize) -> Option<usize> {
        self.hosts.get(dummy).map(|h| h.frame_index)
    }

    /// Dummy in frame slot `frame`, or `None` for a real atom.
    pub fn dummy_of_frame(&self, frame: usize) -> Option<usize> {
        if self.is_dummy(frame) {
            Some(frame - self.n_real)
        } else {
            None
        }
    }

    /// The dummies, in frame order.
    pub fn hosts(&self) -> &[DummyHost] {
        &self.hosts
    }
}

/// Rank and completeness of a chart against the rigid quotient.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct TopologyCheck {
    /// Internals in the chart.
    pub n_int: usize,
    /// Cartesian slots of the frame, `3 (n_real + n_dummy)`.
    pub frame_dof: usize,
    /// Rigid motions of the frame: 6, 5 for a line, 3 when the bond
    /// graph wraps the cell and only translations are free.
    pub rigid_dof: usize,
    /// Singular values of the Wilson matrix above the cutoff.
    pub rank: usize,
    /// `frame_dof - rigid_dof`: the rank a complete chart reaches.
    pub expected_rank: usize,
    /// Internals beyond the rank.
    pub redundancy: usize,
    /// Dimension of the rigid quotient of the real atoms alone,
    /// `3 n_real` less their rigid motions.
    pub real_dof: usize,
    /// Rank of the dummy pins; zero without dummies.
    pub pin_rank: usize,
    /// Rank of the Wilson matrix on the displacements the pins allow.
    pub pinned_rank: usize,
    /// What `pinned_rank` reaches when nothing is missing: the pinned
    /// subspace less the rigid motions.
    pub pinned_expected: usize,
    /// Smallest singular value counted in `rank`. Zero for an empty chart.
    pub smallest_sv: f64,
    /// Largest `|B w|` over a unit rigid-motion vector `w`: how far the
    /// internals are from invariant under rigid motion.
    pub rigid_leak: f64,
    /// The bond graph wraps the cell, so rotations are not free.
    pub periodic_network: bool,
    /// Rank, pinned rank and invariance all meet their targets.
    pub complete: bool,
}

/// Internals of one geometry: the chart, the dummy pins, and the map of
/// real atoms to Wilson frame slots.
#[derive(Clone, Debug)]
pub struct InternalTopology {
    chart: Constraints,
    pins: Constraints,
    map: AtomMap,
    dummy_xyz: Array1<f64>,
    linear: Vec<LinearBend>,
    planar: Vec<PlanarWatch>,
    check: TopologyCheck,
    config: TopologyConfig,
}

impl InternalTopology {
    /// Build the internals of `x` for atoms `elements` (atomic
    /// numbers) with the paper's default rules. `cell` makes the search
    /// periodic: rows of the lattice are the cell vectors of
    /// [`Cell`].
    pub fn from_geometry(
        elements: &[u8],
        x: ArrayView1<f64>,
        cell: Option<&Cell>,
    ) -> Result<Self, SaddleError> {
        Self::from_geometry_with(elements, x, cell, &TopologyConfig::default())
    }

    /// [`Self::from_geometry`] with explicit rules.
    pub fn from_geometry_with(
        elements: &[u8],
        x: ArrayView1<f64>,
        cell: Option<&Cell>,
        config: &TopologyConfig,
    ) -> Result<Self, SaddleError> {
        Self::from_lattice(elements, x, cell.map(lattice_of), config)
    }

    /// [`Self::from_geometry_with`] for a lattice given as one cell
    /// vector per row.
    pub(crate) fn from_lattice(
        elements: &[u8],
        x: ArrayView1<f64>,
        lattice: Option<[[f64; 3]; 3]>,
        config: &TopologyConfig,
    ) -> Result<Self, SaddleError> {
        config.validate()?;
        let mut b = Builder::new(elements, x, lattice, config)?;
        b.find_bonds()?;
        b.find_angles_and_dummies();
        b.add_planar_impropers();
        let topology = b.finish().assemble(lattice, config)?;
        if config.require_complete && !topology.check.complete {
            let k = &topology.check;
            return Err(SaddleError::Solver(format!(
                "automatic internals are incomplete: rank {} of {} (pinned {} of {}), rigid leak {:.3e}",
                k.rank, k.expected_rank, k.pinned_rank, k.pinned_expected, k.rigid_leak
            )));
        }
        Ok(topology)
    }

    /// [`Self::from_geometry`] from element symbols.
    pub fn from_symbols(
        symbols: &[&str],
        x: ArrayView1<f64>,
        cell: Option<&Cell>,
    ) -> Result<Self, SaddleError> {
        let mut z = Vec::with_capacity(symbols.len());
        for s in symbols {
            z.push(
                atomic_number(s)
                    .ok_or_else(|| SaddleError::Invalid(format!("unknown element symbol {s:?}")))?,
            );
        }
        Self::from_geometry(&z, x, cell)
    }

    /// The internals as equalities on the real-plus-dummy frame, in
    /// Sella packing order: bonds, then angles, then dihedrals.
    pub fn constraints(&self) -> &Constraints {
        &self.chart
    }

    /// The pins that hold each dummy in place: its bond length and two
    /// 90 degree angles. Empty when there is no dummy.
    pub fn pins(&self) -> &Constraints {
        &self.pins
    }

    /// Real and dummy slots of the frame.
    pub fn atom_map(&self) -> &AtomMap {
        &self.map
    }

    /// Initial dummy positions, `3 n_dummy` long, in dummy order.
    pub fn dummy_positions(&self) -> ArrayView1<'_, f64> {
        self.dummy_xyz.view()
    }

    /// Bends the build found near-linear and replaced by dummies or
    /// improper dihedrals. A rebuild is due when one leaves the window.
    pub fn linear_bends(&self) -> &[LinearBend] {
        &self.linear
    }

    /// Three-neighbour centres that were clear of the near-planar window
    /// and so carry no improper dihedral.
    pub fn planar_watch(&self) -> &[PlanarWatch] {
        &self.planar
    }

    /// The rank and completeness measured when the topology was built.
    pub fn check(&self) -> &TopologyCheck {
        &self.check
    }

    /// The rules the topology was built with.
    pub fn config(&self) -> &TopologyConfig {
        &self.config
    }

    /// Real atoms.
    pub fn n_real(&self) -> usize {
        self.map.n_real()
    }

    /// Dummy atoms.
    pub fn n_dummy(&self) -> usize {
        self.map.n_dummy()
    }

    /// Internals in the chart.
    pub fn n_int(&self) -> usize {
        self.chart.equalities().len()
    }

    /// The Wilson frame of real positions `x`: `x` followed by the
    /// dummy positions.
    pub fn frame(&self, x: ArrayView1<f64>) -> Result<Array1<f64>, SaddleError> {
        if x.len() != 3 * self.map.n_real() {
            return Err(SaddleError::Shape(
                "positions must be 3 N for the topology's real atoms".into(),
            ));
        }
        Ok(join_frame(x, self.dummy_xyz.view()))
    }

    /// Measure the topology at `x` (with the stored dummy positions).
    pub fn check_at(&self, x: ArrayView1<f64>) -> Result<TopologyCheck, SaddleError> {
        let frame = self.frame(x)?;
        check_chart(
            &self.chart,
            Some(&self.pins),
            frame.view(),
            self.map.n_real(),
            self.config.sv_tol,
        )
    }
}

pub(crate) fn join_frame(x: ArrayView1<f64>, dummies: ArrayView1<f64>) -> Array1<f64> {
    let mut f = Array1::zeros(x.len() + dummies.len());
    for (i, v) in x.iter().enumerate() {
        f[i] = *v;
    }
    for (i, v) in dummies.iter().enumerate() {
        f[x.len() + i] = *v;
    }
    f
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale3(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm3(a: [f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}

fn unit3(a: [f64; 3]) -> [f64; 3] {
    scale3(a, 1.0 / norm3(a))
}

/// Angle between `u` and `v` in `[0, pi]` without an arccosine domain.
fn angle_between(u: [f64; 3], v: [f64; 3]) -> f64 {
    norm3(cross3(u, v)).atan2(dot3(u, v))
}

fn negate(s: Shift) -> Shift {
    [-s[0], -s[1], -s[2]]
}

fn add_shift(a: Shift, b: Shift) -> Shift {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

const ZERO_SHIFT: Shift = [0; 3];

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct BondKey {
    i: usize,
    j: usize,
    shift: Shift,
}

impl BondKey {
    /// `shift` is the image of `j` relative to `i`. Stored with
    /// `i < j`, or `i == j` and a lexicographically positive shift.
    fn canonical(i: usize, j: usize, shift: Shift) -> Self {
        if i < j || (i == j && shift > ZERO_SHIFT) {
            Self { i, j, shift }
        } else {
            Self {
                i: j,
                j: i,
                shift: negate(shift),
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct AngleKey {
    centre: usize,
    a: Nbr,
    c: Nbr,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct DihedralKey {
    atoms: [usize; 4],
    shifts: [Shift; 4],
}

impl DihedralKey {
    /// Translate so the first atom sits at image zero, then take the
    /// smaller of the chain and its reverse.
    fn canonical(atoms: [usize; 4], shifts: [Shift; 4]) -> Self {
        let rel = |from: usize| {
            let mut out = [ZERO_SHIFT; 4];
            for (k, s) in shifts.iter().enumerate() {
                out[k] = add_shift(*s, negate(shifts[from]));
            }
            out
        };
        let fwd = Self {
            atoms,
            shifts: rel(0),
        };
        let mut ra = atoms;
        ra.reverse();
        let mut rs = rel(3);
        rs.reverse();
        let rev = Self {
            atoms: ra,
            shifts: rs,
        };
        if fwd <= rev { fwd } else { rev }
    }
}

struct Builder<'a> {
    cfg: &'a TopologyConfig,
    z: &'a [u8],
    n: usize,
    pos: Vec<[f64; 3]>,
    lattice: Option<[[f64; 3]; 3]>,
    inv_rows: Option<[[f64; 3]; 3]>,
    widths: [f64; 3],
    radii: Vec<f64>,
    bonds: BTreeSet<BondKey>,
    nbrs: Vec<Vec<Nbr>>,
    angles: BTreeSet<AngleKey>,
    dihedrals: BTreeSet<DihedralKey>,
    dummies: Vec<DummyBuild>,
    linear: Vec<LinearBend>,
    planar: Vec<PlanarWatch>,
}

struct DummyBuild {
    centre: usize,
    partners: [Nbr; 2],
    pins: BTreeSet<AngleKey>,
}

struct Primitives {
    n: usize,
    pos: Vec<[f64; 3]>,
    bonds: Vec<BondKey>,
    angles: Vec<AngleKey>,
    dihedrals: Vec<DihedralKey>,
    dummies: Vec<DummyBuild>,
    linear: Vec<LinearBend>,
    planar: Vec<PlanarWatch>,
}

impl<'a> Builder<'a> {
    fn new(
        z: &'a [u8],
        x: ArrayView1<f64>,
        lattice: Option<[[f64; 3]; 3]>,
        cfg: &'a TopologyConfig,
    ) -> Result<Self, SaddleError> {
        let n = z.len();
        if n < 2 {
            return Err(SaddleError::Shape(
                "internal coordinates need at least two atoms".into(),
            ));
        }
        if x.len() != 3 * n {
            return Err(SaddleError::Shape(
                "positions must be 3 N for the element list".into(),
            ));
        }
        if !x.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("topology positions"));
        }
        let mut radii = Vec::with_capacity(n);
        for &zi in z {
            radii.push(covalent_radius(zi).ok_or_else(|| {
                SaddleError::Invalid(format!(
                    "no Cordero covalent radius for atomic number {zi} ({})",
                    element_symbol(zi).map_or("?", |s| s)
                ))
            })?);
        }
        let pos = (0..n)
            .map(|i| [x[3 * i], x[3 * i + 1], x[3 * i + 2]])
            .collect();
        let (inv_rows, widths) = match lattice {
            None => (None, [0.0; 3]),
            Some(l) => {
                let inv = m3_inv(l)?;
                let det = dot3(l[0], cross3(l[1], l[2])).abs();
                let w = [
                    det / norm3(cross3(l[1], l[2])),
                    det / norm3(cross3(l[2], l[0])),
                    det / norm3(cross3(l[0], l[1])),
                ];
                (Some(inv), w)
            }
        };
        Ok(Self {
            cfg,
            z,
            n,
            pos,
            lattice,
            inv_rows,
            widths,
            radii,
            bonds: BTreeSet::new(),
            nbrs: vec![Vec::new(); n],
            angles: BTreeSet::new(),
            dihedrals: BTreeSet::new(),
            dummies: Vec::new(),
            linear: Vec::new(),
            planar: Vec::new(),
        })
    }

    fn shift_vec(&self, s: Shift) -> [f64; 3] {
        match self.lattice {
            None => [0.0; 3],
            Some(l) => {
                let mut v = [0.0; 3];
                for d in 0..3 {
                    for c in 0..3 {
                        v[c] += f64::from(s[d]) * l[d][c];
                    }
                }
                v
            }
        }
    }

    /// Position of a neighbour entry of `owner`'s image.
    fn nbr_pos(&self, e: Nbr) -> [f64; 3] {
        add3(self.pos[e.atom], self.shift_vec(e.shift))
    }

    /// Vector from `centre` to its neighbour `e`.
    fn arm(&self, centre: usize, e: Nbr) -> [f64; 3] {
        sub3(self.nbr_pos(e), self.pos[centre])
    }

    /// Every image shift `t` of atom `j` whose distance to atom `i` is
    /// under `cut`, with that distance.
    fn images_within(&self, i: usize, j: usize, cut: f64) -> Vec<(Shift, f64)> {
        let d0 = sub3(self.pos[j], self.pos[i]);
        let mut out = Vec::new();
        let (Some(inv), Some(l)) = (self.inv_rows, self.lattice) else {
            let d = norm3(d0);
            if d < cut {
                out.push((ZERO_SHIFT, d));
            }
            return out;
        };
        let mut frac = [0.0; 3];
        for k in 0..3 {
            for c in 0..3 {
                frac[k] += inv[c][k] * d0[c];
            }
        }
        let mut lo = [0i32; 3];
        let mut hi = [0i32; 3];
        for k in 0..3 {
            let reach = cut / self.widths[k];
            lo[k] = (-frac[k] - reach).ceil() as i32;
            hi[k] = (-frac[k] + reach).floor() as i32;
        }
        for t0 in lo[0]..=hi[0] {
            for t1 in lo[1]..=hi[1] {
                for t2 in lo[2]..=hi[2] {
                    let t = [t0, t1, t2];
                    let mut v = d0;
                    for d in 0..3 {
                        for c in 0..3 {
                            v[c] += f64::from(t[d]) * l[d][c];
                        }
                    }
                    let dist = norm3(v);
                    if dist < cut {
                        out.push((t, dist));
                    }
                }
            }
        }
        out
    }

    fn components(&self) -> (Vec<usize>, usize) {
        let mut parent: Vec<usize> = (0..self.n).collect();
        fn find(parent: &mut [usize], mut a: usize) -> usize {
            while parent[a] != a {
                parent[a] = parent[parent[a]];
                a = parent[a];
            }
            a
        }
        for b in &self.bonds {
            let ra = find(&mut parent, b.i);
            let rb = find(&mut parent, b.j);
            if ra != rb {
                parent[ra] = rb;
            }
        }
        let mut label = vec![usize::MAX; self.n];
        let mut next = 0;
        let mut out = vec![0; self.n];
        for (a, slot) in out.iter_mut().enumerate() {
            let r = find(&mut parent, a);
            if label[r] == usize::MAX {
                label[r] = next;
                next += 1;
            }
            *slot = label[r];
        }
        (out, next)
    }

    /// Bonds by covalent radius, then welding of the fragments.
    fn find_bonds(&mut self) -> Result<(), SaddleError> {
        let coincident = f64::EPSILON.sqrt();
        let mut sigma = self.cfg.bond_scale;
        let mut labels = vec![0usize; self.n];
        let mut first = true;
        let mut rounds = 0usize;
        loop {
            let mut found = Vec::new();
            for i in 0..self.n {
                for j in i..self.n {
                    if !first && labels[i] == labels[j] {
                        continue;
                    }
                    let rsum = self.radii[i] + self.radii[j];
                    let cut = sigma * rsum;
                    for (t, d) in self.images_within(i, j, cut) {
                        if i == j && t == ZERO_SHIFT {
                            continue;
                        }
                        if d <= coincident * rsum {
                            return Err(SaddleError::Shape(format!(
                                "atoms {i} and {j} sit on top of each other"
                            )));
                        }
                        found.push(BondKey::canonical(i, j, t));
                    }
                }
            }
            for b in found {
                self.bonds.insert(b);
            }
            first = false;
            let (lab, count) = self.components();
            labels = lab;
            if count == 1 {
                break;
            }
            rounds += 1;
            if rounds > self.cfg.max_weld_rounds {
                return Err(SaddleError::Solver(
                    "fragments did not weld into one molecular graph".into(),
                ));
            }
            sigma *= self.cfg.weld_growth;
        }
        for b in &self.bonds {
            self.nbrs[b.i].push(Nbr {
                atom: b.j,
                shift: b.shift,
            });
            self.nbrs[b.j].push(Nbr {
                atom: b.i,
                shift: negate(b.shift),
            });
        }
        for list in &mut self.nbrs {
            list.sort();
        }
        Ok(())
    }

    fn in_window(&self, theta: f64) -> bool {
        is_bent(theta, self.cfg.linear_tol)
    }

    fn bend(&self, centre: usize, a: Nbr, c: Nbr) -> f64 {
        angle_between(self.arm(centre, a), self.arm(centre, c))
    }

    fn push_angle(&mut self, centre: usize, p: Nbr, q: Nbr) {
        let (a, c) = if p <= q { (p, q) } else { (q, p) };
        self.angles.insert(AngleKey { centre, a, c });
    }

    fn find_angles_and_dummies(&mut self) {
        for j in 0..self.n {
            let list = self.nbrs[j].clone();
            let mut linear: Vec<(Nbr, Nbr)> = Vec::new();
            for p in 0..list.len() {
                for q in (p + 1)..list.len() {
                    let theta = self.bend(j, list[p], list[q]);
                    if self.in_window(theta) {
                        self.push_angle(j, list[p], list[q]);
                    } else {
                        linear.push((list[p], list[q]));
                    }
                }
            }
            if linear.is_empty() {
                continue;
            }
            let mut dummy: Option<usize> = None;
            for (a, c) in linear {
                self.linear.push(LinearBend {
                    centre: j,
                    arms: [(a.atom, a.shift), (c.atom, c.shift)],
                });
                if list.len() >= 3
                    && let Some(d) = self.third_bond(j, a, c, &list)
                {
                    self.add_dihedral(
                        [a.atom, j, d.atom, c.atom],
                        [a.shift, ZERO_SHIFT, d.shift, c.shift],
                    );
                    continue;
                }
                let k = match dummy {
                    Some(k) => k,
                    None => {
                        let k = self.make_dummy(j, a, c, &list);
                        dummy = Some(k);
                        k
                    }
                };
                self.pin_dummy(k, j, a, c);
            }
        }
        self.add_proper_dihedrals();
    }

    /// The improper `a-centre-d-c` of a triple of bonds: the nearest of
    /// the three is `d`. `None` when a bend among them is near-linear.
    fn improper_of(&self, j: usize, mut triple: [Nbr; 3]) -> Option<([usize; 4], [Shift; 4])> {
        triple.sort_by(|u, v| {
            norm3(self.arm(j, *u))
                .total_cmp(&norm3(self.arm(j, *v)))
                .then(self.z[u.atom].cmp(&self.z[v.atom]))
                .then(u.cmp(v))
        });
        let d = triple[0];
        let (a, c) = if triple[1] <= triple[2] {
            (triple[1], triple[2])
        } else {
            (triple[2], triple[1])
        };
        (self.in_window(self.bend(j, a, d)) && self.in_window(self.bend(j, d, c))).then_some((
            [a.atom, j, d.atom, c.atom],
            [a.shift, ZERO_SHIFT, d.shift, c.shift],
        ))
    }

    fn real_neighbours(&self, j: usize) -> Vec<Nbr> {
        self.nbrs[j]
            .iter()
            .copied()
            .filter(|e| e.atom < self.n)
            .collect()
    }

    /// An improper dihedral for every triple of bonds whose three bends
    /// come within `planar_tol` of a full turn; the triples clear of the
    /// window are noted so a search can tell when one enters it.
    ///
    /// A centre that already sits inside a torsion has its out-of-plane
    /// motion in that torsion, as in ethylene, and is left alone.
    fn add_planar_impropers(&mut self) {
        let full_turn = 2.0 * std::f64::consts::PI;
        let mut inside_torsion = vec![false; self.n];
        for d in &self.dihedrals {
            for k in [1, 2] {
                if d.atoms[k] < self.n {
                    inside_torsion[d.atoms[k]] = true;
                }
            }
        }
        for (j, covered) in inside_torsion.into_iter().enumerate() {
            if covered {
                continue;
            }
            let list = self.real_neighbours(j);
            for p in 0..list.len() {
                for q in (p + 1)..list.len() {
                    for r in (q + 1)..list.len() {
                        let triple = [list[p], list[q], list[r]];
                        let sum = self.bend(j, triple[0], triple[1])
                            + self.bend(j, triple[1], triple[2])
                            + self.bend(j, triple[0], triple[2]);
                        if full_turn - sum < self.cfg.planar_tol {
                            if let Some((atoms, shifts)) = self.improper_of(j, triple) {
                                self.add_dihedral(atoms, shifts);
                            }
                        } else {
                            self.planar.push(PlanarWatch {
                                centre: j,
                                arms: triple.map(|e| (e.atom, e.shift)),
                            });
                        }
                    }
                }
            }
        }
    }

    /// The bonded atom closest to `centre` other than `a` and `c` that
    /// makes `a-centre-d-c` a well-defined improper dihedral.
    fn third_bond(&self, centre: usize, a: Nbr, c: Nbr, list: &[Nbr]) -> Option<Nbr> {
        let mut cand: Vec<(f64, u8, Nbr)> = list
            .iter()
            .filter(|e| **e != a && **e != c)
            .map(|e| (norm3(self.arm(centre, *e)), self.z[e.atom], *e))
            .collect();
        cand.sort_by(|u, v| u.0.total_cmp(&v.0).then(u.1.cmp(&v.1)).then(u.2.cmp(&v.2)));
        cand.into_iter().map(|(_, _, e)| e).find(|e| {
            self.in_window(self.bend(centre, a, *e)) && self.in_window(self.bend(centre, *e, c))
        })
    }

    fn add_dihedral(&mut self, atoms: [usize; 4], shifts: [Shift; 4]) {
        self.dihedrals.insert(DihedralKey::canonical(atoms, shifts));
    }

    /// Order the pair by bond length (equal lengths by atomic number,
    /// then index), place the dummy, and register it.
    fn make_dummy(&mut self, centre: usize, a: Nbr, c: Nbr, list: &[Nbr]) -> usize {
        let (a, c) = self.order_pair(centre, a, c);
        let va = self.arm(centre, a);
        let vc = self.arm(centre, c);
        let dir = dummy_direction(va, vc);
        let place = add3(self.pos[centre], scale3(dir, self.cfg.dummy_distance));
        let frame = self.pos.len();
        self.pos.push(place);
        self.nbrs.push(vec![Nbr {
            atom: centre,
            shift: ZERO_SHIFT,
        }]);
        let x = Nbr {
            atom: frame,
            shift: ZERO_SHIFT,
        };
        self.bonds
            .insert(BondKey::canonical(centre, frame, ZERO_SHIFT));
        self.nbrs[centre].push(x);
        self.nbrs[centre].sort();
        for e in list {
            let theta = self.bend(centre, *e, x);
            if self.in_window(theta) {
                self.push_angle(centre, *e, x);
            }
        }
        self.dummies.push(DummyBuild {
            centre,
            partners: [a, c],
            pins: BTreeSet::new(),
        });
        self.dummies.len() - 1
    }

    fn order_pair(&self, centre: usize, a: Nbr, c: Nbr) -> (Nbr, Nbr) {
        let la = norm3(self.arm(centre, a));
        let lc = norm3(self.arm(centre, c));
        let tie = (la - lc).abs() <= f64::EPSILON.sqrt() * la.max(lc);
        let first = if tie {
            (self.z[a.atom], a) <= (self.z[c.atom], c)
        } else {
            la < lc
        };
        if first { (a, c) } else { (c, a) }
    }

    /// Record the pins and the improper dihedral of one replaced bend.
    fn pin_dummy(&mut self, k: usize, centre: usize, a: Nbr, c: Nbr) {
        let frame = self.n + k;
        let x = Nbr {
            atom: frame,
            shift: ZERO_SHIFT,
        };
        for e in [a, c] {
            let (p, q) = if e <= x { (e, x) } else { (x, e) };
            self.dummies[k].pins.insert(AngleKey { centre, a: p, c: q });
        }
        let (first, second) = if self.dummy_pair_order(k, a, c) {
            (a, c)
        } else {
            (c, a)
        };
        self.add_dihedral(
            [first.atom, centre, frame, second.atom],
            [first.shift, ZERO_SHIFT, ZERO_SHIFT, second.shift],
        );
    }

    /// Whether `(a, c)` is the order the dummy was placed with.
    fn dummy_pair_order(&self, k: usize, a: Nbr, c: Nbr) -> bool {
        let centre = self.dummies[k].centre;
        let (fa, _) = self.order_pair(centre, a, c);
        fa == a
    }

    fn add_proper_dihedrals(&mut self) {
        let bonds: Vec<BondKey> = self.bonds.iter().copied().collect();
        for b in bonds {
            let (j, k, t) = (b.i, b.j, b.shift);
            let to_k = Nbr { atom: k, shift: t };
            let to_j = Nbr {
                atom: j,
                shift: negate(t),
            };
            let at_j: Vec<Nbr> = self.nbrs[j]
                .iter()
                .copied()
                .filter(|e| *e != to_k)
                .collect();
            let at_k: Vec<Nbr> = self.nbrs[k]
                .iter()
                .copied()
                .filter(|e| *e != to_j)
                .collect();
            for p in &at_j {
                if !self.has_angle(j, *p, to_k) {
                    continue;
                }
                for q in &at_k {
                    if !self.has_angle(k, to_j, *q) {
                        continue;
                    }
                    let q_shift = add_shift(t, q.shift);
                    if p.atom == q.atom && p.shift == q_shift {
                        continue;
                    }
                    self.add_dihedral([p.atom, j, k, q.atom], [p.shift, ZERO_SHIFT, t, q_shift]);
                }
            }
        }
    }

    fn has_angle(&self, centre: usize, p: Nbr, q: Nbr) -> bool {
        let (a, c) = if p <= q { (p, q) } else { (q, p) };
        self.angles.contains(&AngleKey { centre, a, c })
    }

    fn finish(self) -> Primitives {
        Primitives {
            n: self.n,
            pos: self.pos,
            bonds: self.bonds.into_iter().collect(),
            angles: self.angles.into_iter().collect(),
            dihedrals: self.dihedrals.into_iter().collect(),
            dummies: self.dummies,
            linear: self.linear,
            planar: self.planar,
        }
    }
}

/// Whether a bend of `theta` radians is clear of the near-linear window
/// of half-width `tol` around 0 and pi.
pub(crate) fn is_bent(theta: f64, tol: f64) -> bool {
    theta > tol && theta < std::f64::consts::PI - tol
}

/// The bend at `centre` between arms `a` and `c`, each `(atom, lattice
/// image)`, measured in the frame `x` with lattice rows `lattice`.
pub(crate) fn arm_angle(
    x: ArrayView1<f64>,
    lattice: Option<[[f64; 3]; 3]>,
    centre: usize,
    a: (usize, [i32; 3]),
    c: (usize, [i32; 3]),
) -> f64 {
    let at = |e: (usize, [i32; 3])| {
        let mut p = [x[3 * e.0], x[3 * e.0 + 1], x[3 * e.0 + 2]];
        if let Some(l) = lattice {
            for d in 0..3 {
                for k in 0..3 {
                    p[k] += f64::from(e.1[d]) * l[d][k];
                }
            }
        }
        sub3(p, [x[3 * centre], x[3 * centre + 1], x[3 * centre + 2]])
    };
    angle_between(at(a), at(c))
}

/// The direction of a dummy atom for arms `va` and `vc`: along
/// `ba x bc`, or the Cartesian-axis rule once that cross product is
/// numerically unstable.
fn dummy_direction(va: [f64; 3], vc: [f64; 3]) -> [f64; 3] {
    let cr = cross3(unit3(va), unit3(vc));
    let len = norm3(cr);
    if len >= PAPER_CROSS_TOL {
        return scale3(cr, 1.0 / len);
    }
    let ac = unit3(sub3(vc, va));
    let mut axis = 0;
    for k in 1..3 {
        if ac[k].abs() < ac[axis].abs() {
            axis = k;
        }
    }
    let mut w = [0.0; 3];
    w[axis] = 1.0;
    unit3(sub3(w, scale3(ac, dot3(ac, w))))
}

impl Primitives {
    fn assemble(
        self,
        lattice: Option<[[f64; 3]; 3]>,
        cfg: &TopologyConfig,
    ) -> Result<InternalTopology, SaddleError> {
        let n_real = self.n;
        let n_frame = self.pos.len();
        let mut frame = Array1::zeros(3 * n_frame);
        for (i, p) in self.pos.iter().enumerate() {
            for c in 0..3 {
                frame[3 * i + c] = p[c];
            }
        }
        let mut chart = Constraints::new(n_frame)?;
        let mut pins = Constraints::new(n_frame)?;
        if let Some(l) = lattice {
            chart.set_lattice(l);
            pins.set_lattice(l);
        }
        for b in &self.bonds {
            chart.fix_bond_image([b.i, b.j], [ZERO_SHIFT, b.shift], frame.view(), None)?;
        }
        for a in &self.angles {
            chart.fix_angle_image(
                [a.a.atom, a.centre, a.c.atom],
                [a.a.shift, ZERO_SHIFT, a.c.shift],
                frame.view(),
                None,
            )?;
        }
        for d in &self.dihedrals {
            chart.fix_dihedral_image(d.atoms, d.shifts, frame.view(), None)?;
        }
        let mut hosts = Vec::with_capacity(self.dummies.len());
        for (k, d) in self.dummies.iter().enumerate() {
            let x = n_real + k;
            pins.fix_bond([d.centre, x], frame.view(), Some(cfg.dummy_distance))?;
            for a in &d.pins {
                pins.fix_angle_image(
                    [a.a.atom, a.centre, a.c.atom],
                    [a.a.shift, ZERO_SHIFT, a.c.shift],
                    frame.view(),
                    Some(std::f64::consts::FRAC_PI_2),
                )?;
            }
            hosts.push(DummyHost {
                frame_index: x,
                centre: d.centre,
                partners: [d.partners[0].atom, d.partners[1].atom],
            });
        }
        let mut dummy_xyz = Array1::zeros(3 * hosts.len());
        for k in 0..hosts.len() {
            for c in 0..3 {
                dummy_xyz[3 * k + c] = frame[3 * (n_real + k) + c];
            }
        }
        let check = check_chart(&chart, Some(&pins), frame.view(), n_real, cfg.sv_tol)?;
        Ok(InternalTopology {
            chart,
            pins,
            map: AtomMap { n_real, hosts },
            dummy_xyz,
            linear: self.linear,
            planar: self.planar,
            check,
            config: cfg.clone(),
        })
    }
}

/// Truncated singular value decomposition of a Wilson matrix `B`
/// (`m x p`), through the eigenpairs of `B^T B`.
///
/// Singular values at or under `tol` are dropped, as the paper drops
/// them, so the pseudo-inverse never divides by a vanishing one.
pub(crate) struct Wilson {
    b: Array2<f64>,
    /// Right singular vectors kept, `p x r`.
    v: Array2<f64>,
    /// Singular values kept, length `r`, ascending.
    sigma: Array1<f64>,
}

impl Wilson {
    pub(crate) fn new(b: &Array2<f64>, tol: f64) -> Result<Self, SaddleError> {
        let p = b.ncols();
        if p == 0 || b.nrows() == 0 {
            return Ok(Self {
                b: b.clone(),
                v: Array2::zeros((p, 0)),
                sigma: Array1::zeros(0),
            });
        }
        let g = b.t().dot(b);
        let (lam, vec) = exact_eigh(g.view())?;
        let all = lam.mapv(|l| if l > 0.0 { l.sqrt() } else { 0.0 });
        let keep: Vec<usize> = (0..p).filter(|&k| all[k] > tol).collect();
        let mut v = Array2::zeros((p, keep.len()));
        let mut sigma = Array1::zeros(keep.len());
        for (c, &k) in keep.iter().enumerate() {
            sigma[c] = all[k];
            for r in 0..p {
                v[(r, c)] = vec[(r, k)];
            }
        }
        Ok(Self {
            b: b.clone(),
            v,
            sigma,
        })
    }

    pub(crate) fn rank(&self) -> usize {
        self.sigma.len()
    }

    pub(crate) fn smallest_kept(&self) -> f64 {
        self.sigma.iter().copied().fold(f64::INFINITY, f64::min)
    }

    /// `B^+ dq`: the min-norm Cartesian move that realizes the part of
    /// `dq` in the range of `B`.
    pub(crate) fn solve(&self, dq: &Array1<f64>) -> Array1<f64> {
        let w = self.v.t().dot(&self.b.t().dot(dq));
        let scaled = Array1::from_shape_fn(w.len(), |k| w[k] / (self.sigma[k] * self.sigma[k]));
        self.v.dot(&scaled)
    }

    /// `(B^T)^+ g`: the internals gradient of a Cartesian gradient.
    pub(crate) fn gradient(&self, g: &Array1<f64>) -> Array1<f64> {
        let w = self.v.t().dot(g);
        let scaled = Array1::from_shape_fn(w.len(), |k| w[k] / (self.sigma[k] * self.sigma[k]));
        self.b.dot(&self.v.dot(&scaled))
    }

    /// The truncated pseudo-inverse `B^+` itself, `p x m`.
    pub(crate) fn pinv(&self) -> Array2<f64> {
        let r = self.rank();
        if r == 0 {
            return Array2::zeros((self.v.nrows(), self.b.nrows()));
        }
        let scaled = Array2::from_shape_fn(self.v.dim(), |(i, k)| {
            self.v[(i, k)] / (self.sigma[k] * self.sigma[k])
        });
        scaled.dot(&self.v.t()).dot(&self.b.t())
    }

    /// Orthonormal basis `U` of the range of `B`, `m x r`.
    pub(crate) fn range_basis(&self) -> Array2<f64> {
        let r = self.rank();
        let bv = self.b.dot(&self.v);
        Array2::from_shape_fn((self.b.nrows(), r), |(i, k)| bv[(i, k)] / self.sigma[k])
    }

    /// `V Sigma^-1` as a `p x r` map from the range basis to Cartesian
    /// space: `B^+ = (V Sigma^-1) U^T`.
    pub(crate) fn pinv_cols(&self) -> Array2<f64> {
        let r = self.rank();
        Array2::from_shape_fn((self.v.nrows(), r), |(i, k)| self.v[(i, k)] / self.sigma[k])
    }
}

/// Measure `chart` at the Wilson frame `frame` against the rigid
/// quotient. `n_real` is the number of real atoms; `pins` are the dummy
/// pins, if any.
pub fn check_chart(
    chart: &Constraints,
    pins: Option<&Constraints>,
    frame: ArrayView1<f64>,
    n_real: usize,
    sv_tol: f64,
) -> Result<TopologyCheck, SaddleError> {
    let p = frame.len();
    let n_int = chart.equalities().len();
    let b = chart.jacobian(frame)?;
    let wilson = Wilson::new(&b, sv_tol)?;
    let rank = wilson.rank();

    let (unwrapped, periodic_network) = unwrap_frame(chart, frame)?;
    let rigid = rigid_vertical(&unwrapped, periodic_network);
    let rigid_dof = rigid.1;
    let mut leak = 0.0_f64;
    for k in 0..p {
        let w = rigid.0.column(k).to_owned();
        let bw = b.dot(&w);
        leak = leak.max(bw.dot(&bw).sqrt());
    }
    let expected_rank = p.saturating_sub(rigid_dof);

    let (pin_rank, pinned_rank, pinned_expected) = match pins {
        Some(pc) if !pc.equalities().is_empty() => {
            let jp = pc.jacobian(frame)?;
            let gp = jp.t().dot(&jp);
            let (lam, vec) = exact_eigh(gp.view())?;
            let tol2 = sv_tol * sv_tol;
            let free: Vec<usize> = (0..p).filter(|&k| lam[k] <= tol2).collect();
            let pin_rank = p - free.len();
            let mut z = Array2::zeros((p, free.len()));
            for (c, &k) in free.iter().enumerate() {
                for r in 0..p {
                    z[(r, c)] = vec[(r, k)];
                }
            }
            let bz = b.dot(&z);
            let wz = Wilson::new(&bz, sv_tol)?;
            (pin_rank, wz.rank(), free.len().saturating_sub(rigid_dof))
        }
        _ => (0, rank, expected_rank),
    };

    let real_rigid = rigid_vertical(
        &unwrapped.slice(ndarray::s![..3 * n_real]).to_owned(),
        periodic_network,
    )
    .1;
    let real_dof = (3 * n_real).saturating_sub(real_rigid);
    let complete = rank == expected_rank && pinned_rank == pinned_expected && leak <= sv_tol;
    Ok(TopologyCheck {
        n_int,
        frame_dof: p,
        rigid_dof,
        rank,
        expected_rank,
        redundancy: n_int.saturating_sub(rank),
        real_dof,
        pin_rank,
        pinned_rank,
        pinned_expected,
        smallest_sv: if rank == 0 {
            0.0
        } else {
            wilson.smallest_kept()
        },
        rigid_leak: leak,
        periodic_network,
        complete,
    })
}

/// Positions with each atom moved to the image the bond graph reaches
/// it in, and whether the graph closes a loop that crosses the cell.
fn unwrap_frame(
    chart: &Constraints,
    frame: ArrayView1<f64>,
) -> Result<(Array1<f64>, bool), SaddleError> {
    let n = frame.len() / 3;
    let mut adj: Vec<Vec<(usize, Shift)>> = vec![Vec::new(); n];
    for (idx, eq) in chart.equalities().iter().enumerate() {
        if let Equality::Bond { atoms, .. } = eq {
            let (iu, iv) = match chart.images(idx) {
                Some(im) => (im[0], im[1]),
                None => (ZERO_SHIFT, ZERO_SHIFT),
            };
            let rel = add_shift(iv, negate(iu));
            adj[atoms[0]].push((atoms[1], rel));
            adj[atoms[1]].push((atoms[0], negate(rel)));
        }
    }
    let mut cum: Vec<Option<Shift>> = vec![None; n];
    let mut wraps = false;
    for root in 0..n {
        if cum[root].is_some() {
            continue;
        }
        cum[root] = Some(ZERO_SHIFT);
        let mut queue = VecDeque::from([root]);
        while let Some(u) = queue.pop_front() {
            let Some(cu) = cum[u] else { continue };
            for &(v, rel) in &adj[u] {
                let want = add_shift(cu, rel);
                match cum[v] {
                    None => {
                        cum[v] = Some(want);
                        queue.push_back(v);
                    }
                    Some(have) if have != want => wraps = true,
                    Some(_) => {}
                }
            }
        }
    }
    let lattice = chart.lattice();
    let mut out = frame.to_owned();
    if let Some(l) = lattice {
        for a in 0..n {
            let Some(c) = cum[a] else { continue };
            for d in 0..3 {
                for k in 0..3 {
                    out[3 * a + k] += f64::from(c[d]) * l[d][k];
                }
            }
        }
    }
    Ok((out, wraps))
}

/// The projector onto the rigid motions of `x`, as a `p x p` matrix
/// `I - P`, and its trace. `P` is the horizontal projector of the rigid
/// quotient ([`rgmin::ManifoldKind::RigidQuotient`]), or the removal of
/// the mean per axis when only translations are free.
fn rigid_vertical(x: &Array1<f64>, translations_only: bool) -> (Array2<f64>, usize) {
    let p = x.len();
    let n = p / 3;
    let mut vert = Array2::zeros((p, p));
    for k in 0..p {
        let mut e = Array1::zeros(p);
        e[k] = 1.0;
        let horizontal = if translations_only {
            let mut h = e.clone();
            for c in 0..3 {
                let mean = (0..n).map(|a| e[3 * a + c]).sum::<f64>() / n as f64;
                for a in 0..n {
                    h[3 * a + c] -= mean;
                }
            }
            h
        } else {
            ManifoldKind::RigidQuotient.project(x, &e)
        };
        for r in 0..p {
            vert[(r, k)] = e[r] - horizontal[r];
        }
    }
    let trace: f64 = (0..p).map(|k| vert[(k, k)]).sum();
    (vert, trace.round() as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::internal::pack_cart;

    fn water() -> (Vec<u8>, Array1<f64>) {
        (
            vec![8, 1, 1],
            pack_cart(&[[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2399, 0.9266, 0.0]]),
        )
    }

    #[test]
    fn water_has_two_bonds_one_angle_and_is_complete() {
        let (z, x) = water();
        let t = InternalTopology::from_geometry(&z, x.view(), None).unwrap();
        let c = t.constraints().counts();
        assert_eq!((c.nbonds, c.nangles, c.ndihedrals), (2, 1, 0));
        assert_eq!(t.n_dummy(), 0);
        assert!(t.check().complete);
        assert_eq!(t.check().rank, 3);
    }

    fn ethane() -> (Vec<u8>, Array1<f64>) {
        let rcc = 1.54;
        let rch = 1.09;
        let cos_t = -1.0 / 3.0;
        let sin_t = (8.0_f64).sqrt() / 3.0;
        let mut rows = vec![[0.0, 0.0, -rcc / 2.0], [0.0, 0.0, rcc / 2.0]];
        let mut z = vec![6u8, 6];
        for (side, zc, flip, phase) in [(0, -rcc / 2.0, 1.0, 0.0), (1, rcc / 2.0, -1.0, 60.0)] {
            let _ = side;
            for k in 0..3 {
                let phi = (phase + 120.0 * f64::from(k)).to_radians();
                rows.push([
                    rch * sin_t * phi.cos(),
                    rch * sin_t * phi.sin(),
                    zc + flip * rch * cos_t,
                ]);
                z.push(1);
            }
        }
        (z, pack_cart(&rows))
    }

    fn co2() -> (Vec<u8>, Array1<f64>) {
        (
            vec![8, 6, 8],
            pack_cart(&[[-1.16, 0.0, 0.0], [0.0, 0.0, 0.0], [1.16, 0.0, 0.0]]),
        )
    }

    fn hcn() -> (Vec<u8>, Array1<f64>) {
        (
            vec![1, 6, 7],
            pack_cart(&[[-1.064, 0.0, 0.0], [0.0, 0.0, 0.0], [1.156, 0.0, 0.0]]),
        )
    }

    fn acetylene() -> (Vec<u8>, Array1<f64>) {
        (
            vec![1, 6, 6, 1],
            pack_cart(&[
                [-1.66, 0.0, 0.0],
                [-0.6, 0.0, 0.0],
                [0.6, 0.0, 0.0],
                [1.66, 0.0, 0.0],
            ]),
        )
    }

    #[test]
    fn ethane_has_the_textbook_primitive_counts() {
        let (z, x) = ethane();
        let t = InternalTopology::from_geometry(&z, x.view(), None).unwrap();
        let c = t.constraints().counts();
        assert_eq!((c.nbonds, c.nangles, c.ndihedrals), (7, 12, 9));
        assert_eq!(t.n_dummy(), 0);
        let k = t.check();
        assert!(k.complete);
        assert_eq!(k.rank, 18);
        assert_eq!(k.expected_rank, 18);
        assert_eq!(k.redundancy, 28 - 18);
        assert!(!k.periodic_network);
    }

    #[test]
    fn co2_gets_one_perpendicular_dummy_with_pins() {
        let (z, x) = co2();
        let t = InternalTopology::from_geometry(&z, x.view(), None).unwrap();
        assert_eq!(t.n_dummy(), 1);
        let h = &t.atom_map().hosts()[0];
        assert_eq!((h.centre, h.frame_index), (1, 3));
        let d = t.dummy_positions();
        let carbon = [0.0, 0.0, 0.0];
        let v = [d[0] - carbon[0], d[1] - carbon[1], d[2] - carbon[2]];
        assert!((norm3(v) - 1.0).abs() < 1.0e-12);
        assert!(v[0].abs() < 1.0e-12);
        let c = t.constraints().counts();
        assert_eq!((c.nbonds, c.nangles, c.ndihedrals), (3, 2, 1));
        assert_eq!(t.pins().equalities().len(), 3);
        let k = t.check();
        assert!(k.complete, "{k:?}");
        assert_eq!(k.real_dof, 4);
        assert_eq!(k.pin_rank, 3);
        // Both 90 degree pins stay first-order independent in the frame,
        // which holds the dummy's azimuth but also blocks the bend that
        // tilts the arms toward it. A rebuild takes that move.
        assert_eq!(k.pinned_rank, 3);
    }

    #[test]
    fn hcn_is_linear_with_one_dummy_and_a_full_rank_chart() {
        let (z, x) = hcn();
        let t = InternalTopology::from_geometry(&z, x.view(), None).unwrap();
        assert_eq!(t.n_dummy(), 1);
        assert_eq!(t.atom_map().hosts()[0].centre, 1);
        let k = t.check();
        assert!(k.complete, "{k:?}");
        assert_eq!(k.rank, k.expected_rank);
    }

    #[test]
    fn acetylene_has_a_dummy_on_each_carbon() {
        let (z, x) = acetylene();
        let t = InternalTopology::from_geometry(&z, x.view(), None).unwrap();
        assert_eq!(t.n_dummy(), 2);
        let centres: Vec<usize> = t.atom_map().hosts().iter().map(|h| h.centre).collect();
        assert_eq!(centres, vec![1, 2]);
        assert!(t.check().complete);
        assert_eq!(t.check().pinned_rank, 6);
    }

    #[test]
    fn dummy_placement_is_deterministic_and_swap_stable() {
        let (z, x) = co2();
        let a = InternalTopology::from_geometry(&z, x.view(), None).unwrap();
        let b = InternalTopology::from_geometry(&z, x.view(), None).unwrap();
        assert_eq!(a.dummy_positions(), b.dummy_positions());
        // Exactly equal bonds order by atomic number, then index.
        let z2 = [8u8, 6, 8];
        let c = InternalTopology::from_geometry(&z2, x.view(), None).unwrap();
        assert_eq!(c.atom_map().hosts()[0].partners, [0, 2]);
    }

    #[test]
    fn a_bend_just_outside_the_window_keeps_its_angle_and_makes_no_dummy() {
        let theta = (180.0_f64 - 16.0).to_radians();
        let x = pack_cart(&[
            [-1.16, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [-1.16 * theta.cos(), 1.16 * theta.sin(), 0.0],
        ]);
        let t = InternalTopology::from_geometry(&[8, 6, 8], x.view(), None).unwrap();
        assert_eq!(t.n_dummy(), 0);
        assert_eq!(t.constraints().counts().nangles, 1);
        let inside = (180.0_f64 - 14.0).to_radians();
        let x = pack_cart(&[
            [-1.16, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [-1.16 * inside.cos(), 1.16 * inside.sin(), 0.0],
        ]);
        let t = InternalTopology::from_geometry(&[8, 6, 8], x.view(), None).unwrap();
        assert_eq!(t.n_dummy(), 1);
    }

    #[test]
    fn a_t_shaped_vertex_takes_the_improper_dihedral_not_a_dummy() {
        let x = pack_cart(&[
            [0.0, 0.0, 0.0],
            [1.1, 0.0, 0.0],
            [-1.1, 0.0, 0.0],
            [0.0, 1.1, 0.0],
        ]);
        let t = InternalTopology::from_geometry(&[6, 1, 1, 1], x.view(), None).unwrap();
        assert_eq!(t.n_dummy(), 0);
        let c = t.constraints().counts();
        assert_eq!((c.nbonds, c.nangles, c.ndihedrals), (3, 2, 1));
        assert!(t.check().complete);
    }

    #[test]
    fn ammonia_and_methane_are_complete_without_impropers() {
        let nh3 = pack_cart(&[
            [0.0, 0.0, 0.1],
            [0.94, 0.0, -0.3],
            [-0.47, 0.814, -0.3],
            [-0.47, -0.814, -0.3],
        ]);
        let t = InternalTopology::from_geometry(&[7, 1, 1, 1], nh3.view(), None).unwrap();
        let c = t.constraints().counts();
        assert_eq!((c.nbonds, c.nangles, c.ndihedrals), (3, 3, 0));
        assert!(t.check().complete);
        let a = 0.629;
        let ch4 = pack_cart(&[[0.0; 3], [a, a, a], [a, -a, -a], [-a, a, -a], [-a, -a, a]]);
        let t = InternalTopology::from_geometry(&[6, 1, 1, 1, 1], ch4.view(), None).unwrap();
        assert_eq!(t.check().rank, 9);
        assert!(t.check().complete);
    }

    fn planar_ammonia() -> (Vec<u8>, Array1<f64>) {
        let r = 1.0;
        let x = pack_cart(&[
            [0.0; 3],
            [r, 0.0, 0.0],
            [-0.5 * r, 0.866_025_403_784_438_6 * r, 0.0],
            [-0.5 * r, -0.866_025_403_784_438_6 * r, 0.0],
        ]);
        (vec![7u8, 1, 1, 1], x)
    }

    #[test]
    fn a_planar_centre_gets_an_improper_dihedral_from_the_window() {
        let (z, x) = planar_ammonia();
        let t = InternalTopology::from_geometry(&z, x.view(), None).unwrap();
        assert_eq!(t.constraints().counts().ndihedrals, 1);
        assert!(t.planar_watch().is_empty());
        assert!(t.check().complete);
        assert_eq!(t.check().rank, 6);
    }

    #[test]
    fn a_pyramidal_centre_is_watched_until_it_flattens() {
        let nh3 = pack_cart(&[
            [0.0, 0.0, 0.38],
            [0.94, 0.0, 0.0],
            [-0.47, 0.814, 0.0],
            [-0.47, -0.814, 0.0],
        ]);
        let t = InternalTopology::from_geometry(&[7, 1, 1, 1], nh3.view(), None).unwrap();
        assert_eq!(t.planar_watch().len(), 1);
        assert_eq!(t.constraints().counts().ndihedrals, 0);
        let wide = TopologyConfig::default().with_planar_tol(1.0);
        let t =
            InternalTopology::from_geometry_with(&[7, 1, 1, 1], nh3.view(), None, &wide).unwrap();
        assert!(t.planar_watch().is_empty());
        assert_eq!(t.constraints().counts().ndihedrals, 1);
    }

    #[test]
    fn a_tetrahedral_centre_is_clear_of_the_planar_window() {
        let a = 0.629;
        let ch4 = pack_cart(&[[0.0; 3], [a, a, a], [a, -a, -a], [-a, a, -a], [-a, -a, a]]);
        let t = InternalTopology::from_geometry(&[6, 1, 1, 1, 1], ch4.view(), None).unwrap();
        assert_eq!(t.planar_watch().len(), 4);
        assert_eq!(t.constraints().counts().ndihedrals, 0);
    }

    #[test]
    fn far_fragments_are_welded_into_one_graph() {
        let x = pack_cart(&[
            [0.0, 0.0, 0.0],
            [0.74, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.74, 3.0, 0.0],
        ]);
        let t = InternalTopology::from_geometry(&[1, 1, 1, 1], x.view(), None).unwrap();
        assert!(t.constraints().counts().nbonds >= 3);
        assert!(t.check().complete);
    }

    #[test]
    fn welding_stops_when_the_round_budget_is_spent() {
        let x = pack_cart(&[[0.0; 3], [0.0, 0.0, 40.0]]);
        let cfg = TopologyConfig::default().with_max_weld_rounds(3);
        assert!(InternalTopology::from_geometry_with(&[1, 1], x.view(), None, &cfg).is_err());
        let t = InternalTopology::from_geometry(&[1, 1], x.view(), None).unwrap();
        assert_eq!(t.constraints().counts().nbonds, 1);
    }

    #[test]
    fn coincident_atoms_are_refused() {
        let x = pack_cart(&[[0.0; 3], [0.0; 3]]);
        assert!(InternalTopology::from_geometry(&[1, 1], x.view(), None).is_err());
    }

    #[test]
    fn a_water_split_across_the_cell_boundary_matches_the_free_molecule() {
        let (z, x) = water();
        let free = InternalTopology::from_geometry(&z, x.view(), None).unwrap();
        let cell = Cell::ortho(10.0, 10.0, 10.0).unwrap();
        let mut shifted = x.clone();
        // Put the oxygen at the edge and wrap one hydrogen to the far side.
        for a in 0..3 {
            shifted[3 * a] += 9.7;
            shifted[3 * a + 1] += 5.0;
            shifted[3 * a + 2] += 5.0;
        }
        for a in 0..3 {
            if shifted[3 * a] >= 10.0 {
                shifted[3 * a] -= 10.0;
            }
        }
        let t = InternalTopology::from_geometry(&z, shifted.view(), Some(&cell)).unwrap();
        assert!(t.constraints().is_periodic());
        let fc = free.constraints().counts();
        let tc = t.constraints().counts();
        assert_eq!((fc.nbonds, fc.nangles), (tc.nbonds, tc.nangles));
        let vf = free.constraints().values(x.view()).unwrap();
        let frame = t.frame(shifted.view()).unwrap();
        let vt = t.constraints().values(frame.view()).unwrap();
        for k in 0..vf.len() {
            assert!(
                (vf[k] - vt[k]).abs() < 1.0e-10,
                "{k}: {} vs {}",
                vf[k],
                vt[k]
            );
        }
        assert!(t.check().complete);
        assert!(!t.check().periodic_network);
    }

    #[test]
    fn a_zigzag_chain_wraps_the_cell_and_leaves_only_translations_free() {
        let cell = Cell::from_vectors(
            [2.5, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
            [0.0; 3],
        )
        .unwrap();
        let x = pack_cart(&[[0.0, 0.0, 0.0], [1.25, 0.8, 0.0]]);
        let t = InternalTopology::from_geometry(&[6, 6], x.view(), Some(&cell)).unwrap();
        let k = t.check();
        assert!(k.periodic_network);
        assert_eq!(k.rigid_dof, 3);
        assert_eq!(k.expected_rank, 3);
        assert!(k.complete, "{k:?}");
        let c = t.constraints().counts();
        assert_eq!(c.nbonds, 2);
    }

    #[test]
    fn an_incomplete_chart_is_flagged_and_can_be_refused() {
        let (z, x) = ethane();
        let t = InternalTopology::from_geometry(&z, x.view(), None).unwrap();
        let mut thin = Constraints::new(8).unwrap();
        for (k, eq) in t.constraints().equalities().iter().enumerate() {
            if let Equality::Bond { atoms, .. } = eq {
                let _ = k;
                thin.fix_bond(*atoms, x.view(), None).unwrap();
            }
        }
        let k = check_chart(&thin, None, x.view(), 8, PAPER_SV_TOL).unwrap();
        assert!(!k.complete);
        assert_eq!(k.rank, 7);
    }

    #[test]
    fn a_single_atom_has_no_internals() {
        let x = Array1::zeros(3);
        assert!(InternalTopology::from_geometry(&[1], x.view(), None).is_err());
    }

    #[test]
    fn heavy_elements_without_a_radius_are_refused() {
        let x = pack_cart(&[[0.0; 3], [1.5, 0.0, 0.0]]);
        let err = InternalTopology::from_geometry(&[1, 118], x.view(), None).unwrap_err();
        assert!(format!("{err}").contains("Cordero"));
    }

    #[test]
    fn invalid_rules_are_rejected() {
        let (z, x) = water();
        let bad = TopologyConfig::default().with_weld_growth(1.0);
        assert!(InternalTopology::from_geometry_with(&z, x.view(), None, &bad).is_err());
        let bad = TopologyConfig::default().with_planar_tol(2.0);
        assert!(
            InternalTopology::from_geometry_with(&[1], pack_cart(&[[0.0; 3]]).view(), None, &bad)
                .is_err()
        );
        let bad = TopologyConfig::default().with_linear_tol(2.0);
        assert!(InternalTopology::from_geometry_with(&z, x.view(), None, &bad).is_err());
    }

    #[test]
    fn config_setters_chain() {
        let c = TopologyConfig::default()
            .with_bond_scale(1.3)
            .with_dummy_distance(2.0)
            .with_require_complete(false);
        assert_eq!(c.bond_scale, 1.3);
        assert_eq!(c.dummy_distance, 2.0);
        assert!(!c.require_complete);
    }

    #[test]
    fn dummy_direction_axis_rule_is_perpendicular_to_the_line() {
        let va = [-1.1, 0.0, 0.0];
        let vc = [1.1, 0.0, 0.0];
        let d = dummy_direction(va, vc);
        assert!(dot3(d, [1.0, 0.0, 0.0]).abs() < f64::EPSILON.sqrt());
        assert!((norm3(d) - 1.0).abs() < f64::EPSILON.sqrt());
    }

    #[test]
    fn atom_map_round_trips() {
        let m = AtomMap::without_dummies(4);
        assert_eq!(m.n_frame(), 4);
        assert_eq!(m.frame_of_real(3), Some(3));
        assert_eq!(m.real_of_frame(4), None);
        assert!(!m.is_dummy(2));
    }
}
