//! Symmetry-resolved kinetic network of a mobile defect next to a fixed one.
//!
//! A configuration is the relative lattice vector from the fixed defect to
//! the mobile one together with the mobile defect's orientation director.
//! Both are integer vectors in a unit the caller chooses (for an fcc
//! lattice, `a/2`). A point group of integer matrices acts on both. A state
//! is one orbit of configurations, so every symmetry-distinct relative
//! vector and orientation is its own state; nothing is lumped by radius or
//! by axis.
//!
//! Moves are the mobile defect's elementary hops: a displacement, the
//! orientation before and after, and a saddle director `via` that tells
//! apart distinct paths between the same two configurations (two rotation
//! saddles, for example). A hop class is one orbit of the unordered pair of
//! configurations with its `via`, so one saddle serves the hop and its
//! reverse. Rates are looked up per edge through a closure that sees the
//! hop class, the direction, and the source state.
//!
//! [`PairNetwork::sites`] turns the network into [`VolumeSite`] rows for
//! [`crate::kinetic_recombination_volume`] and returns the hops that leave
//! the enumerated region as [`PairExit`] rows, which a host may hand to
//! [`crate::embed_sphere_return`].

use std::collections::BTreeMap;

use crate::defect_pair::{SiteOutcome, VolumeSite};
use crate::error::SaddleError;

/// Integer lattice vector.
pub type Vec3i = [i32; 3];

/// Integer point-group operation acting on [`Vec3i`] as `g * v`.
pub type Mat3i = [[i32; 3]; 3];

/// The 48 signed permutation matrices of the full cubic group `O_h`.
pub fn cubic_group() -> Vec<Mat3i> {
    let permutations = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let mut group = Vec::with_capacity(48);
    for permutation in permutations {
        for signs in 0..8 {
            let mut matrix = [[0; 3]; 3];
            for (row, column) in permutation.into_iter().enumerate() {
                matrix[row][column] = if signs & (1 << row) == 0 { 1 } else { -1 };
            }
            group.push(matrix);
        }
    }
    group
}

fn apply(g: &Mat3i, v: Vec3i) -> Vec3i {
    let mut out = [0; 3];
    for (row, slot) in out.iter_mut().enumerate() {
        *slot = g[row][0] * v[0] + g[row][1] * v[1] + g[row][2] * v[2];
    }
    out
}

fn add(left: Vec3i, right: Vec3i) -> Vec3i {
    [left[0] + right[0], left[1] + right[1], left[2] + right[2]]
}

/// Sign-normalized director: the first non-zero component is positive.
pub fn director(v: Vec3i) -> Vec3i {
    match v.iter().find(|c| **c != 0) {
        Some(c) if *c < 0 => [-v[0], -v[1], -v[2]],
        _ => v,
    }
}

/// Squared length of an integer vector.
pub fn norm2(v: Vec3i) -> i64 {
    v.iter().map(|c| i64::from(*c) * i64::from(*c)).sum()
}

/// Relative vector and orientation of the mobile defect.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PairConfig {
    /// Lattice vector from the fixed defect to the mobile one.
    pub offset: Vec3i,
    /// Orientation director of the mobile defect; the sign carries no
    /// meaning.
    pub orientation: Vec3i,
}

impl PairConfig {
    /// Configuration with a sign-normalized orientation.
    pub fn new(offset: Vec3i, orientation: Vec3i) -> Self {
        Self {
            offset,
            orientation: director(orientation),
        }
    }

    /// Image under `g`.
    pub fn transformed(&self, g: &Mat3i) -> Self {
        Self::new(apply(g, self.offset), apply(g, self.orientation))
    }

    /// Lexicographically least image under `group`: the orbit's label.
    pub fn canonical(&self, group: &[Mat3i]) -> Self {
        group
            .iter()
            .map(|g| self.transformed(g))
            .min()
            .unwrap_or(*self)
    }

    /// Number of distinct images under `group`.
    pub fn orbit_size(&self, group: &[Mat3i]) -> usize {
        let mut images: Vec<Self> = group.iter().map(|g| self.transformed(g)).collect();
        images.sort();
        images.dedup();
        images.len()
    }
}

/// One elementary move of the mobile defect.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct PairMove {
    /// Mechanism name; rates are looked up by it.
    pub kind: String,
    /// Change of the relative vector.
    pub displacement: Vec3i,
    /// Orientation before the move.
    pub from: Vec3i,
    /// Orientation after the move.
    pub to: Vec3i,
    /// Saddle director separating distinct paths between the same two
    /// configurations; zero when the configurations fix the path.
    pub via: Vec3i,
}

impl PairMove {
    /// Move with sign-normalized directors.
    pub fn new(
        kind: impl Into<String>,
        displacement: Vec3i,
        from: Vec3i,
        to: Vec3i,
        via: Vec3i,
    ) -> Self {
        Self {
            kind: kind.into(),
            displacement,
            from: director(from),
            to: director(to),
            via: director(via),
        }
    }

    fn transformed(&self, g: &Mat3i) -> Self {
        Self::new(
            self.kind.clone(),
            apply(g, self.displacement),
            apply(g, self.from),
            apply(g, self.to),
            apply(g, self.via),
        )
    }

    /// Every image of `moves` under `group`, without repeats.
    pub fn orbit(moves: &[PairMove], group: &[Mat3i]) -> Vec<PairMove> {
        let mut all: Vec<PairMove> = moves
            .iter()
            .flat_map(|item| group.iter().map(move |g| item.transformed(g)))
            .collect();
        all.sort();
        all.dedup();
        all
    }
}

/// Where an edge lands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PairTarget {
    /// An enumerated state, by index into [`PairNetwork::states`].
    State(usize),
    /// The mobile defect reaches the fixed defect (zero relative vector).
    Contact,
    /// A configuration beyond the enumerated radius.
    Outside,
}

/// Aggregated moves from a state's representative into one target through
/// one hop class.
#[derive(Clone, Debug, PartialEq)]
pub struct PairEdge {
    /// Hop class, by index into [`PairNetwork::hops`].
    pub hop: usize,
    /// True when the edge runs from [`PairHop::from`] to [`PairHop::to`]
    /// in the hop class's own frame.
    pub forward: bool,
    /// Landing state.
    pub target: PairTarget,
    /// One landing configuration, in the source representative's frame.
    pub config: PairConfig,
    /// Saddle director of that move, in the same frame.
    pub via: Vec3i,
    /// Number of moves from the representative that this edge collects.
    pub count: usize,
}

/// One symmetry-distinct state.
#[derive(Clone, Debug, PartialEq)]
pub struct PairStateClass {
    /// Canonical representative.
    pub config: PairConfig,
    /// Number of configurations in the orbit.
    pub multiplicity: usize,
    /// Outgoing edges from the representative.
    pub edges: Vec<PairEdge>,
}

/// One symmetry-distinct saddle, shared by a hop and its reverse.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PairHop {
    /// Mechanism name from [`PairMove::kind`].
    pub kind: String,
    /// One end, in the class's canonical frame.
    pub from: PairConfig,
    /// The other end, in the same frame.
    pub to: PairConfig,
    /// Saddle director in the same frame.
    pub via: Vec3i,
}

/// An outward hop from an enumerated state past the enumerated radius.
#[derive(Clone, Debug, PartialEq)]
pub struct PairExit {
    /// Source site id from [`PairNetwork::id`].
    pub source: String,
    /// Summed rate of the outward moves to this landing.
    pub rate: f64,
    /// Squared landing offset in lattice units.
    pub norm2: i64,
}

/// States and hop classes within a radius.
#[derive(Clone, Debug, PartialEq)]
pub struct PairNetwork {
    /// States ordered by squared radius, then by representative.
    pub states: Vec<PairStateClass>,
    /// Hop classes in order of first use.
    pub hops: Vec<PairHop>,
    /// Number of orientations a placement may take.
    pub orientations: usize,
    /// Largest enumerated squared radius.
    pub max_norm2: i64,
}

fn hop_key(
    group: &[Mat3i],
    kind: &str,
    source: PairConfig,
    target: PairConfig,
    via: Vec3i,
) -> (PairHop, bool) {
    let mut best: Option<((PairConfig, PairConfig, Vec3i), bool)> = None;
    for g in group {
        let a = source.transformed(g);
        let b = target.transformed(g);
        let v = director(apply(g, via));
        let (key, forward) = if a <= b {
            ((a, b, v), true)
        } else {
            ((b, a, v), false)
        };
        if best.as_ref().is_none_or(|(prior, _)| key < *prior) {
            best = Some((key, forward));
        }
    }
    let ((from, to, via), forward) = best.unwrap_or(((source, target, via), true));
    (
        PairHop {
            kind: kind.to_owned(),
            from,
            to,
            via,
        },
        forward,
    )
}

impl PairNetwork {
    /// Enumerate states and hop classes.
    ///
    /// `offsets` are the admissible relative vectors (the lattice sites);
    /// those with squared length in `1..=max_norm2` become states with every
    /// orientation in `orientations`. `moves` must already be closed under
    /// `group` (see [`PairMove::orbit`]). A move from an enumerated state
    /// that lands inside the radius on an offset not in `offsets` is an
    /// error.
    pub fn build(
        group: &[Mat3i],
        orientations: &[Vec3i],
        moves: &[PairMove],
        offsets: impl IntoIterator<Item = Vec3i>,
        max_norm2: i64,
    ) -> Result<Self, SaddleError> {
        if group.is_empty() || orientations.is_empty() {
            return Err(SaddleError::Invalid(
                "a pair network needs a group and at least one orientation".into(),
            ));
        }
        let mut orbits: BTreeMap<(i64, PairConfig), usize> = BTreeMap::new();
        let mut seen_orientations: Vec<Vec3i> = orientations.iter().map(|o| director(*o)).collect();
        seen_orientations.sort();
        seen_orientations.dedup();
        for offset in offsets {
            let length = norm2(offset);
            if length == 0 || length > max_norm2 {
                continue;
            }
            for orientation in &seen_orientations {
                let canonical = PairConfig::new(offset, *orientation).canonical(group);
                *orbits.entry((length, canonical)).or_insert(0) += 1;
            }
        }
        let mut states: Vec<PairStateClass> = orbits
            .into_iter()
            .map(|((_, config), multiplicity)| PairStateClass {
                config,
                multiplicity,
                edges: Vec::new(),
            })
            .collect();
        let index: BTreeMap<PairConfig, usize> = states
            .iter()
            .enumerate()
            .map(|(i, state)| (state.config, i))
            .collect();
        let mut hops: Vec<PairHop> = Vec::new();
        let mut hop_index: BTreeMap<(String, PairConfig, PairConfig, Vec3i), usize> =
            BTreeMap::new();
        for state in &mut states {
            let source = state.config;
            let mut edges: Vec<PairEdge> = Vec::new();
            for item in moves.iter().filter(|m| m.from == source.orientation) {
                let target = PairConfig::new(add(source.offset, item.displacement), item.to);
                let length = norm2(target.offset);
                let landing = if length == 0 {
                    PairTarget::Contact
                } else if length > max_norm2 {
                    PairTarget::Outside
                } else {
                    let canonical = target.canonical(group);
                    match index.get(&canonical) {
                        Some(i) => PairTarget::State(*i),
                        None => {
                            return Err(SaddleError::Invalid(format!(
                                "move {:?} from {:?} lands off the supplied offsets",
                                item.displacement, source.offset
                            )));
                        }
                    }
                };
                let (hop, forward) = hop_key(group, &item.kind, source, target, item.via);
                let key = (hop.kind.clone(), hop.from, hop.to, hop.via);
                let hop_id = *hop_index.entry(key).or_insert_with(|| {
                    hops.push(hop);
                    hops.len() - 1
                });
                let merge = edges.iter_mut().find(|edge| {
                    edge.hop == hop_id
                        && edge.forward == forward
                        && edge.target == landing
                        && (landing != PairTarget::Outside || norm2(edge.config.offset) == length)
                });
                match merge {
                    Some(edge) => edge.count += 1,
                    None => edges.push(PairEdge {
                        hop: hop_id,
                        forward,
                        target: landing,
                        config: target,
                        via: item.via,
                        count: 1,
                    }),
                }
            }
            state.edges = edges;
        }
        Ok(Self {
            states,
            hops,
            orientations: seen_orientations.len(),
            max_norm2,
        })
    }

    /// Site id of state `index`: offset and orientation components.
    pub fn id(&self, index: usize) -> String {
        let c = self.states[index].config;
        format!(
            "{}_{}_{}:{}{}{}",
            c.offset[0],
            c.offset[1],
            c.offset[2],
            c.orientation[0],
            c.orientation[1],
            c.orientation[2]
        )
    }

    /// Index of the state holding `config`.
    pub fn locate(&self, config: PairConfig, group: &[Mat3i]) -> Option<usize> {
        let canonical = config.canonical(group);
        self.states
            .iter()
            .position(|state| state.config == canonical)
    }

    /// Volume sites and outward exits.
    ///
    /// `spontaneous(state)` marks states that recombine without a barrier.
    /// `rate(state, edge)` is the rate of one move along `edge` from the
    /// state's representative; it is multiplied by [`PairEdge::count`].
    /// A site's count is its multiplicity over the number of orientations,
    /// so the counts sum to lattice sites. Contact edges go to
    /// `"recombined"`; outward edges are left out of the sites and returned
    /// as exits.
    pub fn sites(
        &self,
        spontaneous: &dyn Fn(usize) -> bool,
        rate: &dyn Fn(usize, &PairEdge) -> f64,
    ) -> Result<(Vec<VolumeSite>, Vec<PairExit>), SaddleError> {
        let mut sites = Vec::with_capacity(self.states.len());
        let mut exits = Vec::new();
        for (i, state) in self.states.iter().enumerate() {
            let id = self.id(i);
            let count = state.multiplicity as f64 / self.orientations as f64;
            if spontaneous(i) {
                sites.push(VolumeSite {
                    id,
                    count,
                    outcome: SiteOutcome::Spontaneous,
                    rates: Vec::new(),
                });
                continue;
            }
            let mut rates: Vec<(String, f64)> = Vec::new();
            for edge in &state.edges {
                let value = rate(i, edge) * edge.count as f64;
                if !(value.is_finite() && value >= 0.0) {
                    return Err(SaddleError::Invalid(format!(
                        "rate from {id} through hop {} is not finite and non-negative",
                        edge.hop
                    )));
                }
                match edge.target {
                    PairTarget::Contact => rates.push(("recombined".into(), value)),
                    PairTarget::State(j) => rates.push((self.id(j), value)),
                    PairTarget::Outside => exits.push(PairExit {
                        source: id.clone(),
                        rate: value,
                        norm2: norm2(edge.config.offset),
                    }),
                }
            }
            sites.push(VolumeSite {
                id,
                count,
                outcome: SiteOutcome::Kinetic,
                rates,
            });
        }
        Ok((sites, exits))
    }
}

/// Which classes to compute so the rest stay within a tolerance.
#[derive(Clone, Debug)]
pub struct SensitivityCover {
    /// Class indices by decreasing `|sensitivity|`.
    pub order: Vec<usize>,
    /// How many leading entries of [`Self::order`] are to be computed.
    pub covered: usize,
    /// `spread` times the summed `|sensitivity|` of the classes after
    /// each rank: the worst coherent shift the remaining model rates can
    /// make when all of them move by `spread` in `ln k`.
    pub remaining: Vec<f64>,
}

/// Ranks classes by `|d result / d ln k|` and covers the fewest leading
/// ones after which `spread * sum |sensitivity|` over the rest is below
/// `tolerance`. Non-finite sensitivities rank first and are always covered.
pub fn sensitivity_cover(sensitivity: &[f64], spread: f64, tolerance: f64) -> SensitivityCover {
    let magnitude = |i: usize| {
        let value = sensitivity[i].abs();
        if value.is_finite() {
            value
        } else {
            f64::INFINITY
        }
    };
    let mut order: Vec<usize> = (0..sensitivity.len()).collect();
    order.sort_by(|l, r| magnitude(*r).total_cmp(&magnitude(*l)).then(l.cmp(r)));
    let mut remaining = vec![0.0; order.len()];
    let mut tail = 0.0;
    for rank in (0..order.len()).rev() {
        remaining[rank] = spread.abs() * tail;
        tail += magnitude(order[rank]);
    }
    let head = spread.abs() * tail;
    let covered = if head < tolerance {
        0
    } else {
        remaining
            .iter()
            .position(|rest| *rest < tolerance)
            .map_or(order.len(), |rank| rank + 1)
    };
    SensitivityCover {
        order,
        covered,
        remaining,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fcc_offsets(reach: i32) -> Vec<Vec3i> {
        let mut out = Vec::new();
        for x in -reach..=reach {
            for y in -reach..=reach {
                for z in -reach..=reach {
                    if (x + y + z).rem_euclid(2) == 0 {
                        out.push([x, y, z]);
                    }
                }
            }
        }
        out
    }

    fn dumbbell_moves() -> Vec<PairMove> {
        let seed = [
            PairMove::new("translate", [1, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]),
            PairMove::new("rotate", [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]),
        ];
        PairMove::orbit(&seed, &cubic_group())
    }

    #[test]
    fn sensitivity_cover_stops_once_the_rest_is_inside_the_tolerance() {
        let cover = sensitivity_cover(&[1.0, -8.0, 0.5, 4.0, -2.0], 0.5, 1.6);
        assert_eq!(cover.order, vec![1, 3, 4, 0, 2]);
        assert_eq!(cover.remaining, vec![3.75, 1.75, 0.75, 0.25, 0.0]);
        assert_eq!(cover.covered, 3);
        assert_eq!(sensitivity_cover(&[1.0, 1.0], 0.5, 1.5).covered, 0);
        assert_eq!(sensitivity_cover(&[1.0, f64::NAN], 0.5, 10.0).order[0], 1);
        assert_eq!(sensitivity_cover(&[1.0, f64::NAN], 0.5, 10.0).covered, 1);
    }

    #[test]
    fn cubic_group_is_closed() {
        let group = cubic_group();
        assert_eq!(group.len(), 48);
        for a in &group {
            for b in &group {
                let mut product = [[0; 3]; 3];
                for (r, row) in product.iter_mut().enumerate() {
                    for (c, slot) in row.iter_mut().enumerate() {
                        *slot = (0..3).map(|k| a[r][k] * b[k][c]).sum();
                    }
                }
                assert!(group.contains(&product));
            }
        }
    }

    #[test]
    fn dumbbell_moves_have_eight_translations_and_four_rotations() {
        let moves = dumbbell_moves();
        let from_x = |kind: &str| {
            moves
                .iter()
                .filter(|m| m.kind == kind && m.from == [1, 0, 0])
                .count()
        };
        assert_eq!(from_x("translate"), 8);
        assert_eq!(from_x("rotate"), 4);
    }

    #[test]
    fn multiplicities_count_every_configuration() {
        let group = cubic_group();
        let network = PairNetwork::build(
            &group,
            &[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            &dumbbell_moves(),
            fcc_offsets(6),
            8,
        )
        .unwrap();
        let sites = fcc_offsets(6)
            .into_iter()
            .filter(|o| (1..=8).contains(&norm2(*o)))
            .count();
        let total: usize = network.states.iter().map(|s| s.multiplicity).sum();
        assert_eq!(total, 3 * sites);
        // nn shell <110>: a dumbbell along the bond plane or normal to it.
        let nn: Vec<_> = network
            .states
            .iter()
            .filter(|s| norm2(s.config.offset) == 2)
            .map(|s| s.multiplicity)
            .collect();
        assert_eq!(nn.iter().sum::<usize>(), 36);
        assert_eq!(nn.len(), 2);
    }

    #[test]
    fn reverse_moves_share_a_hop_class() {
        let group = cubic_group();
        let network = PairNetwork::build(
            &group,
            &[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            &dumbbell_moves(),
            fcc_offsets(8),
            24,
        )
        .unwrap();
        for (i, state) in network.states.iter().enumerate() {
            for edge in &state.edges {
                let PairTarget::State(j) = edge.target else {
                    continue;
                };
                let back = network.states[j]
                    .edges
                    .iter()
                    .find(|other| other.hop == edge.hop);
                let back =
                    back.unwrap_or_else(|| panic!("no reverse of hop {} from {i}", edge.hop));
                if i != j {
                    assert_ne!(back.forward, edge.forward);
                }
            }
        }
    }

    #[test]
    fn uniform_rates_with_spontaneous_contact_give_detailed_balance() {
        let group = cubic_group();
        let network = PairNetwork::build(
            &group,
            &[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            &dumbbell_moves(),
            fcc_offsets(8),
            12,
        )
        .unwrap();
        let (sites, exits) = network
            .sites(&|i| norm2(network.states[i].config.offset) <= 2, &|_, _| {
                1.0
            })
            .unwrap();
        let count: f64 = sites.iter().map(|s| s.count).sum();
        let lattice = fcc_offsets(8)
            .into_iter()
            .filter(|o| (1..=12).contains(&norm2(*o)))
            .count();
        assert!((count - lattice as f64).abs() < 1.0e-9);
        assert!(!exits.is_empty());
        // Outward flux from one state class equals the inward flux of its
        // reverse with uniform rates, weighted by multiplicity.
        let mut flux: BTreeMap<(usize, usize), f64> = BTreeMap::new();
        for (i, state) in network.states.iter().enumerate() {
            for edge in &state.edges {
                if let PairTarget::State(j) = edge.target {
                    *flux.entry((i, j)).or_insert(0.0) +=
                        state.multiplicity as f64 * edge.count as f64;
                }
            }
        }
        for ((i, j), value) in &flux {
            let back = flux.get(&(*j, *i)).copied().unwrap_or(0.0);
            assert!((value - back).abs() < 1.0e-9, "{i}->{j} {value} vs {back}");
        }
        let volume = crate::kinetic_recombination_volume(
            &crate::embed_sphere_return(
                &sites,
                12f64.sqrt(),
                &exits
                    .iter()
                    .map(|exit| crate::SphereLanding {
                        source: exit.source.clone(),
                        boundary: exit.source.clone(),
                        rate: exit.rate,
                        distance: (exit.norm2 as f64).sqrt(),
                    })
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
            1.0,
        )
        .unwrap();
        assert!(volume.volume > 0.0);
    }

    #[test]
    fn matched_capture_radius_does_not_drift_with_the_cutoff() {
        let group = cubic_group();
        let moves = dumbbell_moves();
        let matched = |max_norm2: i64| {
            let network = PairNetwork::build(
                &group,
                &[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                &moves,
                fcc_offsets(16),
                max_norm2,
            )
            .unwrap();
            let (sites, exits) = network
                .sites(&|i| norm2(network.states[i].config.offset) <= 4, &|_, _| {
                    1.0
                })
                .unwrap();
            let radius_of = |i: usize| (norm2(network.states[i].config.offset) as f64).sqrt();
            let rmax = (max_norm2 as f64).sqrt();
            let outer: Vec<(String, f64)> = (0..network.states.len())
                .filter(|i| radius_of(*i) >= rmax - 1.5)
                .map(|i| (network.id(i), radius_of(i)))
                .collect();
            let hops: Vec<crate::OuterHop> = exits
                .iter()
                .map(|exit| crate::OuterHop {
                    source: exit.source.clone(),
                    rate: exit.rate,
                    distance: (exit.norm2 as f64).sqrt(),
                })
                .collect();
            let found = crate::matched_capture_radius(&sites, &hops, &outer, 1.0).unwrap();
            let profile: Vec<f64> = found
                .volume
                .placements
                .iter()
                .enumerate()
                .filter(|(i, _)| radius_of(*i) >= 0.5 * rmax)
                .map(|(i, place)| place.probability * radius_of(i))
                .collect();
            (found.radius, profile)
        };
        let (small, _) = matched(64);
        let (large, profile) = matched(144);
        assert!((small - large).abs() < 0.01 * large, "{small} vs {large}");
        let mean = profile.iter().sum::<f64>() / profile.len() as f64;
        for value in profile {
            assert!(
                (value - mean).abs() < 0.05 * mean,
                "P r {value} against {mean}"
            );
        }
    }
}
