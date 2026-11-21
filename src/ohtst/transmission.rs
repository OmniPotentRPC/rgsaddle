//! The transmission coefficient of a dividing plane from short
//! trajectories.
//!
//! Each shot starts from a configuration of the plane's own ensemble with
//! a velocity from the flux-weighted Maxwell–Boltzmann distribution,
//! `(ν · v)₊ exp(-vᵀ M v / 2kT)`, and is run forward with that velocity
//! and backward with its reverse; the backward half stands for the past
//! of the crossing. With `h_P` one in the product state:
//!
//! - [`Counting::Final`] is the reactive flux of Bennett (1977) and
//!   Chandler, J. Chem. Phys. 68, 2959 (1978), Eq. 8.1 of the design
//!   survey with both halves of the velocity distribution folded onto the
//!   positive one: `κ(t) = ⟨h_P(t) - h_P(-t)⟩`. A shot scores minus one,
//!   zero or one.
//! - [`Counting::FirstCrossing`] is the effective positive flux of van
//!   Erp, Moroni and Bolhuis, J. Chem. Phys. 118, 7762 (2003),
//!   doi:10.1063/1.1562614, after Anderson, J. Chem. Phys. 58, 4684
//!   (1973), doi:10.1063/1.1679032: `κ(t) = ⟨χ_R(-t) h_P(t)⟩`, where
//!   `χ_R(-t)` is one when the past reaches the reactant without crossing
//!   the plane again. A reactive trajectory then counts once, at its first
//!   crossing, and a shot scores zero or one, which keeps the variance
//!   down when trajectories recross many times.
//!
//! The two agree once the halves are long enough to commit every
//! trajectory to a state.
//!
//! The end state is the side of the plane ([`EndState::Side`]) or the
//! basin that a quench from the configuration reaches
//! ([`EndState::Quench`]): steepest-descent flow to a minimum, then the
//! state of the nearest reference minimum. The distance to the references
//! can be the intensive permutation-invariant one of Goswami and Jónsson,
//! ChemPhysChem 27, e202500730 (2026), doi:10.1002/cphc.202500730, Eq. 12.5
//! of the design survey ([`PermutationDistance`]), so that configurations
//! that differ only in which like atoms sit on which sites are one state.
//! A trajectory still near the plane at the end is counted by its basin
//! rather than by the side it happens to be on.
//!
//! The dynamics is Langevin BAOAB with a mass for every coordinate, and
//! Newtonian when the friction is zero. It must be the dynamics that `κ`
//! corrects: for a Langevin thermostat the same friction, since `κ`
//! depends on it (Kramers, Physica 7, 284 (1940)). On a Cartesian surface
//! with masses, the velocities are drawn and the noise added in the
//! mass-weighted frame, so the trajectories are those of the
//! mass-weighted surface with unit masses, step for step.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, Array2, ArrayView1, Zip};
use rayon::prelude::*;
use rgmin::{Control, Method, Oracle, Solver, StopNorm};

use crate::bennett::Transmission;
use crate::error::SaddleError;
use crate::mic::Cell;
use crate::minmode::PointSurface;
use crate::ohtst::plane::{Hyperplane, orthonormal_rows};
use crate::region::{BoundaryViolation, Region};
use crate::stats::{Rng, block_mean};

/// A transmission coefficient and its standard error, the factor
/// [`OhTstRate::transmitted`](crate::ohtst::OhTstRate::transmitted) applies.
pub trait TransmissionEstimate {
    /// `κ` and its standard error.
    fn kappa(&self) -> (f64, f64);
}

impl TransmissionEstimate for Transmission {
    fn kappa(&self) -> (f64, f64) {
        (self.kappa, self.error)
    }
}

/// The two states a configuration can be assigned to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Endpoint {
    /// The reactant.
    Reactant,
    /// The product.
    Product,
}

/// The intensive permutation-invariant distance between two sets of atom
/// positions, Eq. 12.5 of the design survey:
///
/// `D = max_t (1/N_t) min_π Σ_k |r¹_k - r²_π(k)|`,
///
/// with `π` over the permutations of the `N_t` atoms of species `t`, found
/// by the Hungarian method on minimum-image distances. Rigid-body motion
/// is not removed.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct PermutationDistance {
    /// The species of each atom. Atoms of one species may be exchanged.
    pub species: Vec<usize>,
    /// The periodic cell for minimum-image distances. `None` is open.
    pub cell: Option<Cell>,
    /// Atom masses when the coordinates are mass-weighted, `q = √m x`, so
    /// that the distance is between Cartesian positions.
    pub masses: Option<Vec<f64>>,
    /// The atoms that enter the distance, such as those near a defect.
    /// The average of each species runs over these alone, so a hop is not
    /// diluted by the rest of the cell. `None` takes every atom.
    pub atoms: Option<Vec<usize>>,
}

impl PermutationDistance {
    /// All atoms, open boundaries and Cartesian coordinates.
    pub fn new(species: Vec<usize>) -> Self {
        Self {
            species,
            cell: None,
            masses: None,
            atoms: None,
        }
    }

    /// Set [`Self::species`].
    #[must_use]
    pub fn with_species(mut self, species: Vec<usize>) -> Self {
        self.species = species;
        self
    }

    /// Set [`Self::cell`].
    #[must_use]
    pub fn with_cell(mut self, cell: Cell) -> Self {
        self.cell = Some(cell);
        self
    }

    /// Set [`Self::masses`].
    #[must_use]
    pub fn with_masses(mut self, masses: Vec<f64>) -> Self {
        self.masses = Some(masses);
        self
    }

    /// Set [`Self::atoms`].
    #[must_use]
    pub fn with_atoms(mut self, atoms: Vec<usize>) -> Self {
        self.atoms = Some(atoms);
        self
    }

    /// The distance between the flat `(x, y, z)` coordinates `a` and `b`.
    pub fn distance(&self, a: ArrayView1<f64>, b: ArrayView1<f64>) -> Result<f64, SaddleError> {
        let atoms = self.species.len();
        if a.len() != 3 * atoms || b.len() != 3 * atoms {
            return Err(SaddleError::Shape(
                "the configurations need three coordinates for every species entry".into(),
            ));
        }
        if let Some(masses) = &self.masses
            && (masses.len() != atoms || !masses.iter().all(|m| m.is_finite() && *m > 0.0))
        {
            return Err(SaddleError::Invalid(
                "the distance needs one positive mass per atom".into(),
            ));
        }
        let position = |x: ArrayView1<f64>, k: usize| {
            let scale = self.masses.as_ref().map_or(1.0, |m| m[k].sqrt());
            [x[3 * k] / scale, x[3 * k + 1] / scale, x[3 * k + 2] / scale]
        };
        let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        match &self.atoms {
            Some(chosen) => {
                for &k in chosen {
                    if k >= atoms {
                        return Err(SaddleError::Shape(
                            "a chosen atom is past the end of the configuration".into(),
                        ));
                    }
                    groups.entry(self.species[k]).or_default().push(k);
                }
            }
            None => {
                for (k, &t) in self.species.iter().enumerate() {
                    groups.entry(t).or_default().push(k);
                }
            }
        }
        let mut largest: f64 = 0.0;
        for members in groups.values() {
            let m = members.len();
            let mut cost = vec![0.0; m * m];
            for (i, &k) in members.iter().enumerate() {
                let p = position(a, k);
                for (j, &l) in members.iter().enumerate() {
                    let q = position(b, l);
                    let squared = match &self.cell {
                        Some(cell) => cell.dist2_euclidean(p, q),
                        None => (0..3).map(|c| (q[c] - p[c]).powi(2)).sum(),
                    };
                    cost[i * m + j] = squared.sqrt();
                }
            }
            if !cost.iter().all(|c| c.is_finite()) {
                return Err(SaddleError::NonFinite("permutation distance"));
            }
            let (total, _) = assignment(&cost, m);
            largest = largest.max(total / m as f64);
        }
        Ok(largest)
    }
}

/// How a quenched configuration is compared with the reference minima.
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub enum StateDistance {
    /// The Euclidean distance in the surface's coordinates.
    #[default]
    Euclidean,
    /// The permutation-invariant distance between atom positions.
    Permutation(Box<PermutationDistance>),
}

impl From<PermutationDistance> for StateDistance {
    fn from(distance: PermutationDistance) -> Self {
        Self::Permutation(Box::new(distance))
    }
}

impl StateDistance {
    /// The distance between `a` and `b`.
    pub fn distance(&self, a: ArrayView1<f64>, b: ArrayView1<f64>) -> Result<f64, SaddleError> {
        match self {
            Self::Euclidean => {
                if a.len() != b.len() {
                    return Err(SaddleError::Shape(
                        "the configurations differ in length".into(),
                    ));
                }
                Ok((&a - &b).dot(&(&a - &b)).sqrt())
            }
            Self::Permutation(distance) => distance.distance(a, b),
        }
    }
}

/// The quench that takes a configuration to the minimum of its basin.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct QuenchConfig {
    /// The minimizer. [`Method::Ode12r`] follows the steepest-descent flow,
    /// whose end is the basin of attraction; momentum methods can cross
    /// into the next basin from near a ridge.
    pub method: Method,
    /// Largest force on any atom, a triple of coordinates, accepted as a
    /// minimum.
    pub force_tol: f64,
    /// Most steps; a quench that does not converge leaves the
    /// configuration unassigned.
    pub max_steps: usize,
    /// Largest move of one atom in one step.
    pub max_move: f64,
    /// The opening step.
    pub initial_step: f64,
}

impl Default for QuenchConfig {
    fn default() -> Self {
        Self {
            method: Method::Ode12r,
            force_tol: 1.0e-4,
            max_steps: 5000,
            max_move: 0.1,
            initial_step: 0.1,
        }
    }
}

impl QuenchConfig {
    /// Set [`Self::method`].
    #[must_use]
    pub fn with_method(mut self, method: Method) -> Self {
        self.method = method;
        self
    }

    /// Set [`Self::force_tol`].
    #[must_use]
    pub fn with_force_tol(mut self, force_tol: f64) -> Self {
        self.force_tol = force_tol;
        self
    }

    /// Set [`Self::max_steps`].
    #[must_use]
    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// Set [`Self::max_move`].
    #[must_use]
    pub fn with_max_move(mut self, max_move: f64) -> Self {
        self.max_move = max_move;
        self
    }

    /// Set [`Self::initial_step`].
    #[must_use]
    pub fn with_initial_step(mut self, initial_step: f64) -> Self {
        self.initial_step = initial_step;
        self
    }

    fn check(&self) -> Result<(), SaddleError> {
        if !(self.force_tol.is_finite() && self.force_tol > 0.0)
            || self.max_steps == 0
            || !(self.max_move.is_finite() && self.max_move > 0.0)
            || !(self.initial_step.is_finite() && self.initial_step > 0.0)
        {
            return Err(SaddleError::Invalid(
                "the quench needs a positive tolerance, step count, move cap and opening step"
                    .into(),
            ));
        }
        Ok(())
    }
}

/// Quench `start`; the minimum, or `None` when the quench did not
/// converge, and the evaluations it took.
fn quench<S: PointSurface>(
    surface: &S,
    start: ArrayView1<f64>,
    config: &QuenchConfig,
) -> Result<(Option<Array1<f64>>, usize), SaddleError> {
    let n = start.len();
    let calls = AtomicUsize::new(0);
    let mut solver = Solver::new(
        config.method.clone(),
        Control::default()
            .with_maxiter(config.max_steps)
            .with_gtol(config.force_tol)
            .with_stop_norm(StopNorm::MaxAtom)
            .with_istep(config.initial_step),
        n,
    );
    solver.set_atom_maxmove(config.max_move);
    let oracle = Oracle::unbounded(n, |x: ArrayView1<f64>| {
        calls.fetch_add(1, Ordering::Relaxed);
        surface
            .eval(x)
            .unwrap_or_else(|_| (f64::NAN, Array1::from_elem(x.len(), f64::NAN)))
    });
    let mut x = start.to_owned();
    if solver.run(&oracle, &mut x).is_err() {
        return Ok((None, calls.into_inner()));
    }
    let (energy, gradient) = surface.eval(x.view())?;
    let evaluations = calls.into_inner() + 1;
    let converged = energy.is_finite() && StopNorm::MaxAtom.of(gradient.view()) < config.force_tol;
    Ok((converged.then_some(x), evaluations))
}

/// Reference minima of the two states and the rule that assigns a
/// quenched configuration to one of them.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct Basins {
    /// Minima of the reactant state.
    pub reactants: Vec<Array1<f64>>,
    /// Minima of the product state, for instance the symmetry-equivalent
    /// ends of the hops through one plane.
    pub products: Vec<Array1<f64>>,
    /// How configurations are compared with the minima.
    pub distance: StateDistance,
    /// A quenched configuration takes the state of its nearest minimum
    /// when it lies within this fraction of the smallest distance between
    /// a reactant and a product minimum, and is unassigned otherwise.
    pub tolerance: f64,
    /// An absolute acceptance distance in place of [`Self::tolerance`],
    /// such as a fraction of the nearest-neighbour distance.
    pub threshold: Option<f64>,
    /// The quench.
    pub quench: QuenchConfig,
}

impl Basins {
    /// Reference minima, the Euclidean distance and a tolerance of a
    /// quarter of the reactant–product distance.
    pub fn new(reactants: Vec<Array1<f64>>, products: Vec<Array1<f64>>) -> Self {
        Self {
            reactants,
            products,
            distance: StateDistance::Euclidean,
            tolerance: 0.25,
            threshold: None,
            quench: QuenchConfig::default(),
        }
    }

    /// Set [`Self::reactants`].
    #[must_use]
    pub fn with_reactants(mut self, reactants: Vec<Array1<f64>>) -> Self {
        self.reactants = reactants;
        self
    }

    /// Set [`Self::products`].
    #[must_use]
    pub fn with_products(mut self, products: Vec<Array1<f64>>) -> Self {
        self.products = products;
        self
    }

    /// Set [`Self::distance`].
    #[must_use]
    pub fn with_distance(mut self, distance: StateDistance) -> Self {
        self.distance = distance;
        self
    }

    /// Set [`Self::tolerance`].
    #[must_use]
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set [`Self::threshold`].
    #[must_use]
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = Some(threshold);
        self
    }

    /// Set [`Self::quench`].
    #[must_use]
    pub fn with_quench(mut self, quench: QuenchConfig) -> Self {
        self.quench = quench;
        self
    }

    /// The distance within which a minimum is assigned.
    pub fn acceptance(&self) -> Result<f64, SaddleError> {
        if self.reactants.is_empty() || self.products.is_empty() {
            return Err(SaddleError::Invalid(
                "the basins need at least one reactant and one product minimum".into(),
            ));
        }
        if let Some(threshold) = self.threshold {
            if !(threshold.is_finite() && threshold > 0.0) {
                return Err(SaddleError::Invalid(
                    "the acceptance threshold must be finite and positive".into(),
                ));
            }
            return Ok(threshold);
        }
        if !(self.tolerance > 0.0 && self.tolerance <= 0.5) {
            return Err(SaddleError::Invalid(
                "the acceptance tolerance must lie in (0, 0.5]".into(),
            ));
        }
        let mut separation = f64::INFINITY;
        for r in &self.reactants {
            for p in &self.products {
                separation = separation.min(self.distance.distance(r.view(), p.view())?);
            }
        }
        if !(separation.is_finite() && separation > 0.0) {
            return Err(SaddleError::Invalid(
                "a reactant and a product minimum coincide".into(),
            ));
        }
        Ok(self.tolerance * separation)
    }

    /// The state of the minimum nearest to `q`, without a quench, or
    /// `None` when it is farther than [`Self::acceptance`].
    pub fn classify(&self, q: ArrayView1<f64>) -> Result<Option<Endpoint>, SaddleError> {
        let acceptance = self.acceptance()?;
        self.classify_within(q, acceptance)
    }

    fn classify_within(
        &self,
        q: ArrayView1<f64>,
        acceptance: f64,
    ) -> Result<Option<Endpoint>, SaddleError> {
        let nearest = |minima: &[Array1<f64>]| -> Result<f64, SaddleError> {
            let mut best = f64::INFINITY;
            for m in minima {
                best = best.min(self.distance.distance(q, m.view())?);
            }
            Ok(best)
        };
        let (reactant, product) = (nearest(&self.reactants)?, nearest(&self.products)?);
        Ok(if reactant <= product && reactant <= acceptance {
            Some(Endpoint::Reactant)
        } else if product < reactant && product <= acceptance {
            Some(Endpoint::Product)
        } else {
            None
        })
    }

    /// Quench `q` and classify the minimum it reaches; `None` when the
    /// quench does not converge or reaches neither state. Also returns
    /// the evaluations the quench took.
    pub fn assign<S: PointSurface>(
        &self,
        surface: &S,
        q: ArrayView1<f64>,
    ) -> Result<(Option<Endpoint>, usize), SaddleError> {
        self.quench.check()?;
        let acceptance = self.acceptance()?;
        self.assign_within(surface, q, acceptance)
    }

    fn assign_within<S: PointSurface>(
        &self,
        surface: &S,
        q: ArrayView1<f64>,
        acceptance: f64,
    ) -> Result<(Option<Endpoint>, usize), SaddleError> {
        let (minimum, evaluations) = quench(surface, q, &self.quench)?;
        let state = match minimum {
            Some(minimum) => self.classify_within(minimum.view(), acceptance)?,
            None => None,
        };
        Ok((state, evaluations))
    }
}

/// Hard transition region whose exterior is any basin reached by an rgmin
/// steepest-descent quench.
///
/// The reflection normal joins the reference minimum of the reached state
/// nearest the quenched configuration to the reference minimum of the
/// opposite state nearest that configuration. Membership itself comes from
/// [`Basins::assign`], not from that linear normal model.
pub struct QuenchedTransitionRegion<S> {
    surface: Arc<S>,
    basins: Basins,
}

impl<S> std::fmt::Debug for QuenchedTransitionRegion<S> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("QuenchedTransitionRegion")
            .field("basins", &self.basins)
            .finish_non_exhaustive()
    }
}

impl<S> QuenchedTransitionRegion<S> {
    /// Build a basin-excluding predicate.
    pub fn new(surface: Arc<S>, basins: Basins) -> Self {
        Self { surface, basins }
    }

    /// Basin assignment used by the predicate.
    pub fn basins(&self) -> &Basins {
        &self.basins
    }
}

impl<S: PointSurface + Send + Sync> Region for QuenchedTransitionRegion<S> {
    fn violation(
        &self,
        point: ArrayView1<'_, f64>,
    ) -> Result<Option<BoundaryViolation>, SaddleError> {
        self.basins.quench.check()?;
        let acceptance = self.basins.acceptance()?;
        let (reached, _) = quench(self.surface.as_ref(), point, &self.basins.quench)?;
        let Some(reached) = reached else {
            return Ok(None);
        };
        let Some(state) = self.basins.classify_within(reached.view(), acceptance)? else {
            return Ok(None);
        };
        let (outside, opposite) = match state {
            Endpoint::Reactant => (&self.basins.reactants, &self.basins.products),
            Endpoint::Product => (&self.basins.products, &self.basins.reactants),
        };
        let nearest = |minima: &[Array1<f64>]| -> Result<Array1<f64>, SaddleError> {
            let mut best: Option<(f64, Array1<f64>)> = None;
            for minimum in minima {
                let distance = self
                    .basins
                    .distance
                    .distance(reached.view(), minimum.view())?;
                if best.as_ref().is_none_or(|known| distance < known.0) {
                    best = Some((distance, minimum.clone()));
                }
            }
            best.map(|(_, minimum)| minimum)
                .ok_or_else(|| SaddleError::Invalid("basin reference set is empty".into()))
        };
        let normal = &nearest(outside)? - &nearest(opposite)?;
        Ok(Some(BoundaryViolation::predicate(normal)?))
    }
}

/// How the end of a half trajectory is assigned to a state.
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub enum EndState {
    /// The side of the plane: the reactant behind it, the product ahead.
    #[default]
    Side,
    /// The basin that a quench from the configuration reaches.
    Quench(Box<Basins>),
}

impl From<Basins> for EndState {
    fn from(basins: Basins) -> Self {
        Self::Quench(Box::new(basins))
    }
}

/// Which crossings count.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Counting {
    /// Bennett–Chandler: the product indicator at the end of the forward
    /// half minus that at the end of the backward half.
    Final,
    /// The effective positive flux: one when the backward half reaches the
    /// reactant without crossing the plane again and the forward half ends
    /// in the product.
    #[default]
    FirstCrossing,
}

/// Shooting from one plane.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct TransmissionConfig {
    /// `kT` in the energy unit of the surface.
    pub thermal: f64,
    /// Langevin friction `γ` in inverse time. Zero is Newtonian.
    pub friction: f64,
    /// BAOAB time step.
    pub time_step: f64,
    /// Steps of each half, forward and backward.
    pub steps: usize,
    /// Steps between checkpoints, where the end state is assigned and
    /// `κ(t)` recorded. The last step is always a checkpoint.
    pub check_every: usize,
    /// A half ends once this many checkpoints in a row agree on a state,
    /// which keeps slow returns from a shallow well out of `κ`. The window
    /// must outlast the time a trajectory lingers at the barrier: a shorter
    /// one commits trajectories that cross back later and raises `κ`. Zero
    /// runs every half to the last step.
    pub commit: usize,
    /// Shots, spread evenly over the starts in their order, so that the
    /// shots from one start are consecutive.
    pub shots: usize,
    /// Most blocks of consecutive shots for the standard error. Blocks
    /// are widened to twice the autocorrelation time of the shots, which
    /// correlated starts or repeated ones lengthen.
    pub blocks: usize,
    /// Seed of the velocities and the noise; shot `i` uses the seed plus
    /// `i`, so the result does not depend on the number of threads.
    pub seed: u64,
    /// The mass of every coordinate of a Cartesian surface. `None` for
    /// mass-weighted coordinates, where every mass is one.
    pub masses: Option<Vec<f64>>,
    /// Which crossings count.
    pub counting: Counting,
    /// How the end of a half is assigned.
    pub end_state: EndState,
    /// Draw no velocity and add no noise along the surface's excluded
    /// modes at the start, such as rigid translations.
    pub hold_excluded: bool,
}

impl Default for TransmissionConfig {
    fn default() -> Self {
        Self {
            thermal: 0.02585,
            friction: 0.0,
            time_step: 0.05,
            steps: 400,
            check_every: 40,
            commit: 0,
            shots: 256,
            blocks: 16,
            seed: 1,
            masses: None,
            counting: Counting::FirstCrossing,
            end_state: EndState::Side,
            hold_excluded: true,
        }
    }
}

impl TransmissionConfig {
    /// Set [`Self::thermal`].
    #[must_use]
    pub fn with_thermal(mut self, thermal: f64) -> Self {
        self.thermal = thermal;
        self
    }

    /// Set [`Self::friction`].
    #[must_use]
    pub fn with_friction(mut self, friction: f64) -> Self {
        self.friction = friction;
        self
    }

    /// Set [`Self::time_step`].
    #[must_use]
    pub fn with_time_step(mut self, time_step: f64) -> Self {
        self.time_step = time_step;
        self
    }

    /// Set [`Self::steps`].
    #[must_use]
    pub fn with_steps(mut self, steps: usize) -> Self {
        self.steps = steps;
        self
    }

    /// Set [`Self::check_every`].
    #[must_use]
    pub fn with_check_every(mut self, check_every: usize) -> Self {
        self.check_every = check_every;
        self
    }

    /// Set [`Self::commit`].
    #[must_use]
    pub fn with_commit(mut self, commit: usize) -> Self {
        self.commit = commit;
        self
    }

    /// Set [`Self::shots`].
    #[must_use]
    pub fn with_shots(mut self, shots: usize) -> Self {
        self.shots = shots;
        self
    }

    /// Set [`Self::blocks`].
    #[must_use]
    pub fn with_blocks(mut self, blocks: usize) -> Self {
        self.blocks = blocks;
        self
    }

    /// Set [`Self::seed`].
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set [`Self::masses`].
    #[must_use]
    pub fn with_masses(mut self, masses: Vec<f64>) -> Self {
        self.masses = Some(masses);
        self
    }

    /// Set [`Self::counting`].
    #[must_use]
    pub fn with_counting(mut self, counting: Counting) -> Self {
        self.counting = counting;
        self
    }

    /// Set [`Self::end_state`].
    #[must_use]
    pub fn with_end_state(mut self, end_state: EndState) -> Self {
        self.end_state = end_state;
        self
    }

    /// Set [`Self::hold_excluded`].
    #[must_use]
    pub fn with_hold_excluded(mut self, hold_excluded: bool) -> Self {
        self.hold_excluded = hold_excluded;
        self
    }

    fn check(&self, n: usize) -> Result<(), SaddleError> {
        if !(self.thermal.is_finite() && self.thermal > 0.0) {
            return Err(SaddleError::Invalid(
                "shooting kT must be finite and positive".into(),
            ));
        }
        if !(self.friction.is_finite() && self.friction >= 0.0) {
            return Err(SaddleError::Invalid(
                "shooting friction must be finite and non-negative".into(),
            ));
        }
        if !(self.time_step.is_finite() && self.time_step > 0.0) || self.steps == 0 {
            return Err(SaddleError::Invalid(
                "shooting needs a positive time step and at least one step".into(),
            ));
        }
        if self.check_every == 0 || self.blocks < 2 || self.shots < self.blocks {
            return Err(SaddleError::Invalid(
                "shooting needs a checkpoint interval, two blocks and a shot per block".into(),
            ));
        }
        if let Some(masses) = &self.masses
            && (masses.len() != n || !masses.iter().all(|m| m.is_finite() && *m > 0.0))
        {
            return Err(SaddleError::Invalid(
                "shooting needs one positive mass per coordinate".into(),
            ));
        }
        if let EndState::Quench(basins) = &self.end_state {
            basins.quench.check()?;
            if basins
                .reactants
                .iter()
                .chain(&basins.products)
                .any(|m| m.len() != n)
            {
                return Err(SaddleError::Shape(
                    "the reference minima differ in length from the plane".into(),
                ));
            }
        }
        Ok(())
    }
}

/// `κ` from shooting, with `κ(t)` and the counts behind it.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct TransmissionCoefficient {
    /// `κ` at the end of the halves.
    pub kappa: f64,
    /// Block standard error of [`Self::kappa`].
    pub error: f64,
    /// Times of the checkpoints.
    pub time: Vec<f64>,
    /// `κ(t)` at those times.
    pub kappa_of_t: Vec<f64>,
    /// Assigned final-time score of each shot, in shot order.
    ///
    /// These are the observations averaged into [`Self::kappa`]. They allow
    /// callers to inspect the block-size plateau without repeating the
    /// trajectories.
    pub scores: Vec<f64>,
    /// Shots whose halves both reached a state at the end, the ones in
    /// [`Self::kappa`] and [`Self::scores`].
    pub shots: usize,
    /// Shots left out because a half ended in no state: the quench did
    /// not converge or found neither state.
    pub unassigned: usize,
    /// Shots from the reactant into the product.
    pub reactive: usize,
    /// Backward halves that crossed the plane again.
    pub recrossed: usize,
    /// Surface evaluations, quenches included.
    pub evaluations: usize,
}

impl TransmissionEstimate for TransmissionCoefficient {
    fn kappa(&self) -> (f64, f64) {
        (self.kappa, self.error)
    }
}

/// Engine boundary for unconstrained shooting from hyperplane samples.
///
/// Implementations own velocity generation, unconstrained propagation,
/// deterministic seed mapping and basin classification. The operation is
/// separate from [`crate::HyperplaneSampler`] because constrained sampling
/// cannot supply a recrossing correction.
pub trait HyperplaneDynamics {
    /// Shoot trajectories from `starts` and estimate the transmission
    /// coefficient of `plane`.
    fn shoot_hyperplane(
        &mut self,
        plane: &Hyperplane,
        starts: &[Array1<f64>],
        config: &TransmissionConfig,
    ) -> Result<TransmissionCoefficient, SaddleError>;
}

/// Analytic [`PointSurface`] reference implementation of
/// [`HyperplaneDynamics`].
pub struct AnalyticSurfaceDynamics<'a, S> {
    surface: &'a S,
}

impl<'a, S: PointSurface> AnalyticSurfaceDynamics<'a, S> {
    /// Bind unconstrained shooting to `surface`.
    pub fn new(surface: &'a S) -> Self {
        Self { surface }
    }
}

impl<S: PointSurface> HyperplaneDynamics for AnalyticSurfaceDynamics<'_, S> {
    fn shoot_hyperplane(
        &mut self,
        plane: &Hyperplane,
        starts: &[Array1<f64>],
        config: &TransmissionConfig,
    ) -> Result<TransmissionCoefficient, SaddleError> {
        analytic_transmission_coefficient(self.surface, plane, starts, config)
    }
}

/// Shoot from `starts`, configurations of `plane`'s ensemble, and return
/// the transmission coefficient of the plane. The starts are projected
/// onto the plane; its normal points to the product.
pub fn transmission_coefficient<D: HyperplaneDynamics>(
    dynamics: &mut D,
    plane: &Hyperplane,
    starts: &[Array1<f64>],
    config: &TransmissionConfig,
) -> Result<TransmissionCoefficient, SaddleError> {
    dynamics.shoot_hyperplane(plane, starts, config)
}

/// Analytic-surface compatibility wrapper for [`transmission_coefficient`].
#[deprecated(
    since = "0.5.0",
    note = "construct AnalyticSurfaceDynamics and call transmission_coefficient"
)]
pub fn transmission_coefficient_on_surface<S: PointSurface>(
    surface: &S,
    plane: &Hyperplane,
    starts: &[Array1<f64>],
    config: &TransmissionConfig,
) -> Result<TransmissionCoefficient, SaddleError> {
    let mut dynamics = AnalyticSurfaceDynamics::new(surface);
    transmission_coefficient(&mut dynamics, plane, starts, config)
}

fn analytic_transmission_coefficient<S: PointSurface>(
    surface: &S,
    plane: &Hyperplane,
    starts: &[Array1<f64>],
    config: &TransmissionConfig,
) -> Result<TransmissionCoefficient, SaddleError> {
    let n = plane.dim();
    config.check(n)?;
    if starts.is_empty() {
        return Err(SaddleError::Invalid(
            "shooting needs at least one start on the plane".into(),
        ));
    }
    if starts.iter().any(|s| s.len() != n) {
        return Err(SaddleError::Shape(
            "a start differs in length from the plane".into(),
        ));
    }
    let root = match &config.masses {
        Some(masses) => Array1::from_iter(masses.iter().map(|m| m.sqrt())),
        None => Array1::ones(n),
    };
    let normal = plane.normal().to_owned();
    let mut flux_normal = &normal / &root;
    flux_normal /= flux_normal.dot(&flux_normal).sqrt();
    let acceptance = match &config.end_state {
        EndState::Side => 0.0,
        EndState::Quench(basins) => basins.acceptance()?,
    };
    let mut checkpoints: Vec<usize> = (1..=config.steps / config.check_every)
        .map(|k| k * config.check_every)
        .collect();
    if checkpoints.last() != Some(&config.steps) {
        checkpoints.push(config.steps);
    }
    let shooter = Shooter {
        surface,
        config,
        point: plane.point().to_owned(),
        normal,
        inverse_mass: root.mapv(|r| 1.0 / (r * r)),
        root,
        flux_normal,
        acceptance,
        checkpoints: &checkpoints,
    };
    let shots: Vec<Shot> = (0..config.shots)
        .into_par_iter()
        .map(|shot| {
            let start = plane.project(starts[shot * starts.len() / config.shots].view());
            shooter.shot(&start, config.seed.wrapping_add(shot as u64))
        })
        .collect::<Result<_, _>>()?;

    let last = checkpoints.len() - 1;
    let mut kappa_of_t = Vec::with_capacity(checkpoints.len());
    for k in 0..checkpoints.len() {
        let values: Vec<f64> = shots.iter().filter_map(|s| s.values[k]).collect();
        kappa_of_t.push(if values.is_empty() {
            f64::NAN
        } else {
            values.iter().sum::<f64>() / values.len() as f64
        });
    }
    let finals: Vec<f64> = shots.iter().filter_map(|s| s.values[last]).collect();
    if finals.len() < config.blocks {
        return Err(SaddleError::Invalid(
            "fewer shots reached a state than there are blocks".into(),
        ));
    }
    let assigned = finals.len();
    let reactive = finals.iter().filter(|value| **value > 0.5).count();
    let (_, error, _, _) = block_mean(&finals, config.blocks);
    Ok(TransmissionCoefficient {
        kappa: kappa_of_t[last],
        error,
        time: checkpoints
            .iter()
            .map(|&s| s as f64 * config.time_step)
            .collect(),
        kappa_of_t,
        shots: assigned,
        scores: finals,
        unassigned: shots.len() - assigned,
        reactive,
        recrossed: shots.iter().filter(|s| s.recrossed).count(),
        evaluations: shots.iter().map(|s| s.evaluations).sum(),
    })
}

/// One shot: its score at every checkpoint, `None` where a half had no
/// state.
struct Shot {
    values: Vec<Option<f64>>,
    recrossed: bool,
    evaluations: usize,
}

/// One half: its state at every checkpoint, and the first checkpoint at
/// or after a crossing of the plane back toward the product.
struct Half {
    states: Vec<Option<Endpoint>>,
    recrossed: Option<usize>,
    evaluations: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Watch {
    Forward,
    Backward,
}

struct Shooter<'a, S> {
    surface: &'a S,
    config: &'a TransmissionConfig,
    point: Array1<f64>,
    normal: Array1<f64>,
    /// `√m` of every coordinate.
    root: Array1<f64>,
    inverse_mass: Array1<f64>,
    /// `M^{-1/2} ν`, normalized: the flux direction in the mass-weighted
    /// frame.
    flux_normal: Array1<f64>,
    acceptance: f64,
    checkpoints: &'a [usize],
}

impl<S: PointSurface> Shooter<'_, S> {
    fn shot(&self, start: &Array1<f64>, seed: u64) -> Result<Shot, SaddleError> {
        let config = self.config;
        let mut rng = Rng::new(seed);
        let held = if config.hold_excluded {
            let modes = self.surface.excluded_modes(start.view())?;
            if modes.nrows() > 0 && modes.ncols() != start.len() {
                return Err(SaddleError::Shape(
                    "excluded modes differ in length from the plane".into(),
                ));
            }
            let mut weighted = Array2::zeros((modes.nrows(), start.len()));
            for (mut row, mode) in weighted.rows_mut().into_iter().zip(modes.rows()) {
                row.assign(&(&mode * &self.root));
            }
            orthonormal_rows(weighted.view(), start.len())
        } else {
            Vec::new()
        };
        let mut drawn: Array1<f64> = (0..start.len()).map(|_| rng.normal()).collect();
        project(&mut drawn, &held);
        let along = drawn.dot(&self.flux_normal);
        drawn.scaled_add(-along, &self.flux_normal);
        let mut free = self.flux_normal.clone();
        project(&mut free, &held);
        let free_norm = free.dot(&free).sqrt();
        if free_norm < 1.0e-12 {
            return Err(SaddleError::Invalid(
                "the plane normal lies in the held modes".into(),
            ));
        }
        free /= free_norm;
        let thermal = config.thermal.sqrt();
        let rayleigh = (-2.0 * (1.0 - rng.uniform()).ln()).sqrt();
        let mut velocity = &drawn * thermal;
        velocity.scaled_add(rayleigh * thermal, &free);
        velocity /= &self.root;

        let backward = self.half(start, -&velocity, &held, &mut rng, Watch::Backward)?;
        let forward = self.half(start, velocity, &held, &mut rng, Watch::Forward)?;
        let product =
            |state: Option<Endpoint>| state.map(|s| f64::from(u8::from(s == Endpoint::Product)));
        let values = (0..self.checkpoints.len())
            .map(|k| match config.counting {
                Counting::Final => Some(product(forward.states[k])? - product(backward.states[k])?),
                Counting::FirstCrossing => {
                    if backward.recrossed.is_some_and(|r| r <= k) {
                        return Some(0.0);
                    }
                    let past = backward.states[k]?;
                    let future = forward.states[k]?;
                    Some(f64::from(u8::from(
                        past == Endpoint::Reactant && future == Endpoint::Product,
                    )))
                }
            })
            .collect();
        Ok(Shot {
            values,
            recrossed: backward.recrossed.is_some(),
            evaluations: backward.evaluations + forward.evaluations,
        })
    }

    fn side(&self, x: &Array1<f64>, scratch: &mut Array1<f64>) -> f64 {
        Zip::from(&mut *scratch)
            .and(x)
            .and(&self.point)
            .for_each(|d, &x, &p| *d = x - p);
        scratch.dot(&self.normal)
    }

    fn classify(
        &self,
        x: &Array1<f64>,
        scratch: &mut Array1<f64>,
    ) -> Result<(Option<Endpoint>, usize), SaddleError> {
        match &self.config.end_state {
            EndState::Side => {
                let xi = self.side(x, scratch);
                Ok((
                    if xi < 0.0 {
                        Some(Endpoint::Reactant)
                    } else if xi > 0.0 {
                        Some(Endpoint::Product)
                    } else {
                        None
                    },
                    0,
                ))
            }
            EndState::Quench(basins) => {
                basins.assign_within(self.surface, x.view(), self.acceptance)
            }
        }
    }

    /// Integrate one half from `start` with `velocity`. A backward half
    /// records its first return to the product side of the plane, which
    /// ends it when only first crossings count.
    fn half(
        &self,
        start: &Array1<f64>,
        mut velocity: Array1<f64>,
        held: &[Array1<f64>],
        rng: &mut Rng,
        watch: Watch,
    ) -> Result<Half, SaddleError> {
        let config = self.config;
        let h = config.time_step;
        let decay = (-config.friction * h).exp();
        let kick = (config.thermal * (1.0 - decay * decay)).sqrt();
        let mut x = start.clone();
        let mut scratch = Array1::zeros(x.len());
        let (_, mut gradient) = self.surface.eval(x.view())?;
        let mut evaluations = 1;
        let mut states = vec![None; self.checkpoints.len()];
        let mut recrossed = None;
        let mut next = 0;
        let mut run: (Option<Endpoint>, usize) = (None, 0);
        let kick_half = |velocity: &mut Array1<f64>, gradient: &Array1<f64>| {
            Zip::from(velocity)
                .and(gradient)
                .and(&self.inverse_mass)
                .for_each(|v, &g, &w| *v += -0.5 * h * (g * w));
        };
        for step in 1..=config.steps {
            kick_half(&mut velocity, &gradient);
            x.scaled_add(0.5 * h, &velocity);
            if config.friction > 0.0 {
                scratch.iter_mut().for_each(|n| *n = rng.normal());
                project(&mut scratch, held);
                Zip::from(&mut velocity)
                    .and(&scratch)
                    .and(&self.root)
                    .for_each(|v, &n, &r| *v = *v * decay + kick * (n / r));
            }
            x.scaled_add(0.5 * h, &velocity);
            let (energy, g) = self.surface.eval(x.view())?;
            evaluations += 1;
            if !energy.is_finite() || !g.iter().all(|v| v.is_finite()) {
                return Err(SaddleError::NonFinite("shooting gradient"));
            }
            gradient = g;
            kick_half(&mut velocity, &gradient);
            if watch == Watch::Backward && recrossed.is_none() && self.side(&x, &mut scratch) > 0.0
            {
                recrossed = Some(next);
                if config.counting == Counting::FirstCrossing {
                    break;
                }
            }
            if step == self.checkpoints[next] {
                let (state, cost) = self.classify(&x, &mut scratch)?;
                evaluations += cost;
                states[next] = state;
                run = match (state, run.0) {
                    (Some(s), Some(r)) if s == r => (state, run.1 + 1),
                    (Some(_), _) => (state, 1),
                    (None, _) => (None, 0),
                };
                next += 1;
                if config.commit > 0 && run.1 >= config.commit {
                    for slot in &mut states[next..] {
                        *slot = state;
                    }
                    break;
                }
                if next == self.checkpoints.len() {
                    break;
                }
            }
        }
        Ok(Half {
            states,
            recrossed,
            evaluations,
        })
    }
}

fn project(v: &mut Array1<f64>, held: &[Array1<f64>]) {
    for direction in held {
        let c = direction.dot(v);
        v.scaled_add(-c, direction);
    }
}

/// The minimum-cost perfect matching of the square, row-major `cost` of
/// order `n` by the Hungarian method with potentials, `O(n³)`: the total
/// cost and the column of every row.
fn assignment(cost: &[f64], n: usize) -> (f64, Vec<usize>) {
    let mut u = vec![0.0; n + 1];
    let mut v = vec![0.0; n + 1];
    let mut owner = vec![0usize; n + 1];
    let mut way = vec![0usize; n + 1];
    for row in 1..=n {
        owner[0] = row;
        let mut column = 0;
        let mut slack = vec![f64::INFINITY; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[column] = true;
            let i = owner[column];
            let mut delta = f64::INFINITY;
            let mut next = 0;
            for j in 1..=n {
                if !used[j] {
                    let reduced = cost[(i - 1) * n + (j - 1)] - u[i] - v[j];
                    if reduced < slack[j] {
                        slack[j] = reduced;
                        way[j] = column;
                    }
                    if slack[j] < delta {
                        delta = slack[j];
                        next = j;
                    }
                }
            }
            for j in 0..=n {
                if used[j] {
                    u[owner[j]] += delta;
                    v[j] -= delta;
                } else {
                    slack[j] -= delta;
                }
            }
            column = next;
            if owner[column] == 0 {
                break;
            }
        }
        loop {
            let previous = way[column];
            owner[column] = owner[previous];
            column = previous;
            if column == 0 {
                break;
            }
        }
    }
    let mut columns = vec![0; n];
    for j in 1..=n {
        if owner[j] > 0 {
            columns[owner[j] - 1] = j - 1;
        }
    }
    let total = (0..n).map(|i| cost[i * n + columns[i]]).sum();
    (total, columns)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn permutations(n: usize) -> Vec<Vec<usize>> {
        if n == 0 {
            return vec![vec![]];
        }
        let mut out = Vec::new();
        for p in permutations(n - 1) {
            for slot in 0..n {
                let mut q = p.clone();
                q.insert(slot, n - 1);
                out.push(q);
            }
        }
        out
    }

    #[test]
    fn hungarian_matches_every_permutation() {
        let mut rng = Rng::new(4);
        for n in 1..=6 {
            for _ in 0..20 {
                let cost: Vec<f64> = (0..n * n).map(|_| rng.uniform() * 10.0).collect();
                let (total, columns) = assignment(&cost, n);
                let best = permutations(n)
                    .iter()
                    .map(|p| (0..n).map(|i| cost[i * n + p[i]]).sum::<f64>())
                    .fold(f64::INFINITY, f64::min);
                assert!((total - best).abs() < 1e-12, "{total} against {best}");
                let mut seen = columns.clone();
                seen.sort_unstable();
                assert_eq!(seen, (0..n).collect::<Vec<_>>());
            }
        }
    }

    /// Exchanging like atoms costs nothing; moving one atom of a species
    /// with `N_t` atoms by `d` costs `d / N_t`; unlike atoms are never
    /// exchanged; and a move across the periodic boundary is its minimum
    /// image.
    #[test]
    fn permutation_distance_of_simple_moves() {
        let a = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 3.0, 3.0, 3.0];
        let swapped = array![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 3.0, 3.0, 3.0];
        let same = PermutationDistance::new(vec![0, 0, 0, 1]);
        assert!(same.distance(a.view(), swapped.view()).unwrap() < 1e-15);
        let unlike = PermutationDistance::new(vec![0, 1, 0, 1]);
        let d = unlike.distance(a.view(), swapped.view()).unwrap();
        assert!((d - 0.5).abs() < 1e-14, "{d}");
        let mut moved = a.clone();
        moved[7] += 0.6;
        let d = same.distance(a.view(), moved.view()).unwrap();
        assert!((d - 0.6 / 3.0).abs() < 1e-14, "{d}");
        let chosen = same.clone().with_atoms(vec![2, 3]);
        assert!((chosen.distance(a.view(), moved.view()).unwrap() - 0.6).abs() < 1e-14);
        let cell = Cell::ortho(4.0, 4.0, 4.0).unwrap();
        let mut wrapped = a.clone();
        wrapped[9] = -0.9;
        let periodic = PermutationDistance::new(vec![0, 0, 0, 1]).with_cell(cell);
        let d = periodic.distance(a.view(), wrapped.view()).unwrap();
        assert!((d - 0.1).abs() < 1e-12, "{d}");
        let masses: Vec<f64> = vec![4.0, 4.0, 4.0, 9.0];
        let weighted: Array1<f64> = a
            .iter()
            .enumerate()
            .map(|(i, x)| x * masses[i / 3].sqrt())
            .collect();
        let moved_weighted: Array1<f64> = moved
            .iter()
            .enumerate()
            .map(|(i, x)| x * masses[i / 3].sqrt())
            .collect();
        let massive = same.with_masses(masses);
        let d = massive
            .distance(weighted.view(), moved_weighted.view())
            .unwrap();
        assert!((d - 0.2).abs() < 1e-14, "{d}");
    }

    /// A product minimum with two like atoms relabelled is still the
    /// product under the permutation-invariant distance and is in neither
    /// state under the Euclidean one.
    #[test]
    fn relabelled_minimum_keeps_its_state() {
        let reactant = array![0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0];
        let product = array![0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 1.0, 0.0];
        let relabelled = array![2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0];
        let permuted = Basins::new(vec![reactant.clone()], vec![product.clone()])
            .with_distance(PermutationDistance::new(vec![0, 0, 1]).into());
        assert_eq!(
            permuted.classify(relabelled.view()).unwrap(),
            Some(Endpoint::Product)
        );
        let euclidean = Basins::new(vec![reactant], vec![product]);
        assert_eq!(euclidean.classify(relabelled.view()).unwrap(), None);
    }

    /// `(a² - 1)² + b² + ab/2`: minima at `a = ±(1 + 1/32)^{1/2}`,
    /// `b = -a/4`, and a saddle at the origin.
    struct TiltedWell;

    impl PointSurface for TiltedWell {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let (a, b) = (x[0], x[1]);
            Ok((
                (a * a - 1.0).powi(2) + b * b + 0.5 * a * b,
                array![4.0 * a * (a * a - 1.0) + 0.5 * b, 2.0 * b + 0.5 * a],
            ))
        }
    }

    /// Quenches end in the basin of their start. Near the saddle the
    /// separatrix runs along the stable mode, `a ≈ 0.083 b`, so a point on
    /// the reactant side of the plane `a = 0` at `b = -1` but past
    /// `a = -0.083` goes to the product. The saddle itself is a stationary
    /// point in neither state.
    #[test]
    fn quench_finds_the_basin() {
        let a = (1.0_f64 + 1.0 / 32.0).sqrt();
        let basins = Basins::new(vec![array![-a, a / 4.0]], vec![array![a, -a / 4.0]])
            .with_quench(QuenchConfig::default().with_force_tol(1.0e-8));
        for (start, expected) in [
            ([-0.4, 0.9], Some(Endpoint::Reactant)),
            ([0.5, 2.0], Some(Endpoint::Product)),
            ([-0.02, -1.0], Some(Endpoint::Product)),
            ([0.0, 0.0], None),
        ] {
            let (state, evaluations) = basins
                .assign(&TiltedWell, array![start[0], start[1]].view())
                .unwrap();
            assert_eq!(state, expected, "{start:?}");
            assert!(evaluations > 0);
        }
    }

    #[test]
    fn a_basin_escape_supplies_a_hard_reflection_normal() {
        let a = (1.0_f64 + 1.0 / 32.0).sqrt();
        let basins = Basins::new(vec![array![-a, a / 4.0]], vec![array![a, -a / 4.0]])
            .with_quench(QuenchConfig::default().with_force_tol(1.0e-8));
        let region = QuenchedTransitionRegion::new(Arc::new(TiltedWell), basins);
        let escaped = array![-0.02, -1.0];
        let violation = region
            .violation(escaped.view())
            .unwrap()
            .expect("the quench reaches a basin");
        let plane = Hyperplane::new(escaped, array![0.0, 1.0]).unwrap();
        let mut velocity = array![0.7, 0.0];
        let speed = velocity.dot(&velocity);
        crate::reflect_velocity(&mut velocity, violation.normal.view(), &plane).unwrap();
        assert!(velocity[0] < 0.0);
        assert!(velocity[1].abs() <= 16.0 * f64::EPSILON);
        assert!((velocity.dot(&velocity) - speed).abs() <= 64.0 * f64::EPSILON * speed);
    }

    /// Every start quenches to the product at `(1, 0)`. The query sits next
    /// to a second reactant, so the nearest reactant of the query and of
    /// the quenched configuration disagree.
    struct ProductBowl;

    impl PointSurface for ProductBowl {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let (a, b) = (x[0] - 1.0, x[1]);
            Ok((0.5 * (a * a + b * b), array![a, b]))
        }
    }

    #[test]
    fn the_reflection_normal_uses_the_quenched_configuration() {
        let basins = Basins::new(
            vec![array![-1.0, 0.0], array![1.0, 3.0]],
            vec![array![1.0, 0.0]],
        )
        .with_quench(QuenchConfig::default().with_force_tol(1.0e-8));
        let region = QuenchedTransitionRegion::new(Arc::new(ProductBowl), basins);
        let violation = region
            .violation(array![0.2, 2.9].view())
            .unwrap()
            .expect("the quench reaches the product");
        assert!(violation.normal[0] > 0.5, "{:?}", violation.normal);
    }
}
