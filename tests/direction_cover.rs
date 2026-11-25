//! Signed per-atom direction cover and arm picker.
//!
//! The covering radius is checked against dense sampling of the sphere,
//! the active mask against the seeds it produces, the picker on
//! Müller-Brown (embedded as one atom with a stiff third coordinate)
//! against its two saddles out of the shallow minimum, and on a 13-atom
//! Lennard-Jones cluster against a bare displacement along the soft mode.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use ndarray::{Array1, Array2, ArrayView1, array};
use rgsaddle::{
    ArmOutcome, ArmPolicy, ArtConfig, CoverBudget, DirectionCover, DirectionCoverConfig,
    MinModeConfig, MinModeKind, MinModeSession, PointSurface, RelaxConfig, SaddleError,
    SaddleIndexCheck, covering_radius_s2, relax_to_tolerance, sphere_cover,
};

const OPEN: [[f64; 3]; 3] = [[60.0, 0.0, 0.0], [0.0, 60.0, 0.0], [0.0, 0.0, 60.0]];

struct Lj {
    calls: AtomicUsize,
}

impl Lj {
    fn new() -> Self {
        Self {
            calls: AtomicUsize::new(0),
        }
    }
}

impl PointSurface for Lj {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let n = x.len() / 3;
        let mut energy = 0.0;
        let mut gradient = Array1::zeros(x.len());
        for i in 0..n {
            for j in (i + 1)..n {
                let mut delta = [0.0; 3];
                let mut square = 0.0;
                for axis in 0..3 {
                    delta[axis] = x[3 * i + axis] - x[3 * j + axis];
                    square += delta[axis] * delta[axis];
                }
                if !(square > 0.0 && square.is_finite()) {
                    return Err(SaddleError::NonFinite("Lennard-Jones overlap"));
                }
                let inverse = 1.0 / square;
                let inverse3 = inverse * inverse * inverse;
                let inverse6 = inverse3 * inverse3;
                energy += 4.0 * (inverse6 - inverse3);
                let slope = 4.0 * (-6.0 * inverse6 * inverse + 3.0 * inverse3 * inverse);
                for axis in 0..3 {
                    let component = slope * 2.0 * delta[axis];
                    gradient[3 * i + axis] += component;
                    gradient[3 * j + axis] -= component;
                }
            }
        }
        Ok((energy, gradient))
    }

    fn excluded_modes(&self, x: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
        Ok(rigid_modes(x))
    }
}

fn rigid_modes(x: ArrayView1<f64>) -> Array2<f64> {
    let n = x.len() / 3;
    let mut com = [0.0; 3];
    for atom in 0..n {
        for axis in 0..3 {
            com[axis] += x[3 * atom + axis] / n as f64;
        }
    }
    let mut rows = Vec::new();
    for axis in 0..3 {
        let mut row = Array1::zeros(x.len());
        for atom in 0..n {
            row[3 * atom + axis] = 1.0;
        }
        rows.push(row);
    }
    for axis in 0..3 {
        let mut row = Array1::zeros(x.len());
        for atom in 0..n {
            let r = [
                x[3 * atom] - com[0],
                x[3 * atom + 1] - com[1],
                x[3 * atom + 2] - com[2],
            ];
            let v = match axis {
                0 => [0.0, -r[2], r[1]],
                1 => [r[2], 0.0, -r[0]],
                _ => [-r[1], r[0], 0.0],
            };
            for k in 0..3 {
                row[3 * atom + k] = v[k];
            }
        }
        rows.push(row);
    }
    let mut basis: Vec<Array1<f64>> = Vec::new();
    for mut row in rows {
        for previous in &basis {
            let overlap = row.dot(previous);
            row = &row - &(previous * overlap);
        }
        let norm = row.dot(&row).sqrt();
        if norm > 1e-10 {
            basis.push(row / norm);
        }
    }
    let mut out = Array2::zeros((basis.len(), x.len()));
    for (index, row) in basis.iter().enumerate() {
        out.row_mut(index).assign(row);
    }
    out
}

fn quench_config() -> RelaxConfig {
    RelaxConfig::default()
        .with_force_tol(1e-4)
        .with_max_steps(20_000)
        .with_max_atom_move(0.05)
        .with_initial_step(0.05)
}

fn relaxed(surface: &Lj, x: Array1<f64>) -> Array1<f64> {
    relax_to_tolerance(surface, x, OPEN, &quench_config())
        .expect("cluster relaxes")
        .positions
}

fn icosahedron() -> Array1<f64> {
    let phi = 0.5 * (1.0 + 5.0_f64.sqrt());
    let mut x = vec![0.0, 0.0, 0.0];
    let scale = 1.1 / (1.0 + phi * phi).sqrt();
    for (a, b) in [(1.0, phi), (-1.0, phi), (1.0, -phi), (-1.0, -phi)] {
        x.extend([0.0, a * scale, b * scale]);
        x.extend([a * scale, b * scale, 0.0]);
        x.extend([b * scale, 0.0, a * scale]);
    }
    relaxed(&Lj::new(), Array1::from(x))
}

/// A cluster of `n` atoms: points of a simple cubic grid nearest the
/// origin, relaxed. Any local minimum serves the scaling run.
fn grid_cluster(n: usize) -> Array1<f64> {
    let mut sites = Vec::new();
    for i in -4i32..=4 {
        for j in -4i32..=4 {
            for k in -4i32..=4 {
                let p = [i as f64 + 0.1, j as f64 + 0.2, k as f64 + 0.3];
                sites.push((p[0] * p[0] + p[1] * p[1] + p[2] * p[2], p));
            }
        }
    }
    sites.sort_by(|a, b| a.0.total_cmp(&b.0));
    let x: Vec<f64> = sites
        .iter()
        .take(n)
        .flat_map(|(_, p)| p.iter().map(|v| v * 1.12))
        .collect();
    relaxed(&Lj::new(), Array1::from(x))
}

fn lj_art() -> ArtConfig {
    let mode = MinModeConfig::default()
        .with_kind(MinModeKind::Lanczos)
        .with_saddle_check(None)
        .with_krylov_dim(30)
        .with_max_rotations(40)
        .with_force_tol(0.01)
        .with_max_move(0.08)
        .with_method(rgmin::Method::lbfgs_with_memory(10))
        .with_seed_curvature(true)
        .with_dr(0.001);
    let perpendicular = RelaxConfig::default()
        .with_force_tol(0.01)
        .with_max_steps(3000)
        .with_max_atom_move(0.08)
        .with_initial_step(0.04);
    let quench = RelaxConfig::default()
        .with_force_tol(0.001)
        .with_max_steps(6_000)
        .with_max_atom_move(0.08)
        .with_initial_step(0.05);
    ArtConfig::default()
        .with_mode(mode)
        .with_perpendicular(perpendicular)
        .with_quench(quench)
        .with_eigen_threshold(-0.3)
        .with_push_length(0.4)
        .with_push_step(0.15)
        .with_max_pushes(16)
        .with_max_climbs(80)
        .with_push_over(0.4)
        .with_basin_rmsd(0.04)
        .with_duplicate_rmsd(0.04)
        .with_saddle(SaddleIndexCheck::default().with_max_actions(400))
}

fn lj_cover_config() -> DirectionCoverConfig {
    DirectionCoverConfig::default()
        .with_art(lj_art())
        .with_kick_radius(1.35)
        .with_kick_width(0.2)
        .with_saddle_rmsd_tol(0.02)
        .with_minimum_rmsd_tol(0.02)
}

struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (u, v) = (self.next(), self.next());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

fn brute_force_radius(points: &[[f64; 3]], samples: usize) -> f64 {
    let mut rng = SplitMix(11);
    let mut best = 0.0_f64;
    for _ in 0..samples {
        let v = [rng.normal(), rng.normal(), rng.normal()];
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        let x = [v[0] / n, v[1] / n, v[2] / n];
        let nearest = points
            .iter()
            .map(|p| {
                (x[0] * p[0] + x[1] * p[1] + x[2] * p[2])
                    .clamp(-1.0, 1.0)
                    .acos()
            })
            .fold(f64::INFINITY, f64::min);
        best = best.max(nearest);
    }
    best
}

#[test]
fn covering_radius_shrinks_and_matches_brute_force() {
    let cover = sphere_cover(48, 400, 3);
    let mut previous = covering_radius_s2(&[]);
    for k in 1..=cover.len() {
        let radius = covering_radius_s2(&cover[..k]);
        assert!(
            radius <= previous + 1e-12,
            "radius rose from {previous} to {radius} at {k} directions"
        );
        previous = radius;
    }
    for k in [2, 3, 5, 12, 30, 48] {
        let exact = covering_radius_s2(&cover[..k]);
        let sampled = brute_force_radius(&cover[..k], 400_000);
        assert!(
            exact >= sampled - 1e-9,
            "{k}: exact {exact} below a sampled point {sampled}"
        );
        assert!(
            exact - sampled < 0.01,
            "{k}: exact {exact} far above dense sampling {sampled}"
        );
    }
}

#[test]
fn active_mask_confines_centres_and_kicks() {
    let minimum = icosahedron();
    let surface = Lj::new();
    let mut active = vec![false; 13];
    active[3] = true;
    active[7] = true;
    let bare = DirectionCover::new(
        &surface,
        minimum.clone(),
        Some(&active),
        5,
        lj_cover_config().with_kick_radius(0.0),
    )
    .unwrap();
    assert_eq!(bare.centres(), &[3, 7]);
    for &atom in bare.centres() {
        for point in 0..bare.cover().len() {
            let seed = bare
                .seed(rgsaddle::ArmId { atom, point })
                .expect("arm seed");
            for other in 0..13 {
                let norm: f64 = (0..3).map(|k| seed.seed[3 * other + k].powi(2)).sum();
                let mask: f64 = (0..3).map(|k| seed.mask[3 * other + k]).sum();
                if other == atom {
                    assert!((norm - 1.0).abs() < 1e-12);
                    assert_eq!(mask, 3.0);
                } else {
                    assert_eq!(norm, 0.0);
                    assert_eq!(mask, 0.0);
                }
            }
        }
    }
    assert!(bare.seed(rgsaddle::ArmId { atom: 0, point: 0 }).is_err());

    let radius = 1.35;
    let kicked = DirectionCover::new(
        &surface,
        minimum.clone(),
        Some(&active),
        5,
        lj_cover_config(),
    )
    .unwrap();
    let seed = kicked.seed(rgsaddle::ArmId { atom: 3, point: 4 }).unwrap();
    let mut neighbours = 0;
    for other in 0..13 {
        let d: f64 = (0..3)
            .map(|k| (minimum[3 * other + k] - minimum[3 * 3 + k]).powi(2))
            .sum::<f64>()
            .sqrt();
        let moved = (0..3).any(|k| seed.seed[3 * other + k] != 0.0);
        if d > radius {
            assert!(!moved, "atom {other} at {d} kicked");
        } else if other != 3 {
            assert!(moved, "neighbour {other} at {d} not kicked");
            neighbours += 1;
        }
    }
    assert!(neighbours >= 4, "{neighbours} neighbours");

    let wrong = DirectionCover::new(
        &surface,
        minimum.clone(),
        Some(&[true; 4]),
        5,
        lj_cover_config(),
    );
    assert!(matches!(wrong, Err(SaddleError::Shape(_))));
    let empty = DirectionCover::new(
        &surface,
        minimum.clone(),
        Some(&[false; 13]),
        5,
        lj_cover_config(),
    );
    assert!(matches!(empty, Err(SaddleError::Invalid(_))));
    let bad = DirectionCover::new(
        &surface,
        minimum.clone(),
        None,
        5,
        lj_cover_config().with_cover_points(2),
    );
    assert!(matches!(bad, Err(SaddleError::Invalid(_))));
    let ei = DirectionCover::new(
        &surface,
        minimum,
        None,
        5,
        lj_cover_config().with_policy(ArmPolicy::ExpectedImprovement),
    )
    .unwrap();
    assert!(matches!(ei.suggest(), Err(SaddleError::Invalid(_))));
    let mut unbounded =
        DirectionCover::new(&surface, icosahedron(), None, 5, lj_cover_config()).unwrap();
    assert!(matches!(
        unbounded.run(&surface, CoverBudget::default()),
        Err(SaddleError::Invalid(_))
    ));
}

struct Bowl;

impl PointSurface for Bowl {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        Ok((0.5 * x.dot(&x), x.to_owned()))
    }
}

#[test]
fn gap_policy_tries_every_centre_before_a_second_direction() {
    let mut cover = DirectionCover::new(
        &Bowl,
        Array1::zeros(12),
        None,
        9,
        DirectionCoverConfig::default(),
    )
    .unwrap();
    let mut seen = Vec::new();
    for _ in 0..4 {
        let pick = cover.suggest().unwrap().unwrap();
        assert!(!seen.contains(&pick.arm.atom), "{pick:?} after {seen:?}");
        seen.push(pick.arm.atom);
        cover
            .record(rgsaddle::ArmTrial::returned(pick.arm, 1))
            .unwrap();
    }
    let report = cover.coverage().unwrap();
    assert_eq!(report.subspace_dim, 12);
    assert!(report.atoms.iter().all(|atom| atom.tried == 1));
    assert!(report.covering_radius > 3.0);
    let again = report.records[0].arm;
    assert!(matches!(
        cover.record(rgsaddle::ArmTrial::returned(again, 1)),
        Err(SaddleError::Invalid(_))
    ));
}

fn bowl_connected(cover: &mut DirectionCover, point: usize, product: f64) {
    let arm = rgsaddle::ArmId { atom: 0, point };
    cover
        .record(rgsaddle::ArmTrial::connected(
            arm,
            array![0.5 * product, 0.0, 0.0],
            0.125 * product * product,
            array![product, 0.0, 0.0],
            0.5 * product * product,
            1,
        ))
        .unwrap();
}

/// Largest angle from an untried point to its nearest tried one.
fn widest_gap(cover: &DirectionCover, tried: &[usize]) -> f64 {
    let points = cover.cover();
    (0..points.len())
        .filter(|point| !tried.contains(point))
        .map(|point| {
            tried
                .iter()
                .map(|&other| {
                    let u = points[point];
                    let v = points[other];
                    (u[0] * v[0] + u[1] * v[1] + u[2] * v[2])
                        .clamp(-1.0, 1.0)
                        .acos()
                })
                .fold(std::f64::consts::PI, f64::min)
        })
        .fold(0.0, f64::max)
}

#[test]
fn gap_counts_the_start_beside_one_product() {
    let mut cover = DirectionCover::new(
        &Bowl,
        Array1::zeros(3),
        None,
        9,
        DirectionCoverConfig::default(),
    )
    .unwrap();
    bowl_connected(&mut cover, 0, 1.0);
    let pick = cover.suggest().unwrap().unwrap();
    let expected = 2.0 * widest_gap(&cover, &[0]);
    assert_eq!(pick.policy, ArmPolicy::Gap);
    assert!(
        (pick.score - expected).abs() < 1e-12,
        "score {} against start and one product {expected}",
        pick.score
    );
}

#[test]
fn gap_counts_the_start_beside_two_products() {
    let mut cover = DirectionCover::new(
        &Bowl,
        Array1::zeros(3),
        None,
        9,
        DirectionCoverConfig::default(),
    )
    .unwrap();
    bowl_connected(&mut cover, 0, 1.0);
    bowl_connected(&mut cover, 1, 2.0);
    let report = cover.coverage().unwrap();
    assert_eq!(report.products.len(), 2);
    let pick = report.next.unwrap();
    let expected = 3.0 * widest_gap(&cover, &[0, 1]);
    assert!(
        (pick.score - expected).abs() < 1e-12,
        "score {} against start and two products {expected}",
        pick.score
    );
}

#[test]
fn coverage_states_count_the_start() {
    let mut cover = DirectionCover::new(
        &Bowl,
        Array1::zeros(3),
        None,
        9,
        DirectionCoverConfig::default(),
    )
    .unwrap();
    let states = |cover: &DirectionCover| cover.coverage().unwrap().atoms[0].states;
    assert_eq!(states(&cover), 1, "nothing tried");
    let failed = rgsaddle::ArmId { atom: 0, point: 2 };
    cover
        .record(rgsaddle::ArmTrial::failed(
            failed,
            rgsaddle::ArmFailure::NotIndexOne,
            1,
        ))
        .unwrap();
    assert_eq!(states(&cover), 1, "a failed arm adds no state");
    let returned = rgsaddle::ArmId { atom: 0, point: 3 };
    cover
        .record(rgsaddle::ArmTrial::returned(returned, 1))
        .unwrap();
    assert_eq!(states(&cover), 1, "a return is the start");
    bowl_connected(&mut cover, 0, 1.0);
    assert_eq!(states(&cover), 2, "start and one product");
    bowl_connected(&mut cover, 1, 2.0);
    assert_eq!(states(&cover), 3, "start and two products");
}

/// Müller-Brown in (x, y) plus a stiff harmonic z: one atom.
struct MullerBrown;

const MB_A: [f64; 4] = [-200.0, -100.0, -170.0, 15.0];
const MB_AX: [f64; 4] = [-1.0, -1.0, -6.5, 0.7];
const MB_B: [f64; 4] = [0.0, 0.0, 11.0, 0.6];
const MB_AY: [f64; 4] = [-10.0, -10.0, -6.5, 0.7];
const MB_X0: [f64; 4] = [1.0, 0.0, -0.5, -1.0];
const MB_Y0: [f64; 4] = [0.0, 0.5, 1.5, 1.0];
const MB_Z: f64 = 2000.0;

impl PointSurface for MullerBrown {
    fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (x, y, z) = (q[0], q[1], q[2]);
        let mut energy = 0.5 * MB_Z * z * z;
        let mut gx = 0.0;
        let mut gy = 0.0;
        for i in 0..4 {
            let dx = x - MB_X0[i];
            let dy = y - MB_Y0[i];
            let term =
                MB_A[i] * (MB_AX[i] * dx * dx + MB_B[i] * dx * dy + MB_AY[i] * dy * dy).exp();
            energy += term;
            gx += term * (2.0 * MB_AX[i] * dx + MB_B[i] * dy);
            gy += term * (MB_B[i] * dx + 2.0 * MB_AY[i] * dy);
        }
        Ok((energy, array![gx, gy, MB_Z * z]))
    }
}

#[test]
fn muller_brown_low_saddles_are_recovered() {
    let relax = RelaxConfig::default()
        .with_force_tol(1e-6)
        .with_max_steps(20_000)
        .with_max_atom_move(0.01)
        .with_initial_step(0.001);
    let minimum = relax_to_tolerance(&MullerBrown, array![-0.05, 0.47, 0.0], OPEN, &relax)
        .unwrap()
        .positions;
    let mode = MinModeConfig::default()
        .with_kind(MinModeKind::Lanczos)
        .with_saddle_check(None)
        .with_krylov_dim(3)
        .with_max_rotations(6)
        .with_force_tol(1e-3)
        .with_max_move(0.01)
        .with_dr(1e-5);
    let art = ArtConfig::default()
        .with_mode(mode)
        .with_perpendicular(
            RelaxConfig::default()
                .with_force_tol(1e-3)
                .with_max_steps(2_000)
                .with_max_atom_move(0.01)
                .with_initial_step(0.001),
        )
        .with_quench(relax.clone().with_force_tol(1e-4))
        .with_eigen_threshold(-1.0)
        .with_push_length(0.4)
        .with_push_step(0.15)
        .with_max_pushes(20)
        .with_max_climbs(400)
        .with_push_over(0.03)
        .with_basin_rmsd(0.02)
        .with_duplicate_rmsd(0.02)
        .with_saddle(SaddleIndexCheck::default().with_dr(1e-5));
    let config = DirectionCoverConfig::default()
        .with_art(art)
        .with_cover_points(64)
        .with_saddle_energy_tol(1e-2)
        .with_saddle_rmsd_tol(1e-2)
        .with_minimum_energy_tol(1e-2)
        .with_minimum_rmsd_tol(1e-2);
    let mut chooser = DirectionCover::new(&MullerBrown, minimum, None, 1, config).unwrap();
    let report = chooser
        .run(&MullerBrown, CoverBudget::default().with_max_arms(48))
        .unwrap();
    let mut lines = Vec::new();
    for record in &report.records {
        lines.push(format!(
            "{:?} {:?} barrier {:?} calls {}",
            record.direction, record.outcome, record.barrier, record.force_calls
        ));
    }
    let found = |x: f64, y: f64| {
        report
            .saddles
            .iter()
            .any(|s| (s.position[0] - x).abs() < 5e-3 && (s.position[1] - y).abs() < 5e-3)
    };
    assert!(
        found(0.212_487, 0.292_988),
        "lower saddle\n{}",
        lines.join("\n")
    );
    assert!(
        found(-0.822_002, 0.624_313),
        "upper saddle\n{}",
        lines.join("\n")
    );
    assert_eq!(report.subspace_dim, 3);
    assert_eq!(report.discovery.len(), report.records.len());
    assert!(report.covering_radius < std::f64::consts::PI);
    println!(
        "Muller-Brown: {} saddles in {} arms, {} force calls",
        report.saddles.len(),
        report.records.len(),
        report.force_calls
    );
}

fn returned_fraction(records: &[rgsaddle::ArmRecord]) -> f64 {
    let back = records
        .iter()
        .filter(|r| r.outcome == ArmOutcome::ReturnedToStart)
        .count();
    back as f64 / records.len() as f64
}

/// Arms that did not reach a minimum other than the start: the bare
/// displacement's returned fraction counts the same thing, since every
/// one of its arms is quenched.
fn stayed_fraction(records: &[rgsaddle::ArmRecord]) -> f64 {
    let left = records.iter().filter(|r| r.product.is_some()).count();
    1.0 - left as f64 / records.len() as f64
}

/// Displace along the lowest Lanczos mode, seeded from a random vector,
/// by `length` in either sign and quench: the bare soft-mode start.
fn soft_mode_returned(minimum: &Array1<f64>, arms: usize, length: f64) -> (f64, usize) {
    let surface = Lj::new();
    let mut rng = SplitMix(29);
    let mut back = 0;
    for arm in 0..arms {
        let seed = Array1::from_iter((0..minimum.len()).map(|_| rng.normal()));
        let mut session = MinModeSession::new(
            MinModeConfig::default()
                .with_kind(MinModeKind::Lanczos)
                .with_krylov_dim(24)
                .with_max_rotations(24),
            minimum.clone(),
            seed,
        )
        .unwrap();
        let mode = session.estimate_mode(&surface).unwrap().mode;
        let sign = if arm % 2 == 0 { 1.0 } else { -1.0 };
        let start = minimum + &(&mode * (sign * length));
        let landed = relax_to_tolerance(&surface, start, OPEN, &quench_config()).unwrap();
        let d = &landed.positions - minimum;
        if (d.dot(&d) / (d.len() / 3) as f64).sqrt() < 0.02 {
            back += 1;
        }
    }
    (
        back as f64 / arms as f64,
        surface.calls.load(Ordering::Relaxed),
    )
}

#[test]
fn lj13_cover_leaves_the_start_more_often_than_soft_mode_displacement() {
    let minimum = icosahedron();
    let surface = Lj::new();
    let arms = 16;
    let mut chooser =
        DirectionCover::new(&surface, minimum.clone(), None, 3, lj_cover_config()).unwrap();
    let report = chooser
        .run(&surface, CoverBudget::default().with_max_arms(arms))
        .unwrap();
    let returned = returned_fraction(&report.records);
    let stayed = stayed_fraction(&report.records);
    let (soft, soft_calls) = soft_mode_returned(&minimum, arms, lj_art().push_length);
    println!(
        "LJ13 over {arms} arms: cover returned to start {returned:.3}, cover did not leave the start {stayed:.3} ({} saddles, {} calls); soft-mode displacement returned to start {soft:.3} ({soft_calls} calls)",
        report.saddles.len(),
        report.force_calls,
    );
    for record in &report.records {
        println!(
            "  {:?} {:?} {:?}",
            record.arm, record.outcome, record.barrier
        );
    }
    assert!(!report.saddles.is_empty(), "no saddle out of LJ13");
    let left = report
        .records
        .iter()
        .filter(|r| r.product.is_some())
        .count();
    assert!(left >= 2, "{left} of {arms} arms reached another minimum");
    assert!(
        returned < soft,
        "returned {returned} against soft mode {soft}"
    );
    assert!(stayed < soft, "stayed {stayed} against soft mode {soft}");
}

fn run_in_pool(threads: usize) -> rgsaddle::CoverageReport {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    pool.install(|| {
        let surface = Lj::new();
        let mut chooser = DirectionCover::new(
            &surface,
            icosahedron(),
            None,
            17,
            lj_cover_config().with_batch(4),
        )
        .unwrap();
        chooser
            .run(&surface, CoverBudget::default().with_max_arms(8))
            .unwrap()
    })
}

#[test]
fn records_are_identical_at_one_and_four_threads() {
    let one = run_in_pool(1);
    let four = run_in_pool(4);
    assert_eq!(one.records.len(), 8);
    assert_eq!(one, four);
    for (a, b) in one.saddles.iter().zip(&four.saddles) {
        assert_eq!(a.energy.to_bits(), b.energy.to_bits());
    }
}

/// Time per arm and force calls per saddle against cluster size.
/// `cargo test --release --test direction_cover -- --ignored --nocapture`
#[test]
#[ignore]
fn scaling_with_cluster_size() {
    let fixture = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/lj38/start_0.xyz"
    );
    let text = std::fs::read_to_string(fixture).unwrap();
    let lj38: Vec<f64> = text
        .lines()
        .skip(2)
        .flat_map(|line| {
            line.split_whitespace()
                .skip(1)
                .take(3)
                .map(|v| v.parse::<f64>().unwrap())
                .collect::<Vec<_>>()
        })
        .collect();
    let systems = [
        ("LJ13", icosahedron()),
        ("LJ38", relaxed(&Lj::new(), Array1::from(lj38))),
        ("LJ75", grid_cluster(75)),
    ];
    let arms = 32;
    for (name, minimum) in systems {
        let surface = Lj::new();
        let mut chooser =
            DirectionCover::new(&surface, minimum.clone(), None, 3, lj_cover_config()).unwrap();
        let clock = Instant::now();
        let report = chooser
            .run(&surface, CoverBudget::default().with_max_arms(arms))
            .unwrap();
        let elapsed = clock.elapsed().as_secs_f64();
        let returned = returned_fraction(&report.records);
        let stayed = stayed_fraction(&report.records);
        let (soft, _) = soft_mode_returned(&minimum, arms, lj_art().push_length);
        let count = |want: fn(&ArmOutcome) -> bool| {
            report.records.iter().filter(|r| want(&r.outcome)).count()
        };
        let per_saddle = if report.saddles.is_empty() {
            "none found".to_string()
        } else {
            format!(
                "{:.0}",
                report.force_calls as f64 / report.saddles.len() as f64
            )
        };
        println!(
            "{name}: dim {} arms {} saddles {} draws {} calls {} calls/saddle {per_saddle} calls/arm {:.0} ms/arm {:.1} new {} duplicate {} returned {} failed {} nonconverged {} returned fraction {returned:.3} stayed fraction {stayed:.3} soft-mode returned {soft:.3}",
            minimum.len(),
            report.records.len(),
            report.saddles.len(),
            report.saddle_draws,
            report.force_calls,
            report.force_calls as f64 / report.records.len() as f64,
            1e3 * elapsed / report.records.len() as f64,
            count(|o| *o == ArmOutcome::NewSaddle),
            count(|o| matches!(o, ArmOutcome::DuplicateSaddle(_))),
            count(|o| *o == ArmOutcome::ReturnedToStart),
            count(|o| matches!(o, ArmOutcome::Failed(_))),
            count(|o| *o == ArmOutcome::NonConverged),
        );
    }
}

/// Quick linear-cost sanity comparison used by `NOTES.md`.
/// `cargo test +1.88.0 --release --test direction_cover quick_direction_cover_sanity -- --ignored --nocapture`
#[test]
#[ignore]
fn quick_direction_cover_sanity() {
    struct IndependentWells {
        calls: AtomicUsize,
    }

    impl PointSurface for IndependentWells {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            let mut energy = 0.0;
            let mut gradient = Array1::zeros(x.len());
            for atom in 0..x.len() / 3 {
                let a = x[3 * atom] - 2.0 * atom as f64;
                let b = x[3 * atom + 1];
                let c = x[3 * atom + 2];
                energy += 0.25 * (a * a - 1.0).powi(2) + 0.5 * (b * b + c * c);
                gradient[3 * atom] = a * a * a - a;
                gradient[3 * atom + 1] = b;
                gradient[3 * atom + 2] = c;
            }
            Ok((energy, gradient))
        }
    }

    fn config() -> DirectionCoverConfig {
        let mode = MinModeConfig::default()
            .with_kind(MinModeKind::Lanczos)
            .with_saddle_check(None)
            .with_krylov_dim(4)
            .with_max_rotations(4)
            .with_force_tol(0.02)
            .with_max_move(0.1);
        let relax = RelaxConfig::default()
            .with_force_tol(0.02)
            .with_max_steps(100)
            .with_max_atom_move(0.1)
            .with_initial_step(0.05);
        DirectionCoverConfig::default()
            .with_cover_points(16)
            .with_kick_radius(0.0)
            .with_art_max_steps(200)
            .with_saddle_rmsd_tol(0.005)
            .with_minimum_rmsd_tol(0.005)
            .with_art(
                ArtConfig::default()
                    .with_mode(mode)
                    .with_perpendicular(relax.clone())
                    .with_quench(relax.with_force_tol(0.001))
                    .with_push_length(0.8)
                    .with_push_step(0.3)
                    .with_max_pushes(12)
                    .with_max_climbs(25)
                    .with_push_over(0.25)
                    .with_basin_rmsd(0.005)
                    .with_duplicate_rmsd(0.005)
                    .with_saddle(SaddleIndexCheck::default().with_max_actions(20)),
            )
    }

    for atoms in [75, 500, 4_000] {
        let arms = 12;
        let mut minimum = Array1::zeros(3 * atoms);
        for atom in 0..atoms {
            minimum[3 * atom] = 2.0 * atom as f64 - 1.0;
        }
        let mut active = vec![false; atoms];
        active[0] = true;
        let surface = IndependentWells {
            calls: AtomicUsize::new(0),
        };
        let mut cover =
            DirectionCover::new(&surface, minimum.clone(), Some(&active), 19, config()).unwrap();
        surface.calls.store(0, Ordering::Relaxed);
        let started = Instant::now();
        let report = cover
            .run(&surface, CoverBudget::default().with_max_arms(arms))
            .unwrap();
        let cover_ms = 1e3 * started.elapsed().as_secs_f64() / report.records.len() as f64;
        let left = report
            .records
            .iter()
            .filter(|record| record.product.is_some())
            .count();
        let calls_per_saddle = if report.saddles.is_empty() {
            f64::NAN
        } else {
            report.force_calls as f64 / report.saddles.len() as f64
        };
        surface.calls.store(0, Ordering::Relaxed);
        let started = Instant::now();
        let (returned, _) = soft_mode_returned_independent(&surface, &minimum, arms, 0.8);
        let soft_calls = surface.calls.load(Ordering::Relaxed);
        let soft_ms = 1e3 * started.elapsed().as_secs_f64() / arms as f64;
        println!(
            "{atoms} cover: left {left}/{} saddles {} calls/saddle {calls_per_saddle:.0} ms/arm {cover_ms:.3}; soft: left {:.0}/{arms} saddles 0 calls/saddle -- ms/arm {soft_ms:.3} calls {soft_calls}",
            report.records.len(),
            report.saddles.len(),
            arms as f64 * (1.0 - returned),
        );
    }

    fn soft_mode_returned_independent(
        surface: &IndependentWells,
        minimum: &Array1<f64>,
        arms: usize,
        length: f64,
    ) -> (f64, usize) {
        let mut back = 0;
        for arm in 0..arms {
            let mut seed = Array1::zeros(minimum.len());
            seed[1 + arm % (minimum.len() - 1)] = 1.0;
            let mut session = MinModeSession::new(
                MinModeConfig::default()
                    .with_kind(MinModeKind::Lanczos)
                    .with_krylov_dim(4)
                    .with_max_rotations(4),
                minimum.clone(),
                seed,
            )
            .unwrap();
            let mode = session.estimate_mode(surface).unwrap().mode;
            let landed = relax_to_tolerance(
                surface,
                minimum + &(&mode * length),
                OPEN,
                &quench_config().with_max_steps(100),
            )
            .unwrap();
            let delta = &landed.positions - minimum;
            if (delta.dot(&delta) / (delta.len() / 3) as f64).sqrt() < 0.02 {
                back += 1;
            }
        }
        (
            back as f64 / arms as f64,
            surface.calls.load(Ordering::Relaxed),
        )
    }
}
