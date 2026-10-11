//! ART nouveau on Muller-Brown and the double-well atom used by the
//! minimum-mode and band tests.
//!
//! The double well is `(x^2 - 1)^2 + 2 y^2 + 2 z^2` at `(1, 0, 0)`.
//! The reaction coordinate is the stiffest mode there. An isotropic
//! kick has a negative x component about half the time and points
//! toward the saddle at the origin. The comparison counts success
//! among those seeds.
//!
//! ART, the dimer, and Lanczos min-mode following share one force-call
//! budget and the same displaced start.

use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use ndarray::{Array1, Array2, ArrayView1, array};
use rgsaddle::{
    ArtConfig, ArtDisplacement, ArtEvent, ArtOutcome, ArtSession, ArtStatus, ArtSupport,
    FiniteDifference, ForceGate, MinModeConfig, MinModeKind, MinModeSession, MinModeStatus,
    PointSurface, SaddleError, event_seed, run_events,
};

/// V = (x^2 - 1)^2 + 2 y^2 + 2 z^2 on every Cartesian atom.
struct DoubleWell;

impl PointSurface for DoubleWell {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        if x.is_empty() || !x.len().is_multiple_of(3) {
            return Err(SaddleError::Shape("double well expects triples".into()));
        }
        let mut energy = 0.0;
        let mut gradient = Array1::zeros(x.len());
        for atom in 0..x.len() / 3 {
            let (a, b, c) = (x[3 * atom], x[3 * atom + 1], x[3 * atom + 2]);
            energy += (a * a - 1.0).powi(2) + 2.0 * b * b + 2.0 * c * c;
            gradient[3 * atom] = 4.0 * a * (a * a - 1.0);
            gradient[3 * atom + 1] = 4.0 * b;
            gradient[3 * atom + 2] = 4.0 * c;
        }
        Ok((energy, gradient))
    }
}

fn double_well_index(x: ArrayView1<f64>) -> i32 {
    let mut negative = 0;
    for atom in 0..x.len() / 3 {
        let curvature = 12.0 * x[3 * atom] * x[3 * atom] - 4.0;
        if curvature < 0.0 {
            negative += 1;
        }
    }
    negative
}

struct Counted<S> {
    inner: S,
    calls: AtomicUsize,
}

impl<S: PointSurface> PointSurface for Counted<S> {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.inner.eval(x)
    }
}

const MB_A: [f64; 4] = [-200.0, -100.0, -170.0, 15.0];
const MB_AA: [f64; 4] = [-1.0, -1.0, -6.5, 0.7];
const MB_B: [f64; 4] = [0.0, 0.0, 11.0, 0.6];
const MB_C: [f64; 4] = [-10.0, -10.0, -6.5, 0.7];
const MB_X0: [f64; 4] = [1.0, 0.0, -0.5, -1.0];
const MB_Y0: [f64; 4] = [0.0, 0.5, 1.5, 1.0];

fn muller_brown(x: f64, y: f64) -> (f64, [f64; 2], [[f64; 2]; 2]) {
    let mut energy = 0.0;
    let mut gx = 0.0;
    let mut gy = 0.0;
    let mut hxx = 0.0;
    let mut hyy = 0.0;
    let mut hxy = 0.0;
    for i in 0..4 {
        let dx = x - MB_X0[i];
        let dy = y - MB_Y0[i];
        let exponent = MB_AA[i] * dx * dx + MB_B[i] * dx * dy + MB_C[i] * dy * dy;
        let weight = MB_A[i] * exponent.exp();
        let px = 2.0 * MB_AA[i] * dx + MB_B[i] * dy;
        let py = MB_B[i] * dx + 2.0 * MB_C[i] * dy;
        energy += weight;
        gx += weight * px;
        gy += weight * py;
        hxx += weight * (px * px + 2.0 * MB_AA[i]);
        hyy += weight * (py * py + 2.0 * MB_C[i]);
        hxy += weight * (px * py + MB_B[i]);
    }
    (energy, [gx, gy], [[hxx, hxy], [hxy, hyy]])
}

struct MullerBrown;

impl PointSurface for MullerBrown {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (energy, g, _) = muller_brown(x[0], x[1]);
        Ok((energy, array![g[0], g[1]]))
    }
}

fn muller_index(x: f64, y: f64) -> i32 {
    let (_, _, h) = muller_brown(x, y);
    let tr = h[0][0] + h[1][1];
    let det = h[0][0] * h[1][1] - h[0][1] * h[1][0];
    let disc = (tr * tr - 4.0 * det).max(0.0).sqrt();
    let low = 0.5 * (tr - disc);
    let high = 0.5 * (tr + disc);
    i32::from(low < 0.0) + i32::from(high < 0.0)
}

const MIN_A: [f64; 2] = [-0.558_223_634_633, 1.441_725_841_97];
const TS1: [f64; 2] = [-0.822_001_558_732, 0.624_304_440_872];

struct Boom;

impl PointSurface for Boom {
    fn eval(&self, _x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        Err(SaddleError::Surface("scf did not converge".into()))
    }
}

#[test]
fn new_displaces_without_a_surface_call_and_a_failure_stays_put() {
    let minimum = array![1.0, 0.0, 0.0];
    let mut session = ArtSession::new(
        ArtConfig::default(),
        minimum,
        ArtDisplacement::Targeted {
            direction: array![-1.0, 0.0, 0.0],
            support: ArtSupport::Global,
            amplitude: 0.2,
        },
    )
    .unwrap();
    assert!((session.position()[0] - 0.8).abs() < 1e-12);
    assert_eq!(session.evaluations(), 0);
    let start = session.position().to_owned();
    let err = session.step(&Boom).unwrap_err();
    assert!(matches!(err, SaddleError::Surface(ref message) if message.contains("scf")));
    assert_eq!(session.position(), start.view());
}

#[test]
fn local_targeted_displacement_leaves_the_other_atom() {
    let minimum = array![1.0, 0.0, 0.0, 1.0, 3.0, 0.0];
    let support = ArtSupport::local_atoms(minimum.view(), 0, 0.0).unwrap();
    let ArtSupport::Local(indices) = &support else {
        panic!("expected a local support");
    };
    assert_eq!(indices, &vec![0, 1, 2]);
    let session = ArtSession::new(
        ArtConfig::default(),
        minimum,
        ArtDisplacement::Targeted {
            direction: array![-1.0, 0.4, 0.0, 0.2, 0.0, 0.0],
            support,
            amplitude: 0.25,
        },
    )
    .unwrap();
    let x = session.position();
    let shift = ((x[0] - 1.0).powi(2) + x[1].powi(2) + x[2].powi(2)).sqrt();
    assert!((shift - 0.25).abs() < 1e-12, "shift {shift}");
    assert!((x[3] - 1.0).abs() < 1e-12);
    assert!((x[4] - 3.0).abs() < 1e-12);
    assert!(x[5].abs() < 1e-12);
    assert!(session.push_direction()[3].abs() < 1e-12);
}

#[test]
fn zero_amplitude_is_rejected() {
    let err = ArtSession::new(
        ArtConfig::default(),
        array![1.0, 0.0, 0.0],
        ArtDisplacement::Random {
            seed: 1,
            support: ArtSupport::Global,
            amplitude: 0.0,
        },
    );
    assert!(matches!(err, Err(SaddleError::Invalid(_))));
}

#[test]
fn targeted_double_well_reaches_the_origin() {
    let surface = Counted {
        inner: DoubleWell,
        calls: AtomicUsize::new(0),
    };
    let mut session = ArtSession::new(
        ArtConfig::default(),
        array![1.0, 0.0, 0.0],
        ArtDisplacement::Targeted {
            direction: array![-1.0, 0.0, 0.0],
            support: ArtSupport::Global,
            amplitude: 0.2,
        },
    )
    .unwrap();
    let report = session.run(&surface, 40).unwrap();
    assert_eq!(
        report.status,
        ArtStatus::Converged,
        "status {:?} force {} curvature {} x {:?}",
        report.status,
        report.max_force,
        report.curvature,
        session.position()
    );
    assert!(session.position().iter().all(|c| c.abs() < 1e-3));
    assert!(report.curvature < 0.0);
    assert_eq!(double_well_index(session.position()), 1);
    assert_eq!(
        report.total_evaluations,
        surface.calls.load(Ordering::Relaxed)
    );
    assert!(report.total_evaluations > 0);
    eprintln!(
        "ART_BENCH surface=double_well_targeted art_calls={}",
        report.total_evaluations
    );
}

#[test]
fn local_push_on_one_atom_of_the_double_well_keeps_the_other_in_its_well() {
    let mut minimum = Array1::zeros(6);
    minimum[0] = 1.0;
    minimum[3] = 1.0;
    let support = ArtSupport::local_atoms(minimum.view(), 0, 0.0).unwrap();
    let mut session = ArtSession::new(
        ArtConfig::default(),
        minimum,
        ArtDisplacement::Targeted {
            direction: array![-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            support,
            amplitude: 0.2,
        },
    )
    .unwrap();
    session.set_force_gate(ForceGate::MaxForceOnAtom);
    let report = session.run(&DoubleWell, 50).unwrap();
    assert_eq!(report.status, ArtStatus::Converged, "{report:?}");
    let x = session.position();
    assert!(
        x[0].abs() < 1e-2 && x[1].abs() < 1e-2 && x[2].abs() < 1e-2,
        "moved atom {x:?}"
    );
    assert!(
        (x[3] - 1.0).abs() < 1e-2 && x[4].abs() < 1e-2 && x[5].abs() < 1e-2,
        "spectator {x:?}"
    );
    assert_eq!(double_well_index(x), 1);
}

#[test]
fn lanczos_waits_until_the_push_passes_the_activation_distance() {
    let config = ArtConfig::builder()
        .activation_distance(10.0)
        .perp_steps_push(0)
        .build()
        .unwrap();
    let mut session = ArtSession::new(
        config,
        array![1.0, 0.0, 0.0],
        ArtDisplacement::Targeted {
            direction: array![-1.0, 0.0, 0.0],
            support: ArtSupport::Global,
            amplitude: 0.2,
        },
    )
    .unwrap();
    let report = session.step(&DoubleWell).unwrap();
    assert_eq!(report.lanczos_evaluations, 0);
    assert!(report.curvature.is_nan());
    assert_eq!(report.status, ArtStatus::Pushing);
}

fn double_well_config() -> ArtConfig {
    ArtConfig::default()
}

fn mb_config() -> ArtConfig {
    ArtConfig {
        mode: MinModeConfig {
            kind: MinModeKind::Lanczos,
            force_tol: 5e-3,
            max_move: 0.05,
            // The reflected climb is the min-mode step. L-BFGS keeps
            // that force; FIRE's first capped step leaves this ridge.
            method: rgmin::Method::Lbfgs { memory: 8 },
            dr: 1e-4,
            difference: FiniteDifference::Central,
            rotation_tol: 1e-2,
            krylov_dim: 4,
            ..MinModeConfig::default()
        },
        push_step: 0.03,
        max_pushes: 40,
        max_climbs: 80,
        // The ridge is narrower than one extra FIRE step. The climb
        // already moves on the min-mode reflected force, which keeps
        // the perpendicular part once the curvature is negative.
        perp_steps_push: 0,
        perp_steps_climb: 0,
        krylov_min: 1,
        eigen_threshold: -0.01,
        activation_distance: 0.0,
        nsmooth: 0,
        push_over: 0.0,
    }
}

#[test]
fn targeted_muller_brown_reaches_the_saddle_between_the_deep_wells() {
    let minimum = array![MIN_A[0], MIN_A[1]];
    let toward = array![TS1[0] - MIN_A[0], TS1[1] - MIN_A[1]];
    let surface = Counted {
        inner: MullerBrown,
        calls: AtomicUsize::new(0),
    };
    let mut session = ArtSession::new(
        mb_config(),
        minimum,
        ArtDisplacement::Targeted {
            direction: toward,
            support: ArtSupport::Global,
            amplitude: 0.05,
        },
    )
    .unwrap();
    let report = session.run(&surface, 120).unwrap();
    let x = session.position();
    let distance = ((x[0] - TS1[0]).powi(2) + (x[1] - TS1[1]).powi(2)).sqrt();
    assert_eq!(
        report.status,
        ArtStatus::Converged,
        "status {:?} force {} curvature {} x {:?} distance {distance}",
        report.status,
        report.max_force,
        report.curvature,
        x
    );
    assert_eq!(muller_index(x[0], x[1]), 1, "x={x:?}");
    assert!(
        distance < 0.05,
        "landed at {x:?}, {distance} from the Muller-Brown saddle"
    );
    assert_eq!(
        report.total_evaluations,
        surface.calls.load(Ordering::Relaxed)
    );
    eprintln!(
        "ART_BENCH surface=muller_brown_targeted art_calls={} distance={distance}",
        report.total_evaluations
    );
}

fn batch_events(count: usize) -> Vec<ArtEvent> {
    (0..count)
        .map(|index| ArtEvent {
            minimum: array![1.0, 0.0, 0.0],
            displacement: ArtDisplacement::Random {
                seed: event_seed(0xA47E_5EED, index as u64),
                support: ArtSupport::Global,
                amplitude: 0.15,
            },
        })
        .collect()
}

fn assert_bitwise(left: &[ArtOutcome], right: &[ArtOutcome]) {
    assert_eq!(left.len(), right.len());
    for (index, (a, b)) in left.iter().zip(right).enumerate() {
        assert_eq!(a.status, b.status, "status at {index}");
        assert_eq!(a.energy.to_bits(), b.energy.to_bits(), "energy at {index}");
        assert_eq!(
            a.curvature.to_bits(),
            b.curvature.to_bits(),
            "curvature at {index}"
        );
        assert_eq!(
            a.max_force.to_bits(),
            b.max_force.to_bits(),
            "force at {index}"
        );
        assert_eq!(a.pushes, b.pushes, "pushes at {index}");
        assert_eq!(a.climbs, b.climbs, "climbs at {index}");
        assert_eq!(a.evaluations, b.evaluations, "evaluations at {index}");
        assert_eq!(a.position.len(), b.position.len());
        for (x, y) in a.position.iter().zip(b.position.iter()) {
            assert_eq!(x.to_bits(), y.to_bits(), "position at {index}");
        }
    }
}

fn serial_outcomes(events: &[ArtEvent], max_steps: usize) -> Vec<ArtOutcome> {
    events
        .iter()
        .map(|event| {
            let mut session = ArtSession::new(
                double_well_config(),
                event.minimum.clone(),
                event.displacement.clone(),
            )
            .unwrap();
            let report = session.run(&DoubleWell, max_steps).unwrap();
            ArtOutcome {
                status: report.status,
                position: session.position().to_owned(),
                energy: session.energy(),
                curvature: session.curvature(),
                max_force: report.max_force,
                pushes: report.pushes,
                climbs: report.climbs,
                evaluations: report.total_evaluations,
            }
        })
        .collect()
}

#[test]
fn batch_matches_across_thread_counts_and_rank_splits() {
    let events = batch_events(9);
    let serial = serial_outcomes(&events, 40);
    let serial_hash = outcome_hash(&serial);
    for threads in [1usize, 2, 3, 8] {
        let parallel =
            run_events(&double_well_config(), &events, &DoubleWell, 40, threads).expect("batch");
        assert_bitwise(&serial, &parallel);
        assert_eq!(outcome_hash(&parallel), serial_hash);
    }
    let third = events.len() / 3;
    let mut split =
        run_events(&double_well_config(), &events[..third], &DoubleWell, 40, 1).expect("rank 0");
    split.extend(
        run_events(
            &double_well_config(),
            &events[third..2 * third],
            &DoubleWell,
            40,
            1,
        )
        .expect("rank 1"),
    );
    split.extend(
        run_events(
            &double_well_config(),
            &events[2 * third..],
            &DoubleWell,
            40,
            3,
        )
        .expect("rank 2"),
    );
    assert_bitwise(&serial, &split);
    assert_eq!(outcome_hash(&split), serial_hash);
}

fn outcome_hash(outcomes: &[ArtOutcome]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for outcome in outcomes {
        hash = mix_bytes(hash, &(outcome.status as u8).to_le_bytes());
        hash = mix_bytes(hash, &outcome.energy.to_le_bytes());
        hash = mix_bytes(hash, &outcome.curvature.to_le_bytes());
        hash = mix_bytes(hash, &outcome.max_force.to_le_bytes());
        for value in outcome.position.iter() {
            hash = mix_bytes(hash, &value.to_le_bytes());
        }
        hash = mix_bytes(hash, &outcome.evaluations.to_le_bytes());
    }
    hash
}

fn mix_bytes(mut hash: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    hash
}

struct FailByX;

impl PointSurface for FailByX {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        Err(SaddleError::Surface(format!("{}", x[0])))
    }
}

#[test]
fn batch_error_is_the_lowest_event_index() {
    let events = vec![
        ArtEvent {
            minimum: array![1.0, 0.0, 0.0],
            displacement: ArtDisplacement::Targeted {
                direction: array![-1.0, 0.0, 0.0],
                support: ArtSupport::Global,
                amplitude: 0.2,
            },
        },
        ArtEvent {
            minimum: array![-1.0, 0.0, 0.0],
            displacement: ArtDisplacement::Targeted {
                direction: array![-1.0, 0.0, 0.0],
                support: ArtSupport::Global,
                amplitude: 0.2,
            },
        },
    ];
    for threads in [1usize, 2, 3, 8] {
        let err = run_events(&ArtConfig::default(), &events, &FailByX, 4, threads)
            .expect_err("event 0 fails");
        let message = err.to_string();
        assert!(
            message.contains("0.8"),
            "threads {threads} reported {message}"
        );
    }
}

#[test]
#[ignore = "release timings for independent ART events"]
fn strong_scaling_of_independent_art_events() {
    let count = std::env::var("ART_SCALE_EVENTS")
        .ok()
        .and_then(|text| text.parse().ok())
        .unwrap_or(512);
    let events = batch_events(count);
    let max_steps = 40;
    let reference =
        run_events(&double_well_config(), &events, &DoubleWell, max_steps, 1).expect("reference");
    let max_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .max(1);
    let mut widths = vec![1usize, 2, 4, 8];
    if !widths.contains(&max_threads) {
        widths.push(max_threads);
    }
    widths.retain(|width| *width >= 1);
    widths.sort_unstable();
    widths.dedup();
    let reps = 5;
    let mut samples = vec![Vec::with_capacity(reps); widths.len()];
    for _rep in 0..reps {
        for (slot, threads) in widths.iter().enumerate() {
            let started = std::time::Instant::now();
            let outcomes = run_events(
                &double_well_config(),
                &events,
                &DoubleWell,
                max_steps,
                *threads,
            )
            .expect("scaled batch");
            let elapsed = started.elapsed().as_secs_f64() * 1.0e3;
            assert_bitwise(&reference, &outcomes);
            samples[slot].push(elapsed);
        }
    }
    let mut medians = Vec::with_capacity(widths.len());
    for (threads, sample) in widths.iter().zip(samples.iter()) {
        let mut ordered = sample.clone();
        ordered.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = ordered[ordered.len() / 2];
        let low = ordered[0];
        let high = ordered[ordered.len() - 1];
        medians.push(median);
        eprintln!(
            "ART_SCALE threads={threads} median_ms={median:.3} range_ms=[{low:.3}, {high:.3}] reps={reps}"
        );
    }
    let t1 = medians[0];
    for (threads, median) in widths.iter().zip(medians.iter()).skip(1) {
        let speedup = t1 / median;
        let efficiency = speedup / *threads as f64;
        eprintln!("ART_SCALE threads={threads} speedup={speedup:.3} efficiency={efficiency:.3}");
    }
    let p = *widths.last().unwrap();
    let tp = *medians.last().unwrap();
    let speedup = t1 / tp;
    let efficiency = speedup / p as f64;
    let serial = if p > 1 {
        (p as f64 / speedup - 1.0) / (p as f64 - 1.0)
    } else {
        0.0
    };
    let limiter = if efficiency >= 0.7 {
        "hardware thread count"
    } else if serial >= 0.15 {
        "serial dispatch and join around the per-event search"
    } else {
        "uneven event cost across the fixed index buckets"
    };
    eprintln!(
        "ART_SCALE limiter={limiter} serial_fraction={serial:.3} widest_threads={p} widest_efficiency={efficiency:.3}"
    );
}

/// Henkelman–Jónsson model II: LEPS plus a harmonic oscillator.
/// The third coordinate is a unit spring so the vector is a Cartesian triple.
struct LepsHo;

fn leps_morse(r: f64, d: f64) -> (f64, f64) {
    let alpha = 1.942;
    let r0 = 0.742;
    let z = alpha * (r - r0);
    let attract = (-z).exp();
    let repel = (-2.0 * z).exp();
    let q = d / 2.0 * (1.5 * repel - attract);
    let j = d / 4.0 * (repel - 6.0 * attract);
    (q, j)
}

fn leps_ho_energy(r_ab: f64, x: f64) -> f64 {
    let r_ac = 3.742;
    let r_bc = r_ac - r_ab;
    let (q_ab, j_ab) = leps_morse(r_ab, 4.746);
    let (q_bc, j_bc) = leps_morse(r_bc, 4.746);
    let (q_ac, j_ac) = leps_morse(r_ac, 3.445);
    let a = 0.05;
    let b = 0.80;
    let c = 0.05;
    let q = q_ab / (1.0 + a) + q_bc / (1.0 + b) + q_ac / (1.0 + c);
    let j_ab = j_ab / (1.0 + a);
    let j_bc = j_bc / (1.0 + b);
    let j_ac = j_ac / (1.0 + c);
    let inside = j_ab * j_ab + j_bc * j_bc + j_ac * j_ac - j_ab * j_bc - j_bc * j_ac - j_ab * j_ac;
    let kc = 0.2025;
    let c_ho = 1.154;
    let u = r_ab - r_ac / 2.0 + x / c_ho;
    q - inside.max(0.0).sqrt() + 2.0 * kc * u * u
}

impl PointSurface for LepsHo {
    fn eval(&self, point: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (r_ab, x, z) = (point[0], point[1], point[2]);
        let dr = 1e-6;
        let energy = leps_ho_energy(r_ab, x) + 0.5 * z * z;
        let grad_r = (leps_ho_energy(r_ab + dr, x) - leps_ho_energy(r_ab - dr, x)) / (2.0 * dr);
        let grad_x = (leps_ho_energy(r_ab, x + dr) - leps_ho_energy(r_ab, x - dr)) / (2.0 * dr);
        Ok((energy, array![grad_r, grad_x, z]))
    }
}

fn leps_index(point: ArrayView1<f64>) -> i32 {
    let dr = 1e-5;
    let g = |r: f64, x: f64| {
        let shift = 1e-6;
        let gr = (leps_ho_energy(r + shift, x) - leps_ho_energy(r - shift, x)) / (2.0 * shift);
        let gx = (leps_ho_energy(r, x + shift) - leps_ho_energy(r, x - shift)) / (2.0 * shift);
        (gr, gx)
    };
    let (r, x) = (point[0], point[1]);
    let (gp, _) = g(r + dr, x);
    let (gm, _) = g(r - dr, x);
    let (_, hp) = g(r, x + dr);
    let (_, hm) = g(r, x - dr);
    let hrr = (gp - gm) / (2.0 * dr);
    let hxx = (hp - hm) / (2.0 * dr);
    let hrx = (g(r, x + dr).0 - g(r, x - dr).0) / (2.0 * dr);
    let tr = hrr + hxx;
    let det = hrr * hxx - hrx * hrx;
    let disc = (tr * tr - 4.0 * det).max(0.0).sqrt();
    let low = 0.5 * (tr - disc);
    let high = 0.5 * (tr + disc);
    i32::from(low < -1e-3) + i32::from(high < -1e-3)
}

fn leps_stationary(guess_r: f64, guess_x: f64) -> (f64, f64) {
    let mut r = guess_r;
    let mut x = guess_x;
    for _ in 0..40 {
        let (_, gradient) = LepsHo.eval(array![r, x, 0.0].view()).unwrap();
        if gradient[0].hypot(gradient[1]) < 1e-8 {
            break;
        }
        let dr = 1e-5;
        let g = |r: f64, x: f64| {
            let (_, gradient) = LepsHo.eval(array![r, x, 0.0].view()).unwrap();
            (gradient[0], gradient[1])
        };
        let (gr, gx) = g(r, x);
        let (grp, _) = g(r + dr, x);
        let (grm, _) = g(r - dr, x);
        let (_, gxp) = g(r, x + dr);
        let (_, gxm) = g(r, x - dr);
        let hrr = (grp - grm) / (2.0 * dr);
        let hxx = (gxp - gxm) / (2.0 * dr);
        let hrx = (g(r, x + dr).0 - g(r, x - dr).0) / (2.0 * dr);
        let det = hrr * hxx - hrx * hrx;
        if det.abs() < 1e-14 {
            break;
        }
        r -= (hxx * gr - hrx * gx) / det;
        x -= (hrr * gx - hrx * gr) / det;
    }
    (r, x)
}

fn leps_references() -> (Array1<f64>, Array1<f64>) {
    let mut best_min = (f64::INFINITY, 0.74, 0.0);
    let mut best_sad = (f64::INFINITY, 1.5, 0.0);
    let mut r = 0.55;
    while r < 3.2 {
        let mut x = -1.6;
        while x < 1.6 {
            let point = array![r, x, 0.0];
            let (energy, gradient) = LepsHo.eval(point.view()).unwrap();
            let force = gradient[0].hypot(gradient[1]);
            if force < 8.0 {
                let index = leps_index(point.view());
                if index == 0 && energy < best_min.0 {
                    best_min = (energy, r, x);
                }
                if index == 1 && force < best_sad.0 {
                    best_sad = (force, r, x);
                }
            }
            x += 0.08;
        }
        r += 0.08;
    }
    let (r_min, x_min) = leps_stationary(best_min.1, best_min.2);
    let (r_sad, x_sad) = leps_stationary(best_sad.1, best_sad.2);
    (array![r_min, x_min, 0.0], array![r_sad, x_sad, 0.0])
}

struct Lj7;

fn lj_pair(r2: f64) -> (f64, f64) {
    let s = 1.0 / r2;
    let s3 = s * s * s;
    let s6 = s3 * s3;
    let energy = 4.0 * (s6 - s3);
    let d_dr2 = 4.0 * (-6.0 * s6 * s + 3.0 * s3 * s);
    (energy, d_dr2)
}

impl PointSurface for Lj7 {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let n = x.len() / 3;
        let mut energy = 0.0;
        let mut gradient = Array1::zeros(x.len());
        for i in 0..n {
            for j in (i + 1)..n {
                let mut r2 = 0.0;
                let mut delta = [0.0; 3];
                for c in 0..3 {
                    delta[c] = x[3 * i + c] - x[3 * j + c];
                    r2 += delta[c] * delta[c];
                }
                if r2 < 1e-12 {
                    return Err(SaddleError::NonFinite("lj overlap"));
                }
                let (pair, d_dr2) = lj_pair(r2);
                energy += pair;
                for c in 0..3 {
                    let component = d_dr2 * 2.0 * delta[c];
                    gradient[3 * i + c] += component;
                    gradient[3 * j + c] -= component;
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
    for i in 0..n {
        for c in 0..3 {
            com[c] += x[3 * i + c];
        }
    }
    for value in &mut com {
        *value /= n as f64;
    }
    let mut rows = Vec::new();
    for axis in 0..3 {
        let mut row = Array1::zeros(x.len());
        for i in 0..n {
            row[3 * i + axis] = 1.0;
        }
        rows.push(row / (n as f64).sqrt());
    }
    for axis in 0..3 {
        let mut row = Array1::zeros(x.len());
        for i in 0..n {
            let r = [
                x[3 * i] - com[0],
                x[3 * i + 1] - com[1],
                x[3 * i + 2] - com[2],
            ];
            let v = match axis {
                0 => [0.0, -r[2], r[1]],
                1 => [r[2], 0.0, -r[0]],
                _ => [-r[1], r[0], 0.0],
            };
            row[3 * i] = v[0];
            row[3 * i + 1] = v[1];
            row[3 * i + 2] = v[2];
        }
        let norm = row.dot(&row).sqrt();
        if norm > 1e-8 {
            rows.push(row / norm);
        }
    }
    let mut out = Array2::zeros((rows.len(), x.len()));
    for (index, row) in rows.iter().enumerate() {
        out.row_mut(index).assign(row);
    }
    out
}

fn relax_lj(start: Array1<f64>) -> Array1<f64> {
    let mut solver = rgmin::Solver::new(
        rgmin::Method::Fire {
            kind: rgmin::FireKind::V2,
        },
        rgmin::Control {
            maxiter: usize::MAX,
            gtol: 0.0,
            istep: 0.05,
            maxmove: None,
            ftol_rel: None,
        },
        start.len(),
    );
    solver.set_atom_maxmove(0.05);
    let mut x = start;
    for _ in 0..400 {
        let oracle = rgmin::Oracle::unbounded(x.len(), |point: ArrayView1<f64>| {
            Lj7.eval(point)
                .unwrap_or((f64::INFINITY, Array1::zeros(point.len())))
        });
        solver.step(&oracle, &mut x).expect("lj relax");
        let (_, gradient) = Lj7.eval(x.view()).unwrap();
        if gradient.iter().all(|v| v.abs() < 1e-6) {
            break;
        }
    }
    x
}

fn pentagonal_bipyramid() -> Array1<f64> {
    let bond = 2f64.powf(1.0 / 6.0);
    let radius = bond / (2.0 * (std::f64::consts::PI / 5.0).sin());
    let height = (bond * bond - radius * radius).max(0.0).sqrt();
    let mut x = Array1::zeros(21);
    x[2] = height;
    x[5] = -height;
    for k in 0..5 {
        let angle = 2.0 * std::f64::consts::PI * (k as f64) / 5.0;
        let base = 6 + 3 * k;
        x[base] = radius * angle.cos();
        x[base + 1] = radius * angle.sin();
    }
    relax_lj(x)
}

struct ModeExcluding<'a, S> {
    inner: &'a S,
    extra: Array1<f64>,
}

impl<S: PointSurface> PointSurface for ModeExcluding<'_, S> {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        self.inner.eval(x)
    }

    fn excluded_modes(&self, x: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
        let mut modes = self.inner.excluded_modes(x)?;
        if modes.ncols() == 0 {
            modes = Array2::zeros((0, x.len()));
        }
        let mut stacked = Array2::zeros((modes.nrows() + 1, x.len()));
        if modes.nrows() > 0 {
            stacked
                .slice_mut(ndarray::s![0..modes.nrows(), ..])
                .assign(&modes);
        }
        stacked.row_mut(modes.nrows()).assign(&self.extra);
        Ok(stacked)
    }
}

fn lowest_curvature(surface: &impl PointSurface, x: ArrayView1<f64>) -> f64 {
    let config = MinModeConfig {
        kind: MinModeKind::Lanczos,
        krylov_dim: 12,
        rotation_tol: 1e-5,
        dr: 1e-4,
        ..MinModeConfig::default()
    };
    let mut session = MinModeSession::new(config, x.to_owned(), Array1::ones(x.len())).unwrap();
    session.estimate_mode(surface).unwrap().curvature
}

fn lj_index(x: ArrayView1<f64>) -> i32 {
    let lowest = lowest_curvature(&Lj7, x);
    if lowest >= -1e-2 {
        return 0;
    }
    let session_mode = {
        let config = MinModeConfig {
            kind: MinModeKind::Lanczos,
            krylov_dim: 12,
            rotation_tol: 1e-5,
            dr: 1e-4,
            ..MinModeConfig::default()
        };
        let mut session = MinModeSession::new(config, x.to_owned(), Array1::ones(x.len())).unwrap();
        session.estimate_mode(&Lj7).unwrap().mode
    };
    let second = lowest_curvature(
        &ModeExcluding {
            inner: &Lj7,
            extra: session_mode,
        },
        x,
    );
    if second < -1e-2 { 2 } else { 1 }
}

struct Tally<'a, S> {
    inner: &'a S,
    calls: AtomicUsize,
}

impl<S: PointSurface> PointSurface for Tally<'_, S> {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.inner.eval(x)
    }

    fn excluded_modes(&self, x: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
        self.inner.excluded_modes(x)
    }
}

#[derive(Clone)]
struct BenchRow {
    system: &'static str,
    method: &'static str,
    seed: u64,
    status: &'static str,
    converged: bool,
    censored: bool,
    iters: usize,
    calls: usize,
    energy: f64,
    fmax: f64,
    curvature: f64,
    wall_s: f64,
    endpoint: Array1<f64>,
    index: i32,
    reference: bool,
    inward: bool,
    pre_lanczos: usize,
    pre_push: usize,
    pre_perp: usize,
}

fn follower(kind: MinModeKind, max_move: f64) -> MinModeConfig {
    MinModeConfig {
        kind,
        max_move,
        ..MinModeConfig::default()
    }
}

fn art_status_name(status: ArtStatus) -> &'static str {
    match status {
        ArtStatus::Pushing => "pushing",
        ArtStatus::Climbing => "climbing",
        ArtStatus::Converged => "converged",
        ArtStatus::Exhausted => "exhausted",
    }
}

fn distance(left: &Array1<f64>, right: &Array1<f64>) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt()
}

/// One method from a shared displaced start, stopped when the counting
/// surface passes `budget`. A finish past that count is censored.
fn drive_art<S: PointSurface>(
    minimum: &Array1<f64>,
    seed: u64,
    amplitude: f64,
    surface: &S,
    budget: usize,
) -> (Array1<f64>, Array1<f64>, BenchRow) {
    let mut session = ArtSession::new(
        ArtConfig::default(),
        minimum.clone(),
        ArtDisplacement::Random {
            seed,
            support: ArtSupport::Global,
            amplitude,
        },
    )
    .unwrap();
    let start = session.position().to_owned();
    let direction = session.push_direction().to_owned();
    let tally = Tally {
        inner: surface,
        calls: AtomicUsize::new(0),
    };
    let started = Instant::now();
    let mut status = ArtStatus::Pushing;
    let mut iters = 0usize;
    let mut converged = false;
    let mut curvature = f64::NAN;
    let mut fmax = f64::NAN;
    let mut pre_lanczos = 0usize;
    let mut pre_push = 0usize;
    let mut pre_perp = 0usize;
    let censored = loop {
        if tally.calls.load(Ordering::Relaxed) > budget {
            break true;
        }
        let report = session.step(&tally).unwrap();
        iters += 1;
        status = report.status;
        curvature = report.curvature;
        fmax = report.max_force;
        pre_lanczos = report.pre_inflection_lanczos;
        pre_push = report.pre_inflection_push;
        pre_perp = report.pre_inflection_perp;
        let over = tally.calls.load(Ordering::Relaxed) > budget;
        if report.status == ArtStatus::Converged {
            converged = !over;
            break over;
        }
        if report.status == ArtStatus::Exhausted || over || iters > budget {
            converged = false;
            break over || iters > budget;
        }
    };
    let row = BenchRow {
        system: "",
        method: "art",
        seed,
        status: art_status_name(status),
        converged,
        censored,
        iters,
        calls: tally.calls.load(Ordering::Relaxed),
        energy: session.energy(),
        fmax,
        curvature,
        wall_s: started.elapsed().as_secs_f64(),
        endpoint: session.position().to_owned(),
        index: 0,
        reference: false,
        inward: direction.first().copied().unwrap_or(0.0) < 0.0,
        pre_lanczos,
        pre_push,
        pre_perp,
    };
    (start, direction, row)
}

fn drive_minmode<S: PointSurface>(
    kind: MinModeKind,
    method: &'static str,
    start: &Array1<f64>,
    direction: &Array1<f64>,
    max_move: f64,
    surface: &S,
    budget: usize,
) -> (BenchRow, Vec<(f64, f64)>) {
    let mut session =
        MinModeSession::new(follower(kind, max_move), start.clone(), direction.clone()).unwrap();
    let tally = Tally {
        inner: surface,
        calls: AtomicUsize::new(0),
    };
    let started = Instant::now();
    let mut status = "running";
    let mut iters = 0usize;
    let mut converged = false;
    let mut curvature = f64::NAN;
    let mut fmax = f64::NAN;
    let mut steps = Vec::new();
    let censored = loop {
        if tally.calls.load(Ordering::Relaxed) > budget {
            break true;
        }
        let report = session.step(&tally).unwrap();
        iters += 1;
        curvature = report.curvature;
        fmax = report.max_force;
        steps.push((curvature, fmax));
        let over = tally.calls.load(Ordering::Relaxed) > budget;
        if report.status == MinModeStatus::Converged {
            status = "converged";
            converged = !over;
            break over;
        }
        if over || iters > budget {
            break true;
        }
    };
    let row = BenchRow {
        system: "",
        method,
        seed: 0,
        status,
        converged,
        censored,
        iters,
        calls: tally.calls.load(Ordering::Relaxed),
        energy: session.energy().unwrap_or(f64::NAN),
        fmax,
        curvature,
        wall_s: started.elapsed().as_secs_f64(),
        endpoint: session.position().to_owned(),
        index: 0,
        reference: false,
        inward: direction.first().copied().unwrap_or(0.0) < 0.0,
        pre_lanczos: 0,
        pre_push: 0,
        pre_perp: 0,
    };
    (row, steps)
}

fn median_range(values: &[usize]) -> (f64, usize, usize) {
    if values.is_empty() {
        return (0.0, 0, 0);
    }
    let mut ordered = values.to_vec();
    ordered.sort_unstable();
    (
        ordered[ordered.len() / 2] as f64,
        ordered[0],
        *ordered.last().unwrap(),
    )
}

fn distinct_points(points: &[&Array1<f64>], tol: f64) -> usize {
    let mut reps: Vec<&Array1<f64>> = Vec::new();
    for point in points {
        let known = reps.iter().any(|rep| distance(point, rep) < tol);
        if !known {
            reps.push(point);
        }
    }
    reps.len()
}

fn atoms_of(x: &Array1<f64>) -> Vec<[f64; 3]> {
    (0..x.len() / 3)
        .map(|i| [x[3 * i], x[3 * i + 1], x[3 * i + 2]])
        .collect()
}

#[allow(clippy::needless_range_loop)]
fn jacobi_eig(mut a: [[f64; 3]; 3]) -> ([[f64; 3]; 3], [f64; 3]) {
    let mut vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..24 {
        let (p, q) = if a[0][1].abs() >= a[0][2].abs() && a[0][1].abs() >= a[1][2].abs() {
            (0, 1)
        } else if a[0][2].abs() >= a[1][2].abs() {
            (0, 2)
        } else {
            (1, 2)
        };
        if a[p][q].abs() < 1e-15 {
            break;
        }
        let tau = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
        let root = (1.0 + tau * tau).sqrt();
        let t = if tau >= 0.0 {
            1.0 / (tau + root)
        } else {
            -1.0 / (-tau + root)
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];
        a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;
        for r in 0..3 {
            if r == p || r == q {
                continue;
            }
            let arp = a[r][p];
            let arq = a[r][q];
            a[r][p] = c * arp - s * arq;
            a[p][r] = a[r][p];
            a[r][q] = s * arp + c * arq;
            a[q][r] = a[r][q];
        }
        for r in 0..3 {
            let vip = vectors[r][p];
            let viq = vectors[r][q];
            vectors[r][p] = c * vip - s * viq;
            vectors[r][q] = s * vip + c * viq;
        }
    }
    (vectors, [a[0][0], a[1][1], a[2][2]])
}

fn det3(m: [[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

fn matmul3(left: [[f64; 3]; 3], right: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                out[i][j] += left[i][k] * right[k][j];
            }
        }
    }
    out
}

fn transpose3(m: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

/// Root-mean-square distance after a rigid alignment, reflections included.
#[allow(clippy::needless_range_loop)]
fn kabsch_rmsd(left: &[[f64; 3]], right: &[[f64; 3]]) -> f64 {
    let n = left.len() as f64;
    let mut ca = [0.0; 3];
    let mut cb = [0.0; 3];
    for i in 0..left.len() {
        for c in 0..3 {
            ca[c] += left[i][c];
            cb[c] += right[i][c];
        }
    }
    for c in 0..3 {
        ca[c] /= n;
        cb[c] /= n;
    }
    let mut cov = [[0.0; 3]; 3];
    for k in 0..left.len() {
        for i in 0..3 {
            for j in 0..3 {
                cov[i][j] += (left[k][i] - ca[i]) * (right[k][j] - cb[j]);
            }
        }
    }
    let mut gram = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                gram[i][j] += cov[k][i] * cov[k][j];
            }
        }
    }
    let (mut v, evals) = jacobi_eig(gram);
    let mut sigma = [0.0; 3];
    for i in 0..3 {
        sigma[i] = evals[i].max(0.0).sqrt();
    }
    let mut u = [[0.0; 3]; 3];
    for col in 0..3 {
        if sigma[col] > 1e-12 {
            for i in 0..3 {
                let mut sum = 0.0;
                for j in 0..3 {
                    sum += cov[i][j] * v[j][col];
                }
                u[i][col] = sum / sigma[col];
            }
        }
    }
    // Complete a rank-deficient left basis so the rotation stays orthogonal.
    if sigma.iter().filter(|s| **s <= 1e-12).count() > 0 {
        for col in 0..3 {
            if sigma[col] > 1e-12 {
                continue;
            }
            let mut axis = [0.0, 0.0, 0.0];
            axis[col] = 1.0;
            for known in 0..3 {
                if sigma[known] <= 1e-12 {
                    continue;
                }
                let mut dot = 0.0;
                for i in 0..3 {
                    dot += axis[i] * u[i][known];
                }
                for i in 0..3 {
                    axis[i] -= dot * u[i][known];
                }
            }
            let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
            if norm > 1e-12 {
                for i in 0..3 {
                    u[i][col] = axis[i] / norm;
                }
            }
        }
    }
    let mut rotation = matmul3(v, transpose3(u));
    if det3(rotation) < 0.0 {
        let flip = sigma
            .iter()
            .enumerate()
            .min_by(|left, right| left.1.partial_cmp(right.1).unwrap())
            .map(|(index, _)| index)
            .unwrap_or(2);
        for i in 0..3 {
            v[i][flip] = -v[i][flip];
        }
        rotation = matmul3(v, transpose3(u));
    }
    let proper = rmsd_of(left, right, &ca, &cb, rotation);
    // The mirror image of a free cluster is the same saddle.
    for i in 0..3 {
        v[i][2] = -v[i][2];
    }
    let mirrored = matmul3(v, transpose3(u));
    proper.min(rmsd_of(left, right, &ca, &cb, mirrored))
}

fn rmsd_of(
    left: &[[f64; 3]],
    right: &[[f64; 3]],
    ca: &[f64; 3],
    cb: &[f64; 3],
    rotation: [[f64; 3]; 3],
) -> f64 {
    let mut err = 0.0;
    for k in 0..left.len() {
        for i in 0..3 {
            let mut moved = 0.0;
            for j in 0..3 {
                moved += rotation[i][j] * (right[k][j] - cb[j]);
            }
            let delta = left[k][i] - ca[i] - moved;
            err += delta * delta;
        }
    }
    (err / left.len() as f64).sqrt()
}

fn heap_permutations(n: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut order: Vec<usize> = (0..n).collect();
    let mut count = vec![0usize; n];
    out.push(order.clone());
    let mut i = 0;
    while i < n {
        if count[i] < i {
            if i % 2 == 0 {
                order.swap(0, i);
            } else {
                order.swap(count[i], i);
            }
            out.push(order.clone());
            count[i] += 1;
            i = 0;
        } else {
            count[i] = 0;
            i += 1;
        }
    }
    out
}

fn aligned_rmsd(left: &Array1<f64>, right: &Array1<f64>, permutations: &[Vec<usize>]) -> f64 {
    let a = atoms_of(left);
    let b = atoms_of(right);
    let mut best = f64::INFINITY;
    let mut trial = vec![[0.0; 3]; b.len()];
    for order in permutations {
        for (slot, &index) in order.iter().enumerate() {
            trial[slot] = b[index];
        }
        best = best.min(kabsch_rmsd(&a, &trial));
    }
    best
}

struct LjSaddle {
    energy: f64,
    point: Array1<f64>,
}

fn lj_reference_list(rows: &[BenchRow], permutations: &[Vec<usize>]) -> Vec<LjSaddle> {
    let mut saddles = Vec::new();
    for row in rows.iter().filter(|row| row.converged && !row.censored) {
        let known = saddles.iter().any(|saddle: &LjSaddle| {
            (saddle.energy - row.energy).abs() <= 1e-4
                && aligned_rmsd(&saddle.point, &row.endpoint, permutations) <= 0.05
        });
        if !known {
            saddles.push(LjSaddle {
                energy: row.energy,
                point: row.endpoint.clone(),
            });
        }
    }
    saddles
}

fn quote_endpoint(point: &Array1<f64>) -> String {
    let body = point
        .iter()
        .map(|value| format!("{value:.8}"))
        .collect::<Vec<_>>()
        .join(" ");
    format!("\"{body}\"")
}

fn write_csv(path: &std::path::Path, rows: &[BenchRow]) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("bench directory");
    }
    let mut file = std::fs::File::create(path).expect("bench csv");
    writeln!(
        file,
        "system,method,seed,converged,iters,calls,energy,fmax,wall_s,endpoint,censored,reference"
    )
    .unwrap();
    for row in rows {
        writeln!(
            file,
            "{},{},{},{},{},{},{:.8},{:.8},{:.6},{},{},{}",
            row.system,
            row.method,
            row.seed,
            u8::from(row.converged),
            row.iters,
            row.calls,
            row.energy,
            row.fmax,
            row.wall_s,
            quote_endpoint(&row.endpoint),
            u8::from(row.censored),
            u8::from(row.reference)
        )
        .unwrap();
    }
}

fn summarize(rows: &[BenchRow], lj: bool, permutations: &[Vec<usize>]) {
    let system = rows.first().map(|row| row.system).unwrap_or("none");
    let lj_list = if lj {
        lj_reference_list(rows, permutations)
    } else {
        Vec::new()
    };
    for method in ["art", "dimer", "lanczos"] {
        let mine: Vec<&BenchRow> = rows.iter().filter(|row| row.method == method).collect();
        let successes: Vec<&BenchRow> = mine
            .iter()
            .copied()
            .filter(|row| row.converged && !row.censored)
            .collect();
        let calls: Vec<usize> = successes.iter().map(|row| row.calls).collect();
        let (median, low, high) = median_range(&calls);
        let censored = mine.iter().filter(|row| row.censored).count();
        let reference = mine.iter().filter(|row| row.reference).count();
        let exhausted = mine.iter().filter(|row| row.status == "exhausted").count();
        let curvature_median = {
            let mut values: Vec<f64> = successes.iter().map(|row| row.curvature).collect();
            values.sort_by(|left, right| {
                left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
            });
            values.get(values.len() / 2).copied().unwrap_or(f64::NAN)
        };
        let distinct = if lj {
            let hits: Vec<BenchRow> = successes.iter().map(|row| (*row).clone()).collect();
            lj_reference_list(&hits, permutations).len()
        } else {
            let points: Vec<&Array1<f64>> = successes.iter().map(|row| &row.endpoint).collect();
            distinct_points(&points, 0.15)
        };
        eprintln!(
            "ART_BENCH system={system} method={method} successes={} censored={censored} exhausted={exhausted} reference={reference} median_calls={median:.1} range=[{low}, {high}] distinct={distinct} curvature_median={curvature_median:.4}",
            successes.len()
        );
    }
    if lj {
        eprintln!(
            "ART_BENCH system={system} reference_saddles={}",
            lj_list.len()
        );
    }
    let art: Vec<&BenchRow> = rows.iter().filter(|row| row.method == "art").collect();
    for (label, values) in [
        (
            "lanczos",
            art.iter().map(|row| row.pre_lanczos).collect::<Vec<_>>(),
        ),
        (
            "push",
            art.iter().map(|row| row.pre_push).collect::<Vec<_>>(),
        ),
        (
            "perp",
            art.iter().map(|row| row.pre_perp).collect::<Vec<_>>(),
        ),
    ] {
        let (median, low, high) = median_range(&values);
        eprintln!(
            "ART_PUSH system={system} phase={label} median={median:.1} range=[{low}, {high}]"
        );
    }
}

fn mb_index(point: ArrayView1<f64>) -> i32 {
    muller_index(point[0], point[1])
}

fn finish_row(
    mut row: BenchRow,
    system: &'static str,
    index_of: &dyn Fn(ArrayView1<f64>) -> i32,
    named: Option<&Array1<f64>>,
) -> BenchRow {
    row.system = system;
    row.index = index_of(row.endpoint.view());
    if let Some(target) = named {
        row.reference = distance(&row.endpoint, target) < 0.15;
    }
    row
}

#[test]
fn fair_force_budget_art_dimer_and_lanczos() {
    let (leps_min, leps_saddle) = leps_references();
    assert_eq!(leps_index(leps_min.view()), 0, "leps minimum {leps_min:?}");
    assert_eq!(
        leps_index(leps_saddle.view()),
        1,
        "leps saddle {leps_saddle:?}"
    );
    eprintln!(
        "ART_BENCH surface=leps_ho minimum={:?} saddle={:?}",
        leps_min, leps_saddle
    );
    let lj_min = pentagonal_bipyramid();
    let (_, lj_force) = Lj7.eval(lj_min.view()).unwrap();
    assert!(
        lj_force.iter().all(|value| value.abs() < 1e-4),
        "lj7 minimum force {lj_force:?}"
    );

    let permutations = heap_permutations(7);
    assert!(aligned_rmsd(&lj_min, &lj_min, &permutations) < 1e-6);
    let mut swapped = lj_min.clone();
    for c in 0..3 {
        swapped.swap(c, 3 + c);
    }
    assert!(
        aligned_rmsd(&lj_min, &swapped, &permutations) < 1e-6,
        "a permutation of LJ7 must align"
    );
    let mut rows = Vec::new();
    let mut leps_dimer_trace = Vec::new();

    let dw_min = array![1.0, 0.0, 0.0];
    let dw_saddle = array![0.0, 0.0, 0.0];
    let mb_min = array![MIN_A[0], MIN_A[1]];
    let mb_saddle = array![TS1[0], TS1[1]];
    rows.extend(compare_named(
        &Compare {
            system: "double_well",
            minimum: &dw_min,
            named: Some(&dw_saddle),
            amplitude: 0.15,
            seed0: 1_000,
            seeds: 32,
            budget: 2_000,
            max_move: 0.1,
            surface: &DoubleWell,
            index_of: &double_well_index,
            trace_dimer: false,
        },
        &mut leps_dimer_trace,
    ));
    rows.extend(compare_named(
        &Compare {
            system: "muller_brown",
            minimum: &mb_min,
            named: Some(&mb_saddle),
            amplitude: 0.08,
            seed0: 4_000,
            seeds: 32,
            budget: 2_000,
            max_move: 0.1,
            surface: &MullerBrown,
            index_of: &mb_index,
            trace_dimer: false,
        },
        &mut leps_dimer_trace,
    ));
    rows.extend(compare_named(
        &Compare {
            system: "leps_ho",
            minimum: &leps_min,
            named: Some(&leps_saddle),
            amplitude: 0.35,
            seed0: 7_000,
            seeds: 32,
            budget: 2_000,
            max_move: 0.1,
            surface: &LepsHo,
            index_of: &leps_index,
            trace_dimer: true,
        },
        &mut leps_dimer_trace,
    ));
    rows.extend(compare_named(
        &Compare {
            system: "lj7",
            minimum: &lj_min,
            named: None,
            amplitude: 0.3,
            seed0: 8_000,
            seeds: 32,
            budget: 5_000,
            max_move: 0.2,
            surface: &Lj7,
            index_of: &lj_index,
            trace_dimer: false,
        },
        &mut leps_dimer_trace,
    ));

    let lj_rows: Vec<BenchRow> = rows
        .iter()
        .filter(|row| row.system == "lj7")
        .cloned()
        .collect();
    let lj_list = lj_reference_list(&lj_rows, &permutations);
    for row in rows.iter_mut().filter(|row| row.system == "lj7") {
        row.reference = lj_list
            .iter()
            .any(|saddle| (saddle.energy - row.energy).abs() <= 1e-4);
    }

    for system in ["double_well", "muller_brown", "leps_ho", "lj7"] {
        let slice: Vec<BenchRow> = rows
            .iter()
            .filter(|row| row.system == system)
            .cloned()
            .collect();
        summarize(&slice, system == "lj7", &permutations);
        if system == "double_well" {
            let inward = slice
                .iter()
                .filter(|row| row.method == "art" && row.inward)
                .count();
            eprintln!("ART_BENCH system=double_well inward_seeds={inward}");
            for method in ["art", "dimer", "lanczos"] {
                let hits = slice
                    .iter()
                    .filter(|row| {
                        row.method == method && row.inward && row.converged && !row.censored
                    })
                    .count();
                eprintln!(
                    "ART_BENCH system=double_well method={method} inward_success={hits}/{inward}"
                );
            }
        }
    }

    let leps_dimer_ok = rows
        .iter()
        .filter(|row| row.system == "leps_ho" && row.method == "dimer" && row.converged)
        .count();
    if leps_dimer_ok == 0 {
        eprintln!(
            "ART_DIMER_SETTINGS kind=dimer max_move=0.1 force_tol=1e-3 dr=1e-3 rotation_tol=1e-4 max_rotations=20 method=fire_v2 difference=forward"
        );
        for (step, (curvature, force)) in leps_dimer_trace.iter().enumerate() {
            eprintln!("ART_DIMER_TRAJ step={step} curvature={curvature:.6} fmax={force:.6}");
        }
    }

    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/art_bench.csv");
    write_csv(&path, &rows);
    eprintln!("ART_BENCH csv={}", path.display());
    assert_eq!(rows.len(), 4 * 32 * 3);
    let art_inward = rows
        .iter()
        .filter(|row| {
            row.system == "double_well" && row.method == "art" && row.inward && row.converged
        })
        .count();
    assert!(
        art_inward >= 1,
        "no inward double-well ART search converged inside the force budget"
    );
}

struct Compare<'a, S> {
    system: &'static str,
    minimum: &'a Array1<f64>,
    named: Option<&'a Array1<f64>>,
    amplitude: f64,
    seed0: u64,
    seeds: u64,
    budget: usize,
    max_move: f64,
    surface: &'a S,
    index_of: &'a dyn Fn(ArrayView1<f64>) -> i32,
    trace_dimer: bool,
}

fn compare_named<S: PointSurface>(
    job: &Compare<'_, S>,
    trace_out: &mut Vec<(f64, f64)>,
) -> Vec<BenchRow> {
    let mut rows = Vec::new();
    for offset in 0..job.seeds {
        let seed = job.seed0 + offset;
        let (start, direction, art) =
            drive_art(job.minimum, seed, job.amplitude, job.surface, job.budget);
        let art = finish_row(art, job.system, job.index_of, job.named);
        let (mut dimer, trace) = drive_minmode(
            MinModeKind::Dimer,
            "dimer",
            &start,
            &direction,
            job.max_move,
            job.surface,
            job.budget,
        );
        dimer.seed = seed;
        if job.trace_dimer && offset == 0 {
            *trace_out = trace;
        }
        let dimer = finish_row(dimer, job.system, job.index_of, job.named);
        let (mut lanczos, _) = drive_minmode(
            MinModeKind::Lanczos,
            "lanczos",
            &start,
            &direction,
            job.max_move,
            job.surface,
            job.budget,
        );
        lanczos.seed = seed;
        let lanczos = finish_row(lanczos, job.system, job.index_of, job.named);
        rows.push(art);
        rows.push(dimer);
        rows.push(lanczos);
    }
    rows
}
