//! ART nouveau on Muller-Brown and the double-well atom used by the
//! minimum-mode and band tests. Seeded pushes report a success rate and
//! the force calls per saddle next to the dimer driver.

use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, ArrayView1, array};
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
        ArtConfig {
            force_tol: 1e-3,
            ..ArtConfig::default()
        },
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

fn dimer_from(
    start: Array1<f64>,
    seed: Array1<f64>,
    surface: &impl PointSurface,
    force_tol: f64,
    max_move: f64,
    max_steps: usize,
) -> (bool, usize, Array1<f64>) {
    let config = MinModeConfig {
        kind: MinModeKind::Dimer,
        force_tol,
        max_move,
        ..MinModeConfig::default()
    };
    let mut session = MinModeSession::new(config, start, seed).unwrap();
    let mut total = 0;
    let mut converged = false;
    for _ in 0..max_steps {
        let report = session.step(surface).unwrap();
        total += report.evaluations;
        if report.status == MinModeStatus::Converged {
            converged = true;
            break;
        }
    }
    (converged, total, session.position().to_owned())
}

struct GateCount {
    calls: AtomicUsize,
}

impl PointSurface for GateCount {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        DoubleWell.eval(x)
    }
}

fn double_well_config() -> ArtConfig {
    ArtConfig {
        // The reaction coordinate is the stiffest mode at the minimum.
        // Perpendicular relaxation during the opening dumps the push
        // into the soft axes and the well never opens.
        perp_steps_push: 0,
        ..ArtConfig::default()
    }
}

fn random_double_well(seed: u64) -> (bool, usize, bool, usize) {
    let minimum = array![1.0, 0.0, 0.0];
    let config = double_well_config();
    let displacement = ArtDisplacement::Random {
        seed,
        support: ArtSupport::Global,
        amplitude: 0.15,
    };
    let mut session = ArtSession::new(config.clone(), minimum, displacement).unwrap();
    let start = session.position().to_owned();
    let direction = session.push_direction().to_owned();
    let surface = GateCount {
        calls: AtomicUsize::new(0),
    };
    let report = session.run(&surface, 50).unwrap();
    assert_eq!(
        report.total_evaluations,
        surface.calls.load(Ordering::Relaxed)
    );
    let art_ok =
        report.status == ArtStatus::Converged && double_well_index(session.position()) == 1;
    let (dimer_ok, dimer_calls, dimer_x) = dimer_from(
        start,
        direction,
        &DoubleWell,
        config.force_tol,
        config.max_move,
        50,
    );
    let dimer_saddle = dimer_ok && double_well_index(dimer_x.view()) == 1;
    (art_ok, report.total_evaluations, dimer_saddle, dimer_calls)
}

#[test]
fn seeded_random_pushes_on_the_double_well_against_the_dimer() {
    let seeds: Vec<u64> = (0..16).map(|i| 1_000 + i).collect();
    let mut art_ok = 0;
    let mut art_calls = 0;
    let mut dimer_ok = 0;
    let mut dimer_calls = 0;
    for seed in &seeds {
        let (art_success, art_n, dimer_success, dimer_n) = random_double_well(*seed);
        if art_success {
            art_ok += 1;
            art_calls += art_n;
        }
        if dimer_success {
            dimer_ok += 1;
            dimer_calls += dimer_n;
        }
    }
    let art_mean = if art_ok == 0 {
        0.0
    } else {
        art_calls as f64 / art_ok as f64
    };
    let dimer_mean = if dimer_ok == 0 {
        0.0
    } else {
        dimer_calls as f64 / dimer_ok as f64
    };
    eprintln!(
        "ART_BENCH surface=double_well seeds={} art_success={art_ok} art_mean_calls={art_mean:.1} dimer_success={dimer_ok} dimer_mean_calls={dimer_mean:.1}",
        seeds.len()
    );
    assert!(
        art_ok * 4 >= seeds.len(),
        "ART success {art_ok}/{} is below a quarter of the seeded pushes",
        seeds.len()
    );
    assert!(
        art_ok >= dimer_ok,
        "ART {art_ok} converged fewer seeded pushes than the dimer {dimer_ok}"
    );
}

fn mb_config() -> ArtConfig {
    ArtConfig {
        force_tol: 5e-3,
        push_step: 0.03,
        max_move: 0.05,
        max_pushes: 40,
        max_climbs: 80,
        perp_steps_push: 0,
        perp_steps_climb: 4,
        dr: 1e-4,
        difference: FiniteDifference::Central,
        rotation_tol: 1e-2,
        krylov_dim: 4,
        eigen_threshold: 0.0,
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

struct MbCount {
    calls: AtomicUsize,
}

impl PointSurface for MbCount {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        MullerBrown.eval(x)
    }
}

fn random_muller(seed: u64) -> (bool, usize, bool, usize) {
    let minimum = array![MIN_A[0], MIN_A[1]];
    let config = mb_config();
    let mut session = ArtSession::new(
        config.clone(),
        minimum,
        ArtDisplacement::Random {
            seed,
            support: ArtSupport::Global,
            amplitude: 0.08,
        },
    )
    .unwrap();
    let start = session.position().to_owned();
    let direction = session.push_direction().to_owned();
    let surface = MbCount {
        calls: AtomicUsize::new(0),
    };
    let report = session.run(&surface, 200).unwrap();
    assert_eq!(
        report.total_evaluations,
        surface.calls.load(Ordering::Relaxed)
    );
    let x = session.position();
    let art_ok = report.status == ArtStatus::Converged && muller_index(x[0], x[1]) == 1;
    let (dimer_ok, dimer_calls, dimer_x) = dimer_from(
        start,
        direction,
        &MullerBrown,
        config.force_tol,
        config.max_move,
        40,
    );
    let dimer_saddle = dimer_ok && muller_index(dimer_x[0], dimer_x[1]) == 1;
    (art_ok, report.total_evaluations, dimer_saddle, dimer_calls)
}

#[test]
fn seeded_random_pushes_on_muller_brown_against_the_dimer() {
    let seeds: Vec<u64> = (0..12).map(|i| 4_000 + i).collect();
    let mut art_ok = 0;
    let mut art_calls = 0;
    let mut dimer_ok = 0;
    let mut dimer_calls = 0;
    for seed in &seeds {
        let (art_success, art_n, dimer_success, dimer_n) = random_muller(*seed);
        if art_success {
            art_ok += 1;
            art_calls += art_n;
        }
        if dimer_success {
            dimer_ok += 1;
            dimer_calls += dimer_n;
        }
    }
    let art_mean = if art_ok == 0 {
        0.0
    } else {
        art_calls as f64 / art_ok as f64
    };
    let dimer_mean = if dimer_ok == 0 {
        0.0
    } else {
        dimer_calls as f64 / dimer_ok as f64
    };
    eprintln!(
        "ART_BENCH surface=muller_brown seeds={} art_success={art_ok} art_mean_calls={art_mean:.1} dimer_success={dimer_ok} dimer_mean_calls={dimer_mean:.1}",
        seeds.len()
    );
    assert!(
        art_ok >= 1,
        "no seeded Muller-Brown push reached an index-1 saddle"
    );
    assert!(
        art_ok >= dimer_ok,
        "ART {art_ok} converged fewer seeded pushes than the dimer {dimer_ok}"
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
    let events = batch_events(8);
    let serial = serial_outcomes(&events, 40);
    let width = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(2)
        .max(2);
    for threads in [1usize, 2, width] {
        let parallel =
            run_events(&double_well_config(), &events, &DoubleWell, 40, threads).expect("batch");
        assert_bitwise(&serial, &parallel);
    }
    let mid = events.len() / 2;
    let mut split =
        run_events(&double_well_config(), &events[..mid], &DoubleWell, 40, 1).expect("low rank");
    split.extend(
        run_events(
            &double_well_config(),
            &events[mid..],
            &DoubleWell,
            40,
            width,
        )
        .expect("high rank"),
    );
    assert_bitwise(&serial, &split);
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
    let width = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(2)
        .max(2);
    for threads in [1usize, 2, width] {
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
    let mut widths = vec![1usize, 2, 4];
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
