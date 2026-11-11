//! Force calls to a converged climbing-image saddle.
//!
//! The eOn column is the iteration in `NudgedElasticBand::compute`, the
//! climbing-image choice in `updateForces`, `FIRE::step`, and `LBFGS::step`
//! from TheochemUI/eOn (client/NudgedElasticBand.cpp, client/FIRE.cpp,
//! client/LBFGS.cpp). The tangent, spring and projection are this crate's
//! port of those eOn routines. Endpoints are evaluated once. FIRE does not
//! rescale its velocity when the per-atom cap shortens the step, and L-BFGS
//! takes the step with no energy line search, which is eOn's default
//! `lbfgs_accept = none`. A negative curvature takes a max-move step.
//!
//! rgsaddle is `BandSession` with the same images, spring, per-atom cap,
//! tolerance and optimizer. FIRE uses the native velocity-Verlet step
//! (`FireKind::V1`) and the eOn time step `1/10.1805055`, whose cap is
//! `2.5` times that step. The stop, for both, is eOn's: the largest
//! per-image Euclidean norm of the projected force, or only the climbing
//! image's norm when every image is within ten times the tolerance.
//!
//! `scripts/neb_survey.sh` writes `bench/raw/neb_survey.csv`.

mod eon;
mod geom;
mod model;
mod pot;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use ndarray::{Array1, Array2, ArrayView2};
use rgsaddle::{
    BandConfig, BandSession, BandSurface, Cell as BandCell, CiConfig, GrowingString, IdppConfig,
    SaddleError, SpringKind, grow_string, interpolated_path,
};

use eon::run_eon;
use geom::{build_cases, minimize};
use model::{
    Counted, DT, K_SPRING, MAX_MOVE, Optim, Problem, TOL, band_config, run_rgsaddle, run_tuned,
};
use pot::Pot;

fn main() {
    let cases = match build_cases() {
        Ok(cases) => cases,
        Err(err) => {
            eprintln!("setup failed: {err}");
            std::process::exit(1);
        }
    };
    println!(
        "{:<16} {:<8} {:<8} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8}",
        "case", "optim", "path", "eon", "eon_s", "rgsaddle", "rgs_s", "eon_ci", "rgs_ci"
    );
    let mut csv = vec!["case,method,calls,seconds,converged,ci,held_out".to_string()];
    for case in &cases {
        for optim in [Optim::Fire, Optim::Lbfgs] {
            let eon = match run_eon(case, optim) {
                Ok(row) => row,
                Err(err) => {
                    println!("{:<16} {:<8} eon failed: {err}", case.name, optim.label());
                    match run_rgsaddle(case, optim, false) {
                        Ok(rgs) => println!(
                            "{:<16} {:<8} {:>10} {:>10} {:>10} {:>10.3} {:>8} {:>8}",
                            case.name,
                            optim.label(),
                            "fail",
                            "-",
                            rgs.calls,
                            rgs.seconds,
                            "-",
                            rgs.ci.map(|i| i.to_string()).unwrap_or_else(|| "-".into()),
                        ),
                        Err(rerr) => println!(
                            "{:<16} {:<8} rgsaddle also failed: {rerr}",
                            case.name,
                            optim.label()
                        ),
                    }
                    continue;
                }
            };
            let rgs = match run_rgsaddle(case, optim, false) {
                Ok(row) => row,
                Err(err) => {
                    println!(
                        "{:<16} {:<8} {:>10} {:>10.3} rgsaddle failed: {err}",
                        case.name,
                        optim.label(),
                        eon.calls,
                        eon.seconds
                    );
                    continue;
                }
            };
            println!(
                "{:<16} {:<8} {:<8} {:>10} {:>10.3} {:>10} {:>10.3} {:>8} {:>8}",
                case.name,
                optim.label(),
                case.path_kind,
                eon.calls,
                eon.seconds,
                rgs.calls,
                rgs.seconds,
                eon.ci.map(|i| i.to_string()).unwrap_or_else(|| "-".into()),
                rgs.ci.map(|i| i.to_string()).unwrap_or_else(|| "-".into()),
            );
            push_csv(
                &mut csv,
                case,
                &format!("eon-{}", optim.label()),
                eon.calls,
                eon.seconds,
                eon.converged,
                eon.ci,
            );
            push_csv(
                &mut csv,
                case,
                &format!("ci-{}", optim.label()),
                rgs.calls,
                rgs.seconds,
                rgs.converged,
                rgs.ci,
            );
            if matches!(optim, Optim::Fire) {
                match run_rgsaddle(case, optim, true) {
                    Ok(local) => println!(
                        "{:<16} {:<8} {:<8} {:>10} {:>10} {:>10} {:>10.3} {:>8} {:>8}{}",
                        case.name,
                        "fire-img",
                        case.path_kind,
                        "",
                        "",
                        local.calls,
                        local.seconds,
                        "",
                        local
                            .ci
                            .map(|i| i.to_string())
                            .unwrap_or_else(|| "-".into()),
                        if case.held_out { " held-out" } else { "" },
                    ),
                    Err(err) => println!("{:<16} fire-img failed: {err}", case.name),
                }
                survey(case, &mut csv);
            }
        }
    }
    if let Err(err) = pt_adatom(&mut csv) {
        eprintln!("pt adatom failed: {err}");
    }
    if let Err(err) = pt_step(&mut csv) {
        eprintln!("pt step failed: {err}");
    }
    if let Err(err) = large_vacancy(&mut csv) {
        eprintln!("large vacancy failed: {err}");
    }
    profile_band();
    std::fs::create_dir_all("bench/raw").expect("bench/raw");
    std::fs::write("bench/raw/neb_survey.csv", csv.join("\n") + "\n").expect("csv");
    println!("wrote bench/raw/neb_survey.csv");
}

pub(crate) fn push_csv(
    csv: &mut Vec<String>,
    case: &Problem,
    method: &str,
    calls: usize,
    seconds: f64,
    converged: bool,
    ci: Option<usize>,
) {
    csv.push(format!(
        "{case},{method},{calls},{seconds:.6},{converged},{ci},{held}",
        case = case.name,
        ci = ci.map(|i| i.to_string()).unwrap_or_default(),
        held = if case.held_out { 1 } else { 0 },
    ));
}

pub(crate) fn survey(case: &Problem, csv: &mut Vec<String>) {
    for (method, spring, oci) in [
        (
            "energy-weighted",
            SpringKind::EnergyWeighted {
                k_min: 0.97,
                k_max: 9.7,
            },
            false,
        ),
        ("oci", SpringKind::Uniform { k: K_SPRING }, true),
    ] {
        match run_tuned(case, spring, oci) {
            Ok(row) => {
                println!(
                    "{:<16} {:<8} {:<8} {:>10} {:>10} {:>10} {:>10.3} {:>8} {:>8}",
                    case.name,
                    method,
                    case.path_kind,
                    "",
                    "",
                    row.calls,
                    row.seconds,
                    "",
                    row.ci.map(|i| i.to_string()).unwrap_or_else(|| "-".into()),
                );
                push_csv(
                    csv,
                    case,
                    method,
                    row.calls,
                    row.seconds,
                    row.converged,
                    row.ci,
                );
            }
            Err(err) => {
                println!("{:<16} {method} failed: {err}", case.name);
                push_csv(csv, case, method, 0, 0.0, false, None);
            }
        }
    }
    let surface = Counted {
        pot: case.pot.clone(),
        cell: case.cell,
        calls: AtomicUsize::new(0),
    };
    let config = GrowingString {
        images: case.positions.nrows(),
        force_tol: TOL,
        max_move: MAX_MOVE,
        ..GrowingString::default()
    };
    let started = Instant::now();
    let last = case.positions.nrows() - 1;
    match grow_string(
        case.positions.row(0),
        case.positions.row(last),
        &config,
        &surface,
    ) {
        Ok(report) => {
            let seconds = started.elapsed().as_secs_f64();
            let calls = surface.calls.load(Ordering::Relaxed);
            println!(
                "{:<16} {:<8} {:<8} {:>10} {:>10} {:>10} {:>10.3} {:>8} {:>8}",
                case.name,
                "gsm",
                "grow",
                "",
                "",
                calls,
                seconds,
                "",
                report
                    .ci_index
                    .map(|i| i.to_string())
                    .unwrap_or_else(|| "-".into()),
            );
            push_csv(
                csv,
                case,
                "growing-string",
                calls,
                seconds,
                report.converged,
                report.ci_index,
            );
        }
        Err(err) => {
            println!("{:<16} gsm failed: {err}", case.name);
            push_csv(csv, case, "growing-string", 0, 0.0, false, None);
        }
    }
}

/// Vacancy hop in a 500-site fcc cell (499 atoms). Six images is two
/// per Cartesian dimension. The clock covers endpoint relaxation, the
/// initial path, and the FIRE band.
pub(crate) fn large_vacancy(csv: &mut Vec<String>) -> Result<(), SaddleError> {
    let nn = 2.0_f64.powf(1.0 / 6.0);
    let a = nn * 2.0_f64.sqrt();
    let ncell = 5usize;
    let length = ncell as f64 * a;
    let cutoff = 2.5 * nn;
    let basis = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ];
    let mut sites = Vec::new();
    for i in 0..ncell {
        for j in 0..ncell {
            for k in 0..ncell {
                for b in basis {
                    sites.push([
                        a * (i as f64 + b[0]),
                        a * (j as f64 + b[1]),
                        a * (k as f64 + b[2]),
                    ]);
                }
            }
        }
    }
    let vacancy = ncell / 2 * ncell * ncell * 4;
    let hopper = vacancy + 1;
    let mut reactant = Array1::zeros(3 * (sites.len() - 1));
    let mut product = reactant.clone();
    let mut slot = 0;
    for (index, site) in sites.iter().enumerate() {
        if index == vacancy {
            continue;
        }
        for k in 0..3 {
            reactant[3 * slot + k] = site[k];
            product[3 * slot + k] = if index == hopper {
                sites[vacancy][k]
            } else {
                site[k]
            };
        }
        slot += 1;
    }
    let pot = Pot::LjCrystal { length, cutoff };
    let started = Instant::now();
    minimize(&mut reactant, &pot, None)?;
    minimize(&mut product, &pot, None)?;
    let relaxed = started.elapsed().as_secs_f64();
    let mic = linkcell::Cell::ortho(length, length, length)
        .map_err(|e| SaddleError::Invalid(e.to_string()))?;
    let path_started = Instant::now();
    // Two images along each Cartesian direction, endpoints included.
    let n_images = 6;
    let built = interpolated_path(
        reactant.view(),
        product.view(),
        n_images,
        Some(&mic),
        &IdppConfig::default(),
    )?;
    let path_seconds = path_started.elapsed().as_secs_f64();
    let band_cell = BandCell([[length, 0.0, 0.0], [0.0, length, 0.0], [0.0, 0.0, length]]);
    let case = Problem {
        name: "lj-vacancy-499",
        path_kind: if built.idpp { "idpp-fb" } else { "linear" },
        held_out: false,
        positions: built.positions,
        cell: Some(band_cell),
        translate: true,
        pot,
    };
    let mut config = band_config(&case, SpringKind::Uniform { k: K_SPRING });
    config.method = rgmin::Method::Fire {
        kind: rgmin::FireKind::V1,
    };
    let surface = Counted {
        pot: case.pot.clone(),
        cell: case.cell,
        calls: AtomicUsize::new(0),
    };
    let mut session = BandSession::new(config, case.positions.clone())?;
    let band_started = Instant::now();
    let mut ci = None;
    let mut max_atom = f64::INFINITY;
    let mut steps = 0usize;
    // Image L2 grows with sqrt(N). The large-cell stop is the largest
    // per-atom projected force, eOn's max_atom metric, at the same 0.01.
    for _ in 0..400 {
        let report = session.step(&surface)?;
        steps += 1;
        ci = report.ci_index;
        let Some(ev) = session.evaluation() else {
            break;
        };
        max_atom = 0.0;
        for row in ev.projected.outer_iter() {
            let n = row.len() / 3;
            for atom in 0..n {
                let nrm =
                    (row[3 * atom].powi(2) + row[3 * atom + 1].powi(2) + row[3 * atom + 2].powi(2))
                        .sqrt();
                max_atom = max_atom.max(nrm);
            }
        }
        if max_atom < TOL {
            break;
        }
    }
    let band_seconds = band_started.elapsed().as_secs_f64();
    let calls = surface.calls.load(Ordering::Relaxed);
    let dof = 3 * (sites.len() - 1);
    let per = band_seconds / (n_images as f64 * dof as f64);
    println!(
        "large N={n} images={n_images} (2 per dimension) dof={dof} path={} relaxed_s={relaxed:.3} path_s={path_seconds:.3} steps={steps} fire_calls={calls} fire_s={band_seconds:.3} s/(image*dof)={per:.3e} max_atom={max_atom:.3e} ci={ci:?} pipeline_s={pipeline:.3}",
        case.path_kind,
        n = sites.len() - 1,
        pipeline = relaxed + path_seconds + band_seconds,
    );
    csv.push(format!(
        "lj-vacancy-499,ci-fire,{calls},{band_seconds:.6},{},{ci},0",
        max_atom < TOL,
        ci = ci.map(|i| i.to_string()).unwrap_or_default(),
    ));
    Ok(())
}

/// Harmonic chain, so the clock is the band and not an O(N²) potential.

pub(crate) fn profile_band() {
    let n_atoms = 499usize;
    let n_images = 6usize;
    let dof = 3 * n_atoms;
    let mut positions = Array2::zeros((n_images, dof));
    for i in 0..n_images {
        for a in 0..n_atoms {
            positions[(i, 3 * a)] = a as f64 * 1.2 + 0.01 * i as f64;
        }
    }
    struct Harmonic {
        nanos: std::sync::atomic::AtomicU64,
    }
    impl BandSurface for Harmonic {
        fn eval(
            &self,
            positions: ArrayView2<f64>,
            energies: &mut Array1<f64>,
            gradients: &mut Array2<f64>,
        ) -> Result<(), SaddleError> {
            let t0 = Instant::now();
            for (i, row) in positions.outer_iter().enumerate() {
                let mut e = 0.0;
                for k in 0..row.len() {
                    let x = row[k];
                    e += x * x;
                    gradients[(i, k)] = 2.0 * x;
                }
                energies[i] = e;
            }
            self.nanos
                .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            Ok(())
        }
    }
    let surface = Harmonic {
        nanos: std::sync::atomic::AtomicU64::new(0),
    };
    let config = BandConfig {
        climbing: Some(CiConfig::always()),
        force_tol: 1e-8,
        max_move: 0.2,
        time_step: DT,
        min_mode: None,
        method: rgmin::Method::Fire {
            kind: rgmin::FireKind::V1,
        },
        ..BandConfig::default()
    };
    let mut session = BandSession::new(config, positions).unwrap();
    let started = Instant::now();
    for _ in 0..40 {
        session.step(&surface).unwrap();
    }
    let total = started.elapsed().as_secs_f64();
    let surface_s = surface.nanos.load(Ordering::Relaxed) as f64 / 1e9;
    println!(
        "profile 499-atom harmonic 6 images 40 steps: total_s={total:.4} surface_s={surface_s:.4} band_s={:.4}",
        total - surface_s
    );
}

pub(crate) fn pt_adatom(csv: &mut Vec<String>) -> Result<(), SaddleError> {
    // 10 by 10 fcc(111), three layers: 300 slab atoms plus the adatom.
    let nn = 2.897;
    let a1 = [nn, 0.0, 0.0];
    let a2 = [nn / 2.0, nn * 3.0_f64.sqrt() / 2.0, 0.0];
    let h = nn * (2.0_f64 / 3.0).sqrt();
    let mut slab = Vec::new();
    for layer in 0..3 {
        for i in 0..10 {
            for j in 0..10 {
                slab.push([
                    i as f64 * a1[0] + j as f64 * a2[0],
                    i as f64 * a1[1] + j as f64 * a2[1],
                    layer as f64 * h,
                ]);
            }
        }
    }
    let hollow = |i: f64, j: f64, shift: f64| {
        [
            (i + shift) * a1[0] + (j + shift) * a2[0],
            (i + shift) * a1[1] + (j + shift) * a2[1],
            3.0 * h,
        ]
    };
    let pot = Pot::MorseSlab { slab: slab.clone() };
    let mut reactant = Array1::from_vec(hollow(4.0, 4.0, 1.0 / 3.0).to_vec());
    let mut product = Array1::from_vec(hollow(4.0, 4.0, 2.0 / 3.0).to_vec());
    minimize(&mut reactant, &pot, None)?;
    minimize(&mut product, &pot, None)?;
    let built = interpolated_path(
        reactant.view(),
        product.view(),
        7,
        None,
        &IdppConfig::default(),
    )?;
    let mut case = Problem {
        name: "pt111-adatom",
        path_kind: if built.idpp { "idpp-fb" } else { "linear" },
        held_out: true,
        positions: built.positions,
        cell: None,
        translate: false,
        pot,
    };
    println!(
        "setup pt111-adatom: {} slab atoms, images {}, spring {K_SPRING}, tol {TOL}, norm per-image, path {}",
        slab.len(),
        case.positions.nrows(),
        case.path_kind
    );
    for optim in [Optim::Fire, Optim::Lbfgs] {
        let eon = run_eon(&case, optim);
        let rgs = run_rgsaddle(&case, optim, false);
        match (&eon, &rgs) {
            (Ok(eon), Ok(rgs)) => {
                println!(
                    "{:<16} {:<8} {:<8} {:>10} {:>10.3} {:>10} {:>10.3} {:>8} {:>8}",
                    case.name,
                    optim.label(),
                    case.path_kind,
                    eon.calls,
                    eon.seconds,
                    rgs.calls,
                    rgs.seconds,
                    eon.ci.map(|i| i.to_string()).unwrap_or_else(|| "-".into()),
                    rgs.ci.map(|i| i.to_string()).unwrap_or_else(|| "-".into()),
                );
                push_csv(
                    csv,
                    &case,
                    &format!("eon-{}", optim.label()),
                    eon.calls,
                    eon.seconds,
                    eon.converged,
                    eon.ci,
                );
                push_csv(
                    csv,
                    &case,
                    &format!("ci-{}", optim.label()),
                    rgs.calls,
                    rgs.seconds,
                    rgs.converged,
                    rgs.ci,
                );
            }
            (Err(err), _) => println!("{:<16} {:<8} eon failed: {err}", case.name, optim.label()),
            (Ok(_), Err(err)) => println!(
                "{:<16} {:<8} rgsaddle failed: {err}",
                case.name,
                optim.label()
            ),
        }
    }
    survey(&mut case, csv);
    Ok(())
}

/// Adatom hop from the lower terrace, across the step edge, onto the
/// upper terrace. Eleven images, the same spring, tolerance and
/// per-image norm as the flat terrace, from [`interpolated_path`].
pub(crate) fn pt_step(csv: &mut Vec<String>) -> Result<(), SaddleError> {
    let nn = 2.897;
    let a1 = [nn, 0.0, 0.0];
    let a2 = [nn / 2.0, nn * 3.0_f64.sqrt() / 2.0, 0.0];
    let h = nn * (2.0_f64 / 3.0).sqrt();
    let mut slab = Vec::new();
    for layer in 0..3 {
        for i in 0..10 {
            for j in 0..10 {
                slab.push([
                    i as f64 * a1[0] + j as f64 * a2[0],
                    i as f64 * a1[1] + j as f64 * a2[1],
                    layer as f64 * h,
                ]);
            }
        }
    }
    // Partial fourth layer: the upper terrace, five rows, leaving the
    // other half as the lower terrace.
    for i in 0..10 {
        for j in 0..5 {
            slab.push([
                i as f64 * a1[0] + j as f64 * a2[0],
                i as f64 * a1[1] + j as f64 * a2[1],
                3.0 * h,
            ]);
        }
    }
    let site = |i: f64, j: f64, z: f64| {
        let shift = 1.0 / 3.0;
        [
            (i + shift) * a1[0] + (j + shift) * a2[0],
            (i + shift) * a1[1] + (j + shift) * a2[1],
            z,
        ]
    };
    let pot = Pot::MorseSlab { slab: slab.clone() };
    // Lower terrace, one row past the edge, and the upper terrace
    // just onto the island. The straight path crosses the step edge.
    let mut reactant = Array1::from_vec(site(4.0, 6.0, 3.0 * h).to_vec());
    let mut product = Array1::from_vec(site(4.0, 3.0, 4.0 * h).to_vec());
    minimize(&mut reactant, &pot, None)?;
    minimize(&mut product, &pot, None)?;
    let n_images = 11;
    let built = interpolated_path(
        reactant.view(),
        product.view(),
        n_images,
        None,
        &IdppConfig::default(),
    )?;
    let case = Problem {
        name: "pt111-step",
        path_kind: if built.idpp { "idpp-fb" } else { "linear" },
        held_out: false,
        positions: built.positions,
        cell: None,
        translate: false,
        pot,
    };
    println!(
        "setup pt111-step: {} slab atoms, images {n_images}, spring {K_SPRING}, tol {TOL}, norm per-image, path {}",
        slab.len(),
        case.path_kind
    );
    for optim in [Optim::Fire, Optim::Lbfgs] {
        let eon = run_eon(&case, optim);
        let rgs = run_rgsaddle(&case, optim, false);
        match (&eon, &rgs) {
            (Ok(eon), Ok(rgs)) => {
                println!(
                    "{:<16} {:<8} {:<8} {:>10} {:>10.3} {:>10} {:>10.3} {:>8} {:>8}",
                    case.name,
                    optim.label(),
                    case.path_kind,
                    eon.calls,
                    eon.seconds,
                    rgs.calls,
                    rgs.seconds,
                    eon.ci.map(|i| i.to_string()).unwrap_or_else(|| "-".into()),
                    rgs.ci.map(|i| i.to_string()).unwrap_or_else(|| "-".into()),
                );
                push_csv(
                    csv,
                    &case,
                    &format!("eon-{}", optim.label()),
                    eon.calls,
                    eon.seconds,
                    eon.converged,
                    eon.ci,
                );
                push_csv(
                    csv,
                    &case,
                    &format!("ci-{}", optim.label()),
                    rgs.calls,
                    rgs.seconds,
                    rgs.converged,
                    rgs.ci,
                );
            }
            (Err(err), _) => println!("{:<16} {:<8} eon failed: {err}", case.name, optim.label()),
            (Ok(_), Err(err)) => println!(
                "{:<16} {:<8} rgsaddle failed: {err}",
                case.name,
                optim.label()
            ),
        }
    }
    survey(&case, csv);
    Ok(())
}
