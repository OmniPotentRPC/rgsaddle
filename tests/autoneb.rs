//! ASE AutoNEB on Müller-Brown, and the pure insertion decisions.

use std::process::Command;

use ndarray::{Array1, Array2, ArrayView2, array};
use rgsaddle::{
    AutoNeb, AutoNebTuning, BandConfig, BandStatus, BandSurface, CiConfig, SaddleError,
};

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

fn mb_index(x: f64, y: f64) -> i32 {
    let (_, _, h) = muller_brown(x, y);
    let tr = h[0][0] + h[1][1];
    let det = h[0][0] * h[1][1] - h[0][1] * h[1][0];
    let disc = (tr * tr - 4.0 * det).sqrt();
    let low = 0.5 * (tr - disc);
    let high = 0.5 * (tr + disc);
    i32::from(low < 0.0) + i32::from(high < 0.0)
}

#[derive(Clone, Copy)]
struct MullerBrown;

impl BandSurface for MullerBrown {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (i, row) in positions.outer_iter().enumerate() {
            let (energy, g, _) = muller_brown(row[0], row[1]);
            energies[i] = energy;
            gradients[(i, 0)] = g[0];
            gradients[(i, 1)] = g[1];
            gradients[(i, 2)] = 0.0;
        }
        Ok(())
    }
}

fn near_ts2(pos: ArrayView2<f64>, ci: usize) -> bool {
    let ts = [0.21248658, 0.29298833];
    let dist = ((pos[(ci, 0)] - ts[0]).powi(2) + (pos[(ci, 1)] - ts[1]).powi(2)).sqrt();
    dist < 0.05 && mb_index(pos[(ci, 0)], pos[(ci, 1)]) == 1 && pos[(ci, 2)].abs() < 1e-8
}

fn json_field<'a>(text: &'a str, key: &str) -> &'a str {
    let pat = format!("\"{key}\": ");
    let start = text
        .find(&pat)
        .unwrap_or_else(|| panic!("missing {key} in {text}"))
        + pat.len();
    let rest = &text[start..];
    if let Some(stripped) = rest.strip_prefix('[') {
        let end = stripped.find(']').unwrap();
        &stripped[..end]
    } else {
        let end = rest.find([',', '}']).unwrap();
        &rest[..end]
    }
}

fn json_ints(text: &str, key: &str) -> Vec<usize> {
    json_field(text, key)
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| part.trim().parse().unwrap())
        .collect()
}

fn json_floats(text: &str, key: &str) -> Vec<f64> {
    json_field(text, key)
        .split(',')
        .map(|part| part.trim().parse().unwrap())
        .collect()
}

#[test]
fn muller_brown_matches_an_ase_autoneb_run() {
    let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/ase_autoneb_mb.py");
    let run = Command::new("python3")
        .arg(&script)
        .output()
        .expect("run ASE AutoNEB");
    assert!(
        run.status.success(),
        "ASE AutoNEB failed: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    let text = String::from_utf8(run.stdout).unwrap();
    let dynamic = json_ints(&text, "dynamic");
    let ase_ci = json_floats(&text, "ci");
    let ts = [0.21248658_f64, 0.29298833];
    let ase_dist = ((ase_ci[0] - ts[0]).powi(2) + (ase_ci[1] - ts[1]).powi(2)).sqrt();
    assert!(ase_dist < 0.05, "ASE ci {ase_ci:?} distance {ase_dist}");
    assert!(ase_ci[2].abs() < 1e-8, "ASE z {}", ase_ci[2]);

    let band = BandConfig {
        force_tol: 0.05,
        max_move: 0.05,
        method: rgmin::Method::Fire {
            kind: rgmin::FireKind::V1,
        },
        spring: rgsaddle::SpringKind::Uniform { k: 5.0 },
        climbing: Some(CiConfig {
            trigger_factor: 1.0,
            trigger_force: f64::INFINITY,
        }),
        autoneb: Some(AutoNebTuning {
            n_simul: 3,
            n_max: 7,
            local_force_tol: 0.5,
            local_max_steps: 25,
            climb_max_steps: 80,
            space_energy_ratio: 0.5,
            climb: true,
            full_band: false,
            threads: 1,
            record_samples: false,
        }),
        ..BandConfig::default()
    };
    let initial = array![
        [-0.05001082, 0.46669410, 0.0],
        [0.62349940, 0.02803776, 0.0]
    ];
    let mut search = AutoNeb::new(band, initial).unwrap();
    let report = search.run(&MullerBrown).unwrap();
    let ci = report.ci_index;
    eprintln!(
        "rust status={:?} force={:.4} insertions={:?} ci={ci:?}",
        report.status,
        report.max_force,
        search.insertions(),
    );
    assert_eq!(
        report.status,
        BandStatus::Converged,
        "force {}",
        report.max_force
    );
    let ci = report.ci_index.expect("climbing image");
    assert!(near_ts2(search.positions(), ci), "{:?}", search.positions());
    // Two endpoints and n_simul = 3: three repeated midpoints. ASE logs
    // that fill as one "max length" at 0. FIRE 1.0 is ASE's original
    // FIRE (alpha start 0.1, Nmin 5). The dynamic indices match.
    assert_eq!(search.insertions()[..3], [0, 0, 2]);
    let rust_dynamic = search.insertions()[3..].to_vec();
    for cut in search.cuts() {
        eprintln!(
            "rust cut index={} energies={:?} lengths={:?}",
            cut.index, cut.energies, cut.lengths
        );
    }
    eprintln!("ASE dynamic={dynamic:?} rust dynamic={rust_dynamic:?}");
    assert_eq!(
        rust_dynamic, dynamic,
        "dynamic insertion sequence, ASE {dynamic:?} rust {rust_dynamic:?}"
    );
    let dx = search.positions()[(ci, 0)] - ase_ci[0];
    let dy = search.positions()[(ci, 1)] - ase_ci[1];
    let ci_gap = (dx * dx + dy * dy).sqrt();
    eprintln!("ci_gap={ci_gap:.6e}");
    assert!(ci_gap < 1e-3, "climbing images differ by {ci_gap}");
}

#[test]
fn ase_decision_functions_match_worst_segment_and_the_window() {
    let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/ase_autoneb_mb.py");
    let run = Command::new("python3")
        .arg(&script)
        .arg("--decisions")
        .output()
        .expect("ASE decision cases");
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    let text = String::from_utf8(run.stdout).unwrap();
    assert!(text.contains("\"window\": [1, 2, 3, 4, 5]"), "{text}");
    assert!(text.contains("\"window\": [2, 3, 4, 5, 6]"), "{text}");
    let energies = [0.0, 0.0, 10.0, 0.0, 0.0];
    assert_eq!(
        rgsaddle::autoneb::worst_segment(&[1.0, 1.0, 1.0, 1.0], &energies, 4.0, 0.5).unwrap(),
        1
    );
    assert_eq!(
        rgsaddle::autoneb::worst_segment(&[1.0, 1.0, 1.0, 10.0], &energies, 4.0, 0.5).unwrap(),
        3
    );
    assert!(text.contains("\"index\": 1"), "{text}");
    assert!(text.contains("\"index\": 3"), "{text}");
    assert!(text.contains("\"index\": 0"), "{text}");
}
