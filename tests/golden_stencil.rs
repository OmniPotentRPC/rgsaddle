//! Golden hashes of positions and energies after a fixed step count.
//!
//! Recorded on main before the stencil refactor. Each line is
//! `preset fixture threads sha256`. The digest is SHA-256 over the
//! raw little-endian f64 bytes of the final positions, row-major,
//! followed by the energies at those positions. Threads are 1, 2, 3
//! and 8. A pure refactor must reproduce every digest. The
//! Müller-Brown dimer digest is the path after the rotation stops on
//! a resolved mode instead of extrapolating a right angle.

use std::fs;
use std::path::PathBuf;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rayon::ThreadPoolBuilder;
use rgsaddle::{
    AutoNeb, AutoNebTuning, BandConfig, BandSession, BandSurface, CiConfig, FiniteDifference,
    MinModeConfig, MinModeKind, MinModeSession, PointSurface, Preset, ProjectionKind, SaddleError,
    SpringKind, Stencil, TangentKind,
};
use sha2::{Digest, Sha256};

const THREADS: [usize; 4] = [1, 2, 3, 8];
const BAND_STEPS: usize = 4;
const MODE_STEPS: usize = 3;

fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/stencil.txt")
}

fn hash_bytes(positions: ArrayView2<f64>, energies: &[f64]) -> String {
    let mut hasher = Sha256::new();
    for value in positions.iter() {
        hasher.update(value.to_le_bytes());
    }
    for value in energies {
        hasher.update(value.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn on_pool<T: Send>(threads: usize, body: impl FnOnce() -> T + Send) -> T {
    let pool = ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("thread pool");
    pool.install(body)
}

struct DoubleWell;

fn double_well_energy(x: f64, y: f64, z: f64) -> (f64, [f64; 3]) {
    let energy = (x * x - 1.0).powi(2) + 2.0 * y * y + 2.0 * z * z;
    let gradient = [4.0 * x * (x * x - 1.0), 4.0 * y, 4.0 * z];
    (energy, gradient)
}

impl BandSurface for DoubleWell {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (i, row) in positions.outer_iter().enumerate() {
            let (energy, gradient) = double_well_energy(row[0], row[1], row[2]);
            energies[i] = energy;
            gradients[(i, 0)] = gradient[0];
            gradients[(i, 1)] = gradient[1];
            gradients[(i, 2)] = gradient[2];
        }
        Ok(())
    }
}

impl PointSurface for DoubleWell {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (energy, gradient) = double_well_energy(x[0], x[1], x[2]);
        Ok((energy, Array1::from_vec(gradient.to_vec())))
    }
}

const MB_A: [f64; 4] = [-200.0, -100.0, -170.0, 15.0];
const MB_AA: [f64; 4] = [-1.0, -1.0, -6.5, 0.7];
const MB_B: [f64; 4] = [0.0, 0.0, 11.0, 0.6];
const MB_C: [f64; 4] = [-10.0, -10.0, -6.5, 0.7];
const MB_X0: [f64; 4] = [1.0, 0.0, -0.5, -1.0];
const MB_Y0: [f64; 4] = [0.0, 0.5, 1.5, 1.0];

fn muller_brown(x: f64, y: f64) -> (f64, [f64; 2]) {
    let mut energy = 0.0;
    let mut gx = 0.0;
    let mut gy = 0.0;
    for i in 0..4 {
        let dx = x - MB_X0[i];
        let dy = y - MB_Y0[i];
        let exponent = MB_AA[i] * dx * dx + MB_B[i] * dx * dy + MB_C[i] * dy * dy;
        let weight = MB_A[i] * exponent.exp();
        energy += weight;
        gx += weight * (2.0 * MB_AA[i] * dx + MB_B[i] * dy);
        gy += weight * (MB_B[i] * dx + 2.0 * MB_C[i] * dy);
    }
    (energy, [gx, gy])
}

struct MullerBrown;

impl BandSurface for MullerBrown {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (i, row) in positions.outer_iter().enumerate() {
            let (energy, gradient) = muller_brown(row[0], row[1]);
            energies[i] = energy;
            gradients[(i, 0)] = gradient[0];
            gradients[(i, 1)] = gradient[1];
            if gradients.ncols() > 2 {
                gradients[(i, 2)] = 0.0;
            }
        }
        Ok(())
    }
}

impl PointSurface for MullerBrown {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (energy, gradient) = muller_brown(x[0], x[1]);
        Ok((energy, array_from(&[gradient[0], gradient[1]])))
    }
}

/// Three atoms on a line. Neighbours interact by a harmonic bond of
/// length 1. The third coordinate of each atom is a stiff spring, so
/// the chain is a small atomistic path rather than a 1-D well.
struct HarmonicChain;

fn chain_eval(positions: ArrayView2<f64>) -> (Array1<f64>, Array2<f64>) {
    let rows = positions.nrows();
    let mut energies = Array1::zeros(rows);
    let mut gradients = Array2::zeros((rows, 9));
    for (i, row) in positions.outer_iter().enumerate() {
        let mut energy = 0.0;
        let mut g = [0.0; 9];
        for bond in 0..2 {
            let a = bond * 3;
            let b = a + 3;
            let dx = row[b] - row[a];
            let dy = row[b + 1] - row[a + 1];
            let dz = row[b + 2] - row[a + 2];
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            let delta = r - 1.0;
            energy += delta * delta;
            let scale = if r > 1e-15 { 2.0 * delta / r } else { 0.0 };
            let fx = scale * dx;
            let fy = scale * dy;
            let fz = scale * dz;
            g[a] -= fx;
            g[a + 1] -= fy;
            g[a + 2] -= fz;
            g[b] += fx;
            g[b + 1] += fy;
            g[b + 2] += fz;
        }
        for atom in 0..3 {
            let z = row[atom * 3 + 2];
            energy += 4.0 * z * z;
            g[atom * 3 + 2] += 8.0 * z;
        }
        energies[i] = energy;
        for k in 0..9 {
            gradients[(i, k)] = g[k];
        }
    }
    (energies, gradients)
}

impl BandSurface for HarmonicChain {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        let (e, g) = chain_eval(positions);
        energies.assign(&e);
        gradients.assign(&g);
        Ok(())
    }
}

impl PointSurface for HarmonicChain {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let mut row = Array2::zeros((1, x.len()));
        row.row_mut(0).assign(&x);
        let (e, g) = chain_eval(row.view());
        Ok((e[0], g.row(0).to_owned()))
    }
}

fn array_from(values: &[f64]) -> Array1<f64> {
    Array1::from_vec(values.to_vec())
}

fn lerp_band(start: &[f64], end: &[f64], n_images: usize) -> Array2<f64> {
    let dof = start.len();
    let mut band = Array2::zeros((n_images, dof));
    for i in 0..n_images {
        let t = i as f64 / (n_images - 1) as f64;
        for k in 0..dof {
            band[(i, k)] = (1.0 - t) * start[k] + t * end[k];
        }
    }
    band
}

fn double_well_band() -> Array2<f64> {
    let n = 7;
    let mut band = Array2::zeros((n, 3));
    for i in 0..n {
        let t = i as f64 / (n - 1) as f64;
        band[(i, 0)] = -1.0 + 2.0 * t;
        band[(i, 1)] = 0.3 * (std::f64::consts::PI * t).sin();
    }
    band
}

fn muller_band() -> Array2<f64> {
    lerp_band(&[-0.558, 1.442, 0.0], &[0.623, 0.028, 0.0], 5)
}

fn chain_band() -> Array2<f64> {
    lerp_band(
        &[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        &[1.0, 0.2, 0.0, 0.0, 0.4, 0.0, -1.0, 0.0, 0.0],
        5,
    )
}

fn band_config(preset: &str) -> BandConfig {
    let mut config = BandConfig {
        tangent: TangentKind::Improved,
        spring: SpringKind::Uniform { k: 5.0 },
        projection: ProjectionKind::Neb,
        climbing: None,
        force_tol: 1e-8,
        max_move: 0.2,
        method: rgmin::Method::Fire {
            kind: rgmin::FireKind::V2,
        },
        ..BandConfig::default()
    };
    match preset {
        "neb" => {}
        "ci-neb" => {
            config.climbing = Some(CiConfig {
                trigger_factor: 0.5,
                trigger_force: 0.0,
            });
        }
        "dneb" => {
            config.projection = ProjectionKind::DoublyNudged;
        }
        "weighted-neb" => {
            config.spring = SpringKind::Weighted {
                ks: vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0],
            };
            config.climbing = Some(CiConfig {
                trigger_factor: 0.5,
                trigger_force: 0.0,
            });
        }
        other => panic!("unknown band preset {other}"),
    }
    config
}

fn run_band(preset: &str, fixture: &str) -> (Array2<f64>, Vec<f64>) {
    let initial = match fixture {
        "double-well" => double_well_band(),
        "muller-brown" => muller_band(),
        "harmonic-chain" => chain_band(),
        other => panic!("unknown fixture {other}"),
    };
    let mut config = band_config(preset);
    if preset == "weighted-neb" {
        let n = initial.nrows();
        config.spring = SpringKind::Weighted {
            ks: (0..n - 1).map(|i| 1.0 + i as f64).collect(),
        };
    }
    let mut session = BandSession::new(config, initial.clone()).expect("band");
    match fixture {
        "double-well" => {
            session.run(&DoubleWell, BAND_STEPS).expect("band steps");
        }
        "muller-brown" => {
            session.run(&MullerBrown, BAND_STEPS).expect("band steps");
        }
        "harmonic-chain" => {
            session.run(&HarmonicChain, BAND_STEPS).expect("band steps");
        }
        _ => unreachable!(),
    }
    let positions = session.positions().to_owned();
    let energies = energies_of(fixture, positions.view());
    (positions, energies)
}

fn energies_of(fixture: &str, positions: ArrayView2<f64>) -> Vec<f64> {
    let n = positions.nrows();
    let mut energies = Array1::zeros(n);
    let mut gradients = Array2::zeros(positions.dim());
    match fixture {
        "double-well" => {
            BandSurface::eval(&DoubleWell, positions, &mut energies, &mut gradients).unwrap()
        }
        "muller-brown" => {
            BandSurface::eval(&MullerBrown, positions, &mut energies, &mut gradients).unwrap()
        }
        "harmonic-chain" => {
            BandSurface::eval(&HarmonicChain, positions, &mut energies, &mut gradients).unwrap()
        }
        _ => unreachable!(),
    }
    energies.to_vec()
}

fn mode_config(preset: &str) -> MinModeConfig {
    MinModeConfig {
        kind: match preset {
            "dimer" => MinModeKind::Dimer,
            "lanczos" => MinModeKind::Lanczos,
            other => panic!("unknown mode preset {other}"),
        },
        dr: 1e-3,
        difference: FiniteDifference::Forward,
        rotation_tol: 1e-4,
        rotation_angle_tol: 0.0,
        max_rotations: 8,
        krylov_dim: 6,
        force_tol: 1e-12,
        max_move: 0.1,
        method: rgmin::Method::Fire {
            kind: rgmin::FireKind::V2,
        },
    }
}

fn run_mode(preset: &str, fixture: &str) -> (Array2<f64>, Vec<f64>) {
    let (x0, mode) = match fixture {
        "double-well" => (vec![0.4, 0.25, 0.05], vec![1.0, 0.0, 0.0]),
        "muller-brown" => (vec![-0.7, 0.5], vec![1.0, 0.0]),
        "harmonic-chain" => (
            vec![-1.0, 0.1, 0.0, 0.05, 0.3, 0.02, 1.0, -0.1, 0.0],
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        other => panic!("unknown fixture {other}"),
    };
    let config = mode_config(preset);
    let mut session =
        MinModeSession::new(config, array_from(&x0), array_from(&mode)).expect("minmode");
    match fixture {
        "double-well" => {
            session.run(&DoubleWell, MODE_STEPS).expect("mode steps");
        }
        "muller-brown" => {
            session.run(&MullerBrown, MODE_STEPS).expect("mode steps");
        }
        "harmonic-chain" => {
            session.run(&HarmonicChain, MODE_STEPS).expect("mode steps");
        }
        _ => unreachable!(),
    }
    let x = session.position().to_owned();
    let positions = x.to_shape((1, x.len())).unwrap().to_owned();
    let energies = energies_of(fixture, positions.view());
    (positions, energies)
}

fn run_autoneb(fixture: &str, threads: usize) -> (Array2<f64>, Vec<f64>) {
    let initial = match fixture {
        "double-well" => lerp_band(&[-1.0, 0.0, 0.0], &[1.0, 0.0, 0.0], 2),
        "muller-brown" => lerp_band(&[-0.558, 1.442, 0.0], &[0.623, 0.028, 0.0], 2),
        "harmonic-chain" => lerp_band(
            &[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
            2,
        ),
        other => panic!("unknown fixture {other}"),
    };
    let config = BandConfig {
        force_tol: 1e-3,
        max_move: 0.1,
        method: rgmin::Method::Fire {
            kind: rgmin::FireKind::V2,
        },
        spring: SpringKind::Uniform { k: 5.0 },
        climbing: Some(CiConfig {
            trigger_factor: 0.5,
            trigger_force: 0.0,
        }),
        autoneb: Some(AutoNebTuning {
            n_simul: 1,
            n_max: 4,
            local_force_tol: 1.0,
            local_max_steps: 2,
            climb_max_steps: 2,
            space_energy_ratio: 0.5,
            climb: true,
            full_band: false,
            threads,
            record_samples: false,
        }),
        ..BandConfig::default()
    };
    let mut search = AutoNeb::new(config, initial).expect("autoneb");
    match fixture {
        "double-well" => {
            search.run(&DoubleWell).expect("autoneb");
        }
        "muller-brown" => {
            search.run(&MullerBrown).expect("autoneb");
        }
        "harmonic-chain" => {
            search.run(&HarmonicChain).expect("autoneb");
        }
        _ => unreachable!(),
    }
    let positions = search.positions().to_owned();
    let energies = energies_of(fixture, positions.view());
    (positions, energies)
}

fn cases() -> Vec<(&'static str, &'static str)> {
    let fixtures = ["double-well", "muller-brown", "harmonic-chain"];
    let mut out = Vec::new();
    for preset in [
        "neb",
        "ci-neb",
        "dneb",
        "weighted-neb",
        "dimer",
        "lanczos",
        "autoneb",
    ] {
        for fixture in fixtures {
            out.push((preset, fixture));
        }
    }
    out
}

fn digest(preset: &str, fixture: &str, threads: usize) -> String {
    let (positions, energies) = if preset == "dimer" || preset == "lanczos" {
        run_mode(preset, fixture)
    } else if preset == "autoneb" {
        run_autoneb(fixture, threads)
    } else {
        run_band(preset, fixture)
    };
    hash_bytes(positions.view(), &energies)
}

fn collect_lines() -> Vec<String> {
    let mut lines = Vec::new();
    for (preset, fixture) in cases() {
        for threads in THREADS {
            let hash = on_pool(threads, || digest(preset, fixture, threads));
            lines.push(format!("{preset} {fixture} {threads} {hash}"));
        }
    }
    lines
}

#[test]
fn dimer_keeps_the_resolved_muller_brown_mode() {
    // One rotation brings the rotational force under 1. The next
    // Kastner-Sherwood trial is about 1e-3 rad and the Fourier fit asks
    // for a right angle. Measuring that step rejects it, so the estimate
    // stays on the negative mode (curvature near -113) instead of the
    // extrapolated curvature near -4000, and it does not spend the budget.
    let mut session = MinModeSession::new(
        mode_config("dimer"),
        array_from(&[-0.7, 0.5]),
        array_from(&[1.0, 0.0]),
    )
    .unwrap();
    let estimate = session.estimate_mode(&MullerBrown).unwrap();
    assert!(
        estimate.rotations < 8,
        "rotations {} curvature {}",
        estimate.rotations,
        estimate.curvature
    );
    assert!(
        (-200.0..=-50.0).contains(&estimate.curvature),
        "curvature {} mode {:?}",
        estimate.curvature,
        estimate.mode
    );
}

#[test]
fn stencil_dimer_and_lanczos_match_the_recorded_bytes() {
    let recorded = fs::read_to_string(golden_path()).expect("golden file");
    for preset in ["dimer", "lanczos"] {
        for fixture in ["double-well", "muller-brown", "harmonic-chain"] {
            let (positions, energies) = run_stencil_mode(preset, fixture);
            let digest = hash_bytes(positions.view(), &energies);
            let prefix = format!("{preset} {fixture} 1 ");
            let line = recorded
                .lines()
                .find(|line| line.starts_with(&prefix))
                .unwrap_or_else(|| panic!("missing golden line {prefix}"));
            let want = line.rsplit(' ').next().expect("hash");
            assert_eq!(digest, want, "{preset} {fixture}");
            let (direct, direct_e) = run_mode(preset, fixture);
            assert_eq!(positions.len(), direct.len());
            for (got, reference) in positions.iter().zip(direct.iter()) {
                assert_eq!(got.to_le_bytes(), reference.to_le_bytes());
            }
            for (got, reference) in energies.iter().zip(direct_e.iter()) {
                assert_eq!(got.to_le_bytes(), reference.to_le_bytes());
            }
        }
    }
}

fn run_stencil_mode(preset: &str, fixture: &str) -> (Array2<f64>, Vec<f64>) {
    let (x0, mode) = match fixture {
        "double-well" => (vec![0.4, 0.25, 0.05], vec![1.0, 0.0, 0.0]),
        "muller-brown" => (vec![-0.7, 0.5], vec![1.0, 0.0]),
        "harmonic-chain" => (
            vec![-1.0, 0.1, 0.0, 0.05, 0.3, 0.02, 1.0, -0.1, 0.0],
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        other => panic!("unknown fixture {other}"),
    };
    let kind = match preset {
        "dimer" => Preset::Dimer,
        "lanczos" => Preset::Lanczos,
        other => panic!("unknown mode preset {other}"),
    };
    let mut search = Stencil::point(array_from(&x0))
        .preset(kind)
        .mode(array_from(&mode))
        .dr(1e-3)
        .max_rotations(8)
        .krylov_dim(6)
        .force_tol(1e-12)
        .max_move(0.1)
        .build()
        .expect("stencil");
    match fixture {
        "double-well" => {
            search.run(&DoubleWell, MODE_STEPS).expect("stencil steps");
        }
        "muller-brown" => {
            search.run(&MullerBrown, MODE_STEPS).expect("stencil steps");
        }
        "harmonic-chain" => {
            search
                .run(&HarmonicChain, MODE_STEPS)
                .expect("stencil steps");
        }
        _ => unreachable!(),
    }
    let x = search.position().to_owned();
    let positions = x.to_shape((1, x.len())).expect("row").to_owned();
    let energies = energies_of(fixture, positions.view());
    (positions, energies)
}

#[test]
fn golden_hashes_match_the_recorded_bytes() {
    let lines = collect_lines();
    let path = golden_path();
    if std::env::var_os("RGSADDLE_RECORD_GOLDEN").is_some() {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, lines.join("\n") + "\n").unwrap();
        return;
    }
    let recorded = fs::read_to_string(&path).unwrap_or_else(|err| {
        panic!(
            "missing {} ({err}); record it with RGSADDLE_RECORD_GOLDEN=1",
            path.display()
        )
    });
    let expected: Vec<&str> = recorded.lines().filter(|line| !line.is_empty()).collect();
    assert_eq!(
        lines.len(),
        expected.len(),
        "golden line count changed\n{}",
        lines.join("\n")
    );
    for (got, want) in lines.iter().zip(expected.iter()) {
        assert_eq!(got, want, "golden mismatch");
    }
    for (preset, fixture) in cases() {
        let hashes: Vec<&str> = lines
            .iter()
            .filter(|line| line.starts_with(&format!("{preset} {fixture} ")))
            .map(|line| line.rsplit(' ').next().unwrap())
            .collect();
        assert!(
            hashes.windows(2).all(|pair| pair[0] == pair[1]),
            "{preset} {fixture} changed with the thread count: {hashes:?}"
        );
    }
}
