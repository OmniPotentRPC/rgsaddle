//! Bitwise trajectories of the public saddle searches on the analytic
//! surfaces the other tests already use.
//!
//! The digest is SHA-256 over the little-endian `to_bits` of every
//! recorded coordinate, energy, force and curvature, in the order the
//! searches below emit them. A refactor that keeps those bits keeps
//! this digest.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, array};
use rgsaddle::{
    ArcPath, BandConfig, BandSession, BandSurface, MinModeConfig, MinModeKind, MinModeSession,
    PafiConfig, PointSurface, SaddleError, pafi_profile,
};
use sha2::{Digest, Sha256};

/// `V = (x^2 - 1)^2 + 2 y^2 + 2 z^2`.
struct DoubleWell;

impl BandSurface for DoubleWell {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (i, row) in positions.outer_iter().enumerate() {
            let (x, y, z) = (row[0], row[1], row[2]);
            energies[i] = (x * x - 1.0).powi(2) + 2.0 * y * y + 2.0 * z * z;
            gradients[(i, 0)] = 4.0 * x * (x * x - 1.0);
            gradients[(i, 1)] = 4.0 * y;
            gradients[(i, 2)] = 4.0 * z;
        }
        Ok(())
    }
}

/// `V = (x^2 - 1)^2 + 2 y^2` on the two coordinates PAFI walks.
struct Widening;

impl PointSurface for Widening {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (a, b) = (x[0], x[1]);
        let energy = (a * a - 1.0).powi(2) + 2.0 * b * b;
        Ok((energy, array![4.0 * a * (a * a - 1.0), 4.0 * b]))
    }
}

/// `V = -x^2 + y^2 + z^2`.
struct QuadraticSaddle;

impl PointSurface for QuadraticSaddle {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let energy = -x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
        Ok((energy, array![-2.0 * x[0], 2.0 * x[1], 2.0 * x[2]]))
    }
}

struct Dig {
    hash: Sha256,
    lines: Vec<String>,
}

impl Dig {
    fn new() -> Self {
        Self {
            hash: Sha256::new(),
            lines: Vec::new(),
        }
    }

    fn mark(&mut self, name: &str) {
        self.hash.update(name.as_bytes());
        self.lines.push(format!("# {name}"));
    }

    fn f64(&mut self, label: &str, value: f64) {
        self.hash.update(value.to_bits().to_le_bytes());
        self.lines
            .push(format!("{label} {value:.16e} {:016x}", value.to_bits()));
    }

    fn row(&mut self, label: &str, values: impl IntoIterator<Item = f64>) {
        for (i, value) in values.into_iter().enumerate() {
            self.f64(&format!("{label}[{i}]"), value);
        }
    }

    fn finish(&self) -> String {
        hex(&self.hash.clone().finalize())
    }
}

fn hex(bytes: &[u8]) -> String {
    const LUT: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(LUT[(byte >> 4) as usize] as char);
        out.push(LUT[(byte & 0xf) as usize] as char);
    }
    out
}

fn band(n_images: usize) -> Array2<f64> {
    let mut positions = Array2::zeros((n_images, 3));
    for i in 0..n_images {
        let t = i as f64 / (n_images - 1) as f64;
        positions[(i, 0)] = -1.0 + 2.0 * t;
        positions[(i, 1)] = 0.3 * (std::f64::consts::PI * t).sin();
    }
    positions
}

fn record_band(dig: &mut Dig, session: &BandSession) {
    let positions = session.positions();
    for (i, row) in positions.outer_iter().enumerate() {
        dig.row(&format!("x{i}"), row.iter().copied());
        let (x, y, z) = (row[0], row[1], row[2]);
        let energy = (x * x - 1.0).powi(2) + 2.0 * y * y + 2.0 * z * z;
        dig.f64(&format!("e{i}"), energy);
    }
}

fn neb(dig: &mut Dig, name: &str, climbing: bool) {
    dig.mark(name);
    let config = BandConfig::default()
        .with_force_tol(1e-8)
        .with_max_move(0.1)
        .with_min_mode(None)
        .with_saddle_check(None)
        .with_climbing(climbing.then(rgsaddle::CiConfig::default));
    let mut session = BandSession::new(config, band(7)).expect(name);
    for step in 0..12 {
        let report = session.step(&DoubleWell).expect(name);
        dig.mark(&format!("{name}-step{step}"));
        dig.f64("max_force", report.max_force);
        record_band(dig, &session);
    }
}

fn mode(dig: &mut Dig, name: &str, kind: MinModeKind) {
    dig.mark(name);
    let config = MinModeConfig::default()
        .with_kind(kind)
        .with_force_tol(1e-6)
        .with_max_move(0.1)
        .with_saddle_check(None);
    let mut session =
        MinModeSession::new(config, array![0.35, 0.4, -0.3], array![0.8, 0.5, 0.1]).expect(name);
    for step in 0..8 {
        let report = session.step(&QuadraticSaddle).expect(name);
        dig.mark(&format!("{name}-step{step}"));
        dig.f64("max_force", report.max_force);
        dig.f64("curvature", report.curvature);
        let position = session.position();
        dig.row("x", position.iter().copied());
        let energy =
            -position[0] * position[0] + position[1] * position[1] + position[2] * position[2];
        dig.f64("energy", energy);
    }
}

fn pafi(dig: &mut Dig) {
    dig.mark("pafi");
    let frames: Vec<Array1<f64>> = (0..5)
        .map(|i| {
            let x = -1.0 + i as f64 / 4.0;
            array![x, 0.15 * (1.0 - x * x)]
        })
        .collect();
    let path = ArcPath::from_positions(&frames).expect("pafi path");
    let arcs = path.arcs();
    let config = PafiConfig::default()
        .with_thermal(0.02)
        .with_mobility(1.0)
        .with_time_step(0.01)
        .with_thermalize(20)
        .with_stride(2)
        .with_samples(64)
        .with_blocks(4)
        .with_seed(11)
        .with_fix_com(false)
        .with_retain(2)
        .with_target_error(None)
        .with_max_rounds(1);
    let (profile, planes) = pafi_profile(&Widening, &path, &arcs, &config).expect("pafi");
    dig.f64("barrier", profile.barrier);
    dig.row("free_energy", profile.free_energy.iter().copied());
    dig.row("gradient", profile.gradient.iter().copied());
    for (i, plane) in planes.iter().enumerate() {
        dig.f64(&format!("plane{i}-gradient"), plane.gradient);
        dig.f64(&format!("plane{i}-psi"), plane.psi);
        dig.f64(&format!("plane{i}-error"), plane.error);
        for (k, kept) in plane.retained.iter().enumerate() {
            dig.row(&format!("plane{i}-kept{k}"), kept.iter().copied());
        }
    }
}

fn trajectory() -> (String, String) {
    let mut dig = Dig::new();
    neb(&mut dig, "neb", false);
    neb(&mut dig, "ci-neb", true);
    mode(&mut dig, "dimer", MinModeKind::Dimer);
    mode(&mut dig, "lanczos", MinModeKind::Lanczos);
    pafi(&mut dig);
    (dig.finish(), dig.lines.join("\n"))
}

/// SHA-256 of the baseline dump at develop e799819, before any refactor
/// in this series.
///
/// Measured with the same command under default and
/// `readcon,readcon-db`; both feature sets produce this same digest. The
/// PAFI section of the dump depends on `pafi_profile`, so a change to the
/// PAFI walk moves the digest and has to re-measure it.
const EXPECTED: &str = "c4caf5f773a618cb3856b0c521236fb2810887dec5afc9b9668b4ae37b85429b";

#[test]
fn trajectories_match_the_baseline_bits() {
    let (digest, dump) = trajectory();
    if let Ok(path) = std::env::var("DRY_BITS_OUT") {
        std::fs::write(&path, format!("{digest}\n{dump}\n")).expect("write dump");
    }
    assert_eq!(digest, EXPECTED, "trajectory bits moved");
}
