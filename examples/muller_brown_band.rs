//! Find the saddle between minima A and B of the Müller-Brown surface
//! twice: with the climbing-image nudged elastic band (CI-NEB), and with
//! the default band, which also refines its climbing image along the
//! lowest curvature mode (off-path climbing image NEB, OCI-NEB).
//!
//! ```text
//! cargo run --release --example muller_brown_band -- <out-dir>
//! ```
//!
//! For each method this writes `<method>_trace.csv`, one row per band
//! step, and `<method>_band.csv`, the converged images. The start path
//! is `initial_band.csv`. The summary goes to stdout.

use std::error::Error;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, Array2, ArrayView2, array};
use rgsaddle::{BandConfig, BandSession, BandStatus, BandSurface, SaddleError, linear_path};

const PREFACTOR: [f64; 4] = [-200.0, -100.0, -170.0, 15.0];
const XX: [f64; 4] = [-1.0, -1.0, -6.5, 0.7];
const XY: [f64; 4] = [0.0, 0.0, 11.0, 0.6];
const YY: [f64; 4] = [-10.0, -10.0, -6.5, 0.7];
const X0: [f64; 4] = [1.0, 0.0, -0.5, -1.0];
const Y0: [f64; 4] = [0.0, 0.5, 1.5, 1.0];

/// Energy and gradient of the Müller-Brown surface at (x, y).
fn muller_brown(x: f64, y: f64) -> (f64, f64, f64) {
    let (mut energy, mut gx, mut gy) = (0.0, 0.0, 0.0);
    for i in 0..4 {
        let (dx, dy) = (x - X0[i], y - Y0[i]);
        let term = PREFACTOR[i] * (XX[i] * dx * dx + XY[i] * dx * dy + YY[i] * dy * dy).exp();
        energy += term;
        gx += term * (2.0 * XX[i] * dx + XY[i] * dy);
        gy += term * (XY[i] * dx + 2.0 * YY[i] * dy);
    }
    (energy, gx, gy)
}

/// One atom in the plane z = 0. Each image evaluated is one force call.
#[derive(Default)]
struct MullerBrown {
    force_calls: AtomicUsize,
}

impl BandSurface for MullerBrown {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        self.force_calls
            .fetch_add(positions.nrows(), Ordering::Relaxed);
        for (i, row) in positions.outer_iter().enumerate() {
            let (energy, gx, gy) = muller_brown(row[0], row[1]);
            energies[i] = energy;
            gradients[(i, 0)] = gx;
            gradients[(i, 1)] = gy;
            gradients[(i, 2)] = 0.0;
        }
        Ok(())
    }
}

fn write_band(path: &Path, band: ArrayView2<f64>) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    writeln!(file, "image,x,y,energy")?;
    for (i, row) in band.outer_iter().enumerate() {
        let (energy, _, _) = muller_brown(row[0], row[1]);
        writeln!(file, "{i},{:.10},{:.10},{:.10}", row[0], row[1], energy)?;
    }
    Ok(())
}

/// Gradient norm at the highest interior image, climbing or not. This
/// is a diagnostic outside the force-call count.
fn top_gradient(band: ArrayView2<f64>) -> f64 {
    let energy = |i: usize| muller_brown(band[(i, 0)], band[(i, 1)]).0;
    let top = (1..band.nrows() - 1)
        .max_by(|&i, &j| energy(i).total_cmp(&energy(j)))
        .unwrap();
    let (_, gx, gy) = muller_brown(band[(top, 0)], band[(top, 1)]);
    gx.hypot(gy)
}

fn run(
    name: &str,
    config: BandConfig,
    initial: &Array2<f64>,
    out: &Path,
) -> Result<(), Box<dyn Error>> {
    let surface = MullerBrown::default();
    let mut session = BandSession::new(config, initial.clone())?;
    let mut trace = File::create(out.join(format!("{name}_trace.csv")))?;
    writeln!(
        trace,
        "iteration,force_calls,max_force,climbing_image,top_gradient"
    )?;
    let mut force_calls = 0;
    let report = loop {
        let report = session.step(&surface)?;
        force_calls += report.surface_rows;
        assert_eq!(force_calls, surface.force_calls.load(Ordering::Relaxed));
        let ci = report.ci_index.map_or(-1, |i| i as i64);
        let top = top_gradient(session.positions());
        writeln!(
            trace,
            "{},{force_calls},{:e},{ci},{top:e}",
            report.iteration, report.max_force
        )?;
        if report.status == BandStatus::Converged || report.iteration >= 5000 {
            break report;
        }
    };
    write_band(&out.join(format!("{name}_band.csv")), session.positions())?;
    let ci = report.ci_index.expect("a climbing image");
    let (x, y) = (session.positions()[(ci, 0)], session.positions()[(ci, 1)]);
    println!(
        "{name:<8} {:?} after {} steps, {force_calls} force calls, max force {:.1e}, \
         climbing image {ci} at ({x:.6}, {y:.6}), energy {:.6}",
        report.status,
        report.iteration,
        report.max_force,
        muller_brown(x, y).0,
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let out = std::env::args()
        .nth(1)
        .unwrap_or("target/muller_brown".into());
    let out = Path::new(&out);
    fs::create_dir_all(out)?;

    let minimum_a = array![-0.558223634633, 1.44172584195, 0.0];
    let minimum_b = array![0.623499404931, 0.028037758529, 0.0];
    let initial = linear_path(minimum_a.view(), minimum_b.view(), 11, None)?;
    write_band(&out.join("initial_band.csv"), initial.view())?;

    let oci_neb = BandConfig {
        max_move: 0.05,
        ..BandConfig::default()
    };
    let ci_neb = BandConfig {
        min_mode: None,
        ..oci_neb.clone()
    };
    run("ci-neb", ci_neb, &initial, out)?;
    run("oci-neb", oci_neb, &initial, out)?;
    Ok(())
}
