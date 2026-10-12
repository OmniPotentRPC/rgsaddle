//! OCI-NEB against the analytic saddles.
//!
//! The stop is an interior climbing image whose distance to the saddle
//! is below 1e-3 and whose Cartesian gradient RMS is below `force_tol`.
//! On the double well the dimer lands on the saddle. On Müller-Brown the
//! curvature minimum along the unstable mode is offset from TS1, so the
//! same stop is reached by CI-NEB and missed by OCI-NEB.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rgsaddle::stencil::Preset;
use rgsaddle::{BandSurface, SaddleError, Stencil};

const FORCE_TOL: f64 = 1e-3;
const DIST_TOL: f64 = 1e-3;

fn double_well_energy(x: f64, y: f64, z: f64) -> (f64, [f64; 3]) {
    let energy = (x * x - 1.0).powi(2) + 2.0 * y * y + 2.0 * z * z;
    let gradient = [4.0 * x * (x * x - 1.0), 4.0 * y, 4.0 * z];
    (energy, gradient)
}

struct DoubleWell;

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
    let start = [-0.558, 1.442, 0.0];
    let end = [0.623, 0.028, 0.0];
    let n = 5;
    let mut band = Array2::zeros((n, 3));
    for i in 0..n {
        let t = i as f64 / (n - 1) as f64;
        for k in 0..3 {
            band[(i, k)] = (1.0 - t) * start[k] + t * end[k];
        }
    }
    band
}

fn distance(row: ArrayView1<f64>, saddle: &[f64]) -> f64 {
    row.iter()
        .zip(saddle.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn gradient_rms(row: ArrayView1<f64>, well: bool) -> f64 {
    let g = if well {
        double_well_energy(row[0], row[1], row[2]).1.to_vec()
    } else {
        let (_, g) = muller_brown(row[0], row[1]);
        vec![g[0], g[1], 0.0]
    };
    let n = row.len();
    (g.iter().take(n).map(|v| v * v).sum::<f64>() / n as f64).sqrt()
}

struct Sample {
    calls: usize,
    dist: f64,
    rms: f64,
    dimer: bool,
}

fn chase(preset: Preset, band: Array2<f64>, saddle: [f64; 3], well: bool, steps: usize) -> Sample {
    let mut search = Stencil::band(band)
        .preset(preset)
        .threads(1)
        .force_tol(FORCE_TOL)
        .max_move(0.2)
        .build()
        .expect("band");
    let mut dimer = false;
    let mut last = Sample {
        calls: 0,
        dist: f64::INFINITY,
        rms: f64::INFINITY,
        dimer: false,
    };
    for _ in 0..steps {
        let report = if well {
            search.step_band(&DoubleWell).expect("step")
        } else {
            search.step_band(&MullerBrown).expect("step")
        };
        if report.curvature.is_finite() && report.curvature < 0.0 {
            dimer = true;
        }
        let ci = report.ci_index.expect("climbing index");
        let positions = search.positions();
        let row = positions.row(ci);
        last = Sample {
            calls: report.total_evaluations,
            dist: distance(row, &saddle),
            rms: gradient_rms(row, well),
            dimer,
        };
        let interior = ci > 0 && ci + 1 < positions.nrows();
        if interior && last.dist < DIST_TOL && last.rms < FORCE_TOL {
            return last;
        }
    }
    last
}

#[test]
fn oci_neb_reaches_the_double_well_saddle_in_fewer_calls() {
    let saddle = [0.0, 0.0, 0.0];
    let oci = chase(Preset::OciNeb, double_well_band(), saddle, true, 40);
    let ci = chase(Preset::CiNeb, double_well_band(), saddle, true, 40);
    assert!(oci.dimer, "the dimer did not run");
    assert!(
        oci.dist < 1e-12,
        "OCI distance {} is not on the saddle",
        oci.dist
    );
    assert!(
        oci.rms < FORCE_TOL,
        "OCI RMS {} is above the tolerance",
        oci.rms
    );
    assert!(ci.dist < DIST_TOL, "CI distance {}", ci.dist);
    assert!(ci.rms < FORCE_TOL, "CI RMS {}", ci.rms);
    assert_eq!(oci.calls, 96, "OCI force calls");
    assert_eq!(ci.calls, 371, "CI force calls");
    assert!(oci.calls <= ci.calls);
}

#[test]
fn oci_neb_misses_muller_brown_ts1_inside_the_ci_budget() {
    let saddle = [-0.822001558732, 0.624312802418, 0.0];
    let ci = chase(Preset::CiNeb, muller_band(), saddle, false, 200);
    assert_eq!(ci.calls, 1045, "CI force calls");
    assert!(ci.dist < 1e-5, "CI distance {}", ci.dist);
    assert!(ci.rms < FORCE_TOL, "CI RMS {}", ci.rms);

    let mut search = Stencil::band(muller_band())
        .preset(Preset::OciNeb)
        .threads(1)
        .force_tol(FORCE_TOL)
        .max_move(0.2)
        .build()
        .expect("oci");
    let mut dimer = false;
    let mut at_budget: Option<Sample> = None;
    let mut best_dist = f64::INFINITY;
    let mut best_rms = f64::INFINITY;
    let mut best_calls = 0usize;
    let mut met = false;
    for _ in 0..200 {
        let report = search.step_band(&MullerBrown).expect("step");
        if report.curvature.is_finite() && report.curvature < 0.0 {
            dimer = true;
        }
        let ci_index = report.ci_index.expect("climbing index");
        let positions = search.positions();
        let row = positions.row(ci_index);
        let sample = Sample {
            calls: report.total_evaluations,
            dist: distance(row, &saddle),
            rms: gradient_rms(row, false),
            dimer,
        };
        if sample.dist < best_dist {
            best_dist = sample.dist;
            best_rms = sample.rms;
            best_calls = sample.calls;
        }
        let interior = ci_index > 0 && ci_index + 1 < positions.nrows();
        if interior && sample.dist < DIST_TOL && sample.rms < FORCE_TOL {
            met = true;
            break;
        }
        if at_budget.is_none() && sample.calls >= ci.calls {
            at_budget = Some(sample);
        }
    }
    assert!(dimer, "the dimer did not run");
    assert!(
        !met,
        "OCI met the saddle tolerance; the call comparison has to be measured again"
    );
    let over = at_budget.expect("OCI never reached the CI call count");
    assert_eq!(over.calls, 9757, "OCI calls at the CI budget");
    assert!(over.calls > ci.calls);
    assert!(
        over.dist > 0.2,
        "OCI distance {} at {} calls",
        over.dist,
        over.calls
    );
    assert!(
        over.rms > FORCE_TOL,
        "OCI RMS {} is under the tolerance",
        over.rms
    );
    assert!(
        best_rms > FORCE_TOL,
        "closest OCI image RMS {} at {} calls",
        best_rms,
        best_calls
    );
    assert!(best_calls > ci.calls);
}
