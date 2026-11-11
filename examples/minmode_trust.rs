// Copyright (c) rgsaddle developers
//!
//! Gradient calls and wall time of one min-mode search with the curvature
//! trust region off and on. Off is the stepper `MinModeConfig::default`
//! uses. Each case is warmed once, then run seven times in series. The
//! printed calls and milliseconds are the medians of those seven.
//!
//! ```text
//! cargo run --release --example minmode_trust
//! ```

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use ndarray::{Array1, ArrayView1};
use rgsaddle::{
    MinModeConfig, MinModeKind, MinModeSession, MinModeStatus, PointSurface, SaddleError,
};

const REPEATS: usize = 7;

fn main() {
    let jobs = [
        Job {
            name: "quadratic",
            surface: Surface::Quadratic,
            start: array3(0.35, 0.4, -0.3),
            seed: array3(0.8, 0.5, 0.1),
            method: MethodKind::Fire,
            steps: 400,
        },
        Job {
            name: "quadratic",
            surface: Surface::Quadratic,
            start: array3(0.35, 0.4, -0.3),
            seed: array3(0.8, 0.5, 0.1),
            method: MethodKind::Lbfgs,
            steps: 400,
        },
        Job {
            name: "double-well",
            surface: Surface::DoubleWell,
            start: array3(0.7, 0.2, -0.15),
            seed: array3(0.3, 0.8, 0.2),
            method: MethodKind::Fire,
            steps: 800,
        },
        Job {
            name: "double-well",
            surface: Surface::DoubleWell,
            start: array3(0.7, 0.2, -0.15),
            seed: array3(0.3, 0.8, 0.2),
            method: MethodKind::Lbfgs,
            steps: 800,
        },
        Job {
            name: "muller-brown",
            surface: Surface::Muller,
            start: array3(-0.6, 0.8, 0.05),
            seed: array3(-1.0, 0.4, 0.0),
            method: MethodKind::Fire,
            steps: 1500,
        },
        Job {
            name: "muller-brown",
            surface: Surface::Muller,
            start: array3(-0.6, 0.8, 0.05),
            seed: array3(-1.0, 0.4, 0.0),
            method: MethodKind::Lbfgs,
            steps: 1500,
        },
        Job {
            name: "spread-12",
            surface: Surface::Spread { stiff: 2.0 },
            start: Array1::from_elem(12, 0.15),
            seed: Array1::from_elem(12, 1.0),
            method: MethodKind::Fire,
            steps: 800,
        },
        Job {
            name: "spread-12",
            surface: Surface::Spread { stiff: 2.0 },
            start: Array1::from_elem(12, 0.15),
            seed: Array1::from_elem(12, 1.0),
            method: MethodKind::Lbfgs,
            steps: 800,
        },
        Job {
            name: "stiff-12",
            surface: Surface::Spread { stiff: 40.0 },
            start: Array1::from_elem(12, 0.08),
            seed: Array1::from_elem(12, 1.0),
            method: MethodKind::Fire,
            steps: 800,
        },
        Job {
            name: "stiff-12",
            surface: Surface::Spread { stiff: 40.0 },
            start: Array1::from_elem(12, 0.08),
            seed: Array1::from_elem(12, 1.0),
            method: MethodKind::Lbfgs,
            steps: 800,
        },
    ];
    println!("case\tmethod\ttrust\tconverged\tcalls\tms");
    for job in &jobs {
        for trust in [false, true] {
            let _ = job.run(trust);
            let mut calls = Vec::with_capacity(REPEATS);
            let mut millis = Vec::with_capacity(REPEATS);
            let mut converged = true;
            for _ in 0..REPEATS {
                let sample = job.run(trust);
                calls.push(sample.calls);
                millis.push(sample.millis);
                converged &= sample.converged;
            }
            calls.sort();
            millis.sort_by(f64::total_cmp);
            println!(
                "{}\t{}\t{}\t{}\t{}\t{:.3}",
                job.name,
                job.method.label(),
                trust,
                converged,
                calls[REPEATS / 2],
                millis[REPEATS / 2],
            );
        }
    }
}

struct Job {
    name: &'static str,
    surface: Surface,
    start: Array1<f64>,
    seed: Array1<f64>,
    method: MethodKind,
    steps: usize,
}

struct Sample {
    calls: usize,
    millis: f64,
    converged: bool,
}

impl Job {
    fn run(&self, trust: bool) -> Sample {
        let config = MinModeConfig {
            kind: MinModeKind::Dimer,
            trust,
            force_tol: 1e-3,
            max_move: 0.2,
            method: self.method.method(),
            ..MinModeConfig::default()
        };
        let mut session =
            MinModeSession::new(config, self.start.clone(), self.seed.clone()).expect("session");
        let surface = Counting {
            inner: self.surface,
            calls: AtomicUsize::new(0),
        };
        let started = Instant::now();
        let report = session.run(&surface, self.steps).expect("search");
        let millis = started.elapsed().as_secs_f64() * 1.0e3;
        Sample {
            calls: surface.calls.load(Ordering::Relaxed),
            millis,
            converged: report.status == MinModeStatus::Converged,
        }
    }
}

#[derive(Clone, Copy)]
enum MethodKind {
    Fire,
    Lbfgs,
}

impl MethodKind {
    fn method(self) -> rgmin::Method {
        match self {
            Self::Fire => rgmin::Method::Fire {
                kind: rgmin::FireKind::V2,
            },
            Self::Lbfgs => rgmin::Method::Lbfgs { memory: 10 },
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Fire => "fire",
            Self::Lbfgs => "lbfgs",
        }
    }
}

#[derive(Clone, Copy)]
enum Surface {
    Quadratic,
    DoubleWell,
    Muller,
    Spread { stiff: f64 },
}

struct Counting {
    inner: Surface,
    calls: AtomicUsize,
}

impl PointSurface for Counting {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(self.inner.eval(x))
    }
}

impl Surface {
    fn eval(self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        match self {
            Self::Quadratic => {
                let energy = -x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
                let gradient = Array1::from_vec(vec![-2.0 * x[0], 2.0 * x[1], 2.0 * x[2]]);
                (energy, gradient)
            }
            Self::DoubleWell => {
                let (a, b, c) = (x[0], x[1], x[2]);
                let energy = (a * a - 1.0).powi(2) + 2.0 * b * b + 2.0 * c * c;
                let gradient = Array1::from_vec(vec![4.0 * a * (a * a - 1.0), 4.0 * b, 4.0 * c]);
                (energy, gradient)
            }
            Self::Muller => {
                let (energy_xy, gx, gy) = muller_brown(x[0], x[1]);
                let energy = energy_xy + 2.0 * x[2] * x[2];
                let gradient = Array1::from_vec(vec![gx, gy, 4.0 * x[2]]);
                (energy, gradient)
            }
            Self::Spread { stiff } => {
                let mut energy = 0.0;
                let mut gradient = Array1::zeros(x.len());
                for (i, component) in x.iter().enumerate() {
                    let curvature = if i == 0 { -2.0 } else { stiff };
                    energy += 0.5 * curvature * component * component;
                    gradient[i] = curvature * component;
                }
                (energy, gradient)
            }
        }
    }
}

fn array3(x: f64, y: f64, z: f64) -> Array1<f64> {
    Array1::from_vec(vec![x, y, z])
}

const PREFACTOR: [f64; 4] = [-200.0, -100.0, -170.0, 15.0];
const XX: [f64; 4] = [-1.0, -1.0, -6.5, 0.7];
const XY: [f64; 4] = [0.0, 0.0, 11.0, 0.6];
const YY: [f64; 4] = [-10.0, -10.0, -6.5, 0.7];
const X0: [f64; 4] = [1.0, 0.0, -0.5, -1.0];
const Y0: [f64; 4] = [0.0, 0.5, 1.5, 1.0];

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
