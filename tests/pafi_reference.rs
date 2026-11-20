//! The examples of Swinburne and Marinica's reference PAFI code, run with
//! rgsaddle's PAFI.
//!
//! The reference code, github.com/tomswinburne/pafi, ships two NEB paths
//! with their EAM tables in `examples/systems`: a vacancy hop in W on the
//! Marinica04 potential (`EAM-VAC-W`) and a dumbbell hop in Fe on the
//! Marinica07 potential (`EAM-SIA-Fe`). Its `tests/test_pafi.py` pins the
//! zero-temperature gradient on nine planes of the Fe path, and its README
//! plots the W barrier at 0, 1000 and 2000 K from
//! `examples/input_python.py`. `benchmarks/pafi_reference.sh` fetches both
//! systems at a pinned commit, points `RGSADDLE_PAFI_REFERENCE` at them
//! and runs the ignored tests here: one writes
//! `benchmarks/pafi_reference.csv` and `benchmarks/pafi_reference_planes.csv`,
//! and one samples six W planes at two time steps.
//!
//! The path is the one the reference code builds: the frames unwrapped
//! against the first by the minimum image, each at its distance from the
//! first over that of the last, and planes evenly spaced in that `r`.

use std::fmt::Write as _;
use std::num::NonZeroUsize;
use std::path::PathBuf;

use ndarray::{Array1, ArrayView1, s};
use rayon::prelude::*;
use rgsaddle::{
    ArcPath, BOLTZMANN_EV_PER_K, FreeEnergyProfile, HyperplaneAverage, PafiConfig, PathFrame,
    PathGeometry, PointSurface, average_hyperplane, pafi_profile, zero_temperature_barrier,
};

include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/common/eam_fs.rs"
));

/// `FreeEnergyGradient` at 0 K on the planes `r = k/8` of the Fe path,
/// from `test_zero_temperature_run` in `tests/test_pafi.py` of the
/// reference code.
const FE_GRADIENT: [f64; 9] = [
    0.0, 0.614145, 0.846192, 0.59226, 0.089912, -0.426347, -0.853405, -0.792713, 0.0,
];

/// Gradient-descent steps that relax a reference plane at `kT → 0`.
const RELAX_STEPS: usize = 20_000;

/// Gradient-descent steps from a sampled configuration to the minimum of
/// its plane, for [`max_jump`].
const JUMP_STEPS: usize = 5_000;

/// An NEB path as the reference code reads it.
struct ReferencePath {
    crystal: EamCrystal,
    path: ArcPath,
    atoms: NonZeroUsize,
    /// Energy of the first frame.
    start: f64,
}

impl ReferencePath {
    fn load(dir: &Path, potential: &str, images: &[String]) -> Self {
        let mut lengths = None;
        let mut frames = Vec::with_capacity(images.len());
        for image in images {
            let (box_lengths, positions) =
                read_lammps_data(&dir.join(image)).expect("reference image");
            assert!(
                lengths.is_none_or(|known| known == box_lengths),
                "{image}: every image needs the same box"
            );
            lengths = Some(box_lengths);
            frames.push(positions);
        }
        let table = EamTable::read(&dir.join(potential)).expect("reference potential");
        let crystal = EamCrystal::new(table, lengths.expect("images")).expect("crystal");
        Self::new(crystal, frames)
    }

    fn new(crystal: EamCrystal, mut frames: Vec<Vec<f64>>) -> Self {
        let atoms = NonZeroUsize::new(frames[0].len() / 3).expect("reference atoms");
        for atom in frames[0].chunks_mut(3) {
            for (value, length) in atom.iter_mut().zip(crystal.lengths) {
                *value -= length * (*value / length).floor();
            }
        }
        let first = frames[0].clone();
        for frame in frames.iter_mut().skip(1) {
            for (atom, origin) in frame.chunks_mut(3).zip(first.chunks(3)) {
                let mut step = [
                    atom[0] - origin[0],
                    atom[1] - origin[1],
                    atom[2] - origin[2],
                ];
                crystal.image(&mut step);
                for ((value, start), step) in atom.iter_mut().zip(origin).zip(step) {
                    *value = start + step;
                }
            }
        }
        let distance: Vec<f64> = frames
            .iter()
            .map(|frame| {
                frame
                    .iter()
                    .zip(&first)
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f64>()
                    .sqrt()
            })
            .collect();
        let total = distance.last().copied().expect("frames");
        let path = ArcPath::new(
            frames
                .into_iter()
                .zip(&distance)
                .map(|(frame, distance)| PathFrame {
                    positions: Array1::from(frame),
                    arc: distance / total,
                })
                .collect(),
        )
        .expect("reference path");
        let first = path.geometry(path.bounds().0).expect("first frame");
        let (start, _) = crystal.eval(first.positions.view()).expect("first frame");
        Self {
            crystal,
            path,
            atoms,
            start,
        }
    }
}

/// `count` planes evenly spaced on `0 ≤ r ≤ 1`, the reference default for
/// nine.
fn planes(count: usize) -> Vec<f64> {
    (0..count).map(|k| k as f64 / (count - 1) as f64).collect()
}

/// Gradient descent in a plane at `kT → 0` for `steps` steps, without the
/// basin test: a relaxation, not a sampling. A relaxation of 100 steps or
/// more records 64 samples in 4 blocks, so that the drift test of the
/// plane average has more than one record in each block of a half. A
/// shorter run only probes the neighborhood of the path and records 4
/// samples in 2 blocks.
fn descent(steps: usize, atoms: NonZeroUsize) -> PafiConfig {
    let (samples, blocks) = if steps >= 100 { (64, 4) } else { (4, 2) };
    PafiConfig::default()
        .with_thermal(1.0e-9)
        .with_time_step(0.01)
        .with_thermalize(steps)
        .with_stride(1)
        .with_samples(samples)
        .with_blocks(blocks)
        .with_com_atoms(atoms)
        .with_retain(1)
        .with_max_jump(None)
}

/// Every plane relaxed at `kT → 0`, which is what the reference code
/// samples at `Temperature = 0`.
fn relaxed(
    system: &ReferencePath,
    path: &ArcPath,
    planes: &[f64],
    steps: usize,
) -> (FreeEnergyProfile, Vec<HyperplaneAverage>) {
    pafi_profile(&system.crystal, path, planes, &descent(steps, system.atoms))
        .expect("relaxed planes")
}

/// Overdamped sampling of every plane at `temperature` kelvin.
fn sampled(
    system: &ReferencePath,
    path: &ArcPath,
    planes: &[f64],
    temperature: f64,
    sampling: &Sampling,
    seed: u64,
) -> (FreeEnergyProfile, Vec<HyperplaneAverage>) {
    let config = PafiConfig::default()
        .with_thermal(BOLTZMANN_EV_PER_K * temperature)
        .with_mobility(1.0)
        .with_time_step(sampling.time_step)
        .with_thermalize(sampling.thermalize)
        .with_stride(sampling.stride)
        .with_samples(sampling.samples)
        .with_blocks(sampling.blocks)
        .with_seed(seed)
        .with_com_atoms(system.atoms)
        .with_retain(4)
        .with_max_jump(sampling.max_jump);
    pafi_profile(&system.crystal, path, planes, &config).expect("sampled planes")
}

/// The `MaxJump` test of the reference code, which drops a worker whose
/// plane minimum after sampling lies more than 0.4 Å from the one before,
/// on the configurations `average` retained: the largest atomic
/// displacement between the minimum of the plane at `arc` reached from
/// the path and those reached from them, by an independent gradient
/// descent rather than the quench of [`PafiConfig::max_jump`].
fn max_jump(
    system: &ReferencePath,
    path: &ArcPath,
    arc: f64,
    average: &HyperplaneAverage,
    steps: usize,
) -> f64 {
    let geometry = path.geometry(arc).expect("plane");
    let config = descent(steps, system.atoms);
    let minimum = |start: &Array1<f64>| {
        let from = PathGeometry {
            positions: start.clone(),
            ..geometry.clone()
        };
        let mut relaxed =
            average_hyperplane(&system.crystal, &from, &config).expect("plane minimum");
        relaxed.retained.swap_remove(0)
    };
    let home = minimum(&geometry.positions);
    average
        .retained
        .iter()
        .map(|start| {
            let end = minimum(start);
            end.iter()
                .zip(&home)
                .collect::<Vec<_>>()
                .chunks(3)
                .map(|atom| {
                    let mut step = [0.0; 3];
                    for (slot, (a, b)) in step.iter_mut().zip(atom) {
                        *slot = *a - *b;
                    }
                    system.crystal.image(&mut step);
                    step.iter().map(|value| value * value).sum::<f64>().sqrt()
                })
                .fold(0.0, f64::max)
        })
        .fold(0.0, f64::max)
}

/// [`max_jump`] on every plane of a sampled profile.
fn jumps(
    system: &ReferencePath,
    path: &ArcPath,
    (profile, averages): &(FreeEnergyProfile, Vec<HyperplaneAverage>),
    steps: usize,
) -> Vec<f64> {
    profile
        .arc
        .par_iter()
        .zip(averages)
        .map(|(arc, average)| max_jump(system, path, *arc, average, steps))
        .collect()
}

struct Sampling {
    time_step: f64,
    thermalize: usize,
    stride: usize,
    samples: usize,
    blocks: usize,
    /// [`PafiConfig::max_jump`].
    max_jump: Option<f64>,
}

/// 5000 thermalization steps, then 400 000 steps recorded every tenth in
/// 40 blocks, with the basin test at 0.4 Å.
const SAMPLING: Sampling = Sampling {
    time_step: 1.0e-3,
    thermalize: 5_000,
    stride: 10,
    samples: 40_000,
    blocks: 40,
    max_jump: Some(0.4),
};

/// The planes `r = k/32` of [`reference_step_size`], on both sides of the
/// saddle of the W path.
const STEP_PLANES: [f64; 6] = [4.0, 11.0, 17.0, 21.0, 26.0, 29.0];

/// A barrier with its standard error and the free energy of the last
/// plane above the first.
#[derive(Clone, Copy)]
struct Integral {
    barrier: f64,
    error: f64,
    end: f64,
    end_error: f64,
}

/// The trapezoid sums of the profile itself.
fn trapezoid(profile: &FreeEnergyProfile) -> Integral {
    let last = profile.free_energy.len() - 1;
    Integral {
        barrier: profile.barrier,
        error: profile.barrier_error,
        end: profile.free_energy[last],
        end_error: profile.free_energy_error[last],
    }
}

/// The integral of the reference `ResultsProcessor`: the not-a-knot cubic
/// through the plane gradients on `10 n` evenly spaced points, then the
/// cumulative trapezoid over them and its maximum. The cubic is linear in
/// the gradients, so the errors propagate through the weights of that
/// maximum and of the last point.
fn remeshed(profile: &FreeEnergyProfile) -> Integral {
    let arcs = &profile.arc;
    let count = arcs.len();
    let basis = ArcPath::new(
        arcs.iter()
            .enumerate()
            .map(|(k, arc)| {
                let mut positions = Array1::zeros(count + 1);
                positions[0] = *arc;
                positions[k + 1] = 1.0;
                PathFrame {
                    positions,
                    arc: *arc,
                }
            })
            .collect(),
    )
    .expect("remesh basis");
    let dense = 10 * count;
    let (first, last) = (arcs[0], arcs[count - 1]);
    let points: Vec<f64> = (0..dense)
        .map(|j| first + (last - first) * j as f64 / (dense - 1) as f64)
        .collect();
    let weights: Vec<Array1<f64>> = points
        .iter()
        .map(|point| {
            basis
                .geometry(*point)
                .expect("remesh")
                .positions
                .slice(s![1..])
                .to_owned()
        })
        .collect();
    let mut cumulative = vec![Array1::<f64>::zeros(count)];
    for j in 1..dense {
        let step = 0.5 * (points[j] - points[j - 1]);
        let next = &cumulative[j - 1] + &((&weights[j - 1] + &weights[j]) * step);
        cumulative.push(next);
    }
    let gradient = Array1::from(profile.gradient.clone());
    let error = |weights: &Array1<f64>| {
        weights
            .iter()
            .zip(&profile.gradient_error)
            .map(|(w, e)| (w * e).powi(2))
            .sum::<f64>()
            .sqrt()
    };
    let (top, barrier) = cumulative
        .iter()
        .map(|weights| weights.dot(&gradient))
        .enumerate()
        .fold((0, f64::NEG_INFINITY), |best, (j, value)| {
            if value > best.1 { (j, value) } else { best }
        });
    let end = &cumulative[dense - 1];
    Integral {
        barrier,
        error: error(&cumulative[top]),
        end: end.dot(&gradient),
        end_error: error(end),
    }
}

/// One temperature and plane count, integrated from either end.
struct Run {
    system: &'static str,
    temperature: f64,
    forward: (FreeEnergyProfile, Vec<HyperplaneAverage>),
    reversed: (FreeEnergyProfile, Vec<HyperplaneAverage>),
    /// In-plane relaxed energies above the first frame, at `kT → 0`.
    energies: Option<Vec<f64>>,
    /// [`jumps`] of the forward and the reversed profile when sampled.
    jumps: Option<[Vec<f64>; 2]>,
}

impl Run {
    fn rows(&self, summary: &mut String, planes: &mut String) {
        let count = self.forward.0.arc.len();
        for (name, integrate) in [
            ("trapezoid", trapezoid as fn(&FreeEnergyProfile) -> Integral),
            ("remeshed", remeshed),
        ] {
            let forward = integrate(&self.forward.0);
            let reversed = integrate(&self.reversed.0);
            let jump = self.jumps.as_ref().map_or(String::new(), |jumps| {
                let top = jumps.iter().flatten().copied().fold(0.0, f64::max);
                format!("{top:.4}")
            });
            let rejected: usize = [&self.forward.1, &self.reversed.1]
                .into_iter()
                .flatten()
                .map(|average| average.rejected)
                .sum();
            writeln!(
                summary,
                "{},{},{count},{name},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{rejected},{jump}",
                self.system,
                self.temperature,
                forward.barrier,
                forward.error,
                reversed.barrier,
                reversed.error,
                forward.end,
                forward.end_error,
                self.forward.0.samples + self.reversed.0.samples,
            )
            .expect("summary row");
        }
        for (side, (direction, (profile, averages))) in
            [("forward", &self.forward), ("reversed", &self.reversed)]
                .into_iter()
                .enumerate()
        {
            let sign = if direction == "forward" { 1.0 } else { -1.0 };
            for (k, average) in averages.iter().enumerate() {
                let jump = self
                    .jumps
                    .as_ref()
                    .map_or(String::new(), |jumps| format!("{:.4}", jumps[side][k]));
                let arc = if direction == "forward" {
                    profile.arc[k]
                } else {
                    1.0 - profile.arc[k]
                };
                let energy = match (&self.energies, direction) {
                    (Some(energies), "forward") => format!("{:.6}", energies[k]),
                    _ => String::new(),
                };
                writeln!(
                    planes,
                    "{},{},{count},{direction},{arc:.6},{:.6},{:.6},{:.6},{:.1},{},{energy},{jump}",
                    self.system,
                    self.temperature,
                    sign * average.gradient,
                    average.error,
                    average.psi,
                    average.n_eff,
                    average.rejected,
                )
                .expect("plane row");
            }
        }
    }
}

fn relaxed_energies(system: &ReferencePath, averages: &[HyperplaneAverage]) -> Vec<f64> {
    averages
        .iter()
        .map(|average| {
            let (energy, _) = system
                .crystal
                .eval(average.retained[0].view())
                .expect("relaxed energy");
            energy - system.start
        })
        .collect()
}

fn reference_dir() -> PathBuf {
    PathBuf::from(std::env::var_os("RGSADDLE_PAFI_REFERENCE").expect(
        "RGSADDLE_PAFI_REFERENCE must name the reference examples/systems directory; \
         benchmarks/pafi_reference.sh sets it",
    ))
}

fn images(names: impl Iterator<Item = usize>) -> Vec<String> {
    names.map(|k| format!("image_{k}.dat")).collect()
}

#[test]
#[ignore = "release; needs RGSADDLE_PAFI_REFERENCE, writes benchmarks/pafi_reference*.csv"]
fn reference_examples_reproduce() {
    let root = reference_dir();
    let mut summary = String::from(
        "system,temperature_k,planes,integration,barrier,barrier_error,reversed,reversed_error,end,end_error,samples,rejected,max_jump\n",
    );
    let mut rows = String::from(
        "system,temperature_k,planes,direction,r,gradient,gradient_error,psi,n_eff,rejected,relaxed_energy,max_jump\n",
    );

    let iron = ReferencePath::load(&root.join("EAM-SIA-Fe"), "Fe.eam.fs", &images(1..=9));
    let reversed_iron = iron.path.reversed().expect("reversed Fe path");
    let fe_planes = planes(9);
    let forward = relaxed(&iron, &iron.path, &fe_planes, RELAX_STEPS);
    for (k, (average, expected)) in forward.1.iter().zip(FE_GRADIENT).enumerate() {
        assert!(
            (average.gradient - expected).abs() < 1.0e-3,
            "Fe plane {k}: {} against {expected}",
            average.gradient
        );
    }
    let run = Run {
        system: "EAM-SIA-Fe",
        temperature: 0.0,
        energies: Some(relaxed_energies(&iron, &forward.1)),
        jumps: None,
        reversed: relaxed(&iron, &reversed_iron, &fe_planes, RELAX_STEPS),
        forward,
    };
    run.rows(&mut summary, &mut rows);

    let tungsten = ReferencePath::load(&root.join("EAM-VAC-W"), "W.eam.fs", &images(0..=8));
    let reversed_tungsten = tungsten.path.reversed().expect("reversed W path");
    let line = zero_temperature_barrier(&tungsten.crystal, &tungsten.path, &planes(33))
        .expect("zero-temperature integral");
    println!("W energy along the spline: max {:.6} eV", line.barrier);
    for count in [9, 33] {
        let w_planes = planes(count);
        let forward = relaxed(&tungsten, &tungsten.path, &w_planes, RELAX_STEPS);
        let run = Run {
            system: "EAM-VAC-W",
            temperature: 0.0,
            energies: Some(relaxed_energies(&tungsten, &forward.1)),
            jumps: None,
            reversed: relaxed(&tungsten, &reversed_tungsten, &w_planes, RELAX_STEPS),
            forward,
        };
        run.rows(&mut summary, &mut rows);
        for temperature in [1000.0, 2000.0] {
            let seed = 1000 * count as u64 + temperature as u64;
            let forward = sampled(
                &tungsten,
                &tungsten.path,
                &w_planes,
                temperature,
                &SAMPLING,
                seed,
            );
            let reversed = sampled(
                &tungsten,
                &reversed_tungsten,
                &w_planes,
                temperature,
                &SAMPLING,
                seed + 500,
            );
            let run = Run {
                system: "EAM-VAC-W",
                temperature,
                energies: None,
                jumps: Some([
                    jumps(&tungsten, &tungsten.path, &forward, JUMP_STEPS),
                    jumps(&tungsten, &reversed_tungsten, &reversed, JUMP_STEPS),
                ]),
                forward,
                reversed,
            };
            run.rows(&mut summary, &mut rows);
            print!("{summary}");
        }
    }
    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks");
    std::fs::write(out.join("pafi_reference.csv"), &summary).expect("summary csv");
    std::fs::write(out.join("pafi_reference_planes.csv"), &rows).expect("planes csv");
    print!("{summary}");
}

/// The W planes of [`STEP_PLANES`] at 1000 K, sampled as
/// [`reference_examples_reproduce`] samples them at `μ dt = 10⁻³` and
/// again at `8 × 10⁻³` with other seeds: the gradients agree within four
/// errors in quadrature.
#[test]
#[ignore = "release; needs RGSADDLE_PAFI_REFERENCE"]
fn reference_step_size() {
    let tungsten = ReferencePath::load(
        &reference_dir().join("EAM-VAC-W"),
        "W.eam.fs",
        &images(0..=8),
    );
    let arcs: Vec<f64> = STEP_PLANES.iter().map(|k| k / 32.0).collect();
    let long = Sampling {
        time_step: 8.0e-3,
        ..SAMPLING
    };
    let (_, short) = sampled(&tungsten, &tungsten.path, &arcs, 1000.0, &SAMPLING, 7000);
    let (_, long) = sampled(&tungsten, &tungsten.path, &arcs, 1000.0, &long, 8000);
    let mut chi2 = 0.0;
    for ((arc, a), b) in arcs.iter().zip(&short).zip(&long) {
        let z = (b.gradient - a.gradient) / a.error.hypot(b.error);
        println!(
            "W r = {arc:.5}: {:.6} ± {:.6} (n_eff {:.0}) at 1e-3, {:.6} ± {:.6} (n_eff {:.0}) at 8e-3, z {z:+.2}",
            a.gradient, a.error, a.n_eff, b.gradient, b.error, b.n_eff
        );
        assert!(z.abs() < 4.0, "r = {arc}: z = {z}");
        chi2 += z * z;
    }
    println!("chi2/{} {:.2}", arcs.len(), chi2 / arcs.len() as f64);
}

/// A one-element `eam/fs` table: `F = -√ρ`, `ρ = 10 (1 - r/3.6)⁴` and
/// `φ = 3 (1 - r/3.6)³`, tabulated as LAMMPS reads them.
fn synthetic_table() -> String {
    let cutoff: f64 = 3.6;
    let (nrho, drho, nr, dr) = (2001, 0.005, 2001, 0.002);
    let mut text = String::from("synthetic eam/fs\nfor the rgsaddle PAFI smoke test\n\n1 Xx\n");
    writeln!(text, "{nrho} {drho:e} {nr} {dr:e} {cutoff:e}").expect("grid");
    text.push_str("1 50.0 2.9 bcc\n");
    let fall = |r: f64| (1.0 - r / cutoff).max(0.0);
    for k in 0..nrho {
        writeln!(text, "{:e}", -(k as f64 * drho).sqrt()).expect("F");
    }
    for k in 0..nr {
        writeln!(text, "{:e}", 10.0 * fall(k as f64 * dr).powi(4)).expect("rho");
    }
    for k in 0..nr {
        let r = k as f64 * dr;
        writeln!(text, "{:e}", r * 3.0 * fall(r).powi(3)).expect("r phi");
    }
    text
}

/// A 4 × 4 × 4 bcc box with `a = 2.9` Å less the corner site, and five
/// frames of the body-center atom next to it hopping across the periodic
/// boundary into the vacancy with a sideways bow. The other atoms sit off
/// their sites by a fixed pattern, and every frame is wrapped into the box.
fn synthetic_hop() -> ([f64; 3], Vec<Vec<f64>>) {
    let (a, cells) = (2.9, 4);
    let length = a * cells as f64;
    let mut sites = Vec::new();
    for i in 0..cells {
        for j in 0..cells {
            for k in 0..cells {
                let corner = [i as f64, j as f64, k as f64];
                sites.push(corner);
                sites.push(corner.map(|value| value + 0.5));
            }
        }
    }
    let mut atoms: Vec<[f64; 3]> = sites
        .iter()
        .skip(1)
        .enumerate()
        .map(|(id, site)| {
            let mut position = site.map(|value| value * a);
            for (k, value) in position.iter_mut().enumerate() {
                *value += 0.03 * (1.3 * id as f64 + k as f64).sin();
            }
            position
        })
        .collect();
    let hopper = atoms.len() - 1;
    let start = [3.5 * a; 3];
    let frames = (0..5)
        .map(|frame| {
            let t = frame as f64 / 4.0;
            let bow = 0.2 * (std::f64::consts::PI * t).sin() / 2.0_f64.sqrt();
            atoms[hopper] = [
                start[0] + 0.5 * a * t + bow,
                start[1] + 0.5 * a * t - bow,
                start[2] + 0.5 * a * t,
            ];
            atoms
                .iter()
                .flat_map(|atom| atom.map(|value| value - length * (value / length).floor()))
                .collect()
        })
        .collect();
    ([length; 3], frames)
}

/// LAMMPS (22 Jul 2025) `pair_style eam/fs` on [`synthetic_table`]: the
/// energy of every frame of [`synthetic_hop`], then the force on the
/// hopping atom and on atom 1 in the middle frame.
const LAMMPS_ENERGIES: [f64; 5] = [
    -59.87604793277296,
    -59.92718956469189,
    -59.996503663583894,
    -59.918536110455825,
    -59.85582258741337,
];
const LAMMPS_FORCES: [[f64; 3]; 2] = [
    [
        0.2223937861707214,
        -0.21966278950120138,
        -0.014214921963748243,
    ],
    [
        0.06742918002306868,
        0.09987914422355347,
        0.13917641718024384,
    ],
];

#[test]
fn eam_table_matches_lammps() {
    let table = EamTable::parse(&synthetic_table()).expect("table");
    let (lengths, frames) = synthetic_hop();
    let crystal = EamCrystal::new(table, lengths).expect("crystal");
    for (k, (frame, expected)) in frames.iter().zip(LAMMPS_ENERGIES).enumerate() {
        let (energy, _) = crystal.eval(ArrayView1::from(frame)).expect("energy");
        assert!(
            (energy - expected).abs() < 1.0e-10,
            "frame {k}: {energy} against {expected}"
        );
    }
    let (_, gradient) = crystal.eval(ArrayView1::from(&frames[2])).expect("forces");
    let hopper = gradient.len() / 3 - 1;
    for (atom, expected) in [hopper, 0].into_iter().zip(LAMMPS_FORCES) {
        for (k, force) in expected.iter().enumerate() {
            let ours = -gradient[3 * atom + k];
            assert!(
                (ours - force).abs() < 1.0e-10,
                "atom {atom} axis {k}: {ours} against {force}"
            );
        }
    }
}

/// The reference integral of the gradients in `test_pafi.py`, from scipy
/// 1.18.1: `interp1d(kind="cubic")` on 90 points, `cumulative_trapezoid`.
#[test]
fn remeshed_integral_is_the_reference_one() {
    let arcs = planes(9);
    let errors = vec![0.01; 9];
    let profile = rgsaddle::integrate_profile(&arcs, &FE_GRADIENT, &errors).expect("profile");
    let integral = remeshed(&profile);
    assert!((integral.barrier - 0.27622876229655097).abs() < 1.0e-12);
    assert!((integral.end - 0.003824869324197502).abs() < 1.0e-12);
    assert!(integral.error > 0.0 && integral.end_error > integral.error);
}

/// The pipeline of the reference benchmark on a small hop: unwrapping
/// across the boundary, the `r` parameter, the spline through the frame
/// energies, relaxed and sampled planes from both ends, and the rows.
#[test]
fn reference_pipeline_runs_on_a_small_hop() {
    let table = EamTable::parse(&synthetic_table()).expect("table");
    let (lengths, frames) = synthetic_hop();
    let crystal = EamCrystal::new(table, lengths).expect("crystal");
    let system = ReferencePath::new(crystal, frames);
    let arcs = system.path.arcs();
    assert!(arcs[0] == 0.0 && (arcs[4] - 1.0).abs() < 1.0e-15);
    let hop = system.path.geometry(1.0).expect("last frame");
    let corner = hop.positions.len() - 3;
    for value in hop.positions.slice(s![corner..]) {
        assert!((value - 11.6).abs() < 1.0e-9, "unwrapped {value}");
    }
    let line = zero_temperature_barrier(&system.crystal, &system.path, &arcs).expect("line");
    for (k, expected) in LAMMPS_ENERGIES.iter().enumerate() {
        let along = line.free_energy[64 * k];
        let truth = expected - LAMMPS_ENERGIES[0];
        assert!(
            (along - truth).abs() < 1.0e-3,
            "frame {k}: {along} against {truth}"
        );
    }
    let reversed = system.path.reversed().expect("reversed");
    let five = planes(5);
    let forward = relaxed(&system, &system.path, &five, 50);
    let mut summary = String::new();
    let mut rows = String::new();
    Run {
        system: "synthetic",
        temperature: 0.0,
        energies: Some(relaxed_energies(&system, &forward.1)),
        jumps: None,
        reversed: relaxed(&system, &reversed, &five, 50),
        forward,
    }
    .rows(&mut summary, &mut rows);
    // The box is far from a minimum of the synthetic potential: descent in
    // a plane runs on into a collapse, so there is no basin to hold.
    let sampling = Sampling {
        time_step: 1.0e-3,
        thermalize: 10,
        stride: 2,
        samples: 16,
        blocks: 4,
        max_jump: None,
    };
    let forward = sampled(&system, &system.path, &five, 300.0, &sampling, 3);
    let backward = sampled(&system, &reversed, &five, 300.0, &sampling, 4);
    let run = Run {
        system: "synthetic",
        temperature: 300.0,
        energies: None,
        jumps: Some([
            jumps(&system, &system.path, &forward, 20),
            jumps(&system, &reversed, &backward, 20),
        ]),
        forward,
        reversed: backward,
    };
    run.rows(&mut summary, &mut rows);
    assert!(run.jumps.iter().flatten().flatten().all(|jump| *jump < 0.4));
    assert_eq!(summary.lines().count(), 4);
    assert_eq!(rows.lines().count(), 20);
    for line in summary.lines().chain(rows.lines()) {
        for field in line.split(',').skip(4).filter(|field| !field.is_empty()) {
            let value: f64 = field.parse().expect("number");
            assert!(value.is_finite(), "{line}");
        }
    }
}
