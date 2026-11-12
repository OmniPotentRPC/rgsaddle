//! A held atom stays put under every band stepper, and a masked
//! exponential metric on a small Lennard-Jones slab finds the same
//! barrier as the unpreconditioned band.

use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, Array2, ArrayView2};
use rgmin::Method;
use rgsaddle::{
    BandConfig, BandSession, BandStatus, BandSurface, ExpBandPrecon, Ode12rTolerances, SaddleError,
    SpringKind,
};

struct Pull;

impl BandSurface for Pull {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (image, row) in positions.outer_iter().enumerate() {
            let mut energy = 0.0;
            for atom in 0..row.len() / 3 {
                let (x, y, z) = (row[3 * atom], row[3 * atom + 1], row[3 * atom + 2]);
                energy += 0.5 * (x * x + y * y + z * z);
                gradients[(image, 3 * atom)] = x;
                gradients[(image, 3 * atom + 1)] = y;
                gradients[(image, 3 * atom + 2)] = z;
            }
            energies[image] = energy;
        }
        Ok(())
    }
}

fn two_atom_band() -> Array2<f64> {
    let mut band = Array2::zeros((5, 6));
    for image in 0..5 {
        let t = image as f64 / 4.0;
        band[(image, 0)] = 0.4;
        band[(image, 1)] = -0.2;
        band[(image, 2)] = 0.1;
        band[(image, 3)] = -1.0 + 2.0 * t;
        band[(image, 4)] = 0.3 * (std::f64::consts::PI * t).sin();
    }
    band
}

fn holds(method: Method, quickmin: bool, rtr: bool) {
    let initial = two_atom_band();
    let config = BandConfig {
        method,
        quickmin,
        min_mode: None,
        climbing: None,
        remove_translation: false,
        force_tol: 1e-12,
        max_move: 0.2,
        ..BandConfig::default()
    };
    let mut session = BandSession::new(config, initial.clone()).unwrap();
    if rtr {
        session
            .set_rtr(Some(rgsaddle::RtrConfig {
                radius_max: 0.2,
                ..rgsaddle::RtrConfig::default()
            }))
            .unwrap();
    }
    session.set_fixed_mask(Some(&[true, false])).unwrap();
    for _ in 0..6 {
        session.step(&Pull).unwrap();
    }
    let got = session.positions();
    for image in 0..5 {
        for axis in 0..3 {
            assert_eq!(
                got[(image, axis)],
                initial[(image, axis)],
                "image {image} axis {axis} moved"
            );
        }
    }
}

#[test]
fn every_stepper_leaves_a_masked_atom_where_it_stands() {
    holds(
        Method::Fire {
            kind: rgmin::FireKind::V2,
        },
        false,
        false,
    );
    holds(Method::Lbfgs { memory: 8 }, false, false);
    holds(Method::QuickMin, true, false);
    holds(Method::Ode12r, false, false);
    holds(
        Method::Fire {
            kind: rgmin::FireKind::V2,
        },
        false,
        true,
    );
}

#[test]
fn default_ode12r_tolerances_do_not_change_the_step() {
    let initial = two_atom_band();
    let config = || BandConfig {
        method: Method::Ode12r,
        min_mode: None,
        climbing: None,
        remove_translation: false,
        force_tol: 1e-12,
        ..BandConfig::default()
    };
    let mut plain = BandSession::new(config(), initial.clone()).unwrap();
    let mut named = BandSession::new(config(), initial).unwrap();
    named
        .set_ode12r_tolerances(Ode12rTolerances::default())
        .unwrap();
    assert_eq!(named.ode12r_tolerances(), Ode12rTolerances::default());
    for _ in 0..4 {
        plain.step(&Pull).unwrap();
        named.step(&Pull).unwrap();
    }
    assert_eq!(plain.positions(), named.positions());
}

struct Stiff;

impl BandSurface for Stiff {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (image, row) in positions.outer_iter().enumerate() {
            let y = row[1];
            energies[image] = 20.0 * y * y;
            gradients[(image, 0)] = 0.0;
            gradients[(image, 1)] = 40.0 * y;
            gradients[(image, 2)] = 0.0;
        }
        Ok(())
    }
}

#[test]
fn a_tighter_rtol_rejects_an_overshoot_the_default_accepts() {
    // One atom, y = 0.02, curvature 40, opening step 0.06. The Euler
    // step overshoots (hk = 2.4) so the residual grows, and the local
    // error sits between 1e-8 and 0.1: the default bound accepts it and
    // a tight rtol rejects it.
    let mut initial = Array2::zeros((5, 3));
    for image in 0..5 {
        let t = image as f64 / 4.0;
        initial[(image, 0)] = -1.0 + 2.0 * t;
        if image > 0 && image < 4 {
            initial[(image, 1)] = 0.02;
        }
    }
    let config = || BandConfig {
        method: Method::Ode12r,
        min_mode: None,
        climbing: None,
        force_tol: 1e-12,
        max_move: 10.0,
        time_step: 0.06,
        spring: SpringKind::Uniform { k: 0.0 },
        ..BandConfig::default()
    };
    let mut loose = BandSession::new(config(), initial.clone()).unwrap();
    let mut tight = BandSession::new(config(), initial.clone()).unwrap();
    tight
        .set_ode12r_tolerances(Ode12rTolerances {
            rtol: 1e-8,
            ..Ode12rTolerances::default()
        })
        .unwrap();
    loose.step(&Stiff).unwrap();
    tight.step(&Stiff).unwrap();
    let loose_move = (loose.positions()[(2, 1)] - initial[(2, 1)]).abs();
    let tight_move = (tight.positions()[(2, 1)] - initial[(2, 1)]).abs();
    assert!(
        tight_move < loose_move,
        "tight {tight_move} vs loose {loose_move}"
    );
}

fn lj(r: f64) -> (f64, f64) {
    const CUT: f64 = 3.5;
    if !(r > 1e-8 && r < CUT) {
        return (0.0, 0.0);
    }
    let sr6 = (1.0 / r).powi(6);
    let tail = {
        let c6 = (1.0 / CUT).powi(6);
        4.0 * (c6 * c6 - c6)
    };
    let energy = 4.0 * (sr6 * sr6 - sr6) - tail;
    let dedr = 24.0 * (sr6 - 2.0 * sr6 * sr6) / r;
    (energy, dedr)
}

struct Slab {
    calls: AtomicUsize,
}

impl BandSurface for Slab {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        self.calls.fetch_add(positions.nrows(), Ordering::Relaxed);
        gradients.fill(0.0);
        for (image, row) in positions.outer_iter().enumerate() {
            let n = row.len() / 3;
            let mut energy = 0.0;
            for a in 0..n {
                for b in a + 1..n {
                    let d = [
                        row[3 * b] - row[3 * a],
                        row[3 * b + 1] - row[3 * a + 1],
                        row[3 * b + 2] - row[3 * a + 2],
                    ];
                    let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                    let (e, dedr) = lj(r);
                    energy += e;
                    if r > 1e-8 {
                        for axis in 0..3 {
                            let g = dedr * d[axis] / r;
                            gradients[(image, 3 * b + axis)] += g;
                            gradients[(image, 3 * a + axis)] -= g;
                        }
                    }
                }
            }
            energies[image] = energy;
        }
        Ok(())
    }
}

fn layer() -> Vec<[f64; 3]> {
    let h = 1.2 * 3.0_f64.sqrt() / 2.0;
    vec![
        [0.0, 0.0, 0.0],
        [1.2, 0.0, 0.0],
        [2.4, 0.0, 0.0],
        [0.6, h, 0.0],
        [1.8, h, 0.0],
    ]
}

fn hollows() -> ([f64; 2], [f64; 2]) {
    let sites = layer();
    let mean = |ids: [usize; 3]| {
        let mut p = [0.0; 2];
        for id in ids {
            p[0] += sites[id][0];
            p[1] += sites[id][1];
        }
        [p[0] / 3.0, p[1] / 3.0]
    };
    (mean([0, 1, 3]), mean([1, 2, 4]))
}

fn energy_at(layer: &[[f64; 3]], adatom: [f64; 3]) -> (f64, [f64; 3]) {
    let mut energy = 0.0;
    let mut g = [0.0; 3];
    for atom in layer {
        let d = [
            adatom[0] - atom[0],
            adatom[1] - atom[1],
            adatom[2] - atom[2],
        ];
        let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        let (e, dedr) = lj(r);
        energy += e;
        if r > 1e-8 {
            for axis in 0..3 {
                g[axis] += dedr * d[axis] / r;
            }
        }
    }
    (energy, g)
}

fn relax_hollow(xy: [f64; 2]) -> [f64; 3] {
    let sites = layer();
    let mut best_z = 1.0;
    let mut best_e = f64::INFINITY;
    let mut z = 0.5;
    while z < 2.2 {
        let (energy, _) = energy_at(&sites, [xy[0], xy[1], z]);
        if energy < best_e {
            best_e = energy;
            best_z = z;
        }
        z += 0.02;
    }
    z = best_z;
    for _ in 0..30 {
        let (_, g) = energy_at(&sites, [xy[0], xy[1], z]);
        let step = (-0.05 * g[2]).clamp(-0.02, 0.02);
        z += step;
    }
    [xy[0], xy[1], z]
}

fn slab_band(n_images: usize) -> (Array2<f64>, Vec<bool>) {
    let sites = layer();
    let (left, right) = hollows();
    let a = relax_hollow(left);
    let b = relax_hollow(right);
    let n_atoms = sites.len() + 1;
    let mut band = Array2::zeros((n_images, 3 * n_atoms));
    for image in 0..n_images {
        let t = image as f64 / (n_images - 1) as f64;
        for (atom, site) in sites.iter().enumerate() {
            for axis in 0..3 {
                band[(image, 3 * atom + axis)] = site[axis];
            }
        }
        let ad = n_atoms - 1;
        for axis in 0..3 {
            band[(image, 3 * ad + axis)] = (1.0 - t) * a[axis] + t * b[axis];
        }
    }
    let mut mask = vec![true; sites.len()];
    mask.push(false);
    (band, mask)
}

fn barrier(positions: ArrayView2<f64>, ci: usize) -> f64 {
    let surface = Slab {
        calls: AtomicUsize::new(0),
    };
    let mut energies = Array1::zeros(positions.nrows());
    let mut gradients = Array2::zeros(positions.dim());
    surface
        .eval(positions, &mut energies, &mut gradients)
        .unwrap();
    energies[ci] - energies[0].min(energies[positions.nrows() - 1])
}

fn relax_slab(precon: bool) -> (f64, usize, Array2<f64>) {
    let (initial, mask) = slab_band(5);
    let config = BandConfig {
        method: Method::Ode12r,
        min_mode: None,
        remove_translation: false,
        force_tol: 1e-4,
        max_move: 0.15,
        spring: SpringKind::Uniform { k: 0.5 },
        exp_precon: precon.then(|| ExpBandPrecon {
            mu: Some(1.0),
            ..ExpBandPrecon::default()
        }),
        ..BandConfig::default()
    };
    let mut session = BandSession::new(config, initial.clone()).unwrap();
    session.set_fixed_mask(Some(&mask)).unwrap();
    let surface = Slab {
        calls: AtomicUsize::new(0),
    };
    let mut status = BandStatus::Running;
    for _ in 0..4000 {
        let report = session.step(&surface).unwrap();
        status = report.status;
        if status == BandStatus::Converged {
            break;
        }
    }
    assert_eq!(status, BandStatus::Converged, "slab did not converge");
    let got = session.positions();
    for image in 0..got.nrows() {
        for atom in 0..mask.len() - 1 {
            for axis in 0..3 {
                assert_eq!(
                    got[(image, 3 * atom + axis)],
                    initial[(image, 3 * atom + axis)],
                    "layer atom {atom} moved"
                );
            }
        }
    }
    let ci = session.climbing_image().expect("climbing image");
    (
        barrier(got.view(), ci),
        surface.calls.load(Ordering::Relaxed),
        got.to_owned(),
    )
}

#[test]
fn masked_exp_matches_the_unpreconditioned_slab_barrier() {
    let (plain, plain_calls, _) = relax_slab(false);
    let (precon, precon_calls, _) = relax_slab(true);
    eprintln!(
        "LJ slab barrier plain {plain:.6} eV ({plain_calls} rows), exp {precon:.6} ({precon_calls} rows)"
    );
    assert!(
        (plain - precon).abs() < 1e-4,
        "barriers {plain} and {precon}"
    );
    assert!(plain > 1e-3, "barrier collapsed to {plain}");
}

fn morse_pt_pair(r: f64) -> (f64, f64) {
    let (de, alpha, re, cut) = (0.7102, 1.6047, 2.897, 6.5);
    if !(r > 1e-8 && r < cut) {
        return (0.0, 0.0);
    }
    let d = 1.0 - (-alpha * (r - re)).exp();
    let dc = 1.0 - (-alpha * (cut - re)).exp();
    let energy = de * (d * d - dc * dc);
    let dedr = 2.0 * de * alpha * d * (1.0 - d);
    (energy, dedr)
}

/// Fixed-bottom Morse slab in the Pt parameters of the IRC fixture.
/// Ignored in the default suite; run it from the benchmark note.
#[test]
#[ignore]
fn morse_pt_fixed_bottom_benchmark() {
    let re = 2.897;
    let h = re * 3.0_f64.sqrt() / 2.0;
    let z1 = (2.0_f64 / 3.0).sqrt() * re;
    let mut sites = Vec::new();
    for row in 0..3 {
        for col in 0..3 {
            let shift = if row % 2 == 0 { 0.0 } else { 0.5 * re };
            sites.push([col as f64 * re + shift, row as f64 * h, 0.0]);
        }
    }
    let bottom = sites.len();
    for row in 0..3 {
        for col in 0..3 {
            let shift = if row % 2 == 0 { 0.5 * re } else { 0.0 };
            sites.push([col as f64 * re + shift, row as f64 * h, z1]);
        }
    }
    let top = sites.len();
    let left = [
        0.5 * (sites[0][0] + sites[1][0] + sites[3][0]),
        0.5 * (sites[0][1] + sites[1][1] + sites[3][1]) / 1.5,
        z1 + 0.8 * re,
    ];
    // The two hollows sit over the first-layer triangles, one lattice
    // step apart, a height above the second layer.
    let hollow = |origin: usize| {
        let mut p = [0.0; 3];
        for id in [origin, origin + 1, origin + 3] {
            p[0] += sites[id][0];
            p[1] += sites[id][1];
        }
        [p[0] / 3.0, p[1] / 3.0, z1 + 0.85 * re]
    };
    let _ = left;
    let mut a = hollow(0);
    let mut b = hollow(1);
    // Sit the adatom on the Morse minimum above the second layer.
    for end in [&mut a, &mut b] {
        let mut best_z = end[2];
        let mut best = f64::INFINITY;
        let mut z = z1 + 0.4 * re;
        while z < z1 + 1.6 * re {
            let mut energy = 0.0;
            for site in &sites {
                let d = [end[0] - site[0], end[1] - site[1], z - site[2]];
                let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                energy += morse_pt_pair(r).0;
            }
            if energy < best {
                best = energy;
                best_z = z;
            }
            z += 0.02;
        }
        end[2] = best_z;
    }
    let n_atoms = sites.len() + 1;
    let n_images = 5;
    let mut band = Array2::zeros((n_images, 3 * n_atoms));
    for image in 0..n_images {
        let t = image as f64 / (n_images - 1) as f64;
        for (atom, site) in sites.iter().enumerate() {
            for axis in 0..3 {
                band[(image, 3 * atom + axis)] = site[axis];
            }
        }
        let ad = n_atoms - 1;
        for axis in 0..3 {
            band[(image, 3 * ad + axis)] = (1.0 - t) * a[axis] + t * b[axis];
        }
    }
    let mut mask = vec![true; bottom];
    mask.extend(std::iter::repeat_n(false, top - bottom + 1));
    let grad_of = |x: &[f64]| {
        let n = x.len() / 3;
        let mut energy = 0.0;
        let mut g = vec![0.0; x.len()];
        for i in 0..n {
            for j in i + 1..n {
                let d = [
                    x[3 * j] - x[3 * i],
                    x[3 * j + 1] - x[3 * i + 1],
                    x[3 * j + 2] - x[3 * i + 2],
                ];
                let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                let (e, dedr) = morse_pt_pair(r);
                energy += e;
                if r > 1e-8 {
                    for axis in 0..3 {
                        let c = dedr * d[axis] / r;
                        g[3 * j + axis] += c;
                        g[3 * i + axis] -= c;
                    }
                }
            }
        }
        (energy, g)
    };
    let relax_free = |x: &mut [f64]| {
        let mut v = vec![0.0; x.len()];
        for _ in 0..800 {
            let (_, g) = grad_of(x);
            let mut max_f = 0.0_f64;
            for i in 0..x.len() {
                if mask[i / 3] {
                    continue;
                }
                max_f = max_f.max(g[i].abs());
                v[i] = (0.5 * v[i] - 0.015 * g[i]).clamp(-0.04, 0.04);
                x[i] += v[i];
            }
            if max_f < 5e-3 {
                break;
            }
        }
    };
    let mut left: Vec<f64> = band.row(0).iter().copied().collect();
    let mut right: Vec<f64> = band.row(n_images - 1).iter().copied().collect();
    relax_free(&mut left);
    relax_free(&mut right);
    for image in 0..n_images {
        let t = image as f64 / (n_images - 1) as f64;
        for coord in 0..left.len() {
            let atom = coord / 3;
            band[(image, coord)] = if mask[atom] {
                left[coord]
            } else {
                (1.0 - t) * left[coord] + t * right[coord]
            };
        }
    }

    struct Morse {
        calls: AtomicUsize,
        pair: fn(f64) -> (f64, f64),
    }
    impl BandSurface for Morse {
        fn eval(
            &self,
            positions: ArrayView2<f64>,
            energies: &mut Array1<f64>,
            gradients: &mut Array2<f64>,
        ) -> Result<(), SaddleError> {
            self.calls.fetch_add(positions.nrows(), Ordering::Relaxed);
            gradients.fill(0.0);
            for (image, row) in positions.outer_iter().enumerate() {
                let n = row.len() / 3;
                let mut energy = 0.0;
                for i in 0..n {
                    for j in i + 1..n {
                        let d = [
                            row[3 * j] - row[3 * i],
                            row[3 * j + 1] - row[3 * i + 1],
                            row[3 * j + 2] - row[3 * i + 2],
                        ];
                        let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                        let (e, dedr) = (self.pair)(r);
                        energy += e;
                        if r > 1e-8 {
                            for axis in 0..3 {
                                let g = dedr * d[axis] / r;
                                gradients[(image, 3 * j + axis)] += g;
                                gradients[(image, 3 * i + axis)] -= g;
                            }
                        }
                    }
                }
                energies[image] = energy;
            }
            Ok(())
        }
    }

    let run = |precon: bool| {
        let config = BandConfig {
            method: if precon {
                Method::Ode12r
            } else {
                Method::Fire {
                    kind: rgmin::FireKind::V2,
                }
            },
            min_mode: None,
            remove_translation: false,
            force_tol: 1e-2,
            max_move: 0.1,
            time_step: 0.1,
            spring: SpringKind::Uniform { k: 0.5 },
            exp_precon: precon.then(|| ExpBandPrecon {
                mu: Some(1.0),
                r_cut_factor: 2.0,
                ..ExpBandPrecon::default()
            }),
            ..BandConfig::default()
        };
        let mut session = BandSession::new(config, band.clone()).unwrap();
        session.set_fixed_mask(Some(&mask)).unwrap();
        let surface = Morse {
            calls: AtomicUsize::new(0),
            pair: morse_pt_pair,
        };
        let mut status = BandStatus::Running;
        let mut steps = 0;
        let mut note = String::new();
        for _ in 0..2500 {
            match session.step(&surface) {
                Ok(report) => {
                    steps += 1;
                    status = report.status;
                    if status == BandStatus::Converged {
                        break;
                    }
                }
                Err(err) => {
                    note = err.to_string();
                    break;
                }
            }
        }
        let positions = session.positions().to_owned();
        let ci = session.climbing_image().unwrap_or(n_images / 2);
        let mut energies = Array1::zeros(n_images);
        let mut gradients = Array2::zeros((n_images, 3 * n_atoms));
        surface
            .eval(positions.view(), &mut energies, &mut gradients)
            .unwrap();
        let height = energies[ci] - energies[0].min(energies[n_images - 1]);
        (
            status,
            steps,
            surface.calls.load(Ordering::Relaxed),
            height,
            note,
        )
    };
    let plain = run(false);
    let precon = run(true);
    eprintln!(
        "Morse Pt-like slab: plain FIRE status {:?} steps {} calls {} barrier {:.6} {}; Exp+ODE12r status {:?} steps {} calls {} barrier {:.6} {}",
        plain.0,
        plain.1,
        plain.2,
        plain.3,
        plain.4,
        precon.0,
        precon.1,
        precon.2,
        precon.3,
        precon.4
    );
}
