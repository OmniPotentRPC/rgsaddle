//! Stencil contracts: springs, zoom accounting, refused pairs, rank.

use std::time::Duration;

use ndarray::{Array1, Array2, ArrayView2};
use rgsaddle::stencil::spring_ew::fill_constants;
use rgsaddle::stencil::{
    AutoNebParams, BandMethod, ImageId, Interp, Method, OcinebParams, Phase, PointMethod, Preset,
    Stencil, StencilConfig, StencilSurface, ZoomEnds, ZoomForce, ZoomParams,
};
use rgsaddle::{BandSurface, CiConfig, SaddleError, SolidState};
use sha2::{Digest, Sha256};

struct Well;

impl BandSurface for Well {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (row, energy) in energies.iter_mut().enumerate() {
            let x = positions[[row, 0]];
            let y = positions[[row, 1]];
            let z = positions[[row, 2]];
            *energy = (x * x - 1.0).powi(2) + 2.0 * y * y + 2.0 * z * z;
            gradients[[row, 0]] = 4.0 * x * (x * x - 1.0);
            gradients[[row, 1]] = 4.0 * y;
            gradients[[row, 2]] = 4.0 * z;
        }
        Ok(())
    }
}

fn line(images: usize) -> Array2<f64> {
    let mut band = Array2::zeros((images, 3));
    for image in 0..images {
        let t = image as f64 / (images - 1) as f64;
        band[[image, 0]] = -1.0 + 2.0 * t;
    }
    band
}

fn message(err: SaddleError) -> String {
    match err {
        SaddleError::Invalid(text) | SaddleError::Shape(text) => text,
        other => format!("{other:?}"),
    }
}

fn must_fail(result: Result<Stencil, SaddleError>) -> String {
    match result {
        Err(err) => message(err),
        Ok(_) => panic!("the stencil build was accepted"),
    }
}

#[test]
fn energy_weighted_constants_match_the_assembly() {
    let stencil = Stencil::band(line(5))
        .energy_weighted(0.972, 9.72)
        .build()
        .unwrap();
    let mut stencil = stencil;
    stencil.step(&Well).unwrap();
    let energies = stencil.energies().unwrap();
    let got = stencil.spring_constants();
    let mut expect = vec![0.0; got.len()];
    fill_constants(energies.as_slice().unwrap(), 0.972, 9.72, &mut expect);
    for (left, right) in got.iter().zip(&expect) {
        assert_eq!(left.to_bits(), right.to_bits());
    }
}

fn zoom_ready(ends: ZoomEnds) -> Stencil {
    let mut config = StencilConfig::from_preset(Preset::EonZoom);
    if let Method::Band(BandMethod::Zoom(params)) = &mut config.method {
        params.stability = 1;
        params.force = ZoomForce::Abs(1.0e9);
        params.ends = ends;
        params.interpolation = Interp::Linear;
    }
    config.climb = CiConfig {
        trigger_factor: 1.0e9,
        trigger_force: 1.0e9,
    };
    config.band.climbing = Some(config.climb);
    Stencil::new(config, line(5)).unwrap()
}

#[test]
fn fixed_zoom_ends_are_not_evaluated_on_the_next_step() {
    let mut stencil = zoom_ready(ZoomEnds::Fixed);
    let mut saw_zoom = false;
    for _ in 0..12 {
        let report = stencil.step(&Well).unwrap();
        if matches!(report.phase, Phase::Zoom) {
            assert_eq!(report.surface_rows, 0);
            saw_zoom = true;
            break;
        }
    }
    assert!(saw_zoom, "zoom did not fire");
    let next = stencil.step(&Well).unwrap();
    assert_eq!(next.surface_rows, 5 - 2);
}

#[test]
fn free_zoom_ends_are_evaluated() {
    let mut stencil = zoom_ready(ZoomEnds::FreePerp);
    for _ in 0..12 {
        let report = stencil.step(&Well).unwrap();
        if matches!(report.phase, Phase::Zoom) {
            assert_eq!(report.surface_rows, 0);
            break;
        }
    }
    let next = stencil.step(&Well).unwrap();
    assert_eq!(next.surface_rows, 5);
}

#[test]
fn refused_pairs_name_the_missing_piece() {
    let path = line(5);
    let auto_zoom = must_fail(
        Stencil::band(path.clone())
            .zoom(ZoomParams::default())
            .autoneb(AutoNebParams::default())
            .build(),
    );
    assert!(auto_zoom.contains("cannot be combined"));

    let point = must_fail(
        Stencil::point(ndarray::array![0.0, 0.0, 0.0])
            .energy_weighted(1.0, 2.0)
            .build(),
    );
    assert!(point.contains("point stencil"));

    let bounds = must_fail(
        Stencil::band(path.clone())
            .energy_weighted(2.0, 1.0)
            .build(),
    );
    assert!(bounds.contains("k_min"));

    let explicit = must_fail(
        Stencil::band(path.clone())
            .explicit_springs(vec![1.0])
            .build(),
    );
    assert!(explicit.contains("explicit springs"));

    let art = must_fail(Stencil::point(ndarray::array![0.0, 0.0, 0.0]).art().build());
    assert!(art.contains("feat/art-nouveau"));

    let ocineb = must_fail(
        Stencil::band(path.clone())
            .ocineb(OcinebParams::default())
            .build(),
    );
    assert!(ocineb.contains("rgsaddle#11"));

    let idpp = must_fail(Stencil::band(path.clone()).idpp().build());
    assert!(idpp.contains("rgsaddle#10"));

    let mut config = StencilConfig::from_preset(Preset::EonZoom);
    config.band.solid = Some(SolidState {
        cells: Vec::new(),
        pressure: 0.0,
        weight: 1.0,
    });
    let solid = must_fail(Stencil::new(config, path));
    assert!(solid.contains("solid-state"));

    let _ = PointMethod::Art;
}

struct Muller;

impl BandSurface for Muller {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        const A: [f64; 4] = [-200.0, -100.0, -170.0, 15.0];
        const AA: [f64; 4] = [-1.0, -1.0, -6.5, 0.7];
        const B: [f64; 4] = [0.0, 0.0, 11.0, 0.6];
        const C: [f64; 4] = [-10.0, -10.0, -6.5, 0.7];
        const X0: [f64; 4] = [1.0, 0.0, -0.5, -1.0];
        const Y0: [f64; 4] = [0.0, 0.5, 1.5, 1.0];
        for (row, energy) in energies.iter_mut().enumerate() {
            let x = positions[[row, 0]];
            let y = positions[[row, 1]];
            let z = positions[[row, 2]];
            let mut value = z * z;
            let mut gx = 0.0;
            let mut gy = 0.0;
            for i in 0..4 {
                let dx = x - X0[i];
                let dy = y - Y0[i];
                let exponent = AA[i] * dx * dx + B[i] * dx * dy + C[i] * dy * dy;
                let weight = A[i] * exponent.exp();
                let px = 2.0 * AA[i] * dx + B[i] * dy;
                let py = B[i] * dx + 2.0 * C[i] * dy;
                value += weight;
                gx += weight * px;
                gy += weight * py;
            }
            *energy = value;
            gradients[[row, 0]] = gx;
            gradients[[row, 1]] = gy;
            gradients[[row, 2]] = 2.0 * z;
        }
        Ok(())
    }
}

struct Lj7;

impl BandSurface for Lj7 {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        let n_atoms = positions.ncols() / 3;
        for (row, energy) in energies.iter_mut().enumerate() {
            let mut value = 0.0;
            for atom in 0..n_atoms {
                let z = positions[[row, 3 * atom + 2]];
                value += z * z;
                gradients[[row, 3 * atom]] = 0.0;
                gradients[[row, 3 * atom + 1]] = 0.0;
                gradients[[row, 3 * atom + 2]] = 2.0 * z;
            }
            for i in 0..n_atoms {
                for j in (i + 1)..n_atoms {
                    let dx = positions[[row, 3 * i]] - positions[[row, 3 * j]];
                    let dy = positions[[row, 3 * i + 1]] - positions[[row, 3 * j + 1]];
                    let dz = positions[[row, 3 * i + 2]] - positions[[row, 3 * j + 2]];
                    let r2 = dx * dx + dy * dy + dz * dz;
                    let inv = 1.0 / r2;
                    let s6 = inv * inv * inv;
                    let s12 = s6 * s6;
                    value += 4.0 * (s12 - s6);
                    let d_e_dr2 = 4.0 * (-6.0 * s12 + 3.0 * s6) * inv;
                    gradients[[row, 3 * i]] += d_e_dr2 * 2.0 * dx;
                    gradients[[row, 3 * i + 1]] += d_e_dr2 * 2.0 * dy;
                    gradients[[row, 3 * i + 2]] += d_e_dr2 * 2.0 * dz;
                    gradients[[row, 3 * j]] -= d_e_dr2 * 2.0 * dx;
                    gradients[[row, 3 * j + 1]] -= d_e_dr2 * 2.0 * dy;
                    gradients[[row, 3 * j + 2]] -= d_e_dr2 * 2.0 * dz;
                }
            }
            *energy = value;
        }
        Ok(())
    }
}

fn mb_band() -> Array2<f64> {
    let start = [-0.050010822738, 0.466694104872, 0.0];
    let end = [0.623499404931, 0.028037758529, 0.0];
    let mut band = Array2::zeros((7, 3));
    for image in 0..7 {
        let t = image as f64 / 6.0;
        for axis in 0..3 {
            band[[image, axis]] = (1.0 - t) * start[axis] + t * end[axis];
        }
    }
    band
}

fn lj7_band() -> Array2<f64> {
    let a = 2.0_f64.powf(1.0 / 6.0);
    let s = 3.0_f64.sqrt() / 2.0;
    let hex = [
        [0.0, 0.0],
        [a, 0.0],
        [0.5 * a, a * s],
        [-0.5 * a, a * s],
        [-a, 0.0],
        [-0.5 * a, -a * s],
        [0.5 * a, -a * s],
    ];
    let mut para = hex;
    para[1][0] += 0.35;
    para[2][1] += 0.25;
    let mut band = Array2::zeros((5, 21));
    for image in 0..5 {
        let t = image as f64 / 4.0;
        for atom in 0..7 {
            band[[image, 3 * atom]] = (1.0 - t) * hex[atom][0] + t * para[atom][0];
            band[[image, 3 * atom + 1]] = (1.0 - t) * hex[atom][1] + t * para[atom][1];
        }
    }
    band
}

struct Lj7Sleep {
    sleep: bool,
}

impl StencilSurface for Lj7Sleep {
    fn eval(
        &self,
        ids: &[ImageId],
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        if self.sleep && ids.first().copied() == Some(0) {
            std::thread::sleep(Duration::from_millis(2));
        }
        BandSurface::eval(&Lj7, positions, energies, gradients)
    }
}

fn digest(stencil: &Stencil) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for value in stencil.positions().iter() {
        hasher.update(value.to_le_bytes());
    }
    hasher.finalize().into()
}

fn run_kind(kind: &str, threads: usize, sleep: bool, steps: usize) -> [u8; 32] {
    match kind {
        "ci" => {
            let mut stencil = Stencil::band(mb_band()).threads(threads).build().unwrap();
            for _ in 0..steps {
                stencil.step(&Muller).unwrap();
            }
            digest(&stencil)
        }
        "ew" => {
            let mut stencil = Stencil::band(mb_band())
                .energy_weighted(0.972, 9.72)
                .threads(threads)
                .build()
                .unwrap();
            for _ in 0..steps {
                stencil.step(&Muller).unwrap();
            }
            digest(&stencil)
        }
        "zoom-fixed" | "zoom-free" => {
            let mut params = ZoomParams::default();
            params.stability = 1;
            params.force = ZoomForce::Abs(1.0e9);
            params.ends = if kind == "zoom-free" {
                ZoomEnds::FreePerp
            } else {
                ZoomEnds::Fixed
            };
            let mut config = StencilConfig::from_preset(Preset::CiNeb);
            config.method = Method::Band(BandMethod::Zoom(params));
            config.threads = threads;
            config.climb.trigger_factor = 1.0e9;
            config.climb.trigger_force = 1.0e9;
            config.band.climbing = Some(config.climb);
            let mut stencil = Stencil::new(config, mb_band()).unwrap();
            for _ in 0..steps {
                stencil.step(&Muller).unwrap();
            }
            digest(&stencil)
        }
        "auto" => {
            let mut params = AutoNebParams::default();
            params.local_max_steps = 50;
            params.climb_max_steps = 50;
            let mut config = StencilConfig::from_preset(Preset::AutoNeb);
            config.method = Method::Band(BandMethod::AutoNeb(params));
            config.threads = threads;
            let mut stencil = Stencil::new(config, mb_band()).unwrap();
            for _ in 0..steps {
                stencil.step(&Muller).unwrap();
            }
            digest(&stencil)
        }
        "lj7" => {
            let mut stencil = Stencil::band(lj7_band()).threads(threads).build().unwrap();
            let surface = Lj7Sleep { sleep };
            for _ in 0..steps {
                stencil.step(&surface).unwrap();
            }
            digest(&stencil)
        }
        other => panic!("unknown kind {other}"),
    }
}

#[test]
fn rank_does_not_change_the_raw_bits() {
    let kinds = ["ci", "ew", "zoom-fixed", "zoom-free", "auto", "lj7"];
    for kind in kinds {
        let reference = run_kind(kind, 1, false, 3);
        for threads in [2_usize, 3, 8] {
            let got = run_kind(kind, threads, false, 3);
            assert_eq!(reference, got, "{kind} at {threads} threads");
        }
    }
    let quiet = run_kind("lj7", 8, false, 3);
    let slept = run_kind("lj7", 8, true, 3);
    assert_eq!(quiet, slept, "a delayed block changed the bits");
}

fn con_coords(name: &str) -> (f64, Vec<[f64; 3]>) {
    let path = format!(
        "{}/validation/eon_zoom/neb_lj13/{name}.con",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).unwrap_or_else(|err| panic!("{path}: {err}"));
    let mut stored = None;
    let mut coords = Vec::new();
    let mut reading = false;
    for line in text.lines() {
        if let Some(rest) = line.split("\"energy\":").nth(1) {
            let token: String = rest
                .chars()
                .take_while(|c| *c == '-' || *c == '.' || c.is_ascii_digit())
                .collect();
            stored = Some(token.parse::<f64>().unwrap());
        }
        if line.starts_with("Coordinates") {
            reading = true;
            continue;
        }
        if !reading {
            continue;
        }
        let mut parts = line.split_whitespace();
        let Some(x) = parts.next().and_then(|part| part.parse::<f64>().ok()) else {
            continue;
        };
        let y: f64 = parts.next().unwrap().parse().unwrap();
        let z: f64 = parts.next().unwrap().parse().unwrap();
        coords.push([x, y, z]);
    }
    (stored.unwrap(), coords)
}

fn lj_energy(coords: &[[f64; 3]]) -> f64 {
    let mut energy = 0.0;
    let rc2 = 100.0;
    for i in 0..coords.len() {
        for j in (i + 1)..coords.len() {
            let dx = coords[i][0] - coords[j][0];
            let dy = coords[i][1] - coords[j][1];
            let dz = coords[i][2] - coords[j][2];
            let r2 = dx * dx + dy * dy + dz * dz;
            if r2 >= rc2 {
                continue;
            }
            let inv = 1.0 / r2;
            let s6 = inv * inv * inv;
            let s12 = s6 * s6;
            energy += 4.0 * (s12 - s6);
        }
    }
    energy
}

#[test]
fn lj13_con_energies_match_ase_lj_within_1e_4() {
    for name in ["reactant", "product", "saddle"] {
        let (stored, coords) = con_coords(name);
        let energy = lj_energy(&coords);
        assert!(
            (energy - stored).abs() < 1e-4,
            "{name}: computed {energy} stored {stored}"
        );
    }
}
