//! Joint-chart solid-state NEB.
//!
//! A frozen cell is the ordinary band, including the climbing image.
//! The variable-cell double well is the shear-coupled model whose
//! saddle energy is 1.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, array};
use rgmin::Method;
use rgsaddle::{
    BandConfig, BandSession, BandStatus, BandSurface, CellMask, CiConfig, PointSurface,
    SaddleError, SolidNeb,
};

struct DoubleWell;

fn well(row: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let (x, y, z) = (row[0], row[1], row[2]);
    let energy = (x * x - 1.0).powi(2) + 2.0 * y * y + 2.0 * z * z;
    let gradient = array![4.0 * x * (x * x - 1.0), 4.0 * y, 4.0 * z];
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
            let (energy, gradient) = well(row);
            energies[i] = energy;
            gradients.row_mut(i).assign(&gradient);
        }
        Ok(())
    }
}

impl PointSurface for DoubleWell {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        Ok(well(x))
    }
}

fn bowed(n_images: usize) -> Array2<f64> {
    Array2::from_shape_fn((n_images, 3), |(i, k)| {
        let t = i as f64 / (n_images - 1) as f64;
        [-1.0 + 2.0 * t, 0.3 * (std::f64::consts::PI * t).sin(), 0.0][k]
    })
}

fn ci_config() -> BandConfig {
    BandConfig {
        climbing: Some(CiConfig::always()),
        force_tol: 1e-8,
        max_move: 0.2,
        min_mode: None,
        method: Method::Lbfgs { memory: 12 },
        ..BandConfig::default()
    }
}

fn same_positions(left: ArrayView2<f64>, right: ArrayView2<f64>) {
    assert_eq!(left.dim(), right.dim());
    for (a, b) in left.iter().zip(right.iter()) {
        assert_eq!(a, b, "{a} vs {b}");
    }
}

#[test]
fn frozen_cell_matches_the_ordinary_band_and_its_climbing_image() {
    let initial = bowed(9);
    let plain = BandConfig {
        climbing: None,
        force_tol: 1e-8,
        max_move: 0.2,
        min_mode: None,
        method: Method::Lbfgs { memory: 12 },
        ..BandConfig::default()
    };
    let mut ordinary = BandSession::new(plain.clone(), initial.clone()).unwrap();
    let mut frozen = SolidNeb::frozen(plain, initial.clone()).unwrap();
    assert!(frozen.is_frozen());
    for _ in 0..4 {
        ordinary.step(&DoubleWell).unwrap();
        frozen.step_surface(&DoubleWell).unwrap();
        same_positions(ordinary.positions(), frozen.cartesian().unwrap().view());
    }

    let config = ci_config();
    let mut ordinary = BandSession::new(config.clone(), initial.clone()).unwrap();
    let mut frozen = SolidNeb::frozen(config, initial).unwrap();
    let mut ordinary_report = ordinary.step(&DoubleWell).unwrap();
    let mut frozen_report = frozen.step_surface(&DoubleWell).unwrap();
    for _ in 0..400 {
        if ordinary_report.status == BandStatus::Converged
            && frozen_report.status == BandStatus::Converged
        {
            break;
        }
        ordinary_report = ordinary.step(&DoubleWell).unwrap();
        frozen_report = frozen.step_surface(&DoubleWell).unwrap();
        same_positions(ordinary.positions(), frozen.cartesian().unwrap().view());
        assert_eq!(ordinary_report.status, frozen_report.status);
        assert_eq!(ordinary_report.ci_index, frozen_report.ci_index);
        assert_eq!(
            ordinary_report.max_force.to_bits(),
            frozen_report.max_force.to_bits()
        );
    }
    assert_eq!(ordinary_report.status, BandStatus::Converged);
    assert_eq!(frozen_report.status, BandStatus::Converged);
    let ci = ordinary_report.ci_index.expect("climbing image");
    let pos = ordinary.positions();
    assert!(pos[(ci, 0)].abs() < 1e-5, "saddle x {}", pos[(ci, 0)]);
    assert!(pos[(ci, 1)].abs() < 1e-5, "saddle y {}", pos[(ci, 1)]);
    // The point-surface entry is the same band.
    let mut via_point = SolidNeb::frozen(ci_config(), bowed(9)).unwrap();
    let mut via_surface = BandSession::new(ci_config(), bowed(9)).unwrap();
    for _ in 0..3 {
        via_point.step(&DoubleWell).unwrap();
        via_surface.step(&DoubleWell).unwrap();
    }
    same_positions(
        via_surface.positions(),
        via_point.cartesian().unwrap().view(),
    );
}

const L: f64 = 5.0;

fn eye() -> [[f64; 3]; 3] {
    [[L, 0.0, 0.0], [0.0, L, 0.0], [0.0, 0.0, L]]
}

fn sheared(gamma: f64) -> [[f64; 3]; 3] {
    [[L, 0.0, 0.0], [gamma * L, L, 0.0], [0.0, 0.0, L]]
}

fn invert(h: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let d = h[0][0] * (h[1][1] * h[2][2] - h[1][2] * h[2][1])
        - h[0][1] * (h[1][0] * h[2][2] - h[1][2] * h[2][0])
        + h[0][2] * (h[1][0] * h[2][1] - h[1][1] * h[2][0]);
    let mut c = [[0.0; 3]; 3];
    c[0][0] = (h[1][1] * h[2][2] - h[1][2] * h[2][1]) / d;
    c[0][1] = (h[0][2] * h[2][1] - h[0][1] * h[2][2]) / d;
    c[0][2] = (h[0][1] * h[1][2] - h[0][2] * h[1][1]) / d;
    c[1][0] = (h[1][2] * h[2][0] - h[1][0] * h[2][2]) / d;
    c[1][1] = (h[0][0] * h[2][2] - h[0][2] * h[2][0]) / d;
    c[1][2] = (h[0][2] * h[1][0] - h[0][0] * h[1][2]) / d;
    c[2][0] = (h[1][0] * h[2][1] - h[1][1] * h[2][0]) / d;
    c[2][1] = (h[0][1] * h[2][0] - h[0][0] * h[2][1]) / d;
    c[2][2] = (h[0][0] * h[1][1] - h[0][1] * h[1][0]) / d;
    c
}

fn mul_row(row: [f64; 3], m: [[f64; 3]; 3]) -> [f64; 3] {
    [
        row[0] * m[0][0] + row[1] * m[1][0] + row[2] * m[2][0],
        row[0] * m[0][1] + row[1] * m[1][1] + row[2] * m[2][1],
        row[0] * m[0][2] + row[1] * m[1][2] + row[2] * m[2][2],
    ]
}

/// `(u² − 1)² + 6 (γ − 0.2 u)²` plus penalties that vanish on the path.
/// `u = 0`, `γ = 0` is energy 1. Both minima are energy 0.
fn model_energy(r: &[f64], h: [[f64; 3]; 3]) -> f64 {
    let s = mul_row([r[0], r[1], r[2]], invert(h));
    let u = s[0] / 0.25;
    let mut eps = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            eps[i][j] = h[i][j] / L;
        }
        eps[i][i] -= 1.0;
    }
    let gamma = eps[1][0];
    (u * u - 1.0).powi(2)
        + 6.0 * (gamma - 0.2 * u).powi(2)
        + 4.0 * s[1] * s[1]
        + 4.0 * s[2] * s[2]
        + 3.0
            * (eps[0][0] * eps[0][0]
                + eps[1][1] * eps[1][1]
                + eps[2][2] * eps[2][2]
                + eps[2][0] * eps[2][0]
                + eps[2][1] * eps[2][1])
}

struct ShearWell;

impl PointSurface for ShearWell {
    fn eval(&self, r: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        self.eval_in_cell(r, &[L, 0.0, 0.0, 0.0, L, 0.0, 0.0, 0.0, L])
    }

    fn eval_in_cell(
        &self,
        r: ArrayView1<f64>,
        cell: &[f64; 9],
    ) -> Result<(f64, Array1<f64>), SaddleError> {
        let h = [
            [cell[0], cell[1], cell[2]],
            [cell[3], cell[4], cell[5]],
            [cell[6], cell[7], cell[8]],
        ];
        let row = [r[0], r[1], r[2]];
        let energy = model_energy(&row, h);
        let step = 1e-7;
        let mut gradient = Array1::zeros(3);
        for k in 0..3 {
            let mut plus = row;
            let mut minus = row;
            plus[k] += step;
            minus[k] -= step;
            gradient[k] = (model_energy(&plus, h) - model_energy(&minus, h)) / (2.0 * step);
        }
        Ok((energy, gradient))
    }
}

#[test]
fn variable_cell_double_well_barrier_is_one() {
    let left_r = array![0.25 * L, 0.0, 0.0];
    let right_r = array![-0.25 * L, 0.0, 0.0];
    let left_h = sheared(0.2);
    let right_h = sheared(-0.2);
    assert!(model_energy(left_r.as_slice().unwrap(), left_h).abs() < 1e-12);
    assert!(model_energy(right_r.as_slice().unwrap(), right_h).abs() < 1e-12);
    assert!((model_energy(&[0.0, 0.0, 0.0], eye()) - 1.0).abs() < 1e-12);

    let n_images = 9;
    // Linear in (r0, J ε), the coordinate the spring and the tangent see.
    let reference = eye();
    let probe =
        rgsaddle::JointChart::new(reference, 1, CellMask::all(), [[0.0; 3]; 3], 1.0).unwrap();
    let q_left =
        rgsaddle::joint_coordinate(&probe, CellMask::all(), left_r.view(), left_h).unwrap();
    let q_right =
        rgsaddle::joint_coordinate(&probe, CellMask::all(), right_r.view(), right_h).unwrap();
    let mut chart_pos = Array2::zeros((n_images, 3));
    let mut chart_cells = Vec::with_capacity(n_images);
    for i in 0..n_images {
        let t = i as f64 / (n_images - 1) as f64;
        let q = &q_left * (1.0 - t) + &q_right * t;
        let (r, h) = probe.geometry(q.view()).unwrap();
        chart_pos.row_mut(i).assign(&r);
        chart_cells.push(h);
    }
    let mut band = SolidNeb::joint(
        BandConfig {
            climbing: Some(CiConfig::always()),
            force_tol: 1e-6,
            max_move: 0.2,
            remove_translation: false,
            method: Method::Lbfgs { memory: 20 },
            min_mode: None,
            ..BandConfig::default()
        },
        reference,
        chart_pos,
        &chart_cells,
        CellMask::all(),
        [[0.0; 3]; 3],
        1.0,
    )
    .unwrap();
    assert!(!band.is_frozen());
    let mut report = band.step(&ShearWell).unwrap();
    for _ in 0..300 {
        if report.status == BandStatus::Converged {
            break;
        }
        report = band.step(&ShearWell).unwrap();
    }
    assert_eq!(
        report.status,
        BandStatus::Converged,
        "max force {}",
        report.max_force
    );
    let ci = report.ci_index.expect("climbing image");
    let images = band.images().expect("evaluation");
    let energy = images[ci].energy;
    let enthalpy = images[ci].enthalpy;
    let barrier = enthalpy - images[0].enthalpy.min(images[n_images - 1].enthalpy);
    assert!(
        (energy - 1.0).abs() < 1e-3,
        "climbing energy {energy}, barrier {barrier}"
    );
    assert!((barrier - 1.0).abs() < 1e-3, "barrier {barrier}");
    assert!((enthalpy - energy).abs() < 1e-12, "no external stress");
}

/// A narrow repulsive wall on top of the double well. The wall stands in
/// for two atoms that the initial path drives through each other.
struct SteepWall;

fn steep(row: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let (x, y, z) = (row[0], row[1], row[2]);
    let well = (x * x - 1.0).powi(2) + 2.0 * y * y + 2.0 * z * z;
    let dx = x - 0.35;
    let denom = dx * dx + 1.0e-3;
    let bump = 0.08 / denom;
    let d_bump = -0.16 * dx / (denom * denom);
    let energy = well + bump;
    let gradient = array![4.0 * x * (x * x - 1.0) + d_bump, 4.0 * y, 4.0 * z];
    (energy, gradient)
}

impl PointSurface for SteepWall {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        Ok(steep(x))
    }
}

impl BandSurface for SteepWall {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (i, row) in positions.outer_iter().enumerate() {
            let (energy, gradient) = steep(row);
            energies[i] = energy;
            gradients.row_mut(i).assign(&gradient);
        }
        Ok(())
    }
}

/// L-BFGS on a smooth climbing band converges: the two-loop update is not
/// the defect. Immediate climbing follows the highest image, so a path
/// that still contains a collision climbs that collision. The band's
/// L-BFGS guards drop a step that faces away from the force or hits the
/// cap; they do not move the collision off the band. FIRE with climbing
/// off does.
#[test]
fn lbfgs_diverges_when_it_climbs_a_collision_image() {
    let mut smooth = BandConfig {
        force_tol: 1.0e-3,
        ..ci_config()
    };
    smooth.min_mode = None;
    let mut band = BandSession::new(smooth, bowed(7)).unwrap();
    let mut report = band.step(&DoubleWell).unwrap();
    for _ in 0..80 {
        if report.status == BandStatus::Converged {
            break;
        }
        report = band.step(&DoubleWell).unwrap();
    }
    assert!(
        report.max_force < 1.0e-2,
        "a smooth climbing band stays with L-BFGS, force {}",
        report.max_force
    );

    let mut spiked = bowed(7);
    spiked[(3, 0)] = 0.35;
    spiked[(3, 1)] = 0.0;
    let climb_now = BandConfig {
        force_tol: 1.0e-2,
        max_move: 0.2,
        time_step: 0.1,
        climbing: Some(CiConfig::always()),
        method: Method::Lbfgs { memory: 12 },
        min_mode: None,
        ..BandConfig::default()
    };
    let mut climb = BandSession::new(climb_now, spiked.clone()).unwrap();
    let mut force = climb.step(&SteepWall).unwrap().max_force;
    let mut energy = 0.0;
    for _ in 0..25 {
        let step = climb.step(&SteepWall).unwrap();
        force = step.max_force;
        let row = climb.positions().row(3).to_owned();
        energy = steep(row.view()).0;
        if !force.is_finite() || force > 50.0 {
            break;
        }
    }
    assert!(
        !force.is_finite() || force > 5.0 || energy > 10.0,
        "immediate climbing should stay on the collision or blow up, force {force} energy {energy}"
    );

    let plain = BandConfig {
        force_tol: 1.0e-2,
        max_move: 0.15,
        time_step: 0.1,
        climbing: None,
        method: Method::Fire {
            kind: rgmin::FireKind::V2,
        },
        min_mode: None,
        ..BandConfig::default()
    };
    let mut relax = BandSession::new(plain, spiked).unwrap();
    let mut force = f64::MAX;
    for _ in 0..200 {
        let step = relax.step(&SteepWall).unwrap();
        force = step.max_force;
        if force < 0.5 {
            break;
        }
    }
    let row = relax.positions().row(3).to_owned();
    let energy = steep(row.view()).0;
    assert!(
        energy < 3.0,
        "FIRE without climbing leaves the collision, force {force} energy {energy}"
    );
}
