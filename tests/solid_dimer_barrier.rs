//! A barrier that moves when the cell can shear.
//!
//! The variable-cell dimer and the solid-state band meet on the same
//! saddle. Freezing the cell changes the barrier, and that changed
//! barrier matches a one-dimensional scan at the imposed shear.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, array};
use rgmin::Method;
use rgsaddle::{
    BandConfig, BandSession, BandStatus, BandSurface, Cell, CellMask, CiConfig, MinModeConfig,
    MinModeStatus, PointSurface, SaddleError, SolidDimer, SolidDimerConfig, SolidState,
    finite_difference_cauchy_stress,
};

const L: f64 = 5.0;

fn eye_cell() -> [[f64; 3]; 3] {
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

/// Reaction coordinate `u = s_x / 0.25` and shear `ε_yx` off the cube of side `L`.
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

fn flat(cell: [[f64; 3]; 3]) -> [f64; 9] {
    [
        cell[0][0], cell[0][1], cell[0][2], cell[1][0], cell[1][1], cell[1][2], cell[2][0],
        cell[2][1], cell[2][2],
    ]
}

struct Model {
    /// Cell used by [`PointSurface::eval`]. The joint search passes its own.
    fixed: [[f64; 3]; 3],
}

impl Model {
    fn energy_gradient(&self, r: ArrayView1<f64>, cell: [[f64; 3]; 3]) -> (f64, Array1<f64>) {
        let row = [r[0], r[1], r[2]];
        let energy = model_energy(&row, cell);
        let step = 1e-6;
        let mut g = Array1::zeros(3);
        for k in 0..3 {
            let mut plus = row;
            let mut minus = row;
            plus[k] += step;
            minus[k] -= step;
            g[k] = (model_energy(&plus, cell) - model_energy(&minus, cell)) / (2.0 * step);
        }
        (energy, g)
    }
}

impl PointSurface for Model {
    fn eval(&self, r: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        Ok(self.energy_gradient(r, self.fixed))
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
        Ok(self.energy_gradient(r, h))
    }

    fn cauchy_stress(
        &self,
        r: ArrayView1<f64>,
        cell: &[f64; 9],
    ) -> Result<Option<[f64; 9]>, SaddleError> {
        let h = [
            [cell[0], cell[1], cell[2]],
            [cell[3], cell[4], cell[5]],
            [cell[6], cell[7], cell[8]],
        ];
        // Energy only: the gradient finite difference inside `eval_in_cell`
        // is not part of the stress derivative. A local surface returns
        // the energy and a zero gradient so the stress helper reads energy.
        struct EnergyOnly;
        impl PointSurface for EnergyOnly {
            fn eval(&self, r: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
                self.eval_in_cell(r, &flat([[L, 0.0, 0.0], [0.0, L, 0.0], [0.0, 0.0, L]]))
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
                Ok((model_energy(&row, h), Array1::zeros(r.len())))
            }
        }
        let sigma = finite_difference_cauchy_stress(&EnergyOnly, r, h, 1e-6)?;
        Ok(Some(flat(sigma)))
    }
}

impl BandSurface for Model {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (i, row) in positions.outer_iter().enumerate() {
            let (energy, g) = PointSurface::eval(self, row)?;
            energies[i] = energy;
            gradients.row_mut(i).assign(&g);
        }
        Ok(())
    }

    fn eval_cells(
        &self,
        positions: ArrayView2<f64>,
        cells: &[Cell],
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
        stresses: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (i, row) in positions.outer_iter().enumerate() {
            let packed = flat(cells[i].0);
            let (energy, g) = self.eval_in_cell(row, &packed)?;
            let stress = self.cauchy_stress(row, &packed)?.unwrap();
            energies[i] = energy;
            gradients.row_mut(i).assign(&g);
            for k in 0..9 {
                stresses[(i, k)] = stress[k];
            }
        }
        Ok(())
    }
}

fn search(mask: CellMask, start: Array1<f64>, cell: [[f64; 3]; 3]) -> SolidDimer {
    let config = MinModeConfig {
        method: Method::Lbfgs { memory: 12 },
        force_tol: 1e-7,
        max_move: 0.15,
        dr: 1e-5,
        rotation_tol: 1e-7,
        max_rotations: 16,
        ..MinModeConfig::default()
    };
    SolidDimer::new(
        SolidDimerConfig::new(config).with_mask(mask),
        start,
        cell,
        array![1.0, 0.1, 0.0],
    )
    .unwrap()
}

/// Stationary points of `f(u) = (u²-1)² + 6 (γ - 0.2 u)²` by Newton.
fn scan_barrier(gamma: f64) -> (f64, f64) {
    let f = |u: f64| (u * u - 1.0).powi(2) + 6.0 * (gamma - 0.2 * u).powi(2);
    let df = |u: f64| 4.0 * u * (u * u - 1.0) - 2.4 * (gamma - 0.2 * u);
    let ddf = |u: f64| 12.0 * u * u - 4.0 + 0.48;
    let mut roots = Vec::new();
    for guess in [-1.2, -0.2, 0.0, 0.2, 1.2] {
        let mut u = guess;
        for _ in 0..30 {
            let second = ddf(u);
            if second.abs() < 1e-12 {
                break;
            }
            u -= df(u) / second;
        }
        if df(u).abs() < 1e-8 && roots.iter().all(|known: &f64| (known - u).abs() > 1e-4) {
            roots.push(u);
        }
    }
    let energies: Vec<f64> = roots.iter().copied().map(f).collect();
    let saddle = energies.iter().copied().fold(f64::MIN, f64::max);
    let minimum = energies.iter().copied().fold(f64::MAX, f64::min);
    (saddle, saddle - minimum)
}

#[test]
fn variable_cell_dimer_matches_the_solid_state_band() {
    let left = sheared(0.2);
    let right = sheared(-0.2);
    let r_left = array![0.25 * L, 0.0, 0.0];
    let r_right = array![-0.25 * L, 0.0, 0.0];
    assert!(model_energy(r_left.as_slice().unwrap(), left).abs() < 1e-12);
    assert!(model_energy(r_right.as_slice().unwrap(), right).abs() < 1e-12);
    assert!((model_energy(&[0.0, 0.0, 0.0], eye_cell()) - 1.0).abs() < 1e-12);

    let surface = Model { fixed: eye_cell() };
    let mut dimer = search(CellMask::all(), array![0.35, 0.04, -0.02], eye_cell());
    let mut dimer_evals = 0;
    let mut report = dimer.step(&surface).unwrap();
    dimer_evals += report.evaluations;
    for _ in 1..80 {
        if report.status == MinModeStatus::Converged {
            break;
        }
        report = dimer.step(&surface).unwrap();
        dimer_evals += report.evaluations;
    }
    assert_eq!(
        report.status,
        MinModeStatus::Converged,
        "max force {}",
        report.max_force
    );
    assert!(report.curvature < 0.0);
    let dimer_energy = model_energy(dimer.positions().as_slice().unwrap(), dimer.cell());
    assert!(
        (dimer_energy - 1.0).abs() < 2e-3,
        "dimer saddle {dimer_energy}, cell {:?}",
        dimer.cell()
    );
    // (u^2 - 1)^2 is 1 at u = 0, and the shear term is 0 at γ = 0.
    eprintln!("analytic variable dimer evaluations {dimer_evals} energy {dimer_energy}");

    let n_images = 7;
    let mut positions = Array2::zeros((n_images, 3));
    let mut cells = Vec::with_capacity(n_images);
    for i in 0..n_images {
        let t = i as f64 / (n_images - 1) as f64;
        for k in 0..3 {
            positions[(i, k)] = (1.0 - t) * r_left[k] + t * r_right[k];
        }
        let mut h = [[0.0; 3]; 3];
        for a in 0..3 {
            for b in 0..3 {
                h[a][b] = (1.0 - t) * left[a][b] + t * right[a][b];
            }
        }
        cells.push(Cell(h));
    }
    let mut band = BandSession::new(
        BandConfig {
            climbing: Some(CiConfig::always()),
            force_tol: 5e-4,
            max_move: 0.2,
            remove_translation: false,
            method: Method::Lbfgs { memory: 20 },
            solid: Some(SolidState::new(cells)),
            min_mode: None,
            ..BandConfig::default()
        },
        positions,
    )
    .unwrap();
    let mut band_rows = 0;
    let mut band_report = band.step(&surface).unwrap();
    band_rows += band_report.surface_rows;
    for _ in 0..250 {
        if band_report.status == BandStatus::Converged {
            break;
        }
        band_report = band.step(&surface).unwrap();
        band_rows += band_report.surface_rows;
    }
    assert_eq!(
        band_report.status,
        BandStatus::Converged,
        "band force {}",
        band_report.max_force
    );
    let eval = band.evaluation().unwrap();
    let ci = band_report.ci_index.expect("climbing image");
    let band_energy = eval.energies[ci];
    assert!(
        (band_energy - dimer_energy).abs() < 2e-3,
        "band {band_energy} dimer {dimer_energy}"
    );
    eprintln!("analytic band rows {band_rows} climbing energy {band_energy}");
    let (fixed_saddle, fixed_barrier) = scan_barrier(0.0);
    let (free_saddle, free_barrier) = scan_barrier(0.0);
    // Free shear relaxes the minima to energy 0, so the barrier is the
    // saddle energy. The scan at γ = 0 is the frozen-cell section.
    let _ = (free_saddle, free_barrier);
    assert!((fixed_saddle - 1.0).abs() < 1e-8);
    assert!((fixed_barrier - 0.7744).abs() < 2e-3);

    let mut frozen = search(CellMask::frozen(), array![0.35, 0.04, -0.02], eye_cell());
    let mut frozen_evals = 0;
    let mut frozen_report = frozen.step(&surface).unwrap();
    frozen_evals += frozen_report.evaluations;
    for _ in 1..80 {
        if frozen_report.status == MinModeStatus::Converged {
            break;
        }
        frozen_report = frozen.step(&surface).unwrap();
        frozen_evals += frozen_report.evaluations;
    }
    eprintln!("analytic frozen dimer evaluations {frozen_evals}");
    assert_eq!(frozen_report.status, MinModeStatus::Converged);
    let frozen_energy = model_energy(frozen.positions().as_slice().unwrap(), eye_cell());
    assert!((frozen_energy - fixed_saddle).abs() < 2e-3);
    let frozen_barrier = frozen_energy - (fixed_saddle - fixed_barrier);
    let variable_barrier = dimer_energy;
    assert!(
        (variable_barrier - 1.0).abs() < 2e-3 && (frozen_barrier - fixed_barrier).abs() < 2e-3,
        "variable {variable_barrier} frozen {frozen_barrier}"
    );
    assert!((variable_barrier - frozen_barrier).abs() > 0.1);

    // An imposed shear, cell frozen, moves the barrier off the γ = 0 value.
    let gamma = 0.1;
    let (imposed_saddle, imposed_barrier) = scan_barrier(gamma);
    let imposed = Model {
        fixed: sheared(gamma),
    };
    let mut held = search(CellMask::frozen(), array![0.2, 0.0, 0.0], sheared(gamma));
    let held_report = held.run(&imposed, 80).unwrap();
    assert_eq!(held_report.status, MinModeStatus::Converged);
    let held_energy = model_energy(held.positions().as_slice().unwrap(), sheared(gamma));
    assert!(
        (held_energy - imposed_saddle).abs() < 5e-3,
        "held {held_energy} scan {imposed_saddle}"
    );
    let held_barrier = held_energy - (imposed_saddle - imposed_barrier);
    assert!((held_barrier - imposed_barrier).abs() < 5e-3);
    assert!((held_barrier - fixed_barrier).abs() > 1e-3);
}
