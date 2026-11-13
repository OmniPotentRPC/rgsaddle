//! Bain path of the switched Lennard-Jones crystal, through the
//! solid-state band and the solid-state dimer.
//!
//! The potential is the unshifted Lennard-Jones pair, epsilon 1 eV and
//! sigma 1 Å, times the C^2 quintic switch over [2.0, 2.2] Å. Both
//! endpoints are the two-atom body-centred tetragonal cell. bcc is
//! c/a = 1 and fcc is c/a = sqrt(2). The scan in
//! `validation/bain_lj_reference.py` puts the index-1 saddle at
//! c/a = 1.09331985 with a barrier of 0.0590356373 eV/atom above bcc.
//! ASE 3.29's NEB on a `FrechetCellFilter` of the same pair
//! (`validation/ssneb_bain_ase.py`) climbs to c/a = 1.09332479 and
//! 0.0590356369 eV/atom in 18 steps and 126 calculator calls.
//! Stillinger-Weber silicon has no metastable beta-tin minimum: a
//! two-atom beta-tin cell falls to the diamond energy.

use ndarray::{Array1, Array2, ArrayView2};
use rgmin::{FireKind, Method};
use rgsaddle::{
    BandConfig, BandSession, BandStatus, BandSurface, Cell, CiConfig, SaddleError, SolidState,
};

const RC: f64 = 2.2;
const WIDTH: f64 = 0.2;
const A_BCC: f64 = 1.228441371;
const A_FCC: f64 = 1.100691058;
const C_FCC: f64 = 1.556612222;
const E_BCC: f64 = -7.4148256256;
const Q_SADDLE: f64 = 1.09331985;
const BARRIER: f64 = 0.0590356373;

fn pair(r: f64) -> (f64, f64) {
    let sr6 = (1.0 / r).powi(6);
    let v = 4.0 * (sr6 * sr6 - sr6);
    let dv = 4.0 * (-12.0 * sr6 * sr6 + 6.0 * sr6) / r;
    let t = ((r - (RC - WIDTH)) / WIDTH).clamp(0.0, 1.0);
    let s = 1.0 + t * t * t * (-10.0 + t * (15.0 - 6.0 * t));
    let ds = -30.0 * t * t * (1.0 - t) * (1.0 - t) / WIDTH;
    (v * s, dv * s + v * ds)
}

fn inverse(h: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let det = h[0][0] * (h[1][1] * h[2][2] - h[1][2] * h[2][1])
        - h[0][1] * (h[1][0] * h[2][2] - h[1][2] * h[2][0])
        + h[0][2] * (h[1][0] * h[2][1] - h[1][1] * h[2][0]);
    let s = 1.0 / det;
    [
        [
            (h[1][1] * h[2][2] - h[1][2] * h[2][1]) * s,
            (h[0][2] * h[2][1] - h[0][1] * h[2][2]) * s,
            (h[0][1] * h[1][2] - h[0][2] * h[1][1]) * s,
        ],
        [
            (h[1][2] * h[2][0] - h[1][0] * h[2][2]) * s,
            (h[0][0] * h[2][2] - h[0][2] * h[2][0]) * s,
            (h[0][2] * h[1][0] - h[0][0] * h[1][2]) * s,
        ],
        [
            (h[1][0] * h[2][1] - h[1][1] * h[2][0]) * s,
            (h[0][1] * h[2][0] - h[0][0] * h[2][1]) * s,
            (h[0][0] * h[1][1] - h[0][1] * h[1][0]) * s,
        ],
    ]
}

/// Energy, Cartesian forces and Cauchy stress of one periodic cell.
fn lj(x: &[f64], h: &[[f64; 3]; 3]) -> (f64, Vec<f64>, [f64; 9]) {
    let n = x.len() / 3;
    let inv = inverse(h);
    let mut reach = [0i32; 3];
    for k in 0..3 {
        let col = (inv[0][k] * inv[0][k] + inv[1][k] * inv[1][k] + inv[2][k] * inv[2][k]).sqrt();
        reach[k] = (RC * col + 0.5).ceil() as i32;
    }
    let mut energy = 0.0;
    let mut grad = vec![0.0; 3 * n];
    let mut w = [0.0; 9];
    for i in 0..n {
        for j in i..n {
            let mut d0 = [
                x[3 * i] - x[3 * j],
                x[3 * i + 1] - x[3 * j + 1],
                x[3 * i + 2] - x[3 * j + 2],
            ];
            let mut f = [0.0; 3];
            for c in 0..3 {
                f[c] = d0[0] * inv[0][c] + d0[1] * inv[1][c] + d0[2] * inv[2][c];
            }
            for c in 0..3 {
                let shift = f[c].round();
                for k in 0..3 {
                    d0[k] -= shift * h[c][k];
                }
            }
            let weight = if i == j { 0.5 } else { 1.0 };
            for a in -reach[0]..=reach[0] {
                for b in -reach[1]..=reach[1] {
                    for c in -reach[2]..=reach[2] {
                        if i == j && a == 0 && b == 0 && c == 0 {
                            continue;
                        }
                        let d = [
                            d0[0] + a as f64 * h[0][0] + b as f64 * h[1][0] + c as f64 * h[2][0],
                            d0[1] + a as f64 * h[0][1] + b as f64 * h[1][1] + c as f64 * h[2][1],
                            d0[2] + a as f64 * h[0][2] + b as f64 * h[1][2] + c as f64 * h[2][2],
                        ];
                        let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                        if r2 >= RC * RC {
                            continue;
                        }
                        let r = r2.sqrt();
                        let (phi, dphi) = pair(r);
                        energy += weight * phi;
                        let g = weight * dphi / r;
                        for k in 0..3 {
                            grad[3 * i + k] += g * d[k];
                            grad[3 * j + k] -= g * d[k];
                        }
                        for p in 0..3 {
                            for q in 0..3 {
                                w[3 * p + q] += g * d[p] * d[q];
                            }
                        }
                    }
                }
            }
        }
    }
    let vol = (h[0][0] * (h[1][1] * h[2][2] - h[1][2] * h[2][1])
        - h[0][1] * (h[1][0] * h[2][2] - h[1][2] * h[2][0])
        + h[0][2] * (h[1][0] * h[2][1] - h[1][1] * h[2][0]))
        .abs();
    let mut stress = [0.0; 9];
    for k in 0..9 {
        stress[k] = w[k] / vol;
    }
    (energy, grad, stress)
}

fn bct(a: f64, c: f64) -> ([[f64; 3]; 3], [f64; 6]) {
    let h = [[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, c]];
    let x = [0.0, 0.0, 0.0, 0.5 * a, 0.5 * a, 0.5 * c];
    (h, x)
}

struct Crystal;

impl Crystal {
    fn at(&self, x: &[f64], h: &[[f64; 3]; 3]) -> (f64, Vec<f64>, [f64; 9]) {
        lj(x, h)
    }
}

impl BandSurface for Crystal {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        let (h, _) = bct(A_BCC, A_BCC);
        for row in 0..positions.nrows() {
            let x: Vec<f64> = positions.row(row).to_vec();
            let (e, g, _) = self.at(&x, &h);
            energies[row] = e;
            for k in 0..g.len() {
                gradients[(row, k)] = g[k];
            }
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
        for row in 0..positions.nrows() {
            let x: Vec<f64> = positions.row(row).to_vec();
            let (e, g, s) = self.at(&x, &cells[row].0);
            energies[row] = e;
            for k in 0..g.len() {
                gradients[(row, k)] = g[k];
            }
            for k in 0..9 {
                stresses[(row, k)] = s[k];
            }
        }
        Ok(())
    }
}

fn ratio(cell: &Cell) -> f64 {
    cell.0[2][2] / cell.0[0][0]
}

#[test]
fn switched_lj_bain_band_matches_the_scan() {
    let n_images = 9;
    let mut positions = Array2::zeros((n_images, 6));
    let mut cells = Vec::with_capacity(n_images);
    for i in 0..n_images {
        let t = i as f64 / (n_images - 1) as f64;
        let a = (1.0 - t) * A_BCC + t * A_FCC;
        let c = (1.0 - t) * A_BCC + t * C_FCC;
        let (h, x) = bct(a, c);
        cells.push(Cell(h));
        positions.row_mut(i).assign(&Array1::from_vec(x.to_vec()));
    }
    let mut band = BandSession::new(
        BandConfig {
            climbing: Some(CiConfig::after(0.5)),
            force_tol: 1e-4,
            max_move: 0.1,
            min_mode: None,
            solid: Some(SolidState::new(cells)),
            method: Method::Fire { kind: FireKind::V2 },
            ..BandConfig::default()
        },
        positions,
    )
    .unwrap();
    let mut rows = 0usize;
    let mut report = band.step(&Crystal).unwrap();
    rows += report.surface_rows;
    let mut steps = 1usize;
    while report.status == BandStatus::Running && steps < 4000 {
        report = band.step(&Crystal).unwrap();
        rows += report.surface_rows;
        steps += 1;
    }
    assert_eq!(
        report.status,
        BandStatus::Converged,
        "force {}",
        report.max_force
    );
    let ci = report.ci_index.expect("climbing image");
    let q = ratio(&band.image_cells().unwrap()[ci]);
    let (e_ci, _, _) = lj(
        band.positions().row(ci).as_slice().unwrap(),
        &band.image_cells().unwrap()[ci].0,
    );
    let barrier = e_ci / 2.0 - E_BCC;
    eprintln!("bain band steps {steps} surface_rows {rows} c/a {q} barrier {barrier}");
    assert!((q - Q_SADDLE).abs() < 2.5e-4, "c/a {q}");
    assert!((barrier - BARRIER).abs() < 1e-4, "barrier {barrier}");
}
