//! Mishin copper, the EAM alloy table NIST converted from the 2001 files.
//!
//! Y. Mishin, M. J. Mehl, D. A. Papaconstantopoulos, A. F. Voter, and
//! J. D. Kress, Phys. Rev. B 63, 224106 (2001), doi:10.1103/PhysRevB.63.224106.
//! The spline is the natural cubic spline used for a setfl table: value
//! and derivative come from the same second-derivative array, so the
//! force is the gradient of the interpolated energy.

use ndarray::{Array1, ArrayView1};

pub struct Eam {
    embed: Table,
    density: Table,
    pair: Table,
    cutoff: f64,
}

struct Table {
    y: Vec<f64>,
    y2: Vec<f64>,
    dx: f64,
}

impl Table {
    fn new(y: Vec<f64>, dx: f64) -> Self {
        let y2 = spline_y2(&y, dx);
        Self { y, y2, dx }
    }

    fn at(&self, x: f64) -> (f64, f64) {
        let n = self.y.len();
        if n < 2 || x <= 0.0 {
            return (self.y.first().copied().unwrap_or(0.0), 0.0);
        }
        let i = (x / self.dx) as usize;
        if i + 1 >= n {
            return (self.y[n - 1], 0.0);
        }
        let h = self.dx;
        let b = (x - i as f64 * h) / h;
        let a = 1.0 - b;
        let y = &self.y;
        let y2 = &self.y2;
        let val = a * y[i]
            + b * y[i + 1]
            + ((a * a * a - a) * y2[i] + (b * b * b - b) * y2[i + 1]) * (h * h) / 6.0;
        let der = (y[i + 1] - y[i]) / h
            + ((3.0 * b * b - 1.0) * y2[i + 1] - (3.0 * a * a - 1.0) * y2[i]) * h / 6.0;
        (val, der)
    }
}

fn spline_y2(y: &[f64], h: f64) -> Vec<f64> {
    let n = y.len();
    let mut y2 = vec![0.0; n];
    let mut u = vec![0.0; n];
    for i in 1..n.saturating_sub(1) {
        let sig = 0.5;
        let p = sig * y2[i - 1] + 2.0;
        y2[i] = (sig - 1.0) / p;
        let yy = (y[i + 1] - y[i]) / h - (y[i] - y[i - 1]) / h;
        u[i] = (6.0 * yy / (2.0 * h) - sig * u[i - 1]) / p;
    }
    for k in (0..n.saturating_sub(1)).rev() {
        y2[k] = y2[k] * y2[k + 1] + u[k];
    }
    y2
}

impl Eam {
    pub fn load(text: &str) -> Self {
        let tokens: Vec<&str> = text.split_whitespace().collect();
        let idx = tokens
            .windows(2)
            .position(|pair| pair[0] == "1" && pair[1] == "Cu")
            .expect("Mishin setfl header");
        let nrho: usize = tokens[idx + 2].parse().unwrap();
        let drho: f64 = tokens[idx + 3].parse().unwrap();
        let nr: usize = tokens[idx + 4].parse().unwrap();
        let dr: f64 = tokens[idx + 5].parse().unwrap();
        let cutoff: f64 = tokens[idx + 6].parse().unwrap();
        let nums: Vec<f64> = tokens[idx + 11..]
            .iter()
            .map(|token| token.parse().unwrap())
            .collect();
        assert_eq!(nums.len(), nrho + 2 * nr, "setfl table length");
        let embed = Table::new(nums[..nrho].to_vec(), drho);
        let density = Table::new(nums[nrho..nrho + nr].to_vec(), dr);
        let pair = Table::new(nums[nrho + nr..].to_vec(), dr);
        Self {
            embed,
            density,
            pair,
            cutoff,
        }
    }

    pub fn cutoff(&self) -> f64 {
        self.cutoff
    }

    /// Energy and Cartesian gradient. `box` is cubic and larger than the cutoff.
    pub fn energy_gradient(&self, x: ArrayView1<f64>, box_len: f64) -> (f64, Array1<f64>) {
        let n = x.len() / 3;
        let cut2 = self.cutoff * self.cutoff;
        let images = if box_len > 2.0 * self.cutoff { 0 } else { 1 };
        let mut rho = vec![0.0; n];
        let mut pair_energy = 0.0;
        let mut pairs: Vec<(usize, usize, [f64; 3], f64)> = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                for na in -images..=images {
                    for nb in -images..=images {
                        for nc in -images..=images {
                            if images == 0 && (na != 0 || nb != 0 || nc != 0) {
                                continue;
                            }
                            let dr = [
                                x[3 * j] - x[3 * i] + na as f64 * box_len,
                                x[3 * j + 1] - x[3 * i + 1] + nb as f64 * box_len,
                                x[3 * j + 2] - x[3 * i + 2] + nc as f64 * box_len,
                            ];
                            if images == 0 {
                                // Minimum image. The box is wider than twice the cutoff,
                                // so at most one image lies inside it.
                            }
                            let mut dr = dr;
                            if images == 0 {
                                for k in 0..3 {
                                    dr[k] -= box_len * (dr[k] / box_len).round();
                                }
                            }
                            let r2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                            if r2 >= cut2 || r2 == 0.0 {
                                continue;
                            }
                            let r = r2.sqrt();
                            let (density, _) = self.density.at(r);
                            rho[i] += density;
                            rho[j] += density;
                            let (z, _) = self.pair.at(r);
                            pair_energy += z / r;
                            pairs.push((i, j, dr, r));
                        }
                    }
                }
            }
        }
        let mut fp = vec![0.0; n];
        let mut energy = pair_energy;
        for i in 0..n {
            let (f, df) = self.embed.at(rho[i]);
            energy += f;
            fp[i] = df;
        }
        let mut g = vec![0.0; x.len()];
        for (i, j, dr, r) in pairs {
            let (z, dz) = self.pair.at(r);
            let dphi = (dz - z / r) / r;
            let (_, drho) = self.density.at(r);
            let dedr = (fp[i] + fp[j]) * drho + dphi;
            for k in 0..3 {
                let comp = dedr * dr[k] / r;
                g[3 * i + k] -= comp;
                g[3 * j + k] += comp;
            }
        }
        let mut mean = [0.0; 3];
        for atom in 0..n {
            for k in 0..3 {
                mean[k] += g[3 * atom + k];
            }
        }
        for m in &mut mean {
            *m /= n as f64;
        }
        for atom in 0..n {
            for k in 0..3 {
                g[3 * atom + k] -= mean[k];
            }
        }
        (energy, Array1::from(g))
    }
}
