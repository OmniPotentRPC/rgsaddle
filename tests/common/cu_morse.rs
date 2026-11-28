use ndarray::{Array1, Array2, ArrayView1};
use rgsaddle::{Cell, PointSurface, SaddleError};

pub const ENERGY_SCALE: f64 = 0.3429;
pub const LENGTH_SCALE: f64 = 2.866;
const RHO: f64 = 1.3588 * LENGTH_SCALE;
const CUTOFF: f64 = 7.7;

pub struct CuMorse {
    cell: Cell,
    translations: Vec<[f64; 3]>,
}

impl CuMorse {
    pub fn vacancy_4x4x4() -> (Self, Array1<f64>, [f64; 3]) {
        let lattice = 2.0_f64.sqrt() * LENGTH_SCALE;
        let primitive = [
            [0.0, 0.5 * lattice, 0.5 * lattice],
            [0.5 * lattice, 0.0, 0.5 * lattice],
            [0.5 * lattice, 0.5 * lattice, 0.0],
        ];
        let cell = Cell(primitive.map(|row| row.map(|value| 4.0 * value)));
        let mut positions = Vec::with_capacity(3 * 63);
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    if i == 0 && j == 0 && k == 0 {
                        continue;
                    }
                    // The axis addresses all three lattice vectors at once.
                    #[allow(clippy::needless_range_loop)]
                    for axis in 0..3 {
                        positions.push(
                            i as f64 * primitive[0][axis]
                                + j as f64 * primitive[1][axis]
                                + k as f64 * primitive[2][axis],
                        );
                    }
                }
            }
        }
        let translations = (-1..=1)
            .flat_map(|i| {
                (-1..=1).flat_map(move |j| {
                    (-1..=1).map(move |k| {
                        std::array::from_fn(|axis| {
                            i as f64 * cell.0[0][axis]
                                + j as f64 * cell.0[1][axis]
                                + k as f64 * cell.0[2][axis]
                        })
                    })
                })
            })
            .collect();
        (
            Self { cell, translations },
            Array1::from_vec(positions),
            [0.0; 3],
        )
    }

    pub fn displacement(&self, from: ArrayView1<f64>, to: [f64; 3]) -> Array1<f64> {
        let mut displacement = Array1::from_iter((0..3).map(|axis| to[axis] - from[axis]));
        self.cell.minimum_image(&mut displacement);
        displacement
    }
}

impl PointSurface for CuMorse {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let atoms = x.len() / 3;
        let mut energy = 0.0;
        let mut gradient = Array1::zeros(x.len());
        let (cutoff_energy, cutoff_coefficient, _) =
            rgpot_pair::morse_pair(ENERGY_SCALE, RHO, LENGTH_SCALE, CUTOFF * CUTOFF);
        let cutoff_slope = cutoff_coefficient * CUTOFF;
        for i in 0..atoms {
            for j in (i + 1)..atoms {
                for translation in &self.translations {
                    let delta: [f64; 3] = std::array::from_fn(|axis| {
                        x[3 * i + axis] - x[3 * j + axis] - translation[axis]
                    });
                    let square = delta.iter().map(|value| value * value).sum::<f64>();
                    if !(square > 0.0 && square < CUTOFF * CUTOFF) {
                        continue;
                    }
                    let r = square.sqrt();
                    let (pair_energy, coefficient, _) =
                        rgpot_pair::morse_pair(ENERGY_SCALE, RHO, LENGTH_SCALE, square);
                    energy += pair_energy - cutoff_energy - (r - CUTOFF) * cutoff_slope;
                    let shifted_coefficient = coefficient - cutoff_slope / r;
                    for axis in 0..3 {
                        let component = shifted_coefficient * delta[axis];
                        gradient[3 * i + axis] += component;
                        gradient[3 * j + axis] -= component;
                    }
                }
            }
        }
        Ok((energy, gradient))
    }

    fn excluded_modes(&self, x: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
        let mut translations = Array2::zeros((3, x.len()));
        for atom in 0..x.len() / 3 {
            for axis in 0..3 {
                translations[(axis, 3 * atom + axis)] = 1.0;
            }
        }
        Ok(translations)
    }
}
