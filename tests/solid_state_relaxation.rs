//! Fast checks of atom-strain relaxation and stress loading.

use ndarray::{Array1, ArrayView1, array};
use rgmin::{Control, Method, Oracle, Solver, StopNorm};
use rgsaddle::{
    CellMask, ElasticConstants, JointChart, PointSurface, SaddleError,
    finite_difference_cauchy_stress,
};

const LENGTH: f64 = 3.0;
const SITE: [f64; 3] = [0.25, 0.4, 0.1];
const ATOM_STIFFNESS: f64 = 5.0;
const C11: f64 = 12.0;
const C12: f64 = 4.0;
const C44: f64 = 3.0;
const COUPLING: f64 = 0.4;
const SHEAR_COUPLING: f64 = 0.25;

type Tensor4 = [[[[f64; 3]; 3]; 3]; 3];

fn reference() -> [[f64; 3]; 3] {
    [[LENGTH, 0.0, 0.0], [0.0, LENGTH, 0.0], [0.0, 0.0, LENGTH]]
}

/// `h0 (I + ε)` for the cubic reference.
fn strained(strain: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut cell = reference();
    for (i, row) in cell.iter_mut().enumerate() {
        for (j, value) in row.iter_mut().enumerate() {
            *value += LENGTH * strain[i][j];
        }
    }
    cell
}

fn flat(cell: [[f64; 3]; 3]) -> [f64; 9] {
    [
        cell[0][0], cell[0][1], cell[0][2], cell[1][0], cell[1][1], cell[1][2], cell[2][0],
        cell[2][1], cell[2][2],
    ]
}

fn matrix(cell: &[f64; 9]) -> [[f64; 3]; 3] {
    [
        [cell[0], cell[1], cell[2]],
        [cell[3], cell[4], cell[5]],
        [cell[6], cell[7], cell[8]],
    ]
}

fn volume(cell: [[f64; 3]; 3]) -> f64 {
    (cell[0][0] * (cell[1][1] * cell[2][2] - cell[1][2] * cell[2][1])
        - cell[0][1] * (cell[1][0] * cell[2][2] - cell[1][2] * cell[2][0])
        + cell[0][2] * (cell[1][0] * cell[2][1] - cell[1][1] * cell[2][0]))
        .abs()
}

/// Lower-triangular index of each Voigt strain, as in `CellMask::voigt`.
const VOIGT: [(usize, usize); 6] = [(0, 0), (1, 1), (2, 2), (2, 1), (2, 0), (1, 0)];

/// Closed-form cubic response `(ε_load, ε_transverse)` to a normal load
/// `p` in the chart's pressure sign.
fn normal_response(pressure: f64) -> (f64, f64) {
    let scale = (C11 - C12) * (C11 + 2.0 * C12);
    (-pressure * (C11 + C12) / scale, pressure * C12 / scale)
}

/// One atom tied to the fractional site `SITE`, shifted by the xx and
/// yx strains, in a cell whose strain energy is `V0 e : C : e / 2` with
/// `e` the symmetric part of `h0⁻¹ h - I`.
struct CoupledCell {
    stiffness: Tensor4,
    analytic_stress: bool,
}

impl CoupledCell {
    fn new(analytic_stress: bool) -> Self {
        Self {
            stiffness: ElasticConstants::cubic(C11, C12, C44).unwrap().tensor(),
            analytic_stress,
        }
    }

    fn evaluate(
        &self,
        positions: ArrayView1<f64>,
        cell: [[f64; 3]; 3],
    ) -> (f64, Array1<f64>, [[f64; 3]; 3]) {
        let reference_volume = LENGTH.powi(3);
        let mut strain = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                strain[i][j] = cell[i][j] / LENGTH - if i == j { 1.0 } else { 0.0 };
            }
        }
        let symmetric = |k: usize, l: usize| 0.5 * (strain[k][l] + strain[l][k]);
        let mut traction = [[0.0; 3]; 3];
        for (row, block) in traction.iter_mut().zip(&self.stiffness) {
            for (value, plane) in row.iter_mut().zip(block) {
                for (k, line) in plane.iter().enumerate() {
                    for (l, entry) in line.iter().enumerate() {
                        *value += entry * symmetric(k, l);
                    }
                }
            }
        }
        let mut energy = 0.0;
        let mut cell_derivative = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                energy += 0.5 * reference_volume * traction[i][j] * symmetric(i, j);
                cell_derivative[i][j] = reference_volume * traction[i][j] / LENGTH;
            }
        }
        let offset = [COUPLING * strain[0][0], SHEAR_COUPLING * strain[1][0], 0.0];
        let mut gradient = Array1::zeros(positions.len());
        for c in 0..3 {
            let site: f64 = (0..3).map(|k| SITE[k] * cell[k][c]).sum();
            let displacement = positions[c] - site - offset[c];
            energy += 0.5 * ATOM_STIFFNESS * displacement * displacement;
            gradient[c] = ATOM_STIFFNESS * displacement;
            for (k, row) in cell_derivative.iter_mut().enumerate() {
                row[c] -= gradient[c] * SITE[k];
            }
        }
        cell_derivative[0][0] -= gradient[0] * COUPLING / LENGTH;
        cell_derivative[1][0] -= gradient[1] * SHEAR_COUPLING / LENGTH;
        let v = volume(cell);
        let mut stress = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                let cell_part: f64 = (0..3).map(|k| cell[k][i] * cell_derivative[k][j]).sum();
                stress[i][j] = (positions[i] * gradient[j] + cell_part) / v;
            }
        }
        (energy, gradient, stress)
    }

    /// Cartesian position that zeroes the atom force in `cell`.
    fn equilibrium(&self, cell: [[f64; 3]; 3]) -> [f64; 3] {
        let offset = [
            COUPLING * (cell[0][0] / LENGTH - 1.0),
            SHEAR_COUPLING * cell[1][0] / LENGTH,
            0.0,
        ];
        std::array::from_fn(|c| (0..3).map(|k| SITE[k] * cell[k][c]).sum::<f64>() + offset[c])
    }
}

impl PointSurface for CoupledCell {
    fn eval(&self, positions: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        self.eval_in_cell(positions, &flat(reference()))
    }

    fn eval_in_cell(
        &self,
        positions: ArrayView1<f64>,
        cell: &[f64; 9],
    ) -> Result<(f64, Array1<f64>), SaddleError> {
        let (energy, gradient, _) = self.evaluate(positions, matrix(cell));
        Ok((energy, gradient))
    }

    fn cauchy_stress(
        &self,
        positions: ArrayView1<f64>,
        cell: &[f64; 9],
    ) -> Result<Option<[f64; 9]>, SaddleError> {
        Ok(self
            .analytic_stress
            .then(|| flat(self.evaluate(positions, matrix(cell)).2)))
    }
}

/// Enthalpy and its chart gradient, with the surface stress or its
/// finite difference.
fn joint(surface: &CoupledCell, chart: &JointChart, q: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let (positions, cell) = chart.geometry(q).expect("valid joint geometry");
    let packed = flat(cell);
    let (energy, cartesian) = surface
        .eval_in_cell(positions.view(), &packed)
        .expect("toy surface");
    let stress = match surface
        .cauchy_stress(positions.view(), &packed)
        .expect("toy stress")
    {
        Some(stress) => matrix(&stress),
        None => finite_difference_cauchy_stress(surface, positions.view(), cell, 1.0e-6)
            .expect("finite difference stress"),
    };
    (
        chart.enthalpy(energy, q).expect("finite enthalpy"),
        chart
            .gradient(q, cartesian.view(), stress)
            .expect("joint gradient"),
    )
}

/// Central differences of the joint enthalpy against its gradient.
fn assert_gradient_matches(surface: &CoupledCell, chart: &JointChart, q: ArrayView1<f64>) {
    let step = 1.0e-5;
    let (_, gradient) = joint(surface, chart, q);
    for k in 0..q.len() {
        let mut plus = q.to_owned();
        let mut minus = q.to_owned();
        plus[k] += step;
        minus[k] -= step;
        let finite = (joint(surface, chart, plus.view()).0 - joint(surface, chart, minus.view()).0)
            / (2.0 * step);
        assert!(
            (finite - gradient[k]).abs() < 1.0e-7 * (1.0 + gradient[k].abs()),
            "gradient [{k}] finite {finite} analytic {}",
            gradient[k]
        );
    }
}

fn relax(surface: &CoupledCell, chart: &JointChart, mut generalized: Array1<f64>) -> Array1<f64> {
    assert_gradient_matches(surface, chart, generalized.view());
    let dimension = generalized.len();
    let oracle = Oracle::unbounded(dimension, |q: ArrayView1<f64>| joint(surface, chart, q));
    let mut solver = Solver::new(
        Method::lbfgs_with_memory(8),
        Control::default()
            .with_maxiter(300)
            .with_gtol(1.0e-10)
            .with_stop_norm(StopNorm::Inf)
            .with_istep(0.1),
        dimension,
    );
    solver
        .run(&oracle, &mut generalized)
        .expect("joint relaxation");
    generalized
}

/// Generalized vector of `chart`, built with mask `free`, holding the
/// free entries of `strain`.
fn generalized(
    chart: &JointChart,
    free: [bool; 6],
    positions: [f64; 3],
    strain: [[f64; 3]; 3],
) -> Array1<f64> {
    let mut q = Array1::zeros(chart.dimension());
    for c in 0..3 {
        q[c] = positions[c];
    }
    let mut k = 3;
    for (on, (i, j)) in free.into_iter().zip(VOIGT) {
        if on {
            q[k] = strain[i][j] * chart.jacobian();
            k += 1;
        }
    }
    q
}

const ALL_FREE: [bool; 6] = [true; 6];
const NORMAL_AND_XY: [bool; 6] = [true, true, true, false, false, true];

fn assert_relaxed(
    surface: &CoupledCell,
    chart: &JointChart,
    relaxed: ArrayView1<f64>,
    expected: [[f64; 3]; 3],
) {
    let strain = chart.strain(relaxed).unwrap();
    for (i, j) in VOIGT {
        assert!(
            (strain[i][j] - expected[i][j]).abs() < 1.0e-8,
            "strain [{i}, {j}] relaxed {} expected {}",
            strain[i][j],
            expected[i][j]
        );
    }
    let (positions, cell) = chart.geometry(relaxed).unwrap();
    let site = surface.equilibrium(cell);
    for c in 0..3 {
        assert!(
            (positions[c] - site[c]).abs() < 1.0e-8,
            "atom [{c}] relaxed {} expected {}",
            positions[c],
            site[c]
        );
    }
}

#[test]
fn chart_strain_geometry_and_enthalpy_follow_the_stored_coordinates() {
    let mut external = [[0.0; 3]; 3];
    external[0][0] = 0.3;
    external[2][1] = -0.2;
    external[1][0] = 0.15;
    let chart = JointChart::new(reference(), 1, CellMask::voigt(ALL_FREE), external, 1.0).unwrap();
    assert!((chart.jacobian() - LENGTH).abs() < 1.0e-12);
    let q = array![0.9, 1.1, 0.4, 0.06, -0.03, 0.09, 0.12, -0.15, 0.21];
    let strain = chart.strain(q.view()).unwrap();
    for (k, (i, j)) in VOIGT.into_iter().enumerate() {
        assert!((strain[i][j] - q[3 + k] / LENGTH).abs() < 1.0e-15);
    }
    for (i, j) in [(0, 1), (0, 2), (1, 2)] {
        assert_eq!(strain[i][j], 0.0);
    }
    let (positions, cell) = chart.geometry(q.view()).unwrap();
    let expected_cell = strained(strain);
    for i in 0..3 {
        for j in 0..3 {
            assert!((cell[i][j] - expected_cell[i][j]).abs() < 1.0e-14);
        }
        let row: f64 = (0..3)
            .map(|c| q[c] * (strain[c][i] + if c == i { 1.0 } else { 0.0 }))
            .sum();
        assert!((positions[i] - row).abs() < 1.0e-14);
    }
    let energy = 1.7;
    let work = 0.3 * strain[0][0] - 0.2 * strain[2][1] + 0.15 * strain[1][0];
    let enthalpy = chart.enthalpy(energy, q.view()).unwrap();
    assert!((enthalpy - (energy + LENGTH.powi(3) * work)).abs() < 1.0e-12);
}

#[test]
fn joint_gradient_matches_finite_differences_of_the_enthalpy() {
    let mut external = [[0.0; 3]; 3];
    external[0][0] = 0.3;
    external[2][2] = -0.1;
    external[2][1] = -0.2;
    external[2][0] = 0.05;
    external[1][0] = 0.15;
    let chart = JointChart::new(reference(), 1, CellMask::voigt(ALL_FREE), external, 1.0).unwrap();
    let strain = [[0.04, 0.0, 0.0], [0.07, -0.03, 0.0], [-0.05, 0.06, 0.02]];
    let q = generalized(&chart, ALL_FREE, [1.3, 0.2, -0.4], strain);
    for analytic_stress in [true, false] {
        assert_gradient_matches(&CoupledCell::new(analytic_stress), &chart, q.view());
    }
}

#[test]
fn finite_difference_stress_relaxation_reaches_the_loaded_minimum() {
    let pressure = 0.3;
    let shear = 0.2;
    let mut external = [[0.0; 3]; 3];
    external[0][0] = pressure;
    external[1][0] = shear;
    let chart = JointChart::new(
        reference(),
        1,
        CellMask::voigt(NORMAL_AND_XY),
        external,
        1.0,
    )
    .unwrap();
    let surface = CoupledCell::new(false);
    let start = generalized(
        &chart,
        NORMAL_AND_XY,
        [SITE[0] * LENGTH + 0.3, SITE[1] * LENGTH - 0.1, 0.2],
        [[0.08, 0.0, 0.0], [0.05, -0.02, 0.0], [0.0, 0.0, 0.03]],
    );
    let relaxed = relax(&surface, &chart, start);
    let (load, transverse) = normal_response(pressure);
    let expected = [
        [load, 0.0, 0.0],
        [-shear / C44, transverse, 0.0],
        [0.0, 0.0, transverse],
    ];
    assert_relaxed(&surface, &chart, relaxed.view(), expected);
}

#[test]
fn external_stress_gives_the_cubic_compliance_response() {
    let tension = -0.25;
    let mut external = [[0.0; 3]; 3];
    external[2][2] = tension;
    external[2][1] = 0.15;
    external[2][0] = -0.1;
    let chart = JointChart::new(reference(), 1, CellMask::voigt(ALL_FREE), external, 1.0).unwrap();
    let surface = CoupledCell::new(true);
    let start = generalized(
        &chart,
        ALL_FREE,
        [SITE[0] * LENGTH, SITE[1] * LENGTH, SITE[2] * LENGTH],
        [[0.0; 3]; 3],
    );
    let relaxed = relax(&surface, &chart, start);
    let (load, transverse) = normal_response(tension);
    let expected = [
        [transverse, 0.0, 0.0],
        [0.0, transverse, 0.0],
        [0.1 / C44, -0.15 / C44, load],
    ];
    assert_relaxed(&surface, &chart, relaxed.view(), expected);
}

#[test]
fn finite_difference_stress_matches_the_analytic_stress() {
    let surface = CoupledCell::new(true);
    let cell = strained([[0.07, 0.0, 0.0], [0.04, -0.02, 0.0], [-0.03, 0.05, 0.01]]);
    let positions = array![SITE[0] * cell[0][0] + 0.11, -0.2, 0.3];
    let analytic = surface.evaluate(positions.view(), cell).2;
    let finite =
        finite_difference_cauchy_stress(&CoupledCell::new(false), positions.view(), cell, 1.0e-6)
            .unwrap();
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (finite[i][j] - analytic[i][j]).abs() < 1.0e-8,
                "stress [{i}, {j}] finite {} analytic {}",
                finite[i][j],
                analytic[i][j]
            );
        }
    }
}
