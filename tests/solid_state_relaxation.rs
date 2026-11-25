//! Fast checks of atom-strain relaxation and stress loading.

use ndarray::{Array1, ArrayView1, array};
use rgmin::{Control, Method, Oracle, Solver, StopNorm};
use rgsaddle::{
    CellMask, ElasticConstants, JointChart, PointSurface, SaddleError,
    finite_difference_cauchy_stress,
};

const LENGTH: f64 = 3.0;
const SITE: f64 = 0.25;
const ATOM_STIFFNESS: f64 = 5.0;
const STRAIN_STIFFNESS: f64 = 12.0;
const COUPLING: f64 = 0.4;

fn cell(strain: f64) -> [[f64; 3]; 3] {
    [
        [LENGTH * (1.0 + strain), 0.0, 0.0],
        [0.0, LENGTH, 0.0],
        [0.0, 0.0, LENGTH],
    ]
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
    cell[0][0] * cell[1][1] * cell[2][2]
}

struct CoupledCell {
    analytic_stress: bool,
}

impl CoupledCell {
    fn evaluate(
        &self,
        positions: ArrayView1<f64>,
        cell: [[f64; 3]; 3],
    ) -> (f64, Array1<f64>, [[f64; 3]; 3]) {
        let reference_volume = LENGTH.powi(3);
        let strain = cell[0][0] / LENGTH - 1.0;
        let displacement = positions[0] - SITE * cell[0][0];
        let coupled = displacement - COUPLING * strain;
        let energy = 0.5 * ATOM_STIFFNESS * coupled * coupled
            + 0.5 * reference_volume * STRAIN_STIFFNESS * strain * strain;
        let mut gradient = Array1::zeros(positions.len());
        gradient[0] = ATOM_STIFFNESS * coupled;
        let cell_derivative = -ATOM_STIFFNESS * coupled * (SITE + COUPLING / LENGTH)
            + reference_volume * STRAIN_STIFFNESS * strain / LENGTH;
        let mut stress = [[0.0; 3]; 3];
        for i in 0..3 {
            stress[i][0] =
                (gradient[0] * positions[i] + cell_derivative * cell[0][i]) / volume(cell);
        }
        (energy, gradient, stress)
    }
}

impl PointSurface for CoupledCell {
    fn eval(&self, positions: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        self.eval_in_cell(positions, &flat(cell(0.0)))
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

fn relax(surface: &CoupledCell, chart: &JointChart, mut generalized: Array1<f64>) -> Array1<f64> {
    let dimension = generalized.len();
    let oracle = Oracle::unbounded(dimension, |q: ArrayView1<f64>| {
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
    });
    let mut solver = Solver::new(
        Method::lbfgs_with_memory(8),
        Control::default()
            .with_maxiter(100)
            .with_gtol(1.0e-9)
            .with_stop_norm(StopNorm::Inf)
            .with_istep(0.1),
        dimension,
    );
    solver
        .run(&oracle, &mut generalized)
        .expect("joint relaxation");
    generalized
}

#[test]
fn coupled_atom_strain_relaxation_reaches_the_small_cell_minimum() {
    let reference = cell(0.0);
    let chart = JointChart::new(
        reference,
        1,
        CellMask::voigt([true, false, false, false, false, false]),
        [[0.0; 3]; 3],
        1.0,
    )
    .unwrap();
    let mut start = Array1::zeros(chart.dimension());
    start[0] = SITE * LENGTH + 0.3;
    start[3] = 0.08 * chart.jacobian();
    let relaxed = relax(
        &CoupledCell {
            analytic_stress: true,
        },
        &chart,
        start,
    );
    let (positions, relaxed_cell) = chart.geometry(relaxed.view()).unwrap();
    assert!((relaxed_cell[0][0] / LENGTH - 1.0).abs() < 1.0e-8);
    assert!((positions[0] - SITE * relaxed_cell[0][0]).abs() < 1.0e-8);
}

#[test]
fn external_stress_gives_the_elastic_strain_response() {
    let constants = ElasticConstants::cubic(STRAIN_STIFFNESS, 4.0, 3.0).unwrap();
    let c11 = constants.tensor()[0][0][0][0];
    let pressure = 0.3;
    let mut external = [[0.0; 3]; 3];
    external[0][0] = pressure;
    let chart = JointChart::new(
        cell(0.0),
        1,
        CellMask::voigt([true, false, false, false, false, false]),
        external,
        1.0,
    )
    .unwrap();
    let mut start = Array1::zeros(chart.dimension());
    start[0] = SITE * LENGTH;
    let relaxed = relax(
        &CoupledCell {
            analytic_stress: true,
        },
        &chart,
        start,
    );
    let strain = chart.strain(relaxed.view()).unwrap()[0][0];
    assert!((strain + pressure / c11).abs() < 1.0e-8);
}

#[test]
fn finite_difference_stress_matches_the_analytic_stress() {
    let surface = CoupledCell {
        analytic_stress: true,
    };
    let cell = cell(0.07);
    let positions = array![SITE * cell[0][0] + 0.11, -0.2, 0.3];
    let analytic = surface.evaluate(positions.view(), cell).2;
    let finite = finite_difference_cauchy_stress(
        &CoupledCell {
            analytic_stress: false,
        },
        positions.view(),
        cell,
        1.0e-6,
    )
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
