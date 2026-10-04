//! The 2012 solid-state cell block on the band.

use ndarray::{Array1, Array2, ArrayView2, array};
use rgsaddle::{BandConfig, BandSession, BandStatus, BandSurface, Cell, SaddleError, SolidState};

fn diagonal(length: f64) -> Cell {
    Cell([[length, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]])
}

struct Push {
    gradient: [f64; 3],
    stress_xx: f64,
}

impl BandSurface for Push {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        energies.fill(0.0);
        for row in 0..positions.nrows() {
            gradients[(row, 0)] = self.gradient[0];
            gradients[(row, 1)] = self.gradient[1];
            gradients[(row, 2)] = self.gradient[2];
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
        assert_eq!(cells.len(), positions.nrows());
        self.eval(positions, energies, gradients)?;
        stresses.fill(0.0);
        for row in 0..positions.nrows() {
            stresses[(row, 0)] = self.stress_xx;
        }
        Ok(())
    }
}

#[test]
fn equal_cells_and_zero_stress_match_the_atomic_band() {
    let positions = array![[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [2.0, 0.0, 0.0]];
    let surface = Push {
        gradient: [0.0, 1.0, 0.0],
        stress_xx: 0.0,
    };
    let mut atomic_session = BandSession::new(
        BandConfig {
            climbing: None,
            max_move: 0.2,
            ..BandConfig::default()
        },
        positions.clone(),
    )
    .unwrap();
    let mut solid_session = BandSession::new(
        BandConfig {
            climbing: None,
            max_move: 0.2,
            solid: Some(SolidState {
                cells: vec![diagonal(5.0), diagonal(5.0), diagonal(5.0)],
                pressure: 0.0,
                weight: 1.0,
            }),
            ..BandConfig::default()
        },
        positions,
    )
    .unwrap();
    let atomic_report = atomic_session.step(&surface).unwrap();
    let solid_report = solid_session.step(&surface).unwrap();
    assert_eq!(atomic_report.status, BandStatus::Running);
    assert_eq!(solid_report.status, BandStatus::Running);
    for (left, right) in atomic_session
        .positions()
        .iter()
        .zip(solid_session.positions().iter())
    {
        assert!((left - right).abs() < 1e-9, "{left} vs {right}");
    }
    let cells = solid_session.image_cells().unwrap();
    for cell in cells {
        assert!((cell.0[0][0] - 5.0).abs() < 1e-12);
    }
}

#[test]
fn positive_pressure_shrinks_the_interior_cell() {
    let config = BandConfig {
        climbing: None,
        max_move: 0.05,
        solid: Some(SolidState {
            cells: vec![diagonal(5.0), diagonal(5.0), diagonal(5.0)],
            pressure: 0.02,
            weight: 1.0,
        }),
        ..BandConfig::default()
    };
    let mut session = BandSession::new(
        config,
        array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
    )
    .unwrap();
    assert!(session.solid_jacobian().unwrap() > 0.0);
    session
        .step(&Push {
            gradient: [0.0, 0.0, 0.0],
            stress_xx: 0.0,
        })
        .unwrap();
    let cells = session.image_cells().unwrap();
    assert!(cells[1].0[0][0] < 5.0, "length {}", cells[1].0[0][0]);
    assert!(cells[1].0[1][1] < 4.0, "width {}", cells[1].0[1][1]);
    assert!((cells[0].0[0][0] - 5.0).abs() < 1e-12);
    assert!((cells[2].0[0][0] - 5.0).abs() < 1e-12);
    assert!((cells[1].0[0][1]).abs() < 1e-12);
}

#[test]
fn a_longer_product_cell_pulls_the_middle_cell_out() {
    let config = BandConfig {
        climbing: None,
        max_move: 0.2,
        solid: Some(SolidState {
            cells: vec![diagonal(4.0), diagonal(4.0), diagonal(6.0)],
            pressure: 0.0,
            weight: 1.0,
        }),
        ..BandConfig::default()
    };
    // Matching fractional coordinates, so the atomic block of the
    // displacement is zero and only the strain remains.
    let mut session = BandSession::new(
        config,
        array![[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [3.0, 2.0, 2.0]],
    )
    .unwrap();
    session
        .step(&Push {
            gradient: [0.0, 0.0, 0.0],
            stress_xx: 0.0,
        })
        .unwrap();
    let cells = session.image_cells().unwrap();
    assert!(cells[1].0[0][0] > 4.0, "length {}", cells[1].0[0][0]);
    assert!((cells[0].0[0][0] - 4.0).abs() < 1e-12);
    assert!((cells[2].0[0][0] - 6.0).abs() < 1e-12);
}
