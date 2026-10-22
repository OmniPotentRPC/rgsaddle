//! A `.con` file written in the middle of a band matches the run that
//! never stopped.
//!
//! The file is the geometry. Reloading those bits with
//! `BandSession::set_positions` leaves FIRE, L-BFGS, and the climbing
//! baseline in place, because the rows did not move. `BandSession::restart`
//! is the in-memory wipe and is not this resume.

#![cfg(feature = "readcon")]

use ndarray::{Array1, Array2, ArrayView2};
use rgsaddle::{
    BandConfig, BandSession, BandSurface, ConImage, ConSpecies, MinModeRefine, SaddleError,
    frames_from_con, write_band,
};

const IMAGES: usize = 7;
const BEFORE: usize = 4;
const AFTER: usize = 4;

struct DoubleWell;

impl BandSurface for DoubleWell {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (i, row) in positions.outer_iter().enumerate() {
            let (x, y, z) = (row[0], row[1], row[2]);
            energies[i] = (x * x - 1.0).powi(2) + 2.0 * y * y + 2.0 * z * z;
            gradients[(i, 0)] = 4.0 * x * (x * x - 1.0);
            gradients[(i, 1)] = 4.0 * y;
            gradients[(i, 2)] = 4.0 * z;
        }
        Ok(())
    }
}

fn initial_band(n_images: usize) -> Array2<f64> {
    let mut band = Array2::zeros((n_images, 3));
    for i in 0..n_images {
        let t = i as f64 / (n_images - 1) as f64;
        band[(i, 0)] = -1.0 + 2.0 * t;
        band[(i, 1)] = 0.3 * (std::f64::consts::PI * t).sin();
    }
    band
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().copied().map(f64::to_bits).collect()
}

fn session() -> BandSession {
    let config = BandConfig {
        force_tol: 1e-12,
        max_move: 0.05,
        min_mode: Some(MinModeRefine::default()),
        ..BandConfig::default()
    };
    BandSession::new(config, initial_band(IMAGES)).unwrap()
}

fn images(rows: &[Vec<f64>]) -> Vec<ConImage<'_>> {
    rows.iter()
        .map(|row| ConImage {
            positions: row.as_slice(),
            energy: None,
            forces: None,
            neb_bead: None,
        })
        .collect()
}

fn owned_rows(positions: ArrayView2<f64>) -> Vec<Vec<f64>> {
    positions
        .rows()
        .into_iter()
        .map(|row| row.to_vec())
        .collect()
}

struct Tmp(std::path::PathBuf);

impl Tmp {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "rgsaddle-{name}-{}-{}.con",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        Self(path)
    }
}

impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[test]
fn mid_run_con_restart_matches_an_uninterrupted_run() {
    let symbols = ["H"];
    let masses = [1.008];
    let species = ConSpecies {
        symbols: &symbols,
        masses: &masses,
        cell: None,
    };
    let surface = DoubleWell;
    let mut straight = session();
    let mut resumed = session();
    for _ in 0..BEFORE {
        straight.step(&surface).unwrap();
        resumed.step(&surface).unwrap();
    }
    let checkpoint = resumed.positions().to_owned();
    assert_eq!(checkpoint.len(), IMAGES * 3);
    assert_ne!(
        bits(checkpoint.as_slice().unwrap()),
        bits(initial_band(IMAGES).as_slice().unwrap())
    );

    let path = Tmp::new("mid-run");
    let rows = owned_rows(checkpoint.view());
    write_band(&path.0, &species, &images(&rows)).unwrap();
    let loaded = frames_from_con(&path.0).unwrap();
    assert_eq!(loaded.len(), IMAGES);
    for (image, frame) in loaded.iter().enumerate() {
        assert_eq!(
            bits(frame.positions.as_slice().unwrap()),
            bits(checkpoint.row(image).as_slice().unwrap())
        );
    }
    let mut flat = Vec::with_capacity(IMAGES * 3);
    for frame in &loaded {
        flat.extend_from_slice(frame.positions.as_slice().unwrap());
    }
    let stacked = Array2::from_shape_vec((IMAGES, 3), flat).unwrap();
    resumed.set_positions(stacked).unwrap();

    let mut refined = false;
    for _ in 0..AFTER {
        refined |= straight.step(&surface).unwrap().curvature.is_some();
        resumed.step(&surface).unwrap();
    }
    assert!(refined, "the climbing image was not refined");
    let straight_final = straight.positions().to_owned();
    let resumed_final = resumed.positions().to_owned();
    assert_eq!(straight_final.len(), 21);
    assert_ne!(
        bits(straight_final.as_slice().unwrap()),
        bits(checkpoint.as_slice().unwrap())
    );
    assert_eq!(
        bits(resumed_final.as_slice().unwrap()),
        bits(straight_final.as_slice().unwrap())
    );

    let straight_file = Tmp::new("straight");
    let resumed_file = Tmp::new("resumed");
    let straight_rows = owned_rows(straight_final.view());
    let resumed_rows = owned_rows(resumed_final.view());
    write_band(&straight_file.0, &species, &images(&straight_rows)).unwrap();
    write_band(&resumed_file.0, &species, &images(&resumed_rows)).unwrap();
    let straight_bytes = std::fs::read(&straight_file.0).unwrap();
    let resumed_bytes = std::fs::read(&resumed_file.0).unwrap();
    assert_eq!(straight_bytes, resumed_bytes);
    assert_eq!(straight_bytes.len(), 1602);
}
