//! Molecular I/O: readcon-core, optional readcon-db and chemfiles.
//!
//! This crate does not parse CON itself. Frames come from
//! [readcon-core](https://github.com/lode-org/readcon-core). The
//! `chemfiles` feature is readcon-core's chemfiles import (foreign
//! trajectories). `readcon-db` is the mmap corpus for traces.

use ndarray::Array1;
use readcon_core::iterators::read_first_frame;

use crate::error::SaddleError;
use crate::mic::Cell;

/// One CON frame as a 3N Cartesian, per-atom masses, and a linkcell box.
pub struct MolecularFrame {
    pub positions: Array1<f64>,
    pub masses: Array1<f64>,
    pub cell: Option<Cell>,
}

/// Load the first frame of a CON file for IRC / min-mode / band hosts.
pub fn frame_from_con(path: impl AsRef<std::path::Path>) -> Result<MolecularFrame, SaddleError> {
    let frame = read_first_frame(path.as_ref()).map_err(|e| SaddleError::Surface(e.to_string()))?;
    let n = frame.positions.nrows();
    if n == 0 || frame.positions.ncols() != 3 {
        return Err(SaddleError::Shape("CON frame must be N by 3".into()));
    }
    let mut positions = Array1::zeros(n * 3);
    for i in 0..n {
        let row = frame.positions.as_f64_row(i);
        positions[3 * i] = row[0];
        positions[3 * i + 1] = row[1];
        positions[3 * i + 2] = row[2];
    }
    let masses = if frame.masses.len() == n {
        Array1::from((0..n).map(|i| frame.masses.get_f64(i)).collect::<Vec<_>>())
    } else {
        Array1::ones(n)
    };
    let cell = cell_from_header(&frame.header)?;
    Ok(MolecularFrame {
        positions,
        masses,
        cell,
    })
}


fn cell_from_header(header: &readcon_core::types::FrameHeader) -> Result<Option<Cell>, SaddleError> {
    let vectors = if let Some(vectors) = header.lattice_vectors() {
        vectors
    } else {
        let lengths = header.boxl;
        if lengths.iter().all(|length| *length == 0.0) { return Ok(None); }
        if !lengths.iter().all(|length| length.is_finite() && *length > 0.0) {
            return Err(SaddleError::Invalid("CON cell lengths must be finite and positive, or all zero".into()));
        }
        if !header.angles.iter().all(|angle| angle.is_finite() && *angle > 0.0 && *angle < 180.0) {
            return Err(SaddleError::Invalid("CON cell angles must lie between zero and 180 degrees".into()));
        }
        let [alpha, beta, gamma] = header.angles.map(f64::to_radians);
        let cy = (alpha.cos() - beta.cos() * gamma.cos()) / gamma.sin();
        let cz_squared = 1.0 - beta.cos().powi(2) - cy * cy;
        if !(cz_squared.is_finite() && cz_squared > 0.0) {
            return Err(SaddleError::Invalid("CON cell angles define a singular or invalid lattice".into()));
        }
        [
            [lengths[0], 0.0, 0.0],
            [lengths[1] * gamma.cos(), lengths[1] * gamma.sin(), 0.0],
            [lengths[2] * beta.cos(), lengths[2] * cy, lengths[2] * cz_squared.sqrt()],
        ]
    };
    Cell::from_vectors(vectors[0], vectors[1], vectors[2], [0.0; 3])
        .map(Some).map_err(|error| SaddleError::Invalid(format!("invalid CON cell: {error}")))
}
