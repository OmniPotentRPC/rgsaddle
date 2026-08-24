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
    let boxl = frame.header.boxl;
    let cell = if boxl.iter().all(|l| *l > 0.0) {
        Cell::ortho(boxl[0], boxl[1], boxl[2]).ok()
    } else {
        None
    };
    Ok(MolecularFrame {
        positions,
        masses,
        cell,
    })
}
