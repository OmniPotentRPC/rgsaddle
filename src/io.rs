//! Molecular I/O through readcon-core.
//!
//! This crate does not parse CON itself. Frames come from
//! [readcon-core](https://github.com/lode-org/readcon-core). Writers
//! use [`ConFrameWriter::new`]: RoundTrip, canonical left off, no
//! decimal precision. The `.con` file is the resume. A saddle step
//! does not open a path. [`crate::band::BandSession::restart`] stays
//! the in-memory wipe of FIRE, L-BFGS, and the climbing baseline.
//!
//! The `chemfiles` feature is readcon-core's chemfiles import.
//! `store_band` is the corpus append. It stays behind the `readcon-db`
//! cfg until that crate pins readcon-core 0.17.1.

use std::path::Path;

use ndarray::Array1;
use readcon_core::iterators::{read_all_frames, read_first_frame, read_nth_frame};
use readcon_core::types::{ConFrameBuilder, cell_vectors, lengths_and_angles};
use readcon_core::writer::ConFrameWriter;

use crate::error::SaddleError;
use crate::mic::Cell;

/// One CON frame as a 3N Cartesian, per-atom masses, and a linkcell box.
pub struct MolecularFrame {
    pub positions: Array1<f64>,
    pub masses: Array1<f64>,
    pub cell: Option<Cell>,
}

/// Species, masses, and cell shared by a geometry and by every image of a band.
///
/// `symbols` is one entry per atom, in the same order as the coordinate
/// triples. A missing vector is an error. This crate does not invent an element.
pub struct ConSpecies<'a> {
    pub symbols: &'a [&'a str],
    pub masses: &'a [f64],
    pub cell: Option<&'a Cell>,
}

/// One geometry or one band image.
///
/// `forces` is the flat 3N gradient. `None` omits the forces section.
/// `neb_bead` set here wins. On [`write_band`], `None` uses the file index.
pub struct ConImage<'a> {
    pub positions: &'a [f64],
    pub energy: Option<f64>,
    pub forces: Option<&'a [f64]>,
    pub neb_bead: Option<u64>,
}

/// Load the first frame of a CON file.
///
/// Positions are one copy of the frame's float64 slice. Column 5 is the
/// host order when those ids are a permutation of `0..N`. Otherwise the
/// buffer stays in file order.
pub fn frame_from_con(path: impl AsRef<Path>) -> Result<MolecularFrame, SaddleError> {
    let frame = read_first_frame(path.as_ref()).map_err(io_error)?;
    molecular_frame(frame)
}

/// Load frame `index` of a CON file.
///
/// `index == 0` is [`frame_from_con`]. This does not parse the other frames.
pub fn frame_from_con_nth(
    path: impl AsRef<Path>,
    index: usize,
) -> Result<MolecularFrame, SaddleError> {
    let frame = read_nth_frame(path.as_ref(), index).map_err(io_error)?;
    molecular_frame(frame)
}

/// Every frame of a band file, in file order.
pub fn frames_from_con(path: impl AsRef<Path>) -> Result<Vec<MolecularFrame>, SaddleError> {
    let frames = read_all_frames(path.as_ref()).map_err(io_error)?;
    frames.into_iter().map(molecular_frame).collect()
}

/// Write one geometry with [`ConFrameWriter::new`].
///
/// RoundTrip, canonical off. Forces are written only when the image has them.
pub fn write_restart(
    path: impl AsRef<Path>,
    species: &ConSpecies<'_>,
    image: &ConImage<'_>,
) -> Result<(), SaddleError> {
    let frame = build_frame(species, image, None)?;
    write_frames(path.as_ref(), std::slice::from_ref(&frame))
}

/// Write one frame per band image with [`ConFrameWriter::new`].
///
/// Symbols stay in coordinate order on every image. `neb_bead` falls
/// back to the file index. A frame carries forces only when that image
/// has a gradient.
pub fn write_band(
    path: impl AsRef<Path>,
    species: &ConSpecies<'_>,
    images: &[ConImage<'_>],
) -> Result<(), SaddleError> {
    if images.is_empty() {
        return Err(SaddleError::Invalid(
            "a band file needs one frame per image".into(),
        ));
    }
    let mut frames = Vec::with_capacity(images.len());
    for (index, image) in images.iter().enumerate() {
        frames.push(build_frame(species, image, Some(index as u64))?);
    }
    write_frames(path.as_ref(), &frames)
}

/// Append a band in one corpus call.
///
/// The numeric spelling is [`readcon_core::writer::FloatFormat::RoundTrip`].
/// A band under 64 images is one append, which is the call that can group.
/// This is not a release gate until a host calls it.
#[cfg(feature = "readcon-db")]
pub fn store_band(
    corpus: &readcon_db::ConCorpus,
    traj_id: u64,
    species: &ConSpecies<'_>,
    images: &[ConImage<'_>],
) -> Result<u32, SaddleError> {
    if images.is_empty() {
        return Err(SaddleError::Invalid(
            "a band corpus append needs one frame per image".into(),
        ));
    }
    let mut frames = Vec::with_capacity(images.len());
    for (index, image) in images.iter().enumerate() {
        frames.push(build_frame(species, image, Some(index as u64))?);
    }
    corpus
        .append_trajectory_frames_with_float_format(
            traj_id,
            &frames,
            "rgsaddle",
            readcon_core::writer::FloatFormat::RoundTrip,
        )
        .map_err(io_error)
}

fn molecular_frame(frame: readcon_core::types::ConFrame) -> Result<MolecularFrame, SaddleError> {
    let n = frame.positions.nrows();
    if n == 0 || frame.positions.ncols() != 3 {
        return Err(SaddleError::Shape("CON frame must be N by 3".into()));
    }
    let grouped = frame
        .positions
        .f64_slice()
        .ok_or_else(|| SaddleError::Invalid("CON positions are not float64".into()))?;
    if grouped.len() != n * 3 {
        return Err(SaddleError::Shape("CON frame must be N by 3".into()));
    }
    let order = frame.atom_ids.as_slice().and_then(|ids| host_order(ids, n));
    let positions = scatter(grouped, order.as_deref(), 3);
    let masses = if let Some(grouped_masses) = frame.masses.f64_slice()
        && grouped_masses.len() == n
    {
        scatter(grouped_masses, order.as_deref(), 1)
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

/// `grouped_for_host[host] = grouped index` when column 5 is a permutation of `0..N`.
fn host_order(ids: &[u64], n: usize) -> Option<Vec<usize>> {
    if ids.len() != n {
        return None;
    }
    let mut grouped_for_host = vec![0usize; n];
    let mut seen = vec![false; n];
    for (grouped, &id) in ids.iter().enumerate() {
        let host = usize::try_from(id).ok()?;
        if host >= n || seen[host] {
            return None;
        }
        seen[host] = true;
        grouped_for_host[host] = grouped;
    }
    Some(grouped_for_host)
}

fn scatter(grouped: &[f64], grouped_for_host: Option<&[usize]>, width: usize) -> Array1<f64> {
    let mut out = vec![0.0; grouped.len()];
    if let Some(index) = grouped_for_host {
        for (host, &grouped_index) in index.iter().enumerate() {
            let from = grouped_index * width;
            let to = host * width;
            out[to..to + width].copy_from_slice(&grouped[from..from + width]);
        }
    } else {
        out.copy_from_slice(grouped);
    }
    Array1::from_vec(out)
}

fn cell_from_header(
    header: &readcon_core::types::FrameHeader,
) -> Result<Option<Cell>, SaddleError> {
    let vectors = if let Some(vectors) = header.lattice_vectors() {
        vectors
    } else if header.boxl.iter().all(|length| *length == 0.0) {
        return Ok(None);
    } else {
        if !header
            .boxl
            .iter()
            .all(|length| length.is_finite() && *length > 0.0)
        {
            return Err(SaddleError::Invalid(
                "CON cell lengths must be finite and positive, or all zero".into(),
            ));
        }
        if !header
            .angles
            .iter()
            .all(|angle| angle.is_finite() && *angle > 0.0 && *angle < 180.0)
        {
            return Err(SaddleError::Invalid(
                "CON cell angles must lie between zero and 180 degrees".into(),
            ));
        }
        // cell_vectors clamps a negative cz^2 in (-1e-12, 0) to zero.
        cell_vectors(header.boxl, header.angles)
    };
    if vectors.iter().flatten().any(|value| !value.is_finite()) {
        return Err(SaddleError::Invalid(
            "CON cell angles define a singular or invalid lattice".into(),
        ));
    }
    Cell::from_vectors(vectors[0], vectors[1], vectors[2], [0.0; 3])
        .map(Some)
        .map_err(|error| SaddleError::Invalid(format!("invalid CON cell: {error}")))
}

fn build_frame(
    species: &ConSpecies<'_>,
    image: &ConImage<'_>,
    bead_fallback: Option<u64>,
) -> Result<readcon_core::types::ConFrame, SaddleError> {
    let n = species.symbols.len();
    if n == 0 {
        return Err(SaddleError::Invalid(
            "a CON write needs the symbol vector, in coordinate order".into(),
        ));
    }
    if species.symbols.iter().any(|symbol| symbol.is_empty()) {
        return Err(SaddleError::Invalid("CON symbol is empty".into()));
    }
    if species.masses.len() != n {
        return Err(SaddleError::Shape(
            "CON masses and symbols differ in length".into(),
        ));
    }
    if image.positions.len() != 3 * n {
        return Err(SaddleError::Shape(
            "CON positions must be 3N, one symbol per atom".into(),
        ));
    }
    require_finite(image.positions, "CON positions")?;
    require_finite(species.masses, "CON masses")?;
    if let Some(forces) = image.forces {
        if forces.len() != 3 * n {
            return Err(SaddleError::Shape("CON forces must be 3N".into()));
        }
        require_finite(forces, "CON forces")?;
    }
    if let Some(energy) = image.energy
        && !energy.is_finite()
    {
        return Err(SaddleError::NonFinite("CON energy"));
    }
    agree_masses(species)?;

    let mut lattice = None;
    let (lengths, angles) = if let Some(cell) = species.cell {
        let rows = [cell.a(), cell.b(), cell.c()];
        if rows.iter().flatten().any(|value| !value.is_finite()) {
            return Err(SaddleError::NonFinite("CON cell"));
        }
        let (lengths, angles) = lengths_and_angles(rows);
        if !(lengths
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
            && angles
                .iter()
                .all(|value| value.is_finite() && *value > 0.0 && *value < 180.0))
        {
            return Err(SaddleError::Invalid(
                "CON cell is singular or invalid".into(),
            ));
        }
        lattice = Some(rows);
        (lengths, angles)
    } else {
        ([0.0; 3], [0.0; 3])
    };

    let mut builder = ConFrameBuilder::new(lengths, angles);
    builder.prebox_header("rgsaddle");
    for (i, symbol) in species.symbols.iter().enumerate() {
        builder.add_atom(
            symbol,
            0.0,
            0.0,
            0.0,
            [false; 3],
            i as u64,
            species.masses[i],
        );
    }
    builder
        .set_positions_from_flat(image.positions)
        .map_err(io_error)?;
    if let Some(forces) = image.forces {
        builder.set_forces_from_flat(forces).map_err(io_error)?;
    }
    if let Some(energy) = image.energy {
        builder.set_energy(energy);
    }
    if let Some(bead) = image.neb_bead.or(bead_fallback) {
        builder.set_neb_bead(bead);
    }
    let mut frame = builder.build().map_err(io_error)?;
    if let Some(rows) = lattice {
        frame.header.set_lattice_vectors(rows);
    }
    Ok(frame)
}

fn agree_masses(species: &ConSpecies<'_>) -> Result<(), SaddleError> {
    for i in 0..species.symbols.len() {
        for j in 0..i {
            if species.symbols[i] == species.symbols[j] && species.masses[i] != species.masses[j] {
                return Err(SaddleError::Invalid(format!(
                    "symbol {} has more than one mass",
                    species.symbols[i]
                )));
            }
        }
    }
    Ok(())
}

fn require_finite(values: &[f64], what: &'static str) -> Result<(), SaddleError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(SaddleError::NonFinite(what))
    }
}

fn write_frames(path: &Path, frames: &[readcon_core::types::ConFrame]) -> Result<(), SaddleError> {
    let mut bytes = Vec::new();
    {
        let mut writer = ConFrameWriter::new(&mut bytes);
        for frame in frames {
            writer.write_frame(frame).map_err(io_error)?;
        }
        writer.flush().map_err(io_error)?;
    }
    std::fs::write(path, bytes).map_err(io_error)
}

fn io_error(err: impl std::fmt::Display) -> SaddleError {
    SaddleError::Surface(err.to_string())
}
