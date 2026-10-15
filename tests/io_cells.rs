#![cfg(feature = "readcon")]

use rgsaddle::{
    ConImage, ConSpecies, SaddleError, frame_from_con, frame_from_con_nth, frames_from_con,
    write_band, write_restart,
};

fn vectors(cell: &rgsaddle::mic::Cell) -> [[f64; 3]; 3] {
    [
        cell.cartesian([1.0, 0.0, 0.0]),
        cell.cartesian([0.0, 1.0, 0.0]),
        cell.cartesian([0.0, 0.0, 1.0]),
    ]
}

#[test]
fn con_lengths_and_angles_preserve_the_triclinic_metric() {
    let frame = frame_from_con(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/triclinic.con"
    ))
    .unwrap();
    assert_eq!(frame.positions.to_vec(), vec![1.0, 2.0, 3.0]);
    assert_eq!(frame.masses.to_vec(), vec![12.011]);
    let lattice = vectors(frame.cell.as_ref().unwrap());
    let lengths = [10.0_f64, 11.0, 12.0];
    for i in 0..3 {
        let squared = lattice[i].iter().map(|v| v * v).sum::<f64>();
        assert!((squared - lengths[i].powi(2)).abs() < 1e-11);
    }
    for (i, j, angle) in [(1, 2, 80.0_f64), (0, 2, 100.0), (0, 1, 60.0)] {
        let dot = lattice[i]
            .iter()
            .zip(lattice[j])
            .map(|(a, b)| a * b)
            .sum::<f64>();
        assert!((dot / (lengths[i] * lengths[j]) - angle.to_radians().cos()).abs() < 1e-13);
    }
}

#[test]
fn exact_lattice_metadata_takes_precedence_over_rounded_header_angles() {
    let frame = frame_from_con(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/exact_lattice.con"
    ))
    .unwrap();
    let actual = vectors(frame.cell.as_ref().unwrap());
    let expected = [
        [2.000000000002, 0.400000000003, 0.0],
        [0.0, 3.000000000004, 0.200000000005],
        [0.100000000006, 0.0, 4.000000000007],
    ];
    for i in 0..3 {
        for j in 0..3 {
            assert!((actual[i][j] - expected[i][j]).abs() < 1e-13);
        }
    }
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().copied().map(f64::to_bits).collect()
}

fn cell_bits(cell: &rgsaddle::mic::Cell) -> [u64; 9] {
    let rows = [cell.a(), cell.b(), cell.c()];
    let mut out = [0; 9];
    let mut k = 0;
    for row in rows {
        for value in row {
            out[k] = value.to_bits();
            k += 1;
        }
    }
    out
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
fn write_read_matches_bits_on_a_nonorthogonal_cell() {
    let neg_zero = -0.0_f64;
    let subnormal = f64::from_bits(1);
    let after_one = f64::from_bits(0x3ff0_0000_0000_0001);
    assert!(neg_zero.is_sign_negative());
    let symbols = ["Cu", "H", "Cu"];
    let masses = [63.546, 1.00784, 63.546];
    let cell = rgsaddle::mic::Cell::from_vectors(
        [10.0, 0.1, 0.0],
        [1.5, 8.0, 0.2],
        [0.25, -0.4, 9.0],
        [0.0; 3],
    )
    .expect("cell");
    let species = ConSpecies {
        symbols: &symbols,
        masses: &masses,
        cell: Some(&cell),
    };
    let first = [
        1.25, neg_zero, subnormal, 0.5, -2.0, 3.5, after_one, 4.0, 8.0,
    ];
    let second = [0.2, 0.3, 0.4, 1.1, 1.2, 1.3, -0.7, 0.8, 0.9];
    let forces = [0.01, -1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5];
    let energy = -12.25;
    let path = Tmp::new("roundtrip");
    let images = [
        ConImage {
            positions: &first,
            energy: Some(energy),
            forces: Some(&forces),
            neb_bead: None,
        },
        ConImage {
            positions: &second,
            energy: None,
            forces: None,
            neb_bead: None,
        },
    ];
    write_band(&path.0, &species, &images).unwrap();
    let again = Tmp::new("roundtrip-again");
    write_band(&again.0, &species, &images).unwrap();
    let first_bytes = std::fs::read(&path.0).unwrap();
    let second_bytes = std::fs::read(&again.0).unwrap();
    assert_eq!(first_bytes, second_bytes);
    assert_eq!(first_bytes.len(), 1008);

    let text = String::from_utf8(first_bytes.clone()).unwrap();
    let mut frames = text.split("rgsaddle\n").filter(|part| !part.is_empty());
    let first_frame = frames.next().expect("first frame");
    let second_frame = frames.next().expect("second frame");
    assert!(frames.next().is_none());
    // Cu and H are two components, and only the first image carries forces.
    assert_eq!(first_frame.matches("Forces of Component").count(), 2);
    assert_eq!(second_frame.matches("Forces of Component").count(), 0);
    assert!(first_frame.contains("\"neb_bead\":0"));
    assert!(second_frame.contains("\"neb_bead\":1"));

    let loaded = frames_from_con(&path.0).unwrap();
    assert_eq!(loaded.len(), 2);
    assert_eq!(bits(loaded[0].positions.as_slice().unwrap()), bits(&first));
    assert_eq!(bits(loaded[1].positions.as_slice().unwrap()), bits(&second));
    assert_eq!(bits(loaded[0].masses.as_slice().unwrap()), bits(&masses));
    assert_eq!(bits(loaded[1].masses.as_slice().unwrap()), bits(&masses));
    assert_eq!(
        cell_bits(loaded[0].cell.as_ref().unwrap()),
        cell_bits(&cell)
    );
    assert_eq!(
        cell_bits(loaded[1].cell.as_ref().unwrap()),
        cell_bits(&cell)
    );
    assert_eq!(loaded[0].positions[1].to_bits(), neg_zero.to_bits());
    assert_eq!(loaded[0].positions[2].to_bits(), 1);

    let head = frame_from_con(&path.0).unwrap();
    let nth = frame_from_con_nth(&path.0, 0).unwrap();
    let image_one = frame_from_con_nth(&path.0, 1).unwrap();
    assert_eq!(
        bits(head.positions.as_slice().unwrap()),
        bits(nth.positions.as_slice().unwrap())
    );
    assert_eq!(bits(image_one.positions.as_slice().unwrap()), bits(&second));
    assert!(frame_from_con_nth(&path.0, 2).is_err());

    let restart = Tmp::new("restart");
    write_restart(
        &restart.0,
        &species,
        &ConImage {
            positions: &first,
            energy: Some(energy),
            forces: None,
            neb_bead: None,
        },
    )
    .unwrap();
    let restart_text = std::fs::read_to_string(&restart.0).unwrap();
    assert!(!restart_text.contains("Forces of Component"));
    assert!(!restart_text.contains("neb_bead"));
    let restored = frame_from_con(&restart.0).unwrap();
    assert_eq!(bits(restored.positions.as_slice().unwrap()), bits(&first));
    assert_eq!(cell_bits(restored.cell.as_ref().unwrap()), cell_bits(&cell));
}

#[test]
fn write_restart_rejects_a_missing_symbol_vector() {
    let path = Tmp::new("symbols");
    let err = write_restart(
        &path.0,
        &ConSpecies {
            symbols: &[],
            masses: &[],
            cell: None,
        },
        &ConImage {
            positions: &[0.0, 0.0, 0.0],
            energy: None,
            forces: None,
            neb_bead: None,
        },
    )
    .unwrap_err();
    assert!(
        matches!(err, SaddleError::Invalid(ref message) if message.contains("symbol")),
        "{err}"
    );
}

#[test]
fn write_restart_rejects_two_masses_for_one_symbol() {
    let path = Tmp::new("masses");
    let symbols = ["Cu", "Cu"];
    let masses = [63.546, 65.0];
    let positions = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let err = write_restart(
        &path.0,
        &ConSpecies {
            symbols: &symbols,
            masses: &masses,
            cell: None,
        },
        &ConImage {
            positions: &positions,
            energy: None,
            forces: None,
            neb_bead: None,
        },
    )
    .unwrap_err();
    assert!(
        matches!(err, SaddleError::Invalid(ref message) if message.contains("mass")),
        "{err}"
    );
}

#[test]
fn write_restart_rejects_a_non_finite_coordinate() {
    let path = Tmp::new("nan");
    let symbols = ["H"];
    let masses = [1.008];
    let positions = [0.0, f64::NAN, 0.0];
    let err = write_restart(
        &path.0,
        &ConSpecies {
            symbols: &symbols,
            masses: &masses,
            cell: None,
        },
        &ConImage {
            positions: &positions,
            energy: None,
            forces: None,
            neb_bead: None,
        },
    )
    .unwrap_err();
    assert!(
        matches!(err, SaddleError::NonFinite("CON positions")),
        "{err}"
    );
}
