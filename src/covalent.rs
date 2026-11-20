//! Covalent radii and element symbols for the automatic internals.
//!
//! The radii are the single-bond covalent radii of Cordero, Gomez,
//! Platero-Prats, Reves, Echeverria, Cremades, Barragan and Alvarez,
//! "Covalent radii revisited", Dalton Trans. (2008) 2832-2838,
//! doi:10.1039/B801115J, Table 2, for atomic numbers 1 to 96. Where the
//! table splits an element, this one takes the value Sella reads from
//! `ase.data.covalent_radii`: the sp3 radius for carbon (0.76 A) and the
//! low-spin radius for manganese, iron and cobalt (1.39, 1.32 and
//! 1.26 A). Cordero et al. give nothing beyond curium, so heavier
//! elements have no radius here and the topology refuses them rather
//! than guess.
//!
//! `scripts/check_covalent_radii.py` compares this table with an
//! independent transcription of Table 2 and with the ASE table.

/// Atomic number of the last element with a Cordero et al. radius.
pub const LAST_CORDERO_Z: u8 = 96;

/// Cordero et al. (2008) covalent radii in picometres, index `Z - 1`.
pub const CORDERO_2008_PM: [u16; 96] = [
    31, 28, // H He
    128, 96, 84, 76, 71, 66, 57, 58, // Li-Ne
    166, 141, 121, 111, 107, 105, 102, 106, // Na-Ar
    203, 176, 170, 160, 153, 139, 139, 132, 126, 124, 132, 122, 122, 120, 119, 120, 120,
    116, // K-Kr
    220, 195, 190, 175, 164, 154, 147, 146, 142, 139, 145, 144, 142, 139, 139, 138, 139,
    140, // Rb-Xe
    244, 215, // Cs Ba
    207, 204, 203, 201, 199, 198, 198, 196, 194, 192, 192, 189, 190, 187, // La-Yb
    187, 175, 170, 162, 151, 144, 141, 136, 136, 132, 145, 146, 148, 140, 150, 150, // Lu-Rn
    260, 221, // Fr Ra
    215, 206, 200, 196, 190, 187, 180, 169, // Ac-Cm
];

const SYMBOLS: [&str; 118] = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl",
    "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As",
    "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In",
    "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
    "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl",
    "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk",
    "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh",
    "Fl", "Mc", "Lv", "Ts", "Og",
];

/// Covalent radius of element `z` in angstrom, or `None` outside 1..=96.
pub fn covalent_radius(z: u8) -> Option<f64> {
    if z == 0 || z > LAST_CORDERO_Z {
        return None;
    }
    Some(f64::from(CORDERO_2008_PM[usize::from(z) - 1]) / 100.0)
}

/// Element symbol of atomic number `z` (1..=118).
pub fn element_symbol(z: u8) -> Option<&'static str> {
    if z == 0 {
        return None;
    }
    SYMBOLS.get(usize::from(z) - 1).copied()
}

/// Atomic number of an element symbol, matched case-insensitively on
/// the standard capitalization (`"cl"`, `"Cl"` and `"CL"` are chlorine).
pub fn atomic_number(symbol: &str) -> Option<u8> {
    let s = symbol.trim();
    SYMBOLS
        .iter()
        .position(|c| c.eq_ignore_ascii_case(s))
        .map(|i| (i + 1) as u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_has_one_entry_per_element() {
        assert_eq!(CORDERO_2008_PM.len(), usize::from(LAST_CORDERO_Z));
        assert_eq!(SYMBOLS.len(), 118);
    }

    #[test]
    fn spot_values_match_cordero_table_two() {
        let pm = |z: u8| (covalent_radius(z).unwrap() * 100.0).round() as u16;
        let spots = [
            (1u8, 31u16),
            (6, 76),
            (7, 71),
            (8, 66),
            (9, 57),
            (16, 105),
            (17, 102),
            (26, 132),
            (29, 132),
            (79, 136),
            (96, 169),
        ];
        for (z, want) in spots {
            assert_eq!(pm(z), want, "Z={z}");
        }
    }

    #[test]
    fn symbols_round_trip_and_bounds() {
        for z in 1..=118u8 {
            let s = element_symbol(z).unwrap();
            assert_eq!(atomic_number(s), Some(z), "{s}");
        }
        assert_eq!(atomic_number("cl"), Some(17));
        assert_eq!(atomic_number("Xx"), None);
        assert!(covalent_radius(0).is_none());
        assert!(covalent_radius(97).is_none());
        assert!(element_symbol(0).is_none());
    }
}
