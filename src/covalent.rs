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

/// Standard atomic weights in unified atomic mass units (`u` / `Da`),
/// indexed by `Z - 1` for elements 1 through 118, matching ASE's
/// `ase.data.atomic_masses`.
pub const ATOMIC_MASSES_AMU: [f64; 118] = [
    1.008,
    4.002602,
    6.94,
    9.0121831,
    10.81,
    12.011,
    14.007,
    15.999,
    18.998403163,
    20.1797,
    22.98976928,
    24.305,
    26.9815385,
    28.085,
    30.973761998,
    32.06,
    35.45,
    39.948,
    39.0983,
    40.078,
    44.955908,
    47.867,
    50.9415,
    51.9961,
    54.938044,
    55.845,
    58.933194,
    58.6934,
    63.546,
    65.38,
    69.723,
    72.63,
    74.921595,
    78.971,
    79.904,
    83.798,
    85.4678,
    87.62,
    88.90584,
    91.224,
    92.90637,
    95.95,
    97.90721,
    101.07,
    102.9055,
    106.42,
    107.8682,
    112.414,
    114.818,
    118.71,
    121.76,
    127.6,
    126.90447,
    131.293,
    132.90545196,
    137.327,
    138.90547,
    140.116,
    140.90766,
    144.242,
    144.91276,
    150.36,
    151.964,
    157.25,
    158.92535,
    162.5,
    164.93033,
    167.259,
    168.93422,
    173.054,
    174.9668,
    178.49,
    180.94788,
    183.84,
    186.207,
    190.23,
    192.217,
    195.084,
    196.966569,
    200.592,
    204.38,
    207.2,
    208.9804,
    208.98243,
    209.98715,
    222.01758,
    223.01974,
    226.02541,
    227.02775,
    232.0377,
    231.03588,
    238.02891,
    237.04817,
    244.06421,
    243.06138,
    247.07035,
    247.07031,
    251.07959,
    252.083,
    257.09511,
    258.09843,
    259.101,
    262.11,
    267.122,
    268.126,
    271.134,
    270.133,
    269.1338,
    278.156,
    281.165,
    281.166,
    285.177,
    286.182,
    289.19,
    289.194,
    293.204,
    293.208,
    294.214,
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

/// Standard atomic mass of element `z` in u (`Da`), or `None` outside `1..=118`.
pub fn atomic_mass(z: u8) -> Option<f64> {
    if z == 0 {
        return None;
    }
    ATOMIC_MASSES_AMU.get(usize::from(z) - 1).copied()
}

/// Standard atomic mass in u (`Da`) for an element symbol, matched case-insensitively.
pub fn atomic_mass_by_symbol(symbol: &str) -> Option<f64> {
    atomic_number(symbol).and_then(atomic_mass)
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

    #[test]
    fn masses_match_reference_spots_and_reject_out_of_range() {
        assert_eq!(ATOMIC_MASSES_AMU.len(), 118);
        assert!((atomic_mass(1).unwrap() - 1.008).abs() < 1e-12);
        assert!((atomic_mass(13).unwrap() - 26.9815385).abs() < 1e-12);
        assert!((atomic_mass(29).unwrap() - 63.546).abs() < 1e-12);
        assert!((atomic_mass_by_symbol("Cu").unwrap() - 63.546).abs() < 1e-12);
        assert!((atomic_mass_by_symbol("cu").unwrap() - 63.546).abs() < 1e-12);
        assert!((atomic_mass_by_symbol("Pt").unwrap() - 195.084).abs() < 1e-12);
        assert!(atomic_mass(0).is_none());
        assert!(atomic_mass(119).is_none());
        assert!(atomic_mass_by_symbol("Xx").is_none());
    }
}
