//! Counts from Sella 2.6 on fixed geometries, recorded once by
//! `scripts/gen_sella_topology_fixture.py`. Sella is an oracle for the
//! test, not a dependency.

use ndarray::Array1;
use rgsaddle::{Equality, InternalTopology};

struct Entry {
    name: String,
    numbers: Vec<u8>,
    x: Array1<f64>,
    bonds: Vec<[usize; 2]>,
    counts: [usize; 4],
}

fn load() -> (String, Vec<Entry>) {
    let text = include_str!("fixtures/sella_topology.txt");
    let mut version = String::new();
    let mut out: Vec<Entry> = Vec::new();
    for line in text.lines() {
        let (key, rest) = line.split_once(' ').unwrap_or((line, ""));
        match key {
            "sella" => version = rest.to_string(),
            "molecule" => out.push(Entry {
                name: rest.to_string(),
                numbers: Vec::new(),
                x: Array1::zeros(0),
                bonds: Vec::new(),
                counts: [0; 4],
            }),
            "numbers" => {
                out.last_mut().unwrap().numbers = rest
                    .split_whitespace()
                    .map(|v| v.parse().unwrap())
                    .collect();
            }
            "positions" => {
                out.last_mut().unwrap().x = rest
                    .split_whitespace()
                    .map(|v| v.parse().unwrap())
                    .collect();
            }
            "bonds" => {
                out.last_mut().unwrap().bonds = rest
                    .split_whitespace()
                    .map(|p| {
                        let (a, b) = p.split_once('-').unwrap();
                        [a.parse().unwrap(), b.parse().unwrap()]
                    })
                    .collect();
            }
            "counts" => {
                let v: Vec<usize> = rest
                    .split_whitespace()
                    .map(|v| v.parse().unwrap())
                    .collect();
                out.last_mut().unwrap().counts = [v[0], v[1], v[2], v[3]];
            }
            _ => {}
        }
    }
    (version, out)
}

/// Molecules on which this crate and Sella 2.6 build the same chart:
/// the same bonds, and as many angles, dihedrals and dummy atoms.
const AGREE: [&str; 13] = [
    "H2O", "H2O2", "C2H6", "C2H4", "CH3OH", "CH3SH", "CO2", "HCN", "C2H2", "CH3CH2OH", "C6H6",
    "HCOOH", "CH3CHO",
];

/// Sella adds improper dihedrals at a centre with three or more bonds
/// even when the bonds and angles already fix it; this crate adds one
/// only where it is needed.
const SELLA_EXTRA_IMPROPERS: [&str; 2] = ["NH3", "CH4"];

fn real_bonds(t: &InternalTopology, n: usize) -> Vec<[usize; 2]> {
    let mut out: Vec<[usize; 2]> = t
        .constraints()
        .equalities()
        .iter()
        .filter_map(|e| match e {
            Equality::Bond { atoms, .. } if atoms[0] < n && atoms[1] < n => {
                Some([atoms[0].min(atoms[1]), atoms[0].max(atoms[1])])
            }
            _ => None,
        })
        .collect();
    out.sort_unstable();
    out
}

#[test]
fn the_fixture_names_the_sella_release_it_was_recorded_with() {
    let (version, list) = load();
    assert_eq!(version, "2.6.0");
    assert_eq!(list.len(), AGREE.len() + SELLA_EXTRA_IMPROPERS.len());
}

#[test]
fn bonds_match_sella_on_every_molecule() {
    let (_, list) = load();
    for e in &list {
        let n = e.numbers.len();
        let t = InternalTopology::from_geometry(&e.numbers, e.x.view(), None).unwrap();
        let mut theirs: Vec<[usize; 2]> = e
            .bonds
            .iter()
            .filter(|b| b[0] < n && b[1] < n)
            .copied()
            .collect();
        theirs.sort_unstable();
        assert_eq!(real_bonds(&t, n), theirs, "{}", e.name);
    }
}

#[test]
fn counts_match_sella_where_the_two_build_the_same_chart() {
    let (_, list) = load();
    for e in list.iter().filter(|e| AGREE.contains(&e.name.as_str())) {
        let t = InternalTopology::from_geometry(&e.numbers, e.x.view(), None).unwrap();
        let c = t.constraints().counts();
        assert_eq!(
            [c.nbonds, c.nangles, c.ndihedrals, t.n_dummy()],
            e.counts,
            "{}",
            e.name
        );
        assert!(t.check().complete, "{}", e.name);
    }
}

#[test]
fn sella_only_adds_impropers_where_this_crate_stops_at_bonds_and_angles() {
    let (_, list) = load();
    for e in list
        .iter()
        .filter(|e| SELLA_EXTRA_IMPROPERS.contains(&e.name.as_str()))
    {
        let t = InternalTopology::from_geometry(&e.numbers, e.x.view(), None).unwrap();
        let c = t.constraints().counts();
        assert_eq!(
            [c.nbonds, c.nangles, t.n_dummy()],
            [e.counts[0], e.counts[1], e.counts[3]]
        );
        assert_eq!(c.ndihedrals, 0, "{}", e.name);
        assert!(e.counts[2] > 0, "{}", e.name);
        assert!(t.check().complete, "{}", e.name);
    }
}
