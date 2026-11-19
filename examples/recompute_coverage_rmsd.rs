//! Recompute the coverage RMSD overlay with the canonical `minimage` crate.

use minimage::Cell;
use serde_json::{Value, json};

const RMSD_MATCH_TOLERANCE_ANGSTROM: f64 = 1.0e-12;

fn triples(value: &Value) -> Vec<[f64; 3]> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            let row = row.as_array().unwrap();
            [
                row[0].as_f64().unwrap(),
                row[1].as_f64().unwrap(),
                row[2].as_f64().unwrap(),
            ]
        })
        .collect()
}

fn rmsd(cell: &Cell, reference: &[[f64; 3]], trial: &[[f64; 3]]) -> f64 {
    let sum: f64 = reference
        .iter()
        .zip(trial)
        .map(|(&left, &right)| cell.dist2_euclidean(left, right))
        .sum();
    (sum / reference.len() as f64).sqrt()
}

fn main() {
    let payload: Value = serde_json::from_str(include_str!("../bench/results.json")).unwrap();
    let system = &payload["systems"]["pt_exchange"];
    let rows = triples(&system["cell"]);
    let cell = Cell::from_ase([rows[0], rows[1], rows[2]]).unwrap();
    let fixed = system["fixed"].as_array().unwrap();
    let reference: Vec<_> = triples(&system["reference"]["positions"])
        .into_iter()
        .enumerate()
        .filter_map(|(index, point)| (fixed[index].as_u64() == Some(0)).then_some(point))
        .collect();
    let cause = "The rejected independent check included fixed slab atoms and used a custom radius-2 lattice-image search; the benchmark uses free atoms only and minimage 0.1.2's Smith/Minkowski-reduced 27-image Euclidean convention. The legacy harness passes raw stored positions directly to minimage::Cell::displacement_euclidean through the C ABI, performs no ASE get_distances call, explicit wrapping, centering, or alignment, and computes sqrt(mean(per-atom squared 3D displacement)) over free atoms. For ase-neb it compares the reference with the highest-energy interior image selected from the final unconverged band. All rounds are identical because ASE BFGS starts from the same linear band without a seed and deterministically reaches the same final selected structure; each run stores that structure independently.";
    let mut corrections = Vec::new();
    for run in payload["runs"].as_array().unwrap() {
        if run["system"] == "pt_exchange" && run["method"] == "ase-neb" && run["converged"] == false
        {
            let trial: Vec<_> = triples(&run["positions"])
                .into_iter()
                .enumerate()
                .filter_map(|(index, point)| (fixed[index].as_u64() == Some(0)).then_some(point))
                .collect();
            let previous = run["rmsd"].as_f64().unwrap();
            let recomputed = rmsd(&cell, &reference, &trial);
            assert!((previous - recomputed).abs() < RMSD_MATCH_TOLERANCE_ANGSTROM);
            corrections.push(json!({
                "cause": cause,
                "images": run["images"].as_u64().unwrap(),
                "method": "ase-neb",
                "rmsd_new": recomputed,
                "rmsd_old": previous,
                "round": run["round"].as_u64().unwrap(),
                "system": "pt_exchange",
            }));
        }
    }
    corrections.sort_by_key(|row| {
        (
            row["images"].as_u64().unwrap(),
            row["round"].as_u64().unwrap(),
        )
    });
    assert_eq!(corrections.len(), 24);
    let output = json!({
        "generator": "examples/recompute_coverage_rmsd.rs",
        "legacy_code": {
            "base": "bench/harness.py rmsd -> mic_vectors -> mi_displacement_euclidean",
            "cell_revision": "7c1d432 uses minimage 0.1.2; src/mic.rs uses Cell fractional wrapping for rgsaddle band internals",
        },
        "rows": corrections,
        "source_results": "bench/results.json",
    });
    println!("{}", serde_json::to_string_pretty(&output).unwrap());
}
