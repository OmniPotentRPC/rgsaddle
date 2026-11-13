//! Dipole field of a cubic crystal, and the hop barriers it produces.

use rgsaddle::{
    CubicElasticity, DirectCutoff, ElasticDipole, HopErrorSample, Kelvin, Strain, StrainResponse,
    Trust, cutoffs_by_hop,
};

fn shear_dipole() -> ElasticDipole {
    ElasticDipole::from_tensor([[0.0, 1.5, 0.2], [1.5, 0.0, -0.4], [0.2, -0.4, 0.3]]).unwrap()
}

#[test]
fn isotropic_cubic_strain_matches_kelvin() {
    let solid = CubicElasticity::isotropic(40.0, 0.30).unwrap();
    let kelvin = Kelvin::new(40.0, 0.30).unwrap();
    let dipole = shear_dipole();
    for r in [
        [3.0, 0.0, 0.0],
        [1.2, -0.7, 0.4],
        [0.2, 2.5, -1.1],
        [-0.8, -0.3, 1.7],
    ] {
        let from_cubic = solid.strain_of_dipole(r, dipole).unwrap().tensor();
        let from_kelvin = kelvin.strain_from_dipole(r, dipole).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let scale = from_kelvin[i][j].abs().max(1.0e-8);
                assert!(
                    (from_cubic[i][j] - from_kelvin[i][j]).abs() < 1.0e-8 * scale,
                    "r {r:?} component {i}{j}: cubic {} kelvin {}",
                    from_cubic[i][j],
                    from_kelvin[i][j]
                );
            }
        }
    }
}

#[test]
fn anisotropic_strain_scales_as_the_inverse_cube() {
    let solid = CubicElasticity::new(1.70, 1.22, 0.76).unwrap();
    let dipole = shear_dipole();
    let near = [2.0, 0.5, -0.25];
    let far = [4.0, 1.0, -0.5];
    let left = solid.strain_of_dipole(near, dipole).unwrap().tensor();
    let right = solid.strain_of_dipole(far, dipole).unwrap().tensor();
    for i in 0..3 {
        for j in 0..3 {
            let scale = left[i][j].abs().max(1.0e-12);
            assert!(
                (left[i][j] - 8.0 * right[i][j]).abs() < 1.0e-6 * scale,
                "component {i}{j}: {} versus 8 * {}",
                left[i][j],
                right[i][j]
            );
        }
    }
    let isotropic = CubicElasticity::isotropic(0.76, 1.22 / (2.0 * (1.22 + 0.76))).unwrap();
    let other = isotropic.strain_of_dipole(near, dipole).unwrap().tensor();
    let difference = (0..3)
        .flat_map(|i| (0..3).map(move |j| (left[i][j] - other[i][j]).abs()))
        .fold(0.0, f64::max);
    assert!(
        difference > 1.0e-4,
        "anisotropy changed nothing: {difference}"
    );
}

#[test]
fn periodic_strain_approaches_the_infinite_medium_field() {
    let solid = CubicElasticity::isotropic(25.0, 0.25).unwrap();
    let dipole = shear_dipole();
    let r = [2.4, -0.6, 0.3];
    let infinite = solid.strain_of_dipole(r, dipole).unwrap();
    let length = 80.0;
    let cell = [[length, 0.0, 0.0], [0.0, length, 0.0], [0.0, 0.0, length]];
    let periodic = solid.periodic_strain(r, cell, dipole, 3).unwrap();
    for i in 0..3 {
        for j in 0..3 {
            let scale = infinite.tensor()[i][j].abs().max(1.0e-8);
            assert!(
                (periodic.tensor()[i][j] - infinite.tensor()[i][j]).abs() < 0.05 * scale,
                "component {i}{j}: periodic {} infinite {}",
                periodic.tensor()[i][j],
                infinite.tensor()[i][j]
            );
        }
    }
}

#[test]
fn every_hop_uses_the_strain_of_the_source_dipole() {
    let minimum =
        ElasticDipole::from_tensor([[2.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]).unwrap();
    let saddle =
        ElasticDipole::from_tensor([[0.5, 0.4, 0.0], [0.4, 0.5, 0.0], [0.0, 0.0, -0.2]]).unwrap();
    let response = StrainResponse::new(0.47, minimum, saddle)
        .unwrap()
        .with_point_group(&[
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ])
        .unwrap();
    let solid = CubicElasticity::isotropic(30.0, 0.28).unwrap();
    let source = shear_dipole();
    let separation = [6.0, 1.0, -2.0];
    let hops = response
        .hops_from_dipole(&solid, separation, source)
        .unwrap();
    assert!(hops.len() >= 2, "images {}", hops.len());
    let strain = solid.strain_of_dipole(separation, source).unwrap();
    for hop in &hops {
        let direct = response.hops_at(&strain);
        let matched = direct.iter().find(|item| item.name == hop.name).unwrap();
        assert!((hop.barrier - matched.barrier).abs() < 1.0e-12);
        assert_eq!(hop.strain.tensor(), strain.tensor());
        let variant = response
            .variants()
            .iter()
            .find(|variant| variant.name == hop.name)
            .unwrap();
        let linear = variant.linear_barrier(0.47, strain);
        assert!((hop.barrier - linear).abs() < 1.0e-12);
    }
    let lowest = response.barrier(&strain);
    let best = hops
        .iter()
        .map(|hop| hop.barrier)
        .fold(f64::INFINITY, f64::min);
    assert!((lowest.barrier - best).abs() < 1.0e-12);
    assert!(matches!(
        hops[0].trust,
        Trust::Linear | Trust::NeedsSearch(_)
    ));
    let _ = Strain::zero();
}

#[test]
fn a_displaced_saddle_keeps_both_strains() {
    let minimum =
        ElasticDipole::from_tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).unwrap();
    let saddle =
        ElasticDipole::from_tensor([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]]).unwrap();
    let response = StrainResponse::new(0.1, minimum, saddle).unwrap();
    let at_minimum = Strain::uniaxial_x(0.01);
    let at_saddle =
        Strain::from_tensor([[0.0, 0.0, 0.0], [0.0, 0.02, 0.0], [0.0, 0.0, 0.0]]).unwrap();
    let hops = response.hops_at_ends(&at_minimum, &at_saddle);
    assert!((hops[0].barrier - 0.07).abs() < 1.0e-12);
    assert!((hops[0].strain.tensor()[0][0] - 0.01).abs() < 1.0e-15);
    assert!((hops[0].strain_saddle.tensor()[1][1] - 0.02).abs() < 1.0e-15);
}

#[test]
fn residual_stress_dipole_matches_the_fixed_cell_formula() {
    let defect = [
        [0.01, 0.002, 0.0],
        [0.002, -0.004, 0.001],
        [0.0, 0.001, 0.003],
    ];
    let bulk = [[0.001, 0.0, 0.0], [0.0, 0.001, 0.0], [0.0, 0.0, 0.001]];
    let volume = 100.0;
    let from_route = ElasticDipole::from_residual_stress(volume, defect, bulk).unwrap();
    let from_stress = ElasticDipole::from_stress(volume, defect, bulk).unwrap();
    assert_eq!(from_route.tensor(), from_stress.tensor());
    assert!((from_route.tensor()[0][0] + volume * 0.009).abs() < 1.0e-12);
}

#[test]
fn direct_cutoff_starts_at_the_first_clean_tail() {
    let samples = [(1.0, 0.02), (2.0, 0.004), (2.0, 0.006), (3.0, 0.001)];
    let cutoff = DirectCutoff::from_errors(&samples, 0.005).unwrap();
    assert!((cutoff.radius() - 3.0).abs() < 1.0e-12);
    assert!(cutoff.search_directly(2.0));
    assert!(cutoff.use_dipole(3.0));
    assert!(DirectCutoff::from_errors(&[(1.0, 0.02)], 0.005).is_err());
    let whole = DirectCutoff::from_errors(&[(4.0, 0.001), (5.0, 0.0)], 0.005).unwrap();
    assert!((whole.radius() - 4.0).abs() < 1.0e-12);
    assert!(whole.search_directly(3.9));
}

/// Largest |direct − dipole| at each shell of the 8×8×8 Mishin vacancy
/// survey, translation and rotation together. The tail inside 5 meV
/// starts at 10.5394 Å.
#[test]
fn mishin_direct_cutoff_is_the_10_5394_shell() {
    let samples = [
        (3.6150, 0.306954),
        (5.1124, 0.025721),
        (5.7158, 0.005997),
        (6.2614, 0.027840),
        (6.7630, 0.045336),
        (7.2300, 0.007803),
        (7.6686, 0.014160),
        (8.0834, 0.007306),
        (8.4779, 0.003204),
        (8.8549, 0.005578),
        (9.2165, 0.001644),
        (9.9001, 0.002802),
        (10.2248, 0.006408),
        (10.5394, 0.002762),
        (10.8450, 0.000631),
    ];
    let cutoff = DirectCutoff::from_errors(&samples, 0.005).unwrap();
    assert!((cutoff.radius() - 10.5394).abs() < 1.0e-9);
    assert!(cutoff.search_directly(10.2248));
    assert!(cutoff.use_dipole(10.5394));
    assert!(cutoff.use_dipole(10.8450));
}

/// Per-hop cutoffs from the 8×8×8 survey. A shell is safe only when every
/// later measured shell of that hop stays inside 5 meV. 7.2300 Å is one
/// rotation shell, not that tail: 7.6686 Å is 0.014160 eV out, 8.0834 Å is
/// 0.007306 eV out, and the [100] translation at 10.2248 Å is 0.005478 eV out.
#[test]
fn per_hop_cutoff_follows_the_measured_tail() {
    let mut samples = Vec::new();
    let mut rotations = Vec::new();
    for line in include_str!("../benchmarks/dipole_cutoff_cu.csv")
        .lines()
        .skip(1)
    {
        let mut cols = line.split(',');
        let kind = cols.next().unwrap();
        let orientation = cols.next().unwrap();
        let separation: f64 = cols.next().unwrap().parse().unwrap();
        let _sense = cols.next().unwrap();
        let _direct = cols.next().unwrap();
        let _dipole = cols.next().unwrap();
        let error = cols.next().unwrap();
        let status = cols.next().unwrap();
        if status != "converged" || error.is_empty() {
            continue;
        }
        let sample = HopErrorSample {
            kind: kind.to_string(),
            orientation: orientation.to_string(),
            separation,
            absolute_error: error.parse().unwrap(),
        };
        if kind == "rotation" {
            rotations.push(HopErrorSample {
                orientation: "all".to_string(),
                ..sample.clone()
            });
        }
        samples.push(sample);
    }
    samples.extend(rotations);
    let cuts = cutoffs_by_hop(&samples, 0.005).unwrap();
    let radius = |kind: &str, orientation: &str| {
        cuts.iter()
            .find(|cut| cut.kind == kind && cut.orientation == orientation)
            .and_then(|cut| cut.radius)
    };
    assert!((radius("translation", "100").unwrap() - 10.5394).abs() < 1.0e-6);
    assert!((radius("translation", "010").unwrap() - 7.2300).abs() < 1.0e-6);
    assert!((radius("translation", "001").unwrap() - 7.6686).abs() < 1.0e-6);
    assert!((radius("rotation", "all").unwrap() - 10.5394).abs() < 1.0e-6);
    let at_eight = samples.iter().any(|sample| {
        sample.kind == "rotation"
            && sample.orientation == "all"
            && (sample.separation - 8.0834).abs() < 1.0e-6
            && sample.absolute_error > 0.005
    });
    assert!(at_eight);
}
