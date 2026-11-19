use ndarray::{Array1, Array2, array};
use rgsaddle::{CrystalSymmetryConfig, CrystalSymmetryOperation, Region, SymmetryUniqueRegion};

fn operation(rotation: Array2<f64>) -> CrystalSymmetryOperation {
    CrystalSymmetryOperation::default().with_rotation(rotation)
}

fn c4v() -> Vec<CrystalSymmetryOperation> {
    let mut operations = Vec::new();
    for rotation in [
        array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        array![[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        array![[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        array![[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    ] {
        operations.push(operation(rotation.clone()));
        let reflection = array![[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]];
        operations.push(operation(rotation.dot(&reflection)));
    }
    operations
}

fn cubic_group(centre: Array1<f64>) -> Vec<CrystalSymmetryOperation> {
    let mut operations = Vec::new();
    for permutation in [
        [0_usize, 1_usize, 2_usize],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        for signs in 0_u8..8 {
            let mut rotation = Array2::zeros((3, 3));
            for row in 0..3 {
                rotation[[row, permutation[row]]] =
                    if signs & (1 << row) == 0 { 1.0 } else { -1.0 };
            }
            operations.push(
                CrystalSymmetryOperation::default()
                    .with_rotation(rotation)
                    .with_centre(centre.clone()),
            );
        }
    }
    operations
}

#[test]
fn analytic_c4v_splits_four_paths_from_the_in_plane_mirror() {
    let reactant = array![[0.0, 0.0, 0.0]];
    let saddle = array![[1.0, 0.0, 0.0]];
    let config = CrystalSymmetryConfig::default();
    let operations = c4v();
    let built = SymmetryUniqueRegion::from_crystal_operations(
        reactant.view(),
        saddle.view(),
        &[0],
        &operations,
        &config,
    )
    .unwrap();
    let joint = joint_stabilizer_order(&operations, &reactant, &saddle, &config);
    assert_eq!(built.path_multiplicity, 4);
    assert_eq!(
        built.path_multiplicity,
        distinct_saddle_images(&operations, &reactant, &saddle, &config)
    );
    assert_eq!(built.region_degeneracy, joint);
    assert_eq!(built.region.degeneracy(), joint);
    assert!(joint > 1);
    assert_ne!(built.region_degeneracy, built.path_multiplicity);
    assert!(built.region.contains(flatten(&saddle).view()).unwrap());
    assert_exactly_one_side(&built.region, &saddle, 1, 0.4);
}

#[test]
fn fcc_vacancy_nearest_neighbour_hop_has_twelve_paths() {
    let centre = array![5.0, 5.0, 5.0];
    let mut reactant = Array2::zeros((12, 3));
    let mut row = 0;
    for zero_axis in 0..3 {
        for first_sign in [-1.0, 1.0] {
            for second_sign in [-1.0, 1.0] {
                let mut offset = [first_sign, second_sign, 0.0];
                offset.rotate_right(zero_axis);
                for axis in 0..3 {
                    reactant[[row, axis]] = centre[axis] + offset[axis];
                }
                row += 1;
            }
        }
    }
    let mut saddle = reactant.clone();
    for axis in 0..3 {
        saddle[[0, axis]] = 0.5 * (reactant[[0, axis]] + centre[axis]);
    }
    let config = CrystalSymmetryConfig::default()
        .with_cell(minimage::Cell::ortho(10.0, 10.0, 10.0).unwrap());
    let operations = cubic_group(centre);
    let built = SymmetryUniqueRegion::from_crystal_operations(
        reactant.view(),
        saddle.view(),
        &[0; 12],
        &operations,
        &config,
    )
    .unwrap();
    let joint = joint_stabilizer_order(&operations, &reactant, &saddle, &config);
    assert_eq!(built.path_multiplicity, 12);
    assert_eq!(
        built.path_multiplicity,
        distinct_saddle_images(&operations, &reactant, &saddle, &config)
    );
    assert_eq!(built.region_degeneracy, joint);
    assert_eq!(built.region.degeneracy(), joint);
    assert_ne!(joint, built.path_multiplicity);
    assert!(built.region.contains(flatten(&saddle).view()).unwrap());
    assert!(
        built
            .region
            .contains(built.region.representative())
            .unwrap()
    );
    assert_one_joint_mirror_cuts_the_plane(&built.region, &operations, &reactant, &saddle, &config);
}

#[test]
fn a_lattice_translation_is_not_the_identity() {
    let reactant = array![[0.0, 0.0, 0.0]];
    let saddle = array![[1.0, 0.0, 0.0]];
    let translated = operation(Array2::eye(3)).with_translation(array![4.0, 0.0, 0.0]);
    let error = SymmetryUniqueRegion::from_crystal_operations(
        reactant.view(),
        saddle.view(),
        &[0],
        &[translated],
        &CrystalSymmetryConfig::default(),
    )
    .expect_err("a lattice translation is not the identity");
    assert!(
        matches!(error, rgsaddle::SaddleError::Invalid(_)),
        "{error:?}"
    );
    let guide = include_str!("../docs/orgmode/explanation/symmetry-regions.org");
    assert!(
        guide.to_ascii_lowercase().contains("lattice translation"),
        "the symmetry reference must say a lattice translation is not the identity"
    );
    let source = include_str!("../src/region.rs");
    assert!(
        source.to_ascii_lowercase().contains("lattice translation"),
        "rustdoc must say a lattice translation is not the identity"
    );
}

#[test]
fn identity_group_gives_the_trivial_region() {
    let geometry = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let identity = operation(Array2::eye(3)).with_permutation(vec![0, 1]);
    let built = SymmetryUniqueRegion::from_crystal_operations(
        geometry.view(),
        geometry.view(),
        &[0, 0],
        &[identity],
        &CrystalSymmetryConfig::default(),
    )
    .unwrap();
    assert_eq!(built.path_multiplicity, 1);
    assert_eq!(built.region_degeneracy, 1);
    assert_eq!(built.region.degeneracy(), 1);
    assert!(built.region.planar_boundaries().unwrap().is_empty());
    assert!(
        built
            .region
            .contains(array![9.0, -2.0, 4.0, 3.0, 7.0, 1.0].view())
            .unwrap()
    );
}

#[test]
fn invalid_operations_are_rejected() {
    let reactant = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let non_orthogonal = operation(array![[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    assert!(
        SymmetryUniqueRegion::from_crystal_operations(
            reactant.view(),
            reactant.view(),
            &[0, 1],
            &[non_orthogonal],
            &CrystalSymmetryConfig::default(),
        )
        .is_err()
    );

    let cross_species = operation(Array2::eye(3)).with_permutation(vec![1, 0]);
    assert!(
        SymmetryUniqueRegion::from_crystal_operations(
            reactant.view(),
            reactant.view(),
            &[0, 1],
            &[cross_species],
            &CrystalSymmetryConfig::default(),
        )
        .is_err()
    );
}

/// Al(100) adatom hop, following the half-line construction of Jóhannesson
/// and Jónsson, J. Chem. Phys. 115, 9644 (2001), Secs. II D and III D.
///
/// The adatom sits in a fourfold hollow. Four bridge directions are the
/// equivalent products. One hop, along +x, still has the vertical mirror that
/// contains the hop, so the dividing plane's asymmetric unit is a proper
/// subset of that plane and the region degeneracy is greater than one.
#[test]
fn al100_hop_half_line_has_in_plane_region_degeneracy() {
    let reactant = array![
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, -1.0, 0.0],
        [-1.0, 1.0, 0.0],
        [-1.0, -1.0, 0.0],
    ];
    let mut saddle = reactant.clone();
    saddle[[0, 0]] = 1.0;
    let config = CrystalSymmetryConfig::default();
    let operations = c4v();
    let built = SymmetryUniqueRegion::from_crystal_operations(
        reactant.view(),
        saddle.view(),
        &[0; 5],
        &operations,
        &config,
    )
    .unwrap();
    let joint = joint_stabilizer_order(&operations, &reactant, &saddle, &config);
    let paths = distinct_saddle_images(&operations, &reactant, &saddle, &config);
    assert_eq!(built.path_multiplicity, paths);
    assert_eq!(paths, 4);
    assert!(joint > 1);
    assert_eq!(built.region_degeneracy, joint);
    assert_eq!(built.region.degeneracy(), joint);
    assert_ne!(built.region_degeneracy, built.path_multiplicity);
    assert!(built.region.contains(flatten(&saddle).view()).unwrap());
    assert_exactly_one_side(&built.region, &saddle, 1, 0.25);
    let hop = flatten(&saddle) - flatten(&reactant);
    for boundary in built.region.planar_boundaries().unwrap() {
        let normal = boundary.normal();
        let scale = hop.dot(&hop).sqrt() * normal.dot(&normal).sqrt();
        assert!(hop.dot(&normal).abs() <= config.orthogonality_tolerance * scale.max(1.0));
    }
}

fn flatten(geometry: &Array2<f64>) -> Array1<f64> {
    Array1::from_iter(geometry.iter().copied())
}

fn assert_exactly_one_side(
    region: &SymmetryUniqueRegion,
    saddle: &Array2<f64>,
    axis: usize,
    shift: f64,
) {
    let mut positive = saddle.clone();
    let mut negative = saddle.clone();
    positive[[0, axis]] += shift;
    negative[[0, axis]] -= shift;
    let positive_inside = region.contains(flatten(&positive).view()).unwrap();
    let negative_inside = region.contains(flatten(&negative).view()).unwrap();
    assert!(
        positive_inside ^ negative_inside,
        "the in-plane asymmetric unit keeps one side of the mirror"
    );
}

fn assert_one_joint_mirror_cuts_the_plane(
    region: &SymmetryUniqueRegion,
    operations: &[CrystalSymmetryOperation],
    reactant: &Array2<f64>,
    saddle: &Array2<f64>,
    config: &CrystalSymmetryConfig,
) {
    let mirror = operations.iter().find(|operation| {
        !is_identity(operation)
            && fixes_geometry(operation, reactant, config, config.matching_tolerance)
            && fixes_geometry(operation, saddle, config, config.orbit_tolerance)
    });
    let Some(mirror) = mirror else {
        panic!("the joint stabilizer should contain a non-identity operation");
    };
    for axis in 0..3 {
        let mut displaced = saddle.clone();
        displaced[[0, axis]] += 0.3;
        let image = map_geometry(mirror, &displaced);
        if same_point_set(&displaced, &image, config, config.orbit_tolerance) {
            continue;
        }
        let displaced_inside = region.contains(flatten(&displaced).view()).unwrap();
        let image_inside = region.contains(flatten(&image).view()).unwrap();
        assert!(
            displaced_inside ^ image_inside,
            "a joint-stabilizer mirror should cut the plane into one retained side"
        );
        return;
    }
    panic!("no coordinate displacement was moved by the joint stabilizer");
}

fn joint_stabilizer_order(
    operations: &[CrystalSymmetryOperation],
    reactant: &Array2<f64>,
    saddle: &Array2<f64>,
    config: &CrystalSymmetryConfig,
) -> usize {
    operations
        .iter()
        .filter(|operation| {
            fixes_geometry(operation, reactant, config, config.matching_tolerance)
                && fixes_geometry(operation, saddle, config, config.orbit_tolerance)
        })
        .count()
}

fn distinct_saddle_images(
    operations: &[CrystalSymmetryOperation],
    reactant: &Array2<f64>,
    saddle: &Array2<f64>,
    config: &CrystalSymmetryConfig,
) -> usize {
    let mut images = vec![saddle.clone()];
    for operation in operations {
        if !fixes_geometry(operation, reactant, config, config.matching_tolerance) {
            continue;
        }
        let image = map_geometry(operation, saddle);
        let mut known = false;
        for previous in &images {
            if same_point_set(previous, &image, config, config.orbit_tolerance) {
                known = true;
                break;
            }
        }
        if !known {
            images.push(image);
        }
    }
    images.len()
}

fn is_identity(operation: &CrystalSymmetryOperation) -> bool {
    operation.translation.is_none()
        && operation.permutation.is_none()
        && operation
            .rotation
            .indexed_iter()
            .all(|((row, column), value)| *value == if row == column { 1.0 } else { 0.0 })
}

fn apply_operation(operation: &CrystalSymmetryOperation, point: [f64; 3]) -> [f64; 3] {
    let centre = match &operation.centre {
        Some(centre) => [centre[0], centre[1], centre[2]],
        None => [0.0, 0.0, 0.0],
    };
    let shifted = [
        point[0] - centre[0],
        point[1] - centre[1],
        point[2] - centre[2],
    ];
    let rotation = &operation.rotation;
    let mut mapped = [0.0; 3];
    for row in 0..3 {
        mapped[row] = rotation[[row, 0]] * shifted[0]
            + rotation[[row, 1]] * shifted[1]
            + rotation[[row, 2]] * shifted[2]
            + centre[row];
    }
    if let Some(translation) = &operation.translation {
        for row in 0..3 {
            mapped[row] += translation[row];
        }
    }
    mapped
}

fn map_geometry(operation: &CrystalSymmetryOperation, geometry: &Array2<f64>) -> Array2<f64> {
    let mut image = Array2::zeros(geometry.raw_dim());
    for atom in 0..geometry.nrows() {
        let mapped = apply_operation(
            operation,
            [
                geometry[[atom, 0]],
                geometry[[atom, 1]],
                geometry[[atom, 2]],
            ],
        );
        for axis in 0..3 {
            image[[atom, axis]] = mapped[axis];
        }
    }
    image
}

fn separation(config: &CrystalSymmetryConfig, from: [f64; 3], to: [f64; 3]) -> f64 {
    let delta = match &config.cell {
        Some(cell) => cell.displacement_euclidean(from, to),
        None => [to[0] - from[0], to[1] - from[1], to[2] - from[2]],
    };
    delta
        .into_iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt()
}

fn fixes_geometry(
    operation: &CrystalSymmetryOperation,
    geometry: &Array2<f64>,
    config: &CrystalSymmetryConfig,
    tolerance: f64,
) -> bool {
    same_point_set(
        &map_geometry(operation, geometry),
        geometry,
        config,
        tolerance,
    )
}

fn same_point_set(
    left: &Array2<f64>,
    right: &Array2<f64>,
    config: &CrystalSymmetryConfig,
    tolerance: f64,
) -> bool {
    if left.nrows() != right.nrows() {
        return false;
    }
    let mut used = vec![false; right.nrows()];
    for source in 0..left.nrows() {
        let from = [left[[source, 0]], left[[source, 1]], left[[source, 2]]];
        let mut target = None;
        for candidate in 0..right.nrows() {
            if used[candidate] {
                continue;
            }
            let to = [
                right[[candidate, 0]],
                right[[candidate, 1]],
                right[[candidate, 2]],
            ];
            if separation(config, from, to) <= tolerance {
                if target.is_some() {
                    return false;
                }
                target = Some(candidate);
            }
        }
        let Some(target) = target else {
            return false;
        };
        used[target] = true;
    }
    true
}

#[test]
fn a_group_without_the_identity_is_rejected() {
    let reactant = array![[0.0, 0.0, 0.0]];
    let saddle = array![[1.0, 0.0, 0.0]];
    let mirror = operation(array![[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0],]);
    let error = SymmetryUniqueRegion::from_crystal_operations(
        reactant.view(),
        saddle.view(),
        &[0],
        &[mirror],
        &CrystalSymmetryConfig::default(),
    )
    .expect_err("a crystal group without the identity");
    assert!(error.to_string().contains("identity"), "{error}");
}
