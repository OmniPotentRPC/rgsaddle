//! Downward pathway maps on two surfaces whose critical points are known.
//!
//! The separable well is the surface in `tests/saddle_index.rs`.
//! Mueller-Brown is the surface in `tests/simplified_string.rs`, with the
//! same published stationary points.

use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, ArrayView1, array};
use rgsaddle::{
    Cell, CriticalPoint, ForceGate, IndexKConfig, IndexKSession, LandscapeBoundKind,
    LandscapeConfig, LandscapeSession, MinModeConfig, MinModeStatus, NEGATIVE_CURVATURE_TOL,
    PathwayMap, PointSurface, RelaxConfig, SaddleError, SaddleIndexCheck, saddle_index,
};

const SIDE: f64 = std::f64::consts::FRAC_1_SQRT_2;
const POSITION_TOL: f64 = 1.0e-3;

struct Wells;

fn wells_energy(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let (a, b, c) = (x[0], x[1], x[2]);
    let energy = (a.powi(4) - 2.0 * a * a) / 4.0 + (b.powi(4) - b * b) / 4.0 + c * c / 2.0;
    (energy, array![a.powi(3) - a, b.powi(3) - 0.5 * b, c])
}

impl PointSurface for Wells {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        Ok(wells_energy(x))
    }

    fn hessian_vector(
        &self,
        x: ArrayView1<f64>,
        v: ArrayView1<f64>,
    ) -> Result<Option<Array1<f64>>, SaddleError> {
        Ok(Some(array![
            (3.0 * x[0] * x[0] - 1.0) * v[0],
            (3.0 * x[1] * x[1] - 0.5) * v[1],
            v[2],
        ]))
    }
}

struct Counted<S> {
    inner: S,
    calls: AtomicUsize,
}

impl<S: PointSurface> PointSurface for Counted<S> {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.inner.eval(x)
    }

    fn hessian_vector(
        &self,
        x: ArrayView1<f64>,
        v: ArrayView1<f64>,
    ) -> Result<Option<Array1<f64>>, SaddleError> {
        self.inner.hessian_vector(x, v)
    }
}

fn wells_config() -> LandscapeConfig {
    let search = MinModeConfig::default()
        .with_force_tol(1.0e-8)
        .with_max_move(0.1)
        .with_saddle_check(None);
    LandscapeConfig::default()
        .with_push(0.5)
        .with_duplicate_distance(1.0e-2)
        .with_max_steps(800)
        .with_search(IndexKConfig::new(1, search))
        .with_relax(
            RelaxConfig::default()
                .with_force_tol(1.0e-8)
                .with_max_atom_move(0.1)
                .with_initial_step(0.05),
        )
}

fn near(point: &CriticalPoint, target: [f64; 3]) -> bool {
    point
        .position
        .iter()
        .zip(target)
        .all(|(found, expected)| (found - expected).abs() <= POSITION_TOL)
}

fn require(nodes: &[CriticalPoint], target: [f64; 3]) -> &CriticalPoint {
    nodes
        .iter()
        .find(|node| near(node, target))
        .unwrap_or_else(|| panic!("missing {target:?} in {nodes:?}"))
}

#[test]
fn the_wells_map_from_the_origin_has_every_saddle_and_minimum() {
    let surface = Counted {
        inner: Wells,
        calls: AtomicUsize::new(0),
    };
    let session = LandscapeSession::new(wells_config(), array![0.0, 0.0, 0.0], None).unwrap();
    let map = session.run(&surface).unwrap();
    assert_eq!(map.evaluations, surface.calls.load(Ordering::Relaxed));
    check_wells(&map);
    eprintln!(
        "WELLS nodes={} edges={} evaluations={}",
        map.nodes.len(),
        map.edges.len(),
        map.evaluations
    );
    for (i, node) in map.nodes.iter().enumerate() {
        eprintln!(
            "WELLS_NODE {i} index={} energy={:.8} evaluations={} position={:?}",
            node.index,
            node.energy,
            node.evaluations,
            node.position.as_slice()
        );
    }
}

#[test]
fn an_indexk_block_is_accepted_as_the_parent_modes() {
    let search = MinModeConfig::default()
        .with_force_tol(1.0e-8)
        .with_max_move(0.1)
        .with_saddle_check(None);
    let mut indexk = IndexKSession::new(
        IndexKConfig::new(2, search).with_residual_tol(1.0e-8),
        array![0.25, -0.2, 0.15],
        None,
    )
    .unwrap();
    let report = indexk.run(&Wells, 400).unwrap();
    assert_eq!(report.status, MinModeStatus::Converged);
    assert!(report.position.iter().all(|value| value.abs() < 1.0e-4));
    let session = LandscapeSession::new(
        wells_config(),
        report.position,
        Some(indexk.modes().to_owned()),
    )
    .unwrap();
    let map = session.run(&Wells).unwrap();
    check_wells(&map);
}

fn check_wells(map: &PathwayMap) {
    assert_eq!(map.nodes.len(), 9, "{map:?}");
    assert_eq!(map.edges.len(), 12, "{:?}", map.edges);
    let origin = &map.nodes[0];
    assert_eq!(origin.index, 2);
    assert!(near(origin, [0.0, 0.0, 0.0]), "{}", origin.position);
    assert!(origin.energy.abs() < 1.0e-8, "{}", origin.energy);
    assert!(
        (origin.curvatures[0] + 1.0).abs() < 1.0e-6,
        "{:?}",
        origin.curvatures
    );
    assert!(
        (origin.curvatures[1] + 0.5).abs() < 1.0e-6,
        "{:?}",
        origin.curvatures
    );

    let x_saddles = [
        require(&map.nodes, [1.0, 0.0, 0.0]),
        require(&map.nodes, [-1.0, 0.0, 0.0]),
    ];
    let y_saddles = [
        require(&map.nodes, [0.0, SIDE, 0.0]),
        require(&map.nodes, [0.0, -SIDE, 0.0]),
    ];
    for saddle in x_saddles {
        assert_eq!(saddle.index, 1);
        assert!((saddle.energy + 0.25).abs() < 1.0e-8, "{}", saddle.energy);
    }
    for saddle in y_saddles {
        assert_eq!(saddle.index, 1);
        assert!(
            (saddle.energy + 1.0 / 16.0).abs() < 1.0e-8,
            "{}",
            saddle.energy
        );
    }
    let minima = [
        [1.0, SIDE, 0.0],
        [1.0, -SIDE, 0.0],
        [-1.0, SIDE, 0.0],
        [-1.0, -SIDE, 0.0],
    ];
    for target in minima {
        let node = require(&map.nodes, target);
        assert_eq!(node.index, 0, "{target:?} {node:?}");
        assert!((node.energy + 5.0 / 16.0).abs() < 1.0e-8, "{}", node.energy);
        assert!(node.modes.ncols() == 0);
    }

    let id = |target: [f64; 3]| {
        map.nodes
            .iter()
            .position(|node| near(node, target))
            .unwrap()
    };
    let from_origin = |direction: usize| -> Vec<usize> {
        map.edges
            .iter()
            .filter(|edge| edge.parent == 0 && edge.direction == direction)
            .map(|edge| edge.child)
            .collect()
    };
    let mut along_x = from_origin(0);
    along_x.sort_unstable();
    let mut expect_x = vec![id([1.0, 0.0, 0.0]), id([-1.0, 0.0, 0.0])];
    expect_x.sort_unstable();
    assert_eq!(along_x, expect_x, "direction 0 is the most negative mode");
    let mut along_y = from_origin(1);
    along_y.sort_unstable();
    let mut expect_y = vec![id([0.0, SIDE, 0.0]), id([0.0, -SIDE, 0.0])];
    expect_y.sort_unstable();
    assert_eq!(along_y, expect_y);

    for target in [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]] {
        let children = children_of(map, id(target));
        assert_eq!(children.len(), 2, "{target:?}");
        assert!(children.contains(&id([target[0], SIDE, 0.0])));
        assert!(children.contains(&id([target[0], -SIDE, 0.0])));
    }
    for target in [[0.0, SIDE, 0.0], [0.0, -SIDE, 0.0]] {
        let children = children_of(map, id(target));
        assert_eq!(children.len(), 2, "{target:?}");
        assert!(children.contains(&id([1.0, target[1], 0.0])));
        assert!(children.contains(&id([-1.0, target[1], 0.0])));
    }
    for target in minima {
        let parents: Vec<usize> = map
            .edges
            .iter()
            .filter(|edge| edge.child == id(target))
            .map(|edge| edge.parent)
            .collect();
        assert_eq!(parents.len(), 2, "{target:?} {parents:?}");
    }
    let edge_sum: usize = map.edges.iter().map(|edge| edge.evaluations).sum();
    assert_eq!(map.evaluations, map.nodes[0].evaluations + edge_sum);
    let mut seen = vec![false; map.nodes.len()];
    seen[0] = true;
    for edge in &map.edges {
        if !seen[edge.child] {
            assert_eq!(map.nodes[edge.child].evaluations, edge.evaluations);
            seen[edge.child] = true;
        }
        assert_eq!(
            map.nodes[edge.child].index + 1,
            map.nodes[edge.parent].index
        );
        assert!(edge.sign == 1 || edge.sign == -1);
    }
}

fn children_of(map: &PathwayMap, parent: usize) -> Vec<usize> {
    let mut children: Vec<usize> = map
        .edges
        .iter()
        .filter(|edge| edge.parent == parent)
        .map(|edge| edge.child)
        .collect();
    children.sort_unstable();
    children.dedup();
    children
}

const MB_A: [f64; 4] = [-200.0, -100.0, -170.0, 15.0];
const MB_XX: [f64; 4] = [-1.0, -1.0, -6.5, 0.7];
const MB_XY: [f64; 4] = [0.0, 0.0, 11.0, 0.6];
const MB_YY: [f64; 4] = [-10.0, -10.0, -6.5, 0.7];
const MB_X0: [f64; 4] = [1.0, 0.0, -0.5, -1.0];
const MB_Y0: [f64; 4] = [0.0, 0.5, 1.5, 1.0];

fn muller_brown(x: f64, y: f64) -> (f64, [f64; 2], [[f64; 2]; 2]) {
    let (mut energy, mut gx, mut gy) = (0.0, 0.0, 0.0);
    let (mut hxx, mut hxy, mut hyy) = (0.0, 0.0, 0.0);
    for i in 0..4 {
        let (dx, dy) = (x - MB_X0[i], y - MB_Y0[i]);
        let q = MB_XX[i] * dx * dx + MB_XY[i] * dx * dy + MB_YY[i] * dy * dy;
        let term = MB_A[i] * q.exp();
        let qx = 2.0 * MB_XX[i] * dx + MB_XY[i] * dy;
        let qy = MB_XY[i] * dx + 2.0 * MB_YY[i] * dy;
        energy += term;
        gx += term * qx;
        gy += term * qy;
        hxx += term * (qx * qx + 2.0 * MB_XX[i]);
        hxy += term * (qx * qy + MB_XY[i]);
        hyy += term * (qy * qy + 2.0 * MB_YY[i]);
    }
    (energy, [gx, gy], [[hxx, hxy], [hxy, hyy]])
}

struct MullerBrown;

impl PointSurface for MullerBrown {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (energy, gradient, _) = muller_brown(x[0], x[1]);
        Ok((energy, Array1::from(gradient.to_vec())))
    }

    fn hessian_vector(
        &self,
        x: ArrayView1<f64>,
        v: ArrayView1<f64>,
    ) -> Result<Option<Array1<f64>>, SaddleError> {
        let (_, _, hessian) = muller_brown(x[0], x[1]);
        Ok(Some(array![
            hessian[0][0] * v[0] + hessian[0][1] * v[1],
            hessian[1][0] * v[0] + hessian[1][1] * v[1],
        ]))
    }
}

fn mb_config() -> LandscapeConfig {
    let search = MinModeConfig::default()
        .with_force_tol(1.0e-6)
        .with_max_move(0.05)
        .with_saddle_check(None);
    LandscapeConfig::default()
        .with_push(0.2)
        .with_duplicate_distance(5.0e-2)
        .with_max_steps(2_000)
        .with_search(IndexKConfig::new(1, search))
        .with_relax(
            RelaxConfig::default()
                .with_force_tol(1.0e-6)
                .with_max_steps(4_000)
                .with_max_atom_move(0.05)
                .with_initial_step(0.02),
        )
}

#[test]
fn muller_brown_saddles_connect_the_published_minima() {
    let saddles = [
        (
            [-0.822_001_558_733, 0.624_312_802_815],
            -40.664_843_508_657_f64,
        ),
        ([0.212_486_582_001, 0.292_988_325_107], -72.248_940_112_325),
    ];
    let basins = [
        [
            (
                [-0.558_223_634_633, 1.441_725_841_805],
                -146.699_517_209_954_f64,
            ),
            ([-0.050_010_822_998, 0.466_694_104_872], -80.767_818_129_659),
        ],
        [
            ([0.623_499_404_931, 0.028_037_758_529], -108.166_724_116_852),
            ([-0.050_010_822_998, 0.466_694_104_872], -80.767_818_129_659),
        ],
    ];
    let surface = Counted {
        inner: MullerBrown,
        calls: AtomicUsize::new(0),
    };
    for (saddle, ends) in saddles.iter().zip(basins) {
        let session =
            LandscapeSession::new(mb_config(), array![saddle.0[0], saddle.0[1]], None).unwrap();
        let map = session.run(&surface).unwrap();
        assert_eq!(map.nodes.len(), 3, "{:?}", map.nodes);
        assert_eq!(map.edges.len(), 2, "{:?}", map.edges);
        assert_eq!(map.nodes[0].index, 1);
        assert!(
            (map.nodes[0].energy - saddle.1).abs() < 1.0e-4,
            "{}",
            map.nodes[0].energy
        );
        assert!(
            (map.nodes[0].position[0] - saddle.0[0]).abs() < 1.0e-4
                && (map.nodes[0].position[1] - saddle.0[1]).abs() < 1.0e-4,
            "{}",
            map.nodes[0].position
        );
        for (end, energy) in ends {
            let node = map
                .nodes
                .iter()
                .find(|node| {
                    (node.position[0] - end[0]).abs() < 1.0e-2
                        && (node.position[1] - end[1]).abs() < 1.0e-2
                })
                .unwrap_or_else(|| panic!("missing {end:?} in {:?}", map.nodes));
            assert_eq!(node.index, 0);
            assert!((node.energy - energy).abs() < 1.0e-3, "{}", node.energy);
        }
        let edge_sum: usize = map.edges.iter().map(|edge| edge.evaluations).sum();
        assert_eq!(map.evaluations, map.nodes[0].evaluations + edge_sum);
        eprintln!(
            "MB saddle ({:.6}, {:.6}) nodes={} evaluations={}",
            saddle.0[0],
            saddle.0[1],
            map.nodes.len(),
            map.evaluations
        );
        for (i, node) in map.nodes.iter().enumerate() {
            eprintln!(
                "MB_NODE {i} index={} energy={:.6} evaluations={} position={:?}",
                node.index,
                node.energy,
                node.evaluations,
                node.position.as_slice()
            );
        }
    }
    eprintln!(
        "MB_TOTAL evaluations={}",
        surface.calls.load(Ordering::Relaxed)
    );
}

struct Cosine {
    length: f64,
}

impl PointSurface for Cosine {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let theta = std::f64::consts::TAU * x[0] / self.length;
        let scale = std::f64::consts::TAU / self.length;
        let energy = -theta.cos() + 0.5 * (x[1] * x[1] + x[2] * x[2]);
        Ok((energy, array![scale * theta.sin(), x[1], x[2]]))
    }

    fn hessian_vector(
        &self,
        x: ArrayView1<f64>,
        v: ArrayView1<f64>,
    ) -> Result<Option<Array1<f64>>, SaddleError> {
        let theta = std::f64::consts::TAU * x[0] / self.length;
        let scale = std::f64::consts::TAU / self.length;
        Ok(Some(
            array![scale * scale * theta.cos() * v[0], v[1], v[2],],
        ))
    }
}

fn cosine_config(cell: Option<Cell>) -> LandscapeConfig {
    let search = MinModeConfig::default()
        .with_force_tol(1.0e-8)
        .with_max_move(0.2)
        .with_saddle_check(None);
    LandscapeConfig::default()
        .with_push(0.4)
        .with_duplicate_distance(1.0e-2)
        .with_max_steps(400)
        .with_search(IndexKConfig::new(1, search))
        .with_relax(
            RelaxConfig::default()
                .with_force_tol(1.0e-8)
                .with_max_atom_move(0.2)
                .with_initial_step(0.05),
        )
        .with_cell(cell)
}

#[test]
fn a_periodic_cell_identifies_the_two_images_of_one_minimum() {
    let length = 4.0;
    let start = array![length / 2.0, 0.0, 0.0];
    let open = LandscapeSession::new(cosine_config(None), start.clone(), None)
        .unwrap()
        .run(&Cosine { length })
        .unwrap();
    assert_eq!(open.nodes.len(), 3, "{:?}", open.nodes);
    assert_eq!(open.nodes[0].index, 1);
    assert_eq!(open.nodes.iter().filter(|node| node.index == 0).count(), 2);

    let cell = Cell([[length, 0.0, 0.0], [0.0, length, 0.0], [0.0, 0.0, length]]);
    let wrapped = LandscapeSession::new(cosine_config(Some(cell)), start, None)
        .unwrap()
        .run(&Cosine { length })
        .unwrap();
    assert_eq!(wrapped.nodes.len(), 2, "{:?}", wrapped.nodes);
    assert_eq!(wrapped.edges.len(), 2, "{:?}", wrapped.edges);
    assert_eq!(wrapped.nodes[0].index, 1);
    assert_eq!(wrapped.nodes[1].index, 0);
    let image = wrapped.nodes[1].position[0].rem_euclid(length);
    assert!(image < 1.0e-3 || (image - length).abs() < 1.0e-3, "{image}");
}

#[test]
fn node_and_evaluation_limits_are_errors() {
    let refused = LandscapeSession::new(
        wells_config().with_max_nodes(1),
        array![0.0, 0.0, 0.0],
        None,
    )
    .unwrap()
    .run(&Wells);
    match refused {
        Err(SaddleError::LandscapeBound {
            kind: LandscapeBoundKind::Nodes,
            limit: 1,
            nodes: 1,
            evaluations,
        }) => assert!(evaluations > 0),
        other => panic!("{other:?}"),
    }

    // The analytic action certifies the parent on its one gradient, so the
    // next translation is the call the budget refuses. The parent is already
    // a node; the pathway is still not returned.
    let starved = LandscapeSession::new(
        wells_config().with_max_evaluations(1),
        array![0.0, 0.0, 0.0],
        None,
    )
    .unwrap()
    .run(&Wells);
    match starved {
        Err(SaddleError::LandscapeBound {
            kind: LandscapeBoundKind::Evaluations,
            limit: 1,
            nodes: 1,
            evaluations: 1,
        }) => {}
        other => panic!("{other:?}"),
    }

    struct FiniteDifference;
    impl PointSurface for FiniteDifference {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            Wells.eval(x)
        }
    }
    let during_certificate = LandscapeSession::new(
        wells_config().with_max_evaluations(1),
        array![0.0, 0.0, 0.0],
        None,
    )
    .unwrap()
    .run(&FiniteDifference);
    match during_certificate {
        Err(SaddleError::LandscapeBound {
            kind: LandscapeBoundKind::Evaluations,
            limit: 1,
            nodes: 0,
            evaluations: 1,
        }) => {}
        other => panic!("{other:?}"),
    }

    let done = LandscapeSession::new(wells_config(), array![0.0, 0.0, 0.0], None)
        .unwrap()
        .run(&Wells)
        .unwrap();
    let short = done.evaluations - 1;
    let stopped = LandscapeSession::new(
        wells_config().with_max_evaluations(short),
        array![0.0, 0.0, 0.0],
        None,
    )
    .unwrap()
    .run(&Wells);
    match stopped {
        Err(SaddleError::LandscapeBound {
            kind: LandscapeBoundKind::Evaluations,
            limit,
            evaluations,
            ..
        }) => {
            assert_eq!(limit, short);
            assert!(evaluations <= short, "{evaluations} {short}");
        }
        other => panic!("{other:?}"),
    }

    let invalid = LandscapeSession::new(
        wells_config().with_max_nodes(0),
        array![0.0, 0.0, 0.0],
        None,
    );
    assert!(matches!(invalid, Err(SaddleError::Invalid(_))));
}

#[test]
fn a_cell_on_a_coordinate_count_not_divisible_by_three_is_refused() {
    let cell = Cell([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    let session = LandscapeSession::new(mb_config().with_cell(cell), array![0.0, 0.0], None);
    assert!(matches!(session, Err(SaddleError::Invalid(_))));
}

/// Diagonal quadratic `V = ½ Σ λ_i x_i²`. The origin is the only critical point.
struct Diagonal {
    curvatures: [f64; 3],
}

impl PointSurface for Diagonal {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let mut energy = 0.0;
        let mut gradient = Array1::zeros(3);
        for i in 0..3 {
            energy += 0.5 * self.curvatures[i] * x[i] * x[i];
            gradient[i] = self.curvatures[i] * x[i];
        }
        Ok((energy, gradient))
    }

    fn hessian_vector(
        &self,
        _x: ArrayView1<f64>,
        v: ArrayView1<f64>,
    ) -> Result<Option<Array1<f64>>, SaddleError> {
        Ok(Some(array![
            self.curvatures[0] * v[0],
            self.curvatures[1] * v[1],
            self.curvatures[2] * v[2],
        ]))
    }
}

fn indexk_status(surface: &Diagonal, k: usize) -> MinModeStatus {
    let search = MinModeConfig::default()
        .with_force_tol(1.0e-8)
        .with_max_move(0.1)
        .with_saddle_check(None);
    let config = IndexKConfig::new(k, search).with_residual_tol(1.0e-10);
    let mut session = IndexKSession::new(config, array![0.0, 0.0, 0.0], None).unwrap();
    session.run(surface, 6).unwrap().status
}

fn certified_index(surface: &Diagonal) -> usize {
    let found = saddle_index(
        &SaddleIndexCheck::default(),
        surface,
        array![0.0, 0.0, 0.0].view(),
        None,
        None,
        &[],
    )
    .unwrap();
    assert!(found.converged, "{found:?}");
    found.index
}

#[test]
fn a_curvature_between_the_threshold_and_zero_is_not_a_negative_mode() {
    assert_eq!(
        SaddleIndexCheck::default().negative_tol,
        NEGATIVE_CURVATURE_TOL
    );
    // -5e-4 lies strictly between -1e-3 and 0.
    let soft = -0.5 * NEGATIVE_CURVATURE_TOL;
    assert!(soft < 0.0 && soft > -NEGATIVE_CURVATURE_TOL);
    let block = Diagonal {
        curvatures: [soft, -1.0, 1.0],
    };
    assert_ne!(
        indexk_status(&block, 2),
        MinModeStatus::Converged,
        "index-2 converged while one curvature is inside (-1e-3, 0)"
    );
    assert_eq!(certified_index(&block), 1);
    let one = Diagonal {
        curvatures: [soft, 1.0, 1.0],
    };
    assert_ne!(indexk_status(&one, 1), MinModeStatus::Converged);
    assert_eq!(certified_index(&one), 0);
}

#[test]
fn a_curvature_below_the_shared_threshold_certifies_as_index_k() {
    let below = -(NEGATIVE_CURVATURE_TOL + 1.0e-4);
    let surface = Diagonal {
        curvatures: [below, -1.0, 1.0],
    };
    assert_eq!(indexk_status(&surface, 2), MinModeStatus::Converged);
    assert_eq!(certified_index(&surface), 2);
    let edge = Diagonal {
        curvatures: [-NEGATIVE_CURVATURE_TOL, -1.0, 1.0],
    };
    assert_ne!(
        indexk_status(&edge, 2),
        MinModeStatus::Converged,
        "a curvature equal to -NEGATIVE_CURVATURE_TOL counts as negative"
    );
    assert_eq!(certified_index(&edge), 1);
}

fn axial_epsilon(soft: f64) -> f64 {
    2.0 * (1.0 + soft)
}

/// `V = Σ (x_i⁴ - ½ x_i²) + ε Σ_{i<j} x_i² x_j²` with `ε = 2(1+soft)`.
/// The origin has curvature -1 on each axis. Each axial point `(±0.5, 0, 0)`
/// and its permutations is critical, with one positive curvature and two
/// curvatures equal to `soft`.
struct Axial {
    soft: f64,
}

impl Axial {
    fn epsilon(&self) -> f64 {
        axial_epsilon(self.soft)
    }
}

fn quartic_well(t: f64) -> (f64, f64, f64) {
    // f = t^4 - 0.5 t^2, so f(0.5) is a minimum: f'(t) = 4 t^3 - t, f'(0.5) = 0.5 - 0.5 = 0.
    // f''(t) = 12 t^2 - 1, f''(0) = -1, f''(0.5) = 2.
    (
        t.powi(4) - 0.5 * t * t,
        4.0 * t.powi(3) - t,
        12.0 * t * t - 1.0,
    )
}

impl PointSurface for Axial {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let e = self.epsilon();
        let (mut energy, mut gradient) = (0.0, Array1::zeros(3));
        for i in 0..3 {
            let (value, first, _) = quartic_well(x[i]);
            energy += value;
            gradient[i] = first;
        }
        for (i, j) in [(0, 1), (0, 2), (1, 2)] {
            energy += e * x[i] * x[i] * x[j] * x[j];
            gradient[i] += 2.0 * e * x[i] * x[j] * x[j];
            gradient[j] += 2.0 * e * x[j] * x[i] * x[i];
        }
        Ok((energy, gradient))
    }

    fn hessian_vector(
        &self,
        x: ArrayView1<f64>,
        v: ArrayView1<f64>,
    ) -> Result<Option<Array1<f64>>, SaddleError> {
        let e = self.epsilon();
        let mut h = [[0.0; 3]; 3];
        for i in 0..3 {
            let (_, _, second) = quartic_well(x[i]);
            h[i][i] = second;
        }
        for (i, j) in [(0, 1), (0, 2), (1, 2)] {
            h[i][i] += 2.0 * e * x[j] * x[j];
            h[j][j] += 2.0 * e * x[i] * x[i];
            h[i][j] += 4.0 * e * x[i] * x[j];
            h[j][i] += 4.0 * e * x[i] * x[j];
        }
        let mut out = Array1::zeros(3);
        for i in 0..3 {
            for j in 0..3 {
                out[i] += h[i][j] * v[j];
            }
        }
        Ok(Some(out))
    }
}

fn axial_session(soft: f64, negative_tol: f64) -> LandscapeSession {
    let search = MinModeConfig::default()
        .with_force_tol(1.0e-8)
        .with_max_move(0.05)
        .with_saddle_check(None);
    let certificate = SaddleIndexCheck::default()
        .with_negative_tol(negative_tol)
        .with_residual_tol(1.0e-8);
    let config = LandscapeConfig::default()
        .with_push(0.5)
        .with_duplicate_distance(1.0e-2)
        .with_max_steps(30)
        .with_search(IndexKConfig::new(1, search).with_residual_tol(1.0e-8))
        .with_certificate(certificate)
        .with_relax(
            RelaxConfig::default()
                .with_force_tol(1.0e-8)
                .with_max_atom_move(0.05)
                .with_initial_step(0.02),
        );
    let modes = ndarray::Array2::eye(3);
    LandscapeSession::new(config, array![0.0, 0.0, 0.0], Some(modes))
        .unwrap_or_else(|err| panic!("axial {soft} session: {err}"))
}

fn assert_index_gap_refused(outcome: Result<PathwayMap, SaddleError>) {
    match outcome {
        Err(SaddleError::Solver(message)) => assert!(
            message.contains("index"),
            "solver error should name the index: {message}"
        ),
        Ok(map) => {
            let edges: Vec<_> = map
                .edges
                .iter()
                .map(|edge| (map.nodes[edge.parent].index, map.nodes[edge.child].index))
                .collect();
            panic!("stored an index gap: {edges:?}");
        }
        Err(err) => panic!("{err}"),
    }
}

#[test]
fn a_curvature_between_the_threshold_and_zero_cannot_skip_an_index() {
    let soft = -0.5 * NEGATIVE_CURVATURE_TOL;
    assert!(soft < 0.0 && soft > -NEGATIVE_CURVATURE_TOL);
    let outcome = axial_session(soft, NEGATIVE_CURVATURE_TOL).run(&Axial { soft });
    assert_index_gap_refused(outcome);
}

#[test]
fn a_downward_leg_requires_the_child_index_to_be_one_less() {
    // -5e-3 is below the index-k threshold and above this certificate
    // threshold, so index-2 dynamics converge and the certificate reports
    // index 0. The parent is index 3, and the edge has to be refused.
    let soft = -5.0e-3;
    let negative_tol = 1.0e-2;
    assert!(soft < -NEGATIVE_CURVATURE_TOL && soft > -negative_tol);
    let outcome = axial_session(soft, negative_tol).run(&Axial { soft });
    match outcome {
        Err(SaddleError::Solver(message)) => {
            assert!(
                message.contains("child index is"),
                "expected the child index to be refused, got {message}"
            );
        }
        other => panic!("expected a refused child index, got {other:?}"),
    }
}

#[test]
fn a_dependent_unstable_block_is_refused() {
    let mut modes = ndarray::Array2::zeros((3, 2));
    modes[(0, 0)] = 1.0;
    modes[(0, 1)] = 2.0;
    let err = LandscapeSession::new(wells_config(), array![0.0, 0.0, 0.0], Some(modes));
    match err {
        Err(SaddleError::Invalid(message)) => {
            assert!(message.contains("dependent"), "{message}");
        }
        Err(other) => panic!("{other}"),
        Ok(_) => panic!("a dependent block was accepted"),
    }
}

#[test]
fn a_non_positive_search_step_is_refused() {
    let mut config = wells_config();
    config.search.search.dr = 0.0;
    let err = LandscapeSession::new(config, array![0.0, 0.0, 0.0], None);
    assert!(matches!(err, Err(SaddleError::Invalid(_))));
}

#[test]
fn a_minimum_leg_stops_on_the_relax_tolerance() {
    let search = MinModeConfig::default()
        .with_force_tol(1.0e-8)
        .with_max_move(0.1)
        .with_saddle_check(None);
    let config = LandscapeConfig::default()
        .with_push(0.70)
        .with_duplicate_distance(1.0e-2)
        .with_max_steps(400)
        .with_search(IndexKConfig::new(1, search))
        .with_relax(
            RelaxConfig::default()
                .with_force_tol(1.0e-2)
                .with_max_atom_move(0.1)
                .with_initial_step(0.05),
        );
    let map = LandscapeSession::new(config, array![1.0, 0.0, 0.0], None)
        .unwrap()
        .run(&Wells)
        .unwrap_or_else(|err| panic!("minimum leg: {err}"));
    assert!(map.nodes.iter().any(|node| node.index == 0));
}

/// Two double wells on collective coordinates, so a Cartesian gradient
/// has equal components and the Euclidean norm is larger than the
/// largest component.
struct Spread;

fn spread_parts(x: ArrayView1<f64>) -> (f64, f64, f64, f64) {
    let s2 = std::f64::consts::FRAC_1_SQRT_2;
    let s = (x[0] + x[3]) * s2;
    let u = (x[1] + x[4]) * s2;
    let t = (x[0] - x[3]) * s2;
    let v = (x[1] - x[4]) * s2;
    (s, u, t, v)
}

fn quartic(q: f64) -> (f64, f64, f64) {
    (
        (q.powi(4) - 2.0 * q * q) / 4.0,
        q.powi(3) - q,
        3.0 * q * q - 1.0,
    )
}

impl PointSurface for Spread {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (s, u, t, v) = spread_parts(x);
        let (es, ds, _) = quartic(s);
        let (eu, du, _) = quartic(u);
        let s2 = std::f64::consts::FRAC_1_SQRT_2;
        let energy = es + eu + 0.5 * (t * t + v * v + x[2] * x[2] + x[5] * x[5]);
        let gradient = array![
            (ds + t) * s2,
            (du + v) * s2,
            x[2],
            (ds - t) * s2,
            (du - v) * s2,
            x[5],
        ];
        Ok((energy, gradient))
    }

    fn hessian_vector(
        &self,
        x: ArrayView1<f64>,
        vector: ArrayView1<f64>,
    ) -> Result<Option<Array1<f64>>, SaddleError> {
        let (s, u, _, _) = spread_parts(x);
        let (_, _, dds) = quartic(s);
        let (_, _, ddu) = quartic(u);
        let (ds, du, dt, dv) = spread_parts(vector);
        let s2 = std::f64::consts::FRAC_1_SQRT_2;
        Ok(Some(array![
            (dds * ds + dt) * s2,
            (ddu * du + dv) * s2,
            vector[2],
            (dds * ds - dt) * s2,
            (ddu * du - dv) * s2,
            vector[5],
        ]))
    }
}

#[test]
fn the_index_k_leg_uses_the_configured_force_gate() {
    let s2 = std::f64::consts::FRAC_1_SQRT_2;
    let mut modes = ndarray::Array2::zeros((6, 2));
    modes[(0, 0)] = s2;
    modes[(3, 0)] = s2;
    modes[(1, 1)] = s2;
    modes[(4, 1)] = s2;
    let search = MinModeConfig::default()
        .with_force_tol(1.6e-2)
        .with_max_move(0.05)
        .with_saddle_check(None);
    let config = LandscapeConfig::default()
        .with_push(0.99)
        .with_duplicate_distance(5.0e-2)
        .with_max_steps(80)
        .with_force_gate(ForceGate::L2Norm)
        .with_search(IndexKConfig::new(1, search).with_residual_tol(1.0e-8))
        .with_relax(
            RelaxConfig::default()
                .with_force_tol(1.6e-2)
                .with_max_atom_move(0.05)
                .with_initial_step(0.02),
        );
    let map = LandscapeSession::new(config, Array1::zeros(6), Some(modes))
        .unwrap()
        .run(&Spread)
        .unwrap_or_else(|err| panic!("configured gate: {err}"));
    assert!(map.nodes.iter().any(|node| node.index == 1));
}

#[test]
fn landscape_config_setters_cover_every_field() {
    let search = IndexKConfig::new(2, MinModeConfig::default().with_force_tol(1.0e-6));
    let relax = RelaxConfig::default()
        .with_force_tol(2.0e-6)
        .with_max_steps(30);
    let certificate = SaddleIndexCheck::default().with_negative_tol(1.0e-4);
    let cell = Cell([[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]]);
    let config = LandscapeConfig::default()
        .with_duplicate_distance(0.02)
        .with_push(0.3)
        .with_max_nodes(8)
        .with_max_evaluations(90)
        .with_max_steps(40)
        .with_search(search.clone())
        .with_relax(relax.clone())
        .with_certificate(certificate)
        .with_force_gate(ForceGate::L2Norm)
        .with_cell(cell);
    assert_eq!(config.duplicate_distance, 0.02);
    assert_eq!(config.push, 0.3);
    assert_eq!(config.max_nodes, 8);
    assert_eq!(config.max_evaluations, 90);
    assert_eq!(config.max_steps, 40);
    assert_eq!(config.search.k, search.k);
    assert_eq!(config.search.search.force_tol, search.search.force_tol);
    assert_eq!(config.relax.force_tol, relax.force_tol);
    assert_eq!(config.relax.max_steps, relax.max_steps);
    assert_eq!(config.certificate, certificate);
    assert_eq!(config.force_gate, ForceGate::L2Norm);
    assert_eq!(config.cell.map(|cell| cell.0), Some(cell.0));
}
