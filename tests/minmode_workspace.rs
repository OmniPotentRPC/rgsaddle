// Copyright (c) rgsaddle developers

//! The dimer rotation and the Lanczos iteration reuse their buffers:
//! apart from the arrays the surface returns, a longer rotation or a
//! larger Krylov space allocates nothing more. The estimates and the
//! searches stay bitwise those of the allocating code, pinned by
//! hashes taken on it.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use ndarray::{Array1, ArrayView1};
use rgsaddle::{
    FiniteDifference, MinModeConfig, MinModeKind, MinModeSession, MinModeStatus, PointSurface,
    SaddleError,
};

struct Counting;

thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    static SURFACE: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.with(|n| n.set(n.get() + 1));
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        ALLOCATIONS.with(|n| n.set(n.get() + 1));
        unsafe { System.realloc(ptr, layout, size) }
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

fn allocations() -> usize {
    ALLOCATIONS.with(Cell::get)
}

/// Allocations made outside the surface's own calls.
fn library_allocations() -> usize {
    allocations() - SURFACE.with(Cell::get)
}

const DOF: usize = 9;

/// A saddle of coupled anharmonic coordinates: a double well along
/// the first, quartic wells along the rest. Its allocations, the
/// returned gradient included, are booked to the surface.
struct Saddle;

impl PointSurface for Saddle {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let before = allocations();
        let mut g = Array1::zeros(x.len());
        let w = x[0] * x[0] - 1.0;
        let mut e = -w * w;
        g[0] = -4.0 * x[0] * w;
        for j in 1..x.len() {
            let k = 1.0 + 0.5 * j as f64;
            e += k * x[j] * x[j] + 0.1 * x[j].powi(4) + 0.2 * x[j] * x[j - 1];
            g[j] += 2.0 * k * x[j] + 0.4 * x[j].powi(3) + 0.2 * x[j - 1];
            g[j - 1] += 0.2 * x[j];
        }
        SURFACE.with(|n| n.set(n.get() + allocations() - before));
        Ok((e, g))
    }
}

fn start() -> (Array1<f64>, Array1<f64>) {
    let x = Array1::from_shape_fn(DOF, |j| 0.3 / (1.0 + j as f64) - 0.05 * j as f64);
    let mode = Array1::from_shape_fn(DOF, |j| 1.0 / (1.0 + j as f64));
    (x, mode)
}

/// Library allocations of one warm estimate, and its rotations.
fn estimate_allocations(config: MinModeConfig) -> (usize, usize) {
    let (x, mode) = start();
    let mut session = MinModeSession::new(config, x, mode).unwrap();
    session.set_torque_window(Some(false));
    // Two estimates leave the buffers a third one reuses.
    session.estimate_mode(&Saddle).unwrap();
    session.estimate_mode(&Saddle).unwrap();
    let before = library_allocations();
    let estimate = session.estimate_mode(&Saddle).unwrap();
    (library_allocations() - before, estimate.rotations)
}

#[test]
fn a_longer_dimer_rotation_allocates_nothing_more() {
    for difference in [FiniteDifference::Forward, FiniteDifference::Central] {
        let config = |max_rotations| MinModeConfig {
            kind: MinModeKind::Dimer,
            difference,
            rotation_tol: 0.0,
            rotation_angle_tol: 0.0,
            max_rotations,
            ..MinModeConfig::default()
        };
        let (short, short_rotations) = estimate_allocations(config(2));
        let (long, long_rotations) = estimate_allocations(config(8));
        assert!(long_rotations > short_rotations, "{difference:?}");
        assert_eq!(
            short, long,
            "{difference:?}: {short_rotations} against {long_rotations} rotations"
        );
    }
}

#[test]
fn a_larger_krylov_space_allocates_nothing_more() {
    for difference in [FiniteDifference::Forward, FiniteDifference::Central] {
        let config = |krylov_dim| MinModeConfig {
            kind: MinModeKind::Lanczos,
            difference,
            rotation_tol: 0.0,
            krylov_dim,
            ..MinModeConfig::default()
        };
        let (small, small_actions) = estimate_allocations(config(2));
        let (large, large_actions) = estimate_allocations(config(8));
        assert!(large_actions > small_actions, "{difference:?}");
        assert_eq!(
            small, large,
            "{difference:?}: {small_actions} against {large_actions} products"
        );
    }
}

fn fnv1a(hash: &mut u64, value: f64) {
    for byte in value.to_bits().to_le_bytes() {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

/// FNV-1a over every step's position, mode, force and curvature, and
/// over the Krylov block of a Lanczos session.
fn search_hash(config: MinModeConfig, steps: usize) -> u64 {
    let (x, mode) = start();
    let mut session = MinModeSession::new(config, x, mode).unwrap();
    let mut hash = 0xcbf2_9ce4_8422_2325;
    for _ in 0..steps {
        let report = session.step(&Saddle).unwrap();
        fnv1a(&mut hash, report.max_force);
        fnv1a(&mut hash, report.curvature);
        session.position().iter().for_each(|&v| fnv1a(&mut hash, v));
        session.mode().iter().for_each(|&v| fnv1a(&mut hash, v));
        for (q, hq) in session.krylov_block() {
            q.iter().chain(hq.iter()).for_each(|&v| fnv1a(&mut hash, v));
        }
        if report.status != MinModeStatus::Running {
            break;
        }
    }
    hash
}

#[test]
fn searches_match_the_allocating_code_bitwise() {
    let lbfgs = rgmin::Method::Lbfgs { memory: 5 };
    let cases = [
        (
            "dimer-forward-fire",
            MinModeKind::Dimer,
            FiniteDifference::Forward,
            None,
            0x6aa6_d128_0d08_9963,
        ),
        (
            "dimer-central-fire",
            MinModeKind::Dimer,
            FiniteDifference::Central,
            None,
            0x27c1_53f7_c02b_944f,
        ),
        (
            "dimer-forward-lbfgs",
            MinModeKind::Dimer,
            FiniteDifference::Forward,
            Some(&lbfgs),
            0x932d_afe7_39a1_2065,
        ),
        (
            "lanczos-forward-fire",
            MinModeKind::Lanczos,
            FiniteDifference::Forward,
            None,
            0xb735_0bce_aca5_a0f0,
        ),
        (
            "lanczos-central-lbfgs",
            MinModeKind::Lanczos,
            FiniteDifference::Central,
            Some(&lbfgs),
            0x62bc_1c25_7ad8_4dbb,
        ),
    ];
    let got: Vec<(&str, u64)> = cases
        .iter()
        .map(|&(label, kind, difference, method, _)| {
            let mut config = MinModeConfig {
                kind,
                difference,
                max_move: 0.1,
                force_tol: 1e-8,
                ..MinModeConfig::default()
            };
            if let Some(method) = method {
                config.method = method.clone();
            }
            (label, search_hash(config, 200))
        })
        .collect();
    let expected: Vec<(&str, u64)> = cases.iter().map(|&(l, _, _, _, h)| (l, h)).collect();
    let shown: Vec<String> = got
        .iter()
        .map(|(l, h)| format!("(\"{l}\", 0x{h:016x})"))
        .collect();
    assert_eq!(got, expected, "{}", shown.join(",\n"));
}
