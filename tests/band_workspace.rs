// Copyright (c) rgsaddle developers

//! The band force reuses its buffers from one evaluation to the next:
//! a warm step allocates the same number of times whatever the image
//! count, so no allocation is made per image. The trajectories stay
//! bitwise those of the allocating assembly, pinned by hashes taken
//! on it.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use ndarray::{Array1, Array2, ArrayView2};
use rgsaddle::{
    BandConfig, BandSession, BandSurface, CiConfig, ProjectionKind, SaddleError, SpringKind,
};

struct Counting;

thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
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

/// Two atoms in three dimensions, each coordinate in a double well,
/// neighbouring coordinates coupled. Allocation-free.
struct Wells;

const ATOMS: usize = 2;
const DOF: usize = 3 * ATOMS;

fn wells(x: &[f64], g: &mut [f64]) -> f64 {
    let mut e = 0.0;
    for j in 0..x.len() {
        let w = x[j] * x[j] - 1.0;
        e += w * w;
        g[j] = 4.0 * x[j] * w;
    }
    for j in 1..x.len() {
        e += 0.3 * x[j] * x[j - 1];
        g[j] += 0.3 * x[j - 1];
        g[j - 1] += 0.3 * x[j];
    }
    e
}

impl BandSurface for Wells {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (r, row) in positions.outer_iter().enumerate() {
            let x = row.as_slice().expect("contiguous rows");
            let g = gradients.row_mut(r).into_slice().expect("contiguous rows");
            energies[r] = wells(x, g);
        }
        Ok(())
    }
}

fn band(n_images: usize) -> Array2<f64> {
    let start: [f64; DOF] = [-1.0, -0.9, -1.1, -1.0, -0.95, -1.05];
    let end: [f64; DOF] = [1.0, 1.1, 0.9, 1.05, 1.0, 0.95];
    Array2::from_shape_fn((n_images, DOF), |(i, c)| {
        let t = i as f64 / (n_images - 1) as f64;
        let bow = 0.2 * (std::f64::consts::PI * t).sin() * if c % 2 == 0 { 1.0 } else { -1.0 };
        (1.0 - t) * start[c] + t * end[c] + bow
    })
}

fn configs() -> Vec<(&'static str, BandConfig)> {
    let base = BandConfig {
        climbing: Some(CiConfig::always()),
        min_mode: None,
        max_move: 0.05,
        force_tol: 1e-9,
        ..BandConfig::default()
    };
    vec![
        ("cineb-uniform-fire", base.clone()),
        (
            "neb-uniform-fire",
            BandConfig {
                climbing: None,
                ..base.clone()
            },
        ),
        (
            "dneb-weighted-fire",
            BandConfig {
                projection: ProjectionKind::DoublyNudged,
                spring: SpringKind::EnergyWeighted {
                    k_min: 1.0,
                    k_max: 5.0,
                },
                ..base.clone()
            },
        ),
        (
            "peb-om-live-fire",
            BandConfig {
                projection: ProjectionKind::PlainElasticBand,
                spring: SpringKind::OnsagerMachlupLive { k: 5.0 },
                climbing: None,
                ..base.clone()
            },
        ),
        (
            "cineb-uniform-quickmin",
            BandConfig {
                method: rgmin::Method::QuickMin,
                quickmin: true,
                ..base.clone()
            },
        ),
    ]
}

fn step_allocations(config: &BandConfig, n_images: usize, warm: usize, steps: usize) -> Vec<usize> {
    let mut session = BandSession::new(config.clone(), band(n_images)).unwrap();
    for _ in 0..warm {
        session.step(&Wells).unwrap();
    }
    (0..steps)
        .map(|_| {
            let before = allocations();
            session.step(&Wells).unwrap();
            allocations() - before
        })
        .collect()
}

fn fnv1a(hash: &mut u64, value: f64) {
    for byte in value.to_bits().to_le_bytes() {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

/// FNV-1a over the bits of every step's band and force.
fn trajectory_hash(config: &BandConfig, n_images: usize, steps: usize) -> u64 {
    let mut session = BandSession::new(config.clone(), band(n_images)).unwrap();
    let mut hash = 0xcbf2_9ce4_8422_2325;
    for _ in 0..steps {
        let report = session.step(&Wells).unwrap();
        fnv1a(&mut hash, report.max_force);
        session
            .positions()
            .iter()
            .for_each(|&v| fnv1a(&mut hash, v));
    }
    hash
}

#[test]
fn band_trajectories_match_the_allocating_assembly_bitwise() {
    let expected = [
        ("cineb-uniform-fire", 0x80c7_3a0c_da8b_a09e),
        ("neb-uniform-fire", 0xcb00_29a5_02ca_20b9),
        ("dneb-weighted-fire", 0x0a11_786f_e60a_cd48),
        ("peb-om-live-fire", 0x3b34_02d1_dec7_90d6),
        ("cineb-uniform-quickmin", 0x2904_fa78_e5d2_b94f),
    ];
    let got: Vec<_> = configs()
        .into_iter()
        .map(|(label, config)| (label, trajectory_hash(&config, 7, 60)))
        .collect();
    let shown: Vec<String> = got
        .iter()
        .map(|(l, h)| format!("(\"{l}\", 0x{h:016x})"))
        .collect();
    assert_eq!(got, expected, "{}", shown.join(",\n"));
}

#[test]
fn warm_band_steps_allocate_independently_of_the_image_count() {
    for (label, config) in configs() {
        let few = step_allocations(&config, 5, 4, 8);
        let many = step_allocations(&config, 13, 4, 8);
        eprintln!("{label}: {few:?} {many:?}");
        assert_eq!(
            few, many,
            "{label}: per-step allocations, 5 against 13 images"
        );
    }
}
