// Copyright (c) rgsaddle developers

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use ndarray::{Array1, Array2, ArrayView2, s};
use rgsaddle::{
    BandConfig, BandSession, BandSurface, Cell, ExpBandPrecon, Reparametrization, SaddleError,
    reparametrize,
};

struct Harmonic;

impl BandSurface for Harmonic {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (image, row) in positions.outer_iter().enumerate() {
            energies[image] = 0.5 * row.dot(&row);
            gradients.row_mut(image).assign(&row);
        }
        Ok(())
    }
}

fn periodic_band(images: usize) -> Array2<f64> {
    Array2::from_shape_fn((images, 96), |(image, coordinate)| {
        let atom = coordinate / 3;
        let axis = coordinate % 3;
        let lattice = [
            (atom % 4) as f64 + 0.2,
            ((atom / 4) % 4) as f64 + 0.2,
            (atom / 16) as f64 + 0.2,
        ];
        lattice[axis] + 0.01 * image as f64 * (atom + axis + 1) as f64
    })
}

fn smoke(c: &mut Criterion) {
    let positions = periodic_band(10);
    let config = BandConfig::default()
        .with_climbing(None)
        .with_min_mode(None)
        .with_cell(Cell([[8.0, 0.0, 0.0], [0.7, 8.0, 0.0], [0.2, 0.4, 8.0]]))
        .with_method(rgmin::Method::Ode12r)
        .with_exp_precon(ExpBandPrecon::default().with_mu(1.0));
    c.bench_function("band-exp-32-atoms-8-interior", |b| {
        b.iter_batched(
            || BandSession::new(config.clone(), positions.clone()).expect("band session"),
            |mut session| session.step(&Harmonic).expect("band step"),
            BatchSize::SmallInput,
        );
    });

    let spline = periodic_band(64);
    c.bench_function("not-a-knot-64-images-96-dof", |b| {
        b.iter_batched(
            || spline.clone(),
            |mut band| reparametrize(&mut band, None, None, Reparametrization::NotAKnot),
            BatchSize::SmallInput,
        );
    });
}

fn slab_lattice_band(images: usize, layers_z: usize) -> (Array2<f64>, Cell) {
    let atoms_per_layer = 16;
    let n_atoms = atoms_per_layer * layers_z;
    let dof = 3 * n_atoms;
    let positions = Array2::from_shape_fn((images, dof), |(image, coordinate)| {
        let atom = coordinate / 3;
        let axis = coordinate % 3;
        let lattice = [
            (atom % 4) as f64 + 0.2,
            ((atom / 4) % 4) as f64 + 0.2,
            (atom / atoms_per_layer) as f64 + 0.2,
        ];
        let rattle = 0.001 * ((atom * 7 + axis * 3 + 1) % 11) as f64;
        lattice[axis] + rattle + 0.002 * image as f64 * ((axis + 1) as f64)
    });
    let z_len = (layers_z as f64 * 2.0).max(8.0);
    let cell = Cell([[8.0, 0.0, 0.0], [0.7, 8.0, 0.0], [0.2, 0.4, z_len]]);
    (positions, cell)
}

fn exp_metric_force_vs_dof(c: &mut Criterion) {
    let images = 10;
    let n_interior = images - 2;
    let mut assembly_group = c.benchmark_group("exp_metric_warm_step_vs_dof");
    for layers_z in [2usize, 8, 32, 128] {
        let (positions, cell) = slab_lattice_band(images, layers_z);
        let dof_per_image = positions.ncols();
        let total_interior_dof = n_interior * dof_per_image;
        let config = BandConfig::default()
            .with_quickmin(true)
            .with_climbing(None)
            .with_min_mode(None)
            .with_saddle_check(None)
            .with_cell(cell)
            .with_time_step(0.01)
            .with_max_move(0.000_000_1)
            .with_force_tol(0.0)
            .with_exp_precon(ExpBandPrecon::default().with_mu(1.0));
        let mut session = BandSession::new(config, positions).expect("band session");
        session.step(&Harmonic).expect("warm-up metric build");

        assembly_group.throughput(Throughput::Elements(total_interior_dof as u64));
        assembly_group.bench_with_input(
            BenchmarkId::new("interior_dof", total_interior_dof),
            &total_interior_dof,
            |b, _| {
                b.iter(|| session.step(&Harmonic).expect("warm exp-metric step"));
            },
        );
    }
    assembly_group.finish();

    let mut scan_group = c.benchmark_group("exp_metric_all_dof_scan_vs_dof");
    for layers_z in [2usize, 8, 32, 128] {
        let (positions, _) = slab_lattice_band(images, layers_z);
        let built_at: Array1<f64> = positions
            .slice(s![1..images - 1, ..])
            .iter()
            .copied()
            .collect();
        let trial = &built_at + 0.000_1;
        let total_interior_dof = built_at.len();

        scan_group.throughput(Throughput::Elements(total_interior_dof as u64));
        scan_group.bench_with_input(
            BenchmarkId::new("interior_dof", total_interior_dof),
            &total_interior_dof,
            |b, _| {
                b.iter(|| {
                    std::hint::black_box(rgmin::max_atom_displacement(
                        std::hint::black_box(built_at.view()),
                        std::hint::black_box(trial.view()),
                    ))
                });
            },
        );
    }
    scan_group.finish();
}

criterion_group!(benches, smoke, exp_metric_force_vs_dof);
criterion_main!(benches);
