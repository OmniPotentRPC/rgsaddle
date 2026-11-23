// Copyright (c) rgsaddle developers

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use ndarray::{Array1, Array2, ArrayView2};
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

criterion_group!(benches, smoke);
criterion_main!(benches);
