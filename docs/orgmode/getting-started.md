A host builds a band, steps until the projected force is under `force_tol`, and never lets this crate own the loop.


# 1. Depend on the published crate

```toml
[dependencies]
rgsaddle = "0.6.0"
ndarray = "0.17"
```

A host that also names rgmin types, such as `QuickMinOptions`, asks for the rgmin release and source in rgsaddle's `Cargo.toml`.
Cargo builds any other release or source as a second crate, and its types do not match the ones rgsaddle takes.


# 2. Implement `BandSurface`

One eval fills energies and gradients for the rows it receives: the whole band on the first call, the interior images afterwards (the endpoints never move, so their energies are cached).
The double well $V = (x^2 - 1)^2 + 2y^2 + 2z^2$ has minima at $(\pm 1, 0, 0)$ and a saddle at the origin with energy 1:

```rust
/// Energy and gradient of the double well at (x, y, z).
fn double_well(x: f64, y: f64, z: f64) -> (f64, [f64; 3]) {
    let energy = (x * x - 1.0).powi(2) + 2.0 * y * y + 2.0 * z * z;
    (energy, [4.0 * x * (x * x - 1.0), 4.0 * y, 4.0 * z])
}

struct DoubleWell;

impl BandSurface for DoubleWell {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (i, row) in positions.outer_iter().enumerate() {
            let (energy, gradient) = double_well(row[0], row[1], row[2]);
            energies[i] = energy;
            for (k, g) in gradient.into_iter().enumerate() {
                gradients[(i, k)] = g;
            }
        }
        Ok(())
    }
}
```


# 3. Step

Nine images run from one minimum to the other, bowed off the path so the band has work to do.
The host calls `step` until the band reports `Converged`:

```rust
// Nine images from (-1, 0, 0) to (1, 0, 0), bowed off the path in y.
let initial = Array2::from_shape_fn((9, 3), |(i, k)| {
    let t = i as f64 / 8.0;
    match k {
        0 => 2.0 * t - 1.0,
        1 => 0.3 * (std::f64::consts::PI * t).sin(),
        _ => 0.0,
    }
});

let config = BandConfig::default()
    .with_force_tol(1e-3)
    .with_max_move(0.1);
let mut session = BandSession::new(config, initial)?;
let report = loop {
    let report = session.step(&DoubleWell)?;
    if report.status == BandStatus::Converged {
        break report;
    }
};

let ci = report.ci_index.expect("a climbing image");
```

Default stepper is FIRE, as in eOn's velocity NEB stepper.
L-BFGS runs under rgmin's `Accept::Step`: the projected band force is not conservative, so a step is taken without an energy test.


# 4. Run it

The code above is `examples/double_well.rs`.
From the repository root:

```console
$ cargo run --example double_well
```

    Converged after 8 steps, max force 2.6e-4
    climbing image 4 at (0.0000, -0.0001, 0.0000), energy 1.000000

The climbing image is image 4, the middle of the nine.
It sits on the saddle at the origin, and its energy is the barrier.


# 5. Reset when the surface changes

`BandSession::reset` is the model-update boundary.
Quasi-Newton history taken on one surface epoch must not survive onto the next.


# Next
