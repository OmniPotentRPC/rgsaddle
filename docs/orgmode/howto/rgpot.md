`rgsaddle` never asks what kind of potential a `PointSurface` wraps. This how-to wraps a real one from [rgpot](https://github.com/OmniPotentRPC/rgpot), the sibling potential project, instead of an analytic model, and runs the crate's usual pieces (relaxation, a dimer search, a PAFI barrier) on it unchanged.


# Which half of rgpot

rgpot splits into two crates. `rgpot-core` is the RPC/callback half: the physics lives in a C++ core reached over capnp, through a server binary or the Python wheel, and pulling it in means capnp, tokio and a running server. `rgpot-pair` is the other half: plain, zero-dependency Rust functions over flat position arrays, the pairwise kernels (Lennard-Jones, Morse, binary LJ, TIP4P sites, Sutton-Chen scales) inlined with no library boundary per pair. This how-to uses `rgpot-pair`, pinned the same way `ci/consumer/Cargo.toml` pins `rgpot-core`, by the 40-hex revision of the release tag:

```toml
[dev-dependencies]
rgpot-pair = { version = "=3.6.0", git = "https://github.com/OmniPotentRPC/rgpot", rev = "20c58fe5a2859cd0a79a3e06d64d567f96d0ccbe" }
```

`rgpot-pair` has no dependencies of its own, so it adds no second copy of `eindir-core`, `dlpk` or `rgmin`, the crates `ci/check_dependency_pins.py` watches for.


# The system

`lj_gradient(epsilon, sigma, cutoff, positions, gradient)` takes the whole configuration, a flat array of 3-D atoms, and returns the total energy while filling `gradient` in place; a non-positive cutoff keeps every pair. The system here is one migrating atom `B` between two fixed walls `A` and `C` at `(-1.5, 0, 0)` and `(1.5, 0, 0)`, unit epsilon and sigma. The walls sit past the pair minimum, $r_\text{min} = 2^{1/6}$, so `B` prefers to sit near one wall or the other: the line between them is a double well with the saddle at the midpoint by symmetry. Wrapping that in `PointSurface` only needs to pick `B`'s three coordinates back out of the gradient `rgpot-pair` wrote for all three atoms:

```rust
/// Half the A-C separation. Past `rmin = 2^(1/6)`, so the midpoint is a
/// barrier rather than the single minimum.
const WALL: f64 = 1.5;

/// `B`'s position in the fixed-wall frame: a thin [`PointSurface`] over
/// `rgpot_pair::lj_gradient`, which wants a flat array of 3 atoms.
struct MigratingAtom;

impl PointSurface for MigratingAtom {
    fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let positions = [-WALL, 0.0, 0.0, q[0], q[1], q[2], WALL, 0.0, 0.0];
        let mut gradient = [0.0; 9];
        let energy = rgpot_pair::lj_gradient(1.0, 1.0, -1.0, &positions, &mut gradient);
        Ok((energy, array![gradient[3], gradient[4], gradient[5]]))
    }
```

Everything past this point is ordinary `rgsaddle`: `relax_to_tolerance`, `MinModeSession` and `zero_temperature_barrier` do not know the gradient came from `rgpot-pair` instead of a closed-form expression.


# Relax to a minimum, then find the saddle

```rust
fn main() -> Result<(), SaddleError> {
    let surface = MigratingAtom;

    let cell = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let outcome = relax_to_tolerance(
        &surface,
        array![-0.6, 0.02, 0.0],
        cell,
        &RelaxConfig::default(),
    )?;
    let near_a = outcome.positions;
    println!(
        "minimum near A: ({:.6}, {:.6}, {:.6}), energy {:.6}, {} steps",
        near_a[0], near_a[1], near_a[2], outcome.energy, outcome.steps
    );

    let config = MinModeConfig::default().with_force_tol(1.0e-8);
    let mut session = MinModeSession::new(config, array![0.0, 0.05, 0.0], array![1.0, 0.0, 0.0])?;
    let mut converged = false;
    for _ in 0..2000 {
        if session.step(&surface)?.status == MinModeStatus::Converged {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(SaddleError::Solver("the dimer did not converge".into()));
    }
    let saddle = session.position().to_owned();
    println!(
        "saddle: ({:.6}, {:.6}, {:.6}), energy {:.6}",
        saddle[0],
        saddle[1],
        saddle[2],
        surface.eval(saddle.view())?.0
```


# Check the barrier two ways

`zero_temperature_barrier` integrates the gradient along a short `ArcPath` through the two relaxed minima and the saddle, a second, independent measurement of the same barrier `relax_to_tolerance` and the dimer already gave directly:

```rust
let saddle_energy = surface.eval(saddle.view())?.0;
let barrier = saddle_energy - outcome.energy;
println!("zero-temperature barrier from the relaxed minimum: {barrier:.6}");

let mirror = array![-near_a[0], near_a[1], near_a[2]];
let path = ArcPath::from_positions(&[near_a.clone(), saddle.clone(), mirror])?;
let cold = zero_temperature_barrier(&surface, &path, &path.arcs())?;
println!(
    "same barrier along a PAFI path over this surface: {:.6}",
    cold.barrier
);
```


# Run it

```console
$ cargo run --release --example rgpot_howto
```

    minimum near A: (-0.372331, 0.002303, 0.000000), energy -1.095419, 49 steps
    saddle: (-0.000000, -0.000000, 0.000000), energy -0.646153
    zero-temperature barrier from the relaxed minimum: 0.449267
    same barrier along a PAFI path over this surface: 0.449099

The relaxed minimum sits at $x = -0.372$, close to the $A$-anchored basin predicted by the pair potential's own shape, and the dimer lands exactly on the midpoint the problem's symmetry puts the saddle at. The two barrier estimates, 0.449267 directly and 0.449099 along the three-point PAFI path, agree to the third digit: the small difference is the spline path's own resolution (`howto/pafi-ohtst-checking-convergence` covers checking a path's resolution this way), not anything specific to using a real potential instead of an analytic one.


# Next
