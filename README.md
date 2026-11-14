# rgsaddle

rgsaddle finds saddle points on a potential energy surface, the transition states between two minima.
It is a Rust library with a C interface and a Python package, and its main searches are:

- a nudged elastic band (NEB) between two endpoints, with a climbing image;
- a minimum-mode search from one point, by the dimer method or by Lanczos iteration;
- an index-1 search on a dense Hessian by restricted-step partitioned rational function optimization (RFO).

By default the band also refines its climbing image along the lowest curvature mode.
That is the off-path climbing image NEB (OCI-NEB) of [Goswami, Gunde, and Jónsson (2026)](https://doi.org/10.3389/fchem.2026.1807063).
Each session takes one step of an [rgmin](https://github.com/OmniPotentRPC/rgmin) optimizer and reports.
The optimizer is the fast inertial relaxation engine (FIRE) by default, or limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS).
The host program owns the loop.

Docs: <https://omnipotentrpc.github.io/rgsaddle/>

Issues and questions: <https://github.com/OmniPotentRPC/rgsaddle/issues>.
The [contributing page][contributing] lists prerequisites, contact and acknowledgments.

[contributing]: https://omnipotentrpc.github.io/rgsaddle/contributing/index.html


## Install

<!-- docs-example: skip "edits the Cargo.toml of the host crate; index.org runs cargo add in a new crate" -->
```sh
cargo add rgsaddle
```

The [Python](#python) section below installs the Python package.
The [C interface guide](https://omnipotentrpc.github.io/rgsaddle/howto/c-abi.html) covers the C library.

## Example

This program relaxes a band of nine images on the double well V = (x² − 1)² + 2y² + 2z².
The surface has minima at (±1, 0, 0) and its saddle at the origin.
It needs `cargo add rgsaddle ndarray@0.17`.

```rust
use ndarray::{Array1, Array2, ArrayView2};
use rgsaddle::{BandConfig, BandSession, BandStatus, BandSurface, SaddleError};

/// V = (x² − 1)² + 2y² + 2z²: minima at (±1, 0, 0), saddle at the origin.
struct DoubleWell;

impl BandSurface for DoubleWell {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (i, r) in positions.outer_iter().enumerate() {
            let (x, y, z) = (r[0], r[1], r[2]);
            energies[i] = (x * x - 1.0).powi(2) + 2.0 * y * y + 2.0 * z * z;
            gradients[(i, 0)] = 4.0 * x * (x * x - 1.0);
            gradients[(i, 1)] = 4.0 * y;
            gradients[(i, 2)] = 4.0 * z;
        }
        Ok(())
    }
}

fn main() -> Result<(), SaddleError> {
    // Nine images from (-1, 0, 0) to (1, 0, 0), bowed off the axis in y.
    let initial = Array2::from_shape_fn((9, 3), |(i, k)| {
        let t = i as f64 / 8.0;
        [2.0 * t - 1.0, 0.3 * (std::f64::consts::PI * t).sin(), 0.0][k]
    });
    let config = BandConfig { force_tol: 1e-3, max_move: 0.1, ..BandConfig::default() };
    let mut session = BandSession::new(config, initial)?;
    let mut report = session.step(&DoubleWell)?;
    while report.status == BandStatus::Running {
        report = session.step(&DoubleWell)?;
    }
    let ci = report.ci_index.expect("a climbing image");
    let saddle = session.positions().row(ci).to_owned();
    println!("{:?} after {} steps, max force {:.1e}", report.status, report.iteration, report.max_force);
    println!("climbing image {ci} at ({:.4}, {:.4}, {:.4})", saddle[0], saddle[1], saddle[2]);
    Ok(())
}
```

With rgsaddle 0.4.0, `cargo run` prints:

<!-- docs-example: output -->
```text
Converged after 8 steps, max force 2.8e-4
climbing image 4 at (0.0000, 0.0001, 0.0000)
```

## Learn more

- [Getting started](https://omnipotentrpc.github.io/rgsaddle/getting-started.html) builds a band and steps it from a host loop.
- The tutorials find a saddle [with a band](https://omnipotentrpc.github.io/rgsaddle/tutorials/band.html), [with the dimer](https://omnipotentrpc.github.io/rgsaddle/tutorials/dimer.html) and [with Sella's step](https://omnipotentrpc.github.io/rgsaddle/tutorials/sella.html), and [follow the intrinsic reaction coordinate (IRC)](https://omnipotentrpc.github.io/rgsaddle/tutorials/quickstart.html) down from a first-order saddle.
- [How-to guides](https://omnipotentrpc.github.io/rgsaddle/howto.html) cover AutoNEB, the minimum-mode search, the C interface and Python.
- [The API documentation](https://docs.rs/rgsaddle) on docs.rs lists every public item.

## Design notes

The band and the minimum-mode search take one `rgmin` `Solver` step
on a force this crate assembled. `BandSession::step` assembles the
NEB force and reports. The sessions have no run-to-completion
contract. The host owns the loop and runs its own policy between
steps. `run` is a loop over `step`.

`Index1Session` is the index-1 search on a dense Hessian. The
displacement is the restricted-step partitioned RFO of Baker
([J. Comput. Chem. 7, 385 (1986)](https://doi.org/10.1002/jcc.540070402)).
It maximizes along the lowest mode, minimizes along the rest, and
solves the scaling that keeps the Cartesian step inside the trust
sphere. Powell and Bofill update the Hessian. `nichols_step` is the
i-PI level shift. `rgsaddle_prfo_step` is the C entry for the
partition.

The force assembly ports eOn's NEB with the same branches. The
tangents are the simple tangent of Mills, Jónsson and Schenter and the
improved tangent of
[Henkelman and Jónsson (2000)](https://doi.org/10.1063/1.1323224).
The springs are uniform, energy-weighted, or Onsager-Machlup. The
projections are the plain elastic band, the NEB, and the doubly nudged
band. The climbing image of
[Henkelman, Uberuaga and Jónsson (2000)](https://doi.org/10.1063/1.1329672)
arms by eOn's trigger rule.

`AutoNeb` grows a short band. It inserts an image where the energy
gap or the segment length is worst, relaxes a local window, and
climbs. With more than one thread it keeps one thread pool, and it
stores each block at its image index after every block has finished.
The tuning is `BandConfig::autoneb`. A weighted spring supplied by the
host is refused.

Positions are unwrapped Cartesian. A `Cell` (rows are lattice
vectors, triclinic allowed) supplies minimum-image differences for the
band mechanics. Neighbour lists belong to the surface, for example
through [vesin](https://github.com/Luthaf/vesin), not to this crate.

A warm band step costs one evaluation of the interior images. The
session caches the endpoint energies and its last evaluation. A host
that hands the band back unchanged (`set_positions`, wrapped into its
cell or not) or restarts the optimizer (`restart`) pays no extra
evaluation. `evaluation` returns the energies, gradients, and
projected forces at the current band, so the host need not compute
them again. `BandSession::reset` is the model-update boundary: it
drops the optimizer history and every cached surface value. FIRE is
the default stepper. L-BFGS steps the band under rgmin's
`Accept::Step`: one evaluation, and no energy test on the
non-conservative band force.

`MinModeSession` lives across a host search. `set_position` moves it,
optionally with the host's gradient, and `estimate_mode` refreshes the
lowest mode without a translation. The dimer rotates by the modified
Newton step with an extrapolated gradient, one evaluation per
rotation, and keeps its Polak-Ribiere rotational history across
translations. Lanczos stops on the Ritz residual.
`FiniteDifference::Central` gives central-difference Hessian actions.
`rotation_angle_tol` stops a dimer rotation whose optimal angle falls
under it: five degrees by default, and zero turns the test off. A
curvature that is not negative takes one full per-atom `max_move`
along the mode. A mode turn inside 45 degrees keeps the translator
memory. Both sessions refuse line-searched methods, because the energy
beside a projected or reflected force is not its potential.

## Validation

`validation/` has the sympy and sollya checks of the rotation
formulas, the Lanczos residual bound, and the finite-difference
curvature error terms
(`uv run --no-project --with sympy python validation/<script>.py`,
`sollya validation/fd_curvature.sollya`).
`validation/bain_lj_reference.py` scans the relaxed Bain path (bcc to
fcc) of a switched Lennard-Jones crystal. It is the reference for
`tests/c/solid_bain_lj.c`, which runs the solid-state band on that
crystal through `rgsaddle_band_create_solid`.

## Python

The package is a nanobind extension. `import rgsaddle` does not import
the Atomic Simulation Environment (ASE).

<!-- docs-example: skip "builds and installs the package; the docs-examples job runs pip install . before the Python blocks" -->
```sh
pip install .
pip install 'rgsaddle[ase]'
```

`ForceGate`, `HessUpdate`, `IrcKind`, `MinModeKind`, `EigenDevice`,
`ExpandKind`, `CellChart` and `RestrictedKind` are `IntEnum` controls.
`NumericalError`, `last_error` and `raise_status` map
`rgsaddle_status_t`. `rgsaddle.ase.NEB` takes the `images`, `k`,
`climb`, `method` and `remove_rotation_and_translation` arguments of
`ase.mep.NEB`, and an ASE optimizer drives it as an NEB or a CI-NEB. `rgsaddle.ase.MinModeAtoms` is the dimer or Lanczos
force for an ASE optimizer, and `rgsaddle.ase.Sella` is an order-1
optimizer with ASE's `run(fmax, steps)`. `rgsaddle.ase.SurfaceAdapter`
evaluates a list of ASE `Atoms` as request rows `(ids, x)`.
