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

With rgsaddle 0.3.2 from crates.io, `cargo run` prints:

```text
Converged after 8 steps, max force 2.8e-4
climbing image 4 at (0.0000, 0.0001, 0.0000)
```

## Learn more

- [Getting started](https://omnipotentrpc.github.io/rgsaddle/getting-started.html) builds a band and steps it from a host loop.
- [The tutorial](https://omnipotentrpc.github.io/rgsaddle/tutorials/quickstart.html) follows the intrinsic reaction coordinate (IRC) down from a first-order saddle.
- [How-to guides](https://omnipotentrpc.github.io/rgsaddle/howto.html) cover AutoNEB, the minimum-mode search, the C interface and Python.
- [The API documentation](https://docs.rs/rgsaddle) on docs.rs lists every public item.

## Design notes

The band and the minimum-mode search take one `rgmin` `Solver` step
on a force this crate assembled. `BandSession::step` assembles the
NEB force and reports. No run-to-completion contract exists; hosts
own the loop and interleave their policy between steps. `run` is a
convenience loop over `step`.

`Index1Session` is the index-1 search on a dense Hessian. The
displacement is Baker's restricted-step partitioned RFO: maximize
the lowest mode, minimize the rest, and solve the scaling so the
Cartesian step lies inside the trust sphere. Powell and Bofill
update the Hessian. `nichols_step` remains the i-PI level shift.
`rgsaddle_prfo_step` is the C entry for the partition.

Force assembly ports eOn's NEB mechanics with identical branch
structure: tangents (Mills-Jonsson-Schenter simple, Henkelman-Jonsson
improved), springs (uniform, energy-weighted, Onsager-Machlup),
projections (plain elastic band, NEB, doubly nudged), and the
climbing-image force with the eOn trigger rule.

`AutoNeb` grows a short band. It inserts an image where the energy
gap or the segment length is worst, relaxes a local window, and
climbs. Image evaluation keeps one thread pool; each block is stored
at its image index after every block has finished. The tuning sits on
`BandConfig::autoneb`. A weighted spring supplied by the host
is refused until zoom-NEB recomputes it.

Positions are unwrapped Cartesian. A `Cell` (rows are lattice
vectors, triclinic allowed) supplies minimum-image differences for the
band mechanics; neighbor lists, when a surface wants them, are
[vesin](https://github.com/Luthaf/vesin)'s job, not this crate's.

A warm band step costs one evaluation of the interior images. The
session caches the endpoint energies and its last evaluation, so a
host that hands the band back unchanged (`set_positions`, wrapped
into its cell or not) or restarts the optimizer (`restart`) pays no
extra evaluation, and `evaluation` returns the energies, gradients,
and projected forces at the current band so the host need not
recompute them. `BandSession::reset` is the model-update boundary: it
drops the optimizer history and every cached surface value.
L-BFGS steps the band under rgmin's `Accept::Step` (one evaluation,
no energy test on the non-conservative band force); FIRE is the
default.

`MinModeSession` is long-lived: `set_position` moves it (optionally
with the host's gradient), `estimate_mode` refreshes the lowest mode
without translating. The dimer rotates by the modified Newton step
with an extrapolated gradient (one evaluation per rotation) and keeps
its Polak-Ribiere rotational history across translations. Lanczos
stops on the Ritz residual; `FiniteDifference::Central` is available
for the actions, and `rotation_angle_tol` stops a dimer rotation whose
optimal angle falls under it (five degrees by default; zero disables
the test). A non-negative curvature takes one full per-atom `max_move`
along the mode, and a mode turn inside 45 degrees keeps the translator
memory. Both sessions refuse line-searched methods: the
oracle value beside a projected or inverted force is not its
potential.

## Validation

`validation/` has the sympy and sollya
checks of the rotation formulas, the Lanczos residual bound, and the
finite-difference curvature error terms
(`uv run --no-project --with sympy python validation/<script>.py`,
`sollya validation/fd_curvature.sollya`).
`validation/bain_lj_reference.py` scans the relaxed Bain path (bcc to
fcc) of a switched Lennard-Jones crystal, the reference for
`tests/c/solid_bain_lj.c`, which runs the solid-state band on it
through `rgsaddle_band_create_solid`.

## Python

The package is a nanobind extension. `import rgsaddle` does not import
the Atomic Simulation Environment (ASE).

```
pip install .
pip install 'rgsaddle[ase]'
```

`ForceGate`, `HessUpdate`, `IrcKind`, `MinModeKind`, `EigenDevice`,
`ExpandKind`, `CellChart` and `RestrictedKind` are `IntEnum` controls
with the same names and values as before. `NumericalError`,
`last_error` and `raise_status` map `rgsaddle_status_t`.
`rgsaddle.ase.SurfaceAdapter` evaluates a list of ASE `Atoms` as
request rows `(ids, x)`.
