# AGENTS.md

Instructions for contributors working in `rgsaddle`.

## Repository layout

- `src/`: Rust library crate (`rgsaddle`).
  - `src/lib.rs`: crate root, module declarations, and top-level re-exports.
  - `src/config.rs`: `with_setters!` macro for chainable `with_*` builders.
  - `src/band.rs`, `src/band/`: NEB, climbing image, OCI-NEB refinement (`min_mode.rs`), simplified string (`string.rs`), growing string (`growing_string.rs`), free-end adaptive NEB (`free_end.rs`), and Zoom-NEB (`zoom.rs`).
  - `src/autoneb.rs`: AutoNEB window insertion and spring scaling.
  - `src/ocineb.rs`: standalone OCI-NEB session with published hand-off controls.
  - `src/minmode.rs`, `src/indexk.rs`, `src/kappa.rs`, `src/seed_direction.rs`: dimer, Lanczos, LOBPCG, block-Lanczos index-k searches, basin-constrained dimer, and soft-mode direction seeding.
  - `src/art.rs`: phased Activation-Relaxation Technique nouveau (`ArtSession`).
  - `src/nichols.rs`, `src/prfo.rs`, `src/prfo/sella.rs`, `src/prfo_restricted.rs`, `src/rfo.rs`, `src/qn.rs`: Index-1 Newton, Nichols shift, RFO, P-RFO, restricted-step P-RFO, and quasi-Newton updates.
  - `src/sella_min.rs`, `src/sella_saddle.rs`, `src/pes.rs`, `src/pes_internal.rs`, `src/constraints.rs`, `src/internal.rs`, `src/vocn.rs`, `src/cell_log.rs`, `src/restricted.rs`, `src/geom.rs`, `src/force_match.rs`, `src/samd.rs`: Sella order-0 and order-1 sessions, internal coordinates, equality constraints, cell log-deformation chart, restricted steps, force-matched Hessians, and SAMD thermostat.
  - `src/irc.rs`: intrinsic reaction coordinate (`SellaSphere`, `Gs2`, `Morokuma`).
  - `src/line_neb.rs`: fixed-energy line-integral NEB (LI-NEB), its trapezoidal action and action force, and the energy-sliced seed from a saddle.
  - `src/idpp.rs`, `src/afir.rs`, `src/connect.rs`, `src/sqnm.rs`, `src/reparam.rs`, `src/rtr.rs`: IDPP and S-IDPP path initialization, AFIR fragment pull, COMPASS and SQNM double-ended connect, equal-arc-length reparametrization, and projected band RTR.
  - `src/solid_dimer.rs`, `src/solid_state.rs`, `src/ssneb.rs`, `src/strain_rate.rs`, `src/elastic.rs`, `src/field.rs`, `src/defect_pair.rs`, `src/pair_network.rs`, `src/occupancy.rs`, `src/relax.rs`: solid-state dimer, generalized solid-state NEB, elastic dipoles/multipoles, Kelvin/FMM fields, defect-pair hop rates, and crystal pair networks.
  - `src/pafi.rs`, `src/bennett.rs`, `src/ohtst/`, `src/adaptive_ohtst.rs`, `src/projector.rs`, `src/region.rs`, `src/lammps.rs`: PAFI free-energy integration, Bennett-Chandler transmission, optimal hyperplanar TST (OH-TST), adaptive OH-TST strings, symmetry-unique hard regions, and the opt-in LAMMPS sampler (`lammps` feature).
  - `src/sparse_hessian.rs`, `src/eigensolve.rs`, `src/linalg.rs`, `src/gpu.rs`: Curtis-Powell-Reid sparse Hessian recovery by distance-2 colouring, Rayleigh-Ritz eigensolvers, numerical Hessian-vector products, and DLPK device policy.
  - `src/stencil/`: unified point stencil (`Stencil`, `Chain`, `Mesh`, `Triad`).
  - `src/capi.rs`, `src/capi/`, `src/capi_hyperplane.rs`, `src/capi_kappa.rs`: C ABI entry points (`capi` feature).
  - `src/io.rs`: CON and molecular frame I/O (`readcon`, `readcon-db`, `chemfiles` features).
- `include/`: hand-written C header (`include/rgsaddle.h`) and C++ RAII wrappers (`include/rgsaddle/session.hpp`, `include/rgsaddle/stencil.hpp`).
- `bind/` and `python/rgsaddle/`: nanobind C++ extension (`bind/ext.cpp`), pure-Python package (`__init__.py`, `ase.py`), and generated stub (`_core.pyi`).
- `docs/`: Org-mode narrative and reference sources (`docs/orgmode/`), Emacs Org-to-rST exporter (`docs/export.el`), Sphinx configuration (`docs/source/conf.py`, `docs/source/references.bib`), C symbol renderer (`docs/render_c_symbols.py`), LLM context renderer (`docs/render_llms.py`), and coverage/hygiene checks (`docs/check_c_symbols.py`, `docs/check_paths.py`).
- `tests/`, `examples/`, `validation/`, `formal/`, `ci/`: Rust, C, C++, and Python tests, runnable examples, symbolic/Sollya validations, Lean 4 formal proofs, and CI scripts.

## Build and test

Build the C ABI library before running the test suite so the C and C++ smoke tests can link `target/debug/librgsaddle.so`:

```sh
cargo build --locked --features capi
cargo test --locked --features capi
cargo clippy --locked --features capi -- -D warnings
cargo fmt --check
```

Check optional feature sets and MSRV (`rust-version` in `Cargo.toml`, currently `1.88`):

```sh
cargo test --locked --features capi,ohtst
cargo test --locked --features readcon,readcon-db
cargo clippy --locked --all-targets --features capi,verify -- -D warnings
python3 ci/check_dependency_pins.py
cargo check --manifest-path ci/consumer/Cargo.toml --target-dir target
```

Every git dependency in `Cargo.toml` and `ci/consumer/Cargo.toml` must pin a full 40-character `rev` alongside an exact `version = "=x.y.z"` specifier so `ci/check_dependency_pins.py` resolves a single copy.

For the Python bindings, build against the pinned `nanobind==3.1.0` in `pyproject.toml`. Never hand-edit `python/rgsaddle/_core.pyi`; regenerate it with `python3 scripts/regen_stub.py`:

```sh
python3 -m pip install 'nanobind==3.1.0' 'scikit-build-core>=1.1'
python3 -m pip install --no-build-isolation -e .
python3 scripts/regen_stub.py --check
python3 -m pytest tests/python
```

## Documentation build and checks

Narrative and reference pages live in `docs/orgmode/*.org`. Do not hand-edit exported `.rst` files under `docs/source/`, `docs/orgmode/reference/c-symbols.org`, `llms.txt`, or `llms-full.txt`; regenerate them from the sources:

```sh
python3 docs/render_c_symbols.py
python3 docs/render_llms.py
python3 docs/check_c_symbols.py
python3 docs/check_paths.py
emacs --batch -l docs/export.el
sphinx-build -b html -n -W --keep-going docs/source docs/build
```

Verify Rust API documentation across all CI feature sets with `-D warnings` and `-D missing_docs`:

```sh
cargo rustdoc --locked --features capi -- -D missing_docs
cargo rustdoc --locked --features readcon,readcon-db -- -D missing_docs
ci/check_rustdoc.sh
cargo test --locked --doc --features capi
```

## C ABI minor version rule

`include/rgsaddle.h` defines `RGSADDLE_ABI_MAJOR` and `RGSADDLE_ABI_MINOR`. Every wire struct opens with `rgsaddle_version_t` and `uint64_t flags`. When adding new C ABI symbols, flags, or wire structs on a branch off `develop`, set `RGSADDLE_ABI_MINOR` to `develop`'s `RGSADDLE_ABI_MINOR` plus one (never skip or invent higher numbers, and never break existing struct layouts or symbol signatures within the same major version). Update `src/capi.rs` and any version assertions in `tests/` to match.

## Configuration struct convention (`#[non_exhaustive]`, `Default`, `with_*`)

Every public configuration, tuning, or options struct must:

1. Carry `#[non_exhaustive]` so adding fields later is not a breaking change.
2. Implement `Default` (or provide a documented canonical constructor when required inputs have no sensible default) with published, unit-agnostic defaults.
3. Provide chainable `#[must_use]` `with_*` builder methods via the `with_setters!` macro in `src/config.rs` (where `val` fields take `T` and `opt` fields take `impl Into<Option<T>>`).

## General and system-agnostic methods

All optimization, saddle-search, path-integration, and rate-theory methods in `rgsaddle` must remain general, unit-agnostic, and system-agnostic:

- Never hardcode chemical elements, pair styles, lattice constants, or material-specific parameters (such as copper EAM tables or Lennard-Jones parameters) inside library modules.
- Surfaces communicate through general traits (`PointSurface`, `BandSurface`, `HyperplaneSampler`, `HyperplaneDynamics`) in the caller's energy, length, mass, and time units.
- Specific potentials, materials, and benchmark systems belong only in `tests/`, `examples/`, `bench/`, or `validation/`.

## Commit message and prose rules

- Follow conventional commits (`feat`, `fix`, `chore`, `docs`, `refactor`, `test`, `bench`, `ci`, `style`).
- Keep the commit subject line at most 72 characters.
- Wrap commit message bodies at 72 characters.
- Put code identifiers, file paths, types, functions, and flags in backticks (`like_this`).
- Never use dashes (double hyphens, en-dash, or em-dash) as punctuation in commit messages or documentation prose; use colons, commas, parentheses, or separate sentences instead.
- Never include pull request numbers, commit SHAs, or ticket IDs in commit messages or documentation.
- Never use `!` after the commit type/scope or frame changes as breaking changes.
- Never add trailers of any kind.
- Write commit messages from a file using `git commit -F <file> --cleanup=whitespace`.
