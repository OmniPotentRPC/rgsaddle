# Changelog

All notable changes to rgsaddle since 0.3.2 are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - 2026-11-19

### Added

- C ABI minor 74 adds standard atomic mass lookup by atomic number or symbol and mass weighted finite difference Vineyard prefactors through `rgsaddle_atomic_mass`, `rgsaddle_atomic_mass_by_symbol`, `rgsaddle_vineyard_config_default`, and `rgsaddle_vineyard_fd`. The new wire structures are `rgsaddle_vineyard_config_t` and `rgsaddle_vineyard_report_t`.
- `ThermodynamicDerivative` (`potential.full_minus_harmonic`) is a generic lambda thermodynamic-integration observable, with harmonic Taylor reference helpers, `ThermodynamicSurfaceSampler`, `ThermodynamicSampler`, a LAMMPS thermodynamic-reference configuration, and the C symbol `rgsaddle_thermodynamic_derivative_observable`.
- An adaptive OH-TST hyperplane string shares constrained samples, runs a correlated pilot whose stride doubles until the autocorrelation time fits fourteen records, relaxes at production resolution, and reports a finite-temperature profile with measured geometry and quadrature errors. `TargetMet`, `SampleCap`, and `Unresolved` name the combined-error outcomes. Correlation-width batching is shared by every engine-backed hyperplane sample. Symmetry multiplicity is applied with `with_region` once inside the estimator.
- `ArtNConfig` adds the opt-in ART nouveau basin-exit and activation protocol to `MinModeSession` and the point builder. C ABI minor 43 adds `rgsaddle_artn_config_t`, its default initializer, and the live-session setter.
- `ArtSession` runs ART nouveau as eight persisted phases and lists every surface evaluation on `ArtSession::force_calls`. `ArtStatus::Reverted` is a quench pair that falls in one basin.
- `SymmetryUniqueRegion::from_crystal_operations` lifts crystal operations and reactant or saddle geometries into a hard region on one hop's dividing plane. Path multiplicity and in-plane region degeneracy stay separate.
- Hyperplane optimization, reversible work, unconstrained shooting, restricted Hessian modes, exact Gaussian sampling, and fixed-plane thermodynamic integration share `HyperplaneSampler` and `HyperplaneDynamics`. Samplers consume a generic hard `Region`.
- The fixed-energy instanton band exposes its mass-weighted optimization, imaginary-time replicas, rates, fluctuation index and evaluation counts through Rust, C and Python.
- The free-end band uses `rgmin = "=0.7.0"` for level-set constraints. Both Rust manifests pin the matching source revision.
- `BandConfig::with_free_end` and `FeaNebSession` add the free-end adaptive nudged elastic band. One terminal image moves on a target-energy level set, and fixed endpoints stay the default. C ABI minor 45 adds `rgsaddle_band_set_free_end`, `rgsaddle_band_free_end_residual`, and `rgsaddle_band_free_end_cuts`.
- `BandConfig::with_string` and `BandSession::set_string` add the simplified string method beside NEB, on a cubic spline or a polyline, with an optional climbing image. C ABI minor 44 adds `RGSADDLE_BAND_STRING` and `rgsaddle_band_set_string`. Python adds `rgsaddle.Band(..., string=True)` and `rgsaddle.ase.SimplifiedString`.
- `GrowingStringSession` implements the growing string method and hands a joined path to the simplified string. C ABI minor 44 adds `rgsaddle_growing_string_*`. Python adds `rgsaddle.GrowingString` and `rgsaddle.ase.GrowingString`.
- `SaddleIndexCheck` certifies the saddle index of a converged band climbing image, min-mode search, or OCI-NEB climbing image. `saddle_index` reports the result, and `SaddleIndexCheck::escape` can leave an index of two or more. The C ABI gained this at minor 40, and Python exposes `saddle_index` on the session report, `MinMode`, and the ASE helpers.
- `pypi.yml` publishes each `v*` tag to PyPI through trusted publishing, with wheels for Linux, macOS, and Windows, and checks that the tag matches `Cargo.toml` and `pyproject.toml`.

### Changed

- A converged band, min-mode, or OCI-NEB step certifies the saddle index of its point by default. The check's surface evaluations count in that step's `surface_rows` or `evaluations`, and every step still reaches the same positions. On the slab of `tests/band_fixed_atoms.rs`, FIRE at `force_tol` 1e-3 takes 141 force rows with the check and 81 without it. `with_saddle_check(None)` turns the check off, and the `fixed_slab_precon`, `minmode_trust`, and `minmode_calls` examples do so to keep their recorded counts. `BandReport`, `MinModeReport`, and `OcinebReport` include `saddle_index`. A C config stamped with ABI minor 40 or newer certifies by default, and one stamped earlier does not. Python's `MinMode` certifies by default.

### Fixed

- A Lanczos min-mode estimate stops on the same torque window as a classic dimer rotation. `MinModeSession::set_torque_window` and `rgsaddle_minmode_set_torque_window` apply to Lanczos sessions. C ABI minor 41.
- `rgsaddle_indexk_step` copies strided mode columns into a contiguous buffer before the C Hessian callback.
- Lambda thermodynamic integration passes the harmonic reference geometry into the LAMMPS mixing hook and evaluates `V_full - V_harm` in the sampler frame.
- `kabsch_rotation` iterates over slices, which clears the Rust 1.88 `needless_range_loop` lints in `src/art.rs`.
- The checked-in Python stub for `hyperplane_thermodynamic_integration` matches the binding.
- The crate overview links the optional `ohtst` module only when that feature is enabled, and the examples job builds `ohtst_skewed_well` with `--features ohtst`.
- The verify workflow's symbolic step skips `validation/benchmark/`, and its `libm-diff` job downloads only the `libm-*` artifacts with a ten-minute timeout.
- `connect` checks endpoint lengths, finiteness, and force tolerance on entry.
- `rayleigh_ritz_iter` returns the exact Ritz pairs once the basis is square, including a 1 by 1 matrix.
- `rgsaddle.ase.GrowingString.run` restores the reactant and product geometries after every step. `join_tol` compares the frontier gap after the step's node additions, `rgsaddle_growing_string_nodes_added` reports nodes grown, and `rgsaddle_growing_string_create` treats -1.0 as no early join.

### Documentation

- `ohtst::work` links `sample_plane` by its crate path.
- The rustdoc of the band, stencil, dimer, P-RFO, IRC, IDPP, OH-TST, and index-k modules states units, shapes, errors, and the method papers. Windows, force tolerances, and eigensolver sizes whose docs disagreed with the code are corrected.

### Developer

- The pt-island proxy tabulates campaign runs at matched horizons and records a SHA-256 of each run's searches. `modes.py` replays the first catalog climbs, and `dimer-lbfgs-tight.toml` is the dimer L-BFGS template with `rotation_tol` 1e-6.

## [0.5.0] - 2026-11-17

### Breaking

- Every public configuration struct outside the C ABI, except `DirectSolver`, is `#[non_exhaustive]`.
  A struct literal of one, including one that ends in `..Default::default()`, or an exhaustive pattern of one no longer compiles outside this crate.
  Each config has a `#[must_use]` `with_*` setter for every public field, so `BandConfig { force_tol: 1e-3, ..BandConfig::default() }` becomes `BandConfig::default().with_force_tol(1e-3)`.
  The fields stay public, no existing default changes, and the `#[repr(C)]` structs of the C ABI keep their layout.
- The crate requires rgmin 0.6.0, pinned by full git rev to its v0.6.0 tag.
  A host that passes rgmin types, such as the `Method` of `BandConfig::with_method`, takes them from that release, whose `Method` and `LineSearch` struct variants are non-exhaustive and built through constructors such as `Method::fire(FireKind::V2)`.
  FIRE 2.0 follows Guénolé et al. (2020), Algorithm 2, so the FIRE band, dimer, Lanczos and AutoNEB trajectories move, and a FIRE band step that backs up costs two evaluations of the interior images.

### Added

- `BandConfig::with_fixed_atoms` and `BandSession::set_fixed_atoms` hold a per-atom mask fixed in every image of a band.
  Fixed atoms never move under FIRE, L-BFGS, quick-min, ODE12r or the trust-region band, and are left out of the tangents, the springs, the climbing image and the min-mode refinement.
  C ABI minor 39 adds `rgsaddle_band_set_fixed_atoms` and `rgsaddle_band_project_fixed`, the C++ `Band` gains `set_fixed_atoms`, Python's `project_band` takes `fixed`, and `rgsaddle.ase.NEB` reads `FixAtoms`.
- `MinModeSession::set_rotation_stop_length` adds the curvature-aware rotation stop of Kästner and Sherwood, off by default.
  C ABI minor 38 adds `rgsaddle_minmode_set_rotation_stop_length`, and the C++ and Python bindings expose it.
- `Reparametrization::NotAKnot` places band images at equal arc length on a not-a-knot cubic spline; the polyline placement stays the default.
  `RtrConfig::reparametrization` selects it for the trust-region band and `Mesh::RemeshSpline` for a stencil chain, `rgsaddle::reparametrize` applies either placement to a band, and C ABI minor 38 adds `rgsaddle_reparametrize`.
- `SolidDimer` is the minimum-mode search on atoms plus cell strain, and `SolidNeb` the climbing-image band on the same chart.
  `IndexCheck` checks that a converged solid dimer is index 1, and `StressCheck` compares a surface's Cauchy stress with finite differences of its energy before the first free-cell step.
- `StrainResponse` estimates the barrier of one hop under strain from the elastic dipoles of its minimum and saddle, and asks for a search (`NeedsSearch`) when the estimate leaves its bounds.
  `ElasticDipole::from_stress`, `ElasticMultipole`, `Kelvin`, `DirectSolver` and `FmmSolver` give the dipoles and the elastic fields of defects.
- `rgsaddle::ohtst` is optimal hyperplanar transition state theory: a hyperplane seeded from a converged dimer or Lanczos session, sampled by projected BAOAB or GHMC, optimized by rgmin's Riemannian trust region, and turned into a rate with its standard error.
  `transmission_coefficient` and `plane_transmission` give the transmission coefficient, `reversible_work_along` the reversible work along a path of planes, and `anharmonic_correction` the first-order correction to harmonic transition state theory.
- `relax_to_tolerance` relaxes a configuration in a fixed cell until the largest per-atom force is below `RelaxConfig::force_tol`; reaching the step cap is an error, never a geometry.
  `site_occupancy` is the Wigner–Seitz defect analysis.
- `defect_pair_rates` searches a caller-supplied hop catalogue with the solid-state dimer and rates each hop by PAFI with a Bennett–Chandler factor.
  `PairNetwork` builds the symmetry-resolved kinetic network of a mobile defect next to a fixed one, and `matched_capture_radius` closes its chain against the infinite crystal.
  The Mishin copper vacancy–interstitial volumes are in `benchmarks/defect_pair_*`, and the direct-MD runs they are compared with in `validation/direct_md_*`.
- A top-level `meson.build` provides the C and C++ headers to meson projects as a wrap subproject or through `rgsaddle-headers.pc`.

### Changed

- The `msrv` CI job runs clippy and the tests on the `rust-version` toolchain, 1.88, and the `readcon` job covers both readcon feature sets.
- The API documentation builds with every rustdoc warning denied, and `ci/check_dependency_pins.py` fails when a git dependency is not pinned by full rev at the exact version of that rev.
- The `meson-consumer` CI job builds `ci/meson-consumer` against the headers, as a wrap-git subproject and from an installed `rgsaddle-headers.pc`, and steps a climbing band through the cdylib.
- Pushes to `develop` run the CI and docs workflows.

### Fixed

- The min-mode refinement's climb model counts every periodic image of a pair within the cutoff, so a cell shorter than twice the cutoff keeps the lattice's symmetry.
  A cell more than twice the cutoff high along every wrapped axis steps as before.
- rgmin 0.6.0 judges an ODE12r step that no cap or bound shortened at its own length, and the recorded ODE12r trajectories moved.
  ODE12r with the exponential metric takes 18.33 force evaluations per image on the 5-image Cu vacancy band, against 19.33 with rgmin 0.5.2.
- linkcell and rgmin are git dependencies pinned by full rev with the exact version at that rev, so a dependent that pins either by the same rev locks one copy.
- `ConImage::forces` is documented as forces, the `fx fy fz` columns, in place of the gradient its docs named.
  Every field of `MolecularFrame`, `ConSpecies` and `ConImage` documents its length, atom order, unit and ownership.

## [0.4.0] - 2026-11-13

### Breaking

- The reaction-coordinate kinds now take the steps they name.
  `IrcKind::Gs2` is the Gonzalez and Schlegel step: the energy minimum on the mass-weighted sphere of radius `dx / 2` about a pivot `dx / 2` down the gradient.
  The first pivot lies `dx / 2` from the saddle along the launch mode.
  Before this release that kind took the minimum on the sphere of radius `dx` centred on the last point.
  That step is now `IrcKind::SellaSphere`, and it is the default, so a session built from `IrcConfig::default()` keeps the step it took in 0.3.2.
  A session that names `IrcKind::Gs2` follows the new step.
  A `match` on `IrcKind` needs an arm for `SellaSphere`.
- `IrcConfig` no longer has `max_move` or `method`.
  Those fields configured a stepper the IRC session never used.
  A struct literal that names either field does not compile.
  The C struct `rgsaddle_irc_config_t` still has `max_move`, so the layout is unchanged, and create does not read it.
- A C config whose version minor is 34 or older still runs the old sphere for every kind except Morokuma.
  Minor 34 is the curvature probe and does not change the IRC kinds.
  From minor 35, kind 0 is the new `Gs2` step, kind 2 is `RGSADDLE_IRC_SELLA_SPHERE`, and create returns NULL for any other kind.
- `SpringKind` gained `OnsagerMachlupLive` and `EnergyWeighted`.
  An exhaustive `match` has to name them.
- C spring ordinal 2 (`RGSADDLE_SPRING_ONSAGER_MACHLUP`) now builds the live Onsager-Machlup spring from the band forces.
  In 0.3.2 that ordinal held every displacement at zero, so a band that uses it steps differently, and a non-finite `spring_k` makes `rgsaddle_band_create` return NULL.
  `RGSADDLE_ABI_MINOR` is 33 for that change. The layout and the symbols are unchanged.
- The Sella small-system atom ceiling is on by default.
  A Cartesian saddle search whose starting radius is one length unit or less steps differently from 0.3.2.
  Set the ceiling false, or call `rgsaddle_sella_saddle_set_small_system_atom_ceiling` with 0, for the previous step.
- A stalled FIRE translation on a stiff negative mode takes one short step along the reflected force and then stays on that step while the curvature remains steep.
  The probe is on by default.
  `set_fire_curvature(false)` is the integrator from 0.3.2.
  `RGSADDLE_ABI_MINOR` 34 names that default. No field was added.

### Added

- `IndexKSession` and `indexk_step` take one minimum-mode step in the k lowest modes.
  `k = 1` is the dimer or Lanczos step this crate already had.
  C ABI minor 37 adds `rgsaddle_indexk_create`, `rgsaddle_indexk_step`, the position, mode, and translation reads, and `rgsaddle_reflect_force`.
- `halve_raised_energy` and `trust_update` decide a halved step and a trust radius from energies the host already holds.
  Neither function evaluates a surface.
  C ABI minor 36 publishes both decisions, and the Python binding exposes the same names.
- Python drivers shaped like ASE: `rgsaddle.ase.NEB`, `MinModeAtoms`, and `Sella`.
  `rgsaddle_band_project` projects an evaluated band at ABI minor 35.
- `BandSession::set_image_threads` evaluates contiguous blocks of images on a pool the session keeps.
  One thread is the default. Zero threads is refused. Independent image rows keep the one-thread trajectory.
- `SpringKind::OnsagerMachlupLive` rebuilds the Onsager-Machlup displacements from the band forces on every evaluation.
  `sidpp_path` builds an initial band by sequential image-dependent pair potential.
  AutoNEB can place a new image with that potential.
  `SpringKind::EnergyWeighted` recomputes the energy-weighted spring constants from the band energies.
- Zoom-NEB, off by default. `rgsaddle_band_set_zoom` is ABI minor 31.
- Opt-in Sella Ritz pairs, and `rgsaddle_sella_saddle_set_symmetrized_ritz` at ABI minor 24.
- `rgsaddle_minmode_set_torque_window` sets the dimer torque window apart from `rotation_tol`. The header records that entry at minor 32.
- An airspeed-velocity suite times one dimer step and one connect call through the Python binding.

### Changed

- The crate requires rgmin 0.5.2.
  The band, minimum-mode, stencil, IDPP, and IRC solvers still use the 0.5.1 control defaults: no relative energy slack and no noise-floor stop.
- linkcell is pinned to tag v0.3.5, and minimage is pinned to 0.1.2.
- A P-RFO block wider than 32, and the climbing model's wide TS-BFGS update, use faer's self-adjoint eigensolver.
  Dimension 32 and below stays on cyclic Jacobi.
- The band and minimum-mode sessions reuse their workspaces.
  A warm band step allocates as often at 13 images as at 5.
  A longer dimer rotation or a larger Lanczos space allocates nothing more outside the surface.
  The steps are bitwise the same.
- `MinModeConfig::seed_curvature` can scale the first empty translation by `1/|λ|`.
  It is off by default.
- The OCI-NEB force-call benchmark is a regression test.
  It requires both methods to converge, the climbing images to agree with each other and with the analytic saddle, and OCI-NEB to use at most half as many force calls as CI-NEB.
- The C smoke tests find `librgsaddle.so` by walking up from the test binary to the profile directory.

### Fixed

- A band refuses inputs the sessions used to misread: an unresolved energy-weighted or live Onsager-Machlup spring, a non-finite minimum-mode start, and a two-image band.
- The Python NEB reports the highest interior energy, as ASE's NEB does.
- An endpoint at or above the highest interior image no longer pulls the climbing image up a slope into itself when its force toward the next image exceeds `force_tol`.
- A GS2 branch stops when the force is at or below `force_tol`, the step is past the kick, and the curvature along the launch mode is positive.
  Each inner trial computes one spectrum of its quasi-Newton model.
  A rejected trial still halves the radius, and a later accepted step doubles it back up to the requested radius.
- The IRC launch from a surface follows the lowest eigenvector of the mass-weighted Hessian.
  An inner trial puts the model on the sphere by bisection.
  Each sphere step hands its quasi-Newton pairs to rgmin at unit length, so a short step is not dropped by the curvature floor.
- The curvature seed applies only to the first translation that still has an empty memory.
- The Sella bibliography entry gives volume 18 and page 6974.

## [0.3.2] - 2026-10-25

- Builds on rgmin 0.5.1.
  A negative `maxmove` in that control means no step cap.
  The band and minimum-mode sessions over atoms keep the rgmin 0.5.0 behaviour.

[0.6.0]: https://github.com/OmniPotentRPC/rgsaddle/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/OmniPotentRPC/rgsaddle/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/OmniPotentRPC/rgsaddle/compare/v0.3.2...v0.4.0
[0.3.2]: https://github.com/OmniPotentRPC/rgsaddle/releases/tag/v0.3.2
