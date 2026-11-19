# Changelog

All notable changes to rgsaddle since 0.3.2 are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - 2026-11-19

### Breaking

- Converged band, minimum-mode, and OCI-NEB sessions certify the saddle
  index by default. The certificate evaluations are included in each
  report, and `with_saddle_check(None)` restores the earlier cost.
  A C config stamped with ABI minor 40 or newer certifies by default.
- `BandReport`, `MinModeReport`, and `OcinebReport` add the public
  `saddle_index` field, so a struct literal or an exhaustive pattern of
  one no longer compiles outside this crate.

### Added

- `SaddleIndexCheck` certifies the lowest curvatures with block Lanczos
  and can escape a higher-index saddle along its extra unstable modes.
  Rust, C, C++, and Python expose the certificate.
- `GrowingStringSession` implements the growing string method, with a
  hand-off to the simplified string method on join, and
  `BandConfig::with_string` adds the simplified string method as an
  opt-in band method. C ABI minor 44 adds `rgsaddle_growing_string_*` and
  `rgsaddle_band_set_string`, and Python adds `rgsaddle.GrowingString`,
  `rgsaddle.ase.GrowingString`, and `rgsaddle.ase.SimplifiedString`.
- `ArtNConfig` adds the opt-in ART nouveau protocol to `MinModeSession`
  and the point builder, and `ArtSession` runs it as eight persisted
  phases. C ABI minor 43 adds `rgsaddle_artn_config_t`.
- An adaptive OH-TST hyperplane string shares constrained samples with
  the other hyperplane samplers and reports whether its combined error
  target was met, capped, or left unresolved.
- `SymmetryUniqueRegion::from_crystal_operations` constructs a hard
  symmetry-unique dividing-plane region from crystal operations while
  keeping region degeneracy separate from path multiplicity.
- Hyperplane optimization, reversible work, shooting, restricted modes,
  exact Gaussian sampling, and fixed-plane thermodynamic integration
  share `HyperplaneSampler` and `HyperplaneDynamics`. Fixed-plane
  thermodynamic integration is also available from C and Python.
- Lambda thermodynamic integration gains `ThermodynamicDerivative`,
  harmonic-reference samplers, LAMMPS configuration, and the
  `rgsaddle_thermodynamic_derivative_observable` C symbol.
- Tagged releases publish tested wheels and an sdist to PyPI.

### Changed

- `Stencil::band` and the C, C++, and Python stencil builders report the
  error `band presets are not implemented`, in place of text that named a
  development branch.
- The `rgmin` 0.6.0 dependency resolves from crates.io so downstream
  builds do not split the package by source.
- Lanczos minimum-mode estimation uses the same torque window as the
  classic dimer rotation. `set_torque_window` controls both methods.

### Fixed

- `connect` validates endpoint lengths, finiteness, step budget, and
  force tolerance on entry instead of panicking, and `rayleigh_ritz_iter`
  returns the exact Ritz pairs once its basis spans the matrix.
- `rgsaddle_indexk_step` copies strided mode columns before calling the
  C Hessian callback, and the growing string reports its added nodes
  through `rgsaddle_growing_string_nodes_added`.
- Lambda thermodynamic integration evaluates the full-minus-harmonic
  observable in the sampler frame and supplies the harmonic reference
  geometry to the LAMMPS mixing hook.
- The Python thermodynamic-integration stub, optional OH-TST rustdoc
  link, OH-TST example feature, symbolic validation selection, and
  libm artifact selection match their implementations.

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
