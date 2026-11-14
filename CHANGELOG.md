# Changelog

All notable changes to rgsaddle since 0.3.2 are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.4.0]: https://github.com/OmniPotentRPC/rgsaddle/compare/v0.3.2...v0.4.0
[0.3.2]: https://github.com/OmniPotentRPC/rgsaddle/releases/tag/v0.3.2
