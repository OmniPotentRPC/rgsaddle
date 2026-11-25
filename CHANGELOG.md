# Changelog

All notable changes to rgsaddle since 0.3.2 are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.0] - 2026-11-25

### Added

- `ElasticConstants` and `AnisotropicGreen` evaluate point-defect interactions from a general stiffness tensor with Barnett's analytic Green-function derivative. `PeriodicElastic` applies a spherical periodic-image correction to pair landscapes and harmonic rate catalogues and reports angular and image-shell error estimates.
- `ElasticDipole::from_energy_strain_gradient` builds a defect dipole from the homogeneous-strain energy derivatives of the defect and bulk cells, beside the stress and force routes.
- `FixedEnergyTrajectory` splines a path in mass-weighted coordinates and integrates the motion on the inverted potential along it, with `half`, `period`, and `arc_at_time`. `PathGeometry::on_spline` reads the cubic path geometry. `Spline` is public with `SplineEnds::Natural` beside the not-a-knot ends string reparametrization keeps, and `Spline::sample` returns a `SplineSample`. A spent step or surface-call budget is `SaddleError::Invalid` with a message that starts `work budget exhausted`, so the error enum is unchanged.
- `LineNebConfig::with_endpoint_below_energy` selects the endpoint policy for a running fixed-energy LI-NEB session.
  `EndpointBelowEnergy::Restrained` is the default and retains the published restraint and zero action weight while an endpoint is below `E`.
  `EndpointBelowEnergy::Strict` stops with `SaddleError::Invalid`, so the error enum is unchanged.
  `LineNebEndpointViolation` reports the endpoint, `V - E`, and the one-based session step.
- `LineNebSession` optimizes the fixed-energy JWKB tunnelling path with the line-integral extension of the nudged elastic band, one host-visible quick-min step at a time. `fixed_energy_action` is the trapezoidal abbreviated action over every image and refuses an endpoint below the fixed energy; `fixed_energy_action_force` is its gradient. `LineNebStatus::Converged` needs the transverse action force, the interior spring force, and the endpoint force within tolerance, and the improved tangent falls back to the secant on an energy plateau. `fixed_energy_mep_seed` builds the starting band from a saddle and its unstable mode. `LineNebConfig` and `FixedEnergySeed` are non-exhaustive with `Default` and `with_*` setters. Ásgeirsson, Arnaldsson, and Jónsson, J. Chem. Phys. 148, 102334 (2018), doi:10.1063/1.5007180.
- `LinebConfig::with_gradient_end_tangents(true)` ends the orbit spline of a fixed-energy LI-NEB path along the surface gradient at each turning point. The not-a-knot ends remain the default. The option separates the two sources of ring force residual on a coarse band: it removes the turning-replica term |∇V| sin θ, with θ the angle between the spline end tangent and the gradient, and shows the remaining interpolation error inside the first and last intervals.
- `action_pair_blocks` reads the Hessian blocks of caller-chosen atom pairs from an exact `PointSurface::hessian_vector`, with no finite difference, and refuses a surface without a Hessian action.
- `anchored_barriers` takes the same inputs as `balanced_barriers` and keeps the forward barrier of the path: the reverse barrier is the forward one less the free-energy difference of the ends, the error is that of the forward barrier, and the drift is what the barrier from the last plane adds to that difference. It is the estimate to take when a path integrated again from its other end shows the drift past the dividing surface. `FreeEnergyProfile::anchored` takes the two barriers from a profile.
- `balanced_barriers` takes the forward and reverse barriers of one PAFI path and the free-energy difference of its ends, and treats the forward barrier and the reverse one plus that gap as two estimates of one barrier. `BalancedBarriers` holds their inverse-variance mean, the reverse barrier the gap below it, the drift the integral adds to the gap, and an error with half the drift in quadrature. `FreeEnergyProfile::balanced` takes the two barriers from a profile.
- `calibrate_surrogate` fits `x · β` to the residuals of a surrogate `ln k` against computed rates by weighted least squares, with weights `1/(σ² + τ²)` and the Paule–Mandel scatter `τ`, zero when the errors explain the residuals. `SurrogateCalibration` gives the `shift` of a rate left on the surrogate, its `shift_error`, its `spread` as one more draw from the same population, and `draw` of the coefficients within their covariance for error draws.
- `docs/orgmode/howto/pafi.org` is the guide to PAFI for a user with NEB images: what the surface must provide, how `ArcPath` is built from the images and why its hyperplanes follow the spline through them, every `PafiConfig` field with its default, how to choose the arcs, how to read `FreeEnergyProfile` and `HyperplaneAverage` including the block standard error, the `zero_temperature_barrier` check, what the sampler does about thermalization and the automated equilibration test (check `equilibration`, `tau` and `n_eff`), and the comparison with the reference code of Swinburne and Marinica.
  The module documentation of `rgsaddle::pafi` has a compiled example on an analytic two-dimensional surface, and the public items of the module are documented. `PafiConfig::max_rounds` is documented as the most passes, the first included, which is what it does.
- `sensitivity_cover` ranks classes by `|d result / d ln k|` and returns how many leading ones to compute so that the rest, all moved together by the spread of the rate model, shift the result by less than a tolerance; `SensitivityCover` holds the order, that count and the remaining shift after each rank. `FreeEnergyProfile::reverse_barrier` is `F(dividing) - F(last)` with its trapezoid error, so one PAFI path that ends in the product minimum gives both directions of a hop.
- Fixed-energy LI-NEB aligns the images of a free molecule to the first image by default and to their neighbours on request: `Masses::with_adjacent_alignment(true)` in Rust and `molecular` 2 in the C API. Neighbour alignment gives every chord its minimum mass-weighted length, as in Ásgeirsson, Arnaldsson and Jónsson, J. Chem. Phys. 148, 102334 (2018).
- Python exposes `rgsaddle.lineb_solve` and `LinebResult` for the fixed-energy LI-NEB path and optional harmonic quantum TST rate, with the same one-image surface trampoline as `connect`, shape checks on `(n_images, 3N)` positions and per-atom masses, and preserved calculator exceptions.
  `LinebResult.surface_calls` reports the number of geometries evaluated in each phase and their total. A path-only calculation reports zero prefactor evaluations.
- The `solid_prefactor` module builds dense atom-strain Hessians with `joint_hessian` and computes variable-cell Vineyard prefactors with `joint_vineyard_prefactor`, keeping the homogeneous strains and requiring one unstable saddle mode.
- The Rust LI-NEB continuation API accepts a callback after each completed energy level, allowing callers to record provider work including seed preparation.
- The zero-temperature defect-pair hop search (dimer, connectivity, climbing band, Hessian helpers) lives in `hop_search`. `search_hop` takes a `PairSetup`, `HopTemplate` and `HopSearchConfig` and returns a `HopSearchResult` for the rate half. `DefectPairConfig` stays whole; `HopSearchConfig::from` borrows the fields the search needs. Existing `rgsaddle::*` and `rgsaddle::defect_pair::*` paths keep resolving through re-exports.
- Tutorials for PAFI and OH-TST (`tutorials/pafi`, `tutorials/oh-tst`) take a newcomer from a checkout to a free-energy profile, a rate, and a transmission-corrected rate, in Rust and in Python, on the same analytic surface.
  How-tos cover wrapping an `rgpot-pair` potential as a `PointSurface` (`howto/rgpot`), restarting a PAFI or OH-TST run (`howto/pafi-ohtst-restarting`), parallel sampling (`howto/pafi-ohtst-parallel-sampling`), and checking convergence (`howto/pafi-ohtst-checking-convergence`).
  `explanation/pafi-ohtst` gives the equations and assumptions of harmonic TST, PAFI and OH-TST, and when to use each.
  `reference/pafi` and `reference/ohtst` list every public configuration and result type of both modules.
  Six new runnable examples back the code blocks these tutorials and how-tos include and run in CI alongside the existing ones.
- `DynamicNeb` is an opt-in on `BandConfig` (default off) that follows ASE 3.29 `DyNEB`: an interior image whose largest per-atom projected force is strictly below `fmax` keeps its last energy and force, stays in the spring and the tangent, and is evaluated again when a neighbour moves the projected force back over the threshold.
  `scale_fmax` scales that threshold by the Euclidean distance from the highest-energy image, which is never given a zero force.
  `rgsaddle.ase.DyNEB` is that band; `NEB` still refuses `dynamic_relaxation`, `fmax`, and `scale_fmax`.
  The C ABI adds `rgsaddle_band_set_dynamic` and `rgsaddle_dyneb_mask` (minor 65).
- `InternalTopology::from_geometry` builds redundant internals from atomic numbers and a geometry: bonds by Cordero covalent radius, angles, proper dihedrals, and dummy atoms on near-linear bends, after Hermes, Sargsyan, Najm, and Zador, J. Chem. Theory Comput. 18, 6974 (2022), doi:10.1021/acs.jctc.2c00395, with a rank and invariance check against the rigid quotient.
  `InternalPes::auto`, `SellaSaddleSession::on_auto_internal` and `SellaMinSession::on_auto_internal` rebuild that chart at the accepted geometry when its Wilson matrix goes singular, a bend nears linear, a centre goes planar or a back-transformation fails, carry the Hessian across, and restart the trust history; a chart given by hand is never rebuilt.
  A cell makes the construction periodic, and a hand chart can name lattice images with `Constraints::fix_bond_image`, `fix_angle_image` and `fix_dihedral_image`.
- `LinebSession` finds the optimal tunneling path at a fixed energy by the line integral nudged elastic band (LI-NEB) of Ásgeirsson, Arnaldsson and Jónsson, J. Chem. Phys. 148, 102334 (2018), on any `PointSurface` in mass-weighted coordinates.
  `LinebConfig::new` takes the two stopping tolerances, which have no default. `Masses::coordinates` leaves rigid motions alone and `Masses::molecule` removes them by quaternion alignment.
  `LinebPath` reports the action, the half period from the quadrature and from constrained dynamics, the instanton temperature of each, and `LinebCalls`, the force calls of the optimizer, the quadrature and the dynamics counted apart.
  `continue_from_mep` runs a ladder of energies, a `Continuation`, down from the top of a minimum energy path.
  `hqtst_rate` gives the ring-polymer harmonic quantum TST rate from Hessian-vector products at the distinct replicas and a deflated Lanczos log-determinant, in `(P/2 + 1)(1 + 2 dof)` force calls without an analytic product.
  The paper's interpolated finite-difference Hessian is `hqtst_rate_interpolated_hessian` behind the `lineb-oracle` feature.
  On H2 desorbing from rgpot's Cu(110) slab over RPC, 9 energies from 235 K to 63 K took 466.7 optimizer force calls per energy on average.
- `MD_SIA_TRACER=1` in `examples/langevin_nvt.rs` runs the free <100> dumbbell replicas of `MD_MSD=1` with the same seeds and records, for every lag between block ends, the squared displacements of the doubly occupied site, the dumbbell centre and its farther atom, the sample-to-sample jump sum, and the squared displacement summed over all atoms, N times the per-atom G_MSD of B. Zhang et al., Phys. Rev. Materials 6, 053605 (2022), Eq. (5). `MD_SIA_TRACER_NOTE=1` writes the tracer notes and `validation/direct_md_dynamics.md` from `validation/direct_md_sia_tracer_*.csv`.
  The Table 1 D_i is a tracer sum, not the diffusivity of the defect. Its source is their Ref. [12], B. Zhang et al., Phys. Rev. Materials 6, 053605 (2022), OSTI 1870695, which fits the per-atom G_MSD ` (1/N) Σ_i of its Eq. (5) (p. 4). The erratum, Phys. Rev. Materials 9, 069901(E) (2025), doi:10.1103/8hb2-86lm, calls that the self-diffusivity of Cu and multiplies it by the 4951 atoms of the cell. 4951 times the zero-strain 1.84 µm²/s of its Fig. 4 is 0.911 Å²/ps, the 700 K D_i of Table 1 (0.910±0.012). Table 1 therefore lists N D* ` f D_defect. The ν of K_iv = V_iv ν is the rate of one jump path, and (2/3)a²ν is D_defect when successive defect jumps are uncorrelated. `MD_SIA_TRACER=1` measures both on one free <100> dumbbell in the 13³ cell, with eight 10 fs trajectories per temperature at zero total pressure. Under Langevin at γ ` 0.1/ps, at 300, 500 and 700 K, N D* by the Ref. [12] fit is 0.973±0.050, 1.016±0.022 and 1.034±0.025 of Table 1, the Wigner-Seitz defect D is 1.900±0.136, 2.013±0.088 and 2.225±0.138 of Table 1, and f ` N D*/D_defect is 0.512±0.033, 0.504±0.023 and 0.465±0.030. Under NPH then NVE, N D* is 1.056±0.056, 1.085±0.027 and 1.031±0.030 of Table 1, and f is 0.473±0.026, 0.460±0.028 and 0.439±0.026. An ideal-lattice Johnson walk has f = 0.4397±0.0003 after 10⁵ jumps. Our defect D is about twice the Table 1 D_i because the two are different quantities, and our N D* is the Table 1 quantity.
  The whole-cell k ensembles of `validation/direct_md_long.csv` and `validation/direct_md_ws.csv` and the tracer campaigns integrate at 10 fs. Neither paper states an MD timestep. A 2 fs repeat of the Langevin tracer campaign at 500 K counts the same site changes as the 10 fs run, 1.169±0.020 against 1.171±0.010 per ps, and the same jump-sum D, 1.296±0.022 against 1.297±0.011 Å²/ps. Its N D* is 0.985±0.028 of Table 1. Its defect D, however, is 1.227±0.082 times the 10 fs value (z = 2.98). Defect D over jump-sum D is 1.003±0.054 at 2 fs and 0.817±0.035 at 10 fs, so at 10 fs successive sample-to-sample site changes are correlated; the committed jumps of `validation/direct_md_sia_d.md`, held for 0.3 ps, are not. f is 0.399±0.023 at 2 fs.
- `SaddleError` has a new `Singular(usize)` variant, returned by `solve_dense`, `solve_dense_matrix`, `dense_logabsdet` and `expand` when a pivot falls at or below `relative_zero_tolerance`, and by `spd_logdet` for an all-zero matrix.
  The C ABI reports it as `RGSADDLE_SOLVER`, as before.
  `SaddleError` is now `#[non_exhaustive]`, so a `match` on it outside rgsaddle needs a wildcard arm, and later variants will not break such a match again.
- `SolidDimerConfig::frozen_first_order` builds a fixed-cell dimer that certifies the projected index at convergence. A point that is still not index one after the escapes are spent is returned with `SolidDimerReport::rejected` set, and its status stays converged, so callers read that flag.
  `certify_vineyard_pair` checks a minimum and a saddle before Vineyard rates: each must have a projected gradient no larger than the force tolerance, the minimum must have no projected eigenvalue below the index cutoff, and the saddle exactly one.
  Directions that a surface names in `PointSurface::excluded_modes` are projected out of the force and of every Hessian action; nothing else is projected, so an open cluster has to name its rotations as well as its translations.
  `VineyardIndexError` says which test failed, and `VineyardIndexReport` carries the lowest projected eigenvalues of both points.
  `certify_vineyard_pair` needs `IndexCheck::lowest` of at least two, so that the second eigenvalue of the saddle is resolved and an index-two point or a maximum is refused. It validates the points and the check before it evaluates either surface. `VineyardIndexReport` and `VineyardIndexError` are non-exhaustive.
- `StepRefine` (`BandConfig::with_refine`, `BandSession::set_refine`, `BandSession::is_refining`) chooses the band stepper on each step. The default is no refine stepper. `StepRefine::default` is L-BFGS with memory 20 and a threshold of 0.5, in the host's force units. A step whose convergence force is at or under the threshold uses that method, and any other step uses the primary FIRE stepper. Both keep their own state.
  C ABI minor 60 adds `rgsaddle_band_set_refine` and `rgsaddle_band_is_refining` (`Band::set_refine` and `Band::is_refining` in C++ and Python). A host checks `rgsaddle_abi_version()` for minor 60 or newer. A threshold of 0 selects 0.5, and a memory less than or equal to 0 selects 20 pairs.
- `benchmarks/pafi_reference.sh` reruns the examples of the pinned Swinburne and Marinica reference PAFI code at github.com/tomswinburne/pafi with rgsaddle's PAFI: the W vacancy hop on the Marinica04 EAM at 0, 1000 and 2000 K on 9 and 33 planes from both ends, and the Fe dumbbell hop on the Marinica07 EAM at `kT → 0` against the nine gradients of the reference `tests/test_pafi.py`.
  One ignored test in `tests/pafi_reference.rs` writes `benchmarks/pafi_reference.csv` and `benchmarks/pafi_reference_planes.csv`, another samples six W planes at `μ dt` of 10⁻³ and 8 × 10⁻³ and checks that their gradients agree, and `PAFI_REFERENCE_CODE=1` also runs the reference code through `benchmarks/pafi_reference/reference_code.py` into `benchmarks/pafi_reference_code.csv` and `benchmarks/pafi_reference_code_planes.csv`.
  Against the forward barriers in `benchmarks/pafi_reference_code.csv`, with σ combining rgsaddle's block error and the reference code's worker-trajectory spread in quadrature, the forward column of `benchmarks/pafi_reference.csv` differs by up to 1.1e-5 eV at 0 K and, at 1000 and 2000 K, up to 0.29σ on 33 planes and 1.93σ on 9 planes; against the same forward reference column, the reversed column differs by up to 2.86e-3 eV (33 planes) and 2.94e-2 eV (9 planes) at 0 K, and at 1000 and 2000 K up to 0.41σ on 33 planes while reaching 0.0513 eV (6.08σ) on 9-plane trapezoid at 1000 K and 0.0504 eV (3.98σ) at 2000 K (9-plane remeshed stays within 0.17σ).
  That 9-plane reversed gap is a coarse-grid quadrature asymmetry on the curved path under `r ` |X - X₀| / |X_end - X₀|`, which shifts interior knots toward `r ` 1` and leaves a non-zero endpoint drift `F(1) - F(0)` in both codes separating `F_max - F(1)` from the forward `F_max - F(0)`; against the reversed integral of `benchmarks/pafi_reference_code_planes.csv`, the reversed column differs by up to 1.4e-5 eV at 0 K and up to 1.20σ (9 planes) or 0.52σ (33 planes) at 1000 and 2000 K.
  That comparison is read from the CSV files, not asserted in a test.
  The tests read LAMMPS `eam/fs` tables and data files; CI checks that reader against LAMMPS on a synthetic table and runs the pipeline on a small hop.
- `geodesic_path`, `geodesic_interpolate`, `geodesic_lengths`, `GeodesicConfig`, and `MorsePairChart` minimize the redundant coordinate path length of Zhu, Thompson, and Martinez, J. Chem. Phys. 150, 164103 (2019), doi:10.1063/1.5090303. The chart is caller supplied. `MorsePairChart` is their Eq. (7) with alpha 1.7 and beta 0.01. When a `minimage::Cell` is set, pair distances, Cartesian midpoints, subdivision samples, and inserted images all follow the minimum image displacement, and `rgsaddle_geodesic_path` takes an optional cell for a host chart. Interior images take an L-BFGS step from rgmin on the analytic length gradient. The endpoints stay fixed. `geodesic_interpolate` follows their Algorithm 3. The target is the average of the redundant coordinates, \((q(A) + q(B)) / 2\). Ten seeded noisy starts alternate between the reactant and the product, the middle image is adjusted on the three point path, and the images are copied onto thirds. `GeodesicConfig::seed` sets the noise. The C entry points check `n_images * dof`, `n_coords * dof`, and the Morse pair count times `3 * n_atoms` with `checked_mul` and a slice size limit, and return `RGSADDLE_INVALID_PARAMETER` on overflow. A run that uses all of `max_sweeps` with an image still moving is reported: `GeodesicReport::converged` is false, and the C entry points return `RGSADDLE_NOT_CONVERGED` with the last iterate in the outputs. C ABI minor 64 adds `rgsaddle_geodesic_config_default`, `rgsaddle_geodesic_path`, and `rgsaddle_geodesic_morse`.
- `hessian_mode_sweep_directions`, `ModeSweepDirectionPlan` and `mode_sweep_schedule` seed a saddle search from the Hessian modes of a minimum, soft to stiff.
  The Hessian is recovered by central differences and diagonalized in full, so every non-rigid mode gets a direction, including for systems larger than the Lanczos dimension.
  Each direction is the mode plus a carrier-weighted, optionally localized copy of itself, normalized; its squared overlap with the mode is at least `(carrier / (1 + carrier))^2`, which the plan reports.
  Each eigenvector takes a fixed sign, so the directions do not depend on the sign the eigensolver returned.
  `mode_sweep_schedule` picks a short, stratified list of modes from the band, and a list at least as long as the band visits every mode once before it cycles.
  Above 128 degrees of freedom the sweep needs the `faer` feature and returns `SaddleError::Solver` without it.
  Directions are formed at a scale that cannot overflow or underflow, so the overlap bound holds for every finite positive carrier, and a direction that cannot be formed is an error instead of a coordinate axis. `ModeSweepDirectionPlan` is non-exhaustive.
- `npt_lattice`, `NptConfig` and `NptReport` sample the lattice parameter of a crystal under isotropic Langevin dynamics with a Berendsen barostat.
  `NptConfig` is non-exhaustive: start from `Default` and chain the `with_*` setters.
  The thermostat stream is xoshiro256** seeded through SplitMix64, so every seed, including consecutive ones, starts an unrelated stream.
  The run discards `equilibration_steps` and reports the mean lattice parameter and pressure of the rest.
  The standard errors come from blocking the per-step series (Flyvbjerg and Petersen) and take the largest estimate over the block lengths that keep at least sixteen blocks, because the barostat correlates successive steps over many hundreds of steps and the error of fixed 250-step block means falls short of the true error for such a series.
  `lattice_error_resolved` and `pressure_error_resolved` say whether the estimate stopped rising with block length; when they are false the standard error is a lower bound.
  `pressure_converged` is true when the pressure error is resolved and the mean pressure is within two standard errors of the target, which is zero by default. It is an indicative screen, not a proof of equilibrium.
  A run that misses the criterion extends by `extension_steps` until it passes or reaches `maximum_steps`.
  `npt_lattice` refuses a single atom, which has no thermal degrees of freedom once the centre-of-mass momentum is removed, a control that is NaN or infinite, positions or a cell that are not finite, and a cell with no volume, all before the first evaluation. A value that turns non-finite during the run is an error rather than a report with a NaN mean and a zero standard error.
- `rgsaddle_lineb_config_default` and `rgsaddle_lineb_solve` expose fixed-energy LI-NEB and harmonic quantum TST through the C ABI at minor 46.
  `molecular` 0 expands atom masses onto Cartesian coordinates without alignment; `molecular` 1 uses quaternion molecule alignment.
  `beads` 0 returns the path only with a NaN `log_rate_over_reactant`; an unconverged search returns `RGSADDLE_STALLED`.
  `rgsaddle_lineb_solve_with_kappa` accepts an explicit endpoint restraint coefficient from ABI minor 47.
  `rgsaddle_lineb_solve_options` accepts flags for a straight initial band and separate endpoint and perpendicular convergence tests from minor 48. Minor 49 adds projected velocity Verlet.
  Minor 50 adds `rgsaddle_lineb_solve_counted`. Its `rgsaddle_lineb_counts_t` partitions surface evaluations among seeding, optimization, path preparation, quadrature, dynamics and the rate prefactor.

### Documentation

- Bibliography DOI metadata now matches Crossref for the AFIR and Simons and Nichols citations, and IDPP references cite Smidstrup et al. (2014), doi:10.1063/1.4878664.
- Every public module, type, function, constant, configuration field, C ABI symbol, and Python binding is indexed in `docs/orgmode/reference.org` and `docs/orgmode/reference/c-symbols.org`, and verified in CI by `docs/check_c_symbols.py`.
  Added `llms.txt` and `llms-full.txt` rendered by `docs/render_llms.py`.
- The PAFI reference describes `BalancedBarriers` and the calls that build one from a profile or from two barrier estimates.
  The public reference describes `SensitivityCover` and how its order, covered count and remaining shift choose which hop classes to compute.
- The copper validation lattice table now comes from `npt_lattice` with the kinetic pressure in the barostat, and the older table, which left it out, is marked stale where it is still recorded.
  The defect-pair guide lists the new lattice parameters and the `cargo test --release --test mishin_cu npt_lattice_table -- --ignored --nocapture` command that regenerates them.
  The guide gives the run settings, the 0.0006 Å seed to seed spread of a 4000 step run at 600 K, and a measured comparison of eight 4000 step runs with four 40000 step runs (`npt_lattice_run_length`), which replaces an estimate of the bias that the run length leaves.
- The defect-pair guide no longer says that the 0.0006 Å seed spread is larger than the lattice offsets. Compared with the differences of the listed columns, that spread is smaller than four of the five offsets and larger only than the 500 K offset.
- The readme now follows the C ABI constants: config minors 5 through 62 read `rotation_angle_tol` as given, and minor 63 keeps the default five degree stop when that field is 0 unless `RGSADDLE_MINMODE_EXPLICIT_ROTATION_ANGLE_TOL` is set.
  The minimum-mode explanation marks the Lanczos torque window bound as `rotation_tol >= 1e-4`, which Org can render, and `llms-full.txt` is regenerated from the Org sources.
- The reference tables and how-to pages now cover `npt_lattice`, `NptConfig` and `NptReport`, `hessian_mode_sweep_directions`, `ModeSweepDirectionPlan` and `mode_sweep_schedule`, and `certify_vineyard_pair`, `VineyardIndexReport` and `VineyardIndexError`.
  The NPT lattice how-to explains the blocking standard errors and why `pressure_converged` is indicative.
  `llms.txt` and `llms-full.txt` are regenerated, which refreshes the defect-pair-rates table that they carried.
- The reference tables in the main reference and the PAFI types page export as valid tables, so the strict documentation build passes again. The two tutorial scripts under `examples` return a typed result, which keeps `mypy --strict` clean, and `llms-full.txt` is regenerated from its sources. The crate lists the `either` and `zerocopy` features that its benchmark dependency enables, so `cargo test` and the library build used by the C ABI tests share one build of `rgmin` and the doctests compile under the minimum supported Rust version.
- The repository now carries citation metadata for rgsaddle in `CITATION.cff`, `codemeta.json` and `.zenodo.json`, with the author ORCID and the Zenodo concept DOI for all versions. `tests/package_version.rs` holds their version fields equal to `Cargo.toml`, and the release steps in the contributing guide list the three files.

### Developer

- Copper defect-pair readers now reject an empty, whitespace only, or separator
  only `RGSADDLE_TEMPS` list by naming the variable and leaving the target output
  file unchanged. The check covers 11 files across 7 readers.

  Table 1 lattice lookup no longer maps both 400 K and 600 K to 3.643, and an
  unlisted temperature now panics. Rotation rates now panic instead of returning
  zero when no rated PAFI row, and no barrier with an attempt frequency, exists
  for the temperature.
- The Lanczos and ART preset test now includes positive and negative infinity among the `rotation_angle_tol` values that must fail as finite and non-negative, with no default substituted.
- The all feature test suite now runs independent numerical campaigns, seed sweeps, and molecular checks in parallel and builds only the C ABI shared library in the shared target directory while preserving every fixture size, sample count, tolerance, and assertion.
- The default tests cover coupled atom and strain relaxation, external stress response, and finite difference stress.
- The wheel tests now copy `bench/harness.py` into the test directory, so `tests/python/test_coverage_harness.py` can load the saddle-coverage harness inside cibuildwheel without the large `bench/results.json`. The OH-TST and PAFI tutorial scripts check that the result file holds a JSON object and convert it to `dict[str, float]`, which clears the `no-any-return` error from `mypy --strict`.

### Changed

- A converged band, min-mode or OCI-NEB step now certifies the saddle index of its point by default, and the check's surface evaluations count in that step's `surface_rows` or `evaluations`.
  Every step still reaches the same positions.
  `with_saddle_check(None)` turns the check off when a caller needs the earlier evaluation budget.
  `BandReport`, `MinModeReport` and `OcinebReport` gain the public field `saddle_index`, so code that builds one with a struct literal or matches it exhaustively must handle the new field.
  A C config stamped with ABI minor 40 or newer certifies by default, while one stamped earlier does not; Python's `MinMode` certifies by default.
- Free-end adaptive NEB and dynamic NEB are exclusive. A session built with both, `BandSession::set_free_end` on a dynamic band, and `BandSession::set_dynamic` on a free-end band return an invalid-configuration error that names both options.
- The interpolated LI-NEB flux restricts its symmetric Hessian to the retained modes with Householder reflections and computes only eigenvalues on the faer backend.
- The `minimage` requirement is now `>=0.1.2, <0.2` in place of the exact `=0.1.2`, and `Cargo.lock` holds 0.1.4.
  A dependent that needs `minimage` 0.1.3 or newer now resolves beside rgsaddle, and linkcell v0.3.5 and rgsaddle still share one `Cell`, because linkcell asks for the caret `^0.1.2`.
  From 0.1.3, `Cell::displacement` on an orthorhombic cell wraps a coordinate difference across every box image, and `Cell::displacement_euclidean` returns the nearest image on a triclinic cell for any separation.
  `SparsityPattern::from_positions` on an orthorhombic cell therefore finds every pair for atoms that sit more than one box length outside it, where 0.1.2 missed pairs once a difference reached 1.5 box lengths.
  The public API of `minimage` only grows, and the test suites, golden hashes and reference values do not move.
- `ArtConfig::default` changes its values and no longer holds Lennard-Jones reduced units.
  `push_length` goes from 0.4 to 1.0, `push_step` from 0.3 to 1.0 and `push_over` from 0.15 to 1.0.
  The perpendicular force tolerance goes from 0.05 to 0.001 and the mode force tolerance from 0.01 to 0.001, so the two are equal.
  The eigenvalue threshold goes from -0.01 to -0.001, the Lanczos rotation budget from 12 to 20, the perpendicular and quench step caps from 800 and 4000 to 10000, and both relaxation initial steps from 0.05 to 0.1.
  The old values are in `ArtConfig::lj_art`, except that its perpendicular force tolerance goes from 0.05 to 0.01 and its saddle certificate uses central differences.
  A configuration whose perpendicular force tolerance is looser than its mode force tolerance is now rejected: `ArtConfig::validate`, `ArtSession::new` and `run_events` return `SaddleError::Invalid` for it, naming both values, before any search starts.
  The old default pair of 0.05 and 0.01 is one such configuration.
  The same calls reject a mode force tolerance that is not finite and positive, and a Lanczos block that `MinModeSession::new` refuses, with an `ART mode:` message.
  `ArtConfig::for_units` scales `lj_art` by each field's own unit: lengths, forces and curvatures by their factors, FIRE and quick-min time steps including the new `MinModeConfig::time_step` as length over the square root of energy, the opening steps of steepest descent, L-BFGS, nonlinear CG, DIIS and Barzilai-Borwein as length squared over energy, and Adam's as a length.
  The rotation tolerances keep their values, and a relaxation method with no single conversion factor is an error.
- `PafiMeanForce::PublishedEquation8` evaluates both terms of Swinburne and Marinica, Phys. Rev. Lett. 120, 135503 (2018), Eq. (8), including the analytic `kT d/ds ln(|psi| / |X'|)` Jacobian term and its locality check. `PafiMeanForce::ReferenceCode` remains the default and reproduces the force-only estimator recorded by LAMMPS `fix pafi` and the public PAFI driver.

  `PafiIntegrator::EulerMaruyama` selects the paper's direct overdamped Eq. (9) step. `PafiConfig::trajectories` runs independent trajectories and reports the ensemble error used by the public driver.

  `ArcPath::geometry`, and so `pafi_profile`, return `SaddleError::Invalid` for an arc outside `ArcPath::bounds` instead of clamping it to the end. `PathGeometry` gains the field `jerk`, the third derivative of the spline.
- `average_hyperplane` keeps its walk in the basin of the path by default, as the `MaxJump` test of Swinburne and Marinica's reference code (github.com/tomswinburne/pafi, `MaxJumpThresh` 0.4 Å) drops a trajectory that left it.
  With `PafiConfig::max_jump` set, 0.4 Å by default, the production run is cut into `blocks` stretches; the thermalized configuration and the end of every stretch are quenched in the plane under `PafiConfig::quench`, the steepest-descent flow (`Method::Ode12r`) by default, and compared with the plane minimum quenched from the path.
  A stretch whose minimum has one atom farther than `max_jump` from that one is sampled again from the end of the last kept stretch; `HyperplaneAverage::rejected` counts them, `HyperplaneAverage::evaluations` includes the quenches, and more rejections than `blocks` is an error.
  Before, a walk that crossed into another basin of its plane was averaged with it: on a plane of two wells the gradient left the quadrature over the basin of the path by more than 0.25, against an error of 0.004.
  `with_max_jump(None)` samples as before; `DefectPairConfig::pafi` takes the new default.
- `average_hyperplane` steps its overdamped walk as Leimkuhler and Matthews, Appl. Math. Res. Express 2013, 34 (2013), doi:10.1093/amrx/abs010, Eq. (3), the high-friction limit of their BAOAB step: the noise of a step is the sum of a new Gaussian draw and the draw of the step before, and a step is still one force call.
  Stationary averages are second order in `PafiConfig::time_step` instead of first, and on a harmonic plane at `μ k dt = 1.2` the tests find `⟨y²⟩` within four errors of its exact value, where the Euler-Maruyama step it replaces samples `⟨y²⟩` 2.5 times too wide.
  The same `PafiConfig::seed` gives other trajectories than before.
- `rgmin` is the crates.io release 0.8.0 instead of a pinned git revision of 0.7.0, which was never published. This lets `rgsaddle` publish.
- `rgsaddle.ase.NEB` takes `k` as one spring or one per segment, `interpolate(method='idpp', mic=True)`, and `remove_rotation_and_translation` as a rigid fit on a free non-periodic band.
  `idpp_path` is the image-dependent pair-potential path those calls run.
  `DimerControl`, `MinModeAtoms(atoms, control)`, `displace`, and `MinModeTranslate` follow ASE's dimer call, with the mode from this dimer.
  An ASE option with no equivalent, including `spline`, `string`, `precon`, and an MPI world, raises `NotImplementedError` naming it.
  `tests/python/ase_upstream` runs ASE 3.29.0's NEB and dimer tests against these drivers, and the Python CI job sets `RGSADDLE_REQUIRE_ASE=1`.

### Fixed

- `FixedEnergyTrajectory::from_gradients` refuses a non-finite position, gradient, or mass, and a mass that is not positive.
- `SplineEnds::NotAKnot` now uses the same knot, shape, and finiteness checks as `SplineEnds::Natural` and returns the same typed error for an empty or non-finite input.
- A band step that accepts a geometry other than its last evaluation
  assembles the preconditioned driving force when an exponential metric
  is installed, for every stepper.
- A free-end step no longer evaluates the fixed terminal image again after the first step, and the step that follows starts from the evaluation the last step ended on instead of evaluating the band a second time. The trajectory is unchanged. On the narrow-barrier example the surface sees 1662 rows where it saw 2522.
- A free-end step now rebuilds the exponential band metric after its interior images move far enough to make it stale, and after a cut and remesh. Before, the reported forces and the cached evaluation were assembled without a current metric until the next step.
- A joint `SolidNeb` band now clears its optimizer history when the climbing image moves to another image, so steps for the new climber are not shaped by curvature pairs from the old one.
- A not-a-knot spline refuses a pivot that is not larger than one unit
  in the last place of the knot span.
- Band evaluations now cache exponential-metric driving forces, build metric
  neighbours with linkcell, solve not-a-knot splines in linear time, and
  parallelize per-image force assembly when image threads are enabled.
- Cached band forces are keyed on the current exponential-metric build
  and the metric is refreshed on every step exit, including converged
  min-mode refinements, so an evaluation cannot assemble or reuse forces
  from a stale metric.
- Clearing dynamic NEB drops the cached assembly and the solver's stored force, so an exponential metric band does not keep a masked driving force.
  `BandSession::reset` drops the dynamic image answers, the step pin and the step origin when the surface changes.
  `DyNEB.set_positions` reads the band force once, then moves every image that is still free.
  A non-finite `fmax` or `scale_fmax` on `rgsaddle_band_set_dynamic` and `rgsaddle_dyneb_mask` returns `RGSADDLE_NON_FINITE`.
  The simplified string method and free-end adaptive NEB stay on the refused list of `rgsaddle_band_set_dynamic`.
- For C min-mode configs built at ABI minor 63, passing
  `rotation_angle_tol = 0` now keeps the documented default improved-dimer
  angle stop unless `RGSADDLE_MINMODE_EXPLICIT_ROTATION_ANGLE_TOL` marks
  that zero as explicit.
  Callers built against minors 5 through 62 keep the older behaviour,
  where `rotation_angle_tol = 0` disables the angle stop.
- Free-end NEB cut thresholds at or below the final \(4\,f_{\mathrm{tol}}\) threshold now stay at that threshold instead of increasing between cuts.
- Free-end NEB cuts now reset the endpoint step size together with the main band optimizer state, so a remeshed phase does not inherit stale step history.
- Geodesic interpolation rejects an odd `bound_samples` count, because that equal partition skips the Cartesian midpoint and the upper bound need not dominate the midpoint length. A C `max_images` above the output image count is `RGSADDLE_INVALID_PARAMETER` instead of being clamped. A refused optimizer trial no longer counts as a converged step when it leaves the image unmoved. `MorsePairChart::with_radii` checks that every radius is finite and positive, and a Morse evaluation whose finite parameters overflow returns `SaddleError::NonFinite`.
- Improved dimer rotation now stops on its five degree trial and fitted
  angle test by default instead of also inferring the classic torque
  window, and checks the trial angle against `rotation_angle_tol` before
  evaluating the trial rotation, as eOn's improved dimer does.
  The angle criterion follows Heyden, Bell, and Keil, J. Chem. Phys. 123,
  224101 (2005), doi:10.1063/1.2104507, and Kästner and Sherwood,
  J. Chem. Phys. 128, 014106 (2008), doi:10.1063/1.2815812.
  Set `rotation_angle_tol` to zero to infer the classic window, or use
  `MinModeSession::set_torque_window` to select it explicitly.
  Under a `rotation_tol` tighter than 1e-4 every angle exit, before the
  trial rotation, after the anharmonic fallback, and after the fit, is
  skipped unless `MinModeConfig::with_rotation_angle_tol` set the angle,
  so the default drives the rotational force under `rotation_tol`; an
  explicit five degrees keeps the angle test. A C or Python session
  always passes its angle, so it keeps the angle test.
  A positive curvature climb keeps the seed mode orientation when the
  parallel gradient is within half of that climb of the minimum along the
  mode, in `MinModeSession` and in eigen following.
  The climb still moves the largest atom by `max_move` under FIRE and
  L-BFGS; under `set_cg_translate` it has Euclidean norm `max_move`, as in
  eOn's conjugate gradients.
  The stalled FIRE probe opens past a curvature magnitude of 4, the
  bound of the opening timestep.
  `Hyperplane::from_min_mode` now prefers the lowest Ritz vector of a
  converged saddle-index certificate at the session position, oriented
  along the session mode, and keeps the session mode when no certificate
  converged.
- LI-NEB endpoint quadrature now uses the finite turning-point limit despite positive roundoff in the endpoint energy, and its fine-grid regression has a bounded refinement.
- Lanczos now stops when the lowest Ritz residual divided by the absolute Ritz value is at most `rotation_tol`, together with the relative change of that value.
  A Ritz value near zero keeps that comparison absolute, so the stop never divides by zero.
  The Lanczos torque window uses the same ratio and stops once two vectors leave it under 1.
  The dimer keeps an absolute rotational force in curvature units.
  ART unit conversion therefore leaves `rotation_tol` unchanged for Lanczos and scales it by energy over length squared when the dimer is selected.
- LineB passes warning-denied Clippy checks, and wheel smoke tests copy only test sources present in the source distribution.
- PAFI hyperplanes follow the not-a-knot cubic spline through the images, the spline interpolation of Swinburne and Marinica, Phys. Rev. Lett. 120, 135503 (2018), p. 135503-2, instead of the chords between them.
  `ArcPath::geometry` returned the chord of the segment as the normal and no curvature, so the normal jumped at every image, `ψ` was zero, and each average was the mean force on a plane of no common family; the normal also kept its center-of-mass part when the center of mass was held.
  On the Fe self-interstitial path of the reference code (github.com/tomswinburne/pafi, commit 1661a1d) at `kT → 0` the chord planes gave `dF/dr` ` 0.378 at `r ` 1/8` against the reference 0.614, and a trapezoid barrier of 0.228 eV against 0.262 eV; the spline planes give all nine reference gradients within 1e-3.
  `ArcPath` now shares the spline of `Reparametrization::NotAKnot`, `ψ = (x - X) · X'' / |t|²` with `t` the center-of-mass-free `X'`, and the drift, the noise and the return to the plane use that one normal.
  `ArcPath::reversed` is the same curve from the last frame, for the reversal test of a profile.
  `zero_temperature_barrier` integrates the energy along the spline with 64 points per span, and `defect_pair_rates` takes its reactant attempt frequency and Bennett–Chandler normal from the spline tangent.
  The copper rows of `benchmarks/defect_pair_network_pafi.csv` and the volumes built on them were integrated on the chord planes and are not recomputed here.
- Rebuilding the exponential metric drops the cached band force, so the
  next assembly uses the new matrices.
- Solid-state NEB activates the climbing image only when the highest interior image exceeds both endpoint enthalpies. A band that slopes toward an endpoint continues with the ordinary NEB forces.
- The C ABI tests now link their fresh shared library in a separate target directory, so `cargo test` no longer replaces the library the doctests link against.
- The C header now documents that `rgsaddle_band_free_end_residual` returns `RGSADDLE_INVALID_PARAMETER` for a null output pointer, consistently with the other band getters.
- The NPT pressure now includes the kinetic term. The barostat drive and the reported pressure add the kinetic pressure of the thermostatted velocities to the configurational pressure, so the sampled lattice is the zero-pressure lattice rather than the one where the configurational stress alone vanishes.
  Tests check an ideal gas, which must read (N - 1) k T / V because the thermostat removes the centre-of-mass momentum, and a crystal of atoms tethered to cell-scaled sites, whose total pressure is zero at every volume.
- The all-features suite no longer stalls in `colour_block_strong_scaling_on_vacancy_cells`.
  That test recovers shifted-force Hessians of a 10975-atom vacancy and a 97555-atom vacancy, five times at each thread count.
  The test build compiled the neighbour walk unoptimized, so the test stayed in that walk for hours.
  Test builds now use opt-level 3, with debug assertions still on.
  A 12³ vacancy recovery that took 105s unoptimized takes 13s optimized, and the test fails if that recovery exceeds one minute.
  The C ABI smoke tests also find `librgsaddle.so` in the cdylib unit directory, which is where Cargo leaves it when the test profile's opt-level differs from dev.
- The evaluation a free-end step caches now holds the preconditioned driving force under the metric it is tagged with, and no driving force when the band has no metric. Before, the cache held the unpreconditioned force under a metric tag, so a later read that trusted the tag used the wrong force.
- The fixed-energy MEP seed opens each branch with a step of sqrt(machine epsilon) times a distance along the path, and each energy slice brackets its crossing from the same fraction of the spacing of the images already placed on that branch. The opening distance is measured on the surface, from the saddle along the branch's mode, to the point where the energy has fallen by a tenth of the first energy drop. No step depends on an absolute coordinate, so translating the saddle and the surface together leaves the seed unchanged, and a change of length unit scales the step with the coordinates. A mode along which the energy never falls to that level is SaddleError::Invalid.
  `FixedEnergyTrajectory::from_images` checks the shape, finiteness, and positive masses of its input, and that the mass-weighted path has arc length, before it calls the surface. A bad input returns SaddleError::Shape, NonFinite, or Invalid without evaluating any image.
- The near-shell rates of the copper V_iv that carry its weight are now PAFI free-energy rates, not the barrier-shift model. `benchmarks/defect_pair_network_sensitivity.csv` ranks every hop class inside the census radius by `dV/d ln k` of the 30 Å cell-decay volume, by central differences of 0.1 in `ln k` on both directions of the class. Classes are taken in that order until the rest, all moved by the model spread together, shift the volume by less than the smaller Table 1 error (21, 15 and 10 Ω at 300, 500 and 700 K). With the spread of 0.576 in `ln k` from the eight recorded PAFI rows that selects 18, 27 and 35 of 162, 166 and 166 classes. Once they are rated the cover is repeated on the classes still on the model, with the spread of the calibrated model below and half the Table 1 error. Earlier passes with the uncalibrated spread called for 4 and then 2 more at 300 K, and one during the calibration fit for 3 more at 700 K; the 13 unrated 300 K classes ranked next were rated as well, to test the coverage. With 37, 27 and 38 classes rated the rest shift the decay volume by at most 6.74, 7.22 and 8.08 Ω against 10.5, 7.5 and 5 Ω. The cover selects none at 300 and 500 K. At 700 K it selects 10 more, which would bring the bound to 4.97 Ω; they are not rated, and the bound there stays below the full Table 1 error of 10 Ω. The largest sensitivities, 8.8, 10.3 and 10.4 Ω per unit `ln k`, are translations into the spontaneous (3,3,0) state; on the rated network every class moved together moves the decay volume by 102.0, 103.5 and 105.2 Ω per unit `ln k`.
  `benchmarks/defect_pair_network_pafi.csv` holds the rates: 37, 27 and 38 hop classes and the isolated translation and rotation at each temperature, all rated, in the 13³ cell on the Table 1 lattices. Each class's saddle is searched again as in the network campaign. A climbing band of thirteen images with energy-weighted springs joins the bound state, the verified dimer saddle (the band top for one class at 300 and one at 500 K) and the relaxed target basin; a band into recombination ends 0.3 Å past the saddle. Bands are kept at a force of at most 0.01 eV/Å. PAFI samples each image's hyperplane until the barrier error is at most 0.2 kT, or four passes. One path gives both directions: the forward barrier above the reactant and `FreeEnergyProfile::reverse_barrier` above the last plane, each with the attempt frequency from the curvature at its end. A direction runs at `k_kind (nu/nu_iso) exp(-(F - F_iso)/kT)` against the isolated hop of the same kind. The two barriers of a path should differ by the gap of its ends, but the integration drifts from it. The isolated hop ends in a state equivalent to its start, yet its forward barrier exceeds the back one by +0.0131, +0.0172 and +0.0300 eV translating and by -0.0021, +0.0212 and -0.0004 eV rotating; on the 31, 19 and 27 classes rated both ways the forward less the reverse barrier drifts from the zero-temperature gap of the ends by -0.460, -0.282 and +0.078 kT on average and 0.776, 0.528 and 0.513 kT in root mean square, a χ² of 168, 79 and 135 against their errors. Four 300 K classes integrated again from the other end place the drift past the dividing surface: the barriers read from the start of each integration agree, and those read from its end do not. `anchored_barriers` therefore keeps the forward barrier of each path and its error and puts the reverse barrier the zero-temperature gap below it, and the isolated hops take their forward barriers, so that a class and its reference are read alike. The 6, 8 and 11 recombinations are rated one way only and keep their forward error. Against the 37, 27 and 38 rated classes, one residual each from the mean over its directions, the barrier-shift model's `ln k` is off by 0.570, 0.343 and 0.520 in root mean square and by +0.102, -0.055 and +0.200 on average, computed less model. `calibrate_surrogate` fits that bias by weighted least squares with a Paule–Mandel scatter, +0.0665±0.0784, -0.0623±0.0642 and +0.1982±0.0798 with a scatter of 0.4408, 0.2856 and 0.4617, and every direction still on the model runs at `exp(c)` times its model rate, both directions of a class alike.
  In `benchmarks/defect_pair_network_volume.csv` a rated direction replaces the model. In the 64 draws each rated class draws one normal for both its directions, each isolated kind one that moves its rated and calibrated model directions together, the calibration its coefficients within their covariance, and the remaining near-shell rates a common factor with the calibration scatter. At the 30 Å cutoff the 13³-cell decay volume is 228.2±22.5, 194.0±20.3 and 206.4±19.6 Ω (was 210.8±62.3, 191.1±60.9 and 177.9±51.2), the mean-time volume 237.8±24.8, 200.4±22.0 and 213.9±21.3 Ω, and the free sphere 142.7±9.2, 128.0±9.2 and 133.5±8.6 Ω, against 254±21 / 286±28, 185±23 / 161±15 and 203±10 / 210±21 Ω in Table 1. The PAFI rates move the decay volume by +16.4, +4.4 and +25.3 Ω and the calibration by +1.0, -1.5 and +3.3 Ω. The decay volume lies within one combined error of alpha at every temperature: 25.8 Ω below it at 300 K against 30.8, and 9.0 and 3.4 Ω above it at 500 and 700 K against 30.7 and 22.0. Balancing the two barriers of each path on the gap of its ends instead, with `balanced_barriers`, gives 181.2±29.6, 160.6±21.1 and 180.2±27.5 Ω on the same rows: the drift past the dividing surface then slows the classes against their reference. The volumes stay within 0.4 % from the 14 Å cutoff on; at 10.8 Å, the census radius, they are 2.1 %, 0.3 % and 4.0 % lower. The campaign took 128.4 min for a probe of 8 jobs, 711.1 min for the main pass and 49.7 and 33.1 min for two 300 K passes on 8 threads; the 13 further 300 K classes took 296.9 min and the 3 further 700 K classes 74.9 min, sharing the machine. An earlier pass was stopped after two rows, and its wall time was not recorded. The job time summed over the rows is 75.9, 37.5 and 42.3 h. The sensitivity pass takes 3.5 min and the assembly 2.8 min.
- V_iv from a capture rate had α inverted. Mwungeli, Liu, and Demkowicz, Phys. Rev. Materials 9, 023601 (2025), write K_iv ` V_iv D_i/(αa²) ` V_iv ν with α ` 2/3 (section 1, p. 2 of the accepted manuscript, OSTI 2513909) and V_iv ` K_iv αa²/D_i (section 2, p. 5), so V_iv/Ω = 4αK_iv/(aD_i). In `tests/mishin_cu.rs` the run 12 assembler took ν ` (2/3)k and ν ` (2/3)D/a², `sphere_volumes` took ν = (2/3)D_graph/a² for the run 13, converged, census and network volumes, the isotropic and no-strain-bias rows of run 13 used 24πR/a, and `direct_md_capture_radius` converted with 24πR/a and 6(k/D)/a. The Smoluchowski check of `tests/defect_pair_rates.rs` expected 24πR/a, and `first_method_report` of `examples/langevin_nvt.rs` converted k as k a²/(αD_i). Each put V 1/α² ` 9/4 too high (3/2 for the ν ` (2/3)k rows). `dumbbell_channel_rate(a, D) = D/((2/3)a²)`, in `tests/common/fcc_dumbbell.rs` and in the example, is now the one conversion, and the free sphere is 16παR/a. With the 600 K slope of their Fig. 1, 0.00173/ps for 8788 atoms, a ` 3.649 Å and D_i ` 0.728 Å²/ps, it gives 185.4 Ω, against the 186±14 Ω of their Table 1.
  The regenerated volumes are 4/9 of the old ones; radii, site counts, site sums, mean times and rates do not move. At the 30 Å cutoff, at 300, 500 and 700 K: the run 13 Smoluchowski volume is 48.8652, 46.4938 and 44.3997 Ω (`benchmarks/defect_pair_run13_volume.csv`); the converged cell-decay volume is 90.0885±8.3344, 96.0389±14.9314 and 79.6626±7.3322 Ω, the mean-time volume 93.2813±9.0583, 99.7810±16.4374 and 82.0072±7.9637 Ω and the free sphere 58.6329±3.6935, 61.1935±6.2258 and 53.8943±3.2605 Ω (`benchmarks/defect_pair_converged_volume.csv`); the census cell-decay volume is 84.5791, 81.5963 and 79.6701 Ω (`benchmarks/defect_pair_census_volume.csv`); and the network cell-decay volume with the PAFI rates is 101.4151±10.0061, 86.2091±9.0214 and 91.7372±8.7054 Ω, the mean-time volume 105.6960±11.0431, 89.0686±9.7593 and 95.0771±9.4774 Ω and the free sphere 63.4231±4.1061, 56.9056±4.1103 and 59.3527±3.8280 Ω (`benchmarks/defect_pair_network_volume.csv`). Table 1 at 8788 atoms is 254±21 / 286±28, 185±23 / 161±15 and 203±10 / 210±21 Ω.
  With the same tolerances the sensitivity cover (`benchmarks/defect_pair_network_sensitivity.csv`) selects 3, 9 and 17 of 162, 166 and 166 hop classes at the model stage and none after PAFI, so the 37, 27 and 38 rated classes of `benchmarks/defect_pair_network_pafi.csv` cover it, and the classes still on the model shift the decay volume by at most 2.99, 3.21 and 3.59 Ω (`benchmarks/defect_pair_network_notes.txt`). In `benchmarks/defect_pair_run19_capture_radius.csv`, with Wigner-Seitz capture and starts beyond 8.1 Å, the 13³-cell decay volume at the fitted MD radius is 92.7664 [86.8492, 99.0611], 83.1149 [78.4902, 88.0287] and 66.3105 [58.8358, 73.9265] Ω, the mean-capture-time volume 96.2000, 85.7238 and 67.7797 Ω, and the free sphere 59.7979, 55.4975 and 47.2849 Ω. The first-method notes `validation/direct_md_long.md` and `validation/direct_md_ws.md` give 98.48±18.26 and 16.57±2.41 Ω at 300 and 500 K with 12-fold capture and 98.49±18.39, 27.32±5.98 and 16.55±3.75 Ω at 300, 500 and 700 K with Wigner-Seitz capture, using the Table 1 D_i, and now also convert k with the D_msd of the free dumbbell under the same dynamics.
  Every V_iv from a rate in the earlier changelog entries of the copper work used the inverted ν and is 9/4 of what the same inputs give now: the free sphere 24πR_eff/a, the volume 4πR_eff a²/α of k ` 4πD_graph R_eff (it is 4παR_eff a²) and the 13³-cell volumes at the fitted MD radius of `+defect-pair-cu.added.org=, the converged and census volumes of `+converged-campaigns.fixed.org`, and the network volumes and their four steps of `+converged-capture.fixed.org`, all three in the 0.5.0 changelog; and the network volumes, the cover counts and bounds, and the `balanced_barriers` comparison, which was not rerun, of `+pafi-near-shell.fixed.org`. The decay volume no longer lies within one combined error of Table 1 at any temperature.
  The Table 1 V_iv were converted with the Table 1 D_i, which is N D* = f D_defect (`validation/direct_md_dynamics.md`). A capture rate converted with D_i gives a volume larger by D/D_i than one converted with a defect diffusivity D, and for the free dumbbell defect D / Table 1 runs 1.900 to 2.360 there.
  `benchmarks/defect_pair_run12_volume.csv` is left as it was and is stale. Its mean-time rows still carry ν ` (2/3)k and ν ` (2/3)D/a², and its first-passage rows, which do not use ν, predate the current assembler: they give 202.4328 Ω at 300 K, where the assembler gives 276.8703 Ω.
- Vineyard prefactor validation now requires exactly one more mode at the minimum than at the saddle, so the prefactor always has units of frequency.
- `ArtConfig::validate`, `ArtSession::new` and `run_events` now hold `ArtConfig::mode` to every check that `MinModeSession::new` makes, including the new `MinModeConfig::time_step`.
  A zero Krylov or rotation budget, a finite-difference step or time step that is not finite and positive, a negative or non-finite rotation tolerance, an infinite or NaN `max_move` and a stepper that tests the oracle value return `SaddleError::Invalid` with an `ART mode:` message before the first evaluation, where an event could stop part way through.
  `ArtConfig::for_units` leaves both rotation tolerances as they are, scales the climb time step and each relaxation opening step by its own dimension, and a configuration converted to another unit system and back keeps every field.
- `ElasticConstants` and the Barnett quadrature now invert an acoustic tensor only
  when every leading minor is positive, so a tensor with two negative eigenvalues
  and a positive determinant, such as a cubic tensor with a negative `C44`,
  returns `SaddleError::Invalid` instead of being treated as stable.

  `PeriodicElastic::correct_rate_catalogue` now returns `SaddleError::NonFinite`
  when a corrected barrier or harmonic rate is not finite, such as the overflow
  of a strongly negative barrier at low temperature, where it used to return an
  infinite `rate_hz`.
- `mode_sweep_schedule` evaluates its stratum index with a wide product, so a mode count near the top of `usize` no longer overflows, and a schedule too long to store is an error.
- `npt_lattice` samples the isotropic NPT ensemble with stochastic cell rescaling (Bernetti and Bussi, J. Chem. Phys. 153, 114107 (2020)).
  `NptBarostat::Berendsen` remains available for equilibration and reports means only, with no ensemble fluctuation statistics.
  A cell that is not cubic is refused, and an extension that would overflow the step counter is an error instead of a panic or a stale report.

## [0.6.0] - 2026-11-19

### Added

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
