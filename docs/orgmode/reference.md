

# Public Rust types

The hosted [Rust API reference](https://docs.rs/rgsaddle) includes the default solver, C ABI, OH-TST, and readcon interfaces.
`cargo doc --no-deps --features capi,ohtst,readcon-db --open` builds these pages from a checkout.
The `lammps` feature adds the LAMMPS adapter to a local documentation build.
Runnable examples open `BandSession`, `AutoNeb`, `OcinebSession`, `MinModeSession`, `Index1Session`, `IndexKSession`, `SolidDimer`, `SolidNeb`, `Stencil` and the `ohtst` module, and, with `--features capi`, the C band, min-mode and stencil handles.
`cargo test --doc` runs them.
Every public configuration struct is non-exhaustive.
Start from `Default` or the type's named constructor and chain the `with_*` setters, as in `BandConfig::default().with_force_tol(1e-3)`.
The table below lists the main types.

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Type</th>
<th scope="col" class="org-left">Role</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left"><code>BandSession</code></td>
<td class="org-left">Outer seam. <code>new</code>, <code>step</code>, <code>run</code>, <code>restart</code>, <code>reset</code>, <code>set_positions</code>, <code>positions</code>, <code>evaluation</code></td>
</tr>

<tr>
<td class="org-left"><code>OcinebSession</code></td>
<td class="org-left">The band's refinement with the published controls (<code>OcinebConfig::refine</code>). <code>step</code> reports <code>Band</code>, <code>Align</code>, or <code>MinMode</code></td>
</tr>

<tr>
<td class="org-left"><code>AutoNeb</code></td>
<td class="org-left">Insert on the worst gap, relax a local window, and climb. The full band runs only when <code>full_band</code> is set. <code>run</code> is one shot</td>
</tr>

<tr>
<td class="org-left"><code>AutoNebTuning</code></td>
<td class="org-left"><code>n_simul</code>, <code>n_max</code>, local tolerance and step caps, <code>space_energy_ratio</code>, <code>climb</code>, <code>full_band</code>, <code>threads</code>, <code>record_samples</code>, <code>trics</code>. Stored on <code>BandConfig</code></td>
</tr>

<tr>
<td class="org-left"><code>BandConfig</code></td>
<td class="org-left">Tangent, spring, projection, climbing, cell, tolerances, method, optional AutoNEB tuning, <code>min_mode</code> refinement, <code>zoom</code>, <code>string</code>, <code>free_end</code>, optional <code>dynamic</code> (<code>DynamicNeb</code>, off unless set)</td>
</tr>

<tr>
<td class="org-left"><code>FreeEndConfig</code>, <code>FeaNebSession</code></td>
<td class="org-left">Free-end adaptive NEB of Zhang, Zhang, Ye, and Zheng (2016) and Zhu et al. (2007) on <code>BandConfig::free_end</code>. One terminal image (<code>FreeEndSide::First</code> or <code>FreeEndSide::Last</code>, default <code>Last</code>) moves on the target-energy level set with adaptive cut-and-remesh around the detected peak (<code>FreeEndReport</code>, <code>FeaNebReport</code>)</td>
</tr>

<tr>
<td class="org-left"><code>StringConfig</code></td>
<td class="org-left">Simplified string method of E, Ren, and Vanden-Eijnden (2007) on <code>BandConfig::string</code>. Non-exhaustive, off unless set. Evolves interior images under the full potential force and reparametrizes to equal arc length on <code>Reparametrization::NotAKnot</code> (default, <code>StringConfig::spline()</code>) or <code>Linear</code> (<code>StringConfig::linear()</code>), anchoring an armed climbing image</td>
</tr>

<tr>
<td class="org-left"><code>StringSession</code></td>
<td class="org-left">Standalone simplified string session preserving per-image <code>ImageIds</code> and batched surface evaluations; target of <code>GrowingStringSession</code> hand-off</td>
</tr>

<tr>
<td class="org-left"><code>GrowingStringSession</code></td>
<td class="org-left">Growing string method of Peters, Heyden, Bell, and Chakraborty (2004). Reactant and product fragments with frontier tangent constraints, configurable node-add criterion (<code>GrowingStringConfig</code>), separate fragment relaxation, join detection, and hand-off to <code>StringSession</code></td>
</tr>

<tr>
<td class="org-left"><code>MinModeRefine</code></td>
<td class="org-left">OCI-NEB refinement of the climbing image. Non-exhaustive. Trigger at any force, stability 1, start alignment 0.85, no alignment exit, retry under 0.5 of the starting force, refresh at 0.3 of the measured curvature or overlap 0.8 every 4 steps at most, budget 1000, Lanczos with twenty actions and relative tolerance 0.05</td>
</tr>

<tr>
<td class="org-left"><code>DynamicNeb</code></td>
<td class="org-left">Dynamic NEB on <code>BandConfig::dynamic</code>. Non-exhaustive, off unless set. <code>fmax</code> 0.05 and <code>scale_fmax</code> 0 match ASE 3.29 <code>DyNEB</code>: an interior image whose largest per-atom projected force is strictly below <code>fmax * (1 + distance * scale_fmax)</code> keeps its last energy and force. <code>distance</code> is the Euclidean norm from the highest-energy image, with no minimum image. That image stays in the spring and the tangent and is evaluated again when the projected force exceeds the threshold. The highest-energy image is never given a zero force.</td>
</tr>

<tr>
<td class="org-left"><code>ZoomConfig</code></td>
<td class="org-left">The nudged elastic band (NEB) zoom, Zoom-NEB, on <code>BandConfig::zoom</code>. Non-exhaustive, off unless set. Recommended settings for a band that opts in are the defaults: <code>alpha</code> 0.5, <code>offset</code> 1, <code>stability</code> 5, cubic placement, and <code>after</code> 0. An <code>after</code> of 0 zooms once the largest per-image force, measured by the session's convergence norm, is below twice <code>force_tol</code>. The window always reaches both neighbours of the climbing image. The min-mode refinement waits for the zoom decision. Plain climbing image NEB (CI-NEB) on the 51 benchmark pairs of the 38-atom Lennard-Jones cluster (LJ38) converged 20 pairs with the zoom on and 21 without it. It used more force calls on 10 of the 17 pairs that converged both ways. With the default min-mode refinement on, one LJ38 pair ran to the 250000 call cap. On Müller-Brown at 13 images with FIRE and <code>max_move</code> 0.05, plain CI-NEB took 30593 force calls with the zoom and 31308 without it. With L-BFGS of memory 100 and climbing from half the first force, the zoom does not help on Müller-Brown: the band took 6503 force calls with it and 3280 without it. The zoom is an opt-in, not a general default.</td>
</tr>

<tr>
<td class="org-left"><code>BandSurface</code></td>
<td class="org-left">Host potential on a band (energies + gradients per image)</td>
</tr>

<tr>
<td class="org-left"><code>BandReport</code></td>
<td class="org-left"><code>status</code>, <code>max_force</code>, <code>ci_index</code>, <code>iteration</code>, <code>surface_rows</code>, <code>curvature</code> (refinement steps only), <code>saddle_index</code> (converged steps)</td>
</tr>

<tr>
<td class="org-left"><code>SaddleIndexCheck</code></td>
<td class="org-left">The saddle-index check of a converged band, min-mode or OCI-NEB search, on <code>BandConfig::saddle_check</code> and <code>MinModeConfig::saddle_check</code>, on by default. Non-exhaustive. Block 2, lowest 4, residual 1e-2, curvature threshold -1e-3, 100 forward actions of 1e-3. <code>escape</code> takes a <code>SaddleEscape</code> (push 0.1, two attempts)</td>
</tr>

<tr>
<td class="org-left"><code>SaddleIndex</code></td>
<td class="org-left"><code>index</code>, ascending <code>curvatures</code> and their <code>residuals</code>, <code>converged</code>, <code>evaluations</code>. <code>saddle_index</code> certifies one point on a <code>PointSurface</code></td>
</tr>

<tr>
<td class="org-left"><code>MinModeSession</code></td>
<td class="org-left">Dimer or Lanczos lowest-mode search, same stepping contract; <code>set_position</code>, <code>set_mode</code>, <code>estimate_mode</code> for reuse; <code>set_torque_window</code> sets the dimer and Lanczos torque window apart from <code>rotation_tol</code>; <code>set_rotation_stop_length</code> turns on the curvature-aware rotation stop; <code>set_artn</code> enables or disables the ART nouveau protocol</td>
</tr>

<tr>
<td class="org-left"><code>SoftModeDirectionPlan</code></td>
<td class="org-left">Soft-mode-subspace-first direction seeding (<code>soft_mode_then_full_directions</code>, <code>spectral_gap_rank</code>). Recovers a sparse Hessian at a minimum, selects the soft-subspace rank <code>k</code> from the largest spectral gap, and emits <code>2 * phase_count</code> unit directions</td>
</tr>

<tr>
<td class="org-left"><code>ModeSweepDirectionPlan</code></td>
<td class="org-left">Deterministic soft-to-stiff Hessian-mode sweep (<code>hessian_mode_sweep_directions</code>, <code>mode_sweep_schedule</code>). Recovers the Hessian by central differences, diagonalizes it in full, and emits one carrier-weighted, optionally localized unit direction for every non-rigid mode. <code>minimum_squared_mode_weight</code> is the proven overlap floor. <code>mode_sweep_schedule</code> picks a stratified list of band positions. Non-exhaustive</td>
</tr>

<tr>
<td class="org-left"><code>ArtSession</code></td>
<td class="org-left">ART nouveau in eight persisted phases: masked host seed, perpendicular relaxation, Lanczos, lowest-mode climb, index-1 certificate, push both ways, unconstrained quench, reversibility. The climb checks curvature and consecutive-mode overlap, recovers the carried mode or restarts from the last accepted index-1 point. When recovery is exhausted it finishes as <code>ArtStatus::LostCurvature</code> if the rejected curvature is finite and non-negative and returns <code>SaddleError::ArtIndexOneLost</code> otherwise. <code>run_events</code> marks duplicate saddles. <code>ArtStatus::Reverted</code> is not a completed exit. <code>force_calls</code> is the bounded recent window; <code>force_call_count</code> is the total. Non-exhaustive <code>ArtConfig</code> and <code>ArtOutcome</code></td>
</tr>

<tr>
<td class="org-left"><code>DirectionCover</code></td>
<td class="org-left">Signed per-atom direction cover and arm picker for saddle searches from one minimum. Each active atom is its own centre with an even signed cover of <code>cover_points</code> on its sphere and an optional Gaussian kick on neighbours within <code>kick_radius</code>. The climb is <code>ArtSession</code>. <code>run</code>, or <code>suggest</code>, <code>run_arm</code> and <code>record</code> for a host loop. <code>coverage</code> is the <code>CoverageReport</code>: exact covering radius per centre, per arm records, distinct saddles against draws, frequency counts, and the next arm with its score. No Hessian. Non-exhaustive <code>DirectionCoverConfig</code> and <code>CoverBudget</code></td>
</tr>

<tr>
<td class="org-left"><code>IndexKSession</code></td>
<td class="org-left">Index-k minimum-mode step. <code>k = 1</code> is <code>MinModeSession</code>. <code>indexk_step</code> is one shot. <code>reflect_force</code> and <code>translated_force</code> are the subspace maps</td>
</tr>

<tr>
<td class="org-left"><code>LandscapeSession</code></td>
<td class="org-left">Downward pathway map. <code>PathwayMap</code> of <code>CriticalPoint</code> nodes and <code>LandscapeEdge</code> connections. <code>LandscapeConfig</code> bounds the nodes and the force evaluations. <code>LandscapeBoundKind</code> names the limit <code>SaddleError::LandscapeBound</code> reports</td>
</tr>

<tr>
<td class="org-left"><code>SolidDimer</code></td>
<td class="org-left">That search on atoms plus Jacobian-scaled cell strain. <code>CellMask::frozen</code> is the fixed-cell dimer. <code>stress</code> and <code>elastic_dipole</code> are <code>σ</code> and <code>V σ</code>. <code>SolidDimerConfig</code>, <code>IndexCheck</code>, <code>StressCheck</code>, <code>SolidDimerReport</code> and <code>IndexReport</code> are non-exhaustive</td>
</tr>

<tr>
<td class="org-left"><code>VineyardIndexReport</code></td>
<td class="org-left">The <code>IndexReport</code> of a certified <code>minimum</code> and <code>saddle</code>, returned by <code>certify_vineyard_pair</code>. The function needs each point stationary to a force tolerance, a minimum with no projected negative mode, a saddle with exactly one, and <code>IndexCheck::lowest</code> of at least two. Non-exhaustive</td>
</tr>

<tr>
<td class="org-left"><code>VineyardIndexError</code></td>
<td class="org-left">Why a pair is not Vineyard-ready: <code>Evaluation</code>, <code>MinimumForce</code>, <code>SaddleForce</code>, <code>Minimum</code> or <code>Saddle</code>, the last two with the lowest projected spectrum. Non-exhaustive</td>
</tr>

<tr>
<td class="org-left"><code>StrainResponse</code></td>
<td class="org-left">One hop. <code>barrier(ε)</code> is the lowest symmetry image of <code>ΔE0 - (P_saddle - P_minimum):ε</code>, or <code>NeedsSearch</code> past the validity bound. <code>ElasticDipole::from_stress</code> is <code>-V (σ_defect - σ_bulk)</code>. <code>refine</code> caches an index-1 solid dimer. <code>with_prefactor</code>, <code>with_variants</code>, <code>with_point_group</code>, <code>with_policy</code>, and <code>with_multipoles</code> attach options. <code>to_json</code> and <code>from_json</code> are the event record</td>
</tr>

<tr>
<td class="org-left"><code>defect_pair_rates</code></td>
<td class="org-left">One relaxed geometry and a hop catalogue. Solid dimer, index check, connectivity, fixed CI-NEB, PAFI <code>ΔF</code>, Bennett–Chandler <code>κ</code>. <code>k = κ ν exp(-ΔF/kT)</code>. <code>RateCatalogue</code> is <code>rgsaddle.defect_pair_rates</code></td>
</tr>

<tr>
<td class="org-left"><code>npt_lattice</code></td>
<td class="org-left">Isotropic Langevin dynamics with a Berendsen barostat on a crystal, with the kinetic pressure in the drive. <code>NptConfig</code> is non-exhaustive: start from <code>Default</code> and chain the <code>with_*</code> setters, including <code>with_repeats</code>, <code>with_maximum_steps</code>, <code>with_extension_steps</code> and <code>with_block_steps</code>. <code>NptReport</code> carries the mean lattice parameter and pressure with blocking standard errors. <code>pressure_converged</code> is an indicative screen, not a proof. See </td>
</tr>

<tr>
<td class="org-left"><code>PairNetwork</code></td>
<td class="org-left">States are orbits of (relative vector, orientation) under a caller's integer point group; hop classes are orbits of the unordered configuration pair and saddle director. <code>sites</code> takes a per-edge rate closure and returns <code>VolumeSite</code> rows and outer hops. <code>cubic_group</code> is O<sub>h</sub></td>
</tr>

<tr>
<td class="org-left"><code>matched_capture_radius</code></td>
<td class="org-left">Closes a site chain at its cutoff with capture <code>R/d</code> for an outer hop to <code>d</code>, <code>R</code> matched to the mean <code>P r</code> of the outermost sites, in closed form</td>
</tr>

<tr>
<td class="org-left"><code>SensitivityCover</code></td>
<td class="org-left">Which hop classes to compute so the rest of a rate model stay inside a tolerance. <code>sensitivity_cover(sensitivity, spread, tolerance)</code> ranks every class by <code>abs(d result / d ln k)</code> and returns one. <code>order</code> is the class indices, largest absolute sensitivity first. A non-finite sensitivity ranks first and is always covered. <code>covered</code> is how many leading entries of <code>order</code> to compute: the shortest prefix after which <code>spread</code> times the summed absolute sensitivity of the classes still out is below <code>tolerance</code>. <code>remaining</code> has one entry per rank, and the entry at that rank is <code>spread</code> times the summed absolute sensitivity of the classes after it, the worst coherent shift those model rates can still make if every one of them moves by <code>spread</code> in <code>ln k</code>. Compute the prefix of <code>order</code> whose length is <code>covered</code>, and leave the tail on the model. When the shift from every class together is already below <code>tolerance</code>, <code>covered</code> is zero. When no tail falls below <code>tolerance</code>, <code>covered</code> equals the length of <code>order</code> and every class is computed. <code>howto/defect-pair-rates</code> builds <code>sensitivity</code> by perturbing <code>ln k</code> in the rate closure, then passes <code>SurrogateCalibration::spread</code> back in as the next <code>spread</code></td>
</tr>

<tr>
<td class="org-left"><code>calibrate_surrogate</code></td>
<td class="org-left">Fits <code>x · β</code> to the residuals of a surrogate <code>ln k</code> against computed rates, weights <code>1/(σ² + τ²)</code> with the Paule–Mandel scatter <code>τ</code>. <code>SurrogateCalibration</code> gives <code>shift</code>, <code>shift_error</code>, <code>spread</code> of one more rate, and <code>draw</code> of the coefficients within their covariance</td>
</tr>

<tr>
<td class="org-left"><code>ArcPath</code></td>
<td class="org-left">PAFI path through ordered frames, a cubic spline over chord length, with its curvature and jerk. See </td>
</tr>

<tr>
<td class="org-left"><code>PafiConfig</code></td>
<td class="org-left">PAFI's overdamped Langevin sampling settings, with <code>target_error</code> and <code>max_rounds</code> for automatic extra rounds. See </td>
</tr>

<tr>
<td class="org-left"><code>pafi_profile</code></td>
<td class="org-left">Samples every plane of a path in parallel and integrates <code>dF/ds</code> (Swinburne and Marinica, Phys. Rev. Lett. 120, 135503 (2018), <a href="https://doi.org/10.1103/PhysRevLett.120.135503">https://doi.org/10.1103/PhysRevLett.120.135503</a>) to a <code>FreeEnergyProfile</code>. See </td>
</tr>

<tr>
<td class="org-left"><code>FreeEnergyProfile::reverse_barrier</code></td>
<td class="org-left"><code>F(dividing) - F(last)</code> with its trapezoid error: the reverse barrier of a path that ends in the product minimum</td>
</tr>

<tr>
<td class="org-left"><code>BalancedBarriers</code></td>
<td class="org-left">Both barriers of one path, made to differ by a known <code>end_gap</code> <code>F(last) - F(first)</code>. <code>forward</code> is the barrier above the first end, <code>reverse</code> the barrier above the last end (<code>end_gap</code> below <code>forward</code>), <code>error</code> the standard error of both, and <code>drift</code> what the integral adds to <code>end_gap</code>. Build it with <code>FreeEnergyProfile::balanced</code> or <code>balanced_barriers</code> when the two barriers are two estimates of one barrier, or with <code>FreeEnergyProfile::anchored</code> or <code>anchored_barriers</code> when the drift lies past the dividing surface and only the forward barrier should set the value. See </td>
</tr>

<tr>
<td class="org-left"><code>balanced_barriers</code></td>
<td class="org-left">The forward and reverse barriers of one path against a known <code>F(last) - F(first)</code>: their inverse-variance mean, the reverse barrier that gap below it, the <code>drift</code> the integral adds, and half of it in the error, returned as a <code>BalancedBarriers</code>. <code>FreeEnergyProfile::balanced</code> takes them from a profile</td>
</tr>

<tr>
<td class="org-left"><code>anchored_barriers</code></td>
<td class="org-left">The same inputs kept on the forward barrier and returned as a <code>BalancedBarriers</code>: the reverse barrier the gap below it, the error of the forward barrier, and the <code>drift</code> the barrier from the last plane adds. <code>FreeEnergyProfile::anchored</code> takes them from a profile</td>
</tr>

<tr>
<td class="org-left"><code>Hyperplane</code></td>
<td class="org-left">OH-TST plane: a point and a unit normal toward the product in mass-weighted coordinates. <code>from_min_mode</code> seeds it from a converged dimer or Lanczos session, preferring the vector of a converged saddle-index certificate over the session mode; <code>packed</code> and <code>from_packed</code> are its <code>[p; ν]</code> layout, the one of <code>PlaneSample::directional</code>; <code>from_cartesian</code> and <code>to_cartesian</code> convert; <code>orthogonal_to_mass_weighted_translations</code> drops rigid translations from the normal. See for every OH-TST type</td>
</tr>

<tr>
<td class="org-left"><code>LammpsHyperplaneSampler</code></td>
<td class="org-left">Opt-in lammpc backend for <code>HyperplaneSampler</code>. <code>LammpsSystem</code> owns the rgpot configuration and geometry; <code>LammpsSamplerConfig</code> owns the time step, friction, thermostat and excluded-mode count; records expose <code>lammps.thermostat</code> and <code>lammps.effective_degrees_of_freedom</code></td>
</tr>

<tr>
<td class="org-left"><code>MassWeighted</code></td>
<td class="org-left"><code>q = M^{1/2} x</code> through <code>MassMetric</code> over a Cartesian <code>PointSurface</code>; <code>mass_weighted_rigid_translations</code> returns the orthonormal rigid-translation rows in these coordinates; <code>FixedCell</code> holds the cell of a crystal fixed per temperature. <code>flux_mass</code> is <code>1 / Σ nᵢ²/mᵢ</code>, not the 2001 Eq. 24</td>
</tr>

<tr>
<td class="org-left"><code>sample_plane</code></td>
<td class="org-left">Plane-restricted Langevin sampling at <code>kT</code> by projected BAOAB or GHMC (<code>PlaneIntegrator</code>), reproducible from a seed. <code>PlaneSample</code> holds the Euclidean gradient blocks, the centroid torque and the virial, each with a block standard error; <code>HarmonicReference</code> adds Stein control variates</td>
</tr>

<tr>
<td class="org-left"><code>PlaneOptimizer</code></td>
<td class="org-left">Riemannian trust region (rgmin <code>minimize_rtr_gauged</code>) for the plane free energy on the product of the sphere of normals, off the held modes or within a rotation basis, and the offset along the normal; the pivot is a gauge moved to each accepted plane's centroid. <code>PlaneMetric</code> weights the sphere or makes the harmonic Hessian its <code>TangentMetric</code>. Norm-test sample growth, error-based stop. <code>PlaneOptimizerConfig</code> is non-exhaustive</td>
</tr>

<tr>
<td class="org-left"><code>basin_density</code></td>
<td class="org-left"><code>Z_H/Z_R</code> on a plane through a minimum from the mean force on parallel planes: windows per side that end on a cap, a turn or a width, a not-a-knot spline refined by leave-one-out, Stein control variates from the Hessian at the minimum, and a delta-method error. <code>reactant_density</code> keeps the 2001 crossing estimator, whose variance is infinite</td>
</tr>

<tr>
<td class="org-left"><code>PlanePath</code></td>
<td class="org-left">Planes between two planes for <code>reversible_work_along</code>: PAFI planes along a spline through a curve with clamped or natural ends (<code>curve</code>), the optimizer's accepted planes (<code>trace</code>), straight legs (<code>legs</code>), joined by <code>then</code></td>
</tr>

<tr>
<td class="org-left"><code>reversible_work</code></td>
<td class="org-left"><code>ΔA</code> along straight legs; <code>reversible_work_along</code> along any <code>PlanePath</code>. Simpson or trapezoid integration of <code>dA</code>, BAR between neighbours with their overlap, MBAR across all nodes (<code>with_mbar</code>), and <code>WorkEstimate</code> for the value the rate uses. <code>Transport</code> is the rigid map or the <code>GaussianTransport</code> optimal-transport map within the plane, shrunk toward the rigid map by a James–Stein factor; <code>ReversibleWork::warnings</code> names poor overlaps and fallbacks</td>
</tr>

<tr>
<td class="org-left"><code>ohtst_rate</code></td>
<td class="org-left"><code>k = sqrt(kT/2π) (Z_H/Z_R) exp(-ΔA/kT)</code> with unit mass in mass-weighted coordinates, and its error</td>
</tr>

<tr>
<td class="org-left"><code>transmission_coefficient</code></td>
<td class="org-left"><code>κ</code> from shots off a plane sample, with masses and Langevin friction or none: <code>Counting::Final</code> (Bennett–Chandler) or <code>Counting::FirstCrossing</code> (effective positive flux), end states by side (<code>EndState::Side</code>) or by quench into <code>Basins</code> with a Euclidean or <code>PermutationDistance</code>. <code>plane_transmission</code> shoots from a plane sample directly</td>
</tr>

<tr>
<td class="org-left"><code>anharmonic_correction</code></td>
<td class="org-left">First-order correction to harmonic TST and the first-order offset of the optimal plane, from Hessian actions only</td>
</tr>

<tr>
<td class="org-left"><code>Index1Session</code></td>
<td class="org-left">Restricted-step partitioned RFO, Powell or Bofill update. <code>Index1Config</code> is non-exhaustive. <code>with_sparse_pattern</code> selects a CPR seed</td>
</tr>

<tr>
<td class="org-left"><code>SparsityPattern</code></td>
<td class="org-left">Atom-level symmetric CSR from a linkcell cutoff walk. Diagonal blocks are implied and not stored as loops</td>
</tr>

<tr>
<td class="org-left"><code>Colouring</code></td>
<td class="org-left">Smallest-last distance-2 colouring and the max-degree lower bound</td>
</tr>

<tr>
<td class="org-left"><code>SparseHessian</code></td>
<td class="org-left">Frozen recovered blocks. Implements <code>ApplyHessian</code>. <code>matvec</code> and <code>apply_hessian</code> panic on a length mismatch. <code>to_dense</code> and <code>lowest_eigenpairs</code> export the matrix. <code>LowestEigenpairs</code> is non-exhaustive</td>
</tr>

<tr>
<td class="org-left"><code>SparseHessianOpts</code></td>
<td class="org-left">Non-exhaustive. <code>Default</code> and <code>with_difference</code>, <code>with_step</code>, <code>with_memory_budget</code>, <code>with_skin</code>. Default step 1e-4. The step carries \(O(h^2)\), \(O(\varepsilon_g / h)\), and the optimum near \((3 \varepsilon_g / \lvert g''' \rvert)^{1/3}\)</td>
</tr>

<tr>
<td class="org-left"><code>prefactor_hessian</code></td>
<td class="org-left">Direct CPR seed. <code>one_atom_hessian</code> is the per-atom baseline eOn uses when a potential has no finite cutoff</td>
</tr>

<tr>
<td class="org-left"><code>nichols_step</code></td>
<td class="org-left">i-PI Nichols shift from a supplied spectrum</td>
</tr>

<tr>
<td class="org-left"><code>restricted_prfo_displacement</code></td>
<td class="org-left">Baker restricted-step partitioned RFO from <code>H</code> and <code>g</code></td>
</tr>

<tr>
<td class="org-left"><code>SellaSaddleConfig</code></td>
<td class="org-left">Settings of the order-1 Sella session for partitioned rational function optimization</td>
</tr>

<tr>
<td class="org-left"><code>Cell</code></td>
<td class="org-left">3×3 lattice (rows are vectors), fractional minimum-image wrap</td>
</tr>

<tr>
<td class="org-left"><code>SaddleError</code></td>
<td class="org-left">Shape, surface, non-finite, solver, invalid, stalled, ART index-1 loss, not-equilibrated, and insufficient-effective-samples errors. The enum is unchanged by fixed-energy LI-NEB: a strict endpoint below <code>E</code> is <code>SaddleError::Invalid</code>, and <code>LineNebEndpointViolation</code> carries the endpoint (<code>first</code> or <code>last</code>), <code>V - E</code>, and the one-based step</td>
</tr>

<tr>
<td class="org-left"><code>Stencil</code></td>
<td class="org-left">Point builder. Dimer and Lanczos wrap <code>MinModeSession</code>. <code>Stencil::band</code> rejects band presets as not implemented. See </td>
</tr>

<tr>
<td class="org-left"><code>Chain</code></td>
<td class="org-left">Positions, ids, and roles. <code>evaluate</code> uses a fixed block of one image. <code>Mesh</code> remeshes (linear, Hermite, or not-a-knot spline), inserts, or windows with no force call</td>
</tr>

<tr>
<td class="org-left"><code>reparametrize</code></td>
<td class="org-left">Equal arc-length placement of a band, <code>Reparametrization::Linear</code> on the polyline or <code>NotAKnot</code> on the not-a-knot cubic spline. <code>RtrConfig::reparametrization</code> selects it for the trust-region band</td>
</tr>

<tr>
<td class="org-left"><code>Preset</code></td>
<td class="org-left"><code>Min</code>, <code>Dimer</code>, <code>Lanczos</code>, <code>Art</code>, <code>EigenFollow</code>, and the named band presets</td>
</tr>

<tr>
<td class="org-left"><code>ConnectConfig</code></td>
<td class="org-left">IDPP Lanczos climb and the COMPASS fallback. <code>optbench</code> sets the LJ38 force tolerance, eight Lanczos actions, relative tolerance 0.05, and quenched split nodes</td>
</tr>

<tr>
<td class="org-left"><code>sidpp_path</code></td>
<td class="org-left">Sequential IDPP (S-IDPP) initial path. <code>SidppConfig</code> holds the spring, the two force stops, and the quick-min step. <code>idpp_path</code> is the per-image IDPP start</td>
</tr>

<tr>
<td class="org-left"><code>idpp_path_fixed</code></td>
<td class="org-left">IDPP path with a frozen-atom mask. An empty mask is the same minimization as <code>idpp_path</code>: each interior image starts on the straight minimum-image line and relaxes toward linearly interpolated pair distances under <code>IdppConfig</code>. A true entry keeps that atom at the reactant coordinates on every interior image. A non-empty mask whose length is not the atom count is a shape error. A fixed atom that moves between the endpoints is rejected. The endpoints are copied through</td>
</tr>

<tr>
<td class="org-left"><code>sidpp_alternating_path</code></td>
<td class="org-left">Alternating IDPP path. This is not <code>sidpp_path</code>. <code>SidppAlternatingConfig</code> holds the gap fraction, the collective stops, and the optional pull. Images are inserted from both ends, starting at the reactant. A new image takes <code>alpha</code> of the minimum-image step from the current frontier toward the other side of the open gap. Frozen components of that step are cleared, so a frozen atom keeps the frontier coordinates, and the interior rows are then pinned to the reactant. Each insertion is followed by at most <code>short_iterations</code> quick-min steps, and a budget of no steps skips that stage. There is no further stage before the next insertion. The collective force on an interior image is the part of the IDPP force perpendicular to the simple tangent, plus <code>spring</code> times (the distance to the next image minus the distance to the previous image) along that tangent. The tangent is the unit minimum-image vector from the previous image to the next. The quick-min step is fixed, and it is not a field of <code>SidppAlternatingConfig</code>. A single velocity projection covers the whole interior, and <code>max_move</code> caps the per-atom displacement. A stage stops early when the largest absolute component of the free force is under <code>force_tol</code>. When every slot is filled and <code>reparametrize</code> is set, the band is placed at equal minimum-image arc length on the polyline, including a path with a single interior image. A final quick-min of at most <code>final_iterations</code> steps follows. A budget of no steps skips that final stage. When <code>pull_to_target</code> is set, each interior image is then minimized on the IDPP objective at its fraction of the finished path. <code>inverse_distance</code> selects the inverse-distance blend for every target, including that pull, and <code>pull</code> is the <code>IdppConfig</code> of the minimization. <code>n_images</code> counts the endpoints. An empty fixed mask moves every atom. The endpoints are copied through. The call is rejected when <code>alpha</code> is non-finite or outside the open unit interval, when <code>force_tol</code> or <code>spring</code> is non-finite or negative, when <code>max_move</code> is non-finite or not positive, when a fixed atom moves between the endpoints, or when <code>pull_to_target</code> is set and <code>pull</code> fails the <code>IdppConfig</code> checks. It also returns the shape and non-finite errors of <code>linear_path</code></td>
</tr>

<tr>
<td class="org-left"><code>sidpp_alternating_inverse_distance_path</code></td>
<td class="org-left">Inverse-distance form of <code>sidpp_alternating_path</code>. The call turns <code>inverse_distance</code> and <code>pull_to_target</code> on, then inserts, relaxes, and reparametrizes on that same loop, so neither flag can be left off for this entry point. Target pair distances at a path fraction are the reciprocal of the linear blend of the reciprocal endpoint distances. A short endpoint pair uses the IDPP floor in the denominators, and the distance is not left below that floor. The pull minimizes each interior image on its finished-path target and reads <code>pull</code>, which must be a usable <code>IdppConfig</code>. The errors are those of <code>sidpp_alternating_path</code>. This is not <code>geodesic_interpolate</code></td>
</tr>

<tr>
<td class="org-left"><code>geodesic_path</code></td>
<td class="org-left">Geodesic interpolation on a caller supplied chart (Zhu, Thompson, and Martinez, J. Chem. Phys. 150, 164103 (2019)). <code>GeodesicConfig</code> and <code>MorsePairChart</code> are non-exhaustive. <code>geodesic_interpolate</code> builds the path from the two endpoints</td>
</tr>

<tr>
<td class="org-left"><code>PathInitializer</code></td>
<td class="org-left">Initial path selector (<code>Linear</code>, <code>Idpp</code>, <code>Sidpp</code>, <code>SidppAlternating</code>, <code>Trics</code>) for <code>initialize_path</code> and <code>ConnectConfig::path_initializer</code> (<code>None</code> keeps the built-in steepest-descent IDPP band; <code>Some</code> forwards the chosen initializer to <code>initialize_path</code>). <code>Sidpp</code> calls <code>sidpp_path</code>. <code>SidppAlternating</code> calls <code>sidpp_alternating_inverse_distance_path</code> when <code>inverse_distance</code> is set and <code>sidpp_alternating_path</code> otherwise, both with an empty frozen-atom mask. <code>TricsStartConfig</code> pairs a <code>TricsSeed</code> (<code>Linear</code> or <code>Idpp</code>) with <code>TricsConfig</code></td>
</tr>

<tr>
<td class="org-left"><code>trics_path</code></td>
<td class="org-left">Translation-rotation internal coordinates with splicing (TRICS). Builds per-image <code>TricSystem</code> charts (<code>tric_systems</code>) over <code>TricPrimitive</code> coordinates with <code>TricConfig</code>, scores the stability <code>diagnostic_matrix</code> (re-exported at the crate root as <code>tric_diagnostic_matrix</code>), selects overlapping <code>TricsSegment</code> intervals (<code>tric_segments</code>), inverts each segment (<code>interpolate_tric_segment</code> with <code>TricInversionConfig</code>), and splices to <code>TricsConfig::splice_tolerance</code> (<code>TRICS_SPLICE_TOLERANCE_ANGSTROM</code>), returning a <code>TricsReport</code></td>
</tr>

<tr>
<td class="org-left"><code>connect</code></td>
<td class="org-left">Double-ended search. The callback returns energy and the Cartesian gradient. <code>evaluations</code> counts finite differences that decide a step or an index. The Lanczos acceptance does not build a dense Hessian</td>
</tr>
</tbody>
</table>

`SellaSaddleConfig` has options that follow Sella.
Each finite-difference diagonalization runs `symm=2` products in its residual test.
With `symmetrized_ritz` on, it takes its Ritz pairs from those products.
The option is off by default, and `SellaSaddleSession::set_symmetrized_ritz` sets it on a live session.
`small_system_atom_ceiling` bounds only the largest atom move, by Sella's `delta0` ceiling.
That bound applies from the first step of a Cartesian trust-region search with iterative diagonalization.
The search's radius `delta * n_free` must start at 1 or less.
The ceiling is on by default, and `SellaSaddleSession::set_small_system_atom_ceiling` sets it on a live session.


## Public Rust module index

Every public module exported by `src/lib.rs`, with its public types, free functions, and constants.

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Module</th>
<th scope="col" class="org-left">Public items</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left"><code>adaptive_ohtst</code></td>
<td class="org-left">Types: <code>AdaptiveOhTstConfig</code>, <code>AdaptiveOhTstEstimate</code>, <code>AdaptiveOhTstProfile</code>, <code>AdaptiveOhTstStatus</code>. Functions: <code>adaptive_ohtst_sample_count</code>, <code>adaptive_ohtst_string_rate</code>. Constants: <code>ADAPTIVE_OHTST_MAX_SAMPLES</code>, <code>ADAPTIVE_OHTST_PILOT_SAMPLES</code>, <code>ADAPTIVE_OHTST_SAMPLE_SAFETY</code>, <code>ADAPTIVE_OHTST_TARGET_LOG_ERROR</code></td>
</tr>

<tr>
<td class="org-left"><code>afir</code></td>
<td class="org-left">Functions: <code>afir_force</code>, <code>try_afir_force</code></td>
</tr>

<tr>
<td class="org-left"><code>art</code></td>
<td class="org-left">Types: <code>ArtBatch</code>, <code>ArtConfig</code>, <code>ArtDuplicate</code>, <code>ArtForceCall</code>, <code>ArtOutcome</code>, <code>ArtPhase</code>, <code>ArtReport</code>, <code>ArtReversibility</code>, <code>ArtSeed</code>, <code>ArtSession</code>, <code>ArtStatus</code>. Functions: <code>atom_mask</code>, <code>connection_verdict</code>, <code>full_mask</code>, <code>run_events</code>, <code>structure_distance</code></td>
</tr>

<tr>
<td class="org-left"><code>autoneb</code></td>
<td class="org-left">Types: <code>AutoNeb</code>, <code>AutoNebCut</code>, <code>AutoNebInterpolant</code>, <code>AutoNebReport</code>, <code>AutoNebSample</code>, <code>AutoNebStage</code>, <code>AutoNebStageLog</code>, <code>AutoNebStep</code>, <code>AutoNebTuning</code>. Functions: <code>worst_segment</code></td>
</tr>

<tr>
<td class="org-left"><code>band</code></td>
<td class="org-left">Types: <code>BandConfig</code>, <code>BandEvaluation</code>, <code>BandReport</code>, <code>BandSession</code>, <code>BandStatus</code>, <code>BandSurface</code>, <code>Cell</code>, <code>CiConfig</code>, <code>ExpBandPrecon</code>, <code>FeaNebReport</code>, <code>FeaNebSession</code>, <code>FreeEndConfig</code>, <code>FreeEndReport</code>, <code>FreeEndSide</code>, <code>GrowingStringConfig</code>, <code>GrowingStringPhase</code>, <code>GrowingStringReport</code>, <code>GrowingStringSession</code>, <code>MinModeRefine</code>, <code>SolidState</code>, <code>StringConfig</code>, <code>StringSession</code>, <code>ZoomConfig</code>. Functions: <code>climbing_image</code></td>
</tr>

<tr>
<td class="org-left"><code>bennett</code></td>
<td class="org-left">Types: <code>ShootingConfig</code>, <code>Transmission</code>. Functions: <code>bennett_chandler</code></td>
</tr>

<tr>
<td class="org-left"><code>block_cyclic</code></td>
<td class="org-left">Types: <code>BlockInertia</code>, <code>CyclicFactor</code>, <code>SolveReport</code>. Functions: <code>factor_cyclic</code>, <code>logdet_cyclic_blocks</code>, <code>logdet_cyclic_strict</code>. Constants: <code>SOLVE_REFINE_TOLERANCE</code>. Cyclic block-tridiagonal log absolute determinant and inertia. See and </td>
</tr>

<tr>
<td class="org-left"><code>cell_log</code></td>
<td class="org-left">Types: <code>CellChart</code>, <code>CellState</code>, <code>M3</code>, <code>PackedHess</code>. Functions: <code>expm_3x3</code>, <code>expm_frechet</code>, <code>logm_3x3</code>, <code>m3_det</code>, <code>m3_from_row9</code>, <code>m3_inv</code>, <code>m3_mul</code>, <code>m3_to_row9</code>, <code>transform_cell_block</code></td>
</tr>

<tr>
<td class="org-left"><code>connect</code></td>
<td class="org-left">Types: <code>ConnectConfig</code>, <code>ConnectProfile</code>, <code>ConnectReport</code>. Functions: <code>connect</code></td>
</tr>

<tr>
<td class="org-left"><code>constraints</code></td>
<td class="org-left">Types: <code>Constraints</code>, <code>Equality</code>, <code>InternalCounts</code></td>
</tr>

<tr>
<td class="org-left"><code>covalent</code></td>
<td class="org-left">Functions: <code>atomic_number</code>, <code>covalent_radius</code>, <code>element_symbol</code>. Constants: <code>CORDERO_2008_PM</code>, <code>LAST_CORDERO_Z</code></td>
</tr>

<tr>
<td class="org-left"><code>defect_pair</code></td>
<td class="org-left">Types: <code>BasinSaddle</code>, <code>CaptureRecord</code>, <code>CellMode</code>, <code>DefectPairConfig</code>, <code>DimerSeed</code>, <code>DirectionClass</code>, <code>FreeMigration</code>, <code>HopOutcome</code>, <code>HopTemplate</code>, <code>MatchedCapture</code>, <code>OuterHop</code>, <code>PairSetup</code>, <code>PeriodicSinkRate</code>, <code>PeriodicSinkSurvival</code>, <code>RateCatalogue</code>, <code>RateEntry</code>, <code>RecombinationVolume</code>, <code>SiteOutcome</code>, <code>SphereLanding</code>, <code>VolumePlacement</code>, <code>VolumeSite</code>. Functions: <code>attempt_frequency</code>, <code>capture_is_spontaneous</code>, <code>capture_is_spontaneous_within</code>, <code>classify_separation</code>, <code>debye_hit_probability</code>, <code>debye_matched_radius</code>, <code>debye_radius</code>, <code>debye_resistance</code>, <code>defect_pair_rates</code>, <code>descend_to_floor</code>, <code>effective_capture_radius</code>, <code>embed_free_migration</code>, <code>embed_sphere_return</code>, <code>kinetic_recombination_volume</code>, <code>matched_capture_radius</code>, <code>mean_absorption_time</code>, <code>pair_separation</code>, <code>periodic_capture_log_likelihood</code>, <code>periodic_sink_rate</code>, <code>periodic_sink_survival</code>, <code>product_tangent</code>, <code>seeded_saddles</code>, <code>segment_band</code>, <code>shell_pair_factor</code>, <code>smoluchowski_rate</code>, <code>sphere_hit_probability</code>, <code>spherical_return_probability</code>, <code>volume_from_frequency</code>. Constants: <code>FREE_CRYSTAL</code>, <code>SPONTANEOUS_BARRIER_EV</code></td>
</tr>

<tr>
<td class="org-left"><code>direction_cover</code></td>
<td class="org-left">Types: <code>ArmFailure</code>, <code>ArmId</code>, <code>ArmOutcome</code>, <code>ArmPolicy</code>, <code>ArmRecord</code>, <code>ArmSuggestion</code>, <code>ArmTrial</code>, <code>AtomCoverage</code>, <code>CoverBudget</code>, <code>CoverageReport</code>, <code>DirectionCover</code>, <code>DirectionCoverConfig</code>, <code>ProductState</code>, <code>SaddleEntry</code>. Functions: <code>covering_radius_s2</code>, <code>sphere_cover</code></td>
</tr>

<tr>
<td class="org-left"><code>eigensolve</code></td>
<td class="org-left">Types: <code>EigenDevice</code>, <code>ExpandKind</code>, <code>RitzDecomposition</code>. Functions: <code>dense_logabsdet</code>, <code>eigh_on</code>, <code>exact_eigh</code>, <code>exact_eigvalsh</code>, <code>expand</code>, <code>faer_eigh</code>, <code>lowest_on</code>, <code>rayleigh_ritz</code>, <code>rayleigh_ritz_iter</code>, <code>relative_zero_tolerance</code>, <code>solve_dense</code>, <code>solve_dense_matrix</code>, <code>spd_logdet</code></td>
</tr>

<tr>
<td class="org-left"><code>elastic</code></td>
<td class="org-left">Types: <code>ElasticMultipole</code>, <code>LocalField</code>, <code>Octupole</code>, <code>Quadrupole</code>, <code>StrainGradient</code>. Functions: <code>quadrupole_energy</code></td>
</tr>

<tr>
<td class="org-left"><code>elastic_dipole</code></td>
<td class="org-left">Types: <code>AnisotropicGreen</code>, <code>AnisotropicGreenConfig</code>, <code>CorrectedEnergy</code>, <code>CorrectedRate</code>, <code>DefectDipole</code>, <code>ElasticConstants</code>, <code>ElasticCorrection</code>, <code>ElasticPairState</code>, <code>ElasticRate</code>, <code>GreenSecondDerivative</code>, <code>InteractionEstimate</code>, <code>PeriodicElastic</code>, <code>PeriodicElasticConfig</code>, <code>Tensor4</code></td>
</tr>

<tr>
<td class="org-left"><code>error</code></td>
<td class="org-left">Types: <code>SaddleError</code></td>
</tr>

<tr>
<td class="org-left"><code>field</code></td>
<td class="org-left">Types: <code>DefectSource</code>, <code>DirectSolver</code>, <code>FieldSolver</code>, <code>FmmSolver</code>, <code>Kelvin</code></td>
</tr>

<tr>
<td class="org-left"><code>force</code></td>
<td class="org-left">Types: <code>ForceGate</code></td>
</tr>

<tr>
<td class="org-left"><code>force_match</code></td>
<td class="org-left">Types: <code>ForceMatchOpts</code>, <code>ForceMatchReport</code>, <code>ForceMatchedHessian</code>, <code>PairKind</code>. Functions: <code>bond_hessian</code>, <code>covalent_pairs</code>, <code>covalent_radius</code>, <code>fit_bond_ks</code>, <code>force_match</code>, <code>force_match_hessian</code>, <code>force_match_on</code>, <code>project_force_match</code>, <code>retract_force_match</code>, <code>transport_force_match</code></td>
</tr>

<tr>
<td class="org-left"><code>geom</code></td>
<td class="org-left">Types: <code>SellaGeom</code>, <code>TrustSchedule</code>. Functions: <code>trust_update</code>, <code>u_t_h_u</code>, <code>u_t_vec</code>, <code>u_vec</code>, <code>update_trust</code></td>
</tr>

<tr>
<td class="org-left"><code>gpu</code></td>
<td class="org-left">Types: <code>GpuPolicy</code>. Functions: <code>clear_oom_floor</code>, <code>cuda_available</code>, <code>cuda_device</code>, <code>fail_next_cuda_claim</code>, <code>gpu_eigh</code>, <code>gpu_eigh_env</code>, <code>gpu_eigh_t</code>, <code>gpu_eigh_with</code>, <code>gpu_ok</code>, <code>gpu_ok_env</code>, <code>gpu_project</code>, <code>gpu_project_env</code>, <code>gpu_project_with</code>, <code>gpu_qr</code>, <code>gpu_qr_env</code>, <code>gpu_qr_with</code>, <code>lock_oom_for_test</code>, <code>oom_floor</code>, <code>record_oom</code>, <code>to_gpu</code>, <code>to_gpu_matrix</code>, <code>to_gpu_view</code>. Constants: <code>DEFAULT_MIN_DIM</code>, <code>GPU_MIN_DIM</code></td>
</tr>

<tr>
<td class="org-left"><code>idpp</code></td>
<td class="org-left">Types: <code>IdppConfig</code>, <code>PathInitializer</code>, <code>SidppAlternatingConfig</code>, <code>SidppConfig</code>, <code>TricsSeed</code>, <code>TricsStartConfig</code>. Functions: <code>idpp_path</code>, <code>idpp_path_fixed</code>, <code>initialize_path</code>, <code>linear_path</code>, <code>sidpp_alternating_inverse_distance_path</code>, <code>sidpp_alternating_path</code>, <code>sidpp_path</code></td>
</tr>

<tr>
<td class="org-left"><code>indexk</code></td>
<td class="org-left">Types: <code>BlockEigenKind</code>, <code>HessianProduct</code>, <code>IndexKConfig</code>, <code>IndexKReport</code>, <code>IndexKSession</code>, <code>WithHessian</code>. Functions: <code>indexk_step</code>, <code>reflect_force</code>, <code>translated_force</code></td>
</tr>

<tr>
<td class="org-left"><code>internal</code></td>
<td class="org-left">Types: <code>CartAxis</code>, <code>Displacement</code>, <code>InternalSlot</code>, <code>Rotation</code>, <code>Translation</code>. Functions: <code>pack_cart</code>, <code>seam_nrm2</code></td>
</tr>

<tr>
<td class="org-left"><code>io</code></td>
<td class="org-left">Types: <code>ConImage</code>, <code>ConSpecies</code>, <code>MolecularFrame</code>. Functions: <code>frame_from_con</code>, <code>frame_from_con_nth</code>, <code>frames_from_con</code>, <code>store_band</code>, <code>write_band</code>, <code>write_restart</code></td>
</tr>

<tr>
<td class="org-left"><code>irc</code></td>
<td class="org-left">Types: <code>IrcConfig</code>, <code>IrcDirection</code>, <code>IrcKind</code>, <code>IrcReport</code>, <code>IrcSession</code></td>
</tr>

<tr>
<td class="org-left"><code>kappa</code></td>
<td class="org-left">Types: <code>KappaDimerConfig</code>, <code>KappaDimerForce</code>. Functions: <code>kappa_dimer_force</code></td>
</tr>

<tr>
<td class="org-left"><code>landscape</code></td>
<td class="org-left">Types: <code>CriticalPoint</code>, <code>LandscapeBoundKind</code>, <code>LandscapeConfig</code>, <code>LandscapeEdge</code>, <code>LandscapeSession</code>, <code>PathwayMap</code>. Constants: <code>DEFAULT_DUPLICATE_DISTANCE</code>, <code>DEFAULT_LANDSCAPE_MAX_EVALUATIONS</code>, <code>DEFAULT_LANDSCAPE_MAX_NODES</code>, <code>DEFAULT_LANDSCAPE_MAX_STEPS</code>, <code>DEFAULT_LANDSCAPE_PUSH</code></td>
</tr>

<tr>
<td class="org-left"><code>lammps</code></td>
<td class="org-left">Types: <code>LammpsHyperplaneSampler</code>, <code>LammpsSamplerConfig</code>, <code>LammpsSystem</code>, <code>LammpsThermodynamicReference</code>, <code>LammpsThermostatKind</code>, <code>LammpsThermostatRecord</code></td>
</tr>

<tr>
<td class="org-left"><code>lineb</code></td>
<td class="org-left">Types: <code>Continuation</code>, <code>HqtstRate</code>, <code>LinebCalls</code>, <code>LinebConfig</code>, <code>LinebPath</code>, <code>LinebReport</code>, <code>LinebSession</code>, <code>LinebStatus</code>, <code>Masses</code>. Functions: <code>continue_from_mep</code>, <code>continue_from_mep_with_observer</code>, <code>continue_from_path_with_observer</code>, <code>hqtst_rate</code>, <code>hqtst_rate_interpolated_hessian</code></td>
</tr>

<tr>
<td class="org-left"><code>linalg</code></td>
<td class="org-left">Types: <code>NumericalHessian</code>. Functions: <code>hessian_vec_sign</code>, <code>lift_through</code>, <code>modified_gram_schmidt</code>, <code>numerical_hvp</code>, <code>numerical_hvp_on</code>, <code>numerical_hvp_proj</code>, <code>project_hvp</code>, <code>project_through</code>, <code>retract_hvp</code>, <code>symmetrize_vt_av</code>, <code>transport_hvp</code></td>
</tr>

<tr>
<td class="org-left"><code>line_neb</code></td>
<td class="org-left">Types: <code>EndpointBelowEnergy</code>, <code>FixedEnergySeed</code>, <code>LineNebConfig</code>, <code>LineNebEndpointViolation</code>, <code>LineNebReport</code>, <code>LineNebSession</code>, <code>LineNebStatus</code>. Functions: <code>fixed_energy_action</code>, <code>fixed_energy_action_force</code>, <code>fixed_energy_mep_seed</code></td>
</tr>

<tr>
<td class="org-left"><code>mass_metric</code></td>
<td class="org-left">Types: <code>MassMetric</code>, <code>MetricCovector</code>, <code>MetricPoint</code>, <code>MetricTangent</code></td>
</tr>

<tr>
<td class="org-left"><code>mic</code></td>
<td class="org-left">Types: <code>Cell</code>. Functions: <code>wrap_difference</code></td>
</tr>

<tr>
<td class="org-left"><code>minmode</code></td>
<td class="org-left">Types: <code>ArtNConfig</code>, <code>FiniteDifference</code>, <code>KrylovBlock</code>, <code>MinModeConfig</code>, <code>MinModeKind</code>, <code>MinModeReport</code>, <code>MinModeSession</code>, <code>MinModeStatus</code>, <code>ModeEstimate</code>, <code>PointSurface</code>, <code>RaisedStep</code>. Functions: <code>halve_raised_energy</code>, <code>rotation_force_error</code>. Constants: <code>DEFAULT_ROTATION_ANGLE_TOL</code>, <code>MODE_RESET_ANGLE</code></td>
</tr>

<tr>
<td class="org-left"><code>nichols</code></td>
<td class="org-left">Types: <code>HessianUpdate</code>, <code>Index1Config</code>, <code>Index1Report</code>, <code>Index1Session</code>, <code>Index1Status</code>, <code>NicholsMode</code>. Functions: <code>bofill_update</code>, <code>cap_max_abs</code>, <code>nichols_displacement</code>, <code>nichols_step</code>, <code>powell_update</code></td>
</tr>

<tr>
<td class="org-left"><code>npt</code></td>
<td class="org-left">Types: <code>NptConfig</code>, <code>NptReport</code>. Functions: <code>npt_lattice</code></td>
</tr>

<tr>
<td class="org-left"><code>occupancy</code></td>
<td class="org-left">Types: <code>SiteOccupancy</code>. Functions: <code>site_occupancy</code></td>
</tr>

<tr>
<td class="org-left"><code>ocineb</code></td>
<td class="org-left">Types: <code>OcinebConfig</code>, <code>OcinebPhase</code>, <code>OcinebReport</code>, <code>OcinebSession</code>, <code>OcinebStatus</code>. Functions: <code>backoff_threshold</code>, <code>initial_threshold</code>, <code>mode_alignment</code>, <code>should_trigger</code>, <code>success_threshold</code>, <code>walk_helped</code>. Constants: <code>MIN_ALIGNMENT</code></td>
</tr>

<tr>
<td class="org-left"><code>ohtst</code></td>
<td class="org-left">Types: <code>AnalyticSurfaceDynamics</code>, <code>AnalyticSurfaceSampler</code>, <code>AnalyticThermodynamicSampler</code>, <code>AnharmonicConfig</code>, <code>AnharmonicCorrection</code>, <code>BasinConfig</code>, <code>BasinDensity</code>, <code>BasinEnd</code>, <code>Basins</code>, <code>Counting</code>, <code>EndState</code>, <code>Endpoint</code>, <code>ExactGaussianPlaneSampler</code>, <code>FixedCell</code>, <code>GaussianTransport</code>, <code>HarmonicExpansion</code>, <code>HarmonicReference</code>, <code>Hyperplane</code>, <code>HyperplaneDynamics</code>, <code>HyperplaneObservable</code>, <code>LangevinConfig</code>, <code>MassWeighted</code>, <code>OhTstRate</code>, <code>PermutationDistance</code>, <code>PlaneDensity</code>, <code>PlaneIntegrator</code>, <code>PlaneMetric</code>, <code>PlaneOptimizer</code>, <code>PlaneOptimizerConfig</code>, <code>PlaneOptimizerStatus</code>, <code>PlaneOptimum</code>, <code>PlanePath</code>, <code>PlaneSample</code>, <code>PlaneTrace</code>, <code>QuadraticModel</code>, <code>QuenchConfig</code>, <code>QuenchedTransitionRegion</code>, <code>RateMultiplicity</code>, <code>ReactantDensity</code>, <code>ReversibleWork</code>, <code>ReversibleWorkConfig</code>, <code>StateDistance</code>, <code>ThermodynamicDerivative</code>, <code>ThermodynamicIntegrationConfig</code>, <code>ThermodynamicIntegrationEstimate</code>, <code>ThermodynamicIntegrationNode</code>, <code>ThermodynamicSampler</code>, <code>ThermodynamicSurfaceSampler</code>, <code>TransmissionCoefficient</code>, <code>TransmissionConfig</code>, <code>TransmissionEstimate</code>, <code>Transport</code>, <code>Uncontrolled</code>, <code>WorkEstimate</code>, <code>WorkNode</code>, <code>EquilibrationReport</code>. Functions: <code>anharmonic_correction</code>, <code>basin_density</code>, <code>flux_mass</code>, <code>harmonic_energy_at</code>, <code>mass_weighted_rigid_translations</code>, <code>ohtst_rate</code>, <code>optimize_plane</code>, <code>plane_transmission</code>, <code>reactant_density</code>, <code>restricted_thermodynamic_integration</code>, <code>reversible_work</code>, <code>reversible_work_along</code>, <code>reversible_work_along_on_surface</code>, <code>reversible_work_on_surface</code>, <code>sample_plane</code>, <code>sample_plane_with_sampler</code>, <code>thermodynamic_integration</code>, <code>transmission_coefficient</code>, <code>transmission_coefficient_on_surface</code>. Constants: <code>THERMODYNAMIC_DERIVATIVE</code></td>
</tr>

<tr>
<td class="org-left"><code>pafi</code></td>
<td class="org-left">Types: <code>ArcPath</code>, <code>BalancedBarriers</code>, <code>FixedEnergyTrajectory</code>, <code>FreeEnergyProfile</code>, <code>HyperplaneAverage</code>, <code>PafiConfig</code>, <code>PathFrame</code>, <code>PathGeometry</code>, <code>TrajectoryPoint</code>. Functions: <code>anchored_barriers</code>, <code>arc_at_time</code>, <code>average_hyperplane</code>, <code>average_hyperplane_with_sampler</code>, <code>balanced_barriers</code>, <code>integrate_profile</code>, <code>pafi_profile</code>, <code>zero_temperature_barrier</code></td>
</tr>

<tr>
<td class="org-left"><code>pair_network</code></td>
<td class="org-left">Types: <code>Mat3i</code>, <code>PairConfig</code>, <code>PairEdge</code>, <code>PairExit</code>, <code>PairHop</code>, <code>PairMove</code>, <code>PairNetwork</code>, <code>PairStateClass</code>, <code>PairTarget</code>, <code>SensitivityCover</code>, <code>Vec3i</code>. Functions: <code>cubic_group</code>, <code>director</code>, <code>norm2</code>, <code>sensitivity_cover</code></td>
</tr>

<tr>
<td class="org-left"><code>pes</code></td>
<td class="org-left">Types: <code>CartesianHessian</code>, <code>CartesianPes</code>, <code>HessUpdate</code></td>
</tr>

<tr>
<td class="org-left"><code>pes_internal</code></td>
<td class="org-left">Types: <code>CellCartesianPes</code>, <code>CellInternalPes</code>, <code>HessianCarry</code>, <code>InternalPes</code>, <code>RebuildReason</code>, <code>RebuildReport</code>, <code>SellaPes</code>. Functions: <code>niggli_reduce_cell</code>, <code>niggli_reduce_vectors</code>, <code>place_perp_dummy</code></td>
</tr>

<tr>
<td class="org-left"><code>placement</code></td>
<td class="org-left">Types: <code>SitePlacement</code>. Functions: <code>push_past</code>, <code>translate_onto_sites</code></td>
</tr>

<tr>
<td class="org-left"><code>prfo</code></td>
<td class="org-left">Types: <code>PartitionedRationalFunctionOptimization</code>, <code>PrfoKind</code>. Functions: <code>partitioned_rfo_eigen</code>, <code>prfo_get_s</code>, <code>prfo_stepper</code>, <code>prfo_trust_region</code>, <code>restricted_partitioned_rfo</code>, <code>restricted_prfo_displacement</code>. Constants: <code>SYNONYMS</code></td>
</tr>

<tr>
<td class="org-left"><code>prfo_restricted</code></td>
<td class="org-left">Types: <code>PrfoKind</code>. Functions: <code>partitioned_rfo_eigen</code>, <code>restricted_partitioned_rfo</code>, <code>restricted_prfo_displacement</code></td>
</tr>

<tr>
<td class="org-left"><code>projection</code></td>
<td class="org-left">Types: <code>ProjectionKind</code>. Functions: <code>climbing_image_force</code>, <code>dneb_component</code>, <code>force_perp</code></td>
</tr>

<tr>
<td class="org-left"><code>projector</code></td>
<td class="org-left">Types: <code>Hyperplane</code>, <code>HyperplaneCoordinates</code>, <code>HyperplaneEnsemble</code>, <code>HyperplaneModes</code>, <code>HyperplaneObservation</code>, <code>HyperplaneSampleBudget</code>, <code>HyperplaneSampler</code>, <code>ManifoldTangent</code>, <code>ProjectedConstraint</code>, <code>TangentProjector</code>, <code>UnitNormal</code></td>
</tr>

<tr>
<td class="org-left"><code>qn</code></td>
<td class="org-left">Types: <code>HessUpdate</code>, <code>QuasiNewton</code>, <code>StepperKind</code>. Functions: <code>get_stepper</code>, <code>matches</code>, <code>qn_get_s</code>, <code>qn_restricted</code>, <code>qn_retract</code>, <code>qn_tangent</code>, <code>retract_qn</code>, <code>symmetrize_y</code>, <code>symmetrize_y_cols</code>, <code>symmetrize_y_pair</code>, <code>update_h</code>, <code>update_h_ms</code>, <code>update_hessian</code>, <code>update_hessian_cols</code>. Constants: <code>ALPHA0</code>, <code>ALPHA_MAX</code>, <code>ALPHA_MIN</code>, <code>NEWTON_SAFE</code>, <code>SLOPE</code>, <code>SYNONYMS</code></td>
</tr>

<tr>
<td class="org-left"><code>region</code></td>
<td class="org-left">Types: <code>AffineSymmetry</code>, <code>BoundaryViolation</code>, <code>CrystalSymmetryConfig</code>, <code>CrystalSymmetryOperation</code>, <code>CrystalSymmetryRegion</code>, <code>PathSlab</code>, <code>PeriodicSymmetryOperations</code>, <code>PlanarBoundary</code>, <code>Region</code>, <code>RegionIntersection</code>, <code>SymmetryOperations</code>, <code>SymmetrySource</code>, <code>SymmetryUniqueRegion</code>. Functions: <code>reflect_velocity</code></td>
</tr>

<tr>
<td class="org-left"><code>relax</code></td>
<td class="org-left">Types: <code>RelaxConfig</code>, <code>Relaxation</code>. Functions: <code>max_atom_force</code>, <code>relax_to_tolerance</code>. Constants: <code>DEFAULT_FORCE_TOL</code>, <code>DEFAULT_MAX_STEPS</code></td>
</tr>

<tr>
<td class="org-left"><code>reparam</code></td>
<td class="org-left">Types: <code>Reparametrization</code>, <code>Spline</code>, <code>SplineEnds</code>, <code>SplineSample</code>. Functions: <code>reparametrize</code>, <code>segment_lengths</code></td>
</tr>

<tr>
<td class="org-left"><code>restricted</code></td>
<td class="org-left">Types: <code>InternalWeights</code>, <code>MaxInternalStep</code>, <code>RestrictedAtomicStep</code>, <code>RestrictedKind</code>, <code>TrustRegion</code>. Functions: <code>cons_max</code>, <code>mis_clip</code>, <code>pack_weights</code>, <code>ras_cons</code>, <code>weights_for_equalities</code>, <code>weights_for_equalities_with</code>. Constants: <code>RAS_SYNONYMS</code>, <code>TRUST_SYNONYMS</code></td>
</tr>

<tr>
<td class="org-left"><code>rfo</code></td>
<td class="org-left">Types: <code>RationalFunctionOptimization</code>. Functions: <code>rfo_stepper</code>. Constants: <code>SYNONYMS</code></td>
</tr>

<tr>
<td class="org-left"><code>rtr</code></td>
<td class="org-left">Types: <code>BandForces</code>, <code>BandRtr</code>, <code>RtrConfig</code>, <code>RtrRadius</code>, <code>RtrReport</code>, <code>TcgResult</code>, <code>TcgStop</code>. Functions: <code>band_forces</code>, <code>reparametrize_equal_arc</code>, <code>reparametrize_equal_arc_in_cell</code>, <code>truncated_cg</code></td>
</tr>

<tr>
<td class="org-left"><code>saddle_index</code></td>
<td class="org-left">Types: <code>SaddleEscape</code>, <code>SaddleIndex</code>, <code>SaddleIndexCheck</code>. Functions: <code>saddle_index</code>. Constants: <code>NEGATIVE_CURVATURE_TOL</code></td>
</tr>

<tr>
<td class="org-left"><code>samd</code></td>
<td class="org-left">Types: <code>SamdConfig</code>, <code>SamdReport</code>, <code>SamdSession</code>, <code>VelocitySofteningConfig</code>, <code>VelocitySofteningReport</code>. Functions: <code>project_velocity</code>, <code>retract_samd</code>, <code>soften_velocity_on</code>, <code>t_exp</code>, <code>t_linear</code>, <code>transport_velocity</code>, <code>verlet_increment</code></td>
</tr>

<tr>
<td class="org-left"><code>seed_direction</code></td>
<td class="org-left">Types: <code>ModeSweepDirectionPlan</code>, <code>SoftModeDirectionPlan</code>. Functions: <code>hessian_mode_sweep_directions</code>, <code>mode_sweep_schedule</code>, <code>soft_mode_then_full_directions</code>, <code>spectral_gap_rank</code></td>
</tr>

<tr>
<td class="org-left"><code>sella_min</code></td>
<td class="org-left">Types: <code>SellaMinConfig</code>, <code>SellaMinReport</code>, <code>SellaMinSession</code></td>
</tr>

<tr>
<td class="org-left"><code>sella_saddle</code></td>
<td class="org-left">Types: <code>SellaSaddleConfig</code>, <code>SellaSaddleReport</code>, <code>SellaSaddleSession</code></td>
</tr>

<tr>
<td class="org-left"><code>solid_dimer</code></td>
<td class="org-left">Types: <code>CellMask</code>, <code>IndexCheck</code>, <code>IndexReport</code>, <code>JointChart</code>, <code>SolidDimer</code>, <code>SolidDimerConfig</code>, <code>SolidDimerReport</code>, <code>StressCheck</code>, <code>VineyardIndexError</code>, <code>VineyardIndexReport</code>. Functions: <code>certify_vineyard_pair</code>, <code>check_cauchy_stress</code>, <code>finite_difference_cauchy_stress</code>, <code>index_at</code>, <code>virial</code>. Constants: <code>STRAIN_VOIGT</code></td>
</tr>

<tr>
<td class="org-left"><code>solid_prefactor</code></td>
<td class="org-left">Types: <code>JointHessian</code>, <code>JointHessianConfig</code>, <code>JointVineyard</code>, <code>JointVineyardConfig</code>. Functions: <code>joint_hessian</code>, <code>joint_vineyard_prefactor</code></td>
</tr>

<tr>
<td class="org-left"><code>solid_state</code></td>
<td class="org-left">Functions: <code>cartesian_step</code>, <code>cell_neb_force</code>, <code>joint_displacement</code>, <code>orient_lower_triangular</code>, <code>solid_state_enthalpy</code>, <code>solid_state_jacobian</code></td>
</tr>

<tr>
<td class="org-left"><code>sparse_hessian</code></td>
<td class="org-left">Types: <code>Colouring</code>, <code>LowestEigenpairs</code>, <code>SparseHessian</code>, <code>SparseHessianOpts</code>, <code>SparsityPattern</code>. Functions: <code>distance2_colouring</code>, <code>eval_displacements</code>, <code>one_atom_hessian</code>, <code>prefactor_hessian</code>, <code>sparse_hessian</code></td>
</tr>

<tr>
<td class="org-left"><code>spring</code></td>
<td class="org-left">Types: <code>SpringForces</code>, <code>SpringKind</code>. Functions: <code>energy_weighted_constants</code>, <code>onsager_machlup_displacements</code>. Constants: <code>OM_RECIP_EPS</code></td>
</tr>

<tr>
<td class="org-left"><code>ssneb</code></td>
<td class="org-left">Types: <code>SolidImage</code>, <code>SolidNeb</code>. Functions: <code>joint_coordinate</code></td>
</tr>

<tr>
<td class="org-left"><code>stencil</code></td>
<td class="org-left">Types: <code>Action</code>, <code>ArtControl</code>, <code>BandBuilder</code>, <code>Chain</code>, <code>Climb</code>, <code>Controller</code>, <code>Coupling</code>, <code>CouplingForces</code>, <code>Delta</code>, <code>Euclidean</code>, <code>EvalPool</code>, <code>Fixed</code>, <code>FromBand</code>, <code>FromPoint</code>, <code>ImageId</code>, <code>Mesh</code>, <code>MeshEvent</code>, <code>Metric</code>, <code>Point</code>, <code>PointBuilder</code>, <code>Preset</code>, <code>Report</code>, <code>Role</code>, <code>Sides</code>, <code>Stencil</code>, <code>StencilReport</code>, <code>StencilStatus</code>, <code>StencilView</code>, <code>Surface</code>, <code>TangentRule</code>, <code>Triad</code>. Functions: <code>ascend</code>, <code>coupling_forces</code>, <code>coupling_forces_into</code>, <code>curvature_along</code>, <code>perp</code>, <code>perp_into</code>, <code>reflect</code>, <code>reflect_into</code>, <code>unit_tangent</code>, <code>unit_tangent_into</code>. Constants: <code>BAND_PRESET_SEAM</code>, <code>EVAL_BLOCK</code></td>
</tr>

<tr>
<td class="org-left"><code>strain_rate</code></td>
<td class="org-left">Types: <code>CurvatureBound</code>, <code>ElasticDipole</code>, <code>Estimate</code>, <code>HopVariant</code>, <code>SaddleSeed</code>, <code>SearchReason</code>, <code>Strain</code>, <code>StrainResponse</code>, <code>TranslationalSplit</code>, <code>Trust</code>, <code>TrustPolicy</code>. Functions: <code>activation_volume</code>, <code>angular_scale_rad_s</code>, <code>arrhenius</code>, <code>compliance_from_stiffness</code>, <code>cubic_rotations</code>, <code>double_dot</code>, <code>elastic_dipole</code>, <code>from_voigt</code>, <code>rotation_x</code>, <code>split_translational_modes</code>, <code>symmetrize</code>, <code>transform_tensor</code>, <code>vineyard_prefactor</code>, <code>voigt</code>. Constants: <code>BOLTZMANN_EV_PER_K</code>, <code>LINEAR_FOLD_EV</code>, <code>NEGATIVE_CURVATURE</code>, <code>VOIGT</code></td>
</tr>

<tr>
<td class="org-left"><code>tangent</code></td>
<td class="org-left">Types: <code>TangentKind</code>, <code>SplineTangents</code>. Functions: <code>compute_tangent</code>, <code>spline_tangents</code>, <code>spline_resample</code></td>
</tr>

<tr>
<td class="org-left"><code>topology</code></td>
<td class="org-left">Types: <code>AtomMap</code>, <code>DummyHost</code>, <code>InternalTopology</code>, <code>LinearBend</code>, <code>PlanarWatch</code>, <code>TopologyCheck</code>, <code>TopologyConfig</code>. Functions: <code>check_chart</code>. Constants: <code>PAPER_SV_TOL</code></td>
</tr>

<tr>
<td class="org-left"><code>trics</code></td>
<td class="org-left">Types: <code>TricConfig</code>, <code>TricInversionConfig</code>, <code>TricPrimitive</code>, <code>TricSystem</code>, <code>TricsConfig</code>, <code>TricsReport</code>, <code>TricsSegment</code>. Functions: <code>diagnostic_matrix</code>, <code>interpolate_tric_segment</code>, <code>tric_segments</code>, <code>tric_systems</code>, <code>trics_path</code>. Constants: <code>TRICS_SPLICE_TOLERANCE_ANGSTROM</code></td>
</tr>

<tr>
<td class="org-left"><code>trust</code></td>
<td class="org-left">Types: <code>RestrictedStep</code>, <code>TrustRegion</code></td>
</tr>

<tr>
<td class="org-left"><code>vocn</code></td>
<td class="org-left">Types: <code>Found</code>, <code>Primitive</code>, <code>VocnFound</code>, <code>VocnPrimitive</code></td>
</tr>
</tbody>
</table>


## Configuration fields and `with_*` builders

Every public configuration struct, its public fields, and the corresponding chainable `with_*` builder method.

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Configuration struct</th>
<th scope="col" class="org-left">Fields and builders</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left"><code>AdaptiveOhTstConfig</code></td>
<td class="org-left"><code>max_iterations</code> (<code>with_max_iterations</code>), <code>max_planes</code> (<code>with_max_planes</code>), <code>sampler</code> (<code>with_sampler</code>, <code>with_thermal</code>)</td>
</tr>

<tr>
<td class="org-left"><code>AffineSymmetry</code></td>
<td class="org-left"><code>linear</code> (<code>with_linear</code>), <code>translation</code> (<code>with_translation</code>)</td>
</tr>

<tr>
<td class="org-left"><code>AnisotropicGreenConfig</code></td>
<td class="org-left"><code>angular_points</code> (<code>with_angular_points</code>)</td>
</tr>

<tr>
<td class="org-left"><code>AnharmonicConfig</code></td>
<td class="org-left"><code>hvp_step</code> (<code>with_hvp_step</code>), <code>krylov</code> (<code>with_krylov</code>), <code>krylov_tolerance</code> (<code>with_krylov_tolerance</code>), <code>probes</code> (<code>with_probes</code>), <code>seed</code> (<code>with_seed</code>), <code>step</code> (<code>with_step</code>)</td>
</tr>

<tr>
<td class="org-left"><code>ArtConfig</code></td>
<td class="org-left"><code>basin_rmsd</code> (<code>with_basin_rmsd</code>), <code>duplicate_rmsd</code> (<code>with_duplicate_rmsd</code>), <code>eigen_threshold</code> (<code>with_eigen_threshold</code>), <code>force_call_window</code> (<code>with_force_call_window</code>), <code>index_one_recovery_attempts</code> (<code>with_index_one_recovery_attempts</code>), <code>max_climbs</code> (<code>with_max_climbs</code>), <code>max_pushes</code> (<code>with_max_pushes</code>), <code>mode</code> (<code>with_mode</code>), <code>mode_overlap_tol</code> (<code>with_mode_overlap_tol</code>), <code>permute_identical</code> (<code>with_permute_identical</code>), <code>perpendicular</code> (<code>with_perpendicular</code>), <code>push_length</code> (<code>with_push_length</code>), <code>push_over</code> (<code>with_push_over</code>), <code>push_step</code> (<code>with_push_step</code>), <code>quench</code> (<code>with_quench</code>), <code>saddle</code> (<code>with_saddle</code>)</td>
</tr>

<tr>
<td class="org-left"><code>ArtNConfig</code></td>
<td class="org-left"><code>activation_relaxations</code> (<code>with_activation_relaxations</code>), <code>basin_push</code> (<code>with_basin_push</code>), <code>basin_relaxations</code> (<code>with_basin_relaxations</code>), <code>curvature_floor</code> (<code>with_curvature_floor</code>), <code>eigenvalue_threshold</code> (<code>with_eigenvalue_threshold</code>), <code>lanczos_actions</code> (<code>with_lanczos_actions</code>), <code>lanczos_start</code> (<code>with_lanczos_start</code>), <code>relaxation_step</code> (<code>with_relaxation_step</code>)</td>
</tr>

<tr>
<td class="org-left"><code>AutoNebTuning</code></td>
<td class="org-left"><code>climb</code> (<code>with_climb</code>), <code>climb_max_steps</code> (<code>with_climb_max_steps</code>), <code>full_band</code> (<code>with_full_band</code>), <code>interpolant</code> (<code>with_interpolant</code>), <code>local_force_tol</code> (<code>with_local_force_tol</code>), <code>local_max_steps</code> (<code>with_local_max_steps</code>), <code>n_max</code> (<code>with_n_max</code>), <code>n_simul</code> (<code>with_n_simul</code>), <code>record_samples</code> (<code>with_record_samples</code>), <code>space_energy_ratio</code> (<code>with_space_energy_ratio</code>), <code>threads</code> (<code>with_threads</code>), <code>trics</code> (<code>with_trics</code>)</td>
</tr>

<tr>
<td class="org-left"><code>BandConfig</code></td>
<td class="org-left"><code>autoneb</code> (<code>with_autoneb</code>), <code>cell</code> (<code>with_cell</code>), <code>climbing</code> (<code>with_climbing</code>), <code>exp_precon</code> (<code>with_exp_precon</code>), <code>fixed_atoms</code> (<code>with_fixed_atoms</code>), <code>force_tol</code> (<code>with_force_tol</code>), <code>free_end</code> (<code>with_free_end</code>), <code>max_move</code> (<code>with_max_move</code>), <code>method</code> (<code>with_method</code>), <code>min_mode</code> (<code>with_min_mode</code>), <code>projection</code> (<code>with_projection</code>), <code>quickmin</code> (<code>with_quickmin</code>), <code>quickmin_options</code> (<code>with_quickmin_options</code>), <code>remove_translation</code> (<code>with_remove_translation</code>), <code>saddle_check</code> (<code>with_saddle_check</code>), <code>solid</code> (<code>with_solid</code>), <code>spring</code> (<code>with_spring</code>), <code>string</code> (<code>with_string</code>), <code>tangent</code> (<code>with_tangent</code>), <code>time_step</code> (<code>with_time_step</code>), <code>zoom</code> (<code>with_zoom</code>)</td>
</tr>

<tr>
<td class="org-left"><code>BasinConfig</code></td>
<td class="org-left"><code>cap</code> (<code>with_cap</code>), <code>control</code> (<code>with_control</code>), <code>hvp_step</code> (<code>with_hvp_step</code>), <code>refine</code> (<code>with_refine</code>), <code>refinements</code> (<code>with_refinements</code>), <code>resolution</code> (<code>with_resolution</code>), <code>sampler</code> (<code>with_sampler</code>), <code>spacing</code> (<code>with_spacing</code>), <code>tolerance</code> (<code>with_tolerance</code>), <code>width</code> (<code>with_width</code>)</td>
</tr>

<tr>
<td class="org-left"><code>Basins</code></td>
<td class="org-left"><code>distance</code> (<code>with_distance</code>), <code>products</code> (<code>with_products</code>), <code>quench</code> (<code>with_quench</code>), <code>reactants</code> (<code>with_reactants</code>), <code>threshold</code> (<code>with_threshold</code>), <code>tolerance</code> (<code>with_tolerance</code>)</td>
</tr>

<tr>
<td class="org-left"><code>CartesianPes</code></td>
<td class="org-left"><code>with_proj</code> (<code>CartesianPes::with_proj</code>, <code>set_proj</code>)</td>
</tr>

<tr>
<td class="org-left"><code>CiConfig</code></td>
<td class="org-left"><code>trigger_factor</code> (<code>with_trigger_factor</code>), <code>trigger_force</code> (<code>with_trigger_force</code>)</td>
</tr>

<tr>
<td class="org-left"><code>ConnectConfig</code></td>
<td class="org-left"><code>alpha0</code> (<code>with_alpha0</code>), <code>band_images</code> (<code>with_band_images</code>), <code>distance_step_max</code> (<code>with_distance_step_max</code>), <code>distance_step_min</code> (<code>with_distance_step_min</code>), <code>energy_difference_threshold</code> (<code>with_energy_difference_threshold</code>), <code>forcetol</code> (<code>with_forcetol</code>), <code>forcetol_per_distance</code> (<code>with_forcetol_per_distance</code>), <code>lanczos_iterations</code> (<code>with_lanczos_iterations</code>), <code>lanczos_rel_tol</code> (<code>with_lanczos_rel_tol</code>), <code>minimize_fmax</code> (<code>with_minimize_fmax</code>), <code>minimize_maxrise</code> (<code>with_minimize_maxrise</code>), <code>minimize_nsteps</code> (<code>with_minimize_nsteps</code>), <code>minmode_max_move</code> (<code>with_minmode_max_move</code>), <code>minmode_memory</code> (<code>with_minmode_memory</code>), <code>n_loops</code> (<code>with_n_loops</code>), <code>nhistx</code> (<code>with_nhistx</code>), <code>nsteps</code> (<code>with_nsteps</code>), <code>path_initializer</code> (<code>with_path_initializer</code>), <code>quench_nodes</code> (<code>with_quench_nodes</code>), <code>saddle_cosp</code> (<code>with_saddle_cosp</code>), <code>target_distance_other</code> (<code>with_target_distance_other</code>), <code>target_distance_saddle</code> (<code>with_target_distance_saddle</code>), <code>trusted_force_difference</code> (<code>with_trusted_force_difference</code>), <code>trustradius</code> (<code>with_trustradius</code>)</td>
</tr>

<tr>
<td class="org-left"><code>Continuation</code></td>
<td class="org-left"><code>delta</code> (<code>with_delta</code>), <code>images</code> (<code>with_images</code>), <code>levels</code> (<code>with_levels</code>), <code>max_steps</code> (<code>with_max_steps</code>)</td>
</tr>

<tr>
<td class="org-left"><code>CoverBudget</code></td>
<td class="org-left"><code>max_arms</code> (<code>with_max_arms</code>), <code>max_force_calls</code> (<code>with_max_force_calls</code>)</td>
</tr>

<tr>
<td class="org-left"><code>CrystalSymmetryConfig</code></td>
<td class="org-left"><code>cell</code> (<code>with_cell</code>), <code>matching_tolerance</code> (<code>with_matching_tolerance</code>), <code>orbit_tolerance</code> (<code>with_orbit_tolerance</code>), <code>orthogonality_tolerance</code> (<code>with_orthogonality_tolerance</code>)</td>
</tr>

<tr>
<td class="org-left"><code>CrystalSymmetryOperation</code></td>
<td class="org-left"><code>centre</code> (<code>with_centre</code>), <code>permutation</code> (<code>with_permutation</code>), <code>rotation</code> (<code>with_rotation</code>), <code>translation</code> (<code>with_translation</code>)</td>
</tr>

<tr>
<td class="org-left"><code>CurvatureBound</code></td>
<td class="org-left"><code>fraction</code> (<code>with_fraction</code>), <code>lambda0</code> (<code>with_lambda0</code>), <code>slope</code> (<code>with_slope</code>)</td>
</tr>

<tr>
<td class="org-left"><code>DefectPairConfig</code></td>
<td class="org-left"><code>angular_scale</code> (<code>with_angular_scale</code>), <code>band</code> (<code>with_band</code>), <code>boltzmann</code> (<code>with_boltzmann</code>), <code>bulk_stress</code> (<code>with_bulk_stress</code>), <code>cell_mode</code> (<code>with_cell_mode</code>), <code>connect_rmsd</code> (<code>with_connect_rmsd</code>), <code>dimer</code> (<code>with_dimer</code>), <code>direction_tol</code> (<code>with_direction_tol</code>), <code>external_stress</code> (<code>with_external_stress</code>), <code>finite_temperature</code> (<code>with_finite_temperature</code>), <code>harmonic_atoms</code> (<code>with_harmonic_atoms</code>), <code>harmonic_step</code> (<code>with_harmonic_step</code>), <code>images</code> (<code>with_images</code>), <code>mass</code> (<code>with_mass</code>), <code>max_band_steps</code> (<code>with_max_band_steps</code>), <code>max_dimer_steps</code> (<code>with_max_dimer_steps</code>), <code>pafi</code> (<code>with_pafi</code>), <code>planes</code> (<code>with_planes</code>), <code>push</code> (<code>with_push</code>), <code>relax</code> (<code>with_relax</code>), <code>seed_fraction</code> (<code>with_seed_fraction</code>), <code>shooting</code> (<code>with_shooting</code>), <code>temperatures</code> (<code>with_temperatures</code>), <code>zero_t_tol</code> (<code>with_zero_t_tol</code>)</td>
</tr>

<tr>
<td class="org-left"><code>DirectionCoverConfig</code></td>
<td class="org-left"><code>art</code> (<code>with_art</code>), <code>art_max_steps</code> (<code>with_art_max_steps</code>), <code>batch</code> (<code>with_batch</code>), <code>cell</code> (<code>with_cell</code>), <code>cover_points</code> (<code>with_cover_points</code>), <code>cover_sweeps</code> (<code>with_cover_sweeps</code>), <code>ei_margin</code> (<code>with_ei_margin</code>), <code>kick_radius</code> (<code>with_kick_radius</code>), <code>kick_width</code> (<code>with_kick_width</code>), <code>length_scale</code> (<code>with_length_scale</code>), <code>minimum_energy_tol</code> (<code>with_minimum_energy_tol</code>), <code>minimum_rmsd_tol</code> (<code>with_minimum_rmsd_tol</code>), <code>policy</code> (<code>with_policy</code>), <code>saddle_energy_tol</code> (<code>with_saddle_energy_tol</code>), <code>saddle_rmsd_tol</code> (<code>with_saddle_rmsd_tol</code>)</td>
</tr>

<tr>
<td class="org-left"><code>ExpBandPrecon</code></td>
<td class="org-left"><code>a</code> (<code>with_a</code>), <code>c_stab</code> (<code>with_c_stab</code>), <code>mu</code> (<code>with_mu</code>), <code>r_cut_factor</code> (<code>with_r_cut_factor</code>)</td>
</tr>

<tr>
<td class="org-left"><code>FmmSolver</code></td>
<td class="org-left"><code>green</code> (<code>with_green</code>), <code>leaf</code> (<code>with_leaf</code>), <code>theta</code> (<code>with_theta</code>)</td>
</tr>

<tr>
<td class="org-left"><code>FixedEnergySeed</code></td>
<td class="org-left"><code>images</code> (<code>with_images</code>), <code>max_evaluations</code> (<code>with_max_evaluations</code>), <code>saddle_energy</code> (<code>with_saddle_energy</code>), <code>target_energy</code> (<code>with_target_energy</code>)</td>
</tr>

<tr>
<td class="org-left"><code>ForceMatchOpts</code></td>
<td class="org-left"><code>bond_scale</code> (<code>with_bond_scale</code>), <code>kinds</code> (<code>with_kinds</code>), <code>rcut_factor</code> (<code>with_rcut_factor</code>), <code>tvecs</code> (<code>with_tvecs</code>)</td>
</tr>

<tr>
<td class="org-left"><code>FreeEndConfig</code></td>
<td class="org-left"><code>cubic</code> (<code>with_cubic</code>), <code>cut_force</code> (<code>with_cut_force</code>), <code>energy_tol</code> (<code>with_energy_tol</code>), <code>max_cuts</code> (<code>with_max_cuts</code>), <code>side</code> (<code>with_side</code>), <code>stability</code> (<code>with_stability</code>), <code>target_energy</code> (<code>with_target_energy</code>), <code>window_offset</code> (<code>with_window_offset</code>)</td>
</tr>

<tr>
<td class="org-left"><code>GaussianTransport</code></td>
<td class="org-left"><code>rank</code> (<code>with_rank</code>), <code>shrink_to_rigid</code> (<code>with_shrink_to_rigid</code>), <code>shrinkage</code> (<code>with_shrinkage</code>)</td>
</tr>

<tr>
<td class="org-left"><code>GpuPolicy</code></td>
<td class="org-left"><code>enabled</code> (<code>with_enabled</code>), <code>min_dim</code> (<code>with_min_dim</code>)</td>
</tr>

<tr>
<td class="org-left"><code>GrowingStringConfig</code></td>
<td class="org-left"><code>band</code> (<code>with_band</code>, <code>with_cell</code>, <code>with_climbing</code>, <code>with_fixed_atoms</code>, <code>with_force_tol</code>, <code>with_max_move</code>, <code>with_method</code>, <code>with_min_mode</code>, <code>with_quickmin</code>, <code>with_remove_translation</code>, <code>with_reparametrization</code>, <code>with_saddle_check</code>, <code>with_time_step</code>), <code>force_gate</code> (<code>with_force_gate</code>), <code>frontier_tol</code> (<code>with_frontier_tol</code>), <code>join_tol</code> (<code>with_join_tol</code>), <code>max_steps_per_node</code> (<code>with_max_steps_per_node</code>), <code>min_steps_per_node</code> (<code>with_min_steps_per_node</code>), <code>n_images</code> (<code>with_n_images</code>)</td>
</tr>

<tr>
<td class="org-left"><code>HarmonicExpansion</code></td>
<td class="org-left"><code>reference</code> (<code>with_reference</code>), <code>surface</code></td>
</tr>

<tr>
<td class="org-left"><code>HarmonicReference</code></td>
<td class="org-left"><code>hvp_step</code> (<code>with_hvp_step</code>), <code>point</code> (<code>with_point</code>), <code>tolerance</code> (<code>with_tolerance</code>)</td>
</tr>

<tr>
<td class="org-left"><code>HyperplaneObservation</code></td>
<td class="org-left"><code>constraint_force</code> (<code>with_constraint_force</code>), <code>jacobian</code> (<code>with_jacobian</code>), <code>observables</code> (<code>with_observable</code>), <code>pafi_gradient</code> (<code>with_pafi_gradient</code>), <code>position</code> (<code>with_position</code>), <code>projected_force</code> (<code>with_projected_force</code>), <code>projected_velocity</code> (<code>with_projected_velocity</code>)</td>
</tr>

<tr>
<td class="org-left"><code>HyperplaneSampleBudget</code></td>
<td class="org-left"><code>coordinates</code> (<code>with_coordinates</code>), <code>lambda</code> (<code>with_lambda</code>), <code>observables</code> (<code>with_observable</code>), <code>positions</code> (<code>with_positions</code>), <code>region</code> (<code>with_region</code>), <code>samples</code> (<code>with_samples</code>), <code>seed</code> (<code>with_seed</code>), <code>stride</code> (<code>with_stride</code>), <code>thermal</code> (<code>with_thermal</code>), <code>thermalize</code> (<code>with_thermalize</code>)</td>
</tr>

<tr>
<td class="org-left"><code>IdppConfig</code></td>
<td class="org-left"><code>grad_tol</code> (<code>with_grad_tol</code>), <code>max_iterations</code> (<code>with_max_iterations</code>), <code>max_move</code> (<code>with_max_move</code>), <code>memory</code> (<code>with_memory</code>)</td>
</tr>

<tr>
<td class="org-left"><code>Index1Config</code></td>
<td class="org-left"><code>fd_dr</code> (<code>with_fd_dr</code>), <code>force_tol</code> (<code>with_force_tol</code>), <code>mode</code> (<code>with_mode</code>), <code>sparse_pattern</code> (<code>with_sparse_pattern</code>), <code>trust_radius</code> (<code>with_trust_radius</code>), <code>update</code> (<code>with_update</code>)</td>
</tr>

<tr>
<td class="org-left"><code>IndexCheck</code></td>
<td class="org-left"><code>escape</code> (<code>with_escape</code>), <code>escapes</code> (<code>with_escapes</code>), <code>lowest</code> (<code>with_lowest</code>), <code>negative</code> (<code>with_negative</code>), <code>step</code> (<code>with_step</code>)</td>
</tr>

<tr>
<td class="org-left"><code>IndexKConfig</code></td>
<td class="org-left"><code>block</code> (<code>with_block</code>), <code>k</code> (<code>with_k</code>), <code>max_block_iterations</code> (<code>with_max_block_iterations</code>), <code>residual_tol</code> (<code>with_residual_tol</code>), <code>search</code> (<code>with_search</code>)</td>
</tr>

<tr>
<td class="org-left"><code>InternalPes</code></td>
<td class="org-left"><code>with_dummies</code> (<code>InternalPes::with_dummies</code>, <code>with_vocn</code>)</td>
</tr>

<tr>
<td class="org-left"><code>InternalWeights</code></td>
<td class="org-left"><code>angle</code> (<code>with_angle</code>), <code>bond</code> (<code>with_bond</code>), <code>dihedral</code> (<code>with_dihedral</code>), <code>other</code> (<code>with_other</code>), <code>translation</code> (<code>with_translation</code>)</td>
</tr>

<tr>
<td class="org-left"><code>IrcConfig</code></td>
<td class="org-left"><code>dr</code> (<code>with_dr</code>), <code>dx</code> (<code>with_dx</code>), <code>eigen_kind</code> (<code>with_eigen_kind</code>), <code>force_gate</code> (<code>with_force_gate</code>), <code>force_tol</code> (<code>with_force_tol</code>), <code>kind</code> (<code>with_kind</code>), <code>krylov_dim</code> (<code>with_krylov_dim</code>), <code>max_inner</code> (<code>with_max_inner</code>)</td>
</tr>

<tr>
<td class="org-left"><code>JointHessianConfig</code></td>
<td class="org-left"><code>coordinate_step</code> (<code>with_coordinate_step</code>), <code>stationary_tolerance</code> (<code>with_stationary_tolerance</code>), <code>stress_step</code> (<code>with_stress_step</code>)</td>
</tr>

<tr>
<td class="org-left"><code>JointVineyardConfig</code></td>
<td class="org-left"><code>mass_amu</code> (<code>with_mass_amu</code>), <code>mode_tolerance</code> (<code>with_mode_tolerance</code>)</td>
</tr>

<tr>
<td class="org-left"><code>KappaDimerConfig</code></td>
<td class="org-left"><code>beta</code> (<code>with_beta</code>), <code>eigen</code> (<code>with_eigen</code>)</td>
</tr>

<tr>
<td class="org-left"><code>LammpsSamplerConfig</code></td>
<td class="org-left"><code>excluded_modes</code> (<code>with_excluded_modes</code>), <code>friction_per_ps</code> (<code>with_friction_per_ps</code>), <code>library_path</code> (<code>with_library_path</code>), <code>thermodynamic_reference</code> (<code>with_thermodynamic_reference</code>), <code>thermostat</code> (<code>with_thermostat</code>), <code>time_step_ps</code> (<code>with_time_step_ps</code>)</td>
</tr>

<tr>
<td class="org-left"><code>LammpsSystem</code></td>
<td class="org-left"><code>atomic_numbers</code> (<code>with_atomic_numbers</code>), <code>box_matrix</code> (<code>with_box_matrix</code>), <code>config_bytes</code> (<code>with_config_bytes</code>), <code>masses</code> (<code>with_masses</code>), <code>positions</code> (<code>with_positions</code>)</td>
</tr>

<tr>
<td class="org-left"><code>LammpsThermodynamicReference</code></td>
<td class="org-left"><code>energy</code>, <code>gradient</code>, <code>hessian</code>, <code>point</code> (<code>LammpsThermodynamicReference::new</code>)</td>
</tr>

<tr>
<td class="org-left"><code>LandscapeConfig</code></td>
<td class="org-left"><code>cell</code> (<code>with_cell</code>), <code>certificate</code> (<code>with_certificate</code>), <code>duplicate_distance</code> (<code>with_duplicate_distance</code>), <code>force_gate</code> (<code>with_force_gate</code>), <code>max_evaluations</code> (<code>with_max_evaluations</code>), <code>max_nodes</code> (<code>with_max_nodes</code>), <code>max_steps</code> (<code>with_max_steps</code>), <code>push</code> (<code>with_push</code>), <code>relax</code> (<code>with_relax</code>), <code>search</code> (<code>with_search</code>)</td>
</tr>

<tr>
<td class="org-left"><code>LineNebConfig</code></td>
<td class="org-left"><code>endpoint_below_energy</code> (<code>with_endpoint_below_energy</code>), <code>endpoint_kappa</code> (<code>with_endpoint_kappa</code>), <code>energy</code> (<code>with_energy</code>), <code>force_tol</code> (<code>with_force_tol</code>), <code>masses</code> (<code>with_masses</code>), <code>reduced_planck</code> (<code>with_reduced_planck</code>), <code>spring_k</code> (<code>with_spring_k</code>), <code>time_step</code> (<code>with_time_step</code>)</td>
</tr>

<tr>
<td class="org-left"><code>LangevinConfig</code></td>
<td class="org-left"><code>blocks</code> (<code>with_blocks</code>), <code>burn_in</code> (<code>with_burn_in</code>), <code>control</code> (<code>with_control</code>), <code>friction</code> (<code>with_friction</code>), <code>hold_excluded</code> (<code>with_hold_excluded</code>), <code>integrator</code> (<code>with_integrator</code>), <code>records</code> (<code>with_records</code>), <code>region</code> (<code>with_region</code>), <code>retain</code> (<code>with_retain</code>), <code>seed</code> (<code>with_seed</code>), <code>stride</code> (<code>with_stride</code>), <code>thermal</code> (<code>with_thermal</code>), <code>time_step</code> (<code>with_time_step</code>)</td>
</tr>

<tr>
<td class="org-left"><code>LinebConfig</code></td>
<td class="org-left"><code>boltzmann</code> (<code>with_boltzmann</code>), <code>endpoint_scale</code> (<code>with_endpoint_scale</code>), <code>gradient_end_tangents</code> (<code>with_gradient_end_tangents</code>), <code>hbar</code> (<code>with_hbar</code>), <code>max_move</code> (<code>with_max_move</code>), <code>method</code> (<code>with_method</code>), <code>opt_tol</code> (<code>with_opt_tol</code>), <code>perp_tol</code> (<code>with_perp_tol</code>), <code>quadrature_pairs</code> (<code>with_quadrature_pairs</code>), <code>spring</code> (<code>with_spring</code>), <code>time_step</code> (<code>with_time_step</code>), <code>turning_width</code> (<code>with_turning_width</code>)</td>
</tr>

<tr>
<td class="org-left"><code>MinModeConfig</code></td>
<td class="org-left"><code>artn</code> (<code>with_artn</code>), <code>difference</code> (<code>with_difference</code>), <code>dr</code> (<code>with_dr</code>), <code>force_tol</code> (<code>with_force_tol</code>), <code>kind</code> (<code>with_kind</code>), <code>krylov_dim</code> (<code>with_krylov_dim</code>), <code>max_move</code> (<code>with_max_move</code>), <code>max_rotations</code> (<code>with_max_rotations</code>), <code>method</code> (<code>with_method</code>), <code>rotation_angle_tol</code> (<code>with_rotation_angle_tol</code>), <code>rotation_tol</code> (<code>with_rotation_tol</code>), <code>saddle_check</code> (<code>with_saddle_check</code>), <code>seed_curvature</code> (<code>with_seed_curvature</code>), <code>time_step</code> (<code>with_time_step</code>)</td>
</tr>

<tr>
<td class="org-left"><code>MinModeRefine</code></td>
<td class="org-left"><code>alignment_tol</code> (<code>with_alignment_tol</code>), <code>climb_alignment</code> (<code>with_climb_alignment</code>), <code>curvature_retained</code> (<code>with_curvature_retained</code>), <code>difference</code> (<code>with_difference</code>), <code>dr</code> (<code>with_dr</code>), <code>kind</code> (<code>with_kind</code>), <code>krylov_dim</code> (<code>with_krylov_dim</code>), <code>lanczos_rel_tol</code> (<code>with_lanczos_rel_tol</code>), <code>max_rotations</code> (<code>with_max_rotations</code>), <code>max_steps</code> (<code>with_max_steps</code>), <code>min_overlap</code> (<code>with_min_overlap</code>), <code>refresh_interval</code> (<code>with_refresh_interval</code>), <code>restore_unhelpful</code> (<code>with_restore_unhelpful</code>), <code>retry_factor</code> (<code>with_retry_factor</code>), <code>rotation_angle_tol</code> (<code>with_rotation_angle_tol</code>), <code>rotation_tol</code> (<code>with_rotation_tol</code>), <code>stable_steps</code> (<code>with_stable_steps</code>), <code>trigger_factor</code> (<code>with_trigger_factor</code>), <code>trigger_force</code> (<code>with_trigger_force</code>)</td>
</tr>

<tr>
<td class="org-left"><code>NptConfig</code></td>
<td class="org-left"><code>berendsen_tau_s</code> (<code>with_berendsen_tau_s</code>), <code>block_steps</code> (<code>with_block_steps</code>), <code>compressibility_per_ev_a3</code> (<code>with_compressibility_per_ev_a3</code>), <code>equilibration_steps</code> (<code>with_equilibration_steps</code>), <code>extension_steps</code> (<code>with_extension_steps</code>), <code>langevin_gamma_per_s</code> (<code>with_langevin_gamma_per_s</code>), <code>maximum_steps</code> (<code>with_maximum_steps</code>), <code>repeats</code> (<code>with_repeats</code>), <code>seed</code> (<code>with_seed</code>), <code>steps</code> (<code>with_steps</code>), <code>target_pressure_ev_a3</code> (<code>with_target_pressure_ev_a3</code>), <code>timestep_s</code> (<code>with_timestep_s</code>)</td>
</tr>

<tr>
<td class="org-left"><code>OcinebConfig</code></td>
<td class="org-left"><code>angle_tol</code> (<code>with_angle_tol</code>), <code>band</code> (<code>with_band</code>), <code>max_mmf_steps</code> (<code>with_max_mmf_steps</code>), <code>minmode</code> (<code>with_minmode</code>), <code>restore_unhelpful</code> (<code>with_restore_unhelpful</code>), <code>stability_count</code> (<code>with_stability_count</code>), <code>trigger_factor</code> (<code>with_trigger_factor</code>), <code>trigger_force</code> (<code>with_trigger_force</code>)</td>
</tr>

<tr>
<td class="org-left"><code>PafiConfig</code></td>
<td class="org-left"><code>blocks</code> (<code>with_blocks</code>), <code>fix_com</code> (<code>with_fix_com</code>), <code>max_rounds</code> (<code>with_max_rounds</code>), <code>mobility</code> (<code>with_mobility</code>), <code>region</code> (<code>with_region</code>), <code>retain</code> (<code>with_retain</code>), <code>samples</code> (<code>with_samples</code>), <code>seed</code> (<code>with_seed</code>), <code>stride</code> (<code>with_stride</code>), <code>target_error</code> (<code>with_target_error</code>), <code>thermal</code> (<code>with_thermal</code>), <code>thermalize</code> (<code>with_thermalize</code>), <code>time_step</code> (<code>with_time_step</code>)</td>
</tr>

<tr>
<td class="org-left"><code>PairConfig</code></td>
<td class="org-left"><code>offset</code>, <code>orientation</code> (<code>PairConfig::new</code>)</td>
</tr>

<tr>
<td class="org-left"><code>PeriodicElasticConfig</code></td>
<td class="org-left"><code>green</code> (<code>with_green</code>), <code>image_shells</code> (<code>with_image_shells</code>)</td>
</tr>

<tr>
<td class="org-left"><code>PermutationDistance</code></td>
<td class="org-left"><code>atoms</code> (<code>with_atoms</code>), <code>cell</code> (<code>with_cell</code>), <code>masses</code> (<code>with_masses</code>), <code>species</code> (<code>with_species</code>)</td>
</tr>

<tr>
<td class="org-left"><code>PlaneOptimizerConfig</code></td>
<td class="org-left"><code>confidence</code> (<code>with_confidence</code>), <code>max_evaluations</code> (<code>with_max_evaluations</code>), <code>max_iterations</code> (<code>with_max_iterations</code>), <code>max_records</code> (<code>with_max_records</code>), <code>max_rotation</code> (<code>with_max_rotation</code>), <code>max_translation</code> (<code>with_max_translation</code>), <code>metric</code> (<code>with_metric</code>), <code>metric_rank</code> (<code>with_metric_rank</code>), <code>norm_test</code> (<code>with_norm_test</code>), <code>patience</code> (<code>with_patience</code>), <code>recenter</code> (<code>with_recenter</code>), <code>rotate</code> (<code>with_rotate</code>), <code>rotation_basis</code> (<code>with_rotation_basis</code>), <code>sampler</code> (<code>with_sampler</code>), <code>tolerance</code> (<code>with_tolerance</code>), <code>translate</code> (<code>with_translate</code>)</td>
</tr>

<tr>
<td class="org-left"><code>QuadraticModel</code></td>
<td class="org-left"><code>center</code> (<code>with_center</code>), <code>energy</code> (<code>with_energy</code>), <code>gradient</code> (<code>with_gradient</code>), <code>hessian</code> (<code>with_hessian</code>)</td>
</tr>

<tr>
<td class="org-left"><code>QuenchConfig</code></td>
<td class="org-left"><code>force_tol</code> (<code>with_force_tol</code>), <code>initial_step</code> (<code>with_initial_step</code>), <code>max_move</code> (<code>with_max_move</code>), <code>max_steps</code> (<code>with_max_steps</code>), <code>method</code> (<code>with_method</code>)</td>
</tr>

<tr>
<td class="org-left"><code>RateMultiplicity</code></td>
<td class="org-left"><code>in_plane_images</code> (<code>with_in_plane_images</code>, <code>with_region</code>), <code>paths</code> (<code>with_paths</code>)</td>
</tr>

<tr>
<td class="org-left"><code>RelaxConfig</code></td>
<td class="org-left"><code>force_tol</code> (<code>with_force_tol</code>), <code>initial_step</code> (<code>with_initial_step</code>), <code>max_atom_move</code> (<code>with_max_atom_move</code>), <code>max_steps</code> (<code>with_max_steps</code>), <code>method</code> (<code>with_method</code>)</td>
</tr>

<tr>
<td class="org-left"><code>ReversibleWorkConfig</code></td>
<td class="org-left"><code>bar_samples</code> (<code>with_bar_samples</code>), <code>estimate</code> (<code>with_estimate</code>), <code>mbar</code> (<code>with_mbar</code>), <code>min_overlap</code> (<code>with_min_overlap</code>), <code>nodes</code> (<code>with_nodes</code>), <code>sampler</code> (<code>with_sampler</code>), <code>simpson</code> (<code>with_simpson</code>), <code>transport</code> (<code>with_transport</code>), <code>warm_burn_in</code> (<code>with_warm_burn_in</code>)</td>
</tr>

<tr>
<td class="org-left"><code>RtrConfig</code></td>
<td class="org-left"><code>fd_step</code> (<code>with_fd_step</code>), <code>kappa</code> (<code>with_kappa</code>), <code>max_cg</code> (<code>with_max_cg</code>), <code>radius_max</code> (<code>with_radius_max</code>), <code>reparametrization</code> (<code>with_reparametrization</code>), <code>theta</code> (<code>with_theta</code>)</td>
</tr>

<tr>
<td class="org-left"><code>SaddleEscape</code></td>
<td class="org-left"><code>max_attempts</code> (<code>with_max_attempts</code>), <code>push</code> (<code>with_push</code>)</td>
</tr>

<tr>
<td class="org-left"><code>SaddleIndexCheck</code></td>
<td class="org-left"><code>block</code> (<code>with_block</code>), <code>difference</code> (<code>with_difference</code>), <code>dr</code> (<code>with_dr</code>), <code>escape</code> (<code>with_escape</code>), <code>lowest</code> (<code>with_lowest</code>), <code>max_actions</code> (<code>with_max_actions</code>), <code>negative_tol</code> (<code>with_negative_tol</code>), <code>residual_tol</code> (<code>with_residual_tol</code>), <code>seed</code> (<code>with_seed</code>)</td>
</tr>

<tr>
<td class="org-left"><code>SamdConfig</code></td>
<td class="org-left"><code>dt</code> (<code>with_dt</code>), <code>exponential</code> (<code>with_exponential</code>), <code>ngen</code> (<code>with_ngen</code>), <code>t0</code> (<code>with_t0</code>), <code>tau</code> (<code>with_tau</code>), <code>tf</code> (<code>with_tf</code>)</td>
</tr>

<tr>
<td class="org-left"><code>SellaMinConfig</code></td>
<td class="org-left"><code>delta</code> (<code>with_delta</code>), <code>delta_min</code> (<code>with_delta_min</code>), <code>force_gate</code> (<code>with_force_gate</code>), <code>force_tol</code> (<code>with_force_tol</code>), <code>restricted</code> (<code>with_restricted</code>), <code>rho_dec</code> (<code>with_rho_dec</code>), <code>rho_inc</code> (<code>with_rho_inc</code>), <code>sigma_dec</code> (<code>with_sigma_dec</code>), <code>sigma_inc</code> (<code>with_sigma_inc</code>)</td>
</tr>

<tr>
<td class="org-left"><code>SellaSaddleConfig</code></td>
<td class="org-left"><code>delta</code> (<code>with_delta</code>), <code>delta_min</code> (<code>with_delta_min</code>), <code>eig</code> (<code>with_eig</code>), <code>eigen_device</code> (<code>with_eigen_device</code>), <code>expand</code> (<code>with_expand</code>), <code>force_gate</code> (<code>with_force_gate</code>), <code>force_tol</code> (<code>with_force_tol</code>), <code>gamma</code> (<code>with_gamma</code>), <code>nsteps_per_diag</code> (<code>with_nsteps_per_diag</code>), <code>order</code> (<code>with_order</code>), <code>restricted</code> (<code>with_restricted</code>), <code>rho_dec</code> (<code>with_rho_dec</code>), <code>rho_inc</code> (<code>with_rho_inc</code>), <code>sigma_dec</code> (<code>with_sigma_dec</code>), <code>sigma_inc</code> (<code>with_sigma_inc</code>), <code>small_system_atom_ceiling</code> (<code>with_small_system_atom_ceiling</code>), <code>symmetrized_ritz</code> (<code>with_symmetrized_ritz</code>)</td>
</tr>

<tr>
<td class="org-left"><code>ShootingConfig</code></td>
<td class="org-left"><code>blocks</code> (<code>with_blocks</code>), <code>friction</code> (<code>with_friction</code>), <code>mass</code> (<code>with_mass</code>), <code>plateau_record</code> (<code>with_plateau_record</code>), <code>record_every</code> (<code>with_record_every</code>), <code>seed</code> (<code>with_seed</code>), <code>shots</code> (<code>with_shots</code>), <code>steps</code> (<code>with_steps</code>), <code>thermal</code> (<code>with_thermal</code>), <code>time_step</code> (<code>with_time_step</code>)</td>
</tr>

<tr>
<td class="org-left"><code>SidppAlternatingConfig</code></td>
<td class="org-left"><code>alpha</code> (<code>with_alpha</code>), <code>final_iterations</code> (<code>with_final_iterations</code>), <code>force_tol</code> (<code>with_force_tol</code>), <code>inverse_distance</code> (<code>with_inverse_distance</code>), <code>max_move</code> (<code>with_max_move</code>), <code>pull</code> (<code>with_pull</code>), <code>pull_to_target</code> (<code>with_pull_to_target</code>), <code>reparametrize</code> (<code>with_reparametrize</code>), <code>short_iterations</code> (<code>with_short_iterations</code>), <code>spring</code> (<code>with_spring</code>)</td>
</tr>

<tr>
<td class="org-left"><code>SidppConfig</code></td>
<td class="org-left"><code>force_tol</code> (<code>with_force_tol</code>), <code>max_iterations</code> (<code>with_max_iterations</code>), <code>max_move</code> (<code>with_max_move</code>), <code>rms_tol</code> (<code>with_rms_tol</code>), <code>spring</code> (<code>with_spring</code>), <code>time_step</code> (<code>with_time_step</code>)</td>
</tr>

<tr>
<td class="org-left"><code>SolidDimerConfig</code></td>
<td class="org-left"><code>external_stress</code> (<code>with_external_stress</code>, <code>with_pressure</code>), <code>flat</code> (<code>with_flat</code>), <code>index</code> (<code>with_index</code>, <code>with_index_check</code>), <code>mask</code> (<code>with_mask</code>), <code>search</code> (<code>with_search</code>), <code>stress_check</code> (<code>with_stress_check</code>), <code>weight</code> (<code>with_weight</code>)</td>
</tr>

<tr>
<td class="org-left"><code>SolidState</code></td>
<td class="org-left"><code>cells</code> (<code>with_cells</code>), <code>pressure</code> (<code>with_pressure</code>), <code>weight</code> (<code>with_weight</code>)</td>
</tr>

<tr>
<td class="org-left"><code>SparseHessianOpts</code></td>
<td class="org-left"><code>difference</code> (<code>with_difference</code>), <code>memory_budget</code> (<code>with_memory_budget</code>), <code>skin</code> (<code>with_skin</code>), <code>step</code> (<code>with_step</code>)</td>
</tr>

<tr>
<td class="org-left"><code>StrainResponse</code></td>
<td class="org-left"><code>with_multipoles</code>, <code>with_point_group</code>, <code>with_policy</code>, <code>with_prefactor</code>, <code>with_variants</code></td>
</tr>

<tr>
<td class="org-left"><code>StressCheck</code></td>
<td class="org-left"><code>absolute</code> (<code>with_absolute</code>, <code>with_tolerance</code>), <code>relative</code> (<code>with_relative</code>), <code>step</code> (<code>with_step</code>)</td>
</tr>

<tr>
<td class="org-left"><code>StringConfig</code></td>
<td class="org-left"><code>reparametrization</code> (<code>with_reparametrization</code>)</td>
</tr>

<tr>
<td class="org-left"><code>SymmetryOperations</code></td>
<td class="org-left"><code>operations</code> (<code>with_operations</code>)</td>
</tr>

<tr>
<td class="org-left"><code>ThermodynamicDerivative</code></td>
<td class="org-left"><code>full</code>, <code>reference</code> (<code>with_reference</code>)</td>
</tr>

<tr>
<td class="org-left"><code>ThermodynamicIntegrationConfig</code></td>
<td class="org-left"><code>blocks</code> (<code>with_blocks</code>), <code>nodes</code> (<code>with_nodes</code>), <code>observable</code> (<code>with_observable</code>), <code>samples</code> (<code>with_samples</code>), <code>seed</code> (<code>with_seed</code>), <code>stride</code> (<code>with_stride</code>), <code>thermal</code> (<code>with_thermal</code>), <code>thermalize</code> (<code>with_thermalize</code>)</td>
</tr>

<tr>
<td class="org-left"><code>TopologyConfig</code></td>
<td class="org-left"><code>back_transform_iterations</code> (<code>with_back_transform_iterations</code>), <code>back_transform_tol</code> (<code>with_back_transform_tol</code>), <code>bond_scale</code> (<code>with_bond_scale</code>), <code>dummy_distance</code> (<code>with_dummy_distance</code>), <code>linear_tol</code> (<code>with_linear_tol</code>), <code>max_weld_rounds</code> (<code>with_max_weld_rounds</code>), <code>planar_tol</code> (<code>with_planar_tol</code>), <code>require_complete</code> (<code>with_require_complete</code>), <code>sv_tol</code> (<code>with_sv_tol</code>), <code>weld_growth</code> (<code>with_weld_growth</code>)</td>
</tr>

<tr>
<td class="org-left"><code>TransmissionConfig</code></td>
<td class="org-left"><code>blocks</code> (<code>with_blocks</code>), <code>check_every</code> (<code>with_check_every</code>), <code>commit</code> (<code>with_commit</code>), <code>counting</code> (<code>with_counting</code>), <code>end_state</code> (<code>with_end_state</code>), <code>friction</code> (<code>with_friction</code>), <code>hold_excluded</code> (<code>with_hold_excluded</code>), <code>masses</code> (<code>with_masses</code>), <code>seed</code> (<code>with_seed</code>), <code>shots</code> (<code>with_shots</code>), <code>steps</code> (<code>with_steps</code>), <code>thermal</code> (<code>with_thermal</code>), <code>time_step</code> (<code>with_time_step</code>)</td>
</tr>

<tr>
<td class="org-left"><code>TricConfig</code></td>
<td class="org-left"><code>bond_factor</code> (<code>with_bond_factor</code>), <code>bonds</code> (<code>with_bonds</code>), <code>rank_tolerance</code> (<code>with_rank_tolerance</code>)</td>
</tr>

<tr>
<td class="org-left"><code>TricInversionConfig</code></td>
<td class="org-left"><code>max_atom_move</code> (<code>with_max_atom_move</code>), <code>max_iterations</code> (<code>with_max_iterations</code>), <code>tolerance</code> (<code>with_tolerance</code>)</td>
</tr>

<tr>
<td class="org-left"><code>TricsConfig</code></td>
<td class="org-left"><code>coordinates</code> (<code>with_coordinates</code>), <code>diagnostic_threshold</code> (<code>with_diagnostic_threshold</code>), <code>inversion</code> (<code>with_inversion</code>), <code>length_per_angstrom</code> (<code>with_length_per_angstrom</code>), <code>max_splice_iterations</code> (<code>with_max_splice_iterations</code>), <code>splice_tolerance</code> (<code>with_splice_tolerance</code>)</td>
</tr>

<tr>
<td class="org-left"><code>TricsStartConfig</code></td>
<td class="org-left"><code>seed</code> (<code>with_seed</code>), <code>trics</code> (<code>with_trics</code>)</td>
</tr>

<tr>
<td class="org-left"><code>TrustPolicy</code></td>
<td class="org-left"><code>curvature</code> (<code>with_curvature</code>), <code>error_tolerance</code> (<code>with_error_tolerance</code>), <code>floor</code> (<code>with_floor</code>), <code>quadratic</code> (<code>with_quadratic</code>), <code>radius</code> (<code>with_radius</code>)</td>
</tr>

<tr>
<td class="org-left"><code>TrustSchedule</code></td>
<td class="org-left"><code>delta_min</code> (<code>with_delta_min</code>), <code>rho_dec</code> (<code>with_rho_dec</code>), <code>rho_inc</code> (<code>with_rho_inc</code>), <code>sigma_dec</code> (<code>with_sigma_dec</code>), <code>sigma_inc</code> (<code>with_sigma_inc</code>)</td>
</tr>

<tr>
<td class="org-left"><code>VelocitySofteningConfig</code></td>
<td class="org-left"><code>displacement</code> (<code>with_displacement</code>), <code>mixing</code> (<code>with_mixing</code>), <code>steps</code> (<code>with_steps</code>)</td>
</tr>

<tr>
<td class="org-left"><code>ZoomConfig</code></td>
<td class="org-left"><code>after</code> (<code>with_after</code>), <code>alpha</code> (<code>with_alpha</code>), <code>cubic</code> (<code>with_cubic</code>), <code>offset</code> (<code>with_offset</code>), <code>stability</code> (<code>with_stability</code>)</td>
</tr>
</tbody>
</table>


# C ABI (feature `capi`)

This section lists the entry points, codes and flags of `include/rgsaddle.h`.
A C host reaches them through the application binary interface (ABI).
`RGSADDLE_ABI_MAJOR` is 1.
Status codes include `RGSADDLE_OK`, `RGSADDLE_SHAPE`, `RGSADDLE_SURFACE_FAILED`, `RGSADDLE_ABI_MISMATCH`, `RGSADDLE_NOT_EQUILIBRATED`, `RGSADDLE_INSUFFICIENT_EFFECTIVE_SAMPLES` and, from minor 64, `RGSADDLE_NOT_CONVERGED`.
The methods on the wire are the fast inertial relaxation engine (FIRE) and quick-min.
Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) is also on the wire.
From minor 22 so is ODE12r (`RGSADDLE_METHOD_ODE12R`), the adaptive ordinary differential equation (ODE) step.
From minor 22, `rgsaddle_band_set_exp_precon` installs the per-image exponential metric on the band solver.
With ODE12r that gives ODE12r-Exp.
From minor 39, `rgsaddle_band_set_fixed_atoms` fixes atoms on a band and the exponential metric is then built on the free atoms; `rgsaddle_band_project_fixed` projects with the same mask.
From minor 40, a band, min-mode or OCI-NEB config stamped with that minor or newer certifies the saddle index of the converged point with the `rgsaddle_saddle_check_t` that `rgsaddle_saddle_check_default` fills.
`rgsaddle_band_set_saddle_check`, `rgsaddle_minmode_set_saddle_check` and `rgsaddle_ocineb_set_saddle_check` replace it, NULL for none, the matching `_saddle_index` calls read the certificate into an `rgsaddle_saddle_index_t`, and `rgsaddle_saddle_index` certifies one point.
Min-mode sessions estimate the mode by the dimer or by Lanczos.
For index-1 steps the header has `rgsaddle_nichols_step`, `rgsaddle_prfo_step` and `rgsaddle_index1_step`.
It also has `rgsaddle_hessian_powell`, `rgsaddle_hessian_bofill` and `rgsaddle_cap_max_abs` for those steps.
`rgsaddle_band_restart` and `rgsaddle_band_evaluation` use the band cache.
`rgsaddle_band_evaluation` returns `RGSADDLE_NO_EVALUATION` when nothing is cached at the band's positions.
A min-mode session is reused through three entry points.
They are `rgsaddle_minmode_set_position`, `rgsaddle_minmode_set_mode` and `rgsaddle_minmode_estimate`.
A Lanczos estimate advances one Hessian action and keeps the Krylov basis until `set_position`, `set_mode`, or `reset`.
That basis grows to the free dimension.
After k actions the mode lies in the k + 1 Lanczos vectors, with the Ritz value of the k as its curvature.
Each dimer estimate is the full rotation.
Dimer Polak-Ribiere history survives `set_position` and is cleared by `set_mode` and `reset`.
A failed callback leaves it unchanged.
`rgsaddle_minmode_config_t.difference` picks `RGSADDLE_DIFFERENCE_FORWARD` or `_CENTRAL`.
Its `rotation_angle_tol` field stops a dimer rotation once the optimal angle falls under this tolerance in radians.
For config minors 5 through 62 this field is read as given, so 0 disables the angle stop.
From minor 63, 0 keeps the Rust default unless flag bit 0, `RGSADDLE_MINMODE_EXPLICIT_ROTATION_ANGLE_TOL`, is set; then 0 disables it.
The Rust default is five degrees. With an explicit 0 (legacy minors, or minor 63 with the explicit flag) the stop is off.
Both fields are read only from a config whose version minor is 5 or newer.
`RGSADDLE_ABI_MINOR` is 71.
`rgsaddle_band_set_dynamic` (minor 65) sets or clears `DynamicNeb` on a band.
`rgsaddle_dyneb_mask` (minor 65) applies that mask to a projected band.
From minor 24, `rgsaddle_sella_saddle_set_symmetrized_ritz` sets `SellaSaddleConfig::symmetrized_ritz` from 0 or 1.
It returns `RGSADDLE_INVALID_PARAMETER` for any other value.
`rgsaddle_sella_saddle_set_small_system_atom_ceiling` (minor 25) sets `SellaSaddleConfig::small_system_atom_ceiling` from 0 or 1.
It returns `RGSADDLE_INVALID_PARAMETER` for any other value.
`rgsaddle_band_set_zoom` (minor 31) sets or clears Zoom-NEB on a band.
Recommended when enabled: `alpha` 0.5, `offset` 1, `after` 0 (twice `force_tol` on the largest per-image force), `stability` 5 and `cubic` 1.
Its window always reaches both neighbours of the climbing image, and the min-mode refinement waits for the zoom decision.
`rgsaddle_minmode_set_torque_window` (minor 32) calls `MinModeSession::set_torque_window` with -1, 0 or 1.
The value -1 infers the window as a new session does: on when `rotation_tol` is at least 1e-4 and the session is Lanczos or a dimer whose angle stop is off. 0 turns it off and 1 turns it on.
For config minors 5 through 62, `rotation_angle_tol` 0 keeps the angle stop off. From minor 63, `rotation_angle_tol` 0 keeps the Rust default unless `RGSADDLE_MINMODE_EXPLICIT_ROTATION_ANGLE_TOL` is set.
It returns `RGSADDLE_INVALID_PARAMETER` for any other value.
From minor 41 a Lanczos session reads it too.
From minor 46, -1 leaves the window off for an improved dimer with a positive `rotation_angle_tol`.
`rgsaddle_minmode_set_rotation_stop_length` (minor 38) calls `MinModeSession::set_rotation_stop_length`; 0 turns the stop off and a negative or non-finite length returns `RGSADDLE_INVALID_PARAMETER`.
From minor 43, `rgsaddle_artn_config_default` fills `rgsaddle_artn_config_t` with the published defaults and `rgsaddle_minmode_set_artn` enables them on a Lanczos session or disables them with NULL.
`rgsaddle_reparametrize` (minor 38) places a band at equal arc length with an `rgsaddle_reparametrization_t`, and band flag bit 4, `RGSADDLE_BAND_SPLINE_REPARAM`, puts the trust-region band on the not-a-knot spline.
From minor 44, band flag bit 5, `RGSADDLE_BAND_STRING`, runs the simplified string method of E, Ren, and Vanden-Eijnden (2007) (with `RGSADDLE_BAND_SPLINE_REPARAM` for the not-a-knot spline, or the polyline without it), and `rgsaddle_band_set_string` enables or clears it on a live band.
From minor 44, `rgsaddle_growing_string_config_default`, `rgsaddle_growing_string_create`, `rgsaddle_growing_string_step`, `rgsaddle_growing_string_n_images`, `rgsaddle_growing_string_nodes_added`, `rgsaddle_growing_string_phase`, `rgsaddle_growing_string_positions`, `rgsaddle_growing_string_ids`, `rgsaddle_growing_string_set_fixed_atoms`, `rgsaddle_growing_string_saddle_index`, and `rgsaddle_growing_string_free` expose the growing string method of Peters, Heyden, Bell, and Chakraborty (2004). The growing string `join_tol` uses -1.0 for no early join and treats zero as an inclusive zero-gap threshold.
From minor 45, `rgsaddle_band_set_free_end`, `rgsaddle_band_free_end_residual`, and `rgsaddle_band_free_end_cuts` expose the free-end adaptive NEB of Zhang, Zhang, Ye, and Zheng (2016). The call is a setter, not a flag. `rgsaddle_band_evaluation` returns that step's energies, gradients, and projected interior forces. A host checks `rgsaddle_abi_version()` for minor 45 or newer. The worked example is `tests/free_end_neb.rs`.
From minor 47, `rgsaddle_lineb_config_default` and `rgsaddle_lineb_solve` expose fixed-energy LI-NEB and harmonic quantum TST. `rgsaddle_lineb_config_t` holds `opt_tol`, `perp_tol`, `spring`, `endpoint_scale`, `time_step`, `max_move`, `turning_width`, `hbar`, `boltzmann` and `quadrature_pairs`. `rgsaddle_lineb_result_t` holds `action`, both half periods and temperatures, `log_rate_over_reactant`, and the optimizer, measure, dynamics and rate call counts. Positions are contiguous image, atom, xyz. `molecular` 0 expands atom masses onto Cartesian coordinates without alignment; `molecular` 1 uses molecule alignment. `beads` 0 is path-only with a NaN log rate. Unconverged searches return `RGSADDLE_STALLED`. A host checks `rgsaddle_abi_version()` for minor 47 or newer. The smoke test is `tests/c/lineb_abi.c`.
From minor 47, `RGSADDLE_LINEB_PROJECTED_VERLET` `(1u << 2)` on `rgsaddle_lineb_solve_options` selects the projected Verlet band integrator (`with_projected_verlet`).
The feature macro `RGSADDLE_HAS_LINEB_RATE_DIAGNOSTICS` advertises
`rgsaddle_lineb_solve_counted_diagnostics`. Its final two optional signed
64-bit outputs give the number of negative and total retained fluctuation
modes. Each reports `-1` for a path-only or failed solve.

From minor 60, `rgsaddle_band_set_refine` installs or clears the per-step L-BFGS refine stepper, and `rgsaddle_band_is_refining` reports whether the last first-order step used it. A host checks `rgsaddle_abi_version()` for minor 60 or newer. The worked example is `tests/band_refine.rs`.
From minor 68, `rgsaddle_autoneb_run` grows an AutoNEB band to completion. `rgsaddle_autoneb_config_t` and `rgsaddle_autoneb_report_t` are the inputs and the result. `RGSADDLE_AUTONEB_CLIMB`, `RGSADDLE_AUTONEB_IDPP`, and `RGSADDLE_AUTONEB_KEEP_TRANSLATION` are the flag bits. The callback stays on the calling thread.
The request flags are `RGSADDLE_REQ_ONE_IMAGE` (bit 0) and `RGSADDLE_REQ_CLIMBING_IMAGE` (bit 1).
From a band config at minor 22, the refinement's one-image requests also set bit 1.
Their `image` is then the climbing image's band index.
`rgsaddle_idpp_path` writes the image-dependent pair potential initial path with its endpoints.
It takes a row-major cell or NULL.
From minor 64, `rgsaddle_geodesic_config_default`, `rgsaddle_geodesic_path`, and `rgsaddle_geodesic_morse` interpolate a path in a redundant coordinate chart (Zhu, Thompson, and Martinez, J. Chem. Phys. 150, 164103 (2019), https://doi.org/10.1063/1.5090303). `rgsaddle_geodesic_path` takes a host chart. `rgsaddle_geodesic_morse` uses Eq. (7) with alpha 1.7 and beta 0.01. A NULL config selects the defaults, including an image cap of 32, so it requires `n_images` of at least 32. A supplied cap above `n_images` is `RGSADDLE_INVALID_PARAMETER`.
From minor 66, `rgsaddle_trics_path` refines an image-major unwrapped Cartesian path in translation-rotation internal coordinates with splicing and writes the refined path to `out` (which may alias `initial` for in-place refinement). Its length controls are in angstrom and `length_per_angstrom` converts the caller's coordinates.
From minor 69, `rgsaddle_landscape_config_default`, `rgsaddle_landscape_create`, `rgsaddle_landscape_run`, `rgsaddle_landscape_node`, `rgsaddle_landscape_modes`, `rgsaddle_landscape_edge`, and `rgsaddle_landscape_free` expose the downward pathway search. `RGSADDLE_LANDSCAPE_BOUND` reports a node or evaluation limit and does not return a pathway. `rgsaddle_landscape_run` takes the geometry-aware Hessian action `rgsaddle_hessian_at_fn`, which minor 69 also adds with `rgsaddle_indexk_step_at`. `rgsaddle_hessian_fn` keeps its original signature from minor 37. `rgsaddle_landscape_report_t` has `flags` after `version`. The worked example is `tests/landscape.rs`.
From minor 70, `RGSADDLE_TANGENT_SPLINE` on `rgsaddle_band_project` and `rgsaddle_band_step` assembles the cubic-spline tangent and curvature spring, and `rgsaddle_spline_resample` places the interior images at equal steps of the normalized chord length. A band session refuses that tangent together with the simplified string, a solid-state cell, free-end adaptive NEB, or the exponential metric.
`rgsaddle_ocineb_step` reports the hand-off phase (band, tangent alignment, or minimum-mode translation).
`rgsaddle_connect` runs the double-ended search of `connect` to completion on two endpoint arrays and a host surface.
The surface writes energy and the Cartesian gradient for one image.
`rgsaddle_connect_report_t` carries the energy and the Euclidean gradient norm of the search.
It also carries the kind (`s`, `u`, or `f`), the Hessian index and the evaluation count.
That count includes the finite differences that decide a step or an index.
The Lanczos acceptance does not build a dense Hessian.
Band flag bit 2 is `RGSADDLE_BAND_CI_AFTER`.
It arms the climbing image under `ci_trigger_force` or `ci_trigger_factor` times the first force.
Either of the two may be 0.
From minor 21, band flag bit 3 is `RGSADDLE_BAND_NO_MIN_MODE`.
It turns off the min-mode refinement of the climbing image.
Otherwise the refinement runs and reports its curvature in `rgsaddle_report_t.curvature`.
`RGSADDLE_STALLED` reports a band step that left the band unchanged above `force_tol`.
From minor 42, `RGSADDLE_NOT_EQUILIBRATED` and `RGSADDLE_INSUFFICIENT_EFFECTIVE_SAMPLES` report constrained-ensemble tails that fail the drift guard or the effective-sample floor.
On the band, `RGSADDLE_METHOD_QUICKMIN` runs quick-min, and a min-mode session returns null for that code.
`rgsaddle_band_set_quickmin_variant` selects `RGSADDLE_QUICKMIN_RGMIN`, `_EON` or `_PER_ATOM`.
`rgsaddle_band_create_solid` builds a solid-state band.
`rgsaddle_band_step_solid` steps it on an `rgsaddle_solid_surface_fn`.
Cells go in and Cauchy stresses come out through `rgsaddle_solid_surface_request_t`.
`rgsaddle_band_cells` and `rgsaddle_band_solid_jacobian` read the solid-state band back.
The report's `evaluations` field took the place of `reserved` at minor 5.
`cargo cinstall` installs `include/rgsaddle.h` as written.


# Python package

The importable package is `rgsaddle`. `rgsaddle._core` is the nanobind extension. `import rgsaddle` does not import ASE.

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Name</th>
<th scope="col" class="org-left">Role</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left"><code>ForceGate</code>, <code>HessUpdate</code>, <code>IrcKind</code>, <code>MinModeKind</code>, <code>EigenDevice</code>, <code>ExpandKind</code>, <code>CellChart</code>, <code>RestrictedKind</code></td>
<td class="org-left"><code>enum.IntEnum</code> controls. Names and values match the previous module</td>
</tr>

<tr>
<td class="org-left"><code>NumericalError</code></td>
<td class="org-left"><code>ArithmeticError</code> for <code>RGSADDLE_NON_FINITE</code></td>
</tr>

<tr>
<td class="org-left"><code>last_error</code></td>
<td class="org-left">Thread-local text reserved for the C ABI last-error call. Empty unless an error text was set. <code>raise_status</code> reads it and then clears the slot</td>
</tr>

<tr>
<td class="org-left"><code>raise_status</code></td>
<td class="org-left">Map an <code>rgsaddle_status_t</code> value. The message is the stored text, or the status name when that text is empty. <code>RGSADDLE_OK</code> returns true</td>
</tr>

<tr>
<td class="org-left"><code>rgsaddle.ase.SurfaceAdapter</code></td>
<td class="org-left"><code>list[Atoms]</code> evaluated as <code>(ids, x) -&gt; (energies, gradients)</code> request rows (<code>claim</code>, <code>owners</code>, <code>reentrant</code>, <code>__call__</code>). Optional extra <code>ase</code></td>
</tr>

<tr>
<td class="org-left"><code>rgsaddle.Stencil</code></td>
<td class="org-left">Point search. <code>point</code>, <code>band</code>, <code>step</code>, <code>run</code>, <code>positions</code>, <code>mode</code>, <code>curvature</code>. <code>band</code> raises <code>ValueError</code> because band presets are not implemented. Step report is <code>StencilReport</code> (<code>status</code>, <code>max_force</code>, <code>curvature</code>, <code>iteration</code>, <code>evaluations</code>, <code>converged</code>)</td>
</tr>

<tr>
<td class="org-left"><code>rgsaddle.__version__</code></td>
<td class="org-left">Package version string returned by <code>rgsaddle._core.version</code></td>
</tr>

<tr>
<td class="org-left"><code>rgsaddle.THERMODYNAMIC_DERIVATIVE</code></td>
<td class="org-left">Observable name <code>potential.full_minus_harmonic</code> used by <code>rgsaddle.hyperplane_thermodynamic_integration</code></td>
</tr>

<tr>
<td class="org-left"><code>rgsaddle.hyperplane_thermodynamic_integration</code></td>
<td class="org-left">Constrained hyperplane thermodynamic integration, BAR, MBAR, and PAFI profile estimation from hyperplane sample batches</td>
</tr>

<tr>
<td class="org-left"><code>rgsaddle.saddle_index</code></td>
<td class="org-left">Standalone point saddle-index certification on a callback surface, returning <code>SaddleIndex</code> (<code>certified</code>, <code>index</code>, <code>converged</code>, <code>curvatures</code>, <code>evaluations</code>, <code>total_evaluations</code>, <code>escapes</code>)</td>
</tr>

<tr>
<td class="org-left"><code>rgsaddle.reparametrize</code></td>
<td class="org-left">Equal-arc-length reparametrization of a path with <code>scheme="linear"</code> or <code>"not-a-knot"</code> (<code>"spline"</code>)</td>
</tr>

<tr>
<td class="org-left"><code>rgsaddle.spline_resample</code></td>
<td class="org-left">Interior images at equal steps of the normalized cumulative chord length on the not-a-knot cubic (ABI minor 70)</td>
</tr>

<tr>
<td class="org-left"><code>rgsaddle._core</code></td>
<td class="org-left">Extension symbols: <code>ClaimSlot</code> (<code>__enter__</code>, <code>__exit__</code>), <code>MinMode</code> (<code>estimate</code>, <code>step</code>, <code>set_position</code>, <code>position</code>, <code>mode</code>, <code>saddle_index</code>, <code>set_saddle_check</code>), <code>SellaSaddle</code> (<code>step</code>, <code>position</code>), <code>version</code>, <code>project_band</code>, <code>dyneb_mask</code>, <code>halve_raised_energy</code>, <code>trust_update</code>, <code>RGSADDLE_FORCE_L2</code>, <code>RGSADDLE_FORCE_LINF</code>, <code>RGSADDLE_FORCE_MAX_ATOM</code>, <code>RGSADDLE_MINMODE_DIMER</code>, <code>RGSADDLE_MINMODE_LANCZOS</code>, <code>RGSADDLE_IRC_GS2</code>, <code>RGSADDLE_IRC_MOROKUMA</code>, <code>RGSADDLE_IRC_SELLA_SPHERE</code></td>
</tr>

<tr>
<td class="org-left"><code>rgsaddle.ase.NEB</code>, <code>AutoNEB</code>, <code>MinModeAtoms</code>, <code>Sella</code>, <code>saddle_index</code></td>
<td class="org-left">ASE drop-in wrappers for nudged elastic band (<code>interpolate</code>, <code>get_positions</code>, <code>set_positions</code>, <code>get_potential_energy</code>, <code>get_forces</code>, <code>saddle_index</code>), AutoNEB (<code>run</code> returns <code>AutoNEBReport</code> (<code>status</code>, <code>max_force</code>, <code>ci_index</code>, <code>surface_rows</code>, <code>n_images</code>, <code>iteration</code>, <code>positions</code>, <code>images</code>)), minimum-mode effective forces (<code>get_forces</code>), order-1 Sella optimization (<code>attach</code>, <code>run</code>), and saddle-index certification</td>
</tr>

<tr>
<td class="org-left"><code>rgsaddle.connect</code></td>
<td class="org-left">Run-to-completion double-ended search. <code>surface(x)</code> is the one-image <code>(energy, gradient)</code> callback. <code>ConnectReport</code> holds <code>position</code>, <code>energy</code>, <code>force_norm</code>, <code>kind</code>, <code>hessian_index</code>, and <code>evaluations</code></td>
</tr>

<tr>
<td class="org-left"><code>rgsaddle.lineb_solve</code></td>
<td class="org-left">Fixed-energy LI-NEB path and optional harmonic rate. <code>positions</code> is <code>(n_images, 3N)</code>, <code>atom_masses</code> one mass per atom, <code>molecular</code> selects coordinate expansion or molecule alignment, <code>beads=0</code> leaves <code>log_rate_over_reactant</code> as NaN. <code>LinebResult</code> holds <code>positions</code>, <code>action</code>, half periods, temperatures, call tallies, and the log rate</td>
</tr>

<tr>
<td class="org-left"><code>rgsaddle.Band</code></td>
<td class="org-left">Stepping NEB, simplified-string (<code>string=True</code>, <code>reparametrization</code>), and free-end adaptive NEB session. <code>step</code>, <code>run</code>, <code>positions</code>, <code>set_positions</code>, <code>evaluation</code>, <code>set_string</code>, <code>set_free_end</code>, <code>target_energy_residual</code>, <code>free_end_cuts</code>, <code>set_fixed_atoms</code>, <code>saddle_index</code>, <code>set_saddle_check</code>, <code>restart</code>, <code>reset</code></td>
</tr>

<tr>
<td class="org-left"><code>rgsaddle.GrowingString</code></td>
<td class="org-left">Stepping growing-string session (Peters et al. 2004) with hand-off to <code>StringSession</code> upon join. <code>step</code>, <code>run</code>, <code>n_images</code>, <code>nodes_added</code>, <code>joined</code>, <code>positions</code>, <code>ids</code>, <code>set_fixed_atoms</code>, <code>saddle_index</code></td>
</tr>

<tr>
<td class="org-left"><code>rgsaddle.SessionReport</code></td>
<td class="org-left">Step report of <code>Band</code>, <code>GrowingString</code>, <code>MinMode</code>, and <code>SellaSaddle</code> (<code>status</code>, <code>max_force</code>, <code>curvature</code>, <code>iteration</code>, <code>evaluations</code>, <code>ci_index</code>, <code>rotations</code>, <code>saddle_index</code>)</td>
</tr>

<tr>
<td class="org-left"><code>rgsaddle.ase.SimplifiedString</code></td>
<td class="org-left">ASE <code>Optimizer</code> driver and <code>Optimizable</code> for the simplified string method over <code>list[Atoms]</code> (<code>attach</code>, <code>interpolate</code>, <code>get_positions</code>, <code>set_positions</code>, <code>get_potential_energy</code>, <code>get_forces</code>, <code>run</code>, <code>saddle_index</code>), with <code>FixAtoms</code> and optional climbing image</td>
</tr>

<tr>
<td class="org-left"><code>rgsaddle.ase.GrowingString</code></td>
<td class="org-left">ASE driver for the growing string method over <code>list[Atoms]</code> (<code>attach</code>, <code>run</code>, <code>saddle_index</code>), preserving image IDs and batched calculator evaluation</td>
</tr>

<tr>
<td class="org-left"><code>rgsaddle.ase.Point</code></td>
<td class="org-left">ASE <code>Optimizer</code> driver for one point preset (<code>attach</code>, <code>run</code>). <code>run(fmax, steps)</code> is ASE's own loop</td>
</tr>

<tr>
<td class="org-left"><code>rgsaddle.ase.NEB</code></td>
<td class="org-left">ASE's band call. <code>k</code> is a scalar or one value per spring. <code>method</code> is <code>improvedtangent</code>, <code>aseneb</code>, <code>eb</code>, <code>dneb</code>, <code>spline</code>, or <code>string</code>. <code>spline</code> is the cubic-spline tangent and curvature spring. <code>string</code> uses that tangent with no spring and redistributes interior images on <code>set_positions</code>. <code>precon</code> and the DyNEB keywords <code>dynamic_relaxation</code>, <code>fmax</code>, and <code>scale_fmax</code> raise <code>NotImplementedError</code>. <code>interpolate(method='idpp')</code> and <code>idpp_interpolate</code> run <code>idpp_path</code>. <code>get_residual</code> is the largest absolute force component on the interior images. <code>get_forces</code> stores the infinity norm, a call before that raises <code>RuntimeError</code>, and the climbing image is omitted only when one was chosen</td>
</tr>

<tr>
<td class="org-left"><code>rgsaddle.ase.NEBOptimizer</code></td>
<td class="org-left">ASE's band optimizer. <code>method</code> is <code>ODE</code> (adaptive ODE12r), <code>static</code> (step <code>alpha</code>), or <code>krylov</code> (stored, and <code>run</code> raises <code>ValueError</code>). The constructor lowercases <code>method</code>. <code>run</code> does not. <code>run(fmax, steps)</code> stops on <code>NEB.get_residual</code>. The static loop returns false when <code>steps</code> is spent. The ODE path returns false only on <code>OptimizerConvergenceError</code></td>
</tr>

<tr>
<td class="org-left"><code>rgsaddle.ase.SingleCalculatorNEB</code></td>
<td class="org-left">ASE's deprecated shared-calculator band. Constructing it warns with ASE's <code>FutureWarning</code> and is <code>NEB(allow_shared_calculator=True)</code></td>
</tr>

<tr>
<td class="org-left"><code>rgsaddle.ase.DyNEB</code></td>
<td class="org-left">ASE 3.29 dynamic NEB. <code>dynamic_relaxation</code> (default true) leaves an image in place once its per-atom force is below <code>fmax</code>. <code>scale_fmax</code> scales that test by the distance from the highest-energy image. <code>dynamic_relaxation</code> false is the ordinary band</td>
</tr>

<tr>
<td class="org-left"><code>rgsaddle.idpp_path</code></td>
<td class="org-left">Image-dependent pair-potential path, endpoints included, shape <code>(n_images, 3N)</code>. <code>cell</code> is the 3x3 row-major lattice</td>
</tr>

<tr>
<td class="org-left"><code>rgsaddle.ase.DimerControl</code>, <code>MinModeAtoms</code>, <code>MinModeTranslate</code></td>
<td class="org-left">ASE's dimer. The mode and curvature come from <code>MinMode</code>. The trial step and conjugate gradient are ASE's. <code>order</code> other than 1 and a gauss displacement raise <code>NotImplementedError</code></td>
</tr>
</tbody>
</table>

`SurfaceAdapter` keeps `get_forces(apply_constraint=False)` on `raw_forces`. The returned gradient is minus `get_forces()` after constraints. The adapter evaluates the given rows in order on the calling thread. `owners()` and `reentrant()` are what a driver uses to group. A second call while the first holds the adapter raises `RuntimeError`.


# Tests that pin the contract

-   `tests/ocineb.rs` — the hand-off waits for a stable climbing image, restores it when the curvature is positive, and reaches the same double-well saddle as a climbing-image band. `tests/ocineb_abi.rs` runs `tests/c/ocineb_abi.c`
-   `tests/connect_driver.rs` — two-barrier toy, the C ABI, LJ38 pair 0 and pair 39, and ignored release samples of the 51-pair connect score and the CI-NEB templates on 20 pairs
-   `tests/double_well.rs` — band finds the origin saddle, endpoints pinned
-   `tests/simplified_string.rs` — simplified string method of E, Ren, and Vanden-Eijnden (2007) on Müller-Brown (spline vs linear MEP and both saddles against analytic Newton and CI-NEB), across `FIRE`, `L-BFGS`, `QuickMin`, `Bb`, and `ODE12r`, with fixed atoms, periodic cells, exponential preconditioning, image threads, and step guards
-   `tests/growing_string.rs` — growing string method of Peters, Heyden, Bell, and Chakraborty (2004) on a 2D curved channel with a central obstacle and on Müller-Brown against CI-NEB, verifying frontier tangent constraints, configurable node-add criteria, separate fragment relaxation, join hand-off to `StringSession`, and preserved `ImageIds` and batching
-   `tests/free_end_neb.rs`: free-end adaptive NEB of Zhang, Zhang, Ye, and Zheng (2016) and Zhu et al. (2007) on a narrow barrier along a long tilted valley (`FreeEndSide::Last` and `FreeEndSide::First`), recovering the fixed-end CI-NEB saddle and barrier height with a third as many images, plus C ABI and Python parity.
-   `tests/band_refine.rs`: a FIRE band steps with L-BFGS when the convergence force is at or under the `StepRefine` threshold and keeps the primary stepper otherwise. `tests/band_refine_abi.rs` runs `tests/c/band_refine_abi.c`.
-   `tests/softmode_seeding.rs`: soft-mode-subspace-first direction seeding (`soft_mode_then_full_directions`, `spectral_gap_rank`) against isotropic random starts on coupled wells and Müller-Brown.
-   `tests/makri_vacancy.rs` — Makri, Ortner, and Kermode (2019) Table II 107-atom Morse Cu fcc vacancy counts for ODE12r and ODE12r-Exp, plus simplified string (spline, linear, and climbing spline) barrier parity with NEB
-   `tests/autoneb.rs` — ASE AutoNEB decisions and the Müller-Brown insertion sequence. `tests/autoneb_gate.rs` hashes the insertion list and the trajectory at 1, 2, 3 and 8 threads, and an ignored release run compares AutoNEB with CI-NEB at one force-call budget. Strong scaling of a full AutoNEB run is an ignored test in the crate.
-   `tests/autoneb_abi.rs` runs `tests/c/autoneb_abi.c`: `rgsaddle_autoneb_run` on a double well reports `status` 1 with `max_force` under the stop and a repeatable `iteration` count, reports `status` 0 and `iteration` equal to the step budget when the climb is cut short, and returns `RGSADDLE_INVALID_PARAMETER`, `RGSADDLE_NULL_SURFACE`, `RGSADDLE_SHAPE`, `RGSADDLE_NON_FINITE`, `RGSADDLE_ABI_MISMATCH`, and `RGSADDLE_SURFACE_FAILED` for the matching bad inputs.
-   `tests/minmode_saddle.rs` — dimer / Lanczos
-   `tests/band_workspace.rs`: a warm band step allocates as often at 13
    images as at 5, for five spring, projection and optimizer settings,
    and the trajectories keep the hashes of the allocating code.
-   `tests/band_image_threads.rs`: trajectories and surface rows at 2, 3,
    8 and 16 image threads are bitwise those of one thread, each call is
    split into contiguous blocks, the first failing image is reported, and
    zero threads is refused.
-   `tests/band_repeat_evaluations.rs`: no band reaches the surface twice
    under the trust-region stepper on coupled double wells or under ODE12r
    on LJ38 through its stall.
-   `tests/minmode_workspace.rs`: outside the surface,
    a dimer estimate allocates as often at eight rotations as at two,
    and Lanczos as often at eight Krylov products as at two,
    and five searches keep the hashes of the allocating code.
-   `tests/nichols_saddle.rs` — Nichols step against i-PI on the same Hessian and gradient, Muller-Brown saddles, quadratic saddle
-   `tests/prfo_saddle.rs` — partitioned RFO against the augmented Hessian, trust sphere changes the mode ratio
-   `tests/sella_symmetrized_ritz.rs`: the Sella saddle walk on a linear gradient field with an antisymmetric Jacobian part.
    With the atom ceiling off, the walk keeps its evaluation counts and positions bit for bit.
    `symmetrized_ritz` stops the first finite-difference diagonalization two evaluations sooner.
-   `tests/sella_atom_ceiling.rs`: a six-coordinate Cartesian saddle search with `small_system_atom_ceiling` on.
    No atom moves past the ceiling from the first step, and the ceiling never passes its starting Euclidean radius.
    The search still reaches the saddle.
    A twelve-coordinate search and a search without iterative diagonalization step the same with the option on or off.
    The default takes the ceiling.
-   `tests/c_abi.rs` — header and symbols with `capi`
-   `tests/solid_bain_lj.rs` — `tests/c/solid_bain_lj.c` runs the solid-state band from bcc to fcc for switched LJ on the two-atom tetragonal cell through `rgsaddle_band_create_solid`; the climbing image matches the relaxed scan of `validation/bain_lj_reference.py` in c/a within 2.5e-4 and in barrier within 1.2e-6 eV/atom, and every image keeps a diagonal tetragonal cell and its fractional coordinates
-   `tests/solid_dimer_barrier.rs` — the variable-cell dimer and the solid-state band meet on the shear-coupled double well; a frozen cell reproduces the one-dimensional barrier
-   `tests/ssneb.rs` — a frozen `SolidNeb` matches the ordinary band and CI-NEB; the variable-cell double well barrier is 1
-   `tests/ssneb_validate_cu.rs` — release CI-SS-NEB on the 109-atom Mishin cell against `benchmarks/solid_dimer_cu.csv`
-   `tests/mishin_cu.rs` — Mishin copper, four-atom cohesive energy and virial, the <100> to <110> dumbbell saddle under shear, and the ignored vacancy/SIA catalogue
-   `tests/pafi_analytic.rs` — PAFI on a 1D barrier, on a 2D well with a known transverse free-energy shift, and on a bent valley whose spline hyperplanes have a quadrature free energy; a harmonic plane sampled exactly at a step where Euler–Maruyama is 2.5 times too wide; the basin test keeps a plane of two wells on the basin of the path and refuses one that cannot hold it; the spline path reproduces a cubic and reverses exactly; the reverse barrier of an integrated profile
-   `tests/pafi_reference.rs` — the `eam/fs` reader of `tests/common/eam_fs.rs` against LAMMPS on a synthetic table, the integral of the reference `ResultsProcessor` against scipy on the gradients of the reference `tests/test_pafi.py`, the benchmark pipeline on a small hop, and, behind `benchmarks/pafi_reference.sh`, the ignored rerun of the reference W vacancy and Fe dumbbell examples and six W planes sampled at `μ dt` of 10⁻³ and 8 × 10⁻³
-   `tests/pafi_swinburne_fast.rs` and `tests/pafi_swinburne_full.rs` — the pinned public PAFI Fe gradient fixture in default CI and the ignored W-vacancy statistical regression through `rgpot/lammpc`
-   `tests/bennett_kramers.rs` — Bennett–Chandler `κ = 1` for a short Newtonian crossing, and the Kramers value at `γ = 2`
-   `tests/defect_pair_rates.rs` — direction labels, equal toward/away rates on a flat control, and the catalogue round trip
-   `tests/ohtst_estimators.rs` — OH-TST force, torque and centroid torque against closed forms on a Gaussian and line quadrature on LEPS; control variates exact on the Gaussian and unbiased on LEPS
-   `tests/ohtst_transmission.rs` — `κ` by both countings against Kramers' closed form on a plane normal to the barrier and on a turned one, with and without masses
-   `tests/ohtst_bligaard_jonsson.rs` — the Eckart angles of Bligaard and Jónsson against line quadrature, the optimizer at both maxima, the work between them, and the ratio of `κ` at the two maxima against the quadrature ratio of their `k_TST`
-   `tests/ohtst_leps.rs` — the dimer-seeded plane, the climb, `Z_H/Z_R` by parallel planes and by crossings, the rate against quadrature and harmonic TST, the approach to harmonic TST at low temperature, `κ k_TST` on two lines with end states by side and by quench, and the work along the PAFI planes of the minimum energy path by integration, BAR and MBAR with rigid and Gaussian maps
-   `tests/ohtst_skewed_well.rs` — the first-order correction, the optimum and rate at high temperature with the work along the optimizer's trace, `Z_H/Z_R` against two-dimensional quadrature, `κ k_TST` on two lines, and the work along the PAFI planes of the minimum energy path
-   `tests/ohtst_coupled_modes.rs` — 48 coupled modes against product quadrature: the gradient frame, the first-order coefficients, plane free energies, control variates, `Z_H/Z_R`, `ln κ + ln Z` on two planes, and the work to the saddle plane with rigid and Gaussian maps
-   `validation/` — sympy and sollya checks of the dimer rotation, the Lanczos residual bound, and the finite-difference curvature; `bain_lj_reference.py` (numpy, scipy) scans the relaxed Bain path that `tests/c/solid_bain_lj.c` checks against, and `ssneb_bain_cell.py` (sympy) checks its strain parametrisation and Jacobian
