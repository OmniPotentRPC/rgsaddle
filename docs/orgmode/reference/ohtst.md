Full API documentation for every item below is on [docs.rs/rgsaddle](https://docs.rs/rgsaddle), under `rgsaddle::ohtst`; `cargo doc --no-deps --features ohtst --open` builds the same pages from a checkout. This page is the inventory: every public configuration and result type the module exports. `howto/oh-tst` is the dense narrative reference for how the pieces fit together, with every citation; `tutorials/oh-tst` walks through using them once, end to end; `explanation/pafi-ohtst` has the theory.


# Coordinates and the plane

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
<td class="org-left"><code>MassWeighted</code></td>
<td class="org-left">Wraps a Cartesian <code>PointSurface</code> in mass-weighted coordinates, <code>q = M^(1/2) x</code>, through <code>MassMetric</code></td>
</tr>

<tr>
<td class="org-left"><code>FixedCell</code></td>
<td class="org-left">Holds a crystal's cell fixed, for one temperature at a time</td>
</tr>

<tr>
<td class="org-left"><code>Hyperplane</code></td>
<td class="org-left">A point and a unit normal, in mass-weighted coordinates, pointing from reactant to product. <code>from_min_mode</code> seeds one from a converged <code>MinModeSession</code> or Lanczos search, normal to the lowest Ritz vector of a converged saddle-index certificate at that point and otherwise to the session mode; <code>oriented_toward</code> fixes the sign toward a given configuration; <code>packed</code> / <code>from_packed</code> give the <code>[p; normal]</code> layout <code>PlaneSample::directional</code> also uses; <code>from_cartesian</code> / <code>to_cartesian</code> convert</td>
</tr>
</tbody>
</table>


# Sampling

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
<td class="org-left"><code>LangevinConfig</code></td>
<td class="org-left">Non-exhaustive; chain <code>with_*</code> setters from <code>default</code>. Defaults: <code>thermal = 0.02585</code>, <code>time_step = 0.05</code>, <code>friction = 1.0</code>, <code>integrator = PlaneIntegrator::Baoab</code>, <code>control = None</code>, <code>burn_in = 500</code>, <code>stride = 5</code>, <code>records = 2000</code>, <code>blocks = 20</code>, <code>seed = 1</code>, <code>retain = 16</code>, <code>hold_excluded = true</code>, <code>region = None</code>. <code>with_control</code> adds Stein control variates from a <code>HarmonicReference</code>. <code>sample_plane</code>'s own <code>start</code> argument (not a field of this config) continues a chain from a previous <code>PlaneSample::last</code> instead of the plane's own point (<code>howto/pafi-ohtst-restarting</code>)</td>
</tr>

<tr>
<td class="org-left"><code>PlaneIntegrator</code></td>
<td class="org-left"><code>Baoab</code> (projected BAOAB, the default, exact on a harmonic plane for any stable step) or <code>Ghmc</code> (generalized hybrid Monte Carlo, exact for any step)</td>
</tr>

<tr>
<td class="org-left"><code>HarmonicReference</code></td>
<td class="org-left">A reference point (usually the saddle) whose Hessian, reached only through Hessian actions, builds the zero-variance control variates <code>LangevinConfig::with_control</code> adds. <code>new(point)</code> defaults to <code>hvp_step = 1e-4</code> and <code>tolerance = 1e-8</code></td>
</tr>

<tr>
<td class="org-left"><code>sample_plane</code></td>
<td class="org-left"><code>sample_plane(surface, plane, start, config)</code> runs plane-restricted Langevin dynamics once and returns a <code>PlaneSample</code>; <code>start</code> is <code>None</code> to begin from the plane's own point, or <code>Some(configuration)</code> to continue a chain</td>
</tr>

<tr>
<td class="org-left"><code>PlaneSample</code></td>
<td class="org-left">One plane's sampled result: <code>normal_force</code> and <code>normal_force_error</code> (the negative normal component of the mean gradient), <code>grad_point</code>, <code>grad_normal</code> and <code>grad_normal_error</code> (<code>Uncontrolled</code> holds the raw values when control variates are applied), <code>centroid_torque</code> and <code>centroid_torque_error</code> (the torque about <code>centroid</code>), the in-plane <code>virial</code> and <code>virial_error</code> (compare against <code>free_dimensions * thermal</code> at equilibrium, <code>howto/pafi-ohtst-checking-convergence</code>), <code>tau</code> and <code>effective_samples</code> (the same autocorrelation pair as PAFI's <code>HyperplaneAverage</code>), <code>records</code>, <code>evaluations</code>, <code>acceptance</code>, <code>retained</code> configurations, and <code>last</code> (the final configuration, for restarting)</td>
</tr>

<tr>
<td class="org-left"><code>Uncontrolled</code></td>
<td class="org-left">The raw (uncontrolled) <code>normal_force</code>, its error, <code>grad_normal</code> and its error, and <code>centroid_torque</code> and its error, kept on <code>PlaneSample</code> whenever control variates were applied</td>
</tr>
</tbody>
</table>


# Engines

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
<td class="org-left"><code>HyperplaneSampler</code></td>
<td class="org-left">The trait every constrained-ensemble engine implements: it samples one <code>Hyperplane</code> under a <code>HyperplaneSampleBudget</code> and returns a <code>HyperplaneEnsemble</code> of <code>HyperplaneObservation</code>. <code>reversible_work</code>, <code>reversible_work_along</code> and PAFI's <code>average_hyperplane_with_sampler</code> take one</td>
</tr>

<tr>
<td class="org-left"><code>AnalyticSurfaceSampler</code></td>
<td class="org-left">The reference <code>HyperplaneSampler</code> for a <code>PointSurface</code>: projected BAOAB or GHMC dynamics. <code>new(&amp;surface, config)</code> then <code>with_start(position)</code> seeds the first plane, as the <code>start</code> argument of <code>sample_plane</code> does</td>
</tr>

<tr>
<td class="org-left"><code>HyperplaneDynamics</code></td>
<td class="org-left">The trait that shoots trajectories from a plane for <code>transmission_coefficient</code></td>
</tr>

<tr>
<td class="org-left"><code>AnalyticSurfaceDynamics</code></td>
<td class="org-left">The reference <code>HyperplaneDynamics</code> for a <code>PointSurface</code>: <code>new(&amp;surface)</code></td>
</tr>
</tbody>
</table>


# Plane optimization

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
<td class="org-left"><code>PlaneOptimizerConfig</code></td>
<td class="org-left">Non-exhaustive. Defaults: <code>sampler = LangevinConfig::default()</code>, <code>max_records = 16_000</code>, <code>norm_test = 0.5</code>, <code>max_translation = 0.1</code>, <code>max_rotation = 0.05</code>, <code>metric = PlaneMetric::Blocks</code>, <code>metric_rank = 32</code>, <code>rotation_basis = None</code>, <code>translate = true</code>, <code>rotate = true</code>, <code>recenter = true</code>, <code>confidence = 2.0</code>, <code>tolerance = 0.0</code>, <code>patience = 3</code>, <code>max_iterations = 100</code>, <code>max_evaluations = 10_000_000</code>. <code>with_sampler</code> sets the <code>LangevinConfig</code> each trial plane is sampled with; <code>with_rotation_basis</code> restricts the sphere of normals to a subspace</td>
</tr>

<tr>
<td class="org-left"><code>PlaneMetric</code></td>
<td class="org-left"><code>Blocks</code> (the default, weights the sphere by <code>l^2</code> with <code>l = max_translation / max_rotation</code>) or <code>Harmonic(HarmonicReference)</code> (the Hessian at a reference point as the sphere's tangent metric, so a step on a quadratic saddle is a Newton step)</td>
</tr>

<tr>
<td class="org-left"><code>PlaneOptimizer</code></td>
<td class="org-left"><code>new(seed, config)</code> then <code>run(surface)</code>: a Riemannian trust region on the product manifold of the sphere of normals and the offset along it, reporting the optimized plane in <code>PlaneOptimum::plane</code></td>
</tr>

<tr>
<td class="org-left"><code>PlaneOptimum</code></td>
<td class="org-left">The optimized <code>plane</code>, outer and inner (truncated conjugate-gradient) iteration counts, accepted planes, planes sampled, and force (and Hessian-action) evaluations</td>
</tr>

<tr>
<td class="org-left"><code>PlaneTrace</code></td>
<td class="org-left">One of the optimizer's accepted planes along its own iterates: <code>plane</code>, <code>normal_force</code> and its error, and the length <code>rotation</code> of the torque about the pivot, with its <code>rotation_error</code> and the <code>records</code> sampled. <code>PlanePath::trace</code> threads these into a path for <code>reversible_work_along</code></td>
</tr>

<tr>
<td class="org-left"><code>PlaneOptimizerStatus</code></td>
<td class="org-left">Why <code>PlaneOptimizer::run</code> stopped: <code>Converged</code> when the accepted-step criterion passed for the configured patience, <code>MaxIterations</code> when the outer iteration cap was reached with a finite last estimate, or <code>MaxEvaluations</code> when another sample would exceed the evaluation cap</td>
</tr>
</tbody>
</table>


# Reactant normalization

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
<td class="org-left"><code>BasinConfig</code></td>
<td class="org-left">Non-exhaustive. Defaults: <code>sampler = LangevinConfig::default()</code>, <code>spacing = 1.0</code>, <code>width = 8.0</code>, <code>cap = 10.0</code>, <code>refine = 0.5</code>, <code>resolution = 1e-4</code>, <code>refinements = 8</code>, <code>control = true</code>, <code>hvp_step = 1e-4</code>, <code>tolerance = 1e-10</code>. <code>with_sampler</code> sets the per-plane <code>LangevinConfig</code>; <code>cap</code>, <code>width</code> and <code>refine</code> bound the window on each side of the minimum and the leave-one-out spline refinement; <code>with_control</code> adds Stein control variates from the Hessian at the minimum</td>
</tr>

<tr>
<td class="org-left"><code>basin_density</code></td>
<td class="org-left"><code>basin_density(surface, minimum, normal, config)</code> samples planes parallel to <code>normal</code>, spreading from <code>minimum</code>, and returns a <code>BasinDensity</code></td>
</tr>

<tr>
<td class="org-left"><code>BasinDensity</code></td>
<td class="org-left"><code>plane</code> (through the minimum, with the integration normal), <code>thermal</code>, <code>density</code> (<code>Z_H(0)/Z_R</code>) and its <code>density_error</code>, the harmonic <code>spread</code> (<code>sigma_xi</code>) and <code>harmonic_density</code>, per-plane <code>offsets</code>, <code>slopes</code> and their errors, the integrated <code>profile</code>, the <code>window</code> and how each side <code>ends</code> (<code>BasinEnd</code>), the extrapolated-tail fraction <code>tail</code>, a leave-one-out <code>quadrature_error</code>, and <code>evaluations</code>. Implements <code>PlaneDensity</code>, the trait <code>ohtst_rate</code> reads a reactant normalization through</td>
</tr>

<tr>
<td class="org-left"><code>BasinEnd</code></td>
<td class="org-left"><code>Cap</code> (the free energy rose past <code>BasinConfig::cap</code>), <code>Turn</code> (it turned over: the true edge of the basin), or <code>Width</code> (the window ran out at <code>BasinConfig::width</code> before either)</td>
</tr>

<tr>
<td class="org-left"><code>PlaneDensity</code></td>
<td class="org-left">Trait <code>BasinDensity</code> and <code>ReactantDensity</code> both implement: <code>plane</code>, <code>thermal</code>, and <code>density()</code> returning <code>(value, error)</code>. <code>ohtst_rate</code> takes any <code>PlaneDensity</code></td>
</tr>

<tr>
<td class="org-left"><code>ReactantDensity</code></td>
<td class="org-left"><code>reactant_density</code> returns this: the 2001 paper's own crossing estimator (Eq. 22) on one sampled basin, kept for reference; its weight has a second moment that diverges at slow crossings, so <code>basin_density</code>'s mean-force integral is the one to use for anything but a direct comparison</td>
</tr>
</tbody>
</table>


# The path between planes, and the work along it

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
<td class="org-left"><code>PlanePath</code></td>
<td class="org-left"><code>legs</code> (straight legs between given planes), <code>curve</code> (PAFI-style planes through a curve of points, normal to a chord-length cubic spline, with optional clamped end tangents), <code>trace</code> (an optimizer's own accepted <code>PlaneTrace</code> iterates), <code>then</code> (joins two paths that share an endpoint plane)</td>
</tr>

<tr>
<td class="org-left"><code>ReversibleWorkConfig</code></td>
<td class="org-left">Non-exhaustive. Defaults: <code>sampler = LangevinConfig::default()</code>, <code>warm_burn_in = 100</code>, <code>nodes = 9</code>, <code>simpson = true</code>, <code>bar_samples = 128</code>, <code>transport = Transport::Rigid</code>, <code>mbar = false</code>, <code>estimate = WorkEstimate::Integration</code>, <code>min_overlap = 0.03</code>. <code>with_sampler</code> sets the per-node <code>LangevinConfig</code>; <code>nodes &gt;= 2</code> always, and <code>with_simpson(true)</code> (the default) requires an odd node count, so <code>with_nodes(2)</code> requires <code>with_simpson(false)</code>; <code>with_bar_samples</code> bounds samples used for BAR; <code>with_transport</code> picks <code>Transport</code>; <code>with_mbar(true)</code> turns on MBAR across every node; <code>with_estimate</code> picks the <code>WorkEstimate</code> that <code>ReversibleWork::work</code> reads; <code>min_overlap</code> is the two-state overlap threshold below which a node pair is named in <code>ReversibleWork::warnings</code></td>
</tr>

<tr>
<td class="org-left"><code>Transport</code></td>
<td class="org-left"><code>Rigid</code> (the default: the minimal rotation between neighbouring planes) or <code>Gaussian(GaussianTransport)</code>, built with <code>GaussianTransport::default().into()</code>: an optimal-transport map between the planes' fitted Gaussian moments, shrunk toward the rigid map by a James–Stein factor</td>
</tr>

<tr>
<td class="org-left"><code>GaussianTransport</code></td>
<td class="org-left">Settings of the Gaussian map: fitting <code>rank</code> (default <code>4</code>), <code>shrinkage</code> of each fitted covariance toward the identity (default <code>0.0</code>), and <code>shrink_to_rigid</code> (default <code>true</code>, James–Stein shrinkage of the fitted change toward the rigid one)</td>
</tr>

<tr>
<td class="org-left"><code>WorkEstimate</code></td>
<td class="org-left"><code>Integration</code> (default), <code>Bar</code> or <code>Mbar</code>: which value <code>ReversibleWork::work</code> (and so the rate functions) read</td>
</tr>

<tr>
<td class="org-left"><code>reversible_work</code></td>
<td class="org-left"><code>reversible_work(sampler, planes, config)</code> integrates <code>dA</code> along straight legs between the given planes, sampling them with any <code>HyperplaneSampler</code></td>
</tr>

<tr>
<td class="org-left"><code>reversible_work_along</code></td>
<td class="org-left"><code>reversible_work_along(sampler, path, config)</code> integrates <code>dA</code> along any <code>PlanePath</code></td>
</tr>

<tr>
<td class="org-left"><code>ReversibleWork</code></td>
<td class="org-left"><code>work</code> / <code>error</code> (the <code>WorkEstimate</code> choice) and <code>estimate</code> (which one); <code>integrated_work</code> / <code>integrated_error</code> (quadrature, always computed); <code>bar_work</code> / <code>bar_error</code> (<code>Option</code>, Bennett's acceptance ratio between neighbours); <code>mbar_work</code> / <code>mbar_error</code> (<code>Option</code>, only when <code>with_mbar(true)</code>); per-node <code>nodes</code> (<code>WorkNode</code>, each with its own <code>overlap</code> and <code>bar_work</code> / <code>mbar_work</code>); the MBAR <code>overlap</code> matrix; the sample <code>end</code>; total <code>evaluations</code>; and <code>warnings</code> naming any pair below <code>min_overlap</code> or a Gaussian map that fell back to the rigid one</td>
</tr>

<tr>
<td class="org-left"><code>WorkNode</code></td>
<td class="org-left">One node along a <code>PlanePath</code>: <code>leg</code> and <code>s</code> (position on the segment), the sampled <code>plane</code>, <code>derivative</code> (<code>dA/ds</code>) and its <code>error</code>, trapezoid <code>work</code> from the first plane, <code>bar_work</code> and <code>overlap</code> with the node before, <code>transport_weight</code> of the Gaussian map, and <code>mbar_work</code></td>
</tr>
</tbody>
</table>


# The rate and the transmission correction

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
<td class="org-left"><code>ohtst_rate</code></td>
<td class="org-left"><code>ohtst_rate(density, work)</code> with any <code>PlaneDensity</code> and a <code>ReversibleWork</code>: <code>k = sqrt(kT / 2*pi) * (Z_H/Z_R) * exp(-DeltaA/kT)</code>, unit mass in mass-weighted coordinates</td>
</tr>

<tr>
<td class="org-left"><code>OhTstRate</code></td>
<td class="org-left"><code>rate</code> and <code>error</code>, plus <code>log_rate</code> / <code>log_rate_error</code>, <code>flux</code> (<code>sqrt(kT/2*pi)</code>), <code>density</code> / <code>density_error</code> and <code>free_energy</code> / <code>free_energy_error</code>: the bare rate before any transmission correction, and the pieces that built it. <code>transmitted(&amp;transmission) -&gt; (f64, f64)</code> multiplies by any <code>TransmissionEstimate</code>'s <code>kappa</code> and propagates the error</td>
</tr>

<tr>
<td class="org-left"><code>RateMultiplicity</code></td>
<td class="org-left">Explicit, separate multiplicities of a rate: <code>paths</code> (equivalent reaction paths or hop directions) and <code>in_plane_images</code> (degeneracy of one symmetry-unique in-plane region)</td>
</tr>

<tr>
<td class="org-left"><code>TransmissionConfig</code></td>
<td class="org-left">Non-exhaustive. Defaults: <code>thermal = 0.02585</code>, <code>friction = 0.0</code> (Newtonian), <code>time_step = 0.05</code>, <code>steps = 400</code>, <code>check_every = 40</code>, <code>commit = 0</code>, <code>shots = 256</code>, <code>blocks = 16</code>, <code>seed = 1</code>, <code>masses = None</code>, <code>counting = Counting::FirstCrossing</code>, <code>end_state = EndState::Side</code>, <code>hold_excluded = true</code>. Setters <code>with_thermal</code>, <code>with_friction</code>, <code>with_time_step</code>, <code>with_steps</code>, <code>with_check_every</code>, <code>with_commit</code>, <code>with_shots</code>, <code>with_blocks</code>, <code>with_seed</code>, <code>with_masses</code>, <code>with_counting</code>, <code>with_end_state</code>, <code>with_hold_excluded</code> (the friction must match the dynamics <code>kappa</code> corrects)</td>
</tr>

<tr>
<td class="org-left"><code>Counting</code></td>
<td class="org-left"><code>Final</code> (the Bennett–Chandler reactive flux, counting only each trajectory's final state) or <code>FirstCrossing</code> (the default, the effective positive flux of van Erp, Moroni and Bolhuis, counting a reactive trajectory once even if it recrosses)</td>
</tr>

<tr>
<td class="org-left"><code>EndState</code></td>
<td class="org-left"><code>Side</code> (the default, which side of the plane a shot ends on) or <code>Quench(Box&lt;Basins&gt;)</code> (the basin a steepest-descent quench reaches, built with <code>Basins::...into()</code>)</td>
</tr>

<tr>
<td class="org-left"><code>Endpoint</code></td>
<td class="org-left"><code>Reactant</code> or <code>Product</code>: which side a shot ended on, under <code>EndState::Side</code></td>
</tr>

<tr>
<td class="org-left"><code>Basins</code></td>
<td class="org-left">Reference minima (<code>reactants</code>, <code>products</code>), a <code>StateDistance</code> (default <code>Euclidean</code>), <code>tolerance = 0.25</code>, optional <code>threshold = None</code>, and a <code>QuenchConfig</code> for <code>EndState::Quench</code></td>
</tr>

<tr>
<td class="org-left"><code>QuenchedTransitionRegion</code></td>
<td class="org-left">Hard transition region whose exterior is any basin reached by an rgmin steepest-descent quench. Membership comes from <code>Basins::assign</code></td>
</tr>

<tr>
<td class="org-left"><code>QuenchConfig</code></td>
<td class="org-left">Non-exhaustive. Steepest-descent quench settings used by <code>EndState::Quench</code>. Defaults: <code>method = Method::Ode12r</code>, <code>force_tol = 1e-4</code>, <code>max_steps = 5000</code>, <code>max_move = 0.1</code>, <code>initial_step = 0.1</code></td>
</tr>

<tr>
<td class="org-left"><code>StateDistance</code></td>
<td class="org-left"><code>Euclidean</code> (the default) or <code>Permutation(Box&lt;PermutationDistance&gt;)</code>, built with <code>PermutationDistance::...into()</code>, so relabelled like atoms count as one state</td>
</tr>

<tr>
<td class="org-left"><code>PermutationDistance</code></td>
<td class="org-left">The intensive permutation-invariant distance of Goswami and Jónsson, ChemPhysChem 27, e202500730 (2026), <a href="https://doi.org/10.1002/cphc.202500730">https://doi.org/10.1002/cphc.202500730</a>: <code>species</code> (which atoms may be exchanged), an optional periodic <code>cell</code>, optional <code>masses</code> when the coordinates are mass-weighted, and optional <code>atoms</code> subset</td>
</tr>

<tr>
<td class="org-left"><code>TransmissionEstimate</code></td>
<td class="org-left">Trait implemented by <code>TransmissionCoefficient</code> and <code>rgsaddle::bennett::Transmission</code>: <code>kappa() -&gt; (value, error)</code>. <code>OhTstRate::transmitted</code> takes any <code>TransmissionEstimate</code></td>
</tr>

<tr>
<td class="org-left"><code>transmission_coefficient</code></td>
<td class="org-left"><code>transmission_coefficient(dynamics, plane, starts, config)</code> shoots trajectories from a <code>PlaneSample</code>'s <code>retained</code> configurations with any <code>HyperplaneDynamics</code> and returns a <code>TransmissionCoefficient</code></td>
</tr>

<tr>
<td class="org-left"><code>plane_transmission</code></td>
<td class="org-left"><code>plane_transmission(surface, sample, config)</code> is the plain Bennett-Chandler shooting of a <code>PlaneSample</code>'s <code>retained</code> configurations under a <code>ShootingConfig</code>, with unit mass in mass-weighted coordinates</td>
</tr>

<tr>
<td class="org-left"><code>TransmissionCoefficient</code></td>
<td class="org-left"><code>kappa</code> and its <code>error</code>, plus the checkpoint <code>time</code> and <code>kappa_of_t</code> series, <code>shots</code> that reached a state on both halves, <code>unassigned</code> shots that did not, <code>reactive</code> shots and <code>recrossed</code> backward halves, and <code>evaluations</code>. Implements <code>TransmissionEstimate</code>, the trait <code>OhTstRate::transmitted</code> reads <code>kappa</code> through</td>
</tr>

<tr>
<td class="org-left"><code>anharmonic_correction</code></td>
<td class="org-left">First-order correction to harmonic TST, <code>ln(k / k_HTST) = c_1 * kT + O(kT^2)</code>, and the first-order offset of the optimal plane, from Hessian actions and directional derivatives at Gaussian probes; no full Hessian is formed. <code>AnharmonicConfig</code> defaults: <code>probes = 2000</code>, <code>step = 1e-2</code>, <code>hvp_step = 1e-4</code>, <code>krylov = 64</code>, <code>krylov_tolerance = 1e-8</code>, <code>seed = 1</code>. <code>AnharmonicCorrection</code> holds the first-order anharmonic coefficients <code>reactant</code> / <code>reactant_error</code> (<code>A_R</code> at the minimum) and <code>saddle</code> / <code>saddle_error</code> (<code>A^\ddagger</code> on the saddle plane), <code>slope</code> / <code>slope_error</code> (<code>S</code>), <code>curvature</code> (\(\lvert \lambda_u \rvert\)), <code>coefficient</code> / <code>coefficient_error</code> (<code>c_1</code>) and <code>evaluations</code>, with methods <code>log_ratio</code>, <code>fixed_log_ratio</code> and <code>offset</code></td>
</tr>

<tr>
<td class="org-left"><code>flux_mass</code></td>
<td class="org-left"><code>flux_mass(normal, masses)</code> is <code>1 / sum(n_i^2 / m_i)</code> for normalized <code>n</code>, the Cartesian flux mass along a normal; the 2001 paper's Eq. 24, <code>mu = sum(m_i n_i^2)</code>, agrees with it only for equal masses</td>
</tr>
</tbody>
</table>


# Thermodynamic integration on one plane

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
<td class="org-left"><code>HarmonicExpansion</code></td>
<td class="org-left">Second-order Taylor expansion of a <code>PointSurface</code> about a <code>HarmonicReference</code> point</td>
</tr>

<tr>
<td class="org-left"><code>ThermodynamicDerivative</code></td>
<td class="org-left">Observable <code>V_full - V_harmonic</code> at a sampled configuration</td>
</tr>

<tr>
<td class="org-left"><code>ThermodynamicSampler</code></td>
<td class="org-left">Records <code>V_full - V_harmonic</code> from sampled positions and attaches that observable to an inner sampler</td>
</tr>

<tr>
<td class="org-left"><code>ThermodynamicSurfaceSampler</code></td>
<td class="org-left">Projected BAOAB or GHMC sampler on <code>V_lambda</code> built from a harmonic Taylor expansion of the full surface</td>
</tr>

<tr>
<td class="org-left"><code>ThermodynamicIntegrationConfig</code></td>
<td class="org-left">Settings for fixed-plane lambda thermodynamic integration</td>
</tr>

<tr>
<td class="org-left"><code>ThermodynamicIntegrationNode</code></td>
<td class="org-left">One lambda node and its block estimate</td>
</tr>

<tr>
<td class="org-left"><code>ThermodynamicIntegrationEstimate</code></td>
<td class="org-left"><code>F_full - F_harmonic</code> on one fixed hyperplane</td>
</tr>

<tr>
<td class="org-left"><code>AnalyticThermodynamicSampler</code></td>
<td class="org-left">Analytic reference backend for <code>V_lambda = (1-lambda) V_harmonic + lambda V_full</code></td>
</tr>
</tbody>
</table>


# Exact samples of a quadratic plane

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
<td class="org-left"><code>QuadraticModel</code></td>
<td class="org-left">Quadratic potential <code>V(q) = energy + gradient·(q-center) + ½(q-center)ᵀ H (q-center)</code></td>
</tr>

<tr>
<td class="org-left"><code>HyperplaneObservable</code></td>
<td class="org-left">A named scalar measured at an exact Gaussian sample</td>
</tr>

<tr>
<td class="org-left"><code>ExactGaussianPlaneSampler</code></td>
<td class="org-left">Independent exact samples of a quadratic potential restricted to a plane</td>
</tr>
</tbody>
</table>


# What the tests check

`tests/ohtst_estimators.rs`, `tests/ohtst_transmission.rs`, `tests/ohtst_bligaard_jonsson.rs`, `tests/ohtst_skewed_well.rs`, `tests/ohtst_coupled_modes.rs` and `tests/ohtst_leps.rs` check every piece above against closed forms, line or product quadrature, or exact draws; the full list with the numbers each test checks is in `reference.org`'s main table and `howto/oh-tst`'s own. `examples/ohtst_tutorial.rs` runs the same kind of pipeline on a simpler, reflection-symmetric surface, with numbers printed rather than asserted; `tutorials/oh-tst` walks through it.
