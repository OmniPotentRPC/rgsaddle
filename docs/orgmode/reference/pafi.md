Full API documentation for every item below is on [docs.rs/rgsaddle](https://docs.rs/rgsaddle), under `rgsaddle::pafi`; `cargo doc --open` builds the same pages from a checkout. This page is the inventory: every public configuration and result type the module exports, and the functions that connect them. `tutorials/pafi` walks through using them; `explanation/pafi-ohtst` has the theory.


# Path types

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
<td class="org-left"><code>ArcPath</code></td>
<td class="org-left">A cubic-spline path through ordered frames. <code>from_positions</code> builds it from a list of Cartesian configurations, reactant first, parameterized by the summed chord lengths from the first; <code>new</code> takes frames with a parameter of the caller's; <code>reversed</code> is the same curve run from the other end; <code>bounds</code> gives the parameter of the first and last frame, <code>arcs</code> every frame's parameter, and <code>geometry(arc)</code> the position, tangent, curvature and jerk of the spline at a parameter inside <code>bounds</code>, and <code>SaddleError::Invalid</code> outside them</td>
</tr>

<tr>
<td class="org-left"><code>PathFrame</code></td>
<td class="org-left">One frame of a path: <code>positions</code> (the Cartesian configuration) and <code>arc</code> (its path parameter; the chord length from the first frame for <code>from_positions</code>)</td>
</tr>

<tr>
<td class="org-left"><code>PathGeometry</code></td>
<td class="org-left"><code>positions</code> (<code>X(s)</code>), <code>tangent</code> (<code>X'</code>), <code>curvature</code> (<code>X''</code>), <code>jerk</code> (<code>X'''</code>, constant on each spline segment) and <code>arc</code> (<code>s</code>) at one point along a path</td>
</tr>
</tbody>
</table>


# Fixed-energy trajectory

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
<td class="org-left"><code>TrajectoryPoint</code></td>
<td class="org-left">One sample of inverted-potential motion on a normalized path: <code>time</code> from the left turning point, the normalized <code>arc</code>, and <code>velocity</code> (<code>d(arc)/dt</code>)</td>
</tr>

<tr>
<td class="org-left"><code>FixedEnergyTrajectory</code></td>
<td class="org-left">Mass-weighted path and the parallel force along it. Knots are the normalized polyline arc length of the mass-weighted images</td>
</tr>
</tbody>
</table>


# Sampling configuration and results

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
<td class="org-left"><code>PafiConfig</code></td>
<td class="org-left">Non-exhaustive; chain <code>with_*</code> setters from <code>default</code>. Defaults: <code>thermal = 0.02585</code> (300 K in eV), <code>mobility = 1.0</code>, <code>time_step = 1e-3</code>, <code>integrator = PafiIntegrator::LeimkuhlerMatthews</code>, <code>mean_force = PafiMeanForce::ReferenceCode</code>, <code>thermalize = 200</code>, <code>stride = 5</code>, <code>samples = 200</code>, <code>blocks = 5</code>, <code>trajectories = 1</code>, <code>seed = 1</code>, <code>fix_com = true</code>, <code>retain = 16</code>, <code>target_error = None</code>, <code>max_rounds = 4</code>, <code>region = None</code>, <code>max_jump = Some(0.4)</code>, <code>quench = RelaxConfig::default().with_method(Method::Ode12r)</code>. <code>thermal</code> is <code>kT</code> in the surface's energy unit; <code>mobility</code> and <code>time_step</code> set the overdamped Langevin step; <code>integrator</code> picks <code>PafiIntegrator</code> and <code>mean_force</code> picks <code>PafiMeanForce</code>; <code>thermalize</code> is the minimum warm-up before the automated equilibration test, <code>stride</code>, <code>samples</code> and <code>blocks</code> control the production run and its block standard error, and <code>trajectories</code> runs that many independent trajectories per plane; <code>seed</code> fixes the noise stream; <code>fix_com</code> removes the mean Cartesian drift on systems with at least two atoms; <code>retain</code> keeps that many recorded configurations on each plane for later reuse, evenly spaced from the first sample through the last (both ends kept when it is greater than one), and it must not exceed <code>samples</code>; <code>region</code> restricts the walk to a hard <code>Region</code>; <code>max_jump</code> and <code>quench</code> (a <code>RelaxConfig</code>) reject a stretch of the walk whose in-plane minimum has moved more than <code>max_jump</code> from the path's own, so the walk stays in the basin of the path; <code>target_error</code> and <code>max_rounds</code> run extra sampling rounds on every plane, inverse-variance combined with the first, until the barrier error meets the target or the round budget is spent (<code>howto/pafi-ohtst-restarting</code>)</td>
</tr>

<tr>
<td class="org-left"><code>PafiIntegrator</code></td>
<td class="org-left"><code>LeimkuhlerMatthews</code> (the default, second order for stationary averages) or <code>EulerMaruyama</code> (the direct step of the paper's Eq. (9))</td>
</tr>

<tr>
<td class="org-left"><code>PafiMeanForce</code></td>
<td class="org-left"><code>ReferenceCode</code> (the default: the first term of the paper's Eq. (8), as LAMMPS <code>fix pafi</code> averages it) or <code>PublishedEquation8</code> (both terms, the second being \(k T\, \mathrm{d}/\mathrm{d}s\, \ln(\lvert\psi\rvert / \lvert X'\rvert)\))</td>
</tr>

<tr>
<td class="org-left"><code>HyperplaneAverage</code></td>
<td class="org-left">One plane's result: <code>gradient</code> (<code>dF/ds</code>), its <code>error</code>, the mean <code>psi</code> (the correction term, zero on a straight path), <code>samples</code> recorded (summed over rounds), the integrated autocorrelation time <code>tau</code> in samples, <code>n_eff</code> (<code>(samples - discard) / max(1, 2 tau)</code>), total force <code>evaluations</code>, the <code>retained</code> configurations on the plane, the <code>equilibration</code> report and the number of stretches <code>rejected</code> for leaving the basin (<code>howto/pafi-ohtst-checking-convergence</code> covers reading <code>tau</code> and <code>n_eff</code>)</td>
</tr>

<tr>
<td class="org-left"><code>FreeEnergyProfile</code></td>
<td class="org-left">The potential of mean force integrated from the first plane: per-plane <code>arc</code>, <code>gradient</code> and <code>gradient_error</code>, the trapezoid-integrated <code>free_energy</code> and its propagated <code>free_energy_error</code>, the <code>barrier</code> (the profile's maximum, with the reactant held at zero) and its <code>barrier_error</code>, the index of that maximum in <code>dividing</code>, and total recorded <code>samples</code> across every plane and round. <code>reverse_barrier</code> is <code>F(dividing) - F(last)</code> with the trapezoid error of the planes between them. <code>balanced(end_gap)</code> and <code>anchored(end_gap)</code> turn that pair into a <code>BalancedBarriers</code></td>
</tr>

<tr>
<td class="org-left"><code>BalancedBarriers</code></td>
<td class="org-left">Both barriers of one path, made to differ by the free-energy difference of its ends. <code>forward</code> is the barrier above the first end. <code>reverse</code> is the barrier above the last end, and it sits <code>end_gap</code> below <code>forward</code>. <code>error</code> is the standard error of both. <code>drift</code> is <code>forward - reverse - end_gap</code> of the numbers that went in: what the integral adds to the free-energy difference of the ends. <code>end_gap</code> is the free energy of the last end less that of the first, known apart from this profile; zero is a path between two equivalent states. Both barrier errors must be positive, and every input must be finite, or the call returns <code>SaddleError</code>. Use <code>FreeEnergyProfile::balanced(end_gap)</code>, or <code>balanced_barriers((forward, forward_error), (reverse, reverse_error), end_gap)</code>, when the forward barrier and the reverse barrier plus <code>end_gap</code> are two estimates of one barrier. The result is their inverse-variance mean, the reverse barrier <code>end_gap</code> below it, and an error with half of <code>drift</code> added in quadrature. Use <code>FreeEnergyProfile::anchored(end_gap)</code>, or <code>anchored_barriers</code> with the same arguments, when a second integration from the other end shows that the drift lies past the dividing surface: the forward barrier and its error are kept, and the reverse barrier is the forward one less <code>end_gap</code>. <code>drift</code> is still filled in, so a caller can see what the barrier from the last plane would have added. Read <code>forward</code>, <code>reverse</code> and <code>error</code> as the two directions of the hop; <code>howto/defect-pair-rates</code> does that for a path that ends in the product minimum</td>
</tr>
</tbody>
</table>


# Functions

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Function</th>
<th scope="col" class="org-left">Role</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left"><code>average_hyperplane</code></td>
<td class="org-left">Samples one hyperplane by overdamped Langevin dynamics restricted to the plane, with <code>PafiConfig::trajectories</code> trajectories, and returns its <code>HyperplaneAverage</code></td>
</tr>

<tr>
<td class="org-left"><code>average_hyperplane_with_sampler</code></td>
<td class="org-left">The same estimator over any <code>HyperplaneSampler</code>, such as a molecular-dynamics engine</td>
</tr>

<tr>
<td class="org-left"><code>pafi_profile</code></td>
<td class="org-left">Samples every arc length in parallel on rayon's global thread pool and integrates to a <code>FreeEnergyProfile</code>, returning it alongside every plane's <code>HyperplaneAverage</code>; runs extra rounds when <code>PafiConfig::target_error</code> is set</td>
</tr>

<tr>
<td class="org-left"><code>integrate_profile</code></td>
<td class="org-left">Trapezoid-integrates a set of arc lengths, gradients and their errors directly, with no sampling, into a <code>FreeEnergyProfile</code></td>
</tr>

<tr>
<td class="org-left"><code>balanced_barriers</code></td>
<td class="org-left"><code>balanced_barriers((forward, forward_error), (reverse, reverse_error), end_gap)</code> returns a <code>BalancedBarriers</code> whose <code>forward</code> is the inverse-variance mean of the forward barrier and of <code>reverse + end_gap</code>. <code>FreeEnergyProfile::balanced</code> is that call on <code>barrier</code> / <code>barrier_error</code> and <code>reverse_barrier</code></td>
</tr>

<tr>
<td class="org-left"><code>anchored_barriers</code></td>
<td class="org-left"><code>anchored_barriers((forward, forward_error), (reverse, reverse_error), end_gap)</code> returns a <code>BalancedBarriers</code> that keeps the forward barrier and its error and sets <code>reverse</code> to <code>forward - end_gap</code>. <code>FreeEnergyProfile::anchored</code> is that call on a profile. Use it instead of <code>balanced_barriers</code> when the drift lies past the dividing surface</td>
</tr>

<tr>
<td class="org-left"><code>zero_temperature_barrier</code></td>
<td class="org-left">Integrates the potential's own gradient along a path with no sampling (<code>PafiConfig::thermal</code> plays no part): a check of the path itself, exact on a converged minimum-energy path</td>
</tr>

<tr>
<td class="org-left"><code>arc_at_time</code></td>
<td class="org-left">Cubic Hermite arc at a time on a half-trajectory of <code>TrajectoryPoint</code>. An exact node time returns that node's arc; between nodes the interpolant uses both endpoint velocities</td>
</tr>

<tr>
<td class="org-left"><code>attempt_frequency</code></td>
<td class="org-left">Not part of <code>rgsaddle::pafi</code> itself (it is exported from <code>rgsaddle::defect_pair</code>), but the usual prefactor for turning a PAFI barrier into a rate: <code>nu = angular * sqrt(lambda / mass) / (2 pi)</code> from a curvature <code>lambda</code> along the path tangent at the reactant. <code>rgsaddle::arrhenius</code> (from <code>rgsaddle::strain_rate</code>, fixed to eV and kelvin) is the wrong call for a reduced-units surface; multiply <code>nu</code> by <code>exp(-barrier / kT)</code> directly instead, as <code>tutorials/pafi</code> does</td>
</tr>
</tbody>
</table>


# What the tests check

`tests/pafi_analytic.rs` checks the sampled profile on a one-dimensional barrier and on a two-dimensional well with a known transverse free-energy shift, against the closed form the shift gives. `examples/pafi_tutorial.rs` runs the same kind of check, on a different surface, with numbers printed rather than asserted; `tutorials/pafi` walks through it.
