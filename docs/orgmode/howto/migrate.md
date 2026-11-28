The Rust sessions are the source of the mechanics. The Python classes in `rgsaddle.ase` follow the call shape of ASE's nudged elastic band, minimum-mode atoms, and Sella's `run(fmax, steps)`. Where the numbers differ, the difference is the one the tests lock in.


# From 0.5.x to 0.6.0

A converged band, minimum-mode, or OCI-NEB session now certifies the saddle index by default.
Those finite-difference evaluations count in the step report, so force-call totals can increase even when the search takes the same path.
Use `with_saddle_check(None)` on `BandConfig` or `MinModeConfig` when reproducing 0.5.x counts.

`BandReport`, `MinModeReport`, and `OcinebReport` add `saddle_index`.
Code that constructs or exhaustively matches one of those reports must accept the new field.
The C ABI introduced the saddle-check structs and functions at minor 40; a config stamped with an older minor keeps the check off.
Python's `_core.MinMode` enables the check by default and returns it through `SessionReport.saddle_index` and `MinMode.saddle_index()`.


# From ASE's NEB

`rgsaddle.ase.NEB(images, k=0.1, climb=False)` takes the same list of `Atoms` with calculators. `k` is one spring or one constant per segment. `interpolate()` places the interior images on the straight line between the ends, and `interpolate(method='idpp')` then runs `idpp_path`. `mic` uses the minimum image for the straight line. `get_positions` and `get_forces` are the interior images only, and `ase.optimize` (`BFGS`, `FIRE`, `MDMin`) drives them through `__ase_optimizable__`. `get_potential_energy` is the highest interior energy. `NEB.get_residual` is the largest absolute component of the projected force on the interior images, stored by `get_forces` as an infinity norm per image. The climbing image is omitted only when one was chosen. A call before `get_forces` raises `RuntimeError`. One interior image that does climb leaves that list empty and the maximum raises `ValueError`. `NEBOptimizer` (`howto/neb-optimizer`) stops on that residual.

The spring constant defaults to 0.1, ASE's value. A `BandSession` still defaults to 5, eOn's value. Pass the number you mean.

`method='improvedtangent'` and `method='aseneb'` build the same force. The tangent is a unit vector, so ASE's extra division by the tangent norm in the `aseneb` climbing step is 1. `eb` keeps the full spring. `dneb` adds the doubly nudged correction. `spline` takes the unit tangent and the curvature spring from the not-a-knot cubic through the images in normalized chord length (Makri, Ortner, and Kermode, 2019). `string` uses that tangent with no spring, and `set_positions` places the interior images at equal steps of the same parameter. A `precon` raises `NotImplementedError`. The exponential metric stays `BandConfig.exp_precon` on the Rust session.

`climb=True` climbs on the call, with no arming delay. It does not climb when a higher endpoint slopes down into the band. ASE climbs the highest image anyway. That image then feels the reflected force and walks up a slope that has no saddle between it and the endpoint. `tests/python/test_ase_dropin.py` and `tests/api_input.rs` lock the refusal. A `BandSession` still waits until the band force drops before it arms the climb; the one-shot projection does not, because the caller already asked for it.

`remove_rotation_and_translation=True` on a free, non-periodic band moves each image onto the previous one with ASE's rigid fit before the force is built, and `interpolate` fits the product onto the reactant. A fixed atom is not rotated, and a periodic image is not rotated: the mean Cartesian force is removed instead, and a fixed atom keeps that mean. `parallel=True` on one rank evaluates on the calling thread. `world.size > 1` raises `NotImplementedError`.

`FixAtoms` on the images is read into the band's fixed-atom mask, as `rgsaddle_band_set_fixed_atoms` takes it from C. The fixed atoms get zero force rows, keep the mean force even with `remove_rotation_and_translation=True`, and stay where the first image has them through `interpolate()`. Every image must fix the same atoms. Other constraints act through `get_forces()` as before.

`dynamic_relaxation`, `fmax`, and `scale_fmax` on `NEB` raise `NotImplementedError`. `DyNEB` takes them: an image under `fmax` is left in place, `scale_fmax` loosens that test with the distance from the highest-energy image, and the default band is unchanged. A variable cell in a periodic direction raises `NotImplementedError` as well.

`AutoNEB` reads `prefix000.traj` and `prefix001.traj` and grows the band with `autoneb_run`. The constructor arguments match `ase.mep.AutoNEB`. The stored images must agree on the atom count, the atomic numbers, `pbc`, and the periodic cell vectors, as ASE's `NEB` requires, and a mismatch raises the same error. The endpoints keep their own `Atoms` records, and a new interior image copies the start, as ASE copies the image left of the cut. `world=None` is `ase.parallel.world`. `parallel=True` on more than one rank raises `NotImplementedError`; with `parallel=False` every rank runs the same search and rank 0 alone writes the trajectories. `remove_rotation_and_translation=True` removes the mean force during the search and, for a free, non-periodic band without `FixAtoms`, fits each image onto the one before it by the Kabsch rotation before and after the search, which is the fit of `minimize_rotation_and_translation`. The steps are the band's FIRE solver. `maxstep` is read from the optimizer class signature when that parameter has a numeric default. A default of None uses the optimizer's documented `defaults['maxstep']`, which is the value ASE applies. `smooth_curve=True`, a non-FIRE optimizer, and `method` `spline` or `string` raise `NotImplementedError`. The final images are written back to `prefixNNN.traj`. Per-iteration trajectory files are not written. The initial fill matches ASE when `n_simul` is 1 or 3. ASE's `run()` returns nothing, and this one returns an `AutoNEBReport` (`status`, `max_force`, `ci_index`, `surface_rows`, `n_images`, `iteration`, `positions`, `images`). `status` is 1 when the last stage met its force stop and 0 when it ran out of steps. `max_force` is the largest force component of that stage and `iteration` is its step count. `positions` is the finished band, shaped `(n_images, 3N)`, and `images` is that band with its endpoints. A spent step budget is not an error, so a caller tests `status`. `all_images`, `status`, `max_force`, and `iteration` on the object hold the same values after the call. With a partially periodic first image, a slab or a wire, the minimum image wraps only the periodic axes, so the band is not folded across its vacuum.


# From ASE's dimer and from Sella

`rgsaddle.ase.MinModeAtoms(atoms, mode=...)` estimates a dimer or Lanczos mode and returns the effective force. Above the well the perpendicular force is dropped. Below it the force along the mode is reflected. An ASE optimizer can drive that object.

`MinModeAtoms(atoms, control)` with a `DimerControl` is ASE's dimer call. `displace()` with the default `displacement_method='gauss'` draws the same `RandomState` normals as ASE 3.29 when `random_seed` is set, one coordinate at a time. A mask, or a `displacement_center` with `radius` or `number_of_atoms`, limits that draw, and `mic` is the minimum image when the center is an atom index. `method='vector'` still applies a Cartesian displacement. `initial_eigenmode_method='gauss'` (the default) builds the mode from one further draw and restores the positions. `'displacement'` uses the move away from the original positions. `get_potential_energy` stores that guess without rotating it. `get_eigenmode`, `get_curvature`, and `get_barrier_energy` read the estimate, and `MinModeTranslate` takes the trial step and the conjugate-gradient direction from ASE while the rotated mode comes from this dimer. `order` other than 1, an `eigenmode_method` other than `dimer`, and a communicator with `size > 1` raise `NotImplementedError` naming the option. The mode's sign follows the guess. Convergence wants the force below `fmax` and a negative curvature.

`rgsaddle.ase.Sella(atoms, order=1)` is the saddle optimizer. `run(fmax, steps)` is ASE's loop. One step is the partitioned rational-function trust step, not a dimer translation. Masses come from `atoms.get_masses`.


# From eOn

The band tangents, springs, projections, and the climbing-image arm follow eOn's branch structure. `CiConfig::after(0.5)` is eOn's `ci_after`. `after_rel` is `ci_after_rel`. `always` climbs from the first evaluation. The minimum-mode dimer is eOn's improved dimer: `rotation_angle_tol` defaults to five degrees (`converged_angle`), and a rotation stops when the trial or the fitted angle falls under it, on `rotation_tol`, or at `max_rotations`. It does not infer the classic torque window; `rotation_angle_tol` 0 restores that window at the stock residual, and `set_torque_window(Some(true))` adds it to the improved dimer. Lanczos stops on the window at two vectors. eOn's Lanczos stops on the relative change of its lowest Ritz value (`lanczos_tolerance`), which `set_lanczos_rel_tol` sets here without the window.

eOn's client reads a config file and a potential plugin. Here the host passes positions and a surface. There is no system name, species table, or potential key in the session. A solid-state band takes one cell per image and the Cauchy stress through `BandSurface::eval_cells`.

Live Onsager-Machlup displacements are the atomic force over `2k`. A solid-state band has a cell block in the same vector, so that spring is refused at construction rather than applied as zeros.


# ASE 3.29 tests

`tests/python/ase_upstream` vendors ASE 3.29.0's NEB and dimer tests (LGPL-2.1-or-later) and substitutes `rgsaddle.ase.NEB`, `DyNEB`, `MinModeAtoms`, `DimerControl`, `MinModeTranslate`, `interpolate`, and `idpp_interpolate`. The Python CI job sets `RGSADDLE_REQUIRE_ASE=1`, so a missing ASE install fails collection. `ase/optimize/test/neb.py` is not vendored: it is a script with no pytest entry and no assertion.

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Result</th>
<th scope="col" class="org-left">Test</th>
<th scope="col" class="org-left">Why</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left">passing</td>
<td class="org-left"><code>test_neb</code></td>
<td class="org-left">Al(100)+Au, <code>FixAtoms</code>, BFGS, constrained and unconstrained forces</td>
</tr>

<tr>
<td class="org-left">passing</td>
<td class="org-left"><code>test_idpp</code></td>
<td class="org-left">ethane methyl permutation through <code>idpp_path</code></td>
</tr>

<tr>
<td class="org-left">passing</td>
<td class="org-left"><code>test_interpolate_images</code></td>
<td class="org-left">linear placement, <code>mic</code>, constraints, cell and scaled coordinates</td>
</tr>

<tr>
<td class="org-left">passing</td>
<td class="org-left"><code>test_COCu111</code>, <code>test_COCu111_2</code></td>
<td class="org-left">IDPP start, BFGS, shared calculator, trajectory</td>
</tr>

<tr>
<td class="org-left">passing</td>
<td class="org-left"><code>test_neb_methods</code> <code>aseneb</code> and <code>improvedtangent</code>, precon <code>None</code></td>
<td class="org-left">Morse vacancy barrier</td>
</tr>

<tr>
<td class="org-left">passing</td>
<td class="org-left"><code>test_deprecations</code>, <code>test_neb_default</code>, <code>test_raising_parallel_errors</code>, <code>test_no_shared_calc</code>, <code>test_init_checks</code>, <code>test_freeze_method</code></td>
<td class="org-left">constructor, shared calculator, image checks, frozen results</td>
</tr>

<tr>
<td class="org-left">passing</td>
<td class="org-left"><code>test_dimer_method</code></td>
<td class="org-left">Pt hop with <code>DimerControl</code>, <code>displace</code>, and <code>MinModeTranslate</code></td>
</tr>

<tr>
<td class="org-left">skipped</td>
<td class="org-left"><code>test_autoneb.py</code></td>
<td class="org-left">drives ASE AutoNEB image insertion, not these drivers</td>
</tr>

<tr>
<td class="org-left">skipped</td>
<td class="org-left"><code>test_dimer.py</code></td>
<td class="org-left"><code>FixBondLength</code> on a two-atom cluster, not the minimum-mode method</td>
</tr>

<tr>
<td class="org-left">skipped</td>
<td class="org-left"><code>test_get_neb_method</code></td>
<td class="org-left">asserts ASE's private NEB method classes, not the public force</td>
</tr>

<tr>
<td class="org-left">skipped</td>
<td class="org-left"><code>test_neb_optimizers</code></td>
<td class="org-left">drives ASE <code>NEBOptimizer</code> on a spline band with an Exp preconditioner</td>
</tr>

<tr>
<td class="org-left">skipped</td>
<td class="org-left"><code>test_precon_initialisation</code>, <code>test_single_precon_initialisation</code>, <code>test_list_precon_initialisation</code>, <code>test_precon_assembly</code></td>
<td class="org-left">asserts ASE Exp preconditioner assembly, not the band force</td>
</tr>

<tr>
<td class="org-left">skipped</td>
<td class="org-left"><code>test_spline_fit</code></td>
<td class="org-left">asserts ASE's cubic-spline fit object, not the band force</td>
</tr>

<tr>
<td class="org-left">skipped</td>
<td class="org-left"><code>test_integrate_forces</code></td>
<td class="org-left">integrates forces on ASE's spline fit, not the band force</td>
</tr>

<tr>
<td class="org-left">skipped</td>
<td class="org-left"><code>test_neb_methods</code> <code>spline</code> and <code>string</code>, and any precon</td>
<td class="org-left">those rows drive NEBOptimizer or an Exp preconditioner; the identity spline and string forces are <code>tests/python/test_ase_spline_string.py</code></td>
</tr>

<tr>
<td class="org-left">skipped</td>
<td class="org-left"><code>test_neb_tr</code></td>
<td class="org-left">FIRE step-count ratio follows ASE's partial BFGS IDPP path; the band force matches on one geometry and the barriers agree, but the aligned run is not faster by the factor 0.75</td>
</tr>

<tr>
<td class="org-left">passing</td>
<td class="org-left"><code>test_dynamic_neb</code></td>
<td class="org-left">Pt(111)+O, EMT, BFGS; dynamic and scaled DyNEB use fewer force calls than the plain band, and the barriers agree</td>
</tr>

<tr>
<td class="org-left">not supported</td>
<td class="org-left"><code>precon</code></td>
<td class="org-left">preconditioned spline and string forces are not implemented; the name is in the <code>NotImplementedError</code></td>
</tr>

<tr>
<td class="org-left">not supported</td>
<td class="org-left"><code>parallel</code> with <code>world.size &gt; 1</code></td>
<td class="org-left">no MPI image distribution</td>
</tr>

<tr>
<td class="org-left">not supported</td>
<td class="org-left"><code>idpp</code> <code>mic</code> on a partially periodic cell</td>
<td class="org-left"><code>idpp_path</code> wraps every axis</td>
</tr>

<tr>
<td class="org-left">not supported</td>
<td class="org-left"><code>idpp_interpolate</code> <code>optimizer</code> other than <code>MDMin</code> or <code>BFGS</code>, and a custom <code>traj</code> or <code>log</code></td>
<td class="org-left">the path is <code>idpp_path</code> and it writes no trajectory</td>
</tr>

<tr>
<td class="org-left">not supported</td>
<td class="org-left">gauss displacement, <code>order</code> other than 1, <code>eigenmode_method</code> other than <code>dimer</code></td>
<td class="org-left">the estimator is the dimer or Lanczos mode of order 1</td>
</tr>

<tr>
<td class="org-left">not supported</td>
<td class="org-left">variable cell in a periodic direction</td>
<td class="org-left">the band cell is one lattice</td>
</tr>
</tbody>
</table>
