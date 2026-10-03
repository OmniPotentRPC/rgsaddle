Explanation
===========

Band and minimum-mode
---------------------

The band and the minimum-mode search assemble a force and ask
``rgmin::Solver`` for one step. Hosts interleave trust regions, drift
budgets, acquisition, or a hybrid MMF between steps. ``run`` is a
loop over ``step``.

Index-1 Newton
--------------

``Index1Session`` searches an index-1 saddle on a Cartesian energy
Hessian. The displacement is Baker's restricted-step partitioned
RFO: a 2 by 2 rational-function problem maximizes the lowest mode,
a second rational-function problem minimizes the complement, and the
Besalu-Bofill scaling is solved so the Cartesian step lies inside
the trust sphere. Powell or Bofill updates the Hessian from the
gradient difference. The trust radius shrinks when that model
predicts the energy change poorly and grows when a step on the
sphere agrees with it.

``nichols_step`` is the separate i-PI level shift. A host that
already holds a spectrum can call it. ``rgsaddle_prfo_step`` is the
restricted-step partition. ``rgsaddle_hessian_powell`` and
``rgsaddle_hessian_bofill`` update the Hessian.

eOn branch structure
--------------------

Tangents, springs, projections, and the climbing-image force follow
eOn's ``NEBTangent`` / ``NEBSpringForce`` / ``NEBForceProjection``
with the same branches. The climbing image arms when the convergence
force falls under ``factor * baseline`` or under an absolute trigger.

Periodicity
-----------

Positions stay unwrapped Cartesian. ``Cell`` wraps per-atom
differences for the band mechanics through fractional coordinates
(rows are lattice vectors; a zero row is a non-periodic axis).
Neighbor lists, when a surface wants them, are vesin's job.

Evaluations per step
--------------------

A warm band step costs ``n_images - 2`` image evaluations. The
endpoint energies are evaluated once. The session keeps its last
evaluation, keyed on the interior positions, so a step that starts
where the previous one ended is assembled from it: a host resync
through ``set_positions`` (rows equal up to the minimum image keep the
session's own copy), an optimizer ``restart``, or a solver that
dropped its own cache costs nothing. ``BandReport.surface_rows``
counts what the surface evaluated. The climbing image is chosen on
the energies of the evaluation it climbs on, so the step that arms it
already climbs.

Steppers
--------

FIRE is the default, as in eOn's velocity NEB stepper. The projected
NEB force is not the gradient of the summed image energy the oracle
reports, so L-BFGS runs under rgmin's ``Accept::Step``: the two-loop
direction, capped per atom, with one evaluation and no energy test.

Minimum mode
------------

The dimer rotation is the modified Newton step of Heyden, Bell, and
Keil (2005) and Kastner and Sherwood (2008): one trial gradient fixes
the Fourier model of the curvature in the rotation plane, the optimal
angle is closed form, and the gradient at the rotated dimer is the
exact (on a quadratic) combination of the two measured ones, so each
rotation costs one evaluation. Rotation planes are Polak-Ribiere
conjugate. Lanczos stops when the Ritz residual ``beta_j |s_j|`` meets
``rotation_tol``. Both use the forward difference against the centre
gradient by default; ``validation/fd_curvature.py`` gives its error,
``dr |T| / 2 + 2 eps / dr``, and the ``dr`` that minimizes it for a
given gradient noise. ``FiniteDifference::Central`` trades one more
gradient per action for an ``O(dr^2)`` error.

The translation steps on the inverted force, whose value (the true
energy) is not its potential, as the band's pseudo-energy is not the
potential of the projected force. Both sessions accept only methods
that step on the force alone (FIRE, L-BFGS under ``Accept::Step``,
Barzilai-Borwein) and refuse line-searched ones. Between steps the
solver's cached force, inverted along the previous mode, is dropped
(rgmin ``forget_evaluation``) and the centre answers the re-read
without a surface call; a turn past ``MODE_RESET_ANGLE`` (5 degrees,
``validation/mode_reset.py``) also drops the optimizer memory.
