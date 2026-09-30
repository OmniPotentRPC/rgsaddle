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
differences for the band mechanics on an orthorhombic box. Neighbor
lists, when a surface wants them, are vesin's job.

FIRE as the default
-------------------

The projected NEB force is not a gradient of a potential. rgmin's
session L-BFGS still applies an energy-decrease acceptance on that
path, so every NEB step is refused. FIRE steps unconditionally, which
is what eOn's velocity NEB stepper does. Hosts that want L-BFGS set
``BandConfig.method`` and own the acceptance gap.
