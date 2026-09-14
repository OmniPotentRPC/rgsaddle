Explanation
===========

Two seams
---------

rgsaddle does not implement an optimizer. rgmin does. This crate
assembles a force (NEB band or inverted lowest mode) and asks
``rgmin::Solver`` for one step. Hosts interleave trust regions, drift
budgets, acquisition, or a hybrid MMF between steps.

That is why ``run`` is documented as a convenience loop over ``step``
and nothing more.

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
