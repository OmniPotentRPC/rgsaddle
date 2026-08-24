

Changelog
---------

Unreleased
~~~~~~~~~~

- Sella ``RationalFunctionOptimization`` stepper: ``alpha`` /
  ``order`` over ``rgmin.rfo_get_s``. Distinct from
  ``NewtonKind.Rfo``. Riemannian ``step_on`` retracts onto
  ``ManifoldKind``.

- ``IrcSession`` forward / reverse roll-down over rgmin ``IrcTrust``.
- ``IrcSession::step`` matches Sella ``IRC.step``: optional kick,
  inner GS2 on ``||(s+d1) odot sqrt(m)|| = dx``, then ``d1 = 0``.
  Not ``ManifoldKind::Sphere``. ``MwRigid`` only when ``N >= 3``.

- Matrix-free lowest-mode kick (``from_surface``) through the
  rgmin ``EigensolverKind`` waist.

- Diataxis book: one tutorial (MEP roll-down), howtos, MEP
  explanation, Shibuya ecosystem nav.
