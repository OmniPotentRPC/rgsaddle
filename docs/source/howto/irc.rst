

Drive an IRC session
--------------------

Hold an ``IrcSession``. Call ``step`` once per outer IRC move. The
inner geometry is rgmin ``IrcTrust``:



.. math::

    \|(s + d_1)\odot\sqrt{m}\| = dx.

The increment is Sella ``QuasiNewtonIRC``
(``rgmin::qn_irc_restricted``) on a mass-weighted BFGS Hessian that
starts as the identity. ``reset`` and ``set_direction`` drop that
curvature.

Directions
~~~~~~~~~~

``IrcDirection::Forward`` and ``Reverse`` are the sign of the kick.
``set_direction`` restores the saddle, rebuilds ``d1``, and forgets
solver history.

Kick
~~~~

Prefer ``IrcSession::from_surface`` so the mode comes from
the rgmin lowest-mode waist (default Lanczos; closed
``EigensolverKind``). Pass a precomputed mode to ``new`` only when
the host already has it.

What not to do
~~~~~~~~~~~~~~

- Do not treat ``ManifoldKind::Sphere`` as the GS2 sphere.

- Do not full-diagonalize a 3N Hessian just to kick. ELPA and
  SLATE ``heev`` belong behind a Hessian that already exists and
  ``n >= 512``, with ``nev`` small.
