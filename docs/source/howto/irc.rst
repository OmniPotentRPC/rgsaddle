

Drive an IRC session
--------------------

Hold an ``IrcSession``. Call ``step`` once per outer IRC move. The
inner geometry is rgmin ``IrcTrust``:



.. math::

    \|(s + d_1)\odot\sqrt{m}\| = dx.

The increment is Sella ``QuasiNewtonIRC``
(``rgmin::qn_irc_restricted``) on a mass-weighted BFGS Hessian
seeded with the imaginary mode (``lambda = -1``). Interior Newton
is allowed only after that model is positive definite and the arc
is past eight radii. ``reset`` and ``set_direction`` drop the
pairs and re-seed.

Directions
~~~~~~~~~~

``IrcKind::Gs2`` (default) is Gonzalez--Schlegel / Sella.
``IrcKind::Morokuma`` is the Ishida--Morokuma--Komornicki
predictor-corrector used by ``gpr_optim`` ``IRCDriver`` (fixed
mass-weighted step ``h = dx``).

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
