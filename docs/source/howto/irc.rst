

Drive an IRC session
--------------------

Hold an ``IrcSession``. Call ``step`` once per outer IRC move. The
inner geometry is rgmin ``IrcTrust``:



.. math::

    \|(s + d_1)\odot\sqrt{m}\| = dx.

Directions
~~~~~~~~~~

``IrcDirection::Forward`` and ``Reverse`` are the sign of the kick.
``set_direction`` restores the saddle, rebuilds ``d1``, and forgets
solver history.

Kick
~~~~

Prefer ``IrcSession::from_surface`` so the mode comes from
matrix-free Lanczos. Pass a precomputed mode to ``new`` only when
the host already has it.

What not to do
~~~~~~~~~~~~~~

- Do not treat ``ManifoldKind::Sphere`` as the GS2 sphere.

- Do not full-diagonalize a 3N Hessian just to kick. ELPA and
  SLATE ``heev`` belong behind a Hessian that already exists and
  ``n >= 512``, with ``nev`` small.
