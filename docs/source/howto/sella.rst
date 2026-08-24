

Drive a Sella minimum or saddle session
---------------------------------------

The Sella QuasiNewton stepper is ``rgsaddle.QuasiNewton``
(``rgmin.qn_get_s``). ``get_stepper("qn")`` is the factory;
``step_on`` / ``retract_qn`` is project then retract.
``transport_step`` is the vector transport. This is a
stepper, not a session.

``SellaMinSession`` is order 0, ``eig=false``: project on
``SellaGeom`` (RigidQuotient at N>=3, or a ``Constraints``
chart via ``with_chart``), ``qn_restricted``, retract,
transport, ``CartesianPes.kick``, then Sella's ``delta0`` /
``sigma`` / ``rho`` trust schedule. ``SellaSaddleSession`` is
order 1: the same geometry and the saddle trust numbers,
then ``prfo_restricted``. Both sit on ``CartesianPes``, not
on ``IrcSession``. ``n_free`` is ``3N-6`` on the quotient
and ``3N - ncons`` on a chart.

``exact_eigh`` / ``rayleigh_ritz`` are Sella
``eigensolvers.py``. ``numerical_hvp`` is
``linalg.NumericalHessian``. ``SamdSession`` is the BDP
thermostat (``samd.py``). ``force_match_hessian`` is the
bond arm of ``force_match.pyx``.

The C wire exposes ``rgsaddle_sella_min_*`` and
``rgsaddle_sella_saddle_*`` (ABI minor 2). Unknown
``force_gate`` refuses create.

``RationalFunctionOptimization`` is Sella ``method=rfo``:
``alpha`` in ``[0, 1]``, ``order`` selects the
Banerjee-augmented mode. The increment is
``rgmin.rfo_get_s``. Distinct from ``rgmin.NewtonKind.Rfo``.
A host that needs a point on a set calls ``step_on``
(``project`` / ``retract`` / ``transport``).

Equality internals are ``Constraints`` on the same manifold:
fix translations (COM), rotations (Kabsch quaternion),
displacements, and host-supplied bonds / angles / dihedrals.
``project`` / ``retract`` / ``transport`` keep the point on the
level set. Bond / angle / dihedral topology stays in vocn.

``MaxInternalStep`` is the Sella per-coordinate clip
(``max |s_i w_i| <= delta``) on an internals packing. It sits
in front of the Euclidean trust radius. Distinct from
``rgmin.ras_clip`` (per-atom Cartesian) and from
``qn_restricted`` (``||s||``).

IRC stays ``IrcSession`` (``IrcKind.Gs2`` or ``Morokuma``).
Minimum-mode following stays ``MinModeSession`` (dimer / Lanczos).

Hessian
~~~~~~~

``CartesianPes`` defaults to MW BFGS. ``HessUpdate.TsBfgs`` is
Sella TS-BFGS (``|B|`` in the secant weight) so a negative mode
is not forced positive.

What not to do
~~~~~~~~~~~~~~

- Do not use ``SellaSaddleSession`` as a substitute for
  ``MinModeSession`` or ``IrcSession``.

- Do not treat ``ManifoldKind.Sphere`` as the GS2 sphere.
