

Changelog
---------

Unreleased
~~~~~~~~~~

- ``SellaSaddleSession`` TrustRegion is ``prfo_trust_region`` (Sella
  Optimizer ``rs=tr``: one P-RFO alpha search on ``||s||``). dest
  ``prfo_restricted`` stays the stepper clip. Golden-master
  ``tests/sella_gold.rs`` compares RFO / QN / P-RFO / TS-BFGS / RAS
  and SellaMin / SellaSaddle one-step to dest Sella gold JSON.

- ``numerical_hvp`` is Sella ``linalg.NumericalHessian``. Reductions
  go through ``rgmin::vecops`` so ``par`` applies. A host that needs
  the displaced point on a set calls ``numerical_hvp_on``
  (``project`` / ``retract`` / ``transport``). Dense eigen stays in
  ``eigensolve``. Approximate Hessian stays in ``qn``. Sparse
  internals stay in vocn.

- ``CartesianPes::with_proj`` hangs Sella ``proj_trans`` / ``proj_rot``
  (``fix_translation`` / ``fix_rotation``). ``project`` / ``retract`` /
  ``transport`` stay on that set. ``new`` is the unconstrained
  Hessian store ``InternalPes`` wraps. C ABI ``rgsaddle_pes_*``,
  ABI minor 12.

- ``exact_eigh`` / ``rayleigh_ritz`` / ``expand`` / ``rayleigh_ritz_iter``
  are Sella ``eigensolvers.py``. ``eig=true`` walks a Ritz subspace
  (``jd0`` default). Reductions go through ``rgmin::vecops`` so
  ``par`` applies. A Ritz vector lives on the sphere: ``project`` /
  ``retract`` / ``transport`` keep it on the set. ``EigenDevice::Dlpk``
  is the rgmin ``lowest_mode`` waist, not a second GPU stack.

- ``SamdSession`` is Sella ``samd.py`` BDP: Euclidean ``step`` on
  the C wire (``rgsaddle_samd_*``), and ``step_on`` /
  ``retract_samd`` / ``transport_velocity`` so a retracted
  Verlet increment stays on the set.

- Force gate is the eOn / gpr\ :sub:`optim`\ closed enum
  (``L2`` / ``Linf`` / ``MaxForceOnAtom``), not a single implied
  scalar. Sessions take ``ForceGate``; the C ABI is
  ``rgsaddle_force_gate_t``. Maturin / PyO3 emit a real Python
  enum (``ForceGate.L2_NORM``, ``LINF_NORM``, ``MAX_FORCE_ON_ATOM``).

- Sella ``Translation`` / ``Rotation`` / ``Displacement`` internals
  implement ``rgmin::Manifold`` (``project`` / ``retract`` /
  ``transport`` stay on each coordinate's level set) and feed
  the ``Constraints`` equality residual on the same chart.
  Bond / angle / dihedral topology stays in vocn.
  ``InternalPes::from_find`` loads a vocn auto-find primitive
  set; dest does not generate Bond / Angle / Dihedral
  topology. C ABI ``rgsaddle_internal_pes_create_from_find``,
  ABI minor 11.

- ``CellInternalPes`` is Sella ``CellInternalPES``: packed
  ``[q_int; cell_params]``, log-cell ``CellChart::LogDeform``,
  ``kick_packed`` evaluates in the living cell. Mask-false
  cell DOF drop out of the packed chart. Internals stay
  on the ``Constraints`` level set.

- ``TrustRegion``: Sella ``cons(s) = ||s||`` over
  ``rgmin::qn_restricted``. Distinct from ``IRCTrustRegion``
  (``||(s+d1) odot sqrt(m)||``). ``step_on`` / ``transport_step``
  keep the clipped increment on the set.

- ``RestrictedAtomicStep``: Sella ``cons(s) = max_i ||s_i||``
  over ``rgmin::ras_clip`` (3N Cartesian). Internals refuse
  it. Cartesian / cell sessions take unrestricted QN or
  P-RFO, then ``ras_clip``. ``step_on`` / ``transport_step``
  keep the clipped increment on the set.

- ``MaxInternalStep``: per-coordinate clip ``max |s_i w_i|``
  on an internals packing, applied before the Euclidean
  trust radius.

- ``SellaMinSession`` is Sella order-0: QN + TrustRegion,
  ``delta0~/~sigma~/~rho`` schedule, ``eig=false``. Isolated
  molecules retract on ``ManifoldKind::RigidQuotient``; the
  increment stays horizontal. ``CartesianPes::kick`` stores
  Cartesian BFGS pairs; the force gate is per-atom ``||F||_2``.

- Sella ``QuasiNewton`` stepper: ``get_stepper("qn")``,
  ``get_s`` / ``restricted`` over ``rgmin::qn_get_s``.
  ``step_on`` / ``retract_qn`` is project then retract;
  ``transport_step`` is the vector transport.

- Sella ``RationalFunctionOptimization`` stepper: ``alpha`` /
  ``order`` over ``rgmin::rfo_get_s``. Distinct from
  ``NewtonKind::Rfo``. Riemannian ``step_on`` retracts onto
  ``ManifoldKind``.

- Sella ``PartitionedRationalFunctionOptimization`` stepper:
  RFO in the ``order`` uphill modes plus RFO in the downhill
  complement, over ``rgmin::rfo_get_s`` / ``prfo_restricted``.
  ``step_on`` / ``transport_step`` keep the increment on the set.

- ``IrcSession`` forward / reverse roll-down over rgmin ``IrcTrust``.

- ``IrcSession::step`` matches Sella ``IRC.step``: optional kick,
  inner GS2 on ``||(s+d1) odot sqrt(m)|| = dx``, then ``d1 = 0``.
  Not ``ManifoldKind::Sphere``. ``MwRigid`` only when ``N >= 3``.

- Matrix-free lowest-mode kick (``from_surface``) through the
  rgmin ``EigensolverKind`` waist.

- Diataxis book: one tutorial (MEP roll-down), howtos, MEP
  explanation, Shibuya ecosystem nav.
