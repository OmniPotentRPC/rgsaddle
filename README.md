# rgsaddle

Band and minimum-mode saddle mechanics over
[rgmin](https://github.com/OmniPotentRPC/rgmin) steppers.

Docs: <https://omnipotentrpc.github.io/rgsaddle/>


The band and the minimum-mode search take one `rgmin` `Solver` step
on a force this crate assembled. `BandSession::step` assembles the
NEB force and reports. No run-to-completion contract exists; hosts
own the loop and interleave their policy between steps. `run` is a
convenience loop over `step`.

`Index1Session` is the index-1 search on a dense Hessian. The
displacement is Baker's restricted-step partitioned RFO: maximize
the lowest mode, minimize the rest, and solve the scaling so the
Cartesian step lies inside the trust sphere. Powell and Bofill
update the Hessian. `nichols_step` remains the i-PI level shift.
`rgsaddle_prfo_step` is the C entry for the partition.

Force assembly ports eOn's NEB mechanics with identical branch
structure: tangents (Mills-Jonsson-Schenter simple, Henkelman-Jonsson
improved), springs (uniform, energy-weighted, Onsager-Machlup),
projections (plain elastic band, NEB, doubly nudged), and the
climbing-image force with the eOn trigger rule.

Positions are unwrapped Cartesian. A `Cell` (rows are lattice
vectors, triclinic allowed) supplies minimum-image differences for the
band mechanics; neighbor lists, when a surface wants them, are
[vesin](https://github.com/Luthaf/vesin)'s job, not this crate's.

A warm band step costs one evaluation of the interior images. The
session caches the endpoint energies and its last evaluation, so a
host that hands the band back unchanged (`set_positions`, wrapped
into its cell or not) or restarts the optimizer (`restart`) pays no
extra evaluation, and `evaluation` returns the energies, gradients,
and projected forces at the current band so the host need not
recompute them. `BandSession::reset` is the model-update boundary: it
drops the optimizer history and every cached surface value.
L-BFGS steps the band under rgmin's `Accept::Step` (one evaluation,
no energy test on the non-conservative band force); FIRE is the
default.

`MinModeSession` is long-lived: `set_position` moves it (optionally
with the host's gradient), `estimate_mode` refreshes the lowest mode
without translating. The dimer rotates by the modified Newton step
with an extrapolated gradient (one evaluation per rotation); Lanczos
stops on the Ritz residual; `FiniteDifference::Central` is available
for the actions, and `rotation_angle_tol` stops a dimer rotation whose
optimal angle falls under it. Both sessions refuse line-searched methods: the
oracle value beside a projected or inverted force is not its
potential. `validation/` holds the sympy and sollya
checks of the rotation formulas, the Lanczos residual bound, and the
finite-difference curvature error terms
(`uv run --with sympy python validation/<script>.py`,
`sollya validation/fd_curvature.sollya`).
