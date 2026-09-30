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

Positions are unwrapped Cartesian. A `Cell` supplies orthorhombic
minimum-image differences for the band mechanics; neighbor lists,
when a surface wants them, are [vesin](https://github.com/Luthaf/vesin)'s
job, not this crate's.

`BandSession::reset` is the model-update boundary: quasi-Newton
history taken on one surface epoch must not survive onto the next.
