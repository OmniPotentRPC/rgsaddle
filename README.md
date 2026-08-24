# rgsaddle

<p align="center">
  <img src="branding/logo/rgsaddle-logo-light.svg" width="420" alt="rgsaddle">
</p>

Band, min-mode, and IRC mechanics over
[rgmin](https://github.com/OmniPotentRPC/rgmin) steppers.

Two seams, stepping at both. The inner seam is rgmin's `Solver`: one
optimizer step over an assembled band force; this crate defines no
optimizer of its own. The outer seam is `BandSession::step`: assemble
NEB forces on the caller's surface, take one solver step, report. No
run-to-completion contract exists; hosts own the loop and interleave
their policy (trust regions, drift budgets, acquisition, hybrid MMF)
between steps. `run` is a convenience loop over `step` and nothing
more.

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

Narrative docs live in [`docs/orgmode/`](docs/orgmode/index.org)
(Diataxis). The one tutorial is MEP roll-down from a first-order
saddle. Export with `emacs --script docs/export.el` from `docs/`;
Sphinx (Shibuya) reads `docs/source/`. The mark is in
[`branding/logo/`](branding/logo/README.md).

MIT. Build and test on the remote builder, not a laptop.
