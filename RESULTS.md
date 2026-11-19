# Solid-state saddle results

## Scope and base

This series extends the solid-state saddle implementation already present on
`develop`; it does not replace `SolidDimer` or `SolidNeb`. The exact base is
`aa6dd9371d6a63103010a6829cb62c74a39c7ff1`.

`AUDIT.md` records the implementation and repository audit, including the four
named feature-branch tips supplied by the owner.

## Cu benchmark failures

The 109-atom embedded Mishin-Cu validation now passes all ten fixed- and
variable-cell rows in `benchmarks/ssneb_validate_cu.csv`, including the three
previous failures:

- The zero-shear fixed-cell band now ends in the basin connected to the
  first-order dimer saddle. Both signs of the unstable mode are escaped and
  quenched, and the band is seeded through the dimer saddle. This removes the
  unrelated index-2 stationary point reached from the former prescribed
  endpoint and linear path.
- The 3% fixed-cell force target is derived from the smallest certified stable
  dimer eigenvalue and the existing 0.01 Å geometry target. It therefore
  resolves the soft stable mode instead of loosening the force or geometry
  acceptance criteria.
- The 5% variable-cell band is first relaxed with FIRE to the geometric mean
  of its initial and final force targets, then climbed with L-BFGS. In the
  library, the L-BFGS history is cleared whenever the climbing image changes
  or is armed. This prevents curvature history from a different objective
  driving the free cell away.

Variable-cell potential energies are compared after mapping the climbing image
to the dimer cell; enthalpies remain compared in each state’s own cell. The CSV
retains both values explicitly.

The benchmark uses its existing embedded Mishin EAM implementation. It does not
claim a separate LAMMPS/`rgpot` reproduction.

## Strain-coupled Vineyard prefactor

`solid_prefactor` supplies a finite-difference Hessian in the joint coordinate
`(r, J eps)`, where `J = V^(1/3) N^(1/6)`, for the active atomic and strain
degrees of freedom. The gradient combines atomic forces and the
external-stress-coupled strain force. `joint_vineyard_prefactor` requires:

- caller-provided coordinate, stress, stationarity, eigenvalue, and mode-count
  tolerances;
- a minimum with no unstable modes;
- a saddle with exactly one unstable mode;
- the same explicit atomic mass convention for atomic and scaled strain
  coordinates.

No missing mode contributes a silent zero. Non-finite inputs and spectra are
errors. The API documentation states the generalized-mass convention so callers
can choose a physically appropriate mapping.

## Reproduction

The focused Cu benchmark is:

```text
cargo +1.88 test --release --test ssneb_validate_cu -- --ignored --nocapture
```

The complete verification commands and their outcomes are recorded in the
handoff `RESULTS.md` beside the generated mbox.
