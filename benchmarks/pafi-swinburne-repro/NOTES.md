# PAFI Swinburne reproduction

## Verdict

**Partially reproduced.** The full 100-trajectory Swinburne run is **not
yet reproduced**: it has not completed, and no number from it is reported.
The fast test exactly replays the public PAFI repository's pinned Fe-SIA
zero-temperature gradient fixture and its cubic postprocessing. The ignored
test is a statistical regression against the three numbers printed in the
public repository's W-vacancy example and sends the EAM evaluations through
`rgpot` and `lammpc`. A four-trajectory moderate smoke passes all three
rows: 0 K, 1000 K and 2000 K, and that smoke is the only finite-temperature
evidence here. The equilibration gate is not loosened to make the full run
pass. A full run with 100 trajectories, 2000 thermalization steps and 5000
production steps was started once, on three threads, and stopped after 1021
seconds in the 1000 K profile with `NotEquilibrated`: the split-half drift
of a trajectory was 0.2606 against a standard error of 0.0455, and the gate
is 4 standard errors. The ensemble returns the first trajectory error, so
one trajectory that fails the gate ends the profile call. The cause of that
drift was not investigated, and the run was not repeated with other
settings.

The two applications in the PRL itself were **not reproduced**. The article
does not publish their atomic paths, cells, or numerical free-energy tables,
and no supplementary-information file is linked by the PRL or HAL records or
present in the public PAFI repository. Figure ordinates alone are not a
regression oracle. The W application also contains about 120,000 atoms and is
outside the requested moderate scale.

## Primary-source record

The method source is T. D. Swinburne and M.-C. Marinica, “Unsupervised
Calculation of Free Energy Barriers in Large Crystalline Systems,” *Physical
Review Letters* **120**, 135503 (2018),
<https://doi.org/10.1103/PhysRevLett.120.135503>. Page numbers below are the
printed article pages. The implementation source is
`tomswinburne/pafi` at commit
`1661a1d61b2ccbbd33ec4c9656d60947b8b74cc1`.

Exact numerical statements available in the article are:

- p. 135503-3: “typically 100–1000 independent trajectories,” “initially 10
  values of ξ,” “1000–4000 integration steps” for thermalization,
  “2000–5000 steps” for production, and an ensemble size “100–500” for
  acceptable error.
- Fig. 1, p. 135503-4: the displayed Fe profiles are labelled 0, 100, 300,
  500, 700, and 900 K. The text says 1000 independent simulations at 10
  temperatures up to 900 K, agreement with the harmonic approximation below
  200 K, and reversal of the harmonic stability prediction above 450 K.
  The caption says the ensemble error is not visible for ensemble size above
  300.
- Fig. 2 and pp. 135503-4–5: the W double-kink NEB formation energy is
  exactly 1.67 eV for a system of about 120,000 atoms. The displayed profiles
  are labelled 0, 200, 400, 600, 800, and 1000 K. The text reports a velocity
  boost “over ×10³” at 800 K and anharmonic corrections increasing from
  around 200 K.

The public repository, not the PRL, supplies a runnable 249-atom W-vacancy
example. `doc/test_output.png` prints, exactly:

| source case | temperature (K) | source ΔF (eV) | this work ΔF (eV) | status |
|---|---:|---:|---:|---|
| public W vacancy | 0 | 1.97 ± 0.00 | 1.96817368 ± 0.00000000 | pass in moderate smoke |
| public W vacancy | 1000 | 2.07 ± 0.04 | 2.04436424 ± 0.01559574 | pass in moderate smoke |
| public W vacancy | 2000 | 1.82 ± 0.09 | 1.80367161 ± 0.03325187 | pass in moderate smoke |
| PRL W double kink, Fig. 2 | 0 | 1.67 (NEB) | not run | different, unavailable 120,000-atom case |
| PRL Fe C15, Fig. 1 | 0–900 | no numerical table | not run | path and configuration unavailable |

The public `tests/test_pafi.py` pins the nine Fe-SIA gradients
`[0, 0.614145, 0.846192, 0.592260, 0.089912, -0.426347, -0.853405,
-0.792713, 0]`. Default CI replays those exact values and obtains the public
cubic-remesh barrier `0.27622876229655097 eV` and end difference
`0.003824869324197502 eV`. This checks the path interpolation and integration;
it deliberately does not claim an independent EAM calculation.

## Protocol and implementation

| item | PRL protocol | this reproduction |
|---|---|---|
| path | NEB, then a spline for every atomic coordinate (p. 135503-2) | rgsaddle climbing-image NEB, initialized by the nine public W frames, then a not-a-knot cubic spline |
| hyperplanes | initially 10 equally spaced ξ values; add planes where the mean force is large | 9 equally spaced planes, the public example's default; this is a known difference from the unavailable PRL applications |
| thermostat | projected overdamped Brownian dynamics, Eq. (9) | `PafiIntegrator::EulerMaruyama`, projected Gaussian Brownian increments |
| thermalization | 1000–4000 steps | 2000 steps |
| production | 2000–5000 steps | 5000 steps |
| ensemble | typically 100–1000 trajectories; 100–500 acceptable | 100 independent trajectories |
| temperatures | application-dependent sets shown above | 0, 1000, and 2000 K, because those are the public W-example oracle |
| 0 K reference | constrained minimum in every hyperplane | the planes of the remeshed path, relaxed in the plane as described below |
| potential path | EAM | `rgpot` profile session backed by the canonical `lammpc` LAMMPS embed |

The measured rows above are a smoke with four trajectories, 2000
thermalization steps, and 2000 production steps. Rgsaddle NEB converged in
86 steps at a maximum force of 0.0008863843282684947 eV/Å, and no basin
stretch was rejected. The release test, including NEB and both
finite-temperature rows, took 144 seconds. The same smoke with 1000
thermalization steps stopped at 1000 K with `NotEquilibrated` (drift
0.8129 against a standard error of 0.0981), so 1000 steps are too few for
this path and the table uses 2000. The smoke uses fewer trajectories than
the article's stated acceptable range and is not substituted for the full
campaign.

### The 0 K row

Reading the 0 K barrier from the unrelaxed planes of an NEB path that
used no climbing image gives 1.92446638 eV against the published 1.97 eV.
That reference has two defects. Without a climbing image the highest image
is not at the saddle. Without relaxation the planes are not the ones the
method samples: in Eq. (9) of the PRL the dynamics is projected onto the
hyperplane through the normal, so as the temperature goes to zero it is a
steepest descent in the plane, and the force is read at that minimum.

The test now climbs the highest image, requires that the band reports a
climbing image, and evaluates the 0 K profile on the planes of the remeshed
spline. Each plane is relaxed with the steepest-descent flow on the force
without its center-of-mass part and without its component along the path
normal. The gradient at the relaxed point is the force along the normal
times `1 - psi`, with `psi` from the spline curvature, and the profile is
the integral of that gradient over the arc. The measured values are:

| quantity | value (eV) |
|---|---:|
| climbing-image energy above the first image | 1.842811 |
| remeshed path, planes unrelaxed | 1.932339 |
| remeshed path, planes relaxed in the plane | 1.968174 |
| published | 1.97 |

The relaxed value differs from the published 1.97 eV by 0.001826 eV, which
is inside the half display unit of 0.005 eV. The relaxation lowered the
energy of the seven interior planes by between 0.00021 and 0.0221 eV, and no
atom moved more than 0.0344 Å. The climbing-image energy is lower than the
0 K profile because the profile integrates the force on planes of one fixed
family, which is a constrained quantity, while the climbing image is a
saddle estimate of the unconstrained surface. The two are not expected to
be equal and the test does not compare them.

The public W example is not itself the paper protocol:
`examples/input_python.py` requests underdamped Langevin
(`OverDamped = 0`), one repeat, and 2000 thermalization plus 2000 sampling
steps. Its image title says “Short (10 ps),” but does not record the worker
count or all parameters that produced the error bars. The ignored regression
uses the paper-range ensemble and lengths, so agreement tests the published
equilibrium values rather than bitwise trajectory identity.

Two estimator variants are explicit:

- `PafiMeanForce::ReferenceCode` is the default and records
  `(∇E·X′) ψ`, matching LAMMPS `fix pafi` and the public driver.
- `PafiMeanForce::PublishedEquation8` adds the analytic
  `kT ∂ξ ln(|ψ|/|X′|)` term in PRL Eq. (8), using the spline's analytic
  one-sided third derivative, and rejects violation of Eq. (6).

The public W numbers are tested with `ReferenceCode`, because that is the
implementation that produced them.

The pinned lammpc embed exposed three generic owning-library defects: C helper
implementations were linked into a C++ translation unit without `extern "C"`
declarations; its capability schema identifier omitted the canonical `0x`
prefix required by rgpot; and its direct LAMMPS setup did not allocate
per-type atom arrays before assigning masses. The lammpc fixes
are carried by the `lammpc` revision that the full-run script pins and
fix those ABI and setup defects. The moderate EAM smoke exercises them
through the real embed. No potential code is copied into rgsaddle.

## Test-only EAM reader

`tests/common/eam_fs.rs` is a test-only reader of the LAMMPS `eam/fs`
tables and still exists, as does `tests/pafi_reference.rs`. Both serve only
the rerun of the public reference examples, which
`benchmarks/pafi_reference.sh` drives and which compares rgsaddle with the
reference code on the same potential. The Swinburne reproduction in this
directory does not use them. It evaluates EAM only through
`rgpot` and `lammpc`, so its numbers do not depend on that reader.

## Statistical acceptance

For each public W row, the ignored test requires

`|ΔF_ours - ΔF_source| ≤ 2 sqrt(σ_ours² + σ_source² + (0.005 eV)²)`.

The last term is half of the source image's 0.01 eV display unit. The two
standard errors give a predeclared approximately 95% sampling band; this is
the derived regression tolerance, not a hand-tuned absolute threshold.

The BenchBot question is in `QUESTION.md`. A full run must add its generated
CSV values separately from the smoke values above and retain any failure
rather than widening the acceptance band.
