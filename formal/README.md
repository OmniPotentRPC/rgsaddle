# Index-one runtime contracts

`Index1Session::step` uses one decomposition of the mass-weighted Hessian
for both its displacement and its curvature report. The report refers to
the Hessian that produced the step, before Powell or Bofill updates it.
`cartesian_of` reads the eigenvectors and masses through borrowed views.

## Proofs and their assumptions

`RgsaddleRuntimeContracts/Index1Reuse.lean` proves five contracts:

- Ordered dot products agree when every indexed input agrees.
- Cartesian components agree under identical multiply, add, square-root,
  and divide operations. The proof assumes no associativity of arithmetic.
- Reusing a deterministic decomposition preserves success and error results
  while its Hessian and masses remain unchanged.
- An exact weighted eigenpair gives a generalized Cartesian eigenpair.
- Dividing by square-root masses preserves the corresponding weighted norm.

The first three contracts model the operations in `src/prfo_restricted.rs`
and the reuse in `src/nichols.rs`. Their correspondence with the Rust source
requires valid array dimensions, equal copied and borrowed entries, and a
fixed arithmetic environment, including across surface callbacks. The
proofs are mathematical models; they do not verify Rust or compiler output.

The exact eigenpair theorem assumes an eigenpair. The Jacobi eigensolver
uses a stopping tolerance, so its computed eigenvalues need not equal the
exact spectrum. For example, a two-dimensional matrix with zero diagonal
and off-diagonal entries `2^-54` meets its stopping tolerance immediately.
Reusing that computation preserves its result and its approximation error.

`../validation/index1_reuse.py` checks mass weighting and the step norm with
SymPy. Its exact trust-step fixture has weighted curvature -2 and an updated
curvature of -4; `../tests/index1_contract.rs` exercises the same case for
Powell and Bofill. The Cartesian trust norm differs from the mass-weighted
norm when masses differ.

`../validation/index1_reuse.sollya` encloses the absolute rounding-error
coefficient for 3, 75, 300, and 4096 modes. It assumes finite, normal
intermediates and correctly rounded operations. For unit roundoff `u`,
`eta = 2*n*u/(1 - 2*n*u)`, and `A_i = sum_k |Q_ik*z_k|`, the bound is

```text
|computed_step_i - exact_step_i|
  <= A_i / sqrt(m_i) * ((1 + eta)*(1 + u)/(1 - u) - 1).
```

This bound allows cancellation. Overflow and underflow lie outside its
relative-error model. The certificates bound Cartesian reconstruction;
they do not establish eigensolver accuracy or global convergence.

## Run

The Lean toolchain and Mathlib revision are pinned in this directory.
With Lean, Python, SymPy, and Sollya installed, run from the repository root:

```sh
./formal/check.sh
python validation/index1_reuse.py
cargo build --locked --release --features capi
cargo test --locked --release --features capi
```

The audit rejects extra axioms and missing proofs. Successful output reports
five audited theorems, two SymPy identity groups, four Cartesian error
certificates, and the exact trust fixture. Python optimization is refused.

`validation/index1_probe.c` uses only `include/rgsaddle.h`. Compile it against
each runtime library and give the two executables to the comparison driver:

```sh
cc -O2 -Iinclude validation/index1_probe.c /path/to/baseline.so -lm -o probe-baseline
cc -O2 -Iinclude validation/index1_probe.c /path/to/candidate.so -lm -o probe-candidate
python validation/check_index1.py ./probe-baseline ./probe-candidate ./comparison
```

The driver compares complete binary traces for 128 cases and 768 steps,
including reports, positions, Hessians, and callback counts. Its paired
timings cover dimensions 3, 15, 75, 150, and 300 and retain every sample.
They measure the index-one runtime on synthetic surfaces; molecular campaign
performance needs separate measurements.
