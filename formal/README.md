# Formal contracts in Lean 4 and Mathlib

Machine-checked statements of what the band projections, the
minimum-mode step and the finite-difference curvature guarantee. Each
theorem names the Rust function it constrains and the precondition the
Rust code must establish. The proofs hold over the reals (exact
arithmetic); the Rust tests hold the floating-point implementation to
them at fixed tolerances.

## Check

```sh
cd formal
./check.sh                # lake exe cache get, lake build, sorry/axiom audit
SKIP_CACHE=1 ./check.sh   # when the Mathlib oleans are already present
```

`check.sh` fails when the build fails, when a source contains `sorry`,
`admit` or an `axiom` declaration, when the build log reports a sorry,
or when `#print axioms` of any theorem names an axiom beyond
`propext`, `Classical.choice` and `Quot.sound`. The toolchain is pinned
in `lean-toolchain` (`leanprover/lean4:v4.34.0-rc2`) and Mathlib in
`lakefile.toml` and `lake-manifest.json`
(`f2916a54665af851fc9a4da901cfc242c47a8922`). Install the toolchain
with [elan](https://github.com/leanprover/elan).

## Theorem map

| Theorem (`RgsaddleContracts.*`) | Source | What the Rust code must guarantee |
|---|---|---|
| `perp_add_parallel` | `src/projection.rs::force_perp` | None. Any force splits into perpendicular and parallel parts. |
| `perp_orth`, `perp_idem` | `src/projection.rs::force_perp` | A unit tangent. `F - (F.t) t` is then orthogonal to `t`, and projecting twice changes nothing. |
| `perp_parallel` | `src/spring.rs::SpringKind::compute` | A unit tangent. Every spring model returns `parallel = c t`, which has no perpendicular part. |
| `neb_force_parts` | `src/projection.rs::ProjectionKind::project` (`Neb`) | A unit tangent. The NEB force's perpendicular part is the true force's, and its parallel part is the spring's alone. |
| `reflect_parallel`, `reflect_perp`, `reflect_reflect`, `norm_reflect` | `src/projection.rs::climbing_image_force` | A unit tangent. `F - 2 (F.t) t` reverses the parallel part, keeps the perpendicular part, is an involution, and keeps the norm. |
| `climb_dneb_parallel` | `src/projection.rs::climbing_image_force` | A DNEB term orthogonal to `t`, which `dneb_orth_tangent` supplies. The climbing image still reverses the parallel part. |
| `dneb_orth_tangent`, `dneb_orth_force`, `dneb_force_parallel` | `src/projection.rs::dneb_component` | A unit tangent, and a nonzero perpendicular force (the code requires a norm above `1e-10`). The DNEB term is then orthogonal to `t` and to `F_perp`, and leaves the parallel part `c`. |
| `perp_not_orth_of_short` | `src/tangent.rs::normalize_tangent` | Shows the unit hypothesis matters. A tangent of norm `1/2` leaves a parallel residue of `3/8`. `normalize_tangent` returns a unit vector or an error. |
| `normalized_mode_unit`, `normalized_mode_reflect_norm` | `src/minmode.rs::checked_mode`, called by `new` and `set_mode` | A nonzero mode. Division by its norm yields a unit direction whose reflected force preserves the norm. |
| `quad_eq_sum`, `rayleigh_bounds`, `rayleigh_between_extremes`, `rayleigh_at_eigenvector` | `src/minmode.rs::rotate_dimer` and `lanczos_mode` (the curvature `h.n` of a unit mode) | A unit mode and a symmetric Hessian. The reported curvature then lies between the extreme eigenvalues and reaches each one at its eigenvector. |
| `rotational_force_zero_iff` | `src/minmode.rs::rotate_dimer` | None. The rotational force vanishes exactly at an eigenvector. |
| `minmode_force_parts`, `reflect_neg` | `src/minmode.rs::MinModeSession::step` | A unit mode (`normalize` keeps it unit). The inverted gradient flips the mode component and keeps the rest. |
| `quad_key`, `minmode_descends_to_saddle` | `src/minmode.rs::MinModeSession::step` | A quadratic model with an exact unit eigenvector of negative eigenvalue and positive curvature orthogonal to it. The inverted force then points at the saddle from every other point. |
| `switched_force_neg`, `switched_force_nonneg`, `switched_force_parallel`, `switched_force_perp_nonneg`, `switched_force_perp_neg` | `src/minmode.rs::translated_gradient` | A unit mode. Both branches reverse the mode component; at a nonnegative curvature nothing across the mode remains, below zero the force across it is kept. |
| `switched_force_energy_rate` | `src/minmode.rs::translated_gradient` (nonnegative branch) | None. The true energy rises along the switched force at the rate `(F.v)^2`. |
| `switched_force_leaves_minimum`, `switched_descends_to_saddle` | `src/minmode.rs::MinModeSession::step` | The quadratic model with an exact unit eigenvector. From a minimum basin the switched force is `lam a v` and points away from the minimum; from the saddle region it is the inverted force and points at the saddle. |
| `reflect_comp_symm`, `minmode_energy_mismatch` | `src/minmode.rs::MinModeSession::step` | Same model (symmetry needs only the eigenvector). The inverted operator is symmetric. Its form differs from the true energy by `2 a^2 |lam|`, `a = d.v`. |
| `fence` | (lemma) | Comparison principle on `[0, h]`. |
| `forward_grad_diff_error`, `forward_grad_diff_exact_cubic` | `src/minmode.rs::Counted::action` (`FiniteDifference::Forward`) | A bound on the third derivative along the mode. The forward-difference curvature error is then at most `M3 dr / 2`, and this order is attained. |
| `central_grad_diff_error` | `src/minmode.rs::Counted::action` (`FiniteDifference::Central`) | A bound on the fourth derivative. The error is then at most `M4 dr^2 / 6`, for one more gradient per action. |
| `central_second_diff_error`, `central_second_diff_error_contDiff` | (energy-only curvature) | A bound on the fourth derivative. The error of `(E(x+h) - 2E(x) + E(x-h)) / h^2` is then at most `M4 h^2 / 12`. |
| `climbingImageForce_eq_reflect` | `src/projection.rs::climbing_image_force` | A zero DNEB vector. The force is `reflect` of the true force. |
| `translatedForce_eq_reflect`, `translated_neg_curvature`, `dimer_translation_eq_climbing` | `src/minmode.rs::translated_gradient` | Negative curvature and a unit mode. Minus the translated gradient of `g`, with force `F = -g`, is `reflect` of `F`. |
| `reflect_eq_zero_iff` | `src/projection.rs::climbing_image_force` | A unit tangent. The reflected force is zero exactly when the force is zero. |
| `ascend_eq_reflect_iff` | `src/minmode.rs::translated_gradient` (non-negative curvature) | A unit mode. The ascend branch `-(F · v) v` equals `reflect` exactly when the component of `F` orthogonal to `v` is zero. |
| `improvedTangent_dimer` | `src/tangent.rs::compute_tangent` (`TangentKind::Improved`) | `R± = R0 ± δ N`, `|N| = 1`, `δ > 0`, and equal side energies strictly below the centre. The improved tangent is `N`. The Lean normalisation uses a nonzero test. Rust uses the floor `1e-10` (`scripts/constants.json`). |
| `dimer_forces_eq` | `climbing_image_force` and `translated_gradient` | The hypotheses of `improvedTangent_dimer`, negative curvature, and the improved tangent already equal to `N`. The climbing-image force and the dimer translation force then agree. |
| `frozenAxis_two_dim` | `src/projection.rs::climbing_image_force` (the reflector `I - 2 v vᵀ` applied to `H = diag(-a, b)`) | `a > 0`, `b > 0`, and a unit `v = (c, s)`. The eigenvalues of the reflected Hessian have positive real part exactly when `|c| > 1/√2`. |

## Stencil notes

`lake exe mirror` writes `formal/golden/stencil.json` over the rationals.
`tests/formal_mirror.rs` compares `climbing_image_force` and
`compute_tangent` to those bits. `translated_gradient` is `pub(crate)`.
Its rows are compared from `src/kani_proofs.rs` under `#[cfg(test)]`.
There is no public shim.

The n-dimensional frozen-axis statement is open. The two-dimensional
theorem is the one above. A search for `n = 3..6`
(`validation/stencil/fortyfive.py`) did not produce a counterexample
with `|v · e_min| > 1/√2` and a negative real part below `1e-6`.
The best candidates sit just above the threshold, with smallest real
part about `8.7e-4`. That is not a proof.

Signed zero, from `validation/signed_zero_z3.py` (de Moura and Bjørner,
Z3, TACAS 2008, doi:10.1007/978-3-540-78800-3_24). Z3 proves that
`x + 0.0` preserves every binary64 bit pattern except `-0`, which
becomes `+0`. Adding a zero DNEB vector can therefore change a `-0`
component. The ascend branch scales the mode by `-parallel`, so a zero
component becomes `-0` while the rational golden writes `0/1`. On inputs that are both signed zeros, the two scalar
reflector formulas agree as bits or both return zero. Z3 did not decide
the formulas on every finite binary64 value, so the Rust forms stay as
written.

`SpringKind::Weighted` does not evaluate
`α = (E_max - E_i) / (E_max - E_ref)`. The caller must not divide when
`E_max = E_ref`. On `E_i ≤ E_max` that case is `k_i = k_min`.
The golden rows check this formula, not the spring method.

## Preconditions and how the Rust code meets them

* `normalize_tangent` falls back from the tangent to the forward and
  then the backward difference, and returns `SaddleError::Invalid`
  when the image coincides with both neighbours, so every projection
  receives a unit tangent (`perp_not_orth_of_short`).
* `MinModeSession::step` hands the solver the true energy beside the
  inverted gradient; they disagree by `2 a^2 |lam|` on the model
  (`minmode_energy_mismatch`). `MinModeSession::new` and
  `BandSession::new` therefore accept only force-driven methods (FIRE,
  L-BFGS under rgmin `Accept::Step`, Barzilai-Borwein) and refuse the
  rest with `SaddleError::Invalid`.
* The Hessian actions use the forward difference by default; its
  curvature error is first order in `dr` (`forward_grad_diff_error`,
  `forward_grad_diff_exact_cubic`). `FiniteDifference::Central` is
  second order (`central_grad_diff_error`) for one more gradient per
  action. With `dr = 1e-3`, the forward error is `M3 * 5e-4` and the
  central error is `M4 * 1.7e-7`.

## Host mode checks

`MinModeSession::new` and `set_mode` use the same norm check.
Both reject a computed norm that is nonfinite or at most `1e-14`.
The accepted vector is divided by that norm.
The Lean normalization theorem supplies the unit-direction premise for the reflection theorem over the reals.

Run the symbolic identities and the interval certificates with:

```sh
python validation/check_mode_normalization.py
```

The command requires SymPy and Sollya on the same environment's path.
It fails if any certificate is absent or any check fails.
The Python scripts reject optimization flags that remove assertions.
The Rust and C tests exercise valid and invalid seeds through the public entry points.

For the floating-point bound, let `q` denote the exact squared norm and `u = 2^-53`.
Assume the dot product has relative error at most `eta`, and the square root and divisions have relative error at most `u`.
The normalized vector's squared norm then lies in:

```text
(1-u)^2 / ((1+eta)(1+u)^2) <= ||v_hat||^2
||v_hat||^2 <= (1+u)^2 / ((1-eta)(1-u)^2)
```

Sollya encloses both bounds with interval arithmetic for dimensions 3, 75, 512, and 4096, using `eta = 2*n*u/(1-2*n*u)`.
The checked error budget is `(4*n+16)*u`.
This relative-error model assumes finite, normal intermediate results.
Overflow and underflow require separate analysis; these certificates do not cover them.
The entry-point tests include zero, tiny, infinite, NaN, and overflowing norms.
