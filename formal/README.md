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
| `quad_eq_sum`, `rayleigh_bounds`, `rayleigh_between_extremes`, `rayleigh_at_eigenvector` | `src/minmode.rs::rotate_dimer` and `lanczos_mode` (the curvature `h.n` of a unit mode) | A unit mode and a symmetric Hessian. The reported curvature then lies between the extreme eigenvalues and reaches each one at its eigenvector. |
| `rotational_force_zero_iff` | `src/minmode.rs::rotate_dimer` | None. The rotational force vanishes exactly at an eigenvector. |
| `minmode_force_parts`, `reflect_neg` | `src/minmode.rs::MinModeSession::step` | A unit mode (`normalize` keeps it unit). The inverted gradient flips the mode component and keeps the rest. |
| `quad_key`, `minmode_descends_to_saddle` | `src/minmode.rs::MinModeSession::step` | A quadratic model with an exact unit eigenvector of negative eigenvalue and positive curvature orthogonal to it. The inverted force then points at the saddle from every other point. |
| `reflect_comp_symm`, `minmode_energy_mismatch` | `src/minmode.rs::MinModeSession::step` | Same model (symmetry needs only the eigenvector). The inverted operator is symmetric. Its form differs from the true energy by `2 a^2 |lam|`, `a = d.v`. |
| `fence` | (lemma) | Comparison principle on `[0, h]`. |
| `forward_grad_diff_error`, `forward_grad_diff_exact_cubic` | `src/minmode.rs::Counted::action` (`FiniteDifference::Forward`) | A bound on the third derivative along the mode. The forward-difference curvature error is then at most `M3 dr / 2`, and this order is attained. |
| `central_grad_diff_error` | `src/minmode.rs::Counted::action` (`FiniteDifference::Central`) | A bound on the fourth derivative. The error is then at most `M4 dr^2 / 6`, for one more gradient per action. |
| `central_second_diff_error`, `central_second_diff_error_contDiff` | (energy-only curvature) | A bound on the fourth derivative. The error of `(E(x+h) - 2E(x) + E(x-h)) / h^2` is then at most `M4 h^2 / 12`. |

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
