import Mathlib

/-! # Finite-difference curvature: error orders

Contract for `src/minmode.rs::hessian_action` and `curvature_along`.

Along a unit mode `v` write `phi(t) = E(x + t v)`; then
`phi'(t) = g(x + t v).v` and `phi''(0) = v.H v`. The code estimates the
curvature with the **forward** difference of gradients,
`C_fwd = (g(x + dr v) - g(x)).v / dr = (phi'(dr) - phi'(0)) / dr`.

* `forward_grad_diff_error`: `|C_fwd - phi''(0)| <= M3 dr / 2`, first
  order in `dr`, and `forward_grad_diff_exact_cubic` shows the order is
  attained (the error is exactly `dr / 2` for `phi = t^3 / 6`).
* `central_grad_diff_error`: the central difference
  `(phi'(dr) - phi'(-dr)) / (2 dr)` has error `<= M4 dr^2 / 6`, at the
  price of one more gradient per action.
* `central_second_diff_error`: the energy-only central second
  difference `(phi(h) - 2 phi(0) + phi(-h)) / h^2` has error
  `<= M4 h^2 / 12`; `central_second_diff_error_contDiff` states it for a
  `C^4` function with Mathlib's `iteratedDeriv`.

Each bound uses only the derivative chain `f -> f1 -> f2 -> ...` as
`HasDerivAt` facts and a bound on the top derivative over the stencil.
The proofs fence each remainder with Mathlib's
`image_norm_le_of_norm_deriv_right_le_deriv_boundary` (a comparison
principle), which gives the sharp Taylor-remainder constants.
-/

open Set

namespace RgsaddleContracts

/-- Comparison on `[0, h]`: a function starting inside a bound, whose
derivative stays below the bound's derivative, stays inside it. -/
theorem fence {g g' B B' : ℝ → ℝ} {h : ℝ}
    (hg : ∀ t, HasDerivAt g (g' t) t) (hB : ∀ t, HasDerivAt B (B' t) t)
    (h0 : |g 0| ≤ B 0) (bd : ∀ t ∈ Ico 0 h, |g' t| ≤ B' t) :
    ∀ t ∈ Icc 0 h, |g t| ≤ B t := by
  intro t ht
  have := image_norm_le_of_norm_deriv_right_le_deriv_boundary (a := 0) (b := h) (f := g)
    (f' := g') (fun s _ => (hg s).continuousAt.continuousWithinAt)
    (fun s _ => (hg s).hasDerivWithinAt) (by simpa [Real.norm_eq_abs] using h0) hB
    (fun s hs => by simpa [Real.norm_eq_abs] using bd s hs) ht
  simpa [Real.norm_eq_abs] using this

/-- Transport a derivative across a pointwise equal function and an
equal derivative value. -/
theorem hasDerivAt_of_eq {f g : ℝ → ℝ} {a b s : ℝ} (h : HasDerivAt f a s)
    (hfg : ∀ y, g y = f y) (hab : a = b) : HasDerivAt g b s := by
  have e : g = f := funext hfg
  subst e
  subst hab
  exact h

theorem hasDerivAt_shift_add {f f' : ℝ → ℝ} (hf : ∀ t, HasDerivAt f (f' t) t) (x s : ℝ) :
    HasDerivAt (fun s => f (x + s)) (f' (x + s)) s :=
  hasDerivAt_of_eq ((hf (x + s)).comp s ((hasDerivAt_id s).const_add x)) (fun _ => rfl)
    (mul_one _)

theorem hasDerivAt_shift_sub {f f' : ℝ → ℝ} (hf : ∀ t, HasDerivAt f (f' t) t) (x s : ℝ) :
    HasDerivAt (fun s => f (x - s)) (-f' (x - s)) s :=
  hasDerivAt_of_eq ((hf (x - s)).comp s ((hasDerivAt_id s).const_sub x)) (fun _ => rfl)
    (by ring)

theorem dB1 (c t : ℝ) : HasDerivAt (fun s : ℝ => c * s) c t :=
  hasDerivAt_of_eq ((hasDerivAt_id t).const_mul c) (fun _ => rfl) (mul_one c)

theorem dB2 (c t : ℝ) : HasDerivAt (fun s : ℝ => c * s ^ 2 / 2) (c * t) t :=
  hasDerivAt_of_eq (((hasDerivAt_pow 2 t).const_mul c).div_const 2) (fun _ => rfl)
    (by norm_num [pow_succ]; ring)

theorem dB3 (c t : ℝ) : HasDerivAt (fun s : ℝ => c * s ^ 3 / 3) (c * t ^ 2) t :=
  hasDerivAt_of_eq (((hasDerivAt_pow 3 t).const_mul c).div_const 3) (fun _ => rfl)
    (by norm_num [pow_succ]; ring)

theorem dB4 (c t : ℝ) : HasDerivAt (fun s : ℝ => c * s ^ 4 / 12) (c * t ^ 3 / 3) t :=
  hasDerivAt_of_eq (((hasDerivAt_pow 4 t).const_mul c).div_const 12) (fun _ => rfl)
    (by norm_num [pow_succ]; ring)

/-- **Forward difference of gradients is first order.** With `f1` the
first derivative of the line function, `f2` the second, `f3` the third,
and `|f3| <= M` on `[x, x + h]`,
`|f1(x + h) - f1(x) - h f2(x)| <= M h^2 / 2`. Dividing by `h` bounds
`|C_fwd - f2(x)|` by `M h / 2`. -/
theorem forward_grad_diff_error {f1 f2 f3 : ℝ → ℝ} {x h M : ℝ}
    (h1 : ∀ t, HasDerivAt f1 (f2 t) t) (h2 : ∀ t, HasDerivAt f2 (f3 t) t)
    (hh : 0 ≤ h) (hM : ∀ t ∈ Icc x (x + h), |f3 t| ≤ M) :
    |f1 (x + h) - f1 x - h * f2 x| ≤ M * h ^ 2 / 2 := by
  have hj : ∀ s ∈ Icc 0 h, |f2 (x + s) - f2 x| ≤ M * s :=
    fence (g := fun s => f2 (x + s) - f2 x) (g' := fun s => f3 (x + s))
      (B := fun s => M * s) (B' := fun _ => M)
      (fun s => hasDerivAt_of_eq ((hasDerivAt_shift_add h2 x s).sub_const (f2 x))
        (fun _ => rfl) rfl)
      (dB1 M) (by simp)
      (fun s hs => hM (x + s) ⟨by linarith [hs.1], by linarith [hs.2]⟩)
  have hk := fence (h := h) (g := fun s => f1 (x + s) - f1 x - s * f2 x)
    (g' := fun s => f2 (x + s) - f2 x) (B := fun s => M * s ^ 2 / 2)
    (B' := fun s => M * s)
    (fun s => hasDerivAt_of_eq
      (((hasDerivAt_shift_add h1 x s).sub_const (f1 x)).sub
        ((hasDerivAt_id s).mul_const (f2 x))) (fun _ => rfl) (by simp))
    (dB2 M) (by simp) (fun s hs => hj s (Ico_subset_Icc_self hs)) h ⟨hh, le_rfl⟩
  simpa using hk

/-- **The forward order is attained.** For the line function `t^3 / 6`
(first derivative `t^2 / 2`, second `t`, third `1`) at `x = 0` the
forward error `(f1(h) - f1(0)) / h - f2(0)` is exactly `h / 2`. -/
theorem forward_grad_diff_exact_cubic (h : ℝ) (hh : h ≠ 0) :
    ((h ^ 2 / 2 - 0 ^ 2 / 2) / h - 0) = h / 2 := by
  field_simp
  ring

/-- **Central difference of gradients is second order.** With the
chain `f1 -> f2 -> f3 -> f4` and `|f4| <= M` on `[x - h, x + h]`,
`|f1(x + h) - f1(x - h) - 2 h f2(x)| <= M h^3 / 3`. Dividing by `2 h`
bounds the central curvature error by `M h^2 / 6`. -/
theorem central_grad_diff_error {f1 f2 f3 f4 : ℝ → ℝ} {x h M : ℝ}
    (h1 : ∀ t, HasDerivAt f1 (f2 t) t) (h2 : ∀ t, HasDerivAt f2 (f3 t) t)
    (h3 : ∀ t, HasDerivAt f3 (f4 t) t)
    (hh : 0 ≤ h) (hM : ∀ t ∈ Icc (x - h) (x + h), |f4 t| ≤ M) :
    |f1 (x + h) - f1 (x - h) - 2 * h * f2 x| ≤ M * h ^ 3 / 3 := by
  have hk3 : ∀ s ∈ Icc 0 h, |f3 (x + s) - f3 (x - s)| ≤ 2 * M * s := by
    refine fence (g := fun s => f3 (x + s) - f3 (x - s))
      (g' := fun s => f4 (x + s) + f4 (x - s)) (B := fun s => 2 * M * s) (B' := fun _ => 2 * M)
      (fun s => hasDerivAt_of_eq
        ((hasDerivAt_shift_add h3 x s).sub (hasDerivAt_shift_sub h3 x s))
        (fun _ => rfl) (by ring))
      (dB1 (2 * M)) (by simp) fun s hs => ?_
    have a := hM (x + s) ⟨by linarith [hs.1], by linarith [hs.2]⟩
    have b := hM (x - s) ⟨by linarith [hs.2], by linarith [hs.1]⟩
    calc |f4 (x + s) + f4 (x - s)| ≤ |f4 (x + s)| + |f4 (x - s)| := abs_add_le _ _
      _ ≤ 2 * M := by linarith
  have hk2 : ∀ s ∈ Icc 0 h, |f2 (x + s) + f2 (x - s) - 2 * f2 x| ≤ 2 * M * s ^ 2 / 2 :=
    fence (h := h) (g := fun s => f2 (x + s) + f2 (x - s) - 2 * f2 x)
      (g' := fun s => f3 (x + s) - f3 (x - s)) (B := fun s => 2 * M * s ^ 2 / 2)
      (B' := fun s => 2 * M * s)
      (fun s => hasDerivAt_of_eq
        (((hasDerivAt_shift_add h2 x s).add (hasDerivAt_shift_sub h2 x s)).sub_const
          (2 * f2 x)) (fun _ => rfl) (by ring))
      (dB2 (2 * M)) (by norm_num [two_mul])
      (fun s hs => hk3 s (Ico_subset_Icc_self hs))
  have hk1 := fence (h := h) (g := fun s => f1 (x + s) - f1 (x - s) - 2 * s * f2 x)
    (g' := fun s => f2 (x + s) + f2 (x - s) - 2 * f2 x) (B := fun s => M * s ^ 3 / 3)
    (B' := fun s => M * s ^ 2)
    (fun s => hasDerivAt_of_eq
      (((hasDerivAt_shift_add h1 x s).sub (hasDerivAt_shift_sub h1 x s)).sub
        (((hasDerivAt_id s).const_mul 2).mul_const (f2 x))) (fun _ => rfl) (by simp))
    (dB3 M) (by simp)
    (fun s hs => by
      have := hk2 s (Ico_subset_Icc_self hs)
      have e : 2 * M * s ^ 2 / 2 = M * s ^ 2 := by ring
      rw [e] at this
      exact this)
    h ⟨hh, le_rfl⟩
  simpa using hk1

/-- **The energy central second difference is second order.** With the
chain `f -> f1 -> f2 -> f3 -> f4` and `|f4| <= M` on `[x - h, x + h]`,
`|f(x + h) - 2 f(x) + f(x - h) - h^2 f2(x)| <= M h^4 / 12`. -/
theorem central_second_diff_error {f f1 f2 f3 f4 : ℝ → ℝ} {x h M : ℝ}
    (h0 : ∀ t, HasDerivAt f (f1 t) t) (h1 : ∀ t, HasDerivAt f1 (f2 t) t)
    (h2 : ∀ t, HasDerivAt f2 (f3 t) t) (h3 : ∀ t, HasDerivAt f3 (f4 t) t)
    (hh : 0 ≤ h) (hM : ∀ t ∈ Icc (x - h) (x + h), |f4 t| ≤ M) :
    |f (x + h) - 2 * f x + f (x - h) - h ^ 2 * f2 x| ≤ M * h ^ 4 / 12 := by
  have hk1 : ∀ s ∈ Icc 0 h, |f1 (x + s) - f1 (x - s) - 2 * s * f2 x| ≤ M * s ^ 3 / 3 := by
    intro s hs
    exact central_grad_diff_error h1 h2 h3 hs.1
      (fun t ht => hM t ⟨by linarith [ht.1, hs.2], by linarith [ht.2, hs.2]⟩)
  have hk0 := fence (h := h) (g := fun s => f (x + s) - 2 * f x + f (x - s) - s ^ 2 * f2 x)
    (g' := fun s => f1 (x + s) - f1 (x - s) - 2 * s * f2 x) (B := fun s => M * s ^ 4 / 12)
    (B' := fun s => M * s ^ 3 / 3)
    (fun s => hasDerivAt_of_eq
      ((((hasDerivAt_shift_add h0 x s).sub_const (2 * f x)).add
        (hasDerivAt_shift_sub h0 x s)).sub ((hasDerivAt_pow 2 s).mul_const (f2 x)))
      (fun _ => rfl) (by norm_num; ring))
    (dB4 M) (by norm_num [two_mul])
    (fun s hs => hk1 s (Ico_subset_Icc_self hs)) h ⟨hh, le_rfl⟩
  simpa using hk0

/-- **The same bound for a `C^4` function**, with the derivatives taken
as Mathlib's `iteratedDeriv`. -/
theorem central_second_diff_error_contDiff {f : ℝ → ℝ} (hf : ContDiff ℝ 4 f) {x h M : ℝ}
    (hh : 0 ≤ h) (hM : ∀ t ∈ Icc (x - h) (x + h), |iteratedDeriv 4 f t| ≤ M) :
    |f (x + h) - 2 * f x + f (x - h) - h ^ 2 * iteratedDeriv 2 f x| ≤ M * h ^ 4 / 12 := by
  have step : ∀ k : ℕ, k < 4 → ∀ t,
      HasDerivAt (iteratedDeriv k f) (iteratedDeriv (k + 1) f t) t := by
    intro k hk t
    have hd := hf.differentiable_iteratedDeriv k (by exact_mod_cast hk)
    rw [iteratedDeriv_succ]
    exact (hd t).hasDerivAt
  have h0 : ∀ t, HasDerivAt f (iteratedDeriv 1 f t) t := by
    intro t
    have := step 0 (by norm_num) t
    simpa [iteratedDeriv_zero] using this
  exact central_second_diff_error h0 (step 1 (by norm_num)) (step 2 (by norm_num))
    (step 3 (by norm_num)) hh hM

end RgsaddleContracts
