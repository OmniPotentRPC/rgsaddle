import Mathlib
import RgsaddleContracts.Projection

/-! # Minimum-mode following: Rayleigh bounds and the inverted force

Contracts for `src/minmode.rs`: `curvature_along` (the Rayleigh
quotient along the unit mode), the stopping test of `rotate_dimer`,
and the switched effective force of `effective_gradient` that
`MinModeSession::step` hands the translation solver: `F - 2 (F.tau) tau`
below a negative curvature, `-(F.tau) tau` at a nonnegative one.

The quadratic model is `E(x) = 1/2 <H d, d>`, `d = x - x_s`, with `H`
symmetric, a unit eigenvector `v` with eigenvalue `lam < 0`, and `H`
positive definite on the hyperplane orthogonal to `v`: an index-1
saddle at `x_s`. Its gradient is `H d`.
-/

open RealInnerProductSpace

namespace RgsaddleContracts

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- `checked_mode` divides a nonzero host seed by its norm. The Rust
entry points also require a finite computed norm above `1e-14`. -/
theorem normalized_mode_unit {v : E} (hv : v ≠ 0) :
    ‖(‖v‖⁻¹ : ℝ) • v‖ = 1 := by
  rw [norm_smul, norm_inv, norm_norm]
  exact inv_mul_cancel₀ (norm_ne_zero_iff.mpr hv)

/-- Normalizing the accepted seed establishes the hypothesis of the
reflected-force norm contract. -/
theorem normalized_mode_reflect_norm {v : E} (hv : v ≠ 0) (F : E) :
    ‖reflect ((‖v‖⁻¹ : ℝ) • v) F‖ = ‖F‖ :=
  norm_reflect (normalized_mode_unit hv) F

section Rayleigh

variable [FiniteDimensional ℝ E] {n : ℕ} {T : E →ₗ[ℝ] E}

/-- The quadratic form in the eigenbasis: `<T x, x> = sum lam_i <x, b_i>^2`. -/
theorem quad_eq_sum (hT : T.IsSymmetric) (hn : Module.finrank ℝ E = n) (x : E) :
    ⟪T x, x⟫ = ∑ i, hT.eigenvalues hn i * ⟪x, hT.eigenvectorBasis hn i⟫ ^ 2 := by
  rw [← (hT.eigenvectorBasis hn).sum_inner_mul_inner (T x) x]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [hT x, hT.apply_eigenvectorBasis]
  simp only [RCLike.ofReal_real_eq_id, id, real_inner_smul_right]
  rw [real_inner_comm (hT.eigenvectorBasis hn i) x]
  ring

theorem norm_sq_eq_sum (hT : T.IsSymmetric) (hn : Module.finrank ℝ E = n) (x : E) :
    ‖x‖ ^ 2 = ∑ i, ⟪x, hT.eigenvectorBasis hn i⟫ ^ 2 := by
  rw [← real_inner_self_eq_norm_sq, ← (hT.eigenvectorBasis hn).sum_inner_mul_inner x x]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [real_inner_comm (hT.eigenvectorBasis hn i) x]
  ring

/-- **The Rayleigh quotient lies between the extreme eigenvalues.**
For any bounds `m <= lam_i <= M` on the spectrum,
`m |x|^2 <= <T x, x> <= M |x|^2`; with `|x| = 1` this bounds the
curvature `curvature_along` reports (exact Hessian action). -/
theorem rayleigh_bounds (hT : T.IsSymmetric) (hn : Module.finrank ℝ E = n) {m M : ℝ}
    (hm : ∀ i, m ≤ hT.eigenvalues hn i) (hM : ∀ i, hT.eigenvalues hn i ≤ M) (x : E) :
    m * ‖x‖ ^ 2 ≤ ⟪T x, x⟫ ∧ ⟪T x, x⟫ ≤ M * ‖x‖ ^ 2 := by
  rw [quad_eq_sum hT hn, norm_sq_eq_sum hT hn, Finset.mul_sum, Finset.mul_sum]
  constructor
  · exact Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_right (hm i) (sq_nonneg _)
  · exact Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_right (hM i) (sq_nonneg _)

/-- The same bounds with the extreme eigenvalues themselves. -/
theorem rayleigh_between_extremes (hT : T.IsSymmetric) (hn : Module.finrank ℝ E = n) (x : E) :
    (⨅ i, hT.eigenvalues hn i) * ‖x‖ ^ 2 ≤ ⟪T x, x⟫ ∧
      ⟪T x, x⟫ ≤ (⨆ i, hT.eigenvalues hn i) * ‖x‖ ^ 2 :=
  rayleigh_bounds hT hn (fun i => ciInf_le (Set.finite_range _).bddBelow i)
    (fun i => le_ciSup (Set.finite_range _).bddAbove i) x

/-- **The quotient attains each eigenvalue at its eigenvector**, so the
bounds are tight. -/
theorem rayleigh_at_eigenvector (hT : T.IsSymmetric) (hn : Module.finrank ℝ E = n) (i : Fin n) :
    ⟪T (hT.eigenvectorBasis hn i), hT.eigenvectorBasis hn i⟫ = hT.eigenvalues hn i := by
  rw [hT.apply_eigenvectorBasis]
  simp only [RCLike.ofReal_real_eq_id, id, real_inner_smul_left]
  rw [real_inner_self_eq_norm_sq, (hT.eigenvectorBasis hn).orthonormal.1 i]
  ring

end Rayleigh

/-- **The rotation stops exactly at an eigenvector.** `rotate_dimer`
stops when the rotational force `H tau - (tau.H tau) tau` vanishes;
that holds iff `tau` is an eigenvector with eigenvalue the reported
curvature. -/
theorem rotational_force_zero_iff (T : E → E) (τ : E) :
    perp τ (T τ) = 0 ↔ T τ = ⟪T τ, τ⟫ • τ := by
  rw [perp, sub_eq_zero]

/-- **The inverted force flips the mode component and keeps the rest.**
The min-mode effective force is the reflection of `projection.lean`. -/
theorem minmode_force_parts {v : E} (hv : ‖v‖ = 1) (F : E) :
    ⟪reflect v F, v⟫ = -⟪F, v⟫ ∧ perp v (reflect v F) = perp v F :=
  ⟨reflect_parallel hv F, reflect_perp hv F⟩

/-- The switched force of `src/minmode.rs::effective_gradient`
(negated, `F = -g`): the reflection below a negative curvature `c`,
the reversed mode component `-(F.v) v` alone otherwise. -/
noncomputable def switchedForce (c : ℝ) (v F : E) : E :=
  if c < 0 then reflect v F else -(⟪F, v⟫ • v)

theorem switched_force_neg {c : ℝ} (hc : c < 0) (v F : E) :
    switchedForce c v F = reflect v F := by
  simp [switchedForce, hc]

theorem switched_force_nonneg {c : ℝ} (hc : 0 ≤ c) (v F : E) :
    switchedForce c v F = -(⟪F, v⟫ • v) := by
  simp [switchedForce, not_lt.mpr hc]

/-- **Both branches reverse the mode component.** -/
theorem switched_force_parallel {v : E} (hv : ‖v‖ = 1) (c : ℝ) (F : E) :
    ⟪switchedForce c v F, v⟫ = -⟪F, v⟫ := by
  unfold switchedForce
  split_ifs
  · exact reflect_parallel hv F
  · rw [inner_neg_left, real_inner_smul_left, inner_self_of_unit hv, mul_one]

/-- **At a nonnegative curvature nothing across the mode remains.** -/
theorem switched_force_perp_nonneg {v : E} (hv : ‖v‖ = 1) {c : ℝ} (hc : 0 ≤ c) (F : E) :
    perp v (switchedForce c v F) = 0 := by
  rw [switched_force_nonneg hc, ← neg_smul]
  exact perp_parallel hv _

/-- **Below a negative curvature the force across the mode is kept.** -/
theorem switched_force_perp_neg {v : E} (hv : ‖v‖ = 1) {c : ℝ} (hc : c < 0) (F : E) :
    perp v (switchedForce c v F) = perp v F := by
  rw [switched_force_neg hc]
  exact reflect_perp hv F

/-- **In the convex region every step climbs.** Along the switched force
at a nonnegative curvature the true energy changes at the rate
`<g, F_eff> = (F.v)^2 >= 0` (`g = -F`): no step spends itself relaxing
back into the basin. The reflected force's rate,
`(F.v)^2 - |perp v F|^2`, has no sign. -/
theorem switched_force_energy_rate {c : ℝ} (hc : 0 ≤ c) (v F : E) :
    ⟪-F, switchedForce c v F⟫ = ⟪F, v⟫ ^ 2 := by
  rw [switched_force_nonneg hc, inner_neg_left, inner_neg_right, neg_neg,
    real_inner_smul_right]
  ring

section Quadratic

variable {H : E →ₗ[ℝ] E} {v : E} {lam : ℝ}

theorem quad_key (hS : H.IsSymmetric) (hv : ‖v‖ = 1) (hHv : H v = lam • v)
    (a : ℝ) {w : E} (hw : ⟪w, v⟫ = 0) :
    ⟪reflect v (H (a • v + w)), a • v + w⟫ = -(a ^ 2 * lam) + ⟪H w, w⟫ ∧
      ⟪H (a • v + w), a • v + w⟫ = a ^ 2 * lam + ⟪H w, w⟫ := by
  have hvv : ⟪v, v⟫ = 1 := inner_self_of_unit hv
  have hw' : ⟪v, w⟫ = 0 := by rw [real_inner_comm]; exact hw
  have hHw : ⟪H w, v⟫ = 0 := by
    rw [hS w v, hHv, real_inner_smul_right, hw, mul_zero]
  have hHw' : ⟪v, H w⟫ = 0 := by rw [real_inner_comm]; exact hHw
  constructor
  · simp only [reflect, map_add, map_smul, hHv, inner_add_left, inner_add_right,
      inner_sub_left, real_inner_smul_left, real_inner_smul_right, hvv, hw', hHw]
    ring
  · simp only [map_add, map_smul, hHv, inner_add_left, inner_add_right,
      real_inner_smul_left, real_inner_smul_right, hvv, hw', hHw]
    ring

/-- **The inverted force points at the saddle of the quadratic model.**
With `g = H d` and the min-mode force `F_mm = -(g - 2 (g.v) v)`,
`<F_mm, x_s - x> > 0` for every `x /= x_s`: gradient descent on the
inverted force contracts the distance to the index-1 saddle, the
stability statement behind `MinModeSession::step`. -/
theorem minmode_descends_to_saddle (hS : H.IsSymmetric) (hv : ‖v‖ = 1)
    (hHv : H v = lam • v) (hlam : lam < 0)
    (hpos : ∀ w, ⟪w, v⟫ = 0 → w ≠ 0 → 0 < ⟪H w, w⟫) {d : E} (hd : d ≠ 0) :
    0 < ⟪-reflect v (H d), -d⟫ := by
  rw [inner_neg_left, inner_neg_right, neg_neg]
  set a := ⟪d, v⟫
  set w := d - a • v with hwdef
  have hdw : d = a • v + w := by rw [hwdef]; abel
  have hw : ⟪w, v⟫ = 0 := by
    rw [hwdef, inner_sub_left, real_inner_smul_left, inner_self_of_unit hv]
    ring
  rw [hdw, (quad_key hS hv hHv a hw).1]
  by_cases hw0 : w = 0
  · have ha : a ≠ 0 := by
      intro ha
      apply hd
      rw [hdw, ha, hw0, zero_smul, add_zero]
    rw [hw0, map_zero, inner_zero_left, add_zero]
    have : 0 < a ^ 2 := lt_of_le_of_ne (sq_nonneg _) (Ne.symm (pow_ne_zero 2 ha))
    nlinarith
  · have := hpos w hw hw0
    nlinarith [sq_nonneg a]

/-- **The inverted operator is symmetric** on the model: the inverted
force is the negative gradient of the quadratic form
`1/2 <R H d, d>`, `R` the reflection (a standard fact about symmetric
forms, not formalised here). -/
theorem reflect_comp_symm (hS : H.IsSymmetric) (hHv : H v = lam • v) (x y : E) :
    ⟪reflect v (H x), y⟫ = ⟪x, reflect v (H y)⟫ := by
  have hx : ⟪H x, v⟫ = lam * ⟪x, v⟫ := by rw [hS x v, hHv, real_inner_smul_right]
  have hy : ⟪H y, v⟫ = lam * ⟪y, v⟫ := by rw [hS y v, hHv, real_inner_smul_right]
  simp only [reflect, inner_sub_left, inner_sub_right, real_inner_smul_left,
    real_inner_smul_right, hx, hy]
  rw [hS x y, real_inner_comm v y]
  ring

/-- **The energy the translation oracle reports is not the potential of
the force it reports.** `MinModeSession::step` returns the true energy
`e` with the inverted gradient. On the model the form whose gradient is
the inverted one exceeds the true one by `-2 a^2 lam = 2 a^2 |lam|`,
`a = d.v`: the two agree only on the hyperplane `a = 0`. An
energy-based acceptance test (Armijo, `take_step`'s `nval < value`)
therefore sees the climb along the mode as uphill. -/
theorem minmode_energy_mismatch (hS : H.IsSymmetric) (hv : ‖v‖ = 1) (hHv : H v = lam • v)
    (a : ℝ) {w : E} (hw : ⟪w, v⟫ = 0) :
    ⟪reflect v (H (a • v + w)), a • v + w⟫ - ⟪H (a • v + w), a • v + w⟫ = -2 * a ^ 2 * lam := by
  rw [(quad_key hS hv hHv a hw).1, (quad_key hS hv hHv a hw).2]
  ring

/-- **From a minimum basin the switched force points out along the
mode.** On the model with a unit eigenvector `v` of eigenvalue `lam`,
the switched force at a nonnegative curvature is `lam a v`, `a = d.v`,
and `<F_eff, d> = lam a^2`: with `lam > 0` the walker moves away from
the minimum along `v` and keeps its coordinates across `v`. -/
theorem switched_force_leaves_minimum (hS : H.IsSymmetric) (hHv : H v = lam • v)
    {c : ℝ} (hc : 0 ≤ c) (d : E) :
    switchedForce c v (-H d) = (lam * ⟪d, v⟫) • v ∧
      ⟪switchedForce c v (-H d), d⟫ = lam * ⟪d, v⟫ ^ 2 := by
  have hHd : ⟪H d, v⟫ = lam * ⟪d, v⟫ := by rw [hS d v, hHv, real_inner_smul_right]
  have h1 : switchedForce c v (-H d) = (lam * ⟪d, v⟫) • v := by
    rw [switched_force_nonneg hc, inner_neg_left, hHd, neg_smul, neg_neg]
  refine ⟨h1, ?_⟩
  rw [h1, real_inner_smul_left, real_inner_comm d v]
  ring

/-- **From the saddle region the switched force is the inverted force**
and points at the saddle (`minmode_descends_to_saddle`). -/
theorem switched_descends_to_saddle (hS : H.IsSymmetric) (hv : ‖v‖ = 1)
    (hHv : H v = lam • v) (hlam : lam < 0)
    (hpos : ∀ w, ⟪w, v⟫ = 0 → w ≠ 0 → 0 < ⟪H w, w⟫) {c : ℝ} (hc : c < 0)
    {d : E} (hd : d ≠ 0) :
    0 < ⟪switchedForce c v (-H d), -d⟫ := by
  rw [switched_force_neg hc, reflect_neg]
  exact minmode_descends_to_saddle hS hv hHv hlam hpos hd

end Quadratic

end RgsaddleContracts
