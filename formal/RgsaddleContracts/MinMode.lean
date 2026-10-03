import Mathlib
import RgsaddleContracts.Projection

/-! # Minimum-mode following: Rayleigh bounds and the inverted force

Contracts for `src/minmode.rs`: `curvature_along` (the Rayleigh
quotient along the unit mode), the stopping test of `rotate_dimer`,
and the effective gradient `g - 2 (g.tau) tau` that
`MinModeSession::step` hands the translation solver.

The quadratic model is `E(x) = 1/2 <H d, d>`, `d = x - x_s`, with `H`
symmetric, a unit eigenvector `v` with eigenvalue `lam < 0`, and `H`
positive definite on the hyperplane orthogonal to `v`: an index-1
saddle at `x_s`. Its gradient is `H d`.
-/

open RealInnerProductSpace

namespace RgsaddleContracts

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

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

end Quadratic

end RgsaddleContracts
