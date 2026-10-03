import Mathlib

/-! # NEB force projections

Contracts for `src/projection.rs` (`force_perp`, `climbing_image_force`,
`dneb_component`, `ProjectionKind::project`) and the parallel spring of
`src/spring.rs::SpringKind::compute`.

Every statement about orthogonality needs a **unit** tangent `t`.
`src/tangent.rs::normalize_tangent` divides by the norm when it exceeds
`1e-10` and falls back to the normalised `pos_diff_next`; when that
fallback also has norm `<= 1e-10` (two coincident images) it returns the
vector unnormalised, and `perp_not_orth_of_short` shows the
perpendicular projection then keeps a tangent component.
-/

open RealInnerProductSpace

namespace RgsaddleContracts

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- `force_perp`: `F - (F.t) t`. -/
noncomputable def perp (t F : E) : E := F - ⟪F, t⟫ • t

/-- `climbing_image_force` without the DNEB term: `F - 2 (F.t) t`.
The same reflection is the min-mode effective force. -/
noncomputable def reflect (t F : E) : E := F - (2 * ⟪F, t⟫) • t

theorem inner_self_of_unit {t : E} (ht : ‖t‖ = 1) : ⟪t, t⟫ = 1 := by
  rw [real_inner_self_eq_norm_sq, ht, one_pow]

/-- Any force splits into its perpendicular and parallel parts (no unit
hypothesis). -/
theorem perp_add_parallel (t F : E) : perp t F + ⟪F, t⟫ • t = F := by
  simp [perp]

/-- **The perpendicular projection is orthogonal to the tangent.** -/
theorem perp_orth {t : E} (ht : ‖t‖ = 1) (F : E) : ⟪perp t F, t⟫ = 0 := by
  simp [perp, inner_sub_left, real_inner_smul_left, ht]

/-- **The perpendicular projection is idempotent.** -/
theorem perp_idem {t : E} (ht : ‖t‖ = 1) (F : E) : perp t (perp t F) = perp t F := by
  conv_lhs => rw [perp, perp_orth ht F, zero_smul, sub_zero]

/-- **A spring force along the tangent has no perpendicular part.**
`SpringKind::compute` returns `parallel = c t` for every spring model
(`c = k (d_next - d_prev)`, `k_next d_next - k_prev d_prev`, or the
Onsager-Machlup projection `f_om.t`). -/
theorem perp_parallel {t : E} (ht : ‖t‖ = 1) (c : ℝ) : perp t (c • t) = 0 := by
  simp [perp, real_inner_smul_left, ht]

/-- **The NEB force keeps the spring along the band and the true force
across it.** For `ProjectionKind::Neb`, `F_neb = c t + perp t F`: its
perpendicular part is the true force's and its parallel part is the
spring's alone. -/
theorem neb_force_parts {t : E} (ht : ‖t‖ = 1) (c : ℝ) (F : E) :
    perp t (c • t + perp t F) = perp t F ∧ ⟪c • t + perp t F, t⟫ = c := by
  constructor
  · simp only [perp, inner_add_left, real_inner_smul_left, inner_self_of_unit ht,
      inner_sub_left]
    module
  · rw [inner_add_left, perp_orth ht, real_inner_smul_left, inner_self_of_unit ht]
    ring

/-- **The climbing image reverses the parallel component.** -/
theorem reflect_parallel {t : E} (ht : ‖t‖ = 1) (F : E) : ⟪reflect t F, t⟫ = -⟪F, t⟫ := by
  simp only [reflect, inner_sub_left, real_inner_smul_left, inner_self_of_unit ht]
  ring

/-- **The climbing image leaves the perpendicular component.** -/
theorem reflect_perp {t : E} (ht : ‖t‖ = 1) (F : E) : perp t (reflect t F) = perp t F := by
  simp only [perp, reflect_parallel ht]
  simp only [reflect]
  module

/-- **The reflection is an involution.** Reflecting twice returns the
force. -/
theorem reflect_reflect {t : E} (ht : ‖t‖ = 1) (F : E) : reflect t (reflect t F) = F := by
  conv_lhs => rw [reflect, reflect_parallel ht]
  simp only [reflect]
  module

/-- **The reflection commutes with negation**, so inverting the gradient
along the mode (`minmode.rs` builds `g - 2 (g.tau) tau`) is inverting
the force along it. -/
theorem reflect_neg (t F : E) : reflect t (-F) = -reflect t F := by
  simp only [reflect, inner_neg_left]
  module

/-- **The reflection preserves the norm.** -/
theorem norm_reflect {t : E} (ht : ‖t‖ = 1) (F : E) : ‖reflect t F‖ = ‖F‖ := by
  have h : ‖reflect t F‖ ^ 2 = ‖F‖ ^ 2 := by
    rw [← real_inner_self_eq_norm_sq, ← real_inner_self_eq_norm_sq]
    simp only [reflect, inner_sub_left, inner_sub_right, real_inner_smul_left,
      real_inner_smul_right, inner_self_of_unit ht, real_inner_comm t F]
    ring
  exact (pow_left_inj₀ (norm_nonneg _) (norm_nonneg _) two_ne_zero).mp h

/-- **A DNEB term orthogonal to the tangent keeps the climbing image's
parallel component reversed** (`climbing_image_force` adds
`force_dneb`). -/
theorem climb_dneb_parallel {t : E} (ht : ‖t‖ = 1) (F D : E) (hD : ⟪D, t⟫ = 0) :
    ⟪reflect t F + D, t⟫ = -⟪F, t⟫ := by
  rw [inner_add_left, reflect_parallel ht, hD, add_zero]

/-- The DNEB correction of `dneb_component`: the perpendicular spring
with its component along the unit perpendicular force removed, scaled
by the switching factor `sigma`. -/
noncomputable def dneb (σ : ℝ) (t S F : E) : E :=
  σ • (perp t S - ⟪perp t S, ‖perp t F‖⁻¹ • perp t F⟫ • (‖perp t F‖⁻¹ • perp t F))

/-- **The DNEB correction is orthogonal to the tangent.** -/
theorem dneb_orth_tangent {t : E} (ht : ‖t‖ = 1) (σ : ℝ) (S F : E) :
    ⟪dneb σ t S F, t⟫ = 0 := by
  simp only [dneb, real_inner_smul_left, inner_sub_left, perp_orth ht]
  ring

/-- **The DNEB correction is orthogonal to the perpendicular true
force** when that force is nonzero (the code requires its norm to
exceed `1e-10`). -/
theorem dneb_orth_force {t : E} (σ : ℝ) (S F : E) (hF : perp t F ≠ 0) :
    ⟪dneb σ t S F, perp t F⟫ = 0 := by
  have hn : ‖perp t F‖ ≠ 0 := norm_ne_zero_iff.mpr hF
  have hpp : ⟪perp t F, perp t F⟫ = ‖perp t F‖ ^ 2 := real_inner_self_eq_norm_sq _
  simp only [dneb, real_inner_smul_left, real_inner_smul_right, inner_sub_left, hpp]
  field_simp
  ring

/-- **The DNEB projection keeps the spring's parallel part.**
`ProjectionKind::DoublyNudged` returns `c t + perp t F + D`; its
component along `t` is `c`. -/
theorem dneb_force_parallel {t : E} (ht : ‖t‖ = 1) (c σ : ℝ) (S F : E) :
    ⟪c • t + perp t F + dneb σ t S F, t⟫ = c := by
  rw [inner_add_left, inner_add_left, dneb_orth_tangent ht, perp_orth ht,
    real_inner_smul_left, inner_self_of_unit ht]
  ring

/-- **A short tangent breaks orthogonality.** With `t = (1/2) e` for a
unit `e` (the unnormalised fallback of `normalize_tangent`), the
"perpendicular" force `perp t e` keeps a component `3/8` along `t`. -/
theorem perp_not_orth_of_short {e : E} (he : ‖e‖ = 1) :
    ⟪perp ((1 / 2 : ℝ) • e) e, (1 / 2 : ℝ) • e⟫ = 3 / 8 := by
  simp only [perp, inner_sub_left, real_inner_smul_left, real_inner_smul_right,
    inner_self_of_unit he]
  norm_num

end RgsaddleContracts
