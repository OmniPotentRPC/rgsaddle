import Mathlib

/-! # Euclidean radial restriction

These statements cover exact real arithmetic. Floating-point radius
checks are executable tests of the returned displacement.
-/

namespace RgsaddleContracts

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

theorem radial_scale_norm (v : E) (radius : ℝ)
    (hr : 0 ≤ radius) (hv : 0 < ‖v‖) :
    ‖(radius/‖v‖) • v‖ = radius := by
  rw [norm_smul, Real.norm_eq_abs,
    abs_of_nonneg (div_nonneg hr (le_of_lt hv))]
  exact div_mul_cancel₀ radius (ne_of_gt hv)

noncomputable def radialCap (radius : ℝ) (v : E) : E :=
  if ‖v‖ ≤ radius then v else (radius/‖v‖) • v

theorem radial_cap_bound (v : E) (radius : ℝ) (hr : 0 ≤ radius) :
    ‖radialCap radius v‖ ≤ radius := by
  by_cases h : ‖v‖ ≤ radius
  · simpa [radialCap, h] using h
  · have hv : 0 < ‖v‖ := lt_of_le_of_lt hr (lt_of_not_ge h)
    simp only [radialCap, if_neg h]
    rw [radial_scale_norm v radius hr hv]

end RgsaddleContracts
