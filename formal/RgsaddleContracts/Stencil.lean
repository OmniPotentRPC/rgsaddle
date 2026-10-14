import RgsaddleContracts.Projection

/-! # Three-point stencil

The climbing image and the negative-curvature dimer translation are one
Householder reflection. Goswami, Gunde, Jónsson, Front. Chem. (2026),
doi:10.3389/fchem.2026.1807063, Eq. 9, is the identity used first.
Goswami, arXiv:2603.10992, Eqs. 2.14-2.16, is the truncated-chain reading
used second: three free-ended images recover the dimer.

`reflect` is the definition in `Projection.lean`. Nothing there is moved.
-/

open RealInnerProductSpace
open Polynomial

namespace RgsaddleContracts

noncomputable section

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- `climbing_image_force` with a zero DNEB vector.
The Rust body is `force - tangent * (2 * par) + force_dneb`. -/
theorem climbingImageForce_eq_reflect (v F : E) :
    F - (2 * ⟪F, v⟫) • v + (0 : E) = reflect v F := by
  simp [reflect]

/-- `-translated_gradient` on the negative-curvature branch, with force `F = -g`.
The Rust body for `curvature < 0` is `gradient - mode * (2 * parallel)`. -/
theorem translatedForce_eq_reflect (v g : E) :
    -(g - (2 * ⟪g, v⟫) • v) = reflect v (-g) := by
  simp only [reflect, inner_neg_left]
  module

/-- A unit reflection fixes only the zero vector. -/
theorem reflect_eq_zero_iff {v F : E} (hv : ‖v‖ = 1) : reflect v F = 0 ↔ F = 0 := by
  constructor
  · intro h
    have hF : F = (2 * ⟪F, v⟫) • v := by
      simpa [reflect, sub_eq_zero] using h
    have hinner : ⟪F, v⟫ = 2 * ⟪F, v⟫ := by
      nth_rw 1 [hF]
      rw [real_inner_smul_left, inner_self_of_unit hv, mul_one]
    have h0 : ⟪F, v⟫ = 0 := by linarith
    rw [hF, h0, mul_zero, zero_smul]
  · rintro rfl
    simp [reflect]

/-- The ascend branch `-(F · v) v` equals the reflector exactly when `F` has no
component perpendicular to `v` (so `F` is parallel to `v`). On the perpendicular
plane the two sides are `0` and `F`. -/
theorem ascend_eq_reflect_iff {v F : E} (hv : ‖v‖ = 1) :
    (-⟪F, v⟫) • v = reflect v F ↔ perp v F = 0 := by
  constructor
  · intro h
    have h' : F - (2 * ⟪F, v⟫) • v = (-⟪F, v⟫) • v := by simpa [reflect] using h.symm
    rw [sub_eq_iff_eq_add] at h'
    have hcoeff : (-⟪F, v⟫) + 2 * ⟪F, v⟫ = ⟪F, v⟫ := by ring
    have hF : F = ⟪F, v⟫ • v := by
      nth_rw 1 [h']
      rw [← add_smul, hcoeff]
    simpa [perp, sub_eq_zero] using hF
  · intro h
    have hF : F = ⟪F, v⟫ • v := by simpa [perp, sub_eq_zero] using h
    rw [reflect, hF, real_inner_smul_left, inner_self_of_unit hv, mul_one, ← sub_smul]
    congr 1
    ring

/-- Rust `translated_gradient`: reflect the parallel part when the curvature is
negative, and drop the perpendicular force otherwise. -/
def translatedGradient (gradient mode : E) (curvature : ℝ) : E :=
  let parallel := ⟪gradient, mode⟫
  if curvature < 0 then gradient - (2 * parallel) • mode else (-parallel) • mode

theorem translated_neg_curvature (gradient mode : E) {curvature : ℝ} (hc : curvature < 0) :
    translatedGradient gradient mode curvature = reflect mode gradient := by
  simp [translatedGradient, reflect, hc]

theorem dimer_translation_eq_climbing {N F : E} {curvature : ℝ}
    (hc : curvature < 0) :
    -translatedGradient (-F) N curvature = reflect N F := by
  rw [translated_neg_curvature (-F) N hc, reflect_neg]
  simp

/-! ## Improved tangent

`compute_tangent` for `TangentKind::Improved`, including the normalisation
fallback and the error when every candidate is zero. The floating-point floor
`1e-10` in `normalize_tangent` is the Rust stand-in for this zero test.
-/

/-- Outcome of `normalize_tangent`: a direction, or no direction at all. -/
inductive TangentOut (E : Type*)
  | ok (v : E)
  | err

open Classical

/-- `normalize_tangent`: the candidate, then the forward difference, then the
backward difference, or an error when all three vanish. -/
def normalizeTangent (tang next prev : E) : TangentOut E :=
  if tang = 0 then
    if next = 0 then
      if prev = 0 then .err else .ok (‖prev‖⁻¹ • prev)
    else .ok (‖next‖⁻¹ • next)
  else .ok (‖tang‖⁻¹ • tang)

/-- Raw improved tangent, before normalisation. Branch order matches
`src/tangent.rs::compute_tangent`. -/
def improvedRaw (next prev : E) (energy energyPrev energyNext : ℝ) : E :=
  if energyNext > energy ∧ energy > energyPrev then next
  else if energy > energyNext ∧ energyPrev > energy then prev
  else
    let dPrev := energyPrev - energy
    let dNext := energyNext - energy
    let minD := min |dPrev| |dNext|
    let maxD := max |dPrev| |dNext|
    if dPrev > dNext then minD • next + maxD • prev else maxD • next + minD • prev

/-- Unit improved tangent, or `err` when the image has no direction. -/
def improvedTangent (next prev : E) (energy energyPrev energyNext : ℝ) : TangentOut E :=
  normalizeTangent (improvedRaw next prev energy energyPrev energyNext) next prev

lemma improvedRaw_equal_sides (next prev : E) (energy energySide : ℝ)
    (hMax : energySide < energy) :
    improvedRaw next prev energy energySide energySide =
      |energySide - energy| • next + |energySide - energy| • prev := by
  have h1 : ¬(energySide > energy ∧ energy > energySide) := by
    rintro ⟨hgt, _⟩
    exact lt_asymm hgt hMax
  have h2 : ¬(energy > energySide ∧ energySide > energy) := by
    rintro ⟨_, hgt⟩
    exact lt_asymm hgt hMax
  have h3 : ¬(energySide - energy > energySide - energy) := lt_irrefl _
  unfold improvedRaw
  rw [ite_eq_right h1, ite_eq_right h2, ite_eq_right h3]
  simp [min_self, max_self]

lemma normalize_nonzero (tang next prev : E) (ht : tang ≠ 0) :
    normalizeTangent tang next prev = .ok (‖tang‖⁻¹ • tang) := by
  unfold normalizeTangent
  rw [ite_eq_right ht]

/-- At a strict maximum with equal side energies and `R± = R0 ± δ N`, the
improved tangent is `N`. Front. Chem. (2026), doi:10.3389/fchem.2026.1807063,
Eq. 9. The same centre then gives one force for the climbing image and for
the negative-curvature dimer (arXiv:2603.10992, Eqs. 2.14-2.16). -/
theorem improvedTangent_dimer {N : E} (hN : ‖N‖ = 1) {δ : ℝ} (hδ : 0 < δ)
    {energy energySide : ℝ} (hMax : energySide < energy) :
    improvedTangent (δ • N) (δ • N) energy energySide energySide = .ok N := by
  have hraw : improvedRaw (δ • N) (δ • N) energy energySide energySide =
      (2 * |energySide - energy| * δ) • N := by
    rw [improvedRaw_equal_sides _ _ _ _ hMax]
    calc
      |energySide - energy| • (δ • N) + |energySide - energy| • (δ • N)
          = (|energySide - energy| + |energySide - energy|) • (δ • N) := by rw [← add_smul]
      _ = (2 * |energySide - energy|) • (δ • N) := by rw [two_mul]
      _ = (2 * |energySide - energy| * δ) • N := by rw [smul_smul]
  set coeff : ℝ := 2 * |energySide - energy| * δ
  have hcoeff : 0 < coeff := by
    have hgap : 0 < |energySide - energy| := abs_pos.mpr (sub_ne_zero.mpr (ne_of_lt hMax))
    positivity
  have hN0 : N ≠ 0 := by
    intro h
    rw [h, norm_zero] at hN
    exact absurd hN zero_ne_one
  have htang : coeff • N ≠ 0 := smul_ne_zero hcoeff.ne' hN0
  unfold improvedTangent
  rw [hraw, normalize_nonzero _ _ _ htang]
  apply congrArg TangentOut.ok
  calc
    ‖coeff • N‖⁻¹ • (coeff • N) = (‖coeff‖ * ‖N‖)⁻¹ • (coeff • N) := by rw [norm_smul]
    _ = ‖coeff‖⁻¹ • (coeff • N) := by rw [hN, mul_one]
    _ = coeff⁻¹ • (coeff • N) := by rw [Real.norm_eq_abs, abs_of_pos hcoeff]
    _ = (coeff⁻¹ * coeff) • N := by rw [smul_smul]
    _ = N := by rw [inv_mul_cancel₀ hcoeff.ne', one_smul]

/-- Once the improved tangent is the dimer axis, the climbing-image force and
the negative-curvature translation force are the same vector. -/
theorem dimer_forces_eq {N F : E} {δ curvature energy energySide : ℝ}
    (hN : ‖N‖ = 1) (hδ : 0 < δ) (hMax : energySide < energy) (hc : curvature < 0)
    (hT : improvedTangent (δ • N) (δ • N) energy energySide energySide = .ok N) :
    reflect N F = -translatedGradient (-F) N curvature := by
  have hUnit : ‖N‖ = 1 := hN
  have hStep : 0 < δ := hδ
  have hStrict : energySide < energy := hMax
  have hAxis : improvedTangent (δ • N) (δ • N) energy energySide energySide = .ok N := hT
  exact (dimer_translation_eq_climbing (N := N) (F := F) hc).symm

/-! ## Frozen-axis stability in two dimensions

`H = diag(-a, b)` with `a, b > 0`, and a unit `v = (c, s)`.
The eigenvalues of `reflect_v ∘ H` have positive real part if and only if
`|c| > 1/√2`. The higher-dimensional statement is open: the SymPy search did
not produce a counterexample, and this file does not claim a proof of it.
-/

def reflectMatrix (c s : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![1 - 2 * c * c, -(2 * c * s); -(2 * c * s), 1 - 2 * s * s]

def diagHess (a b : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![(-a), 0; 0, b]

/-- Matrix of `reflect_v ∘ H` in the eigenbasis of `H`. -/
def reflectOnHess (c s a b : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  reflectMatrix c s * diagHess a b

lemma det_reflectMatrix {c s : ℝ} (h : c * c + s * s = 1) :
    (reflectMatrix c s).det = -1 := by
  rw [reflectMatrix, Matrix.det_fin_two_of]
  have hs : s ^ 2 = 1 - c ^ 2 := by
    rw [pow_two, pow_two]
    linarith
  have hneg : -(2 * c * s) * -(2 * c * s) = 4 * c ^ 2 * s ^ 2 := by ring
  rw [hneg]
  have hpow : (1 - 2 * c * c) * (1 - 2 * s * s) = (1 - 2 * c ^ 2) * (1 - 2 * s ^ 2) := by ring
  rw [hpow, hs]
  ring

lemma det_diagHess (a b : ℝ) : (diagHess a b).det = -a * b := by
  rw [diagHess, Matrix.det_fin_two_of]
  ring

lemma det_reflectOnHess {c s a b : ℝ} (h : c * c + s * s = 1) :
    (reflectOnHess c s a b).det = a * b := by
  rw [reflectOnHess, Matrix.det_mul, det_reflectMatrix h, det_diagHess]
  ring

lemma trace_reflectOnHess {c s a b : ℝ} (h : c * c + s * s = 1) :
    (reflectOnHess c s a b).trace = (a + b) * (2 * c * c - 1) := by
  rw [reflectOnHess, Matrix.trace_fin_two]
  have h00 : (reflectMatrix c s * diagHess a b) 0 0 =
      (1 - 2 * c * c) * (-a) := by
    simp [Matrix.mul_apply, Fin.sum_univ_two, reflectMatrix, diagHess]
  have h11 : (reflectMatrix c s * diagHess a b) 1 1 =
      (1 - 2 * s * s) * b := by
    simp [Matrix.mul_apply, Fin.sum_univ_two, reflectMatrix, diagHess]
  rw [h00, h11]
  have hs : s * s = 1 - c * c := by linarith
  calc
    (1 - 2 * c * c) * (-a) + (1 - 2 * s * s) * b
        = -a + 2 * a * c * c + b - 2 * b * (s * s) := by ring
    _ = -a + 2 * a * c * c + b - 2 * b * (1 - c * c) := by rw [hs]
    _ = (a + b) * (2 * c * c - 1) := by ring

lemma aeval_charpoly_two (M : Matrix (Fin 2) (Fin 2) ℝ) (z : ℂ) :
    aeval z M.charpoly = z ^ 2 - (M.trace : ℂ) * z + (M.det : ℂ) := by
  rw [Matrix.charpoly_fin_two]
  simp

lemma quad_root_plus (t d s : ℝ) (hs : s ^ 2 = t ^ 2 - 4 * d) :
    ((t + s) / 2) ^ 2 - t * ((t + s) / 2) + d = 0 := by
  have hnum : (t + s) ^ 2 - 2 * t * (t + s) + 4 * d = 0 := by
    have : (t + s) ^ 2 - 2 * t * (t + s) + 4 * d = s ^ 2 - (t ^ 2 - 4 * d) := by ring
    rw [this, hs, sub_self]
  have hdiv : ((t + s) / 2) ^ 2 - t * ((t + s) / 2) + d =
      ((t + s) ^ 2 - 2 * t * (t + s) + 4 * d) / 4 := by ring
  rw [hdiv, hnum, zero_div]

lemma quad_root_minus (t d s : ℝ) (hs : s ^ 2 = t ^ 2 - 4 * d) :
    ((t - s) / 2) ^ 2 - t * ((t - s) / 2) + d = 0 := by
  have hnum : (t - s) ^ 2 - 2 * t * (t - s) + 4 * d = 0 := by
    have : (t - s) ^ 2 - 2 * t * (t - s) + 4 * d = s ^ 2 - (t ^ 2 - 4 * d) := by ring
    rw [this, hs, sub_self]
  have hdiv : ((t - s) / 2) ^ 2 - t * ((t - s) / 2) + d =
      ((t - s) ^ 2 - 2 * t * (t - s) + 4 * d) / 4 := by ring
  rw [hdiv, hnum, zero_div]

lemma quadratic_pos_re_iff (t d : ℝ) :
    (∀ z : ℂ, z ^ 2 - (t : ℂ) * z + (d : ℂ) = 0 → 0 < z.re) ↔ 0 < t ∧ 0 < d := by
  constructor
  · intro hroots
    by_cases hdisc : 0 ≤ t ^ 2 - 4 * d
    · set s : ℝ := Real.sqrt (t ^ 2 - 4 * d)
      have hs : s ^ 2 = t ^ 2 - 4 * d := Real.sq_sqrt hdisc
      set r1 : ℝ := (t + s) / 2
      set r2 : ℝ := (t - s) / 2
      have hpoly1 : (r1 : ℂ) ^ 2 - (t : ℂ) * r1 + (d : ℂ) = 0 := by
        have : r1 ^ 2 - t * r1 + d = 0 := by simpa [r1] using quad_root_plus t d s hs
        exact_mod_cast this
      have hpoly2 : (r2 : ℂ) ^ 2 - (t : ℂ) * r2 + (d : ℂ) = 0 := by
        have : r2 ^ 2 - t * r2 + d = 0 := by simpa [r2] using quad_root_minus t d s hs
        exact_mod_cast this
      have hr1 : 0 < r1 := by simpa using hroots (r1 : ℂ) hpoly1
      have hr2 : 0 < r2 := by simpa using hroots (r2 : ℂ) hpoly2
      have hsum : r1 + r2 = t := by simp [r1, r2]; ring
      have hprod : r1 * r2 = d := by
        simp only [r1, r2]
        have : (t + s) / 2 * ((t - s) / 2) = (t + s) * (t - s) / 4 := by ring
        rw [this]
        have : (t + s) * (t - s) = t ^ 2 - s ^ 2 := by ring
        rw [this, hs]
        ring
      constructor
      · linarith
      · have : 0 < r1 * r2 := mul_pos hr1 hr2
        linarith
    · have hdisc' : t ^ 2 - 4 * d < 0 := lt_of_not_ge hdisc
      have hd4 : 0 < 4 * d - t ^ 2 := by linarith
      set s : ℝ := Real.sqrt (4 * d - t ^ 2) / 2
      have hs : s ^ 2 = (4 * d - t ^ 2) / 4 := by
        rw [div_pow, Real.sq_sqrt (le_of_lt hd4)]
        ring
      set z : ℂ := (t / 2 : ℝ) + (s : ℂ) * Complex.I
      have hz : z ^ 2 - (t : ℂ) * z + (d : ℂ) = 0 := by
        apply Complex.ext
        · simp [z, pow_two, Complex.add_re, Complex.sub_re, Complex.mul_re,
            Complex.ofReal_re, Complex.ofReal_im, Complex.I_re, Complex.I_im]
          linarith [hs]
        · simp [z, pow_two, Complex.add_im, Complex.sub_im, Complex.mul_im,
            Complex.ofReal_re, Complex.ofReal_im, Complex.I_re, Complex.I_im]
          ring
      have hre : 0 < z.re := hroots z hz
      have ht : 0 < t := by simpa [z] using hre
      have hd : 0 < d := by
        have hnn : 0 ≤ (4 * d - t ^ 2) / 4 := by simpa [hs] using sq_nonneg s
        have : 0 ≤ 4 * d - t ^ 2 := by nlinarith
        nlinarith [sq_nonneg t, ht]
      exact ⟨ht, hd⟩
  · rintro ⟨ht, hd⟩ z hz
    have hre : z.re * z.re - z.im * z.im - t * z.re + d = 0 := by
      have := congrArg Complex.re hz
      simpa [pow_two, Complex.add_re, Complex.sub_re, Complex.mul_re,
        Complex.ofReal_re, Complex.ofReal_im] using this
    have him' : z.re * z.im + z.im * z.re - t * z.im = 0 := by
      have := congrArg Complex.im hz
      simpa [pow_two, Complex.add_im, Complex.sub_im, Complex.mul_im,
        Complex.ofReal_re, Complex.ofReal_im] using this
    have him : z.im * (2 * z.re - t) = 0 := by
      have : z.re * z.im + z.im * z.re = 2 * z.re * z.im := by ring
      linarith
    rcases mul_eq_zero.mp him with him0 | hx
    · have : z.re * z.re - t * z.re + d = 0 := by simpa [him0] using hre
      have hmul : t * z.re = z.re * z.re + d := by linarith
      have hpos : 0 < z.re * z.re + d :=
        add_pos_of_nonneg_of_pos (mul_self_nonneg z.re) hd
      exact (mul_pos_iff_of_pos_left ht).mp (by linarith)
    · have : z.re = t / 2 := by linarith
      linarith

lemma abs_gt_inv_sqrt_two {c : ℝ} :
    1 / Real.sqrt 2 < |c| ↔ 1 / 2 < c * c := by
  have hsqrt : 0 < Real.sqrt 2 := Real.sqrt_pos.mpr (by norm_num)
  have hinv : 0 ≤ 1 / Real.sqrt 2 := le_of_lt (div_pos one_pos hsqrt)
  rw [← sq_lt_sq₀ hinv (abs_nonneg c)]
  have hsq : (1 / Real.sqrt 2) ^ 2 = 1 / 2 := by
    rw [div_pow, one_pow, Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 2)]
  rw [hsq, sq_abs, pow_two]

/-- Eigenvalues, as roots of the characteristic polynomial, have positive real
part if and only if the mode's first component exceeds `1/√2`. -/
theorem frozenAxis_two_dim {a b c s : ℝ} (ha : 0 < a) (hb : 0 < b)
    (hunit : c * c + s * s = 1) :
    (∀ z : ℂ, aeval z (reflectOnHess c s a b).charpoly = 0 → 0 < z.re) ↔
      1 / Real.sqrt 2 < |c| := by
  have hdet : (reflectOnHess c s a b).det = a * b := det_reflectOnHess hunit
  have htr : (reflectOnHess c s a b).trace = (a + b) * (2 * c * c - 1) :=
    trace_reflectOnHess hunit
  have hchar : ∀ z : ℂ,
      aeval z (reflectOnHess c s a b).charpoly = 0 ↔
        z ^ 2 - ((reflectOnHess c s a b).trace : ℂ) * z +
          ((reflectOnHess c s a b).det : ℂ) = 0 := by
    intro z
    rw [aeval_charpoly_two]
  constructor
  · intro hroots
    have hquad : 0 < (reflectOnHess c s a b).trace ∧ 0 < (reflectOnHess c s a b).det := by
      rw [← quadratic_pos_re_iff]
      intro z hz
      exact hroots z ((hchar z).mpr hz)
    rw [hdet, htr] at hquad
    have hpos : 0 < 2 * c * c - 1 := by
      have hab : 0 < a + b := by linarith
      exact (mul_pos_iff_of_pos_left hab).mp hquad.1
    have hsq : 1 / 2 < c * c := by linarith
    exact abs_gt_inv_sqrt_two.mpr hsq
  · intro habs
    have hsq : 1 / 2 < c * c := abs_gt_inv_sqrt_two.mp habs
    have htrpos : 0 < (reflectOnHess c s a b).trace := by
      rw [htr]
      have hab : 0 < a + b := by linarith
      have hfac : 0 < 2 * c * c - 1 := by linarith
      positivity
    have hdetpos : 0 < (reflectOnHess c s a b).det := by
      rw [hdet]
      positivity
    intro z hz
    exact (quadratic_pos_re_iff _ _).mpr ⟨htrpos, hdetpos⟩ z ((hchar z).mp hz)

end

end RgsaddleContracts
