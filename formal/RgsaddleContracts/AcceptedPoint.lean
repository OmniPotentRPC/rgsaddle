import RgsaddleContracts.FiniteDiff

namespace RgsaddleContracts

noncomputable def cubicEnergy (a x : ℝ) : ℝ := a * x ^ 3 / 3 - x ^ 2 / 2 + (1 - a) * x

def cubicGradient (a x : ℝ) : ℝ := a * x ^ 2 - x + 1 - a

def cubicCurvature (a x : ℝ) : ℝ := 2 * a * x - 1

theorem cubic_energy_derivative (a x : ℝ) :
    HasDerivAt (cubicEnergy a) (cubicGradient a x) x := by
  exact hasDerivAt_of_eq (((dB3 a x).sub (dB2 1 x)).add (dB1 (1 - a) x))
    (fun _ => by dsimp [cubicEnergy]; ring) (by unfold cubicGradient; ring)

theorem cubic_gradient_derivative (a x : ℝ) :
    HasDerivAt (cubicGradient a) (cubicCurvature a x) x := by
  exact hasDerivAt_of_eq
    (((((hasDerivAt_pow 2 x).const_mul a).sub (hasDerivAt_id x)).add_const 1).sub_const a)
    (fun _ => rfl) (by unfold cubicCurvature; norm_num; ring)

theorem cubic_forward_curvature (a x h : ℝ) (hh : h ≠ 0) :
    (cubicGradient a (x + h) - cubicGradient a x) / h = cubicCurvature a x + a * h := by
  unfold cubicGradient cubicCurvature
  field_simp
  ring

theorem cubic_reflected_unit_step (a : ℝ) :
    (-1 : ℝ) + cubicGradient a (-1) = 1 ∧ cubicGradient a 1 = 0 := by
  unfold cubicGradient
  constructor <;> ring

theorem cubic_minimum_witness :
    cubicCurvature 1 (-1) < 0 ∧ cubicGradient 1 1 = 0 ∧ 0 < cubicCurvature 1 1 := by
  norm_num [cubicCurvature, cubicGradient]

theorem cubic_saddle_witness :
    0 < cubicCurvature (-1) (-1) ∧ cubicGradient (-1) 1 = 0 ∧ cubicCurvature (-1) 1 < 0 := by
  norm_num [cubicCurvature, cubicGradient]

theorem cubic_probe_signs :
    cubicCurvature 1 (-1) + (1 : ℝ) / 1000 < 0 ∧
    0 < cubicCurvature 1 1 + (1 : ℝ) / 1000 ∧
    0 < cubicCurvature (-1) (-1) - (1 : ℝ) / 1000 ∧
    cubicCurvature (-1) 1 - (1 : ℝ) / 1000 < 0 := by
  norm_num [cubicCurvature]

end RgsaddleContracts
