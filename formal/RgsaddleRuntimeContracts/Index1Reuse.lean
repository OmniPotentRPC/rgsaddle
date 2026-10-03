import Mathlib

namespace RgsaddleRuntimeContracts

/-- Operations on machine values under a fixed arithmetic environment. -/
structure ScalarOps (F : Type) where
  zero : F
  add : F → F → F
  mul : F → F → F
  div : F → F → F
  sqrt : F → F

/-- The ordered accumulation in `cartesian_of`. -/
def orderedDot {F : Type} (ops : ScalarOps F)
    (a x : ℕ → F) : ℕ → F
  | 0 => ops.zero
  | k + 1 =>
      ops.add (orderedDot ops a x k) (ops.mul (a k) (x k))

theorem orderedDot_congr {F : Type} (ops : ScalarOps F)
    (a b x y : ℕ → F) :
    ∀ n,
      (∀ k, k < n → a k = b k) →
      (∀ k, k < n → x k = y k) →
      orderedDot ops a x n = orderedDot ops b y n := by
  intro n
  induction n with
  | zero =>
      intro _ _
      rfl
  | succ n ih =>
      intro ha hx
      have hd :=
        ih
          (fun k hk => ha k (Nat.lt_succ_of_lt hk))
          (fun k hk => hx k (Nat.lt_succ_of_lt hk))
      simp only [orderedDot]
      rw [hd, ha n (Nat.lt_succ_self n), hx n (Nat.lt_succ_self n)]

def cartesianRow {F : Type} (ops : ScalarOps F)
    (vectors : ℕ → ℕ → F) (step mass : ℕ → F)
    (i modes : ℕ) : F :=
  ops.div
    (orderedDot ops (vectors i) step modes)
    (ops.sqrt (mass i))

/-- Equal indexed entries preserve each Cartesian output component.
No algebraic laws for the machine operations are required. -/
theorem cartesianRow_congr {F : Type} (ops : ScalarOps F)
    (va vb : ℕ → ℕ → F)
    (sa sb ma mb : ℕ → F) (i modes : ℕ)
    (hv : ∀ k, k < modes → va i k = vb i k)
    (hs : ∀ k, k < modes → sa k = sb k)
    (hm : ma i = mb i) :
    cartesianRow ops va sa ma i modes =
      cartesianRow ops vb sb mb i modes := by
  unfold cartesianRow
  rw [orderedDot_congr ops (va i) (vb i) sa sb modes hv hs, hm]

/-- Decomposition includes mass weighting and spectrum sorting.
The remaining calculation may fail independently. -/
def recomputedResult {H M S D C E : Type}
    (decompose : H → M → Except E S)
    (displace : S → Except E D)
    (lowest : S → C) (h : H) (mass : M) : Except E (D × C) :=
  match decompose h mass with
  | .error e => .error e
  | .ok spectrum =>
      match displace spectrum with
      | .error e => .error e
      | .ok dx =>
          match decompose h mass with
          | .error e => .error e
          | .ok repeated => .ok (dx, lowest repeated)

def reusedResult {H M S D C E : Type}
    (decompose : H → M → Except E S)
    (displace : S → Except E D)
    (lowest : S → C) (h : H) (mass : M) : Except E (D × C) :=
  match decompose h mass with
  | .error e => .error e
  | .ok spectrum =>
      match displace spectrum with
      | .error e => .error e
      | .ok dx => .ok (dx, lowest spectrum)

/-- Reusing a deterministic decomposition preserves the result when
its Hessian and mass inputs are unchanged. -/
theorem reusedResult_eq_recomputedResult {H M S D C E : Type}
    (decompose : H → M → Except E S)
    (displace : S → Except E D)
    (lowest : S → C) (h : H) (mass : M) :
    reusedResult decompose displace lowest h mass =
      recomputedResult decompose displace lowest h mass := by
  cases hd : decompose h mass with
  | error e => simp [reusedResult, recomputedResult, hd]
  | ok spectrum =>
      cases hs : displace spectrum with
      | error e => simp [reusedResult, recomputedResult, hd, hs]
      | ok dx => simp [reusedResult, recomputedResult, hd, hs]

section ExactMassWeighting

variable {ι : Type} [Fintype ι]

/-- With root_i = sqrt(m_i), an exact eigenpair of the weighted
matrix gives a generalized eigenpair of the Cartesian Hessian.
S is the exact symmetric part of the supplied Hessian. -/
theorem generalized_eigenpair
    (S : ι → ι → ℝ) (root q : ι → ℝ) (lam : ℝ)
    (hr : ∀ i, root i ≠ 0)
    (heig : ∀ i,
      (∑ j, (S i j / (root i * root j)) * q j) = lam * q i) :
    ∀ i,
      (∑ j, S i j * (q j / root j)) =
        lam * (root i) ^ 2 * (q i / root i) := by
  intro i
  have hscale :
      (∑ j, S i j * (q j / root j)) =
        root i * (∑ j, (S i j / (root i * root j)) * q j) := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro j _
    field_simp [hr i, hr j]
  rw [hscale, heig i]
  field_simp [hr i]

/-- Cartesian division by square-root masses preserves the weighted
norm of the transformed vector. -/
theorem cartesian_mass_norm
    (root q : ι → ℝ) (hr : ∀ i, root i ≠ 0) :
    (∑ i, (root i) ^ 2 * (q i / root i) ^ 2) = ∑ i, (q i) ^ 2 := by
  apply Finset.sum_congr rfl
  intro i _
  field_simp [hr i]

end ExactMassWeighting
end RgsaddleRuntimeContracts
