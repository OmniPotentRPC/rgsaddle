This page explains how the exact references used by `tests/ohtst_pafi_exact.rs` are computed. The aim is to validate sampled PAFI and OH-TST estimates against deterministic references on surfaces where the relevant integrals are tractable.


# Hyperplane free energy by quadrature

For a hyperplane family parameterized by reaction coordinate `r`, the reference free energy is

`F(r) = -kT ln Z(r)`, with
`Z(r) = ∫ exp(-V(q)/kT) δ(g_r(q)) |J_r(q)| dq`.

In two dimensions the hyperplanes are lines, so the constrained integral is one-dimensional. The tests evaluate each line integral with adaptive Simpson quadrature on an automatically truncated support where the Boltzmann factor is non-negligible.

Along a prescribed path (PAFI hyperplane family), the reference profile is this line integral evaluated at each path station. The reference `dF/dr` used in comparisons is the central difference of the quadrature profile over a step of 1e-4 in `r`, because the spacing of the stations is too wide for a difference of the profile itself.

A sampled plane is held in the basin of the path by `PafiConfig.max_jump`. On the Müller-Brown surface a line a few units long also crosses the other, deeper wells, which no walk in the plane of the path visits. The reference there integrates only the span of the line that drains to the in-line minimum nearest the path: down to that minimum, then up to the first local maximum on each side.

This matches the PAFI estimator of Swinburne and Marinica (Phys. Rev. Lett. 120, 135503, 2018, https://doi.org/10.1103/PhysRevLett.120.135503).


# Exact TST rate on a line or hyperplane

The tests use the mass-weighted OH-TST rate form

`k_TST = sqrt(kT / 2π) * Z_H / Z_R`,

with:

-   `Z_H` from the constrained line integral on the dividing hyperplane, and
-   `Z_R` from a half-space integral behind that hyperplane.

For 2D systems this is evaluated by nested quadrature: an outer integral over signed distance along the hyperplane normal, and one constrained line integral per outer node.

The optimal hyperplane reference is obtained by maximizing the line free energy with respect to line shift and angle (Newton iterations on finite-difference gradients/Hessian of the quadrature objective), following the OH-TST variational idea of Jóhannesson and Jónsson (J. Chem. Phys. 115, 9644, 2001, https://doi.org/10.1063/1.1415499) and Bligaard and Jónsson (Comput. Phys. Commun. 169, 284, 2005, https://doi.org/10.1016/j.cpc.2005.03.064).


# Overdamped reference rate (MFPT / Kramers limit)

For the 1D double well, the overdamped Langevin reference rate is computed from the mean first-passage time equation (Smoluchowski dynamics), with a reflecting boundary in the reactant well and an absorbing boundary at the dividing surface:

`k_MFPT = 1 / τ`,

`τ(x0) = (1 / μkT) ∫_{x0}^{x‡} dy exp(V(y)/kT) ∫_{xL}^{y} dz exp(-V(z)/kT)`.

The inner cumulative integral is tabulated once on a fine grid; the outer integral is then evaluated by adaptive quadrature. The time to reach the dividing surface is half of the time to cross it: a trajectory on the surface returns to the reactant as often as it commits to the product, so the reactive rate is `1 / (2τ)` and the transmission coefficient compared with the sampled one is `κ = 1 / (2 τ k_TST)`. For `kT = 0.05` and a friction of 14 this is 0.141, against 0.140 from the Kramers formula for a barrier frequency of 2. This provides the exact overdamped benchmark used for transmission/rate consistency checks, in the spirit of the Kramers-to-diffusion discussion summarized by Hänggi, Talkner and Borkovec (Rev. Mod. Phys. 62, 251, 1990, https://doi.org/10.1103/RevModPhys.62.251).


# Straight chord versus optimal hyperplane

The curved tilted-well case intentionally separates two questions:

1.  Does PAFI recover the free-energy profile of the **chosen** hyperplane family (straight chord planes)?
2.  Is that family variationally optimal for rate estimation?

The chord family is validated against its own quadrature profile. Then the OH-TST optimizer is validated against the quadrature optimum over all nearby lines. The tests assert that the optimized-line rate agrees with quadrature within statistical uncertainty while the straight-chord line remains measurably biased.


# Harmonic limit and Vineyard scaling

At low temperature, exact quadrature rates are compared to harmonic HTST:

`ln k_HTST = ln( sqrt(det H_R / det' H_‡) / (2π) ) - ΔV / kT`.

For the tested 2D systems this is evaluated from reactant and saddle Hessians (one unstable saddle mode removed). The tests check that `|ln k_exact - ln k_HTST|` decreases monotonically as `kT` is lowered, consistent with the Vineyard harmonic limit (J. Phys. Chem. Solids 3, 121, 1957, https://doi.org/10.1016/0022-3697(57)90059-8).


# Step sizes on a stiff surface

The explicit overdamped step of PAFI is stable for `μ dt λ < 2`, with `λ` the stiffest curvature in the plane, and the BAOAB step of the plane sampler for `h ω < 2`. The Müller-Brown surface has `λ = 4068` at its deepest minimum, so the steps that suit the double well (`λ = 8`) make the walk leave the basin or diverge. The Müller-Brown cases use `dt = 2e-4` for PAFI and `h = 4e-3` for the plane sampler.


# Running the ignored cases

The cases that take longer than a few seconds are ignored, and the file needs the `ohtst` feature:

`cargo test --release --features ohtst --test ohtst_pafi_exact -- --ignored`


# Uncertainty policy in assertions

All sampler-vs-reference checks use uncertainty-aware tolerances:

-   sampled mean-force/profile quantities are compared with a multiple of their own autocorrelation-aware standard errors, and
-   deterministic discretization terms (path spacing, finite quadrature resolution, high-friction remainder) are added explicitly when needed.

No fixed absolute `1e-N` threshold is used as the main pass criterion.
