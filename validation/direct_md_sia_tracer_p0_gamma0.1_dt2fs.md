# Free dumbbell tracer diffusivity, Mishin Cu

One <100> dumbbell in the perfect 13×13×13 cell (8789 atoms) on the Mishin EAM1 of `tests/mishin_cu.rs`: Maxwellian start, 2 ps of Langevin at γ = 1/ps, then Langevin at γ = 0.1/ps, timestep 2 fs. The a column is the production lattice constant. Every 0.1 ps the centre-of-mass drift is removed and every atom is assigned its nearest ideal-lattice site. The defect is located three ways: the one doubly occupied Wigner-Seitz cell (as in `direct_md_sia_d.md`), the midpoint of the two atoms in it, and the one of those two farther from the site. Each atom's site is unwrapped from sample to sample, and every atom's position less the drift is stored at the start of production and at the end of each block of τ.

Two diffusivities are measured from the same trajectories. The defect diffusivity is the mean-square displacement of the defect over 6t. The tracer quantity is Σ_i |Δr_i|² over 6t, summed over every atom i; this is N D*, with D* the per-atom self-diffusivity that Eq. (5) of Ref. [12] (B. Zhang et al., Phys. Rev. Materials 6, 053605 (2022)) computes, G_MSD = (1/N) Σ_i Δx_i Δx_j with 2Dt = G_MSD at long times. Its erratum (Phys. Rev. Materials 9, 069901(E) (2025)) multiplies that D* by N = 4951 to report an SIA diffusivity, and Table 1 of Mwungeli, Liu, and Demkowicz lists those values. N D* is measured from atom sites (one atom changes site by λ = a/√2 per dumbbell jump, so the vibrational term is absent) and from the drift-removed positions. Every displacement is in the frame of the centre of mass.

Block values are the lag-τ mean over every block. Slope values fit the mean over all origins against lag, with an intercept, from lag L/8 to L/2 of the L blocks. The Ref. [12] column follows that paper: one origin, the start of production; each axis fitted as 2Dt with an intercept over every lag; the three axes averaged. f is N D* over the block D of the Wigner-Seitz site, each the mean over trajectories, with both standard errors in quadrature: f takes the sites slope and f (Ref. [12]) the Ref. [12] fit. The block D of the defect equals its slope when successive defect jumps are uncorrelated beyond τ, which the slope column tests; the block N D* lies above the slope because a block of n jumps has the lattice-walk f(n) of `direct_md_dynamics.md`, about 0.8/n above its limit. The jump column sums |Δr|² of the Wigner-Seitz site from sample to sample over 6t: the jump rate times λ²/6 when every move is one nearest-neighbour jump. Every value is the mean over trajectories ± the standard error across them.

## Defect position

| T (K) | a (Å) | seeds | blocks | τ (ps) | site changes/ps | D, Wigner-Seitz (Å²/ps) | D, pair midpoint | D, farther atom | D, jump sum | D, Wigner-Seitz slope |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 3.644035 | 8 | 20 | 20.0 | 1.169 | 1.2999 ± 0.0670 | 1.2993 ± 0.0666 | 1.3273 ± 0.0642 | 1.2957 ± 0.0220 | 1.1859 ± 0.2750 |

## Tracer

| T (K) | N D*, sites, block (Å²/ps) | N D*, sites, slope | N D*, positions, slope | N D*, Ref. [12] fit | Table 1 D_i [12] | Ref. [12] fit / Table 1 | z | f | f (Ref. [12]) | sum-rule breaks | unresolved samples |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 0.5623 ± 0.0113 | 0.5182 ± 0.0159 | 0.5183 ± 0.0168 | 0.5182 ± 0.0134 | 0.526 ± 0.006 | 0.985 ± 0.028 | -0.53 | 0.3987 ± 0.0239 | 0.3986 ± 0.0230 | 0 | 0 |

## Lag dependence

Means over trajectories of the multi-origin mean at each lag t, over 6t.

| T (K) | t (ps) | origins | site changes | D, Wigner-Seitz (Å²/ps) | N D*, sites | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 20 | 20 | 23.4 | 1.2999 | 0.5623 | 0.433 |
| 500 | 40 | 19 | 46.7 | 1.2176 | 0.5391 | 0.443 |
| 500 | 80 | 17 | 93.5 | 1.2313 | 0.5230 | 0.425 |
| 500 | 160 | 13 | 186.0 | 1.2055 | 0.5210 | 0.432 |
| 500 | 200 | 11 | 232.5 | 1.1993 | 0.5228 | 0.436 |
| 500 | 320 | 5 | 373.4 | 1.4102 | 0.5177 | 0.367 |
| 500 | 400 | 1 | 467.5 | 1.5956 | 0.5177 | 0.324 |

Raw lag statistics per trajectory are the CSV this note was written from.
