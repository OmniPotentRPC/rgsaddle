# Free dumbbell tracer diffusivity, Mishin Cu

One <100> dumbbell in the perfect 13×13×13 cell (8789 atoms) on the Mishin EAM1 of `tests/mishin_cu.rs`: Maxwellian start, 2 ps of Langevin at γ = 1/ps, then Langevin at γ = 0.1/ps, timestep 10 fs. The a column is the production lattice constant. Every 0.1 ps the centre-of-mass drift is removed and every atom is assigned its nearest ideal-lattice site. The defect is located three ways: the one doubly occupied Wigner-Seitz cell (as in `direct_md_sia_d.md`), the midpoint of the two atoms in it, and the one of those two farther from the site. Each atom's site is unwrapped from sample to sample, and every atom's position less the drift is stored at the start of production and at the end of each block of τ.

Two diffusivities are measured from the same trajectories. The defect diffusivity is the mean-square displacement of the defect over 6t. The tracer quantity is Σ_i |Δr_i|² over 6t, summed over every atom i; this is N D*, with D* the per-atom self-diffusivity that Eq. (5) of Ref. [12] (B. Zhang et al., Phys. Rev. Materials 6, 053605 (2022)) computes, G_MSD = (1/N) Σ_i Δx_i Δx_j with 2Dt = G_MSD at long times. Its erratum (Phys. Rev. Materials 9, 069901(E) (2025)) multiplies that D* by N = 4951 to report an SIA diffusivity, and Table 1 of Mwungeli, Liu, and Demkowicz lists those values. N D* is measured from atom sites (one atom changes site by λ = a/√2 per dumbbell jump, so the vibrational term is absent) and from the drift-removed positions. Every displacement is in the frame of the centre of mass.

Block values are the lag-τ mean over every block. Slope values fit the mean over all origins against lag, with an intercept, from lag L/8 to L/2 of the L blocks. The Ref. [12] column follows that paper: one origin, the start of production; each axis fitted as 2Dt with an intercept over every lag; the three axes averaged. f is N D* over the block D of the Wigner-Seitz site, each the mean over trajectories, with both standard errors in quadrature: f takes the sites slope and f (Ref. [12]) the Ref. [12] fit. The block D of the defect equals its slope when successive defect jumps are uncorrelated beyond τ, which the slope column tests; the block N D* lies above the slope because a block of n jumps has the lattice-walk f(n) of `direct_md_dynamics.md`, about 0.8/n above its limit. The jump column sums |Δr|² of the Wigner-Seitz site from sample to sample over 6t: the jump rate times λ²/6 when every move is one nearest-neighbour jump. Every value is the mean over trajectories ± the standard error across them.

## Defect position

| T (K) | a (Å) | seeds | blocks | τ (ps) | site changes/ps | D, Wigner-Seitz (Å²/ps) | D, pair midpoint | D, farther atom | D, jump sum | D, Wigner-Seitz slope |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 300 | 3.631851 | 8 | 40 | 40.0 | 0.340 | 0.2717 ± 0.0157 | 0.2712 ± 0.0159 | 0.2879 ± 0.0172 | 0.3736 ± 0.0086 | 0.2730 ± 0.0443 |
| 500 | 3.644035 | 8 | 40 | 20.0 | 1.171 | 1.0590 ± 0.0449 | 1.0561 ± 0.0460 | 1.0680 ± 0.0471 | 1.2969 ± 0.0114 | 1.0111 ± 0.3823 |
| 700 | 3.657362 | 8 | 40 | 10.0 | 1.931 | 2.0246 ± 0.1230 | 2.0162 ± 0.1233 | 2.0219 ± 0.1183 | 2.1615 ± 0.0279 | 1.3506 ± 0.2023 |

## Tracer

| T (K) | N D*, sites, block (Å²/ps) | N D*, sites, slope | N D*, positions, slope | N D*, Ref. [12] fit | Table 1 D_i [12] | Ref. [12] fit / Table 1 | z | f | f (Ref. [12]) | sum-rule breaks | unresolved samples |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 300 | 0.1597 ± 0.0032 | 0.1372 ± 0.0052 | 0.1373 ± 0.0053 | 0.1392 ± 0.0042 | 0.143 ± 0.006 | 0.973 ± 0.050 | -0.52 | 0.5050 ± 0.0349 | 0.5122 ± 0.0334 | 0 | 0 |
| 500 | 0.5647 ± 0.0087 | 0.5325 ± 0.0134 | 0.5325 ± 0.0134 | 0.5342 ± 0.0097 | 0.526 ± 0.006 | 1.016 ± 0.022 | 0.72 | 0.5028 ± 0.0248 | 0.5044 ± 0.0233 | 0 | 0 |
| 700 | 1.0032 ± 0.0154 | 0.9332 ± 0.0214 | 0.9330 ± 0.0211 | 0.9412 ± 0.0195 | 0.910 ± 0.012 | 1.034 ± 0.025 | 1.36 | 0.4609 ± 0.0299 | 0.4649 ± 0.0298 | 0 | 0 |

## Lag dependence

Means over trajectories of the multi-origin mean at each lag t, over 6t.

| T (K) | t (ps) | origins | site changes | D, Wigner-Seitz (Å²/ps) | N D*, sites | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 300 | 40 | 40 | 13.6 | 0.2717 | 0.1597 | 0.588 |
| 300 | 80 | 39 | 27.2 | 0.2623 | 0.1488 | 0.567 |
| 300 | 160 | 37 | 54.3 | 0.2575 | 0.1447 | 0.562 |
| 300 | 320 | 33 | 108.8 | 0.2725 | 0.1425 | 0.523 |
| 300 | 640 | 25 | 217.7 | 0.2787 | 0.1401 | 0.503 |
| 300 | 800 | 21 | 272.2 | 0.2633 | 0.1387 | 0.527 |
| 300 | 1280 | 9 | 435.2 | 0.2410 | 0.1409 | 0.585 |
| 300 | 1600 | 1 | 543.8 | 0.2336 | 0.1396 | 0.598 |
| 500 | 20 | 40 | 23.4 | 1.0590 | 0.5647 | 0.533 |
| 500 | 40 | 39 | 46.8 | 1.0779 | 0.5505 | 0.511 |
| 500 | 80 | 37 | 93.7 | 1.0318 | 0.5429 | 0.526 |
| 500 | 160 | 33 | 187.4 | 1.0028 | 0.5377 | 0.536 |
| 500 | 320 | 25 | 377.4 | 0.9911 | 0.5348 | 0.540 |
| 500 | 400 | 21 | 473.2 | 1.0207 | 0.5346 | 0.524 |
| 500 | 640 | 9 | 750.3 | 0.8242 | 0.5343 | 0.648 |
| 500 | 800 | 1 | 937.0 | 0.6854 | 0.5343 | 0.780 |
| 700 | 10 | 40 | 19.3 | 2.0246 | 1.0032 | 0.496 |
| 700 | 20 | 39 | 38.7 | 2.0624 | 0.9748 | 0.473 |
| 700 | 40 | 37 | 77.5 | 2.0959 | 0.9582 | 0.457 |
| 700 | 80 | 33 | 154.7 | 1.8863 | 0.9461 | 0.502 |
| 700 | 160 | 25 | 310.6 | 1.6184 | 0.9399 | 0.581 |
| 700 | 200 | 21 | 389.3 | 1.5248 | 0.9383 | 0.615 |
| 700 | 320 | 9 | 618.8 | 1.2365 | 0.9413 | 0.761 |
| 700 | 400 | 1 | 772.5 | 1.3467 | 0.9430 | 0.700 |

Raw lag statistics per trajectory are the CSV this note was written from.
