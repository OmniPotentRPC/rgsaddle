# Free dumbbell tracer diffusivity, Mishin Cu

One <100> dumbbell in the perfect 13×13×13 cell (8789 atoms) on the Mishin EAM1 of `tests/mishin_cu.rs`: Maxwellian start, 30 ps of isotropic zero-pressure NPH (Andersen piston, period 2.0 ps), then NVE in the mean box of the second half of that stage, both rescaled to the target temperature every 0.10 ps, timestep 10 fs. The a column is the production lattice constant. Every 0.1 ps the centre-of-mass drift is removed and every atom is assigned its nearest ideal-lattice site. The defect is located three ways: the one doubly occupied Wigner-Seitz cell (as in `direct_md_sia_d.md`), the midpoint of the two atoms in it, and the one of those two farther from the site. Each atom's site is unwrapped from sample to sample, and every atom's position less the drift is stored at the start of production and at the end of each block of τ.

Two diffusivities are measured from the same trajectories. The defect diffusivity is the mean-square displacement of the defect over 6t. The tracer quantity is Σ_i |Δr_i|² over 6t, summed over every atom i; this is N D*, with D* the per-atom self-diffusivity that Eq. (5) of Ref. [12] (B. Zhang et al., Phys. Rev. Materials 6, 053605 (2022)) computes, G_MSD = (1/N) Σ_i Δx_i Δx_j with 2Dt = G_MSD at long times. Its erratum (Phys. Rev. Materials 9, 069901(E) (2025)) multiplies that D* by N = 4951 to report an SIA diffusivity, and Table 1 of Mwungeli, Liu, and Demkowicz lists those values. N D* is measured from atom sites (one atom changes site by λ = a/√2 per dumbbell jump, so the vibrational term is absent) and from the drift-removed positions. Every displacement is in the frame of the centre of mass.

Block values are the lag-τ mean over every block. Slope values fit the mean over all origins against lag, with an intercept, from lag L/8 to L/2 of the L blocks. The Ref. [12] column follows that paper: one origin, the start of production; each axis fitted as 2Dt with an intercept over every lag; the three axes averaged. f is N D* over the block D of the Wigner-Seitz site, each the mean over trajectories, with both standard errors in quadrature: f takes the sites slope and f (Ref. [12]) the Ref. [12] fit. The block D of the defect equals its slope when successive defect jumps are uncorrelated beyond τ, which the slope column tests; the block N D* lies above the slope because a block of n jumps has the lattice-walk f(n) of `direct_md_dynamics.md`, about 0.8/n above its limit. The jump column sums |Δr|² of the Wigner-Seitz site from sample to sample over 6t: the jump rate times λ²/6 when every move is one nearest-neighbour jump. Every value is the mean over trajectories ± the standard error across them.

## Defect position

| T (K) | a (Å) | seeds | blocks | τ (ps) | site changes/ps | D, Wigner-Seitz (Å²/ps) | D, pair midpoint | D, farther atom | D, jump sum | D, Wigner-Seitz slope |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 300 | 3.632602 | 8 | 40 | 40.0 | 0.365 | 0.3198 ± 0.0143 | 0.3185 ± 0.0142 | 0.3217 ± 0.0150 | 0.4021 ± 0.0038 | 0.1959 ± 0.0383 |
| 500 | 3.645083 | 8 | 40 | 20.0 | 1.232 | 1.2415 ± 0.0698 | 1.2413 ± 0.0681 | 1.2639 ± 0.0618 | 1.3653 ± 0.0207 | 1.0912 ± 0.2247 |
| 700 | 3.658765 | 8 | 40 | 10.0 | 2.009 | 2.1380 ± 0.1123 | 2.1359 ± 0.1122 | 2.1890 ± 0.1112 | 2.2488 ± 0.0336 | 3.1112 ± 0.9489 |

## Tracer

| T (K) | N D*, sites, block (Å²/ps) | N D*, sites, slope | N D*, positions, slope | N D*, Ref. [12] fit | Table 1 D_i [12] | Ref. [12] fit / Table 1 | z | f | f (Ref. [12]) | sum-rule breaks | unresolved samples |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 300 | 0.1739 ± 0.0035 | 0.1514 ± 0.0058 | 0.1513 ± 0.0059 | 0.1511 ± 0.0050 | 0.143 ± 0.006 | 1.056 ± 0.056 | 1.04 | 0.4735 ± 0.0280 | 0.4725 ± 0.0262 | 0 | 0 |
| 500 | 0.6008 ± 0.0125 | 0.5692 ± 0.0115 | 0.5691 ± 0.0112 | 0.5705 ± 0.0127 | 0.526 ± 0.006 | 1.085 ± 0.027 | 3.17 | 0.4585 ± 0.0274 | 0.4595 ± 0.0278 | 0 | 0 |
| 700 | 0.9998 ± 0.0260 | 0.9387 ± 0.0294 | 0.9400 ± 0.0301 | 0.9382 ± 0.0245 | 0.910 ± 0.012 | 1.031 ± 0.030 | 1.03 | 0.4390 ± 0.0269 | 0.4388 ± 0.0257 | 0 | 0 |

## Lag dependence

Means over trajectories of the multi-origin mean at each lag t, over 6t.

| T (K) | t (ps) | origins | site changes | D, Wigner-Seitz (Å²/ps) | N D*, sites | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 300 | 40 | 40 | 14.6 | 0.3198 | 0.1739 | 0.544 |
| 300 | 80 | 39 | 29.3 | 0.3204 | 0.1627 | 0.508 |
| 300 | 160 | 37 | 58.4 | 0.3129 | 0.1564 | 0.500 |
| 300 | 320 | 33 | 116.5 | 0.2895 | 0.1536 | 0.530 |
| 300 | 640 | 25 | 232.7 | 0.2454 | 0.1528 | 0.623 |
| 300 | 800 | 21 | 290.4 | 0.2212 | 0.1520 | 0.687 |
| 300 | 1280 | 9 | 466.8 | 0.1841 | 0.1503 | 0.817 |
| 300 | 1600 | 1 | 584.8 | 0.2777 | 0.1493 | 0.538 |
| 500 | 20 | 40 | 24.6 | 1.2415 | 0.6008 | 0.484 |
| 500 | 40 | 39 | 49.3 | 1.2331 | 0.5839 | 0.473 |
| 500 | 80 | 37 | 98.8 | 1.2354 | 0.5748 | 0.465 |
| 500 | 160 | 33 | 198.0 | 1.2980 | 0.5698 | 0.439 |
| 500 | 320 | 25 | 397.0 | 1.2324 | 0.5685 | 0.461 |
| 500 | 400 | 21 | 497.3 | 1.1126 | 0.5710 | 0.513 |
| 500 | 640 | 9 | 791.9 | 0.8222 | 0.5686 | 0.692 |
| 500 | 800 | 1 | 985.9 | 0.9000 | 0.5687 | 0.632 |
| 700 | 10 | 40 | 20.1 | 2.1380 | 0.9998 | 0.468 |
| 700 | 20 | 39 | 40.2 | 2.2400 | 0.9629 | 0.430 |
| 700 | 40 | 37 | 80.5 | 2.4558 | 0.9501 | 0.387 |
| 700 | 80 | 33 | 160.6 | 2.7581 | 0.9419 | 0.342 |
| 700 | 160 | 25 | 320.3 | 2.8664 | 0.9408 | 0.328 |
| 700 | 200 | 21 | 400.8 | 3.0286 | 0.9407 | 0.311 |
| 700 | 320 | 9 | 642.5 | 3.4846 | 0.9405 | 0.270 |
| 700 | 400 | 1 | 803.6 | 3.9106 | 0.9343 | 0.239 |

Raw lag statistics per trajectory are the CSV this note was written from.
