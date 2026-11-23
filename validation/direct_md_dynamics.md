# Table 1 D_i and the free-dumbbell tracer, Mishin Cu

Our free-dumbbell defect D (`direct_md_sia_d.md`) is about twice the Table 1 D_i of Mwungeli, Liu, and Demkowicz, Phys. Rev. Materials 9, 023601 (2025), which they quote from Ref. [12], B. Zhang et al., Phys. Rev. Materials 6, 053605 (2022). The first section shows from both papers and from tracer trajectories that the two are different quantities; the second repeats one campaign at a shorter timestep. The tracer campaigns ran at zero total pressure, at the lattice constants of their notes rather than those of `measured_lattice`, under Langevin with γ = 0.1/ps and under the paper's protocol (NPH for 30 ps, then NVE with velocity rescaling). Every campaign uses the same potential, 13³ cell, timestep, seeds, and blocks, except the shorter-timestep repeat of the Timestep section, which has half the blocks. `MD_SIA_TRACER=1` in `examples/langevin_nvt.rs` takes the same measurements on the `MD_MSD` replicas, and `MD_SIA_TRACER_NOTE=1` rewrites this note and the tracer notes from their CSVs. Each number below is read from the CSV named in its row.

## What the Table 1 D_i is

Page numbers are those of the accepted manuscripts on OSTI: Mwungeli, Liu, and Demkowicz, Phys. Rev. Materials 9, 023601 (2025), OSTI 2513909; B. Zhang et al., Phys. Rev. Materials 6, 053605 (2022), their Ref. [12], OSTI 1870695.

- Section 1 (p. 2): "The recombination rate coefficient K_iv may be written as K_iv = V_iv D_i/(αa²), where V_iv is the vacancy-SIA recombination volume, D_i is SIA diffusivity, a is the cubic lattice parameter, and α is a numerical factor that depends on the SIA migration mechanism (α = 2/3 for [100] split dumbbell SIAs in Cu [10]). The quotient D_i/(αa²) is just the SIA migration frequency, ν."
- Section 2 (p. 5): "we calculate the corresponding recombination volume as V_iv = K_iv αa²/D_i. The values of SIA diffusivity D_i for the temperatures of interest are taken from Ref. [12], which obtained D_i directly from mean squared displacements of SIAs in MD simulations". With Ω = a³/4 this is V_iv/Ω = 4αK_iv/(aD_i); α multiplies K_iv.
- Section 5 (p. 10), after Eq. (4), K_iv = (1/3)Ω Σ_j ν_j over the Johnson jump paths that lead to recombination: "If we further assume that all the migration rates ν_j equal ν—the rate of SIA migration in the interior of a perfect crystal, far from any other defects—then, from the relation K_iv = V_iv D_i/(αa²) = V_iv ν (see section 1), we get V_iv = (1/3)Ωs". ν is the rate of one jump path of the defect, and D_i = (2/3)a²ν is the diffusivity of the defect for eight uncorrelated jumps of length a/√2; section 6 (p. 15) writes the same D_i = (2/3)a²ν₀^i e^(−ΔE_i/k_BT) for TST.
- Ref. [12], Eq. (5) (p. 4): the "generalized mean square displacement (MSD) per atom matrix" G_MSD = (1/N) Σ_i [Δx_i², Δx_iΔy_i, …], with "displacements of individual atoms from their initial positions", "2Dt = G_MSD in the limit of a long time", and "linear least-squares-fits to the corresponding elements of G_MSD over time" after the first 1 ps. The cell has 4951 atoms with the SIA (p. 3); the volume follows thermal expansion and the runs are 2 ns of NVE (p. 4).
- Erratum, Phys. Rev. Materials 9, 069901(E) (2025), doi:10.1103/8hb2-86lm, replaces the p. 4 sentence that this D "is the SIA diffusivity" with "The diffusivity thereby obtained is the self-diffusivity of Cu, not the SIA diffusivity, because G_MSD is computed on a per-atom basis", and states that the diffusivities of Figs. 4 and 5(b) "may be converted to SIA diffusivities by multiplying them by the number of atoms in the model, i.e., 4951." Fig. 4 of Ref. [12] (p. 9) plots those diffusivities at 700 K in µm²/s; at zero strain the components read 1.84 µm²/s = 1.84×10⁻⁴ Å²/ps, and 4951 times that is 0.911 Å²/ps, the 700 K D_i of Table 1 (0.910 ± 0.012).
- Potential: section 3 (p. 6), "All atomistic simulations are carried out using the embedded atom method (EAM) potential for Cu by Mishin et al. [32]"; Ref. [12] (p. 3), "an embedded atom method (EAM) [24] potential for Cu developed by Mishin et al. [25]". Both references are Y. Mishin, M. J. Mehl, D. A. Papaconstantopoulos, A. F. Voter, and J. D. Kress, Phys. Rev. B 63, 224106 (2001), the EAM1 of `tests/mishin_cu.rs` that every campaign here uses.

Table 1 D_i is therefore 4951 D*, with D* the per-atom diffusivity of Eq. (5): N D* = Σ_i ⟨Δx_i²⟩/2t, a sum of tracer displacements. One Johnson jump moves one atom by the defect's own jump vector, so N D* would equal the defect diffusivity only if successive jumps moved uncorrelated atoms. They do not: the atom that has just moved is one of the two atoms of the new dumbbell, so it moves again at the next jump with probability 1/2, and that move points back along its last one on average. N D* = f D_defect defines the tracer correlation factor f of the mechanism, while the ν of section 1 is the defect's own jump rate, with (2/3)a²ν = D_defect when successive defect jumps are uncorrelated. Both are measured below on the same free-dumbbell trajectories (`direct_md_sia_tracer_*.md`). N D* is the Ref. [12] fit of drift-removed positions; defect D is the Wigner-Seitz block value of the same trajectories; f is N D* over defect D, the ratio of the two means. Ratios carry both errors in quadrature, and z is the difference over the two errors in quadrature.

| T (K) | dynamics | CSV | seeds | N D*, Ref. [12] fit (Å²/ps) | Table 1 D_i [12] | N D* / Table 1 | z | defect D (Å²/ps) | defect D / Table 1 | f |
|---:|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| 300 | Langevin, γ = 0.1/ps, zero total pressure | `direct_md_sia_tracer_p0_gamma0.1_dt10fs.csv` | 8 | 0.1392 ± 0.0042 | 0.143 ± 0.006 | 0.973 ± 0.050 | -0.52 | 0.2717 ± 0.0157 | 1.900 ± 0.136 | 0.5122 ± 0.0334 |
| 300 | NPH then NVE, rescaled | `direct_md_sia_tracer_nph_nve_dt10fs.csv` | 8 | 0.1511 ± 0.0050 | 0.143 ± 0.006 | 1.056 ± 0.056 | 1.04 | 0.3198 ± 0.0143 | 2.236 ± 0.137 | 0.4725 ± 0.0262 |
| 500 | Langevin, γ = 0.1/ps, zero total pressure | `direct_md_sia_tracer_p0_gamma0.1_dt10fs.csv` | 8 | 0.5342 ± 0.0097 | 0.526 ± 0.006 | 1.016 ± 0.022 | 0.72 | 1.0590 ± 0.0449 | 2.013 ± 0.088 | 0.5044 ± 0.0233 |
| 500 | NPH then NVE, rescaled | `direct_md_sia_tracer_nph_nve_dt10fs.csv` | 8 | 0.5705 ± 0.0127 | 0.526 ± 0.006 | 1.085 ± 0.027 | 3.17 | 1.2415 ± 0.0698 | 2.360 ± 0.135 | 0.4595 ± 0.0278 |
| 700 | Langevin, γ = 0.1/ps, zero total pressure | `direct_md_sia_tracer_p0_gamma0.1_dt10fs.csv` | 8 | 0.9412 ± 0.0195 | 0.910 ± 0.012 | 1.034 ± 0.025 | 1.36 | 2.0246 ± 0.1230 | 2.225 ± 0.138 | 0.4649 ± 0.0298 |
| 700 | NPH then NVE, rescaled | `direct_md_sia_tracer_nph_nve_dt10fs.csv` | 8 | 0.9382 ± 0.0245 | 0.910 ± 0.012 | 1.031 ± 0.030 | 1.03 | 2.1380 ± 0.1123 | 2.349 ± 0.127 | 0.4388 ± 0.0257 |

Across these rows N D* / Table 1 runs 0.973 to 1.085, defect D / Table 1 runs 1.900 to 2.360, and f runs 0.439 to 0.512.

The same correlation on an ideal lattice: a <100> dumbbell that moves only by Johnson jumps, the moving atom ending on the near side of the new dumbbell (`johnson_walk`). f(n) is Σ_i |Δs_i|² over n λ² after n jumps, mean ± standard error over the walks, seed 0x5EED. The lattice value is not an input to any number above.

| jumps n | walks | f(n) |
|---:|---:|---:|
| 10 | 40000 | 0.5175 ± 0.0010 |
| 100 | 10000 | 0.4480 ± 0.0008 |
| 1000 | 2000 | 0.4413 ± 0.0006 |
| 10000 | 400 | 0.4405 ± 0.0004 |
| 100000 | 100 | 0.4397 ± 0.0003 |

## Timestep

Every other campaign in this note integrates at 10 fs. `direct_md_sia_tracer_p0_gamma0.1_dt2fs.csv` repeats `direct_md_sia_tracer_p0_gamma0.1_dt10fs.csv` at 2 fs with the same seeds and τ and half the blocks. Neither Mwungeli, Liu, and Demkowicz nor Ref. [12] states an MD timestep.

| T (K) | quantity | 2 fs | 10 fs | ratio | z |
|---:|:---|---:|---:|---:|---:|
| 500 | site changes/ps | 1.1687 ± 0.0197 (8 seeds) | 1.1712 ± 0.0104 (8 seeds) | 0.998 ± 0.019 | -0.11 |
| 500 | D, jump sum | 1.2957 ± 0.0220 (8 seeds) | 1.2969 ± 0.0114 (8 seeds) | 0.999 ± 0.019 | -0.05 |
| 500 | defect D | 1.2999 ± 0.0670 (8 seeds) | 1.0590 ± 0.0449 (8 seeds) | 1.227 ± 0.082 | 2.98 |
| 500 | N D*, Ref. [12] fit | 0.5182 ± 0.0134 (8 seeds) | 0.5342 ± 0.0097 (8 seeds) | 0.970 ± 0.031 | -0.97 |
| 500 | f | 0.3986 ± 0.0230 (8 seeds) | 0.5044 ± 0.0233 (8 seeds) | 0.790 ± 0.058 | -3.23 |

At 500 K, defect D over the jump-sum D, which is 1 when successive defect jumps are uncorrelated, is 1.003 ± 0.054 at 2 fs and 0.817 ± 0.035 at 10 fs. N D* over the Table 1 D_i is 0.985 ± 0.028 (z = -0.53) at 2 fs and 1.016 ± 0.022 (z = 0.72) at 10 fs.
