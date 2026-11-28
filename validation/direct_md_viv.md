# Direct-MD vacancy-SIA volume, Mishin Cu

Independent NVT trajectories of one vacancy and one <100> dumbbell on the Mishin EAM1 evaluated by `tests/mishin_cu.rs`. The driver is `examples/langevin_nvt.rs`. The cell is 13×13×13 conventional cells (8788 atoms) at the branch's NPT lattice constant. Capture is the paper's 12-fold test: coordination cutoff 3.1 Å, sampled every 0.25 ps (Mwungeli, Liu, and Demkowicz, Phys. Rev. Materials 9, 023601 (2025), first method). The volume is not their survival-curve slope. It is the shell sum used by `kinetic_recombination_volume`: each orientation owns one third of the shell, and a shell enters only when all three <100> orientations were run.

The clock starts at the ideal dumbbell (offset 1.025 Å) with Maxwellian velocities. There is no separate NPH equilibration. The thermostat is the Langevin velocity Verlet in `npt_lattice`, without the barostat. Friction and timestep are the columns `gamma_per_ps` and `dt_fs`.

A 3 ps control at 500 K, 2 fs, and the same friction, sampled every 0.25 ps, fixes what a zero in the 12-fold count means. The perfect 13³ crystal started at 0 atoms off 12-fold coordination and then fluctuated: minimum 0, mean 6.2, maximum 14, with 4 of 12 samples exactly 0. The nearest shell, a <100> dumbbell, started at 13 and fell into that same cloud (minimum 0, mean 5.7, maximum 16, 2 of 12 samples exactly 0). The outermost shell inside 10.8 Å started at 20 and stayed separated (minimum 22, mean 32.0, maximum 47, 0 of 12 samples exactly 0). Capture in the table is the first sample whose off-12 count is 0.

Binomial standard error of a capture probability is sqrt(p (1-p) / N). The volume error adds the orientation contributions in quadrature. A shell whose trajectories all agree has a binomial error of zero.

## 500 K, window 6.00 ps

| shell (Å) | orientation | sites | captured | N | p | binomial σ |
|---:|:---:|---:|---:|---:|---:|---:|
| 2.5729 | 001 | 12 | 6 | 8 | 0.7500 | 0.1531 |
| 2.5729 | 010 | 12 | 8 | 8 | 1.0000 | 0.0000 |
| 2.5729 | 100 | 12 | 8 | 8 | 1.0000 | 0.0000 |
| 3.6387 | 001 | 6 | 5 | 8 | 0.6250 | 0.1712 |
| 3.6387 | 010 | 6 | 6 | 8 | 0.7500 | 0.1531 |
| 3.6387 | 100 | 6 | 6 | 8 | 0.7500 | 0.1531 |
| 4.4564 | 001 | 24 | 6 | 8 | 0.7500 | 0.1531 |
| 4.4564 | 010 | 24 | 7 | 8 | 0.8750 | 0.1169 |
| 4.4564 | 100 | 24 | 4 | 8 | 0.5000 | 0.1768 |
| 5.1459 | 001 | 12 | 1 | 8 | 0.1250 | 0.1169 |
| 5.1459 | 010 | 12 | 8 | 8 | 1.0000 | 0.0000 |
| 5.1459 | 100 | 12 | 8 | 8 | 1.0000 | 0.0000 |
| 5.7532 | 001 | 24 | 0 | 8 | 0.0000 | 0.0000 |
| 5.7532 | 010 | 24 | 2 | 8 | 0.2500 | 0.1531 |
| 5.7532 | 100 | 24 | 0 | 8 | 0.0000 | 0.0000 |
| 6.3024 | 001 | 8 | 0 | 8 | 0.0000 | 0.0000 |
| 6.3024 | 010 | 8 | 0 | 8 | 0.0000 | 0.0000 |
| 6.3024 | 100 | 8 | 1 | 8 | 0.1250 | 0.1169 |
| 6.8073 | 001 | 48 | 2 | 8 | 0.2500 | 0.1531 |
| 6.8073 | 010 | 48 | 0 | 8 | 0.0000 | 0.0000 |
| 6.8073 | 100 | 48 | 3 | 8 | 0.3750 | 0.1712 |
| 7.2773 | 001 | 6 | 0 | 8 | 0.0000 | 0.0000 |
| 7.2773 | 010 | 6 | 1 | 8 | 0.1250 | 0.1169 |
| 7.2773 | 100 | 6 | 0 | 8 | 0.0000 | 0.0000 |
| 7.7188 | 001 | 36 | 0 | 8 | 0.0000 | 0.0000 |
| 7.7188 | 010 | 36 | 4 | 8 | 0.5000 | 0.1768 |
| 7.7188 | 100 | 36 | 4 | 8 | 0.5000 | 0.1768 |
| 8.1363 | 001 | 24 | 0 | 8 | 0.0000 | 0.0000 |
| 8.1363 | 010 | 24 | 0 | 8 | 0.0000 | 0.0000 |
| 8.1363 | 100 | 24 | 0 | 8 | 0.0000 | 0.0000 |
| 8.5334 | 001 | 24 | 0 | 8 | 0.0000 | 0.0000 |
| 8.5334 | 010 | 24 | 0 | 8 | 0.0000 | 0.0000 |
| 8.5334 | 100 | 24 | 0 | 8 | 0.0000 | 0.0000 |
| 8.9129 | 001 | 24 | 0 | 8 | 0.0000 | 0.0000 |
| 8.9129 | 010 | 24 | 1 | 8 | 0.1250 | 0.1169 |
| 8.9129 | 100 | 24 | 0 | 8 | 0.0000 | 0.0000 |
| 9.2768 | 001 | 72 | 0 | 8 | 0.0000 | 0.0000 |
| 9.2768 | 010 | 72 | 0 | 8 | 0.0000 | 0.0000 |
| 9.2768 | 100 | 72 | 0 | 8 | 0.0000 | 0.0000 |
| 9.9649 | 001 | 48 | 0 | 8 | 0.0000 | 0.0000 |
| 9.9649 | 010 | 48 | 0 | 8 | 0.0000 | 0.0000 |
| 9.9649 | 100 | 48 | 0 | 8 | 0.0000 | 0.0000 |
| 10.2917 | 001 | 12 | 0 | 8 | 0.0000 | 0.0000 |
| 10.2917 | 010 | 12 | 0 | 8 | 0.0000 | 0.0000 |
| 10.2917 | 100 | 12 | 0 | 8 | 0.0000 | 0.0000 |
| 10.6084 | 001 | 48 | 0 | 8 | 0.0000 | 0.0000 |
| 10.6084 | 010 | 48 | 0 | 8 | 0.0000 | 0.0000 |
| 10.6084 | 100 | 48 | 0 | 8 | 0.0000 | 0.0000 |

V_iv = 66.33 ± 5.50 Ω over 16 shells with all three orientations.

Commit c7688cf3 on feat/defect-pair-rates counts capture once, treats a nearest-neighbor hop onto the vacancy as capture, and builds the return layer from the sites one nearest-neighbor hop past the outer shell. Its first-passage volumes on the 8³ cell (2048 atoms), from benchmarks/defect_pair_passage.csv, are 185.4028 ± 5.0671 Ω at 300 K, 271.6158 ± 2.8552 Ω at 500 K, and 185.7878 ± 3.9133 Ω at 700 K. Table 1 of Mwungeli, Liu and Demkowicz, 8788 atoms at 500 K, is 185 ± 23 Ω (k-method) and 161 ± 15 Ω (mean-time method). Those volumes are long-time recombination volumes. This table counts captures inside the window above, on the 8788-atom cell.

Raw per-trajectory outcomes are `validation/direct_md_viv.csv`.
