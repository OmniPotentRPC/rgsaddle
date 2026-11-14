# Free `<100>` dumbbell diffusivity, Mishin Cu

Each row of `direct_md_sia_d.csv` is one independent replica. A replica is one `<100>` dumbbell in the perfect 13×13×13 cell (8789 atoms) at the lattice constant `measured_lattice` gives for that temperature. The integrator is the Langevin velocity Verlet of `npt_lattice` in `examples/langevin_nvt.rs` (`MD_MSD=1`). A replica starts with 2 ps at friction 1/ps, and then the production window runs at the friction and timestep in the row. The two protocols are 2 fs with 1/ps, the run-18 thermostat, and 10 fs with 0.1/ps, which produced the capture trajectories in `direct_md_long.csv`.

Tracking. Every 0.1 ps each atom is assigned to its nearest fcc site after the mean lattice drift is removed. The site holding two atoms is the interstitial. No sample in any replica had zero, two or more doubly occupied sites (`rejected` is 0 throughout). A move to a new site counts as a committed jump once it has been held for 0.3 ps. Two hops inside that time merge into one jump, so the mean squared committed jump is 1.02, 1.16 and 1.40 times a²/2 at 300, 500 and 700 K.

Estimators. `D_msd` is the slope of the multi-origin MSD, divided by 6, over lags from 5 to 20 ps. `D_jump` is the sum of squared committed jumps over 6t, the diffusivity the same jumps would give if they were uncorrelated. The quoted error on a mean is the standard error over replicas. The run-18 estimator used one end-to-end displacement per run, whose relative error is about 80% for a single replica. That estimator gave 0.136 at 300 K.

`benchmarks/defect_pair_run19_diffusivity.csv` is generated from this file by `direct_md_diffusivity_summary` in `tests/mishin_cu.rs`:

| T (K) | dt (fs) | γ (1/ps) | replicas | jumps | D_msd (Å²/ps) | D_jump (Å²/ps) | f = D_msd/D_jump | D_graph (Å²/ps) | D_msd/D_graph |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 300 | 2 | 1.0 | 8 | 453 | 0.269 ± 0.023 | 0.251 ± 0.006 | 1.07 ± 0.09 | 0.297 | 0.91 ± 0.08 |
| 500 | 2 | 1.0 | 16 | 1797 | 1.081 ± 0.077 | 0.956 ± 0.031 | 1.13 ± 0.09 | 1.276 | 0.85 ± 0.06 |
| 700 | 2 | 1.0 | 16 | 1677 | 1.752 ± 0.149 | 1.584 ± 0.044 | 1.11 ± 0.10 | 1.521 | 1.15 ± 0.10 |
| 300 | 10 | 0.1 | 8 | 376 | 0.278 ± 0.033 | 0.264 ± 0.019 | 1.06 ± 0.15 | 0.297 | 0.94 ± 0.11 |
| 500 | 10 | 0.1 | 8 | 1140 | 0.860 ± 0.071 | 0.919 ± 0.016 | 0.94 ± 0.08 | 1.276 | 0.67 ± 0.06 |
| 700 | 10 | 0.1 | 8 | 1763 | 1.886 ± 0.188 | 1.806 ± 0.066 | 1.04 ± 0.11 | 1.521 | 1.24 ± 0.12 |

`D_graph = (2/3) k a²` uses the one-channel PAFI translation rate of the rate model (`benchmarks/defect_pair_run13_volume.csv`). The correlation factor against the MD's own jumps is 1 within one or two standard errors at every temperature and in both protocols. The dumbbell's successive jumps are not correlated, as the centrosymmetric eight-channel argument says.

Table 1 of Mwungeli, Liu and Demkowicz, Phys. Rev. Materials 9, 023601 (2025), quotes 0.143 ± 0.006, 0.526 ± 0.006 and 0.910 ± 0.012 Å²/ps from their Ref. [12]. The 2 fs replicas here are 1.88, 2.05 and 1.93 times those values.
