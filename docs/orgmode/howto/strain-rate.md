

# What this is

`StrainResponse` is one migration event. It holds the zero-strain barrier, the elastic dipoles of the minimum and the saddle, an optional Vineyard prefactor and its strain slope, and the symmetry images of that hop. It does not store a strain grid. The strain at which a rate is needed is an argument of `barrier`.


# The dipole

Varvenne and Clouet, Phys. Rev. B 96, 224103 (2017), https://doi.org/10.1103/PhysRevB.96.224103, define the interaction of a point defect with an applied strain as `-P : ε` (their Eq. (5)). On a fixed cell their Eq. (8) is `P = -V σ`. The paragraph after that equation removes a nonzero homogeneous stress of the perfect cell. `ElasticDipole::from_stress` is that correction:

    P = -V (σ_defect - σ_bulk)

and the result is symmetrized, because the defect has no net torque (their Eq. (1)). `from_forces` is the same tensor as the first moment of a set of Kanzaki forces with zero resultant and zero torque. Any potential and any cell can supply the stress or the forces. Nothing in the library is specific to copper.

The barrier of one hop is the difference of the two interaction energies:

    ΔE(ε) = ΔE0 - (P_saddle - P_minimum) : ε

Dederichs and Schroeder, Phys. Rev. B 17, 2524 (1978), https://doi.org/10.1103/PhysRevB.17.2524, show that a homogeneous strain changes the jump rate through the saddle dipole relative to the equilibrium site. The activation volume `Ω = ∂ΔG/∂τ` is `activation_volume`.


# Higher moments

`ElasticMultipole::about` expands the atomic forces about one centre, the same single-centre idea as Wikfeldt, Batista, Vila, and Jónsson, Phys. Chem. Chem. Phys. 15, 16542 (2013), https://doi.org/10.1039/C3CP52097H. The dipole is the first moment. The quadrupole is `Q_ijk = Σ F_i X_j X_k` and the octupole is the next moment. `barrier_in` adds `-(1/2) (Q_saddle - Q_minimum)_ijk ∂_k ε_ij` when the caller supplies a strain gradient. A homogeneous `barrier` leaves that term out.


# Symmetry images

`with_point_group` rotates both dipoles by each supplied matrix, `P' = R P R^T`, and drops images that repeat a dipole. `cubic_rotations` is the 24 proper rotations of the cube. A caller that already knows the distinct hops can pass them to `with_variants` instead. `barrier` returns the image with the lowest barrier, and names it in `Estimate::variant`.


# Trust

`TrustPolicy` is the validity bound.

-   `radius` is a cap on the Frobenius norm of `ε`.
-   `quadratic` is a second-order error estimate, `quadratic * |ε|²`, compared with `error_tolerance` (0.01 eV by default, `LINEAR_FOLD_EV`).
-   `curvature` is the unstable-mode curvature `λ0` and its strain slope. The line is trusted while `λ(ε) = λ0 + slope : ε` stays negative and at least as negative as `fraction * λ0`.
-   `floor` is a barrier floor. A prediction at or below it needs a search, which is how a barrier that has collapsed is reported.

`Estimate::trust` is `Linear`, `Explicit` after a cached search, or `NeedsSearch` with the reason. `rate_hz` returns the Arrhenius rate only for `Linear` and `Explicit`.


# Refine

`refine` affine-strains a `SaddleSeed`, runs a fixed-cell `SolidDimer` with an index-1 check, and caches the explicit barrier. The surface is evaluated in Cartesian coordinates, so bind it to the strained cell first. `minimum_energy` is the reactant energy in that cell. A later `barrier` at the same strain returns `Trust::Explicit`.


# Serialization

`to_json` / `from_json` round-trip the event, including the policy, the images, and the explicit cache. The document name is `rgsaddle.strain_response`. An AKMC store can keep one document per event.


# The field of the other defects

The strain in `barrier` is the applied strain plus the strain produced by every other defect. `Kelvin` is the isotropic Green's function. `DirectSolver` sums it. `FmmSolver` walks an octree: neighboring leaves are direct, and a well-separated cell is translated into a local expansion, the multipole-to-local step of Greengard and Rokhlin, J. Comput. Phys. 73, 325 (1987), https://doi.org/10.1016/0021-9991(87)90140-9. `examples/elastic_field_bench.rs` times both on 100 to 100000 defects and writes `benchmarks/elastic_field_scaling.csv`.


# A sweep

`examples/akmc_strain.rs` places four copies of one hop along a shear that grows with the site coordinate, takes the trusted Arrhenius rate, and refines the sites past the radius.


# Copper check

`benchmarks/strain_response_cu.json` is the zero-strain dipole of the 109-atom Mishin copper `[100]` dumbbell and its `[110]` saddle, plus the quadratic error coefficient fitted at 0.5 percent shear. It is a validation record, not a rate table. It was recorded on rgmin 0.5.2, so its `barrier0_eV`, 0.473916239370908, is the zero-shear fixed-cell saddle of that run; the rgmin 0.6.0 rerun in `benchmarks/solid_dimer_cu.csv` is 0.473923553294469 eV. At `ε_yx = 0.005` the line is inside 0.01 eV. At `ε_yx = 0.02` the quadratic estimate is past 0.01 eV, so `barrier` returns `NeedsSearch`. The same barrier is returned for `ε_zx = 0.02` from the rotated image of the hop.


# It does not save calls at equal accuracy

`benchmarks/strain_response_head_to_head.csv` is a release run of the ignored test `strain_response_head_to_head`. The test system is the same 109-atom Mishin EAM1 dumbbell. The numbers below are that file. Nothing here is estimated.

The eOn values 0.47381310, 0.24354609, 0.14829202, 0.06786217 and 0.00779374 eV, and the variable-cell values 0.34700694, 0.16255572, 0.08805194 and 0.02792458 eV, are the xy simple shear of that dumbbell, not a vacancy hop. On TheochemUI/eOn branch `feat/solid-state-dimer` at e96b0881 the test is "copper dumbbell saddle follows the shear-aligned switch" in `client/unit_tests/SolidStateDimerTest.cpp`, and `docs/source/user_guide/solid_state.md` ("Copper interstitial") describes it. The cell map is `h = L * [[1, 0, 0], [γ, 1, 0], [0, 0, 1]]`, so `x' = x + γ y`, on a 3×3×3 cell with one <100> dumbbell (109 atoms). Fixed barriers are at 0, 2, 3, 4 and 5%. Variable-cell enthalpy barriers are at 0, 2, 3 and 4%. At 5% variable that test's reactant is a minimum and the index-1 point is not the shear-aligned <110> switch, so it is not a barrier. The full digits are that branch's `benchmarks/solid_dimer_cu.csv`.

The explicit fixed-cell search here, minus those eOn barriers, is +0.0001104559736973, +0.0000057825280919, -0.0000053800793580, -0.0000274446736626 and -0.0000208262501360 eV at 0, 2, 3, 4 and 5%. The variable-cell enthalpy minus the eOn enthalpy is -0.0000057668697195, +0.0002091265311606, +0.0000605272978760 and -0.0006349075244430 eV at 0, 2, 3 and 4%. At 5% variable this search is also not the switch (`explicit_is_barrier` 0, reactant lowest eigenvalue +0.0181147653518677). The enthalpy it did stop on is 0.9152325828545145 eV, which is a different index-1 point from eOn's and is not the barrier either. These rows are the rgmin 0.6.0 run with the 0.004 Å dimer cap of `howto/solid-dimer`.

`StrainResponse::barrier` with the quadratic bound fitted at 0.5% xy shear (coefficient 70.3686534924141682, tolerance 0.01 eV) does not reproduce those barriers once the line leaves that neighborhood, and the refine hook does not repair it. The one-time build is 2975 force calls: 2 for the bulk stress, 1494 for the zero-strain saddle, 1479 for the 0.5% calibration search. A force multipole about the relaxed atoms was refused, because the residual forces are not torque-free, so the build is the stress dipole only. Both rates use one prefactor. The relative rate error is `exp(-(E_method - E_explicit) / kT) - 1` with `k` = `BOLTZMANN_EV_PER_K`. The prefactor cancels. The response object carries 1 Hz only so a rate can be formed.

On the 25 fixed-cell points (xy, xz, uniaxial x, hydrostatic, at 0, 0.5, 1, 2, 3, 4 and 5%, with 0% stored once) the explicit searches cost 40627 calls. The method costs 59338, the build plus every refine attempt. Eight of the seventeen points that `NeedsSearch` were still `Running` after 2500 steps, so the reported barrier is the dipole line. Nine refine calls did stop on an index-1 cache entry. Three of those match the explicit barrier: hydrostatic 2%, 4% and 5%, to -0.0000430058427696, -0.0002751516212811 and -0.0002487851893989 eV, for 3376, 2346 and 2365 calls against explicit searches of 1058, 1527 and 2248. The other six do not match. xy 2% and xz 2% return 0.0658510203606397 and 0.0658510203593892 eV against 0.2435518720826053 and 0.2435622382795373 eV. xy 4% and xz 4% are low by 0.0417736394587109 and 0.0417755757029568 eV. xy 5% and xz 5% return 0.0974351674951208 and 0.0974351674948934 eV against explicit barriers of 0.0077729111648068 and 0.0077711936276046 eV.

Where the line is trusted and no refine runs, the largest absolute barrier error outside the calibration point is 0.0067642793531877 eV, on xy at 1%. The relative rate error there is 0.2990769941275599 at 300 K and 0.1397705883762574 at 600 K. A barrier inside the 0.01 eV fold is not the same rate.

The 200-event batch is the field in `examples/akmc_strain.rs`: four sites, a saddle dipole with `P_xy = 8`, Kelvin modulus 40 and Poisson ratio 0.3, spacing 6, plus `Strain::shear_yx(0.01)`, then symmetrized. All four stay on the line, so the queries add 0 calls. With the build, the method costs 2975. The four sites are two strains. One explicit search at each costs 1612 and 1600 calls, sum 3212. Fifty events at each site, each event searching again, cost 321200. The barrier errors are -0.0069949937062344 eV and -0.0071769982823058 eV. The 300 K relative rate errors are 0.3107224023749027 and 0.3199827769541861. The 600 K ones are 0.1448678536734722 and 0.1489050339145468. Repeating a trusted query skips the search. It does not return the explicit rate. One search at each of the two strains costs 3212 calls, against 2975 for the build.

Variable-cell rows are the two shears, including 0%. Uniaxial and hydrostatic are imposed strains, so only the fixed cell is the ensemble the dipole line claims. The 13 variable searches cost 20439 calls. The method does not answer those enthalpies.
