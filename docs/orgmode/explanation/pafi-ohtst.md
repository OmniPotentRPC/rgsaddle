Every rate in this crate starts from the same transition-state-theory (TST) picture: pick a dividing surface between a reactant and a product, assume the reactant stays in quasi-equilibrium (Boltzmann-weighted) right up to that surface, and the one-way flux of that equilibrium ensemble through the surface is an upper bound on the true rate (the variational principle of TST). Any TST rate built this way can only overestimate the real one, because it counts every crossing as reactive; a trajectory that crosses and comes straight back is still counted once by the flux, so a separate, dynamical correction, the transmission coefficient `kappa <= 1`, removes that overcount. The three methods below differ only in which dividing surface they use and how they get `kappa`, not in this basic picture.


# Harmonic TST (Vineyard)

The cheapest dividing surface is implicit: approximate both wells and the barrier as quadratic and read the rate off their curvatures. `vineyard_prefactor` and `attempt_frequency` are this crate's pieces of it. With `lambda_min` the positive Cartesian Hessian eigenvalues at the minimum (translations removed) and `lambda_sad` those at the saddle (its one negative eigenvalue excluded), each is a set of normal-mode frequencies `nu_i = omega_i / (2 pi)` with `omega_i = sqrt(lambda_i / m)`, and the rate is

$$
k_{\text{HTST}} = \frac{\prod_i \nu_i^{\text{min}}}{\prod_i \nu_i^{\text{sad}}} \exp\!\left(-\frac{E_b}{kT}\right),
$$

Vineyard, J. Phys. Chem. Solids 3, 121–127 (1957), https://doi.org/10.1016/0022-3697(57)90059-8. `E_b` is the bare potential-energy barrier, with no finite-temperature correction: harmonic TST assumes the free-energy barrier is the harmonic approximation's own, so no sampling enters it at all, only Hessians at two points. `defect_pair_rates` (`howto/defect-pair-rates`) reports this rate beside its dynamically sampled one for exactly that reason, as a sanity check that costs two Hessians instead of a sampling campaign; the two agree where the wells really are quadratic and diverge where they are not.

Harmonic TST assumes classical statistics (no zero-point energy or tunnelling), that a single reaction path's local curvature stands in for the whole transverse free energy at every temperature, and `kappa = 1`, no recrossing correction. None of the three methods here tests the first assumption; PAFI and OH-TST relax the second and, through transmission shooting, the third.


# PAFI: a free energy along a fixed path

Projected average force integration (PAFI) keeps harmonic TST's idea of a single reaction path but replaces its curvature with a sampled, anharmonic, finite-temperature free energy along that path. With a path `X(s)`, the cubic spline through the frames over their chord lengths, tangent `n = X'`, `X''` its second derivative, and `psi = (x - X) . X'' / |n|^2` the path-curvature correction, the free-energy gradient averaged over the hyperplane normal to the path at `s` is

$$
\frac{dF}{ds} = \left\langle (\nabla E \cdot n)(1 - \psi) \right\rangle,
$$

Swinburne and Marinica, Phys. Rev. Lett. 120, 135503 (2018), https://doi.org/10.1103/PhysRevLett.120.135503: the same estimator LAMMPS' `fix pafi` averages (`PafiMeanForce::ReferenceCode`; `PafiMeanForce::PublishedEquation8` adds the paper's second term, `kT d/ds ln(|psi| / |X'|)`). `rgsaddle::pafi` samples that average by overdamped Langevin dynamics restricted to the plane, trapezoid-integrates `dF/ds` along the path to the potential of mean force `F(s)`, and reports its maximum as the barrier `Delta F`. On a straight path `psi = 0` exactly, since `X'' = 0`; a curved path needs the correction term to stay unbiased, and `tutorials/pafi` picks a surface with a straight valley floor for exactly that reason, to make the first run simple. The rate is harmonic TST's own attempt frequency, now against the sampled barrier instead of the bare one,

$$
k_{\text{PAFI}} = \nu \exp\!\left(-\frac{\Delta F(T)}{kT}\right).
$$

PAFI needs a path, not a point: the whole result is only as good as that path's agreement with the true valley floor, since the plane's own normal and curvature correction come from the path, not from the surface. `zero_temperature_barrier` checks a path against the potential's own zero-temperature barrier before spending any sampling budget on it, exactly the check `tutorials/pafi` and `howto/rgpot` both run first. The usual source of that path in this crate is its own climbing-image nudged elastic band, whose refinement follows Goswami, Gunde and Jónsson, Front. Chem. 14, 1807063 (2026), https://doi.org/10.3389/fchem.2026.1807063; `defect_pair_rates` (`howto/defect-pair-rates`) builds exactly that path internally before handing it to PAFI. `rgsaddle::pafi` reports no transmission coefficient of its own: a host that wants one samples it separately, the way `defect_pair_rates` adds the Bennett–Chandler `kappa` from `rgsaddle::bennett` on top of the PAFI barrier.


# OH-TST: the best possible hyperplane

Optimal hyperplanar transition state theory (OH-TST) keeps harmonic TST's dividing-surface picture, a single plane, but drops the fixed path: instead of committing to one reaction coordinate in advance, it moves the plane itself, by the normal and an offset along it, to the maximum of its own free energy, the hyperplane the variational principle of TST picks as tightest among every plane through that region. Jóhannesson and Jónsson, J. Chem. Phys. 115, 9644–9656 (2001), https://doi.org/10.1063/1.1415499, introduced the method; Bligaard and Jónsson, Comput. Phys. Commun. 169, 284–288 (2005), https://doi.org/10.1016/j.cpc.2005.03.064, worked it out on two-dimensional test surfaces where the free energy of a line is a single integral and the optimum can be checked by quadrature. In mass-weighted coordinates, `q = M^{1/2} x`, a plane is a point `p` and unit normal `nu`, its free energy is

$$
A(p, \nu) = -kT \ln \int \delta(\nu \cdot (q - p)) \, e^{-V(q)/kT} \, dq,
$$

and the bare rate, before any transmission correction, is

$$
k_{\text{OH}} = \sqrt{\frac{kT}{2\pi}} \, \frac{Z_H}{Z_R} \exp\!\left(-\frac{A^\ddagger - A_H}{kT}\right),
$$

with unit mass in these coordinates. `Z_H/Z_R` is the reactant normalization, `basin_density`'s mean-force integral over planes parallel to the optimal one; `A^\ddagger - A_H` is the reversible work `reversible_work_along` integrates between those two planes, by thermodynamic integration, BAR, or MBAR. Unlike PAFI, OH-TST samples a transmission coefficient as part of the same pipeline: `transmission_coefficient` shoots trajectories from the optimal plane's own retained configurations, and `OhTstRate::transmitted` applies `kappa` to `k_OH` directly. Assigning a shot's end state by the basin a quench reaches, rather than by side of the plane, needs a distance between atomic configurations that does not care which identical atom is which; `PermutationDistance`, the intensive permutation-invariant distance of Goswami and Jónsson, ChemPhysChem 27, e202500730 (2026), https://doi.org/10.1002/cphc.202500730, is that distance.

OH-TST's plane is still flat: it assumes a hyperplane is a fair stand-in for the true dividing surface (the isocommittor surface, where a trajectory is equally likely to relax to either side), which is only exact when that surface really is flat near the optimum, as it is on the reflection-symmetric surface `tutorials/oh-tst` runs first. A curved true dividing surface still gets a hyperplane fit to its best orientation and position, but `kappa` then has more work to do correcting for it, since every recrossing a flat plane cannot avoid shows up there instead. `PlanePath::curve` reuses PAFI's own plane integral, through a curve of planes along a path rather than one optimized plane, so a host that already has a trusted path and wants OH-TST's transmission-coefficient machinery on top of it, instead of a fixed attempt frequency, is not choosing between the two methods: `howto/oh-tst` shows that combination directly.


# Picking one

Harmonic TST costs two Hessians and no sampling: use it first, as a sanity check or when the wells are known to be close to quadratic and `kT` is small against the curvature scale (low temperature, stiff wells). It has no free-energy sampling and no recrossing correction, so it is also the one most easily and silently wrong on an anharmonic or floppy system.

PAFI costs one sampling campaign, parallel over planes, and needs a path already in hand: use it when a converged band or dimer search already gives a reaction path trusted well enough that `zero_temperature_barrier` on it reproduces the known zero-temperature barrier, and the thing wanted is that path's own finite-temperature free energy, possibly at several temperatures reusing the same path. It reports no transmission coefficient on its own.

OH-TST costs more, a plane optimization, a basin normalization and a transmission-coefficient shooting campaign, in exchange for not needing a trusted path at all: use it when the reaction coordinate itself is in question, when the variational bound a fixed path gives is suspect, or when the recrossing correction is wanted from the same pipeline that produced the bare rate, rather than bolted on afterward. On a surface where a path-based plane is already known to be good, as in `howto/oh-tst`'s PAFI-plane combination, the two are not competitors: OH-TST's sampling and transmission machinery can run on exactly the planes PAFI would have used.


# Next
