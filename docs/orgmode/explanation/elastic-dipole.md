

# Dipole of a relaxed defect

The sign convention is the one of Varvenne and Clouet, Phys. Rev. B 96, 224103 (2017), https://doi.org/10.1103/PhysRevB.96.224103: a dipole $P$ in an imposed strain has energy $-P:\epsilon$.  For a fixed periodic cell,

    P = -V (sigma_defect - sigma_bulk).

`ElasticDipole::from_stress` applies that residual-stress formula and symmetrizes the result.  `ElasticDipole::from_energy_strain_gradient` accepts the two homogeneous-strain energy gradients directly, $P=-(dE_defect/d\epsilon-dE_bulk/d\epsilon)$.  `ElasticDipole::from_forces` computes the symmetric first moment of equilibrated Kanzaki forces.  It refuses a force set with a resultant or torque above its scale-dependent roundoff allowance.  These three routes and the perfect-cell stress subtraction are reviewed by Clouet, Varvenne, and Jourdan, Comput. Mater. Sci. 147, 49 (2018), https://doi.org/10.1016/j.commatsci.2018.01.053.

The potential is not part of these types.  A host evaluates the relaxed bulk and defect configurations with rgpot, obtains stress (or homogeneous-strain derivatives) from rgpot strain requests, and passes the results to rgsaddle.  Atomic geometry remains in `linkcell::Cell`; its re-exported minimage displacement is used for the defect separation.  No second neighbour list or minimum-image convention is hidden in the elastic layer.


# Anisotropic Green function

`ElasticConstants` accepts a full $C_{ijkl}$, conventional Voigt $C_{IJ}$, cubic $C_{11},C_{12},C_{44}$, or isotropic Lamé constants.  It checks elastic symmetries and the acoustic tensor.  `AnisotropicGreen` evaluates Barnett's half-circle integral and Barnett's analytic second derivative, D. M. Barnett, Phys. Status Solidi B 49, 741 (1972), https://doi.org/10.1002/pssb.2220490238.  There is no finite-difference fallback.

For separation $r$, source $P^a$, and target $P^b$, the interaction is

    E_ab = P^b_ij G_ik,jl(r) P^a_kl.

The periodic trapezoid rule is evaluated on nested angular meshes.
Their difference is the angular error estimate.
The isotropic test compares the dipole interaction with Kelvin's closed form result.
The cubic test uses the exact `[100]` result of Dederichs and Leibfried, Phys. Rev. 188, 1175 (1969), https://doi.org/10.1103/PhysRev.188.1175: $G_{xx,xx}=1/(2 pi C_{44}r^3)$, while choosing constants that are not isotropic.


# Periodic correction

The $1/r^3$ image sum is conditionally convergent.  `PeriodicElastic` fixes its order by growing Cartesian spheres, as required when applying the periodic image correction of Varvenne, Bruneval, Marinica, and Clouet, Phys. Rev. B 88, 134102 (2013), https://doi.org/10.1103/PhysRevB.88.134102.  It reports the change on removing the outer sphere plus both nested angular errors.  For a pair,

    correction = E_infinite - E_periodic_images
    E_corrected = E_atomistic_periodic + correction.

`correct_energy` handles one pair state, `correct_landscape` maps a whole energy landscape, and `correct_rate_catalogue` applies the reactant and saddle corrections before rebuilding the harmonic rate.  Every result carries an absolute error estimate.  The periodic energy includes the cross interaction with every image and one half of each defect's interaction with its own nonzero images.  Consequently a barrier remains correct when a defect dipole changes between the reactant and saddle states.

The smoke cell-size test synthesizes the atomistic long-range contribution from the same periodic sum, then verifies that adding the correction removes its cell drift for several separations and box lengths.
The full copper job is driven by `scripts/benchmark-elastic-dipole.sh`.
It builds the defect-pair catalogue once, runs the network volume calculation for every size in `RGSADDLE_REPEATS`, and records the vector-return and vector-dipole $V_{iv}$ estimators as separate rows.
Both rows carry the vector- and orientation-resolved dipole bias on every hop of the same per-vector network.
The vector-return row sends a hop past the cutoff back to its source through the return layer of the lumped model.
The vector-dipole row matches the network to the field-free crystal outside the cutoff.
Neither row is uncorrected: the run without the dipole interaction is the no-strain-bias row of `benchmarks/defect_pair_run13_volume.csv`.


# Copper reference status

The opt-in Mishin Cu EAM test uses a reference supercell with 2048 lattice sites by default, relaxes a vacancy and a `[100]` dumbbell, derives their dipoles from the residual stress, and records finite symmetric tensors.
No published tensor for those two defects using exactly the 2001 Mishin parameterization and this fixed-cell convention was identified, so the test does not assert an unrelated potential's numbers.
Mishin et al., Phys. Rev. B 63, 224106 (2001), https://doi.org/10.1103/PhysRevB.63.224106, report defect energetics and elastic constants, not these residual-stress dipole tensors.
