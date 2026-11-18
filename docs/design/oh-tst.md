# Optimal hyperplane transition state theory: design

Ticket Software-ecn5. Base revision: `develop` at
3fe97278e3a7641b1ff88ae10c48e534d8e69e5c. This is a design document. No code
changes accompany it.

Citation keys in brackets refer to `docs/design/oh-tst-lit.bib`, which
Section 17 lists. Every DOI and arXiv identifier in that file was checked
against Crossref, DataCite (for the Zenodo software record) or the arXiv
API. Two theses have neither identifier; their entries give the repository
handle or the ISBN instead. Sources that could not be read in full are
listed in Section 16.4. Every
equation below was derived again for this document. The gradient, Hessian,
control-variate and sampler-bias statements were also checked numerically,
as described where they appear. Published reference values were recomputed
where the source gives enough information (Section 10). Where they disagree
with the source, both values are given.

## Status

`rgsaddle::ohtst` implements this design, and
`docs/orgmode/howto/oh-tst.org` is its guide. The plane optimizer runs on
rgmin's product manifold and takes from it items 1 to 4 and 7 of
Section 4.6:

- the offset $d$ is a Euclidean factor of weight one, not
  $\mathrm{SPD}(1)$;
- the normal lies on `ManifoldKind::sphere_excluding` of the sampler's
  held modes, or on `sphere_within` the span of a rotation basis $W$
  (Section 4.5);
- the sphere factor carries the block weight $\ell^2$ of Section 4.2, with
  $\ell$ the largest translation over the largest rotation, or the
  harmonic Hessian (3.15) as its `TangentMetric`, with $H_\parallel$ from
  a fixed number of Lanczos steps of Hessian actions at the reference
  point, the low-rank option of Section 4.2;
- `rtr::minimize_rtr_gauged` re-centres the pivot on the centroid of each
  accepted plane's sample;
- the rotation gradient is passed as a tangent vector, the gradient of the
  degree-zero extension, so rgmin's Weingarten term on the sphere
  vanishes.

The stochastic drivers of item 5 are not in rgmin, and the optimizer does
without them. It runs rgmin's Riemannian trust region on a sample-average
approximation: every plane is sampled from the same start configuration
with the same seed, and the change of $A$ from the current plane to a
trial is the trapezoid of their sampled gradients along the chord, exact
for a quadratic $A$, in place of the BAR difference of Section 4.4. The
model Hessian is the harmonic one, or one curvature per factor measured
along the trial steps; Hessian estimates with error bars (item 6) are not
used. The norm test grows the sample at accepted planes, and the run stops
when the translation and rotation blocks of the gradient have been within
a few block standard errors of zero for several accepted planes in a row,
in place of `control::NoiseFloor`. Averaged stochastic gradient ascent,
step 2 of Section 4.4, is not implemented.

## Contents

1. Scope and existing pieces
2. Decisions
3. Statistical mechanics of a hyperplane dividing surface
4. Riemannian formulation
5. Sampling in a hyperplane
6. Estimators and variance reduction
7. Free-energy differences, basin normalization and the rate
8. Dynamical correction
9. Anharmonicity and thermal expansion: the Cu vacancy hop at 700 K and 900 K
10. Test problems and reference values
11. What to take from PAFI
12. What to take from optimal transport
13. Neighbouring codes
14. Ranked improvements over the 2001 method
15. Implementation plan
16. Literature survey and access notes
17. Bibliography

## 1. Scope and existing pieces

Optimal hyperplanar transition state theory (OH-TST)
[johannessonOptimizationHyperplanar2001] estimates a classical rate constant.
It finds the hyperplane in configuration space with the highest free energy
and uses it as the transition state dividing surface. For classical dynamics
the TST rate through any dividing surface is an upper bound on the exact rate
[wignerTransitionState1938; horiutiStatisticalMechanical1938;
keckVariationalTheory1967]. The plane of maximum free energy therefore gives
the lowest, and best, planar estimate. A dynamical correction removes the
remaining recrossing error.

The 2001 paper moves the plane with damped "plane dynamics" while it
integrates the reversible work from the reactant. Bligaard and Jónsson
[bligaardOptimizationHyperplanar2005] applied it to two-dimensional test
problems. eOn has an `oh_tst` job that ports the 2001 paper, including its
flux-mass error (Section 13.1). An implementation of the 2001 method as
published is also being written separately for rgsaddle. This document does
six things:

- It restates the statistical mechanics, including the corrections the 2001
  equations need (Section 3.8).
- It recasts the plane search as stochastic optimization on the manifold of
  hyperplanes, implemented with rgmin.
- It chooses the sampler and the estimators, with an error analysis.
- It says how to treat the Cu vacancy hop at 700 K and 900 K without
  Cu-specific code. The Vineyard harmonic rate is reported to fall below the
  MD rate for that hop; Section 9.6 traces how much evidence supports this.
- It records what to take from PAFI and from optimal transport.
- It lists test problems with reference values traceable to sources.

The design reuses these existing rgsaddle pieces:

- `src/pafi.rs`: projected average force integration on an arc-length path.
  It runs overdamped Langevin (Euler–Maruyama) dynamics in the plane and
  averages `dF/ds = ⟨(∇E·n)(1 − ψ)⟩` with block-averaged errors (`PafiConfig`,
  `ArcPath`, `average_hyperplane`, `integrate_profile`, `pafi_profile`). The
  path is a polyline, whose curvature is zero, so ψ = 0 there.
- `src/bennett.rs`: the Bennett–Chandler transmission coefficient with BAOAB
  dynamics and a scalar mass (`ShootingConfig`, `bennett_chandler`,
  `Transmission`).
- The defect-pair rate pipeline (`src/defect_pair.rs`,
  `docs/orgmode/howto/defect-pair-rates.org`). It reports κν exp(−ΔF/kT)
  beside the Vineyard rate (`vineyard_prefactor` in `src/strain_rate.rs`).
- `tests/mishin_cu.rs`: the Mishin embedded-atom Cu potential
  [mishinStructuralStability2001], with NPT lattice parameters from 300 K to
  700 K.

rgmin, pinned at d3f94d11c583864d980c750a9240333d1244bd00 (0.6.0) when this
was written, provided:

- the `Manifold` trait (`project`, `egrad2rgrad`, `retract`, `transport`) and
  `Sphere`;
- `ManifoldKind`, including `Sphere`, `Oblique` and `MwRigid`;
- the runtime product `RiemannianManifold`, with Stiefel and SPD factors and
  `gradient_step`;
- the trust-region pieces `rtr::truncated_cg_projected` and `RtrRadius`;
- the Hessian-vector machinery `hvp::FdHvp`, `steihaug_pcg` and
  `NystromPrecond`;
- Moré–Wild noise estimation (`noise`) and a noise-floor stopping rule
  (`control::NoiseFloor`).

It lacked a Euclidean factor in the runtime product, per-factor metric
weights and a stochastic-approximation driver. Section 4.6 lists what to add,
written as requirements on rgmin's product manifold, which was derived
separately. rgsaddle now pins rgmin at
731e48b67585519f5e2807ea02134cc0900e4212, tag v0.6.0, which carries that product
manifold: `ProductManifold` of weighted `ProductFactor`s, Euclidean factors,
`ManifoldKind::sphere_within` and `sphere_excluding`, a `TangentMetric` per
factor, and `rtr::minimize_rtr_gauged`.

## 2. Decisions

1. Work in mass-weighted coordinates $q = M^{1/2}x$. Then the flux factor is
   the constant $(kT/2\pi)^{1/2}$, and maximizing the plane free energy is
   exactly minimizing $k_\mathrm{TST}$ over planes. The effective mass in the
   2001 paper (its Eq. 24) is wrong for unequal masses (Section 3.2).
2. The objective is the plane free energy
   $A(c,n) = -kT \ln \int \delta(n\cdot(q-c))\, e^{-\beta V}\, dq$ on the
   manifold of oriented hyperplanes, maximized. Its exact gradient has a
   translation part $\langle n\cdot\nabla V\rangle$ and a rotation part
   $\langle (n\cdot F)(q-c)\rangle$ (Section 3.3). An exact fluctuation
   formula gives the Hessian (Section 3.5). This is a maximization over
   planes, not a saddle search, because each plane already integrates out
   the in-plane degrees of freedom.
3. Start from the zero-temperature optimum. That is the plane through the
   first-order saddle, normal to the mass-weighted unstable mode, which
   rgsaddle's saddle searches already produce. Reach the reactant with PAFI
   planes along the minimum energy path (existing code, area measure), then
   optimize the plane at temperature. The barrier
   $A^\ddagger - A_R$ is the PAFI part plus the optimization part, and each
   part is exact (Section 7.1).
4. The optimizer is preconditioned, averaged Riemannian stochastic gradient
   ascent with an adaptive sample size. Near convergence it switches to a
   sampled trust region whose actual change is measured by the Bennett
   acceptance ratio (BAR) between planes. The preconditioner comes from the
   harmonic model at the saddle. Rotation is restricted to a low-dimensional
   subspace, which keeps the variational bound (Section 4.5). Iteration stops
   at the noise floor.
5. The sampler is projected BAOAB in the plane. It enforces the constraint
   exactly without iteration, is exact for the harmonic part of the
   potential, and has an O(Δt²) anharmonic bias. PAFI's in-plane quench
   check detects escape from the channel (Section 5).
6. The mean force and torque are estimated with Stein control variates built
   from the zero-temperature Hessian. These are exactly unbiased and remove
   the harmonic part of the variance (Section 6.2). Errors come from block
   averages plus independent replicas. Differences between planes use BAR
   with rigid maps (Section 7.1). Thermodynamic integration over parallel
   planes replaces the 2001 crossing estimator of
   $Z_{H_R}/Z_R$, which has infinite variance (Section 7.2).
7. The transmission coefficient is Bennett–Chandler from the optimal plane
   (`bennett.rs`, generalized to masses), with quench-based basin assignment
   (Section 8).
8. Thermal expansion is a caller-supplied cell at each temperature. All
   stationary points are re-relaxed in that cell, and sampling runs at that
   fixed cell. Nothing is material-specific (Section 9).

## 3. Statistical mechanics of a hyperplane dividing surface

### 3.1 Coordinates

Let $x \in \mathbb{R}^D$ be the free Cartesian coordinates, with $D$ equal to
three times the number of mobile atoms, and let $M$ be the diagonal mass
matrix. Mass-weighted coordinates are $q = M^{1/2}x$, the potential is
$V(q)$, the force is $F = -\nabla_q V = M^{-1/2}F_x$, and the Hessian is
$H = M^{-1/2} H_x M^{-1/2}$. In these coordinates the Maxwell distribution
of $v = \dot q$ is isotropic, with variance $kT$ in every direction.
$\beta = 1/kT$.

### 3.2 TST rate through a plane

A plane is

$$\Sigma(c,n) = \{ q : n\cdot(q-c) = 0 \}, \qquad |n| = 1. \tag{3.1}$$

$|\nabla_q (n\cdot q)| = 1$, so $\delta(n\cdot(q-c))\,dq$ is the
$(D-1)$-dimensional area measure $d\sigma$ on $\Sigma$. The plane
configuration integral and plane free energy are

$$Z_H(c,n) = \int \delta(n\cdot(q-c))\, e^{-\beta V(q)}\, dq
= \int_\Sigma e^{-\beta V}\, d\sigma, \qquad A(c,n) = -kT\ln Z_H(c,n).
\tag{3.2}$$

The one-way classical TST rate out of the reactant region $R$ through
$\Sigma$ is

$$k_\mathrm{TST}(\Sigma) = \langle (n\cdot v)_+\rangle\, \frac{Z_H}{Z_R}
= \left(\frac{kT}{2\pi}\right)^{1/2} \frac{Z_H(c,n)}{Z_R},
\qquad Z_R = \int_R e^{-\beta V}\, dq. \tag{3.3}$$

This is Eq. (22) of [vandenEijndenTransitionState2005], which those authors
write with the same mass-weighting. In Cartesian coordinates, with a
Cartesian unit normal $\nu$, the same rate is

$$k_\mathrm{TST} = \left(\frac{kT}{2\pi\mu}\right)^{1/2}
\frac{Z_{H,x}}{Z_{R,x}}, \qquad \mu^{-1} = \nu^\mathsf{T} M^{-1}\nu
= \sum_i \frac{\nu_i^2}{m_i}. \tag{3.4}$$

To derive (3.4), note that $n = M^{-1/2}\nu/|M^{-1/2}\nu|$. Then
$\delta(n\cdot(q-c)) = |M^{-1/2}\nu|\,\delta(\nu\cdot(x-c_x))$ and
$dq = \det M^{1/2}\, dx$, and the determinants cancel in the ratio. Eq. (24)
of the 2001 paper uses $\mu = \sum_i m_i \nu_i^2$. That agrees with (3.4)
only when all masses are equal. The variance of $\nu\cdot\dot x$ is
$kT\sum_i \nu_i^2/m_i$ for independent Maxwellian components. The 2001
argument from the velocity ellipse gives instead the inertia of a rigid
collective displacement. The 2001 examples use equal masses, so their
numbers are unaffected. In Cartesian coordinates the prefactor depends on
the plane orientation through $\mu$. Optimizing $A$ alone is then the wrong
objective. In mass-weighted coordinates it is exact.

The 2001 paper splits (3.3) using a plane $H_R$ in the reactant region:

$$k = \left(\frac{kT}{2\pi}\right)^{1/2} \frac{Z_{H_R}}{Z_R}\,
e^{-\beta (A^\ddagger - A_R)}, \qquad A_R = A(H_R),\quad
A^\ddagger = A(\Sigma^\ddagger). \tag{3.5}$$

This is its Eq. (3), written there as $(\langle|v_\perp|\rangle/2)
(Q_{Z_R}/Q_R)\, e^{-\Delta A/kT}$. The ratio called $Z_{H_R}/Z_R$ in the
ticket is $Q_{Z_R}/Q_R$ there.

**Variational principle.** For classical dynamics,
$k \le k_\mathrm{TST}(\Sigma)$ for every surface $\Sigma$ that separates $R$
from the product region $P$
[wignerTransitionState1938; horiutiStatisticalMechanical1938;
keckVariationalTheory1967; truhlarVariationalTransition1984;
vandenEijndenTransitionState2005]. Among planes, the best estimate is
therefore the plane of maximum $A$.

Vanden-Eijnden and Tal (their Section V) show that "the optimal TST surface
maximizes the free energy" is false for the reaction-coordinate free energy
$F(z)$ of a curved family of surfaces. It is true for the surface free
energy $G(S)$ of their Eq. (51). For planes with a unit normal in
mass-weighted coordinates $|\nabla q| = 1$, so $F$ and $G$ coincide up to a
constant and the statement is exact. It fails for Cartesian planes with
unequal masses, because of the orientation-dependent $\mu$ in (3.4).

### 3.3 Reversible work for rigid motions of the plane

Let the plane move rigidly, $t \mapsto (c(t), n(t))$ with $|n| = 1$. The
normal velocity of the plane at a point $q \in \Sigma$ is

$$w(q) = n\cdot\dot c - \dot n\cdot(q-c). \tag{3.6}$$

Then

$$\frac{dA}{dt} = \big\langle (n\cdot\nabla V)\, w \big\rangle_\Sigma,
\tag{3.7}$$

where $\langle\cdot\rangle_\Sigma$ is the canonical average over $\Sigma$
with the area measure. The derivation takes one line. With
$g = n\cdot(q-c)$,

$$\dot Z_H = \int (\dot n\cdot(q-c) - n\cdot\dot c)\,\delta'(g)\,
e^{-\beta V}\, dq.$$

Since $|n| = 1$, $\delta'(g) = n\cdot\nabla\delta(g)$. Integrating by parts
along $n$, and using $n\cdot\dot n = 0$, gives
$\dot Z_H = -\beta\int \delta(g)\, w\,(n\cdot\nabla V)\, e^{-\beta V}\,dq$.

The components follow, with the pivot $c$ on the plane:

$$\frac{\partial A}{\partial d} = \langle n\cdot\nabla V\rangle_\Sigma
\quad (c \to c + d\,n), \tag{3.8}$$

$$\operatorname{grad}_n A = \langle (n\cdot F)(q-c)\rangle_\Sigma
= -\langle (n\cdot\nabla V)(q-c)\rangle_\Sigma . \tag{3.9}$$

(3.9) is the Riemannian gradient on the unit sphere. It is tangent to the
sphere because $q - c \perp n$ on $\Sigma$. These are Eqs. (33)–(34) of
[vandenEijndenTransitionState2005]. They were also checked against central
differences of quadrature free energies (Section 3.9).

- Moving the pivot within the plane ($\dot c \perp n$, $\dot n = 0$) gives
  $w = 0$. It is a gauge freedom.
- The rotation gradient depends on the pivot. Moving the pivot by $t$ within
  the plane changes (3.9) by $-\langle n\cdot\nabla V\rangle_\Sigma\, t$.
  That vanishes only where the translational force vanishes, as Vanden-Eijnden
  and Tal remark. A pivot at the in-plane centroid $\langle q\rangle_\Sigma$
  decouples the two blocks of the Hessian (Section 3.6).
- For a path of planes $c = X(s)$, $n = \hat n(s)$,

$$\frac{dA}{ds} = \big\langle (\hat n\cdot\nabla V)\,
(\hat n\cdot X' - \hat n'\cdot(q - X)) \big\rangle_\Sigma . \tag{3.10}$$

  For $\hat n = X'/|X'|$ this becomes
  $\langle (X'\cdot\nabla V)(1 - (q-X)\cdot X''/|X'|^2)\rangle$. It is the
  same quantity as Eq. (17) of Schenter, Mills and Jónsson
  [schenterReversibleWork1994], Eq. (4) of the 2001 paper (arc length), and
  Eq. (10) of the finite-temperature string paper
  [eFiniteTemperature2005]. It is also PAFI's
  $\langle\psi\,\partial_\xi X_0\cdot\nabla V\rangle$
  [swinburneUnsupervisedCalculation2018] and the
  $(f\cdot n)(1-\psi)$ of LAMMPS `fix pafi`.
- The full rigid-motion work $\langle\nabla V\cdot\dot q_\mathrm{rigid}\rangle$
  has the same mean, because rigid motions preserve $d\sigma$ and the
  in-plane part integrates to zero by parts. Its variance is larger, so only
  the normal component should be averaged.

### 3.4 Area measure and PAFI's closest-point measure

PAFI defines $F(\xi)$ with $\delta(\xi - \tilde\xi(X))$, where $\tilde\xi$
is the closest-point projection onto the path (PRL Eq. 1). By the co-area
formula, $\delta(\xi-\tilde\xi)\,dX = d\sigma/|\nabla\tilde\xi|$ on
$\Sigma_\xi$. PRL Eq. (4) gives
$|\nabla\tilde\xi| = 1/(|\psi|\,|\partial_\xi X_0|)$. Hence, exactly,

$$F_\mathrm{PAFI}(\xi) = A(\xi) - kT\ln\langle|\psi|\rangle_\xi -
kT\ln|\partial_\xi X_0| . \tag{3.11}$$

The Jacobian term in PRL Eq. (8) comes from these extra factors. The TST
flux through $\Sigma_\xi$ is the area measure times $|n\cdot v|$, so the
rate needs $A$, not $F_\mathrm{PAFI}$. The reversible work for $A$ is
exactly (3.10), with no Jacobian term. LAMMPS `fix pafi` and the PAFI driver
average exactly (3.10), with no Jacobian term (Section 11). Their profile is
therefore the area-measure $A$, and so is the one `pafi.rs` computes. Where
$\psi \approx 1$ the two profiles differ by a slowly varying function. With
PAFI's $k = \omega_0 e^{-\beta\Delta F}$ the difference disappears into an
unspecified $\omega_0$. With (3.3) it does not, so rgsaddle should keep the
area measure throughout.

### 3.5 Second derivatives

Parametrize the plane as $(c(\theta), n(\theta))$, $\theta \in
\mathbb{R}^m$, and write $w_a = n\cdot\partial_a c - \partial_a n\cdot(q-c)$.
Differentiating (3.7) once more gives

$$\partial_a\partial_b A = \langle (n^\mathsf{T} H n)\, w_a w_b\rangle +
\langle (\partial_b n\cdot\nabla V)\, w_a\rangle +
\langle (n\cdot\nabla V)\, \partial_b w_a\rangle -
\beta\, \mathrm{Cov}\big((n\cdot\nabla V) w_a,\, (n\cdot\nabla V) w_b\big),
\tag{3.12}$$

with $\partial_b w_a = \partial_b n\cdot\partial_a c + \partial_a n\cdot
\partial_b c + n\cdot\partial_a\partial_b c - \partial_a\partial_b n\cdot
(q-c)$. For translation alone,

$$\frac{\partial^2 A}{\partial d^2} = \langle n^\mathsf{T} H n\rangle_\Sigma -
\beta\,\mathrm{Var}_\Sigma(n\cdot\nabla V). \tag{3.13}$$

The covariance term is the Fisher information of the family of planes. The
first term needs one directional second derivative per sample.

### 3.6 Harmonic limit

Near a saddle, $V = V^\ddagger + \tfrac12 q^\mathsf{T} H q$, with eigenvalues
$\lambda_1 < 0 < \lambda_2 \le \dots$ and eigenvectors $e_j$. For the plane
$n\cdot q = d$, the restriction of $H$ to $n^\perp$ has determinant
$\det H\, (n^\mathsf{T}H^{-1}n)$. The in-plane minimum energy is
$V^\ddagger + \tfrac12 d^2/(n^\mathsf{T}H^{-1}n)$. Hence

$$A(d,n) = V^\ddagger + \frac{kT}{2}\ln\!\big[\det H\,
(n^\mathsf{T}H^{-1}n)\big] - \frac{D-1}{2}kT\ln(2\pi kT) +
\frac{d^2}{2\, n^\mathsf{T}H^{-1}n}, \tag{3.14}$$

which is finite when $n^\mathsf{T}H^{-1}n < 0$. The maximum is at
$n = e_1$, $d = 0$. There (3.3) with a harmonic $Z_R$ is Vineyard's rate
$\nu = \prod_{i=1}^D \omega_i^R / (2\pi \prod_{j\ge2}\omega_j^\ddagger)$
[vineyardFrequencyFactors1957]. Write $\phi_j$ for the rotation angle of
$n$ from $e_1$ towards $e_j$. The Hessian at the optimum, in the
coordinates $(d, \phi_2, \dots, \phi_D)$, is

$$\frac{\partial^2 A}{\partial d^2} = -|\lambda_1|, \qquad
\frac{\partial^2 A}{\partial\phi_j\,\partial\phi_k}
= -kT\Big(1 + \frac{|\lambda_1|}{\lambda_j}\Big)\delta_{jk}, \qquad
\frac{\partial^2 A}{\partial d\,\partial\phi_j} = 0 . \tag{3.15}$$

The cross term vanishes because the pivot is the in-plane centroid, which is
the saddle at $d = 0$. For a plane through a minimum with Hessian $H_R$ and
normal $n$, $A(d) - A(0) = d^2/(2n^\mathsf{T}H_R^{-1}n)$. The harmonic basin
normalization is therefore
$Z_{H_R}/Z_R = (2\pi kT\, n^\mathsf{T}H_R^{-1}n)^{-1/2}$.

Expanding (3.12) around the harmonic optimum gives the temperature
dependence of the optimum. The translational force on the zero-temperature
plane is $O(kT)$, from third derivatives of $V$. The torque is $O((kT)^2)$,
because odd Gaussian moments vanish. The optimal plane therefore shifts by
$O(kT)$, and the gain $\beta(A^* - A(\Sigma_0^\ddagger))$ from optimizing
starts at first order in $kT$. Most of the translational part is already
captured by the maximum of a PAFI profile along the minimum energy path. For
weakly anharmonic solids the extra gain from rotating is small. There the
anharmonic free energy of the planes themselves dominates the departure from
Vineyard. Bligaard and Jónsson's model is the opposite regime: a strongly
coupled bath coordinate makes the optimal plane jump by about 70° at
$T = 0.227\,E_a/k$ (Section 10.2).

### 3.7 What a single plane cannot represent

The 2001 paper and Bligaard and Jónsson record three failure modes, and
each needs a check in the implementation.

- Symmetry-equivalent channels. For the Al adatom on Al(100), without a
  symmetry restriction the optimization "converged to a hyperplane that
  resembled an average plane for two symmetrically equivalent concerted
  displacement processes" (2001 paper, Section III D). The sampled plane
  must be restricted to one channel (Section 5.4).
- Multiple maxima. In Bligaard and Jónsson's model a second maximum appears
  at $T = 0.1182\,E_a/k$ and becomes the global one at
  $T = 0.2266\,E_a/k$. Local ascent from the zero-temperature plane stays on
  the old branch, so candidates must be compared (Section 4.4).
- Curved dividing surfaces. At high temperature, or with several product
  channels, a plane is a poor surface. The remaining error appears as
  $\kappa < 1$. Bligaard and Jónsson propose piecewise-planar surfaces for
  multichannel cases.

### 3.8 Corrections to the 2001 equations

Vanden-Eijnden and Tal state that the 2001 paper gave "incorrect equations
for $b$ and $\hat n$" and give the correct ones, their Eq. (33). Those agree
with (3.8)–(3.9). Reading the 2001 text against (3.7)–(3.9) identifies the
specific problems:

1. The effective mass, Eq. (24), as in Section 3.2.
2. The rotational force is
   $\tau_i = \langle (\hat n_i\cdot F_{\parallel,i}) R_i/(a_i|R_i|^2)\rangle$,
   with the arm $R_i = r - G_{s_i}$ measured from the intersection of the
   plane with the guideline. The factor $1/|R_i|^2$ sits inside the average,
   so $\tau$ reweights samples and is not the gradient (3.9). Its zero
   differs from the stationarity condition except where $n\cdot F$ vanishes
   pointwise, as in the harmonic limit at the optimum. In high dimension
   $|R|^2$ concentrates and the difference shrinks. In the two-dimensional
   tests it does not.
3. The sign. Eq. (8), $\ddot{\hat n} = -\tau$, moves the normal along
   $-\langle(n\cdot F)R\rangle$, which lowers $A$. The text says the plane
   is "rotated against the rotational force", and Appendix A rotates "by
   adding $t = R(\hat n\cdot F)$ to the normal". By (3.9), Appendix A is the
   ascent direction. The translational update, Eq. (5), has the ascent sign.
4. The pivot. Rotation about the guideline intersection instead of the
   centroid couples rotation to the translational force (Section 3.3).
5. Plane dynamics. Velocity Verlet with velocity zeroing, separate $s$ and
   $\hat n$ steps, caps of 1° to 5° per step and an inertia parameter $a$ is
   a heuristic optimizer with no convergence statement for noisy forces. For
   the Al adatom, changing the inertia parameter changed which mechanism the
   plane converged to (0.154 against 0.00220 in Section III D).
6. Transport between planes, Eqs. (11)–(12), is first order in the rotation
   and rescales the arm by hand. Section 7.1 uses an exact rigid map instead.
7. The reactant normalization, Eq. (22), has an infinite-variance estimator
   (Section 7.2).
8. Integration. The trapezoid rule over plane-to-plane averages, with planes
   discarded by "grooming", gives no propagated error.
9. A label. Section III B reports "$Q^\ddagger/Q_{Z_R}$ … 0.765 ± 0.005" and
   combines it with $\langle|v|\rangle = 0.399$ into
   $k = (4.5\pm0.2)\times10^{-7}$. By Eq. (3) the only ratio of
   configuration integrals in the prefactor is $Q_{Z_R}/Q_R$. A value of
   0.765 for $Q^\ddagger/Q_{Z_R} = e^{-\beta\Delta A}$ would contradict the
   barrier of about $13\,kT$, so the label is $Q_{Z_R}/Q_R$. The same
   paragraph says $k_BT=0.1E_A$, but Appendix B gives $E_A=3.63395$ from the
   reactant minimum. Its printed speed and Vineyard rate instead both select
   $k_BT=0.25$ in the Appendix B energy units with unit mass. At that
   convention the paper's crossing estimator and trapezoidal work, repeated
   over eight seeds, give $(4.340\pm0.132)\times10^{-7}$, inside the printed
   $(4.5\pm0.2)\times10^{-7}$ band. Section 10.3 gives the full check.

### 3.9 Numerical checks

The checks used two-dimensional anharmonic potentials with
$K = ((1, 0.6), (0.6, 2))$, $kT = 0.7$ and the pivot at $(0.5, 0.2)$. Plane
integrals were computed by adaptive quadrature.

- With $V = \tfrac12 q^\mathsf{T}Kq + 0.03q_1^3$, (3.8) and (3.9) agree with
  central differences of $A$ to eight digits at three orientations.
- With $V = \tfrac12 q^\mathsf{T}Kq + 0.03q_1^3 + 0.02q_2^4$, (3.12) agrees
  with a finite-difference Hessian to a few parts in $10^6$, which is the
  finite-difference error.
- The sign of (3.9) was checked separately against the 2001 Eq. (8) and its
  Appendix A, with the result stated in item 3 of Section 3.8.
- The control variates of Section 6.2 and the sampler biases of Section 5.2
  were checked as described there.

## 4. Riemannian formulation

### 4.1 Manifold

An oriented hyperplane is a pair $(n, d)$, a unit normal and a signed
offset, relative to a pivot $c$:
$\Sigma = \{q : n\cdot(q-c) = d\}$. The space of planes is
$S^{D-1}\times\mathbb{R}$. It is the quotient of the point-normal space
$\mathbb{R}^D\times S^{D-1}$ by in-plane moves of the point, which are an
$(D-1)$-dimensional gauge. The unoriented planes are a further quotient by
$(n,d)\sim(-n,-d)$. TST needs an orientation, so fix it by
$n\cdot n_0 > 0$, with $n_0$ pointing from $R$ to $P$. A small step never
flips it.

In practice the driver holds $(c, n)$ with $c$ on the plane. A tangent
vector is $(u, \delta)$, with $u \perp n$ a rotation and $\delta$ a normal
translation. After each accepted step the pivot is re-centred on the current
estimate of the in-plane centroid. This changes the representation, not the
plane.

Linear constraints shrink the sphere. The sampler removes the centre of
mass motion. In mass-weighted coordinates its three directions are
$e_\alpha = (\sqrt{m_1}\,\hat e_\alpha, \sqrt{m_2}\,\hat e_\alpha,
\dots)/\sqrt{M_\mathrm{tot}}$, which are mutually orthogonal. Fixed atoms
are simply removed from $q$. A normal with a component along a constrained
direction is meaningless, because the sampler cannot move along it. So
$n \in S(U)$, the unit sphere of $U = \operatorname{span}\{e_\alpha\}^\perp$.
The projection is $P_U n = n - \sum_\alpha (e_\alpha\cdot n)\,e_\alpha$,
followed by normalization. With $n \in U$,
$P = I - nn^\mathsf{T} - \sum_\alpha e_\alpha e_\alpha^\mathsf{T}$ is an
orthogonal projector. LAMMPS `fix pafi` applies its centre-of-mass and
normal projections one after the other, without mass weighting, which is
correct only when the directions are orthogonal (Section 11.4).

### 4.2 Metric and block weights

The natural candidate metrics differ by large factors, and (3.15) fixes the
choice.

- **Unweighted product metric.** The rotation block of the Hessian has
  eigenvalues $-kT(1+|\lambda_1|/\lambda_j)$. The translation eigenvalue is
  $-|\lambda_1|$. Rotation angles are dimensionless and offsets are
  lengths, so the two blocks differ by a dimensional factor that the metric
  must supply. Within the rotation block the condition number is at most
  $1 + |\lambda_1|/\lambda_\mathrm{min}$, which is large only when soft
  in-plane modes exist.
- **Block-weighted metric.** Use $g((u,\delta),(u,\delta)) = \ell^2|u|^2 +
  \delta^2$. Then the rotation eigenvalues become
  $-kT(1+|\lambda_1|/\lambda_j)/\ell^2$. Choose
  $\ell^2 = kT(1+|\lambda_1|/\bar\lambda)/|\lambda_1|$, with $\bar\lambda$ a
  typical in-plane eigenvalue, so the two blocks have comparable
  curvature. When most in-plane eigenvalues exceed $|\lambda_1|$, as in
  close-packed metals, $\ell^2 \approx 2kT/|\lambda_1|$ is a reasonable
  start. $\ell$ is a length, so $\ell^2|u|^2$ and $\delta^2$ have the same
  units.
- **Harmonic Newton preconditioner** (recommended). Use the negative of the
  harmonic Hessian (3.15) as the metric: $|\lambda_1|$ on the translation
  and $kT(I + |\lambda_1|H_\parallel^{-1})$ on $T_nS(U)$, where $H_\parallel$
  is the in-plane Hessian at the saddle. Apply its inverse by conjugate
  gradients on Hessian-vector products of $V$ at the saddle. A cheaper
  option is a low-rank Lanczos approximation of the softest in-plane modes,
  which dominate $1+|\lambda_1|/\lambda_j$. In the harmonic limit,
  preconditioned ascent converges in a few iterations whatever the
  conditioning.
- **Wasserstein pullback metric.** The normal part of the Wasserstein-2
  metric of the plane distributions is
  $g = \mathbb{E}_\Sigma[w^2] = \delta^2 + u^\mathsf{T}\Sigma_\parallel u$,
  with $\Sigma_\parallel$ the in-plane covariance (Section 12.3). In the
  harmonic limit $\Sigma_\parallel = kT H_\parallel^{-1}$. The natural
  gradient is then the regression of $n\cdot\nabla V$ on $(1, q-c)$, and the
  rotation direction is $-P_n H n$. That is the dimer rotation force
  [henkelmanDimerMethod1999], so the zero-temperature limit of OH-TST in
  this metric is the dimer method. The rotation rates of its gradient flow
  are $\lambda_j + |\lambda_1|$, a condition number of
  $(\lambda_\mathrm{max}+|\lambda_1|)/(\lambda_\mathrm{min}+|\lambda_1|)$.
  That is worse than the unweighted metric in stiff solids. It is useful for
  interpretation and for units, not as the optimizer's metric.

### 4.3 Gradient, retraction, transport and Hessian

- The Riemannian gradient is $G^{-1}$ applied to
  $(\operatorname{grad}_n A, \partial A/\partial d)$ from (3.8)–(3.9), where
  $G$ is the metric operator.
- The retraction is $n \leftarrow P_U(n+u)/|P_U(n+u)|$ and
  $c \leftarrow c + \delta\, n$, followed by re-centring. This is rgmin
  `Sphere::retract` restricted to $U$.
- Vector transport is projection onto the new tangent space
  (`Sphere::transport`). That is enough for averaging and quasi-Newton
  memory.
- The Riemannian Hessian on the sphere is
  $\operatorname{Hess} f(n)[u] = P_n(\nabla^2\bar f(n)\,u) -
  (n\cdot\nabla\bar f(n))\,u$ for an extension $\bar f$. Use the degree-zero
  extension $\bar f(n) = A(n/|n|)$. Then $n\cdot\nabla\bar f = 0$ and the
  Weingarten term vanishes. An implementation that uses another extension
  and drops the term gets the wrong curvature.
- Hessian-vector products have three sources:
  1. the fluctuation formula (3.12), which is exact but needs
     $n^\mathsf{T}H(q)\,n$ for every sample. That is two extra force calls
     by central differences, or one Hessian-vector product. Its covariance
     term is noisy.
  2. common-random-number finite differences of the gradient estimate
     between nearby planes. The two runs share the noise sequence, and their
     starting configurations are related by the rigid map of Section 7.1.
     For short trajectories the variance of the difference is $O(\epsilon^2)$.
     For long ones it grows with the trajectory length through chaotic
     divergence, so use many short pairs.
  3. the harmonic model (3.15) from the zero-temperature saddle Hessian. It
     costs nothing extra and is exact as $T\to0$.

  Use the harmonic model as the preconditioner, common-random-number
  differences to check curvature inside trust-region steps, and the
  fluctuation formula for diagnostics and tests.

### 4.4 Optimizer for noisy gradients

The gradient estimates have a bias, from sampler discretization, and a
covariance $\Sigma_g/n_\mathrm{eff}$. $A$ itself is never observed, but
differences of $A$ between planes are, through BAR (Section 7.1).

1. Start at the zero-temperature optimum, the saddle plane. For weakly
   anharmonic solids this is inside the basin of attraction of the
   finite-temperature optimum (Section 3.6).
2. Iterate preconditioned Riemannian stochastic gradient ascent
   [bonnabelStochasticGradient2013]:
   $\eta_k = \gamma_k G^{-1}\hat g_k$, with
   $\gamma_k = \gamma_0(1+k/k_0)^{-\alpha}$ and $\alpha\in(1/2,1]$, then
   retract. Average the iterates in the tangent space at the running mean
   [polyakAccelerationStochastic1992; tripuraneniAveragingStochastic2018].
   Bonnabel proves almost-sure convergence on Riemannian manifolds under the
   Robbins–Monro step conditions [robbinsStochasticApproximation1951].
   Tripuraneni et al. show that averaging attains the asymptotically
   efficient $O(1/k)$ rate near a nondegenerate optimum.
3. Grow the sample size with the norm test
   [byrdSampleSize2012]. Increase $n_k$ when the estimated standard error of
   $\hat g_k$ exceeds a fraction $\theta$ (0.5 to 0.9) of its norm.
4. Near the optimum, switch to a sampled trust region
   [absilTrustRegion2007; blanchetConvergenceRate2019]. Build the model from
   the harmonic Hessian, corrected by common-random-number curvature. Solve
   for the step with `rtr::truncated_cg_projected`, then measure the actual
   change of $A$ by BAR between the current and the trial plane. Update the
   radius with `RtrRadius::update(actual, model, |η|)`. Accept only when the
   BAR difference exceeds a few of its standard errors, or the model
   predicts a change below the noise. This is the condition under which
   stochastic trust-region convergence results hold: function estimates
   accurate to $O(\Delta^2)$ and gradients to $O(\Delta)$ with fixed
   probability, where $\Delta$ is the radius.
5. Stop at the noise floor, when the preconditioned gradient norm is below
   a few standard errors (rgmin `control::NoiseFloor`, with the noise level
   from the replica covariance). With rotation restricted to a subspace
   (Section 4.5) the test is a $\chi^2$ test in a few tens of dimensions,
   which is meaningful. In the full $D-1$ dimensions it is not.
6. Report the optimal plane, $A^*$ with its standard error, and the
   uncertainty of the plane. The plane's uncertainty is the inverse Hessian
   applied to the gradient covariance. Because the optimum is stationary, an
   error $\delta\theta$ in the plane changes $A$ only by
   $\tfrac12\delta\theta^\mathsf{T}\operatorname{Hess}\,\delta\theta$. Noise
   in the plane therefore enters the rate at second order. This is the main
   practical advantage of a variational formulation.

Multiple maxima (Section 3.7) are handled by comparing candidates. The
candidates are the optimum reached from the zero-temperature plane, the
optimum reached from the maximum of the PAFI profile, and, when the
low-dimensional rotation subspace allows it, a coarse scan over angles. They
are compared by BAR or thermodynamic integration between them. Temperature
continuation alone cannot see a maximum that appears away from the tracked
branch, as Bligaard and Jónsson's second maximum does at 43°.

Adam and FIRE-like schemes are not recommended. They normalize the step by
noisy moment estimates and ignore the curvature information that is
available here. The 2001 plane dynamics is a FIRE-like scheme. Per-factor
adaptive scaling on product manifolds [becigneulRiemannianAdaptive2019]
contains the right idea: the translation and rotation blocks have different
noise and curvature. The block weights of Section 4.2 give that scaling
deterministically.

The plane sampler may continue one Markov chain across iterations instead
of restarting. The gradient noise is then correlated. Durmus et al.
[durmusRiemannianStochastic2021] treat fixed-step Riemannian stochastic
approximation with Markovian noise. The iterates have a stationary
distribution that stays in an $O(\eta^{1/2})$ neighbourhood of the optimum
for step $\eta$. That supports a fixed step during the transient, followed
by decreasing steps or averaging, as in step 2.

### 4.5 Rotation dimension and subspace restriction

A cell with 1000 mobile atoms has about 3000 rotational degrees of freedom.
The noise in a gradient estimate grows like $\sqrt{D}\,\sigma/\sqrt{n}$ in
norm. Near the optimum the true torque along stiff in-plane modes is
$O((kT)^2)$, so those components are mostly noise. Unrestricted ascent then
random-walks the normal in those directions, and every unit of random
rotation costs $\tfrac12 kT(1+|\lambda_1|/\lambda_j)\phi_j^2$ of free
energy. Preconditioning does not help, because those curvatures are about
$kT$ anyway. Three remedies:

1. Restrict rotation to a subspace $W$. $W$ is spanned by the unstable mode,
   the $k$ softest in-plane modes at the saddle (Lanczos), and averaged
   torque directions that become significant during the run. A direction
   becomes significant when its signal-to-noise ratio exceeds a threshold.
   Typically $k$ is 10 to 50.
2. Restrict by locality. In a solid the normal is concentrated on a few
   atoms. For the Al adatom, seven atoms carry essentially all of the
   normal's weight (2001 paper, Section III D). Allow rotation only on atoms
   within a cutoff of the moving atoms, with a smooth taper.
3. Grow $W$ until $A^*$ stops changing within its error.

Any restriction keeps the bound. The restricted maximum satisfies
$A^*_W \le A^*$, so $k_W \ge k_\mathrm{opt} \ge k$. Restricting makes the
estimate less tight, never wrong, and enlarging $W$ checks how much was lost.

### 4.6 What rgmin needs

1. A Euclidean factor in the runtime product `RiemannianManifold`. Today it
   has only Stiefel and SPD factors. $\mathrm{SPD}(1)$ is the positive
   half-line with the affine-invariant metric, which is wrong for a signed
   offset. $\mathrm{St}(D,1)$ is the sphere.
2. Per-factor metric weights, and preferably a caller-supplied SPD
   preconditioner on each factor's tangent space, used by `egrad2rgrad` and
   by the trust-region inner product.
3. A sphere restricted to a linear subspace (project, then normalize), or a
   wrapper that takes an orthonormal basis of excluded directions. This
   covers the centre of mass, fixed atoms and the rotation subspace $W$.
4. A gauge hook. A callback after each accepted step re-centres the pivot.
   It changes the packed point without changing the plane.
5. Stochastic drivers:
   - Riemannian stochastic gradient ascent with a step schedule and iterate
     averaging;
   - norm-test control of the sample size;
   - a trust region that accepts an externally measured actual decrease
     with its standard error;
   - `NoiseFloor` driven by a caller-supplied covariance.
6. A Hessian-vector oracle that returns an estimate with an error bar, and
   an `hvp::Preconditioner` built from the harmonic model (Lanczos low rank
   plus a scalar).
7. A documented convention for sphere Hessians: the degree-zero extension,
   or the explicit Weingarten term.

## 5. Sampling in a hyperplane

### 5.1 Requirements

The sampler must draw from $\pi_\Sigma \propto e^{-\beta V}\,d\sigma$ on the
plane, with the centre-of-mass constraint. It must support many short
independent replicas, in the PAFI manner. It must accept starting
configurations mapped from a neighbouring plane. Each sample costs one force
call. In mass-weighted coordinates all the constraints are linear and
$|\nabla(n\cdot q)| = 1$, so the constraint never needs an iterative solve.
There is no Fixman or blue-moon correction either, because the mass-metric
tensor $\nabla\xi^\mathsf{T}M^{-1}\nabla\xi$ is constant
[carterConstrainedReaction1989; sprikFreeEnergy1998;
denOtterCalculationFree1998; ciccottiBlueMoon2005]. In Cartesian
coordinates that tensor is $1/\mu$ from (3.4), which is also constant.

### 5.2 Options and their errors

**Overdamped Euler–Maruyama** (the current `pafi.rs`, and PAFI). For an
in-plane harmonic mode of curvature $\lambda$ at unit mobility, the
stationary variance is

$$\langle r^2\rangle_{\Delta t} = \frac{kT}{\lambda(1-\lambda\Delta t/2)},
\tag{5.1}$$

which overestimates the variance at $O(\Delta t)$ and diverges as
$\lambda\Delta t\to2$. A simulation at $\lambda\Delta t = 0.05$ and 0.2 gave
1.0230 and 1.1107, against 1.0256 and 1.1111 from (5.1). The mean of a
linear observable is exact. The anharmonic part of the mean force, which is
the reason to sample at all, is not. A normal-force term
$\tfrac12 r^\mathsf{T}T_n r$ (third derivatives) is biased by the relative
amount $\lambda\Delta t/2$. That is the same order as the anharmonic signal
times the step. Stability needs $\Delta t < 2/\lambda_\mathrm{max}$, and
decorrelating the softest mode takes about
$\lambda_\mathrm{max}/\lambda_\mathrm{min}$ steps.

**MALA in the plane.** A Metropolis step makes the invariant measure exact
at the same cost per step. The step must shrink like $D^{-1/3}$ to keep the
acceptance rate [robertsOptimalScaling1998]. It is useful as an exactness
check.

**Projected BAOAB** (recommended) [leimkuhlerRationalConstruction2012]. Run
underdamped Langevin dynamics in the plane: project the forces and momenta
onto the tangent space and restore positions to the plane. For linear
constraints this is the geodesic BAOAB scheme
[leimkuhlerEfficientMolecular2016] with an exact constraint solve. Its
configurational sampling is exact for harmonic potentials at any stable step
$\omega\Delta t < 2$. A simulation at $\omega\Delta t = 0.5$, 1.0 and 1.5
gave $\langle q^2\rangle = 0.9996$, 1.0002 and 0.9997 against 1. For
anharmonic potentials the bias is $O(\Delta t^2)$ and comes only from the
anharmonic terms. With friction near critical for the softest in-plane
mode, decorrelation takes about
$(\lambda_\mathrm{max}/\lambda_\mathrm{min})^{1/2}$ steps. That is the
square root of the Euler–Maruyama cost.

**RATTLE and SHAKE** with nonlinear constraints
[ryckaertNumericalIntegration1977; andersenRattleVelocity1983], as in
rgmin's `holonomic` module, are not needed. Neither is hybrid Monte Carlo.

The recommended sampler is projected BAOAB in mass-weighted coordinates,
which means unit masses there. Choose the friction near critical for the
softest in-plane mode and the step from the stiffest one, with
$\Delta t \lesssim 1/\omega_\mathrm{max}$. Re-project positions onto the
plane every step to remove rounding drift. The BAOAB core in `bennett.rs`
generalizes directly once it carries masses.

### 5.3 Error analysis per plane

Each plane average has four error sources:

1. Discretization bias, $O(\Delta t^2)$ from anharmonic terms only. Check it
   with two step sizes, and extrapolate when it matters.
2. Statistical error. The standard error is
   $\sigma_f\sqrt{2\tau_\mathrm{int}/T_\mathrm{run}}$ for one replica. Block
   averages with blocks at least twice $\tau_\mathrm{int}$
   [flyvbjergErrorEstimates1989], which `pafi.rs` already uses, give it
   within a replica. The spread between independent replicas gives it
   across replicas, as in PAFI's ensemble estimate.
3. Initialization bias. Starting configurations come from the rigid map of
   a neighbouring plane's samples, so they are close to equilibrium. The
   overdamped plane dynamics contracts the relative entropy to
   $\pi_\Sigma$. If $\nabla^2_\parallel V \ge \lambda_\mathrm{min} > 0$ on
   the region visited, then
   $\mathrm{KL}(\rho_t|\pi_\Sigma) \le e^{-2\lambda_\mathrm{min}t}\,
   \mathrm{KL}(\rho_0|\pi_\Sigma)$ in the time units of
   $dq = -\nabla V\,dt + \sqrt{2kT}\,dW$
   [bakryDiffusionsHypercontractives1985; ottoGeneralizationInequality2000].
   The thermalization time therefore needs to exceed
   $\ln(\mathrm{KL}_0/\epsilon)/(2\lambda_\mathrm{min})$, with
   $\lambda_\mathrm{min}$ from the in-plane Lanczos run at the saddle.
   Section 12.1 derives from this a bound on the bias of the averages and
   a better starting distribution.
4. Metastability within the plane. The sampler can leave the channel: a
   different atom hops into the vacancy, or the system crosses into a
   symmetry-equivalent channel. For the Al adatom the 2001 paper saw escape
   within the plane during the hop progression. When the in-plane potential
   is not convex the entropy-contraction constant degrades exponentially,
   and this is that regime.

### 5.4 Channel restriction and escape detection

- After each replica, minimize within the plane and compare with the
  in-plane minimum found before sampling. Flag the replica when the largest
  per-atom displacement exceeds a threshold. The PAFI Python driver's
  default is 0.4 Å (`MaxJumpThresh`). It computes a resampling threshold
  (`ReSampleThresh`, default 0.5) but never uses it, and drops invalid
  samples at post-processing. The removed C++ driver resampled, and aborted
  when fewer than 10% of samples were valid.
- Count flagged replicas and report them. Do not silently drop them.
  Dropping conditions on the channel, which changes the ensemble. If the
  flagged fraction is above a few percent, shorten the replicas and use more
  of them, or add a soft in-plane restraint on the order parameter that
  separates the channels and reweight.
- For symmetric channels the 2001 paper restricts sampling to symmetry
  half-lines (its Section II D). The restraint and quench test above handle
  the same case without hand-built symmetry operations.

## 6. Estimators and variance reduction

### 6.1 Mean force and torque

Each sample needs one force call. It gives $f = n\cdot\nabla V(q)$ and the
torque sample $\tau = -(n\cdot\nabla V)(q - c)$, which is a vector in $U$.
Accumulate the sums and the replica-level means. Never average the
in-plane force components into the work (Section 3.3).

### 6.2 Stein control variates from the saddle Hessian

The plane measure satisfies a Stein identity. For any smooth in-plane
vector field $\varphi$ with enough decay,

$$\mathbb{E}_\Sigma[\nabla_\parallel\cdot\varphi]
= \beta\,\mathbb{E}_\Sigma[\varphi\cdot\nabla_\parallel V]. \tag{6.1}$$

This gives control variates with known means
[assarafZeroVariance1999; miraZeroVariance2013; oatesControlFunctionals2017].
Let $H_0$ be the zero-temperature saddle Hessian, $H_\parallel = PH_0P$ its
in-plane block, and $h = PH_0n$. Harmonically, $n\cdot\nabla V \approx
\mathrm{const} + h\cdot r$ and $P\nabla V \approx \mathrm{const} +
H_\parallel r$, with $r = q - c$. Then:

- With $\varphi = H_\parallel^{-1}h$, a constant field, the control variate
  $(H_\parallel^{-1}h)\cdot P\nabla V$ has mean zero. The corrected force
  estimator is
  $$\hat f = \overline{\,n\cdot\nabla V - (H_\parallel^{-1}h)\cdot P\nabla V\,}.
  \tag{6.2}$$
- With $\varphi = (h\cdot r)\,H_\parallel^{-1}e_j$, the control variate
  $(h\cdot r)(H_\parallel^{-1}e_j)\cdot P\nabla V$ has mean
  $kT(H_\parallel^{-1}h)_j$. The corrected torque estimator is
  $$\hat\tau = \overline{\,-(n\cdot\nabla V)\,r +
  (h\cdot r)\,H_\parallel^{-1}P\nabla V\,} - kT\,H_\parallel^{-1}h .
  \tag{6.3}$$

In a harmonic potential both corrected estimators have zero variance. The
cost is one linear solve with $H_\parallel$ per plane for (6.2). For (6.3)
it is one solve per batch: accumulate $\sum_s (h\cdot r_s)\,P\nabla V_s$ and
apply $H_\parallel^{-1}$ once. Use conjugate gradients on Hessian-vector
products at the saddle. The estimators stay exactly unbiased for any fixed
$H_0$. The quality of $H_0$ affects only the variance. $h$ and $H_0$ must
not depend on the samples they correct. The pivot $c$ should be the
previous round's centroid, or else the sample mean adds an $O(1/n)$ bias.

The check used a two-dimensional anharmonic saddle,
$V = \tfrac12q^\mathsf{T}Kq + 0.08q_1^3 + 0.05q_2^3 + 0.03q_2^4$ with
$K = ((-1, 0.4), (0.4, 2))$ and $kT = 0.3$, with exact quadrature on three
tilted, offset lines.

- The control variate means agree with (6.1) to $10^{-16}$.
- The force variance fell by factors of 350 to 700.
- The torque variance fell by factors of 6 to 180. The gain is smallest
  for strongly tilted planes, where the constant force terms dropped in the
  derivation matter.

The control variates matter most where $h$ is large: for planes away from
the optimum during optimization, and for PAFI planes away from the saddle,
where the path tangent is not an eigenvector of the local Hessian. At the
optimum $h\to0$, and the remaining variance is purely anharmonic. For the
PAFI planes, use the Hessian at the path point $X_0(\xi)$ instead of the
saddle Hessian.

### 6.3 Errors and the gradient covariance

- Use block averages within each replica and the spread between independent
  replicas. PAFI's ensemble estimator is the replica-level one.
- The torque is a vector in $U$, or in the subspace $W$. Estimate its
  covariance from replica means, at low rank when the replicas are fewer
  than the dimension. Feed the trace to `NoiseFloor` and the restricted
  covariance to the $\chi^2$ stopping test of Section 4.4.
- Common random numbers between neighbouring planes reduce the variance of
  differences and of curvature estimates (Section 4.3).

## 7. Free-energy differences, basin normalization and the rate

### 7.1 Path of planes and the barrier

$A$ is a function on the manifold of planes. The difference between two
planes therefore does not depend on the path between them, provided the
restricted ensembles change continuously along the path and stay in one
channel. The 2001 paper integrates along the path the optimizer happened to
take. Here the barrier is split into two parts, each computed along a path
chosen for accuracy:

1. **Reactant to saddle plane.** Use PAFI planes along the minimum energy
   path: the normal is the unit tangent and the pivot is the path point.
   Average (3.10), with the Stein control variates of Section 6.2 built from
   the Hessian at the path point. This is the existing `pafi.rs`, with the
   sampler of Section 5 and the error propagation below.
2. **Saddle plane to optimal plane.** Use the geodesic in $(n, d)$ between
   the two, or the optimizer's own iterates when the steps are small. Use
   (3.7) for thermodynamic integration and BAR between neighbouring planes.

The barrier is the sum, $A^* - A_R = [A(\Sigma_0^\ddagger) - A_R] +
[A^* - A(\Sigma_0^\ddagger)]$, and the second term is non-negative.

**Rigid maps between planes.** For planes $(c_a, n_a)$ and $(c_b, n_b)$,
let $R_{ab}$ be the minimal rotation taking $n_a$ to $n_b$. It acts in
$\operatorname{span}\{n_a, n_b\}$ and is the identity on the orthogonal
complement. With $\cos\theta = n_a\cdot n_b$ and
$u = (n_b - \cos\theta\, n_a)/|n_b - \cos\theta\, n_a|$,

$$R_{ab} = I + \sin\theta\,(u n_a^\mathsf{T} - n_a u^\mathsf{T}) +
(\cos\theta - 1)(n_a n_a^\mathsf{T} + u u^\mathsf{T}). \tag{7.1}$$

The map $T_{ab}(q) = c_b + R_{ab}(q - c_a)$ takes $\Sigma_a$ onto $\Sigma_b$
and preserves the area measure. Therefore

$$\frac{Z_H(\Sigma_b)}{Z_H(\Sigma_a)}
= \big\langle e^{-\beta[V(T_{ab}q) - V(q)]}\big\rangle_{\Sigma_a}, \tag{7.2}$$

exactly. BAR [bennettEfficientEstimation1976] combines the forward and
reverse work distributions. Over many planes, MBAR
[shirtsStatisticallyOptimal2008] combines them all. (7.2) is targeted free
energy perturbation [jarzynskiTargetedFree2002] with an exact
volume-preserving map. Eqs. (11)–(12) of the 2001 paper are its first-order
version, applied to the mean arm only. The affine extension of Section 12.2
adds an in-plane linear map that matches the harmonic covariances of the two
planes. After that, only anharmonic differences reduce the overlap.

**Errors.** For thermodynamic integration with weights $w_i$ over planes
sampled independently, $\mathrm{Var}(\Delta A) = \sum_i w_i^2\sigma_i^2$.
Add the covariances if neighbouring planes share random numbers. Estimate
the quadrature error by comparing the trapezoid rule with Simpson's rule or
a spline, and add planes where they disagree. The PRL adds samples where the
mean force is large. The PAFI Python driver integrates the variance as
$\sqrt{\int\sigma^2\,dr}$. That is larger than the weighted sum by about
$1/\sqrt{\Delta r}$, so do not copy it (Section 11.4). A bootstrap over
replicas is the simplest consistent alternative when splines correlate the
nodes. BAR gives its own asymptotic variance.

### 7.2 Basin normalization $Z_{H_R}/Z_R$

**The 2001 estimator.** Eq. (22) of the 2001 paper is
$Q_{Z_R}/Q_R = (\Delta t/t_\mathrm{tot})\sum_i 1/|(r_i - r_{i+1})\cdot n|$,
summed over the crossings of the plane $H_R$ by an unconstrained trajectory
in the basin. Each crossing contributes about $1/|v_n|$, and crossings occur
at a rate proportional to $|v_n|$, so the mean is right. The second moment
is $\int (1/v^2)\,|v|\,e^{-v^2/2kT}\,dv$, which diverges logarithmically at
$v\to0$. Only the finite time step keeps the estimator's variance finite,
and it grows like $\ln(1/\Delta t)$. The error bar from a few long runs
therefore understates the error.

**Replacement A: parallel planes** (recommended). By the co-area formula,
$Z_R = \int d\xi\, Z_H(c_R + \xi n_R, n_R)$ for any fixed unit $n_R$, up to
the part of $R$ beyond the dividing surface, which is of relative order
$e^{-\beta\Delta A}$. Hence

$$\frac{Z_{H_R}}{Z_R} = \Big[\int d\xi\,
e^{-\beta(A_R(\xi) - A_R(0))}\Big]^{-1}, \qquad
A_R'(\xi) = \langle n_R\cdot\nabla V\rangle_\xi . \tag{7.3}$$

$A_R(\xi)$ is nearly quadratic. Start from the harmonic value (Section 3.6),
sample five to nine parallel planes out to about $\pm4$ standard deviations
of $\xi$, and fit $A_R'$ with a low-order polynomial. The same sampler and
control variates apply. With $\xi_0$ the minimum of $A_R$, the Laplace
approximation of (7.3) turns (3.5) into

$$k \approx \frac{\omega_0}{2\pi}\, e^{-\beta(A^* - A_R(\xi_0))}, \qquad
\omega_0 = \sqrt{A_R''(\xi_0)}. \tag{7.4}$$

This gives PAFI's attempt frequency explicitly. Harmonically,
$\omega_0^2 = 1/(n_R^\mathsf{T}H_R^{-1}n_R)$, and (7.4) reduces to
Vineyard's rate. Use (7.3) itself, not (7.4), when $A_R$ is visibly
anharmonic.

**Replacement B: a density estimate.** $Z_{H_R}/Z_R$ is the marginal
density at zero of $\xi = n_R\cdot(q - c_R)$ in the basin. A kernel density
estimate from unconstrained basin dynamics has $O(h^2)$ bias and
$O(1/(nh))$ variance for bandwidth $h$, so its variance is finite. It needs
a separate unconstrained run and is a useful cross-check.

**Replacement C: adaptive biasing force along $\xi = n_R\cdot q$.** For a
linear coordinate with unit $n_R$, the ABF instantaneous force reduces to
$-n_R\cdot\nabla V$, with no divergence term
[comerAdaptiveBiasing2015; lelievreLongtimeConvergence2008]. One ABF run
along $\xi$ therefore gives the whole profile $A_R(\xi)$, by the same
estimator as the constrained planes. ABF samples without a constraint, so
it cannot apply the in-plane channel restriction of Section 5.4. That is
harmless inside the basin and is not harmless near the barrier. Lelièvre,
Rousset and Stoltz bound the convergence rate of ABF by the log-Sobolev
constant of the measures conditioned on $\xi$, the same in-plane mixing
constant that controls the plane sampler (Section 5.3).

### 7.3 Rate and error budget

$$k = \kappa\left(\frac{kT}{2\pi}\right)^{1/2}\frac{Z_{H_R}}{Z_R}\,
e^{-\beta(A^* - A_R)} \tag{7.5}$$

is in the time unit of the mass-weighted coordinates. The three statistical
factors are independent:

$$\mathrm{Var}(\ln k) = \beta^2\,\mathrm{Var}(A^* - A_R) +
\mathrm{Var}\big(\ln(Z_{H_R}/Z_R)\big) + \mathrm{Var}(\ln\kappa).
\tag{7.6}$$

Report $\ln k$ with its standard error and each component separately: the
PAFI part of the barrier, the optimization gain, the basin term and
$\kappa$. Report the plane uncertainty from Section 4.4 as well. The bound
$k \le k_\mathrm{TST}$ holds for the exact averages. A noisy estimate can
violate it by about its error, so a check against MD is two-sided at the
level of the error bars.

## 8. Dynamical correction

The transmission coefficient follows Bennett and Chandler
[bennettMolecularDynamics1977; chandlerStatisticalMechanics1978]:

$$\kappa = \lim_{t\to\infty} \frac{\langle (n\cdot v_0)\,
\theta(n\cdot(q_t - c))\rangle_{\Sigma^*}}{\langle (n\cdot v_0)_+
\rangle_{\Sigma^*}}, \tag{8.1}$$

where trajectories start from decorrelated samples of the optimal plane,
with Maxwellian velocities. Vanden-Eijnden and Tal write it as a
flux-weighted average of products of committors, their Eq. (67). The rate
does not depend on the dividing surface but $k_\mathrm{TST}$ does, so "the
variational TST dividing surface which minimizes $\nu_S^\mathrm{TST}$ also
maximizes $\kappa_S$". The optimal plane therefore gives the $\kappa$
closest to one among planes. With $N_\mathrm{traj}$ trajectories the
relative error of $\kappa$ is $\sqrt{(1-\kappa)/(N_\mathrm{traj}\kappa)}$
(their Eq. 76).

Implementation notes:

- `bennett_chandler` in `src/bennett.rs` already integrates BAOAB
  trajectories with a scalar mass and detects the plateau. In mass-weighted
  coordinates the mass is one. Pass the optimal plane as the surface and
  the plane sampler's output as the starting configurations.
- The dynamics must be the dynamics of the comparison. For a comparison
  with NVE molecular dynamics, use Newtonian trajectories. For a Langevin
  thermostat, use the same friction, since $\kappa$ depends on it
  [kramersBrownianMotion1940].
- Assign the end state by quenching: steepest descent to a minimum, then
  matching against the known minima with the permutation-invariant
  distance (12.5). Do not classify from instantaneous positions; at 900 K
  they fluctuate too much to classify reliably.
- When recrossings are many, count only the first crossing out of the
  reactant region, as in the effective positive flux of transition
  interface sampling [vanErpNovelPath2003]. The positive-flux formulation
  goes back to Anderson [andersonStatisticalTheories1973]. For Bligaard and
  Jónsson's model, Makarov and Metiu found trajectories that recross
  "thousands of times" at the higher temperature, while the TST rate is off
  by only a factor of two [makarovReactionRate1997], as quoted by Bligaard
  and Jónsson.
- $\kappa \le 1$, so recrossing can only lower the rate. If MD is faster
  than $\kappa\,k_\mathrm{OH}$, the cause is in the free energy (sampling,
  potential, cell), in extra channels, or in how the MD counts correlated
  jumps. It is never in $\kappa$.
- **Small $\kappa$.** When $\kappa$ is small, (8.1) needs many long
  trajectories. Mökkönen, Ala-Nissilä and Jónsson use a sequence of planes
  parallel to the transition-state plane, out to near the final state
  [mokkonenEfficientDynamical2016]. Configurations are equilibrated in
  the transition-state plane and given Maxwellian velocities, and
  $\kappa = \prod_i P(\lambda_{i+1}|\lambda_i)$. For polymer escape at
  $T = 1.0$ (their Table 1), the costs in force evaluations were:

  | Chain length | Forward-flux correction | Voter–Doll correction | Direct Langevin dynamics |
  |---|---|---|---|
  | $N = 8$ | 4% error, $8.0\times10^6$ | 3%, $8.0\times10^7$ | 6%, $6.4\times10^9$ |
  | $N = 64$ | 3%, $2.88\times10^{10}$ | 9%, $2.6\times10^{11}$ | 10%, $3.0\times10^{13}$ |

  Planes parallel to the optimal plane use the same sampler. This is an
  option behind the same interface as `bennett_chandler`.
- **Consistency test.** The exact rate does not depend on the dividing
  surface. $\kappa_S\,k_\mathrm{TST}(S)$ is therefore the same for every
  plane $S$ that separates the states. Computing both factors on the
  saddle plane and on the optimal plane tests the free energies, the flux
  factor and $\kappa$ together.

  The same identity makes maximizing $\kappa$ over planes equivalent to
  minimizing $k_\mathrm{TST}$. Zinovjev and Tuñón optimize a hyperplane in
  collective-variable space by maximizing the transmission coefficient
  [zinovjevTransitionState2015]. For the Eckart barrier, the 2001 paper
  notes that the plane with the fewest recrossings "is clearly different
  from the plane with maximum free energy" (its Fig. 10 caption). A count
  of recrossings is not $\kappa$, so this does not contradict the
  identity.
- **Diffusive crossings.** Take overdamped dynamics with a constant
  diffusion matrix $D$ in mass-weighted coordinates. The reactive flux is
  the minimum of the Dirichlet form $\int e^{-\beta V}\nabla u^\mathsf{T}D
  \nabla u\,dq/Z$ over trial committors $u$, so every trial function bounds
  the rate from above [petersenCommittorsReaction2026, Eq. 5].

  Restrict $u$ to functions of $n\cdot q$, with planes at $b_R$ and $b_P$
  bounding the reactant and product regions. The optimal profile has
  $du/db \propto e^{\beta A(n,b)}$, and

  $$k \le \frac{n^\mathsf{T}Dn}{Z_R\int_{b_R}^{b_P} e^{\beta A(n,b)}\,db},
  \tag{8.2}$$

  with $Z_R$ as in (3.3). Like (3.3), this bounds the rate for every $n$.
  Unlike (3.3), it holds only for overdamped dynamics, where TST
  overestimates by about $\gamma/\omega_b$.

  For a sharp barrier along $b$ with curvature $-\omega_b^2$ and
  $D = (kT/\gamma)I$, (8.2) reduces to $\omega_b/\gamma$ times
  $k_\mathrm{TST}$ through the same plane. That is Kramers' high-friction
  result [kramersBrownianMotion1940]. (8.2) needs the profile $A(n,b)$ on
  parallel planes, which the plane sampler provides, and minimizing it
  over $n$ is the overdamped analogue of OH-TST.
- **Committor-likelihood check.** The trajectories of (8.1) also give
  committor outcomes for their starting points. Peters and Trout fit
  $p_B = \sigma(w\cdot q + w_0)$ to such outcomes by maximum likelihood,
  for linear combinations of collective variables
  [petersObtainingReaction2006; petersExtensionsLikelihood2007]. Lechner
  et al. use the same linear model as the baseline for nonlinear
  coordinates [lechnerNonlinearReaction2010]. Best and Hummer instead
  optimize the direction of a fixed-norm weight vector by the peak of
  $p(\mathrm{TP}|r)$ [bestReactionCoordinates2005].

  Points on one plane determine only the in-plane part $Pw$, because
  $n\cdot q$ is constant there. If $Pw$ is zero within its error, the
  committor does not vary on the plane, as on an isocommittor. A
  significant $Pw$ gives the direction, though not the angle, in which to
  rotate the plane toward an isocommittor. Starting points on the parallel
  planes of the forward-flux scheme determine all of $w$.

  The two normals need not agree. OH-TST minimizes the one-way flux, not
  the committor mismatch, and the difference measures how far the
  optimal plane is from an isocommittor.

## 9. Anharmonicity and thermal expansion: the Cu vacancy hop at 700 K and 900 K

### 9.1 Inputs

Nothing in this section is specific to Cu. The inputs are:

- a potential, through the `PointSurface` interface;
- a cell with one vacancy and its cell matrix $h(T)$ at each temperature;
- the masses and the temperature.

The cell at temperature comes from the caller, for example an NPT run with
the same potential, or a fitted $a(T)$ applied as a uniform scaling. PAFI uses
the per-axis form $a(T) = a_0(1 + \alpha_1 T + \alpha_2 T^2)$, with
parameters `LinearThermalExpansion` and `QuadraticThermalExpansion`. The
same form serves here.

The repository holds NPT lattice parameters for the Mishin potential at
300–700 K, in the notes of `tests/mishin_cu.rs`, and the Table 1 values at
300, 500 and 700 K in `benchmarks/defect_pair_network_notes.txt`. It has no
900 K value. That value must be measured with the same potential before the
900 K calculation. This document does not supply one.

### 9.2 Procedure

1. Scale the perfect-crystal positions affinely to $h(T)$ and create the
   vacancy.
2. Relax the reactant and product at fixed $h(T)$. Find the saddle (CI-NEB
   followed by a dimer or Lanczos refinement), the mass-weighted unstable
   mode and the Vineyard rate, all at $h(T)$. Repeat at $h(0)$ for the
   reference.
3. Compute the PAFI profile from the reactant plane to the saddle plane at
   $T$ and $h(T)$ (Section 7.1, part 1).
4. Optimize the plane at $T$, starting from the saddle plane (Section 4.4),
   with rotation restricted to the moving atoms and their near shells
   (Section 4.5).
5. Compute the basin normalization by parallel planes (Section 7.2).
6. Compute $\kappa$ from the optimal plane with the comparison dynamics
   (Section 8).
7. Assemble (7.5). For a nearest-neighbour vacancy hop in fcc there are 12
   equivalent channels, so the vacancy jump frequency is $12k$. Compute one
   channel. Keep the sampler in it with the quench test, not with hand-built
   symmetry half-lines.

### 9.3 Decomposition against Vineyard

Write $k_V(h)$ for the Vineyard rate at cell $h$. Then

$$\ln\frac{k}{k_V(h(0))} = \underbrace{\ln\frac{k_V(h(T))}{k_V(h(0))}}_{\text{thermal expansion}} +
\underbrace{\ln\frac{k_\mathrm{PAFI}(h(T))}{k_V(h(T))}}_{\text{anharmonic planes}} +
\underbrace{\ln\frac{k_\mathrm{OH}(h(T))}{k_\mathrm{PAFI}(h(T))}}_{\text{orientation}\ \le\ 0} +
\ln\kappa . \tag{9.1}$$

Here $k_\mathrm{PAFI}$ is (3.3) through the maximum of the PAFI profile,
with the basin term from (7.3). Each term answers a separate question:

1. How much of the discrepancy with MD is the lattice parameter alone, with
   all vibrations still harmonic?
2. How much is explicit anharmonicity of the planes? This is the term PAFI
   measures.
3. How much does finite temperature move the best plane? This term is never
   positive, and Section 3.6 predicts it is small for a weakly anharmonic
   hop.
4. How much is recrossing? $\ln\kappa \le 0$.

Terms 3 and 4 can only lower the rate. If MD is faster than Vineyard, the
first two terms must carry the difference. If they do not, the free energy
calculation or the MD rate needs re-examination.

### 9.4 Fixed cell or fixed pressure

At fixed $h(T)$ the cell cannot relax to the activation volume
$\Delta V^\ddagger$. In linear elasticity the fixed-volume barrier exceeds
the fixed-pressure one by about $B(\Delta V^\ddagger)^2/(2V)$, with $B$ the
bulk modulus. That vanishes like $1/N$. Check it with two cell sizes, or
apply the elastic correction. The MD to compare with must use the same
ensemble: NVT at the same $h(T)$, or NPT if the correction is applied.

### 9.5 Classical statistics

The comparison is classical OH-TST against classical MD with the same
potential. Quantum vibrational effects are absent on both sides and cannot
explain a difference between them. They matter only for comparison with
experiment. Namakian et al. restrict their Cu comparison with experiment to
temperatures above the Debye temperature for this reason
[namakianTemperatureDependent2023].

### 9.6 Literature context

The statement that harmonic TST is slow against MD for vacancy hops at
these temperatures traces to one sentence. Sørensen and Voter compared HTST
with direct MD for several processes with EAM potentials, including the
"bulk vacancy hop in Ag and Cu". They found the MD rate constants "at
suitable temperatures (700–1000 K) are within a factor of 2 of the HTST
rate constants" [sorensenTemperatureAccelerated2000, p. 9603].

They show no data for the bulk cases and do not state the cell, the
lattice-parameter protocol or which rate is larger. Their Fig. 4(b), for an
Ag(100) adatom, plots the ratio of the MD to the HTST rate constant from
500 K to 1000 K. Read from the figure, every point lies above one and the
ratio grows with temperature. The text gives HTST within 60% for the hop,
and off by a factor of 4 for exchange at 1000 K. Sørensen, Mishin and Voter
restate the result as deviations "by up to a factor of 2–4" at
700–1000 K, again without a direction
[sorensenDiffusionMechanisms2000, p. 3671].

No primary source found tabulates direct MD against Vineyard rates for the
bulk Cu vacancy as a function of temperature. Sivak et al. report an
Arrhenius migration energy of 0.73 eV from MD at 750–1300 K, with no
harmonic comparison [sivakDiffusionCharacteristics2022]; only the abstract
was read. The calculation of Section 9.2 therefore has no published
counterpart, and its MD side has to be run.

Other results bear on the expected size and sign:

- **Thermal expansion.** Mishin et al. report, for their EAM1 potential, a
  migration energy of 0.689 eV, a Vineyard prefactor of 7.69 THz and a
  migration volume $V_m = 0.107\,V_0$
  [mishinStructuralStability2001, Table IV]. Since
  $\partial G_m/\partial p = V_m > 0$, lattice expansion lowers the
  barrier, so the thermal-expansion term of (9.1) should be positive. Its
  size needs $a(T)$ for the potential, which that paper shows only as a
  figure (Section 9.1).

  Sørensen and Voter call thermal expansion "a special kind of anharmonic
  effect". They suggest expanding the cell for MD and contracting it for
  minimizations and saddle searches. The decomposition (9.1) is the
  consistent version of that: every term at the same $h(T)$, and the 0 K
  cell only in the first term.
- **Published prefactors disagree.** Sørensen, Mishin and Voter list a
  vacancy migration energy of 0.690 eV and a prefactor of 5.3 THz
  [sorensenDiffusionMechanisms2000, Table I]. That energy is the EAM2 value
  of Mishin et al., whose EAM2 prefactor is 16.22 THz. This survey did not
  reconcile the two. The Vineyard reference should be recomputed with the
  potential and cell in use, not quoted.
- **Anharmonicity of the vacancy.** For Cu with density-functional theory,
  Glensk et al. find that the vacancy formation entropy rises almost
  linearly with temperature, from explicit anharmonicity. Arrhenius
  extrapolation then misplaces the 0 K formation energy by 0.22 eV
  [glenskBreakdownArrhenius2014].

  Foiles found that the quasi-harmonic approximation underestimates the
  temperature dependence of the vacancy formation free energy in EAM Cu,
  significantly from about 600 K [foilesEvaluationHarmonic1994]. That
  paper was not accessed and is known through the quotation in
  [sorensenDiffusionMechanisms2000]. No migration counterpart for Cu was
  found. The W result of Section 11.5 is the nearest.
- **Full TST against MD in Cu.** The only Cu case found where an
  anharmonic TST rate is compared with MD is the adatom on Cu(100) with
  an EAM potential [boisvertSurfaceDiffusion1998]. Thermodynamic
  integration of the mean force along the path gives activation energies
  and prefactors that agree with MD within the stated uncertainties. For
  the jump, the activation energies are 0.51(2) eV against 0.49(1) eV,
  and $\ln\Gamma_0$ ($\Gamma_0$ in THz) is 2.9(2) against 3.0(2).

  They report $\kappa \approx 0.9$ for jumps and 0.6 for exchanges. They
  also note that a harmonic calculation with the same potential gave a
  jump prefactor ten times smaller.
- **Recrossing.** DeLorenzi, Flynn and Jacucci find at most about 5%
  return jumps for models of Ag, Al, Ar and Cu, even near melting. They
  describe the watershed saddle surface as "curved rather than planar"
  [delorenziEffectAnharmonicity1984]; only the abstract was read.
  $\kappa$ close to one is expected for the vacancy hop, and $\kappa$
  cannot make MD faster than TST (Section 8).

  Paci and Ciccotti computed vacancy migration rates in a Lennard-Jones
  crystal by reactive flux, with a holonomic constraint on the reaction
  coordinate [paciVacancyMigration1992]; only the abstract was read.
- **Boundary conditions.** For Voter's EAM Cu, the harmonic vacancy
  formation entropy is $1.231\,k_B$ at zero pressure and $2.969\,k_B$ at
  constant volume, both supercell values
  [mishinCalculationPointDefect2001, Table 5]. The authors expect
  migration entropies to be less sensitive, because the number of atoms is
  conserved. This supports the requirement of Section 9.4 that MD and
  OH-TST use the same ensemble.
- **The MD rate.** Sørensen and Voter list correlated double jumps as
  outside TST [sorensenTemperatureAccelerated2000, p. 9605]. The MD rate
  should count net site changes, with rapid double jumps counted
  separately.

Together these suggest that MD should be faster than Vineyard at $h(0)$ by
up to about a factor of 2 at 700–900 K for an EAM potential. Thermal
expansion and the anharmonic migration free energy would both contribute,
with $\kappa$ close to one. A much larger gap would point to an error in
the MD rate or in the cell. This is an expectation drawn from the sources
above, not a published result.

## 10. Test problems and reference values

All values below were recomputed for this document unless they are marked
as published. Recomputed values come from adaptive quadrature with relative
tolerance $10^{-11}$ to $10^{-13}$. Published values are quoted with their
source.

### 10.1 Quadratic saddle (closed form)

Use (3.14)–(3.15) in any dimension. Example: $D = 3$,
$H = \operatorname{diag}(-1, 1, 4)$, $kT = 0.1$. The optimum is $n = e_1$,
$d = 0$, with $\partial^2 A/\partial d^2 = -1$ and rotation curvatures
$-0.2$ (towards $e_2$) and $-0.125$ (towards $e_3$). With a harmonic
reactant, (7.5) with $\kappa = 1$ must equal Vineyard's rate. The tests:

- the gradient estimators are unbiased, and the control variates give zero
  variance in this potential;
- projected BAOAB samples the plane exactly at any stable step;
- the optimizer converges from a tilted, offset start, for example 20°
  towards $e_2$ and $d = 0.3$;
- the noise-floor stop triggers;
- the curvatures above come out of (3.12).

### 10.2 Eckart barrier coupled to an oscillator

This is the model of Bligaard and Jónsson (their Eq. 1), also Eq. (25) of
the 2001 paper and studied by Makarov and Metiu:

$$V(x,y) = \frac{V_0}{\cosh^2(\alpha x/2)} + \tfrac12\mu\omega^2(y - Cx)^2,
\qquad V_0 = 0.0156,\ \alpha = 3.97,\ \tfrac12\mu\omega^2 = 1.04\times10^{-4},
\ C = 10.$$

$E_a = V_0$. The lines pass through $(0,0)$, and $\theta$ is measured from
the $y$ axis. Bligaard and Jónsson give 1.16° for the saddle line. The
recomputed zero-temperature limit is 1.1638°.
$A - A_\mathrm{ref}$ is relative to their reference line at
$-\arctan(1/10)/2$, in units of $E_a$.

| $kT/E_a$ | Published maxima (Bligaard–Jónsson) | Recomputed maxima | Recomputed $A - A_\mathrm{ref}$ |
|---|---|---|---|
| 0.01 | 1.24° | 1.170° | 0.46334 |
| 0.1 | — | 1.226° | 0.55289 |
| 0.1182 | secondary maximum appears at 43° | 1.239°, 43.435° | 0.57187, 0.37167 |
| 0.2266 | 1.33° and 71.2°, equally high | 1.327°, 71.795° | 0.68730, 0.68699 |
| 0.3 | 1.43° and 75.6° | 1.405°, 75.602° | 0.76509, 0.90532 |
| $\to\infty$ | 84.29° | — | — |

- At $kT/E_a = 0.1180$ only one maximum exists, so the secondary maximum
  appears between 0.1180 and 0.1182, as published.
- The recomputed maxima are equally high at $kT/E_a = 0.22676$, at 1.328°
  and 71.807°.
- At 0.3 the high-angle maximum is higher by $0.1402\,E_a = 0.467\,kT$.
- At 0.01 the free energy is flat to $3\times10^{-4}\,kT$ between 1.17° and
  1.24°. A small integration error moves the published peak that far.
  Bligaard and Jónsson integrated out to $V_0 + 22\,kT$. The recomputation
  goes to $V_0 + 40\,kT$.
- A second-order Laplace expansion in $kT$ gives 1.1696° at 0.01 and
  1.2316° at 0.1, consistent with the quadrature.
- Published dynamical results: Makarov and Metiu found the smallest TST
  rate at about 1° for $kT = 0.1E_a$ and 70° for $kT = 0.3E_a$, as quoted by
  Bligaard and Jónsson. The 2001 paper quotes 0° and about 70°. The 2001
  OH-TST runs through $(0,0)$ gave 1° and 74°.

This model tests the multiple-maxima logic of Section 4.4 and the
first-order switch of the optimal plane.

### 10.3 LEPS plus harmonic oscillator

This is the potential of the 2001 paper, its Appendix B:

- LEPS parameters $a = 0.05$, $b = 0.80$, $c = 0.05$,
  $d_{AB} = d_{BC} = 4.746$, $d_{AC} = 3.445$, $r_0 = 0.742$,
  $\alpha = 1.942$;
- $V(x,y) = V_\mathrm{LEPS}(x, r_{AC} - x) + 2k_c(x - (r_{AC}/2 - y/c))^2$
  with $r_{AC} = 3.742$, $k_c = 0.2025$, $c = 1.154$;
- no mass or physical unit is specified in the paper.

Recomputed stationary points:

| Point | $(x, y)$ | $V$ | Hessian eigenvalues |
|---|---|---|---|
| saddle | (2.02083, −0.17290) | −0.87522 | −8.0027, 0.66545 |
| deep minimum | (0.74152, 1.30342) | −4.50918 | 0.59393, 35.0438 |
| shallow minimum | (3.00128, −1.30434) | −2.62029 | 0.58397, 20.9082 |

The barrier is $E_A = 3.63395$ from the deep minimum and 1.74506 from the
shallow one. The first table is the independent low-temperature check at
$kT = 0.1E_A$, in units with unit mass and the potential's own energy and
length units.

| Reactant | $k_\mathrm{HTST}$ | TST, saddle line | TST, optimal line | Optimal line relative to saddle line |
|---|---|---|---|---|
| deep | 4.0410e−5 | 3.8594e−5 | 3.8589e−5 | offset −0.00092, tilt −0.257° |
| shallow | 3.0951e−5 | 2.9846e−5 | 2.9845e−5 | offset −0.00043, tilt −0.125° |

**Fixed orientation.** The fixed normal is the guideline from reactant to
product, which is 44.43° from the unstable mode. Barriers are measured from
the line through the reactant normal to the guideline.

- From the deep minimum: $\beta\Delta A$ is 8.308 for the optimal line and
  5.734 for the best fixed-orientation line, at $s = 0.692$. The fixed
  barrier is 31.0% lower, and its rate is 13.1 times larger.
- From the shallow minimum: 8.540 against 2.718, 68.2% lower.
- The 2001 paper reports "about 25% lower, corresponding to a 20 times
  larger rate" (published), without the exact orientation or reactant
  plane.

**Basin normalization.** $Q_{Z_R}/Q_R$ for lines through the reactant,
with the harmonic value and the quadrature value in units of inverse
length:

| Reactant | Guideline normal | Soft-mode normal | Stiff-mode normal |
|---|---|---|---|
| deep | 0.6594, 0.6516 | 0.5100, 0.5080 | 3.9177, 3.7495 |
| shallow | 0.9297, 0.8266 | 0.7298, 0.8072 | 4.3668, 3.8018 |

**Section III B reproduction.** The paper gives
$(4.5\pm0.2)\times10^{-7}$ for OH-TST, $(4.4\pm0.1)\times10^{-7}$ from
hyperdynamics, $4.3\times10^{-7}$ from harmonic TST and
$\langle|v|\rangle=0.399$, with no units. Although the same paragraph says
$k_BT=0.1E_A$, the latter two numbers uniquely reproduce at $k_BT=0.25$ in
the Appendix B energy units with unit mass: $\langle|v|\rangle=0.398942$ and
$k_\mathrm{hTST}=4.33127\times10^{-7}$. At that convention:

| Estimate | Rate |
|---|---|
| analytic Vineyard hTST | $4.33127\times10^{-7}$ |
| optimal-line quadrature | $4.20148\times10^{-7}$ |
| Eq. (22) crossings plus trapezoidal work, 8 independent seeds | $(4.340\pm0.132)\times10^{-7}$ |
| paper | $(4.5\pm0.2)\times10^{-7}$ |

The sampled estimate uses 1000 records for each plane optimization, 8000
records for Eq. (22), and 2000 records on each of 17 work planes. The paper
states only that canonical averages use one molecular-dynamics trajectory
with stochastic collisions; it does not report a time step, collision
frequency, trajectory length, number of planes, masses, or physical units.
The test therefore states its sample counts rather than attributing them to
the paper. It keeps the paper's crossing estimator and trapezoidal
integration; Simpson quadrature gives $4.211\times10^{-7}$ and exposes the
small upward discretization bias that puts the paper-style estimate inside
the printed band.

The apparent $Q^\ddagger/Q_{Z_R}=0.765$ is a label error: Eq. (3) needs
$Q_{Z_R}/Q_R$ in the prefactor, while $Q^\ddagger/Q_{Z_R}$ is the
exponentially small work factor. This model also tests the
PAFI-plus-optimization path on a twisted valley. The tilt of the optimal
line is small here, so most of the gain over a fixed orientation comes from
following the path, not from the finite-temperature optimization.

### 10.4 System–bath models with exact references in high dimension

The Eckart model is a one-dimensional barrier coupled linearly to one
harmonic bath mode. Use a low-dimensional system $s$ (one or two
coordinates) and many bath modes,
$V = V_s(s) + \sum_k \tfrac12\omega_k^2(z_k - c_k\cdot s)^2$. The plane
integral stays exact for any normal. For fixed $s$, the plane constraint
leaves the bath on an affine subspace on which the bath energy is quadratic.
Its Gaussian integral is closed-form, and the remaining integral over $s$ is
one- or two-dimensional quadrature. This gives exact references for $A$, its gradient and Hessian,
and the optimal plane in hundreds of dimensions with an anharmonic system
part. It is the main correctness test for the high-dimensional optimizer,
the subspace restriction and the control variates. Section 15 schedules
it.

With a parabolic barrier for $V_s$ the potential is quadratic. The plane of
maximal $A$ is normal to the unstable mode of the mass-weighted Hessian
(Section 10.1). The normal coordinate along that mode evolves independently
under Newtonian dynamics, so no trajectory recrosses the plane and
$\kappa = 1$ exactly. That tests the $\kappa$ machinery of Section 8 in high
dimension.

This is the finite-bath form of Pollak's result that the Kramers–Grote–Hynes
rate is the continuum limit of multidimensional harmonic TST
[pollakTheoryActivated1986]. For an anharmonic system in the continuum
(generalized Langevin) limit, Berezhkovskii, Pollak and Zitserman give the
optimal planar dividing surface explicitly, and it depends on temperature
[berezhkovskiiActivatedRate1992]. Only the abstracts of these two papers
were read, so neither supplies a reference value here.

### 10.5 Al adatom on Al(100) (published benchmark)

The 2001 paper, Section III D, used an EAM of the Voter–Chen form
[voterAccurateInteratomic1986], for which it cites Chen and Voter
[chenReconstruction1991]: 201 atoms, the two bottom layers fixed, 210 K.
Published values:

- energy barriers 0.23 eV (exchange) and 0.37 eV (hop);
- OH-TST free-energy barriers 0.16 eV and 0.33 eV, that is entropic
  reductions of 30% and 11%;
- quenches from the optimal plane reached the exchange product in 38% of
  cases (62% returned to the initial state);
- the hop plane, found with a large inertia parameter, gave 41% and 59%;
- normal components on the leading atoms: 0.667, 0.592, 0.108, … for the
  exchange and 0.973, 0.078, 0.078, … for the hop.

Feibelman's density-functional result motivates the exchange mechanism
[feibelmanDiffusionPath1990]. This is a literature benchmark. It needs the
Voter–Chen Al potential, and its free-energy barriers come from one group's
sampling with unstated error bars, so it is not a CI test.

### 10.6 Cu vacancy hop against MD

This is the integration test of Section 9: Mishin EAM Cu at 700 K and
900 K, compared with direct MD rates computed with the same potential, cell
and ensemble. Section 9.6 lists the literature context.

## 11. What to take from PAFI

PAFI [swinburneUnsupervisedCalculation2018] is the closest existing method
and the only one with maintained code. The sources read are:

- the PRL (HAL deposit of the APS version);
- Swinburne's review [swinburneUncertaintyAnharmonicity2021];
- the error-analysis note in the code repository;
- the applications listed in Section 11.5;
- the code at the revisions below.

| Code | Revision |
|---|---|
| Python driver (`tomswinburne/pafi`, `main`) | 1661a1d |
| C++ driver (removed in cfe812d) | 427b0af |
| ASE branch | e2c53d0 |
| LAMMPS `fix pafi` | 582a54b |

### 11.1 What PAFI computes

PAFI samples a fixed family of hyperplanes, each normal to a splined
zero-temperature minimum energy path. The planes are not updated with
temperature: "configurations … are sampled at all temperatures from the
same hyperplane, which is perpendicular to the initial NEB path"
[alleraActivationEntropy2025]. Every code version averages only
$\langle\psi\,\partial_\xi X_0\cdot\nabla V\rangle$: the Python driver, the
C++ driver, the ASE branch and `fix pafi` (output element 0, $\psi\,f\cdot
n$). This is (3.10), the exact derivative of the area-measure free energy
of the plane (Section 3.2). Two consequences follow:

1. The maximum of the integrated PAFI profile is the variational optimum
   within the one-parameter family of planes normal to the path, an
   optimal-hyperplane calculation restricted to one dimension.
2. The PRL's Eq. (1) is a different quantity. It equals the plane free
   energy minus $kT\ln\langle|\psi|\rangle$ minus
   $kT\ln|\partial_\xi X_0|$, by the co-area formula (Section 3.2).

The printed PRL contains a sign error. Eqs. (3) and (8) carry the
$\beta^{-1}$ divergence term with a plus sign. The standard identity has a
minus sign [comerAdaptiveBiasing2015, Eqs. 5, 6 and 13], and the PAFI
report's two-dimensional check agrees with the minus sign. Crossref lists no
erratum. No code implements the term, so no published PAFI number is
affected. Do not port Eq. (8) literally.

PAFI computes no rate. No code version computes an attempt frequency, flux
factor or reactant partition function. The applications either fit the
prefactor to MD or assume it cancels:

- the PRL's tungsten "boost" is $\exp(-\beta[\Delta\mathcal F -
  \Delta\mathcal F_\mathrm{harm}])$;
- Sato et al. fit $t_0 = 0.025$ ps to MD;
- Allera et al. fit $\nu = 3.8\times10^9$ Hz for harmonic TST and
  $9.2\times10^{10}$ Hz for their variational harmonic TST, and call $\nu$
  "the only fitting parameter".

Section 7.2 closes this gap: the basin term (7.3) and the attempt frequency
(7.4) follow from the same plane sampler.

### 11.2 What to take

1. **The path of planes as stage one.** The maximum of the PAFI profile is
   a better starting plane than the saddle plane when the free-energy
   landscape along the path changes with temperature. For their Fe kink
   pair, Allera et al. evaluate their variational harmonic theory at the
   configurations that set the barrier, $z_2 \approx 0.7$ and, above $T_0$,
   $z_1 \approx 0.3$. For EAM Fe the profile minimum leaves $z = 0$ above
   200 K. `pafi.rs` already provides the profile.
2. **Independent short replicas.** Each plane gets many short, independent
   replicas, each started on the path point, thermalized and then averaged.
   - The PRL used 1000–4000 thermalization steps, 2000–5000 sampling steps
     and 100–1000 replicas per plane, and found 100–500 typically
     sufficient.
   - Swinburne's error-analysis note takes the error from the spread of
     replica means (its Eq. 8): $\hat\sigma^2 = \sum_j(\langle f\rangle_j -
     \hat\mu)^2/N_E^2$.
   - Independent replicas need no autocorrelation analysis and parallelize
     trivially. `average_hyperplane` currently runs one chain per plane and
     takes the error from block means, with blocks at least twice the
     integrated autocorrelation time. Replicas complement that, and they
     also give the escape statistics of item 3.
3. **The escape test.** After sampling, minimize within the plane and
   compare with the reference in-plane minimum (Section 5.4).
4. **Thermal expansion by uniform cell scaling**, with the polynomial
   $a(T)$ of Section 9.1. Allera et al. checked expansion separately and
   found "only a slight change" for their Fe kink pair, using
   $a_0(T) = a_0(0)(1 + 1.093\times10^{-5}T + 5.117\times10^{-9}T^2)$.
   That check is specific to their system.
5. **Adaptive placement of planes.** Add planes where the mean force
   changes rapidly. The PRL did this by hand, and Sato et al. note that
   "the high number of hyperplanes could have been reduced by a more
   judicious choice of the hyperplane positions".

### 11.3 What not to copy

1. The printed Eq. (8), for the sign error above.
2. Fixed planes at all temperatures. The optimization of Section 4 is what
   this design adds.
3. Prefactors fitted to MD. Use (7.3)–(7.5).
4. $\Delta\mathcal F = \max\mathcal F - \min\mathcal F$ along the path as
   the barrier in $k = \omega_0 e^{-\beta\Delta\mathcal F}$. The profile
   minimum is not the reactant partition function. The correct statement is
   (7.4), with $\omega_0$ and $A_R(\xi_0)$ from parallel planes through the
   reactant.
5. Overdamped Euler–Maruyama sampling. The PRL argues that overdamped
   dynamics is optimal when the free energy is parabolic, citing its ref.
   [9]. The Python driver defaults to Langevin (`OverDamped` 0), the C++
   driver defaulted to overdamped. Euler–Maruyama carries the $O(\Delta t)$
   bias of (5.1). Projected BAOAB has no bias in the harmonic part
   (Section 5.2).
6. Knots at the chord distance from image 0, as in the Python driver,
   which marks this "TODO: symmetric??". The C++ driver symmetrized
   distances from both ends instead. The chord distance is not monotone on
   a path that curves back, and the README warns that the unsmoothed spline
   oscillates between similar images. Use arc length along the spline, or
   the image index with the spline's own derivative.
7. The Python error pipeline (Section 11.4).

### 11.4 Code-level issues to avoid in a port

**`fix pafi`.** Per atom, it applies
$f \leftarrow f - (f\cdot n)n - F_\mathrm{com}/N$, and the same to the
velocity and the noise. This is the orthogonal projector onto
$\{n, \text{centre of mass}\}^\perp$ only when $n$ has zero centre-of-mass
component, which the documentation requires but the fix does not check.
Several other problems follow:

- The fix never normalizes $n$, so the operation is a projection only for
  a unit tangent. The Python driver supplies one.
- All projections are Euclidean. With unequal masses, the overdamped drift
  is $PM^{-1}P$ and the noise covariance is proportional to
  $PM^{-1/2}PM^{-1/2}P$, which do not satisfy detailed balance with
  $e^{-\beta V}$ on the plane. Every published PAFI application is a
  single-species metal. In mass-weighted coordinates (Section 4.1) the
  issue disappears.
- The noise is uniform with matched variance (`uniform() - 0.5`). That is
  adequate for Langevin dynamics in the weak sense, but the harmonic
  exactness of BAOAB needs Gaussian noise.
- The fix documents four output elements but has five. The fifth reads
  `f[i][3..5]`, past the end of the force row, since LAMMPS commit
  6ec3aac3f0. PAFI reads only four elements, so its results are
  unaffected.

**PAFI Python driver.**

- `FreeEnergyGradientVariance` squares an already averaged square.
- The per-worker time-series variance is added to the squared standard
  error.
- Knot errors and the "95% confidence interval" bounds are integrated
  variances without a square root, applied as $\pm1$ of that value.
- The variance is integrated as $\sqrt{\int\sigma^2\,dr}$, not as
  $\sum_i w_i^2\sigma_i^2$ (Section 7.1).
- No time step is set. The default script uses LAMMPS metal units, whose
  default step is 0.001 ps, and `Friction` 0.05 is passed as the damping
  time.
- The resampling threshold `ReSampleThresh` is computed but unused.

**Removed C++ driver.** It passed the `OverDamped` flag as `com` to the
fix.

**README.** Its DOIs for Sato et al. and for Namakian et al. (Cu) do not
resolve. The correct ones are in the bibliography.

### 11.5 Anharmonic results computed with PAFI and related methods

| System and potential | Result, as published | Direction of the HTST error |
|---|---|---|
| Fe, C15 cluster to ⟨110⟩ dumbbells, EAM [swinburneUnsupervisedCalculation2018] | Harmonic free energy "only valid up to around 200 K". Harmonic theory gives C15 relative stability above 450 K, "the full anharmonic calculation completely reverses this picture" | qualitative reversal |
| W ½⟨111⟩ screw double kink, EAM, about 120 000 atoms [swinburneUnsupervisedCalculation2018] | "the harmonic approximation breaks down by 200 K, with the sign of the entropy changing". The velocity is "over ×10³ greater at 800 K" | HTST too slow |
| Mg {10-12} twin nucleation, MEAM, 320 000 atoms [satoAnharmonicEffect2021] | HTST "underestimates the nucleation time by several orders of magnitude" at 0.2 GPa, and predicts a vanishing barrier above about 250 K, where MD nucleation still takes nanoseconds | HTST too fast |
| Cu {111}⟨110⟩ generalized stacking fault, Mishin EAM and ACE [namakianTemperatureDependent2023] | Intrinsic stacking-fault free energy drops 29.1 mJ/m² from 0 to 1294 K (EAM) and 12.93 mJ/m² from 0 to 1244 K (ACE). Anharmonicity grows beyond 200 K with EAM and is "insignificant" with ACE | profile only, no rate |
| Fe ½⟨111⟩ screw kink pair, ML potential, 96 000 atoms [alleraActivationEntropy2025] | Harmonic regime up to $T_0 \simeq 100$ K. Variational HTST agrees with PAFI "over the entire temperature range". MD velocities match with a fitted $\nu$ | HTST adequate with the variational correction |
| Fe kink pair, EAM [alleraActivationEntropy2025] | "a drastic departure from the harmonic prediction … above temperatures as low as 20 K" | potential-dependent |
| W vacancy formation and migration, moment tensor potential, TSTI rather than PAFI [zhangAbInitio2025] | Explicit anharmonicity lowers $G_\mathrm{mig}$ by at most 10% and $G_\mathrm{form}$ by at most 20%, up to 3400 K. MD diffusivities (1024 atoms, 50 NVT runs of 20 ns at 3000–3400 K) "agree very well not only in the absolute values but also in the temperature dependence" | HTST too slow at high $T$ |

What this means for the Cu vacancy problem:

- The sign and size of the anharmonic correction depend on the system and
  on the potential. For the same Cu stacking fault, the Mishin EAM and an
  ACE potential disagree on whether anharmonicity matters. A Cu vacancy
  result with the Mishin EAM is a statement about that potential.
- The only point-defect migration case with an MD check (W vacancy, by
  TSTI) shows a modest reduction of the migration free energy, and MD
  agrees. A localized hop at 700–900 K, below the melting point, should
  show a correction of that order.
- PAFI's cost is "around $10^5 - 10^6$ force calls" per profile
  [swinburneUncertaintyAnharmonicity2021]. Allera et al. report
  $5\times10^4$ to $2.5\times10^5$ CPU hours per condition for their
  96 000-atom cells. The optimization stage adds a few times the cost of
  one plane per iteration (Section 15 budgets it). For a vacancy cell of a
  few hundred atoms with an EAM potential, the whole calculation is cheap.

## 12. What to take from optimal transport

Two different things called optimal transport bear on OH-TST:

- **Optimal transport between atomic configurations.** This is the
  permutation-invariant distance of Goswami and Jónsson's OT-GP saddle
  search [goswamiAdaptivePruning2026].
- **Optimal transport between probability measures.** This is the
  Wasserstein-2 geometry in which overdamped Langevin dynamics is the
  gradient flow of the relative entropy [jordanVariationalFormulation1998],
  together with transport maps between distributions.

Each idea is assessed against the five places it could act: the sampler,
the gradient estimator, the optimizer, the reversible-work integral and the
metric.

| Idea | Acts on | Verdict | Section |
|---|---|---|---|
| Jarzynski's derivative identity along a family of maps | gradient estimator | already used, as the Stein control variates | 6.2 |
| Log-Sobolev contraction of the Langevin flow | sampler: thermalization and starting distribution | use | 12.1 |
| Covariance preconditioning, as in affine-invariant samplers | sampler | use, in the deterministic form of a constant mass tensor | 12.1 |
| Kernel Stein discrepancy | sampler diagnostics | optional | 12.1 |
| Gaussian (Bures–Wasserstein) maps between planes | reversible work: FEP and BAR | use | 12.2 |
| Learned flow maps | reversible work | not in the first version | 12.2 |
| Wasserstein metric on the planes | optimizer metric | interpretation only | 12.3 |
| Rigid map as the retraction acting on samples | optimizer and sampler warm starts | use | 12.3 |
| Intensive Earth mover's distance between configurations | quench and shooting end-state assignment | use | 12.4 |
| SVGD and other interacting-particle transport | sampler | do not use | 12.5 |
| Transport between basin densities | choice of the normal | do not use | 12.5 |
| Schrödinger bridges and stochastic control of paths | $\kappa$, path sampling | out of scope | 12.5 |
| Minimal-dissipation protocols | nonequilibrium $\Delta A$ | does not apply to TI and BAR | 12.5 |

The force and torque are the rigid-map case of Jarzynski's identity for the
derivative of a free energy along a family of maps
[jarzynskiTargetedFree2002, Eq. 18]. Take maps between planes whose
generator $u$ has tangential part $u_\parallel = Pu$. Changing variables in
$Z_H$ gives

$$\frac{dA}{d\lambda} = \big\langle u\cdot\nabla V -
kT\,\operatorname{div}_\Sigma u_\parallel\big\rangle_\Sigma.$$

Rigid motions have $u_\parallel = 0$, which gives (3.7)–(3.9). Adding any
tangential field to $u$ leaves the mean unchanged. That is how the Stein
control variates of Section 6.2 arise [oatesControlFunctionals2017]. The
variance reduction thus comes from a transport identity, as targeted FEP
does, but the field is chosen to minimize variance, not a transport cost.
Optimal transport proper contributes nothing to the gradient estimator.

### 12.1 Sampler

**Contraction and bias.** Section 5.3 gives the contraction of the relative
entropy of overdamped plane dynamics at rate $2\lambda_\mathrm{min}$. By
Otto and Villani, the log-Sobolev inequality also gives a transport-cost
inequality [ottoGeneralizationInequality2000]. With the same constant,
$W_2(\rho,\pi_\Sigma)^2 \le (2kT/\lambda_\mathrm{min})\,
\mathrm{KL}(\rho|\pi_\Sigma)$. For any observable $f$ with Lipschitz
constant $L_f$,

$$|\mathbb{E}_\rho f - \mathbb{E}_{\pi_\Sigma} f|
\le L_f \sqrt{2kT\,\mathrm{KL}(\rho|\pi_\Sigma)/\lambda_\mathrm{min}}.
\tag{12.1}$$

The thermalization stage can stop when this bound on the bias of the force
average falls below its target standard error. The rate does not need
convexity. Vempala and Wibisono prove convergence in relative entropy of
the unadjusted Langevin algorithm from a log-Sobolev inequality and a
bounded Hessian alone [vempalaRapidConvergence2019]. In a plane that cuts
two channels, the log-Sobolev constant is exponentially small in the
in-plane barrier [lelievreTwoMathematical2012]. This is the quantitative
form of the channel problem of Section 5.4.

The constant cannot be computed for a real potential, but it can be bounded
from above. A log-Sobolev inequality implies a Poincaré inequality with the
same constant. The slowest observed decay rate of the force and torque
autocorrelations, which Section 5.3 measures anyway, therefore bounds
$\lambda$ from above.

Lelièvre, Rousset and Stoltz bound the error of a conditional mean force by
$M^2/\rho$ [lelievreLongtimeConvergence2008]. Here $\rho$ is the
log-Sobolev constant of the conditional measures. For planes, $M$ bounds
the normal–tangential coupling $P\nabla^2V n$, which vanishes at the
harmonic optimum (Section 3.6). The optimal plane is thus also the one
whose force averages are least sensitive to incomplete sampling.

The torque is a force–position covariance. It is more sensitive to
step-size bias than the mean force. For the unadjusted Langevin algorithm
in a harmonic plane, the stationary covariance is wrong at first order in
the step while the mean is exact [wibisonoSamplingOptimization2018,
Example 2]. That is one more reason for projected BAOAB, which is exact in
the harmonic part (Section 5.2).

**Starting distribution.** A replica started on the pivot has infinite
relative entropy at time zero. After a short transient, its relative
entropy grows with the number of in-plane modes. Through the bound above,
this costs about $\ln D$ extra relaxation times of the slowest in-plane
mode. Start instead from one of two distributions:

- a draw from the in-plane harmonic Gaussian
  $N(c, kT H_\parallel^{-1})$, with $H_\parallel$ from the saddle Hessian
  or the Lanczos modes of Section 4.2;
- the previous plane's samples, carried over by the rigid map (7.1) and
  the affine map of Section 12.2.

Either way the initial relative entropy is the anharmonic mismatch alone.

**Preconditioning.** Interacting-particle samplers derived from the
Wasserstein gradient structure precondition drift and noise by the ensemble
covariance. Examples are the ensemble Kalman sampler
[garbunoInigoInteractingLangevin2019] and affine-invariant interacting
Langevin dynamics [garbunoInigoAffineInvariant2019]. With this
preconditioning, the contraction rate does not depend on the conditioning
of the target.

In a plane with a known harmonic model, mass tensor dynamics gives the same
effect without interacting particles [bennettMassTensor1975]:

- Use a constant symmetric positive definite in-plane mass matrix $G$.
  Configurational averages do not depend on $G$, so $A$ and its gradients
  are unchanged.
- The trajectories of Section 8 must use the physical masses.
- With $G = H_\parallel$, every harmonic mode has unit frequency. BAOAB
  then samples the harmonic part exactly for any step below 2 in those
  units, and every mode relaxes at the same rate.
- A dense $G^{-1/2}$ costs one matrix-vector product per step. A low-rank
  $G$ that stiffens only the softest $K$ in-plane modes, from the same
  Lanczos run, costs $O(KD)$ per step.

In a crystal supercell the soft in-plane modes are long-wavelength phonons.
Their frequencies fall as the inverse of the cell length, so the gain grows
with the cell size.

**Diagnostics.** The kernel Stein discrepancy measures how far a sample is
from a density known up to normalization
[liuKernelizedStein2016; chwialkowskiKernelTest2016]. With a suitable
kernel it detects non-convergence [gorhamMeasuringSample2017]. On a plane
it needs only the projected score $-\beta P\nabla V$, which the sampler
computes anyway. It can complement the escape test of Section 5.4 as a
check on step-size bias and thermalization. It costs $O(n^2)$ in the
number of retained samples, so apply it to a thinned subsample.

### 12.2 Reversible work: Gaussian transport maps between planes

Section 7.1 maps plane $a$ onto plane $b$ by the rigid map $T_{ab}$ of
(7.1). Write $y = Q_a^\mathsf{T}(q - c_a)$ for in-plane coordinates on
$a$, with $Q_a$ an orthonormal basis of the plane and
$Q_b = R_{ab}Q_a$. Let $(m_a, \Sigma_a)$ and $(m_b, \Sigma_b)$ be the
in-plane means and covariances of the two restricted distributions. The
optimal transport map between the Gaussians with these moments is affine,
$y \mapsto m_b + L(y - m_a)$, with

$$L = \Sigma_a^{-1/2}\big(\Sigma_a^{1/2}\Sigma_b\Sigma_a^{1/2}\big)^{1/2}
\Sigma_a^{-1/2}, \tag{12.2}$$

the symmetric positive definite matrix with $L\Sigma_a L = \Sigma_b$. It is
the only positive definite solution of that equation
[mccannConvexityPrinciple1997; olkinDistanceTwo1982;
takatsuWassersteinGeometry2008; bhatiaBuresWasserstein2019]. Let
$T(q) = c_b + Q_b[m_b + L(Q_a^\mathsf{T}(q - c_a) - m_a)]$. Targeted FEP
[jarzynskiTargetedFree2002] then gives

$$\frac{Z_H(\Sigma_b)}{Z_H(\Sigma_a)}
= |\det L|\,\big\langle e^{-\beta[V(Tq) - V(q)]}\big\rangle_{\Sigma_a}
\tag{12.3}$$

exactly. BAR uses the same work values with the inverse map on the
samples from $b$.

If $V$ is quadratic on both planes, with $\Sigma = kT K^{-1}$, then
$L\Sigma_a L = \Sigma_b$ implies $L^\mathsf{T}K_bL = K_a$. The work
$V(Tq) - V(q)$ is then constant, and (12.3) has zero variance. Only
anharmonicity and errors in the moments reduce the overlap. Among all
linear maps with this property, (12.2) is the one closest to the identity
in mean squared displacement, so it does not rotate the in-plane
coordinates. A Cholesky-based map does rotate them, and its result
depends on the atom ordering.

**Check on the LEPS model.** The table gives the effective sample fraction
$(\mathbb{E}_a w)^2/\mathbb{E}_a w^2$ of the importance weights for the rigid
and affine maps. The model is that of Section 10.3 at $kT = 0.1E_A$. Values
are computed by quadrature on the lines, so they carry no sampling error.
Plane $a$ is the line through the saddle normal to the unstable mode. The
affine map uses the exact moments of both lines.

| Plane $b$ | Deep, rigid | Deep, affine | Shallow, rigid | Shallow, affine |
|---|---|---|---|---|
| rotated 5° about the saddle | 0.9934 | 1.0000 | 0.9937 | 1.0000 |
| rotated 10° | 0.7829 | 1.0000 | 0.7754 | 1.0000 |
| rotated 20° | 0.0000 | 0.5047 | 0.0000 | 0.0086 |
| guideline normal, 44.4° | 0.0582 | 0.1616 | 0.0041 | 0.0000 |
| translated by 0.1 along $n$ | 0.9997 | 1.0000 | 0.9999 | 1.0000 |

Harmonic moments from the saddle Hessian give 0.9999 to 1.0000 up to 10°.
They are undefined beyond the angle at which the rotated line has negative
curvature at the saddle.

The losses follow from the harmonic model. Rotating the normal from the
unstable mode, eigenvalue $\lambda_1 < 0$, by $\theta$ toward an in-plane
eigenvector with eigenvalue $\lambda_u$ multiplies that mode's variance by
$r = \lambda_u/(\lambda_u\cos^2\theta - |\lambda_1|\sin^2\theta)$. The rigid
map's effective fraction for that mode is

$$\sqrt{r(2-r)}, \tag{12.4}$$

and it is zero once $r \ge 2$, that is, once
$\tan^2\theta \ge \lambda_u/(\lambda_u + 2|\lambda_1|)$. For LEPS,
$\lambda_u/|\lambda_1| = 0.0832$, so the threshold is 11.3°. (12.4) gives
0.76 at 10°, against 0.78 by quadrature. The affine map keeps a positive
overlap up to the angle at which the plane loses in-plane binding,
$\tan^2\theta = \lambda_u/|\lambda_1|$, which is 16.1° for LEPS.

$R_{ab}$ differs from the identity only on $\operatorname{span}\{n_a,
n_b\}$. After the rigid map, the in-plane Hessians of two planes through
one pivot therefore differ by a matrix of rank at most two; a numerical
check with $D = 20$ gives rank two. Rotation steps therefore behave as in
two dimensions whatever $D$ is.

In close-packed metals, most in-plane eigenvalues exceed $|\lambda_1|$
(Section 4.2). There $r - 1 \approx (1 + |\lambda_1|/\lambda_u)\theta^2$
is small, and rigid maps suffice for moderate rotations. The affine map
pays off for soft in-plane modes and between planes along a path, where
many local modes change. For large rotations neither map has useful
overlap, and the affine map can be the worse of the two, as in the shallow
44.4° case. Steps must stay small, and the BAR overlap remains the
acceptance test of the trust region (Section 4.4).

**Fitting and cost.** The map needs the two in-plane covariances. Use
either the harmonic ones from the saddle Hessian, or moments estimated from
samples that are not used in the FEP or BAR averages. (12.3) is exact for
any map fixed independently of the evaluation samples. A map fitted to the
evaluation samples biases the estimate, as Rizzi et al. show for learned
maps: the estimate is below the true value on the training data and above
it, on average, on fresh data [rizziTargetedFree2021, Eqs. 12–13]. Shrink
the sample covariance, or use identity outside a leading subspace, when
the number of effective samples is not much larger than $D$. The matrix
square roots cost $O(D^3)$ once per pair of planes.

**Learned maps.** (12.2) is the zero-training member of a family of
transport maps for targeted FEP. Normalizing flows learn more general maps
[wirnsbergerTargetedFree2020; noeBoltzmannGenerators2019]. Rizzi et al.
learn the map from a cheaper reference potential to an expensive target
[rizziTargetedFree2021]. For deca-alanine, learned maps were accurate
between restraint centres 1 Å and sometimes 2 Å apart, and failed for more
distant states [willowLearnedMappings2023]. Flows need training data and
their own error control. They are worth considering only if the affine map
leaves poor overlap between planes that cannot be placed closer together.

### 12.3 Optimizer: metric, retraction and transport

**Wasserstein metric.** A rigid motion of the plane with translation
$\delta$ and rotation $u$ displaces each point of the plane along $n$ by
$w = \delta - u\cdot(q - c)$. Any velocity field that carries the restricted
distribution to that of the moved plane must have this normal component.
$g = \mathbb{E}_\Sigma[w^2] = \delta^2 + u^\mathsf{T}\Sigma_\parallel u$,
the metric of Section 4.2, is therefore the normal part of the
Wasserstein-2 metric of the family of plane distributions, and a lower
bound on it. The in-plane reshaping of the density adds to it. The plane
family is a statistical manifold with a closed-form transport metric of
the kind studied as the Wasserstein natural gradient
[liNaturalGradient2018; chenOptimalTransport2020]. In the harmonic limit,
the natural gradient turns the plane toward $-P_nHn$, the dimer rotation
force (Section 4.2). Its conditioning is worse than that of the
unweighted metric in stiff solids, so it is used for interpretation and
units. The optimizer uses the harmonic Newton metric. Computing the full
transport metric would need an elliptic solve per parameter direction
[chenOptimalTransport2020]. The positivity and smoothness that this
construction assumes fail for distributions supported on moving planes.
The Fisher metric does not exist here at all: plane distributions with
different normals are mutually singular.

**Retraction and transport.** The manifold $S(U)\times\mathbb R$ has a
closed-form retraction and a vector transport (Section 4.3). In Riemannian
optimization, "vector transport" moves tangent vectors and is unrelated to
optimal transport, except in Wasserstein space itself, where a transport
map is the natural vector transport [parkAccelerationSilver2025]. No work
was found that uses a transport map as the retraction or vector transport
of a sphere or a similar manifold. Transport-based retractions therefore
add nothing to the optimizer.
At the level of samples, the rigid map (7.1) is the retraction pushed
forward to the plane distributions. After each accepted step, carry the
current samples to the new plane with $T_{ab}$ and the affine correction
of Section 12.2, then thermalize from there. By Section 12.1 this
shortens the thermalization after a step.

### 12.4 Optimal transport between configurations: assigning end states

Goswami and Jónsson define a distance between configurations by solving
one linear assignment problem for each atom type
[goswamiAdaptivePruning2026]:

$$\bar d_t = \frac{1}{N_t}\min_{\pi}\sum_{k=1}^{N_t}
\lvert r^{(1)}_{k,t} - r^{(2)}_{\pi(k),t}\rvert, \qquad D = \max_t \bar d_t,
\tag{12.5}$$

with $\pi$ over the permutations of the $N_t$ atoms of type $t$. This is
the Earth mover's (Wasserstein-1) distance between the two sets of atom
positions, made intensive. It is invariant to permutations of identical
atoms and respects element types. It does not remove rigid-body motion.
In OT-GP it selects a diverse training subset by farthest-point sampling
and sets the trust radius of the dimer translation.

OH-TST needs exactly this invariance when it assigns end states:

- Section 5.4 assigns each quenched replica to a state.
- Section 8 assigns the end points of the shooting trajectories.
- In a monatomic crystal with one vacancy, a state is the set of occupied
  sites. A labelled Euclidean distance treats configurations that differ
  only in which atoms occupy which sites as different.

With minimum-image costs, compute $D$ from the quenched configuration to
the reactant and to each product reference; the fcc vacancy has twelve. If
the nearest is below a threshold that is a fraction of the nearest-neighbour
distance, assign it to that state, and otherwise flag the replica.

The assignment can be limited to atoms within a cutoff of the defect, the
rest being matched by identity, so its cost is negligible. The plane
itself cannot be permutation invariant: a hyperplane in labelled Cartesian
coordinates belongs to one labelled channel. For the vacancy this does not
matter, because the twelve channels are related by symmetry (Section 9.2).

For potentials that are expensive to evaluate, the OT-GP dimer is also a
cheaper way to obtain the saddle point and unstable mode that OH-TST
starts from.

### 12.5 What optimal transport does not improve

1. **Choosing the normal from the basin densities.** The max-sliced
   Wasserstein distance picks the direction along which the projected
   reactant and product distributions differ most
   [deshpandeMaxSliced2019]. Subspace-robust distances pick a subspace
   [patySubspaceRobust2019]. For Gaussian basins, the squared distance
   between the projections onto $\omega$ is
   $(\omega\cdot\Delta\mu)^2 + (\sqrt{\omega^\mathsf{T}\Sigma_R\omega} -
   \sqrt{\omega^\mathsf{T}\Sigma_P\omega})^2$. The criterion therefore
   rewards displacement and differences in spread, not separation relative
   to spread.

   For basins of equal covariance the maximizer is the mean difference
   $P - R$, which is the guideline. In the LEPS test of Section 10.3,
   planes normal to the guideline lower the barrier by 31.0% from the deep
   minimum and by 68.2% from the shallow one. The rate is then too large by
   13.1 and about 340 times.

   Semi-discrete transport from the Boltzmann density to two point masses
   at $R$ and $P$ assigns configurations to Laguerre cells. Their common
   boundary is a plane normal to $P - R$, the guideline again. A Brenier map
   between the two basins describes a transport between them, not the
   equilibrium flux through a surface, and has no variational relation to
   $k_\mathrm{TST}$.

   Harmonic linear discriminant analysis uses
   $n_0 \propto (\Sigma_R^{-1} + \Sigma_P^{-1})(\mu_P - \mu_R)$, which does
   normalize by the spread [mendelsCollectiveVariables2018]. It is not a
   transport construction. It is the cheapest reasonable start when no
   saddle is known. OH-TST starts from the saddle's unstable mode, which is
   better whenever a saddle is available.

   The search behind this section found no paper that defines a dividing
   surface or committor from a transport map or plan between reactant and
   product densities. The closest is the sliced committor of Petersen,
   Lichtinger and Covino [petersenCommittorsReaction2026]. It is a
   ridge-function ansatz for trial committors, "in the spirit of sliced
   optimal transport", inside a variational bound on the rate. With one
   direction it reduces to the planar bound (8.2), which needs only the
   plane free energies that OH-TST computes.
2. **Stein variational gradient descent.** SVGD moves interacting
   particles deterministically toward the target
   [liuSteinVariational2016]. Its continuum limit is a gradient flow of the
   relative entropy in a kernel-induced geometry
   [liuSteinVariational2017]. Its particles collapse toward modes, and the
   repulsive force weakens as the dimension grows
   [zhuoMessagePassing2018]. That biases the second moments that the torque
   (3.9) and the control variates need. For smooth kernels the flow does
   not converge exponentially to equilibrium
   [duncanGeometryStein2019, Lemma 39]. Its cost is quadratic in the number
   of particles. Independent Langevin replicas have no such bias and run in
   parallel.
3. **Interacting-particle Langevin samplers.** Their gain on a plane is
   the preconditioning, which a constant mass tensor provides without
   interaction (Section 12.1). Affine-invariant interacting Langevin
   dynamics is ergodic only with more walkers than dimensions plus one.
   With fewer, the ensemble stays in the affine span of its initial
   positions [garbunoInigoAffineInvariant2019]. On a plane with
   $D \approx 10^3$ that means more than $10^3$ force calls per step. For
   the ensemble Kalman sampler, the rigorous derivation of the mean-field
   equations is left open by its authors
   [garbunoInigoInteractingLangevin2019].
4. **Schrödinger bridges and stochastic optimal control of paths.** These
   construct or sample reactive trajectories between end states
   [chenStochasticControl2021; holdijkStochasticOptimal2022;
   duDoobsLagrangian2024]. They could serve $\kappa$ or the committor, but
   not the plane free energy, and they need training. They are out of
   scope.
5. **Minimal-dissipation protocols.** Optimal transport gives the driving
   protocol of least dissipation for nonequilibrium free-energy estimates
   [aurellOptimalProtocols2011; chennakesavaluUnifiedGeometric2023]. It
   would matter if the barrier were computed by fast switching of the
   plane. This design uses equilibrium TI and BAR.
6. **Wasserstein-barycentre paths between end states.** Decherchi and
   Cavalli build alchemical paths from Wasserstein barycentres
   [decherchiOptimalTransport2023]. In their harmonic example this reduces
   to moving a restraint centre, which the path of planes already does.
   They state that the protocol is not immediately amenable to
   high-dimensional spaces.

## 13. Neighbouring codes

Revisions read:

| Code | Revision |
|---|---|
| eOn (TheochemUI) | 579738a9, release heading 3.6.0 |
| PLUMED | c0beebb5, 2.11.0-dev |
| LAMMPS | 582a54bc |
| OpenMM | 3e34aa20, 8.6.0 |
| ASE | c879478d, 3.30.0b1 |
| DL-FIND, as packaged by libdlfind | 4167998d |
| ChemShell | online manuals only, source not accessed |
| Polyrate | 17-C manual |

### 13.1 eOn already has an OH-TST job

The premise that no OH-TST code exists is not quite right. eOn's changelog
lists an "Optimized hyperplanar TST (OH-TST) job" under release 3.0.0,
dated 2026-07-26: `job = oh_tst`, in `client/OHTSTJob.cpp` (761 lines).
It is a port of the 2001 paper:

- a straight reactant-to-product guideline;
- velocity Verlet on the plane with Andersen or GLE thermostats;
- trapezoid reversible work for translation and rotation, with grooming;
- damped Verlet updates of the plane position and normal, with clamps;
- the Eq. (22) crossing estimator for the basin term;
- the Eq. (24) flux mass, as `mu = Σ m n²`.

Four findings, checked against the source:

1. **The flux mass is wrong for mixed masses**, as in the 2001 paper
   (Section 3.8). The correct value is $(n^\mathsf{T}M^{-1}n)^{-1}$ for a
   Cartesian unit normal. For $m = (1, 100)$ and $n = (1,1)/\sqrt2$,
   Eq. (24) gives 50.5 instead of 1.98, and the prefactor is off by a
   factor of 5.05.
2. **All projections are Euclidean**, although a code comment calls the
   scheme RATTLE. The Andersen velocities are drawn from
   $N(0, kTM^{-1})$ and projected with $P = I - nn^\mathsf{T}$. The scheme
   is then velocity Verlet for $\tfrac12 p^\mathsf{T}Wp + V$ on the plane,
   with the constant $W = PM^{-1}P$, and the thermostat matches $W$. The
   configurational plane averages are therefore unbiased up to the time
   step, but the trajectories are not physical, so $\kappa$ cannot be
   taken from them.
3. **The normal is driven by the 2001 normalized torque**,
   $\langle(n\cdot F)(x - \Gamma)/(\alpha|x - \Gamma|^2)\rangle$, while the
   reversible work uses the unnormalized $\langle(n\cdot F)(x -
   \Gamma)\rangle$, which is the gradient (3.9). The zeros of the two
   differ, so the converged plane is in general not a stationary point of
   $A$.
4. **The OH-TST tests check bookkeeping**: the scan grid, the 2 kT
   re-anchoring rule, restart unwrapping, the plane count and half-line
   distances. None compares a barrier or a rate with an analytic value.

eOn's path-integral QTST code is the better template inside eOn. It works
in mass-weighted coordinates, projects momenta with the mass metric, and
normalizes its rate with $\int e^{-\beta F(s)}ds$, the analogue of (7.3).
It also has analytic tests:

- one bead reduces to classical TST;
- a harmonic profile gives harmonic TST;
- $\kappa = 1$ at the top of a one-dimensional barrier;
- a rotated barrier;
- the Craig–Manolopoulos Eckart rates [craigChemicalReaction2005].

Its jackknifed $\kappa$ plateau, its trapezoid error propagation and its
overflow-safe product of Vineyard frequencies (`client/Prefactor.cpp`) are
worth copying. The tests of Sections 10.1–10.3 would catch findings 1 and
3 in the eOn job.

### 13.2 PLUMED

- A plane coordinate $n\cdot(x - \gamma)$ is a one-component `PCAVARS`
  with weights $n$ and no alignment. With optimal alignment the coordinate
  depends on rotation, its level sets are no longer planes, and Section 3
  no longer applies.
- PLUMED has no holonomic constraints. It leaves them to the MD engine.
  `RESTRAINT` and `MOVINGRESTRAINT` give a soft plane, and `DRR` gives
  eABF with the CZAR estimator [lesageSmoothedBiasing2017]. There is no
  plain ABF action.
- `COMMITTOR` stops at axis-aligned box basins and computes no committor
  statistics.
- For users of other MD engines, a stiff `RESTRAINT` on a `PCAVARS` plane
  with umbrella integration or CZAR estimates $\partial A/\partial d$
  without a hard constraint. The torque has no such shortcut.

### 13.3 LAMMPS

- `fix rigid` sums $\tau = \sum_i (x_i - x_\mathrm{cm})\times f_i$ every
  step. Its NVE variants rotate exactly (NO_SQUISH,
  [millerSymplecticQuaternion2002]); the default variants use one
  Richardson extrapolation. The lesson for plane dynamics is to rotate the
  normal exactly, not by adding and renormalizing. This design moves the
  normal by optimization steps, where the retraction only needs to be
  first-order accurate, so the point matters only if plane dynamics is
  ever used.
- `fix shake` and `fix rattle` solve small clusters in closed form, and
  `store yes` keeps the constraint forces. For one linear constraint, the
  RATTLE multiplier $\lambda = -(n^\mathsf{T}M^{-1}F)/(n^\mathsf{T}M^{-1}n)$
  is closed-form. In mass-weighted coordinates it is $-n\cdot\nabla V$
  with a sign, the sample of the translational force (3.8). The sampler
  should return it.
- `fix pafi`: Section 11.4.

### 13.4 OpenMM

- `CustomCVForce` with a `CustomExternalForce` can express
  $n\cdot(x - x_0)$, but absolute coordinates are not safe under periodic
  boundaries. OpenMM has no API for a hard plane constraint, since its
  constraints are distances only.
- `LangevinMiddleIntegrator` uses the LFMiddle scheme
  [zhangUnifiedEfficient2019]. It produces the same configurations as
  BAOAB [leimkuhlerEfficientMolecular2016], with leapfrog velocities, and
  applies constraints after the kick and after the drift. Projected BAOAB
  in Section 5.2 uses the same placement.
- The OpenMM headers disagree on whether `LangevinIntegrator` and
  `LangevinMiddleIntegrator` differ.

### 13.5 ASE

- ASE has no rate or TST code. `HarmonicThermo` and
  `from_transition_state`, which drops exactly one imaginary mode, help
  with harmonic comparisons.
- `FixedMode` is a $3N$-dimensional hyperplane constraint. It projects
  positions and forces in the Euclidean metric, does not enforce
  $n\cdot v = 0$ for unequal masses, and reports `len(atoms)` removed
  degrees of freedom instead of one. Temperatures under `FixedMode`
  therefore count $2N$ degrees of freedom instead of $3N - 1$.
- `Langevin` implements Vanden-Eijnden and Ciccotti's scheme with RATTLE
  [vandenEijndenSecondorderIntegrators2006]. `LangevinBAOAB` warns that it
  is untested with constraints.

### 13.6 ChemShell and DL-FIND

- DL-FIND provides harmonic TST, classical and with quantized vibrations,
  Wigner and symmetric Eckart tunnelling corrections, the crossover
  temperature $T_c = \hbar\omega_b/(2\pi k_B)$, and instanton rates. None
  of it samples a dividing surface.
- Tcl-ChemShell has QM/MM free-energy perturbation along a constrained
  coordinate, with forward and backward runs and convergence tests. The
  Py-ChemShell manual documents no rate features. The ChemShell source was
  not accessed.
- For the classical comparison of Section 9, report $T_c$ as a check that
  quantum effects on the hop are small at 700 K and 900 K.

### 13.7 Polyrate

- Canonical variational TST minimizes $k_\mathrm{GT}(s)$ over generalized
  transition states orthogonal to the minimum energy path in mass-scaled
  coordinates. It locates the maximum of the free energy with parabolic or
  quartic fits. This is the harmonic counterpart of the PAFI profile
  maximum. Fit the maximum; do not take the largest sample.
- RODS optimizes "the orientation of the dividing surface at each point
  along the reaction path in order to maximize the free energy of the
  generalized transition state containing that point"
  [gonzalezLafontVariationalTransition1998; villaVariationalTransition1997].
  That is the orientation half of OH-TST in the harmonic approximation,
  with the pivot on the path. It removes spurious dips in the free-energy
  profile caused by imperfect paths. The default is NORODS, with the
  dividing surface normal to the gradient.
- Anharmonicity enters only mode by mode (Morse, torsions). Polyrate
  samples no plane integral.

### 13.8 What to take

| Code | Take | Avoid |
|---|---|---|
| eOn | PI-QTST statistics and analytic tests; overflow-safe Vineyard product; restart logic | Eq. (24) flux mass; Euclidean "RATTLE"; normalized torque as the driving force; crossing estimator; tests without physics |
| PLUMED | `PCAVARS` plane for soft restraints; CZAR | alignment on a plane coordinate |
| LAMMPS | closed-form constraint multipliers, stored; exact rotation if plane dynamics is used | Euclidean projection followed by mass-scaled friction and noise |
| OpenMM | constraint placement in LFMiddle and BAOAB | absolute-coordinate variables under periodic boundaries |
| ASE | `from_transition_state` for harmonic comparisons | `FixedMode` momenta and degree-of-freedom count |
| DL-FIND | log-space harmonic rates; $T_c$ check | — |
| Polyrate | fitted maximum; RODS as precedent | per-mode anharmonicity as a substitute for sampling |

## 14. Ranked improvements over the 2001 method

Ranked by their effect on the correctness of the rate first, then on cost.

1. **Correct measure and gradients** (Section 3). Use mass-weighted
   coordinates and a unit normal. The plane free energy then has no
   Jacobian, and the flux factor is $\sqrt{kT/2\pi}$. This removes:
   - the effective-mass error of Eq. (24);
   - the sign inconsistency of Eq. (8);
   - the $1/(a|R|^2)$ factor inside the torque average;
   - the dependence of the torque on the guideline intersection as pivot.

   Without this the optimum and the rate are wrong for unequal masses, and
   the iteration does not converge to a stationary point of $A$.
2. **Finite-variance basin normalization** (Section 7.2). Eq. (22) has a
   logarithmically divergent second moment. Parallel planes through the
   reactant (7.3) give a smooth, low-variance estimate and the attempt
   frequency (7.4) as a by-product.
3. **A convergent stochastic optimizer** (Section 4.4). The 2001 plane
   dynamics is heuristic. Its inertia parameter changed the mechanism found
   for the Al adatom (0.154 against 0.00220), and it has no stopping rule.
   This design uses preconditioned Riemannian stochastic gradient ascent
   with averaging, sample-size control, a trust region whose steps are
   checked by BAR, and a noise-floor stop. Convergence results apply.
4. **Path-independent barrier assembly** (Section 7.1). Run PAFI from the
   reactant to the saddle plane, then a short optimization, with BAR on
   exact rigid maps between planes. This replaces integration along the
   optimizer's own trajectory, with first-order transport (Eqs. 11–12) and
   trapezoid integration, and removes the need for grooming.
5. **Control variates** (Section 6.2). Stein control variates built from
   the saddle Hessian cut the variance of the force by factors of 350–700
   and of the torque by 6–180 in the two-dimensional check. Cost scales
   with the variance, so this is the largest single saving.
6. **An unbiased plane sampler** (Section 5). Projected BAOAB has no
   discretization bias in the harmonic part, for any stable step. The 2001
   sampler was velocity Verlet with Andersen collisions, whose
   configurational bias is $O(\Delta t^2)$ even for a harmonic system.
7. **Restricting rotation to a subspace** (Section 4.5). Rotating only in
   the subspace of the moving atoms keeps the variational bound and removes
   the random walk of the normal in thousands of stiff directions. It also
   makes the noise-floor test meaningful.
8. **Channel restriction by quench tests** (Section 5.4). The 2001 paper
   restricted the sampler to one channel with symmetry half-planes and
   velocity reflection at a distance criterion (Eqs. 15–17). Without that
   restriction, the Al adatom plane averaged two channels.    A
   post-replica quench test does not depend on hand-built geometry. With
   the permutation-invariant distance of Section 12.4, it does not depend
   on atom labels either. Multiple maxima are handled by explicit
   candidate comparison (Section 4.4).
9. **Dynamical correction with an error estimate** (Section 8). Bennett–
   Chandler from the optimal plane, with $N_\mathrm{traj}$ set from the
   Vanden-Eijnden–Tal bound and end states assigned by quenching.
10. **A complete error budget** (Section 7.3). Report the standard error of
    $\ln k$ and its independent components. The absolute rates of the 2001
    paper cannot be reproduced from its text (Section 10.3).
11. **Decomposition against harmonic theory** (Section 9.3). Split
    $\ln(k/k_V)$ into thermal expansion, anharmonic planes, orientation and
    recrossing, so a discrepancy with MD can be located.
12. **Transport-based savings** (Sections 12.1–12.3). Start replicas from
    the harmonic Gaussian or from mapped samples. Precondition soft
    in-plane modes with a constant mass tensor. Use the Gaussian transport
    map (12.2) for FEP and BAR between planes. These reduce cost and do
    not change the rate. In the LEPS check, the Gaussian map keeps the full
    effective sample size for 10° rotation steps, where the rigid map
    keeps 78%.

## 15. Implementation plan

The plan reuses `PointSurface`, `pafi.rs` and `bennett.rs` in rgsaddle. It
needs from rgmin the product manifold being derived separately, plus
existing pieces: `rtr`, `RtrRadius`, `control::NoiseFloor`, the `hvp`
module and the noise estimators. Each stage ends with tests that need no
external potential.

1. **Plane geometry.**
   - A `Plane` type: pivot and unit normal in mass-weighted coordinates,
     the projector with the centre-of-mass directions, and the rigid map
     (7.1).
   - The closed-form harmonic model (3.14)–(3.15) from a saddle Hessian.

   Tests: projector identities, rigid-map invariance of the area measure,
   and harmonic curvatures against finite differences.
2. **Plane sampler.**
   - Projected BAOAB in mass-weighted coordinates with Gaussian noise.
   - Independent replicas started from the in-plane harmonic Gaussian, or
     from the previous plane's samples carried over by the plane maps.
     Thermalization is set by the in-plane curvature and the bias bound
     (12.1) (Sections 5.3 and 12.1).
   - Optional constant mass tensor on the softest in-plane modes
     (Section 12.1).
   - The post-replica quench test (Section 5.4), with end states assigned
     by the permutation-invariant distance (12.5).

   Tests: exact second moments in a quadratic potential at several steps,
   and the escape test on a double-well plane.
3. **Estimators.**
   - Force and torque, (3.8)–(3.9), with Stein control variates (6.2)–(6.3).
   - Replica covariance.
   - BAR and MBAR between planes on the rigid map (7.1) and the Gaussian
     map (12.2).
   - The fluctuation Hessian (3.12) as a diagnostic.

   Tests: zero variance of the controlled estimators and of (12.3) in a
   quadratic potential, the variance reductions of Section 6.2 on the
   cubic check potential, and the LEPS overlaps of Section 12.2.
4. **Optimizer.**
   - Averaged preconditioned Riemannian stochastic gradient ascent on
     $S(U)\times\mathbb R$ with the harmonic preconditioner.
   - The norm test for sample size, the trust region checked by BAR, the
     noise-floor stop, the rotation subspace and candidate comparison.

   Tests: Sections 10.1, 10.2 and 10.4.
5. **Rate.**
   - Basin normalization (7.3)–(7.4).
   - Barrier assembly from the PAFI profile plus the optimization leg.
   - $\kappa$ from `bennett_chandler` started on the optimal plane.
   - The error budget (7.6) and the decomposition (9.1).

   Tests: Vineyard's rate recovered in the harmonic limit, Section 10.3,
   and agreement with `pafi.rs` on the profile part.
6. **Examples outside CI.** The Cu vacancy at 700 K and 900 K (Section 9),
   once the 900 K lattice parameter has been measured with the same
   potential. Optionally, the Al adatom (Section 10.5).

**Cost model.** Let $C_p$ be the cost of one converged plane average. The
calculation takes about:

- 10–20 $C_p$ for the PAFI profile;
- 5–9 $C_p$ for the basin;
- a few $C_p$ per optimization iteration, for 10–30 iterations, fewer with
  control variates and a good start;
- $10^2$–$10^3$ short trajectories for $\kappa$.

Section 6.3 sets $C_p$ through the target error.

**Interface sketch.**

```rust
pub struct OhTstConfig {
    pub thermal: f64,
    pub sampler: PlaneSamplerConfig,
    pub optimizer: PlaneOptimizerConfig,
    pub basin: BasinConfig,
    pub transmission: Option<ShootingConfig>,
}

pub struct OhTstResult {
    pub plane: Plane,
    pub barrier: Estimate,
    pub basin_ratio: Estimate,
    pub kappa: Option<Estimate>,
    pub ln_rate: Estimate,
    pub decomposition: RateDecomposition,
}

pub fn optimal_hyperplane<S: PointSurface>(
    surface: &S,
    masses: ArrayView1<f64>,
    reactant: ArrayView1<f64>,
    saddle: ArrayView1<f64>,
    path: Option<&ArcPath>,
    config: &OhTstConfig,
) -> Result<OhTstResult, SaddleError>;
```

`Estimate` holds a value and its standard error. A first implementation
that follows the 2001 paper is being written separately. Sections 3.8 and
14 list what has to change in such a version. The tests of Sections
10.1–10.3 apply to it unchanged.

## 16. Literature survey and access notes

Forward citations of the 2001 paper came from OpenAlex and Semantic Scholar:
53 records after merging, 50 with DOIs. Each citation was confirmed in the
citing work's reference list where Crossref holds one. Theses, the citers of
the 2005 follow-up, and the Jónsson and Goswami publication lists were
searched as well. Related methods were found by topic searches and by
following references.

A citing work is classed as follows:

- it *uses* OH-TST if it applies it;
- it *extends* OH-TST if it modifies or re-derives it;
- it *criticizes* OH-TST if it states a limitation or an error;
- otherwise it *mentions* OH-TST.

### 16.1 Jónsson group and Goswami

Only two published works run OH-TST. The 2001 paper treats an Eckart
barrier and an Al adatom on Al(100)
[johannessonOptimizationHyperplanar2001]. Bligaard and Jónsson treat
two-dimensional models, including a mosaic of four lines
[bligaardOptimizationHyperplanar2005]. Later Jónsson-group papers cite
OH-TST as the full-TST alternative to HTST without running it.

| Work | Content relevant here | Section |
|---|---|---|
| Schenter, Mills and Jónsson [schenterReversibleWork1994] | Reversible-work quantum TST. The plane free energy changes by translation work, from the mean normal force, and by rotation work, from the torque about a pivot | 3.3 |
| Mills, Jónsson and Schenter [millsReversibleWork1995] | The classical form. The optimal plane has zero mean force and zero mean torque | 3.3 |
| Jóhannesson and Jónsson [johannessonOptimizationHyperplanar2001] | The method, the Eckart test and the Al adatom benchmark | 3.8, 10.5, 13.1 |
| Bligaard and Jónsson [bligaardOptimizationHyperplanar2005] | Two orientation maxima of equal height at $kT = 0.2266\,E_a$ in a twisted valley, and a mosaic of planes | 3.7, 10.2 |
| Mökkönen et al. [mokkonenPolymerEscape2014; mokkonenTransitionState2015; mokkonenEfficientDynamical2016] | Polymer escape: a centroid plane; then HTST with anharmonic and Voter–Doll corrections; then a forward-flux correction on parallel planes. All three name orientation optimization as future work | 8 |
| Jónsson [jonssonSimulationSurface2011] | "The normal of the hyperplane after maximizing the free energy gives the reaction coordinate of the transition in the bottleneck region." Full free-energy optimization within adaptive kinetic Monte Carlo "remains a challenge for the future" | 1 |
| Maronsson, Jónsson and Vegge [maronssonMethodFinding2012] | Ridge between saddle points for Al(100) adatom processes. The ratio of ridge to harmonic partition functions is about 0.5 for a four-atom process at 350 K | 10.5 |
| Bessarab, Uzdin and Jónsson [bessarabPotentialEnergy2013] | Harmonic TST for spin systems. "In its most elaborate form, the transition state is determined by maximizing the free energy" | – |
| Smidstrup et al. [smidstrupImprovedInitial2014] | Initial paths, proposed as input for free-energy paths sampled in hyperplanes | 7.1 |
| Koistinen et al., Liashko et al., Ásgeirsson et al., Müller et al., Re Fiorentin et al. [koistinenNudgedElastic2017; liashkoEffectTemperature2017; asgeirssonEfficientEvaluation2018; mullerDuplicationCollapse2018; reFiorentinMethodologicalFrameworks2026] | Cite optimization of the location and orientation of the plane as the full-TST alternative to HTST | – |
| Ásgeirsson, PhD thesis [asgeirssonDevelopmentEvaluation2021] | "At present, finding the optimal transition state dividing surface is a computationally infeasible task for high dimensional chemical systems" | 4.5, 15 |

| Goswami's work | Content relevant here | Section |
|---|---|---|
| Goswami et al. [goswamiEfficientImplementation2025] | Gaussian-process dimer searches in eOn. Its only citation of OH-TST: where HTST fails, "an optimal dividing surface should be determined by maximizing the free energy" | 1 |
| Goswami and Jónsson [goswamiAdaptivePruning2026] | Pruning of Gaussian-process training data by an intensive, permutation-invariant earth mover's distance | 12.4 |
| Goswami, Gunde and Jónsson [goswamiEnhancedClimbing2026] | Climbing-image NEB with Hessian eigenmode alignment. HTST is framed as a plane through the saddle with its normal along the unstable mode | 9.2 |
| Goswami [goswamiTutorialReview2026] | Review of Gaussian-process stationary-point searches, with the HTST rate as its Eq. (1.1) | – |
| Goswami, PhD thesis [goswamiEfficientExploration2025] | Cites OH-TST only in the reprinted 2025 paper | – |
| eOn [chillEONSoftware2014; goswamiTheochemUIEOn2026] | The `oh_tst` job and the path-integral QTST code | 13.1 |

Apart from the 2001 and 2005 papers, no work in either table optimizes a
dividing surface by its free energy. The `oh_tst` job in eOn is the only
implementation of OH-TST found.

### 16.2 Other citing work

The citation records reduce to 49 citing works, after merging duplicates
and removing one false positive. Works with two classes count in both:

- 5 extend OH-TST;
- 4 criticize it;
- 1 uses it (Bligaard and Jónsson, on two-dimensional models);
- 25 mention it in passing;
- for 17, the citing passage could not be read.

One more criticism comes from a thesis outside the citation databases.

- **Vanden-Eijnden and Tal** [vandenEijndenTransitionState2005] (extends,
  criticizes). The 2001 paper "gave incorrect equations for b and n̂", and
  the correct planar conditions are their Eq. (33). They trace the error to
  a "misconception about the free energy". Section 3.8 locates it
  concretely, in the normalized torque and the flux mass.
- **E and Vanden-Eijnden** [eMetastabilityConformation2004] (extends,
  criticizes). They derive different equations for $n$ and $b$, and write
  that "the very concept of a dividing surface for the transition may be
  flawed". This design answers with $\kappa$, the consistency test and the
  planar diffusive bound (8.2), all in Section 8.
- **Kudin and Car** [kudinFreeEnergy2005] (extends; abstract only). The
  free energy along a sequence of hyperplanes splits into a translational
  part, from the constraint force, and a rotational part, from the torque
  about reference points. The rotational part is "sizable". This is (3.8)
  and (3.9), with the pivot as a gauge choice (Sections 3.3 and 3.6).
- **Zinovjev and Tuñón** [zinovjevTransitionState2015] (extends). They vary
  a hyperplane in a space of collective variables to maximize $\kappa$, and
  obtain $\kappa = 0.8$ with 18 coordinates for an enzyme reaction.

  They later call it a "generalized" hyperplanar transition state, to avoid
  confusion with the Cartesian method
  [zinovjevReactionCoordinates2018]. Only the citation context of that
  review was seen.
- **van Erp** [vanErpEfficiencyAnalysis2006] (criticizes). The method is
  said to rely on "harmonic approximations" and on Cartesian linear
  combinations. In this design, harmonic information enters only the
  control variates, the preconditioner and the starting distribution. None
  of them biases the averages.
- **Tao et al.** [taoComputingFree2014] (criticizes). "The hyperplane
  constraint is only valid within the region close to the reference path."
  This is the escape problem of Sections 3.7 and 5.4.
- **Zinovjev** [zinovjevEnzymaticReaction2017] (criticizes; PhD thesis).
  The dividing surface "should be invariant to the rotation and translation
  of the system", which the 2001 definition is not. The method was "never
  applied to any real world problem".

  This design removes the centre-of-mass direction from the normal
  (Section 4.6). Overall rotation is not a symmetry of a periodic cell. A
  free molecule or cluster would need the plane in a body-fixed frame,
  which this design does not cover.

### 16.3 Related methods

| Topic | Sources | Section |
|---|---|---|
| Variational TST | [wignerTransitionState1938; horiutiStatisticalMechanical1938; keckVariationalTheory1967; truhlarVariationalTransition1984; villaVariationalTransition1997; gonzalezLafontVariationalTransition1998] | 1, 3.2, 13.7 |
| Flux minimized over linear combinations of coordinates | [martinGeneralProcedure1982] | – |
| Optimal planes in system–bath models | [pollakTheoryActivated1986; berezhkovskiiActivatedRate1992; messinaQuantumActivated1993; frishmanOptimizedPlanar1994] | 10.4 |
| Planar VTST conditions and the finite-temperature string | [vandenEijndenTransitionState2005; eFiniteTemperature2005; eTransitionPathways2005; vandenEijndenRevisitingFinite2009] | 3.2, 3.3, 8 |
| Reversible work and free-energy gradients | [schenterReversibleWork1994; millsReversibleWork1995; neriaSimulationActivation1996; kudinFreeEnergy2005; okuyamaYoshidaStructureOptimization2000; schenterGeneralizedTransition2003] | 3.3 |
| Blue moon and constrained ensembles | [carterConstrainedReaction1989; sprikFreeEnergy1998; denOtterCalculationFree1998; ciccottiBlueMoon2005] | 5.1 |
| Stochastic approximation, Euclidean and Riemannian | [robbinsStochasticApproximation1951; polyakAccelerationStochastic1992; bonnabelStochasticGradient2013; tripuraneniAveragingStochastic2018; becigneulRiemannianAdaptive2019; durmusRiemannianStochastic2021; blanchetConvergenceRate2019; byrdSampleSize2012; absilTrustRegion2007] | 4.4 |
| Adaptive biasing force | [comerAdaptiveBiasing2015; lelievreLongtimeConvergence2008; lesageSmoothedBiasing2017] | 7.2, 11.1, 13.2 |
| Linear reaction coordinates from path sampling | [bestReactionCoordinates2005; petersObtainingReaction2006; petersExtensionsLikelihood2007; lechnerNonlinearReaction2010; hummerTransitionPaths2004; maAutomaticMethod2005; vanErpNovelPath2003] | 8 |
| Committor and machine-learned dividing surfaces | [khooSolvingHighdimensional2019; liComputingCommittor2019; rotskoffActiveImportance2022; kangComputingCommittor2024; jungMachineguidedPath2023; chenDiscoveringReaction2023; pozunOptimizingTransition2012; schraftNeuralNetwork2018; petersReactionCoordinates2016; petersenCommittorsReaction2026] | 8, 12.5 |
| Hyperplane free-energy paths in solids | [swinburneUnsupervisedCalculation2018; swinburneUncertaintyAnharmonicity2021; alleraActivationEntropy2025; satoAnharmonicEffect2021; namakianTemperatureDependent2023; zhangAbInitio2025] | 11 |
| Optimal transport | the sources of Section 12 | 12 |

Martin and Raff minimized the classical flux over dividing surfaces that
are linear combinations of internal coordinates, for three-body exchange
reactions [martinGeneralProcedure1982]. Only the abstract was read.

Path-sampling methods optimize the same variable as OH-TST, a plane normal,
but fit it to committor data rather than to the flux. Section 8 uses this
as a check. Kang, Trizio and Parrinello bias the sampling by
$-\beta^{-1}\ln|\nabla q|^2$ [kangComputingCommittor2024]. For a sharp
sigmoid of $n\cdot q$, this concentrates the samples on the plane, so in
that limit their sampler becomes a plane sampler.

The machine-learning work minimizes the committor (Dirichlet) functional,
whose restriction to planar trial functions gives (8.2). No paper from
2018–2026 was found that trains a neural or linear dividing surface by
directly minimizing the configurational TST flux. The closest prior
methods optimize a plane in collective-variable space for $\kappa$
[zinovjevTransitionState2015], or train a classifier on basin labels
[pozunOptimizingTransition2012]. This reflects the coverage of the search,
not proof that no such work exists.

### 16.4 Sources not read in full

Statements that rest on these sources are attributed to them where they
appear.

- **Not accessed, or metadata only.**
  - Horiuti, and Wigner [horiutiStatisticalMechanical1938;
    wignerTransitionState1938].
  - Keck: section titles only [keckVariationalTheory1967].
  - Foiles [foilesEvaluationHarmonic1994].
  - Assaraf and Caffarel [assarafZeroVariance1999].
  - Feibelman, Voter and Chen, and Chen and Voter
    [feibelmanDiffusionPath1990; voterAccurateInteratomic1986;
    chenReconstruction1991]. These are cited through the 2001 paper.
  - The ChemShell source code.
- **Abstract only.**
  - Chandler [chandlerStatisticalMechanics1978].
  - Carter et al., Sprik and Ciccotti, and den Otter and Briels
    [carterConstrainedReaction1989; sprikFreeEnergy1998;
    denOtterCalculationFree1998]. The constrained-ensemble formulas used
    here come from Ciccotti, Kapral and Vanden-Eijnden
    [ciccottiBlueMoon2005], read as a preprint.
  - Polyak and Juditsky [polyakAccelerationStochastic1992].
  - The planar-surface and free-energy-gradient papers of Section 16.3:
    Martin and Raff; Pollak; Berezhkovskii, Pollak and Zitserman; Messina,
    Schenter and Garrett; Frishman, Berezhkovskii and Pollak;
    Okuyama-Yoshida et al.; Schenter, Garrett and Truhlar.
  - Kudin and Car; Hummer; Ma and Dinner; Schraft et al.
  - DeLorenzi, Flynn and Jacucci; Paci and Ciccotti; Sivak et al.
- **Citation context only.** Zinovjev and Tuñón's review
  [zinovjevReactionCoordinates2018].
- **Text extractions with unreliable symbols.** Peters and Trout; Peters,
  Beckham and Trout; Pozun et al.; Chen, Roux and Chipot; Neria, Fischer
  and Karplus. Their equations are not quoted.
- **Partial or secondary.**
  - Makarov and Metiu [makarovReactionRate1997], quoted from Bligaard and
    Jónsson.
  - Zinovjev and Tuñón 2015, read as reprinted in Zinovjev's thesis.
  - Lelièvre's two-scale criterion, used through his review
    [lelievreTwoMathematical2012].
  - Swinburne 2021: Section VI only
    [swinburneUncertaintyAnharmonicity2021].
  - Namakian et al. 2023: a repository text capture without page numbers.
  - Sørensen and Voter's Fig. 4(b), read from the figure without
    digitizing.
- **Read as arXiv versions, journal versions not checked.** Takatsu;
  Duncan, Nüsken and Szpruch; both Garbuno-Inigo papers; Vempala and
  Wibisono; Wibisono; Liu 2017; Holdijk et al.; Deshpande et al. Equation
  and lemma numbers follow the arXiv versions.
- **Cited for standard results that were derived again here; full text not
  read for this document.**
  - Vineyard [vineyardFrequencyFactors1957]; Anderson
    [andersonStatisticalTheories1973]; Bennett
    [bennettMolecularDynamics1977]; Kramers [kramersBrownianMotion1940].
  - Villà and Truhlar, and González-Lafont et al.
    [villaVariationalTransition1997;
    gonzalezLafontVariationalTransition1998].
  - Integrators and constrained dynamics: [bennettMassTensor1975;
    ryckaertNumericalIntegration1977; andersenRattleVelocity1983;
    millerSymplecticQuaternion2002; vandenEijndenSecondorderIntegrators2006;
    leimkuhlerRationalConstruction2012; leimkuhlerEfficientMolecular2016;
    zhangUnifiedEfficient2019].
  - Estimators and diagnostics: [bennettEfficientEstimation1976;
    shirtsStatisticallyOptimal2008; flyvbjergErrorEstimates1989;
    robertsOptimalScaling1998; miraZeroVariance2013;
    liuKernelizedStein2016; chwialkowskiKernelTest2016;
    gorhamMeasuringSample2017].
  - Functional inequalities: [bakryDiffusionsHypercontractives1985;
    ottoGeneralizationInequality2000].
  - Optimization: [byrdSampleSize2012; absilTrustRegion2007;
    blanchetConvergenceRate2019].
  - Transport and flows: [bhatiaBuresWasserstein2019;
    patySubspaceRobust2019; noeBoltzmannGenerators2019].
  - Test reference: [craigChemicalReaction2005].

## 17. Bibliography

This list is generated from `docs/design/oh-tst-lit.bib` and sorted by
key. Entries with more than three authors give the first author only.

- `absilTrustRegion2007`: P.-A. Absil, C.G. Baker and K.A. Gallivan,
  "Trust-Region Methods on Riemannian Manifolds", *Foundations of
  Computational Mathematics* 7(3), 303–330 (2007).
  [doi:10.1007/s10208-005-0179-9](https://doi.org/10.1007/s10208-005-0179-9).
- `alleraActivationEntropy2025`: Arnaud Allera et al., "Activation entropy
  of dislocation glide in body-centered cubic metals from atomistic
  simulations", *Nature Communications* 16(1), 8367 (2025).
  [doi:10.1038/s41467-025-62390-w](https://doi.org/10.1038/s41467-025-62390-w).
- `andersenRattleVelocity1983`: Hans C Andersen, "Rattle: A “velocity”
  version of the shake algorithm for molecular dynamics calculations",
  *Journal of Computational Physics* 52(1), 24–34 (1983).
  [doi:10.1016/0021-9991(83)90014-1](https://doi.org/10.1016/0021-9991%2883%2990014-1).
- `andersonStatisticalTheories1973`: James B. Anderson, "Statistical
  theories of chemical reactions. Distributions in the transition region",
  *The Journal of Chemical Physics* 58(10), 4684–4692 (1973).
  [doi:10.1063/1.1679032](https://doi.org/10.1063/1.1679032).
- `asgeirssonDevelopmentEvaluation2021`: Vilhjálmur Ásgeirsson, "Development
  and evaluation of computational methods for studies of chemical
  reactions", PhD thesis, University of Iceland (2021). No DOI or arXiv
  identifier found; ISBN 978-9935-9564-9-1.
- `asgeirssonEfficientEvaluation2018`: Vilhjálmur Ásgeirsson, Andri
  Arnaldsson and Hannes Jónsson, "Efficient evaluation of atom tunneling
  combined with electronic structure calculations", *The Journal of Chemical
  Physics* 148(10), 102334 (2018).
  [doi:10.1063/1.5007180](https://doi.org/10.1063/1.5007180).
- `assarafZeroVariance1999`: Roland Assaraf and Michel Caffarel,
  "Zero-Variance Principle for Monte Carlo Algorithms", *Physical Review
  Letters* 83(23), 4682–4685 (1999).
  [doi:10.1103/PhysRevLett.83.4682](https://doi.org/10.1103/PhysRevLett.83.4682).
- `aurellOptimalProtocols2011`: Erik Aurell, Carlos Mejía-Monasterio and
  Paolo Muratore-Ginanneschi, "Optimal Protocols and Optimal Transport in
  Stochastic Thermodynamics", *Physical Review Letters* 106(25), 250601
  (2011).
  [doi:10.1103/PhysRevLett.106.250601](https://doi.org/10.1103/PhysRevLett.106.250601).
- `bakryDiffusionsHypercontractives1985`: D. Bakry and M. Émery, "Diffusions
  hypercontractives", in *Séminaire de Probabilités XIX 1983/84*, Lecture
  Notes in Mathematics 1123, 177–206 (1985).
  [doi:10.1007/BFb0075847](https://doi.org/10.1007/BFb0075847).
- `becigneulRiemannianAdaptive2019`: Gary Bécigneul and Octavian-Eugen
  Ganea, "Riemannian Adaptive Optimization Methods", in *International
  Conference on Learning Representations* (2019).
  [arXiv:1810.00760](https://arxiv.org/abs/1810.00760).
- `bennettEfficientEstimation1976`: Charles H Bennett, "Efficient estimation
  of free energy differences from Monte Carlo data", *Journal of
  Computational Physics* 22(2), 245–268 (1976).
  [doi:10.1016/0021-9991(76)90078-4](https://doi.org/10.1016/0021-9991%2876%2990078-4).
- `bennettMassTensor1975`: Charles H. Bennett, "Mass tensor molecular
  dynamics", *Journal of Computational Physics* 19(3), 267–279 (1975).
  [doi:10.1016/0021-9991(75)90077-7](https://doi.org/10.1016/0021-9991%2875%2990077-7).
- `bennettMolecularDynamics1977`: Charles H. Bennett, "Molecular Dynamics
  and Transition State Theory: The Simulation of Infrequent Events", in
  *Algorithms for Chemical Computations*, ACS Symposium Series 46, 63–97
  (1977).
  [doi:10.1021/bk-1977-0046.ch004](https://doi.org/10.1021/bk-1977-0046.ch004).
- `berezhkovskiiActivatedRate1992`: Alexander M. Berezhkovskii, Eli Pollak
  and Vladimir Yu. Zitserman, "Activated rate processes: Generalization of
  the Kramers–Grote–Hynes and Langer theories", *The Journal of Chemical
  Physics* 97(4), 2422–2437 (1992).
  [doi:10.1063/1.463081](https://doi.org/10.1063/1.463081).
- `bessarabPotentialEnergy2013`: Pavel F. Bessarab, Valery M. Uzdin and
  Hannes Jónsson, "Potential Energy Surfaces and Rates of Spin Transitions",
  *Zeitschrift für Physikalische Chemie*, 130708000310008 (2013).
  [doi:10.1524/zpch.2013.0403](https://doi.org/10.1524/zpch.2013.0403).
- `bestReactionCoordinates2005`: Robert B. Best and Gerhard Hummer,
  "Reaction coordinates and rates from transition paths", *Proceedings of
  the National Academy of Sciences* 102(19), 6732–6737 (2005).
  [doi:10.1073/pnas.0408098102](https://doi.org/10.1073/pnas.0408098102).
- `bhatiaBuresWasserstein2019`: Rajendra Bhatia, Tanvi Jain and Yongdo Lim,
  "On the Bures–Wasserstein distance between positive definite matrices",
  *Expositiones Mathematicae* 37(2), 165–191 (2019).
  [doi:10.1016/j.exmath.2018.01.002](https://doi.org/10.1016/j.exmath.2018.01.002).
- `blanchetConvergenceRate2019`: Jose Blanchet et al., "Convergence Rate
  Analysis of a Stochastic Trust-Region Method via Supermartingales",
  *INFORMS Journal on Optimization* 1(2), 92–119 (2019).
  [doi:10.1287/ijoo.2019.0016](https://doi.org/10.1287/ijoo.2019.0016).
- `bligaardOptimizationHyperplanar2005`: Thomas Bligaard and Hannes Jónsson,
  "Optimization of hyperplanar transition states: Application to 2D test
  problems", *Computer Physics Communications* 169(1-3), 284–288 (2005).
  [doi:10.1016/j.cpc.2005.03.064](https://doi.org/10.1016/j.cpc.2005.03.064).
- `boisvertSurfaceDiffusion1998`: Ghyslain Boisvert, Normand Mousseau and
  Laurent J. Lewis, "Surface diffusion coefficients by thermodynamic
  integration: Cu on Cu(100)", *Physical Review B* 58(19), 12667–12670
  (1998).
  [doi:10.1103/PhysRevB.58.12667](https://doi.org/10.1103/PhysRevB.58.12667).
- `bonnabelStochasticGradient2013`: Silvere Bonnabel, "Stochastic Gradient
  Descent on Riemannian Manifolds", *IEEE Transactions on Automatic Control*
  58(9), 2217–2229 (2013).
  [doi:10.1109/TAC.2013.2254619](https://doi.org/10.1109/TAC.2013.2254619).
- `byrdSampleSize2012`: Richard H. Byrd et al., "Sample size selection in
  optimization methods for machine learning", *Mathematical Programming*
  134(1), 127–155 (2012).
  [doi:10.1007/s10107-012-0572-5](https://doi.org/10.1007/s10107-012-0572-5).
- `carterConstrainedReaction1989`: E.A. Carter et al., "Constrained reaction
  coordinate dynamics for the simulation of rare events", *Chemical Physics
  Letters* 156(5), 472–477 (1989).
  [doi:10.1016/S0009-2614(89)87314-2](https://doi.org/10.1016/S0009-2614%2889%2987314-2).
- `chandlerStatisticalMechanics1978`: David Chandler, "Statistical mechanics
  of isomerization dynamics in liquids and the transition state
  approximation", *The Journal of Chemical Physics* 68(6), 2959–2970 (1978).
  [doi:10.1063/1.436049](https://doi.org/10.1063/1.436049).
- `chenDiscoveringReaction2023`: Haochuan Chen, Benoît Roux and Christophe
  Chipot, "Discovering Reaction Pathways, Slow Variables, and Committor
  Probabilities with Machine Learning", *Journal of Chemical Theory and
  Computation* 19(14), 4414–4426 (2023).
  [doi:10.1021/acs.jctc.3c00028](https://doi.org/10.1021/acs.jctc.3c00028).
- `chennakesavaluUnifiedGeometric2023`: Shriram Chennakesavalu and Grant M.
  Rotskoff, "Unified, Geometric Framework for Nonequilibrium Protocol
  Optimization", *Physical Review Letters* 130(10), 107101 (2023).
  [doi:10.1103/PhysRevLett.130.107101](https://doi.org/10.1103/PhysRevLett.130.107101).
- `chenOptimalTransport2020`: Yifan Chen and Wuchen Li, "Optimal transport
  natural gradient for statistical manifolds with continuous sample space",
  *Information Geometry* 3(1), 1–32 (2020).
  [doi:10.1007/s41884-020-00028-0](https://doi.org/10.1007/s41884-020-00028-0).
- `chenReconstruction1991`: S. P. Chen and A. F. Voter, "Reconstruction of
  the (310), (210) and (110) surfaces in fcc metals", *Surface Science*
  244(1-2), L107–L112 (1991).
  [doi:10.1016/0039-6028(91)90162-L](https://doi.org/10.1016/0039-6028%2891%2990162-L).
- `chenStochasticControl2021`: Yongxin Chen, Tryphon T. Georgiou and Michele
  Pavon, "Stochastic Control Liaisons: Richard Sinkhorn Meets Gaspard Monge
  on a Schrödinger Bridge", *SIAM Review* 63(2), 249–313 (2021).
  [doi:10.1137/20M1339982](https://doi.org/10.1137/20M1339982).
- `chillEONSoftware2014`: Samuel T Chill et al., "EON: software for long
  time simulations of atomic scale systems", *Modelling and Simulation in
  Materials Science and Engineering* 22(5), 055002 (2014).
  [doi:10.1088/0965-0393/22/5/055002](https://doi.org/10.1088/0965-0393/22/5/055002).
- `chwialkowskiKernelTest2016`: Kacper Chwialkowski, Heiko Strathmann and
  Arthur Gretton, "A Kernel Test of Goodness of Fit", preprint (2016).
  [arXiv:1602.02964](https://arxiv.org/abs/1602.02964).
- `ciccottiBlueMoon2005`: Giovanni Ciccotti, Raymond Kapral and Eric
  Vanden-Eijnden, "Blue Moon Sampling, Vectorial Reaction Coordinates, and
  Unbiased Constrained Dynamics", *ChemPhysChem* 6(9), 1809–1814 (2005).
  [doi:10.1002/cphc.200400669](https://doi.org/10.1002/cphc.200400669).
- `comerAdaptiveBiasing2015`: Jeffrey Comer et al., "The Adaptive Biasing
  Force Method: Everything You Always Wanted To Know but Were Afraid To
  Ask", *The Journal of Physical Chemistry B* 119(3), 1129–1151 (2015).
  [doi:10.1021/jp506633n](https://doi.org/10.1021/jp506633n).
- `craigChemicalReaction2005`: Ian R. Craig and David E. Manolopoulos,
  "Chemical reaction rates from ring polymer molecular dynamics", *The
  Journal of Chemical Physics* 122(8), 084106 (2005).
  [doi:10.1063/1.1850093](https://doi.org/10.1063/1.1850093).
- `decherchiOptimalTransport2023`: S. Decherchi and A. Cavalli, "Optimal
  Transport for Free Energy Estimation", *The Journal of Physical Chemistry
  Letters* 14(6), 1618–1625 (2023).
  [doi:10.1021/acs.jpclett.2c03523](https://doi.org/10.1021/acs.jpclett.2c03523).
- `delorenziEffectAnharmonicity1984`: G. DeLorenzi, C. P. Flynn and G.
  Jacucci, "Effect of anharmonicity on diffusion jump rates", *Physical
  Review B* 30(10), 5430–5448 (1984).
  [doi:10.1103/PhysRevB.30.5430](https://doi.org/10.1103/PhysRevB.30.5430).
- `denOtterCalculationFree1998`: W. K. den Otter and W. J. Briels, "The
  calculation of free-energy differences by constrained molecular-dynamics
  simulations", *The Journal of Chemical Physics* 109(11), 4139–4146 (1998).
  [doi:10.1063/1.477019](https://doi.org/10.1063/1.477019).
- `deshpandeMaxSliced2019`: Ishan Deshpande et al., "Max-Sliced Wasserstein
  Distance and its use for GANs", preprint (2019).
  [arXiv:1904.05877](https://arxiv.org/abs/1904.05877).
- `duDoobsLagrangian2024`: Yuanqi Du et al., "Doob's Lagrangian: A
  Sample-Efficient Variational Approach to Transition Path Sampling", in
  *Advances in Neural Information Processing Systems 37* (2024).
  [arXiv:2410.07974](https://arxiv.org/abs/2410.07974).
- `duncanGeometryStein2019`: A. Duncan, N. Nuesken and L. Szpruch, "On the
  geometry of Stein variational gradient descent", preprint (2019).
  [arXiv:1912.00894](https://arxiv.org/abs/1912.00894).
- `durmusRiemannianStochastic2021`: Alain Durmus et al., "On Riemannian
  Stochastic Approximation Schemes with Fixed Step-Size", in *Proceedings of
  the 24th International Conference on Artificial Intelligence and
  Statistics*, Proceedings of Machine Learning Research 130, 1018–1026
  (2021). [arXiv:2102.07586](https://arxiv.org/abs/2102.07586).
- `eFiniteTemperature2005`: Weinan E, Weiqing Ren and Eric Vanden-Eijnden,
  "Finite Temperature String Method for the Study of Rare Events", *The
  Journal of Physical Chemistry B* 109(14), 6688–6693 (2005).
  [doi:10.1021/jp0455430](https://doi.org/10.1021/jp0455430).
- `eMetastabilityConformation2004`: Weinan E and Eric Vanden-Eijnden,
  "Metastability, conformation dynamics, and transition pathways in complex
  systems", in *Multiscale Modelling and Simulation*, Lecture Notes in
  Computational Science and Engineering, 35–68 (2004).
  [doi:10.1007/978-3-642-18756-8_3](https://doi.org/10.1007/978-3-642-18756-8_3).
- `eTransitionPathways2005`: Weinan E, Weiqing Ren and Eric Vanden-Eijnden,
  "Transition pathways in complex systems: Reaction coordinates,
  isocommittor surfaces, and transition tubes", *Chemical Physics Letters*
  413(1-3), 242–247 (2005).
  [doi:10.1016/j.cplett.2005.07.084](https://doi.org/10.1016/j.cplett.2005.07.084).
- `feibelmanDiffusionPath1990`: Peter J. Feibelman, "Diffusion path for an
  Al adatom on Al(001)", *Physical Review Letters* 65(6), 729–732 (1990).
  [doi:10.1103/PhysRevLett.65.729](https://doi.org/10.1103/PhysRevLett.65.729).
- `flyvbjergErrorEstimates1989`: H. Flyvbjerg and H. G. Petersen, "Error
  estimates on averages of correlated data", *The Journal of Chemical
  Physics* 91(1), 461–466 (1989).
  [doi:10.1063/1.457480](https://doi.org/10.1063/1.457480).
- `foilesEvaluationHarmonic1994`: Stephen M. Foiles, "Evaluation of harmonic
  methods for calculating the free energy of defects in solids", *Physical
  Review B* 49(21), 14930–14938 (1994).
  [doi:10.1103/PhysRevB.49.14930](https://doi.org/10.1103/PhysRevB.49.14930).
- `frishmanOptimizedPlanar1994`: Anatoli M. Frishman, Alexander M.
  Berezhkovskii and Eli Pollak, "Optimized planar dividing surfaces for
  asymmetric activated-rate processes", *Physical Review E* 49(2), 1216–1224
  (1994).
  [doi:10.1103/PhysRevE.49.1216](https://doi.org/10.1103/PhysRevE.49.1216).
- `garbunoInigoAffineInvariant2019`: Alfredo Garbuno-Inigo, Nikolas Nüsken
  and Sebastian Reich, "Affine invariant interacting Langevin dynamics for
  Bayesian inference", preprint (2019).
  [arXiv:1912.02859](https://arxiv.org/abs/1912.02859).
- `garbunoInigoInteractingLangevin2019`: Alfredo Garbuno-Inigo et al.,
  "Interacting Langevin Diffusions: Gradient Structure And Ensemble Kalman
  Sampler", preprint (2019).
  [arXiv:1903.08866](https://arxiv.org/abs/1903.08866).
- `glenskBreakdownArrhenius2014`: A. Glensk et al., "Breakdown of the
  Arrhenius Law in Describing Vacancy Formation Energies: The Importance of
  Local Anharmonicity Revealed byAb initioThermodynamics", *Physical Review
  X* 4(1), 011018 (2014).
  [doi:10.1103/PhysRevX.4.011018](https://doi.org/10.1103/PhysRevX.4.011018).
- `gonzalezLafontVariationalTransition1998`: Angels González-Lafont et al.,
  "Variational Transition State Theory and Tunneling Calculations with
  Reorientation of the Generalized Transition States for Methyl Cation
  Transfer", *The Journal of Physical Chemistry A* 102(19), 3420–3428
  (1998). [doi:10.1021/jp9807672](https://doi.org/10.1021/jp9807672).
- `gorhamMeasuringSample2017`: Jackson Gorham and Lester Mackey, "Measuring
  Sample Quality with Kernels", preprint (2017).
  [arXiv:1703.01717](https://arxiv.org/abs/1703.01717).
- `goswamiAdaptivePruning2026`: Rohit Goswami and Hannes Jónsson, "Adaptive
  Pruning for Increased Robustness and Reduced Computational Overhead in
  Gaussian Process Accelerated Saddle Point Searches", *ChemPhysChem* 27(4),
  e202500730 (2026).
  [doi:10.1002/cphc.202500730](https://doi.org/10.1002/cphc.202500730).
- `goswamiEfficientExploration2025`: Rohit Goswami, "Efficient Exploration
  of Chemical Kinetics", preprint (2025).
  [arXiv:2510.21368](https://arxiv.org/abs/2510.21368). PhD thesis,
  University of Iceland.
- `goswamiEfficientImplementation2025`: Rohit Goswami et al., "Efficient
  Implementation of Gaussian Process Regression Accelerated Saddle Point
  Searches with Application to Molecular Reactions", *Journal of Chemical
  Theory and Computation* 21(16), 7935–7943 (2025).
  [doi:10.1021/acs.jctc.5c00866](https://doi.org/10.1021/acs.jctc.5c00866).
- `goswamiEnhancedClimbing2026`: Rohit Goswami, Miha Gunde and Hannes
  Jónsson, "Enhanced climbing image nudged elastic band method with Hessian
  eigenmode alignment", *Frontiers in Chemistry* 14, 1807063 (2026).
  [doi:10.3389/fchem.2026.1807063](https://doi.org/10.3389/fchem.2026.1807063).
- `goswamiTheochemUIEOn2026`: Rohit Goswami et al., "TheochemUI/eOn:
  v3.6.0", Zenodo (2026).
  [doi:10.5281/zenodo.18529495](https://doi.org/10.5281/zenodo.18529495).
  Software release, version v3.6.0.
- `goswamiTutorialReview2026`: Rohit Goswami, "A Tutorial Review of Bayesian
  Optimization with Gaussian Processes to Accelerate Stationary Point
  Searches", *ACS Physical Chemistry Au* 6(4), 633–664 (2026).
  [doi:10.1021/acsphyschemau.6c00038](https://doi.org/10.1021/acsphyschemau.6c00038).
- `henkelmanDimerMethod1999`: Graeme Henkelman and Hannes Jónsson, "A dimer
  method for finding saddle points on high dimensional potential surfaces
  using only first derivatives", *The Journal of Chemical Physics* 111(15),
  7010–7022 (1999).
  [doi:10.1063/1.480097](https://doi.org/10.1063/1.480097).
- `holdijkStochasticOptimal2022`: Lars Holdijk et al., "Stochastic Optimal
  Control for Collective Variable Free Sampling of Molecular Transition
  Paths", preprint (2022).
  [arXiv:2207.02149](https://arxiv.org/abs/2207.02149).
- `horiutiStatisticalMechanical1938`: Juro Horiuti, "On the Statistical
  Mechanical Treatment of the Absolute Rate of Chemical Reaction", *Bulletin
  of the Chemical Society of Japan* 13(1), 210–216 (1938).
  [doi:10.1246/bcsj.13.210](https://doi.org/10.1246/bcsj.13.210).
- `hummerTransitionPaths2004`: Gerhard Hummer, "From transition paths to
  transition states and rate coefficients", *The Journal of Chemical
  Physics* 120(2), 516–523 (2004).
  [doi:10.1063/1.1630572](https://doi.org/10.1063/1.1630572).
- `jarzynskiTargetedFree2002`: C. Jarzynski, "Targeted free energy
  perturbation", *Physical Review E* 65(4), 046122 (2002).
  [doi:10.1103/PhysRevE.65.046122](https://doi.org/10.1103/PhysRevE.65.046122).
- `johannessonOptimizationHyperplanar2001`: Gísli H. Jóhannesson and Hannes
  Jónsson, "Optimization of hyperplanar transition states", *The Journal of
  Chemical Physics* 115(21), 9644–9656 (2001).
  [doi:10.1063/1.1415499](https://doi.org/10.1063/1.1415499).
- `jonssonSimulationSurface2011`: Hannes Jónsson, "Simulation of surface
  processes", *Proceedings of the National Academy of Sciences* 108(3),
  944–949 (2011).
  [doi:10.1073/pnas.1006670108](https://doi.org/10.1073/pnas.1006670108).
- `jordanVariationalFormulation1998`: Richard Jordan, David Kinderlehrer and
  Felix Otto, "The Variational Formulation of the Fokker–Planck Equation",
  *SIAM Journal on Mathematical Analysis* 29(1), 1–17 (1998).
  [doi:10.1137/S0036141096303359](https://doi.org/10.1137/S0036141096303359).
- `jungMachineguidedPath2023`: Hendrik Jung et al., "Machine-guided path
  sampling to discover mechanisms of molecular self-organization", *Nature
  Computational Science* 3(4), 334–345 (2023).
  [doi:10.1038/s43588-023-00428-z](https://doi.org/10.1038/s43588-023-00428-z).
- `kangComputingCommittor2024`: Peilin Kang, Enrico Trizio and Michele
  Parrinello, "Computing the committor with the committor to study the
  transition state ensemble", *Nature Computational Science* 4(6), 451–460
  (2024).
  [doi:10.1038/s43588-024-00645-0](https://doi.org/10.1038/s43588-024-00645-0).
- `keckVariationalTheory1967`: James C. Keck, "Variational Theory of
  Reaction Rates", in *Advances in Chemical Physics*, 85–121 (1967).
  [doi:10.1002/9780470140154.ch5](https://doi.org/10.1002/9780470140154.ch5).
- `khooSolvingHighdimensional2019`: Yuehaw Khoo, Jianfeng Lu and Lexing
  Ying, "Solving for high-dimensional committor functions using artificial
  neural networks", *Research in the Mathematical Sciences* 6(1), 1 (2019).
  [doi:10.1007/s40687-018-0160-2](https://doi.org/10.1007/s40687-018-0160-2).
- `koistinenNudgedElastic2017`: Olli-Pekka Koistinen et al., "Nudged elastic
  band calculations accelerated with Gaussian process regression", *The
  Journal of Chemical Physics* 147(15), 152720 (2017).
  [doi:10.1063/1.4986787](https://doi.org/10.1063/1.4986787).
- `kramersBrownianMotion1940`: H.A. Kramers, "Brownian motion in a field of
  force and the diffusion model of chemical reactions", *Physica* 7(4),
  284–304 (1940).
  [doi:10.1016/S0031-8914(40)90098-2](https://doi.org/10.1016/S0031-8914%2840%2990098-2).
- `kudinFreeEnergy2005`: Konstantin N. Kudin and Roberto Car, "Free energy
  profile along a discretized reaction path via the hyperplane constraint
  force and torque", *The Journal of Chemical Physics* 122(11), 114108
  (2005). [doi:10.1063/1.1874832](https://doi.org/10.1063/1.1874832).
- `lechnerNonlinearReaction2010`: Wolfgang Lechner et al., "Nonlinear
  reaction coordinate analysis in the reweighted path ensemble", *The
  Journal of Chemical Physics* 133(17), 174110 (2010).
  [doi:10.1063/1.3491818](https://doi.org/10.1063/1.3491818).
- `leimkuhlerEfficientMolecular2016`: Benedict Leimkuhler and Charles
  Matthews, "Efficient molecular dynamics using geodesic integration and
  solvent–solute splitting", *Proceedings of the Royal Society A:
  Mathematical, Physical and Engineering Sciences* 472(2189), 20160138
  (2016).
  [doi:10.1098/rspa.2016.0138](https://doi.org/10.1098/rspa.2016.0138).
- `leimkuhlerRationalConstruction2012`: B. Leimkuhler and C. Matthews,
  "Rational Construction of Stochastic Numerical Methods for Molecular
  Sampling", *Applied Mathematics Research eXpress* (2012).
  [doi:10.1093/amrx/abs010](https://doi.org/10.1093/amrx/abs010).
- `lelievreLongtimeConvergence2008`: Tony Lelièvre, Mathias Rousset and
  Gabriel Stoltz, "Long-time convergence of an adaptive biasing force
  method", *Nonlinearity* 21(6), 1155–1181 (2008).
  [doi:10.1088/0951-7715/21/6/001](https://doi.org/10.1088/0951-7715/21/6/001).
- `lelievreTwoMathematical2012`: Tony Lelièvre, "Two mathematical tools to
  analyze metastable stochastic processes", preprint (2012).
  [arXiv:1201.3775](https://arxiv.org/abs/1201.3775).
- `lesageSmoothedBiasing2017`: Adrien Lesage et al., "Smoothed Biasing
  Forces Yield Unbiased Free Energies with the Extended-System Adaptive
  Biasing Force Method", *The Journal of Physical Chemistry B* 121(15),
  3676–3685 (2017).
  [doi:10.1021/acs.jpcb.6b10055](https://doi.org/10.1021/acs.jpcb.6b10055).
- `liashkoEffectTemperature2017`: Sergei Y Liashko, Hannes Jónsson and
  Valery M Uzdin, "The effect of temperature and external field on
  transitions in elements of kagome spin ice", *New Journal of Physics*
  19(11), 113008 (2017).
  [doi:10.1088/1367-2630/aa8b96](https://doi.org/10.1088/1367-2630/aa8b96).
- `liComputingCommittor2019`: Qianxiao Li, Bo Lin and Weiqing Ren,
  "Computing committor functions for the study of rare events using deep
  learning", *The Journal of Chemical Physics* 151(5), 054112 (2019).
  [doi:10.1063/1.5110439](https://doi.org/10.1063/1.5110439).
- `liNaturalGradient2018`: Wuchen Li and Guido Montúfar, "Natural gradient
  via optimal transport", *Information Geometry* 1(2), 181–214 (2018).
  [doi:10.1007/s41884-018-0015-3](https://doi.org/10.1007/s41884-018-0015-3).
- `liuKernelizedStein2016`: Qiang Liu, Jason D. Lee and Michael I. Jordan,
  "A Kernelized Stein Discrepancy for Goodness-of-fit Tests and Model
  Evaluation", preprint (2016).
  [arXiv:1602.03253](https://arxiv.org/abs/1602.03253).
- `liuSteinVariational2016`: Qiang Liu and Dilin Wang, "Stein Variational
  Gradient Descent: A General Purpose Bayesian Inference Algorithm", in
  *Advances in Neural Information Processing Systems 29* (2016).
  [arXiv:1608.04471](https://arxiv.org/abs/1608.04471).
- `liuSteinVariational2017`: Qiang Liu, "Stein Variational Gradient Descent
  as Gradient Flow", preprint (2017).
  [arXiv:1704.07520](https://arxiv.org/abs/1704.07520).
- `maAutomaticMethod2005`: Ao Ma and Aaron R. Dinner, "Automatic Method for
  Identifying Reaction Coordinates in Complex Systems", *The Journal of
  Physical Chemistry B* 109(14), 6769–6779 (2005).
  [doi:10.1021/jp045546c](https://doi.org/10.1021/jp045546c).
- `makarovReactionRate1997`: Dmitrii E. Makarov and Horia Metiu, "The
  reaction rate constant in a system with localized trajectories in the
  transition region: Classical and quantum dynamics", *The Journal of
  Chemical Physics* 107(19), 7787–7799 (1997).
  [doi:10.1063/1.475092](https://doi.org/10.1063/1.475092).
- `maronssonMethodFinding2012`: Jón Bergmann Maronsson, Hannes Jónsson and
  Tejs Vegge, "A method for finding the ridge between saddle points applied
  to rare event rate estimates", *Physical Chemistry Chemical Physics*
  14(8), 2884 (2012).
  [doi:10.1039/c2cp23421a](https://doi.org/10.1039/c2cp23421a).
- `martinGeneralProcedure1982`: David L. Martin and L. M. Raff, "A general
  procedure for classical variational rate calculations for three-body
  exchange reactions", *The Journal of Chemical Physics* 77(3), 1235–1247
  (1982). [doi:10.1063/1.443999](https://doi.org/10.1063/1.443999).
- `mccannConvexityPrinciple1997`: Robert J. McCann, "A Convexity Principle
  for Interacting Gases", *Advances in Mathematics* 128(1), 153–179 (1997).
  [doi:10.1006/aima.1997.1634](https://doi.org/10.1006/aima.1997.1634).
- `mendelsCollectiveVariables2018`: Dan Mendels, GiovanniMaria Piccini and
  Michele Parrinello, "Collective Variables from Local Fluctuations", *The
  Journal of Physical Chemistry Letters* 9(11), 2776–2781 (2018).
  [doi:10.1021/acs.jpclett.8b00733](https://doi.org/10.1021/acs.jpclett.8b00733).
- `messinaQuantumActivated1993`: Michael Messina, Gregory K. Schenter and
  Bruce C. Garrett, "Quantum activated rate theory: Variational optimization
  of planar dividing surfaces", *The Journal of Chemical Physics* 99(11),
  8644–8653 (1993).
  [doi:10.1063/1.465588](https://doi.org/10.1063/1.465588).
- `millerSymplecticQuaternion2002`: T. F. Miller et al., "Symplectic
  quaternion scheme for biophysical molecular dynamics", *The Journal of
  Chemical Physics* 116(20), 8649–8659 (2002).
  [doi:10.1063/1.1473654](https://doi.org/10.1063/1.1473654).
- `millsReversibleWork1995`: Gregory Mills, Hannes Jónsson and Gregory K.
  Schenter, "Reversible work transition state theory: application to
  dissociative adsorption of hydrogen", *Surface Science* 324(2-3), 305–337
  (1995).
  [doi:10.1016/0039-6028(94)00731-4](https://doi.org/10.1016/0039-6028%2894%2900731-4).
- `miraZeroVariance2013`: Antonietta Mira, Reza Solgi and Daniele Imparato,
  "Zero variance Markov chain Monte Carlo for Bayesian estimators",
  *Statistics and Computing* 23(5), 653–662 (2013).
  [doi:10.1007/s11222-012-9344-6](https://doi.org/10.1007/s11222-012-9344-6).
- `mishinCalculationPointDefect2001`: Yuri Mishin, Mads R. Sørensen and
  Arthur F. Voter, "Calculation of point-defect entropy in metals",
  *Philosophical Magazine A* 81(11), 2591–2612 (2001).
  [doi:10.1080/01418610108216657](https://doi.org/10.1080/01418610108216657).
- `mishinStructuralStability2001`: Y. Mishin et al., "Structural stability
  and lattice defects in copper: Ab initio, tight-binding, and embedded-atom
  calculations", *Physical Review B* 63(22), 224106 (2001).
  [doi:10.1103/PhysRevB.63.224106](https://doi.org/10.1103/PhysRevB.63.224106).
- `mokkonenEfficientDynamical2016`: Harri Mökkönen, Tapio Ala-Nissila and
  Hannes Jónsson, "Efficient dynamical correction of the transition state
  theory rate estimate for a flat energy barrier", *The Journal of Chemical
  Physics* 145(9), 094901 (2016).
  [doi:10.1063/1.4962167](https://doi.org/10.1063/1.4962167).
- `mokkonenPolymerEscape2014`: Harri Mökkönen et al., "Polymer escape from a
  confining potential", *The Journal of Chemical Physics* 140(5), 054907
  (2014). [doi:10.1063/1.4863920](https://doi.org/10.1063/1.4863920).
- `mokkonenTransitionState2015`: Harri Mökkönen et al., "Transition state
  theory approach to polymer escape from a one dimensional potential well",
  *The Journal of Chemical Physics* 142(22), 224906 (2015).
  [doi:10.1063/1.4921959](https://doi.org/10.1063/1.4921959).
- `mullerDuplicationCollapse2018`: Gideon P. Müller et al., "Duplication,
  Collapse, and Escape of Magnetic Skyrmions Revealed Using a Systematic
  Saddle Point Search Method", *Physical Review Letters* 121(19), 197202
  (2018).
  [doi:10.1103/PhysRevLett.121.197202](https://doi.org/10.1103/PhysRevLett.121.197202).
- `namakianTemperatureDependent2023`: Reza Namakian, Dorel Moldovan and
  Thomas D. Swinburne, "Temperature dependent stacking fault free energy
  profiles and partial dislocation separation in FCC Cu", *Computational
  Materials Science* 218, 111971 (2023).
  [doi:10.1016/j.commatsci.2022.111971](https://doi.org/10.1016/j.commatsci.2022.111971).
- `neriaSimulationActivation1996`: Eyal Neria, Stefan Fischer and Martin
  Karplus, "Simulation of activation free energies in molecular systems",
  *The Journal of Chemical Physics* 105(5), 1902–1921 (1996).
  [doi:10.1063/1.472061](https://doi.org/10.1063/1.472061).
- `noeBoltzmannGenerators2019`: Frank Noé et al., "Boltzmann generators:
  Sampling equilibrium states of many-body systems with deep learning",
  *Science* 365(6457), eaaw1147 (2019).
  [doi:10.1126/science.aaw1147](https://doi.org/10.1126/science.aaw1147).
- `oatesControlFunctionals2017`: Chris J. Oates, Mark Girolami and Nicolas
  Chopin, "Control Functionals for Monte Carlo Integration", *Journal of the
  Royal Statistical Society Series B: Statistical Methodology* 79(3),
  695–718 (2017).
  [doi:10.1111/rssb.12185](https://doi.org/10.1111/rssb.12185).
- `okuyamaYoshidaStructureOptimization2000`: Naoto Okuyama-Yoshida et al.,
  "Structure optimization via free energy gradient method: Application to
  glycine zwitterion in aqueous solution", *The Journal of Chemical Physics*
  113(9), 3519–3524 (2000).
  [doi:10.1063/1.1287785](https://doi.org/10.1063/1.1287785).
- `olkinDistanceTwo1982`: I. Olkin and F. Pukelsheim, "The distance between
  two random vectors with given dispersion matrices", *Linear Algebra and
  its Applications* 48, 257–263 (1982).
  [doi:10.1016/0024-3795(82)90112-4](https://doi.org/10.1016/0024-3795%2882%2990112-4).
- `ottoGeneralizationInequality2000`: F. Otto and C. Villani,
  "Generalization of an Inequality by Talagrand and Links with the
  Logarithmic Sobolev Inequality", *Journal of Functional Analysis* 173(2),
  361–400 (2000).
  [doi:10.1006/jfan.1999.3557](https://doi.org/10.1006/jfan.1999.3557).
- `paciVacancyMigration1992`: E Paci and G Ciccotti, "Vacancy migration
  rates by molecular dynamics with constraints", *Journal of Physics:
  Condensed Matter* 4(9), 2173–2184 (1992).
  [doi:10.1088/0953-8984/4/9/011](https://doi.org/10.1088/0953-8984/4/9/011).
- `parkAccelerationSilver2025`: Jiyoung Park et al., "Acceleration via
  silver step-size on Riemannian manifolds with applications to Wasserstein
  space", preprint (2025).
  [arXiv:2506.06160](https://arxiv.org/abs/2506.06160).
- `patySubspaceRobust2019`: François-Pierre Paty and Marco Cuturi, "Subspace
  Robust Wasserstein Distances", preprint (2019).
  [arXiv:1901.08949](https://arxiv.org/abs/1901.08949).
- `petersenCommittorsReaction2026`: Magnus Petersen, Simon Lichtinger and
  Roberto Covino, "Committors and Reaction Rates from Trial Functions That
  Violate the Boundary Conditions", preprint (2026).
  [arXiv:2608.02536](https://arxiv.org/abs/2608.02536).
- `petersExtensionsLikelihood2007`: Baron Peters, Gregg T. Beckham and
  Bernhardt L. Trout, "Extensions to the likelihood maximization approach
  for finding reaction coordinates", *The Journal of Chemical Physics*
  127(3), 034109 (2007).
  [doi:10.1063/1.2748396](https://doi.org/10.1063/1.2748396).
- `petersObtainingReaction2006`: Baron Peters and Bernhardt L. Trout,
  "Obtaining reaction coordinates by likelihood maximization", *The Journal
  of Chemical Physics* 125(5), 054108 (2006).
  [doi:10.1063/1.2234477](https://doi.org/10.1063/1.2234477).
- `petersReactionCoordinates2016`: Baron Peters, "Reaction Coordinates and
  Mechanistic Hypothesis Tests", *Annual Review of Physical Chemistry*
  67(1), 669–690 (2016).
  [doi:10.1146/annurev-physchem-040215-112215](https://doi.org/10.1146/annurev-physchem-040215-112215).
- `pollakTheoryActivated1986`: Eli Pollak, "Theory of activated rate
  processes: A new derivation of Kramers' expression", *The Journal of
  Chemical Physics* 85(2), 865–867 (1986).
  [doi:10.1063/1.451294](https://doi.org/10.1063/1.451294).
- `polyakAccelerationStochastic1992`: B. T. Polyak and A. B. Juditsky,
  "Acceleration of Stochastic Approximation by Averaging", *SIAM Journal on
  Control and Optimization* 30(4), 838–855 (1992).
  [doi:10.1137/0330046](https://doi.org/10.1137/0330046).
- `pozunOptimizingTransition2012`: Zachary D. Pozun et al., "Optimizing
  transition states via kernel-based machine learning", *The Journal of
  Chemical Physics* 136(17), 174101 (2012).
  [doi:10.1063/1.4707167](https://doi.org/10.1063/1.4707167).
- `reFiorentinMethodologicalFrameworks2026`: Michele Re Fiorentin et al.,
  "Methodological Frameworks for Computational Electrocatalysis: From Theory
  to Practice", *Small Methods* 10(5), e01542 (2026).
  [doi:10.1002/smtd.202501542](https://doi.org/10.1002/smtd.202501542).
- `rizziTargetedFree2021`: Andrea Rizzi, Paolo Carloni and Michele
  Parrinello, "Targeted Free Energy Perturbation Revisited: Accurate Free
  Energies from Mapped Reference Potentials", *The Journal of Physical
  Chemistry Letters* 12(39), 9449–9454 (2021).
  [doi:10.1021/acs.jpclett.1c02135](https://doi.org/10.1021/acs.jpclett.1c02135).
- `robbinsStochasticApproximation1951`: Herbert Robbins and Sutton Monro, "A
  Stochastic Approximation Method", *The Annals of Mathematical Statistics*
  22(3), 400–407 (1951).
  [doi:10.1214/aoms/1177729586](https://doi.org/10.1214/aoms/1177729586).
- `robertsOptimalScaling1998`: Gareth O. Roberts and Jeffrey S. Rosenthal,
  "Optimal Scaling of Discrete Approximations to Langevin Diffusions",
  *Journal of the Royal Statistical Society Series B: Statistical
  Methodology* 60(1), 255–268 (1998).
  [doi:10.1111/1467-9868.00123](https://doi.org/10.1111/1467-9868.00123).
- `rotskoffActiveImportance2022`: Grant M. Rotskoff, Andrew R. Mitchell and
  Eric Vanden-Eijnden, "Active Importance Sampling for Variational
  Objectives Dominated by Rare Events: Consequences for Optimization and
  Generalization", in *Proceedings of the 2nd Mathematical and Scientific
  Machine Learning Conference*, Proceedings of Machine Learning Research
  145, 757–780 (2022). [arXiv:2008.06334](https://arxiv.org/abs/2008.06334).
- `ryckaertNumericalIntegration1977`: Jean-Paul Ryckaert, Giovanni Ciccotti
  and Herman J.C Berendsen, "Numerical integration of the cartesian
  equations of motion of a system with constraints: molecular dynamics of
  n-alkanes", *Journal of Computational Physics* 23(3), 327–341 (1977).
  [doi:10.1016/0021-9991(77)90098-5](https://doi.org/10.1016/0021-9991%2877%2990098-5).
- `satoAnharmonicEffect2021`: Yuji Sato et al., "Anharmonic effect on the
  thermally activated migration of $\{10\bar{1}2\}$ twin interfaces in
  magnesium", *Materials Research Letters* 9(5), 231–238 (2021).
  [doi:10.1080/21663831.2021.1875079](https://doi.org/10.1080/21663831.2021.1875079).
- `schenterGeneralizedTransition2003`: Gregory K. Schenter, Bruce C. Garrett
  and Donald G. Truhlar, "Generalized transition state theory in terms of
  the potential of mean force", *The Journal of Chemical Physics* 119(12),
  5828–5833 (2003).
  [doi:10.1063/1.1597477](https://doi.org/10.1063/1.1597477).
- `schenterReversibleWork1994`: Gregory K. Schenter, Gregory Mills and
  Hannes Jónsson, "Reversible work based quantum transition state theory",
  *The Journal of Chemical Physics* 101(10), 8964–8971 (1994).
  [doi:10.1063/1.468447](https://doi.org/10.1063/1.468447).
- `schraftNeuralNetwork2018`: Philippe Schraft et al., "Neural network
  approach to time-dependent dividing surfaces in classical reaction
  dynamics", *Physical Review E* 97(4), 042309 (2018).
  [doi:10.1103/PhysRevE.97.042309](https://doi.org/10.1103/PhysRevE.97.042309).
- `shirtsStatisticallyOptimal2008`: Michael R. Shirts and John D. Chodera,
  "Statistically optimal analysis of samples from multiple equilibrium
  states", *The Journal of Chemical Physics* 129(12), 124105 (2008).
  [doi:10.1063/1.2978177](https://doi.org/10.1063/1.2978177).
- `sivakDiffusionCharacteristics2022`: A. B. Sivak, D. N. Demidov and P. A.
  Sivak, "Diffusion Characteristics of Self-Point Defects in Copper:
  Molecular Dynamic Study", *Physics of Atomic Nuclei* 85(7), 1245–1255
  (2022).
  [doi:10.1134/S1063778822070171](https://doi.org/10.1134/S1063778822070171).
- `smidstrupImprovedInitial2014`: Søren Smidstrup et al., "Improved initial
  guess for minimum energy path calculations", *The Journal of Chemical
  Physics* 140(21), 214106 (2014).
  [doi:10.1063/1.4878664](https://doi.org/10.1063/1.4878664).
- `sorensenDiffusionMechanisms2000`: Mads R. Sørensen, Yuri Mishin and
  Arthur F. Voter, "Diffusion mechanisms in Cu grain boundaries", *Physical
  Review B* 62(6), 3658–3673 (2000).
  [doi:10.1103/PhysRevB.62.3658](https://doi.org/10.1103/PhysRevB.62.3658).
- `sorensenTemperatureAccelerated2000`: Mads R. Sørensen and Arthur F.
  Voter, "Temperature-accelerated dynamics for simulation of infrequent
  events", *The Journal of Chemical Physics* 112(21), 9599–9606 (2000).
  [doi:10.1063/1.481576](https://doi.org/10.1063/1.481576).
- `sprikFreeEnergy1998`: Michiel Sprik and Giovanni Ciccotti, "Free energy
  from constrained molecular dynamics", *The Journal of Chemical Physics*
  109(18), 7737–7744 (1998).
  [doi:10.1063/1.477419](https://doi.org/10.1063/1.477419).
- `swinburneUncertaintyAnharmonicity2021`: Thomas D. Swinburne, "Uncertainty
  and anharmonicity in thermally activated dynamics", *Computational
  Materials Science* 193, 110256 (2021).
  [doi:10.1016/j.commatsci.2020.110256](https://doi.org/10.1016/j.commatsci.2020.110256).
- `swinburneUnsupervisedCalculation2018`: Thomas D. Swinburne and
  Mihai-Cosmin Marinica, "Unsupervised Calculation of Free Energy Barriers
  in Large Crystalline Systems", *Physical Review Letters* 120(13), 135503
  (2018).
  [doi:10.1103/PhysRevLett.120.135503](https://doi.org/10.1103/PhysRevLett.120.135503).
- `takatsuWassersteinGeometry2008`: Asuka Takatsu, "On Wasserstein geometry
  of the space of Gaussian measures", preprint (2008).
  [arXiv:0801.2250](https://arxiv.org/abs/0801.2250).
- `taoComputingFree2014`: Peng Tao et al., "Computing the Free Energy along
  a Reaction Coordinate Using Rigid Body Dynamics", *Journal of Chemical
  Theory and Computation* 10(10), 4198–4207 (2014).
  [doi:10.1021/ct500342h](https://doi.org/10.1021/ct500342h).
- `tripuraneniAveragingStochastic2018`: Nilesh Tripuraneni et al.,
  "Averaging Stochastic Gradient Descent on Riemannian Manifolds", in
  *Proceedings of the 31st Conference on Learning Theory*, Proceedings of
  Machine Learning Research 75, 650–687 (2018).
  [arXiv:1802.09128](https://arxiv.org/abs/1802.09128).
- `truhlarVariationalTransition1984`: D G Truhlar and B C Garrett,
  "Variational Transition State Theory", *Annual Review of Physical
  Chemistry* 35(1), 159–189 (1984).
  [doi:10.1146/annurev.pc.35.100184.001111](https://doi.org/10.1146/annurev.pc.35.100184.001111).
- `vandenEijndenRevisitingFinite2009`: Eric Vanden-Eijnden and Maddalena
  Venturoli, "Revisiting the finite temperature string method for the
  calculation of reaction tubes and free energies", *The Journal of Chemical
  Physics* 130(19), 194103 (2009).
  [doi:10.1063/1.3130083](https://doi.org/10.1063/1.3130083).
- `vandenEijndenSecondorderIntegrators2006`: Eric Vanden-Eijnden and
  Giovanni Ciccotti, "Second-order integrators for Langevin equations with
  holonomic constraints", *Chemical Physics Letters* 429(1-3), 310–316
  (2006).
  [doi:10.1016/j.cplett.2006.07.086](https://doi.org/10.1016/j.cplett.2006.07.086).
- `vandenEijndenTransitionState2005`: Eric Vanden-Eijnden and Fabio A. Tal,
  "Transition state theory: Variational formulation, dynamical corrections,
  and error estimates", *The Journal of Chemical Physics* 123(18), 184103
  (2005). [doi:10.1063/1.2102898](https://doi.org/10.1063/1.2102898).
- `vanErpEfficiencyAnalysis2006`: Titus S. van Erp, "Efficiency analysis of
  reaction rate calculation methods using analytical models I: The
  two-dimensional sharp barrier", *The Journal of Chemical Physics* 125(17),
  174106 (2006). [doi:10.1063/1.2363996](https://doi.org/10.1063/1.2363996).
- `vanErpNovelPath2003`: Titus S. van Erp, Daniele Moroni and Peter G.
  Bolhuis, "A novel path sampling method for the calculation of rate
  constants", *The Journal of Chemical Physics* 118(17), 7762–7774 (2003).
  [doi:10.1063/1.1562614](https://doi.org/10.1063/1.1562614).
- `vempalaRapidConvergence2019`: Santosh S. Vempala and Andre Wibisono,
  "Rapid Convergence of the Unadjusted Langevin Algorithm: Isoperimetry
  Suffices", preprint (2019).
  [arXiv:1903.08568](https://arxiv.org/abs/1903.08568).
- `villaVariationalTransition1997`: Jordi Villà and Donald G. Truhlar,
  "Variational transition state theory without the minimum-energy path",
  *Theoretical Chemistry Accounts* 97(1-4), 317–323 (1997).
  [doi:10.1007/s002140050267](https://doi.org/10.1007/s002140050267).
- `vineyardFrequencyFactors1957`: George H. Vineyard, "Frequency factors and
  isotope effects in solid state rate processes", *Journal of Physics and
  Chemistry of Solids* 3(1-2), 121–127 (1957).
  [doi:10.1016/0022-3697(57)90059-8](https://doi.org/10.1016/0022-3697%2857%2990059-8).
- `voterAccurateInteratomic1986`: Arthur F. Voter and Shao Ping Chen,
  "Accurate Interatomic Potentials for Ni, Al and Ni$_3$Al", *MRS
  Proceedings* 82, 175 (1986).
  [doi:10.1557/PROC-82-175](https://doi.org/10.1557/PROC-82-175).
- `wibisonoSamplingOptimization2018`: Andre Wibisono, "Sampling as
  optimization in the space of measures: The Langevin dynamics as a
  composite optimization problem", preprint (2018).
  [arXiv:1802.08089](https://arxiv.org/abs/1802.08089).
- `wignerTransitionState1938`: E. Wigner, "The transition state method",
  *Transactions of the Faraday Society* 34, 29 (1938).
  [doi:10.1039/TF9383400029](https://doi.org/10.1039/TF9383400029).
- `willowLearnedMappings2023`: Soohaeng Yoo Willow, Lulu Kang and David D.
  L. Minh, "Learned mappings for targeted free energy perturbation between
  peptide conformations", *The Journal of Chemical Physics* 159(12), 124104
  (2023). [doi:10.1063/5.0164662](https://doi.org/10.1063/5.0164662).
- `wirnsbergerTargetedFree2020`: Peter Wirnsberger et al., "Targeted free
  energy estimation via learned mappings", *The Journal of Chemical Physics*
  153(14), 144112 (2020).
  [doi:10.1063/5.0018903](https://doi.org/10.1063/5.0018903).
- `zhangAbInitio2025`: Xi Zhang, Sergiy V. Divinski and Blazej Grabowski,
  "Ab initio machine-learning unveils strong anharmonicity in non-Arrhenius
  self-diffusion of tungsten", *Nature Communications* 16(1), 394 (2025).
  [doi:10.1038/s41467-024-55759-w](https://doi.org/10.1038/s41467-024-55759-w).
- `zhangUnifiedEfficient2019`: Zhijun Zhang et al., "Unified Efficient
  Thermostat Scheme for the Canonical Ensemble with Holonomic or Isokinetic
  Constraints via Molecular Dynamics", *The Journal of Physical Chemistry A*
  123(28), 6056–6079 (2019).
  [doi:10.1021/acs.jpca.9b02771](https://doi.org/10.1021/acs.jpca.9b02771).
- `zhuoMessagePassing2018`: Jingwei Zhuo et al., "Message Passing Stein
  Variational Gradient Descent", in *Proceedings of the 35th International
  Conference on Machine Learning* (2018).
  [arXiv:1711.04425](https://arxiv.org/abs/1711.04425).
- `zinovjevEnzymaticReaction2017`: Kirill Zinovjev, "Enzymatic reaction
  coordinates: methodological developments and applications", PhD thesis,
  Universitat de València (2017). No DOI or arXiv identifier; repository
  handle hdl:10550/58600.
- `zinovjevReactionCoordinates2018`: Kirill Zinovjev and Iñaki Tuñón,
  "Reaction coordinates and transition states in enzymatic catalysis",
  *WIREs Computational Molecular Science* 8(1), e1329 (2018).
  [doi:10.1002/wcms.1329](https://doi.org/10.1002/wcms.1329).
- `zinovjevTransitionState2015`: Kirill Zinovjev and Iñaki Tuñón,
  "Transition state ensemble optimization for reactions of arbitrary
  complexity", *The Journal of Chemical Physics* 143(13), 134111 (2015).
  [doi:10.1063/1.4931596](https://doi.org/10.1063/1.4931596).
