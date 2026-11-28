# Adaptive OH-TST strings

## Scope

PAFI and OH-TST use the same constrained hyperplane samples but solve
different problems. PAFI keeps the planes normal to a supplied zero-temperature
path and translates them only along that path. The method in
`adaptive_ohtst` is OH-TST: it translates and rotates interior planes to make
a finite-temperature string. The PAFI implementation remains in `pafi`.

For an ambient dimension \(D\), the unit normal lies on \(S^{D-1}\), so this
implementation optimizes the full \(D-1=3N-1\) orientation coordinates and
one physical plane translation. Jóhannesson and Jónsson,
*J. Chem. Phys.* **115**, 9644 (2001), p. 9645, state that “the orientational
optimization is \(D-2\) dimensional” as a size comparison with the
one-dimensional reaction-coordinate optimization; that statement is not the
dimension of this implementation. In-plane translation of the plane point is
gauge.

## Sampled force and string update

For plane point \(p\), unit normal \(\nu\), and constrained canonical average,

\[
 A(p,\nu)=-kT\log\int\delta(\nu\mathbin{\cdot}(q-p))
                  e^{-V(q)/kT}\,dq ,
\]
\[
 \nabla_p A=\langle\nu\mathbin{\cdot}\nabla V\rangle\nu,\qquad
 \nabla_\nu A=-\langle(\nu\mathbin{\cdot}\nabla V)(q-p)\rangle .
\]

One `PlaneSample` supplies both blocks and their block covariance. The
rotation block is scaled by the sampled root-mean-square lever arm. At each
interior plane the component tangent to the string is removed. The remaining
gradient translates and rotates the plane; equal-arc redistribution is only
a parametrization step. A move is bounded by half the shorter neighboring
arc, so adjacent planes cannot cross.

No comparison of two independently noisy residuals is made. If
\(\widehat g_\perp\) is the projected mean gradient and \(C\) its
autocorrelation-aware batch covariance, a plane is statistically unresolved
when

\[
 \|\widehat g_\perp\|^2\leq\bigl(\sqrt{\operatorname{tr}C}+\gamma_k s\bigr)^2 .
\]

Here \(s\) is the unprojected gradient norm plus \(kT\) over the lever arm, and
\(k=8m\) for a gradient of length \(m\): four length-\(m\) dots (tangent norm,
unprojected norm, tangent coefficient, projected norm) and one axpy, at two
flops per component. Squaring a bare \(\gamma_k s\) would put it below the ulp
of a physical \(\|\widehat g_\perp\|^2\), so the comparison uses that expansion.
For resolved gradients, separate translation and rotation secants bound the
local curvature. Relaxation also stops when the quadratic residual free-energy
bound is below \(0.03\,kT/m\) for each of \(m\) interior planes. The trust
radius is half the neighboring separation, the largest step that cannot cross.
Floating-point cancellation elsewhere uses the same
\(\gamma_k=k\epsilon/(1-k\epsilon)\). The pilot drift comparison counts one
addition per record, two absolute values, one maximum, and the product with
\(\gamma\), so its operation count is \(n+4\).

## Sampling and uncertainty

The first sample has 14 records. The ordinary stationarity test is not
meaningful with so short a pilot. Instead, the split-half diagnostic compares
equal even-length ends, leaving the central pair out so stationary period-two
integration signals are phase balanced. It uses a familywise false-rejection probability
\(0.03/N\) for the \(N\) initial planes and Chebyshev's
\(1/\sqrt{\alpha}\) threshold. Each half's standard error is inflated by that
half's integrated autocorrelation time. A six-point half still includes lag 1;
longer series stop at the original cutoff \(\mathrm{lag}<n/4\). At least four
effective samples are required, so each half has two observations to estimate
its mean and variance. Fourteen records support \(\tau\le 14/8=1.75\). An AR(1) of
parameter \(\varphi\) has \(\tau=(1+\varphi)/(2(1-\varphi))\); \(\varphi=0.9\)
is \(\tau\approx 9.5\) at stride 1 and is rejected. When the effective-sample
gate fails, the pilot doubles its stride and burn-in and tries again, up to
14 times the configured stride. Retaining every \(s\)-th step replaces
\(\varphi\) by \(\varphi^s\). From the default stride of 5 the attempts are
10, 20, and 40; the next doubling would pass that cap. One step to stride
10 replaces \(\varphi=0.9\) by \(0.9^{10}\approx 0.35\), which has
\(\tau\approx 1.0\). The sampler must
honor that stride; the pilot is the correlated chain, not an independent
draw. A sampler that ignores stride still returns the effective-sample error
once the cap is hit. Production samples keep the configured stride; the
stride that let the pilot pass is not copied onto them, so a chain that
needed thinning can still spend its production budget at the original
spacing and finish at the record cap. A failed
pilot returns `NotEquilibrated` or `InsufficientEffectiveSamples`; it never
enters a profile. Production samples use automated transient discard. For a
retained tail of length \(r\), requested count \(b\), and correlation width
\(w=\lceil 2\tau\rceil\), the batch width is
\(\max(\lfloor r/b\rfloor, w, 1)\). The actual block count is the floor of
\(r\) divided by that width, and the remainder is dropped. Every retained
block has that same width. A width that leaves fewer than two blocks is an
error, rather than an empty or short block. The rule is the same for every
requested count. At \(b=2\) and \(r=15\), \(\lfloor r/b\rfloor=7\) gives two
blocks and drops the last record. \(\lceil r/b\rceil=8\) leaves only one
block, which is the failure this width avoids.

The effective-sample floor is the block count the caller passed in,
\(n_{\mathrm{eff}}\ge b\). It is not one literal for every caller. The plane
sampler and thermodynamic integration pass 20. PAFI passes 5. The adaptive
string passes 7 at 14 records and \(\lfloor n^{1/3}\rfloor\) above that, so
record counts 15 through 26 pass 2. The pilot keeps a separate gate of four
effective samples, two in each half, so that each half can estimate a mean
and a variance.

The production split-half comparison is an absolute four-standard-error
screen on one plane. On a stationary Gaussian series the statistic is the
two-sample \(t\) (equal half lengths make the Welch and pooled standard
errors the same). With \(\nu=n_1+n_2-2\) degrees of freedom, \(n_i\) the
number of block means in half \(i\), the two-sided false-rejection
probability at four standard errors is the Student-\(t\) tail
\(I_{\nu/(\nu+16)}(\nu/2,\,1/2)\). Two block means in each half, which is
what \(b=2\) produces, give \(\nu=2\) and a tail of 5.72%. Seven in each
half, a 14-record production sample, give \(\nu=12\) and a tail of 0.176%.
Six and seven, the neighbouring odd tail, give \(\nu=11\) and a tail of
0.209%, about 0.22% per plane. A run that tests several planes rejects when
any plane rejects. Those rejections stay visible. Bonferroni or Holm would
rescale this screen by the number of planes. The screen is not a family-wise
level-\(\alpha\) test: the pilot already spends the family-wise budget
\(0.03/N\). The production threshold stays at four standard errors.

This width is applied inside the shared engine-backed plane sampler, so every
engine hyperplane sample, not only the adaptive string, inherits the
correlation-width batching. Its reported standard errors can therefore differ
from estimates made with the caller's requested fixed block count. The
adaptive method requests seven blocks at 14 records and
\(\lfloor n^{1/3}\rfloor\) blocks above that, so both block width and the
number of variance degrees of freedom grow.

The required record count is

\[
 n'=\min\!\left(2048,
   \left\lceil1.1\,n\left({\operatorname{se}(\log k)\over0.03}\right)^2
   \right\rceil\right).
\]

A plane sample can reduce only
\(\operatorname{se}(\Delta A^\ddagger)/(kT)\). Reactant-density uncertainty
is propagated to the returned rate but is excluded from this growth rule, so
an independently noisy density cannot force futile hyperplane sampling.

A fresh production sample is drawn after the pilot, including when the
pilot predicts \(n'=14\). The production loop still returns the first
sample whose own standard error lies inside the target, so that reported
error is conditional on stopping. For iid noise with standard deviation
0.25, on a chain that ends at the barrier, the conditional sampled error
averages 0.83 of the unconditional barrier deviation. The dividing plane
stays on the last plane and the string does not move, so the deficit is
the stopping rule, not geometry. A fixed record count, with the noise
small enough that the target never stops the run, recovers the chi
correction in the validation section to about one percent. The estimator
does not draw a confirmation sample: a second draw that was kept only when
its own error met the target would carry the same bias, and a confirmation
whose error exceeded the target would have no honest status below the cap.

At the cap the status is `SampleCap` whenever the total error is still
above the target. That includes a sample whose own error is already inside
the target while geometry or quadrature is not. The cap was reached, and
the combined target was not. `Unresolved` is that same split only below
the cap, where further records would not close the gap.

The free-energy profile integrates the sampled directional derivatives.
Pairs of segments use the three-node Lagrange (nonuniform Simpson) rule and
an unpaired segment uses trapezoidal integration. Applying the same weights
to every plane's block gradients propagates the sampling standard error.
`sampled_barrier_error`, `geometry_error`, and `quadrature_error` report the
three contributions separately; `barrier_error` combines them in quadrature.
`TargetMet` requires that combined energy error to be at most \(0.03\,kT\).
Geometry is the barrier change when each interior normal
is replaced by the midpoint of its neighbors, combined with the change under
a trust-limited Newton step. Quadrature is the absolute difference between
the profile integral and the same integral on every other node. When the
reactant and dividing-plane normals agree, it also includes the difference
between the barrier and a Simpson rule in the offset \(n\cdot p\), integrated
only up to that dividing plane. The free-energy drop on the product side of
an interior barrier is not a quadrature error. Either term is at least the
roundoff floor \(\gamma kT\) of the packed dot products in the integral.
While the even-grid gap exceeds both \(0.03\,kT\) and a Chebyshev band on its
own standard error, and doubling would not pass the plane cap, the string
inserts a sampled midpoint on every edge and integrates again. Sample growth
still uses only `sampled_barrier_error`.
The rate is

\[
 k=\sqrt{kT/(2\pi)}\,\rho_R\,e^{-\Delta A^\ddagger/kT},
\qquad
 \operatorname{se}(\log k)^2=
 \left({\operatorname{se}\rho_R\over\rho_R}\right)^2+
 \left({\operatorname{se}_{\rm total}\Delta A^\ddagger\over kT}\right)^2 .
\]

## Hard regions and multiplicity

Sampling uses the `Region` attached to `LangevinConfig`. Analytic BAOAB
reflects position and velocity, GHMC rejects a crossing proposal, and engine
backends consume the same region. No soft collective-variable restraint is
introduced.

A symmetry-unique region carries its exact degeneracy. The estimator first
computes the unique-region rate, then applies
`RateMultiplicity::with_region` once. That call sets `in_plane_images` and
leaves `paths` at 1. The caller applies `with_paths` to the returned rate
and must not call `with_region` again; a twelve-fold hop is `with_paths(12)`
at the call site. The estimator has no independent degeneracy field, so
dropping, doubling, or squaring the region factor changes the end-to-end
symmetry test.

The current plane-gradient records contain no wall-contact impulse. A bounded
adaptive string is therefore accepted only when all exact planar faces are
tangent to one common plane orientation and the update does not rotate that
orientation. Translation along the common normal then leaves the sampled
cross-section fixed. A non-planar predicate, an oblique face, or a rotating
bounded plane returns an explicit wall-contact-derivative error instead of
silently applying the infinite-plane torque formula.

## Validation

The analytic tests cover the abrupt low- to high-temperature rotation on the
twisted path of Bligaard and Jónsson, *Comput. Phys. Commun.* **169**, 284
(2005), a straight finite-temperature string, hard-reflected sampling of a
two-channel symmetric surface, and independent noisy samplers at noise
standard deviations 0.25 and 0.5. The noisy ensemble checks return rate and
profile-standard-error coverage. Fixed-count iid validation uses three
pre-registered, disjoint 200-seed blocks and a two-sided two-standard-error
interval. Coverage is the fraction of returned estimates. A floor rejection
is not a miss of that interval: fourteen records pass seven blocks, so
`n_eff` must be at least 7, and a literal floor of 4 fails that check.
Split-half rejections stay visible and are not counted as covered draws.
At nominal 95% coverage the binomial standard error for 200 returned
trials is 1.54 percentage points, so the band extends 3.2 standard errors
below nominal and 2.6 above it. The combined error is what the adaptive
coverage checks use, because geometry and quadrature sit on top of the
sampled term. The 0.83 ratio quoted above is the optional-stopping bias at
noise 0.25, not a property of the fixed-count estimator.
Cu production sampling remains an engine-backed benchmark, not part of
this generic method.

The orientation probe skips a Newton step whose normal and point move by
at most \(\gamma_{2D+1}\). A step that sits between \(\gamma_1\) and that
bound changes the reintegrated barrier by far less than the integral's
roundoff floor, and a neighbor force large enough to push the change over
the floor leaves the quadrature unresolved. Replacing that count by
\(\gamma_1\) does not change a returned estimate. The other named counts
are tested by a configuration that sits between \(\gamma_1\) and the count.
