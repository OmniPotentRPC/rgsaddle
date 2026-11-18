# Jóhannesson–Jónsson Section III B reproduction

Source: G. H. Jóhannesson and H. Jónsson, “Optimization of hyperplanar
transition states,” *J. Chem. Phys.* **115**, 9644–9656 (2001),
doi:10.1063/1.1415499. The checked author-hosted PDF is
<https://hj.hi.is/papers/paperOHTST-JCP01.pdf> (SHA-256
`7392008f98919d708db90c4c07aa4ea076261dea5e8b6ae00d7fc8e57188c274`).
Page numbers below are the printed journal pages.

## What the paper states

Potential (Appendix B, pp. 9655–9656):

> “The two-dimensional potentials used in the test calculations shown in
> Figs. 6(a)–6(d) are based on a LEPS plus harmonic oscillator potential,
> \(V_{\mathrm{LEPS+HO}}\), shown in Fig. 6(a).”

Equation (B1), p. 9655, is

\[
V_{\mathrm{LEPS}}(r_{AB},r_{BC}) =
\frac{Q_{AB}}{1+a}+\frac{Q_{BC}}{1+b}+\frac{Q_{AC}}{1+c}
-\left[
\frac{J_{AB}^2}{(1+a)^2}+\frac{J_{BC}^2}{(1+b)^2}
+\frac{J_{AC}^2}{(1+c)^2}
-\frac{J_{AB}J_{BC}}{(1+a)(1+b)}
-\frac{J_{BC}J_{AC}}{(1+b)(1+c)}
-\frac{J_{AB}J_{AC}}{(1+a)(1+c)}
\right]^{1/2}.
\]

The pair terms are, exactly as Eqs. (B2) and (B3), p. 9655,

\[
Q(r)=\frac d2\left(\frac32e^{-2\alpha(r-r_0)}
-e^{-\alpha(r-r_0)}\right),
\qquad
J(r)=\frac d4\left(e^{-2\alpha(r-r_0)}
-6e^{-\alpha(r-r_0)}\right).
\]

The parameter sentence on p. 9655 states:

> “The parameters used in the calculations presented here are
> \(a=c=0.05\), \(b=0.80\),
> \(r_{0,AB}=r_{0,BC}=r_{0,AC}=0.742\),
> \(\alpha_{AB}=\alpha_{BC}=\alpha_{AC}=1.942\),
> \(d_{AB}=d_{BC}=4.746\), and \(d_{AC}=3.445\).”

The coupling is Eq. (B4), p. 9656,

\[
V_{\mathrm{LEPS+HO}}(x,y)=
V_{\mathrm{LEPS}}(x,r_{AC}-x)
+2k_c\left[x-\left(r_{AC}/2-y/c\right)\right]^2,
\]

followed by:

> “where \(k_c=0.2025\), \(r_{AC}=3.742\), and \(c=1.154\).”

Here the second \(c\) is the oscillator coordinate scale, distinct from the
LEPS parameter \(c=0.05\). The paper then states on p. 9656 that
“The coordinate \((x,y)=(2.02083,-0.172881)\) corresponds to the first
order saddle point on the \(V_{\mathrm{LEPS+HO}}\) surface.”

Temperature and barrier convention:

* Section III A, p. 9650, says: “the system was simulated at a very low
  temperature, \(k_BT=10^{-7}E_A\), where \(E_A\) is the energy of the
  saddle point with respect to \(R\).”
* Section III B, p. 9651, says: “The two-dimensional potential surface
  \(V_a\) shown in Fig. 6(a) was used to calculate the free energy barrier
  at a temperature of \(k_BT=0.1E_A\).”
* The same paragraph on p. 9651 says:
  “The ratio of the configuration integrals,
  \(Q^\ddagger/Q_{Z_R}\) was found to be \(0.765\pm0.005\) for \(V_a\) at
  \(k_BT=0.1E_A\). The value of \(\langle|v|\rangle\) was found to be
  0.399, so the rate constant is estimated to be
  \(k_{\mathrm{OH-TST}}=(4.5\pm0.2)\times10^{-7}\).”
* It continues: “harmonic TST gives a very similar value for the rate
  constant \(k_{\mathrm{hTST}}=4.3\times10^{-7}\).”

Velocity and mass are Eqs. (23) and (24), p. 9650:

\[
\frac{\langle|v|\rangle}{2}=\sqrt{\frac{k_BT}{2\pi m}},
\qquad
m=\sum_i m_i(n_x^2+n_y^2+n_z^2).
\]

The text calls \(m\) “the effective mass for motion across the dividing
surface in the direction of the normal.” The paper does **not** specify the
individual masses \(m_i\), a numerical effective mass, or physical units for
the Appendix B parameters or rates. It also does not print a numerical
\(E_A\).

Sampling details that are actually stated are on p. 9650:

> “In all calculations of canonical ensemble averages we use a single
> molecular dynamics trajectory. We simulate a system at a given
> temperature by applying stochastic collisions to the trajectory.”

Equation (22), p. 9649, estimates \(Q_{Z_R}/Q_R\) from trajectory crossings,
weighted by the inverse normal displacement. Section III A, p. 9650, says
“The integral was evaluated by using the trapezoidal rule.” The paper does not
state the trajectory length, time step, collision frequency, number of
samples, or number of work planes.

## Term-by-term implementation check

`tests/common/ohtst_surfaces.rs::LepsHo` matches Eqs. (B1)–(B4):

| Paper term | Implementation |
|---|---|
| \(r_{AB}=x,\ r_{BC}=r_{AC}-x\) | `x`, `R_AC - x` |
| three \(Q/(1+a,b,c)\) terms | `qab/sa + qbc/sb + qac/sc` |
| six-term exchange square root | `s = u²+w²+z²-uw-wz-uz; sqrt(s)` |
| \(Q(r)\), \(J(r)\) | `coulomb_exchange`, including analytic derivatives |
| \(2k_c[x-(r_{AC}/2-y/c)]^2\) | `2.0*K_C*stretch*stretch` |
| all Appendix B parameters | exact decimal values listed above |

No potential mismatch was found. The analytic Hessian now used for Vineyard
hTST differentiates those same terms; the reproduction does not use hand
finite differences.

## Convention decision

Evaluating Appendix B gives the reactant minimum
\((0.74152,1.30342)\), the printed saddle, and
\(E_A=3.63395\) in the potential's own energy unit. Thus the literal
\(k_BT=0.1E_A=0.363395\), with unit mass, gives
\(\langle|v|\rangle=0.480983\) and
\(k_{\mathrm{hTST}}=4.0410\times10^{-5}\), not the printed values.

Both independently printed quantities select \(k_BT=0.25\) in the Appendix B
energy unit with unit mass:

* Eq. (23): \(\sqrt{2k_BT/\pi}=0.398942\);
* analytic Vineyard hTST: \(4.33127\times10^{-7}\).

Therefore the numerical convention behind the printed rates is
\(k_BT=0.25\), unit mass, and the potential's own energy, length, and time
units. The prose \(0.1E_A\) is inconsistent with Appendix B by a factor
\(0.25/(0.1\times3.63395)=0.688\). The printed
\(Q^\ddagger/Q_{Z_R}=0.765\) is also mislabeled: Eq. (3), p. 9646, uses
\(Q_{Z_R}/Q_R\) in the prefactor, while \(Q^\ddagger/Q_{Z_R}\) is the
exponentially small reversible-work factor.

## Rates

The executable reproduction uses the paper's Eq. (22) crossing estimator,
sampled plane optimization, and trapezoidal reversible work. Because the
paper omits all sample counts, the test declares its own: eight independent
seeds; 1000 records per plane-optimization sample; 8000 records for the
crossing trajectory; and 2000 records on each of 17 work planes. Rgsaddle's
BAOAB thermostat samples the same canonical distribution; an exact
Andersen-collision schedule cannot be reconstructed because the paper does
not give its time step or collision frequency.

| Quantity | Result | Published |
|---|---:|---:|
| \(\langle|v|\rangle\) | 0.398942 | 0.399 |
| analytic Vineyard \(k_{\mathrm{hTST}}\) | \(4.33127\times10^{-7}\) | \(4.3\times10^{-7}\) |
| optimal-line quadrature \(k_{\mathrm{OH-TST}}\) | \(4.20148\times10^{-7}\) | — |
| Eq. (22) + trapezoidal work \(k_{\mathrm{OH-TST}}\) | \((4.340\pm0.132)\times10^{-7}\) | \((4.5\pm0.2)\times10^{-7}\) |

The uncertainty is the standard error of the mean over independent seeds.
The estimate lies in the unmodified published band
\([4.3,4.7]\times10^{-7}\). Simpson integration gives
\((4.211\pm0.136)\times10^{-7}\), 2.06% below its lower edge; the likely
source of the difference is the paper's trapezoidal discretization, since
the converged quadrature value is also just below the band.

## Verification

Test counts and log hashes are filled from the packaged logs after the
required full Rust, Meson consumer, and Python runs.
