
# Table of Contents

-   [Cargo feature](#orgbe0b808)
-   [Public types](#org8e012fd)
-   [Units and coordinates](#org7a5d4ac)
-   [Thermostat metadata and records](#orgfd8fa33)
-   [lammpc constrained MD extension](#org19f1589)
-   [Thermodynamic integration coupling](#orgcbfe966)
-   [Errors](#org9ddddcb)



<a id="orgbe0b808"></a>

# Cargo feature

`lammps` is opt-in and enables `ohtst`. It adds the pinned
`rgpot-core` profile dependency. `liblammpc` and `liblammps` are runtime
shared libraries and are not needed by default builds.


<a id="org8e012fd"></a>

# Public types

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Type</th>
<th scope="col" class="org-left">Contract</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left"><code>LammpsSystem</code></td>
<td class="org-left">Flat <code>PotentialConfig</code>, Cartesian positions, atomic numbers, masses, and optional cell</td>
</tr>

<tr>
<td class="org-left"><code>LammpsSamplerConfig</code></td>
<td class="org-left">Shared library path, BAOAB time step in ps, Langevin friction in inverse ps, thermostat kind, the excluded-mode count used in \(3N - 1 - n_{\mathrm{excluded}}\), and an optional Taylor reference for lambda mixing</td>
</tr>

<tr>
<td class="org-left"><code>LammpsThermodynamicReference</code></td>
<td class="org-left">Reference geometry, energy, gradient, and dense Hessian in the sampler coordinates of the plane</td>
</tr>

<tr>
<td class="org-left"><code>LammpsThermostatKind</code></td>
<td class="org-left">Thermostat enum stored in records; currently <code>BaoabLangevin</code></td>
</tr>

<tr>
<td class="org-left"><code>LammpsThermostatRecord</code></td>
<td class="org-left">Serializable thermostat metadata with <code>thermostat</code> and <code>effective_degrees_of_freedom</code></td>
</tr>

<tr>
<td class="org-left"><code>LammpsHyperplaneSampler</code></td>
<td class="org-left">Persistent lammpc session implementing <code>HyperplaneSampler</code></td>
</tr>
</tbody>
</table>

All configuration structs are `#[non_exhaustive]`, implement `Default`, and
provide `with_*` builders. A default system is incomplete and validation
returns `SaddleError::Invalid` or `SaddleError::Shape` before loading the
engine.


<a id="org7a5d4ac"></a>

# Units and coordinates

The rgpot `ForceInput` uses angstrom and eV. LAMMPS uses `metal` units.
The sampler controls use:

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Quantity</th>
<th scope="col" class="org-left">Unit</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left">time step</td>
<td class="org-left">ps</td>
</tr>

<tr>
<td class="org-left">friction</td>
<td class="org-left">ps<sup>-1</sup></td>
</tr>

<tr>
<td class="org-left">thermal energy</td>
<td class="org-left">eV</td>
</tr>

<tr>
<td class="org-left">mass</td>
<td class="org-left">amu</td>
</tr>
</tbody>
</table>

`HyperplaneCoordinates::MassWeighted` passes the plane in
$q=M^{1/2}x$. Returned positions are $q$, forces are
$F_q=M^{-1/2}F_x$, and velocities are $\dot q=M^{1/2}\dot x$.

`HyperplaneCoordinates::Cartesian` passes Cartesian point and normal. lammpc
uses the diagonal mass metric when restoring positions and projecting
velocity and force.


<a id="orgfd8fa33"></a>

# Thermostat metadata and records

Each `HyperplaneObservation` from `LammpsHyperplaneSampler` carries two stable
scalar observables:

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Observable</th>
<th scope="col" class="org-left">Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left"><code>lammps.thermostat</code></td>
<td class="org-left">Numeric code of <code>LammpsThermostatKind</code> (1 for <code>BaoabLangevin</code>)</td>
</tr>

<tr>
<td class="org-left"><code>lammps.effective_degrees_of_freedom</code></td>
<td class="org-left">\(3N - 1 - n_{\mathrm{excluded}}\)</td>
</tr>
</tbody>
</table>

`LammpsThermostatRecord::to_json` writes
`rgsaddle.lammps_thermostat` documents, and `from_json` reads the same
format.


<a id="org19f1589"></a>

# lammpc constrained MD extension

The adapter requires these additive C ABI functions:

```c
int lammpc_session_md_initialize(...);
int lammpc_session_md_step(LammpcSession *, size_t);
int lammpc_session_md_state(...);
```

`md_initialize` receives one rgpot `ForceInput`, a point, a normal, and
`LammpcMdConfig`. `md_step` advances exactly the requested number of BAOAB
steps. `md_state` returns positions, projected forces, removed velocities,
the scalar constraint force, the force-evaluation count, and the completed
step count.

LAMMPS hard-region sampling additionally requires lammpc to advertise that
`md_step` applies a registered hard-boundary callback, restores the complete
accepted state on rejection, and refreshes projected forces after reflection.
The callback registration symbols alone do not establish this behavior.
Current lammpc main does not provide it, so the adapter fails closed for a
sampling budget with a region.

The extension is generic. It does not contain an OH-TST, PAFI, pair-style,
or material type. rgsaddle maps those arrays into `HyperplaneObservation`.


<a id="orgcbfe966"></a>

# Thermodynamic integration coupling

When `HyperplaneSampleBudget::lambda` is set, `LammpsHyperplaneSampler`
requires `LammpsSamplerConfig::thermodynamic_reference`: a Taylor harmonic
model stored as reference geometry, energy, gradient, and dense Hessian in
the sampler coordinates of the plane. Those are the coordinates selected by
`HyperplaneSampleBudget::coordinates` and passed to `md_initialize`.

Until lammpc exposes a mixing hook, lambda sampling fails with a surface
error. The additive ABI is:

```c
typedef struct {
  double lambda;
  double reference_energy;
} lammpc_md_thermodynamic_coupling_t;

int lammpc_session_md_set_thermodynamic_coupling(
    LammpcSession *session,
    const lammpc_md_thermodynamic_coupling_t *coupling,
    const double *reference_point,
    const double *reference_gradient,
    const double *reference_hessian,
    size_t dimension);
```

`reference_point`, `reference_gradient`, and `reference_hessian` are in the
sampler frame. The Hessian is passed already symmetrized. Each MD step adds
`(1-lambda) F_harmonic` in that frame, with
`F_harmonic = -(g_0 + H (x - x_0))`, and the engine converts that force to
the Cartesian LAMMPS force with the same mass metric it uses for the
constraint. The hook is not tied to a pair style or a material.

The observable also needs the unmixed full potential. A missing symbol is a
surface error when lambda is set:

```c
int lammpc_session_md_potential_energy(
    LammpcSession *session,
    const double *position,
    size_t dimension,
    double *energy);
```

`position` is the Cartesian geometry returned by `md_state`. The scalar
energy is frame invariant. rgsaddle evaluates `V_harm` itself at the
sampler-frame image of that geometry, including the mass-weighted scaling
`q = M^{1/2} x` when the budget asks for mass-weighted coordinates.


<a id="org9ddddcb"></a>

# Errors

Library lookup follows rgpot's `lammpc` profile candidates. An explicit path
has priority, followed by `LAMMPC_LIBRARY`, `RGPOT_LAMMPC_ENGINE`, and
platform library names. Missing symbols, an unavailable stub build, ABI
mismatch, invalid geometry, and LAMMPS failures become
`SaddleError::Surface` with lammpc's diagnostic.
