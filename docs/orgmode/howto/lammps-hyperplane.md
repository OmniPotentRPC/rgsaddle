
# Table of Contents

-   [Build the optional stack](#orgbe0b808)
-   [Describe the system](#org8e012fd)
-   [Run PAFI or OH-TST](#org7a5d4ac)



<a id="orgbe0b808"></a>

# Build the optional stack

Enable the `lammps` feature. It also enables `ohtst`. Default builds do not
compile rgpot or load LAMMPS.

The supported native stack is:

-   `OmniPotentRPC/lammpc` at `dc443997` plus the constrained MD API patch
-   `OmniPotentRPC/rgpot` at `97ebba0b`
-   LAMMPS `stable_22Jul2025_update2`, built as a serial shared library

Build lammpc with its `with_lammps` Meson option. Set `LAMMPC_LIBRARY` to the
resulting `liblammpc.so`, or set an explicit path with
`LammpsSamplerConfig::with_library_path`.

```console
$ cargo test --features lammps --test lammps_sampler
```

The test uses `RGSADDLE_CU_EAM` for the optional `eam/alloy` copper smoke
case. The value is the path to a LAMMPS alloy table containing Cu.


<a id="org8e012fd"></a>

# Describe the system

`LammpsSystem` takes flat rgpot `PotentialConfig` bytes with the `lammps`
arm, Cartesian positions in angstrom, atomic numbers, masses in amu, and an
optional row-major cell. The masses must match the `LammpsParams.masses`
entries selected by the atomic numbers.

```rust
let system = LammpsSystem::default()
    .with_config_bytes(config_bytes)
    .with_positions(positions)
    .with_atomic_numbers(atomic_numbers)
    .with_masses(masses)
    .with_box_matrix(cell);

let mut sampler = LammpsHyperplaneSampler::new(
    system,
    LammpsSamplerConfig::default()
        .with_library_path("liblammpc.so")
        .with_time_step_ps(0.001)
        .with_friction_per_ps(1.0)
        .with_thermostat(LammpsThermostatKind::BaoabLangevin)
        .with_excluded_modes(3),
);
```

The interaction model remains inside the rgpot configuration. The sampler
does not name a pair style, element, or crystal.


<a id="org7a5d4ac"></a>

# Run PAFI or OH-TST

Use `average_hyperplane_with_sampler` for PAFI and
`sample_plane_with_sampler` for one OH-TST plane. Both use the same
`HyperplaneSampler` contract.

```rust
let plane_sample = sample_plane_with_sampler(
    &mut sampler,
    &plane,
    &LangevinConfig::default()
        .with_thermal(0.02585)
        .with_burn_in(500)
        .with_stride(5)
        .with_records(2_000),
)?;
```

The configured burn-in is only a minimum floor. The estimator applies
`equilibrate` to the returned constraint-force series, selects the retained
tail, and rejects split-half drift. The backend adds no hidden burn-in.

lammpc advances one projected BAOAB step at a time. Each step projects the
force and velocity, restores the position to the plane after each drift,
and thermostats tangent velocity with Langevin noise. LAMMPS hard-region
sampling is pending a lammpc capability that invokes its registered boundary
callback on every constrained MD step. Current lammpc main has the callback
registration entry points but not that constrained-step behavior, so the
adapter must reject a region until the capability is available. The force
evaluation is the live in-process LAMMPS session. No penalty restraint is
used.

Mass-weighted sampling returns $q=M^{1/2}x$, its tangent force, and its
removed velocity. Cartesian sampling leaves the vectors Cartesian and uses
the mass metric for projection. PAFI and OH-TST statistics are computed by
rgsaddle after lammpc returns the records.

Each record exposes two thermostat metadata observables:
`lammps.thermostat` and `lammps.effective_degrees_of_freedom`. The thermostat
field is a stable enum code (currently BAOAB Langevin). The effective degrees
of freedom are $3N - 1 - n_{\mathrm{excluded}}$, with
`n_{\mathrm{excluded}}` supplied through
`LammpsSamplerConfig::with_excluded_modes`.
