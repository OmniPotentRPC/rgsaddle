Use `PathInitializer` when a driver needs the same choice as NEB or Gaussian
process NEB.  It accepts two endpoint geometries and returns an image-major
Cartesian path.


# Choose the method

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Initializer</th>
<th scope="col" class="org-left">Choose it when</th>
<th scope="col" class="org-left">Important limit</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left"><code>Linear</code></td>
<td class="org-left">The endpoint mapping is already good and no atom collision or large group rotation lies between the endpoints.</td>
<td class="org-left">It does not preserve molecular geometry.</td>
</tr>

<tr>
<td class="org-left"><code>Idpp</code></td>
<td class="org-left">Pair-distance interpolation is enough to remove collisions.</td>
<td class="org-left">Each image is relaxed independently.</td>
</tr>

<tr>
<td class="org-left"><code>Trics</code></td>
<td class="org-left">Bonds, bends, torsions, or fragment rotations should change in molecular internal coordinates, especially when one coordinate chart becomes singular.</td>
<td class="org-left">Coordinates must be unwrapped and nonperiodic.</td>
</tr>
</tbody>
</table>

TRICS refines a seed path.  `TricsSeed::Linear` is the default.
`TricsSeed::Idpp` first creates an IDPP path and then applies the TRICS
diagnostic, segmented interpolation, and splicing.

```rust
use rgsaddle::{
    IdppConfig, PathInitializer, TricsSeed, TricsStartConfig,
    initialize_path,
};

let start = TricsStartConfig::default()
    .with_seed(TricsSeed::Idpp(IdppConfig::default()));
let path = initialize_path(
    reactant.view(),
    product.view(),
    9,
    None,
    &PathInitializer::Trics(start),
)?;
```

Pass `path` to `BandSession::new` or to a Gaussian process NEB host.  The
initializer has no potential energy callback and does not evaluate a surface.
The direct `trics_path` function accepts an existing path from any source.


# AutoNEB

Select TRICS for every inserted AutoNEB image with
`AutoNebTuning::default().with_trics(TricsStartConfig::default())`, leaving
`interpolant` at its default.  AutoNEB samples a
seven-image TRICS segment and takes its midpoint.  If iterative inversion of
that local segment fails or produces a degenerate midpoint, insertion returns a
solver error.  Periodic AutoNEB rejects TRICS because rotations and the
diagnostic require unwrapped free-space coordinates.


# C

ABI minor 66 adds `rgsaddle_trics_path`.  The caller supplies an initial path
from linear interpolation, `rgsaddle_idpp_path`, or another source.  Input and
output each hold `n_images * 3 * n_atoms` doubles in image-major order.

```c
rgsaddle_status_t status = rgsaddle_trics_path(
    n_images, n_atoms, initial,
    50, 1.0e-8, 0.1,
    1, 0.0018, 1000,
    1.0, output);
```

The tolerances and the move cap are in angstrom.  Pass the length of one
angstrom in the caller's coordinates as the argument before `output`, here
1.0 for a path in angstrom and 0.1 for nanometres.  In Rust the same value is
`TricsConfig::length_per_angstrom`.

The C function uses geometry-only bond inference.  Rust callers that know the
topology can set `TricConfig::bonds` explicitly.  Python does not expose an
IDPP path initializer, so there is no separate Python TRICS wrapper.
