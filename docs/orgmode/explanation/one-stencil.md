The dimer translation and the climbing image are one Householder reflection of the true force. Goswami, Gunde, and Jónsson, Front. Chem. (2026), https://doi.org/10.3389/fchem.2026.1807063, Eqs. 3, 8, and 9, and Sec. 2.3.1 and Sec. 4. CI-NEB and dimer translation share that reflection, using the path tangent and minimum mode, respectively, as the distinguished direction. Goswami (2026), arXiv:2603.10992, https://doi.org/10.48550/arXiv.2603.10992, Eqs. 2.14-2.16.

A preset chooses the sides, the tangent, the coupling, the climb, and the mesh. `Stencil::point` runs a minimiser, the dimer, Lanczos, ART, and eigen following. Dimer and Lanczos construct `MinModeSession`, so the recorded position and energy hashes stay the same. `Stencil::band` names the band presets but returns an error because band presets are not implemented. Use the dedicated band sessions instead.

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Preset</th>
<th scope="col" class="org-left">Tangent</th>
<th scope="col" class="org-left">Climb</th>
<th scope="col" class="org-left">Mode</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left">minimiser</td>
<td class="org-left">none</td>
<td class="org-left">none</td>
<td class="org-left">none</td>
</tr>

<tr>
<td class="org-left">dimer</td>
<td class="org-left">axis</td>
<td class="org-left">reflect if the curvature is negative, otherwise ascend</td>
<td class="org-left">rotation</td>
</tr>

<tr>
<td class="org-left">Lanczos</td>
<td class="org-left">Ritz</td>
<td class="org-left">as the dimer</td>
<td class="org-left">Lanczos</td>
</tr>

<tr>
<td class="org-left">ART</td>
<td class="org-left">seed, then Ritz</td>
<td class="org-left">ascend until the curvature drops, then reflect</td>
<td class="org-left">Lanczos</td>
</tr>

<tr>
<td class="org-left">eigen following</td>
<td class="org-left">lowest eigenvector</td>
<td class="org-left">as the dimer</td>
<td class="org-left">dense central-difference <code>exact_eigh</code></td>
</tr>
</tbody>
</table>

rgmin started as Rohit Goswami's xtsci-optimize (2023-2024), which composed minimisers from small strategy objects. The stencil core extends that to saddle and band methods. eOn v2.9.0.1 is https://doi.org/10.5281/zenodo.18529496. Goswami (2025), arXiv:2510.21368, https://doi.org/10.48550/arXiv.2510.21368.

`Chain::evaluate` owns one block per image. The block size is fixed at one image and does not depend on the thread count. Each block is written back by index after the surface call. Remesh, insert, and window do not call the surface. Image ids survive an insert.

The C surface is `rgsaddle_stencil_*` with an opaque config, typed key setters, and a thread-local `rgsaddle_last_error`. `include/rgsaddle/stencil.hpp` is the RAII wrapper. Python `rgsaddle.Stencil.point` and `rgsaddle.ase.Point` call that wrapper. `Point.run` uses an ASE `Optimizer` subclass. The extension is nanobind with `Py_MOD_GIL_NOT_USED` on a free-threaded build.

Method papers, after the citations above: Henkelman and Jónsson (1999), https://doi.org/10.1063/1.480097; Henkelman, Uberuaga, and Jónsson (2000), https://doi.org/10.1063/1.1329672; Henkelman and Jónsson (2000), https://doi.org/10.1063/1.1323224; Jónsson, Mills, and Jacobsen (1998); Ásgeirsson et al. (2021), https://doi.org/10.1021/acs.jctc.1c00462; Kolsbjerg, Groves, and Hammer (2016), https://doi.org/10.1063/1.4961868; Malek and Mousseau (2000), https://doi.org/10.1103/PhysRevE.62.7723; Smidstrup et al. (2014), https://doi.org/10.1063/1.4878664; Schmerwitz, Ásgeirsson, and Jónsson (2024), https://doi.org/10.1021/acs.jctc.3c01111; Curtis, Powell, and Reid (1974), https://doi.org/10.1093/imamat/13.1.117.

A benchmark that changes iterates cites Goswami, AIP Adv. (2025), https://doi.org/10.1063/5.0283639, first.
