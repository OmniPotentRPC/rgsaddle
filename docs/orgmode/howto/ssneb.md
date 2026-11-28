`SolidNeb` is a climbing-image nudged elastic band on the coordinate already used by `SolidDimer`. Sheppard, Xiao, Chemelewski, Johnson, and Henkelman, J. Chem. Phys. 136, 074103 (2012), https://doi.org/10.1063/1.3684549, put the strain next to the atoms and spring the band in that joint vector. The chart is the solid-state dimer of Xiao, Sheppard, Rogal, and Henkelman, J. Chem. Phys. 140, 174104 (2014), https://doi.org/10.1063/1.4873437:

$$ h = h_0 (I + \varepsilon), \quad r = r_0 (I + \varepsilon), \quad J = V^{1/3} N^{1/6}. $$

`J` is fixed from `h_0`. The free strains are the lower triangle, stored as `J ε`. The enthalpy is `E + V_0 σ_ext : ε`, the same sign as `JointChart`: a positive diagonal entry of `σ_ext` pushes the cell inward. The tangent, the spring and the climbing image are the ordinary band's, applied to the joint vector. The improved tangent reads the enthalpy. A surface supplies the Cauchy stress; `None` is finite-differenced.

`SolidNeb::frozen` is a `BandSession`. The same configuration and the same surface produce the same positions, including the climbing image. `SolidNeb::joint` rejects `BandConfig` settings with `BandConfig::solid`, `BandConfig::min_mode` (atomic min-mode refinement), `BandConfig::zoom` (Zoom-NEB), `BandConfig::exp_precon` (exponential metric), `BandConfig::autoneb` (AutoNEB), `BandConfig::cell` (periodic minimum-image cell), or any `true` entry in `BandConfig::fixed_atoms`; `SpringKind::OnsagerMachlup` or `SpringKind::OnsagerMachlupLive` springs, or `SpringKind::Weighted` whose `ks` length is not `n_images - 1`; a non-positive or non-finite `BandConfig::time_step`; and, when `BandConfig::quickmin` is `false`, any `BandConfig::method` other than `Method::Fire`, `Method::Lbfgs`, `Method::Bb`, `Method::QuickMin`, or `Method::Ode12r`. It also rejects `positions` with fewer than 3 images, zero columns, or a column count not divisible by 3, or a `cells` slice whose length differs from `positions.nrows()`; in `JointChart::new`, a `reference` cell that is not lower triangular (upper-triangle entries above 1e-8 in magnitude) or has non-positive or non-finite volume `|det(reference)|`, a non-positive or non-finite `weight`, or non-finite entries in the folded lower triangle of `external`; and, in `joint_coordinate` for each image, a singular `reference` or `reference^-1 * cell`, an image `cell` whose strain `ε = reference^-1 * cell - I` has upper-triangle entries above 1e-8 in magnitude or a strain component above 1e-8 in magnitude on an entry held fixed (`false`) by `mask`, a deformed cell with volume `|det(h)| <= 1e-18`, non-finite coordinates, or positions or cell entries that do not round-trip through `chart.geometry` within 1e-6. `joint_coordinate` maps a Cartesian geometry onto the chart. `SolidDimer::set_generalized` moves a search that was built on `h_0` onto a point of that same chart, so a dimer started at a climbing image keeps the enthalpy the band climbed. After a step, `images` returns one `SolidImage` per image with its positions, cell, joint coordinate, energy, enthalpy and stress. It is non-exhaustive: read its fields.

No C field was added.


# Runnable example: solid-state NEB on the joint chart

```rust
use ndarray::{Array1, ArrayView1, array, s};
use rgsaddle::{
    BandConfig, BandStatus, CellMask, JointChart, PointSurface, SaddleError, SolidNeb,
    joint_coordinate, linear_path,
};

const CUBE: [f64; 9] = [4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.0];

struct WellInCell;

impl PointSurface for WellInCell {
    fn eval(&self, r: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        self.eval_in_cell(r, &CUBE)
    }

    fn eval_in_cell(
        &self,
        r: ArrayView1<f64>,
        cell: &[f64; 9],
    ) -> Result<(f64, Array1<f64>), SaddleError> {
        let (x, y, z) = (r[0], r[1], r[2]);
        let elastic: f64 = cell.iter().zip(CUBE).map(|(h, h0)| (h - h0).powi(2)).sum();
        let energy = (x * x - 1.0).powi(2) + 2.0 * y * y + 2.0 * z * z + elastic;
        Ok((energy, array![4.0 * x * (x * x - 1.0), 4.0 * y, 4.0 * z]))
    }
}

fn main() -> Result<(), SaddleError> {
    let cube = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]];
    let chart = JointChart::new(cube, 1, CellMask::all(), [[0.0; 3]; 3], 1.0)?;
    let _q0 = joint_coordinate(&chart, CellMask::all(), array![-1.0, 0.0, 0.0].view(), cube)?;
    let mut positions = linear_path(array![-1.0, 0.0, 0.0].view(), array![1.0, 0.0, 0.0].view(), 5, None)?;
    positions.slice_mut(s![1..4, 1]).fill(0.2);
    let config = BandConfig::default()
        .with_force_tol(1e-3)
        .with_max_move(0.1)
        .with_min_mode(None);
    let mut neb = SolidNeb::joint(config, cube, positions, &[cube; 5], CellMask::all(), [[0.0; 3]; 3], 1.0)?;
    let mut report = neb.step(&WellInCell)?;
    while report.status == BandStatus::Running && report.iteration < 400 {
        report = neb.step(&WellInCell)?;
    }
    assert_eq!(report.status, BandStatus::Converged);
    Ok(())
}
```


## The shear-coupled double well

`tests/ssneb.rs` uses `(u^2 - 1)^2 + 6 (γ - 0.2 u)^2` plus penalties that vanish on the path. The minima have energy 0. The saddle `u = 0`, `γ = 0` has energy 1, so the barrier is 1. The variable-cell climbing image converges to that barrier. The frozen cell on the plain double well matches the ordinary CI-NEB at every step.


## Copper under shear

The check against the index-checked dimer is `tests/ssneb_validate_cu.rs`, release, `--ignored`. The system is the same 109-atom Mishin EAM1 cell. The <100> reactant and the shear-aligned <110> are relaxed at the imposed shear. A variable-cell product is then relaxed on the enthalpy `E + V0 σ_ext : ε` with `σ_ext` equal to minus the stress of the <100> minimum, and at 5% the opposite <110> is treated the same way. Eleven images start linear in the joint coordinate. The spring constant is 1. Climbing starts on the first evaluation. The largest force component is driven to 5e-4, or to `0.005 |λ|` when the dimer curvature is softer than that and still at least 5e-5. The negative-mode count projects out the rigid translations and uses a central step of 1e-5. A negative mode is an eigenvalue below -1e-4. The distance removes the mean minimum-image atomic displacement and takes the larger of that maximum and the maximum cell-matrix element. The dimer column is the index-checked search, not the earlier table. Raw rows are `benchmarks/ssneb_validate_cu.csv`.

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-right" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-left" />
</colgroup>
<tbody>
<tr>
<td class="org-right">shear</td>
<td class="org-left">cell</td>
<td class="org-left">product</td>
<td class="org-right">E dimer</td>
<td class="org-right">E CI</td>
<td class="org-right">H dimer</td>
<td class="org-right">H CI</td>
<td class="org-right">geometry (Å)</td>
<td class="org-right">n<sub>neg</sub></td>
<td class="org-left">result</td>
</tr>

<tr>
<td class="org-right">0</td>
<td class="org-left">fixed</td>
<td class="org-left">aligned</td>
<td class="org-right">0.47392355</td>
<td class="org-right">0.47420776</td>
<td class="org-right">0.47392355</td>
<td class="org-right">0.47420776</td>
<td class="org-right">0.087922</td>
<td class="org-right">2</td>
<td class="org-left">fail</td>
</tr>

<tr>
<td class="org-right">0</td>
<td class="org-left">variable</td>
<td class="org-left">aligned</td>
<td class="org-right">0.33255762</td>
<td class="org-right">0.33256222</td>
<td class="org-right">0.34700117</td>
<td class="org-right">0.34701664</td>
<td class="org-right">0.004087</td>
<td class="org-right">1</td>
<td class="org-left">pass</td>
</tr>

<tr>
<td class="org-right">0.02</td>
<td class="org-left">fixed</td>
<td class="org-left">aligned</td>
<td class="org-right">0.24355187</td>
<td class="org-right">0.24356221</td>
<td class="org-right">0.24355187</td>
<td class="org-right">0.24356221</td>
<td class="org-right">0.003215</td>
<td class="org-right">1</td>
<td class="org-left">pass</td>
</tr>

<tr>
<td class="org-right">0.02</td>
<td class="org-left">variable</td>
<td class="org-left">aligned</td>
<td class="org-right">0.31321352</td>
<td class="org-right">0.31313601</td>
<td class="org-right">0.16276484</td>
<td class="org-right">0.16277097</td>
<td class="org-right">0.004157</td>
<td class="org-right">1</td>
<td class="org-left">pass</td>
</tr>

<tr>
<td class="org-right">0.03</td>
<td class="org-left">fixed</td>
<td class="org-left">aligned</td>
<td class="org-right">0.14828664</td>
<td class="org-right">0.14829576</td>
<td class="org-right">0.14828664</td>
<td class="org-right">0.14829576</td>
<td class="org-right">0.001921</td>
<td class="org-right">1</td>
<td class="org-left">fail</td>
</tr>

<tr>
<td class="org-right">0.03</td>
<td class="org-left">variable</td>
<td class="org-left">aligned</td>
<td class="org-right">0.28604258</td>
<td class="org-right">0.28597114</td>
<td class="org-right">0.08811247</td>
<td class="org-right">0.08811325</td>
<td class="org-right">0.000435</td>
<td class="org-right">1</td>
<td class="org-left">pass</td>
</tr>

<tr>
<td class="org-right">0.04</td>
<td class="org-left">fixed</td>
<td class="org-left">aligned</td>
<td class="org-right">0.06783472</td>
<td class="org-right">0.06784900</td>
<td class="org-right">0.06783472</td>
<td class="org-right">0.06784900</td>
<td class="org-right">0.006941</td>
<td class="org-right">1</td>
<td class="org-left">pass</td>
</tr>

<tr>
<td class="org-right">0.04</td>
<td class="org-left">variable</td>
<td class="org-left">aligned</td>
<td class="org-right">0.24337840</td>
<td class="org-right">0.24328451</td>
<td class="org-right">0.02728968</td>
<td class="org-right">0.02729164</td>
<td class="org-right">0.007535</td>
<td class="org-right">1</td>
<td class="org-left">pass</td>
</tr>

<tr>
<td class="org-right">0.05</td>
<td class="org-left">fixed</td>
<td class="org-left">aligned</td>
<td class="org-right">0.00777291</td>
<td class="org-right">0.00776732</td>
<td class="org-right">0.00777291</td>
<td class="org-right">0.00776732</td>
<td class="org-right">0.009611</td>
<td class="org-right">1</td>
<td class="org-left">pass</td>
</tr>

<tr>
<td class="org-right">0.05</td>
<td class="org-left">variable</td>
<td class="org-left">aligned</td>
<td class="org-right">0.28153212</td>
<td class="org-right">69.40842902</td>
<td class="org-right">0.91523258</td>
<td class="org-right">129.51686918</td>
<td class="org-right">20.417387</td>
<td class="org-right">7</td>
<td class="org-left">band runs away</td>
</tr>

<tr>
<td class="org-right">0.05</td>
<td class="org-left">variable</td>
<td class="org-left">opposite</td>
<td class="org-right">0.28153212</td>
<td class="org-right">293.10519184</td>
<td class="org-right">0.91523258</td>
<td class="org-right">359.42770067</td>
<td class="org-right">21.622133</td>
<td class="org-right">8</td>
<td class="org-left">band runs away</td>
</tr>
</tbody>
</table>

Seven of the ten published comparisons pass. A dimer started at each passing climbing image does not move.

At zero shear the fixed-cell band stops at the 800-step limit with a largest force of 0.009468. Its climbing image is index 2: the projected spectrum, central step 1e-5, starts -0.98787712, -0.01541131. The index-checked dimer saddle at 0.473923553294469 eV has one negative eigenvalue, -0.98724793, and is 0.087922 Å from the climbing image. The energies differ by 2.842e-4 eV. The row fails the force, geometry and index tests. At 3% fixed the band also stops at 800 steps, at 0.001267; the energy matches to 9.1e-6 eV and the geometry to 0.001921 Å on an index-1 point, so only the force target fails.

At 4% variable the flat tolerance runs from call 855, where the force is first under 2e-3, to call 2291. The largest force component is then 1.196e-4 on a curvature of -0.02542812. The enthalpy differs from the climbing image by 1.96e-6 eV and the atoms by 0.007535 Å. At 5% fixed the same tolerance runs from call 2182 to call 3981. The largest component is 1.702e-4 on a curvature of -0.03412715. The energy differs by 5.59e-6 eV and the atoms by 0.009611 Å.

At 5% variable the reactant has no negative projected eigenvalue. The lowest is +0.018114765351867675, so it is a shallow minimum, not a saddle. The shear-aligned <110>, relaxed on the enthalpy, lies 0.00809061 eV below the reactant, and the opposite orientation relaxed on the same enthalpy stops 23.4° from <100> and 21.6° from <110>, 0.00062033 eV below it. Neither band stays on its path. The L-BFGS band climbs from its first step, holds a largest force between 1e-3 and 6e-3 for about 150 steps, and then runs away: at 800 steps the climbing images are 69.4 eV and 293.1 eV above the minimum. On rgmin 0.5.2 the same bands stayed bounded and did not rise; on rgmin 0.6.0 these rows carry no barrier. The index check accepts an index-1 saddle of the opposite <110>, enthalpy 0.9152325828545145 eV, lowest eigenvalue -1.26503090. It is not on either band and is not the barrier of either path.

The 4% variable band passes through 0.02159001 eV with a true force of 0.01048197, and its climbing image is at 0.02729164 eV. That 0.021 eV point is on the slope.
