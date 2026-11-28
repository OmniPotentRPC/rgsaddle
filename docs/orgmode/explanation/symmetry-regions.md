
# Table of Contents

1.  [Construction](#orgbe0b808)
2.  [Path multiplicity and region degeneracy](#org8e012fd)
3.  [Validation and tolerances](#org7a5d4ac)



<a id="orgbe0b808"></a>

# Construction

A host supplies three-dimensional orthogonal matrices, optional affine
translations or rotation centres, and reactant and saddle Cartesian
geometries.  An operation acts on an atom at $x$ as

$$
x' = R(x-c)+c+t.
$$

The operation may also carry a source-to-target atom permutation.  When it
does not, rgsaddle matches transformed reactant atoms to reactant atoms of
the same species.  Periodic distances and the final unwrapping use
`minimage::Cell::displacement_euclidean`.  The implementation does not
duplicate a minimum-image convention.

This is a consumer API, not a space-group finder.  The host remains
responsible for supplying the candidate crystallographic operations.
The affine form and the distinction between point and space-group
operations follow the conventions in International Tables for
Crystallography, Volume A, edited by Theo Hahn, 5th edition (2005),
https://doi.org/10.1107/97809553602060000100.


<a id="org8e012fd"></a>

# Path multiplicity and region degeneracy

Operations which map the reactant onto itself form the reactant stabilizer
$H$.  Applied to the selected saddle, they produce an orbit.  The path
multiplicity is the number of distinct images

$$
m_{\rm path} = |H \cdot s| = {|H| \over |H_s|},
$$

where $H_s$ is the joint stabilizer, the subgroup of $H$ that also maps
that saddle to itself.  This is the orbit-stabilizer theorem; see
Jean-Pierre Serre, *Linear Representations of Finite Groups*, Springer
(1977), https://doi.org/10.1007/978-1-4684-9458-7.  $m_{\rm path}$ is the number of
equivalent hops.  Callers pass it to `RateMultiplicity::with_paths`.  It is
not the degeneracy of the sampled cell.

$H_s$ is the symmetry of one hop.  Each of its operations fixes the
reactant and the saddle, and therefore maps the dividing plane of that hop
onto itself.  The region degeneracy is the order $|H_s|$.  The returned
`SymmetryUniqueRegion` is the asymmetric unit of that plane, the Dirichlet
cell of one generic in-plane point under $H_s$, and `Region::degeneracy`
reports the same order.  Faces lie in the plane.  Sampling the cell and
multiplying by $|H_s|$ restores the full plane of that one hop.
Multiplying again by $m_{\rm path}$ counts the other hops.

Jóhannesson and Jónsson, *Optimization of hyperplanar transition states*,
*The Journal of Chemical Physics* 115, 9644-9656 (2001),
https://doi.org/10.1063/1.1415499, build this kind of restricted sample in Sec. II D
("Incorporation of symmetry").  For an $n$-fold axis, half-lines start at
the reactant minimum and run toward each symmetry-equivalent product.  The
sample is kept in the sector of one product: a step that comes closer to
another half-line is rejected and the velocity is reflected in the mirror
that exchanges the two product directions.  Their Eqs. 13-17 remove the
part of that mirror normal parallel to the hyperplane normal, so the
reflection stays in the plane.  Sec. III D ("Al adatom diffusion on
Al(100)") uses the fourfold form of this construction for every thermal
sample.  Without the constraint, the optimized hyperplane was an average of
two symmetrically equivalent concerted-displacement channels.

An identity-only group has path multiplicity one, region degeneracy one,
and no faces.  The full octahedral stabilizer of an fcc vacancy sends one
nearest-neighbour hop to twelve images.  Those twelve are the path
multiplicity.  The region degeneracy is $|H_s|$ for that hop, the
operations that fix both the vacancy reactant and the chosen saddle.


<a id="org7a5d4ac"></a>

# Validation and tolerances

The constructor rejects non-finite data, non-orthogonal matrices,
non-bijective permutations, and permutations that map unlike species.
Operations that are valid group operations but do not stabilize the
particular reactant are omitted from $H$.  Matching, orbit, and
orthogonality tolerances are explicit fields of
`CrystalSymmetryConfig`.  Callers should choose them for their coordinate
units and structure accuracy instead of relying on a hidden physical
tolerance.

For periodic cells the nearest-image displacement is the standard
simulation-cell convention described by Daan Frenkel and Berend Smit,
*Understanding Molecular Simulation*, 2nd edition, Academic Press (2002),
https://doi.org/10.1016/B978-0-12-267351-1.X5000-7.
