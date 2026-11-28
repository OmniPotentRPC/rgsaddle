`block_cyclic` is the cyclic block-tridiagonal factor.
It is exported with the default `faer` feature.
Full API documentation is on [docs.rs/rgsaddle](https://docs.rs/rgsaddle), under `rgsaddle::block_cyclic`.
This page is the inventory of each public item.
records the rotation fallback, the singular criterion, the symmetric part, and the residual norm.


# Module

The matrix is a ring of at least three beads.
Each diagonal block is an `n` by `n` matrix.
Every neighbor coupling, including the corner that closes the ring, is `-spring` times the identity.
The open chain is eliminated by block Thomas.
The closing bead is the Schur complement of that chain.
The log absolute determinant and the negative count add across those complements.


# Types

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Type</th>
<th scope="col" class="org-left">Role</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left"><code>BlockInertia</code></td>
<td class="org-left"><code>log_abs_det</code> is the sum of \(\ln\lvert\lambda\rvert\) over eigenvalues outside the relative zero tolerance. <code>n_negative</code> counts eigenvalues below the negative side of that tolerance. A successful call has already refused every eigenvalue inside the tolerance, so both fields cover the spectrum that was kept</td>
</tr>

<tr>
<td class="org-left"><code>CyclicFactor</code></td>
<td class="org-left">Factor returned by <code>factor_cyclic</code>. It holds the open-chain inverses and the closing Schur inverse, or the dense inverse of the assembled matrix when no rotation factors. <code>CyclicFactor::solve</code> returns the solution of \(M x = b\). <code>CyclicFactor::solve_report</code> returns that solution together with a <code>SolveReport</code></td>
</tr>

<tr>
<td class="org-left"><code>SolveReport</code></td>
<td class="org-left"><code>solution</code> is in the caller's bead order. <code>residual</code> is the Euclidean norm of <code>M x - b</code> after the optional correction. <code>corrections</code> is 1 when that correction ran and 0 otherwise</td>
</tr>
</tbody>
</table>


# Functions

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Function</th>
<th scope="col" class="org-left">Role</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left"><code>logdet_cyclic_blocks</code></td>
<td class="org-left"><code>logdet_cyclic_blocks(diag, n, spring)</code> returns a <code>BlockInertia</code>. A pivot inside the relative zero tolerance, or a non-finite pivot, is an error. Substituting a finite value for that pivot would change every later Schur complement. <code>SaddleError::Invalid</code> when there are fewer than three blocks, <code>n</code> is zero, or a block is asymmetric beyond the tolerance of its largest absolute entry. <code>SaddleError::Shape</code> when a block is not <code>n</code> by <code>n</code>. <code>SaddleError::NonFinite</code> when an entry, the spring, or a pivot is non-finite. <code>SaddleError::Singular</code> only when the assembled matrix has an eigenvalue inside that tolerance. A zero complement of one rotation is not singular when another rotation, or the assembled spectrum, has full rank. The singular index is the preferred rotation's complement</td>
</tr>

<tr>
<td class="org-left"><code>logdet_cyclic_strict</code></td>
<td class="org-left"><code>logdet_cyclic_strict(diag, n, spring)</code> is the same computation as <code>logdet_cyclic_blocks</code>. It refuses a pivot inside the relative zero tolerance and does not substitute a finite value. The errors are the same</td>
</tr>

<tr>
<td class="org-left"><code>factor_cyclic</code></td>
<td class="org-left"><code>factor_cyclic(diag, n, spring, tau)</code> adds <code>tau</code> to every diagonal entry after the symmetric part is taken, then returns a <code>CyclicFactor</code>. <code>CyclicFactor::solve</code> refines one step at <code>SOLVE_REFINE_TOLERANCE</code>. <code>SaddleError::Invalid</code> below three blocks, for a zero block size, or for an asymmetric block. <code>SaddleError::Singular</code> when the assembled spectrum has an eigenvalue inside the relative zero tolerance. <code>SaddleError::NonFinite</code> when <code>spring</code> or <code>tau</code> is non-finite, and the shape and finiteness errors of <code>logdet_cyclic_blocks</code></td>
</tr>
</tbody>
</table>


# Constant

`SOLVE_REFINE_TOLERANCE` is `1e-11`.
`CyclicFactor::solve_report` applies one correction when the residual exceeds this value times the Euclidean norm of the right-hand side.
A zero right-hand side is corrected when the residual is positive.
The reported norm is measured again after that single correction.
