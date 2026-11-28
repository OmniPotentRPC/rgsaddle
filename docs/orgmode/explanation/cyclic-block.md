`block_cyclic` factors a ring of at least three beads.
Each diagonal block is an `n` by `n` matrix supplied by the caller.
Every neighbor coupling, including the corner that closes the ring, is `-spring` times the identity.
The open chain is eliminated by block Thomas.
The last bead is the Schur complement of that chain.
The log absolute determinant and the negative count add across those complements (Haynsworth).
Each accepted complement is inverted once, as $V \mathrm{diag}(\lambda^{-1}) V^{T}$, by one matrix product.
Later block products use that same product.
The public items are listed in .


# Rotation fallback

Pivot selection reads the eigenvalues of the original blocks and does not form those eigenvalues again while eliminating one rotation.
The quality of a block is the ratio of its smallest absolute eigenvalue to its largest.
A block whose largest absolute eigenvalue is zero has quality zero.
The best-conditioned block is eliminated first.
The bead order is rotated so that block opens the chain.
That rotation is the preferred one.

A complement with an eigenvalue inside the relative zero tolerance fails that rotation.
The failure does not, by itself, make the ring singular.
The next rotation, starting one bead later, is tried, and so on around the ring.
The first rotation whose complements all lie outside the tolerance is the factor.
It stores the open-chain inverses and the inverse of the closing Schur complement.

The spring-squared term in each complement is applied as two scalings by `spring`.
Forming `spring * spring` overflows for a large finite spring, and scaling twice keeps a well-conditioned complement finite.

When every rotation meets a zero complement, the ring is assembled into one dense matrix of `beads * n` rows and factored from its spectrum.
If that spectrum has full rank, the assembled inverse replaces the block Thomas factors.
The log absolute determinant and the negative count then come from the assembled eigenvalues.


# Singular criterion

`SaddleError::Singular` is returned only when an assembled eigenvalue lies inside the relative zero tolerance.
The tolerance is `relative_zero_tolerance(scale, n)`, with `scale` the largest absolute eigenvalue and `n` the number of rows.
An eigenvalue is inside the tolerance when its absolute value is less than or equal to that product.
A zero complement of one rotation is not this error when another rotation, or the assembled spectrum, has full rank.

The index on the error is the preferred rotation's complement.
It is the index, in that rotation's bead order, of the first complement whose eigenvalue fell inside the tolerance.
It is not an index into the assembled spectrum.
A pivot inside the tolerance is not replaced by a finite value, because that substitution would change every later Schur complement.

`logdet_cyclic_blocks` and `logdet_cyclic_strict` share this test.
Both refuse the pivot.
`logdet_cyclic_strict` is the same computation as `logdet_cyclic_blocks`.


# Symmetric part

Before a block is factored, its off-diagonal pairs are compared with the relative zero tolerance of its largest absolute entry.
The scale in that tolerance is the largest absolute entry, and the length is the block size `n`.
A pair whose absolute difference is greater than the tolerance is `SaddleError::Invalid`.
The factor would otherwise use the symmetric part while a residual formed from the original entries would not.
A smaller skew, including a difference equal to the tolerance, is replaced by the symmetric part: both entries become the mean of the two.
That replacement is made before the factor and before the residual.
`factor_cyclic` adds `tau` to every diagonal entry after the replacement.
A non-finite entry, spring, shift, or pivot is `SaddleError::NonFinite`.


# Residual norm

`CyclicFactor::solve_report` solves $M x = b$ and returns a `SolveReport`.
The residual vector is $M x - b$.
`M` is built from the symmetrized blocks stored on the factor, including the `tau` shift when `factor_cyclic` built it, and from the spring couplings.
The reported `residual` is the Euclidean norm of that vector.

The norm scales its accumulator the way LAPACK `dnrm2` does.
Zero entries are skipped.
The largest magnitude seen so far is factored out of the sum of squares, and the square root is multiplied by that magnitude at the end.
Squaring an entry near `1e200` would overflow.
A non-finite solution or a non-finite residual is `SaddleError::NonFinite`.

One correction is applied when the residual exceeds `SOLVE_REFINE_TOLERANCE` times the norm of `b`.
`SOLVE_REFINE_TOLERANCE` is `1e-11`.
A zero right-hand side is corrected when the residual is positive.
The correction solves `M` against the residual vector and subtracts that step from the solution.
The residual is then measured again.
There is no second correction.
`corrections` is 1 when the step ran and 0 otherwise.
A right-hand side whose length is not `beads * n` is `SaddleError::Shape`.
