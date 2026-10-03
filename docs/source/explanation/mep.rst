Mass-weighted reaction paths
============================

An intrinsic reaction coordinate follows the gradient in mass-weighted coordinates. With ``x_mw = sqrt(m) x``, its tangent is the normalized negative mass-weighted gradient.

The GS2 inner step uses a moving constraint whose radius is measured in that metric. The displacement and the stored path offset appear together in the constraint, so the sphere is not the unit sphere centred at the coordinate origin. Cartesian rigid-motion removal and mass weighting are separate parts of the calculation.

The launch direction is one extremal Hessian mode. A surface that supports Hessian actions can supply it without materializing a dense Hessian. An existing dense Hessian can use the corresponding eigensolver. The host chooses the method and verifies the physical saddle before tracing its branches.
