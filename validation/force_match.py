"""Derive the radial force and Cartesian Hessian coefficients of pair kernels."""

import sympy as sp

if not __debug__:
    raise RuntimeError("proof assertions require normal Python execution")

r, rho, r0 = sp.symbols("r rho r0", positive=True)
a, b, c6, c12, k = sp.symbols("a b c6 c12 k", real=True)
exp = sp.exp(-rho * r)
kernels = {
    "lj": (c12 / r**12 - c6 / r**6,
           -12*c12/r**14 + 6*c6/r**8,
           168*c12/r**16 - 48*c6/r**10),
    "buckingham": (a*exp - c6/r**6,
                   6*c6/r**8 - a*rho*exp/r,
                   -48*c6/r**10 + a*rho*exp/r**3 + a*rho**2*exp/r**2),
    "morse": (a*exp**2 - b*exp,
              rho*(b*exp - 2*a*exp**2)/r,
              rho*((2*a*exp**2 - b*exp)/r + rho*(4*a*exp**2 - b*exp))/r**2),
    "bond": (k*(r-r0)**2, 2*k*(r-r0)/r, 2*k*r0/r**3),
}
for name, (energy, diagonal, outer) in kernels.items():
    force_scale = sp.diff(energy, r) / r
    hessian_outer = sp.diff(force_scale, r) / r
    assert sp.simplify(force_scale - diagonal) == 0, name
    assert sp.simplify(hessian_outer - outer) == 0, name
    print(f"{name}: force and Hessian coefficients verified")
