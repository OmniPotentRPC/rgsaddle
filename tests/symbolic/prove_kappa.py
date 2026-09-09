"""Exact identities for Xiao et al., JCP 141, 164111 (2014), Eqs. 6--7."""
import json
import sympy as s

x, y, a, b = s.symbols("x y a b", real=True)
a, b = s.symbols("a b", positive=True)
g = s.Matrix([2*a*x, -2*b*y])
h = s.diag(2*a, -2*b)
t = s.Matrix([-g[1], g[0]])
nu = (t.T*h*t)[0] / (g.dot(g))
kappa = -nu / s.sqrt(g.dot(g))
checks = {
    "contour_tangent": s.simplify(t.dot(g)),
    "harmonic_curvature": s.simplify(kappa - a*b*(a*x*x-b*y*y)/(a*a*x*x+b*b*y*y)**s.Rational(3,2)),
}
z = s.symbols("z", real=True)
p = 1/(1+s.exp(z))
gamma1, gamma2 = 2*p-1, 1-p
checks["gamma_relation"] = s.simplify(gamma1+2*gamma2-1)
checks["regular_dimer_equal_coefficients"] = s.simplify((gamma1-gamma2).subs(z,-s.log(2)))
checks["uphill_parallel_limit"] = s.limit(gamma1,z,-s.oo)-1
checks["uphill_perpendicular_limit"] = s.limit(gamma2,z,-s.oo)
checks["downhill_parallel_limit"] = s.limit(gamma1,z,s.oo)+1
checks["downhill_perpendicular_limit"] = s.limit(gamma2,z,s.oo)-1
u = s.Matrix([s.Rational(1,3),s.Rational(2,3),s.Rational(2,3)])
q = s.eye(3)-2*u*u.T
checks["householder_orthogonality"] = s.simplify(q.T*q-s.eye(3)).norm()
basis=q[:,1:]
w=s.Matrix(s.symbols("w0:2"))
h=s.Matrix([[1,2,3],[2,4,5],[3,5,-6]])
checks["reduced_rayleigh"] = s.expand(((basis*w).T*h*(basis*w))[0]-(w.T*(basis.T*h*basis)*w)[0])
checks["reduced_norm"] = s.expand((basis*w).dot(basis*w)-w.dot(w))
checks["excluded_direction"] = s.simplify(q[:,0].dot(basis*w))
assert all(value == 0 for value in checks.values()), checks
print(json.dumps({"sympy":s.__version__,"identities":{k:str(v) for k,v in checks.items()},
    "harmonic_axis_limit":"kappa(x,0)=b/(a*abs(x)); no direction-independent zero limit at the saddle",
    "source":"https://doi.org/10.1063/1.4898664"}, indent=2))
