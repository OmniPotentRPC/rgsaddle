"""Exact weighted-basis identities and conditional binary64 certificates."""
if not __debug__:
    raise SystemExit("Symbolic validation requires Python assertions; omit -O and PYTHONOPTIMIZE")

from pathlib import Path
import subprocess
import sys
import sympy as s

t = s.symbols("t", real=True)
r = (1 - t**2) / (1 + t**2)
c = 2 * t / (1 + t**2)
Q = s.Matrix([[r, -c, 0], [c, r, 0], [0, 0, 1]])
s1, s2, s3 = s.symbols("s1 s2 s3", positive=True)
S = s.diag(s1, s2, s3)
lam = s.symbols("l0:3", real=True)
z = s.Matrix(s.symbols("z0:3", real=True))
W = Q * s.diag(*lam) * Q.T
H = S * W * S
x = S.inv() * Q * z
assert s.simplify(Q.T * Q - s.eye(3)) == s.zeros(3)
assert s.simplify(S.inv() * H * S.inv() - W) == s.zeros(3)
assert s.simplify((x.T * S**2 * x)[0] - (z.T * z)[0]) == 0
for i in range(3):
    direction = S.inv() * Q[:, i]
    assert s.simplify(H * direction - lam[i] * S**2 * direction) == s.zeros(3, 1)
print("SYMPY_WEIGHTED_EIGENPAIRS_AND_STEP_NORM", flush=True)

H0 = s.diag(2, -8, 45)
M = s.diag(1, 4, 9)
S0 = s.diag(1, 2, 3)
x0 = s.Matrix([0, s.Rational(1, 4), 0])
g = 2 * H0 * x0
weighted = S0.inv() * H0 * S0.inv()
assert sorted(weighted.eigenvals()) == [-2, 2, 5]
alpha = s.Integer(12)
g0 = g[1] / 2
mu = (-2 + s.sqrt(4 + 4 * alpha * g0**2)) / 2
dx = s.Matrix([0, g0 / (mu + 2) / 2, 0])
assert dx == s.Matrix([0, -s.Rational(1, 8), 0])
residual = (2 * H0 - H0) * dx
ss = (dx.T * dx)[0]
rs = (residual.T * dx)[0]
powell = H0 + (residual * dx.T + dx * residual.T) / ss - rs * dx * dx.T / ss**2
sr1 = H0 + residual * residual.T / rs
assert powell == sr1 == s.diag(2, -16, 45)
assert sorted((S0.inv() * powell * S0.inv()).eigenvals()) == [-4, 2, 5]
print("SYMPY_TRUST_STEP_AND_PRE_UPDATE_CURVATURE", flush=True)

result = subprocess.run(["sollya", "--flush", str(Path(__file__).with_suffix(".sollya"))],
                        capture_output=True, text=True)
print(result.stdout, end="")
if result.stderr:
    print(result.stderr, file=sys.stderr, end="")
result.check_returncode()
expected = [f"CERTIFIED_CARTESIAN_ERROR {n}" for n in (3, 75, 300, 4096)]
expected.append("CERTIFIED_TRUST_FIXTURE")
assert result.stdout.splitlines() == expected and not result.stderr, "missing interval certificate"
