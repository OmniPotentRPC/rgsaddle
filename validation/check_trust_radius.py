"""Check the exact radial identity and two binary64 rounding witnesses."""

if not __debug__:
    raise SystemExit("Symbolic validation requires Python assertions; disable optimization.")

from pathlib import Path
import subprocess
import sys

import sympy as sp

sx, sy = sp.symbols("sx sy", positive=True)
radius = sp.symbols("radius", nonnegative=True)
norm_sq = sx**2 + sy**2
scale = radius/sp.sqrt(norm_sq)
assert sp.simplify((scale * sx)**2 + (scale * sy)**2 - radius**2) == 0
print("EXACT_RADIAL_NORM_SQUARED")

result = subprocess.run(
    ["sollya", "--flush", str(Path(__file__).with_name("trust_radius.sollya"))],
    capture_output=True,
    text=True,
)
print(result.stdout, end="")
if result.stderr:
    print(result.stderr, file=sys.stderr, end="")
result.check_returncode()
expected = ["CERTIFIED_RADIAL_WITNESS", "CERTIFIED_RADIAL_WITNESS"]
if result.stdout.splitlines() != expected or result.stderr:
    raise SystemExit("Sollya did not certify both binary64 radial witnesses.")
