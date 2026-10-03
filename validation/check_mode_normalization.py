"""Run the normalization identities and require every interval certificate."""

if not __debug__:
    raise SystemExit("Symbolic validation requires Python assertions; disable optimization.")

from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
subprocess.run([sys.executable, str(root / "mode_normalization.py")], check=True)
result = subprocess.run(
    ["sollya", "--flush", str(root / "mode_normalization.sollya")],
    capture_output=True,
    text=True,
)
print(result.stdout, end="")
if result.stderr:
    print(result.stderr, file=sys.stderr, end="")
result.check_returncode()
lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
expected = [f"CERTIFIED_MODE_NORM {n}" for n in (3, 75, 512, 4096)]
if lines != expected or result.stderr:
    raise SystemExit("Sollya did not emit exactly the four required certificates")
