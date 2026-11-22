"""OH-TST tutorial, Python side.

Like PAFI, OH-TST has no Python session: this script runs the compiled
example and reads its JSON result back. The surface, the dimer seed, the
plane optimiser, the sampler, MBAR, the transmission coefficient and the
rate are all in ``examples/ohtst_tutorial.rs``; see that file and
``docs/orgmode/tutorials/oh-tst.org`` for the physics. The settings here
are fixed in the Rust example, not passed in, since the point of this
script is reading the result, not configuring the run.

Run from a checkout after building the example once:

.. code-block:: console

    $ cargo build --release --features ohtst --example ohtst_tutorial
    $ python3 examples/ohtst_tutorial.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def find_binary(repo_root: Path) -> Path:
    """Locate the compiled example, building it if it is missing."""
    binary = repo_root / "target" / "release" / "examples" / "ohtst_tutorial"
    if not binary.is_file():
        subprocess.run(
            ["cargo", "build", "--release", "--features", "ohtst", "--example", "ohtst_tutorial"],
            cwd=repo_root,
            check=True,
        )
    if not binary.is_file():
        raise FileNotFoundError(f"did not find {binary} after building it")
    return binary


def run_ohtst(repo_root: Path) -> dict[str, float]:
    """Run the compiled OH-TST example and return its JSON result."""
    binary = find_binary(repo_root)
    with tempfile.TemporaryDirectory() as workdir:
        result_path = Path(workdir) / "result.json"
        subprocess.run([str(binary), str(result_path)], check=True)
        raw = json.loads(result_path.read_text())
        if not isinstance(raw, dict):
            raise ValueError("OH-TST tutorial output must be a JSON object")
        result: dict[str, float] = {str(key): float(value) for key, value in raw.items()}
        return result


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    result = run_ohtst(repo_root)

    print(f"optimal plane turned {result['turn_degrees']:.4f} degrees from the dimer's seed")
    print(f"Z_H / Z_R = {result['density']:.4f} +/- {result['density_error']:.4f}")
    print(
        "reversible work reactant plane -> optimum: "
        f"MBAR {result['mbar_work']:.5f} +/- {result['mbar_error']:.5f}, "
        f"closed form {result['closed_form']:.5f}"
    )
    print(
        f"k_OH (no transmission correction) = {result['bare_rate']:.6e} "
        f"+/- {result['bare_rate_error']:.1e}"
    )
    print(f"kappa = {result['kappa']:.5f} +/- {result['kappa_error']:.5f}")
    print(f"k = kappa * k_OH = {result['final_rate']:.6e} +/- {result['final_error']:.1e}")

    deviation = abs(result["mbar_work"] - result["closed_form"]) / result["mbar_error"]
    print(f"MBAR work is {deviation:.2f} standard errors from the closed form")


if __name__ == "__main__":
    main()
