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
from typing import TypedDict


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


class OhTstTutorialResult(TypedDict):
    """JSON written by ``examples/ohtst_tutorial.rs``."""

    turn_degrees: float
    density: float
    density_error: float
    integrated_work: float
    integrated_error: float
    bar_work: float
    bar_error: float
    mbar_work: float
    mbar_error: float
    closed_form: float
    bare_rate: float
    bare_rate_error: float
    kappa: float
    kappa_error: float
    final_rate: float
    final_error: float


def _json_fields(text: str) -> dict[str, object]:
    parsed: object = json.loads(text)
    if not isinstance(parsed, dict):
        raise TypeError("tutorial result must be a JSON object")
    fields: dict[str, object] = {}
    for key, value in parsed.items():
        if not isinstance(key, str):
            raise TypeError("tutorial result keys must be strings")
        fields[key] = value
    return fields


def _number(fields: dict[str, object], key: str) -> float:
    try:
        value = fields[key]
    except KeyError as exc:
        raise TypeError(f"tutorial result is missing {key}") from exc
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"tutorial result field {key} must be a number")
    return float(value)


def load_ohtst_result(text: str) -> OhTstTutorialResult:
    """Parse the example's JSON object into floats."""
    fields = _json_fields(text)
    return {
        "turn_degrees": _number(fields, "turn_degrees"),
        "density": _number(fields, "density"),
        "density_error": _number(fields, "density_error"),
        "integrated_work": _number(fields, "integrated_work"),
        "integrated_error": _number(fields, "integrated_error"),
        "bar_work": _number(fields, "bar_work"),
        "bar_error": _number(fields, "bar_error"),
        "mbar_work": _number(fields, "mbar_work"),
        "mbar_error": _number(fields, "mbar_error"),
        "closed_form": _number(fields, "closed_form"),
        "bare_rate": _number(fields, "bare_rate"),
        "bare_rate_error": _number(fields, "bare_rate_error"),
        "kappa": _number(fields, "kappa"),
        "kappa_error": _number(fields, "kappa_error"),
        "final_rate": _number(fields, "final_rate"),
        "final_error": _number(fields, "final_error"),
    }


def run_ohtst(repo_root: Path) -> OhTstTutorialResult:
    """Run the compiled OH-TST example and return its JSON result."""
    binary = find_binary(repo_root)
    with tempfile.TemporaryDirectory() as workdir:
        result_path = Path(workdir) / "result.json"
        subprocess.run([str(binary), str(result_path)], check=True)
        return load_ohtst_result(result_path.read_text())


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
