"""PAFI tutorial, Python side.

PAFI has no Python session yet (see ``howto/python``): the sampling in
``rgsaddle::pafi`` is a Rust-only entry point. This script plays the role
Python usually takes in a mixed workflow: it owns the model parameters,
writes them out, runs the compiled example that does the sampling, and
reads the result back for printing. The potential, the path and the
closed-form check are the same ones in ``examples/pafi_tutorial.rs``;
see that file and ``docs/orgmode/tutorials/pafi.org`` for the physics.

Run from a checkout after building the example once:

.. code-block:: console

    $ cargo build --release --example pafi_tutorial
    $ python3 examples/pafi_tutorial.py
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TypedDict


def stiffness(x: float) -> float:
    """``k(x) = 2.5 + 1.5 x``, the transverse stiffness of the valley."""
    return 2.5 + 1.5 * x


def closed_form_barrier(kt: float) -> float:
    """The exact transverse-Gaussian barrier at ``kt``.

    Matches ``closed_form_barrier`` in ``examples/pafi_tutorial.rs``:
    with the transverse coordinate integrated out exactly, ``dF = 1 +
    0.5 kt ln(k(0) / k(-1))``.
    """
    return 1.0 + 0.5 * kt * math.log(stiffness(0.0) / stiffness(-1.0))


def find_binary(repo_root: Path) -> Path:
    """Locate the compiled example, building it if it is missing."""
    binary = repo_root / "target" / "release" / "examples" / "pafi_tutorial"
    if not binary.is_file():
        subprocess.run(
            ["cargo", "build", "--release", "--example", "pafi_tutorial"],
            cwd=repo_root,
            check=True,
        )
    if not binary.is_file():
        raise FileNotFoundError(f"did not find {binary} after building it")
    return binary


class PafiTutorialResult(TypedDict):
    """JSON written by ``examples/pafi_tutorial.rs``."""

    cold_barrier: float
    barrier: float
    barrier_error: float
    closed_form: float
    nu: float
    rate: float
    tau_max: float
    n_eff_min: float


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


def load_pafi_result(text: str) -> PafiTutorialResult:
    """Parse the example's JSON object into floats."""
    fields = _json_fields(text)
    return {
        "cold_barrier": _number(fields, "cold_barrier"),
        "barrier": _number(fields, "barrier"),
        "barrier_error": _number(fields, "barrier_error"),
        "closed_form": _number(fields, "closed_form"),
        "nu": _number(fields, "nu"),
        "rate": _number(fields, "rate"),
        "tau_max": _number(fields, "tau_max"),
        "n_eff_min": _number(fields, "n_eff_min"),
    }


def run_pafi(repo_root: Path, config: dict[str, float]) -> PafiTutorialResult:
    """Run the compiled PAFI example on ``config`` and return its result."""
    binary = find_binary(repo_root)
    with tempfile.TemporaryDirectory() as workdir:
        config_path = Path(workdir) / "config.json"
        result_path = Path(workdir) / "result.json"
        config_path.write_text(json.dumps(config))
        subprocess.run(
            [str(binary), str(config_path), str(result_path)],
            check=True,
        )
        return load_pafi_result(result_path.read_text())


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    config = {
        "kt": 0.1,
        "n_images": 21,
        "n_planes": 49,
        "thermalize": 2000,
        "stride": 4,
        "samples": 30000,
        "blocks": 24,
        "seed": 7,
    }
    result = run_pafi(repo_root, config)

    print(f"zero-temperature barrier along the path: {result['cold_barrier']:.6f}")
    print(
        f"PAFI barrier at kT = {config['kt']}: "
        f"{result['barrier']:.5f} +/- {result['barrier_error']:.5f}"
    )
    print(f"closed-form transverse-Gaussian barrier: {closed_form_barrier(config['kt']):.5f}")
    print(f"reactant attempt frequency (unit mass): {result['nu']:.5f}")
    print(f"rate k = nu * exp(-barrier / kT): {result['rate']:.6e}")

    reference = closed_form_barrier(config["kt"])
    deviation = abs(result["barrier"] - reference) / result["barrier_error"]
    print(f"measured barrier is {deviation:.2f} standard errors from the closed form")


if __name__ == "__main__":
    main()
