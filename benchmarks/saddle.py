"""Saddle-search timings and memray peaks for the Python binding."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

from rgsaddle import Stencil, connect

_RECORDED = False


def _record() -> None:
    """Write the loaded extension checksum once per process.

    Each round is its own process. The extension contains the Rust
    static library the search calls, so its checksum is that commit's
    code.
    """
    global _RECORDED
    if _RECORDED:
        return
    _RECORDED = True
    env = os.environ.get("ASV_ENV_DIR")
    if not env:
        return
    import rgsaddle._core as core

    path = Path(core.__file__).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    commit = os.environ.get("ASV_COMMIT", "")
    line = f"{commit}\t{path}\t{digest}\n"
    with (Path(env) / "lib-load.log").open("a", encoding="utf-8") as handle:
        handle.write(line)


def bowl(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    row = np.ascontiguousarray(x, dtype=np.float64)
    v = row[0]
    energy = np.array([2.0 * v[0] * v[0] + 5.0 * v[1] * v[1]], dtype=np.float64)
    gradient = np.array([[4.0 * v[0], 10.0 * v[1]]], dtype=np.float64)
    return energy, gradient


def barrier(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    row = np.ascontiguousarray(x, dtype=np.float64)
    a, b, c = row[0]
    u = a * a * a - 4.0 * a
    energy = np.array([u * u + 2.0 * b * b + 2.0 * c * c], dtype=np.float64)
    gradient = np.array(
        [[2.0 * u * (3.0 * a * a - 4.0), 4.0 * b, 4.0 * c]], dtype=np.float64
    )
    return energy, gradient


class SaddleSearch:
    """Wall time of one dimer step and one double-ended connect.

    Each call builds its own coordinates, mode, and endpoints. There is
    no setup object to retain. ``warmup_time`` repeats those calls and
    drops them before sampling, so the cold call is not a sample. A
    recorded sample is the mean of ``number`` fresh calls from the same
    numeric start, divided back to one call. Nothing in the warm calls
    is passed into the timed calls.
    """

    rounds = 10
    repeat = 1
    number = 400
    warmup_time = 0.5
    min_run_count = 1
    timeout = 300

    def time_dimer_step(self) -> None:
        _record()
        x = np.array([0.4, 0.3], dtype=np.float64)
        mode = np.array([1.0, 0.0], dtype=np.float64)
        search = Stencil.point(x, preset="dimer", mode=mode, max_rotations=4)
        search.step(bowl)

    def time_connect(self) -> None:
        _record()
        left = np.array([-2.0, 0.0, 0.0], dtype=np.float64)
        right = np.array([2.0, 0.0, 0.0], dtype=np.float64)
        connect(barrier, left, right, force_tol=1e-3, max_calls=2000, quench_nodes=True)


class SaddleMemory:
    """Peak bytes from asv_bench_memray (methods named ray_*)."""

    rounds = 2
    number = 1
    repeat = 1
    timeout = 300

    def ray_dimer_step(self) -> None:
        _record()
        x = np.array([0.4, 0.3], dtype=np.float64)
        mode = np.array([1.0, 0.0], dtype=np.float64)
        search = Stencil.point(x, preset="dimer", mode=mode, max_rotations=4)
        search.step(bowl)

    def ray_connect(self) -> None:
        _record()
        left = np.array([-2.0, 0.0, 0.0], dtype=np.float64)
        right = np.array([2.0, 0.0, 0.0], dtype=np.float64)
        connect(barrier, left, right, force_tol=1e-3, max_calls=2000, quench_nodes=True)
