"""The tutorial scripts parse their JSON results into floats."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Callable

import pytest

ROOT = Path(__file__).resolve().parents[2]
PAFI = ROOT / "examples" / "pafi_tutorial.py"
OHTST = ROOT / "examples" / "ohtst_tutorial.py"
if not PAFI.is_file() or not OHTST.is_file():
    pytest.skip(
        f"the tutorial scripts are not in this source tree ({ROOT / 'examples'})",
        allow_module_level=True,
    )


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pafi = _load("pafi_tutorial_json", PAFI)
ohtst = _load("ohtst_tutorial_json", OHTST)


def _payload(keys: set[str], **overrides: object) -> str:
    values: dict[str, object] = {key: float(index + 1) for index, key in enumerate(sorted(keys))}
    values.update(overrides)
    return json.dumps(values)


def _check(loader: Callable[[str], dict[str, float]], keys: set[str]) -> None:
    parsed = loader(_payload(keys, nu=2) if "nu" in keys else _payload(keys, kappa=3))
    assert set(parsed) == keys
    for key, value in parsed.items():
        assert type(value) is float
        assert value > 0.0
    with pytest.raises(TypeError, match="JSON object"):
        loader("[]")
    missing = set(keys)
    dropped = missing.pop()
    with pytest.raises(TypeError, match=dropped):
        loader(_payload(missing))
    with pytest.raises(TypeError, match="must be a number"):
        loader(_payload(keys, **{dropped: True}))
    with pytest.raises(TypeError, match="must be a number"):
        loader(_payload(keys, **{dropped: "nope"}))


def test_pafi_result_is_a_float_record() -> None:
    _check(pafi.load_pafi_result, set(pafi.PafiTutorialResult.__required_keys__))


def test_ohtst_result_is_a_float_record() -> None:
    _check(ohtst.load_ohtst_result, set(ohtst.OhTstTutorialResult.__required_keys__))
