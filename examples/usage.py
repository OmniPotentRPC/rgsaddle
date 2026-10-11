"""Public enum controls.

The values are the C discriminants. ``import rgsaddle`` does not import ASE.
"""

from __future__ import annotations

import rgsaddle


def gate_name(gate: rgsaddle.ForceGate) -> str:
    """Return the ``IntEnum`` member name."""
    return gate.name


def main() -> None:
    assert rgsaddle.__version__ == "0.2.2"
    assert gate_name(rgsaddle.ForceGate.L2_NORM) == "L2_NORM"
    assert int(rgsaddle.HessUpdate.TS_BFGS) == 1
    assert int(rgsaddle.MinModeKind.DIMER) == 0
    assert int(rgsaddle.IrcKind.MOROKUMA) == 1


if __name__ == "__main__":
    main()
