# Copyright (c) rgsaddle developers
"""The two decisions a host can take from energies it already holds.

    python3 examples/step_controls.py
"""

from rgsaddle._core import halve_raised_energy, trust_update


def main() -> None:
    raised, step = halve_raised_energy(1.0, 1.0002, 0.4)
    assert raised is True
    assert step == 0.4 * 0.5
    kept, same = halve_raised_energy(1.0, 1.0, 0.4)
    assert kept is False
    assert same == 0.4
    # Sella minimum schedule.
    shrunk = trust_update(0.2, -1.0, 0.0, 0.2, 1.15, 0.90, 1.035, 100.0, 1e-4)
    assert shrunk == 0.2 * 0.90
    held = trust_update(0.2, 1e-16, 1.0, 0.2, 1.15, 0.90, 1.035, 100.0, 1e-4)
    assert held == 0.2
    print(f"raised {raised} step {step} radius {shrunk} held {held}")


if __name__ == "__main__":
    main()
