"""ART nouveau is a set of options on the saddle-search configuration."""

from __future__ import annotations

from rgsaddle import MinMode, SearchConfig


def surface(x):
    a, b, c = x
    energy = (a * a - 1.0) ** 2 + 2.0 * b * b + 2.0 * c * c
    gradient = [4.0 * a * (a * a - 1.0), 4.0 * b, 4.0 * c]
    return energy, gradient


def test_art_step_leaves_the_minimum():
    config = SearchConfig()
    config.set_eigen_threshold(-0.01)
    config.set_push_step(0.3)
    config.set_activation_distance(0.0)
    config.set_perp_steps_push(0)
    config.set_displacement_amplitude(0.2)
    config.set_max_move(0.3)
    try:
        config.set_push_step(0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("a zero push must be rejected")
    session = MinMode(config, [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0])
    report = session.step(surface)
    assert session.position()[0] < 1.0
    assert report.status in (0, 1)
