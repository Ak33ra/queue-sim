"""Tests for queue_sim.animate module."""

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # non-interactive backend for CI

import matplotlib.animation  # noqa: E402

from queue_sim import FCFS, QueueSystem, genExp  # noqa: E402
from queue_sim.animate import animate_network  # noqa: E402

NUM_EVENTS = 5_000


@pytest.fixture()
def single_server_log():
    """Event log from a single-server M/M/1 queue."""
    system = QueueSystem([FCFS(sizefn=genExp(2.0))], arrivalfn=genExp(1.0))
    system.sim(num_events=NUM_EVENTS, seed=42, track_events=True)
    return system.event_log


@pytest.fixture()
def tandem_log():
    """Event log from a 2-server tandem."""
    s0 = FCFS(sizefn=genExp(3.0))
    s1 = FCFS(sizefn=genExp(3.0))
    system = QueueSystem([s0, s1], arrivalfn=genExp(1.0))
    system.sim(num_events=NUM_EVENTS, seed=42, track_events=True)
    return system.event_log


class TestAnimateNetwork:
    def test_returns_func_animation(self, single_server_log):
        anim = animate_network(single_server_log, n_frames=10)
        assert isinstance(anim, matplotlib.animation.FuncAnimation)

    def test_single_server(self, single_server_log):
        anim = animate_network(single_server_log, n_frames=10)
        assert anim is not None

    def test_tandem(self, tandem_log):
        anim = animate_network(tandem_log, n_frames=10)
        assert isinstance(anim, matplotlib.animation.FuncAnimation)

    def test_custom_positions(self, tandem_log):
        positions = {0: (0.2, 0.5), 1: (0.8, 0.5)}
        anim = animate_network(tandem_log, positions=positions, n_frames=10)
        assert isinstance(anim, matplotlib.animation.FuncAnimation)

    def test_custom_transition_matrix(self, tandem_log):
        tm = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        anim = animate_network(
            tandem_log, transition_matrix=tm, n_frames=10
        )
        assert isinstance(anim, matplotlib.animation.FuncAnimation)

    def test_no_colorbar(self, single_server_log):
        anim = animate_network(
            single_server_log, n_frames=10, show_colorbar=False
        )
        assert isinstance(anim, matplotlib.animation.FuncAnimation)

    def test_with_title(self, single_server_log):
        anim = animate_network(
            single_server_log, n_frames=10, title="M/M/1 Queue"
        )
        assert isinstance(anim, matplotlib.animation.FuncAnimation)
