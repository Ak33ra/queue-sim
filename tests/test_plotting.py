"""Tests for queue_sim.plotting module."""

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # non-interactive backend for CI

import matplotlib.axes  # noqa: E402
import matplotlib.figure  # noqa: E402

from queue_sim import FCFS, PS, SRPT, QueueSystem, genExp  # noqa: E402
from queue_sim.plotting import (  # noqa: E402
    compare_policies,
    plot_cdf,
    plot_server_occupancy,
    plot_system_state,
    plot_tail,
)

NUM_EVENTS = 10_000


@pytest.fixture()
def response_times():
    """Generate response times from an M/M/1-FCFS queue."""
    system = QueueSystem([FCFS(sizefn=genExp(2.0))], arrivalfn=genExp(1.0))
    system.sim(num_events=NUM_EVENTS, seed=42, track_response_times=True)
    return system.response_times


@pytest.fixture()
def multi_policy_rts():
    """Generate response times for FCFS, PS, and SRPT."""
    rts = {}
    for name, cls in [("FCFS", FCFS), ("PS", PS), ("SRPT", SRPT)]:
        system = QueueSystem([cls(sizefn=genExp(2.0))], arrivalfn=genExp(1.0))
        system.sim(num_events=NUM_EVENTS, seed=42, track_response_times=True)
        rts[name] = system.response_times
    return rts


class TestPlotCdf:
    def test_returns_fig_ax(self, response_times):
        fig, ax = plot_cdf(response_times)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_accepts_existing_ax(self, response_times):
        import matplotlib.pyplot as plt

        fig0, ax0 = plt.subplots()
        fig, ax = plot_cdf(response_times, ax=ax0)
        assert ax is ax0
        assert fig is fig0
        plt.close("all")

    def test_with_label(self, response_times):
        fig, ax = plot_cdf(response_times, label="FCFS")
        texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert "FCFS" in texts

    def test_y_range(self, response_times):
        fig, ax = plot_cdf(response_times)
        lines = ax.get_lines()
        assert len(lines) == 1
        ydata = lines[0].get_ydata()
        assert ydata[-1] == pytest.approx(1.0)
        assert ydata[0] == pytest.approx(1.0 / NUM_EVENTS)


class TestPlotTail:
    def test_returns_fig_ax(self, response_times):
        fig, ax = plot_tail(response_times)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_log_scale_default(self, response_times):
        fig, ax = plot_tail(response_times)
        assert ax.get_yscale() == "log"

    def test_linear_scale(self, response_times):
        fig, ax = plot_tail(response_times, log=False)
        assert ax.get_yscale() == "linear"

    def test_accepts_existing_ax(self, response_times):
        import matplotlib.pyplot as plt

        fig0, ax0 = plt.subplots()
        fig, ax = plot_tail(response_times, ax=ax0)
        assert ax is ax0
        plt.close("all")


class TestComparePolicies:
    def test_cdf_overlay(self, multi_policy_rts):
        fig, ax = compare_policies(multi_policy_rts, kind="cdf")
        assert isinstance(fig, matplotlib.figure.Figure)
        # One line per policy
        assert len(ax.get_lines()) == len(multi_policy_rts)

    def test_tail_overlay(self, multi_policy_rts):
        fig, ax = compare_policies(multi_policy_rts, kind="tail")
        assert len(ax.get_lines()) == len(multi_policy_rts)
        assert ax.get_yscale() == "log"

    def test_legend_labels(self, multi_policy_rts):
        fig, ax = compare_policies(multi_policy_rts)
        texts = {t.get_text() for t in ax.get_legend().get_texts()}
        assert texts == set(multi_policy_rts.keys())

    def test_invalid_kind_raises(self, multi_policy_rts):
        with pytest.raises(ValueError, match="kind must be"):
            compare_policies(multi_policy_rts, kind="histogram")

    def test_accepts_existing_ax(self, multi_policy_rts):
        import matplotlib.pyplot as plt

        fig0, ax0 = plt.subplots()
        fig, ax = compare_policies(multi_policy_rts, ax=ax0)
        assert ax is ax0
        plt.close("all")


@pytest.fixture()
def event_log():
    """Generate an event log from an M/M/1-FCFS queue."""
    system = QueueSystem([FCFS(sizefn=genExp(2.0))], arrivalfn=genExp(1.0))
    system.sim(num_events=NUM_EVENTS, seed=42, track_events=True)
    return system.event_log


@pytest.fixture()
def tandem_event_log():
    """Generate an event log from a 2-server tandem."""
    s0 = FCFS(sizefn=genExp(3.0))
    s1 = FCFS(sizefn=genExp(3.0))
    system = QueueSystem([s0, s1], arrivalfn=genExp(1.0))
    system.sim(num_events=NUM_EVENTS, seed=42, track_events=True)
    return system.event_log


class TestPlotSystemState:
    def test_returns_fig_ax(self, event_log):
        fig, ax = plot_system_state(event_log)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_accepts_existing_ax(self, event_log):
        import matplotlib.pyplot as plt

        fig0, ax0 = plt.subplots()
        fig, ax = plot_system_state(event_log, ax=ax0)
        assert ax is ax0
        assert fig is fig0
        plt.close("all")

    def test_has_lines(self, event_log):
        fig, ax = plot_system_state(event_log)
        assert len(ax.get_lines()) >= 1


class TestPlotServerOccupancy:
    def test_returns_fig_ax(self, tandem_event_log):
        fig, ax = plot_server_occupancy(tandem_event_log)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_accepts_existing_ax(self, tandem_event_log):
        import matplotlib.pyplot as plt

        fig0, ax0 = plt.subplots()
        fig, ax = plot_server_occupancy(tandem_event_log, ax=ax0)
        assert ax is ax0
        plt.close("all")

    def test_has_collections(self, tandem_event_log):
        fig, ax = plot_server_occupancy(tandem_event_log)
        # pcolormesh creates a QuadMesh collection
        assert len(ax.collections) >= 1

    def test_single_server(self, event_log):
        fig, ax = plot_server_occupancy(event_log)
        assert isinstance(fig, matplotlib.figure.Figure)
