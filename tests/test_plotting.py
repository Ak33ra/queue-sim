"""Tests for queue_sim.plotting module."""

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # non-interactive backend for CI

import matplotlib.axes  # noqa: E402
import matplotlib.figure  # noqa: E402

from queue_sim import FCFS, PS, SRPT, QueueSystem, genExp  # noqa: E402
from queue_sim.plotting import compare_policies, plot_cdf, plot_tail  # noqa: E402

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
