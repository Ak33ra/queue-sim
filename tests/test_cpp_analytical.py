"""Validate C++ backend against closed-form queueing results.

Mirrors test_analytical.py but uses the C++ extension module.
"""

import pytest

_queue_sim_cpp = pytest.importorskip("_queue_sim_cpp")

NUM_EVENTS = 500_000
RTOL = 0.05  # 5% relative tolerance


class TestMM1Cpp:
    """M/M/1 queue via C++ backend."""

    @pytest.mark.parametrize("lam,mu", [
        (1.0, 2.0),
        (5.0, 10.0),
        (8.0, 10.0),
    ])
    def test_fcfs_mean_response_time(self, lam: float, mu: float) -> None:
        """E[T] for M/M/1 FCFS should match 1/(mu - lambda)."""
        expected_T = 1.0 / (mu - lam)
        server = _queue_sim_cpp.FCFS(_queue_sim_cpp.ExponentialDist(mu))
        system = _queue_sim_cpp.QueueSystem(
            [server], _queue_sim_cpp.ExponentialDist(lam)
        )
        N, T = system.sim(num_events=NUM_EVENTS, seed=42)
        assert T == pytest.approx(expected_T, rel=RTOL), (
            f"lam={lam}, mu={mu}: simulated E[T]={T:.4f}, "
            f"expected={expected_T:.4f}"
        )

    @pytest.mark.parametrize("lam,mu", [
        (1.0, 2.0),
        (5.0, 10.0),
        (8.0, 10.0),
    ])
    def test_fcfs_mean_number_in_system(self, lam: float, mu: float) -> None:
        """E[N] for M/M/1 FCFS should match rho/(1-rho)."""
        rho = lam / mu
        expected_N = rho / (1 - rho)
        server = _queue_sim_cpp.FCFS(_queue_sim_cpp.ExponentialDist(mu))
        system = _queue_sim_cpp.QueueSystem(
            [server], _queue_sim_cpp.ExponentialDist(lam)
        )
        N, T = system.sim(num_events=NUM_EVENTS, seed=42)
        assert N == pytest.approx(expected_N, rel=RTOL), (
            f"lam={lam}, mu={mu}: simulated E[N]={N:.4f}, "
            f"expected={expected_N:.4f}"
        )
