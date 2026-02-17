"""Validate simulation output against closed-form queueing results.

These tests run moderately large simulations and compare the empirical
E[T] (mean response time) to known analytical formulas. Because the
simulator is stochastic, we allow a relative tolerance (default 5%).
"""

import pytest

from queue_sim import FCFS, QueueSystem, genExp

NUM_EVENTS = 500_000
RTOL = 0.05  # 5% relative tolerance


class TestMM1:
    """M/M/1 queue: Poisson arrivals, exponential service, one server."""

    @pytest.mark.parametrize("lam,mu", [
        (1.0, 2.0),    # rho = 0.5 (moderate load)
        (5.0, 10.0),   # rho = 0.5
        (8.0, 10.0),   # rho = 0.8 (heavy load)
    ])
    def test_fcfs_mean_response_time(self, lam: float, mu: float) -> None:
        """E[T] for M/M/1 FCFS should match 1/(mu - lambda)."""
        expected_T = 1.0 / (mu - lam)
        system = QueueSystem([FCFS(sizefn=genExp(mu))], arrivalfn=genExp(lam))
        _N, T = system.sim(num_events=NUM_EVENTS, seed=42)
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
        system = QueueSystem([FCFS(sizefn=genExp(mu))], arrivalfn=genExp(lam))
        N, _T = system.sim(num_events=NUM_EVENTS, seed=42)
        assert N == pytest.approx(expected_N, rel=RTOL), (
            f"lam={lam}, mu={mu}: simulated E[N]={N:.4f}, "
            f"expected={expected_N:.4f}"
        )
