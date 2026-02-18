"""Validate simulation output against closed-form queueing results.

These tests run moderately large simulations and compare the empirical
E[T] (mean response time) to known analytical formulas. Because the
simulator is stochastic, we allow a relative tolerance (default 5%).
"""

import pytest

from queue_sim import FB, FCFS, PS, QueueSystem, genExp, genUniform

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


class TestMG1:
    """M/G/1 queue with Uniform service — tests P-K formula for FCFS
    and E[S]/(1-rho) for PS. These produce *different* E[T] values,
    unlike M/M/1 where all work-conserving policies agree.

    Uniform(a, b): E[S] = (a+b)/2, E[S^2] = (a^2 + ab + b^2)/3.
    FCFS P-K:  E[T] = E[S] + lam * E[S^2] / (2 * (1 - rho))
    PS:        E[T] = E[S] / (1 - rho)
    """

    A, B = 0.3, 0.7  # Uniform service parameters
    ES = (A + B) / 2                          # 0.5
    ES2 = (A**2 + A * B + B**2) / 3          # 0.79/3

    @pytest.mark.parametrize("lam", [1.0, 1.6])
    def test_fcfs_pk_formula(self, lam: float) -> None:
        """M/G/1-FCFS E[T] via Pollaczek-Khinchine."""
        rho = lam * self.ES
        expected_T = self.ES + lam * self.ES2 / (2 * (1 - rho))
        system = QueueSystem(
            [FCFS(sizefn=genUniform(self.A, self.B))],
            arrivalfn=genExp(lam),
        )
        _N, T = system.sim(num_events=NUM_EVENTS, seed=42)
        assert T == pytest.approx(expected_T, rel=RTOL), (
            f"lam={lam}: simulated E[T]={T:.4f}, expected={expected_T:.4f}"
        )

    @pytest.mark.parametrize("lam", [1.0, 1.6])
    def test_ps_mean_response_time(self, lam: float) -> None:
        """M/G/1-PS E[T] = E[S] / (1 - rho)."""
        rho = lam * self.ES
        expected_T = self.ES / (1 - rho)
        system = QueueSystem(
            [PS(sizefn=genUniform(self.A, self.B))],
            arrivalfn=genExp(lam),
        )
        _N, T = system.sim(num_events=NUM_EVENTS, seed=42)
        assert T == pytest.approx(expected_T, rel=RTOL), (
            f"lam={lam}: simulated E[T]={T:.4f}, expected={expected_T:.4f}"
        )


class TestMM1PS:
    """M/M/1-PS: E[T] = 1/(mu - lambda), same as FCFS for exponential."""

    @pytest.mark.parametrize("lam,mu", [
        (1.0, 2.0),
        (8.0, 10.0),
    ])
    def test_ps_mean_response_time(self, lam: float, mu: float) -> None:
        expected_T = 1.0 / (mu - lam)
        system = QueueSystem([PS(sizefn=genExp(mu))], arrivalfn=genExp(lam))
        _N, T = system.sim(num_events=NUM_EVENTS, seed=42)
        assert T == pytest.approx(expected_T, rel=RTOL), (
            f"lam={lam}, mu={mu}: simulated E[T]={T:.4f}, "
            f"expected={expected_T:.4f}"
        )


class TestMM1FB:
    """M/M/1-FB: E[T] = 1/(mu - lambda), same as FCFS for exponential."""

    @pytest.mark.parametrize("lam,mu", [
        (1.0, 2.0),
        (8.0, 10.0),
    ])
    def test_fb_mean_response_time(self, lam: float, mu: float) -> None:
        expected_T = 1.0 / (mu - lam)
        system = QueueSystem([FB(sizefn=genExp(mu))], arrivalfn=genExp(lam))
        _N, T = system.sim(num_events=NUM_EVENTS, seed=42)
        assert T == pytest.approx(expected_T, rel=RTOL), (
            f"lam={lam}, mu={mu}: simulated E[T]={T:.4f}, "
            f"expected={expected_T:.4f}"
        )
