"""Validate C++ backend against closed-form queueing results.

Mirrors test_analytical.py but uses the C++ extension module.
"""

import pytest

_queue_sim_cpp = pytest.importorskip("_queue_sim_cpp")

from .helpers import erlang_b, mm1k_ploss, mmk_expected_T  # noqa: E402

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


class TestMG1Cpp:
    """M/G/1 with Uniform service via C++ backend.

    Uniform(a, b): E[S] = (a+b)/2, E[S^2] = (a^2 + ab + b^2)/3.
    FCFS P-K:  E[T] = E[S] + lam * E[S^2] / (2 * (1 - rho))
    PS:        E[T] = E[S] / (1 - rho)
    """

    A, B = 0.3, 0.7
    ES = (A + B) / 2
    ES2 = (A**2 + A * B + B**2) / 3

    @pytest.mark.parametrize("lam", [1.0, 1.6])
    def test_fcfs_pk_formula(self, lam: float) -> None:
        """M/G/1-FCFS E[T] via Pollaczek-Khinchine."""
        rho = lam * self.ES
        expected_T = self.ES + lam * self.ES2 / (2 * (1 - rho))
        server = _queue_sim_cpp.FCFS(_queue_sim_cpp.UniformDist(self.A, self.B))
        system = _queue_sim_cpp.QueueSystem(
            [server], _queue_sim_cpp.ExponentialDist(lam)
        )
        N, T = system.sim(num_events=NUM_EVENTS, seed=42)
        assert T == pytest.approx(expected_T, rel=RTOL), (
            f"lam={lam}: simulated E[T]={T:.4f}, expected={expected_T:.4f}"
        )

    @pytest.mark.parametrize("lam", [1.0, 1.6])
    def test_ps_mean_response_time(self, lam: float) -> None:
        """M/G/1-PS E[T] = E[S] / (1 - rho)."""
        rho = lam * self.ES
        expected_T = self.ES / (1 - rho)
        server = _queue_sim_cpp.PS(_queue_sim_cpp.UniformDist(self.A, self.B))
        system = _queue_sim_cpp.QueueSystem(
            [server], _queue_sim_cpp.ExponentialDist(lam)
        )
        N, T = system.sim(num_events=NUM_EVENTS, seed=42)
        assert T == pytest.approx(expected_T, rel=RTOL), (
            f"lam={lam}: simulated E[T]={T:.4f}, expected={expected_T:.4f}"
        )


class TestMM1PSCpp:
    """M/M/1-PS via C++ backend: E[T] = 1/(mu - lambda)."""

    @pytest.mark.parametrize("lam,mu", [
        (1.0, 2.0),
        (8.0, 10.0),
    ])
    def test_ps_mean_response_time(self, lam: float, mu: float) -> None:
        expected_T = 1.0 / (mu - lam)
        server = _queue_sim_cpp.PS(_queue_sim_cpp.ExponentialDist(mu))
        system = _queue_sim_cpp.QueueSystem(
            [server], _queue_sim_cpp.ExponentialDist(lam)
        )
        N, T = system.sim(num_events=NUM_EVENTS, seed=42)
        assert T == pytest.approx(expected_T, rel=RTOL), (
            f"lam={lam}, mu={mu}: simulated E[T]={T:.4f}, "
            f"expected={expected_T:.4f}"
        )


class TestMM1FBCpp:
    """M/M/1-FB via C++ backend: E[T] = 1/(mu - lambda)."""

    @pytest.mark.parametrize("lam,mu", [
        (1.0, 2.0),
        (8.0, 10.0),
    ])
    def test_fb_mean_response_time(self, lam: float, mu: float) -> None:
        expected_T = 1.0 / (mu - lam)
        server = _queue_sim_cpp.FB(_queue_sim_cpp.ExponentialDist(mu))
        system = _queue_sim_cpp.QueueSystem(
            [server], _queue_sim_cpp.ExponentialDist(lam)
        )
        N, T = system.sim(num_events=NUM_EVENTS, seed=42)
        assert T == pytest.approx(expected_T, rel=RTOL), (
            f"lam={lam}, mu={mu}: simulated E[T]={T:.4f}, "
            f"expected={expected_T:.4f}"
        )


class TestMMkCpp:
    """M/M/k via C++ backend: validates FCFS and PS against Erlang-C."""

    @pytest.mark.parametrize("lam,mu,k,expected_T", [
        (1.0, 1.0, 2, 4.0 / 3.0),
        (1.5, 1.0, 2, 16.0 / 7.0),
    ])
    def test_fcfs_mmk(self, lam: float, mu: float, k: int, expected_T: float) -> None:
        assert expected_T == pytest.approx(mmk_expected_T(lam, mu, k), rel=1e-10)
        server = _queue_sim_cpp.FCFS(
            _queue_sim_cpp.ExponentialDist(mu), num_servers=k
        )
        system = _queue_sim_cpp.QueueSystem(
            [server], _queue_sim_cpp.ExponentialDist(lam)
        )
        N, T = system.sim(num_events=NUM_EVENTS, seed=42)
        assert T == pytest.approx(expected_T, rel=RTOL), (
            f"M/M/{k} FCFS: lam={lam}, mu={mu}: simulated E[T]={T:.4f}, "
            f"expected={expected_T:.4f}"
        )

    @pytest.mark.parametrize("lam,mu,k,expected_T", [
        (1.0, 1.0, 2, 4.0 / 3.0),
        (1.5, 1.0, 2, 16.0 / 7.0),
    ])
    def test_ps_mmk(self, lam: float, mu: float, k: int, expected_T: float) -> None:
        assert expected_T == pytest.approx(mmk_expected_T(lam, mu, k), rel=1e-10)
        server = _queue_sim_cpp.PS(
            _queue_sim_cpp.ExponentialDist(mu), num_servers=k
        )
        system = _queue_sim_cpp.QueueSystem(
            [server], _queue_sim_cpp.ExponentialDist(lam)
        )
        N, T = system.sim(num_events=NUM_EVENTS, seed=42)
        assert T == pytest.approx(expected_T, rel=RTOL), (
            f"M/M/{k} PS: lam={lam}, mu={mu}: simulated E[T]={T:.4f}, "
            f"expected={expected_T:.4f}"
        )


class TestErlangBCpp:
    """M/M/c/c (Erlang loss system) via C++ backend."""

    @pytest.mark.parametrize("lam,mu,c", [
        (2.0, 1.0, 3),
        (5.0, 1.0, 5),
        (1.0, 2.0, 2),
    ])
    def test_erlang_b_loss_probability(
        self, lam: float, mu: float, c: int
    ) -> None:
        a = lam / mu
        expected_ploss = erlang_b(c, a)
        server = _queue_sim_cpp.FCFS(
            _queue_sim_cpp.ExponentialDist(mu),
            num_servers=c,
            buffer_capacity=c,
        )
        system = _queue_sim_cpp.QueueSystem(
            [server], _queue_sim_cpp.ExponentialDist(lam)
        )
        system.sim(num_events=NUM_EVENTS, seed=42)
        ploss = server.num_rejected / server.num_arrivals
        assert ploss == pytest.approx(expected_ploss, abs=0.02), (
            f"M/M/{c}/{c}: lam={lam}, mu={mu}: simulated P(loss)={ploss:.4f}, "
            f"expected={expected_ploss:.4f}"
        )


class TestMM1KCpp:
    """M/M/1/K (finite buffer) via C++ backend."""

    @pytest.mark.parametrize("lam,mu,K", [
        (1.0, 2.0, 5),
        (3.0, 4.0, 3),
        (8.0, 10.0, 10),
    ])
    def test_mm1k_loss_probability(
        self, lam: float, mu: float, K: int
    ) -> None:
        rho = lam / mu
        expected_ploss = mm1k_ploss(rho, K)
        server = _queue_sim_cpp.FCFS(
            _queue_sim_cpp.ExponentialDist(mu), buffer_capacity=K
        )
        system = _queue_sim_cpp.QueueSystem(
            [server], _queue_sim_cpp.ExponentialDist(lam)
        )
        system.sim(num_events=NUM_EVENTS, seed=42)
        ploss = server.num_rejected / server.num_arrivals
        assert ploss == pytest.approx(expected_ploss, abs=0.02), (
            f"M/M/1/{K}: lam={lam}, mu={mu}: simulated P(loss)={ploss:.4f}, "
            f"expected={expected_ploss:.4f}"
        )
