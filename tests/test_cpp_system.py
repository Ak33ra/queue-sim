"""Smoke tests for the C++ backend.

Covers seed reproducibility, transition matrix validation,
tandem networks, and type checking.
"""

import pytest

_queue_sim_cpp = pytest.importorskip("_queue_sim_cpp")


class TestSeedReproducibility:
    """Same seed must produce identical results."""

    def test_same_seed_same_result(self) -> None:
        def make_system():
            server = _queue_sim_cpp.FCFS(_queue_sim_cpp.ExponentialDist(2.0))
            return _queue_sim_cpp.QueueSystem(
                [server], _queue_sim_cpp.ExponentialDist(1.0)
            )

        r1 = make_system().sim(num_events=100_000, seed=123)
        r2 = make_system().sim(num_events=100_000, seed=123)
        assert r1 == r2

    def test_different_seed_different_result(self) -> None:
        def make_system():
            server = _queue_sim_cpp.FCFS(_queue_sim_cpp.ExponentialDist(2.0))
            return _queue_sim_cpp.QueueSystem(
                [server], _queue_sim_cpp.ExponentialDist(1.0)
            )

        r1 = make_system().sim(num_events=100_000, seed=1)
        r2 = make_system().sim(num_events=100_000, seed=2)
        assert r1 != r2


class TestTransitionMatrix:
    """Transition matrix validation."""

    def test_wrong_dimensions_raises(self) -> None:
        s1 = _queue_sim_cpp.FCFS(_queue_sim_cpp.ExponentialDist(2.0))
        system = _queue_sim_cpp.QueueSystem(
            [s1],
            _queue_sim_cpp.ExponentialDist(1.0),
            transitionMatrix=[[0.5, 0.5], [0.5, 0.5]],  # 2 rows for 1 server
        )
        with pytest.raises(Exception):
            system.sim(num_events=100, seed=1)

    def test_row_sum_not_one_raises(self) -> None:
        s1 = _queue_sim_cpp.FCFS(_queue_sim_cpp.ExponentialDist(2.0))
        system = _queue_sim_cpp.QueueSystem(
            [s1],
            _queue_sim_cpp.ExponentialDist(1.0),
            transitionMatrix=[[0.5, 0.3]],  # sums to 0.8
        )
        with pytest.raises(Exception):
            system.sim(num_events=100, seed=1)


class TestTandemNetwork:
    """Two-server tandem queue (no transition matrix)."""

    def test_tandem_runs_without_error(self) -> None:
        s1 = _queue_sim_cpp.FCFS(_queue_sim_cpp.ExponentialDist(3.0))
        s2 = _queue_sim_cpp.FCFS(_queue_sim_cpp.ExponentialDist(3.0))
        system = _queue_sim_cpp.QueueSystem(
            [s1, s2], _queue_sim_cpp.ExponentialDist(1.0)
        )
        N, T = system.sim(num_events=50_000, seed=42)
        assert N > 0
        assert T > 0


class TestSRPT:
    """SRPT policy smoke tests."""

    def test_srpt_single_server(self) -> None:
        server = _queue_sim_cpp.SRPT(_queue_sim_cpp.ExponentialDist(2.0))
        system = _queue_sim_cpp.QueueSystem(
            [server], _queue_sim_cpp.ExponentialDist(1.0)
        )
        N, T = system.sim(num_events=100_000, seed=42)
        # SRPT should have lower or equal E[T] compared to M/M/1 FCFS
        # M/M/1 FCFS E[T] = 1/(mu - lam) = 1.0
        assert T < 1.1  # allow some slack


class TestDistributions:
    """Verify distribution types can be constructed."""

    def test_exponential(self) -> None:
        d = _queue_sim_cpp.ExponentialDist(1.5)
        assert d is not None

    def test_uniform(self) -> None:
        d = _queue_sim_cpp.UniformDist(0.0, 1.0)
        assert d is not None

    def test_bounded_pareto(self) -> None:
        d = _queue_sim_cpp.BoundedParetoDist(1.0, 10.0, 1.5)
        assert d is not None

    def test_invalid_dist_type_raises(self) -> None:
        with pytest.raises(TypeError):
            _queue_sim_cpp.FCFS("not_a_distribution")
