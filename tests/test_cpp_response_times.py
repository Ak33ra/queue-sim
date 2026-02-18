"""Tests for response time distribution tracking (C++ backend)."""

import statistics

import pytest

_queue_sim_cpp = pytest.importorskip("_queue_sim_cpp")

NUM_EVENTS = 100_000
RTOL = 0.05  # 5% relative tolerance


def _make_system(policy_cls, **kwargs):
    """Create a QueueSystem with the given policy, lam=1, mu=2."""
    server = policy_cls(_queue_sim_cpp.ExponentialDist(2.0), **kwargs)
    return _queue_sim_cpp.QueueSystem(
        [server], _queue_sim_cpp.ExponentialDist(1.0)
    )


class TestResponseTimesLength:
    """len(response_times) == num_events for every policy."""

    @pytest.mark.parametrize("policy_cls", [
        _queue_sim_cpp.FCFS, _queue_sim_cpp.PS,
        _queue_sim_cpp.FB, _queue_sim_cpp.SRPT,
    ])
    def test_length_matches_num_events(self, policy_cls):
        system = _make_system(policy_cls)
        system.sim(num_events=NUM_EVENTS, seed=42, track_response_times=True)
        assert len(system.response_times) == NUM_EVENTS


class TestResponseTimesPositive:
    """All response times must be positive."""

    @pytest.mark.parametrize("policy_cls", [
        _queue_sim_cpp.FCFS, _queue_sim_cpp.PS,
        _queue_sim_cpp.FB, _queue_sim_cpp.SRPT,
    ])
    def test_all_positive(self, policy_cls):
        system = _make_system(policy_cls)
        system.sim(num_events=NUM_EVENTS, seed=42, track_response_times=True)
        assert all(t > 0 for t in system.response_times)


class TestResponseTimesMeanMatchesET:
    """mean(response_times) ≈ system E[T] within tolerance."""

    @pytest.mark.parametrize("policy_cls", [
        _queue_sim_cpp.FCFS, _queue_sim_cpp.PS,
        _queue_sim_cpp.FB, _queue_sim_cpp.SRPT,
    ])
    def test_mean_approx_ET(self, policy_cls):
        system = _make_system(policy_cls)
        _N, T = system.sim(
            num_events=NUM_EVENTS, seed=42, track_response_times=True
        )
        mean_rt = statistics.mean(system.response_times)
        assert mean_rt == pytest.approx(T, rel=RTOL)


class TestSRPTNowReportsT:
    """SRPT server.T should be > 0 after simulation (was 0 before rework)."""

    def test_srpt_server_T_positive(self):
        server = _queue_sim_cpp.SRPT(_queue_sim_cpp.ExponentialDist(2.0))
        system = _queue_sim_cpp.QueueSystem(
            [server], _queue_sim_cpp.ExponentialDist(1.0)
        )
        system.sim(num_events=NUM_EVENTS, seed=42)
        assert server.T > 0


class TestDefaultEmpty:
    """When track_response_times=False (default), response_times is empty."""

    def test_empty_by_default(self):
        system = _make_system(_queue_sim_cpp.FCFS)
        system.sim(num_events=1000, seed=42)
        assert len(system.response_times) == 0


class TestBackwardCompat:
    """Return type of sim() is unchanged."""

    def test_return_type(self):
        system = _make_system(_queue_sim_cpp.FCFS)
        result = system.sim(
            num_events=1000, seed=42, track_response_times=True
        )
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestWithBufferCapacity:
    """Tracking works correctly with finite buffers."""

    def test_buffer_capacity(self):
        server = _queue_sim_cpp.FCFS(
            _queue_sim_cpp.ExponentialDist(2.0), buffer_capacity=10
        )
        system = _queue_sim_cpp.QueueSystem(
            [server], _queue_sim_cpp.ExponentialDist(1.0)
        )
        system.sim(num_events=NUM_EVENTS, seed=42, track_response_times=True)
        assert len(system.response_times) == NUM_EVENTS
        assert all(t > 0 for t in system.response_times)


class TestMultiServer:
    """Tracking works with num_servers=2."""

    @pytest.mark.parametrize("policy_cls", [_queue_sim_cpp.FCFS, _queue_sim_cpp.PS])
    def test_multiserver(self, policy_cls):
        server = policy_cls(
            _queue_sim_cpp.ExponentialDist(2.0), num_servers=2
        )
        system = _queue_sim_cpp.QueueSystem(
            [server], _queue_sim_cpp.ExponentialDist(1.0)
        )
        _N, T = system.sim(
            num_events=NUM_EVENTS, seed=42, track_response_times=True
        )
        assert len(system.response_times) == NUM_EVENTS
        assert all(t > 0 for t in system.response_times)
        mean_rt = statistics.mean(system.response_times)
        assert mean_rt == pytest.approx(T, rel=RTOL)


class TestDeterminism:
    """Tracking doesn't change sim results; response times are deterministic."""

    def test_tracking_does_not_change_results(self):
        """E[N], E[T] identical whether tracking is on or off."""
        system1 = _make_system(_queue_sim_cpp.FCFS)
        N1, T1 = system1.sim(num_events=NUM_EVENTS, seed=42)

        system2 = _make_system(_queue_sim_cpp.FCFS)
        N2, T2 = system2.sim(
            num_events=NUM_EVENTS, seed=42, track_response_times=True
        )

        assert N1 == N2
        assert T1 == T2

    def test_response_times_deterministic(self):
        """Same seed produces identical response_times."""
        system1 = _make_system(_queue_sim_cpp.FCFS)
        system1.sim(num_events=1000, seed=42, track_response_times=True)

        system2 = _make_system(_queue_sim_cpp.FCFS)
        system2.sim(num_events=1000, seed=42, track_response_times=True)

        assert list(system1.response_times) == list(system2.response_times)
