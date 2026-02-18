"""Tests for response time distribution tracking (Python backend)."""

import statistics

import pytest

from queue_sim import FB, FCFS, PS, SRPT, QueueSystem, genExp

NUM_EVENTS = 100_000
RTOL = 0.05  # 5% relative tolerance


class TestResponseTimesLength:
    """len(response_times) == num_events for every policy."""

    @pytest.mark.parametrize("policy_cls", [FCFS, PS, FB, SRPT])
    def test_length_matches_num_events(self, policy_cls):
        server = policy_cls(sizefn=genExp(2.0))
        system = QueueSystem([server], arrivalfn=genExp(1.0))
        system.sim(num_events=NUM_EVENTS, seed=42, track_response_times=True)
        assert len(system.response_times) == NUM_EVENTS


class TestResponseTimesPositive:
    """All response times must be positive."""

    @pytest.mark.parametrize("policy_cls", [FCFS, PS, FB, SRPT])
    def test_all_positive(self, policy_cls):
        server = policy_cls(sizefn=genExp(2.0))
        system = QueueSystem([server], arrivalfn=genExp(1.0))
        system.sim(num_events=NUM_EVENTS, seed=42, track_response_times=True)
        assert all(t > 0 for t in system.response_times)


class TestResponseTimesMeanMatchesET:
    """mean(response_times) ≈ system E[T] within tolerance."""

    @pytest.mark.parametrize("policy_cls", [FCFS, PS, FB, SRPT])
    def test_mean_approx_ET(self, policy_cls):
        server = policy_cls(sizefn=genExp(2.0))
        system = QueueSystem([server], arrivalfn=genExp(1.0))
        _N, T = system.sim(
            num_events=NUM_EVENTS, seed=42, track_response_times=True
        )
        mean_rt = statistics.mean(system.response_times)
        assert mean_rt == pytest.approx(T, rel=RTOL)


class TestSRPTNowReportsT:
    """SRPT server.T should be > 0 after simulation (was 0 before rework)."""

    def test_srpt_server_T_positive(self):
        server = SRPT(sizefn=genExp(2.0))
        system = QueueSystem([server], arrivalfn=genExp(1.0))
        system.sim(num_events=NUM_EVENTS, seed=42)
        assert server.T > 0


class TestDefaultNoAttribute:
    """When track_response_times=False (default), no response_times attribute."""

    def test_no_attribute_by_default(self):
        server = FCFS(sizefn=genExp(2.0))
        system = QueueSystem([server], arrivalfn=genExp(1.0))
        system.sim(num_events=1000, seed=42)
        assert not hasattr(system, "response_times")


class TestBackwardCompat:
    """Return type of sim() is unchanged."""

    def test_return_type(self):
        server = FCFS(sizefn=genExp(2.0))
        system = QueueSystem([server], arrivalfn=genExp(1.0))
        result = system.sim(
            num_events=1000, seed=42, track_response_times=True
        )
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestWithBufferCapacity:
    """Tracking works correctly with finite buffers."""

    def test_buffer_capacity(self):
        server = FCFS(sizefn=genExp(2.0), buffer_capacity=10)
        system = QueueSystem([server], arrivalfn=genExp(1.0))
        system.sim(num_events=NUM_EVENTS, seed=42, track_response_times=True)
        assert len(system.response_times) == NUM_EVENTS
        assert all(t > 0 for t in system.response_times)


class TestMultiServer:
    """Tracking works with num_servers=2."""

    @pytest.mark.parametrize("policy_cls", [FCFS, PS])
    def test_multiserver(self, policy_cls):
        server = policy_cls(sizefn=genExp(2.0), num_servers=2)
        system = QueueSystem([server], arrivalfn=genExp(1.0))
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
        server1 = FCFS(sizefn=genExp(2.0))
        system1 = QueueSystem([server1], arrivalfn=genExp(1.0))
        N1, T1 = system1.sim(num_events=NUM_EVENTS, seed=42)

        server2 = FCFS(sizefn=genExp(2.0))
        system2 = QueueSystem([server2], arrivalfn=genExp(1.0))
        N2, T2 = system2.sim(
            num_events=NUM_EVENTS, seed=42, track_response_times=True
        )

        assert N1 == N2
        assert T1 == T2

    def test_response_times_deterministic(self):
        """Same seed produces identical response_times."""
        server1 = FCFS(sizefn=genExp(2.0))
        system1 = QueueSystem([server1], arrivalfn=genExp(1.0))
        system1.sim(num_events=1000, seed=42, track_response_times=True)

        server2 = FCFS(sizefn=genExp(2.0))
        system2 = QueueSystem([server2], arrivalfn=genExp(1.0))
        system2.sim(num_events=1000, seed=42, track_response_times=True)

        assert system1.response_times == system2.response_times
