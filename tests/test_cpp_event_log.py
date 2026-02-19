"""Tests for event log tracking (C++ backend)."""

import pytest

_queue_sim_cpp = pytest.importorskip("_queue_sim_cpp")

from queue_sim.event_log import per_server_states  # noqa: E402

NUM_EVENTS = 10_000
EventLog = _queue_sim_cpp.EventLog
VALID_KINDS = {EventLog.ARRIVAL, EventLog.DEPARTURE, EventLog.ROUTE, EventLog.REJECTION}


def _make_system(policy_cls, **kwargs):
    """Create a QueueSystem with the given policy, lam=1, mu=2."""
    server = policy_cls(_queue_sim_cpp.ExponentialDist(2.0), **kwargs)
    return _queue_sim_cpp.QueueSystem(
        [server], _queue_sim_cpp.ExponentialDist(1.0)
    )


def _make_tandem():
    """Create a 2-server tandem QueueSystem."""
    s0 = _queue_sim_cpp.FCFS(_queue_sim_cpp.ExponentialDist(3.0))
    s1 = _queue_sim_cpp.FCFS(_queue_sim_cpp.ExponentialDist(3.0))
    return _queue_sim_cpp.QueueSystem(
        [s0, s1], _queue_sim_cpp.ExponentialDist(1.0)
    )


class TestEventLogLength:
    """Events are logged when track_events=True."""

    def test_events_logged(self):
        system = _make_system(_queue_sim_cpp.FCFS)
        system.sim(num_events=NUM_EVENTS, seed=42, track_events=True)
        assert len(system.event_log) > 0


class TestDefaultEmpty:
    """event_log is empty when track_events=False."""

    def test_empty_by_default(self):
        system = _make_system(_queue_sim_cpp.FCFS)
        system.sim(num_events=1000, seed=42)
        assert len(system.event_log) == 0


class TestAllKindsValid:
    """Every logged kind is one of the 4 constants."""

    def test_kinds_valid(self):
        system = _make_system(_queue_sim_cpp.FCFS)
        system.sim(num_events=NUM_EVENTS, seed=42, track_events=True)
        assert all(k in VALID_KINDS for k in system.event_log.kinds)


class TestTimesNonDecreasing:
    """Event times must be monotonically non-decreasing."""

    def test_non_decreasing(self):
        system = _make_system(_queue_sim_cpp.FCFS)
        system.sim(num_events=NUM_EVENTS, seed=42, track_events=True)
        times = system.event_log.times
        assert all(a <= b for a, b in zip(times, times[1:]))


class TestDepartureCount:
    """Single-server no-buffer: departures == num_events."""

    def test_departure_count(self):
        system = _make_system(_queue_sim_cpp.FCFS)
        system.sim(num_events=NUM_EVENTS, seed=42, track_events=True)
        log = system.event_log
        departures = sum(1 for k in log.kinds if k == EventLog.DEPARTURE)
        assert departures == NUM_EVENTS


class TestArrivalCount:
    """Single-server no-buffer: arrivals - departures == final state."""

    def test_arrival_departure_balance(self):
        system = _make_system(_queue_sim_cpp.FCFS)
        system.sim(num_events=NUM_EVENTS, seed=42, track_events=True)
        log = system.event_log
        arrivals = sum(1 for k in log.kinds if k == EventLog.ARRIVAL)
        departures = sum(1 for k in log.kinds if k == EventLog.DEPARTURE)
        assert arrivals >= departures
        assert arrivals - departures == log.states[-1]


class TestStateAlwaysNonNegative:
    """System state is never negative."""

    def test_state_non_negative(self):
        system = _make_system(_queue_sim_cpp.FCFS)
        system.sim(num_events=NUM_EVENTS, seed=42, track_events=True)
        assert all(s >= 0 for s in system.event_log.states)


class TestFromToConsistency:
    """Departures have to==SYSTEM_EXIT, arrivals have from==EXTERNAL."""

    def test_departure_to(self):
        system = _make_system(_queue_sim_cpp.FCFS)
        system.sim(num_events=NUM_EVENTS, seed=42, track_events=True)
        log = system.event_log
        for i, k in enumerate(log.kinds):
            if k == EventLog.DEPARTURE:
                assert log.to_servers[i] == EventLog.SYSTEM_EXIT
            if k == EventLog.ARRIVAL:
                assert log.from_servers[i] == EventLog.EXTERNAL


class TestBufferRejection:
    """With buffer_capacity and high load, REJECTION events appear."""

    def test_rejection_events(self):
        server = _queue_sim_cpp.FCFS(
            _queue_sim_cpp.ExponentialDist(0.5), buffer_capacity=2
        )
        system = _queue_sim_cpp.QueueSystem(
            [server], _queue_sim_cpp.ExponentialDist(1.0)
        )
        system.sim(num_events=NUM_EVENTS, seed=42, track_events=True)
        log = system.event_log
        rejections = sum(1 for k in log.kinds if k == EventLog.REJECTION)
        assert rejections > 0


class TestNetworkRouting:
    """2-server tandem: ROUTE events appear with correct from/to."""

    def test_route_events(self):
        system = _make_tandem()
        system.sim(num_events=NUM_EVENTS, seed=42, track_events=True)
        log = system.event_log
        routes = [(log.from_servers[i], log.to_servers[i])
                  for i, k in enumerate(log.kinds) if k == EventLog.ROUTE]
        assert len(routes) > 0
        for from_s, to_s in routes:
            assert from_s == 0
            assert to_s == 1


class TestDeterminism:
    """Same seed produces identical event logs."""

    def test_same_seed_same_log(self):
        system1 = _make_system(_queue_sim_cpp.FCFS)
        system1.sim(num_events=1000, seed=42, track_events=True)

        system2 = _make_system(_queue_sim_cpp.FCFS)
        system2.sim(num_events=1000, seed=42, track_events=True)

        log1, log2 = system1.event_log, system2.event_log
        assert list(log1.times) == list(log2.times)
        assert list(log1.kinds) == list(log2.kinds)
        assert list(log1.from_servers) == list(log2.from_servers)
        assert list(log1.to_servers) == list(log2.to_servers)
        assert list(log1.states) == list(log2.states)


class TestBackwardCompat:
    """track_events doesn't change E[N], E[T]."""

    def test_results_unchanged(self):
        system1 = _make_system(_queue_sim_cpp.FCFS)
        N1, T1 = system1.sim(num_events=NUM_EVENTS, seed=42)

        system2 = _make_system(_queue_sim_cpp.FCFS)
        N2, T2 = system2.sim(num_events=NUM_EVENTS, seed=42, track_events=True)

        assert N1 == N2
        assert T1 == T2


class TestParallelVectorsSameLength:
    """All 5 vectors have equal length."""

    def test_same_length(self):
        system = _make_system(_queue_sim_cpp.FCFS)
        system.sim(num_events=NUM_EVENTS, seed=42, track_events=True)
        log = system.event_log
        n = len(log)
        assert len(log.times) == n
        assert len(log.kinds) == n
        assert len(log.from_servers) == n
        assert len(log.to_servers) == n
        assert len(log.states) == n


class TestAllPolicies:
    """Event logging works with all scheduling policies."""

    @pytest.mark.parametrize("policy_cls", [
        _queue_sim_cpp.FCFS, _queue_sim_cpp.PS,
        _queue_sim_cpp.FB, _queue_sim_cpp.SRPT,
    ])
    def test_policy(self, policy_cls):
        system = _make_system(policy_cls)
        system.sim(num_events=NUM_EVENTS, seed=42, track_events=True)
        log = system.event_log
        assert len(log) > 0
        assert all(k in VALID_KINDS for k in log.kinds)


class TestConstants:
    """Constants are accessible on the C++ EventLog class."""

    def test_kind_constants(self):
        assert EventLog.ARRIVAL == "arrival"
        assert EventLog.DEPARTURE == "departure"
        assert EventLog.ROUTE == "route"
        assert EventLog.REJECTION == "rejection"

    def test_server_constants(self):
        assert EventLog.EXTERNAL == -1
        assert EventLog.SYSTEM_EXIT == -1


class TestPerServerStates:
    """Tests for per_server_states() with C++ EventLog."""

    def test_single_server_matches_system_state(self):
        system = _make_system(_queue_sim_cpp.FCFS)
        system.sim(num_events=NUM_EVENTS, seed=42, track_events=True)
        log = system.event_log
        data = per_server_states(log)
        assert data["server_states"][0] == list(log.states)

    def test_tandem_sum_equals_system_state(self):
        system = _make_tandem()
        system.sim(num_events=NUM_EVENTS, seed=42, track_events=True)
        log = system.event_log
        data = per_server_states(log)
        for i in range(len(log)):
            total = sum(data["server_states"][s][i] for s in range(2))
            assert total == log.states[i]

    def test_all_pops_non_negative(self):
        system = _make_system(_queue_sim_cpp.FCFS)
        system.sim(num_events=NUM_EVENTS, seed=42, track_events=True)
        data = per_server_states(system.event_log)
        for s_states in data["server_states"]:
            assert all(v >= 0 for v in s_states)

    def test_n_servers_inferred(self):
        system = _make_tandem()
        system.sim(num_events=NUM_EVENTS, seed=42, track_events=True)
        data = per_server_states(system.event_log)
        assert len(data["server_states"]) == 2

    def test_empty_log_raises(self):
        system = _make_system(_queue_sim_cpp.FCFS)
        system.sim(num_events=1000, seed=42)
        # C++ backend: event_log exists but is empty
        with pytest.raises(ValueError, match="empty"):
            per_server_states(system.event_log)
