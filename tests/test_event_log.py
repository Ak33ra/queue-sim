"""Tests for event log tracking (Python backend)."""

import pytest

from queue_sim import FB, FCFS, PS, SRPT, EventLog, QueueSystem, genExp, per_server_states

NUM_EVENTS = 10_000
VALID_KINDS = {EventLog.ARRIVAL, EventLog.DEPARTURE, EventLog.ROUTE, EventLog.REJECTION}


class TestEventLogLength:
    """Events are logged when track_events=True."""

    def test_events_logged(self):
        server = FCFS(sizefn=genExp(2.0))
        system = QueueSystem([server], arrivalfn=genExp(1.0))
        system.sim(num_events=NUM_EVENTS, seed=42, track_events=True)
        assert len(system.event_log) > 0


class TestNoAttributeByDefault:
    """event_log not created when track_events=False."""

    def test_no_attribute(self):
        server = FCFS(sizefn=genExp(2.0))
        system = QueueSystem([server], arrivalfn=genExp(1.0))
        system.sim(num_events=1000, seed=42)
        assert not hasattr(system, "event_log")


class TestAllKindsValid:
    """Every logged kind is one of the 4 constants."""

    def test_kinds_valid(self):
        server = FCFS(sizefn=genExp(2.0))
        system = QueueSystem([server], arrivalfn=genExp(1.0))
        system.sim(num_events=NUM_EVENTS, seed=42, track_events=True)
        assert all(k in VALID_KINDS for k in system.event_log.kinds)


class TestTimesNonDecreasing:
    """Event times must be monotonically non-decreasing."""

    def test_non_decreasing(self):
        server = FCFS(sizefn=genExp(2.0))
        system = QueueSystem([server], arrivalfn=genExp(1.0))
        system.sim(num_events=NUM_EVENTS, seed=42, track_events=True)
        times = system.event_log.times
        assert all(a <= b for a, b in zip(times, times[1:]))


class TestDepartureCount:
    """Single-server no-buffer: departures == num_events."""

    def test_departure_count(self):
        server = FCFS(sizefn=genExp(2.0))
        system = QueueSystem([server], arrivalfn=genExp(1.0))
        system.sim(num_events=NUM_EVENTS, seed=42, track_events=True)
        log = system.event_log
        departures = sum(1 for k in log.kinds if k == EventLog.DEPARTURE)
        assert departures == NUM_EVENTS


class TestArrivalCount:
    """Single-server no-buffer: arrivals - departures == final state."""

    def test_arrival_departure_balance(self):
        server = FCFS(sizefn=genExp(2.0))
        system = QueueSystem([server], arrivalfn=genExp(1.0))
        system.sim(num_events=NUM_EVENTS, seed=42, track_events=True)
        log = system.event_log
        arrivals = sum(1 for k in log.kinds if k == EventLog.ARRIVAL)
        departures = sum(1 for k in log.kinds if k == EventLog.DEPARTURE)
        assert arrivals >= departures
        assert arrivals - departures == log.states[-1]


class TestStateAlwaysNonNegative:
    """System state is never negative."""

    def test_state_non_negative(self):
        server = FCFS(sizefn=genExp(2.0))
        system = QueueSystem([server], arrivalfn=genExp(1.0))
        system.sim(num_events=NUM_EVENTS, seed=42, track_events=True)
        assert all(s >= 0 for s in system.event_log.states)


class TestFromToConsistency:
    """Departures have to==SYSTEM_EXIT, arrivals have from==EXTERNAL."""

    def test_departure_to(self):
        server = FCFS(sizefn=genExp(2.0))
        system = QueueSystem([server], arrivalfn=genExp(1.0))
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
        server = FCFS(sizefn=genExp(0.5), buffer_capacity=2)
        system = QueueSystem([server], arrivalfn=genExp(1.0))
        system.sim(num_events=NUM_EVENTS, seed=42, track_events=True)
        log = system.event_log
        rejections = sum(1 for k in log.kinds if k == EventLog.REJECTION)
        assert rejections > 0


class TestNetworkRouting:
    """2-server tandem: ROUTE events appear with correct from/to."""

    def test_route_events(self):
        s0 = FCFS(sizefn=genExp(3.0))
        s1 = FCFS(sizefn=genExp(3.0))
        system = QueueSystem([s0, s1], arrivalfn=genExp(1.0))
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
        for _ in range(2):
            server = FCFS(sizefn=genExp(2.0))
            system = QueueSystem([server], arrivalfn=genExp(1.0))
            system.sim(num_events=1000, seed=42, track_events=True)
            if _ == 0:
                log1 = system.event_log
            else:
                log2 = system.event_log

        assert log1.times == log2.times
        assert log1.kinds == log2.kinds
        assert log1.from_servers == log2.from_servers
        assert log1.to_servers == log2.to_servers
        assert log1.states == log2.states


class TestBackwardCompat:
    """track_events doesn't change E[N], E[T]."""

    def test_results_unchanged(self):
        server1 = FCFS(sizefn=genExp(2.0))
        system1 = QueueSystem([server1], arrivalfn=genExp(1.0))
        N1, T1 = system1.sim(num_events=NUM_EVENTS, seed=42)

        server2 = FCFS(sizefn=genExp(2.0))
        system2 = QueueSystem([server2], arrivalfn=genExp(1.0))
        N2, T2 = system2.sim(num_events=NUM_EVENTS, seed=42, track_events=True)

        assert N1 == N2
        assert T1 == T2


class TestParallelVectorsSameLength:
    """All 5 vectors have equal length."""

    def test_same_length(self):
        server = FCFS(sizefn=genExp(2.0))
        system = QueueSystem([server], arrivalfn=genExp(1.0))
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

    @pytest.mark.parametrize("policy_cls", [FCFS, PS, FB, SRPT])
    def test_policy(self, policy_cls):
        server = policy_cls(sizefn=genExp(2.0))
        system = QueueSystem([server], arrivalfn=genExp(1.0))
        system.sim(num_events=NUM_EVENTS, seed=42, track_events=True)
        log = system.event_log
        assert len(log) > 0
        assert all(k in VALID_KINDS for k in log.kinds)


class TestConstants:
    """Constants are accessible on the class."""

    def test_kind_constants(self):
        assert EventLog.ARRIVAL == "arrival"
        assert EventLog.DEPARTURE == "departure"
        assert EventLog.ROUTE == "route"
        assert EventLog.REJECTION == "rejection"

    def test_server_constants(self):
        assert EventLog.EXTERNAL == -1
        assert EventLog.SYSTEM_EXIT == -1


class TestPerServerStates:
    """Tests for per_server_states() reconstruction utility."""

    def test_single_server_matches_system_state(self):
        """For a single server, server_states[0] should equal log.states."""
        server = FCFS(sizefn=genExp(2.0))
        system = QueueSystem([server], arrivalfn=genExp(1.0))
        system.sim(num_events=NUM_EVENTS, seed=42, track_events=True)
        log = system.event_log
        data = per_server_states(log)
        assert data["server_states"][0] == list(log.states)

    def test_tandem_sum_equals_system_state(self):
        """For a tandem network, sum of per-server pops == system state."""
        s0 = FCFS(sizefn=genExp(3.0))
        s1 = FCFS(sizefn=genExp(3.0))
        system = QueueSystem([s0, s1], arrivalfn=genExp(1.0))
        system.sim(num_events=NUM_EVENTS, seed=42, track_events=True)
        log = system.event_log
        data = per_server_states(log)
        for i in range(len(log)):
            total = sum(data["server_states"][s][i] for s in range(2))
            assert total == log.states[i]

    def test_all_pops_non_negative(self):
        """Per-server populations should never go negative."""
        server = FCFS(sizefn=genExp(2.0))
        system = QueueSystem([server], arrivalfn=genExp(1.0))
        system.sim(num_events=NUM_EVENTS, seed=42, track_events=True)
        data = per_server_states(system.event_log)
        for s_states in data["server_states"]:
            assert all(v >= 0 for v in s_states)

    def test_pops_non_negative_with_buffer(self):
        """Populations stay non-negative even with buffer rejections."""
        s0 = FCFS(sizefn=genExp(0.5), buffer_capacity=2)
        s1 = FCFS(sizefn=genExp(3.0), buffer_capacity=3)
        system = QueueSystem([s0, s1], arrivalfn=genExp(1.0))
        system.sim(num_events=NUM_EVENTS, seed=42, track_events=True)
        data = per_server_states(system.event_log)
        for s_states in data["server_states"]:
            assert all(v >= 0 for v in s_states)

    def test_n_servers_inferred(self):
        """n_servers is inferred correctly from a tandem log."""
        s0 = FCFS(sizefn=genExp(3.0))
        s1 = FCFS(sizefn=genExp(3.0))
        system = QueueSystem([s0, s1], arrivalfn=genExp(1.0))
        system.sim(num_events=NUM_EVENTS, seed=42, track_events=True)
        data = per_server_states(system.event_log)
        assert len(data["server_states"]) == 2

    def test_n_servers_override(self):
        """User can override n_servers to a higher value."""
        server = FCFS(sizefn=genExp(2.0))
        system = QueueSystem([server], arrivalfn=genExp(1.0))
        system.sim(num_events=1000, seed=42, track_events=True)
        data = per_server_states(system.event_log, n_servers=3)
        assert len(data["server_states"]) == 3
        # Extra servers should be all zeros
        assert all(v == 0 for v in data["server_states"][1])
        assert all(v == 0 for v in data["server_states"][2])

    def test_times_match_log(self):
        """Returned times should match the log times."""
        server = FCFS(sizefn=genExp(2.0))
        system = QueueSystem([server], arrivalfn=genExp(1.0))
        system.sim(num_events=1000, seed=42, track_events=True)
        data = per_server_states(system.event_log)
        assert data["times"] == list(system.event_log.times)

    def test_empty_log_raises(self):
        """Empty log should raise ValueError."""
        log = EventLog()
        with pytest.raises(ValueError, match="empty"):
            per_server_states(log)
