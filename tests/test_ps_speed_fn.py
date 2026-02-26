"""Tests for the configurable speed_fn parameter on PS policy.

Covers both the Python and C++ backends.
"""

import pytest

from queue_sim import PS, QueueSystem, genExp

_queue_sim_cpp = pytest.importorskip("_queue_sim_cpp")

NUM_EVENTS = 500_000
RTOL = 0.05  # 5% relative tolerance


# ---------------------------------------------------------------------------
# Python backend tests
# ---------------------------------------------------------------------------


class TestPSSpeedFnPython:
    """PS speed_fn tests using the pure-Python backend."""

    def test_mm_inf_speed_fn(self) -> None:
        """speed_fn(n) = n  =>  M/M/inf: each job gets rate 1, E[T] = 1/mu."""
        mu = 2.0
        lam = 5.0  # can exceed mu since infinite servers
        system = QueueSystem(
            [PS(sizefn=genExp(mu), speed_fn=lambda n: n)],
            arrivalfn=genExp(lam),
        )
        _N, T = system.sim(num_events=NUM_EVENTS, seed=42)
        expected_T = 1.0 / mu
        assert T == pytest.approx(expected_T, rel=RTOL), (
            f"M/M/inf: simulated E[T]={T:.4f}, expected={expected_T:.4f}"
        )

    def test_explicit_default_matches_standard(self) -> None:
        """speed_fn(n) = min(2, n)  should match num_servers=2 default."""
        mu, lam = 4.0, 3.0
        k = 2
        sys_default = QueueSystem(
            [PS(sizefn=genExp(mu), num_servers=k)],
            arrivalfn=genExp(lam),
        )
        sys_explicit = QueueSystem(
            [PS(sizefn=genExp(mu), num_servers=k, speed_fn=lambda n: min(2, n))],
            arrivalfn=genExp(lam),
        )
        _N1, T1 = sys_default.sim(num_events=NUM_EVENTS, seed=42)
        _N2, T2 = sys_explicit.sim(num_events=NUM_EVENTS, seed=42)
        assert T2 == pytest.approx(T1, rel=1e-6), (
            f"Explicit default T={T2:.6f} != standard T={T1:.6f}"
        )

    def test_concave_speedup_higher_ET(self) -> None:
        """n^0.7 speedup should produce higher E[T] than standard k=2 PS."""
        mu, lam = 4.0, 3.0
        k = 2
        sys_standard = QueueSystem(
            [PS(sizefn=genExp(mu), num_servers=k)],
            arrivalfn=genExp(lam),
        )
        sys_concave = QueueSystem(
            [PS(sizefn=genExp(mu), num_servers=k, speed_fn=lambda n: n**0.7)],
            arrivalfn=genExp(lam),
        )
        _N1, T_std = sys_standard.sim(num_events=NUM_EVENTS, seed=42)
        _N2, T_conc = sys_concave.sim(num_events=NUM_EVENTS, seed=42)
        assert T_conc > T_std, (
            f"Concave E[T]={T_conc:.4f} should exceed standard E[T]={T_std:.4f}"
        )

    def test_speed_fn_overrides_num_servers(self) -> None:
        """speed_fn(n) = 1  with num_servers=5 should behave like M/M/1-PS."""
        mu, lam = 2.0, 1.0
        expected_T = 1.0 / (mu - lam)  # M/M/1: 1/(mu-lam)
        system = QueueSystem(
            [PS(sizefn=genExp(mu), num_servers=5, speed_fn=lambda n: 1)],
            arrivalfn=genExp(lam),
        )
        _N, T = system.sim(num_events=NUM_EVENTS, seed=42)
        assert T == pytest.approx(expected_T, rel=RTOL), (
            f"Override: simulated E[T]={T:.4f}, expected M/M/1 E[T]={expected_T:.4f}"
        )

    def test_speed_fn_never_called_with_zero(self) -> None:
        """speed_fn should never be invoked with n=0."""
        called_with = []

        def guarded_speed(n: int) -> float:
            assert n > 0, f"speed_fn called with n={n}"
            called_with.append(n)
            return min(1, n)

        system = QueueSystem(
            [PS(sizefn=genExp(2.0), speed_fn=guarded_speed)],
            arrivalfn=genExp(1.0),
        )
        system.sim(num_events=10_000, seed=42)
        assert len(called_with) > 0, "speed_fn was never called"


# ---------------------------------------------------------------------------
# C++ backend tests
# ---------------------------------------------------------------------------


class TestPSSpeedFnCpp:
    """PS speed_fn tests using the C++ backend."""

    def test_mm_inf_speed_fn(self) -> None:
        """speed_fn(n) = n  =>  M/M/inf: each job gets rate 1, E[T] = 1/mu."""
        mu = 2.0
        lam = 5.0
        server = _queue_sim_cpp.PS(
            _queue_sim_cpp.ExponentialDist(mu), speed_fn=lambda n: n
        )
        system = _queue_sim_cpp.QueueSystem(
            [server], _queue_sim_cpp.ExponentialDist(lam)
        )
        _N, T = system.sim(num_events=NUM_EVENTS, seed=42)
        expected_T = 1.0 / mu
        assert T == pytest.approx(expected_T, rel=RTOL), (
            f"M/M/inf C++: simulated E[T]={T:.4f}, expected={expected_T:.4f}"
        )

    def test_explicit_default_matches_standard(self) -> None:
        """speed_fn(n) = min(2, n)  should match num_servers=2 default."""
        mu, lam = 4.0, 3.0
        k = 2
        server1 = _queue_sim_cpp.PS(
            _queue_sim_cpp.ExponentialDist(mu), num_servers=k
        )
        sys1 = _queue_sim_cpp.QueueSystem(
            [server1], _queue_sim_cpp.ExponentialDist(lam)
        )
        server2 = _queue_sim_cpp.PS(
            _queue_sim_cpp.ExponentialDist(mu),
            num_servers=k,
            speed_fn=lambda n: min(2, n),
        )
        sys2 = _queue_sim_cpp.QueueSystem(
            [server2], _queue_sim_cpp.ExponentialDist(lam)
        )
        _N1, T1 = sys1.sim(num_events=NUM_EVENTS, seed=42)
        _N2, T2 = sys2.sim(num_events=NUM_EVENTS, seed=42)
        assert T2 == pytest.approx(T1, rel=1e-6), (
            f"C++ explicit default T={T2:.6f} != standard T={T1:.6f}"
        )

    def test_concave_speedup_higher_ET(self) -> None:
        """n^0.7 speedup should produce higher E[T] than standard k=2 PS."""
        mu, lam = 4.0, 3.0
        k = 2
        server1 = _queue_sim_cpp.PS(
            _queue_sim_cpp.ExponentialDist(mu), num_servers=k
        )
        sys1 = _queue_sim_cpp.QueueSystem(
            [server1], _queue_sim_cpp.ExponentialDist(lam)
        )
        server2 = _queue_sim_cpp.PS(
            _queue_sim_cpp.ExponentialDist(mu),
            num_servers=k,
            speed_fn=lambda n: n**0.7,
        )
        sys2 = _queue_sim_cpp.QueueSystem(
            [server2], _queue_sim_cpp.ExponentialDist(lam)
        )
        _N1, T1 = sys1.sim(num_events=NUM_EVENTS, seed=42)
        _N2, T2 = sys2.sim(num_events=NUM_EVENTS, seed=42)
        assert T2 > T1, (
            f"C++ concave E[T]={T2:.4f} should exceed standard E[T]={T1:.4f}"
        )

    def test_speed_fn_overrides_num_servers(self) -> None:
        """speed_fn(n) = 1  with num_servers=5 should behave like M/M/1-PS."""
        mu, lam = 2.0, 1.0
        expected_T = 1.0 / (mu - lam)
        server = _queue_sim_cpp.PS(
            _queue_sim_cpp.ExponentialDist(mu),
            num_servers=5,
            speed_fn=lambda n: 1,
        )
        system = _queue_sim_cpp.QueueSystem(
            [server], _queue_sim_cpp.ExponentialDist(lam)
        )
        _N, T = system.sim(num_events=NUM_EVENTS, seed=42)
        assert T == pytest.approx(expected_T, rel=RTOL), (
            f"C++ override: E[T]={T:.4f}, expected={expected_T:.4f}"
        )
