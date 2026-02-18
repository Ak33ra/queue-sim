"""Tests for the C++ backend replicate() method."""

import pytest

_queue_sim_cpp = pytest.importorskip("_queue_sim_cpp")

from queue_sim.results import _build_replication_result  # noqa: E402


def _make_mm1_cpp(lam: float = 1.0, mu: float = 2.0):
    server = _queue_sim_cpp.FCFS(_queue_sim_cpp.ExponentialDist(mu))
    return _queue_sim_cpp.QueueSystem([server], _queue_sim_cpp.ExponentialDist(lam))


class TestCppReplicateRaw:

    def test_returns_raw_result(self) -> None:
        sys = _make_mm1_cpp()
        raw = sys.replicate(n_replications=5, num_events=10_000, seed=42)
        assert isinstance(raw, _queue_sim_cpp.ReplicationRawResult)

    def test_correct_lengths(self) -> None:
        sys = _make_mm1_cpp()
        raw = sys.replicate(n_replications=10, num_events=10_000, seed=42)
        assert len(raw.raw_N) == 10
        assert len(raw.raw_T) == 10

    def test_seed_determinism(self) -> None:
        sys = _make_mm1_cpp()
        r1 = sys.replicate(n_replications=5, num_events=10_000, seed=42)
        r2 = sys.replicate(n_replications=5, num_events=10_000, seed=42)
        assert r1.raw_T == r2.raw_T

    def test_different_seed(self) -> None:
        sys = _make_mm1_cpp()
        r1 = sys.replicate(n_replications=5, num_events=50_000, seed=42)
        r2 = sys.replicate(n_replications=5, num_events=50_000, seed=99)
        assert r1.raw_T != r2.raw_T


class TestCppReplicateWrapped:

    def test_build_replication_result(self) -> None:
        sys = _make_mm1_cpp()
        raw = sys.replicate(n_replications=10, num_events=50_000, seed=42)
        result = _build_replication_result(
            tuple(raw.raw_N), tuple(raw.raw_T), 0.95,
        )
        assert result.n_replications == 10
        assert result.confidence_level == 0.95
        lo, hi = result.ci_T
        assert lo < result.mean_T < hi

    def test_ci_covers_analytical_ET(self) -> None:
        lam, mu = 1.0, 2.0
        expected_T = 1.0 / (mu - lam)
        sys = _make_mm1_cpp(lam, mu)
        raw = sys.replicate(n_replications=30, num_events=200_000, seed=42)
        result = _build_replication_result(
            tuple(raw.raw_N), tuple(raw.raw_T), 0.95,
        )
        lo, hi = result.ci_T
        assert lo <= expected_T <= hi, (
            f"95% CI [{lo:.4f}, {hi:.4f}] does not contain E[T]={expected_T}"
        )


class TestCppWarmup:

    def test_warmup_runs(self) -> None:
        sys = _make_mm1_cpp()
        raw = sys.replicate(n_replications=5, num_events=10_000, seed=42, warmup=1000)
        assert len(raw.raw_T) == 5

    def test_sim_warmup(self) -> None:
        sys = _make_mm1_cpp()
        n1, t1 = sys.sim(num_events=10_000, seed=42, warmup=0)
        n2, t2 = sys.sim(num_events=10_000, seed=42, warmup=5000)
        assert t1 != t2
