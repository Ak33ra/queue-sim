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


class TestCppParallelReplicate:

    def test_parallel_matches_sequential(self) -> None:
        """n_threads=1 vs n_threads=4 with same seed -> identical results."""
        sys = _make_mm1_cpp()
        r1 = sys.replicate(n_replications=10, num_events=10_000, seed=42, n_threads=1)
        r2 = sys.replicate(n_replications=10, num_events=10_000, seed=42, n_threads=4)
        assert list(r1.raw_T) == list(r2.raw_T)
        assert list(r1.raw_N) == list(r2.raw_N)

    def test_thread_capping(self) -> None:
        """n_threads=16 with n_replications=3 -> doesn't crash."""
        sys = _make_mm1_cpp()
        raw = sys.replicate(n_replications=3, num_events=10_000, seed=42, n_threads=16)
        assert len(raw.raw_T) == 3

    def test_default_threads(self) -> None:
        """n_threads=0 (default) -> works."""
        sys = _make_mm1_cpp()
        raw = sys.replicate(n_replications=5, num_events=10_000, seed=42)
        assert len(raw.raw_T) == 5

    def test_warmup_parallel(self) -> None:
        """Warmup + parallel -> matches sequential."""
        sys = _make_mm1_cpp()
        r1 = sys.replicate(
            n_replications=10, num_events=10_000, seed=42, warmup=1000, n_threads=1,
        )
        r2 = sys.replicate(
            n_replications=10, num_events=10_000, seed=42, warmup=1000, n_threads=4,
        )
        assert list(r1.raw_T) == list(r2.raw_T)

    def test_tandem_parallel(self) -> None:
        """Tandem queue (multi-server) -> parallel matches sequential."""
        s1 = _queue_sim_cpp.FCFS(_queue_sim_cpp.ExponentialDist(3.0))
        s2 = _queue_sim_cpp.FCFS(_queue_sim_cpp.ExponentialDist(3.0))
        tm = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        sys = _queue_sim_cpp.QueueSystem(
            [s1, s2], _queue_sim_cpp.ExponentialDist(1.0), tm,
        )
        r1 = sys.replicate(n_replications=10, num_events=10_000, seed=42, n_threads=1)
        r2 = sys.replicate(n_replications=10, num_events=10_000, seed=42, n_threads=4)
        assert list(r1.raw_T) == list(r2.raw_T)
        assert list(r1.raw_N) == list(r2.raw_N)

    def test_srpt_parallel(self) -> None:
        """SRPT -> parallel matches sequential (verifies SRPT clone)."""
        server = _queue_sim_cpp.SRPT(_queue_sim_cpp.ExponentialDist(2.0))
        sys = _queue_sim_cpp.QueueSystem(
            [server], _queue_sim_cpp.ExponentialDist(1.0),
        )
        r1 = sys.replicate(n_replications=10, num_events=10_000, seed=42, n_threads=1)
        r2 = sys.replicate(n_replications=10, num_events=10_000, seed=42, n_threads=4)
        assert list(r1.raw_T) == list(r2.raw_T)

    def test_parallel_ci_covers_analytical(self) -> None:
        """Parallel CI still covers analytical M/M/1 E[T]."""
        lam, mu = 1.0, 2.0
        expected_T = 1.0 / (mu - lam)
        sys = _make_mm1_cpp(lam, mu)
        raw = sys.replicate(
            n_replications=30, num_events=200_000, seed=42, n_threads=4,
        )
        result = _build_replication_result(
            tuple(raw.raw_N), tuple(raw.raw_T), 0.95,
        )
        lo, hi = result.ci_T
        assert lo <= expected_T <= hi, (
            f"95% CI [{lo:.4f}, {hi:.4f}] does not contain E[T]={expected_T}"
        )

    def test_ps_parallel(self) -> None:
        """PS -> parallel matches sequential (verifies PS clone)."""
        server = _queue_sim_cpp.PS(_queue_sim_cpp.ExponentialDist(2.0))
        sys = _queue_sim_cpp.QueueSystem(
            [server], _queue_sim_cpp.ExponentialDist(1.0),
        )
        r1 = sys.replicate(n_replications=10, num_events=10_000, seed=42, n_threads=1)
        r2 = sys.replicate(n_replications=10, num_events=10_000, seed=42, n_threads=4)
        assert list(r1.raw_T) == list(r2.raw_T)

    def test_fb_parallel(self) -> None:
        """FB -> parallel matches sequential (verifies FB clone)."""
        server = _queue_sim_cpp.FB(_queue_sim_cpp.ExponentialDist(2.0))
        sys = _queue_sim_cpp.QueueSystem(
            [server], _queue_sim_cpp.ExponentialDist(1.0),
        )
        r1 = sys.replicate(n_replications=10, num_events=10_000, seed=42, n_threads=1)
        r2 = sys.replicate(n_replications=10, num_events=10_000, seed=42, n_threads=4)
        assert list(r1.raw_T) == list(r2.raw_T)

    def test_fcfs_k2_parallel(self) -> None:
        """FCFS k=2 -> parallel matches sequential (verifies multi-server clone)."""
        server = _queue_sim_cpp.FCFS(
            _queue_sim_cpp.ExponentialDist(1.0), num_servers=2
        )
        sys = _queue_sim_cpp.QueueSystem(
            [server], _queue_sim_cpp.ExponentialDist(1.0),
        )
        r1 = sys.replicate(n_replications=10, num_events=10_000, seed=42, n_threads=1)
        r2 = sys.replicate(n_replications=10, num_events=10_000, seed=42, n_threads=4)
        assert list(r1.raw_T) == list(r2.raw_T)
        assert list(r1.raw_N) == list(r2.raw_N)

    def test_ps_k2_parallel(self) -> None:
        """PS k=2 -> parallel matches sequential (verifies multi-server clone)."""
        server = _queue_sim_cpp.PS(
            _queue_sim_cpp.ExponentialDist(1.0), num_servers=2
        )
        sys = _queue_sim_cpp.QueueSystem(
            [server], _queue_sim_cpp.ExponentialDist(1.0),
        )
        r1 = sys.replicate(n_replications=10, num_events=10_000, seed=42, n_threads=1)
        r2 = sys.replicate(n_replications=10, num_events=10_000, seed=42, n_threads=4)
        assert list(r1.raw_T) == list(r2.raw_T)
        assert list(r1.raw_N) == list(r2.raw_N)
