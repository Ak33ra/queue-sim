"""Tests for the Python-backend replicate() method and statistical helpers."""

import pytest

from queue_sim import FCFS, QueueSystem, ReplicationResult, genExp
from queue_sim.results import _ci_half_width, _derive_seed, _t_inv_cdf

# -- t-distribution inverse CDF ---------------------------------------------

class TestTInvCdf:
    """Verify the Hill (1970) approximation against known table values."""

    def test_t_975_29(self) -> None:
        assert _t_inv_cdf(0.975, 29) == pytest.approx(2.045, abs=0.005)

    def test_t_975_9(self) -> None:
        assert _t_inv_cdf(0.975, 9) == pytest.approx(2.262, abs=0.005)

    def test_t_95_29(self) -> None:
        assert _t_inv_cdf(0.95, 29) == pytest.approx(1.699, abs=0.005)

    def test_symmetry(self) -> None:
        assert _t_inv_cdf(0.025, 29) == pytest.approx(-_t_inv_cdf(0.975, 29))

    def test_invalid_p(self) -> None:
        with pytest.raises(ValueError):
            _t_inv_cdf(0.0, 10)
        with pytest.raises(ValueError):
            _t_inv_cdf(1.0, 10)

    def test_invalid_df(self) -> None:
        with pytest.raises(ValueError):
            _t_inv_cdf(0.975, 0)


# -- seed derivation --------------------------------------------------------

class TestDeriveSeed:

    def test_deterministic(self) -> None:
        assert _derive_seed(42, 0) == _derive_seed(42, 0)

    def test_all_distinct(self) -> None:
        seeds = [_derive_seed(42, i) for i in range(100)]
        assert len(set(seeds)) == 100

    def test_different_base(self) -> None:
        assert _derive_seed(0, 0) != _derive_seed(1, 0)


# -- ci_half_width -----------------------------------------------------------

class TestCIHalfWidth:

    def test_needs_two_values(self) -> None:
        with pytest.raises(ValueError):
            _ci_half_width((1.0,), 0.95)

    def test_positive(self) -> None:
        values = (1.0, 2.0, 3.0, 4.0, 5.0)
        h = _ci_half_width(values, 0.95)
        assert h > 0


# -- replicate() basic behaviour --------------------------------------------

def _make_mm1(lam: float = 1.0, mu: float = 2.0) -> QueueSystem:
    return QueueSystem([FCFS(sizefn=genExp(mu))], arrivalfn=genExp(lam))


class TestReplicateBasics:

    def test_returns_replication_result(self) -> None:
        result = _make_mm1().replicate(n_replications=5, num_events=10_000, seed=42)
        assert isinstance(result, ReplicationResult)

    def test_correct_lengths(self) -> None:
        result = _make_mm1().replicate(n_replications=10, num_events=10_000, seed=42)
        assert len(result.raw_N) == 10
        assert len(result.raw_T) == 10

    def test_stores_confidence_level(self) -> None:
        result = _make_mm1().replicate(
            n_replications=5, num_events=10_000, seed=42, confidence=0.99,
        )
        assert result.confidence_level == 0.99

    def test_n_replications_stored(self) -> None:
        result = _make_mm1().replicate(n_replications=7, num_events=10_000, seed=42)
        assert result.n_replications == 7

    def test_ci_properties(self) -> None:
        result = _make_mm1().replicate(n_replications=5, num_events=10_000, seed=42)
        lo, hi = result.ci_T
        assert lo < result.mean_T < hi
        lo_n, hi_n = result.ci_N
        assert lo_n < result.mean_N < hi_n


# -- seed determinism -------------------------------------------------------

class TestSeedDeterminism:

    def test_same_seed_same_result(self) -> None:
        sys = _make_mm1()
        r1 = sys.replicate(n_replications=5, num_events=10_000, seed=42)
        r2 = sys.replicate(n_replications=5, num_events=10_000, seed=42)
        assert r1.raw_T == r2.raw_T

    def test_different_seed_different_result(self) -> None:
        sys = _make_mm1()
        r1 = sys.replicate(n_replications=5, num_events=50_000, seed=42)
        r2 = sys.replicate(n_replications=5, num_events=50_000, seed=99)
        assert r1.raw_T != r2.raw_T


# -- CI covers analytical truth ---------------------------------------------

class TestCICoversAnalytical:

    def test_mm1_ci_covers_true_ET(self) -> None:
        lam, mu = 1.0, 2.0
        expected_T = 1.0 / (mu - lam)  # = 1.0
        result = _make_mm1(lam, mu).replicate(
            n_replications=30, num_events=200_000, seed=42,
        )
        lo, hi = result.ci_T
        assert lo <= expected_T <= hi, (
            f"95% CI [{lo:.4f}, {hi:.4f}] does not contain E[T]={expected_T}"
        )


# -- CI narrows with more replications --------------------------------------

class TestCINarrows:

    def test_more_reps_narrower_ci(self) -> None:
        sys = _make_mm1()
        r_small = sys.replicate(n_replications=5, num_events=100_000, seed=42)
        r_large = sys.replicate(n_replications=30, num_events=100_000, seed=42)
        assert r_large.ci_half_T < r_small.ci_half_T


# -- validation --------------------------------------------------------------

class TestValidation:

    def test_rejects_too_few_replications(self) -> None:
        with pytest.raises(ValueError, match="n_replications"):
            _make_mm1().replicate(n_replications=1, num_events=1000, seed=42)

    def test_rejects_confidence_zero(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            _make_mm1().replicate(confidence=0.0, num_events=1000, seed=42)

    def test_rejects_confidence_one(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            _make_mm1().replicate(confidence=1.0, num_events=1000, seed=42)


# -- warmup ------------------------------------------------------------------

class TestWarmup:

    def test_runs_without_error(self) -> None:
        result = _make_mm1().replicate(
            n_replications=5, num_events=10_000, seed=42, warmup=1000,
        )
        assert isinstance(result, ReplicationResult)

    def test_warmup_changes_results(self) -> None:
        sys = _make_mm1()
        r_no = sys.replicate(n_replications=5, num_events=10_000, seed=42, warmup=0)
        r_wu = sys.replicate(n_replications=5, num_events=10_000, seed=42, warmup=5000)
        assert r_no.raw_T != r_wu.raw_T
