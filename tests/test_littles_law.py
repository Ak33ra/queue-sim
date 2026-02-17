"""Property-based test: Little's Law must hold for any stable open system.

Little's Law: E[N] = lambda * E[T]

We use Hypothesis to generate random (but stable) system configurations
and verify that the simulated E[N] and E[T] satisfy this identity.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from queue_sim import FCFS, SRPT, QueueSystem, genExp


@settings(max_examples=10, deadline=None)
@given(
    lam=st.floats(min_value=0.5, max_value=5.0),
    mu=st.floats(min_value=6.0, max_value=20.0),
    seed=st.integers(min_value=0, max_value=2**31),
)
def test_littles_law_fcfs(lam: float, mu: float, seed: int) -> None:
    """E[N] / E[T] should approximate lambda for a stable M/M/1 FCFS queue."""
    system = QueueSystem([FCFS(sizefn=genExp(mu))], arrivalfn=genExp(lam))
    N, T = system.sim(num_events=100_000, seed=seed)
    assert T > 0, "E[T] must be positive for a stable system"
    observed_lam = N / T
    assert observed_lam == pytest.approx(lam, rel=0.10), (
        f"Little's Law violated: E[N]/E[T] = {observed_lam:.4f}, lambda = {lam:.4f}"
    )


@settings(max_examples=10, deadline=None)
@given(
    lam=st.floats(min_value=0.5, max_value=5.0),
    mu=st.floats(min_value=6.0, max_value=20.0),
    seed=st.integers(min_value=0, max_value=2**31),
)
def test_littles_law_srpt(lam: float, mu: float, seed: int) -> None:
    """E[N] / E[T] should approximate lambda for a stable M/M/1 SRPT queue."""
    system = QueueSystem([SRPT(sizefn=genExp(mu))], arrivalfn=genExp(lam))
    N, T = system.sim(num_events=100_000, seed=seed)
    assert T > 0, "E[T] must be positive for a stable system"
    observed_lam = N / T
    assert observed_lam == pytest.approx(lam, rel=0.10), (
        f"Little's Law violated: E[N]/E[T] = {observed_lam:.4f}, lambda = {lam:.4f}"
    )
