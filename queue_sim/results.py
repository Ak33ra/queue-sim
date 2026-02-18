"""Replication result container and statistical helpers.

Provides the ``ReplicationResult`` dataclass returned by
``QueueSystem.replicate()`` along with internal helpers for seed
derivation, t-distribution quantiles, and confidence interval
computation.  No external dependencies beyond the standard library.
"""

from __future__ import annotations

import dataclasses
import math

# -- seed derivation (SplitMix64) -------------------------------------------

_PHI = 0x9E3779B97F4A7C15  # golden-ratio constant
_MASK64 = (1 << 64) - 1


def _splitmix64(x: int) -> int:
    """One round of SplitMix64 (Steele / Vigna)."""
    x = (x + _PHI) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (x ^ (x >> 31)) & _MASK64


def _derive_seed(base_seed: int, index: int) -> int:
    """Deterministic per-replication seed from *base_seed* and *index*."""
    return _splitmix64((base_seed + index * _PHI) & _MASK64)


# -- t-distribution inverse CDF (Hill 1970) ---------------------------------

def _t_inv_cdf(p: float, df: int) -> float:
    """Return *t* such that P(T <= t) = *p* for Student's t with *df* dof.

    Uses the Hill (1970) rational approximation.  Accurate to ~1e-5
    for all df >= 1 — negligible compared to simulation variance.
    """
    if not (0 < p < 1):
        raise ValueError(f"p must be in (0, 1), got {p}")
    if df < 1:
        raise ValueError(f"df must be >= 1, got {df}")

    # Use symmetry so we only need the upper tail
    if p < 0.5:
        return -_t_inv_cdf(1.0 - p, df)

    # Normal quantile via Abramowitz & Stegun 26.2.23 (rational approx)
    a = math.sqrt(-2.0 * math.log(1.0 - p))
    zp = a - (2.515517 + 0.802853 * a + 0.010328 * a * a) / (
        1.0 + 1.432788 * a + 0.189269 * a * a + 0.001308 * a * a * a
    )

    # Hill's correction from normal to t
    g1 = (zp ** 3 + zp) / 4.0
    g2 = ((5 * zp ** 5 + 16 * zp ** 3 + 3 * zp) / 96.0)
    g3 = ((3 * zp ** 7 + 19 * zp ** 5 + 17 * zp ** 3 - 15 * zp) / 384.0)
    g4 = (
        (79 * zp ** 9 + 776 * zp ** 7 + 1482 * zp ** 5
         - 1920 * zp ** 3 - 945 * zp)
        / 92160.0
    )

    tp = (
        zp
        + g1 / df
        + g2 / (df ** 2)
        + g3 / (df ** 3)
        + g4 / (df ** 4)
    )
    return tp


# -- confidence interval helper ---------------------------------------------

def _ci_half_width(values: tuple[float, ...], confidence: float) -> float:
    """Return the half-width *h* of a *confidence*-level CI for the mean."""
    n = len(values)
    if n < 2:
        raise ValueError("Need at least 2 values for a CI")
    x_bar = math.fsum(values) / n
    s2 = math.fsum((x - x_bar) ** 2 for x in values) / (n - 1)
    s = math.sqrt(s2)
    alpha = 1.0 - confidence
    t_crit = _t_inv_cdf(1.0 - alpha / 2.0, n - 1)
    return t_crit * s / math.sqrt(n)


# -- ReplicationResult -------------------------------------------------------

@dataclasses.dataclass(frozen=True, slots=True)
class ReplicationResult:
    """Aggregated output of multiple independent simulation replications."""

    mean_N: float
    mean_T: float
    ci_half_N: float
    ci_half_T: float
    confidence_level: float
    n_replications: int
    raw_N: tuple[float, ...]
    raw_T: tuple[float, ...]

    @property
    def ci_N(self) -> tuple[float, float]:
        """Confidence interval for E[N] as (lower, upper)."""
        return (self.mean_N - self.ci_half_N, self.mean_N + self.ci_half_N)

    @property
    def ci_T(self) -> tuple[float, float]:
        """Confidence interval for E[T] as (lower, upper)."""
        return (self.mean_T - self.ci_half_T, self.mean_T + self.ci_half_T)


def _build_replication_result(
    raw_N: tuple[float, ...],
    raw_T: tuple[float, ...],
    confidence: float,
) -> ReplicationResult:
    """Construct a :class:`ReplicationResult` from raw per-replication data."""
    n = len(raw_N)
    mean_N = math.fsum(raw_N) / n
    mean_T = math.fsum(raw_T) / n
    return ReplicationResult(
        mean_N=mean_N,
        mean_T=mean_T,
        ci_half_N=_ci_half_width(raw_N, confidence),
        ci_half_T=_ci_half_width(raw_T, confidence),
        confidence_level=confidence,
        n_replications=n,
        raw_N=raw_N,
        raw_T=raw_T,
    )


__all__ = ["ReplicationResult"]
