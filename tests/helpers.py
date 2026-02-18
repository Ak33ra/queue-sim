"""Shared analytical helpers for queueing tests."""

from math import factorial


def erlang_c(k: int, a: float) -> float:
    """Erlang-C formula: probability an arriving customer must wait in M/M/k.

    Args:
        k: number of servers
        a: offered load (lambda / mu)

    Returns:
        P(wait) — the probability a customer must queue.
    """
    rho = a / k
    num = a**k / (factorial(k) * (1 - rho))
    denom = sum(a**n / factorial(n) for n in range(k)) + num
    return num / denom


def mmk_expected_T(lam: float, mu: float, k: int) -> float:
    """Expected mean response time E[T] for an M/M/k queue.

    E[T] = 1/mu + C(k, a) / (k*mu - lambda)
    """
    a = lam / mu
    return 1 / mu + erlang_c(k, a) / (k * mu - lam)
