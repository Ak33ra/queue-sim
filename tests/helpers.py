"""Shared analytical helpers for queueing tests."""

from math import factorial


def erlang_b(c: int, a: float) -> float:
    """Erlang-B formula: blocking probability for M/M/c/c (loss system).

    Uses the numerically stable recursive form (Jagerman's algorithm).

    Args:
        c: number of servers (= system capacity)
        a: offered load (lambda / mu)

    Returns:
        P(loss) — the probability an arriving customer is rejected.
    """
    b = 1.0
    for n in range(1, c + 1):
        b = (a * b) / (n + a * b)
    return b


def mm1k_ploss(rho: float, K: int) -> float:
    """Loss probability for M/M/1/K queue.

    Args:
        rho: traffic intensity (lambda / mu)
        K:   total system capacity (in-service + waiting)

    Returns:
        P(loss) — fraction of arrivals rejected.
    """
    if abs(rho - 1.0) < 1e-10:
        return 1.0 / (K + 1)
    return (1 - rho) * rho**K / (1 - rho ** (K + 1))


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
