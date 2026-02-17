"""Functions to sample from common probability distributions.

Two flavors:

1. **Generator functions** (``gen*``) return a zero-argument callable that
   produces a fresh sample on each call.  Useful when a distribution is
   reused many times (e.g. as a ``sizefn`` for a Server).

   >>> exponential = genExp(1.5)
   >>> sample = exponential()          # float ~ Exp(1.5)

2. **Direct samplers** return a single value immediately.

   >>> sample = Uniform(2, 5)          # float ~ Unif(2, 5)
"""

import math
import random
from typing import Callable

# ---------------------------------------------------------------------------
# Generator functions (return Callable[[], float])
# ---------------------------------------------------------------------------

def genExp(mu: float) -> Callable[[], float]:
    """X ~ Exponential(mu), with E[X] = 1/mu."""
    return lambda: -(1 / mu) * math.log(1 - random.random())


def genUniform(a: float, b: float) -> Callable[[], float]:
    """X ~ Uniform(a, b)."""
    d = b - a
    return lambda: d * random.random() + a


def genBoundedPareto(k: float, p: float, alpha: float) -> Callable[[], float]:
    """X ~ BoundedPareto(k, p, alpha)."""
    C = (k ** alpha) / (1 - (k / p) ** alpha)
    return lambda: (-random.random() / C + k ** (-alpha)) ** (-1 / alpha)


def genBernoulli(p: float) -> Callable[[], int]:
    """X ~ Bernoulli(p), returns 0 or 1."""
    return lambda: 1 if random.random() <= p else 0


# ---------------------------------------------------------------------------
# Direct samplers (return a single value)
# ---------------------------------------------------------------------------

def Uniform(a: float, b: float) -> float:
    """Return a single sample from Uniform(a, b)."""
    d = b - a
    return d * random.random() + a


def BoundedPareto(k: float, p: float, alpha: float) -> float:
    """Return a single sample from BoundedPareto(k, p, alpha)."""
    return genBoundedPareto(k, p, alpha)()


def Bernoulli(p: float) -> int:
    """Return a single sample from Bernoulli(p)."""
    return genBernoulli(p)()
