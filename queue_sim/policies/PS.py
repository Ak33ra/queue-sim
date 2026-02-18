"""Processor Sharing (PS) scheduling policy.

All jobs in service share the server(s) equally. With k servers and n jobs:
- n <= k: each job gets rate 1 (dedicated server)
- n > k:  each job gets rate k/n

For k=1: rate = 1/n, identical to standard M/G/1-PS.
"""

import math
from typing import Callable

from ..server import Server


class PS(Server):

    def __init__(self, sizefn: Callable[[], float], num_servers: int = 1) -> None:
        super().__init__(sizefn, num_servers)
        self.remaining: list[float] = []
        self.jobArrivals: list[float] = []

    def reset(self) -> None:
        super().reset()
        self.remaining = []
        self.jobArrivals = []

    def nextJob(self) -> float:
        return self.genSize()

    def updateET(self) -> None:
        # PS computes response times directly in update(); no-op here.
        return

    def arrival(self) -> None:
        self.remaining.append(self.genSize())
        self.jobArrivals.append(self.clock)
        self.state += 1
        self._recalc_ttnc()

    def update(self, time_elapsed: float) -> bool:
        self.TTNC -= time_elapsed
        self.clock += time_elapsed
        if self.state == 0:
            return False

        work = time_elapsed * min(self.num_servers, self.state) / self.state
        for i in range(len(self.remaining)):
            self.remaining[i] -= work

        if self.TTNC <= 0.0:
            idx = min(range(len(self.remaining)), key=lambda i: self.remaining[i])
            response_time = self.clock - self.jobArrivals[idx]
            del self.remaining[idx]
            del self.jobArrivals[idx]
            self.state -= 1
            self.num_completions += 1
            n = self.num_completions
            self.T = self.T * (n - 1) / n + response_time / n
            self._recalc_ttnc()
            return True
        return False

    def _recalc_ttnc(self) -> None:
        if not self.remaining:
            self.TTNC = math.inf
            return
        min_rem = min(self.remaining)
        self.TTNC = min_rem * self.state / min(self.num_servers, self.state)


__all__ = ['PS']
