"""Shortest Remaining Processing Time (SRPT) scheduling policy.

Always serves the job with the lowest remaining processing time,
preempting the current job if a shorter one arrives. Locally optimal
for minimizing mean response time in a single-server queue, but may
make globally suboptimal decisions in networks.
"""

import heapq
import math
from typing import Callable

from ..server import Server


class SRPT(Server):

    def __init__(
        self,
        sizefn: Callable[[], float],
        buffer_capacity: int | None = None,
    ) -> None:
        super().__init__(sizefn, 1, buffer_capacity)
        self.jobs: list[tuple[float, float]] = []  # (remaining, arrival_time)
        self._running_arrival_time: float = 0.0

    def reset(self) -> None:
        super().reset()
        self.jobs = []
        self._running_arrival_time = 0.0

    def nextJob(self) -> float:
        remaining, arr = heapq.heappop(self.jobs)
        self._running_arrival_time = arr
        return remaining

    def updateET(self) -> None:
        t = self.clock - self._running_arrival_time
        self._last_response_time = t
        n = self.num_completions
        self.T = self.T * (n - 1) / n + t / n

    def arrival(self) -> None:
        if self.state > 0:
            heapq.heappush(self.jobs, (self.TTNC, self._running_arrival_time))
        heapq.heappush(self.jobs, (self.genSize(), self.clock))
        remaining, arr = heapq.heappop(self.jobs)
        self.TTNC = remaining
        self._running_arrival_time = arr
        self.state += 1

    # Critical ordering: updateET() BEFORE nextJob()
    def update(self, time_elapsed: float) -> bool:
        self.TTNC -= time_elapsed
        self.clock += time_elapsed
        if self.TTNC <= 0.0:
            self.state -= 1
            self.num_completions += 1
            self.updateET()
            self.TTNC = self.nextJob() if self.state > 0 else math.inf
            return True
        return False


__all__ = ['SRPT']
