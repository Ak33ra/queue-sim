"""Shortest Remaining Processing Time (SRPT) scheduling policy.

Always serves the job with the lowest remaining processing time,
preempting the current job if a shorter one arrives. Locally optimal
for minimizing mean response time in a single-server queue, but may
make globally suboptimal decisions in networks.
"""

import heapq
from typing import Callable

from ..server import Server


class SRPT(Server):

    def __init__(
        self,
        sizefn: Callable[[], float],
        buffer_capacity: int | None = None,
    ) -> None:
        super().__init__(sizefn, 1, buffer_capacity)
        self.jobs: list[float] = []

    def reset(self) -> None:
        super().reset()
        self.jobs = []

    def nextJob(self) -> float:
        return heapq.heappop(self.jobs)

    def updateET(self) -> None:
        # SRPT reorders jobs, so the FIFO-based E[T] tracker is invalid.
        return

    def arrival(self) -> None:
        if self.state > 0:
            heapq.heappush(self.jobs, self.TTNC)
        heapq.heappush(self.jobs, self.genSize())
        self.TTNC = heapq.heappop(self.jobs)
        self.state += 1


__all__ = ['SRPT']
