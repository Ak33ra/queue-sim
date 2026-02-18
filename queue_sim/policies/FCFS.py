"""First Come First Served (FCFS) scheduling policy.

Jobs are served in arrival order. Supports multiple parallel servers
(G/G/k): up to k jobs served simultaneously, rest wait in FIFO queue.
For k=1, delegates to the base class for bit-for-bit backward compat.
"""

import math
from collections import deque
from typing import Callable

from ..server import Server


class FCFS(Server):

    def __init__(
        self,
        sizefn: Callable[[], float],
        num_servers: int = 1,
        buffer_capacity: int | None = None,
    ) -> None:
        super().__init__(sizefn, num_servers, buffer_capacity)
        self.channelRemaining: list[float] = []
        self.channelArrivals: list[float] = []
        self.waitQueue: deque[float] = deque()

    def reset(self) -> None:
        super().reset()
        self.channelRemaining = []
        self.channelArrivals = []
        self.waitQueue.clear()

    def nextJob(self) -> float:
        return self.genSize()

    def updateET(self) -> None:
        if self.num_servers == 1:
            super().updateET()
            return
        # For k>1, jobs depart out of arrival order — no-op.
        # Response time is computed directly in update().

    def arrival(self) -> None:
        if self.num_servers == 1:
            super().arrival()
            return
        self.state += 1
        if len(self.channelRemaining) < self.num_servers:
            self.channelRemaining.append(self.genSize())
            self.channelArrivals.append(self.clock)
            self._recalc_ttnc()
        else:
            self.waitQueue.append(self.clock)

    def update(self, time_elapsed: float) -> bool:
        if self.num_servers == 1:
            return super().update(time_elapsed)

        self.clock += time_elapsed
        for i in range(len(self.channelRemaining)):
            self.channelRemaining[i] -= time_elapsed
        self.TTNC -= time_elapsed

        if self.TTNC <= 0.0:
            idx = min(
                range(len(self.channelRemaining)),
                key=lambda i: self.channelRemaining[i],
            )
            response_time = self.clock - self.channelArrivals[idx]
            self.num_completions += 1
            n = self.num_completions
            self.T = self.T * (n - 1) / n + response_time / n

            del self.channelRemaining[idx]
            del self.channelArrivals[idx]
            self.state -= 1

            if self.waitQueue:
                arr_time = self.waitQueue.popleft()
                self.channelRemaining.append(self.genSize())
                self.channelArrivals.append(arr_time)
                self.state += 1

            self._recalc_ttnc()
            return True
        return False

    def _recalc_ttnc(self) -> None:
        if not self.channelRemaining:
            self.TTNC = math.inf
            return
        self.TTNC = min(self.channelRemaining)


__all__ = ['FCFS']
