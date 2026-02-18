"""Foreground-Background (FB) / Least Attained Service scheduling policy.

Always serves the job(s) with the least attained service time.
Ties share the server equally (processor-sharing among tied jobs).
Optimal for minimizing mean response time when job sizes are unknown.
For M/M/1, E[T] = 1/(mu - lambda), same as FCFS and PS.
"""

import math
from typing import Callable

from ..server import Server


class FB(Server):

    def __init__(
        self,
        sizefn: Callable[[], float],
        buffer_capacity: int | None = None,
    ) -> None:
        super().__init__(sizefn, 1, buffer_capacity)
        # Each job: [remaining, attained, arrival_time]
        self.jobs: list[list[float]] = []

    def reset(self) -> None:
        super().reset()
        self.jobs = []

    def nextJob(self) -> float:
        return self.genSize()

    def updateET(self) -> None:
        # FB computes response times directly in update(); no-op here.
        return

    def arrival(self) -> None:
        self.jobs.append([self.genSize(), 0.0, self.clock])
        self.state += 1
        self._recalc_ttnc()

    def update(self, time_elapsed: float) -> bool:
        self.TTNC -= time_elapsed
        self.clock += time_elapsed
        if not self.jobs:
            return False

        # Find active set (minimum attained service)
        min_att = min(j[1] for j in self.jobs)
        active = [i for i, j in enumerate(self.jobs) if j[1] <= min_att + 1e-12]
        num_active = len(active)

        work = time_elapsed / num_active
        for i in active:
            self.jobs[i][0] -= work   # remaining
            self.jobs[i][1] += work   # attained

        if self.TTNC <= 0.0:
            # Check for completion (remaining ≈ 0)
            for i, j in enumerate(self.jobs):
                if j[0] <= 1e-12:
                    response_time = self.clock - j[2]
                    del self.jobs[i]
                    self.state -= 1
                    self.num_completions += 1
                    n = self.num_completions
                    self.T = self.T * (n - 1) / n + response_time / n
                    self._recalc_ttnc()
                    return True
            # Level crossing — active set expanded, recalculate
            self._recalc_ttnc()
        return False

    def _recalc_ttnc(self) -> None:
        if not self.jobs:
            self.TTNC = math.inf
            return

        min_att = min(j[1] for j in self.jobs)
        min_rem_active = math.inf
        next_level = math.inf
        num_active = 0

        for j in self.jobs:
            if j[1] <= min_att + 1e-12:
                num_active += 1
                min_rem_active = min(min_rem_active, j[0])
            else:
                next_level = min(next_level, j[1])

        time_to_completion = min_rem_active * num_active
        time_to_crossing = (next_level - min_att) * num_active
        self.TTNC = min(time_to_completion, time_to_crossing)


__all__ = ['FB']
