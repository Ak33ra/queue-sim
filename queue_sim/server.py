"""Base class for scheduling policies.

Implements core server mechanics: processing jobs, tracking time, and
reporting event times back to the system driver (QueueSystem).

To create a custom policy, subclass Server and implement nextJob().
See queue_sim/policies/ for examples.
"""

import math
from abc import ABC, abstractmethod
from collections import deque
from typing import Callable


class Server(ABC):

    def __init__(
        self,
        sizefn: Callable[[], float],
        num_servers: int = 1,
        buffer_capacity: int | None = None,
    ) -> None:
        if buffer_capacity is not None and buffer_capacity < 1:
            raise ValueError("buffer_capacity must be >= 1 or None (unlimited)")
        self.genSize: Callable[[], float] = sizefn
        self.num_servers: int = num_servers
        self.buffer_capacity: int | None = buffer_capacity
        self._init_state()

    def _init_state(self) -> None:
        """Initialize (or reset) all mutable simulation state."""
        self.clock: float = 0.0
        self.arrivalTimes: deque[float] = deque()
        self.TTNC: float = math.inf
        self.T: float = 0.0
        self.num_completions: int = 0
        self.state: int = 0
        self.num_rejected: int = 0
        self.num_arrivals: int = 0

    def is_full(self) -> bool:
        """Return True if the server's buffer is at capacity."""
        return self.buffer_capacity is not None and self.state >= self.buffer_capacity

    def reset(self) -> None:
        """Reset runtime state so the server can be reused across sim() calls."""
        self._init_state()

    @abstractmethod
    def nextJob(self) -> float:
        """Return the service time for the next job to process."""
        ...

    def updateET(self) -> None:
        """Update running mean response time E[T] via incremental average.

        Only valid for FIFO-ordered policies. Policies that reorder jobs
        (e.g. SRPT) should override this.
        """
        t = self.clock - self.arrivalTimes.popleft()
        n = self.num_completions
        self.T = self.T * (n - 1) / n + t / n

    def arrival(self) -> None:
        """Register a new job arrival at this server."""
        self.arrivalTimes.append(self.clock)
        if self.state == 0:
            self.TTNC = self.nextJob()
        self.state += 1

    def queryTTNC(self) -> float:
        """Return time until this server's next completion (inf if idle)."""
        return self.TTNC

    def update(self, time_elapsed: float) -> bool:
        """Advance server clock by time_elapsed.

        Returns True if a job completed during this time step.
        """
        self.TTNC -= time_elapsed
        self.clock += time_elapsed
        if self.TTNC <= 0.0:
            self.state -= 1
            self.TTNC = self.nextJob() if self.state > 0 else math.inf
            self.num_completions += 1
            self.updateET()
            return True
        return False


__all__ = ["Server"]
