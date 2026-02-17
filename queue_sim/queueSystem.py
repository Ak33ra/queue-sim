"""System driver that coordinates servers in a queueing network.

Simulates a sequence of arrivals and collects performance statistics
(mean number in system E[N], mean response time E[T]).

The execution is event-driven: servers report the time until their next
completion back to the system, which advances a global clock to the
nearest event. This allows the simulation to skip idle time rather than
stepping in real-time increments.
"""

import random
from typing import Callable

from .server import Server


class QueueSystem:

    def __init__(
        self,
        servers: list[Server],
        arrivalfn: Callable[[], float],
        transitionMatrix: list[list[float]] | None = None,
    ) -> None:
        self.servers = servers
        self.genArrival = arrivalfn
        self.transitionMatrix = transitionMatrix or []
        self.T: float = 0.0

    # -- configuration helpers ------------------------------------------------

    def addServer(self, server: Server) -> None:
        self.servers.append(server)

    def updateTransitionMatrix(self, M: list[list[float]]) -> None:
        self.transitionMatrix = M

    # -- internal helpers -----------------------------------------------------

    def _verify_transition_matrix(self) -> None:
        if not self.transitionMatrix:
            return
        n_servers = len(self.servers)
        n_rows = len(self.transitionMatrix)
        if n_rows != n_servers or any(
            len(row) != n_servers + 1 for row in self.transitionMatrix
        ):
            raise ValueError(
                f"Transition matrix must be {n_servers} x {n_servers + 1} "
                f"(one row per server, last column = exit probability). "
                f"Got {n_rows} x {len(self.transitionMatrix[0])}."
            )
        for i, row in enumerate(self.transitionMatrix):
            if abs(sum(row) - 1.0) > 1e-9:
                raise ValueError(
                    f"Transition matrix row {i} sums to {sum(row)}, expected 1.0"
                )

    def _min_ttnc(self) -> float:
        """Return the minimum time-to-next-completion across all servers."""
        return min(s.queryTTNC() for s in self.servers)

    def _route_job(self, server_idx: int) -> int:
        """Return the index of the next server for a completed job.

        If the returned index == len(self.servers), the job exits the system.
        For deterministic (tandem) routing, jobs proceed to server_idx + 1.
        """
        if not self.transitionMatrix:
            return server_idx + 1

        u = random.random()
        acc = 0.0
        for i, p in enumerate(self.transitionMatrix[server_idx]):
            acc += p
            if u < acc:
                return i
        # Numerical safety: if we fall through, treat as exit
        return len(self.servers)

    # -- main simulation loop -------------------------------------------------

    def sim(
        self,
        num_events: int = 10**6,
        seed: int | None = None,
    ) -> tuple[float, float]:
        """Run the simulation.

        Args:
            num_events: Number of job completions (departures from the system)
                        before stopping.
            seed:       Optional RNG seed for reproducibility.

        Returns:
            (E[N], E[T]): mean number in system and mean response time.
        """
        if seed is not None:
            random.seed(seed)

        self._verify_transition_matrix()
        for server in self.servers:
            server.reset()

        num_completions = 0
        ttna = self.genArrival()       # time to next arrival
        area_n: float = 0.0
        state = 0                      # total jobs in the network
        clock: float = 0.0

        while num_completions < num_events:
            ttnc = self._min_ttnc()
            ttne = min(ttnc, ttna)

            clock += ttne
            area_n += state * ttne

            # Advance all servers, collect indices of those that completed
            completed = [
                idx for idx, server in enumerate(self.servers)
                if server.update(ttne)
            ]

            # Route completed jobs
            for idx in completed:
                dest = self._route_job(idx)
                if dest >= len(self.servers):
                    num_completions += 1
                    state -= 1
                else:
                    self.servers[dest].arrival()

            # Handle arrival if it fires at or before the next completion
            if ttna <= ttnc:
                state += 1
                self.servers[0].arrival()
                ttna = self.genArrival()
            else:
                ttna -= ttne

        mean_n = area_n / clock
        mean_t = area_n / max(1, num_completions)
        self.T = mean_t
        return (mean_n, mean_t)


__all__ = ['QueueSystem']
