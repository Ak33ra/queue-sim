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

from .results import ReplicationResult, _build_replication_result, _derive_seed
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
        *,
        _warmup: int = 0,
        track_response_times: bool = False,
        track_events: bool = False,
    ) -> tuple[float, float]:
        """Run the simulation.

        Args:
            num_events: Number of job completions (departures from the system)
                        before stopping.
            seed:       Optional RNG seed for reproducibility.
            _warmup:    Number of departures to discard before measurement.
            track_response_times: If True, record every job's response time
                        in ``self.response_times`` (list of floats).
            track_events: If True, record every event during the measurement
                        phase in ``self.event_log`` (:class:`EventLog`).

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
        state = 0                      # total jobs in the network

        # -- warmup phase (no accumulation) -----------------------------------
        if _warmup > 0:
            warmup_done = 0
            while warmup_done < _warmup:
                ttnc = self._min_ttnc()
                ttne = min(ttnc, ttna)
                completed = [
                    idx for idx, server in enumerate(self.servers)
                    if server.update(ttne)
                ]
                for idx in completed:
                    dest = self._route_job(idx)
                    if dest >= len(self.servers):
                        warmup_done += 1
                        state -= 1
                    else:
                        self.servers[dest].num_arrivals += 1
                        if self.servers[dest].is_full():
                            self.servers[dest].num_rejected += 1
                            warmup_done += 1
                            state -= 1
                        else:
                            self.servers[dest].arrival()
                if ttna <= ttnc:
                    self.servers[0].num_arrivals += 1
                    if self.servers[0].is_full():
                        self.servers[0].num_rejected += 1
                    else:
                        state += 1
                        self.servers[0].arrival()
                    ttna = self.genArrival()
                else:
                    ttna -= ttne

        # Clear per-server rejection counters so measurement reflects
        # only the measurement phase.
        for server in self.servers:
            server.num_rejected = 0
            server.num_arrivals = 0

        # -- measurement phase ------------------------------------------------
        if track_response_times:
            self.response_times: list[float] = []

        if track_events:
            from .event_log import EventLog

            self.event_log = EventLog()
            log = self.event_log

        area_n: float = 0.0
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
                    if track_response_times:
                        self.response_times.append(
                            self.servers[idx]._last_response_time
                        )
                    if track_events:
                        log._append(clock, EventLog.DEPARTURE, idx, EventLog.SYSTEM_EXIT, state)
                else:
                    self.servers[dest].num_arrivals += 1
                    if self.servers[dest].is_full():
                        self.servers[dest].num_rejected += 1
                        num_completions += 1
                        state -= 1
                        if track_events:
                            log._append(clock, EventLog.REJECTION, idx, dest, state)
                    else:
                        self.servers[dest].arrival()
                        if track_events:
                            log._append(clock, EventLog.ROUTE, idx, dest, state)

            # Handle arrival if it fires at or before the next completion
            if ttna <= ttnc:
                self.servers[0].num_arrivals += 1
                if self.servers[0].is_full():
                    self.servers[0].num_rejected += 1
                    if track_events:
                        log._append(clock, EventLog.REJECTION, EventLog.EXTERNAL, 0, state)
                else:
                    state += 1
                    self.servers[0].arrival()
                    if track_events:
                        log._append(clock, EventLog.ARRIVAL, EventLog.EXTERNAL, 0, state)
                ttna = self.genArrival()
            else:
                ttna -= ttne

        mean_n = area_n / clock
        mean_t = area_n / max(1, num_completions)
        self.T = mean_t
        return (mean_n, mean_t)


    # -- replications ---------------------------------------------------------

    def replicate(
        self,
        n_replications: int = 30,
        num_events: int = 10**6,
        seed: int | None = None,
        confidence: float = 0.95,
        warmup: int = 0,
    ) -> ReplicationResult:
        """Run multiple independent replications and return a CI.

        Args:
            n_replications: Number of independent runs (>= 2).
            num_events:     Departures per replication.
            seed:           Base seed (deterministic seed derivation per rep).
            confidence:     Confidence level in (0, 1).
            warmup:         Warmup departures discarded per replication.

        Returns:
            :class:`ReplicationResult` with grand means and CIs.
        """
        if n_replications < 2:
            raise ValueError("n_replications must be >= 2")
        if not (0 < confidence < 1):
            raise ValueError("confidence must be in (0, 1)")

        base_seed = seed if seed is not None else random.randrange(2**63)

        raw_N: list[float] = []
        raw_T: list[float] = []
        for i in range(n_replications):
            rep_seed = _derive_seed(base_seed, i)
            n, t = self.sim(num_events=num_events, seed=rep_seed, _warmup=warmup)
            raw_N.append(n)
            raw_T.append(t)

        return _build_replication_result(
            tuple(raw_N), tuple(raw_T), confidence,
        )


__all__ = ['QueueSystem']
