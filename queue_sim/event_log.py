"""Parallel-vector event log for simulation trajectory reconstruction."""

from __future__ import annotations

import numpy as np


class EventLog:
    """Record of simulation events with parallel-vector storage."""

    # -- Event kind constants --
    ARRIVAL: str = "arrival"
    DEPARTURE: str = "departure"
    ROUTE: str = "route"
    REJECTION: str = "rejection"

    # -- Special server indices --
    EXTERNAL: int = -1
    SYSTEM_EXIT: int = -1

    __slots__ = ("times", "kinds", "from_servers", "to_servers", "states")

    def __init__(self) -> None:
        self.times: list[float] = []
        self.kinds: list[str] = []
        self.from_servers: list[int] = []
        self.to_servers: list[int] = []
        self.states: list[int] = []

    def _append(
        self, time: float, kind: str, from_server: int, to_server: int, state: int
    ) -> None:
        self.times.append(time)
        self.kinds.append(kind)
        self.from_servers.append(from_server)
        self.to_servers.append(to_server)
        self.states.append(state)

    def __len__(self) -> int:
        return len(self.times)


def per_server_states(
    log,
    n_servers: int | None = None,
) -> dict[str, list]:
    """Reconstruct per-server occupancy from an event log.

    Works with both Python and C++ EventLog objects via duck typing.

    Args:
        log: An EventLog (Python or C++) with times, kinds, from_servers,
             to_servers attributes.
        n_servers: Number of servers. If None, inferred from the log.

    Returns:
        ``{"times": list[float], "server_states": list[list[int]]}`` where
        ``server_states[s][i]`` is the occupancy of server *s* after event *i*.

    Raises:
        ValueError: If the log is empty.
    """
    if len(log) == 0:
        raise ValueError("Event log is empty")

    kinds = log.kinds
    from_servers = log.from_servers
    to_servers = log.to_servers
    times = log.times

    if n_servers is None:
        max_idx = -1
        for v in from_servers:
            if v >= 0 and v > max_idx:
                max_idx = v
        for v in to_servers:
            if v >= 0 and v > max_idx:
                max_idx = v
        n_servers = max_idx + 1

    pops = [0] * n_servers
    result_times: list[float] = []
    server_states: list[list[int]] = [[] for _ in range(n_servers)]

    for i in range(len(log)):
        kind = kinds[i]
        fr = from_servers[i]
        to = to_servers[i]

        if kind == "arrival":
            # External arrival: from == -1, to == server index
            pops[to] += 1
        elif kind == "departure":
            # Departure from system: from == server index, to == -1
            pops[fr] -= 1
        elif kind == "route":
            # Routed: from server -> to server
            pops[fr] -= 1
            pops[to] += 1
        elif kind == "rejection":
            # Rejection: if from >= 0 (routed rejection), decrement source
            # If from == -1 (external rejection), no change
            if fr >= 0:
                pops[fr] -= 1

        result_times.append(times[i])
        for s in range(n_servers):
            server_states[s].append(pops[s])

    return {"times": result_times, "server_states": server_states}


def _bin_step_function(
    times: list[float] | np.ndarray,
    values: list | np.ndarray,
    bin_edges: np.ndarray,
) -> np.ndarray:
    """Compute time-weighted average of a step function over bins.

    Given a step function defined by *times* and *values* (value changes at
    each time), compute the time-weighted average in each bin defined by
    *bin_edges*.

    Args:
        times: Sorted event times (length *n*).
        values: Value of the step function after each event (length *n*).
        bin_edges: Sorted bin boundaries (length *n_bins + 1*).

    Returns:
        Array of shape ``(n_bins,)`` with the time-weighted average per bin.
    """
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    n_bins = len(bin_edges) - 1

    result = np.zeros(n_bins)
    if len(times) == 0:
        return result

    # Build the full step-function breakpoints: prepend time=0 with value=0,
    # then each event time with the corresponding value.
    bp_times = np.empty(len(times) + 1)
    bp_values = np.empty(len(times) + 1)
    bp_times[0] = 0.0
    bp_values[0] = 0.0
    bp_times[1:] = times
    bp_values[1:] = values

    for b in range(n_bins):
        lo = bin_edges[b]
        hi = bin_edges[b + 1]
        if hi <= lo:
            continue

        # Find breakpoints that fall within [lo, hi)
        i_start = np.searchsorted(bp_times, lo, side="right") - 1
        i_end = np.searchsorted(bp_times, hi, side="right") - 1

        area = 0.0
        cursor = lo
        for j in range(max(i_start, 0), min(i_end + 1, len(bp_times))):
            seg_end = bp_times[j + 1] if j + 1 < len(bp_times) else hi
            seg_end = min(seg_end, hi)
            if seg_end > cursor:
                area += bp_values[j] * (seg_end - cursor)
                cursor = seg_end

        # Handle remaining time in the bin after the last breakpoint
        if cursor < hi:
            area += bp_values[min(i_end, len(bp_values) - 1)] * (hi - cursor)

        result[b] = area / (hi - lo)

    return result
