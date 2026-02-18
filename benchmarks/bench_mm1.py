"""Benchmark: Python vs C++ backend for M/M/1 FCFS queue.

Usage:
    python benchmarks/bench_mm1.py
"""

import time


def bench_python(num_events: int, seed: int = 42) -> tuple[float, float, float]:
    """Run Python backend, return (E[N], E[T], elapsed_seconds)."""
    from queue_sim import FCFS, QueueSystem, genExp

    system = QueueSystem([FCFS(sizefn=genExp(2.0))], arrivalfn=genExp(1.0))
    t0 = time.perf_counter()
    N, T = system.sim(num_events=num_events, seed=seed)
    elapsed = time.perf_counter() - t0
    return N, T, elapsed


def bench_cpp(num_events: int, seed: int = 42) -> tuple[float, float, float]:
    """Run C++ backend, return (E[N], E[T], elapsed_seconds)."""
    import _queue_sim_cpp as cpp

    server = cpp.FCFS(cpp.ExponentialDist(2.0))
    system = cpp.QueueSystem([server], cpp.ExponentialDist(1.0))
    t0 = time.perf_counter()
    N, T = system.sim(num_events=num_events, seed=seed)
    elapsed = time.perf_counter() - t0
    return N, T, elapsed


def main() -> None:
    sizes = [100_000, 500_000, 1_000_000, 5_000_000]

    print(f"{'events':>12s}  {'Python (s)':>12s}  {'C++ (s)':>12s}  {'speedup':>8s}")
    print("-" * 52)

    for n in sizes:
        _, _, t_py = bench_python(n)
        _, _, t_cpp = bench_cpp(n)
        speedup = t_py / t_cpp if t_cpp > 0 else float("inf")
        print(f"{n:>12,d}  {t_py:>12.3f}  {t_cpp:>12.3f}  {speedup:>7.1f}x")


if __name__ == "__main__":
    main()
