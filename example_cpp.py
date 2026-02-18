import _queue_sim_cpp as cpp

# --- Scheduling policies ---

# M/M/1-FCFS
system = cpp.QueueSystem([cpp.FCFS(cpp.ExponentialDist(2.0))], cpp.ExponentialDist(1.0))
N, T = system.sim(num_events=10**6, seed=42)

# M/G/1-PS with uniform service
system = cpp.QueueSystem([cpp.PS(cpp.UniformDist(0.3, 0.7))], cpp.ExponentialDist(1.0))
N, T = system.sim(num_events=10**6, seed=42)

# M/M/1-FB
system = cpp.QueueSystem([cpp.FB(cpp.ExponentialDist(2.0))], cpp.ExponentialDist(1.0))
N, T = system.sim(num_events=10**6, seed=42)

# --- Multi-server (G/G/k) ---

# M/M/2-FCFS
system = cpp.QueueSystem(
    [cpp.FCFS(cpp.ExponentialDist(1.0), num_servers=2)], cpp.ExponentialDist(1.0)
)
N, T = system.sim(num_events=10**6, seed=42)

# M/M/2-PS
system = cpp.QueueSystem(
    [cpp.PS(cpp.ExponentialDist(1.0), num_servers=2)], cpp.ExponentialDist(1.0)
)
N, T = system.sim(num_events=10**6, seed=42)

# --- Finite buffers (loss queues) ---

# M/M/3/3 (Erlang-B)
server = cpp.FCFS(cpp.ExponentialDist(1.0), num_servers=3, buffer_capacity=3)
system = cpp.QueueSystem([server], cpp.ExponentialDist(2.0))
system.sim(num_events=10**6, seed=42)
print(f"P(loss) = {server.num_rejected / server.num_arrivals:.4f}")

# M/M/1/5
server = cpp.FCFS(cpp.ExponentialDist(2.0), buffer_capacity=5)
system = cpp.QueueSystem([server], cpp.ExponentialDist(1.0))
system.sim(num_events=10**6, seed=42)
print(f"P(loss) = {server.num_rejected / server.num_arrivals:.4f}")

# --- Networks ---

# Tandem: PS -> FCFS
system = cpp.QueueSystem(
    [cpp.PS(cpp.ExponentialDist(4.0)), cpp.FCFS(cpp.ExponentialDist(4.0))],
    cpp.ExponentialDist(1.0),
)
N, T = system.sim(num_events=10**6, seed=42)

# --- Parallel replications ---

system = cpp.QueueSystem([cpp.FCFS(cpp.ExponentialDist(2.0))], cpp.ExponentialDist(1.0))
raw = system.replicate(
    n_replications=30, num_events=10**6, seed=42, warmup=10_000, n_threads=4,
)
# raw.raw_T and raw.raw_N are lists of per-replication results

# Wrap with CI computation (no scipy needed)
from queue_sim.results import _build_replication_result
result = _build_replication_result(tuple(raw.raw_N), tuple(raw.raw_T), 0.95)
print(f"E[T] = {result.mean_T:.4f}  95% CI: {result.ci_T}")

# --- Response time distribution tracking ---

import numpy as np

system = cpp.QueueSystem([cpp.FCFS(cpp.ExponentialDist(2.0))], cpp.ExponentialDist(1.0))
system.sim(num_events=10**6, seed=42, track_response_times=True)
rt = np.array(system.response_times)
print(f"Median: {np.median(rt):.4f}, P99: {np.percentile(rt, 99):.4f}")